import argparse
import os
import sys
import time
import torch
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    # Ensure we load the local package instead of any installed version.
    sys.path.insert(0, str(REPO_ROOT))

from flash_attn.flash_attn_triton_amd.fwd_prefill import (
    ENV_USE_GLUON_PREFILL,
    attention_prefill_forward_triton_impl,
)
from flash_attn.flash_attn_triton_amd.utils import input_helper

PATH_COOLDOWN_S = 1.0
CASE_COOLDOWN_S = 2.0

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Gluon vs Triton prefill forward performance."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bf16"],
        help="Computation dtype.",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations per path.")
    parser.add_argument("--iters", type=int, default=50, help="Timed iterations per path.")
    parser.add_argument("--seed", type=int, default=123, help="Torch random seed.")
    return parser.parse_args()


def dtype_from_str(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def run_prefill_once(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    metadata,
) -> None:
    attention_prefill_forward_triton_impl(
        q,
        k,
        v,
        o,
        metadata.sm_scale,
        metadata.alibi_slopes,
        metadata.causal,
        metadata.bias,
        metadata.layout,
        metadata.cu_seqlens_q,
        metadata.cu_seqlens_k,
        metadata.max_seqlens_q,
        metadata.max_seqlens_k,
        metadata.cache_seqlens,
        metadata.cache_batch_idx,
        metadata.dropout_p,
        metadata.philox_seed,
        metadata.philox_offset,
        metadata.return_scores,
        metadata.use_exp2,
        None,
        None,
        None,
        None,
    )


def plot_results(results: List[Dict[str, float]], chunk_idx: int, total_chunks: int) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless-friendly
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot.")
        return

    labels = [r["label"] for r in results]
    gluon = [r["gluon_ms"] for r in results]
    triton = [r["triton_ms"] for r in results]
    ratios = [r["ratio"] for r in results]
    gains = [r["gain_pct"] for r in results]

    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width / 2 for i in x], gluon, width, label="Gluon")
    ax.bar([i + width / 2 for i in x], triton, width, label="Triton")
    for i, ratio in enumerate(ratios):
        label = f"ratio={ratio:.3f}\ngain={gains[i]:.2f}%"
        ax.text(
            i,
            max(gluon[i], triton[i]) * 1.02,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_ylabel("ms per iteration")
    if total_chunks > 1:
        ax.set_title(f"Prefill forward: Gluon vs Triton (part {chunk_idx + 1}/{total_chunks})")
    else:
        ax.set_title("Prefill forward: Gluon vs Triton")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15)
    ax.legend()
    fig.tight_layout()
    if total_chunks > 1:
        out_path = Path(__file__).with_name(
            f"{Path(__file__).stem}_part{chunk_idx + 1}.png"
        )
    else:
        out_path = Path(__file__).with_suffix(".png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    plt.close(fig)


def benchmark_path(
    use_gluon: bool,
    base_inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    metadata,
    warmup: int,
    iters: int,
) -> float:
    os.environ[ENV_USE_GLUON_PREFILL] = "1" if use_gluon else "0"

    # Clone once so both paths use identical inputs without timing the copy cost.
    q = base_inputs[0].clone()
    k = base_inputs[1].clone()
    v = base_inputs[2].clone()
    o = torch.empty_like(q)

    def _call():
        run_prefill_once(q, k, v, o, metadata)

    torch.cuda.synchronize()
    for _ in range(warmup):
        _call()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _call()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / float(iters)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required for this benchmark.")

    torch.manual_seed(args.seed)
    device = "cuda"
    dtype = dtype_from_str(args.dtype)

    # Predefined configs to keep the script simple and reproducible.
    input_configs = [
        # LLaMA 3 8B
        #(1, 32, 8, 8192, 8192, 128, True, 0.0, "thd", True),
        # LLaMA 3 70B
        #(1, 64, 8, 8192, 8192, 128, True, 0.0, "thd", True),
        # Wan2.2 T2V/I2V A14B 480p (832x480, 81 frames)
        (1, 40, 40, 32760, 32760, 128, True, 0.0, "thd", True),
        (1, 40, 40, 32760, 32760, 128, True, 0.0, "thd", False),
        (1, 40, 40, 32760, 32760, 128, False, 0.0, "thd", True),
        (1, 40, 40, 32760, 32760, 128, False, 0.0, "thd", False),
        # Wan2.2 T2V/I2V A14B 720p (1280x720, 81 frames)
        (1, 40, 40, 75600, 75600, 128, False, 0.0, "thd", True),
        (1, 40, 40, 75600, 75600, 128, False, 0.0, "thd", False),
        # Wan2.2 TI2V-5B 720p (1280x704, 121 frames)
        (1, 24, 24, 27280, 27280, 128, False, 0.0, "thd", True),
        (1, 24, 24, 27280, 27280, 128, False, 0.0, "thd", False),
        # Wan2.2 S2V-14B (1024x704, 81 frames)
        (1, 40, 40, 59136, 59136, 128, False, 0.0, "thd", True),
        (1, 40, 40, 59136, 59136, 128, False, 0.0, "thd", False),
        # Wan2.2 Animate-14B (1280x720, 77 frames)
        (1, 40, 40, 72000, 72000, 128, False, 0.0, "thd", True),
        (1, 40, 40, 72000, 72000, 128, False, 0.0, "thd", False),
    ]

    print("Prefill forward benchmark (Gluon vs Triton)")
    all_results: List[Dict[str, float]] = []
    for (
        batch,
        hq,
        hk,
        seqlen_q,
        seqlen_k,
        d_head,
        causal,
        dropout,
        layout,
        use_exp2,
    ) in input_configs:
        # Generate inputs; metadata carries the layout/config for the kernel.
        q, k, v, _, metadata = input_helper(
            batch,
            hq,
            hk,
            seqlen_q,
            seqlen_k,
            d_head,
            causal,
            dropout,
            dtype,
            layout=layout,
            device=device,
        )
        metadata.use_exp2 = use_exp2
        metadata.return_scores = False  # Benchmark the normal path without returning scores.

        base_inputs = (q.detach(), k.detach(), v.detach())

        with torch.inference_mode():
            results: Dict[str, float] = {}
            results["gluon_ms"] = benchmark_path(True, base_inputs, metadata, args.warmup, args.iters)
            time.sleep(PATH_COOLDOWN_S)
            results["triton_ms"] = benchmark_path(False, base_inputs, metadata, args.warmup, args.iters)

        ratio = results["gluon_ms"] / results["triton_ms"]  # Gluon relative to Triton baseline.
        gain_pct = (1.0 - ratio) * 100.0  # Percent improvement vs Triton; positive means Gluon is faster.

        print(
            f"\nConfig: batch={batch}, hq={hq}, hk={hk}, seqlen_q={seqlen_q}, seqlen_k={seqlen_k}, "
            f"d_head={d_head}, causal={causal}, dropout={dropout}, layout={layout}, dtype={args.dtype}, "
            f"use_exp2={use_exp2}"
        )
        print(f"warmup={args.warmup}, iters={args.iters}")
        print(f"Gluon : {results['gluon_ms']:.3f} ms")
        print(f"Triton: {results['triton_ms']:.3f} ms")
        print(f"Ratio : {ratio:.3f}  (Gluon / Triton; <1 means Gluon is faster)")
        print(f"Gain  : {gain_pct:.2f}% (relative improvement vs Triton baseline)")

        all_results.append(
            {
                "label": f"B{batch}-HQ{hq}-HK{hk}-SQ{seqlen_q}-SK{seqlen_k}-D{d_head}",
                "gluon_ms": results["gluon_ms"],
                "triton_ms": results["triton_ms"],
                "ratio": ratio,
                "gain_pct": gain_pct,
            }
        )
        time.sleep(CASE_COOLDOWN_S)

    max_cases_per_plot = 8
    total_chunks = (len(all_results) + max_cases_per_plot - 1) // max_cases_per_plot
    for chunk_idx in range(total_chunks):
        start = chunk_idx * max_cases_per_plot
        end = start + max_cases_per_plot
        plot_results(all_results[start:end], chunk_idx, total_chunks)


if __name__ == "__main__":
    main()
