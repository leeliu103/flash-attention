import pytest
import torch

from .fwd_prefill import attention_prefill_forward_triton_impl
from .utils import input_helper

# Match tolerances used elsewhere in the suite.
ATOL, RTOL = 1e-2, 1e-2


@pytest.mark.parametrize(
    "BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD",
    [
        (1, 12, 12, 32760, 32760, 128),
    ],
)
@pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("dropout_p", [0.0])
@pytest.mark.parametrize("alibi_slopes", [None])
@pytest.mark.parametrize("layout", ["thd"])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("use_exp2", [True, False])
def test_op_prefill_triton_vs_gluon(
    BATCH,
    HQ,
    HK,
    N_CTX_Q,
    N_CTX_K,
    D_HEAD,
    causal,
    dropout_p,
    alibi_slopes,
    layout,
    dtype,
    use_exp2,
    monkeypatch,
):
    """Compare Triton and Gluon prefill forward paths for correctness."""

    torch.manual_seed(123)
    device = "cuda"

    q, k, v, _, metadata = input_helper(
        BATCH,
        HQ,
        HK,
        N_CTX_Q,
        N_CTX_K,
        D_HEAD,
        causal,
        dropout_p,
        dtype,
        layout=layout,
        device=device,
    )
    metadata.use_exp2 = use_exp2
    if alibi_slopes is not None:
        metadata.need_alibi(alibi_slopes, BATCH, HQ)

    def run_path(use_gluon: bool):
        flag = "1" if use_gluon else "0"
        monkeypatch.setenv("FLASH_ATTENTION_TRITON_AMD_USE_GLUON_PREFILL", flag)
        q_run, k_run, v_run = q.clone(), k.clone(), v.clone()
        o_run = torch.empty_like(q_run)

        softmax_lse_run, _ = attention_prefill_forward_triton_impl(
            q_run,
            k_run,
            v_run,
            o_run,
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
        return o_run, softmax_lse_run

    o_gluon, lse_gluon = run_path(True)
    o_triton, lse_triton = run_path(False)

    torch.testing.assert_close(o_triton, o_gluon, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(lse_triton, lse_gluon, atol=ATOL, rtol=RTOL)
