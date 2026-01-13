from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl


@gluon.jit
def compute_alibi_block(
    alibi_slope,
    seqlen_q,
    seqlen_k,
    offs_m,
    offs_n,
    transpose: ttgl.constexpr = False,
):
    relative_pos = offs_m[:, None] + seqlen_k - seqlen_q - offs_n[None, :]
    alibi_block = -1 * alibi_slope * ttgl.abs(relative_pos)
    return alibi_block.permute((1, 0)) if transpose else alibi_block


@gluon.jit
def compute_fp8_scaling_factors(x, fp8_max: ttgl.constexpr):
    x_abs = ttgl.abs(x)
    x_amax = ttgl.max(x_abs)
    x_amax = ttgl.where(x_amax <= 1e-9, 1e-9, x_amax)
    scale_x = fp8_max / x_amax
    descale_x = x_amax / fp8_max
    return scale_x, descale_x


@gluon.jit
def load_fn(ptr, offsets, mask=None):
    offsets = ttgl.cast(offsets, ttgl.int32)
    return ttgl.amd.rdna4.buffer_load(ptr, offsets, mask=mask, other=0.0)


@gluon.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q_frag,
    q_rows,
    k_ptr,
    v_ptr,
    base_k,
    base_v,
    offs_m_wmma,
    offs_n_wmma,
    offs_n_k,
    offs_n_v,
    offs_d_k,
    offs_d_v,
    stride_kn,
    stride_vk,
    stride_kk,
    stride_vn,
    seqlen_q,
    seqlen_k,
    block_min,
    block_max,
    causal_shift,
    APPLY_MASKS: ttgl.constexpr,
    APPLY_CAUSAL: ttgl.constexpr,
    SM_SCALE: ttgl.constexpr,
    BLOCK_M: ttgl.constexpr,
    BLOCK_N: ttgl.constexpr,
    BLOCK_DMODEL: ttgl.constexpr,
    USE_EXP2: ttgl.constexpr,
):
    wmma_layout: ttgl.constexpr = ttgl.amd.AMDWMMALayout(
        version=2, transposed=True, warp_bases=[[1, 0], [2, 0]]
    )
    blocked_k: ttgl.constexpr = ttgl.BlockedLayout([8, 1], [16, 2], [1, 4], [0, 1])
    blocked_v: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [2, 16], [4, 1], [1, 0])
    shared_k: ttgl.constexpr = ttgl.SwizzledSharedLayout(8, 1, 16, order=[0, 1])
    shared_v: ttgl.constexpr = ttgl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    dot_k: ttgl.constexpr = ttgl.DotOperandLayout(1, wmma_layout, 8)
    dot_v: ttgl.constexpr = ttgl.DotOperandLayout(1, wmma_layout, 8)
    dot_p: ttgl.constexpr = ttgl.DotOperandLayout(0, wmma_layout, 8)
    RCP_LN2: ttgl.constexpr = 1.4426950408889634

    zero_qk = ttgl.zeros([BLOCK_M, BLOCK_N], ttgl.float32, layout=wmma_layout)
    neg_inf = ttgl.full(
        [BLOCK_M, BLOCK_N], -float("inf"), ttgl.float32, layout=wmma_layout
    )

    acc_local = acc
    l_local = l_i
    m_local = m_i

    k_smem = ttgl.allocate_shared_memory(ttgl.float16, [BLOCK_DMODEL, BLOCK_N], shared_k)
    v_smem = ttgl.allocate_shared_memory(ttgl.float16, [BLOCK_N, BLOCK_DMODEL], shared_v)

    for start_n in range(block_min, block_max, BLOCK_N):
        k_rows = start_n + offs_n_k
        v_rows = start_n + offs_n_v

        k_offsets = (
            base_k
            + ttgl.expand_dims(offs_d_k, 1) * stride_kk
            + ttgl.expand_dims(k_rows, 0) * stride_kn
        )
        v_offsets = (
            base_v
            + ttgl.expand_dims(v_rows, 1) * stride_vk
            + ttgl.expand_dims(offs_d_v, 0) * stride_vn
        )

        if APPLY_MASKS:
            mask_k = ttgl.expand_dims(ttgl.cast(k_rows, ttgl.int64), 0) < seqlen_k
            mask_v = ttgl.expand_dims(ttgl.cast(v_rows, ttgl.int64), 1) < seqlen_k
            k_tile = load_fn(k_ptr, k_offsets, mask=mask_k)
            v_tile = load_fn(v_ptr, v_offsets, mask=mask_v)
        else:
            k_tile = load_fn(k_ptr, k_offsets)
            v_tile = load_fn(v_ptr, v_offsets)

        k_smem.store(k_tile)
        v_smem.store(v_tile)

        k_frag = k_smem.load(dot_k)
        v_frag = v_smem.load(dot_v)

        if APPLY_MASKS:
            k_rows_i64 = ttgl.cast(start_n + offs_n_wmma, ttgl.int64)
            valid_n = ttgl.expand_dims(k_rows_i64, 0) < seqlen_k
            qk_init = ttgl.where(valid_n, zero_qk, neg_inf)
        else:
            qk_init = zero_qk

        if APPLY_CAUSAL:
            causal_cols = ttgl.cast(start_n + offs_n_wmma + causal_shift, ttgl.int64)
            causal_mask = (
                ttgl.expand_dims(q_rows, 1) >= ttgl.expand_dims(causal_cols, 0)
            )
            qk_init = ttgl.where(causal_mask, qk_init, neg_inf)

        qk = ttgl.amd.rdna4.wmma(q_frag, k_frag, qk_init)
        qk_scaled = qk * SM_SCALE

        m_ij = ttgl.max(qk_scaled, 1)
        m_new = ttgl.maximum(m_local, m_ij)
        q_shifted = qk_scaled - ttgl.expand_dims(m_new, 1)
        if USE_EXP2:
            p = ttgl.exp2(q_shifted * RCP_LN2)
        else:
            p = ttgl.exp(q_shifted)
        l_ij = ttgl.sum(p, 1)

        if USE_EXP2:
            alpha = ttgl.exp2((m_local - m_new) * RCP_LN2)
        else:
            alpha = ttgl.exp(m_local - m_new)
        acc_scaled = acc_local * ttgl.expand_dims(alpha, 1)

        p_frag = ttgl.convert_layout(p.to(ttgl.float16), layout=dot_p)
        acc_next = ttgl.amd.rdna4.wmma(p_frag, v_frag, acc_scaled)
        l_next = l_local * alpha + l_ij

        acc_local = acc_next
        l_local = l_next
        m_local = m_new

    return acc_local, l_local, m_local


@gluon.jit
def attn_fwd(
    Q,
    K,
    V,
    bias,
    Cache_seqlens,
    Cache_batch_idx,
    Descale_Q,
    Descale_K,
    Descale_V,
    Descale_O,
    stride_descale_q_z,
    stride_descale_k_z,
    stride_descale_v_z,
    stride_descale_o_z,
    SM_SCALE: ttgl.constexpr,
    LSE,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    stride_bz,
    stride_bh,
    stride_bm,
    stride_bn,
    stride_az,
    stride_ah,
    stride_sz,
    stride_sh,
    stride_sm,
    stride_sn,
    stride_lse_z,
    stride_lse_h,
    stride_lse_m,
    cu_seqlens_q,
    cu_seqlens_k,
    dropout_p,
    philox_seed,
    philox_offset_base,
    sd_mask,
    dropout_mask,
    alibi_slopes,
    HQ: ttgl.constexpr,
    HK: ttgl.constexpr,
    ACTUAL_BLOCK_DMODEL: ttgl.constexpr,
    MAX_SEQLENS_Q: ttgl.constexpr,
    MAX_SEQLENS_K: ttgl.constexpr,
    IS_VARLEN: ttgl.constexpr,
    IS_INFERENCE: ttgl.constexpr,
    IS_CAUSAL: ttgl.constexpr,
    BLOCK_M: ttgl.constexpr,
    BLOCK_DMODEL: ttgl.constexpr,
    BLOCK_N: ttgl.constexpr,
    PRE_LOAD_V: ttgl.constexpr,
    USE_BIAS: ttgl.constexpr,
    ENABLE_DROPOUT: ttgl.constexpr,
    RETURN_SCORES: ttgl.constexpr,
    USE_ALIBI: ttgl.constexpr,
    USE_EXP2: ttgl.constexpr,
    IS_FP8: ttgl.constexpr,
    FP8_MAX: ttgl.constexpr,
    FP8_OUTPUT: ttgl.constexpr,
    num_warps: ttgl.constexpr,
    num_stages: ttgl.constexpr,
):
    wmma_layout: ttgl.constexpr = ttgl.amd.AMDWMMALayout(
        version=2, transposed=True, warp_bases=[[1, 0], [2, 0]]
    )
    blocked_q: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [2, 16], [4, 1], [1, 0])
    blocked_k: ttgl.constexpr = ttgl.BlockedLayout([8, 1], [16, 2], [1, 4], [0, 1])
    blocked_v: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [2, 16], [4, 1], [1, 0])
    blocked_m: ttgl.constexpr = ttgl.BlockedLayout([1], [32], [4], [0])
    shared_q: ttgl.constexpr = ttgl.SwizzledSharedLayout(8, 1, 16, order=[1, 0])
    RCP_LN2: ttgl.constexpr = 1.4426950408889634
    LN2: ttgl.constexpr = 0.6931471824645996

    pid_m = ttgl.program_id(0)
    off_h_q = ttgl.program_id(1)
    off_z = ttgl.program_id(2)

    cu_seqlens_q_start = ttgl.load(cu_seqlens_q + off_z)
    cu_seqlens_q_end = ttgl.load(cu_seqlens_q + off_z + 1)
    seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start

    if pid_m * BLOCK_M > seqlen_q:
        return

    cu_seqlens_k_start = ttgl.load(cu_seqlens_k + off_z)
    cu_seqlens_k_end = ttgl.load(cu_seqlens_k + off_z + 1)
    seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start

    group_size: ttgl.constexpr = HQ // HK
    off_h_k = off_h_q // group_size

    offs_m_blocked = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, blocked_q))
    offs_m_wmma = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, wmma_layout))
    offs_m_store = ttgl.arange(0, BLOCK_M, layout=blocked_m)
    offs_n_k = ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, blocked_k))
    offs_n_v = ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(1, blocked_v))
    offs_n_wmma = ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, wmma_layout))
    offs_d_q = ttgl.arange(0, BLOCK_DMODEL, layout=ttgl.SliceLayout(0, blocked_q))
    offs_d_k = ttgl.arange(0, BLOCK_DMODEL, layout=ttgl.SliceLayout(1, blocked_k))
    offs_d_v = ttgl.arange(0, BLOCK_DMODEL, layout=ttgl.SliceLayout(0, blocked_q))
    offs_d_out = ttgl.arange(0, BLOCK_DMODEL, layout=ttgl.SliceLayout(0, wmma_layout))

    q_rows = pid_m * BLOCK_M + offs_m_blocked
    q_rows_wmma = ttgl.cast(pid_m * BLOCK_M + offs_m_wmma, ttgl.int64)

    q_base = (
        off_z * stride_qz
        + off_h_q * stride_qh
        + cu_seqlens_q_start * stride_qm
    )
    q_offsets = q_base + q_rows[:, None] * stride_qm + offs_d_q[None, :] * stride_qk
    q_offsets = ttgl.cast(q_offsets, ttgl.int32)

    q_mask = (ttgl.expand_dims(q_rows, 1) < seqlen_q) & (
        ttgl.expand_dims(offs_d_q, 0) < ACTUAL_BLOCK_DMODEL
    )
    q_tile = ttgl.amd.rdna4.buffer_load(Q, q_offsets, mask=q_mask, other=0.0)
    q_smem = ttgl.allocate_shared_memory(ttgl.float16, [BLOCK_M, BLOCK_DMODEL], shared_q)
    q_smem.store(q_tile)
    q_frag = q_smem.load(ttgl.DotOperandLayout(0, wmma_layout, 8))

    acc = ttgl.zeros([BLOCK_M, BLOCK_DMODEL], ttgl.float32, layout=wmma_layout)
    l_i = ttgl.full(
        [BLOCK_M], 1.0, ttgl.float32, layout=ttgl.SliceLayout(1, wmma_layout)
    )
    m_i = ttgl.full(
        [BLOCK_M], -float("inf"), ttgl.float32, layout=ttgl.SliceLayout(1, wmma_layout)
    )

    k_base = (
        off_z * stride_kz
        + off_h_k * stride_kh
        + cu_seqlens_k_start * stride_kn
    )
    v_base = (
        off_z * stride_vz
        + off_h_k * stride_vh
        + cu_seqlens_k_start * stride_vk
    )

    n_blocks = ttgl.cdiv(seqlen_k, BLOCK_N)
    if IS_CAUSAL:
        n_blocks_seqlen = ttgl.cdiv(
            (pid_m + 1) * BLOCK_M + seqlen_k - seqlen_q, BLOCK_N
        )
        n_blocks = ttgl.minimum(n_blocks, n_blocks_seqlen)
        if n_blocks <= 0:
            o_base = (
                off_z * stride_oz
                + off_h_q * stride_oh
                + cu_seqlens_q_start * stride_om
            )
            o_offsets = (
                o_base
                + ttgl.expand_dims(pid_m * BLOCK_M + offs_m_wmma, 1) * stride_om
                + ttgl.expand_dims(offs_d_out, 0) * stride_on
            )
            o_offsets = ttgl.cast(o_offsets, ttgl.int32)
            o_mask = ttgl.expand_dims(pid_m * BLOCK_M + offs_m_wmma, 1) < seqlen_q
            zero_out = ttgl.zeros(
                [BLOCK_M, BLOCK_DMODEL], Out.type.element_ty, layout=wmma_layout
            )
            ttgl.amd.rdna4.buffer_store(zero_out, Out, o_offsets, mask=o_mask)

            l_offset = (
                off_z * stride_lse_z
                + off_h_q * stride_lse_h
                + cu_seqlens_q_start * stride_lse_m
            )
            l_ptrs = l_offset + (pid_m * BLOCK_M + offs_m_store) * stride_lse_m
            l_ptrs = ttgl.cast(l_ptrs, ttgl.int32)
            l_ptrs_mask = offs_m_store < MAX_SEQLENS_Q
            l_zero = ttgl.zeros([BLOCK_M], ttgl.float32, layout=blocked_m)
            ttgl.amd.rdna4.buffer_store(l_zero, LSE, l_ptrs, mask=l_ptrs_mask)
            return

    n_extra_tokens = ttgl.where(
        seqlen_k < BLOCK_N,
        BLOCK_N - seqlen_k,
        ttgl.where(seqlen_k % BLOCK_N != 0, seqlen_k % BLOCK_N, 0),
    )
    is_modulo_mn = (n_extra_tokens == 0) & ((seqlen_q % BLOCK_M) == 0)
    base_masked_blocks = ttgl.full((), BLOCK_M // BLOCK_N, ttgl.int64)
    masked_blocks = base_masked_blocks + ttgl.where(
        is_modulo_mn, ttgl.full((), 0, ttgl.int64), ttgl.full((), 1, ttgl.int64)
    )
    masked_blocks = ttgl.minimum(masked_blocks, n_blocks)
    n_full_blocks = n_blocks - masked_blocks
    block_min = n_full_blocks * BLOCK_N
    block_max = n_blocks * BLOCK_N
    causal_shift = seqlen_q - seqlen_k

    if n_full_blocks > 0:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q_frag,
            q_rows_wmma,
            K,
            V,
            k_base,
            v_base,
            offs_m_wmma,
            offs_n_wmma,
            offs_n_k,
            offs_n_v,
            offs_d_k,
            offs_d_v,
            stride_kn,
            stride_vk,
            stride_kk,
            stride_vn,
            seqlen_q,
            seqlen_k,
            0,
            block_min,
            causal_shift,
            False,
            False,
            SM_SCALE,
            BLOCK_M,
            BLOCK_N,
            BLOCK_DMODEL,
            USE_EXP2,
        )

    if masked_blocks > 0:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q_frag,
            q_rows_wmma,
            K,
            V,
            k_base,
            v_base,
            offs_m_wmma,
            offs_n_wmma,
            offs_n_k,
            offs_n_v,
            offs_d_k,
            offs_d_v,
            stride_kn,
            stride_vk,
            stride_kk,
            stride_vn,
            seqlen_q,
            seqlen_k,
            block_min,
            block_max,
            causal_shift,
            True,
            IS_CAUSAL,
            SM_SCALE,
            BLOCK_M,
            BLOCK_N,
            BLOCK_DMODEL,
            USE_EXP2,
        )

    acc = acc / ttgl.expand_dims(l_i, 1)

    l_offset = (
        off_z * stride_lse_z
        + off_h_q * stride_lse_h
        + cu_seqlens_q_start * stride_lse_m
    )
    l_ptrs = l_offset + (pid_m * BLOCK_M + offs_m_store) * stride_lse_m
    l_ptrs = ttgl.cast(l_ptrs, ttgl.int32)
    if USE_EXP2:
        mi_base2 = m_i * RCP_LN2
        softmax_lse = (mi_base2 + ttgl.log2(l_i)) * LN2
    else:
        softmax_lse = m_i + ttgl.log(l_i)

    start_m_idx = pid_m * BLOCK_M
    end_m_idx = (pid_m + 1) * BLOCK_M
    overflow_size = end_m_idx - seqlen_q
    causal_start_idx = seqlen_q - seqlen_k

    if IS_CAUSAL:
        lse_mask = (start_m_idx + offs_m_wmma) < causal_start_idx
        softmax_lse = ttgl.where(lse_mask, 0.0, softmax_lse)

    l_store = ttgl.convert_layout(softmax_lse, layout=blocked_m)

    if overflow_size > 0:
        boundary = BLOCK_M - overflow_size
        l_mask = offs_m_store < boundary
        ttgl.amd.rdna4.buffer_store(l_store, LSE, l_ptrs, mask=l_mask)
    else:
        ttgl.amd.rdna4.buffer_store(l_store, LSE, l_ptrs)

    o_base = (
        off_z * stride_oz
        + off_h_q * stride_oh
        + cu_seqlens_q_start * stride_om
    )
    o_offsets = (
        o_base
        + ttgl.expand_dims(pid_m * BLOCK_M + offs_m_wmma, 1) * stride_om
        + ttgl.expand_dims(offs_d_out, 0) * stride_on
    )
    o_offsets = ttgl.cast(o_offsets, ttgl.int32)

    if IS_CAUSAL:
        if causal_start_idx > start_m_idx and causal_start_idx < end_m_idx:
            out_mask_boundary = ttgl.full(
                [BLOCK_M],
                causal_start_idx,
                ttgl.int32,
                layout=ttgl.SliceLayout(1, wmma_layout),
            )
            out_ptrs_mask = (
                ttgl.expand_dims(pid_m * BLOCK_M + offs_m_wmma, 1)
                >= ttgl.expand_dims(out_mask_boundary, 1)
            )
            acc = ttgl.where(out_ptrs_mask, acc, ttgl.zeros_like(acc))

    o_mask = (ttgl.expand_dims(pid_m * BLOCK_M + offs_m_wmma, 1) < seqlen_q) & (
        ttgl.expand_dims(offs_d_out, 0) < ACTUAL_BLOCK_DMODEL
    )

    acc_out = acc.to(Out.type.element_ty)
    ttgl.amd.rdna4.buffer_store(acc_out, Out, o_offsets, mask=o_mask)
