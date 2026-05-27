import math

import torch
import triton
import triton.language as tl

__all__ = ["fused_attention", "flash_attention_fwd"]


@triton.jit
def _fused_attention_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    seq_len, head_dim, scale,
    stride_qb, stride_qs, stride_qd,
    stride_kb, stride_ks, stride_kd,
    stride_vb, stride_vs, stride_vd,
    stride_ob, stride_os, stride_od,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_row = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim

    # Load one query row
    q_ptrs = q_ptr + pid_b * stride_qb + pid_row * stride_qs + offs_d * stride_qd
    q = tl.load(q_ptrs, mask=d_mask, other=0.0)

    # Compute attention scores for this row against all key rows
    m_max = float("-inf")
    sum_exp = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for s_start in range(0, seq_len, BLOCK_S):
        offs_s = s_start + tl.arange(0, BLOCK_S)
        s_mask = offs_s < seq_len

        # Load K block: [BLOCK_S, BLOCK_D]
        k_ptrs = k_ptr + pid_b * stride_kb + offs_s[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k_mask = s_mask[:, None] & d_mask[None, :]
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # Scores: dot(q, k_j) for each j in block
        scores = tl.sum(q[None, :] * k, axis=1) * scale

        # Online softmax: update running max and sum
        block_max = tl.max(tl.where(s_mask, scores, float("-inf")))
        new_max = tl.maximum(m_max, block_max)
        correction = tl.exp(m_max - new_max)
        exp_scores = tl.exp(scores - new_max)
        exp_scores = tl.where(s_mask, exp_scores, 0.0)

        # Load V block: [BLOCK_S, BLOCK_D]
        v_ptrs = v_ptr + pid_b * stride_vb + offs_s[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=k_mask, other=0.0)

        # Update accumulator
        acc = acc * correction + tl.sum(exp_scores[:, None] * v, axis=0)
        sum_exp = sum_exp * correction + tl.sum(exp_scores)
        m_max = new_max

    acc = acc / sum_exp

    o_ptrs = o_ptr + pid_b * stride_ob + pid_row * stride_os + offs_d * stride_od
    tl.store(o_ptrs, acc, mask=d_mask)


def fused_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float | None = None
) -> torch.Tensor:
    """Fused multi-head attention: softmax(Q @ K^T * scale) @ V per batch element."""
    assert q.ndim == 3, "Inputs must be [batch, seq_len, head_dim]"
    batch, seq_len, head_dim = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    o = torch.empty_like(q)
    BLOCK_D = triton.next_power_of_2(head_dim)
    BLOCK_S = 64
    grid = (batch, seq_len)
    _fused_attention_kernel[grid](
        q, k, v, o,
        seq_len, head_dim, scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        BLOCK_D=BLOCK_D, BLOCK_S=BLOCK_S,
    )
    return o


@triton.jit
def _flash_attention_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    seq_len, head_dim, scale,
    stride_qb, stride_qs, stride_qd,
    stride_kb, stride_ks, stride_kd,
    stride_vb, stride_vs, stride_vd,
    stride_ob, stride_os, stride_od,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_q = tl.program_id(1)

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, BLOCK_D)
    q_mask = (offs_q[:, None] < seq_len) & (offs_d[None, :] < head_dim)

    # Load Q block: [BLOCK_Q, BLOCK_D]
    q_ptrs = q_ptr + pid_b * stride_qb + offs_q[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Running softmax state per query row
    m_i = tl.full((BLOCK_Q,), value=float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

    for kv_start in range(0, seq_len, BLOCK_KV):
        offs_kv = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = offs_kv < seq_len

        # Load K block: [BLOCK_KV, BLOCK_D]
        k_ptrs = k_ptr + pid_b * stride_kb + offs_kv[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=kv_mask[:, None] & (offs_d[None, :] < head_dim), other=0.0)

        # S = Q @ K^T: [BLOCK_Q, BLOCK_KV]
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(
            (offs_q[:, None] < seq_len) & kv_mask[None, :],
            s,
            float("-inf"),
        )

        # Online softmax
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])

        # Load V block: [BLOCK_KV, BLOCK_D]
        v_ptrs = v_ptr + pid_b * stride_vb + offs_kv[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=kv_mask[:, None] & (offs_d[None, :] < head_dim), other=0.0)

        # Update accumulator
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    acc = acc / l_i[:, None]

    o_ptrs = o_ptr + pid_b * stride_ob + offs_q[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc, mask=q_mask)


def flash_attention_fwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float | None = None
) -> torch.Tensor:
    """Flash-attention-inspired forward pass with online softmax and block-wise K/V iteration."""
    assert q.ndim == 3, "Inputs must be [batch, seq_len, head_dim]"
    batch, seq_len, head_dim = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    o = torch.empty_like(q)
    BLOCK_Q = 64
    BLOCK_KV = 64
    BLOCK_D = triton.next_power_of_2(head_dim)
    grid = (batch, triton.cdiv(seq_len, BLOCK_Q))
    _flash_attention_fwd_kernel[grid](
        q, k, v, o,
        seq_len, head_dim, scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        BLOCK_Q=BLOCK_Q, BLOCK_KV=BLOCK_KV, BLOCK_D=BLOCK_D,
    )
    return o
