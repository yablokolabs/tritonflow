"""Scientific computing kernels: prefix sum, stencil, and batched outer product."""

import torch
import triton
import triton.language as tl

__all__ = [
    "parallel_prefix_sum",
    "stencil_1d",
    "batched_outer_product",
]


# ---------------------------------------------------------------------------
# Prefix Sum (inclusive scan) – Blelloch-style within a single block
# ---------------------------------------------------------------------------


@triton.jit
def _prefix_sum_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Inclusive scan via Hillis-Steele (iterative doubling).
    # Each iteration adds the value `stride` positions to the left.
    val = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    local = tl.arange(0, BLOCK_SIZE)
    for i in tl.static_range(0, 10):  # up to BLOCK_SIZE 1024
        stride = 1 << i
        if stride >= BLOCK_SIZE:
            break
        tl.store(out_ptr + offsets, val, mask=mask)
        src = offsets - stride
        src_mask = (local >= stride) & mask
        prev = tl.load(out_ptr + src, mask=src_mask, other=0.0)
        val = val + prev

    tl.store(out_ptr + offsets, val, mask=mask)


def parallel_prefix_sum(x: torch.Tensor) -> torch.Tensor:
    """Inclusive prefix sum (scan) of a 1D tensor.

    Uses Hillis-Steele iterative doubling within a single block.
    For multi-block scans a second pass to propagate block totals is required.
    """
    assert x.ndim == 1, "Input must be 1D"
    n = x.numel()
    BLOCK_SIZE = triton.next_power_of_2(n)
    BLOCK_SIZE = max(BLOCK_SIZE, 1)
    out = torch.empty_like(x)
    _prefix_sum_kernel[(1,)](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


# ---------------------------------------------------------------------------
# 1-D Stencil (convolution with small kernel)
# ---------------------------------------------------------------------------


@triton.jit
def _stencil_1d_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    n_elements,
    half_w: tl.constexpr,
    W_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    out_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    out_mask = out_offsets < n_elements

    # Accumulate weighted contributions from each position in the stencil
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for k in tl.static_range(0, W_SIZE):
        src = block_start - half_w + tl.arange(0, BLOCK_SIZE) + k
        src_mask = (src >= 0) & (src < n_elements)
        val = tl.load(x_ptr + src, mask=src_mask, other=0.0)
        w_k = tl.load(w_ptr + k)
        acc += w_k * val

    tl.store(out_ptr + out_offsets, acc, mask=out_mask)


def stencil_1d(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """1D stencil (convolution) with a small kernel.

    Args:
        x: 1D input tensor.
        weights: 1D weight tensor of odd length.

    Returns:
        Output tensor of same shape as x (zero-padded boundary).
    """
    assert x.ndim == 1, "Input must be 1D"
    assert weights.ndim == 1, "Weights must be 1D"
    assert weights.numel() % 2 == 1, "Weight length must be odd"

    n = x.numel()
    w_size = weights.numel()
    half_w = w_size // 2

    BLOCK_SIZE = 1024
    # Ensure BLOCK_SIZE + 2*half_w is a power of two (Triton requirement on arange)
    halo_total = BLOCK_SIZE + 2 * half_w
    halo_total = triton.next_power_of_2(halo_total)
    # Adjust BLOCK_SIZE so that BLOCK_SIZE + 2*half_w == halo_total
    BLOCK_SIZE = halo_total - 2 * half_w

    out = torch.empty_like(x)
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _stencil_1d_kernel[grid](
        x, weights, out, n,
        half_w=half_w,
        W_SIZE=w_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


# ---------------------------------------------------------------------------
# Batched Outer Product
# ---------------------------------------------------------------------------

@triton.jit
def _batched_outer_product_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    M,
    N,
    BLOCK_N: tl.constexpr,
):
    # Each program computes one (batch, row) combination
    pid = tl.program_id(0)  # linear index over B*M
    batch = pid // M
    row = pid % M

    # Load scalar x[batch, row]
    x_val = tl.load(x_ptr + batch * M + row)

    # Load y[batch, :] in chunks of BLOCK_N
    col_offsets = tl.arange(0, BLOCK_N)
    y_base = batch * N
    out_base = batch * M * N + row * N

    for col_start in range(0, N, BLOCK_N):
        cols = col_start + col_offsets
        mask = cols < N
        y_val = tl.load(y_ptr + y_base + cols, mask=mask, other=0.0)
        tl.store(out_ptr + out_base + cols, x_val * y_val, mask=mask)


def batched_outer_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Batched outer product: x[B,M] ⊗ y[B,N] → out[B,M,N].

    Args:
        x: Tensor of shape [B, M].
        y: Tensor of shape [B, N].

    Returns:
        Tensor of shape [B, M, N].
    """
    assert x.ndim == 2 and y.ndim == 2, "Inputs must be 2D [B, *]"
    B, M = x.shape
    B2, N = y.shape
    assert B == B2, "Batch dimensions must match"

    out = torch.empty((B, M, N), device=x.device, dtype=x.dtype)
    BLOCK_N = triton.next_power_of_2(N)
    BLOCK_N = min(BLOCK_N, 1024)
    grid = (B * M,)
    _batched_outer_product_kernel[grid](x, y, out, M, N, BLOCK_N=BLOCK_N)
    return out
