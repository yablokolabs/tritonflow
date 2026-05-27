"""Triton kernels for normalization operations."""

import torch
import triton
import triton.language as tl

__all__ = ["layer_norm", "rms_norm"]


@triton.jit
def _layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(x_ptr + row * N + offsets, mask=mask, other=0.0)

    # Two-pass: mean then variance
    mean = tl.sum(x, axis=0) / N
    centered = x - mean
    var = tl.sum(centered * centered, axis=0) / N
    inv_std = 1.0 / tl.sqrt(var + eps)

    x_norm = centered * inv_std
    w = tl.load(weight_ptr + offsets, mask=mask)
    b = tl.load(bias_ptr + offsets, mask=mask)
    out = x_norm * w + b

    tl.store(output_ptr + row * N + offsets, out, mask=mask)


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Layer normalization over the last dimension of a 2D input [M, N]."""
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.ndim == 2
    M, N = x.shape
    output = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    _layer_norm_kernel[(M,)](x, weight, bias, output, N, eps, BLOCK_SIZE=BLOCK_SIZE)
    return output


@triton.jit
def _rms_norm_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(x_ptr + row * N + offsets, mask=mask, other=0.0)

    ms = tl.sum(x * x, axis=0) / N
    rms = tl.sqrt(ms + eps)

    w = tl.load(weight_ptr + offsets, mask=mask)
    out = x * w / rms

    tl.store(output_ptr + row * N + offsets, out, mask=mask)


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """RMS normalization over the last dimension of a 2D input [M, N]."""
    assert x.is_cuda and weight.is_cuda
    assert x.ndim == 2
    M, N = x.shape
    output = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    _rms_norm_kernel[(M,)](x, weight, output, N, eps, BLOCK_SIZE=BLOCK_SIZE)
    return output
