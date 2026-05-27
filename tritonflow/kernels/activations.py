"""Triton kernels for activation functions."""

import torch
import triton
import triton.language as tl

__all__ = ["softmax", "gelu", "fused_gelu_bias"]


@triton.jit
def _softmax_kernel(x_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(x_ptr + row * N + offsets, mask=mask, other=float("-inf"))
    x_max = tl.max(x, axis=0)
    x_exp = tl.exp(x - x_max)
    x_exp = tl.where(mask, x_exp, 0.0)
    x_sum = tl.sum(x_exp, axis=0)
    out = x_exp / x_sum

    tl.store(output_ptr + row * N + offsets, out, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax for 2D input [M, N]."""
    assert x.is_cuda and x.ndim == 2
    M, N = x.shape
    output = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    _softmax_kernel[(M,)](x, output, N, BLOCK_SIZE=BLOCK_SIZE)
    return output


@triton.jit
def _gelu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    # tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    inner = c * (x + 0.044715 * x * x * x)
    out = 0.5 * x * (1.0 + tl.math.tanh(inner))

    tl.store(output_ptr + offsets, out, mask=mask)


def gelu(x: torch.Tensor) -> torch.Tensor:
    """GELU activation using tanh approximation."""
    assert x.is_cuda
    output = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _gelu_kernel[grid](x, output, n, BLOCK_SIZE=1024)
    return output


@triton.jit
def _fused_gelu_bias_kernel(
    x_ptr, bias_ptr, output_ptr, n_rows, N, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row = pid // triton.cdiv(N, BLOCK_SIZE)
    col_block = pid % triton.cdiv(N, BLOCK_SIZE)
    offsets = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(x_ptr + row * N + offsets, mask=mask)
    b = tl.load(bias_ptr + offsets, mask=mask)
    x = x + b

    c = 0.7978845608028654  # sqrt(2/pi)
    inner = c * (x + 0.044715 * x * x * x)
    out = 0.5 * x * (1.0 + tl.math.tanh(inner))

    tl.store(output_ptr + row * N + offsets, out, mask=mask)


def fused_gelu_bias(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Fused GELU activation with bias addition in a single kernel pass."""
    assert x.is_cuda and bias.is_cuda
    assert x.ndim == 2
    M, N = x.shape
    output = torch.empty_like(x)
    BLOCK_SIZE = 1024
    n_col_blocks = triton.cdiv(N, BLOCK_SIZE)
    grid = (M * n_col_blocks,)
    _fused_gelu_bias_kernel[grid](x, bias, output, M, N, BLOCK_SIZE=BLOCK_SIZE)
    return output
