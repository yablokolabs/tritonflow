"""Triton kernels for vector operations."""

import torch
import triton
import triton.language as tl

__all__ = [
    "vector_add",
    "fused_add_mul",
    "batched_sum",
    "batched_mean",
    "batched_max",
    "batched_min",
]


@triton.jit
def _vector_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Element-wise addition of two 1D tensors."""
    assert x.is_cuda and y.is_cuda
    output = torch.empty_like(x)
    n = output.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _vector_add_kernel[grid](x, y, output, n, BLOCK_SIZE=1024)
    return output


@triton.jit
def _fused_add_mul_kernel(x_ptr, y_ptr, z_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = tl.load(z_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, (x + y) * z, mask=mask)


def fused_add_mul(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Computes (x + y) * z in a single kernel pass."""
    assert x.is_cuda and y.is_cuda and z.is_cuda
    output = torch.empty_like(x)
    n = output.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _fused_add_mul_kernel[grid](x, y, z, output, n, BLOCK_SIZE=1024)
    return output


@triton.jit
def _batched_sum_kernel(x_ptr, output_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + row * N + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + row, tl.sum(x, axis=0))


def batched_sum(x: torch.Tensor) -> torch.Tensor:
    """Row-wise sum reduction of a 2D tensor [M, N] -> [M]."""
    assert x.is_cuda and x.ndim == 2
    M, N = x.shape
    output = torch.empty(M, device=x.device, dtype=x.dtype)
    BLOCK_SIZE = triton.next_power_of_2(N)
    _batched_sum_kernel[(M,)](x, output, M, N, BLOCK_SIZE=BLOCK_SIZE)
    return output


def batched_mean(x: torch.Tensor) -> torch.Tensor:
    """Row-wise mean of a 2D tensor [M, N] -> [M]."""
    assert x.is_cuda and x.ndim == 2
    return batched_sum(x) / x.shape[1]


@triton.jit
def _batched_max_kernel(x_ptr, output_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + row * N + offsets, mask=mask, other=float("-inf"))
    tl.store(output_ptr + row, tl.max(x, axis=0))


def batched_max(x: torch.Tensor) -> torch.Tensor:
    """Row-wise max of a 2D tensor [M, N] -> [M]."""
    assert x.is_cuda and x.ndim == 2
    M, N = x.shape
    output = torch.empty(M, device=x.device, dtype=x.dtype)
    BLOCK_SIZE = triton.next_power_of_2(N)
    _batched_max_kernel[(M,)](x, output, M, N, BLOCK_SIZE=BLOCK_SIZE)
    return output


@triton.jit
def _batched_min_kernel(x_ptr, output_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + row * N + offsets, mask=mask, other=float("inf"))
    tl.store(output_ptr + row, tl.min(x, axis=0))


def batched_min(x: torch.Tensor) -> torch.Tensor:
    """Row-wise min of a 2D tensor [M, N] -> [M]."""
    assert x.is_cuda and x.ndim == 2
    M, N = x.shape
    output = torch.empty(M, device=x.device, dtype=x.dtype)
    BLOCK_SIZE = triton.next_power_of_2(N)
    _batched_min_kernel[(M,)](x, output, M, N, BLOCK_SIZE=BLOCK_SIZE)
    return output
