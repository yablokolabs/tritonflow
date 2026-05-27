import torch
import triton
import triton.language as tl

__all__ = ["matrix_transpose", "matmul"]


@triton.jit
def _matrix_transpose_kernel(
    input_ptr,
    output_ptr,
    M,
    N,
    stride_im,
    stride_in,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    in_ptrs = input_ptr + offs_m[:, None] * stride_im + offs_n[None, :] * stride_in
    x = tl.load(in_ptrs, mask=mask, other=0.0)

    out_ptrs = output_ptr + offs_n[None, :] * stride_om + offs_m[:, None] * stride_on
    tl.store(out_ptrs, x, mask=mask)


def matrix_transpose(x: torch.Tensor, BLOCK: int = 32) -> torch.Tensor:
    """Transpose a 2D matrix using a tiled Triton kernel."""
    assert x.ndim == 2, "Input must be 2D"
    M, N = x.shape
    output = torch.empty(N, M, device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(M, BLOCK), triton.cdiv(N, BLOCK))
    _matrix_transpose_kernel[grid](
        x, output, M, N,
        x.stride(0), x.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK, BLOCK_N=BLOCK,
    )
    return output


@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Matrix multiplication using a tiled Triton kernel with blocked accumulation."""
    assert a.ndim == 2 and b.ndim == 2, "Inputs must be 2D"
    assert a.shape[1] == b.shape[0], "Inner dimensions must match"
    M, K = a.shape
    _, N = b.shape
    c = torch.empty(M, N, device=a.device, dtype=torch.float32)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c
