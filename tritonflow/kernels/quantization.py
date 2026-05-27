import torch
import triton
import triton.language as tl

__all__ = ["quantized_matmul_fp16", "dynamic_quantize", "quantized_matmul_int8"]


@triton.jit
def _quantized_matmul_fp16_kernel(
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
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float16)

        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float16)

        acc += tl.dot(a, b)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def quantized_matmul_fp16(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Matrix multiply with FP16 computation, returning FP32 result."""
    assert a.ndim == 2 and b.ndim == 2
    assert a.shape[1] == b.shape[0]
    M, K = a.shape
    _, N = b.shape
    c = torch.empty(M, N, device=a.device, dtype=torch.float32)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _quantized_matmul_fp16_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c


@triton.jit
def _dynamic_quantize_kernel(
    x_ptr, out_ptr, scale_ptr,
    cols,
    stride_xm, stride_xd,
    stride_om, stride_od,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)

    # Compute per-row max absolute value
    row_max = tl.zeros((1,), dtype=tl.float32)
    for d_start in range(0, cols, BLOCK_D):
        d_offs = d_start + offs_d
        d_mask = d_offs < cols
        x = tl.load(x_ptr + pid * stride_xm + d_offs * stride_xd, mask=d_mask, other=0.0)
        row_max = tl.maximum(row_max, tl.max(tl.abs(x)))

    scale = row_max / 127.0
    scale = tl.maximum(scale, 1e-10)
    tl.store(scale_ptr + pid, scale)

    # Quantize
    for d_start in range(0, cols, BLOCK_D):
        d_offs = d_start + offs_d
        d_mask = d_offs < cols
        x = tl.load(x_ptr + pid * stride_xm + d_offs * stride_xd, mask=d_mask, other=0.0)
        q = libdevice_round(x / scale)
        q = tl.minimum(tl.maximum(q, -127.0), 127.0)
        tl.store(out_ptr + pid * stride_om + d_offs * stride_od, q, mask=d_mask)


@triton.jit
def libdevice_round(x):
    return (x + 0.5 * tl.where(x >= 0, 1.0, -1.0)).to(tl.int32).to(tl.float32)


def dynamic_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize FP32 tensor to INT8 with per-row scaling. Returns (quantized, scales)."""
    assert x.ndim == 2
    M, D = x.shape
    quantized = torch.empty(M, D, device=x.device, dtype=torch.int8)
    scales = torch.empty(M, device=x.device, dtype=torch.float32)
    # Use float32 intermediate then cast to int8
    q_float = torch.empty(M, D, device=x.device, dtype=torch.float32)
    BLOCK_D = min(triton.next_power_of_2(D), 1024)
    _dynamic_quantize_kernel[(M,)](
        x, q_float, scales,
        D,
        x.stride(0), x.stride(1),
        q_float.stride(0), q_float.stride(1),
        BLOCK_D=BLOCK_D,
    )
    quantized = q_float.to(torch.int8)
    return quantized, scales


@triton.jit
def _quantized_matmul_int8_kernel(
    a_ptr, b_ptr, c_ptr,
    scale_a_ptr, scale_b_ptr,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0).to(tl.int32)

        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0).to(tl.int32)

        # Accumulate in int32
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32)).to(tl.int32)

    # Dequantize: apply per-row scales
    sa = tl.load(scale_a_ptr + offs_m, mask=offs_m < M, other=1.0)
    sb = tl.load(scale_b_ptr + offs_n, mask=offs_n < N, other=1.0)

    result = acc.to(tl.float32) * sa[:, None] * sb[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, result, mask=c_mask)


def quantized_matmul_int8(
    a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor
) -> torch.Tensor:
    """INT8 matrix multiply with dequantization via per-row scales."""
    assert a.ndim == 2 and b.ndim == 2
    assert a.shape[1] == b.shape[0]
    M, K = a.shape
    _, N = b.shape
    c = torch.empty(M, N, device=a.device, dtype=torch.float32)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _quantized_matmul_int8_kernel[grid](
        a, b, c,
        scale_a, scale_b,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c
