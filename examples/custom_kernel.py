"""Writing Custom Kernels with TritonFlow.

Demonstrates how to write your own Triton kernel following TritonFlow patterns,
and integrate it with the profiling and benchmarking infrastructure.
"""

import torch
import triton
import triton.language as tl

from tritonflow.utils.gpu import is_gpu_available, require_gpu
from tritonflow.profiling.profiler import TritonProfiler


@triton.jit
def _elu_kernel(x_ptr, output_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    """ELU activation: x if x > 0, alpha * (exp(x) - 1) otherwise."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    positive = tl.where(x > 0, x, 0.0)
    negative = tl.where(x <= 0, alpha * (tl.exp(x) - 1.0), 0.0)

    tl.store(output_ptr + offsets, positive + negative, mask=mask)


def elu(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """GPU-accelerated ELU activation using a custom Triton kernel."""
    assert x.is_cuda, "Input must be on CUDA device"
    output = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _elu_kernel[grid](x, output, alpha, n, BLOCK_SIZE=1024)
    return output


@require_gpu
def custom_kernel_demo():
    """Demonstrate a custom ELU kernel with profiling."""
    profiler = TritonProfiler()
    x = torch.randn(1_000_000, device="cuda")

    with profiler.profile_kernel("custom_elu"):
        result = elu(x)

    with profiler.profile_kernel("torch_elu"):
        expected = torch.nn.functional.elu(x)

    print(profiler.summary())

    diff = (result - expected).abs().max().item()
    print(f"\nMax difference vs PyTorch: {diff:.2e}")
    print(f"Output shape: {result.shape}")


if __name__ == "__main__":
    if not is_gpu_available():
        print("GPU required for this example. Skipping.")
    else:
        custom_kernel_demo()
