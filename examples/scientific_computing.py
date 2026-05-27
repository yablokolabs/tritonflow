"""Scientific Computing with TritonFlow.

Demonstrates GPU-accelerated scientific computing operations:
prefix sums, stencil computations, and batched outer products.
"""

import torch
from tritonflow.utils.gpu import is_gpu_available, require_gpu


@require_gpu
def scientific_computing_demo():
    """Run scientific computing operations using TritonFlow kernels."""
    from tritonflow.kernels.scientific import parallel_prefix_sum, stencil_1d, batched_outer_product
    from tritonflow.kernels.vector_ops import batched_sum, batched_mean
    from tritonflow.profiling.profiler import TritonProfiler

    profiler = TritonProfiler()

    # Parallel prefix sum (scan)
    x = torch.arange(1, 1025, dtype=torch.float32, device="cuda")
    with profiler.profile_kernel("parallel_prefix_sum"):
        cumsum = parallel_prefix_sum(x)

    # 1D stencil (smoothing filter)
    signal = torch.randn(4096, device="cuda")
    weights = torch.tensor([0.25, 0.5, 0.25], device="cuda")
    with profiler.profile_kernel("stencil_1d"):
        smoothed = stencil_1d(signal, weights)

    # Batched outer product
    a = torch.randn(16, 64, device="cuda")
    b = torch.randn(16, 128, device="cuda")
    with profiler.profile_kernel("batched_outer_product"):
        outer = batched_outer_product(a, b)

    # Batched reductions
    data = torch.randn(32, 1024, device="cuda")
    with profiler.profile_kernel("batched_sum"):
        row_sums = batched_sum(data)
    with profiler.profile_kernel("batched_mean"):
        row_means = batched_mean(data)

    print(profiler.summary())
    print(f"\nResults:")
    print(f"  Prefix sum of 1..1024: last element = {cumsum[-1].item():.0f} "
          f"(expected: {1024 * 1025 / 2:.0f})")
    print(f"  Smoothed signal shape: {smoothed.shape}")
    print(f"  Outer product shape:   {outer.shape}")
    print(f"  Row sums shape:        {row_sums.shape}")
    print(f"  Row means shape:       {row_means.shape}")


if __name__ == "__main__":
    if not is_gpu_available():
        print("GPU required for this example. Skipping.")
    else:
        scientific_computing_demo()
