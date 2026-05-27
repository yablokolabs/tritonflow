"""Core benchmark framework for timing and comparing kernel implementations."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["BenchmarkResult", "BenchmarkRunner"]

try:
    import torch

    _HAS_TORCH = True
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    _HAS_TORCH = False
    _HAS_CUDA = False


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str
    category: str
    input_size: str
    triton_ms: float
    pytorch_ms: float | None = None
    numpy_ms: float | None = None
    speedup_vs_pytorch: float | None = None
    speedup_vs_numpy: float | None = None
    memory_bytes: int | None = None
    dtype: str = "float32"
    device: str = "cuda"


class BenchmarkRunner:
    """Run and collect benchmarks comparing Triton, PyTorch, and NumPy."""

    def __init__(self, warmup_iters: int = 10, bench_iters: int = 100):
        self.warmup_iters = warmup_iters
        self.bench_iters = bench_iters
        self.results: list[BenchmarkResult] = []

    def time_fn(self, fn: Callable, *args: Any, **kwargs: Any) -> float:
        """Time a function with warmup. Returns median time in ms.

        Uses torch.cuda.synchronize() if CUDA is available, otherwise
        falls back to wall-clock timing.
        """
        for _ in range(self.warmup_iters):
            fn(*args, **kwargs)

        times: list[float] = []
        for _ in range(self.bench_iters):
            if _HAS_CUDA:
                torch.cuda.synchronize()
                start = time.perf_counter()
                fn(*args, **kwargs)
                torch.cuda.synchronize()
            else:
                start = time.perf_counter()
                fn(*args, **kwargs)
            end = time.perf_counter()
            times.append((end - start) * 1000.0)

        return statistics.median(times)

    def benchmark(
        self,
        name: str,
        category: str,
        input_size: str,
        triton_fn: Callable,
        pytorch_fn: Callable | None = None,
        numpy_fn: Callable | None = None,
        dtype: str = "float32",
    ) -> BenchmarkResult:
        """Run a benchmark comparing Triton vs PyTorch vs NumPy."""
        triton_ms = self.time_fn(triton_fn)
        pytorch_ms = self.time_fn(pytorch_fn) if pytorch_fn is not None else None
        numpy_ms = self.time_fn(numpy_fn) if numpy_fn is not None else None

        speedup_vs_pytorch = pytorch_ms / triton_ms if pytorch_ms and triton_ms > 0 else None
        speedup_vs_numpy = numpy_ms / triton_ms if numpy_ms and triton_ms > 0 else None

        mem = None
        if _HAS_CUDA:
            mem = torch.cuda.max_memory_allocated()

        device = "cuda" if _HAS_CUDA else "cpu"
        result = BenchmarkResult(
            name=name,
            category=category,
            input_size=input_size,
            triton_ms=triton_ms,
            pytorch_ms=pytorch_ms,
            numpy_ms=numpy_ms,
            speedup_vs_pytorch=speedup_vs_pytorch,
            speedup_vs_numpy=speedup_vs_numpy,
            memory_bytes=mem,
            dtype=dtype,
            device=device,
        )
        self.results.append(result)
        return result

    def run_suite(self, suite: list[dict]) -> list[BenchmarkResult]:
        """Run a list of benchmark configs.

        Each dict must have keys matching :meth:`benchmark` parameters.
        """
        results: list[BenchmarkResult] = []
        for config in suite:
            result = self.benchmark(**config)
            results.append(result)
        return results
