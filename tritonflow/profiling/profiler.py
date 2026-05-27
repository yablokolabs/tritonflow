"""Profiling utility for Triton kernel execution."""

import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch

__all__ = ["KernelProfile", "TritonProfiler"]

_CUDA_AVAILABLE = torch.cuda.is_available()


@dataclass
class KernelProfile:
    """Timing and memory snapshot for a single kernel invocation."""

    name: str
    elapsed_ms: float
    memory_allocated_bytes: int
    memory_reserved_bytes: int
    num_calls: int = 1


class TritonProfiler:
    """Profile Triton kernel execution."""

    def __init__(self) -> None:
        self.profiles: list[KernelProfile] = []

    @contextmanager
    def profile_kernel(self, name: str):
        """Context manager to profile a kernel execution."""
        if _CUDA_AVAILABLE:
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated()
            res_before = torch.cuda.memory_reserved()

        start = time.perf_counter()
        yield

        if _CUDA_AVAILABLE:
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if _CUDA_AVAILABLE:
            mem_after = torch.cuda.memory_allocated()
            res_after = torch.cuda.memory_reserved()
            mem_alloc = mem_after - mem_before
            mem_res = res_after - res_before
        else:
            mem_alloc = 0
            mem_res = 0

        self.profiles.append(
            KernelProfile(
                name=name,
                elapsed_ms=elapsed_ms,
                memory_allocated_bytes=mem_alloc,
                memory_reserved_bytes=mem_res,
            )
        )

    def summary(self) -> str:
        """Return formatted summary table of all profiled kernels."""
        if not self.profiles:
            return "No profiles recorded."

        header = f"{'Kernel':<30} {'Calls':>6} {'Time (ms)':>12} {'Mem Alloc (B)':>15} {'Mem Rsv (B)':>15}"
        sep = "-" * len(header)
        lines = [sep, header, sep]

        aggregated: dict[str, KernelProfile] = {}
        for p in self.profiles:
            if p.name in aggregated:
                agg = aggregated[p.name]
                aggregated[p.name] = KernelProfile(
                    name=p.name,
                    elapsed_ms=agg.elapsed_ms + p.elapsed_ms,
                    memory_allocated_bytes=agg.memory_allocated_bytes + p.memory_allocated_bytes,
                    memory_reserved_bytes=agg.memory_reserved_bytes + p.memory_reserved_bytes,
                    num_calls=agg.num_calls + p.num_calls,
                )
            else:
                aggregated[p.name] = KernelProfile(
                    name=p.name,
                    elapsed_ms=p.elapsed_ms,
                    memory_allocated_bytes=p.memory_allocated_bytes,
                    memory_reserved_bytes=p.memory_reserved_bytes,
                    num_calls=p.num_calls,
                )

        for p in aggregated.values():
            lines.append(
                f"{p.name:<30} {p.num_calls:>6} {p.elapsed_ms:>12.3f} "
                f"{p.memory_allocated_bytes:>15} {p.memory_reserved_bytes:>15}"
            )
        lines.append(sep)
        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all recorded profiles."""
        self.profiles.clear()
