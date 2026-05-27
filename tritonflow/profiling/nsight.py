"""Nsight Systems integration hooks via NVTX range markers."""

from contextlib import contextmanager

__all__ = ["nsight_range_push", "nsight_range_pop", "nsight_range"]


def nsight_range_push(name: str) -> None:
    """Push an NVTX range marker for Nsight profiling."""
    try:
        import torch.cuda.nvtx

        torch.cuda.nvtx.range_push(name)
    except (ImportError, AttributeError, RuntimeError):
        pass


def nsight_range_pop() -> None:
    """Pop the current NVTX range marker."""
    try:
        import torch.cuda.nvtx

        torch.cuda.nvtx.range_pop()
    except (ImportError, AttributeError, RuntimeError):
        pass


@contextmanager
def nsight_range(name: str):
    """Context manager for NVTX range markers."""
    nsight_range_push(name)
    try:
        yield
    finally:
        nsight_range_pop()
