"""GPU detection and device info utilities."""

from __future__ import annotations

import functools
from typing import Any

try:
    import torch
    import torch.cuda

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def is_gpu_available() -> bool:
    """Check if a CUDA GPU is available."""
    if not _TORCH_AVAILABLE:
        return False
    return torch.cuda.is_available()


def get_device_info() -> dict[str, Any] | None:
    """Return basic info about the default CUDA device.

    Returns None if no GPU is available.
    """
    if not is_gpu_available():
        return None

    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "compute_capability": (props.major, props.minor),
        "total_memory_gb": round(props.total_memory / (1024**3), 2),
        "sm_count": props.multi_processor_count,
        "driver_version": torch.version.cuda or "unknown",
    }


def get_device_properties(device_id: int = 0) -> dict[str, Any]:
    """Return detailed properties for a specific CUDA device.

    Raises RuntimeError if no GPU is available.
    """
    if not is_gpu_available():
        raise RuntimeError("No CUDA GPU available")

    props = torch.cuda.get_device_properties(device_id)
    return {
        "name": props.name,
        "compute_capability": (props.major, props.minor),
        "total_memory_bytes": props.total_memory,
        "total_memory_gb": round(props.total_memory / (1024**3), 2),
        "sm_count": props.multi_processor_count,
        "max_threads_per_sm": props.max_threads_per_multi_processor,
        "warp_size": props.warp_size,
        "max_shared_memory_per_block": props.shared_memory_per_block,
        "driver_version": torch.version.cuda or "unknown",
    }


def require_gpu():
    """Decorator that raises RuntimeError if no CUDA GPU is available."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not is_gpu_available():
                raise RuntimeError(
                    f"Function '{fn.__name__}' requires a CUDA GPU, but none is available."
                )
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def get_memory_stats(device_id: int = 0) -> dict[str, int]:
    """Return current GPU memory statistics in bytes.

    Returns dict with allocated, reserved, and free memory.
    Raises RuntimeError if no GPU is available.
    """
    if not is_gpu_available():
        raise RuntimeError("No CUDA GPU available")

    torch.cuda.set_device(device_id)
    total = torch.cuda.get_device_properties(device_id).total_memory
    allocated = torch.cuda.memory_allocated(device_id)
    reserved = torch.cuda.memory_reserved(device_id)
    free = total - allocated

    return {
        "allocated": allocated,
        "reserved": reserved,
        "free": free,
        "total": total,
    }
