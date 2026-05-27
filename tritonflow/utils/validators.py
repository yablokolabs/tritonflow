"""Input validation utilities for tensor operations."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class TensorSpec(BaseModel):
    """Specification for validating tensor properties."""

    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    device: str | None = None
    min_ndim: int | None = None
    max_ndim: int | None = None


def validate_tensor(tensor: Any, spec: TensorSpec) -> None:
    """Validate that a tensor matches the given specification.

    Raises ValueError with a descriptive message on mismatch.
    Raises TypeError if the input is not a torch Tensor.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required for validate_tensor")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor).__name__}")

    if spec.shape is not None and tuple(tensor.shape) != spec.shape:
        raise ValueError(
            f"Expected shape {spec.shape}, got {tuple(tensor.shape)}"
        )

    if spec.dtype is not None:
        actual_dtype = str(tensor.dtype).replace("torch.", "")
        expected = spec.dtype.replace("torch.", "")
        if actual_dtype != expected:
            raise ValueError(f"Expected dtype {expected}, got {actual_dtype}")

    if spec.device is not None:
        actual_device = str(tensor.device)
        if not actual_device.startswith(spec.device):
            raise ValueError(f"Expected device {spec.device}, got {actual_device}")

    ndim = tensor.ndim
    if spec.min_ndim is not None and ndim < spec.min_ndim:
        raise ValueError(
            f"Expected at least {spec.min_ndim} dimensions, got {ndim}"
        )
    if spec.max_ndim is not None and ndim > spec.max_ndim:
        raise ValueError(
            f"Expected at most {spec.max_ndim} dimensions, got {ndim}"
        )


def validate_same_device(*tensors: Any) -> None:
    """Ensure all tensors are on the same device.

    Raises ValueError if tensors reside on different devices.
    Raises TypeError if any input is not a torch Tensor.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required for validate_same_device")
    if len(tensors) < 2:
        return

    devices = []
    for i, t in enumerate(tensors):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"Argument {i} is not a torch.Tensor (got {type(t).__name__})")
        devices.append(str(t.device))

    if len(set(devices)) > 1:
        raise ValueError(f"All tensors must be on the same device, got {devices}")


def validate_contiguous(tensor: Any) -> None:
    """Ensure a tensor is contiguous in memory.

    Raises ValueError if the tensor is not contiguous.
    Raises TypeError if the input is not a torch Tensor.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required for validate_contiguous")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor).__name__}")
    if not tensor.is_contiguous():
        raise ValueError(
            f"Tensor with shape {tuple(tensor.shape)} is not contiguous"
        )
