"""Dtype helper utilities for tensor operations."""

from __future__ import annotations

import enum
from typing import Any

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from tritonflow.utils.gpu import is_gpu_available


class DType(enum.Enum):
    """Supported data types for tritonflow operations."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"

    def to_torch(self):
        """Convert to the corresponding torch dtype."""
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required to convert DType to torch dtype")
        return _DTYPE_TO_TORCH[self]


if _TORCH_AVAILABLE:
    _DTYPE_TO_TORCH = {
        DType.FP32: torch.float32,
        DType.FP16: torch.float16,
        DType.BF16: torch.bfloat16,
        DType.INT8: torch.int8,
    }


_DTYPE_INFO: dict[DType, dict[str, Any]] = {
    DType.FP32: {
        "bits": 32,
        "max_value": 3.4028235e38,
        "min_value": -3.4028235e38,
        "eps": 1.1920929e-7,
    },
    DType.FP16: {
        "bits": 16,
        "max_value": 65504.0,
        "min_value": -65504.0,
        "eps": 9.765625e-4,
    },
    DType.BF16: {
        "bits": 16,
        "max_value": 3.3895314e38,
        "min_value": -3.3895314e38,
        "eps": 0.0078125,
    },
    DType.INT8: {
        "bits": 8,
        "max_value": 127,
        "min_value": -128,
        "eps": 1,
    },
}


def supports_bf16() -> bool:
    """Check if the current GPU supports BF16 (compute capability >= 8.0)."""
    if not _TORCH_AVAILABLE or not is_gpu_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return bool(props.major >= 8)


def get_dtype_info(dtype: DType) -> dict[str, Any]:
    """Return numeric properties for the given dtype."""
    return dict(_DTYPE_INFO[dtype])


def cast_tensor(tensor: Any, target_dtype: DType) -> Any:
    """Cast a torch tensor to the target dtype with validation.

    Raises TypeError if the input is not a torch Tensor.
    Raises ValueError if casting to BF16 on unsupported hardware.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required for cast_tensor")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor).__name__}")

    torch_dtype = target_dtype.to_torch()

    if target_dtype == DType.BF16 and tensor.is_cuda and not supports_bf16():
        raise ValueError("Current GPU does not support BF16")

    if target_dtype == DType.INT8 and tensor.is_floating_point():
        tensor = tensor.clamp(
            _DTYPE_INFO[DType.INT8]["min_value"],
            _DTYPE_INFO[DType.INT8]["max_value"],
        )
        tensor = tensor.round()

    return tensor.to(torch_dtype)
