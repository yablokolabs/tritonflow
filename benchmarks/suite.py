"""Predefined benchmark suites for tritonflow kernels."""

from __future__ import annotations

from benchmarks.configs import TENSOR_SIZES

__all__ = ["vector_ops_suite", "ml_kernels_suite", "similarity_suite", "all_suites"]

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _torch_dtype(name: str) -> torch.dtype:
    return getattr(torch, name)


def _make_tensor(shape: tuple[int, ...], dtype: str = "float32") -> torch.Tensor:
    """Create a random tensor on the best available device."""
    device = "cuda" if _HAS_TORCH and torch.cuda.is_available() else "cpu"
    return torch.randn(shape, dtype=_torch_dtype(dtype), device=device)


# ------------------------------------------------------------------
# Vector operations
# ------------------------------------------------------------------


def vector_ops_suite() -> list[dict]:
    """Benchmarks for vector operations (vector_add, fused_add_mul, batched_sum)."""
    from tritonflow.kernels import batched_sum, fused_add_mul, vector_add

    configs: list[dict] = []
    for size_name, shape in TENSOR_SIZES.items():
        a = _make_tensor(shape)
        b = _make_tensor(shape)
        c = _make_tensor(shape)

        # vector_add
        configs.append(
            {
                "name": "vector_add",
                "category": "vector_ops",
                "input_size": size_name,
                "triton_fn": lambda _a=a, _b=b: vector_add(_a, _b),
                "pytorch_fn": lambda _a=a, _b=b: _a + _b,
                "numpy_fn": None,
                "dtype": "float32",
            }
        )

        # fused_add_mul
        configs.append(
            {
                "name": "fused_add_mul",
                "category": "vector_ops",
                "input_size": size_name,
                "triton_fn": lambda _a=a, _b=b, _c=c: fused_add_mul(_a, _b, _c),
                "pytorch_fn": lambda _a=a, _b=b, _c=c: (_a + _b) * _c,
                "numpy_fn": None,
                "dtype": "float32",
            }
        )

        # batched_sum
        configs.append(
            {
                "name": "batched_sum",
                "category": "vector_ops",
                "input_size": size_name,
                "triton_fn": lambda _a=a: batched_sum(_a),
                "pytorch_fn": lambda _a=a: _a.sum(),
                "numpy_fn": None,
                "dtype": "float32",
            }
        )

    return configs


# ------------------------------------------------------------------
# ML kernels
# ------------------------------------------------------------------


def ml_kernels_suite() -> list[dict]:
    """Benchmarks for ML kernels (softmax, gelu, layernorm)."""
    from tritonflow.kernels import gelu, layernorm, softmax

    configs: list[dict] = []
    for size_name, shape in TENSOR_SIZES.items():
        x = _make_tensor(shape)

        # softmax
        configs.append(
            {
                "name": "softmax",
                "category": "ml_kernels",
                "input_size": size_name,
                "triton_fn": lambda _x=x: softmax(_x),
                "pytorch_fn": lambda _x=x: torch.nn.functional.softmax(_x, dim=-1),
                "numpy_fn": None,
                "dtype": "float32",
            }
        )

        # gelu
        configs.append(
            {
                "name": "gelu",
                "category": "ml_kernels",
                "input_size": size_name,
                "triton_fn": lambda _x=x: gelu(_x),
                "pytorch_fn": lambda _x=x: torch.nn.functional.gelu(_x),
                "numpy_fn": None,
                "dtype": "float32",
            }
        )

        # layernorm
        weight = _make_tensor(shape)
        bias = _make_tensor(shape)
        configs.append(
            {
                "name": "layernorm",
                "category": "ml_kernels",
                "input_size": size_name,
                "triton_fn": lambda _x=x, _w=weight, _b=bias: layernorm(_x, _w, _b),
                "pytorch_fn": lambda _x=x, _w=weight, _b=bias: torch.nn.functional.layer_norm(
                    _x, _x.shape, _w, _b
                ),
                "numpy_fn": None,
                "dtype": "float32",
            }
        )

    return configs


# ------------------------------------------------------------------
# Similarity operations
# ------------------------------------------------------------------


def similarity_suite() -> list[dict]:
    """Benchmarks for similarity operations (cosine, dot product)."""
    from tritonflow.kernels import cosine_similarity, dot_product

    configs: list[dict] = []
    for size_name, shape in TENSOR_SIZES.items():
        a = _make_tensor(shape)
        b = _make_tensor(shape)

        configs.append(
            {
                "name": "cosine_similarity",
                "category": "similarity",
                "input_size": size_name,
                "triton_fn": lambda _a=a, _b=b: cosine_similarity(_a, _b),
                "pytorch_fn": lambda _a=a, _b=b: torch.nn.functional.cosine_similarity(
                    _a.unsqueeze(0), _b.unsqueeze(0)
                ),
                "numpy_fn": None,
                "dtype": "float32",
            }
        )

        configs.append(
            {
                "name": "dot_product",
                "category": "similarity",
                "input_size": size_name,
                "triton_fn": lambda _a=a, _b=b: dot_product(_a, _b),
                "pytorch_fn": lambda _a=a, _b=b: torch.dot(_a, _b),
                "numpy_fn": None,
                "dtype": "float32",
            }
        )

    return configs


# ------------------------------------------------------------------
# Combined
# ------------------------------------------------------------------


def all_suites() -> list[dict]:
    """All benchmarks combined."""
    return vector_ops_suite() + ml_kernels_suite() + similarity_suite()
