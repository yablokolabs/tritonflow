"""Shared test fixtures and configuration."""

import pytest
import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import triton

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

HAS_GPU = HAS_TORCH and torch.cuda.is_available()

gpu = pytest.mark.skipif(not HAS_GPU, reason="CUDA GPU not available")
requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
requires_triton = pytest.mark.skipif(not HAS_TRITON, reason="Triton not installed")


@pytest.fixture
def device():
    return "cuda" if HAS_GPU else "cpu"


@pytest.fixture
def cuda_device():
    if not HAS_GPU:
        pytest.skip("CUDA GPU not available")
    return "cuda"


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def torch_rng():
    if not HAS_TORCH:
        pytest.skip("PyTorch not installed")
    g = torch.Generator()
    g.manual_seed(42)
    return g


def make_tensor(*shape, dtype=None, device="cuda", requires_grad=False):
    """Create a random tensor on the specified device."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not installed")
    if device == "cuda" and not HAS_GPU:
        pytest.skip("CUDA GPU not available")
    dtype = dtype or torch.float32
    return torch.randn(*shape, dtype=dtype, device=device, requires_grad=requires_grad)


def assert_close(actual, expected, atol=1e-5, rtol=1e-5):
    """Assert two tensors are close within tolerance."""
    if not HAS_TORCH:
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    else:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
