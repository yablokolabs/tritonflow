"""Tests for activation kernels."""

import pytest
import math
from tests.conftest import gpu, make_tensor, assert_close, HAS_TORCH

if HAS_TORCH:
    import torch


@gpu
class TestSoftmax:
    def test_basic(self):
        x = make_tensor(32, 128)
        from tritonflow.kernels.activations import softmax

        result = softmax(x)
        expected = torch.nn.functional.softmax(x, dim=-1)
        assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_sums_to_one(self):
        x = make_tensor(16, 64)
        from tritonflow.kernels.activations import softmax

        result = softmax(x)
        row_sums = result.sum(dim=-1)
        assert_close(row_sums, torch.ones(16, device="cuda"), atol=1e-5, rtol=1e-5)

    def test_numerical_stability(self):
        x = torch.tensor([[1000.0, 1001.0, 1002.0]], device="cuda")
        from tritonflow.kernels.activations import softmax

        result = softmax(x)
        assert not torch.any(torch.isnan(result))
        assert not torch.any(torch.isinf(result))


@gpu
class TestGELU:
    def test_basic(self):
        x = make_tensor(1024)
        from tritonflow.kernels.activations import gelu

        result = gelu(x)
        expected = torch.nn.functional.gelu(x, approximate="tanh")
        assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_zero_input(self):
        x = torch.zeros(128, device="cuda")
        from tritonflow.kernels.activations import gelu

        result = gelu(x)
        assert_close(result, torch.zeros(128, device="cuda"))


@gpu
class TestFusedGELUBias:
    def test_basic(self):
        x = make_tensor(32, 128)
        bias = make_tensor(128)
        from tritonflow.kernels.activations import fused_gelu_bias

        result = fused_gelu_bias(x, bias)
        expected = torch.nn.functional.gelu(x + bias, approximate="tanh")
        assert_close(result, expected, atol=1e-4, rtol=1e-4)
