"""Tests for normalization kernels."""

from tests.conftest import HAS_TORCH, assert_close, gpu, make_tensor

if HAS_TORCH:
    import torch


@gpu
class TestLayerNorm:
    def test_basic(self):
        x = make_tensor(32, 128)
        weight = torch.ones(128, device="cuda")
        bias = torch.zeros(128, device="cuda")
        from tritonflow.kernels.normalization import layer_norm

        result = layer_norm(x, weight, bias)
        expected = torch.nn.functional.layer_norm(x, [128], weight, bias)
        assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_with_learned_params(self):
        x = make_tensor(16, 64)
        weight = torch.randn(64, device="cuda")
        bias = torch.randn(64, device="cuda")
        from tritonflow.kernels.normalization import layer_norm

        result = layer_norm(x, weight, bias)
        expected = torch.nn.functional.layer_norm(x, [64], weight, bias)
        assert_close(result, expected, atol=1e-4, rtol=1e-4)


@gpu
class TestRMSNorm:
    def test_basic(self):
        x = make_tensor(32, 128)
        weight = torch.ones(128, device="cuda")
        from tritonflow.kernels.normalization import rms_norm

        result = rms_norm(x, weight)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        expected = x / rms * weight
        assert_close(result, expected, atol=1e-4, rtol=1e-4)
