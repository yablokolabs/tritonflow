"""Tests for quantization kernels."""

from tests.conftest import HAS_TORCH, assert_close, gpu, make_tensor

if HAS_TORCH:
    import torch


@gpu
class TestQuantizedMatmulFP16:
    def test_basic(self):
        a = make_tensor(32, 64)
        b = make_tensor(64, 48)
        from tritonflow.kernels.quantization import quantized_matmul_fp16

        result = quantized_matmul_fp16(a, b)
        expected = torch.matmul(a.half(), b.half()).float()
        assert result.shape == (32, 48)
        assert_close(result, expected, atol=1e-2, rtol=1e-2)


@gpu
class TestDynamicQuantize:
    def test_basic(self):
        x = make_tensor(16, 64)
        from tritonflow.kernels.quantization import dynamic_quantize

        quantized, scales = dynamic_quantize(x)
        assert quantized.dtype == torch.int8
        assert scales.shape == (16,)
        # Dequantize and check closeness
        dequantized = quantized.float() * scales.unsqueeze(1)
        assert_close(dequantized, x, atol=0.5, rtol=0.1)


@gpu
class TestQuantizedMatmulINT8:
    def test_basic(self):
        a = make_tensor(32, 64)
        b = make_tensor(64, 48)
        from tritonflow.kernels.quantization import dynamic_quantize, quantized_matmul_int8

        qa, sa = dynamic_quantize(a)
        qb, sb = dynamic_quantize(b.T)
        result = quantized_matmul_int8(qa, qb.T, sa, sb)
        expected = torch.matmul(a, b)
        assert result.shape == (32, 48)
        # INT8 matmul has larger tolerance
        assert_close(result, expected, atol=2.0, rtol=0.2)
