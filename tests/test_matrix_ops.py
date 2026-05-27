"""Tests for matrix operation kernels."""

from tests.conftest import HAS_TORCH, assert_close, gpu, make_tensor

if HAS_TORCH:
    import torch


@gpu
class TestMatrixTranspose:
    def test_square(self):
        x = make_tensor(64, 64)
        from tritonflow.kernels.matrix_ops import matrix_transpose

        result = matrix_transpose(x)
        assert_close(result, x.T)

    def test_rectangular(self):
        x = make_tensor(32, 128)
        from tritonflow.kernels.matrix_ops import matrix_transpose

        result = matrix_transpose(x)
        assert result.shape == (128, 32)
        assert_close(result, x.T)


@gpu
class TestMatmul:
    def test_square(self):
        a = make_tensor(64, 64)
        b = make_tensor(64, 64)
        from tritonflow.kernels.matrix_ops import matmul

        result = matmul(a, b)
        expected = torch.matmul(a, b)
        assert_close(result, expected, atol=1e-3, rtol=1e-3)

    def test_rectangular(self):
        a = make_tensor(32, 64)
        b = make_tensor(64, 48)
        from tritonflow.kernels.matrix_ops import matmul

        result = matmul(a, b)
        expected = torch.matmul(a, b)
        assert result.shape == (32, 48)
        assert_close(result, expected, atol=1e-3, rtol=1e-3)
