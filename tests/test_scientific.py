"""Tests for scientific computing kernels."""

from tests.conftest import HAS_TORCH, assert_close, gpu, make_tensor

if HAS_TORCH:
    import torch


@gpu
class TestParallelPrefixSum:
    def test_basic(self):
        x = torch.arange(1, 129, dtype=torch.float32, device="cuda")
        from tritonflow.kernels.scientific import parallel_prefix_sum

        result = parallel_prefix_sum(x)
        expected = torch.cumsum(x, dim=0)
        assert_close(result, expected)

    def test_small(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
        from tritonflow.kernels.scientific import parallel_prefix_sum

        result = parallel_prefix_sum(x)
        expected = torch.tensor([1.0, 3.0, 6.0, 10.0], device="cuda")
        assert_close(result, expected)


@gpu
class TestStencil1D:
    def test_averaging_filter(self):
        x = torch.ones(128, device="cuda")
        weights = torch.tensor([1.0 / 3, 1.0 / 3, 1.0 / 3], device="cuda")
        from tritonflow.kernels.scientific import stencil_1d

        result = stencil_1d(x, weights)
        # Interior elements of all-ones should remain ~1.0
        assert_close(result[1:-1], torch.ones(126, device="cuda"), atol=1e-5, rtol=1e-5)


@gpu
class TestBatchedOuterProduct:
    def test_basic(self):
        x = make_tensor(4, 8)
        y = make_tensor(4, 16)
        from tritonflow.kernels.scientific import batched_outer_product

        result = batched_outer_product(x, y)
        expected = torch.einsum("bi,bj->bij", x, y)
        assert result.shape == (4, 8, 16)
        assert_close(result, expected, atol=1e-4, rtol=1e-4)
