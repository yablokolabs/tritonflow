"""Tests for vector operations kernels."""

from tests.conftest import HAS_TORCH, assert_close, gpu, make_tensor

if HAS_TORCH:
    pass


@gpu
class TestVectorAdd:
    def test_basic_addition(self):
        x = make_tensor(1024)
        y = make_tensor(1024)
        from tritonflow.kernels.vector_ops import vector_add

        result = vector_add(x, y)
        assert_close(result, x + y)

    def test_large_tensor(self):
        x = make_tensor(1_000_000)
        y = make_tensor(1_000_000)
        from tritonflow.kernels.vector_ops import vector_add

        result = vector_add(x, y)
        assert_close(result, x + y)

    def test_non_power_of_two_size(self):
        x = make_tensor(1023)
        y = make_tensor(1023)
        from tritonflow.kernels.vector_ops import vector_add

        result = vector_add(x, y)
        assert_close(result, x + y)


@gpu
class TestFusedAddMul:
    def test_basic(self):
        x = make_tensor(1024)
        y = make_tensor(1024)
        z = make_tensor(1024)
        from tritonflow.kernels.vector_ops import fused_add_mul

        result = fused_add_mul(x, y, z)
        assert_close(result, (x + y) * z)

    def test_large_tensor(self):
        x = make_tensor(500_000)
        y = make_tensor(500_000)
        z = make_tensor(500_000)
        from tritonflow.kernels.vector_ops import fused_add_mul

        result = fused_add_mul(x, y, z)
        assert_close(result, (x + y) * z)


@gpu
class TestBatchedReductions:
    def test_batched_sum(self):
        x = make_tensor(64, 128)
        from tritonflow.kernels.vector_ops import batched_sum

        result = batched_sum(x)
        assert_close(result, x.sum(dim=1), atol=1e-4, rtol=1e-4)

    def test_batched_mean(self):
        x = make_tensor(32, 256)
        from tritonflow.kernels.vector_ops import batched_mean

        result = batched_mean(x)
        assert_close(result, x.mean(dim=1), atol=1e-4, rtol=1e-4)

    def test_batched_max(self):
        x = make_tensor(16, 512)
        from tritonflow.kernels.vector_ops import batched_max

        result = batched_max(x)
        assert_close(result, x.max(dim=1).values)

    def test_batched_min(self):
        x = make_tensor(16, 512)
        from tritonflow.kernels.vector_ops import batched_min

        result = batched_min(x)
        assert_close(result, x.min(dim=1).values)
