"""Tests for utility modules (CPU-runnable)."""


class TestGPUUtils:
    def test_is_gpu_available_returns_bool(self):
        from tritonflow.utils.gpu import is_gpu_available

        result = is_gpu_available()
        assert isinstance(result, bool)

    def test_get_device_info_without_gpu(self):
        from tritonflow.utils.gpu import get_device_info, is_gpu_available

        info = get_device_info()
        if not is_gpu_available():
            assert info is None
        else:
            assert isinstance(info, dict)
            assert "name" in info


class TestDTypes:
    def test_dtype_enum_members(self):
        from tritonflow.utils.dtypes import DType

        assert hasattr(DType, "FP32")
        assert hasattr(DType, "FP16")
        assert hasattr(DType, "BF16")
        assert hasattr(DType, "INT8")

    def test_get_dtype_info(self):
        from tritonflow.utils.dtypes import DType, get_dtype_info

        info = get_dtype_info(DType.FP32)
        assert info["bits"] == 32


class TestAutotuning:
    def test_standard_block_sizes(self):
        from tritonflow.utils.autotuning import standard_block_sizes

        sizes = standard_block_sizes()
        assert isinstance(sizes, list)
        assert 128 in sizes
        assert 1024 in sizes

    def test_estimate_occupancy(self):
        from tritonflow.utils.autotuning import estimate_occupancy

        occ = estimate_occupancy(256, 0)
        assert 0.0 <= occ <= 1.0
