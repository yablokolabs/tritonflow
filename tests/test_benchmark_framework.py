"""Tests for benchmark framework (CPU-runnable)."""

import time


class TestBenchmarkResult:
    def test_creation(self):
        from benchmarks.framework import BenchmarkResult

        result = BenchmarkResult(
            name="test_kernel",
            category="vector_ops",
            input_size="1024",
            triton_ms=0.5,
            pytorch_ms=1.0,
            speedup_vs_pytorch=2.0,
        )
        assert result.name == "test_kernel"
        assert result.speedup_vs_pytorch == 2.0


class TestBenchmarkRunner:
    def test_time_fn_cpu(self):
        from benchmarks.framework import BenchmarkRunner

        runner = BenchmarkRunner(warmup_iters=2, bench_iters=5)

        def dummy():
            time.sleep(0.001)

        elapsed = runner.time_fn(dummy)
        assert elapsed > 0


class TestBenchmarkReporter:
    def test_markdown_report(self, tmp_path):
        from benchmarks.framework import BenchmarkResult
        from benchmarks.reporters import BenchmarkReporter

        results = [
            BenchmarkResult(
                name="vector_add",
                category="vector_ops",
                input_size="1M",
                triton_ms=0.1,
                pytorch_ms=0.3,
                speedup_vs_pytorch=3.0,
            ),
        ]
        reporter = BenchmarkReporter(results)
        md = reporter.to_markdown()
        assert "vector_add" in md
        assert "vector_ops" in md

    def test_csv_export(self, tmp_path):
        from benchmarks.framework import BenchmarkResult
        from benchmarks.reporters import BenchmarkReporter

        results = [
            BenchmarkResult(
                name="softmax",
                category="ml_kernels",
                input_size="1K",
                triton_ms=0.05,
            ),
        ]
        reporter = BenchmarkReporter(results)
        csv_path = str(tmp_path / "results.csv")
        reporter.to_csv(csv_path)

        with open(csv_path) as f:
            content = f.read()
        assert "softmax" in content
