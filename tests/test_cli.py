"""Tests for CLI commands (CPU-runnable)."""

import pytest
from typer.testing import CliRunner


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def app():
    from tritonflow.cli.app import app

    return app


class TestCLIInfo:
    def test_info_command(self, runner, app):
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "TritonFlow" in result.output


class TestCLIBenchmark:
    def test_benchmark_help(self, runner, app):
        result = runner.invoke(app, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "benchmark" in result.output.lower()


class TestCLIProfile:
    def test_profile_help(self, runner, app):
        result = runner.invoke(app, ["profile", "--help"])
        assert result.exit_code == 0


class TestCLICompare:
    def test_compare_help(self, runner, app):
        result = runner.invoke(app, ["compare", "--help"])
        assert result.exit_code == 0


class TestCLIVisualize:
    def test_visualize_help(self, runner, app):
        result = runner.invoke(app, ["visualize", "--help"])
        assert result.exit_code == 0
