.PHONY: install install-dev install-all test test-cpu test-gpu lint format typecheck bench clean docker-build docker-run pre-commit

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[all]"

test:
	pytest tests/ -v --tb=short

test-cpu:
	pytest tests/ -v --tb=short -m "not gpu"

test-gpu:
	pytest tests/ -v --tb=short -m "gpu"

lint:
	ruff check tritonflow/ tests/ benchmarks/
	mypy tritonflow/

format:
	ruff check --fix tritonflow/ tests/ benchmarks/
	black tritonflow/ tests/ benchmarks/

typecheck:
	mypy tritonflow/

bench:
	tritonflow benchmark --suite all

bench-report:
	tritonflow benchmark --suite all --export-md results/benchmark_report.md --export-csv results/benchmark_results.csv

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info .mypy_cache .pytest_cache .ruff_cache

docker-build:
	docker build -f docker/Dockerfile -t tritonflow:latest .

docker-run:
	docker run --gpus all -it tritonflow:latest

pre-commit:
	pre-commit install
	pre-commit run --all-files
