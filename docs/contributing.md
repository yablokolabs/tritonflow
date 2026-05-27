# Contributing to TritonFlow

## Development Setup

```bash
git clone https://github.com/yablokolabs/tritonflow.git
cd tritonflow
./scripts/setup_dev.sh
# Or manually:
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
pre-commit install
```

## Code Standards

### Style
- **Formatter**: Black (line length 100)
- **Linter**: Ruff
- **Type checker**: mypy

Run all checks:
```bash
make lint
make format
```

### Kernel Conventions

1. Private kernel functions are prefixed with underscore: `_softmax_kernel`
2. Public wrapper functions match the kernel name without prefix: `softmax`
3. Every kernel module has an `__all__` list
4. Use `tl.constexpr` for block size parameters
5. Always mask loads/stores for edge blocks
6. Docstrings on public wrappers only

### Testing

- GPU-dependent tests use `@pytest.mark.gpu` (via the `@gpu` decorator from `conftest.py`)
- CPU tests must pass without a GPU
- Compare kernel output against PyTorch reference implementations
- Use `atol=1e-5` for FP32, `atol=1e-2` for FP16, `atol=0.5` for INT8

```bash
make test-cpu    # Run without GPU
make test-gpu    # Run GPU tests
make test        # Run everything
```

## Adding a New Kernel

1. Create the kernel in `tritonflow/kernels/your_module.py` following the standard pattern
2. Add the wrapper function to `__all__`
3. Write tests in `tests/test_your_module.py` comparing against PyTorch
4. Add a benchmark in `benchmarks/suite.py`
5. Document in `docs/kernels.md`

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-kernel`
3. Write tests first (TDD)
4. Implement the kernel
5. Run `make lint && make test-cpu`
6. Submit a PR with description of the kernel and expected performance characteristics

## Reporting Issues

Include:
- GPU model and driver version (`tritonflow info`)
- Python, PyTorch, and Triton versions
- Minimal reproduction script
- Expected vs actual behavior
