#!/usr/bin/env bash
set -euo pipefail

echo "=== TritonFlow Development Environment Setup ==="

# Check Python version
python_version=$(python3 --version 2>&1)
echo "Python: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing tritonflow with all dependencies..."
pip install -e ".[all]"

echo "Installing pre-commit hooks..."
pre-commit install

# Check GPU availability
echo ""
echo "=== GPU Status ==="
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA: {torch.version.cuda}')
    mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'Memory: {mem:.1f} GB')
else:
    print('No GPU detected. CPU-only mode.')
"

echo ""
echo "=== Setup Complete ==="
echo "Run 'source .venv/bin/activate' to activate the environment"
echo "Run 'make test-cpu' to run CPU tests"
echo "Run 'tritonflow info' for system information"
