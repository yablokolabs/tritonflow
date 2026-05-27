<div align="center">

# ⚡ TritonFlow

**High-performance GPU kernels for modern AI, vector search, analytics, and scientific computing.**

Built with [OpenAI Triton](https://openai.com/index/triton/) · By [Yabloko Labs](https://github.com/yablokolabs)

[![CI](https://github.com/yablokolabs/tritonflow/actions/workflows/ci.yml/badge.svg)](https://github.com/yablokolabs/tritonflow/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

---

## Overview

TritonFlow is a modular GPU acceleration library that provides optimized Triton kernels for AI/ML workloads, similarity search, and scientific computing. It includes a benchmark framework, CLI tooling, and profiling infrastructure designed for real-world deployment.

### Why Triton?

[OpenAI Triton](https://openai.com/index/triton/) compiles Python-level kernel definitions into optimized GPU machine code. Compared to raw CUDA:

- **~90% less boilerplate** — no manual memory management, grid/block configuration is declarative
- **Portable across GPU architectures** — Triton's compiler handles SM-specific tuning
- **Competitive performance** — within 5–15% of hand-written CUDA for most workloads
- **Python-native** — integrates directly with PyTorch tensors, no C++ build step

TritonFlow uses Triton to deliver production-grade kernels that are readable, maintainable, and fast.

---

## Kernel Library

| Category | Kernels | Description |
|----------|---------|-------------|
| **Vector Ops** | `vector_add`, `fused_add_mul`, `batched_sum`, `batched_mean`, `batched_max`, `batched_min` | Element-wise and reduction operations with fused variants |
| **Normalization** | `layer_norm`, `rms_norm` | Layer normalization and RMS normalization (LLaMA-style) |
| **Activations** | `softmax`, `gelu`, `fused_gelu_bias` | Numerically stable softmax, GELU with fused bias |
| **Matrix Ops** | `matrix_transpose`, `matmul` | Tiled transpose and blocked matrix multiplication |
| **Attention** | `fused_attention`, `flash_attention_fwd` | Fused multi-head attention and flash-attention-inspired forward pass |
| **Embeddings** | `embedding_lookup`, `fused_embedding_layernorm` | Accelerated lookup with optional fused normalization |
| **Similarity** | `cosine_similarity`, `l2_distance`, `top_k_similarity` | Batched distance kernels and top-k retrieval |
| **Quantization** | `quantized_matmul_fp16`, `dynamic_quantize`, `quantized_matmul_int8` | FP16/INT8 quantized operations |
| **Scientific** | `parallel_prefix_sum`, `stencil_1d`, `batched_outer_product` | Scan, stencil, and batched tensor operations |

---

## Architecture

```
tritonflow/
├── tritonflow/                 # Core library
│   ├── kernels/                # Triton kernel implementations
│   │   ├── vector_ops.py       # Vector operations
│   │   ├── normalization.py    # LayerNorm, RMSNorm
│   │   ├── activations.py      # Softmax, GELU
│   │   ├── matrix_ops.py       # Transpose, matmul
│   │   ├── attention.py        # Fused & flash attention
│   │   ├── embedding.py        # Embedding lookup
│   │   ├── similarity.py       # Cosine sim, L2, top-k
│   │   ├── quantization.py     # FP16/INT8 quantized ops
│   │   └── scientific.py       # Prefix sum, stencil
│   ├── cli/                    # CLI application (typer)
│   ├── profiling/              # Profiler, Nsight hooks, timeline
│   └── utils/                  # GPU detection, dtypes, autotuning
├── benchmarks/                 # Benchmark framework & suites
├── tests/                      # Test suite (pytest)
├── examples/                   # Usage examples
├── docs/                       # Documentation
├── docker/                     # Containerized environments
├── scripts/                    # Dev & benchmark scripts
└── .github/workflows/          # CI/CD pipelines
```

### Kernel Design Pattern

Every kernel follows the same structure:

```python
@triton.jit
def _kernel(x_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    result = x * 2  # computation
    tl.store(output_ptr + offsets, result, mask=mask)

def kernel(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    _kernel[grid](x, output, n, BLOCK_SIZE=1024)
    return output
```

- **Private kernel** (`_kernel`): `@triton.jit` decorated, handles the GPU computation
- **Public wrapper** (`kernel`): Allocates output tensors, computes the grid, launches the kernel
- **Masking**: Every kernel masks out-of-bounds threads for non-power-of-2 inputs
- **Block sizes**: Configurable via `tl.constexpr`, defaults tuned for common GPU architectures

---

## Quick Start

### Installation

```bash
# From source
git clone https://github.com/yablokolabs/tritonflow.git
cd tritonflow
pip install -e ".[all]"

# Or minimal install
pip install -e .
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- OpenAI Triton 2.1+
- NVIDIA GPU with CUDA support

### Usage Examples

#### Vector Operations

```python
import torch
from tritonflow.kernels.vector_ops import vector_add, fused_add_mul, batched_sum, batched_mean

x = torch.randn(1_000_000, device="cuda")
y = torch.randn(1_000_000, device="cuda")
z = torch.randn(1_000_000, device="cuda")

# Element-wise addition
result = vector_add(x, y)

# Fused (x + y) * z in a single GPU pass — no intermediate allocation
fused = fused_add_mul(x, y, z)

# Row-wise reductions on 2D tensors
matrix = torch.randn(256, 1024, device="cuda")
row_sums = batched_sum(matrix)     # [256]
row_means = batched_mean(matrix)   # [256]
```

#### Normalization

```python
from tritonflow.kernels.normalization import layer_norm, rms_norm

hidden = torch.randn(32, 768, device="cuda")
weight = torch.ones(768, device="cuda")
bias = torch.zeros(768, device="cuda")

# Standard layer normalization (transformer-style)
normed = layer_norm(hidden, weight, bias)

# RMS normalization (LLaMA / Gemma style)
rms_normed = rms_norm(hidden, weight)
```

#### Activations

```python
from tritonflow.kernels.activations import softmax, gelu, fused_gelu_bias

logits = torch.randn(64, 50257, device="cuda")
probs = softmax(logits)  # Numerically stable row-wise softmax

x = torch.randn(32, 3072, device="cuda")
activated = gelu(x)

# Fused bias + GELU in one kernel (saves a full memory round-trip)
bias = torch.randn(3072, device="cuda")
out = fused_gelu_bias(x, bias)
```

#### Matrix Operations

```python
from tritonflow.kernels.matrix_ops import matmul, matrix_transpose

a = torch.randn(1024, 512, device="cuda")
b = torch.randn(512, 768, device="cuda")

product = matmul(a, b)          # [1024, 768]
transposed = matrix_transpose(a)  # [512, 1024]
```

#### Fused & Flash Attention

```python
from tritonflow.kernels.attention import fused_attention, flash_attention_fwd

batch, seq_len, head_dim = 8, 512, 64
q = torch.randn(batch, seq_len, head_dim, device="cuda")
k = torch.randn(batch, seq_len, head_dim, device="cuda")
v = torch.randn(batch, seq_len, head_dim, device="cuda")

# Standard fused attention: softmax(QK^T / √d) @ V
attn_out = fused_attention(q, k, v)

# Flash-attention-inspired: memory-efficient, avoids materializing S×S matrix
flash_out = flash_attention_fwd(q, k, v)
```

#### Embedding Lookup

```python
from tritonflow.kernels.embedding import embedding_lookup, fused_embedding_layernorm

vocab_size, embed_dim = 32000, 4096
weight = torch.randn(vocab_size, embed_dim, device="cuda")
token_ids = torch.randint(0, vocab_size, (64,), device="cuda")

embeddings = embedding_lookup(weight, token_ids)  # [64, 4096]

# Fused embedding + layer norm in a single kernel pass
ln_weight = torch.ones(embed_dim, device="cuda")
ln_bias = torch.zeros(embed_dim, device="cuda")
normed_embeddings = fused_embedding_layernorm(weight, token_ids, ln_weight, ln_bias)
```

#### Similarity Search

```python
from tritonflow.kernels.similarity import cosine_similarity, l2_distance, top_k_similarity

queries = torch.randn(32, 256, device="cuda")
database = torch.randn(100_000, 256, device="cuda")

# Pairwise cosine similarity (batched)
batch_keys = database[:32]
cos_scores = cosine_similarity(queries, batch_keys)  # [32]

# L2 distance
distances = l2_distance(queries, batch_keys)  # [32]

# Top-K retrieval: find 10 most similar vectors for each query
indices, scores = top_k_similarity(queries, database, k=10)
# indices: [32, 10], scores: [32, 10] (sorted descending)
```

#### Quantization

```python
from tritonflow.kernels.quantization import quantized_matmul_fp16, dynamic_quantize

a = torch.randn(512, 1024, device="cuda")
b = torch.randn(1024, 768, device="cuda")

# FP16 matrix multiply (2× memory savings)
result_fp16 = quantized_matmul_fp16(a, b)

# Dynamic INT8 quantization
quantized, scales = dynamic_quantize(a)
# quantized: int8 tensor, scales: per-row FP32 scales
```

#### Scientific Computing

```python
from tritonflow.kernels.scientific import parallel_prefix_sum, stencil_1d, batched_outer_product

# Inclusive prefix sum (scan)
x = torch.arange(1, 1025, dtype=torch.float32, device="cuda")
cumsum = parallel_prefix_sum(x)  # [1, 3, 6, 10, ...]

# 1D stencil / convolution
signal = torch.randn(8192, device="cuda")
kernel = torch.tensor([0.25, 0.5, 0.25], device="cuda")
smoothed = stencil_1d(signal, kernel)

# Batched outer product
a = torch.randn(16, 64, device="cuda")
b = torch.randn(16, 128, device="cuda")
outer = batched_outer_product(a, b)  # [16, 64, 128]
```

#### Profiling

```python
from tritonflow.profiling.profiler import TritonProfiler
from tritonflow.profiling.nsight import nsight_range
from tritonflow.profiling.timeline import ExecutionTimeline
from tritonflow.kernels.activations import softmax

profiler = TritonProfiler()
x = torch.randn(4096, 4096, device="cuda")

# Time kernel execution with memory tracking
with profiler.profile_kernel("softmax"):
    out = softmax(x)

print(profiler.summary())
# ┌────────┬────────────┬──────────────┐
# │ Kernel │ Time (ms)  │ Memory (MB)  │
# ├────────┼────────────┼──────────────┤
# │ softmax│ 0.42       │ 67.1         │
# └────────┴────────────┴──────────────┘

# Nsight Systems integration (for `nsys profile python script.py`)
with nsight_range("forward_pass"):
    out = softmax(x)

# Chrome trace export
timeline = ExecutionTimeline()
timeline.start()
with timeline.record("softmax"):
    out = softmax(x)
trace = timeline.to_chrome_trace()  # Open in chrome://tracing
```

#### GPU Detection

```python
from tritonflow.utils.gpu import is_gpu_available, get_device_info, require_gpu

if is_gpu_available():
    info = get_device_info()
    print(info)
    # {'name': 'NVIDIA A100', 'compute_capability': (8, 0),
    #  'total_memory_gb': 80.0, 'sm_count': 108, ...}

# Decorator: raises RuntimeError if no GPU
@require_gpu
def train_step(model, batch):
    ...
```

### CLI

```bash
# System information
tritonflow info

# Run benchmarks
tritonflow benchmark run --suite all
tritonflow benchmark run --suite vector_ops --iterations 200 --export-csv results.csv

# Profile a specific kernel
tritonflow profile kernel softmax --size 1048576 --repeats 50

# Compare Triton vs PyTorch vs NumPy
tritonflow compare ops vector_add --sizes 1024,65536,1048576

# Generate charts from benchmark CSV
tritonflow visualize charts results.csv --output-dir charts/
```

---

## Benchmarking

TritonFlow includes a benchmark framework that measures throughput, latency, and memory utilization against PyTorch and NumPy baselines.

```bash
# Full benchmark suite
make bench

# With report export
make bench-report

# Specific suite
tritonflow benchmark run --suite vector_ops --export-csv results.csv
```

### Benchmark Methodology

- **Warmup**: Configurable warmup iterations (default: 10) to stabilize GPU clocks
- **Timing**: Median of N iterations (default: 100) with `torch.cuda.synchronize()` barriers
- **Memory**: Peak allocated/reserved memory tracked per kernel
- **Baselines**: PyTorch native ops and NumPy CPU implementations
- **Reproducibility**: Fixed random seeds, configurable tensor sizes and dtypes

### Sample Results

Results will vary by GPU. Run `make bench-report` to generate results for your hardware.

| Kernel | Input Size | Triton (ms) | PyTorch (ms) | Speedup |
|--------|-----------|-------------|-------------|---------|
| `vector_add` | 16M | — | — | — |
| `softmax` | 4096×4096 | — | — | — |
| `layer_norm` | 4096×1024 | — | — | — |
| `matmul` | 1024×1024 | — | — | — |
| `fused_attention` | B=8, S=512 | — | — | — |

*Run on your hardware to fill in actual numbers.*

---

## Profiling

```python
from tritonflow.profiling.profiler import TritonProfiler
from tritonflow.profiling.nsight import nsight_range
from tritonflow.profiling.timeline import ExecutionTimeline

# Kernel profiling
profiler = TritonProfiler()
with profiler.profile_kernel("my_kernel"):
    result = my_kernel(x)
print(profiler.summary())

# Nsight integration
with nsight_range("attention_block"):
    output = fused_attention(q, k, v)

# Timeline capture (Chrome trace format)
timeline = ExecutionTimeline()
timeline.start()
with timeline.record("kernel_1"):
    result = softmax(x)
trace = timeline.to_chrome_trace()  # Load in chrome://tracing
```

---

## Development

```bash
# Setup
make install-dev
make pre-commit

# Testing
make test-cpu          # CPU-only tests (no GPU required)
make test-gpu          # GPU tests
make test              # All tests

# Code quality
make lint              # Ruff + mypy
make format            # Ruff fix + Black

# Docker
make docker-build
make docker-run        # Requires nvidia-docker
```

### Project Standards

- **Linting**: Ruff with isort, Black for formatting
- **Type checking**: mypy with strict mode
- **Testing**: pytest with GPU skip markers
- **CI**: GitHub Actions for lint + CPU tests on every push
- **GPU CI**: Self-hosted runner for GPU tests (weekly + manual)

---

## Examples

| Example | Description |
|---------|-------------|
| [`transformer_acceleration.py`](examples/transformer_acceleration.py) | Accelerate transformer ops: LayerNorm, GELU, attention |
| [`vector_search.py`](examples/vector_search.py) | GPU-accelerated similarity search and top-k retrieval |
| [`scientific_computing.py`](examples/scientific_computing.py) | Prefix sums, stencils, batched outer products |
| [`custom_kernel.py`](examples/custom_kernel.py) | Write your own Triton kernel with TritonFlow patterns |

---

## Documentation

- [Architecture Guide](docs/architecture.md) — system design and kernel patterns
- [Kernel Reference](docs/kernels.md) — detailed kernel documentation
- [Benchmarking Guide](docs/benchmarking.md) — methodology and reproducibility
- [Optimization Guide](docs/optimization.md) — Triton performance techniques
- [Contributing](docs/contributing.md) — development setup and code standards

---

## License

Apache License 2.0. See [LICENSE](LICENSE).

Built by [Yabloko Labs](https://github.com/yablokolabs).
