# Architecture Guide

## System Design

TritonFlow is organized as a Python package with clear separation between kernel implementations, infrastructure (benchmarks, profiling, CLI), and developer tooling.

### Layer Model

```
┌─────────────────────────────────────────────┐
│                  CLI (typer)                 │  User interface
├─────────────────────────────────────────────┤
│         Benchmarks │ Profiling               │  Measurement
├─────────────────────────────────────────────┤
│              Kernel Wrappers                 │  Python API
├─────────────────────────────────────────────┤
│           @triton.jit Kernels                │  GPU computation
├─────────────────────────────────────────────┤
│        Utils (GPU, dtypes, validators)       │  Shared infrastructure
└─────────────────────────────────────────────┘
```

### Kernel Module Structure

Each kernel module (`tritonflow/kernels/*.py`) follows a consistent pattern:

1. **Private kernel function** (`_kernel_name_kernel`): Decorated with `@triton.jit`, contains the Triton DSL logic that runs on the GPU. Uses `tl.constexpr` for compile-time block size parameters.

2. **Public wrapper function** (`kernel_name`): Takes and returns `torch.Tensor` objects. Handles output allocation, grid computation, and kernel launch.

3. **Module exports** (`__all__`): Lists only the public wrapper functions.

### Grid and Block Design

Triton programs are organized in a 1D, 2D, or 3D grid of program instances. Each program processes a tile of the input data.

- **1D kernels** (vector ops, activations): One program per `BLOCK_SIZE` elements
- **2D kernels** (matmul, attention): One program per `(BLOCK_M, BLOCK_N)` output tile
- **Row-wise kernels** (softmax, normalization): One program per row

### Memory Access Patterns

- **Coalesced loads**: `tl.arange(0, BLOCK_SIZE)` produces contiguous offsets for coalesced global memory access
- **Masking**: `mask = offsets < n_elements` prevents out-of-bounds memory access
- **Shared memory**: Implicitly managed by Triton for tiled algorithms (matmul, transpose)

## Profiling Architecture

The profiling system has three components:

- **TritonProfiler**: Measures wall-clock time and GPU memory per kernel invocation
- **Nsight hooks**: NVTX range markers for integration with NVIDIA Nsight Systems
- **ExecutionTimeline**: Records events for export to Chrome trace format

## Benchmark Architecture

The benchmark system separates concerns:

- **BenchmarkRunner**: Handles timing with warmup, CUDA synchronization, and statistical aggregation
- **BenchmarkReporter**: Converts results to markdown, CSV, rich tables, and matplotlib charts
- **Suites**: Predefined sets of benchmarks organized by kernel category
- **Configs**: Centralized constants for tensor sizes, dtypes, and iteration counts
