# Benchmarking Guide

## Methodology

TritonFlow benchmarks follow a rigorous methodology to produce reproducible, meaningful results.

### Timing Protocol

1. **Warmup phase**: Run the kernel N times (default: 10) to warm up GPU caches, trigger JIT compilation, and stabilize clock frequencies.

2. **Measurement phase**: Run the kernel M times (default: 100) with `torch.cuda.synchronize()` barriers before and after each invocation to ensure accurate GPU timing.

3. **Aggregation**: Report the **median** execution time to reduce the impact of outliers from OS scheduling, memory allocation, or thermal throttling.

### Baselines

Each benchmark compares against up to three baselines:

| Baseline | Implementation | Purpose |
|----------|---------------|---------|
| **Triton** | TritonFlow kernel | Primary measurement |
| **PyTorch** | `torch.*` / `torch.nn.functional.*` | GPU baseline (cuBLAS/cuDNN backed) |
| **NumPy** | `numpy.*` | CPU baseline for speedup context |

### Memory Tracking

Peak GPU memory is measured using `torch.cuda.max_memory_allocated()`, reset before each kernel invocation.

## Running Benchmarks

### Full Suite

```bash
make bench-report
```

### Specific Suite

```bash
tritonflow benchmark run --suite vector_ops
tritonflow benchmark run --suite ml_kernels
tritonflow benchmark run --suite similarity
```

### Custom Configuration

```bash
tritonflow benchmark run \
    --suite all \
    --iterations 200 \
    --warmup 20 \
    --export-md report.md \
    --export-csv results.csv
```

### Docker (Reproducible Environment)

```bash
docker compose -f docker/docker-compose.yml run tritonflow-bench
```

## Export Formats

| Format | Command | Use Case |
|--------|---------|----------|
| Rich table | Default (console) | Quick inspection |
| Markdown | `--export-md report.md` | Documentation, GitHub |
| CSV | `--export-csv results.csv` | Data analysis, plotting |
| Charts | `tritonflow visualize charts results.csv` | Visual reports |

## Reproducibility Checklist

- [ ] Fix random seeds (`torch.manual_seed(42)`)
- [ ] Report GPU model, driver version, CUDA version
- [ ] Use the same tensor sizes across comparisons
- [ ] Run with exclusive GPU access (no competing workloads)
- [ ] Report median, not mean
- [ ] Specify warmup and iteration count
