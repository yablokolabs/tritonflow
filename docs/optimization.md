# Triton Optimization Guide

## Block Size Selection

Block size is the most impactful tuning parameter. It controls how many elements each program instance processes.

| Block Size | Characteristics |
|-----------|----------------|
| 64–128 | Lower register pressure, higher occupancy, good for simple kernels |
| 256–512 | Balanced for most workloads |
| 1024+ | Better memory coalescing, higher register pressure |

Use `triton.autotune` to let the compiler test multiple configurations:

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    ...
```

## Memory Access Patterns

### Coalesced Access
Triton programs should access memory in contiguous chunks. Use `tl.arange(0, BLOCK_SIZE)` for sequential offsets.

### Avoiding Bank Conflicts
For shared memory (implicit in Triton), ensure stride patterns avoid 32-way bank conflicts. Tiled algorithms with power-of-2 tile sizes generally work well.

### Masking
Always mask loads and stores to prevent out-of-bounds access:

```python
mask = offsets < n_elements
x = tl.load(ptr + offsets, mask=mask, other=0.0)
```

## Kernel Fusion

Fusing multiple operations into a single kernel reduces memory bandwidth pressure:

```python
# Unfused: 3 global memory round-trips
y = x + bias          # read x, bias → write y
z = gelu(y)           # read y → write z
out = layer_norm(z)   # read z → write out

# Fused: 1 global memory round-trip
out = fused_bias_gelu_layernorm(x, bias, weight)
```

TritonFlow provides fused variants like `fused_add_mul`, `fused_gelu_bias`, and `fused_embedding_layernorm`.

## Numerical Stability

### Softmax
Always subtract the row maximum before exponentiation to prevent overflow:

```python
row_max = tl.max(x, axis=0)
safe = tl.exp(x - row_max)
output = safe / tl.sum(safe, axis=0)
```

### Reductions
For large reductions, accumulate in FP32 even when inputs are FP16/BF16 to maintain precision.

## GPU Architecture Considerations

| Feature | Ampere (A100) | Ada (RTX 4090) | Hopper (H100) |
|---------|--------------|----------------|---------------|
| FP32 TFLOPS | 19.5 | 82.6 | 60 |
| FP16 TFLOPS | 312 | 165 | 990 (w/ sparsity) |
| BF16 | ✅ | ✅ | ✅ |
| INT8 | ✅ | ✅ | ✅ |
| HBM bandwidth | 2 TB/s | 1 TB/s | 3.35 TB/s |

Kernel performance is typically **memory-bandwidth-bound** for element-wise operations and **compute-bound** for matmul/attention. Choose block sizes and fusion strategies accordingly.

## Profiling Workflow

1. **Identify bottleneck**: Use `TritonProfiler` to find the slowest kernel
2. **Profile with Nsight**: Wrap with `nsight_range()` and run under `nsys profile`
3. **Check occupancy**: Use `estimate_occupancy()` from autotuning utils
4. **Tune block sizes**: Try `triton.autotune` with various configurations
5. **Verify correctness**: Always compare against PyTorch reference after tuning
