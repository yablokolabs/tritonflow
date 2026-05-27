"""Accelerating Transformer Operations with TritonFlow.

Demonstrates how to use TritonFlow kernels for common transformer building blocks:
layer normalization, GELU activation, softmax attention, and fused operations.
"""

import torch
from tritonflow.utils.gpu import is_gpu_available, require_gpu


@require_gpu
def transformer_block_demo():
    """Run core transformer operations using TritonFlow kernels."""
    from tritonflow.kernels.normalization import layer_norm, rms_norm
    from tritonflow.kernels.activations import softmax, gelu, fused_gelu_bias
    from tritonflow.kernels.attention import fused_attention
    from tritonflow.profiling.profiler import TritonProfiler

    batch_size = 8
    seq_len = 512
    hidden_dim = 768
    profiler = TritonProfiler()

    x = torch.randn(batch_size * seq_len, hidden_dim, device="cuda")
    ln_weight = torch.ones(hidden_dim, device="cuda")
    ln_bias = torch.zeros(hidden_dim, device="cuda")

    # Layer Normalization
    with profiler.profile_kernel("layer_norm"):
        normed = layer_norm(x, ln_weight, ln_bias)

    # RMS Normalization (used in LLaMA-style models)
    rms_weight = torch.ones(hidden_dim, device="cuda")
    with profiler.profile_kernel("rms_norm"):
        rms_normed = rms_norm(x, rms_weight)

    # GELU activation (feed-forward network)
    bias = torch.randn(hidden_dim, device="cuda")
    with profiler.profile_kernel("fused_gelu_bias"):
        activated = fused_gelu_bias(normed, bias)

    # Fused attention
    q = torch.randn(batch_size, seq_len, 64, device="cuda")
    k = torch.randn(batch_size, seq_len, 64, device="cuda")
    v = torch.randn(batch_size, seq_len, 64, device="cuda")
    with profiler.profile_kernel("fused_attention"):
        attn_out = fused_attention(q, k, v)

    print(profiler.summary())
    print(f"\nOutput shapes:")
    print(f"  LayerNorm: {normed.shape}")
    print(f"  RMSNorm:   {rms_normed.shape}")
    print(f"  GELU:      {activated.shape}")
    print(f"  Attention: {attn_out.shape}")


if __name__ == "__main__":
    if not is_gpu_available():
        print("GPU required for this example. Skipping.")
    else:
        transformer_block_demo()
