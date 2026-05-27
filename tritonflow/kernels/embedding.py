import torch
import triton
import triton.language as tl

__all__ = ["embedding_lookup", "fused_embedding_layernorm"]


@triton.jit
def _embedding_lookup_kernel(
    weight_ptr,
    indices_ptr,
    output_ptr,
    embed_dim,
    stride_wv,
    stride_wd,
    stride_ob,
    stride_od,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = tl.load(indices_ptr + pid)

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < embed_dim

    w_ptrs = weight_ptr + idx * stride_wv + offs_d * stride_wd
    embedding = tl.load(w_ptrs, mask=d_mask, other=0.0)

    o_ptrs = output_ptr + pid * stride_ob + offs_d * stride_od
    tl.store(o_ptrs, embedding, mask=d_mask)


def embedding_lookup(weight: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Accelerated embedding table lookup."""
    assert weight.ndim == 2 and indices.ndim == 1
    batch_size = indices.shape[0]
    embed_dim = weight.shape[1]
    output = torch.empty(batch_size, embed_dim, device=weight.device, dtype=weight.dtype)
    BLOCK_D = triton.next_power_of_2(embed_dim)
    _embedding_lookup_kernel[(batch_size,)](
        weight,
        indices,
        output,
        embed_dim,
        weight.stride(0),
        weight.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_D=BLOCK_D,
    )
    return output


@triton.jit
def _fused_embedding_layernorm_kernel(
    weight_ptr,
    indices_ptr,
    ln_w_ptr,
    ln_b_ptr,
    output_ptr,
    embed_dim,
    eps,
    stride_wv,
    stride_wd,
    stride_ob,
    stride_od,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = tl.load(indices_ptr + pid)

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < embed_dim

    # Load embedding
    w_ptrs = weight_ptr + idx * stride_wv + offs_d * stride_wd
    x = tl.load(w_ptrs, mask=d_mask, other=0.0)

    # Layer norm: compute mean and variance
    mean = tl.sum(x, axis=0) / embed_dim
    x_centered = tl.where(d_mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / embed_dim
    inv_std = 1.0 / tl.sqrt(var + eps)

    x_norm = x_centered * inv_std

    # Apply affine
    ln_w = tl.load(ln_w_ptr + offs_d, mask=d_mask, other=1.0)
    ln_b = tl.load(ln_b_ptr + offs_d, mask=d_mask, other=0.0)
    out = x_norm * ln_w + ln_b

    o_ptrs = output_ptr + pid * stride_ob + offs_d * stride_od
    tl.store(o_ptrs, out, mask=d_mask)


def fused_embedding_layernorm(
    weight: torch.Tensor,
    indices: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Fused embedding lookup + layer normalization in a single kernel pass."""
    assert weight.ndim == 2 and indices.ndim == 1
    batch_size = indices.shape[0]
    embed_dim = weight.shape[1]
    output = torch.empty(batch_size, embed_dim, device=weight.device, dtype=weight.dtype)
    BLOCK_D = triton.next_power_of_2(embed_dim)
    _fused_embedding_layernorm_kernel[(batch_size,)](
        weight,
        indices,
        ln_weight,
        ln_bias,
        output,
        embed_dim,
        eps,
        weight.stride(0),
        weight.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_D=BLOCK_D,
    )
    return output
