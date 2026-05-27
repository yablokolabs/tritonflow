"""Tests for attention kernels."""

import math

from tests.conftest import HAS_TORCH, assert_close, gpu, make_tensor

if HAS_TORCH:
    import torch


def _reference_attention(q, k, v, scale=None):
    """Reference attention implementation."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


@gpu
class TestFusedAttention:
    def test_basic(self):
        batch, seq, dim = 2, 32, 64
        q = make_tensor(batch, seq, dim)
        k = make_tensor(batch, seq, dim)
        v = make_tensor(batch, seq, dim)
        from tritonflow.kernels.attention import fused_attention

        result = fused_attention(q, k, v)
        expected = _reference_attention(q, k, v)
        assert result.shape == (batch, seq, dim)
        assert_close(result, expected, atol=1e-2, rtol=1e-2)


@gpu
class TestFlashAttention:
    def test_basic(self):
        batch, seq, dim = 2, 64, 32
        q = make_tensor(batch, seq, dim)
        k = make_tensor(batch, seq, dim)
        v = make_tensor(batch, seq, dim)
        from tritonflow.kernels.attention import flash_attention_fwd

        result = flash_attention_fwd(q, k, v)
        expected = _reference_attention(q, k, v)
        assert result.shape == (batch, seq, dim)
        assert_close(result, expected, atol=1e-2, rtol=1e-2)
