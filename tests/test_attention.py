"""Tests for attention kernels."""

import pytest
import math
from tests.conftest import gpu, make_tensor, assert_close, HAS_TORCH

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
        B, S, D = 2, 32, 64
        q = make_tensor(B, S, D)
        k = make_tensor(B, S, D)
        v = make_tensor(B, S, D)
        from tritonflow.kernels.attention import fused_attention

        result = fused_attention(q, k, v)
        expected = _reference_attention(q, k, v)
        assert result.shape == (B, S, D)
        assert_close(result, expected, atol=1e-2, rtol=1e-2)


@gpu
class TestFlashAttention:
    def test_basic(self):
        B, S, D = 2, 64, 32
        q = make_tensor(B, S, D)
        k = make_tensor(B, S, D)
        v = make_tensor(B, S, D)
        from tritonflow.kernels.attention import flash_attention_fwd

        result = flash_attention_fwd(q, k, v)
        expected = _reference_attention(q, k, v)
        assert result.shape == (B, S, D)
        assert_close(result, expected, atol=1e-2, rtol=1e-2)
