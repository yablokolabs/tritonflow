"""Tests for embedding kernels."""

from tests.conftest import HAS_TORCH, assert_close, gpu, make_tensor

if HAS_TORCH:
    import torch


@gpu
class TestEmbeddingLookup:
    def test_basic(self):
        vocab_size, embed_dim = 1000, 128
        weight = make_tensor(vocab_size, embed_dim)
        indices = torch.randint(0, vocab_size, (32,), device="cuda")
        from tritonflow.kernels.embedding import embedding_lookup

        result = embedding_lookup(weight, indices)
        expected = torch.nn.functional.embedding(indices, weight)
        assert_close(result, expected)

    def test_single_index(self):
        weight = make_tensor(100, 64)
        indices = torch.tensor([42], device="cuda")
        from tritonflow.kernels.embedding import embedding_lookup

        result = embedding_lookup(weight, indices)
        expected = weight[42].unsqueeze(0)
        assert_close(result, expected)


@gpu
class TestFusedEmbeddingLayerNorm:
    def test_basic(self):
        vocab_size, embed_dim = 500, 64
        weight = make_tensor(vocab_size, embed_dim)
        indices = torch.randint(0, vocab_size, (16,), device="cuda")
        ln_weight = torch.ones(embed_dim, device="cuda")
        ln_bias = torch.zeros(embed_dim, device="cuda")
        from tritonflow.kernels.embedding import fused_embedding_layernorm

        result = fused_embedding_layernorm(weight, indices, ln_weight, ln_bias)
        embedded = torch.nn.functional.embedding(indices, weight)
        expected = torch.nn.functional.layer_norm(embedded, [embed_dim], ln_weight, ln_bias)
        assert_close(result, expected, atol=1e-4, rtol=1e-4)
