"""Tests for similarity kernels."""

import pytest
from tests.conftest import gpu, make_tensor, assert_close, HAS_TORCH

if HAS_TORCH:
    import torch


@gpu
class TestCosineSimilarity:
    def test_basic(self):
        x = make_tensor(32, 128)
        y = make_tensor(32, 128)
        from tritonflow.kernels.similarity import cosine_similarity

        result = cosine_similarity(x, y)
        expected = torch.nn.functional.cosine_similarity(x, y, dim=1)
        assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_identical_vectors(self):
        x = make_tensor(16, 64)
        from tritonflow.kernels.similarity import cosine_similarity

        result = cosine_similarity(x, x)
        assert_close(result, torch.ones(16, device="cuda"), atol=1e-4, rtol=1e-4)


@gpu
class TestL2Distance:
    def test_basic(self):
        x = make_tensor(32, 128)
        y = make_tensor(32, 128)
        from tritonflow.kernels.similarity import l2_distance

        result = l2_distance(x, y)
        expected = torch.norm(x - y, dim=1)
        assert_close(result, expected, atol=1e-3, rtol=1e-3)

    def test_same_vectors(self):
        x = make_tensor(16, 64)
        from tritonflow.kernels.similarity import l2_distance

        result = l2_distance(x, x)
        assert_close(result, torch.zeros(16, device="cuda"), atol=1e-5, rtol=1e-5)


@gpu
class TestTopKSimilarity:
    def test_basic(self):
        queries = make_tensor(4, 64)
        keys = make_tensor(100, 64)
        from tritonflow.kernels.similarity import top_k_similarity

        indices, scores = top_k_similarity(queries, keys, k=5)
        assert indices.shape == (4, 5)
        assert scores.shape == (4, 5)
        # Scores should be sorted descending
        for i in range(4):
            assert torch.all(scores[i, :-1] >= scores[i, 1:])
