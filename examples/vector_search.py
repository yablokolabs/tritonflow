"""Vector Search Acceleration with TritonFlow.

Demonstrates GPU-accelerated similarity search using TritonFlow kernels:
cosine similarity, L2 distance, and top-k retrieval.
"""

import torch
from tritonflow.utils.gpu import is_gpu_available, require_gpu


@require_gpu
def vector_search_demo():
    """Run similarity search operations using TritonFlow kernels."""
    from tritonflow.kernels.similarity import cosine_similarity, l2_distance, top_k_similarity
    from tritonflow.profiling.profiler import TritonProfiler

    profiler = TritonProfiler()

    num_queries = 32
    num_keys = 10_000
    dim = 256

    # Simulate a vector database
    queries = torch.randn(num_queries, dim, device="cuda")
    keys = torch.randn(num_keys, dim, device="cuda")

    # Pairwise cosine similarity (batch)
    batch_keys = keys[:num_queries]
    with profiler.profile_kernel("cosine_similarity"):
        cos_sim = cosine_similarity(queries, batch_keys)

    # L2 distance
    with profiler.profile_kernel("l2_distance"):
        l2_dist = l2_distance(queries, batch_keys)

    # Top-K retrieval
    with profiler.profile_kernel("top_k_similarity"):
        top_indices, top_scores = top_k_similarity(queries, keys, k=10)

    print(profiler.summary())
    print(f"\nResults:")
    print(f"  Cosine similarities shape: {cos_sim.shape}")
    print(f"  L2 distances shape:        {l2_dist.shape}")
    print(f"  Top-10 indices shape:      {top_indices.shape}")
    print(f"  Top-10 scores shape:       {top_scores.shape}")
    print(f"\n  Best match for query 0: key {top_indices[0, 0].item()} "
          f"(score: {top_scores[0, 0].item():.4f})")


if __name__ == "__main__":
    if not is_gpu_available():
        print("GPU required for this example. Skipping.")
    else:
        vector_search_demo()
