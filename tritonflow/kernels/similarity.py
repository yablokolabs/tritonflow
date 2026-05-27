import torch
import triton
import triton.language as tl

__all__ = ["cosine_similarity", "l2_distance", "top_k_similarity"]


@triton.jit
def _cosine_similarity_kernel(
    x_ptr, y_ptr, out_ptr,
    D,
    stride_xm, stride_xd,
    stride_ym, stride_yd,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)

    dot = tl.zeros((1,), dtype=tl.float32)
    norm_x = tl.zeros((1,), dtype=tl.float32)
    norm_y = tl.zeros((1,), dtype=tl.float32)

    for d_start in range(0, D, BLOCK_D):
        d_offs = d_start + offs_d
        d_mask = d_offs < D

        x = tl.load(x_ptr + pid * stride_xm + d_offs * stride_xd, mask=d_mask, other=0.0)
        y = tl.load(y_ptr + pid * stride_ym + d_offs * stride_yd, mask=d_mask, other=0.0)

        dot += tl.sum(x * y)
        norm_x += tl.sum(x * x)
        norm_y += tl.sum(y * y)

    result = dot / (tl.sqrt(norm_x) * tl.sqrt(norm_y) + 1e-8)
    tl.store(out_ptr + pid, result)


def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Batched cosine similarity between corresponding rows of x and y."""
    assert x.ndim == 2 and y.ndim == 2 and x.shape == y.shape
    M, D = x.shape
    out = torch.empty(M, device=x.device, dtype=torch.float32)
    BLOCK_D = min(triton.next_power_of_2(D), 1024)
    _cosine_similarity_kernel[(M,)](
        x, y, out, D,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_D=BLOCK_D,
    )
    return out


@triton.jit
def _l2_distance_kernel(
    x_ptr, y_ptr, out_ptr,
    D,
    stride_xm, stride_xd,
    stride_ym, stride_yd,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)

    acc = tl.zeros((1,), dtype=tl.float32)

    for d_start in range(0, D, BLOCK_D):
        d_offs = d_start + offs_d
        d_mask = d_offs < D

        x = tl.load(x_ptr + pid * stride_xm + d_offs * stride_xd, mask=d_mask, other=0.0)
        y = tl.load(y_ptr + pid * stride_ym + d_offs * stride_yd, mask=d_mask, other=0.0)

        diff = x - y
        acc += tl.sum(diff * diff)

    tl.store(out_ptr + pid, tl.sqrt(acc))


def l2_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Batched L2 (Euclidean) distance between corresponding rows of x and y."""
    assert x.ndim == 2 and y.ndim == 2 and x.shape == y.shape
    M, D = x.shape
    out = torch.empty(M, device=x.device, dtype=torch.float32)
    BLOCK_D = min(triton.next_power_of_2(D), 1024)
    _l2_distance_kernel[(M,)](
        x, y, out, D,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_D=BLOCK_D,
    )
    return out


@triton.jit
def _batched_cosine_sim_kernel(
    queries_ptr, keys_ptr, out_ptr,
    N, D,
    stride_qm, stride_qd,
    stride_kn, stride_kd,
    stride_om, stride_on,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_D)

    dot = tl.zeros((1,), dtype=tl.float32)
    norm_q = tl.zeros((1,), dtype=tl.float32)
    norm_k = tl.zeros((1,), dtype=tl.float32)

    for d_start in range(0, D, BLOCK_D):
        d_offs = d_start + offs_d
        d_mask = d_offs < D

        q = tl.load(queries_ptr + pid_m * stride_qm + d_offs * stride_qd, mask=d_mask, other=0.0)
        k = tl.load(keys_ptr + pid_n * stride_kn + d_offs * stride_kd, mask=d_mask, other=0.0)

        dot += tl.sum(q * k)
        norm_q += tl.sum(q * q)
        norm_k += tl.sum(k * k)

    sim = dot / (tl.sqrt(norm_q) * tl.sqrt(norm_k) + 1e-8)
    tl.store(out_ptr + pid_m * stride_om + pid_n * stride_on, sim)


def top_k_similarity(
    queries: torch.Tensor, keys: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """For each query, find top-k most similar keys by cosine similarity.

    Returns (indices [M, k], scores [M, k]).
    """
    assert queries.ndim == 2 and keys.ndim == 2 and queries.shape[1] == keys.shape[1]
    M, D = queries.shape
    N = keys.shape[0]

    # Compute full similarity matrix with Triton kernel
    sim_matrix = torch.empty(M, N, device=queries.device, dtype=torch.float32)
    BLOCK_D = min(triton.next_power_of_2(D), 1024)
    grid = (M, N)
    _batched_cosine_sim_kernel[grid](
        queries, keys, sim_matrix,
        N, D,
        queries.stride(0), queries.stride(1),
        keys.stride(0), keys.stride(1),
        sim_matrix.stride(0), sim_matrix.stride(1),
        BLOCK_D=BLOCK_D,
    )

    scores, indices = torch.topk(sim_matrix, k=k, dim=1)
    return indices, scores
