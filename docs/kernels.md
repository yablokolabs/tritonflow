# Kernel Reference

## Vector Operations (`tritonflow.kernels.vector_ops`)

### `vector_add(x, y) → Tensor`
Element-wise addition of two 1D tensors. Each Triton program processes `BLOCK_SIZE` elements with masking for non-aligned sizes.

### `fused_add_mul(x, y, z) → Tensor`
Computes `(x + y) * z` in a single kernel pass, avoiding intermediate memory allocation.

### `batched_sum(x) → Tensor`
Row-wise sum reduction. Input `[M, N]` → output `[M]`. Each program handles one row using `tl.sum`.

### `batched_mean(x) → Tensor`
Row-wise mean. Divides `batched_sum` result by row length.

### `batched_max(x) → Tensor` / `batched_min(x) → Tensor`
Row-wise max/min reduction using `tl.max`/`tl.min`.

---

## Normalization (`tritonflow.kernels.normalization`)

### `layer_norm(x, weight, bias, eps=1e-5) → Tensor`
Layer normalization for 2D input `[M, N]`. Two-pass algorithm:
1. Compute row mean
2. Compute row variance, normalize, apply affine transform

Matches `torch.nn.functional.layer_norm` output.

### `rms_norm(x, weight, eps=1e-6) → Tensor`
RMS normalization (used in LLaMA, Gemma). Single-pass: `x * weight / sqrt(mean(x²) + eps)`.

---

## Activations (`tritonflow.kernels.activations`)

### `softmax(x) → Tensor`
Row-wise softmax for 2D input. Numerically stable implementation: subtract row max before exponentiation.

### `gelu(x) → Tensor`
GELU activation using the tanh approximation: `0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715x³)))`.

### `fused_gelu_bias(x, bias) → Tensor`
Fused bias addition + GELU in a single kernel pass.

---

## Matrix Operations (`tritonflow.kernels.matrix_ops`)

### `matrix_transpose(x) → Tensor`
Tiled 2D transpose. Uses `BLOCK × BLOCK` tiles for coalesced memory access.

### `matmul(a, b) → Tensor`
Blocked matrix multiplication. Each program computes a `BLOCK_M × BLOCK_N` output tile by iterating over the K dimension in `BLOCK_K` chunks using `tl.dot`.

---

## Attention (`tritonflow.kernels.attention`)

### `fused_attention(q, k, v, scale=None) → Tensor`
Fused multi-head attention. Inputs `[B, S, D]`. Computes `softmax(QK^T * scale) @ V` with online softmax for numerical stability.

### `flash_attention_fwd(q, k, v, scale=None) → Tensor`
Flash-attention-inspired forward pass. Processes Q in blocks, iterates over K/V blocks with running max and sum for memory-efficient online softmax. Avoids materializing the full `S×S` attention matrix.

---

## Embeddings (`tritonflow.kernels.embedding`)

### `embedding_lookup(weight, indices) → Tensor`
Accelerated embedding table lookup. `weight [V, D]`, `indices [B]` → output `[B, D]`.

### `fused_embedding_layernorm(weight, indices, ln_weight, ln_bias, eps=1e-5) → Tensor`
Single-pass embedding lookup + layer normalization.

---

## Similarity (`tritonflow.kernels.similarity`)

### `cosine_similarity(x, y) → Tensor`
Batched cosine similarity. `x, y [M, D]` → output `[M]`. Each program computes `dot(x_i, y_i) / (‖x_i‖ · ‖y_i‖)`.

### `l2_distance(x, y) → Tensor`
Batched L2 (Euclidean) distance. `x, y [M, D]` → output `[M]`.

### `top_k_similarity(queries, keys, k) → (indices, scores)`
Top-k cosine similarity retrieval. `queries [M, D]`, `keys [N, D]` → `indices [M, k]`, `scores [M, k]`.

---

## Quantization (`tritonflow.kernels.quantization`)

### `quantized_matmul_fp16(a, b) → Tensor`
Matrix multiplication in FP16 precision with FP32 output.

### `dynamic_quantize(x) → (quantized, scales)`
Per-row dynamic INT8 quantization. Returns int8 tensor and per-row scales.

### `quantized_matmul_int8(a, b, scale_a, scale_b) → Tensor`
INT8 matrix multiplication with int32 accumulation and scale dequantization.

---

## Scientific (`tritonflow.kernels.scientific`)

### `parallel_prefix_sum(x) → Tensor`
Inclusive prefix sum (scan) using Hillis-Steele algorithm within a single block.

### `stencil_1d(x, weights) → Tensor`
1D stencil convolution. Applies a small kernel to each element with halo loading.

### `batched_outer_product(x, y) → Tensor`
Batched outer product. `x [B, M]`, `y [B, N]` → output `[B, M, N]`.
