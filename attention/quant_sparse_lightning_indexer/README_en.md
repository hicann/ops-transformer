# QuantSparseLightningIndexer

The **second stage** operator (consumer, candidate-consuming) of the two-level TopK scheme (coarse block-level screening followed by fine-grained selection inside the candidate blocks). It must be used together with the first-stage operator [QuantLightningIndexerV2](../quant_lightning_indexer_v2/README.md): the first stage selects candidate blocks at block granularity from the whole sequence and outputs the block-level index `candidate_block_indices`, and this operator performs token-level TopK within the set of candidate blocks.

## Product Support

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function Description

- Operator role: this is the second stage of the two-level TopK. The first-stage operator `QuantLightningIndexerV2` enables source with `candidate_topk_blocks=2048`, splits each token's row (the key sequence) into blocks of `candidate_block_size` positions, scores the blocks and selects `candidate_topk_blocks` candidate blocks, then outputs the block-level index `candidate_block_indices`. This operator receives that index, performs token-level TopK only on the positions covered by candidate blocks, and outputs the final `sparse_indices` for sparse attention.
- API function: sparse-attention preprocessing in inference scenarios; selects the key sparse tokens within candidate blocks and quantizes the input query and key (INT8 8-bit storage and computation / INT4 4-bit storage and computation) for maximum benefit.
- Formula: identical to QuantLightningIndexerV2, except that the TopK domain is restricted to candidate blocks:

    $$out = \text{Top-}k\left\{[1]_{1\times g}@\left[(W@[1]_{1\times S_{k}})\odot\text{ReLU}\left(\left(Scale_Q@Scale_K^T\right)\odot\left(Q_{index}^{Quant}@{\left(K_{index}^{Quant}\right)}^T\right)\right)\right]\right\},\quad p \in \text{candidate block}$$

    Block-domain definition: position $p$ belongs to block $\lfloor p / \text{candidate\_block\_size} \rfloor$; only positions whose block index appears in `candidate_block_indices` and that satisfy valid sequence-length and mask constraints participate in TopK, and all other positions are set to $-\infty$; slots left when the number of valid positions is less than Top-k output `-1`.
    Main computation steps:
    1. Multiply the input `q` of a token ($Q_{index}^{Quant}\in\R^{g\times d}$) by the given context `k` ($K_{index}^{Quant}\in\R^{S_{k}\times d}$) to obtain the relevance.
    2. Multiply the relevance by the dequantization scales `q_descale` ($Scale_Q$) and `k_descale` ($Scale_K^T$) of `q` and `k`, and filter out invalid negative correlations with $ReLU$ to obtain the relevance score vector between the current token and preceding tokens.
    3. Multiply the score vector by the weights `w` ($W$) and sum along g. Set positions outside the candidates or excluded by sequence-length and mask constraints to $-\infty$, then select the top $Top-k$ indices along Key positions to obtain the output $out$.

- Typical call chain: QuantLightningIndexerV2 produces candidates and valid counts → QuantSparseLightningIndexerMetadata produces second-stage metadata → this operator selects token indices within candidates → sparse attention.

## Parameter Description

| Parameter | Input/Output/Attribute | Description | Data Type | Format |
|-----------|-------------------|-------------|-----------|--------|
| q | Input | $Q_{index}^{Quant}\in\R^{g\times d}$ in the formula, representing input Index Query; non-contiguous is not supported. | INT8, FLOAT4_e2m1, INT4 | ND |
| k | Input | $K_{index}^{Quant}\in\R^{S_{k}\times d}$ in the formula, representing the compressed input Index Key; non-contiguity on axis 0 is supported. See Constraints for layout restrictions. | INT8, FLOAT4_e2m1, INT4 | ND |
| w | Input | $W$ in the formula, representing the weight coefficient; non-contiguous is not supported. | FLOAT16, FLOAT32 | ND |
| q_descale | Input | $Scale_Q$ in the formula, representing the dequantization coefficient of Index Query; non-contiguous is not supported. When `quant_mode` is 5, the shape is obtained by replacing the D axis of `q` with (D/64, 2); in other scenarios, the shape is consistent with w | FLOAT16, FLOAT8_e8m0 | ND |
| k_descale | Input | $Scale_K$ in the formula, representing the dequantization coefficient of Index Key; non-contiguity on axis 0 is supported. When `quant_mode` is 5, the shape is obtained by replacing the D axis of `k` with (D/64, 2); in other scenarios, the shape is obtained by removing the D axis of `k`. See Constraints for paged storage requirements. | FLOAT16, FLOAT8_e8m0 | ND |
| cu_seqlens_q | Optional input | Must be passed in when layout_q is TND, representing the prefix sum of the number of valid tokens of `q` in each batch; it cannot be passed in when layout_q is BSND. The default value is None. | INT32 | ND |
| cu_seqlens_k | Optional input | Must be passed in when layout_k is TND, representing the prefix sum of the number of valid tokens of `k` in each batch; it cannot be passed in when layout_k is PA_BBND or BSND. The default value is None. | INT32 | ND |
| seqused_q | Optional input | Optionally passed in when layout_q is BSND, representing the number of valid tokens of `q` in each batch. The default value is None. | INT32 | ND |
| seqused_k | Optional input | Used when layout_k is PA_BBND or BSND, representing the number of valid tokens of `k` in each batch. The default value is None. | INT32 | ND |
| cmp_residual_k | Optional input | Residual length of Key in compression scenarios, which must satisfy 0 \<= cmp_residual_k\[i\] \< cmp_ratio. The default value is None. | INT32 | ND |
| block_table | Optional input | Represents the block mapping table used for KV storage in PageAttention. The default value is None. | INT32 | ND |
| output_idx_offset | Optional input | Offset of the output indices, added only to sparse_indices. | INT32 | ND |
| candidate_block_length | Optional input | Actual number of valid candidate blocks per Query row; see Constraints for passing requirements. | INT32 | ND |
| metadata | Optional input | Core-partitioning information produced by the QuantSparseLightningIndexerMetadata operator, including the number of cores used, block size, and the start point of the data handled by each core. The shape is (1024,). The default value is None. | INT32 | ND |
| candidate_block_indices | Input | Candidate block indices produced by QuantLightningIndexerV2; see Constraints for validity requirements. | INT32 | ND |
| topk | Attribute | Represents the number of indices to retain in the topK stage, with a default value of 2048. | INT32 | - |
| quant_mode | Attribute | Identifies the quantization mode of inputs. | INT32 | - |
| max_seqlen_q | Optional attribute | Maximum sequence length of Query; the default value -1 means any possible length | INT32 | - |
| layout_q | Optional attribute | Identifies the data layout format of input `q`, with a default value of "BSND". | STRING | - |
| layout_k | Optional attribute | Identifies the data layout format of input `k`, with a default value of "PA_BBND". | STRING | - |
| mask_mode | Optional attribute | Represents the mask mode, with a default value of 0. | INT32 | - |
| cmp_ratio | Optional attribute | Used for sparse computation; represents the compression ratio of k, with a default value of 1. | INT32 | - |
| return_value | Optional attribute | Indicates whether to output `sparse_values`, with a default value of 0. | INT32 | - |
| candidate_block_size | Optional attribute | Candidate block size (number of positions per block); default 8; currently only 8 is supported. | INT32 | - |
| sparse_indices | Output | Token indices used by sparse attention. Shape is (B,S1,N2,topk) for BSND and (T,N2,topk) for TND. | INT32 | ND |
| sparse_values | Output | Value corresponding to the Indices output in the formula. When `return_value` is 1, the shape is consistent with `sparse_indices`; when `return_value` is 0, the shape is (0,) | BFLOAT16 | ND |

## Constraints

- This operator is used for inference.
- candidate_block_indices is required and has a fixed capacity of 2048 candidates per Query row.
- candidate_block_size must be 8 and must match the first-stage operator.
- metadata is required and must not be empty.
- output_idx_offset must have one fewer dimension than sparse_indices.

<!-- npu="950" id8 -->
- <term>Ascend 950PR&950DT products</term>:
  - `quant_mode` supports only 5 (MXFP4). `q` and non-paged `k` support `float4_e2m1fn_x2` or packed `uint8` storage, with a physical last dimension of 64.
  - `w` supports only `float32`. `q_descale` and non-paged `k_descale` support `float8_e8m0fnu` or equivalent `uint8` storage. Their shapes are obtained by removing the last dimension of the corresponding `q` or `k` shape and appending `(2,2)`.
  - The N dimension of `q` is in [1,32]; the N dimension of `k` must be 1. The logical head dimension must be 128.
  - `topk` is in [1,8192].
  - `layout_q` supports TND and BSND. `layout_k` supports TND, BSND and PA_BBND. For non-paged keys, `layout_q` and `layout_k` must be the same.
  - When `layout_k` is PA_BBND, `k` uses `uint8` storage with shape `(block_num,block_size/8,544)`. `block_size` must be a multiple of 16 in [16,1024]. Each group of eight tokens stores 512 bytes of key data followed by 32 bytes of scales. Pass an empty `uint8` tensor for `k_descale`.
  - When `layout_q` is TND, `cu_seqlens_q` is required. When `layout_k` is TND, `cu_seqlens_k` is required and `block_table` must not be passed.
  - BSND layouts do not accept `cu_seqlens_q` or `cu_seqlens_k`. When `layout_k` is PA_BBND, `cu_seqlens_k` must not be passed.
  - When `layout_k` is PA_BBND, `seqused_k` and a non-empty `block_table` must be passed.
  - `q`, `w`, `q_descale`, `candidate_block_indices`, and `candidate_block_length` must be contiguous. When `layout_k` is TND or BSND, `k` and `k_descale` must be contiguous. When `layout_k` is PA_BBND, non-contiguity is supported only on axis 0 of `k`, and the axis-0 stride must be at least the size of one page.
  - `candidate_block_size` supports only 8. The shape of `candidate_block_indices` is `(T,1,2048)` or `(B,S1,1,2048)`.
  - `candidate_block_length` is required, with shape `(T,1)` or `(B,S1,1)`. Each element is the actual valid candidate count for the corresponding row, in [0,2048]. The first `candidate_block_length` entries in that row must be valid, unique block indices. Do not substitute the candidate capacity for the actual valid count.
  - `return_value` supports 0 and 1. With 0, `sparse_values` is empty. With 1, `sparse_values` has BFLOAT16 dtype and the same shape as `sparse_indices`. Scores correspond to indices element by element; output indices are not guaranteed to be sorted.
  - If the number of valid positions is less than `topk`, remaining indices are filled with -1 and corresponding scores with negative infinity. `output_idx_offset` applies only to valid indices; invalid indices remain -1.
  - `mask_mode` supports 0 and 3. `cmp_ratio` is an integer in [1,128]. Pass `cmp_residual_k` only when `mask_mode` is 3 and `cmp_ratio` is not 1; it is required in that case.
  - All input tensors must be on the same NPU device. After obtaining the first-stage candidate indices and actual valid counts, generate this operator's metadata and pass it to the computation. Regenerate metadata when candidate counts or sequence information change.
<!-- end id8 -->

<!-- npu="A3,910b" id7 -->
- <term>Atlas A3 products</term>, <term>Atlas A2 products</term>:
  - `quant_mode` supports 2 and 7: 2 means INT8 (8-bit storage and computation), and 7 means INT4 (W4A4 4-bit storage and computation).
  - `q` and `k` support INT8 and INT4; FLOAT8_e4m3fn, HIFLOAT8 and FLOAT4_e2m1 are not supported. When `quant_mode` is 2, `q` and `k` must be INT8; when it is 7, they must be INT4 (stored as packed uint8, with the low nibble preceding the high nibble in each byte, no zero-point offset, and a logical element count equal to twice the packed byte count; passed through the PyTorch Adapter as a `torch.uint8` tensor).
  - `q_descale` and `k_descale` support FLOAT16; FLOAT32 and FLOAT8_e8m0 are not supported. When `quant_mode` is 7, the dequantization granularity is the same as for 2 (per token and head for q, and per token for k), with no shape change.
  - `w` supports FLOAT16; FLOAT32 is not supported.
  - `return_value=1` is not supported (`sparse_values` is always of shape (0,)).
  - N of `k` supports only 1; the ratio of N of `q` to N of `k` supports only 64 or 32 (that is, N of `q` supports 64 or 32).
  - `topk` supports [1, 2048].
  - `layout_k` supports only `PA_BBND`, so `block_table` and `seqused_k` are required and `cu_seqlens_k` must not be passed.
  - `layout_q` supports `BSND` and `TND`; with `TND`, `cu_seqlens_q` is required and `layout_k` must be `PA_BBND`.
  - `candidate_block_length` is a reserved interface (not consumed in this release) and only accepts an empty tensor; passing a non-empty tensor makes tiling fail.
  - q_descale has the same shape as w; k_descale has the shape of k with its last axis removed.
  - cmp_ratio supports only 1, 2, 4, 8, 16, 32, 64 and 128.
  - The logical head dimension is 128. For PA_BBND, k has shape (block_num,block_size,1,128), with block_size a multiple of 16. k and k_descale support padding on axis 0; all inner axes must be contiguous. When `quant_mode` is 7, this shape still describes logical elements (128 INT4 elements are packed into 64 bytes), and each PA page row occupies 64 physical bytes.
<!-- end id7 -->

## Usage Example

| Calling method | Example | Description |
|----------------|---------|-------------|
| PyTorch API | [test_torch_quant_sparse_lightning_indexer](examples/test_torch_quant_sparse_lightning_indexer.py) | Call through [quant_sparse_lightning_indexer](docs/torchapi_quant_sparse_lightning_indexer.md). |
| aclnn API | [test_aclnn_quant_sparse_lightning_indexer](examples/test_aclnn_quant_sparse_lightning_indexer.cpp) | Call this operator through the two-phase interface [aclnnQuantSparseLightningIndexer](docs/aclnnQuantSparseLightningIndexer.md). |
