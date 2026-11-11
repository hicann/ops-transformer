# QuantSparseLightningIndexer

The **second stage** operator (consumer, candidate-consuming) of the two-level TopK scheme (coarse block-level screening followed by fine-grained selection inside the candidate blocks). It must be used together with the first-stage operator [QuantLightningIndexerV2](../quant_lightning_indexer_v2/README.md): the first stage selects candidate blocks at block granularity from the whole sequence and outputs the block-level index `candidate_block_indices`, and this operator performs token-level TopK within the set of candidate blocks.

## Product Support

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Not supported
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

> Not supported on Ascend 950PR&950DT series products: only the arch22 (910B/910_93) implementation is provided.

## Function Description

- Operator role: this is the second stage of the two-level TopK. The first-stage operator `QuantLightningIndexerV2` enables source with `candidate_topk_blocks=2048`, splits each token's row (the key sequence) into blocks of `candidate_block_size` positions, scores the blocks and selects `candidate_topk_blocks` candidate blocks, then outputs the block-level index `candidate_block_indices`. This operator receives that index, performs token-level TopK only on the positions covered by candidate blocks, and outputs the final `sparse_indices` for sparse attention.
- API function: sparse-attention preprocessing in inference scenarios; selects the key sparse tokens within candidate blocks and quantizes the input query and key (8-bit storage, 8-bit computation) for maximum benefit.
- Formula: identical to QuantLightningIndexerV2, except that the TopK domain is restricted to candidate blocks:

    $$out = \text{Top-}k\left\{[1]_{1\times g}@\left[(W@[1]_{1\times S_{k}})\odot\text{ReLU}\left(\left(Scale_Q@Scale_K^T\right)\odot\left(Q_{index}^{Quant}@{\left(K_{index}^{Quant}\right)}^T\right)\right)\right]\right\},\quad p \in \text{candidate block}$$

    Block-domain definition: position $p$ belongs to block $\lfloor p / \text{candidate\_block\_size} \rfloor$; only positions whose block index appears in `candidate_block_indices` participate in TopK, and all other positions are set to $-\infty$; slots left when the number of valid positions is less than Top-k output `-1`.
    Main computation steps:
    1. Multiply the input `q` of a token ($Q_{index}^{Quant}\in\R^{g\times d}$) by the given context `k` ($K_{index}^{Quant}\in\R^{S_{k}\times d}$) to obtain the relevance.
    2. Multiply the relevance by the dequantization scales `q_descale` ($Scale_Q$) and `k_descale` ($Scale_K^T$) of `q` and `k`, and filter out invalid negative correlations with $ReLU$ to obtain the relevance score vector between the current token and preceding tokens.
    3. Mask the score vector with candidate blocks (set to $-\infty$), multiply by the weights `w` ($W$), and select the top $Top-k$ indices along the g direction to obtain the output $out$.
- Typical call chain: QuantSparseLightningIndexerMetadata (the preceding operator of this operator, produces the core-partitioning `metadata`) → QuantLightningIndexerV2 (`candidate_topk_blocks=2048`, source enabled, outputs `candidate_block_indices`) → this operator (input `candidate_block_indices`) → sparse attention.

## Parameter Description

| Parameter | Input/Output/Attr | Description | Data Type | Format |
|-----------|-------------------|-------------|-----------|--------|
| q | Input | $Q_{index}^{Quant}\in\R^{g\times d}$ in the formula, the input Index Query; non-contiguous input is not supported. | INT8 | ND |
| k | Input | $K_{index}^{Quant}\in\R^{S_{k}\times d}$ in the formula, the compressed input Index Key; `k`/`k_descale` support padding-style non-contiguity on axis 0 (PA_BBND layout only, other axes must be contiguous). The stride is carried by the input descriptor and the kernel addresses by block index × stride. | INT8 | ND |
| w | Input | $W$ in the formula, the weight coefficients; non-contiguous input is not supported. | FLOAT16 | ND |
| q_descale | Input | $Scale_Q$ in the formula, the dequantization scale of Index Query; non-contiguous input is not supported; shape is the same as `w`. | FLOAT16 | ND |
| k_descale | Input | $Scale_K$ in the formula, the dequantization scale of Index Key; non-contiguity on axis 0 is supported; shape is `k` with the D axis removed. | FLOAT16 | ND |
| cu_seqlens_q | Optional input | Required when layout_q is TND; prefix sums of the valid token counts of `q` per batch; must not be passed when layout_q is BSND. | INT32 | ND |
| cu_seqlens_k | Optional input | Must not be passed when layout_k is PA_BBND. | INT32 | ND |
| seqused_q | Optional input | May be passed when layout_q is BSND; the number of valid tokens of `q` per batch. | INT32 | ND |
| seqused_k | Optional input | The number of valid tokens of `k` per batch; **required** when layout_k is PA_BBND. | INT32 | ND |
| cmp_residual_k | Optional input | Residual length of Key in compression scenarios; must satisfy 0 \<= cmp_residual_k\[i\] \< cmp_ratio. | INT32 | ND |
| block_table | Optional input | Block mapping table used by KV storage in PageAttention; **required** when layout_k is PA_BBND. | INT32 | ND |
| output_idx_offset | Optional input | Offset of the output indices, added only to `sparse_indices`; one dimension less than `sparse_indices`: (B,S1,N2) for BSND and (T,N2) for TND. | INT32 | ND |
| candidate_block_length | Optional input | **Reserved interface**; only an empty tensor is accepted, and a non-empty tensor makes tiling fail. | INT32 | ND |
| metadata | Input (required) | Core-partitioning information produced by the QuantSparseLightningIndexerMetadata operator, including the number of cores used, block size, and the start point of the data handled by each core; shape is fixed to (1024,); **must not be empty**. | INT32 | ND |
| candidate_block_indices | Input (required) | Block-level candidate index output by the first-stage operator QuantLightningIndexerV2 when source is enabled (`candidate_topk_blocks=2048`). **The last dimension is the number of candidate blocks (currently only 2048 is supported; it is no longer an input of this operator)**; the element count must equal B\*S1\*N2\*last-dim for BSND, and T\*N2\*last-dim for TND (T is dim 0 of `q`). Values are in [0, numBlocks) or -1 (invalid slot). | INT32 | ND |
| topk | Attr | Number of indices to keep in the TopK stage; default 2048. | INT32 | - |
| quant_mode | Attr | Quantization mode of the input; this operator **supports only 2** (INT8 quantization). The OpDef default value 1 is illegal for this operator. | INT32 | - |
| max_seqlen_q | Optional attr | Maximum query sequence length; the default -1 means any possible length. | INT32 | - |
| layout_q | Optional attr | Data layout of `q`; "BSND"/"TND" are supported; default "BSND". | STRING | - |
| layout_k | Optional attr | Data layout of `k`; default "BSND"; on 910B/910_93 **only "PA_BBND"** is supported (paged addressing through block_table). | STRING | - |
| mask_mode | Optional attr | Mask mode; only 0 (no mask) and 3 (lower triangular) are supported; default 0. | INT32 | - |
| cmp_ratio | Optional attr | Key compression ratio used for sparse computation; default 1. | INT32 | - |
| return_value | Optional attr | Whether to output `sparse_values`; default 0; on 910B/910_93 only 0 is supported (no value output). | INT32 | - |
| candidate_block_size | Optional attr | Candidate block size (number of positions per block); must be > 0; currently only 8 is supported. | INT32 | - |
| sparse_indices | Output | Token indices used by sparse attention. Shape is (B,S1,N2,topk) for BSND and (T,N2,topk) for TND. | INT32 | ND |
| sparse_values | Output | Values corresponding to `sparse_indices`. On 910B/910_93 `return_value` is fixed to 0 and the shape is (0,). | BFLOAT16 | ND |

## Constraints

<!-- npu="A3,910b" id7 -->
- <term>Atlas A3 products</term>, <term>Atlas A2 products</term>:
  - This operator is currently available only on these two product lines (arch22).
  - `quant_mode` supports only 2.
  - `q` and `k` support INT8; FLOAT8_e4m3fn, HIFLOAT8 and FLOAT4_e2m1 are not supported.
  - `q_descale` and `k_descale` support FLOAT16; FLOAT32 and FLOAT8_e8m0 are not supported.
  - `w` supports FLOAT16; FLOAT32 is not supported.
  - `return_value=1` is not supported (`sparse_values` is always of shape (0,)).
  - N of `k` supports only 1; the ratio of N of `q` to N of `k` supports only 64 or 32 (that is, N of `q` supports 64 or 32).
  - `topk` supports [1, 2048].
  - `layout_k` supports only `PA_BBND`, so `block_table` and `seqused_k` are required and `cu_seqlens_k` must not be passed.
  - `layout_q` supports `BSND` and `TND`; with `TND`, `cu_seqlens_q` is required and `layout_k` must be `PA_BBND`.
  - `candidate_block_length` is a reserved interface (not consumed in this release) and only accepts an empty tensor; passing a non-empty tensor makes tiling fail.
<!-- end id7 -->
- Candidate-related constraints:
  - `candidate_block_indices` is a required input of type INT32; **its last dimension is the number of candidate blocks (currently only 2048 is supported)**, and the element count must equal `B*S1*N2*last-dim` for BSND and `T*N2*last-dim` for TND, otherwise tiling fails.
  - The number of candidate blocks is no longer an input of this operator: tiling derives and validates it from the last dimension of `candidate_block_indices` (the accumulator, extraction and copy-out logic are all designed with 64 alignment, and it must be consistent with the candidate dimension output by the first stage).
  - `candidate_block_size` must be > 0; currently only 8 is supported (BlockReduceMax reduces over a 32B block, i.e. 8 float32 values).
  - `candidate_block_length` is a reserved interface; only an empty tensor is accepted, and a non-empty value makes tiling fail.
  - `metadata` is a required input and must not be empty.
  - The dimension of `output_idx_offset` must be one less than that of `sparse_indices`.

## Usage Example

| Calling method | Example | Description |
|----------------|---------|-------------|
| PyTorch API | [quant_sparse_lightning_indexer.py](torch_extension/quant_sparse_lightning_indexer.py) | Call this operator through `torch.ops.cann_ops_transformer.quant_sparse_lightning_indexer` (the argument is `candidate_block_indices`, the block-level candidate index output by the first-stage operator). Graph mode is not supported yet. `metadata` must be generated by the metadata operator first (this extension does not build it across operators automatically). |
| aclnn API | [test_aclnn_quant_sparse_lightning_indexer](examples/test_aclnn_quant_sparse_lightning_indexer.cpp) | Call this operator through the two-phase interface [aclnnQuantSparseLightningIndexer](docs/aclnnQuantSparseLightningIndexer.md). |
