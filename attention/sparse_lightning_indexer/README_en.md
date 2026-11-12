# SparseLightningIndexer

English | [简体中文](./README.md)

## Product Support

| Product                                                      | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                      |    ×     |
| <term>Atlas A3 products</term>                                |    √     |
| <term>Atlas A2 products</term>                                |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                       |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                               |    ×     |

## Description

- Operator function: SparseLightningIndexer (candidate consumer) uses the weighted ReLU scoring and TopK framework of LightningIndexerV2. It **consumes the candidate block indices produced by the upstream source operator** (`candidate_block_indices`) and performs **leak-free TopK within candidate blocks** for each query row. Reachable positions covered by candidate blocks are ranked using their actual scores. All other positions, including unreachable positions and reachable positions outside candidate blocks, are invalid and are not selected. **Indices outside candidate blocks never appear in the topk output**, and when the number of valid entries within candidates is less than topk, the remaining slots are filled with -1. This operator is used together with `lightning_indexer` (source mode, `candidate_topk_blocks != -1`) to form the consumer side of the two-level TopK over candidate blocks.

- Calculation formula (for each batch b and query row i, with N2=1):

  $$
  \begin{aligned}
  score(b,i,:) &= \mathrm{DoReduce}(W \odot ReLU(Q_{index} @ K_{index}^{T})) \in \R^{S_2} \\
  score'(b,i,p) &= \begin{cases}
      score(b,i,p) & p < vl(b,i) \text{ and } \lfloor p / blockSize \rfloor \in candSet(b,i) \\
      -\infty      & \text{otherwise (} p \geq vl(b,i) \text{, or reachable but not in candidates)}
  \end{cases} \\
  nvalid(b,i) &= \left|\{\, p : score'(b,i,p) > -\infty \,\}\right| \\
  sparseIndices(b,i,k) &= \begin{cases}
      \mathrm{topK}(score'(b,i,:))_k & k < \min(topk,\ nvalid(b,i)) \\
      -1 & \min(topk,\ nvalid(b,i)) \le k < topk
  \end{cases} \\
  sparseValues(b,i,k) &= \begin{cases}
      score(b,i,\ sparseIndices(b,i,k)) & k < \min(topk,\ nvalid(b,i)),\ \text{return\_value}=1 \\
      \text{always-empty tensor } (0,) & \text{return\_value}=0
  \end{cases}
  \end{aligned}
  $$

  where `vl(b,i)` is the reachable prefix length of the row (derived from the causal mask, seqused_k/cu_seqlens_k, and cmp_ratio), `blockSize` = `candidate_block_size`, and `candSet` is the set of candidate block numbers (duplicates, out-of-range values, and -1 are treated as "non-candidate").

- The main computation process is as follows:

  1. The same scoring process as LIV2: QK correlation → ReLU → W-weighted sum over the g dimension to obtain the score row.
  2. Builds the position-level candidate mask based on the candidate block indexes: positions covered by candidate blocks are candidates, and the remaining positions are non-candidates.
  3. Merges reachable positions outside candidates and unreachable positions into the same category of invalid positions (score set to -inf and index set to -1), which are not selected.
  4. Outputs indexes through the same position-level TopK pipeline as LIV2 (when nvalid is less than topk, the remaining slots are -1, and a fully invalid row outputs all -1).

## Parameter Description

> **Description:**<br>
> Parameter dimension meanings: B indicates Batch Size, S1 and S2 indicate the Sequence Length of q and k respectively, N1 and N2 indicate the Head Num of q and k respectively (N2 supports only 1), D indicates Head Dim (only 128 is supported), and T indicates the Total Tokens of q/k. block_num and block_size indicate the total number of blocks and the number of tokens contained in each block in the PageAttention scenario respectively (block_size is in the range [16, 1024] and is a multiple of 16).

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
|--------|---------------|------|----------|----------|
| q | Input | Input Q in the formula (the same as the source operator). Empty tensors are not supported. When layout_q is BSND, the shape is [B, S1, N1, D]. When layout_q is TND, the shape is [T, N1, D]. | FLOAT16, BFLOAT16 | ND |
| k | Input | Input K in the formula (the same as the source operator). Empty tensors are not supported. When layout_k is PA_BBND, the shape is [block_num, block_size, N2, D]. When layout_k is BSND, the shape is [B, S2, N2, D]. When layout_k is TND, the shape is [T, N2, D]. | FLOAT16, BFLOAT16 | ND |
| w | Input | Input W in the formula (the same as the source operator). Empty tensors are not supported. When layout_q is BSND, the shape is [B, S1, N1]. When layout_q is TND, the shape is [T, N1]. | FLOAT | ND |
| cu_seqlens_q | Input | Cumulative sum (prefix sum) of the number of valid tokens of q in the current and preceding batches. This parameter is required when layout_q is TND. Passing this parameter in the BSND scenario is not recommended. The shape is [B+1, ]. | INT32 | ND |
| cu_seqlens_k | Input | Cumulative sum (prefix sum) of the number of valid tokens of k in the current and preceding batches. This parameter is required when layout_k is TND. Passing this parameter in the BSND scenario causes an error. This parameter is not passed in the PA_BBND scenario. The shape is [B+1, ]. | INT32 | ND |
| seqused_q | Input | Reserved slot for API alignment. This operator (arch22) does not support passing this parameter in. A non-empty value is rejected by the host-side check. Pass an empty value. | INT32 | ND |
| seqused_k | Input | Actual used length of k (after compression) in each batch. This parameter is required when layout_k is PA_BBND. It is optional in the BSND/TND scenarios. When passed in, it participates in the valid length computation of k (actS2Orig = seqused_k×cmp_ratio + cmp_residual_k). The shape is [B, ]. | INT32 | ND |
| cmp_residual_k | Input | Remainder of the number of tokens of k before compression divided by cmp_ratio. The host side does not check the shape/dtype (cross-operator constraint). The shape is [B, ]. | INT32 | ND |
| block_table | Input | Block mapping table used for KV storage in PageAttention. This parameter is required when layout_k is PA_BBND (dim0 must equal B). Passing this parameter in non-PA scenarios causes an error. The shape is [B, maxBlockNumPerSeq]. | INT32 | ND |
| output_idx_offset | Input | Reserved slot for API alignment. This operator (arch22) does not support passing this parameter in (no offset is added to the output index). A non-empty value is rejected by the host-side check. Pass an empty value. | INT32 | ND |
| metadata | Input | Metadata (reserved slot for API alignment). The arch22 kernel does not consume it. A non-empty value is rejected by the host-side check. Pass an empty value. | INT32 | ND |
| candidate_block_indices | Input | Candidate block indexes (directly connected to the output of the source operator lightning_indexer). This parameter is required, and empty tensors are not supported. The value is the relative block number in the compressed K position space within a batch (block number × candidate_block_size is the start position) or -1 (invalid slot). Slots are unordered. Duplicates, out-of-range values, and -1 are allowed and are treated as "non-candidate". output_idx_offset is not added. When layout_q is BSND, the shape is [B, S1, N2, candBlocks]. When layout_q is TND, the shape is [T, N2, candBlocks]. candBlocks is the last dimension of the shape, with the value range (0, 2048] and a multiple of 64. | INT32 | ND |
| candidate_block_length | Input | Reserved API. Only empty tensors are accepted (None/numel==0; a non-empty value causes an error). The shape is (0,). | INT32 | ND |
| sparse_indices | Output | Output Top-k index in the formula (leak-free TopK index). Valid slots are reachable positions within candidates sorted by score in descending order (the order of equal scores is not guaranteed). When the number of valid entries nvalid within candidates is less than topk, all subsequent slots are filled with -1 (all invalid slots, including reachable positions outside candidates, are -1, and no index outside candidates is output). A fully invalid row outputs all -1. When layout_q is BSND, the shape is [B, S1, N2, topk]. When layout_q is TND, the shape is [T, N2, topk]. | INT32 | ND |
| sparse_values | Output | Output Top-k values in the formula. When return_value is 1, the output has the same shape as sparse_indices (dtype FLOAT32). Valid slots are the real TopK scores of selected positions (directly output in the fp32 scoring domain), and the positions corresponding to -1 slots are filled with 0. When return_value is 0 (or default), the output is always an empty tensor (0,), and the kernel does not write to it. | FLOAT | ND |
| topk | Attribute | k value of Top-k. The value range is (0, 2048] (a value out of range directly causes an error). | INT64 | - |
| max_seqlen_q | Attribute | Maximum sequence length of q. Only -1 is supported (the arch22 kernel does not consume it). A value other than -1 causes an error. | INT64 | - |
| layout_q | Attribute | Data layout format of q. BSND/TND is supported. The default value is BSND. | STRING | - |
| layout_k | Attribute | Data layout format of k. BSND/TND/PA_BBND is supported. The default value is BSND. In non-PA_BBND scenarios, the value must be consistent with layout_q. | STRING | - |
| mask_mode | Attribute | Mask mode. 0 indicates that no mask is used, and 3 indicates the causal mask (rightDownCausal). The default value is 0. | INT64 | - |
| cmp_ratio | Attribute | Compression ratio. The value is in the range (0, 128] and is a power of 2. The default value is 1. The value must be consistent with the configuration on the source side. | INT64 | - |
| return_value | Attribute | Whether to output TopK scores. The value is 0 or 1. The default value is 0. 1 = output real TopK scores (sparse_values has the same shape as sparse_indices, and the value of an invalid slot is 0). Other values are rejected. | INT64 | - |
| candidate_block_size | Attribute | Candidate block granularity. A power of 2 in [2,64] (2/4/8/16/32/64). The default value is 8. The value must be consistent with the source side. | INT64 | - |

## Constraints

1. This API can be used in inference scenarios and supports single-operator mode invocation.
2. `candidate_block_indices` is required (REQUIRED, INT32). candBlocks (the last dimension of the shape) is in (0, 2048]
   and is a multiple of 64. This operator has no `candidate_topk_blocks` attribute (the width is inferred from the input).
3. `candidate_block_size` is a power of 2 in [2, 64]. The default value is 8. Other values are rejected.
4. `topk` in (0, 2048] is always checked (a value greater than 2048 causes an error). `return_value` is 0 or 1, and other values cause an error and are rejected.
5. `candidate_block_length` is reserved: only None/empty tensors are accepted. A non-empty value causes the error
   "candidate_block_length is reserved and only empty tensor is supported yet".
6. Other constraints, such as layout combinations, PA block_table, N2=1, and head_dim=128, are the same as those of the upstream operator `lightning_indexer`.
7. Cross-operator constraints (source → consumer):
   - This operator is used in pair with the `lightning_indexer` source mode (`candidate_topk_blocks != -1`);
   - The `candidate_block_size` values on both sides must be the same. **This is a caller constraint, and the framework cannot perform cross-operator checks** (the host on each side only checks the value range. The behavior of a mismatch is undefined);
   - Candidates are valid only for cross-layer sharing **within the same forward**. The source and the consumer should share the same KV domain. When the KV on the consumer side is shorter, out-of-range candidate block numbers naturally degrade to "non-candidate" (robust, and no additional check is required);
   - Candidates are relative block numbers in the **compressed position domain** within a batch (block number × blockSize = start position).
     -1 is an invalid slot, slots are unordered, and **output_idx_offset is not added**;
   - Leak-free semantics: **non-candidate block indexes never enter the topk output**. Reachable positions outside candidates and unreachable positions are both invalid positions
     (score set to -inf and index set to -1; they do not participate in valid sorting and are not selected). When the number of valid entries nvalid within candidates is less than topk
     (including regime 2: valid count within candidates < topk < reachable count), all remaining slots are filled with -1, no index outside candidates is output,
     and a fully invalid row outputs all -1;
   - The output slot order (including the -1 slot filling order and the order of equal real scores) is not externally guaranteed. Only the consistency of the valid slot set and the correctness of the number of -1 slots are guaranteed.

## Calling Example

| Calling Method | Sample Code | Description |
|---------|--------|------|
| aclnn API | [test_aclnn_sparse_lightning_indexer](examples/test_aclnn_sparse_lightning_indexer.cpp) | Calls the SparseLightningIndexer operator through the [aclnnSparseLightningIndexer](docs/aclnnSparseLightningIndexer.md) API. |
| PyTorch API | [sparse_lightning_indexer](docs/torchapi_sparse_lightning_indexer.md) | Calls the SparseLightningIndexer operator through the [cann_ops_transformer.sparse_lightning_indexer](docs/torchapi_sparse_lightning_indexer.md) API (used in pair with the lightning_indexer source mode). |
