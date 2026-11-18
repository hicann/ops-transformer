# QuantSparseLightningIndexerMetadata

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

- Operator function: `QuantSparseLightningIndexerMetadata` is the preceding operator of the `QuantSparseLightningIndexer` operator (the second stage / consumer of the two-level TopK); it produces a load-balanced task partitioning plan. It does not perform the actual LightningIndexer computation. Instead, it computes on the AI CPU the start and end ranges of the computation each AI Core should handle, based on the input parameters, so as to maximize the utilization of compute resources and avoid load imbalance across cores.

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 180px">
  <col style="width: 100px">
  <col style="width: 700px">
  <col style="width: 90px">
  <col style="width: 80px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data Type</th>
      <th>Format</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>cu_seqlens_q</td>
      <td>Optional input</td>
      <td>Prefix sums of the valid sequence length of q per batch, with shape (B+1, ); required only when layout_q is TND; the first value is fixed to 0.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_k</td>
      <td>Optional input</td>
      <td>Prefix sums of the valid sequence length of k per batch, with shape (B+1, ); required only when layout_k is TND; the first value is fixed to 0.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_q</td>
      <td>Optional input</td>
      <td>The sequence length of q actually involved in the computation per batch, with shape (B, ).</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_k</td>
      <td>Optional input</td>
      <td>The sequence length of k actually involved in the computation per batch, with shape (B, ).</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_residual_k</td>
      <td>Optional input</td>
      <td>The remainder of the compressed sequence length of k per batch; used together with cmp_ratio for masking and load computation, with shape (B, ). Required when cmp_ratio is not 1 and mask_mode is 3.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>candidate_block_length</td>
      <td>Optional input</td>
      <td>Actual valid candidate count for each Query row, in [0,2048]. Passing requirements are described in Constraints.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num_heads_q</td>
      <td>Attribute</td>
      <td>Number of Query heads; supports [1,32] and 64.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_heads_k</td>
      <td>Attribute</td>
      <td>Number of heads of k; currently only 1 is supported.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>head_dim</td>
      <td>Attribute</td>
      <td>Dimension of the attention head; currently only 128 is supported.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>topk</td>
      <td>Attribute</td>
      <td>Number of key sparse tokens selected from q; currently supports [1, 8192].</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quant_mode</td>
      <td>Attribute</td>
      <td>Indicates the quantization mode of the input.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>batch_size</td>
      <td>Optional attribute</td>
      <td>Batch size; default 0.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_q</td>
      <td>Optional attribute</td>
      <td>Maximum sequence length of q; -1 means any possible length; default -1.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_k</td>
      <td>Optional attribute</td>
      <td>Maximum sequence length of k; -1 means any possible length; default -1.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_q</td>
      <td>Optional attribute</td>
      <td>Layout of q; BSND and TND are supported; default BSND.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_k</td>
      <td>Optional attribute</td>
      <td>Layout of k; BSND, TND and PA_BBND are supported; default BSND.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mask_mode</td>
      <td>Optional attribute</td>
      <td>Sparse mode; 0 means No mask and 3 means rightDownCausal; default 0.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_ratio</td>
      <td>Optional attribute</td>
      <td>Compression ratio of k, in the range [1, 128]; default 1, meaning no compression.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>candidate_block_size</td>
      <td>Optional attribute</td>
      <td>Number of Key positions in each candidate block; default 8. Only 8 is supported and it must match QuantSparseLightningIndexer.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadata</td>
      <td>Output</td>
      <td>Load-balancing result; shape is fixed to (1024, )</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

## Constraints

- `QuantSparseLightningIndexerMetadata` must be used together with the `QuantSparseLightningIndexer` operator.
- B (Batch) is the input batch size; q and k are the inputs of the matching `QuantSparseLightningIndexer` operator; S1 is the size of the S axis in the shape of q when layout_q = BSND; S2 is the size of the S axis in the shape of k when layout_k = BSND.
- `cu_seqlens_q` and `cu_seqlens_k` must contain cumulative values of the valid token counts of the current batch and all preceding batches; the first element is fixed to 0 and each following element must be greater than or equal to the previous one.
- `seqused_q` and `seqused_k` must represent the number of valid tokens in each batch.
- `cmp_residual_k` must satisfy cmp_residual_k[i] < cmp_ratio.
- For details about the mask modes represented by mask_mode, see [sparse_mode parameter description](../../docs/zh/context/sparse_mode_introduction.md).

<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT products</term>:
  - `quant_mode` supports only 5 (MXFP4).
  - `num_heads_q` is in [1,32], `num_heads_k` must be 1, and `head_dim` must be 128.
  - `topk` is in [1,8192]. `candidate_block_size` supports only 8.
  - `candidate_block_length` is required and must have INT32 dtype. Each element is the actual valid candidate count for the corresponding query, in [0,2048].
  - The output `metadata` is a contiguous INT32 tensor with shape `(1024,)`, on the same NPU device as the input.
  - Sequence information, layouts, masking and compression parameters must match the computation. Regenerate metadata when these parameters or candidate counts change.
  - The generated metadata is used only by QuantSparseLightningIndexer and must not be substituted with QuantLightningIndexerV2 metadata.
  - layout_q supports BSND and TND; layout_k supports BSND, TND and PA_BBND. Non-paged layouts must match.
  - TND Query or Key requires the corresponding cu_seqlens_q or cu_seqlens_k. BSND layouts do not accept prefix sums.
  - For BSND Key, max_seqlen_k must equal the physical Key sequence length. For PA_BBND, at least one of seqused_k and max_seqlen_k is required; if seqused_k is omitted, max_seqlen_k must match the logical Key capacity used by the compute interface.
<!-- end id7 -->

<!-- npu="A3,910b" id8 -->
- <term>Atlas A3 products</term>, <term>Atlas A2 products</term>:
  - num_heads_q supports only 32 and 64; quant_mode supports only 2; topk supports only [1,2048]; layout_k supports only PA_BBND; cmp_ratio supports only 1, 2, 4, 8, 16, 32, 64 and 128.
  - candidate_block_length is reserved; omit it or pass an empty tensor. A non-empty tensor is not supported.
  - layout_q = BSND scenario
    - max_seqlen_q must be set to the value of S1.
  - layout_q = TND scenario
    - cu_seqlens_q must be passed.
  - layout_k = PA_BBND scenario
    - seqused_k must be passed.
  - Batch derivation rules
    - When layout_q is BSND, batch is preferably derived from the shape of seqused_q; if seqused_q is not passed, batch is taken from batch_size.
    - When layout_q is TND, batch is preferably derived from the shape of seqused_q; if seqused_q is not passed, batch is derived from the shape of cu_seqlens_q.
  - q seqlen derivation rules
    - When layout_q is BSND, seqlen is preferably taken from the elements of seqused_q; if seqused_q is not passed, seqlen is taken from max_seqlen_q.
    - When layout_q is TND, seqlen is preferably taken from the elements of seqused_q; if seqused_q is not passed, seqlen is taken from the elements of cu_seqlens_q.
<!-- end id8 -->

## Usage

| Calling method | Sample code | Description |
| -------------- | ----------- | ----------- |
| aclnn API | [test_aclnn_quant_sparse_lightning_indexer_metadata](./examples/test_aclnn_quant_sparse_lightning_indexer_metadata.cpp) | Call the QuantSparseLightningIndexerMetadata operator through the [aclnnQuantSparseLightningIndexerMetadata](./docs/aclnnQuantSparseLightningIndexerMetadata.md) interface |
| PyTorch API | [test_torch_quant_sparse_lightning_indexer_metadata](./examples/test_torch_quant_sparse_lightning_indexer_metadata.py) | Call the QuantSparseLightningIndexerMetadata operator through the [quant_sparse_lightning_indexer_metadata](docs/torchapi_quant_sparse_lightning_indexer_metadata.md) interface |
