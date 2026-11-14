# QuantSparseLightningIndexerMetadata

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
      <th>Input/Output/Attr</th>
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
      <td>Reserved interface for the candidate block length; only an empty tensor is accepted.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num_heads_q</td>
      <td>Attr</td>
      <td>Number of heads of q; currently supports [1, 64]; 64 or 32 when used as the preceding operator of the second-stage consumer.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_heads_k</td>
      <td>Attr</td>
      <td>Number of heads of k; currently only 1 is supported.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>head_dim</td>
      <td>Attr</td>
      <td>Dimension of the attention head; currently only 128 is supported.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>topk</td>
      <td>Attr</td>
      <td>Number of key sparse tokens selected from q; currently supports [1, 8192].</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quant_mode</td>
      <td>Attr</td>
      <td>Quantization mode; currently supports 1/2/3/4/5. 1 means qk: fp8(e4m3) per-token-head, scale: fp32; 2 means qk: int8 per-token-head, scale: fp16, w: fp16; 3 means qk: mxfp8(e4m3), scale: fp8(e8m0); 4 means qk: hif8 per-tensor, scale: fp32; 5 means qk: mxfp4(e2m1), scale: fp8(e8m0).</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>batch_size</td>
      <td>Optional attr</td>
      <td>Batch size; default 0.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_q</td>
      <td>Optional attr</td>
      <td>Maximum sequence length of q; -1 means any possible length; default -1.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_k</td>
      <td>Optional attr</td>
      <td>Maximum sequence length of k; -1 means any possible length; default -1.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_q</td>
      <td>Optional attr</td>
      <td>Layout of q; BSND and TND are supported; default BSND.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_k</td>
      <td>Optional attr</td>
      <td>Layout of k; BSND, TND and PA_BBND are supported; default BSND.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mask_mode</td>
      <td>Optional attr</td>
      <td>Sparse mode; 0 means No mask and 3 means rightDownCausal; default 0.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_ratio</td>
      <td>Optional attr</td>
      <td>Compression ratio of k, in the range [1, 128]; default 1, meaning no compression.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>candidate_block_size</td>
      <td>Optional attr</td>
      <td>Candidate block size (number of positions per block); consistent with candidate_block_size of QuantSparseLightningIndexer.</td>
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

  <ul>
    <li><term>Atlas A3 products</term> : num_heads_q supports 64 or 32 (gSize of the second-stage consumer supports 64/32); quant_mode = 1/3/4/5 is not supported; topk supports only [1, 2048]; layout_k = BSND/TND is not supported; cmp_ratio does not accept arbitrary values in [1, 128], only cmp_ratio = 1/2/4/8/16/32/64/128.</li>
    <li><term>Atlas A2 products</term> : num_heads_q supports 64 or 32 (gSize of the second-stage consumer supports 64/32); quant_mode = 1/3/4/5 is not supported; topk supports only [1, 2048]; layout_k = BSND/TND is not supported; cmp_ratio does not accept arbitrary values in [1, 128], only cmp_ratio = 1/2/4/8/16/32/64/128.</li>
  </ul>

## Constraints

- `QuantSparseLightningIndexerMetadata` must be used together with the `QuantSparseLightningIndexer` operator.
- B (Batch) is the input batch size; q and k are the inputs of the matching `QuantSparseLightningIndexer` operator; S1 is the size of the S axis in the shape of q when layout_q = BSND; S2 is the size of the S axis in the shape of k when layout_k = BSND.
- `cu_seqlens_q` and `cu_seqlens_k` must contain cumulative values of the valid token counts of the current batch and all preceding batches; the first element is fixed to 0 and each following element must be greater than or equal to the previous one.
- `seqused_q` and `seqused_k` must represent the number of valid tokens in each batch.
- `cmp_residual_k` must satisfy cmp_residual_k[i] < cmp_ratio.
- `candidate_block_length` is a reserved interface and only accepts an empty tensor (shapeSize 0); a non-empty tensor is rejected.
- `candidate_block_size` currently supports only 8, consistent with `candidate_block_size` of `QuantSparseLightningIndexer`.
- For details about the mask modes represented by mask_mode, see [sparse_mode parameter description](../../docs/zh/context/sparse_mode_introduction.md).
- In non-PA scenarios, layout_q and layout_k must be the same.
- layout_q = BSND scenario
  - max_seqlen_q must be set to the value of S1.
- layout_k = BSND scenario
  - max_seqlen_k must be set to the value of S2.
- layout_q = TND scenario
  - cu_seqlens_q must be passed.
- layout_k = TND scenario
  - cu_seqlens_k must be passed.
- layout_k = PA_BBND scenario
  - seqused_k must be passed.
- Batch derivation rules
  - When layout_q is BSND, batch is preferably derived from the shape of seqused_q; if seqused_q is not passed, batch is taken from batch_size.
  - When layout_q is TND, batch is preferably derived from the shape of seqused_q; if seqused_q is not passed, batch is derived from the shape of cu_seqlens_q.
- q seqlen derivation rules
  - When layout_q is BSND, seqlen is preferably taken from the elements of seqused_q; if seqused_q is not passed, seqlen is taken from max_seqlen_q.
  - When layout_q is TND, seqlen is preferably taken from the elements of seqused_q; if seqused_q is not passed, seqlen is taken from the elements of cu_seqlens_q.
- k seqlen derivation rules
  - When layout_k is BSND, seqlen is preferably taken from the elements of seqused_k; if seqused_k is not passed, seqlen is taken from max_seqlen_k.
  - When layout_k is TND, seqlen is preferably taken from the elements of seqused_k; if seqused_k is not passed, seqlen is taken from the elements of cu_seqlens_k.

## Usage

| Calling method | Sample code | Description |
| -------------- | ----------- | ----------- |
| aclnn API | [test_aclnn_quant_sparse_lightning_indexer_metadata](./examples/test_aclnn_quant_sparse_lightning_indexer_metadata.cpp) | Call the QuantSparseLightningIndexerMetadata operator through the [aclnnQuantSparseLightningIndexerMetadata](./docs/aclnnQuantSparseLightningIndexerMetadata.md) interface |
| PyTorch API | [test_torch_quant_sparse_lightning_indexer_metadata](./examples/test_torch_quant_sparse_lightning_indexer_metadata.py) | Call the QuantSparseLightningIndexerMetadata operator through the [quant_sparse_lightning_indexer_metadata](../quant_sparse_lightning_indexer/torch_extension/quant_sparse_lightning_indexer.py) interface |
