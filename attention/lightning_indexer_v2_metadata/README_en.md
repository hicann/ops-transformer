# LightningIndexerV2Metadata

English | [简体中文](./README.md)

## Product Support

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
|<term>Ascend 950PR&950DT products</term>|      √     |
|<term>Atlas A3 products</term>|      √     |
|<term>Atlas A2 products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function Description

- Operator function: `LightningIndexerV2Metadata` is the preprocessing operator of the `LightningIndexerV2` operator and is used to generate a load-balanced task partitioning scheme. This operator does not perform the actual LightningIndexer computation. Instead, based on the input parameters, it computes the start and end ranges of the computation that each AI Core should process on the AI CPU, maximizing the utilization of computing resources and avoiding load imbalance between cores.

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
      <th>Parameter Name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data Type</th>
      <th>Data Format</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>cu_seqlens_q</td>
      <td>Optional Input</td>
      <td>Cumulative prefix sum of the Sequence Lengths of q in different batches. The shape is (B+1, ), and the first value is 0. The default value is None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_k</td>
      <td>Optional Input</td>
      <td>Cumulative prefix sum of the Sequence Lengths of k in different batches. The shape is (B+1, ), and the first value is 0. The default value is None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_q</td>
      <td>Optional Input</td>
      <td>Sequence Length of q that actually participates in the computation in each batch. The shape is (B, ). The default value is None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_k</td>
      <td>Optional Input</td>
      <td>Sequence Length of k that actually participates in the computation in each batch. The shape is (B, ). The default value is None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_residual_k</td>
      <td>Optional Input</td>
      <td>Remainder of the pre-compression Sequence Length of k in each batch divided by cmp_ratio. It is used together with cmp_ratio to restore the pre-compression length. The shape is (B, ). It must be passed in the scenario where cmp_ratio is not 1 and mask_mode is 3. The default value is None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num_heads_q</td>
      <td>Attribute</td>
      <td>Number of heads of q. Currently, [1, 64] is supported.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_heads_k</td>
      <td>Attribute</td>
      <td>Number of heads of k. Currently, only 1 is supported.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>head_dim</td>
      <td>Attribute</td>
      <td>Dimension of the attention head. Currently, only 128 is supported.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>topk</td>
      <td>Attribute</td>
      <td>Number of k token indexes retained for each q token. Currently, only [1, 8192] is supported.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>batch_size</td>
      <td>Optional Attribute</td>
      <td>Number of batches.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_q</td>
      <td>Optional Attribute</td>
      <td>Maximum Sequence Length of q. -1 indicates any possible length.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_k</td>
      <td>Optional Attribute</td>
      <td>Maximum Sequence Length of k. -1 indicates any possible length.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_q</td>
      <td>Optional Attribute</td>
      <td>Arrangement format of q. BSND and TND are supported.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_k</td>
      <td>Optional Attribute</td>
      <td>Arrangement format of k. BSND, TND, and PA_BBND are supported.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mask_mode</td>
      <td>Optional Attribute</td>
      <td>Sparse mode. 0 indicates No mask, and 3 indicates the rightDownCausal mode.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_ratio</td>
      <td>Optional Attribute</td>
      <td>Compression ratio of k. The value range is [1, 128], and 1 indicates no compression.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadata</td>
      <td>Output</td>
      <td>Load balancing result output. The shape is fixed to (1024, ).</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

## Constraints

- The LightningIndexerV2Metadata operator must be used together with the LightningIndexerV2 operator.
- B (Batch) indicates the batch size of input samples. q and k are input parameters of the supporting LightningIndexerV2 operator. S1 indicates the size of the S axis in the q shape when layout_q=BSND, and S2 indicates the size of the S axis in the k shape when layout_k=BSND.
- The values of parameters cu_seqlens_q and cu_seqlens_k must be the cumulative numbers of valid tokens in the current batch and preceding batches. The first element is fixed to 0, and the value of each subsequent element must be greater than or equal to that of the preceding element.
- The values of parameters seqused_q and seqused_k must indicate the number of valid tokens in each batch.
- Parameter cmp_residual_k must satisfy cmp_residual_k[i] < cmp_ratio.
- For details about the mask modes indicated by mask_mode, see [sparse_mode Parameter Description](../../docs/zh/context/sparse_mode_introduction_en.md).
- In the non-PA scenario, layout_q and layout_k must be the same.
- layout_q=BSND scenario
  - max_seqlen_q must be passed the value of S1.
- layout_k=BSND scenario
  - max_seqlen_k must be passed the value of S2.
- layout_q=TND scenario
  - cu_seqlens_q must be passed.
- layout_k=TND scenario
  - cu_seqlens_k must be passed.
- layout_k=PA_BBND scenario
  - seqused_k must be passed.
- Batch value rules
  - When layout_q is BSND, the batch is preferentially derived from the shape of seqused_q; if seqused_q is not passed, the number of batches is obtained from batch_size.
  - When layout_q is TND, the batch is preferentially derived from the shape of seqused_q; if seqused_q is not passed, the batch is derived from the shape of cu_seqlens_q.
- q Seqlen value rules
  - When layout_q is BSND, the seqlen is preferentially obtained from the elements in seqused_q; if seqused_q is not passed, the seqlen is obtained from max_seqlen_q.
  - When layout_q is TND, the seqlen is preferentially obtained from the elements in seqused_q; if seqused_q is not passed, the seqlen is obtained from the elements in cu_seqlens_q.
- k Seqlen value rules
  - When layout_k is BSND, the seqlen is preferentially obtained from the elements in seqused_k; if seqused_k is not passed, the seqlen is obtained from max_seqlen_k.
  - When layout_k is TND, the seqlen is preferentially obtained from the elements in seqused_k; if seqused_k is not passed, the seqlen is obtained from the elements in cu_seqlens_k.

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn API | [test_aclnn_lightning_indexer_v2_metadata](./examples/test_aclnn_lightning_indexer_v2_metadata.cpp) | Call the LightningIndexerV2Metadata operator through the [aclnnLightningIndexerV2Metadata](docs/aclnnLightningIndexerV2Metadata_en.md) interface |
| PyTorch API | [test_torch_lightning_indexer_v2_metadata](./examples/test_torch_lightning_indexer_v2_metadata.py) | Call the LightningIndexerV2Metadata operator through the [lightning_indexer_metadata](../lightning_indexer_v2/docs/torchapi_lightning_indexer_en.md) interface |
