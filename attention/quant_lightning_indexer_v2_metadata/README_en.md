# QuantLightningIndexerV2Metadata

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

## Description

- Operator function: `QuantLightningIndexerV2Metadata` is a preceding operator of the `QuantLightningIndexerV2` operator and generates a load-balanced task partition scheme. This operator does not perform the actual LightningIndexer computation. Instead, it computes on the AI CPU the start and end ranges that each AI Core should process based on the input parameters, thereby maximizing the utilization of computing resources and preventing unbalanced loads among Cores.

## Parameters

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
      <td>Optional input</td>
      <td>Prefix sum of accumulated sequence lengths of q in different batches. The shape is (B+1, ). This parameter is required only when layout_q is TND, and the first value is fixed to 0. Default value: None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_k</td>
      <td>Optional input</td>
      <td>Prefix sum of accumulated sequence lengths of k in different batches. The shape is (B+1, ). This parameter is required only when layout_k is TND, and the first value is fixed to 0. Default value: None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_q</td>
      <td>Optional input</td>
      <td>Actual sequence length of q that participates in the operation in each batch. The shape is (B, ). Default value: None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_k</td>
      <td>Optional input</td>
      <td>Actual sequence length of k that participates in the operation in each batch. The shape is (B, ). Default value: None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_residual_k</td>
      <td>Optional input</td>
      <td>Remainder of the sequence length of k before compression divided by cmp_ratio in each batch. This parameter is used together with cmp_ratio to restore the length before compression. The shape is (B, ). This parameter is required when cmp_ratio is not 1 and mask_mode is 3. Default value: None.</td>
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
      <td>Number of k token indexes retained for each q token. Currently, [1, 8192] is supported.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quant_mode</td>
      <td>Attribute</td>
      <td>Quantization mode. Currently, 1/2/3/4/5 is supported. 1 indicates qk: fp8(e4m3) per-token-head, scale: fp32; 2 indicates qk: int8 per-token-head, scale: fp16, w: fp16; 3 indicates qk: mxfp8(e4m3), scale: fp8(e8m0); 4 indicates qk: hif8 per-tensor, scale: fp32; 5 indicates qk: mxfp4(e2m1), scale: fp8(e8m0).</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>batch_size</td>
      <td>Optional attribute</td>
      <td>Number of batches.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_q</td>
      <td>Optional attribute</td>
      <td>Maximum sequence length of q. -1 indicates any possible length.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_k</td>
      <td>Optional attribute</td>
      <td>Maximum sequence length of k. -1 indicates any possible length.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_q</td>
      <td>Optional attribute</td>
      <td>Layout format of q. BSND and TND are supported.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_k</td>
      <td>Optional attribute</td>
      <td>Layout format of k. BSND, TND, and PA_BBND are supported.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mask_mode</td>
      <td>Optional attribute</td>
      <td>Sparse mode. 0 indicates No mask, and 3 indicates the rightDownCausal mode.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_ratio</td>
      <td>Optional attribute</td>
      <td>Compression ratio of k. The value range is [1, 128], and 1 indicates no compression.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadata</td>
      <td>Output</td>
      <td>Output of the load balancing result. The shape is fixed to (1024, )</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

  <ul>
    <li><term>Atlas A3 products</term>: num_heads_q supports only 64; quant_mode = 1/3/4/5 is not supported; topk supports only [1, 2048]; layout_k = BSND/TND is not supported; not every value in [1, 128] is supported for cmp_ratio; only cmp_ratio = 1/2/4/8/16/32/64/128 is supported.</li>
    <li><term>Atlas A2 products</term>: num_heads_q supports only 64; quant_mode = 1/3/4/5 is not supported; topk supports only [1, 2048]; layout_k = BSND/TND is not supported; not every value in [1, 128] is supported for cmp_ratio; only cmp_ratio = 1/2/4/8/16/32/64/128 is supported.</li>
  </ul>

## Constraints

- The QuantLightningIndexerV2Metadata operator must be used together with the QuantLightningIndexerV2 operator.
- B (Batch) indicates the batch size of input samples. q and k are input parameters of the accompanying QuantLightningIndexerV2 operator. S1 indicates the size of the S axis in the q shape when layout_q=BSND, and S2 indicates the size of the S axis in the k shape when layout_k=BSND.
- The values of cu_seqlens_q and cu_seqlens_k must be the cumulative number of valid tokens in the current batch and all preceding batches. The first element is fixed to 0, and the value of each subsequent element must be greater than or equal to that of the previous element.
- The values of seqused_q and seqused_k indicate the number of valid tokens in each batch.
- cmp_residual_k must satisfy cmp_residual_k[i] < cmp_ratio.
- For details about the mask mode indicated by mask_mode, see [sparse_mode parameter description](../../docs/zh/context/sparse_mode_introduction_en.md).
- In non-PA scenarios, layout_q and layout_k must be the same.
- layout_q=BSND scenario
  - max_seqlen_q must be set to S1.
- layout_k=BSND scenario
  - max_seqlen_k must be set to S2.
- layout_q=TND scenario
  - cu_seqlens_q must be passed in.
- layout_k=TND scenario
  - cu_seqlens_k must be passed in.
- layout_k=PA_BBND scenario
  - seqused_k must be passed in.
- Batch value rules
  - When layout_q is BSND, the batch is preferentially derived from the shape of seqused_q; if seqused_q is not passed in, the number of batches is obtained from batch_size.
  - When layout_q is TND, the batch is preferentially derived from the shape of seqused_q; if seqused_q is not passed in, the batch is derived from the shape of cu_seqlens_q.
- q Seqlen rules
  - When layout_q is BSND, the seqlen is preferentially obtained from the elements in seqused_q; if seqused_q is not passed in, the seqlen is obtained from max_seqlen_q.
  - When layout_q is TND, the seqlen is preferentially obtained from the elements in seqused_q; if seqused_q is not passed in, the seqlen is obtained from the elements in cu_seqlens_q.
- k Seqlen rules
  - When layout_k is BSND, the seqlen is preferentially obtained from the elements in seqused_k; if seqused_k is not passed in, the seqlen is obtained from max_seqlen_k.
  - When layout_k is TND, the seqlen is preferentially obtained from the elements in seqused_k; if seqused_k is not passed in, the seqlen is obtained from the elements in cu_seqlens_k.

## Invocation

| Invocation Method  | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn API | [test_aclnn_quant_lightning_indexer_v2_metadata](./examples/test_aclnn_quant_lightning_indexer_v2_metadata.cpp) | Calls the QuantLightningIndexerV2Metadata operator through the [aclnnQuantLightningIndexerV2Metadata](docs/aclnnQuantLightningIndexerV2Metadata_en.md) API |
| PyTorch API | [test_torch_quant_lightning_indexer_v2_metadata](./examples/test_torch_quant_lightning_indexer_v2_metadata.py) | Calls the QuantLightningIndexerV2Metadata operator through the [quant_lightning_indexer_metadata](../quant_lightning_indexer_v2/docs/torchapi_quant_lightning_indexer_en.md) API |
