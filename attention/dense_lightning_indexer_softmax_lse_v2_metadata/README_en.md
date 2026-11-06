# DenseLightningIndexerSoftmaxLseV2Metadata

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
|<term>Ascend 950PR&950DT products</term>|      √     |
|<term>Atlas A3 products</term>|      ×     |
|<term>Atlas A2 products</term>|      ×     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function Description

- Operator function: The `DenseLightningIndexerSoftmaxLseV2Metadata` operator computes and outputs the core splitting metadata based on the input shape, layout, mask, and compression ratio information of the `DenseLightningIndexerSoftmaxLseV2` operator. The metadata is used by the subsequent `DenseLightningIndexerSoftmaxLseV2` operator.

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
      <td>Optional input</td>
      <td>Represents the cumulative sequence length of Query in different batches. The shape is (B+1, ). Mandatory only when layout_q is TND. The first value is fixed to 0.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_k</td>
      <td>Optional input</td>
      <td>Represents the cumulative sequence length of Key in different batches. The shape is (B+1, ). Mandatory only when layout_k is TND. The first value is fixed to 0.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_q</td>
      <td>Optional input</td>
      <td>Represents the sequence length of Query that actually participates in the computation in different batches. The shape is (B, ).</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_k</td>
      <td>Optional input</td>
      <td>Represents the sequence length of Key that actually participates in the computation in different batches. The shape is (B, ).</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_residual_k</td>
      <td>Optional input</td>
      <td>Represents the residual of the Key sequence length related to cmp_ratio in different batches. The shape is (B, ). Mandatory when mask_mode is 3 and cmp_ratio is greater than 1. Each value must be less than cmp_ratio.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>batch_size</td>
      <td>Optional attribute</td>
      <td>Represents the number of batches. Supports non-negative values. In TND scenarios, you can set this parameter to 0 and infer the value from cu_seqlens_q.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_q</td>
      <td>Optional attribute</td>
      <td>Represents the maximum sequence length of Query. Supports non-negative values. In BSND scenarios, the value must be a positive number.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_k</td>
      <td>Optional attribute</td>
      <td>Represents the maximum sequence length of Key. Supports non-negative values. In BSND scenarios, the value must be a positive number.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_heads_q</td>
      <td>Attribute</td>
      <td>Represents the number of Query heads. Value range: [1, 128]. It must be divisible by num_heads_k.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_heads_k</td>
      <td>Attribute</td>
      <td>Represents the number of Key heads. Currently, only 1 is supported.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>head_dim</td>
      <td>Attribute</td>
      <td>Represents the head dimension of q/k. Currently, only 128 is supported.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_q</td>
      <td>Optional attribute</td>
      <td>Represents the layout format of Query. Supports BSND and TND. The default value is BSND.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_k</td>
      <td>Optional attribute</td>
      <td>Represents the layout format of Key. Supports BSND and TND. The default value is BSND. It must be consistent with layout_q.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mask_mode</td>
      <td>Attribute</td>
      <td>Represents the sparse mask mode. 0 indicates No mask, and 3 indicates Causal.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_ratio</td>
      <td>Attribute</td>
      <td>Represents the compression ratio of Keys. Value range: [1, 128].</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadata</td>
      <td>Output</td>
      <td>Represents the load balancing result output. The shape is fixed to [64].</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

## Constraints

- The DenseLightningIndexerSoftmaxLseV2Metadata operator must be used together with the DenseLightningIndexerSoftmaxLseV2 operator.
- B (Batch) indicates the batch size of input samples.
- In BSND scenarios, the batch_size, max_seqlen_q, and max_seqlen_k parameters must be passed to obtain shape information.
- In TND scenarios, the cu_seqlens_q and cu_seqlens_k parameters must be passed to obtain correct shape information. When batch_size is 0, the batch is inferred from the shape of cu_seqlens_q.
- The values of cu_seqlens_q and cu_seqlens_k must be the cumulative numbers of valid tokens in the current batch and all preceding batches. Each element must be greater than or equal to the previous element.
- The values of seqused_q and seqused_k must indicate the numbers of valid tokens in each batch.
- cmp_residual_k must satisfy cmp_residual_k[i] < cmp_ratio.
- Currently, mask_mode supports only 0 and 3. For details about the mask modes, see [sparse_mode Parameter Description](../../docs/zh/context/sparse_mode_introduction_en.md).
- layout_q must be BSND or TND, and layout_k must be consistent with layout_q.
- The value range of num_heads_q is [1, 128], and it must be divisible by num_heads_k. Currently, num_heads_k supports only 1, and head_dim supports only 128.

## Troubleshooting

- For common execution problems or exceptions of AI CPU operator kernels, refer to "[Fault Cases > Operator Execution Problems > AI CPU Operator Kernel Execution Errors](https://www.hiascend.com/document/detail/zh/canncommercial/latest/maintenref/troubleshooting/troubleshooting_0151.html)" in the Troubleshooting guide.

## Calling Description

| Calling Method  | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn API | [test_aclnn_dense_lightning_indexer_softmax_lse_v2_metadata](./examples/test_aclnn_dense_lightning_indexer_softmax_lse_v2_metadata.cpp) | Calls the DenseLightningIndexerSoftmaxLseV2Metadata operator through the [aclnnDenseLightningIndexerSoftmaxLseV2Metadata](docs/aclnnDenseLightningIndexerSoftmaxLseV2Metadata_en.md) API. |
