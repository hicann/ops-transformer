# QuantSparseFlashMlaMetadata

English | [简体中文](./README.md)

## Product Support

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
|<term>Ascend 950PR&950DT products</term>|      √     |
|<term>Atlas A3 products</term>|      ×     |
|<term>Atlas A2 products</term>|      ×     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function Description

- Operator function: `QuantSparseFlashMlaMetadata` is a preceding operator of the `QuantSparseFlashMla` operator, and is used to generate a load-balanced task partition scheme for subsequent Attention computation. This operator does not perform the actual Attention computation. Instead, it computes the start and end ranges of the Attention computation that each AI Core should process on the AI CPU based on the input parameters, so as to maximize the utilization of computing resources and avoid load imbalance among Cores.

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
      <td>Indicates the valid sequence length of Query in each batch. The shape is (B+1, ). This parameter must be passed in only for the scenario where layout_q is TND. Default value: None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_ori_kv</td>
      <td>Optional input</td>
      <td>Indicates the valid sequence length of ori_kv in each batch. The shape is (B+1, ). This parameter must be passed in only for the scenario where layout_kv is TND. Default value: None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_cmp_kv</td>
      <td>Optional input</td>
      <td>Indicates the valid sequence length of cmp_kv in each batch. The shape is (B+1, ). This parameter must be passed in only for the scenario where layout_kv is TND. Default value: None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_q</td>
      <td>Optional input</td>
      <td>Indicates the sequence length of Query that actually participates in the computation in each batch. The shape is (B, ). Default value: None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_ori_kv</td>
      <td>Optional input</td>
      <td>Indicates the sequence length of ori_kv that actually participates in the computation in each batch. The shape is (B, ). Default value: None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_cmp_kv</td>
      <td>Optional input</td>
      <td>Indicates the sequence length of cmp_kv that actually participates in the computation in each batch. The shape is (B, ). Default value: None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_residual_kv</td>
      <td>Optional input</td>
      <td>Indicates the remainder of the compressed sequence length of cmp_kv in each batch. It works with cmp_ratio to implement the mask and load computation for the cmp_kv part. This parameter must be passed in when cmp_mask_mode=3 and cmp_ratio≠1. The shape is (B, ). Default value: None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ori_topk_length</td>
      <td>Optional input</td>
      <td>Reserved parameter, which does not take effect currently. Default value: None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_topk_length</td>
      <td>Optional input</td>
      <td>Reserved parameter, which does not take effect currently. Default value: None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num_heads_q</td>
      <td>Attribute</td>
      <td>Indicates the number of heads of Query. The value ranges from 1 to 128.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_heads_kv</td>
      <td>Attribute</td>
      <td>Indicates the number of heads corresponding to Key and Value. Currently, only 1 is supported.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>head_dim</td>
      <td>Attribute</td>
      <td>Indicates the dimension of the attention head. Currently, only 512 is supported.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quant_mode</td>
      <td>Attribute</td>
      <td>Indicates the quantization mode. The value 1 indicates that Q, K, and V use per-tensor quantization, and the data types of Q, K, and V are HIFLOAT8.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>batch_size</td>
      <td>Optional attribute</td>
      <td>Indicates the number of batches.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_q</td>
      <td>Optional attribute</td>
      <td>Indicates the maximum sequence length of Query.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_ori_kv</td>
      <td>Optional attribute</td>
      <td>Indicates the maximum sequence length of ori_kv.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_cmp_kv</td>
      <td>Optional attribute</td>
      <td>Indicates the maximum sequence length of cmp_kv.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_topk</td>
      <td>Optional attribute</td>
      <td>Indicates the number of key sparse tokens selected from ori_kv. The value 0 indicates a non-sparse scenario.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_topk</td>
      <td>Optional attribute</td>
      <td>Indicates the number of key sparse tokens selected from cmp_kv. The value 0 indicates a non-sparse scenario.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_ratio</td>
      <td>Optional attribute</td>
      <td>Indicates the compression ratio of cmp_kv.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_mask_mode</td>
      <td>Optional attribute</td>
      <td>Indicates the mask mode for the computation between q and ori_kv.<br/>0: No Mask.<br/>3: RightDownCausal mode.<br/>4: Band mode.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_mask_mode</td>
      <td>Optional attribute</td>
      <td>Indicates the mask mode for the computation between q and cmp_kv.<br/>0: No Mask.<br/>3: RightDownCausal mode.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_win_left</td>
      <td>Optional attribute</td>
      <td>Indicates the number of past tokens that q computes against in the computation between q and ori_kv. The value can be -1 or a non-negative number, where -1 indicates that the window is not limited.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_win_right</td>
      <td>Optional attribute</td>
      <td>Indicates the number of future tokens that q computes against in the computation between q and ori_kv. The value can be -1 or a non-negative number, where -1 indicates that the window is not limited.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_q</td>
      <td>Optional attribute</td>
      <td>Indicates the layout format of Query. The supported values are BSND and TND.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_kv</td>
      <td>Optional attribute</td>
      <td>Indicates the layout format of Key. The supported values are BSND, TND, and PA_BBND.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>has_ori_kv</td>
      <td>Optional attribute</td>
      <td>Indicates whether ori_kv exists.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>has_cmp_kv</td>
      <td>Optional attribute</td>
      <td>Indicates whether cmp_kv exists.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadata</td>
      <td>Output</td>
      <td>Indicates the output of the load balancing result. The shape is fixed to [2048].</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

## Constraints

- The QuantSparseFlashMlaMetadata operator must be used together with the QuantSparseFlashMla operator.
- B (batch) indicates the batch size of input samples.
- The values of cu_seqlens_q, cu_seqlens_ori_kv, and cu_seqlens_cmp_kv must be the cumulative number of valid tokens in the current batch and all preceding batches, and the value of each element must be greater than or equal to that of the previous element.
- The values of seqused_q, seqused_ori_kv, and seqused_cmp_kv must indicate the number of valid tokens in each batch.
- The parameter cmp_residual_kv must satisfy cmp_residual_kv[i] < cmp_ratio.
- For details about the mask modes indicated by ori_mask_mode and cmp_mask_mode, see [sparse_mode parameter description](../../docs/zh/context/sparse_mode_introduction_en.md).

## Calling Instructions

| Calling Method | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn API | [test_aclnn_quant_sparse_flash_mla_metadata](./examples/test_aclnn_quant_sparse_flash_mla_metadata.cpp) | Calls the QuantSparseFlashMlaMetadata operator through the [aclnnQuantSparseFlashMlaMetadata](docs/aclnnQuantSparseFlashMlaMetadata_en.md) API |
| PyTorch API | [test_torch_quant_sparse_flash_mla_metadata](./examples/test_torch_quant_sparse_flash_mla_metadata.py) | Calls the QuantSparseFlashMlaMetadata operator through the [quant_sparse_flash_mla_metadata](../quant_sparse_flash_mla/docs/torchapi_quant_sparse_flash_mla_en.md) API |
