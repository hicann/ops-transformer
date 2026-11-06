# MixedQuantSparseFlashMlaMetadata

English | [简体中文](./README.md)

## Product Support

| Product                                                         |  Supported |
| :----------------------------------------------------------- | :------: |
|<term>Ascend 950PR&950DT products</term>|      √     |
|<term>Atlas A3 products</term>|      √     |
|<term>Atlas A2 products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function Description

- Operator function: `MixedQuantSparseFlashMlaMetadata` is the preceding operator of the `MixedQuantSparseFlashMla` operator, and generates a load-balanced task partitioning scheme for the subsequent Attention computation. This operator does not perform the actual Attention computation. Instead, it computes on the AI CPU the start and end ranges of the Attention computation that each AI Core should process based on the input parameters, maximizing the utilization of computing resources and avoiding load imbalance between Cores.

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
      <td>Represents the valid Sequence Length of q in different batches, with shape (B+1, ). The default value is None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_ori_kv</td>
      <td>Optional input</td>
      <td>Represents the valid Sequence Length of ori_kv in different batches, with shape (B+1, ). The default value is None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_cmp_kv</td>
      <td>Optional input</td>
      <td>Represents the valid Sequence Length of cmp_kv in different batches, with shape (B+1, ). The default value is None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_q</td>
      <td>Optional input</td>
      <td>Represents the Sequence Length of q that actually participates in the computation in each batch, with shape (B, ). The default value is None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_ori_kv</td>
      <td>Optional input</td>
      <td>Represents the Sequence Length of ori_kv that actually participates in the computation in each batch, with shape (B, ). The default value is None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_cmp_kv</td>
      <td>Optional input</td>
      <td>Represents the Sequence Length of cmp_kv that actually participates in the computation in each batch, with shape (B, ). The default value is None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_residual_kv</td>
      <td>Optional input</td>
      <td>Represents the remainder of the compressed Sequence Length of cmp_kv in different batches, and works with cmp_ratio to implement the mask and load computation of the cmp_kv part. It is mandatory when cmp_mask_mode=3 and cmp_ratio≠1, with shape (B, ). The default value is None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ori_topk_length</td>
      <td>Optional input</td>
      <td>Represents the number of key sparse tokens in the ori_kv part corresponding to each q token, with shape (B, S1, N2) or (T1, N2). The default value is None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_topk_length</td>
      <td>Optional input</td>
      <td>Represents the number of key sparse tokens in the cmp_kv part corresponding to each q token, with shape (B, S1, N2) or (T1, N2). The default value is None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num_heads_q</td>
      <td>Attribute</td>
      <td>Represents the number of heads of q. Currently, [1, 128] is supported.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_heads_kv</td>
      <td>Attribute</td>
      <td>Represents the number of heads corresponding to ori_kv and cmp_kv. Currently, only 1 is supported.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>head_dim</td>
      <td>Attribute</td>
      <td>Represents the dimension of the attention head. Currently, only 512 is supported.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quant_mode</td>
      <td>Attribute</td>
      <td>Represents the quantization mode. 1 indicates the BF16 scale quantization layout; 2 indicates the FLOAT8_E8M0 scale quantization layout; 3 indicates the metadata for TurboQuant TQ4. The Metadata API accepts 1, 2, and 3. For platform-specific constraints, see Constraints.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>batch_size</td>
      <td>Optional attribute</td>
      <td>Represents the number of batches.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_q</td>
      <td>Optional attribute</td>
      <td>Represents the maximum Sequence Length of q.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_ori_kv</td>
      <td>Optional attribute</td>
      <td>Represents the maximum Sequence Length of ori_kv.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_cmp_kv</td>
      <td>Optional attribute</td>
      <td>Represents the maximum Sequence Length of cmp_kv.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_topk</td>
      <td>Optional attribute</td>
      <td>Represents the number of key sparse tokens selected from ori_kv. 0 indicates a non-sparse scenario.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_topk</td>
      <td>Optional attribute</td>
      <td>Represents the number of key sparse tokens selected from cmp_kv. 0 indicates a non-sparse scenario.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rope_head_dim</td>
      <td>Optional attribute</td>
      <td>Represents the dimension of the rope head. Currently, only 64 is supported.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_ratio</td>
      <td>Optional attribute</td>
      <td>Represents the compression ratio of cmp_kv. Currently, [1, 128] is supported.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_mask_mode</td>
      <td>Optional attribute</td>
      <td>Represents the mask mode for the computation between q and ori_kv. Currently, 0, 3, and 4 are supported. 0 indicates No Mask, 3 indicates the RightDownCausal mode, and 4 indicates the sliding window mode.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_mask_mode</td>
      <td>Optional attribute</td>
      <td>Represents the mask mode for the computation between q and cmp_kv. Currently, 0 and 3 are supported. 0 indicates No Mask, and 3 indicates the RightDownCausal mode.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_win_left</td>
      <td>Optional attribute</td>
      <td>Represents the number of past tokens computed for q in the computation between q and ori_kv. -1 indicates infinity.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_win_right</td>
      <td>Optional attribute</td>
      <td>Represents the number of future tokens computed for q in the computation between q and ori_kv. -1 indicates infinity.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_q</td>
      <td>Optional attribute</td>
      <td>Represents the layout format of q. BSND and TND are supported.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_kv</td>
      <td>Optional attribute</td>
      <td>Represents the layout format of ori_kv and cmp_kv. BSND, TND, and PA_BBND are supported.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>has_ori_kv</td>
      <td>Optional attribute</td>
      <td>Identifies whether ori_kv exists.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>has_cmp_kv</td>
      <td>Optional attribute</td>
      <td>Identifies whether cmp_kv exists.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadata</td>
      <td>Output</td>
      <td>Represents the load balancing result output, with shape fixed to (2048, )</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

## Constraints

- `quant_mode` supports 1, 2, and 3. The quantization modes supported by specific products are as follows.

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: When used together with the `MixedQuantSparseFlashMla` operator, only `quant_mode=1/2` is supported.
<!-- end id1 -->

<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: When used together with the `MixedQuantSparseFlashMla` operator, only `quant_mode=3` is supported.
<!-- end id2 -->

<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: When used together with the `MixedQuantSparseFlashMla` operator, only `quant_mode=3` is supported.
<!-- end id3 -->
- The MixedQuantSparseFlashMlaMetadata operator must be used together with the MixedQuantSparseFlashMla operator.
- B (Batch) indicates the batch size of input samples. q, ori_kv, and cmp_kv are input parameters of the accompanying MixedQuantSparseFlashMla operator. S1 indicates the size of the S axis in the q shape when layout_q=BSND, T1 indicates the size of the T axis in the q shape when layout_q=TND, S2 indicates the size of the S axis in the ori_kv shape when layout_kv=BSND, S3 indicates the size of the S axis in the cmp_kv shape when layout_kv=BSND, and N2 indicates the size of the N axis in the ori_kv and cmp_kv shapes.
- The values of cu_seqlens_q, cu_seqlens_ori_kv, and cu_seqlens_cmp_kv must be the cumulative number of valid tokens in the current batch and previous batches. The first element is fixed to 0, and the value of each subsequent element must be greater than or equal to the value of the previous element.
- The values of seqused_q, seqused_ori_kv, and seqused_cmp_kv must indicate the number of valid tokens in each batch.
- cmp_residual_kv must satisfy cmp_residual_kv[i] < cmp_ratio.
- For details about the mask modes represented by ori_mask_mode and cmp_mask_mode, see [sparse_mode parameter description](../../docs/zh/context/sparse_mode_introduction_en.md).
- In non-PA scenarios, layout_q and layout_kv must be the same.
- When has_ori_kv is true, ori_topk greater than 0 indicates that the ori_kv part is sparse, and ori_topk equal to 0 indicates that the ori_kv part is non-sparse.
- When has_cmp_kv is true, cmp_topk greater than 0 indicates that the cmp_kv part is sparse, and cmp_topk equal to 0 indicates that the cmp_kv part is non-sparse.
- When has_ori_kv is true, ori_topk is not 0, and ori_mask_mode is 0, ori_topk_length must be passed in. In this case, the minimum value between the ori_mask_mode rule and the elements of ori_topk_length is used as the valid seqlen of ori_kv corresponding to the current q token. In other ori_kv sparse scenarios, the minimum value between the ori_mask_mode rule and ori_topk is used as the valid seqlen of ori_kv corresponding to the current q token.
- When has_cmp_kv is true, cmp_topk is not 0, and cmp_mask_mode is 0, cmp_topk_length must be passed in. In this case, the minimum value between the cmp_mask_mode rule and the elements of cmp_topk_length is used as the valid seqlen of cmp_kv corresponding to the current q token. In other cmp_kv sparse scenarios, the minimum value between the cmp_mask_mode rule and cmp_topk is used as the valid seqlen of cmp_kv corresponding to the current q token.
- layout_q=BSND scenario
  - max_seqlen_q must be passed in with the value of S1.
- layout_kv=BSND scenario
  - When has_ori_kv is true, max_seqlen_ori_kv must be passed in with the value of S2.
  - When has_cmp_kv is true, max_seqlen_cmp_kv must be passed in with the value of S3.
- layout_q=TND scenario
  - cu_seqlens_q must be passed in.
- layout_kv=TND scenario
  - When has_ori_kv is true, cu_seqlens_ori_kv must be passed in.
  - When has_cmp_kv is true, cu_seqlens_cmp_kv must be passed in.
- layout_kv=PA_BBND scenario
  - When has_ori_kv is true, ori_topk is not 0, and ori_mask_mode is 0 (the scenario where ori_topk_length is mandatory), seqused_ori_kv is optional. In other scenarios, seqused_ori_kv is mandatory.
  - When has_cmp_kv is true, cmp_topk is not 0, and cmp_mask_mode is 0 (the scenario where cmp_topk_length is mandatory), seqused_cmp_kv is optional. In other scenarios, seqused_cmp_kv is mandatory.
- Batch value rules
  - When layout_q is BSND, the batch is derived preferentially from the shape of seqused_q. If seqused_q is not passed in, the batch count is obtained from batch_size.
  - When layout_q is TND, the batch is derived preferentially from the shape of seqused_q. If seqused_q is not passed in, the batch is derived from the shape of cu_seqlens_q.
- q Seqlen value rules
  - When layout_q is BSND, the seqlen is obtained preferentially from the elements in seqused_q. If seqused_q is not passed in, the seqlen is obtained from max_seqlen_q.
  - When layout_q is TND, the seqlen is obtained preferentially from the elements in seqused_q. If seqused_q is not passed in, the seqlen is obtained from the elements in cu_seqlens_q.
- ori_kv Seqlen value rules
  - When layout_kv is BSND, the seqlen is obtained preferentially from the elements in seqused_ori_kv. If seqused_ori_kv is not passed in, the seqlen is obtained from max_seqlen_ori_kv.
  - When layout_kv is TND, the seqlen is obtained preferentially from the elements in seqused_ori_kv. If seqused_ori_kv is not passed in, the seqlen is obtained from the elements in cu_seqlens_ori_kv.
  - When layout_kv is PA_BBND, the seqlen is obtained preferentially from the elements in seqused_ori_kv. If seqused_ori_kv is not passed in, the seqlen is obtained from ori_topk_length.
- cmp_kv Seqlen value rules
  - When layout_kv is BSND, the seqlen is obtained preferentially from the elements in seqused_cmp_kv. If seqused_cmp_kv is not passed in, the seqlen is obtained from max_seqlen_cmp_kv.
  - When layout_kv is TND, the seqlen is obtained preferentially from the elements in seqused_cmp_kv. If seqused_cmp_kv is not passed in, the seqlen is obtained from the elements in cu_seqlens_cmp_kv.
  - When layout_kv is PA_BBND, the seqlen is obtained preferentially from the elements in seqused_cmp_kv. If seqused_cmp_kv is not passed in, the seqlen is obtained from cmp_topk_length.

## Calling Instructions

| Calling Method  | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------- | ------------------------------------------------------------- |
| aclnn API | [test_aclnn_mixed_quant_sparse_flash_mla_metadata](./examples/test_aclnn_mixed_quant_sparse_flash_mla_metadata.cpp) | Calls the MixedQuantSparseFlashMlaMetadata operator through the [aclnnMixedQuantSparseFlashMlaMetadata](docs/aclnnMixedQuantSparseFlashMlaMetadata_en.md) API |
| PyTorch API | [test_torch_mixed_quant_sparse_flash_mla_metadata](./examples/test_torch_mixed_quant_sparse_flash_mla_metadata.py) | Calls the MixedQuantSparseFlashMlaMetadata operator through the [mixed_quant_sparse_flash_mla_metadata](../mixed_quant_sparse_flash_mla/docs/torchapi_mixed_quant_sparse_flash_mla_en.md) API |
