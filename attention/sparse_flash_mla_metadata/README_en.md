# SparseFlashMlaMetadata

English | [简体中文](./README.md)

## Product Support

| Product                                                      | Supported |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR&950DT products</term>                        | √  |
|<term>Atlas A3 products</term>        | √  |
|<term>Atlas A2 products</term>        | √  |
|<term>Atlas 200I/500 A2 inference products</term>                    | ×  |
|<term>Atlas inference products</term>                                | ×  |
|<term>Atlas training products</term>                                | ×  |

## Description

- Operator function: `SparseFlashMlaMetadata` is a preceding operator of the `SparseFlashMla` operator. It generates a load-balanced task partitioning scheme for the subsequent Attention computation. This operator does not perform the actual Attention computation. Instead, based on the input parameters, it computes on the AI CPU the start and end ranges of the Attention computation to be processed by each AI Core, maximizing the utilization of computing resources and preventing load imbalance across Cores.

- Scenario abbreviations: SWA (Sliding Window Attention), CSA (Compressed Sparse Attention), and HCA (Heavily Compressed Attention).

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1576px">
  <colgroup>
  <col style="width: 220px">
  <col style="width: 150px">
  <col style="width: 700px">
  <col style="width: 180px">
  <col style="width: 100px">
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
      <td>num_heads_q</td>
      <td>Attribute</td>
      <td>Number of heads of `q`. The value range is [1, 128].</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_heads_kv</td>
      <td>Attribute</td>
      <td>Number of heads of `ori_kv` and `cmp_kv`. Only 1 is supported.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>head_dim</td>
      <td>Attribute</td>
      <td>Dimension of the attention head. Only 512 is supported.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cu_seqlens_q</td>
      <td>Optional input</td>
      <td>Cumulative sequence lengths of `q` in different batches under the TND layout. The shape is (B+1, ). The default value is None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_ori_kv</td>
      <td>Optional input</td>
      <td>Cumulative sequence lengths of `ori_kv` in different batches under the TND layout. The shape is (B+1, ). The default value is None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_cmp_kv</td>
      <td>Optional input</td>
      <td>Cumulative sequence lengths of `cmp_kv` in different batches under the TND layout. The shape is (B+1, ). The default value is None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_q</td>
      <td>Optional input</td>
      <td>Number of tokens of `q` that actually participate in the computation in each batch. The shape is (B, ). The default value is None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_ori_kv</td>
      <td>Optional input</td>
      <td>Number of tokens of `ori_kv` that actually participate in the computation in each batch. The shape is (B, ). The default value is None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_cmp_kv</td>
      <td>Optional input</td>
      <td>Number of tokens of `cmp_kv` that actually participate in the computation in each batch. The shape is (B, ). The default value is None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_residual_kv</td>
      <td>Optional input</td>
      <td>Remainder of the compressed KV, used to restore the KV length before compression on the cmp side for the mask. The shape is (B, ). The default value is None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ori_topk_length</td>
      <td>Optional input</td>
      <td>In the SWA sparse ori_kv scenario, this parameter indicates the number of key sparse tokens in the ori_kv part corresponding to each q token. This parameter must be passed in. The shape is (B, S1, N2) or (T1, N2). The default value is None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_topk_length</td>
      <td>Optional input</td>
      <td>Number of key sparse tokens in the cmp_kv part corresponding to each q token. The shape is (B, S1, N2) or (T1, N2). The default value is None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>batch_size</td>
      <td>Optional attribute</td>
      <td>Batch size of the input samples. If 0 is passed in, the value is inferred by the API.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_q</td>
      <td>Optional attribute</td>
      <td>Maximum number of valid tokens of `q` across all batches. If 0 is passed in, the value is inferred by the API.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_ori_kv</td>
      <td>Optional attribute</td>
      <td>Maximum number of valid tokens of `ori_kv` across all batches. If 0 is passed in, the value is inferred by the API.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_cmp_kv</td>
      <td>Optional attribute</td>
      <td>Maximum number of valid tokens of `cmp_kv` across all batches. If 0 is passed in, the value is inferred by the API.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_topk</td>
      <td>Optional attribute</td>
      <td>Number of key sparse tokens selected from `ori_kv`. In the SWA sparse ori_kv scenario, this value is K, the last dimension of `ori_sparse_indices` of the main operator, and must be greater than 0. In other scenarios, the value is 0.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_topk</td>
      <td>Optional attribute</td>
      <td>Number of key sparse tokens selected from `cmp_kv`.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_ratio</td>
      <td>Optional attribute</td>
      <td>Compression ratio of `cmp_kv` relative to the KV length before compression, used to restore the KV length before compression on the cmp side for the mask. This parameter is not involved in the compressed KV computation when only `ori_kv` is passed in. The value range is [1, 128].</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_mask_mode</td>
      <td>Optional attribute</td>
      <td>Mask mode for the computation of `q` and `ori_kv`.<br/>0: No Mask.<br/>3: RightDownCausal mode.<br/>4: Band mode.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_mask_mode</td>
      <td>Optional attribute</td>
      <td>Mask mode for the computation of `q` and `cmp_kv`.<br/>0: No Mask.<br/>3: RightDownCausal mode.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_win_left</td>
      <td>Optional attribute</td>
      <td>Number of past tokens computed with `q` in the computation of `q` and `ori_kv`. The value is -1 or a non-negative number, where -1 indicates that the window is not limited.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_win_right</td>
      <td>Optional attribute</td>
      <td>Number of future tokens computed with `q` in the computation of `q` and `ori_kv`. The value is -1 or a non-negative number, where -1 indicates that the window is not limited.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_q</td>
      <td>Optional attribute</td>
      <td>Data layout format of the input `q`. "BSND" and "TND" are supported.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_kv</td>
      <td>Optional attribute</td>
      <td>Data layout format of the inputs `ori_kv` and `cmp_kv`. "BSND", "TND", and "PA_BBND" are supported.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>has_ori_kv</td>
      <td>Optional attribute</td>
      <td>Whether `ori_kv` is passed in to the `SparseFlashMla` main operator.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>has_cmp_kv</td>
      <td>Optional attribute</td>
      <td>Whether `cmp_kv` is passed in to the `SparseFlashMla` main operator.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadata</td>
      <td>Output</td>
      <td>Task partitioning result used by the `SparseFlashMla` main operator. The shape is fixed to (2048, ).</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>
<ul>
  <li><term>Atlas A3 products</term>: For num_heads_q/num_heads_kv, only 1, 2, 4, 8, 16, 32, 64, and 128 are supported, and seqused_q and cmp_topk_length are not supported. In the SWA sparse ori_kv scenario, ori_topk_length, ori_topk greater than 0, and ori_mask_mode set to 0 are supported, and ori_win_left and ori_win_right support non-negative numbers. In other SWA scenarios, ori_topk is 0, ori_mask_mode is 4, ori_win_left is 127, and ori_win_right is 0. cmp_topk supports [0, 8192], and only 3 is supported for cmp_mask_mode. In the SWA scenario, cmp_kv is not passed in and cmp_ratio is not involved in the computation; 1, 2, or 4 is supported in the CSA scenario, and 128 is supported in the HCA scenario.</li>
  <li><term>Atlas A2 products</term>: For num_heads_q/num_heads_kv, only 1, 2, 4, 8, 16, 32, 64, and 128 are supported, and seqused_q and cmp_topk_length are not supported. In the SWA sparse ori_kv scenario, ori_topk_length, ori_topk greater than 0, and ori_mask_mode set to 0 are supported, and ori_win_left and ori_win_right support non-negative numbers. In other SWA scenarios, ori_topk is 0, ori_mask_mode is 4, ori_win_left is 127, and ori_win_right is 0. cmp_topk supports [0, 8192], and only 3 is supported for cmp_mask_mode. In the SWA scenario, cmp_kv is not passed in and cmp_ratio is not involved in the computation; 1, 2, or 4 is supported in the CSA scenario, and 128 is supported in the HCA scenario.</li>
  <li><term>Ascend 950PR&950DT products</term>: Set cmp_ratio to 1 in the SWA scenario. In CSA/HCA scenarios, values from 1 to 128 are supported.</li>
</ul>

## Constraints

- This API can be used in both training and inference scenarios.
- This API supports the aclgraph mode.
- The general specification constraints are as follows:
  - B (Batch) indicates the batch size of the input samples. q, ori_kv, and cmp_kv are input parameters of the paired SparseFlashMla operator. S1 indicates the size of the S axis in the q shape when layout_q=BSND. T1 indicates the size of the T axis in the q shape when layout_q=TND. S2 indicates the size of the S axis in the ori_kv shape when layout_kv=BSND. S3 indicates the size of the S axis in the cmp_kv shape when layout_kv=BSND. N2 indicates the size of the N axis in the ori_kv and cmp_kv shapes.
  - The value of `cu_seqlens_q`, `cu_seqlens_ori_kv`, or `cu_seqlens_cmp_kv` must be the cumulative number of valid tokens in the current batch and preceding batches. The first element is fixed to 0, and each subsequent element must be greater than or equal to the preceding element.
  - The value of `seqused_q`, `seqused_ori_kv`, or `seqused_cmp_kv` must indicate the number of valid tokens in each batch.
  - Only the combinations "BSND"/"BSND", "TND"/"TND", "BSND"/"PA_BBND", and "TND"/"PA_BBND" are supported for `layout_q` and `layout_kv`. In scenarios other than PA_BBND, `layout_q` and `layout_kv` must be the same.
  - `cmp_residual_kv` must satisfy the condition that `cmp_residual_kv`[i] < `cmp_ratio`.
- Constraints on Ascend 950PR&950DT products:
  - When has_ori_kv is true, the ori_kv part is considered sparse if ori_topk is greater than 0, and non-sparse if ori_topk is 0.
  - When has_cmp_kv is true, the cmp_kv part is considered sparse if cmp_topk is greater than 0, and non-sparse if cmp_topk is 0.
  - When has_ori_kv is true, ori_topk is not 0, and ori_mask_mode is 0, ori_topk_length must be passed in. In this case, the valid seqlen of ori_kv corresponding to the current q token is the minimum of the value determined by the ori_mask_mode rule and the elements of ori_topk_length. In other ori_kv sparse scenarios, the valid seqlen of ori_kv corresponding to the current q token is the minimum of the value determined by the ori_mask_mode rule and ori_topk.
  - When has_cmp_kv is true, cmp_topk is not 0, and cmp_mask_mode is 0, cmp_topk_length must be passed in. In this case, the valid seqlen of cmp_kv corresponding to the current q token is the minimum of the value determined by the cmp_mask_mode rule and the elements of cmp_topk_length. In other cmp_kv sparse scenarios, the valid seqlen of cmp_kv corresponding to the current q token is the minimum of the value determined by the cmp_mask_mode rule and cmp_topk.
  - layout_q=BSND scenario
    - max_seqlen_q must be set to S1.
  - layout_kv=BSND scenario
    - When has_ori_kv is true, max_seqlen_ori_kv must be set to S2.
    - When has_cmp_kv is true, max_seqlen_cmp_kv must be set to S3.
  - layout_q=TND scenario
    - cu_seqlens_q must be passed in.
  - layout_kv=TND scenario
    - When has_ori_kv is true, cu_seqlens_ori_kv must be passed in.
    - When has_cmp_kv is true, cu_seqlens_cmp_kv must be passed in.
  - layout_kv=PA_BBND scenario
    - When has_ori_kv is true, ori_topk is not 0, and ori_mask_mode is 0 (scenario where ori_topk_length must be passed in), seqused_ori_kv is optional. In other scenarios, seqused_ori_kv must be passed in.
    - When has_cmp_kv is true, cmp_topk is not 0, and cmp_mask_mode is 0 (scenario where cmp_topk_length must be passed in), seqused_cmp_kv is optional. In other scenarios, seqused_cmp_kv must be passed in.
  - Batch value rules
    - When layout_q is BSND, the batch is first inferred from the shape of seqused_q. If seqused_q is not passed in, the number of batches is obtained from batch_size.
    - When layout_q is TND, the batch is first inferred from the shape of seqused_q. If seqused_q is not passed in, the batch is inferred from the shape of cu_seqlens_q.
  - q Seqlen value rules
    - When layout_q is BSND, the seqlen is first obtained from the elements of seqused_q. If seqused_q is not passed in, the seqlen is obtained from max_seqlen_q.
    - When layout_q is TND, the seqlen is first obtained from the elements of seqused_q. If seqused_q is not passed in, the seqlen is obtained from the elements of cu_seqlens_q.
  - ori_kv Seqlen value rules
    - When layout_kv is BSND, the seqlen is first obtained from the elements of seqused_ori_kv. If seqused_ori_kv is not passed in, the seqlen is obtained from max_seqlen_ori_kv.
    - When layout_kv is TND, the seqlen is first obtained from the elements of seqused_ori_kv. If seqused_ori_kv is not passed in, the seqlen is obtained from the elements of cu_seqlens_ori_kv.
    - When layout_kv is PA_BBND, the seqlen is first obtained from the elements of seqused_ori_kv. If seqused_ori_kv is not passed in, the seqlen is obtained from ori_topk_length.
  - cmp_kv Seqlen value rules
    - When layout_kv is BSND, the seqlen is first obtained from the elements of seqused_cmp_kv. If seqused_cmp_kv is not passed in, the seqlen is obtained from max_seqlen_cmp_kv.
    - When layout_kv is TND, the seqlen is first obtained from the elements of seqused_cmp_kv. If seqused_cmp_kv is not passed in, the seqlen is obtained from the elements of cu_seqlens_cmp_kv.
    - When layout_kv is PA_BBND, the seqlen is first obtained from the elements of seqused_cmp_kv. If seqused_cmp_kv is not passed in, the seqlen is obtained from cmp_topk_length.
- Constraints on Atlas A3 products:
  - In the SWA sparse ori_kv scenario, only the SWA template is supported. Set `has_ori_kv` to true, `has_cmp_kv` to false, and `ori_mask_mode` to 0, set `ori_topk` to a value greater than 0, and set `ori_win_left` and `ori_win_right` to non-negative numbers. `ori_topk_length` must be passed in. `ori_topk` must be consistent with K, the last dimension of `ori_sparse_indices` of the paired main operator. `ori_topk_length` indicates the number of left-aligned valid index entries for each q token and KV head, and its value must be in the range [0, K]. Metadata uses only `ori_topk_length` to generate the task partitioning. The paired main operator still requires `seqused_ori_kv` to be passed in for the PA_BBND scenario.
  - `cmp_ratio` indicates the compression ratio of `cmp_kv` relative to the KV length before compression. This parameter is not involved in the compressed KV computation when only `ori_kv` is passed in. Set the value to 1, 2, or 4 in the CSA scenario, and 128 in the HCA scenario.
  - `cmp_topk` supports any integer in [1, 8192] in the CSA scenario. Set the value to 0 in the SWA and HCA scenarios.
- Constraints on Atlas A2 products:
  - In the SWA sparse ori_kv scenario, only the SWA template is supported. Set `has_ori_kv` to true, `has_cmp_kv` to false, and `ori_mask_mode` to 0, set `ori_topk` to a value greater than 0, and set `ori_win_left` and `ori_win_right` to non-negative numbers. `ori_topk_length` must be passed in. `ori_topk` must be consistent with K, the last dimension of `ori_sparse_indices` of the paired main operator. `ori_topk_length` indicates the number of left-aligned valid index entries for each q token and KV head, and its value must be in the range [0, K]. Metadata uses only `ori_topk_length` to generate the task partitioning. The paired main operator still requires `seqused_ori_kv` to be passed in for the PA_BBND scenario.
  - `cmp_ratio` indicates the compression ratio of `cmp_kv` relative to the KV length before compression. This parameter is not involved in the compressed KV computation when only `ori_kv` is passed in. Set the value to 1, 2, or 4 in the CSA scenario, and 128 in the HCA scenario.
  - `cmp_topk` supports any integer in [1, 8192] in the CSA scenario. Set the value to 0 in the SWA and HCA scenarios.

## Calling Example

| Calling Method | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn API | [test_aclnn_sparse_flash_mla_metadata](./examples/test_aclnn_sparse_flash_mla_metadata.cpp) | Calls the SparseFlashMlaMetadata operator through [aclnnSparseFlashMlaMetadata](docs/aclnnSparseFlashMlaMetadata_en.md) |
| PyTorch API | [test_torch_sparse_flash_mla_metadata](./examples/test_torch_sparse_flash_mla_metadata.py) | Generates the metadata used by the SparseFlashMla main operator through the [sparse_flash_mla_metadata](../sparse_flash_mla/docs/torchapi_sparse_flash_mla_en.md) API |
