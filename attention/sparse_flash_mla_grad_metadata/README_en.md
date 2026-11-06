# SparseFlashMlaGradMetadata

English | [简体中文](./README.md)

## Product Support Status

| Product                                                      | Supported |
| :----------------------------------------------------------- | :------: |
|<term>Ascend 950PR&950DT products</term>|      √     |
|<term>Atlas A3 products</term>|      ×     |
|<term>Atlas A2 products</term>|      ×     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function Description

- Operator function: The `SparseFlashMlaGradMetadata` operator generates a task list that contains the start and end Batch and Head indices and the Q and K block indices of the attention computation task of each AIcore, for use by the subsequent `SparseFlashMlaGrad` operator.

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
      <td>Valid Sequence Length of q in different batches, with shape (B+1, ).</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_ori_kv</td>
      <td>Optional input</td>
      <td>Valid Sequence Length of ori_kv in different batches, with shape (B+1, ).</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_cmp_kv</td>
      <td>Optional input</td>
      <td>Valid Sequence Length of cmp_kv in different batches, with shape (B+1, ).</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_q</td>
      <td>Optional input</td>
      <td>Sequence Length of q that actually participates in the computation in different batches, with shape (B, ).</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_ori_kv</td>
      <td>Optional input</td>
      <td>Sequence Length of ori_kv that actually participates in the computation in different batches, with shape (B, ).</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_cmp_kv</td>
      <td>Optional input</td>
      <td>Sequence Length of cmp_kv that actually participates in the computation in different batches, with shape (B, ).</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_residual_kv</td>
      <td>Optional input</td>
      <td>Remainder of the compressed Sequence Length of cmp_kv in different batches. Used together with cmp_ratio to compute the mask and load of the cmp_kv part. Must be passed in when cmp_mask_mode=3 and cmp_ratio≠1, with shape (B, ).</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ori_topk_length</td>
      <td>Optional input</td>
      <td>Number of key sparse tokens in the ori_kv part corresponding to each q token, with shape (B, S1, N2) or (T1, N2).</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_topk_length</td>
      <td>Optional input</td>
      <td>Number of key sparse tokens in the cmp_kv part corresponding to each q token, with shape (B, S1, N2) or (T1, N2).</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num_heads_q</td>
      <td>Attribute</td>
      <td>Number of heads of q. Currently supports [1, 128].</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_heads_kv</td>
      <td>Attribute</td>
      <td>Number of heads corresponding to ori_kv and cmp_kv. Currently supports only 1.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>head_dim</td>
      <td>Attribute</td>
      <td>Dimension of the attention head. Currently supports only 512.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>batch_size</td>
      <td>Optional attribute</td>
      <td>Number of batches. The default value is 0.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_q</td>
      <td>Optional attribute</td>
      <td>Maximum Sequence Length of q. The default value is 0.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_ori_kv</td>
      <td>Optional attribute</td>
      <td>Maximum Sequence Length of ori_kv. The default value is 0.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_cmp_kv</td>
      <td>Optional attribute</td>
      <td>Maximum Sequence Length of cmp_kv. The default value is 0.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_topk</td>
      <td>Optional attribute</td>
      <td>Number of key sparse tokens filtered from ori_kv. 0 indicates a non-sparse scenario. The default value is 0.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_topk</td>
      <td>Optional attribute</td>
      <td>Number of key sparse tokens filtered from cmp_kv. 0 indicates a non-sparse scenario. The default value is 0.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_ratio</td>
      <td>Optional attribute</td>
      <td>Compression ratio of cmp_kv. The default value is 1. Currently supports [1, 128].</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_mask_mode</td>
      <td>Optional attribute</td>
      <td>Mask mode for the computation between q and ori_kv. 0 indicates No mask, 3 indicates the rightDownCausal mode, and 4 indicates the sliding window mode. The default value is 0.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_mask_mode</td>
      <td>Optional attribute</td>
      <td>Mask mode for the computation between q and cmp_kv. 0 indicates No mask and 3 indicates the rightDownCausal mode. The default value is 0.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_win_left</td>
      <td>Optional attribute</td>
      <td>Number of past tokens computed by q in the computation between q and ori_kv. -1 indicates infinity. The default value is -1.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_win_right</td>
      <td>Optional attribute</td>
      <td>Number of future tokens computed by q in the computation between q and ori_kv. -1 indicates infinity. The default value is -1.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_q</td>
      <td>Optional attribute</td>
      <td>Arrangement format of q. Supports BSND and TND. The default value is BSND.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_kv</td>
      <td>Optional attribute</td>
      <td>Arrangement format of ori_kv and cmp_kv. Supports BSND and TND. The default value is BSND.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>has_ori_kv</td>
      <td>Optional attribute</td>
      <td>Identifies whether ori_kv exists. The default value is true.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>has_cmp_kv</td>
      <td>Optional attribute</td>
      <td>Identifies whether cmp_kv exists. The default value is true.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadata</td>
      <td>Output</td>
      <td>Output of the load balancing result, with a fixed shape of (1024, )</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

## Constraints

- The SparseFlashMlaGradMetadata operator must be used together with the SparseFlashMlaGrad operator.
- B (Batch) indicates the batch size of the input samples. q, ori_kv, and cmp_kv are inputs of the supporting SparseFlashMlaGrad operator. S1 indicates the size of the S axis in the q shape when layout_q=BSND, T1 indicates the size of the T axis in the q shape when layout_q=TND, S2 indicates the size of the S axis in the ori_kv shape when layout_kv=BSND, S3 indicates the size of the S axis in the cmp_kv shape when layout_kv=BSND, and N2 indicates the size of the N axis in the ori_kv and cmp_kv shapes.
- layout_q and layout_kv must be the same.
- The values of cu_seqlens_q, cu_seqlens_ori_kv, and cu_seqlens_cmp_kv must be the cumulative number of valid tokens in the current batch and all preceding batches. The first element is fixed to 0, and each subsequent element must be greater than or equal to the previous element.
- The values of seqused_q, seqused_ori_kv, and seqused_cmp_kv must indicate the number of valid tokens in each batch.
- cmp_residual_kv must satisfy cmp_residual_kv[i] < cmp_ratio.
- For details about the mask modes represented by ori_mask_mode and cmp_mask_mode, see [sparse_mode Parameter Description](../../docs/zh/context/sparse_mode_introduction_en.md).
- When has_ori_kv is true, ori_topk greater than 0 indicates that the ori_kv part is sparse, and ori_topk of 0 indicates that the ori_kv part is non-sparse.
- When has_cmp_kv is true, cmp_topk greater than 0 indicates that the cmp_kv part is sparse, and cmp_topk of 0 indicates that the cmp_kv part is non-sparse.
- When has_ori_kv is true, ori_topk is not 0, and ori_mask_mode is 0, ori_topk_length must be passed in. In this case, the minimum of the ori_mask_mode rule and the ori_topk_length element is used as the valid seqlen of ori_kv corresponding to the current q token. In other ori_kv sparse scenarios, the minimum of the ori_mask_mode rule and ori_topk is used as the valid seqlen of ori_kv corresponding to the current q token.
- When has_cmp_kv is true, cmp_topk is not 0, and cmp_mask_mode is 0, cmp_topk_length must be passed in. In this case, the minimum of the cmp_mask_mode rule and the cmp_topk_length element is used as the valid seqlen of cmp_kv corresponding to the current q token. In other cmp_kv sparse scenarios, the minimum of the cmp_mask_mode rule and cmp_topk is used as the valid seqlen of cmp_kv corresponding to the current q token.
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
- Batch value rules
  - When layout_q is BSND, the batch is derived from the shape of seqused_q first; if seqused_q is not passed in, the batch number is obtained from batch_size.
  - When layout_q is TND, the batch is derived from the shape of seqused_q first; if seqused_q is not passed in, the batch is derived from the shape of cu_seqlens_q.
- q Seqlen value rules
  - When layout_q is BSND, the seqlen is obtained from the elements of seqused_q first; if seqused_q is not passed in, the seqlen is obtained from max_seqlen_q.
  - When layout_q is TND, the seqlen is obtained from the elements of seqused_q first; if seqused_q is not passed in, the seqlen is obtained from the elements of cu_seqlens_q.
- ori_kv Seqlen value rules
  - When layout_kv is BSND, the seqlen is obtained from the elements of seqused_ori_kv first; if seqused_ori_kv is not passed in, the seqlen is obtained from max_seqlen_ori_kv.
  - When layout_kv is TND, the seqlen is obtained from the elements of seqused_ori_kv first; if seqused_ori_kv is not passed in, the seqlen is obtained from the elements of cu_seqlens_ori_kv.
- cmp_kv Seqlen value rules
  - When layout_kv is BSND, the seqlen is obtained from the elements of seqused_cmp_kv first; if seqused_cmp_kv is not passed in, the seqlen is obtained from max_seqlen_cmp_kv.
  - When layout_kv is TND, the seqlen is obtained from the elements of seqused_cmp_kv first; if seqused_cmp_kv is not passed in, the seqlen is obtained from the elements of cu_seqlens_cmp_kv.

## Calling Instructions

| Calling Method | Sample Code                                                  | Description                                                  |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn API | [test_aclnn_sparse_flash_mla_grad_metadata](./examples/test_aclnn_sparse_flash_mla_grad_metadata.cpp) | Calls the SparseFlashMlaGradMetadata operator through the [aclnnSparseFlashMlaGradMetadata](docs/aclnnSparseFlashMlaGradMetadata_en.md) API. |
| PyTorch API | [test_torch_sparse_flash_mla_grad_metadata](./examples/test_torch_sparse_flash_mla_grad_metadata.py) | Calls the SparseFlashMlaGradMetadata operator through the [sparse_flash_mla_grad_metadata](../sparse_flash_mla_grad/docs/torchapi_sparse_flash_mla_grad_en.md) API. |
