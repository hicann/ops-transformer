# QuantSparseFlashMla

English | [简体中文](./README.md)

## Product Support

| Product                                                         | Supported |
| :------------------------------------------------------------ | :------: |
|<term>Ascend 950PR&950DT products</term>                        | √  |
|<term>Atlas A3 products</term>        | ×  |
|<term>Atlas A2 products</term>        | ×  |
|<term>Atlas 200I/500 A2 inference products</term>                    | ×  |
|<term>Atlas inference products</term>                                | ×  |
|<term>Atlas training products</term>                                | ×  |

## Description

- Operator function:

  The `QuantSparseFlashMla` operator performs MLA (Multi-head Latent Attention) computation in fully quantized and sparse scenarios. The API supports the following three computation modes:
  - **SWA (Sliding Window Attention)**: Pass in only `ori_kv` to perform sliding window attention on the original KV.
  - **CSA (Compressed Sparse Attention)**: Pass in `ori_kv`, `cmp_kv`, and `cmp_sparse_indices` together to perform attention on both the original KV window and the compressed KV selected by topK.
  - **HCA (Heavily Compressed Attention)**: Pass in `ori_kv` and `cmp_kv` together to perform attention on both the original KV window and the continuously compressed KV segments.

  `QuantSparseFlashMlaMetadata` provides the core partition information of `QuantSparseFlashMla` and is generated before the main operator is executed. The current version requires the main operator to receive this metadata. The typical invocation flow is as follows:

  1. Prepare the corresponding inputs according to the invocation scenario.
  2. Call `QuantSparseFlashMlaMetadata` to generate `metadata`, which is passed to `QuantSparseFlashMla` as an input parameter.
  3. Call `QuantSparseFlashMla` to generate the computation result.

- Calculation formula:
  `QuantSparseFlashMla` uses MLA (Multi-head Latent Attention) to compute sparse attention with KV-shared inputs. The principle is to selectively compress and quantize the input KV, and then obtain attention weights by applying Softmax to the product of the Query and the concatenated KV computation result.

  The calculation formula of MLA is generally defined as follows, where $\tilde{K}=\tilde{V}$ is the KV that actually participates in the computation based on input parameter control. It consists of the sliding window part of `ori_kv` and the compressed part of `cmp_kv`. The range of KV that actually participates in the computation is determined by `cmp_ratio`, `ori_mask_mode`, `cmp_mask_mode`, `ori_win_left`, `ori_win_right`, and `cmp_sparse_indices`.

  $$
  O = \text{softmax}(Q@\tilde{K}^T \cdot \text{softmax\_scale})@\tilde{V}
  $$

## Parameters

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
      <td>q</td>
      <td>Input</td>
      <td>Corresponds to Q in the formula.</td>
      <td>HIFLOAT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ori_kv</td>
      <td>Optional input</td>
      <td>Part of K and V in the formula, indicating the original quantized KV without compression. Default value: None.</td>
      <td>HIFLOAT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_kv</td>
      <td>Optional input</td>
      <td>Part of K and V in the formula, indicating the compressed quantized KV. Default value: None.</td>
      <td>HIFLOAT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>q_descale</td>
      <td>Optional input</td>
      <td>Quantization parameter of q. Default value: None.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ori_kv_descale</td>
      <td>Optional input</td>
      <td>Quantization parameter of ori_kv. Default value: None.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_kv_descale</td>
      <td>Optional input</td>
      <td>Quantization parameter of cmp_kv. Default value: None.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ori_sparse_indices</td>
      <td>Optional input</td>
      <td>Indexes for discrete data fetch from ori_kv; invalid positions are filled with -1. Default value: None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_sparse_indices</td>
      <td>Optional input</td>
      <td>Indexes for discrete data fetch from cmp_kv; invalid positions are filled with -1. Default value: None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ori_block_table</td>
      <td>Optional input</td>
      <td>Block mapping table used by ori_kv in PageAttention. Default value: None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_block_table</td>
      <td>Optional input</td>
      <td>Block mapping table used by cmp_kv in PageAttention. Default value: None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_q</td>
      <td>Optional input</td>
      <td>Cumulative sequence lengths of q in different batches under the TND layout. Default value: None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_ori_kv</td>
      <td>Optional input</td>
      <td>Cumulative sequence lengths of ori_kv in different batches under the TND layout. Default value: None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_cmp_kv</td>
      <td>Optional input</td>
      <td>Cumulative sequence lengths of cmp_kv in different batches under the TND layout. Default value: None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_q</td>
      <td>Optional input</td>
      <td>Number of tokens of q that actually participate in the computation in each batch. Default value: None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_ori_kv</td>
      <td>Optional input</td>
      <td>Number of tokens of ori_kv that actually participate in the computation in each batch. Default value: None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_cmp_kv</td>
      <td>Optional input</td>
      <td>Number of tokens of cmp_kv that actually participate in the computation in each batch. Default value: None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_residual_kv</td>
      <td>Optional input</td>
      <td>Remainder of the compressed KV, used to restore the KV length before compression for the cmp-side mask. Default value: None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ori_topk_length</td>
      <td>Optional input</td>
      <td>Identifies the length of ori_sparse_indices that actually participates in the computation. Default value: None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_topk_length</td>
      <td>Optional input</td>
      <td>Identifies the length of cmp_sparse_indices that actually participates in the computation. Default value: None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sinks</td>
      <td>Optional input</td>
      <td>Sets an independent learnable virtual offset for each attention head to maintain stability during long-text inference. Default value: None.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>metadata</td>
      <td>Required input</td>
      <td>The task partition result generated by QuantSparseFlashMlaMetadata must be passed in.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>quant_mode</td>
      <td>Attribute</td>
      <td>Quantization mode. Quantization mode 1 indicates that Q, K, and V are per-tensor quantized, and the data types of Q, K, and V are HIFLOAT8.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmax_scale</td>
      <td>Optional attribute</td>
      <td>Corresponds to softmax_scale in the formula.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_ratio</td>
      <td>Optional attribute</td>
      <td>Compression ratio of cmp_kv relative to the KV length before compression, used to restore the KV length before compression for the cmp-side mask. Default value: 1.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_mask_mode</td>
      <td>Optional attribute</td>
      <td>Mask mode for the computation of q and ori_kv.<br/>0: No Mask.<br/>3: RightDownCausal mode.<br/>4: Band mode. Default value: 0.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_mask_mode</td>
      <td>Optional attribute</td>
      <td>Mask mode for the computation of q and cmp_kv.<br/>0: No Mask.<br/>3: RightDownCausal mode. Default value: 0.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_win_left</td>
      <td>Optional attribute</td>
      <td>Number of past tokens computed by q in the computation of q and ori_kv. -1 or non-negative numbers are supported, where -1 indicates that the window is unrestricted. Default value: -1.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_win_right</td>
      <td>Optional attribute</td>
      <td>Number of future tokens computed by q in the computation of q and ori_kv. -1 or non-negative numbers are supported, where -1 indicates that the window is unrestricted. Default value: -1.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_q</td>
      <td>Optional attribute</td>
      <td>Data layout format of input q. "BSND" and "TND" are supported. Default value: "BSND".</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_kv</td>
      <td>Optional attribute</td>
      <td>Data layout format of input ori_kv and cmp_kv. "BSND", "TND", and "PA_BBND" are supported. Default value: "BSND".</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>topk_value_mode</td>
      <td>Optional attribute</td>
      <td>TopK index value mode. The default value is 1, and 1 and 2 are supported. When the value is 1, the sparse indexes indicate logical token indexes. When the value is 2, the sparse indexes under the PA_BBND layout indicate physical token offsets, that is, physical_block_id * block_size + offset_in_block.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>return_softmax_lse</td>
      <td>Optional attribute</td>
      <td>Indicates whether to return softmax_lse. Default value: False.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attn_out</td>
      <td>Output</td>
      <td>Corresponds to output O in the formula.</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>softmax_lse</td>
      <td>Optional output</td>
      <td>Returns the log-sum-exp result of softmax. By default, no valid result is returned (return_softmax_lse=False). When the value is True, a valid value is output.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## Constraints

- This API is used in inference scenarios.
- This API supports the aclgraph mode.
- This API currently supports three computation scenarios: the SWA (Sliding Window Attention) scenario, in which only `ori_kv` is passed in; the CSA (Compressed Sparse Attention) scenario, in which `ori_kv`, `cmp_kv`, and `cmp_sparse_indices` are passed in together; and the HCA (Heavily Compressed Attention) scenario, in which `ori_kv` and `cmp_kv` are passed in together;
- The general specification constraints are as follows:
  - N2 supports only 1, and D supports only 512.
  - `cmp_ratio` indicates the compression ratio of `cmp_kv` relative to the KV length before compression. When only `ori_kv` is passed in, `cmp_ratio` does not participate in the compressed KV computation and must remain at the default value 1. Values from 1 to 128 are supported.
  - `ori_mask_mode` supports 0/3/4, `cmp_mask_mode` supports 0/3, `ori_win_left` supports -1 or non-negative numbers, and `ori_win_right` supports -1 or non-negative numbers. Only when `ori_mask_mode` is 4 can `ori_win_left` and `ori_win_right` be >=0.
  - The combinations of `layout_q` and `layout_kv` support only "BSND"/"BSND", "TND"/"TND", "BSND"/"PA_BBND", and "TND"/"PA_BBND". In non-PA_BBND scenarios, `layout_q` and `layout_kv` must be the same. In PA_BBND scenarios, `block_size` supports 1 to 1024.

- When `layout_q` is TND, the function usage restrictions are as follows:
  - The shape of `q` must be [T1,N1,D].
  - The shape dimensions of `ori_sparse_indices` are [Q\_T, KV\_N, K1], where K1 is the number of tokens discretely selected from `ori_kv` at a time.
  - The shape of `cmp_sparse_indices` must be [Q\_T, KV\_N, K2], where K2 is the number of tokens discretely selected from `cmp_kv` at a time.
  - `cu_seqlens_q` must be passed in. The input dimension is B+1, and the value of each element indicates the total number of tokens in the current batch and all preceding batches, that is, the prefix sum. Therefore, the value of each subsequent element must be >= that of the previous element, and the first element must be 0.

- When `layout_q` is BSND, the function usage restrictions are as follows:
  - The shape of `q` must be [B, Q\_S, Q\_N, D].
  - The shape of `ori_sparse_indices` must be [B, Q\_S, KV\_N, K1], where K1 is the number of tokens discretely selected from `ori_kv` at a time.
  - The shape of `cmp_sparse_indices` must be [B, Q\_S, KV\_N, K2], where K2 is the number of tokens discretely selected from `cmp_kv` at a time.

- In PageAttention scenarios, the function usage restrictions are as follows:
  - The shapes of `ori_kv` and `cmp_kv` are [ori\_block\_num, ori\_block\_size, KV\_N, D] and [cmp\_block\_num, cmp\_block\_size, KV\_N, D] respectively, where ori\_block\_num and cmp\_block\_num are the total number of blocks in PageAttention, and ori\_block\_size and cmp\_block\_size are the number of tokens in a block. ori\_block\_size and cmp\_block\_size range from 1 to 1024, and KV_N supports only 1.
  - The shapes of `ori_block_table` and `cmp_block_table` are 2-dimensional, where the length of the first dimension is B, and the length of the second dimension is not less than the number of blocks corresponding to the maximum S2 and S3 among all batches, that is, S2\_max / block\_size and S3\_max / block\_size rounded up.
- `metadata` is the core partition result actually used by the operator. Currently, this parameter is required, and the shape size is fixed to [2048].
- `layout_kv` supports "BSND", "TND", and "PA_BBND" as inputs, which must satisfy the preceding combination constraints of `layout_q` and `layout_kv`.
  - When the input is PA_BBND, `seqused_ori_kv` and `ori_block_table` must be passed in. When the input is BSND, `seqused_ori_kv` can express the valid length of `ori_kv` in each batch. When the input is TND, the valid length of `ori_kv` is expressed by `cu_seqlens_ori_kv`.
  - When the input is BSND, the layouts of both `ori_kv` and `cmp_kv` must be BSND. The shape of ori_kv is [B, S2, N2,D], and the shape of cmp_kv is [B, S3, N2,D].
  - When the input is TND, `cu_seqlens_ori_kv` must be passed in. If `cmp_kv` exists, `cu_seqlens_cmp_kv` must also be passed in.
- When `return_softmax_lse` is False, a placeholder tensor is returned. When it is True, the log-sum-exp result of softmax is returned.
- The passed-in tensors cannot be empty.
- `seqused_cmp_kv` is an optional input for all `layout_kv` values. When explicitly passed in, it overrides the logical valid length on the cmp side. When it is not passed in, the length is derived from the `cmp_kv` shape, `cu_seqlens_cmp_kv`, or PA block table semantics.
- `cmp_residual_kv` is an optional input parameter of the operator. After being passed in, it is used to restore the KV length before compression for the cmp-side mask by `cmp_len * cmp_ratio + residual`, where `cmp_len` preferentially comes from the explicitly passed `seqused_cmp_kv`.
- The data layout formats of `q`, `ori_kv`, and `cmp_kv` can be interpreted from multiple dimensions. B (Batch) indicates the batch size of input samples, S (Seq-Length) indicates the sequence length of input samples, H (Hidden-Size) indicates the size of the hidden layer, N (Head-Num) indicates the number of heads, D (Head-Dim) indicates the size of the smallest unit in the hidden layer and satisfies D=H/N, and T indicates the sum of sequence lengths of input samples across all batches.
- Q\_S and S1 indicate the S in the q shape, S2 indicates the S in the ori_kv shape, and S3 indicates the S in the cmp_kv shape. Q\_N and N1 indicate num\_q\_heads, and KV\_N and N2 indicate num\_ori_kv\_heads and num\_cmp_kv\_heads. Q\_T and T1 indicate the sum of sequence lengths of input samples in the q shape.

## Invocation

| Invocation Method  | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn API | [test_aclnn_quant_sparse_flash_mla](./examples/test_aclnn_quant_sparse_flash_mla.cpp) | Calls the QuantSparseFlashMla operator through [aclnnQuantSparseFlashMla](docs/aclnnQuantSparseFlashMla_en.md) |
| PyTorch API | [quant_sparse_flash_mla](docs/torchapi_quant_sparse_flash_mla_en.md) | Calls the QuantSparseFlashMla operator through `cann_ops_transformer.quant_sparse_flash_mla` |
