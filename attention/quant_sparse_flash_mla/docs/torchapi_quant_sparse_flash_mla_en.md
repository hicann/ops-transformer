# quant_sparse_flash_mla

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T12:52:42.881Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
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

## Function

- Description:
  `quant_sparse_flash_mla` is an extended `torch` API of `cann_ops_transformer`. It calls the `QuantSparseFlashMla` operator to perform MLA (Multi-head Latent Attention) attention computation in quantization and sparsity use cases, with unified support for training and inference. The three typical use cases of this operator are as follows:

  - **SWA (Sliding Window Attention)**: Pass in only `ori_kv` to perform sliding window attention over the original KV.
  - **CSA (Compressed Sparse Attention)**: Pass in `ori_kv`, `cmp_kv`, and `cmp_sparse_indices` to perform attention over both the original KV window and the compressed KV selected by topK.
  - **HCA (Heavily Compressed Attention)**: Pass both `ori_kv` and `cmp_kv` to jointly perform attention on the original KV window and the contiguous compressed KV segments.

  `quant_sparse_flash_mla_metadata` provides the core splitting information for `quant_sparse_flash_mla` and is called before the main operator is executed. In the current version, the main operator requires this metadata. The typical call flow is as follows:

  1. Prepare the corresponding inputs based on the use case.
  2. Call `quant_sparse_flash_mla_metadata` to generate `metadata` as an input parameter of `quant_sparse_flash_mla`.
  3. Call `quant_sparse_flash_mla` to generate the computation result.

- **Formula**:

  quant_sparse_flash_mla uses MLA to compute sparse attention for inputs with shared KV. The operator selectively compresses and quantizes the input KV, then applies Softmax to the Query and the concatenated KV computation result to obtain the attention weights.

  The MLA computation formula is generally defined as follows, where $\tilde{K}=\tilde{V}$ denotes the KV actually involved in the computation under the control of input parameters. It consists of the sliding window part of `ori_kv` and the compressed part of `cmp_kv`. The range of the KV actually involved in the computation is determined by `cmp_ratio`, `ori_mask_mode`, `cmp_mask_mode`, `ori_win_left`, `ori_win_right`, and `cmp_sparse_indices`.

  $$
  O = \text{softmax}(Q @ \tilde{K}^T \cdot \text{softmax\_scale}) @ \tilde{V}
  $$

## Function Prototype

Before calling the quant_sparse_flash_mla API, call the prerequisite API quant_sparse_flash_mla_metadata first.

```python
cann_ops_transformer.quant_sparse_flash_mla_metadata(
    num_heads_q,
    num_heads_kv,
    head_dim,
    quant_mode,
    *,
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    cmp_residual_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    batch_size=0,
    max_seqlen_q=0,
    max_seqlen_ori_kv=0,
    max_seqlen_cmp_kv=0,
    ori_topk=0,
    cmp_topk=0,
    cmp_ratio=1,
    ori_mask_mode=0,
    cmp_mask_mode=0,
    ori_win_left=-1,
    ori_win_right=-1,
    layout_q="BSND",
    layout_kv="BSND",
    has_ori_kv=True,
    has_cmp_kv=True
) -> Tensor
```

```python
cann_ops_transformer.quant_sparse_flash_mla(
    q,
    quant_mode,
    *,
    ori_kv=None,
    cmp_kv=None,
    q_descale=None,
    ori_kv_descale=None,
    cmp_kv_descale=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    cmp_residual_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    sinks=None,
    metadata=None,
    softmax_scale=1.0,
    cmp_ratio=1,
    ori_mask_mode=0,
    cmp_mask_mode=0,
    ori_win_left=-1,
    ori_win_right=-1,
    layout_q="BSND",
    layout_kv="BSND",
    topk_value_mode=1,
    return_softmax_lse=False
) -> (Tensor, Tensor)
```

## Parameters

### Common Field Definitions

| Name | Meaning |
| :---------: | :---------------------------------------------------------: |
| b | Batch size of input samples. |
| q_s | Sequence length of input q. |
| ori_kv_s | Sequence length of input ori_kv. |
| cmp_kv_s | Sequence length of input cmp_kv. |
| q_n | Number of heads of input q. |
| kv_n | Number of heads of input ori_kv/cmp_kv. |
| q_d | Dimension of the attention head of input q. |
| kv_d | Dimension of the attention head of input ori_kv/cmp_kv. |
| q_t | Cumulative sum of sequence lengths of all batches in input q. |
| ori_kv_t | Cumulative sum of sequence lengths of all batches in input ori_kv. |
| cmp_kv_t | Cumulative sum of sequence lengths of all batches in input cmp_kv. |
| ori_kv_k | Number of tokens selected by topK in input ori_sparse_indices. |
| cmp_kv_k | Number of tokens selected by topK in input cmp_sparse_indices. |
| ori_kv_s_max | Maximum sequence length of input ori_kv. |
| cmp_kv_s_max | Maximum sequence length of input cmp_kv. |
| ori_kv_block_size | Block size of input ori_kv in the PagedAttention use case. |
| cmp_kv_block_size | Block size of input cmp_kv in the PagedAttention use case. |
| ori_kv_block_nums | Number of blocks of input ori_kv in the PagedAttention use case. |
| cmp_kv_block_nums | Number of blocks of input cmp_kv in the PagedAttention use case. |

### quant_sparse_flash_mla_metadata

| Name | Type | Required/Optional | Description | Data Type | Data Format | Shape
| :--- | :--- | :--- | :--- | :--- | :--- | :---
| num_heads_q | int | Required | Number of q heads. | int32 | - | -
| num_heads_kv | int | Required | Number of ori_kv/cmp_kv heads. | int32 | - | -
| head_dim | int | Required | Dimension of each attention head. | int32 | - | -
| quant_mode | int | Required | Quantization mode. Currently, only 1 is supported. 1 indicates per-tensor quantization for Q, K, and V with the hifloat8 data type. | int32 | - | -
| cu_seqlens_q | tensor | Optional | Cumulative sequence lengths of the variable-length sequences for input q. | int32 | ND | <ul><li>(b+1,)</li></ul>
| cu_seqlens_ori_kv | tensor | Optional | Cumulative sequence lengths of the variable-length sequences for input ori_kv. | int32 | ND | <ul><li>(b+1,)</li></ul>
| cu_seqlens_cmp_kv | tensor | Optional | Cumulative sequence lengths of the variable-length sequences for input cmp_kv. | int32 | ND | <ul><li>(b+1,)</li></ul>
| seqused_q | tensor | Optional | Actual sequence lengths participating in computation in each batch of input q. | int32 | ND | <ul><li>(b,)</li></ul>
| seqused_ori_kv | tensor | Optional | Actual sequence lengths participating in computation in each batch of input ori_kv. | int32 | ND | <ul><li>(b,)</li></ul>
| seqused_cmp_kv | tensor | Optional | Actual sequence lengths participating in computation in each batch of input cmp_kv. | int32 | ND | <ul><li>(b,)</li></ul>
| cmp_residual_kv | tensor | Optional | Remainder of the compressed sequence length of cmp_kv in each batch. | int32 | ND | <ul><li>(b,)</li></ul>
| ori_topk_length | tensor | Optional | Actual length of ori_sparse_indices participating in computation. | int32 | ND | <ul><li>(b, q_s, kv_n)</li><li>(q_t, kv_n)</li></ul>
| cmp_topk_length | tensor | Optional | Actual length of cmp_sparse_indices participating in computation. | int32 | ND | <ul><li>(b, q_s, kv_n)</li><li>(q_t, kv_n)</li></ul>
| batch_size | int | Optional | Batch size. | int32 | - | -
| max_seqlen_q | int | Optional | Maximum length of the query q sequence. | int32 | - | -
| max_seqlen_ori_kv | int | Optional | Maximum length of the ori_kv sequence. | int32 | - | -
| max_seqlen_cmp_kv | int | Optional | Maximum length of the cmp_kv sequence. | int32 | - | -
| ori_topk | int | Optional | Number of key sparse tokens selected from ori_kv. 0 indicates a non-sparse use case. Defaults to 0. | int32 | - | -
| cmp_topk | int | Optional | Number of key sparse tokens selected from cmp_kv. 0 indicates a non-sparse use case. Defaults to 0. | int32 | - | -
| cmp_ratio | int | Optional | Compression ratio of cmp_kv. Defaults to 1. | int32 | - | -
| ori_mask_mode | int | Optional | Mask mode for the computation between q and ori_kv. 0: No mask. 3: rightDownCausal mode. 4: sliding window mode. Defaults to 0. | int32 | - | -
| cmp_mask_mode | int | Optional | Mask mode for the computation between q and cmp_kv. 0: No mask. 3: rightDownCausal mode. Defaults to 0. | int32 | - | -
| ori_win_left | int | Optional | Number of past tokens involved in the computation between q and ori_kv. -1 indicates infinity, meaning all tokens participate in computation. Defaults to -1. | int32 | - | -
| ori_win_right | int | Optional | Number of future tokens involved in the computation between q and ori_kv. -1 indicates infinity, meaning all tokens participate in computation. Defaults to -1. | int32 | - | -
| layout_q | string | Optional | Layout format of input q. Defaults to BSND. | string | - | -
| layout_kv | string | Optional | Layout format of input ori_kv/cmp_kv. Defaults to BSND. | string | - | -
| has_ori_kv | bool | Optional | Whether ori_kv is included. Defaults to True. | bool | - | -
| has_cmp_kv | bool | Optional | Whether cmp_kv is included. Defaults to True. | bool | - | -

### quant_sparse_flash_mla

| Name | Type | Required/Optional | Description | Data Type | Data Format | Dimension
| :--- | :--- | :--- | :--- | :--- | :--- | :---
| q | tensor | Required | q in the formula. | hifloat8 | ND | <ul><li>(b, q_s, q_n, q_d)</li><li>(q_t, q_n, q_d)</li></ul>
| quant_mode | int | Required | Quantization mode. Currently only 1 is supported. 1 indicates per-tensor quantization of Q, K, and V, with data type hifloat8. | int32 | - | -
| ori_kv | tensor | Optional | Original quantized KV input. Key and Value share the same data. | hifloat8 | ND | <ul><li>(b, ori_kv_s, kv_n, kv_d)</li><li>(ori_kv_t, kv_n, kv_d)</li><li>(ori_kv_block_nums, ori_kv_block_size, kv_n, kv_d)</li></ul>
| cmp_kv | tensor | Optional | Compressed quantized KV input. Key and Value share the same data. | hifloat8 | ND | <ul><li>(b, cmp_kv_s, kv_n, kv_d)</li><li>(cmp_kv_t, kv_n, kv_d)</li><li>(cmp_kv_block_nums, cmp_kv_block_size, kv_n, kv_d)</li></ul>
| q_descale | tensor | Optional | Quantization parameter corresponding to q. | float32 | ND | <ul><li>[1]</li></ul>
| ori_kv_descale | tensor | Optional | Quantization parameter corresponding to ori_kv. | float32 | ND | <ul><li>[1]</li></ul>
| cmp_kv_descale | tensor | Optional | Quantization parameter corresponding to cmp_kv. | float32 | ND | <ul><li>[1]</li></ul>
| ori_sparse_indices | tensor | Optional | TopK indices of the original KV. Invalid positions are filled with -1. | int32 | ND | <ul><li>(q_t, kv_n, ori_kv_k)</li><li>(b, q_s, kv_n, ori_kv_k)</li></ul>
| cmp_sparse_indices | tensor | Optional | TopK indices of the compressed KV. Invalid positions are filled with -1. | int32 | ND | <ul><li>(q_t, kv_n, cmp_kv_k)</li><li>(b, q_s, kv_n, cmp_kv_k)</li></ul>
| ori_block_table | tensor | Optional | Block mapping table used by ori_kv in PagedAttention use cases. | int32 | ND | <ul><li>(b, ceil(ori_kv_s_max/ori_kv_block_size))</li></ul>
| cmp_block_table | tensor | Optional | Block mapping table used by cmp_kv in PagedAttention use cases. | int32 | ND | <ul><li>(b, ceil(cmp_kv_s_max/cmp_kv_block_size))</li></ul>
| cu_seqlens_q | tensor | Optional | Cumulative sequence lengths for variable-length sequences of input q. | int32 | ND | <ul><li>(b+1,)</li></ul>
| cu_seqlens_ori_kv | tensor | Optional | Cumulative sequence lengths for variable-length sequences of input ori_kv. | int32 | ND | <ul><li>(b+1,)</li></ul>
| cu_seqlens_cmp_kv | tensor | Optional | Cumulative sequence lengths for variable-length sequences of input cmp_kv. | int32 | ND | <ul><li>(b+1,)</li></ul>
| seqused_q | tensor | Optional | Actual sequence length of input q participating in computation in each batch. | int32 | ND | <ul><li>(b,)</li></ul>
| seqused_ori_kv | tensor | Optional | Actual sequence length of input ori_kv participating in computation in each batch. | int32 | ND | <ul><li>(b,)</li></ul>
| seqused_cmp_kv | tensor | Optional | Actual sequence length of input cmp_kv participating in computation in each batch. | int32 | ND | <ul><li>(b,)</li></ul>
| cmp_residual_kv | tensor | Optional | Remainder of the sequence length of cmp_kv after compression in each batch. Required when cmp_mask_mode=3 and cmp_ratio!=1. Not allowed when cmp_mask_mode=0 or cmp_ratio=1. | int32 | ND | <ul><li>(b,)</li></ul>
| ori_topk_length | tensor | Optional | Actual length of ori_sparse_indices participating in computation. | int32 | ND | <ul><li>(b, q_s, kv_n)</li><li>(q_t, kv_n)</li></ul>
| cmp_topk_length | tensor | Optional | Actual length of cmp_sparse_indices participating in computation. | int32 | ND | <ul><li>(b, q_s, kv_n)</li><li>(q_t, kv_n)</li></ul>
| sinks | tensor | Optional | Independent learnable virtual offset set for each attention head, used to maintain stability in long-text inference. | float32 | ND | <ul><li>(q_n,)</li></ul>
| metadata | tensor | Optional | Kernel splitting information generated by quant_sparse_flash_mla_metadata. | int32 | ND | <ul><li>(1024,)</li></ul>
| softmax_scale | float | Optional | Scaling factor that can be explicitly set. Defaults to 1.0. | float32 | - | -
| cmp_ratio | int | Optional | Compression ratio of cmp_kv relative to the KV length before compression, used to restore the pre-compression KV length for the cmp-side mask. Defaults to 1. Value range: 1-128. | int32 | - | -
| ori_mask_mode | int | Optional | Mask mode for the computation between q and ori_kv. 0: No mask. 3: rightDownCausal mode. 4: sliding window mode. Defaults to 0. | int32 | - | -
| cmp_mask_mode | int | Optional | Mask mode for the computation between q and cmp_kv. 0: No mask. 3: rightDownCausal mode. Defaults to 0. | int32 | - | -
| ori_win_left | int | Optional | Number of historical tokens that q attends to in the computation between q and ori_kv. -1 indicates infinity, that is, all tokens participate in the computation. Defaults to -1. | int32 | - | -
| ori_win_right | int | Optional | Number of future tokens that q attends to in the computation between q and ori_kv. -1 indicates infinity, that is, all tokens participate in the computation. Defaults to -1. | int32 | - | -
| layout_q | string | Optional | Layout format of input q. Defaults to BSND. | string | - | -
| layout_kv | string | Optional | Layout format of input ori_kv/cmp_kv. Defaults to BSND. | string | - | -
| topk_value_mode | int | Optional | Value mode of topK indices. Defaults to 1. | int32 | - | -
| return_softmax_lse | bool | Optional | Whether to return the lse result of softmax. Defaults to False. | bool | - | -

## Return Values

### quant_sparse_flash_mla_metadata

| Name | Type | Required/Optional | Description | Data Type | Data Format | Dimension
| :--- | :--- | :--- | :--- | :--- | :--- | :---
| metadata | tensor | Required | Splitting information of quant_sparse_flash_mla. | int32 | ND | <ul><li>(1024,)</li></ul>

### quant_sparse_flash_mla

| Name | Type | Required/Optional | Description | Data Type | Data Format | Dimensions
| :--- | :--- | :--- | :--- | :--- | :--- | :---
| attention_out | tensor | Required | Computation output of quant_sparse_flash_mla. | bfloat16 | ND | <ul><li>(b, q_s, q_n, q_d)</li><li>(q_t, q_n, q_d)</li></ul>
| softmax_lse | tensor | Optional | Result obtained by taking the max of the query × key result to obtain softmax_max, subtracting softmax_max from the query × key result, applying exp and sum to obtain softmax_sum, and finally adding softmax_max to the log of softmax_sum. | float32 | ND | <ul><li>(b, kv_n, q_s, q_n/kv_n)</li><li>(kv_n, q_t, q_n/kv_n)</li></ul>

## Constraints

- Statements
  - Inputs such as cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_q, seqused_ori_kv, seqused_cmp_kv, cmp_residual_kv, ori_block_table, and cmp_block_table are tensors. Since the operator cannot obtain the actual values of tensors during the Tiling phase, tiling does not validate the values, and users must ensure their correctness. Passing illegal values to the preceding parameters triggers undefined behavior (such as precision issues and program crashes caused by invalid memory access).
  - The input parameters of quant_sparse_flash_mla_metadata and quant_sparse_flash_mla must be consistent when the APIs are called. Since the operator is split into two APIs called in phases, it cannot verify the inputs itself, and users must ensure correctness. If the parameters passed to the APIs are inconsistent, undefined behavior occurs (such as precision issues and program crashes caused by invalid memory access).
  - ori_topk_length and cmp_topk_length indicate the actual lengths of the ori/cmp sparse_indices participating in the computation. The values cannot be greater than the size of the last dimension of sparse_indices. When seqused_q is passed in, the values in the valid part of topk_length must be greater than or equal to 0.
  - When ori_mask_mode/cmp_mask_mode is 0, ori_kv_k/cmp_kv_k must be greater than or equal to the maximum value of ori_topk_length/cmp_topk_length.
  - cmp_residual_kv is used together with cmp_ratio to restore the KV length before compression. The value of each batch must be less than cmp_ratio. It can be passed in only when cmp_mask_mode=3 and cmp_ratio!=1; it cannot be passed in when cmp_mask_mode=0 or cmp_ratio=1.
  - attention_out: tensor, the output in the formula. Supported data type: bfloat16. Supported data format: ND. Constraint: the shape of this output parameter must be the same as the shape of the input q, and the dtype must be the same as that of q.
  - If return_softmax_lse=False, a tensor of shape [1] with value 0 is returned; if return_softmax_lse=True, the float32 log-sum-exp result is returned.
  - cu_seqlens_q, cu_seqlens_ori_kv, and cu_seqlens_cmp_kv must start with 0 and be non-decreasing; that is, each element is not smaller than its previous element.
  - When layout_kv is PA_BBND, ori_kv and cmp_kv support a non-contiguous axis 0.
  - For dimensions denoted by the same symbol in the shapes of these parameters, the actual values of the corresponding axes must be consistent.

### Feature Parameter Group

|      Feature Parameter Group      |     Parameter Field Name     |
| :-------------------: | :-------------------: |
|      Common Parameter Group      | q, quant_mode, ori_kv, cmp_kv, q_descale, ori_kv_descale, cmp_kv_descale, metadata, softmax_scale, layout_q, layout_kv, attention_out |
|      Mask Parameter Group      | ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right |
|   SeqLens Parameter Group   | cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_q, seqused_ori_kv, seqused_cmp_kv |
|   Sparse Compression Parameter Group    | cmp_ratio, cmp_residual_kv, ori_sparse_indices, cmp_sparse_indices, ori_topk_length, cmp_topk_length, topk_value_mode |
| Paged Attention Parameter Group | ori_block_table, cmp_block_table |
|   Sinks Parameter Group   | sinks |
|   SoftmaxLse Parameter Group   | return_softmax_lse, softmax_lse |

### Computation Modes

|    Name    |    Parameters Required for Typical Use Cases    |    Parameters Required for Fully Sparse Use Cases    |
| :---------: | :---------------------------------------------------------: | :---------------------------------------------------------: |
|      SWA      | ori_kv | ori_kv, ori_sparse_indices |
|      HCA      | ori_kv, cmp_kv|-|
|      CSA      | ori_kv, cmp_kv, cmp_sparse_indices| ori_kv, ori_sparse_indices, cmp_kv, cmp_sparse_indices|

### Parameter Group Constraints

#### Common Parameter Group

- Handling of empty input parameters:
  - An empty tensor is a required input, or an input or output required in a specific use case, whose shape size is 0, that is, any axis is 0.
  - Use cases that trigger an empty tensor are all intercepted and an error is reported.

- Validation of q, ori_kv, cmp_kv, and attention_out

<table style="undefined;table-layout: fixed; width:1625px"><colgroup>
<col style="width: 147px">
<col style="width: 232px">
<col style="width: 232px">
<col style="width: 293px">
<col style="width: 185px">
</colgroup>
<thead>
<tr>
    <th>Parameter</th>
    <th>Single Parameter Validation</th>
    <th>Existence Validation</th>
    <th>Consistency Validation</th>
    <th>Cross-Feature Validation</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>q</td>
        <td>            <ul>
                <li>The dtype is hifloat8.</li>
                <li>When layout_q is BSND, the shape of q is (b, q_s, q_n, q_d).</li>
                <li>When layout_q is TND, the shape of q is (q_t, q_n, q_d).</li>
            </ul>
        </td>
        <td>            Required
        </td>
        <td rowspan="4">            <ul>
                <li>The shapes of q and attention_out must be the same.</li>
                <li>If cmp_kv is passed in, ori_kv and cmp_kv must have the same dtype.</li>
                <li>If layout_kv is not PA_BBND, layout_q and layout_kv must be the same.</li>
                <li>If layout_kv is PA_BBND, layout_q can be BSND or TND.</li>
                <li>If quant_mode is 1, the dtypes of q, ori_kv, and cmp_kv must be hifloat8.</li>
            </ul>
        </td>
        <td rowspan="4">            Axis constraints:
            <ul>
                <li>b > 0</li>
                <li>q_s > 0</li>
                <li>0 < q_n <= 128</li>
                <li>q_d = 512</li>
                <li>q_t > 0</li>
                <li>ori_kv_s > 0</li>
                <li>cmp_kv_s > 0</li>
                <li>kv_n = 1</li>
                <li>kv_d = 512</li>
                <li>ori_kv_t > 0</li>
                <li>cmp_kv_t > 0</li>
                <li>ori_kv_block_nums > 0</li>
                <li>cmp_kv_block_nums > 0</li>
                <li>1 <= ori_kv_block_size <= 1024</li>
                <li>1 <= cmp_kv_block_size <= 1024</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>ori_kv</td>
        <td>            <ul>
                <li>The dtype is hifloat8.</li>
                <li>When layout_kv is BSND, the shape of ori_kv is (b, ori_kv_s, kv_n, kv_d).</li>
                <li>When layout_kv is TND, the shape of ori_kv is (ori_kv_t, kv_n, kv_d).</li>
                <li>When layout_kv is PA_BBND, the shape of ori_kv is (ori_kv_block_nums, ori_kv_block_size, kv_n, kv_d).</li>
            </ul>
        </td>
        <td>            Required in the current version
        </td>
    </tr>
    <tr>
        <td>attention_out</td>
        <td>            <ul>
                <li>The dtype is bfloat16.</li>
                <li>When layout_q is BSND, the shape of attention_out is (b, q_s, q_n, q_d).</li>
                <li>When layout_q is TND, the shape of attention_out is (q_t, q_n, q_d).</li>
            </ul>
        </td>
        <td>            Required
        </td>
    </tr>
    <tr>
        <td>cmp_kv</td>
        <td>            <ul>
                <li>dtype supports hifloat8.</li>
                <li>When layout_kv is BSND, the shape of cmp_kv is (b, cmp_kv_s, kv_n, kv_d).</li>
                <li>When layout_kv is TND, the shape of cmp_kv is (cmp_kv_t, kv_n, kv_d).</li>
                <li>When layout_kv is PA_BBND, the shape of cmp_kv is (cmp_kv_block_nums, cmp_kv_block_size, kv_n, kv_d).</li>
            </ul>
        </td>
        <td>            Optional input
        </td>
    </tr>
    <tr>
        <td>q_descale</td>
        <td>            <ul>
                <li>dtype supports float32.</li>
            </ul>
        </td>
        <td>            Required in the current version
        </td>
        <td rowspan="3">            <ul>
                <li>When quant_mode=1, the shape is [1].</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>ori_kv_descale</td>
        <td>            <ul>
                <li>dtype supports float32.</li>
            </ul>
        </td>
        <td>            Required in the current version
        </td>
    </tr>
    <tr>
        <td>cmp_kv_descale</td>
        <td>            <ul>
                <li>dtype: float32</li>
            </ul>
        </td>
        <td>            <ul>
                <li>Mandatory when cmp_kv is passed in.</li>
                <li>Not supported when cmp_kv is empty.</li>
            </ul>
        </td>
    </tr>
</tbody>
</table>
Layout matching table:
<table style="undefined;table-layout: fixed; width:1625px"><colgroup>
<col style="width: 400px">
<col style="width: 400px">
</colgroup>
<thead>
<tr>
    <th>layout_q</th>
    <th>layout_kv</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>BSND</td>
        <td>          <li>BSND</li>
          <li>PA_BBND</li>
        </td>
    </tr>
    <tr>
        <td>TND</td>
        <td>          <li>TND</li>
          <li>PA_BBND</li>
        </td>
    </tr>
</tbody>
</table>

Metadata validation
<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Single-Parameter Validation</th>
            <th>Existence Validation</th>
            <th>Consistency Validation</th>
            <th>Feature Cross-Validation</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>metadata</td>
            <td>                <ul>
                    <li>dtype supports int32 only.</li>
                    <li>shape is dynamically calculated by quant_sparse_flash_mla_metadata.</li>
                </ul>
            </td>
            <td>Required in the current version.</td>
            <td>None</td>
            <td>If specified, must be consistent with the result generated by quant_sparse_flash_mla_metadata.</td>
        </tr>
    </tbody>
</table>

#### Mask Parameter Group

<ul>
    <li>ori_mask_mode/cmp_mask_mode=0, full computation mode (default value)</li>
    <li>ori_mask_mode/cmp_mask_mode=3, Causal mode</li>
    <li>ori_mask_mode=4, SlidingWindow mode</li>
</ul>

<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 200px">
        <col style="width: 100px">
        <col style="width: 200px">
    </colgroup>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Single-Parameter Validation</th>
            <th>Existence Validation</th>
            <th>Consistency Validation</th>
            <th>Feature Cross-Validation</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>ori_mask_mode</td>
            <td>                <ul>
                    <li>Only int32 is supported for dtype.</li>
                    <li>Only 0, 3, and 4 are supported. The default value is 0.</li>
                </ul>
            </td>
            <td>                Optional. If this parameter is not passed, the default value is 0.
            </td>
            <td>                <ul>
                    <li>None</li>
                </ul>
            </td>
            <td>                <ul>
                     <li>In SWA use cases, ori_mask_mode is 0, 3, or 4.</li>
                 </ul>
             </td>
         </tr>
         <tr>
             <td>cmp_mask_mode</td>
             <td>                 <ul>
                     <li>dtype: int32</li>
                     <li>Only 0 and 3 are supported. Default value: 0.</li>
                 </ul>
             </td>
             <td>                 Optional. If this parameter is not passed, the default value is 0.
             </td>
             <td>                 <ul>
                     <li>None</li>
                 </ul>
             </td>
             <td>                <ul>
                    <li>In SWA use cases, cmp_mask_mode must be 0. </li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>ori_win_left/ori_win_right</td>
            <td>                <ul>
                    <li>dtype: int32</li>
                    <li>Only -1 or a non-negative integer is supported.</li>
                </ul>
            </td>
            <td>                Optional. If this parameter is not passed, the default value is -1.
            </td>
            <td>                <ul>
                    None
                </ul>
            </td>
            <td>                <ul>
                    <li>ori_win_left/ori_win_right can be >=0 only when ori_mask_mode is 4.</li>
                </ul>
            </td>
        </tr>
    </tbody>
</table>

#### SeqLens Parameter Group

<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Single-Parameter Validation</th>
            <th>Existence Validation</th>
            <th>Consistency Validation</th>
            <th>Feature Cross-Validation</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>seqused_q</td>
            <td>                <ul>
                    <li>dtype supports int32.</li>
                    <li>shape is (b,).</li>
                    <li>Values must be non-negative integers.</li>
                    <li>Values in seqused_q must be less than or equal to q_s.</li>
                </ul>
            </td>
            <td>None</td>
            <td>None</td>
            <td>None</td>
        </tr>
        <tr>
            <td>seqused_ori_kv</td>
            <td>                <ul>
                    <li>dtype supports int32.</li>
                    <li>shape is (b,).</li>
                    <li>Only non-negative integers are supported.</li>
                    <li>Values in seqused_ori_kv must be less than or equal to ori_kv_s.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Optional when layout_kv is BSND.</li>
                    <li>Required when layout_kv is PA_BBND.</li>
                    <li>Can be omitted when ori_topk_length is provided.</li>
                </ul>
            </td>
            <td>None</td>
            <td>None</td>
        </tr>
        <tr>
            <td>seqused_cmp_kv</td>
            <td>                <ul>
                    <li>dtype supports int32.</li>
                    <li>shape is (b,).</li>
                    <li>Only non-negative integers are supported.</li>
                    <li>Values in seqused_cmp_kv must be less than or equal to kv_s.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Optional when layout_kv is BSND.</li>
                    <li>Required when layout_kv is PA_BBND and cmp_kv is provided.</li>
                    <li>Can be omitted when cmp_topk_length is provided.</li>
                </ul>
            </td>
            <td>None</td>
            <td>None</td>
        </tr>
        <tr>
            <td>cu_seqlens_q</td>
            <td>                <ul>
                    <li>dtype: int32.</li>
                    <li>shape: (b+1,).</li>
                    <li>Only non-negative integers are supported.</li>
                    <li>Values must be non-decreasing (greater than or equal to the previous value). The first element is 0 and the last element equals q_t.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Required when layout_q is TND.</li>
                    <li>Not supported when layout_q is not TND.</li>
                </ul>
            </td>
            <td>None</td>
            <td>None</td>
        </tr>
        <tr>
            <td>cu_seqlens_ori_kv</td>
            <td>                <ul>
                    <li>dtype: int32.</li>
                    <li>shape: (b+1,).</li>
                    <li>Only non-negative integers are supported.</li>
                    <li>Values must be non-decreasing (greater than or equal to the previous value). The first element is 0 and the last element equals ori_kv_t.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Required when layout_kv is TND.</li>
                    <li>Not supported when layout_kv is not TND.</li>
                </ul>
            </td>
            <td>None</td>
            <td>None</td>
        </tr>
        <tr>
            <td>cu_seqlens_cmp_kv</td>
            <td>                <ul>
                    <li>dtype supports int32.</li>
                    <li>shape is (b+1,).</li>
                    <li>Only non-negative integers are supported.</li>
                    <li>The values must be arranged in non-descending order (each value is greater than or equal to the preceding one). The first element is 0, and the last element equals kv_t.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Must be passed in when layout_kv is TND.</li>
                    <li>Passing in is not supported when layout_kv is not TND.</li>
                </ul>
            </td>
            <td>None</td>
            <td>None</td>
        </tr>
    </tbody>
</table>

#### Sparse Compression Parameter Group

<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Single-Parameter Validation</th>
            <th>Existence Validation</th>
            <th>Consistency Validation</th>
            <th>Feature Cross-Validation</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>cmp_ratio</td>
            <td>                <ul>
                    <li>data_type supports int32.</li>
                    <li>Represents the compression ratio of cmp_kv relative to the KV length before compression. The value must be greater than 0.</li>
                    <li>The default value is 1.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Optional. The default value is 1.</li>
                </ul>
            </td>
            <td>None</td>
            <td>                <ul>
                    <li>In the typical SWA use case, only the default value 1 is supported.</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>cmp_residual_kv</td>
            <td>                <ul>
                    <li>dtype supports int32.</li>
                    <li>The shape is (b,)</li>
                    <li>Only non-negative integers are supported.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Validated only when cmp_kv is passed in.</li>
                    <li>Optional</li>
                    <li>Required when cmp_mask_mode=3 and cmp_ratio!=1.</li>
                    <li>Not allowed when cmp_mask_mode=0 or cmp_ratio=1.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>None</li>
                </ul>
            </td>
            <td>None</td>
        </tr>
        <tr>
            <td>ori_sparse_indices</td>
            <td>                <ul>
                    <li>dtype supports int32.</li>
                    <li>The shape is (q_t, kv_n, ori_kv_k) or (b, q_s, kv_n, ori_kv_k).</li>
                    <li>Invalid positions are filled with -1, and others are non-negative integers.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Optional</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>When layout_q is TND, the shape is (q_t, kv_n, ori_kv_k).</li>
                    <li>When layout_q is BSND, the shape is (b, q_s, kv_n, ori_kv_k).</li>
                </ul>
            </td>
            <td>None</td>
        </tr>
        <tr>
            <td>cmp_sparse_indices</td>
            <td>                <ul>
                    <li>Supports int32 dtype.</li>
                    <li>The shape is (q_t, kv_n, cmp_kv_k) or (b, q_s, kv_n, cmp_kv_k).</li>
                    <li>Invalid positions are filled with -1, and others are non-negative integers.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Validated only when cmp_kv is passed in.</li>
                    <li>Optional.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>When layout_q is TND, the shape is (q_t, kv_n, cmp_kv_k).</li>
                    <li>When layout_q is BSND, the shape is (b, q_s, kv_n, cmp_kv_k).</li>
                </ul>
            </td>
            <td>None</td>
        </tr>
        <tr>
            <td>ori_topk_length</td>
            <td>                <ul>
                    <li>Supports int32 dtype.</li>
                    <li>The shape is (b, q_s, kv_n) or (q_t, kv_n).</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Required when ori_mask_mode=0 and ori_sparse_indices is not empty.</li>
                    <li>Not supported in other use cases.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>When layout_q is TND, the shape is (q_t, kv_n).</li>
                    <li>When layout_q is BSND, the shape is (b, q_s, kv_n).</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Not supported when ori_mask_mode is not 0</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>cmp_topk_length</td>
            <td>                <ul>
                    <li>dtype supports int32</li>
                    <li>shape is (b, q_s, kv_n) or (q_t, kv_n)</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Verified only when cmp_kv is passed in</li>
                    <li>Must be passed in when cmp_mask_mode=0 and cmp_sparse_indices is not empty</li>
                    <li>Not supported in other use cases</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>When layout_q is TND, the shape is (q_t, kv_n)</li>
                    <li>When layout_q is BSND, the shape is (b, q_s, kv_n)</li>
                </ul>
            </td>
            <td>None</td>
        </tr>
        <tr>
            <td>topk_value_mode</td>
            <td>                <ul>
                    <li>data_type supports int32</li>
                    <li>topK index value mode. Default value: 1</li>
                </ul>
            </td>
            <td>Optional attribute. Default value: 1</td>
            <td>None</td>
            <td>None</td>
        </tr>
    </tbody>
</table>

#### Paged Attention Parameter Group

Paged Attention is enabled when layout_kv is PA_BBND.
<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Single-Parameter Validation</th>
            <th>Existence Validation</th>
            <th>Consistency Validation</th>
            <th>Feature Cross-Validation</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>ori_block_table</td>
            <td>                <ul>
                    <li>dtype supports int32 only.</li>
                    <li>shape is (b, ceil(ori_kv_s_max/ori_kv_block_size)).</li>
                    <li>values must be positive integers.</li>
                </ul>
            </td>
            <td>Optional</td>
            <td>None</td>
            <td>                <ul>
                    <li>If ori_block_table exists, seqused_ori_kv must be passed in.</li>
                    <li>block_table must not be empty when PagedAttention is enabled.</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>cmp_block_table</td>
            <td>                <ul>
                    <li>Only int32 is supported for dtype.</li>
                    <li>The shape is (b, cmp_kv_s_max/cmp_kv_block_size).</li>
                    <li>Values must be positive integers.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Validated only when cmp_kv is passed in.</li>
                    <li>Optional</li>
                </ul>
            </td>
            <td>None</td>
            <td>                <ul>
                    <li>If cmp_block_table exists, seqused_cmp_kv must be passed in.</li>
                    <li>block_table must not be empty when PagedAttention is enabled.</li>
                </ul>
            </td>
        </tr>
    </tbody>
</table>

#### Sinks Parameter Group

<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Single-Parameter Validation</th>
            <th>Existence Validation</th>
            <th>Consistency Validation</th>
            <th>Feature Cross-Validation</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>sinks</td>
            <td>                <ul>
                    <li>dtype is float32.</li>
                    <li>shape is (q_n, ).</li>
                </ul>
            </td>
            <td> Required in the current version </td>
            <td> None </td>
            <td> None </td>
        </tr>
    </tbody>
</table>

#### SoftmaxLse Parameter Group

<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Single-Parameter Validation</th>
            <th>Existence Validation</th>
            <th>Consistency Validation</th>
            <th>Feature Cross-Validation</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>return_softmax_lse</td>
            <td>                <ul>
                    <li>data_type supports only bool.</li>
                    <li>true enables softmax_lse, and false disables softmax_lse.</li>
                </ul>
            </td>
            <td>Optional, default value: false</td>
            <td rowspan="2">                <ul>
                     <li>If return_softmax_lse is false, the output is a tensor with shape [1] and value 0.</li>
                    <li>If return_softmax_lse is true, the relationship between the shape of softmax_lse and layout_q is as follows:<ul><li>If layout_q is BSND, the shape of softmax_lse is (b, kv_n, q_s, q_n/kv_n).</li><li>If layout_q is TND, the shape of softmax_lse is (kv_n, q_t, q_n/kv_n).</li></ul></li>
                </ul>
            </td>
            <td>None</td>
        </tr>
        <tr>
            <td>softmax_lse</td>
            <td>                <ul>
                    <li>data_type only supports float32.</li>
                </ul>
            </td>
            <td>None</td>
            <td>None</td>
        </tr>
    </tbody>
</table>

## Example

### SWA, BSND Input

```python
import torch
import torch_npu
import math
import cann_ops_transformer

torch_npu.npu.set_device(0)

qDtype = torch.uint8 # Use uint8 to disguise hifloat8.
kvDtype = torch.uint8
B = 1
S1 = 16
S2 = 64
N1 = 64
N2 = 1
Dq = 512
Dkv = 512
quant_mode = 1
cmp_ratio = 1  # In the SWA example, only ori_kv is passed. cmp_ratio does not participate in compressed KV computation, so keep the default value 1.

seqused_q = torch.tensor([16], dtype=torch.int32, device="npu")
seqused_ori_kv = torch.tensor([64], dtype=torch.int32, device="npu")
q = torch.full((B, S1, N1, Dq), 8, dtype=qDtype, device="npu")  # 8 in uint8 equals 1.0 in hifloat8.
ori_kv = torch.full((B, S2, N2, Dkv), 8 ,dtype=kvDtype, device="npu")
cmp_kv = None
q_descale = torch.tensor([2.0], dtype=torch.float32, device="npu")
ori_kv_descale = torch.tensor([2.0], dtype=torch.float32, device="npu")
cmp_kv_descale = None
sinks = torch.zeros(N1, dtype=torch.float32, device="npu")

layout_q = "BSND"
layout_kv = "BSND"

metadata = torch.ops.cann_ops_transformer.quant_sparse_flash_mla_metadata(
    num_heads_q = N1,
    num_heads_kv = N2,
    head_dim = Dq,
    seqused_q=seqused_q,
    seqused_ori_kv=seqused_ori_kv,
    quant_mode=quant_mode,
    batch_size = B,
    max_seqlen_q = max(seqused_q),
    max_seqlen_ori_kv = max(seqused_ori_kv),
    ori_topk = 0,
    cmp_ratio = cmp_ratio,
    ori_mask_mode = 4,
    cmp_mask_mode = 0,
    ori_win_left = 127,
    ori_win_right = 0,
    layout_q = layout_q,
    layout_kv = layout_kv,
    has_ori_kv = ori_kv is not None,
    has_cmp_kv = cmp_kv is not None,
    )

npu_result, _ = torch.ops.cann_ops_transformer.quant_sparse_flash_mla(
    q=q,
    ori_kv=ori_kv,
    cmp_kv=cmp_kv,
    q_descale = q_descale,
    ori_kv_descale = ori_kv_descale,
    cmp_kv_descale = cmp_kv_descale,
    seqused_q=seqused_q,
    seqused_ori_kv=seqused_ori_kv,
    sinks=sinks,
    metadata=metadata,
    quant_mode=quant_mode,
    softmax_scale=1.0 / math.sqrt(Dq),
    cmp_ratio=cmp_ratio,
    ori_mask_mode=4,
    cmp_mask_mode=0,
    ori_win_left=127,
    ori_win_right=0,
    layout_q=layout_q,
    layout_kv=layout_kv,
    topk_value_mode=1,
    return_softmax_lse=False)

torch.npu.synchronize()
```

### HCA, LayoutQ BSND, LayoutKv PA_BBND Input

```python
import torch
import torch_npu
import math
import cann_ops_transformer

torch_npu.npu.set_device(0)

def scatter_to_pa(kv_bnsd, block_size, block_num, max_block_num_per_batch):
    B, N2, S, D = kv_bnsd.shape
    kv_expand = torch.zeros(B, N2, max_block_num_per_batch * block_size, D, dtype=kv_bnsd.dtype, device=kv_bnsd.device)
    kv_expand[:, :, :S, :] = kv_bnsd
    kv_pa = torch.zeros(block_num, block_size, N2, D, dtype=kv_bnsd.dtype, device=kv_bnsd.device)
    block_table = torch.zeros(B, max_block_num_per_batch, dtype=torch.int32, device=kv_bnsd.device)
    cur_block_id = 0
    for i_B in range(B):
        for i_block in range(max_block_num_per_batch):
            block_table[i_B, i_block] = cur_block_id
            block_start = i_block * block_size
            for i_N2 in range(N2):
                kv_pa[cur_block_id, :, i_N2, :] = kv_expand[i_B, i_N2, block_start:block_start + block_size, :]
            cur_block_id += 1
    return kv_pa, block_table

qDtype = torch.uint8 # Use uint8 to disguise hifloat8.
kvDtype = torch.uint8
B = 1
S1 = 16
S2 = 2050
N1 = 64
N2 = 1
Dq = 512
Dkv = 512
quant_mode = 1
cmp_ratio = 128

block_size1 = 128
block_size2 = 128
ori_max_block_num_per_batch = math.ceil(S2 / block_size1)
block_num1 = ori_max_block_num_per_batch * B

seqused_ori_kv = torch.tensor([S2], dtype=torch.int32, device="npu")
cmp_max_s2 = S2 // cmp_ratio
seqused_cmp_kv = torch.tensor([cmp_max_s2], dtype=torch.int32, device="npu")
cmp_residual_kv = torch.tensor([S2 % cmp_ratio], dtype=torch.int32, device="npu")
seqused_q = torch.tensor([16], dtype=torch.int32, device="npu")

q = torch.full((B, S1, N1, Dq), 8, dtype=qDtype, device="npu") # 8 in uint8 equals 1.0 in hifloat8.
ori_kv_bnsd = torch.full((B, N2, S2, Dkv), 8, dtype=kvDtype, device="npu")
ori_kv, ori_block_table = scatter_to_pa(ori_kv_bnsd, block_size1, block_num1, ori_max_block_num_per_batch)

if cmp_max_s2 > 0:
    cmp_max_block_num_per_batch = math.ceil(cmp_max_s2 / block_size2)
    block_num2 = cmp_max_block_num_per_batch * B
    cmp_kv_bnsd = torch.full((B, N2, cmp_max_s2, Dkv), 8, dtype=kvDtype, device="npu")
    cmp_kv, cmp_block_table = scatter_to_pa(cmp_kv_bnsd, block_size2, block_num2, cmp_max_block_num_per_batch)
else:
    block_num2 = 0
    cmp_kv = None
    cmp_block_table = None

q_descale = torch.tensor([2.0], dtype=torch.float32, device="npu")
ori_kv_descale = torch.tensor([2.0], dtype=torch.float32, device="npu")
cmp_kv_descale = torch.tensor([2.0], dtype=torch.float32, device="npu")

sinks = torch.zeros(N1, dtype=torch.float32, device="npu")

layout_q = "BSND"
layout_kv = "PA_BBND"

metadata = torch.ops.cann_ops_transformer.quant_sparse_flash_mla_metadata(
    num_heads_q = N1,
    num_heads_kv = N2,
    head_dim = Dq,
    seqused_q=seqused_q,
    seqused_ori_kv=seqused_ori_kv,
    seqused_cmp_kv=seqused_cmp_kv,
    cmp_residual_kv=cmp_residual_kv,
    quant_mode=quant_mode,
    batch_size = B,
    max_seqlen_q = max(seqused_q),
    max_seqlen_ori_kv = max(seqused_ori_kv),
    max_seqlen_cmp_kv = max(seqused_cmp_kv),
    ori_topk = 0,
    cmp_topk = 0,
    cmp_ratio = cmp_ratio,
    ori_mask_mode = 4,
    cmp_mask_mode = 3,
    ori_win_left = 127,
    ori_win_right = 0,
    layout_q = layout_q,
    layout_kv = layout_kv,
    has_ori_kv = ori_kv is not None,
    has_cmp_kv = cmp_kv is not None,
    )

npu_result, _ = torch.ops.cann_ops_transformer.quant_sparse_flash_mla(
    q=q,
    ori_kv=ori_kv,
    cmp_kv=cmp_kv,
    q_descale=q_descale,
    ori_kv_descale=ori_kv_descale,
    cmp_kv_descale=cmp_kv_descale,
    ori_block_table=ori_block_table,
    cmp_block_table=cmp_block_table,
    seqused_q=seqused_q,
    seqused_ori_kv=seqused_ori_kv,
    seqused_cmp_kv=seqused_cmp_kv,
    cmp_residual_kv=cmp_residual_kv,
    sinks=sinks,
    metadata=metadata,
    quant_mode=quant_mode,
    softmax_scale=1.0 / math.sqrt(Dq),
    cmp_ratio=cmp_ratio,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=127,
    ori_win_right=0,
    layout_q=layout_q,
    layout_kv=layout_kv,
    topk_value_mode=1,
    return_softmax_lse=False)

torch.npu.synchronize()
```

### CSA, TND Input

```python
import torch
import torch_npu
import math
import cann_ops_transformer
import torchair
from torchair.configs.compiler_config import CompilerConfig

torch_npu.npu.set_device(0)

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

    def forward(self, q, ori_kv, cmp_kv, q_descale, ori_kv_descale, cmp_kv_descale,
                cmp_sparse_indices, cu_seqlens_q, seqused_q, cu_seqlens_ori_kv, seqused_ori_kv,
                cu_seqlens_cmp_kv, seqused_cmp_kv, cmp_residual_kv,
                sinks, metadata, quant_mode, softmax_scale, cmp_ratio,
                ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right,
                layout_q, layout_kv, topk_value_mode, return_softmax_lse):
        npu_result, _ = torch.ops.cann_ops_transformer.quant_sparse_flash_mla(
            q=q,
            ori_kv=ori_kv,
            cmp_kv=cmp_kv,
            q_descale=q_descale,
            ori_kv_descale=ori_kv_descale,
            cmp_kv_descale=cmp_kv_descale,
            cmp_sparse_indices=cmp_sparse_indices,
            cu_seqlens_q=cu_seqlens_q,
            seqused_q=seqused_q,
            cu_seqlens_ori_kv=cu_seqlens_ori_kv,
            seqused_ori_kv=seqused_ori_kv,
            cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
            seqused_cmp_kv=seqused_cmp_kv,
            cmp_residual_kv=cmp_residual_kv,
            sinks=sinks,
            metadata=metadata,
            quant_mode=quant_mode,
            softmax_scale=softmax_scale,
            cmp_ratio=cmp_ratio,
            ori_mask_mode=ori_mask_mode,
            cmp_mask_mode=cmp_mask_mode,
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right,
            layout_q=layout_q,
            layout_kv=layout_kv,
            topk_value_mode=topk_value_mode,
            return_softmax_lse=return_softmax_lse)
        return npu_result

qDtype = torch.uint8 # Use uint8 to disguise hifloat8
kvDtype = torch.uint8
B = 1
S1 = 16
S2 = 2050
N1 = 64
N2 = 1
Dq = 512
Dkv = 512
K = 512
quant_mode = 1
cmp_ratio = 128

q_seqlens = [S1] * B
ori_kv_seqlens = [S2] * B

cu_seqlens_q = torch.tensor([sum(q_seqlens[:i]) for i in range(B + 1)], dtype=torch.int32, device="npu")
cu_seqlens_ori_kv = torch.tensor([sum(ori_kv_seqlens[:i]) for i in range(B + 1)], dtype=torch.int32, device="npu")
cu_seqlens_cmp_kv = torch.tensor([sum(s // cmp_ratio for s in ori_kv_seqlens[:i]) for i in range(B + 1)], dtype=torch.int32, device="npu")

total_q = cu_seqlens_q[-1].item()
total_ori_kv = cu_seqlens_ori_kv[-1].item()
total_cmp_kv = cu_seqlens_cmp_kv[-1].item()

seqused_q = torch.tensor(q_seqlens, dtype=torch.int32, device="npu")
seqused_ori_kv = torch.tensor(ori_kv_seqlens, dtype=torch.int32, device="npu")
seqused_cmp_kv = torch.tensor([s // cmp_ratio for s in ori_kv_seqlens], dtype=torch.int32, device="npu")
cmp_residual_kv = torch.tensor([s % cmp_ratio for s in ori_kv_seqlens], dtype=torch.int32, device="npu")

q = torch.full((total_q, N1, Dq), 8, dtype=qDtype, device="npu") # 8 in uint8 equals 1.0 in hifloat8
ori_kv = torch.full((total_ori_kv, N2, Dkv), 8, dtype=kvDtype, device="npu")
cmp_kv = torch.full((total_cmp_kv, N2, Dkv), 8, dtype=kvDtype, device="npu")
q_descale = torch.tensor([2.0], dtype=torch.float32, device="npu")
ori_kv_descale = torch.tensor([2.0], dtype=torch.float32, device="npu")
cmp_kv_descale = torch.tensor([2.0], dtype=torch.float32, device="npu")

cmp_sparse_indices = torch.full((total_q, N2, K), fill_value=-1, dtype=torch.int32, device="npu")
for i_B in range(B):
    cur_act_q = seqused_q[i_B].item()
    s1_prefix = cu_seqlens_q[i_B].item()
    cur_ori_kv = seqused_ori_kv[i_B].item()
    for i_N2 in range(N2):
        for i_S1 in range(cur_act_q):
            cur_valid_s2_max = math.floor((cur_ori_kv - cur_act_q + i_S1 + 1) / cmp_ratio)
            valid_blocks_max = max(0, cur_valid_s2_max)
            block_indices = torch.randperm(valid_blocks_max, device="npu").to(torch.int32)
            valid_blocks_topk = min(valid_blocks_max, K)
            cmp_sparse_indices[s1_prefix + i_S1, i_N2, :valid_blocks_topk] = block_indices[0:valid_blocks_topk]

sinks = torch.zeros(N1, dtype=torch.float32, device="npu")

layout_q = "TND"
layout_kv = "TND"

print("quant_sparse_flash_mla_metadata...")
metadata = torch.ops.cann_ops_transformer.quant_sparse_flash_mla_metadata(
    num_heads_q = N1,
    num_heads_kv = N2,
    head_dim = Dq,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_ori_kv=cu_seqlens_ori_kv,
    cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
    seqused_q=seqused_q,
    seqused_ori_kv=seqused_ori_kv,
    seqused_cmp_kv=seqused_cmp_kv,
    cmp_residual_kv=cmp_residual_kv,
    quant_mode=quant_mode,
    batch_size = B,
    max_seqlen_q = max(q_seqlens),
    max_seqlen_ori_kv = max(ori_kv_seqlens),
    max_seqlen_cmp_kv = max(s // cmp_ratio for s in ori_kv_seqlens),
    ori_topk = 0,
    cmp_topk = K,
    cmp_ratio = cmp_ratio,
    ori_mask_mode = 4,
    cmp_mask_mode = 3,
    ori_win_left = 127,
    ori_win_right = 0,
    layout_q = layout_q,
    layout_kv = layout_kv,
    has_ori_kv = ori_kv is not None,
    has_cmp_kv = cmp_kv is not None,
    )
torch.npu.synchronize()
metadata.npu()

print("torch.compile...")
torch._dynamo.reset()
npu_mode = Network().npu()
config = CompilerConfig()
config.mode = "reduce-overhead"
config.experimental_config.aclgraph._aclnn_static_shape_kernel = True
config.experimental_config.frozen_parameter = True
npu_backend = torchair.get_npu_backend(compiler_config=config)
npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)

print("quant_sparse_flash_mla (graph)...")
npu_result = npu_mode(
    q=q,
    ori_kv=ori_kv,
    cmp_kv=cmp_kv,
    q_descale=q_descale,
    ori_kv_descale=ori_kv_descale,
    cmp_kv_descale=cmp_kv_descale,
    cmp_sparse_indices=cmp_sparse_indices,
    cu_seqlens_q=cu_seqlens_q,
    seqused_q=seqused_q,
    cu_seqlens_ori_kv=cu_seqlens_ori_kv,
    seqused_ori_kv=seqused_ori_kv,
    cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
    seqused_cmp_kv=seqused_cmp_kv,
    cmp_residual_kv=cmp_residual_kv,
    sinks=sinks,
    metadata=metadata,
    quant_mode=quant_mode,
    softmax_scale=1.0 / math.sqrt(Dq),
    cmp_ratio=cmp_ratio,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=127,
    ori_win_right=0,
    layout_q=layout_q,
    layout_kv=layout_kv,
    topk_value_mode=1,
    return_softmax_lse=False)

torch.npu.synchronize()
```
