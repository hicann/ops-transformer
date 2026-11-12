# mixed_quant_sparse_flash_mla

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T11:29:18.862Z -->

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
  `mixed_quant_sparse_flash_mla` is an extended `torch` API of `cann_ops_transformer`, used to call the `MixedQuantSparseFlashMla` operator to perform MLA (Multi-head Latent Attention) computation in quantization and sparse use cases, unifying training and inference. The operator has three typical use cases:

  - **SWA (Sliding Window Attention)**: Passes in only `ori_kv` to perform sliding window attention on the original KV.
  - **CSA (Compressed Sparse Attention)**: Passes in `ori_kv`, `cmp_kv`, and `cmp_sparse_indices` to perform attention on both the original KV window and the compressed KV selected by topK.
  - **HCA (Heavily Compressed Attention)**: Both `ori_kv` and `cmp_kv` are passed in, and attention is computed on both the raw KV window and the continuously compressed KV segments.

  `mixed_quant_sparse_flash_mla_metadata` provides the core splitting information for `mixed_quant_sparse_flash_mla` and is generated before the main operator is executed. In the current version, this metadata must be passed in to the main operator. The typical call flow is as follows:

  1. Prepare the corresponding inputs based on the use case.
  2. Call `mixed_quant_sparse_flash_mla_metadata` to generate `metadata`, which is passed in as an input parameter of `mixed_quant_sparse_flash_mla`.
  3. Call `mixed_quant_sparse_flash_mla` to generate the computation result.

- Formula:

  mixed_quant_sparse_flash_mla uses MLA to compute sparse attention on KV-shared inputs. The input KV is selectively compressed and quantized. The attention weights are then obtained by applying Softmax to the Query and the concatenated KV computation result.

  The MLA formula is generally defined as follows, where $\tilde{K}=\tilde{V}$ is the KV actually participating in the computation, controlled by input parameters. It consists of the sliding window part of `ori_kv` and the compressed part of `cmp_kv`. The range of the KV actually participating in the computation is determined by `cmp_ratio`, `ori_mask_mode`, `cmp_mask_mode`, `ori_win_left`, `ori_win_right`, and `cmp_sparse_indices`.

  $$
  O = \text{softmax}(Q @ \tilde{K}^T \cdot \text{softmax\_scale}) @  \tilde{V}
  $$

## Function Prototype

Before calling the mixed_quant_sparse_flash_mla API, call the prerequisite API mixed_quant_sparse_flash_mla_metadata first.

```python
cann_ops_transformer.mixed_quant_sparse_flash_mla_metadata(
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
    rope_head_dim=64,
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
cann_ops_transformer.mixed_quant_sparse_flash_mla(
    q,
    quant_mode,
    *,
    ori_kv=None,
    cmp_kv=None,
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
    rope_head_dim=64,
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

|    Name    |                            Meaning                            |
| :---------: | :---------------------------------------------------------: |
|      b      |      Batch size of input samples.                |
|     q_s     |      Sequence length of the input q.      |
|    ori_kv_s    |  Sequence length of the input ori_kv.  |
|    cmp_kv_s    |  Sequence length of the input cmp_kv.  |
|     q_n     |        Number of heads of the input q.        |
|    kv_n    |    Number of heads of the input ori_kv/cmp_kv.    |
|      q_d      |          Attention head dimension of the input q.         |
|      kv_d      |          Attention head dimension of the input ori_kv/cmp_kv.         |
|     q_t     |          Sum of sequence lengths of all batches of the input q.          |
|     ori_kv_t    |          Sum of sequence lengths of all batches of the input ori_kv.          |
|     cmp_kv_t    |          Sum of sequence lengths of all batches of the input cmp_kv.          |
|      ori_kv_k      |           Number of tokens selected by topK in the input ori_sparse_indices.         |
|      cmp_kv_k      |           Number of tokens selected by topK in the input cmp_sparse_indices.         |
|      ori_kv_s_max      |           Maximum sequence length of the input ori_kv.         |
|      cmp_kv_s_max      |           Maximum sequence length of the input cmp_kv.         |
|      ori_kv_block_size      |           Block size of the input ori_kv in PagedAttention use cases.         |
|      cmp_kv_block_size      |           Block size of the input cmp_kv in PagedAttention use cases.         |
|      ori_kv_block_nums      |           Number of blocks of the input ori_kv in PagedAttention use cases.         |
|      cmp_kv_block_nums      |           Number of blocks of the input cmp_kv in PagedAttention use cases.         |

### mixed_quant_sparse_flash_mla_metadata

| Name | Type | Type | Description | data type | data format | Dimensions |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| num_heads_q | int | Required | Indicates the number of q heads. | int32 | - | -
| num_heads_kv | int | Required | Indicates the number of ori_kv/cmp_kv heads. | int32 | - | -
| head_dim | int | Required | Indicates the dimension of each attention head. | int32 | - | -
| quant_mode | int | Required | Indicates the quantization mode. Currently only 1 and 2 are supported. When quant_mode is 1, the data is concatenated in the order of rope (64, bfloat16), nope (448, float8_e4m3fn), scale (7, bfloat16), and pad (18B). When quant_mode is 2, the data is concatenated in the order of nope (448, float8_e4m3fn), rope (64, bfloat16), scale (7, float8_e8m0), and pad (1B). | int32 | - | -
| cu_seqlens_q | tensor | Optional | Indicates the cumulative sequence lengths of variable-length sequences for input q. | int32 | ND | <ul><li>(b+1,)</li></ul>
| cu_seqlens_ori_kv | tensor | Optional | Indicates the cumulative sequence lengths of variable-length sequences for input ori_kv. | int32 | ND | <ul><li>(b+1,)</li></ul>
| cu_seqlens_cmp_kv | tensor | Optional | Indicates the cumulative sequence lengths of variable-length sequences for input cmp_kv. | int32 | ND | <ul><li>(b+1,)</li></ul>
| seqused_q | tensor | Optional | Indicates the sequence lengths actually participating in computation in each batch of input q. | int32 | ND | <ul><li>(b,)</li></ul>
| seqused_ori_kv | tensor | Optional | Indicates the sequence lengths actually participating in computation in each batch of input ori_kv. | int32 | ND | <ul><li>(b,)</li></ul>
| seqused_cmp_kv | tensor | Optional | Indicates the sequence lengths actually participating in computation in each batch of input cmp_kv. | int32 | ND | <ul><li>(b,)</li></ul>
| cmp_residual_kv | tensor | Optional | Indicates the remainder of the compressed sequence length of cmp_kv in each batch. | int32 | ND | <ul><li>(b,)</li></ul>
| ori_topk_length | tensor | Optional | Indicates the lengths of ori_sparse_indices actually participating in computation. | int32 | ND | <ul><li>(b, q_s, kv_n)</li><li>(q_t, kv_n)</li></ul>
| cmp_topk_length | tensor | Optional | Indicates the lengths of cmp_sparse_indices actually participating in computation. | int32 | ND | <ul><li>(b, q_s, kv_n)</li><li>(q_t, kv_n)</li></ul>
| batch_size | int | Optional | Indicates the batch size. | int32 | - | -
| max_seqlen_q | int | Optional | Indicates the maximum sequence length of query q. | int32 | - | -
| max_seqlen_ori_kv | int | Optional | Indicates the maximum sequence length of ori_kv. | int32 | - | -
| max_seqlen_cmp_kv | int | Optional | Indicates the maximum sequence length of cmp_kv. | int32 | - | -
| ori_topk | int | Optional | Indicates the number of key sparse tokens selected from ori_kv. 0 indicates a non-sparse use case. The default value is 0. | int32 | - | -
| cmp_topk | int | Optional | Indicates the number of key sparse tokens selected from cmp_kv. 0 indicates a non-sparse use case. The default value is 0. | int32 | - | -
| rope_head_dim | int | Optional | Indicates the dimension of the rope head. The default value is 64. | int32 | - | -
| cmp_ratio | int | Optional | Indicates the compression ratio of cmp_kv. The default value is 1. | int32 | - | -
| ori_mask_mode | int | Optional | Indicates the mask mode for the computation between q and ori_kv. 0: No mask. 3: rightDownCausal mode. 4: sliding window mode. The default value is 0. | int32 | - | -
| cmp_mask_mode | int | Optional | Indicates the mask mode for the computation between q and cmp_kv. 0: No mask. 3: rightDownCausal mode. The default value is 0. | int32 | - | -
| ori_win_left | int | Optional | Indicates the number of past tokens involved in the computation between q and ori_kv. -1 represents infinity, that is, all tokens participate in the computation. The default value is -1. | int32 | - | -
| ori_win_right | int | Optional | Indicates the number of future tokens involved in the computation between q and ori_kv. -1 represents infinity, that is, all tokens participate in the computation. The default value is -1. | int32 | - | -
| layout_q | string | Optional | Indicates the layout format of input q. The default value is BSND. | string | - | -
| layout_kv | string | Optional | Indicates the layout format of input ori_kv/cmp_kv. The default value is BSND. | string | - | -
| has_ori_kv | bool | Optional | Indicates whether ori_kv exists. The default value is True. | bool | - | -
| has_cmp_kv | bool | Optional | Indicates whether cmp_kv exists. The default value is True. | bool | - | -

### mixed_quant_sparse_flash_mla

| Name | Type | Type | Description | data type | data format | Dimension |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| q | tensor | Required | Represents q in the formula. | bfloat16 | ND | <ul><li>(b, q_s, q_n, q_d)</li><li>(q_t, q_n, q_d)</li></ul>
| quant_mode | int | Required | Represents the quantization mode. Currently, only 1 and 2 are supported. When quant_mode is 1, the quantized KV is concatenated in the following order: rope (64, bfloat16), nope (448, float8_e4m3fn), scale (7, bfloat16), and pad (18B). When quant_mode is 2, the quantized KV is concatenated in the following order: nope (448, float8_e4m3fn), rope (64, bfloat16), scale (7, float8_e8m0), and pad (1B). All quantization modes support UINT8 and FLOAT8_E4M3FN as the single-byte storage view, with the underlying byte content unchanged. | int32 | - | -
| ori_kv | tensor | Optional | Represents the original quantized KV input. Key and Value share the same data. | fp8_e4m3 | ND | <ul><li>(b, ori_kv_s, kv_n, kv_d)</li><li>(ori_kv_t, kv_n, kv_d)</li><li>(ori_kv_block_nums, ori_kv_block_size, kv_n, kv_d)</li></ul>
| cmp_kv | tensor | Optional | Represents the compressed quantized KV input. Key and Value share the same data. | fp8_e4m3 | ND | <ul><li>(b, cmp_kv_s, kv_n, kv_d)</li><li>(cmp_kv_t, kv_n, kv_d)</li><li>(cmp_kv_block_nums, cmp_kv_block_size, kv_n, kv_d)</li></ul>
| ori_sparse_indices | tensor | Optional | Represents the topK indices of the original KV. -1 is filled at invalid positions. | int32 | ND | <ul><li>(q_t, kv_n, ori_kv_k)</li><li>(b, q_s, kv_n, ori_kv_k)</li></ul>
| cmp_sparse_indices | tensor | Optional | Represents the topK indices of the compressed KV. -1 is filled at invalid positions. | int32 | ND | <ul><li>(q_t, kv_n, cmp_kv_k)</li><li>(b, q_s, kv_n, cmp_kv_k)</li></ul>
| ori_block_table | tensor | Optional | Represents the block mapping table used by ori_kv in the PagedAttention use case. | int32 | ND | <ul><li>(b, ceil(ori_kv_s_max/ori_kv_block_size))</li></ul>
| cmp_block_table | tensor | Optional | Represents the block mapping table used by cmp_kv in the PagedAttention use case. | int32 | ND | <ul><li>(b, ceil(cmp_kv_s_max/cmp_kv_block_size))</li></ul>
| cu_seqlens_q | tensor | Optional | Represents the cumulative sequence lengths for processing the variable-length sequences of input q. | int32 | ND | <ul><li>(b+1,)</li></ul>
| cu_seqlens_ori_kv | tensor | Optional | Represents the cumulative sequence lengths for processing the variable-length sequences of input ori_kv. | int32 | ND | <ul><li>(b+1,)</li></ul>
| cu_seqlens_cmp_kv | tensor | Optional | Represents the cumulative sequence lengths for processing the variable-length sequences of input cmp_kv. | int32 | ND | <ul><li>(b+1,)</li></ul>
| seqused_q | tensor | Optional | Represents the sequence length actually participating in computation for each batch of input q. | int32 | ND | <ul><li>(b,)</li></ul>
| seqused_ori_kv | tensor | Optional | Represents the sequence length actually participating in computation for each batch of input ori_kv. | int32 | ND | <ul><li>(b,)</li></ul>
| seqused_cmp_kv | tensor | Optional | Represents the sequence length actually participating in computation for each batch of input cmp_kv. | int32 | ND | <ul><li>(b,)</li></ul>
| cmp_residual_kv | tensor | Optional | Represents the remainder of the sequence length of cmp_kv after compression in each batch. Must be passed in when cmp_mask_mode=3 and cmp_ratio!=1; must not be passed in when cmp_mask_mode=0 or cmp_ratio=1. | int32 | ND | <ul><li>(b,)</li></ul>
| ori_topk_length | tensor | Optional | Represents the length of ori_sparse_indices actually participating in computation. | int32 | ND | <ul><li>(b, q_s, kv_n)</li><li>(q_t, kv_n)</li></ul>
| cmp_topk_length | tensor | Optional | Represents the length of cmp_sparse_indices actually participating in computation. | int32 | ND | <ul><li>(b, q_s, kv_n)</li><li>(q_t, kv_n)</li></ul>
| sinks | tensor | Optional | Represents the independent learnable virtual offset set for each attention head, used to maintain stability during long-text inference. | float32 | ND | <ul><li>(q_n,)</li></ul>
| metadata | tensor | Optional | Represents the core partitioning information generated by mixed_quant_sparse_flash_mla_metadata. | int32 | ND | <ul><li>(1024,)</li></ul>
| rope_head_dim | int | Optional | Represents the dimension of the rope head. The default value is 64. | int32 | - | -
| softmax_scale | float | Optional | Represents the scaling factor that can be explicitly set. The default value is 1.0. | float32 | - | -
| cmp_ratio | int | Optional | Represents the compression ratio of cmp_kv relative to the KV length before compression. It can restore the KV length before compression used by the cmp-side mask. The default value is 1, and the value range is 1-128.| int32 | - | -
| ori_mask_mode | int | Optional | Represents the mask mode for computation between q and ori_kv. 0: No mask. 3: rightDownCausal mode. 4: sliding window mode. The default value is 0. | int32 | - | -
| cmp_mask_mode | int | Optional | Represents the mask mode for computation between q and cmp_kv. 0: No mask. 3: rightDownCausal mode. The default value is 0. | int32 | - | -
| ori_win_left | int | Optional | Represents the number of historical tokens that q computes against in the computation between q and ori_kv. -1 represents infinity, meaning all tokens participate in computation. The default value is -1. | int32 | - | -
| ori_win_right | int | Optional | Represents the number of future tokens that q computes against in the computation between q and ori_kv. -1 represents infinity, meaning all tokens participate in computation. The default value is -1. | int32 | - | -
| layout_q | string | Optional | Represents the layout format of input q. The default value is BSND. | string | - | -
| layout_kv | string | Optional | Represents the layout format of input ori_kv/cmp_kv. The default value is BSND. | string | - | -
| topk_value_mode | int | Optional | Represents the value mode of topK indices. The default value is 1. | int32 | - | -
| return_softmax_lse | bool | Optional | Represents whether to return the lse result of softmax. The default value is False. | bool | - | -

## Return Values

### mixed_quant_sparse_flash_mla_metadata

| Name | Type | Type | Description | data type | data format | Dimensions |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| metadata | tensor | Required | Core splitting information of mixed_quant_sparse_flash_mla. | int32 | ND | <ul><li>(1024,)</li></ul>

### mixed_quant_sparse_flash_mla

| Name | Type | Type | Description | Data Type | Data Format | Dimensions |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| attention_out | tensor | Required | Computation output of mixed_quant_sparse_flash_mla. | bfloat16 | ND | <ul><li>(b, q_s, q_n, q_d)</li><li>(q_t, q_n, q_d)</li></ul>
| softmax_lse | tensor | Optional | First, softmax_max is obtained by taking the max of the query × key result. Then, softmax_sum is obtained by subtracting softmax_max from the query × key result, taking exp, and summing. Finally, the result is obtained by taking the log of softmax_sum and adding softmax_max. | float32 | ND | <ul><li>(b, kv_n, q_s, q_n/kv_n)</li><li>(kv_n, q_t, q_n/kv_n)</li></ul>

## Constraints

- Declaration
  - Inputs such as cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_q, seqused_ori_kv, seqused_cmp_kv, cmp_residual_kv, ori_block_table, and cmp_block_table are tensors. Because the operator cannot obtain the specific values of the tensors in the Tiling phase, tiling does not validate the values. You must ensure their correctness. If invalid values are passed in for the preceding parameters, undefined behavior is triggered (accuracy issues, program crashes caused by illegal memory access, etc.).
  - The input parameters of mixed_quant_sparse_flash_mla_metadata and mixed_quant_sparse_flash_mla must be consistent during the call. The operator is called in two phases through two APIs and cannot perform the validation itself. You must ensure correctness. If the parameters passed to the APIs are inconsistent, undefined behavior occurs (accuracy issues, program crashes caused by illegal memory access, etc.).
  - ori_topk_length and cmp_topk_length indicate the length of ori/cmp sparse_indices that actually participates in the computation. The value cannot be greater than the size of the last dimension of sparse_indices. When seqused_q is passed in, the values of the valid part corresponding to topk_length must be greater than or equal to 0.
  - When ori_mask_mode/cmp_mask_mode is 0, ori_kv_k/cmp_kv_k must be greater than or equal to the maximum value of ori_topk_length/cmp_topk_length.
  - cmp_residual_kv is used with cmp_ratio to restore the KV length before compression. The value of each batch must be less than cmp_ratio, that is, cmp_residual_kv[i] < cmp_ratio. cmp_residual_kv can be passed in only when cmp_mask_mode=3 and cmp_ratio!=1. It cannot be passed in when cmp_mask_mode=0 or cmp_ratio=1.
  - attention_out: tensor type, the output in the formula. bfloat16 is supported for the data type. ND is supported for the data format. Constraint: The shape of this output parameter must be the same as the shape of input parameter q, and the dtype must be the same as that of q.
  - When return_softmax_lse=False, a tensor with shape [1] and value 0 is returned. When return_softmax_lse=True, the float32 log-sum-exp result is returned.
  - For cu_seqlens_q, cu_seqlens_ori_kv, and cu_seqlens_cmp_kv, the first element must be 0, and the sequence must be non-decreasing as a whole, that is, each element is not less than its previous value.
  - When layout_kv is PA_BBND, ori_kv and cmp_kv support non-contiguity on axis 0.
  - For dimensions denoted by the same symbol in the shapes of parameters, the actual values on the corresponding axes must be consistent.

### Feature Parameter Group

|      Feature Parameter Group      |     Parameter Field Name     |
| :-------------------: | :-------------------: |
|      Common Parameter Group      | q, quant_mode, ori_kv, cmp_kv, metadata, rope_head_dim, softmax_scale, layout_q, layout_kv, attention_out |
|      Mask Parameter Group      | ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right |
|   SeqLens Parameter Group   | cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_q, seqused_ori_kv, seqused_cmp_kv |
|   Sparse Compression Parameter Group    | cmp_ratio, cmp_residual_kv, ori_sparse_indices, cmp_sparse_indices, ori_topk_length, cmp_topk_length, topk_value_mode |
| Paged Attention Parameter Group | ori_block_table, cmp_block_table |
|   Sinks Parameter Group   | sinks |
|   SoftmaxLse Parameter Group   | return_softmax_lse, softmax_lse |

### Baseline Information Description

### Computation Mode Description

|    Name    |    Parameters Required for Typical Use Cases    |    Parameters Required for Fully Sparse Scenarios    |
| :---------: | :---------------------------------------------------------: | :---------------------------------------------------------: |
|      SWA      | ori_kv | ori_kv, ori_sparse_indices |
|      HCA      | ori_kv, cmp_kv|-|
|      CSA      | ori_kv, cmp_kv, cmp_sparse_indices| ori_kv, ori_sparse_indices, cmp_kv, cmp_sparse_indices|

### Parameter Group Constraints

#### Common Parameter Group

- Handling of empty input parameters:
  - An empty tensor indicates that the shape size of a required input, or an input or output that must be passed in a specific call use case, is 0, that is, any axis is 0.
  - All use cases that trigger an empty tensor are intercepted and an error is reported.

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
    <th>Single-Parameter Validation</th>
    <th>Existence Validation</th>
    <th>Consistency Validation</th>
    <th>Feature Cross-Validation</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>q</td>
        <td>            <ul>
                <li>dtype supports bfloat16.</li>
                <li>If layout_q is BSND, the shape of q is (b, q_s, q_n, q_d).</li>
                <li>If layout_q is TND, the shape of q is (q_t, q_n, q_d).</li>
            </ul>
        </td>
        <td>            Must Be Passed In
        </td>
        <td rowspan="4">            <ul>
                <li>q and attention_out must have the same dtype and shape.</li>
                <li>If cmp_kv is passed in, ori_kv and cmp_kv must have the same dtype.</li>
                <li>If layout_kv is not PA_BBND, layout_q must be the same as layout_kv.</li>
                <li>If layout_kv is PA_BBND, layout_q can be BSND or TND.</li>
            </ul>
        </td>
        <td rowspan="4">            Axis validation:
            <ul>
                <li>b > 0</li>
                <li>q_s > 0</li>
                <li>0 < q_n <= 128</li>
                <li>q_d = 512</li>
                <li>q_t > 0</li>
                <li>ori_kv_s > 0</li>
                <li>cmp_kv_s > 0</li>
                <li>kv_n = 1</li>
                <li>If quant_mode=1, kv_d=608; if quant_mode=2, kv_d=584.</li>
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
                <li>dtype supports fp8_e4m3.</li>
                <li>If layout_kv is BSND, the shape of ori_kv is (b, ori_kv_s, kv_n, kv_d).</li>
                <li>If layout_kv is TND, the shape of ori_kv is (ori_kv_t, kv_n, kv_d).</li>
                <li>If layout_kv is PA_BBND, the shape of ori_kv is (ori_kv_block_nums, ori_kv_block_size, kv_n, kv_d).</li>
            </ul>
        </td>
        <td>            Must Be Passed In in the Current Version
        </td>
    </tr>
    <tr>
        <td>attention_out</td>
        <td>            <ul>
                <li>dtype supports bfloat16.</li>
                <li>If layout_q is BSND, the shape of attention_out is (b, q_s, q_n, q_d).</li>
                <li>If layout_q is TND, the shape of attention_out is (q_t, q_n, q_d).</li>
            </ul>
        </td>
        <td>            Must Be Passed In
        </td>
    </tr>
    <tr>
        <td>cmp_kv</td>
        <td>            <ul>
                <li>dtype supports fp8_e4m3.</li>
                <li>If layout_kv is BSND, the shape of cmp_kv is (b, cmp_kv_s, kv_n, kv_d).</li>
                <li>If layout_kv is TND, the shape of cmp_kv is (cmp_kv_t, kv_n, kv_d).</li>
                <li>If layout_kv is PA_BBND, the shape of cmp_kv is (cmp_kv_block_nums, cmp_kv_block_size, kv_n, kv_d).</li>
            </ul>
        </td>
        <td>Optional input</td>
    </tr>
</tbody>
</table>

Layout matching relationship table:
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
        <td>              <ul>
                        <li>BSND</li>
                        <li>PA_BBND</li>
               </ul>
        </td>
    </tr>
    <tr>
        <td>TND</td>
        <td>              <ul>
                        <li>TND</li>
                        <li>PA_BBND</li>
               </ul>
        </td>
    </tr>
</tbody>
</table>

metadata validation
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
                    <li>dtype only supports int32.</li>
                    <li>The shape is dynamically computed by mixed_quant_sparse_flash_mla_metadata.</li>
                </ul>
            </td>
            <td>Must be passed in for the current version.</td>
            <td>None</td>
            <td>If passed in, the value must be consistent with the result generated by mixed_quant_sparse_flash_mla_metadata.</td>
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
                    <li>dtype supports int32.</li>
                    <li>Supports an input range of only 0, 3, and 4. The default value is 0.</li>
                </ul>
            </td>
            <td>                Optional. If not passed, the default value is 0.
            </td>
            <td>                <ul>
                    <li>None</li>
                </ul>
            </td>
            <td>                <ul>
                     <li>In SWA use cases, ori_mask_mode must be 0, 3, or 4.</li>
                 </ul>
             </td>
         </tr>
         <tr>
             <td>cmp_mask_mode</td>
             <td>                 <ul>
                     <li>dtype: int32</li>
                     <li>The supported input range is only 0 and 3. Default value is 0.</li>
                 </ul>
             </td>
             <td>                Optional. If not passed, the default value is 0.
            </td>
             <td>                 <ul>
                     <li>None</li>
                 </ul>
             </td>
             <td>                   <ul>
                              <li>If cmp_kv is not passed, cmp_mask_mode must be 0.</li>
                     </ul>
            </td>
        </tr>
        <tr>
            <td>ori_win_left/ori_win_right</td>
            <td>                <ul>
                    <li>dtype: int32</li>
                    <li>The supported input range is only -1 or a non-negative integer.</li>
                </ul>
            </td>
            <td>                Optional. If not passed, the default value is -1.
            </td>
            <td>                <ul>
                    None
                </ul>
            </td>
            <td>                <ul>
                      <li>Only when ori_mask_mode is 4 can ori_win_left/ori_win_right be >=0.</li>
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
                    <li>Value only supports non-negative integers.</li>
                    <li>The value in seqused_q must be less than or equal to q_s.</li>
                </ul>
            </td>
            <td>None</td>
            <td>None</td>
            <td>None</td>
        </tr>
        <tr>
            <td>seqused_ori_kv</td>
            <td>                <ul>
                    <li>Supports int32 dtype.</li>
                    <li>The shape is (b,).</li>
                    <li>Value only supports non-negative integers.</li>
                    <li>Values in seqused_ori_kv must be less than or equal to ori_kv_s.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Optional when layout_kv is BSND.</li>
                    <li>Must be passed in when layout_kv is PA_BBND.</li>
                    <li>Can be omitted when ori_topk_length is passed in.</li>
                </ul>
            </td>
            <td>None</td>
            <td>None</td>
        </tr>
        <tr>
            <td>seqused_cmp_kv</td>
            <td>                <ul>
                    <li>Supports int32 dtype.</li>
                    <li>The shape is (b,).</li>
                    <li>Value only supports non-negative integers.</li>
                    <li>Values in seqused_cmp_kv must be less than or equal to cmp_kv_s.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Optional when layout_kv is BSND.</li>
                    <li>Must be passed in when layout_kv is PA_BBND and cmp_kv is passed in.</li>
                    <li>Can be omitted when cmp_topk_length is passed in.</li>
                </ul>
            </td>
            <td>None</td>
            <td>None</td>
        </tr>
        <tr>
            <td>cu_seqlens_q</td>
            <td>                <ul>
                    <li>dtype supports int32.</li>
                    <li>shape is (b+1,).</li>
                    <li>Value only supports non-negative integers.</li>
                    <li>Value should be non-descending (greater than or equal to the previous value), the first element is 0 and the last element equals q_t.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Must be passed in when layout_q is TND.</li>
                    <li>Not supported when layout_q is not TND.</li>
                </ul>
            </td>
            <td>None</td>
            <td>None</td>
        </tr>
        <tr>
            <td>cu_seqlens_ori_kv</td>
            <td>                <ul>
                    <li>dtype supports int32.</li>
                    <li>shape is (b+1,).</li>
                    <li>Value only supports non-negative integers.</li>
                    <li>Value should be non-descending (greater than or equal to the previous value), the first element is 0 and the last element equals ori_kv_t.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Must be passed in when layout_kv is TND.</li>
                    <li>Not supported when layout_kv is not TND.</li>
                </ul>
            </td>
            <td>None</td>
            <td>None</td>
        </tr>
        <tr>
            <td>cu_seqlens_cmp_kv</td>
            <td>                <ul>
                    <li>Supports the int32 data type.</li>
                    <li>The shape is (b+1,).</li>
                    <li>Value only supports non-negative integers.</li>
                    <li>Values should be arranged in non-descending order (greater than or equal to the previous value). The first element is 0 and the last element equals cmp_kv_t.</li>
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
                    <li>Represents the compression ratio of cmp_kv relative to the KV length before compression. Value must be greater than 0.</li>
                    <li>Default value is 1.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Optional, default value is 1.</li>
                </ul>
            </td>
            <td>None</td>
            <td>                <ul>
                    <li>In the SWA typical use case, only the default value 1 is supported.</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>cmp_residual_kv</td>
            <td>                <ul>
                    <li>dtype supports int32.</li>
                    <li>The shape is (b,).</li>
                    <li>Value only supports non-negative integers.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Validated only when cmp_kv is passed in.</li>
                    <li>Optional</li>
                    <li>Must be passed in when cmp_mask_mode=3 and cmp_ratio!=1.</li>
                    <li>Cannot be passed in when cmp_mask_mode=0 or cmp_ratio=1.</li>
                </ul>
            </td>
            <td>None</td>
            <td>None</td>
        </tr>
        <tr>
            <td>ori_sparse_indices</td>
            <td>                <ul>
                    <li>dtype supports int32.</li>
                    <li>The shape is (q_t, kv_n, ori_kv_k) or (b, q_s, kv_n, ori_kv_k).</li>
                    <li>Invalid positions are filled with -1, and the rest are non-negative integers.</li>
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
                    <li>dtype supports int32.</li>
                    <li>shape is (q_t, kv_n, cmp_kv_k) or (b, q_s, kv_n, cmp_kv_k).</li>
                    <li>Invalid positions are filled with -1, and other values are non-negative integers.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Validation is performed only when cmp_kv is passed in.</li>
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
                    <li>dtype supports int32.</li>
                    <li>shape is (b, q_s, kv_n) or (q_t, kv_n).</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Must be passed in when ori_mask_mode=0 and ori_sparse_indices is not empty.</li>
                    <li>Not supported in other use cases.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>When layout_q is TND, the shape is (q_t, kv_n).</li>
                    <li>When layout_q is BSND, the shape is (b, q_s, kv_n).</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Not supported when ori_mask_mode is not 0.</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>cmp_topk_length</td>
            <td>                <ul>
                    <li>dtype: int32</li>
                    <li>shape: (b, q_s, kv_n) or (q_t, kv_n)</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Validation is performed only when cmp_kv is passed in.</li>
                    <li>Must be passed in when cmp_mask_mode=0 and cmp_sparse_indices is not empty.</li>
                    <li>Not supported in other use cases.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>If layout_q is TND, the shape is (q_t, kv_n).</li>
                    <li>If layout_q is BSND, the shape is (b, q_s, kv_n).</li>
                </ul>
            </td>
            <td>None</td>
        </tr>
        <tr>
            <td>topk_value_mode</td>
            <td>                <ul>
                    <li>data_type: int32</li>
                    <li>topK index value mode. Default value is 1.</li>
                </ul>
            </td>
            <td>Optional attribute. Default value is 1.</td>
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
                    <li>Supports only the int32 data type.</li>
                    <li>Shape is (b, ceil(ori_kv_s_max/ori_kv_block_size)).</li>
                    <li>Values must be positive integers.</li>
                </ul>
            </td>
            <td>Optional</td>
            <td>None</td>
            <td>                <ul>
                    <li>When ori_block_table exists, seqused_ori_kv must be passed in.</li>
                    <li>When PagedAttention is enabled, block_table must not be empty.</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>cmp_block_table</td>
            <td>                <ul>
                    <li>The data type must be int32.</li>
                    <li>The shape is (b, ceil(cmp_kv_s_max/cmp_kv_block_size)).</li>
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
                    <li>When cmp_block_table exists, seqused_cmp_kv must be passed in.</li>
                    <li>When PagedAttention is enabled, block_table must not be empty.</li>
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
                    <li>dtype supports float32.</li>
                    <li>shape is (q_n, ).</li>
                </ul>
            </td>
            <td> Must be passed in the current version </td>
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
                    <li>data_type only supports bool.</li>
                    <li>true enables softmax_lse, and false disables softmax_lse.</li>
                </ul>
            </td>
            <td>Optional, default value is false</td>
            <td rowspan="2">                <ul>
                     <li>When return_softmax_lse is false, the output is a tensor with shape [1] and value 0.</li>
                    <li>When return_softmax_lse is true, the shape of softmax_lse depends on layout_q as follows:<ul><li>When layout_q is BSND, the shape of softmax_lse is (b, kv_n, q_s, q_n/kv_n).</li><li>When layout_q is TND, the shape of softmax_lse is (kv_n, q_t, q_n/kv_n).</li></ul></li>
                </ul>
            </td>
            <td>None</td>
        </tr>
        <tr>
            <td>softmax_lse</td>
            <td>                <ul>
                    <li>data_type supports only float32.</li>
                </ul>
            </td>
            <td>None</td>
            <td>None</td>
        </tr>
    </tbody>
</table>

## Example<a name="zh-cn_topic_0000002168254826_section14459801435"></a>

### SWA, BSND Input

```python
import torch
import torch_npu
import math
import cann_ops_transformer

torch_npu.npu.set_device(0)

qDtype = torch.bfloat16
kvDtype = torch.float8_e4m3fn
B = 1
S1 = 16
S2 = 64
N1 = 64
N2 = 1
Dq = 512
Dkv = 608
quant_mode = 1
cmp_ratio = 1  # In the SWA example, only ori_kv is passed. cmp_ratio does not participate in compressed KV computation. Keep the default value 1.

seqused_q = torch.tensor([16], dtype=torch.int32, device="npu")
seqused_ori_kv = torch.tensor([64], dtype=torch.int32, device="npu")

q = torch.randn(B, S1, N1, Dq, dtype=qDtype, device="npu")
ori_kv = torch.randn(B, S2, N2, Dkv, dtype=qDtype, device="npu").to(kvDtype)
cmp_kv = None
sinks = torch.zeros(N1, dtype=torch.float32, device="npu")

layout_q = "BSND"
layout_kv = "BSND"

metadata = torch.ops.cann_ops_transformer.mixed_quant_sparse_flash_mla_metadata(
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
    rope_head_dim=64,
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

npu_result, npu_lse = torch.ops.cann_ops_transformer.mixed_quant_sparse_flash_mla(
    q=q,
    ori_kv=ori_kv,
    cmp_kv=cmp_kv,
    seqused_q=seqused_q,
    seqused_ori_kv=seqused_ori_kv,
    sinks=sinks,
    metadata=metadata,
    quant_mode=quant_mode,
    rope_head_dim=64,
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

qDtype = torch.bfloat16
kvDtype = torch.float8_e4m3fn
B = 1
S1 = 16
S2 = 2050
N1 = 64
N2 = 1
Dq = 512
Dkv = 608
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

q = torch.randn(B, S1, N1, Dq, dtype=qDtype, device="npu")

ori_kv_bnsd = torch.randn(B, N2, S2, Dkv, dtype=qDtype, device="npu").to(kvDtype)
ori_kv, ori_block_table = scatter_to_pa(ori_kv_bnsd, block_size1, block_num1, ori_max_block_num_per_batch)

if cmp_max_s2 > 0:
    cmp_max_block_num_per_batch = math.ceil(cmp_max_s2 / block_size2)
    block_num2 = cmp_max_block_num_per_batch * B
    cmp_kv_bnsd = torch.randn(B, N2, cmp_max_s2, Dkv, dtype=qDtype, device="npu").to(kvDtype)
    cmp_kv, cmp_block_table = scatter_to_pa(cmp_kv_bnsd, block_size2, block_num2, cmp_max_block_num_per_batch)
else:
    block_num2 = 0
    cmp_kv = None
    cmp_block_table = None

sinks = torch.zeros(N1, dtype=torch.float32, device="npu")

layout_q = "BSND"
layout_kv = "PA_BBND"

metadata = torch.ops.cann_ops_transformer.mixed_quant_sparse_flash_mla_metadata(
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
    rope_head_dim=64,
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

npu_result, npu_lse = torch.ops.cann_ops_transformer.mixed_quant_sparse_flash_mla(
    q=q,
    ori_kv=ori_kv,
    cmp_kv=cmp_kv,
    ori_block_table=ori_block_table,
    cmp_block_table=cmp_block_table,
    seqused_q=seqused_q,
    seqused_ori_kv=seqused_ori_kv,
    seqused_cmp_kv=seqused_cmp_kv,
    cmp_residual_kv=cmp_residual_kv,
    sinks=sinks,
    metadata=metadata,
    quant_mode=quant_mode,
    rope_head_dim=64,
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

    def forward(self, q, ori_kv, cmp_kv, cmp_sparse_indices,
                cu_seqlens_q, seqused_q, cu_seqlens_ori_kv, seqused_ori_kv,
                cu_seqlens_cmp_kv, seqused_cmp_kv, cmp_residual_kv,
                sinks, metadata, quant_mode, rope_head_dim, softmax_scale, cmp_ratio,
                ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right,
                layout_q, layout_kv, topk_value_mode, return_softmax_lse):
        npu_result, npu_lse = torch.ops.cann_ops_transformer.mixed_quant_sparse_flash_mla(
            q=q,
            ori_kv=ori_kv,
            cmp_kv=cmp_kv,
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
            rope_head_dim=rope_head_dim,
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
        return npu_result, npu_lse

qDtype = torch.bfloat16
kvDtype = torch.float8_e4m3fn
B = 1
S1 = 16
S2 = 2050
N1 = 64
N2 = 1
Dq = 512
Dkv = 608
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

q = torch.randn(total_q, N1, Dq, dtype=qDtype, device="npu")
ori_kv = torch.randn(total_ori_kv, N2, Dkv, dtype=qDtype, device="npu").to(kvDtype)
cmp_kv = torch.randn(total_cmp_kv, N2, Dkv, dtype=qDtype, device="npu").to(kvDtype)

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

print("mixed_quant_sparse_flash_mla_metadata...")
metadata = torch.ops.cann_ops_transformer.mixed_quant_sparse_flash_mla_metadata(
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
    rope_head_dim=64,
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

print("mixed_quant_sparse_flash_mla ...")
npu_result, npu_lse = npu_mode(
    q=q,
    ori_kv=ori_kv,
    cmp_kv=cmp_kv,
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
    rope_head_dim=64,
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
