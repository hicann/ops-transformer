# sparse_flash_mla <a name="ZH-CN_TOPIC_SPARSE_FLASH_MLA"></a>

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T13:18:27.528Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id8 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id8 -->
<!-- npu="910" id9 -->
- <term>Atlas training products</term>: Not supported
<!-- end id9 -->

## Function<a name="zh-cn_topic_sparse_flash_mla_function"></a>

- **Description**:

  `sparse_flash_mla` is a torch API extended by `cann_ops_transformer`. It calls the `SparseFlashMla` operator to perform sparse attention computation with shared KV (Key and Value use the same input). The API supports both training and inference use cases, and combines the KV context using `ori_kv` and the optional `cmp_kv`.

  `sparse_flash_mla_metadata` is the pre-processing metadata generation API of the main operator. It generates the task splitting result for AI Core/AI Vector Core. `metadata` must be generated using parameters exactly the same as those of the main operator, and passed in to `sparse_flash_mla`. The metadata is required in this scenario. The typical call procedure is as follows:

  1. Prepare inputs based on the corresponding use case, such as `q`, `ori_kv`, the optional `cmp_kv`, sequence length, sparse index, or Block Table.
  2. Call `sparse_flash_mla_metadata` to generate `metadata` as the input parameter of the `sparse_flash_mla` API.
  3. Call `sparse_flash_mla` and pass in the computation result of `sparse_flash_mla_metadata`.

  The API supports the following three typical computation modes:

  - **SWA (Sliding Window Attention)**: Uses only `ori_kv` to perform sliding window attention on the original KV.
  - **CSA (Compressed Sparse Attention)**: Uses `ori_kv`, `cmp_kv`, and `cmp_sparse_indices` together to perform attention on both the original KV window and the compressed KV selected by TopK.
  - **HCA (Heavily Compressed Attention)**: Uses `ori_kv` and `cmp_kv` together to perform attention on both the original KV window and the continuous compressed KV segments.

- **Formula**:

  $$
  O = \text{softmax}(Q \cdot \tilde{K}^{T} \cdot \text{softmax\_scale}) \cdot \tilde{V}
  $$

  where $\tilde{K}=\tilde{V}$ is composed of the sliding window part of `ori_kv` and the compressed part of `cmp_kv`. The KV range actually participating in the calculation is jointly determined by `cmp_ratio`, `ori_mask_mode`, `cmp_mask_mode`, `ori_win_left`, `ori_win_right`, and `cmp_sparse_indices`.

  When `return_softmax_lse` is enabled, the second output is the log-sum-exp for each Query position:

  $$
  \operatorname{LSE}=\log\sum\exp(S-\max(S))+\max(S),\quad S=Q\tilde{K}^{T}\cdot\text{softmax\_scale}
  $$

## Prototype<a name="zh-cn_topic_sparse_flash_mla_prototype"></a>

Before calling the `sparse_flash_mla` API, call the preceding API `sparse_flash_mla_metadata` first to complete core splitting. The parameters of the two calls must be consistent.

> [!NOTE]
>
> The default values of the mask and window parameters of `sparse_flash_mla_metadata` are `ori_mask_mode=0`, `cmp_mask_mode=0`, `ori_win_left=-1`, and `ori_win_right=-1`, while the corresponding default values of `sparse_flash_mla` are `4`, `3`, `-1`, and `-1`. To ensure parameter consistency between the two calls, explicitly pass in these four parameters instead of relying on the default values of both APIs.

```python
cann_ops_transformer.sparse_flash_mla_metadata(
    num_heads_q,
    num_heads_kv,
    head_dim,
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
cann_ops_transformer.sparse_flash_mla(
    q,
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
    softmax_scale=1.0,
    cmp_ratio=1,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=-1,
    ori_win_right=-1,
    layout_q="BSND",
    layout_kv="BSND",
    topk_value_mode=1,
    return_softmax_lse=False
) -> (Tensor, Tensor)
```

## Parameters<a name="zh-cn_topic_sparse_flash_mla_parameters"></a>

### Baseline Information

| Name | Meaning |
| :--- | :--- |
| b | Indicates the batch size of input samples. |
| q_n / kv_n | Number of heads of `q` / number of heads of kv. Currently, only 1 is supported for `kv_n`. |
| q_s / ori_kv_s / cmp_kv_s | Sequence length per batch of `q`, `ori_kv`, and `cmp_kv`. |
| q_t / ori_kv_t / cmp_kv_t | Cumulative sum of sequence lengths of all batches in `q`, `ori_kv`, and `cmp_kv` under the TND layout. |
| d | Dimension of each attention head. Currently, only 512 is supported. |
| ori_kv_k / cmp_kv_k | Number of tokens selected by topK from input `ori_sparse_indices` and `cmp_sparse_indices`. |
| ori_kv_s_max / cmp_kv_s_max | Maximum sequence length of input `ori_kv` and `cmp_kv`. |
| ori_kv_block_size / cmp_kv_block_size | Block size of input `ori_kv` and `cmp_kv` in the PagedAttention use case. |
| ori_kv_block_nums / cmp_kv_block_nums | Number of blocks of input `ori_kv` and `cmp_kv` in the PagedAttention use case. |

### sparse_flash_mla_metadata

| Name | Type | Required/Optional | Description | Data Type | Data Format | Dimension |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| num_heads_q | int32 | Required | Number of heads of `q`, ranging from 1 to 128. | int32 | - | - |
| num_heads_kv | int32 | Required | Number of heads of `ori_kv` and `cmp_kv`. Only 1 is supported. | int32 | - | - |
| head_dim | int32 | Required | Dimension of each attention head. Only 512 is supported. | int32 | - | - |
| cu_seqlens_q | Tensor | Optional | Cumulative sum of the number of valid tokens in each batch of `q` in the TND use case. The first element must be 0. | int32 | ND | (b+1,) |
| cu_seqlens_ori_kv | Tensor | Optional | Cumulative sum of the number of valid tokens in each batch of `ori_kv` in the TND use case. The first element must be 0. | int32 | ND | (b+1,) |
| cu_seqlens_cmp_kv | Tensor | Optional | Cumulative sum of the number of valid tokens in each batch of `cmp_kv` in the TND use case. The first element must be 0. | int32 | ND | (b+1,) |
| seqused_q | Tensor | Optional | Number of tokens of `q` actually participating in computation in each batch. | int32 | ND | (b,) |
| seqused_ori_kv | Tensor | Optional | Number of tokens of `ori_kv` actually participating in computation in each batch. | int32 | ND | (b,) |
| seqused_cmp_kv | Tensor | Optional | Number of tokens of `cmp_kv` actually participating in computation in each batch. | int32 | ND | (b,) |
| cmp_residual_kv | Tensor | Optional | Remainder of the kv length before compression divided by `cmp_ratio` in each batch, used to restore the mask length on the cmp side. | int32 | ND | (b,) |
| ori_topk_length | Tensor | Optional |  Indicates the length of ori_sparse_indices actually participating in computation. | int32 | ND | `BSND`: (b, q_s, kv_n)<br>`TND`: (q_t, kv_n) |
| cmp_topk_length | Tensor | Optional |  Indicates the length of cmp_sparse_indices actually participating in computation. | int32 | ND | `BSND`: (b, q_s, kv_n)<br>`TND`: (q_t, kv_n) |
| batch_size | int32 | Optional | Batch size; used in the BSND use case. Default value is 0. | int32 | - | - |
| max_seqlen_q | int32 | Optional | Maximum effective sequence length of `q`. In the TND use case, it must be equal to the actual maximum length. Default value is 0. | int32 | - | - |
| max_seqlen_ori_kv | int32 | Optional | Maximum effective sequence length of `ori_kv`. Default value is 0. | int32 | - | - |
| max_seqlen_cmp_kv | int32 | Optional | Maximum effective sequence length of `cmp_kv`. Default value is 0. | int32 | - | - |
| ori_topk | int32 | Optional | Number of key sparse tokens selected from `ori_kv`. 0 indicates the non-sparse use case. Default value is 0. | int32 | - | - |
| cmp_topk | int32 | Optional | Number of key sparse tokens selected from `cmp_kv`. 0 indicates the non-sparse use case. Default value is 0. | int32 | - | - |
| cmp_ratio | int32 | Optional | Compression ratio of `cmp_kv` relative to the kv before compression. Ranges from 1 to 128. Default value is 1. | int32 | - | - |
| ori_mask_mode | int32 | Optional | Mask mode between `q` and `ori_kv`: 0 (No Mask), 3 (RightDownCausal), or 4 (Band). | int32 | - | - |
| cmp_mask_mode | int32 | Optional | Mask mode between `q` and `cmp_kv`: 0 (No Mask) or 3 (RightDownCausal). | int32 | - | - |
| ori_win_left | int32 | Optional | Left boundary of the `ori_kv` sliding window. The value is -1 or not less than 0; -1 indicates no limit. | int32 | - | - |
| ori_win_right | int32 | Optional | Right boundary of the `ori_kv` sliding window. The value is -1 or not less than 0; -1 indicates no limit. | int32 | - | - |
| layout_q | string | Optional | Data layout of `q`. Supports `BSND` and `TND`. Default value is `BSND`. | string | - | - |
| layout_kv | string | Optional | Data layout of `ori_kv` and `cmp_kv`. Supports `BSND`, `TND`, and `PA_BBND`. Default value is `BSND`. | string | - | - |
| has_ori_kv | bool | Optional | Whether `ori_kv` is passed in to the main operator. Default value is True. | bool | - | - |
| has_cmp_kv | bool | Optional | Whether `cmp_kv` is passed in to the main operator. Default value is True. | bool | - | - |
| metadata | Tensor | output | Task splitting result used by the main operator. | int32 | ND | (1024,) |

### sparse_flash_mla

| Name | Type | Required/Optional | Description | Data Type | Data Format | Dimension |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| q | Tensor | Required | Query in the formula. | bfloat16/float16 | ND | `BSND`: (b, q_s, q_n, d)<br>`TND`: (q_t, q_n, d) |
| ori_kv | Tensor | Optional | Original KV input. Key and Value share the same data. | bfloat16/float16 | ND | `BSND`: (b, ori_kv_s, kv_n, d)<br>`TND`: (ori_kv_t, kv_n, d)<br>`PA_BBND`: (ori_kv_block_nums, ori_kv_block_size, kv_n, d) |
| cmp_kv | Tensor | Optional | Compressed KV input. Key and Value share the same data. | bfloat16/float16 | ND | `BSND`: (b, cmp_kv_s, kv_n, d)<br>`TND`: (cmp_kv_t, kv_n, d)<br>`PA_BBND`: (cmp_kv_block_nums, cmp_kv_block_size, kv_n, d) |
| ori_sparse_indices | Tensor | Optional | Original KV sparse index. Reserved field. | int32 | ND | `TND`: (q_t, kv_n, ori_kv_k) `BSND`: (b, q_s, kv_n, ori_kv_k) |
| cmp_sparse_indices | Tensor | Optional | TopK indices of the compressed KV. Invalid positions are filled with -1. Passed in only for CSA use cases. | int32 | ND | `BSND`: (b, q_s, kv_n, cmp_kv_k)<br>`TND`: (q_t, kv_n, cmp_kv_k) |
| ori_block_table | Tensor | Optional | Block mapping table used by `ori_kv` in PagedAttention use cases. | int32 | ND | (b, max_num_blocks_per_seq) |
| cmp_block_table | Tensor | Optional | Block mapping table used by `cmp_kv` in PagedAttention use cases. | int32 | ND | (b, max_num_blocks_per_seq) |
| cu_seqlens_q | Tensor | Optional | Cumulative sum of the number of effective tokens in each batch of `q` in TND use cases. | int32 | ND | (b+1,) |
| cu_seqlens_ori_kv | Tensor | Optional | Cumulative sum of the number of effective tokens in each batch of `ori_kv` in TND use cases. | int32 | ND | (b+1,) |
| cu_seqlens_cmp_kv | Tensor | Optional | Cumulative sum of the number of effective tokens in each batch of `cmp_kv` in TND use cases. | int32 | ND | (b+1,) |
| seqused_q | Tensor | Optional | Number of `q` tokens actually participating in computation in each batch. | int32 | ND | (b,) |
| seqused_ori_kv | Tensor | Optional | Number of `ori_kv` tokens actually participating in computation in each batch. | int32 | ND | (b,) |
| seqused_cmp_kv | Tensor | Optional | Number of `cmp_kv` tokens actually participating in computation in each batch. | int32 | ND | (b,) |
| cmp_residual_kv | Tensor | Optional | Compressed remainder of each batch. Must be passed in when `cmp_kv` exists, `cmp_mask_mode=3`, and `cmp_ratio!=1`. Not allowed when `cmp_mask_mode=0` or `cmp_ratio=1`. | int32 | ND | (b,) |
| ori_topk_length | Tensor | Optional | Length of `ori_sparse_indices` actually participating in computation. | int32 | ND | - |
| cmp_topk_length | Tensor | Optional | Length of `cmp_sparse_indices` actually participating in computation. | int32 | ND | - |
| sinks | Tensor | Optional | Independently learnable virtual offset terms set for each attention head, used to maintain stability during long-text inference. | float32 | ND | (q_n,) |
| metadata | Tensor | Required | Core splitting information generated by `sparse_flash_mla_metadata`. | int32 | ND | (1024,) |
| softmax_scale | float | Optional | Scaling coefficient after QK matrix multiplication. The default value is 1.0. | float32 | - | - |
| cmp_ratio | int32 | Optional | Compression ratio. Value range: 1-128. The default value is 1. | int32 | - | - |
| ori_mask_mode | int32 | Optional | Mask mode between `q` and `ori_kv`. 0: No mask. 3: rightDownCausal mode. 4: sliding window mode. The default value is 4. | int32 | - | - |
| cmp_mask_mode | int32 | Optional | Mask mode between `q` and `cmp_kv`. 0: No mask. 3: rightDownCausal mode. The default value is 3. | int32 | - | - |
| ori_win_left | int32 | Optional | Left boundary of the `ori_kv` sliding window, indicating the number of historical tokens computed by `q` in the computation between `q` and `ori_kv`. Value: -1 or not less than 0. -1 indicates no limit. The default value is -1. | int32 | - | - |
| ori_win_right | int32 | Optional | Right boundary of the `ori_kv` sliding window, indicating the number of future tokens computed by `q` in the computation between `q` and `ori_kv`. Value: -1 or not less than 0. -1 indicates no limit. The default value is -1. | int32 | - | - |
| layout_q | string | Optional | Data layout of `q`. Options: `BSND` and `TND`. The default value is `BSND`. | string | - | - |
| layout_kv | string | Optional | Data layout of KV. Options: `BSND`, `TND`, and `PA_BBND`. The default value is `BSND`. | string | - | - |
| topk_value_mode | int32 | Optional | TopK index value mode. Only 1 is supported. The default value is 1. | int32 | - | - |
| return_softmax_lse | bool | Optional | Whether to return the log-sum-exp result of softmax. The default value is False. | bool | - | - |
| attention_out | Tensor | Required | Attention computation output. Its shape and data type are the same as those of `q`. | bfloat16/float16 | ND | Same as `q` |
| softmax_lse | Tensor | Optional | Log-sum-exp result of softmax. | float32 | ND | `BSND`: (b, kv_n, q_s, q_n/kv_n)<br>`TND`: (kv_n, q_t, q_n/kv_n) |

## Return Values<a name="zh-cn_topic_sparse_flash_mla_returns"></a>

### sparse_flash_mla_metadata

| Name | Type | Description | Data Type | Data Format | Dimension |
| :--- | :--- | :--- | :--- | :--- | :--- |
| metadata | Tensor | Core partitioning information for `sparse_flash_mla`. | int32 | ND | (1024,) |

### sparse_flash_mla

| Name | Type | Description | Data Type | Data Format | Dim |
| :--- | :--- | :--- | :--- | :--- | :--- |
| attention_out | Tensor | Attention computation output. Its shape and data type are the same as `q`. | bfloat16/float16 | ND | Same as `q` |
| softmax_lse | Tensor | Returns the log-sum-exp of softmax when `return_softmax_lse=True`; otherwise returns a float32 scalar placeholder tensor. | float32 | ND | `BSND`: (b, kv_n, q_s, q_n/kv_n)<br>`TND`: (kv_n, q_t, q_n/kv_n) |

## Constraints<a name="zh-cn_topic_sparse_flash_mla_constraints"></a>

- Declarations
  - `cu_seqlens_q`, `cu_seqlens_ori_kv`, `cu_seqlens_cmp_kv`, `seqused_q`, `seqused_ori_kv`, `seqused_cmp_kv`, `cmp_residual_kv`, the sparse index, and the Block Table are all tensors. The operator cannot verify their specific values in the Tiling phase. You must ensure their validity. Passing invalid values may cause precision anomalies or illegal memory access.
  - When the input is PA_BBND, `seqused_ori_kv` and `ori_block_table` must be passed in. When the input is BSND, `seqused_ori_kv` can be used to indicate the effective length of `ori_kv` in each batch. When the input is TND, the maximum length of `ori_kv` is indicated by `cu_seqlens_ori_kv`. If `seqused_ori_kv` is passed in, the effective length is indicated by `seqused_ori_kv`. The same applies to `cmp_kv`.
  - `cu_seqlens_q`, `cu_seqlens_ori_kv`, and `cu_seqlens_cmp_kv` must satisfy the following requirements: the first element is 0, and the entire sequence is monotonically non-decreasing, that is, each element is not less than its preceding element.
  - When layout_kv is PA_BBND, `ori_kv` and `cmp_kv` can be non-contiguous along axis 0.
  - When `ori_mask_mode` and `cmp_mask_mode` are 0, `ori_kv_k` and `cmp_kv_k` need to be greater than or equal to the maximum value of `ori_topk_length` and `cmp_topk_length`, respectively.
  - `ori_topk_length` and `cmp_topk_length` indicate the lengths of ori/cmp sparse_indices actually participating in the computation. The value cannot be greater than the size of the last dimension of sparse_indices. When `seqused_q` is passed in, the values of the effective part corresponding to topk_length need to be greater than or equal to 0.
  - `cmp_residual_kv` is used together with `cmp_ratio` to restore the KV length before compression. The value of each batch needs to be less than `cmp_ratio`, that is, `cmp_residual_kv[i]` < `cmp_ratio`. It can be passed in only when `cmp_kv` exists, `cmp_mask_mode=3`, and `cmp_ratio!=1`. It is not allowed when `cmp_mask_mode=0` or `cmp_ratio=1`.
  - `attention_out`: of the tensor type. It is the output in the formula. The supported data types are bfloat16 and float16. The supported data format is ND. Constraint: The shape of this output parameter is the same as that of input parameter q, and its dtype is the same as that of q.
  - When `return_softmax_lse` is False, a tensor of shape [1] with value 0 is returned. When `return_softmax_lse` is True, the float32 log-sum-exp result is returned.
  - `cu_seqlens_q`, `cu_seqlens_ori_kv`, and `cu_seqlens_cmp_kv` must satisfy: the first element is 0, and the entire sequence is monotonically non-decreasing, that is, each element is not less than its previous element.
  - `sparse_flash_mla_metadata` and `sparse_flash_mla` are called in two phases. The input parameters that participate in task splitting must be consistent between the two calls. Otherwise, undefined behavior may occur.
  - This API supports the single-operator mode and TorchAir graph mode, and can be used for training and inference use cases.

### Feature Parameter Group

| Feature Parameter Group | Parameter Field Names |
| :--- | :--- |
| common parameter group | layout_q, layout_kv, q, ori_kv, cmp_kv, attention_out, softmax_scale |
| metadata parameter group | num_heads_q, num_heads_kv, head_dim, batch_size, max_seqlen_q, max_seqlen_ori_kv, max_seqlen_cmp_kv, metadata, has_ori_kv, has_cmp_kv |
| sparse compression parameter group | ori_topk, cmp_topk, cmp_ratio, ori_sparse_indices, cmp_sparse_indices |
| Mask and window parameter group | ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right, cmp_residual_kv |
| Paged Attention parameter group | ori_block_table, cmp_block_table |
| SeqLengths parameter group | cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_q, seqused_ori_kv, seqused_cmp_kv |
| Sinks parameter group | sinks |
| SoftmaxLse parameter group | return_softmax_lse, softmax_lse |

### Parameter Group Constraints

The following table describes the parameter requirements in terms of single-parameter check, existence check, consistency check, and feature cross-check. For specific values in a tensor, those that cannot be read or fully validated in the tiling phase are marked with "user guarantee". If such constraints are violated, no error may be reported at the API entry, but undefined behavior may occur.

#### Parameter Groups of the Preceding metadata API

Parameters involved in task splitting in `sparse_flash_mla_metadata` and `sparse_flash_mla` must be passed in pairs with exactly the same values. In particular, the call should be blocked when the layout, length, compression ratio, TopK length, Mask mode, window, or kv existence flag is inconsistent.

| Parameter | Single-Parameter Check | Existence Check | Consistency Check | Feature Cross-Check |
| :--- | :--- | :--- | :--- | :--- |
| num_heads_q | int32; range 1-128. | Required. | Must be consistent with the `q_n` dimension of `q`. | <term>Atlas A3 products</term>: Blocked if the value is not 1, 2, 4, 8, 16, 32, 64, or 128; <term>Ascend 950PR&950DT products</term>: Blocked if the value is less than 1 or greater than 128. |
| num_heads_kv | int32; only 1 is supported. | Required. | Must be consistent with the kv_n dimension of `ori_kv` and `cmp_kv`. | Must be consistent with the group dimension of `num_heads_q` and `softmax_lse`. |
| head_dim | int32; only 512 is supported. | Required. | Must be consistent with the d dimension of `q`, `ori_kv`, and `cmp_kv`. | Blocked if this is not satisfied; automatic inference is not allowed. |
| batch_size | int32; must be greater than 0. | Required in the BSND scenario. | Must be consistent with the b dimension of `q` and kv in the BSND layout. | In the TND scenario, even when derived from the length of `cu_seqlens_q`, it must still be consistent with all length-related tensors. |
| max_seqlen_q | int32; must be greater than 0. | Required in the TND scenario. | Must be equal to the maximum value of the actual lengths of all batches in `q`. | Must be consistent with `cu_seqlens_q` and the q_t dimension of q. |
| max_seqlen_ori_kv | int32; must be greater than 0. | Required when `ori_kv` is in TND layout. | Must be equal to the maximum value of the actual lengths of all batches in `ori_kv`. | Must be consistent with `cu_seqlens_ori_kv` and ori_kv_t. |
| max_seqlen_cmp_kv | int32; must be greater than 0. | Required when `cmp_kv` is in TND layout. | Must be equal to the maximum value of the actual lengths of all batches in `cmp_kv`. | Must be consistent with `cu_seqlens_cmp_kv` and cmp_kv_t. |
| ori_topk | int32; currently only 0 is supported. | Optional, default 0. | Must be consistent with the passed-in state of `ori_sparse_indices` and `ori_topk_length`. | A non-empty `ori_sparse_indices` is currently not supported, so the value must be 0. |
| cmp_topk | int32; value is 0 in SWA/HCA scenarios, and the TopK length of compressed kv (greater than 0) in the CSA scenario. | Required and non-zero in the CSA scenario; 0 in other scenarios. | Must be equal to the last dimension of `cmp_sparse_indices`. | <term>Atlas A3 products</term>: Blocked if the value is greater than 8192 or less than or equal to 0 in the CSA scenario, or non-zero in SWA/HCA scenarios; <term>Ascend 950PR&950DT products</term>: Blocked if the value is less than or equal to 0 in the CSA scenario, or non-zero in SWA/HCA scenarios. |
| cmp_ratio | int32; value is 1 in the SWA scenario, and 1-128 in CSA/HCA scenarios. | Optional, default 1. | Must be consistent with the compression relationship between the main API, `cmp_residual_kv`, and `cmp_kv`. | <term>Atlas A3 products</term>: No restriction for SWA; blocked if the value is not 1, 2, or 4 in the CSA scenario, or not 128 in the HCA scenario; <term>Ascend 950PR&950DT products</term>: Blocked if the value is not 1 in the SWA scenario, or not 1-128 in CSA/HCA scenarios. |
| ori_mask_mode | int32; 0, 3, and 4 are supported in the interface definition. | Optional. | None. | When 4 is passed in, used together with `ori_win_left` and `ori_win_right`. |
| cmp_mask_mode | int32; 0 and 3 are supported in the interface definition. | Optional. | None. | 0 for SWA; 3 for CSA/HCA. |
| ori_win_left | int32; -1 or non-negative in the interface definition. | Optional. | None. | <term>Atlas A3 products</term>: Blocked if the value is not 127; <term>Ascend 950PR&950DT products</term>: Blocked if the value is less than -1. Applies only to the `ori_kv` side. |
| ori_win_right | int32; -1 or non-negative in the interface definition. | Optional. | None. | <term>Atlas A3 products</term>: Blocked if the value is not 0; <term>Ascend 950PR&950DT products</term>: Blocked if the value is less than -1. Applies only to the `ori_kv` side. |
| layout_q | string; only `BSND` and `TND` are supported. | Optional, default `BSND`. | Must be consistent with the dimensions of q, the main API, and length-related tensors. | Only the following combinations with `layout_kv` are supported: `BSND`/`BSND`, `TND`/`TND`, `BSND`/`PA_BBND`, and `TND`/`PA_BBND`. |
| layout_kv | string; only `BSND`, `TND`, and `PA_BBND` are supported. | Optional, default `BSND`. | Must be consistent with the dimensions of `ori_kv` and `cmp_kv` and with the main API. | Must be the same as `layout_q` in non-PA scenarios; Block Table and `seqused_ori_kv` are required in the PA scenario. |
| has_ori_kv | bool. | Required; must be consistent with the actual passed-in state of the main API. | Must be equivalent to `ori_kv is not None`. | Must be True in all three supported scenarios. |
| has_cmp_kv | bool. | Required; must be consistent with the actual passed-in state of the main API. | Must be equivalent to `cmp_kv is not None`. | False for SWA; True for CSA/HCA. |
| metadata (output) | Tensor; `int32`, ND, shape (1024,). | Always output. | Must be used as the `metadata` input of the main API with the same set of parameters. | Reusing `metadata` generated with different layout, length, or mode parameters is not allowed. |

Length and optional tensor checks:

| Parameter | Single-Parameter Check | Existence Check | Consistency Check | Feature Cross-Check |
| :--- | :--- | :--- | :--- | :--- |
| cu_seqlens_q | `int32`, ND, shape (b+1,); first element is 0, monotonically non-decreasing, and the last element is q_t. | Required when `layout_q="TND"`; not passed in other layouts. | Must be consistent with q_t, b, and `max_seqlen_q` of `q`. | Tensor values are guaranteed by the user. |
| cu_seqlens_ori_kv | `int32`, ND, shape (b+1,); first element is 0, monotonically non-decreasing, and the last element is ori_kv_t. | Required when `layout_kv="TND"` and `ori_kv` is passed in; not passed in other layouts. | Must be consistent with ori_kv_t, b, and `max_seqlen_ori_kv` of `ori_kv`. | Tensor values are guaranteed by the user. |
| cu_seqlens_cmp_kv | `int32`, ND, shape (b+1,); first element is 0, monotonically non-decreasing, and the last element is cmp_kv_t. | Required when `layout_kv="TND"` and `cmp_kv` is passed in; not passed in other layouts. | Must be consistent with cmp_kv_t, b, and `max_seqlen_cmp_kv` of `cmp_kv`. | Tensor values are guaranteed by the user. |
| seqused_q | `int32`, ND, shape (b,); each item is non-negative and does not exceed the corresponding `q` length. | Optional. | b must be consistent with `q`. | This parameter is not required by Paged Attention. Tensor values are guaranteed by the user. |
| seqused_ori_kv | `int32`, ND, shape (b,); each item is non-negative and does not exceed the corresponding `ori_kv` length. | Required in the PA scenario; optional in other scenarios. | b must be consistent with `ori_kv` and the Block Table. | Determines the effective kv prefix of each sequence in the PA scenario. Tensor values are guaranteed by the user. |
| seqused_cmp_kv | `int32`, ND, shape (b,); each item is non-negative and does not exceed the corresponding `cmp_kv` length. | Optional. | b must be consistent with `cmp_kv`. | Overrides the effective length logic on the cmp side when explicitly passed in. Tensor values are guaranteed by the user. |
| cmp_residual_kv | `int32`, ND, shape (b,); each item satisfies `0 <= value < cmp_ratio`. | Required for CSA/HCA; not passed for SWA. | Must be passed to both the `metadata` API and the main API. | Must satisfy `cmp_len * cmp_ratio + residual = ori_len_for_cmp_mask`. Tensor values are guaranteed by the user. |
| ori_topk_length | `int32`, ND, shape (b, q_s, kv_n) or (q_t, kv_n). | Must be passed in when ori_mask_mode=0 and `ori_sparse_indices` is not empty; passing it in is not supported in other scenarios. | Must be consistent with `ori_topk=0` and `ori_sparse_indices=None`. | Passing it in is not supported when ori_mask_mode is not 0. |
| cmp_topk_length | `int32`, ND, shape (b, q_s, kv_n) or (q_t, kv_n). | Checked only when `cmp_kv` is passed in; must be passed in when cmp_mask_mode=0 and `cmp_sparse_indices` is not empty; passing it in is not supported in other scenarios. | Must be consistent with the state of `cmp_topk` and `cmp_sparse_indices`. | None. |

#### Common Parameter Group

- The data types of `q`, `ori_kv`, and `cmp_kv` must be the same. Only `float16` and `bfloat16` are supported.
- `layout_q` and `layout_kv` support only the `BSND`/`BSND`, `TND`/`TND`, `BSND`/`PA_BBND`, and `TND`/`PA_BBND` combinations. In use cases not involving `PA_BBND`, the two must be the same.
- When `layout_q="BSND"`, `q` must be a 4D tensor. When `layout_q="TND"`, `q` must be a 3D tensor and `cu_seqlens_q` must be passed in.
- When `layout_kv="BSND"` or `layout_kv="PA_BBND"`, kv must be a 4D tensor. When `layout_kv="TND"`, kv must be a 3D tensor. When `layout_kv="TND"`, `cu_seqlens_ori_kv` must be passed in. If `cmp_kv` is passed in, `cu_seqlens_cmp_kv` must also be passed in.
- `metadata` must contain 1024 `int32` elements. `topk_value_mode` supports only 1.
- `ori_kv` and `cmp_kv` allow non-contiguous memory with inter-row padding. The API obtains stride information through aclNN and passes it to the underlying operator.

| Parameter | Single-Parameter Check | Existence Check | Consistency Check | Feature Cross-Check |
| :--- | :--- | :--- | :--- | :--- |
| q | `float16` or `bfloat16`; (b, q_s, q_n, d) for `BSND`, (q_t, q_n, d) for `TND`; blocked if any axis is 0. | Required. | The dtype and shape of `q` and `attention_out` must be the same. | b > 0; d must be 512; q_n/kv_n must be positive integers; the layout must comply with the following table. q_s > 0; 0 < q_n <= 128; q_t > 0. |
| ori_kv | Same data type as `q`; (b, ori_kv_s, kv_n, d) for `BSND`, (ori_kv_t, kv_n, d) for `TND`, (ori_kv_block_nums, ori_kv_block_size, kv_n, d) for `PA_BBND`. | Required in all supported SWA, CSA, and HCA use cases. | When layout_kv is not PA_BBND, layout_q and layout_kv must be the same; when layout_kv is PA_BBND, layout_q can be BSND or TND. | The PA use case requires `ori_block_table` and `seqused_ori_kv`; block_size must meet product specifications. ori_kv_s > 0; ori_kv_n = 1; ori_kv_t > 0; ori_kv_block_nums > 0; 1 <= ori_kv_block_size <= 1024. |
| cmp_kv | Same data type as `q`; (b, cmp_kv_s, kv_n, d) for `BSND`, (cmp_kv_t, kv_n, d) for `TND`, (cmp_kv_block_nums, cmp_kv_block_size, kv_n, d) for `PA_BBND`. | Must not be passed in the SWA use case; required for CSA/HCA. | If cmp_kv is passed in, the dtype of ori_kv and cmp_kv must be the same. | The CSA use case requires `cmp_sparse_indices`; the index is prohibited in the HCA use case; the PA use case requires `cmp_block_table`. cmp_kv_s > 0; cmp_kv_n = 1; cmp_kv_t > 0; cmp_kv_block_nums > 0; 1 <= cmp_kv_block_size <= 1024. |
| metadata | `int32`, ND, shape (1024,). | Required; missing input must be blocked. | Must be the result generated using the layout, sequence lengths, mode, window, compression rate, and kv existence of this call. | Blocked if metadata is inconsistent with the first-phase API or reused from other inputs. |
| softmax_scale | float; must be a finite value. | Optional, default 1.0. | Must match the scaling policy expected by the user. | Not involved in metadata generation; commonly set to `1.0 / sqrt(d)`. |
| layout_q | Only `BSND` and `TND` are supported. | Optional, default `BSND`. | Must be consistent with the rank of q and whether `cu_seqlens_q` is passed in. | See layout matching relationship table. |
| layout_kv | Only `BSND`, `TND`, and `PA_BBND` are supported. | Optional, default `BSND`. | Must be consistent with the ranks of all non-empty kv tensors and the corresponding length/Block Table parameters. | See layout matching relationship table. |
| attention_out (output) | Same data type, rank, and shape as `q`. | Always output. | Corresponds one-to-one with q. | Blocked before output construction if `q` is empty or the shape is invalid. |

Layout matching relationship table:

| layout_q | layout_kv | q/kv Dimension | softmax_lse Dimension (When Enabled) |
| :--- | :--- | :--- | :--- |
| `BSND` | `BSND` | q: (b, q_s, q_n, d)<br>kv: (b, ori_kv_s, kv_n, d) | (b, kv_n, q_s, q_n/kv_n) |
| `TND` | `TND` | q: (q_t, q_n, d)<br>kv: (ori_kv_t, kv_n, d) | (kv_n, q_t, q_n/kv_n) |
| `BSND` | `PA_BBND` | q: (b, q_s, q_n, d)<br>kv: (ori_kv_block_nums, ori_kv_block_size, kv_n, d) | (b, kv_n, q_s, q_n/kv_n) |
| `TND` | `PA_BBND` | q: (q_t, q_n, d)<br>kv: (ori_kv_block_nums, ori_kv_block_size, kv_n, d) | (kv_n, q_t, q_n/kv_n) |

#### Compute Mode Parameter Group

| Use Case | Required Inputs and Attributes | Inputs and Attributes Not Allowed or Fixed |
| :--- | :--- | :--- |
| SWA | Pass in only `ori_kv`. | Do not pass in `cmp_kv`, `cmp_sparse_indices`, or `cmp_block_table`; `cmp_topk=0` and `cmp_mask_mode=0`. |
| CSA | Pass in `ori_kv`, `cmp_kv`, and `cmp_sparse_indices`; `cmp_mask_mode=3` and `cmp_topk` is non-zero; pass in `cmp_residual_kv` when `cmp_ratio!=1`. | `cmp_residual_kv` is not allowed when `cmp_mask_mode=0` or `cmp_ratio=1`. |
| HCA | Pass in `ori_kv` and `cmp_kv`; `cmp_mask_mode=3`; pass in `cmp_residual_kv` when `cmp_ratio!=1`. | Do not pass in `cmp_sparse_indices`; `cmp_topk=0`; `cmp_residual_kv` is not allowed when `cmp_mask_mode=0` or `cmp_ratio=1`. |

| Parameter | Single-Parameter Check | Existence Check | Consistency Check | Feature Cross-Check |
| :--- | :--- | :--- | :--- | :--- |
| ori_sparse_indices | `int32`, ND. | Must not be passed in. | Must be consistent with `ori_topk=0` and `ori_topk_length=None`. | When mask mode != 0, the valid length must be consistent with the sequence length actually participating in computation, and topk_length cannot be passed in. When mask mode == 0, ori_kv_k needs to be greater than or equal to the corresponding topklength. |
| cmp_sparse_indices | `int32`, ND; (b, q_s, kv_n, cmp_kv_k) for `BSND`, (q_t, kv_n, cmp_kv_k) for `TND`; values must be -1 or valid cmp token indices. | Required for CSA only; must not be passed in for SWA/HCA. | b/q_t and kv_n must be consistent with `q`, and cmp_kv_k must be consistent with `cmp_topk`. | <term>Atlas A3 products</term>: intercepted when cmp_kv_k is greater than 8192 or less than or equal to 0; <term>Ascend 950PR&950DT products</term>: intercepted when cmp_kv_k is less than or equal to 0. Invalid positions are filled with -1, and users must ensure the actual index values. |
| ori_topk_length | `int32`, ND, shape (b, q_s, kv_n) or (q_t, kv_n). | `ori_topk_length` must be passed in for ori+cmp sparsity. | Must be consistent with `ori_sparse_indices=None` and `ori_topk=0`. | Optional for CSA/ALL_CSA use cases; cannot be passed in for other use cases. If passed in, `seqused_ori_kv` is not required. |
| cmp_topk_length | `int32`, ND, shape (b, q_s, kv_n) or (q_t, kv_n). | `cmp_topk_length` must be passed in for ori+cmp sparsity. | Must be consistent with the states of `cmp_sparse_indices` and `cmp_topk`. | Optional for CSA/ALL_CSA use cases; cannot be passed in for other use cases. If passed in, `seqused_cmp_kv` is not required. |
| cmp_ratio | int32; 1 for the SWA use case, range 1-128 for CSA/HCA use cases. | Optional, default 1. | Must be consistent with `metadata`, the length of `cmp_kv`, and `cmp_residual_kv`. | <term>Atlas A3 products</term>: no restriction for SWA; intercepted when the CSA value is not 1, 2, or 4 or the HCA value is not 128; <term>Ascend 950PR&950DT products</term>: intercepted when the SWA value is not 1 or the CSA/HCA value is out of 1-128. |
| topk_value_mode | int32; only 1 is supported. | Optional, default 1. | Must be consistent with the index value convention of `cmp_sparse_indices`. | Not involved in `metadata` generation; values other than 1 should be intercepted. |

#### SeqLengths and Mask Parameter Group

- `cu_seqlens_q`, `cu_seqlens_ori_kv`, and `cu_seqlens_cmp_kv` must be the accumulated numbers of valid tokens in the current and preceding batches. The first element is 0, and each subsequent element must not be less than the previous one.
- `seqused_q`, `seqused_ori_kv`, and `seqused_cmp_kv` indicate the actual numbers of valid tokens in each batch. `seqused_cmp_kv` is optional for all kv layouts. When explicitly passed in, it overrides the logical valid length on the cmp side.
- `cmp_residual_kv[i]` must be less than `cmp_ratio`. In CSA and HCA use cases, its length must be equal to the batch size.
- For details about the meanings of `ori_mask_mode` and `cmp_mask_mode`, see [sparseMode Introduction](../../../docs/en/context/sparse_mode_introduction.md). In the current specification, `ori_mask_mode` supports 0, 3, and 4, `cmp_mask_mode` supports 3 and 4, and `ori_win_left` and `ori_win_right` support -1 and non-negative values.

| Parameter | Single-Parameter Check | Existence Check | Consistency Check | Feature Cross-Check |
| :--- | :--- | :--- | :--- | :--- |
| cu_seqlens_q | `int32`, ND, shape (b+1,); the first element is 0, the values are monotonically non-decreasing, and the last element is q_t. | Required when `layout_q="TND"`; must not be passed in for other layouts. | Consistent with q_t, B, `max_seqlen_q`, and `metadata` of `q`. | The specific length values are guaranteed by the user. |
| cu_seqlens_ori_kv | `int32`, ND, shape (b+1,); the first element is 0, the values are monotonically non-decreasing, and the last element is ori_kv_t. | Required when `layout_kv="TND"` and `ori_kv` is not empty; must not be passed in for other layouts. | Consistent with ori_kv_t, b, `max_seqlen_ori_kv`, and `metadata` of `ori_kv`. | The specific length values are guaranteed by the user. |
| cu_seqlens_cmp_kv | `int32`, ND, shape (b+1,); the first element is 0, the values are monotonically non-decreasing, and the last element is cmp_kv_t. | Required when `layout_kv="TND"` and `cmp_kv` is not empty; must not be passed in for other layouts. | Consistent with cmp_kv_t, b, `max_seqlen_cmp_kv`, and `metadata` of `cmp_kv`. | The specific length values are guaranteed by the user. |
| seqused_q | `int32`, ND, shape (b,); each item is a non-negative integer and does not exceed the length of the corresponding q. | Optional. | b must be consistent with `q` and `metadata`. | The specific tensor values are guaranteed by the user. |
| seqused_ori_kv | `int32`, ND, shape (b,); each item is a non-negative integer and does not exceed the length of the corresponding `ori_kv`. | Required in the PA scenario; optional in other scenarios. | b must be consistent with `ori_kv`, `ori_block_table`, and `metadata`. | None. |
| seqused_cmp_kv | `int32`, ND, shape (b,); each item is a non-negative integer and does not exceed the length of the corresponding `cmp_kv`. | Optional. | b must be consistent with `cmp_kv` and `metadata`. | None. |
| cmp_residual_kv | `int32`, ND, shape (b,); each item is in the range [0, cmp_ratio). | Not allowed when `cmp_mask_mode=0` or `cmp_ratio=1`. | Must be consistent with `metadata`, `cmp_ratio`, and the length of `cmp_kv`. | The recovered length must satisfy `cmp_len * cmp_ratio + residual = ori_len_for_cmp_mask`. |
| ori_mask_mode | `int32`; the interface definition supports 0, 3, and 4. | Optional. | Must be consistent with `metadata`. | Currently supports 0, 3, and 4. |
| cmp_mask_mode | `int32`; the interface definition supports 0 and 3. | Optional. | Must be consistent with `metadata`. | Currently supports 0 and 3. |
| ori_win_left | `int32`; the interface definition is -1 or non-negative. | Optional. | Must be consistent with `metadata`. | <term>Atlas A3 products</term>: intercepted when the value is not 127; <term>Ascend 950PR&950DT products</term>: intercepted when the value is less than -1. |
| ori_win_right | `int32`; the interface definition is -1 or non-negative. | Optional. | Must be consistent with `metadata`. | <term>Atlas A3 products</term>: intercepted when the value is not 0; <term>Ascend 950PR&950DT products</term>: intercepted when the value is less than -1. |

#### Paged Attention Parameter Group

- When `layout_kv="PA_BBND"`, `seqused_ori_kv` and `ori_block_table` must be passed in; when `cmp_kv` is passed in, `cmp_block_table` must also be passed in.
- The `block_size` of PagedAttention must be greater than 0. For <term>Atlas A2 products</term> and <term>Atlas A3 products</term>, an error is reported when the value is not a multiple of 16 or is greater than 1024; for <term>Ascend 950PR&950DT products</term>, an error is reported when the value is less than or equal to 0.
- When topk_value_mode=2, `ori_block_table` and `cmp_block_table` can be omitted.

| Parameter | Single-Parameter Check | Existence Check | Consistency Check | Feature Cross-Check |
| :--- | :--- | :--- | :--- | :--- |
| ori_block_table | `int32`, ND, shape (b, max_num_blocks_per_seq); values can only be positive integers. | Must be passed in when `layout_kv="PA_BBND"`; must not be passed in for other layouts. | b must be consistent with `q`, `ori_kv`, `seqused_ori_kv`, and metadata. | When ori_block_table exists, `seqused_ori_kv` must be passed in; when PagedAttention is enabled, `ori_block_table` must not be empty. |
| cmp_block_table | `int32`, ND, shape (b, max_num_blocks_per_seq); values can only be positive integers. | Must be passed in when `layout_kv="PA_BBND"` and `cmp_kv` is not empty; must not be passed in for other cases. | b must be consistent with `q`, `cmp_kv`, `seqused_cmp_kv` (if passed in), and metadata. | When cmp_block_table exists, `seqused_cmp_kv` must be passed in; when PagedAttention is enabled, `cmp_block_table` must not be empty. |

#### Sinks Parameter Group

| Parameter | Single-Parameter Check | Existence Check | Consistency Check | Feature Cross-Check |
| :--- | :--- | :--- | :--- | :--- |
| sinks | `float32`, ND, shape (q_n,). | Required in the current version. | Length must be equal to q_n of `q`. | None. |

#### SoftmaxLse Parameter Group

| Parameter | Single-Parameter Check | Existence Check | Consistency Check | Feature Cross-Check |
| :--- | :--- | :--- | :--- | :--- |
| return_softmax_lse | bool; true indicates enabling softmax_lse, false indicates disabling softmax_lse. | Optional parameter, default value is False. | When return_softmax_lse is false, a tensor with shape [1] and value 0 is output. When return_softmax_lse is true, the shape of softmax_lse is related to layout_q as follows: when layout_q is BSND, the shape of softmax_lse is (b, kv_n, q_s, q_n/kv_n); when layout_q is TND, the shape of softmax_lse is (kv_n, q_t, q_n/kv_n). | None. |
| softmax_lse | `float32`, ND; BSND is (b, kv_n, q_s, q_n/kv_n), TND is (kv_n, q_t, q_n/kv_n). | Valid results are returned when `return_softmax_lse=True`. | kv_n, q_s/q_t, and q_n/kv_n must be consistent with `q` and kv. | A float32 scalar placeholder tensor when `return_softmax_lse=False`; must not be read as a valid LSE. |

## Deterministic Computation<a name="zh-cn_topic_sparse_flash_mla_deterministic"></a>

- Defaults to a deterministic implementation.
- Batch invariance is supported by default.

## Example<a name="zh-cn_topic_sparse_flash_mla_examples"></a>

### SWA Use Case (BSND Input)

```python
import math
import torch
import torch_npu
import cann_ops_transformer

torch_npu.npu.set_device(0)

dtype = torch.bfloat16
b = 1
q_s = 16
ori_kv_s = 64
q_n = 64
kv_n = 1
d = 512
cmp_ratio = 1  # In the SWA example, only ori_kv is passed in. cmp_ratio does not participate in compressed kv computation, so keep the default value 1.

q = torch.randn(b, q_s, q_n, d, dtype=dtype, device="npu")
ori_kv = torch.randn(b, ori_kv_s, kv_n, d, dtype=dtype, device="npu")
sinks = torch.zeros(q_n, dtype=torch.float32, device="npu")

metadata = cann_ops_transformer.sparse_flash_mla_metadata(
    q_n,
    kv_n,
    d,
    batch_size=b,
    max_seqlen_q=q_s,
    max_seqlen_ori_kv=ori_kv_s,
    ori_topk=0,
    cmp_topk=0,
    cmp_ratio=cmp_ratio,
    ori_mask_mode=4,
    cmp_mask_mode=0,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="BSND",
    layout_kv="BSND",
    has_ori_kv=True,
    has_cmp_kv=False,
)

attn_out, softmax_lse = cann_ops_transformer.sparse_flash_mla(
    q,
    ori_kv=ori_kv,
    sinks=sinks,
    metadata=metadata,
    softmax_scale=1.0 / math.sqrt(d),
    cmp_ratio=cmp_ratio,
    ori_mask_mode=4,
    cmp_mask_mode=0,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="BSND",
    layout_kv="BSND",
    return_softmax_lse=False,
)
torch_npu.npu.synchronize()
assert attn_out.shape == q.shape
assert attn_out.dtype == q.dtype
assert softmax_lse.shape == torch.Size([])
assert torch.isfinite(attn_out.float()).all().item()
```

### HCA Use Case (BSND Input)

```python
import math
import torch
import torch_npu
import cann_ops_transformer

torch_npu.npu.set_device(0)

dtype = torch.bfloat16
b = 1
q_s = 16
ori_kv_s = 128
cmp_kv_s = 1
q_n = 64
kv_n = 1
d = 512
cmp_ratio = 128

q = torch.randn(b, q_s, q_n, d, dtype=dtype, device="npu")
ori_kv = torch.randn(b, ori_kv_s, kv_n, d, dtype=dtype, device="npu")
cmp_kv = torch.randn(b, cmp_kv_s, kv_n, d, dtype=dtype, device="npu")
cmp_residual_kv = torch.zeros(b, dtype=torch.int32, device="npu")
sinks = torch.zeros(q_n, dtype=torch.float32, device="npu")

metadata = cann_ops_transformer.sparse_flash_mla_metadata(
    q_n,
    kv_n,
    d,
    batch_size=b,
    max_seqlen_q=q_s,
    max_seqlen_ori_kv=ori_kv_s,
    max_seqlen_cmp_kv=cmp_kv_s,
    cmp_residual_kv=cmp_residual_kv,
    ori_topk=0,
    cmp_topk=0,
    cmp_ratio=cmp_ratio,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="BSND",
    layout_kv="BSND",
    has_ori_kv=True,
    has_cmp_kv=True,
)

attn_out, softmax_lse = cann_ops_transformer.sparse_flash_mla(
    q,
    ori_kv=ori_kv,
    cmp_kv=cmp_kv,
    cmp_residual_kv=cmp_residual_kv,
    sinks=sinks,
    metadata=metadata,
    softmax_scale=1.0 / math.sqrt(d),
    cmp_ratio=cmp_ratio,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="BSND",
    layout_kv="BSND",
    return_softmax_lse=False,
)
torch_npu.npu.synchronize()
assert attn_out.shape == q.shape
assert attn_out.dtype == q.dtype
assert softmax_lse.shape == torch.Size([])
assert torch.isfinite(attn_out.float()).all().item()
```

### CSA use case (TND input, using cmp_residual_kv)

```python
import math
import torch
import torch_npu
import cann_ops_transformer

torch_npu.npu.set_device(0)

dtype = torch.float16
b = 1
q_lens = [1]
ori_lens = [6]
cmp_lens = [1]
q_n = 64
kv_n = 1
d = 512
cmp_kv_k = 512
cmp_ratio = 4

cu_q = torch.tensor([0, 1], dtype=torch.int32, device="npu")
cu_ori = torch.tensor([0, 6], dtype=torch.int32, device="npu")
cu_cmp = torch.tensor([0, 1], dtype=torch.int32, device="npu")
cmp_residual_kv = torch.tensor([2], dtype=torch.int32, device="npu")

q = torch.randn(sum(q_lens), q_n, d, dtype=dtype, device="npu")
ori_kv = torch.randn(sum(ori_lens), kv_n, d, dtype=dtype, device="npu")
cmp_kv = torch.randn(sum(cmp_lens), kv_n, d, dtype=dtype, device="npu")
sinks = torch.zeros(q_n, dtype=torch.float32, device="npu")

cmp_sparse_indices = torch.full((sum(q_lens), kv_n, cmp_kv_k), -1, dtype=torch.int32, device="npu")
cmp_sparse_indices[:, :, :1] = torch.arange(1, dtype=torch.int32, device="npu").view(1, 1, 1)

metadata = cann_ops_transformer.sparse_flash_mla_metadata(
    q_n,
    kv_n,
    d,
    cu_seqlens_q=cu_q,
    cu_seqlens_ori_kv=cu_ori,
    cu_seqlens_cmp_kv=cu_cmp,
    max_seqlen_q=max(q_lens),
    max_seqlen_ori_kv=max(ori_lens),
    max_seqlen_cmp_kv=max(cmp_lens),
    ori_topk=0,
    cmp_topk=cmp_kv_k,
    cmp_ratio=cmp_ratio,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    cmp_residual_kv=cmp_residual_kv,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="TND",
    layout_kv="TND",
    has_ori_kv=True,
    has_cmp_kv=True,
)

attn_out, softmax_lse = cann_ops_transformer.sparse_flash_mla(
    q,
    ori_kv=ori_kv,
    cmp_kv=cmp_kv,
    cmp_sparse_indices=cmp_sparse_indices,
    cu_seqlens_q=cu_q,
    cu_seqlens_ori_kv=cu_ori,
    cu_seqlens_cmp_kv=cu_cmp,
    cmp_residual_kv=cmp_residual_kv,
    sinks=sinks,
    metadata=metadata,
    softmax_scale=1.0 / math.sqrt(d),
    cmp_ratio=cmp_ratio,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="TND",
    layout_kv="TND",
    return_softmax_lse=False,
)
torch_npu.npu.synchronize()
assert attn_out.shape == q.shape
assert attn_out.dtype == q.dtype
assert softmax_lse.shape == torch.Size([])
assert torch.isfinite(attn_out.float()).all().item()
```

### CP Splitting Example (TND + CSA, rank0 splits the second sequence)

The following example simulates two CP ranks sequentially in a single process to show the relationship between the global TND data and the input parameters of each rank. Assume there are 2 sequences in total, with `cmp_ratio=4`:

| View | q range | ori_kv range | cmp_kv range | cu_seqlens_q | cu_seqlens_ori_kv | cu_seqlens_cmp_kv | cmp_residual_kv |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Global | seq0 `[0,16)`, seq1 `[0,18)` | seq0 `[0,16)`, seq1 `[0,18)` | seq0 `[0,4)`, seq1 `[0,4)` | `[0,16,34]` | `[0,16,34]` | `[0,4,8]` | `[0,2]` |
| rank0 | seq0 `[0,16)`, seq1 `[0,8)` | seq0 `[0,16)`, seq1 `[0,8)` | seq0 `[0,4)`, seq1 `[0,2)` | `[0,16,24]` | `[0,16,24]` | `[0,4,6]` | `[0,0]` |
| rank1 | seq1 `[8,18)` | seq1 `[0,18)` | seq1 `[0,4)` | `[0,10]` | `[0,18]` | `[0,4]` | `[2]` |

Although rank1 computes only `[8,18)` of seq1, `ori_kv` and `cmp_kv` must be passed in as the prefix up to the end of the current position. In this case, `ori_prefix_len - q_len = 18 - 10 = 8`, and the q start point derived by the kernel is exactly the CP split point. Each local batch must satisfy `cmp_len * cmp_ratio + cmp_residual_kv[b] == ori_prefix_len`.

```python
import math
import torch
import torch_npu
import cann_ops_transformer


torch_npu.npu.set_device(0)

dtype = torch.float16
cmp_ratio = 4
cmp_kv_k = 512
q_n = 64
kv_n = 1
d = 512

# Global packed TND view: seq0 has length 16, and seq1 has length 18.
global_q_lens = [16, 18]
global_ori_lens = [16, 18]
global_cmp_lens = [4, 4]
global_cmp_residual = [0, 2]

q_global = torch.randn(sum(global_q_lens), q_n, d, dtype=dtype, device="npu")
ori_global = torch.randn(sum(global_ori_lens), kv_n, d, dtype=dtype, device="npu")
cmp_global = torch.randn(sum(global_cmp_lens), kv_n, d, dtype=dtype, device="npu")
sinks = torch.zeros(q_n, dtype=torch.float32, device="npu")


def make_cu(lengths):
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    return torch.tensor(cu, dtype=torch.int32, device="npu")


def make_cmp_sparse_indices(q_lens, ori_prefix_lens, cmp_lens):
    indices = torch.full((sum(q_lens), kv_n, cmp_kv_k), -1, dtype=torch.int32, device="npu")
    q_base = 0
    for q_len, ori_prefix_len, cmp_len in zip(q_lens, ori_prefix_lens, cmp_lens):
        q_start = ori_prefix_len - q_len
        for row in range(q_len):
            q_pos = q_start + row
            cmp_end = min(cmp_len, (q_pos + 1) // cmp_ratio)
            if cmp_end > 0:
                indices[q_base + row, :, :cmp_end] = torch.arange(
                    cmp_end, dtype=torch.int32, device="npu"
                ).view(1, cmp_end)
        q_base += q_len
    return indices


def run_one_rank(name, q, ori_kv, cmp_kv, q_lens, ori_prefix_lens, cmp_lens, residuals):
    cu_q = make_cu(q_lens)
    cu_ori = make_cu(ori_prefix_lens)
    cu_cmp = make_cu(cmp_lens)
    cmp_residual_kv = torch.tensor(residuals, dtype=torch.int32, device="npu")
    cmp_sparse_indices = make_cmp_sparse_indices(q_lens, ori_prefix_lens, cmp_lens)

    metadata = cann_ops_transformer.sparse_flash_mla_metadata(
        q_n,
        kv_n,
        d,
        cu_seqlens_q=cu_q,
        cu_seqlens_ori_kv=cu_ori,
        cu_seqlens_cmp_kv=cu_cmp,
        max_seqlen_q=max(q_lens),
        max_seqlen_ori_kv=max(ori_prefix_lens),
        max_seqlen_cmp_kv=max(cmp_lens),
        ori_topk=0,
        cmp_topk=cmp_kv_k,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        cmp_residual_kv=cmp_residual_kv,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="TND",
        has_ori_kv=True,
        has_cmp_kv=True,
    )

    attn_out, softmax_lse = cann_ops_transformer.sparse_flash_mla(
        q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        cmp_sparse_indices=cmp_sparse_indices,
        cu_seqlens_q=cu_q,
        cu_seqlens_ori_kv=cu_ori,
        cu_seqlens_cmp_kv=cu_cmp,
        cmp_residual_kv=cmp_residual_kv,
        sinks=sinks,
        metadata=metadata,
        softmax_scale=1.0 / math.sqrt(d),
        cmp_ratio=cmp_ratio,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="TND",
        return_softmax_lse=False,
    )
    torch_npu.npu.synchronize()
    assert attn_out.shape == q.shape, name
    assert attn_out.dtype == q.dtype, name
    assert softmax_lse.shape == torch.Size([]), name
    assert torch.isfinite(attn_out.float()).all().item(), name
    return attn_out


# rank0: contains the complete seq0 and splits the first 8 tokens of seq1.
rank0_q = torch.cat([q_global[0:16], q_global[16:24]], dim=0)
rank0_ori = torch.cat([ori_global[0:16], ori_global[16:24]], dim=0)
rank0_cmp = torch.cat([cmp_global[0:4], cmp_global[4:6]], dim=0)
rank0_out = run_one_rank(
    "rank0", rank0_q, rank0_ori, rank0_cmp,
    q_lens=[16, 8], ori_prefix_lens=[16, 8], cmp_lens=[4, 2], residuals=[0, 0]
)

# rank1: computes only the last 10 tokens of seq1, but passes the prefix of seq1 up to 18 as ori_kv/cmp_kv.
rank1_q = q_global[24:34]
rank1_ori = ori_global[16:34]
rank1_cmp = cmp_global[4:8]
rank1_out = run_one_rank(
    "rank1", rank1_q, rank1_ori, rank1_cmp,
    q_lens=[10], ori_prefix_lens=[18], cmp_lens=[4], residuals=[2]
)

seq0_out = rank0_out[:16]
seq1_out = torch.cat([rank0_out[16:24], rank1_out], dim=0)
assert seq0_out.shape == (16, q_n, d)
assert seq1_out.shape == (18, q_n, d)
```
