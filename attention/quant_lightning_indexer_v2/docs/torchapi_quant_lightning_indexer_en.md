# quant\_lightning\_indexer

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T12:41:59.113Z -->

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

## Function

- **Description**:

  The `quant_lightning_indexer_metadata` API generates a task list that contains the batch and head of the start and end points of the Attention computation task on each AI Core, as well as the tile indices of Q and K, for use by the subsequent `quant_lightning_indexer` operator.

  The `quant_lightning_indexer` API obtains the top-k positions corresponding to each token through a series of operations. The main computation procedure is as follows:

  1. Multiply the input parameter `q` ($Q_{index}^{Quant}\in\R^{g\times d}$) of a token by the given context `k` ($K_{index}^{Quant}\in\R^{S_{k}\times d}$) to obtain the relevance.
  2. The relevance results are multiplied by the dequantization scales `q_descale` ($Scale_Q$) and `k_descale` ($Scale_K^T$) corresponding to `q` and `k`. After the invalid negative relevance signals are filtered out through the activation function $ReLU$, the relevance score vector between the current token and all preceding tokens is obtained.
  3. After being multiplied by the weight coefficient `w` ($W$), the top $Top-k$ index values are selected along the g dimension to obtain the output $sparseIndices$, and the corresponding $sparseValues$ are output as the input to Attention.

- **Formula**:

  $$
  out = Top\text{-}k\left\{[1]_{1\times g}@\left[\left(W@[1]_{1\times S_{k}}\right)\odot ReLU\left(\left(Scale_Q@Scale_K^T\right)\odot\left(Q_{index}^{Quant}@{\left(K_{index}^{Quant}\right)}^T\right)\right)\right]\right\}
  $$

## Function Prototype

Before calling the quant_lightning_indexer API, call the prerequisite API quant_lightning_indexer_metadata to complete the load balancing computation for quant_lightning_indexer.

```python
cann_ops_transformer.quant_lightning_indexer_metadata(
  num_heads_q,
  num_heads_k,
  head_dim,
  topk,
  quant_mode,
  *,
  cu_seqlens_q=None,
  cu_seqlens_k=None,
  seqused_q=None,
  seqused_k=None,
  cmp_residual_k=None,
  batch_size=0,
  max_seqlen_q=-1,
  max_seqlen_k=-1,
  layout_q="BSND",
  layout_k="BSND",
  mask_mode=0,
  cmp_ratio=1
) -> Tensor
```

```python
cann_ops_transformer.quant_lightning_indexer(
  q,
  k,
  w,
  q_descale,
  k_descale,
  topk,
  quant_mode,
  *,
  cu_seqlens_q=None,
  cu_seqlens_k=None,
  seqused_q=None,
  seqused_k=None,
  cmp_residual_k=None,
  block_table=None,
  output_idx_offset=None,
  metadata=None,
  max_seqlen_q=-1,
  layout_q="BSND",
  layout_k="BSND",
  mask_mode=0,
  cmp_ratio=1,
  return_value=0
) -> (Tensor, Tensor)
```

## Parameters

>**NOTE:**
>
> The meanings of the dimensions of parameters q, k, w, q_descale, and k_descale: b (batch size) denotes the batch size of input samples, q_s denotes the sequence length of input samples for q, k_s denotes the sequence length of input samples for k, q_n denotes the number of heads for q, k_n denotes the number of heads for k, d (head dim) denotes the dimension of the attention head, q_t denotes the cumulative sum of the sequence lengths of input samples for q, k_t denotes the cumulative sum of the sequence lengths of input samples for k, block_num denotes the total number of blocks in the PageAttention use case, block_size denotes the number of tokens in each block in the PageAttention use case, and g denotes the group size of GQA (g = q_n / k_n). The value of d in parameter q is equal to that in parameter k, and d currently only supports 128.

### quant_lightning_indexer_metadata

| Name | Type | Required/Optional | Description | Data Type | Dimension (Shape) |
|--------|----------|-----------|------|----------|-------------|
| num_heads_q | int | Required | Number of heads for q. | int64 | - |
| num_heads_k | int | Required | Number of heads for k. Currently only supports 1. | int64 | - |
| head_dim | int | Required | Dimension of the attention head. Currently only supports 128. | int64 | - |
| topk | int | Required | Number of k token indices to be retained in the topK stage. Currently only supports [1, 8192]. | int64 | - |
| quant_mode | int | Required | Quantization mode. Currently supports 1/2/3/4/5. 1 indicates qk: fp8(e4m3fn) per-token-head, scale: fp32; 2 indicates qk: int8 per-token-head, scale: fp16, w: fp16; 3 indicates qk: mxfp8(e4m3fn), scale: fp8(e8m0); 4 indicates qk: hifloat8 per-tensor, scale: fp32; 5 indicates qk: mxfp4(e2m1), scale: fp8(e8m0). | int64 | - |
| cu_seqlens_q | Tensor | Optional | Cumulative prefix sum of the valid Sequence Length of q in each batch. Must be passed only when layout_q is TND. The first value is fixed to 0. The data format is ND. Non-contiguous tensors are supported. | int32 | (b+1, ) |
| cu_seqlens_k | Tensor | Optional | Cumulative prefix sum of the valid Sequence Length of k in each batch. Must be passed only when layout_k is TND. The first value is fixed to 0. The data format is ND. Non-contiguous tensors are supported. | int32 | (b+1, ) |
| seqused_q | Tensor | Optional | Sequence Length of q actually participating in the computation in each batch. The data format is ND. Non-contiguous tensors are supported. | int32 | (b, ) |
| seqused_k | Tensor | Optional | Sequence Length of k actually participating in the computation in each batch. The data format is ND. Non-contiguous tensors are supported. | int32 | (b, ) |
| cmp_residual_k | Tensor | Optional | Remainder of the Sequence Length of cmp_kv before compression divided by cmp_ratio in each batch. Used together with cmp_ratio to implement mask and load computation for the cmp_kv part. Must be passed when cmp_ratio is not 1 and mask_mode is 3. Must satisfy 0 \<= cmp_residual_k\[i\] \< cmp_ratio. The data format is ND. Non-contiguous tensors are supported. | int32 | (b, ) |
| batch_size | int | Optional | Number of batches. The default value is 0. | int64 | - |
| max_seqlen_q | int | Optional | Maximum Sequence Length of q. -1 indicates any possible length. The default value is -1. | int64 | - |
| max_seqlen_k | int | Optional | Maximum Sequence Length of k. -1 indicates any possible length. The default value is -1. | int64 | - |
| layout_q | str | Optional | Layout of q. Supports BSND and TND. The default value is BSND. | string | - |
| layout_k | str | Optional | Layout of k. Supports BSND, TND, and PA_BBND. The default value is BSND. | string | - |
| mask_mode | int | Optional | Sparse mode. 0 indicates No mask, and 3 indicates rightDownCausal mode. The default value is 0. | int64 | - |
| cmp_ratio | int | Optional | Compression ratio of k. Value range: [1, 128]. The default value is 1, indicating no compression. | int64 | - |

### quant_lightning_indexer

| Name | Type | Required/Optional | Description | Data Type | Dimension (Shape) |
|--------|----------|-----------|------|----------|-------------|
| q | Tensor | Required | Quantized input $Q_{index}^{Quant}$ in the formula. Empty tensor not supported. Data format: ND. | int8, float8_e4m3fn, HIfloat8, float4_e2m1 | shape is (b,q_s,q_n,d) when layout_q is BSND; shape is (q_t,q_n,d) when layout_q is TND |
| k | Tensor | Required | Quantized input $K_{index}^{Quant}$ in the formula. Empty tensor not supported. Data format: ND. Only dimension 0 can be non-contiguous in PA_BBND use cases. In PA_BBND use cases, block_size must be a multiple of 16, with a maximum of 1024. | int8, float8_e4m3fn, HIfloat8, float4_e2m1 | shape is (b,k_s,k_n,d) when layout_k is BSND; shape is (k_t,k_n,d) when layout_k is TND; shape is (block_num,block_size,k_n,d) when layout_k is PA_BBND |
| w | Tensor | Required | Weight coefficient $W$ in the formula. Empty tensor not supported. Data format: ND. | float16, float32 | shape is (b,q_s,q_n) when layout_q is BSND; shape is (q_t,q_n) when layout_q is TND |
| q_descale | Tensor | Required | $Scale_Q$ in the formula, which is the dequantization scale of Index q. Empty tensor not supported. Data format: ND. | float16, float32, float8_e8m0 | When quant_mode is 3/5, shape is (b,q_s,q_n,d/64,2) when layout_q is BSND, and shape is (q_t,q_n,d/64,2) when layout_q is TND; when quant_mode is 4, shape must be (1,); in other use cases, the shape is the same as that of w |
| k_descale | Tensor | Required | $Scale_K$ in the formula, which is the dequantization scale of Index k. Empty tensor not supported. Data format: ND. Only dimension 0 can be non-contiguous in PA_BBND use cases. | float16, float32, float8_e8m0 | When quant_mode is 3/5, shape is (block_num,block_size,k_n,d/64,2) when layout_k is PA_BBND, shape is (b,k_s,k_n,d/64,2) when layout_k is BSND, and shape is (k_t,k_n,d/64,2) when layout_k is TND; when quant_mode is 4, shape must be (1,); in other use cases, the shapes are (block_num,block_size,k_n), (b,k_s,k_n), or (k_t,k_n) respectively |
| topk | int | Required | Number of k token indices to retain in the topK phase. Currently supports [1, 8192]. | int64 | - |
| quant_mode | int | Required | Quantization mode. Currently supports 1/2/3/4/5. 1 indicates qk: fp8(e4m3fn) per-token-head, scale: fp32; 2 indicates qk: int8 per-token-head, scale: fp16, w: fp16; 3 indicates qk: mxfp8(e4m3fn), scale: fp8(e8m0); 4 indicates qk: hifloat8 per-tensor, scale: fp32; 5 indicates qk: mxfp4(e2m1), scale: fp8(e8m0). | int64 | - |
| cu_seqlens_q | Tensor | Optional | Cumulative sum of the number of valid q tokens in the current and preceding batches. The value of each element must be greater than or equal to that of the preceding element. Must be passed only when layout_q is TND. The first value is fixed to 0. Data format: ND. | int32 | (b+1,) |
| cu_seqlens_k | Tensor | Optional | Cumulative sum of the number of valid k tokens in the current and preceding batches. The value of each element must be greater than or equal to that of the preceding element. Must be passed only when layout_k is TND. The first value is fixed to 0. Data format: ND. | int32 | (b+1,) |
| seqused_q | Tensor | Optional | Actual used length of q in each batch. The number of valid tokens in each batch must not exceed dimension S of q and must not be less than 0. Data format: ND. | int32 | (b,) |
| seqused_k | Tensor | Optional | Actual used length of k in each batch. The number of valid tokens in each batch must not exceed dimension S of k and must not be less than 0. Data format: ND. Must be passed when layout_k is PA_BBND. | int32 | (b,) |
| cmp_residual_k | Tensor | Optional | Remainder of the number of k tokens before compression divided by cmp_ratio, which must satisfy 0 \<= cmp_residual_k\[i\] \< cmp_ratio. Used when mask_mode is 3 and cmp_ratio is not 1. Data format: ND. | int32 | (b,) |
| block_table | Tensor | Optional | Block mapping table used for KV storage in PageAttention. Empty tensor not supported. Must be passed when layout_k is PA_BBND. Data format: ND. | int32 | (b, k_s_max/block_size) |
| output_idx_offset | Tensor | Optional | Offset to be added to the output indices of the topK result. The value must be greater than 0, and the topk index plus the offset must not exceed the maximum value of int32. Data format: ND. | int32 | shape is (b,q_s,k_n) when layout_q is BSND; shape is (q_t,k_n) when layout_q is TND |
| metadata | Tensor | Optional | Core allocation information passed by the quant_lightning_indexer_metadata operator, including the number of cores used, block size, and the start point of data processed by each core. Empty tensor not supported. Data format: ND. | int32 | (8 * (1 + aic_num + aiv_num) + 256,) |
| max_seqlen_q | int | Optional | Maximum sequence length of q. -1 indicates any possible length. Default value is -1. | int64 | - |
| layout_q | str | Optional | Data layout format of input q. Supports BSND and TND. Default value is BSND. | string | - |
| layout_k | str | Optional | Data layout format of input k. Supports BSND, TND, and PA_BBND. Default value is BSND. | string | - |
| mask_mode | int | Optional | Mask mode. 0 indicates the defaultMask mode, and 3 indicates the rightDownCausal mode. Default value is 0. | int64 | - |
| cmp_ratio | int | Optional | Used for sparse computation, indicating the compression ratio of k. Default value is 1, indicating no compression. | int64 | - |
| return_value | int | Optional | Whether to return the sparseValues corresponding to sparseIndices. 0 indicates not returning, and 1 indicates returning. Default value is 0. | int64 | - |

## Return Values

### quant_lightning_indexer_metadata

| Name | Type | Required/Optional | Description | Data Type | Dimension (Shape) |
|--------|----------|-----------|------|----------|-------------|
| metadata | Tensor | Required | Index of batch, head, and Q and K chunking for the Attention computation task of each AIcore. The data format is ND, and non-contiguous tensors are not supported. | int32 | (8 * (1 + aic_num + aiv_num) + 256,)  |

### quant_lightning_indexer

| Name | Type | Required/Optional | Description | Data Type | Dimension (Shape) |
|--------|----------|-----------|------|----------|-------------|
| sparse_indices | Tensor | Required | Indices output in the formula. Empty tensors are not supported. Invalid parts are filled with -1. The data format is ND. | int32 | (b,q_s,k_n,topk) when layout_q is BSND; (q_t,k_n,topk) when layout_q is TND |
| sparse_values | Tensor | conditional output | Values output corresponding to Indices in the formula. When return_value is 1, corresponding values are output; when return_value is 0, an empty tensor with shape [0] is output. Invalid parts are filled with -inf. The data format is ND. | bfloat16 | (b,q_s,k_n,topk) when layout_q is BSND; (q_t,k_n,topk) when layout_q is TND; (0,) when return_value is 0 |

## Constraints

- This API supports use in inference scenarios.
- This API can be called in single-operator mode and TorchAir (aclgraph) graph mode.
- The quant_lightning_indexer_metadata API must be used together with the quant_lightning_indexer operator.
- b (batch) indicates the batch size of input samples.
- The values of cu_seqlens_q and cu_seqlens_k must be the cumulative number of valid tokens in the current batch and all preceding batches. The first element must be 0, and each subsequent element must be greater than or equal to the previous element.
- The values of seqused_q and seqused_k indicate the number of valid tokens in each batch.
- cmp_residual_k must satisfy 0 <= cmp_residual_k\[i\] < cmp_ratio.
- For details about the mask mode indicated by mask_mode, see [sparse_mode Parameter Description](../../../docs/en/context/sparse_mode_introduction.md).
- pa_kv_cache supports non-contiguity on axis 0; pa_block_size supports values in [16, 1024] and must be a multiple of 16.
- The data types of parameters q and k must be the same.
- Sorting NaN values in the TopK sorting process of this API is undefined behavior.
- When layout_q is BSND, passing cu_seqlens_q is not supported; when layout_k is BSND or PA_BBND, passing cu_seqlens_k is not supported.
- When cmp_ratio > 1 and mask_mode = 3, cmp_residual_k must be passed; in other cases, do not pass it.
- Invalid parts of sparse_indices are filled with -1; invalid parts of sparse_values are filled with -inf.
<!-- npu="A3,910b" id7 -->
- x1 is INT8, x2 is INT8, x1Scale is FLOAT32, and x2Scale is FLOAT32.
  - The value range of topk currently only supports [1, 2048].
  - When layout_k is TND, passing seqused_k is not supported.
  - output_idx_offset is not supported.
  - Only num_heads_q = 64 and q_n = 64 are supported.
  - Only quant_mode = 2 is supported.
  - q and k only support int8; float8_e4m3fn, HIfloat8, and float4_e2m1 are not supported.
  - q_descale and k_descale support float16; float32 and float8_e8m0 are not supported.
  - w supports float16; float32 is not supported.
  - layout_k only supports PA_BBND, and seqused_k must be passed.
  - cmp_ratio only supports powers of 2: 1/2/4/8/16/32/64/128.
  - The return_value function is not supported; passing this parameter is not recommended.
<!-- end id7 -->
<!-- npu="950" id5 -->
- <term>Ascend 950PR&950DT products</term>:
  - topk currently only supports a value range of [1, 8192].
  - num_heads_q = 1~64, q_n = 1~64 are supported.
  - cmp_ratio supports [1, 128].
  - When output_idx_offset is passed, index offsets greater than or equal to 0 are supported. In addition, the following constraint must be satisfied: after the passed index offset is added, the resulting sparseIndice value must not exceed the maximum value of INT32.
  - When layout_q is TND, cu_seqlens_q must be passed. If seqused_q is also passed, ensure that the q length of each batch passed by seqused_q does not exceed the q sequence length of each batch calculated based on cu_seqlens_q. When the q sequence length seqlen1 of a batch passed by seqused_q is less than the q length seqlen2 calculated based on cu_seqlens_q, TND Padding is enabled, and the sparse_indices and sparse_values of the q output in the batch for the part corresponding to the difference between seqlen2 and seqlen1 are all set to invalid values. In some long-sequence use cases, if the invalid data to be padded is too large, the aicore execution may time out due to hardware limitations. The amount of data to be padded can be calculated as (seqlen2 - seqlen1) * topk. It is recommended to control this data amount within 400 million.
  - The parameter metadata must be passed, and its shape is (8 * (1 + aic_num + aiv_num) + 256,), where the core counts are obtained from the current device.
  - For q and k, float8_e4m3fn is supported when quant_mode is 1/3, HIfloat8 is supported when quant_mode is 4, float4_e2m1 is supported when quant_mode is 5, and int8 is supported when quant_mode is 2.
  - For q_descale and k_descale, float8_e8m0 is supported when quant_mode is 3/5, float32 is supported when quant_mode is 1/4, and float16 is supported when quant_mode is 2.
  - For w, float16 is supported when quant_mode is 2, and float32 is supported when quant_mode is 1/3/4/5.
<!-- end id5 -->

### Feature Parameter Group

|      Feature Parameter Group      |      Parameter Field Name      |
| :-------------------: | :-------------------: |
|      Common parameter group      | q, k, w, q_descale, k_descale, metadata, output_idx_offset, topk, quant_mode, layout_q, layout_k, sparse_indices, sparse_values |
|      Mask parameter group      | mask_mode |
|   SeqLens parameter group   | cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, max_seqlen_q |
|   Sparse compression parameter group    | cmp_ratio, cmp_residual_k |
| Paged Attention parameter group | block_table |

### Baseline Information

#### Common Parameter Group

- Handling of empty input parameters:
    - An empty tensor refers to a required input or output with a shape size of 0, that is, any axis is 0.
    - All use cases that trigger an empty tensor are intercepted and an error is reported.

- Validation of q, k, sparse_indices, and sparse_values

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
                <li>tensor_type supports INT8, FLOAT8_e4m3fn, HIFLOAT8, and FLOAT4_e2m1fn</li>
                <li>BSND -> (b, q_s, q_n, d)</li>
                <li>TND -> (q_t, q_n, d)</li>
            </ul>
        </td>
        <td rowspan="4">            Must exist.
        </td>
        <td rowspan="4">            <ul>
                <li>q and k must have the same data type</li>
                <li>For layout validation rules, see the layout matching relationship table</li>
            </ul>
        </td>
        <td rowspan="4">            Axis validation:
            <ul>
                <li>65536 > b > 0</li>
                <li>q_t > 0</li>
                <li>k_t > 0</li>
                <li>q_n > 0</li>
                <li>k_n = 1</li>
                <li>q_s > 0</li>
                <li>k_s > 0</li>
                <li>d = 128</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>k</td>
        <td rowspan="1">            <ul>
                <li>tensor_type supports INT8, FLOAT8_e4m3fn, HIFLOAT8, and FLOAT4_e2m1fn</li>
                <li>BSND -> (b, k_s, k_n, d)</li>
                <li>TND -> (k_t, k_n, d)</li>
                <li>PA_BBND -> (num_blocks, block_size, k_n, d)</li>
                <li>1024 >= block_size >= 16, and block_size % 16 == 0</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>sparse_indices</td>
        <td rowspan="1">            <ul>
                <li>tensor_type supports INT32</li>
                <li>If layout_q is BSND, the shape of sparse_indices is (b, q_s, k_n, topk)</li>
                <li>If layout_q is TND, the shape of sparse_indices is (q_t, k_n, topk)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>sparse_values</td>
        <td rowspan="1">            <ul>
                <li>tensor_type supports BFLOAT16</li>
                <li>If layout_q is BSND, the shape of sparse_indices is (b, q_s, k_n, topk)</li>
                <li>If layout_q is TND, the shape of sparse_indices is (q_t, k_n, topk)</li>
            </ul>
        </td>
    </tr>
</tbody>
</table>

Layout matching relationships:
<table style="undefined;table-layout: fixed; width:1625px"><colgroup>
<col style="width: 247px">
<col style="width: 132px">
<col style="width: 232px">
<col style="width: 293px">
<col style="width: 185px">
<col style="width: 119px">
<col style="width: 272px">
<col style="width: 145px">
</colgroup>
<thead>
<tr>
    <th>layout_q</th>
    <th>layout_k</th>
    <th>layout_out</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>BSND</td>
        <td>          <li>BSND</li>
          <li>PA_BBND</li>
        </td>
        <td>BSND</td>
    </tr>
    <tr>
        <td>TND</td>
        <td>          <li>TND</li>
          <li>PA_BBND</li>
        </td>
        <td>TND</td>
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
                    <li>tensor_type only supports INT32.</li>
                    <li>The shape is dynamically calculated by quant_lightning_indexer_v2_metadata.</li>
                    <li>Currently, this parameter must be passed; otherwise, an interception alarm is triggered.</li>
                </ul>
            </td>
            <td>Optional parameter</td>
            <td>None</td>
            <td>If passed in, must be consistent with the result generated by quant_lightning_indexer_v2_metadata.</td>
        </tr>
    </tbody>
</table>

mask_mode parameter description:
<ul>
    <li>mask_mode=0, full computation mode (default value)</li>
    <li>mask_mode=3, Causal mode</li>
</ul>

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
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>mask_mode</td>
            <td>                <ul>
                    <li>data_type supports INT.</li>
                    <li>Only 0 and 3 are supported; the default value is 0.</li>
                </ul>
            </td>
            <td>                Optional input. If this parameter is not passed, the default value is 0.
            </td>
        </tr>
    </tbody>
</table>

#### SeqLengths parameter group

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
            <td rowspan="2">                <ul>
                    <li>tensor_type supports INT32.</li>
                    <li>tensor_shape is (b,).</li>
                    <li>Only non-negative integers are supported.</li>
                    <li>Values in seqused_q must be less than or equal to q_s.</li>
                    <li>Values in seqused_k must be less than or equal to k_s.</li>
                </ul>
            </td>
            <td rowspan="6">Optional parameter</td>
            <td rowspan="6">None</td>
            <td rowspan="2">None</td>
        </tr>
        <tr>
            <td>seqused_k</td>
        </tr>
        <tr>
            <td>cu_seqlens_q</td>
            <td>                <ul>
                    <li>tensor_type supports INT32</li>
                    <li>tensor_shape is (b+1,)</li>
                    <li>Only non-negative integers are supported</li>
                    <li>Values must be arranged in non-decreasing order (greater than or equal to the previous value), with the first element being 0 and the last element equal to q_t</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Must be passed when layout_q is TND</li>
                    <li>Passing not supported when layout_q is not TND</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>cu_seqlens_k</td>
            <td>                <ul>
                    <li>tensor_type supports INT32</li>
                    <li>tensor_shape is (b+1,)</li>
                    <li>Only non-negative integers are supported</li>
                    <li>Values must be arranged in non-decreasing order (greater than or equal to the previous value), with the first element being 0 and the last element equal to k_t</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Must be passed when layout_k is TND</li>
                    <li>Passing not supported when layout_k is not TND</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>max_seqlen_q</td>
            <td rowspan="2">                <ul>
                    <li>data_type supports INT</li>
                    <li>The value must be greater than or equal to -1</li>
                    <li>Default value is -1</li>
                </ul>
            </td>
            <td rowspan="2">                <ul>
                    <li>None</li>
                </ul>
            </td>
        </tr>
    </tbody>
</table>

#### Paged Attention Parameter Group

Paged Attention is enabled when block_table is not empty.
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
            <td>block_table</td>
            <td>                <ul>
                    <li>tensor_type only supports INT32</li>
                    <li>tensor_shape is (b, max_num_blocks_per_seq)</li>
                    <li>The value must be a positive integer</li>
                </ul>
            </td>
            <td>Optional parameter</td>
            <td>None</td>
            <td>                <ul>
                    <li>When PagedAttention is enabled, seqused_k must be passed.</li>
                    <li>When PagedAttention is enabled, block_table must not be empty.</li>
                </ul>
            </td>
        </tr>
    </tbody>
</table>

## Deterministic Computation

Deterministic computation is supported by default.

## Example

- Single-operator mode call:

  ```python
  import torch
  import torch_npu
  from cann_ops_transformer.ops import quant_lightning_indexer, quant_lightning_indexer_metadata

  B = 2
  S1 = 64
  N1 = 64
  N2 = 1
  D = 128
  blockNum = 128
  blockSize = 128
  topk = 32

  # Construct the inputs.
  query = torch.randn(B, S1, N1, D).to(torch.float8_e4m3fn).npu()
  key = torch.randn(blockNum, blockSize, N2, D).to(torch.float8_e4m3fn).npu()
  weights = torch.randn(B, S1, N1).npu()
  query_dequant_scale = torch.randn(B, S1, N1).npu()
  key_dequant_scale = torch.randn(blockNum, blockSize, N2).npu()
  block_table = torch.arange(blockNum, dtype=torch.int32).expand(B, -1).npu()
  seqused_k = torch.full((B,), blockNum * blockSize, dtype=torch.int32).npu()

  # Generate metadata.
  metadata = quant_lightning_indexer_metadata(
      num_heads_q=N1,
      num_heads_k=N2,
      head_dim=D,
      topk=topk,
      quant_mode=1,
      seqused_k=seqused_k,
      batch_size=B,
      max_seqlen_q=S1,
      max_seqlen_k=blockNum * blockSize,
      layout_q="BSND",
      layout_k="PA_BBND",
      mask_mode=0,
      cmp_ratio=1
  )

  # Execute quant_lightning_indexer.
  sparse_indices, sparse_values = quant_lightning_indexer(
      query, key, weights, query_dequant_scale, key_dequant_scale,
      topk=topk,
      quant_mode=1,
      block_table=block_table,
      metadata=metadata,
      seqused_k=seqused_k,
      max_seqlen_q=-1,
      layout_q="BSND",
      layout_k="PA_BBND",
      mask_mode=0,
      cmp_ratio=1,
      return_value=1
  )
  print(f"sparse_indices shape: {sparse_indices.shape}")
  print(f"sparse_values shape: {sparse_values.shape}")
  ```

- aclgraph graph mode call:

  ```python
  import torch
  import torch_npu
  import torchair
  from cann_ops_transformer.ops import quant_lightning_indexer, quant_lightning_indexer_metadata

  B = 2
  S1 = 64
  S2 = 128
  N1 = 64
  N2 = 1
  D = 128
  topk = 32

  query = torch.randn(B, S1, N1, D).to(torch.float8_e4m3fn).npu()
  key = torch.randn(B, S2, N2, D).to(torch.float8_e4m3fn).npu()
  weights = torch.randn(B, S1, N1).npu()
  query_dequant_scale = torch.randn(B, S1, N1).npu()
  key_dequant_scale = torch.randn(B, S2, N2).npu()

  class QuantLightningIndexerNetwork(torch.nn.Module):
      def __init__(self):
          super(QuantLightningIndexerNetwork, self).__init__()

      def forward(self, query, key, weights, query_dequant_scale, key_dequant_scale):
          metadata = torch.ops.cann_ops_transformer.quant_lightning_indexer_metadata(
              num_heads_q=N1,
              num_heads_k=N2,
              head_dim=D,
              topk=topk,
              quant_mode=1,
              batch_size=B,
              max_seqlen_q=S1,
              max_seqlen_k=S2,
              layout_q="BSND",
              layout_k="BSND",
              mask_mode=0,
              cmp_ratio=1
          )

          return torch.ops.cann_ops_transformer.quant_lightning_indexer(
              query, key, weights, query_dequant_scale, key_dequant_scale,
              topk=topk,
              quant_mode=1,
              metadata=metadata,
              max_seqlen_q=-1,
              layout_q="BSND",
              layout_k="BSND",
              mask_mode=0,
              cmp_ratio=1,
              return_value=1
          )

  from torchair.configs.compiler_config import CompilerConfig
  config = CompilerConfig()
  config.mode = "reduce-overhead"
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  torch._dynamo.reset()
  npu_mode = torch.compile(QuantLightningIndexerNetwork(), fullgraph=True, backend=npu_backend, dynamic=False)
  sparse_indices, sparse_values = npu_mode(query, key, weights, query_dequant_scale, key_dequant_scale)
  print(f"sparse_indices shape: {sparse_indices.shape}")
  print(f"sparse_values shape: {sparse_values.shape}")
  ```
