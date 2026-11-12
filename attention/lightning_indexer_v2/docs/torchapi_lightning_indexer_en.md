# lightning\_indexer

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T11:08:41.265Z -->

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
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- **Description**:

  The `lightning_indexer_metadata` API generates a task list, which contains the indexes of the batch, head, and Q and K chunks at the start and end points of the Attention computation task on each AI Core. The list is used by the subsequent `lightning_indexer` operator.

  The `lightning_indexer` API obtains the top-k positions for each token through a series of operations. The main computation process is as follows:

  1. Multiply the input parameter `q` ($Q_{index}\in\R^{g\times d}$) corresponding to a token by the given context `k` ($K_{index}\in\R^{S_{k}\times d}$) to obtain the correlation.
  2. Filter invalid negative correlation signals through the activation function $ReLU$ to obtain the correlation score vector between the current token and all preceding tokens.
  3. Multiply it by the weight coefficient `w` ($W$), then along the g direction, select the top $Top-k$ index values to obtain the output $sparseIndices$, and output the corresponding $sparseValues$ as the input of Attention.

- **Formula**:

  $$
  Top-k \left\{  \left[ 1 \left] \mathop{{}}\nolimits_{{1 \times \text{  }g}}\text{@} \left[  \left( W\text{@} \left[ 1 \left] \mathop{{}}\nolimits_{{1\text{ } \times \text{  }S\mathop{{}}\nolimits_{{k}}}} \left) \text{ } \odot \text{ }ReLU \left( Q\mathop{{}}\nolimits_{{index}}\text{@}K\mathop{{}}\nolimits_{{T}}^{{index}} \left)  \left]  \right\} \right. \right. \right. \right. \right. \right. \right. \right. \right. \right.
  $$

## Function Prototype

Before calling the lightning_indexer API, call the prerequisite API lightning_indexer_metadata to complete the load balancing computation of lightning_indexer.

```python
cann_ops_transformer.lightning_indexer_metadata(
  num_heads_q,
  num_heads_k,
  head_dim,
  topk,
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
cann_ops_transformer.lightning_indexer(
  q,
  k,
  w,
  topk,
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

>**NOTE:**<br>
>
>- Dimension meanings of parameters q, k, and w: b (batch size) indicates the batch size of input samples, q_s indicates the sequence length of q input samples, k_s indicates the sequence length of k input samples, q_n indicates the number of heads of q, k_n indicates the number of heads of k, d (head dim) indicates the dimension of an attention head, q_t indicates the cumulative sum of the sequence lengths of q input samples, and k_t indicates the cumulative sum of the sequence lengths of k input samples. The value of d in parameter q is equal to that of d in parameter k, and currently only 128 is supported.

### lightning_indexer_metadata

| Name | Type | Required/Optional | Description | Data Type | Dimension (Shape) |
|--------|----------|-----------|------|----------|-------------|
| num_heads_q | int | Required | Number of q heads. | int64 | - |
| num_heads_k | int | Required | Number of k heads. Currently, only 1 is supported. | int64 | - |
| head_dim | int | Required | Dimension of the attention head. Currently, only 128 is supported. | int64 | - |
| topk | int | Required | Number of Key token indexes retained for each q token. Currently, values in [1, 8192] are supported. | int64 | - |
| cu_seqlens_q | Tensor | Optional | Cumulative prefix sum of valid Sequence Lengths of q in different batches. Required only when layout_q is TND. The first value is fixed to 0. Data format: ND. Non-contiguous tensors are supported. | int32 | (b+1, ) |
| cu_seqlens_k | Tensor | Optional | Cumulative prefix sum of valid Sequence Lengths of k in different batches. Required only when layout_k is TND. The first value is fixed to 0. Data format: ND. Non-contiguous tensors are supported. | int32 | (b+1, ) |
| seqused_q | Tensor | Optional | Actual Sequence Lengths of q that participate in the computation in different batches. Data format: ND. Non-contiguous tensors are supported. | int32 | (b, ) |
| seqused_k | Tensor | Optional | Actual Sequence Lengths of k that participate in the computation in different batches. Data format: ND. Non-contiguous tensors are supported. | int32 | (b, ) |
| cmp_residual_k | Tensor | Optional | Remainder of the pre-compression Sequence Length of cmp_kv divided by cmp_ratio in each batch. Used together with cmp_ratio to compute the mask and load of the cmp_kv part. Required when cmp_ratio is not 1 and mask_mode is 3. Data format: ND. Non-contiguous tensors are supported. | int32 | (b, ) |
| batch_size | int | Optional | Number of batches. The default value is 0. | int64 | - |
| max_seqlen_q | int | Optional | Maximum Sequence Length of q. -1 indicates any possible length. The default value is -1. | int64 | - |
| max_seqlen_k | int | Optional | Maximum Sequence Length of k. -1 indicates any possible length. The default value is -1. | int64 | - |
| layout_q | str | Optional | Layout format of q. Supported values: BSND and TND. The default value is BSND. | string | - |
| layout_k | str | Optional | Layout format of k. Supported values: BSND, TND, and PA_BBND. The default value is BSND. | string | - |
| mask_mode | int | Optional | Sparse mode. 0 indicates No mask, and 3 indicates the rightDownCausal mode. The default value is 0. | int64 | - |
| cmp_ratio | int | Optional | Compression ratio of k. Value range: [1, 128]. The default value is 1, indicating no compression. | int64 | - |

### lightning_indexer

| Name | Type | Required/Optional | Description | Data Type | Dimension (Shape) |
|--------|----------|-----------|------|----------|-------------|
| q | Tensor | Required | Input Q in the formula. Empty tensors are not supported. Data format: ND. | bfloat16, float16 | (b,q_s,q_n,d) when layout_q is BSND; (q_t,q_n,d) when layout_q is TND |
| k | Tensor | Required | Input K in the formula. Empty tensors are not supported. Data format: ND. Non-contiguous tensors are supported (only axis 0 supports non-contiguity in the PA_BBND use case). | bfloat16, float16 | (b,k_s,k_n,d) when layout_k is BSND; (k_t,k_n,d) when layout_k is TND; (block_num,block_size,k_n,d) when layout_k is PA_BBND |
| w | Tensor | Required | Input W in the formula. Empty tensors are not supported. Data format: ND. | float | (b,q_s,q_n) when layout_q is BSND; (q_t,q_n) when layout_q is TND |
| topk | int | Required | Number of Key token indices to retain in the topK phase. Value range: [1, 8192]. | int64 | - |
| cu_seqlens_q | Tensor | Optional | Cumulative sum of the number of valid q tokens in the current and preceding batches. Required only when layout_q is TND. The first value is fixed to 0. Data format: ND. | int32 | (b+1,) |
| cu_seqlens_k | Tensor | Optional | Cumulative sum of the number of valid k tokens in the current and preceding batches. Required only when layout_k is TND. The first value is fixed to 0. Data format: ND. | int32 | (b+1,) |
| seqused_q | Tensor | Optional | Actual used length of q in different batches. Data format: ND. | int32 | (b,) |
| seqused_k | Tensor | Optional | Actual used length of k in different batches. Data format: ND. | int32 | (b,) |
| cmp_residual_k | Tensor | Optional | Remainder of dividing the number of k tokens before compression by cmp_ratio. Required when mask_mode is 3 and cmp_ratio is not 1. Data format: ND. | int32 | (b,) |
| block_table | Tensor | Optional | Block mapping table used for KV storage in PagedAttention. Empty tensors are not supported. Data format: ND. | int32 | (b, k_s_max/block_size) |
| output_idx_offset | Tensor | Optional | Offset to add to the output indices of the topK result. The value must be greater than or equal to 0, and the topK index plus the offset must not exceed the maximum value of int32. Data format: ND. | int32 | (b,q_s,k_n) when layout_q is BSND; (q_t,k_n) when layout_q is TND |
| metadata | Tensor | Optional | Core splitting information obtained from lightning_indexer_metadata, including the number of used cores, block size, and the start point of the data processed by each core. Empty tensors are not supported. Data format: ND. | int32 | (1024,) |
| max_seqlen_q | int | Optional | Maximum sequence length of q. -1 indicates any possible length. Default value: -1. | int64 | - |
| layout_q | str | Optional | Data layout format of input q. Supports BSND and TND. Default value: BSND. | string | - |
| layout_k | str | Optional | Data layout format of input k. Supports BSND, TND, and PA_BBND. Default value: BSND. | string | - |
| mask_mode | int | Optional | Mask mode. 0 indicates the defaultMask mode, and 3 indicates the rightDownCausal mode. Default value: 0. | int64 | - |
| cmp_ratio | int | Optional | Used for sparse computation, indicating the compression ratio of k. Value range: 1-128. Default value: 1. | int64 | - |
| return_value | int | Optional | Whether to return the Values corresponding to Indices. 0 indicates no return, and 1 indicates return. Default value: 0. | int64 | - |

## Return Values

### lightning_indexer_metadata

| Name | Type | Required/Optional | Description | Data type | Dimension (Shape) |
|--------|----------|-----------|------|----------|-------------|
| metadata | Tensor | Required | Indices of batch, head, and chunks of Q and K in the Attention computing task on each AI Core. The data format is ND. Non-contiguous tensors are not supported. | int32 | (1024,) |

### lightning_indexer

| Name | Type | Required/Optional | Description | Data Type | Dimension (Shape) |
|--------|----------|-----------|------|----------|-------------|
| sparse_indices | Tensor | Required | Indices output in the formula. Empty tensors are not supported. Invalid parts are filled with -1. The data format is ND. | int32 | (b,q_s,k_n,topk) when layout_q is BSND; (q_t,k_n,topk) when layout_q is TND |
| sparse_values | Tensor | Conditional output | Values output corresponding to the Indices in the formula. Outputs the corresponding values when return_value is 1; outputs an empty tensor with a shape of [0] when return_value is 0. Invalid parts are filled with -inf. The data format is ND. | float | (b,q_s,k_n,topk) when layout_q is BSND; (q_t,k_n,topk) when layout_q is TND; (0,) when return_value is 0 |

## Constraints

- This API supports use in inference scenarios.
- This API supports calls in single-operator mode and TorchAir (aclgraph) graph mode.
- The lightning_indexer_metadata API must be used together with the lightning_indexer operator.
- b (batch) indicates the batch size of input samples.
- The values of cu_seqlens_q and cu_seqlens_k must be the cumulative number of valid tokens in the current batch and preceding batches. The first element must be 0, and each subsequent element must be greater than or equal to the previous one.
- The values of seqused_q and seqused_k indicate the number of valid tokens in each batch.
- cmp_residual_k must satisfy cmp_residual_k\[i\] < cmp_ratio.
- For details about the mask mode indicated by mask_mode, see [sparseMode Introduction](../../../docs/en/context/sparse_mode_introduction.md).
- pa_kv_cache supports non-contiguous data on axis 0; pa_block_size supports [16, 1024] and must be a multiple of 16.
- The data types of q and k must be consistent.
- The TopK sorting process of this API exhibits undefined behavior when sorting NaN values.
- When layout_q is BSND, passing cu_seqlens_q is not supported; when layout_k is BSND or PA_BBND, passing cu_seqlens_k is not supported.
- When the input cmp_ratio > 1 and mask_mode = 3, cmp_residual_k must be passed in; otherwise, do not pass it in.
- The invalid parts of sparse_indices are filled with -1; the invalid parts of sparse_values are filled with -inf.
<!-- npu="A3,910b" id7 -->
- <term>Atlas A3 products</term>, <term>Atlas A2 products</term>:
  - The value range of topk currently supports only [1, 2048], as well as 3072, 4096, 5120, 6144, 7168, and 8192.
  - The lightning_indexer API currently does not support seqused_q, output_idx_offset, or max_seqlen_q. Passing in these parameters is not recommended.
  - num_heads_q = 1~64 and q_n = 1~64 are supported.
  - layout_k supports BSND, TND, and PA_BBND; when layout_k is not PA_BBND, layout_q must be the same as layout_k.
  - seqused_k must be passed in when layout_k is PA_BBND and is optional when layout_k is BSND or TND.
  - cmp_ratio supports [1, 128].
  - The return_value function is supported. 0 indicates that the corresponding value is not returned, and 1 indicates that sparse_values of the FLOAT32 type is returned.
<!-- end id7 -->
<!-- npu="950" id8 -->
- <term>Ascend 950PR&950DT products</term>:
  - The value range of topk is currently only [1, 8192].
  - num_heads_q = 1~64 and q_n = 1~64 are supported.
  - cmp_ratio supports [1, 128].
  - When output_idx_offset is passed in, index offset values greater than or equal to 0 are supported; in addition, the following constraint must be met: after the passed index offset value is added, the obtained sparseIndice value must not exceed the maximum value of INT32.
  - When layout_q is TND, cu_seqlens_q must be passed in. If seqused_q is also passed in, ensure that the q length of each batch passed in through seqused_q does not exceed the q sequence length of each batch calculated based on cu_seqlens_q. When the q sequence length seqlen1 of a batch passed in through seqused_q is less than the q length seqlen2 calculated based on cu_seqlens_q, the TND Padding function is enabled, and the sparse_indices and sparse_values of the q output for the difference between seqlen2 and seqlen1 of the batch are all set to invalid values. In some long-sequence use cases, if too much invalid data needs to be padded, aicore execution may time out due to hardware limitations. Calculate the amount of data to be padded by (seqlen2 - seqlen1) * topk. It is recommended that this amount be less than 400 million.
  - The metadata parameter must be passed in, and its shape is (1024,).
<!-- end id8 -->

### Feature Parameter Group

|      Feature Parameter Group      |     Parameter Field Name     |
| :-------------------: | :-------------------: |
|      Common parameter group      | q, k, w, metadata, output_idx_offset, topk, layout_q, layout_k, sparse_indices, sparse_values |
|      Mask parameter group      | mask_mode |
|   SeqLens parameter group   | cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, max_seqlen_q |
|   Sparse compression parameter group    | cmp_ratio, cmp_residual_k |
| Paged Attention parameter group | block_table |

### Baseline Information

#### Common Parameter Group

- Handling of empty input parameters:
    - An empty tensor refers to a required input or output whose shape size is 0, that is, any axis is 0.
    - Use cases that trigger an empty tensor will all be intercepted with errors.

- q, k, sparse_indices, and sparse_values validation

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
                <li>tensor_type supports BFLOAT16 and FLOAT16.</li>
                <li>BSND -> (b, q_s, q_n, d)</li>
                <li>TND -> (q_t, q_n, d)</li>
            </ul>
        </td>
        <td rowspan="4">            Must exist.
        </td>
        <td rowspan="4">            <ul>
                <li>The data types of q and k must be the same.</li>
                <li>For layout validation rules, see the layout matching relationship table.</li>
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
                <li>tensor_type supports BFLOAT16 and FLOAT16.</li>
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
                <li>tensor_type supports INT32.</li>
                <li>If layout_q is BSND, the shape of sparse_indices is (b, q_s, k_n, topk).</li>
                <li>If layout_q is TND, the shape of sparse_indices is (q_t, k_n, topk).</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>sparse_values</td>
        <td rowspan="1">            <ul>
                <li>tensor_type supports FLOAT32.</li>
                <li>If layout_q is BSND, the shape of sparse_indices is (b, q_s, k_n, topk).</li>
                <li>If layout_q is TND, the shape of sparse_indices is (q_t, k_n, topk).</li>
            </ul>
        </td>
    </tr>
</tbody>
</table>

layout matching relationship table:
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
            <th>Cross-Feature Validation</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>metadata</td>
            <td>                <ul>
                    <li>tensor_type supports only INT32.</li>
                    <li>shape is dynamically computed by lightning_indexer_v2_metadata.</li>
                    <li>Currently, the parameter cannot be omitted; otherwise, an interception alarm is reported.</li>
                </ul>
            </td>
            <td>Optional parameter</td>
            <td>None</td>
            <td>Must be consistent with the result generated by lightning_indexer_v2_metadata when passed in.</td>
        </tr>
    </tbody>
</table>

mask_mode parameter description
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
                    <li>The supported input value range is only 0 and 3, and the default value is 0.</li>
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
                    <li>Values must be non-decreasing (greater than or equal to the previous value), with the first element being 0 and the last element equal to q_t</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Required when layout_q is TND</li>
                    <li>Not supported when layout_q is not TND</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>cu_seqlens_k</td>
            <td>                <ul>
                    <li>tensor_type supports INT32</li>
                    <li>tensor_shape is (b+1,)</li>
                    <li>Only non-negative integers are supported</li>
                    <li>Values must be non-decreasing (greater than or equal to the previous value), with the first element being 0 and the last element equal to k_t</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Required when layout_k is TND</li>
                    <li>Not supported when layout_k is not TND</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>max_seqlen_q</td>
            <td rowspan="2">                <ul>
                    <li>data_type supports INT</li>
                    <li>Not yet effective; only -1 is supported</li>
                    <li>Default value: -1</li>
                </ul>
            </td>
            <td rowspan="2">                <ul>
                    <li>Currently not effective; only -1 is supported.</li>
                </ul>
            </td>
        </tr>
    </tbody>
</table>

#### Paged Attention parameter group

When block_table is not empty, Paged Attention is enabled
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
            <th>Single Parameter Validation</th>
            <th>Existence Validation</th>
            <th>Consistency Validation</th>
            <th>Feature Cross-Validation</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>block_table</td>
            <td>                <ul>
                    <li>tensor_type supports only INT32</li>
                    <li>tensor_shape is (b, max_num_blocks_per_seq)</li>
                    <li>The value can only be a positive integer</li>
                </ul>
            </td>
            <td>Optional parameter</td>
            <td>None</td>
            <td>                <ul>
                    <li>When PagedAttention is enabled, seqused_k must be passed in.</li>
                    <li>When PagedAttention is enabled, block_table must not be empty.</li>
                </ul>
            </td>
        </tr>
    </tbody>
</table>

## Deterministic Computation

Deterministic computation is supported by default.

## Example

- Call in single-operator mode:

  ```python
  import torch
  import torch_npu
  import numpy as np
  from cann_ops_transformer.ops import lightning_indexer, lightning_indexer_metadata

  B = 2
  S1 = 64
  S2 = 130
  N1 = 16
  N2 = 1
  D = 128
  topk = 32
  cmp_ratio = 4

  # Calculate the compressed S2 length and remainder.
  S2_compressed = S2 // cmp_ratio
  res_k = S2 % cmp_ratio

  # Construct the inputs.
  q = torch.randn(B, S1, N1, D, dtype=torch.float16).npu()
  k = torch.randn(B, S2_compressed, N2, D, dtype=torch.float16).npu()
  w = torch.randn(B, S1, N1, dtype=torch.float32).npu()
  cmp_residual_k = torch.tensor([res_k] * B, dtype=torch.int32).npu()

  # Generate metadata.
  metadata = lightning_indexer_metadata(
      num_heads_q=N1,
      num_heads_k=N2,
      head_dim=D,
      topk=topk,
      cmp_residual_k=cmp_residual_k,
      batch_size=B,
      max_seqlen_q=S1,
      max_seqlen_k=S2_compressed,
      layout_q="BSND",
      layout_k="BSND",
      mask_mode=3,
      cmp_ratio=cmp_ratio
  )

  # Execute lightning_indexer.
  sparse_indices, sparse_values = lightning_indexer(
      q, k, w, topk,
      cmp_residual_k=cmp_residual_k,
      metadata=metadata,
      max_seqlen_q=-1,
      layout_q="BSND",
      layout_k="BSND",
      mask_mode=3,
      cmp_ratio=cmp_ratio,
      return_value=0
  )
  print(f"sparse_indices shape: {sparse_indices.shape}")
  ```

- Call in TorchAir (aclgraph) graph mode:

  ```python
  import torch
  import torch_npu
  import numpy as np
  import torchair
  from cann_ops_transformer.ops import lightning_indexer, lightning_indexer_metadata

  B = 2
  S1 = 64
  S2 = 130
  N1 = 16
  N2 = 1
  D = 128
  topk = 32
  cmp_ratio = 4

  S2_compressed = S2 // cmp_ratio
  res_k = S2 % cmp_ratio

  q = torch.randn(B, S1, N1, D, dtype=torch.float16).npu()
  k = torch.randn(B, S2_compressed, N2, D, dtype=torch.float16).npu()
  w = torch.randn(B, S1, N1, dtype=torch.float32).npu()
  cmp_residual_k = torch.tensor([res_k] * B, dtype=torch.int32).npu()

  class LightningIndexerNetwork(torch.nn.Module):
      def __init__(self):
          super(LightningIndexerNetwork, self).__init__()

      def forward(self, q, k, w, cmp_residual_k):
          metadata = torch.ops.cann_ops_transformer.lightning_indexer_metadata(
              num_heads_q=N1,
              num_heads_k=N2,
              head_dim=D,
              topk=topk,
              cmp_residual_k=cmp_residual_k,
              batch_size=B,
              max_seqlen_q=S1,
              max_seqlen_k=S2_compressed,
              layout_q="BSND",
              layout_k="BSND",
              mask_mode=3,
              cmp_ratio=cmp_ratio
          )

          return torch.ops.cann_ops_transformer.lightning_indexer(
              q, k, w, topk,
              cmp_residual_k=cmp_residual_k,
              metadata=metadata,
              max_seqlen_q=-1,
              layout_q="BSND",
              layout_k="BSND",
              mask_mode=3,
              cmp_ratio=cmp_ratio,
              return_value=1
          )

  from torchair.configs.compiler_config import CompilerConfig
  config = CompilerConfig()
  config.mode = "reduce-overhead"
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  torch._dynamo.reset()
  npu_mode = torch.compile(LightningIndexerNetwork(), fullgraph=True, backend=npu_backend, dynamic=False)
  sparse_indices, sparse_values = npu_mode(q, k, w, cmp_residual_k)
  print(f"sparse_indices shape: {sparse_indices.shape}")
  ```
