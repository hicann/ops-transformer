# quant_flash_attn

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T12:42:00.425Z -->

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

- **Description**:

  `quant_flash_attn` is an extension API of `cann_ops_transformer` based on `torch_npu`, used to call the `QuantFlashAttn` operator to perform full-quantization attention computation in quantization use cases, unifying training and inference. Currently, three quantization use cases are supported:
  - HIF8 use case: Q/K/V all use HIFLOAT8 per-tensor quantization.
  - MxFP8 use case: Q/K/V all use MXFP8 quantization.
  - FP8 use case: Q/K use FP8_E4M3 per-token-head quantization, and V uses FP8_E4M3 per-head quantization.

  `quant_flash_attn_metadata` is the metadata generation API of `quant_flash_attn`, used to generate metadata before the main operator executes. The metadata records the AICore/AIVCore task partitioning result. The main operator can optionally pass in the metadata to optimize scheduling. The typical call flow is as follows:

  1. Prepare inputs such as `q`, `k`, and `v`.
  2. Call `quant_flash_attn_metadata` to generate `metadata`.
  3. Call `quant_flash_attn` and pass the `metadata` obtained in the previous step to the main operator.

- **Formula**:

  self-attention builds an attention model using the relationships within the input samples. The principle is: assume an input sample sequence $x$ of length $n$, where each element of $x$ is a $d$-dimensional vector. Each $d$-dimensional vector can be regarded as a token embedding. The sequence is transformed through 3 weight matrices to obtain 3 matrices of dimension $n \times d$.

  The computation formula of self-attention is generally defined as follows, where $Q$, $K$, and $V$ are important attribute elements of the input samples, that is, matrices obtained by transforming the input samples in space, and they can be unified into one feature space. $Q$, $K$, and $V$ are input in low-precision format and carry the corresponding dequantization scale. "Attention" in the formula and the operator name is short for "self-attention".

  $$
  Attention(Q,K,V)=Score(Q,\ K) V
  $$

  In this operator, the Score function adopts the Softmax function. The self-attention computation formula is:

  $$
  Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
  $$

  The product of $Q$ and $K^T$ represents the attention of the input $x$. To prevent this value from becoming excessively large, it is usually divided by $\sqrt{d}$ for scaling, and each row is normalized by softmax. The result is then multiplied by $V$ to obtain an $n \times d$ matrix.

  After **return_softmax_lse** is enabled, the return value softmax_lse is calculated as follows:

  $$
  S = \frac{QK^T}{\sqrt{d}}
  $$

  $$
  softmax\_max = max(S)
  $$

  $$
  softmax\_lse = log{\sum e^{S-softmax\_max}} + softmax\_max
  $$

> [!NOTE]
>
> The Q, K, V data layout can be interpreted in multiple dimensions. B (Batch) indicates the batch size batch_size of input samples, S (Seq-Length) indicates the sequence length of input samples, H (Hidden-Size) indicates the hidden size, N (Head-Num) indicates the number of heads, D (Head-Dim) indicates the size of the smallest unit headdim in the hidden layer, and D=H/N. Q_T indicates the cumulative sum of the sequence lengths of all query Batch input samples, and KV_T indicates the cumulative sum of the sequence lengths of all K and V Batch input samples. Q_S indicates the sequence length of the input q tensor, Q_N indicates the number of heads of the input q tensor, KV_S indicates the sequence length of the input k/v tensor, and KV_N indicates the number of heads of the input k/v tensor.

## Function Prototype

Before calling the quant_flash_attn API, call the prerequisite API quant_flash_attn_metadata to complete the load balancing computation for quant_flash_attn.

```python
cann_ops_transformer.quant_flash_attn_metadata(
    num_heads_q,
    num_heads_kv,
    head_dim,
    quant_mode,
    *,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    batch_size=None,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    head_dim_v=None,
    mask_mode=0,
    win_left=-1,
    win_right=-1,
    layout_q="BSND",
    layout_q_descale="BSND",
    layout_kv="BSND",
    layout_out="BSND",
    is_grad_enabled=False
) -> Tensor
```

```python
cann_ops_transformer.quant_flash_attn(
    q,
    k,
    v,
    q_descale,
    k_descale,
    v_descale,
    quant_mode,
    *,
    block_table=None,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    p_scale=None,
    sinks=None,
    attn_mask=None,
    metadata=None,
    softmax_scale=1.0,
    mask_mode=0,
    win_left=-1,
    win_right=-1,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    layout_q="BSND",
    layout_q_descale="BSND",
    layout_kv="BSND",
    layout_out="BSND",
    return_softmax_lse=False
) -> (Tensor, Tensor)
```

## Enumerations

The Python API supports passing the `IntEnum` enumeration or the corresponding int value for `quant_mode` and `mask_mode`. The enumerations are defined in `cann_ops_transformer.ops.quant_flash_attn`:

### quant_mode Enum

| Enumeration Name                                                 |  Value  | Meaning |
| :------------------------------------------------------------ | :-: | :------ |
| `A8C8_QKV_HIF8_PER_TENSOR_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32` |  0  | HIF8 use case. |
| `A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32`           |  1  | MxFP8 use case. |
| `A8C8_QK_FP8_E4M3_PER_TOKEN_HEAD_V_FP8_E4M3_PER_HEAD_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32` |  6  | FP8 use case. |

### mask_mode Enumeration

| Enumeration Name | Value | Meaning                             |
| :--------------- | :-: | :-------------------------------------- |
| `NO_MASK`        |  0  | Full computation mode (default value)   |
| `CAUSAL`         |  3  | Causal mode                             |

> [!NOTE]
>
> The current version only supports mask_mode = 0 (`NO_MASK`) and 3 (`CAUSAL`).

## Baseline Information Description

In the document constraints, common fields are defined as follows:

|   Name  |                                                                             Meaning                                                                             |
| :---: | :--------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   B   |                                                                      Batch, indicating the input sample batch size.                                                                       |
|  Q_N |                                                                 Number of heads of the input q tensor, corresponding to N in the q shape.                                                                 |
| KV_N |                                                               Number of heads of the input k/v tensor, corresponding to N in the k/v shape.                                                               |
|  Q_S |                                                                Sequence length of the input q tensor, corresponding to S in the q shape.                                                                |
| KV_S |                                                              Sequence length of the input k/v tensor, corresponding to S in the k/v shape.                                                              |
|  Q_T |                                                                  Cumulative sum of sequence lengths of all batches in the input q tensor.                                                                  |
| KV_T |                                                                 Cumulative sum of sequence lengths of all batches in the input k/v tensor.                                                                 |
|   D   |                                                        Minimum hidden layer unit size (headdim) of the input q/k/v tensors and the output attn_out.                                                       |
|   Bs  |                                                               Block size of the KV cache in the Paged Attention use case.                                                              |
|   Bn  | Number of blocks of the KV cache in the Paged Attention use case. In the shape of k/v, Bn is the total number of physically stored blocks in the KV cache, Bn = Σ ⌈KV sequence length of each batch / Bs⌉; in the shape of block_table, Bn is the maximum number of blocks for a single batch, Bn = max ⌈KV sequence length of each batch / Bs⌉. |

## Parameters

### quant_flash_attn_metadata

| Name | Type | Required/Optional | Description | Data Type | Data Format | Dimension | Non-contiguous Tensor |
| :----------------- | :------------ | :---- | :-------------------------------------------------------------------------------- | :----- | :--- | :----- | :-------- |
| num_heads_q      | int           | Required    | Number of Query heads.                                                                       | int32  | -    | -      | -         |
| num_heads_kv     | int           | Required    | Number of Key/Value heads.                                                                   | int32  | -    | -      | -         |
| head_dim          | int           | Required    | Dimension of each attention head.                                                                         | int32  | -    | -      | -         |
| quant_mode        | int/QuantMode | Required    | Quantization mode. Supports passing the enumeration or the corresponding int value. For the enumeration definition, see "quant_mode Enumeration".                                        | int32  | -    | -      | -         |
| cu_seqlens_q     | Tensor        | Optional    | Cumulative sequence length of Q, used to handle variable-length sequences. The first element must be 0.                                                       | int32  | ND   | (B+1,) | ×         |
| cu_seqlens_kv    | Tensor        | Optional    | Cumulative sequence length of KV, used to handle variable-length sequences. The first element must be 0.                                                      | int32  | ND   | (B+1,) | ×         |
| seqused_q         | Tensor        | Optional    | Specifies the actual sequence length used in each batch of q, to truncate redundancy.                                                          | int32  | ND   | (B,)   | ×         |
| seqused_kv        | Tensor        | Optional    | Specifies the actual sequence length used in each batch of kv, to truncate redundancy.                                                         | int32  | ND   | (B,)   | ×         |
| batch_size        | int           | Optional    | Batch size. If not passed, it is derived from cu_seqlens_q or seqused_q. The default value is None.                               | int32  | -    | -      | -         |
| max_seqlen_q     | int           | Optional    | Specifies the upper limit of the query q sequence length.                                                                      | int32  | -    | -      | -         |
| max_seqlen_kv    | int           | Optional    | Specifies the upper limit of the sequence lengths of key k and value v.                                                                    | int32  | -    | -      | -         |
| head_dim_v       | int           | Optional    | Dimension of each attention head of v. The default value is None (equivalent to head_dim). Currently, only None, a value equal to head_dim, or 128 when head_dim is 192 can be passed. Other values will be intercepted.                     | int32  | -    | -      | -         |
| mask_mode         | int/MaskMode  | Optional    | Mask mode. Supports passing the enumeration or the corresponding int value. For the enumeration definition, see "mask_mode Enumeration". The current version only supports 0/3                              | int32  | -    | -      | -         |
| win_left          | int           | Optional    | Left boundary of the window.                                                                         | int32  | -    | -      | -         |
| win_right         | int           | Optional    | Right boundary of the window.                                                                         | int32  | -    | -      | -         |
| layout_q          | string        | Optional    | Defines the layout format of the input q tensor.                                                                      | string | -    | -      | -         |
| layout_q_descale | string        | Optional    | Defines the layout format of the input q_descale tensor.                                                             | string | -    | -      | -         |
| layout_kv         | string        | Optional    | Defines the layout format of the input k and v tensors.                                                                    | string | -    | -      | -         |
| layout_out        | string        | Optional    | Defines the layout format of the output tensor.                                                                       | string | -    | -      | -         |
| is_grad_enabled  | bool          | Optional    | Whether to enable metadata generation for the backward gradient use case. The default value is False. When set to True, the metadata is used by the companion backward operator quant_flash_attn_grad. | BOOL   | -    | -      | -         |

### quant_flash_attn

| Name             | Type | Required/Optional | Description | Data Type                | Data Format | Dimensions                                                                                                         | Non-contiguous Tensor |
| :------------------- | :------------ | :---- | :--------------------------------------------------- | :---------------------- | :--- | :----------------------------------------------------------------------------------------------------------------- | :-------- |
| q                    | Tensor        | Required | Q in the formula. | float8_e4m3fn/hifloat8 | ND   | (Q_T, Q_N, D)                                                                                                    | ×         |
| k                    | Tensor        | Required | K in the formula. | float8_e4m3fn/hifloat8 | ND   | <ul><li>(KV_T, KV_N, D)</li><li>(Bn, KV_N, Bs, D)</li><li>(Bn, KV_N, D/32, Bs, 32)</li></ul>                   | √         |
| v                    | Tensor        | Required | V in the formula. | float8_e4m3fn/hifloat8 | ND   | <ul><li>(KV_T, KV_N, D)</li><li>(Bn, KV_N, Bs, D)</li><li>(Bn, KV_N, D/32, Bs, 32)</li></ul>                   | √         |
| q_descale           | Tensor        | Required | Dequantization scale of q. | float8_e8m0/float32    | ND   | <ul><li>(Q_T, Q_N, ⌈D/64⌉, 2)</li><li>(KV_N, Q_T, G, ⌈D/64⌉, 2)</li><li>(Q_N, Q_T)</li></ul>                     | ×         |
| k_descale           | Tensor        | Required | Dequantization scale of k. | float8_e8m0/float32    | ND   | <ul><li>(KV_T, KV_N, ⌈D/64⌉, 2)</li><li>(Bn, KV_N, Bs, ⌈D/64⌉, 2)</li><li>(Bn, KV_N, Bs/16, ⌈D/64⌉, 16, 2)</li><li>(Bn, KV_N, Bs)</li></ul> | √         |
| v_descale           | Tensor        | Required | Dequantization scale of v. | float8_e8m0/float32    | ND   | <ul><li>(KV_T/64, KV_N, D, 2)</li><li>(Bn, KV_N, Bs/64, D, 2)</li><li>(Bn, KV_N, D/16, Bs/64, 16, 2)</li><li>(KV_N,)</li></ul> | √         |
| quant_mode          | int/QuantMode | Required | Quantization mode. Supports passing the enumeration or the corresponding int value. See "quant_mode Enumeration" for the enumeration definition. | int32                   | -    | -                                                                                                                  | -         |
| block_table         | Tensor        | Optional | Block index mapping in block-wise attention computation. | int32                   | ND   | (B, Bn)                                                                                                            | ×         |
| cu_seqlens_q       | Tensor        | Optional | Cumulative sequence length of Q, used to handle variable-length sequences. The first element must be 0. | int32                   | ND   | (B+1,)                                                                                                             | ×         |
| cu_seqlens_kv      | Tensor        | Optional | Cumulative sequence length of KV, used to handle variable-length sequences. The first element must be 0. | int32                   | ND   | (B+1,)                                                                                                             | ×         |
| seqused_q           | Tensor        | Optional | Specifies the actual sequence length used in each batch to truncate redundant computation. | int32                   | ND   | (B,)                                                                                                               | ×         |
| seqused_kv          | Tensor        | Optional | Specifies the actual sequence length used in each batch to truncate redundant computation. | int32                   | ND   | (B,)                                                                                                               | ×         |
| p_scale             | Tensor        | Optional | Quantization parameter of P. | float32                 | ND   | (1,)                                                                                                               | ×         |
| sinks                | Tensor        | Optional | Input tensor in the sink use case. Not supported in the current version. Pass None. | float32                 | ND   | (Q_N,)                                                                                                            | ×         |
| attn_mask           | Tensor        | Optional | Mask matrix. | int8                    | ND   | (2048, 2048)                                                                                                       | ×         |
| metadata             | Tensor        | Optional | Task splitting result generated by `quant_flash_attn_metadata`. Passing it optimizes scheduling. | int32                   | ND   | (2, max_schedule_size)                                                                                             | ×         |
| softmax_scale       | float         | Optional | Scaling factor that can be explicitly set to override the default computation. | float32                 | -    | -                                                                                                                  | -         |
| mask_mode           | int/MaskMode  | Optional | Mask mode. Supports passing the enumeration or the corresponding int value. See "mask_mode Enumeration" for the enumeration definition. Only 0/3 is supported in the current version. | int32                   | -    | -                                                                                                                  | -         |
| win_left            | int           | Optional | Left boundary of the window. | int32                   | -    | -                                                                                                                  | -         |
| win_right           | int           | Optional | Right boundary of the window. | int32                   | -    | -                                                                                                                  | -         |
| max_seqlen_q       | int           | Optional | Specifies the maximum length of the query q sequence. The value must be greater than or equal to 0 when passed. -1 indicates that the parameter is not passed. | int32                   | -    | -                                                                                                                  | -         |
| max_seqlen_kv      | int           | Optional | Specifies the maximum length of the key k and value v sequences. The value must be greater than or equal to 0 when passed. -1 indicates that the parameter is not passed. | int32                   | -    | -                                                                                                                  | -         |
| layout_q            | string        | Optional | Defines the layout format of input tensor q. | string                  | -    | -                                                                                                                  | -         |
| layout_q_descale   | string        | Optional | Defines the layout format of input tensor q_descale. | string                  | -    | -                                                                                                                  | -         |
| layout_kv           | string        | Optional | Defines the layout format of input tensors k and v. | string                  | -    | -                                                                                                                  | -         |
| layout_out          | string        | Optional | Defines the layout format of the output tensor. | string                  | -    | -                                                                                                                  | -         |
| return_softmax_lse | bool          | Optional | Whether to obtain the LSE result of softmax. | BOOL                    | -    | -                                                                                                                  | -         |

> [!NOTE]
>
> k/v/k_descale/v_descale support non-contiguous tensors only in the Paged Attention use case. For the axis constraints supported in the non-contiguous case, see the feature cross validation in <a href="#paged-attention-parameter-group">Paged Attention Parameter Group</a>.<br>Note: The stride of each dimension of a non-contiguous tensor cannot be 0.

## Returns

### quant_flash_attn_metadata

| Name | Type | Required/Optional | Description                        | Data type | Data format | Dim                      | Non-contiguous Tensor |
| :------- | :----- | :---- | :------------------------ | :---- | :--- | :----------------------- | :-------- |
| metadata | Tensor | Required    | Task split data of quant_flash_attn. | int32 | ND   | (2, max_schedule_size) | ×         |

### quant_flash_attn

| Name      | Type   | Required/Optional | Description | Data Type     | Data Format | Dimension              | Non-contiguous Tensor |
| :----------- | :----- | :---- | :--- | :------- | :--- | :--- | :-------- |
| attn_out    | Tensor | Required    | Computation output of quant_flash_attn. | bfloat16 | ND   | <ul><li>(Q_T, Q_N, D)</li><li>(B, Q_S, Q_N, D) (HIF8 use case)</li><li>(B, Q_N, Q_S, D) (HIF8 use case)</li></ul> | ×         |
| softmax_lse | Tensor | Optional    | LSE result of softmax. When `return_softmax_lse` is True, a Tensor is output with a shape that depends on the layout_q; when `return_softmax_lse` is False, an empty Tensor is output. | float32  | ND   | <ul><li>(Q_N, Q_T) (TND)</li><li>(B, Q_N, Q_S) (BSND)</li><li>(B, Q_N, Q_S) (BNSD)</li></ul> | ×         |

## Constraints

- Parameters cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv, block_table, and attn_mask are tensors. Since the operator cannot obtain the specific values of tensors in the Tiling phase, the tiling side does not validate the values, and users must ensure correctness. If invalid values are passed for the preceding parameters, undefined behavior occurs (such as precision issues and program crashes caused by invalid memory access).
- The input parameters of quant_flash_attn_metadata and quant_flash_attn must be consistent during the call. Since the operator is divided into two APIs for a two-phase call, it cannot perform the check itself, and customers must ensure correctness. If the parameters passed to the APIs are inconsistent, undefined behavior occurs (such as precision issues and program crashes caused by invalid memory access).

### Feature Parameter Group

|       Feature Parameter Group       | Parameter Field Name |  Field Grouping  | Field Type |
| :----------------: | :------------------: | :--------------: | :----: |
|       Common parameter group        |           q          |       INPUT      | Tensor |
|               <br />                |           k          |       INPUT      | Tensor |
|               <br />                |           v          |       INPUT      | Tensor |
|               <br />                |       metadata       |  INPUT(OPTIONAL) | Tensor |
|               <br />                |    softmax_scale    |  ATTR(OPTIONAL)  | double |
|               <br />                |       layout_q      |  ATTR(OPTIONAL)  | string |
|               <br />                |  layout_q_descale  |  ATTR(OPTIONAL)  | string |
|               <br />                |      layout_kv      |  ATTR(OPTIONAL)  | string |
|               <br />                |      layout_out     |  ATTR(OPTIONAL)  | string |
|               <br />                |       attn_out      |      OUTPUT      | Tensor |
|  Full quantization parameter group  |      quant_mode     |       ATTR       |   int  |
|               <br />                |      q_descale      |       INPUT      | Tensor |
|               <br />                |      k_descale      |       INPUT      | Tensor |
|               <br />                |      v_descale      |       INPUT      | Tensor |
|               <br />                |       p_scale       |  INPUT(OPTIONAL) | Tensor |
|        Mask parameter group         |      mask_mode      |  ATTR(OPTIONAL)  |   int  |
|               <br />                |       win_left      |  ATTR(OPTIONAL)  |   int  |
|               <br />                |      win_right      |  ATTR(OPTIONAL)  |   int  |
|               <br />                |      attn_mask      |  INPUT(OPTIONAL) | Tensor |
|     SeqLengths parameter group      |    cu_seqlens_q    |  INPUT(OPTIONAL) | Tensor |
|               <br />                |    cu_seqlens_kv   |  INPUT(OPTIONAL) | Tensor |
|               <br />                |      seqused_q      |  INPUT(OPTIONAL) | Tensor |
|               <br />                |      seqused_kv     |  INPUT(OPTIONAL) | Tensor |
|               <br />                |    max_seqlen_q    |  ATTR(OPTIONAL)  |   int  |
|               <br />                |    max_seqlen_kv   |  ATTR(OPTIONAL)  |   int  |
|   Paged Attention parameter group   |     block_table     |  INPUT(OPTIONAL) | Tensor |
|        Sinks parameter group        |         sinks        |  INPUT(OPTIONAL) | Tensor |
|     SoftmaxLSE parameter group      | return_softmax_lse |  ATTR(OPTIONAL)  |  bool  |
|               <br />                |     softmax_lse     | OUTPUT(OPTIONAL) | Tensor |

### Parameter Group Constraints

#### Common Parameter Group

- Handling of empty input parameters:
  - An empty Tensor refers to a required input or output whose shape size is 0, i.e., any axis is 0.
  - All use cases that trigger an empty Tensor are intercepted with an error reported.

- q, k, v, and attn_out validation:

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
    <th>Single-Parameter Check</th>
    <th>Existence Check</th>
    <th>Consistency Check</th>
    <th>Feature Cross-Check</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>q</td>
        <td>            <ul>
                <li>tensor_type supports float8_e4m3fn/hifloat8.</li>
                <li>shape dim supports 3, 4.</li>
            </ul>
        </td>
        <td rowspan="4">            Must exist.
        </td>
        <td rowspan="4">            <ul>
                <li>The data types of q, k, and v must be the same.</li>
            </ul>
        </td>
        <td rowspan="4">            <ul>
                <li>q/k/v dtype must exactly match quant_mode: float8_e4m3fn for the MxFP8 use case, hifloat8 for the HIF8 use case, and float8_e4m3fn for the FP8 use case.</li>
                <li>q/attn_out shape dim must exactly match quant_mode: the MxFP8 use case only supports 3D, the HIF8 use case supports 3D/4D, and the FP8 use case only supports 3D.</li>
                <li>Axis validation:
                    <ul>
                        <li>65536 > B > 0</li>
                        <li>Q_S > 0; KV_S > 0</li>
                        <li>Q_T > 0, KV_T > 0</li>
                        <li>D only supports 64, 72, 128, 192, or 256; the HIF8 use case only supports D=128; the FP8 use case only supports D=128.</li>
                        <li>Q_N % KV_N == 0 and Q_N / KV_N > 0</li>
                        <li>Q_N ≤ 256; KV_N ≤ 256; Q_N / KV_N ≤ 64</li>
                    </ul>
                </li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>k</td>
        <td rowspan="2">            <ul>
                <li>tensor_type supports float8_e4m3fn/hifloat8.</li>
                <li>shape dim supports 3, 4, 5.</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>v</td>
    </tr>
    <tr>
        <td>attn_out</td>
        <td>            <ul>
                <li>data_type only supports bfloat16.</li>
                <li>shape dim supports 3, 4.</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>layout_q</td>
        <td>            <ul>
                <li>MxFP8 use case supports TND.</li>
                <li>HIF8 use case supports TND/BSND/BNSD.</li>
                <li>FP8 use case supports NTD.</li>
            </ul>
        </td>
        <td rowspan="4">Currently, this parameter must be passed. An interception alarm is triggered if it is not passed.</td>
        <td rowspan="4">None</td>
        <td rowspan="4">None</td>
    </tr>
    <tr>
        <td>layout_q_descale</td>
        <td>            <ul>
                <li>MxFP8 use case supports TND/N2TGD.</li>
                <li>HIF8 use case supports BSND.</li>
                <li>FP8 use case supports NT.</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>layout_kv</td>
        <td>            <ul>
                <li>MxFP8 use case supports TND/PA_BBND/PA_BNBD/PA_NZ (PA_NZ is not supported when D=72).</li>
                <li>HIF8 use case supports TND/BSND/BNSD.</li>
                <li>FP8 use case supports PA_BNBD.</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>layout_out</td>
        <td>            <ul>
                <li>MxFP8 use case supports TND.</li>
                <li>HIF8 use case supports TND/BSND/BNSD.</li>
                <li>FP8 use case supports TND.</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>metadata</td>
        <td>            <ul>
                <li>tensor_type supports only int32.</li>
                <li>shape is dynamically calculated by quant_flash_attn_metadata.</li>
                <li>Omitting this parameter is not supported currently. If it is not passed, an interception alarm is reported.</li>
            </ul>
        </td>
        <td>Optional parameter</td>
        <td>None</td>
        <td>Must be consistent with the result generated by quant_flash_attn_metadata when passed.</td>
    </tr>
</tbody>
</table>

#### Full Quantization Parameter Group

Explanation of the quant_mode parameter:

<ul>
    <li>quant_mode=0, A8C8_QKV_HIF8_PER_TENSOR_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32 (HIF8 use case)</li>
    <li>quant_mode=1, A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32 (MxFP8 use case)</li>
    <li>quant_mode=6, A8C8_QK_FP8_E4M3_PER_TOKEN_HEAD_V_FP8_E4M3_PER_HEAD_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32 (FP8 use case)</li>
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
            <th>Single-Parameter Check</th>
            <th>Existence Check</th>
            <th>Consistency Check</th>
            <th>Feature Cross-Check</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>quant_mode</td>
            <td>                <ul>
                    <li>The data_type supports int32.</li>
                    <li>The supported input range is 0, 1, and 6.</li>
                </ul>
            </td>
            <td>Required attribute</td>
            <td>None</td>
            <td rowspan="5">                <ul>
                    <li>For the layout validation rules, see the <a href="#layout-matching-relationship-table">layout matching relationship table</a>.</li>
                    <li>For the q, k, v, and attn_out shape validation rules, see the <a href="#qkv_attn_out_shape-matching-relationship-table">q/k/v/attn_out shape matching relationship table</a>.</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>q_descale</td>
            <td>                <ul>
                    <li>tensor_type supports float8_e8m0 and float32.</li>
                    <li>shape dim: 4 or 5 for the MxFP8 use case; 2 for the FP8 use case; 1 for the HIF8 use case.</li>
                </ul>
            </td>
            <td rowspan="3">Must exist</td>
            <td rowspan="3">                <ul>
                    <li>For the descale shape validation rules, see the <a href="#descale_shape-matching-relationship-table">descale_shape matching relationship table</a>.</li>
                    <li>For the descale dtype validation rules, see the <a href="#descale_dtype-matching-relationship-table">descale_dtype matching relationship table</a>.</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>k_descale</td>
            <td>                <ul>
                    <li>tensor_type supports float8_e8m0 and float32.</li>
                    <li>shape dim: 4, 5, or 6 for the MxFP8 use case; 3 for the FP8 use case; 1 for the HIF8 use case.</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>v_descale</td>
            <td>                <ul>
                    <li>tensor_type supports float8_e8m0 and float32.</li>
                    <li>shape dim: the MxFP8 use case supports 4, 5, and 6; the FP8 use case supports 1; the HIF8 use case supports 1.</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>p_scale</td>
            <td>                <ul>
                    <li>tensor_type supports only float32.</li>
                    <li>shape supports only (1,).</li>
                </ul>
            </td>
            <td>optional parameter</td>
            <td>None</td>
        </tr>
    </tbody>
</table>

layout matching relationship table:<a name="layout-matching-relationship-table"></a>

<table style="undefined;table-layout: fixed; width:1625px"><colgroup>
<col style="width: 110px">
<col style="width: 130px">
<col style="width: 130px">
<col style="width: 180px">
<col style="width: 110px">
<col style="width: 160px">
</colgroup>
<thead>
<tr>
    <th>quant_mode</th>
    <th>layout_q</th>
    <th>layout_q_descale</th>
    <th>layout_kv</th>
    <th>layout_out</th>
    <th>layout_softmax_lse</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>quant_mode=0 (HIF8)</td>
        <td>TND/BSND/BNSD</td>
        <td>BSND</td>
        <td>TND/BSND/BNSD</td>
        <td>TND/BSND/BNSD</td>
        <td>
            <ul>
                <li>(Q_N, Q_T) (layout_q=TND)</li>
                <li>(B, Q_N, Q_S) (layout_q=BSND/BNSD)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>quant_mode=1 (MxFP8)</td>
        <td>TND</td>
        <td>            <ul>
                <li>TND (prefill)</li>
                <li>N2TGD (decode)</li>
            </ul>
        </td>
        <td>            <ul>
                <li>TND</li>
                <li>PA_BBND</li>
                <li>PA_BNBD</li>
                <li>PA_NZ (not supported when D=72)</li>
            </ul>
        </td>
        <td>TND</td>
        <td>(Q_N, Q_T)</td>
    </tr>
    <tr>
        <td>quant_mode=6 (FP8)</td>
        <td>NTD</td>
        <td>NT</td>
        <td>PA_BNBD</td>
        <td>TND</td>
        <td>(Q_N, Q_T)</td>
    </tr>
</tbody>
</table>

> [!NOTE]
>
> In the HIF8 use case (`quant_mode=0`), layout_q, layout_kv, and layout_out must be the same.

q/k/v descale dtype matching relationship table:<a name="descale_dtype-matching-relationship-table"></a>

<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 120px">
        <col style="width: 147px">
        <col style="width: 293px">
    </colgroup>
    <thead>
        <tr>
            <th>quant_mode</th>
            <th>Parameter</th>
            <th>dtype</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="3">0</td>
            <td>q_descale</td>
            <td>float32</td>
        </tr>
        <tr>
            <td>k_descale</td>
            <td>float32</td>
        </tr>
        <tr>
            <td>v_descale</td>
            <td>float32</td>
        </tr>
        <tr>
            <td rowspan="3">1</td>
            <td>q_descale</td>
            <td>float8_e8m0</td>
        </tr>
        <tr>
            <td>k_descale</td>
            <td>float8_e8m0</td>
        </tr>
        <tr>
            <td>v_descale</td>
            <td>float8_e8m0</td>
        </tr>
        <tr>
            <td rowspan="3">6</td>
            <td>q_descale</td>
            <td>float32</td>
        </tr>
        <tr>
            <td>k_descale</td>
            <td>float32</td>
        </tr>
        <tr>
            <td>v_descale</td>
            <td>float32</td>
        </tr>
    </tbody>
</table>

q_descale shape matching relationship table: <a name="descale_shape-matching-relationship-table"></a>

<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 120px">
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 293px">
    </colgroup>
    <thead>
        <tr>
            <th>quant_mode</th>
            <th>Parameter</th>
            <th>layout_q_descale</th>
            <th>shape</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>0</td>
            <td>q_descale</td>
            <td>BSND</td>
            <td>(1,)<br>per-tensor quantization, 1D</td>
        </tr>
        <tr>
            <td rowspan="2">1</td>
            <td rowspan="2">q_descale</td>
            <td>TND</td>
            <td>(Q_T, Q_N, ⌈D/64⌉, 2)<br>Used in the Prefill use case. It is recommended to pass in when G*Q_S > 80.</td>
        </tr>
        <tr>
            <td>N2TGD</td>
            <td>(KV_N, Q_T, G, ⌈D/64⌉, 2)<br>Used in the Decode use case. It is recommended to pass in when G*Q_S <= 80.</td>
        </tr>
        <tr>
            <td>6</td>
            <td>q_descale</td>
            <td>NT</td>
            <td>(Q_N, Q_T)<br>per-token-head quantization, layout_q is NTD.</td>
        </tr>
    </tbody>
</table>

k_descale/v_descale shape matching relationship table:

<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 120px">
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 293px">
    </colgroup>
    <thead>
        <tr>
            <th>quant_mode</th>
            <th>Parameter</th>
            <th>layout_kv</th>
            <th>shape</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="2">0</td>
            <td>k_descale</td>
            <td>-</td>
            <td>(1,)<br>per-tensor quantization, 1D</td>
        </tr>
        <tr>
            <td>v_descale</td>
            <td>-</td>
            <td>(1,)<br>per-tensor quantization, 1D</td>
        </tr>
        <tr>
            <td rowspan="8">1</td>
            <td rowspan="4">k_descale</td>
            <td>TND</td>
            <td>(KV_T, KV_N, ⌈D/64⌉, 2)</td>
        </tr>
        <tr>
            <td>PA_BBND</td>
            <td>(Bn, Bs, KV_N, ⌈D/64⌉, 2)</td>
        </tr>
        <tr>
            <td>PA_BNBD</td>
            <td>(Bn, KV_N, Bs, ⌈D/64⌉, 2)</td>
        </tr>
        <tr>
            <td>PA_NZ</td>
            <td>(Bn, KV_N, Bs/16, ⌈D/64⌉, 16, 2)</td>
        </tr>
        <tr>
            <td rowspan="4">v_descale</td>
            <td>TND</td>
            <td>(KV_T/64, KV_N, D, 2)</td>
        </tr>
        <tr>
            <td>PA_BBND</td>
            <td>(Bn, Bs/64, KV_N, D, 2)</td>
        </tr>
        <tr>
            <td>PA_BNBD</td>
            <td>(Bn, KV_N, Bs/64, D, 2)</td>
        </tr>
        <tr>
            <td>PA_NZ</td>
            <td>(Bn, KV_N, D/16, Bs/64, 16, 2)</td>
        </tr>
        <tr>
            <td rowspan="2">6</td>
            <td>k_descale</td>
            <td>PA_BNBD</td>
            <td>(Bn, KV_N, Bs)<br>per-token-head quantization, forced to PA_BNBD.</td>
        </tr>
        <tr>
            <td>v_descale</td>
            <td>-</td>
            <td>(KV_N,)<br>per-head quantization, 1D</td>
        </tr>
    </tbody>
</table>

> [!NOTE]
>
> In the MxFP8 use case, for v_descale in TND (non-Paged Attention) use cases, the first dimension KV_T/64 in the shape is not a simple integer division but the cumulative sum of the actual KV sequence lengths of all batches, each rounded up to a multiple of 64, that is, Σ ceil(cu_seqlens_kv[b+1] - cu_seqlens_kv[b], 64), where cu_seqlens_kv is the cumulative sequence length. Therefore, the KV sequence lengths of batches do not need to be strictly aligned to 64.

q/k/v/attn_out shape matching relationship table:<a name="qkv_attn_out_shape-matching-relationship-table"></a>

<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 120px">
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 293px">
    </colgroup>
    <thead>
        <tr>
            <th>quant_mode</th>
            <th>Parameter</th>
            <th>layout_kv</th>
            <th>shape</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="9">0</td>
            <td rowspan="3">q</td>
            <td>TND</td>
            <td>(Q_T, Q_N, D)</td>
        </tr>
        <tr>
            <td>BSND</td>
            <td>(B, Q_S, Q_N, D)</td>
        </tr>
        <tr>
            <td>BNSD</td>
            <td>(B, Q_N, Q_S, D)</td>
        </tr>
        <tr>
            <td rowspan="3">k/v</td>
            <td>TND</td>
            <td>(KV_T, KV_N, D)</td>
        </tr>
        <tr>
            <td>BSND</td>
            <td>(B, KV_S, KV_N, D)</td>
        </tr>
        <tr>
            <td>BNSD</td>
            <td>(B, KV_N, KV_S, D)</td>
        </tr>
        <tr>
            <td rowspan="3">attn_out</td>
            <td>TND</td>
            <td>(Q_T, Q_N, D)</td>
        </tr>
        <tr>
            <td>BSND</td>
            <td>(B, Q_S, Q_N, D)</td>
        </tr>
        <tr>
            <td>BNSD</td>
            <td>(B, Q_N, Q_S, D)</td>
        </tr>
        <tr>
            <td rowspan="6">1</td>
            <td>q</td>
            <td>TND</td>
            <td>(Q_T, Q_N, D)</td>
        </tr>
        <tr>
            <td rowspan="4">k/v</td>
            <td>TND</td>
            <td>(KV_T, KV_N, D)</td>
        </tr>
        <tr>
            <td>PA_BBND</td>
            <td>(Bn, Bs, KV_N, D)</td>
        </tr>
        <tr>
            <td>PA_BNBD</td>
            <td>(Bn, KV_N, Bs, D)</td>
        </tr>
        <tr>
            <td>PA_NZ</td>
            <td>(Bn, KV_N, D/32, Bs, 32)</td>
        </tr>
        <tr>
            <td>attn_out</td>
            <td>TND</td>
            <td>(Q_T, Q_N, D)</td>
        </tr>
        <tr>
            <td rowspan="3">6</td>
            <td>q</td>
            <td>NTD</td>
            <td>(Q_N, Q_T, D)</td>
        </tr>
        <tr>
            <td>k/v</td>
            <td>PA_BNBD</td>
            <td>(Bn, KV_N, Bs, D)</td>
        </tr>
        <tr>
            <td>attn_out</td>
            <td>TND</td>
            <td>(Q_T, Q_N, D)</td>
        </tr>
    </tbody>
</table>

#### Mask parameter group

mask_mode parameter description:

<ul>
    <li>mask_mode=0, NO_MASK, full computation mode (default value)</li>
    <li>mask_mode=3, CAUSAL, Causal mode</li>
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
            <th>Single-Parameter Check</th>
            <th>Existence Check</th>
            <th>Consistency Check</th>
            <th>Feature Cross Check</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>mask_mode</td>
            <td>                <ul>
                    <li>data_type supports int32.</li>
                    <li>The input supports 0/3.</li>
                </ul>
            </td>
            <td>                Optional attribute, default value is 0.
            </td>
            <td rowspan="3">                <ul>
                    <li>Does not support passing attn_mask when mask_mode is 0.</li>
                    <li>The attn_mask matrix must be passed when mask_mode is 3.</li>
                    <li>win_left and win_right must be -1.</li>
                </ul>
            </td>
            <td rowspan="3">None</td>
        </tr>
        <tr>
            <td>attn_mask</td>
            <td>                <ul>
                    <li>tensor_type only supports int8</li>
                    <li>tensor_shape is (2048, 2048)</li>
                </ul>
            </td>
            <td>                Optional input.
            </td>
        </tr>
        <tr>
            <td>win_left<br>win_right</td>
            <td>                <ul>
                    <li>data_type supports int32</li>
                    <li>The value must be ≥ -1</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Optional attribute, default value is -1, indicating infinity (an extremely large value). The default value must be kept in the current version.</li>
                </ul>
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
            <th>Single-Parameter Check</th>
            <th>Existence Check</th>
            <th>Consistency Check</th>
            <th>Feature Cross-Check</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>seqused_q</td>
            <td rowspan="2">                <ul>
                    <li>tensor_type supports int32.</li>
                    <li>tensor_shape is (B,).</li>
                    <li>Value only supports non-negative integers.</li>
                    <li>Values in seqused_q must be less than or equal to Q_S.</li>
                    <li>Values in seqused_kv must be less than or equal to KV_S.</li>
                </ul>
            </td>
            <td rowspan="6">Optional parameter</td>
            <td rowspan="6">None</td>
            <td>                <ul>
                    <li>When layout_q is neither TND nor NTD, at least one of seqused_q and max_seqlen_q must be passed.</li>
                    <li>When layout_q is TND/NTD or layout_kv is TND, passing batch_size is not supported.</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>seqused_kv</td>
            <td>                <ul>
                    <li>When layout_kv is the PA use case, it must be passed.</li>
                    <li>When layout_kv is neither TND nor the PA use case, at least one of seqused_kv and max_seqlen_kv must be passed.</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>cu_seqlens_q</td>
            <td>                <ul>
                    <li>tensor_type supports int32.</li>
                    <li>tensor_shape is (B+1,).</li>
                    <li>Values only support non-negative integers.</li>
                    <li>Values must be arranged in non-decreasing order (greater than or equal to the previous value), with the first element being 0 and the last element equal to Q_T.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>When layout_q is TND or NTD, it must be passed (in this case, both seqused_q and max_seqlen_q are optional).</li>
                    <li>When layout_q is neither TND nor NTD, passing it is not supported.</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>cu_seqlens_kv</td>
            <td>                <ul>
                    <li>tensor_type supports int32.</li>
                    <li>tensor_shape is (B+1,).</li>
                    <li>Values only support non-negative integers.</li>
                    <li>Values must be arranged in non-decreasing order (greater than or equal to the previous value), with the first element being 0 and the last element equal to KV_T.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>When layout_kv is TND, it must be passed (in this case, both seqused_kv and max_seqlen_kv are optional).</li>
                    <li>When layout_kv is not TND, passing it is not supported.</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>max_seqlen_q</td>
            <td>                <ul>
                    <li>data_type supports int32.</li>
                    <li>The default value is -1.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>When layout_q is neither TND nor NTD, at least one of seqused_q and max_seqlen_q must be passed.</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>max_seqlen_kv</td>
            <td>                <ul>
                    <li>data_type supports int32.</li>
                    <li>The default value is -1.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>When layout_kv is not TND and the use case is not PA, at least one of seqused_kv and max_seqlen_kv must be passed.</li>
                </ul>
            </td>
        </tr>
    </tbody>
</table>

> [!NOTE]
>
> The operator does not check whether the maximum values in `seqused_q`/`seqused_kv` are consistent with `max_seqlen_q`/`max_seqlen_kv`. If both are passed, ensure that `max(seqused_q) <= max_seqlen_q` and `max(seqused_kv) <= max_seqlen_kv`.

#### Paged Attention parameter group <a name="paged-attention-parameter-group"></a>

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
            <th>Single-Parameter Check</th>
            <th>Existence Check</th>
            <th>Consistency Check</th>
            <th>Feature Cross-Check</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>block_table</td>
            <td>                <ul>
                    <li>tensor_type supports only int32.</li>
                    <li>tensor_shape is (B, Bn).</li>
                    <li>The value must be a positive integer.</li>
                </ul>
            </td>
            <td>Optional parameter</td>
            <td>None</td>
            <td>                <ul>
                    <li>If layout_kv is PA, seqused_kv must be passed.</li>
                    <li>If layout_kv is PA, block_table must not be empty.</li>
                    <li>In the Paged Attention use case, k/v/k_descale/v_descale support non-contiguous tensors. The supported non-contiguous axes depend on layout_kv: when layout_kv is PA_BNBD/PA_NZ, axis 0 or axes 0 and 1 can be non-contiguous; when layout_kv is PA_BBND, only axis 0 can be non-contiguous.</li>
                    <li>Block size is supported only in the MxFP8 use case. The value can be 64, 128, 256, 512, or 1024.</li>
                    <li>FP8 (quant_mode=6) enforces the PA use case. block_table is required, and Bs is fixed to 128.</li>
                </ul>
            </td>
        </tr>
    </tbody>
</table>

#### Sinks parameter group

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
            <th>Single-Parameter Check</th>
            <th>Existence Check</th>
            <th>Consistency Check</th>
            <th>Feature Cross-Check</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>sinks</td>
            <td>                <ul>
                    <li>Not supported currently.</li>
                </ul>
            </td>
            <td>Optional parameter</td>
            <td>None</td>
            <td>Not supported currently.</td>
        </tr>
    </tbody>
</table>

#### SoftmaxLSE Parameter Group

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
            <th>Single-Parameter Check</th>
            <th>Existence Check</th>
            <th>Consistency Check</th>
            <th>Feature Cross-Check</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>return_softmax_lse</td>
            <td>                <ul>
                    <li>data_type only supports BOOL.</li>
                    <li>The value only supports True and False. True indicates that softmax_lse is enabled, and False indicates that softmax_lse is disabled.</li>
                </ul>
            </td>
            <td>Optional attribute. The default value is False.</td>
            <td rowspan="2">                <ul>
                    <li>If return_softmax_lse is False, an empty Tensor is output.</li>
                    <li>If return_softmax_lse is True, softmax_lse must not be empty, and the output shape is described in the <a href="#layout-matching-relationship-table">layout matching relationship table</a>.</li>
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

- quant_flash_attn_metadata + quant_flash_attn joint call example (TND, non-PA use case)

  ```python
  import torch
  import torch_npu
  import cann_ops_transformer

  torch_npu.npu.set_device(0)

  dtype = torch.float8_e4m3fn
  out_dtype = torch.bfloat16
  B = 2
  Q_S = 16
  KV_S = 16
  Q_N = 8
  KV_N = 2
  D = 128
  G = Q_N // KV_N

  # TND layout, where Q_T and KV_T are the cumulative sum of sequence lengths for each batch
  Q_T = B * Q_S
  KV_T = B * KV_S

  q = torch.randn(Q_T, Q_N, D, dtype=dtype, device="npu")
  k = torch.randn(KV_T, KV_N, D, dtype=dtype, device="npu")
  v = torch.randn(KV_T, KV_N, D, dtype=dtype, device="npu")

  # descale (4D, prefill use case)
  q_descale = torch.randn(Q_T, Q_N, D // 64, 2, dtype=torch.float8_e8m0, device="npu")
  k_descale = torch.randn(KV_T, KV_N, D // 64, 2, dtype=torch.float8_e8m0, device="npu")
  # The first dimension of v_descale is the sum of sequence lengths of each batch rounded up to a multiple of 64
  v_descale = torch.randn(B * ((KV_S + 63) // 64), KV_N, D, 2, dtype=torch.float8_e8m0, device="npu")

  # Cumulative sequence length (with leading 0)
  cu_seqlens_q = torch.tensor([0, Q_S, Q_S * 2], dtype=torch.int32, device="npu")
  cu_seqlens_kv = torch.tensor([0, KV_S, KV_S * 2], dtype=torch.int32, device="npu")

  metadata = cann_ops_transformer.ops.quant_flash_attn_metadata(
      Q_N,
      KV_N,
      D,
      quant_mode=1,
      cu_seqlens_q=cu_seqlens_q,
      cu_seqlens_kv=cu_seqlens_kv,
      max_seqlen_q=Q_S,
      max_seqlen_kv=KV_S,
      mask_mode=0,
      win_left=-1,
      win_right=-1,
      layout_q="TND",
      layout_q_descale="TND",
      layout_kv="TND",
      layout_out="TND",
      is_grad_enabled=False,
  )

  attn_out, softmax_lse = cann_ops_transformer.ops.quant_flash_attn(
      q, k, v,
      q_descale, k_descale, v_descale,
      quant_mode=1,
      cu_seqlens_q=cu_seqlens_q,
      cu_seqlens_kv=cu_seqlens_kv,
      metadata=metadata,
      softmax_scale=1.0 / (D ** 0.5),
      mask_mode=0,
      win_left=-1,
      win_right=-1,
      layout_q="TND",
      layout_q_descale="TND",
      layout_kv="TND",
      layout_out="TND",
      return_softmax_lse=False,
  )
  torch_npu.npu.synchronize()
  assert attn_out.shape == (Q_T, Q_N, D)
  assert attn_out.dtype == out_dtype
  assert torch.isfinite(attn_out.float()).all().item()
  ```

- Example of calling quant_flash_attn_metadata and quant_flash_attn together (TND + PA use case, causal mask)

  ```python
  import torch
  import torch_npu
  import cann_ops_transformer

  torch_npu.npu.set_device(0)

  dtype = torch.float8_e4m3fn
  out_dtype = torch.bfloat16
  B = 2
  Q_S = 16
  KV_S = 16
  Q_N = 8
  KV_N = 2
  D = 128
  G = Q_N // KV_N
  pa_block_size = 512

  # In the PA use case, the KV cache layout is BnNBsD.
  num_blocks_per_seq = (KV_S + pa_block_size - 1) // pa_block_size
  total_blocks = num_blocks_per_seq * B

  q = torch.randn(B * Q_S, Q_N, D, dtype=dtype, device="npu")
  k = torch.randn(total_blocks, KV_N, pa_block_size, D, dtype=dtype, device="npu")
  v = torch.randn(total_blocks, KV_N, pa_block_size, D, dtype=dtype, device="npu")

  # descale (PA-BnNBsD)
  q_descale = torch.randn(B * Q_S, Q_N, D // 64, 2, dtype=torch.float8_e8m0, device="npu")
  k_descale = torch.randn(total_blocks, KV_N, pa_block_size, D // 64, 2, dtype=torch.float8_e8m0, device="npu")
  v_descale = torch.randn(total_blocks, KV_N, pa_block_size // 64, D, 2, dtype=torch.float8_e8m0, device="npu")

  # block_table
  block_table = torch.arange(total_blocks, dtype=torch.int32, device="npu").reshape(B, num_blocks_per_seq)

  # Cumulative sequence lengths (with a leading 0)
  cu_seqlens_q = torch.tensor([0, Q_S, Q_S * 2], dtype=torch.int32, device="npu")

  # In the PA use case, seqused_kv is required.
  seqused_kv = torch.tensor([KV_S, KV_S], dtype=torch.int32, device="npu")

  # attn_mask (causal, 2048*2048)
  attn_mask = torch.tril(torch.ones(2048, 2048, dtype=torch.int8, device="npu"))

  metadata = cann_ops_transformer.ops.quant_flash_attn_metadata(
      Q_N,
      KV_N,
      D,
      quant_mode=1,
      cu_seqlens_q=cu_seqlens_q,
      seqused_kv=seqused_kv,
      max_seqlen_q=Q_S,
      max_seqlen_kv=KV_S,
      mask_mode=3,
      win_left=-1,
      win_right=-1,
      layout_q="TND",
      layout_q_descale="TND",
      layout_kv="PA_BNBD",
      layout_out="TND",
      is_grad_enabled=False,
  )

  attn_out, softmax_lse = cann_ops_transformer.ops.quant_flash_attn(
      q, k, v,
      q_descale, k_descale, v_descale,
      quant_mode=1,
      block_table=block_table,
      cu_seqlens_q=cu_seqlens_q,
      seqused_kv=seqused_kv,
      attn_mask=attn_mask,
      metadata=metadata,
      softmax_scale=1.0 / (D ** 0.5),
      mask_mode=3,
      win_left=-1,
      win_right=-1,
      layout_q="TND",
      layout_q_descale="TND",
      layout_kv="PA_BNBD",
      layout_out="TND",
      return_softmax_lse=False,
  )
  torch_npu.npu.synchronize()
  assert attn_out.shape == (B * Q_S, Q_N, D)
  assert attn_out.dtype == out_dtype
  assert torch.isfinite(attn_out.float()).all().item()
  ```
