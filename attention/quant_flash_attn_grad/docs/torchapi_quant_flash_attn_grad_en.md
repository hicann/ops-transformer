# quant_flash_attn_grad

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T12:41:42.277Z -->

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

  `quant_flash_attn_grad` is an extension API of `cann_ops_transformer` based on `torch_npu`. It calls the `QuantFlashAttnGrad` operator to compute attention backward gradients in the HIFLOAT8 quantization scenario. This API is the backward counterpart of the `quant_flash_attn` forward operator. It computes the gradients of Query, Key, and Value (dq, dk, dv) and the sink gradient (dsink). Currently, the HIFLOAT8 quantized data type is supported, with two data layout formats: BSND and BNSD.
   The metadata generation API of `quant_flash_attn_grad` reuses `quant_flash_attn_metadata` to generate metadata before the main operator is executed. The metadata records the task split results of the AICore/AIVCore. The main operator can optionally pass in this metadata to optimize scheduling. The typical call flow is as follows:

  1. Prepare inputs such as `q`, `k`, and `v`.
  2. Call `quant_flash_attn_metadata` to generate `metadata`.
  3. Call `quant_flash_attn_grad` and pass the `metadata` obtained in the previous step to the main operator.

- **Formula**:

  The backward gradient computation of quantized attention consists of the following stages:

  Stage 1: Dequantize the input data, converting the quantized input to the high-precision floating-point domain

  $$
  Q_{fp} = Q_{quant} \times q\_descale
  $$

  $$
  K_{fp} = K_{quant} \times k\_descale
  $$

  $$
  V_{fp} = V_{quant} \times v\_descale
  $$

  $$
  dO_{fp} = dO_{quant} \times do\_descale
  $$

  Stage 2: Compute the Softmax gradient (PreSfmg stage)

  $$
  dP = dO \times V
  $$

  $$
  dS = dP \times p\_scale
  $$

  $$
  dS_{grad} = (dS - ds\_scale \times softmax(lse)) \times softmaxLse
  $$

  Stage 3: Compute Key/Value gradients (Cube main computation stage)

  $$
  dV = dS_{grad}^{T} \times Q
  $$

  $$
  dK = dS_{grad}^{T} \times Q
  $$

  $$
  dQ = dS_{grad} \times K
  $$

  Stage 4: Post-processing (Post stage), which converts intermediate results to the output dtype

  $$
  dq = dQ \times softmax\_scale
  $$

  $$
  dk = dK \times softmax\_scale
  $$

  $$
  dv = dV \times softmax\_scale
  $$

  $$
  dsink = reduce(dS_{grad}, dim=S)
  $$

  where $q\_descale$, $k\_descale$, $v\_descale$, and $do\_descale$ are quantization scale factors, $p\_scale$ is the scale factor of the P matrix, $ds\_scale$ is the dequantization scale factor, and $softmax\_scale$ is the scale coefficient (the recommended value is the reciprocal of $\sqrt{D}$).

> [!NOTE]
>
> The data layout formats of Q, K, and V can be interpreted from multiple dimensions: B (Batch) indicates the batch size of input samples, batch_size; S (Seq-Length) indicates the sequence length of input samples; N (Head-Num) indicates the number of heads; D (Head-Dim) indicates the size of the smallest unit of the hidden layer, headdim. Q_S indicates the sequence length of the input q tensor, Q_N indicates the number of heads of the input q tensor, KV_S indicates the sequence length of the input k/v tensor, and KV_N indicates the number of heads of the input k/v tensor.

## Function Prototype

Before calling the quant_flash_attn_grad API, call the prerequisite API quant_flash_attn_metadata first to complete the load balancing computation of quant_flash_attn_grad.

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
    v_descale=None,
    batch_size=None,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    mask_mode=0,
    win_left=-1,
    win_right=-1,
    layout_q="BSND",
    layout_q_descale="BSND",
    layout_kv="BSND",
    layout_out="BSND",
    is_grad_enabled=True
) -> Tensor
```

```python
cann_ops_transformer.quant_flash_attn_grad(
    q,
    k,
    v,
    dout,
    attn_out,
    q_descale,
    k_descale,
    v_descale,
    do_descale,
    p_scale,
    ds_scale,
    softmax_lse,
    *,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    sinks=None,
    attn_mask=None,
    metadata=None,
    quant_mode=0,
    softmax_scale=1.0,
    mask_mode=0,
    win_left=-1,
    win_right=-1,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    layout_q="BSND",
    layout_kv="BSND"
) -> (Tensor, Tensor, Tensor, Tensor)
```

## Parameters

### quant_flash_attn_metadata

| Name | Type | Required/Optional | Description | Data Type | Data Format | Dimension | Non-contiguous Tensor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| num_heads_q | int | Required | Number of Query heads. | int32 | - | - | - |
| num_heads_kv | int | Required | Number of Key/Value heads. | int32 | - | - | - |
| head_dim | int | Required | Dimension of each attention head. | int32 | - | - | - |
| quant_mode | int/QuantMode | Required | Quantization mode. Supports passing an enumeration or the corresponding int value. For the enumeration definition, see "quant_mode enumeration". | int32 | - | - | - |
| cu_seqlens_q | Tensor | Optional | Cumulative sequence length of Q, used to handle variable-length sequences. The first element must be 0. | int32 | ND | (B+1,) | × |
| cu_seqlens_kv | Tensor | Optional | Cumulative sequence length of KV, used to handle variable-length sequences. The first element must be 0. | int32 | ND | (B+1,) | × |
| seqused_q | Tensor | Optional | Specifies the actual sequence length used in each batch of Q to truncate redundant computation. | int32 | ND | (B,) | × |
| seqused_kv | Tensor | Optional | Specifies the actual sequence length used in each batch of KV to truncate redundant computation. | int32 | ND | (B,) | × |
| v_descale | Tensor | Optional | Dequantization scale factor of v. | float32 | ND | (1,) | × |
| batch_size | int | Optional | Batch size. Derived from cu_seqlens_q or seqused_q if not passed. The default value is None. | int32 | - | - | - |
| max_seqlen_q | int | Optional | Upper limit of the sequence length of query q. | int32 | - | - | - |
| max_seqlen_kv | int | Optional | Upper limit of the sequence length of key k and value v. | int32 | - | - | - |
| mask_mode | int/MaskMode | Optional | Mask mode. Supports passing an enumeration or the corresponding int value. For the enumeration definition, see "mask_mode enumeration". | int32 | - | - | - |
| win_left | int | Optional | Left boundary of the window. | int32 | - | - | - |
| win_right | int | Optional | Right boundary of the window. | int32 | - | - | - |
| layout_q | string | Optional | Defines the layout format of the input q tensor. | string | - | - | - |
| layout_q_descale | string | Optional | Defines the layout format of the input q_descale tensor. | string | - | - | - |
| layout_kv | string | Optional | Defines the layout format of the input k and v tensors. | string | - | - | - |
| layout_out | string | Optional | Defines the layout format of the output tensor. | string | - | - | - |
| is_grad_enabled | bool | Optional | Specifies whether to enable metadata generation for the backward gradient use case. The default value is True. When True, the metadata is used by the corresponding backward operator quant_flash_attn_grad. | BOOL | - | - | - |

### quant_flash_attn_grad

| Name | Type | Required/Optional | Description | Data Type | Data Format | Dimension | Non-contiguous Tensor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| q | Tensor | Required | Q in the formula, quantized data. | uint8 | ND | BSND: (B, Q_S, Q_N, D)<br>BNSD: (B, Q_N, Q_S, D) | × |
| k | Tensor | Required | K in the formula, quantized data. | uint8 | ND | BSND: (B, KV_S, KV_N, D)<br>BNSD: (B, KV_N, KV_S, D) | × |
| v | Tensor | Required | V in the formula, quantized data. | uint8 | ND | BSND: (B, KV_S, KV_N, D)<br>BNSD: (B, KV_N, KV_S, D) | × |
| dout | Tensor | Required | Gradient of the forward output attn_out, quantized data. | uint8 | ND | Same shape as q | × |
| attn_out | Tensor | Required | attn_out output of the forward computation. | bfloat16 | ND | Same shape as q | × |
| q_descale | Tensor | Required | Dequantization scale factor of q. | float32 | ND | (1,) | × |
| k_descale | Tensor | Required | Dequantization scale factor of k. | float32 | ND | (1,) | × |
| v_descale | Tensor | Required | Dequantization scale factor of v. | float32 | ND | (1,) | × |
| do_descale | Tensor | Required | Dequantization scale factor of dout. | float32 | ND | (1,) | × |
| p_scale | Tensor | Required | Quantization scale factor of the P matrix. | float32 | ND | (1,) | × |
| ds_scale | Tensor | Required | Dequantization scale factor. | float32 | ND | (1,) | × |
| softmax_lse | Tensor | Required | softmaxLse output of the attention forward computation. | float32 | ND | (B, Q_N, Q_S, 1) | × |
| cu_seqlens_q | Tensor | Optional | Cumulative sequence length of Q, used to handle variable-length sequences. The first element must be 0. | int32 | ND | (B+1,) | × |
| cu_seqlens_kv | Tensor | Optional | Cumulative sequence length of KV, used to handle variable-length sequences. The first element must be 0. | int32 | ND | (B+1,) | × |
| seqused_q | Tensor | Optional | Specifies the actual sequence length of q used in each batch, truncating redundant computation. | int32 | ND | (B,) | × |
| seqused_kv | Tensor | Optional | Specifies the actual sequence length of kv used in each batch, truncating redundant computation. | int32 | ND | (B,) | × |
| sinks | Tensor | Optional | Input tensor for the sink use case. Not supported in the current version, pass None. | float32 | ND | (Q_N,) | × |
| attn_mask | Tensor | Optional | Mask matrix. | int8/uint8/bool | ND | (2048, 2048) | × |
| metadata | Tensor | Optional | Output result of the AICPU operator with tiling offloaded. | int32 | ND | (2, max_schedule_size) | × |
| quant_mode | int/QuantMode | Required | Quantization mode. Supports passing an enumeration or the corresponding int value. See "quant_mode Enumeration" for the enumeration definition. | int32 | - | - | - |
| softmax_scale | float | Optional | Scaling factor, default value is 1.0. Recommended value: reciprocal of sqrt(head_dim). | float32 | - | - | - |
| mask_mode | int/MaskMode | Optional | Mask mode. Supports passing an enumeration or the corresponding int value. See "mask_mode Enumeration" for the enumeration definition. Only 0 is supported in the current version. | int32 | - | - | - |
| win_left | int | Optional | Left boundary of the window, default value is -1. | int32 | - | - | - |
| win_right | int | Optional | Right boundary of the window, default value is -1. | int32 | - | - | - |
| max_seqlen_q | int | Optional | Specifies the upper limit of the query q sequence length. -1 indicates automatic inference. Default value is -1. | int32 | - | - | - |
| max_seqlen_kv | int | Optional | Specifies the upper limit of the sequence lengths of key k and value v. -1 indicates automatic inference. Default value is -1. | int32 | - | - | - |
| layout_q | string | Optional | Defines the layout format of the input q Tensor. Supports "BSND" and "BNSD", default value is "BSND". | string | - | - | - |
| layout_kv | string | Optional | Defines the layout format of the input k/v Tensor. Supports "BSND" and "BNSD", default value is "BSND". | string | - | - | - |

### quant_mode Enumeration

`quant_mode` supports passing the `IntEnum` enumeration or the corresponding int value in the Python API. The enumeration is defined in `cann_ops_transformer.ops.quant_flash_attn_grad`.

| Enumeration name | Value | Meaning |
| :--- | :---: | :--- |
| `HIF8_PER_TENSOR` | 0 | HIFLOAT8 per-tensor quantization. |

### mask_mode Enumeration

In the Python API, `mask_mode` supports passing the `IntEnum` enumeration or the corresponding int value. The enumeration is defined in `cann_ops_transformer.ops.quant_flash_attn_grad`.

| Enumeration Name | Value | Meaning |
| :--- | :---: | :--- |
| `ALL` | 0 | Full computation mode (default value) |
| `CAUSAL` | 3 | Causal mode |
| `WINDOW` | 4 | Sliding Window mode |

> [!NOTE]
>
> The enumeration is `IntEnum` and can be directly passed as an int to the underlying operator. The API only supports passing the enumeration or the corresponding int value. The current version only supports mask_mode = 0 (`ALL`); other modes are not supported yet.

## Return Values

### quant_flash_attn_metadata

| Name | Type | Required/Optional | Description | Data Type | Data Format | Dimension | Non-contiguous Tensor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| metadata | Tensor | Required | Task splitting data of quant_flash_attn_grad. | int32 | ND | (2, max_schedule_size) | × |

### quant_flash_attn_grad

| Name | Type | Required/Optional | Description | Data Type | Data Format | Dimension | Non-contiguous Tensor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| dq | Tensor | Required | gradient of Query. | bfloat16 | ND | BSND: (B, Q_S, Q_N, D)<br>BNSD: (B, Q_N, Q_S, D) | × |
| dk | Tensor | Required | gradient of Key. | bfloat16 | ND | BSND: (B, KV_S, KV_N, D)<br>BNSD: (B, KV_N, KV_S, D) | × |
| dv | Tensor | Required | gradient of Value. | bfloat16 | ND | BSND: (B, KV_S, KV_N, D)<br>BNSD: (B, KV_N, KV_S, D) | × |
| dsink | Tensor | Required | gradient of sink. | float32 | ND | (Q_N,) | × |

## Constraints

### Feature Parameter Group

<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 200px">
        <col style="width: 200px">
        <col style="width: 250px">
        <col style="width: 150px">
    </colgroup>
    <thead>
        <tr>
            <th>Feature Parameter Group</th>
            <th>Parameter Field Name</th>
            <th>Field Group</th>
            <th>Field Type</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="12">Common Parameter Group</td>
            <td>q</td>
            <td>INPUT</td>
            <td>Tensor</td>
        </tr>
        <tr>
            <td>k</td>
            <td>INPUT</td>
            <td>Tensor</td>
        </tr>
        <tr>
            <td>v</td>
            <td>INPUT</td>
            <td>Tensor</td>
        </tr>
        <tr>
            <td>dout</td>
            <td>INPUT</td>
            <td>Tensor</td>
        </tr>
        <tr>
            <td>attn_out</td>
            <td>INPUT</td>
            <td>Tensor</td>
        </tr>
        <tr>
            <td>metadata</td>
            <td>INPUT(OPTIONAL)</td>
            <td>Tensor</td>
        </tr>
        <tr>
            <td>softmax_scale</td>
            <td>ATTR(OPTIONAL)</td>
            <td>float</td>
        </tr>
        <tr>
            <td>layout_q</td>
            <td>ATTR(OPTIONAL)</td>
            <td>string</td>
        </tr>
        <tr>
            <td>layout_kv</td>
            <td>ATTR(OPTIONAL)</td>
            <td>string</td>
        </tr>
        <tr>
            <td>dq</td>
            <td>OUTPUT</td>
            <td>Tensor</td>
        </tr>
        <tr>
            <td>dk</td>
            <td>OUTPUT</td>
            <td>Tensor</td>
        </tr>
        <tr>
            <td>dv</td>
            <td>OUTPUT</td>
            <td>Tensor</td>
        </tr>
        <tr>
            <td rowspan="8">Full quantization parameter group</td>
            <td>quant_mode</td>
            <td>ATTR</td>
            <td>int</td>
        </tr>
        <tr>
            <td>q_descale</td>
            <td>INPUT</td>
            <td>Tensor</td>
        </tr>
        <tr>
            <td>k_descale</td>
            <td>INPUT</td>
            <td>Tensor</td>
        </tr>
        <tr>
            <td>v_descale</td>
            <td>INPUT</td>
            <td>Tensor</td>
        </tr>
        <tr>
            <td>do_descale</td>
            <td>INPUT</td>
            <td>Tensor</td>
        </tr>
        <tr>
            <td>p_scale</td>
            <td>INPUT</td>
            <td>Tensor</td>
        </tr>
        <tr>
            <td>ds_scale</td>
            <td>INPUT</td>
            <td>Tensor</td>
        </tr>
        <tr>
            <td>softmax_lse</td>
            <td>INPUT</td>
            <td>Tensor</td>
        </tr>
        <tr>
            <td rowspan="4">Mask parameter group</td>
            <td>mask_mode</td>
            <td>ATTR(OPTIONAL)</td>
            <td>int</td>
        </tr>
        <tr>
            <td>win_left</td>
            <td>ATTR(OPTIONAL)</td>
            <td>int</td>
        </tr>
        <tr>
            <td>win_right</td>
            <td>ATTR(OPTIONAL)</td>
            <td>int</td>
        </tr>
        <tr>
            <td>attn_mask</td>
            <td>INPUT(OPTIONAL)</td>
            <td>Tensor</td>
        </tr>
        <tr>
            <td rowspan="6">SeqLengths parameter group</td>
            <td>cu_seqlens_q</td>
            <td>INPUT(OPTIONAL)</td>
            <td>Tensor</td>
        </tr>
        <tr>
            <td>cu_seqlens_kv</td>
            <td>INPUT(OPTIONAL)</td>
            <td>Tensor</td>
        </tr>
        <tr>
            <td>seqused_q</td>
            <td>INPUT(OPTIONAL)</td>
            <td>Tensor</td>
        </tr>
        <tr>
            <td>seqused_kv</td>
            <td>INPUT(OPTIONAL)</td>
            <td>Tensor</td>
        </tr>
        <tr>
            <td>max_seqlen_q</td>
            <td>ATTR(OPTIONAL)</td>
            <td>int</td>
        </tr>
        <tr>
            <td>max_seqlen_kv</td>
            <td>ATTR(OPTIONAL)</td>
            <td>int</td>
        </tr>
        <tr>
            <td>Sinks parameter group</td>
            <td>sinks</td>
            <td>INPUT(OPTIONAL)</td>
            <td>Tensor</td>
        </tr>
        <tr>
            <td>DSink output parameter group.</td>
            <td>dsink</td>
            <td>OUTPUT</td>
            <td>Tensor</td>
        </tr>
    </tbody>
</table>

### Baseline Information

Common fields in the documentation constraints are defined as follows:

|    Name    |                            Meaning                            |
| :---------: | :---------------------------------------------------------: |
|      B      |                Batch, indicating the batch size of input samples.                |
|     Q_N     |        Number of heads of the input q tensor, corresponding to N in the q shape.        |
|    KV_N    |    Number of heads of the input k/v tensor, corresponding to N in the k/v shape.    |
|     Q_S     |      Sequence length of the input q tensor, corresponding to S in the q shape.      |
|    KV_S    |  Sequence length of the input k/v tensor, corresponding to S in the k/v shape.  |
|     D     |          Minimum unit size of the hidden layer (headdim) of the input q/k/v tensors and the output dq/dk/dv.         |

### Parameter Group Constraints

#### Common Parameter Group

- Handling of empty input parameters:
  - An empty Tensor means that the shape size of a required input or output is 0, that is, any axis is 0.
  - All use cases that trigger an empty Tensor will be intercepted and report errors.

- The parameters cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv, and attn_mask are tensors. Since the operator cannot obtain the specific values of the tensors during the Tiling phase, the tiling side does not validate the values, and users must ensure correctness. If invalid values are passed for the preceding parameters, undefined behavior (such as precision issues and program crashes caused by illegal memory access) will be triggered.
- The input parameters of quant_flash_attn_metadata and quant_flash_attn_grad must be consistent during the call. Since the operator is called through two APIs in phases, the operator cannot validate them itself, and customers must ensure correctness. If the parameters passed to the APIs are inconsistent, undefined behavior (such as precision issues and program crashes caused by illegal memory access) occurs.

- Validation of q, k, v, dout, and attn_out:

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
            <td>                <ul>
                    <li>tensor_type supports only uint8.</li>
                    <li>shape dim supports only 4.</li>
                </ul>
            </td>
            <td rowspan="5">                Must exist.
            </td>
            <td rowspan="5">                <ul>
                    <li>The data types of q, k, v, and dout must be the same (all uint8).</li>
                    <li>The shape of q must be consistent with that of dout and attn_out.</li>
                    <li>The shape of k must be consistent with that of v.</li>
                </ul>
            </td>
            <td rowspan="5">                Axis checks:
                <ul>
                    <li>65536 > B > 0.</li>
                    <li>Q_S ≥ 0; KV_S ≥ 0.</li>
                    <li>D supports only 128.</li>
                    <li>Q_N % KV_N == 0 and Q_N / KV_N > 0 (GQA constraint).</li>
                    <li>Q_N ≤ 128; KV_N ≤ 128.</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>k</td>
            <td>                <ul>
                    <li>tensor_type supports only uint8.</li>
                    <li>shape dim supports only 4.</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>v</td>
            <td>                <ul>
                    <li>tensor_type supports only uint8.</li>
                    <li>shape dim supports only 4.</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>dout</td>
            <td>                <ul>
                    <li>tensor_type only supports uint8.</li>
                    <li>shape dim only supports 4.</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>attn_out</td>
            <td>                <ul>
                    <li>data_type only supports bfloat16.</li>
                    <li>shape dim only supports 4.</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>layout_q</td>
            <td>Supports BSND/BNSD.</td>
            <td rowspan="2">Omitting this parameter is not supported currently. If it is not passed, an interception alarm is raised.</td>
            <td rowspan="2">layout_q and layout_kv must be consistent.</td>
            <td rowspan="2">None</td>
        </tr>
        <tr>
            <td>layout_kv</td>
            <td>Supports BSND/BNSD.</td>
        </tr>
        <tr>
            <td>metadata</td>
            <td>                <ul>
                    <li>tensor_type only supports int32.</li>
                    <li>shape is (2, max_schedule_size).</li>
                </ul>
            </td>
            <td>optional parameter</td>
            <td>None</td>
            <td>None</td>
        </tr>
    </tbody>
    </table>

#### Full Quantization Parameter Group

- Explanation of quant_mode parameter:

    <ul>
        <li>quant_mode=0, HIF8_PER_TENSOR (HIFLOAT8 per-tensor quantization scenario)</li>
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
                <td>                    <ul>
                        <li>data_type supports int32.</li>
                        <li>The supported input range is 0.</li>
                    </ul>
                </td>
                <td>Required attribute</td>
                <td>None</td>
                <td rowspan="8">                    <ul>
                        <li>Non-contiguous Tensor is not supported.</li>
                        <li>D is fixed at 128.</li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td>q_descale</td>
                <td>                    <ul>
                        <li>tensor_type only supports float32</li>
                        <li>shape only supports (1,)</li>
                    </ul>
                </td>
                <td rowspan="7">Must exist</td>
                <td rowspan="7">                    <ul>
                        <li>The dtype of all descale/scale must be float32.</li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td>k_descale</td>
                <td>                    <ul>
                        <li>tensor_type only supports float32</li>
                        <li>shape only supports (1,)</li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td>v_descale</td>
                <td>                    <ul>
                        <li>tensor_type only supports float32.</li>
                        <li>shape only supports (1,).</li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td>do_descale</td>
                <td>                    <ul>
                        <li>tensor_type only supports float32.</li>
                        <li>shape only supports (1,).</li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td>p_scale</td>
                <td>                    <ul>
                        <li>tensor_type only supports float32.</li>
                        <li>shape only supports (1,).</li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td>ds_scale</td>
                <td>                    <ul>
                        <li>tensor_type only supports float32.</li>
                        <li>shape only supports (1,).</li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td>softmax_lse</td>
                <td>                    <ul>
                        <li>tensor_type only supports float32.</li>
                        <li>shape is (B, Q_N, Q_S, 1).</li>
                    </ul>
                </td>
            </tr>
        </tbody>
    </table>

#### Mask parameter group

mask_mode parameter description
<ul>
    <li>mask_mode=0, ALL, all computation mode (default value)</li>
    <li>mask_mode=3, CAUSAL, Causal mode</li>
    <li>mask_mode=4, WINDOW, Window mode</li>
</ul>

> [!NOTE]
>
> The current version supports only mask_mode=0 (`ALL`). Other modes are not supported yet.

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
            <td>mask_mode</td>
            <td>                <ul>
                    <li>data_type supports int32.</li>
                    <li>The current version supports only 0 as input.</li>
                </ul>
            </td>
            <td>                Optional attribute. The default value is 0.
            </td>
            <td rowspan="3">                <ul>
                    <li>When mask_mode is 0, passing attn_mask is not supported.</li>
                </ul>
            </td>
            <td rowspan="3">                <ul>
                    <li>The current version supports only mask_mode=0. Other modes are not supported yet.</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>attn_mask</td>
            <td>                <ul>
                    <li>tensor_type supports int8/uint8/bool.</li>
                    <li>tensor_shape is (2048, 2048).</li>
                </ul>
            </td>
            <td>                Optional input
            </td>
        </tr>
        <tr>
            <td>win_left<br>win_right</td>
            <td>                <ul>
                    <li>data_type supports int32.</li>
                    <li>The value must be ≥ -1.</li>
                </ul>
            </td>
            <td>                Optional attribute, effective only when mask_mode=4.
                <li>The default value is -1, indicating infinity (an extremely large value).</li>
            </td>
        </tr>
    </tbody>
</table>

#### SeqLengths Parameter Group

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
                    <li>The value only supports non-negative integers.</li>
                    <li>Values in seqused_q must be less than or equal to Q_S.</li>
                    <li>Values in seqused_kv must be less than or equal to KV_S.</li>
                </ul>
            </td>
            <td rowspan="6">Optional parameter</td>
            <td rowspan="6">None</td>
            <td>None</td>
        </tr>
        <tr>
            <td>seqused_kv</td>
            <td>None</td>
        </tr>
        <tr>
            <td>cu_seqlens_q</td>
            <td>                <ul>
                    <li>tensor_type supports int32.</li>
                    <li>tensor_shape is (B+1,).</li>
                    <li>The value only supports non-negative integers.</li>
                    <li>The values must be non-decreasing (greater than or equal to the previous value), with the first element being 0 and the last element equal to Q_T.</li>
                </ul>
            </td>
            <td>None</td>
        </tr>
        <tr>
            <td>cu_seqlens_kv</td>
            <td>                <ul>
                    <li>tensor_type supports int32.</li>
                    <li>tensor_shape is (B+1,).</li>
                    <li>The value only supports non-negative integers.</li>
                    <li>The values must be non-decreasing (greater than or equal to the previous value), with the first element being 0 and the last element equal to KV_T.</li>
                </ul>
            </td>
            <td>None</td>
        </tr>
        <tr>
            <td>max_seqlen_q</td>
            <td>                <ul>
                    <li>data_type supports int32.</li>
                    <li>Default value is -1.</li>
                </ul>
            </td>
            <td>None</td>
        </tr>
        <tr>
            <td>max_seqlen_kv</td>
            <td>                <ul>
                    <li>data_type supports int32.</li>
                    <li>Default value is -1.</li>
                </ul>
            </td>
            <td>None</td>
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
                    <li>Not supported in the current version.</li>
                </ul>
            </td>
            <td>optional parameter</td>
            <td>None</td>
            <td>Not supported in the current version. Pass None.</td>
        </tr>
    </tbody>
</table>

## Deterministic Computation

- Deterministic computation: quant_flash_attn_grad defaults to a deterministic implementation.

## Example

- Example of calling quant_flash_attn_grad (BSND layout, HIFLOAT8 quantization scenario)

    ```python
    import torch
    import torch_npu
    import cann_ops_transformer
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    torch_npu.npu.set_device(0)

    B = 1
    Q_S = 512
    KV_S = 512
    Q_N = 1
    KV_N = 1
    D = 128

    # BSND layout
    q = torch.zeros(B, Q_S, Q_N, D, dtype=torch.uint8, device="npu")
    k = torch.zeros(B, KV_S, KV_N, D, dtype=torch.uint8, device="npu")
    v = torch.zeros(B, KV_S, KV_N, D, dtype=torch.uint8, device="npu")
    dout = torch.zeros(B, Q_S, Q_N, D, dtype=torch.uint8, device="npu")
    attn_out = torch.zeros(B, Q_S, Q_N, D, dtype=torch.bfloat16, device="npu")

    # descale / scale: FP32, shape=(1,)
    q_descale = torch.ones(1, dtype=torch.float32, device="npu")
    k_descale = torch.ones(1, dtype=torch.float32, device="npu")
    v_descale = torch.ones(1, dtype=torch.float32, device="npu")
    do_descale = torch.ones(1, dtype=torch.float32, device="npu")
    p_scale = torch.ones(1, dtype=torch.float32, device="npu")
    ds_scale = torch.ones(1, dtype=torch.float32, device="npu")

    # softmax_lse: FP32, shape=(B, Q_N, Q_S, 1)
    softmax_lse = torch.zeros(B, Q_N, Q_S, 1, dtype=torch.float32, device="npu")

    try:
        metadata = cann_ops_transformer.quant_flash_attn_metadata(
            num_heads_q=Q_N,
            num_heads_kv=KV_N,
            head_dim=D,
            quant_mode=0,
            cu_seqlens_q=None,
            cu_seqlens_kv=None,
            seqused_q=None,
            seqused_kv=None,
            batch_size=B,
            max_seqlen_q=Q_S,
            max_seqlen_kv=KV_S,
            mask_mode=0,
            layout_q="BSND",
            layout_kv="BSND",
            is_grad_enabled=True
        )
    except Exception as e:
        logger.error("[MAIN_WRAPPER] quant_flash_attn_metadata rebuild failed: %s", str(e))
        raise

    dq, dk, dv, dsink = cann_ops_transformer.ops.quant_flash_attn_grad(
        q, k, v, dout, attn_out,
        q_descale, k_descale, v_descale, do_descale, p_scale, ds_scale, softmax_lse,
        metadata=metadata,
        quant_mode=0,
        softmax_scale=1.0 / (D ** 0.5),
        mask_mode=0,
        win_left=-1,
        win_right=-1,
        max_seqlen_q=Q_S,
        max_seqlen_kv=KV_S,
        layout_q="BSND",
        layout_kv="BSND",
    )
    torch_npu.npu.synchronize()
    assert dq.shape == (B, Q_S, Q_N, D)
    assert dk.shape == (B, KV_S, KV_N, D)
    assert dv.shape == (B, KV_S, KV_N, D)
    assert dq.dtype == torch.bfloat16
    ```
