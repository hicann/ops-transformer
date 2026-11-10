# flash_attn

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T09:21:12.320Z -->

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

  `flash_attn` is a `cann_ops_transformer` extension API based on `TorchNPU`. It calls the `FlashAttn` operator to perform non-quantized attention computation with shared KV (Key and Value use the same input), unifying training and inference.

  `flash_attn_metadata` is the metadata generation API of `flash_attn`, used to generate metadata before the main operator executes. The metadata records the task splitting results of AICore/AIVCore. The main operator can optionally pass in this metadata to optimize scheduling. The typical call flow is as follows:

  1. Prepare inputs such as `q`, `k`, and `v`.
  2. Call `flash_attn_metadata` to generate `metadata`.
  3. Call `flash_attn` and pass the `metadata` obtained in the previous step to the main operator.

- **Formula**:

  Self-attention builds an attention model using the relationships within the input samples. It assumes an input sample sequence $x$ of length $n$, where each element of $x$ is a $d$-dimensional vector. Each $d$-dimensional vector can be viewed as a token embedding. The sequence is transformed by 3 weight matrices to obtain 3 matrices of dimension $n \times d$.

  The self-attention formula is generally defined as follows, where $Q$, $K$, and $V$ are important property elements of the input sample, obtained by spatially transforming the input sample and unified into a single feature space. "Attention" in the formula and operator name is an abbreviation of "self-attention".

  $$
  Attention(Q,K,V)=Score(Q,K)V
  $$

  In this operator, the Score function uses the Softmax function. The self-attention calculation formula is:

  $$
  Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
  $$

  The product of $Q$ and $K^T$ represents the attention of input $x$. To prevent this value from becoming too large, it is usually scaled by dividing by $\sqrt{d}$. After softmax normalization on each row, multiplying by $V$ yields an $n \times d$ matrix.

  After **sink** is added, the calculation logic is as follows. The softmax_max and softmax_sum calculation parts are mainly modified.

  $$
  S = \frac{QK^T}{\sqrt{d}}
  $$

  $$
  m = max(sink, max(S))
  $$

  $$
  Attention = \frac{e^{S - m} \times V}{\sum e^{S-m} + e^{sink - m}}
  $$

  After **return_softmax_lse** is enabled, the softmax_lse return value is calculated as follows:

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
> The data layout formats of q, k, and v can be interpreted from multiple dimensions. B (Batch) indicates the batch size of input samples, batch_size; S (Seq-Length) indicates the sequence length of input samples; H (Hidden-Size) indicates the size of the hidden layer; N (Head-Num) indicates the number of heads; and D (Head-Dim) indicates the size of the smallest unit of the hidden layer, headdim, where D = H/N. Q_T indicates the cumulative sum of sequence lengths of all query Batch input samples, and KV_T indicates the cumulative sum of sequence lengths of all k and v Batch input samples.
> Q_S indicates seqlen_q, Q_N indicates nheads_q, KV_S indicates seqlen_kv, and KV_N indicates nheads_kv.

## Function Prototype

Before calling the flash_attn API, call the prerequisite API flash_attn_metadata first to complete the load balancing computation of flash_attn.

```python
cann_ops_transformer.flash_attn_metadata(
    num_heads_q,
    num_heads_kv,
    head_dim,
    *,
    head_dim_v=None,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_kv=None,
    mask_mode=None,
    win_left=None,
    win_right=None,
    layout_q=None,
    layout_kv=None,
    layout_out=None,
    is_grad_enabled=False
) -> Tensor
```

```python
cann_ops_transformer.flash_attn(
    q,
    k,
    v,
    block_table=None,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    sinks=None,
    metadata=None,
    softmax_scale=1.0,
    mask_mode=0,
    attn_mask=None,
    win_left=-1,
    win_right=-1,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    layout_q="BSND",
    layout_kv="BSND",
    layout_out="BSND",
    return_softmax_lse=False
) -> (Tensor, Tensor)
```

## Parameters

### flash_attn_metadata

| Name | Type | Required/Optional | Description | Data Type | Data Format | Dimension |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| num_heads_q | int | Required | Number of query heads | int32 | - | - |
| num_heads_kv | int | Required | Number of Key/Value heads | int32 | - | - |
| head_dim | int | Required | Dimension of each Query/Key attention head | int32 | - | - |
| head_dim_v | int | Optional | Dimension of each Value attention head. Default None (internally -1) indicates that it equals head_dim. Must be passed in when the QK and V dimensions differ (e.g., QK 192 / V 128), and must be consistent with the head_dim of value in the subsequent flash_attn call (the last dimension of value in contiguous layout; in PA_NZ layout, the value shape is (blocks, N2, D/16, block_size, 16) and head_dim = dim2*dim4) | int32 | - | - |
| cu_seqlens_q | Tensor | Optional | Cumulative sequence length, used to process variable-length sequences. The first element must be 0 | int32 | ND | (B+1,)
| cu_seqlens_kv | Tensor | Optional | Cumulative sequence length, used to process variable-length sequences. The first element must be 0 | int32 | ND | (B+1,)
| seqused_q | Tensor | Optional | Specifies the actual sequence length used in each batch, to truncate redundant computation | int32 | ND | (B,)
| seqused_kv | Tensor | Optional | Specifies the actual sequence length used in each batch, to truncate redundant computation | int32 | ND | (B,)
| batch_size | int | Optional | Batch size. If not passed in, it is derived from cu_seqlens_q or seqused_q. Default value is None | int32 | - | - |
| max_seqlen_q | int | Optional | Specifies the upper limit of the query q sequence length, and should be passed when seeking optimal performance | int32 | - | - |
| max_seqlen_kv | int | Optional | Specifies the upper limit of the key k and value v sequence lengths, and should be passed when seeking optimal performance | int32 | - | - |
| mask_mode | int | Optional | Mask mode | int32 | - | - |
| win_left | int | Optional | Left boundary of window | int32 | - | - |
| win_right | int | Optional | Right boundary of window | int32 | - | - |
| layout_q | string | Optional | Defines the layout format of input tensor q | string | - | - |
| layout_kv | string | Optional | Defines the layout format of input tensors k and v | string | - | - |
| layout_out | string | Optional | Defines the layout format of the output tensor | string | - | - |
| is_grad_enabled | bool | Optional | Whether to generate metadata required for backward gradients. Default value is False. When set to True, the metadata contains flash_attn_grad information required for backward | bool | - | - |

**About `head_dim_v`**: `flash_attn_metadata` has no q/k/v tensor input, so the head_dim of V cannot be inferred from the tensor shape. It is therefore explicitly declared by the scalar parameter `head_dim_v`:

- Default None (internally -1) indicates that **the head_dim of V equals `head_dim`** (QK dimension), consistent with the behavior in the common use case where qk==v.
- When the QK and V dimensions differ (e.g., QK 192 / V 128, MLA non-absorbed form), it **must be passed in** and must be consistent with the head_dim of value in the subsequent `flash_attn` call. In contiguous layout (BSND/BNSD/TND), it is the last dimension of value; in PA_NZ layout, the value shape is (blocks, N2, D/16, block_size, 16) and head_dim = dim2 × dim4. If inconsistent, the load balancing split differs from that of the main operator, which may cause incorrect results.
- The main operator `flash_attn` has no corresponding parameter: the head_dim of its V is taken directly from the last dimension of the value tensor. The currently supported (QK D, V DV) combinations are subject to validation of `flash_attn`: (64,64), (72,72), (128,128), (256,256), (192,128).

### flash_attn

| Name | Type | Required/Optional | Description | Data Type | Data Format | Dimension |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| q | Tensor | Required | q in the formula. | bfloat16/float16 | ND | <ul><li>(B, Q_S, Q_N, D)</li><li>(B, Q_N, Q_S, D)</li><li>(Q_T, Q_N, D)</li></ul> |
| k | Tensor | Required | k in the formula. | bfloat16/float16 | ND | <ul><li>(B, KV_S, KV_N, D)</li><li>(B, KV_N, KV_S, D)</li><li>(KV_T, KV_N, D)</li><li>(num_blocks, block_size, KV_N, D)</li><li>(num_blocks, KV_N, block_size, D)</li></ul> |
| v | Tensor | Required | v in the formula. | bfloat16/float16 | ND | <ul><li>(B, KV_S, KV_N, D)</li><li>(B, KV_N, KV_S, D)</li><li>(KV_T, KV_N, D)</li><li>(num_blocks, block_size, KV_N, D)</li><li>(num_blocks, KV_N, block_size, D)</li></ul> |
| block_table | Tensor | Optional | Block index mapping for paged attention computation. | int32 | ND | (B, max_num_blocks_per_seq) |
| cu_seqlens_q | Tensor | Optional | Cumulative sequence length used for variable-length sequences; the first element must be 0. | int32 | ND | (B+1,) |
| cu_seqlens_kv | Tensor | Optional | Cumulative sequence length used for variable-length sequences; the first element must be 0. | int32 | ND | (B+1,) |
| seqused_q | Tensor | Optional | Specifies the actual sequence length used in each batch to truncate redundant computation. | int32 | ND | (B,) |
| seqused_kv | Tensor | Optional | Specifies the actual sequence length used in each batch to truncate redundant computation. | int32 | ND | (B,) |
| sinks | Tensor | Optional | Attention sink parameter used to improve the numerical stability of self-attention computation. | float32 | ND | (Q_N,) |
| metadata | Tensor | Optional | Task splitting result generated by `flash_attn_metadata`. Passing it in optimizes scheduling. | int32 | ND | (max_schedule_size,) |
| softmax_scale | float | Optional | Explicitly sets the scaling factor to override the default computation. | float32 | - | - |
| mask_mode | int | Optional | Mask mode. | int32 | - | - |
| attn_mask | Tensor | Optional | Mask matrix. | int8 | ND | (2048, 2048)
| win_left | int | Optional | Left boundary of the window. | int32 | - | - |
| win_right | int | Optional | Right boundary of the window. | int32 | - | - |
| max_seqlen_q | int | Optional | Specifies the upper limit of the q sequence length, and should be passed when seeking optimal performance. | int32 | - | - |
| max_seqlen_kv | int | Optional | Specifies the upper limit of the k and v sequence lengths, and should be passed when seeking optimal performance. | int32 | - | - |
| layout_q | string | Optional | Defines input q tensor layout format. | string | - | - |
| layout_kv | string | Optional | Defines input k and v tensor layout format. | string | - | - |
| layout_out | string | Optional | Defines output tensor layout format. | string | - | - |
| return_softmax_lse | bool | Optional | Whether to obtain the LSE result of softmax. | bool | - | - |

## Returns

### flash_attn_metadata

| Name | Type | Required/Optional | Description | Data Type | Data Format | Dimension |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| metadata | Tensor | Required | Task splitting data of flash_attn. | int32 | ND | Shape is dynamically calculated based on batch_size and num_heads_kv. |

### flash_attn

| Name | Type | Required/Optional | Description | Data Type | Data Format | Dimension |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| attn_out | Tensor | Required | Computation output of flash_attn. | bfloat16/float16 | ND | <ul><li>(B, Q_S, Q_N, D)</li><li>(B, Q_N, Q_S, D)</li><li>(Q_T, Q_N, D )</li></ul> |
| softmax_lse | Tensor | Optional | LSE result of softmax. | float32 | ND | <ul><li>(B, Q_N, Q_S)</li><li>(Q_N, Q_T)</li></ul> |

**Note**

- attn_out: Constraint: The D dimension of this output parameter must be the same as the D of value, and the other dimensions must be consistent with the shape of the input query.
- softmax_lse: If return_softmax_lse is True, the output is generally a Tensor with shape (B, Q_N, Q_S), or a Tensor with shape (Q_N, Q_T) when input_q is TND. If return_softmax_lse is False, the output is a Tensor with shape [1] and a value of 0.

## Constraints

- The parameters cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv, block_table, and attn_mask are tensors. Since the operator cannot obtain the actual values of the tensors during the Tiling phase, the tiling side does not validate the values, and you must ensure their correctness. If invalid values are passed in for these parameters, undefined behavior occurs, such as precision issues or program crashes caused by invalid memory access.
- The input parameters of flash_attn_metadata and flash_attn must be consistent when the APIs are called. Since the operator is called in two phases through two APIs, the operator cannot validate the consistency itself, and you must ensure correctness. If inconsistent parameters are passed in, undefined behavior occurs, such as precision issues or program crashes caused by invalid memory access.
- **Optimal performance constraint**: For optimal performance, pass both `max_seqlen_q` and `max_seqlen_kv` to both `flash_attn_metadata` and `flash_attn`, as optimal performance cannot be guaranteed if either parameter is omitted.

### Feature Parameter Group

|      Feature Parameter Group      |     Parameter Field Name     |    Field Grouping    |  Field Type  |
| :-------------------: | :-------------------: | :-------------: | :--------: |
|      Common Parameter Group      |         q         |      INPUT      |   Tensor   |
|                      |          k          |      INPUT      | Tensor |
|                      |         v         |      INPUT      | Tensor |
|                      |         metadata        |      INPUT(OPTIONAL)      | Tensor |
|                      |      softmax_scale      | ATTR(OPTIONAL) |   float   |
|                      |      layout_q      | ATTR(OPTIONAL) |   string   |
|                      |      layout_kv      | ATTR(OPTIONAL) |   string   |
|                      |      layout_out      | ATTR(OPTIONAL) |   string   |
|                      |     attn_out     |     OUTPUT     |   Tensor   |
|      Mask Parameter Group      |       mask_mode       | ATTR(OPTIONAL) |   int   |
|                      |       win_left       | ATTR(OPTIONAL) |   int   |
|                      |      win_right      | ATTR(OPTIONAL) |   int   |
|                      |      attn_mask      | INPUT(OPTIONAL) |   Tensor   |
| SeqLens Parameter Group  |   cu_seqlens_q   | INPUT(OPTIONAL) |  Tensor  |
|                      |  cu_seqlens_kv  | INPUT(OPTIONAL) |  Tensor  |
|                      |  seqused_q  | INPUT(OPTIONAL) |  Tensor  |
|                      |  seqused_kv  | INPUT(OPTIONAL) |  Tensor  |
|                      |  max_seqlen_q  | ATTR(OPTIONAL) |  int  |
|                      |  max_seqlen_kv  | ATTR(OPTIONAL) |  int  |
| Paged Attention Parameter Group |      block_table      | INPUT(OPTIONAL) |   Tensor   |
|  Sinks Parameter Group  |     sinks     | INPUT(OPTIONAL) |   Tensor   |
|   SoftmaxLSE Parameter Group   |    return_softmax_lse    | ATTR(OPTIONAL) |    bool    |
|                      |      softmax_lse      |     OUTPUT     |   Tensor   |

### Baseline Information

In the documentation constraints, the meanings of common fields are as follows:

|    Naming    |                            Meaning                            |
| :---------: | :---------------------------------------------------------: |
|      B      |                Batch, indicating the batch size of input samples.                |
|     Q_N     |        Number of heads of the input q tensor, corresponding to N in the q shape.        |
|    KV_N    |    Number of heads of the input k/v tensor, corresponding to N in the k/v shape.    |
|     Q_S     |      Sequence length of the input q tensor, corresponding to S in the q shape.      |
|    KV_S    |  Sequence length of the input k/v tensor, corresponding to S in the k/v shape.  |
|     Q_T     |          Cumulative sum of sequence lengths of all batches in the input q tensor.          |
|     KV_T     |          Cumulative sum of sequence lengths of all batches in the input k/v tensor.          |
|     D     |          Minimum unit size (headdim) of the hidden layer of the input q/k/v tensors and the output attn_out.         |

### Parameter Group Constraints

#### Common Parameter Group

- Handling of empty input parameters:
  - An empty Tensor indicates required input or output with a shape size of 0, that is, any axis is 0.
  - Use cases triggering an empty tensor are all intercepted with an error.

- Validation of q, k, v, and attn_out:

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
                <li>tensor_type supports bfloat16 and float16.</li>
                <li>BNSD -> (B, N_Q, S_Q, D)</li>
                <li>BSND -> (B, S_Q, N_Q, D)</li>
                <li>TND -> (T_Q, N_Q, D)</li>
            </ul>
        </td>
        <td rowspan="4">            Must exist
        </td>
        <td rowspan="4">            <ul>
                <li>The data types of q, k, v, and attn_out must be the same.</li>
                <li>k and v must have the same shape.</li>
                <li>For layout validation rules, see the layout matching relationship table.</li>
            </ul>
        </td>
        <td rowspan="4">            Axis validation:
            <ul>
                <li>65536 > B > 0</li>
                <li>Q_T > 0</li>
                <li>KV_T > 0</li>
                <li>Q_N > 0</li>
                <li>KV_N > 0</li>
                <li>Q_S > 0</li>
                <li>KV_S > 0</li>
                <li>D supports 64, 72, 128, and 256.</li>
                <li>q and k/v must have the same D dimension.</li>
                <li>Q_N % KV_N == 0 and Q_N / KV_N > 0</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>k</td>
        <td rowspan="2">            <ul>
                <li>tensor_type supports bfloat16 and float16.</li>
                <li>BNSD -> (B, N_KV, S_KV, D)</li>
                <li>BSND -> (B, S_KV, N_KV, D)</li>
                <li>TND -> (T_KV, N_KV, D)</li>
                <li>PA_BBND -> (num_blocks, block_size, KV_N, D)</li>
                <li>PA_BNBD -> (num_blocks, KV_N, block_size, D)</li>
                <li>PA_NZ -> (num_blocks, KV_N, D//16, block_size, 16)</li>
                <li>1024 >= block_size >= 16, block_size % 16 == 0</li>
                <li>Non-contiguous tensor constraints:
                    <ul>
                        <li>BNSD/BSND/TND: k/v must be contiguous. An error is reported if a non-contiguous tensor is passed in.</li>
                        <li>PA_BBND: only dim0 supports non-contiguous tensors.</li>
                        <li>PA_BNBD/PA_NZ: dim0 or dim1 supports non-contiguous tensors.</li>
                    </ul>
                </li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>v</td>
    </tr>
    <tr>
        <td>attn_out</td>
        <td>            <ul>
                <li>data_type supports bfloat16 and float16.</li>
                <li>The shape supports 3 or 4 dimensions.</li>
            </ul>
        </td>
    </tr>
    </tbody>
    </table>

- Layout matching relationship table:

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
        <th>layout_kv</th>
        <th>layout_out</th>
        <th>layout_softmax_lse</th>
    </tr>
    </thead>
    <tbody>
        <tr>
            <td>BNSD</td>
            <td>            <li>BNSD</li>
            <li>PA_BBND</li>
            <li>PA_BNBD</li>
            <li>PA_NZ</li>
            </td>
            <td>            <li>BNSD</li>
            <li>BSND</li>
            </td>
            <td>(B, Q_N, Q_S)</td>
        </tr>
        <tr>
            <td>BSND</td>
            <td>            <li>BSND</li>
            <li>PA_BBND</li>
            <li>PA_BNBD</li>
            <li>PA_NZ</li>
            </td>
            <td>BSND</td>
            <td>(B, Q_N, Q_S)</td>
        </tr>
        <tr>
            <td>TND</td>
            <td>            <li>TND</li>
            <li>PA_BBND</li>
            <li>PA_BNBD</li>
            <li>PA_NZ</li>
            </td>
            <td>TND</td>
            <td>(Q_N, Q_T)</td>
        </tr>
    </tbody>
    </table>

- metadata validation:

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
                <th>Consistency validation</th>
                <th>Feature Cross-Validation</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>metadata</td>
                <td>                    <ul>
                        <li>tensor_type only supports int32.</li>
                        <li>shape is dynamically computed by flash_attn_metadata.</li>
                        <li>Currently, omitting this parameter is not supported. If it is not passed in, an interception alarm is raised.</li>
                    </ul>
                </td>
                <td>Optional parameter</td>
                <td>None</td>
                <td>When passed in, it must be consistent with the result generated by flash_attn_metadata.</td>
            </tr>
        </tbody>
    </table>

#### Mask parameter group

mask_mode parameter description
<ul>
    <li>mask_mode=0, full computation mode (default)</li>
    <li>mask_mode=3, Causal mode</li>
    <li>mask_mode=4, Window mode</li>
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
            <th>Consistency Validation</th>
            <th>Feature Cross-Validation</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>mask_mode</td>
            <td>                <ul>
                    <li>data_type supports INT.</li>
                    <li>The supported input range is only 0, 3, and 4. The default value is 0.</li>
                </ul>
            </td>
            <td>                Optional input. If this parameter is not passed in, the default value is 0.
            </td>
            <td rowspan="4">                <ul>
                    <li>When mask_mode is 0, win_left and win_right must be -1, and passing in attn_mask is not supported.</li>
                    <li>When mask_mode is 3, win_left and win_right must be -1, and the attn_mask matrix must be passed in.</li>
                    <li>When mask_mode is 4, the attn_mask matrix must be passed in.</li>
                </ul>
            </td>
            <td rowspan="4">                <ul>
                    <li>None</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>attn_mask</td>
            <td>                <ul>
                    <li>tensor_type supports int8.</li>
                    <li>The tensor_shape dimension is (2048, 2048). The matrix is fixed as a 2048*2048 lower triangular matrix with all diagonal elements set to 0.</li>
                </ul>
            </td>
            <td rowspan="3">None</td>
        </tr>
        <tr>
            <td>win_left</td>
            <td rowspan="2">                <ul>
                    <li>data_type supports INT.</li>
                    <li>If this value is not passed in or is set to -1, it indicates positive infinity.</li>
                    <li>The value must be >= -1.</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>win_right</td>
        </tr>
    </tbody>
</table>

#### Paged Attention parameter group

When block_table is not empty, Paged Attention is enabled.
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
                    <li>tensor_type only supports int32.</li>
                    <li>tensor_shape is (B, max_num_blocks_per_seq).</li>
                    <li>The value must be a positive integer.</li>
                    <li>1024 >= block_size >= 16, block_size % 16 == 0.</li>
                </ul>
            </td>
            <td>Optional parameter</td>
            <td>None</td>
            <td>                <ul>
                    <li>When PagedAttention is enabled, seqused_kv must be passed in.</li>
                    <li>When PagedAttention is enabled, block_table must not be empty.</li>
                    <li>When layout_kv is PA_NZ, D must be a multiple of 16.</li>
                </ul>
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
                    <li>tensor_type supports int32.</li>
                    <li>tensor_shape is (B,).</li>
                    <li>Only non-negative integers are supported.</li>
                    <li>Values in seqused_q must be less than or equal to Q_S.</li>
                    <li>Values in seqused_kv must be less than or equal to KV_S.</li>
                </ul>
            </td>
            <td>Optional parameter</td>
            <td>None</td>
            <td>                <ul>
                    <li>When layout_q is TND, seqused_q is optional.</li>
                    <li>When layout_q is not TND, pass one of the two: seqused_q or max_seqlen_q.</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>seqused_kv</td>
            <td>Optional parameter</td>
            <td>None</td>
            <td>                <ul>
                    <li>When layout_kv is TND, seqused_kv is optional.</li>
                    <li>When layout_kv is PA, seqused_kv must be passed in.</li>
                    <li>When layout_kv is not TND and not PA, pass one of the two: seqused_kv or max_seqlen_kv.</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>cu_seqlens_q</td>
            <td rowspan="2">                <ul>
                    <li>tensor_type supports int32.</li>
                    <li>tensor_shape is (B+1,).</li>
                    <li>Only non-negative integers are supported.</li>
                    <li>Values must be arranged in non-decreasing order (greater than or equal to the previous value).</li>
                    <li>cu_seqlens_q: the first element is 0 and the last element equals Q_T.</li>
                    <li>cu_seqlens_kv: the first element is 0 and the last element equals KV_T.</li>
                </ul>
            </td>
            <td rowspan="2">Optional parameter</td>
            <td rowspan="2">None</td>
            <td rowspan="2">                <ul>
                    <li>When layout_q is TND, cu_seqlens_q must be passed in.</li>
                    <li>When layout_q is not TND, passing cu_seqlens_q is not supported.</li>
                    <li>When layout_kv is TND, cu_seqlens_kv must be passed in.</li>
                    <li>When layout_kv is not TND, passing cu_seqlens_kv is not supported.</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>cu_seqlens_kv</td>
        </tr>
        <tr>
            <td>max_seqlen_q</td>
            <td rowspan="2">                <ul>
                    <li>data_type supports INT.</li>
                    <li>The default value is -1.</li>
                </ul>
            </td>
            <td rowspan="2">Optional parameter</td>
            <td rowspan="2">                <ul>
                    <li>The value must be greater than or equal to -1; if passed in, it must be equal to the actual maximum sequence length; otherwise, the behavior is undefined.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>When layout_q is TND, max_seqlen_q is optional.</li>
                    <li>When layout_q is not TND, pass one of the two: max_seqlen_q and seqused_q.</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>max_seqlen_kv</td>
            <td>                <ul>
                    <li>When layout_kv is TND or PA, max_seqlen_kv is optional.</li>
                    <li>When layout_kv is not TND and not PA, pass one of the two: max_seqlen_kv and seqused_kv.</li>
                </ul>
            </td>
        </tr>
        </tbody>
</table>

Combination relationship between layout_q, layout_kv, and SeqLens parameters:

- When layout_q is not TND, pass one of the two: seqused_q and max_seqlen_q; passing cu_seqlens_q is not supported. When layout_q is TND, cu_seqlens_q must be passed in, and both seqused_q and max_seqlen_q are optional parameters.
- When layout_kv is neither TND nor PA, pass one of the two: seqused_kv or max_seqlen_kv. Passing cu_seqlens_kv is not supported. When layout_kv is PA, seqused_kv must be passed in and max_seqlen_kv is an optional parameter. When layout_kv is TND, cu_seqlens_kv must be passed in and both seqused_kv and max_seqlen_kv are optional parameters.

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
                    <li>data_type only supports float32 (independent of the data type of q).</li>
                    <li>shape must be 1D, and its length equals Q_N (number of heads of q, that is, the number of heads before GQA grouping).</li>
                </ul>
            </td>
            <td> Optional parameter </td>
            <td> None </td>
            <td> None </td>
        </tr>
    </tbody>
</table>

#### SoftmaxLSE parameter group

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
                    <li>data_type only supports BOOL.</li>
                    <li>The value only supports True and False. True indicates that softmax_lse is enabled, and False indicates that softmax_lse is disabled.</li>
                </ul>
            </td>
            <td>Optional attribute. The default value is False.</td>
            <td rowspan="2">                <ul>
                    <li>When return_softmax_lse is False, the output is a tensor with shape [1] and value 0.</li>
                    <li>When return_softmax_lse is True, the output shape is described in the layout matching relationship table.</li>
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

- Example of calling flash_attn_metadata + flash_attn (BSND)

    ```python
    import torch
    import torch_npu
    import cann_ops_transformer

    torch_npu.npu.set_device(0)

    dtype = torch.bfloat16
    B = 2
    S = 16
    Q_N = 8
    KV_N = 2
    D = 128

    q = torch.randn(B, S, Q_N, D, dtype=dtype, device="npu")
    k = torch.randn(B, S, KV_N, D, dtype=dtype, device="npu")
    v = torch.randn(B, S, KV_N, D, dtype=dtype, device="npu")

    metadata = cann_ops_transformer.ops.flash_attn_metadata(
        Q_N,
        KV_N,
        D,
        batch_size=B,
        max_seqlen_q=S,
        max_seqlen_kv=S,
        mask_mode=0,
        win_left=-1,
        win_right=-1,
        layout_q="BSND",
        layout_kv="BSND",
        layout_out="BSND",
    )

    attn_out, softmax_lse = cann_ops_transformer.ops.flash_attn(
        q, k, v,
        metadata=metadata,
        softmax_scale=1.0 / (D ** 0.5),
        mask_mode=0,
        win_left=-1,
        win_right=-1,
        layout_q="BSND",
        layout_kv="BSND",
        layout_out="BSND",
        return_softmax_lse=False,
    )
    torch_npu.npu.synchronize()
    assert attn_out.shape == q.shape
    assert attn_out.dtype == q.dtype
    assert torch.isfinite(attn_out.float()).all().item()
    ```
