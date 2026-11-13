# generic_block_sparse_attention

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T10:01:34.631Z -->

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

  `generic_block_sparse_attention` is an extension API of `cann_ops_transformer` based on `TorchNPU`, used to call the `GenericBlockSparseAttention` operator to perform block sparse attention computation with arbitrary granularity.

  `generic_block_sparse_attention_metadata` is the metadata generation API of `generic_block_sparse_attention`, used to generate metadata before the main operator is executed. The metadata records the task splitting result of AICore/AIVCore. The main operator can optionally pass in the metadata to optimize scheduling. The typical call flow is as follows:

  1. Prepare inputs such as `q`, `k`, `v`, `sparse_block_idx`, and `sparse_block_count`.
  2. Call `generic_block_sparse_attention_metadata` to generate `metadata`.
  3. Call `generic_block_sparse_attention` and pass the `metadata` obtained in the previous step to the main operator.

- **Formula**:

  This operator performs sparse computation block by block along the sequence dimension on the basis of standard Attention. The sparse block size is $blockShapeX \times blockShapeY$: Q is divided by $blockShapeX$, and KV by $blockShapeY$; `sparse_block_idx` gives the index of the KV block selected for each Q block, and `sparse_block_count` gives the number of KV blocks actually used in computation for each Q block. Only these blocks are used to compute $q k^{T}$, Softmax, and the product with $v$.

  $$
  attention\_out = Softmax(softmax\_scale \cdot q \cdot k_{sparse}^{T} + attn\_mask) \cdot v_{sparse}
  $$

  In the formula, $softmax\_scale$ is the scale factor; $k_{sparse}$ and $v_{sparse}$ indicate the KV blocks selected based on the preceding indexes.

> [!NOTE]
>
> The data layout formats of q, k, and v can be interpreted from multiple dimensions, where B (Batch) indicates the batch size batch_size of input samples, S (Seq-Length) indicates the sequence length of input samples, H (Hidden-Size) indicates the hidden layer size, N (Head-Num) indicates the number of heads, and D (Head-Dim) indicates the size of the smallest unit of the hidden layer (headdim), satisfying D=H/N. Q_T indicates the cumulative sum of the sequence lengths of all query batch input samples, and KV_T indicates the cumulative sum of the sequence lengths of all k and v batch input samples.
> Q_S indicates seqlen_q, Q_N indicates nheads_q, KV_S indicates seqlen_kv, and KV_N indicates nheads_kv.

## Function Prototype

Before calling the generic_block_sparse_attention API, call the prerequisite API generic_block_sparse_attention_metadata first to complete the load balancing computation for generic_block_sparse_attention.

```python
cann_ops_transformer.generic_block_sparse_attention_metadata(
    sparse_block_idx,
    sparse_block_count,
    num_heads_q,
    num_heads_kv,
    head_dim,
    block_shape,
    *,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    is_packed_gqa=True,
    layout_q="TND",
    layout_kv="PA_BBND",
    mask_mode=1,
    quant_mode=0,
    softmax_precision=1,
    win_left=-1,
    win_right=-1,
) -> Tensor
```

```python
cann_ops_transformer.generic_block_sparse_attention(
    q,
    k,
    v,
    sparse_block_idx,
    sparse_block_count,
    block_shape,
    *,
    metadata=None,
    attn_mask=None,
    q_dequant_scale=None,
    k_dequant_scale=None,
    v_dequant_scale=None,
    p_quant_scale=None,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    block_table=None,
    is_packed_gqa=True,
    layout_q="TND",
    layout_kv="PA_BBND",
    softmax_scale=0.0,
    mask_mode=1,
    quant_mode=0,
    dst_type_max=0.0,
    softmax_precision=1,
    win_left=-1,
    win_right=-1,
    return_softmax_lse=False,
    attention_out_dtype=None,
) -> (Tensor, Tensor)
```

## Enumerations

`quant_mode` and `mask_mode` in the Python API support passing an `IntEnum` enumeration or the corresponding int value. The enumerations are defined in `cann_ops_transformer.ops.generic_block_sparse_attention`:

### quant_mode Enumeration

| Enumeration Name | Value | Meaning |
| :--- | :---: | :--- |
| `NO_QUANT` | 0 | Non-quantization (default value). |
| `FP8_E4M3_STATIC_PER_GROUP` | 1 | FP8_E4M3 static per-group. |
| `FP8_E4M3_DYNAMIC_MX` | 2 | FP8_E4M3 dynamic MX. |
| `FP4_E2M1_DYNAMIC_OCP` | 3 | FP4_E2M1 dynamic OCP. |
| `FP4_E2M1_DYNAMIC_CX` | 4 | FP4_E2M1 dynamic CX. |
| `FP8_E4M3_STATIC_CAST_P` | 5 | FP8_E4M3 static cast P. |

### mask_mode Enumeration

| Enumeration Name | Value | Meaning |
| :--- | :---: | :--- |
| `NO_MASK` | 0 | No mask |
| `CAUSAL` | 1 | Causal mode (default value) |
| `WINDOW` | 2 | Window mode |
| `UNION` | 3 | Union exactMask mode |

> [!NOTE]
>
> The enumeration is of type `IntEnum` and can be directly passed to the underlying operator as an int. The API only supports passing the enumeration or the corresponding int value. mask_mode supports 1 (`CAUSAL`); non-quant FLOAT16/BFLOAT16 scenarios also support 0 (`NO_MASK`), and so does MXFP8 on Ascend 950PR&950DT products; non-quant FLOAT16/BFLOAT16 (all architectures) and FP8 quantization on Ascend 950PR&950DT products also support 3 (Union), as described under product-specific constraints. quant_mode currently supports 0 (`NO_QUANT`), 5 (`FP8_E4M3_STATIC_CAST_P`), and 6 (`MXFP8_E4M3_PER_CHANNEL_V`).

## Parameters

### generic_block_sparse_attention_metadata

| Name | Type | Required/Optional | Description | Data Type | Data Format | Dimension |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| sparse_block_idx | Tensor | Required | KV block indexes selected by each Q block. | int32 | ND | (KV_N, totalQBlocks, topK) |
| sparse_block_count | Tensor | Required | Number of KV blocks actually retained for each Q block. | int32 | ND | (KV_N, totalQBlocks) |
| num_heads_q | int | Required | Number of Query heads. | int32 | - | - |
| num_heads_kv | int | Required | Number of Key/Value heads. | int32 | - | - |
| head_dim | int | Required | Dimension of each attention head. | int32 | - | - |
| block_shape | list[int] | Required | Sparse block shape `[block_x, block_y]`. | int64 | - | Length of 2 |
| cu_seqlens_q | Tensor | Optional | Cumulative sequence lengths, used to process variable-length sequences. The first element must be 0. | int64 | ND | (B+1,) |
| cu_seqlens_kv | Tensor | Optional | Cumulative sequence lengths, used to process variable-length sequences. The first element must be 0. | int64 | ND | (B+1,) |
| seqused_q | Tensor | Optional | Specifies the sequence length actually used in each batch to truncate redundant computation. | int32 | ND | (B,) |
| seqused_kv | Tensor | Optional | Specifies the KV sequence length actually used in each batch to truncate redundant computation. | int32 | ND | (B,) |
| max_seqlen_q | int | Optional | Specifies the upper limit of the query q sequence length. | int32 | - | - |
| max_seqlen_kv | int | Optional | Specifies the upper limit of the key k and value v sequence lengths. | int32 | - | - |
| is_packed_gqa | bool | Optional | Whether to enable Packed GQA. | bool | - | - |
| layout_q | string | Optional | Defines the layout format of the input query tensor. | string | - | - |
| layout_kv | string | Optional | Defines the layout format of the input key/value tensor. | string | - | - |
| mask_mode | int/MaskMode | Optional | Mask mode. Supports passing an enumeration or the corresponding int value. For the enumeration definition, see "mask_mode enumeration". | int32 | - | - |
| quant_mode | int/QuantMode | Optional | Quantization mode. Supports passing an enumeration or the corresponding int value. For the enumeration definition, see "quant_mode enumeration". | int32 | - | - |
| softmax_precision | int | Optional | Softmax precision mode. | int32 | - | - |
| win_left | int | Optional | Left boundary of the window. | int32 | - | - |
| win_right | int | Optional | Right boundary of the window. | int32 | - | - |

### generic_block_sparse_attention

| Name | Type | Required/Optional | Description | Data Type | Data Format | Dimension |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| q | Tensor | Required | q in the formula | bfloat16/float16/float8_e4m3fn | ND | (Q_T, Q_N, D) |
| k | Tensor | Required | k in the formula | bfloat16/float16/float8_e4m3fn | ND | (num_blocks, block_size, KV_N, D) |
| v | Tensor | Required | v in the formula | bfloat16/float16/float8_e4m3fn | ND | (num_blocks, block_size, KV_N, D) |
| sparse_block_idx | Tensor | Required | KV block index selected by each Q block | int32 | ND | (KV_N, totalQBlocks, topK) |
| sparse_block_count | Tensor | Required | Number of KV blocks actually retained for each Q block | int32 | ND | (KV_N, totalQBlocks) |
| block_shape | list[int] | Required | Sparse block shape `[block_x, block_y]` | int64 | - | length of 2 |
| metadata | Tensor | Optional | Task split result generated by `generic_block_sparse_attention_metadata`. Passing it optimizes scheduling | int32 | ND | (1024,) |
| attn_mask | Tensor | Optional | Must be None when mask_mode=0/1; must be INT32 exactMask when mask_mode=3 (Union) | bool/int32 | ND | Union: (KV_N, G, U×block_y) |
| q_dequant_scale | Tensor | Optional | query dequantization scale factor | float32/float8_e8m0 | ND | - |
| k_dequant_scale | Tensor | Optional | key dequantization scale factor | float32/float8_e8m0 | ND | - |
| v_dequant_scale | Tensor | Optional | value dequantization scale factor | float32/float8_e8m0 | ND | - |
| p_quant_scale | Tensor | Optional | P quantization scale factor | float32 | ND | - |
| cu_seqlens_q | Tensor | Optional | Cumulative sequence length, used to process variable-length sequences. The first element must be 0 | int64 | ND | (B+1,) |
| cu_seqlens_kv | Tensor | Optional | Cumulative sequence length, used to process variable-length sequences. The first element must be 0 | int64 | ND | (B+1,) |
| seqused_q | Tensor | Optional | Specifies the sequence length actually used in each batch to truncate redundant computation | int32 | ND | (B,) |
| seqused_kv | Tensor | Optional | Specifies the kv sequence length actually used in each batch to truncate redundant computation | int32 | ND | (B,) |
| block_table | Tensor | Optional | Block index mapping used in Paged Attention computation | int32 | ND | (B, max_num_blocks_per_seq) |
| is_packed_gqa | bool | Optional | Whether to enable Packed GQA | bool | - | - |
| layout_q | string | Optional | Defines the layout format of the input query tensor | string | - | - |
| layout_kv | string | Optional | Defines the layout format of the input key/value tensor | string | - | - |
| softmax_scale | float | Optional | Scale factor that can be explicitly set to override the default computation | float32 | - | - |
| mask_mode | int/MaskMode | Optional | Mask mode. Supports passing an enumeration or the corresponding int value. See "mask_mode enumeration" for the enumeration definition | int32 | - | - |
| quant_mode | int/QuantMode | Optional | Quantization mode. Supports passing an enumeration or the corresponding int value. See "quant_mode enumeration" for the enumeration definition | int32 | - | - |
| dst_type_max | float | Optional | Quantization-related parameter | float32 | - | - |
| softmax_precision | int | Optional | Softmax precision mode | int32 | - | - |
| win_left | int | Optional | Window left boundary | int32 | - | - |
| win_right | int | Optional | Window right boundary | int32 | - | - |
| return_softmax_lse | bool | Optional | Whether to obtain the LSE result of softmax | bool | - | - |
| attention_out_dtype | dtype | Optional | Output dtype. Required when `quant_mode!=0`. When `quant_mode=0` and not specified, it is the same as q | ScalarType | - | - |

## Return Values

### generic_block_sparse_attention_metadata

| Name | Type | Required/Optional | Description | Data Type | Data Format | Dimension |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| metadata | Tensor | Required | Task splitting data of generic_block_sparse_attention | int32 | ND | (1024,) |

### generic_block_sparse_attention

| Name | Type | Required/Optional | Description | Data Type | Data Format | Dimensions |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| attention_out | Tensor | Required | Computation output of generic_block_sparse_attention. | bfloat16/float16 | ND | (Q_T, Q_N, D) |
| softmax_lse | Tensor | Optional | LSE result of softmax. | float32 | ND | (Q_T, Q_N, 1) or empty |

**NOTE**

- attention_out: Constraint: The shape of this output parameter must be the same as the shape of the input parameter query. When `quant_mode=0`, the dtype defaults to that of q. When `quant_mode!=0`, the dtype is specified by attention_out_dtype.
- softmax_lse: If return_softmax_lse is True, the output is a Tensor with shape (Q_T, Q_N, 1) in TND layout. If return_softmax_lse is False, the output is an empty Tensor.

## Constraints

- Parameters cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv, sparse_block_idx, sparse_block_count, and block_table are tensors. The operator cannot obtain the specific values of these tensors during the Tiling phase. Therefore, the tiling side does not validate the values, and users must ensure their correctness. If invalid values are passed for the preceding parameters, undefined behavior may occur (such as accuracy issues or program crashes caused by invalid memory access).
- The input parameters of generic_block_sparse_attention_metadata and generic_block_sparse_attention must be consistent during the call. The operator is called in two phases through two APIs and cannot perform the check itself. Customers must ensure correctness. If the parameters passed to the APIs are inconsistent, undefined behavior may occur (such as accuracy issues or program crashes caused by invalid memory access).

### Product-Specific Constraints

- Ascend 950PR&amp;950DT non-quant FLOAT16/BFLOAT16: `mask_mode` supports 0/1/3; non-union uses `block_shape=[1,Y]` (`1≤Y≤128`); union uses `block_shape=[8,Y]` (currently only 8); `residual_block_mode` supports 0 or 1 (union supports only 0).
- Non-quant FLOAT16/BFLOAT16 on Atlas A2/A3 products: `head_dim` supports 1 to 256; `layout_q/layout_kv` supports `TND/PA_BBND`, `TND/PA_BNBD`, `TND/TND`, `BSND/BSND`, `BNSD/BNSD`, `BSND/PA_BBND`, `BNSD/PA_BBND`, `BSND/PA_BNBD`, and `BNSD/PA_BNBD`; `mask_mode` supports 0/1/3; non-union uses `block_shape=[1,Y]` (Y supports 1, 2, 4, 8, 16, 32, 64, 128) and union (mask_mode=3) uses `block_shape=[8,Y]`; `residual_block_mode` supports 0 or 1 (union supports only 0). TND KV requires `cu_seqlens_kv`, and dense Q/K/V are stored contiguously in the corresponding layout. In causal mode, the actual KV length of each batch must be no less than the actual Q length.
- Union (mask_mode=3) on Atlas A2/A3 products supports non-quant FP16/BF16 with `block_shape=[8,Y]` and `residual_block_mode=0`. Y supports 1, 2, 4, 8, 16, 32, 64, 128, and the PA KV page size is 128. Dense KV with the same layout (TND, BSND, BNSD) and paged KV (PA_BBND, PA_BNBD) are supported. INT32 exactMask is required, and the product of the GQA group size and blockShapeX must not exceed 128.
- Regular FP8 on Ascend 950PR&amp;950DT: `head_dim=128`, `layout_q/layout_kv=TND/PA_BBND`, `mask_mode=1`, `residual_block_mode=0`, `block_shape=[1,Y]` (`1≤Y≤128`). Union FP8 requires INT32 exactMask, `block_shape=[8,Y]` (`1≤Y≤128`), `residual_block_mode=0`, and `head_dim=128`. Union does not support `quant_mode=6`.
- Sparse-index last-dimension capacity is 1 to INT32_MAX for non-quant and FP8 paths.

### Feature Parameter Group

|      Feature Parameter Group      |     Parameter Field Name     |    Field Grouping    |  Field Type  |
| :-------------------: | :-------------------: | :-------------: | :--------: |
|      Common Parameter Group      |         q         |      INPUT      |   Tensor   |
|                      |         k         |      INPUT      |   Tensor   |
|                      |         v         |      INPUT      |   Tensor   |
|                      |  sparse_block_idx  |      INPUT      |   Tensor   |
|                      | sparse_block_count  |      INPUT      |   Tensor   |
|                      |     block_shape     |   ATTR(REQUIRED) |  int[]  |
|                      |    is_packed_gqa    | ATTR(OPTIONAL) |   bool   |
|                      |       metadata       | INPUT(OPTIONAL) |   Tensor   |
|                      |    softmax_scale    | ATTR(OPTIONAL) |   float   |
|                      |  softmax_precision  | ATTR(OPTIONAL) |   int   |
|                      |      layout_q       | ATTR(OPTIONAL) |   string   |
|                      |      layout_kv      | ATTR(OPTIONAL) |   string   |
|                      |   attention_out   |     OUTPUT     |   Tensor   |
|   metadata Parameter Group   |   max_seqlen_q   | ATTR(OPTIONAL) |   int   |
|                      |  max_seqlen_kv  | ATTR(OPTIONAL) |   int   |
|                      |   num_heads_q   | ATTR(REQUIRED) |   int   |
|                      |  num_heads_kv  | ATTR(REQUIRED) |   int   |
|                      |     head_dim     | ATTR(REQUIRED) |   int   |
|                      |       metadata       |     OUTPUT     |   Tensor   |
|      Mask Parameter Group      |      mask_mode      | ATTR(OPTIONAL) |   int   |
|                      |      win_left       | ATTR(OPTIONAL) |   int   |
|                      |     win_right      | ATTR(OPTIONAL) |   int   |
|                      |      attn_mask      | INPUT(OPTIONAL) |   Tensor   |
| SeqLengths Parameter Group  |   cu_seqlens_q   | INPUT(OPTIONAL) |  Tensor  |
|                      |  cu_seqlens_kv  | INPUT(OPTIONAL) |  Tensor  |
|                      |    seqused_q    | INPUT(OPTIONAL) |  Tensor  |
|                      |   seqused_kv   | INPUT(OPTIONAL) |  Tensor  |
| Paged Attention Parameter Group |     block_table     | INPUT(OPTIONAL) |   Tensor   |
|    Quant Parameter Group    |    quant_mode    | ATTR(OPTIONAL) |   int   |
|                      | q_dequant_scale | INPUT(OPTIONAL) |  Tensor  |
|                      | k_dequant_scale | INPUT(OPTIONAL) |  Tensor  |
|                      | v_dequant_scale | INPUT(OPTIONAL) |  Tensor  |
|                      |  p_quant_scale  | INPUT(OPTIONAL) |  Tensor  |
|                      |  dst_type_max  | ATTR(OPTIONAL) |  float  |
|                      | attention_out_dtype | ATTR(OPTIONAL) | ScalarType |
|   SoftmaxLSE Parameter Group   | return_softmax_lse | ATTR(OPTIONAL) |    bool    |
|                      |    softmax_lse    |     OUTPUT     |   Tensor   |

### Baseline Information

In the constraints, the definitions of common fields are as follows:

| Name | Meaning |
| :---: | :---: |
| B | Batch, indicating input sample batch size. |
| Q_N | Number of heads of the input q tensor, corresponding to N in the q shape. |
| KV_N | Number of heads of the input k/v tensor, corresponding to N in the k/v shape. |
| Q_T | Cumulative sum of sequence lengths of all batches in the input q tensor. |
| KV_T | Cumulative sum of sequence lengths of all batches in the input k/v. |
| Q_S | Logical sequence length of the query in each batch, generally obtained by differencing cu_seqlens_q in TND layout. |
| KV_S | Logical sequence length of the key/value in each batch; in TND layout, generally obtained by differencing cu_seqlens_kv; in Paged Attention, the logical KV length corresponding to the block_table mapping. |
| D | headdim, the minimum unit size of the hidden layer in the input q/k/v tensor. |
| topK | Last dimension of sparse_block_idx, indicating the maximum number of KV blocks selected for each Q block. |
| totalQBlocks | Total number of Q blocks after each batch is split by Q_S in TND layout; $\sum_i\mathrm{ceil}(Q\_S_i / block\_x)$, the second dimension of `sparse_block_idx`/`sparse_block_count`. |
| num_blocks | Total number of physical pages in the Paged KV Cache. |
| block_size | Number of tokens on a single page of the Paged KV Cache. |

### Parameter Group Constraints

#### Common Parameter Group

- Handling of empty input parameters:
  - An empty tensor indicates that the shape size of a required input or output is 0, that is, at least one axis is 0.
  - All use cases that trigger an empty tensor are intercepted and report an error.

- q, k, v, and attention_out checks:

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
                <li>tensor_type supports bfloat16, float16, and float8_e4m3fn.</li>
                <li>TND -> (Q_T, Q_N, D)</li>
            </ul>
        </td>
        <td rowspan="4">            Must exist.
        </td>
        <td rowspan="4">            <ul>
                <li>The data types of q, k, and v must be the same.</li>
                <li>The shapes of k and v must be the same.</li>
                <li>For the layout check rules, see <a href="#layout-matching-relationship-table">layout matching relationship table</a>.</li>
            </ul>
        </td>
        <td rowspan="4">            Axis check:
            <ul>
                <li>Q_T > 0</li>
                <li>Q_N > 0</li>
                <li>KV_N > 0</li>
                <li>Currently, D only supports 128.</li>
                <li>Q_N % KV_N == 0 and Q_N / KV_N > 0</li>
                <li>Currently, groupSize = Q_N / KV_N must not exceed 128.</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>k</td>
        <td rowspan="2">            <ul>
                <li>tensor_type supports bfloat16, float16, and float8_e4m3fn.</li>
                <li>PA_BBND -> (num_blocks, block_size, KV_N, D)</li>
                <li>Currently, block_size must be 128 and equal to block_shape[1].</li>
                <li>Non-contiguous tensor constraint: only dim0 of PA_BBND/PA_BNBD supports non-contiguous layout; MXFP8 normalizes page storage as described below.</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>v</td>
    </tr>
    <tr>
        <td>attention_out</td>
        <td>            <ul>
                <li>When quant_mode=0, data_type is the same as that of q; when quant_mode!=0, data_type is specified by attention_out_dtype.</li>
                <li>The shape is the same as that of q.</li>
            </ul>
        </td>
    </tr>
    </tbody>
    </table>

- layout matching relationship table: <a name="layout-matching-relationship-table"></a>

    <table style="undefined;table-layout: fixed; width:1625px"><colgroup>
    <col style="width: 247px">
    <col style="width: 232px">
    <col style="width: 293px">
    <col style="width: 293px">
    </colgroup>
    <thead>
    <tr>
        <th>layout_q</th>
        <th>layout_kv</th>
        <th>attention_out</th>
        <th>softmax_lse</th>
    </tr>
    </thead>
    <tbody>
        <tr>
            <td>TND</td>
            <td>PA_BBND</td>
            <td>(Q_T, Q_N, D)</td>
            <td>(Q_T, Q_N, 1)</td>
        </tr>
    </tbody>
    </table>

- Check of sparse_block_idx, sparse_block_count, block_shape, and is_packed_gqa:

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
    <th>Single-parameter check</th>
    <th>Existence Check</th>
    <th>Consistency Check</th>
    <th>Feature Cross-Check</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td>sparse_block_idx</td>
        <td>            <ul>
                <li>tensor_type only supports int32.</li>
                <li>shape is (KV_N, totalQBlocks, topK).</li>
            </ul>
        </td>
        <td rowspan="3">Required</td>
        <td rowspan="3">            <ul>
                <li>The metadata interface and main operator must pass the same sparse_block_idx, sparse_block_count, and block_shape.</li>
                <li>topK must be no less than the maximum value of all elements in sparse_block_count.</li>
            </ul>
        </td>
        <td rowspan="3">            <ul>
                <li>topK capacity upper limit is INT32_MAX (must be ≥ max of sparse_block_count).</li>
                <li>Non-union: block_shape=[1, Y]; union (mask_mode=3): block_shape=[8, Y] (unionSize from block_shape[0], currently only 8).</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>sparse_block_count</td>
        <td>            <ul>
                <li>tensor_type only supports int32.</li>
                <li>shape is (KV_N, totalQBlocks).</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>block_shape</td>
        <td>            <ul>
                <li>int list with a length of 2</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>is_packed_gqa</td>
        <td>            <ul>
                <li>data_type only supports BOOL.</li>
                <li>Currently only supports True.</li>
            </ul>
        </td>
        <td>Optional attribute, default value is True.</td>
        <td>The metadata API and main operator must be consistent.</td>
        <td>None</td>
    </tr>
    </tbody>
    </table>

- metadata check:

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
                <td>metadata</td>
                <td>                    <ul>
                        <li>tensor_type only supports int32.</li>
                        <li>The shape is fixed to (1024,).</li>
                        <li>Currently, omitting this parameter is not supported. An interception alarm is reported if it is not passed.</li>
                    </ul>
                </td>
                <td>Optional Parameter</td>
                <td>None</td>
                <td>If passed, it must be consistent with the result generated by generic_block_sparse_attention_metadata.</td>
            </tr>
        </tbody>
    </table>

- softmax_precision check:

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
                <td>softmax_precision</td>
                <td>                    <ul>
                        <li>data_type supports INT.</li>
                        <li>The value is 0 or 1. In non-Union scenarios (mask_mode=0/1) of Ascend 950PR&950DT products, FP8 quantization supports only 1; in Union scenarios (mask_mode=3), only 0 is supported. On Atlas A2/A3 products, FLOAT16 supports 0 or 1 and BFLOAT16 supports only 0 in non-Union scenarios, and only 0 is supported in Union scenarios (mask_mode=3).</li>
                    </ul>
                </td>
                <td>Optional attribute, default value is 1.</td>
                <td>The metadata interface and main operator must be consistent.</td>
                <td>None</td>
            </tr>
        </tbody>
    </table>

#### Mask Parameter Group

For details about the mask_mode parameter, see "mask_mode Enumeration".

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
                    <li>data_type supports INT</li>
                    <li>Supports 0 (non-causal) and 1 (causal); Ascend 950 and non-quantized Atlas A2/A3 also support 3 (union). See product-specific constraints.</li>
                </ul>
            </td>
            <td>Optional input, default value is 1</td>
            <td rowspan="4">                <ul>
                    <li>The mask_mode of the metadata interface and the main operator must be consistent.</li>
                    <li>Currently win_left and win_right must be -1.</li>
                </ul>
            </td>
            <td rowspan="4">                <ul>
                    <li>When mask_mode=1, the built-in causal mask of the operator is used.</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>attn_mask</td>
            <td>                <ul>
                    <li>mask_mode=0/1: must be None</li>
                    <li>mask_mode=3: must pass INT32 exactMask, shape [KV_N, G, U×block_y]; bit i marks whether the i-th Q in the group can see that column</li>
                </ul>
            </td>
            <td>None</td>
        </tr>
        <tr>
            <td>win_left</td>
            <td rowspan="2">                <ul>
                    <li>data_type supports INT.</li>
                    <li>Currently only supports -1.</li>
                </ul>
            </td>
            <td rowspan="2">Optional input. The default value is -1.</td>
        </tr>
        <tr>
            <td>win_right</td>
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
                    <li>Only non-negative integers are supported.</li>
                    <li>Values in seqused_q must be less than or equal to Q_S.</li>
                    <li>Values in seqused_kv must be less than or equal to KV_S.</li>
                </ul>
            </td>
            <td rowspan="2">Optional parameter</td>
            <td rowspan="4">None</td>
            <td rowspan="4">None</td>
        </tr>
        <tr>
            <td>seqused_kv</td>
        </tr>
        <tr>
            <td>cu_seqlens_q</td>
            <td rowspan="2">                <ul>
                    <li>tensor_type supports int64.</li>
                    <li>tensor_shape is (B+1,).</li>
                    <li>Only non-negative integers are supported.</li>
                    <li>Values must be in non-decreasing order (greater than or equal to the preceding value).</li>
                    <li>cu_seqlens_q: the first element is 0 and the last element equals Q_T.</li>
                    <li>cu_seqlens_kv: the first element is 0 and the last element equals KV_T.</li>
                </ul>
            </td>
            <td rowspan="2">                <ul>
                    <li>If layout_q is TND, cu_seqlens_q must be passed.</li>
                    <li>If layout_q is not TND, passing cu_seqlens_q is not supported.</li>
                    <li>If layout_kv is TND, cu_seqlens_kv must be passed.</li>
                    <li>If layout_kv is not TND, passing cu_seqlens_kv is not supported.</li>
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
                    <li>Default value is -1.</li>
                </ul>
            </td>
            <td rowspan="2">Optional parameter</td>
            <td rowspan="2">                <ul>
                    <li>The value must be greater than or equal to -1; when passed, it must equal the actual maximum sequence length. Otherwise, the behavior is undefined.</li>
                </ul>
            </td>
            <td rowspan="2">None</td>
        </tr>
        <tr>
            <td>max_seqlen_kv</td>
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
            <th>Single-parameter Check</th>
            <th>Existence Check</th>
            <th>Consistency Check</th>
            <th>Feature Cross-Check</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>block_table</td>
            <td>                <ul>
                    <li>tensor_type only supports int32.</li>
                    <li>tensor_shape is (B, max_num_blocks_per_seq).</li>
                    <li>The value must be a positive integer.</li>
                </ul>
            </td>
            <td>Optional parameter</td>
            <td>None</td>
            <td>                <ul>
                    <li>When PagedAttention is enabled, seqused_kv must be passed.</li>
                    <li>When PagedAttention is enabled, block_table must not be empty.</li>
                </ul>
            </td>
        </tr>
    </tbody>
</table>

#### Quant Parameter Group

For details about the quant_mode parameter, see the quant_mode enumeration.

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
                    <li>data_type supports INT.</li>
                    <li>Currently, only 0 and 5 are supported.</li>
                </ul>
            </td>
            <td>Optional input. The default value is 0.</td>
            <td rowspan="5">                <ul>
                    <li>The quant_mode of the metadata interface and the main operator must be the same.</li>
                </ul>
            </td>
            <td rowspan="5">                <ul>
                    <li>If quant_mode=0, q/k/v_dequant_scale must be None.</li>
                    <li>If quant_mode!=0, attention_out_dtype must be passed, and the output dtype in current quantization use cases must be float16 or bfloat16.</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>q_dequant_scale</td>
            <td rowspan="3">                <ul>
                    <li>Must be None in non-quantization use cases.</li>
                </ul>
            </td>
            <td rowspan="3">optional parameter</td>
        </tr>
        <tr>
            <td>k_dequant_scale</td>
        </tr>
        <tr>
            <td>v_dequant_scale</td>
        </tr>
        <tr>
            <td>p_quant_scale</td>
            <td>                <ul>
                    <li>Currently must be None.</li>
                </ul>
            </td>
            <td>None</td>
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
            <th>Single-parameter Check</th>
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
                    <li>Non-quant supports False/True; FP8 union (D=128) supports False/True; other FP8 paths support only False.</li>
                </ul>
            </td>
            <td>Optional attribute, default value is False.</td>
            <td rowspan="2">                <ul>
                    <li>If return_softmax_lse is False, softmax_lse outputs an empty Tensor.</li>
                    <li>If return_softmax_lse is True, see <a href="#layout-matching-relationship-table">layout matching relationship table</a> for the output shape.</li>
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

- Example of calling generic_block_sparse_attention_metadata and generic_block_sparse_attention together (TND + PA_BBND)

    ```python
    import math
    import torch
    import torch_npu
    import cann_ops_transformer

    torch_npu.npu.set_device(0)

    B, Q_N, KV_N, Q_S, KV_S, D = 1, 32, 8, 128, 256, 128
    block_x, block_y = 1, 128
    block_size = 128
    top_k = 16

    Q_T = B * Q_S
    num_blocks = math.ceil(KV_S / block_size)
    total_q_blocks = math.ceil(Q_S / block_x)

    q = torch.randn(Q_T, Q_N, D, dtype=torch.float16, device="npu")
    k = torch.randn(num_blocks, block_size, KV_N, D, dtype=torch.float16, device="npu")
    v = torch.randn(num_blocks, block_size, KV_N, D, dtype=torch.float16, device="npu")

    sparse_block_idx = torch.randint(
        0, num_blocks, (KV_N, total_q_blocks, top_k), dtype=torch.int32, device="npu"
    )
    sparse_block_count = torch.full((KV_N, total_q_blocks), top_k, dtype=torch.int32, device="npu")
    cu_seqlens_q = torch.tensor([0, Q_S], dtype=torch.int64, device="npu")
    seqused_kv = torch.tensor([KV_S], dtype=torch.int32, device="npu")
    block_table = torch.arange(num_blocks, dtype=torch.int32, device="npu").view(1, -1)
    block_shape = [block_x, block_y]

    metadata = cann_ops_transformer.ops.generic_block_sparse_attention_metadata(
        sparse_block_idx,
        sparse_block_count,
        Q_N,
        KV_N,
        D,
        block_shape,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seqused_kv,
        max_seqlen_q=Q_S,
        max_seqlen_kv=KV_S,
        is_packed_gqa=True,
        layout_q="TND",
        layout_kv="PA_BBND",
        mask_mode=1,
        quant_mode=0,
        softmax_precision=1,
    )

    attention_out, softmax_lse = cann_ops_transformer.ops.generic_block_sparse_attention(
        q,
        k,
        v,
        sparse_block_idx,
        sparse_block_count,
        block_shape,
        metadata=metadata,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seqused_kv,
        block_table=block_table,
        is_packed_gqa=True,
        layout_q="TND",
        layout_kv="PA_BBND",
        softmax_scale=1.0 / (D ** 0.5),
        mask_mode=1,
        quant_mode=0,
        softmax_precision=1,
        return_softmax_lse=False,
    )
    torch_npu.npu.synchronize()
    assert attention_out.shape == q.shape
    assert attention_out.dtype == q.dtype
    ```
