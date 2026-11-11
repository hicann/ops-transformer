# generic_block_sparse_attention_grad

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T10:14:23.191Z -->

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

  `generic_block_sparse_attention_grad` is a `cann_ops_transformer` extension API based on `TorchNPU`. It calls the `GenericBlockSparseAttentionGrad` operator to perform backward gradient computation for generic block sparse attention. `sparse_block_idx` specifies each Q block/token index selected by each KV block, and `sparse_block_count` specifies the quantity of Q actually retained by each KV block. `dq`/`dk`/`dv` are computed and returned only on selected sparse blocks.

  `generic_block_sparse_attention_grad_metadata` is the metadata generation API of `generic_block_sparse_attention_grad`. It generates `metadata` before the main operator is executed. The `metadata` records the AICore task splitting and load balancing results. The main operator must be passed this `metadata` to optimize scheduling. The typical call flow is as follows:

  1. Prepare inputs such as `q`, `k`, `v`, `dout`, `attn_out`, `softmax_lse`, `sparse_block_idx`, `sparse_block_count`, `cu_seqlens_q`, and `cu_seqlens_kv`.
  2. Call `generic_block_sparse_attention_grad_metadata` to generate `metadata`.
  3. Call `generic_block_sparse_attention_grad` and pass the `metadata` obtained in the previous step to the main operator.
- **Formula**:

  The sparse block size is $blockShapeX \times blockShapeY$. After selecting Q–KV block pairs for computation based on `sparse_block_idx`/`sparse_block_count`:

  $$
  P = SimpleSoftmax(Mask(Q @ selectedK^{T} \cdot scale), lse)
  $$

  $$
  dP = dO @ selectedV^{T}
  $$

  $$
  dS = P \odot (dP - SoftmaxGrad(dO, O))
  $$

  $$
  dQ = dS @ selectedK \cdot scale
  $$

  $$
  dK = dS^{T} @ Q \cdot scale
  $$

  $$
  dV = P^{T} @ dO
  $$

  where $scale$ is specified by `softmax_scale`, and the recommended value is $D^{-0.5}$.

> [!NOTE]
> The q, k, and v data layout formats can be interpreted from multiple dimensions, where B (Batch) indicates the input sample batch size, S (Seq-Length) indicates the input sample sequence length, N (Head-Num) indicates the number of heads, and D (Head-Dim) indicates headdim, the smallest unit size of the hidden layer.
> In TND, when N is the headNum of q, it is denoted as N1 (Q_N); the headNum of k/v is denoted as N2 (KV_N). T1 indicates the cumulative sum of sequence lengths of all q Batch input samples; T2 indicates the cumulative sum of sequence lengths of all kv Batch input samples.
> $J = \lceil max\_seqlen\_kv / blockShapeY \rceil$. Unlike the forward `generic_block_sparse_attention`, the `sparse_block_idx`/`sparse_block_count` of this backward API is in the **KV→Q** direction (Q tokens selected by each KV block), rather than the forward Q→KV direction.

## Function Prototype

Before calling the `generic_block_sparse_attention_grad` API, call the prerequisite API `generic_block_sparse_attention_grad_metadata` first to calculate load balancing and core splitting information.

```python
cann_ops_transformer.generic_block_sparse_attention_grad_metadata(
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
    max_seqlen_q=None,
    max_seqlen_kv=None,
    is_packed_gqa=True,
    layout_q="TND",
    layout_kv="TND",
    mask_mode=MaskMode.CAUSAL,
    softmax_precision=0,
    win_left=-1,
    win_right=-1,
) -> Tensor
```

```python
cann_ops_transformer.generic_block_sparse_attention_grad(
    q,
    k,
    v,
    dout,
    attn_out,
    softmax_lse,
    sparse_block_idx,
    sparse_block_count,
    block_shape,
    *,
    metadata=None,
    attn_mask=None,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    is_packed_gqa=True,
    layout_q="TND",
    layout_kv="TND",
    softmax_scale=1.0,
    mask_mode=MaskMode.CAUSAL,
    softmax_precision=0,
    win_left=-1,
    win_right=-1,
) -> (Tensor, Tensor, Tensor)
```

## Enumerations

In the Python API, `mask_mode` supports passing an `IntEnum` enumeration or the corresponding int value. The enumeration is defined in `cann_ops_transformer.ops.generic_block_sparse_attention_grad`:

### mask_mode Enumeration

| Enumeration Name | Value | Meaning |
| :--------- | :-: | :-------------------------------- |
| `CAUSAL` | 1 | Causal mode (currently the default and the only supported mode). |

> [!NOTE]
> The enumeration is of type `IntEnum` and can be directly passed to the underlying operator as an int. The API accepts an enumeration, the corresponding int value, or the enumeration name string. The current version only supports `mask_mode = 1` (`CAUSAL`).

## Parameters

### generic_block_sparse_attention_grad_metadata

| Name             | Type | Type | Description                                                                                                | Data Type | Data Format | Dimensions                                                |
| :----------------- | :----------- | :-------- | :-------------------------------------------------------------------------------------------------- | :------- | :------- | :-------------------------------------------------------- |
| sparse_block_idx   | Tensor       | Required      | Sparse block index array, specifying the Q block/token indices selected for each KV block.                                                     | int32    | ND       | TND: `(B, N2, J, maxS1)`; BNSD/BSND: `(B, N2, J, S1)` |
| sparse_block_count | Tensor       | Required      | Specifies the actual quantity of Q blocks selected for each KV block.                                                                         | int32    | ND       | `(B, N2, J)`                                            |
| num_heads_q        | int          | Required      | Number of query heads (N1).                                                                                  | int32    | -        | -                                                         |
| num_heads_kv       | int          | Required      | Number of Key/Value heads (N2).                                                                              | int32    | -        | -                                                         |
| head_dim           | int          | Required      | Dimension of each attention head, currently fixed to 128.                                                                     | int32    | -        | -                                                         |
| block_shape        | list[int]    | Required      | Sparse block shape [block_x, block_y]: block_x only supports 1; block_y must be >=128 and a multiple of 64.                    | int64    | -        | Length 2                                                   |
| cu_seqlens_q       | Tensor       | Optional      | Cumulative sequence length of q. The first element must be 0.                                                                    | int64    | ND       | `(B+1,)`                                                |
| cu_seqlens_kv      | Tensor       | Optional      | Cumulative sequence length of k/value. The first element must be 0.                                                              | int64    | ND       | `(B+1,)`                                                |
| seqused_q          | Tensor       | Optional      | Actual sequence length of q used in each batch.                                                                        | int32    | ND       | `(B,)`                                                  |
| seqused_kv         | Tensor       | Optional      | Actual sequence length of kv used in each batch.                                                                       | int32    | ND       | `(B,)`                                                  |
| max_seqlen_q       | int          | Optional      | Maximum sequence length of q across all batches; inferred by the extension if omitted (preferring `seqused_q`/`cu_seqlens_q`, otherwise taking the last dimension of `sparse_block_idx`); must be ≥0 if explicitly passed. | int32    | -        | -                                                         |
| max_seqlen_kv      | int          | Optional      | Maximum sequence length of kv across all batches, used to calculate $J$; inferred by the extension if omitted (preferring `seqused_kv`/`cu_seqlens_kv`, otherwise `J * block_y`); must be ≥0 if explicitly passed. | int32    | -        | -                                                         |
| is_packed_gqa      | bool         | Optional      | Whether qHeads in the same group share a sparse pattern. Currently only supports `True`. Defaults to `True`.                               | bool     | -        | -                                                         |
| layout_q           | string       | Optional      | Layout of q. Supports `"TND"`/`"BNSD"`/`"BSND"`. Defaults to `"TND"`.                                           | string   | -        | -                                                         |
| layout_kv          | string       | Optional      | Layout of k/value. Must be consistent with `layout_q`. Defaults to `"TND"`.                                                    | string   | -        | -                                                         |
| mask_mode          | int/MaskMode | Optional      | Mask mode. Supports passing an enumeration or the corresponding int value. See "mask_mode enumeration" for the enumeration definition. Currently only supports 1 (`CAUSAL`); defaults to 1. | int32    | -        | -                                                         |
| softmax_precision  | int          | Optional      | Softmax precision level. Currently only supports 0. Defaults to 0.                                                                   | int32    | -        | -                                                         |
| win_left           | int          | Optional      | Number of tokens included forward in the sliding window. Must be -1 when disabled. Defaults to -1.                                                       | int32    | -        | -                                                         |
| win_right          | int          | Optional      | Number of tokens included backward in the sliding window. Must be -1 when disabled. Defaults to -1.                                                      | int32    | -        | -                                                         |

### generic_block_sparse_attention_grad

| Name | Type | Required/Optional | Description | Data Type | Data Format | Dimension |
| :----------------- | :----------- | :-------- | :------------------------------------------------------------------------------------ | :--------------- | :------- | :----------------------------------------------------------------------- |
| q | Tensor | Required | $Q$ in the formula. | bfloat16/float16 | ND | TND: `(T1, N1, D)`; BNSD: `(B, N1, S1, D)`; BSND: `(B, S1, N1, D)` |
| k | Tensor | Required | $K$ in the formula. | bfloat16/float16 | ND | TND: `(T2, N2, D)`; BNSD: `(B, N2, S2, D)`; BSND: `(B, S2, N2, D)` |
| v | Tensor | Required | $V$ in the formula, with the same shape as k. | bfloat16/float16 | ND | consistent with k |
| dout | Tensor | Required | Gradient of the attention output $dO$. | bfloat16/float16 | ND | consistent with q |
| attn_out | Tensor | Required | Forward attention output $O$. | bfloat16/float16 | ND | consistent with q |
| softmax_lse | Tensor | Required | Softmax LSE saved in the forward pass. | float32 | ND | TND: `(T1,N1,1)`; BNSD: `(B,N1,S1,1)`; BSND: `(B,S1,N1,1)` or `(B,N1,S1,1)`. See Constraints for details. |
| sparse_block_idx | Tensor | Required | Sparse block index array (KV→Q). | int32 | ND | consistent with the metadata API |
| sparse_block_count | Tensor | Required | Quantity of Q actually selected for each KV block. | int32 | ND | consistent with the metadata API |
| metadata | Tensor | Optional | Task splitting result generated by `generic_block_sparse_attention_grad_metadata`. If passed in, scheduling can be optimized. Must be passed in as a keyword argument. | int32 | ND | `(metaSize,)`. See Return Values |
| block_shape | list[int] | Required | Sparse block shape [block_x, block_y]: block_x only supports 1; block_y must be >=128 and a multiple of 64. | int64 | - | length 2 |
| attn_mask | Tensor | Optional | Mask matrix. Currently not supported. Pass `None`. | bool | ND | - |
| cu_seqlens_q | Tensor | Optional | Cumulative sequence length of q. Must be passed when `layout_q="TND"`. | int64 | ND | `(B+1,)` |
| cu_seqlens_kv | Tensor | Optional | Cumulative sequence length of kv. Must be passed when `layout_kv="TND"`. | int64 | ND | `(B+1,)` |
| seqused_q | Tensor | Optional | Actual sequence length of q used in each batch. | int32 | ND | `(B,)` |
| seqused_kv | Tensor | Optional | Actual sequence length of kv used in each batch. | int32 | ND | `(B,)` |
| is_packed_gqa | bool | Optional | Packed GQA switch. Currently only supports `True`; defaults to `True`. | bool | - | - |
| layout_q | string | Optional | Layout of q. Supports `"TND"`/`"BNSD"`/`"BSND"`; defaults to `"TND"`. | string | - | - |
| layout_kv | string | Optional | Layout of k/value. Must be consistent with `layout_q`; defaults to `"TND"`. | string | - | - |
| softmax_scale | float | Optional | Scaling factor. Recommended value $1/\sqrt{D}$; defaults to 1.0. | float32 | - | - |
| mask_mode | int/MaskMode | Optional | Mask mode. Supports passing an enumeration or the corresponding int value. See the mask_mode enumeration section for the enumeration definition. Currently only supports 1; defaults to 1. | int32 | - | - |
| softmax_precision | int | Optional | Softmax precision level. Currently only supports 0; defaults to 0. | int32 | - | - |
| win_left | int | Optional | Number of tokens included forward by the sliding window. Must be -1 when disabled; defaults to -1. | int32 | - | - |
| win_right | int | Optional | Number of tokens included backward by the sliding window. Must be -1 when disabled; defaults to -1. | int32 | - | - |

## Return Values

### generic_block_sparse_attention_grad_metadata

| Name | Type | Required/Optional | Description | Data Type | Data Format | Dimension |
| :------- | :------- | :-------- | :----------------------- | :------- | :------- | :-------------- |
| metadata | Tensor   | Required | Core partitioning and load balancing data of the main operator | int32    | ND       | `(metaSize,)` |

Where:
<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT products</term>

    $$
    metaSize = 80 + B \times N1 \times J \times 4,\quad J = \lceil max\_seqlen\_kv / blockShapeY \rceil
    $$

<!-- end id7 -->

<!-- npu="A3,910b" id8 -->
- <term>Atlas A3 products</term>, <term>Atlas A2 products</term>:

    $$
    metaSize = 198
    $$

<!-- end id8 -->

### generic_block_sparse_attention_grad

| Name | Type | Required/Optional | Description | Data Type | Data Format | Dimension |
| :----- | :------- | :-------- | :------ | :--------------- | :------- | :------ |
| dq     | Tensor   | Required      | Gradient of q. | bfloat16/float16 | ND       | Consistent with q. |
| dk     | Tensor   | Required      | Gradient of k. | bfloat16/float16 | ND       | Consistent with k. |
| dv     | Tensor   | Required      | Gradient of v. | bfloat16/float16 | ND       | Consistent with v. |

**NOTE**

- The dtype of `dq`/`dk`/`dv` must be consistent with the corresponding inputs `q`/`k`/`v`.
- The length of `metadata` must satisfy `shape[0] ≥ 80 + B × num_heads_q × J × 4`; the upper bound of the number of tasks is $B \times N1 \times J \le 1048576$.

## Constraints

- Deterministic computation: `generic_block_sparse_attention_grad` defaults to a non-deterministic implementation. Deterministic implementation is currently not supported. The deterministic computation configuration does not take effect.
- Parameters `cu_seqlens_q`, `cu_seqlens_kv`, `seqused_q`, `seqused_kv`, `sparse_block_idx`, and `sparse_block_count` are tensors. Because the operator cannot obtain the specific tensor values in the Tiling phase, tiling does not validate the values. You must ensure their correctness. If invalid values are passed for the preceding parameters, undefined behavior occurs (such as precision issues or program crashes caused by invalid memory access).
- The input parameters of `generic_block_sparse_attention_grad_metadata` and `generic_block_sparse_attention_grad` must be consistent during the call. Because the operator is divided into two APIs for phased calls, the operator cannot validate the parameters itself. You must ensure their correctness. If the parameters passed to the APIs are inconsistent, undefined behavior occurs (such as precision issues or program crashes caused by invalid memory access).
- Successfully call `generic_block_sparse_attention_grad_metadata` to generate `metadata`, and then pass it to the main operator;
- Softmax LSE: TND is `(T1,N1,1)`, BNSD is `(B,N1,S1,1)`; for BSND, GSAG on A5 requires `(B,N1,S1,1)`, GSAG on A2/A3 uses `(B,S1,N1,1)`. For QSAG (layout_sparse_pattern=0), LSE follows the forward output: `(T1,N1)` or `(T1,N1,1)` for TND, and `(B,N1,S1)` or `(B,N1,S1,1)` for both BNSD and BSND.
- `softmax_scale` must be finite in float32 (not NaN/Inf, and still finite after cast to float).
- In the Torch extension, `max_seqlen_q`/`max_seqlen_kv` can be omitted: the extension infers a value that is `≥0` before calling the underlying API and then passes it to the host; when explicitly passed, the values must be `≥0`, and `J=ceilDiv(max_seqlen_kv, block_y)` and `maxS1≥max_seqlen_q` must still be satisfied.

### Feature Parameter Group

|   Feature Parameter Group   |    Parameter Field Name    |    Field Grouping    |   Field Type   |
| :------------: | :----------------: | :-------------: | :----------: |
|   Common parameter group   |         q         |      INPUT      |    Tensor    |
|                |         k         |      INPUT      |    Tensor    |
|                |         v         |      INPUT      |    Tensor    |
|                |        dout        |      INPUT      |    Tensor    |
|                |      attn_out      |      INPUT      |    Tensor    |
|                |    softmax_lse    |      INPUT      |    Tensor    |
|                |  sparse_block_idx  |      INPUT      |    Tensor    |
|                | sparse_block_count |      INPUT      |    Tensor    |
|                |      metadata      |      INPUT      |    Tensor    |
|                |    block_shape    | ATTR(REQUIRED) |    int[]    |
|                |   softmax_scale   | ATTR(OPTIONAL) |    float    |
|                |      layout_q      | ATTR(OPTIONAL) |    string    |
|                |     layout_kv     | ATTR(OPTIONAL) |    string    |
|                |         dq         |     OUTPUT     |    Tensor    |
|                |         dk         |     OUTPUT     |    Tensor    |
|                |         dv         |     OUTPUT     |    Tensor    |
| Metadata parameter group |    max_seqlen_q    | ATTR(OPTIONAL) |     int     |
|                |   max_seqlen_kv   | ATTR(OPTIONAL) |     int     |
|                |    num_heads_q    | ATTR(REQUIRED) |     int     |
|                |    num_heads_kv    | ATTR(REQUIRED) |     int     |
|                |      head_dim      | ATTR(REQUIRED) |     int     |
|                |      metadata      |     OUTPUT     |    Tensor    |
|   Mask parameter group   |     mask_mode     | ATTR(OPTIONAL) | int/MaskMode |
|                |      win_left      | ATTR(OPTIONAL) |     int     |
|                |     win_right     | ATTR(OPTIONAL) |     int     |
|                |     attn_mask     | INPUT(OPTIONAL) |    Tensor    |
| SeqLens parameter group |    cu_seqlens_q    | INPUT(OPTIONAL) |    Tensor    |
|                |   cu_seqlens_kv   | INPUT(OPTIONAL) |    Tensor    |
|                |     seqused_q     | INPUT(OPTIONAL) |    Tensor    |
|                |     seqused_kv     | INPUT(OPTIONAL) |    Tensor    |
| Softmax parameter group | softmax_precision | ATTR(OPTIONAL) |     int     |
|                |   is_packed_gqa   | ATTR(OPTIONAL) |     bool     |

### Baseline Information

In data constraints, common fields are defined as follows:

|   Naming   |                          Meaning                          |
| :-------: | :-----------------------------------------------------: |
|     B     |               Batch, indicating the batch size of input samples.               |
| N1 / Q_N |                   Number of heads of the input q tensor.                   |
| N2 / KV_N |                 Number of heads of the input key/v tensor.                 |
| S1 / Q_S |                      Sequence length of input q.                      |
| S2 / KV_S |                  Sequence length of input key/value.                  |
|    T1    |             Cumulative sum of sequence lengths of all batches in input q.             |
|    T2    |             Cumulative sum of sequence lengths of all batches in input kv.             |
|     D     |                  headdim, currently fixed to 128.                  |
|     J     | $\lceil max\_kv\_seqlen / blockShapeY \rceil$, number of KV blocks. |
|   maxS1   | Capacity of the last dimension of `sparse_block_idx`, must be ≥ `max_seqlen_q`. |

### Parameter Group Constraints

#### Common Parameter Group

- Handling of empty input parameters:

  - An empty tensor refers to a required input or output whose shape size is 0, that is, any axis is 0.
  - All use cases that trigger an empty tensor are intercepted and an error is reported.
- q, k, v, dout, attn_out, dq, dk, dv validation:

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
                <li>TND → (T1, N1, D); BNSD → (B, N1, S1, D); BSND → (B, S1, N1, D)</li>
            </ul>
        </td>
        <td rowspan="5">Must exist.</td>
        <td rowspan="5">            <ul>
                <li>The data types of q, k, v, dout, attn_out, dq, dk, and dv must be the same.</li>
                <li>The shapes of dout, attn_out, and dq are consistent with q.</li>
                <li>The shapes of k, v, dk, and dv must be the same.</li>
                <li>layout_q must be consistent with layout_kv. TND/BNSD/BSND are supported.</li>
            </ul>
        </td>
        <td rowspan="5">            Axis validation:
            <ul>
                <li>Currently, D supports only 128.</li>
                <li>Value range of N1 and N2: [1, 128]</li>
                <li>N1 % N2 == 0</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>k / v</td>
        <td>            <ul>
                <li>tensor_type supports bfloat16 and float16.</li>
                <li>TND → (T2, N2, D); BNSD → (B, N2, S2, D); BSND → (B, S2, N2, D)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>dout / attn_out</td>
        <td>            <ul>
                <li>dtype consistent with q.</li>
                <li>shape consistent with q.</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>dq</td>
        <td>shape/dtype consistent with q.</td>
    </tr>
    <tr>
        <td>dk / dv</td>
        <td>shape/dtype consistent with k.</td>
    </tr>
    </tbody>
    </table>
- lse validation:

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
    <th>Single-parameter Validation</th>
    <th>Existence Validation</th>
    <th>Consistency Validation</th>
    <th>feature Cross-Validation</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td>softmax_lse</td>
        <td>            <ul>
                <li>tensor_type only supports float32</li>
                <li>TND: (T1,N1,1); BNSD: (B,N1,S1,1); BSND: (B,S1,N1,1) or (B,N1,S1,1). See Constraints for details.</li>
            </ul>
        </td>
        <td>Must exist.</td>
        <td>For BSND axis order, see Constraints.</td>
        <td>None.</td>
    </tr>
    </tbody>
    </table>
- Validation of sparse_block_idx, sparse_block_count, and block_shape:

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
    <th>Single-parameter Validation</th>
    <th>Existence Validation</th>
    <th>Consistency Validation</th>
    <th>Feature Cross-Validation</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td>sparse_block_idx</td>
        <td>            <ul>
                <li>tensor_type only supports int32.</li>
                <li>TND: (B, N2, J, maxS1); BNSD/BSND: (B, N2, J, S1)</li>
            </ul>
        </td>
        <td rowspan="3">Must exist.</td>
        <td rowspan="3">            <ul>
                <li>The metadata API and the main operator must be passed the same sparse_block_idx, sparse_block_count, and block_shape.</li>
                <li>J must equal ceilDiv(max_seqlen_kv, block_y).</li>
                <li>maxS1 / S1 must be ≥ max_seqlen_q and ≥ the maximum value of all elements in sparse_block_count.</li>
            </ul>
        </td>
        <td rowspan="3">            <ul>
                <li>block_shape: block_x=1; block_y >= 128 and a multiple of 64</li>
                <li>is_packed_gqa currently only supports True.</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>sparse_block_count</td>
        <td>            <ul>
                <li>tensor_type only supports int32.</li>
                <li>shape: (B, N2, J)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>block_shape</td>
        <td>            <ul>
                <li>An int list of length 2</li>
            </ul>
        </td>
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
                <th>Single-parameter Validation</th>
                <th>Existence Validation</th>
                <th>Consistency Validation</th>
                <th>Feature Cross-Validation</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>metadata</td>
                <td>
                    <ul>
                        <li>`layout_sparse_pattern=1`: tensor_type only supports int32; supported lengths differ by product, see notes below</li>
                    </ul>
                </td>
                <td>`layout_sparse_pattern=1` must exist; `layout_sparse_pattern=0` must be None</td>
                <td>`layout_sparse_pattern=1` must be the output of generic_block_sparse_attention_grad_metadata</td>
                <td>`layout_sparse_pattern=1` upper bound of task quantity: B×N1×J ≤ 1048576</td>
            </tr>
        </tbody>
    </table>

<!-- npu="A3,910b" id9 -->
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>:
    metadata length ≥ 198.
<!-- end id9 -->

<!-- npu="950" id10 -->
  - <term>Ascend 950PR&950DT products</term>: metadata length ≥ 80+B×N1×J×4.
<!-- end id10 -->

#### Mask parameter group

mask_mode parameter description:

- `MaskMode.CAUSAL` (1): Supported in both modes; default value.
- `MaskMode.NO_MASK` (0): Supported only when `layout_sparse_pattern=0`.
- win_left / win_right: must be -1 when disabled; attn_mask should currently be set to `None`

#### SeqLens Parameter Group

- `cu_seqlens_q` is required when `layout_q="TND"`; `cu_seqlens_kv` is required when `layout_kv="TND"`.
- For BNSD/BSND layouts, pass `None` for the cumulative sequence lengths described above.
- The first element of `cu_seqlens_q`/`cu_seqlens_kv` must be 0, and the last elements equal T1/T2 respectively.

#### Softmax Parameter Group

- `softmax_precision` currently only supports 0.
- `is_packed_gqa` currently only supports `True`.

## Example

- generic_block_sparse_attention_grad_metadata + generic_block_sparse_attention_grad joint call example (BNSD)

  ```python
  import math
  import torch
  import torch_npu
  import cann_ops_transformer
  from cann_ops_transformer.ops.generic_block_sparse_attention_grad import MaskMode

  torch_npu.npu.set_device(0)

  B, N1, N2, S1, S2, D = 1, 1, 1, 128, 128, 128
  block_x, block_y = 1, 128
  J = math.ceil(S2 / block_y)
  scale = 1.0 / math.sqrt(D)
  block_shape = [block_x, block_y]

  q = torch.randn(B, N1, S1, D, dtype=torch.float16, device="npu")
  k = torch.randn(B, N2, S2, D, dtype=torch.float16, device="npu")
  v = torch.randn(B, N2, S2, D, dtype=torch.float16, device="npu")
  dout = torch.randn(B, N1, S1, D, dtype=torch.float16, device="npu")
  attn_out = torch.randn(B, N1, S1, D, dtype=torch.float16, device="npu")
  softmax_lse = torch.randn(B, N1, S1, 1, dtype=torch.float32, device="npu")

  sparse_block_idx = torch.full((B, N2, J, S1), -1, dtype=torch.int32, device="npu")
  sparse_block_count = torch.zeros((B, N2, J), dtype=torch.int32, device="npu")
  for qi in range(S1):
      sparse_block_idx[0, 0, 0, qi] = qi
  sparse_block_count[0, 0, 0] = S1

  metadata = cann_ops_transformer.generic_block_sparse_attention_grad_metadata(
      sparse_block_idx,
      sparse_block_count,
      N1,
      N2,
      D,
      block_shape,
      is_packed_gqa=True,
      layout_q="BNSD",
      layout_kv="BNSD",
      mask_mode=MaskMode.CAUSAL,
      softmax_precision=0,
      win_left=-1,
      win_right=-1,
  )

  dq, dk, dv = cann_ops_transformer.generic_block_sparse_attention_grad(
      q,
      k,
      v,
      dout,
      attn_out,
      softmax_lse,
      sparse_block_idx,
      sparse_block_count,
      block_shape,
      metadata=metadata,
      attn_mask=None,
      is_packed_gqa=True,
      layout_q="BNSD",
      layout_kv="BNSD",
      softmax_scale=scale,
      mask_mode=MaskMode.CAUSAL,
      softmax_precision=0,
      win_left=-1,
      win_right=-1,
  )
  torch_npu.npu.synchronize()
  assert dq.shape == q.shape
  assert dk.shape == k.shape
  assert dv.shape == v.shape
  assert dq.dtype == q.dtype
  ```
