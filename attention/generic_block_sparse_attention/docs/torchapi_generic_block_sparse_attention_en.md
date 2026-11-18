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

  `generic_block_sparse_attention_metadata` is the metadata generation API of `generic_block_sparse_attention`, used to generate metadata before the main operator is executed. The metadata records the task splitting result of AICore/AIVCore. The main operator requires matching metadata. The Python signature defaults to `None`, but callers must explicitly pass the result of the metadata API. The typical call flow is as follows:

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
    layout_q="TND",
    layout_kv="PA_BBND",
    layout_sparse_pattern=4,
    mask_mode=1,
    quant_mode=0,
    softmax_precision=1,
    win_left=-1,
    win_right=-1,
    residual_block_mode=0,
    is_consistent_topk=False,
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
    layout_q="TND",
    layout_kv="PA_BBND",
    layout_sparse_pattern=4,
    softmax_scale=0.0,
    mask_mode=1,
    quant_mode=0,
    dst_type_max=0.0,
    softmax_precision=1,
    win_left=-1,
    win_right=-1,
    return_softmax_lse=False,
    residual_block_mode=0,
    is_consistent_topk=False,
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
| `FP8_E4M3_STATIC_CAST_P` | 5 | FP8_E4M3 static cast P on Ascend 950PR&950DT products. |
| `MXFP8_E4M3_PER_CHANNEL_V` | 6 | A8C8: Q/K use one E8M0 scale per 32 elements along D; V is quantized per channel. Explicitly set softmax_precision=0. |
| `INT8_FULL_QUANT` | 7 | INT8 full quantization on <term>Atlas A2 products</term> and <term>Atlas A3 products</term>, with all four external scales required. |
| `Q_FP8_PER_TENSOR_KV_FP4_PER_TOKEN_PER_GROUP` | 8 | FP8 E4M3FN Q, packed FP4 E2M1 K/V, and BF16 K/V scales. |

### mask_mode Enumeration

| Enumeration Name | Value | Meaning |
| :--- | :---: | :--- |
| `NO_MASK` | 0 | No mask |
| `CAUSAL` | 1 | Causal mode (default value) |
| `WINDOW` | 2 | Window mode |
| `UNION` | 3 | Union exactMask mode |

> [!NOTE]
>
> Pass an `IntEnum` or its corresponding integer. `quant_mode` supports 0 (NO_QUANT), 5 (FP8), 6 (A8C8), 7 (<term>Atlas A2 products</term> and <term>Atlas A3 products</term> INT8), and 8 (FP8 Q with FP4 K/V). Modes 5, 6 and 8 apply to <term>Ascend 950PR&950DT products</term>. INT8 supports `mask_mode=0/1` and requires `softmax_precision=0`. See the product-specific constraints for each mode.

## Parameters

### generic_block_sparse_attention_metadata

| Name | Type | Required/Optional | Description | Data Type | Data Format | Dimension |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| sparse_block_idx | Tensor | Required | KV block indexes selected by each Q block. | int32 | ND | (KV_N, totalQBlocks, topK) |
| sparse_block_count | Tensor | Required | Number of KV blocks actually retained for each Q block. | int32 | ND | (KV_N, totalQBlocks) |
| num_heads_q | int | Required | Number of Query heads. A4C4: num_heads_q/num_heads_kv is in [1,16]. | int32 | - | - |
| num_heads_kv | int | Required | Number of Key/Value heads. | int32 | - | - |
| head_dim | int | Required | Dimension of each attention head. A4C4 uses logical D=192 or 256. | int32 | - | - |
| block_shape | list[int] | Required | Sparse block shape `[block_x, block_y]`. A4C4 uses [1,Y], Y in {4,8,16,32,64,128}. | int64 | - | Length of 2 |
| cu_seqlens_q | Tensor | Conditional | Required for TND Q. Cumulative sequence lengths, used to process variable-length sequences. The first element must be 0. | int64 | ND | (B+1,) |
| cu_seqlens_kv | Tensor | Conditional | Required for TND KV. Pass `None` for paged KV. Cumulative sequence lengths, used to process variable-length sequences. The first element must be 0. | int64 | ND | (B+1,) |
| seqused_q | Tensor | Conditional | Required for BSND/BNSD. For TND Q, this is optional with non-quantized inputs or `quant_mode=8` on <term>Ascend 950PR&950DT products</term>. Pass `None` for INT8 TND. | int32 | ND | (B,) |
| seqused_kv | Tensor | Conditional | Actual KV length for each batch. Required for BSND, BNSD, PA_BBND, and PA_BNBD KV. Optional for TND KV. | int32 | ND | (B,) |
| max_seqlen_q | int | Optional | Specifies the upper limit of the query q sequence length. | int32 | - | - |
| max_seqlen_kv | int | Optional | Specifies the upper limit of the key k and value v sequence lengths. | int32 | - | - |
| layout_q | string | Optional | Defines the layout format of the input query tensor. | string | - | - |
| layout_kv | string | Optional | Defines the layout format of the input key/value tensor. | string | - | - |
| layout_sparse_pattern | int | Optional | Layout of `sparse_block_idx` and `sparse_block_count`. The supported value is `4`. | int64 | - | - |
| mask_mode | int/MaskMode | Optional | Mask mode. Supports passing an enumeration or the corresponding int value. For the enumeration definition, see "mask_mode enumeration". | int32 | - | - |
| quant_mode | int/QuantMode | Optional | Quantization mode. Supports passing an enumeration or the corresponding int value. For the enumeration definition, see "quant_mode enumeration". | int32 | - | - |
| softmax_precision | int | Optional | Softmax precision mode. | int32 | - | - |
| win_left | int | Optional | Left boundary of the window. | int32 | - | - |
| win_right | int | Optional | Right boundary of the window. | int32 | - | - |
| residual_block_mode | int | Optional | Controls how the final partial KV block is handled. Defaults to `0`. | int64 | - | - |
| is_consistent_topk | bool | Optional | Indicates whether query blocks in the same batch and head share the same maximum selected KV block count. Defaults to `False`. | bool | - | - |

### generic_block_sparse_attention

D in the table denotes the logical head dimension. A4C4 Torch tensors use a physical final dimension D/2; see [A4C4](#a4c4quantmode4) for packed storage and scale conventions.

| Name | Type | Required/Optional | Description | Data Type | Data Format | Dimension |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| q | Tensor | Required | Query in the formula; mode 4 uses packed uint8 with physical final dimension D/2. | bfloat16/float16/float8_e4m3fn/int8/uint8 | ND | TND: (Q_T, Q_N, D); BSND: (B, Sq, Q_N, D); BNSD: (B, Q_N, Sq, D) |
| k | Tensor | Required | Key in the formula. The last Torch dimension `D_kv` is logical `D/2` for `quant_mode=4/8` and `D` for other modes. For `quant_mode=8`, K and V must have the same data type. Use `float4_e2m1fn_x2` for both, or `uint8` containing packed FP4 data for both. | bfloat16/float16/float8_e4m3fn/int8/float4_e2m1fn_x2/uint8 | ND | PA_BBND: (num_blocks, block_size, KV_N, D_kv); PA_BNBD: (num_blocks, KV_N, block_size, D_kv); TND: (KV_T, KV_N, D_kv); BSND: (B, Sk, KV_N, D_kv); BNSD: (B, KV_N, Sk, D_kv) |
| v | Tensor | Required | Value in the formula, with the same `D_kv` and supported data types as k. | bfloat16/float16/float8_e4m3fn/int8/float4_e2m1fn_x2/uint8 | ND | Same as k |
| sparse_block_idx | Tensor | Required | KV block index selected by each Q block | int32 | ND | (KV_N, totalQBlocks, topK) |
| sparse_block_count | Tensor | Required | Number of KV blocks actually retained for each Q block | int32 | ND | (KV_N, totalQBlocks) |
| block_shape | list[int] | Required | Sparse block shape `[block_x, block_y]` A4C4 uses [1,Y], Y in {4,8,16,32,64,128}. | int64 | - | length of 2 |
| metadata | Tensor | Required | Task split result from `generic_block_sparse_attention_metadata` with matching layout, inputs, and attributes. | int32 | ND | (1024,) |
| attn_mask | Tensor | Optional | Must be None when mask_mode=0/1; must be INT32 exactMask when mask_mode=3 (Union) | bool/int32 | ND | Union: (KV_N, G, U×block_y) |
| q_dequant_scale | Tensor | Required for INT8; conditional for other modes; required for A4C4 | query dequantization scale factor; <term>Atlas A2 products</term> and <term>Atlas A3 products</term> INT8 (quant_mode=7) requires a contiguous, positive, finite float32 Tensor A4C4 (mode 4) requires E8M0 or uint8 encoding. | float32/float8_e8m0/uint8 | ND | INT8: TND: [Q_T,Q_N]; BSND: [B,Q_S,Q_N]; BNSD: [B,Q_N,Q_S] <br>A4C4: [scale shapes](#a4c4quantmode4). |
| k_dequant_scale | Tensor | Required for INT8 and `quant_mode=8`; required for A4C4 | key dequantization scale factor; <term>Atlas A2 products</term> and <term>Atlas A3 products</term> INT8 (quant_mode=7) requires a contiguous, positive, finite float32 Tensor A4C4 (mode 4) requires E8M0 or uint8 encoding. | float32/bfloat16/uint8 | ND | INT8: PA_BBND/PA_BNBD: [num_blocks,KV_N,128]; TND: [KV_N,KV_T]; BSND/BNSD: [B,KV_N,KV_S] <br>For `quant_mode=8`, use the logical K/V shape with D replaced by 8. <br>A4C4: [scale shapes](#a4c4quantmode4). |
| v_dequant_scale | Tensor | Required for INT8 and `quant_mode=8`; required for A4C4 | value dequantization scale factor; <term>Atlas A2 products</term> and <term>Atlas A3 products</term> INT8 (quant_mode=7) requires a contiguous, positive, finite float32 Tensor A4C4 (mode 4) requires E8M0 or uint8 encoding. | float32/bfloat16/uint8 | ND | INT8: [KV_N,D] <br>For `quant_mode=8`, use the logical K/V shape with D replaced by 8. <br>A4C4: [scale shapes](#a4c4quantmode4). |
| p_quant_scale | Tensor | Required for INT8; conditional for other modes |P quantization scale factor; <term>Atlas A2 products</term> and <term>Atlas A3 products</term> INT8 (quant_mode=7) requires a contiguous, positive, finite float32 Tensor | float32 | ND | INT8: [1] |
| cu_seqlens_q | Tensor | Conditional | Required for TND Q. Cumulative sequence length, used to process variable-length sequences. The first element must be 0 | int64 | ND | (B+1,) |
| cu_seqlens_kv | Tensor | Conditional | Required for TND KV. Pass `None` for paged KV. Cumulative sequence length, used to process variable-length sequences. The first element must be 0 | int64 | ND | (B+1,) |
| seqused_q | Tensor | Conditional | Required for BSND/BNSD. For TND Q, this is optional with non-quantized inputs or `quant_mode=8` on <term>Ascend 950PR&950DT products</term>. Pass `None` for INT8 TND. | int32 | ND | (B,) |
| seqused_kv | Tensor | Conditional | Actual KV length for each batch. Required for BSND, BNSD, PA_BBND, and PA_BNBD KV. Optional for TND KV. | int32 | ND | (B,) |
| block_table | Tensor | Optional | Block index mapping used in Paged Attention computation | int32 | ND | (B, max_num_blocks_per_seq) |
| layout_q | string | Optional | Defines the layout format of the input query tensor | string | - | - |
| layout_kv | string | Optional | Defines the layout format of the input key/value tensor | string | - | - |
| layout_sparse_pattern | int | Optional | Layout of `sparse_block_idx` and `sparse_block_count`. The supported value is `4`. | int64 | - | - |
| softmax_scale | float | Optional | Scale factor that can be explicitly set to override the default computation | float32 | - | - |
| mask_mode | int/MaskMode | Optional | Mask mode. Supports passing an enumeration or the corresponding int value. See "mask_mode enumeration" for the enumeration definition | int32 | - | - |
| quant_mode | int/QuantMode | Optional | Quantization mode. Supports passing an enumeration or the corresponding int value. See "quant_mode enumeration" for the enumeration definition | int32 | - | - |
| dst_type_max | float | Optional | Mode 4: 0 selects CX=6; explicit values must be finite and in [6,12]. Other modes require 0. | float32 | - | - |
| softmax_precision | int | Optional | Softmax precision mode | int32 | - | - |
| win_left | int | Optional | Window left boundary | int32 | - | - |
| win_right | int | Optional | Window right boundary | int32 | - | - |
| return_softmax_lse | bool | Optional | Whether to obtain the LSE result of softmax | bool | - | - |
| residual_block_mode | int | Optional | Controls how the final partial KV block is handled. Defaults to `0`. | int64 | - | - |
| is_consistent_topk | bool | Optional | Indicates whether query blocks in the same batch and head share the same maximum selected KV block count. Defaults to `False`. | bool | - | - |
| attention_out_dtype | dtype | Optional | Output dtype. Required when `quant_mode!=0`. When `quant_mode=0` and not specified, it is the same as q | ScalarType | - | - |

## Return Values

### generic_block_sparse_attention_metadata

| Name | Type | Required/Optional | Description | Data Type | Data Format | Dimension |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| metadata | Tensor | Required | Task splitting data of generic_block_sparse_attention | int32 | ND | (1024,) |

### generic_block_sparse_attention

| Name | Type | Required/Optional | Description | Data Type | Data Format | Dimensions |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| attention_out | Tensor | Required | Computation output of generic_block_sparse_attention. | bfloat16/float16 | ND | Same as q |
| softmax_lse | Tensor | Optional | LSE result of softmax. | float32 | ND | TND: (Q_T, Q_N, 1); BSND/BNSD: (B, Q_N, Sq, 1); empty when disabled |

**NOTE**

- attention_out: Constraint: The shape of this output parameter must be the same as the shape of the input parameter query. When `quant_mode=0`, the dtype defaults to that of q. When `quant_mode!=0`, the dtype is specified by attention_out_dtype.
- softmax_lse: When enabled, LSE has shape (Q_T, Q_N, 1) for TND or (B, Q_N, Sq, 1) for BSND/BNSD. If return_softmax_lse is False, the output is an empty Tensor.

## Constraints

- INT8 quantization on <term>Atlas A2 products</term> and <term>Atlas A3 products</term> (quant_mode=7): D is 1..256 and the GQA group size is 1..128. Q supports TND, BSND, and BNSD; KV uses the same nonpaged layout as Q or PA_BBND/PA_BNBD with 128-token pages. block_shape=[1,Y], with Y in {1,2,4,8,16,32,64,128}; mask_mode=0/1, residual_block_mode=0/1, and softmax_precision=0. For TND Q, `seqused_q` must be `None`. Metadata and attention must use matching inputs and attributes.

- Parameters cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv, sparse_block_idx, sparse_block_count, and block_table are tensors. The operator cannot obtain the specific values of these tensors during the Tiling phase. Therefore, the tiling side does not validate the values, and users must ensure their correctness. If invalid values are passed for the preceding parameters, undefined behavior may occur (such as accuracy issues or program crashes caused by invalid memory access).
- The input parameters of generic_block_sparse_attention_metadata and generic_block_sparse_attention must be consistent during the call. The operator is called in two phases through two APIs and cannot perform the check itself. Customers must ensure correctness. If the parameters passed to the APIs are inconsistent, undefined behavior may occur (such as accuracy issues or program crashes caused by invalid memory access).

### Product-Specific Constraints

- A4C4 (quant_mode=4) on <term>Ascend 950PR&950DT products</term>: D=192/256, block_shape=[1,Y] with Y in {4,8,16,32,64,128}, GQA group size 1–16, mask_mode=0/1, residual_block_mode=0/1, and softmax_precision=0. Q/K/V use FLOAT4_E2M1 with E8M0 scales; FP16/BF16 output and optional FP32 LSE are supported. [A4C4 API constraints](#a4c4quantmode4).
- On <term>Ascend 950PR&950DT products</term>, set `quant_mode=QuantMode.Q_FP8_PER_TENSOR_KV_FP4_PER_TOKEN_PER_GROUP` (8) in both Metadata and Attention. Call `generic_block_sparse_attention_metadata` first and explicitly pass its result as `metadata` to Attention. Use logical D=128, `block_shape=[1,128]`, GQA group size 1–32, `mask_mode=0` (non-causal) or `1` (causal), `softmax_precision=0`, and `residual_block_mode=0`. Set `attention_out_dtype` to float16 or bfloat16. LSE is optional.
  - `max_seqlen_q` and the actual Q sequence length of each batch item are at most 16. For BSND and BNSD, the Q tensor sequence axis is also limited to 16. `num_heads_kv` is at most five for PA_BBND, BSND, and TND KV layouts. This five-head limit does not apply to BNSD or PA_BNBD.
  - Q supports `TND`, `BSND`, and `BNSD`. Dense K/V must use the same layout as Q. Each Q layout can also use `PA_BBND` or `PA_BNBD` with 128-token pages. See the [layout matching relationship table](#layout-matching-relationship-table) for output shapes.
  - Pass Q as `torch.float8_e4m3fn`. The operator uses a fixed Q scale of `1.0`. Pass `None` for `q_dequant_scale`. Pass both K and V as `torch.float4_e2m1fn_x2` or both as `torch.uint8` containing packed FP4 bytes. In either representation, the last Torch dimension is logical `D/2` (64 bytes for D=128). Each byte holds the even-indexed FP4 E2M1 value in the low four bits and the odd-indexed value in the high four bits. For each token and KV head, K and V each use a BF16 scale per group of 16 elements along D. The last dimension of each scale tensor is logical `D/16`. Use contiguous ND tensors for dense scales. Pass `None` for `p_quant_scale`.
  - Paged K, V, K scale, and V scale can each use a different first-axis stride. Each page must be contiguous. For packed K/V, the Torch stride is measured in bytes and must be at least `block_size × KV_N × 64` and a multiple of `KV_N × 64`. For the scales, the stride is measured in BF16 elements and must be at least `block_size × KV_N × 8` and a multiple of `KV_N × 8`. See [ACLNN constraints](./aclnnGenericBlockSparseAttention_en.md).

- Ascend 950PR&amp;950DT non-quant FLOAT16/BFLOAT16: `mask_mode` supports 0/1/3; non-union uses `block_shape=[1,Y]` (`1≤Y≤128`); union uses `block_shape=[8,Y]` (currently only 8); `residual_block_mode` supports 0 or 1 (union supports only 0).
- Non-quant FLOAT16/BFLOAT16 on <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: `head_dim` supports 1 to 256; `layout_q/layout_kv` supports `TND/PA_BBND`, `TND/PA_BNBD`, `TND/TND`, `BSND/BSND`, `BNSD/BNSD`, `BSND/PA_BBND`, `BNSD/PA_BBND`, `BSND/PA_BNBD`, and `BNSD/PA_BNBD`; `mask_mode` supports 0/1/3; non-union uses `block_shape=[1,Y]` (Y supports 1, 2, 4, 8, 16, 32, 64, 128) and union (mask_mode=3) uses `block_shape=[8,Y]`; `residual_block_mode` supports 0 or 1 (union supports only 0). TND KV requires `cu_seqlens_kv`, and dense Q/K/V are stored contiguously in the corresponding layout. In causal mode, the actual KV length of each batch must be no less than the actual Q length.
- Union (mask_mode=3) on <term>Atlas A2 products</term> and <term>Atlas A3 products</term> supports only `quant_mode=0` (non-quantized FP16/BF16) with `block_shape=[8,Y]` and `residual_block_mode=0`. Y supports 1, 2, 4, 8, 16, 32, 64, 128, and the PA KV page size is 128. Dense KV with the same layout (TND, BSND, BNSD) and paged KV (PA_BBND, PA_BNBD) are supported. INT32 exactMask is required, and the product of the GQA group size and `block_shape[0]` must not exceed 128.
- Regular FP8 on Ascend 950PR&amp;950DT: `head_dim=128`, `mask_mode=1`, `residual_block_mode=0`, `block_shape=[1,Y]` (`1≤Y≤128`). Union on Ascend 950 supports only `quant_mode=0` (non-quantized FP16/BF16) and `quant_mode=5` (FP8). Union FP8 requires INT32 exactMask, `block_shape=[8,Y]` (`1≤Y≤128`), `residual_block_mode=0`, and `head_dim=128`. For paged KV, Y must divide the page size of 128.
- On <term>Ascend 950PR&950DT products</term>, regular FP8 (`quant_mode=5`, `mask_mode=1`) with `softmax_precision=0` requires `head_dim=128` and `1≤group_size≤16`. Supported Q/KV layouts are `TND/TND`, `BSND/BSND`, `BNSD/BNSD`, and each Q layout paired with `PA_BBND` or `PA_BNBD`. Set `return_softmax_lse` to 0 or 1. Low precision (`softmax_precision=1`) supports only `TND/PA_BBND` with LSE disabled.
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
|                      | layout_sparse_pattern | ATTR(OPTIONAL) | int64 |
|                      | residual_block_mode | ATTR(OPTIONAL) | int64 |
|                      | is_consistent_topk | ATTR(OPTIONAL) | bool |
|                      |       metadata       |      INPUT      |   Tensor   |
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
                <li>See the q dimensions in the parameter table.</li>
            </ul>
        </td>
        <td rowspan="4">            Must exist.
        </td>
        <td rowspan="4">            <ul>
                <li>For quant_mode=8, use FP8 E4M3FN q and packed FP4 E2M1 k/v. Other modes require matching q/k/v data types.</li>
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
                <li>See the k/v data types and dimensions in the parameter table. For quant_mode=8, the packed final Torch dimension is D/2.</li>
                <li>Currently, block_size must be 128 and equal to block_shape[1].</li>
                <li>Non-contiguous tensor constraint: only dim0 of PA_BBND supports non-contiguous layout.</li>
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

- Layout matching relationship table. See [Product-Specific Constraints](#product-specific-constraints) for each mode's supported combinations. All combinations listed below are supported for `quant_mode=8`. <a name="layout-matching-relationship-table"></a>

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
            <td>TND, PA_BBND, or PA_BNBD</td>
            <td>(Q_T, Q_N, D)</td>
            <td>(Q_T, Q_N, 1)</td>
        </tr>
        <tr>
            <td>BSND</td>
            <td>BSND, PA_BBND, or PA_BNBD</td>
            <td>(B, Sq, Q_N, D)</td>
            <td>(B, Q_N, Sq, 1)</td>
        </tr>
        <tr>
            <td>BNSD</td>
            <td>BNSD, PA_BBND, or PA_BNBD</td>
            <td>(B, Q_N, Sq, D)</td>
            <td>(B, Q_N, Sq, 1)</td>
        </tr>
    </tbody>
    </table>

- Constraints for `sparse_block_idx`, `sparse_block_count`, and `block_shape`

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
                    </ul>
                </td>
                <td>Required.</td>
                <td>None</td>
                <td>Pass the result of generic_block_sparse_attention_metadata for the same inputs and attributes.</td>
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
                        <li>The value is 0 or 1. In non-Union scenarios (mask_mode=0/1) of Ascend 950PR&950DT products, FP8 quantization supports 0 or 1; in Union scenarios (mask_mode=3), only 0 is supported. On <term>Atlas A2 products</term> and <term>Atlas A3 products</term>, FLOAT16 supports 0 or 1 and BFLOAT16 supports only 0 in non-Union scenarios, and only 0 is supported in Union scenarios (mask_mode=3).</li>
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
                    <li>Supports 0 (non-causal) and 1 (causal); <term>Ascend 950PR&950DT products</term> and non-quantized inputs on <term>Atlas A2 products</term> and <term>Atlas A3 products</term> also support 3 (union). See product-specific constraints.</li>
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
                    <li>Supported values are 0 (non-quantized), 4 (A4C4), 5 (FP8), 6 (A8C8), 7 (INT8 on <term>Atlas A2 products</term> and <term>Atlas A3 products</term>), and 8 (FP8 Q with FP4 K/V). Modes 4, 5, 6, and 8 apply to <term>Ascend 950PR&950DT products</term>.</li>
                </ul>
            </td>
            <td>Optional input. The default value is 0.</td>
            <td rowspan="5">                <ul>
                    <li>The quant_mode of the metadata interface and the main operator must be the same.</li>
                </ul>
            </td>
            <td rowspan="5">                <ul>
                    <li>If quant_mode=0, q/k/v_dequant_scale must be None. For quant_mode=8, the Q scale is fixed at 1.0. Pass independent BF16 K/V scales and leave q_dequant_scale and p_quant_scale as None.</li>
                    <li>If quant_mode!=0, attention_out_dtype must be passed, and the output dtype in current quantization use cases must be float16 or bfloat16.</li>
                </ul>
             A4C4 (mode 4) requires three E8M0 dequant scales and p_quant_scale=None; see <a href="#a4c4quantmode4">A4C4</a>.</td>
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
                    <li>Non-quant (quant_mode=0), A4C4 (quant_mode=4), and INT8 on <term>Atlas A2 products</term> and <term>Atlas A3 products</term> (quant_mode=7), and A8C4 (quant_mode=8) support False and True; high-precision FP8 on Ascend 950PR&950DT products (quant_mode=5, softmax_precision=0, D=128) supports False/True; low-precision FP8 on Ascend 950PR&950DT products (quant_mode=5, softmax_precision=1) support only False.</li>
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

    # top_k is the index capacity; the valid count is bounded by the number of logical KV blocks.
    num_sparse_kv_blocks = math.ceil(KV_S / block_y)
    valid_count = min(top_k, num_sparse_kv_blocks)
    sparse_block_idx = torch.full(
        (KV_N, total_q_blocks, top_k), -1, dtype=torch.int32, device="npu"
    )
    sparse_block_idx[..., :valid_count] = torch.arange(valid_count, dtype=torch.int32, device="npu")
    sparse_block_count = torch.full((KV_N, total_q_blocks), valid_count, dtype=torch.int32, device="npu")

    # Only the valid prefix is used. Its indices must be nonnegative, strictly ascending, and unique.
    valid_idx = sparse_block_idx[..., :valid_count].cpu()
    assert torch.all(sparse_block_count.cpu() == valid_count).item()
    assert torch.all((valid_idx >= 0) & (valid_idx < num_sparse_kv_blocks)).item()
    assert torch.all(valid_idx[..., 1:] > valid_idx[..., :-1]).item()
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

    # The identity page table maps the first valid_count logical KV blocks.
    # Apply the right-aligned causal mask used by mask_mode=1.
    selected_kv = min(valid_count * block_y, KV_S)
    q_ref = q.cpu().float()
    k_ref = k.cpu().float().reshape(-1, KV_N, D)[:selected_kv]
    v_ref = v.cpu().float().reshape(-1, KV_N, D)[:selected_kv]
    k_ref = k_ref.repeat_interleave(Q_N // KV_N, dim=1)
    v_ref = v_ref.repeat_interleave(Q_N // KV_N, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q_ref, k_ref) / math.sqrt(D)
    visible_kv = KV_S - Q_S + torch.arange(Q_S) + 1
    visible_mask = torch.arange(selected_kv)[None, :] < visible_kv[:, None]
    scores.masked_fill_(~visible_mask[None, :, :], float("-inf"))
    reference = torch.einsum("hqk,khd->qhd", scores.softmax(dim=-1), v_ref)
    torch.testing.assert_close(attention_out.cpu().float(), reference, atol=1e-2, rtol=1e-2)
    ```

## INT8 quantization on <term>Atlas A2 products</term> and <term>Atlas A3 products</term>

Use `quant_mode=QuantMode.INT8_FULL_QUANT` (7) for INT8 Q/K/V; pass the same value to both metadata and attention. All four external scales are required. `softmax_precision=0` is required; the output is `torch.float16` or `torch.bfloat16`. Scale shapes and layout constraints are described in the [ACLNN quantization contract](aclnnGenericBlockSparseAttention_en.md#quantization-related-description). `quant_mode=5` is for FP8 on <term>Ascend 950PR&950DT products</term> and cannot be used for INT8.

### A8C8 precision and LSE

For `quant_mode=6`, explicitly set `softmax_precision=0` in both Metadata and Attention. Value 1 is rejected; the general Python default remains 1 for compatibility with other modes. `return_softmax_lse=True` returns FP32 LSE with shape `[Tq,Nq,1]` for TND or `[B,Nq,Sq,1]` for BSND/BNSD. LSE uses dequantized Q/K before P quantization; empty rows and padding contain `-FLT_MAX`. Disabling LSE returns an empty Tensor. See the [A8C8 contract](./aclnnGenericBlockSparseAttention.md#a8c8quanttype6).

<a name="a4c4quantmode4"></a>

## A4C4 (MXFP4 CX, quant_mode=4)

`QuantMode.FP4_E2M1_DYNAMIC_CX` (4) selects A4C4. Use mode 4 and `softmax_precision=0` in Metadata and Attention.
Mode 6 is exclusively MXFP8; OCP mode 3 remains unsupported.
Logical D is 192 or 256, `block_shape=[1,Y]` with Y in {4,8,16,32,64,128}, and the Q/KV head ratio is in [1,16].
Both `mask_mode=0/1` and `residual_block_mode=0/1` are supported, with optional FP32 LSE.
See the [ACLNN A4C4 description](./aclnnGenericBlockSparseAttention_en.md#a4c4quanttype4) for the nine layout pairs.

### Inputs and Outputs

| Parameter | Torch type and convention |
| --- | --- |
| q, k, v | packed uint8; two E2M1 values per byte, physical final dimension D/2; even D elements use the low four bits |
| q_dequant_scale, k_dequant_scale | E8M0 or uint8 encoding; preserve the data prefix axes and replace the final D/2 axis with [D/64,2] |
| v_dequant_scale | E8M0 or uint8 encoding; follow the ACLNN A4C4 scale table and independent alignment of each batch |
| p_quant_scale | None; P is dynamically quantized inside the operator |
| dst_type_max | 0 selects CX=6; explicit values must be finite and in [6,12] |
| attention_out_dtype | Explicit torch.float16 or torch.bfloat16 is required |
| attention_out | Same layout as q, with logical final dimension D |
| softmax_lse | FP32; [Tq,Nq,1] for TND or [B,Nq,Sq,1] for BNSD/BSND; empty when disabled |

All three dequantization scales are required. For each effective sequence, KV head and channel, the V scale remains constant across tokens and physical pages and is repeated in the documented storage shape.
For TND V scales, round each batch's storage length from `cu_seqlens_kv` up independently before concatenation.
Do not align only the total length or derive batch offsets from shorter `seqused_kv` values.
Paged K/V and their scales share physical IDs from `block_table`, including non-contiguous or shuffled IDs; the interface normalizes input storage to contiguous tensors.

### Length and Numerical Conventions

TND Q requires `cu_seqlens_q`; TND KV requires `cu_seqlens_kv`.
Paged KV requires `block_table` and `seqused_kv` with `cu_seqlens_kv=None`.
BNSD/BSND Q requires `seqused_q`; BNSD/BSND KV requires `seqused_kv`.
Use logical D for Metadata's `head_dim` and matching inputs and attributes in both calls.
In causal mode, actual KV length must be at least actual Q length in every batch.

P uses dynamic CX quantization; the Softmax denominator and LSE use FP32 exponentials before quantization, with no additional output dequantization.
Quantization error, block pairing and FD accumulation order can prevent bitwise equality across layouts or schedules.
See the [ACLNN A4C4 description](./aclnnGenericBlockSparseAttention_en.md#a4c4quanttype4) for the scale and numerical conventions.
