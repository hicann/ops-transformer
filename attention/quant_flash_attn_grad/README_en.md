# QuantFlashAttnGrad

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Support |
| ------------------------------------------------------------ | ------ |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>     |    ×     |
| <term>Atlas A2 products</term>     |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×    |
| <term>Atlas inference products</term>                              |    ×     |
| <term>Atlas training products</term>                              |    ×    |

## Function Description

- Operator function: Computes the backward gradients of quantized attention (QuantFlashAttention). This operator is the paired backward interface of the forward operator `aclnnQuantFlashAttention`. It computes the gradients of Query, Key, and Value (dq, dk, dv) and the sink gradient (dsink). It supports the HIFLOAT8 quantized data type, three data layout formats (BSND, BNSD, and TND), and three mask modes: full computation (mask_mode=0), causal mask (rightDownCausal, mask_mode=3), and sliding-window mask (band, mask_mode=4). Under the TND layout, Q/K/V are stored packed by variable-length sequences. cu_seqlens_q and cu_seqlens_kv describe the sequence boundaries of each batch, and seqused_q and seqused_kv can further specify the sequence lengths that actually participate in the computation for each batch. Under the BSND and BNSD layouts, seqused_q and seqused_kv can truncate the redundant computation of each batch.

- Prerequisite: Before calling this operator, you must first call the `quant_flash_attn_metadata` operator to generate metadata and pass it in. The metadata records the task partitioning (core scheduling) result of AICore/AIVCore. In variable-length scenarios with the TND layout or with seqused_q and seqused_kv passed in, the metadata also carries per-batch scheduling information (round prefix, number of S1/S2 blocks per batch, band bottom-right anchor token, scheduling type, and so on). The input parameters of the main operator and `quant_flash_attn_metadata` must be consistent; otherwise, undefined behavior occurs.

- Computation formula:

    The backward gradient computation of quantized attention consists of the following stages:

    Stage 1: Dequantize the input data and convert the quantized inputs to the high-precision floating-point domain

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

    Stage 3: Compute the Key/Value gradients (main Cube computation stage)

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

    Here, $q\_descale$, $k\_descale$, $v\_descale$, and $do\_descale$ are quantization scale factors, $p\_scale$ is the scale factor of the P matrix, $ds\_scale$ is the dequantization scale factor, and $softmax\_scale$ is the scaling coefficient (the recommended value is the reciprocal of $\sqrt{D}$).

### Data Layout Description

The Q, K, and V data layout formats can be interpreted across multiple dimensions, where B (Batch) denotes the batch size (batch_size) of input samples, S (Seq-Length) denotes the sequence length of input samples, N (Head-Num) denotes the number of heads, D (Head-Dim) denotes the size of the smallest hidden-layer unit (headdim), and T (Total-Token-Num) denotes the cumulative sum of sequence lengths of all batches under the TND layout. The shape conventions under each layout are as follows:

| Layout Format | q/dout/attn_out/dq | k/v/dk/dv | softmax_lse |
| -------- | ------------------ | --------- | ----------- |
| BSND | (B, S1, N1, D) | (B, S2, N2, D) | (B, N1, S1) |
| BNSD | (B, N1, S1, D) | (B, N2, S2, D) | (B, N1, S1) |
| TND | (T1, N1, D) | (T2, N2, D) | (N1, T1) |

TND layout description:

- TND (Token-Num-Dim) is a packed layout for variable-length scenarios: the sequences of all batches are concatenated continuously along the T dimension, and batch boundaries are described by cu_seqlens_q (cu_seqlens_kv on the K/V side), that is, T1 = cu_seqlens_q[B] and T2 = cu_seqlens_kv[B].
- The batch count B is derived as the length of cu_seqlens_q (or cu_seqlens_kv) minus 1 and is not passed in separately.
- The kernel uses cu_seqlens_q/cu_seqlens_kv to compute the starting token offset (q_start/kv_start) of each batch in GM for addressing, and Q/K/V share the contiguous storage along the T dimension; mask coordinates are always local coordinates within a batch.
- Under TND, the shapes of dq/dk/dv are consistent with those of q/k/v, that is, dq is (T1, N1, D) and dk/dv are (T2, N2, D); the shape of dsink is (N1,).

### Variable-Length Sequence Description (cu_seqlens and seqused)

| Parameter | Description |
| ---- | ---- |
| cu_seqlens_q | Cumulative sum form of the Query sequence lengths of all batches, INT32, shape (B+1,). The first element must be 0, the values must be non-decreasing, and the last element equals T1. |
| cu_seqlens_kv | Cumulative sum form of the Key/Value sequence lengths of all batches, INT32, shape (B+1,). The first element must be 0, the values must be non-decreasing, the last element equals T2, and the length must be consistent with cu_seqlens_q. |
| seqused_q | Sequence length of Query that actually participates in the computation for each batch (taking the prefix of the corresponding cu_seqlens segment or S1), INT32, shape (B,). |
| seqused_kv | Sequence length of Key/Value that actually participates in the computation for each batch (taking the prefix of the corresponding cu_seqlens segment or S2), INT32, shape (B,). |

- cu_seqlens_q and cu_seqlens_kv are valid only under the TND layout: they must be passed in under the TND layout, and passing them in is not supported under the BSND/BNSD layouts.
- seqused_q and seqused_kv are optional under both the TND and BSND/BNSD layouts: under the TND layout, their values must be less than or equal to the length of the cu_seqlens segment of the corresponding batch; under the BSND/BNSD layouts, their values must be less than or equal to S1/S2.
- After seqused_q and seqused_kv are passed in, only the first seqused tokens of each batch participate in the computation, which truncates redundant computation: under the BSND/BNSD layouts, the padding part of each batch is skipped; under the TND layout, they are used together with cu_seqlens (cu_seqlens defines the packing boundaries, and seqused defines the actual valid lengths).
- When the TND layout is used or seqused_q and seqused_kv are passed in, the variable-length (varlen) path is taken: in the PreSfmg stage, dO/O are loaded truncated to the valid lengths (under TND with seqused_q passed in, truncation uses effective_q = seqused_q[b] and addressing uses global_q = cu_seqlens_q[b] + local_q); in the Cube stage, the block counts (s1_outer/s2_outer) and mask boundaries are all computed based on the actual valid lengths of each batch, and per-batch information is delivered by the metadata.
- cu_seqlens_q, cu_seqlens_kv, seqused_q, and seqused_kv are tensor inputs. Their specific values cannot be obtained at the tiling stage, so no value validation is performed and you must ensure their correctness yourself. Passing invalid values triggers undefined behavior (accuracy problems, program crashes caused by illegal memory access, and so on).

### mask_mode Description

| mask_mode Value | Name | Description |
| -------------- | ---- | ---- |
| 0 | No mask (ALL_MASK) | Full computation without applying a mask. attn_mask must be empty. |
| 3 | rightDownCausal | Lower-triangular mask anchored at the bottom-right corner: row i of the query attends only to keys that satisfy j ≤ i + (S2 - S1). When S1 = S2, this is the standard causal (lower-triangular) mask; when S1 > S2, the first S1 - S2 query rows fall entirely outside the causal region, and the corresponding redundant blocks are skipped automatically. |
| 4 | band (Sliding Window) | Sliding-window mask anchored at the bottom-right corner: after query/key are aligned at the bottom-right corner, row i of the query attends only to keys whose distance falls within the [-win_right, win_left] interval. When S1 ≠ S2, the host applies the S1 - S2 correction to win_left/win_right; in variable-length (TND/seqused) scenarios, anchors are computed per batch in the metadata based on the actual sequence length of each batch. |

- attn_mask must be passed in when mask_mode=3 or 4; passing in attn_mask is not allowed when mask_mode=0.
- attn_mask is an INT8 mask template with a fixed shape of (2048, 2048). A typical construction is a lower-triangular template (for example, `torch.tril(torch.ones(2048, 2048, dtype=torch.int8))`). The kernel extracts the corresponding window from the template based on the offset of the current tile relative to the mask boundary: in the causal scenario, one window is taken aligned with the causal line; in the band scenario, two windows are taken aligned with the right/left boundaries of the sliding window respectively, and the set difference yields the sliding-window region.
- win_left/win_right can be set to values other than -1 only when mask_mode=4; under all other mask_mode values they must remain at the default value -1. The values must be greater than or equal to -1. -1 means no window limit in that direction (the host treats it as infinity), and other negative values are not yet supported.
- mask_mode=3/4 uses sparse scheduling: blocks that do not intersect the mask region are skipped entirely, and element-level masks are applied to boundary blocks; this skipping logic applies to both the fixed-length BSND/BNSD scenarios and the variable-length TND/seqused scenarios.

### Kernel Computation Flow Description

The overall computation flow of the operator corresponds to the four stages above. The variable-length and mask related flows are as follows:

- PreSfmg stage (Vector core): dequantizes dout/attn_out and computes the Softmax gradient. In variable-length scenarios, the sfmg and dk/dv workspaces are first zero-initialized (sparse scheduling may revisit the same head/KV column, and variable-length scenarios require zero padding for unvisited columns).
- Main Cube loop stage (Cube core): traverses (batch, head, s1_block, s2_block) according to the deterministic schedule. In variable-length scenarios, the schedule is delivered by the metadata (dense swizzle for BSND/BNSD with seqused, and line swizzle anti-diagonal scheduling for TND), which guarantees unique cross-core ownership of dk/dv and a deterministic accumulation order; in sparse scenarios (mask_mode=3/4), block-level skipping is performed according to the mask region.
- Post stage (Vector core): writes the accumulated results in the workspace out as dq/dk/dv (BF16) and computes dsink (FLOAT).

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
| ------ | -------------- | ---- | -------- | -------- |
| q | Input | Input Q of the attention structure, quantized data. Shape: BSND (B, S1, N1, D), BNSD (B, N1, S1, D), TND (T1, N1, D). | HIFLOAT8 | ND |
| k | Input | Input K of the attention structure, quantized data. Shape: BSND (B, S2, N2, D), BNSD (B, N2, S2, D), TND (T2, N2, D). | HIFLOAT8 | ND |
| v | Input | Input V of the attention structure, quantized data. Shape consistent with k. | HIFLOAT8 | ND |
| dout | Input | Gradient corresponding to the forward output attn_out, quantized data. Shape consistent with q. | HIFLOAT8 | ND |
| attn_out | Input | attn_out from the forward computation output. Shape consistent with q. | BF16 | ND |
| q_descale | Input | Dequantization scale factor of Query; one scale factor per quantization block. | FLOAT | ND |
| k_descale | Input | Dequantization scale factor of Key; one scale factor per quantization block. | FLOAT | ND |
| v_descale | Input | Dequantization scale factor of Value; the shape must be consistent with k_descale. | FLOAT | ND |
| do_descale | Input | Dequantization scale factor of dout; the shape must be consistent with q_descale. | FLOAT | ND |
| p_scale | Input | Quantization scale factor of the P matrix. | FLOAT | ND |
| ds_scale | Input | Dequantization scale factor. | FLOAT | ND |
| softmax_lse | Input | softmaxLse output of the forward attention computation. Shape: BSND/BNSD (B, N1, S1), TND (N1, T1). | FLOAT | ND |
| cu_seqlens_q | Optional input | Cumulative sequence lengths of Query, used to divide batch boundaries for TND variable-length sequences. INT32, shape (B+1,). The first element must be 0, the values must be non-decreasing, and the last element equals T1. Supported and required only under the TND layout; passing it in is not supported under non-TND layouts. | INT32 | ND |
| cu_seqlens_kv | Optional input | Cumulative sequence lengths of Key/Value, used to divide batch boundaries for TND variable-length sequences. INT32, shape (B+1,). The first element must be 0, the values must be non-decreasing, the last element equals T2, and the length must be consistent with cu_seqlens_q. Supported and required only under the TND layout; passing it in is not supported under non-TND layouts. | INT32 | ND |
| seqused_q | Optional input | Sequence length of q that actually participates in the computation for each batch, used to truncate redundant computation. INT32, shape (B,), with non-negative integer values. Under the TND layout, the value must be less than or equal to the length of the cu_seqlens_q segment of the corresponding batch; under the BSND/BNSD layouts, the value must be less than or equal to S1. | INT32 | ND |
| seqused_kv | Optional input | Sequence length of k/v that actually participates in the computation for each batch, used to truncate redundant computation. INT32, shape (B,), with non-negative integer values. Under the TND layout, the value must be less than or equal to the length of the cu_seqlens_kv segment of the corresponding batch; under the BSND/BNSD layouts, the value must be less than or equal to S2. | INT32 | ND |
| sinks | Optional input | Input tensor for the sink scenario. The current version does not support non-empty inputs; pass an empty tensor. | FLOAT | ND |
| attn_mask | Optional input | Attention mask template, INT8, with a fixed shape of (2048, 2048). Required when mask_mode=3/4; not allowed when mask_mode=0. | INT8 | ND |
| metadata | Optional input | Output of the aicpu operator (quant_flash_attn_metadata) with tiling offloading. The current version requires it to be passed in, and it must be generated by quant_flash_attn_metadata. INT32, shape (2, max_schedule_size). The second dimension must be no less than: 17+10×B for the TND layout; 17+3×B for non-TND layouts with seqused_q/seqused_kv passed in (17+10×B when mask_mode=3/4); 16 for other scenarios. | INT32 | ND |
| dq | Output | Gradient of Query, with dtype fixed to BF16; shape consistent with q. | BF16 | ND |
| dk | Output | Gradient of Key, with dtype fixed to BF16; shape consistent with k. | BF16 | ND |
| dv | Output | Gradient of Value, with dtype fixed to BF16; shape consistent with v. | BF16 | ND |
| dsink | Output | Gradient of sink, with dtype fixed to FLOAT; shape (N1,). | FLOAT | ND |
| quant_mode | Required attribute | Quantization mode. 0: HIFLOAT8 per-tensor quantization. | INT64 | - |
| softmax_scale | Optional attribute | Scaling coefficient, with a default value of 1.0. Recommended value: the reciprocal of sqrt(head_dim). | FLOAT | - |
| mask_mode | Optional attribute | Mask mode for the q and k computation. 0: No mask (full computation; attn_mask must be empty); 3: rightDownCausal (lower-triangular causal anchored at the bottom-right corner); 4: band (sliding window, used together with win_left/win_right). Default value: 0. See mask_mode Description for details. | INT64 | - |
| win_left | Optional attribute | Left window width of the sliding-window mask. Valid only when mask_mode=4. -1 means no window limit in that direction, and other negative values are not supported; it must be -1 when mask_mode is not 4. Default value: -1. | INT64 | - |
| win_right | Optional attribute | Right window width of the sliding-window mask. Valid only when mask_mode=4. -1 means no window limit in that direction, and other negative values are not supported; it must be -1 when mask_mode is not 4. Default value: -1. | INT64 | - |
| max_seqlen_q | Optional attribute | Maximum sequence length of Query. Only the TND layout supports passing in values greater than 0; -1 means automatic derivation. Non-TND layouts support only -1. Default value: -1. | INT64 | - |
| max_seqlen_kv | Optional attribute | Maximum sequence length of Key/Value. Only the TND layout supports passing in values greater than 0; -1 means automatic derivation. Non-TND layouts support only -1. Default value: -1. | INT64 | - |
| layout_q | Optional attribute | Data layout format of input q. Supports "BSND", "BNSD", and "TND", with a default value of "BSND". layout_q and layout_kv must be consistent. | STRING | - |
| layout_kv | Optional attribute | Data layout format of input k/v. Supports "BSND", "BNSD", and "TND", with a default value of "BSND". layout_q and layout_kv must be consistent. | STRING | - |

## Constraint Description

- Determinism: aclnnQuantFlashAttnGrad uses a deterministic implementation by default.
- Layout constraints:
    - The BSND, BNSD, or TND layout is supported, and layout_q and layout_kv must be consistent.
    - Under the TND layout, q/k/v/dout/attn_out are all 3-dimensional tensors; under the BSND/BNSD layouts, they are all 4-dimensional tensors.
- Data shape constraints:
    - B: Under the BSND/BNSD layouts, 0 < B < 65536 is supported; under the TND layout, B is derived as the length of cu_seqlens_q (or cu_seqlens_kv) minus 1, requiring B ≥ 1, B ≤ T1, and B ≤ T2.
    - S1 and S2: generally supported, and S1 and S2 may differ in length; under the TND layout, the sequence lengths of batches may differ (divided per batch by cu_seqlens).
    - N1: 1 to 128 is supported. Under the BSND/BNSD layouts, the GQA constraint must be satisfied (N1 = N2 × G, and num_heads_q must be divisible by num_heads_k); the TND layout does not support GQA and requires N1 = N2.
    - N2: 1 to 128 is supported.
    - D: fixed to 128 in quantization scenarios.
- In quantization scenarios, the dimensions of q, k, v, dout, and attn_out must be consistent, and the shapes of k and v must be consistent.
- The shapes of q_descale and do_descale must be consistent, and the shapes of k_descale and v_descale must be consistent.
- TND layout constraints:
    - Only quant_mode=0 is supported, and mask_mode supports only 0/3/4.
    - cu_seqlens_q and cu_seqlens_kv must be passed in (INT32, shape (B+1,); the first element is 0, the values are non-decreasing, the last elements equal T1/T2 respectively, and the two lengths are consistent); passing in cu_seqlens_q and cu_seqlens_kv is not supported under the BSND/BNSD layouts.
    - The shape of softmax_lse is (N1, T1); the shapes of dout and attn_out must be exactly consistent with q.
    - max_seqlen_q/max_seqlen_kv support passing in values greater than 0 only under the TND layout (-1 means automatic derivation); non-TND layouts support only -1.
- seqused constraints:
    - INT32, shape (B,), with non-negative integer values.
    - Under the TND layout, the values of seqused_q/seqused_kv must be less than or equal to the length of the cu_seqlens segment of the corresponding batch; under the BSND/BNSD layouts, they must be less than or equal to S1/S2.
    - When seqused_q/seqused_kv are passed in, the variable-length path is taken. The metadata must be generated by quant_flash_attn_metadata and passed in, and it must be consistent with the input parameters of the main operator.
- mask_mode support:
    - 0: no mask operation (ALL_MASK); attn_mask must be empty.
    - 3: mask of the rightDownCausal mode, which divides the lower-triangular scenario anchored at the bottom-right corner; attn_mask cannot be empty.
    - 4: mask of the band mode, which divides the sliding-window scenario anchored at the bottom-right corner; it must be used together with the win_left and win_right attributes, and attn_mask cannot be empty.
    - win_left/win_right can be set to values other than -1 only when mask_mode=4, and the values must be greater than or equal to -1 (-1 means no window limit in that direction); they must remain -1 when mask_mode is not 4.
- metadata constraints:
    - In the current version, metadata must be passed in, must be generated by quant_flash_attn_metadata, and has shape (2, max_schedule_size).
    - The second dimension must be no less than: 17 + 10×B (TND layout); 17 + 3×B (BSND/BNSD with seqused_q/seqused_kv passed in and mask_mode=0); 17 + 10×B (BSND/BNSD with seqused_q/seqused_kv passed in and mask_mode=3/4); 16 (other scenarios).
- The values of cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv, and attn_mask are not validated at the tiling stage. You must ensure their correctness yourself; passing invalid values triggers undefined behavior.
- Empty tensor (shape size of 0) scenarios are all intercepted with errors.

## Invocation Description

| Invocation Method | Invocation Example | Description |
| -------- | -------- | ---- |
| PyTorch API | - | Call the operator through the [quant_flash_attn_grad](docs/torchapi_quant_flash_attn_grad_en.md) interface. |

Invocation procedure:

1. Prepare inputs such as q, k, and v (under the TND layout, also prepare cu_seqlens_q/cu_seqlens_kv, and prepare seqused_q/seqused_kv as needed).
2. Call `quant_flash_attn_metadata` to generate metadata (in the backward scenario, set is_grad_enabled=True, and keep the input parameters consistent with the main operator).
3. Call `quant_flash_attn_grad` and pass the metadata into the main operator; when mask_mode=3/4, also pass in the attn_mask template.

## Reference Resources (Optional)

None
