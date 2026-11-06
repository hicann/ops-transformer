# QkvRmsNormRopeCacheWithKScale

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
|:---|:---:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Description

- Operator function: Takes the fused Q/K/V tensor `qkv` as input, splits Q, K, and V according to `head_nums=[Nq, Nk, Nv]`, and updates the K/V cache and the K scale cache according to `slot_mapping`. Three complete semantic scenarios are currently supported:
  - RoPE: Q/K perform `RMSNorm -> RoPE -> rotation`, and Q/K undergo PerTokenPerHead dynamic quantization;
  - M-RoPE: Q/K perform `RMSNorm -> M-RoPE -> rotation`. Q is not quantized, and K undergoes PerTokenPerHead INT8 dynamic quantization;
  - M-RoPE MX: Q/K perform `RMSNorm -> M-RoPE -> Dynamic MX Quant`, and `rotation` is not performed. Each D32 block is quantized to FP8 E4M3FN by using a power-of-two FLOAT8_E8M0 scale in the cuBLAS MX FP8 style.
  - The V branch in all three scenarios performs neither RMSNorm, positional encoding, nor `rotation`. RoPE uses `v_scale[Nv]`, and M-RoPE and M-RoPE MX use the per-head, per-channel `v_scale[Nv,D]` to quantize to FP8 E4M3FN.
- Usage scenario: Applicable to PagedAttention KV cache updates in inference scenarios. Currently only <term>Ascend 950PR&950DT products</term> are supported.
- Computation formulas:

  Q, K, and V are split from the fused input according to `head_nums=[Nq, Nk, Nv]`:

  $$
  q, k, v = split(qkv, [Nq, Nk, Nv])
  $$

  The Q and K branches perform RMSNorm using `q_gamma` and `k_gamma` respectively:

  $$
  y = \frac{x}{\sqrt{mean(x^2) + epsilon}} * gamma
  $$

  The Q and K branches perform positional encoding, where `cos_sin[..., :D/2]` is cos and `cos_sin[..., D/2:]` is sin. The V branch does not perform positional encoding:

  $$
  y_{pos} = concat(y_{low} * cos - y_{high} * sin,\ y_{high} * cos + y_{low} * sin)
  $$

  In RoPE and M-RoPE, Q and K share the `rotation` matrix:

  $$
  q_{rot} = q_{pos} @ rotation,\quad k_{rot} = k_{pos} @ rotation
  $$

  When PerTokenPerHead dynamic quantization is enabled in the RoPE and M-RoPE scenarios, one FP32 scale is computed for each token and head:

  $$
  scale = max(abs(x)) / quant\_max,\quad x_{quant} = cast(x / scale)
  $$

  Here `quant_max` is the quantization upper bound of the target data type. M-RoPE MX does not perform the preceding whole-D quantization; instead, it divides the D dimension of each Q/K head into D32 blocks. For each block $x_b$ of finite input, compute the following in FP32:

  $$
  a_b=\max_i|x_{b,i}|,\qquad
  e_b=\max\left(-127,\left\lceil\log_2\frac{a_b}{448}\right\rceil\right),\qquad
  s_b=2^{e_b},\qquad
  q_{b,i}=\operatorname{cast}^{\mathrm{rint,sat}}_{\mathrm{FP8\ E4M3FN}}\left(\frac{x_{b,i}}{s_b}\right).
  $$

  When $a_b=0$, $e_b=-127$ is used. $s_b$ is stored as FLOAT8_E8M0, and the raw encoding of a finite scale is $e_b+127$; when the block contains Inf or NaN, the raw encoding of the scale is `0xFF`. This rule is exactly the cuBLAS MX FP8 scale semantics (corresponding to `scaleAlg=1` of `DynamicMxQuantV3`). Valid scales are stored linearly in D32 blocks, and the number of scales is `ceil(D/32)`.

  The V branch is uniformly multiplied by `v_scale` before conversion to FP8 E4M3FN. `v_scale` is broadcast to the logical shape of V according to its input shape:

  $$
  v_{fp8} = cast_{FP8\ E4M3FN}(v * v\_scale)
  $$

  The cache write-back position is determined by `slot_mapping`:

  $$
  blockId = slot\_mapping[t] / BlockSize,\quad blockOffset = slot\_mapping[t]\ \%\ BlockSize
  $$

  $$
  k\_cache[blockId, nk, blockOffset, :] = k_{quant}[t, nk, :]
  $$

  $$
  v\_cache[blockId, nv, blockOffset, :] = v_{fp8}[t, nv, :]
  $$

  $$
  k\_scale\_cache[blockId, nk, blockOffset, ...] = k\_scale[t, nk, ...]
  $$

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
|---|---|---|---|---|
| qkv | Input | Fused Q/K/V input. When `layout_qkv="TND"`, the shape is `[T, Nq+Nk+Nv, D]`; when `layout_qkv="NTD"`, the shape is `[Nq+Nk+Nv, T, D]`. | BFLOAT16 | ND |
| q_gamma | Input | RMSNorm weight of the Q branch, with shape `[D]`. | FLOAT | ND |
| k_gamma | Input | RMSNorm weight of the K branch, with shape `[D]`. | FLOAT | ND |
| cos_sin | Input | RoPE/M-RoPE positional encoding table, with shape `[MaxSeqLen, D]`. The first `D/2` columns are cos and the last `D/2` columns are sin. | FLOAT | ND |
| slot_mapping | Input | Slot index at which each token is written to the cache, with shape `[T]`. | INT32 | ND |
| k_cache | Input/Output | K cache. The input and output share the same address, with shape `[BlockNum,Nk,BlockSize,D]`. Non-contiguous tensors are supported. FP8 E4M3FN in RoPE and M-RoPE MX; INT8 in M-RoPE. | FLOAT8_E4M3FN / INT8 | ND |
| v_cache | Input/Output | V cache. The input and output share the same address, with shape `[BlockNum, Nv, BlockSize, D]`. Non-contiguous tensors are supported. | FLOAT8_E4M3FN | ND |
| k_scale_cache | Input/Output | K dynamic quantization scale cache. The input and output share the same address. FP32 `[BlockNum,Nk,BlockSize,1]` in RoPE/M-RoPE; FLOAT8_E8M0 `[BlockNum,Nk,BlockSize,ceil(D/32)]` in M-RoPE MX. | FLOAT / FLOAT8_E8M0 | ND |
| query_start_loc | Optional input | Prefix sum of the token count of each batch within the current invocation in the RoPE scenario, with shape `[Batch+1]`. Not passed in the M-RoPE scenario. | INT32 | ND |
| seq_lens | Optional input | Actual sequence length of each batch after the current token is appended in the RoPE scenario, with shape `[Batch]`. Not passed in the M-RoPE scenario. | INT32 | ND |
| rotation | Optional input | Matrix multiplication weight shared by Q/K. BF16 `[D,D]` must be passed in RoPE/M-RoPE; it must not be passed in M-RoPE MX. | BFLOAT16 | ND |
| v_scale | Optional input | Quantization scale factor of the V branch. `[Nv]` in RoPE; per-head, per-channel `[Nv,D]` in M-RoPE and M-RoPE MX. A valid tensor must be passed in all three scenarios. | FLOAT | ND |
| mrope_position | Optional input | Token-major position index in the M-RoPE scenario, with shape `[T,3]`. The three columns are T/H/W in order. Not passed in the RoPE scenario. | INT32 | ND |
| q_out | Output | Q output. FP8 E4M3FN in RoPE, with the shape determined by `layout_q_out`; BF16 `[T,Nq,D]` in M-RoPE; FP8 E4M3FN `[T,Nq,D]` in M-RoPE MX. | FLOAT8_E4M3FN / BFLOAT16 | ND |
| q_scale | Output | Q dynamic quantization scale. FP32 `[T,Nq]` or `[Nq,T]` in RoPE; the exposed output is `nullptr`/`None` in M-RoPE; FLOAT8_E8M0 `[T,Nq,ceil(D/32)]` in M-RoPE MX. | FLOAT / FLOAT8_E8M0 | ND |
| head_nums | Attribute | Numbers of Q/K/V heads, passed as `[Nq,Nk,Nv]`. All three scenarios require `0<Nq<=64`, `Nq=8*Nk`, and `Nk=Nv`. | INT64 array | - |
| layout_qkv | Optional attribute | Layout of the N/T axes of `qkv`. The default value is `"TND"`. Only case-sensitive `"TND"` and `"NTD"` are supported. | STRING | - |
| layout_q_out | Optional attribute | Layout of the N/T axes of `q_out` and `q_scale`. The default value is `"NTD"`. Only case-sensitive `"TND"` and `"NTD"` are supported. | STRING | - |
| epsilon | Optional attribute | Parameter that prevents division by zero in RMSNorm. The default value is `1e-6`. | FLOAT | - |
| mrope_section | Optional attribute | `[t,h,w]` in the M-RoPE scenario. The length must be 3 when the value is not empty. An empty list is equivalent to not passing the attribute. | INT64 array | - |
| q_quant_mode | Optional attribute | Q quantization mode. `PerTokenPerHead` in RoPE, `NoQuant` in M-RoPE, and `Mx` in M-RoPE MX. | STRING | - |
| q_out_dtype | Optional attribute | Q output data type. FP8 E4M3FN in RoPE and M-RoPE MX; BF16 in M-RoPE. | INT64 | - |
| k_quant_mode | Optional attribute | K quantization algorithm mode. The default value is `PerTokenPerHead`; `PerTokenPerHead` and `Mx` are supported. This attribute does not select the dtype. The storage dtype of K is determined by `k_cache` and the scenario contract. | STRING | - |

## Constraints

- The parameter combinations of the three scenarios are as follows. Other cross combinations are not supported:

  | Scenario | Position Input | rotation | Q / q_scale | K / k_scale_cache | V |
  |---|---|---|---|---|---|
  | RoPE | Pass `query_start_loc/seq_lens`; do not pass `mrope_position/mrope_section` | `[D,D]` BF16 must be passed | `q_quant_mode=PerTokenPerHead`; Q is FP8 E4M3FN, and the scale is FP32 rank 2 | `k_quant_mode=PerTokenPerHead`; K is FP8 E4M3FN, and the scale is FP32 rank 4 | `v_scale=[Nv]`, output FP8 E4M3FN |
  | M-RoPE | Pass `mrope_position/mrope_section`; do not pass `query_start_loc/seq_lens` | `[D,D]` BF16 must be passed | `q_quant_mode=NoQuant`; Q is BF16 with no exposed scale | `k_quant_mode=PerTokenPerHead`; K is INT8, and the scale is FP32 rank 4 | `v_scale=[Nv,D]`, output FP8 E4M3FN |
  | M-RoPE MX | Pass `mrope_position/mrope_section`; do not pass `query_start_loc/seq_lens` | Must not be passed | `q_quant_mode=Mx`; Q is FP8 E4M3FN, and the scale is FLOAT8_E8M0 rank 3 | `k_quant_mode=Mx`; K is FP8 E4M3FN, and the scale is FLOAT8_E8M0 rank 4 | `v_scale=[Nv,D]`, statically quantized per head/channel to FP8 E4M3FN |

- Input shape restrictions:
  - The current implementation supports only `D=128`. All three scenarios require `1<=T<=262144`, `0<Nq<=64`, `Nq=8*Nk`, and `Nk=Nv`.
  - In the RoPE scenario, `query_start_loc/seq_lens` must both be passed, `mrope_position` must not be passed, and `mrope_section` must be empty or not passed;
    `q_quant_mode=k_quant_mode=PerTokenPerHead`, `q_out_dtype=FP8 E4M3FN`, and `q_scale` must not be empty, and the K cache is FP8.
  - In the M-RoPE scenario, `mrope_position[T,3]` and non-empty `mrope_section=[t,h,w]` must both be passed, and `query_start_loc/seq_lens` must not be passed;
    M-RoPE requires `q_quant_mode=NoQuant`, `k_quant_mode=PerTokenPerHead`, `q_out_dtype=BF16`, and exposed `q_scale=None`, the K cache is INT8, and only
    `layout_qkv="TND"` and `layout_q_out="TND"` are supported. The two groups of position inputs must not both exist or both be absent.
  - The M-RoPE MX scenario fixes `layout_qkv=layout_q_out="TND"` and `q_quant_mode=k_quant_mode="Mx"`, and `rotation` must not be passed;
    Q/K outputs are FP8 E4M3FN, the Q scale is FLOAT8_E8M0 `[T,Nq,ceil(D/32)]`, and the K scale cache is FLOAT8_E8M0
    `[BlockNum,Nk,BlockSize,ceil(D/32)]`.
  - The shape of `v_scale` must be `[Nv]` in the RoPE scenario and `[Nv,D]` in the M-RoPE and M-RoPE MX scenarios.
  - The `mrope_section` of M-RoPE must have a length of 3, and all three items must be non-negative. With D128, both the H and W items range from `[0,21]`, item 0 has no separate lane limit, and the sum of the three items must not exceed 64. `mrope_position` must be INT32, rank 2, with shape `[T,3]`.
  - `layout_qkv` controls the layout of the N/T axes of `qkv`, with the default value `"TND"`. `layout_q_out` controls the layout of the N/T axes of `q_out` and `q_scale`, with the default value `"NTD"`:
    - `layout_qkv="TND"`, `layout_q_out="TND"`: `qkv=[T, Nq+Nk+Nv, D]`, `q_out=[T, Nq, D]`, `q_scale=[T, Nq]`.
    - `layout_qkv="TND"`, `layout_q_out="NTD"`: `qkv=[T, Nq+Nk+Nv, D]`, `q_out=[Nq, T, D]`, `q_scale=[Nq, T]`.
    - `layout_qkv="NTD"`, `layout_q_out="NTD"`: `qkv=[Nq+Nk+Nv, T, D]`, `q_out=[Nq, T, D]`, `q_scale=[Nq, T]`.
  - The second dimension of `cos_sin` must be `D`, and the first dimension must cover the RoPE/M-RoPE positions accessed in the current invocation.
  - `k_cache`, `v_cache`, and `k_scale_cache` support non-contiguous tensors and must meet the following constraints:
    - `k_cache` and `v_cache` are both 4-dimensional with positive strides, the last-dimension stride is 1, and the strides of both the head and token dimensions are no less than `D=128`;
      the K cache dtype is FP8 or INT8 depending on the scenario, and the V cache is always FP8.
    - The `k_scale_cache` of RoPE/M-RoPE is 4-dimensional with positive strides and a last-dimension stride of 1. M-RoPE MX is also 4-dimensional, the trailing `ceil(D/32)` scale axis is contiguous, and the last-dimension stride must be 1.
    - The first three dimensions of `k_cache` and `v_cache` must have identical strides.
- Input value range restrictions:
  - In the RoPE scenario, `query_start_loc` denotes the batch prefix sum of tokens within the current invocation. `query_start_loc[0]` should be 0, and `query_start_loc[Batch]` should equal `T`.
  - In the RoPE scenario, `seq_lens[b]` denotes the actual sequence length of batch `b` after the tokens of this invocation are appended. For the `i`-th token in the batch, the RoPE position is computed as `seq_lens[b] - (query_start_loc[b+1] - query_start_loc[b]) + i`; the caller must ensure `seq_lens[b] >= query_start_loc[b+1] - query_start_loc[b]`. If `seq_lens[b]` is less than the number of tokens of the batch in this invocation, the behavior is undefined.
  - In both M-RoPE scenarios, every position index in `mrope_position` must satisfy `0 <= value < MaxSeqLen`.
  - The value range of `slot_mapping` should be `[0, BlockNum * BlockSize - 1]`. M-RoPE MX requires that slots within the same invocation be unique. In the RoPE and M-RoPE scenarios, when duplicate slots exist, the final write order and result are undefined.
- Input attribute restrictions:
  - `head_nums` must contain 3 positive integers in the order `[Nq, Nk, Nv]` and satisfy `Nq<=64`, `Nq=8*Nk`,
    and `Nk=Nv`.
  - `layout_qkv` and `layout_q_out` are case-sensitive. Only `"TND"` and `"NTD"` are supported, and `layout_qkv="NTD"` with `layout_q_out="TND"` is currently not supported.
- Input data type restrictions:
  - The data types and data formats of all inputs must comply with the parameter description. Implicit type conversion is not supported.
  - All inputs are in ND format. Private formats are not supported.
- Other restrictions:
  - A valid `v_scale` tensor must be passed in all three scenarios. A valid `rotation` tensor must be passed in RoPE/M-RoPE, and `rotation` must not be passed in M-RoPE MX.
  - Non-optional inputs and outputs do not support empty tensors. Optional position inputs use `None` according to the scenario contract, and empty `mrope_section=[]` is equivalent to not passing it.

## Calling Description

<term>Ascend 950PR&950DT products</term>

| Calling Method | Calling Sample | Description |
|---|---|---|
| aclnn API | [test_aclnn_qkv_rms_norm_rope_cache_with_k_scale](examples/test_aclnn_qkv_rms_norm_rope_cache_with_k_scale.cpp) | Invokes the QkvRmsNormRopeCacheWithKScale operator through the [aclnnQkvRmsNormRopeCacheWithKScale](docs/aclnnQkvRmsNormRopeCacheWithKScale_en.md) API. |
| PyTorch API | - | Invokes the QkvRmsNormRopeCacheWithKScale operator through the [qkv_rms_norm_rope_cache_with_k_scale](docs/torchapi_qkv_rms_norm_rope_cache_with_k_scale_en.md) API. |
