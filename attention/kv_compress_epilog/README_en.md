# KvCompressEpilog

English | [简体中文](./README.md)

## Product Support

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                              |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Description

- Operator function: This operator performs in-place compression update of the cache in the Epilog stage of the KV Cache. After the operator quantizes and compresses the `bfloat16` activation value `x`, it scatters the results to `cache` according to `slot_mapping`. Tokens with a value of -1 in `slot_mapping` are skipped, and tokens with out-of-range values are silently skipped. The last 64 columns of the tail axis `d` of `x` are the rope segment, and the first `d-64` columns are the nope segment. Five quantization modes are supported:
  - `quant_mode=0/1`: `x` is dynamically quantized group by group (Group-wise Dynamic Quantization) and compressed to FP8 (packed as `uint8`); the last 64 columns of the tail axis `d` of `x` form the rope segment, the first `d-64` columns form the nope segment, and the rope segment is kept as `bfloat16`;
  - `quant_mode=2`: the rope segment undergoes hifloat8 **static** quantization (multiplied by `x_scale` and then converted to hifloat8), and the nope segment undergoes per-group **FLOAT4_E2M1 dynamic** quantization (`scale=amax in the group/6`, written out as `bfloat16`);
  - `quant_mode=1` (carries INT8 on <term>Atlas A2 products</term> and <term>Atlas A3 products</term>): the nope segment (first 448 columns, 7 groups of 64 elements each) undergoes INT8 **symmetric dynamic** quantization (`scale=amax in the group/127`, written out as `bfloat16`), the rope segment keeps the original `bfloat16` bits, and the results are scattered according to `slot_mapping`; each token has a fixed length of 608B (rope 128B | nope 448B | scale 14B | pad 18B).
  - `quant_mode=3` (MXFP8): the entire row of `x` is grouped by `quant_group_size` (only 32 is supported) and dynamically quantized to MXFP8 (E4M3FN/E5M2, `scale=group amax/FP8_MAX`, written out as `bfloat16`; when `round_scale=true`, `scale` is rounded up to a power of 2). There is no rope/nope split;
  - `quant_mode=4` (MXFP4 packed): the entire row of `x` is dynamically quantized to MXFP4 (two elements are packed into one E2M1 byte, and the shared exponent scale is bit-encoded as `bfloat16`); `quant_group_size` supports 16/32.

- Calculation formula:

  Quantization is performed on the last dimension (d axis) of `x`. Denote the g-th group as $x_g$:

  - `quant_mode=0/1` (group quantization, FP8):

    $$
    scale_g = \frac{\max(|x_g|)}{FP8\_MAX}, \quad q_i = \mathrm{round}\left(\frac{x_i}{scale_g}\right)
    $$

    `scale` is stored as `bfloat16` / `float8_e8m0` respectively (when `round_scale=true`, `scale` is rounded up to a power of 2).

  - `quant_mode=2` (rope hifloat8 static + nope FLOAT4_E2M1 dynamic, `FP4\_MAX=6.0`):

    $$
    rope_i = \mathrm{hifloat8}(x_i \cdot x\_scale), \qquad
    scale_g = \frac{\max(|x_g|)}{FP4\_MAX}, \quad nope_i = \mathrm{FLOAT4\_E2M1}\left(\frac{x_i}{scale_g}\right)
    $$

    `scale` is written out as `bfloat16`; `round_scale` does not take effect in this mode.

  - On <term>Atlas A2 products</term> and <term>Atlas A3 products</term>, `quant_mode=1` carries INT8 symmetric dynamic quantization (value range [-128, 127]):

    $$
    scale_g = \frac{\max(|x_g|)}{127}, \quad q_i = \mathrm{RNE}\left(\frac{x_i}{scale_g}\right)
    $$

    `d` is fixed to 512; `scale_g=1.0` when `amax_g=0`; `scale` is written out as `bfloat16` (single RNE rounding); the 64-column rope segment is written with the original `bfloat16` bits; the 18B pad in each row is filled with 0.

  - `quant_mode=3` (MXFP8 whole row, `group=32`, `FP8\_MAX` is 448/57344 depending on the cache dtype):

    $$
    scale_g = \frac{\max(|x_g|, 10^{-4})}{FP8\_MAX}, \quad q_i = \mathrm{round}\left(\frac{x_i}{scale_g}\right)
    $$

    When `round_scale=true`, `scale` is rounded up to a power of 2. The flat row layout is `[dataCol data][scaleCol bf16 scales]` (no pad, tail not written).

  - `quant_mode=4` (MXFP4 packed whole row): the shared exponent scale is built from the maximum exponent in the group (bit-encoded as `bfloat16`, with bit-level fixes for Inf/NaN/0); data is packed as two E2M1 elements per byte (RNE).

- Cache layout (`quant_mode=3/4`): supports 2D flat `[N, concatCol]` (slot is the row index; each row writes only `[data|scale]`, no pad) and 4D paged `[blockNum, blockSize, 1, cacheCol]` (per-token interleaved `[data|scale]` bundle; block tail/stride gaps are not written).

## Parameter Description

| Parameter Name           | Input/Output/Attribute | Description                                                             | Data Type      | Data Format |
| ---------------- | -------------- |----------------------------------------------------------------| ------------- | -------- |
| cache            | Input/Output      | KV Cache of the current layer, updated in place, that is, the quantization target in the formula. When `quant_mode=3`, the dtype is FLOAT8_E4M3FN/FLOAT8_E5M2; otherwise it is UINT8.     | UINT8, FLOAT8_E4M3FN, FLOAT8_E5M2         | ND       |
| x                | Input           | Activation input to be quantized, that is, `x` in the formula. The shape is [bs, d].                                | BFLOAT16      | ND       |
| slot_mapping     | Input           | Index mapping from tokens to cache slots. A value of -1 indicates that the token is skipped. The shape is [bs].              | INT32, INT64  | ND       |
| cache            | Output           | KV Cache after in-place update, which is the same tensor as the input `cache`.                            | Same as input cache         | ND       |
| quant_group_size | Optional attribute       | Quantization group size. The recommended value is 64. In `quant_mode=2`, only 16/32/64 are supported, and `(d-64) % quant_group_size == 0` is required; `quant_mode=3` supports only 32; `quant_mode=4` supports 16/32 and requires `d % quant_group_size == 0`; on <term>Atlas A2 products</term> and <term>Atlas A3 products</term>, `quant_mode=1` requires 64. | INT           | -        |
| quant_mode       | Optional attribute       | Quantization mode: 0=group (bf16 scale), 1=group (e8m0 scale), 2=rope hifloat8 static + nope FLOAT4_E2M1 dynamic, 3=MXFP8 whole row, 4=MXFP4 packed whole row; on <term>Atlas A2 products</term> and <term>Atlas A3 products</term>, 1 carries nope INT8 symmetric dynamic quantization + rope bf16. The recommended value is 1. | INT           | -        |
| round_scale      | Optional attribute       | Whether to round the scale of each group up to a power of 2 in group mode. The recommended value is true. It does not take effect under `quant_mode=2/4` (nor under `quant_mode=1` on <term>Atlas A2 products</term> and <term>Atlas A3 products</term>).                  | BOOL          | -        |
| x_scale          | Optional attribute       | In `quant_mode=2`, the scaling factor for hifloat8 static quantization of the rope segment; reserved for other modes and must be 1.0 under `quant_mode=3/4` (the arch22 INT8 mode ignores it as well). The recommended value is 1.0.     | FLOAT         | -        |

## Constraints

- `quant_mode=0/1/2`: `cache` supports only the 4D `[blockNum, blockSize, 1, headDim]` (num_slots = blockNum × blockSize). The second-to-last dimension is fixed to 1; non-contiguous layout is supported only in the blockNum dimension. headDim must be ≥ kvCacheCol, the number of bytes written per row. The last dimension (d axis) of `x` must satisfy `d % 64 == 0` and `64 < d ≤ 8192`.
- `quant_mode=3/4`: `cache` supports 2D `[N, concatCol]` (flat; slot is the row index; the last axis must be ≥ `concatCol`; no 32B rounding requirement) or 4D `[blockNum, blockSize, 1, cacheCol]` (paged; the last axis must be ≥ `concatCol`; the block stride is derived from the cache descriptor view stride, and is `blockSize × cacheCol` when contiguous). The last dimension of `x` must satisfy `0 < d ≤ 8192`, `d % quant_group_size == 0`, and `d % 128 == 0`. `quant_mode=3` requires the cache dtype to be float8_e4m3fn/float8_e5m2 and group=32; `quant_mode=4` requires the cache dtype to be uint8 and group=16/32.
- For `quant_mode=3/4` with a 4D paged cache, **non-contiguous blocks are supported**: the block stride is taken from view stride(0) and must be no less than `blockSize × cacheCol`; 32B alignment is recommended but no longer mandatory. Tokens within a block are packed as `[data|scale]` bundles, and gaps between blocks retain their original values.
- In `quant_mode=2`, `quant_group_size` supports only 16/32/64, and the length of the nope segment `(d-64)` must be divisible by `quant_group_size`; `x` must be `bfloat16`.
- On <term>Atlas A2 products</term> and <term>Atlas A3 products</term>, `quant_mode=1` carries INT8 quantization: `d` must be 512, `quant_group_size` must be 64, and `headDim` must be ≥ 608 (kvCacheCol=608); `round_scale`/`x_scale` do not take effect. On <term>Ascend 950PR&950DT products</term>, `quant_mode=1` performs FP8 (e8m0) quantization; other `quant_mode` values on <term>Atlas A2 products</term> and <term>Atlas A3 products</term> report an error.
- The dimensions of `slot_mapping` should be the dimensions of `x` minus 1, that is, `slot_mapping` is `x` flattened over all dimensions except the last dimension.
- Tokens with a value of -1 in `slot_mapping` are skipped. The value range of other valid elements is [0, num_slots - 1] (out-of-range values are silently skipped for `quant_mode=3/4`). Element values must be unique; if they are duplicated, the cross-core write order is not guaranteed and the correctness of the result is not guaranteed.

## Calling Instructions

| Calling Method  | Sample Code                                                                       | Description                                                                        |
| --------- | ------------------------------------------------------------------------------ | --------------------------------------------------------------------------- |
| aclnn API | [test_aclnn_kv_compress_epilog](./examples/test_aclnn_kv_compress_epilog.cpp)   | Calls the KvCompressEpilog operator through [aclnnKvCompressEpilog](docs/aclnnKvCompressEpilog_en.md). |
| Graph mode    | -                                                                              | Calls the KvCompressEpilog operator by connecting to the GE graph mode through the [operator IR](./op_graph/kv_compress_epilog_proto.h). |
| PyTorch API | - | Calls the kv_compress_epilog operator through the [kv_compress_epilog](docs/torchapi_kv_compress_epilog_en.md) API. |
