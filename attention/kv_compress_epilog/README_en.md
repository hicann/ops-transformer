# KvCompressEpilog

English | [简体中文](./README.md)

## Product Support

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>     |    ×     |
| <term>Atlas A2 products</term>     |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                              |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Description

- Operator function: This operator performs in-place compression update of the cache in the Epilog stage of the KV Cache. After the operator quantizes and compresses the `bfloat16` activation value `x`, it scatters the results to `cache` according to `slot_mapping`. Tokens with a value of -1 in `slot_mapping` are skipped. The last 64 columns of the tail axis `d` of `x` are the rope segment, and the first `d-64` columns are the nope segment. Three quantization modes are supported:
  - `quant_mode=0/1`: `x` is dynamically quantized group by group (Group-wise Dynamic Quantization) and compressed to FP8 (packed as `uint8`), and the rope segment is kept as `bfloat16`;
  - `quant_mode=2`: the rope segment undergoes hifloat8 **static** quantization (multiplied by `x_scale` and then converted to hifloat8), and the nope segment undergoes per-group **FLOAT4_E2M1 dynamic** quantization (`scale=amax in the group/6`, written out as `bfloat16`).

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

## Parameter Description

| Parameter Name           | Input/Output/Attribute | Description                                                             | Data Type      | Data Format |
| ---------------- | -------------- |----------------------------------------------------------------| ------------- | -------- |
| cache            | Input/Output      | KV Cache of the current layer, updated in place, that is, the quantization target in the formula.     | UINT8         | ND       |
| x                | Input           | Activation input to be quantized, that is, `x` in the formula. The shape is [bs, d].                                | BFLOAT16      | ND       |
| slot_mapping     | Input           | Index mapping from tokens to cache slots. A value of -1 indicates that the token is skipped. The shape is [bs].              | INT32, INT64  | ND       |
| cache            | Output           | KV Cache after in-place update, which is the same tensor as the input `cache`.                            | UINT8         | ND       |
| quant_group_size | Optional attribute       | Quantization group size. The recommended value is 64. In `quant_mode=2`, only 16/32/64 are supported, and `(d-64) % quant_group_size == 0` is required. | INT           | -        |
| quant_mode       | Optional attribute       | Quantization mode: 0=group (bf16 scale), 1=group (e8m0 scale), 2=rope hifloat8 static + nope FLOAT4_E2M1 dynamic. The recommended value is 1. | INT           | -        |
| round_scale      | Optional attribute       | Whether to round the scale of each group up to a power of 2 in group mode. The recommended value is true. It does not take effect under `quant_mode=2`.                  | BOOL          | -        |
| x_scale          | Optional attribute       | In `quant_mode=2`, the scaling factor for hifloat8 static quantization of the rope segment; reserved and unused under `quant_mode=0/1`. The recommended value is 1.0.     | FLOAT         | -        |

## Constraints

- `cache` supports only the 4D `[blockNum, blockSize, 1, headDim]` (num_slots = blockNum × blockSize). The second-to-last dimension is fixed to 1; non-contiguous layout is supported only in the blockNum dimension. headDim must be ≥ kvCacheCol, the number of bytes written per row.
- The dimensions of `slot_mapping` should be the dimensions of `x` minus 1, that is, `slot_mapping` is `x` flattened over all dimensions except the last dimension.
- The last dimension (d axis) of `x` must satisfy `d % 64 == 0` and `64 < d ≤ 8192`, and group-wise quantization is performed with every 64 elements as a group.
- In `quant_mode=2`, `quant_group_size` supports only 16/32/64, and the length of the nope segment `(d-64)` must be divisible by `quant_group_size`; `x` must be `bfloat16`.
- Tokens with a value of -1 in `slot_mapping` are skipped. The value range of other valid elements is [0, num_slots - 1], and the element values must be unique. If they are duplicated, the correctness of the result is not guaranteed.

## Calling Instructions

| Calling Method  | Sample Code                                                                       | Description                                                                        |
| --------- | ------------------------------------------------------------------------------ | --------------------------------------------------------------------------- |
| aclnn API | [test_aclnn_kv_compress_epilog](./examples/test_aclnn_kv_compress_epilog.cpp)   | Calls the KvCompressEpilog operator through [aclnnKvCompressEpilog](docs/aclnnKvCompressEpilog_en.md). |
| Graph mode    | -                                                                              | Calls the KvCompressEpilog operator by connecting to the GE graph mode through the [operator IR](./op_graph/kv_compress_epilog_proto.h). |
| PyTorch API | - | Calls the kv_compress_epilog operator through the [kv_compress_epilog](docs/torchapi_kv_compress_epilog_en.md) API. |
