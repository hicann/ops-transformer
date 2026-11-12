# IndexerQuantCache

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

- Operator function: Compresses and updates the KV Cache in place during the Epilog phase of the Indexer attention mechanism. The operator compresses the `float16`/`bfloat16` activation value `x` into FP8 (E4M3/E5M2), HiFloat8 (floating-point codes stored in UINT8), MX-FP4, or HiFloat4 format based on the quantization mode, and scatters the quantization results to `cache` based on `slot_mapping`. Except in HiFloat8 mode, the corresponding scale is also written to `cache_scale`. Tokens with a value of -1 in `slot_mapping` are skipped. Five quantization modes are supported: MX-FP8, Normal, HiFloat8, MX-FP4, and HiFloat4.

- Computation formula:

  Quantization is performed on the last dimension of `x` (the d dimension). In MX-FP8 and MX-FP4 modes, elements are grouped every 32 elements. In Normal mode, each row (one token) is a group and the entire row shares one scale. The quantization group is denoted as $x_g$:

  $$
  scale_g = \frac{\max(|x_g|)}{Q\_MAX}, \quad q_i = \mathrm{round}\left(\frac{x_i}{scale_g}\right)
  $$

  - `quant_mode=0` (MX-FP8): The scale is stored as `float8_e8m0`, and is always rounded up to a power of 2, independently of `round_scale`.
  - `quant_mode=1` (Normal): Dynamic quantization is performed row by row. One scale is computed for each row and stored as `float32`. When `round_scale=true`, the positive scale computed by the formula above is rounded up to `2^ceil(log2(scale))` for quantization and writeback.
  - `quant_mode=2` (HiFloat8): $x \times x\_scale$ is converted to hifloat8, and `cache_scale` is neither computed nor written back.
  - `quant_mode=3` (MX-FP4): The values are quantized to FP4 in standard MX blocks (every 32 elements) (uint8 packs 2 fp4 values per byte), and the scale is stored as `float8_e8m0`.

  - `quant_mode=4` (HiFloat4): Supported only on ascend960dt (DAV_9201). Requires `d % 64 == 0`, HIFLOAT4 cache, and HIFLOAT4_SCALE cache_scale, with d/64 scale elements per row.

## Parameter Description

| Parameter       | Input/Output/Attribute | Description                                                                          | Data Type                                                      | Data Format |
| ------------ | -------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------ | -------- |
| cache        | Input/Output      | KV Cache vector cache of the current layer, updated in place, that is, the quantization target in the formula. In MX-FP4 mode, it is uint8, with 2 fp4 values packed per byte. | FLOAT8_E4M3FN, FLOAT8_E5M2, UINT8, FLOAT4_E2M1, FLOAT4_E1M2, HIFLOAT4   | ND       |
| cache_scale  | Input/Output      | Quantization scale tensor. In HiFloat8 mode, this input remains unchanged and is not written back.                        | FLOAT, FLOAT8_E8M0, HIFLOAT4_SCALE                             | ND       |
| x            | Input           | Activation input to be quantized, that is, `x` in the formula. The shape is [bs, d].                              | FLOAT16, BFLOAT16                                             | ND       |
| slot_mapping | Input           | Index mapping from token to cache slot. A value of -1 indicates that the token is skipped. The shape is [bs].       | INT32                                                        | ND       |
| cache        | Output           | KV Cache updated in place, which is the same tensor as the input `cache`.                            | FLOAT8_E4M3FN, FLOAT8_E5M2, UINT8, FLOAT4_E2M1, FLOAT4_E1M2, HIFLOAT4   | ND       |
| cache_scale  | Output           | Scale written back in place in MX-FP8, Normal, MX-FP4, and HiFloat4 modes. In HiFloat8 mode, the input value remains unchanged.   | FLOAT, FLOAT8_E8M0, HIFLOAT4_SCALE                             | ND       |
| quant_mode   | Optional attribute       | Quantization mode: 0=MX-FP8, 1=Normal, 2=HiFloat8, 3=MX-FP4, 4=HiFloat4. Default value: 1.                | INT                                                          | -        |
| round_scale  | Optional attribute       | Whether to round the scale up to a power of 2, only in Normal mode (quant_mode=1). Default value: true.              | BOOL                                                         | -        |
| x_scale      | Optional attribute       | Global scale multiplier in HiFloat8 mode (quant_mode=2). Default value: 1.0.                 | FLOAT                                                        | -        |

## Constraints

- The last dimension `d` of `x` must be divisible by 32 and `d ≤ 8192`.
- The dimensions of `slot_mapping` must equal the dimensions of `x` minus 1, that is, `slot_mapping` is the flattening of all dimensions of `x` except the last dimension.
- The data type combination must match the quantization mode: In Normal/HiFloat8 mode, cache is FP8/UINT8 and cache_scale is FLOAT. In MX-FP8 mode, cache is FP8 and cache_scale is FLOAT8_E8M0. In MX-FP4 mode, cache is FP4 and cache_scale is FLOAT8_E8M0.
- Tokens with a value of -1 in `slot_mapping` are skipped. The value range of other valid elements is [0, num_slots - 1], and element values must not repeat. Correctness of results is not guaranteed when duplicates exist.

## Calling Instructions

| Calling Method  | Sample Code                                                                          | Description                                                                          |
| --------- | --------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| aclnn API | [test_aclnn_indexer_quant_cache](./examples/test_aclnn_indexer_quant_cache.cpp)    | Calls the IndexerQuantCache operator through [aclnnIndexerQuantCache](docs/aclnnIndexerQuantCache_en.md). |
| Graph mode    | -                                                                                 | Calls the IndexerQuantCache operator by connecting to the GE graph mode through the [operator IR](./op_graph/indexer_quant_cache_proto.h). |
| PyTorch API | - | Calls the indexer_quant_cache operator through the [indexer_quant_cache](docs/torchapi_indexer_quant_cache_en.md) API. |
