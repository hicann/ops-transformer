# UndGenQkvRmsNormRopeCache

English | [简体中文](./README.md)

## Product Support

| Product                                                      | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |    √     |
| <term>Atlas A3 products</term>                               |    ×     |
| <term>Atlas A2 products</term>                               |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>            |    ×     |
| <term>Atlas inference products</term>                        |    ×     |
| <term>Atlas training products</term>                         |    ×     |

## Description

- Operator function: A fused Q/K/V backbone operator for the prefill/decode stages of multimodal large model inference. It assembles the output sequence by indirectly addressing the undecoded segment (und_qkv) and the generated segment (gen_qkv) according to `cat_indices`, and performs the following for each token:

  1. **Split**: Splits each token into Q[Hq, D] / K[Hk, D] / V[Hv, D] along the N dimension;
  2. **RMSNorm**: Normalizes Q/K using the und/gen weight sets respectively (selected by `src_t < und_len`);
  3. **MRoPE**: Obtains the three-axis (time/height/width) cos_sin values according to `positions[3, T]`, merges them into one set based on `mrope_section`, and then applies standard RoPE to Q/K; V does not participate in norm/rope;
  4. **Output**: Q is provided as an independent output; K/V are cast from float32 to bf16 and then written into the paged KV Cache according to `slot_mapping`.

- The inputs and outputs support the following data scenario:

    ```text
    und_qkv:       [und_len, N, D]           N = Hq + Hk + Hv
    gen_qkv:       [gen_len, N, D]           Currently required, gen_len > 0
    und_weights_q: [D]   und_weights_k: [D]
    gen_weights_q: [D]   gen_weights_k: [D]  Optional, provided together with gen_qkv
    cos_sin_cache: [max_pos, D]              First half cos, second half sin
    slot_mapping:  [T]                       slot = block_idx * Bs + row_idx
    positions:     [3, T]
    cat_indices:   [T]                       Currently required (the identity mapping path without it is not supported yet)
    k_cache:       [Bn, Bs, Hk, D]           Pre-allocated by the caller, written in place
    v_cache:       [Bn, Bs, Hv, D]           Pre-allocated by the caller, written in place
    q:             [T, Hq, D]                T = und_len + gen_len
    ```

- **The KV Cache layout is fixed to contiguous BBND** (`[Bn, Bs, N, D]`). Non-contiguous/BNBD physical layouts are not supported.
- Index tensors (`slot_mapping` / `positions` / `cat_indices`) all use int64.
- When `mrope_section` is an empty list, it degrades to `[D/2, 0, 0]`, which is equivalent to standard RoPE.

## Parameter Description

| Parameter Name | Input/Output | Data Type | Data Format | Description |
| :------------- | :----------- | :-------- | :---------- | :---------- |
| und_qkv | Input | BFLOAT16 | ND | QKV of the undecoded segment, [und_len, N, D] |
| und_weights_q | Input | BFLOAT16 | ND | RMSNorm weight of Q in the und segment, [D] |
| und_weights_k | Input | BFLOAT16 | ND | RMSNorm weight of K in the und segment, [D] |
| cos_sin_cache | Input | FLOAT32 | ND | cos/sin cache, [max_pos, D] |
| k_cache | Input/Output | BFLOAT16 | ND | K Cache, [Bn, Bs, Hk, D], written in place |
| v_cache | Input/Output | BFLOAT16 | ND | V Cache, [Bn, Bs, Hv, D], written in place |
| slot_mapping | Input | INT64 | ND | Cache row offset of each token, [T] |
| positions | Input | INT64 | ND | MRoPE three-axis positions, [3, T] |
| gen_qkv | Optional input (currently required) | BFLOAT16 | ND | QKV of the generated segment, [gen_len, N, D], gen_len > 0 |
| gen_weights_q | Optional input (currently required) | BFLOAT16 | ND | RMSNorm weight of Q in the gen segment, [D] |
| gen_weights_k | Optional input (currently required) | BFLOAT16 | ND | RMSNorm weight of K in the gen segment, [D] |
| cat_indices | Optional input (currently required) | INT64 | ND | out_t → src_t mapping, [T] |
| q | Output | BFLOAT16 | ND | Processed Q, [T, Hq, D] |
| num_heads_q | Attribute | INT | - | Number of Q heads Hq |
| num_heads_k | Attribute | INT | - | Number of K heads Hk |
| num_heads_v | Attribute | INT | - | Number of V heads Hv |
| norm_eps | Attribute | FLOAT | - | RMSNorm epsilon, default 1e-6 |
| mrope_section | Attribute | LIST_INT | - | MRoPE three-axis sections, default [] |

## Constraints

- Only <term>Ascend 950PR&950DT products</term> are supported (DAV_3510, arch35).
- The entire computation uses float32, and the KV Cache is stored in BF16. FP8/INT8 quantization is not supported.
- headDim is fixed to 128.
- (Hq, Hk, Hv) supports the two combinations (8, 1, 1) and (16, 2, 2).
- The cache block_size has no limit: Bs does not participate in any addressing or splitting (the cache is forced to be contiguous BBND, and a slot is directly used as a flattened row number). The only constraint is the capacity. Bs ∈ {16, 64, 100, 128, 256, 512}, including non-powers of 2, has been verified on the device.
- T = und_len + gen_len has no upper limit, and it only needs to be positive. The actual upper limit is the KV Cache capacity (`Bn * Bs >= T` is required). T = 64K has been verified on the device.
- The values in `slot_mapping` must be unique and within [0, Bn*Bs). Duplicate slots lead to non-deterministic results in multi-core scenarios.
- The values in `positions` must be within [0, max_pos), and the values in `cat_indices` must be within [0, T). These three types of values are runtime data. The host side cannot see the tensor content and cannot validate them, so the caller must guarantee them.
- Input parameter validation is unified at the tiling layer (aclnn L2 checks only null pointers, and InferShape checks only inference prerequisites). Therefore, invalid inputs do not cause errors in the `aclnnUndGenQkvRmsNormRopeCacheGetWorkspaceSize` phase; instead, tiling returns a failure in the execution phase.
- **The current implementation requires all four optional inputs `gen_qkv`, `gen_weights_q`, `gen_weights_k`, and `cat_indices` to be provided, and `gen_len > 0`**. The degraded path with missing inputs (pure prefill, single-sequence identity mapping) is not supported yet, and tiling directly blocks it. The OPTIONAL declarations in IR/OpDef are retained. When the restriction is lifted later, you only need to delete the corresponding checks in `CheckSupportRange`.
- `k_cache`/`v_cache` must be memory-contiguous.

## Calling Instructions

| Calling Method | Calling Sample | Description |
| :------------- | :------------- | :---------- |
| PyTorch API | [test_torch_und_gen_qkv_rms_norm_rope_cache.py](./examples/test_torch_und_gen_qkv_rms_norm_rope_cache.py) | Calls the UndGenQkvRmsNormRopeCache operator through the [und_gen_qkv_rms_norm_rope_cache](docs/torchapi_und_gen_qkv_rms_norm_rope_cache_en.md) API. |
| aclnn call | [test_aclnn_und_gen_qkv_rms_norm_rope_cache.cpp](./examples/test_aclnn_und_gen_qkv_rms_norm_rope_cache.cpp) | Calls the UndGenQkvRmsNormRopeCache operator through the [aclnnUndGenQkvRmsNormRopeCache](docs/aclnnUndGenQkvRmsNormRopeCache_en.md) API. Two-phase invocation; the sample includes slot_mapping precomputation. |
| Graph mode call | [test_geir_und_gen_qkv_rms_norm_rope_cache.cpp](./examples/test_geir_und_gen_qkv_rms_norm_rope_cache.cpp) | Calls the UndGenQkvRmsNormRopeCache operator by building a graph through the [operator IR](./op_graph/und_gen_qkv_rms_norm_rope_cache_proto.h). |
