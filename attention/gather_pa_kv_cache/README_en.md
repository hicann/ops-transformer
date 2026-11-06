# GatherPaKvCache

English | [简体中文](./README.md)

## Product Support

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                 |    √     |
| <term>Atlas A3 products</term> |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term> |      ×     |
| <term>Atlas inference products</term> |      ×     |
| <term>Atlas training products</term> |      ×     |

## Function Description

- Operator function: Based on the blockId values in blockTables and the seqLen of key/value in seqLens, the operator moves and concatenates the memory-discontinuous tokens from keyCache/valueCache into continuous key/value sequences.

- Example:

  ```python
    keyCache_shape: [128, 128, 16, 144]
    valueCache_shape: [128, 128, 16, 128]
    blockTables_shape: [16, 12]
    seqLens_shape: [16]
    keyRef_shape: [8931, 16, 144]
    valueRef_shape: [8931, 16, 128]
    seqOffset_shape: [16]
    out1_shape: [8931, 16, 144]
    out2_shape: [8931, 16, 128]
  ```

## Formula/Algorithm Description

- The operator locates tokens in `keyCache` and `valueCache` based on the physical block indexes in `blockTables`, and moves the tokens to `keyRef` and `valueRef` in batch order. The operator does not perform numerical computation or data type conversion.
- When `isSeqLensCumsum` is `true`, `seqLens` is a cumulative sum, `seqLens[0]` is 0, the sequence length of the i-th batch is `seqLens[i + 1] - seqLens[i]`, and the total number of output tokens is `seqLens[-1]`.
- When `isSeqLensCumsum` is `false`, `seqLens[i]` is the sequence length of the i-th batch, and the total number of output tokens is `sum(seqLens)`.

## Parameter Description

| Parameter                     | Input/Output/Attribute | Description  | Data Type       | Data Format   |
|----------------------------|-----------|----------------------------------------------------------------------|----------------|------------|
| keyCache                     | Input | Key vector cache stored in the current layer | INT8, FLOAT16, BFLOAT16, FLOAT, UINT8, INT16, UINT16, INT32, UINT32, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN | ND, FRACTAL_NZ |
| valueCache                     | Input | Value vector cache stored in the current layer | INT8, FLOAT16, BFLOAT16, FLOAT, UINT8, INT16, UINT16, INT32, UINT32, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN | ND, FRACTAL_NZ |
| blockTables                     | Input | Mapping between logical blocks and physical blocks of the KV cache in each batch | INT32, INT64       | ND         |
| seqLens                     | Input | Sequence length corresponding to each batch | INT32, INT64       | ND         |
| keyRef                     | Input/Output | Key vector of the current layer | INT8, FLOAT16, BFLOAT16, FLOAT, UINT8, INT16, UINT16, INT32, UINT32, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN       | ND         |
| valueRef                     | Input/Output | Value vector of the current layer | INT8, FLOAT16, BFLOAT16, FLOAT, UINT8, INT16, UINT16, INT32, UINT32, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN       | ND         |
| seqOffset                     | Optional input | First offset that exists when blockTables obtains the blockId; if this parameter is not passed, no first offset exists | INT32, INT64       | ND         |
| cacheMode                     | Input (IR attribute) | Data layout format of the input; supports `Norm` and `PA_NZ`; corresponds to the IR attribute `cache_mode`, with the default value `Norm` | String      | -         |
| isSeqLensCumsum                     | Input (IR attribute) | Whether seqLens is a cumulative sum; corresponds to the IR attribute `is_seq_lens_cumsum`, with the default value `true` | BOOL       | -         |

Note: `keyRef` and `valueRef` are aclnn API parameter names, corresponding to `key` and `value` in the operator IR respectively. The table above lists the union of data types and data formats supported by all products. For product-specific constraints, see the following text.

## Constraints

- When `cacheMode` is `Norm`, the data formats of `keyCache` and `valueCache` are both ND, with shapes `[num_blocks, block_size, num_heads, head_size_k]` and `[num_blocks, block_size, num_heads, head_size_v]` respectively; the shapes of `keyRef` and `valueRef` are `[num_tokens, num_heads, head_size_k]` and `[num_tokens, num_heads, head_size_v]` respectively.
- When `cacheMode` is `PA_NZ`, the data formats of `keyCache` and `valueCache` are both FRACTAL_NZ, with shapes `[num_blocks, num_heads * head_size_k / elenum_aligned, block_size, elenum_aligned]` and `[num_blocks, num_heads * head_size_v / elenum_aligned, block_size, elenum_aligned]` respectively; the shapes of `keyRef` and `valueRef` are `[num_tokens, num_heads * head_size_k]` and `[num_tokens, num_heads * head_size_v]` respectively. The values of `elenum_aligned` corresponding to 8bit, 16bit, and 32bit data types are 32, 16, and 8 respectively.
- The shape of `blockTables` is `[batch, block_indices]`, and the value range of elements is `[0, num_blocks)`; the shape of `seqLens` is one-dimensional. When `isSeqLensCumsum` is `false`, the length of `seqLens` is `batch`; when `isSeqLensCumsum` is `true`, the length of `seqLens` is `batch + 1` and the first element is 0.
- `blockTables` must meet the full coverage condition: for each batch `i`, let `length[i]` be the request sequence length of this batch (`length[i] = seqLens[i]` when `isSeqLensCumsum` is `false`, and `length[i] = seqLens[i + 1] - seqLens[i]` when it is `true`), and let `tableStart[i]` be the start column offset in the table (0 when `seqOffset` is not passed; `seqOffset[i] / block_size` with the quotient of the integer division when it is passed). The conditions `tableStart[i] >= 0` and `tableStart[i] + ceil(length[i] / block_size) <= block_indices` must be met. Then all elements of the output `keyRef`/`valueRef` come from the actual elements in the Cache.
- `seqOffset` may not be passed. When it is passed, its shape is `[batch]` and its data type must be the same as that of `blockTables`.
- The data type of `keyRef` is the same as that of `keyCache`, and the data type of `valueRef` is the same as that of `valueCache`; the data types of `blockTables`, `seqLens`, and `seqOffset` are the same.
- The data volume of a single key or value token does not exceed 148 KiB. For example, in the FLOAT16 or BFLOAT16 scenario, `num_heads * head_size` can be `128 * 576`.
- <term>Atlas A2 products</term> and <term>Atlas A3 products</term>:
  - `cacheMode` supports only `Norm`, and `keyCache` and `valueCache` support only the ND format.
  - `keyCache`, `valueCache`, `keyRef`, and `valueRef` support only INT8, FLOAT16, and BFLOAT16; `blockTables`, `seqLens`, and `seqOffset` support only INT32.
- <term>Ascend 950PR&950DT products</term>:
  - `cacheMode` supports `Norm` and `PA_NZ`; in `Norm` mode, `keyCache` and `valueCache` are both in ND format, and in `PA_NZ` mode, both are in FRACTAL_NZ format.
  - When `keyCache` is FLOAT8_E4M3FN, `valueCache` can also use FLOAT16 or BFLOAT16; in other scenarios, the data types of key and value are the same.
  - When the width of `blockTables` does not meet the preceding coverage condition, the tokens corresponding to logical blocks whose `blockTableOffset` exceeds the range of `block_indices` are filled with zeros before being written to the output, and blocks whose `blockId` is out of range (beyond `[0, num_blocks)`) are also filled with zeros. This behavior is undefined on other product paths, and the caller must ensure that `blockTables` meets the preceding coverage condition.

## Accuracy Description

GatherPaKvCache only moves data in the cache and does not perform numerical computation or data type conversion. When the inputs meet the constraints, each element in `keyRef` and `valueRef` is bitwise identical to its corresponding element in `keyCache` and `valueCache`.

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn API | [test_aclnn_gather_pa_kv_cache](./examples/test_aclnn_gather_pa_kv_cache.cpp) | Invokes the GatherPaKvCache operator through [aclnnGatherPaKvCache](docs/aclnnGatherPaKvCache_en.md) |
