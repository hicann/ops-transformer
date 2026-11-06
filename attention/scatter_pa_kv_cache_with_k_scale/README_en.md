# ScatterPaKvCacheWithKScale

English | [简体中文](./README.md)

## Product Support

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                 |    √     |
| <term>Atlas A3 products</term> |    ×     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term> |      ×     |
| <term>Atlas inference products</term> |      ×     |
| <term>Atlas training products</term> |      ×     |

## Function Description

- Operator function: updates the key and value at specified positions in the KvCache, and also updates the scale value of the key.

- The following input and output scenarios are supported:
  - Scenario 1:

    ```python
    key:[batch * seq_len, num_head, head_size]
    value:[batch * seq_len, num_head, head_size]
    key_cache:[num_blocks, num_head, block_size, head_size]
    value_cache:[num_blocks, num_head, block_size, head_size]
    slot_mapping:[batch * seq_len]
    key_scale:[batch * seq_len, num_head]
    key_scale_cache:[num_blocks, num_head, block_size, 1]
    cache_layout:"BNBD"
    ```

    The dtypes of key and value are FLOAT8_E5M2 or FLOAT8_E4M3FN, and the dtypes of key_scale and key_scale_cache are FLOAT.

    Computation formula:

    For each token (i ∈ [0, num_tokens)) and each head (j ∈ [0, num_head)):

    ```python
    block_idx = slot_mapping[i] // block_size
    block_offset = slot_mapping[i] % block_size

    key_cache[block_idx][j][block_offset][:] = key[i][j][:]
    value_cache[block_idx][j][block_offset][:] = value[i][j][:]
    key_scale_cache[block_idx][j][block_offset][0] = key_scale[i][j]
    ```

    Where:
    - num_tokens = batch * seq_len
    - block_idx: index of the block to which slot_mapping is mapped
    - block_offset: offset within the block

- <term>Ascend 950PR&950DT products</term>: support scenario 1 only.

## Parameter Description

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter Name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data Type</th>
      <th>Data Format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>key</td>
      <td>Input</td>
      <td>Key values to be updated, which are the keys of multiple tokens in the current step.</td>
      <td>FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value</td>
      <td>Input</td>
      <td>Value to be updated, which is the value of multiple tokens in the current step.</td>
      <td>FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key_cache</td>
      <td>Input/Output</td>
      <td>Key cache to be updated, which is the key cache of the current layer.</td>
      <td>Same as key</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value_cache</td>
      <td>Input/Output</td>
      <td>Value cache to be updated, which is the value cache of the current layer.</td>
      <td>Same as value</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>slot_mapping</td>
      <td>Input</td>
      <td>Storage offset of the key or value of each token in the cache.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key_scale</td>
      <td>Input</td>
      <td>Key scale values to be updated, which are the key scales of multiple tokens in the current step.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key_scale_cache</td>
      <td>Input/Output</td>
      <td>Key scale cache to be updated, which is the key scale cache of the current layer.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cache_layout</td>
      <td>Attribute</td>
      <td>Indicates the memory layout format of key_cache and value_cache. When a null pointer or "BNBD" is passed, the format is [num_blocks, num_head, block_size, head_size].</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- Deterministic computation:
  - The implementation is deterministic by default.
- The data types of key, value, key_cache, and value_cache must be the same;
- The value range of slot_mapping is [0, num_blocks*block_size-1]. The values of elements in slot_mapping must be unique. If values are duplicated, correctness is not guaranteed;
- The shapes of the first two dimensions of key and value must be the same;
- key_scale is a 2-dimensional tensor with the shape [batch * seq_len, num_head]. The trailing axis can be non-contiguous;
- key_scale_cache is a 4-dimensional tensor with the shape [num_blocks, num_head, block_size, 1]. The last dimension must be 1, and the trailing axis must be contiguous.

## Calling Instructions

| Calling Method | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn API | [test_aclnn_ScatterPaKvCacheWithKScale](./examples/test_aclnn_scatter_pa_kv_cache_with_k_scale.cpp) | Calls the ScatterPaKvCacheWithKScale operator through [aclnnScatterPaKvCacheWithKScale](docs/aclnnScatterPaKvCacheWithKScale_en.md) |
| Graph mode | [test_geir_ScatterPaKvCacheWithKScale](./examples/test_geir_scatter_pa_kv_cache_with_k_scale.cpp) | Calls the ScatterPaKvCacheWithKScale operator through [operator IR](./op_graph/scatter_pa_kv_cache_with_k_scale_proto.h) |
| PyTorch API | - | Calls the scatter_pa_kv_cache_with_k_scale operator through the [scatter_pa_kv_cache_with_k_scale](docs/torchapi_scatter_pa_kv_cache_with_k_scale_en.md) API. |
