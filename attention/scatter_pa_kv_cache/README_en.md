# ScatterPaKvCache

English | [简体中文](./README.md)

## Product Support

| Product                                                         | Support Status |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                 |    √     |
| <term>Atlas A3 products</term> |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term> |      ×     |
| <term>Atlas inference products</term> |      ×     |
| <term>Atlas training products</term> |      ×     |

## Description

- Operator function: Updates the key and value at the specified positions in KvCache.

- The input and output scenarios are distinguished by the constructed parameters. The inputs and outputs support the following scenarios. Scenario 1, Scenario 2, and Scenario 6 do not include the three optional parameters compressLensOptional, seqLensOptional, and compressSeqOffsetOptional, and Scenario 4 does not include the optional parameter compressSeqOffsetOptional:

  - Scenario 1:

    ```python
    key:[batch * seq_len, num_head, k_head_size]
    value:[batch * seq_len, num_head, v_head_size]
    keyCache:[num_blocks, num_head * k_head_size // last_dim_k, block_size, last_dim_k]/[num_blocks, num_head, k_head_size // last_dim_k, block_size, last_dim_k]
    valueCache:[num_blocks, num_head * v_head_size // last_dim_v, block_size, last_dim_v]/[num_blocks, num_head, v_head_size // last_dim_v, block_size, last_dim_v]
    slotMapping:[batch * seq_len]
    cacheMode:"PA_NZ"

    last_dim_k = 32 / sizeof(dtypeKey)
    last_dim_v = 32 / sizeof(dtypeValue)
    (k_head_size * sizeof(dtypeKey)) % 32 == 0
    (v_head_size * sizeof(dtypeValue)) % 32 == 0
    ```

  - Scenario 2:

    ```python
    key:[batch * seq_len, num_head, k_head_size]
    value:[batch * seq_len, num_head, v_head_size]
    keyCache:[num_blocks, block_size, num_head, k_head_size]
    valueCache:[num_blocks, block_size, num_head, v_head_size]
    slotMapping:[batch * seq_len]
    cacheMode:"Norm"
    scatter_mode:"None"/"Nct"
    ```

    k_head_size and v_head_size can differ or be the same.

  - Scenario 3:

    ```python
    key:[batch, seq_len, num_head, k_head_size]
    value:[batch, seq_len, num_head, v_head_size]
    keyCache:[num_blocks, block_size, 1, k_head_size]
    valueCache:[num_blocks, block_size, 1, v_head_size]
    slotMapping:[batch, num_head]
    compressLensOptional:[batch, num_head]
    seqLensOptional:[batch]
    compressSeqOffsetOptional:[batch * num_head]
    cacheMode:"Norm"
    ```

  - Scenario 4:

    ```python
    key:[num_tokens, num_head, k_head_size]
    value:[num_tokens, num_head, v_head_size]
    keyCache:[num_blocks, block_size, 1, k_head_size]
    valueCache:[num_blocks, block_size, 1, v_head_size]
    slotMapping:[batch * num_head]
    compressLensOptional:[batch * num_head]
    seqLensOptional:[batch]
    cacheMode:"Norm"
    scatter_mode:"Alibi"
    ```

  - Scenario 5:

    ```python
    key:[num_tokens, num_head, k_head_size]
    value:[num_tokens, num_head, v_head_size]
    keyCache:[num_blocks, block_size, 1, k_head_size]
    valueCache:[num_blocks, block_size, 1, v_head_size]
    slotMapping:[batch * num_head]
    compressLensOptional:[batch * num_head]
    seqLensOptional:[batch]
    compressSeqOffsetOptional:[batch * num_head]
    cacheMode:"Norm"
    scatter_mode:"Rope"/"Omni"
    ```

  - Scenario 6:

    ```python
    key:[batch * seq_len, num_head, k_head_size]
    value:[]
    keyCache:[num_blocks, block_size, num_head, k_head_size]
    valueCache:[]
    slotMapping:[batch * seq_len]
    cacheMode:"Norm"
    scatter_mode:"None"/"Nct"
    ```

  - Scenario 7:

    ```python
    key:[num_tokens, num_head, k_head_size]
    value:[num_tokens, num_head, v_head_size]
    keyCache:[num_blocks, num_head, block_size, k_head_size]
    valueCache:[num_blocks, num_head, block_size, v_head_size]
    slotMapping:[num_tokens]
    cacheMode:"Norm"
    scatter_mode:"NHSD"
    ```

- <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: Only Scenarios 1, 2, 4, 5, 6, and 7 are supported.
- <term>Ascend 950PR&950DT products</term>: Only Scenarios 1, 2, 3, 4, and 5 are supported.

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
      <td>Key value to be updated, the key of multiple tokens in the current step.</td>
      <td>FLOAT16, FLOAT, BFLOAT16, INT8, UINT8, INT16, UINT16, INT32, UINT32, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E1M2, FLOAT4_E2M1</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>keyCacheRef</td>
      <td>Input/Output</td>
      <td>Key cache to be updated, the key cache of the current layer.</td>
      <td>FLOAT16, FLOAT, BFLOAT16, INT8, UINT8, INT16, UINT16, INT32, UINT32, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E1M2, FLOAT4_E2M1</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>slotMapping</td>
      <td>Input</td>
      <td>Storage offset of the key or value of each token in the cache.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value</td>
      <td>Input</td>
      <td>Value to be updated, the value of multiple tokens in the current step.</td>
      <td>FLOAT16, FLOAT, BFLOAT16, INT8, UINT8, INT16, UINT16, INT32, UINT32, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E1M2, FLOAT4_E2M1</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>valueCacheRef</td>
      <td>Input/Output</td>
      <td>Value cache to be updated, the value cache of the current layer.</td>
      <td>FLOAT16, FLOAT, BFLOAT16, INT8, UINT8, INT16, UINT16, INT32, UINT32, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E1M2, FLOAT4_E2M1</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>compressLensOptional</td>
      <td>Optional Input</td>
      <td>Compression length.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>compressSeqOffsetOptional</td>
      <td>Optional Input</td>
      <td>Compression start point of each head in each batch.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqLensOptional</td>
      <td>Optional Input</td>
      <td>Actual seqLens of each batch.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cacheMode</td>
      <td>Input</td>
      <td>Indicates the memory layout format of keyCacheRef and valueCacheRef.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scatterMode</td>
      <td>Input</td>
      <td>Indicates the state of the updated key and value.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>Input</td>
      <td>Strides of key and value in the non-contiguous state, with an array length of 2. The value must be greater than 0.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>offsets</td>
      <td>Input</td>
      <td>Offsets of key and value in the non-contiguous state, with an array length of 2. The value must be greater than or equal to 0.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
  </tbody></table>

- Support ranges of the data types of key/value/keyCacheRef/valueCacheRef for each product:
  - <term>Ascend 950PR&950DT products</term>: All data types listed in the parameter table are supported; among them, FLOAT4_E1M2 and FLOAT4_E2M1 are supported only in Scenario 1 and Scenario 2 where scatterMode is "None".
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: Only FLOAT16, BFLOAT16, and INT8 are supported.

## Constraints

  * Input shape restrictions:
      * Except key and value, the input parameters do not support non-contiguous inputs.
      * When both key and value are three-dimensional, the shapes of the first two dimensions of key and value must be the same.
      * When both key and value are four-dimensional, the shapes of the first three dimensions of key and value must be the same, and the third dimension of keyCacheRef and valueCacheRef must be 1.
      * When key and value are four-dimensional, compressLensOptional and seqLensOptional are required parameters; when key and value are three-dimensional, compressLensOptional, compressSeqOffsetOptional, and seqLensOptional are optional parameters.
      * When cacheMode is "PA_NZ", the penultimate dimension of keyCacheRef and valueCacheRef must be less than UINT16_MAX (corresponding to Scenario 1).
      * The 32-byte alignment constraints of k_head_size and v_head_size are as follows:
        * Scenario 7: `(k_head_size * sizeof(dtypeKey)) % 32 == 0` and `(v_head_size * sizeof(dtypeValue)) % 32 == 0` must be satisfied.
        * Scenario 4 and Scenario 5: <term>Atlas A2 products</term> and <term>Atlas A3 products</term> must satisfy the preceding 32-byte alignment constraints.
      * num_head must be less than 4095 (corresponding to Scenario 7).
  * Input value range restrictions:
      * The value range of slotMapping is [0,num_blocks*block_size-1], and the element values in slotMapping must be unique; correctness is not guaranteed when values are duplicated.
      * When both key and value are four-dimensional, slotMapping is two-dimensional; the first dimension of slotMapping equals the first dimension of key, which is batch, and the second dimension of slotMapping equals the third dimension of key, which is num_head (corresponding to Scenario 3).
      * When both key and value are four-dimensional, seqLensOptional is one-dimensional, and the value of seqLensOptional equals the first dimension of key, which is batch (corresponding to Scenario 3).
      * When key and value are three-dimensional and seqLensOptional exists, the sum of all values in seqLensOptional equals the first dimension of key, which is num_blocks (corresponding to Scenario 4 and Scenario 5).
      * Each element value in seqLensOptional and compressLensOptional must satisfy the formula: reduceSum(seqLensOptional[i] - compressLensOptional[i]) <= num_blocks * block_size (corresponding to Scenarios 3, 4, and 5).
      * block_size *k_head_size and block_size* v_head_size must be less than UINT32_MAX (corresponding to Scenario 7).
  * Input attribute restrictions:
      * The data types of key, value, keyCacheRef, and valueCacheRef must be consistent.
      * When cacheModeOptional is "PA_NZ", the data types of key and keyCacheRef, and of value and valueCacheRef, can be inconsistent.
      * The data types of slotMapping, compressLensOptional, compressSeqOffsetOptional, and seqLensOptional must be consistent.
      * When cacheMode is a null pointer or "Norm", only the ND memory layout format is supported; when it is "PA_NZ", only the FRACTAL_NZ memory layout format is supported.
      * When scatterMode is a null pointer or "None", the updated key and value are uncompressed and contiguous; when it is "Alibi", the updated key and value are in a compressed state based on the Alibi structure; when it is "Rope", the updated key and value are in a compressed state based on the Rope structure; when it is "Omni", the updated key and value are in a compressed state based on the Omni structure; when it is "Nct", the updated key and value are uncompressed but non-contiguous.
      * strides and offsets take effect only when scatterMode is "Nct"; they represent strideK and strideV, and offsetK and offsetV, respectively.

## Calling Description

| Calling Method  | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn API | [test_aclnn_ScatterPaKvCache](./examples/test_aclnn_scatter_pa_kv_cache.cpp) | Call the ScatterPaKvCache operator through [aclnnScatterPaKvCache](docs/aclnnScatterPaKvCache_en.md) |
| Graph Mode | [test_geir_ScatterPaKvCache](./examples/test_geir_scatter_pa_kv_cache.cpp) | Call the ScatterPaKvCache operator through the [operator IR](./op_graph/scatter_pa_kv_cache_proto.h) |
