# aclnnScatterPaCache

English | [简体中文](./README.md)

## Product Support

| Product                                                         | Support Status |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    ×     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Description

- Operator function: Updates the key at the specified position in KCache.

- Computation formula:
  - Scenario 1:

    ```python
    key:[batch * seq_len, num_head, k_head_size]
    keyCache:[num_blocks, block_size, num_head, k_head_size]
    slotMapping:[batch * seq_len]
    cacheMode:"Norm"
    ```

    $$
    keyCache = slotMapping(key)
    $$

  - Scenario 2:

    ```python
    key:[batch, seq_len, num_head, k_head_size]
    keyCache:[num_blocks, block_size, 1, k_head_size]
    slotMapping:[batch, num_head]
    compressLensOptional:[batch, num_head]
    compressSeqOffsetOptional:[batch * num_head]
    seqLensOptional:[batch]
    cacheMode:"Norm"
    ```

    $$
    \begin{aligned}
    keyCache =\ & slotMapping(key[: compressSeqOffset], \\
    & ReduceMean(key[compressSeqOffset : compressSeqOffset + compressLens]), \\
    & key[compressSeqOffset + compressLens : seqLens])
    \end{aligned}
    $$

  - Scenario 3:

    ```python
    key:[batch, seq_len, num_head, k_head_size]
    keyCache:[num_blocks, block_size, 1, k_head_size]
    slotMapping:[batch, num_head]
    compressLensOptional:[batch * num_head]
    seqLensOptional:[batch]
    cacheMode:"Norm"
    ```

    $$
    keyCache = slotMapping(key[seqLens - compressLens : seqLens])
    $$

  The preceding scenarios are distinguished by the constructed parameters. If the first input construction is met, Scenario 1 applies; if the second construction is met, Scenario 2 applies; if the third construction is met, Scenario 3 applies. Scenario 1 does not include the three optional parameters compressLensOptional, seqLensOptional, and compressSeqOffsetOptional, and Scenario 3 does not include the optional parameter compressSeqOffsetOptional.

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 312px">
  <col style="width: 213px">
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
      <td>Key value to be updated, which is key in the formula.</td>
      <td>FLOAT16, FLOAT, BFLOAT16, INT8, UINT8, INT16, UINT16, INT32, UINT32, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1, FLOAT4_E1M2</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>keyCacheRef</td>
      <td>Input/Output</td>
      <td>keyCache to be updated, which is keyCache in the formula.</td>
      <td>Consistent with key.</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>slotMapping</td>
      <td>Input</td>
      <td>Storage offset of each token of key in the cache, which is slotMapping in the formula.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>compressLensOptional</td>
      <td>Optional Input</td>
      <td>Compression length, which is compressLens in the formula.</td>
      <td>Consistent with slotMapping.</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>compressSeqOffsetOptional</td>
      <td>Optional Input</td>
      <td>Compression start point of each head in each batch, which is compressSeqOffset in the formula.</td>
      <td>Consistent with slotMapping.</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqLensOptional</td>
      <td>Optional Input</td>
      <td>Actual seqLens of each batch, which is seqLens in the formula.</td>
      <td>Consistent with slotMapping.</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cacheMode</td>
      <td>Input</td>
      <td>Memory layout format of keyCacheRef.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- The variables used in the shapes in Parameter Description are described as follows:
  - batch: number of currently input sequences (number of samples processed at a time); the value is a positive integer.
  - seq_len: length of a sequence; the value is a positive integer.
  - num_head: number of heads in multi-head attention; the value is a positive integer.
  - k_head_size: feature dimension of key in each attention head (length of the key of a single head); the value is a positive integer.
  - num_blocks: total number of pre-allocated blocks in keyCache, used to store the key data of all sequences; the value is a positive integer.
  - block_size: number of tokens contained in each cache block; the value is a positive integer.
- Input value range restriction (the summation in the following formulas covers the elements of all batches and heads):
  - Scenario 2: each element value in seqLensOptional and compressLensOptional must satisfy the formula: reduceSum(seqLensOptional[i] - compressLensOptional[i] + 1) <= num_blocks * block_size, and each write range must satisfy: slotMapping[b][h] >= 0, slotMapping[b][h] + seqLensOptional[b] - compressLensOptional[b][h] + 1 <= num_blocks * block_size.
  - Scenario 3: each element value in compressLensOptional must satisfy the formula: reduceSum(compressLensOptional[i]) <= num_blocks * block_size, and each write range must satisfy: slotMapping[b][h] >= 0, slotMapping[b][h] + compressLensOptional[b * num_head + h] <= num_blocks * block_size.
- Input data type restriction: HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1, and FLOAT4_E1M2 support only the scenario where key is three-dimensional (corresponding to Scenario 1).

## Calling Description

| Calling Method  | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn | [test_aclnn_ScatterPaCache](./examples/test_aclnn_scatter_pa_cache.cpp) | Call the ScatterPaCache operator through [aclnnScatterPaCache](docs/aclnnScatterPaCache_en.md) |
