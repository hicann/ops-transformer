# MoeTokenUnpermuteWithEp

English | [简体中文](./README.md)

## Product Support

|Product             |  Supported  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |
|  <term>Kirin X90 processor series products</term> | √ |
|  <term>Kirin 9030 processor series products</term> | √ |

## Function Description

- Operator function: obtains the input data in permutedTokens based on the index positions stored in sortedIndices, multiplies the data by probs, and performs merged accumulation.
- Computation formula:

$$
sortedIndices = sortedIndices[rangeOptional[0]<=i<rangeOptional[1]]
$$

(1) When probs is not None, the computation formula is as follows, where $i \in {0, 1, 2, ..., num\_tokens - 1}$, $j \in {0, 1, 2, ..., numTopk - 1}$, and $k \in {0, 1, 2, ..., num\_tokens * numTopk}$:

$$
permutedTokens = permutedTokens.indexSelect(0, sortedIndices)
$$

$$
permutedTokens_{k} = permutedTokens_{k} * probs_{i,j}
$$

$$
out_{i} = \sum_{k=i*numTopk}^{(i+1)*numTopk - 1 } permutedTokens_{k}
$$

(2) When probs is None, the computation formula is as follows, where $i \in {0, 1, 2, ..., num\_tokens - 1}$ and $j \in {0, 1, 2, ..., numTopk - 1}$:

$$
permutedTokens = permutedTokens.indexSelect(0, sortedIndices)
$$

$$
out_{i} = \sum_{k=i*numTopk}^{(i+1)*numTopk - 1 } permutedTokens_{k}
$$

## Parameter Description

  <table style="table-layout: fixed; width: 1576px">
  <colgroup>
    <col style="width: 170px">
    <col style="width: 170px">
    <col style="width: 200px">
    <col style="width: 200px">
    <col style="width: 170px">
  </colgroup>
  <thead>
    <tr>
      <th style="white-space: nowrap">Parameter Name</th>
      <th style="white-space: nowrap">Input/Output/Attribute</th>
      <th style="white-space: nowrap">Description</th>
      <th style="white-space: nowrap">Data Type</th>
      <th style="white-space: nowrap">Data Format</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>permutedTokens</td>
      <td>Input</td>
      <td>Tokens that have been expanded and sorted, corresponding to `permutedTokens` in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sortedIndices</td>
      <td>Input</td>
      <td>Indicates the positions of the data to be computed in permutedTokens, corresponding to `sortedIndices` in the formula.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>probsOptional</td>
      <td>Optional Input</td>
      <td>Expert probabilities corresponding to the input tokens, corresponding to `probs` in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>numTopk</td>
      <td>Attribute</td>
      <td>Number of selected experts.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rangeOptional</td>
      <td>Attribute</td>
      <td>Valid range for EP partitioning; the size is 2.</td>
      <td>aclIntArray*</td>
      <td>-</td>
    </tr>
    <tr>
      <td>paddedMode</td>
      <td>Attribute</td>
      <td>Currently only false is supported.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>restoreShapeOptional</td>
      <td>Attribute</td>
      <td>Currently only nullptr is supported.</td>
      <td>aclIntArray*</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output result of the unpermute operation on permutedTokens, corresponding to `out` in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>: BFLOAT16 is not supported.

## Constraints

- numTopk must be greater than or equal to 1. When probsOptional is not empty, numTopk must be less than or equal to 512.
- Broadcast is not supported.
- paddedMode set to `True` is not supported.
- When rangeOptional is empty, the default value {0,0} is used, the output is all 0s, and no fallback call to other operators is performed.

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_moe_token_unpermute_with_ep](examples/test_aclnn_moe_token_unpermute_with_ep.cpp) | Invokes the MoeTokenUnpermuteWithEp operator through the [aclnnMoeTokenUnpermuteWithEp](docs/aclnnMoeTokenUnpermuteWithEp_en.md) interface. |
