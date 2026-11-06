# MoeTokenPermute

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

## Function Description

- **Operator function**: Performs the permute computation of MoE. This operator broadcasts and sorts tokens by the index indices.
- **Computation formula**:
  - When paddedMode is `false`

    $$
    sortedIndicesFirst=argSort(indices)
    $$

    $$
    sortedIndicesOut=argSort(sortedIndicesFirst)
    $$

    $$
    permuteTokens[sortedIndicesFirst[i]]=tokens[i//topK]
    $$

  - When paddedMode is `true`

    $$
    permuteTokensOut[i]=tokens[Indices[i]]
    $$

    $$
    sortedIndicesOut=indices
    $$

## Parameters

  <table style="table-layout: auto; width: 100%">
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
      <td>tokens</td>
      <td>Input</td>
      <td>Input tokens, corresponding to `tokens` in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32, INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Input</td>
      <td>Input indices, corresponding to `Indices` in the formula. When the data type of tokens is INT8, the value range is `[0, 10240)`.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>numOutTokens</td>
      <td>Attribute</td>
      <td>Number of valid output tokens. When set to 0, no token is deleted. When not 0, tokens sorted by indices are sliced by numOutTokens, and the part exceeding numOutTokens is discarded. When it is a negative number, it is processed as a negative slice index.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>paddedMode</td>
      <td>Attribute</td>
      <td>When paddedMode is true, indices has been padded to represent the token indices selected by each expert, and indices is not sorted in this case. Currently, only paddedMode=false is supported.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>permuteTokensOut</td>
      <td>Output</td>
      <td>Tokens expanded and sorted by indices, corresponding to `permuteTokensOut` in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32, INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sortedIndicesOut</td>
      <td>Output</td>
      <td>Mapping between permuteTokensOut and tokens, corresponding to `sortedIndicesOut` in the formula.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## Constraints

- indices requires that the number of elements be less than `16777215` and the values be greater than or equal to `0` and less than `16777215` (only the maximum or minimum value of int32 or int64 is supported as a single point; other values outside the range produce incorrect sorting results). When `aclnnMoeTokenPermute` is called on Ascend 950 and the data type of tokens is INT8, indices indicates the expert ID, and the value range is `[0, 10240)`.
- When `aclnnMoeTokenPermute` is called, tokens must be 2D, and indices must be 1D or 2D.
- When `aclnnMoeTokenPermuteV2` is called, tokens must be 2D, and indices must be 1D or 2D.
- paddedMode `true` is not supported.
- <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: topK must be less than or equal to 512.
- <term>Ascend 950PR&950DT products</term>:
  When this API is called, the framework internally calls the [aclnnMoeInitRoutingV2](../moe_init_routing_v2/docs/aclnnMoeInitRoutingV2_en.md) API instead. If a parameter error message is displayed, refer to the following parameter mapping:
  - The token input is equivalent to the x input of the aclnnMoeInitRoutingV2 API.
  - The indices input is equivalent to the expertIdx input of the aclnnMoeInitRoutingV2 API.
  - The numOutTokens input is equivalent to the activeNum input of the aclnnMoeInitRoutingV2 API.
  - The paddedMode input is equivalent to the dropPadMode input of the aclnnMoeInitRoutingV2 API.
  - The permuteTokensOut output is equivalent to the expandedXOut output of the aclnnMoeInitRoutingV2 API.
  - The sortedIndicesOut output is equivalent to the expandedRowIdxOut output of the aclnnMoeInitRoutingV2 API.

## Invocation

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_moe_token_permute](examples/test_aclnn_moe_token_permute.cpp) | Invoke the MoeTokenPermute operator through the [aclnnMoeTokenPermute](docs/aclnnMoeTokenPermute_en.md) API. |
| PyTorch API | - | Invoke the moe_token_permute operator through the [moe_token_permute](docs/torchapi_moe_token_permute_en.md) API. |
