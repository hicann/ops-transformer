# MoeTokenUnpermuteWithEpGrad

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

- Operator function: performs the backward propagation of aclnnMoeTokenUnpermuteWithEp.
- Computation formula:

  $$
  sortedIndices= sortedIndices[sortedIndices[rangeOptional[0]]<=i<sortedIndices[rangeOptional[1]]]
  $$

- When probsOptional is not None:

  $$
  unpermutedTokens[i] = permutedTokensOptional[sortedIndices[i]]
  $$

  $$
  unpermutedTokens = unpermutedTokens.reshape(-1, topkNum, hiddenSize)
  $$

  $$
  unpermutedTokens = unpermutedTokensGrad.unsqueeze(1) * unpermutedTokens
  $$

  $$
  probsGrad = \sum_{k=0}^{topkNum}(unpermutedTokens_{i,j,k})
  $$

  $$
  permutedTokensGradOut[sortedIndices[i]] = ((unpermutedTokensGrad.unsqueeze(1) * probs.unsqueeze(-1)).reshape(-1, hiddenSize))[i]
  $$

- When probsOptional is None:

  $$
  permutedTokensGradOut[sortedIndices[i]] = unpermutedTokensGrad[i]
  $$

## Parameter Description

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
      <td>unpermutedTokensGrad</td>
      <td>Input</td>
      <td>Gradient of the forward output unpermutedTokens, corresponding to `unpermutedTokensGrad` in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sortedIndices</td>
      <td>Input</td>
      <td>Corresponds to `sortedIndices` in the formula.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>permutedTokensOptional</td>
      <td>Optional Input</td>
      <td>Corresponds to `permutedTokensOptional` in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>probsOptional</td>
      <td>Optional Input</td>
      <td>Corresponds to `probsOptional` in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>paddedMode</td>
      <td>Attribute</td>
      <td>The default value is false, and it does not take effect currently.</td>
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
      <td>rangeOptional</td>
      <td>Attribute</td>
      <td>Valid range for EP partitioning, corresponding to `rangeOptional` in the formula.</td>
      <td>aclIntArray*</td>
      <td>-</td>
    </tr>
    <tr>
      <td>topkNum</td>
      <td>Attribute</td>
      <td>Number of experts selected for each token, corresponding to `topkNum` in the formula.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>permutedTokensGradOut</td>
      <td>Output</td>
      <td>Gradient of the input permutedTokens, corresponding to `permutedTokensGradOut` in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>probsGradOut</td>
      <td>Output</td>
      <td>Gradient of the input probs, corresponding to `probsGradOut` in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- topkNum <= 512.

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn Interface  | [test_aclnn_moe_token_unpermute_with_ep_grad](examples/test_aclnn_moe_token_unpermute_with_ep_grad.cpp) | Invokes the MoeTokenUnpermuteWithEpGrad operator through the [aclnnMoeTokenUnpermuteWithEpGrad](docs/aclnnMoeTokenUnpermuteWithEpGrad_en.md) interface. |
