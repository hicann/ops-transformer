# MoeTokenPermuteWithEpGrad

English | [简体中文](./README.md)

## Product Support

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

Operator function: performs the backward propagation computation of aclnnMoeTokenPermuteWithEp.

Computation formula:

  Let rangeOptional = [start, end) and numTokens = sortedIndices.numel / numTopk:

  - First compute tokenGradOut:
    - When start <= sortedIndices[i] < end:

      $$
      tokenGradOut[i] = permutedTokensOutputGrad[sortedIndices[i] - start]
      $$

    - Otherwise:

      $$
      tokenGradOut[i] = 0
      $$

  - Then compute:

    $$
    tokenGradOut = tokenGradOut.reshape(-1, numTopk, hiddenSize)
    $$

    $$
    tokenGradOut = tokenGradOut.sum(dim = 1)
    $$

  - Next compute probsGradOut:
    - When start <= sortedIndices[i] < end:

      $$
      probsGradOut[i] = permutedProbsOutputGradOptional[sortedIndices[i] - start]
      $$

    - Otherwise:

      $$
      probsGradOut[i] = 0
      $$

    - Finally:

      $$
      probsGradOut = probsGradOut.reshape(numTokens, numTopk)
      $$

  For BFLOAT16/FLOAT16 inputs, tokenGradOut is accumulated in float32 intermediate precision and rounded back to the input dtype; probsGradOut remains in the input dtype.

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
   <td>permutedTokensOutputGrad</td>
   <td>Input</td>
   <td>Gradient of the forward output permutedTokens, that is, `permutedTokensOutputGrad` in the formula.</td>
   <td>BFLOAT16, FLOAT16, FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sortedIndices</td>
   <td>Input</td>
   <td>Mapping between the forward output permuteTokensOut and the forward input tokens, that is, `sortedIndices` in the formula.</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>permutedProbsOutputGradOptional</td>
   <td>Input</td>
   <td>Optional computation input, corresponding to the computation output probsGradOut. If an empty tensor is passed in, probsGradOut is not output.</td>
   <td>BFLOAT16, FLOAT16, FLOAT32</td>
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
   <td>Valid range for EP partitioning.</td>
   <td>aclIntArray</td>
   <td>-</td>
  </tr>
  <tr>
   <td>paddedMode</td>
   <td>Attribute</td>
   <td>true indicates that paddedMode is enabled, and false indicates that paddedMode is disabled. The default value is false, and it does not take effect currently.</td>
   <td>BOOL</td>
   <td>-</td>
  </tr>
  <tr>
   <td>tokenGradOut</td>
   <td>Output</td>
   <td>Gradient of the input token.</td>
   <td>BFLOAT16, FLOAT16, FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>probsGradOut</td>
   <td>Output</td>
   <td>Gradient of the input probs.</td>
   <td>BFLOAT16, FLOAT16, FLOAT32</td>
   <td>ND</td>
  </tr>
 </tbody></table>

## Constraints

 - numTopk <= 512.
 - paddedMode set to `True` is not supported.
 - When rangeOptional is empty, permutedProbsOutputGradOptional and probsGradOut are ignored, and the execution logic falls back to [aclnnMoeTokenPermuteGrad](../moe_token_permute_grad/docs/aclnnMoeTokenPermuteGrad_en.md).

## Invocation Description

| Invocation Method  | Sample Code                                  | Description                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn Interface | [test_aclnn_moe_token_permute_with_ep_grad.cpp](examples/test_aclnn_moe_token_permute_with_ep_grad.cpp) | Invokes the MoeTokenPermuteWithEpGrad operator through the [aclnnMoeTokenPermuteWithEpGrad](docs/aclnnMoeTokenPermuteWithEpGrad_en.md) interface. |
