# MoeTokenPermuteGrad

English | [简体中文](./README.md)

## Product Support

| Product                                                        | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

Operator function: Backward propagation computation of aclnnMoeTokenPermute.

Computation formula:
  $$
  inputGrad = permutedOutputGrad.indexSelect(0, sortedIndices)
  $$

  $$
  inputGrad = inputGrad.reshape(-1, numTopk, hiddenSize)
  $$

  $$
  inputGrad = inputGrad.sum(dim = 1)
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
   <td>permutedOutputGrad</td>
   <td>Input</td>
   <td>Gradient of the forward output permutedTokens.</td>
   <td>BFLOAT16, FLOAT16, FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sortedIndices</td>
   <td>Input</td>
   <td>Sorted index values.</td>
   <td>INT32</td>
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
   <td>paddedMode</td>
   <td>Attribute</td>
   <td>Switch of the pad mode.</td>
   <td>BOOL</td>
   <td>-</td>
  </tr>
  <tr>
   <td>out</td>
   <td>Output</td>
   <td>Gradient of the output tokens.</td>
   <td>BFLOAT16, FLOAT16, FLOAT32</td>
   <td>ND</td>
  </tr>
 </tbody></table>

## Constraints

- <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: numTopk <= 512.
- <term>Ascend 950PR&950DT products</term>:
  When this API is called, the framework internally calls the [aclnnMoeInitRoutingV2Grad](../moe_init_routing_v2_grad/docs/aclnnMoeInitRoutingV2Grad_en.md) API instead. If a parameter error message is displayed, refer to the following parameter mapping:
  - The permutedOutputGrad input is equivalent to the gradExpandedX input of the aclnnMoeInitRoutingV2Grad API.
  - The sortedIndices input is equivalent to the expandedRowIdx input of the aclnnMoeInitRoutingV2Grad API.
  - The numTopk input is equivalent to the topK input of the aclnnMoeInitRoutingV2Grad API.
  - The paddedMode input is equivalent to the dropPadMode input of the aclnnMoeInitRoutingV2Grad API.
  - The out output is equivalent to the out output of the aclnnMoeInitRoutingV2Grad API.

- <term>Atlas A3 products</term>: The value range of the communication volume on a single device is [2MB, 100MB].

## Invocation

| Invocation Method  | Sample Code                                  | Description                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn API | [test_aclnn_moe_token_permute_grad.cpp](examples/test_aclnn_moe_token_permute_grad.cpp) | Invoke the MoeTokenPermuteGrad operator through the [aclnnMoeTokenPermuteGrad](docs/aclnnMoeTokenPermuteGrad_en.md) API. |
