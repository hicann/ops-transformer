# MoeTokenUnpermuteGrad

English | [简体中文](./README.md)

## Product Support

| Product                                                         |  Supported   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR&950DT products</term>                      |     √    |
| <term>Atlas A3 products</term>      |    √    |
| <term>Atlas A2 products</term>      |    √    |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- Operator function: performs the backward propagation of aclnnMoeTokenUnpermute.
- Computation formula:

  - When probs is not None:

    $$
    unpermutedTokens[i] = permutedTokens[sortedIndices[i]]
    $$

    $$
    unpermutedTokens = unpermutedTokens.reshape(-1, topK\_num, hiddenSize)
    $$

    $$
    unpermutedTokens = unpermutedTokensGrad.unsqueeze(1) * unpermutedTokens
    $$

    $$
    probsGrad = \sum_{k=0}^{K}(unpermutedTokens_{i,j,k})
    $$

    $$
    permutedTokensGrad[sortedIndices[i]] = ((unpermutedTokensGrad.unsqueeze(1) * probs.unsqueeze(-1)).reshape(-1, hiddenSize))[i]
    $$

  - When probs is None:

    $$
    permutedTokensGrad[sortedIndices[i]] = unpermutedOutputGrad[i]
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
      <td>permutedTokensOptional</td>
      <td>Optional Input</td>
      <td>Input token, corresponding to `permutedTokens` in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
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
      <td>probs</td>
      <td>Optional Input</td>
      <td>Corresponds to `probs` in the formula.</td>
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
      <td>permutedTokensGradOut</td>
      <td>Output</td>
      <td>Corresponds to `permutedTokensGrad` in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>probsGradOut</td>
      <td>Output</td>
      <td>Corresponds to `probsGrad` in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: topK_num <= 512.
- <term>Ascend 950PR&950DT products</term>:
  When this interface is called, the framework internally forwards the call to the [aclnnMoeFinalizeRoutingV2Grad](../moe_finalize_routing_v2_grad/docs/aclnnMoeFinalizeRoutingV2Grad_en.md) interface. If a parameter error message is displayed, refer to the following parameter mappings:
  - The permutedTokensOptional input is equivalent to the expandedXOptional input of the aclnnMoeFinalizeRoutingV2Grad interface.
  - The unpermutedTokensGrad input is equivalent to the gradY input of the aclnnMoeFinalizeRoutingV2Grad interface.
  - The sortedIndices input is equivalent to the expandedRowIdx input of the aclnnMoeFinalizeRoutingV2Grad interface.
  - The probsOptional input is equivalent to the scalesOptional input of the aclnnMoeFinalizeRoutingV2Grad interface.
  - The paddedMode input is equivalent to the dropPadMode input of the aclnnMoeFinalizeRoutingV2Grad interface.
  - The permutedTokensGradOut output is equivalent to the gradExpandedXOut output of the aclnnMoeFinalizeRoutingV2Grad interface.
  - The probsGradOut output is equivalent to the gradScalesOut output of the aclnnMoeFinalizeRoutingV2Grad interface.

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn Interface  | [test_aclnn_moe_token_unpermute_grad](examples/test_aclnn_moe_token_unpermute_grad.cpp) | Invokes the MoeTokenUnpermuteGrad operator through the [aclnnMoeTokenUnpermuteGrad](docs/aclnnMoeTokenUnpermuteGrad_en.md) interface. |
