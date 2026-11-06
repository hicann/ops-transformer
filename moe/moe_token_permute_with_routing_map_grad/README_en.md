# MoeTokenPermuteWithRoutingMapGrad

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

Operator function: performs the backward propagation of aclnnMoeTokenPermuteWithRoutingMap.

Computation formula:

$$
permuteTokenId, outIndex= sortedIndices.sort(dim=-1)
$$

$$
capacity = permutedTokenOutputGrad.size(0) / experts\_num
$$

- When probs is not None:

  $$
  probsGradOutOptional = zeros(tokens\_num, experts\_num)
  $$

  - When paddedMode is true

  $$
  probsGradOutOptional [sortedIndices[i], i/capacity] = permutedProbsOutputGradOptional[i]
  $$

  - When paddedMode is false

  $$
  probsGradOutOptional = maskedscatter(probsGradOutOptional,routingMap,permutedProbsOutputGradOptional)
  $$

- When probs is None:

  $$
  tokensGradOut= zeros(restoreShape, dtype=permutedTokens.dtype, device=permutedTokens.device)
  $$

  $$
  tokensGradOut[permuteTokenId[i]] += permutedTokens[outIndex[i]]
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
   <td>permutedTokenOutputGrad</td>
   <td>Input</td>
   <td>Gradient of the forward output permutedTokens.</td>
   <td>BFLOAT16, FLOAT16, FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>permutedProbsOutputGradOptional</td>
   <td>Input</td>
   <td>Optional input. If it is not passed in, probsGradOutOptional does not need to be computed.</td>
   <td>BFLOAT16, FLOAT16, FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sortedIndices</td>
   <td>Input</td>
   <td>In non-dropPadded mode, the shape must be 1D: (tokens_num \* topK_num,).</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>routingMap</td>
   <td>Input</td>
   <td>Represents the mapping from tokens to experts.</td>
   <td>INT8</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>experts_num</td>
   <td>Attribute</td>
   <td>Number of experts involved in the computation.</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>tokens_num</td>
   <td>Attribute</td>
   <td>Number of tokens involved in the computation.</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>dropAndPad</td>
   <td>Attribute</td>
   <td>true indicates that dropPaddedMode is enabled, and false indicates that dropPaddedMode is disabled.</td>
   <td>BOOL</td>
   <td>-</td>
  </tr>
  <tr>
   <td>tokensGradOut</td>
   <td>Output</td>
   <td>Gradient of the input permutedTokens.</td>
   <td>BFLOAT16, FLOAT16, FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>probsGradOutOptional</td>
   <td>Output</td>
   <td>Gradient of the input probs. Optional output.</td>
   <td>BFLOAT16, FLOAT16, FLOAT32</td>
   <td>ND</td>
  </tr>
 </tbody></table>

## Constraints

 - In the non-dropPaddedMode scenario, topK_num <= 512.
 - The data type of `tokensGradOut` must be the same as that of `permutedTokenOutputGrad`. When `probsGradOutOptional` is provided, its data type must be the same as that of `permutedProbsOutputGradOptional`. The two gradient inputs support the same type, and also support the combination where the token gradient is BFLOAT16 and the probs gradient is FLOAT32.

## Invocation Description

| Invocation Method  | Sample Code                                  | Description                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn Interface | [test_aclnn_moe_token_permute_with_routing_map_grad.cpp](examples/test_aclnn_moe_token_permute_with_routing_map_grad.cpp) | Invokes the MoeTokenPermuteWithRoutingMap operator through the [aclnnMoeTokenPermuteWithRoutingMap](docs/aclnnMoeTokenPermuteWithRoutingMapGrad_en.md) interface. |
