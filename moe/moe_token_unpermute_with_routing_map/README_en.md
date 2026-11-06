# MoeTokenUnpermuteWithRoutingMap

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
|  <term>Kirin X90 processor products</term> | √ |
|  <term>Kirin 9030 processor products</term> | √ |

## Description

- Operator function: Accumulates the permutedTokens processed by aclnnMoeTokenpermuteWithRoutingMap back into the original unpermutedTokens. The input data stored in permutedTokens is retrieved based on the indices stored in sortedIndices. If probs data exists, permutedTokens is multiplied by probs, the accumulative sum is computed, and the calculation result is output.
- Computation formula:

  $$
  topK\_num= permutedTokens.size(0) // routingMapOptional.size(0)
  $$

  where `topK_num` indicates the maximum number of expert slots reserved for each token. When `paddedMode` is `false`, the number of experts actually selected by each token can be less than or equal to `topK_num`; unused slots are represented by `-1` in `sortedIndices`, and these slots are skipped during computation.

  $$
  numExperts = probs.size(1)
  $$

  $$
  numTokens = probs.size(0)
  $$

  $$
  capacity = sortedIndices.size(0) // numExperts
  $$

  (1) When probs is not None and paddedMode is true:

  $$
  permutedProbs  [i//capacity,sortedIndices[i]]=probs[i]
  $$

  $$
  permutedTokens = permutedTokens  * permutedProbs
  $$

  $$
  unpermutedTokens= zeros(restoreShape, dtype=permutedTokens.dtype, device=permutedTokens.device)
  $$

  $$
  permuteTokenId, outIndex= sortedIndices.sort(dim=-1)
  $$

  $$
  unpermutedTokens[permuteTokenId[i]] += permutedTokens[outIndex[i]]
  $$

  (2) When probs is not None and paddedMode is false:

  $$
  permutedProbs = probs.T.maskedSelect(routingMap.T)
  $$

  $$
  permutedTokens = permutedTokens  * permutedProbs
  $$

  $$
  unpermutedTokens= zeros(restoreShape, dtype=permutedTokens.dtype, device=permutedTokens.device)
  $$

  $$
  if sortedIndices[i] >= 0:
      unpermutedTokens[i//topK\_num] += permutedTokens[sortedIndices[i]] * permutedProbs[i]
  $$

  (3) When probs is None and paddedMode is true:

  $$
  permuteTokenId, outIndex= sortedIndices.sort(dim=-1)
  $$

  $$
  unpermutedTokens[permuteTokenId[i]] += permutedTokens[outIndex[i]]
  $$

  (4) When probs is None and paddedMode is false:

  $$
  if sortedIndices[i] >= 0:
      unpermutedTokens[i//topK\_num] += permutedTokens[sortedIndices[i]]
  $$

## Parameter Description

  <table style="table-layout: auto; width: 100%">
  <thead>
    <tr>
      <th style="white-space: nowrap">Parameter</th>
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
      <td>Input to be computed, corresponding to `permutedTokens` in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sortedIndices</td>
      <td>Input</td>
      <td>Corresponds to `sortedIndices` in the formula. When `paddedMode` is false, a non-negative value indicates a valid row number in `permutedTokens`, and -1 indicates that the slot has no corresponding expert output and is skipped during computation.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>routingMapOptional</td>
      <td>Optional input</td>
      <td>Indicates whether the token at the corresponding position is processed by the corresponding expert, corresponding to `routingMapOptional` in the formula.</td>
      <td>INT8, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>probsOptional</td>
      <td>Optional input</td>
      <td>Indicates the weight of the result of the token at the corresponding position processed by the corresponding expert in the final result, corresponding to `probs` in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>paddedMode</td>
      <td>Attribute</td>
      <td>true indicates that paddedMode is enabled, and false indicates that paddedMode is disabled. When paddedMode is enabled, each expert processes a fixed number of capacity tokens. When paddedMode is disabled, topK_num expert slots are reserved for each token, which are actually processed by no more than topK_num experts; unused slots are represented by -1 in sortedIndices.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>restoreShapeOptional</td>
      <td>Attribute</td>
      <td>Indicates the shape of unpermutedTokens.</td>
      <td>aclIntArray*</td>
      <td>-</td>
    </tr>
    <tr>
      <td>unpermutedTokens</td>
      <td>Output</td>
      <td>Corresponds to `unpermutedTokens` in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>outIndex</td>
      <td>Output</td>
      <td>Corresponds to `outIndex` in the formula.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>permuteTokenId</td>
      <td>Output</td>
      <td>Corresponds to `permuteTokenId` in the formula.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>permuteProbs</td>
      <td>Output</td>
      <td>Indicates the sorted probs in the output, corresponding to `permutedProbs` in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor products</term> and <term>Kirin 9030 processor products</term>: BFLOAT16 is not supported.

## Constraints

- topkNum <= 512. When paddedMode is false, at most topkNum expert slots are reserved for each token, and the number of elements with the value 1 or true in each row of routingMap must be less than or equal to topkNum; -1 is allowed in sortedIndices to indicate an invalid slot.

## Calling Instructions

| Calling Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_moe_token_unpermute_with_routing_map](examples/test_aclnn_moe_token_unpermute_with_routing_map.cpp) | Calls the MoeTokenUnpermuteWithRoutingMap operator through the [aclnnMoeTokenUnpermuteWithRoutingMap](docs/aclnnMoeTokenUnpermuteWithRoutingMap_en.md) API. |
