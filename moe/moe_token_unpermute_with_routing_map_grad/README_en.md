# MoeTokenUnpermuteWithRoutingMapGrad

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

## Description

- **Operator function**: Backpropagation of aclnnMoeTokenUnpermuteWithRoutingMap.
- **Computation formula**:

  (1) probs is not None:

  $$
  permutedTokensGrad[outIndex[i]] = unpermutedTokensGrad[permuteTokenId[i]]
  $$

  $$
  permutedProbsGrad = permutedTokensGrad * permutedTokensOptional
  $$

  $$
  probsGradExpertOrder = \sum_{j=0}^{hidden\_size-1}(permutedProbsGrad_{i,j})
  $$

    - When dropAndPad is false

  $$
  probsGradOut = masked\_scatter(routingMapOptional^T,probsGradExpertOrder)
  $$

  $$
  permutedProbs = probsOptional^T.masked\_select(routingMapOptional^T)
  $$

  $$
  permutedTokensGradOut = permutedProbs.unsqueeze(-1) * permutedTokensGrad
  $$

    - When dropAndPad is true

  $$
  probsGradOut[permuteTokenId[i], outIndex[i]/capacity] = probsGradExpertOrder[outIndex[i]]
  $$

  $$
  permutedProbs[outIndex[i]] = probsOptional.view(1)[i]
  $$

  $$
  permutedTokensGradOut = permutedProbs * permutedTokensGrad
  $$

    (2) probs is None:

  $$
  permutedTokensGradOut[outIndex[i]] = unpermutedTokensGrad[permuteTokenId[i]]
  $$

  1. hidden_size indicates the size of dimension 1 of unpermutedTokensGrad.
  2. When dropAndPad is true, each expert can process a fixed number of capacity tokens. Dimension 1 of the input routingMapOptional is experts_num, which is the number of experts, and dimension 0 of the input outIndex is experts_num * capacity. capacity can be calculated based on these two dimensions.
  3. When dropAndPad is false, each token can be processed by no more than topK_num experts. Dimension 0 of the input unpermutedTokensGrad is tokens_num, which is the number of tokens, and dimension 0 of the input outIndex is tokens_num * topK_num. topK_num can be calculated based on these two dimensions.

## Parameter Description

  <table style="undefined;table-layout: fixed; width: 1296px"><colgroup>
  <col style="width: 212px">
  <col style="width: 151px">
  <col style="width: 541px">
  <col style="width: 273px">
  <col style="width: 119px">
    </colgroup>
    <thead>
      <tr>
        <th>Parameter</th>
        <th>Input/Output/Attribute</th>
        <th>Description</th>
        <th>Data Type</th>
        <th>Data Format</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>unpermutedTokensGrad</td>
        <td>Input</td>
        <td>aclTensor on the Device side. unpermutedTokensGrad in the computation formula, which indicates the gradient of the forward output unpermutedTokens.</td>
        <td>BFLOAT16, FLOAT16, FLOAT</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>outIndex</td>
        <td>Input</td>
        <td>aclTensor on the Device side. outIndex in the computation formula, which indicates the output position index.</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>permuteTokenId</td>
        <td>Input</td>
        <td>aclTensor on the Device side. permuteTokenId in the computation formula, which indicates the token number corresponding to each position of the input permutedTokens.</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>routingMapOptional</td>
        <td>Optional input</td>
        <td>aclTensor on the Device side. This input is optional and is not required when the input probsOptional is a null pointer; in this case, a null pointer should be passed in. routingMapOptional in the computation formula, which indicates whether the token at the corresponding position is processed by the corresponding expert.</td>
        <td>INT8</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>permutedTokensOptional</td>
        <td>Optional input</td>
        <td>aclTensor on the Device side. This input is optional and is not required when the input probsOptional is a null pointer; in this case, a null pointer should be passed in.</td>
        <td>BFLOAT16, FLOAT16, FLOAT</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>probsOptional</td>
        <td>Optional input</td>
        <td>aclTensor on the Device side. This input is optional and is a null pointer when not required.</td>
        <td>BFLOAT16, FLOAT16, FLOAT</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>dropAndPad</td>
        <td>Attribute</td>
        <td>BOOL on the host side. true indicates that dropAndPad is enabled, and false indicates that dropAndPad is disabled.</td>
        <td>BOOL</td>
        <td>-</td>
      </tr>
      <tr>
        <td>restoreShapeOptional</td>
        <td>Attribute</td>
        <td>aclIntArray on the host side.</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>permutedTokensGradOut</td>
        <td>Output</td>
        <td>Gradient of the input permutedTokens, which must be a 2D tensor.</td>
        <td>BFLOAT16, FLOAT16, FLOAT</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>probsGradOutOptional</td>
        <td>Optional output</td>
        <td>Null pointer when not required. Gradient of the input probs.</td>
        <td>BFLOAT16, FLOAT16, FLOAT</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>

## Constraints

- When the input probsOptional is not null and dropAndPad is false
    - topK_num must be less than or equal to 512 and less than or equal to experts_num.
    - experts_num must satisfy (ubSize - (probTypeLen + 1) *numExpertAlign-(tokenTypeLen + 8)* 256) / (6 * tokenTypeLen + 12) >= 1, where ubSize is the size of the on-chip UB space, probTypeLen is the number of bytes of the data type of the input probsOptional, tokenTypeLen is the number of bytes of the data type of the input unpermutedTokensGrad, and numExpertAlign is the result of aligning experts_num up to 32.
- When the input probsOptional is not null and dropAndPad is true
    - capacity must be less than or equal to tokens_num.
    - hidden_size must be less than or equal to 256 * (ubSize - 2080) / (8 + tokenTypeLen), where ubSize is the size of the on-chip UB space and tokenTypeLen is the number of bytes of the data type of the input unpermutedTokensGrad.

## Calling Instructions

| Calling Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_moe_token_unpermute_with_routing_map_grad](examples/test_aclnn_moe_token_unpermute_with_routing_map_grad.cpp) | Calls the MoeTokenUnpermuteWithRoutingMapGrad operator through the [aclnnMoeTokenUnpermuteWithRoutingMapGrad](docs/aclnnMoeTokenUnpermuteWithRoutingMapGrad_en.md) API. |
