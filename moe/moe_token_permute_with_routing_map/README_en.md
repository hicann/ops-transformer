# MoeTokenPermuteWithRoutingMap

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

Operator function: performs the MoE permute computation. The labels of tokens and experts are passed in as routingMap, and tokens and the optional probsOptional are broadcast and sorted based on routingMap.

Computation formula:
  tokens\_num is the size of dimension 0 of routingMap, and expert\_num is the size of dimension 1 of routingMap.

 1. When dropAndPad is `false`:

  $$
  expertIndex=arrange(tokens\_num).expand(expert\_num,-1)
  $$

  $$
  sortedIndicesFirst=expertIndex.maskedselect(routingMap.T)
  $$

  $$
  sortedIndicesOut=argsort(sortedIndicesFirst)
  $$

  $$
  topK = numOutTokens // tokens\_num
  $$

  $$
  outToken = topK * tokens\_num
  $$

  $$
  permutedTokensOut[sortedIndicesOut[i]]=tokens[i//topK]
  $$

  $$
  permuteProbsOutOptional=probsOptional.T.maskedselect(routingMap.T)
  $$

2. When dropAndPad is `true`:

  $$
  capacity = numOutTokens // expert\_num
  $$

  $$
  outToken = capacity * expert\_num
  $$

  $$
  sortedIndicesOut = argsort(routingMap.T,dim=-1)[:, :capacity]
  $$

  $$
  permutedTokensOut = tokens.index_select(0, sortedIndicesOut)
  $$

- If probsOptional is not `None`:

  $$
  probs\_T\_1D = probsOptional.T.view(-1)
  $$

  $$
  indices\_dim0 = arange(expert\_num)
  $$

  $$
  indices\_dim1 = sortedIndicesOut.view(expert\_num, capacity)
  $$

  $$
  indices\_1D = (indices_dim0 * tokens\_num + indices\_dim1).view(-1)
  $$

  $$
  permuteProbsOutOptional = probs\_T\_1D.index_select(0, indices_1D)
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
   <td>tokens</td>
   <td>Input</td>
   <td>Input tokens of the permute operation, that is, tokens in the formula.</td>
   <td>BFLOAT16, FLOAT16, FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>routingMap</td>
   <td>Input</td>
   <td>routingMap in the formula, representing the mapping from tokens to experts.</td>
   <td>INT8, BOOL</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>probsOptional</td>
   <td>Input</td>
   <td>Optional input, that is, probsOptional in the formula.</td>
   <td>BFLOAT16, FLOAT16, FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>numOutTokens</td>
   <td>Attribute</td>
   <td>numOutTokens in the formula, the number of valid output tokens used to compute topK and capacity in the formula.</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>dropAndPad</td>
   <td>Attribute</td>
   <td>dropAndPad in the formula, indicating whether the dropAndPad mode is enabled.</td>
   <td>BOOL</td>
   <td>-</td>
  </tr>
  <tr>
   <td>permutedTokensOut</td>
   <td>Output</td>
   <td>permutedTokensOut in the formula, tokens expanded and sorted based on indices and then filtered.</td>
   <td>BFLOAT16, FLOAT16, FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sortedIndicesOut</td>
   <td>Output</td>
   <td>sortedIndicesOut in the formula, the mapping between permutedTokensOut and tokens.</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>permuteProbsOutOptional</td>
   <td>Output</td>
   <td>permuteProbsOutOptional in the formula, probsOptional sorted based on indices and then filtered.</td>
   <td>BFLOAT16, FLOAT16, FLOAT32</td>
   <td>ND</td>
  </tr>
 </tbody></table>

## Constraints

 - tokens_num and expert_num must be less than `16777215`.
 - When the pad mode is false, the number of elements equal to 1 or true in each row of routingMap is fixed and less than `512`.

## Invocation Description

| Invocation Method  | Sample Code                                  | Description                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn Interface | [test_aclnn_moe_token_permute_with_routing_map](examples/test_aclnn_moe_token_permute_with_routing_map.cpp) | Invokes the MoeTokenPermuteWithRoutingMap operator through the [aclnnMoeTokenPermuteWithRoutingMap](docs/aclnnMoeTokenPermuteWithRoutingMap_en.md) interface. |
