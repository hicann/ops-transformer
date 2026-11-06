# MoeTokenUnpermute

English | [简体中文](./README.md)

## Product Support

| Product                                                         |  Supported   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR&950DT products</term>                      |     √    |
| <term>Atlas A3 products</term>      |    √    |
| <term>Atlas A2 products</term>      |    √    |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×    |
| <term>Atlas inference products</term>                             |    ×    |
| <term>Atlas training products</term>                              |    ×    |
| <term>Kirin X90 processor series products</term> | √ |
| <term>Kirin 9030 processor series products</term> | √ |

## Function Description

- Operator function: obtains the input data stored in permutedTokens based on the indices stored in sortedIndices. If probs data exists, permutedTokens is multiplied by probs. Finally, the results are accumulated and summed, and the computation result is output.
- Computation formula:

  - When probs is not None, the computation formula is as follows:

    $$
    T[k] = T[S[k]]
    $$

    $$
    T[k] = T[k] * P[i][j]
    $$

    $$
    O[i] = \sum_{k=i*topK}^{(i+1)*topK - 1 } T[k]
    $$

    where $i \in {0,1,...,tokens-1}$; $j \in {0,1,...,topK-1}$; $k \in {0,1,...,tokens*topK-1}$; T indicates permutedTokens; S indicates sortedIndices; P indicates probs; O indicates out; topK indicates topK\_num, the number of experts that process each token; and tokens indicates tokens_num, the number of input tokens.

  - When probs is None, topK\_num=1. The computation formula is as follows:

    $$
    T[i] = T[S[i]]
    $$

    $$
    O[i] = T[i]
    $$

    where $i \in {0,1,...,tokens-1}$; T indicates permutedTokens; S indicates sortedIndices; O indicates out; and tokens indicates tokens_num.

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
      <td>permutedTokens</td>
      <td>Input</td>
      <td>Input to be computed, corresponding to `T` in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sortedIndices</td>
      <td>Input</td>
      <td>Indicates the positions of the data to be computed in permutedTokens, corresponding to `S` in the formula.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>probsOptional</td>
      <td>Optional Input</td>
      <td>Corresponds to `P` in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
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
      <td>Corresponds to `O` in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>: BFLOAT16 is not supported.

## Constraints

- <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: topK_num <= 512.
- <term>Ascend 950PR&950DT products</term>:
  When this interface is called, the framework internally calls the [aclnnMoeFinalizeRoutingV2](../moe_finalize_routing_v2/docs/aclnnMoeFinalizeRoutingV2_en.md) interface. If a parameter error message is displayed, refer to the following parameter mappings:
  - The permutedTokens input is equivalent to the expandedX input of the aclnnMoeFinalizeRoutingV2 interface.
  - The sortedIndices input is equivalent to the expandedRowIdx input of the aclnnMoeFinalizeRoutingV2 interface.
  - The probsOptional input is equivalent to the scalesOptional input of the aclnnMoeFinalizeRoutingV2 interface.
  - The paddedMode input is equivalent to the dropPadMode input of the aclnnMoeFinalizeRoutingV2 interface.
  - The out output is equivalent to the out output of the aclnnMoeFinalizeRoutingV2 interface.
- |<term>Atlas inference products</term>:
  - The supported data types of permutedTokens and probsOptional are FLOAT16 and FLOAT32.
  - topK_num <= 512.
  - hiddensize is a multiple of 128 and less than 10240.

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn Interface  | [test_aclnn_moe_token_unpermute](examples/test_aclnn_moe_token_unpermute.cpp) | Invokes the MoeTokenUnpermute operator through the [aclnnMoeTokenUnpermute](docs/aclnnMoeTokenUnpermute_en.md) interface. |
