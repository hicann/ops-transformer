# MoeTokenPermuteWithEp

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

Operator function: Performs the permute computation of MoE (Mixture of Experts). This operator broadcasts tokens and the optional probs by the index indices, sorts them, and slices them by the range in rangeOptional.

Computation formula:

- When paddedMode is `false`, the formulas are as follows. topK indicates the number of experts selected by each token: when indices is 2-dimensional, topK equals the size of the last dimension of indices; when indices is 1-dimensional, topK is 1.

    $$
    sortedIndicesFirst=argSort(flatten(indices))
    $$

    $$
    sortedIndicesOut=argSort(sortedIndicesFirst)
    $$

    When rangeOptional[0] <= sortedIndicesOut[i] < rangeOptional[1]

    $$
    permuteTokensOut[sortedIndicesOut[i]-rangeOptional[0]]=tokens[i//topK]
    $$

    $$
    permuteProbsOut[sortedIndicesOut[i]-rangeOptional[0]]=probsOptional[i]
    $$

- When paddedMode is `true` (not yet supported)

    $$
    permuteTokensOut[i]=tokens[indices[i]]
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
   <td>Input tokens in permute, corresponding to `tokens` in the formula.</td>
   <td>BFLOAT16, FLOAT16, FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>indices</td>
   <td>Input</td>
   <td>Expert indices corresponding to the input tokens, corresponding to `indices` in the formula.</td>
   <td>INT32, INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>probsOptional</td>
   <td>Optional input</td>
   <td>Expert probabilities corresponding to the input tokens, corresponding to `probsOptional` in the formula. The data type must be the same as that of tokens.</td>
   <td>BFLOAT16, FLOAT16, FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>rangeOptional</td>
   <td>Optional attribute</td>
   <td>Valid range of the EP split, corresponding to `rangeOptional` in the formula. The default value is [0,0].</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>numOutTokens</td>
   <td>Optional attribute</td>
   <td>Parameter related to the number of output tokens. It takes effect only when rangeOptional is empty. In this case, the slice start is 0 and the end is the value of this parameter. When rangeOptional is not empty, this parameter is ignored. The default value is 0. To specify an output range, use rangeOptional.</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>paddedMode</td>
   <td>Optional attribute</td>
   <td>When true, indicates that indices has been padded to represent the token indices selected by each expert. Currently, only false is supported. The default value is false.</td>
   <td>BOOL</td>
   <td>-</td>
  </tr>
  <tr>
   <td>permuteTokensOut</td>
   <td>Output</td>
   <td>Tokens expanded and sorted by indices, corresponding to `permuteTokensOut` in the formula.</td>
   <td>BFLOAT16, FLOAT16, FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sortedIndicesOut</td>
   <td>Output</td>
   <td>Sorted output result.</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>permuteProbsOut</td>
   <td>Output</td>
   <td>Output after permute.</td>
   <td>BFLOAT16, FLOAT16, FLOAT32</td>
   <td>ND</td>
  </tr>
 </tbody></table>

## Constraints

- tokens must be 2D, with the shape `(num_tokens, hidden_size)`, and dimension 0 must be the same as dimension 0 of indices. indices must be 1D or 2D.
- indices requires that the number of elements (`num_tokens * topK`) be less than `16777215`, that is, not greater than `16777214`; the element values must be greater than or equal to `0` and less than or equal to `16777214`. Outside this value range, only the maximum or minimum value of int32 or int64 is supported as a single point (used as an invalid position marker), and the sorting results of other out-of-range values are incorrect.
- topK indicates the number of experts selected by each token. When indices is 2-dimensional, topK equals the size of the last dimension of indices; when indices is 1-dimensional, topK is 1. The value range of topK is `[1, 512]`.
- The inputs tokens and indices do not support empty tensors (`num_tokens` and `hidden_size` must both be greater than 0; calls with `num_tokens` being 0 are out of the support scope). probsOptional is an optional input and can be omitted. When it is passed, its shape must be the same as that of indices and it must not be an empty tensor.
- When probsOptional is passed, its data type must be the same as that of tokens. When it is not passed, permuteProbsOut is not output, and permuteProbsOut does not participate in computation or comparison.
- When rangeOptional is not empty, its size must be `2`, and both elements start and end must be within `[0, num_tokens*topK]`. Otherwise, a failure is returned.
- numOutTokens takes effect only when rangeOptional is empty: the value `0` indicates that no token is deleted (the output has `num_tokens*topK` rows), and a positive integer indicates that the first numOutTokens rows of the sorting result are retained. The value range is `[0, num_tokens*topK]`. Negative values and values exceeding the upper limit are not supported.
- paddedMode `true` is not supported.
- When rangeOptional is empty, probsOptional and permuteProbsOut are ignored, and the execution logic falls back to [aclnnMoeTokenPermute](../moe_token_permute/docs/aclnnMoeTokenPermute_en.md).

## Invocation

| Invocation Method  | Sample Code                                  | Description                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn API | [test_aclnn_moe_token_permute_with_ep.cpp](examples/test_aclnn_moe_token_permute_with_ep.cpp) | Invoke the MoeTokenPermuteWithEp operator through the [aclnnMoeTokenPermuteWithEp](docs/aclnnMoeTokenPermuteWithEp_en.md) API. |
