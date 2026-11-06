# MaskedCausalConv1dBackward

English | [简体中文](./README.md)

## Product Support

| Product                                                        | Supported |
| :---------------------------------------------------------- | :------: |
|  <term>Ascend 950PR&950DT products</term>                     |     √    |
| <term>Atlas A3 products</term>      |    ×     |
| <term>Atlas A2 products</term>      |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                              |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- Operator function: Performs the backward gradient computation for the 1D grouped convolution over the tokens of the hidden layer.

- Computation formula:

  Assume that the shapes of the convolution input x, the gradient gradY of the convolution output, and the gradient gradX of the convolution input are [S, B, H], and the shape of weight is [W, H], where i and j denote the indexes of the S/B axes, and k denotes the index within the convolution window W. Then the computation is expressed as:

  $$
  gradYMasked[i,j] = mask[j,i] * gradY[i,j]
  $$

  $$
  gradX[i,j] = \sum_{k=0}^{W-1} gradYMasked[i+k,j] * weight[W-1-k]
  $$

  $$
  gradWeight[k] = \sum_{j=0}^{B-1}\sum_{i=0}^{S-1} gradYMasked[i+W-1-k,j] * x[i,j]
  $$

  Invalid positions are padded with 0. Currently only 3 is supported for W. The H axis involves an elementwise operation, which is not shown in the preceding formulas.

## Parameter Description

  <table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
    <col style="width: 170px">
    <col style="width: 170px">
    <col style="width: 312px">
    <col style="width: 213px">
    <col style="width: 100px">
    </colgroup>
    <thead>
      <tr>
        <th>Parameter Name</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Data Type</th>
        <th>Data Format</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>gradY</td>
        <td>Input</td>
        <td>Input sequence with shape [S, B, H], corresponding to gradY in the formulas. Empty Tensors are not supported.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>x</td>
        <td>Input</td>
        <td>Input with shape [S, B, H], corresponding to x in the formulas. Empty Tensors are not supported.</td>
        <td>Same as the data type of gradY</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>weight</td>
        <td>Input</td>
        <td>Causal 1D grouped convolution kernel with shape [W, H], where W is fixed to 3, corresponding to weight in the formulas. Empty Tensors are not supported.</td>
        <td>Same as the data type of gradY</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>mask</td>
        <td>Optional input</td>
        <td>Boolean mask with shape [B, S], corresponding to mask in the formulas. The default value is None. A value of None indicates that all values of the mask are True. Empty Tensors are not supported.</td>
        <td>BOOL</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>gradX</td>
        <td>Output</td>
        <td>Output result, which indicates the gradient of the grouped convolution input x, with the same shape as gradY. Empty Tensors are not supported.</td>
        <td>Same as the data type of gradY</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>gradWeight</td>
        <td>Output</td>
        <td>Output result, which indicates the gradient of the grouped convolution input weight, with the same shape as weight. Empty Tensors are not supported.</td>
        <td>Same as the data type of gradY</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>

## Constraints

- This API does not support graph mode.
- The shape restrictions on inputs and outputs are as follows:
  - B (Batchsize): The value range is 1 to 32.
  - B *S (Batchsize* SeqLength): The value range is 1 to 512K.
  - H (hiddenSize): The value range is 384 to 24576, and the value must be a multiple of 64.
  - W: Currently only 3 is supported for W.
- The operator input parameters and intermediate computation results do not exceed the value range of the corresponding running data type (float16/bfloat16).
- The operator does not support inputs containing ±inf or nan.

## Calling Example

  | Calling Method | Sample Code                                                     | Description                                                         |
  | --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | aclnn API | [test_aclnn_masked_causal_conv1d_backward](./examples/test_aclnn_masked_causal_conv1d_backward.cpp) | Calls the MaskedCausalConv1dBackward operator through [aclnnMaskedCausalConv1dBackward](docs/aclnnMaskedCausalConv1dBackward_en.md) |
