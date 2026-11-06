# MaskedCausalConv1d

English | [简体中文](./README.md)

## Product Support

| Product                                                     | Supported |
| :----------------------------------------------------------- | :------: |
|  <term>Ascend 950PR&950DT products</term>   |     √    |
| <term>Atlas A3 products</term>     |    ×     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- Operator function: Performs masked causal 1D grouped convolution over the tokens of the hidden layer.

- Computation formula:

  Assume that the shapes of input x and output y are [S, B, H], and the shape of the convolution weight `weight` is [W, H], where i and j denote the indexes of the S and B axes respectively. Then the output is expressed as:

  $$
  y[i,j] = mask[j,i] * \sum_{k=0}^{W-1} x[i-k,j] * weight[W-1-k]
  $$

  Invalid positions are padded with 0. Currently only 3 is supported for W. The H axis involves an elementwise operation, which is not shown in the preceding formula.

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
        <td>x</td>
        <td>Input</td>
        <td>Input sequence with shape [S, B, H], corresponding to x in the formula. Empty Tensors are not supported.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>weight</td>
        <td>Input</td>
        <td>Causal 1D grouped convolution kernel with shape [W, H], where W is fixed to 3, corresponding to weight in the formula. Empty Tensors are not supported.</td>
        <td>Same as the data type of x</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>mask</td>
        <td>Optional input</td>
        <td>Boolean mask with shape [B, S], corresponding to mask in the formula. The default value is None, which is equivalent to all True. Empty Tensors are not supported.</td>
        <td>BOOL</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>y</td>
        <td>Output</td>
        <td>Output result with the same shape as x. Empty Tensors are not supported.</td>
        <td>Same as the data type of x</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>

## Constraints

- Restrictions on input value ranges:
  - B * S: The value range is 1 to 512K.
  - H (hiddenSize): The value range is 384 to 24576 (a multiple of 64).
  - W: Currently only 3 is supported.
- The operator input parameters and intermediate computation results do not exceed the value range of the corresponding running data type (float16/bfloat16).
- The operator does not support inputs containing ±inf or nan.

## Calling Instructions

  | Calling Method | Sample Code                                                     | Description                                                         |
  | --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | aclnn API | [test_aclnn_masked_causal_conv1d](./examples/test_aclnn_masked_causal_conv1d.cpp) | Calls the MaskedCausalConv1d operator through [aclnnMaskedCausalConv1d](docs/aclnnMaskedCausalConv1d_en.md) |
