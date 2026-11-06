# SwinAttentionFFN

English | [简体中文](./README.md)

## Product Support

<table style="undefined;table-layout: fixed; width: 700px"><colgroup>
<col style="width: 600px">
<col style="width: 100px">
</colgroup>
<thead>
  <tr>
    <th style="text-align: center;">Product</th>
    <th style="text-align: center;">Supported</th>
  </tr></thead>
<tbody>
  <tr>
    <td>Ascend 950PR&950DT products</td>
    <td style="text-align: center;">×</td>
  </tr>
  <tr>
    <td>Atlas A3 training products/Atlas A3 inference products</td>
    <td style="text-align: center;">×</td>
  </tr>
  <tr>
    <td>Atlas A2 training products/Atlas A2 inference products</td>
    <td style="text-align: center;">√</td>
  </tr>
  <tr>
    <td>Atlas 200I/500 A2 inference products</td>
    <td style="text-align: center;">×</td>
  </tr>
  <tr>
    <td>Atlas inference products</td>
    <td style="text-align: center;">×</td>
  </tr>
  <tr>
    <td>Atlas training products</td>
    <td style="text-align: center;">×</td>
  </tr>
  <tr>
    <td>Kirin X90 processor series products</td>
    <td style="text-align: center;">√</td>
  </tr>
  <tr>
    <td>Kirin 9030 processor series products</td>
    <td style="text-align: center;">√</td>
  </tr>
</tbody>
</table>

## Function Description

- Operator function: An FFN operator for the swin transformer scenario. It multiplies the inputs x1 and x2, adds the bias offset, and performs residual addition with x3 to output the final result. Only the float16 data type is supported.

- Computation formula:

    $$
    y=x1*x2+bias +x3
    $$

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
<col style="width: 150px">
<col style="width: 200px">
<col style="width: 450px">
<col style="width: 200px">
<col style="width: 100px">
</colgroup>
<thead>
  <tr>
    <th>Parameter name</th>
    <th>Input/Output/Attribute</th>
    <th>Description</th>
    <th>Data type</th>
    <th>Data format</th>
  </tr></thead>
<tbody>
  <tr>
    <td>x1</td>
    <td>Input</td>
    <td>Required. An aclTensor on the device side. It is the input x1 in the formula. The input supports 3D dimensions [B,M,K], where B is the batch size. [M,K] supports only [64,128].</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x2</td>
    <td>Input</td>
    <td>Required. An aclTensor on the device side. It is the input x2 in the formula. The input supports 2D dimensions [K, N]. [K, N] supports only [128,128].</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>bias</td>
    <td>Input</td>
    <td>Required. An aclTensor on the device side. It is the input bias in the formula. The input supports 1D dimensions [N]. [N] supports only [128].</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x3</td>
    <td>Input</td>
    <td>Optional. An aclTensor on the device side. It is the input x3 in the formula. The input supports 3D dimensions [B,M,N], where B is the batch size. [M,N] supports only [64,128].</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y</td>
    <td>Output</td>
    <td>Required. An aclTensor on the device side. It is the output y in the formula.</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## Constraint Description

  Currently, this operator does not support direct invocation by users
