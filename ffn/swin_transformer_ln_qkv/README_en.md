# SwinTransformerLnQKV

English | [简体中文](./README.md)

## Product Support

|Product      | Supported |
|:----------------------------|:-----------:|
| <term>Ascend 950PR&950DT products</term>              |    ×     |
|<term>Atlas A3 products</term>|      ×     |
|<term>Atlas A2 products</term>|      √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |
|<term>Kirin X90 processor series products</term> | √ |
|<term>Kirin 9030 processor series products</term> | √ |

## Function Description

- Operator function: Computes Q, K, and V of the Swin Transformer network model in the scenario where fp16 weights are used.

- Computation formula:

    $$
    (Q,K,V)=((Layernorm(inputX)).transpose() * weight).transpose().split()
    $$

  where weight is the concatenation of the weights of the three matrices Q, K, and V.

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
    <th>Input/Output</th>
    <th>Description</th>
    <th>Data type</th>
    <th>Data format</th>
  </tr></thead>
<tbody>
  <tr>
    <td>inputX</td>
    <td>Input</td>
    <td>Indicates the target tensor for normalization computation. It is the inputX in the formula and an aclTensor on the device side. The data type supports FLOAT16. Only the dimensions [B,S,H] are supported, where B is the batch size and supports only [1,32], S is the product of the length and width of the original image, and H is the product of the sequence length and the number of channels and is less than or equal to 1024. Non-contiguous tensors are not supported.</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gamma</td>
    <td>Input</td>
    <td>Indicates the scale factor in the layernorm computation. Only the 1D dimension [H] is supported. It is an aclTensor on the device side. Non-contiguous tensors are not supported.</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>beta</td>
    <td>Input</td>
    <td>Indicates the offset in the layernorm computation. Only the 1D dimension [H] is supported. It is an aclTensor on the device side. Non-contiguous tensors are not supported.</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>weight</td>
    <td>Input</td>
    <td>Indicates the weight matrix used for target tensor conversion. Only the 2D dimension [H, 3 * H] is supported. It is an aclTensor on the device side. Non-contiguous tensors are not supported.</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>bias</td>
    <td>Input</td>
    <td>Indicates the offset matrix used for target tensor conversion. Only the 1D dimension [3 * H] is supported. It is an aclTensor on the device side. Non-contiguous tensors are not supported.</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>query_output</td>
    <td>Output</td>
    <td>Indicates the converted tensor. It is Q in the formula and an aclTensor on the device side. Non-contiguous tensors are not supported.</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>key_output</td>
    <td>Output</td>
    <td>Indicates the converted tensor. It is K in the formula and an aclTensor on the device side. Non-contiguous tensors are not supported.</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>value_output</td>
    <td>Output</td>
    <td>Indicates the converted tensor. It is V in the formula and an aclTensor on the device side. Non-contiguous tensors are not supported.</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

- <term>Atlas A2 products</term>: The data type supports FLOAT16.

## Constraint Description

- Currently, users cannot call this operator directly through aclnn.

## Invocation Description

| Invocation method | Sample code | Description |
| ---- | ---- | ---- |
| Graph mode invocation | [test_geir_swin_transformer_ln_qkv.cpp](examples/test_geir_swin_transformer_ln_qkv.cpp) | Invokes the SwinTransformerLnQKV operator by building a graph through the [operator IR](op_graph/swin_transformer_ln_qkv_proto.h). |
