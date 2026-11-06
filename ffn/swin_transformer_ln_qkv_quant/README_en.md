# SwinTransformerLnQkvQuant

English | [简体中文](./README.md)

## Product Support

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    ×     |
| <term>Atlas A3 products</term>     |    ×     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    ×     |
| <term>Kirin X90 processor series products</term> | √ |
| <term>Kirin 9030 processor series products</term> | √ |

## Function Description

- Operator function: Computes Q, K, and V of the Swin Transformer network model.
- Computation formula:

    $$
    (Q,K,V)=(Quant(Layernorm(x).transpose()) * weight).dequant().transpose().split()
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
    <th>Input/Output/Attribute</th>
    <th>Description</th>
    <th>Data type</th>
    <th>Data format</th>
  </tr></thead>
<tbody>
  <tr>
    <td>x</td>
    <td>Input</td>
    <td>Indicates the target tensor for normalization computation. It is the x in the formula and an aclTensor on the device side. The data type supports FLOAT16. Only the dimensions [B,S,H] are supported, where B is the batch size and supports only [1,32], S is the product of the length and width of the original image, and H is the product of the sequence length and the number of channels and is less than or equal to 1024. Non-contiguous tensors are not supported.</td>
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
    <td>Indicates the weight matrix used for target tensor conversion. Only the 2D dimension [3 * H, H] is supported. It is an aclTensor on the device side. Currently, only weightTranspose=true is supported; in this case, dimension 0 is N and dimension 1 is K. Non-contiguous tensors are not supported.</td>
    <td>INT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>bias</td>
    <td>Input</td>
    <td>Indicates the offset matrix used for target tensor conversion. Only the 1D dimension [3 * H] is supported. It is an aclTensor on the device side. Non-contiguous tensors are not supported.</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>quantScale</td>
    <td>Input</td>
    <td>Indicates the scaling parameter for quantization of the target tensor. Only the 1D dimension [H] is supported. It is an aclTensor on the device side. Non-contiguous tensors are not supported.</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>quantOffset</td>
    <td>Input</td>
    <td>Indicates the offset matrix used for target tensor conversion. Only the 1D dimension [H] is supported. It is an aclTensor on the device side. Non-contiguous tensors are not supported.</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>dequantScale</td>
    <td>Input</td>
    <td>Indicates the scaling parameter for dequantization after the target tensor is multiplied by the weight matrix. Only the 1D dimension [3 * H] is supported. It is an aclTensor on the device side. Non-contiguous tensors are not supported.</td>
    <td>UINT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>headNum</td>
    <td>Input</td>
    <td>Indicates the number of channels used for conversion; the supported range is [1,32].</td>
    <td>int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>seqLength</td>
    <td>Input</td>
    <td> Indicates the sequence length used for conversion. Only 32 and 64 are supported.</td>
    <td>int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>epsilon</td>
    <td>Input</td>
    <td>Division-by-zero protection value in the layernorm computation; to ensure precision, the value is recommended to be less than or equal to 1e-4.</td>
    <td>float</td>
    <td>-</td>
  </tr>
  <tr>
    <td>oriHeight</td>
    <td>Input</td>
    <td>Dimension of the transpose along the S axis in layernorm; oriHeight*oriWeight must equal the size of the second dimension S of the input x and be an integer multiple of hWinSize.</td>
    <td>int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>oriWeight</td>
    <td>Input</td>
    <td>Dimension of the transpose along the S axis in layernorm; oriHeight*oriWeight must equal the size of the second dimension S of the input x and be an integer multiple of wWinSize.</td>
    <td>int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>hWinSize</td>
    <td>Attribute</td>
    <td>Width of the feature window used; the supported range is [7,32].</td>
    <td>int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>wWinSize</td>
    <td>Attribute</td>
    <td>Width of the feature window used; the supported range is [7,32].</td>
    <td>int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>weightTranspose</td>
    <td>Attribute</td>
    <td>The weight matrix needs to be transposed. The non-transposed scenario is not supported currently.</td>
    <td>bool</td>
    <td>-</td>
  </tr>
  <tr>
    <td>queryOutputOut</td>
    <td>Output</td>
    <td>Indicates the converted tensor. It is Q in the formula and an aclTensor on the device side. Non-contiguous tensors are not supported.</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>keyOutputOut</td>
    <td>Output</td>
    <td>Indicates the converted tensor. It is K in the formula and an aclTensor on the device side. Non-contiguous tensors are not supported.</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>valueOutputOut</td>
    <td>Output</td>
    <td>Indicates the converted tensor. It is V in the formula and an aclTensor on the device side. Non-contiguous tensors are not supported.</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## Constraint Description

- seqLength supports only 32/64.
- oriHeight*oriWeight equals the second dimension of the input x Tensor, oriHeight is an integer multiple of hWinSize, and oriWeight is an integer multiple of wWinSize.
- The range of hWinSize and wWinSize supports only 7 to 32.
- The first dimension B of the input x Tensor supports only 1 to 32.
- The weight needs to be transposed.
