# DequantRopeQuantKvcache

English | [简体中文](./README.md)

## Product Support Status

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

- Operator function: after optional dequantization of the input tensor (x), splits the trailing axis into q, k, and vOut based on `sizeSplits` (the split lengths), applies rotary position encoding to q and k to generate qOut and kOut, and then quantizes kOut and vOut and updates them to kCacheRef and vCacheRef according to `indices`.

- Computation formula:

  $$
  dequantX = Dequant(x,weightScaleOptional,activationScaleOptional,biasOptional)
  $$

  $$
  q,k,vOut = SplitTensor(dequantX,dim=-1,`sizeSplits`)
  $$

  $$
  qOut,kOut = ApplyRotaryPosEmb(q,k,cos,sin)
  $$

  $$
  quantK = Quant(kOut,scaleK,offsetKOptional)
  $$

  $$
  quantV = Quant(vOut,scaleV,offsetVOptional)
  $$

  If cacheModeOptional is contiguous:

  $$
  kCacheRef[i][indices[i]]=quantK[i]
  $$

  $$
  vCacheRef[i][indices[i]]=quantV[i]
  $$

  If cacheModeOptional is page:

  $$
  kCacheRefView=kCacheRef.view(-1,kCacheRef[-2],kCacheRef[-1])
  $$

  $$
  vCacheRefView=vCacheRef.view(-1,vCacheRef[-2],vCacheRef[-1])
  $$

  $$
  kCacheRefView[indices[i]]=quantK[i]
  $$

  $$
  vCacheRefView[indices[i]]=quantV[i]
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
    </tr></thead>
  <tbody>
    <tr>
      <td style="white-space: nowrap">x</td>
      <td style="white-space: nowrap">Input</td>
      <td style="white-space: nowrap">Input `x` for splitting in the formula. It is an aclTensor on the device.</td>
      <td style="white-space: nowrap">FLOAT16, INT32, BFLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">cos</td>
      <td style="white-space: nowrap">Input</td>
      <td style="white-space: nowrap">Input `cos` for position encoding in the formula. It is an aclTensor on the device.</td>
      <td style="white-space: nowrap">FLOAT16, BFLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">sin</td>
      <td style="white-space: nowrap">Input</td>
      <td style="white-space: nowrap">Input `sin` for position encoding in the formula. It is an aclTensor on the device.</td>
      <td style="white-space: nowrap">BFLOAT16, FLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">kCacheRef</td>
      <td style="white-space: nowrap">Input</td>
      <td style="white-space: nowrap">Input `kCacheRef` for caching k in the formula. It is an aclTensor on the device.</td>
      <td style="white-space: nowrap">INT8</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">vCacheRef</td>
      <td style="white-space: nowrap">Input</td>
      <td style="white-space: nowrap">Input `vCacheRef` for caching v in the formula. It is an aclTensor on the device.</td>
      <td style="white-space: nowrap">INT8</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">indices</td>
      <td style="white-space: nowrap">Input</td>
      <td style="white-space: nowrap">Input `indices` indicating the token position information of the KV cache in the formula.</td>
      <td style="white-space: nowrap">INT32</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">scaleK</td>
      <td style="white-space: nowrap">Input</td>
      <td style="white-space: nowrap">Input `scaleK`, the scale factor for quantizing `k` in the formula. It is an aclTensor on the device.</td>
      <td style="white-space: nowrap">FLOAT</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">scaleV</td>
      <td style="white-space: nowrap">Input</td>
      <td style="white-space: nowrap">Input `scaleV`, the scale factor for quantizing `v` in the formula. It is an aclTensor on the device.</td>
      <td style="white-space: nowrap">FLOAT</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">offsetKOptional</td>
      <td style="white-space: nowrap">Optional input</td>
      <td style="white-space: nowrap">Input `offsetKOptional`, the offset factor for quantizing k in the formula. It is an aclTensor on the device.</td>
      <td style="white-space: nowrap">FLOAT</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">offsetVOptional</td>
      <td style="white-space: nowrap">Optional input</td>
      <td style="white-space: nowrap">Input `offsetVOptional`, the offset factor for quantizing v in the formula. It is an aclTensor on the device.</td>
      <td style="white-space: nowrap">FLOAT</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">weightScaleOptional</td>
      <td style="white-space: nowrap">Optional input</td>
      <td style="white-space: nowrap">Input `weightScaleOptional`, the weight scale factor for dequantization in the formula. It is an aclTensor on the device.</td>
      <td style="white-space: nowrap">FLOAT</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">activationScaleOptional</td>
      <td style="white-space: nowrap">Optional input</td>
      <td style="white-space: nowrap">Input `activationScaleOptional`, the activation scale factor for dequantization in the formula. It is an aclTensor on the device.</td>
      <td style="white-space: nowrap">FLOAT</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">biasOptional</td>
      <td style="white-space: nowrap">Optional input</td>
      <td style="white-space: nowrap">Input `biasOptional`, the bias for dequantization in the formula. It is an aclTensor on the device.</td>
      <td style="white-space: nowrap">FLOAT, FLOAT16(HALF), INT32, BFLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">sizeSplits</td>
      <td style="white-space: nowrap">Input</td>
      <td style="white-space: nowrap">Lengths for splitting the input qkv.</td>
      <td style="white-space: nowrap">INT64</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">quantModeOptional</td>
      <td style="white-space: nowrap">Optional input</td>
      <td style="white-space: nowrap">Host-side expression string. It indicates the supported quantization type. Currently only static is supported.</td>
      <td style="white-space: nowrap">String</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">layoutOptional</td>
      <td style="white-space: nowrap">Optional input</td>
      <td style="white-space: nowrap">Host-side expression string. It indicates the supported data format. Currently only BSND is supported.</td>
      <td style="white-space: nowrap">String</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">kvOutput</td>
      <td style="white-space: nowrap">Input</td>
      <td style="white-space: nowrap">Host-side expression boolean. It indicates whether to output kOut and vOut.</td>
      <td style="white-space: nowrap">BOOL</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">cacheModeOptional</td>
      <td style="white-space: nowrap">Input</td>
      <td style="white-space: nowrap">Host-side expression string. It indicates the update mode of kCacheRef. Currently only page and contiguous are supported, and the default value is contiguous.</td>
      <td style="white-space: nowrap">String</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">qOut</td>
      <td style="white-space: nowrap">Output</td>
      <td style="white-space: nowrap">Processed q. It is an aclTensor on the device.</td>
      <td style="white-space: nowrap">FLOAT16, BFLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">kOut</td>
      <td style="white-space: nowrap">Output</td>
      <td style="white-space: nowrap">Processed k. It is an aclTensor on the device.</td>
      <td style="white-space: nowrap">FLOAT16, BFLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">vOut</td>
      <td style="white-space: nowrap">Output</td>
      <td style="white-space: nowrap">Processed v. It is an aclTensor on the device.</td>
      <td style="white-space: nowrap">FLOAT16, BFLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor products</term> and <term>Kirin 9030 processor products</term>: BFLOAT16 is not supported.

## Constraints

- When cacheModeOptional is contiguous: the 0th dimension of kCacheRef must be greater than or equal to the 0th dimension of x. When x is 3-dimensional, the value in indices must be greater than or equal to 0 and less than or equal to the 1st dimension of kCacheRef minus the 1st dimension of x. When x is 2-dimensional, the value in indices must be greater than or equal to 0 and less than or equal to the 1st dimension of kCacheRef minus 1. When cacheModeOptional is page: the value in indices must be greater than or equal to 0, less than the 0th dimension of kCacheRef multiplied by the 1st dimension of kCacheRef, and unique.
- The trailing axis of x must be less than or equal to 4096 and aligned to 64.
- When input x is not int32, x, cos, sin and the outputs qOut, kOut, vOut must have the same data type, and activationScaleOptional, weightScaleOptional, and biasOptional do not take effect. When x is int32, cos, sin and the outputs qOut, kOut, vOut must have the same data type, weightScaleOptional is required, and activationScaleOptional and biasOptional are optional (biasOptional does not need to match the types of other inputs).
- The number of elements of sizeSplits must be 3, and each element must not be less than 0; sizeSplits[1] and sizeSplits[2] must be equal and both equal to Nkv*D; the trailing axis of x must be equal to the sum of the three elements of sizeSplits.
- kCacheRef and vCacheRef must be 4-dimensional with the same shape; D must be a positive integer multiple of 16. The numbers of elements of scaleK, scaleV, and the optional inputs offsetKOptional and offsetVOptional must all be Nkv*D.
- The shape of qOut must be [B, S, Nq, D] when x is 3-dimensional, and [B, Nq, D] when x is 2-dimensional. When kvOutput is true, the shapes of kOut and vOut must both be [B, S, Nkv, D] when x is 3-dimensional, and both [B, Nkv, D] when x is 2-dimensional. The number of dimensions and the size of each axis must satisfy the constraints. Other shapes that only keep the same number of elements are not supported.
- When kvOutput is false, kOut and vOut must both be empty tensors (with 0 elements), and null pointers must not be passed. When called through aclnn, a failed tiling check on the output shape makes the first-stage API return ACLNN_ERR_INNER_TILING_ERROR (561002).
- Empty tensor scenarios with B=0 or S=0 are supported: when x is 2-dimensional and B=0, or x is 3-dimensional and B=0 or S=0, the numbers of elements of qOut, kOut, and vOut are 0, and kCacheRef and vCacheRef remain unchanged from the input values. This scenario still requires structural parameters such as the trailing axis of x, Nkv, D, scaleK, and scaleV to satisfy the preceding constraints. D=0, Nkv=0, or a required scale being empty does not fall within this supported scenario.

## Calling Instructions

| Calling Method      | Calling Sample                 | Description                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn call | [test_aclnn_dequant_rope_quant_kvcache](examples/test_aclnn_dequant_rope_quant_kvcache.cpp) | Calls the [DequantRopeQuantKvcache](docs/aclnnDequantRopeQuantKvcache_en.md) operator through the API. |
