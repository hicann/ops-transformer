# RopeWithSinCosCache

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
|:---|:---:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |
| <term>Kirin X90 processor products</term> | √ |
| <term>Kirin 9030 processor products</term> | √ |

## Description

- Operator function: To improve performance, the inference network passes the sin and cos inputs through a cache, and the operator performs the rotary positional encoding computation.
- Computation formulas:

    1. **mrope mode**: The shape of positions is [m, numTokens], where m is the element count of mropeSection, and 3 or 4 is supported:

    $$
    cosSin[i] = cosSinCache[positions[i]]
    $$

    $$
    cos, sin = cosSin.chunk(2, dim=-1)
    $$

    (1) When cacheMode is 0:
    - The element count of mropeSection is 3:

      $$
      cos0 = cos[0, :, :mropeSection[0]]
      $$

      $$
      cos1 = cos[1, :, mropeSection[0]:(mropeSection[0] + mropeSection[1])]
      $$

      $$
      cos2 = cos[2, :, (mropeSection[0] + mropeSection[1]):(mropeSection[0] + mropeSection[1] + mropeSection[2])]
      $$

      $$
      cos = torch.cat((cos0, cos1, cos2), dim=-1)
      $$

      $$
      sin0 = sin[0, :, :mropeSection[0]]
      $$

      $$
      sin1 = sin[1, :, mropeSection[0]:(mropeSection[0] + mropeSection[1])]
      $$

      $$
      sin2 = sin[2, :, (mropeSection[0] + mropeSection[1]):(mropeSection[0] + mropeSection[1] + mropeSection[2])]
      $$

      $$
      sin= torch.cat((sin0, sin1, sin2), dim=-1)
      $$

      $$
      queryRot = query[..., :rotaryDim]
      $$

      $$
      queryPass = query[..., rotaryDim:]
      $$

    - The element count of mropeSection is 4:

      $$
      cos = torch.cat([m[i]\ for\ i, m\ in\ enumerate(cos.split(mropeSection, dim=-1))], dim=-1)
      $$

      $$
      sin = torch.cat([m[i]\ for\ i, m\ in\ enumerate(sin.split(mropeSection, dim=-1))], dim=-1)
      $$

      $$
      queryRot = query[..., :rotaryDim]
      $$

      $$
      queryPass = query[..., rotaryDim:]
      $$

    (2) When cacheMode is 1:

    $$
    cosTmp = cos
    $$

    $$
    cos [..., 1:mropeSection[1] * 3:3] = cosTmp[1, ..., 1:mropeSection[1] * 3:3]
    $$

    $$
    cos[..., 2:mropeSection[1] * 3:3] = cosTmp[2, ..., 2:mropeSection[1] * 3:3]
    $$

    $$
    sinTmp = sin
    $$

    $$
    sin[..., 1:mropeSection[1] * 3:3] = sinTmp [1, ..., 1:mropeSection[1] * 3:3]
    $$

    $$
    sin[..., 2:mropeSection[1] * 3:3] = sinTmp [2, ..., 2:mropeSection[1] * 3:3]
    $$

    $$
    queryRot = query[..., :rotaryDim]
    $$

    $$
    queryPass = query[..., rotaryDim:]
    $$

    (1) rotate\_half (GPT-NeoX style) computation mode:

    $$
    x1, x2 = torch.chunk(queryRot, 2, dim=-1)
    $$

    $$
    o1[i] = x1[i] * cos[i] - x2[i] * sin[i]
    $$

    $$
    o2[i] = x2[i] * cos[i] + x1[i] * sin[i]
    $$

    $$
    queryRot = torch.cat((o1, o2), dim=-1)
    $$

    $$
    query = torch.cat((queryRot, queryPass), dim=-1)
    $$

    (2) rotate\_interleaved (GPT-J style) computation mode:

    $$
    x1 = queryRot[..., ::2]
    $$

    $$
    x2 = queryRot[..., 1::2]
    $$

    $$
    o1[i] = x1[i] * cos[i] - x2[i] * sin[i]
    $$

    $$
    o2[i] = x2[i] * cos[i] + x1[i] * sin[i]
    $$

    $$
    queryRot = torch.stack((o1, o2), dim=-1)
    $$

    $$
    query = torch.cat((queryRot, queryPass), dim=-1)
    $$

    2. **rope mode**: The shape of positions is [numTokens]:

    $$
    cosSin[i] = cosSinCache[positions[i]]
    $$

    $$
    cos, sin = cosSin.chunk(2, dim=-1)
    $$

    $$
    queryRot = query[..., :rotaryDim]
    $$

    $$
    queryPass = query[..., rotaryDim:]
    $$

    (1) rotate\_half (GPT-NeoX style) computation mode:

    $$
    x1, x2 = torch.chunk(queryRot, 2, dim=-1)
    $$

    $$
    o1[i] = x1[i] * cos[i] - x2[i] * sin[i]
    $$

    $$
    o2[i] = x2[i] * cos[i] + x1[i] * sin[i]
    $$

    $$
    queryRot = torch.cat((o1, o2), dim=-1)
    $$

    $$
    query = torch.cat((queryRot, queryPass), dim=-1)
    $$

    (2) rotate\_interleaved (GPT-J style) computation mode:

    $$
    x1 = queryRot[..., ::2]
    $$

    $$
    x2 = queryRot[..., 1::2]
    $$

    $$
    o1[i] = x1[i] * cos[i] - x2[i] * sin[i]
    $$

    $$
    o2[i] = x2[i] * cos[i] + x1[i] * sin[i]
    $$

    $$
    queryRot = torch.stack((o1, o2), dim=-1)
    $$

    $$
    query = torch.cat((queryRot, queryPass), dim=-1)
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
      <td style="white-space: nowrap">positions</td>
      <td style="white-space: nowrap">Input</td>
      <td style="white-space: nowrap">aclTensor on the device, the input index.</td>
      <td style="white-space: nowrap">INT64</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">queryIn</td>
      <td style="white-space: nowrap">Input</td>
      <td style="white-space: nowrap">aclTensor on the device, the first tensor on which rotary positional encoding is performed, `query` in the formulas.</td>
      <td style="white-space: nowrap">BFLOAT16, FLOAT16, FLOAT32</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">keyIn</td>
      <td style="white-space: nowrap">Input</td>
      <td style="white-space: nowrap">aclTensor on the device, the second tensor on which rotary positional encoding is performed.</td>
      <td style="white-space: nowrap">BFLOAT16, FLOAT16, FLOAT32</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">cosSinCache</td>
      <td style="white-space: nowrap">Input</td>
      <td style="white-space: nowrap">aclTensor on the device, the positional encoding tensor that participates in the computation.</td>
      <td style="white-space: nowrap">BFLOAT16, FLOAT16, FLOAT32</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">mropeSection</td>
      <td style="white-space: nowrap">Attribute</td>
      <td style="white-space: nowrap">Under the mrope mode, used to integrate the positional encoding tensor information of the inputs, `mropeSection` in the formulas. Passing nullptr indicates the rope mode.</td>
      <td style="white-space: nowrap">aclIntArray</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">headSize</td>
      <td style="white-space: nowrap">Attribute</td>
      <td style="white-space: nowrap">Dimension size of each attention head.</td>
      <td style="white-space: nowrap">INT64</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">isNeoxStyle</td>
      <td style="white-space: nowrap">Attribute</td>
      <td style="white-space: nowrap">true indicates the rotate\_half (GPT-NeoX style) computation mode, and false indicates the rotate\_interleaved (GPT-J style) computation mode.</td>
      <td style="white-space: nowrap">BOOL</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">cacheMode</td>
      <td style="white-space: nowrap">Attribute</td>
      <td style="white-space: nowrap">Concatenation mode of cos and sin. 0 indicates the segmented mode, and 1 indicates the interleaved mode. Provided only by the V2 API.</td>
      <td style="white-space: nowrap">INT64</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">queryOut</td>
      <td style="white-space: nowrap">Output</td>
      <td style="white-space: nowrap">Output result of query after rotary positional encoding.</td>
      <td style="white-space: nowrap">FLOAT, FLOAT16, BFLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">keyOut</td>
      <td style="white-space: nowrap">Output</td>
      <td style="white-space: nowrap">Output result of key after rotary positional encoding.</td>
      <td style="white-space: nowrap">FLOAT, FLOAT16, BFLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor products</term> and <term>Kirin 9030 processor products</term>: BFLOAT16 is not supported.

## Constraints

- queryIn, keyIn, and cosSinCache support only 2-dimensional shape inputs.
- The size of dimension 1 of queryIn and keyIn must be a multiple of headSize.
- The data types of the queryIn, keyIn, and cosSinCache inputs must be identical.
- headSize: when the data type is BFLOAT16 or FLOAT16, it is a multiple of 32; when the data type is FLOAT32, it is a multiple of 16.
- rotaryDim: always less than or equal to headSize; when the data type is BFLOAT16 or FLOAT16, it is a multiple of 32; when the data type is FLOAT32, it is a multiple of 16. In the mrope mode, the sum of all elements of mropeSection should equal half of rotaryDim.
- The values of the input tensor positions should be less than the dimension-0 maxSeqLen of cosSinCache.
- aclnnRopeWithSinCosCache is deterministic by default.
- When invoked through the aclnn APIs, mropeSection and cacheMode are subject to the following constraints:
  - The aclnnRopeWithSinCosCache API supports only mropeSection values of [16, 24, 24], [24, 20, 20], [8, 12, 12], and [16, 16, 16, 16], and cacheMode is fixed to 0. Atlas A5 products support only [16, 24, 24] and [24, 20, 20].
  - The aclnnRopeWithSinCosCacheV2 API supports mropeSection values of [11, 11, 10], [16, 24, 24], [24, 20, 20], [8, 12, 12], and [16, 16, 16, 16], and cacheMode supports only 0 and 1. When mropeSection is [16, 16, 16, 16], cacheMode supports only 0. This API does not support Atlas A5 products.

## Calling Description

| Calling Method | Sample Code | Description |
| --- | --- | --- |
| aclnn API | [test_aclnn_rope_with_sin_cos_cache](examples/test_aclnn_rope_with_sin_cos_cache.cpp) | Invokes the RopeWithSinCosCache operator through the [aclnnRopeWithSinCosCache](docs/aclnnRopeWithSinCosCache_en.md) API. |
