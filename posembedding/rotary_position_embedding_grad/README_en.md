# RotaryPositionEmbeddingGrad

English | [简体中文](./README.md)

## Product Support

| Product                                                      |  Supported  |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR&950DT products</term>                     |    √     |
| <term>Atlas A3 products</term>                               |    √    |
| <term>Atlas A2 products</term>                               |    √    |
| <term>Atlas 200I/500 A2 inference products</term>            |    ×    |
| <term>Atlas inference products</term>                        |    ×    |
| <term>Atlas training products</term>                         |    ×    |

## Description

- **Operator function**: Performs the backward computation of the single-path rotary position embedding [RotaryPositionEmbedding](../rotary_position_embedding/README_en.md).

- **Calculation Formula**:

    In the forward computation of the rotary position embedding, let `dims` be the list of broadcast axes. The calculation formula can be expressed as follows:

    - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>:

    (1) half mode (mode equals 0):
    $$
    dy1, dy2 = chunk(dy, chunks=2, dim=-1)
    $$

    $$
    cos1, cos2 = chunk(cos, chunks=2, dim=-1)
    $$

    $$
    sin1, sin2 = chunk(sin, chunks=2, dim=-1)
    $$

    $$
    x1, x2 = chunk(x, chunks=2, dim=-1)
    $$

    $$
    dx = cat((cos1 * dy1 + sin2 * dy2, cos2 * dy2 - sin1 * dy1), dim=-1)
    $$

    $$
    dcos = sum(dy * x, dims)
    $$

    $$
    dsin = sum(dy * cat((-x2, x1), dim=-1), dims)
    $$

    (2) interleave mode (mode equals 1):
    $$
    dy1, dy2 = dy[..., :: 2], dy[..., 1 :: 2]
    $$

    $$
    cos1, cos2 = cos[..., :: 2], cos[..., 1 :: 2]
    $$

    $$
    sin1, sin2 = sin[..., :: 2], sin[..., 1 :: 2]
    $$

    $$
    x1, x2 = x[..., :: 2], x[..., 1 :: 2]
    $$

    $$
    dx = stack((cos1 * dy1 + sin2 * dy2, cos2 * dy2 - sin1 * dy1), dim=-1).reshape(dy.shape)
    $$

    $$
    dcos = sum(dy * x, dims)
    $$

    $$
    dsin = sum(dy * stack((-x2, x1), dim=-1).reshape(dy.shape), dims)
    $$
    - <term>Ascend 950PR&950DT products</term>:

    (3) quarter mode (mode equals 2):
    $$
    dy1, dy2, dy3, dy4 = chunk(dy, chunks=4, dim=-1)
    $$

    $$
    cos1, cos2, cos3, cos4 = chunk(cos, chunks=4, dim=-1)
    $$

    $$
    sin1, sin2, sin3, sin4 = chunk(sin, chunks=4, dim=-1)
    $$

    $$
    x1, x2, x3, x4 = chunk(x, chunks=4, dim=-1)
    $$

    $$
    dx = cat((cos1 * dy1 + sin2 * dy2, cos2 * dy2 - sin1 * dy1, cos3 * dy3 + sin4 * dy4, cos4 * dy4 - sin3 * dy3), dim=-1)
    $$

    $$
    dcos = sum(dy * x, dims)
    $$

    $$
    dsin = sum(dy * cat((-x2, x1, -x4, x3), dim=-1), dims)
    $$

    (4) interleave-half mode (mode equals 3):
    $$
    dy1, dy2 = chunk(dy, chunks=2, dim=-1)
    $$

    $$
    cos1, cos2 = chunk(cos, chunks=2, dim=-1)
    $$

    $$
    sin1, sin2 = chunk(sin, chunks=2, dim=-1)
    $$

    $$
    x1, x2 = x[..., :: 2], x[..., 1 :: 2]
    $$

    $$
    dx = stack((cos1 * dy1 + sin2 * dy2, cos2 * dy2 - sin1 * dy1), dim=-1).reshape(dy.shape)
    $$

    $$
    dcos = sum(dy * cat((x1, x2), dim=-1), dims)
    $$

    $$
    dsin = sum(dy * cat((-x2, x1), dim=-1), dims)
    $$

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1200px">
<colgroup>
  <col style="width: 50px">
  <col style="width: 50px">
  <col style="width: 200px">
  <col style="width: 100px">
  <col style="width: 50px">
</colgroup>
<thead>
  <tr>
    <th>Parameter Name</th>
    <th>Input/Output/Attribute</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Data Format</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>dy</td>
    <td>Input</td>
    <td>dy in the formula, representing the derivative of the forward computation output y.</td>
    <td>BFLOAT16, FLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>cos</td>
    <td>Input</td>
    <td>cos in the formula, an input of the forward computation. Its data type must be the same as that of dy.</td>
    <td>BFLOAT16, FLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>sin</td>
    <td>Input</td>
    <td>sin in the formula, an input of the forward computation. Its data type must be the same as that of dy.</td>
    <td>BFLOAT16, FLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>xOptional</td>
    <td>Optional input</td>
    <td>x in the formula, an input of the forward computation. If it is a null pointer, dcosOut and dsinOut are not computed.</td>
    <td>BFLOAT16, FLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>mode</td>
    <td>Input</td>
    <td>Rotation mode in the formula.</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>dxOut</td>
    <td>Output</td>
    <td>dx in the formula, the derivative of the input x.</td>
    <td>BFLOAT16, FLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>dcosOut</td>
    <td>Output</td>
    <td>dcos in the formula, the derivative of the input cos. It is valid only when xOptional is not null.</td>
    <td>BFLOAT16, FLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>dsinOut</td>
    <td>Output</td>
    <td>dsin in the formula, the derivative of the input sin. It is valid only when xOptional is not null.</td>
    <td>BFLOAT16, FLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## Constraints

  - <term>Ascend 950PR&950DT products</term>:
    The input tensor dy supports the BNSD, BSND, SBND, and TND layouts. The shape constraints of each parameter are described as follows:
    - The last dimension of the input tensors dy, cos, and sin and the output tensor dx must have the same size, which must be less than or equal to 1024. For the half, interleave, and interleave-half modes, the last dimension must be divisible by 2. For the quarter mode, the last dimension must be divisible by 4.
    - The shapes of the input tensor dy and the output tensor dx must be exactly the same.
    - The shapes of the input tensors cos and sin must be exactly the same. The shapes of cos and sin must satisfy the [broadcast relationship](../../docs/zh/context/broadcast_relationship_en.md) with dy, and the shape after broadcasting must be equal to the shape of dy.
    - When dy is TND, cos and sin support T1D and TND.

  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>:

    - The input tensor dy supports the BNSD, BSND, SBND, and TND layouts.
    - The D dimension of the input tensors dy, cos, sin, and xOptional and the output tensors dxOut, dcosOut, and dsinOut must have the same size, satisfying D<896, and must be a multiple of 2.
    - The shapes of the input tensors dy and xOptional and the output tensor dxOut must be exactly the same.
    - The shapes of the input tensors cos and sin and the output tensors dcosOut and dsinOut must be exactly the same, and the shapes of cos and sin must be exactly the same.
    - half mode:
      - B, N < 1000; when dsin and dcos need to be computed, B * N <= 1024
      - When dy is BNSD, cos and sin support 11SD, B1SD, and BNSD; when cos and sin are B1SD, B < S must be satisfied
      - When dy is BSND, cos and sin support 1S1D, BS1D, and BSND; when cos and sin are BS1D, B < S must be satisfied
      - When dy is SBND, cos and sin support S11D, SB1D, and SBND
      - When dy is TND, cos and sin support T1D and TND
    - interleave mode:
      - B * N < 1000 (N < 1000 when dy is TND)
      - When dy is BNSD, cos and sin support 11SD
      - When dy is BSND, cos and sin support 1S1D
      - When dy is SBND, cos and sin support S11D
      - When dy is TND, cos and sin support T1D

## Calling Instructions

| Calling Method  | Calling Sample                                                                          | Description                                                                                         |
|----------------|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| aclnn call | [test_aclnn_rotary_position_embedding_grad](./examples/test_aclnn_rotary_position_embedding_grad.cpp) | Calls the RotaryPositionEmbeddingGrad operator through the [aclnnRotaryPositionEmbeddingGrad](docs/aclnnRotaryPositionEmbeddingGrad_en.md) API. |
