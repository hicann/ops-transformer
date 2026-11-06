# InterleaveRope

English | [简体中文](./README.md)

## Product Support Status

|Product             |  Supported  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     ×    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Kirin X90 processor products</term> | √ |
|  <term>Kirin 9030 processor products</term> | √ |

## Description

- Operator function: performs rotary position encoding on the single input x.
- Computation formula:
  $$
  q = \text{reshape}(x, [B, N, S, D//2, 2]) \cdot \text{transpose}(-1, -2) \cdot \text{reshape}([B, N, S, D])
  $$

  $$
  q_{\text{embed}} = q \cdot \text{cos} + \text{RotateHalf}(q) \cdot \sin
  $$

  RotateHalf(q) moves the elements in the second half of the D dimension of q to the first half and multiplies them by -1, and fills the second half with the values of the first half.

  $$
  \text{RotateHalf}(q)_{\text{i}} =
  \begin{cases}
  -q_{i+D//2} & \text{if } i < D//2 \\
  q_{i-D//2} & \text{otherwise}
  \end{cases}
  $$

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
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data Type</th>
      <th>Data Format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Tensor to be processed, corresponding to x in the formula.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cos</td>
      <td>Input</td>
      <td>Cosine component of the RoPE rotary position, corresponding to cos in the formula.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sin</td>
      <td>Input</td>
      <td>Sine component of the RoPE rotary position, corresponding to sin in the formula.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Result after rotary encoding, corresponding to q_embed in the formula.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

- <term>Kirin X90 processor products</term> and <term>Kirin 9030 processor products</term>: BFLOAT16 is not supported.

## Constraints

  * This API is supported for use in inference scenarios.
  * x and y must be 4-dimensional tensors with the shape (B, N, S, D).
  * cos and sin must be 4-dimensional tensors with the shape (B, N, S, D). S can be 1 or the same as the S of x, and the data type and data format must be the same as x.
  * The D dimension of the input x, cos, and sin must be 64.
  * The N dimension of cos and sin must be 1.
  * x, cos, sin, and y do not support non-contiguous tensors.

## Calling Instructions

| Calling Method           | Calling Sample                                                                                    | Description                                                                                                  |
|----------------|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| aclnn call | [test_aclnn_interleave_rope](./examples/test_aclnn_interleave_rope.cpp) | Calls the InterleaveRope operator through the [aclnnInterleaveRope](docs/aclnnInterleaveRope_en.md) API.             |
