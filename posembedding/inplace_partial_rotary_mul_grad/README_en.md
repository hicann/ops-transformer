# InplacePartialRotaryMulGrad

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         |  Supported   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    ×    |
| <term>Atlas A2 products</term> |    ×    |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×    |
| <term>Atlas inference products</term>                             |    ×    |
| <term>Atlas training products</term>                              |    ×    |

## Description

- **Operator function**: performs the backward computation of the partial rotary position encoding operator InplacePartialRotaryMul. The operator computes rotary position encoding gradients on the [start, end) slice of the D dimension of the input dy, and writes the dx' result in place back to the [start, end) range of dy. Data outside [start, end) of dy remains unchanged.

- **Computation formula**:

    In the forward computation of rotary position encoding, the list of broadcast axes is `dims`, and the slice range on the D dimension is `[start, end)`. Let the slice data involved in the computation be:
    $$
    dy' = dy[..., start:end]
    $$
    $$
    cos' = cos[..., start:end]
    $$
    $$
    sin' = sin[..., start:end]
    $$
    The gradient computation formulas are as follows:

    (1) half mode (rotary_mode equals 0):
    $$
    dy1', dy2' = chunk(dy', chunks=2, dim=-1)
    $$

    $$
    cos1', cos2' = chunk(cos', chunks=2, dim=-1)
    $$

    $$
    sin1', sin2' = chunk(sin', chunks=2, dim=-1)
    $$

    $$
    dx' = cat((cos1' * dy1' + sin2' * dy2', cos2' * dy2' - sin1' * dy1'), dim=-1)
    $$



    (2) interleave mode (rotary_mode equals 1):
    $$
    dy1', dy2' = dy'[..., :: 2], dy'[..., 1 :: 2]
    $$

    $$
    cos1', cos2' = cos'[..., :: 2], cos'[..., 1 :: 2]
    $$

    $$
    sin1', sin2' = sin'[..., :: 2], sin'[..., 1 :: 2]
    $$

    $$
    dx' = stack((cos1' * dy1' + sin2' * dy2', cos2' * dy2' - sin1' * dy1'), dim=-1).reshape(dy'.shape)
    $$



    (3) quarter mode (rotary_mode equals 2):
    $$
    dy1', dy2', dy3', dy4' = chunk(dy', chunks=4, dim=-1)
    $$

    $$
    cos1', cos2', cos3', cos4' = chunk(cos', chunks=4, dim=-1)
    $$

    $$
    sin1', sin2', sin3', sin4' = chunk(sin', chunks=4, dim=-1)
    $$

    $$
    dx' = cat((cos1' * dy1' + sin2' * dy2', cos2' * dy2' - sin1' * dy1', cos3' * dy3' + sin4' * dy4', cos4' * dy4' - sin3' * dy3'), dim=-1)
    $$



    (4) interleave-half mode (rotary_mode equals 3):
    $$
    dy1', dy2' = chunk(dy', chunks=2, dim=-1)
    $$

    $$
    cos1', cos2' = chunk(cos', chunks=2, dim=-1)
    $$

    $$
    sin1', sin2' = chunk(sin', chunks=2, dim=-1)
    $$

    $$
    dx' = stack((cos1' * dy1' + sin2' * dy2', cos2' * dy2' - sin1' * dy1'), dim=-1).reshape(dy'.shape)
    $$



## Parameter Description

<table style="table-layout: auto; width: 100%">
<colgroup>
  <col style="width: 10%">
  <col style="width: 10%">
  <col style="width: 40%">
  <col style="width: 20%">
  <col style="width: 10%">
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
    <td>dy in the formula, the derivative of the forward computation output y. It is updated in place to the derivative of the forward computation input x. In in-place mode, dy also serves as the output and stores the result.</td>
    <td>BFLOAT16, FLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>cos</td>
    <td>Input</td>
    <td>cos in the formula, the forward computation input. Its data type must be the same as sin.</td>
    <td>BFLOAT16, FLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>sin</td>
    <td>Input</td>
    <td>sin in the formula, the forward computation input. Its data type must be the same as cos.</td>
    <td>BFLOAT16, FLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>rotary_mode</td>
    <td>Attribute</td>
    <td>Rotary mode: 0=half, 1=interleave, 2=quarter, 3=interleave-half. Currently only the interleave mode (rotary_mode=1) is supported.</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>partial_slice</td>
    <td>Attribute</td>
    <td>Slice range on the last dimension (D dimension) of dy, in the form [start, end). It indicates that the rotary position encoding gradient computation is performed on dy[..., start:end].</td>
    <td>IntArray</td>
    <td>-</td>
  </tr>
</tbody>
</table>

## Constraints

- This operator supports only the Ascend 950 AI Processor.
- This operator supports only contiguous tensors and does not support non-contiguous tensors.
- The current version of this operator supports only the interleave mode (`rotary_mode=1`). Other modes are not supported yet.
- In-place execution: the input dy and the output share the same tensor, and the computation result is written back to the input dy directly.
- The input dy currently supports only the BSND layout, and the shapes of the input cos/sin must satisfy the broadcast relationship with dy on the B/S/N dimensions (such as BSND, 111D, 1SND, B1ND, BS1D, 11ND, B11D, 1S1D, and so on). The shape constraints on each parameter are described as follows:
  - The last dimension D of the input tensor dy must be less than or equal to 1024.
  - When the slice length (end - start) is greater than 0, the last dimension of the input tensors cos and sin must equal the slice length.
  - The shapes of the input tensors cos and sin must be exactly the same. The B, S, and N dimensions of cos and sin must satisfy the [broadcast relationship](../../docs/zh/context/broadcast_relationship_en.md) with dy, and the broadcast B, S, and N must equal the B, S, and N of dy.
  - In half, interleave, and interleave-half modes, when the slice length (end - start) is greater than 0, the slice length must be divisible by 2.
  - In quarter mode, when the slice length (end - start) is greater than 0, the slice length must be divisible by 4.
  - The input tensors cos and sin must have the same data type.
- **Empty input support**: when the slice length is 0 (the start of `partial_slice` equals end, such as `[0, 0]`), or an empty tensor exists in dy, cos, or sin (a dimension size is 0), the operator performs a no-op operation, does not compute rotary position encoding, and returns directly.


## Calling Instructions

| Calling Method           | Calling Sample                                                                                    | Description                                                                                                  |
|----------------|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| aclnn call | [test_aclnn_inplace_partial_rotary_mul_grad](./examples/arch35/test_aclnn_inplace_partial_rotary_mul_grad.cpp) | Calls the InplacePartialRotaryMulGrad operator through the [aclnnInplacePartialRotaryMulGrad](docs/aclnnInplacePartialRotaryMulGrad_en.md) API.             |
| Graph mode call | [test_geir_inplace_partial_rotary_mul_grad](./examples/arch35/test_geir_inplace_partial_rotary_mul_grad.cpp) | Calls the InplacePartialRotaryMulGrad operator by building the graph through the [operator IR](./op_graph/inplace_partial_rotary_mul_grad_proto.h). |
| PyTorch API | - | Calls the inplace_partial_rotary_mul_backward operator through the [inplace_partial_rotary_mul_backward](docs/torchapi_inplace_partial_rotary_mul_backward_en.md) API. |
