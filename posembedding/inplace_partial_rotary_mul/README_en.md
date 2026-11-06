# InplacePartialRotaryMul

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Description

- **Operator function**: performs in-place computation of single-path rotary position encoding. It directly modifies the input tensor and does not produce a new output tensor. The `partial_slice` parameter enables partial rotary position encoding on the last dimension of the input tensor, so rotary position encoding applies only to the data in the specified range.
- **Computation formula**:

    interleave mode (rotary_mode equals 1):
    $$
    x1 = x[..., ::2]
    $$

    $$
    x2 = x[..., 1::2]
    $$

    $$
    x\_rotate = torch.cat((-x2, x1), dim=-1)
    $$

    $$
    x = x * cos + x\_rotate * sin
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
    <td>xRef</td>
    <td>Input</td>
    <td>x in the formula, the tensor on which rotary position encoding is performed. In in-place mode, xRef also serves as the output and stores the result.</td>
    <td>BFLOAT16, FLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>cos</td>
    <td>Input</td>
    <td>cos in the formula, the position encoding tensor involved in the computation.</td>
    <td>BFLOAT16, FLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>sin</td>
    <td>Input</td>
    <td>sin in the formula, the position encoding tensor involved in the computation.</td>
    <td>BFLOAT16, FLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>rotary_mode</td>
    <td>Input</td>
    <td>Rotary mode in the formula. Currently only the interleave mode (rotary_mode=1) is supported.</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>partial_slice</td>
    <td>Input</td>
    <td>Slice range [start, end) for partial rotation. It applies to the last dimension. The default value is [0, 0].</td>
    <td>INT64 array</td>
    <td>-</td>
  </tr>
</tbody>
</table>

## Constraints

- Non-contiguous tensors are not supported.
- The last dimension (D) of the input tensor xRef must not exceed 1024.
- In interleave mode (rotary_mode = 1), the last dimension (D) of xRef must be a multiple of 2.
- The last dimensions of the input tensors cos and sin must be the same size, and the size must equal the slice length of partialSlice (that is, partialSlice[1] - partialSlice[0]).
- partialSlice constraints: sliceStart ≥ 0, sliceEnd ≥ 0, sliceEnd ≤ the size of the last dimension (D) of xRef, sliceLength = sliceEnd - sliceStart >= 0. When sliceEnd equals sliceStart, rotary position encoding is not performed and the function returns directly.
- Only the interleave mode (rotary_mode = 1) is supported.
- In-place execution: the input xRef and the output share the same tensor, and the computation result is written back to the input xRef directly.
- The shape of xRef is BSND.
- The shapes of cos/sin must be broadcastable to xRef, subject to the following constraints:

  - <term>Ascend 950PR&950DT products</term>:
    The shapes of cos/sin currently support only the BSND, B1ND, B11D, and 111D layouts.

  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>:
    The shapes of cos/sin currently support only the BS1D and B11D layouts.

## Calling Instructions

| Calling Method           | Calling Sample                                                                                    | Description                                                                                                  |
|----------------|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| aclnn call | [test_aclnn_inplace_partial_rotary_mul](./examples/test_aclnn_inplace_partial_rotary_mul.cpp) | Calls the InplacePartialRotaryMul operator through the [aclnnInplacePartialRotaryMul](docs/aclnnInplacePartialRotaryMul_en.md) API.             |
| PyTorch API | [inplace_partial_rotary_mul](docs/torchapi_inplace_partial_rotary_mul_en.md) | Calls the InplacePartialRotaryMul operator through the [cann_ops_transformer.inplace_partial_rotary_mul](docs/torchapi_inplace_partial_rotary_mul_en.md) API. |
| Graph mode call | [test_geir_inplace_partial_rotary_mul](./examples/test_geir_inplace_partial_rotary_mul.cpp) | Calls the InplacePartialRotaryMul operator by building the graph through the [operator IR](./op_graph/inplace_partial_rotary_mul_proto.h). |
