# PartialRotaryMulQuant

English | [简体中文](./README.md)

## Product Support

| Product | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function Description

- **Operator function**: Performs partial rotary position embedding on the `partial_slice = [start, end)` interval of the last dimension of the input tensor `x`, then multiplies the complete result by the optional scalar `scale` and quantizes the result to the HiFloat8 output `y`. The non-rotated prefix keeps its original values and participates in scaling and quantization. `y_scale` is an empty FLOAT32 tensor.
- **Computation formula**:

    Let `D = x.shape[-1]` and `L = end - start`. For a non-empty rotation interval, `end = D` is required. Define:

    $$
    v = x[..., start:end]
    $$

    `cos` and `sin` are broadcast to the shape of `v`, and `concat` denotes concatenation along the last dimension.

    half mode (`rotary_mode = 0`):

    $$
    a = v[..., :L/2],\quad b = v[..., L/2:]
    $$

    $$
    z = v * cos + concat(-b, a) * sin
    $$

    interleave mode (`rotary_mode = 1`):

    $$
    r[..., 2j] = -v[..., 2j+1],\quad r[..., 2j+1] = v[..., 2j]
    $$

    $$
    z = v * cos + r * sin,\quad 0 \le j < L/2
    $$

    quarter mode (`rotary_mode = 2`): divides `v` into four consecutive segments `a`, `b`, `c`, and `d`, each with the length `L/4`.

    $$
    z = v * cos + concat(-b, a, -d, c) * sin
    $$

    deepseek_interleave mode (`rotary_mode = 3`): groups the even and odd positions into two halves respectively, and the output is concatenated by the two halves.

    $$
    a = v[..., 0::2],\quad b = v[..., 1::2]
    $$

    $$
    z = concat(a, b) * cos + concat(-b, a) * sin
    $$

    Concatenate the non-rotated prefix and the rotation result:

    $$
    t = concat(x[..., :start], z)
    $$

    The computation uses FLOAT32 intermediate results and converts the complete result after scaling:

    $$
    \alpha = \begin{cases}
    1, & \text{scale is not passed in} \\
    scale[0], & \text{a single-element scale tensor is passed in}
    \end{cases}
    $$

    $$
    y = Cast_{HiFloat8}(\alpha * t, round\_mode)
    $$

    In this case, `y_scale` is an empty FLOAT32 tensor with the shape `[0]`. When `start == end`, rotation is skipped and `t = float32(x)`. Scaling and quantization are still performed on the complete input.

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1200px">
<colgroup>
  <col style="width: 100px">
  <col style="width: 100px">
  <col style="width: 650px">
  <col style="width: 250px">
  <col style="width: 100px">
</colgroup>
<thead>
  <tr>
    <th>Parameter</th>
    <th>Input/Output/Attribute</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Data Format</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>x</td>
    <td>Input</td>
    <td>Four-dimensional tensor on which partial rotary position embedding and quantization are performed.</td>
    <td>BFLOAT16, FLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>cos</td>
    <td>Input</td>
    <td>Cosine position embedding tensor that participates in the rotation computation. The data type must be the same as that of x.</td>
    <td>BFLOAT16, FLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>sin</td>
    <td>Input</td>
    <td>Sine position embedding tensor that participates in the rotation computation. The shape and data type must be the same as those of cos.</td>
    <td>BFLOAT16, FLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>scale</td>
    <td>Input</td>
    <td>Optional single-element scaling tensor. It multiplies the complete result including the non-rotated prefix. The default value is 1 when this parameter is not passed in.</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y</td>
    <td>Output</td>
    <td>HiFloat8 quantization result. The shape is the same as that of x.</td>
    <td>HIFLOAT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y_scale</td>
    <td>Output</td>
    <td>Empty FLOAT32 tensor with the shape [0].</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y_dtype</td>
    <td>Attribute</td>
    <td>Output quantization type. Currently only 0 is supported, indicating HiFloat8. The default value is 0.</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>quant_mode</td>
    <td>Attribute</td>
    <td>Quantization mode. The default value is 0. This parameter is retained for interface compatibility and is currently ignored.</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>round_mode</td>
    <td>Attribute</td>
    <td>HiFloat8 rounding mode. Supports round and hybrid. An empty string is processed as round. The default value is round. hybrid corresponds to the CAST_HYBRID hybrid rounding.</td>
    <td>STRING</td>
    <td>-</td>
  </tr>
  <tr>
    <td>axis</td>
    <td>Attribute</td>
    <td>Quantization axis. The default value is -1. This parameter is retained for interface compatibility and is currently ignored.</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>rotary_mode</td>
    <td>Attribute</td>
    <td>Rotation mode. 0 indicates half, 1 indicates interleave, 2 indicates quarter, and 3 indicates deepseek_interleave. The default value is 0.</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>partial_slice</td>
    <td>Attribute</td>
    <td>Rotation interval [start, end) on the last dimension. The default value is [0, 0]. When start equals end, RoPE is skipped, while scaling and quantization are still performed.</td>
    <td>INT64 array</td>
    <td>-</td>
  </tr>
</tbody>
</table>

## Constraints

- The input `x`, `cos`, and `sin` and the output `y` are all four-dimensional ND tensors. The data types of `x`, `cos`, and `sin` must be the same, and the shapes of `cos` and `sin` must be the same.
- Let the shape of `x` be `(B, S, N, D)`. Each dimension size of `x` must be positive, and `0 < D <= 1024`. The shape of `y` is the same as that of `x`.
- When a non-empty array is passed in, `partial_slice` must contain two integers and satisfy `0 <= start <= end <= D`. A non-empty rotation interval requires `end = D`, and the size of the last dimension of `cos` and `sin` must equal `L = end - start`.
- A non-empty rotation interval requires that each dimension size of `cos` and `sin` be positive and that the first three dimensions be broadcastable with `x`.
- The ACLNN API makes non-contiguous inputs contiguous.

- `y_dtype` must be 0, and four rotation modes are supported.
- For a non-empty rotation interval, the half, interleave, and deepseek_interleave modes require that both `D` and `L` be multiples of 2, and the quarter mode requires that both `D` and `L` be multiples of 4. `D` or `start` does not need to be aligned to 64.
- For a non-empty rotation interval, each of the first three dimensions of `cos` and `sin` must be 1 or equal to the corresponding dimension of `x`.
- When `partial_slice` is not passed in or an empty array is passed in, it is processed as `[0, 0]`. When `start == end`, RoPE and the rotation dimension divisibility check are skipped. `cos` and `sin` do not need to satisfy the broadcast relationship or the last-dimension length constraint, but four-dimensional tensors with the same shape and the same data type as `x` must still be provided.
- When `scale` is not empty, it must be a single-element FLOAT32 tensor. Passing scale by channel or by group is not supported.
- `round_mode = "hybrid"` involves random rounding behavior. Multiple executions with the same input may produce different results.
- `y_scale` is an empty FLOAT32 tensor with the shape `[0]`. A tensor description must still be provided during invocation.

## Calling Instructions

| Calling Method | Calling Sample | Description |
|----------------|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| aclnn invocation | [test_aclnn_partial_rotary_mul_quant](./examples/test_aclnn_partial_rotary_mul_quant.cpp) | Calls the PartialRotaryMulQuant operator through the [aclnnPartialRotaryMulQuant](./docs/aclnnPartialRotaryMulQuant.md) API. |
| PyTorch API | [test_torch_partial_rotary_mul_quant](./examples/test_torch_partial_rotary_mul_quant.py) | Calls the operator through the [cann_ops_transformer.partial_rotary_mul_quant](./torch_extension/partial_rotary_mul_quant.py) API and returns a uint8 tensor encoded in HiFloat8 and an empty FLOAT32 tensor. |
| Graph mode invocation | [Operator IR](./op_graph/partial_rotary_mul_quant_proto.h) | Calls the operator through PartialRotaryMulQuant operator IR graph construction, with y_dtype set to 0. |
