# RotaryPositionEmbedding

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |
| <term>Kirin X90 processor products</term> | √ |
| <term>Kirin 9030 processor products</term> | √ |

## Description

- **Operator function**: Performs single-path rotary position embedding computation.
- **Computation formulas**:

    - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>:

    (1) half mode (mode equals 0):
    $$
    x1 = x[..., : x.shape[-1] // 2]
    $$

    $$
    x2 = x[..., x.shape[-1] // 2 :]
    $$

    $$
    x\_rotate = torch.cat((-x2, x1), dim=-1)
    $$

    $$
    y = x * cos + x\_rotate * sin
    $$
    (2) interleave mode (mode equals 1):
    $$
    x1 = x[..., ::2].view(-1, 1)
    $$

    $$
    x2 = x[..., 1::2].view(-1, 1)
    $$
    $$
    x\_rotate = torch.cat((-x2, x1), dim=-1).view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
    $$
    $$
    y = x * cos + x\_rotate * sin
    $$

    - <term>Ascend 950PR&950DT products</term>:

    (3) quarter mode (mode equals 2):
    $$
    x1 = x[..., : x.shape[-1] // 4]
    $$

    $$
    x2 = x[..., x.shape[-1] // 4 : x.shape[-1] // 2]
    $$
    $$
    x3 = x[..., x.shape[-1] // 2 : x.shape[-1] // 4 * 3]
    $$
    $$
    x4 = x[..., x.shape[-1] // 4 * 3 :]
    $$

    $$
    x\_rotate = torch.cat((-x2, x1, -x4, x3), dim=-1)
    $$

    $$
    y = x * cos + x\_rotate * sin
    $$
    (4) interleave-half mode (mode equals 3). This mode first extracts the inputs at odd positions to the first half and the inputs at even positions to the second half, and then performs half processing:
    $$
    x1 = x[..., ::2]
    $$

    $$
    x2 = x[..., 1::2]
    $$
    $$
    x\_part1 = torch.cat((x1, x2), dim=-1)
    $$

    $$
    x\_part2 = torch.cat((-x2, x1), dim=-1)
    $$
    $$
    y = x\_part1 * cos + x\_part2 * sin
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
    <td>x</td>
    <td>Input</td>
    <td>x in the formulas, the tensor on which rotary position embedding is performed.</td>
    <td>BFLOAT16, FLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>cos</td>
    <td>Input</td>
    <td>cos in the formulas, the positional encoding tensor that participates in the computation.</td>
    <td>BFLOAT16, FLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>sin</td>
    <td>Input</td>
    <td>sin in the formulas, the positional encoding tensor that participates in the computation.</td>
    <td>BFLOAT16, FLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>mode</td>
    <td>Input</td>
    <td>The rotation mode in the formulas.</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>out</td>
    <td>Output</td>
    <td>y in the formulas, the rotary position embedding result tensor.</td>
    <td>BFLOAT16, FLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

- <term>Kirin X90 processor products</term> and <term>Kirin 9030 processor products</term>: BFLOAT16 is not supported.

## Constraints

  - <term>Ascend 950PR&950DT products</term>:
    The input tensor x supports the BNSD, BSND, SBND, and TND layouts. The shape constraints of the parameters are described as follows:
    - The last-dimension size of the input tensors x, cos, and sin and the output tensor y must be identical and less than or equal to 1024. In the half, interleave, and interleave-half modes, the last dimension must be divisible by 2. In the quarter mode, the last dimension must be divisible by 4.
    - The shapes of the input tensor x and the output tensor y must be exactly identical.
    - The shapes of the input tensors cos and sin must be exactly identical. The shapes of cos and sin must satisfy the [broadcast relationship](../../docs/zh/context/broadcast_relationship_en.md) with x, and the broadcast shape must equal the shape of x.
    - When x is TND, cos and sin support T1D and TND.

  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>:

    The input tensor x supports the BNSD, BSND, SBND, and TND layouts.
    The D dimension size of the input tensors x, cos, and sin and the output tensor y must be identical, satisfy D<896, and be a multiple of 2.
    The shapes of the input tensor x and the output tensor y must be exactly identical.
    The shapes of the input tensors cos and sin must be exactly identical.
    - half mode:
      - B, N < 1000;
      - When x is BNSD, cos and sin support 11SD, B1SD, and BNSD
      - When x is BSND, cos and sin support 1S1D, BS1D, and BSND
      - When x is SBND, cos and sin support S11D, SB1D, and SBND
      - When x is TND, cos and sin support T1D and TND
    - interleave mode:
      - B * N < 1000 (N<1000 when x is TND)
      - When x is BNSD, cos and sin support 11SD
      - When x is BSND, cos and sin support 1S1D
      - When x is SBND, cos and sin support S11D
      - When x is TND, cos and sin support T1D
    - mode: only 0=half and 1=interleave are supported. If 2=quarter or 3=interleave-half (supported only by Ascend 950) is passed in, the API prints a warning, does not return an error code, and performs the computation with the half semantics; if rotate is not passed in, the computation is performed with the 0=half semantics.

## Calling Description

| Calling Method | Calling Sample | Description |
|---|---|---|
| aclnn invocation | [test_aclnn_rotary_position_embedding](./examples/test_aclnn_rotary_position_embedding.cpp) | Invokes the RotaryPositionEmbedding operator through the [aclnnRotaryPositionEmbedding](docs/aclnnRotaryPositionEmbedding_en.md) API. |
