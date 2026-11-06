# FusedFloydAttention

English | [简体中文](./README.md)

## Product Support Status

|Product      | Supported |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT products</term>|      ×     |
|<term>Atlas A3 products</term>|     √     |
|<term>Atlas A2 products</term>|     √     |
|<term>Atlas 200I/500 A2 inference products</term>|     ×      |
|<term>Atlas inference products</term>|     ×     |
|<term>Atlas training products</term>|     ×     |

## Function Description

- Operator function: In the training scenario, the operator implements the computation of multi-dimensional self-attention using the FloydAttention algorithm.

- Calculation formula:

    The forward calculation formulas of attention are as follows:

    $$
    weights = Softmax(attenMask + scale*(einsum(query, key1^T) + einsum(query, key2^T)))
    $$

    $$
    attention\_out = einsum(weights, value1) + einsum(weights, value2)
    $$

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
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
      <td>query</td>
      <td>Input</td>
      <td>Input query in the formula.</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key1</td>
      <td>Input</td>
      <td>Input key1 in the formula.</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value1</td>
      <td>Input</td>
      <td>Input value1 in the formula.</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>attenMaskOptional</td>
      <td>Optional input</td>
      <td>atten_mask in the formula. In the current version, this parameter must be passed in, and null pointers are not supported. It indicates the attention mask. A value of 1 indicates that the position does not participate in the computation (does not take effect), and a value of 0 indicates that the position participates in the computation.</td>
      <td>BOOL, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scaleValue</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>scale in the formula, which indicates the scaling factor and serves as the scalar value of Muls in the computation flow.</li>
          <li>The default value is 1.0.</li>
        </ul>
      </td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmaxMaxOut</td>
      <td>Output</td>
      <td>Max intermediate result of the Softmax computation, which is used for backward computation.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>softmaxSumOut</td>
      <td>Output</td>
      <td>Sum intermediate result of the Softmax computation, which is used for backward computation.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>attentionOut</td>
      <td>Output</td>
      <td>attention_out in the formula.</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## Constraints

- When this API is used with PyTorch, ensure that the versions of the CANN-related packages match those of the PyTorch-related packages
- The constraints on data shapes are as follows:
  - B: The value range is 1\~2K.
  - H: The value range is 1\~256.
  - N: The value range is 16\~1M and N%16==0.
  - M: The value range is 128\~1M and M%128==0.
  - K: The value range is 128\~1M and K%128==0.
  - D: The value range is 32/64/128.

- Dimensions 0/2/4 of query and key1 must be the same.
- key1 and value1 must have the same shape.
- key2 and value2 must have the same shape.
- softmaxMax and softmaxSum must have the same shape.
- D supports only 32/64/128.
- In the current version, attenMaskOptional must be passed in, and null pointers are not supported. If it is not passed in (a null pointer is passed in), the first-stage API returns 561002.

## Calling Instructions

| Calling Method           | Calling Sample                                                                                    | Description                                                                                                  |
|----------------|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| aclnn invocation | [test_aclnn_fused_floyd_attention](./examples/test_aclnn_fused_floyd_attention.cpp) | Calls the FusedFloydAttention operator through the [aclnnFusedFloydAttention](docs/aclnnFusedFloydAttention_en.md) API. |
