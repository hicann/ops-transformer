# FusedFloydAttentionGrad

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

- Operator function: In the training scenario, the operator computes the backward output of Floyd attention. Compared with traditional FA, FloydAttn additionally treats seq as the batch axis when computing qk/pv attention, thereby converting the computation into batchMatmul.

- Calculation formula:

    The forward calculation formulas of attention are as follows:

    $$
    P=Softmax(Mask(scale*(Q*K_1^T + Q*K_2^T), atten\_mask)) \\
    Y=(P*V_1+P*V_2)
    $$

    The backward calculation formulas of attention are as follows:

    $$
    S=Softmax(S)
    $$

    $$
    dV_1=P^TdY
    $$

    $$
    dV_2=P^TdY
    $$

    $$
    dQ=\frac{((dS)*K_1)}{\sqrt{d}}+\frac{((dS)*K_2)}{\sqrt{d}}
    $$

    $$
    dK_1=\frac{((dS)^T*Q)}{\sqrt{d}}
    $$

    $$
    dK_2=\frac{((dS)^T*Q)}{\sqrt{d}}
    $$

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1576px">
  <colgroup>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>query</td>
      <td>Input</td>
      <td>Input Q in the formula.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key1</td>
      <td>Input</td>
      <td>Input K1 in the formula.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value1</td>
      <td>Input</td>
      <td>Input V1 in the formula.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key2</td>
      <td>Input</td>
      <td>Input K2 in the formula.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value2</td>
      <td>Input</td>
      <td>Input V2 in the formula.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>Input</td>
      <td>Input dY in the formula.</td>
      <td>FLOAT16, BFLOAT16</td>
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
      <td>dqOut</td>
      <td>Output</td>
      <td>dQ in the formula, which indicates the gradient of query.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dk1Out</td>
      <td>Output</td>
      <td>dK1 in the formula, which indicates the gradient of key1.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dv1Out</td>
      <td>Output</td>
      <td>dV1 in the formula, which indicates the gradient of value1.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dk2Out</td>
      <td>Output</td>
      <td>dK2 in the formula, which indicates the gradient of key2.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dv2Out</td>
      <td>Output</td>
      <td>dV2 in the formula, which indicates the gradient of value2.</td>
      <td>FLOAT16, BFLOAT16</td>
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
- query and dy/attentionIn must have the same shape.
- softmaxMax and softmaxSum must have the same shape.
- D supports only 32/64/128.
- In the current version, attenMaskOptional must be passed in, and null pointers are not supported. If it is not passed in, tiling returns a failure.

## Calling Instructions

| Calling Method           | Calling Sample                                                                                                              | Description                                                                                                                    |
|----------------|-------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| aclnn invocation | [test_aclnn_fused_floyd_attention_grad](./examples/test_aclnn_fused_floyd_attention_grad.cpp)                     | Calls the [aclnnFusedFloydAttentionGrad](docs/aclnnFusedFloydAttentionGrad_en.md) operator through the API.                   |
