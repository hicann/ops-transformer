# FlashAttentionScoreGrad

English | [简体中文](./README.md)

## Product Support

|Product      | Supported |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT products</term>|      √     |
|<term>Atlas A3 training products</term>|      √     |
|<term>Atlas A3 inference products</term>|      ×     |
|<term>Atlas A2 training products</term>|      √     |
|<term>Atlas A2 inference products</term>|      ×     |

## Function Description

- Operator function: Computes the backward output of attention in the training scenario, that is, the backward computation of FlashAttentionScore:

  - When pseType=1, add must be performed before mul.
  - When pseType≠1, mul must be performed before add.

- Computation formula:

  The known forward computation formula of attention is:
  - When pseType=1, the formula is as follows:

    $$
    Y=Dropout(Softmax(Mask(\frac{QK^T+pse}{\sqrt{d}}),atten\_mask),keep\_prob)V
    $$

  - When pseType≠1, the formula is as follows:

    $$
    Y=Dropout(Softmax(Mask(\frac{QK^T}{\sqrt{d}}+pse),atten\_mask),keep\_prob)V
    $$

  For ease of expression, the computation formula is represented using the variables $S$ and $P$:

  $$
  S=Mask(\frac{QK^T}{\sqrt{d}}+pse),atten\_mask
  $$

  $$
  P=Dropout(Softmax(S),keep\_prob)
  $$

  $$
  Y=PV
  $$

  The backward computation formula of attention is:

  $$
  dV=P^TdY
  $$

  $$
  dQ=\frac{((dS)*K)}{\sqrt{d}}
  $$

  $$
  dK=\frac{((dS)^T*Q)}{\sqrt{d}}
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
      <th>Parameter</th>
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
      <td>FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key</td>
      <td>Input</td>
      <td>Input K in the formula.</td>
      <td>FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value</td>
      <td>Input</td>
      <td>Input V in the formula.</td>
      <td>FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>Input</td>
      <td>Input dY in the formula.</td>
      <td>FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pseShiftOptional</td>
      <td>Optional input</td>
      <td>pse in the formula, indicating positional encoding.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dropMaskOptional</td>
      <td>Optional input</td>
      <td>Dropout in the formula, indicating the dropout mask. A value of 1 indicates that the data is retained, and a value of 0 indicates that the data is discarded.</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>attenMaskOptional</td>
      <td>Optional input</td>
      <td>atten_mask in the formula, indicating the attention mask. A value of 1 indicates that the position does not participate in computation (does not take effect), and a value of 0 indicates that the position participates in computation.</td>
      <td>BOOL, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>softmaxMaxOptional</td>
      <td>Optional input</td>
      <td>Intermediate output of the forward attention computation.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>softmaxSumOptional</td>
      <td>Optional input</td>
      <td>Intermediate output of the forward attention computation.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>attentionInOptional</td>
      <td>Optional input</td>
      <td>Final output of the forward attention computation.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scaleValue</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>scale in the formula, indicating the scale factor. It serves as the scalar value of Muls in the computation flow.</li>
          <li>Default value: 1.0.</li>
        </ul>
      </td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>keepProb</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>keep_prob in the formula, indicating the probability that data is retained.</li>
          <li>Default value: 1.0.</li>
        </ul>
      </td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dqOut</td>
      <td>Output</td>
      <td>dQ in the formula, indicating the gradient of query.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dkOut</td>
      <td>Output</td>
      <td>dK in the formula, indicating the gradient of key.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dvOut</td>
      <td>Output</td>
      <td>dV in the formula, indicating the gradient of value.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>
<ul>
  <li><term>Atlas A2 training products</term>: FLOAT8_E5M2 and FLOAT8_E4M3FN are not supported.</li>
  <li><term>Atlas A3 training products</term>: FLOAT8_E5M2 and FLOAT8_E4M3FN are not supported.</li>
</ul>

## Constraints

- The data types of the input query, key, value, and pseShiftOptional must be consistent.
- The input_layout of the input query, key, value, and dy must be consistent.
- Constraints on the data shape are described as follows, taking BSND and BNSD of inputLayout as an example (H=N\*D for BSH and SBH):
  - B: The value range is 1\~2M. When prefixOptional is used, the maximum supported value of B is 2K.
  - N: The value range is 1\~256.
  - S: The value range is 1\~1M.
  - D:
      - Atlas A2 training products: The value range is 1\~512.
      - Atlas A3 training products: The value range is 1\~512.
      - Ascend 950PR&950DT products: The value range is 1\~768.
- The value range of keepProb is (0, 1].
- In some scenarios, an excessive amount of computation may cause operator execution to time out (with an aicore error, where errorStr is: timeout or trap error). In this case, axis splitting is recommended. Note: The amount of computation is affected by parameters such as B, S, N, and D; the larger the values, the greater the amount of computation.
- When pseType is 2 or 3, only Sq equal to Skv is supported currently.

## Invocation Description

| Invocation Method           | Sample Code                                                                                                              | Description                                                                                                                    |
|----------------|-------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| aclnn invocation | [test_aclnn_flash_attention_score_grad_v2](./examples/test_aclnn_flash_attention_score_grad_v2.cpp)                     | In non-TND scenarios, the FlashAttentionGrad operator is invoked through the [aclnnFlashAttentionScoreGradV2](docs/aclnnFlashAttentionScoreGradV2_en.md) interface.                   |
| Graph mode invocation | [test_geir_flash_attention_score_grad](./examples/test_geir_flash_attention_score_grad.cpp) | The FlashAttentionGrad operator is invoked by constructing a graph with [Ascend IR](./docs/ascendir_FlashAttentionScoreGrad_en.md). |

## Reference Resources

- [FAG Operator Design Description](./docs/FAG_design_en.md)
