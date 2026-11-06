# BlockAttentionResiduals

English | [简体中文](./README.md)

## Product Support Status

| Product                                                 | Supported |
| :------------------------------------------------------- | :-------: |
| <term>Ascend 950PR&950DT products</term>                 |    √     |
| <term>Atlas A3 products</term> |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>       |    ×     |
| <term>Atlas inference products</term>                   |    ×     |
| <term>Atlas training products</term>                    |    ×     |

## Function Description

- **Operator function**: Concatenates `partialBlock` and `blockRes` along the block dimension, performs RMS normalization, `normWeight ⊙ projWeight` projection scoring, and Softmax weighted fusion, and outputs `hiddenStates`. When `needBackward` is true, `invNorm` and `probs` required for backward propagation are also output.

- **Computation formula**:

  $$
  v_{t,i,h} =
  \begin{cases}
  block\_res_{t,i,h}, & i < N \\
  partial\_block_{t,h}, & i = N
  \end{cases}
  $$

  $$
  variance_{t,i} = \frac{1}{H}\sum_{h=0}^{H-1} v_{t,i,h}^{2}
  $$

  $$
  inv\_rms_{t,i} = \frac{1}{\sqrt{variance_{t,i} + norm\_eps}}
  $$

  $$
  k_{t,i,h} = v_{t,i,h} \cdot inv\_rms_{t,i}
  $$

  $$
  score\_weight_{h} = norm\_weight_{h} \cdot proj\_weight_{0,h}
  $$

  $$
  s_{t,i} = \sum_{h=0}^{H-1} k_{t,i,h} \cdot score\_weight_{h}
  $$

  $$
  probs_{t,i} = \frac{e^{s_{t,i}}}{\sum_{j=0}^{N} e^{s_{t,j}}}
  $$

  $$
  hidden\_states_{t,h} = \sum_{i=0}^{N} probs_{t,i} \cdot v_{t,i,h}
  $$

## Parameter Description

  <table style="table-layout: auto; width: 100%">
    <thead>
      <tr>
        <th style="white-space: nowrap">Parameter</th>
        <th style="white-space: nowrap">Input/Output/Attribute</th>
        <th style="white-space: nowrap">Description</th>
        <th style="white-space: nowrap">Data Type</th>
        <th style="white-space: nowrap">Data Format</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>partialBlock</td>
        <td>Input</td>
        <td>The N-th row of value after concatenation, with shape (T, H).</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>blockRes</td>
        <td>Input</td>
        <td>The first N rows of value, with shape (T, N, H). The dtype must be the same as that of partialBlock.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>projWeight</td>
        <td>Input</td>
        <td>Projection weight, with shape (H) or (1, H). The dtype must be the same as that of partialBlock.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>normWeight</td>
        <td>Input</td>
        <td>RMS scaling weight, with shape (H). The dtype must be the same as that of partialBlock.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>validBlockNum</td>
        <td>Optional attribute</td>
        <td>Number of valid blocks. -1 (default) indicates that N of blockRes is used. Currently only -1 or a value equal to N is supported.</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>normEps</td>
        <td>Optional attribute</td>
        <td>Numerical stability parameter for RMS normalization. The value must be greater than 0. Default value: 1e-6.</td>
        <td>DOUBLE</td>
        <td>-</td>
      </tr>
      <tr>
        <td>needBackward</td>
        <td>Optional attribute</td>
        <td>Whether to output the backward intermediate quantities invNorm and probs. Default value: false.</td>
        <td>BOOL</td>
        <td>-</td>
      </tr>
      <tr>
        <td>hiddenStates</td>
        <td>Output</td>
        <td>Weighted fusion result, with shape (T, H). The dtype must be the same as that of partialBlock.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>invNorm</td>
        <td>Output</td>
        <td>Row-wise normalization coefficient output when backward propagation is required. The output is invalid when needBackward is false.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>probs</td>
        <td>Output</td>
        <td>Softmax probabilities output when backward propagation is required. The output is invalid when needBackward is false.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>

## Constraints

- T is greater than or equal to 0, H is greater than or equal to 1, and the value range of N is 1 to 100.
- The data types of partialBlock, blockRes, projWeight, normWeight, and hiddenStates must be the same.
- validBlockNum must be -1 (default, uses N) or equal to the N of blockRes.
- The computation semantics are consistent on <term>Ascend 950PR&950DT products</term>, <term>Atlas A3 products</term>, and <term>Atlas A2 products</term>.

## Invocation Description

| Invocation Method | Invocation Sample         | Description                                                  |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_block_attention_residuals](examples/test_aclnn_block_attention_residuals.cpp) | Invokes the BlockAttentionResiduals operator through the [aclnnBlockAttentionResiduals](docs/aclnnBlockAttentionResiduals_en.md) API. |
| PyTorch API | - | Invokes the BlockAttentionResiduals operator through the [cann_ops_transformer.block_attention_residuals](docs/torchapi_block_attention_residuals_en.md) API. |
