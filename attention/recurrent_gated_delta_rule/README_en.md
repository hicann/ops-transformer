# RecurrentGatedDeltaRule

English | [简体中文](./README.md)

## Product Support

|Product      | Supported |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT products</term>|      √     |
|<term>Atlas A3 products</term>|      √     |
|<term>Atlas A2 products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function Description

- Operator function: Performs the variable-step Recurrent Gated Delta Rule computation.

- Computation formula:

  Recurrent Gated Delta Rule (RGDR) is an operator applied to recurrent neural networks, and is also applied in a linear attention mechanism.
  At each time step $t$, the network computes the current output $o_t$ and the new hidden state $S_t$ based on the current inputs $q_t$, $k_t$, and $v_t$, and the previous hidden state $S_{t-1}$.
  In this process, the gating units determine how much new information is stored in the hidden state and how much old information is forgotten.

  $$
  S_t := S_{t-1}(\alpha_t Diag(\alpha_{kt})(I - \beta_t k_t k_t^T)) + \beta_t v_t k_t^T = \alpha_t S_{t-1}Diag(\alpha_{kt}) + \beta_t (v_t - \alpha_t S_{t-1}Diag(\alpha_{kt})k_t)k_t^T
  $$

  $$
  o_t := \frac{S_t q_t}{\sqrt{d_k}}
  $$

  where $S_{t-1},S_t \in R^{d_v \times d_k}$, $q_t, k_t \in R^{d_k}$, $v_t \in R^{d_v}$, $\alpha_t \in R$, $\alpha_k \in R^{d_k}$, $\beta_t \in R$, and $o \in R^{d_v}$.

## Parameter Description

<table style="undefined;table-layout: fixed; width: 900px"><colgroup>
<col style="width: 180px">
<col style="width: 120px">
<col style="width: 200px">
<col style="width: 300px">
<col style="width: 100px">
</colgroup>
<thead>
  <tr>
    <th>Parameter Name</th>
    <th>Input/Output</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Data Format</th>
  </tr></thead>
<tbody>
  <tr>
    <td>query</td>
    <td>Input</td>
    <td>The q in the formula.</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>key</td>
    <td>Input</td>
    <td>The input k in the formula.</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>value</td>
    <td>Input</td>
    <td>The input v in the formula.</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>beta</td>
    <td>Input</td>
    <td>The β in the formula.</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>stateRef</td>
    <td>Input&Output</td>
    <td>State matrix, which is the input S in the formula.</td>
    <td>BFLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>g</td>
    <td>Input</td>
    <td>Decay coefficient, which is α=e^g in the formula</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gk</td>
    <td>Input</td>
    <td>Decay coefficient, which is αk=e^gk in the formula</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>out</td>
    <td>Output</td>
    <td>The o in the formula.</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## Constraints

- The shapes of the input tensors must satisfy certain constraints. For details, see [aclnnRecurrentGatedDeltaRule](docs/aclnnRecurrentGatedDeltaRule_en.md).

## Calling Instructions

| Calling Method | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn API | [test_aclnn_recurrent_gated_delta_rule.cpp](./examples/test_aclnn_recurrent_gated_delta_rule.cpp) | Calls the aclnnRecurrentGatedDeltaRule operator through [aclnnRecurrentGatedDeltaRule](docs/aclnnRecurrentGatedDeltaRule_en.md) |
