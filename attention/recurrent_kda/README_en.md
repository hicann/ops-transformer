# RecurrentKda

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

- Operator function: Performs the recurrent forward computation of KDA (Kimi Delta Attention), targeting the decode and MTP (Multi-Token Prediction) short sequence scenarios.

- Computation formula:

  For each time step $t$, let $\bar q_t$ be the query multiplied by the scaling coefficient, $\bar k_t$ be the key, $d_t$ be the decay coefficient corresponding to the gate, and $\bar\beta_t$ be the effective value of beta. The computation formulas of the recurrent state $S_t$ and output $o_t$ are as follows:

  $$
  \bar S_t = S_{t-1}\odot d_t
  $$

  $$
  \Delta_t = \bar\beta_t(v_t-\bar S_t\bar k_t)
  $$

  $$
  S_t = \bar S_t+\Delta_t\bar k_t^T
  $$

  $$
  o_t = S_t\bar q_t
  $$

  where $S_{t-1},S_t\in R^{V_{dim}\times K_{dim}}$, $q_t,k_t\in R^{K_{dim}}$, and $v_t,o_t\in R^{V_{dim}}$. $d_t$ is broadcast along the K dimension of the state matrix. The operator supports L2 normalization of query and key in the kernel, conversion of the raw gate, and sigmoid computation of beta.

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
    <td>The k in the formula.</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>value</td>
    <td>Input</td>
    <td>The v in the formula.</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gate</td>
    <td>Input</td>
    <td>The step log gate or raw gate corresponding to the decay coefficient.</td>
    <td>FLOAT16, BFLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>beta</td>
    <td>Input</td>
    <td>Delta update coefficient.</td>
    <td>FLOAT16, BFLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>initialStateRef</td>
    <td>Input&Output</td>
    <td>State matrix, which is the input S in the formula.</td>
    <td>BFLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>cuSeqlensOptional</td>
    <td>Optional input</td>
    <td>Cumulative offsets of variable-length sequences.</td>
    <td>INT32, INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>ssmStateIndicesOptional</td>
    <td>Optional input</td>
    <td>Slot indices of the state pool.</td>
    <td>INT32, INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>aLogOptional</td>
    <td>Optional input</td>
    <td>A_log parameter used for converting the raw gate in the kernel.</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>dtBiasOptional</td>
    <td>Optional input</td>
    <td>Bias used for converting the raw gate in the kernel.</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>numAcceptedTokensOptional</td>
    <td>Optional input</td>
    <td>Number of tokens accepted by each sequence in speculative decoding.</td>
    <td>INT32, INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>attnOut</td>
    <td>Output</td>
    <td>The o in the formula.</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>finalState</td>
    <td>Output</td>
    <td>Final state matrix.</td>
    <td>BFLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## Constraints

- The input tensor shapes, optional parameter dependencies, and attribute values must satisfy certain constraints. For details, see [aclnnRecurrentKda](docs/aclnnRecurrentKda_en.md).

## Calling Instructions

| Calling Method | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn API | [aclnnRecurrentKda](docs/aclnnRecurrentKda_en.md) | Calls the RecurrentKda operator through [aclnnRecurrentKda](docs/aclnnRecurrentKda_en.md) |
| PyTorch API | - | Calls the recurrent_kda operator through the [recurrent_kda](docs/torchapi_recurrent_kda_en.md) API. |
