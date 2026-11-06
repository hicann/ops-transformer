# ChunkGatedDeltaRule

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

- Operator function: Performs the chunked Gated Delta Rule computation.

- Computation formula:

  The Gated Delta Rule (GDR) is an operator applied in recurrent neural networks and also applied in a linear attention mechanism. At each time step $t$, GDR computes the current attention output $o_t$ and the new hidden state $S_t$ based on the current inputs $q_t$, $k_t$, and $v_t$, the previous hidden state $S_{t-1}$, the decay coefficient $\alpha_t$, and the update strength $\beta_t$. The computation formulas are as follows:
  $$
  S_t := S_{t-1}(\alpha_t(I - \beta_t k_t k_t^T)) + \beta_t v_t k_t^T = \alpha_t S_{t-1} + \beta_t (v_t - \alpha_t S_{t-1}k_t)k_t^T
  $$
  $$
  o_t := S_t (q_t \cdot scale)
  $$

  Here $S_{t-1},S_t \in \mathbb{R}^{D_v \times D_k}$, $q_t, k_t \in \mathbb{R}^{D_k}$, $v_t \in \mathbb{R}^{D_v}$, $\alpha_t \in \mathbb{R}$, $\beta_t \in \mathbb{R}$, and $o_t \in \mathbb{R}^{D_v}$.

  Chunked Gated Delta Rule is the chunked implementation of GDR ([reference paper](https://arxiv.org/abs/2412.06464)). It divides the input sequence into chunks to achieve a certain parallelization effect. In long-context scenarios, its computation efficiency is higher than that of Recurrent Gated Delta Rule, making it suitable for the prefill stage. Given an input sequence of length L, this operator computes the output $o_t, t \in \{1, 2, .., L\}$ at each step and the final state matrix $S_L$.

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
    <th>Parameter</th>
    <th>Input/Output/Attribute</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Data Format</th>
  </tr></thead>
<tbody>
  <tr>
    <td>query</td>
    <td>Input</td>
    <td>q in the formula.</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>key</td>
    <td>Input</td>
    <td>Input k in the formula.</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>value</td>
    <td>Input</td>
    <td>Input v in the formula.</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>beta</td>
    <td>Input</td>
    <td>β in the formula.</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>initial_state</td>
    <td>Input</td>
    <td>Initial state matrix, that is, the input S_0 in the formula.</td>
    <td>BFLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>actual_seq_lengths</td>
    <td>Input</td>
    <td>Sequence length of each batch.</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>g</td>
    <td>Optional input</td>
    <td>Decay coefficient, α=e^g in the formula</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>out</td>
    <td>Output</td>
    <td>Attention result at each step, that is, o_t in the formula.</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>final_state</td>
    <td>Output</td>
    <td>Final state matrix, that is, S_L in the formula.</td>
    <td>BFLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>scale_value</td>
    <td>Optional attribute</td>
    <td>Scaling factor of query, that is, scale in the formula. The default value is 1.0</td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
</tbody>
</table>

## Constraints

- To facilitate understanding of the layout formats later (such as BNSD and TND), the meaning of each abbreviated dimension is uniformly described as follows:
  * B: Batch size of the input samples (Batch).
  * T: If $L_i$ is the length of the $i$-th sequence, $T=\sum_i^B L_i$ indicates the cumulative sequence length.
  * Nk: Number of query and key heads.
  * Nv: Number of value heads.
  * Dk: Hidden layer dimension of query and key.
  * Dv: Hidden layer dimension of value.

- Currently only the TND layout is supported:
  - Shapes of query and key: $(T, Nk, Dk)$
  - Shapes of value and out: $(T, Nv, Dv)$
  - Shapes of beta and g: $(T, Nv)$
  - Shape of actual_seq_lengths: $(B,)$
  - Shapes of initial_state and final_state: $(B, Nv, Dv, Dk)$

  The dimensions must satisfy the following constraints:
  - $0 \lt Nv \le 64, 0 \lt Nk \le 64$, and $Nv \bmod Nk = 0$
  - $0 \lt Dv \le 128, 0 \lt Dk \le 128$
  - $B \gt 0, T \gt 0$

- Due to the numerical characteristics of the algorithm, the following value constraints must be satisfied; otherwise, numerical overflow and precision anomalies easily occur:
  - Tensor elements: $-1 \le query[i][j][k] \le 1$
  - Tensor elements: $-1 \le key[i][j][k] \le 1$
  - Tensor elements: $-1 \le g[i][j] \le 0$
  - Tensor elements: $0 < beta[i][j] < 1$

## Calling Instructions

| Calling Method  | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn | [test_aclnn_chunk_gated_delta_rule.cpp](./examples/test_aclnn_chunk_gated_delta_rule.cpp) | Calls the aclnnChunkGatedDeltaRule operator through [aclnnChunkGatedDeltaRule](docs/aclnnChunkGatedDeltaRule_en.md) |
