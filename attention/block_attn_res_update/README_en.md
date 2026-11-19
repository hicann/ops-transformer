# BlockAttnResUpdate

English | [简体中文](./README.md)

## Product Support

| Product | Supported |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function Description

- Accumulates `delta` into `partial_block` in place, computes the normalized score of the updated residual, and combines it with the historical online Softmax state to produce `h`. Optional output RMSNorm is applied to the BFLOAT16 mixed result.

- [aclnnBlockAttnResUpdateV2](docs/aclnnBlockAttnResUpdateV2_en.md) extends [aclnnBlockAttnResUpdate](docs/aclnnBlockAttnResUpdate_en.md) with optional output RMSNorm, using `output_norm_weight` and the independent `output_norm_eps`.

- Computation

  For each token $t$, update the partial residual and write it back in place.

  $$
  p_t=P_t+\operatorname{FP32}(\Delta_t),\qquad
  P^{\mathrm{out}}_t=p_t
  $$

  Compute the current score using root mean square normalization (RMSNorm) and a query dot product.

  $$
  C_t=\left\langle Q,\operatorname{RMSNorm}(p_t,\mathbf{1},\epsilon)\right\rangle
  $$

  Merge the current residual with the historical online Softmax state.

  $$
  m_t=\max(M_t,C_t),\qquad
  \alpha_t=\exp(M_t-m_t),\qquad
  \beta_t=\exp(C_t-m_t)
  $$

  $$
  H^{(0)}_t=\operatorname{BF16}\left(
  \frac{\alpha_t O_t+\beta_t p_t}{\alpha_t L_t+\beta_t}
  \right)
  $$

  If `output_norm_weight` is omitted, return $H=H^{(0)}$. Otherwise, apply output RMSNorm to the BFLOAT16 mixed result.

  $$
  H_t=\operatorname{BF16}\left(
  \operatorname{RMSNorm}\left(\operatorname{FP32}(H^{(0)}_t),\operatorname{FP32}(W),\epsilon_{\mathrm{out}}\right)
  \right)
  $$

  Here, $W$ is `output_norm_weight` and $\epsilon_{\mathrm{out}}=\operatorname{FP32}(\texttt{output\_norm\_eps})$.

  - $T$ and $D$ are the token count and hidden dimension, with $t\in[0,T)$. The residuals, query, numerator, and output are $D$-dimensional vectors.
  - $P$ and $P^{\mathrm{out}}$ are the values of `partial_block` before and after the call. $\Delta$ is `delta`, $Q$ is `pseudo_query`, and $p$ is the FLOAT32 residual sum.
  - $O$, $M$, and $L$ are `numerator`, `logit_max`, and `exp_sum`. $H$ is `h`.
  - $\epsilon=\operatorname{FP32}(\texttt{eps})$. RMSNorm operates along $D$ and uses an all-ones scale vector for scoring. The scoring scale can be multiplied elementwise into the original query in advance to form $Q$.

  For a non-empty history $\mathcal{H}_t$, let $z_{t,i}$ be a historical logit and $v_{t,i}$ its value vector. The input state satisfies the following relations.

  $$
  \begin{aligned}
  M_t&=\max_{i\in\mathcal{H}_t}z_{t,i} \\
  L_t&=\sum_{i\in\mathcal{H}_t}\exp(z_{t,i}-M_t)>0 \\
  O_t&=\sum_{i\in\mathcal{H}_t}\exp(z_{t,i}-M_t)v_{t,i}
  \end{aligned}
  $$

## Parameters

<table style="table-layout: fixed; width: 1000px">
  <colgroup>
    <col style="width: 160px">
    <col style="width: 150px">
    <col style="width: 450px">
    <col style="width: 140px">
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
      <td>partial_block</td>
      <td>Input</td>
      <td>Partial residual P, with shape <code>[T, D]</code>. Shares storage with the output of the same name.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>delta</td>
      <td>Input</td>
      <td>Residual increment Δ, with shape <code>[T, D]</code>.</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pseudo_query</td>
      <td>Input</td>
      <td>Query vector Q used to score the updated residual, with shape <code>[D]</code>.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>numerator</td>
      <td>Input</td>
      <td>Weighted numerator O of the historical online Softmax state, with shape <code>[T, D]</code>.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>logit_max</td>
      <td>Input</td>
      <td>Maximum historical logit M for each token, with shape <code>[T]</code>.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>exp_sum</td>
      <td>Input</td>
      <td>Historical exponential sum L for each token, with shape <code>[T]</code>.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_norm_weight</td>
      <td>Optional input</td>
      <td>Output RMSNorm scale W, with shape <code>[D]</code>. Omitting it disables output RMSNorm.</td>
      <td>BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>partial_block</td>
      <td>Output</td>
      <td>Updated FLOAT32 residual P<sup>out</sup>, written in place with shape <code>[T, D]</code>.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>h</td>
      <td>Output</td>
      <td>Attention output H, with shape <code>[T, D]</code>. The operator applies output RMSNorm when a weight is supplied.</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>Optional attribute</td>
      <td>Stability term ε for residual scoring. The value must be finite and positive. The default is <code>1e-6</code>.</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output_norm_eps</td>
      <td>Optional attribute</td>
      <td>Independent stability term for output RMSNorm. The default is <code>1e-6</code>. When enabled, its value before FLOAT32 conversion must be in <code>[FLT_MIN, FLT_MAX]</code>. It is ignored when output RMSNorm is disabled.</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## Constraints

- When output RMSNorm is enabled, `output_norm_eps` must be in `[FLT_MIN, FLT_MAX]`, where `FLT_MIN = 2^-126 ≈ 1.1754943508222875e-38` and `FLT_MAX ≈ 3.4028234663852886e38`. Subnormal epsilon values are unsupported because the default Sqrt FTZ mode flushes them to zero.
- For non-empty workloads, `T` and `D` must be known positive integers, with `1 <= D <= 8192`. Empty tensor behavior is described in [aclnnBlockAttnResUpdate](docs/aclnnBlockAttnResUpdate_en.md#constraints), [aclnnBlockAttnResUpdateV2](docs/aclnnBlockAttnResUpdateV2_en.md#constraints), and the [PyTorch API](docs/torchapi_block_attn_res_update_en.md#constraints).
- `partial_block`, `delta`, `numerator`, the output `partial_block`, and `h` have shape `[T, D]`. `pseudo_query` and a supplied `output_norm_weight` have shape `[D]`. `logit_max` and `exp_sum` have shape `[T]`.
- For each token in a non-empty workload, the historical state must represent a non-empty history. `numerator` and `logit_max` must contain finite values, and `exp_sum` must contain finite, strictly positive values. All three tensors must describe the same online Softmax state. Host validation does not read these element values.
- `T * D` must fit in `int64_t`, and the token count assigned to a core must fit in `uint32_t`.
- The full hidden dimension, optional normalization weight, and double buffers must fit within the Unified Buffer (UB) budget.
- Output RMSNorm acts on the BFLOAT16 attention mixture. `partial_block` stores the FLOAT32 residual sum in place. Input zeros, extreme values, NaN, and Inf receive no additional handling.

## Invocation

| Invocation Method | Sample | Description |
| --- | --- | --- |
| aclnn API | [test_aclnn_block_attn_res_update.cpp](examples/arch35/test_aclnn_block_attn_res_update.cpp) | Uses [aclnnBlockAttnResUpdate](docs/aclnnBlockAttnResUpdate_en.md). |
| aclnn API (V2, recommended) | [test_aclnn_block_attn_res_update_v2.cpp](examples/arch35/test_aclnn_block_attn_res_update_v2.cpp) | Uses [aclnnBlockAttnResUpdateV2](docs/aclnnBlockAttnResUpdateV2_en.md), with optional output RMSNorm. |
| PyTorch API | [Example](docs/torchapi_block_attn_res_update_en.md#examples) | Uses [cann_ops_transformer.block_attn_res_update](docs/torchapi_block_attn_res_update_en.md). |

Run the following command from the repository root. See [Compile and Run Samples](../../docs/en/context/compile_and_run_sample.md) for details.

```bash
bash build.sh --soc=ascend950 --ops=block_attn_res_update
```
