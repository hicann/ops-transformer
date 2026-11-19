# BlockAttnResPrepare

English | [简体中文](./README.md)

## Product Support

| Product | Supported |
| ---- | :------: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function Description

- Computes the first stage of Attention Residuals historical residual attention. It produces the online Softmax state $(O,M,L)$ for all $S$ target slots, which the update stage combines with the current residual.

- Computation

  Let $V$ be the FLOAT32 representation of `block_res`, $Q$ be `pseudo_query`, $N_v=\min(\texttt{valid\_blocks[0]},N)$ be the number of valid historical blocks, and $\epsilon=\operatorname{FP32}(\texttt{eps})$.

  For $N_v>0$, compute root mean square normalization (RMSNorm) along the hidden dimension and the attention logits.

  $$
  \widehat V_{t,n}=\operatorname{RMSNorm}(V_{t,n},\mathbf{1},\epsilon),\qquad
  Z_{s,t,n}=\left\langle Q_s,\widehat V_{t,n}\right\rangle
  $$

  Compute the online Softmax state over the valid historical blocks.

  $$
  M_{s,t}=\max_{0\le n<N_v}Z_{s,t,n},\qquad
  E_{s,t,n}=\exp(Z_{s,t,n}-M_{s,t})
  $$

  $$
  L_{s,t}=\sum_{n=0}^{N_v-1}E_{s,t,n},\qquad
  O_{s,t,d}=\sum_{n=0}^{N_v-1}E_{s,t,n}V_{t,n,d}
  $$

  - $T$, $N$, $S$, and $D$ are the token count, historical block capacity, target slot count, and hidden dimension. The corresponding indices are $t\in[0,T)$, $n\in[0,N_v)$, $s\in[0,S)$, and $d\in[0,D)$.
  - RMSNorm uses an all-ones scale vector. The scoring scale can be multiplied elementwise into the original query in advance to form $Q$.
  - $O$, $M$, and $L$ correspond to `numerator`, `logit_max`, and `exp_sum`. The weighted numerator $O$ has not been divided by $L$.

  For $N_v=0$, return the empty online Softmax state.

  $$
  O=\mathbf{0}_{S\times T\times D},\qquad
  M=m_{\min}\mathbf{1}_{S\times T},\qquad
  L=\mathbf{0}_{S\times T},\qquad
  m_{\min}=-3.4028234663852886\times10^{38}
  $$

  Here, $m_{\min}$ is the minimum finite FLOAT32 value.

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
      <td>block_res</td>
      <td>Input</td>
      <td>Historical residual blocks, corresponding to V in the formula, with the shape [T, N, D].</td>
      <td>BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>valid_blocks</td>
      <td>Input</td>
      <td>Number of valid historical residual blocks, with shape [1]. Zero produces an empty history, and values above N are clipped to N.</td>
      <td>UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pseudo_query</td>
      <td>Input</td>
      <td>Pseudo-query matrix Q with shape [S, D], with one row per target slot.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>numerator</td>
      <td>Output</td>
      <td>Softmax weighted numerator, corresponding to O in the formula, with the shape [S, T, D].</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>logit_max</td>
      <td>Output</td>
      <td>Maximum logit of valid historical sources, corresponding to M in the formula, with the shape [S, T].</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>exp_sum</td>
      <td>Output</td>
      <td>Exponential sum after max-shift, corresponding to L in the formula, with the shape [S, T].</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>Optional attribute</td>
      <td>Stability term for RMS normalization. The default is <code>1e-6</code>. The value must be in the positive normal FLOAT32 range <code>[FLT_MIN, FLT_MAX]</code>.</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## Constraints

- `T >= 0`, `S >= 0`, `1 <= N <= 64`, and `1 <= D <= 8192`.
- For non-empty computation (`T > 0` and `S > 0`), `T` and `S` must not exceed `4294967295` (`UINT32_MAX`), and `T * S` must not exceed `9223372036854775807` (`INT64_MAX`).
- All input and output tensors must be contiguous.
- `eps` must be finite and in the positive normal FLOAT32 range `[FLT_MIN, FLT_MAX]` (approximately `[1.1754943508222875e-38, 3.4028234663852886e+38]`). The double value received by ACLNN is validated against this range before conversion to a FLOAT32 attribute. Subnormal values and values that narrow to zero or Inf are not supported.
- If `T == 0` or `S == 0`, empty outputs of the corresponding shapes are returned after parameter validation and shape inference, without launching AICore computation. The limits for non-empty computation above do not apply to this path; the valid `eps` range and other API checks still apply.

## Invocation

| Invocation Method | Sample | Description |
| --- | --- | --- |
| aclnn API | [test_aclnn_block_attn_res_prepare.cpp](examples/arch35/test_aclnn_block_attn_res_prepare.cpp) | Calls [aclnnBlockAttnResPrepare](docs/aclnnBlockAttnResPrepare_en.md). See [Compile and Run Samples](../../docs/en/context/compile_and_run_sample.md) for build instructions. |
| PyTorch API | [Example](docs/torchapi_block_attn_res_prepare_en.md#examples) | Calls [block_attn_res_prepare](docs/torchapi_block_attn_res_prepare_en.md). |

```bash
# Execute in the root directory of the ops-transformer repository
bash build.sh --pkg --soc=ascend950 --ops=block_attn_res_prepare
(cd build_out && ./cann-ops-transformer-*.run)
bash build.sh --run_example block_attn_res_prepare eager cust --soc=ascend950 --vendor_name=custom
```
