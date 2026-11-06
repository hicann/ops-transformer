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

- Operator function: Completes the first stage of the two-stage computation of Attention Residuals historical residual attention. It computes the inter-block attention of all $S$ target slots in parallel at one time and returns the softmax weighted numerator and the statistics $(O,M,L)$ for the next stage to fuse the current residual.

- Computation formula:

    $$
    \operatorname{block\_attn\_res\_prepare}
    \left(V,Q,\operatorname{valid\_blocks};\epsilon\right)
    \longrightarrow (O,M,L)
    $$

    Computes the RMS normalization factor for the $n$-th valid historical source of token $t$:

    $$
    R_{t,n}=\sqrt{\frac{1}{D}\sum_{d=0}^{D-1}V_{t,n,d}^{2}+\epsilon}
    $$

    Computes the logit for each target slot, token, and valid historical source:

    $$
    Z_{s,t,n}=R_{t,n}^{-1}\sum_{d=0}^{D-1}Q_{s,d}V_{t,n,d}
    $$

    Let $N_v$ be the number of valid historical sources specified by `valid_blocks`. Then:

    $$
    M_{s,t}=\max_{0\le n<N_v}Z_{s,t,n}
    $$

    $$
    E_{s,t,n}=\exp\left(Z_{s,t,n}-M_{s,t}\right)
    $$

    $$
    L_{s,t}=\sum_{n=0}^{N_v-1}E_{s,t,n}
    $$

    $$
    O_{s,t,d}=\sum_{n=0}^{N_v-1}E_{s,t,n}V_{t,n,d}
    $$

    When `valid_blocks[0] == 0`, the preceding softmax computation is not performed, and the following values are returned directly:

    $$
    O=\mathbf{0}_{S\times T\times D},\qquad
    M=m_{\min}\mathbf{1}_{S\times T},\qquad
    L=\mathbf{0}_{S\times T},\qquad
    m_{\min}=-3.4028234663852886\times10^{38}
    $$

    That is, `numerator` is an all-0 tensor with the shape `[S, T, D]`, `logit_max` is a tensor with the shape `[S, T]` whose elements are all the minimum finite FLOAT32 value $m_{\min}$, and `exp_sum` is an all-0 tensor with the shape `[S, T]`. This result indicates the empty state of online softmax.

    where:

    - $V\in\mathbb{R}^{T\times N\times D}$ denotes the input `block_res`; $V_{t,n,d}$ denotes the value of the $d$-th hidden feature of the $n$-th historical residual block of the $t$-th token.
    - $Q\in\mathbb{R}^{S\times D}$ denotes the input `pseudo_query`; each target slot corresponds to a query vector, and $Q_{s,d}$ denotes the value of the $d$-th hidden feature of the $s$-th target slot.
    - $T$ denotes the number of tokens, $N$ denotes the number of historical residual blocks that dimension 1 of `block_res` can hold, $S$ denotes the number of target slots, and $D$ denotes HiddenSize.
    - $N_v=\min(\texttt{valid\_blocks[0]},N)$ denotes the number of historical residual blocks that actually participate in the computation.
    - $R$ denotes the RMS normalization factor, $Z$ denotes the normalized attention logit, and $E$ denotes the exponential value after max-shift.
    - $O$, $M$, and $L$ denote the softmax weighted numerator, the maximum logit, and the exponential sum, corresponding to the outputs `numerator`, `logit_max`, and `exp_sum`, respectively.
    - $\epsilon$ denotes the numerical stability term of RMS normalization, corresponding to the attribute `eps`.

## Parameter Description

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 120px">
  <col style="width: 150px">
  <col style="width: 490px">
  <col style="width: 120px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>block_res</td>
      <td>Input</td>
      <td>Historical residual blocks, corresponding to V in the formula, with the shape [T, N, D].</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>valid_blocks</td>
      <td>Input</td>
      <td>Number of currently valid historical sources, with the shape [1]. When the value is 0, numerator returns an all-0 tensor, logit_max returns a tensor whose elements are all the minimum finite FLOAT32 value, and exp_sum returns an all-0 tensor. When the value is greater than N, it is processed as N.</td>
      <td>UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pseudo_query</td>
      <td>Input</td>
      <td>Pseudo query, corresponding to Q in the formula. Each target slot corresponds to a query vector, with the shape [S, D].</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>numerator</td>
      <td>Output</td>
      <td>Softmax weighted numerator, corresponding to O in the formula, with the shape [S, T, D].</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>logit_max</td>
      <td>Output</td>
      <td>Maximum logit of valid historical sources, corresponding to M in the formula, with the shape [S, T].</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>exp_sum</td>
      <td>Output</td>
      <td>Exponential sum after max-shift, corresponding to L in the formula, with the shape [S, T].</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>Attribute</td>
      <td>Numerical stability term of RMS normalization. Optional. The default value is 1e-6, and it must be a finite positive number.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## Constraint Description

- `T >= 0`, `S >= 0`, `1 <= N <= 64`, and `1 <= D <= 8192`.
- The shape of `block_res` is `[T, N, D]`, and the shape of `pseudo_query` is `[S, D]`. The shapes of `numerator`, `logit_max`, and `exp_sum` are `[S, T, D]`, `[S, T]`, and `[S, T]`, respectively.
- The shape of `valid_blocks` must be `[1]`. When `valid_blocks[0] == 0`, `numerator` returns an all-0 tensor, `logit_max` returns a tensor whose elements are all the minimum finite FLOAT32 value, and `exp_sum` returns an all-0 tensor. When `valid_blocks[0] > N`, it is processed as `N`.
- All input and output tensors must be contiguous tensors. Non-contiguous tensors are not supported.
- `eps` must be a finite positive number.
- When `T == 0` or `S == 0`, empty outputs of the corresponding shapes are returned.

## Invocation Description

<table><thead>
  <tr>
    <th>Invocation method</th>
    <th>Invocation sample</th>
    <th>Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td>aclnn invocation</td>
    <td><a href="./examples/arch35/test_aclnn_block_attn_res_prepare.cpp">test_aclnn_block_attn_res_prepare</a></td>
    <td>Invokes the operator through <a href="./docs/aclnnBlockAttnResPrepare_en.md">aclnnBlockAttnResPrepare_en.md</a>, and compiles the operator block_attn_res_prepare. For the detailed operator compilation and running methods, see <a href="../../docs/zh/invocation/quick_op_invocation.md">Operator Invocation</a>.</td>
  </tr>
  <tr>
    <td>PyTorch API</td>
    <td>-</td>
    <td>Invokes the block_attn_res_prepare operator through the <a href="../../attention/block_attn_res_prepare/docs/torchapi_block_attn_res_prepare_en.md">block_attn_res_prepare</a> interface.</td>
  </tr>
</tbody>
</table>

```bash
# Execute in the root directory of the ops-transformer repository
bash build.sh --pkg --soc=ascend950 --ops=block_attn_res_prepare
(cd build_out && ./cann-ops-transformer-*.run)
bash build.sh --run_example block_attn_res_prepare eager cust --soc=ascend950 --vendor_name=custom
```
