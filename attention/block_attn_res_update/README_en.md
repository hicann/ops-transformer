# BlockAttnResUpdate

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| ---- | :----:|
|<term>Ascend 950PR&950DT products</term>|√|
|<term>Atlas A3 products</term>|×|
|<term>Atlas A2 products</term>|×|
|<term>Atlas 200I/500 A2 inference products</term>|×|
|<term>Atlas inference products</term>|×|
|<term>Atlas training products</term>|×|

## Description

- Operator function: Performs the phase 2 update of BlockAttnRes. The operator first updates the local attention result partial_block with the incremental delta, then computes the normalized correlation score between the updated result and pseudo_query, and generates the attention output h by combining the existing online Softmax state numerator, logit_max, and exp_sum.

- Computation formula:

For each token $t$, first update the local attention result:

$$
p_t = partial\_block_t + \operatorname{float}(delta_t)
$$

Compute the RMS and the normalized correlation score of the updated result:

$$
rms_t = \sqrt{\frac{1}{D}\sum_{j=0}^{D-1}p_{t,j}^{2} + eps}
$$

$$
score_t = \frac{\sum_{j=0}^{D-1}p_{t,j} \cdot pseudo\_query_j}{rms_t}
$$

Fuse the existing state and the current result in the online Softmax manner:

$$
max_t = \max(logit\_max_t, score_t)
$$

$$
alpha_t = \exp(logit\_max_t-max_t), \qquad
beta_t = \exp(score_t-max_t)
$$

$$
denominator_t = exp\_sum_t \cdot alpha_t + beta_t
$$

The final outputs are:

$$
partial\_block\_out_t = p_t
$$

$$
h_t = \operatorname{bfloat16}\left(
numerator_t \cdot \frac{alpha_t}{denominator_t} +
p_t \cdot \frac{beta_t}{denominator_t}\right)
$$

Where $T$ denotes the number of tokens, $D$ denotes the hidden dimension, $t \in [0,T)$, and $j \in [0,D)$.

For the non-empty history set $\mathcal{H}_t$ of token $t$, let the historical logits be $s_{t,i}$ and the corresponding $D$-dimensional values be
$v_{t,i}$. The input online Softmax state must satisfy:

$$
logit\_max_t = \max_{i \in \mathcal{H}_t}s_{t,i}
$$

$$
exp\_sum_t = \sum_{i \in \mathcal{H}_t}\exp(s_{t,i}-logit\_max_t) > 0
$$

$$
numerator_t = \sum_{i \in \mathcal{H}_t}\exp(s_{t,i}-logit\_max_t)v_{t,i}
$$

This operator only reads the historical state and fuses the current $p_t$ with it. It does not update numerator, logit_max, or exp_sum.

## Parameter Description

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>partial_block</td>
      <td>Input</td>
      <td>Local attention result to be updated, corresponding to partial_block in the formula, with shape $(T, D)$; reuses storage with the output of the same name.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>delta</td>
      <td>Input</td>
      <td>Increment of the local attention result, corresponding to delta in the formula, with shape $(T, D)$.</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pseudo_query</td>
      <td>Input</td>
      <td>Pseudo Query vector used to compute the normalized correlation score, corresponding to pseudo_query in the formula, with shape $(D)$.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>numerator</td>
      <td>Input</td>
      <td>Numerator corresponding to the non-empty historical online Softmax state, corresponding to numerator in the formula, with shape $(T, D)$; read-only for this operator.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>logit_max</td>
      <td>Input</td>
      <td>Finite maximum value of each token in the non-empty historical online Softmax state, corresponding to logit_max in the formula, with shape $(T)$; read-only for this operator.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>exp_sum</td>
      <td>Input</td>
      <td>Finite positive exponent sum of each token in the non-empty historical online Softmax state, corresponding to exp_sum in the formula, with shape $(T)$; read-only for this operator.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>partial_block</td>
      <td>Output</td>
      <td>Updated local attention result, corresponding to partial_block_out in the formula, with shape $(T, D)$; forms a reference relationship with the input of the same name.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>h</td>
      <td>Output</td>
      <td>Attention result obtained by fusing the updated local result with the existing online Softmax state, corresponding to h in the formula, with shape $(T, D)$.</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>Optional attribute</td>
      <td>Stability term in the RMS computation. The value must be finite and greater than 0. The default value is $10^{-6}$.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- $T$ and $D$ must be known positive integers, and $D$ must be in the range $[1, 8192]$.
- The shapes of partial_block, delta, numerator, the output partial_block, and h must be the same, all $(T, D)$; the shapes of pseudo_query, logit_max, and exp_sum must be $(D)$, $(T)$, and $(T)$, respectively.
- The original format and storage format of all inputs must be ND; the StorageShape of all inputs and outputs must be exactly the same as the OriginShape.
- numerator, logit_max, and exp_sum are runtime read-only states in GM (Global Memory); Host Tiling does not read or verify their element values.
  You must ensure that the historical state of each token is non-empty: the elements in logit_max and numerator are finite values, exp_sum is a finite value strictly greater than 0,
  and the three tensors come from the same historical state. The empty historical state represented by `logit_max=-inf`, `exp_sum=0`, and a zero numerator is currently not supported.
- $T \times D$ cannot exceed the representation range of the signed 64-bit Kernel GM element offset; the value of $T$ after core partitioning cannot exceed the representation range of uint32_t.
- A complete row of $D$-dimensional data must fit into the Unified Buffer (UB) in the double buffering mode; otherwise, Tiling fails.
- partial_block is an in-place update parameter. During framework invocation, the reference relationship ensures that the input and output reuse storage; the Kernel directly reads and writes through the partial_block input address and does not access the corresponding output ABI address.
- The workspace size requested by Host Tiling is currently 0, and the Kernel does not access the workspace address.
- The RMS computation uses FP32 sqrt and division. It does not perform Newton iteration and does not perform extra processing on zero values, extreme values, NaN, or Inf in the input Tensor.

## Calling Description

<table><thead>
  <tr>
    <th>Invocation Method</th>
    <th>Invocation Sample</th>
    <th>Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td>aclnn invocation</td>
    <td><a href="examples/arch35/test_aclnn_block_attn_res_update.cpp">test_aclnn_block_attn_res_update.cpp</a></td>
    <td>Invoked through the aclnnBlockAttnResUpdate API. For details, see the <a href="docs/aclnnBlockAttnResUpdate_en.md">API documentation</a>.</td>
  </tr>
  <tr>
    <td>PyTorch API</td>
    <td>-</td>
    <td>Invoked through the <a href="../../attention/block_attn_res_update/docs/torchapi_block_attn_res_update_en.md">cann_ops_transformer.block_attn_res_update</a> API.</td>
  </tr>
</tbody>
</table>

```bash
## Quick Start
# Run in the repository root directory, assuming the environment variables are ready
bash build.sh --soc=${soc_version} --ops=block_attn_res_update
```
