# BlockAttentionResidualsGrad

English | [简体中文](./README.md)

## Product Support Status

| Product                                                      | Supported |
| :----------------------------------------------------------- | :-------: |
| <term>Ascend 950PR&950DT products</term>                     |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>            |    ×     |
| <term>Atlas inference products</term>                        |    ×     |
| <term>Atlas training products</term>                         |    ×     |

## Function Description

- **Operator function**: `BlockAttentionResidualsGrad` is the backward propagation operator of the forward operator `BlockAttentionResiduals` (attention residual). It fuses the backward computation of Softmax, RMS normalization, and attention output.
- **Main outputs**: `gradPartialBlock`, `gradBlockRes`, `gradProjWeight`, and `gradNormWeight`.
- **Forward cache dependency**: `invNorm` and `probs` saved by the forward operator. The current version directly uses the saved results for computation and does not rely on `validBlockNum` to reconstruct the mask.

- **Computation formula**:

The concatenated value and normalized intermediate quantities in the backward computation are defined as follows:

$$
V_{t,i,h} =
\begin{cases}
block\_res_{t,i,h}, & i < N \\
partial\_block_{t,h}, & i = N
\end{cases}
$$

$$
k_{t,i,h} = V_{t,i,h} \cdot inv\_norm_{t,i}
$$

$$
score\_weight_{h} = norm\_weight_{h} \cdot proj\_weight_{0,h}
$$

Backward gradients of `out` with respect to `V` and `probs`:

$$
g_{t,i} = \sum_{h=0}^{H-1} grad\_output_{t,h} \cdot V_{t,i,h}
$$

$$
grad\_score_{t,i} = probs_{t,i} \cdot \left(g_{t,i} - \sum_{j=0}^{N} probs_{t,j} \cdot g_{t,j}\right)
$$

Gradients propagated back through `score_weight` and RMS normalization:

$$
grad\_k_{t,i,h} = grad\_score_{t,i} \cdot score\_weight_{h}
$$

$$
grad\_score\_weight_{h} = \sum_{t=0}^{T-1}\sum_{i=0}^{N} grad\_score_{t,i} \cdot k_{t,i,h}
$$

$$
grad\_inv\_norm_{t,i} = \sum_{h=0}^{H-1} grad\_k_{t,i,h} \cdot V_{t,i,h}
$$

Total gradient of `V` and final output gradients:

$$
grad\_V_{t,i,h} = grad\_output_{t,h} \cdot probs_{t,i} + grad\_k_{t,i,h} \cdot inv\_norm_{t,i}
                - \frac{grad\_inv\_norm_{t,i} \cdot inv\_norm_{t,i}^{3}}{H} \cdot V_{t,i,h}
$$

$$
grad\_block\_res_{t,i,h} = grad\_V_{t,i,h}, \quad i < N
$$

$$
grad\_partial\_block_{t,h} = grad\_V_{t,N,h}
$$

$$
grad\_norm\_weight_{h} = grad\_score\_weight_{h} \cdot proj\_weight_{0,h}
$$

$$
grad\_proj\_weight_{0,h} = grad\_score\_weight_{h} \cdot norm\_weight_{h}
$$

Here, $T$ is the number of tokens, $N$ is the number of blocks in `blockRes`, and $H$ is the hidden size.

## Parameter Description

<table style="table-layout: fixed; width: 1200px"><colgroup>
  <col style="width: 220px">
  <col style="width: 120px">
  <col style="width: 520px">
  <col style="width: 220px">
  <col style="width: 120px">
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
      <td>partialBlock</td>
      <td>Input</td>
      <td>Forward input prefix sum, used as the $(N+1)$-th value after concatenation, with shape $[T,H]$.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>blockRes</td>
      <td>Input</td>
      <td>Forward input block residual, used as the first $N$ values after concatenation, with shape $[T,N,H]$.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>projWeight</td>
      <td>Input</td>
      <td>Forward projection weight, which forms score_weight together with normWeight, with shape $[1,H]$.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normWeight</td>
      <td>Input</td>
      <td>Forward normalization weight, which forms score_weight together with projWeight, with shape $[H]$.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gradHiddenStates</td>
      <td>Input</td>
      <td>Upstream gradient of the forward output out, with shape $[T,H]$.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>invNorm</td>
      <td>Input</td>
      <td>Row-wise normalization coefficient saved by the forward operator, with shape $[T,N+1]$.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>probs</td>
      <td>Input</td>
      <td>Forward Softmax output probability, with shape $[T,N+1]$.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>validBlockNum</td>
      <td>Attribute</td>
      <td>Reserved attribute. Default value: -1. It does not participate in computation currently. Only -1 or N can be passed in.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradPartialBlock</td>
      <td>Output</td>
      <td>Gradient of partialBlock, with the same shape as partialBlock.</td>
      <td>Same as the main inputs</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gradBlockRes</td>
      <td>Output</td>
      <td>Gradient of blockRes, with the same shape as blockRes.</td>
      <td>Same as the main inputs</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gradProjWeight</td>
      <td>Output</td>
      <td>Gradient of projWeight, with the same shape as projWeight.</td>
      <td>Same as the main inputs</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gradNormWeight</td>
      <td>Output</td>
      <td>Gradient of normWeight, with the same shape as normWeight.</td>
      <td>Same as the main inputs</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## Constraints

- $T \ge 64$, $0 \le N \le 128$, and $2048 \le H \le 500000$. $T$ and $H$ in all tensors and the second dimension $N+1$ of `invNorm/probs` must be consistent.
- The main inputs `partialBlock/blockRes/projWeight/normWeight/gradHiddenStates` support FLOAT16, BFLOAT16, and FLOAT32, and their dtypes must be consistent. Only FLOAT32 is supported for `invNorm/probs`.
- Non-contiguous tensors are supported for inputs, which are converted to Contiguous internally before computation.
- The output dtypes are consistent with the corresponding main inputs.
- `validBlockNum` is a reserved attribute, and different values do not affect the computation result of the current version.
- `aclnnBlockAttentionResidualsGrad` uses a deterministic implementation by default.

## Invocation Description

<table><thead>
  <tr>
    <th>Invocation Method</th>
    <th>Invocation Sample</th>
    <th>Description</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>aclnn invocation</td>
    <td><a href="./examples/test_aclnn_block_attention_residuals_grad.cpp">test_aclnn_block_attention_residuals_grad.cpp</a></td>
    <td>Invokes the operator through <a href="./docs/aclnnBlockAttentionResidualsGrad_en.md">aclnnBlockAttentionResidualsGrad_en.md</a>.</td>
  </tr>
  <tr>
    <td>SPLIT_H example</td>
    <td><a href="./examples/test_aclnn_block_attention_residuals_grad_split_h.cpp">test_aclnn_block_attention_residuals_grad_split_h.cpp</a></td>
    <td>Demonstrates how to invoke the SPLIT_H template in large-H scenarios.</td>
  </tr>
</tbody>
</table>

```bash
# Run in the root directory of the ops-transformer repository
bash build.sh --pkg --soc=ascend950 --ops=block_attention_residuals_grad
bash build.sh --run_example block_attention_residuals_grad eager cust --soc=ascend950 --vendor_name=custom
```
