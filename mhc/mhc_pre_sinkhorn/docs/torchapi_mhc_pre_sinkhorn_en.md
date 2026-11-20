# mhc_pre_sinkhorn

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not Supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not Supported
<!-- end id6 -->

## Function

- **Description**:

  Computes the $\mathbf{H}'_{\text{res}}$ and $\mathbf{H}_{\text{post}}$ projection matrices of the hidden layer in the MHC architecture, as well as the input matrix $\mathbf{h}_{\text{in}}$ of the Attention or MLP layer. Performs Sinkhorn iterative normalization on the $\mathbf{H}'_{\text{res}}$ matrix to obtain the approximately doubly stochastic matrix $\mathbf{H}_{\text{res}}$. Supports outputting intermediate computation results for backward gradient computation. The computation mode used in the Cube is HF32 by default.

  <!-- npu="A3,910b" id13 -->
  <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: The HF32 mode is controlled by the global switch `torch_npu.npu.aclnn.allow_hf32` (default true). The Python wrapper does not expose this parameter separately; it reads it automatically on each call and passes it along with the call. When false semantics are passed in, the mode switches to FP32 computation.
  <!-- end id13 -->

- **Formula**:

  $$
  \begin{aligned}
  \vec{x^{'}_{l}} &= \frac{1}{\sqrt{\frac{1}{d} \sum_{\dim=-2,\text{keepdim}=\text{True}} x_i^2 + \epsilon_{norm}}}\\
  H^{pre}_l &= \alpha^{pre}_{l} \cdot(\vec{x^{'}_{l}}\varphi^{pre}_{l}) + b^{pre}_{l}\\
  H^{post}_l &= \alpha^{post}_{l} \cdot(\vec{x^{'}_{l}}\varphi^{post}_{l}) + b^{post}_{l}\\
  H^{res}_l &= \alpha^{res}_{l} \cdot(\vec{x^{'}_{l}}\varphi^{res}_{l}) + b^{res}_{l}\\
  H^{pre}_l &= \sigma (H^{pre}_{l})\\
  H^{post}_l &= 2\sigma (H^{post}_{l})\\
  h_{in} &=\vec{x_{l}}H^{pre}_l
  \end{aligned}
  $$

  Using $\mathbf{H^{res}_l}$ as input, the Sinkhorn transformation performs a total of $\mathbf{numIters}$ iterations. During the iterative process, intermediate normalization results $\mathbf{normOut}[k]$ and summation results $\mathbf{sumOut}[k]$ are generated. The final output is the $\mathbf{normOut}$ from the last iteration as the transformation result.

  First iteration (initialization):

  $$
  \begin{aligned}
      \mathbf{normOut}[0] &= \text{softmax}(\mathbf{H^{res}_l},  \dim=-1) + \epsilon_{hc}, \\
      \mathbf{sumOut}[1] &= \sum_{\dim=-2,\text{keepdim}=\text{True}} \mathbf{normOut}[0] + \epsilon_{hc}, \\
      \mathbf{normOut}[1] &= \frac{\mathbf{normOut}[0]}{\mathbf{sum\_out}[1]}, \\
  \end{aligned}
  $$

  $i$th iteration ($i = 1, 2, \dots, \mathbf{num\_iters}-1$):

  $$
  \begin{aligned}
      \mathbf{sumOut}[2i] &= \sum_{\dim=-1,\text{keepdim}=\text{True}} \mathbf{normOut}[2i-1] + \epsilon_{hc}, \\
      \mathbf{normOut}[2i] &= \frac{\mathbf{normOut}[2i-1]}{\mathbf{sum\_out}[2i]}, \\
      \mathbf{sumOut}[2i+1] &= \sum_{\dim=-2,\text{keepdim}=\text{True}} \mathbf{normOut}[2i] + \epsilon_{hc}, \\
      \mathbf{normOut}[2i+1] &= \frac{\mathbf{normOut}[2i]}{\mathbf{sum\_out}[2i+1]}, \\
  \end{aligned}
  $$

  Final output:

  $$
  \mathbf{normOut}[2 \times \mathbf{num\_iters} - 1]
  $$

  $$
  \mathbf{sumOut}[2 \times \mathbf{num\_iters} - 1]
  $$

## Prototype

```python
cann_ops_transformer.mhc_pre_sinkhorn(x, phi, alpha, bias, hcMult, numIters, hcEps, normEps, outFlag, *, preMix?: Tensor) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)
```

## Parameters

<table style="undefined;table-layout: fixed; width:1200px"><colgroup>
<col style="width: 120px">
<col style="width: 120px">
<col style="width: 100px">
<col style="width: 300px">
<col style="width: 120px">
<col style="width: 200px">
</colgroup>
<thead>
<tr>
    <th>Parameter Name</th>
    <th>Parameter Type</th>
    <th>Optional/Required</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Shape</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>x</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Input data of the MHC layer. Corresponds to x in the formula.</td>
        <td>float16, bfloat16</td>
        <td>
            <ul>
                <li>(B, S, N, C)</li>
                <li>(T, N, C)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>phi</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Parameter matrix of MHC. Corresponds to <span style="font-family: 'Cambria Math', 'Times New Roman', serif;">φ</span><sup>pre</sup>, <span style="font-family: 'Cambria Math', 'Times New Roman', serif;">φ</span><sup>post</sup>, <span style="font-family: 'Cambria Math', 'Times New Roman', serif;">φ</span><sup>res</sup> in the formula. It is split along the N axis into (N, N*C), (N, N*C), and (N*N, N*C).</td>
        <td>float32</td>
        <td>(N*N+2*N, N*C)</td>
    </tr>
    <tr>
        <td>alpha</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Scaling parameters of MHC. Contains 3 elements corresponding to the scaling coefficients of H<sup>pre</sup>, H<sup>post</sup>, and H<sup>res</sup>. Corresponds to α<sup>pre</sup>, α<sup>post</sup>, α<sup>res</sup> in the formula.</td>
        <td>float32</td>
        <td>(3)</td>
    </tr>
    <tr>
        <td>bias</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Bias parameters of MHC. Contains 2N+N² elements. Corresponds to β<sup>pre</sup>, β<sup>post</sup>, β<sup>res</sup> in the formula, which are N, N, and N*N respectively.</td>
        <td>float32</td>
        <td>(2N+N²)</td>
    </tr>
    <tr>
        <td>hcMult</td>
        <td>int</td>
        <td>Required</td>
        <td>Number of residual streams, size of the HC dimension.</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>numIters</td>
        <td>int</td>
        <td>Required</td>
        <td>Number of Sinkhorn algorithm iterations.</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>hcEps</td>
        <td>float</td>
        <td>Optional</td>
        <td>Epsilon parameter after sigmoid of H<sup>pre</sup>. Corresponds to ε<sub>hc</sub> in the formula. Default value is 1e-6.</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>normEps</td>
        <td>float</td>
        <td>Optional</td>
        <td>Division-by-zero prevention parameter of RmsNorm. Corresponds to ε<sub>norm</sub> in the formula. Default value is 1e-6.</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>preMix</td>
        <td>Tensor</td>
        <td>Optional</td>
        <td><ul><li>premix weighting coefficients (usually filled with the hPre output from the previous folding point). When provided, the hin weighting uses preMix instead of the internally computed hPre; when not provided, the behavior is completely the same as the original interface.</li><li>The shape must match `x.shape[:-1]`.</li></ul></td>
        <td>float32</td>
        <td>2-3</td>
    </tr>
    </tbody>
    </table>

## Output Description

<table style="undefined;table-layout: fixed; width:1200px"><colgroup>
<col style="width: 120px">
<col style="width: 120px">
<col style="width: 100px">
<col style="width: 300px">
<col style="width: 120px">
<col style="width: 200px">
</colgroup>
<thead>
<tr>
    <th>Parameter Name</th>
    <th>Parameter Type</th>
    <th>Optional/Required</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Shape</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>hin</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Output h_in, used as the input of the Attention/MLP layer. Corresponds to h<sub>in</sub> in the formula. Data type is the same as input x.</td>
        <td>float16, bfloat16</td>
        <td>
            <ul>
                <li>(B, S, C)</li>
                <li>(T, C)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>hPost</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Output h_post transformation matrix of MHC. Corresponds to H<sup>post</sup> in the formula.</td>
        <td>float32</td>
        <td>
            <ul>
                <li>(B, S, N)</li>
                <li>(T, N)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>hRes</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Output h_res transformation matrix of MHC (doubly stochastic matrix after Sinkhorn normalization). Corresponds to H<sup>res</sup> in the formula.</td>
        <td>float32</td>
        <td>
            <ul>
                <li>(B, S, N*N)</li>
                <li>(T, N*N)</li>
            </ul>
        </td>
    </tr>
</tbody>
</table>

> Note: A 4D input uses the BSND format, and a 3D input uses the TND format. B indicates the batch size, S indicates the single batch sequence length, T indicates the cumulative sum of all batch sequence lengths, N indicates the number of residual streams, and C indicates the tail-axis hidden dimension size.

## Constraints

- This API supports training and inference scenarios.
- This API supports single operator mode and graph mode.
- Parameter constraints: `phi`, `alpha`, and `bias` must satisfy the shapes in the parameter table. When B, S, or T of `x` is 0, the main outputs are empty and the operator computation is skipped; if `pre_mix` is provided, it must still match `x.shape[:-1]`.
- Data type constraints:
  - The data type of output `hin` must be the same as `x`.
- Specification constraints:
  - numIters: Indicates the number of iterations. Currently only 20 is supported.
  - N: Currently only 4 is supported.
  - Input x supports the 4D BSND format and the 3D TND format, with shapes (B, S, N, C) and (T, N, C) respectively.
  - In TND format, T indicates the cumulative sum of all batch sequence lengths. The corresponding output shape expands along the T axis as described in the parameter description.
  - The tail axis C must be a multiple of 128, with a value range of [1, 100000].
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>:
    - The data range of input phi is limited to within $\pm \frac{1}{\sqrt{NC}}$, and eps is limited to 1e-6. Within this range, good numerical stability is achieved.
  <!-- end id8 -->

## Deterministic Computation/Batch Consistency

- Deterministic computation:
    - Deterministic computation is supported by default.

- Batch consistency:
    <!-- npu="A3,910b" id9 -->
    - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>:
        - Batch consistency is not supported by default.
    <!-- end id9 -->
    <!-- npu="950" id10 -->
    - <term>Ascend 950PR&950DT products</term>:
        - Batch consistency is not supported by default. It can be enabled via the PyTorch switch (see `torch_npu.npu.set_deterministic_level` in [TorchNPU Custom APIs](https://www.hiascend.com/document/detail/en/Pytorch/latest/apiref/customapi/docs/en/custom_APIs/overview.md)). Performance may degrade to some extent after enabling batch consistency.
    <!-- end id10 -->

## Example

- Single operator mode call:

  ```python
  import torch
  import torch_npu
  from cann_ops_transformer.ops import mhc_pre_sinkhorn

  B = 1
  S = 1024
  N = 4
  C = 4096

  x = torch.randn(B, S, N, C, dtype=torch.bfloat16).npu()
  phi = torch.randn(N * N + 2 * N, N * C, dtype=torch.float32).npu()
  alpha = torch.randn(3, dtype=torch.float32).npu()
  bias = torch.randn(N * N + 2 * N, dtype=torch.float32).npu()

  hcMult = 4
  numIters = 20
  hcEps = 1e-6
  normEps = 1e-6

  hin, hPost, hRes = mhc_pre_sinkhorn(
    x, phi, alpha, bias, hcMult, numIters, hcEps, normEps
  )
  ```

- Graph mode call:

  ```python
  import torch
  import torch_npu
  import torchair
  from cann_ops_transformer.ops import mhc_pre_sinkhorn

  torch_npu.npu.set_device(0)

  B = 1
  S = 128
  N = 4
  C = 4096

  class MhcPreSinkhornModel(torch.nn.Module):
      def forward(self, x, phi, alpha, bias):
          hin, hPost, hRes = mhc_pre_sinkhorn(
              x, phi, alpha, bias,
              hc_mult=4,
              num_iters=20,
              hc_eps=1e-6,
              norm_eps=1e-6
          )
          return hin, hPost, hRes

  model = MhcPreSinkhornModel().npu()
  npu_backend = torchair.get_npu_backend()
  model = torch.compile(model, backend=npu_backend, dynamic=False)

  x = torch.randn(B, S, N, C, dtype=torch.bfloat16, device="npu")
  phi = torch.randn(N * N + 2 * N, N * C, dtype=torch.float32, device="npu")
  alpha = torch.randn(3, dtype=torch.float32, device="npu")
  bias = torch.randn(N * N + 2 * N, dtype=torch.float32, device="npu")

  hin, hPost, hRes = model(x, phi, alpha, bias)
  ```
