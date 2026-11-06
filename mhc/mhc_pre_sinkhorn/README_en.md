# MhcPreSinkhorn

English | [简体中文](./README.md)

## Product Support

| Product                                                     | Supported |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                   |    √    |
| <term>Atlas A3 products</term> |    √    |
| <term>Atlas A2 products</term> |    √    |
| <term>Atlas 200I/500 A2 inference products</term>                  |    ×    |
| <term>Atlas inference products</term>                          |    ×    |
| <term>Atlas training products</term>                          |    ×    |

## Function Description

- **Operator function**: The operator obtains the $\mathbf{H}'_{\text{res}}$ and $\mathbf{H}_{\text{post}}$ projection matrices of the hidden layer in the MHC architecture and the input matrix $\mathbf{h}_{\text{in}}$ of the Attention or MLP layer through a series of computations. It performs the Sinkhorn iterative normalization transformation on the $\mathbf{H}'_{\text{res}}$ matrix to obtain the doubly stochastic matrix $\mathbf{H}_{\text{res}}$. The operator can output intermediate computation results for backward gradient computation, including the $\mathbf{H^{pre}_l}$ matrix after the sigmoid computation, the result of the matrix multiplication between $\vec{x^{'}_{l}}$ and $\mathbf{\varphi}$, the RmsNorm result of input x $\mathbf{\vec{x^{'}_{l}}}$, the intermediate normalization result $\mathbf{normOut}$ and the summation result $\mathbf{sumOut}$ generated during the iteration. The operator uses the HF32 computation mode in the Cube.

- **Computation formula**:

  $$
  \begin{aligned}
  \vec{x^{'}_{l}} &= \frac{1}{\sqrt{\frac{1}{d} \sum_{\dim=-2,\text{keepdim}=\text{True}} x_i^2 + \epsilon}}\\
  H^{pre}_l &= \alpha^{pre}_{l} ·(\vec{x^{'}_{l}}\varphi^{pre}_{l}) + b^{pre}_{l}\\
  H^{post}_l &= \alpha^{post}_{l} ·(\vec{x^{'}_{l}}\varphi^{post}_{l}) + b^{post}_{l}\\
  H^{res}_l &= \alpha^{res}_{l} ·(\vec{x^{'}_{l}}\varphi^{res}_{l}) + b^{res}_{l}\\
  H^{pre}_l &= \sigma (H^{pre}_{l})\\
  H^{post}_l &= 2\sigma (H^{post}_{l})\\
  h_{in} &=\vec{x_{l}}H^{pre}_l
  \end{aligned}
  $$

  - With $\mathbf{H^{res}_l}$ as the input, the Sinkhorn transformation performs $\mathbf{numIters}$ iterations in total. During the iterations, the intermediate normalization result $\mathbf{normOut}[k]$ and the summation result $\mathbf{sumOut}[k]$ are generated. Finally, the $\mathbf{normOut}$ of the last iteration is output as the transformation result.

    The first iteration (initialization):

    $$
    \begin{aligned}
        \mathbf{normOut}[0] &= \text{softmax}(\mathbf{H^{res}_l},  \dim=-1) + \epsilon, \\
        \mathbf{sumOut}[1] &= \sum_{\dim=-2,\text{keepdim}=\text{True}} \mathbf{normOut}[0] + \epsilon, \\
        \mathbf{normOut}[1] &= \frac{\mathbf{normOut}[0]}{\mathbf{sum\_out}[1]}, \\
    \end{aligned}
    $$

    The $i$-th iteration ($i = 1, 2, \dots, \mathbf({num\_iters}-1)$):

    $$
    \begin{aligned}
        \mathbf{sumOut}[2i] &= \sum_{\dim=-1,\text{keepdim}=\text{True}} \mathbf{normOut}[2i-1] + \epsilon, \\
        \mathbf{normOut}[2i] &= \frac{\mathbf{normOut}[2i-1]}{\mathbf{sum\_out}[2i]}, \\
        \mathbf{sumOut}[2i+1] &= \sum_{\dim=-2,\text{keepdim}=\text{True}} \mathbf{normOut}[2i] + \epsilon, \\
        \mathbf{normOut}[2i+1] &= \frac{\mathbf{normOut}[2i]}{\mathbf{sum\_out}[2i+1]}, \\
    \end{aligned}
    $$

  - Final output

  $$
  \mathbf{normOut}[2 \times \mathbf{num\_iters} - 1]
  $$

  $$
  \mathbf{sumOut}[2 \times \mathbf{num\_iters} - 1]
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
        <td>x</td>
        <td>Input</td>
        <td>Data to be computed, representing the input data of the mHC layer in the network.</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>phi</td>
        <td>Input</td>
        <td>Parameter matrix of mHC.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>alpha</td>
        <td>Input</td>
        <td>Scaling parameter of mHC.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>bias</td>
        <td>Input</td>
        <td>bias parameter of mHC.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>hcMult</td>
        <td>Optional attribute</td>
        <td>Number of residual streams, that is, the size of the HC dimension. Currently, only 4 is supported.</td>
        <td>INT32</td>
        <td>-</td>
      </tr>
      <tr>
        <td>numIters</td>
        <td>Optional attribute</td>
        <td>Number of iterations of the Sinkhorn algorithm. Currently, only 20 is supported.</td>
        <td>INT32</td>
        <td>-</td>
      </tr>
      <tr>
        <td>hcEps</td>
        <td>Optional attribute</td>
        <td>eps parameter added after the sigmoid of h_pre.</td>
        <td>DOUBLE</td>
        <td>-</td>
      </tr>
      <tr>
        <td>normEps</td>
        <td>Optional attribute</td>
        <td>Parameter that prevents division by zero in RmsNorm.</td>
        <td>DOUBLE</td>
        <td>-</td>
      </tr>
      <tr>
        <td>needBackward</td>
        <td>Optional attribute</td>
        <td>Whether to output additional attributes.</td>
        <td>BOOL</td>
        <td>-</td>
      </tr>
      <tr>
        <td>hIn</td>
        <td>Output</td>
        <td>The output h_in is used as the input of the Attention/MLP layer. The data type is the same as that of input x.</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>hPost</td>
        <td>Output</td>
        <td>Output h_post transformation matrix of mHC.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>hRes</td>
        <td>Output</td>
        <td>Output h_res transformation matrix of mHC.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>hPre</td>
        <td>Output</td>
        <td>Output when backward computation is required: the hPre matrix after the sigmoid computation. If needBackward is false, the output is invalid.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>hcBeforeNorm</td>
        <td>Output</td>
        <td>Output when backward computation is required: the result of the matrix multiplication between x and phi. If needBackward is false, the output is invalid.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>invRms</td>
        <td>Output</td>
        <td>Output when backward computation is required: 1/r computed by RmsNorm. If needBackward is false, the output is invalid.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>sumOut</td>
        <td>Output</td>
        <td>Output when backward computation is required: the colSum/rowSum result of each iteration. If needBackward is false, the output is invalid.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>normOut</td>
        <td>Output</td>
        <td>Output when backward computation is required: the comb result after each colSum/rowSum iteration. If needBackward is false, the output is invalid.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>

## Constraints

- Currently, n supports 4.
- Currently, numIters supports only 20.

- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
  - Input x supports only the 4D BSND format, with the shape (bs, seq_len, n, c).
  - The last dimension c of input x must be aligned to 128, and its value range is [1, 100000].
- <term>Ascend 950PR&950DT products</term>:
  - Input x supports the 4D BSND format and the 3D TND format, with the shapes (bs, seq_len, n, c) and (t, n, c), respectively.
  - In the TND format, t indicates the cumulative sum of the sequence lengths of all batches.
  - The last dimension c of input x supports only 4096 and 7168.
  - The data range of input phi is limited to $\pm \frac{1}{\sqrt{nc}}$, and eps is limited to 1e-6. This range provides good numerical stability.

## Calling Instructions

| Calling Method      | Calling Sample                 | Description                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_mhc_pre_sinkhorn](examples/test_aclnn_mhc_pre_sinkhorn.cpp) | Calls the MhcPreSinkhorn operator through the [aclnnMhcPreSinkhorn](docs/aclnnMhcPreSinkhorn_en.md) API. |
| PyTorch API | - | Calls the MhcPreSinkhorn operator through the [cann_ops_transformer.mhc_pre_sinkhorn](docs/torchapi_mhc_pre_sinkhorn_en.md) API. |
