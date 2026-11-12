# aclnnMhcPreSinkhornBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T14:57:43.170Z -->

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
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- **Description**: MhcPreSinkhornBackward is the backward operator of MhcPreSinkhorn. It computes the backward propagation of the corresponding gradients.
  - The interface uses HF32 as the computation mode for Cube operations.

- **Formula**:

    - **Part 1 computes H_res_grad**: Let the current gradient be $\mathbf{grad}_{\text{curr}} = \mathbf{gradHRes}$ (`gradHRes` is the input parameter, that is, the gradient of the forward output `h_res`). A total of $\mathbf{num\_iters}$ (number of Sinkhorn iterations, corresponding to `sk_iter_count`, currently only 20 is supported) iterations are performed.

        - **First $\mathbf{num\_iters} - 1$ iterations**: For iteration indices $i = \mathbf{num\_iters} - 1, \mathbf{num\_iters} - 2, \dots, 1$, execute in sequence, where $\mathbf{sumOut}$ (corresponding input parameter `sumOut`) is the intermediate sum result computed and saved during the Sinkhorn forward transformation, $\mathbf{normOut}$ (corresponding input parameter `normOut`) is the intermediate norm result computed and saved during the Sinkhorn forward transformation, and $\mathbf{dotProd}_k$ is the result of the $k$-th dot product computation (intermediate variable):

          $$
          \begin{aligned}
              \mathbf{dotProd}_{2i+1} &= \sum_{\dim=-2,\text{keepdim}=\text{True}} \left( \mathbf{grad}_{\text{curr}} \cdot \mathbf{normOut}_{2i+1} \right) \\
              \\
              \mathbf{grad}_{\text{curr}} &\gets \frac{\mathbf{grad}_{\text{curr}} - \mathbf{dotProd}_{2i+1}}{\mathbf{sumOut}_{2i+1}}, \\
              \\
              \mathbf{dotProd}_{2i} &= \sum_{\dim=-1,\text{keepdim}=\text{True}} \left( \mathbf{grad}_{\text{curr}} \cdot \mathbf{normOut}_{2i} \right), \\
              \\
              \mathbf{grad}_{\text{curr}} &\gets \frac{\mathbf{grad}_{\text{curr}} - \mathbf{dotProd}_{2i}}{\mathbf{sumOut}_{2i}}, \\
          \end{aligned}
          $$

        - **Last iteration**:

          $$
          \begin{aligned}
              \mathbf{dotProd}_{1} &= \sum_{\dim=-2,\text{keepdim}=\text{True}} \left( \mathbf{grad}_{\text{curr}} \cdot \mathbf{normOut}_{1} \right), \\
              \\
              \mathbf{grad}_{\text{curr}} &\gets \frac{\mathbf{grad}_{\text{curr}} - \mathbf{dotProd}_{1}}{\mathbf{sumOut}_{1}}, \\
              \\
              \mathbf{dotProd}_{0} &= \sum_{\dim=-1,\text{keepdim}=\text{True}} \left( \mathbf{grad}_{\text{curr}} \cdot \mathbf{normOut}_{0} \right), \\
              \\
              \mathbf{HResGrad} &\gets \left( \mathbf{grad}_{\text{curr}} - \mathbf{dotProd}_{0} \right) \cdot \mathbf{normOut}_{0}, \\
          \end{aligned}
          $$

    - **Input**: $\mathbf{gradHin} \in [B, S, C]$ (`gradHin` is the input parameter, that is, the gradient of the output `h_in`)

    - **Combined output gradient computation**:

        - **Forward computation formula**: where $\mathbf{x}$ is the input parameter (forward input x), $\mathbf{hPre}$ is the input parameter (forward saved intermediate result h_pre), and $N$ is the size of the N dimension in the input tensor.

        $$
        \mathbf{HIn} = \sum_{i=1}^{N} \mathbf{x}[B, S, i, :] \cdot \mathbf{hPre}[B, S, i]
        $$

        - **Backward computation formula**:

        $$
        \begin{aligned}
        \mathbf{HPreGrad} &= \text{Reduce}\left(\mathbf{gradHin}.\text{unsqueeze}(-2) \odot \mathbf{x}, \text{dim}=-1\right) \quad ([B,S,N]) \\
        \mathbf{xGradVec3} &= \mathbf{gradHin} \times \mathbf{hPre} \quad ([B,S,N,C])
        \end{aligned}
        $$

    - **Gated activation layer gradient computation**:

        - **Sigmoid gated backward (H_pre)**:

          - Forward formula:

            $$
            \mathbf{hPre} = \text{Sigmoid}(\mathbf{alphaPre} \cdot \mathbf{hPre1} + \mathbf{biasPre}) + \mathbf{hcEps}
            $$

            where $\mathbf{alphaPre}$ is the first element of the input parameter `alpha` (scaling coefficient corresponding to H_pre), $\mathbf{biasPre}$ is the first N elements of the input parameter `bias`, and $\mathbf{hcEps}$ is the input parameter `hcEps` (numerical stability parameter).

          - Backward computation:

            $$
            \begin{aligned}
            \mathbf{s} &= \mathbf{hPre} - \mathbf{hcEps} \\
            \mathbf{HPre2Grad} &= \mathbf{HPreGrad} \odot \mathbf{s} \odot (1 - \mathbf{s}) \\
            \mathbf{HPre1Grad} &= \mathbf{HPre2Grad} \cdot \mathbf{alphaPre} \\
            \mathbf{alphaPreGrad} &= \sum_{b,s,n}^{B,S,N} \left(\mathbf{HPre2Grad} \cdot \mathbf{hPre1}\right) \\
            \mathbf{biasPreGrad} &= \sum_{b,s}^{B,S} \mathbf{HPre2Grad} \quad ([N])
            \end{aligned}
            $$

        - **Sigmoid gated backward (H_post)**:

          - Forward formula:

            $$
            \mathbf{hPost} = \text{Sigmoid}(\mathbf{alphaPost} \cdot \mathbf{hPost1} + \mathbf{biasPost}) \times 2
            $$

            where $\mathbf{alphaPost}$ is the second element of the input parameter `alpha`, and $\mathbf{biasPost}$ is the middle N elements of the input parameter `bias`.

          - Backward computation:

            $$
            \begin{aligned}
            \mathbf{HPost2Grad} &= \mathbf{gradHPost} \odot \left(\mathbf{hPost} \cdot \left(1 - \frac{\mathbf{hPost}}{2}\right)\right) \\
            \mathbf{HPost1Grad} &= \mathbf{HPost2Grad} \cdot \mathbf{alphaPost} \\
            \mathbf{alphaPostGrad} &= \sum_{b,s,n}^{B,S,N} \left(\mathbf{HPost2Grad} \cdot \mathbf{hPost1}\right) \\
            \mathbf{biasPostGrad} &= \sum_{b,s}^{B,S} \mathbf{HPost2Grad} \quad ([N])
            \end{aligned}
            $$

            where $\mathbf{gradHPost}$ is an input parameter (the gradient of output `h_post`).

        - **Residual Connection Backward (H_res)**:

          - Forward formula:

            $$
            \mathbf{hRes} = \mathbf{alphaRes} \cdot \mathbf{hRes1} + \mathbf{biasRes}
            $$

            where $\mathbf{alphaRes}$ is the third element of input parameter `alpha`, and $\mathbf{biasRes}$ is the last $N^2$ elements of input parameter `bias`.

          - Backward computation:

            $$
            \begin{aligned}
            \mathbf{HRes2Grad} &= \mathbf{HResGrad} \cdot \mathbf{alphaRes} \quad ([B,S,N,N]) \\
            \mathbf{alphaResGrad} &= \sum_{b,s,i,j}^{B,S,N,N} \left(\mathbf{HResGrad} \cdot \mathbf{hRes2}\right) \\
            \mathbf{biasResGrad} &= \sum_{b,s}^{B,S} \mathbf{HResGrad} \quad ([N,N]) \\
            \mathbf{HRes1Grad} &= \text{Reshape}(\mathbf{HRes2Grad}) \quad ([B,S,N^2])
            \end{aligned}
            $$

    - **RMS Normalization Fusion Backward**:

        - **RMSNorm Fusion Backward**:

          - Forward formula:

            $$
            \mathbf{hMixTmp} = \mathbf{hMix} \cdot \mathbf{invRms}
            $$

            where $\mathbf{hMix}$ is the output of the linear projection layer, and $\mathbf{invRms}$ is an input parameter (the forward saved intermediate result inv_rms).

          - Backward computation:

            $$
            \begin{aligned}
            \mathbf{hMixTmpGrad} &= \text{Concat}(\mathbf{HPre1Grad}, \mathbf{HPost1Grad}, \mathbf{HRes1Grad}) \quad ([B,S,2N+N^2]) \\
            \mathbf{hMixGrad} &= \mathbf{hMixTmpGrad} \cdot \mathbf{invRms} \\
            \mathbf{invRmsGrad} &= \sum_{\text{last\_dim}} \left(\mathbf{hMixTmpGrad} \cdot \mathbf{hMix}\right) \quad ([B,S,1])
            \end{aligned}
            $$

    - **Linear projection layer gradient computation**:

        - **Matrix multiplication backward**:

          - Forward formula:

            $$
            \mathbf{hMix} = \mathbf{xRs} @ \mathbf{phi}^T
            $$

            $$
            \mathbf{xRs} = \mathbf{x} \cdot \mathbf{gamma}
            $$

            where $\mathbf{phi}$ is the input parameter (forward parameter phi), and $\mathbf{gamma}$ is the scaling parameter of RMS normalization (computed from input $\mathbf{x}$).

          - Backward computation:

            $$
            \begin{aligned}
            \mathbf{xRsGrad} &= \mathbf{hMixGrad} @ \mathbf{phi} \quad ([B,S,N \cdot C]) \\
            \mathbf{X} &= \text{Reshape}(\mathbf{xRs}, [B \cdot S, N \cdot C]) \\
            \mathbf{G} &= \text{Reshape}(\mathbf{hMixGrad}, [B \cdot S, 2N+N^2]) \\
            \mathbf{phiGrad} &= \mathbf{G}^T @ \mathbf{X} \quad ([2N+N^2, N \cdot C])
            \end{aligned}
            $$

        - **Feature scaling backward**:

          - Forward formula:

            $$
            \mathbf{xRs} = \mathbf{x} \cdot \mathbf{gamma}
            $$

          - Backward computation:

            $$
            \begin{aligned}
            \mathbf{xGradMm} &= \mathbf{xRsGrad} \cdot \mathbf{gamma} \\
            \mathbf{gammaGrad} &= \sum_{b=1}^{B}\sum_{s=1}^{S} (\mathbf{x} \cdot \mathbf{xRsGrad}) \quad ([N,C])
            \end{aligned}
            $$

    - **RMS normalization gradient computation**:

      - Forward formula:

        $$
        \mathbf{invRms} = \frac{1}{\sqrt{\frac{1}{n}\sum_{i=1}^{n}\mathbf{x}_i^2 + \mathbf{eps}}}, \quad where \ n = N \times C
        $$

        where $\mathbf{eps}$ is the numerical stability parameter of RMS normalization.

      - Backward computation:

        $$
        \begin{aligned}
        \mathbf{xRsGradInv} &= - \left(\frac{\mathbf{invRmsGrad} \cdot \mathbf{invRms}^3}{N \times C}\right) \cdot \mathbf{xRs} \\
        \mathbf{xRsGrad} &= \mathbf{xGradMm} + \mathbf{xRsGradInv} \\
        \mathbf{xGradVec1} &= \text{Reshape}(\mathbf{xRsGrad}, [B,S,N,C]) \\
        \mathbf{xGrad} &= \mathbf{xGradVec3} + \mathbf{xGradVec1}
        \end{aligned}
        $$

    - **Symbol Description**:

      | Symbol | Meaning |
      |------|------|
      | $\mathbf{grad}_{\text{curr}}$ | Current gradient (intermediate variable in iterations). |
      | $\mathbf{sumOut}_k$ | Sum result of the $k$-th iteration (corresponding input parameter `sumOut`). |
      | $\mathbf{normOut}_k$ | Norm result of the $k$-th iteration (corresponding input parameter `normOut`). |
      | $\mathbf{dotProd}_k$ | Result of the $k$-th dot product computation (intermediate variable). |
      | $\mathbf{gradHin}$ | Gradient of output `h_in` (input parameter). |
      | $\mathbf{gradHPost}$ | Gradient of output `h_post` (input parameter). |
      | $\mathbf{gradHRes}$ | Gradient of output `h_res` (input parameter). |
      | $\mathbf{x}$ | Forward input x (input parameter). |
      | $\mathbf{hPre}$ | Forward saved intermediate result h_pre (input parameter). |
      | $\mathbf{hMix}$ | Output of the linear projection layer (intermediate variable). |
      | $\mathbf{hMixTmp}$ | Output after RMS normalization (intermediate variable). |
      | $\mathbf{invRms}$ | Forward saved intermediate result inv_rms (input parameter). |
      | $\mathbf{phi}$ | Forward parameter phi (input parameter). |
      | $\mathbf{alpha}$ | Forward parameter alpha (input parameter, containing 3 elements corresponding to the scaling coefficients of H_pre, H_post, and H_res, respectively). |
      | $\mathbf{bias}$ | Forward parameter bias (input parameter, containing 2N+N^2 elements). |
      | $\mathbf{hcEps}$ | Numerical stability parameter (input parameter). |
      | $B, S, N, C$ | Batch size, sequence length, number of experts, and feature dimension, respectively. |
      | $\cdot$ | Dot product (inner product) or element-wise multiplication. |
      | $\odot$ | Element-wise multiplication. |
      | $\sum_{\dim=d,\text{keepdim}=\text{True}}$ | Sums along the specified dimension and keeps the dimension. |

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First, call the `aclnnMhcPreSinkhornBackwardGetWorkspaceSize` API to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call `aclnnMhcPreSinkhornBackward` to perform the actual computation.

```cpp
aclnnStatus aclnnMhcPreSinkhornBackwardGetWorkspaceSize(
    const aclTensor  *gradHin,
    const aclTensor  *gradHPost,
    const aclTensor  *gradHRes,
    const aclTensor  *x,
    const aclTensor  *phi,
    const aclTensor  *alpha,
    const aclTensor  *bias,
    const aclTensor  *hPre,
    const aclTensor  *hcBeforeNorm,
    const aclTensor  *invRms,
    const aclTensor  *sumOut,
    const aclTensor  *normOut,
    double            hcEps,
    const aclTensor  *gradX,
    const aclTensor  *gradPhi,
    const aclTensor  *gradAlpha,
    const aclTensor  *gradBias,
    uint64_t         *workspaceSize,
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnMhcPreSinkhornBackward(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream       stream)
```

## aclnnMhcPreSinkhornBackwardGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1550px">
    <colgroup>
    <col style="width: 265px">
    <col style="width: 120px">
    <col style="width: 223px">
    <col style="width: 391px">
    <col style="width: 181px">
    <col style="width: 111px">
    <col style="width: 126px">
    <col style="width: 145px">
        </colgroup>
    <thead>
    <tr>
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Instruction</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Dimension (shape)</th>
        <th>Non-contiguous Tensor</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>gradHin(aclTensor *)</td>
        <td>Input</td>
        <td>Gradient of output h_in.</td>
        <td>Supports empty tensor.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>(B,S,C) or (T,C)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>gradHPost(aclTensor *)</td>
        <td>Input</td>
        <td>Gradient of output h_post.</td>
        <td>Supports empty tensor.</td>
        <td>FLOAT32 </td>
        <td>ND</td>
        <td>(B,S,N) or (T,N)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>gradHRes(aclTensor *)</td>
        <td>Input</td>
        <td>Gradient of the output h_res.</td>
        <td>Supports empty tensor.</td>
        <td>FLOAT32 </td>
        <td>ND</td>
        <td>(B,S,N*N) or (B,S,N,N) or (T,N*N) or (T,N,N)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>x(aclTensor *)</td>
        <td>Input</td>
        <td>Forward input x.</td>
        <td>Supports empty tensor.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>(B,S,N,C) or (T,N,C)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>phi(aclTensor *)</td>
        <td>Input</td>
        <td>Forward parameter phi.</td>
        <td>Does not support empty tensor.</td>
        <td>FLOAT32 </td>
        <td>ND</td>
        <td>(2N+N^2, N*C)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>alpha(aclTensor *)</td>
        <td>Input</td>
        <td>Forward parameter alpha.</td>
        <td>Does not support empty tensor.</td>
        <td>FLOAT32 </td>
        <td>ND</td>
        <td>(3)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>bias(aclTensor *)</td>
        <td>Input</td>
        <td>Forward parameter bias.</td>
        <td>Does not support empty tensor.</td>
        <td>FLOAT32 </td>
        <td>ND</td>
        <td>(2N+N^2)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>hPre(aclTensor *)</td>
        <td>Input</td>
        <td>Forward saved intermediate result h_pre.</td>
        <td>Supports empty tensor.</td>
        <td>FLOAT32 </td>
        <td>ND</td>
        <td>(B,S,N) or (T,N)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>hcBeforeNorm(aclTensor *)</td>
        <td>Input</td>
        <td>Forward saved intermediate result hc_before_norm.</td>
        <td>Supports empty tensor.</td>
        <td>FLOAT32 </td>
        <td>ND</td>
        <td>(B,S,N*N+2*N) or (T,N*N+2*N)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>invRms(aclTensor *)</td>
        <td>Input</td>
        <td>Forward saved intermediate result inv_rms.</td>
        <td>Supports empty tensor.</td>
        <td>FLOAT32 </td>
        <td>ND</td>
        <td>(B,S,1) or (T,1)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>sumOut(aclTensor *)</td>
        <td>Input</td>
        <td>Intermediate sum result computed and saved during the forward Sinkhorn transformation.</td>
        <td>Supports empty tensor.</td>
        <td>FLOAT32 </td>
        <td>ND</td>
        <td>(2*sk_iter_count,B,S,N) or (2*sk_iter_count,T,N)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>normOut(aclTensor *)</td>
        <td>Input</td>
        <td>Intermediate norm result computed and saved during the forward Sinkhorn transformation.</td>
        <td>Supports empty Tensor.</td>
        <td>FLOAT32 </td>
        <td>ND</td>
        <td>(2*sk_iter_count,B,S,N,N) or (2*sk_iter_count,T,N,N)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>hcEps(double)</td>
        <td>Input</td>
        <td>Numerical stability parameter.</td>
        <td>Recommended value: 1e-6.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>gradX(aclTensor *)</td>
        <td>Output</td>
        <td>Gradient of the input x.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>(B,S,N,C) or (T,N,C)</td>
        <td>-</td>
    </tr>
    <tr>
        <td>gradPhi(aclTensor *)</td>
        <td>Output</td>
        <td>Gradient of parameter phi.</td>
        <td>-</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>(2N+N^2, N*C)</td>
        <td>-</td>
    </tr>
    <tr>
        <td>gradAlpha(aclTensor *)</td>
        <td>Output</td>
        <td>Gradient of parameter alpha.</td>
        <td>-</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>(3)</td>
        <td>-</td>
    </tr>
    <tr>
        <td>gradBias(aclTensor *)</td>
        <td>Output</td>
        <td>Gradient of parameter bias.</td>
        <td>-</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>(2N+N^2)</td>
        <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize(uint64_t *)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be side-applied on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor(aclOpExecutor **)</td>
      <td>Output</td>
      <td>Returns the op executor containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    </tbody>
    </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 1180px"><colgroup>
      <col style="width: 250px">
      <col style="width: 130px">
      <col style="width: 800px">
    </colgroup>
        <thead>
            <th>Return Value</th>
            <th>Error Code</th>
            <th>Description</th>
        </thead>
        <tbody>
            <tr>
                <td>ACLNN_ERR_PARAM_NULLPTR</td>
                <td>161001</td>
                <td>A required parameter or output is a null pointer.</td>
            </tr>
            <tr>
                <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
                <td rowspan="5">161002</td>
                <td>The number of dimensions (dim) of the input or output tensor does not match the description in Parameters.</td>
            </tr>
            <tr>
                <td>The shape of the input or output tensor does not match the description in Parameters.</td>
            </tr>
            <tr>
                <td>The data type and data format of the input are not within the supported range.</td>
            </tr>
            <tr>
                <td>The value of N does not meet the constraint requirements.</td>
            </tr>
            <tr>
                <td>C does not meet the requirement of being greater than 0, less than 100000, and divisible by 128.</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_INNER_NULLPTR</td>
                <td>561103</td>
                <td>sk_iter_count is not equal to 20.</td>
            </tr>
        </tbody>
    </table>

## aclnnMhcPreSinkhornBackward

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1180px"><colgroup>
      <col style="width: 250px">
      <col style="width: 130px">
      <col style="width: 800px">
    </colgroup>
    <thead>
        <tr>
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
        </tr></thead>
    <tbody>
        <tr>
        <td>workspace</td>
        <td>Input</td>
        <td>Memory address of the Device side-applied workspace.</td>
        </tr>
        <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Size of the Device side-applied workspace, obtained by calling the first-phase API aclnnMhcPreSinkhornBackwardGetWorkspaceSize.</td>
        </tr>
        <tr>
        <td>executor</td>
        <td>Input</td>
        <td>op executor, which contains the operator computation flow.</td>
        </tr>
        <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream for executing the task.</td>
        </tr>
    </tbody>
    </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- **Dimension Format Constraints**

  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT products</term>: Inputs support two dimension formats: BSND (4D) and TND (3D), where T = B × S, indicating the cumulative sum of the sequence lengths of all batches. All inputs with B/S dimensions must use the same dimension format (all 4D BSND or all 3D TND).
  <!-- end id9 -->

  <!-- npu="A3,910b" id10 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: Inputs support only the BSND (4D) format.
  <!-- end id10 -->

- **Specification constraints**

  <!-- npu="950" id11 -->
  - N (for Ascend 950PR&950DT products): The N dimension supports values greater than 0 and no more than 8.
  <!-- end id11 -->

  <!-- npu="A3,910b" id12 -->
  - N (for <term>Atlas A2 products</term>, <term>Atlas A3 products</term>): The N dimension supports only 4.
  <!-- end id12 -->

  - C: Greater than 0, less than 100000, and divisible by 128.
  - sk_iter_count: Currently, only 20 Sinkhorn iterations are supported.

- Deterministic computation

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>: defaults to a deterministic implementation. Multiple calls with the same input produce the same result.
  <!-- end id7 -->

  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: default to a nondeterministic implementation. Determinism can be enabled through `aclrtSetSysParamOpt`.
  <!-- end id8 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_mhc_pre_sinkhorn_backward.h"

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                 \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t size = 1;
    for (int64_t dim : shape) {
        size *= dim;
    }
    return size;
}

void PrintTensorDataFloat(const std::vector<int64_t> &shape, void *device_addr)
{
    int64_t size = GetShapeSize(shape);
    std::vector<float> host_data(size, 0.0f);

    aclError ret = aclrtMemcpy(host_data.data(), size * sizeof(float), device_addr, size * sizeof(float),
                               ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Memcpy device to host failed, error: %d\n", ret); return);

    LOG_PRINT("Tensor data (first 10 elements): ");
    for (int64_t i = 0; i < std::min((int64_t)10, size); ++i) {
        LOG_PRINT("%f ", host_data[i]);
    }
    LOG_PRINT("\n");
}

void PrintTensorDataBfloat16(const std::vector<int64_t> &shape, void *device_addr)
{
    int64_t size = GetShapeSize(shape);
    std::vector<uint16_t> host_bf16(size);

    aclError ret = aclrtMemcpy(host_bf16.data(), size * sizeof(uint16_t), device_addr, size * sizeof(uint16_t),
                               ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Memcpy device to host failed, error: %d\n", ret); return);

    LOG_PRINT("Tensor data (first 10 elements, BF16 hex): ");
    for (int64_t i = 0; i < std::min((int64_t)10, size); ++i) {
        LOG_PRINT("0x%04x ", host_bf16[i]);
    }
    LOG_PRINT("\n");
}

int InitAcl(int32_t device_id, aclrtContext &context, aclrtStream &stream)
{
    aclError ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed, error: %d\n", ret); return -1);

    ret = aclrtSetDevice(device_id);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed, error: %d\n", ret); return -1);

    ret = aclrtCreateContext(&context, device_id);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed, error: %d\n", ret); return -1);

    ret = aclrtSetCurrentContext(context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed, error: %d\n", ret); return -1);

    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed, error: %d\n", ret); return -1);

    return 0;
}

int CreateAclTensorBfloat16(const std::vector<float> &host_data, const std::vector<int64_t> &shape, void *&device_addr,
                            aclTensor *&tensor)
{
    int64_t size = GetShapeSize(shape);
    std::vector<uint16_t> host_data_bf16(size);
    for (int64_t i = 0; i < size; ++i) {
        host_data_bf16[i] = static_cast<uint16_t>(static_cast<int16_t>(host_data[i] * 32767));
    }

    int64_t byte_size = size * sizeof(uint16_t);

    aclError ret = aclrtMalloc(&device_addr, byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed, error: %d\n", ret); return -1);

    ret = aclrtMemcpy(device_addr, byte_size, host_data_bf16.data(), byte_size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed, error: %d\n", ret); return -1);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    tensor = aclCreateTensor(shape.data(), shape.size(), ACL_BF16, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                             shape.size(), device_addr);
    CHECK_RET(tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return -1);

    return 0;
}

int CreateAclTensorFloat32(const std::vector<float> &host_data, const std::vector<int64_t> &shape, void *&device_addr,
                           aclTensor *&tensor)
{
    int64_t size = GetShapeSize(shape) * sizeof(float);

    aclError ret = aclrtMalloc(&device_addr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed, error: %d\n", ret); return -1);

    ret = aclrtMemcpy(device_addr, size, host_data.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed, error: %d\n", ret); return -1);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                             shape.size(), device_addr);
    CHECK_RET(tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return -1);

    return 0;
}

int CreateAclTensorBfloat16Output(const std::vector<int64_t> &shape, void *&device_addr, aclTensor *&tensor)
{
    int64_t size = GetShapeSize(shape);
    int64_t byte_size = size * sizeof(uint16_t);

    aclError ret = aclrtMalloc(&device_addr, byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed, error: %d\n", ret); return -1);

    tensor = aclCreateTensor(shape.data(), shape.size(), ACL_BF16, nullptr, 0, ACL_FORMAT_ND, shape.data(),
                             shape.size(), device_addr);
    CHECK_RET(tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return -1);

    return 0;
}

int CreateAclTensorFloat32Output(const std::vector<int64_t> &shape, void *&device_addr, aclTensor *&tensor)
{
    int64_t size = GetShapeSize(shape);
    int64_t byte_size = size * sizeof(float);

    aclError ret = aclrtMalloc(&device_addr, byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed, error: %d\n", ret); return -1);

    tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, shape.data(),
                             shape.size(), device_addr);
    CHECK_RET(tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return -1);

    return 0;
}

struct Tensors {
    void *grad_hin_addr = nullptr, *grad_h_post_addr = nullptr, *grad_h_res_addr = nullptr;
    void *x_addr = nullptr, *phi_addr = nullptr, *alpha_addr = nullptr, *bias_addr = nullptr;
    void *h_pre_addr = nullptr, *hc_before_norm_addr = nullptr, *inv_rms_addr = nullptr;
    void *sum_out_addr = nullptr, *norm_out_addr = nullptr;
    void *grad_x_addr = nullptr, *grad_phi_addr = nullptr, *grad_alpha_addr = nullptr, *grad_bias_addr = nullptr;
    aclTensor *grad_hin = nullptr, *grad_h_post = nullptr, *grad_h_res = nullptr;
    aclTensor *x = nullptr, *phi = nullptr, *alpha = nullptr, *bias = nullptr;
    aclTensor *h_pre = nullptr, *hc_before_norm = nullptr, *inv_rms = nullptr;
    aclTensor *sum_out = nullptr, *norm_out = nullptr;
    aclTensor *grad_x = nullptr, *grad_phi = nullptr, *grad_alpha = nullptr, *grad_bias = nullptr;
};

int CreateInputTensors(const std::vector<int64_t> &grad_hin_shape, const std::vector<int64_t> &grad_h_post_shape,
                       const std::vector<int64_t> &grad_h_res_shape, const std::vector<int64_t> &x_shape,
                       const std::vector<int64_t> &phi_shape, const std::vector<int64_t> &alpha_shape,
                       const std::vector<int64_t> &bias_shape, const std::vector<int64_t> &h_pre_shape,
                       const std::vector<int64_t> &hc_before_norm_shape, const std::vector<int64_t> &inv_rms_shape,
                       const std::vector<int64_t> &sum_out_shape, const std::vector<int64_t> &norm_out_shape,
                       Tensors &tensors)
{
    std::vector<float> grad_hin_host_data(GetShapeSize(grad_hin_shape), 1.0f);
    std::vector<float> grad_h_post_host_data(GetShapeSize(grad_h_post_shape), 1.0f);
    std::vector<float> grad_h_res_host_data(GetShapeSize(grad_h_res_shape), 1.0f);
    std::vector<float> x_host_data(GetShapeSize(x_shape), 1.0f);
    std::vector<float> phi_host_data(GetShapeSize(phi_shape), 1.0f);
    std::vector<float> alpha_host_data(GetShapeSize(alpha_shape), 1.0f);
    std::vector<float> bias_host_data(GetShapeSize(bias_shape), 1.0f);
    std::vector<float> h_pre_host_data(GetShapeSize(h_pre_shape), 1.0f);
    std::vector<float> hc_before_norm_host_data(GetShapeSize(hc_before_norm_shape), 1.0f);
    std::vector<float> inv_rms_host_data(GetShapeSize(inv_rms_shape), 1.0f);
    std::vector<float> sum_out_host_data(GetShapeSize(sum_out_shape), 1.0f);
    std::vector<float> norm_out_host_data(GetShapeSize(norm_out_shape), 1.0f);

    int ret = CreateAclTensorBfloat16(grad_hin_host_data, grad_hin_shape, tensors.grad_hin_addr, tensors.grad_hin);
    CHECK_RET(ret == 0, LOG_PRINT("Create grad_hin_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(grad_h_post_host_data, grad_h_post_shape, tensors.grad_h_post_addr, tensors.grad_h_post);
    CHECK_RET(ret == 0, LOG_PRINT("Create grad_h_post_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(grad_h_res_host_data, grad_h_res_shape, tensors.grad_h_res_addr, tensors.grad_h_res);
    CHECK_RET(ret == 0, LOG_PRINT("Create grad_h_res_tensor failed\n"); return -1);
    ret = CreateAclTensorBfloat16(x_host_data, x_shape, tensors.x_addr, tensors.x);
    CHECK_RET(ret == 0, LOG_PRINT("Create x_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(phi_host_data, phi_shape, tensors.phi_addr, tensors.phi);
    CHECK_RET(ret == 0, LOG_PRINT("Create phi_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(alpha_host_data, alpha_shape, tensors.alpha_addr, tensors.alpha);
    CHECK_RET(ret == 0, LOG_PRINT("Create alpha_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(bias_host_data, bias_shape, tensors.bias_addr, tensors.bias);
    CHECK_RET(ret == 0, LOG_PRINT("Create bias_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(h_pre_host_data, h_pre_shape, tensors.h_pre_addr, tensors.h_pre);
    CHECK_RET(ret == 0, LOG_PRINT("Create h_pre_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(hc_before_norm_host_data, hc_before_norm_shape, tensors.hc_before_norm_addr, tensors.hc_before_norm);
    CHECK_RET(ret == 0, LOG_PRINT("Create hc_before_norm_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(inv_rms_host_data, inv_rms_shape, tensors.inv_rms_addr, tensors.inv_rms);
    CHECK_RET(ret == 0, LOG_PRINT("Create inv_rms_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(sum_out_host_data, sum_out_shape, tensors.sum_out_addr, tensors.sum_out);
    CHECK_RET(ret == 0, LOG_PRINT("Create sum_out_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(norm_out_host_data, norm_out_shape, tensors.norm_out_addr, tensors.norm_out);
    CHECK_RET(ret == 0, LOG_PRINT("Create norm_out_tensor failed\n"); return -1);

    return 0;
}

int CreateOutputTensors(const std::vector<int64_t> &grad_x_shape, const std::vector<int64_t> &grad_phi_shape,
                        const std::vector<int64_t> &grad_alpha_shape, const std::vector<int64_t> &grad_bias_shape,
                        Tensors &tensors)
{
    int ret = CreateAclTensorBfloat16Output(grad_x_shape, tensors.grad_x_addr, tensors.grad_x);
    CHECK_RET(ret == 0, LOG_PRINT("Create grad_x_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(grad_phi_shape, tensors.grad_phi_addr, tensors.grad_phi);
    CHECK_RET(ret == 0, LOG_PRINT("Create grad_phi_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(grad_alpha_shape, tensors.grad_alpha_addr, tensors.grad_alpha);
    CHECK_RET(ret == 0, LOG_PRINT("Create grad_alpha_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(grad_bias_shape, tensors.grad_bias_addr, tensors.grad_bias);
    CHECK_RET(ret == 0, LOG_PRINT("Create grad_bias_tensor failed\n"); return -1);

    return 0;
}

void DestroyTensors(Tensors &tensors)
{
    aclDestroyTensor(tensors.grad_hin);
    aclDestroyTensor(tensors.grad_h_post);
    aclDestroyTensor(tensors.grad_h_res);
    aclDestroyTensor(tensors.x);
    aclDestroyTensor(tensors.phi);
    aclDestroyTensor(tensors.alpha);
    aclDestroyTensor(tensors.bias);
    aclDestroyTensor(tensors.h_pre);
    aclDestroyTensor(tensors.hc_before_norm);
    aclDestroyTensor(tensors.inv_rms);
    aclDestroyTensor(tensors.sum_out);
    aclDestroyTensor(tensors.norm_out);
    aclDestroyTensor(tensors.grad_x);
    aclDestroyTensor(tensors.grad_phi);
    aclDestroyTensor(tensors.grad_alpha);
    aclDestroyTensor(tensors.grad_bias);
}

void FreeDeviceMemory(Tensors &tensors)
{
    aclrtFree(tensors.grad_hin_addr);
    aclrtFree(tensors.grad_h_post_addr);
    aclrtFree(tensors.grad_h_res_addr);
    aclrtFree(tensors.x_addr);
    aclrtFree(tensors.phi_addr);
    aclrtFree(tensors.alpha_addr);
    aclrtFree(tensors.bias_addr);
    aclrtFree(tensors.h_pre_addr);
    aclrtFree(tensors.hc_before_norm_addr);
    aclrtFree(tensors.inv_rms_addr);
    aclrtFree(tensors.sum_out_addr);
    aclrtFree(tensors.norm_out_addr);
    aclrtFree(tensors.grad_x_addr);
    aclrtFree(tensors.grad_phi_addr);
    aclrtFree(tensors.grad_alpha_addr);
    aclrtFree(tensors.grad_bias_addr);
}

int main()
{
    int32_t device_id = 0;
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    Tensors tensors;

    int64_t bs = 2, seq_len = 128, n = 4, c = 256;
    int64_t hc_mult = n;
    int64_t hc_mix = n * n + 2 * n;
    int64_t num_iters = 20;
    double hc_eps = 1e-6;

    std::vector<int64_t> grad_hin_shape = {bs, seq_len, c};
    std::vector<int64_t> grad_h_post_shape = {bs, seq_len, n};
    std::vector<int64_t> grad_h_res_shape = {bs, seq_len, n, n};
    std::vector<int64_t> x_shape = {bs, seq_len, n, c};
    std::vector<int64_t> phi_shape = {hc_mix, n * c};
    std::vector<int64_t> alpha_shape = {3};
    std::vector<int64_t> bias_shape = {hc_mix};
    std::vector<int64_t> h_pre_shape = {bs, seq_len, n};
    std::vector<int64_t> hc_before_norm_shape = {bs, seq_len, hc_mix};
    std::vector<int64_t> inv_rms_shape = {bs, seq_len, 1};
    std::vector<int64_t> sum_out_shape = {num_iters * 2, bs, seq_len, n};
    std::vector<int64_t> norm_out_shape = {num_iters * 2, bs, seq_len, n, n};

    std::vector<int64_t> grad_x_shape = {bs, seq_len, n, c};
    std::vector<int64_t> grad_phi_shape = {hc_mix, n * c};
    std::vector<int64_t> grad_alpha_shape = {3};
    std::vector<int64_t> grad_bias_shape = {hc_mix};

    int ret = InitAcl(device_id, context, stream);
    CHECK_RET(ret == 0, LOG_PRINT("InitAcl failed, error: %d\n", ret); return -1);

    ret = CreateInputTensors(grad_hin_shape, grad_h_post_shape, grad_h_res_shape, x_shape,
                             phi_shape, alpha_shape, bias_shape, h_pre_shape,
                             hc_before_norm_shape, inv_rms_shape, sum_out_shape, norm_out_shape, tensors);
    CHECK_RET(ret == 0, return -1);

    ret = CreateOutputTensors(grad_x_shape, grad_phi_shape, grad_alpha_shape, grad_bias_shape, tensors);
    CHECK_RET(ret == 0, return -1);

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclnn_ret = aclnnMhcPreSinkhornBackwardGetWorkspaceSize(
        tensors.grad_hin, tensors.grad_h_post, tensors.grad_h_res,
        tensors.x, tensors.phi, tensors.alpha, tensors.bias,
        tensors.h_pre, tensors.hc_before_norm, tensors.inv_rms,
        tensors.sum_out, tensors.norm_out,
        hc_eps,
        tensors.grad_x, tensors.grad_phi, tensors.grad_alpha, tensors.grad_bias,
        &workspace_size, &executor);
    CHECK_RET(aclnn_ret == ACL_SUCCESS, LOG_PRINT("aclnnMhcPreSinkhornBackwardGetWorkspaceSize failed, error: %d\n", aclnn_ret);
              return -1);

    void *workspace_addr = nullptr;
    if (workspace_size > 0) {
        ret = aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc workspace failed, error: %d\n", ret); return -1);
    }

    aclnn_ret = aclnnMhcPreSinkhornBackward(workspace_addr, workspace_size, executor, stream);
    CHECK_RET(aclnn_ret == ACL_SUCCESS, LOG_PRINT("aclnnMhcPreSinkhornBackward failed, error: %d\n", aclnn_ret); return -1);

    CHECK_RET(aclrtSynchronizeStream(stream) == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed\n"); return -1);

    LOG_PRINT("MhcPreSinkhornBackward compute success!\n");
    LOG_PRINT("Output tensor grad_x (first 10 elements):\n");
    PrintTensorDataBfloat16(grad_x_shape, tensors.grad_x_addr);

    DestroyTensors(tensors);
    FreeDeviceMemory(tensors);
    if (workspace_size > 0)
        aclrtFree(workspace_addr);
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(device_id);
    aclFinalize();
    LOG_PRINT("All resources released successfully!\n");
    return 0;
}
```
