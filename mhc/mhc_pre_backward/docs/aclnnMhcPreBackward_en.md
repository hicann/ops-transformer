# aclnnMhcPreBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T14:51:32.218Z -->

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

- Description: MhcPreBackward is the backward operator of MhcPre. The MhcPre operator obtains the $H^{res}$ and $H^{post}$ projection matrices and the Attention or MLP layer input matrix $h^{in}$ in the mHC (Manifold-Constrained Hyper-Connections) architecture through a series of computations.

- Formula:

  - Output combined gradient computation:

    - Forward formula:

      $$
      H\_in = \sum_{i=1}^{N} x[{B,S,i,:}] · H\_pre_n[B,S,i]
      $$

    - Backward computation:

      $$
      \begin{aligned}
      H\_pre\_grad &= \text{Reduce}\left(H\_in\_grad.\text{unsqueeze}(-2) \odot x, \text{dim}=-1\right) \quad ([B,S,N]) \\
      x\_grad\_vec3 &= H\_in\_grad \times H\_pre \quad ([B,S,N,D])
      \end{aligned}
      $$

  - Sigmoid gating backward (H_pre):
    - Forward formula:

      $$
      H\_pre = \text{Sigmoid}(\alpha\_pre * H\_pre\_1 + bias\_pre) + hc\_eps
      $$

    - Backward computation:

      $$
      \begin{aligned}
      s &= H\_pre - hc\_eps \\
      H\_pre\_2\_grad &= H\_pre\_grad \odot s \odot (1 - s) \\
      H\_pre\_1\_grad &= H\_pre\_2\_grad \cdot \alpha\_pre \\
      \alpha\_pre\_grad &= \sum_{b,s,n}^{B,S,N} \left(H\_pre\_2\_grad \cdot H\_pre\_1\right) \\
      bias\_pre\_grad &= \sum_{b,s}^{B,S} H\_pre\_2\_grad \quad ([N])
      \end{aligned}
      $$

  - Sigmoid gating backward (H_post):
    - Forward formula:

      $$
      H\_post = \text{Sigmoid}(\alpha\_post * H\_post\_1 + bias\_post) * 2
      $$

    - Backward computation:

      $$
      \begin{aligned}
      H\_post\_2\_grad &= H\_post\_grad \odot \left(H\_post \cdot \left(1 - \frac{H\_post}{2}\right)\right) \\
      H\_post\_1\_grad &= H\_post\_2\_grad \cdot \alpha_{post} \\
      \alpha_{post\_grad} &= \sum_{b,s,n}^{B,S,N} \left(H\_post\_2\_grad \cdot H_{post\_1}\right) \\
      bias\_post\_grad &= \sum_{b,s}^{B,S} H\_post\_2\_grad \quad ([N])
      \end{aligned}
      $$

  - Residual connection backward (H_res):
    - Forward formula:

      $$
      H\_res = \alpha\_res * H\_res\_1 + bias\_res
      $$

    - Backward computation:

      $$
      \begin{aligned}
      H\_res\_2\_grad &= H\_res\_grad \cdot \alpha_{res} \quad ([B,S,N,N]) \\
      \alpha\_res\_grad &= \sum_{b,s,i,j}^{B,S,N,N} \left(H\_res\_grad \cdot H\_res\_2\right) \\
      bias\_res\_grad &= \sum_{b,s}^{B,S} H\_res\_grad \quad ([N,N]) \\
      H\_res\_1\_grad &= \text{Reshape}(H\_res\_2\_grad) \quad ([B,S,N^2])
      \end{aligned}
      $$

  - RMSNorm Fusion backward:
    - Forward formula:

      $$
      H\_mix\_tmp = H\_mix * inv\_rms
      $$

    - Backward computation:

      $$
      \begin{aligned}
      H\_mix\_tmp\_grad &= \text{Concat}(H\_pre\_1\_grad, H\_post\_1\_grad, H\_res\_1\_grad) \quad ([B,S,2N+N^2]) \\
      H\_mix\_grad &= H\_mix\_tmp\_grad \cdot inv\_rms \\
      inv\_rms_{grad} &= \sum_{\text{last\_dim}} \left(H\_mix\_tmp\_grad \cdot H\_mix\right) \quad ([B,S,1])
      \end{aligned}
      $$

  - Matrix multiplication backward:
    - Forward formula:

      $$
      H\_mix = x\_rs @ phi^T
      $$

      $$
      x\_rs = x * gamma
      $$

    - Backward computation:

      $$
      \begin{aligned}
      x\_rs\_grad &= H\_mix\_grad @ phi \quad ([B,S,ND]) \\
      X &= \text{Reshape}(x\_rs, [B\cdot S, ND]) \\
      G &= \text{Reshape}(H\_mix\_grad, [B\cdot S, 2N+N^2]) \\
      phi_{grad} &= G^T @ X \quad ([2N+N^2, ND])
      \end{aligned}
      $$

  - Feature scaling backward:
    - Forward formula:

      $$
      x\_rs = x * gamma
      $$

    - Backward computation:

      $$
      \begin{aligned}
      x\_grad\_mm &= x\_rs\_grad * gamma \\
      gamma\_grad &= \sum_{b=1}^{B}\sum_{s=1}^{S} (x * x\_rs\_grad)\quad ([N,D])
      \end{aligned}
      $$

  - RMS normalization gradient computation:
    - Forward formula:

      $$
      inv\_rms = \frac{1}{\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2 + eps}}, \quad where \ n = N * D
      $$

    - Backward computation:

      $$
      \begin{aligned}
      x\_rs\_grad\_inv &= - \left(\frac{inv\_rms\_grad \cdot {inv\_rms}^3}{N*D}\right) \cdot x\_rs \\
      x\_rs\_grad &= x\_grad\_mm + x\_rs\_grad\_inv \\
      x\_grad\_vec1 &= \text{Reshape}(x\_rs\_grad, [B,S,N,D]) \\
      x\_grad &= x\_grad\_vec3 + x\_grad\_vec1
      \end{aligned}
      $$

  - Fusion of the addition of grad_x from mhc_post:

    $$
    \begin{aligned}
    x\_grad &= x\_grad + grad\_x\_post
    \end{aligned}
    $$

## Function Prototype

Each operator is divided into two phases: [two-phase API](../../../docs/en/context/two_phase_api.md). First call "aclnnMhcPreBackwardGetWorkspaceSize" to obtain the workspace size required for the computation and the executor containing the operator computation process, and then call "aclnnMhcPreBackward" to execute the computation.

```Cpp
aclnnStatus aclnnMhcPreBackwardGetWorkspaceSize(
    const aclTensor     *x,
    const aclTensor     *phi,
    const aclTensor     *alpha,
    const aclTensor     *gradHIn,
    const aclTensor     *gradHPost,
    const aclTensor     *gradHRes,
    const aclTensor     *invRms,
    const aclTensor     *hMix,
    const aclTensor     *hPre,
    const aclTensor     *hPost,
    const aclTensor     *gammaOptional,
    const aclTensor     *gradXPostOptional,
    float               hcEps,
    const aclTensor     *gradX,
    const aclTensor     *gradPhi,
    const aclTensor     *gradAlpha,
    const aclTensor     *gradBias,
    const aclTensor     *gradGamma,
    uint64_t            *workspaceSize,
    aclOpExecutor       **executor)
```

```Cpp
aclnnStatus aclnnMhcPreBackward(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream       stream)
```

## aclnnMhcPreBackwardGetWorkspaceSize

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1550px">
        <colgroup>
            <col style="width: 220px">
            <col style="width: 120px">
            <col style="width: 200px">
            <col style="width: 400px">
            <col style="width: 212px">
            <col style="width: 100px">
            <col style="width: 290px">
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
            <td>x (aclTensor*)</td>
            <td>Input</td>
            <td>Data to be computed, representing the layer input of mHC in the network.</td>
            <td><ul><li>Empty tensors are not supported.</li><li>The shape is (B, S, N, D) or (T, N, D), where B and S support generalization and T=B*S.</li></ul></td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>3-4</td>
            <td>√</td>
        </tr>
        <tr>
            <td>phi (aclTensor*)</td>
            <td>Input</td>
            <td>Parameter matrix of mHC.</td>
            <td><ul><li>Empty tensor not supported.</li><li>The shape is (2N+N*N, N*D) or (2N+N!, N*D). N keeps consistent with N in `x`; D keeps consistent with D in `x`.</li></ul></td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>2</td>
            <td>√</td>
        </tr>
        <tr>
            <td>alpha (aclTensor*)</td>
            <td>Input</td>
            <td>Scaling parameter alpha of mHC.</td>
            <td><ul><li>Empty tensor not supported.</li><li>The shape is (3).</li></ul></td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>1</td>
            <td>√</td>
        </tr>
        <tr>
            <td>gradHIn (aclTensor*)</td>
            <td>Input</td>
            <td>hIn serves as the layer input of the Attention/MLP layers. Outputs the corresponding gradient of the forward output hIn.</td>
            <td><ul><li>Empty tensors are not supported.</li><li>shape is (B, S, D) or (T, D). Where B keeps consistent with the B of `x`; S keeps consistent with the S of `x`; T=B*S; D keeps consistent with the D of `x`.</li></ul></td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>2-3</td>
            <td>√</td>
        </tr>
        <tr>
            <td>gradHPost (aclTensor*)</td>
            <td>Input</td>
            <td>Gradient corresponding to the forward output hPost.</td>
            <td><ul><li>Empty tensor not supported.</li><li>The shape is (B,S,N) or (T,N). Here, B is kept consistent with B of `x`; S is kept consistent with S of `x`; T=B*S; N is kept consistent with N of `x`.</li></ul></td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>2-3</td>
            <td>√</td>
        </tr>
        <tr>
            <td>gradHRes (aclTensor*)</td>
            <td>Input</td>
            <td>Corresponding gradient of the forward output hRes.</td>
            <td><ul><li>Empty Tensor not supported.</li><li>The shape is (B, S, N, N), (T, N, N), (B, S, N!), or (T, N!). B, S, and N keep consistent with the B, S, and N of `x`; T=B*S. N! denotes the number of permutations of N (for example, N!=24 when N=4).</li></ul></td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>2-4</td>
            <td>√</td>
        </tr>
        <tr>
            <td>invRms (aclTensor*)</td>
            <td>Input</td>
            <td>invRms computed by the forward RmsNorm.</td>
            <td><ul><li>Empty tensors not supported.</li><li>The shape is (B, S) or (T), where B keeps consistent with the B of `x`; S keeps consistent with the S of `x`; T=B*S.</li></ul></td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>1-2</td>
            <td>√</td>
        </tr>
        <tr>
            <td>hMix (aclTensor*)</td>
            <td>Input</td>
            <td>Result of x@phi in the forward computation stream.</td>
            <td><ul><li>Empty tensors not supported.</li><li>The shape is (B, S, 2N+N*N), (B, S, 2N+N!), (T, 2N+N*N), or (T, 2N+N!), where B keeps consistent with the B of `x`; S keeps consistent with the S of `x`; T=B*S; N keeps consistent with the N of `x`. The last dimension must keep consistent with dimension 0 of `phi` (fusionSize).</li></ul></td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>2-3</td>
            <td>√</td>
        </tr>
        <tr>
            <td>hPre (aclTensor*)</td>
            <td>Input</td>
            <td>hPre matrix after the forward sigmoid computation.</td>
            <td><ul><li>Empty tensors are not supported.</li><li>The shape is (B, S, N) or (T, N). B must keep consistent with the B of `x`; S must keep consistent with the S of `x`; T=B*S; N must keep consistent with the N of `x`.</li></ul></td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>2-3</td>
            <td>√</td>
        </tr>
        <tr>
            <td>hPost (aclTensor*)</td>
            <td>Input</td>
            <td>Forward hPost output.</td>
            <td><ul><li>Empty tensor not supported.</li><li>The shape is (B, S, N) or (T, N). Here, B is kept consistent with the B of `x`; S is kept consistent with the S of `x`; T=B*S; and N is kept consistent with the N of `x`.</li></ul></td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>2-3</td>
            <td>√</td>
        </tr>
        <tr>
            <td>gammaOptional (aclTensor*)</td>
            <td>Optional input</td>
            <td>Scaling coefficient gamma of RmsNorm.</td>
            <td><ul><li>Empty Tensor not supported.</li><li>If nullptr is passed in, it indicates a tensor of all 1s.</li><li>The shape is (N, D). N keeps consistent with the N of `x`; D keeps consistent with the D of `x`.</li></ul></td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>2</td>
            <td>√</td>
        </tr>
        <tr>
            <td>gradXPostOptional (aclTensor*)</td>
            <td>Optional input</td>
            <td>gradX of the post backward output.</td>
            <td><ul><li>Empty tensors not supported.</li><li>If nullptr is passed in, it indicates an all-zero tensor.</li><li>The shape is (B, S, N, D) or (T, N, D), where B must keep consistent with B of `x`; S must keep consistent with S of `x`; T=B*S; N must keep consistent with N of `x`; and D must keep consistent with D of `x`.</li></ul></td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>3-4</td>
            <td>√</td>
        </tr>
        <tr>
            <td>hcEps (float)</td>
            <td>Optional input</td>
            <td>eps parameter after sigmoid of HPre.</td>
            <td>              <li>Recommended value: 1e-6</li>
               </td>
            <td>FLOAT32</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>gradX (aclTensor*)</td>
            <td>Output</td>
            <td>Gradient corresponding to x.</td>
            <td><ul><li>Empty tensor not supported.</li><li>Keep dimensions and data type consistent with input x.</li><li>shape is (B, S, N, D) or (T, N, D), where B keeps consistent with B of `x`; S keeps consistent with S of `x`; T=B*S; N keeps consistent with N of `x`; D keeps consistent with D of `x`.</li></ul></td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>3-4</td>
            <td>√</td>
        </tr>
        <tr>
            <td>gradPhi (aclTensor*)</td>
            <td>Output</td>
            <td>Corresponding gradient of phi.</td>
            <td><ul><li>Empty tensors not supported.</li><li>Dimensions keep consistent with input phi.</li><li>The shape is (2N+N*N, N*D) or (2N+N!, N*D), where N keeps consistent with N of `x`; D keeps consistent with D of `x`. The shape keeps consistent with `phi`.</li></ul></td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>2</td>
            <td>√</td>
        </tr>
        <tr>
            <td>gradAlpha (aclTensor*)</td>
            <td>Output</td>
            <td>Gradient corresponding to alpha.</td>
            <td><ul><li>Empty tensor not supported.</li><li>Dimensions keep consistent with input alpha.</li><li>Shape is (3).</li></ul></td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>1</td>
            <td>√</td>
        </tr>
          <tr>
            <td>gradBias (aclTensor*)</td>
            <td>Output</td>
            <td>Gradient corresponding to bias.</td>
            <td><ul><li>Empty tensor not supported.</li><li>The shape is (2N+N*N) or (2N+N!). It keeps consistent with the 0th dimension (fusionSize) of `phi`.</li></ul></td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>1</td>
            <td>√</td>
        </tr>
        <tr>
            <td>gradGamma (aclTensor*)</td>
            <td>Optional output</td>
            <td>Gradient corresponding to gamma.</td>
            <td><ul><li>Empty tensor not supported.</li><li>This variable is output only when the input gamma is not nullptr.</li><li>Its shape dimensions keep consistent with those of the input gamma.</li><li>The shape is (N, D), where N keeps consistent with the N of `x`; D keeps consistent with the D of `x`.</li></ul></td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>2</td>
            <td>√</td>
        </tr>
        <tr>
            <td>workspaceSize (uint64_t*)</td>
            <td>Output</td>
            <td>Returns the size of the Device-side-applied workspace.</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>executor (aclOpExecutor**)</td>
            <td>Output</td>
            <td>Returns the op executor, containing the operator computation flow.</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        </tbody>
    </table>

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>:
    - The shape of `phi` supports only (2N+N\*N, N\*D).
    - The shape of `gradHRes` supports only (B, S, N, N) and (T, N, N).
    - The shape of `hMix` supports only (B, S, 2N+N\*N) and (T, 2N+N\*N).
    - The shape of `gradPhi` supports only (2N+N\*N, N\*D).
    - The shape of `gradBias` supports only (2N+N\*N).
  <!-- end id7 -->

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

    <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
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
                <td>ACLNN_ERR_PARAM_INVALID</td>
                <td>161002</td>
                <td>The data types and data formats of the input variables x, phi, gamma, and alpha are not within the supported range.</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_RUNTIME_ERROR</td>
                <td>361001</td>
                <td>The internal API call to the NPU runtime fails.</td>
            </tr>
        </tbody>
    </table>

## aclnnMhcPreBackward

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
    <col style="width: 144px">
    <col style="width: 125px">
    <col style="width: 700px">
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
        <td>Memory address of the workspace side-applied on the Device.</td>
        </tr>
        <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Size of the workspace side-applied on the Device, obtained by the first-phase API aclnnMhcPreBackwardGetWorkspaceSize.</td>
        </tr>
        <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Executor of the op, containing the operator computation flow.</td>
        </tr>
        <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Specifies the Stream for executing tasks.</td>
        </tr>
    </tbody>
    </table>

- **Return Value**

  Returns aclnnStatus status codes. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnMhcPreBackward defaults to a deterministic implementation.

- Specification constraints

  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT products</term>:
    <table style="undefined;table-layout: fixed; width: 600px"><colgroup>
        <col style="width: 100px">
        <col style="width: 500px">
        </colgroup>
        <thead>
            <tr>
                <th>Format</th>
                <th>gradHRes shape</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>BSNN</td>
            <td>(B, S, N, N)</td>
        </tr>
        <tr>
            <td>TNN</td>
            <td>(T, N, N)</td>
        </tr>
        </tbody>
    </table>

    <table style="undefined;table-layout: fixed; width: 600px"><colgroup>
        <col style="width: 100px">
        <col style="width: 500px">
        </colgroup>
        <thead>
            <tr>
                <th>Specification Item</th>
                <th>Specification Description</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>N</td>
            <td>N currently supports 4, 6, and 8.</td>
        </tr>
        <tr>
            <td>D</td>
            <td>D supports the range 1-16384 and requires 64-element alignment.</td>
        </tr>
        <tr>
            <td>fusionSize</td>
            <td>Only N*N+2N is supported.</td>
        </tr>
        </tbody>
    </table>
  <!-- end id9 -->

  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:

    <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
        <col style="width: 100px">
        <col style="width: 300px">
        <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>Format</th>
                <th>gradHRes shape</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>BSNN</td>
            <td>(B, S, N, N)</td>
        </tr>
        <tr>
            <td>BSN!</td>
            <td>(B, S, N!)</td>
        </tr>
        <tr>
            <td>TNN</td>
            <td>(T, N, N)</td>
        </tr>
        <tr>
            <td>TN!</td>
            <td>(T, N!)</td>
        </tr>
        </tbody>
    </table>

    <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
        <col style="width: 100px">
        <col style="width: 300px">
        <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>Specification Item</th>
                <th>Specification Description</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>N</td>
            <td>N currently supports only the value 4.</td>
        </tr>
        <tr>
            <td>D</td>
            <td>D supports values within 100000 and requires 128-element alignment.</td>
        </tr>
        <tr>
            <td>fusionSize</td>
            <td>Supports two types of fusionSize: N!+2N and N*N+2N.</td>
        </tr>
        </tbody>
    </table>
  <!-- end id8 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include <numeric>
#include "acl/acl.h"
#include "aclnnop/aclnn_mhc_pre_backward.h"

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr, const char *name, size_t elemSize = sizeof(float))
{
    auto size = GetShapeSize(shape);
    size_t copyBytes = size * elemSize;
    if (elemSize == 2) {
        // BF16: read raw bytes, convert to float for display
        std::vector<uint16_t> rawData(size, 0);
        auto ret = aclrtMemcpy(rawData.data(), copyBytes, *deviceAddr, copyBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
        LOG_PRINT("%s result (first 10 elements):\n", name);
        for (int64_t i = 0; i < std::min(size, (int64_t)10); i++) {
            union {
                uint32_t i;
                float f;
            } u;
            u.i = (uint32_t)rawData[i] << 16;
            LOG_PRINT("  [%ld] = %f\n", i, u.f);
        }
    } else {
        // float32
        std::vector<float> resultData(size, 0);
        auto ret = aclrtMemcpy(resultData.data(), copyBytes, *deviceAddr, copyBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
        LOG_PRINT("%s result (first 10 elements):\n", name);
        for (int64_t i = 0; i < std::min(size, (int64_t)10); i++) {
            LOG_PRINT("  [%ld] = %f\n", i, resultData[i]);
        }
    }
}

int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    // Boilerplate: initialize AscendCL.
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateContext(context, deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetCurrentContext(*context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate device-side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host-side data to device-side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of a contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. (Boilerplate) device/context/stream initialization. For details, see the list of AscendCL external APIs.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct inputs and outputs, customized according to the API definition
    std::vector<int64_t> xShape = {1024, 4, 512};      // T, N, D
    std::vector<int64_t> phiShape = {24, 2048};        // 2 * N + N * N, N * D
    std::vector<int64_t> alphaShape = {3};             // Fixed size
    std::vector<int64_t> gradHInShape = {1024, 512};   // T, D
    std::vector<int64_t> gradHPostShape = {1024, 4};   // T, N
    std::vector<int64_t> gradHResShape = {1024, 4, 4}; // T, N, N
    std::vector<int64_t> invRmsShape = {1024};         // T
    std::vector<int64_t> hMixShape = {1024, 24};       // T, 2 * N + N * N
    std::vector<int64_t> hPreShape = {1024, 4};        // T, N
    std::vector<int64_t> hPostShape = {1024, 4};       // T, N
    std::vector<int64_t> gammaShape = {4, 512};        // N, D
    std::vector<int64_t> gradXShape = {1024, 4, 512};  // T, N, D
    std::vector<int64_t> gradPhiShape = {24, 2048};    // 2 * N + N * N, N * D
    std::vector<int64_t> gradAlphaShape = {3};         // Fixed size
    std::vector<int64_t> gradGammaShape = {4, 512};    // N, D
    std::vector<int64_t> gradBiasShape = {24};         // 2*N+N*N
    std::vector<int64_t> gradXPostOptionalShape = {1024, 4, 512};

    void *xDeviceAddr = nullptr;
    void *phiDeviceAddr = nullptr;
    void *alphaDeviceAddr = nullptr;
    void *gradHInDeviceAddr = nullptr;
    void *gradHPostDeviceAddr = nullptr;
    void *gradHResDeviceAddr = nullptr;
    void *invRmsDeviceAddr = nullptr;
    void *hMixDeviceAddr = nullptr;
    void *hPreDeviceAddr = nullptr;
    void *hPostDeviceAddr = nullptr;
    void *gammaDeviceAddr = nullptr;
    void *gradXDeviceAddr = nullptr;
    void *gradPhiDeviceAddr = nullptr;
    void *gradAlphaDeviceAddr = nullptr;
    void *gradBiasDeviceAddr = nullptr;
    void *gradGammaDeviceAddr = nullptr;
    void *gradXPostOptionalDeviceAddr = nullptr;

    aclTensor *x = nullptr;
    aclTensor *phi = nullptr;
    aclTensor *alpha = nullptr;
    aclTensor *gradHIn = nullptr;
    aclTensor *gradHPost = nullptr;
    aclTensor *gradHRes = nullptr;
    aclTensor *invRms = nullptr;
    aclTensor *hMix = nullptr;
    aclTensor *hPre = nullptr;
    aclTensor *hPost = nullptr;
    aclTensor *gamma = nullptr;
    aclTensor *gradX = nullptr;
    aclTensor *gradPhi = nullptr;
    aclTensor *gradAlpha = nullptr;
    aclTensor *gradBias = nullptr;
    aclTensor *gradGamma = nullptr;
    aclTensor *gradXPostOptional = nullptr;

    std::vector<short> xHostData(1024 * 4 * 512, 1.0);
    std::vector<float> phiHostData(24 * 2048, 1.0);
    std::vector<float> alphaHostData(3, 1.0);
    std::vector<short> gradHInHostData(1024 * 512, 1.0);
    std::vector<float> gradHPostHostData(1024 * 4, 1.0);
    std::vector<float> gradHResHostData(1024 * 4 * 4, 1.0);
    std::vector<float> invRmsHostData(1024, 1.0);
    std::vector<float> hMixHostData(1024 * 24, 1.0);
    std::vector<float> hPreHostData(1024 * 4, 1.0);
    std::vector<float> hPostHostData(1024 * 4, 1.0);
    std::vector<float> gammaHostData(4 * 512, 1.0);
    std::vector<short> gradXHostData(1024 * 4 * 512, 0);
    std::vector<float> gradPhiHostData(24 * 2048, 0);
    std::vector<float> gradAlphaHostData(3, 0);
    std::vector<float> gradBiasHostData(24, 0);
    std::vector<float> gradGammaHostData(4 * 512, 0);
    std::vector<short> gradXPostOptionalHostData(1024 * 4 * 512, 0);

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(phiHostData, phiShape, &phiDeviceAddr, aclDataType::ACL_FLOAT, &phi);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(alphaHostData, alphaShape, &alphaDeviceAddr, aclDataType::ACL_FLOAT, &alpha);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradHInHostData, gradHInShape, &gradHInDeviceAddr, aclDataType::ACL_BF16, &gradHIn);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradHPostHostData, gradHPostShape, &gradHPostDeviceAddr, aclDataType::ACL_FLOAT, &gradHPost);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradHResHostData, gradHResShape, &gradHResDeviceAddr, aclDataType::ACL_FLOAT, &gradHRes);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(invRmsHostData, invRmsShape, &invRmsDeviceAddr, aclDataType::ACL_FLOAT, &invRms);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hMixHostData, hMixShape, &hMixDeviceAddr, aclDataType::ACL_FLOAT, &hMix);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hPreHostData, hPreShape, &hPreDeviceAddr, aclDataType::ACL_FLOAT, &hPre);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hPostHostData, hPostShape, &hPostDeviceAddr, aclDataType::ACL_FLOAT, &hPost);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradXHostData, gradXShape, &gradXDeviceAddr, aclDataType::ACL_BF16, &gradX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradPhiHostData, gradPhiShape, &gradPhiDeviceAddr, aclDataType::ACL_FLOAT, &gradPhi);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradAlphaHostData, gradAlphaShape, &gradAlphaDeviceAddr, aclDataType::ACL_FLOAT, &gradAlpha);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradBiasHostData, gradBiasShape, &gradBiasDeviceAddr, aclDataType::ACL_FLOAT, &gradBias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradGammaHostData, gradGammaShape, &gradGammaDeviceAddr, aclDataType::ACL_FLOAT, &gradGamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradXPostOptionalHostData, gradXPostOptionalShape, &gradXPostOptionalDeviceAddr,
                          aclDataType::ACL_BF16, &gradXPostOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    float hc_eps = 1e-6;

    // 3. Call the CANN operator library API and replace it with the specific API name
    uint64_t workspaceSize = 80 * 1024 * 1024;
    aclOpExecutor *executor;

    // Call the first-phase API of aclnnMhcPreBackward
    ret = aclnnMhcPreBackwardGetWorkspaceSize(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost,
                                              gamma, gradXPostOptional, hc_eps, gradX, gradPhi, gradAlpha, gradBias,
                                              gradGamma, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMhcPreBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // Allocate device memory based on the workspaceSize computed by the first-phase API.
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // Call the second-phase API of aclnnMhcPreBackward
    ret = aclnnMhcPreBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMhcPreBackward failed. ERROR: %d\n", ret); return ret);

    // 4. (boilerplate) Synchronize and wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device side to the host side, modifying the code based on the specific API definition.
    PrintOutResult(gradXShape, &gradXDeviceAddr, "gradX", sizeof(short));
    PrintOutResult(gradPhiShape, &gradPhiDeviceAddr, "gradPhi");
    PrintOutResult(gradAlphaShape, &gradAlphaDeviceAddr, "gradAlpha");
    PrintOutResult(gradBiasShape, &gradBiasDeviceAddr, "gradBias");
    PrintOutResult(gradGammaShape, &gradGammaDeviceAddr, "gradGamma");

    // 6. Release aclTensor and aclScalar, modifying the code based on the specific API definition.
    aclDestroyTensor(x);
    aclDestroyTensor(phi);
    aclDestroyTensor(alpha);
    aclDestroyTensor(gradHIn);
    aclDestroyTensor(gradHPost);
    aclDestroyTensor(gradHRes);
    aclDestroyTensor(invRms);
    aclDestroyTensor(hMix);
    aclDestroyTensor(hPre);
    aclDestroyTensor(hPost);
    aclDestroyTensor(gamma);
    aclDestroyTensor(gradX);
    aclDestroyTensor(gradPhi);
    aclDestroyTensor(gradAlpha);
    aclDestroyTensor(gradBias);
    aclDestroyTensor(gradGamma);
    aclDestroyTensor(gradXPostOptional);

    // 7. Release device resources.
    aclrtFree(xDeviceAddr);
    aclrtFree(phiDeviceAddr);
    aclrtFree(alphaDeviceAddr);
    aclrtFree(gradHInDeviceAddr);
    aclrtFree(gradHPostDeviceAddr);
    aclrtFree(gradHResDeviceAddr);
    aclrtFree(invRmsDeviceAddr);
    aclrtFree(hMixDeviceAddr);
    aclrtFree(hPreDeviceAddr);
    aclrtFree(hPostDeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(gradXDeviceAddr);
    aclrtFree(gradPhiDeviceAddr);
    aclrtFree(gradAlphaDeviceAddr);
    aclrtFree(gradBiasDeviceAddr);
    aclrtFree(gradGammaDeviceAddr);
    aclrtFree(gradXPostOptionalDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
