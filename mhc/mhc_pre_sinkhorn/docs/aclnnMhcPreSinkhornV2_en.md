# aclnnMhcPreSinkhornV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-20T00:00:00.000Z -->

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

- Description: Obtains the $\mathbf{H'_{res}}$ and $\mathbf{H_{post}}$ projection matrices of the hidden layer in the MHC architecture, as well as the input matrix $h_{in}$ of the Attention or MLP layer, through a series of computations. Performs Sinkhorn iterative normalization on the $\mathbf{H}'_{\text{res}}$ matrix to obtain the doubly stochastic matrix $\mathbf{H}_{\text{res}}$; supports outputting intermediate computation results for backward gradient computation, including the $\mathbf{H^{pre}_l}$ matrix after the sigmoid computation, the result of matrix multiplication between $\mathbf{\vec{x^{'}_{l}}}$ and $\mathbf{\varphi}$, the RmsNorm result $\mathbf{\vec{x^{'}_{l}}}$ of input x, the intermediate normalization result $normOut$ and sum result $sumOut$ during the iteration. The computation mode used in the Cube is HF32 by default (the same as before the premix capability was merged).

  <!-- npu="A3,910b" id12 -->
  <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: Controlled by the optional attribute `enableHf32`. When false is passed in, the mode switches to FP32 computation (default true).
  <!-- end id12 -->

- Formula:

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

  When the optional input premix is provided, the weighting coefficient of $h_{in}$ uses $premix$ to replace $\mathbf{H^{pre}_l}$ ($premix$ is usually filled with the hPre output from the previous layer; the remaining formulas are unchanged):

  $$
  h_{in} =\vec{x_{l}}{premix}_{l}
  $$

  - Using $\mathbf{H^{res}_l}$ as input, the Sinkhorn transformation performs a total of $numIters$ iterations. During the iterative process, intermediate normalization results ${normOut}[k]$ and summation results ${sumOut}[k]$ are generated. The $normOut$ from the last iteration is output as the transformation result.

    First iteration (initialization):

    $$
    \begin{aligned}
        {normOut}[0] &= \text{softmax}({H^{res}_l},  \dim=-1) + \epsilon, \\
        {sumOut}[1] &= \sum_{\dim=-2,\text{keepdim}=\text{True}} {normOut}[0] + \epsilon, \\
        {normOut}[1] &= \frac{{normOut}[0]}{{sumOut}[1]}, \\
    \end{aligned}
    $$

    The $i$-th iteration ($i = 1, 2, \dots, ({numIters}-1)$):

    $$
    \begin{aligned}
        {sumOut}[2i] &= \sum_{\dim=-1,\text{keepdim}=\text{True}} {normOut}[2i-1] + \epsilon, \\
        {normOut}[2i] &= \frac{{normOut}[2i-1]}{{sumOut}[2i]}, \\
        {sumOut}[2i+1] &= \sum_{\dim=-2,\text{keepdim}=\text{True}} {normOut}[2i] + \epsilon, \\
        {normOut}[2i+1] &= \frac{{normOut}[2i]}{{sumOut}[2i+1]}, \\
    \end{aligned}
    $$

  - Final output

    $$
    {normOut}[2 \times {numIters} - 1]
    $$

    $$
    {sumOut}[2 \times {numIters} - 1]
    $$

  - Symbols

    | Symbol                                     | Meaning                                       |
    | ------------------------------------------ | ------------------------------------------------- |
    | $x$                               | Input tensor (the $\mathbf{H}'_{\text{res}}$ matrix of the mHC layer). |
    | $\epsilon$                         | Parameter for preventing division by zero (corresponding to the input parameter `eps`). |
    | $\text{softmax}(\cdot, \dim=-1)$      | Performs softmax normalization along the last dimension. |
    | $\sum_{\dim=d,\text{keepdim}=\text{True}}$                         | Sums over the specified dimension $d$ and keeps the dimension. |
    | ${normOut}[k]$                         | Intermediate normalization result of the $k$-th step. |
    | ${sumOut}[k]$                         | Intermediate summation result of the $k$-th step. |
    | ${numIters}$                         | Number of iterations (input parameter). |

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the `aclnnMhcPreSinkhornV2GetWorkspaceSize` API to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the `aclnnMhcPreSinkhornV2` API to perform the actual computation.

```Cpp
aclnnStatus aclnnMhcPreSinkhornV2GetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *phi,
    const aclTensor *alpha,
    const aclTensor *bias,
    const aclTensor *premix,
    int64_t          hcMult,
    int64_t          numIters,
    double           hcEps,
    double           normEps,
    bool             needBackward,
    bool             enableHf32,
    aclTensor       *hin,
    aclTensor       *hPost,
    aclTensor       *hRes,
    aclTensor       *hPre,
    aclTensor       *hcBeforeNorm,
    aclTensor       *invRms,
    aclTensor       *sumOut,
    aclTensor       *normOut,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnMhcPreSinkhornV2(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnMhcPreSinkhornV2GetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1435px"><colgroup>
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 250px">
    <col style="width: 250px">
    <col style="width: 100px">
    <col style="width: 120px">
    <col style="width: 200px">
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
        <td>x (const aclTensor*)</td>
        <td>Input</td>
        <td>Data to be computed, representing the mHC layer input data in the network.</td>
        <td><ul><li>Supports empty tensor.</li><li>The shape supports (bs, seq_len, n, c) or (t, n, c).</li></ul></td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>3-4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>phi (const aclTensor*)</td>
        <td>Input</td>
        <td>Parameter matrix of mHC.</td>
        <td><ul><li>Supports empty tensor.</li><li>The shape supports (n * n + 2 * n, n * c).</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>alpha (const aclTensor*)</td>
        <td>Input</td>
        <td>Scaling parameter of mHC.</td>
        <td><ul><li>Supports empty tensor.</li><li>The shape supports (3).</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>bias (const aclTensor*)</td>
        <td>Input</td>
        <td>Bias parameter of mHC.</td>
        <td><ul><li>Supports empty tensor.</li><li>The shape supports (n * n + 2 * n).</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>premix (const aclTensor*)</td>
        <td>Input</td>
        <td>Optional input, premix weighting coefficient (usually filled with the hPre output from the previous layer).</td>
        <td><ul><li>Supports empty tensor.</li><li>The shape supports (bs, seq_len, n) or (t, n).</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>2-3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>hcMult (int64_t)</td>
        <td>Input</td>
        <td>Number of residual streams, size of the HC dimension.</td>
        <td>Currently, only 4 is supported.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>numIters (int64_t)</td>
        <td>Input</td>
        <td>Indicates the number of Sinkhorn algorithm iterations.</td>
        <td>Currently, only 20 is supported.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>hcEps (double)</td>
        <td>Input</td>
        <td>The eps parameter after the sigmoid of H<sub>pre</sub>.</td>
        <td>Recommended value: 1e-6.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>normEps (double)</td>
        <td>Input</td>
        <td>Division-by-zero prevention parameter of RmsNorm.</td>
        <td>Recommended value: 1e-6.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>needBackward (bool)</td>
        <td>Input</td>
        <td>Whether to output additional attributes.</td>
        <td>The recommended value is true.</td>
        <td></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>enableHf32 (bool)</td>
        <td>Input</td>
        <td>Optional attribute (V2 interface only), indicating whether the Cube enables the HF32 computation mode.</td>
        <td>Can be true (default, consistent with the V1 interface). Passing false switches to FP32.</td>
        <td></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>hin (aclTensor*)</td>
        <td>Output</td>
        <td>The output hin serves as the Attention/MLP layer input.</td>
        <td><ul><li>Empty tensor is not supported.</li><li>The shape supports (bs, seq_len, c) and (t, c).</li></ul></td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2-3</td>
        <td>×</td>
      </tr>
      <tr>
        <td>hPost (aclTensor*)</td>
        <td>Output</td>
        <td>hPost transformation matrix of the output mHC.</td>
        <td><ul><li>Empty tensor is not supported.</li><li>The shape supports (bs, seq_len, n) and (t, n).</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>2-3</td>
        <td>×</td>
      </tr>
      <tr>
        <td>hRes (aclTensor*)</td>
        <td>Output</td>
        <td>hRes transformation matrix of the output mHC.</td>
        <td><ul><li>Empty tensor is not supported.</li><li>The shape supports (bs, seq_len, n * n) and (t, n * n).</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>2-3</td>
        <td>×</td>
      </tr>
      <tr>
        <td>hPre (aclTensor*)</td>
        <td>Output</td>
        <td>Output when backward is required. The hPre matrix after the sigmoid computation.</td>
        <td><ul><li>Empty tensor is not supported.</li><li>When needBackward is false, this output is invalid; when premix is provided, a valid value is output even if needBackward is false.</li><li>The shape supports (bs, seq_len, n) and (t, n).</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>2-3</td>
        <td>×</td>
      </tr>
      <tr>
        <td>hcBeforeNorm (aclTensor*)</td>
        <td>Output</td>
        <td>Output when backward is required. The result of matrix multiplication between x and phi.</td>
        <td><ul><li>Empty tensor is not supported.</li><li>When needBackward is false, this output is invalid.</li><li>The shape supports (bs, seq_len, n*n + 2*n) and (t, n*n + 2*n).</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>2-3</td>
        <td>×</td>
      </tr>
      <tr>
        <td>invRms (aclTensor*)</td>
        <td>Output</td>
        <td>Output when backward is required, which is 1/r computed by RmsNorm.</td>
        <td><ul><li>Empty tensor is not supported.</li><li>When needBackward is false, this output is invalid.</li><li>The shape supports (bs, seq_len, 1) and (t, 1).</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>2-3</td>
        <td>×</td>
      </tr>
      <tr>
        <td>sumOut (aclTensor*)</td>
        <td>Output</td>
        <td>Output when backward is required. The colSum/rowSum result of each iteration.</td>
        <td><ul><li>Empty tensor is not supported.</li><li>When needBackward is false, this output is invalid.</li><li>The shape supports (sk_iter_count * 2, bs, seq_len, n) and (sk_iter_count * 2, t, n).</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>3-4</td>
        <td>×</td>
      </tr>
      <tr>
        <td>normOut (aclTensor*)</td>
        <td>Output</td>
        <td>Output when backward is required. The comb result after each colSum/rowSum iteration.</td>
        <td><ul><li>Empty tensor is not supported.</li><li>When needBackward is false, this output is invalid.</li><li>The shape supports (sk_iter_count * 2, bs, seq_len, n, n) and (sk_iter_count * 2, t, n, n).</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>4-5</td>
        <td>×</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t*)</td>
        <td>Output</td>
        <td>Returns the workspace size that needs to be applied on the Device side.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor (aclOpExecutor**)</td>
        <td>Output</td>
        <td>Returns the op executor, which contains the operator computation flow.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody>
  </table>

  <!-- npu="950" id11 -->
  - <term>Ascend 950PR&950DT products</term>: The parameter `enableHf32` is invalid in the current product usage scenario.
  <!-- end id11 -->

  > Note: A 4D input is in BSND format, and a 3D input is in TND format. Here, bs indicates the batch size, seq_len indicates the sequence length of a single batch, t indicates the sum of sequence lengths of all batches, n indicates the number of residual streams, and c indicates the hidden dimension size on the tail axis.

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 1202px"><colgroup>
  <col style="width: 262px">
  <col style="width: 121px">
  <col style="width: 819px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input parameter is a required input, output, or attribute and is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type, shape dimension, and data format of the input variables are not within the supported range.</td>
    </tr>
    <tr>
      <td>numIters is not 20.</td>
    </tr>
    <tr>
      <td>The value of hcMult is not 4.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>Calling the NPU Runtime API to allocate memory/create a Tensor fails.</td>
    </tr>
  </tbody>
  </table>

## aclnnMhcPreSinkhornV2

- **Parameters**
  <table style="undefined;table-layout: fixed; width: 1154px"><colgroup>
  <col style="width: 153px">
  <col style="width: 121px">
  <col style="width: 880px">
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
      <td>Device-side-applied workspace memory address.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Device-side-applied workspace size, obtained by the first-phase API aclnnMhcPreSinkhornV2GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream used to execute tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:

  - aclnnMhcPreSinkhornV2 uses a deterministic implementation by default.

- Batch consistency:
  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - Defaults to a non-Batch-consistent implementation. Batch consistency cannot be enabled through aclrtSetSysParamOpt.
  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>:
    - Defaults to a non-Batch-consistent implementation, and supports enabling Batch consistency through aclrtSetSysParamOpt. After Batch consistency is enabled, performance may degrade to some extent.
  <!-- end id8 -->

- Specification constraints

  | Specification Item | Value | Specification                       |
  | :----------------- | :---- | :---------------------------------- |
  | numIters           | 20    | Only 20 is supported.               |
  | hcMult             | 4     | Only 4 is supported.                |

  <!-- npu="A3,910b" id9 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - The input x supports only the 4D BSND format, with shape (bs, seq_len, n, c).
    - The trailing axis c must be a multiple of 128, with a value range of [1, 100000].
  <!-- end id9 -->
  <!-- npu="950" id10 -->
  - <term>Ascend 950PR&950DT products</term>:
    - The input x supports the 4D BSND format and 3D TND format, with shapes (bs, seq_len, n, c) and (t, n, c) respectively.
    - In the TND format, t indicates the cumulative sum of sequence lengths of all batches; the corresponding output shape is expanded along the t axis as described in Parameters.
    - The trailing axis c must be a multiple of 128, with a value range of [1, 100000].
    - The data range of input phi is limited to $\pm \frac{1}{\sqrt{nc}}$, and eps is limited to 1e-6. This range provides good numerical stability.
  <!-- end id10 -->

## Example

The sample code is as follows, for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_mhc_pre_sinkhorn.h"

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (int64_t dim : shape) {
        size *= dim;
    }
    return size;
}

void PrintTensorDataFloat(const std::vector<int64_t>& shape, void* device_addr)
{
    int64_t size = GetShapeSize(shape);
    std::vector<float> host_data(size, 0.0f);

    aclError ret = aclrtMemcpy(host_data.data(), size * sizeof(float), device_addr, size * sizeof(float),
                               ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Memcpy device to host failed, error: %d\n", ret); return);

    LOG_PRINT("Tensor data (first 10 elements): ");
    for (int64_t i = 0; i < std::min(static_cast<int64_t>(10), size); ++i) {
        LOG_PRINT("%f ", host_data[i]);
    }
    LOG_PRINT("\n");
}

void PrintTensorDataBfloat16(const std::vector<int64_t>& shape, void* device_addr)
{
    int64_t size = GetShapeSize(shape);
    std::vector<uint16_t> host_bf16(size);

    aclError ret = aclrtMemcpy(host_bf16.data(), size * sizeof(uint16_t), device_addr, size * sizeof(uint16_t),
                               ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Memcpy device to host failed, error: %d\n", ret); return);

    LOG_PRINT("Tensor data (first 10 elements, BF16 hex): ");
    for (int64_t i = 0; i < std::min(static_cast<int64_t>(10), size); ++i) {
        LOG_PRINT("0x%04x ", host_bf16[i]);
    }
    LOG_PRINT("\n");
}

int InitAcl(int32_t device_id, aclrtContext& context, aclrtStream& stream)
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

int CreateAclTensorFloat32(const std::vector<float>& host_data, const std::vector<int64_t>& shape, void*& device_addr,
                           aclTensor*& tensor)
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

int CreateAclTensorBfloat16(const std::vector<float>& host_data, const std::vector<int64_t>& shape, void*& device_addr,
                            aclTensor*& tensor)
{
    int64_t size = GetShapeSize(shape);
    std::vector<uint16_t> host_data_bf16(size);
    for (int64_t i = 0; i < size; ++i) {
        // fp32 -> bf16 standard rounding: take the upper 16 bits (sign 1 + exponent 8 + mantissa 7) and round the lower bits;
        // x*32767 truncation approximation cannot be used, because it changes the numerical magnitude.
        uint32_t bits = 0;
        std::memcpy(&bits, &host_data[i], sizeof(bits));
        uint32_t lsb = (bits >> 16) & 1U;
        bits = ((bits + (0x7FFFU + lsb)) >> 16) & 0xFFFFU;
        host_data_bf16[i] = static_cast<uint16_t>(bits);
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

int CreateAclTensorBfloat16Output(const std::vector<int64_t>& shape, void*& device_addr, aclTensor*& tensor)
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

int CreateAclTensorFloat32Output(const std::vector<int64_t>& shape, void*& device_addr, aclTensor*& tensor)
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
    void *x_addr = nullptr, *phi_addr = nullptr, *alpha_addr = nullptr, *bias_addr = nullptr;
    void* premix_addr = nullptr;
    void *hin_addr = nullptr, *h_post_addr = nullptr, *h_res_addr = nullptr;
    void *h_pre_addr = nullptr, *hc_before_norm_addr = nullptr, *inv_rms_addr = nullptr;
    void *sum_out_addr = nullptr, *norm_out_addr = nullptr;
    aclTensor *x = nullptr, *phi = nullptr, *alpha = nullptr, *bias = nullptr;
    aclTensor* premix = nullptr;
    aclTensor *hin = nullptr, *h_post = nullptr, *h_res = nullptr;
    aclTensor *h_pre = nullptr, *hc_before_norm = nullptr, *inv_rms = nullptr;
    aclTensor *sum_out = nullptr, *norm_out = nullptr;
};

int CreateInputTensors(const std::vector<int64_t>& x_shape, const std::vector<int64_t>& phi_shape,
                       const std::vector<int64_t>& alpha_shape, const std::vector<int64_t>& bias_shape,
                       const std::vector<int64_t>& premix_shape, Tensors& tensors)
{
    // Mild pseudo-random data: all 1.0 would cause the x*phi matrix multiplication to overflow, saturate the sigmoid, and make the sinkhorn output NaN.
    // Note: use a signed int64_t loop variable; with size_t, (mod - 9) underflows to a huge value.
    std::vector<float> x_host_data(GetShapeSize(x_shape), 0.0f);
    for (int64_t i = 0; i < static_cast<int64_t>(x_host_data.size()); ++i) {
        x_host_data[i] = 0.5f * static_cast<float>((i * 37 % 19) - 9) / 9.0f;
    }
    std::vector<float> phi_host_data(GetShapeSize(phi_shape), 0.0f);
    float phi_scale = 16.0f * static_cast<float>(phi_shape[1]); // 16 * (N*C)
    for (int64_t i = 0; i < static_cast<int64_t>(phi_host_data.size()); ++i) {
        phi_host_data[i] = static_cast<float>((i * 91 % 17) - 8) / phi_scale;
    }
    std::vector<float> alpha_host_data = {0.3f, 0.3f, 0.3f};
    std::vector<float> bias_host_data(GetShapeSize(bias_shape), 0.0f);
    for (int64_t i = 0; i < static_cast<int64_t>(bias_host_data.size()); ++i) {
        bias_host_data[i] = 0.05f * static_cast<float>((i * 13 % 7) - 3);
    }
    std::vector<float> premix_host_data(GetShapeSize(premix_shape), 0.0f);
    for (int64_t i = 0; i < static_cast<int64_t>(premix_host_data.size()); ++i) {
        premix_host_data[i] = 0.5f + static_cast<float>(i * 11 % 10) / 20.0f; // [0.5, 0.95]
    }

    int ret = CreateAclTensorBfloat16(x_host_data, x_shape, tensors.x_addr, tensors.x);
    CHECK_RET(ret == 0, LOG_PRINT("Create x_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(phi_host_data, phi_shape, tensors.phi_addr, tensors.phi);
    CHECK_RET(ret == 0, LOG_PRINT("Create phi_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(alpha_host_data, alpha_shape, tensors.alpha_addr, tensors.alpha);
    CHECK_RET(ret == 0, LOG_PRINT("Create alpha_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(bias_host_data, bias_shape, tensors.bias_addr, tensors.bias);
    CHECK_RET(ret == 0, LOG_PRINT("Create bias_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(premix_host_data, premix_shape, tensors.premix_addr, tensors.premix);
    CHECK_RET(ret == 0, LOG_PRINT("Create premix_tensor failed\n"); return -1);
    return 0;
}

int CreateOutputTensors(const std::vector<int64_t>& hin_shape, const std::vector<int64_t>& h_post_shape,
                        const std::vector<int64_t>& h_res_shape, const std::vector<int64_t>& h_pre_shape,
                        const std::vector<int64_t>& hc_before_norm_shape, const std::vector<int64_t>& inv_rms_shape,
                        const std::vector<int64_t>& sum_out_shape, const std::vector<int64_t>& norm_out_shape,
                        Tensors& tensors)
{
    int ret = CreateAclTensorBfloat16Output(hin_shape, tensors.hin_addr, tensors.hin);
    CHECK_RET(ret == 0, LOG_PRINT("Create hin_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(h_post_shape, tensors.h_post_addr, tensors.h_post);
    CHECK_RET(ret == 0, LOG_PRINT("Create h_post_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(h_res_shape, tensors.h_res_addr, tensors.h_res);
    CHECK_RET(ret == 0, LOG_PRINT("Create h_res_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(h_pre_shape, tensors.h_pre_addr, tensors.h_pre);
    CHECK_RET(ret == 0, LOG_PRINT("Create h_pre_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(hc_before_norm_shape, tensors.hc_before_norm_addr, tensors.hc_before_norm);
    CHECK_RET(ret == 0, LOG_PRINT("Create hc_before_norm_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(inv_rms_shape, tensors.inv_rms_addr, tensors.inv_rms);
    CHECK_RET(ret == 0, LOG_PRINT("Create inv_rms_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(sum_out_shape, tensors.sum_out_addr, tensors.sum_out);
    CHECK_RET(ret == 0, LOG_PRINT("Create sum_out_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(norm_out_shape, tensors.norm_out_addr, tensors.norm_out);
    CHECK_RET(ret == 0, LOG_PRINT("Create norm_out_tensor failed\n"); return -1);
    return 0;
}

void DestroyTensors(Tensors& tensors)
{
    aclDestroyTensor(tensors.x);
    aclDestroyTensor(tensors.phi);
    aclDestroyTensor(tensors.alpha);
    aclDestroyTensor(tensors.bias);
    aclDestroyTensor(tensors.premix);
    aclDestroyTensor(tensors.hin);
    aclDestroyTensor(tensors.h_post);
    aclDestroyTensor(tensors.h_res);
    aclDestroyTensor(tensors.h_pre);
    aclDestroyTensor(tensors.hc_before_norm);
    aclDestroyTensor(tensors.inv_rms);
    aclDestroyTensor(tensors.sum_out);
    aclDestroyTensor(tensors.norm_out);
}

void FreeDeviceMemory(Tensors& tensors)
{
    aclrtFree(tensors.x_addr);
    aclrtFree(tensors.phi_addr);
    aclrtFree(tensors.alpha_addr);
    aclrtFree(tensors.bias_addr);
    aclrtFree(tensors.premix_addr);
    aclrtFree(tensors.hin_addr);
    aclrtFree(tensors.h_post_addr);
    aclrtFree(tensors.h_res_addr);
    aclrtFree(tensors.h_pre_addr);
    aclrtFree(tensors.hc_before_norm_addr);
    aclrtFree(tensors.inv_rms_addr);
    aclrtFree(tensors.sum_out_addr);
    aclrtFree(tensors.norm_out_addr);
}

int main()
{
    int32_t device_id = 0;
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    Tensors tensors;

    int64_t bs = 2, seq_len = 128, n = 4, c = 4096;
    int64_t hc_mult = n;
    int64_t hc_mix = n * n + 2 * n;
    int64_t num_iters = 20;
    float hc_eps = 1e-6f, norm_eps = 1e-6f;
    bool need_backward = true;
    bool enable_hf32 = true;

    std::vector<int64_t> x_shape = {bs, seq_len, n, c};
    std::vector<int64_t> phi_shape = {hc_mix, n * c};
    std::vector<int64_t> alpha_shape = {3};
    std::vector<int64_t> bias_shape = {hc_mix};
    std::vector<int64_t> premix_shape = {bs, seq_len, n};

    std::vector<int64_t> hin_shape = {bs, seq_len, c};
    std::vector<int64_t> h_post_shape = {bs, seq_len, n};
    std::vector<int64_t> h_res_shape = {bs, seq_len, n * n};
    std::vector<int64_t> h_pre_shape = {bs, seq_len, n};
    std::vector<int64_t> hc_before_norm_shape = {bs, seq_len, hc_mix};
    std::vector<int64_t> inv_rms_shape = {bs, seq_len, 1};
    std::vector<int64_t> sum_out_shape = {num_iters * 2, bs, seq_len, n};
    std::vector<int64_t> norm_out_shape = {num_iters * 2, bs, seq_len, n, n};

    int ret = InitAcl(device_id, context, stream);
    CHECK_RET(ret == 0, LOG_PRINT("InitAcl failed, error: %d\n", ret); return -1);

    ret = CreateInputTensors(x_shape, phi_shape, alpha_shape, bias_shape, premix_shape, tensors);
    CHECK_RET(ret == 0, return -1);

    ret = CreateOutputTensors(hin_shape, h_post_shape, h_res_shape, h_pre_shape, hc_before_norm_shape, inv_rms_shape,
                              sum_out_shape, norm_out_shape, tensors);
    CHECK_RET(ret == 0, return -1);

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;

    // V2 interface: 5 tensor inputs (including optional premix) + 6 attrs (including enable_hf32), 8 outputs.
    // When premix is non-null, hin uses premix weighting (replacing the internal sinkhorn weight h_pre);
    // when premix is nullptr, the behavior is the same as V1 (passing nullptr is sufficient; this example demonstrates premix usage).
    aclnnStatus aclnn_ret = aclnnMhcPreSinkhornV2GetWorkspaceSize(
        tensors.x, tensors.phi, tensors.alpha, tensors.bias, tensors.premix, hc_mult, num_iters, hc_eps, norm_eps,
        need_backward, enable_hf32, tensors.hin, tensors.h_post, tensors.h_res, tensors.h_pre, tensors.hc_before_norm,
        tensors.inv_rms, tensors.sum_out, tensors.norm_out, &workspace_size, &executor);
    CHECK_RET(aclnn_ret == ACL_SUCCESS,
              LOG_PRINT("aclnnMhcPreSinkhornV2GetWorkspaceSize failed, error: %d\n", aclnn_ret);
              FreeDeviceMemory(tensors); DestroyTensors(tensors); return -1);

    void* workspace_addr = nullptr;
    if (workspace_size > 0) {
        aclError acl_ret = aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(acl_ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed, error: %d\n", acl_ret);
                  FreeDeviceMemory(tensors); DestroyTensors(tensors); return -1);
    }

    aclnn_ret = aclnnMhcPreSinkhornV2(workspace_addr, workspace_size, executor, stream);
    CHECK_RET(aclnn_ret == ACL_SUCCESS, LOG_PRINT("aclnnMhcPreSinkhornV2 failed, error: %d\n", aclnn_ret);
              FreeDeviceMemory(tensors); DestroyTensors(tensors); return -1);

    aclError acl_ret = aclrtSynchronizeStream(stream);
    CHECK_RET(acl_ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed, error: %d\n", acl_ret);
              FreeDeviceMemory(tensors); DestroyTensors(tensors); return -1);

    PrintTensorDataBfloat16(hin_shape, tensors.hin_addr);
    PrintTensorDataFloat(h_post_shape, tensors.h_post_addr);
    PrintTensorDataFloat(h_res_shape, tensors.h_res_addr);
    PrintTensorDataFloat(h_pre_shape, tensors.h_pre_addr);

    if (workspace_size > 0) {
        (void)aclrtFree(workspace_addr);
    }
    FreeDeviceMemory(tensors);
    DestroyTensors(tensors);

    acl_ret = aclrtDestroyStream(stream);
    CHECK_RET(acl_ret == ACL_SUCCESS, LOG_PRINT("destroy stream failed, error: %d\n", acl_ret); return -1);
    acl_ret = aclrtDestroyContext(context);
    CHECK_RET(acl_ret == ACL_SUCCESS, LOG_PRINT("destroy context failed, error: %d\n", acl_ret); return -1);
    (void)aclFinalize();

    return 0;
}
```
