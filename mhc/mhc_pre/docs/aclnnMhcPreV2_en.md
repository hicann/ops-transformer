# aclnnMhcPreV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T14:48:37.029Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
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

- Description: Obtains the projection matrices $H^{res}$ and $H^{post}$ for the hidden layer and the layer input matrix $h^{in}$ for the Attention or MLP layer in the MHC architecture through a series of computations.

- Compared with the [aclnnMhcPre](aclnnMhcPre_en.md) API, this API adds the `opImplMode` parameter to specify the computation mode of the MhcPre operator:
  - `opImplMode=0`: Performs FP32 mode computation in the Cube.
  - `opImplMode=1`: Performs HF32 mode computation in the Cube.

- Formula:

  where xFlat denotes treating the last two dimensions n and D of x as a vector of length nD, gammaFlat denotes treating gamma as a vector of length nD, @ denotes matrix multiplication, and $\odot$ denotes element-wise multiplication.

  $$
  \begin{aligned}
  invRms &= \left(mean(xFlat^{2}) + normEps\right)^{-\frac{1}{2}}\\
  xGamma &= \begin{cases}
      xFlat \odot gammaFlat, & gamma \ne null \\
      xFlat, & gamma = null
  \end{cases}\\
  hMix &= xGamma @ phi^{T}\\
  w &= hMix \odot invRms\\
  [pPre, pPost, pRes] &= \begin{cases}
      split(w, [n, n, n^{2}]), & alpha.shape=[3] \\
      [split(w, [n, n]), 0], & alpha.shape=[2]
  \end{cases}\\
  hPre &= \sigma(pPre \odot alpha0 + bias0) + hcEps\\
  hPost &= \begin{cases}
      2\sigma(pPost \odot alpha1 + bias1), & alpha.shape=[3] \\
      2\sigma(pPost \odot alpha1 + bias1) + hcEps, & alpha.shape=[2]
  \end{cases}\\
  hRes &= \begin{cases}
      pRes \odot alpha2 + bias2, & alpha.shape=[3] \\
      0, & alpha.shape=[2]
  \end{cases}\\
  hIn_{d} &= \sum_{i=0}^{n-1} hPre_{i} x_{i,d}
  \end{aligned}
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnMhcPreV2GetWorkspaceSize" first to obtain the workspace size required for computation and the executor that contains the operator computation process. Then call "aclnnMhcPreV2" to perform the computation.

```Cpp
aclnnStatus aclnnMhcPreV2GetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *phi,
    const aclTensor *alpha,
    const aclTensor *bias,
    const aclTensor *gammaOptional,
    double           normEps,
    double           hcEps,
    int64_t          opImplMode,
    aclTensor       *hIn,
    aclTensor       *hPost,
    aclTensor       *hRes,
    aclTensor       *invRmsOptional,
    aclTensor       *hMixOptional,
    aclTensor       *hPreOptional,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnMhcPreV2(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnMhcPreV2GetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 187px">
  <col style="width: 121px">
  <col style="width: 287px">
  <col style="width: 387px">
  <col style="width: 187px">
  <col style="width: 187px">
  <col style="width: 187px">
  <col style="width: 146px">
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
      <td>x</td>
      <td>Input</td>
      <td>Data to be computed. It indicates the layer input of the mHC layer in the network, corresponding to x in the formula.</td>
      <td>Must not be an empty tensor.</td>
      <td>BFLOAT16 or FLOAT16</td>
      <td>ND</td>
      <td>[B,S,n,D] or [T,n,D]</td>
      <td>√</td>
  </tr>
  <tr>
      <td>phi</td>
      <td>Input</td>
      <td>Parameter matrix of mHC, corresponding to phi in the formula.</td>
      <td>Must not be an empty tensor.</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>[n^2+2n, nD] or [2n, nD]</td>
      <td>√</td>
  </tr>
  <tr>
      <td>alpha</td>
      <td>Input</td>
      <td>Scaling parameter of mHC, corresponding to alpha in the formula.</td>
      <td>Must not be an empty tensor.</td>
      <td>FLOAT32</td>
      <td>-</td>
      <td>[3] or [2]</td>
      <td>-</td>
  </tr>
  <tr>
      <td>bias</td>
          <td>Input</td>
      <td>The bias parameter of mHC, corresponding to bias in the corresponding formula.</td>
      <td>The tensor must not be empty.</td>
      <td>FLOAT32</td>
      <td>-</td>
      <td>[n^2+2n] or [2n]</td>
      <td>-</td>
  </tr>
  <tr>
      <td>gammaOptional</td>
      <td>Optional input</td>
      <td>Scale factor for the RmsNorm computation, corresponding to gamma in the formula.</td>
      <td>For the inference use case, a null pointer can be passed.</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>[n, D]</td>
      <td>√</td>
  </tr>
  <tr>
      <td>normEps</td>
      <td>Optional input</td>
      <td>The divide-by-zero prevention parameter of RmsNorm, corresponding to normEps in the formula.</td>
      <td>Recommended value: 1e-6.</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  <tr>
      <td>hcEps</td>
      <td>Optional input</td>
      <td>The eps parameter of h_pre after sigmoid, and the eps parameter of h_post after sigmoid when alpha.shape=[2], corresponding to hcEps in the formula.</td>
      <td>Recommended value: 1e-6.</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  <tr>
      <td>opImplMode</td>
      <td>Input</td>
      <td>Specifies the computation mode of the MhcPre operator.</td>
      <td>Supported enumeration values:<ul>
        <li>0: Recommended value, FP32 mode computation in the Cube.</li>
        <li>1: HF32 mode computation in the Cube.</li>
      </ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  <tr>
      <td>hIn</td>
      <td>Output</td>
      <td>The output h_in is used as the Attention/MLP layer input, corresponding to hIn in the formula.</td>
      <td>Must not be an empty tensor.</td>
      <td>BFLOAT16 or FLOAT16</td>
      <td>ND</td>
      <td>[B,S,D] or [T,D]</td>
      <td>-</td>
  </tr>
  <tr>
      <td>hPost</td>
      <td>Output</td>
      <td>The h_post transformation matrix of mHC in the output, corresponding to hPost in the formula.</td>
      <td>Must not be an empty tensor.</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>[B,S,n] or [T,n]</td>
      <td>-</td>
  </tr>
  <tr>
      <td>hRes</td>
      <td>Output</td>
      <td>The h_res transformation matrix of mHC in the output (without sinkhorn transformation), corresponding to hRes in the formula.</td>
      <td>Must not be an empty tensor when alpha=[3]; can be a null pointer when alpha=[2].</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>[B,S,n,n] or [T,n,n]</td>
      <td>-</td>
  </tr>
  <tr>
      <td>invRmsOptional</td>
      <td>Optional output</td>
      <td>1/r computed by RmsNorm, corresponding to invRms in the formula.</td>
      <td>-</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>[B,S] or [T]</td>
      <td>-</td>
  </tr>
  <tr>
      <td>hMixOptional</td>
      <td>Optional output</td>
      <td>Matrix multiplication result of xGamma and phi, which is hMix in the corresponding formula.</td>
      <td>-</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>[B,S,n^2+2n] or [T,n^2+2n] or [B,S,2n] or [T, 2n]</td>
      <td>-</td>
  </tr>
  <tr>
      <td>hPreOptional</td>
      <td>optional output</td>
      <td>The h_pre matrix after sigmoid computation, corresponding to hPre in the formula.</td>
      <td>-</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>[B,S,n] or [T,n]</td>
      <td>-</td>
  </tr>
  <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  </tbody></table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 330px">
  <col style="width: 140px">
  <col style="width: 762px">
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
    <td> ACLNN_ERR_PARAM_NULLPTR </td>
    <td> 161001 </td>
    <td>x, phi, alpha, bias, hIn, hPost, or hRes is a null pointer.</td>
    </tr>
    <tr>
    <td rowspan="3"> ACLNN_ERR_PARAM_INVALID </td>
    <td rowspan="3"> 161002 </td>
    <td>The data types of x, phi, alpha, bias, gammaOptional, hIn, hPost, hRes, invRmsOptional, hMixOptional, and hPreOptional are not supported.</td>
    </tr>
    <tr>
    <td>The shape dimensions of x, phi, alpha, bias, gammaOptional, hIn, hPost, hRes, invRmsOptional, hMixOptional, and hPreOptional are not supported.</td>
    </tr>
    <tr>
    <td>opImplMode is not 0 or 1.</td>
    </tr>
    <tr>
    <td> ACLNN_ERR_RUNTIME_ERROR </td>
    <td> 361001 </td>
    <td>The internal API call to the npu runtime interface is abnormal.</td>
    </tr>
  </tbody></table>

## aclnnMhcPreV2

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1148px"><colgroup>
  <col style="width: 170px">
  <col style="width: 134px">
  <col style="width: 844px">
  </colgroup>
  <thead>
  <tr>
    <th>Parameter</th>
    <th>Input/Output</th>
    <th>Description</th>
  </tr>
  </thead>
  <tbody>
  <tr>
    <td>workspace</td>
    <td>Input</td>
    <td>Workspace memory address allocated on the Device.</td>
  </tr>
  <tr>
    <td>workspaceSize</td>
    <td>Input</td>
    <td>Size of the workspace allocated on the Device, obtained by the first-phase API aclnnMhcPreV2GetWorkspaceSize.</td>
  </tr>
  <tr>
    <td>executor</td>
    <td>Input</td>
    <td>Executor of the op, containing the operator computation flow.</td>
  </tr>
  <tr>
    <td>stream</td>
    <td>Input</td>
    <td>Stream for task execution.</td>
  </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnMhcPreV2 defaults to a deterministic implementation.

- Batch consistency:
  - Defaults to a non-Batch-consistent implementation. Enabling Batch consistency through aclrtSetSysParamOpt is not supported.

- Specification constraints:
  - Currently, n supports 4, 6, and 8.
  - D supports values in the range 16-16384. D must be aligned to 16.
  - When alpha=[3], the hRes output is supported. The following conditions must be met: inputs phi=[n^2+2n, nD], bias=[n^2+2n], and output hMixOptional=[B,S,n^2+2n] or [T,n^2+2n]. When alpha=[2], the hRes output is 0. The following conditions must be met: inputs phi=[2n, nD], bias=[2n], and output hMixOptional=[B,S,2n] or [T,2n].
  - The optional outputs invRmsOptional, hMixOptional, and hPreOptional are mutually dependent. They must be output together or not output at all. Returning only part of them is not supported. Output condition: invRmsOptional!=nullptr && hMixOptional!=nullptr && hPreOptional!=nullptr.

## Example

The following sample code is for reference only. For details about the compilation and execution procedure, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_mhc_pre_v2.h"

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

// Calculate the total number of elements based on the tensor shape.
int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t size = 1;
    for (int64_t dim : shape) {
        size *= dim;
    }
    return size;
}

// Copy the device-side tensor data to the host side and print it (float type).
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

// Copy the device-side tensor data to the host side and print it (float16 type).
void PrintTensorDataFloat16(const std::vector<int64_t> &shape, void *device_addr)
{
    int64_t size = GetShapeSize(shape);
    std::vector<aclFloat16> host_fp16(size);
    std::vector<float> host_data(size, 0.0f);

    aclError ret = aclrtMemcpy(host_fp16.data(), size * sizeof(aclFloat16), device_addr, size * sizeof(aclFloat16),
                               ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Memcpy device to host failed, error: %d\n", ret); return);

    for (int64_t i = 0; i < size; ++i) {
        host_data[i] = aclFloat16ToFloat(host_fp16[i]);
    }

    LOG_PRINT("Tensor data (first 10 elements): ");
    for (int64_t i = 0; i < std::min((int64_t)10, size); ++i) {
        LOG_PRINT("%f ", host_data[i]);
    }
    LOG_PRINT("\n");
}

// Initialize the AscendCL environment (Device/Context/Stream).
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

// Create a FLOAT32 Device-side aclTensor (including data copy).
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

// Create a FLOAT16 Device-side aclTensor (including data copy).
int CreateAclTensorFloat16(const std::vector<float> &host_data, const std::vector<int64_t> &shape, void *&device_addr,
                           aclTensor *&tensor)
{
    int64_t size = GetShapeSize(shape);
    std::vector<aclFloat16> host_data_fp16(size);
    for (int64_t i = 0; i < size; ++i) {
        host_data_fp16[i] = aclFloat16(host_data[i]);
    }

    int64_t byte_size = size * sizeof(aclFloat16);

    aclError ret = aclrtMalloc(&device_addr, byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed, error: %d\n", ret); return -1);

    ret = aclrtMemcpy(device_addr, byte_size, host_data_fp16.data(), byte_size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed, error: %d\n", ret); return -1);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT16, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                             shape.size(), device_addr);
    CHECK_RET(tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return -1);

    return 0;
}

// Create a FLOAT16 output aclTensor (memory allocation only).
int CreateAclTensorFloat16Output(const std::vector<int64_t> &shape, void *&device_addr, aclTensor *&tensor)
{
    int64_t size = GetShapeSize(shape);
    int64_t byte_size = size * sizeof(aclFloat16);

    aclError ret = aclrtMalloc(&device_addr, byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed, error: %d\n", ret); return -1);

    tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, shape.data(),
                             shape.size(), device_addr);
    CHECK_RET(tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return -1);

    return 0;
}

// Create a FLOAT32 output aclTensor (memory allocation only).
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
    void *x_addr = nullptr, *phi_addr = nullptr, *alpha_addr = nullptr, *bias_addr = nullptr, *gamma_addr = nullptr;
    void *hin_addr = nullptr, *h_post_addr = nullptr, *h_res_addr = nullptr, *inv_rms_addr = nullptr;
    void *h_mix_addr = nullptr, *h_pre_addr = nullptr;
    aclTensor *x = nullptr, *phi = nullptr, *alpha = nullptr, *bias = nullptr, *gamma = nullptr;
    aclTensor *hin = nullptr, *h_post = nullptr, *h_res = nullptr, *inv_rms = nullptr;
    aclTensor *h_mix = nullptr, *h_pre = nullptr;
};

int CreateInputTensors(const std::vector<int64_t> &x_shape, const std::vector<int64_t> &phi_shape,
                       const std::vector<int64_t> &alpha_shape, const std::vector<int64_t> &bias_shape,
                       const std::vector<int64_t> &gamma_shape, Tensors &tensors)
{
    std::vector<float> x_host_data(GetShapeSize(x_shape), 1.0f);
    std::vector<float> phi_host_data(GetShapeSize(phi_shape), 1.0f);
    std::vector<float> alpha_host_data(3, 1.0f);
    std::vector<float> bias_host_data(GetShapeSize(bias_shape), 1.0f);
    std::vector<float> gamma_host_data(GetShapeSize(gamma_shape), 1.0f);

    int ret = CreateAclTensorFloat16(x_host_data, x_shape, tensors.x_addr, tensors.x);
    CHECK_RET(ret == 0, LOG_PRINT("Create x_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(phi_host_data, phi_shape, tensors.phi_addr, tensors.phi);
    CHECK_RET(ret == 0, LOG_PRINT("Create phi_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(alpha_host_data, alpha_shape, tensors.alpha_addr, tensors.alpha);
    CHECK_RET(ret == 0, LOG_PRINT("Create alpha_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(bias_host_data, bias_shape, tensors.bias_addr, tensors.bias);
    CHECK_RET(ret == 0, LOG_PRINT("Create bias_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(gamma_host_data, gamma_shape, tensors.gamma_addr, tensors.gamma);
    CHECK_RET(ret == 0, LOG_PRINT("Create gamma_tensor failed\n"); return -1);
    return 0;
}

int CreateOutputTensors(const std::vector<int64_t> &hin_shape, const std::vector<int64_t> &h_post_shape,
                        const std::vector<int64_t> &h_res_shape, const std::vector<int64_t> &inv_rms_shape,
                        const std::vector<int64_t> &h_mix_shape, const std::vector<int64_t> &h_pre_shape,
                        Tensors &tensors)
{
    int ret = CreateAclTensorFloat16Output(hin_shape, tensors.hin_addr, tensors.hin);
    CHECK_RET(ret == 0, LOG_PRINT("Create hin_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(h_post_shape, tensors.h_post_addr, tensors.h_post);
    CHECK_RET(ret == 0, LOG_PRINT("Create h_post_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(h_res_shape, tensors.h_res_addr, tensors.h_res);
    CHECK_RET(ret == 0, LOG_PRINT("Create h_res_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(inv_rms_shape, tensors.inv_rms_addr, tensors.inv_rms);
    CHECK_RET(ret == 0, LOG_PRINT("Create inv_rms_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(h_mix_shape, tensors.h_mix_addr, tensors.h_mix);
    CHECK_RET(ret == 0, LOG_PRINT("Create h_mix_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(h_pre_shape, tensors.h_pre_addr, tensors.h_pre);
    CHECK_RET(ret == 0, LOG_PRINT("Create h_pre_tensor failed\n"); return -1);
    return 0;
}

void DestroyTensors(Tensors &tensors)
{
    aclDestroyTensor(tensors.x);
    aclDestroyTensor(tensors.phi);
    aclDestroyTensor(tensors.alpha);
    aclDestroyTensor(tensors.bias);
    aclDestroyTensor(tensors.gamma);
    aclDestroyTensor(tensors.hin);
    aclDestroyTensor(tensors.h_post);
    aclDestroyTensor(tensors.h_res);
    aclDestroyTensor(tensors.inv_rms);
    aclDestroyTensor(tensors.h_mix);
    aclDestroyTensor(tensors.h_pre);
}

void FreeDeviceMemory(Tensors &tensors)
{
    aclrtFree(tensors.x_addr);
    aclrtFree(tensors.phi_addr);
    aclrtFree(tensors.alpha_addr);
    aclrtFree(tensors.bias_addr);
    aclrtFree(tensors.gamma_addr);
    aclrtFree(tensors.hin_addr);
    aclrtFree(tensors.h_post_addr);
    aclrtFree(tensors.h_res_addr);
    aclrtFree(tensors.inv_rms_addr);
    aclrtFree(tensors.h_mix_addr);
    aclrtFree(tensors.h_pre_addr);
}

int main()
{
    int32_t device_id = 0;
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    Tensors tensors;
    int B = 1, S = 2048, n = 4, D = 2560;
    std::vector<int64_t> x_shape = {B * S, n, D}, phi_shape = {n * n + 2 * n, n * D}, alpha_shape = {3},
                         bias_shape = {n * n + 2 * n}, gamma_shape = {n, D};
    std::vector<int64_t> hin_shape = {B * S, D}, h_post_shape = {B * S, n}, h_res_shape = {B * S, n, n},
                         inv_rms_shape = {B * S}, h_mix_shape = {B * S, n * n + 2 * n}, h_pre_shape = {B * S, n};
    int ret = InitAcl(device_id, context, stream);
    CHECK_RET(ret == 0, LOG_PRINT("InitAcl failed, error: %d\n", ret); return -1);
    ret = CreateInputTensors(x_shape, phi_shape, alpha_shape, bias_shape, gamma_shape, tensors);
    CHECK_RET(ret == 0, return -1);
    ret = CreateOutputTensors(hin_shape, h_post_shape, h_res_shape, inv_rms_shape, h_mix_shape, h_pre_shape, tensors);
    CHECK_RET(ret == 0, return -1);
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    int64_t opImplMode = 1;
    aclnnStatus aclnn_ret = aclnnMhcPreV2GetWorkspaceSize(
        tensors.x, tensors.phi, tensors.alpha, tensors.bias, tensors.gamma, 1e-6, 1e-6, opImplMode, tensors.hin,
        tensors.h_post, tensors.h_res, tensors.inv_rms, tensors.h_mix, tensors.h_pre, &workspace_size, &executor);
    CHECK_RET(aclnn_ret == ACL_SUCCESS, LOG_PRINT("aclnnMhcPreV2GetWorkspaceSize failed, error: %d\n", aclnn_ret);
              return -1);
    void *workspace_addr = nullptr;
    if (workspace_size > 0) {
        ret = aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc workspace failed, error: %d\n", ret); return -1);
    }
    aclnn_ret = aclnnMhcPreV2(workspace_addr, workspace_size, executor, stream);
    CHECK_RET(aclnn_ret == ACL_SUCCESS, LOG_PRINT("aclnnMhcPreV2 failed, error: %d\n", aclnn_ret); return -1);
    CHECK_RET(aclrtSynchronizeStream(stream) == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed\n"); return -1);
    LOG_PRINT("MhcPre compute success!\nOutput tensor data: \n");
    PrintTensorDataFloat16(hin_shape, tensors.hin_addr);
    PrintTensorDataFloat(h_post_shape, tensors.h_post_addr);
    PrintTensorDataFloat(h_res_shape, tensors.h_res_addr);
    PrintTensorDataFloat(inv_rms_shape, tensors.inv_rms_addr);
    PrintTensorDataFloat(h_mix_shape, tensors.h_mix_addr);
    PrintTensorDataFloat(h_pre_shape, tensors.h_pre_addr);
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
