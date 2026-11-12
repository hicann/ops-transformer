# aclnnMhcPostBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T14:46:29.224Z -->

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

- Description: mhc_post performs Post Mapping on the previous layer output $h_{t}^{out}$ in the MHC (Manifold-Constrained Hyper-Connection) architecture through a series of computations, performs ResMapping on the previous layer input $x_l$, and then performs residual connection on the two to obtain the next layer input $x_{l+1}$. This operator implements the backward function of the preceding process.

- Formula:

  When hRes is passed (hRes is not nullptr):
  $$
  grad\_x = H_{l}^{res} \times grad\_output\\
  grad\_h\_res = x_{l} \times {grad\_output}^{T}
  $$
  $$
  grad\_h\_out=({grad\_output} * (H_{l}^{post}.unsqueeze(-1))).sum(dim=-2)\\
  grad\_h\_post=({grad\_output} * (h_{l}^{out}.unsqueeze(-2))).sum(dim=-1)
  $$

  When hRes is not passed (hRes is nullptr):
  $$
  grad\_x = grad\_output
  $$
  $$
  grad\_h\_res = x_{l} \times {grad\_output}^{T}
  $$
  $$
  grad\_h\_out=({grad\_output} * (H_{l}^{post}.unsqueeze(-1))).sum(dim=-2)\\
  grad\_h\_post=({grad\_output} * (h_{l}^{out}.unsqueeze(-2))).sum(dim=-1)
  $$

  > NOTE: hRes as an optional parameter (can be set to nullptr) is supported only on <term>Ascend 950PR&950DT products</term>.

## Function Prototype

The operator execution APIs are the [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnMhcPostBackwardGetWorkspaceSize" first to obtain the input parameters and calculate the required workspace size based on the computation flow, and then call "aclnnMhcPostBackward" to perform the computation.

```c++
aclnnStatus aclnnMhcPostBackwardGetWorkspaceSize(
    const aclTensor     *gradOutput,
    const aclTensor     *x,
    const aclTensor     *hRes,
    const aclTensor     *hOut,
    const aclTensor     *hPost,
    aclTensor           *gradX,
    aclTensor           *gradHRes,
    aclTensor           *gradHOut,
    aclTensor           *gradHPost,
    uint64_t            *workspaceSize,
    aclOpExecutor       **executor)
```

```c++
aclnnStatus aclnnMhcPostBackward(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnMhcPostBackwardGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width:  1400px"><colgroup>
  <col style="width: 145px">
  <col style="width: 90px">
  <col style="width: 441px">
  <col style="width: 158px">
  <col style="width: 186px">
  <col style="width: 80px">
  <col style="width: 155px">
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
      <th>non-contiguous tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>gradOutput</td>
      <td>Input</td>
      <td>Data to be computed, indicating the input data of the MHC layer in the network.</td>
      <td>        <ul>
          <li>Empty tensor not supported</li>
        </ul>
      </td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>[B,S,N,D], [T,N,D]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Data to be computed, indicating the input data of the MHC layer in the network.</td>
      <td>        <ul>
          <li>Empty tensor not supported.</li>
        </ul>
      </td>
      <td>Data type consistent with gradOutput</td>
      <td>ND</td>
      <td>[B,S,N,D], [T,N,D]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hRes</td>
      <td>Input</td>
      <td>hRes transformation matrix of MHC.</td>
      <td>        <ul>
          <li>Optional parameter. Set to nullptr if not passed.</li>
          <li>Empty tensor not supported.</li>
        </ul>
      </td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>[B,S,N,N], [T,N,N]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hOut</td>
      <td>Input</td>
      <td>Output of the Atten/MLP layer.</td>
      <td>        <ul>
          <li>Empty tensor not supported</li>
        </ul>
      </td>
      <td>Data type consistent with gradOutput.</td>
      <td>ND</td>
      <td>[B,S,D], [T,D]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hPost</td>
      <td>Input</td>
      <td>hPost transformation matrix of MHC.</td>
      <td>        <ul>
          <li>Empty tensor not supported</li>
        </ul>
      </td>
      <td>Data type consistent with hRes</td>
      <td>ND</td>
      <td>[B,S,N] and [T,N]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradX</td>
      <td>Output</td>
      <td>Gradient of the MHC layer input data x in the network.</td>
      <td>-</td>
      <td>Data type is consistent with gradOutput.</td>
      <td>ND</td>
      <td>[B,S,N,D] or [T,N,D]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradHRes</td>
      <td>Output</td>
      <td>Gradient of the MHC layer input data hRes in the network.</td>
      <td>-</td>
      <td>Data type consistent with hRes</td>
      <td>ND</td>
      <td>[B,S,N,N], [T,N,N]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradHOut</td>
      <td>Output</td>
      <td>Gradient of hOut, the layer input data of the MHC layer in the network.</td>
      <td>-</td>
      <td>Data type consistent with hOut</td>
      <td>ND</td>
      <td>[B,S,D], [T,D]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradHPost</td>
      <td>Output</td>
      <td>Gradient of the MHC layer input data hPost in the network.</td>
      <td>-</td>
      <td>Data type consistent with hPost</td>
      <td>ND</td>
      <td>[B,S,N], [T,N]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be allocated on the Device.</td>
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

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 1000px"><colgroup>
    <col style="width: 300px">
    <col style="width: 150px">
    <col style="width: 550px">
    </colgroup>
    <thead>
        <th>Return value</th>
        <th>Error code</th>
        <th>Description</th>
    </thead>
    <tbody>
        <tr>
          <td>ACLNN_ERR_PARAM_NULLPTR</td>
          <td>161001</td>
          <td>gradOutput, x, hRes, hOut, or hPost is a null pointer.</td>
        </tr>
        <tr>
          <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
          <td rowspan="3">161002</td>
          <td>The data type of gradOutput, x, hRes, hOut, or hPost is not supported.</td>
        </tr>
          <tr>
          <td>The shape dimension or value of gradOutput, x, hRes, hOut, or hPost is not supported.</td>
        </tr>
        <tr>
          <td>The data type or shape of gradOutput, x, hRes, hOut, or hPost does not match.</td>
        </tr>
  </tbody></table>

## aclnnMhcPostBackward

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 598px"><colgroup>
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
        <td>Workspace memory address requested on the Device.</td>
        </tr>
        <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Workspace size requested on the Device, obtained by the first-phase API aclnnMhcPostBackwardGetWorkspaceSize.</td>
        </tr>
        <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Operator executor that contains the operator computation flow.</td>
        </tr>
        <tr>
        <td>stream</td>
        <td>Input</td>
        <td>AscendCL stream used to execute tasks.</td>
        </tr>
    </tbody>
  </table>

- **Return value**

  Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

<!-- npu="A3,910b" id7 -->
- <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: In the parameter description, the value of dimension N currently only supports 4. The value of dimension D must be aligned to 128, and the value range is [1,100000].
<!-- end id7 -->
<!-- npu="950" id8 -->
- <term>Ascend 950PR&950DT products</term>: In the parameter description, the value of dimension N currently only supports 4, 6, and 8.

<!-- end id8 -->

## Example

The sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_mhc_post_backward.h"
#include "securec.h"

using namespace std;

namespace {

#define CHECK_RET(cond) ((cond) ? true :(false))

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream) {
    // Fixed writing method, AscendCL initialization.
    auto ret = aclInit(nullptr);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtSetDevice(deviceId);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtCreateStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        return ret;
    }
    return 0;
}


template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to request device side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
        return ret;
    }
    // Call aclrtMemcpy to copy host side data to device side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        return ret;
    }

    // Calculate the strides of continuous tensors.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor interface to create aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

} // namespace

int main() {
    // 1. (Fixed writing method)  device/stream initialization. Refer to AscendCL's list of external interfaces.
    // Fill in the deviceId based on your actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    LOG_PRINT("ACL environment initialized successfully!\n");

    // 2. To construct input and output, it is necessary to customize the construction according to the API interface.
    // Example: BSND format (B, S, n, D)
    std::vector<int64_t> gradYShape = {1, 1024, 4, 5120};   // BSND
    std::vector<int64_t> xShape = {1, 1024, 4, 5120};   // BSND
    std::vector<int64_t> hResShape = {1, 1024, 4, 4};   // BSND
    std::vector<int64_t> hOutShape = {1, 1024, 5120};   // BSD
    std::vector<int64_t> hPostShape = {1, 1024, 4};     // BSn
    std::vector<int64_t> gradXShape = {1, 1024, 4, 5120};   // BSND
    std::vector<int64_t> gradHResShape = {1, 1024, 4, 4};   // BSND
    std::vector<int64_t> gradHOutShape = {1, 1024, 5120};   // BSD
    std::vector<int64_t> gradHPostShape = {1, 1024, 4};     // BSn

    void *gradYDeviceAddr = nullptr;
    void *xDeviceAddr = nullptr;
    void *hResDeviceAddr = nullptr;
    void *hOutDeviceAddr = nullptr;
    void *hPostDeviceAddr = nullptr;
    void *gradXDeviceAddr = nullptr;
    void *gradHResDeviceAddr = nullptr;
    void *gradHOutDeviceAddr = nullptr;
    void *gradHPostDeviceAddr = nullptr;

    aclTensor *gradYTensor = nullptr;
    aclTensor *xTensor = nullptr;
    aclTensor *hResTensor = nullptr;
    aclTensor *hOutTensor = nullptr;
    aclTensor *hPostTensor = nullptr;
    aclTensor *gradXTensor = nullptr;
    aclTensor *gradHResTensor = nullptr;
    aclTensor *gradHOutTensor = nullptr;
    aclTensor *gradHPostTensor = nullptr;

    int64_t gradYShapeSize = GetShapeSize(gradYShape);
    int64_t xShapeSize = GetShapeSize(xShape);
    int64_t hResShapeSize = GetShapeSize(hResShape);
    int64_t hOutShapeSize = GetShapeSize(hOutShape);
    int64_t hPostShapeSize = GetShapeSize(hPostShape);
    int64_t gradXShapeSize = GetShapeSize(gradXShape);
    int64_t gradHResShapeSize = GetShapeSize(gradHResShape);
    int64_t gradHOutShapeSize = GetShapeSize(gradHOutShape);
    int64_t gradHPostShapeSize = GetShapeSize(gradHPostShape);
    std::vector<aclFloat16> gradYHostData(gradYShapeSize, aclFloatToFloat16(1.0f));
    std::vector<aclFloat16> xHostData(xShapeSize, aclFloatToFloat16(1.0f));
    std::vector<float> hResHostData(hResShapeSize, 1.0f);
    std::vector<aclFloat16> hOutHostData(hOutShapeSize, aclFloatToFloat16(1.0f));
    std::vector<float> hPostHostData(hPostShapeSize, 1.0f);
    std::vector<aclFloat16> gradXHostData(gradXShapeSize, aclFloatToFloat16(1.0f));
    std::vector<float> gradHResHostData(gradHResShapeSize, 1.0f);
    std::vector<aclFloat16> gradHOutHostData(gradHOutShapeSize, aclFloatToFloat16(1.0f));
    std::vector<float> gradHPostHostData(gradHPostShapeSize, 1.0f);

    ret = CreateAclTensor(gradYHostData, gradYShape, &gradYDeviceAddr, aclDataType::ACL_FLOAT16, &gradYTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &xTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create hRes aclTensor.
    ret = CreateAclTensor(hResHostData, hResShape, &hResDeviceAddr, aclDataType::ACL_FLOAT, &hResTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create hOut aclTensor.
    ret = CreateAclTensor(hOutHostData, hOutShape, &hOutDeviceAddr, aclDataType::ACL_FLOAT16, &hOutTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create hPost aclTensor.
    ret = CreateAclTensor(hPostHostData, hPostShape, &hPostDeviceAddr, aclDataType::ACL_FLOAT, &hPostTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    // Create gradX aclTensor.
    ret = CreateAclTensor(gradXHostData, gradXShape, &gradXDeviceAddr, aclDataType::ACL_FLOAT16, &gradXTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create gradHRes aclTensor.
    ret = CreateAclTensor(gradHResHostData, gradHResShape, &gradHResDeviceAddr, aclDataType::ACL_FLOAT, &gradHResTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create gradHOut aclTensor.
    ret = CreateAclTensor(gradHOutHostData, gradHOutShape, &gradHOutDeviceAddr, aclDataType::ACL_FLOAT16, &gradHOutTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create gradHPost aclTensor.
    ret = CreateAclTensor(gradHPostHostData, gradHPostShape, &gradHPostDeviceAddr, aclDataType::ACL_FLOAT, &gradHPostTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    // 3. Call CANN operator library API.
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // Call the first interface.
    ret = aclnnMhcPostBackwardGetWorkspaceSize(
        gradYTensor, xTensor, hResTensor, hOutTensor, hPostTensor, gradXTensor, gradHResTensor, gradHOutTensor,
        gradHPostTensor, &workspaceSize, &executor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnMhcPostBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }
    // Allocate device memory based on the workspaceSize calculated from the first interface paragraph.
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0U) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }
    // Call the second interface.
    ret = aclnnMhcPostBackward(workspaceAddr, workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnMhcPostBackward failed. ERROR: %d\n", ret);
        return ret;
    }

    // 4. (Fixed writing method) Synchronize and wait for task execution to end.
    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        return ret;
    }
    //gradX, gradHRes, gradHOut, gradHPost
    // 5. Retrieve the output value, copy the gradX from the device side memory to the host side.
    auto gradXsize = GetShapeSize(gradXShape);
    std::vector<aclFloat16> gradXData(gradXsize, aclFloatToFloat16(0.0f));
    ret = aclrtMemcpy(gradXData.data(), gradXData.size() * sizeof(gradXData[0]), gradXDeviceAddr,
                      gradXsize * sizeof(gradXData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy gradX from device to host failed. ERROR: %d\n", ret);
        return ret;
    }
    auto gradHRessize = GetShapeSize(gradHResShape);
    std::vector<float> gradHResData(gradHRessize, 0.0f);
    ret = aclrtMemcpy(gradHResData.data(), gradHResData.size() * sizeof(gradHResData[0]), gradHResDeviceAddr,
                      gradHRessize * sizeof(gradHResData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy gradHRes from device to host failed. ERROR: %d\n", ret);
        return ret;
    }
    auto gradHOutsize = GetShapeSize(gradHOutShape);
    std::vector<aclFloat16> gradHOutData(gradHOutsize, aclFloatToFloat16(0.0f));
    ret = aclrtMemcpy(gradHOutData.data(), gradHOutData.size() * sizeof(gradHOutData[0]), gradHOutDeviceAddr,
                      gradHOutsize * sizeof(gradHOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy gradHOut from device to host failed. ERROR: %d\n", ret);
        return ret;
    }
    auto gradHPostsize = GetShapeSize(gradHPostShape);
    std::vector<float> gradHPostData(gradHPostsize, 0.0f);
    ret = aclrtMemcpy(gradHPostData.data(), gradHPostData.size() * sizeof(gradHPostData[0]), gradHPostDeviceAddr,
                      gradHPostsize * sizeof(gradHPostData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy gradHPost from device to host failed. ERROR: %d\n", ret);
        return ret;
    }

    // 6. Release resources.
    aclDestroyTensor(gradYTensor);
    aclDestroyTensor(xTensor);
    aclDestroyTensor(hResTensor);
    aclDestroyTensor(hOutTensor);
    aclDestroyTensor(hPostTensor);
    aclDestroyTensor(gradXTensor);
    aclDestroyTensor(gradHResTensor);
    aclDestroyTensor(gradHOutTensor);
    aclDestroyTensor(gradHPostTensor);
    aclrtFree(gradYDeviceAddr);
    aclrtFree(xDeviceAddr);
    aclrtFree(hResDeviceAddr);
    aclrtFree(hOutDeviceAddr);
    aclrtFree(hPostDeviceAddr);
    aclrtFree(gradXDeviceAddr);
    aclrtFree(gradHResDeviceAddr);
    aclrtFree(gradHOutDeviceAddr);
    aclrtFree(gradHPostDeviceAddr);
    if (workspaceSize > 0U) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
