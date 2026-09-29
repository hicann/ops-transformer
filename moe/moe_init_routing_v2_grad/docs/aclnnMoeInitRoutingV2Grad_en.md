# aclnnMoeInitRoutingV2Grad

[📄 View Source Code](https://gitcode.com/cann/ops-transformer/tree/master/moe/moe_init_routing_v2_grad)

## Supported Products

| Product                                                        |  Supported  |
| :----------------------------------------------------------- |:-------:|
| Ascend 950PR/Ascend 950DT                            |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference products</term>                            |    ×    |
| <term>Atlas training products</term>                             |    ×    |

## Function

- **Description**: Implements the backward propagation of [aclnnMoeInitRoutingV2](../../moe_init_routing_v2/docs/aclnnMoeInitRoutingV2_en.md) to complete the weighted summation of tokens.
- **Formula**:

  $$
  gradX_i=\sum_{t=0}^{topK}gradExpandedX[expandedRowIdx[i * topK + t]]
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnMoeInitRoutingV2GradGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnMoeInitRoutingV2Grad` is called to perform computation.

```cpp
aclnnStatus aclnnMoeInitRoutingV2GradGetWorkspaceSize(
    const aclTensor  *gradExpandedX, 
    const aclTensor  *expandedRowIdx, 
    int64_t           topK, 
    int64_t           dropPadMode, 
    int64_t           activeNum, 
    const aclTensor  *out, 
    uint64_t         *workspaceSize, 
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnMoeInitRoutingV2Grad(
    void            *workspace, 
    uint64_t         workspaceSize, 
    aclOpExecutor   *executor, 
    aclrtStream      stream)
```

## aclnnMoeInitRoutingV2GradGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1542px"><colgroup>
  <col style="width: 173px">
  <col style="width: 120px">
  <col style="width: 242px">
  <col style="width: 399px">
  <col style="width: 200px">
  <col style="width: 123px">
  <col style="width: 140px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>gradExpandedX</td>
      <td>Input</td>
      <td>Target tensor after routing.</td>
      <td>The input must be a 2D or 3D tensor. The 2D shape is [B*S*K, H] in the dropless scenario or [A, H] in the active scenario. The 3D shape is [E, C, H] in the drop/pad scenario.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2 or 3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>expandedRowIdx</td>
      <td>Input</td>
      <td>Indicates the token index sorted by expert order.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The input must be a 1D tensor with shape [B*S*K]. </li><li>The element value ranges from [–1, EC) in the drop/pad scenario and from [0, BS*K) in other scenarios. Except for –1, each value is unique.</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>topK</td>
      <td>Input</td>
      <td>TopK value.</td>
      <td>The value must be greater than 0, and the size of axis 0 of expandedRowIdx must be exactly divided by topK.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dropPadMode</td>
      <td>Input</td>
      <td>Indicates whether the scenario is in drop/pad mode.</td>
      <td>The value can be <code>0</code> or <code>1</code>. <ul><li>0: indicates the dropless scenario. </li><li>1: indicates the drop/pad scenario.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activeNum</td>
      <td>Input</td>
      <td>Indicates whether the scenario is an active scenario.</td>
      <td>The value is greater than or equal to 0. This parameter is valid only when dropPadMode is set to 0. The value 0 indicates a non-active scenario, and a value greater than 0 indicates an active scenario. In the active scenario, the size of axis 0 of gradExpandedX must be equal to the value of activeNum.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Indicates the reverse output of Routing.</td>
      <td><ul><li>Empty tensors are supported. </li><li>It must be a 2D tensor with shape [B*S, H]. </li><li>The data type is the same as that of the input gradExpandedX.</li></ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Operator executor, containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  Shape symbol description:
  - B: batch size
  - S: number of tokens
  - H: hidden size, that is, the length of each token sequence
  - K: topK, that is, the number of experts that process tokens
  - A: value of activeNum
  - E: number of experts
  - C: expert capacity, that is, the threshold of the number of tokens that can be processed by an expert

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.
  <table style="undefined;table-layout: fixed; width: 1180px"> 
    <colgroup>
      <col style="width: 250px">
      <col style="width: 130px">
      <col style="width: 800px">
    </colgroup>
    <thead>
      <tr>
        <th>Return</th>
        <th>Error Code</th>
        <th>Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>The input and output tensors are null pointers.</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td>The input and output data types are not supported.</td>
      </tr>
      <tr>
        <td rowspan="8">ACLNN_ERR_INNER_TILING_ERROR</td>
        <td rowspan="8">561002</td>
        <td>The value of dropPadMode is not 0 or 1.</td>
      </tr>
      <tr>
        <td>The value of topK is less than or equal to 0.</td>
      </tr>
      <tr>
        <td>The value of activeNum is less than 0.</td>
      </tr>
      <tr>
        <td>When gradExpandedX is not 2D or 3D, or when dropPadMode is 1, gradExpandedX is not 3D.</td>
      </tr>
      <tr>
        <td>When both dropPadMode and activeNum are 0, the size of axis 0 of gradExpandedX is not equal to that of axis 0 of expandedRowIdx.</td>
      </tr>
      <tr>
        <td>When dropPadMode is 0 and activeNum is greater than 0, the size of axis 0 of gradExpandedX is not equal to that of activeNum.</td>
      </tr>
      <tr>
        <td>The sizes of the last axes of out and gradExpandedX are not equal.</td>
      </tr>
      <tr>
        <td>The size of axis 0 of out is not equal to the size of axis 0 of expandedRowIdx divided by topK.</td>
      </tr>
    </tbody>
  </table>

## aclnnMoeInitRoutingV2Grad

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1180px"> 
    <colgroup>
      <col style="width: 250px">
      <col style="width: 130px">
      <col style="width: 800px">
    </colgroup>
    <thead>
      <tr>
        <th>Name</th>
        <th>Input/Output</th>
        <th>Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>workspace</td>
        <td>Input</td>
        <td>Memory address of the workspace to be allocated on the device.</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnMoeInitRoutingV2GradGetWorkspaceSize`.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Operator executor, containing the operator computation process.</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream for executing the task.</td>
      </tr>
    </tbody>
  </table>

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
    
## Constraints

- Deterministic computation:
  - `aclnnMoeInitRoutingV2Grad` defaults to deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_init_routing_v2_grad.h"
#include <iostream>
#include <vector>

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)
#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream) {
    // (Boilerplate) Initialize resources.
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy the data on the host to the memory on the device.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    // 1. (Boilerplate) Initialize the device and stream. For details, see the list of external ACL APIs.
    // Set the device ID in use.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as required.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct inputs and outputs based on the API definition.
    std::vector<int64_t> gradExpandedXShape = {4, 2};
    std::vector<int64_t> expandedRowIdxShape = {4};
    std::vector<int64_t> gradXShape = {2, 2};
    void* gradExpandedXDeviceAddr = nullptr;
    void* expandedRowIdxDeviceAddr = nullptr;
    void* gradXDeviceAddr = nullptr;
    aclTensor* gradExpandedX = nullptr;
    aclTensor* expandedRowIdx = nullptr;
    aclScalar* k = nullptr;
    aclScalar* dropPadMode = nullptr;
    aclScalar* activeNum = nullptr;
    aclTensor* expertIdx = nullptr;
    aclTensor* out = nullptr;
    std::vector<float> gradExpandedXHostData = {0.1, 0.1, 0.3, 0.3, 0.2, 0.2, 0.4, 0.4};
    std::vector<int32_t> expandedRowIdxHostData = {2, 0, 1, 3};
    std::vector<float> gradXOutHostData = {0, 0, 0, 0};
    int32_t kValue = 2;
    int32_t dropPadModeValue = 0;
    int32_t activeNumValue = 0;

    // Create an input aclTensor.
    ret = CreateAclTensor(gradExpandedXHostData, gradExpandedXShape, &gradExpandedXDeviceAddr, aclDataType::ACL_FLOAT, &gradExpandedX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandedRowIdxHostData, expandedRowIdxShape, &expandedRowIdxDeviceAddr, aclDataType::ACL_INT32, &expandedRowIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    k = aclCreateScalar(&kValue, aclDataType::ACL_INT32);
    CHECK_RET(k != nullptr, return ret);
    dropPadMode = aclCreateScalar(&dropPadModeValue, aclDataType::ACL_INT32);
    CHECK_RET(dropPadMode != nullptr, return ret);
    activeNum = aclCreateScalar(&activeNumValue, aclDataType::ACL_INT32);
    CHECK_RET(activeNum != nullptr, return ret);
    // Create an output aclTensor.
    ret = CreateAclTensor(gradXOutHostData, gradXShape, &gradXDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnMoeInitRoutingV2Grad.
    ret = aclnnMoeInitRoutingV2GradGetWorkspaceSize(gradExpandedX, expandedRowIdx, kValue, dropPadModeValue, activeNumValue, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeInitRoutingV2GradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of the aclnnMoeInitRoutingV2Grad.
    ret = aclnnMoeInitRoutingV2Grad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeInitRoutingV2Grad failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
    auto gradXSize = GetShapeSize(gradXShape);
    std::vector<float> gradXData(gradXSize, 0);
    ret = aclrtMemcpy(gradXData.data(), gradXData.size() * sizeof(gradXData[0]), gradXDeviceAddr, gradXSize * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < gradXSize; i++) {
        LOG_PRINT("gradXData[%ld] is: %f\n", i, gradXData[i]);
    }

    // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
    aclDestroyTensor(gradExpandedX);
    aclDestroyTensor(expandedRowIdx);
    aclDestroyScalar(k);
    aclDestroyScalar(dropPadMode);
    aclDestroyScalar(activeNum);
    aclDestroyTensor(out);

    // 7. Release device resources. Modify the code based on the API definition.
    aclrtFree(gradExpandedXDeviceAddr);
    aclrtFree(expandedRowIdxDeviceAddr);
    aclrtFree(gradXDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
