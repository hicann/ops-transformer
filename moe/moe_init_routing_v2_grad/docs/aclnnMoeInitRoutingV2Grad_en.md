# aclnnMoeInitRoutingV2Grad

## Product Support

| Product                                                        |  Supported  |
| :----------------------------------------------------------- |:-------:|
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference products</term>                            |    ×    |
| <term>Atlas training products</term>                             |    ×    |

## Function

- **Description**: Implements the backward propagation of [aclnnMoeInitRoutingV2](../../../moe/moe_init_routing_v2/docs/aclnnMoeInitRoutingV2_en.md) to complete the weighted summation of tokens.

- **Formula**:

    $$
    gradX_i=\sum_{t=0}^{topK}gradExpandedX[expandedRowIdx[i * topK + t]]
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnMoeInitRoutingV2GradGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnMoeInitRoutingV2Grad` is called to perform computation.

* `aclnnStatus aclnnMoeInitRoutingV2GradGetWorkspaceSize(const aclTensor *gradExpandedX, const aclTensor *expandedRowIdx, int64_t topK, int64_t dropPadMode, int64_t activeNum, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnMoeInitRoutingV2Grad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnMoeInitRoutingV2GradGetWorkspaceSize

- **Parameters**:
    - `gradExpandedX` (aclTensor\*, computation input): target tensor after routing. It must be a 2D or 3D tensor. The 2D shape is [B\*S\*K, H] in the dropless scenario or [A, H] in the active scenario. The 3D shape is [E, C, H] in the drop/pad scenario. The data type can be FLOAT16, BFLOAT16, or FLOAT32. The [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported.
    - `expandedRowIdx` (aclTensor\*, computation input): tokens are indexed in expert order. It is a 1D tensor with shape [B\*S\*K]. The value range is [-1, E\*C) in the drop/pad scenario and [0, B\*S\*K) in other scenarios. The value is unique except `-1`. The data type can be INT32. The [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported.
    - `topK` (int64_t, computation input): an integer on the host. The value must be greater than `0` and exactly divisible by the size of axis 0 of `expandedRowIdx`.
    - `dropPadMode` (int64\_t, computation input): indicates whether the scenario is in drop mode. It is an integer on the host. The value range is [0, 1]. `0` indicates the dropless scenario, and `1` indicates the drop/pad scenario.
    - `activeNum` (int64\_t, computation input): indicates whether the active scenario is involved. It is an integer on the host. The value range is greater than or equal to `0`. This parameter is valid when `dropPadMode` is set to `0`, indicating the non-active scenario, and a value greater than 0 indicates the active scenario. In the active scenario, the size of axis 0 of `gradExpandedX` must be equal to the value of `activeNum`.
    - `out` (aclTensor\*, computation output): backward output of routing. It is a 2D tensor with shape [B\*S, H]. The data type can be FLOAT16, BFLOAT16, or FLOAT32. The output type is the same as that of `gradExpandedX`. The [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are not supported.
    - `workspaceSize` (uint64\_t\*, output): size of the workspace to be allocated on the device.
    - `executor` (aclOpExecutor\*\*, output): operator executor, containing the operator computation process.

      Shape symbol description:

      `B`: batch size; `S`: number of tokens; `H`: hidden size, that is, the length of each token sequence; `K`: `topK`, that is, the number of experts that process tokens.
      `A`: `activeNum` value; `E`: number of experts; `C`: expert capacity, that is, the number of tokens that can be processed by an expert.

- **Returns**

    `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

    ```cpp
    The first-phase API implements input parameter verification. The following errors may be thrown:
    161001 (ACLNN_ERR_PARAM_NULLPTR): The input or output tensor is a null pointer.
    161002 (ACLNN_ERR_PARAM_INVALID): The data type of input or output is not supported.
    561002 (ACLNN_ERR_INNER_TILING_ERROR): 1. dropPadMode is not 0 or 1.
                                          2. topK is less than or equal to 0.
                                          3. activeNum is less than 0.
                                          4. The shape of gradExpandedX is not 2D or 3D, or the shape of gradExpandedX is not 3D when dropPadMode is set to 1.
                                          5. When both dropPadMode and activeNum are set to 0, axis 0 of gradExpandedX has a different size from that of expandedRowIdx.
                                          6. When dropPadMode is set to 0 and activeNum is greater than 0, the size of axis 0 of gradExpandedX is different from that of activeNum.
                                          7. The sizes of the last axes of out and gradExpandedX are different.
                                          8. Axis 0 of out is not equal to the size of axis 0 of expandedRowIdx divided by topK.
    ```

## aclnnMoeInitRoutingV2Grad

- **Parameters:**
    - `workspace` (void\*, input): address of the workspace to be allocated on the device.
    - `workspaceSize` (uint64\_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnMoeInitRoutingV2GradGetWorkspaceSize`.
    - `executor` (aclOpExecutor\*, input): operator executor, containing the operator computation process.
    - `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

    `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnMoeInitRoutingV2Grad` defaults to deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
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
    std::vector<float> gradXOutHostData = {0, 0, 0, 0, 0, 0, 0, 0};
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
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
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
