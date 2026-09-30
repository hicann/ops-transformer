# aclnnMoeFinalizeRoutingV2Grad

## Supported Products

| Product                                                        |  Supported  |
| :----------------------------------------------------------- |:-------:|
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference products</term>                            |    ×    |
| <term>Atlas training products</term>                             |    ×    |

## Function

- **Description**: Performs backpropagation of `aclnnMoeFinalizeRoutingV2`.
- **Formula**:
    R: batch × sequence

    H: hidden
    
    K: topK

    gradY: (R, H)

    expandedRowIdx: (R × K)

    expandedXOptional: (R × K, H) or (activeNum, H) or (expertNum, expertCapacity, H)

    scalesOptional: (R, K)

    expertIdxOptional: (R, K)

    biasOptional: (E, H)
   
    i : 0 ~ R × K - 1

    j : 0 ~ H

    (1) `scalesOptional` is a null pointer:

    $$
    gradExpandedXOut[expandedRowIdx[i]][j] = gradY[i / K][j]
    $$

    (2) `scalesOptional` is not a null pointer, and `biasOptional` is a null pointer:

    $$
    gradExpandedXOut[expandedRowIdx[i]][j] = gradY[i / K][j] × scalesOptional[i]
    $$

    $$
    gradScalesOut[i] = sum(expandedXOptional[expandedRowIdx[i]][j] × gradY[i / K][j])
    $$

    (3) `scalesOptional` and `biasOptional` are not null pointers:
    
    $$
    gradExpandedXOut[expandedRowIdx[i]][j] = gradY[i / K][j] × scalesOptional[i]
    $$

    $$
    gradScalesOut[i] = sum((expandedXOptional[expandedRowIdx[i]][j] + biasOptional[expertIdxOptional[i]][j]) × gradY[i / K][j])
    $$

## Prototype

Each operator consists of [two-phase APIs](../../../docs/en/context/two_phase_api.md). You must first call the `aclnnMoeFinalizeRoutingV2GradGetWorkspaceSize` API to obtain the required workspace size and the executor that contains the operator computation flow, and then call the `aclnnMoeFinalizeRoutingV2Grad` API to execute the computation.

* `aclnnStatus aclnnMoeFinalizeRoutingV2GradGetWorkspaceSize(const aclTensor *gradY, const aclTensor *expandedRowIdx, const aclTensor *expandedXOptional, const aclTensor *scalesOptional, const aclTensor *expertIdxOptional, const aclTensor *biasOptional, int64_t dropPadMode, int64_t activeNum, int64_t expertNum, int64_t expertCapacity, const aclTensor *gradExpandedXOut, const aclTensor *gradScalesOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnMoeFinalizeRoutingV2Grad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnMoeFinalizeRoutingV2GradGetWorkspaceSize

- **Parameters:**
    - `gradY` (aclTensor*, computation input): `aclTensor` on the device, derivative of the forward output y of `MoeFinalizeRoutingV2`. It must be a 2D tensor with shape (R, H). The data type can be FLOAT16, BFLOAT16, or FLOAT32. The [data format](../../../docs/en/context/data_format.md) must be ND. Non-contiguous input is supported.
    - `expandedRowIdx` (aclTensor*, computation input): `aclTensor` on the device, indicating the token index sorted by expert sequence. It must be a 1D tensor with shape (R × K). When `scalesOptional` is set to a null pointer, `K` must be 1. When `dropPadMode` is 0, the value range is [`0, R * K – 1`] and no duplicate index is allowed. When `dropPadMode` is 1, the value range is [`–1, expertNum * expertCapacity – 1`] and no duplicate index except –1 is allowed. The data type is INT32. The [data format](../../../docs/en/context/data_format.md) must be ND. Non-contiguous input is supported.
    - `expandedXOptional` (aclTensor*, optional computation input): `aclTensor` on the device, indicating the features extended based on `expertIdx`. When `scalesOptional` is not a null pointer, this parameter cannot be a null pointer either. When `dropPadMode` is 0, it must be a 2D tensor. When `activeNum` is greater than 0 and less than R × K, the shape is (activeNum, H). Otherwise, the shape is (R × K, H). When `dropPadMode` is 1, it must be a 3D tensor with shape (expertNum, expertCapacity, H). The data type is the same as that of `gradY`. The supported data types are FLOAT16, BFLOAT16, and FLOAT32. The [data format](../../../docs/en/context/data_format.md) must be ND. Non-contiguous input is supported.
    - `scalesOptional` (aclTensor*, optional computation input): `aclTensor` on the device, indicating the feature scaling. It must be a 2D tensor with shape (R, K). The data type can be FLOAT16, BFLOAT16, or FLOAT32. The [data format](../../../docs/en/context/data_format.md) must be ND. Non-contiguous input is supported.
      - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type must be the same as that of `gradY`.
    - `expertIdxOptional` (aclTensor*, optional computation input): `aclTensor` on the device, indicating the index of an expert that processes a feature. When `biasOptional` is a non-null pointer, the value cannot be a null pointer. It must be a 2D tensor with shape (R, K) and value range [0, E - 1], E ≥ 1, allowing duplicate indices. The data type is the same as that of `expandedRowIdx` and can be INT32. The [data format](../../../docs/en/context/data_format.md) must be ND. non-contiguous tensor are supported.
    - `biasOptional` (aclTensor*, optional computation input): `aclTensor` on the device, indicating the feature bias. It must be a 2D tensor with shape (E, H). The data type is the same as that of `gradY` and can be FLOAT16, BFLOAT16, or FLOAT32. The [data format](../../../docs/en/context/data_format.md) must be ND. Non-contiguous input is supported.
    - `dropPadMode` (int64_t, computation input): int64 data type, indicating different scenarios. The value can be 0 or 1. 0 indicates the dropless scenario, where `expertNum` and `expertCapacity` are not verified. 1 indicates the drop scenario, where `expertNum` and `expertCapacity` need to be verified. The values exceeding or below `expertCapacity` will be processed accordingly.
    - `activeNum` (int64_t, computation input): int64 data type, indicating the maximum number of rows in `gradExpandedXOut`. When `dropPadMode` is 0, this parameter takes effect only when `activeNum` is greater than 0 and less than R × K. When `dropPadMode` is 1, this parameter does not take effect.
    - `expertNum` (int64_t, computation input): int64 data type, indicating the number of experts. When `dropPadMode` is 0, this parameter does not take effect. When `dropPadMode` is 1 and `biasOptional` is a non-null pointer, expertNum must be equal to E. When `biasOptional` is a null pointer, `expertNum` must be greater than 0. Otherwise, an error is reported.
    - `expertCapacity` (int64_t, computation input): int64 data type, indicating the number of rows that can be processed by each expert. When `dropPadMode` is 0, this parameter does not take effect. When `dropPadMode` is 1, `expertCapacity` must be greater than 0. Otherwise, an error is reported.
    - `gradExpandedXOut` (aclTensor*, computation output): `aclTensor` on the device, indicating the derivative of the forward input `expandedX` of `MoeFinalizeRoutingV2`. When `dropPadMode` is 0, it must be a 2D tensor. When `activeNum` is greater than 0 and less than R × K, the shape is (activeNum, H). Otherwise, the shape is (R × K, H). When `dropPadMode` is 1, it must be a 3D tensor. The shape is (expertNum, expertCapacity, H). The data type is the same as that of `gradY`. The data type can be FLOAT16, BFLOAT16, or FLOAT32. The [data format](../../../docs/en/context/data_format.md) must be ND. Non-contiguous output is not supported.
    - `gradScalesOut` (aclTensor*, computation output): `aclTensor` on the device, indicating the derivative of the forward input scales of `MoeFinalizeRoutingV2`. The output is valid only when `scalesOptional` is not a null pointer. The output must be a 2D tensor with shape (R, K). The data type is the same as that of `scalesOptional` and can be FLOAT16, BFLOAT16, or FLOAT32. The [data format](../../../docs/en/context/data_format.md) must be ND. Non-contiguous input is not supported.
    - `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.
    - `executor` (aclOpExecutor**, output): operator executor, covering the operator computation process.

- **Returns**:

    `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

    ```cpp
    The first-phase API implements input parameter verification. The following errors may be thrown.
    161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The required input or output tensor is a null pointer.
    161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type or format of input or output is not supported.
    561002 (ACLNN_ERR_INNER_TILING_ERROR): 1. The shape or value of input or output does not meet the requirements in the parameter description.
    ```

## aclnnMoeFinalizeRoutingV2Grad

- **Parameters:**
    - `workspace` (void*, input): address of the workspace to be allocated on the device.
    - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling `aclnnMoeFinalizeRoutingV2GradGetWorkspaceSize`.
    - `executor` (aclOpExecutor*, input): operator executor, covering the operator computation process.
    - `stream` (aclrtStream, input): stream for executing the task.

- **Returns**:

    `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computing:
  - `aclnnMoeFinalizeRoutingV2Grad` defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_finalize_routing_v2_grad.h"
#include <iostream>

#define CHECK_RET(cond, return_expr)                                           \
  do {                                                                         \
    if (!(cond)) {                                                             \
      return_expr;                                                             \
    }                                                                          \
  } while (0)

#define LOG_PRINT(message, ...)                                                \
  do {                                                                         \
    printf(message, ##__VA_ARGS__);                                            \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                         size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream *stream) {
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
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
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
  // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> gradYShape = {2, 2};
  std::vector<int64_t> expandedRowIdxShape = {4};
  std::vector<int64_t> expandedXShape = {4, 2};
  std::vector<int64_t> scalesShape = {2, 2};
  std::vector<int64_t> expertIdxShape = {2, 2};
  std::vector<int64_t> biasShape = {2, 2};
  std::vector<int64_t> gradExpandedXShape = {4, 2};
  std::vector<int64_t> gradScalesShape = {2, 2};
  void* gradYDeviceAddr = nullptr;
  void* expandedRowIdxDeviceAddr = nullptr;
  void* expandedXDeviceAddr = nullptr;
  void* scalesDeviceAddr = nullptr;
  void* expertIdxDeviceAddr = nullptr;
  void* biasDeviceAddr = nullptr;
  void* gradExpandedXDeviceAddr = nullptr;
  void* gradScalesDeviceAddr = nullptr;

  aclTensor* gradY = nullptr;
  aclTensor* expandedRowIdx = nullptr;
  aclTensor* expandedX = nullptr;
  aclTensor* scales = nullptr;
  aclTensor* expertIdx = nullptr;
  aclTensor* bias = nullptr;
  int64_t dropPadMode = 0;
  int64_t activeNum = 0;
  int64_t expertNum = 0;
  int64_t expertCapacity = 0;
  aclTensor* gradExpandedX = nullptr;
  aclTensor* gradScales = nullptr;

  std::vector<float> gradYHostData = {0.3816, 0.3939, 0.8474, 0.1652};
  std::vector<int> expandedRowIdxHostData = {1, 3, 0, 2};
  std::vector<float> expandedXHostData = {0.6049, 0.3315, 0.4954, 0.3284, 0.7060, 0.4359, 0.6514, 0.9476};
  std::vector<float> scalesHostData = {0.4708, 0.0656, 0.9652, 0.9512};
  std::vector<int> expertIdxHostData = {0, 1, 0, 1};
  std::vector<float> biasHostData = {0.6452, 0.1981, 0.4159, 0.9575};
  std::vector<float> gradExpandedXHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> gradScalesHostData = {0, 0, 0, 0};

  ret = CreateAclTensor(gradYHostData, gradYShape, &gradYDeviceAddr, aclDataType::ACL_FLOAT, &gradY);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(expandedRowIdxHostData, expandedRowIdxShape, &expandedRowIdxDeviceAddr, aclDataType::ACL_INT32,
                        &expandedRowIdx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(expandedXHostData, expandedXShape, &expandedXDeviceAddr, aclDataType::ACL_FLOAT, &expandedX);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_FLOAT, &scales);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(expertIdxHostData, expertIdxShape, &expertIdxDeviceAddr, aclDataType::ACL_INT32, &expertIdx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gradExpandedXHostData, gradExpandedXShape, &gradExpandedXDeviceAddr, aclDataType::ACL_FLOAT,
                        &gradExpandedX);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gradScalesHostData, gradScalesShape, &gradScalesDeviceAddr, aclDataType::ACL_FLOAT, &gradScales);
  CHECK_RET(ret == ACL_SUCCESS, return ret);  

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;

  // Call the first-phase API of aclnnMoeFinalizeRoutingV2Grad.
  ret = aclnnMoeFinalizeRoutingV2GradGetWorkspaceSize(gradY, expandedRowIdx, expandedX, scales, expertIdx, bias,
                                                      dropPadMode, activeNum, expertNum, expertCapacity, gradExpandedX,gradScales, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeFinalizeRoutingV2GradGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);

  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnMoeFinalizeRoutingV2Grad.
  ret = aclnnMoeFinalizeRoutingV2Grad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeFinalizeRoutingV2Grad failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  LOG_PRINT("gradExpandedX result is: \n");
  PrintOutResult(gradExpandedXShape, &gradExpandedXDeviceAddr);
  LOG_PRINT("gradScales result is: \n");
  PrintOutResult(gradScalesShape, &gradScalesDeviceAddr);

  // 6. Destroy aclTensor objects. Modify the code based on the API definition.
  aclDestroyTensor(gradY);
  aclDestroyTensor(expandedRowIdx);
  aclDestroyTensor(expandedX);
  aclDestroyTensor(scales);
  aclDestroyTensor(expertIdx);
  aclDestroyTensor(bias);
  aclDestroyTensor(gradExpandedX);
  aclDestroyTensor(gradScales);

  // 7. Free device resources.
  aclrtFree(gradYDeviceAddr);
  aclrtFree(expandedRowIdxDeviceAddr);
  aclrtFree(expandedXDeviceAddr);
  aclrtFree(scalesDeviceAddr);
  aclrtFree(expertIdxDeviceAddr);
  aclrtFree(biasDeviceAddr);
  aclrtFree(gradExpandedXDeviceAddr);
  aclrtFree(gradScalesDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
