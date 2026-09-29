# aclnnMoeFinalizeRoutingV2Grad

[📄 View Source Code](https://gitcode.com/cann/ops-transformer/tree/master/moe/moe_finalize_routing_v2_grad)

## Supported Products

| Product                                                        |  Supported  |
| :----------------------------------------------------------- |:-------:|
| Ascend 950PR/Ascend 950DT                     |     √    |
| <term>Atlas A3 training products/Atlas A3 inference products</term>     |    √    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>     |    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference products</term>                            |    ×    |
| <term>Atlas training products</term>                             |    ×    |

## Function

- Description: Performs backpropagation of aclnnMoeFinalizeRoutingV2.
- Formulas:

  $$
  i : 0 \sim R * K - 1
  $$
  
  $$
  j : 0 \sim H
  $$

  (1) `scalesOptional` is a null pointer:

  $$
  gradExpandedXOut[expandedRowIdx[i]][j] = gradY[i / K][j]
  $$

  (2) `scalesOptional` is not a null pointer, and `biasOptional` is a null pointer:

  $$
  gradExpandedXOut[expandedRowIdx[i]][j] = gradY[i / K][j] * scalesOptional[i / K][i \% K]
  $$

  $$
  gradScalesOut[i] = sum(expandedXOptional[expandedRowIdx[i]][j] * gradY[i / K][j])
  $$

  (3) `scalesOptional` and `biasOptional` are not null pointers:
  
  $$
  gradExpandedXOut[expandedRowIdx[i]][j] = gradY[i / K][j] * scalesOptional[i / K][i \% K]
  $$

  $$
  gradScalesOut[i] = sum((expandedXOptional[expandedRowIdx[i]][j] + biasOptional[expertIdxOptional[i]][j]) * gradY[i / K][j])
  $$

  R indicates batch x sequence, H indicates hidden, and K indicates topK.

## Prototype

Each operator consists of [two-phase APIs](../../../docs/en/context/two_phase_api.md). You must first call the `aclnnMoeFinalizeRoutingV2GradGetWorkspaceSize` API to obtain the required workspace size and the executor that contains the operator computation flow, and then call the `aclnnMoeFinalizeRoutingV2Grad` API to execute the computation.

```c++
aclnnStatus aclnnMoeFinalizeRoutingV2GradGetWorkspaceSize(
  const aclTensor *gradY,
  const aclTensor *expandedRowIdx,
  const aclTensor *expandedXOptional,
  const aclTensor *scalesOptional,
  const aclTensor *expertIdxOptional,
  const aclTensor *biasOptional,
  int64_t          dropPadMode,
  int64_t          activeNum,
  int64_t          expertNum,
  int64_t          expertCapacity,
  const aclTensor *gradExpandedXOut,
  const aclTensor *gradScalesOut,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor);
```

```c++
aclnnStatus aclnnMoeFinalizeRoutingV2Grad(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream);
```

## aclnnMoeFinalizeRoutingV2GradGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1656px"><colgroup>
  <col style="width: 226px">
  <col style="width: 124px">
  <col style="width: 288px">
  <col style="width: 288px">
  <col style="width: 193px">
  <col style="width: 193px">
  <col style="width: 193px">
  <col style="width: 151px">
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
      <td>gradY (aclTensor)</td>
      <td>Input</td>
      <td>Gradient of the forward output y of MoeFinalizeRoutingV2.</td>
      <td>-</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>(R, H)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>expandedRowIdx (aclTensor)</td>
      <td>Input</td>
      <td>Index of the token sorted by expert sequence.</td>
      <td>When scalesOptional is passed as a null pointer, K must be 1. When dropPadMode is 0, the value range is [0, R * K – 1], and there is no duplicate index. When dropPadMode is 1, the value range is [–1, expertNum * expertCapacity – 1], and there is no duplicate index except –1.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(R * K)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>expandedXOptional (aclTensor)</td>
      <td>Input</td>
      <td>Features expanded based on `expertIdx`.</td>
      <td>When scalesOptional is a non-null pointer, it cannot be a null pointer.</td>
      <td>Same as `gradY`.</td>
      <td>ND</td>
      <td>When dropPadMode is 0, if activeNum is greater than 0 and less than R x K, the shape is (activeNum, H); otherwise, the shape is (R x K, H).<br>When dropPadMode is 1: (expertNum, expertCapacity, H)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scalesOptional (aclTensor)</td>
      <td>Input</td>
      <td>Feature scaling.</td>
      <td>-</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>(R, K)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>expertIdxOptional (aclTensor)</td>
      <td>Input</td>
      <td>Index of the expert corresponding to each feature.</td>
      <td>When biasOptional is not a null pointer, expertIdxOptional cannot be a null pointer either. The value range is [0, E - 1], E &gt;= 1,. Duplicate indexes are allowed. E indicates the number of experts.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(R, K)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>biasOptional (aclTensor)</td>
      <td>Input</td>
      <td>Offset performed on the feature.</td>
      <td>-</td>
      <td>The value is the same as that of gradY.</td>
      <td>ND</td>
      <td>(E, H)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dropPadMode (int64_t)</td>
      <td>Input</td>
      <td>Indicates that different scenarios are used.</td>
      <td>The value can be 0 or 1. 0 indicates the dropless scenario, in which expertNum and expertCapacity are not verified. 1 indicates the drop scenario, in which expertNum and expertCapacity are verified. If the number of elements processed by each expert exceeds or is less than the value of expertCapacity, the corresponding processing is performed.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activeNum (int64_t)</td>
      <td>Input</td>
      <td>Maximum number of output rows of gradExpandedXOut.</td>
      <td>When dropPadMode is 0, this parameter takes effect only when activeNum is greater than 0 and less than R x K. When dropPadMode is 1, this parameter does not take effect.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertNum (int64_t)</td>
      <td>Input</td>
      <td>Number of experts.</td>
      <td>When dropPadMode is 0, this parameter does not take effect. When dropPadMode is 1 and biasOptional is a non-null pointer, expertNum must be equal to E. When dropPadMode is 1 and biasOptional is a null pointer, expertNum must be greater than 0. Otherwise, an error is reported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertCapacity (int64_t)</td>
      <td>Input</td>
      <td>Number of rows that can be processed by each expert.</td>
      <td>When dropPadMode is 0, this parameter does not take effect. When dropPadMode is 1, expertCapacity must be greater than 0. Otherwise, an error is reported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradExpandedXOut (aclTensor)</td>
      <td>Output</td>
      <td>Derivative of the forward input expandedX of MoeFinalizeRoutingV2.</td>
      <td>-</td>
      <td>Same as gradY.</td>
      <td>ND</td>
      <td>Same as expandedXOptional.</td>
      <td>×</td>
    </tr>
    <tr>
      <td>gradScalesOut (aclTensor)</td>
      <td>Output</td>
      <td>Derivative of the forward input scales of MoeFinalizeRoutingV2.</td>
      <td>This output is valid only when scalesOptional is not a null pointer.</td>
      <td>Same as scalesOptional.</td>
      <td>ND</td>
      <td>(R, K)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t)</td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor)</td>
      <td>Output</td>
      <td>Operator executor, containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>:
      - The data type of scalesOptional must be the same as that of gradY.
  - Ascend 950PR/Ascend 950DT:
      - The data type of scalesOptional can be different from that of gradY.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 330px">
  <col style="width: 140px">
  <col style="width: 762px">
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
    <td> ACLNN_ERR_PARAM_NULLPTR </td>
    <td> 161001 </td>
    <td>The input and output tensors are null pointers.</td>
    </tr>
    <tr>
    <td> ACLNN_ERR_PARAM_INVALID </td>
    <td> 161002 </td>
    <td>The input and output data types and formats are not supported.</td>
    </tr>
    <tr>
    <td> ACLNN_ERR_INNER_TILING_ERROR </td>
    <td> 561002 </td>
    <td>The input and output shapes and values do not meet the requirements specified in the parameter description.</td>
    </tr>
  </tbody></table>

## aclnnMoeFinalizeRoutingV2Grad

- **Parameters**
  <table style="undefined;table-layout: fixed; width: 1148px"><colgroup>
  <col style="width: 170px">
  <col style="width: 134px">
  <col style="width: 844px">
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
      <td>Size of the workspace to be allocated on the device, which is obtained by the first-phase API aclnnMoeFinalizeRoutingV2GradGetWorkspaceSize.</td>
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

- Deterministic computing:
  - `aclnnMoeFinalizeRoutingV2Grad` defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
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

  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
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

  // 6. Destroy aclTensor and aclScalar. Modify the code based on the API definition.
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
