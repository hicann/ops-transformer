# aclnnMoeFusedTopk

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference products</term>   |     ×    |
|  <term>Atlas training products</term>   |     ×    |

## Function

- Description: Applies sigmoid to input `x` during MoE computation, sorts the calculation results by group, and selects the top k experts based on the group sorting result.
- Formula:

    Computes sigmoid of the input element-wise.

    $$
    sigmoidRes=sigmoid(x)
    $$

    Add addNum:

    $$
    normOut = sigmoidRes + addNum
    $$

    Group the computation results based on `groupNum`, sort each group based on the sum value of top N, and select the first `groupTopk` groups.

    $$
    groupOut, groupId = TopK(ReduceSum(TopK(Split(normOut, groupCount), k=2, dim=-1), dim=-1),k=kGroup)
    $$

    Obtain the corresponding element in `normOut` based on the `groupId` obtained in the previous step, and perform topK on the data to obtain the indices result.

    $$
    normY,indices=TopK(normOut[groupId, :],k=k)
    $$

    Select `y` from `sigmoidRes` based on indices.

    $$
    y = gather(sigmoidRes, indices)
    $$

    If `isNorm` is `true`, `y` is computed based on the input scale parameter to obtain the result of `y`.
    
    $$
    y = y / (ReduceSum(y, dim=-1)) × scale
    $$
    
    If `enableExpertMapping` is `true`, the physical experts in indices are mapped to logical experts based on the input `mappingNum` and `mappingTable` to obtain the output indices.

## Prototype

Each operator consists of [two-phase APIs](../../../docs/en/context/two_phase_api.md). You must first call the `aclnnMoeFusedTopkGetWorkspaceSize` API to obtain the required workspace size and the executor that contains the operator computation flow, and then call the `aclnnMoeFusedTopk` API to execute the computation.

- `aclnnMoeFusedTopkGetWorkspaceSize(const aclTensor* x, const aclTensor* addNum, const aclTensor* mappingNum, const aclTensor* mappingTable, uint32_t groupNum, uint32_t groupTopk, uint32_t topN, uint32_t topK, uint32_t activateType, bool isNorm, float scale, bool enableExpertMapping, aclTensor* y, aclTensor* indices, uint64_t* workspaceSize, aclOpExecutor** executor)`

- `aclnnStatus aclnnMoeFusedTopk(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnMoeFusedTopkGetWorkspaceSize

- **Parameters:**

  - `x` (aclTensor*, computation input): `aclTensor` on the device. Each token corresponds to the score of each expert. The shape is (numToken, expertNum). The data type can be FLOAT16, BFLOAT16, or FLOAT32. The [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported.
  - `addNum` (aclTensor\*, computation input): `aclTensor` on the device, which is the bias value for computation with input `x`. The shape is (expertNum). The data type must be the same as that of `x`. The [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported.
  - `mappingNum` (aclTensor\*, computation input): `aclTensor` on the device. This parameter is not enabled when `enableExpertMapping` is set to false. The shape is (expertNum). The number of logical experts to which each physical expert is actually mapped. The data type can be INT32. The [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported.
  - `mappingTable` (aclTensor\*, computation input): `aclTensor` on the device. This parameter is not enabled when `enableExpertMapping` is set to false. The shape is (expertNum, maxMappingNum), indicating the mapping table of each physical expert or logical expert. The value of `maxMappingNum` is less than or equal to 128. The data type must be INT32. The [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported.
  - `groupNum` (uint32_t, computation input): number of groups. The value must be greater than 0.
  - `groupTopk` (uint32_t, computation input): number of selected groups. The value must be greater than 0.
  - `topN` (uint32_t, computation input): number of experts selected from each group for summation. The value must be greater than 0.
  - `topK` (uint32_t, computation input): number of experts selected in the end. The value must be greater than 0.
  - `activateType` (uint32_t, computation input): activation type. Currently, only 0 (ACTIVATION_SIGMOID) is supported.
  - `isNorm` (bool, computation input): whether to normalize the output.
  - `scale` (float, computation input): coefficient multiplication after normalization.
  - `enableExpertMapping` (bool, computation input): whether to enable the mapping from physical experts to logical experts.
  - `y` (aclTensor\*, computation output): `aclTensor` on the device. The shape is (numToken, topK). The data type is FLOAT32. The [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported.
  - `indices` (aclTensor\*, computation output): `aclTensor` on the device. The shape is (numToken, topK). The data type is INT32. The [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported.
  - `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor\*\*, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```cpp
  The first-phase API implements input parameter verification. The following errors may be thrown.
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The input `x` or `addNum` is a null pointer.
                                    2. The output `y` or `indices` is a null pointer.
                                    3. When `enableExpertMapping` is set to true, the input `mappingNum` or `mappingTable` is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The input or output data type or format is not supported.
                                    2. The input parameter does not meet the constraints.
                                    3. The input and output shapes do not meet the constraints.
  ```

## aclnnMoeFusedTopk

- **Parameters:**

  - `workspace` (void \*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnMoeFusedTopkGetWorkspaceSize`.
  - `executor` (aclOpExecutor \*, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns**:

  `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computing:
  - `aclnnMoeFusedTopk` defaults to a deterministic implementation.

- `expertNum` must be an integer multiple of `groupNum`.
- `groupTopk` must be less than or equal to `groupNum`.
- `maxMappingNum` must be less than or equal to 128.
- `TopK` must be less than or equal to `expertNum`.
- `TopN` must be less than or equal to `expertNum`/`groupNum`.
- `expertNum` must be less than or equal to 1024.
- `groupNum` must be less than or equal to 256.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <iostream>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_fused_topk.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
  do {                                     \
      if (!(cond)) {                       \
          Finalize(deviceId, stream);      \
          return_expr;                     \
      }                                    \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
    //(Boilerplate) Perform initialization.
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

  // Compute the stride of the contiguous tensor.
  std::vector<int64_t> stride(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    stride[i] = shape[i + 1] * stride[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, stride.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
}

int aclnnMoeFusedTopkTest(int32_t deviceId, aclrtStream& stream) {
  auto ret = Init(deviceId, &stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs based on the API definition.

  int64_t num_token = 16;
  int64_t expert_num = 32;
  int64_t max_mapping_num = 16;

  uint32_t groupNum = 2;
  uint32_t groupTopk = 2;
  uint32_t topN = 2;
  uint32_t topK = 4;
  uint32_t activateType = 0;
  bool isNorm = false;
  float scale = 1.0;
  bool enableExpertMapping = true;

  std::vector<int64_t> xShape = {num_token, expert_num};
  std::vector<int64_t> addNumShape = {expert_num};
  std::vector<int64_t> mappingNumShape = {expert_num};
  std::vector<int64_t> mappingTableShape = {expert_num, max_mapping_num};

  std::vector<int64_t> yShape = {num_token, topK};
  std::vector<int64_t> indicesShape = {num_token, topK};

  void* xDeviceAddr = nullptr;
  void* addNumDeviceAddr = nullptr;
  void* mappingNumDeviceAddr = nullptr;
  void* mappingTableDeviceAddr = nullptr;

  void* yDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;

  aclTensor* x = nullptr;
  aclTensor* addNum = nullptr;
  aclTensor* mappingNum = nullptr;
  aclTensor* mappingTable = nullptr;

  aclTensor* y = nullptr;
  aclTensor* indices = nullptr;

  std::vector<float> xHostData(GetShapeSize(xShape), 1);
  std::vector<float> addNumHostData(GetShapeSize(addNumShape), 1);
  std::vector<int32_t> mappingNumHostData(GetShapeSize(mappingNumShape), 1);
  std::vector<int32_t> mappingTableHostData(GetShapeSize(mappingTableShape), 1);

  std::vector<float> yHostData(GetShapeSize(yShape), 0);
  std::vector<int32_t> indicesHostData(GetShapeSize(indicesShape), 0);

  // Create an x aclTensor.
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> xTensorPtr(x, aclDestroyTensor);

  // Create an addNum aclTensor.
  ret = CreateAclTensor(addNumHostData, addNumShape, &addNumDeviceAddr, aclDataType::ACL_FLOAT, &addNum);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> addNumTensorPtr(addNum, aclDestroyTensor);

  // Create a mappingNum aclTensor.
  ret = CreateAclTensor(mappingNumHostData, mappingNumShape, &mappingNumDeviceAddr, aclDataType::ACL_INT32, &mappingNum);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> mappingNumTensorPtr(mappingNum, aclDestroyTensor);

  // Create a mappingTable aclTensor.
  ret = CreateAclTensor(mappingTableHostData, mappingTableShape, &mappingTableDeviceAddr, aclDataType::ACL_INT32, &mappingTable);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> mappingTableTensorPtr(mappingTable, aclDestroyTensor);

  // Create a y aclTensor.
  ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> yTensorPtr(y, aclDestroyTensor);

  // Create an indices aclTensor.
  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT32, &indices);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> indicesTensorPtr(indices, aclDestroyTensor);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnMoeFusedTopk.
  ret = aclnnMoeFusedTopkGetWorkspaceSize(x,
                                       addNum,
                                       mappingNum,
                                       mappingTable,
                                       groupNum,
                                       groupTopk,
                                       topN,
                                       topK,
                                       activateType,
                                       isNorm,
                                       scale,
                                       enableExpertMapping,
                                       y,
                                       indices,
                                       &workspaceSize,
                                       &executor);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeFusedTopkGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    workspaceAddrPtr.reset(workspaceAddr);
  }
  // Call the second-phase API of aclnnMoeFusedTopk.
  ret = aclnnMoeFusedTopk(workspaceAddr, workspaceSize, executor, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeFusedTopk failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(yShape);
  std::vector<float> yData(size, 0);
  ret = aclrtMemcpy(yData.data(), yData.size() * sizeof(yData[0]), yDeviceAddr,
                    size * sizeof(yData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("out result[%ld] is: %f\n", i, yData[i]);
  }

  return ACL_SUCCESS;
}

int main() {
  // 1. (Boilerplate) Initialize the device and stream. For details, see the external API list.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = aclnnMoeFusedTopkTest(deviceId, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeFusedTopkTest failed. ERROR: %d\n", ret); return ret);

  Finalize(deviceId, stream);
  return 0;
}
```
