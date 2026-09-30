# aclnnMoeTokenUnpermuteGrad

## Product Support

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- **Description**: Performs the backpropagation of `aclnnMoeTokenUnpermute`.
- **Formula**:

  - If `probs` is not set to `None`:

    $$
    unpermutedTokens[i] = permutedTokens[sortedIndices[i]]
    $$

    $$
    unpermutedTokens = unpermutedTokens.reshape(-1, topK, hiddenSize)
    $$

    $$
    unpermutedTokens = unpermutedTokensGrad.unsqueeze(1) * unpermutedTokens
    $$

    $$
    probsGrad = \sum_{k=0}^{K}(unpermutedTokens_{i,j,k})
    $$

    $$
    permutedTokensGrad[sortedIndices[i]] = ((unpermutedTokensGrad.unsqueeze(1) * probs.unsqueeze(-1)).reshape(-1, hiddenSize))[i]
    $$

  - If `probs` is set to `None`:

    $$
    permutedTokensGrad[sortedIndices[i]] = unpermutedOutputGrad[i]
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnMoeTokenUnpermuteGradGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnMoeTokenUnpermuteGrad` is called to perform computation.

* `aclnnStatus aclnnMoeTokenUnpermuteGradGetWorkspaceSize(const aclTensor* permuteTokens, const aclTensor* unpermutedTokensGrad, const aclTensor* sortedIndices, const aclTensor* probsOptional, bool paddedMode, const aclIntArray* restoreShapeOptional, aclTensor* permutedTokensGradOut, aclTensor* probsGradOut, uint64_t* workspaceSize, aclOpExecutor** executor)`
* `aclnnStatus aclnnMoeTokenUnpermuteGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnMoeTokenUnpermuteGradGetWorkspaceSize

- **Parameters:**
    - `permutedTokens` (aclTensor\*, computation input): aclTensor on the device, which indicates an input token. The value must be a 2D tensor with shape (tokens_num \* topK_num, hidden_size). The data type can be BFLOAT16, FLOAT16, or FLOAT32. The [data format](../../../docs/en/context/data_format.md) must be ND. Non-contiguous input is supported.
        - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The value of `topK_num` is less than or equal to `512`.
    - `unpermutedTokensGrad` (aclTensor\*, computation input): aclTensor on the device, which indicates the gradient of the forward output `unpermutedTokens`. The value must be a 2D tensor with shape (tokens_num, hidden_size). The data type is the same as that of `permutedTokens` and can be BFLOAT16, FLOAT16, or FLOAT32. The [data format](../../../docs/en/context/data_format.md) must be ND. Non-contiguous input is supported.
    - `sortedIndices` (aclTensor\*, computation input): aclTensor on the device. The value must be a 1D shape with size (tokens_num \* topK_num,). The data type can be INT32. The [data format](../../../docs/en/context/data_format.md) must be ND. The value range is [0, tokens_num \* topK_num – 1], and there is no duplicate index. Non-contiguous input is supported.
    - `probsOptional` (aclTensor\*, computation input): aclTensor on the device. This input is optional. The value must be a 2D shape with size (tokens_num, topK_num). The data type can be BFLOAT16, FLOAT16, or FLOAT32. The [data format](../../../docs/en/context/data_format.md) must be ND. When `probs` is passed, the value of `topK_num` equals the second dimension of `probs`. When `probs` is not passed, the value of `topK_num` is `1`. Non-contiguous input is supported.
    - `paddedMode` (bool, computation input): `true` indicates that `paddedMode` is enabled, and `false` indicates that `paddedMode` is disabled. For details about `paddedMode`, see the `restoreShapeOptional` parameter. Currently, only `false` is supported.
    - `restoreShapeOptional` (aclIntArray\*, computation input): This parameter takes effect only when `paddedMode` is set to `true`. Otherwise, no operation is performed on this parameter. When `paddedMode` is set to `true`, the shape is the same as that of `unpermutedTokens`. Currently, only `nullptr` is supported.
    - `permutedTokensGradOut` (aclTensor\*, computation output): gradient of the input `permutedTokens`. The value must be a 2D tensor with shape (tokens_num \* topK_num, hidden_size). The data type is the same as that of `permutedTokens`. The data type can be BFLOAT16, FLOAT16, or FLOAT32. The [data format](../../../docs/en/context/data_format.md) must be ND. Non-contiguous output is not supported.
    - `probsGradOut` (aclTensor\*, output): gradient of the input `probs`. This output is optional. The value must be a 2D tensor with shape (tokens_num, topK_num). The data type is the same as that of `probsOptional`. The data type can be BFLOAT16, FLOAT16, or FLOAT32. The [data format](../../../docs/en/context/data_format.md) must be ND. Non-contiguous output is not supported.
    - `workspaceSize` (uint64\_t\*, output): size of the workspace to be allocated on the device.
    - `executor` (aclOpExecutor\*\*, output): operator executor, containing the operator computation process.

- **Returns:**

    `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
   
    ```cpp
    The first-phase API implements input parameter verification. The following errors may be thrown:
    161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The input and output tensors are null pointers.
    161002 (ACLNN_ERR_PARAM_INVALID): 1. The input and output data types are not supported.
    ```

## aclnnMoeTokenUnpermuteGrad

- **Parameters:**
    - `workspace` (void\*, input): address of the workspace to be allocated on the device.
    - `workspaceSize` (uint64\_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnMoeTokenUnpermuteGradGetWorkspaceSize`.
    - `executor` (aclOpExecutor\*, input): operator executor, containing the operator computation process.
    - `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

    `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnMoeTokenUnpermuteGrad` defaults to deterministic implementation.
- <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The value of `topK_num` is less than or equal to `512`.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_token_unpermute_grad.h"
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
  auto ret = aclrtMemcpy(
      resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
      return );
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream *stream) {
  // (Boilerplate) Initialize resources.
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
            return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData,
                    const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
            return ret);
    // Call aclrtMemcpy to copy the data on the host to the memory on the device.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size,
                    ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
            return ret);

  // Compute the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
                            strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return ret);

  // 2. Construct inputs and outputs based on the API definition.
  std::vector<int64_t> permutedTokensShape = {3, 2};
  std::vector<int64_t> unpermutedTokensGradShape = {1, 2};
  std::vector<int64_t> probsShape = {1, 3};
  std::vector<int64_t> sortedIndicesShape = {3};
  std::vector<int64_t> permutedTokensGradShape = {3, 2};
  std::vector<int64_t> probsGradShape = {1, 3};
  void* permutedTokensDeviceAddr = nullptr;
  void* unpermutedTokensGradDeviceAddr = nullptr;
  void* probsDeviceAddr = nullptr;
  void* sortedIndicesDeviceAddr = nullptr;
  void* permutedTokensGradDeviceAddr = nullptr;
  void* probsGradDeviceAddr = nullptr;

  aclTensor* permutedTokens = nullptr;
  aclTensor* unpermutedTokensGrad = nullptr;
  aclTensor* probs = nullptr;
  aclTensor* sortedIndices = nullptr;
  bool paddedMode = false;
  aclTensor *permutedTokensGrad = nullptr;
  aclTensor *probsGrad = nullptr;

  std::vector<float> permutedTokensHostData = {1, 1, 1, 1, 1, 1};
  std::vector<float> unpermutedTokensGradHostData = {1, 1};
  std::vector<float> probsHostData = {1, 1, 1};
  std::vector<int> sortedIndicesHostData = {0, 1, 2};
  std::vector<float> permutedTokensGradHostData = {0, 0, 0, 0, 0, 0};
  std::vector<float> probsGradHostData = {0, 0, 0};

  ret = CreateAclTensor(permutedTokensHostData, permutedTokensShape,
                        &permutedTokensDeviceAddr, aclDataType::ACL_BF16,
                        &permutedTokens);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(unpermutedTokensGradHostData, unpermutedTokensGradShape, &unpermutedTokensGradDeviceAddr,
                      aclDataType::ACL_BF16, &unpermutedTokensGrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(probsHostData, probsShape, &probsDeviceAddr,
                      aclDataType::ACL_BF16, &probs);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sortedIndicesHostData, sortedIndicesShape, &sortedIndicesDeviceAddr,
                      aclDataType::ACL_INT32, &sortedIndices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(permutedTokensGradHostData, permutedTokensGradShape, &permutedTokensGradDeviceAddr, aclDataType::ACL_BF16,
                        &permutedTokensGrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(probsGradHostData, probsGradShape, &probsGradDeviceAddr, aclDataType::ACL_BF16,
                        &probsGrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;

  // Call the first-phase API of aclnnMoeTokenUnpermuteGrad.
  ret = aclnnMoeTokenUnpermuteGradGetWorkspaceSize(permutedTokens, unpermutedTokensGrad, sortedIndices, probs, paddedMode, nullptr,
                                               permutedTokensGrad, probsGrad, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnMoeTokenUnpermuteGradGetWorkspaceSize failed. ERROR: %d\n", ret);
      return ret);

  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // Call the second-phase API of aclnnMoeTokenUnpermuteGrad.
  ret = aclnnMoeTokenUnpermuteGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnMoeTokenUnpermuteGrad failed. ERROR: %d\n", ret);
            return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  PrintOutResult(permutedTokensGradShape, &permutedTokensGradDeviceAddr);
  PrintOutResult(probsGradShape, &probsGradDeviceAddr);

  // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
  aclDestroyTensor(permutedTokens);
  aclDestroyTensor(unpermutedTokensGrad);
  aclDestroyTensor(sortedIndices);
  aclDestroyTensor(probs);
  aclDestroyTensor(permutedTokensGrad);
  aclDestroyTensor(probsGrad);

  // 7. Release device resources.
  aclrtFree(permutedTokensDeviceAddr);
  aclrtFree(unpermutedTokensGradDeviceAddr);
  aclrtFree(probsDeviceAddr);
  aclrtFree(sortedIndicesDeviceAddr);
  aclrtFree(permutedTokensGradDeviceAddr);
  aclrtFree(probsGradDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
