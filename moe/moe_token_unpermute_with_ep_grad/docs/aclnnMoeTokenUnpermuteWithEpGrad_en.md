# aclnnMoeTokenUnpermuteWithEpGrad

## Product Support

| Product                                                        |  Supported  |
| :----------------------------------------------------------- |:-------:|
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference products</term>                            |    ×    |
| <term>Atlas training products</term>                             |    ×    |

## Function

- **Description**: Computes the backpropagation of `aclnnMoeTokenUnpermuteWithEp`.
- **Formula**:

  - If `probs` is not set to `None`, the formula is as follows, where $i \in {0, 1, 2, ..., num\_tokens – 1}$:
    - First, calculate `unpermutedTokens`.
      - When the condition rangeOptional[0] ≤ sortedIndices[i] < rangeOptional[1] is met,
      
        $$
        unpermutedTokens[i] = permutedTokensOptional[sortedIndices[i]-rangeOptional[0]]
        $$

      - Otherwise,
        
        $$
        unpermutedTokens[i] = 0
        $$
    
    - Then, compute the following:

      $$
      unpermutedTokens = unpermutedTokens.reshape(-1, topkNum, hiddenSize)
      $$
      
      $$
      unpermutedTokens = unpermutedTokensGrad.unsqueeze(1) * unpermutedTokens
      $$
      
      $$
      probsGrad = \sum_{k=0}^{topkNum}(unpermutedTokens_{i,j,k})
      $$
    
    - Finally, when the condition rangeOptional[0] ≤ sortedIndices[i] < rangeOptional[1] is met,
      
      $$
      permutedTokensGradOut[sortedIndices[i]] = ((unpermutedTokensGrad.unsqueeze(1) * probs.unsqueeze(-1)).reshape(-1, hiddenSize))[i]
      $$

  - If `probs` is set to `None`, the formula is as follows, where $i \in {0, 1, 2, ..., num\_tokens – 1}$:
    - When the condition rangeOptional[0] ≤ sortedIndices[i] < rangeOptional[1] is met,
    
    $$
    permutedTokensGradOut[sortedIndices[i]-rangeOptional[0]] = unpermutedOutputGrad[i]
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnMoeTokenUnpermuteWithEpGradGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnMoeTokenUnpermuteWithEpGrad` is called to perform computation.

* `aclnnStatus aclnnMoeTokenUnpermuteWithEpGradGetWorkspaceSize(const aclTensor *unpermutedTokensGrad, const aclTensor *sortedIndices, const aclTensor *permutedTokensOptional, const aclTensor *probsOptional, bool paddedMode, const aclIntArray *restoreShapeOptional, const aclIntArray *rangeOptional, int64_t topkNum, const aclTensor *permutedTokensGradOut, const aclTensor *probsGradOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnMoeTokenUnpermuteWithEpGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnMoeTokenUnpermuteWithEpGradGetWorkspaceSize

- **Parameters:**
  
  - `unpermutedTokensGrad` (aclTensor\*, computation input): aclTensor on the device, that is, `unpermutedTokensGrad` in the formula. It indicates the gradient of the forward output `unpermutedTokens`. The value must be a 2D tensor with shape (tokens_num, hidden_size). `tokens_num` indicates the number of tokens, and `hidden_size` indicates the token dimension size. The data type can be BFLOAT16, FLOAT16, or FLOAT32. The [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported.
  - `sortedIndices` (aclTensor\*, computation input): aclTensor on the device, that is, `sortedIndices` in the formula. The value must be a 1D shape with size (tokens_num \* topkNum). The data type can be INT32. The [data format](../../../docs/en/context/data_format.md) must be ND. The index value range is [0, tokens_num \* topkNum – 1]. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported.
  - `permutedTokensOptional` (aclTensor\*, computation input): aclTensor on the device, that is, `permutedTokensOptional` in the formula. This input is optional. The value must be a 2D tensor with shape (tokens_num \* topkNum, hidden_size), where the value of `topkNum` is less than or equal to `512`. The data type is the same as that of `unpermutedTokensGrad`. The [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported.
  - `probsOptional` (aclTensor\*, computation input): aclTensor on the device, that is, `probsOptional` in the formula. This input is optional. The value must be a 2D shape with size (tokens_num, topkNum). The data type can be BFLOAT16, FLOAT16, or FLOAT32. The [data format](../../../docs/en/context/data_format.md) must be ND. When `probs` is passed, the value of `topkNum` equals the second dimension of `probs`. When `probs` is not passed, the value of `topkNum` is `1`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported.
  - `paddedMode` (bool, computation input): `paddedMode` in the formula. The value `true` indicates that `paddedMode` is enabled, and the value `false` indicates that `paddedMode` is disabled. For details about `paddedMode`, see the description of the `restoreShapeOptional` parameter. Currently, only `false` is supported.
  - `restoreShapeOptional` (aclIntArray*, computation input): `restoreShapeOptional` in the formula. This parameter takes effect only when `paddedMode` is set to `true`. Otherwise, no operation is performed on this parameter. When `paddedMode` is set to `true`, the shape is the same as that of `unpermutedTokens`. Currently, only `nullptr` is supported.
  - `rangeOptional` (aclIntArray\*, computation input): `rangeOptional` in the formula, which indicates the valid range of EP slicing. The start position represented by `rangeOptional[0]` must be less than the end position represented by `rangeOptional[1]`. The size is `2`. This parameter does not take effect when it is left empty.
  - `topkNum` (int64_t, computation input): `topkNum` in the formula, which indicates the number of experts selected for each token.
  - `permutedTokensGradOut` (aclTensor\*, computation output): gradient of the input `permutedTokens`. The value must be a 2D tensor with shape (tokens_num \* topkNum, hidden_size). The data type is the same as that of `permutedTokensOptional`. The data type can be BFLOAT16, FLOAT16, or FLOAT32. The [data format](../../../docs/en/context/data_format.md) must be ND. Non-contiguous output is not supported.
  - `probsGradOut` (aclTensor\*, computation output): optional output, which is the gradient of the input `probs`. The value must be a 2D tensor with shape (tokens_num, topkNum). The data type is the same as that of `probsOptional`. The data type can be BFLOAT16, FLOAT16, or FLOAT32. The [data format](../../../docs/en/context/data_format.md) must be ND. Non-contiguous output is not supported.
  - `workspaceSize` (uint64\_t \*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor \*\*, output): operator executor, containing the operator computation process.
- **Returns:**
  
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
  
  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The required input or output tensors are null pointers.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The input and output data types are not supported.
  561002 (ACLNN_ERR_INNER_TILING_ERROR): 1. The value of topK_num is greater than 512.
                                        2. The input and output shapes do not meet the requirements.
                                        3. rangeOptional[1] < rangeOptional[0]
  ```

## aclnnMoeTokenUnpermuteWithEpGrad

- **Parameters:**
  
  - `workspace` (void\*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64\_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnMoeTokenUnpermuteWithEpGradGetWorkspaceSize`.
  - `executor` (aclOpExecutor\*, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.
- **Returns:**
  
    `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnMoeTokenUnpermuteWithEpGrad` defaults to deterministic implementation.
- The value of `topkNum` is less than or equal to `512`.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_token_unpermute_with_ep_grad.h"
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

  // Call the first-phase API of aclnnMoeTokenUnpermuteWithEpGrad.
  ret = aclnnMoeTokenUnpermuteWithEpGradGetWorkspaceSize(unpermutedTokensGrad, sortedIndices,permutedTokens, probs, paddedMode, nullptr, nullptr, 1, permutedTokensGrad, probsGrad, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnMoeTokenUnpermuteWithEpGradGetWorkspaceSize failed. ERROR: %d\n", ret);
      return ret);

  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // Call the second-phase API of aclnnMoeTokenUnpermuteWithEpGrad.
  ret = aclnnMoeTokenUnpermuteWithEpGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnMoeTokenUnpermuteWithEpGrad failed. ERROR: %d\n", ret);
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
