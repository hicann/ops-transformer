# aclnnMoeTokenPermuteGrad

## Product Support

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- **Description**: Performs the backpropagation of [aclnnMoeTokenPermute](../../../moe/moe_token_permute/docs/aclnnMoeTokenPermute_en.md).
- **Formula**:

  $$
  inputGrad = permutedOutputGrad.indexSelect(0, sortedIndices)
  $$
  
  $$
  inputGrad = inputGrad.reshape(-1, topK, hiddenSize)
  $$
  
  $$
  inputGrad = inputGrad.sum(dim = 1)
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnMoeTokenPermuteGradGetWorkspaceSize` is called to obtain the input parameters, the workspace size required for computation, and the executor that contains the operator computation process. Then, `aclnnMoeTokenPermuteGrad` is called to perform computation.

* `aclnnStatus aclnnMoeTokenPermuteGradGetWorkspaceSize(const aclTensor *permutedOutputGrad, const aclTensor *sortedIndices, int64_t numTopk, bool paddedMode, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnMoeTokenPermuteGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnMoeTokenPermuteGradGetWorkspaceSize

- **Parameters:**
  - `permutedOutputGrad` (aclTensor\*, computation input): aclTensor on the device, which is the gradient of the forward output `permutedTokens`. It must be a 2D tensor with shape (tokens_num * topK_num, hidden_size). `tokens_num` indicates the number of tokens, and `topK_num` indicates the value of `numTopk`. The data type can be BFLOAT16, FLOAT16, or FLOAT32. The [data format](../../../docs/en/context/data_format.md) must be ND.
  - `sortedIndices` (aclTensor\*, computation input): aclTensor on the device, whose shape is (`tokens_num * topK_num`). The data type can be INT32. The value range is [0, tokens_num* topK_num – 1], and there is no duplicate index. The [data format](../../../docs/en/context/data_format.md) must be ND.
  - `numTopk` (int64_t, computation input): number of selected experts.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The value of `numTopk` is less than or equal to `512`.
  - `paddedMode` (bool, computation input): `true` indicates that `paddedMode` is enabled, and `false` indicates that `paddedMode` is disabled. Currently, only `false` is supported.
  - `out` (aclTensor\*, computation output): gradient of the input token. The value must be a 2D tensor with shape (tokens_num, hidden_size). The data type is the same as that of `permutedOutputGrad`, supporting BFLOAT16, FLOAT16, and FLOAT32. The [data format](../../../docs/en/context/data_format.md) must be ND.
  - `workspaceSize` (uint64\_t\*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor\*\*, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```cpp
  The first-phase API implements input parameter verification. The following errors may be thrown.
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The input or output tensor is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type or format of the input or output is not supported.
  ```

## aclnnMoeTokenPermuteGrad

- **Parameters:**
  - `workspace` (void\*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64\_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnMoeTokenPermuteGradGetWorkspaceSize`.
  - `executor` (aclOpExecutor\*, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnMoeTokenPermuteGrad` defaults to deterministic implementation.

- <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The value of `numTopk` is less than or equal to `512`.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp

#include "acl/acl.h"
#include "aclnnop/aclnn_moe_token_permute_grad.h"
#include <iostream>
#include <vector>

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

  int64_t num_topk = 2;
  std::vector<float> permuted_output_grad_Data = {1, 2, 3, 4};
  std::vector<int64_t> permuted_output_grad_Shape = {2, 2};
  void *permuted_output_grad_Addr = nullptr;
  aclTensor *permuted_output_grad = nullptr;

  ret = CreateAclTensor(permuted_output_grad_Data, permuted_output_grad_Shape,
                        &permuted_output_grad_Addr, aclDataType::ACL_BF16,
                        &permuted_output_grad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<float> sortedIndicesData = {0, 1};
  std::vector<int64_t> sortedIndicesShape = {2};
  void *sortedIndicesAddr = nullptr;
  aclTensor *sortedIndices = nullptr;

  ret = CreateAclTensor(sortedIndicesData, sortedIndicesShape, &sortedIndicesAddr,
                      aclDataType::ACL_INT32, &sortedIndices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<float> outData = {0, 0};
  std::vector<int64_t> outShape = {1, 2};
  void *outAddr = nullptr;
  aclTensor *out = nullptr;

  ret = CreateAclTensor(outData, outShape, &outAddr, aclDataType::ACL_BF16,
                        &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;

  // Call the first-phase API of aclnnMoeTokenPermuteGrad.
  ret = aclnnMoeTokenPermuteGradGetWorkspaceSize(permuted_output_grad, sortedIndices,
                                                 num_topk, false,
                                                 out, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnMoeTokenPermuteGradGetWorkspaceSize failed. ERROR: %d\n",
                ret);
      return ret);

  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // Call the second-phase API of aclnnMoeTokenPermuteGrad.
  ret = aclnnMoeTokenPermuteGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnMoeTokenPermuteGrad failed. ERROR: %d\n", ret);
            return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  PrintOutResult(outShape, &outAddr);

  // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
  aclDestroyTensor(permuted_output_grad);
  aclDestroyTensor(sortedIndices);
  aclDestroyTensor(out);

  // 7. Release device resources.
  aclrtFree(permuted_output_grad_Addr);
  aclrtFree(sortedIndicesAddr);
  aclrtFree(outAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
