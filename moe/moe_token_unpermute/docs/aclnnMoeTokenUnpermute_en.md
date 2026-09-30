# aclnnMoeTokenUnpermute

## Product Support

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- **Description**: Obtains the input data of `permutedTokens` based on the subscripts stored in `sortedIndices`. If `probs` data exists, `permutedTokens` is multiplied by `probs`. Then, this operator computes the cumulative sum and outputs the computation result.

- **Formula:**

  - If `probs` is not set to `None`, the formula is as follows:
    
    $$
    T[k] = T[S[k]]
    $$
    
    $$
    T[k] = T[k] * P[i][j]
    $$

    $$
    O[i] = \sum_{k=i*topK}^{(i+1)*topK - 1 } T[k]
    $$
    
    where $i \in {0,1,...,tokens-1}$; $j \in {0,1,...,topK-1}$; $k \in {0,1,...,tokens*topK-1}$; `T` indicates `permutedTokens`; `S` indicates `sortedIndices`; `P` indicates `probs`; `O` indicates `out`; `topK` indicates `topK\_num`; `tokens` indicates `tokens_num`.

  - If `probs` is set to `None`, `topK\_num` is `1`. The formula is as follows:

    $$
    T[i] = T[S[i]]
    $$

    $$
    O[i] = T[i]
    $$

    where $i \in {0,1,...,tokens-1}$; `T` indicates `permutedTokens`; `S` indicates `sortedIndices`; `O` indicates `out`; `tokens` indicates `tokens_num`.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnMoeTokenUnpermuteGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnMoeTokenUnpermute` is called to perform computation.

* `aclnnStatus aclnnMoeTokenUnpermuteGetWorkspaceSize(const aclTensor *permutedTokens, const aclTensor *sortedIndices, const aclTensor *probsOptional, bool paddedMode, const aclIntArray *restoreShapeOptional, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`

* `aclnnStatus aclnnMoeTokenUnpermute(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnMoeTokenUnpermuteGetWorkspaceSize

- **Parameters:**
    - `permutedTokens` (aclTensor*, computation input): input data. The shape is (`tokens_num * topK_num, hidden_size`). The data type can be BFLOAT16, FLOAT16, or FLOAT32. The [data format](../../../docs/en/context/data_format.md) can be ND. Non-contiguous input is supported.

    - `sortedIndices` (aclTensor*, computation input): position of the data to be computed in `permutedTokens`. The shape is (`tokens_num * topK_num`). The value range is [`0, tokens_num * topK_num – 1`], and there is no duplicate index. The data type can be INT32. The [data format](../../../docs/en/context/data_format.md) can be ND. Non-contiguous input is supported.

    - `probsOptional` (aclTensor*, optional computation input): optional input. When `probs` is passed, the value of `topK_num` equals the second dimension of `probs`. When `probs` is not passed, the value of `topK_num` is `1`. The shape is ( tokens_num, topK_num). The data type can be BFLOAT16, FLOAT16, or FLOAT32. The [data format](../../../docs/en/context/data_format.md) can be ND. Non-contiguous input is supported.

    - `paddedMode` (bool, computation input): `true` indicates that `paddedMode` is enabled, and `false` indicates that `paddedMode` is disabled. For details about `paddedMode`, see the `restoreShapeOptional` parameter. Currently, only `false` is supported.

    - `restoreShapeOptional` (aclIntArray*, computation input): This parameter takes effect only when `paddedMode` is set to `true`. Otherwise, no operation is performed on this parameter. When `paddedMode` is set to `true`, the shape of `out` is represented as `restoreShapeOptional`. Currently, only `nullptr` is supported.

    - `out` (aclTensor*, computation output): output result. When `paddedMode` is set to `false`, the shape is ( tokens_num, hidden_size). When `paddedMode` is set to `true`, the shape is the same as that of `restoreShapeOptional`. The data type is the same as that of `permutedTokens`, supporting BFLOAT16, FLOAT16, and FLOAT32. The [data format](../../../docs/en/context/data_format.md) can be ND. Non-contiguous output is not supported.

    - `workspaceSize` (uint64\_t\*, output): size of the workspace to be allocated on the device.

    - `executor` (aclOpExecutor\*\*, output): operator executor, containing the operator computation process.

- **Returns:**

    `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
    
    ```cpp
    The first-phase API implements input parameter verification. The following errors may be thrown:
    161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The input and output tensors are null pointers.
    161002 (ACLNN_ERR_PARAM_INVALID): 1. The input and output data types are not supported.
    ```

## aclnnMoeTokenUnpermute

- **Parameters:**
    - `workspace` (void\*, input): address of the workspace to be allocated on the device.
    - `workspaceSize` (uint64\_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnMoeTokenUnpermuteGetWorkspaceSize`.
    - `executor` (aclOpExecutor\*, input): operator executor, containing the operator computation process.
    - `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

    `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnMoeTokenUnpermute` defaults to deterministic implementation.
- <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The value of `topK_num` is less than or equal to `512`.
- <term>Atlas inference products</term>:
  - The data types supported by `permutedTokens` and `probsOptional` are FLOAT16 and FLOAT32.
  - The value of `topK_num` is less than or equal to `512`.
  - The value of `hiddenSize` must be a multiple of 128 and less than `10240`.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp

#include "acl/acl.h"
#include "aclnnop/aclnn_moe_token_unpermute.h"
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

  std::vector<float> permutedTokensData = {1, 2, 3, 4};
  std::vector<int64_t> permutedTokensShape = {2, 2};
  void *permutedTokensAddr = nullptr;
  aclTensor *permutedTokens = nullptr;

  ret = CreateAclTensor(permutedTokensData, permutedTokensShape,
                        &permutedTokensAddr, aclDataType::ACL_BF16,
                        &permutedTokens);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<float> sortedIndicesData = {0,1};
  std::vector<int64_t> sortedIndicesShape = {2};
  void *sortedIndicesAddr = nullptr;
  aclTensor *sortedIndices = nullptr;

  ret =
      CreateAclTensor(sortedIndicesData, sortedIndicesShape, &sortedIndicesAddr,
                      aclDataType::ACL_INT32, &sortedIndices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<float> probsOptionalData = {1, 1};
  std::vector<int64_t> probsOptionalShape = {1, 2};
  void *probsOptionalAddr = nullptr;
  aclTensor *probsOptional = nullptr;

  ret =
      CreateAclTensor(probsOptionalData, probsOptionalShape, &probsOptionalAddr,
                      aclDataType::ACL_BF16, &probsOptional);
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

  // Call the first-phase API of aclnnMoeTokenUnpermute.
  ret = aclnnMoeTokenUnpermuteGetWorkspaceSize(permutedTokens, sortedIndices,
                                               probsOptional, false, nullptr,
                                               out, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnMoeTokenUnpermuteGetWorkspaceSize failed. ERROR: %d\n",
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

  // Call the second-phase API of aclnnMoeTokenUnpermute.
  ret = aclnnMoeTokenUnpermute(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnMoeTokenUnpermute failed. ERROR: %d\n", ret);
            return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  PrintOutResult(outShape, &outAddr);

  // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
  aclDestroyTensor(permutedTokens);
  aclDestroyTensor(sortedIndices);
  aclDestroyTensor(probsOptional);
  aclDestroyTensor(out);

  // 7. Release device resources.
  aclrtFree(permutedTokensAddr);
  aclrtFree(sortedIndicesAddr);
  aclrtFree(probsOptionalAddr);
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
