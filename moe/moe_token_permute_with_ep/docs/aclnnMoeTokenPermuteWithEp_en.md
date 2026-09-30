# aclnnMoeTokenPermuteWithEp

## Product Support

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |     ×      |

## Function

- **Description**: This operator is used for permutation computation of MoE. It broadcasts and sorts tokens and optional probs based on indexes (`indices`), and performing slicing based on the range specified by `rangeOptional`.
- **Formula**:
  - When `paddedMode` is set to `false`,

    $$
    sortedIndicesFirst=argSort(indices)
    $$

    $$
    sortedIndicesOut=argSort(sortedIndices)
    $$

    When the condition rangeOptional[0] ≤ sortedIndices[i] < rangeOptional[1] is met,

    $$
    permuteTokensOut[sortedIndices[i]-rangeOptional[0]]=tokens[i//topK]
    $$

    $$
    permuteProbsOut[sortedIndices[i]-rangeOptional[0]]=probsOptional[i]
    $$

  - When `paddedMode` is set to `true`,

    $$
    permuteTokensOut[i]=tokens[indices[i]]
    $$

    $$
    sortedIndicesOut=indices
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnMoeTokenPermuteWithEpGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnMoeTokenPermuteWithEp` is called to perform computation.

* `aclnnStatus aclnnMoeTokenPermuteWithEpGetWorkspaceSize(const aclTensor *tokens, const aclTensor *indices, const aclTensor *probsOptional, const aclIntArray *rangeOptional, int64_t numOutTokens, bool paddedMode, const aclTensor *permuteTokensOut, const aclTensor *sortedIndicesOut, const aclTensor *permuteProbsOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnMoeTokenPermuteWithEp(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnMoeTokenPermuteWithEpGetWorkspaceSize

- **Parameters:**

  - `tokens` (aclTensor *, computation input): input tokens in permute, that is, `tokens` in the formula, which is an aclTensor on the device. The 2D shape is supported, and the shape is (num\_tokens, hidden\_size). `num\_tokens` indicates the number of tokens, and `hidden\_size` indicates the length of each token. The data type can be BFLOAT16, FLOAT16, or FLOAT32. The [data format](../../../docs/en/context/data_format.md) can be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported, but empty tensors are not supported.
  - `indices` (aclTensor \*, computation input): expert indexes corresponding to input tokens, that is, `indices` in the formula, which is an aclTensor on the device. The shape supports 1D or 2D. When `paddedMode` is set to `false`, it indicates the indexes of *topK* processing experts corresponding to each input token. The shape is (num\_tokens, topK\_num) or (num\_tokens). When `paddedMode` is set to `true`, it indicates the token indexes (not supported currently) selected by each expert. The number of elements is less than `16777215`, and the value must be within the range of `0` to `16777215` (excluded), and the value of `topK_num` must be less than or equal to `512`. The supported data types are INT32 and INT64. The [data format](../../../docs/en/context/data_format.md) can be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported, but empty tensors are not supported.
  - `probsOptional` (aclTensor \*, computation input): the expert processing probability of input tokens, that is, `probsOptional` in the formula, which is an aclTensor on the device. Optional computation input, which corresponds to the computation output `permuteProbsOut`. If this parameter is left empty, `permuteProbsOut` is not output. Its shape is the same as that of `indices`. The data type can be BFLOAT16, FLOAT16, or FLOAT32. The [data format](../../../docs/en/context/data_format.md) can be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported.
  - `rangeOptional` (aclIntArray \*, computation input): valid range of Expert Parallelism (EP) slicing. The size is `2` (an array of two integers). If this parameter is left empty, `probsOptional` and `permuteTokensOut` are ignored, and the execution logic is rolled back to [aclnnMoeTokenPermute](../../../moe/moe_token_permute/docs/aclnnMoeTokenPermute_en.md).
  - `numOutTokens` (int64\_t, computation input): number of valid output tokens. This parameter is valid only when `rangeOptional` is left empty. When it is set to `0`, no token is deleted. If this parameter is not set to `0`, the tokens that are sorted by `indices` and the part exceeding the value specified by `numOutTokens` are dropped. If this parameter is set to a negative number, tokens are processed according to the rules for negative slice indexes.
  - `paddedMode` (bool, computation input): If `paddedMode` is set to `true`, `indices` have been padded with the token indexes selected by each expert. In this case, `indices` is not sorted. Currently, `paddedMode` can only be set to `false`.
  - `permuteTokensOut` (aclTensor \*, computation output): tokens that are extended and sorted based on `indices`, that is, `permuteTokensOut` in the formula, which is an aclTensor on the device. The shape supports 2D, and the shape is (rangeOptional[1] – rangeOptional[0], hidden\_size). The data type is the same as that of tokens. The [data format](../../../docs/en/context/data_format.md) is ND.
  - `sortedIndicesOut` (aclTensor \*, computation output): mapping between `permuteTokensOut` and `tokens`, that is, `sortedIndicesOut` in the formula, which is an aclTensor on the device. The 1D shape is supported, and the shape is (num\_tokens * topK\_num). The data type is INT32, and the [data format](../../../docs/en/context/data_format.md) must be ND.
  - `permuteProbsOut` (aclTensor \*, computation output): probs that are extended and sorted based on indices, that is, `permuteProbsOut` in the formula, which is an aclTensor on the device. The 1D shape is supported, and the shape is (rangeOptional[1] – rangeOptional[0]). The data type is the same as that of `tokens`. The [data format](../../../docs/en/context/data_format.md) must be ND.
  - `workspaceSize` (uint64\_t \*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor \*\*, output): operator executor, containing the operator computation process.
- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```cpp
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The input and output tensors are null pointers.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The input and output data types are not supported.
  ```

## aclnnMoeTokenPermuteWithEp

- **Parameters:**
  - `workspace` (void\*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64\_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnMoeTokenPermuteWithEpGetWorkspaceSize`.
  - `executor` (aclOpExecutor\*, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.
- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnMoeTokenPermuteWithEp` defaults to deterministic implementation.

- `indices` requires that the number of elements be less than `16777215`, and the value ranges from `0` to `16777215` (excluded). (The maximum or minimum value of int-32 or int-64 integer is supported. If the value is not within the range, the sorting result is incorrect.)
- The value of `topK` is less than or equal to `512`.
- `paddedMode` cannot be set to `True`.
- When `rangeOptional` is left empty, `probsOptional` and `permuteTokensOut` are ignored, and the execution logic is rolled back to [aclnnMoeTokenPermute](../../../moe/moe_token_permute/docs/aclnnMoeTokenPermute_en.md).

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_token_permute_with_ep.h"
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
int CreateAclIntArray(const std::vector<T>& hostData, void** deviceAddr, aclIntArray** intArray) {
  auto size = GetShapeSize(hostData) * sizeof(T);
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy the data on the host to the memory on the device.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Call aclCreateIntArray to create an aclIntArray.
  *intArray = aclCreateIntArray(hostData.data(), hostData.size());
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
    std::vector<int64_t> xShape = {3, 4};
    std::vector<int64_t> idxShape = {3, 2};
    std::vector<int64_t> probsShape = {3, 2};
    std::vector<int64_t> expandedXOutShape = {4, 4};
    std::vector<int64_t> idxOutShape = {6};
    std::vector<int64_t> expandedProbsOutShape = {4};

    void* xDeviceAddr = nullptr;
    void* indicesDeviceAddr = nullptr;
    void* probsDeviceAddr = nullptr;
    void* expandedXOutDeviceAddr = nullptr;
    void* sortedIndicesOutDeviceAddr = nullptr;
    void* expandedProbsOutDeviceAddr = nullptr;
    void* rangeDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* indices = nullptr;
    aclTensor* probs = nullptr;
    aclIntArray* range = nullptr;
    int64_t numTokenOut = 6;
    bool padMode = false;

    aclTensor* expandedXOut = nullptr;
    aclTensor* sortedIndicesOut = nullptr;
    aclTensor* expandedProbsOut = nullptr;
    std::vector<float> xHostData = {0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3};
    std::vector<int> indicesHostData = {1, 2, 3, 1, 2, 3};
    std::vector<float> probsHostData = {0.5, 0.3, 0.4, 0.2, 0.5, 0.4};
    std::vector<float> expandedXOutHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int> sortedIndicesOutHostData = {0, 0, 0, 0, 0, 0};
    std::vector<float> expandedProbsOutHostData = {0, 0, 0, 0};
    std::vector<int64_t> rangeHostData = {1, 5};
    // Create a self aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(indicesHostData, idxShape, &indicesDeviceAddr, aclDataType::ACL_INT32, &indices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(probsHostData, probsShape, &probsDeviceAddr, aclDataType::ACL_BF16, &probs);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an out aclTensor.
    ret = CreateAclTensor(expandedXOutHostData, expandedXOutShape, &expandedXOutDeviceAddr, aclDataType::ACL_BF16, &expandedXOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sortedIndicesOutHostData, idxOutShape, &sortedIndicesOutDeviceAddr, aclDataType::ACL_INT32, &sortedIndicesOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandedProbsOutHostData, expandedProbsOutShape, &expandedProbsOutDeviceAddr, aclDataType::ACL_BF16, &expandedProbsOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create related attr.
    ret = CreateAclIntArray(rangeHostData, &rangeDeviceAddr, &range);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnMoeTokenPermute.
    ret = aclnnMoeTokenPermuteWithEpGetWorkspaceSize(x, indices, probs, range, numTokenOut, padMode, expandedXOut, sortedIndicesOut, expandedProbsOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTokenPermuteGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnMoeTokenPermute.
    ret = aclnnMoeTokenPermuteWithEp(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTokenPermute failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
    auto expandedXSize = GetShapeSize(expandedXOutShape);
    std::vector<float> expandedXData(expandedXSize, 0);
    ret = aclrtMemcpy(expandedXData.data(), expandedXData.size() * sizeof(expandedXData[0]), expandedXOutDeviceAddr, expandedXSize * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expandedXSize; i++) {
        LOG_PRINT("expandedXData[%ld] is: %f\n", i, expandedXData[i]);
    }
    auto sortedIndicesSize = GetShapeSize(idxOutShape);
    std::vector<int> sortedIndicesData(sortedIndicesSize, 0);
    ret = aclrtMemcpy(sortedIndicesData.data(), sortedIndicesData.size() * sizeof(sortedIndicesData[0]), sortedIndicesOutDeviceAddr, sortedIndicesSize * sizeof(int32_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < sortedIndicesSize; i++) {
        LOG_PRINT("sortedIndicesData[%ld] is: %d\n", i, sortedIndicesData[i]);
    }
    auto expandedProbsSize = GetShapeSize(expandedProbsOutShape);
    std::vector<float> expandedProbsData(expandedProbsSize, 0);
    ret = aclrtMemcpy(expandedProbsData.data(), expandedProbsData.size() * sizeof(expandedProbsData[0]), expandedProbsOutDeviceAddr, expandedProbsSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expandedProbsSize; i++) {
        LOG_PRINT("expandedProbsData[%ld] is: %f\n", i, expandedProbsData[i]);
    }
    // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
    aclDestroyTensor(x);
    aclDestroyTensor(indices);
    aclDestroyTensor(probs);
    aclDestroyTensor(expandedXOut);
    aclDestroyTensor(sortedIndicesOut);
    aclDestroyTensor(expandedProbsOut);

    // 7. Release device resources. Modify the code based on the API definition.
    aclrtFree(xDeviceAddr);
    aclrtFree(indicesDeviceAddr);
    aclrtFree(probsDeviceAddr);
    aclrtFree(expandedXOutDeviceAddr);
    aclrtFree(sortedIndicesOutDeviceAddr);
    aclrtFree(expandedProbsOutDeviceAddr);
    aclrtFree(rangeDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
