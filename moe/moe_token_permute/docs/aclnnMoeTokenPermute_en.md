# aclnnMoeTokenPermute

## Product Support

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- **Description**: This operator is used for permutation computation of MoE. It broadcasts and sorts tokens based on indexes (`indices`).
- **Formula**:
  - When `paddedMode` is set to `false`:
  
    $$
    sortedIndicesFirst=argSort(Indices)
    $$
  
    $$
    sortedIndicesOut=argSort(sortedIndicesFirst)
    $$
  
    $$
    permuteTokens[sortedIndicesFirst[i]]=tokens[i//topK]
    $$
  
  - When `paddedMode` is set to `true`:
  
    $$
    permuteTokensOut[i]=tokens[Indices[i]]
    $$
  
    $$
    sortedIndicesOut=Indices
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnMoeTokenPermuteGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnMoeTokenPermute` is called to perform computation.

* `aclnnStatus aclnnMoeTokenPermuteGetWorkspaceSize(const aclTensor *tokens, const aclTensor *indices, int64_t numOutTokens, bool paddedMode, const aclTensor *permuteTokensOut, const aclTensor *sortedIndicesOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnMoeTokenPermute(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnMoeTokenPermuteGetWorkspaceSize

- **Parameters:**
  
  - `tokens` (aclTensor\*, computation input): input token. It must be a tensor with two or more dimensions. The size of the first dimension is specified by `num_tokens`. The data type can be FLOAT16, BFLOAT16, or FLOAT32. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) must be ND.
  - `indices` (aclTensor\*, computation input): input indices. The shape must be 2D or 1D. When `paddedMode` is set to `false`, it indicates the *topK* processing expert indexes corresponding to each input token. The shape is (num\_tokens, topK) or (num\_tokens). When `paddedMode` is set to `true`, it indicates the token indexes (not supported currently) selected by each expert. The data type can be INT32 or INT64. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) must be ND. The number of elements must be greater than or equal to `0` and less than `16777215`. (The maximum or minimum values of a 32-bit integer or a 64-bit integer is supported.)
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The value of `topK` is less than or equal to `512`.
  - `numOutTokens` (int64\_t, computation input): number of valid output tokens. If this parameter is set to `0`, no token is deleted. If this parameter is not set to `0`, the tokens sorted by `indices` and exceeding the value specified by `numOutTokens` are dropped. If this parameter is set to a negative number, tokens are processed according to negative slice index rules.
  - `paddedMode` (bool, computation input): If `paddedMode` is set to `true`, `indices` has been padded with the token indexes selected by each expert. In this case, `indices` is not sorted. Currently, `paddedMode` can only be set to `false`.
  - `permuteTokensOut` (aclTensor\*, computation output): tokens that are extended and ordered based on `indices`. The value must be a tensor with two or more dimensions. The size of the first dimension is min\(num\_tokens \* topK, numOutTokens\). The product of all non-first dimensions is the same as the corresponding product in tokens (excluding the first dimension). The data type is the same as that of `tokens`, supporting FLOAT16, BFLOAT16, and FLOAT32. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are not supported. The [data format](../../../docs/en/context/data_format.md) must be ND.
  - `sortedIndicesOut` (aclTensor\*, computation output): mapping between `permuteTokensOut` and `tokens`. The value must be a 1D tensor with shape (num\_tokens\*topK,). The data type is INT32. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are not supported. The [data format](../../../docs/en/context/data_format.md) must be ND.
  - `workspaceSize` (uint64\_t\*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor\*\*, output): operator executor, containing the operator computation process.
- **Returns:**
  
  `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```cpp
  The first-phase API performs input parameter validation. The following errors may be returned:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The input and output tensors are null pointers.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The input and output data types are not supported.
  ```

## aclnnMoeTokenPermute

- **Parameters:**
  - `workspace` (void\*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64\_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnMoeTokenPermuteGetWorkspaceSize`.
  - `executor` (aclOpExecutor\*, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.
- **Returns:**
  
  `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnMoeTokenPermute` defaults to deterministic implementation.

- `indices` requires that the number of elements be less than `16777215`, and the value ranges from `0` to `16777215` (excluded). (The maximum or minimum value of int-32 or int-64 integer is supported. If the value is not within the range, the sorting result is incorrect.)
- `paddedMode` cannot be set to `True`.
- <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The value of `topK` is less than or equal to `512`.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_token_permute.h"
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
    std::vector<int64_t> xShape = {3, 4};
    std::vector<int64_t> idxShape = {3, 2};
    std::vector<int64_t> expandedXOutShape = {6, 4};
    std::vector<int64_t> idxOutShape = {6};
    void* xDeviceAddr = nullptr;
    void* indicesDeviceAddr = nullptr;
    void* expandedXOutDeviceAddr = nullptr;
    void* sortedIndicesOutDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* indices = nullptr;
    int64_t numTokenOut = 0;
    bool padMode = false;

    aclTensor* expandedXOut = nullptr;
    aclTensor* sortedIndicesOut = nullptr;
    std::vector<float> xHostData = {0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3};
    std::vector<int> indicesHostData = {1, 2, 0, 1, 0, 2};
    std::vector<float> expandedXOutHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int> sortedIndicesOutHostData = {0, 0, 0, 0, 0, 0};
    // Create a self aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(indicesHostData, idxShape, &indicesDeviceAddr, aclDataType::ACL_INT32, &indices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an out aclTensor.
    ret = CreateAclTensor(expandedXOutHostData, expandedXOutShape, &expandedXOutDeviceAddr, aclDataType::ACL_BF16, &expandedXOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sortedIndicesOutHostData, idxOutShape, &sortedIndicesOutDeviceAddr, aclDataType::ACL_INT32, &sortedIndicesOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnMoeTokenPermute.
    ret = aclnnMoeTokenPermuteGetWorkspaceSize(x, indices, numTokenOut, padMode, expandedXOut, sortedIndicesOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTokenPermuteGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnMoeTokenPermute.
    ret = aclnnMoeTokenPermute(workspaceAddr, workspaceSize, executor, stream);
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
    // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
    aclDestroyTensor(x);
    aclDestroyTensor(indices);
    aclDestroyTensor(expandedXOut);
    aclDestroyTensor(sortedIndicesOut);

    // 7. Release device resources. Modify the code based on the API definition.
    aclrtFree(xDeviceAddr);
    aclrtFree(indicesDeviceAddr);
    aclrtFree(expandedXOutDeviceAddr);
    aclrtFree(sortedIndicesOutDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
