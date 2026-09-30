# aclnnMoeTokenPermuteWithRoutingMap

## Product Support

| Product                                                        |  Supported  |
| :----------------------------------------------------------- |:-------:|
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference products</term>                            |    ×    |
| <term>Atlas training products</term>                             |    ×    |

## Function

- **Description**: This operator is used for permutation computation in MoE. It passes tokens and expert labels as `routingMap` and sorts the tokens and optional `probsOptional` based on routing maps after broadcasting them.
- **Formula**:
  `tokens\_num` indicates the size of the zeroth dimension of `routingMap`, and `expert\_num` indicates the size of the first dimension of `routingMap`.
  When `dropAndPad` is set to `false`:
  
  $$
  expertIndex=arrange(tokens\_num).expand(expert\_num,-1)
  $$
  
  $$
  sortedIndicesFirst=expertIndex.maskedselect(routingMap.T)
  $$
  
  $$
  sortedIndicesOut=argSort(sortedIndicesFirst)
  $$
    
  $$
  topK = numOutTokens // tokens\_num
  $$
  
  $$
  outToken = topK * tokens\_num
  $$

  $$
  permuteTokens[sortedIndicesOut[i]]=tokens[i//topK]
  $$
  
  $$
  permuteProbsOutOptional=probsOptional.T.maskedselect(routingMap.T)
  $$
  
  When `dropAndPad` is set to `true`:

  $$
  capacity = numOutTokens // expert\_num
  $$
  
  $$
  outToken = capacity * expert\_num
  $$

  $$
  sortedIndicesOut = argsort(routingMap.T,dim=-1) [:, :capacity]
  $$
  
  $$
  permutedTokensOut = tokens.index_select(0, sorted_indices)
  $$
  
  If `probs` is not set to `none`:
  
  $$
  robs\_T\_1D = probsOptional.T.view(-1)
  $$
  
  $$
  indices\_dim0 = arange(num\_experts)
  $$
  
  $$
  indices\_dim1 = sorted_indices.view(expert\_num, capacity)
  $$
  
  $$
  indices\_1D = (indices_dim0 * tokens\_num + indices\_dim1).view(-1)
  $$
  
  $$
  permuteProbsOutOptional = probs\_T\_1D.index_select(0, indices_1D)
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnMoeTokenPermuteWithRoutingMapGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnMoeTokenPermuteWithRoutingMap` is called to perform computation.

* `aclnnStatus aclnnMoeTokenPermuteWithRoutingMapGetWorkspaceSize(const aclTensor *tokens, const aclTensor *routingMap, const aclTensor *probsOptional,  int64_t numOutTokens,  bool dropAndPad, aclTensor *permuteTokensOut, aclTensor *permuteProbsOutOptional, aclTensor *sortedIndicesOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnMoeTokenPermuteWithRoutingMap(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnMoeTokenPermuteWithRoutingMapGetWorkspaceSize

- **Parameters:**
  
  - `tokens` (aclTensor\*, computation input): aclTensor on the device, which indicates input tokens, that is, `tokens` in the formula. The value must be a 2D tensor with shape \(tokens\_num, hidden\_size). The data type can be BFLOAT16, FLOAT16, or FLOAT. The [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported.
  - `routingMap` (aclTensor\*, computation input): aclTensor on the device, which indicates the mapping between tokens and experts, that is, `routingMap` in the formula. The value must be a 2D shape with size (tokens_num, experts_num). The data type can be INT8 or BOOL. If the data type is INT8, the value can be `0` or `1`. If the data type is BOOL, the value can be `true` or `false`. The [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. In drop/pad-less mode, each row must contain *topK* `true` or `1` values.
  - `probsOptional` (aclTensor\*, computation input): aclTensor on the device. It is an optional input, that is, probsOptional in the formula. The number of elements specified by this parameter must be the same as that of `routingMap`. When `probsOptional` is left empty, the optional output `permuteProbsOutOptional` is also left empty. The data type is the same as that of `tokens`. The [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported.
  - `numOutTokens` (int64\_t, computation input): `numOutTokens` in the formula, which is used to compute the number of valid tokens specified by `topK` and `capacity` in the formula.
  - `dropAndPad` (bool, computation input): `dropAndPad` in the formula, indicating whether to enable the `dropAndPad` mode.
  - `permutedTokensOut` (aclTensor\*, computation output): aclTensor on the device, that is `permutedTokensOut` in the formula. It is used to extend, sort, and filter tokens based on `indices`. The value must be a 2D tensor with the shape \(`outToken`, `hidden_size`), that is, `outToken` in the formula. The data type is the same as that of `tokens`. The [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported.
  - `sortedIndicesOut` (aclTensor\*, computation output): mapping between `sortedIndicesOut`, `permute_tokens`, and `tokens` in the formula. It is an aclTensor on the device. The value must be a 1D tensor with shape \(outToken\), that is, `outToken` in the formula. The data type can be INT32. The [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported.
  - `permuteProbsOutOptional` (aclTensor\*, computation output): `permuteProbsOutOptional` in the formula. It is an aclTensor on the device. It is the `probsOptional` that has been sorted and filtered based on `indices`. The shape is \(outToken\), that is, `outToken` in the formula. The data type is the same as that of `probsOptional`. The [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported.
  - `workspaceSize` (uint64\_t\*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor \*\*, output): operator executor, containing the operator computation process.
- **Returns:**
  
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```cpp
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The input and output tensors are null pointers.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The input and output data types are not supported.
                                   2. The input and output shapes do not meet the requirements.
                                   3. The value of numOutTokens is less than 0 or greater than tokens_num * experts_num.
  561002 (ACLNN_ERR_INNER_TILING_ERROR): 1. The value of topkNum is greater than 512.
  ```

## aclnnMoeTokenPermuteWithRoutingMap

- **Parameters:**
  
  - `workspace` (void\*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64\_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API <code>aclnnMoeTokenPermuteWithRoutingMapGetWorkspaceSize</code>.
  - `executor` (aclOpExecutor\*, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.
- **Returns:**
  
    `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnMoeTokenPermuteWithRoutingMap` defaults to deterministic implementation.

- The values of `tokens_num` and `experts_num` must be less than `16777215`. When `paddedMode` is set to `false`, the number of 1s or `true` values in each row of `routingMap` is fixed and less than `512`.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_token_permute_with_routing_map.h"
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
    int64_t numTokenOut = 6;
    bool padMode = false;

    aclTensor* expandedXOut = nullptr;
    aclTensor* sortedIndicesOut = nullptr;
    std::vector<float> xHostData = {0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3};
    std::vector<int> indicesHostData = {1, 1, 1, 1, 1, 1};
    std::vector<float> expandedXOutHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int> sortedIndicesOutHostData = {0, 0, 0, 0, 0, 0};
    // Create a self aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(indicesHostData, idxShape, &indicesDeviceAddr, aclDataType::ACL_BOOL, &indices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an out aclTensor.
    ret = CreateAclTensor(expandedXOutHostData, expandedXOutShape, &expandedXOutDeviceAddr, aclDataType::ACL_BF16, &expandedXOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sortedIndicesOutHostData, idxOutShape, &sortedIndicesOutDeviceAddr, aclDataType::ACL_INT32, &sortedIndicesOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnMoeTokenPermuteWithRoutingMap.
    ret = aclnnMoeTokenPermuteWithRoutingMapGetWorkspaceSize(x, indices, nullptr, numTokenOut, padMode, expandedXOut, nullptr, sortedIndicesOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTokenPermuteWithRoutingMapGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnMoeTokenPermuteWithRoutingMap(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTokenPermuteWithRoutingMapfailed. ERROR: %d\n", ret); return ret);
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
