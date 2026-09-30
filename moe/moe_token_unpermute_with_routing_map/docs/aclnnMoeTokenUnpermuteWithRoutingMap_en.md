# aclnnMoeTokenUnpermuteWithRoutingMap

## Product Support

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- **Description**: For `permutedTokens` processed by `aclnnMoeTokenpermuteWithRoutingMap`, accumulates them back to the original `unpermutedTokens`. This operator retrieves the input data stored in `permutedTokens` based on the subscripts stored in `sortedIndices`. If `probs` data exists, `permutedTokens` is multiplied by `probs`. Then, this operator computes the cumulative sum and outputs the computation result.
- **Formula**:
  
  $$
  topK\_num= permutedTokens.size(0) // routingMapOptional.size(0)
  $$

  $$
  numExperts = probs.size(1)
  $$

  $$
  numTokens = probs.size(0)
  $$

  $$
  capacity = sortedIndices.size(0) // numExperts
  $$

  (1) When `probs` is not set to `None` and `padMode` is set to `true`:

  $$
  permuteProbs  [i//capacity,sortedIndices[i]]=probs[i]
  $$

  $$
  permutedTokens = permutedTokens  * permuteProbs
  $$

  $$
  unpermutedTokens= zeros(restoreShape, dtype=permutedTokens.dtype, device=permutedTokens.device)
  $$

  $$
  permuteTokenId, outIndex= sortedIndices.sort(dim=-1)
  $$

  $$
  unpermutedTokens[permuteTokenId[i]] += permutedTokens[outIndex[i]]
  $$

  (2) When `probs` is not set to `None` and `padMode` is set to `false`:

  $$
  permuteProbs = probs.T.maskedSelect(routingMap.T)
  $$

  $$
  permutedTokens = permutedTokens  * permuteProbs
  $$

  $$
  unpermutedTokens= zeros(restoreShape, dtype=permutedTokens.dtype, device=permutedTokens.device)
  $$

  $$
  unpermutedTokens[i//topK\_num] += permutedTokens[sortedIndices[i]]
  $$

  (3) When `probs` is set to `None` and `padMode` is set to `true`:

  $$
  permuteTokenId, outIndex= sortedIndices.sort(dim=-1)
  $$

  $$
  unpermutedTokens[permuteTokenId[i]] += permutedTokens[outIndex[i]]
  $$

  (4) When `probs` is set to `None` and `padMode` is set to `false`

  $$
  unpermutedTokens[i//topK\_num] += permutedTokens[sortedIndices[i]]
  $$

## Prototype
  
  Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnMoeTokenUnpermuteWithRoutingMapGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor covering the operator computation process. Then, `aclnnMoeTokenUnpermuteWithRoutingMap` is called to perform computation.
  
* `aclnnStatus aclnnMoeTokenUnpermuteWithRoutingMapGetWorkspaceSize(const aclTensor *permutedTokens, const aclTensor *sortedIndices, const aclTensor* routingMapOptional, const aclTensor *probsOptional, bool paddedMode, const aclIntArray *restoreShapeOptional, aclTensor *unpermutedTokens, aclTensor *outIndex, aclTensor *permuteTokenId, aclTensor *permuteProbs, uint64_t *workspaceSize, aclOpExecutor **executor);`
* `aclnnStatus aclnnMoeTokenUnpermuteWithRoutingMap(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)`
  
## aclnnMoeTokenUnpermuteWithRoutingMapGetWorkspaceSize
  
  - **Parameters:**
    
    - `permutedTokens` (aclTensor*, computation input): aclTensor on the device, which indicates input tokens. The value must be a 2D tensor. When `paddedMode` is set to `false`, the shape is (`tokens_num * topK_num, hidden_size`). When `paddedMode` is set to `true`, the shape is (`experts_num * capacity, hidden_size`). `capacity` indicates the number of tokens that can be processed by each expert. The data type can be BFLOAT16, FLOAT16, or FLOAT. The [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported.
    - `sortedIndices` (aclTensor\*, computation input): aclTensor on the device. In drop/pad-less mode, the value must be a 1D shape with size (tokens_num \* topK_num,). The data type can be INT32. The [data format](../../../docs/en/context/data_format.md) must be ND. The index value range is [0, tokens_num \* topK_num – 1]. In drop/pad mode, the value must be a 1D tensor with shape (experts_num \* capacity). The data type can be INT32. The [data format](../../../docs/en/context/data_format.md) must be ND. The index value range is [0, tokens_num – 1]. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported.
    - routingMapOptional (aclTensor\*, computation input): aclTensor on the device. This input is optional. If the input `probsOptional` is a null pointer, this input is not required, and a null pointer should be passed. In the formula, `routingMapOptional` indicates whether the token at the corresponding position is processed by the corresponding expert. The value must be a 2D shape with size (tokens_num, experts_num). The data type can be INT8 or BOOL. If the data type is INT8, the value can be `0` or `1`. If the data type is BOOL, the value can be `true` or `false`. The [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported.
    - `probsOptional` (aclTensor\*, computation input): aclTensor on the device. This input is optional. If it is not required, pass a null pointer. In the formula, `probsOptional` indicates the weight of the token at a specified position processed by the corresponding expert in the final result. The shape is the same as that of `routingMapOptional`, the data type is the same as that of `permutedTokens`, and the [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported.
    - `paddedMode` (bool, computation input): input of the Boolean type on the host. This input is optional. The value can be `false` or `true`. The value `true` indicates that `paddedMode` is enabled, and the value `false` indicates that `paddedMode` is disabled. When `paddedMode` is enabled, the shapes of the outputs `outIndex` and `permuteTokenId` are (`experts_num * capacity,`). When `paddedMode` is disabled, each token is processed by a fixed number of experts (specified by `topK_num`), and the shapes of the outputs `outIndex` and `permuteTokenId` are (tokens_num * topK_num,).
    - `restoreShapeOptional` (aclIntArray*, computation input): aclIntArray on the host. The supported data type is INT64, and the size is `2`. The shape is the same as that `unpermutedTokens`.
    - `unpermutedTokens` (aclTensor*, computation output): aclTensor on the device, that is, `unpermutedTokens` in the formula. It is the forward output result. The value must be a 2D tensor with shape (tokens_num, hidden_size). The data type can be BFLOAT16, FLOAT16, or FLOAT. The [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported.
    - `outIndex` (aclTensor*, computation output): aclTensor on the device, that is, `outIndex` in the formula. When `paddedMode` is set to `false`, the value must be a 1D shape with size (`tokens_num * topK_num,`). The index value range is [`0, tokens_num * topK_num – 1`]. When `paddedMode` is set to `true`, the value must be a 1D tensor with shape (`experts_num * capacity,`). The index value range is [`0, experts_num * capacity – 1`]. The data type is INT32, and the [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported.
    - `permuteTokenId` (aclTensor*, computation output): aclTensor on the device, that is, `permuteTokenId` in the formula. When `paddedMode` is set to `false`, the value must be a 1D shape with size (`tokens_num * topK_num,`). When `paddedMode` is set to `true`, the value must be a 1D tensor with shape (`experts_num * capacity,`). The index value range is [0, tokens_num – 1]. The data type is INT32, and the [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported.
    - `permuteProbs` (aclTensor *, computation output): aclTensor on the device, that is, `permuteProbs` in the formula, which indicates that permuted `probs` is output. The shape can be 1D. The data type is the same as that of `probsOptional`. The [data format](../../../docs/en/context/data_format.md) must be ND.
    - `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.
    - `executor` (aclOpExecutor**, output): operator executor, containing the operator computation process.
  - **Returns:**
    
    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
    
    ```text
    The first-phase API implements input parameter verification. The following errors may be thrown:
    161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The required input or output tensor is a null pointer.
    161002 (ACLNN_ERR_PARAM_INVALID): 1. The input or output data types are not supported.
    561002 (ACLNN_ERR_INNER_TILING_ERROR): 1. The value of topK_num is greater than 512.
                                          2. The value of topK_num is greater than that of experts_num.
                                          3. The value of capacity is greater than that of tokens_num.
                                          4. The input or output shape does not meet the requirements.
    ```
  
## aclnnMoeTokenUnpermuteWithRoutingMap
  
  - **Parameters:**
    
    - `workspace` (void\*, input): address of the workspace to be allocated on the device.
    - `workspaceSize` (uint64\_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnMoeTokenUnpermuteWithRoutingMapGetWorkspaceSize`.
    - `executor` (aclOpExecutor\*, input): operator executor, containing the operator computation process.
    - `stream` (aclrtStream, input): stream for executing the task.
  - **Returns:**
    
      `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
  
## Constraints

- Deterministic computation:
  - `aclnnMoeTokenUnpermuteWithRoutingMap` defaults to deterministic implementation.
- The value of `topkNum` is less than or equal to `512`. When `paddedMode` is set to `false`, the number of 1s or `true` values in each row of `routingMap` is fixed and less than `512`.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_token_unpermute_with_routing_map.h"
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
    std::vector<int64_t> permutedTokensShape = {2, 2};
    std::vector<int64_t> sortedIndicesShape = {2};
    std::vector<int64_t> routingMapOptionalShape = {2, 2};
    std::vector<int64_t> probsShape = {2, 2};
    std::vector<int64_t> unpermutedTokensShape = {2, 2};
    std::vector<int64_t> outIndexShape = {2};
    std::vector<int64_t> permuteTokenIdShape = {2};
    std::vector<int64_t> permuteProbsShape = {2};

    void* permutedTokensDeviceAddr = nullptr;
    void* sortedIndicesDeviceAddr = nullptr;
    void* routingMapOptionalDeviceAddr = nullptr;
    void* probsDeviceAddr = nullptr;
    void* unpermutedTokensDeviceAddr = nullptr;
    void* outIndexDeviceAddr = nullptr;
    void* permuteTokenIdDeviceAddr = nullptr;
    void* permuteProbsDeviceAddr = nullptr;
    //in
    aclTensor* permutedTokens = nullptr;
    aclTensor* sortedIndices = nullptr;
    aclTensor* routingMapOptional = nullptr;
    aclTensor* probs = nullptr;
    aclTensor* unpermutedTokens = nullptr;
    aclTensor* outIndex = nullptr;
    aclTensor* permuteTokenId = nullptr;
    aclTensor* permuteProbs = nullptr;
    bool padMode = true;
    std::vector<int64_t> restoreShapeOptionalData = {2, 2};
    aclIntArray *restoreShapeOptional = aclCreateIntArray(restoreShapeOptionalData.data(), restoreShapeOptionalData.size());

    // Construct data.
    std::vector<float> permutedTokensHostData = {1.0, 1.0, 1.0, 1.0};
    std::vector<int> sortedIndicesHostData = {1, 1};
    std::vector<char> routingMapOptionalHostData = {1, 1, 1, 1};
    std::vector<float> probsHostData = {1, 1, 1, 1};
    
    std::vector<float> unpermutedTokensHostData = {0, 0, 0, 0};
    std::vector<int> outIndexHostData = {0, 0};
    std::vector<int> permuteTokenIdHostData = {0, 0};
    std::vector<float> permuteProbsHostData = {0, 0};
    // Create a self aclTensor.
    ret = CreateAclTensor(permutedTokensHostData, permutedTokensShape, &permutedTokensDeviceAddr, aclDataType::ACL_FLOAT, &permutedTokens);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sortedIndicesHostData, sortedIndicesShape, &sortedIndicesDeviceAddr, aclDataType::ACL_INT32, &sortedIndices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(routingMapOptionalHostData, routingMapOptionalShape, &routingMapOptionalDeviceAddr, aclDataType::ACL_INT8, &routingMapOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(probsHostData, probsShape, &probsDeviceAddr, aclDataType::ACL_FLOAT, &probs);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an out aclTensor.
    ret = CreateAclTensor(unpermutedTokensHostData, unpermutedTokensShape, &unpermutedTokensDeviceAddr, aclDataType::ACL_FLOAT, &unpermutedTokens);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outIndexHostData, outIndexShape, &outIndexDeviceAddr, aclDataType::ACL_INT32, &outIndex);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(permuteTokenIdHostData, permuteTokenIdShape, &permuteTokenIdDeviceAddr, aclDataType::ACL_INT32, &permuteTokenId);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(permuteProbsHostData, permuteProbsShape, &permuteProbsDeviceAddr, aclDataType::ACL_FLOAT, &permuteProbs);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnMoeTokenUnpermuteWithRoutingMap.
    ret = aclnnMoeTokenUnpermuteWithRoutingMapGetWorkspaceSize(permutedTokens, sortedIndices, routingMapOptional, probs, padMode, restoreShapeOptional, 
                                                               unpermutedTokens, outIndex, permuteTokenId, permuteProbs, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTokenUnpermuteWithRoutingMapGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnMoeTokenUnpermuteWithRoutingMap(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTokenUnpermuteWithRoutingMapfailed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
    auto unpermutedTokensSize = GetShapeSize(unpermutedTokensShape);
    std::vector<float> unpermutedTokensData(unpermutedTokensSize, 0);
    ret = aclrtMemcpy(unpermutedTokensData.data(), unpermutedTokensData.size() * sizeof(unpermutedTokensData[0]), unpermutedTokensDeviceAddr, unpermutedTokensSize * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < unpermutedTokensSize; i++) {
        LOG_PRINT("unpermutedTokensData[%ld] is: %f\n", i, unpermutedTokensData[i]);
    }

    // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
    aclDestroyTensor(permutedTokens);
    aclDestroyTensor(sortedIndices);
    aclDestroyTensor(routingMapOptional);
    aclDestroyTensor(probs);
    aclDestroyTensor(unpermutedTokens);
    aclDestroyTensor(outIndex);
    aclDestroyTensor(permuteTokenId);
    aclDestroyTensor(permuteProbs);

    // 7. Release device resources. Modify the code based on the API definition.
    aclrtFree(permutedTokensDeviceAddr);
    aclrtFree(sortedIndicesDeviceAddr);
    aclrtFree(routingMapOptionalDeviceAddr);
    aclrtFree(probsDeviceAddr);
    aclrtFree(unpermutedTokensDeviceAddr);
    aclrtFree(outIndexDeviceAddr);
    aclrtFree(permuteTokenIdDeviceAddr);
    aclrtFree(permuteProbsDeviceAddr);

    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
