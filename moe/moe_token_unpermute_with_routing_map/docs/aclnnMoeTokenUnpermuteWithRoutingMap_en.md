# aclnnMoeTokenUnpermuteWithRoutingMap

[📄 View Source Code](https://gitcode.com/cann/ops-transformer/tree/master/moe/moe_token_unpermute_with_routing_map)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- **Description**: For `permutedTokens` processed by `aclnnMoeTokenPermuteWithRoutingMap`, accumulates them back to the original `unpermutedTokens`. This operator retrieves the input data stored in `permutedTokens` based on the subscripts stored in `sortedIndices`. If `probs` data exists, `permutedTokens` is multiplied by `probs`. Then, this operator computes the cumulative sum and outputs the computation result.
- Formulas:
  
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

  (1) When probs is not None and paddedMode is true:

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

  (2) When probs is not None and paddedMode is false (T is the transpose operation):

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

  (3) When probs is None and paddedMode is true:

  $$
  permuteTokenId, outIndex= sortedIndices.sort(dim=-1)
  $$

  $$
  unpermutedTokens[permuteTokenId[i]] += permutedTokens[outIndex[i]]
  $$

  (4) When probs is None and paddedMode is false:

  $$
  unpermutedTokens[i//topK\_num] += permutedTokens[sortedIndices[i]]
  $$

## Prototype
  
  Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnMoeTokenUnpermuteWithRoutingMapGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor covering the operator computation process. Then, `aclnnMoeTokenUnpermuteWithRoutingMap` is called to perform computation.

```c++
aclnnStatus aclnnMoeTokenUnpermuteWithRoutingMapGetWorkspaceSize(
  const aclTensor   *permutedTokens,
  const aclTensor   *sortedIndices,
  const aclTensor   *routingMapOptional,
  const aclTensor   *probsOptional,
  bool               paddedMode,
  const aclIntArray *restoreShapeOptional,
  aclTensor         *unpermutedTokens,
  aclTensor         *outIndex,
  aclTensor         *permuteTokenId,
  aclTensor         *permuteProbs,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor);
```

```c++
aclnnStatus aclnnMoeTokenUnpermuteWithRoutingMap(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  const aclrtStream stream)
```
  
## aclnnMoeTokenUnpermuteWithRoutingMapGetWorkspaceSize
  
- **Parameters**
      
  <table style="undefined;table-layout: fixed; width: 1595px"><colgroup>
  <col style="width: 220px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 300px">
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 240px">
  <col style="width: 145px">
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
      <td>permutedTokens (aclTensor*)</td>
      <td>Input</td>
      <td>Input token.</td>
      <td>The capacity in the shape indicates the number of tokens that can be processed by each expert.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>If the value of paddedMode is false: (tokens_num * topK_num, hidden_size)<br>When paddedMode is true: (experts_num* capacity, hidden_size)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sortedIndices (aclTensor*)</td>
      <td>Input</td>
      <td>Mapping between the input and output gradients.</td>
      <td>When paddedMode is false, the index value range is [0, tokens_num * topK_num – 1].<br>When paddedMode is true, the index value range is [0, tokens_num – 1].</td>
      <td>INT32</td>
      <td>ND</td>
      <td>When paddedMode is false: (tokens_num * topK_num)<br>When paddedMode is true: (experts_num * capacity)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>routingMapOptional (aclTensor*)</td>
      <td>Input</td>
      <td>The routingMapOptional in the calculation formula indicates whether the token at the corresponding position is processed by the corresponding expert.</td>
      <td>If the input <code>probsOptional</code> is a null pointer, this input is not required, and a null pointer should be passed.<br>When the data type is INT8, the value can be 0 or 1.<br>When the data type is bool, the value can be true or false.</td>
      <td>INT8, BOOL</td>
      <td>ND</td>
      <td>(tokens_num, experts_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>probsOptional (aclTensor*)</td>
      <td>Input</td>
      <td>The probsOptional in the calculation formula indicates the weight of the token at the corresponding position in the final result after being processed by the corresponding expert.</td>
      <td>The data type is the same as that of permutedTokens. If permutedTokens is of type BFLOAT16, probsOptional can be of type FLOAT.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>The value is the same as that of routingMapOptional.</td>;
      <td>√</td>
    </tr>
    <tr>
      <td>paddedMode (bool)</td>
      <td>Input</td>
      <td>Indicates whether the padding mode is enabled.</td>
      <td>true indicates that paddedMode is enabled.<br>false indicates that paddedMode is disabled.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>restoreShapeOptional (aclIntArray*)</td>
      <td>Input</td>
      <td>Indicates the shape of unpermutedTokens.</td>
      <td>The size is 2.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>unpermutedTokens (aclTensor*)</td>
      <td>Output</td>
      <td>Forward output result, which is the value of unpermutedTokens in the calculation formula.</td>
      <td>-</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>(tokens_num, hidden_size)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>outIndex (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the output index value, which is the value of outIndex in the calculation formula.</td>
      <td>When paddedMode is set to false, the index value range is [0, tokens_num x topK_num – 1].<br>When paddedMode is set to true, the index value range is [0, experts_num x capacity – 1].</td>
      <td>INT32</td>
      <td>ND</td>
      <td>When paddedMode is false: (tokens_num * topK_num)<br>When paddedMode is true: (experts_num* capacity)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>permuteTokenId (aclTensor*)</td>
      <td>Output</td>
      <td>permuteTokenId in the calculation formula.</td>
      <td>The index value range is [0, tokens_num – 1].</td>
      <td>INT32</td>
      <td>ND</td>
      <td>When paddedMode is false: (tokens_num * topK_num)<br>When paddedMode is true: (experts_num* capacity)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>permuteProbs (aclTensor*)</td>
      <td>Output</td>
      <td>permuteProbs in the calculation formula, indicating the sorted probs.</td>
      <td>Same as probsOptional.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Operator executor, containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
  
  The first-phase API implements input parameter verification. The following errors may be thrown:
  
  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 320px">
  <col style="width: 140px">
  <col style="width: 880px">
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
      <td>The pointer to the required input or output tensor is null.</td>
    </tr>
    <tr>
      <td> ACLNN_ERR_PARAM_INVALID </td>
      <td> 161002 </td>
      <td>The input or output data type or shape is not supported.</td>
    </tr>
    <tr>
      <td rowspan="2"> ACLNN_ERR_INNER_NULLPTR </td>
      <td rowspan="2"> 561103 </td>
      <td>topK_num > 512.</td>
    </tr>
    <tr>
      <td>The shape of probsOptional is not supported.</td>
    </tr>
  </tbody></table>

## aclnnMoeTokenUnpermuteWithRoutingMap

- **Parameters**
  <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
    <col style="width: 200px">
    <col style="width: 162px">
    <col style="width: 882px">
    </colgroup>
    <thead>
    <tr>
    <th>Name</th>
    <th>Input/Output</th>
    <th>Description</th>
    </tr></thead>
    <tbody>
    <tr>
    <td>workspace</td>
    <td>Input</td>
    <td>Memory address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>Input</td>
    <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnMoeTokenUnpermuteWithRoutingMapGetWorkspaceSize.</td>
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

- Deterministic computation:
  - `aclnnMoeTokenUnpermuteWithRoutingMap` defaults to deterministic implementation.

- When topK_num <= 512 and paddedMode is false, the number of 1s or true values in each row of routingMap is fixed and less than 512.

- The following scenarios will be intercepted in later versions. If a warning is displayed, you are advised to rectify the fault.
  - paddedMode is true and topK_num > experts_num.
  - paddedMode is true and capacity > tokens_num.
  - The data type or shape of routingMap does not meet the requirements.
  - The data format of the input tensor is not ND.

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
    bool paddedMode = true;
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
    ret = aclnnMoeTokenUnpermuteWithRoutingMapGetWorkspaceSize(permutedTokens, sortedIndices, routingMapOptional, probs, paddedMode, restoreShapeOptional, 
                                                               unpermutedTokens, outIndex, permuteTokenId, permuteProbs, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTokenUnpermuteWithRoutingMapGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnMoeTokenUnpermuteWithRoutingMap(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTokenUnpermuteWithRoutingMap failed. ERROR: %d\n", ret); return ret);
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
