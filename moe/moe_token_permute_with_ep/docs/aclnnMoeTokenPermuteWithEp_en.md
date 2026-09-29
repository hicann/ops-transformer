# aclnnMoeTokenPermuteWithEp

[📄 View Source Code](https://gitcode.com/cann/ops-transformer/tree/master/moe/moe_token_permute_with_ep)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |     ×     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |     ×      |

## Function

- **Description**: Performs permute computation of MoE. It broadcasts tokens and optional probs based on the indices, sorts them, and slices them based on the range in rangeOptional.
- **Formula**:
  - When paddedMode is false, the formula is as follows: topK indicates the number of experts selected for each token. If Indices is two-dimensional, topK is equal to the size of the last dimension of Indices. If Indices is one-dimensional, topK is 1.
    
    $$
    sortedIndicesFirst=argSort(\text{flatten}(Indices))
    $$

    $$
    sortedIndicesOut=argSort(sortedIndicesFirst)
    $$
    
    When rangeOptional[0] <= sortedIndicesOut[i] < rangeOptional[1]:

    $$
    permuteTokensOut[sortedIndicesOut[i] - rangeOptional[1]]=tokens[i//topK]
    $$

  - When paddedMode is true (not supported currently):

    $$
    permuteTokensOut[i]=tokens[Indices[i]]
    $$

    $$
    sortedIndicesOut=Indices
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnMoeTokenPermuteWithEpGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnMoeTokenPermuteWithEp` is called to perform computation.

```cpp
aclnnStatus aclnnMoeTokenPermuteWithEpGetWorkspaceSize(
    const aclTensor    *tokens, 
    const aclTensor    *indices, 
    const aclTensor    *probsOptional, 
    const aclIntArray  *rangeOptional, 
    int64_t             numOutTokens, 
    bool                paddedMode, 
    const aclTensor    *permuteTokensOut, 
    const aclTensor    *sortedIndicesOut, 
    const aclTensor    *permuteProbsOut, 
    uint64_t           *workspaceSize, 
    aclOpExecutor     **executor)
```

```cpp
aclnnStatus aclnnMoeTokenPermuteWithEp(
    void            *workspace, 
    uint64_t         workspaceSize, 
    aclOpExecutor   *executor, 
    aclrtStream      stream)
```

## aclnnMoeTokenPermuteWithEpGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1566px"><colgroup>
  <col style="width: 197px">
  <col style="width: 120px">
  <col style="width: 242px">
  <col style="width: 399px">
  <col style="width: 200px">
  <col style="width: 123px">
  <col style="width: 140px">
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
      <td>tokens</td>
      <td>Input</td>
      <td>: tokens entered in permute.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The value must be a 2D tensor with the shape of (num_tokens, hidden_size).</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Input</td>
      <td>: expert index corresponding to the input tokens.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape must be 1D or 2D. </li><li>If paddedMode is false, it indicates the top K processing expert indexes corresponding to each input token. The shape is (num_tokens, topK_num) or (num_tokens). </li><li>If paddedMode is set to true, this parameter indicates the token index selected by each expert (not supported currently). </li><li> The number of elements is less than 16777215, and the value is greater than or equal to 0 and less than 16777215.</li></ul></td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>1 or 2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>probsOptional</td>
      <td>Input</td>
      <td> indicates the expert probability corresponding to the input tokens.</td>
      <td><ul><li>Empty tensors are supported. </li><li>Correspond to the output of permuteProbsOut. If the input parameter is left blank, permuteProbsOut is not output. </li><li>The shape of shape is the same as that of indices.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>1 or 2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rangeOptional</td>
      <td>Input</td>
      <td>Valid range of the ep segmentation.</td>
      <td><ul><li>The value can be empty. </li><li>The size is 2. </li><li>If this parameter is left empty, probsOptional and permuteTokensOut are ignored, and the execution logic is rolled back to aclnnMoeTokenPermute.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numOutTokens</td>
      <td>Input</td>
      <td>Number of valid output tokens.</td>
      <td>The value is an integer. 0 indicates that no token is deleted. If the value is greater than 0, tokens are sliced based on the number of tokens sorted by the expert, and the first numOutTokens tokens are retained. If the value is less than 0, the tokens are processed based on the negative slicing index.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>paddedMode</td>
      <td>Input</td>
      <td>Whether the mode is padding mode.</td>
      <td>The value can be false or true. <ul><li>false: indicates the non-padding mode. In this case, indices are sorted. </li><li>true: indicates the padding mode. In this case, indices have been filled with the token index selected by each expert. In this case, indices are not sorted (not supported currently).</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>permuteTokensOut</td>
      <td>Output</td>
      <td>Tokens that are extended and sorted based on indices.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The value must be a 2D tensor with the shape of (rangeOptional[1] - rangeOptional[0], hidden_size). </li><li>The data type is the same as that of tokens.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sortedIndicesOut</td>
      <td>Output</td>
      <td>Mapping between permuteTokensOut and tokens.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The value must be a 1D tensor with the shape of (num_tokens * topK_num).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>permuteProbsOut</td>
      <td>Output</td>
      <td>Probs that are extended and sorted based on indices.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The value must be a 1D tensor with the shape of (rangeOptional[1] - rangeOptional[0]). </li><li>The data type is the same as that of tokens.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
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

  The first-phase API implements input parameter verification. The following errors may be thrown.
  <table style="undefined;table-layout: fixed; width: 1180px"> 
    <colgroup>
      <col style="width: 250px">
      <col style="width: 130px">
      <col style="width: 800px">
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
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>The input and output tensors are null pointers.</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td>The data type or format of the input and output is not supported.</td>
      </tr>
      <tr>
        <td rowspan="3">ACLNN_ERR_INNER_TILING_ERROR</td>
        <td rowspan="3">561002</td>
        <td>The shape dimension of tokens is not 2.</td>
      </tr>
      <tr>
        <td>The shape of indices is not 1D or 2D, or the first dimension of the shape of indices is not equal to that of tokens when paddedMode is false.</td>
      </tr>
      <tr>
        <td>paddedMode is true (not supported currently).</td>
      </tr>
    </tbody>
  </table>

## aclnnMoeTokenPermuteWithEp

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1148px"><colgroup>
  <col style="width: 170px">
  <col style="width: 134px">
  <col style="width: 844px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Memory address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnMoeTokenPermuteWithEpGetWorkspaceSize.</td>
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
  - `aclnnMoeTokenPermuteWithEp` defaults to deterministic implementation.

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
    // Call the first-phase API of aclnnMoeTokenPermuteWithEp.
    ret = aclnnMoeTokenPermuteWithEpGetWorkspaceSize(x, indices, probs, range, numTokenOut, padMode, expandedXOut, sortedIndicesOut, expandedProbsOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTokenPermuteWithEpGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnMoeTokenPermuteWithEp.
    ret = aclnnMoeTokenPermuteWithEp(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTokenPermuteWithEp failed. ERROR: %d\n", ret); return ret);
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
