# aclnnMoeTokenPermuteWithRoutingMap

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:32:58.842Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- **Description**: MoE permute computation. Passes token and expert labels as the routingMap, and sorts tokens and optional probsOptional after broadcasting based on the routingMap.
- **Formula**:

  tokens\_num is the size of dimension 0 of routingMap, expert\_num is the size of dimension 1 of routingMap
  when dropAndPad is `false`

  $$
  expertIndex=arange(tokens\_num).expand(expert\_num,-1)
  $$

  $$
  sortedIndicesFirst=expertIndex.masked\_select(routingMap.T)
  $$

  $$
  sortedIndicesOut=argsort(sortedIndicesFirst)
  $$

  $$
  topK = numOutTokens // tokens\_num
  $$

  $$
  outToken = topK * tokens\_num
  $$

  $$
  permuteTokensOut[sortedIndicesOut[i]]=tokens[i//topK]
  $$

  If probs is not none

  $$
  permuteProbsOutOptional=probsOptional.T.masked\_select(routingMap.T)
  $$

  When dropAndPad is `true`

  $$
  capacity = numOutTokens // expert\_num
  $$

  $$
  outToken = capacity * expert\_num
  $$

  $$
  sortedIndicesOut = argsort[routingMap.T,dim=-1](:, :capacity)
  $$

  $$
  permutedTokensOut = tokens.index\_select(0, sortedIndicesOut)
  $$

  If probs is not none

  $$
  probs\_T\_1D = probsOptional.T.view(-1)
  $$

  $$
  indices\_dim0 = arange(expert\_num).view(expert\_num, 1)
  $$

  $$
  indices\_dim1 = sortedIndicesOut.view(expert\_num, capacity)
  $$

  $$
  indices\_1D = (indices\_dim0 * tokens\_num + indices\_dim1).view(-1)
  $$

  $$
  permuteProbsOutOptional = probs\_T\_1D.index\_select(0, indices\_1D)
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First, call the "aclnnMoeTokenPermuteWithRoutingMapGetWorkspaceSize" API to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, call the "aclnnMoeTokenPermuteWithRoutingMap" API to perform computation.

```cpp
aclnnStatus aclnnMoeTokenPermuteWithRoutingMapGetWorkspaceSize(
    const aclTensor  *tokens,
    const aclTensor  *routingMap,
    const aclTensor  *probsOptional,
    int64_t           numOutTokens,
    bool              dropAndPad,
    aclTensor        *permuteTokensOut,
    aclTensor        *permuteProbsOutOptional,
    aclTensor        *sortedIndicesOut,
    uint64_t         *workspaceSize,
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnMoeTokenPermuteWithRoutingMap(
    void            *workspace,
    uint64_t         workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream      stream)
```

## aclnnMoeTokenPermuteWithRoutingMapGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 550px">
  <col style="width: 212px">
  <col style="width: 100px">
  <col style="width: 190px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>tokens</td>
      <td>Input</td>
      <td>Input token features.</td>
      <td><ul><li>Supports empty tensors.</li><li>Must be a 2D tensor with shape (tokens_num, hidden_size).</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>routingMap</td>
      <td>Input</td>
      <td>Mapping from tokens to experts.</td>
      <td><ul><li>Supports empty tensor.</li><li>Requires a 2D shape (tokens_num, experts_num).</li><li>For the INT8 data type, the value can be 0 or 1; for the BOOL data type, the value can be true or false.</li><li>In non-dropAndPad mode, each row must contain fewer than topK true values or 1s.</li></ul></td>
      <td>INT8, BOOL</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>probsOptional</td>
      <td>Input</td>
      <td>Optional input probsOptional.</td>
      <td><ul><li>Supports empty tensor.</li><li>The number of elements is the same as that of routingMap.</li><li>When probsOptional is empty, the optional output permuteProbsOutOptional is empty.</li><li>The data type of probsOptional can differ from that of tokens only when probsOptional is FLOAT and tokens is BFLOAT16; in other use cases, the data type of probsOptional must be the same as that of tokens.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>numOutTokens</td>
      <td>Input</td>
      <td>Number of valid output tokens.</td>
      <td>Used as topK and capacity in the formula. The value ranges from 0 to tokens_num * experts_num.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dropAndPad</td>
      <td>Input</td>
      <td>Specifies whether to enable the dropAndPad mode.</td>
      <td>The value can be false or true.<ul><li>false: indicates the non-dropAndPad mode.</li><li>true: indicates the dropAndPad mode.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>permuteTokensOut</td>
      <td>Output</td>
      <td>Tokens that are expanded and sorted according to indices and then filtered.</td>
      <td><ul><li>Supports empty tensor.</li><li>Must be a 2D tensor with the shape (outToken, hidden_size).</li><li>The data type is the same as that of tokens.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>permuteProbsOutOptional</td>
      <td>Output</td>
      <td>probsOptional that are sorted and filtered according to indices.</td>
      <td><ul><li>Supports empty tensor.</li><li>Shape is (outToken).</li><li>Same data type as probsOptional.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sortedIndicesOut</td>
      <td>Output</td>
      <td>Mapping between permuteTokensOut and tokens.</td>
      <td><ul><li>Supports empty tensor.</li><li>Must be a 1D tensor with shape (outToken).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be requested on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Return value**

  `aclnnStatus`: return status code. For details, see <a href="../../../docs/en/context/aclnn_return_code.md">aclnn return codes</a>.

  The first-phase API performs input parameter validation and returns an error in the following scenarios:
  <table style="undefined;table-layout: fixed; width: 1180px">
    <colgroup>
      <col style="width: 250px">
      <col style="width: 130px">
      <col style="width: 800px">
    </colgroup>
    <thead>
      <tr>
        <th>Return Value</th>
        <th>Error Code</th>
        <th>Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>The input or output tensor is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="3">161002</td>
        <td>The input or output data type is not within the supported range.</td>
      </tr>
      <tr>
        <td>The input or output shape does not meet requirements, except for an invalid second dimension of permuteTokensOut or an invalid rank or length of sortedIndicesOut.</td>
      </tr>
      <tr>
        <td>numOutTokens < 0 or numOutTokens > tokens_num * experts_num</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_NULLPTR</td>
        <td>561103</td>
        <td>The internal pointer is null; or routingMap is non-empty, at least one of permuteTokensOut and sortedIndicesOut is non-empty, and the second dimension of permuteTokensOut or the rank or length of sortedIndicesOut does not meet requirements.</td>
      </tr>
    </tbody>
  </table>

> When routingMap is empty, or both permuteTokensOut and sortedIndicesOut are empty, an invalid second dimension of permuteTokensOut or an invalid rank or length of sortedIndicesOut does not trigger error 561103 described above.

## aclnnMoeTokenPermuteWithRoutingMap

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1180px">
    <colgroup>
      <col style="width: 250px">
      <col style="width: 130px">
      <col style="width: 800px">
    </colgroup>
    <thead>
      <tr>
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>workspace</td>
        <td>Input</td>
        <td>Address of the workspace memory applied for on the device.</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Size of the workspace memory applied for on the device, obtained by the first-phase API <code>aclnnMoeTokenPermuteWithRoutingMapGetWorkspaceSize</code>.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Op executor that contains the operator computation flow.</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Specifies the stream for executing tasks.</td>
      </tr>
    </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnMoeTokenPermuteWithRoutingMap defaults to a deterministic implementation.

- When pad mode is true, tokens_num must be less than `16777215`. When pad mode is false, the number of 1 or true values in each row of routingMap must be less than topK and less than `512`, and topK * tokens_num must be less than `16777215`.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
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
    // Boilerplate: resource initialization.
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
    // Call aclrtMalloc to request device-side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host-side data to device-side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    // Calculate the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    // Call the aclCreateTensor API to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}
int main() {
    // 1. Boilerplate: device/stream initialization. Refer to the ACL external API list.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as needed.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct inputs and outputs as defined by the API.
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
    std::vector<uint8_t> indicesHostData = {1, 1, 1, 1, 1, 1};
    std::vector<float> expandedXOutHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int> sortedIndicesOutHostData = {0, 0, 0, 0, 0, 0};
    // Create the self aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(indicesHostData, idxShape, &indicesDeviceAddr, aclDataType::ACL_INT8, &indices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the out aclTensor.
    ret = CreateAclTensor(expandedXOutHostData, expandedXOutShape, &expandedXOutDeviceAddr, aclDataType::ACL_FLOAT, &expandedXOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sortedIndicesOutHostData, idxOutShape, &sortedIndicesOutDeviceAddr, aclDataType::ACL_INT32, &sortedIndicesOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. Call the CANN operator library API. Modify it to the specific API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnMoeTokenPermuteWithRoutingMap.
    ret = aclnnMoeTokenPermuteWithRoutingMapGetWorkspaceSize(x, indices, nullptr, numTokenOut, padMode, expandedXOut, nullptr, sortedIndicesOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTokenPermuteWithRoutingMapGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnMoeTokenPermuteWithRoutingMap(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTokenPermuteWithRoutingMap failed. ERROR: %d\n", ret); return ret);
    // 4. Boilerplate. Synchronously wait for the task execution to finish.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from the device-side memory to the host side. Modify it based on the specific API definition.
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
    // 6. Release the aclTensor and aclScalar. Modify them based on the specific API definition.
    aclDestroyTensor(x);
    aclDestroyTensor(indices);
    aclDestroyTensor(expandedXOut);
    aclDestroyTensor(sortedIndicesOut);

    // 7. Release the device resources. Modify them based on the specific API definition.
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
