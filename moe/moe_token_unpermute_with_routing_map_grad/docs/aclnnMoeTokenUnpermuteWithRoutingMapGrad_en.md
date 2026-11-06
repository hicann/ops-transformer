# aclnnMoeTokenUnpermuteWithRoutingMapGrad

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:40:38.218Z -->

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
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

- **Description**: Backward propagation of aclnnMoeTokenUnpermuteWithRoutingMap.
- **Formula**:

  (1) probs is not None:

  $$
  permutedTokensGrad[outIndex[i]] = unpermutedTokensGrad[permuteTokenId[i]]
  $$

  $$
  permutedProbsGrad = permutedTokensGrad * permutedTokensOptional
  $$

  $$
  probsGradExpertOrder = \sum_{j=0}^{hidden\_size}(permutedProbsGrad_{i,j})
  $$

    - When dropAndPad is false:

  $$
  probsGradOut = masked\_scatter(routingMapOptional^T,probsGradExpertOrder)
  $$

  $$
  permutedProbs = probsOptional^T.masked\_select(routingMapOptional^T)
  $$

  $$
  permutedTokensGradOut = permutedProbs.unsqueeze(-1) * permutedTokensGrad
  $$

    - When dropAndPad is true:

  $$
  probsGradOut[permuteTokenId[i], outIndex[i]/capacity] = probsGradExpertOrder[outIndex[i]]
  $$

  $$
  permutedProbs[outIndex[i]] = probsOptional.view[1](i)
  $$

  $$
  permutedTokensGradOut = permutedProbs * permutedTokensGrad
  $$

    (2) probs is None:

  $$
  permutedTokensGradOut[outIndex[i]] = unpermutedTokensGrad[permuteTokenId[i]]
  $$

  1. hidden_size refers to the size of the first dimension of unpermutedTokensGrad.
  2. When dropAndPad is true, each expert can process a fixed number of capacity tokens. The first dimension of the input routingMapOptional is experts_num, i.e., the number of experts, and the 0th dimension of the input outIndex is experts_num * capacity. capacity can be derived from these two dimensions.
  3. When dropAndPad is false, each token can be processed by no more than topK_num experts. The 0th dimension of input unpermutedTokensGrad is tokens_num (the number of tokens), and the 0th dimension of input outIndex is tokens_num * topK_num. topK_num can be derived from these two dimensions.

## Function Prototype

Each operator is split into a [two-phase API](../../../docs/en/context/two_phase_api.md). First, call the "aclnnMoeTokenUnpermuteWithRoutingMapGradGetWorkspaceSize" API to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, call the "aclnnMoeTokenUnpermuteWithRoutingMapGrad" API to execute the computation.

```c++
aclnnStatus aclnnMoeTokenUnpermuteWithRoutingMapGradGetWorkspaceSize(
    const aclTensor*   unpermutedTokensGrad,
    const aclTensor*   outIndex,
    const aclTensor*   permuteTokenId,
    const aclTensor*   routingMapOptional,
    const aclTensor*   permutedTokensOptional,
    const aclTensor*   probsOptional,
    bool               dropAndPad,
    const aclIntArray* restoreShapeOptional,
    const aclTensor*   permutedTokensGradOut,
    const aclTensor*   probsGradOutOptional,
    uint64_t*          workspaceSize,
    aclOpExecutor**    executor)
```

```c++
aclnnStatus aclnnMoeTokenUnpermuteWithRoutingMapGrad(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
```

## aclnnMoeTokenUnpermuteWithRoutingMapGradGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1635px"><colgroup>
    <col style="width: 230px">
    <col style="width: 120px">
    <col style="width: 290px">
    <col style="width: 300px">
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 260px">
    <col style="width: 145px">
    </colgroup>
    <thead style="font-size: 13px;">
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
    <td>unpermutedTokensGrad (aclTensor*)</td>
    <td>Input</td>
    <td>unpermutedTokensGrad in the Formula. Represents the gradient of forward output unpermutedTokens.</td>
    <td>-</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
    <td>(tokens_num, hidden_size).</td>
    <td>√</td>
    </tr>
    <tr>
    <td>outIndex (aclTensor*)</td>
    <td>Input</td>
    <td>outIndex in the Formula. Represents the output position index.</td>
    <td><ul><li>If dropAndPad is false, the value range is [0,tokens_num*topK_num-1].</li><li>If dropAndPad is true, the value range is [0,experts_num*capacity-1].</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td><ul><li>If dropAndPad is false, the shape is (tokens_num*topK_num).</li><li>If dropAndPad is true, the shape is (experts_num*capacity).</li></ul></td>
    <td>√</td>
    </tr>
    <tr>
    <td>permuteTokenId (aclTensor*)</td>
    <td>Input</td>
    <td>permuteTokenId in the formula. Indicates the token index corresponding to each position of the input permutedTokens.</td>
    <td>Value range [0,tokens_num-1]</td>
    <td>INT32</td>
    <td>ND</td>
    <td>Same as outIndex.</td>
    <td>√</td>
    </tr>
    <tr>
    <td>routingMapOptional (aclTensor*)</td>
    <td>Optional input</td>
    <td>This input is not required when probsOptional is a null pointer; pass a null pointer in this case. routingMapOptional in the Formula indicates whether the token at the corresponding position is processed by the corresponding expert.</td>
    <td><ul><li>If the data type is INT8, the supported values are 0 and 1.</li><li>If the data type is BOOL, the supported values are true and false.</li></ul></td>
    <td>INT8, BOOL</td>
    <td>ND</td>
    <td>(tokens_num,experts_num).</td>
    <td>√</td>
    </tr>
    <tr>
    <td>permutedTokensOptional (aclTensor*)</td>
    <td>Optional input</td>
    <td>Not required when input probsOptional is a null pointer. Pass a null pointer in this case.</td>
    <td>Same data type as unpermutedTokensGrad.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
    <td><ul><li>If dropAndPad is false, the shape is (tokens_num*topK_num,hidden_size).</li><li>If dropAndPad is true, the shape is (experts_num*capacity,hidden_size).</li></ul></td>
    <td>√</td>
    </tr>
    <tr>
    <td>probsOptional (aclTensor*)</td>
    <td>Optional input</td>
    <td>Set to a null pointer when not required.</td>
    <td>The data type is the same as that of unpermutedTokensGrad, or FLOAT is supported when unpermutedTokensGrad is BFLOAT16.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
    <td>Same as that of routingMapOptional.</td>
    <td>√</td>
    </tr>
    <tr>
    <td>dropAndPad (bool)</td>
    <td>Attribute</td>
    <td>true indicates that dropAndPad is enabled, and false indicates that dropAndPad is disabled.</td>
    <td>-</td>
    <td>BOOL</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>restoreShapeOptional (aclIntArray*)</td>
    <td>Attribute</td>
    <td>aclIntArray of INT64. Represents the shape of unpermutedTokensGrad when dropAndPad is true.</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>permutedTokensGradOut (aclTensor*)</td>
    <td>Output</td>
    <td>permutedTokensGradOut in the Formula, representing the gradient of the input permutedTokens.</td>
    <td>Same data type as unpermutedTokensGrad.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
    <td><ul><li>If dropAndPad is false, the shape is (tokens_num*topK_num,hidden_size).</li><li>If dropAndPad is true, the shape is (experts_num*capacity,hidden_size).</li></ul></td>
    <td>×</td>
    </tr>
    <tr>
    <td>probsGradOutOptional (aclTensor*)</td>
    <td>Optional output.</td>
    <td>Null pointer if probsOptional is not input. Gradient of the input probs.</td>
    <td>Data type is the same as that of probsOptional.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
    <td>Same as routingMapOptional.</td>
    <td>×</td>
    </tr>
    <tr>
    <td>workspaceSize (uint64_t*)</td>
    <td>Output</td>
    <td>Returns the workspace size to be allocated on the Device.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor(aclOpExecutor**)</td>
    <td>Output</td>
    <td>Returns the op executor, which contains the operator computation flow.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 291px">
  <col style="width: 135px">
  <col style="width: 723px">
  </colgroup>
  <thead style="font-size: 13px;">
    <tr>
    <th>Return Value</th>
    <th>Error Code</th>
    <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
    <td> ACLNN_ERR_PARAM_NULLPTR </td>
    <td> 161001 </td>
    <td>A required input, required output, or required attribute is a null pointer.</td>
    </tr>
    <tr>
    <td> ACLNN_ERR_PARAM_INVALID </td>
    <td> 161002 </td>
    <td>The data type and data format of the inputs and outputs are not within the supported range.</td>
    </tr>
    <tr>
    <td rowspan="8"> ACLNN_ERR_INNER_TILING_ERROR </td>
    <td rowspan="8"> 561002 </td>
    <td>When the input probsOptional is non-empty and dropAndPad is false, topK_num > 512.</td>
    </tr>
    <tr>
    <td>When the input probsOptional is non-empty and dropAndPad is false, topK_num is greater than experts_num.</td>
    </tr>
    <tr>
    <td>When the input probsOptional is non-empty and dropAndPad is false, (ubSize - (probTypeLen + 1) * numExpertAlign-(tokenTypeLen + 8) * 256) / (6 * tokenTypeLen + 12) < 1.</td>
    </tr>
    <tr>
    <td>When probsOptional is non-empty and dropAndPad is true, capacity is greater than tokens_num.</td>
    </tr>
    <tr>
    <td>When probsOptional is non-empty and dropAndPad is true, hidden_size > 256 * (ubSize - 2080) / (8 + tokenTypeLen).</td>
    </tr>
    <tr>
    <td>When probsOptional is non-empty, routingMapOptional or permutedTokensOptional is empty.</td>
    </tr>
    <tr>
    <td>When probsOptional is non-empty, the data type of probsOptional is different from that of unpermutedTokensGrad and unpermutedTokensGrad is not BFLOAT16.</td>
    </tr>
    <tr>
    <td>The shape of the input or output does not meet the requirements.</td>
    </tr>
  </tbody>
  </table>

## aclnnMoeTokenUnpermuteWithRoutingMapGrad

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 833px">
  </colgroup>
  <thead style="font-size: 13px;">
  <tr>
    <th>Parameter</th>
    <th>Input/Output</th>
    <th>Description</th>
  </tr></thead>
  <tbody>
  <tr>
    <td>workspace</td>
    <td>Input</td>
    <td>The workspace memory address requested on the Device.</td>
  </tr>
  <tr>
    <td>workspaceSize</td>
    <td>Input</td>
    <td>The workspace size requested on the Device, obtained by the first-phase API <code>aclnnMoeTokenUnpermuteWithRoutingMapGradGetWorkspaceSize</code>.</td>
  </tr>
  <tr>
    <td>executor</td>
    <td>Input</td>
    <td>Op executor, containing the operator computation flow.</td>
  </tr>
  <tr>
    <td>stream</td>
    <td>Input</td>
    <td>Specifies the Stream for task execution.</td>
  </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnMoeTokenUnpermuteWithRoutingMapGrad defaults to a deterministic implementation.
- tokens_num indicates the number of input tokens, hidden_size indicates the word vector dimension, and experts_num indicates the number of experts.
- Two modes are distinguished by dropAndPad: When dropAndPad is true, each expert can process a fixed number of tokens specified by capacity. When dropAndPad is false, each token can be processed by up to topK_num experts.
- When the input probsOptional is non-empty and dropAndPad is false:
  - topK_num must be ≤ 512 and topK_num ≤ experts_num.
  - experts_num must meet the following requirement: (ubSize - (probTypeLen + 1) *numExpertAlign-(tokenTypeLen + 8)* 256) / (6 * tokenTypeLen + 12) >= 1, where ubSize is the size of the on-chip UB space, probTypeLen is the byte count corresponding to the data type of input probsOptional, tokenTypeLen is the byte count corresponding to the data type of input unpermutedTokensGrad, and numExpertAlign is the result of aligning experts_num up to 32.
- When the input probsOptional is non-empty and dropAndPad is true:
  - capacity must be ≤ tokens_num.
  - Requires hidden_size <= 256 * (ubSize - 2080) / (8 + tokenTypeLen), where ubSize is the size of the chip's UB space, and tokenTypeLen is the number of bytes of the data type of the input unpermutedTokensGrad.

## Example

The sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_token_unpermute_with_routing_map_grad.h"
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
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream *stream) {
  // Boilerplate for resource initialization
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
  // Call aclrtMalloc to allocate device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
            return ret);
  // Call aclrtMemcpy to copy host data to device memory.
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
  // 1. (Boilerplate) Initialize the device and stream. For details, see the acl API documentation.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return ret);

  // 2. Construct the inputs and outputs as required by the API.
  bool dropAndPad = false;
  int32_t tokenNum = 1;
  int32_t hiddenSize = 2;
  int32_t expertNum = 2;
  int32_t topK = 2;
  int32_t outTokenNum = tokenNum * topK;
  std::vector<int64_t> permutedTokensShape = {outTokenNum, hiddenSize};
  std::vector<int64_t> unpermutedTokensGradShape = {tokenNum, hiddenSize};
  std::vector<int64_t> probsShape = {tokenNum, expertNum};
  std::vector<int64_t> outIndexShape = {outTokenNum};
  std::vector<int64_t> permuteTokenIdShape = {outTokenNum};
  std::vector<int64_t> routingMapShape = {tokenNum, expertNum};
  std::vector<int64_t> permutedTokensGradShape = {outTokenNum, hiddenSize};
  std::vector<int64_t> probsGradShape = {tokenNum, expertNum};
  void* permutedTokensDeviceAddr = nullptr;
  void* unpermutedTokensGradDeviceAddr = nullptr;
  void* probsDeviceAddr = nullptr;
  void* outIndexDeviceAddr = nullptr;
  void* permuteTokenIdDeviceAddr = nullptr;
  void* routingMapDeviceAddr = nullptr;
  void* permutedTokensGradDeviceAddr = nullptr;
  void* probsGradDeviceAddr = nullptr;

  aclTensor* permutedTokens = nullptr;
  aclTensor* unpermutedTokensGrad = nullptr;
  aclTensor* probs = nullptr;
  aclTensor* outIndex = nullptr;
  aclTensor* permuteTokenId = nullptr;
  aclTensor* routingMap = nullptr;
  aclTensor *permutedTokensGrad = nullptr;
  aclTensor *probsGrad = nullptr;

  std::vector<float> permutedTokensHostData = {1, 1, 1, 1};
  std::vector<float> unpermutedTokensGradHostData = {1, 1};
  std::vector<float> probsHostData = {1, 1};
  std::vector<int> outIndexHostData = {0, 1};
  std::vector<int> permuteTokenIdHostData = {0, 0};
  std::vector<int8_t> routingMapHostData = {1, 1};
  std::vector<float> permutedTokensGradHostData = {0, 0, 0, 0};
  std::vector<float> probsGradHostData = {0, 0};

  ret = CreateAclTensor(unpermutedTokensGradHostData, unpermutedTokensGradShape, &unpermutedTokensGradDeviceAddr, aclDataType::ACL_FLOAT, &unpermutedTokensGrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outIndexHostData, outIndexShape, &outIndexDeviceAddr, aclDataType::ACL_INT32, &outIndex);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(permuteTokenIdHostData, permuteTokenIdShape, &permuteTokenIdDeviceAddr, aclDataType::ACL_INT32, &permuteTokenId);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(routingMapHostData, routingMapShape, &routingMapDeviceAddr, aclDataType::ACL_BOOL, &routingMap);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(permutedTokensHostData, permutedTokensShape, &permutedTokensDeviceAddr, aclDataType::ACL_FLOAT, &permutedTokens);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(probsHostData, probsShape, &probsDeviceAddr, aclDataType::ACL_FLOAT, &probs);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(permutedTokensGradHostData, permutedTokensGradShape, &permutedTokensGradDeviceAddr, aclDataType::ACL_FLOAT, &permutedTokensGrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(probsGradHostData, probsGradShape, &probsGradDeviceAddr, aclDataType::ACL_FLOAT, &probsGrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;

  // Call the aclnnMoeTokenUnpermuteWithRoutingMapGrad first-phase API.
  ret = aclnnMoeTokenUnpermuteWithRoutingMapGradGetWorkspaceSize(unpermutedTokensGrad, outIndex, permuteTokenId, routingMap, permutedTokens, probs, dropAndPad, nullptr, permutedTokensGrad, probsGrad, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnMoeTokenUnpermuteWithRoutingMapGradGetWorkspaceSize failed. ERROR: %d\n", ret);
      return ret);

  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // Call the aclnnMoeTokenUnpermuteWithRoutingMapGrad second-phase API.
  ret = aclnnMoeTokenUnpermuteWithRoutingMapGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnMoeTokenUnpermuteWithRoutingMapGrad failed. ERROR: %d\n", ret);
            return ret);

  // 4. (boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5. Obtain the output value and copy the result from the device side to the host side. Modify the code according to the specific API definition.
  LOG_PRINT("permutedTokensGrad \n");
  PrintOutResult(permutedTokensGradShape, &permutedTokensGradDeviceAddr);
  LOG_PRINT("probsGrad \n");
  PrintOutResult(probsGradShape, &probsGradDeviceAddr);

  // 6. Release aclTensor and aclScalar. Modify the code according to the specific API definition.
  aclDestroyTensor(permutedTokens);
  aclDestroyTensor(unpermutedTokensGrad);
  aclDestroyTensor(outIndex);
  aclDestroyTensor(permuteTokenId);
  aclDestroyTensor(routingMap);
  aclDestroyTensor(probs);
  aclDestroyTensor(permutedTokensGrad);
  aclDestroyTensor(probsGrad);

  // 7. Release device resources.
  aclrtFree(permutedTokensDeviceAddr);
  aclrtFree(unpermutedTokensGradDeviceAddr);
  aclrtFree(probsDeviceAddr);
  aclrtFree(outIndexDeviceAddr);
  aclrtFree(permuteTokenIdDeviceAddr);
  aclrtFree(routingMapDeviceAddr);
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
