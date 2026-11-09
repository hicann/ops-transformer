# aclnnMoeTokenUnpermuteWithEpGrad

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:36:33.686Z -->

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

- Description: Backward propagation of aclnnMoeTokenUnpermuteWithEp.
- Formula:
  - When probs is not None, the formula is as follows, where $i \in {0, 1, 2, ..., num\_tokens - 1}$:
    - First, calculate unpermutedTokens:
      - When rangeOptional[0] <= sortedIndices[i] < rangeOptional[1]:

        $$
        unpermutedTokens[i] = permutedTokensOptional[sortedIndices[i]-rangeOptional[0]]
        $$

      - Otherwise:

        $$
        unpermutedTokens[i] = 0
        $$

    - Then calculate:

      $$
      unpermutedTokens = unpermutedTokens.reshape(-1, topkNum, hiddenSize)
      $$

      $$
      unpermutedTokens = unpermutedTokensGrad.unsqueeze(1) * unpermutedTokens
      $$

      $$
      probsGrad = \sum_{k=0}^{topkNum}(unpermutedTokens_{i,j,k})
      $$

    - Finally, when rangeOptional[0] <= sortedIndices[i] < rangeOptional[1]:

      $$
      permutedTokensGradOut[sortedIndices[i]] = ((unpermutedTokensGrad.unsqueeze(1) * probs.unsqueeze(-1)).reshape(-1, hiddenSize))[i]
      $$

  - When probs is None, the formula is as follows, where $i \in {0, 1, 2, ..., num\_tokens - 1}$:

    - When rangeOptional[0] <= sortedIndices[i] < rangeOptional[1]:

    $$
    permutedTokensGradOut[sortedIndices[i]-rangeOptional[0]] = unpermutedTokensGrad[i]
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnMoeTokenUnpermuteWithEpGradGetWorkspaceSize" API first to obtain the workspace size required for the computation and the executor that contains the operator computation process. Then, call the "aclnnMoeTokenUnpermuteWithEpGrad" API to perform the computation.

```c++
aclnnStatus aclnnMoeTokenUnpermuteWithEpGradGetWorkspaceSize(
  const aclTensor   *unpermutedTokensGrad,
  const aclTensor   *sortedIndices,
  const aclTensor   *permutedTokensOptional,
  const aclTensor   *probsOptional,
  bool               paddedMode,
  const aclIntArray *restoreShapeOptional,
  const aclIntArray *rangeOptional,
  int64_t            topkNum,
  const aclTensor   *permutedTokensGradOut,
  const aclTensor   *probsGradOut,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor)
```

```c++
aclnnStatus aclnnMoeTokenUnpermuteWithEpGrad(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnMoeTokenUnpermuteWithEpGradGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1615px"><colgroup>
  <col style="width: 221px">
  <col style="width: 121px">
  <col style="width: 281px">
  <col style="width: 281px">
  <col style="width: 188px">
  <col style="width: 188px">
  <col style="width: 188px">
  <col style="width: 147px">
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
      <td>unpermutedTokensGrad (aclTensor)</td>
      <td>Input</td>
      <td>unpermutedTokensGrad in the formula, the gradient of forward output unpermutedTokens.</td>
      <td>tokens_num represents the number of tokens, and hidden_size represents the dimension size of tokens.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>(tokens_num, hidden_size)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sortedIndices (aclTensor)</td>
      <td>Input</td>
      <td>sortedIndices in the formula.</td>
      <td>Index value range: [0, tokens_num * topK - 1]. When rangeOptional is set, only elements in the range are processed.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(tokens_num * topK)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>permutedTokensOptional (aclTensor)</td>
      <td>Input</td>
      <td>permutedTokensOptional in the formula.</td>
      <td>The shape must match those of sortedIndices and unpermutedTokensGrad.</td>
      <td>Must be consistent with unpermutedTokensGrad.</td>
      <td>ND</td>
      <td>(tokens_num * topK, hidden_size)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>probsOptional(aclTensor)</td>
      <td>Input</td>
      <td>probsOptional in the formula.</td>
      <td>When probsOptional is passed in, the second dimension of probsOptional is used as the effective topK; when probsOptional is not passed in, topkNum is used as the effective topK, which defaults to 1.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>(tokens_num, topK)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>paddedMode (bool)</td>
      <td>Input</td>
      <td>true indicates that paddedMode is enabled, and false indicates that paddedMode is disabled. paddedMode indicates the padding mode.</td>
      <td>Currently, only false is supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>restoreShapeOptional (aclIntArray)</td>
      <td>Input</td>
      <td>restoreShapeOptional in the formula. It takes effect when paddedMode is true; otherwise, no operation is performed on it.</td>
      <td>Currently, only nullptr is supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rangeOptional(aclIntArray)</td>
      <td>Input</td>
      <td>rangeOptional in the formula. Valid range of the ep split.</td>
      <td>Supports nullptr or an aclIntArray of length 2. Takes no effect when set to nullptr or {-1, -1}; when in effect, the normalized start position must not be greater than the end position.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>topkNum (int64_t)</td>
      <td>Input</td>
      <td>topkNum in the formula. Number of experts selected for each token.</td>
      <td>Serves as the valid topK when probsOptional is not passed in; when probsOptional is passed in, the caller must ensure that topkNum is semantically consistent with the second dimension of probsOptional. The value range of valid topK is [1, 512].</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>permutedTokensGradOut (aclTensor)</td>
      <td>Output</td>
      <td>Gradient of permutedTokens, which is permutedTokensGradOut in the formula.</td>
      <td>-</td>
      <td>Consistent with unpermutedTokensGrad.</td>
      <td>ND</td>
      <td>(tokens_num * topK, hidden_size)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>probsGradOut (aclTensor)</td>
      <td>Output</td>
      <td>probsGradOut in the formula, which is the gradient of probs.</td>
      <td>When probsOptional is passed in, the shape is consistent with probsOptional.</td>
      <td>Consistent with probsOptional.</td>
      <td>ND</td>
      <td>(tokens_num, topK)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t)</td>
      <td>Output</td>
      <td>Returns the size of the workspace that needs to be requested on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the computation process of the operator.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 320px">
  <col style="width: 140px">
  <col style="width: 880px">
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
      <td> ACLNN_ERR_PARAM_NULLPTR </td>
      <td> 161001 </td>
      <td>The Tensor of a required input or required output is a null pointer.</td>
    </tr>
    <tr>
      <td> ACLNN_ERR_PARAM_INVALID </td>
      <td> 161002 </td>
      <td>The data type or data format of the input or output is not within the supported range.</td>
    </tr>
    <tr>
      <td rowspan="3"> ACLNN_ERR_INNER_TILING_ERROR </td>
      <td rowspan="3"> 561002 </td>
      <td>topkNum > 512.</td>
    </tr>
    <tr>
      <td>The shape of the input or output does not meet the requirements.</td>
    </tr>
    <tr>
      <td>rangeOptional[1] < rangeOptional[0].</td>
    </tr>
  </tbody>
  </table>

## aclnnMoeTokenUnpermuteWithEpGrad

- **Parameters**
  <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
    <col style="width: 200px">
    <col style="width: 162px">
    <col style="width: 882px">
    </colgroup>
    <thead>
    <tr>
    <th>Parameter</th>
    <th>Input/Output</th>
    <th>Description</th>
    </tr></thead>
    <tbody>
    <tr>
    <td>workspace</td>
    <td>Input</td>
    <td>Workspace memory address requested on the device.</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>Input</td>
    <td>Workspace size requested on the device, obtained by the first-phase API aclnnMoeTokenUnpermuteWithEpGradGetWorkspaceSize.</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>Input</td>
    <td>Executor of the op, which contains the operator computation process.</td>
    </tr>
    <tr>
    <td>stream</td>
    <td>Input</td>
    <td>Specifies the stream for task execution.</td>
    </tr>
    </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Determinism: aclnnMoeTokenUnpermuteWithEpGrad defaults to a deterministic implementation.

- Shape matching constraints: broadcast is not supported, and the shapes of all inputs and outputs must match explicitly. When permutedTokensOptional is passed in, its 0th dimension must equal the 0th dimension of sortedIndices, and its 1st dimension must equal the 1st dimension of unpermutedTokensGrad (hidden_size); when probsOptional is passed in, its 0th dimension must equal the 0th dimension of unpermutedTokensGrad (tokens_num); the 0th dimension of permutedTokensGradOut must equal the 0th dimension of sortedIndices, and its 1st dimension must equal hidden_size; when probsOptional is passed in, probsGradOut must be consistent with probsOptional.

- Effective topK constraint: when probsOptional is passed in, the effective topK is the 2nd dimension of probsOptional; when probsOptional is not passed in, the effective topK is topkNum (default 1). The effective topK is in the range of [1, 512].

- rangeOptional constraint: nullptr (no effect, but processed throughout) or an aclIntArray of length 2 is supported. When passed in, negative values are supported (normalized by tokens_num * topK) and are clipped to the range [0, tokens_num* topK], which represents a left-closed and right-open interval [start, end). After normalization, start <= end is required, and only sortedIndices elements within this range participate in the write.

- Empty tensor constraint: The empty tensor use case occurs when tokens_num, hidden_size, or valid topK is 0.

<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT products</term>: Empty tensors are supported. The first-phase API returns quickly, workspaceSize is 0, and no valid computation is performed.
<!-- end id7 -->
<!-- npu="A3,910b" id8 -->
- <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: Empty tensors are not supported. The first-phase API returns an error.
<!-- end id8 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_token_unpermute_with_ep_grad.h"
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
    LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream *stream) {
  // Boilerplate for resource initialization.
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
  // Call aclrtMalloc to request device-side memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
            return ret);
  // Call aclrtMemcpy to copy host-side data to device-side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size,
                    ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
            return ret);

  // Calculate the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor API to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
                            strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return ret);

  // 2. Construct input and output. Customize the construction based on the API definition.
  std::vector<int64_t> permutedTokensShape = {3, 2};
  std::vector<int64_t> unpermutedTokensGradShape = {1, 2};
  std::vector<int64_t> probsShape = {1, 3};
  std::vector<int64_t> sortedIndicesShape = {3};
  std::vector<int64_t> permutedTokensGradShape = {3, 2};
  std::vector<int64_t> probsGradShape = {1, 3};
  void* permutedTokensDeviceAddr = nullptr;
  void* unpermutedTokensGradDeviceAddr = nullptr;
  void* probsDeviceAddr = nullptr;
  void* sortedIndicesDeviceAddr = nullptr;
  void* permutedTokensGradDeviceAddr = nullptr;
  void* probsGradDeviceAddr = nullptr;

  aclTensor* permutedTokens = nullptr;
  aclTensor* unpermutedTokensGrad = nullptr;
  aclTensor* probs = nullptr;
  aclTensor* sortedIndices = nullptr;
  bool paddedMode = false;
  aclTensor *permutedTokensGrad = nullptr;
  aclTensor *probsGrad = nullptr;

  std::vector<float> permutedTokensHostData = {1, 1, 1, 1, 1, 1};
  std::vector<float> unpermutedTokensGradHostData = {1, 1};
  std::vector<float> probsHostData = {1, 1, 1};
  std::vector<int> sortedIndicesHostData = {0, 1, 2};
  std::vector<float> permutedTokensGradHostData = {0, 0, 0, 0, 0, 0};
  std::vector<float> probsGradHostData = {0, 0, 0};

  ret = CreateAclTensor(permutedTokensHostData, permutedTokensShape,
                        &permutedTokensDeviceAddr, aclDataType::ACL_FLOAT,
                        &permutedTokens);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(unpermutedTokensGradHostData, unpermutedTokensGradShape, &unpermutedTokensGradDeviceAddr,
                      aclDataType::ACL_FLOAT, &unpermutedTokensGrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(probsHostData, probsShape, &probsDeviceAddr,
                      aclDataType::ACL_FLOAT, &probs);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sortedIndicesHostData, sortedIndicesShape, &sortedIndicesDeviceAddr,
                      aclDataType::ACL_INT32, &sortedIndices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(permutedTokensGradHostData, permutedTokensGradShape, &permutedTokensGradDeviceAddr, aclDataType::ACL_FLOAT,
                        &permutedTokensGrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(probsGradHostData, probsGradShape, &probsGradDeviceAddr, aclDataType::ACL_FLOAT,
                        &probsGrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Change it to the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;

  // Call the first-phase API of aclnnMoeTokenUnpermuteWithEpGrad
  ret = aclnnMoeTokenUnpermuteWithEpGradGetWorkspaceSize(unpermutedTokensGrad, sortedIndices,permutedTokens, probs, paddedMode, nullptr, nullptr, 3, permutedTokensGrad, probsGrad, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnMoeTokenUnpermuteWithEpGradGetWorkspaceSize failed. ERROR: %d\n", ret);
      return ret);

  // Request device memory based on the workspaceSize calculated by the first-phase API
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // Call the second-phase API of aclnnMoeTokenUnpermuteWithEpGrad
  ret = aclnnMoeTokenUnpermuteWithEpGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnMoeTokenUnpermuteWithEpGrad failed. ERROR: %d\n", ret);
            return ret);

  // 4. (Boilerplate) Wait for the task execution to complete synchronously.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5. Obtain the output value and copy the result from device memory to host memory. Modify the code based on the specific API definition.
  PrintOutResult(permutedTokensGradShape, &permutedTokensGradDeviceAddr);
  PrintOutResult(probsGradShape, &probsGradDeviceAddr);

  // 6. Release aclTensor and aclScalar. Modify the code based on the specific API definition.
  aclDestroyTensor(permutedTokens);
  aclDestroyTensor(unpermutedTokensGrad);
  aclDestroyTensor(sortedIndices);
  aclDestroyTensor(probs);
  aclDestroyTensor(permutedTokensGrad);
  aclDestroyTensor(probsGrad);

  // 7. Release device resources.
  aclrtFree(permutedTokensDeviceAddr);
  aclrtFree(unpermutedTokensGradDeviceAddr);
  aclrtFree(probsDeviceAddr);
  aclrtFree(sortedIndicesDeviceAddr);
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
