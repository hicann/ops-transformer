# aclnnMoeTokenUnpermuteWithEpGrad

[📄 View Source Code](https://gitcode.com/cann/ops-transformer/tree/master/moe/moe_token_unpermute_with_ep_grad)

## Supported Products

| Product                                                        |  Supported  |
| :----------------------------------------------------------- |:-------:|
| Ascend 950PR/Ascend 950DT                            |    ×    |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference products</term>                            |    ×    |
| <term>Atlas training products</term>                             |    ×    |

## Function

- **Description**: Computes the backpropagation of `aclnnMoeTokenUnpermuteWithEp`.
- Formulas:

  - If `probs` is not set to `None`, the formula is as follows, where $i \in {0, 1, 2, ..., num\_tokens – 1}$:
    - First, calculate `unpermutedTokens`.
      - When the condition rangeOptional[0] ≤ sortedIndices[i] < rangeOptional[1] is met,
      
        $$
        unpermutedTokens[i] = permutedTokensOptional[sortedIndices[i]-rangeOptional[0]]
        $$

      - Otherwise,
        
        $$
        unpermutedTokens[i] = 0
        $$
    
    - Then, compute the following:

      $$
      unpermutedTokens = unpermutedTokens.reshape(-1, topkNum, hiddenSize)
      $$
      
      $$
      unpermutedTokens = unpermutedTokensGrad.unsqueeze(1) * unpermutedTokens
      $$
      
      $$
      probsGrad = \sum_{k=0}^{topkNum}(unpermutedTokens_{i,j,k})
      $$
    
    - Finally, when the condition rangeOptional[0] ≤ sortedIndices[i] < rangeOptional[1] is met,
      
      $$
      permutedTokensGradOut[sortedIndices[i]] = ((unpermutedTokensGrad.unsqueeze(1) * probs.unsqueeze(-1)).reshape(-1, hiddenSize))[i]
      $$

  - If `probs` is set to `None`, the formula is as follows, where $i \in {0, 1, 2, ..., num\_tokens – 1}$:
    - When the condition rangeOptional[0] ≤ sortedIndices[i] < rangeOptional[1] is met,
    
    $$
    permutedTokensGradOut[sortedIndices[i]-rangeOptional[0]] = unpermutedTokensGrad[i]
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnMoeTokenUnpermuteWithEpGradGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnMoeTokenUnpermuteWithEpGrad` is called to perform computation.

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
      <td>unpermutedTokensGrad (aclTensor)</td>
      <td>Input</td>
      <td>unpermutedTokensGrad in the formula, which is the gradient of the forward output unpermutedTokens.</td>
      <td>tokens_num indicates the number of tokens, and hidden_size indicates the token dimension.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>(tokens_num, hidden_size)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sortedIndices (aclTensor)</td>
      <td>Input</td>
      <td>sortedIndices in the formula.</td>
      <td>The value range of the index is [0, tokens_num * topkNum – 1].</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(tokens_num * topkNum)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>permutedTokensOptional (aclTensor)</td>
      <td>Input</td>
      <td>permutedTokensOptional in the formula.</td>
      <td>The topkNum &lt;= 512 is required.</td>
      <td>Same as that of unpermutedTokensGrad.</td>
      <td>ND</td>
      <td>(tokens_num * topkNum, hidden_size)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>probsOptional (aclTensor)</td>
      <td>Input</td>
      <td>probsOptional in the formula.</td>
      <td>When probs is passed, topkNum is equal to the second dimension of probs. When probs is not passed, topkNum is 1.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>(tokens_num, topkNum)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>paddedMode (bool)</td>
      <td>Input</td>
      <td><code>true</code> indicates that <code>paddedMode</code> is enabled, and <code>false</code> indicates that <code>paddedMode</code> is disabled. paddedMode indicates the padding mode.</td>
      <td>Currently, only false is supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>restoreShapeOptional (aclIntArray)</td>
      <td>Input</td>
      <td>restoreShapeOptional in the formula. It takes effect only when paddedMode is set to true. Otherwise, no operation will be performed on it.</td>
      <td>Currently, only nullptr is supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rangeOptional (aclIntArray)</td>
      <td>Input</td>
      <td>rangeOptional in the formula, which is the valid range of ep splitting.</td>
      <td>The start position represented by rangeOptional[0] must be smaller than the end position represented by rangeOptional[1]. If this parameter is left empty, it does not take effect.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>topkNum (int64_t)</td>
      <td>Input</td>
      <td>Number of selected experts for each token in topkNum in the formula.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>permutedTokensGradOut (aclTensor)</td>
      <td>Output</td>
      <td>Gradient of permutedTokens, that is, permutedTokensGradOut in the formula.</td>
      <td>-</td>
      <td>The value is the same as that of unpermutedTokensGrad.</td>
      <td>ND</td>
      <td>(tokens_num * topkNum, hidden_size)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>probsGradOut (aclTensor)</td>
      <td>Output</td>
      <td>Gradient of probs in the formula, that is, probsGradOut in the formula.</td>
      <td>(tokens_num, topkNum)</td>
      <td>The value is the same as that of probsOptional.</td>
      <td>ND</td>
      <td>(tokens_num, topkNum)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t)</td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor)</td>
      <td>Output</td>
      <td>Operator executor, containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

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
      <td>The input or output tensor that is mandatory is null.</td>
    </tr>
    <tr>
      <td> ACLNN_ERR_PARAM_INVALID </td>
      <td> 161002 </td>
      <td>The data type or format of the input or output is not supported.</td>
    </tr>
    <tr>
      <td rowspan="3"> ACLNN_ERR_INNER_TILING_ERROR </td>
      <td rowspan="3"> 561002 </td>
      <td>topkNum > 512.</td>
    </tr>
    <tr>
      <td>The input or output shape does not meet the requirements.</td>
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
    <td>Size of the workspace to be allocated on the device, which is obtained by the first-phase API aclnnMoeTokenUnpermuteWithEpGradGetWorkspaceSize.</td>
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
  - `aclnnMoeTokenUnpermuteWithEpGrad` defaults to deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

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

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;

  // Call the first-phase API of aclnnMoeTokenUnpermuteWithEpGrad.
  ret = aclnnMoeTokenUnpermuteWithEpGradGetWorkspaceSize(unpermutedTokensGrad, sortedIndices,permutedTokens, probs, paddedMode, nullptr, nullptr, 3, permutedTokensGrad, probsGrad, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnMoeTokenUnpermuteWithEpGradGetWorkspaceSize failed. ERROR: %d\n", ret);
      return ret);

  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // Call the second-phase API of aclnnMoeTokenUnpermuteWithEpGrad.
  ret = aclnnMoeTokenUnpermuteWithEpGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnMoeTokenUnpermuteWithEpGrad failed. ERROR: %d\n", ret);
            return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  PrintOutResult(permutedTokensGradShape, &permutedTokensGradDeviceAddr);
  PrintOutResult(probsGradShape, &probsGradDeviceAddr);

  // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
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
