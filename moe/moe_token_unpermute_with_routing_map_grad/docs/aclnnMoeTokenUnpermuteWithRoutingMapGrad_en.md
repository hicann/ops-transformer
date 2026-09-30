# aclnnMoeTokenUnpermuteWithRoutingMapGrad

## Product Support

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- **Description**: Computes the backpropagation of `aclnnMoeTokenUnpermuteWithRoutingMap`.
- **Formula**:

  (1) When `probs` is not set to `None`:
  
  $$
  permutedTokensGrad[outIndex[i]] = unpermutedTokensGrad[permuteTokenId[i]]
  $$
  
  $$
  permutedProbsGrad = permutedTokensGrad * permutedTokensOptional
  $$
  
  $$
  probsGradExpertOrder = \sum_{j=0}^{hidden\_size}(permutedProbsGrad_{i,j})
  $$

    - When `paddedMode` is set to `false`:
  
  $$
  probsGradOut = masked\_scatter(routingMapOptional^T,probsGradExpertOrder)
  $$
  
  $$
  permutedProbs = probsOptional^T.masked\_select(routingMapOptional^T)
  $$

  $$
  permutedTokensGradOut = permutedProbs.unsqueeze(-1) * permutedTokensGrad
  $$

    - When `paddedMode` is set to `true`:
  
  $$
  probsGradOut[permuteTokenId[i], outIndex[i]/capacity] = probsGradExpertOrder[outIndex[i]]
  $$

  $$
  permutedProbs[outIndex[i]] = probsOptional.view(1) [i]
  $$

  $$
  permutedTokensGradOut = permutedProbs * permutedTokensGrad
  $$

    (2) When `probs` is set to `None`:
    
  $$
  permutedTokensGradOut[outIndex[i]] = unpermutedTokensGrad[permuteTokenId[i]]
  $$

  1. `hidden_size` indicates the size of the first dimension of `unpermutedTokensGrad`.
  2. When `paddedMode` is set to `true`, each expert can process a fixed number of tokens, which is specified by `capacity`. The first dimension of the input `routingMapOptional` is the value specified by `experts_num`, which indicates the number of experts. The 0th dimension of the input `outIndex` is `experts_num` * `capacity`. The value of `capacity` can be computed based on the two dimensions.
  3. When `paddedMode` is set to `false`, each token is processed by a fixed number of experts, which is specified by `topK_num`. The 0th dimension of the input `unpermutedTokensGrad` is specified by `tokens_num`, which indicates the number of tokens. The 0th dimension of the input `outIndex` is the value specified by `tokens_num` * `capacity`. The value of `topK_num` can be computed based on the two dimensions.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnMoeTokenUnpermuteWithRoutingMapGradGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor covering the operator computation process. Then, `aclnnMoeTokenUnpermuteWithRoutingMapGrad` is called to perform computation.

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

    <table style="undefined;table-layout: fixed; width: 1643px"><colgroup>
        <col style="width: 213px">
        <col style="width: 121px">
        <col style="width: 269px">
        <col style="width: 320px">
        <col style="width: 171px">
        <col style="width: 117px">
        <col style="width: 280px">
        <col style="width: 146px">
        </colgroup>
        <thead style="font-size: 13px;">
        <tr>
            <th>Name</th>
            <th>Input/Output</th>
            <th>Description</th>
            <th>Usage Notes</th>
            <th>Data Type</th>
            <th>Data Format</th>
            <th>Dimension (Shape)</th>
            <th>Non-contiguous Tensor</th>
        </tr></thead>
    <tbody>
        <tr>
        <td>unpermutedTokensGrad</td>
        <td>Input</td>
        <td><code>unpermutedTokensGrad</code> in the formula, which indicates the gradient of the forward output <code>unpermutedTokens</code>.</td>
        <td>-</td>
        <td>BFLOAT16, FLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>(tokens_num, hidden_size)</td>
        <td>√</td>
        </tr>
        <tr>
        <td>outIndex</td>
        <td>Input</td>
        <td><code>outIndex</code> in the formula, which indicates the output position index.</td>
        <td><ul><li>If <code>paddedMode</code> is set to <code>false</code>, the value range is [0, tokens_num * topK_num – 1]. </li><li>If <code>paddedMode</code> is set to <code>true</code>, the value range is [0, experts_num * capacity – 1].</li></ul></td>
        <td>INT32</td>
        <td>ND</td>
        <td><ul><li>If <code>paddedMode</code> is set to <code>false</code>, the shape is (tokens_num * topK_num).</li><li>If <code>paddedMode</code> is set to <code>true</code>, the shape is (experts_num * capacity).</li></ul></td>
        <td>√</td>
        </tr>
        <tr>
        <td>permuteTokenId</td>
        <td>Input</td>
        <td><code>permuteTokenId</code> in the formula, which indicates the token ID corresponding to each position in the input <code>permutedTokens</code>.</td>
        <td>The value range is [0, tokens_num – 1].</td>
        <td>INT32</td>
        <td>ND</td>
        <td>Same as that of <code>outIndex</code>.</td>
        <td>√</td>
        </tr>
        <tr>
        <td>routingMapOptional</td>
        <td>Optional input</td>
        <td>If the input <code>probsOptional</code> is a null pointer, this input is not required, and a null pointer should be passed. In the formula, <code>routingMapOptional</code> indicates whether the token at the corresponding position is processed by the corresponding expert.</td>
        <td><ul><li>If the data type is INT8, the value can be <code>0</code> or <code>1</code>. </li><li>If the data type is BOOL, the value can be <code>true</code> or <code>false</code>.</li></ul></td>
        <td>INT8, BOOL</td>
        <td>ND</td>
        <td>(tokens_num,experts_num).</td>
        <td>√</td>
        </tr>
        <tr>
        <td>permutedTokensOptional</td>
        <td>Optional input</td>
        <td>If the input <code>probsOptional</code> is a null pointer, this input is not required, and a null pointer should be passed.</td>
        <td>The data type is the same as that of <code>unpermutedTokensGrad</code>.</td>
        <td>BFLOAT16, FLOAT16, FLOAT32</td>
        <td>ND</td>
        <td><ul><li>If <code>paddedMode</code> is set to <code>false</code>, the shape is (tokens_num * topK_num, hidden_size).</li><li>If <code>paddedMode</code> is set to <code>true</code>, the shape is (experts_num * capacity, hidden_size).</li></ul></td>
        <td>√</td>
        </tr>
        <tr>
        <td>probsOptional</td>
        <td>Optional input</td>
        <td>A null pointer if not required.</td>
        <td>The data type is the same as that of <code>unpermutedTokensGrad</code>.</td>
        <td>BFLOAT16, FLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>Same as that of <code>routingMapOptional</code>.</td>
        <td>√</td>
        </tr>
        <tr>
        <td>paddedMode</td>
        <td>Attribute</td>
        <td><code>true</code> indicates that <code>paddedMode</code> is enabled, and <code>false</code> indicates that <code>paddedMode</code> is disabled.</td>
        <td>-</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        </tr>
        <tr>
        <td>restoreShapeOptional</td>
        <td>Attribute</td>
        <td>aclIntArray of the INT64 type. When <code>paddedMode</code> is set to <code>true</code>, this parameter indicates the shape of <code>unpermutedTokensGrad</code>.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        </tr>
        <tr>
        <td>permutedTokensGradOut</td>
        <td>Output</td>
        <td><code>permutedTokensGradOut</code> in the formula, which indicates the gradient of the input <code>permutedTokens</code>.</td>
        <td>The data type is the same as that of <code>unpermutedTokensGrad</code>.</td>
        <td>BFLOAT16, FLOAT16, FLOAT32</td>
        <td>ND</td>
        <td><ul><li>If <code>paddedMode</code> is set to <code>false</code>, the shape is (tokens_num * topK_num, hidden_size).</li><li>If <code>paddedMode</code> is set to <code>true</code>, the shape is (experts_num * capacity, hidden_size).</li></ul></td>
        <td>×</td>
        </tr>
        <tr>
        <td>probsGradOutOptional</td>
        <td>Optional output</td>
        <td>Null pointer if <code>probsOptional</code> is not specified. Gradient of the input <code>probs</code>.</td>
        <td>The data type is the same as that of <code>unpermutedTokensGrad</code>.</td>
        <td>BFLOAT16, FLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>Same as that of <code>routingMapOptional</code>.</td>
        <td>×</td>
        </tr>
        <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Size of the workspace required to be allocated on the device.</td>
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

    <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
        <col style="width: 250px">
        <col style="width: 130px">
        <col style="width: 650px">
        </colgroup>
        <thead style="font-size: 13px;">
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
            <td>The required input, output, or attribute is passed as a null pointer.</td>
            </tr>
            <tr>
            <td> ACLNN_ERR_PARAM_INVALID </td>
            <td> 161002 </td>
            <td>The input and output data types and data formats are not supported.</td>
            </tr>
            <tr>
            <td rowspan="4"> ACLNN_ERR_INNER_TILING_ERROR </td>
            <td rowspan="4"> 561002 </td>
            <td>When the input <code>probsOptional</code> is not empty and <code>paddedMode</code> is false, the value of <code>topK_num</code> must be greater than <code>512</code>.</td>
            </tr>
            <tr>
            <td>When the value of <code>probsOptional</code> is not empty and <code>paddedMode</code> is set to <code>false</code>, the value of <code>topK_num</code> is greater than that of <code>experts_num</code>.</td>
            </tr>
            <tr>
            <td>When the value of <code>probsOptional</code> is not empty and <code>paddedMode</code> is set to <code>true</code>, the value of <code>capacity</code> is greater than that of <code>tokens_num</code>.</td>
            </tr>
            <tr>
            <td>The input or output shape does not meet the requirements.</td>
            </tr>
        </tbody></table>

## aclnnMoeTokenUnpermuteWithRoutingMapGrad

- **Parameters:**
    <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
        <col style="width: 250px">
        <col style="width: 130px">
        <col style="width: 650px">
        </colgroup>
        <thead style="font-size: 13px;">
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
            <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API <code>aclnnMoeTokenUnpermuteWithRoutingMapGradGetWorkspaceSize</code>.</td>
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
  
    `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnMoeTokenUnpermuteWithRoutingMapGrad` defaults to deterministic implementation.

- When the input `probsOptional` is not left empty and `paddedMode` is set to `false`, the value of `topK_num` is less than or equal to that of `experts_num`, and the value is less than or equal to `512`.
- When the input `probsOptional` is not left empty and `paddedMode` is set to `true`, the value of `capacity` is less than or equal to that of `tokens_num`.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

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
  bool paddedMode = false;
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

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;

  // Call the first-phase API of aclnnMoeTokenUnpermuteWithRoutingMapGrad.
  ret = aclnnMoeTokenUnpermuteWithRoutingMapGradGetWorkspaceSize(unpermutedTokensGrad, outIndex, permuteTokenId, routingMap, permutedTokens, probs, paddedMode, nullptr, permutedTokensGrad, probsGrad, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnMoeTokenUnpermuteWithRoutingMapGradGetWorkspaceSize failed. ERROR: %d\n", ret);
      return ret);

  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // Call the second-phase API of aclnnMoeTokenUnpermuteWithRoutingMapGrad.
  ret = aclnnMoeTokenUnpermuteWithRoutingMapGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnMoeTokenUnpermuteWithRoutingMapGrad failed. ERROR: %d\n", ret);
            return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  LOG_PRINT("permutedTokensGrad \n");
  PrintOutResult(permutedTokensGradShape, &permutedTokensGradDeviceAddr);
  LOG_PRINT("probsGrad \n");
  PrintOutResult(probsGradShape, &probsGradDeviceAddr);

  // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
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
