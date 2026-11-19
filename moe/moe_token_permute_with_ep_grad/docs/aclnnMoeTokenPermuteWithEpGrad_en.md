# aclnnMoeTokenPermuteWithEpGrad

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:32:41.128Z -->

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

- **Description**: Performs backward propagation computation for aclnnMoeTokenPermuteWithEp.
- **Formula**:

  - First, compute tokenGradOut:
    - When rangeOptional[0] <= sortedIndices[i] < rangeOptional[1]:

      $$
      tokenGradOut[i] = permutedTokensOutputGrad[sortedIndices[i]-rangeOptional[0]]
      $$

    - Otherwise:

      $$
      tokenGradOut[i] = 0
      $$

  - Then calculate:

    $$
    tokenGradOut = tokenGradOut.reshape(-1, numTopk, hiddenSize)
    $$

    $$
    tokenGradOut = tokenGradOut.sum(dim = 1)
    $$

  - Then compute probsGradOut (numTokens = sortedIndices.numel / numTopk):
    - When rangeOptional[0] <= sortedIndices[i] < rangeOptional[1]:

      $$
      probsGradOut[i] = permutedProbsOutputGradOptional[sortedIndices[i]-rangeOptional[0]]
      $$

    - Otherwise:

      $$
      probsGradOut[i] = 0
      $$

    - Finally:

      $$
      probsGradOut = probsGradOut.reshape(numTokens, numTopk)
      $$

  - Precision note: For BFLOAT16/FLOAT16 inputs, tokenGradOut is accumulated in float32 intermediate precision and rounded back to the input dtype; probsGradOut keeps the input dtype unchanged.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call "aclnnMoeTokenPermuteWithEpGradGetWorkspaceSize" to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call "aclnnMoeTokenPermuteWithEpGrad" to perform the computation.

```c++
aclnnStatus aclnnMoeTokenPermuteWithEpGradGetWorkspaceSize(
    const aclTensor       *permutedTokensOutputGrad,
    const aclTensor       *sortedIndices,
    const aclTensor       *permutedProbsOutputGradOptional,
    int64_t                numTopk,
    const aclIntArray     *rangeOptional,
    bool                   paddedMode,
    const aclTensor       *tokenGradOut,
    const aclTensor       *probsGradOut,
    uint64_t              *workspaceSize,
    aclOpExecutor         **executor)
```

```c++
aclnnStatus aclnnMoeTokenPermuteWithEpGrad(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnMoeTokenPermuteWithEpGradGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 187px">
  <col style="width: 121px">
  <col style="width: 287px">
  <col style="width: 387px">
  <col style="width: 187px">
  <col style="width: 187px">
  <col style="width: 187px">
  <col style="width: 146px">
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
      <td>permutedTokensOutputGrad</td>
      <td>Input</td>
      <td>Gradient of the forward output permutedTokens.</td>
      <td>The shape supports 2D dimensions; empty tensors are not supported. Row j corresponds to the global permuted position rangeOptional[0]+j.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>((rangeOptional[1] - rangeOptional[0]), hidden_size)</td>
      <td>√</td>
  </tr>
  <tr>
      <td>sortedIndices</td>
      <td>Input</td>
      <td>Mapping between the forward output permuteTokensOut and the forward input tokens.</td>
      <td>The shape supports 1D dimension, and num_tokens is the number of original tokens. Empty tensors are not supported.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(num_tokens * topK_num)</td>
      <td>√</td>
  </tr>
  <tr>
      <td>permutedProbsOutputGradOptional</td>
      <td>Optional input</td>
      <td>Gradient of the forward output permutedProbs.</td>
      <td>      • The shape supports 1D dimension.<br>
      • Corresponds to the computation output probsGradOut. If an empty tensor is passed in, probsGradOut is not output.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>((rangeOptional[1] - rangeOptional[0]))</td>
      <td>√</td>
  </tr>
  <tr>
      <td>numTopk</td>
      <td>Input</td>
      <td>Number of selected experts.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
    <tr>
      <td>rangeOptional</td>
      <td>Input</td>
      <td>Valid range of EP split, with a size of 2.</td>
      <td>If empty, permutedProbsOutputGradOptional and probsGradOut are ignored, and the execution logic falls back to <a href="../../moe_token_permute_grad/docs/aclnnMoeTokenPermuteGrad_en.md">aclnnMoeTokenPermuteGrad</a>.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  <tr>
      <td>paddedMode</td>
      <td>Input</td>
      <td>-</td>
      <td>true indicates that paddedMode is enabled, and false indicates that paddedMode is disabled. Currently, only false is supported.</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  <tr>
      <td>tokenGradOut</td>
      <td>Output</td>
      <td>Gradient of the token.</td>
      <td>Must be a 2D tensor.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>(num_tokens, hidden_size)</td>
      <td>-</td>
  </tr>
  <tr>
      <td>probsGradOut</td>
      <td>Output</td>
      <td>Gradient of the output probs.</td>
      <td>Supports 2D dimensions.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>(num_tokens, topK_num)</td>
      <td>-</td>
  </tr>
  <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the Device.</td>
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

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
      <col style="width: 267px">
      <col style="width: 124px">
      <col style="width: 775px">
      </colgroup>
      <thead>
          <tr>
          <th>Return Values</th>
          <th>Error Code</th>
          <th>Description</th>
          </tr>
      </thead>
      <tbody>
          <tr>
          <td> ACLNN_ERR_PARAM_NULLPTR </td>
          <td> 161001 </td>
          <td>A required input, output, or attribute is a null pointer.</td>
          </tr>
          <tr>
          <td> ACLNN_ERR_PARAM_INVALID </td>
          <td> 161002 </td>
          <td>The data type or data format of the input or output is not within the supported range, or the shape of a parameter does not meet the constraint.</td>
          </tr>
      </tbody>
  </table>

## aclnnMoeTokenPermuteWithEpGrad

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 173px">
  <col style="width: 133px">
  <col style="width: 860px">
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
          <td>Workspace memory address allocated on the Device.</td>
          </tr>
          <tr>
          <td>workspaceSize</td>
          <td>Input</td>
          <td>Size of the workspace allocated on the Device, obtained by calling the first-phase API aclnnMoeTokenPermuteWithEpGradGetWorkspaceSize.</td>
          </tr>
          <tr>
          <td>executor</td>
          <td>Input</td><td> op executor that contains the operator computation flow.</td>
          </tr>
          <tr>
          <td>stream</td>
          <td>Input</td>
          <td> Stream for task execution.</td>
          </tr>
      </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- deterministic computation:
  - aclnnMoeTokenPermuteWithEpGrad defaults to a deterministic implementation.

- 1 <= numTopk <= 512.
- The data types of permutedTokensOutputGrad, permutedProbsOutputGradOptional, tokenGradOut, and probsGradOut must be the same, being one of BFLOAT16, FLOAT16, or FLOAT32.
- The values of sortedIndices represent global permuted positions, ranging from [0, num_tokens * numTopk); only elements with rangeOptional[0] <= value < rangeOptional[1] participate in the computation.
- paddedMode being `True` is not supported.
- If rangeOptional is empty, permutedProbsOutputGradOptional and probsGradOut are ignored, and the execution logic falls back to [aclnnMoeTokenPermuteGrad](../../moe_token_permute_grad/docs/aclnnMoeTokenPermuteGrad_en.md).

## Example

The following sample code is for reference only. For details about how to compile and run the code, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp

#include "acl/acl.h"
#include "aclnnop/aclnn_moe_token_permute_with_ep_grad.h"
#include <iostream>
#include <vector>
#include <cstdio>

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
int CreateAclTensor(const std::vector<T> &hostData,
                    const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
            return ret);
  // Call aclrtMemcpy to copy data from the host to the device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size,
                    ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
            return ret);

  // Calculate the strides of a contiguous tensor.
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
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return ret);

  // 2. Construct the inputs and outputs as required by the API.

  int64_t num_topk = 2;
  std::vector<float> permuted_token_output_grad_Data = {2, 2, 1, 1, 3, 3, 2, 2};
  std::vector<float> permuted_prob_output_grad_Data = {0.2, 0.5, 0.4, 0.4};
  std::vector<int64_t> permuted_token_output_grad_Shape = {4, 2};
  std::vector<int64_t> permuted_prob_output_grad_Shape = {4};
  void *permuted_token_output_grad_Addr = nullptr;
  void *permuted_prob_output_grad_Addr = nullptr;
  aclTensor *permuted_token_output_grad = nullptr;
  aclTensor *permuted_prob_output_grad = nullptr;

  ret = CreateAclTensor(permuted_token_output_grad_Data, permuted_token_output_grad_Shape,
                        &permuted_token_output_grad_Addr, aclDataType::ACL_FLOAT,
                        &permuted_token_output_grad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(permuted_prob_output_grad_Data, permuted_prob_output_grad_Shape,
                        &permuted_prob_output_grad_Addr, aclDataType::ACL_FLOAT,
                        &permuted_prob_output_grad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int> sortedIndicesData = {2, 0, 4, 1, 5, 3};
  std::vector<int64_t> sortedIndicesShape = {6};
  void *sortedIndicesAddr = nullptr;
  aclTensor *sortedIndices = nullptr;

  ret = CreateAclTensor(sortedIndicesData, sortedIndicesShape, &sortedIndicesAddr,
                        aclDataType::ACL_INT32, &sortedIndices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  void* rangeDeviceAddr = nullptr;
  aclIntArray* range = nullptr;
  std::vector<int64_t> rangeHostData = {1, 5};
  ret = CreateAclIntArray(rangeHostData, &rangeDeviceAddr, &range);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<float> tokenOutData = {0, 0, 0, 0, 0, 0};
  std::vector<int64_t> tokenOutShape = {3, 2};
  void *tokenOutAddr = nullptr;
  aclTensor *tokenOut = nullptr;

  ret = CreateAclTensor(tokenOutData, tokenOutShape, &tokenOutAddr, aclDataType::ACL_FLOAT,
                        &tokenOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<float> probOutData = {0, 0, 0, 0, 0, 0};
  std::vector<int64_t> probOutShape = {3, 2};
  void *probOutAddr = nullptr;
  aclTensor *probOut = nullptr;

  ret = CreateAclTensor(probOutData, probOutShape, &probOutAddr, aclDataType::ACL_FLOAT,
                        &probOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the actual API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;

  // Call the first-phase API of aclnnMoeTokenPermuteWithEpGrad
  ret = aclnnMoeTokenPermuteWithEpGradGetWorkspaceSize(permuted_token_output_grad, sortedIndices, permuted_prob_output_grad,
                                                       num_topk, range, false,
                                                       tokenOut, probOut, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnMoeTokenPermuteWithEpGradGetWorkspaceSize failed. ERROR: %d\n",
                ret);
      return ret);

  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // Call the second-phase API of aclnnMoeTokenPermuteWithEpGrad
  ret = aclnnMoeTokenPermuteWithEpGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnMoeTokenPermuteWithEpGrad failed. ERROR: %d\n", ret);
            return ret);

  // 4. (boilerplate) Wait for task execution to complete synchronously.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5. Obtain the output value and copy the result from device memory to host memory. The code needs to be modified according to the specific API definition.
  PrintOutResult(tokenOutShape, &tokenOutAddr);
  PrintOutResult(probOutShape, &probOutAddr);

  // 6. Release aclTensor and aclScalar. The code needs to be modified according to the specific API definition.
  aclDestroyTensor(permuted_token_output_grad);
  aclDestroyTensor(permuted_prob_output_grad);
  aclDestroyTensor(sortedIndices);
  aclDestroyTensor(tokenOut);
  aclDestroyTensor(probOut);

  // 7. Release device resources.
  aclrtFree(permuted_token_output_grad_Addr);
  aclrtFree(permuted_prob_output_grad_Addr);
  aclrtFree(sortedIndicesAddr);
  aclrtFree(tokenOutAddr);
  aclrtFree(probOutAddr);
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
