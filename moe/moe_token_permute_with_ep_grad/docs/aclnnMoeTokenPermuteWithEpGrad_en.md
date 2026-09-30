# aclnnMoeTokenPermuteWithEpGrad

[📄 View source code](https://gitcode.com/cann/ops-transformer/tree/master/moe/moe_token_permute_with_ep_grad)

## Product Support

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- **Description**: Performs the backpropagation of `aclnnMoeTokenPermuteWithEp`.
- **Formula**:
    
  - First, compute `tokenGradOut`.
    - When the condition rangeOptional[0] ≤ sortedIndices[i] < rangeOptional[1] is met,

      $$
      tokenGradOut[i] = permutedTokensOutputGrad[sortedIndices[i]-rangeOptional[0]]
      $$

    - Otherwise,
      
      $$
      tokenGradOut[i] = 0
      $$

  - Then, compute the following:

    $$
    tokenGradOut = tokenGradOut.reshape(-1, topK, hiddenSize)
    $$

    $$
    tokenGradOut = tokenGradOut.sum(dim = 1)
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnMoeTokenPermuteWithEpGradGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnMoeTokenPermuteWithEpGrad` is called to perform computation.

```c++
aclnnStatus aclnnMoeTokenPermuteWithEpGradGetWorkspaceSize(
    const aclTensor       *permutedTokensOutputGrad,
    const aclTensor       *sortedIndices,
    const aclTensor       *permutedProbsOutputGradOptional,
    int64_t                numTopk,
    const aclIntArray     *rangeOptional,
    bool                   paddedMode,
    const aclTensor       *tokenGradOut
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
        <td>permutedTokensOutputGrad</td>
        <td>Input</td>
        <td>Gradient of forward output <code>permutedTokens</code>.</td>
        <td>The shape can be 2D, but empty tensors are not supported. The value of <code>topK_num</code> is equal to that of <code>numTopk</code>.</td>
        <td>BFLOAT16, FLOAT16, FLOAT32</td>
        <td>ND</td>
        <td> (rangeOptional[1] – rangeOptional[0]) * topK_num, hidden_size)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>sortedIndices</td>
        <td>Input</td>
        <td>Mapping between the forward output <code>permuteTokensOut</code> and the forward input <code>tokens</code>.</td>
        <td>The 1D shape is supported. <code>num_tokens</code> indicates the original number of tokens. Empty tensors are not supported.</td>
        <td>INT32</td>
        <td>ND</td>
        <td>(num_tokens * topK_num)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>permutedProbsOutputGradOptional</td>
        <td>Optional input</td>
        <td>Gradient of the forward output <code>permutedProbs</code>.</td>
        <td>
        • The 1D shape is supported, and the value of <code>topK_num</code> is equal to that of <code>numTopk</code>.<br>
        • Corresponds to the computation output <code>probsGradOut</code>. If an empty value is passed, <code>probsGradOut</code> is not output.</td>
        <td>BFLOAT16, FLOAT16, FLOAT32</td>
        <td>ND</td>
        <td> (rangeOptional[1] – rangeOptional[0]) * topK_num)</td>
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
        <td>Valid range of EP slicing. The size is <code>2</code>.</td>
        <td>If this parameter is left empty, <code>permutedProbsOutputGradOptional</code> and <code>probsGradOut</code> are ignored, and the execution logic is rolled back to <a href="../../moe_token_permute_grad/docs/aclnnMoeTokenPermuteGrad_en.md">aclnnMoeTokenPermuteGrad</a>.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>paddedMode</td>
        <td>Input</td>
        <td>-</td>
        <td><code>true</code> indicates that <code>paddedMode</code> is enabled, and <code>false</code> indicates that <code>paddedMode</code> is disabled. Currently, only <code>false</code> is supported.</td>
        <td>bool</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>tokenGradOut</td>
        <td>Output</td>
        <td>Gradient of the input token.</td>
        <td>The value must be a 2D tensor.</td>
        <td>BFLOAT16, FLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>(num_tokens, hidden_size)</td>
        <td>-</td>
    </tr>
    <tr>
        <td>probsGradOut</td>
        <td>Output</td>
        <td>Gradient of the input probs.</td>
        <td>The 2D shape is supported.</td>
        <td>BFLOAT16, FLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>(num_tokens, topK_num)</td>
        <td>-</td>
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

    <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
        <col style="width: 267px">
        <col style="width: 124px">
        <col style="width: 775px">
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
            <td>The required input, output, or attribute is passed as a null pointer.</td>
            </tr>
            <tr>
            <td> ACLNN_ERR_PARAM_INVALID </td>
            <td> 161002 </td>
            <td>The input and output data types and data formats are not supported.</td>
            </tr>
        </tbody></table>

## aclnnMoeTokenPermuteWithEpGrad

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
      <col style="width: 173px">
      <col style="width: 133px">
      <col style="width: 860px">
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
              <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API <code>aclnnMoeTokenPermuteWithEpGradGetWorkspaceSize</code>.</td>
              </tr>
              <tr>
              <td>executor</td>
              <td>Input</td><td>Operator executor, containing the operator computation process.</td>
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
  - `aclnnMoeTokenPermuteWithEpGrad` defaults to deterministic implementation.

  - The value of `top_k` is less than or equal to `512`.
  - `paddedMode` cannot be set to `True`.
  - When `rangeOptional` is left empty, `permutedProbsOutputGradOptional` and `probsGradOut` are ignored, and the execution logic is rolled back to [aclnnMoeTokenPermuteGrad](../../moe_token_permute_grad/docs/aclnnMoeTokenPermuteGrad_en.md).

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp

#include "acl/acl.h"
#include "aclnnop/aclnn_moe_token_permute_with_ep_grad.h"
#include <iostream>
#include <vector>

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
                        &permuted_token_output_grad_Addr, aclDataType::ACL_BF16,
                        &permuted_token_output_grad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(permuted_prob_output_grad_Data, permuted_prob_output_grad_Shape,
                        &permuted_prob_output_grad_Addr, aclDataType::ACL_BF16,
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

  ret = CreateAclTensor(tokenOutData, tokenOutShape, &tokenOutAddr, aclDataType::ACL_BF16,
                        &tokenOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<float> probOutData = {0, 0, 0, 0, 0, 0};
  std::vector<int64_t> probOutShape = {3, 2};
  void *probOutAddr = nullptr;
  aclTensor *probOut = nullptr;

  ret = CreateAclTensor(probOutData, probOutShape, &probOutAddr, aclDataType::ACL_BF16,
                        &probOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;

  // Call the first-phase API of aclnnMoeTokenPermuteWithEpGrad.
  ret = aclnnMoeTokenPermuteWithEpGradGetWorkspaceSize(permuted_token_output_grad, sortedIndices, permuted_prob_output_grad,
                                                       num_topk, range, false,
                                                       tokenOut, probOut, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnMoeTokenPermuteWithEpGradGetWorkspaceSize failed. ERROR: %d\n",
                ret);
      return ret);

  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // Call the second-phase API of aclnnMoeTokenPermuteWithEpGrad.
  ret = aclnnMoeTokenPermuteWithEpGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnMoeTokenPermuteWithEpGrad failed. ERROR: %d\n", ret);
            return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  PrintOutResult(tokenOutShape, &tokenOutAddr);
  PrintOutResult(probOutShape, &probOutAddr);

  // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
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
