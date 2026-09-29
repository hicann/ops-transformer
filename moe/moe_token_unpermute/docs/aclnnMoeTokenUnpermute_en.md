# aclnnMoeTokenUnpermute

[📄 View Source Code](https://gitcode.com/cann/ops-transformer/tree/master/moe/moe_token_unpermute)

## Supported Products

| Product                                                        |  Supported  |
| :----------------------------------------------------------- |:-------:|
| Ascend 950PR/Ascend 950DT                     |     √    |
| <term>Atlas A3 training products/Atlas A3 inference products</term>     |    √    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>     |    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference products</term>                            |    ×    |
| <term>Atlas training products</term>                             |    ×    |

## Function

- **Description**: Obtains the input data of permutedTokens based on sortedIndices. If probs data exists, permutedTokens is multiplied by probs. Then, computes the cumulative sum and outputs the computation result.
- Formulas:

  - If `probs` is not set to `None`, the formula is as follows:
    
    $$
    T[k] = T[S[k]]
    $$
    
    $$
    T[k] = T[k] * P[i][j]
    $$

    $$
    O[i] = \sum_{k=i*topK}^{(i+1)*topK - 1 } T[k]
    $$
    
    where $i \in {0,1,...,tokens-1}$; $j \in {0,1,...,topK-1}$; $k \in {0,1,...,tokens*topK-1}$; `T` indicates `permutedTokens`; `S` indicates `sortedIndices`; `P` indicates `probs`; `O` indicates `out`; `topK` indicates `topK\_num`; `tokens` indicates `tokens_num`.

  - If `probs` is set to `None`, `topK\_num` is `1`. The formula is as follows:

    $$
    T[i] = T[S[i]]
    $$

    $$
    O[i] = T[i]
    $$

    where $i \in {0,1,...,tokens-1}$; `T` indicates `permutedTokens`; `S` indicates `sortedIndices`; `O` indicates `out`; `tokens` indicates `tokens_num`.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnMoeTokenUnpermuteGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnMoeTokenUnpermute` is called to perform computation.

```c++
aclnnStatus aclnnMoeTokenUnpermuteGetWorkspaceSize(
    const aclTensor   *permutedTokens,
    const aclTensor   *sortedIndices,
    const aclTensor   *probsOptional,
    bool               paddedMode,
    const aclIntArray *restoreShapeOptional,
    aclTensor         *out,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnMoeTokenUnpermute(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnMoeTokenUnpermuteGetWorkspaceSize

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
      <td>permutedTokens (aclTensor)</td>
      <td>Input</td>
      <td>Enter Tokens, which is T in the formula.</td>
      <td>shape is (tokens_num * topK_num, hidden_size), where tokens_num indicates the number of input tokens, topK_num indicates the number of experts who process each token, and hidden_size indicates the length of the vector representation of each token.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sortedIndices (aclTensor)</td>
      <td>Input</td>
      <td>: position of the data to be calculated in permutedTokens.</td>
      <td><ul><li>shape is (tokens_num * topK_num). </li><li> The value range is [0, tokens_num * topK_num - 1], and no duplicate index exists.</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>probsOptional (aclTensor)</td>
      <td>Input</td>
      <td>P in the formula.</td>
      <td><ul><li>When probs is passed, topK_num is equal to the second dimension of probs. When probs is not passed, topK_num is 1. </li><li>When probs is passed, topK_num is equal to the second dimension of probs. When probs is not passed, topK_num is 1. </li><li>The shape is (tokens_num, topK_num).</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>paddedMode (bool)</td>
      <td>Input</td>
      <td>Whether to enable paddedMode.</td>
      <td><ul><li>When paddedMode is true, restoreShapeOptional takes effect. Otherwise, no operation is performed on it. </li><li>Currently, only false is supported.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>restoreShapeOptional (aclIntArray)</td>
      <td>Input</td>
      <td>Shape of the output result when paddedMode is true.</td>
      <td><ul><li>When paddedMode is true, restoreShapeOptional takes effect, and the shape of out is represented by restoreShapeOptional. </li><li>Currently, only nullptr is supported.</li></ul></td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor)</td>
      <td>Output</td>
      <td>Output result.</td>
      <td><ul><li>When paddedMode is false, the shape is (tokens_num, hidden_size). When paddedMode is true, the shape is the same as that of restoreShapeOptional. </li><li>The data type is the same as that of permutedTokens.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
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
  </tbody></table>

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown:

  <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 650px">
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
      <tr>
      <td rowspan="2"> ACLNN_ERR_INNER_TILING_ERROR </td>
      <td rowspan="2"> 561002 </td>
      <td>The shapes of multiple input tensors do not match.</td>
      </tr>
      <tr>
      <td>The shape of the input attribute does not match that of the input tensor.</td>
      </tr>
  </tbody>
  </table>

## aclnnMoeTokenUnpermute

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 1030px"> <colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 650px">
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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnMoeTokenUnpermuteGetWorkspaceSize`.</td>
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
  - `aclnnMoeTokenUnpermute` defaults to deterministic implementation.

- <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The value of `topK_num` is less than or equal to `512`.
- Ascend 950PR/Ascend 950DT:
  When this API is called, the framework internally calls the [aclnnMoeFinalizeRoutingV2](../../moe_finalize_routing_v2/docs/aclnnMoeFinalizeRoutingV2_en.md) API. If a parameter error message is displayed, see the following parameter mapping:
  - The permutedTokens input is equivalent to the expandedX input of the aclnnMoeFinalizeRoutingV2 API.
  - The sortedIndices input is equivalent to the expandedRowIdx input of the aclnnMoeFinalizeRoutingV2 API.
  - The probsOptional input is equivalent to the scalesOptional input of the aclnnMoeFinalizeRoutingV2 API.
  - The paddedMode input is equivalent to the dropPadMode input of the aclnnMoeFinalizeRoutingV2 API.
  - The out output is equivalent to the out output of the aclnnMoeFinalizeRoutingV2 API.
- <term>Atlas inference products</term>:
  - The data types supported by `permutedTokens` and `probsOptional` are FLOAT16 and FLOAT32.
  - The value of `topK_num` is less than or equal to `512`.
  - hidden_size is a multiple of 128 and is less than 10240.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp

#include "acl/acl.h"
#include "aclnnop/aclnn_moe_token_unpermute.h"
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

  std::vector<float> permutedTokensData = {1, 2, 3, 4};
  std::vector<int64_t> permutedTokensShape = {2, 2};
  void *permutedTokensAddr = nullptr;
  aclTensor *permutedTokens = nullptr;

  ret = CreateAclTensor(permutedTokensData, permutedTokensShape,
                        &permutedTokensAddr, aclDataType::ACL_FLOAT,
                        &permutedTokens);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int> sortedIndicesData = {0,1};
  std::vector<int64_t> sortedIndicesShape = {2};
  void *sortedIndicesAddr = nullptr;
  aclTensor *sortedIndices = nullptr;

  ret =
      CreateAclTensor(sortedIndicesData, sortedIndicesShape, &sortedIndicesAddr,
                      aclDataType::ACL_INT32, &sortedIndices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<float> probsOptionalData = {1, 1};
  std::vector<int64_t> probsOptionalShape = {1, 2};
  void *probsOptionalAddr = nullptr;
  aclTensor *probsOptional = nullptr;

  ret =
      CreateAclTensor(probsOptionalData, probsOptionalShape, &probsOptionalAddr,
                      aclDataType::ACL_FLOAT, &probsOptional);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<float> outData = {0, 0};
  std::vector<int64_t> outShape = {1, 2};
  void *outAddr = nullptr;
  aclTensor *out = nullptr;

  ret = CreateAclTensor(outData, outShape, &outAddr, aclDataType::ACL_FLOAT,
                        &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;

  // Call the first-phase API of aclnnMoeTokenUnpermute.
  ret = aclnnMoeTokenUnpermuteGetWorkspaceSize(permutedTokens, sortedIndices,
                                               probsOptional, false, nullptr,
                                               out, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnMoeTokenUnpermuteGetWorkspaceSize failed. ERROR: %d\n",
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

  // Call the second-phase API of aclnnMoeTokenUnpermute.
  ret = aclnnMoeTokenUnpermute(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnMoeTokenUnpermute failed. ERROR: %d\n", ret);
            return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  PrintOutResult(outShape, &outAddr);

  // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
  aclDestroyTensor(permutedTokens);
  aclDestroyTensor(sortedIndices);
  aclDestroyTensor(probsOptional);
  aclDestroyTensor(out);

  // 7. Release device resources.
  aclrtFree(permutedTokensAddr);
  aclrtFree(sortedIndicesAddr);
  aclrtFree(probsOptionalAddr);
  aclrtFree(outAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
