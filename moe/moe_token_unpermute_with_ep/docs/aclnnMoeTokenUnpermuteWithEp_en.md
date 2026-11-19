# aclnnMoeTokenUnpermuteWithEp

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:36:00.077Z -->

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

- **Description:** Retrieves the input data from permutedTokens based on the index positions stored in sortedIndices, multiplies it with probs, and performs merge accumulation.

- **Formula:**

  $$
  sortedIndices = sortedIndices[rangeOptional[0]<=i<rangeOptional[1]]
  $$

  (1) When probs is not None, the formula is as follows, where $i \in {0, 1, 2, ..., num\_tokens - 1}$, $j \in {0, 1, 2, ..., topK\_num - 1}$, and $k \in {0, 1, 2, ..., num\_tokens * topK\_num}$:

    $$
    permutedTokens = permutedTokens.indexSelect(0, sortedIndices)
    $$

    $$
    permutedTokens_{k} = permutedTokens_{k} * probs_{i,j}
    $$

    $$
    out_{i} = \sum_{k=i*topK\_num}^{(i+1)*topK\_num - 1 } permutedTokens_{k}
    $$

  (2) When probs is None, the formula is as follows, where $i \in {0, 1, 2, ..., num\_tokens - 1}$ and $j \in {0, 1, 2, ..., topK\_num - 1}$:

    $$
    permutedTokens = permutedTokens.indexSelect(0, sortedIndices)
    $$

    $$
    out_{i} = \sum_{k=i*topK\_num}^{(i+1)*topK\_num - 1 } permutedTokens_{k}
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnMoeTokenUnpermuteWithEpGetWorkspaceSize" API first to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call the "aclnnMoeTokenUnpermuteWithEp" API to execute the computation.

```c++
aclnnStatus aclnnMoeTokenUnpermuteWithEpGetWorkspaceSize(
    const aclTensor   *permutedTokens,
    const aclTensor   *sortedIndices,
    const aclTensor   *probsOptional,
    int64_t            numTopk,
    const aclIntArray *rangeOptional,
    bool               paddedMode,
    const aclIntArray *restoreShapeOptional,
    const aclTensor   *out,
    uint64_t          *workspaceSize,
    aclOpExecutor     **executor);
```

```c++
aclnnStatus aclnnMoeTokenUnpermuteWithEp(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream);
```

## aclnnMoeTokenUnpermuteWithEpGetWorkspaceSize

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
      <td>permutedTokens</td>
      <td>Input</td>
      <td>Represents the expanded and sorted tokens.</td>
      <td>Supports 2D dimensions. Empty tensors are not supported. If rangeOptional is not empty, the length of dimension 0 is not less than rangeOptional[1]-rangeOptional[0].</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>(num_permuted_tokens, hidden_size)</td>
      <td>√</td>
  </tr>
  <tr>
      <td>sortedIndices</td>
      <td>Input</td>
      <td>Represents the positions of the data to be calculated in permutedTokens.</td>
      <td>The shape is 1D with the length of num_tokens*numTopk. Empty tensors are not supported. If an element value is out of the range represented by rangeOptional, the corresponding output contribution is 0.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(num_tokens*numTopk)</td>
      <td>√</td>
  </tr>
  <tr>
      <td>probsOptional</td>
      <td>Optional input</td>
      <td>Represents the expert probability corresponding to the input tokens.</td>
      <td>The shape is 2D. Empty tensors are not supported. num_tokens is the number of original tokens, and the length of the first dimension must be equal to numTopk.<br>When a non-empty and valid tensor is passed in, the input data in permutedTokens is multiplied by probsOptional.<br>When a null pointer is passed in, the input data in permutedTokens is not multiplied.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>(num_tokens, numTopk)</td>
      <td>√</td>
  </tr>
  <tr>
      <td>numTopk</td>
      <td>Input</td>
      <td>Number of selected experts.</td>
      <td>The value must be greater than or equal to 1. When probsOptional is not None, the value must be less than or equal to 512.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
    <tr>
      <td>rangeOptional</td>
      <td>Optional input</td>
      <td>Valid range for EP split, with a size of 2.</td>
      <td>A null pointer can be passed. If a null pointer is passed, the default value {0,0} is used and the output is all 0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  <tr>
      <td>paddedMode</td>
      <td>Input</td>
      <td>-</td>
      <td>true indicates that paddedMode is enabled.<br>false indicates that paddedMode is disabled.<br>Currently only false is supported.</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  <tr>
      <td>restoreShapeOptional</td>
      <td>Optional input</td>
      <td>-</td>
      <td>Reserved parameter. Currently, only a null pointer can be passed.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  <tr>
      <td>out</td>
      <td>Output</td>
      <td>Represents the output result of unpermuting permutedTokens.</td>
      <td>Supports 2D dimensions. Empty tensors are not supported.</td>
      <td>Same as permutedTokens</td>
      <td>ND</td>
      <td>(num_tokens, hidden_size)</td>
      <td>×</td>
  </tr>
  <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator calculation process.</td>
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

  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
      <col style="width: 267px">
      <col style="width: 124px">
      <col style="width: 775px">
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
          <td>permutedTokens, sortedIndices, out, or executor is a null pointer.</td>
          </tr>
          <tr>
          <td> ACLNN_ERR_PARAM_INVALID </td>
          <td> 161002 </td>
          <td>The data type and data format of the input and output are not within the supported range.</td>
          </tr>
      </tbody>
  </table>

## aclnnMoeTokenUnpermuteWithEp

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
          <td>Workspace memory address requested on the Device.</td>
          </tr>
          <tr>
          <td>workspaceSize</td>
          <td>Input</td>
          <td>Size of the workspace requested on the Device, obtained by the first-phase API aclnnMoeTokenUnpermuteWithEpGetWorkspaceSize.</td>
          </tr>
          <tr>
          <td>executor</td>
          <td>Input</td><td> Op executor containing the operator computation flow.</td>
          </tr>
          <tr>
          <td>stream</td>
          <td>Input</td>
          <td> Specifies the stream for executing tasks.</td>
          </tr>
      </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnMoeTokenUnpermuteWithEp defaults to a deterministic implementation.

- numTopk must be greater than or equal to 1. If probsOptional is not null, numTopk must be less than or equal to 512.
- Broadcast is not supported.
- Using -1 in the shape of aclTensor to represent a dynamic dimension or -2 to represent a dynamic rank is not supported.
- paddedMode being `True` is not supported.
- When rangeOptional is empty, the default value {0,0} is used. The output is all zeros and the call does not fall back to other operators.

## Example

The following example shows the basic call process. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp

#include "acl/acl.h"
#include "aclnnop/aclnn_moe_token_unpermute_with_ep.h"
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
  // Boilerplate: resource initialization
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
  // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return ret);

  // 2. Construct the inputs and outputs. Custom construction is required based on the API definition.

  std::vector<float> permutedTokensData = {2, 2, 1, 1, 3, 3, 2, 2};
  std::vector<int64_t> permutedTokensShape = {4, 2};
  void *permutedTokensAddr = nullptr;
  aclTensor *permutedTokens = nullptr;

  ret = CreateAclTensor(permutedTokensData, permutedTokensShape,
                        &permutedTokensAddr, aclDataType::ACL_FLOAT,
                        &permutedTokens);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int> sortedIndicesData = {2, 0, 4, 1, 5, 3};
  std::vector<int64_t> sortedIndicesShape = {6};
  void *sortedIndicesAddr = nullptr;
  aclTensor *sortedIndices = nullptr;

  ret =
      CreateAclTensor(sortedIndicesData, sortedIndicesShape, &sortedIndicesAddr,
                      aclDataType::ACL_INT32, &sortedIndices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<float> probsOptionalData = {1, 1, 1, 1, 1, 1};
  std::vector<int64_t> probsOptionalShape = {3, 2};
  void *probsOptionalAddr = nullptr;
  aclTensor *probsOptional = nullptr;

  ret =
      CreateAclTensor(probsOptionalData, probsOptionalShape, &probsOptionalAddr,
                      aclDataType::ACL_FLOAT, &probsOptional);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  int64_t num_topk = 2;
  void* rangeDeviceAddr = nullptr;
  aclIntArray* range = nullptr;
  std::vector<int64_t> rangeHostData = {1, 5};
  ret = CreateAclIntArray(rangeHostData, &rangeDeviceAddr, &range);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<float> outData = {0, 0, 0, 0, 0, 0};
  std::vector<int64_t> outShape = {3, 2};
  void *outAddr = nullptr;
  aclTensor *out = nullptr;

  ret = CreateAclTensor(outData, outShape, &outAddr, aclDataType::ACL_FLOAT,
                        &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);


  // 3. Call the CANN operator library API and change it to the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;

  // Call the first-phase API of aclnnMoeTokenUnpermuteWithEp.
  ret = aclnnMoeTokenUnpermuteWithEpGetWorkspaceSize(permutedTokens, sortedIndices,
                                                     probsOptional, num_topk, range, false, nullptr,
                                                     out, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnMoeTokenUnpermuteWithEpGetWorkspaceSize failed. ERROR: %d\n",
                ret);
      return ret);

  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // Call the second-phase API of aclnnMoeTokenUnpermuteWithEp.
  ret = aclnnMoeTokenUnpermuteWithEp(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnMoeTokenUnpermuteWithEp failed. ERROR: %d\n", ret);
            return ret);

  // 4. (boilerplate) Synchronously wait for the task execution to finish.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5. Obtain the output value and copy the result from device memory to host memory. Modify the code according to the specific API definition.
  PrintOutResult(outShape, &outAddr);

  // 6. Release aclTensor and aclScalar. Modify the code according to the specific API definition.
  aclDestroyTensor(permutedTokens);
  aclDestroyTensor(sortedIndices);
  aclDestroyTensor(probsOptional);
  aclDestroyTensor(out);

  // 7. Release device resources.
  aclrtFree(permutedTokensAddr);
  aclrtFree(sortedIndicesAddr);
  aclrtFree(probsOptionalAddr);
  aclrtFree(outAddr);
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
