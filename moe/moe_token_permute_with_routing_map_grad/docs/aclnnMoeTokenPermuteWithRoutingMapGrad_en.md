# aclnnMoeTokenPermuteWithRoutingMapGrad

[📄 View Source Code](https://gitcode.com/cann/ops-transformer/tree/master/moe/moe_token_permute_with_routing_map_grad)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- **Description**: Computes the backpropagation of `aclnnMoeTokenPermuteWithRoutingMap`.
- **Formula**:

    $$
    permuteTokenId, outIndex= sortedIndices.sort(dim=-1)
    $$

    $$
    capacity = permutedTokensOutputGrad.size(0) / numExperts
    $$

    - When `probs` is not set to `None`:

        $$
        probsGradOutOptional = zeros(tokens_num, numExperts)
        $$

    - When dropPaddedMode is true:

        $$
        probsGradOutOptional [sortedIndices[i], i/capacity] = permutedProbsOutputGradOptional[i]
        $$

    - When dropPaddedMode is false:

        $$
        probsGradOutOptional = maskedscatter(probsGradOutOptional,routingMapOptional, permutedProbsOutputGradOptional)
        $$
    - If `probs` is set to `None`:

        $$
        tokensGradOut= zeros(restoreShapeOptional, dtype=permutedTokens.dtype, device=permutedTokens.device)
        $$

        $$
        tokensGradOut[permuteTokenId[i]] += permutedTokens[outIndex[i]]
        $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnMoeTokenPermuteWithRoutingMapGradGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnMoeTokenPermuteWithRoutingMapGrad` is called to perform computation.

```c++
aclnnStatus aclnnMoeTokenPermuteWithRoutingMapGradGetWorkspaceSize(
    const aclTensor *permutedTokensOutputGrad,
    const aclTensor *permutedProbsOutputGradOptional,
    const aclTensor *sortedIndices,
    const aclTensor *routingMapOptional,
    int64_t          experts_num,
    int64_t          tokens_num,
    bool             dropAndPad,
    aclTensor       *tokenGradOut,
    aclTensor       *probsGradOutOptional,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```c++
aclnnStatus aclnnMoeTokenPermuteWithRoutingMapGrad(
    void                *workspace,
    uint64_t             workspaceSize,
    aclOpExecutor       *executor,
    const aclrtStream    stream)
```

## aclnnMoeTokenPermuteWithRoutingMapGradGetWorkspaceSize

- **Parameters**:
    <table style="undefined;table-layout: fixed; width: 1580px"><colgroup>
    <col style="width: 231px">
    <col style="width: 120px">
    <col style="width: 242px">
    <col style="width: 332px">
    <col style="width: 161px">
    <col style="width: 121px">
    <col style="width: 228px">
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
      <td>permutedTokensOutputGrad</td>
      <td>Input</td>
      <td>Gradient of forward output <code>permutedTokens</code>.</td>
      <td>The shape supports 2D dimensions and does not support empty tensors. topK_num indicates the number of experts selected for each token, and capacity indicates the number of tokens selected for each expert.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td><ul><li>Non-dropAndPad mode: (tokens_num * topK_num, hidden_size); </li><li>dropAndPad mode: (experts_num * capacity, hidden_size).</li></ul></td>
      <td>√</td>
  </tr>
  <tr>
      <td>permutedProbsOutputGradOptional</td>
      <td>Optional input</td>
      <td>Gradient of the forward output <code>permutedProbs</code>.</td>
      <td><ul><li>If this parameter is not passed, it indicates that probsGradOutOptional does not need to be calculated.</li><li>The shape is a 1D dimension, topK_num indicates the number of experts selected for each token, and capacity indicates the number of tokens selected for each expert.</li><li>The data type is the same as that of permutedTokensOutputGrad or FLOAT is supported when permutedTokensOutputGrad is BFLOAT16.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td><ul><li>Non-dropAndPad mode: (tokens_num * topK_num);</li><li>dropAndPad mode: (experts_num * capacity).</li></ul></td>
      <td>√</td>
  </tr>
  <tr>
      <td>sortedIndices</td>
      <td>Input</td>
      <td>Sorted index value.</td>
      <td>In non-dropAndPad mode, the index value range is [0, tokens_num * topK_num – 1]. In dropAndPad mode, the index value range is [0, experts_num * capacity – 1]. topK_num indicates the number of experts selected for each token, and capacity indicates the number of tokens selected for each expert.</td>
      <td>INT32</td>
      <td>ND</td>
      <td><ul><li>Non-dropAndPad mode: (tokens_num * topK_num,);</li><li>dropAndPad mode: (experts_num * capacity)</li></ul></td>
      <td>√</td>
  </tr>
  <tr>
      <td>routingMapOptional</td>
      <td>Optional input</td>
      <td>Indicates the mapping from tokens to experts.</td>
      <td>The shape must be a 2D tensor. In non-dropAndPad mode, each row must contain topK `true` or `1` values.</td>
      <td>INT8, bool (When the data type is INT8, the value can be 0 or 1. When the data type is bool, the value can be true or false.)</td>
      <td>ND</td>
      <td> (tokens_num, experts_num)</td>
      <td>√</td>
  </tr>
  <tr>
      <td>experts_num</td>
      <td>Input</td>
      <td>Indicates the number of experts involved in the computation.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
      <tr>
      <td>tokens_num</td>
      <td>Input</td>
      <td>Number of tokens involved in the operation.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  <tr>
      <td>dropAndPad</td>
      <td>Input</td>
      <td>-</td>
      <td>true indicates that dropPaddedMode is enabled, and false indicates that dropPaddedMode is disabled.</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  <tr>
      <td>tokensGradOut</td>
      <td>Output</td>
      <td>Gradient of the input permutedTokens.</td>
      <td>The value must be a 2D tensor.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td> (tokens_num, hidden_size)</td>
      <td>×</td>
  </tr>
  <tr>
      <td>probsGradOutOptional</td>
      <td>Optional output</td>
      <td>Gradient of the input probs of the forward operator.</td>
      <td>The 2D shape is supported.</td>
      <td>The values are the same as those of permutedProbsOutputGradOptional</td>
      <td>ND</td>
      <td> (tokens_num, experts_num)</td>
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

  The first-phase API implements input parameter verification. The following errors may be thrown:

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
          <td>
              1. The data types and formats of the input and output are not supported.<br>
              2. The input and output shapes are not supported.
          </td>
          </tr>
      </tbody>
  </table>

## aclnnMoeTokenPermuteWithRoutingMapGrad

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
          <td>Size of the workspace to be allocated on the device, which is obtained by the first-phase API aclnnMoeTokenPermuteWithRoutingMapGradGetWorkspaceSize.</td>
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
  - `aclnnMoeTokenPermuteWithRoutingMapGrad` defaults to deterministic implementation.

- Non-dropPaddedMode scenario: The value of `topK_num` is less than or equal to `512`.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include "aclnnop/aclnn_moe_token_permute_with_routing_map_grad.h"
#include <iostream>
#include <vector>
#include <sys/stat.h>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cassert>
#include <iomanip>
#include <unistd.h>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

template <typename T>
bool WriteFile(const std::string &filePath, int64_t size, std::vector<T>& hostData)
{
    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    if (fd < 0) {
        LOG_PRINT("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    size_t writeSize = write(fd, reinterpret_cast<char*>(hostData.data()), size * sizeof(T));
    (void)close(fd);
    if (writeSize != size * sizeof(T)) {
        LOG_PRINT("Write file Failed.");
        return false;
    }

    return true;
}
void PrintOutResult(std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < 10; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
}

int Init(int32_t deviceId, aclrtStream* stream)
{
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
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

int main()
{

    // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
    // Set the device ID in use.
    int32_t deviceId = 0;
    aclrtStream stream;

    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct inputs and outputs based on the API definition.

    int64_t num_token = 4096;
    int64_t hidden_size = 7168;
    int64_t num_expert = 256;
    int64_t num_capacity = 16;
    std::vector<float> permuted_output_grad_Data(num_expert * num_capacity * hidden_size, 0);
    std::vector<int64_t> permuted_output_grad_Shape = {num_expert * num_capacity, hidden_size};
    void* permuted_output_grad_Addr = nullptr;
    aclTensor* permuted_output_grad = nullptr;

    ret = CreateAclTensor(permuted_output_grad_Data, permuted_output_grad_Shape, &permuted_output_grad_Addr,
                          aclDataType::ACL_FLOAT, &permuted_output_grad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> permutedProbsOutputGradOptional(num_expert * num_capacity, 0.1);
    std::vector<int64_t> permutedProbsOutputGradOptionalShape = {num_expert * num_capacity};
    void* permutedProbsOutputGrad_Addr = nullptr;
    aclTensor* ppermutedProbsOutputGrad = nullptr;
    ret = CreateAclTensor(permutedProbsOutputGradOptional, permutedProbsOutputGradOptionalShape,
                          &permutedProbsOutputGrad_Addr, aclDataType::ACL_FLOAT, &ppermutedProbsOutputGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int> sortedIndicesData(num_expert * num_capacity, 0);
    std::vector<int64_t> sortedIndicesShape = {num_expert * num_capacity};
    void* sortedIndicesAddr = nullptr;
    aclTensor* sortedIndices = nullptr;
    ret = CreateAclTensor(sortedIndicesData, sortedIndicesShape, &sortedIndicesAddr, aclDataType::ACL_INT32,
                          &sortedIndices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<char> routingMapOptionalData(num_token * num_expert, 1);
    std::vector<int64_t> routingMapOptionalShape = {num_token , num_expert};
    void* routingMapOptionalAddr = nullptr;
    aclTensor* proutingMapOptional = nullptr;

    ret = CreateAclTensor(routingMapOptionalData, routingMapOptionalShape, &routingMapOptionalAddr,
                          aclDataType::ACL_INT8, &proutingMapOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> outData(num_token * hidden_size, 0.0f);
    std::vector<int64_t> outShape = {num_token, hidden_size};
    // std::vector<int64_t> outShape = {num_token};
    void* outAddr = nullptr;
    aclTensor* out = nullptr;

    ret = CreateAclTensor(outData, outShape, &outAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> outData2(num_token * num_expert, 0.0f);
    std::vector<int64_t> outShape2 = {num_token, num_expert};
    void* outAddr2 = nullptr;
    aclTensor* out2 = nullptr;

    ret = CreateAclTensor(outData2, outShape2, &outAddr2, aclDataType::ACL_FLOAT, &out2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // Call the first-phase API of aclnnMoeTokenPermuteWithRoutingMapGrad.
    ret = aclnnMoeTokenPermuteWithRoutingMapGradGetWorkspaceSize(permuted_output_grad, ppermutedProbsOutputGrad, sortedIndices, proutingMapOptional, num_expert, num_token, true,
                                                                 out, out2, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnMoeTokenPermuteWithRoutingMapGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // Call the second-phase API of aclnnMoeTokenPermuteWithRoutingMapGrad.
    ret = aclnnMoeTokenPermuteWithRoutingMapGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTokenPermuteWithRoutingMapGrad failed. ERROR: %d\n", ret);
              return ret);

    // 4. (Boilerplate) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
    PrintOutResult(outShape, &outAddr);
    PrintOutResult(outShape2, &outAddr2);

    // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
    aclDestroyTensor(permuted_output_grad);
    aclDestroyTensor(sortedIndices);
    aclDestroyTensor(out);

    // 7. Release device resources.
    aclrtFree(permuted_output_grad_Addr);
    aclrtFree(sortedIndicesAddr);
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
