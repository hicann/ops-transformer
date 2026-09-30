# aclnnMoeGatingTopKSoftmaxV2

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- Description: If `renorm` = 0, performs the Softmax operation on the output of `x` before obtaining the topK value. If `renorm` = 1, performs the Softmax operation on the output of `x` after obtaining the topK value. `yOut` is the topK result of softmax. `expertIdxOut` is the index result of the topK value, that is, the corresponding expert sequence number. If the corresponding row is marked as `true`, the expert sequence number is directly filled with the value of `num_expert` (that is, the size of the last axis of `x`).
- Formula:
    1. renorm = 0,

        $$
        softmaxResultOutOptional=softmax(x,axis=-1)
        $$

        $$
        yOut,expertIdxOut=topK(softmaxResultOutOptional,k=k)
        $$

    2. renorm = 1

        $$
        topkOut,expertIdxOut=topK(x, k=k)
        $$

        $$
        yOut = softmax(topkOut,axis=-1)
        $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnMoeGatingTopKSoftmaxV2GetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnMoeGatingTopKSoftmaxV2` is called to perform computation.

```c++
aclnnStatus aclnnMoeGatingTopKSoftmaxV2GetWorkspaceSize(
    const aclTensor *x, 
    const aclTensor *finishedOptional, 
    int64_t          k, 
    int64_t          renorm, 
    bool             outputSoftmaxResultFlag, 
    const aclTensor *yOut, 
    const aclTensor *expertIdxOut, 
    const aclTensor *softmaxResultOutOptional,
    uint64_t        *workspaceSize, 
    aclOpExecutor  **executor)
```

```c++
aclnnStatus aclnnMoeGatingTopKSoftmaxV2(
    void          *workspace, 
    uint64_t       workspaceSize, 
    aclOpExecutor *executor, 
    aclrtStream    stream)
```

## aclnnMoeGatingTopKSoftmaxV2GetWorkspaceSize

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
        <td>x</td>
        <td>Input</td>
        <td>x in the formula to be calculated.</td>
        <td>The input must be a 2D or 3D tensor, and each shape must be less than or equal to the maximum value of int32 (2147483647).</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>2-3</td>
        <td>√</td>
    </tr>
    <tr>
        <td>finishedOptional</td>
        <td>Input</td>
        <td>finished in the formula, indicating whether the row is involved in the computation.</td>
        <td>The shape is x_shape[:-1].</td>
        <td>BOOL</td>
        <td>ND</td>
        <td>1-2</td>
        <td>√</td>
    </tr>
    <tr>
        <td>k</td>
        <td>Input</td>
        <td>Value of k in topK, number of experts.</td>
        <td>0 ≤ k ≤ Axis -1 of x. The value of k must be less than or equal to 1024.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>renorm</td>
        <td>Input</td>
        <td>Sequence of Softmax and TopK.</td>
        <td>The value can only be 0 or 1. 0 indicates that Softmax is calculated before TopK. 1 indicates the opposite order.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>outputSoftmaxResultFlag</td>
        <td>Input</td>
        <td>Whether to output the softmax result.</td>
        <td>When renorm is set to 0, true indicates that the softmax result is output, and false indicates that the softmax result is not output. When renorm is set to 1, this parameter does not take effect, and the softmax result is not output.</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>yOut</td>
        <td>Output</td>
        <td>TopK value obtained after Softmax is performed on x.</td>
        <td>The non- −1 dimensions of its shape must match the corresponding dimensions of x, and the size of the −1 dimension must equal k.</td>
        <td>The value must be the same as that of the input x.</td>
        <td>ND</td>
        <td>2-3</td>
        <td>x</td>
    </tr>
    <tr>
        <td>softmaxResultOutOptional</td>
        <td>Output</td>
        <td>Softmax result during computation.</td>
        <td>The shape must be the same as that of x.</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>2-3</td>
        <td>x</td>
    </tr>
    <tr>
        <td>rowIdxOut</td>
        <td>Output</td>
        <td>scales in the formula.</td>
        <td>The shape must be the same as that of yOut.</td>
        <td>INT32</td>
        <td>ND</td>
        <td>2-3</td>
        <td>x</td>
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

- **Returns**:

    `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API performs input parameter validation. The following errors may be returned:

    <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
    <col style="width: 253px">
    <col style="width: 140px">
    <col style="width: 762px">
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
        <td>The input and output data types and formats are not supported.</td>
        </tr>
        <tr>
        <td rowspan="8"> ACLNN_ERR_INNER_TILING_ERROR </td>
        <td rowspan="8"> 561002 </td>
        <td>The shapes of multiple input tensors do not match.</td>
        </tr>
        <tr>
        <td>The shape of the input attribute does not match that of the input tensor.</td>
        </tr>
        <tr>
        <td>x is not a 2D or 3D tensor.</td>
        </tr>
        <tr>
        <td>The shape of x does not match that of `finishedOptional`.</td>
        </tr>
        <tr>
        <td>The value of k is less than 0 or greater than the size of axis –1 of x.</td>
        </tr>
        <tr>
        <td>The value of k is greater than 1024.</td>
        </tr>
        <tr>
        <td>The value of renorm is not 0 or 1.</td>
        </tr>
        <tr>
        <td>The data type of `softmaxResultOutOptional` is not supported.</td>
        </tr>
    </tbody></table>

## aclnnMoeGatingTopKSoftmax

- **Parameters:**
    <table>
            <thead>
                <tr><th>Parameter</th><th>Input/Output</th><th>Description</th></tr>
            </thead>
            <tbody>
                <tr><td>workspace</td><td>Input</td><td>Address of the workspace memory allocated on the device.</td></tr>
                <tr><td>workspaceSize</td><td>Input</td><td>The workspace size allocated on the device side, obtained from the first API `aclnnInplaceAddGetWorkspaceSize`.</td></tr>
                <tr><td>executor</td><td>Input</td><td>The operator executor, which contains the computation process of the operator. </td></tr>
                <tr><td>stream</td><td>Input</td><td>Stream for executing a task. </td></tr>
            </tbody>
        </table>

- **Returns**
  
  `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computing:
  - `aclnnMoeGatingTopKSoftmaxV2` defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_gating_top_k_softmax_v2.h"
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
  // (Boilerplate) Initialize resources.
  auto  ret = aclInit(nullptr);
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

int main() {
  // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as required.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2.Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> inputShape = {3, 4};
  std::vector<int64_t> outShape = {3, 2};
  std::vector<int64_t> expertIdOutShape = {3, 2};
  std::vector<int64_t> softmaxResultOutOptionalShape = {3, 4};

  void* inputAddr = nullptr;
  void* outAddr = nullptr;
  void* expertIdOutAddr = nullptr;
  void* softmaxResultOutOptionalAddr = nullptr;

  aclTensor* input = nullptr;
  aclTensor* out = nullptr;
  aclTensor* expertIdOut = nullptr;
  aclTensor* softmaxResultOutOptional = nullptr;

  std::vector<float> inputHostData = {0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1};
  std::vector<float> outHostData = {0.1, 1.1, 2.1, 3.1, 4.1, 5.1};
  std::vector<int32_t> expertIdOutHostData = {1, 1, 1, 1, 1, 1};
  std::vector<int32_t> softmaxResultOutOptionalHostData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  // Create an expandedPermutedRows aclTensor.
  ret = CreateAclTensor(inputHostData, inputShape, &inputAddr, aclDataType::ACL_FLOAT, &input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an expertForSourceRow aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an expandedSrcToDstRow aclTensor.
  ret = CreateAclTensor(expertIdOutHostData, expertIdOutShape, &expertIdOutAddr, aclDataType::ACL_INT32, &expertIdOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a softmaxResultOutOptional aclTensor.
  ret = CreateAclTensor(softmaxResultOutOptionalHostData, softmaxResultOutOptionalShape, &softmaxResultOutOptionalAddr, aclDataType::ACL_FLOAT, &softmaxResultOutOptional);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with a specific operator API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnMoeGatingTopKSoftmaxV2.
  ret = aclnnMoeGatingTopKSoftmaxV2GetWorkspaceSize(input, nullptr, 2, 0, true, out, expertIdOut, softmaxResultOutOptional, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeGatingTopKSoftmaxV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnMoeGatingTopKSoftmaxV2.
  ret = aclnnMoeGatingTopKSoftmaxV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeGatingTopKSoftmaxV2 failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0.0f);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                    outAddr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Destroy aclTensor objects. Modify the code based on the API definition.
  aclDestroyTensor(input);
  aclDestroyTensor(out);
  aclDestroyTensor(expertIdOut);
  aclDestroyTensor(softmaxResultOutOptional);

  // 7. Free device resources. Modify the configuration based on the API definition.
  aclrtFree(inputAddr);
  aclrtFree(outAddr);
  aclrtFree(expertIdOutAddr);
  aclrtFree(softmaxResultOutOptionalAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
