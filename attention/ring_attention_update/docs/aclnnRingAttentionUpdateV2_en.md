# aclnnRingAttentionUpdateV2

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>|    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>|      ×     |
| <term>Atlas inference products</term>|      ×     |
| <term>Atlas training products</term>|      ×     |

## Function

- API function: updates the outputs of two FlashAttentions based on the max and sum of different softmax operations. The difference between this API and the [RingAttentionUpdate](./aclnnRingAttentionUpdate.md) API is as follows: When the input layout is TND, the softmax-related input data layout in the original RingAttentionUpdate API is BNS8. The RingAttentionUpdateV2 API supports the inputSoftmaxLayout parameter to control whether the softmax-related input data layout is consistent with that of attention (that is, the TND layout is used).
- Formulas:

$$
softmax\_max = max(prev\_softmax\_max, cur\_softmax\_max)
$$

$$
softmax\_sum = prev\_softmax\_sum * exp(prev\_softmax\_max - softmax\_max) + cur\_softmax\_sum * exp(cur\_softmax\_max - softmax\_max)
$$

$$
attn\_out = prev\_attn\_out * exp(prev\_softmax\_max - softmax\_max) * prev\_softmax\_sum / softmax\_sum + cur\_attn\_out * exp(cur\_softmax\_max - softmax\_max) * cur\_softmax\_sum / softmax\_sum
$$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnRingAttentionUpdateV2GetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnRingAttentionUpdateV2` is called to perform computation.

```cpp
aclnnStatus aclnnRingAttentionUpdateV2GetWorkspaceSize(
  const aclTensor *prevAttnOut, 
  const aclTensor *prevSoftmaxMax, 
  const aclTensor *prevSoftmaxSum, 
  const aclTensor *curAttnOut, 
  const aclTensor *curSoftmaxMax, 
  const aclTensor *curSoftmaxSum, 
  const aclTensor *actualSeqQlenOptional, 
  char            *inputLayoutOptional, 
  char            *inputSoftmaxLayoutOptional, 
  const aclTensor *attnOutOut, 
  const aclTensor *softmaxMaxOut, 
  const aclTensor *softmaxSumOut, 
  uint64_t        *workspaceSize, 
  aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnRingAttentionUpdateV2(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```

## aclnnRingAttentionUpdateV2GetWorkspaceSize

- **Parameters**
  <table style="undefined;table-layout: fixed; width: 1565px">
      <colgroup>
          <col style="width: 146px">
          <col style="width: 135px">
          <col style="width: 326px">
          <col style="width: 246px">
          <col style="width: 275px">
          <col style="width: 101px">
          <col style="width: 190px">
          <col style="width: 146px">
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
          </tr>
      </thead>
      <tbody>
          <tr>
              <td>prevAttnOut (aclTensor*) </td>
              <td>Input</td>
              <td>prev_attn_out in the formula, which is the output of the first FlashAttention.</td>
              <td>
                  The input shape is the same as that of the inputLayoutOptional attribute.
              </td>
              <td>FLOAT16, FLOAT, BFLOAT16</td>
              <td>ND</td>
              <td>[T,N,D], [S,B,H]</td>
              <td>√</td>
          </tr>
          <tr>
              <td>prevSoftmaxMax (aclTensor*)</td>
              <td>Input</td>
              <td>prev_softmax_max in the formula, which is the max result of the first FlashAttention softmax.</td>
              <td>
                  The eight numbers in the last dimension must be the same and positive.
              </td>
              <td>FLOAT</td>
              <td>ND</td>
              <td>[B,N,S,8], [T,N,8]</td>
              <td>√</td>
          </tr>
          <tr>
              <td>prevSoftmaxSum (aclTensor*)</td>
              <td>Input</td>
              <td>prev_softmax_sum in the formula, which is the sum result of the first FlashAttention softmax.</td>
              <td>
                  The input shape is the same as that of prevSoftmaxMax. The eight numbers in the last dimension must be the same and positive.
              </td>
              <td>FLOAT</td>
              <td>ND</td>
              <td>[B,N,S,8], [T,N,8]</td>
              <td>√</td>
          </tr>
          <tr>
              <td>curAttnOut (aclTensor*)</td>
              <td>Input</td>
              <td>cur_attn_out in the formula, which is the output of the second FlashAttention.</td>
              <td>
                  The data type and input shape are the same as those of prevAttnOut.
              </td>
              <td>FLOAT16, FLOAT, BFLOAT16</td>
              <td>ND</td>
              <td>[T,N,D], [S,B,H]</td>
              <td>√</td>
          </tr>
          <tr>
              <td>curSoftmaxMax (aclTensor*) </td>
              <td>Input</td>
              <td>cur_softmax_max in the formula, which is the max result of the second FlashAttention softmax.</td>
              <td>
                  The input shape must be the same as that of prevSoftmaxMax. The eight numbers in the last dimension must be the same and must be positive numbers.
              </td>
              <td>FLOAT</td>
              <td>ND</td>
              <td>[B,N,S,8], [T,N,8]</td>
              <td>√</td>
          </tr>
          <tr>
              <td>curSoftmaxSum (aclTensor*)</td>
              <td>Input</td>
              <td>cur_softmax_sum in the formula, which is the sum result of the second FlashAttention softmax.</td>
              <td>
                  The input shape must be the same as that of prevSoftmaxMax. The eight numbers in the last dimension must be the same and must be positive numbers.
              </td>
              <td>FLOAT</td>
              <td>ND</td>
              <td>[B,N,S,8], [T,N,8]</td>
              <td>√</td>
          </tr>
          <tr>
              <td>actualSeqQlenOptional (aclTensor*)</td>
              <td>Input</td>
              <td>Accumulated sequence length starting from 0.</td>
              <td>
                  This parameter is mandatory when inputLayoutOptional is set to TND. It is an integer ACL tensor that increases from 0 to T.
              </td>
              <td>INT64</td>
              <td>-</td>
              <td>-</td>
              <td>-</td>
          </tr>
          <tr>
              <td>inputLayoutOptional (char*)</td>
              <td>Input</td>
              <td>Data layout of prevAttnOut and curAttnOut.</td>
              <td>
                  Currently, TND and SBH are supported.
              </td>
              <td>-</td>
              <td>-</td>
              <td>-</td>
              <td>-</td>
          </tr>
          <tr>
              <td>inputSoftmaxLayoutOptional (char*)</td>
              <td>Input</td>
              <td>Data layout of prevSoftmaxMax, prevSoftmaxSum, curSoftmaxMax, and curSoftmaxSum.</td>
              <td>
                  This parameter is valid only when inputLayoutOptional is set to TND. It specifies whether to transpose the inputs related to softmaxMax. Currently, an empty string, SBH, and TND are supported.
              </td>
              <td>-</td>
              <td>-</td>
              <td>-</td>
              <td>-</td>
          </tr>
          <tr>
              <td>attnOutOut (aclTensor*)</td>
              <td>Output</td>
              <td>attn_out in the formula, which is the output after two updates.</td>
              <td>
                  The data type and output shape are the same as those of prevAttnOut.
              </td>
              <td>FLOAT16, FLOAT, BFLOAT16</td>
              <td>ND</td>
              <td>[T,N,D], [S,B,H]</td>
              <td>√</td>
          </tr>
          <tr>
              <td>softmaxMaxOut (aclTensor*)</td>
              <td>Output</td>
              <td>softmax_max in the formula, which is the maximum value of softmax after two result updates.</td>
              <td>
                  The output shape is the same as that of prevSoftmaxMax.
              </td>
              <td>FLOAT</td>
              <td>ND</td>
              <td>[B,N,S,8], [T,N,8]</td>
              <td>√</td>
          </tr>
          <tr>
              <td>softmaxSumOut (aclTensor*)</td>
              <td>Output</td>
              <td>softmax_sum in the formula, which is the sum of softmax after two result updates.</td>
              <td>
                  The output shape is the same as that of prevSoftmaxMax.
              </td>
              <td>FLOAT</td>
              <td>ND</td>
              <td>[B,N,S,8], [T,N,8]</td>
              <td>√</td>
          </tr>
          <tr>
              <td>workspaceSize (uint64_t*)</td>
              <td>Output</td>
              <td>Size of the workspace to be allocated on the device.</td>
              <td>-</td>
              <td>-</td>
              <td>-</td>
              <td>-</td>
              <td>-</td>
          </tr>
          <tr>
              <td>executor (aclOpExecutor**)</td>
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

  - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: When the input data layout (inputLayoutOptional) is TND, D must be a multiple of 64.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.

  <table style="undefined;table-layout: fixed; width: 1218px"><colgroup>
  <col style="width: 325px">
  <col style="width: 124px">
  <col style="width: 769px">
  </colgroup>
  <thead>
    <tr>
      <th>Return</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input prevAttnOut, prevSoftmaxMax, prevSoftmaxSum, curAttnOut, curSoftmaxMax, curSoftmaxSum, attnOutOut, softmaxMaxOut and softmaxSumOut are null pointers.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data types of prevAttnOut, prevSoftmaxMax, prevSoftmaxSum, curAttnOut, curSoftmaxMax, curSoftmaxSum, attnOutOut, softmaxMaxOut and softmaxSumOut are not supported.</td>
    </tr>
    <tr>
      <td>The shapes of prevAttnOut, prevSoftmaxMax, prevSoftmaxSum, curAttnOut, curSoftmaxMax, curSoftmaxSum, attnOutOut, softmaxMaxOut and softmaxSumOut are not supported.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="2">561002</td>
      <td>When actualSeqQlenOptional has an input, the input data format is not supported.</td>
    </tr>
    <tr>
      <td>The input value of inputSoftmaxLayoutOptional is not supported.</td>
    </tr>
  </tbody>
  </table>

## aclnnRingAttentionUpdateV2

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
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
      <td>Size of the workspace to be allocated on the device, which is obtained by the first-phase API aclnnRingAttentionUpdateV2GetWorkspaceSize.</td>
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
  - The deterministic implementation of aclnnRingAttentionUpdateV2 is used by default.
- When `inputLayoutOptional` is TND, `actualSeqQlenOptional` is required.
- When inputLayoutOptional is set to TND:
    - N:
      - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: N <= 256.
    - D:
      - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: D <= 768 and D is a multiple of 64.
- When inputLayoutOptional is set to TND, inputSoftmaxLayoutOptional takes effect. inputSoftmaxLayoutOptional supports only three types of input: empty string, SBH, and TND.

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_ring_attention_update.h"

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
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
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
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the inputs and outputs based on the API definition.
  int64_t batchNum = 1;
  int64_t headNum = 1;
  int64_t seqSize = 2;
  int64_t headDim = 64;
  int64_t headSize = headNum * headDim;

  std::vector<int64_t> prevAttnOutShape = {seqSize, batchNum, headSize};
  std::vector<int64_t> prevSoftmaxMaxShape = {batchNum * seqSize, headNum, 8};
  std::vector<int64_t> prevSoftmaxSumShape = {batchNum * seqSize, headNum, 8};
  std::vector<int64_t> curAttnOutShape = {seqSize, batchNum, headSize};
  std::vector<int64_t> curSoftmaxMaxShape = {batchNum * seqSize, headNum, 8};
  std::vector<int64_t> curSoftmaxSumShape = {batchNum * seqSize, headNum, 8};
  std::vector<int64_t> actualSeqQlenOptionalShape = {batchNum, headNum};

  std::vector<int64_t> attnOutShape = {seqSize, batchNum, headSize};
  std::vector<int64_t> softmaxMaxShape = {batchNum * seqSize, headNum, 8};
  std::vector<int64_t> softmaxSumShape = {batchNum * seqSize, headNum, 8};

  void* prevAttnOutDeviceAddr = nullptr;
  void* prevSoftmaxMaxDeviceAddr = nullptr;
  void* prevSoftmaxSumDeviceAddr = nullptr;
  void* curAttnOutDeviceAddr = nullptr;
  void* curSoftmaxMaxDeviceAddr = nullptr;
  void* curSoftmaxSumDeviceAddr = nullptr;
  void* actualSeqQlenOptionalDeviceAddr = nullptr;

  void* attnOutDeviceAddr = nullptr;
  void* softmaxMaxDeviceAddr = nullptr;
  void* softmaxSumDeviceAddr = nullptr;

  aclTensor* prevAttnOut = nullptr;
  aclTensor* prevSoftmaxMax = nullptr;
  aclTensor* prevSoftmaxSum = nullptr;
  aclTensor* curAttnOut = nullptr;
  aclTensor* curSoftmaxMax = nullptr;
  aclTensor* curSoftmaxSum = nullptr;
  aclTensor* actualSeqQlenOptional = nullptr;

  aclTensor* attnOut = nullptr;
  aclTensor* softmaxMax = nullptr;
  aclTensor* softmaxSum = nullptr;

  std::vector<float> prevAttnOutHostData(seqSize * batchNum * headSize, 1);
  std::vector<float> prevSoftmaxMaxHostData(batchNum * headNum * seqSize * 8, 1);
  std::vector<float> prevSoftmaxSumHostData(batchNum * headNum * seqSize * 8, 1);
  std::vector<float> curAttnOutHostData(seqSize * batchNum * headSize, 1);
  std::vector<float> curSoftmaxMaxHostData(batchNum * headNum * seqSize * 8, 1);
  std::vector<float> curSoftmaxSumHostData(batchNum * headNum * seqSize * 8, 1);
  std::vector<float> actualSeqQlenOptionalHostData(batchNum * headNum, 1);

  std::vector<float> attnOutHostData(seqSize * batchNum * headSize, 1);
  std::vector<float> softmaxMaxHostData(batchNum * headNum * seqSize * 8, 1);
  std::vector<float> softmaxSumHostData(batchNum * headNum * seqSize * 8, 1);

  char* inputLayoutOptional = "TND";
  char* inputSoftmaxLayoutOptional = "TND";
  // Create the prevAttnOut aclTensor.
  ret = CreateAclTensor(prevAttnOutHostData, prevAttnOutShape, &prevAttnOutDeviceAddr, aclDataType::ACL_FLOAT, &prevAttnOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the prevSoftmaxMax aclTensor.
  ret = CreateAclTensor(prevSoftmaxMaxHostData, prevSoftmaxMaxShape, &prevSoftmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &prevSoftmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the prevSoftmaxSum aclTensor.
  ret = CreateAclTensor(prevSoftmaxSumHostData, prevSoftmaxSumShape, &prevSoftmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &prevSoftmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the curAttnOut aclTensor.
  ret = CreateAclTensor(curAttnOutHostData, curAttnOutShape, &curAttnOutDeviceAddr, aclDataType::ACL_FLOAT, &curAttnOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the curSoftmaxMax aclTensor.
  ret = CreateAclTensor(curSoftmaxMaxHostData, curSoftmaxMaxShape, &curSoftmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &curSoftmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the curSoftmaxSum aclTensor.
  ret = CreateAclTensor(curSoftmaxSumHostData, curSoftmaxSumShape, &curSoftmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &curSoftmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the actualSeqQlenOptional aclTensor.
  ret = CreateAclTensor(actualSeqQlenOptionalHostData, actualSeqQlenOptionalShape, &actualSeqQlenOptionalDeviceAddr, aclDataType::ACL_FLOAT, &actualSeqQlenOptional);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the attnOut aclTensor.
  ret = CreateAclTensor(attnOutHostData, attnOutShape, &attnOutDeviceAddr, aclDataType::ACL_FLOAT, &attnOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a softmaxMax aclTensor.
  ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a softmaxSum aclTensor.
  ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Modify the API name to the actual one.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first part of the aclnnRingAttentionUpdateV2 API.
  ret = aclnnRingAttentionUpdateV2GetWorkspaceSize(prevAttnOut, prevSoftmaxMax, prevSoftmaxSum, 
                                                 curAttnOut, curSoftmaxMax, curSoftmaxSum, 
                                                 actualSeqQlenOptional, inputLayoutOptional, inputSoftmaxLayoutOptional,
                                                 attnOut, softmaxMax, softmaxSum, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRingAttentionUpdateV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnRingAttentionUpdateV2.
  ret = aclnnRingAttentionUpdateV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRingAttentionUpdateV2 failed. ERROR: %d\n", ret); return ret);
  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto attnOutSize = GetShapeSize(attnOutShape);
  std::vector<float> attnOutResultData(attnOutSize, 0);
  ret = aclrtMemcpy(attnOutResultData.data(), attnOutResultData.size() * sizeof(attnOutResultData[0]), attnOutDeviceAddr, attnOutSize * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < attnOutSize; i++) {
    LOG_PRINT("attnOutResultData[%ld] is: %f\n", i, attnOutResultData[i]);
  }

  auto softmaxMaxSize = GetShapeSize(softmaxMaxShape);
  std::vector<float> softmaxMaxResultData(softmaxMaxSize, 0);
  ret = aclrtMemcpy(softmaxMaxResultData.data(), softmaxMaxResultData.size() * sizeof(softmaxMaxResultData[0]), softmaxMaxDeviceAddr, softmaxMaxSize * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < softmaxMaxSize; i++) {
    LOG_PRINT("softmaxMaxResultData[%ld] is: %f\n", i, softmaxMaxResultData[i]);
  }

  auto softmaxSumSize = GetShapeSize(softmaxSumShape);
  std::vector<float> softmaxSumResultData(softmaxSumSize, 0);
  ret = aclrtMemcpy(softmaxSumResultData.data(), softmaxSumResultData.size() * sizeof(softmaxSumResultData[0]), softmaxSumDeviceAddr, softmaxSumSize * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < softmaxSumSize; i++) {
    LOG_PRINT("softmaxSumResultData[%ld] is: %f\n", i, softmaxSumResultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(prevAttnOut);
  aclDestroyTensor(prevSoftmaxMax);
  aclDestroyTensor(prevSoftmaxSum);
  aclDestroyTensor(curAttnOut);
  aclDestroyTensor(curSoftmaxMax);
  aclDestroyTensor(curSoftmaxSum);
  aclDestroyTensor(actualSeqQlenOptional);
  aclDestroyTensor(attnOut);
  aclDestroyTensor(softmaxMax);
  aclDestroyTensor(softmaxSum);

  // 7. Release device resources. Modify the code based on the API definition.
  aclrtFree(prevAttnOutDeviceAddr);
  aclrtFree(prevSoftmaxMaxDeviceAddr);
  aclrtFree(prevSoftmaxSumDeviceAddr);
  aclrtFree(curAttnOutDeviceAddr);
  aclrtFree(curSoftmaxMaxDeviceAddr);
  aclrtFree(curSoftmaxSumDeviceAddr);
  aclrtFree(actualSeqQlenOptionalDeviceAddr);
  aclrtFree(attnOutDeviceAddr);
  aclrtFree(softmaxMaxDeviceAddr);
  aclrtFree(softmaxSumDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
