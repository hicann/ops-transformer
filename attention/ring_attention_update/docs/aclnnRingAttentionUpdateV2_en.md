# aclnnRingAttentionUpdateV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T12:58:49.164Z -->

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

- Description: Updates the outputs of two FlashAttention calls based on their different softmax max and sum values. **The difference from the [RingAttentionUpdate](./aclnnRingAttentionUpdate_en.md) API is as follows: When the input layout is TND, the softmax-related inputs of the original RingAttentionUpdate API use the BNS8 data layout, while the RingAttentionUpdateV2 API supports the string parameter inputSoftmaxLayout to control whether the data layout of softmax-related inputs is consistent with that of attention (that is, uses the TND layout).**
- Formula:

$$
softmax\_max = max(prev\_softmax\_max, cur\_softmax\_max)
$$

$$
softmax\_sum = prev\_softmax\_sum * exp(prev\_softmax\_max - softmax\_max) + cur\_softmax\_sum * exp(cur\_softmax\_max - softmax\_max)
$$

$$
attn\_out = prev\_attn\_out * exp(prev\_softmax\_max - softmax\_max) * prev\_softmax\_sum / softmax\_sum + cur\_attn\_out * exp(cur\_softmax\_max - softmax\_max) * cur\_softmax\_sum / softmax\_sum
$$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnRingAttentionUpdateV2GetWorkspaceSize" API first to obtain the workspace size required for computation and the executor containing the operator computation process, and then call the "aclnnRingAttentionUpdateV2" API to perform the computation.

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
              <th>Parameter</th>
              <th>Input/Output</th>
              <th>Description</th>
              <th>Instruction</th>
              <th>Data Type</th>
              <th>Data Format</th>
              <th>Dimension (shape)</th>
              <th>Non-contiguous Tensor</th>
          </tr>
      </thead>
      <tbody>
          <tr>
              <td>prevAttnOut (aclTensor*)</td>
              <td>Input</td>
              <td>prev_attn_out in the formula, the output of the first FlashAttention.</td>
              <td>                  The input shape is consistent with the inputLayoutOptional attribute.
              </td>
              <td>FLOAT16, FLOAT, BFLOAT16</td>
              <td>ND</td>
              <td>[T,N,D], [S,B,H]</td>
              <td>√</td>
          </tr>
          <tr>
              <td>prevSoftmaxMax (aclTensor*)</td>
              <td>Input</td>
              <td>prev_softmax_max in the formula, the max result of the softmax of the first FlashAttention.</td>
              <td>                  The 8 numbers in the last dimension are identical and must be positive.
              </td>
              <td>FLOAT</td>
              <td>ND</td>
              <td>[B,N,S,8], [T,N,8]</td>
              <td>√</td>
          </tr>
          <tr>
              <td>prevSoftmaxSum (aclTensor*)</td>
              <td>Input</td>
              <td>prev_softmax_sum in the formula, indicating the softmax sum result of the first FlashAttention.</td>
              <td>                  The input shape is the same as that of prevSoftmaxMax. The 8 numbers in the last dimension are identical and must be positive.
              </td>
              <td>FLOAT</td>
              <td>ND</td>
              <td>[B,N,S,8], [T,N,8]</td>
              <td>√</td>
          </tr>
          <tr>
              <td>curAttnOut (aclTensor*)</td>
              <td>Input</td>
              <td>cur_attn_out in the formula, the output of the second FlashAttention.</td>
              <td>                  The data type and input shape are the same as those of prevAttnOut.
              </td>
              <td>FLOAT16, FLOAT, BFLOAT16</td>
              <td>ND</td>
              <td>[T,N,D], [S,B,H]</td>
              <td>√</td>
          </tr>
          <tr>
              <td>curSoftmaxMax (aclTensor*)</td>
              <td>Input</td>
              <td>cur_softmax_max in the formula. Max result of the softmax of the second FlashAttention.</td>
              <td>                  The input shape is the same as that of prevSoftmaxMax. The eight numbers in the last dimension are identical and must be positive.
              </td>
              <td>FLOAT</td>
              <td>ND</td>
              <td>[B,N,S,8], [T,N,8]</td>
              <td>√</td>
          </tr>
          <tr>
              <td>curSoftmaxSum (aclTensor*)</td>
              <td>Input</td>
              <td>cur_softmax_sum in the formula. Sum result of the softmax of the second FlashAttention.</td>
              <td>                  The input shape must be the same as that of prevSoftmaxMax. The 8 values in the last dimension must be the same and positive.
              </td>
              <td>FLOAT</td>
              <td>ND</td>
              <td>[B,N,S,8] or [T,N,8]</td>
              <td>√</td>
          </tr>
          <tr>
              <td>actualSeqQlenOptional (aclTensor*)</td>
              <td>Input</td>
              <td>Accumulated sequence length starting from 0.</td>
              <td>                  When the data layout inputLayoutOptional is TND, this parameter is required. It is an integer aclTensor that increments from 0 to T.
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
              <td>                  Currently supports "TND" and "SBH".
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
              <td>                  When inputLayoutOptional is TND, this switch controls whether to transpose softmaxMax-related inputs. Currently, empty string, "SBH", and "TND" are supported.
              </td>
              <td>-</td>
              <td>-</td>
              <td>-</td>
              <td>-</td>
          </tr>
          <tr>
              <td>attnOutOut (aclTensor*)</td>
              <td>Output</td>
              <td>attn_out in the formula. Output updated from the two results.</td>
              <td>                  Data type and output shape are the same as those of prevAttnOut.
              </td>
              <td>FLOAT16, FLOAT, BFLOAT16</td>
              <td>ND</td>
              <td>[T,N,D], [S,B,H]</td>
              <td>√</td>
          </tr>
          <tr>
              <td>softmaxMaxOut (aclTensor*)</td>
              <td>Output</td>
              <td>softmax_max in the formula, that is, the softmax max updated using the two results.</td>
              <td>                  The output shape is the same as that of prevSoftmaxMax.
              </td>
              <td>FLOAT</td>
              <td>ND</td>
              <td>[B,N,S,8], [T,N,8]</td>
              <td>√</td>
          </tr>
          <tr>
              <td>softmaxSumOut (aclTensor*)</td>
              <td>Output</td>
              <td>softmax_sum in the formula, that is, the softmax sum updated using the two results.</td>
              <td>The output shape is the same as that of prevSoftmaxMax.</td>
              <td>FLOAT</td>
              <td>ND</td>
              <td>[B,N,S,8], [T,N,8]</td>
              <td>√</td>
          </tr>
          <tr>
              <td>workspaceSize (uint64_t*)</td>
              <td>Output</td>
              <td>Size of the workspace to be allocated on the Device.</td>
              <td>-</td>
              <td>-</td>
              <td>-</td>
              <td>-</td>
              <td>-</td>
          </tr>
          <tr>
              <td>executor(aclOpExecutor**)</td>
              <td>Output</td>
              <td>Returns an op executor that contains the operator computation flow.</td>
              <td>-</td>
              <td>-</td>
              <td>-</td>
              <td>-</td>
              <td>-</td>
          </tr>
      </tbody>
  </table>

  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: When the input data layout inputLayoutOptional is TND, D is constrained to be a multiple of 64.

  <!-- end id7 -->

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1218px"><colgroup>
  <col style="width: 325px">
  <col style="width: 124px">
  <col style="width: 769px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>prevAttnOut, prevSoftmaxMax, prevSoftmaxSum, curAttnOut, curSoftmaxMax, curSoftmaxSum, attnOutOut, softmaxMaxOut, or softmaxSumOut is a null pointer; or actualSeqQlenOptional is a null pointer when inputLayoutOptional is TND.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data types of prevAttnOut, prevSoftmaxMax, prevSoftmaxSum, curAttnOut, curSoftmaxMax, curSoftmaxSum, attnOutOut, softmaxMaxOut, and softmaxSumOut are not within the supported range.</td>
    </tr>
    <tr>
      <td>The shapes of prevAttnOut, prevSoftmaxMax, prevSoftmaxSum, curAttnOut, curSoftmaxMax, curSoftmaxSum, attnOutOut, softmaxMaxOut, and softmaxSumOut are not supported.</td>
    </tr>
    <tr>
      <td>When actualSeqQlenOptional has an input, the input data format is not within the supported range.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>When the input value of inputSoftmaxLayoutOptional is not within the supported range.</td>
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
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device. It is obtained by the first-phase API aclnnRingAttentionUpdateV2GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation flow.</td>
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

- Deterministic computation:
  - aclnnRingAttentionUpdateV2 defaults to a deterministic implementation.
- When inputLayoutOptional is set to "TND", actualSeqQlenOptional is mandatory.
- When inputLayoutOptional is set to "TND":
    - N:

      <!-- npu="A3,910b,950" id8 -->
      - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>: N<=256.
      <!-- end id8 -->

    - D:

      <!-- npu="A3,910b" id9 -->
      - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: D<=768 and D is a multiple of 64.
      <!-- end id9 -->
      <!-- npu="950" id10 -->
      - <term>Ascend 950PR&950DT products</term>: D<=768.
      <!-- end id10 -->

- **inputSoftmaxLayoutOptional takes effect only when inputLayoutOptional is set to "TND". inputSoftmaxLayoutOptional supports only three inputs: an empty string, "SBH", and "TND".**

## Example

The following is sample code for reference only. For details about the compilation and execution procedure, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

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
  // Boilerplate for resource initialization.
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

  // Call aclrtMemcpy to copy data from the host to the device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of a contiguous tensor.
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
  // 1. (Boilerplate) Initialize device/stream. For details, see the acl API documentation.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct inputs and outputs. You need to customize the construction based on the API.
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
  ret = CreateAclTensor(actualSeqQlenOptionalHostData, actualSeqQlenOptionalShape, &actualSeqQlenOptionalDeviceAddr, aclDataType::ACL_INT64, &actualSeqQlenOptional);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the attnOut aclTensor.
  ret = CreateAclTensor(attnOutHostData, attnOutShape, &attnOutDeviceAddr, aclDataType::ACL_FLOAT, &attnOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the softmaxMax aclTensor.
  ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the softmaxSum aclTensor.
  ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Change it to the actual API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the aclnnRingAttentionUpdateV2 first-phase API.
  ret = aclnnRingAttentionUpdateV2GetWorkspaceSize(prevAttnOut, prevSoftmaxMax, prevSoftmaxSum,
                                                 curAttnOut, curSoftmaxMax, curSoftmaxSum,
                                                 actualSeqQlenOptional, inputLayoutOptional, inputSoftmaxLayoutOptional,
                                                 attnOut, softmaxMax, softmaxSum, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRingAttentionUpdateV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the aclnnRingAttentionUpdateV2 second-phase API.
  ret = aclnnRingAttentionUpdateV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRingAttentionUpdateV2 failed. ERROR: %d\n", ret); return ret);
  // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output values and copy the results from device memory to host memory. Modify the code based on the specific API definition.
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

  // 6. Release aclTensor and aclScalar. Modify the code based on the specific API definition.
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

  // 7. Release device resources. Modify the code based on the specific API definition.
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
