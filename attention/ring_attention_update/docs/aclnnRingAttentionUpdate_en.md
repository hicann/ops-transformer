# aclnnRingAttentionUpdate

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T12:56:44.966Z -->

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

- Description: The RingAttentionUpdate operator updates the outputs of two FlashAttention operations based on their different softmax max and sum values.

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

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnRingAttentionUpdateGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor containing the operator computation process. Then call the "aclnnRingAttentionUpdate" API to execute the computation.

```Cpp
aclnnStatus aclnnRingAttentionUpdateGetWorkspaceSize(
  const aclTensor *prevAttnOut,
  const aclTensor *prevSoftmaxMax,
  const aclTensor *prevSoftmaxSum,
  const aclTensor *curAttnOut,
  const aclTensor *curSoftmaxMax,
  const aclTensor *curSoftmaxSum,
  const aclTensor *actualSeqQlenOptional,
  char            *inputLayoutOptional,
  const aclTensor *attnOutOut,
  const aclTensor *softmaxMaxOut,
  const aclTensor *softmaxSumOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnRingAttentionUpdate(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnRingAttentionUpdateGetWorkspaceSize

- **Parameters:**
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
              <td>                  The input shape must keep consistent with the inputLayoutOptional attribute.
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
              <td>                  The 8 numbers in the last dimension are identical and must be positive numbers.
              </td>
              <td>FLOAT</td>
              <td>ND</td>
              <td>[B,N,S,8], [T,N,8]</td>
              <td>√</td>
          </tr>
          <tr>
              <td>prevSoftmaxSum (aclTensor*)</td>
              <td>Input</td>
              <td>prev_softmax_sum in the formula, the sum result of the softmax in the first FlashAttention.</td>
              <td>                  The input shape keeps consistent with prevSoftmaxMax. The 8 numbers in the last dimension are identical and must be positive numbers.
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
              <td>                  The data type and input shape keep consistent with those of prevAttnOut.
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
              <td>                  The input shape must keep consistent with that of prevSoftmaxMax. The 8 values in the last dimension must be the same and must be positive numbers.
              </td>
              <td>FLOAT</td>
              <td>ND</td>
              <td>[B,N,S,8], [T,N,8]</td>
              <td>√</td>
          </tr>
          <tr>
              <td>curSoftmaxSum (aclTensor*)</td>
              <td>Input</td>
              <td>cur_softmax_sum in the formula, which is the sum result of the softmax of the second FlashAttention.</td>
              <td>                  The input shape must keep consistent with prevSoftmaxMax. The last dimension contains 8 identical numbers, which must be positive numbers.
              </td>
              <td>FLOAT</td>
              <td>ND</td>
              <td>[B,N,S,8], [T,N,8]</td>
              <td>√</td>
          </tr>
          <tr>
              <td>actualSeqQlenOptional (aclTensor*)</td>
              <td>Input</td>
              <td>Cumulative sum of sequence length starting from 0.</td>
              <td>                  Required when the data layout inputLayoutOptional is TND. An aclTensor of integers incrementing from 0 to T.
              </td>
              <td>INT64</td>
              <td>-</td>
              <td>-</td>
              <td>-</td>
          </tr>
          <tr>
              <td>inputLayoutOptional (char*)</td>
              <td>Input</td>
              <td>Data layout of the input related to attn_out.</td>
              <td>                  char* constant on the host. Supported values: "TND" and "SBH".
              </td>
              <td>-</td>
              <td>-</td>
              <td>-</td>
              <td>-</td>
          </tr>
          <tr>
              <td>attnOutOut (aclTensor*)</td>
              <td>Output</td>
              <td>attn_out in the formula. The output after being updated by the two results.</td>
              <td>The data type and output shape keep consistent with prevAttnOut.</td>
              <td>FLOAT16, FLOAT, BFLOAT16</td>
              <td>ND</td>
              <td>[T,N,D], [S,B,H]</td>
              <td>√</td>
          </tr>
          <tr>
              <td>softmaxMaxOut (aclTensor*)</td>
              <td>Output</td>
              <td>softmax_max in the formula, that is, the max of softmax updated based on the two results.</td>
              <td>                  The output shape keeps consistent with prevSoftmaxMax.
              </td>
              <td>FLOAT</td>
              <td>ND</td>
              <td>[B,N,S,8], [T,N,8]</td>
              <td>√</td>
          </tr>
          <tr>
              <td>softmaxSumOut (aclTensor*)</td>
              <td>Output</td>
              <td>softmax_sum in the formula, which is the softmax sum updated based on the two results.</td>
              <td>                  The output shape is consistent with prevSoftmaxMax.
              </td>
              <td>FLOAT</td>
              <td>ND</td>
              <td>[B,N,S,8] and [T,N,8]</td>
              <td>√</td>
          </tr>
          <tr>
              <td>workspaceSize (uint64_t*)</td>
              <td>Output</td>
              <td>Size of the workspace to be applied for on the device.</td>
              <td>-</td>
              <td>-</td>
              <td>-</td>
              <td>-</td>
              <td>-</td>
          </tr>
          <tr>
              <td>executor (aclOpExecutor**)</td>
              <td>Output</td>
              <td>Returns the op executor containing the operator computation process.</td>
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

  <table style="undefined;table-layout: fixed; width: 1154px"><colgroup>
  <col style="width: 276px">
  <col style="width: 122px">
  <col style="width: 756px">
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
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data types of prevAttnOut, prevSoftmaxMax, prevSoftmaxSum, curAttnOut, curSoftmaxMax, curSoftmaxSum, attnOutOut, softmaxMaxOut, and softmaxSumOut are not in the supported range.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>If actualSeqQlenOptional has an input, the input data format is not in the supported range.</td>
    </tr>
    <tr>
      <td>The shapes of prevAttnOut, prevSoftmaxMax, prevSoftmaxSum, curAttnOut, curSoftmaxMax, curSoftmaxSum, attnOutOut, softmaxMaxOut, and softmaxSumOut are empty.</td>
    </tr>
    <tr>
      <td>The shapes of prevAttnOut, prevSoftmaxMax, prevSoftmaxSum, curAttnOut, curSoftmaxMax, curSoftmaxSum, attnOutOut, softmaxMaxOut, and softmaxSumOut are not supported.</td>
    </tr>
  </tbody></table>

## aclnnRingAttentionUpdate

- **Parameters**:

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
      <td>Workspace memory address requested on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size requested on the Device side, obtained by calling the first-phase API aclnnRingAttentionUpdateGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**:

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnRingAttentionUpdate defaults to a deterministic implementation.
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

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

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
  int64_t shape_size = 1;
  for (auto i : shape) {
    shape_size *= i;
  }
  return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // Boilerplate for resource initialization
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
  // Call aclrtMalloc to allocate device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy host data to device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of the contiguous tensor.
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
  // 1. (Boilerplate) Initialize device/stream. For details, see the acl API reference.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as needed.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the inputs and outputs. Customize the construction based on the API.
  int64_t batchNum = 1;
  int64_t headNum = 1;
  int64_t seqSize = 2;
  int64_t headDim = 4;
  int64_t headSize = headNum * headDim;

  std::vector<int64_t> prevAttnOutShape = {seqSize, batchNum, headSize};
  std::vector<int64_t> prevSoftmaxMaxShape = {batchNum, headNum, seqSize, 8};
  std::vector<int64_t> prevSoftmaxSumShape = {batchNum, headNum, seqSize, 8};
  std::vector<int64_t> curAttnOutShape = {seqSize, batchNum, headSize};
  std::vector<int64_t> curSoftmaxMaxShape = {batchNum, headNum, seqSize, 8};
  std::vector<int64_t> curSoftmaxSumShape = {batchNum, headNum, seqSize, 8};
  std::vector<int64_t> actualSeqQlenOptionalShape = {batchNum, headNum};

  std::vector<int64_t> attnOutShape = {seqSize, batchNum, headSize};
  std::vector<int64_t> softmaxMaxShape = {batchNum, headNum, seqSize, 8};
  std::vector<int64_t> softmaxSumShape = {batchNum, headNum, seqSize, 8};

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

  char* inputLayoutOptional = "SBH";
  // Create prevAttnOut aclTensor.
  ret = CreateAclTensor(prevAttnOutHostData, prevAttnOutShape, &prevAttnOutDeviceAddr, aclDataType::ACL_FLOAT, &prevAttnOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create prevSoftmaxMax aclTensor.
  ret = CreateAclTensor(prevSoftmaxMaxHostData, prevSoftmaxMaxShape, &prevSoftmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &prevSoftmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create prevSoftmaxSum aclTensor.
  ret = CreateAclTensor(prevSoftmaxSumHostData, prevSoftmaxSumShape, &prevSoftmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &prevSoftmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create curAttnOut aclTensor.
  ret = CreateAclTensor(curAttnOutHostData, curAttnOutShape, &curAttnOutDeviceAddr, aclDataType::ACL_FLOAT, &curAttnOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create curSoftmaxMax aclTensor.
  ret = CreateAclTensor(curSoftmaxMaxHostData, curSoftmaxMaxShape, &curSoftmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &curSoftmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create curSoftmaxSum aclTensor.
  ret = CreateAclTensor(curSoftmaxSumHostData, curSoftmaxSumShape, &curSoftmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &curSoftmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create actualSeqQlenOptional aclTensor.
  ret = CreateAclTensor(actualSeqQlenOptionalHostData, actualSeqQlenOptionalShape, &actualSeqQlenOptionalDeviceAddr, aclDataType::ACL_INT64, &actualSeqQlenOptional);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create attnOut aclTensor.
  ret = CreateAclTensor(attnOutHostData, attnOutShape, &attnOutDeviceAddr, aclDataType::ACL_FLOAT, &attnOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create softmaxMax aclTensor.
  ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create softmaxSum aclTensor.
  ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call CANN operator library APIs. Replace them with the specific APIs.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnRingAttentionUpdate.
  ret = aclnnRingAttentionUpdateGetWorkspaceSize(prevAttnOut, prevSoftmaxMax, prevSoftmaxSum,
                                                 curAttnOut, curSoftmaxMax, curSoftmaxSum,
                                                 actualSeqQlenOptional, inputLayoutOptional,
                                                 attnOut, softmaxMax, softmaxSum, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRingAttentionUpdateGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnRingAttentionUpdate.
  ret = aclnnRingAttentionUpdate(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRingAttentionUpdate failed. ERROR: %d\n", ret); return ret);
  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value, and copy the result from the device-side memory to the host side. Modify based on the specific API definition.
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

  // 6. Release aclTensor and aclScalar. Modify based on the specific API definition.
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

  // 7. Release device resources. Modify based on the specific API definition.
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
