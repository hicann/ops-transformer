# aclnnRingAttentionUpdateV2

## Supported Products

| Product                                                      | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>     |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function

- Description: Updates the outputs of two FlashAttention operations based on their different softmax max and sum values. **The difference from the [RingAttentionUpdate](./aclnnRingAttentionUpdate_en.md) interface is: in the scenario where the input layout is TND, the data layout of the softmax-related inputs in the original `RingAttentionUpdate` interface is BNS8, while the `RingAttentionUpdateV2` interface supports passing a string parameter `inputSoftmaxLayout` to control whether the data layout of the softmax-related inputs is consistent with the attention layout (the TND layout).**
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

## Prototype

- Each operator is divided into a [two-phase API](../../../docs/en/context/two_phase_api.md). It is necessary to first call the `aclnnRingAttentionUpdateV2GetWorkspaceSize` interface to obtain the required workspace size for computation and the executor that includes the operator's computation process, and then call the `aclnnRingAttentionUpdateV2` interface to perform the computation.

```cpp
aclnnStatus aclnnRingAttentionUpdateV2GetWorkspaceSize(
  const aclTensor *prevAttnOut, 
  const aclTensor *prevSoftmaxMax, 
  const aclTensor *prevSoftmaxSum, 
  const aclTensor *curAttnOut, 
  const aclTensor *curSoftmaxMax, 
  const aclTensor *curSoftmaxSum, 
  const aclTensor *actualSeqQlenOptional, 
  char *inputLayoutOptional, 
  char *inputSoftmaxLayoutOptional, 
  const aclTensor *attnOutOut, 
  const aclTensor *softmaxMaxOut, 
  const aclTensor *softmaxSumOut, 
  uint64_t *workspaceSize, 
  aclOpExecutor **executor)
```

```cpp
aclnnRingAttentionUpdateV2(
  void *workspace, 
  uint64_t workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream stream)
```

## aclnnRingAttentionUpdateV2GetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1603px"><colgroup>
  <col style="width: 238px">
  <col style="width: 123px">
  <col style="width: 232px">
  <col style="width: 402px">
  <col style="width: 193px">
  <col style="width: 120px">
  <col style="width: 149px">
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
      <td>prevAttnOut</td>
      <td>Input</td>
      <td>prev_attn_out in the formula.</td>
      <td>The output of the first FlashAttention, the input shape is consistent with the inputLayoutOptional attribute.<br>When the input data layout inputLayoutOptional is TND, D is limited to a multiple of 64.</td>
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>prevSoftmaxMax</td>
      <td>Input</td>
      <td>The prev_softmax_max in the formula.</td>
      <td>The max result of the first FlashAttention's softmax, with an input shape of (B,N,S,8) or (T,N,8). The last dimension has 8 identical numbers, and they must be positive.<br>B is the batch size, N is the number of heads, S is the sequence length, and T is the time.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>prevSoftmaxSum</td>
      <td>Input</td>
      <td>The prev_softmax_sum in the formula.</td>
      <td>The sum result of the first FlashAttention's softmax, the input shape should be consistent with prevSoftmaxMax, the last dimension has 8 identical numbers, and they must be positive.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>curAttnOut</td>
      <td>Input</td>
      <td>The cur_attn_out in the formula.</td>
      <td>The output of the second FlashAttention, with the same data type and input shape as prevAttnOut.<br>When the input data layout inputLayoutOptional is TND, D is restricted to a multiple of 64.</td>
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>curSoftmaxMax</td>
      <td>Input</td>
      <td>The cur_softmax_max in the formula.</td>
      <td>The max result of the second FlashAttention's softmax, the input shape should be consistent with prevSoftmaxMax, the last dimension has 8 identical numbers, and they need to be positive.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>curSoftmaxSum</td>
      <td>Input</td>
      <td>cur_softmax_sum in the formula</td>
      <td>The sum result of the second FlashAttention's softmax, the input shape should be consistent with prevSoftmaxMax, the last dimension has 8 identical numbers, and they must be positive.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>actualSeqQlenOptional</td>
      <td>Input</td>
      <td>Accumulation of sequence length starting from 0.</td>
      <td>When the data layout inputLayoutOptional is TND, this parameter needs to be passed in. It is an integer aclTensor that increments from 0 to T.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inputLayoutOptional</td>
      <td>Input</td>
      <td>Data layout of prevAttnOut and curAttnOut.</td>
      <td>Currently supports "TND" and "SBH".</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inputSoftmaxLayoutOptional</td>
      <td>Input</td>
      <td>Data layout of prevSoftmaxMax, prevSoftmaxSum, curSoftmaxMax, and curSoftmaxSum.</td>
      <td>Effective when the input data layout inputLayoutOptional is TND, currently supports empty string, "SBH", and "TND".<br>This switch controls whether to perform transpose operations on softmaxMax related inputs.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attnOutOut</td>
      <td>Output</td>
      <td>The attn_out in the formula.</td>
      <td>After two result updates, the data type and output shape remain consistent with prevAttnOut.</td>
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>softmaxMaxOut</td>
      <td>Output</td>
      <td>The softmax_max in the formula is the max of the softmax after two result updates.</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>Consistent with prevSoftmaxMax</td>
      <td>√</td>
    </tr>
    <tr>
      <td>softmaxSumOut</td>
      <td>Output</td>
      <td>The softmax_sum in the formula is the sum of the softmax after two result updates.</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>Consistent with prevSoftmaxMax</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size that the user needs to apply for on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, which includes the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Returns:**

  aclnnStatus: Returns the status code, for details see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.
 
  <table style="undefined;table-layout: fixed; width: 1182px"><colgroup>
  <col style="width: 313px">
  <col style="width: 119px">
  <col style="width: 750px">
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
      <td>The passed prevAttnOut, prevSoftmaxMax, prevSoftmaxSum, curAttnOut, curSoftmaxMax, curSoftmaxSum, attnOutOut, softmaxMaxOut, softmaxSumOut are null pointers.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data types of prevAttnOut, prevSoftmaxMax, prevSoftmaxSum, curAttnOut, curSoftmaxMax, curSoftmaxSum, attnOutOut, softmaxMaxOut, and softmaxSumOut are not within the supported range.</td>
    </tr>
    <tr>
      <td>The shapes of prevAttnOut, prevSoftmaxMax, prevSoftmaxSum, curAttnOut, curSoftmaxMax, curSoftmaxSum, attnOutOut, softmaxMaxOut, and softmaxSumOut are not supported.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="2">561002</td>
      <td>When actualSeqQlenOptional has input, the input data format is not within the supported range.</td>
    </tr>
    <tr>
      <td>When the inputSoftmaxLayoutOptional value is outside the supported range.</td>
    </tr>
  </tbody>
  </table>

## aclnnRingAttentionUpdateV2

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1049px"><colgroup>
  <col style="width: 167px">
  <col style="width: 118px">
  <col style="width: 764px">
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
      <td>The memory address of the workspace allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>The size of the workspace allocated on the Device side, obtained by the first interface aclnnInplaceAddGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, which includes the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specify the stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Returns:**

  aclnnStatus: Returns the status code, for details see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnRingAttentionUpdateV2` defaults to deterministic implementation.
- When `inputLayoutOptional` is TND, the last dimension of `prevAttnOut` must be a multiple of 64.
- When `inputLayoutOptional` is TND, `actualSeqQlenOptional` is required.
- When `inputLayoutOptional` is TND, N must be less than or equal to 256, and D must be less than or equal to 768.
- **When `inputLayoutOptional` is TND, `inputSoftmaxLayoutOptional` takes effect. `inputSoftmaxLayoutOptional` only supports three inputs: empty string, "SBH", "TND"**

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
  // (Fixed writing) Initialize resources.
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
  // Call aclrtMalloc to allocate memory on the device side.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy data from the host side to the device side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of a contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor interface to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Fixed writing) device/stream initialization, refer to the acl API manual.
  // Fill in the deviceId according to your actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the input and output based on the API.
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

  // 3. Call the CANN operator library API. Construct the input and output based on the API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnPrecisionCompare.
  ret = aclnnRingAttentionUpdateV2GetWorkspaceSize(prevAttnOut, prevSoftmaxMax, prevSoftmaxSum, 
                                                 curAttnOut, curSoftmaxMax, curSoftmaxSum, 
                                                 actualSeqQlenOptional, inputLayoutOptional, inputSoftmaxLayoutOptional,
                                                 attnOut, softmaxMax, softmaxSum, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRingAttentionUpdateV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated from the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnPrecisionCompare.
  ret = aclnnRingAttentionUpdateV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRingAttentionUpdateV2 failed. ERROR: %d\n", ret); return ret);
  // 4. (Fixed writing) Synchronize the stream and wait for task completion.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
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

  // 6. Release aclTensor and aclScalar. Make modifications according to the specific API definition.
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

  // 7. Release device resources. Make modifications according to the specific API definition.
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
