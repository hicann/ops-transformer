# aclnnRingAttentionUpdate

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Description

- **API function**: Updates the outputs of two FlashAttention operations based on their respective maximum and sum softmax values.

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

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnRingAttentionUpdateGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnRingAttentionUpdate` is called to perform computation.

- `aclnnStatus aclnnRingAttentionUpdateGetWorkspaceSize(const aclTensor *prevAttnOut, const aclTensor *prevSoftmaxMax, const aclTensor *prevSoftmaxSum, const aclTensor *curAttnOut, const aclTensor *curSoftmaxMax, const aclTensor *curSoftmaxSum, const aclTensor *actualSeqQlenOptional, char *inputLayoutOptional, const aclTensor *attnOutOut, const aclTensor *softmaxMaxOut, const aclTensor *softmaxSumOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnRingAttentionUpdate(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnRingAttentionUpdateGetWorkspaceSize

- **Parameters:**
  - `prevAttnOut` (aclTensor*, compute input): aclTensor on the device, prev_attn_out in the formula, output of the first FlashAttention operation. The data type can be FLOAT16, FLOAT, or BFLOAT16. The input shape must be the same as the `inputLayoutOptional` attribute. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md). The [data format](../../../docs/en/context/data_format.md) can be ND. When `inputLayoutOptional` is TND, D must be a multiple of 64.
  - `prevSoftmaxMax` (aclTensor*, compute input): aclTensor on the device, prev_softmax_max in the formula, Softmax maximum result of the first FlashAttention operation. The data type can be FLOAT. The input shape is (B, N, S, 8) or (T, N, 8). The eight numbers of the last dimension must be identical and positive. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md). The [data format](../../../docs/en/context/data_format.md) can be ND. B indicates the batch size, N indicates the head number, S indicates the sequence length, and T indicates the time.
  - `prevSoftmaxSum` (aclTensor*, compute input): aclTensor on the device, prev_softmax_sum in the formula, Softmax sum result of the first FlashAttention operation. The data type can be FLOAT. The input shape is the same as that of `prevSoftmaxMax`. The eight numbers of the last dimension must be identical and positive. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md). The [data format](../../../docs/en/context/data_format.md) can be ND.
  - `curAttnOut` (aclTensor*, compute input): aclTensor on the device, cur_attn_out in the formula, output of the second FlashAttention operation. The data type can be FLOAT16, FLOAT, or BFLOAT16. The data type and input shape must be the same as those of `prevAttnOut`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md). The [data format](../../../docs/en/context/data_format.md) can be ND. When `inputLayoutOptional` is TND, D must be a multiple of 64.
  - `curSoftmaxMax` (aclTensor*, compute input): aclTensor on the device, cur_softmax_max in the formula, Softmax max result of the second FlashAttention operation. The data type can be FLOAT. The input shape is the same as that of `prevSoftmaxMax`. The eight numbers of the last dimension must be identical and positive. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md). The [data format](../../../docs/en/context/data_format.md) can be ND.
  - `curSoftmaxSum` (aclTensor*, compute input): aclTensor on the device, cur_softmax_sum in the formula, Softmax sum result of the second FlashAttention operation. The data type can be FLOAT. The input shape is the same as that of `prevSoftmaxMax`. The eight numbers of the last dimension must be identical and positive. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md). The [data format](../../../docs/en/context/data_format.md) can be ND.
  - `actualSeqQlenOptional` (aclTensor*, compute input): aclTensor on the device, cumulative sequence lengths starting from 0. The data type can be INT64. This parameter must be provided when `inputLayoutOptional` is TND. It is an integer-type aclTensor whose values increase from 0 to T.
  - `inputLayoutOptional` (char\*, compute input): Host-side char\* constant specifying the data layout of inputs related to attn_out.". Currently, TND and SBH are supported.
  - `attnOutOut` (aclTensor*, compute output): aclTensor on the device, attn_out in the formula, updated output after both results are merged. The data type can be FLOAT16, FLOAT, or BFLOAT16. The data type and output shape must be the same as those of `prevAttnOut`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md). The [data format](../../../docs/en/context/data_format.md) can be ND.
  - `softmaxMaxOut` (aclTensor*, compute output): aclTensor on the device, softmax_max in the formula, updated Softmax maximum after both results are merged. The data type can be FLOAT. The output shape must be the same as those of `prevSoftmaxMax`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md). The [data format](../../../docs/en/context/data_format.md) can be ND.
  - `softmaxSumOut` (aclTensor*, compute output): aclTensor on the device, softmax_sum in the formula, updated Softmax sum after both results are merged. The data type can be FLOAT. The output shape must be the same as those of `prevSoftmaxMax`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md). The [data format](../../../docs/en/context/data_format.md) can be ND.
  - `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor\**, output): operator executor, containing the operator computation process.
  
- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```cpp
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed prevAttnOut, prevSoftmaxMax, prevSoftmaxSum, curAttnOut, curSoftmaxMax, curSoftmaxSum, attnOutOut, softmaxMaxOut, or softmaxSumOut is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of prevAttnOut, prevSoftmaxMax, prevSoftmaxSum, curAttnOut, curSoftmaxMax, curSoftmaxSum, attnOutOut, softmaxMaxOut, or softmaxSumOut is not supported.
  561002 (ACLNN_ERR_INNER_TILING_ERROR): 1. When actualSeqQlenOptional is specified, the input data format is not supported.
                                            2. The shape of prevAttnOut, prevSoftmaxMax, prevSoftmaxSum, curAttnOut, curSoftmaxMax, curSoftmaxSum, attnOutOut, softmaxMaxOut or softmaxSumOut is empty.
                                            3. The shape of prevAttnOut, prevSoftmaxMax, prevSoftmaxSum, curAttnOut, curSoftmaxMax, curSoftmaxSum, attnOutOut, softmaxMaxOut or softmaxSumOut is not supported.
  ```

## aclnnRingAttentionUpdate

- **Parameters:**
  - `workspace` (void*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnRingAttentionUpdateGetWorkspaceSize`.
  - `executor` (aclOpExecutor*, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computing:
  - `aclnnRingAttentionUpdate` defaults to a deterministic implementation.
  - When `inputLayoutOptional` is TND, the last dimension of `prevAttnOut` must be a multiple of 64.
  - When `inputLayoutOptional` is TND, `actualSeqQlenOptional` is required.
  - When `inputLayoutOptional` is TND, N must be less than or equal to 256, and D must be less than or equal to 768.

## Example

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
  int64_t shape_size = 1;
  for (auto i : shape) {
    shape_size *= i;
  }
  return shape_size;
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
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy the data from the host to the memory on the device.
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
  // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as required.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the inputs and outputs based on the API definition.
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

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnRingAttentionUpdate.
  ret = aclnnRingAttentionUpdateGetWorkspaceSize(prevAttnOut, prevSoftmaxMax, prevSoftmaxSum, 
                                                 curAttnOut, curSoftmaxMax, curSoftmaxSum, 
                                                 actualSeqQlenOptional, inputLayoutOptional, 
                                                 attnOut, softmaxMax, softmaxSum, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRingAttentionUpdateGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnRingAttentionUpdate.
  ret = aclnnRingAttentionUpdate(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRingAttentionUpdate failed. ERROR: %d\n", ret); return ret);
  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
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

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
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

  // 7. Release device resources. Modify the configuration based on the API definition.
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
