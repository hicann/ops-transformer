# aclnnGatherPaKvCache

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    √     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function

- Operator function: Fetches discontinuous tokens from `keyCache`/`valueCache` and assembles them into a continuous `key`/`value` sequence based on the `blockId` value in `blockTables` and `seqLen` of `key`/`value` from `seqLens`.
- Computation logic:
  - The first dimension of `keyRef`/`valueRef` depends on the value of `seqLens`.
  - If `isSeqLensCumsum` is `true`, the last value in `seqLens` is the size of the first dimension of `keyRef`/`valueRef`: `keyRef[dim0]` = `seqLens[-1]`.
  - If `isSeqLensCumsum` is `false`, the sum of all values in `seqLens` is the size of the first dimension of `keyRef`/`valueRef`: `keyRef[dim0]` = `sum(seqLens)`.

  Restrictions on `keyRef` and `valueRef`:
  - The size of each token must be less than 148k. For example, for the FP16/BF16 type, `num_heads` × `head_size` (`keyRef`/`valueRef`) is 128 × 576.

- Example:

  ```cpp
    keyCache_shape: [128, 128, 16, 144]
    valueCache_shape: [128, 128, 16, 128]
    blockTables_shape: [16, 12]
    seqLens_shape: [16]
    keyRef_shape: [8931, 16, 144]
    valueRef_shape: [8931, 16, 128]
    seqOffset_shape: [16]
    out1_shape: [8931, 16, 144]  
    out2_shape: [8931, 16, 128]        
  ```

## Function Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnGatherPaKvCacheGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnGatherPaKvCache` is called to perform computation.

- `aclnnStatus aclnnGatherPaKvCacheGetWorkspaceSize(const aclTensor *keyCache, const aclTensor *valueCache, const aclTensor *blockTables, const aclTensor *seqLens, aclTensor *keyRef, aclTensor *valueRef, const aclTensor *seqOffsetOptional, char* cacheMode, bool isSeqLensCumsum, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnGatherPaKvCache(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnGatherPaKvCacheGetWorkspaceSize

- **Parameters:**

  - `keyCache` (`aclTensor*`, an input parameter): Device-side `aclTensor`, indicating the `key` vector cache stored at the current layer. When `cacheMode` is `Norm`, the shape is `[num_blocks, block_size, num_heads, head_size_k]`. When `cacheMode` is `PA_NZ`, the shape is `[num_blocks, num_heads × head_size_k // elenum_aligned, block_size, elenum_aligned]`. (`elenum_aligned` is `32` in the b8 scenario, `16` in the b16 scenario, and `8` in the b32 scenario. In the b8 scenario, the bit width of each data element is 8 bits, for example, INT8. In the b16 scenario, the bit width of each data element is 16 bits, for example, INT16. In the b32 scenario, the bit width of each data element is 32 bits, for example, INT32.) [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) and empty tensors are not supported.
    - <term>Atlas A3 training products/Atlas A3 inference products</term>, <term>Atlas A2 training products/Atlas A2 inference products</term>, <term>Atlas 200I/500 A2 inference products</term>, <term>Atlas inference products</term>, and <term>Atlas training products</term>: The data type can be INT8, FLOAT16, or BFLOAT16. The [data format](../../../docs/en/context/data_format.md) can be ND.
  - `valueCache` (`aclTensor*`, an input parameter): Device-side `aclTensor`, indicating the `value` vector cache stored at the current layer. When `cacheMode` is `Norm`, the shape is `[num_blocks, block_size, num_heads, head_size_v]`. When `cacheMode` is `PA_NZ`, the shape is `[num_blocks, num_heads × head_size_v // elenum_aligned, block_size, elenum_aligned]`. (`elenum_aligned` is `32` in the b8 scenario, `16` in the b16 scenario, and `8` in the b32 scenario.) [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) and empty tensors are not supported.
    - <term>Atlas A3 training products/Atlas A3 inference products</term>, <term>Atlas A2 training products/Atlas A2 inference products</term>, <term>Atlas 200I/500 A2 inference products</term>, <term>Atlas inference products</term>, and <term>Atlas training products</term>: The data type can be INT8, FLOAT16, or BFLOAT16. The [data format](../../../docs/en/context/data_format.md) can be ND.
  - `blockTables` (`aclTensor*`, an input parameter): Device-side `aclTensor`, indicating the physical block index corresponding to each sequence. The shape is `[batch, block_indices]`, and the element value range is `[0, num_blocks)`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) and empty tensors are not supported.
    - <term>Atlas A3 training products/Atlas A3 inference products</term>, <term>Atlas A2 training products/Atlas A2 inference products</term>, <term>Atlas 200I/500 A2 inference products</term>, <term>Atlas inference products</term>, and <term>Atlas training products</term>: The data type can be int32_t. The [data format](../../../docs/en/context/data_format.md) can be ND.
  - `seqLens` (`aclTensor*`, an input parameter): Device-side `aclTensor`, indicating the sequence length corresponding to each batch. The [data format](../../../docs/en/context/data_format.md) can be ND, and the shape is `[batch]` or `[batch + 1]`. When `isSeqLensCumsum` is `false`, the shape is `[batch]`. When `isSeqLensCumsum` is `true`, the shape is `[batch + 1]`. The element value range is `[0, num_blocks)`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) and empty tensors are not supported.
    - <term>Atlas A3 training products/Atlas A3 inference products</term>, <term>Atlas A2 training products/Atlas A2 inference products</term>, <term>Atlas 200I/500 A2 inference products</term>, <term>Atlas inference products</term>, and <term>Atlas training products</term>: The data type can be INT32. The [data format](../../../docs/en/context/data_format.md) can be ND.
  - `keyRef` (`aclTensor*`, an input/output parameter): Device-side `aclTensor`, indicating the `key` vector. The [data format](../../../docs/en/context/data_format.md) can be ND. When `cacheMode` is `Norm`, the shape is `[num_tokens, num_heads, head_size_k]`. When `cacheMode` is `PA_NZ`, the shape is `[num_tokens, num_heads × head_size_k]`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) and empty tensors are not supported.
    - <term>Atlas A3 training products/Atlas A3 inference products</term>, <term>Atlas A2 training products/Atlas A2 inference products</term>, <term>Atlas 200I/500 A2 inference products</term>, <term>Atlas inference products</term>, and <term>Atlas training products</term>: The data type can be INT8, FLOAT16, or BFLOAT16. The [data format](../../../docs/en/context/data_format.md) can be ND.
  - `valueRef` (`aclTensor*`, an input/output parameter): Device-side `aclTensor`, indicating the `value` vector. The [data format](../../../docs/en/context/data_format.md) can be ND. When `cacheMode` is `Norm`, the shape is `[num_tokens, num_heads, head_size_v]`. When `cacheMode` is `PA_NZ`, the shape is `[num_tokens, num_heads × head_size_v]`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) and empty tensors are not supported.
    - <term>Atlas A3 training products/Atlas A3 inference products</term>, <term>Atlas A2 training products/Atlas A2 inference products</term>, <term>Atlas 200I/500 A2 inference products</term>, <term>Atlas inference products</term>, and <term>Atlas training products</term>: The data type can be INT8, FLOAT16, or BFLOAT16. The [data format](../../../docs/en/context/data_format.md) can be ND.
  - `seqOffsetOptional` (`aclTensor*`, an optional input parameter): Device-side `aclTensor`. The [data format](../../../docs/en/context/data_format.md) can be ND, and the shape is `[batch]`. If this parameter is passed, there is an initial offset when obtaining `blockId` from `blockTables` (the offset is `seqOffsetOptional[i]`/`block_size`, where `i` indicates a batch). If this parameter is not passed, no offset is required. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) and empty tensors are not supported.
    - <term>Atlas A3 training products/Atlas A3 inference products</term>, <term>Atlas A2 training products/Atlas A2 inference products</term>, <term>Atlas 200I/500 A2 inference products</term>, <term>Atlas inference products</term>, and <term>Atlas training products</term>: The data type can be INT32. The [data format](../../../docs/en/context/data_format.md) can be ND.
  - `cacheMode` (`char*`, an input parameter): The options include `Norm` and `PA_NZ`, indicating the ND and NZ data formats, respectively. This parameter indicates the shape input mode of `keyCache`/`valueCache`/`keyRef`/`valueRef`.

  - `isSeqLensCumsum` (`bool`, an input parameter): Whether `seqLens` is a cumulative sum. If `isSeqLensCumsum` is `false`, `seqLens` is a non-cumulative sum. For example, `seqLens` is `[1, 3, 5, 3, 7]`. If `isSeqLensCumsum` is `true`, `seqLens` is a cumulative sum. For example, `seqLens` is `[0, 1, 4, 9, 12, 19]`, and the 0th element must be `0`. The cumulative sum of `seqlens[i + 1] – seqlens[i]` is equal to the non-cumulative `seqlens[i]`.

  - `workspaceSize` (`uint64_t*`, an output parameter): Size of the workspace to be allocated on the device.

  - `executor` (`aclOpExecutor**`, an output parameter): Operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```cpp
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The input is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The input data type is not supported.
                                      2. The input number of dimensions does not match.
                                      3. The input data types are inconsistent.
  ```

## aclnnGatherPaKvCache

- **Parameters:**

  - `workspace` (`void*`, an input parameter): Address of the workspace to be allocated on the device.

  - `workspaceSize` (`uint64_t`, an input parameter): Size of the workspace to be allocated on the device, which is obtained by the first-phase API `aclnnGatherPaKvCacheGetWorkspaceSize`.

  - `executor` (`aclOpExecutor*`, an input parameter): Operator executor, containing the operator computation process.

  - `stream` (`aclrtStream`, an input parameter): Stream for executing the task.

- **Returns:**

  `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnGatherPaKvCache` defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_gather_pa_kv_cache.h"

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
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy the data from the host to the device.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of consecutive tensors.
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
  // 1. (Fixed writing) Initialize the device and stream. For details, see the list of external AscendCL APIs.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> keyCacheShape = {2, 2, 32, 2};
  std::vector<int64_t> valueCacheShape = {2, 2, 32, 4};
  std::vector<int64_t> blockTablesShape = {4,6};
  std::vector<int64_t> seqLensShape = {4};
  std::vector<int64_t> keyShape = {12, 32, 2};
  std::vector<int64_t> valueShape = {12, 32, 4};
  std::vector<int64_t> seqOffsetShape = {4};


  void* keyCacheDeviceAddr = nullptr;
  void* valueCacheDeviceAddr = nullptr;
  void* blockTablesDeviceAddr = nullptr;
  void* seqLensDeviceAddr = nullptr;
  void* keyDeviceAddr = nullptr;
  void* valueDeviceAddr = nullptr;
  void* seqOffsetAddr = nullptr;

  aclTensor* keyCache= nullptr;
  aclTensor* valueCache = nullptr;
  aclTensor* blockTables= nullptr;
  aclTensor* seqLens= nullptr;
  aclTensor* key = nullptr;
  aclTensor* value= nullptr;
  aclTensor* seqOffset= nullptr;

  std::vector<uint16_t> keyCacheHostData(256, 1);
  std::vector<uint16_t> valueCacheHostData(512, 1);
  std::vector<int32_t> blockTablesHostData(24, 1);
  std::vector<int32_t> seqLensHostData(4, 3);
  std::vector<uint16_t> keyHostData(768, 0);
  std::vector<uint16_t> valueHostData(1536, 0);
  std::vector<int32_t> seqOffsetHostData(4, 2);

  char cacheMode[] = "Norm";
  const bool isSeqLensCumsum = false;
  // Create a gradOut aclTensor.
  ret = CreateAclTensor(keyCacheHostData, keyCacheShape, &keyCacheDeviceAddr, aclDataType::ACL_FLOAT16, &keyCache);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(valueCacheHostData, valueCacheShape, &valueCacheDeviceAddr, aclDataType::ACL_FLOAT16, &valueCache);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(blockTablesHostData, blockTablesShape, &blockTablesDeviceAddr, aclDataType::ACL_INT32, &blockTables);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(seqLensHostData, seqLensShape, &seqLensDeviceAddr, aclDataType::ACL_INT32, &seqLens);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &key);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &value);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(seqOffsetHostData, seqOffsetShape, &seqOffsetAddr, aclDataType::ACL_INT32, &seqOffset);
  CHECK_RET(ret == ACL_SUCCESS, return ret);


  // 3. Call the CANN operator library API. Modify the API as required.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnGatherPaKvCache.
  ret = aclnnGatherPaKvCacheGetWorkspaceSize(keyCache, valueCache, blockTables, seqLens, key , value, seqOffset,
  cacheMode, isSeqLensCumsum, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGatherPaKvCacheGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnGatherPaKvCache.
  ret = aclnnGatherPaKvCache(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGatherPaKvCache failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = 256;
  std::vector<uint16_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), keyDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }

  // 6. Destroy aclTensor and aclIntArray. Modify the code based on the API definition.
  aclDestroyTensor(keyCache);
  aclDestroyTensor(valueCache);
  aclDestroyTensor(blockTables);
  aclDestroyTensor(seqLens);
  aclDestroyTensor(key);
  aclDestroyTensor(value);
  aclDestroyTensor(seqOffset);

  // 7. Free device resources. Modify the configuration based on the API definition.
  aclrtFree(keyCacheDeviceAddr);
  aclrtFree(valueCacheDeviceAddr);
  aclrtFree(blockTablesDeviceAddr );
  aclrtFree(seqLensDeviceAddr );
  aclrtFree(keyDeviceAddr);
  aclrtFree(valueDeviceAddr);
  aclrtFree(seqOffsetAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
