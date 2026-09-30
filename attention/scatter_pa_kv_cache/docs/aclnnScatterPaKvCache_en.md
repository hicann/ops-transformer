# aclnnScatterPaKvCache

## Supported Products

| Product                                                      | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>     |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function

- Updates the `key` and `value` at the specified position in the KvCache.

- Input and output support the following scenarios:
  - Scenario 1:
   
    ```text
    key:[batch * seq_len, num_head, k_head_size]
    value:[batch, num_head, v_head_size]
    keyCache:[num_blocks, num_head * k_head_size // last_dim_k, block_size, last_dim_k]
    valueCache:[num_blocks, num_head * v_head_size // last_dim_k, block_size, last_dim_k]
    slotMapping:[batch * seq_len]
    cacheMode:"PA_NZ"
    scatter_mode:"None"
    ```  
    
  - Scenario 2:
    
    ```text
    key:[batch * seq_len, num_head, k_head_size]
    value:[batch * seq_len, num_head, v_head_size]
    keyCache:[num_blocks, block_size, num_head, k_head_size]
    valueCache:[num_blocks, block_size, num_head, v_head_size]
    slotMapping:[batch * seq_len]
    cacheMode:"Norm"
    scatter_mode:"None"/"Nct"
    
    ```text
    The `k_head_size` and `v_head_size` can be different or the same.

  - Scenario 3:

    ```text
    key:[batch, seq_len, num_head, k_head_size]
    value:[batch, seq_len, num_head, v_head_size]
    keyCache:[num_blocks, block_size, 1, k_head_size]
    valueCache:[num_blocks, block_size, 1, k_head_size]
    slotMapping:[batch, num_head]
    compressLensOptional:[batch, num_head]
    seqLensOptional:[batch]
    compressSeqOffsetOptional:[batch * num_head]
    cacheMode:"Norm"
    ```

  - Scenario 4:

    ```text
    key:[num_tokens, num_head, k_head_size]
    value:[num_tokens, num_head, v_head_size]
    keyCache:[num_blocks, block_size, 1, k_head_size]
    valueCache:[num_blocks, block_size, 1, k_head_size]
    slotMapping:[batch * num_head]
    compressLensOptional:[batch * num_head]
    seqLensOptional:[batch]
    cacheMode:"Norm"
    scatter_mode:"Alibi"
    ```

  - Scenario 5:

    ```text
    key:[num_tokens, num_head, k_head_size]
    value:[num_tokens, num_head, v_head_size]
    keyCache:[num_blocks, block_size, 1, k_head_size]
    valueCache:[num_blocks, block_size, 1, k_head_size]
    slotMapping:[batch * num_head]
    compressLensOptional:[batch * num_head]
    seqLensOptional:[batch]
    compressSeqOffsetOptional:[batch * num_head]
    cacheMode:"Norm"
    scatter_mode:"Rope"/"Omni"
    ```

    - Scenario 6:

    ```text
    key:[batch * seq_len, num_head, k_head_size]
    value:[]
    keyCache:[num_blocks, block_size, num_head, k_head_size]
    valueCache:[]
    slotMapping:[batch * seq_len]
    cacheMode:"Norm"
    scatter_mode:"None"/"Nct"
    ```

  - The above scenarios are distinguished based on the constructed parameters. If the first type of parameter construction is met, it follows scenario 1; if the second type is met, it follows scenario 2; if the third type is met, it follows scenario 3; if the fourth type is met, it follows scenario 4; if the fifth type is met, it follows scenario 5; and if the sixth type is met, it follows scenario 6. Scenarios 1, 2, and 6 do not have the three optional parameters: `compressLensOptional`, `seqLensOptional`, and `compressSeqOffsetOptional`. Scenario 4 does not have the optional parameter `compressSeqOffsetOptional`.
  - <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term> only support scenarios 1, 2, 4, 5, and 6.
  
## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First call `aclnnScatterPaKvCacheGetWorkspaceSize` to obtain the required workspace size for computation and the executor that includes the operator's computation process. Then call `aclnnScatterPaKvCache` to perform the computation.

* `aclnnStatus aclnnScatterPaKvCacheGetWorkspaceSize(const aclTensor *key, aclTensor *keyCacheRef, const aclTensor *slotMapping, const aclTensor *value, aclTensor *valueCacheRef, const aclTensor *compressLensOptional, const aclTensor *compressSeqOffsetOptional, const aclTensor *seqLensOptional, char *cacheModeOptional, char *scatterModeOptional, const aclIntArray *stridesOptional, const aclIntArray *offsetsOptional, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnScatterPaKvCache(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnScatterPaKvCacheGetWorkspaceSize

- **Parameters:**

  * `key` (aclTensor*, computation input): Device-side aclTensor, supports 3D or 4D. The key value to be updated and the key of multiple tokens in the current step. Data types supported include FLOAT16, FLOAT, BFLOAT16, INT8, UINT8, INT16, UINT16, INT32, UINT32, HIFLOAT8, FLOAT8_E5M2, and FLOAT8_E4M3FN, [data format](../../../docs/en/context/data_format.md) supports ND.
      * <term>Atlas A3 Training Series Products/Atlas A3 Inference Series Products</term>, <term>Atlas A2 Training Series Products/Atlas A2 Inference Series Products</term>: The data types supported are only FLOAT16, BFLOAT16, and INT8.
  * `keyCacheRef` (aclTensor*, computation input/output): Device-side aclTensor, only supports 4 dimensions, the key cache that needs to be updated, and the key cache of the current layer. The data type and format are consistent with the key.
  * `slotMapping` (aclTensor*, computation input): Device-side aclTensor, the storage offset of each token key or value in the cache. Supported data types include INT32, INT64, and the [data format](../../../docs/en/context/data_format.md) supports ND.
  * `value` (aclTensor*, computation input): Device-side aclTensor, supports 0D, 3D, or 4D. In non-0D cases, the shape is consistent with the key. The value to be updated, representing the value of multiple tokens in the current step, with data type and format consistent with the key.
  * `valueCacheRef` (aclTensor*, computation input/output): Device-side aclTensor, supports 0-dimensional or 4-dimensional. In non-0-dimensional cases, the shape is consistent with that of `keyCacheRef`. The value cache that needs to be updated and the value cache of the current layer. The data type and format are consistent with value.
  * `compressLensOptional` (aclTensor*, optional computation input): Device-side aclTensor, compression amount. Data type is consistent with `slotMapping`, and [data format](../../../docs/en/context/data_format.md) supports ND.
  * `compressSeqOffsetOptional` (aclTensor*, optional computation input): Device-side aclTensor, the compression starting point for each batch and each head. The data type is consistent with `slotMapping`, and the [data format](../../../docs/en/context/data_format.md) supports ND.
  * `seqLensOptional` (aclTensor*, optional input): Device-side aclTensor, the actual seqLens for each batch. The data type is consistent with `slotMapping`, and the [data format](../../../docs/en/context/data_format.md) supports ND.
  * `cacheMode` (char*, computation input): The `char*` on the host side represents the memory layout format of `keyCacheRef` and `valueCacheRef`.
      * <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>: When passing a null pointer or `Norm`, only the ND memory layout format is supported. When passing `PA_NZ`, only the `FRACTAL_NZ` memory layout format is supported.
  * `scatterMode` (char*, computation input): The `char*` on the host side represents the state of the updated key and value.
      * <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>: When passing a null pointer or `None`, it indicates that the updated key and value are in a non-compressed and continuous state. When passing `Alibi`, it indicates that the updated key and value are in a compressed state based on the Alibi structure. When passing `Rope`, it indicates that the updated key and value are in a compressed state based on the Rope structure. When passing `Omni`, it indicates that the updated key and value are in a compressed state based on the Omni structure. When passing `Nct`, it indicates that the updated key and value are in a non-compressed but non-continuous state.
  * `strides` (aclIntArray *, computation input): The strides of key and value in non-contiguous states, with an array length of 2. The values should be greater than 0.
      * <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>: Only effective when `scatterMode` is `Nct`, representing `strideK` and `strideV` respectively.
  * `offsets` (aclIntArray *, computation input): The offsets of key and value in a non-contiguous state, with an array length of 2. The values should be greater than 0.
      * <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>: Only effective when `scatterMode` is `Nct`, representing `offsetK` and `offsetV` respectively.
  * `workspaceSize` (uint64_t*, computation output): Returns the size of the workspace that the user needs to allocate on the device side.
  * `executor` (aclOpExecutor**, computation output): Returns the op executor, which includes the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown.
  Return 161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed key, keyCacheRef, slotMapping, value, valueCacheRef are null pointers.
  Return 161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of the parameter key or value is not within the supported range.
                                        2. The data types of key, keyCacheRef, value, and valueCacheRef are inconsistent.
                                        3. The data types of slotMapping, compressLensOptional, compressSeqOffsetOptional, and seqLensOptional are inconsistent.
  Return 561002 (ACLNN_ERR_PARAM_INVALID): 1. The dimension of the key is not equal to 3 or 4, and the dimension of the value is not equal to 0, 3, or 4.
  ```

## aclnnScatterPaKvCache

- **Parameters:**

  * `workspace` (void*, input): The memory address of the workspace allocated on the device side.
  * `workspaceSize` (uint64_t, input): The size of the workspace allocated on the device side, obtained by the first-phase API of `aclnnScatterPaKvCacheGetWorkspaceSize`.
  * `executor` (aclOpExecutor*, input): The op executor, which includes the operator computation process.
  * `stream` (aclrtStream, input): Specifies the stream for task execution.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic Computation:
    - `aclnnScatterPaKvCache` defaults to deterministic implementation.
- In addition to key and value, input parameters do not support non-continuity;
- The data types of key, value, keyCacheRef, and valueCacheRef must be consistent;
- The data types of slotMapping, compressLensOptional, compressSeqOffsetOptional, and seqLensOptional must be consistent;
- The value range of slotMapping is [0, num_blocks * block_size - 1], and the element values within slotMapping are guaranteed to be unique; correctness is not guaranteed in case of duplication.
- When both key and value are 3-dimensional, the first two dimensions of their shapes must be the same;
- When both key and value are 4-dimensional, the first three dimensions of the key and value shapes must be the same, and the third dimension of keyCacheRef and valueCacheRef must be 1;
- When the key and value are 4-dimensional, `compressLensOptional` and `seqLensOptional` are required parameters; when the key and value are 3-dimensional, `compressLensOptional`, `compressSeqOffsetOptional`, and `seqLensOptional` are optional parameters;
- When both key and value are 4-dimensional, `slotMapping` is 2-dimensional, and the first dimension of `slotMapping` equals the first dimension of key as batch, and the second dimension of `slotMapping` equals the third dimension of key as `num_head` (corresponding to scenario three);
- When both key and value are 4-dimensional, seqLensOptional is one-dimensional, and the value of seqLensOptional equals the first dimension of key as batch (corresponding to scenario three);
- When both key and value are 3-dimensional and seqLensOptional exists, the sum of all values in seqLensOptional equals the first dimension of key as num_blocks (corresponding to scenarios four and five);
- Each element value in `seqLensOptional` and `compressLensOptional` must satisfy the formula: `reduceSum(seqLensOptional[i] - compressLensOptional[i]) <= num_blocks * block_size` (corresponding to scenarios three, four, and five).

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

- <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>:

   ```c++
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_scatter_pa_kv_cache.h"

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
  // Fixed writing) Initialize resources.
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
  }

  template <typename T>
  int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType, aclTensor** tensor, aclFormat format) {
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
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
  }

  int main() {
    // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
    // Set the deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output based on the API.
    std::vector<int64_t> keyShape = {2, 2, 32};
    std::vector<int64_t> keyCacheShape = {1, 4, 32, 16};
    std::vector<int64_t> slotMappingShape = {2};
    std::vector<int64_t> valueShape = {2, 2, 32};
    std::vector<int64_t> valueCacheShape = {1, 4, 32, 16};

    void* keyDeviceAddr = nullptr;
    void* valueDeviceAddr = nullptr;
    void* slotMappingDeviceAddr = nullptr;
    void* keyCacheDeviceAddr = nullptr;
    void* valueCacheDeviceAddr = nullptr;
    void* compressLensDeviceAddr = nullptr;
    void* compressSeqOffsetDeviceAddr = nullptr;
    void* seqLensDeviceAddr = nullptr;

    aclTensor* key = nullptr;
    aclTensor* value = nullptr;
    aclTensor* slotMapping = nullptr;
    aclTensor* keyCache = nullptr;
    aclTensor* valueCache = nullptr;
    aclTensor* compressLens = nullptr;
    aclTensor* compressSeqOffset = nullptr;
    aclTensor* seqLens = nullptr;
    char * cacheMode = "PA_NZ";
    char * scatterMode = "None";

    std::vector<int16_t> hostKey(128, 1);
    std::vector<int16_t> hostValue(128, 1);
    std::vector<int32_t> hostSlotMapping(2, 1);
    std::vector<int16_t> hostKeyCacheRef(2048, 1);
    std::vector<int16_t> hostValueCacheRef(2048, 1);
    std::vector<int64_t> hostStrides(2, 1);
    std::vector<int64_t> hostOffsets(2, 0);

    // Create key aclTensor.
    ret = CreateAclTensor(hostKey, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &key, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create value aclTensor.
    ret = CreateAclTensor(hostValue, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &value, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create slotMapping aclTensor.
    ret = CreateAclTensor(hostSlotMapping, slotMappingShape, &slotMappingDeviceAddr, aclDataType::ACL_INT32, &slotMapping, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create keyCache aclTensor.
    ret = CreateAclTensor(hostKeyCacheRef, keyCacheShape, &keyCacheDeviceAddr, aclDataType::ACL_FLOAT16, &keyCache, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create valueCache aclTensor.
    ret = CreateAclTensor(hostValueCacheRef, valueCacheShape, &valueCacheDeviceAddr, aclDataType::ACL_FLOAT16, &valueCache, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    aclIntArray *strides = aclCreateIntArray(hostStrides.data(), 2);
    CHECK_RET(strides != nullptr, return ACL_ERROR_INTERNAL_ERROR);
    aclIntArray *offsets = aclCreateIntArray(hostOffsets.data(), 2);
    CHECK_RET(offsets != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnScatterPaKvCache
    ret = aclnnScatterPaKvCacheGetWorkspaceSize(key, keyCache, slotMapping, value, valueCache, compressLens, compressSeqOffset, seqLens, cacheMode, scatterMode, strides, offsets, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScatterPaKvCacheGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnScatterPaKvCache.
    ret = aclnnScatterPaKvCache(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScatterPaKvCache failed. ERROR: %d\n", ret); return ret);

    // 4. (Fixed writing) Synchronize the stream and wait for task completion.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
    auto size = GetShapeSize(keyShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), keyDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
    aclDestroyTensor(key);
    aclDestroyTensor(value);
    aclDestroyTensor(slotMapping);
    aclDestroyTensor(keyCache);
    aclDestroyTensor(valueCache);
    aclDestroyTensor(compressLens);
    aclDestroyTensor(compressSeqOffset);
    aclDestroyTensor(seqLens);
    aclDestroyIntArray(strides);
    aclDestroyIntArray(offsets);
    // 7. Release device resources. Set the parameters based on the API definition.
    aclrtFree(keyDeviceAddr);
    aclrtFree(valueDeviceAddr);
    aclrtFree(slotMappingDeviceAddr);
    aclrtFree(keyCacheDeviceAddr);
    aclrtFree(valueCacheDeviceAddr);
     if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
  }
  ```
