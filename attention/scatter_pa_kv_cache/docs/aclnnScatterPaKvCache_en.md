# aclnnScatterPaKvCache

[📄 View Source Code](https://gitcode.com/cann/ops-transformer/tree/master/attention/scatter_pa_kv_cache)

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

- API function: Updates the `key` and `value` at a specified position in KvCache.

- The input and output support the following scenarios:
  - Scenario 1:

    ```text
    key:[batch * seq_len, num_head, k_head_size]
    value:[batch * seq_len, num_head, v_head_size]
    keyCache:[num_blocks, num_head * k_head_size // last_dim_k, block_size, last_dim_k]/[num_blocks, num_head, k_head_size // last_dim_k, block_size, last_dim_k]
    valueCache:[num_blocks, num_head * v_head_size // last_dim_v, block_size, last_dim_v]/[num_blocks, num_head, v_head_size // last_dim_v, block_size, last_dim_v]
    slotMapping:[batch * seq_len]
    cacheMode:"PA_NZ"
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
    ```

    `k_head_size` and `v_head_size` can be the same or different.

  - Scenario 3:

    ```text
    key:[batch, seq_len, num_head, k_head_size]
    value:[batch, seq_len, num_head, v_head_size]
    keyCache:[num_blocks, block_size, 1, k_head_size]
    valueCache:[num_blocks, block_size, 1, v_head_size]
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
    valueCache:[num_blocks, block_size, 1, v_head_size]
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
    valueCache:[num_blocks, block_size, 1, v_head_size]
    slotMapping:[batch * num_head]
    compressLensOptional:[batch * num_head]
    seqLensOptional:[batch]
    compressSeqOffsetOptional:[batch * num_head]
    cacheMode:"Norm"
    scatter_mode:"Rope"/"Omni"
    ```

    - Scenario 6

    ```text
    key:[batch * seq_len, num_head, k_head_size]
    value:[]
    keyCache:[num_blocks, block_size, num_head, k_head_size]
    valueCache:[]
    slotMapping:[batch * seq_len]
    cacheMode:"Norm"
    scatter_mode:"None"/"Nct"
    ```

- The preceding scenarios are distinguished based on the constructed parameters. If the first type of input parameter is constructed, scenario 1 is used. If the second type of input parameter is constructed, scenario 2 is used. If the third type of input parameter is constructed, scenario 3 is used. If the fourth type of input parameter is constructed, scenario 4 is used. If the fifth type of input parameter is constructed, scenario 5 is used. If the sixth type of input parameter is constructed, scenario 6 is used. In scenarios 1, 2, and 6, the optional parameters compressLensOptional, seqLensOptional, and compressSeqOffsetOptional are not available. In scenario 4, the optional parameter compressSeqOffsetOptional is not available.
- For <term>Atlas A3 training products/Atlas A3 inference products</term>, <term>Atlas A2 training products/Atlas A2 inference products</term>: Currently, the `aicpu` and `aiv` modes are supported.

## Function Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, aclnnScatterPaKvCacheGetWorkspaceSize is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, aclnnScatterPaKvCache is called to perform computation.

```Cpp
aclnnStatus aclnnScatterPaKvCacheGetWorkspaceSize(
  const aclTensor   *key, 
  aclTensor         *keyCacheRef, 
  const aclTensor   *slotMapping, 
  const aclTensor   *value, 
  aclTensor         *valueCacheRef, 
  const aclTensor   *compressLensOptional, 
  const aclTensor   *compressSeqOffsetOptional, 
  const aclTensor   *seqLensOptional, 
  char              *cacheModeOptional, 
  char              *scatterModeOptional, 
  const aclIntArray *stridesOptional, 
  const aclIntArray *offsetsOptional, 
  uint64_t          *workspaceSize, 
  aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnScatterPaKvCache(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```

## aclnnScatterPaKvCacheGetWorkspaceSize

- **Parameter Description**

  <table class="tg" style="undefined;table-layout: fixed; width: 1548px"><colgroup>
  <col style="width: 265px">
  <col style="width: 86px">
  <col style="width: 269px">
  <col style="width: 462px">
  <col style="width: 172px">
  <col style="width: 111px">
  <col style="width: 87px">
  <col style="width: 96px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">Parameter Name</th>
      <th class="tg-0pky">Input/Output</th>
      <th class="tg-0pky">Description</th>
      <th class="tg-0pky">Usage Description</th>
      <th class="tg-0pky">Data Type</th>
      <th class="tg-0pky">Data Format</th>
      <th class="tg-0pky">Dimension (shape)</th>
      <th class="tg-0pky">Non-continuous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">key (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Key value to be updated, which is the key of multiple tokens in the current step.</td>
      <td class="tg-0pky"></td>
      <td class="tg-0pky">FLOAT16, FLOAT, BFLOAT16, INT8, UINT8, INT16, UINT16, INT32, UINT32, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">3-4</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">keyCacheRef (aclTensor*)</td>
      <td class="tg-0pky">Input/Output</td>
      <td class="tg-0pky">Key cache to be updated, which is the key cache of the current layer.</td>
      <td class="tg-0pky">The 4D or 5D format is supported. When a null pointer or "Norm" is passed, only the ND memory format is supported. When "PA_NZ" is passed, only the FRACTAL_NZ memory format is supported.</td>
      <td class="tg-0pky">Consistent with the key.</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">4-5</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">slotMapping (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Storage offset of each token key or value in the cache.</td>
      <td class="tg-0pky"></td>
      <td class="tg-0pky">INT32, INT64</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">value (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Value to be updated, which is the value of multiple tokens in the current step.</td>
      <td class="tg-0pky">The shape can be 0D, 3D, or 4D. For non-0D shapes, the shape is the same as the key.</td>
      <td class="tg-0pky">Consistent with the key.</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">0, 3, 4</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">valueCacheRef (aclTensor*)</td>
      <td class="tg-0pky">Input/Output</td>
      <td class="tg-0pky">Value cache to be updated, which is the value cache of the current layer.</td>
      <td class="tg-0pky">The shape can be 0D, 4D, or 5D. For non-0D shapes, the shape is the same as the keyCacheRef. When a null pointer or "Norm" is passed, only the ND memory format is supported. When "PA_NZ" is passed, only the FRACTAL_NZ memory format is supported.</td>
      <td class="tg-0pky">Consistent with the key.</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">0, 4, 5</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">compressLensOptional (aclTensor*)</td>
      <td class="tg-0pky">Optional</td>
      <td class="tg-0pky">Compression amount.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">Same as slotMapping.</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">compressSeqOffsetOptional (aclTensor*)</td>
      <td class="tg-0pky">Optional</td>
      <td class="tg-0pky">Compression start point of each head in each batch.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">Same as slotMapping.</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">seqLensOptional (aclTensor*)</td>
      <td class="tg-0pky">Optional</td>
      <td class="tg-0pky">Actual seqLens of each batch.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">Same as slotMapping.</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">cacheModeOptional (char*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Indicates the memory layout format of keyCacheRef and valueCacheRef.</td>
      <td class="tg-0pky">If a null pointer or "Norm" is passed, only the ND memory layout format is supported. If "PA_NZ" is passed, only the FRACTAL_NZ memory layout format is supported.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">scatterModeOptional (char*)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Indicates the status of the updated key and value.</td>
      <td class="tg-0lax">If a null pointer or "None" is passed, the updated key and value are in the non-compressed and contiguous state.<br>If "Alibi" is passed, the updated key and value are in the compressed state based on the Alibi structure.<br>If "Rope" is passed, the updated key and value are in the compressed state based on the Rope structure.<br>If "Omni" is passed, the updated key and value are in the compressed state based on the Omni structure.<br>If "Nct" is passed, the updated key and value are in the non-compressed and non-contiguous state.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">stridesOptional (aclIntArray*)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Step of the key and value in the discontinuous state.</td>
      <td class="tg-0lax">The array length is 2. Ensure that the value of **Version** is positive. This parameter is valid only when scatterModeOptional is set to Nct. It indicates strideK and strideV respectively.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">offsetsOptional (aclIntArray*)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Offset of the key and value in the discontinuous state.</td>
      <td class="tg-0lax">The array length is 2. Ensure that the value of **Version** is positive. This parameter is valid only when scatterModeOptional is set to Nct. It indicates offsetK and offsetV respectively.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">workspaceSize (uint64_t*)</td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">Returns the size of the workspace to be allocated on the device.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">executor (aclOpExecutor**)</td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">Returns the operator executor, including the operator execution process.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
  </tbody></table>

  - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>:

    - The input key, keyCacheRef, value, and valueCacheRef do not support the FLOAT, UINT8, INT16, UINT16, INT32, UINT32, HIFLOAT8, FLOAT8_E5M2 and FLOAT8_E4M3FN data types.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.

  <table style="undefined;table-layout: fixed; width: 1152px"><colgroup>
  <col style="width: 302px">
  <col style="width: 119px">
  <col style="width: 731px">
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
      <td>The input key, keyCacheRef, slotMapping, value, and valueCacheRef are null pointers.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type of key or value is not supported.</td>
    </tr>
    <tr>
      <td>The data types of key, keyCacheRef, value, and valueCacheRef are inconsistent.</td>
    </tr>
    <tr>
      <td>The data types of slotMapping, compressLensOptional, compressSeqOffsetOptional, and seqLensOptional are inconsistent.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>561002</td>
      <td>The key is not 3D or 4D, and the value is not 0D, 3D, or 4D.</td>
    </tr>
  </tbody>
  </table>

## aclnnScatterPaKvCache

- **Parameter Description**

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
      <td>Size of the workspace to be allocated on the device, which is obtained by the first-phase API aclnnScatterPaKvCacheGetWorkspaceSize.</td>
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

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
    - `aclnnScatterPaKvCache` defaults to a deterministic implementation.
    - The data types of key, value, keyCacheRef, and valueCacheRef must be the same.
    - The data types of slotMapping, compressLensOptional, compressSeqOffsetOptional, and seqLensOptional must be the same.
    - The value range of slotMapping is [0, num_blocks*block_size-1]. The values of elements in slotMapping must be unique. If the values are duplicate, the correctness cannot be ensured.
    - If both key and value are 3-dimensional, the first two dimensions of key and value must have the same shape.
    - If both key and value are 4-dimensional, the first three dimensions of key and value must have the same shape, and the third dimension of keyCacheRef and valueCacheRef must be 1.
    - If both key and value are 4-dimensional, compressLensOptional and seqLensOptional are mandatory. If both key and value are 3-dimensional, compressLensOptional, compressSeqOffsetOptional, and seqLensOptional are optional.
    - If both key and value are 4-dimensional, slotMapping is 2-dimensional, the value of the first dimension of slotMapping is equal to the value of the first dimension of key (batch), and the value of the second dimension of slotMapping is equal to the value of the third dimension of key (num_head) (corresponding to scenario 3).
    - If both key and value are 4-dimensional, seqLensOptional is 1-dimensional, and the value of seqLensOptional is equal to the value of the first dimension of key (batch) (corresponding to scenario 3).
    - If both key and value are 3-dimensional and seqLensOptional is available, the sum of all values in seqLensOptional is equal to the value of the first dimension of key (num_blocks) (corresponding to scenarios 4 and 5).
    - The value of each element in seqLensOptional and compressLensOptional must meet the following formula: reduceSum(seqLensOptional[i] - compressLensOptional[i]) <= num_blocks * block_size (corresponding to scenarios 3, 4, and 5).
    - When cacheModeOptional is set to PA_NZ, the second-to-last dimension of keyCacheRef and valueCacheRef must be less than UINT16_MAX (corresponding to scenario 1).

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
  int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType, aclTensor** tensor, aclFormat format) {
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

    // Create a key aclTensor.
    ret = CreateAclTensor(hostKey, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &key, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a value aclTensor.
    ret = CreateAclTensor(hostValue, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &value, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a slotMapping aclTensor.
    ret = CreateAclTensor(hostSlotMapping, slotMappingShape, &slotMappingDeviceAddr, aclDataType::ACL_INT32, &slotMapping, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a keyCache aclTensor.
    ret = CreateAclTensor(hostKeyCacheRef, keyCacheShape, &keyCacheDeviceAddr, aclDataType::ACL_FLOAT16, &keyCache, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a valueCache ACL tensor.
    ret = CreateAclTensor(hostValueCacheRef, valueCacheShape, &valueCacheDeviceAddr, aclDataType::ACL_FLOAT16, &valueCache, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    aclIntArray *strides = aclCreateIntArray(hostStrides.data(), 2);
    CHECK_RET(strides != nullptr, return ACL_ERROR_INTERNAL_ERROR);
    aclIntArray *offsets = aclCreateIntArray(hostOffsets.data(), 2);
    CHECK_RET(offsets != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnScatterPaKvCache.
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
