# aclnnScatterPaCache

[📄 View Source Code](https://gitcode.com/cann/ops-transformer/tree/master/attention/scatter_pa_cache)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    ×     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- Operator function: updates the key at the specified position in the KCache.

- Formulas:
  - Scenario 1:
  
    ```text
    key:[batch * seq_len, num_head, k_head_size]
    keyCache:[num_blocks, block_size, num_head, k_head_size]
    slotMapping:[batch * seq_len]
    cacheMode:"Norm"
    ```  

    $$
    keyCache = slotMapping(key)
    $$

  - Scenario 2:

    ```text
    key:[batch, seq_len, num_head, k_head_size]
    keyCache:[num_blocks, block_size, 1, k_head_size]
    slotMapping:[batch, num_head]
    compressLensOptional:[batch, num_head]
    compressSeqOffsetOptional:[batch * num_head]
    seqLensOptional:[batch]
    cacheMode:"Norm"
    ```

    $$
    \begin{aligned}
    keyCache =\ & slotMapping(key[: compressSeqOffset], \\
    & ReduceMean(key[compressSeqOffset : compressSeqOffset + compressLens]), \\
    & key[compressSeqOffset + compressLens : seqLens])
    \end{aligned}
    $$

  - Scenario 3:

    ```text
    key:[batch, seq_len, num_head, k_head_size]
    keyCache:[num_blocks, block_size, 1, k_head_size]
    slotMapping:[batch, num_head]
    compressLensOptional:[batch * num_head]
    seqLensOptional:[batch]
    cacheMode:"Norm"
    ```

    $$
    keyCache = slotMapping(key[seqLens - compressLens : seqLens])
    $$
  
  The preceding scenarios are distinguished based on the constructed parameters. If the first input parameter is constructed, scenario 1 is used. If the second input parameter is constructed, scenario 2 is used. If the third input parameter is constructed, scenario 3 is used. In scenario 1, the compressLensOptional, seqLensOptional and compressSeqOffsetOptional parameters are unavailable. In scenario 3, the compressSeqOffsetOptional parameter is unavailable.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnScatterPaCacheGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnScatterPaCache` is called to perform computation.

```c++
aclnnStatus aclnnScatterPaCacheGetWorkspaceSize(
  const aclTensor *key, 
  aclTensor       *keyCacheRef, 
  const aclTensor *slotMapping, 
  const aclTensor *compressLensOptional, 
  const aclTensor *compressSeqOffsetOptional, 
  const aclTensor *seqLensOptional, 
  char            *cacheMode, 
  uint64_t        *workspaceSize, 
  aclOpExecutor  **executor)
```

```c++
aclnnStatus aclnnScatterPaCache(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```

## aclnnScatterPaCacheGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1582px"><colgroup>
  <col style="width: 231px">
  <col style="width: 157px">
  <col style="width: 271px">
  <col style="width: 365px">
  <col style="width: 161px">
  <col style="width: 121px">
  <col style="width: 131px">
  <col style="width: 145px">
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
      <th class="tg-0pky">Non-continuous tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">key(aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Key value to be updated, which is the key in the formula.</td>
      <td class="tg-0pky"><ul><li>Empty tensors are supported. </li><li>HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1 and FLOAT4_E1M2 support only the scenario where the key is 3-dimensional. </li><li>When the shape is [batch * seq_len, num_head, k_head_size] or [batch, seq_len, num_head, k_head_size], FLOAT4_E2M1 and FLOAT4_E1M2, k_head_size must be an even number.</li></ul></td>
      <td class="tg-0pky">FLOAT16, FLOAT, BFLOAT16, INT8, UINT8, INT16, UINT16, INT32, UINT32, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1, FLOAT4_E1M2</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">3-4</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">keyCacheRef(aclTensor*)</td>
      <td class="tg-0pky">Input/Output</td>
      <td class="tg-0pky">Key cache to be updated, which is the key cache in the formula.</td>
      <td class="tg-0pky"><ul><li>Empty tensors are supported. </li><li>When the key is 3D, the shape is [num_blocks, block_size, num_head, k_head_size]. When the key is 4D, the shape is [num_blocks, block_size, 1, k_head_size].</li></ul></td>
      <td class="tg-0pky">Same as the key.</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">4</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">slotMapping(aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Storage offset of each token of the key in the cache, which is slotMapping in the formula.</td>
      <td class="tg-0pky"><ul><li>Empty tensors are supported. </li><li>When the key is 3D, the shape is [batch * seq_len]. When the key is 4D, the shape is [batch, num_head]. </li><li>The value range is [0, num_blocks * block_size – 1], and the element values must be unique. If the values are duplicate, the correctness cannot be ensured.</li></ul></td>
      <td class="tg-0pky">INT32, INT64</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-2</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">compressLensOptional(aclTensor*)</td>
      <td class="tg-0pky">Optional input</td>
      <td class="tg-0pky">Compression length, which is compressLens in the formula.</td>
      <td class="tg-0pky"><ul><li>Empty tensors are supported. </li><li>If the key is 4-dimensional and compressSeqOffsetOptional is not a null pointer, the shape is [batch, num_head]. If the key is 4-dimensional and compressSeqOffsetOptional is a null pointer, the shape is [batch * num_head]. </li><li>In scenario 1, a null pointer is transferred.</li></ul></td>
      <td class="tg-0pky">The value is the same as that of slotMapping.</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-2</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">compressSeqOffsetOptional(aclTensor*)</td>
      <td class="tg-0pky">Optional input</td>
      <td class="tg-0pky">Compression start point of each head in each batch, which is compressSeqOffset in the formula.</td>
      <td class="tg-0pky"><ul><li>Empty tensors are supported. </li><li>The shape is [batch * num_head]. </li><li>In scenarios 1 and 3, a null pointer is passed.</li></ul></td>
      <td class="tg-0pky">Same as slotMapping.</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">seqLensOptional(aclTensor*)</td>
      <td class="tg-0pky">Optional input</td>
      <td class="tg-0pky">Actual sequence length of each batch, which is the seqLens in the formula.</td>
      <td class="tg-0pky"><ul><li>Empty tensors are supported. </li><li>The shape meets the [batch] requirement. </li><li>In scenario 1, a null pointer is passed.</li></ul></td>
      <td class="tg-0pky">Same as slotMapping.</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">cacheMode(char*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Memory format of keyCacheRef.</td>
      <td class="tg-0pky">Reserved. Currently, only the ND format is supported. This parameter does not take effect.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">workspaceSize (uint64_t*)</td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">Returns the workspace size to be allocated on the device.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">executor(aclOpExecutor**)</td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">Returns the operator executor, including the operator execution process.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
  </tbody></table>

- **Return Value**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.

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
      <td>The input key, keyCacheRef, or slotMapping is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type of the key is not supported.</td>
    </tr>
    <tr>
      <td>The data types of key and keyCacheRef are inconsistent.</td>
    </tr>
    <tr>
      <td>The data types of slotMapping, compressLensOptional, compressSeqOffsetOptional, and seqLensOptional are inconsistent.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>561002</td>
      <td>The number of dimensions of the key is not 3 or 4.</td>
    </tr>
  </tbody>
  </table>

## aclnnScatterPaCache

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
      <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnScatterPaCacheGetWorkspaceSize.</td>
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

- **Return Value**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computing: The default deterministic implementation of aclnnScatterPaCache is used.
- The variables used by shape in the parameter description are described as follows:
  - batch: number of input sequences (number of samples processed at a time). The value is a positive integer.
  - seq_len: length of the sequence. The value is a positive integer.
  - num_head: number of heads in multi-head attention. The value is a positive integer.
  - k_head_size: feature dimension of the key in each attention head (length of the key in a single head). The value is a positive integer.
  - num_blocks: total number of blocks pre-allocated in keyCache, which is used to store the key data of all sequences. The value is a positive integer.
  - block_size: number of tokens contained in each cache block. The value is a positive integer.
- Input value range restriction: Each element value in seqLensOptional and compressLensOptional must meet the following formula: reduceSum(seqLensOptional[i] - compressLensOptional[i] + 1) <= num_blocks * block_size (corresponding to scenarios 2 and 3).

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

- Ascend 950PR/Ascend 950DT:

  ```c++
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_scatter_pa_cache.h"

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
    std::vector<int64_t> keyShape = {2, 2, 3, 7};
    std::vector<int64_t> keyCacheShape = {2, 2, 1, 7};
    std::vector<int64_t> slotMappingShape = {2, 3};

    std::vector<int64_t> compressLensShape = {2, 3};
    std::vector<int64_t> compressSeqOffsetShape = {6};
    std::vector<int64_t> seqLensShape = {2};
    void* keyDeviceAddr = nullptr;
    void* slotMappingDeviceAddr = nullptr;
    void* keyCacheDeviceAddr = nullptr;
    void* compressLensDeviceAddr = nullptr;
    void* compressSeqOffsetDeviceAddr = nullptr;
    void* seqLensDeviceAddr = nullptr;

    aclTensor* key = nullptr;
    aclTensor* slotMapping = nullptr;
    aclTensor* keyCache = nullptr;
    aclTensor* compressLens = nullptr;
    aclTensor* compressSeqOffset = nullptr;
    aclTensor* seqLens = nullptr;
    char* cacheMode = const_cast<char*>("Norm");

    std::vector<float> hostKey = {1};
    std::vector<int32_t> hostSlotMapping = {0, 3, 6, 9, 12, 15};
    std::vector<float> hostKeyCacheRef = {1};
    std::vector<int32_t> hostCompressLens = {1, 0, 0, 0, 1, 0};
    std::vector<int32_t> hostCompressSeqOffset = {0, 0, 1, 0, 1, 1};
    std::vector<int32_t> hostSeqLens = {2, 1};

    // Create a key aclTensor.
    ret = CreateAclTensor(hostKey, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT, &key);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the slotMappitng aclTensor.
    ret = CreateAclTensor(hostSlotMapping, slotMappingShape, &slotMappingDeviceAddr, aclDataType::ACL_INT32, &slotMapping);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the keyCache aclTensor.
    ret = CreateAclTensor(hostKeyCacheRef, keyCacheShape, &keyCacheDeviceAddr, aclDataType::ACL_FLOAT, &keyCache);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the compressLens aclTensor.
    ret = CreateAclTensor(hostCompressLens, compressLensShape, &compressLensDeviceAddr, aclDataType::ACL_INT32, &compressLens);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the compressSeqOffset aclTensor.
    ret = CreateAclTensor(hostCompressSeqOffset, compressSeqOffsetShape, &compressSeqOffsetDeviceAddr, aclDataType::ACL_INT32, &compressSeqOffset);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the seqLens aclTensor.
    ret = CreateAclTensor(hostSeqLens, seqLensShape, &seqLensDeviceAddr, aclDataType::ACL_INT32, &seqLens);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first part of the aclnnScatterPaCache API.
    ret = aclnnScatterPaCacheGetWorkspaceSize(key, keyCache, slotMapping, compressLens, compressSeqOffset, seqLens, cacheMode, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScatterPaCacheGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second part of the aclnnScatterPaCache API.
    ret = aclnnScatterPaCache(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScatterPaCache failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
    auto size = GetShapeSize(keyShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), keyDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
      LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
    aclDestroyTensor(key);
    aclDestroyTensor(slotMapping);
    aclDestroyTensor(keyCache);
    aclDestroyTensor(compressLens);
    aclDestroyTensor(compressSeqOffset);
    aclDestroyTensor(seqLens);
    // 7. Release device resources. Set the parameters based on the API definition.
    aclrtFree(keyDeviceAddr);
    aclrtFree(slotMappingDeviceAddr);
    aclrtFree(keyCacheDeviceAddr);
    aclrtFree(compressLensDeviceAddr);
    aclrtFree(compressSeqOffsetDeviceAddr);
    aclrtFree(seqLensDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
  }
  ```
