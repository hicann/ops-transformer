# aclnnGatherPaKvCache

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T09:52:37.950Z -->

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

- Description: Based on the blockId values in blockTables and the seqLen of key/value in seqLens, copies and concatenates non-contiguous tokens from keyCache/valueCache into contiguous key/value sequences.
- Formula:
  - The first dimension of keyRef/valueRef depends on the size of seq_lens.
  - If isSeqLensCumsum is true, the last value in seqLens is the size of the first dimension of keyRef/valueRef: keyRef[dim0] = seqLens[-1]
  - If isSeqLensCumsum is false, the cumulative sum of all values in seqLens is the size of the first dimension of keyRef/valueRef: keyRef[dim0] = sum(seqLens)

  The constraints on **keyRef** and **valueRef** are as follows:
  - Keep the size of each token within 148k. For example, for the fp16/bf16 data type, set num_heads *head_size(keyRef/valueRef) to 128*576.
  - To ensure that keyRef/valueRef come entirely from the actual elements in the Cache, blockTables must satisfy the coverage condition: for each batch `i`, let `length[i]` denote the sequence length of that batch (`length[i] = seqLens[i]` when isSeqLensCumsum is false; `length[i] = seqLens[i + 1] - seqLens[i]` when isSeqLensCumsum is true), and let `tableStart[i]` denote the starting table column offset (0 when seqOffsetOptional is not passed in; `seqOffsetOptional[i] / block_size` using integer division when passed in). `tableStart[i] >= 0` and `tableStart[i] + ceil(length[i] / block_size) <= block_indices` must be satisfied.

- Example:

  ```python
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

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnGatherPaKvCacheGetWorkspaceSize" API first to obtain the workspace size required for the computation and the executor that contains the computation flow of the operator, and then call the "aclnnGatherPaKvCache" API to execute the computation.

```Cpp
aclnnStatus aclnnGatherPaKvCacheGetWorkspaceSize(
  const aclTensor *keyCache,
  const aclTensor *valueCache,
  const aclTensor *blockTables,
  const aclTensor *seqLens,
  aclTensor       *keyRef,
  aclTensor       *valueRef,
  const aclTensor *seqOffsetOptional,
  char*            cacheMode,
  bool             isSeqLensCumsum,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnGatherPaKvCache(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnGatherPaKvCacheGetWorkspaceSize

- **Parameters**

  <table class="tg" style="undefined;table-layout: fixed; width: 1602px"><colgroup>
  <col style="width: 247px">
  <col style="width: 132px">
  <col style="width: 232px">
  <col style="width: 369px">
  <col style="width: 208px">
  <col style="width: 119px">
  <col style="width: 150px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">Parameter</th>
      <th class="tg-0pky">Input/Output</th>
      <th class="tg-0pky">Description</th>
      <th class="tg-0pky">Instruction</th>
      <th class="tg-0pky">Data Type</th>
      <th class="tg-0pky">Data Format</th>
      <th class="tg-0pky">Dimension (shape)</th>
      <th class="tg-0pky">Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">keyCache (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Indicates the key vector cache stored in the current layer.</td>
      <td class="tg-0pky">When cacheMode is "Norm", the shape is [num_blocks, block_size, num_heads, head_size_k], and the data format must be ND.<br>When cacheMode is "PA_NZ", the shape is [num_blocks, num_heads * head_size_k // elenum_aligned, block_size, elenum_aligned] (b8 case: elenum_aligned=32; b16 case: 16; b32 case: 8. b8 indicates that the bit width of each data element is 8bit, such as INT8; b16 indicates that the bit width of each data element is 16bit, such as INT16; b32 indicates that the bit width of each data element is 32bit, such as INT32), and the data format must be FRACTAL_NZ.</td>
      <td class="tg-0pky">INT8, FLOAT16, BFLOAT16, FLOAT, UINT8, INT16, UINT16, INT32, UINT32, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td class="tg-0pky">ND, FRACTAL_NZ</td>
      <td class="tg-0pky">4</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">valueCache (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Indicates the value vector cache stored in the current layer.</td>
      <td class="tg-0pky">When cacheMode is "Norm", shape is [num_blocks, block_size, num_heads, head_size_v], and the data format must be ND.<br>When cacheMode is "PA_NZ", shape is [num_blocks, num_heads * head_size_v // elenum_aligned, block_size, elenum_aligned] (In the b8 use case, elenum_aligned=32; in the b16 use case, elenum_aligned=16; in the b32 use case, elenum_aligned=8. b8 indicates that the bit width of each data element is 8 bit, such as INT8; b16 indicates that the bit width of each data element is 16 bit, such as INT16; b32 indicates that the bit width of each data element is 32 bit, such as INT32), and the data format must be FRACTAL_NZ.</td>
      <td class="tg-0pky">INT8, FLOAT16, BFLOAT16, FLOAT, UINT8, INT16, UINT16, INT32, UINT32, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td class="tg-0pky">ND, FRACTAL_NZ</td>
      <td class="tg-0pky">4</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">blockTables (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Indicates the physical block index corresponding to each sequence.</td>
      <td class="tg-0pky">shape is [batch, block_indices], where both batch and block_indices must be greater than 0. The value range of elements is [0, num_blocks), indicating that blockId ranges from 0 to num_blocks-1.<br>To ensure that the output comes entirely from the actual elements in the Cache, the blockTables width must satisfy the coverage condition in the feature description.</td>
      <td class="tg-0pky">INT32, INT64</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">2</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">seqLens (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Indicates the sequence length corresponding to each batch.</td>
      <td class="tg-0pky">The shape is [batch] or [batch + 1].<br>If isSeqLensCumsum is false, the shape is [batch].<br>If isSeqLensCumsum is true, the shape is [batch + 1].<br>The value range of elements is [0, num_blocks).</td>
      <td class="tg-0pky">Keep consistent with blockTables</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">keyRef (aclTensor*)</td>
      <td class="tg-0pky">Input/Output</td>
      <td class="tg-0pky">Indicates the key tensor.</td>
      <td class="tg-0pky">When cacheMode is "Norm", the shape is [num_tokens, num_heads, head_size_k].<br>When cacheMode is "PA_NZ", the shape is [num_tokens, num_heads * head_size_k].</td>
      <td class="tg-0pky">Keep consistent with keyCache</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">2-3</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">valueRef (aclTensor*)</td>
      <td class="tg-0pky">Input/Output</td>
      <td class="tg-0pky">Indicates the value vector.</td>
      <td class="tg-0pky">When cacheMode is "Norm", the shape is [num_tokens, num_heads, head_size_v].<br>When cacheMode is "PA_NZ", the shape is [num_tokens, num_heads * head_size_v].</td>
      <td class="tg-0pky">Keep consistent with valueCache</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">2-3</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">seqOffsetOptional (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">If passed, indicates that a head offset exists when obtaining blockId from blockTables (the offset is `seqOffsetOptional[i] / block_size`, where `i` indicates a batch); if not passed, indicates that no offset is required.</td>
      <td class="tg-0pky">The shape is [batch].</td>
      <td class="tg-0pky">Keep consistent with blockTables</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">cacheMode (char*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Supports ["Norm", "PA_NZ"] modes, indicating that the data formats of input keyCache and valueCache are ND and FRACTAL_NZ respectively.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">isSeqLensCumsum (bool)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Indicates whether seqLens is a cumulative sum.</td>
      <td class="tg-0pky">false indicates a non-cumulative sum, for example, seqLens is [1, 3, 5, 3, 7].<br>true indicates a cumulative sum, for example, seqLens is [0, 1, 4, 9, 12, 19], where the 0th element must be 0.<br>For a cumulative sum, `seqlens[i + 1] - seqlens[i]` equals the non-cumulative `seqlens[i]`.</td>
      <td class="tg-0pky">bool</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">workspaceSize (uint64_t*)</td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">Returns the size of the workspace to be allocated on the Device side.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">executor (aclOpExecutor**)</td>
      <td class="tg-0lax">output</td>
      <td class="tg-0lax">Returns the op executor, which contains the operator computation process.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>:
    - The combination of keyCache in FLOAT8_E4M3FN and valueCache in FLOAT16 or BFLOAT16 is supported.
    - When the blockTables width does not satisfy the coverage condition in the feature description, tokens corresponding to logical blocks whose blockTableOffset exceeds the block_indices range will be zero-filled and written to keyRef/valueRef; blocks whose blockId is out of range (beyond [0, num_blocks)) are also zero-filled. This behavior is undefined on other product paths. Ensure that blockTables satisfies the coverage condition.
  <!-- end id7 -->
  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - cacheMode supports only "Norm", and keyCache and valueCache support only the ND format.
    - The input keyCache, valueCache, keyRef, and valueRef do not support the FLOAT, UINT8, INT16, UINT16, INT32, UINT32, HIFLOAT8, FLOAT8_E5M2, and FLOAT8_E4M3FN data types.
    - The input blockTables, seqLens, and seqOffsetOptional do not support the INT64 data type.

  <!-- end id8 -->

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:
  <table style="undefined;table-layout: fixed; width: 1147px"><colgroup>
    <col style="width: 283px">
    <col style="width: 120px">
    <col style="width: 744px">
    </colgroup>
    <thead>
      <tr>
        <th>Return Values</th>
        <th>Error Code</th>
        <th>Description</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>Input is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="2">161002</td>
        <td>The input data type is not in the supported range.</td>
      </tr>
      <tr>
        <td>The input dimensions do not match.</td>
      </tr>
    </tbody>
    </table>

## aclnnGatherPaKvCache

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
      <td>Workspace memory address allocated on the device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the device side, obtained by the first-phase API aclnnGatherPaKvCacheGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, including the computation process of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream used to execute the task.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnGatherPaKvCache defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

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
  // Call aclrtMalloc to allocate memory on the device side.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host-side data to the device-side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor API to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) device/stream initialization. For details, see the list of external acl APIs.
  // Fill in deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs. Customize the construction as required by the API.
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
  // Create the input and output aclTensors of GatherPaKvCache.
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


  // 3. Call the CANN operator library API. Change it to the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnGatherPaKvCache.
  ret = aclnnGatherPaKvCacheGetWorkspaceSize(keyCache, valueCache, blockTables, seqLens, key , value, seqOffset,
  cacheMode, isSeqLensCumsum, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGatherPaKvCacheGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API aclnnGatherPaKvCache.
  ret = aclnnGatherPaKvCache(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGatherPaKvCache failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait synchronously for task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device side to the host side. Modify the code according to the specific API definition.
  auto size = GetShapeSize(keyShape);
  std::vector<uint16_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), keyDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclIntArray. Modify the code according to the specific API definition.
  aclDestroyTensor(keyCache);
  aclDestroyTensor(valueCache);
  aclDestroyTensor(blockTables);
  aclDestroyTensor(seqLens);
  aclDestroyTensor(key);
  aclDestroyTensor(value);
  aclDestroyTensor(seqOffset);

  // 7. Release device resources. Modify the code according to the specific API definition.
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
