# aclnnIndexerQuantCache

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T10:34:30.390Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
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

- Description: Performs in-place compressed updates on the KV Cache in the Epilog phase of the Indexer attention mechanism. Compresses `float16`/`bfloat16` activations into FP8 (E4M3/E5M2), HiFloat8 (floating-point codes stored in UINT8), MX-FP4, or HiFloat4 format through per-block dynamic quantization, and scatters the quantization results and the corresponding `cacheScale` values to the cache based on `slotMapping`. Tokens with a value of -1 are skipped. Supports five quantization modes: MX-FP8, Normal, HiFloat8, MX-FP4, and HiFloat4.

- Formula:

  Calculates the amax of each group along the last dimension (d axis) of `x` based on the quantization granularity, and quantizes each group into the target dtype. Let $x_g$ denote group g (one group per 32 elements in MX-FP8/MX-FP4 mode; the entire row as one group in Normal/HiFloat8 mode):

  $$
  scale_g = \frac{\max(|x_g|)}{Q\_MAX}, \quad q_i = \mathrm{round}\left(\frac{x_i}{scale_g}\right)
  $$

  - Use case 1 (quantMode=0, MX-FP8): scale is stored as `float8_e8m0`, and is always rounded up to a power of 2, independently of `roundScale`.
  - Use case 2 (quantMode=1, Normal): dynamic per-row quantization; scale is stored as `float32`. When `roundScale=true`, the positive scale computed by the formula above is rounded up to `2^ceil(log2(scale))` for quantization and writeback. UINT8 cache stores HiFloat8 codes in Normal mode.
  - Use case 3 (quantMode=2, HiFloat8): the output is hifloat8 after $x \times xScale$.
  - Use case 4 (quantMode=3, MX-FP4): quantization to FP4 is performed by standard MX block (32 elements per block) (`uint8` packs 2 fp4 values per byte), and scale is stored as `float8_e8m0`.

  - Use case 5 (quantMode=4, HiFloat4): Supported only on ascend960dt (DAV_9201). Requires `d % 64 == 0`, HIFLOAT4 cache, and HIFLOAT4_SCALE cacheScale, with d/64 scale elements per row.

- Example:

  ```python
  # cache/cache_scale are both 4D [blockNum, blockSize, 1, headDim]; num_slots = blockNum*blockSize = 2048
  cache shape:        [128, 16, 1, 128]   # headDim=128 >= d(128)
  cache_scale shape:  [128, 16, 1, 1]     # mode1 scaleCol=1
  x shape:            [1024, 128]
  slot_mapping shape: [1024]
  quantMode = 1
  roundScale = true
  xScale = 1.0
  cache out shape:        [128, 16, 1, 128]   (in-place update)
  cache_scale out shape:  [128, 16, 1, 1]     (in-place update)
  ```

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnIndexerQuantCacheGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call the "aclnnIndexerQuantCache" API to perform the computation.

```cpp
aclnnStatus aclnnIndexerQuantCacheGetWorkspaceSize(
  aclTensor       *cacheRef,
  aclTensor       *cacheScaleRef,
  const aclTensor *x,
  const aclTensor *slotMapping,
  int64_t          quantMode,
  bool             roundScale,
  float            xScale,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnIndexerQuantCache(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnIndexerQuantCacheGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 220px">
  <col style="width: 110px">
  <col style="width: 300px">
  <col style="width: 300px">
  <col style="width: 240px">
  <col style="width: 100px">
  <col style="width: 160px">
  <col style="width: 110px">
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
      <th>Non-Contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>cacheRef (aclTensor*)</td>
      <td>Input/Output</td>
      <td>Indicates the paged KV Cache of the current layer, updated in place, corresponding to the quantization target in the formula. <b>Only four dimensions are supported</b> (see Constraints for the shape and headDim length rules). In MX-FP4 mode, the cache is FP4 (two FP4 values packed per byte).</td>
      <td><li>Empty tensors are not supported.</li><li>Serves as both input and output (in-place operation).</li><li>The data type must match quantMode.</li><li><b>Only four dimensions are supported</b> [blockNum, blockSize, 1, headDim], with dim2 fixed to 1 and <b>only the blockNum dimension supports non-contiguity</b> (paging).</li></td>
      <td>FLOAT8_E4M3FN, FLOAT8_E5M2, UINT8, FLOAT4_E2M1, FLOAT4_E1M2, HIFLOAT4</td>
      <td>ND</td>
      <td>[blockNum, blockSize, 1, headDim] (num_slots = blockNum × blockSize)</td>
      <td>√ (blockNum dimension only)</td>
    </tr>
    <tr>
      <td>cacheScaleRef (aclTensor*)</td>
      <td>Input/Output</td>
      <td>Indicates the scale factor for per-block quantization, updated in place, corresponding to scale in the formula. <b>Only four dimensions are supported</b>, following the same rules as cache. headDim must be ≥ scaleCol (for details, see Constraints).</td>
      <td><li>Empty tensors are not supported.</li><li>Serves as both input and output (in-place operation).</li><li>Data type must match quantMode.</li><li><b>Only four dimensions</b> [blockNum, blockSize, 1, headDim] are supported, with dim2 fixed to 1 and headDim ≥ scaleCol, and <b>only the blockNum dimension</b> supports non-contiguous (paged) layout.</li></td>
      <td>FLOAT, FLOAT8_E8M0, HIFLOAT4_SCALE</td>
      <td>ND</td>
      <td>[blockNum, blockSize, 1, scaleHeadDim] (scaleHeadDim ≥ scaleCol)</td>
      <td>√ (blockNum dimension only)</td>
    </tr>
    <tr>
      <td>x (const aclTensor*)</td>
      <td>Input</td>
      <td>Activation input to be quantized, corresponding to x in the formula.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>[bs, d]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>slotMapping (const aclTensor*)</td>
      <td>Input</td>
      <td>Index mapping from tokens to cache slots. A value of -1 indicates that the token is skipped.</td>
      <td><li>Empty tensors are not supported.</li><li>The number of dimensions must be 1 less than that of x.</li><li>Element values range from -1 to num_slots - 1.</li></td>
      <td>INT32</td>
      <td>ND</td>
      <td>[bs]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>quantMode (int64_t)</td>
      <td>Input</td>
      <td>Quantization mode.</td>
      <td>Enumerated values: 0, 1, 2, 3, and 4. For details, see the use cases above.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>roundScale (bool)</td>
      <td>Input</td>
      <td>Whether to round the scale up to a power of 2 in Normal mode.</td>
      <td>Takes effect only when quantMode=1. The recommended value is true.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>xScale (float)</td>
      <td>Input</td>
      <td>Global scale multiplier in HiFloat8 mode.</td>
      <td>Takes effect only when quantMode=2. The recommended value is 1.0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor containing the operator calculation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  - The data type combination must match the quantization mode: for quantMode=0, cache is FP8 and scale is FLOAT8_E8M0; for quantMode=1/2, cache is FP8/UINT8 and scale is FLOAT; for quantMode=3, cache is FP4 and scale is FLOAT8_E8M0.

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 300px">
  <col style="width: 150px">
  <col style="width: 550px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>cacheRef, cacheScaleRef, x, or slotMapping is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>cacheRef, x, or slotMapping is an empty tensor; or the cacheRef/cacheScaleRef layout violates the constraint (must be contiguous or non-contiguous only along the blockNum dimension).</td>
    </tr>
  </tbody></table>

## aclnnIndexerQuantCache

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 700px">
  </colgroup>
  <thead>
    <tr><th>Parameter</th><th>Input/Output</th><th>Description</th></tr>
  </thead>
  <tbody>
    <tr><td>workspace</td><td>Input</td><td>Workspace memory address requested on the Device.</td></tr>
    <tr><td>workspaceSize</td><td>Input</td><td>Workspace size requested on the Device, obtained by the first-phase API aclnnIndexerQuantCacheGetWorkspaceSize.</td></tr>
    <tr><td>executor</td><td>Input</td><td>Op executor that contains the operator computation flow.</td></tr>
    <tr><td>stream</td><td>Input</td><td>Stream for executing tasks.</td></tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation: aclnnIndexerQuantCache defaults to a deterministic implementation.
- The data type combination must match quantMode: in Normal/HiFloat8 modes, cacheScale is FLOAT; in MX-FP8/MX-FP4 modes, cacheScale is FLOAT8_E8M0; in MX-FP4 mode, cache is FP4 (packed in uint8).
- **Both cache and cacheScale support only the four-dimensional shape** `[blockNum, blockSize, 1, headDim]`. The second-to-last dimension is fixed to 1 (one quantization vector per token). Other dimension counts are not supported; num_slots = blockNum × blockSize.
- cache/cacheScale **supports non-contiguity only in the blockNum dimension** (paging): blocks can be sparsely arranged, but the data within each block (blockSize and headDim dimensions) must be contiguous.
- **headDim length constraints**:
  - **cache.headDim ≥ d** (in MX-FP4/HiFloat4 mode, counted in 4-bit elements; d values occupy ⌈d/2⌉ bytes).
  - **In MX-FP4/HiFloat4 (quantMode=3/4) mode, cache.headDim must be an even number**.
  - **cacheScale.headDim ≥ scaleCol**, where scaleCol is ⌈d/32⌉ for MX-FP8/MX-FP4 (quantMode=0/3), 1 for Normal/HiFloat8 (quantMode=1/2), and d/64 for HiFloat4 (quantMode=4).
  - Example: d=128 and quantMode=0 → scaleCol=4 → cache.headDim ≥ 128 and cacheScale.headDim ≥ 4.
- The last dimension (d axis) of x must be divisible by 32 and d ≤ 8192.
- The dimension of slotMapping should equal the dimension of x minus 1; that is, slotMapping is obtained by flattening all dimensions of x except the last one.
- Tokens with a value of -1 in slotMapping are skipped; the values of other valid elements range from [0, num_slots - 1] and must be unique; otherwise, result correctness is not guaranteed.

## Example

The sample code is as follows, for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_indexer_quant_cache.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
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
    // Call aclrtMalloc to allocate memory on the device
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy data from the host to the device memory
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    // Compute the strides of the contiguous tensor
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    // Call the aclCreateTensor API to create an aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output as required by the API (the Normal mode is used as an example here)
    // cache/cacheScale is 4D [blockNum, blockSize, 1, headDim]; num_slots = 128*16 = 2048
    std::vector<int64_t> cacheShape = {128, 16, 1, 128}; // KV Cache (float8_e4m3fn), headDim=128 >= d
    std::vector<int64_t> scaleShape = {128, 16, 1, 1};   // Scale per block (float32), mode1 scaleCol=1
    std::vector<int64_t> xShape = {1024, 128};          // Activation to be quantized
    std::vector<int64_t> slotShape = {1024};            // slot index mapping
    void* cacheDeviceAddr = nullptr;
    void* scaleDeviceAddr = nullptr;
    void* xDeviceAddr = nullptr;
    void* slotDeviceAddr = nullptr;
    aclTensor* cache = nullptr;
    aclTensor* cacheScale = nullptr;
    aclTensor* x = nullptr;
    aclTensor* slotMapping = nullptr;
    std::vector<uint8_t> cacheHostData(GetShapeSize(cacheShape), 0);   // fp8 stored as uint8
    std::vector<float> scaleHostData(GetShapeSize(scaleShape), 0);
    std::vector<uint16_t> xHostData(GetShapeSize(xShape), 0);          // float16 stored as uint16
    std::vector<int32_t> slotHostData(GetShapeSize(slotShape));
    for (size_t i = 0; i < slotHostData.size(); ++i) {
        slotHostData[i] = static_cast<int32_t>(i);
    }
    // Create the cache aclTensor (in-place input/output).
    ret = CreateAclTensor(cacheHostData, cacheShape, &cacheDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &cache);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the cacheScale aclTensor (in-place input/output).
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &cacheScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the slotMapping aclTensor.
    ret = CreateAclTensor(slotHostData, slotShape, &slotDeviceAddr, aclDataType::ACL_INT32, &slotMapping);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Attributes
    int64_t quantMode = 1;   // Normal per-block dynamic quantization
    bool roundScale = true;
    float xScale = 1.0f;

    // 3. Call the CANN operator library API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnIndexerQuantCache.
    ret = aclnnIndexerQuantCacheGetWorkspaceSize(cache, cacheScale, x, slotMapping, quantMode, roundScale, xScale,
                                                 &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIndexerQuantCacheGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnIndexerQuantCache.
    ret = aclnnIndexerQuantCache(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIndexerQuantCache failed. ERROR: %d\n", ret); return ret);
    // 4. (boilerplate) Synchronize and wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. cache and cacheScale are updated in place. The results are written back to the corresponding device memory and can be copied back to the host as needed.

    // 6. Release the aclTensor. Modify the code according to the definition of the specific API.
    aclDestroyTensor(cache);
    aclDestroyTensor(cacheScale);
    aclDestroyTensor(x);
    aclDestroyTensor(slotMapping);

    // 7. Release device resources.
    aclrtFree(cacheDeviceAddr);
    aclrtFree(scaleDeviceAddr);
    aclrtFree(xDeviceAddr);
    aclrtFree(slotDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
