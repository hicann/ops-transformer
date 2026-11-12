# aclnnKvCompressEpilog

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T10:55:47.358Z -->

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

- Description: In the epilog phase of the KV Cache, performs in-place compressed updates on the KV Cache. Quantizes and compresses the `bfloat16` activations, and then scatters them into the cache based on `slotMapping`. Tokens with the value -1 are skipped. The last 64 columns of the last dimension `d` of `x` form the rope segment, and the first `d-64` columns form the nope segment. Supports three quantization modes: group(bf16 scale), group(e8m0 scale), and rope hifloat8 static + nope FLOAT4_E2M1 dynamic.

- Formula:

  Computes the amax of each group along the last dimension (d axis) of `x` and quantizes each group to the target dtype. Group g is denoted as $x_g$:

  - Scenario 1 (quantMode=0): group(64) is quantized to FP8, `scale` is stored as `bfloat16`; the rope part remains `bfloat16`.

    $$
    scale_g = \frac{\max(|x_g|)}{FP8\_MAX}, \quad q_i = \mathrm{round}\left(\frac{x_i}{scale_g}\right)
    $$

  - Scenario 2 (quantMode=1): Same as scenario 1, but `scale` is stored as `float8_e8m0`. When `roundScale=true`, `scale` is rounded up to a power of 2.

  - Scenario 3 (quantMode=2): The rope part (last 64 columns) applies hifloat8 **static** quantization, and the nope part (first d-64 columns) applies per-group **FLOAT4_E2M1 dynamic** quantization (`FP4\_MAX=6.0`), `scale` is written as `bfloat16`, and `roundScale` does not take effect:

    $$
    rope_i = \mathrm{hifloat8}(x_i \cdot xScale), \qquad
    scale_g = \frac{\max(|x_g|)}{FP4\_MAX}, \quad nope_i = \mathrm{FLOAT4\_E2M1}\left(\frac{x_i}{scale_g}\right)
    $$

- Example:

  ```python
  cache shape:        [128, 16, 1, 384]   # 4D [blockNum, blockSize, 1, headDim], num_slots=2048, headDim≥kvCacheCol(=323)
  x shape:            [1024, 256]
  slot_mapping shape: [1024]
  quantGroupSize = 64
  quantMode = 1
  roundScale = true
  xScale = 1.0
  cache out shape:    [128, 16, 1, 384]   (in-place update)
  ```

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnKvCompressEpilogGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor that contains the operator computation flow, and then call the "aclnnKvCompressEpilog" API to execute the computation.

```cpp
aclnnStatus aclnnKvCompressEpilogGetWorkspaceSize(
  aclTensor       *cacheRef,
  const aclTensor *x,
  const aclTensor *slotMapping,
  int64_t          quantGroupSize,
  int64_t          quantMode,
  bool             roundScale,
  float            xScale,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnKvCompressEpilog(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnKvCompressEpilogGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 200px">
  <col style="width: 110px">
  <col style="width: 320px">
  <col style="width: 320px">
  <col style="width: 180px">
  <col style="width: 100px">
  <col style="width: 120px">
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
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>cacheRef (aclTensor*)</td>
      <td>Input/Output</td>
      <td>Indicates the KV Cache vector cache of the current layer, updated in place, corresponding to the quantization target in the formula.</td>
      <td><li>Empty tensors are not supported.</li><li>Serves as both input and output.</li><li>Only the four-dimension shape [blockNum, blockSize, 1, headDim] is supported, where dim2 is fixed to 1.</li></td>
      <td>UINT8</td>
      <td>ND</td>
      <td>[blockNum, blockSize, 1, headDim]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>x (const aclTensor*)</td>
      <td>Input</td>
      <td>Activation input to be quantized, corresponding to x in the formula.</td>
      <td><li>Empty tensors are not supported.</li><li>The last dimension d must meet d % 64 == 0 and 64 < d ≤ 8192.</li></td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>[bs, d]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>slotMapping (const aclTensor*)</td>
      <td>Input</td>
      <td>Index mapping from token to cache slot. A value of -1 indicates that the token is skipped.</td>
      <td><li>Empty tensors are not supported.</li><li>The dimension must equal the dimension of x minus 1.</li><li>Value range of elements: [-1, num_slots - 1].</li></td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>[bs]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>quantGroupSize (int64_t)</td>
      <td>Input</td>
      <td>Quantization group size.</td>
      <td>The default value is 64. When quantMode=2, only 16/32/64 are supported, and (d-64) % quantGroupSize == 0 must be satisfied.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode (int64_t)</td>
      <td>Input</td>
      <td>Quantization mode.</td>
      <td>Supported enumeration values are 0, 1, and 2, where 0 indicates group quantization (scale stored as bfloat16), 1 indicates group quantization (scale stored as float8_e8m0), and 2 indicates hifloat8 static quantization for the rope part + FLOAT4_E2M1 dynamic quantization for the nope part (scale stored as bfloat16). The default value is 1.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>roundScale (bool)</td>
      <td>Input</td>
      <td>Indicates whether the scale of each group is rounded up to a power of 2 in group mode.</td>
      <td>The default value is true. This parameter does not take effect when quantMode=2.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>xScale (float)</td>
      <td>Input</td>
      <td>Scale factor for static hifloat8 quantization of the rope part when quantMode=2. Reserved and unused when quantMode=0/1.</td>
      <td>The default value is 1.0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be allocated on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs the basic checks below. Success does not mean all shape, data type, and attribute constraints have been checked; subsequent tiling failures may be reported by the second-phase API:

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
      <td>cacheRef, x, or slotMapping contains a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>cacheRef, x, or slotMapping is an empty tensor.</td>
    </tr>
  </tbody></table>

## aclnnKvCompressEpilog

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
    <tr><td>workspaceSize</td><td>Input</td><td>Size of the workspace requested on the Device, obtained by calling the first-phase API aclnnKvCompressEpilogGetWorkspaceSize.</td></tr>
    <tr><td>executor</td><td>Input</td><td>op executor, which contains the operator computation flow.</td></tr>
    <tr><td>stream</td><td>Input</td><td>Stream for task execution.</td></tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status value. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  Violating the constraints below may cause execution preparation or tiling checks to fail with the following internal errors. The exact code depends on the failing stage and runtime version; 161002 is not guaranteed. Consult runtime logs rather than interpreting an internal error code alone as a parameter error.

  | Return Value | Error Code | Description |
  | --- | --- | --- |
  | ACLNN_ERR_INNER_NULLPTR | 561103 | Internal nullptr exception; failures of constraints such as d or quantMode may be reported through this error. |
  | ACLNN_ERR_INNER | 561000 | Internal API exception; execution preparation failures such as unsupported data type combinations may be reported through this error. |

## Constraints

- Deterministic computation: aclnnKvCompressEpilog defaults to a deterministic implementation.
- **cache supports only the 4D shape** `[blockNum, blockSize, 1, headDim]` (num_slots = blockNum × blockSize). The second-to-last dimension is fixed to 1, and other dimension counts are not supported; non-contiguous tensors are supported only along the blockNum dimension.
- **headDim constraint**: The last dimension of cache, headDim, must be ≥ the number of bytes written per row, kvCacheCol = alignment(concatCol). concatCol: (d-64)+128+⌈(d-64)/64⌉×scaleBytes for quantMode=0/1 (mode0=2, mode1=1); 64+(d-64)/2+((d-64)/quantGroupSize)×2 for quantMode=2. Alignment: no padding for quantMode=1, and 32B alignment for quantMode=0/2. Example: d=256, quantMode=1 → kvCacheCol=323 → headDim ≥ 323.
- The dimension of slotMapping must be the dimension of x minus 1, indicating that slotMapping is obtained by flattening all dimensions of x except the last one.
- The last dimension of x (d axis) must satisfy d % 64 == 0 and 64 < d ≤ 8192. Quantization is performed group by group, with each group containing 64 elements.
- When quantMode=2, quantGroupSize supports only 16/32/64, the length of the nope segment (d-64) must be divisible by quantGroupSize, and x must be bfloat16.
- Tokens with a value of -1 in slotMapping are skipped and not processed. Other valid elements must be in the range [0, num_slots - 1] and must not be duplicated; otherwise, result correctness is not guaranteed.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_kv_compress_epilog.h"

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
    // Boilerplate: resource initialization
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
    // Call aclrtMemcpy to copy host data to the device memory.
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
    // 1.(boilerplate)Initialize device/stream, refer to the acl API manual.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2.Construct the inputs and outputs. Custom construction is required based on the API.
    std::vector<int64_t> cacheShape = {128, 16, 1, 384};  // 4D KV Cache [blockNum,blockSize,1,headDim], num_slots=2048
    std::vector<int64_t> xShape = {1024, 256};      // Activation to be quantized
    std::vector<int64_t> slotShape = {1024};        // slot index mapping
    void* cacheDeviceAddr = nullptr;
    void* xDeviceAddr = nullptr;
    void* slotDeviceAddr = nullptr;
    aclTensor* cache = nullptr;
    aclTensor* x = nullptr;
    aclTensor* slotMapping = nullptr;
    std::vector<uint8_t> cacheHostData(GetShapeSize(cacheShape), 0);
    std::vector<uint16_t> xHostData(GetShapeSize(xShape), 0);  // bfloat16 carried as uint16
    std::vector<int32_t> slotHostData(GetShapeSize(slotShape), 0);
    // Create the cache aclTensor (in-place input/output).
    ret = CreateAclTensor(cacheHostData, cacheShape, &cacheDeviceAddr, aclDataType::ACL_UINT8, &cache);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the slotMapping aclTensor.
    ret = CreateAclTensor(slotHostData, slotShape, &slotDeviceAddr, aclDataType::ACL_INT32, &slotMapping);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Attributes
    int64_t quantGroupSize = 64;
    int64_t quantMode = 1;   // group quantization, scale is stored as float8_e8m0
    bool roundScale = true;
    float xScale = 1.0f;

    // 3. Call the CANN operator library API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnKvCompressEpilog.
    ret = aclnnKvCompressEpilogGetWorkspaceSize(cache, x, slotMapping, quantGroupSize, quantMode, roundScale, xScale,
                                                &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnKvCompressEpilogGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnKvCompressEpilog.
    ret = aclnnKvCompressEpilog(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnKvCompressEpilog failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Synchronously wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. cache is updated in place, and the result has been written back to cacheDeviceAddr. Copy it back to the host as needed for viewing.

    // 6. Release aclTensor. Modify it based on the definition of the specific API.
    aclDestroyTensor(cache);
    aclDestroyTensor(x);
    aclDestroyTensor(slotMapping);

    // 7. Release device resources.
    aclrtFree(cacheDeviceAddr);
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
