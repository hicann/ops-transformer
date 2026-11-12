# aclnnApplyRotaryPosEmbGrad

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:41:59.379Z -->

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

- Description: Performs backward computation for the dual-path rotary position embedding [aclnnApplyRotaryPosEmb](../../apply_rotary_pos_emb/docs/aclnnApplyRotaryPosEmb_en.md). Computes the rope backward gradients of query and key, fused into a single kernel call.

- Formula:

    In the forward computation of the rotary position embedding, let `dims` be the list of broadcast axes, that is, the axes with value 1 in `cos`/`sin` and a corresponding dimension greater than 1 in `gradQueryEmbed`/`gradKeyEmbed` (including the N axis and the optional B axis in BSND and SBND layouts). The formula for half mode is as follows:

    $$
    grad\_q_1, grad\_q_2 = chunk(grad\_query\_embed, chunks=2, dim=-1)
    $$

    $$
    grad\_k_1, grad\_k_2 = chunk(grad\_key\_embed, chunks=2, dim=-1)
    $$

    $$
    cos_1, cos_2 = chunk(cos, chunks=2, dim=-1)
    $$

    $$
    sin_1, sin_2 = chunk(sin, chunks=2, dim=-1)
    $$

    $$
    query\_rotate = cat((-query_2, query_1), dim=-1)
    $$

    $$
    key\_rotate = cat((-key_2, key_1), dim=-1)
    $$

    $$
    grad\_query = cat(cos_1 * grad\_q_1 + sin_2 * grad\_q_2, cos_2 * grad\_q_2 - sin_1 * grad\_q_1, dim=-1)
    $$

    $$
    grad\_key = cat(cos_1 * grad\_k_1 + sin_2 * grad\_k_2, cos_2 * grad\_k_2 - sin_1 * grad\_k_1, dim=-1)
    $$

    $$
    grad\_cos = sum(grad\_query\_embed * query + grad\_key\_embed * key, dims)
    $$

    $$
    grad\_sin = sum(grad\_query\_embed * query\_rotate + grad\_key\_embed * key\_rotate, dims)
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnApplyRotaryPosEmbGradGetWorkspaceSize" first to obtain the input parameters and calculate the required workspace size according to the process, and then call "aclnnApplyRotaryPosEmbGrad" to execute the computation.

```c++
aclnnStatus aclnnApplyRotaryPosEmbGradGetWorkspaceSize(
    const aclTensor *gradQueryEmbed,
    const aclTensor *gradKeyEmbed,
    const aclTensor *cos,
    const aclTensor *sin,
    const aclTensor *queryOptional,
    const aclTensor *keyOptional,
    char            *rotaryModeOptional,
    int64_t          layout,
    const aclTensor *gradQueryOut,
    const aclTensor *gradKeyOut,
    const aclTensor *gradCosOut,
    const aclTensor *gradSinOut,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```c++
aclnnStatus aclnnApplyRotaryPosEmbGrad(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnApplyRotaryPosEmbGradGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1532px"><colgroup>
  <col style="width: 162px">
  <col style="width: 121px">
  <col style="width: 403px">
  <col style="width: 169px">
  <col style="width: 275px">
  <col style="width: 118px">
  <col style="width: 138px">
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
      <td>gradQueryEmbed</td>
      <td>Input</td>
      <td>Derivative of the forward output query, corresponding to grad_q_embed in the formula.</td>
      <td>Does not support empty tensor.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>4 (layout: 1 or 2) or 3 (layout: 4)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradKeyEmbed</td>
      <td>Input</td>
      <td>Derivative of the forward output key, corresponding to grad_k_embed in the formula.</td>
      <td>Data type and dimensions must be the same as those of gradQueryEmbed. Does not support empty tensors.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>4 (layout is 1 or 2) or 3 (layout is 4)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cos</td>
      <td>Input</td>
      <td>Forward computation input cos.</td>
      <td>Data type and dimensions must be the same as those of gradQueryEmbed. The N dimension must be equal to 1. Does not support empty tensors.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>4 (layout is 1 or 2) or 3 (layout is 4)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sin</td>
      <td>Input</td>
      <td>Forward computation input sin.</td>
      <td>The data type and dimensions must be consistent with those of gradQueryEmbed. The N dimension must be equal to 1. Empty tensors are not supported.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>4 (layout is 1 or 2) or 3 (layout is 4)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>queryOptional</td>
      <td>Optional input</td>
      <td>Query of the forward computation. If the pointer is null, gradCosOut and gradSinOut are not computed.</td>
      <td>Must be consistent with gradQueryEmbed in data type and dimensions, and must be passed in together with keyOptional or not passed in.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>4 (layout is 1 or 2) or 3 (layout is 4)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>keyOptional</td>
      <td>Optional input</td>
      <td>Forward computation input key. If it is a null pointer, gradCosOut and gradSinOut are not computed.</td>
      <td>The data type and dimensions must be the same as those of gradQueryEmbed. Must be passed in together with queryOptional or omitted together.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>4 (layout is 1 or 2) or 3 (layout is 4)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rotaryModeOptional</td>
      <td>Input</td>
      <td>Rotation mode.</td>
      <td>Only supports "half".</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout</td>
      <td>Input</td>
      <td>Layout format of the input tensor.</td>
      <td>Value range: 1-BSND, 2-SBND, 4-TND, 3-BNSD (reserved).</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradQueryOut</td>
      <td>Output</td>
      <td>Derivative of the forward computation input query.</td>
      <td>Data type and dimensions must be consistent with those of gradQueryEmbed.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>4 (layout: 1 or 2) or 3 (layout: 4)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>gradKeyOut</td>
      <td>Output</td>
      <td>Derivative of the key input in forward computation.</td>
      <td>Data type and dimensions are consistent with gradQueryEmbed.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>4 (layout is 1 or 2) or 3 (layout is 4)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>gradCosOut</td>
      <td>Output</td>
      <td>Derivative of the forward computation input cos. Valid only when queryOptional and keyOptional are not empty.</td>
      <td>Data type and dimensions are consistent with those of gradQueryEmbed.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>4 (layout is 1 or 2) or 3 (layout is 4)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>gradSinOut</td>
      <td>Output</td>
      <td>Derivative of the forward computation input sin. Valid only when queryOptional and keyOptional are not empty.</td>
      <td>The data type and dimensions must be consistent with those of gradQueryEmbed.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>4 (layout is 1 or 2) or 3 (layout is 4)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be allocated on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, including the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 288px">
  <col style="width: 125px">
  <col style="width: 742px">
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
      <td>The required input gradQueryEmbed, gradKeyEmbed, cos, or sin, or the required output gradQueryOut or gradKeyOut, is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The input and output data types and formats are not within the supported range, the shape does not meet the validation conditions, the dimensions are out of the supported range, queryOptional and keyOptional are not both passed in or both omitted, or rotaryMode/layout is not a supported value.</td>
    </tr>
  </tbody>
  </table>

## aclnnApplyRotaryPosEmbGrad

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 173px">
  <col style="width: 133px">
  <col style="width: 849px">
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
      <td>Workspace memory address requested on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size requested on the device, obtained by the first-phase API aclnnApplyRotaryPosEmbGradGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnApplyRotaryPosEmbGrad defaults to a deterministic implementation.
- Constraints on each parameter are as follows:
  - Input and output tensors only support 3D or 4D: 4D when layout is 1 or 2, and 3D when layout is 4.
  - The dtype of input and output tensors must be the same.
  - Input and output tensors do not support empty tensors (each dimension must be greater than 0).
  - The layout of input and output tensors must be the same.
  - The D dimension of input and output tensors must be the same. In half mode, it must be ≤ 1024 and divisible by 2.
  - The shape of `gradQueryEmbed` and `gradQueryOut` must be the same, and the shape of `gradKeyEmbed` and `gradKeyOut` must be the same.
  - For any `layout`, all dimensions except N must be the same between `gradQueryEmbed` and `gradKeyEmbed`.
  - The N dimension of `cos` and `sin` must be equal to 1. If `layout` is 1 (BSND) or 2 (SBND), the B dimension of `cos` and `sin` can be 1 or consistent with the B dimension of `gradQueryEmbed`. If `layout` is 4 (TND), the T dimension of `cos` and `sin` must be consistent with the T dimension of `gradQueryEmbed`. All dimensions except N (and the optionally broadcast B dimension under BSND and SBND layouts) must be consistent with those of `gradQueryEmbed`.
  - The shapes of `cos`, `sin`, `gradCosOut`, and `gradSinOut` must be the same.
  - The dimensions of `query` must be consistent with `gradQueryEmbed`, and the dimensions of `key` must be consistent with `gradKeyEmbed`. `query` and `key` must be both passed in or both omitted.
  - `rotaryModeOptional` only supports "half".
  - `layout` only supports {1, 2, 4}, corresponding to {BSND, SBND, TND}.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_apply_rotary_pos_emb_grad.h"
#include <iostream>
#include <vector>

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

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // Boilerplate for resource initialization.
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
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate device memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy host data to device memory.
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

int main()
{
    // 1. Boilerplate for device/stream initialization. For details, see the acl API manual.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as needed.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the input and output according to the API definition.
    std::vector<int64_t> gradQEmbedShape = {1, 1, 1, 128};
    std::vector<int64_t> gradKEmbedShape = {1, 1, 1, 128};
    std::vector<int64_t> cosShape = {1, 1, 1, 128};
    std::vector<int64_t> sinShape = {1, 1, 1, 128};
    std::vector<int64_t> queryShape = {1, 1, 1, 128};
    std::vector<int64_t> keyShape = {1, 1, 1, 128};
    std::vector<int64_t> gradQueryOutShape = {1, 1, 1, 128};
    std::vector<int64_t> gradKeyOutShape = {1, 1, 1, 128};
    std::vector<int64_t> gradCosOutShape = {1, 1, 1, 128};
    std::vector<int64_t> gradSinOutShape = {1, 1, 1, 128};
    int64_t layout = 1;
    const char* rotaryModeOptional = "half";

    void* gradQEmbedDeviceAddr = nullptr;
    void* gradKEmbedDeviceAddr = nullptr;
    void* cosDeviceAddr = nullptr;
    void* sinDeviceAddr = nullptr;
    void* queryDeviceAddr = nullptr;
    void* keyDeviceAddr = nullptr;
    void* gradQueryOutDeviceAddr = nullptr;
    void* gradKeyOutDeviceAddr = nullptr;
    void* gradCosOutDeviceAddr = nullptr;
    void* gradSinOutDeviceAddr = nullptr;
    aclTensor* gradQueryEmbed = nullptr;
    aclTensor* gradKeyEmbed = nullptr;
    aclTensor* cos = nullptr;
    aclTensor* sin = nullptr;
    aclTensor* query = nullptr;
    aclTensor* key = nullptr;
    aclTensor* gradQueryOut = nullptr;
    aclTensor* gradKeyOut = nullptr;
    aclTensor* gradCosOut = nullptr;
    aclTensor* gradSinOut = nullptr;

    std::vector<float> gradQEmbedHostData = {
        74,  54, 84, 125, 23,  78,  37,  72,  27, 98,  34,  107, 29,  23,  54,  60, 70,  49,  119, 54,  29,  54,
        41,  99, 27, 62,  5,   46,  108, 39,  24, 123, 33,  82,  6,   40,  88,  24, 6,   116, 38,  119, 110, 5,
        30,  79, 87, 18,  29,  100, 90,  24,  21, 93,  63,  68,  34,  112, 119, 48, 74,  43,  85,  64,  14,  49,
        128, 59, 18, 37,  123, 76,  14,  63,  10, 39,  107, 124, 79,  16,  17,  76, 80,  47,  90,  41,  58,  82,
        75,  80, 69, 37,  74,  36,  54,  26,  32, 54,  13,  100, 105, 15,  13,  69, 122, 26,  94,  59,  29,  14,
        60,  8,  24, 17,  45,  33,  107, 122, 63, 111, 75,  128, 68,  31,  105, 6,  82,  99};
    std::vector<float> gradKEmbedHostData = {
        74,  54, 84, 125, 23,  78,  37,  72,  27, 98,  34,  107, 29,  23,  54,  60, 70,  49,  119, 54,  29,  54,
        41,  99, 27, 62,  5,   46,  108, 39,  24, 123, 33,  82,  6,   40,  88,  24, 6,   116, 38,  119, 110, 5,
        30,  79, 87, 18,  29,  100, 90,  24,  21, 93,  63,  68,  34,  112, 119, 48, 74,  43,  85,  64,  14,  49,
        128, 59, 18, 37,  123, 76,  14,  63,  10, 39,  107, 124, 79,  16,  17,  76, 80,  47,  90,  41,  58,  82,
        75,  80, 69, 37,  74,  36,  54,  26,  32, 54,  13,  100, 105, 15,  13,  69, 122, 26,  94,  59,  29,  14,
        60,  8,  24, 17,  45,  33,  107, 122, 63, 111, 75,  128, 68,  31,  105, 6,  82,  99};
    std::vector<float> cosHostData = {
        41, 37,  17, 25, 49, 25,  22,  24,  110, 120, 107, 3,   82, 66,  75,  86,  85,  115, 110, 56,  52,  39,
        86, 23,  36, 71, 20, 73,  113, 25,  114, 56,  125, 80,  95, 82,  31,  63,  99,  62,  23,  55,  30,  99,
        42, 121, 15, 24, 97, 87,  81,  67,  43,  21,  13,  9,   33, 29,  117, 10,  114, 61,  98,  15,  78,  108,
        48, 97,  1,  3,  78, 109, 57,  46,  47,  56,  50,  66,  81, 77,  17,  128, 68,  121, 47,  91,  114, 125,
        51, 108, 31, 15, 47, 78,  109, 115, 113, 26,  53,  97,  1,  111, 103, 58,  106, 68,  11,  104, 22,  79,
        61, 127, 86, 39, 33, 123, 102, 39,  64,  41,  119, 120, 61, 29,  94,  68,  36,  12};
    std::vector<float> sinHostData = {
        46, 56,  56,  101, 66,  10,  96,  16, 86,  57,  102, 66,  12,  105, 76, 58,  90,  6,   79, 128, 126, 82,
        41, 3,   45,  7,   66,  4,   46,  22, 31,  26,  37,  63,  97,  84,  91, 90,  47,  77,  90, 34,  41,  83,
        91, 108, 120, 13,  90,  32,  85,  37, 119, 31,  51,  82,  122, 125, 7,  116, 121, 108, 38, 56,  100, 20,
        97, 119, 10,  4,   53,  13,  46,  82, 103, 119, 124, 80,  23,  67,  78, 56,  119, 122, 40, 58,  128, 27,
        30, 52,  71,  42,  123, 69,  4,   5,  116, 97,  38,  107, 8,   4,   65, 120, 40,  22,  60, 44,  48,  66,
        68, 125, 4,   93,  112, 112, 113, 90, 94,  23,  104, 39,  85,  84,  64, 128, 96,  119};
    std::vector<float> queryHostData = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
        23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
        45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
        67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88,
        89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
        109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128};
    std::vector<float> keyHostData = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
        23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
        45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
        67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88,
        89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
        109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128};
    std::vector<float> gradQueryOutHostData(128, 0.0f);
    std::vector<float> gradKeyOutHostData(128, 0.0f);

    // Create the gradQueryEmbed aclTensor.
    ret = CreateAclTensor(gradQEmbedHostData, gradQEmbedShape, &gradQEmbedDeviceAddr, aclDataType::ACL_FLOAT, &gradQueryEmbed);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the gradKeyEmbed aclTensor.
    ret = CreateAclTensor(gradKEmbedHostData, gradKEmbedShape, &gradKEmbedDeviceAddr, aclDataType::ACL_FLOAT, &gradKeyEmbed);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the cos aclTensor.
    ret = CreateAclTensor(cosHostData, cosShape, &cosDeviceAddr, aclDataType::ACL_FLOAT, &cos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the sin aclTensor.
    ret = CreateAclTensor(sinHostData, sinShape, &sinDeviceAddr, aclDataType::ACL_FLOAT, &sin);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the query aclTensor (used to compute grad_cos/grad_sin).
    ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT, &query);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the key aclTensor (used to compute grad_cos/grad_sin).
    ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT, &key);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the gradQueryOut aclTensor.
    ret = CreateAclTensor(gradQueryOutHostData, gradQueryOutShape, &gradQueryOutDeviceAddr, aclDataType::ACL_FLOAT, &gradQueryOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the gradKeyOut aclTensor.
    ret = CreateAclTensor(gradKeyOutHostData, gradKeyOutShape, &gradKeyOutDeviceAddr, aclDataType::ACL_FLOAT, &gradKeyOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the gradCosOut aclTensor.
    ret = CreateAclTensor(std::vector<float>(128, 0.0f), gradCosOutShape, &gradCosOutDeviceAddr, aclDataType::ACL_FLOAT, &gradCosOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the gradSinOut aclTensor.
    ret = CreateAclTensor(std::vector<float>(128, 0.0f), gradSinOutShape, &gradSinOutDeviceAddr, aclDataType::ACL_FLOAT, &gradSinOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. Call the CANN operator library API. Replace it with the specific API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnApplyRotaryPosEmbGrad.
    ret = aclnnApplyRotaryPosEmbGradGetWorkspaceSize(gradQueryEmbed, gradKeyEmbed, cos, sin, query, key,
                                                      const_cast<char*>(rotaryModeOptional), layout,
                                                      gradQueryOut, gradKeyOut, gradCosOut, gradSinOut,
                                                      &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnApplyRotaryPosEmbGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnApplyRotaryPosEmbGrad.
    ret = aclnnApplyRotaryPosEmbGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyRotaryPosEmbGrad failed. ERROR: %d\n", ret); return ret);
    // 4. Fixed boilerplate. Wait for the task execution to complete synchronously.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output values and copy the result from the device memory to the host memory.
    auto size = GetShapeSize(gradQueryOutShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradQueryOutDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Destroy the aclTensor and aclScalar.
    aclDestroyTensor(gradQueryEmbed);
    aclDestroyTensor(gradKeyEmbed);
    aclDestroyTensor(cos);
    aclDestroyTensor(sin);
    aclDestroyTensor(query);
    aclDestroyTensor(key);
    aclDestroyTensor(gradQueryOut);
    aclDestroyTensor(gradKeyOut);
    aclDestroyTensor(gradCosOut);
    aclDestroyTensor(gradSinOut);

    // 7. Release the device resources.
    aclrtFree(gradQEmbedDeviceAddr);
    aclrtFree(gradKEmbedDeviceAddr);
    aclrtFree(cosDeviceAddr);
    aclrtFree(sinDeviceAddr);
    aclrtFree(queryDeviceAddr);
    aclrtFree(keyDeviceAddr);
    aclrtFree(gradQueryOutDeviceAddr);
    aclrtFree(gradKeyOutDeviceAddr);
    aclrtFree(gradCosOutDeviceAddr);
    aclrtFree(gradSinOutDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}

```
