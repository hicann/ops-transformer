# aclnnRotaryPositionEmbedding

[📄 View Source Code](https://gitcode.com/cann/ops-transformer/tree/master/posembedding/rotary_position_embedding)

## Supported Products

| Product                                                      | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>     |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term> |    √     |
| <term>Atlas 200I/500 A2 inference product</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function

- Description: Perform single-channel rotary position encoding computation.
- Formula:

    - <term>Atlas A3 training products/Atlas A3 inference products</term>, <term>Atlas A2 training products/Atlas A2 inference products</term>:

    (1) Half mode (mode equals 0):

    $$
    x1 = x[..., : x.shape[-1] // 2]
    $$

    $$
    x2 = x[..., x.shape[-1] // 2 :]
    $$

    $$
    x\_rotate = torch.cat((-x2, x1), dim=-1)
    $$

    $$
    y = x * cos + x\_rotate * sin
    $$

    (2) Interleave mode (mode equals 1):

    $$
    x1 = x[..., ::2].view(-1, 1)
    $$

    $$
    x2 = x[..., 1::2].view(-1, 1)
    $$

    $$
    x\_rotate = torch.cat((-x2, x1), dim=-1).view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
    $$

    $$
    y = x * cos + x\_rotate * sin
    $$
    
    (3) Quarter mode (mode equals 2):

    $$
    x1 = x[..., : x.shape[-1] // 4]
    $$

    $$
    x2 = x[..., x.shape[-1] // 4 : x.shape[-1] // 2]
    $$

    $$
    x3 = x[..., x.shape[-1] // 2 : x.shape[-1] // 4 * 3]
    $$

    $$
    x4 = x[..., x.shape[-1] // 4 * 3 :]
    $$

    $$
    x\_rotate = torch.cat((-x2, x1, -x4, x3), dim=-1)
    $$

    $$
    y = x * cos + x\_rotate * sin
    $$

    (4) interleave-half mode (mode equals 3), this mode first extracts the odd-positioned inputs to the first half and the even-positioned inputs to the second half, then performs the half processing:

    $$
    x1 = x[..., ::2]
    $$

    $$
    x2 = x[..., 1::2]
    $$

    $$
    x\_part1 = torch.cat((x1, x2), dim=-1)
    $$

    $$
    x\_part2 = torch.cat((-x2, x1), dim=-1)
    $$

    $$
    y = x\_part1 * cos + x\_part2 * sin
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md)calls. You must first call `aclnnRotaryPositionEmbeddingGetWorkspaceSize` to obtain the input parameters and calculate the required workspace size according to the process, and then call `aclnnRotaryPositionEmbedding` to perform the computation.

```c++
aclnnStatus aclnnRotaryPositionEmbeddingGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *cos,
    const aclTensor *sin,
    int64_t          mode,
    aclTensor       *out,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```c++
aclnnStatus aclnnRotaryPositionEmbedding(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)

```

## aclnnRotaryPositionEmbeddingGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1461px"><colgroup>
  <col style="width: 162px">
  <col style="width: 121px">
  <col style="width: 332px">
  <col style="width: 169px">
  <col style="width: 275px">
  <col style="width: 118px">
  <col style="width: 138px">
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
      <th>Shape</th>
      <th>Non-contiguous Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>The tensor to be rotated for positional encoding, denoted as x in the formula.</td>
      <td>-</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>3 or 4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cos</td>
      <td>Input</td>
      <td>Position encoding tensor, the cos in the formula.</td>
      <td>Consistent with the data type of x.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>3 or 4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sin</td>
      <td>Input</td>
      <td>Position encoding tensor, the sin in the formula.</td>
      <td>Consistent with the data type of x.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>3 or 4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mode</td>
      <td>Input</td>
      <td>Rotation mode.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Rotation position encoding calculation result, y in the formula.</td>
      <td>Consistent with the data type of x.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>4</td>
      <td>x</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the size of the workspace that needs to be applied for on the device side.</td>
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
  </tbody>
  </table>

  - Parameter mode constraints:
    - <term>Atlas A3 training products/Atlas A3 inference products</term>, <term>Atlas A2 training products/Atlas A2 inference products</term>: 0=half, 1=interleave.
- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.

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
      <td>The passed x, cos, sin, or out is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data types and formats of the input x, cos, sin, and out are not within the supported range.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="2">561002</td>
      <td>The shapes of the input x, cos, sin, and out do not match.</td>
    </tr>
    <tr>
      <td>The passed mode parameter is not within the range of 0, 1, 2, 3. </td>
    </tr>
  </tbody>
  </table>

## aclnnRotaryPositionEmbedding

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 173px">
  <col style="width: 133px">
  <col style="width: 849px">
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
      <td>The memory address of the workspace applied for on the device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>The size of the workspace requested on the device side, obtained by the first-phase aclnnRotaryPositionEmbeddingGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, which includes the computation process of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic Computation:
  - `aclnnRotaryPositionEmbedding` defaults to deterministic implementation.

  - <term>Atlas A3 training products/Atlas A3 inference products</term>, <term>Atlas A2 training products/Atlas A2 inference products</term>:
    
    The input tensor `x` supports BNSD, BSND, SBND, and TND layouts.
    The D-dimensional sizes of the input tensor `x`, `cos`, `sin`, and output tensor `y` must be the same, satisfy D < 896, and must be a multiple of 2.
    The shapes of the input tensor `x` and the output tensor `y` must be exactly the same.
    The shapes of the input tensors `cos` and `sin` must be exactly the same.
    - Half mode:
      - B, N < 1000.
      - When `x` is BNSD, `cos` and `sin` support 11SD, B1SD, and BNSD.
        `- When (D/2) % (32/inputDtypeSize) == 0, it must satisfy B * N <= S * 8.`
        `- When (D/2) % (32/inputDtypeSize) != 0, it must satisfy B * N * 2 <= (S + coreNum -1) / coreNum or D >= 80.`
      - When `x` is BSND, `cos` and `sin` support 1S1D, BS1D, BSND.
      - When `x` is SBND, `cos` and `sin` support S11D, SB1D, SBND.
      - When `x` is TND, `cos` and `sin` support T1D and TND.
    - Interleave mode:
      - B * N < 1000 (N<1000 when x is TND)
      - When `x` is BNSD, `cos` and `sin` support 11SD.
      - When `x` is BSND, `cos` and `sin` support 1S1D.
      - When `x` is SBND, `cos` and `sin` support S11D.
      - When `x` is TND, `cos` and `sin` support T1D.

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_rotary_position_embedding.h"
#include <iostream>
#include <vector>

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
    // Fixed writing, resource initialization.
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
    // 1. Fixed writing, device/stream initialization, refer to the acl API manual.
    // Fill in the deviceId according to your actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Process according to your own needs.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct input and output according to the API definition.
    std::vector<int64_t> xShape = {1, 1, 1, 128};
    std::vector<int64_t> cosShape = {1, 1, 1, 128};
    std::vector<int64_t> sinShape = {1, 1, 1, 128};
    std::vector<int64_t> outShape = {1, 1, 1, 128};
    int64_t mode = 1;

    void* xDeviceAddr = nullptr;
    void* cosDeviceAddr = nullptr;
    void* sinDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* cos = nullptr;
    aclTensor* sin = nullptr;
    aclTensor* out = nullptr;

    std::vector<float> xHostData = {74, 54, 84, 125, 23, 78, 37, 72, 27, 98, 34, 107, 29, 23, 54, 60, 70, 49,
                                    119, 54, 29, 54, 41, 99, 27, 62, 5, 46, 108, 39, 24, 123, 33, 82, 6, 40, 88,
                                    24, 6, 116, 38, 119, 110, 5, 30, 79, 87, 18, 29, 100, 90, 24, 21, 93, 63, 68,
                                    34, 112, 119, 48, 74, 43, 85, 64, 14, 49, 128, 59, 18, 37, 123, 76, 14, 63, 10,
                                    39, 107, 124, 79, 16, 17, 76, 80, 47, 90, 41, 58, 82, 75, 80, 69, 37, 74, 36, 54,
                                    26, 32, 54, 13, 100, 105, 15, 13, 69, 122, 26, 94, 59, 29, 14, 60, 8, 24, 17, 45,
                                    33, 107, 122, 63, 111, 75, 128, 68, 31, 105, 6, 82, 99};
    std::vector<float> cosHostData = {41, 37, 17, 25, 49, 25, 22, 24, 110, 120, 107, 3, 82, 66, 75, 86, 85, 115, 110, 56, 52,
                                      39, 86, 23, 36, 71, 20, 73, 113, 25, 114, 56, 125, 80, 95, 82, 31, 63, 99, 62, 23, 55, 30,
                                      99, 42, 121, 15, 24, 97, 87, 81, 67, 43, 21, 13, 9, 33, 29, 117, 10, 114, 61, 98, 15, 78,
                                      108, 48, 97, 1, 3, 78, 109, 57, 46, 47, 56, 50, 66, 81, 77, 17, 128, 68, 121, 47, 91, 114,
                                      125, 51, 108, 31, 15, 47, 78, 109, 115, 113, 26, 53, 97, 1, 111, 103, 58, 106, 68, 11,
                                      104, 22, 79, 61, 127, 86, 39, 33, 123, 102, 39, 64, 41, 119, 120, 61, 29, 94, 68, 36, 12};
    std::vector<float> sinHostData = {46, 56, 56, 101, 66, 10, 96, 16, 86, 57, 102, 66, 12, 105, 76, 58, 90, 6, 79, 128, 126,
                                      82, 41, 3, 45, 7, 66, 4, 46, 22, 31, 26, 37, 63, 97, 84, 91, 90, 47, 77, 90, 34, 41, 83,
                                      91, 108, 120, 13, 90, 32, 85, 37, 119, 31, 51, 82, 122, 125, 7, 116, 121, 108, 38, 56,
                                      100, 20, 97, 119, 10, 4, 53, 13, 46, 82, 103, 119, 124, 80, 23, 67, 78, 56, 119, 122, 40,
                                      58, 128, 27, 30, 52, 71, 42, 123, 69, 4, 5, 116, 97, 38, 107, 8, 4, 65, 120, 40, 22, 60,
                                      44, 48, 66, 68, 125, 4, 93, 112, 112, 113, 90, 94, 23, 104, 39, 85, 84, 64, 128, 96, 119};
    std::vector<float> outHostData(128, 0);                                      

    // Create x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create cos aclTensor.
    ret = CreateAclTensor(cosHostData, cosShape, &cosDeviceAddr, aclDataType::ACL_FLOAT, &cos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create sin aclTensor.
    ret = CreateAclTensor(sinHostData, sinShape, &sinDeviceAddr, aclDataType::ACL_FLOAT, &sin);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);    

    // 3. Call the CANN operator library API. Change it to the specific API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnRotaryPositionEmbedding.
    ret = aclnnRotaryPositionEmbeddingGetWorkspaceSize(x, cos, sin, mode, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRotaryPositionEmbeddingGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated from the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnRotaryPositionEmbedding.
    ret = aclnnRotaryPositionEmbedding(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRotaryPositionEmbedding failed. ERROR: %d\n", ret); return ret);
    // 4. (Fixed writing) Synchronize the stream and wait for task completion.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify the code based on the API definition.
    aclDestroyTensor(x);
    aclDestroyTensor(cos);
    aclDestroyTensor(sin);
    aclDestroyTensor(out);

    // 7. Release device resources.
    aclrtFree(xDeviceAddr);
    aclrtFree(cosDeviceAddr);
    aclrtFree(sinDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
