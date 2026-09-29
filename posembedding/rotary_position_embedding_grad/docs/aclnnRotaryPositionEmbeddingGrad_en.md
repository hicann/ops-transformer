# aclnnRotaryPositionEmbeddingGrad

[📄 View Source Code](https://gitcode.com/cann/ops-transformer/tree/master/posembedding/rotary_position_embedding_grad)

## Supported Products

| Product                                                        |  Supported  |
| :----------------------------------------------------------- |:-------:|
| Ascend 950PR/Ascend 950DT                            |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference products</term>                            |    ×    |
| <term>Atlas training products</term>                             |    ×    |

## Function

- Function: Performs the backward computation of the single-channel rotary position embedding [aclnnRotaryPositionEmbedding](../../rotary_position_embedding/docs/aclnnRotaryPositionEmbedding_en.md).
- Formula:
  
    If the broadcast axis list in the forward computation of the rotary position embedding is `dims`, the calculation formula is as follows:

    - <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>:

    (1) half mode (mode = 0):

    $$
    dy1, dy2 = chunk(dy, chunks=2, dim=-1)
    $$
    
    $$
    cos1, cos2 = chunk(cos, chunks=2, dim=-1)
    $$
    
    $$
    sin1, sin2 = chunk(sin, chunks=2, dim=-1)
    $$
    
    $$
    x1, x2 = chunk(x, chunks=2, dim=-1)
    $$

    $$
    dx = cat((cos1 * dy1 + sin2 * dy2, cos2 * dy2 - sin1 * dy1), dim=-1)
    $$

    $$
    dcos = sum(dy * x, dims)
    $$

    $$
    dsin = sum(dy * cat((-x2, x1), dim=-1), dims)
    $$

    (2) Interleave mode (mode = 1):

    $$
    dy1, dy2 = dy[..., :: 2], dy[..., 1 :: 2]
    $$
    
    $$
    cos1, cos2 = cos[..., :: 2], cos[..., 1 :: 2]
    $$
    
    $$
    sin1, sin2 = sin[..., :: 2], sin[..., 1 :: 2]
    $$
    
    $$
    x1, x2 = x[..., :: 2], x[..., 1 :: 2]
    $$

    $$
    dx = stack((cos1 * dy1 + sin2 * dy2, cos2 * dy2 - sin1 * dy1), dim=-1).reshape(dy.shape)
    $$

    $$
    dcos = sum(dy * x, dims)
    $$

    $$
    dsin = sum(dy * stack((-x2, x1), dim=-1).reshape(dy.shape), dims)
    $$
    
    (3) Quarter mode (mode = 2):

    $$
    dy1, dy2, dy3, dy4 = chunk(dy, chunks=4, dim=-1)
    $$
    
    $$
    cos1, cos2, cos3, cos4 = chunk(cos, chunks=4, dim=-1)
    $$
    
    $$
    sin1, sin2, sin3, sin4 = chunk(sin, chunks=4, dim=-1)
    $$
    
    $$
    x1, x2, x3, x4 = chunk(x, chunks=4, dim=-1)
    $$

    $$
    dx = cat((cos1 * dy1 + sin2 * dy2, cos2 * dy2 - sin1 * dy1, cos3 * dy3 + sin4 * dy4, cos4 * dy4 - sin3 * dy3), dim=-1)
    $$

    $$
    dcos = sum(dy * x, dims)
    $$

    $$
    dsin = sum(dy * cat((-x2, x1, -x4, x3), dim=-1), dims)
    $$

    (4) Interleave-half mode (mode = 3):

    $$
    dy1, dy2 = chunk(dy, chunks=2, dim=-1)
    $$
    
    $$
    cos1, cos2 = chunk(cos, chunks=2, dim=-1)
    $$
    
    $$
    sin1, sin2 = chunk(sin, chunks=2, dim=-1)
    $$
    
    $$
    x1, x2 = x[..., :: 2], x[..., 1 :: 2]
    $$

    $$
    dx = stack((cos1 * dy1 + sin2 * dy2, cos2 * dy2 - sin1 * dy1), dim=-1).reshape(dy.shape)
    $$

    $$
    dcos = sum(dy * cat((x1, x2), dim=-1), dims)
    $$

    $$
    dsin = sum(dy * cat((-x2, x1), dim=-1), dims)
    $$

## Function Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnRotaryPositionEmbeddingGradGetWorkspaceSize` is called to obtain the input parameters and compute the required workspace size based on the process. Then, `aclnnRotaryPositionEmbeddingGrad` is called to perform computation.

```c++
aclnnStatus aclnnRotaryPositionEmbeddingGradGetWorkspaceSize(
    const aclTensor *dy,
    const aclTensor *cos,
    const aclTensor *sin,
    const aclTensor *xOptional,
    int64_t          mode,
    const aclTensor *dxOut,
    const aclTensor *dcosOut,
    const aclTensor *dsinOut,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```c++
aclnnStatus aclnnRotaryPositionEmbeddingGrad(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnRotaryPositionEmbeddingGradGetWorkspaceSize

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
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage Notes</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>dy</td>
      <td>Input</td>
      <td>Forward output derivative of y in the positional encoding of rotation.</td>
      <td>-</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cos</td>
      <td>Input</td>
      <td>Input cosine for forward computation.</td>
      <td>Same as the data type of outputOut.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sin</td>
      <td>Input</td>
      <td>Input sin for forward computation.</td>
      <td>Same as the data type of outputOut.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>xOptional</td>
      <td>Optional input</td>
      <td>Input x for forward computation. If the pointer is null, dcosOut and dsinOut are not computed.</td>
      <td>Same as the data type of outputOut.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>4</td>
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
      <td>dxOut</td>
      <td>Output</td>
      <td>Derivative of the input x for forward computation.</td>
      <td>Same as the data type of outputOut.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>4</td>
      <td>x</td>
    </tr>
    <tr>
      <td>dcosOut</td>
      <td>Output</td>
      <td>Derivative of the input cos in forward propagation. This parameter is valid only when xOptional is not empty.</td>
      <td>Same as the data type of outputOut.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>4</td>
      <td>x</td>
    </tr>
    <tr>
      <td>dsinOut</td>
      <td>Output</td>
      <td>Derivative of the input sin in forward propagation. This parameter is valid only when xOptional is not empty.</td>
      <td>Same as the data type of outputOut.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>4</td>
      <td>x</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Size of the workspace required to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Operator executor, containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  - **Constraints**:
    - <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>: 0=half, 1=interleave.
    - Ascend 950PR/Ascend 950DT: 2=quarter, 3=interleave-half.

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
      <th>Return</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input pointers dy, cos, sin and output pointers dxOut, dcosOut, dsinOut are null.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data types and formats of the input dy, cos, sin, xOptional and output dxOut, dcosOut, dsinOut are not supported.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="2">561002</td>
      <td>The input parameter shape does not meet the conditions described in the constraints section.</td>
    </tr>
    <tr>
      <td>The input mode parameter is not within the range of 0, 1, 2, or 3. </td>
    </tr>
  </tbody>
  </table>

## aclnnRotaryPositionEmbeddingGrad

- **Parameters**

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
      <td>Address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API <code>aclnnRotaryPositionEmbeddingGradGetWorkspaceSize</code>.</td>
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

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnRotaryPositionEmbeddingGrad` defaults to a deterministic implementation.

  - Ascend 950PR/Ascend 950DT:

    The input tensor x has four dimensions. The shape constraints of each parameter are as follows:
    - The last dimension of the input tensor x, cos, sin, and output tensor y must be the same and less than or equal to 1024. In half, interleave, and interleave-half modes, the last dimension must be exactly divisible by 2. In quarter mode, the last dimension must be exactly divisible by 4.
    - The shape of the input tensor x must be the same as that of the output tensor y.
    - The shapes of the input tensors cos and sin must be the same. The shapes of cos and sin must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with x, and the shape after broadcasting must be the same as the shape of x.

  - <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>:

    - The input tensor dy supports the BNSD, BSND, SBND, and TND layouts.
    - The D dimension of the input tensors dy, cos, sin, and xOptional and the output tensors dxOut, dcosOut, and dsinOut must be the same, and the value must be less than 896 and a multiple of 2.
    - The shape of the input tensor dy and xOptional must be the same as that of the output tensor dxOut.
    - The shape of the input tensors cos and sin must be the same as that of the output tensors dcosOut and dsinOut, respectively. In addition, the shape of cos must be the same as that of sin.
    - half mode:
      - B, N < 1000. When dsin and dcos need to be computed, B x N <= 1024.
      - When dy is in BNSD format, cos and sin support 11SD, B1SD, and BNSD. When cos and sin are in B1SD format, B < S must be met.
      - When dy is in BSND format, cos and sin support 1S1D, BS1D, and BSND. When cos and sin are in BS1D format, B < S must be met.
      - When dy is in SBND format, cos and sin support S11D, SB1D, and SBND.
      - When dy is in TND format, cos and sin support T1D and TND.
    - Interleave mode:
      - B x N < 1000 (N < 1000 when dy is in TND format)
      - When dy is in BNSD format, cos and sin support 11SD.
      - When dy is in BSND format, cos and sin support 1S1D.
      - When dy is in SBND format, cos and sin support S11D.
      - When dy is in TND format, cos and sin support T1D.

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_rotary_position_embedding_grad.h"
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
                    aclDataType dataType, aclTensor** tensor)
{
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

int main()
{
    // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
    // Set the deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as required.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the input and output based on the API.
    std::vector<int64_t> dyShape = {1, 1, 1, 128};
    std::vector<int64_t> cosShape = {1, 1, 1, 128};
    std::vector<int64_t> sinShape = {1, 1, 1, 128};
    std::vector<int64_t> dxOutShape = {1, 1, 1, 128};
    int64_t mode = 1;

    void* dyDeviceAddr = nullptr;
    void* cosDeviceAddr = nullptr;
    void* sinDeviceAddr = nullptr;
    void* dxOutDeviceAddr = nullptr;
    aclTensor* dy = nullptr;
    aclTensor* cos = nullptr;
    aclTensor* sin = nullptr;
    aclTensor* dxOut = nullptr;
    aclTensor* dcosOut = nullptr;
    aclTensor* dsinOut = nullptr;

    std::vector<float> dyHostData = {
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
    std::vector<float> dxOutHostData(128, 0);

    // Create a dy aclTensor.
    ret = CreateAclTensor(dyHostData, dyShape, &dyDeviceAddr, aclDataType::ACL_FLOAT, &dy);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a cos aclTensor.
    ret = CreateAclTensor(cosHostData, cosShape, &cosDeviceAddr, aclDataType::ACL_FLOAT, &cos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a sin aclTensor.
    ret = CreateAclTensor(sinHostData, sinShape, &sinDeviceAddr, aclDataType::ACL_FLOAT, &sin);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a dxOut aclTensor.
    ret = CreateAclTensor(dxOutHostData, dxOutShape, &dxOutDeviceAddr, aclDataType::ACL_FLOAT, &dxOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create dsinOut and dcosOut.
    std::vector<int64_t> emptyTensorOutShape = {1, 1, 1, 0};
    std::vector<int64_t> emptyTensorStride(emptyTensorOutShape.size(), 0);
    dcosOut = aclCreateTensor(emptyTensorOutShape.data(), emptyTensorOutShape.size(), aclDataType::ACL_FLOAT,
                              emptyTensorStride.data(), 0, aclFormat::ACL_FORMAT_ND, emptyTensorOutShape.data(),
                              emptyTensorOutShape.size(), nullptr);
    dsinOut = aclCreateTensor(emptyTensorOutShape.data(), emptyTensorOutShape.size(), aclDataType::ACL_FLOAT,
                              emptyTensorStride.data(), 0, aclFormat::ACL_FORMAT_ND, emptyTensorOutShape.data(),
                              emptyTensorOutShape.size(), nullptr);                               
    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnRotaryPositionEmbeddingGrad.
    ret = aclnnRotaryPositionEmbeddingGradGetWorkspaceSize(dy, cos, sin, nullptr, mode, dxOut, dcosOut, dsinOut,
                                                           &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnRotaryPositionEmbeddingGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnRotaryPositionEmbeddingGrad.
    ret = aclnnRotaryPositionEmbeddingGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRotaryPositionEmbeddingGrad failed. ERROR: %d\n", ret); return ret);
    // 4. (Fixed writing) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
    auto size = GetShapeSize(dxOutShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), dxOutDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
    aclDestroyTensor(dy);
    aclDestroyTensor(cos);
    aclDestroyTensor(sin);
    aclDestroyTensor(dxOut);

    // 7. Release device resources.
    aclrtFree(dyDeviceAddr);
    aclrtFree(cosDeviceAddr);
    aclrtFree(sinDeviceAddr);
    aclrtFree(dxOutDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
