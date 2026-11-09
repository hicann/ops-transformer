# aclnnQuantGroupedMatmulInplaceAdd

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T12:47:49.336Z -->

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

- Description: In micro-batch training scenarios where micro-batch gradient accumulation is required, there are numerous fusion scenarios involving GroupedMatMul followed by InplaceAdd. The QuantGroupedMatmulInplaceAdd operator fuses the above operators to improve network performance. It implements grouped matrix multiplication and addition computations. The basic function is a combination of matrix multiplication and addition, for example, in the T-C quantization scenario: $y_i[m,n]=(x1_i[m,k_i] \times x2_i[k_i,n]) * scale2_i[n] * scale1_i + y_i[m,n], i=1...g$, where g is the number of groups and $m/k_i/n$ are the corresponding dimensions.

    Compared with the [aclnnGroupedMatmulV4](../../grouped_matmul/docs/aclnnGroupedMatmulV4_en.md) API, **the changes of this API are as follows:**
    - Both input and output parameters are of the aclTensor type.
    - An InplaceAdd computation is added after the GroupedMatMul computation.
    - Only quantization use cases are supported (1. mx quantization; 2. T-C quantization; 3. T-T quantization). For details about the quantization modes, see [Quantization](../../../docs/en/context/quant_mode_introduction.md).
    - Only x1 and x2 inputs of the FLOAT8_E5M2, FLOAT8_E4M3FN, HIFLOAT8 or FLOAT4_E2M1 type are supported.

- Formula:
    - **mx quantization:**

    $$
     y_i[m,n] = \sum_{j=0}^{kLoops-1} ((\sum_{k=0}^{gsK-1} (x1Slice_i * x2Slice_i)) * (scale1_i[m, j] * scale2_i[j, n])) + y_i[m,n]
    $$

    Where gsK represents the quantization block size of the K axis, i.e., 32; $x1Slice_i$ represents the vector of length gsK in the m-th row of $x1_i$; $x2Slice_i$ represents the vector of length gsK in the n-th column of $x2_i$. The K axis starts slicing from $j*gsK$, with j in the range [0, kLoops), where kLoops = ceil($K_i$ / gsK). The case where the last slice length is less than gsK is supported.
    - **T-T/T-C quantization:**

    $$
     y_i=(x1_i\times x2_i) * scale2_i * scale1_i + y_i
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnQuantGroupedMatmulInplaceAddGetWorkspaceSize" API first to obtain the input parameters and calculate the required workspace size based on the computation process, and then call the "aclnnQuantGroupedMatmulInplaceAdd" API to perform the computation.

```cpp
aclnnStatus aclnnQuantGroupedMatmulInplaceAddGetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *scale1Optional,
    const aclTensor *scale2,
    const aclTensor *groupList,
    aclTensor       *yRef,
    int64_t          groupListType,
    int64_t          groupSize,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnQuantGroupedMatmulInplaceAdd(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnQuantGroupedMatmulInplaceAddGetWorkspaceSize

- **Parameters**
  <table style="undefined;table-layout: fixed;width: 1567px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 330px">
  <col style="width: 212px">
  <col style="width: 100px">
  <col style="width: 190px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th style="white-space: nowrap">Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th><a href="../../../docs/en/context/data_format.md" target="_blank">Data Format</a></th>
      <th style="white-space: nowrap">Dimension (shape)</th>
      <th><a href="../../../docs/en/context/non_contiguous_tensor.md" target="_blank">non-contiguous tensor</a></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x1</td>
      <td>Input</td>
      <td>aclTensor on the device, input x1 in the formula.</td>
      <td>-</td>
      <td>FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, FLOAT4_E2M1</td>
      <td>ND</td>
      <td>2(K, M)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>aclTensor on the device, input x2 in the formula.</td>
      <td>-</td>
      <td>FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, FLOAT4_E2M1</td>
      <td>ND</td>
      <td>2(K, N)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scale1Optional</td>
      <td>Optional input</td>
      <td>Scaling factor introduced by x1 quantization in quantization parameters, aclTensor on the Device side.</td>
      <td>        <ul>
          <li>For overall constraints, see <a href="#constraints" target="_blank">Constraints</a>.</li>
        </ul>
      </td>
      <td>FLOAT32, FLOAT8_E8M0</td>
      <td>ND</td>
      <td>1-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scale2</td>
      <td>Input</td>
      <td>Scale factor introduced by x2 quantization in the quantization parameters, in the format of an aclTensor on the device.</td>
      <td>        <ul>
          <li>For comprehensive constraints, see <a href="#constraints" target="_blank">Constraints</a>.</li>
        </ul>
      </td>
      <td>FLOAT32, FLOAT8_E8M0</td>
      <td>ND</td>
      <td>1-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>groupList</td>
      <td>Input</td>
      <td>Indicates the matmul size distribution of input and output along the group axis, in the form of an aclTensor on the Device.</td>
      <td>        <ul>
          <li>If groupListType is 0, groupList must be a non-negative monotonically non-decreasing sequence, and the last value must not be greater than the first dimension of the tensor in x. If groupListType is 1, groupList must be a non-negative sequence, and the sum of the values must not be greater than the first dimension of the tensor in x.</li>
          <li>The values in groupList constrain the valid part of the output data, and the parts not specified in groupList do not participate in the update.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>ND</td>
      <td>1(g,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>yRef</td>
      <td>Input/Output</td>
      <td>aclTensor on the device, corresponding to the input/output y in the formula.</td>
      <td>When the M dimension of x1 or the N dimension of x2 is 0, yRef is an empty tensor.</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>3(g, M, N)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>groupListType</td>
      <td>Optional input</td>
      <td>Integer parameter. Only 0 and 1 are supported.</td>
      <td>        <ul>
          <li>0: The values in groupList are the cumsum (cumulative sum) results of the sizes along the group axis.</li>
          <li>1: The values in groupList are the size of each group along the group axis.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupSize</td>
      <td>Input</td>
      <td>Integer parameter. Inputs the quantization group sizes in the m, n, and k directions.</td>
      <td>        <ul>
          <li>The groupSize input consists of three values, groupSizeM, groupSizeN, and groupSizeK, one for each of the three directions. Each value occupies 16 bits, and together they occupy the lower 48 bits of the int64_t groupSize (the upper 16 bits of groupSize are invalid). The formula is: groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32.</li>
          <li>Currently, only 0 is supported.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Size of the workspace to be requested on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor containing the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:
  <table style="undefined;table-layout: fixed;width: 1030px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 650px">
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
      <td>A required input, output, or attribute parameter is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data type or data format of x1, x2, scale2, groupList, yRef, or scale1Optional is not supported, or the value of groupListType or groupSize is not supported.</td>
    </tr>
  </tbody></table>

## aclnnQuantGroupedMatmulInplaceAdd

- **Parameters**
  <table>
    <thead>
      <tr><th>Parameter</th><th>Input/Output</th><th>Description</th></tr>
    </thead>
    <tbody>
      <tr><td>workspace</td><td>Input</td><td>Workspace memory address allocated on the Device.</td></tr>
      <tr><td>workspaceSize</td><td>Input</td><td>Workspace size allocated on the Device, obtained by calling the first-phase API aclnnQuantGroupedMatmulInplaceAddGetWorkspaceSize.</td></tr>
      <tr><td>executor</td><td>Input</td><td>Op executor that contains the computation flow of the operator.</td></tr>
      <tr><td>stream</td><td>Input</td><td>Specifies the AscendCL stream for executing tasks.</td></tr>
    </tbody>
  </table>

- **Return Value**

  Returns a status code of type aclnnStatus. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

  - Deterministic computation: aclnnQuantGroupedMatmulInplaceAdd defaults to a deterministic implementation.
  - The size of each dimension of x1 and x2, after 32-byte alignment, must be less than the maximum int32 value 2147483647, and the inner axis size must be less than 2097152.
    - Supported input types for dynamic quantization (T-T/T-C quantization) use cases:
      - The supported data type combinations for non-empty parameters must satisfy the following table:

        | x1       | x2  | scale2 | scale1Optional |yRef     |
        |:-------:|:-------:| :------      | :------   | :------ |
        |HIFLOAT8  |HIFLOAT8| FLOAT32    | FLOAT32   | FLOAT32 |

      - scale1Optional/scale2 must meet the following constraints (where g is the number of matmul groups, i.e., the number of groups):

        | Parameter | Shape Constraint |
        |:---------:| :------ |
        |scale1Optional| 2D tensor or 1D tensor, with shape (g, 1) or (g,)|
        |scale2| pertensor use case: 2D tensor or 1D tensor, with shape (g, 1) or (g,); perchannel use case: 2D tensor, with shape (g, N).|

    - Data types supported in the dynamic quantization (MX quantization) scenario:
      - The data type combinations must meet the following table:

        | x1       | x2  |  scale2  | scale1Optional |yRef     |
        |:-------:|:-------:| :-------    | :------   | :------ |
        |FLOAT8_E5M2/FLOAT8_E4M3FN  |FLOAT8_E5M2/FLOAT8_E4M3FN| FLOAT8_E8M0   | FLOAT8_E8M0    | FLOAT32 |
        |FLOAT4_E2M1  |FLOAT4_E2M1| FLOAT8_E8M0   | FLOAT8_E8M0    | FLOAT32 |

      - scale1Optional/scale2 must satisfy the following constraints (where g is the number of matmul groups, i.e., the group count, and g_i is the i-th group (index starting from 0)):

        | Parameter | Shape Constraint |
        |:---------:| :------ |
        |scale1Optional| 3D tensor with shape ((K / 64) + g, M, 2), scale\_i the start address offset is ((K\_0 + K\_1 + ...+ K\_{i-1})/ 64 + g\_i) \* M \* 2, that is, the start address offset of scale_0 is 0, the start address offset of scale_1 is (K\_0 / 64 + 1) \* M \* 2, the start address offset of scale_2 is ((K\_0 + K\_1) / 64 + 2) \* M \* 2, and so on.|
        |scale2| 3D tensor with shape ((K / 64) + g, N, 2). The start address offset is the same as that of scale1Optional.|

      - In mxfp4, x1/x2 must both be FLOAT4_E2M1 and scale1/scale2 must both be FLOAT8_E8M0; the M axis of x1 and the N axis of x2 must be even; K \* M != 2 and K \* N != 2.

  - The first dimension of groupList supports a maximum of 1024, i.e., a maximum of 1024 groups.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include <memory>
#include <utility>
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_grouped_matmul_inplace_add.h"
#define CHECK_RET(cond, return_expr) \
    do {                               \
        if (!(cond)) {                   \
            return_expr;                   \
        }                                \
    } while (0)
#define CHECK_FREE_RET(cond, return_expr) \
    do {                                  \
        if (!(cond)) {                    \
            Finalize(deviceId, stream);   \
            return_expr;                  \
        }                                 \
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
int CreateTransposeAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                             aclDataType dataType, aclTensor** tensor) {
    std::vector<int64_t> view_shape = {shape[1], shape[0]};

    auto size = GetShapeSize(view_shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the Device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy data on the Host to the memory on the Device.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    // Swap the strides.
    std::swap(strides[0], strides[1]);
    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(view_shape.data(), view_shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}
template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the Device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy data from the Host to the Device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    // Calculate the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    // Call the aclCreateTensor API to create the aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}
void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}
int aclnnQuantGroupedMatmulInplaceAddTest(int32_t deviceId, aclrtStream &stream) {
    auto ret = Init(deviceId, &stream);
    // Handle check as required.
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the inputs and outputs. Custom construction is required based on the API.
    std::vector<int64_t> x1Shape = {2, 3};
    std::vector<int64_t> x2Shape= {2, 3};
    std::vector<int64_t> scale2Shape = {2, 3};
    std::vector<int64_t> yShape = {2, 3, 3};
    std::vector<int64_t> scale1Shape = {2, 1};
    std::vector<int64_t> groupListShape = {2};
    void* x1DeviceAddr = nullptr;
    void* x2DeviceAddr = nullptr;
    void* scale2DeviceAddr = nullptr;
    void* scale1DeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    void* groupListDeviceAddr = nullptr;
    aclTensor* x1 = nullptr;
    aclTensor* x2 = nullptr;
    aclTensor* groupList = nullptr;
    aclTensor* scale2 = nullptr;
    aclTensor* yRef = nullptr;
    aclTensor* scale1 = nullptr;
    aclTensor* out = nullptr;
    int64_t groupListType = 0;
    int64_t groupSize = 0;
    std::vector<uint8_t> xData(GetShapeSize(x1Shape), 0X10); // hifloat8 2.0 converted to hexadecimal 0X10
    std::vector<int64_t> groupListData = {1, 3};
    std::vector<float> scale2Data(GetShapeSize(scale2Shape), 1);
    std::vector<float> yData(GetShapeSize(yShape), 1);
    std::vector<float> scale1Data(GetShapeSize(scale1Shape), 1);
    // Create the x1 aclTensor.
    ret = CreateTransposeAclTensor<uint8_t>(xData, x1Shape, &x1DeviceAddr, aclDataType::ACL_HIFLOAT8, &x1);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1TensorPtr(x1, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the x2 aclTensor.
    ret = CreateAclTensor<uint8_t>(xData, x2Shape, &x2DeviceAddr, aclDataType::ACL_HIFLOAT8, &x2);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2TensorPtr(x2, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the scale2 aclTensor.
    ret = CreateAclTensor<float>(scale2Data, scale2Shape, &scale2DeviceAddr, aclDataType::ACL_FLOAT, &scale2);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scale2TensorPtr(scale2, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> scale2DeviceAddrPtr(scale2DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the y aclTensor.
    ret = CreateAclTensor<float>(yData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &yRef);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> yTensorPtr(yRef, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> yDeviceAddrPtr(yDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create group_list aclTensor.
    ret = CreateAclTensor<int64_t>(groupListData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupList);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> groupListTensorPtr(groupList, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> groupListDeviceAddrPtr(groupListDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create scale1 aclTensor.
    ret = CreateAclTensor<float>(scale1Data, scale1Shape, &scale1DeviceAddr, aclDataType::ACL_FLOAT, &scale1);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scale1TensorPtr(scale1, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> scale1DeviceAddrPtr(scale1DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 3. Call the CANN operator library API.
    // Call the first-phase API of aclnnQuantGroupedMatmulInplaceAdd.
    ret = aclnnQuantGroupedMatmulInplaceAddGetWorkspaceSize(x1, x2, scale1, scale2, groupList, yRef, groupListType, groupSize, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantGroupedMatmulInplaceAddGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    // Call the second-phase API of aclnnQuantGroupedMatmulInplaceAdd.
    ret = aclnnQuantGroupedMatmulInplaceAdd(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantGroupedMatmulInplaceAdd failed. ERROR: %d\n", ret); return ret);
    // 4. (boilerplate) Synchronize and wait for the task to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value, copy the result from Device memory to Host memory, and modify it based on the specific API definition.
    auto size = GetShapeSize(yShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), size * sizeof(uint32_t), yDeviceAddr,
                      size * sizeof(uint32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t j = 0; j < size; j++) {
        LOG_PRINT("result[%ld] is: %f\n", j, resultData[j]);
    }
    return ACL_SUCCESS;
}
int main()
{
    // 1. (boilerplate) Initialize device/stream. Refer to the AscendCL external API list.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnQuantGroupedMatmulInplaceAddTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantGroupedMatmulInplaceAddTest failed. ERROR: %d\n", ret); return ret);
    Finalize(deviceId, stream);
    return 0;
}
```
