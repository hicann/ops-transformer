# aclnnQuantGroupedMatmulInplaceAdd

[📄 View Source Code](https://gitcode.com/cann/ops-transformer/tree/master/gmm/quant_grouped_matmul_inplace_add)

## Supported Products

|Product     | Supported|
|:----------------------------|:-----------:|
|Ascend 950PR/Ascend 950DT|      √     |
|<term>Atlas A3 training products/Atlas A3 inference products</term>|      ×     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>|      ×     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function

- API function: In the micro-batch training scenario, micro-batch gradient accumulation is required, and there are a large number of fusion scenarios where GroupedMatMul is followed by InplaceAdd. The QuantGroupedMatmulInplaceAdd operator is introduced to fuse the preceding operators to improve the network performance. Performs group matrix multiplication and addition. The basic function is the combination of matrix multiplication and addition. For example, in the T-C quantization scenario, $y_i[m,n]=(x1_i[m,k_i] \times x2_i[k_i,n]) * scale2_i[n] * scale1_i + y_i[m,n], i=1...g$, where g indicates the number of groups, and $m/k_i/n$ indicates the corresponding dimension.

  Compared with the [aclnnGroupedMatmulV4](../../grouped_matmul/docs/aclnnGroupedMatmulV4_en.md) API, this API has the following changes:
  - The input and output parameter types are both aclTensor.
  - The InplaceAdd computation is added after the GroupedMatMul computation is complete.
  - Only quantization scenarios (1.MX quantization; 2.T-C quantization) are supported. For details about the quantization modes, see [Quantization Overview](../../../docs/en/context/quant_mode_introduction.md).
  - Only x1 and x2 of FLOAT8_E5M2, FLOAT8_E4M3FN and HIFLOAT8 are supported.

- Formulas:
  - MX quantization:

  $$
    y_i[m,n] = \sum_{j=0}^{kLoops-1} ((\sum_{k=0}^{gsK-1} (x1Slice_i * x2Slice_i)) * (scale1_i[m, j] * scale2_i[j, n])) + y_i[m,n]
  $$

  In the preceding information, gsK indicates the quantized block size of the K axis, that is, 32. $x1Slice_i$ indicates the vector whose length is gsK and that is in the mth row of $x1_i$. $x2Slice_i$ indicates the vector whose length is gsK and that is in the nth column of $x2_i$. The K axis is sliced from $j*gsK$. The value range of j is [0, kLoops), and kLoops is calculated as follows: kLoops = ceil($K_i$ / gsK). The length of the last slice can be less than gsK.

  - T-C quantization:
  
  $$
    y_i=(x1_i\times x2_i) * scale2_i * scale1_i + y_i
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnQuantGroupedMatmulInplaceAddGetWorkspaceSize` is called to obtain the input parameters and compute the required workspace size based on the process. Then, `aclnnQuantGroupedMatmulInplaceAdd` is called to perform computation.

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
      <th>Name</th>
      <th style="white-space: nowrap">Input/Output</th>
      <th>Description</th>
      <th>Usage</th>
      <th>Data Type</th>
      <th><a href="../../../docs/en/context/data_format.md" target="_blank"> Data Format</a></th>
      <th style="white-space: nowrap">Dimension (Shape)</th>
      <th><a href="../../../docs/en/context/non_contiguous_tensor.md" target="_blank">Non-Contiguous Tensor</a></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x1</td>
      <td>Input</td>
      <td>aclTensor on the device, which is the input x1 in the formula.</td>
      <td>-</td>
      <td>FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8</td>
      <td>ND</td>
      <td>2(K, M)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>aclTensor on the device, which is the input x2 in the formula.</td>
      <td>-</td>
      <td>FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8</td>
      <td>ND</td>
      <td>2(K, N)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scale1Optional</td>
      <td>Optional input</td>
      <td>Indicates the scaling factor introduced by x1 quantization in the quantization parameters, which is an aclTensor on the device.</td>
      <td>
        <ul>
          <li>For the comprehensive constraints, see <a href="#constraints" target="_blank">Constraints</a>.</li>
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
      <td>Scale (introduced by x2 quantization) for quantization parameters, aclTensor on the device.</td>
      <td>
        <ul>
          <li>For the comprehensive constraints, see <a href="#constraints" target="_blank">Constraints</a>.</li>
        </ul>
      </td>
      <td>FLOAT32, FLOAT8_E8M0</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>groupList</td>
      <td>Input</td>
      <td>Distribution of Matmul sizes along the input and output group axis, aclTensor on the device side.</td>
      <td>
        <ul>
          <li>When groupListType is set to 0, groupList must be a non-negative monotonic non-decreasing sequence. When groupListType is set to 1, groupList must be a non-negative sequence.</li>
          <li>The last value in groupList restricts the valid part of the output data. The part that is not specified in groupList will not be updated.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>ND</td>
      <td>1(g,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>yRef</td>
      <td>Input and output</td>
      <td>aclTensor on the device side, corresponding to the input and output y in the formula.</td>
      <td>When the M axis of x1 or the N axis of x2 is 0, yRef is an empty tensor.</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>3(g, M, N)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>groupListType</td>
      <td>Optional input</td>
      <td>Integer parameter. Only 0 and 1 are supported.</td>
      <td>
        <ul>
          <li>0: The value in groupList is the cumsum result (cumulative sum) of the group axis size.</li>
          <li>1: The value in groupList is the size of each group on the group axis.</li>
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
      <td>Integer parameter, specifying the quantization group size in the m, n, and k directions.</td>
      <td>
        <ul>
          <li>The groupSize input consists of three values: groupSizeM, groupSizeN, and groupSizeK in three directions. Each value occupies 16 bits, and the total 48 bits are used for the lower 48 bits of the int64_t groupSize (the upper 16 bits of groupSize are invalid). The calculation formula is as follows: groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32.</li>
          <li>Currently, only 0 can be transferred.</li>
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
      <td>Size of the workspace to be allocated on the device.</td>
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
  </tbody></table>

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.
  <table style="undefined;table-layout: fixed;width: 1030px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 650px">
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
      <td>The required input, output, or attribute is passed as a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data types and formats of x1, x2, scale2, groupList, yRef, scale1Optional, groupListType, and groupSize are not supported.</td>
    </tr>
  </tbody></table>

## aclnnQuantGroupedMatmulInplaceAdd

- **Parameters**
  <table>
    <thead>
      <tr><th>Parameter</th><th>Input/Output</th><th>Description</th></tr>
    </thead>
    <tbody>
      <tr><td>workspace</td><td>Input</td><td>Address of the workspace to be allocated on the device.</td></tr>
      <tr><td>workspaceSize</td><td>Input</td><td>Size of the workspace to be allocated on the device, which is obtained by the first-phase API aclnnQuantGroupedMatmulInplaceAddGetWorkspaceSize.</td></tr>
      <tr><td>executor</td><td>Input</td><td>Operator executor, containing the operator computation process.</td></tr>
      <tr><td>stream</td><td>Input</td><td>AscendCL stream for executing a task.</td></tr>
    </tbody>
  </table>

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Determinism description: `aclnnQuantGroupedMatmulInplaceAdd` defaults to a deterministic implementation.
- The size of each dimension of x1 and x2 must be less than the maximum value 2147483647 of int32 after 32-byte alignment, and the size of the inner axis must be less than 2097152.
  - The supported input types in the dynamic quantization (T-C quantization) scenario are as follows:
    - The data type combinations supported by non-empty parameters must meet the requirements listed in the following table.

      | x1       | x2  | scale2 | scale1Optional |yRef     |
      |:-------:|:-------:| :------      | :------   | :------ |
      |HIFLOAT8  |HIFLOAT8| FLOAT32    | FLOAT32   | FLOAT32 |

    - The scale1Optional/scale2 must meet the following constraints (g indicates the number of matmul groups, that is, the number of groups):

      | Parameter| Shape Restriction|
      |:---------:| :------ |
      |scale1Optional| 2D tensor or 1D tensor. The shape is (g, 1) or (g,).|
      |scale2| 2D tensor. The shape is (g, N).|

  - The supported data types in the dynamic quantization (mx quantization) scenario are as follows:
    - The data type combinations must meet the requirements listed in the following table.

      | x1       | x2  |  scale2  | scale1Optional |yRef     |
      |:-------:|:-------:| :-------    | :------   | :------ |
      |FLOAT8_E5M2/FLOAT8_E4M3FN  |FLOAT8_E5M2/FLOAT8_E4M3FN| FLOAT8_E8M0   | FLOAT8_E8M0    | FLOAT32 |

    - The scale1Optional/scale2 must meet the following constraints (g indicates the number of matmul groups, that is, the number of groups, and g_i indicates the ith group (the subscript starts from 0)):

      | Parameter| Shape Restriction|
      |:---------:| :------ |
      |scale1Optional| 3D tensor. The shape is ((K / 64) + g, M, 2). The start address offset of scale_i is ((K_0 + K_1 +...+ K_{i-1})/ 64 + g_i) *M* 2. That is, the start address offset of scale_0 is 0, the start address offset of scale_1 is (K_0 / 64 + 1) *M* 2, and the start address offset of scale_2 is ((K_0 + K_1) / 64 + 2) *M* 2.|
      |scale2| 3D tensor. The shape is ((K / 64) + g, N, 2). The start address offset is the same as that of scale1Optional.|

- The maximum value of the first dimension of groupList is 1024, that is, a maximum of 1024 groups are supported.

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

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
int CreateTransposeAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                             aclDataType dataType, aclTensor** tensor) {
    std::vector<int64_t> view_shape = {shape[1], shape[0]};

    auto size = GetShapeSize(view_shape) * sizeof(T);
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
    // Exchange stride.
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
void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}
int aclnnQuantGroupedMatmulInplaceAddTest(int32_t deviceId, aclrtStream &stream) {
    auto ret = Init(deviceId, &stream);
    // Handle the check as required.
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the inputs and outputs based on the API definition.
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
    std::vector<uint8_t> xData(GetShapeSize(x1Shape), 0X10); // hifloat8 2.0 is converted to 0X10 in hexadecimal format.
    std::vector<int64_t> groupListData = {1, 3};
    std::vector<float> scale2Data(GetShapeSize(scale2Shape), 1);
    std::vector<float> yData(GetShapeSize(yShape), 1);
    std::vector<float> scale1Data(GetShapeSize(scale1Shape), 1);
    // Create an x1 aclTensor.
    ret = CreateTransposeAclTensor<uint8_t>(xData, x1Shape, &x1DeviceAddr, aclDataType::ACL_HIFLOAT8, &x1);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1TensorPtr(x1, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an x2 aclTensor.
    ret = CreateAclTensor<uint8_t>(xData, x2Shape, &x2DeviceAddr, aclDataType::ACL_HIFLOAT8, &x2);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2TensorPtr(x2, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create scale2 aclTensor.
    ret = CreateAclTensor<float>(scale2Data, scale2Shape, &scale2DeviceAddr, aclDataType::ACL_FLOAT, &scale2);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scale2TensorPtr(scale2, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> scale2DeviceAddrPtr(scale2DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a y aclTensor.
    ret = CreateAclTensor<float>(yData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &yRef);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> yTensorPtr(yRef, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> yDeviceAddrPtr(yDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a group_list aclTensor.
    ret = CreateAclTensor<int64_t>(groupListData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupList);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> groupListTensorPtr(groupList, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> groupListDeviceAddrPtr(groupListDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the scale1 aclTensor.
    ret = CreateAclTensor<float>(scale1Data, scale1Shape, &scale1DeviceAddr, aclDataType::ACL_FLOAT, &scale1);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scale1TensorPtr(scale1, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> scale1DeviceAddrPtr(scale1DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 3. Call the CANN operator library API.
    // Call the first API of aclnnQuantGroupedMatmulInplaceAdd.
    ret = aclnnQuantGroupedMatmulInplaceAddGetWorkspaceSize(x1, x2, scale1, scale2, groupList, yRef, groupListType, groupSize, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantGroupedMatmulInplaceAddGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    // Call the second API of aclnnQuantGroupedMatmulInplaceAdd.
    ret = aclnnQuantGroupedMatmulInplaceAdd(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantGroupedMatmulInplaceAdd failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
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
    // 1. (Boilerplate) Initialize the device and stream. For details, see the list of external AscendCL APIs.
    // Set the device ID in use.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnQuantGroupedMatmulInplaceAddTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantGroupedMatmulInplaceAddTest failed. ERROR: %d\n", ret); return ret);
    Finalize(deviceId, stream);
    return 0;
}
```
