# aclnnGroupedMatmulAddV2

[📄 View Source Code](https://gitcode.com/cann/ops-transformer/tree/master/gmm/grouped_matmul_add)

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

- Description: Implements grouped matrix multiplication, supporting non-uniform matrix dimension sizes across multiple groups. The basic function is matrix multiplication, for example, $y_i[m_i,n_i]=x_i[m_i,k_i] \times weight_i[k_i,n_i]+y_i[m_i,n_i], i=1...g$, where $g$ indicates the number of groups and $m_i$, $k_i$, and $n_i$ define the shapes for each group. The input and output data types are both aclTensor, with K-axis grouping.

  - K-axis grouping: $k_i$ can vary, while $m_i$ and $n_i$ remain the same for each group.
  - Compared with the [GroupedMatmulAdd](./aclnnGroupedMatmulAdd.md) API, the following functions are added:
    - The value in groupList can be the size of each group on the group axis.
- Formulas:

  $$
  yRef_i=x_i\times weight_i + y_i
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnGroupedMatmulAddV2GetWorkspaceSize` is called to obtain the input parameters and compute the required workspace size based on the process. Then, `aclnnGroupedMatmulAddV2` is called to perform computation.

```cpp
aclnnStatus aclnnGroupedMatmulAddV2GetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weight,
    const aclTensor *groupList,
    aclTensor       *yRef,
    bool             transposeX,
    bool             transposeWeight,
    int64_t          groupType,
    int64_t          group_list_type,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnGroupedMatmulAddV2(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnGroupedMatmulAddV2GetWorkspaceSize

- **Parameters:**

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
      <th>Data Format</th>
      <th style="white-space: nowrap">Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x (aclTensor)</td>
      <td>Input</td>
      <td>x in the formula.</td>
      <td>
        <ul>
          <li>x must be transposed.</li>
          <li>Empty tensors are not supported.</li>
        </ul>
      </td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight (aclTensor)</td>
      <td>Input</td>
      <td>Weight in the formula.</td>
      <td>
        <ul>
          <li>Weight cannot be transposed.</li>
          <li>Empty tensors are not supported.</li>
        </ul>
      </td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>groupList (aclTensor)</td>
      <td>Input</td>
      <td>Size distribution of the matmul operation in the input and output group axis directions.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y (aclTensor)</td>
      <td>Input</td>
      <td>Input matrix for in-place accumulation, of the aclTensor type on the device, which is y in the formula.</td>
      <td>-</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transposeX (bool)</td>
      <td>Input</td>
      <td>Whether the x matrix is transposed. The value is a Boolean value on the host.</td>
      <td>
        <ul>
          <li>Currently, only True is supported.</li>
        </ul>
      </td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transposeWeight (bool)</td>
      <td>Input</td>
      <td>Specifies whether to transpose the weight matrix.</td>
      <td>
        <ul>
          <li>Currently, only False is supported.</li>
        </ul>
      </td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupType (int64_t)</td>
      <td>Input</td>
      <td>Grouping type.</td>
      <td>
        <ul>
          <li>Currently, only 2 (K-axis grouping) is supported.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupListType (int64_t)</td>
      <td>Input</td>
      <td>Integer. Currently, only two values are supported.</td>
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
      <td>yRef (aclTensor)</td>
      <td>Output</td>
      <td>Reference to the input matrix y that is accumulated in place (the same as y), which is yRef in the formula.</td>
      <td>-</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t)</td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
      <td></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor)</td>
      <td>Output</td>
      <td>Operator executor, containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Returns**

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
      <td>The passed x, weight, groupList, y, or yRef is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data type or format of x, weight, groupList, y, or yRef is not supported.</td>
    </tr>
      <td>The data types of x and weight are inconsistent.</td>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>The shapes of x, weight, y, and yRef do not meet the matrix multiplication requirements.</td>
    </tr>
  </tbody></table>

## aclnnGroupedMatmulAddV2

- **Parameters:**
  <table>
    <thead>
      <tr><th>Parameter</th><th>Input/Output</th><th>Description</th></tr>
    </thead>
    <tbody>
      <tr><td>workspace</td><td>Input</td><td>Address of the workspace to be allocated on the device.</td></tr>
      <tr><td>workspaceSize</td><td>Input</td><td>Size of the workspace allocated on the device, which is obtained by the first API aclnnGroupedMatmulAddV2GetWorkspaceSize.</td></tr>
      <tr><td>executor</td><td>Input</td><td>Operator executor, containing the operator computation process.</td></tr>
      <tr><td>stream</td><td>Input</td><td>Stream for executing a task.</td></tr>
    </tbody>
  </table>

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic description: The default deterministic implementation of aclnnGroupedMatmulAddV2 is used.
- The size of each dimension for every tensor in `x` and `weight`, after 32-byte alignment, should be less than the maximum value of INT32 (2147483647).
- The supported input type combinations are as follows:
  - `x`: FLOAT16; `weight`: FLOAT16; `y`: FLOAT32
  - `x`: BFLOAT16; `weight`: BFLOAT16; `y`: FLOAT32

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include <memory>
#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul_add_v2.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
    do {                                  \
        if (!(cond)) {                    \
            Finalize(deviceId, stream);   \
            return_expr;                  \
        }                                 \
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
int CreateAclTensor_New(const std::vector<int64_t>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
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

template <typename T>
int CreateAclTensor(const std::vector<int64_t>& shape, void** deviceAddr,
                      aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy the data on the host to the memory on the device.
    std::vector<T> hostData(size, 0);
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

int aclnnGroupedMatmulAddV2Test(int32_t deviceId, aclrtStream &stream) {
    auto ret = Init(deviceId, &stream);
    // Handle the check as required.
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs based on the API definition.
    std::vector<int64_t> xShape = {512, 256};
    std::vector<int64_t> weightShape= {512, 256};
    std::vector<int64_t> yShape = {2, 256, 256};
    std::vector<int64_t> groupListShape = {2};
    std::vector<int64_t> groupListData = {256, 512};
    void* xDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    void* groupListDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* groupedList = nullptr;
    aclTensor* y = nullptr;
    aclTensor* yRef = nullptr;
    bool transpose_x = true;
    bool transpose_weight = false;
    int group_type = 2;
    int group_list_type = 0;

    // Create an x aclTensorList.
    ret = CreateAclTensor<uint16_t>(xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a weight aclTensorList.
    ret = CreateAclTensor<uint16_t>(weightShape, &weightDeviceAddr, aclDataType::ACL_BF16, &weight);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> weightTensorPtr(weight, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a y aclTensorList.
    ret = CreateAclTensor<float>(yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> yTensorPtr(y, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> yDeviceAddrPtr(yDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a group_list aclTensor.
    ret = CreateAclTensor_New<int64_t>(groupListData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupedList);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> groupedListTensorPtr(groupedList, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> groupListDeviceAddrPtr(groupListDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    yRef = y;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 3. Call the CANN operator library API.
    // Call the first-phase API of aclnnGroupedMatmulAddV2.
    ret = aclnnGroupedMatmulAddV2GetWorkspaceSize(x, weight, groupedList, yRef, transpose_x, transpose_weight, group_type, group_list_type, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulAddV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    // Call the second-phase API of aclnnGroupedMatmulAddV2.
    ret = aclnnGroupedMatmulAddV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulAddV2 failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
    auto size = GetShapeSize(yShape);
    std::vector<uint16_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), size * sizeof(resultData[0]), yDeviceAddr,
                          size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t j = 0; j < size; j++) {
        LOG_PRINT("result[%ld] is: %d\n", j, resultData[j]);
    }
    return ACL_SUCCESS;
}
int main()
{
    // 1. (Boilerplate) Initialize the device and stream. For details, see the list of external AscendCL APIs.
    // Set the device ID in use.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnGroupedMatmulAddV2Test(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulAddV2Test failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}
```
