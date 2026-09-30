# aclnnMoeInitRouting

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference products</term>   |     ×    |
|  <term>Atlas training products</term>   |     ×    |

## Function

- **Description**: Performs routing on the computation result of [aclnnMoeGatingTopKSoftmax](../../moe_gating_top_k_softmax/docs/aclnnMoeGatingTopKSoftmax_en.md).
- **Formula**:
    Flattens the input `expertIdx` with shape [numRows, k] into a row for sorting.
    
    $$
    expandedExpertIdxOut,sortedRowIdx=keyValueSort(expertIdx,rowIdx)
    $$

    $$
    expandedRowIdxOut[sortedRowIdx[i]]=i
    $$

    $$
    expandedXOut[i]=x[sortedRowIdx[i]\%numRows]
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnMoeInitRoutingGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnMoeInitRouting` is called to perform computation.

```cpp
aclnnStatus aclnnMoeInitRoutingGetWorkspaceSize(
    const aclTensor  *x, 
    const aclTensor  *rowIdx, 
    const aclTensor  *expertIdx, 
    int64_t           activeNum, 
    const aclTensor  *expandedXOut, 
    const aclTensor  *expandedRowIdxOut, 
    const aclTensor  *expandedExpertIdxOut, 
    uint64_t         *workspaceSize, 
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnMoeInitRouting(
    void             *workspace, 
    uint64_t          workspaceSize, 
    aclOpExecutor    *executor, 
    aclrtStream       stream)
```

## aclnnMoeInitRoutingGetWorkspaceSize

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1543px"><colgroup>
        <col style="width: 200px">
        <col style="width: 120px">
        <col style="width: 300px">
        <col style="width: 232px">
        <col style="width: 219px">
        <col style="width: 121px">
        <col style="width: 200px">
        <col style="width: 151px">
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
        </tr></thead>
        <tbody>
        <tr>
            <td>x</td>
            <td>Input</td>
            <td>MoE input, that is, the token feature input.</td>
            <td>-</td>
            <td>FLOAT16, BFLOAT16, FLOAT32</td>
            <td>ND</td>
            <td>The shape is (NUM_ROWS, H).</td>
            <td>√</td>
        </tr>
        <tr>
            <td>rowIdx</td>
            <td>Input</td>
            <td>Original row position corresponding to each position.</td>
            <td>The values of rowIdx start from 0 and increment along dimension 1.</td>
            <td>INT32</td>
            <td>ND</td>
            <td>The shape must be the same as that of expertIdx.</td>
            <td>√</td>
        </tr>
        <tr>
            <td>expertIdx</td>
            <td>Input</td>
            <td>K experts corresponding to each row of features.</td>
            <td>-</td>
            <td>INT32</td>
            <td>ND</td>
            <td>The shape is (NUM_ROWS, K).</td>
            <td>√</td>
        </tr>
        <tr>
            <td>activeNum</td>
            <td>Input</td>
            <td>Number of valid rows of expandedXOut.</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>expandedXOut</td>
            <td>Output</td>
            <td>Extended feature based on expertIdx.</td>
            <td>-</td>
            <td>Same as x.</td>
            <td>ND</td>
            <td>Shape = (min(NUM_ROWS, activeNum) × k, H)</td>
            <td>x</td>
        </tr>
        <tr>
            <td>expandedRowIdxOut</td>
            <td>Output</td>
            <td>Mapping between expandedX and x.</td>
            <td>-</td>
            <td>INT32</td>
            <td>ND</td>
            <td>The shape is (NUM_ROWS × K).</td>
            <td>x</td>
        </tr>
        <tr>
            <td>expandedExpertIdxOut</td>
            <td>Output</td>
            <td>ExpertIdx sorting result.</td>
            <td>-</td>
            <td>INT32</td>
            <td>ND</td>
            <td>The shape is (NUM_ROWS × K).</td>
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

- **Returns**

    `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API implements input parameter verification. The following errors may be thrown.
    <table style="undefined;table-layout: fixed; width: 1180px"> 
      <colgroup>
        <col style="width: 250px">
        <col style="width: 130px">
        <col style="width: 800px">
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
          <td>The input and output tensors are null pointers.</td>
        </tr>
        <tr>
          <td>ACLNN_ERR_PARAM_INVALID</td>
          <td>161002</td>
          <td>The input and output data types are not supported.</td>
        </tr>
        <tr>
          <td rowspan="5">ACLNN_ERR_INNER_TILING_ERROR</td>
          <td rowspan="5">561002</td>
          <td>x is not a 2D tensor.</td>
        </tr>
        <tr>
          <td> The shape of `rowIdx` must be 2D, and the shapes of `rowIdx` and `expertIdx` must match.</td>
        </tr>
        <tr>
          <td>The value of `activeNum` is less than 0.</td>
        </tr>
        <tr>
          <td>The shape of `expandedXOut` is not equal to (min(num_rows, activeNum) × k, H).</td>
        </tr>
        <tr>
          <td>The shape of `expandedRowIdxOut` is different from that of `expandedExpertIdxOut` and is not equal to (num_rows × k, ).</td>
        </tr>
      </tbody>
    </table>

## aclnnMoeInitRouting

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1179px">
    <colgroup>
    <col style="width: 169px">
    <col style="width: 130px">
    <col style="width: 880px">
    </colgroup>
      <thead>
        <tr>
          <th>Name</th>
          <th>Input/Output</th>
          <th>Description</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>workspace</td>
          <td>Input</td>
          <td>Memory address of the workspace to be allocated on the device.</td>
        </tr>
        <tr>
          <td>workspaceSize</td>
          <td>Input</td>
          <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnMoeInitRoutingGetWorkspaceSize`.</td>
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

- **Returns**:

    `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
    
## Constraints

- Deterministic computing:
  - `aclnnMoeInitRouting` defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_init_routing.h"
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
    // 1. (Boilerplate) Initialize the device and stream. For details, see the list of external ACL APIs.
    // Set the device ID in use.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as required.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the inputs and outputs based on the API definition.
    std::vector<int64_t> xShape = {3, 4};
    std::vector<int64_t> idxShape = {3, 2};
    std::vector<int64_t> expandedXOutShape = {6, 4};
    std::vector<int64_t> idxOutShape = {6};
    void* xDeviceAddr = nullptr;
    void* rowIdxDeviceAddr = nullptr;
    void* expertIdxDeviceAddr = nullptr;
    void* expandedXOutDeviceAddr = nullptr;
    void* expandedRowIdxOutDeviceAddr = nullptr;
    void* expandedExpertIdxOutDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* rowIdx = nullptr;
    aclTensor* expertIdx = nullptr;
    int64_t activeNum = 3;
    aclTensor* expandedXOut = nullptr;
    aclTensor* expandedRowIdxOut = nullptr;
    aclTensor* expandedExpertIdxOut = nullptr;
    std::vector<float> xHostData = {0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3};
    std::vector<int> expertIdxHostData = {1, 2, 0, 1, 0, 2};
    std::vector<int> rowIdxHostData = {0, 3, 1, 4, 2, 5};
    std::vector<float> expandedXOutHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int> expandedRowIdxOutHostData = {0, 0, 0, 0, 0, 0};
    std::vector<int> expandedExpertIdxOutHostData = {0, 0, 0, 0, 0, 0};
    // Create a self aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(rowIdxHostData, idxShape, &rowIdxDeviceAddr, aclDataType::ACL_INT32, &rowIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertIdxHostData, idxShape, &expertIdxDeviceAddr, aclDataType::ACL_INT32, &expertIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an out aclTensor.
    ret = CreateAclTensor(expandedXOutHostData, expandedXOutShape, &expandedXOutDeviceAddr, aclDataType::ACL_FLOAT, &expandedXOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandedRowIdxOutHostData, idxOutShape, &expandedRowIdxOutDeviceAddr, aclDataType::ACL_INT32, &expandedRowIdxOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandedExpertIdxOutHostData, idxOutShape, &expandedExpertIdxOutDeviceAddr, aclDataType::ACL_INT32, &expandedExpertIdxOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnMoeInitRouting.
    ret = aclnnMoeInitRoutingGetWorkspaceSize(x, rowIdx, expertIdx, activeNum, expandedXOut, expandedRowIdxOut, expandedExpertIdxOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeInitRoutingGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the computed workspaceSize.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnMoeInitRouting.
    ret = aclnnMoeInitRouting(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeInitRouting failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
    auto expandedXSize = GetShapeSize(expandedXOutShape);
    std::vector<float> expandedXData(expandedXSize, 0);
    ret = aclrtMemcpy(expandedXData.data(), expandedXData.size() * sizeof(expandedXData[0]), expandedXOutDeviceAddr, expandedXSize * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expandedXSize; i++) {
        LOG_PRINT("expandedXData[%ld] is: %f\n", i, expandedXData[i]);
    }
    auto expandedRowIdxSize = GetShapeSize(idxOutShape);
    std::vector<int> expandedRowIdxData(expandedRowIdxSize, 0);
    ret = aclrtMemcpy(expandedRowIdxData.data(), expandedRowIdxData.size() * sizeof(expandedRowIdxData[0]), expandedRowIdxOutDeviceAddr, expandedRowIdxSize * sizeof(int32_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expandedRowIdxSize; i++) {
        LOG_PRINT("expandedRowIdxData[%ld] is: %d\n", i, expandedRowIdxData[i]);
    }
    auto expandedExpertIdxSize = GetShapeSize(idxOutShape);
    std::vector<int> expandedExpertIdxData(expandedExpertIdxSize, 0);
    ret = aclrtMemcpy(expandedExpertIdxData.data(), expandedExpertIdxData.size() * sizeof(expandedExpertIdxData[0]), expandedExpertIdxOutDeviceAddr, expandedExpertIdxSize * sizeof(int32_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expandedExpertIdxSize; i++) {
        LOG_PRINT("expandedExpertIdxData[%ld] is: %d\n", i, expandedExpertIdxData[i]);
    }
    // 6. Destroy aclTensor objects. Modify the code based on the API definition.
    aclDestroyTensor(x);
    aclDestroyTensor(rowIdx);
    aclDestroyTensor(expertIdx);
    aclDestroyTensor(expandedXOut);
    aclDestroyTensor(expandedRowIdxOut);
    aclDestroyTensor(expandedExpertIdxOut);

    // 7. Free device resources. Modify the configuration based on the API definition.
    aclrtFree(xDeviceAddr);
    aclrtFree(rowIdxDeviceAddr);
    aclrtFree(expertIdxDeviceAddr);
    aclrtFree(expandedXOutDeviceAddr);
    aclrtFree(expandedRowIdxOutDeviceAddr);
    aclrtFree(expandedExpertIdxOutDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
