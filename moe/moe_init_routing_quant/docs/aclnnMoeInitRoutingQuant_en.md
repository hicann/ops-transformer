# aclnnMoeInitRoutingQuant

## Supported Products

| Product                                                        |  Supported  |
| :----------------------------------------------------------- |:-------:|
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    ×    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference products</term>                            |    ×    |
| <term>Atlas training products</term>                             |    ×    |

## Function

- **Description**: Performs routing on the computation result of [aclnnMoeGatingTopKSoftmax](../../../moe/moe_gating_top_k_softmax/docs/aclnnMoeGatingTopKSoftmax_en.md), and then quantizes the result.

- **Formula**:

$$
expandedExpertIdx,sortedRowIdx=keyValueSort(expertIdx,rowIdx)
$$

$$
expandedRowIdx[sortedRowIdx[i]]=i
$$

$$
expandedX[i]=quant[x[sortedRowIdx[i]\%numRows]]
$$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnMoeInitRoutingQuantGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnMoeInitRoutingQuant` is called to perform computation.

* `aclnnStatus aclnnMoeInitRoutingQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *rowIdx, const aclTensor *expertIdx, int64_t activeNum, double scale, double offset, const aclTensor *expandedXOut, const aclTensor *expandedRowIdxOut, const aclTensor *expandedExpertIdxOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnMoeInitRoutingQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnMoeInitRoutingQuantGetWorkspaceSize

- **Parameters:**
    - `x` (aclTensor\*, computation input): input of MoE, that is, token feature input. The input must be a 2D tensor with shape `(NUM_ROWS, H)`. The data type can be FLOAT16, BFLOAT16, or FLOAT32. The [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported.
    - `rowIdx` (aclTensor\*, computation input): original row position corresponding to each position. The shape must be the same as that of `expertIdx`. The value starts from 0 and increases along dimension 1. The data type can be INT32. The [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported.
    - `expertIdx` (aclTensor\*, computation input): `k` experts corresponding to features in each row of the output of [aclnnMoeGatingTopKSoftmax](../../../moe/moe_gating_top_k_softmax/docs/aclnnMoeGatingTopKSoftmax_en.md). This tensor must be 2D with shape \(NUM\_ROWS, K\). The data type can be INT32. The [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported.
    - `activeNum` (int64\_t, computation input): maximum number of rows to be processed, which is greater than or equal to 0. Only this many rows in `expandedXOut` are valid.
    - `scale` (double, computation input): required for quantization computation. The typical value is $S=\frac{R_{\max }-R_{\min }}{Q_{\max }-Q_{\min }}$, where $R_{max}$ indicates the maximum floating-point number, $R_{min}$ indicates the minimum floating-point number, $Q_{max}$ indicates the maximum fixed-point number, and $Q_{min}$ indicates the minimum fixed-point number.
    - `offset` (double, computation input): required for quantization computation.
    - `expandedXOut` (aclTensor\*, compute output): 2D tensor, indicating features extended based on `expertIdx`. The shape is \(min\(NUM\_ROWS, activeNum\) \* k, H\). The data type is INT8. The [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are not supported.
    - `expandedRowIdxOut` (aclTensor\*, compute output): mapping between `expandedX` and `x`. It must be a 1D tensor with shape \(NUM\_ROWS\*K, \). The data type can be INT32. The [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are not supported.
    - `expandedExpertIdxOut` (aclTensor\*, compute output): sorted expertIdx output. The data type can be INT32. The [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are not supported.
    - `workspaceSize` (uint64\_t\*, output): size of the workspace to be allocated on the device.
    - `executor` (aclOpExecutor\*\*, output): operator executor, containing the operator computation process.

- **Returns**

    `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

    ```cpp
    The first-phase API implements input parameter verification. The following errors may be thrown.
    161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The input and output tensors are null pointers.
    161002 (ACLNN_ERR_PARAM_INVALID): 1. The input and output data types are not supported.
    561002 (ACLNN_ERR_INNER_TILING_ERROR): 1. The shape of `x` is not 2D.
                                          2. The shape of `rowIdx` must be 2D, and the shapes of `rowIdx` and `expertIdx` must match.
                                          3. The value of `activeNum` is less than 0.
                                          4. The shape of `expandedXOut` is not equal to (min(num_rows, activeNum) × k, H).
                                          5. The shape of `expandedRowIdxOut` is different from that of `expandedExpertIdxOut` and is not equal to (num_rows × k, ).
    ```

## aclnnMoeInitRoutingQuant

- **Parameters:**
    - `workspace` (void*, input): address of the workspace to be allocated on the device.
    - `workspaceSize` (uint64\_t, input): size of the workspace to be allocated on the device, which is obtained by calling `aclnnMoeInitRoutingQuantGetWorkspaceSize`.
    - `executor` (aclOpExecutor*, input): operator executor, containing the operator computation process.
    - `stream` (aclrtStream, input): stream for executing the task.

- **Returns**:

    `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computing:
  - The default deterministic implementation of `aclnnMoeInitRoutingQuant` is used.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_init_routing_quant.h"
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
    double scale = 1.0;
    double offset = 1.0;
    aclTensor* expandedXOut = nullptr;
    aclTensor* expandedRowIdxOut = nullptr;
    aclTensor* expandedExpertIdxOut = nullptr;
    std::vector<float> xHostData = {0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3};
    std::vector<int> expertIdxHostData = {1, 2, 0, 1, 0, 2};
    std::vector<int> rowIdxHostData = {0, 3, 1, 4, 2, 5};
    std::vector<int8_t> expandedXOutHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
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
    ret = CreateAclTensor(expandedXOutHostData, expandedXOutShape, &expandedXOutDeviceAddr, aclDataType::ACL_INT8, &expandedXOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandedRowIdxOutHostData, idxOutShape, &expandedRowIdxOutDeviceAddr, aclDataType::ACL_INT32, &expandedRowIdxOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandedExpertIdxOutHostData, idxOutShape, &expandedExpertIdxOutDeviceAddr, aclDataType::ACL_INT32, &expandedExpertIdxOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnMoeInitRoutingQuant.
    ret = aclnnMoeInitRoutingQuantGetWorkspaceSize(x, rowIdx, expertIdx, activeNum, scale, offset, expandedXOut, expandedRowIdxOut, expandedExpertIdxOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeInitRoutingQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnMoeInitRoutingQuant.
    ret = aclnnMoeInitRoutingQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeInitRoutingQuant failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
    auto expandedXSize = GetShapeSize(expandedXOutShape);
    std::vector<int8_t> expandedXData(expandedXSize, 0);
    ret = aclrtMemcpy(expandedXData.data(), expandedXData.size() * sizeof(expandedXData[0]), expandedXOutDeviceAddr, expandedXSize * sizeof(int8_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expandedXSize; i++) {
        LOG_PRINT("expandedXData[%ld] is: %d\n", i, expandedXData[i]);
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

    // 7. Free device resources. Modify the code based on the API definition.
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
