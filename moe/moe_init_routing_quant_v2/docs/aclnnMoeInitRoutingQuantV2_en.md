# aclnnMoeInitRoutingQuantV2

## Product Support

| Product                                                        |  Supported  |
| :----------------------------------------------------------- |:-------:|
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference products</term>                            |    ×    |
| <term>Atlas training products</term>                             |    ×    |

## Function

- **Description**: Performs routing computation for MoE based on the computation result of [aclnnMoeGatingTopKSoftmaxV2](../../../moe/moe_gating_top_k_softmax_v2/docs/aclnnMoeGatingTopKSoftmaxV2_en.md).

  This API has the following function changes based on [aclnnMoeInitRoutingQuant](../../../moe/moe_init_routing_quant/docs/aclnnMoeInitRoutingQuant_en.md). Select a proper API based on your actual requirements.

  - Added the drop mode. In this mode, the output is processed based on `expertCapacity`. If the output exceeds the value of `expertCapacity`, the exceeded part is not processed. If the output falls short of the value of `expertCapacity`, 0s are padded.
  - Added the optional output `expertTokensCountOrCumsumOutOptional` in dropless mode and the optional output `expertTokensBeforeCapacityOutOptional` in drop mode.
  - Deleted the input `rowIdx`.
  - Added the dynamic quantization mode.

- **Formula**:

  1. Sort the input `expertIdx` to obtain the sorted result `sortedExpertIdx` and the corresponding index `sortedRowIdx`.

      $$
      sortedExpertIdx, sortedRowIdx=keyValueSort(expertIdx)
      $$

  2. Use `sortedRowIdx` for location mapping to obtain `expandedRowIdxOut`.

      $$
      expandedRowIdxOut[sortedRowIdx[i]]=i
      $$

  3. In dropless mode, collect statistics on the histogram of each expert in `sortedExpertIdx` and perform Cumsum to obtain `expertTokensCountOrCumsumOutOptional`.

      $$
      expertTokensCountOrCumsumOutOptional[i]=Cumsum(Histogram(sortedExpertIdx))
      $$

  4. In drop mode, collect statistics on the histogram of each expert in `sortedExpertIdx` to obtain `expertTokensBeforeCapacityOutOptional`.

      $$
      expertTokensBeforeCapacityOutOptional[i]=Histogram(sortedExpertIdx)
      $$

  5. Obtain the quantization result.
      - Static quantization:

          $$
          quantResult = round((x * scaleOptional) + offsetOptional)
          $$

      - Dynamic quantization:
          - If `scale` is not specified:

              $$
              dynamicQuantScaleOutOptional = row\_max(abs(x)) / 127
              $$

              $$
              quantResult = round(x / dynamicQuantScaleOutOptional)
              $$

          - If `scale` is specified:

              $$
              dynamicQuantScaleOutOptional = row\_max(abs(x * scaleOptional)) / 127
              $$

              $$
              quantResult = round(x / dynamicQuantScaleOutOptional)
              $$
            
  6. Obtain the values of the first *NUM\_ROWS* `sortedRowIdx` values for `quantResult` to obtain `expandedXOut`.

    $$
    expandedXOut[i]=quantResult[sortedRowIdx[i]\%NUM\_ROWS]
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnMoeInitRoutingQuantV2GetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnMoeInitRoutingQuantV2` is called to perform computation.

* `aclnnStatus aclnnMoeInitRoutingQuantV2GetWorkspaceSize(const aclTensor *x, const aclTensor *expertIdx, const aclTensor *scaleOptional, const aclTensor *offsetOptional, int64_t activeNum, int64_t expertCapacity, int64_t expertNum, int64_t dropPadMode, int64_t expertTokensCountOrCumsumFlag, bool expertTokensBeforeCapacityFlag, int64_t quantMode, const aclTensor *expandedXOut, const aclTensor *expandedRowIdxOut, const aclTensor *expertTokensCountOrCumsumOutOptional, const aclTensor *expertTokensBeforeCapacityOutOptional, const aclTensor *dynamicQuantScaleOutOptional, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnMoeInitRoutingQuantV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnMoeInitRoutingQuantV2GetWorkspaceSize

- **Parameters**:
    - `x` (aclTensor\*, computation input): MoE input, that is, token feature input. It must be a 2D tensor with shape [NUM\_ROWS, H], where `H` indicates the length of each token. The data type can be FLOAT16, BFLOAT16, or FLOAT32. The [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported.
    - `expertIdx` (aclTensor\*, computation input): *K* experts corresponding to features in each row of the output of [aclnnMoeGatingTopKSoftmaxV2](../../../moe/moe_gating_top_k_softmax_v2/docs/aclnnMoeGatingTopKSoftmaxV2_en.md). The value must be 2D shape [NUM\_ROWS, K]. The data type can be INT32. The [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. In drop/pad scenarios or when `expertTokensCountOrCumsumOutOptional` needs to be output in drop/pad-less scenarios, the value range must be [0, expertNum – 1]. In other scenarios, the value must be greater than or equal to `0`.
    - `scaleOptional` (aclTensor\*, computation input): used to compute the quantization result. This parameter is optional, but mandatory in static quantization scenarios. The value is a 1D shape [1,]. In dynamic quantization scenarios, if this parameter is not set, `scale` is not used during computation. If this parameter is set, the value must be a 2D tensor with shape [expertNum, H] or [1, H]. The data type can be FLOAT32. The [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported.
    - `offsetOptional` (aclTensor\*, computation input): calculates the offset of the quantization result. This parameter is optional, but mandatory in static quantization scenarios. The value is a 1D shape [1,]. The data type can be FLOAT32. The [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported.
    - `activeNum` (int64\_t, computation input): indicates whether the active scenario is involved. This attribute is valid only when `dropPadMode` is set to `0`. The value must be greater than or equal to `0`. The value `0` indicates the dropless scenario. A value greater than `0` indicates the active scenario, which restricts the total number of tokens processed by all experts.
    - `expertCapacity` (int64\_t, computation input): number of tokens that can be processed by each expert. The value is greater than or equal to 0. In the drop/pad scenario, the value range is \(0, NUM\_ROWS\]. In this case, experts drop the tokens that exceed the capacity threshold. If the capacity threshold is not reached, pad all-zero tokens. In other scenarios, this attribute value is not concerned.
    - `expertNum` (int64\_t, computation input): number of experts. The value is greater than or equal to `0`. In the drop/pad scenario or when the value of `expertTokensCountOrCumsumFlag` is greater than `0` and the output `expertTokensCountOrCumsumOutOptional` is required, the value of `expertNum` must be greater than `0`.
    - `dropPadMode` (int64\_t, computation input): indicates whether the scenario is in drop/pad mode. The values are `0` or `1`.
        - `0`: drop/pad-less scenario where `expertCapacity` is not verified.
        - `1`: drop/pad scenario where `expertNum` and `expertCapacity` need to be verified. The corresponding measure will be taken when the number of tokens that can be processed by each expert exceeds or falls short of the `expertCapacity` value.
    - `expertTokensCountOrCumsumFlag` (int64\_t, computation input): The value can be `0`, `1`, or `2`.
        - `0`: `expertTokensCountOrCumsumOutOptional` is not output.
        - `1`: The output value is the cumulative number of tokens processed by each expert.
        - `2`: The output value is the number of tokens processed by each expert.
    - `expertTokensBeforeCapacityFlag` (bool, computation input): The value can be `false` or `true`.
        - `false`: `expertTokensBeforeCapacityOutOptional` is not output.
        - `true`: The output value is the number of tokens processed by each expert before the drop operation.
    - `quantMode` (int64\_t, computation input): The values are `0` or `1`.
        - `0`: static quantization scenario.
        - `1`: dynamic quantization scenario.
    - `expandedXOut` (aclTensor\*, computation output): feature expanded based on `expertIdx`. In the dropless/active scenario, it must be a 2D tensor. In the dropless scenario, the shape is [NUM\_ROWS \* K, H]. In the active scenario, the shape is [min\(activeNum, NUM\_ROWS \* K\), H]. In the drop/pad scenario, the value must be a 3D tensor with shape [expertNum, expertCapacity, H]. The data type is INT8. The [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are not supported.
    - `expandedRowIdxOut` (aclTensor\*, computation output): index mapping between `expandedXOut` and `x`. The value must be a 1D tensor with shape [NUM\_ROWS\*K, ]. The data type is INT32. The [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are not supported.
    - `expertTokensCountOrCumsumOutOptional` (aclTensor\*, computation output): outputs the statistics and cumulative value of the number of tokens processed by each expert. This output is optional. The `expertTokensCountOrCumsumFlag` parameter determines whether to output the value. The value is output only in drop/pad-less scenarios and must be a 1D tensor with shape [expertNum, ]. The data type can be INT32. The [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are not supported.
    - `expertTokensBeforeCapacityOutOptional` (aclTensor\*, computation output): outputs the statistics on the number of tokens processed by each expert before the drop operation. This output is optional. The `expertTokensBeforeCapacityFlag` parameter determines whether to output the value. The value is output only in drop/pad scenarios and must be a 1D tensor with shape [expertNum, ]. The data type can be INT32. The [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are not supported.
    - `dynamicQuantScaleOutOptional` (aclTensor\*, computation output): outputs the intermediate value during dynamic quantization. This output is optional. The value is output only in dynamic quantization scenarios, and must be a 1D tensor. The shape is the product of all dimensions except the last dimension of `expandedXOut`. The data type can be FLOAT32. The [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are not supported.
    - `workspaceSize` (uint64\_t\*, output): size of the workspace to be allocated on the device.
    - `executor` (aclOpExecutor\*\*, output): operator executor, containing the operator computation process.

- **Returns**

    `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

    ```cpp
    The first-phase API implements input parameter verification. The following errors may be thrown:
    161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The computation input or required computation output is a null pointer.
    161002 (ACLNN_ERR_PARAM_INVALID): 1. The data types and formats of the required computation input and output are not supported.
    561002 (ACLNN_ERR_INNER_TILING_ERROR): 1. The shape of x and expertIdx is not 2D, and their first dimensions are not equal.
                                          2. activeNum, expertNum, or expertCapacity is less than 0.
                                          3. dropPadMode, expertTokensCountOrCumsumFlag, expertTokensBeforeCapacityFlag, or quantMode is not within the value range.
                                          4. When dropPadMode is set to 1, expertCapacity and expertNum are set to 0.
                                          5. When expertTokensCountOrCumsumOutOptional needs to be output, expertNum is set to 0.
                                          6. The data types of optional inputs and outputs are not supported.
    ```

## aclnnMoeInitRoutingQuantV2

- **Parameters:**
    - `workspace` (void\*, input): address of the workspace to be allocated on the device.
    - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnMoeInitRoutingQuantV2GetWorkspaceSize`.
    - `executor` (aclOpExecutor\*, input): operator executor, containing the operator computation process.
    - `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnMoeInitRoutingQuantV2` defaults to deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_init_routing_quant_v2.h"
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
    // 2. Construct inputs and outputs based on the API definition.
    std::vector<int64_t> xShape = {3, 4};
    std::vector<int64_t> idxShape = {3, 2};
    std::vector<int64_t> scaleShape = {1};
    std::vector<int64_t> expandedXOutShape = {3, 2, 4};
    std::vector<int64_t> idxOutShape = {6};
    std::vector<int64_t> expertTokenOutShape = {3};
    std::vector<int64_t> dynamicQuantScaleOutOptionalShape = {6};
    void* xDeviceAddr = nullptr;
    void* expertIdxDeviceAddr = nullptr;
    void* scaleDeviceAddr = nullptr;
    void* offsetDeviceAddr = nullptr;
    void* expandedXOutDeviceAddr = nullptr;
    void* expandedRowIdxOutDeviceAddr = nullptr;
    void* expertTokenBeforeCapacityOutDeviceAddr = nullptr;
    void* dynamicQuantScaleOutOptionalDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* expertIdx = nullptr;
    aclTensor* scale = nullptr;
    aclTensor* offset = nullptr;
    int64_t activeNum = 0;
    int64_t expertCapacity = 2;
    int64_t expertNum = 3;
    int64_t dropPadMode = 1;
    int64_t expertTokensCountOrCumsumFlag = 0;
    bool expertTokensBeforeCapacityFlag = true;
    int64_t quantMode = 0;
    aclTensor* expandedXOut = nullptr;
    aclTensor* expandedRowIdxOut = nullptr;
    aclTensor* expertTokensBeforeCapacityOutOptional = nullptr;
    aclTensor* dynamicQuantScaleOutOptional = nullptr;
    std::vector<float> xHostData = {0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3};
    std::vector<int> expertIdxHostData = {1, 2, 0, 1, 0, 2};
    std::vector<float> scaleHostData = {0.3452};
    std::vector<float> offsetHostData = {1.8369};
    std::vector<int8_t> expandedXOutHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int> expandedRowIdxOutHostData = {0, 0, 0, 0, 0, 0};
    std::vector<int> expertTokensBeforeCapacityOutOptionalHostData = {0, 0, 0};
    std::vector<float> dynamicQuantScaleOutOptionalHostData = {0, 0, 0, 0, 0, 0};
    // Create a self aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertIdxHostData, idxShape, &expertIdxDeviceAddr, aclDataType::ACL_INT32, &expertIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(offsetHostData, scaleShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an out aclTensor.
    ret = CreateAclTensor(expandedXOutHostData, expandedXOutShape, &expandedXOutDeviceAddr, aclDataType::ACL_INT8, &expandedXOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandedRowIdxOutHostData, idxOutShape, &expandedRowIdxOutDeviceAddr, aclDataType::ACL_INT32, &expandedRowIdxOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertTokensBeforeCapacityOutOptionalHostData, expertTokenOutShape, &expertTokenBeforeCapacityOutDeviceAddr, aclDataType::ACL_INT32, &expertTokensBeforeCapacityOutOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dynamicQuantScaleOutOptionalHostData, dynamicQuantScaleOutOptionalShape, &dynamicQuantScaleOutOptionalDeviceAddr, aclDataType::ACL_FLOAT, &dynamicQuantScaleOutOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnMoeInitRoutingQuantV2.
    ret = aclnnMoeInitRoutingQuantV2GetWorkspaceSize(x, expertIdx, scale, offset, activeNum, expertCapacity, expertNum, dropPadMode, expertTokensCountOrCumsumFlag, expertTokensBeforeCapacityFlag, quantMode, expandedXOut, expandedRowIdxOut, nullptr, expertTokensBeforeCapacityOutOptional, dynamicQuantScaleOutOptional, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeInitRoutingQuantV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnMoeInitRoutingQuantV2.
    ret = aclnnMoeInitRoutingQuantV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeInitRoutingQuantV2 failed. ERROR: %d\n", ret); return ret);
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
    auto expertTokensBeforeCapacitySize = GetShapeSize(expertTokenOutShape);
    std::vector<int> expertTokenIdxData(expertTokensBeforeCapacitySize, 0);
    ret = aclrtMemcpy(expertTokenIdxData.data(), expertTokenIdxData.size() * sizeof(expertTokenIdxData[0]), expertTokenBeforeCapacityOutDeviceAddr, expertTokensBeforeCapacitySize * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expertTokensBeforeCapacitySize; i++) {
        LOG_PRINT("expertTokenIdxData[%ld] is: %d\n", i, expertTokenIdxData[i]);
    }

    auto dynamicQuantScaleSize = GetShapeSize(dynamicQuantScaleOutOptionalShape);
    std::vector<float> dynamicQuantScaleData(dynamicQuantScaleSize, 0);
    ret = aclrtMemcpy(dynamicQuantScaleData.data(), dynamicQuantScaleData.size() * sizeof(dynamicQuantScaleData[0]), dynamicQuantScaleOutOptionalDeviceAddr, dynamicQuantScaleSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < dynamicQuantScaleSize; i++) {
        LOG_PRINT("dynamicQuantScaleData[%ld] is: %f\n", i, dynamicQuantScaleData[i]);
    }
    // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
    aclDestroyTensor(x);
    aclDestroyTensor(expertIdx);
    aclDestroyTensor(scale);
    aclDestroyTensor(offset);
    aclDestroyTensor(expandedXOut);
    aclDestroyTensor(expandedRowIdxOut);
    aclDestroyTensor(expertTokensBeforeCapacityOutOptional);
    aclDestroyTensor(dynamicQuantScaleOutOptional);

    // 7. Release device resources. Modify the code based on the API definition.
    aclrtFree(xDeviceAddr);
    aclrtFree(expertIdxDeviceAddr);
    aclrtFree(scaleDeviceAddr);
    aclrtFree(offsetDeviceAddr);
    aclrtFree(expandedXOutDeviceAddr);
    aclrtFree(expandedRowIdxOutDeviceAddr);
    aclrtFree(expertTokenBeforeCapacityOutDeviceAddr);
    aclrtFree(dynamicQuantScaleOutOptionalDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
