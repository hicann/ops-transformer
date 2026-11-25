# aclnnMoeInitRoutingQuantV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:17:53.316Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
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

- **Description**: This operator corresponds to the **Routing computation** in MoE (Mixture of Experts). It takes the computation result of the [aclnnMoeGatingTopKSoftmax](../../moe_gating_top_k_softmax/docs/aclnnMoeGatingTopKSoftmax_en.md) operator as input, and outputs the quantized Routing matrix expandedXOut and other results for subsequent computation. This API has the following function changes compared with [aclnnMoeInitRoutingQuant](../../moe_init_routing_quant/docs/aclnnMoeInitRoutingQuant_en.md). Choose a proper API based on actual requirements:

    - A Drop mode is newly added. In Drop mode, the count of tokens to be processed by each expert in the output is aligned to expertCapacity. Tokens exceeding expertCapacity are dropped, and the insufficient part is padded with 0.
    - Newly added optional output expertTokensCountOrCumsumOut in Dropless mode, which outputs the cumulative token count (Cumsum) or the token count (Count) that each expert needs to process.
    - Newly added optional output expertTokensBeforeCapacityOut in Drop mode, which outputs the count of tokens each expert should process before Drop.
    - Removed the rowIdx input.
    - Added the dynamic quant computation mode.

- **Formula**:

    1.Flatten the expertIdx input with shape [NUM_ROWS, K] into one row for sorting, where NUM_ROWS is the count of input tokens and K is the count of experts selected by each token, to derive the sorted result sortedExpertIdx and the corresponding indices sortedRowIdx:

    $$
    sortedExpertIdx, sortedRowIdx=keyValueSort(\text{flatten}(expertIdx))
    $$

    2.Perform position mapping based on sortedRowIdx to derive expandedRowIdxOut:

    $$
    expandedRowIdxOut[sortedRowIdx[i]]=i
    $$

    3.In dropless mode, compute the histogram result for each expert in sortedExpertIdx, and then perform Cumsum to derive expertTokensCountOrCumsumOutOptional:

    $$
    expertTokensCountOrCumsumOutOptional[i]=Cumsum(Histogram(sortedExpertIdx))
    $$

    4.In Drop mode, compute the histogram result for each expert in sortedExpertIdx to derive expertTokensBeforeCapacityOutOptional:

    $$
    expertTokensBeforeCapacityOutOptional[i]=Histogram(sortedExpertIdx)
    $$

    5.Compute the quant result:
    - static quant:

        $$
        quantResult = round((x * scaleOptional) + offsetOptional)
        $$

        <!-- npu="A3,910b" id8 -->
        On <term>Atlas A2 products</term> and <term>Atlas A3 products</term>, when quantMode is 0 and x is FLOAT32, static quantization performs multiplication and addition in FLOAT16 before rounding to INT8.
        <!-- end id8 -->

    - dynamic quant:
        - If scale is not provided:

            $$
            dynamicQuantScaleOutOptional = row\_max(abs(x)) / 127
            $$

            $$
            quantResult = round(x / dynamicQuantScaleOutOptional)
            $$

        - If scale is provided:

            $$
            dynamicQuantScaleOutOptional = row\_max(abs(x * scaleOptional)) / 127
            $$

            $$
            quantResult = round(x * scaleOptional / dynamicQuantScaleOutOptional)
            $$

    6.Derive expandedXOut from quantResult:

    $$
    expandedXOut[expandedRowIdxOut[i]]=quantResult[i // K]
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnMoeInitRoutingQuantV2GetWorkspaceSize" first to obtain the workspace size required for the computation and the executor that includes the computation process of the operator, and then call "aclnnMoeInitRoutingQuantV2" to execute the computation.

```cpp
aclnnStatus aclnnMoeInitRoutingQuantV2GetWorkspaceSize(
    const aclTensor  *x,
    const aclTensor  *expertIdx,
    const aclTensor  *scaleOptional,
    const aclTensor  *offsetOptional,
    int64_t           activeNum,
    int64_t           expertCapacity,
    int64_t           expertNum,
    int64_t           dropPadMode,
    int64_t           expertTokensCountOrCumsumFlag,
    bool              expertTokensBeforeCapacityFlag,
    int64_t           quantMode,
    const aclTensor  *expandedXOut,
    const aclTensor  *expandedRowIdxOut,
    const aclTensor  *expertTokensCountOrCumsumOutOptional,
    const aclTensor  *expertTokensBeforeCapacityOutOptional,
    const aclTensor  *dynamicQuantScaleOutOptional,
    uint64_t         *workspaceSize,
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnMoeInitRoutingQuantV2(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream       stream)
```

## aclnnMoeInitRoutingQuantV2GetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 350px">
    <col style="width: 120px">
    <col style="width: 300px">
    <col style="width: 550px">
    <col style="width: 212px">
    <col style="width: 100px">
    <col style="width: 190px">
    <col style="width: 145px">
    </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dim (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>MoE input, that is, the token feature input.</td>
      <td><ul><li>Supports empty tensors.</li><li>Must be a 2D tensor with shape [NUM_ROWS, H], where NUM_ROWS indicates the token count and H indicates the length of each token.</li></ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>expertIdx</td>
      <td>Input</td>
      <td>Output of aclnnMoeGatingTopKSoftmaxV2, indicating the K experts for processing each feature row.</td>
      <td><ul><li>Supports empty tensor.</li><li>The requirement is a 2D shape [NUM_ROWS, K].</li><li>In the Drop/Pad scenario, or in the non-Drop/Pad scenario where expertTokensCountOrCumsumOutOptional needs to be output, the value range must be [0, expertNum - 1]. In other scenarios, the value must be greater than or equal to 0.</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scaleOptional</td>
      <td>Input</td>
      <td>Parameter used for computing the quant result.</td>
      <td><ul><li>Supports empty tensor.</li><li>Mandatory in the static quant scenario, with a 1D shape [1].</li><li>In the dynamic quant scenario, if this input is not provided, it indicates that scale is not used during computation; if provided, it must be a 2D Tensor with shape [expertNum, H] or [1, H].</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1 or 2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>offsetOptional</td>
      <td>Input</td>
      <td>Offset value used to compute the quant result.</td>
      <td><ul><li>Supports empty tensor.</li><li>Required in static quant scenario, with a 1D shape [1].</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>activeNum</td>
      <td>Input</td>
      <td>Indicates whether it is an Active scenario.</td>
      <td>This attribute takes effect when dropPadMode is 0. The value range is greater than or equal to 0; 0 indicates the Dropless scenario, and a value greater than 0 indicates the Active scenario, which constrains the total number of tokens processed by all experts together.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertCapacity</td>
      <td>Input</td>
      <td>Indicates the count of tokens that each expert can process.</td>
      <td>The value range is greater than or equal to 0; in the Drop/Pad scenario, the value range is (0, NUM_ROWS], that is, the value must be greater than 0 and not exceed NUM_ROWS; in other scenarios, this attribute value is ignored.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertNum</td>
      <td>Input</td>
      <td>Indicates the number of experts.</td>
      <td>The value range is greater than or equal to 0. In the Drop/Pad scenario, or when expertTokensCountOrCumsumFlag is greater than 0 and you need to output expertTokensCountOrCumsumOutOptional, expertNum must be greater than 0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dropPadMode</td>
      <td>Input</td>
      <td>Indicates whether it is a Drop/Pad scenario.</td>
      <td>The value is 0 or 1.<ul><li>0: indicates a non-Drop/Pad scenario. In this scenario, expertCapacity is not verified.</li><li>1: indicates the Drop/Pad scenario. expertNum and expertCapacity need to be verified. For each expert, values exceeding or falling short of expertCapacity are processed accordingly.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertTokensCountOrCumsumFlag</td>
      <td>Input</td>
      <td>Controls whether to output expertTokensCountOrCumsumOutOptional.</td>
      <td>The value is 0, 1, or 2.<ul><li>0: indicates that expertTokensCountOrCumsumOutOptional is not output.</li><li>1: indicates that the output value is the cumulative count of tokens processed by each expert.</li><li>2: indicates that the output value is the count of tokens processed by each expert.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertTokensBeforeCapacityFlag</td>
      <td>Input</td>
      <td>Controls whether to output expertTokensBeforeCapacityOutOptional.</td>
      <td>The value is false or true.<ul><li>false: indicates that expertTokensBeforeCapacityOutOptional is not output.</li><li>true: indicates that the output value is the number of tokens processed by each expert before drop.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode</td>
      <td>Input</td>
      <td>Indicates the quant computation mode.</td>
      <td>The value is 0 or 1.<ul><li>0: indicates the static quant scenario.</li><li>1: indicates the dynamic quant scenario.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expandedXOut</td>
      <td>Output</td>
      <td>Features expanded based on expertIdx.</td>
      <td><ul><li>Supports empty tensors.</li><li>In Dropless/Active scenarios, must be a 2D tensor. The shape is [NUM_ROWS * K, H] in Dropless scenario and [min(activeNum, NUM_ROWS * K), H] in Active scenario.</li><li>In Drop/Pad scenarios, must be a 3D tensor with shape [expertNum, expertCapacity, H].</li><li>Supports the INT8 and INT4 data types. If the data type of expandedXOut is INT4, the following restrictions apply: dropPadMode supports only 0; H must be divisible by 2; when scaleOptional is passed in, the shape of scaleOptional can only be [1, H].</li></ul></td>
      <td>INT8, INT4</td>
      <td>ND</td>
      <td>2 or 3</td>
      <td>×</td>
    </tr>
    <tr>
      <td>expandedRowIdxOut</td>
      <td>Output</td>
      <td>Index mapping between expandedXOut and x.</td>
      <td><ul><li>Supports empty tensors.</li><li>Must be a 1D tensor with shape [NUM_ROWS*K].</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>expertTokensCountOrCumsumOutOptional</td>
      <td>Output</td>
      <td>Outputs the statistics and cumulative values of the token count in each expert handling.</td>
      <td><ul><li>Empty tensors are supported.</li><li>Whether to output is controlled by the expertTokensCountOrCumsumFlag parameter. This value is only output in non-Drop/Pad scenarios. The requirement is a 1D Tensor with shape [expertNum].</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>expertTokensBeforeCapacityOutOptional</td>
      <td>Output</td>
      <td>Outputs statistics on the count of tokens processed by each expert before drop.</td>
      <td><ul><li>Supports empty tensors.</li><li>Whether to output is controlled by the expertTokensBeforeCapacityFlag parameter. This value is output only in the Drop/Pad scenario. It is required to be a 1D tensor with a shape of [expertNum].</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>dynamicQuantScaleOutOptional</td>
      <td>Output</td>
      <td>Outputs intermediate values in the dynamic quant computation.</td>
      <td><ul><li>Supports empty tensor.</li><li>This value is only output in the dynamic quant scenario. It must be a 1D Tensor, and its Shape is the product of all dimensions of expandedXOut except the last one.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the size of the workspace that the user needs to allocate on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, which includes the computation flow of the operator.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>: The output expandedXOut data type supports only INT8.

  <!-- end id7 -->

- **Return value**

  `aclnnStatus`: Return status code. For details, see <a href="../../../docs/en/context/aclnn_return_code.md">aclnn return code</a>.

  The first-phase API performs input parameter validation and returns an error in the following scenarios:
  <table style="undefined;table-layout: fixed; width: 1180px">
    <colgroup>
      <col style="width: 250px">
      <col style="width: 130px">
      <col style="width: 800px">
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
        <td>The computation input and required computation output are null pointers.</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td>The data types and formats of the computation input and output are not within the supported range.</td>
      </tr>
      <tr>
        <td rowspan="6">ACLNN_ERR_INNER_TILING_ERROR</td>
        <td rowspan="6">561002</td>
        <td>The shape dimensions of x and expertIdx are not equal to 2, and the first dimensions are not equal. When the data type of expandedXOut is INT4, if dropPadMode is not 0 or scaleOptional is input, the shape is not [1, H].</td>
      </tr>
      <tr>
        <td>The values of activeNum, expertNum, and expertCapacity are less than 0.</td>
      </tr>
      <tr>
        <td>dropPadMode, expertTokensCountOrCumsumFlag, expertTokensBeforeCapacityFlag, and quantMode are not within the value range.</td>
      </tr>
      <tr>
        <td>When dropPadMode is equal to 1, expertCapacity and expertNum are equal to 0.</td>
      </tr>
      <tr>
        <td>expertNum is equal to 0 when expertTokensCountOrCumsumOutOptional needs to output.</td>
      </tr>
      <tr>
        <td>The data type of the optional input or output is not within the supported range.</td>
      </tr>
    </tbody>
  </table>

## aclnnMoeInitRoutingQuantV2

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1180px">
  <colgroup>
    <col style="width: 250px">
    <col style="width: 130px">
    <col style="width: 800px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>The workspace memory address applied by the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>The workspace size applied by the Device side, obtained by the first-phase API <code>aclnnMoeInitRoutingQuantV2GetWorkspaceSize</code>.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, includes computation flow of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnMoeInitRoutingQuantV2 defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
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
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device side.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host-side data to the device-side memory.
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
    // 1. Boilerplate for device/stream initialization. For details, see the list of external ACL APIs.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as needed.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the input and output based on the API definition.
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
    // Create the self aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertIdxHostData, idxShape, &expertIdxDeviceAddr, aclDataType::ACL_INT32, &expertIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(offsetHostData, scaleShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the out aclTensor.
    ret = CreateAclTensor(expandedXOutHostData, expandedXOutShape, &expandedXOutDeviceAddr, aclDataType::ACL_INT8, &expandedXOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandedRowIdxOutHostData, idxOutShape, &expandedRowIdxOutDeviceAddr, aclDataType::ACL_INT32, &expandedRowIdxOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertTokensBeforeCapacityOutOptionalHostData, expertTokenOutShape, &expertTokenBeforeCapacityOutDeviceAddr, aclDataType::ACL_INT32, &expertTokensBeforeCapacityOutOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dynamicQuantScaleOutOptionalHostData, dynamicQuantScaleOutOptionalShape, &dynamicQuantScaleOutOptionalDeviceAddr, aclDataType::ACL_FLOAT, &dynamicQuantScaleOutOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. Call the CANN operator library API, which needs to be replaced with the specific API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnMoeInitRoutingQuantV2.
    ret = aclnnMoeInitRoutingQuantV2GetWorkspaceSize(x, expertIdx, scale, offset, activeNum, expertCapacity, expertNum, dropPadMode, expertTokensCountOrCumsumFlag, expertTokensBeforeCapacityFlag, quantMode, expandedXOut, expandedRowIdxOut, nullptr, expertTokensBeforeCapacityOutOptional, dynamicQuantScaleOutOptional, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeInitRoutingQuantV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnMoeInitRoutingQuantV2.
    ret = aclnnMoeInitRoutingQuantV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeInitRoutingQuantV2 failed. ERROR: %d\n", ret); return ret);
    // 4. Boilerplate. Wait synchronously for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output values and copy the results from device memory to host memory, which needs to be modified based on the definition of the specific API.
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
    // 6. Release aclTensor and aclScalar, which needs to be modified based on the definition of the specific API.
    aclDestroyTensor(x);
    aclDestroyTensor(expertIdx);
    aclDestroyTensor(scale);
    aclDestroyTensor(offset);
    aclDestroyTensor(expandedXOut);
    aclDestroyTensor(expandedRowIdxOut);
    aclDestroyTensor(expertTokensBeforeCapacityOutOptional);
    aclDestroyTensor(dynamicQuantScaleOutOptional);

    // 7. Release device resources, which needs to be modified based on the definition of the specific API.
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
