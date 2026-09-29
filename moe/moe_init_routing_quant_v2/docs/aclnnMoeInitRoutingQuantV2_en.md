# aclnnMoeInitRoutingQuantV2

[📄 View Source Code](https://gitcode.com/cann/ops-transformer/tree/master/moe/moe_init_routing_quant_v2)

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

- **Description**: This operator corresponds to the **routing computation** in the MoE model. It uses the computation result of the [aclnnMoeGatingTopKSoftmax](../../moe_gating_top_k_softmax/docs/aclnnMoeGatingTopKSoftmax_en.md) operator as the input and outputs the quantized routing matrix, `expandedXOut`, and other results for subsequent computation. This API has the following function changes based on [aclnnMoeInitRoutingQuant](../../moe_init_routing_quant/docs/aclnnMoeInitRoutingQuant_en.md). Select a proper API based on your actual requirements.

  - Added the drop mode. In this mode, the output is based on the number of tokens processed by each expert and equal to the value of·`expertCapacity`. If the output exceeds the value, exceeded tokens are dropped. If the output falls short of the value, 0s are padded.
  - Added the optional output `expertTokensCountOrCumsumOut` in dropless mode. The output is based on the number of cumulative tokens (Cumsum) to be processed by each expert or the number of tokens (Count) to be processed by each expert.
  - Added the optional output `expertTokensBeforeCapacityOut` in drop mode. The output is based on the number of tokens to be processed by each expert before the drop operation.
  - Deleted the input `rowIdx`.
  - Added the dynamic quantization mode.

- **Formula**:

  1. Flatten the input expertIdx with shape [NUM_ROWS, K] into a row for sorting, where NUM_ROWS indicates the number of input tokens and K indicates the number of experts selected for tokens. The sortedExpertIdx and corresponding sortedRowIdx are obtained.
      
      $$
      sortedExpertIdx, sortedRowIdx=keyValueSort(\text{flatten}(expertIdx))
      $$

  2. Use `sortedRowIdx` for location mapping to obtain `expandedRowIdxOut`.

      $$
      expandedRowIdxOut[sortedRowIdx[i]]=i
      $$

  3. In dropless mode, collect statistics on the histogram of each expert in `sortedExpertIdx` and perform Cumsum to obtain `expertTokensCountOrCumsumOutOptional`.

      $$
      expertTokensCountOrCumsumOutOptional[i]=Cumsum(Histogram(sortedExpertIdx))
      $$

  4. In Drop mode, collect the histogram result of each expert in sortedExpertIdx to obtain expertTokensBeforeCapacityOutOptional.

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
          quantResult = round(x * scaleOptional / dynamicQuantScaleOutOptional)
          $$

  6. Obtain expandedXOut based on quantResult.

  $$
  expandedXOut[expandedRowIdxOut[i]]=quantResult[i // K]
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnMoeInitRoutingQuantV2GetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnMoeInitRoutingQuantV2` is called to perform computation.

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
  <table style="undefined;table-layout: fixed; width: 1575px"><colgroup>
  <col style="width: 260px">
  <col style="width: 120px">
  <col style="width: 242px">
  <col style="width: 399px">
  <col style="width: 160px">
  <col style="width: 115px">
  <col style="width: 134px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage</th>
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
      <td><ul><li>Empty tensors are supported. </li><li>The value must be a 2D tensor with shape [NUM_ROWS, H]. NUM_ROWS indicates the number of tokens, and H indicates the length of each token.</li></ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>expertIdx</td>
      <td>Input</td>
      <td>Output of aclnnMoeGatingTopKSoftmaxV2. It indicates the K processed experts corresponding to each row of features.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The value must be a 2D tensor with shape [NUM_ROWS, K]. </li><li>In the Drop/Pad scenario or when expertTokensCountOrCumsumOutOptional needs to be output in the non-Drop/Pad scenario, the value range is [0, expertNum – 1]. In other scenarios, the value must be greater than or equal to 0.</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scaleOptional</td>
      <td>Input</td>
      <td>Parameter used to calculate the quantization result.</td>
      <td><ul><li>Empty tensors are supported. </li><li>This parameter is mandatory in static quantization scenarios. The value is a 1D tensor with shape [1].</li> <li>In dynamic quantization scenarios, if this parameter is not specified, no scale is used during computation. If this parameter is specified, it must be a 2D tensor with the shape of [expertNum, H] or [1, H].</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1 or 2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>offsetOptional</td>
      <td>Input</td>
      <td>Offset value used to calculate the quantization result.</td>
      <td><ul><li>Empty tensors are supported. </li><li>In static quantization scenarios, this parameter is required and must be a 1D tensor with the shape of [1].</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>activeNum</td>
      <td>Input</td>
      <td>Indicates whether the active scenario is involved.</td>
      <td>This attribute takes effect when <code>dropPadMode</code> is set to <code>0</code>. The value must be greater than or equal to <code>0</code>. The value <code>0</code> indicates the dropless scenario, and a value greater than <code>0</code> indicates the active scenario where all experts are required to process the total number of tokens.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertCapacity</td>
      <td>Input</td>
      <td>Number of tokens that can be processed by each expert.</td>
      <td>The value is greater than or equal to 0. In the Drop/Pad scenario, the value range is (0, NUM_ROWS]. In this case, experts drop tokens that exceed the capacity. If the capacity is less than the threshold, pad all zero tokens. The value of this attribute is not concerned in other scenarios.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertNum</td>
      <td>Input</td>
      <td>Number of experts.</td>
      <td>The value must be greater than or equal to 0. In the Drop/Pad scenario or when expertTokensCountOrCumsumFlag is greater than 0 and expertTokensCountOrCumsumOutOptional needs to be output, expertNum must be greater than 0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dropPadMode</td>
      <td>Input</td>
      <td>Indicates whether the scenario is in drop/pad mode.</td>
      <td>The values are <code>0</code> and <code>1</code>. <ul><li><code>0</code>: indicates drop/pad-less scenario where <code>expertCapacity</code> is not verified. </li><li><code>1</code>: indicates the drop/pad scenario where <code>expertNum</code> and <code>expertCapacity</code> need to be verified. The corresponding measure will be taken when the number of tokens that can be processed by each expert exceeds or falls short of the <code>expertCapacity</code> value.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertTokensCountOrCumsumFlag</td>
      <td>Input</td>
      <td>Controls whether to output expertTokensCountOrCumsumOutOptional.</td>
      <td>The value can be <code>0</code>, <code>1</code>, or <code>2</code>. <ul><li>0: expertTokensCountOrCumsumOutOptional is not output. </li><li><code>1</code>: The output value is the cumulative sum of the number of tokens processed by each expert. </li><li><code>2</code>: The output value is the number of tokens processed by each expert.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertTokensBeforeCapacityFlag</td>
      <td>Input</td>
      <td>Controls whether to output expertTokensBeforeCapacityOutOptional.</td>
      <td>The value can be false or true. <ul><li>false: expertTokensBeforeCapacityOutOptional is not output. </li><li>true: The output value is the number of tokens processed by each expert before the drop.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode</td>
      <td>Input</td>
      <td>Indicates the quantization mode.</td>
      <td>The values are <code>0</code> and <code>1</code>. <ul><li>0: static quantization scenario. </li><li>1: dynamic quantization scenario.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expandedXOut</td>
      <td>Output</td>
      <td>Extended feature based on `expertIdx`.</td>
      <td><ul><li>Empty tensors are supported. </li><li>In the Dropless/Active scenario, the tensor must be a 2D tensor. In the Dropless scenario, the shape is [NUM_ROWS * K, H]. In the Active scenario, the shape is [min(activeNum, NUM_ROWS * K), H]. </li><li>In the Drop/Pad scenario, the tensor must be a 3D tensor. The shape is [expertNum, expertCapacity, H]. </li><li>The data type can be INT8 or INT4. When the data type of expandedXOut is INT4, the following restrictions are imposed: dropPadMode can only be 0; H must be exactly divided by 2; when scaleOptional is passed, the shape of scaleOptional can only be s = [1, H].</li></ul></td>
      <td>INT8, INT4</td>
      <td>ND</td>
      <td>2 or 3</td>
      <td>×</td>
    </tr>
    <tr>
      <td>expandedRowIdxOut</td>
      <td>Output</td>
      <td>Index mapping between <code>expandedXOut</code> and <code>x</code>.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The value must be a 1D tensor with the shape of [NUM_ROWS*K].</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>expertTokensCountOrCumsumOutOptional</td>
      <td>Output</td>
      <td>Statistics result and cumulative sum of the number of tokens processed by each expert.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The expertTokensCountOrCumsumFlag parameter is used to determine whether to output the value. This parameter is output only in the non-drop/pad scenario. It must be a 1D tensor with the shape of [expertNum].</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>expertTokensBeforeCapacityOutOptional</td>
      <td>Output</td>
      <td>Statistics result of the number of tokens processed by each expert before the drop operation.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The expertTokensBeforeCapacityFlag parameter is used to determine whether to output the value. This parameter is output only in the drop/pad scenario. It must be a 1D tensor with the shape of [expertNum].</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>dynamicQuantScaleOutOptional</td>
      <td>Output</td>
      <td>Intermediate value during dynamic quantization computation.</td>
      <td><ul><li>Empty tensors are supported. </li><li>This value is output only in the dynamic quantization scenario. It must be a 1D tensor, and the shape is the product of all dimensions except the last dimension of the expandedXOut shape.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
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
  
  - Ascend 950PR/Ascend 950DT: The output expandedXOut data type supports only INT8.

- **Returns:**

  aclnnStatus: status code. For details, see <a href="../../../docs/en/context/aclnn_return_code.md">aclnn Return Code</a>.

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
        <td>The input and required output for computation are null pointers.</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td>The data types and formats of the computation input and output are not supported.</td>
      </tr>
      <tr>
        <td rowspan="6">ACLNN_ERR_INNER_TILING_ERROR</td>
        <td rowspan="6">561002</td>
        <td>The shape dimensions of <code>x</code> and <code>expertIdx</code> are not equal to <code>2</code>, and their first dimensions are not equal. When the data type of expandedXOut is INT4, dropPadMode is not 0 or shape is not [1, H] when scaleOptional is input.</td>
      </tr>
      <tr>
        <td>The values of <code>activeNum</code>, <code>expertNum</code>, and <code>expertCapacity</code> are less than <code>0</code>.</td>
      </tr>
      <tr>
        <td>The values of dropPadMode, expertTokensCountOrCumsumFlag, expertTokensBeforeCapacityFlag, and quantMode are not supported.</td>
      </tr>
      <tr>
        <td>When <code>dropPadMode</code> is set to <code>1</code>, <code>expertCapacity</code> and <code>expertNum</code> are set to <code>0</code>.</td>
      </tr>
      <tr>
        <td>When expertTokensCountOrCumsumOutOptional needs to be output, expertNum is 0.</td>
      </tr>
      <tr>
        <td>The optional input and output data types are not supported.</td>
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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnMoeInitRoutingQuantV2GetWorkspaceSize`.</td>
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
  - `aclnnMoeInitRoutingQuantV2` defaults to deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

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
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
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
