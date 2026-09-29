# aclnnMoeInitRoutingV2

[📄 View Source Code](https://gitcode.com/cann/ops-transformer/tree/master/moe/moe_init_routing_v2)

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  Ascend 950PR/Ascend 950DT  |     √    |
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference products</term>   |     √    |
|  <term>Atlas training products</term>   |     ×    |

## Function

- **Description**: This operator corresponds to the **routing computation** in the MoE model. It uses the computation result of the [aclnnMoeGatingTopKSoftmax](../../moe_gating_top_k_softmax/docs/aclnnMoeGatingTopKSoftmax_en.md) operator as the input and outputs the routing matrix, `expandedXOut`, and other results for subsequent computation. This API has the following function changes based on [aclnnMoeInitRouting](../../moe_init_routing/docs/aclnnMoeInitRouting_en.md). Select a proper API based on your actual requirements.

  - Added the drop mode. In this mode, the output is based on the number of tokens processed by each expert and equal to the value of·`expertCapacity`. If the output exceeds the value, exceeded tokens are dropped. If the output falls short of the value, 0s are padded.
  - Added the optional output `expertTokensCountOrCumsumOut` in dropless mode. The output is based on the number of cumulative tokens (Cumsum) to be processed by each expert or the number of tokens (Count) to be processed by each expert.
  - Added the optional output `expertTokensBeforeCapacityOut` in drop mode. The output is based on the number of tokens to be processed by each expert before the drop operation.
  - Deleted the input `rowIdx`.

  **Note:**
  Routing computation is a phase in an MoE model. The MoE model consists of a group of expert models and a gating model. During computation, the input data is first used to calculate the *k* experts with the highest weights corresponding to each data element based on the gating network (including the MoeGatingTopKSoftmax operator). Then, the result is input to the MoeInitRouting operator to generate the routing matrix. In subsequent operations, each expert in the model processes the data that it should process based on the routing matrix and generates the corresponding output. The outputs of all experts are weighted and summed up to form the final prediction result.
- **Formula**:

  1. Flatten the input `expertIdx` with shape [numRows, k] or [numRows] into a row for sorting, and obtain `sortedExpertIdx` and the corresponding `sortedRowIdx`, where `numRows` indicates the number of tokens, and *k* indicates the number of experts. When `expertIdx` is 1D, *k* is `1`.
      
      $$
      sortedExpertIdx, sortedRowIdx=keyValueSort(\text{flatten}(expertIdx))
      $$

  2. Use `sortedRowIdx` for location mapping to obtain `expandedRowIdxOut`.
      
      $$
      expandedRowIdxOut[sortedRowIdx[i]]=i
      $$
  
  3. Sort tokens by expert in the order of `sortedRowIdx`. When `dropPadMode` is set to `1`, the number of tokens to be processed by each expert is the same as the value of `expertCapacity`. The tokens that exceed the value of `expertCapacity` are dropped, and the tokens that fall short of the value are padded with 0s. The `expandedXOut` is obtained as follows:
  
      $$
      expandedXOut[expandedRowIdxOut[i]]=x[i//k]
      $$
      
  4. Collect statistics on the histogram of each expert in `sortedExpertIdx` and perform Cumsum to obtain `expertTokensCountOrCumsumOut`.
  
      $$
      expertTokensCountOrCumsumOut[i]=Cumsum(Histogram(sortedExpertIdx))
      $$
      
  5. Collect statistics on the histogram of each expert in `sortedExpertIdx` to obtain `expertTokensBeforeCapacityOut`.
  
  $$
  expertTokensBeforeCapacityOut[i]=Histogram(sortedExpertIdx)
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnMoeInitRoutingV2GetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnMoeInitRoutingV2` is called to perform computation.

```cpp
aclnnStatus aclnnMoeInitRoutingV2GetWorkspaceSize(
    const aclTensor  *x, 
    const aclTensor  *expertIdx, 
    int64_t           activeNum, 
    int64_t           expertCapacity, 
    int64_t           expertNum, 
    int64_t           dropPadMode, 
    int64_t           expertTokensCountOrCumsumFlag, 
    bool              expertTokensBeforeCapacityFlag, 
    const aclTensor  *expandedXOut, 
    const aclTensor  *expandedRowIdxOut, 
    const aclTensor  *expertTokensCountOrCumsumOut, 
    const aclTensor  *expertTokensBeforeCapacityOut, 
    uint64_t         *workspaceSize, 
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnMoeInitRoutingV2(
    void             *workspace, 
    uint64_t          workspaceSize, 
    aclOpExecutor    *executor, 
    aclrtStream       stream)
```

## aclnnMoeInitRoutingV2GetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1562px"><colgroup>
    <col style="width: 265px">
    <col style="width: 120px">
    <col style="width: 223px">  
    <col style="width: 391px">  
    <col style="width: 181px">  
    <col style="width: 111px"> 
    <col style="width: 126px">
    <col style="width: 145px">
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
      <td><ul><li>Empty tensors are supported. </li><li>The value must be a 2D tensor with shape [numRows, h]. <code>numRows</code> indicates the number of tokens, and <code>h</code> indicates the length of each token.</li></ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>expertIdx</td>
      <td>Input</td>
      <td>Sequence number corresponding to *k* experts to process each token.</td>
      <td><ul><li>Empty tensors are supported. </li><li>In drop/pad scenarios or when <code>expertTokensCountOrCumsumOut</code> needs to be output in drop/pad-less scenarios, the value range must be [0, expertNum – 1]. In other scenarios, the value must be greater than or equal to <code>0</code>.</li></ul></td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>1 or 2</td>
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
      <td>The value must be greater than or equal to <code>0</code>. In the drop/pad scenario, the value range is (0, numRows]. In this case, each expert drops tokens that exceed the capacity. If the number of tokens is less than the capacity, all-zero tokens are padded. In other scenarios, this attribute is not concerned.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertNum</td>
      <td>Input</td>
      <td>Number of experts.</td>
      <td>The value is greater than or equal to <code>0</code>. In the drop/pad scenario or when the value of <code>expertTokensCountOrCumsumFlag</code> is greater than 0 and the output <code>expertTokensCountOrCumsumOut</code> is required, the value of <code>expertNum</code> must be greater than 0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dropPadMode</td>
      <td>Input</td>
      <td>Indicates whether the scenario is in drop/pad mode.</td>
      <td>The value can be <code>0</code> or <code>1</code>. <ul><li><code>0</code>: indicates drop/pad-less scenario where <code>expertCapacity</code> is not verified. </li><li><code>1</code>: indicates the drop/pad scenario where <code>expertNum</code> and <code>expertCapacity</code> need to be verified. The corresponding measure will be taken when the number of tokens that can be processed by each expert exceeds or falls short of the <code>expertCapacity</code> value.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertTokensCountOrCumsumFlag</td>
      <td>Input</td>
      <td>Indicates whether to output <code>expertTokensCountOrCumsumOut</code>.</td>
      <td>The value can be <code>0</code>, <code>1</code>, or <code>2</code>. <ul><li><code>0</code>: indicates that <code>expertTokensCountOrCumsumOut</code> is not output. </li><li><code>1</code>: The output value is the cumulative sum of the number of tokens processed by each expert. </li><li><code>2</code>: The output value is the number of tokens processed by each expert.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertTokensBeforeCapacityFlag</td>
      <td>Input</td>
      <td>Controls whether to output <code>expertTokensBeforeCapacityOut</code>.</td>
      <td>The value can be<code> false</code> or <code>true</code><ul><li>false, indicating that <code>expertTokensBeforeCapacityOut</code> is not output. </li><li><code>true</code>: <code>expertTokensBeforeCapacityOut</code> is output. The value is the number of tokens processed by each expert before the drop operation.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expandedXOut</td>
      <td>Output</td>
      <td>Extended feature based on <code>expertIdx</code>.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The data type must be the same as that of <code>x</code>. </li><li>In the dropless/active scenario, the value must be a 2D tensor. In the dropless scenario, the shape is [numRows * k, h]. In the active scenario, the shape is [min(activeNum, numRows * k), h].<br>In the drop/pad scenario, the value must be a 3D tensor with shape [expertNum, expertCapacity, h].</li></ul>
      </td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2 or 3</td>
      <td>×</td>
    </tr>
    <tr>
      <td>expandedRowIdxOut</td>
      <td>Output</td>
      <td>Index mapping between <code>expandedXOut</code> and <code>x</code>.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The value must be a 1D tensor with shape [numRows * k].</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>expertTokensCountOrCumsumOut</td>
      <td>Output</td>
      <td>Statistics result and cumulative sum of the number of tokens processed by each expert.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The expertTokensCountOrCumsumFlag parameter determines whether to output the value. This value is output only in the drop/pad-less scenario. The value must be a 1D tensor with shape [expertNum].</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>expertTokensBeforeCapacityOut</td>
      <td>Output</td>
      <td>Statistics result of the number of tokens processed by each expert before the drop operation.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The <code>expertTokensBeforeCapacityFlag</code> parameter determines whether to output the value. This value is output only in the drop/pad scenario. The value must be a 1D tensor with shape [expertNum].</li></ul></td>
      <td>INT32</td>
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

  - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type of input `expertIdx` supports INT32, and the value is a 2D shape [numRows, k].
  - Ascend 950PR/Ascend 950DT: The expertIdx input supports the INT32 and INT64 data types. The shape must be 2D [numRows, k] or 1D [numRows]. When the shape is 1D, k is 1.
  - <term>Atlas inference products</term>: The data type of input `expertIdx` can be INT32. The value must be a 2D shape with size [numRows, k]. The `dropPadMode` parameter supports only `0`.

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
        <td rowspan="5">ACLNN_ERR_INNER_TILING_ERROR</td>
        <td rowspan="5">561002</td>
        <td>When <code>expertTokensCountOrCumsumOut</code> needs to be output, <code>expertNum</code> is <code>0</code>.</td>
      </tr>
      <tr>
        <td>The shape dimensions of <code>x</code> and <code>expertIdx</code> are not equal to <code>2</code>, and their first dimensions are not equal.</td>
      </tr>
      <tr>
        <td>The values of <code>activeNum</code>, <code>expertNum</code>, and <code>expertCapacity</code> are less than <code>0</code>.</td>
      </tr>
      <tr>
        <td>The values of <code>dropPadMode</code>, <code>expertTokensCountOrCumsumFlag</code>, and <code>expertTokensBeforeCapacityFlag</code> are not supported.</td>
      </tr>
      <tr>
        <td>When <code>dropPadMode</code> is set to <code>1</code>, <code>expertCapacity</code> and <code>expertNum</code> are set to <code>0</code>.</td>
      </tr>
    </tbody>
  </table>

## aclnnMoeInitRoutingV2

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1179px"> <colgroup>
  <col style="width: 169px">
  <col style="width: 130px">
  <col style="width: 880px">
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
    <td>Memory address of the workspace to be allocated on the device.</td>
  </tr>
  <tr>
    <td>workspaceSize</td>
    <td>Input</td>
    <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API <code>aclnnMoeInitRoutingV2GetWorkspaceSize</code>.</td>
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
  </tbody></table>

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
    
## Constraints

- Deterministic computation:
  - `aclnnMoeInitRoutingV2` defaults to deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_init_routing_v2.h"
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
    std::vector<int64_t> expandedXOutShape = {3, 2, 4};
    std::vector<int64_t> idxOutShape = {6};
    std::vector<int64_t> expertTokenOutShape = {3};
    void* xDeviceAddr = nullptr;
    void* expertIdxDeviceAddr = nullptr;
    void* expandedXOutDeviceAddr = nullptr;
    void* expandedRowIdxOutDeviceAddr = nullptr;
    void* expertTokenBeforeCapacityOutDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* expertIdx = nullptr;
    int64_t activeNum = 0;
    int64_t expertCapacity = 2;
    int64_t expertNum = 3;
    int64_t dropPadMode = 1;
    int64_t expertTokensCountOrCumsumFlag = 0;
    bool expertTokensBeforeCapacityFlag = true;
    aclTensor* expandedXOut = nullptr;
    aclTensor* expandedRowIdxOut = nullptr;
    aclTensor* expertTokensBeforeCapacityOut = nullptr;
    std::vector<float> xHostData = {0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3};
    std::vector<int> expertIdxHostData = {1, 2, 0, 1, 0, 2};
    std::vector<float> expandedXOutHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int> expandedRowIdxOutHostData = {0, 0, 0, 0, 0, 0};
    std::vector<int> expertTokensBeforeCapacityOutHostData = {0, 0, 0};
    // Create a self aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertIdxHostData, idxShape, &expertIdxDeviceAddr, aclDataType::ACL_INT32, &expertIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an out aclTensor.
    ret = CreateAclTensor(expandedXOutHostData, expandedXOutShape, &expandedXOutDeviceAddr, aclDataType::ACL_FLOAT, &expandedXOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandedRowIdxOutHostData, idxOutShape, &expandedRowIdxOutDeviceAddr, aclDataType::ACL_INT32, &expandedRowIdxOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertTokensBeforeCapacityOutHostData, expertTokenOutShape, &expertTokenBeforeCapacityOutDeviceAddr, aclDataType::ACL_INT32, &expertTokensBeforeCapacityOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnMoeInitRoutingV2.
    ret = aclnnMoeInitRoutingV2GetWorkspaceSize(x, expertIdx, activeNum, expertCapacity, expertNum, dropPadMode, expertTokensCountOrCumsumFlag, expertTokensBeforeCapacityFlag, expandedXOut, expandedRowIdxOut, nullptr, expertTokensBeforeCapacityOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeInitRoutingV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnMoeInitRoutingV2.
    ret = aclnnMoeInitRoutingV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeInitRoutingV2 failed. ERROR: %d\n", ret); return ret);
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
    auto expertTokensBeforeCapacitySize = GetShapeSize(expertTokenOutShape);
    std::vector<int> expertTokenIdxData(expertTokensBeforeCapacitySize, 0);
    ret = aclrtMemcpy(expertTokenIdxData.data(), expertTokenIdxData.size() * sizeof(expertTokenIdxData[0]), expertTokenBeforeCapacityOutDeviceAddr, expertTokensBeforeCapacitySize * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expertTokensBeforeCapacitySize; i++) {
        LOG_PRINT("expertTokenIdxData[%ld] is: %d\n", i, expertTokenIdxData[i]);
    }
    // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
    aclDestroyTensor(x);
    aclDestroyTensor(expertIdx);
    aclDestroyTensor(expandedXOut);
    aclDestroyTensor(expandedRowIdxOut);
    aclDestroyTensor(expertTokensBeforeCapacityOut);

    // 7. Release device resources. Modify the code based on the API definition.
    aclrtFree(xDeviceAddr);
    aclrtFree(expertIdxDeviceAddr);
    aclrtFree(expandedXOutDeviceAddr);
    aclrtFree(expandedRowIdxOutDeviceAddr);
    aclrtFree(expertTokenBeforeCapacityOutDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
