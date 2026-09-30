# aclnnMoeInitRoutingV3

## Product Support

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference products</term>   |     ×    |
|  <term>Atlas training products</term>   |     ×    |

## Function

- Description: Performs routing computation for MoE based on the computation result of [aclnnMoeGatingTopKSoftmaxV2](../../moe_gating_top_k_softmax_v2/docs/aclnnMoeGatingTopKSoftmaxV2_en.md). Both non-quantization and dynamic quantization modes are supported. This API has the following function changes based on the V2 API [aclnnMoeInitRoutingV2](../../moe_init_routing_v2/docs/aclnnMoeInitRoutingV2_en.md). Select a proper API based on your actual requirements.

    1. Added the dynamic quantization function to support the INT8 dynamic quantization output of `expendX`.

    2. Added the `activeExpertRangeOptional` parameter to support `expertId` filtering within the valid range.

    3. Deleted the `expertTokensBeforeCapacityFlag` attribute and the output `expertTokensBeforeCapacityOut` (replaced with `expertTokensCountOrCumsumOut`).

- Formula: 

  1. Sort the input `expertIdx` to obtain the sorted result `sortedExpertIdx` and the corresponding index `sortedRowIdx`.

      $$
      sortedExpertIdx, sortedRowIdx=keyValueSort(expertIdx,rowIdx)
      $$

  2. Use `sortedRowIdx` for location mapping to obtain `expandedRowIdxOut`.

      $$
      expandedRowIdxOut[sortedRowIdx[i]]=i
      $$

  3. In drop mode, collect statistics on the histogram of each expert in `sortedExpertIdx` to obtain `expertTokensCountOrCumsumOutOptional`.

      $$
      expertTokensCountOrCumsumOutOptional[i]=Histogram(sortedExpertIdx)
      $$

  4. Compute the quantization result.
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
  
  5. Obtain the values of the first *NUM\_ROWS* `sortedRowIdx` values for `quantResult` to obtain `expandedXOut`.

      $$
      expandedXOut[i]=quantResult[sortedRowIdx[i]\%NUM\_ROWS]
      $$

  6. The number of valid elements specified by `availableIdxNum` in `expandedRowIdxOut` is equal to the number of elements within the range specified by `activeExpertRangeOptional` in `expertIdx`.
    $$
    availableIdxNum = |\{x\in expertIdx| expert\_start \le x<expert\_end \ \}|
    $$
  
## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnMoeInitRoutingV3GetWorkspaceSize` is called to obtain the input parameters, the workspace size required for computation, and the executor that contains the operator computation process. Then, `aclnnMoeInitRoutingV3` is called to perform computation.

```Cpp
aclnnStatus aclnnMoeInitRoutingV3GetWorkspaceSize(
  const aclTensor   *x, 
  const aclTensor   *expertIdx, 
  const aclTensor   *scaleOptional, 
  const aclTensor   *offsetOptional, 
  int64_t            activeNum, 
  int64_t            expertCapacity, 
  int64_t            expertNum, 
  int64_t            dropPadMode, 
  int64_t            expertTokensNumType, 
  bool               expertTokensNumFlag, 
  int64_t            quantMode, 
  const aclIntArray *activeExpertRangeOptional, 
  int64_t            rowIdxType, 
  const aclTensor   *expandedXOut,
  const aclTensor   *expandedRowIdxOut,
  const aclTensor   *expertTokensCountOrCumsumOut,
  const aclTensor   *expandedScaleOut, 
  uint64_t          *workspaceSize, 
  aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnMoeInitRoutingV3(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```

## aclnnMoeInitRoutingV3GetWorkspaceSize

- **Parameters**:

    <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 158px">
    <col style="width: 120px">
    <col style="width: 333px">
    <col style="width: 375px">
    <col style="width: 212px">
    <col style="width: 100px">
    <col style="width: 107px">
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
        <td>The shape is (NUM_ROWS, H).</td>
        <td>FLOAT16, BFLOAT16, FLOAT32, INT8</td>
        <td>ND</td>
        <td>2</td>
        <td>-</td>
      </tr>
      <tr>
        <td>expertIdx</td>
        <td>Input</td>
        <td> Each row of features corresponds to K processing experts. The number of expert IDs cannot exceed that of experts.</td>
        <td>The shape is (NUM_ROWS, K).</td>
        <td>INT32</td>
        <td>ND</td>
        <td>2</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scaleOptional</td>
        <td>Input</td>
        <td>Parameter used to compute the quantization result.</td>
        <td>If this parameter is not specified, <code>scale</code> is not used during computation.
          <br>In non-quantization scenarios, if this parameter is specified, the input must be a 1D tensor with shape (NUM_ROWS,).
          <br>In dynamic quantization scenarios, if this parameter is specified, the input must be a 2D tensor with shape (expertEnd – expertStart, H).</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1-2</td>
        <td>-</td>
      </tr>
      <tr>
        <td>offsetOptional</td>
        <td>Input</td>
        <td>Offset used to compute the quantization result.</td>
        <td>Not required in non-quantization scenarios.<br>Not required in dynamic quantization scenarios.</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>activeNum</td>
        <td>Input</td>
        <td>Total maximum number of rows that can be processed, that is, the maximum number of rows that are valid in the output <code>expandedXOut</code>.</td>
        <td>Pass a value that is greater than or equal to <code>0</code>.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>expertCapacity</td>
        <td>Input</td>
        <td>Number of tokens that can be processed by each expert.</td>
        <td>The value must be greater than or equal to <code>0</code>.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>expertNum</td>
        <td>Input</td>
        <td>Number of experts.</td>
        <td>When <code>expertTokensNumType</code> is set to <code>key_value</code>, the value range is [0, 5120]. In other modes, the value range is [0, 10240].</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dropPadMode</td>
        <td>Input</td>
        <td>Indicates whether the scenario is in drop/pad mode.</td>
        <td>The values are <code>0</code> and <code>1</code>.
          <br>0: indicates the dropless scenario, where <code>expertCapacity</code> is not verified.
          <br>1: indicates the drop/pad scenario.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>expertTokensNumType</td>
        <td>Input</td>
        <td>Different modes.</td>
        <td>The value can be <code>0</code>, <code>1</code>, or <code>2</code>.
          <br><code>0</code>: cumsum mode.
          <br><code>1</code>: count mode.
          <br><code>2</code>: key_value mode.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>expertTokensNumFlag</td>
        <td>Input</td>
        <td>Indicates whether to output <code>expertTokensCountOrCumsumOut</code>.</td>
        <td>The value can be <code>false</code> or <code>true</code>.</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantMode</td>
        <td>Input</td>
        <td>Different quantization scenarios.</td>
        <td>The value can be <code>0</code>, <code>1</code>, or <code>-1</code>.
          <br><<code>0</code>: static quantization scenario.
          <br><code>1</code>: dynamic quantization scenario.
          <br><code>-1</code>: non-quantization scenario. </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>activeExpertRangeOptional</td>
        <td>Input</td>
        <td>Active expert range.</td>
        <td>The length is <code>2</code>. The value in the array is [expertStart, expertEnd], which is left-closed and right-open. The value must be greater than or equal to <code>0</code>, and the value of <code>expertEnd</code> must be less than or equal to that of <code>expertNum</code></td>.
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>rowIdxType</td>
        <td>Input</td>
        <td>Index type used by <code>expandedRowIdxOut</code>.</td>
        <td>The value can be <code>0</code> or <code>1</code>. (The performance template supports only <code>1</code>.)
          <br><code>0</code>: index of the gather type.
          <br><code>1</code>: index of the scatter type.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>expandedXOut</td>
        <td>Output</td>
        <td>Extended feature based on expertIdx.</td>
        <td>In non-quantization scenarios, the data type is the same as that of <code>x</code>. In quantization scenarios, the data type can be INT8.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32, INT8</td>
        <td>ND</td>
        <td>2</td>
        <td>-</td>
      </tr>
      <tr>
        <td>expandedRowIdxOut</td>
        <td>Output</td>
        <td>Mapping between the indexes of <code>expandedXOut</code> and <code>x</code>.</td>
        <td>The first <code>availableIdNum</code> * <code>H</code> elements are valid data. The other invalid data is determined by <code>rowIdxType</code>.
          <br>When <code>rowIdxType</code> is set to <code>0</code>, invalid data is padded with <code>-1</code>.
          <br>When <code>rowIdxType</code> is set to <code>1</code>, invalid data is not initialized.</td>
        <td>INT32</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>expertTokensCountOrCumsumOut</td>
        <td>Output</td>
        <td>Statistics result and cumulative sum of the number of tokens processed by each expert.</td>
        <td>When <code>expertTokensNumType</code> is set to <code>1</code>, this parameter indicates the total number of tokens processed by experts within the range specified by <code>activeExpertRangeOptional</code>.
            <br>When <code>expertTokensNumType</code> is set to <code>2</code>, this parameter indicates the experts whose total number of tokens within the range specified by <code>activeExpertRangeOptional</code> is not 0, and the total number of tokens processed by the corresponding expert.</td>
        <td>INT64</td>
        <td>ND</td>
        <td>1-2</td>
        <td>-</td>
      </tr>
      <tr>
        <td>expandedScaleOut</td>
        <td>Output</td>
        <td>Intermediate value of <code>scaleOptional</code> in different quantization processes.</td>
        <td>The shape is (NUM_ROWS * K,).
          <br>When <code>scaleOptional</code> is specified, the first <code>availableIdNum</code> * <code>H</code> elements are valid data.
          <br>When <code>scaleOptional</code> is specified, the first *availableIdxNum* elements are valid data. If the data type of <code>x</code> is INT8, the output value is not defined. </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
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

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
  </colgroup>
  <thead>
    <tr>
      <th>Return</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input and output for computation are null pointers.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161002</td>
      <td>The input and output data types are not supported.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>
      The shape of the input or output tensor is not supported.<br>
      The input attribute is not supported.<br>
      </td>
    </tr>
  </table>

## aclnnMoeInitRoutingV3

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API <code>aclnnMoeInitRoutingV3GetWorkspaceSize</code>.</td>
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

  `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnMoeInitRoutingV3` defaults to deterministic implementation.

- Input value range restrictions:
  - `activeNum` is not used currently. The value must be equal to `NUM_ROWS` * `K` upon verification.
  - `expertCapacity` is currently not used. Non-empty verification is performed only.
  - Currently, `dropPadMode` supports only `0`, indicating the dropless scenario.
  - Currently, `expertTokensNumType` supports only `1` and `2`, indicating the count mode and key\_value mode, respectively.
  - `expertTokensNumFlag` supports only `true`, indicating that `expertTokensCountOrCumsumOut` is output.
  - `quantMode` supports only `1` and `-1`, indicating the dynamic quantization scenario and non-quantization scenario, respectively.

- Other restrictions: This operator supports two performance templates. To use either of the two templates, the following conditions must be met. If the conditions are not met, the general template is used.

  - To use the low-latency performance template, the following conditions must be met:
    - The input shapes of `x`, `expertIdx`, and `scaleOptional` must be (1, 7168), (1, 8), and (256, 7168), respectively.
    - The data type of `x` must be BFLOAT16.
    - The attribute requirements are as follows: `activeExpertRangeOptional`=[0, 256]; `quantMode`=`1`; `expertTokensNumType`=`2`; `expertNum`=`256`

  - To use the large-batch performance template, the following conditions must be met:
    - The value range of `NUM_ROWS` is [384, 8192].
    - K=8
    - expertNum=256
    - expertEnd-expertStart<=32
    - quantMode=-1
    - rowIdxType=1
    - expertTokensNumType=1

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_init_routing_v3.h"

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
int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}
int Init(int32_t deviceId, aclrtStream *stream)
{
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
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor)
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
    *tensor = aclCreateTensor(shape.data(),
        shape.size(),
        dataType,
        strides.data(),
        0,
        aclFormat::ACL_FORMAT_ND,
        shape.data(),
        shape.size(),
        *deviceAddr);
    return 0;
}
int main()
{
    // 1. (Boilerplate) Initialize the device and stream. For details, see the list of external ACL APIs.
    // Set the device ID in use.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as required.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct inputs and outputs based on the API definition.
    std::vector<int64_t> xShape = {3, 2};
    std::vector<int64_t> expertIdxShape = {3, 4};
    std::vector<int64_t> scaleShape = {3};
    std::vector<int64_t> offsetShape = {1};

    std::vector<int64_t> expandedXOutShape = {12, 2};
    std::vector<int64_t> expandedRowIdxOutShape = {12};
    std::vector<int64_t> expertTokensCountOrCumsumOutOptionalShape = {4};
    std::vector<int64_t> expandedScaleOutOptionalShape = {12};

    std::vector<int64_t> activeExpertRangeArray = {0, 4};

    void *xDeviceAddr = nullptr;
    void *expertIdxDeviceAddr = nullptr;
    void *scaleDeviceAddr = nullptr;
    void *offsetDeviceAddr = nullptr;

    void *expandedXOutDeviceAddr = nullptr;
    void *expandedRowIdxOutDeviceAddr = nullptr;
    void *expertTokensCountOrCumsumOutOptionalDeviceAddr = nullptr;
    void *expandedScaleOutOptionalDeviceAddr = nullptr;

    aclTensor *x = nullptr;
    aclTensor *expertIdx = nullptr;
    aclTensor *scale = nullptr;
    aclTensor *offset = nullptr;

    int64_t activeNum = 12;
    int64_t expertCapacity = 4;
    int64_t expertNum = 256;
    int64_t dropPadMode = 0;
    int64_t expertTokensNumType = 1;
    bool expertTokensNumFlag = true;
    int64_t quantMode = -1;
    aclIntArray *activeExpertRange = aclCreateIntArray(activeExpertRangeArray.data(), activeExpertRangeArray.size());
    int64_t rowIdxType = 1;

    aclTensor *expandedXOut = nullptr;
    aclTensor *expandedRowIdxOut = nullptr;
    aclTensor *expertTokensCountOrCumsumOutOptional = nullptr;
    aclTensor *expandedScaleOutOptional = nullptr;

    std::vector<float> xHostData = {0.1, 0.1, 0.2, 0.2, 0.3, 0.3};
    std::vector<int> expertIdxHostData = {1, 2, 0, 3, 0, 2, 1, 3, 0, 1, 3, 2};
    std::vector<float> scaleHostData = {0.3423, 0.1652, 0.2652};
    std::vector<float> offsetHostData = {1.8369};

    std::vector<int8_t> expandedXOutHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int> expandedRowIdxOutHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int64_t> expertTokensCountOrCumsumOutOptionalHostData = {0, 0, 0, 0};
    std::vector<float> expandedScaleOutOptionalHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // Create a self aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertIdxHostData, expertIdxShape, &expertIdxDeviceAddr, aclDataType::ACL_INT32, &expertIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(offsetHostData, scaleShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an out aclTensor.
    ret = CreateAclTensor(
        expandedXOutHostData, expandedXOutShape, &expandedXOutDeviceAddr, aclDataType::ACL_INT8, &expandedXOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandedRowIdxOutHostData,
        expandedRowIdxOutShape,
        &expandedRowIdxOutDeviceAddr,
        aclDataType::ACL_INT32,
        &expandedRowIdxOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertTokensCountOrCumsumOutOptionalHostData,
        expertTokensCountOrCumsumOutOptionalShape,
        &expertTokensCountOrCumsumOutOptionalDeviceAddr,
        aclDataType::ACL_INT64,
        &expertTokensCountOrCumsumOutOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandedScaleOutOptionalHostData,
        expandedScaleOutOptionalShape,
        &expandedScaleOutOptionalDeviceAddr,
        aclDataType::ACL_FLOAT,
        &expandedScaleOutOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // Call the first-phase API of aclnnMoeInitRoutingV3.
    ret = aclnnMoeInitRoutingV3GetWorkspaceSize(x,
        expertIdx,
        scale,
        offset,
        activeNum,
        expertCapacity,
        expertNum,
        dropPadMode,
        expertTokensNumType,
        expertTokensNumFlag,
        quantMode,
        activeExpertRange,
        rowIdxType,
        expandedXOut,
        expandedRowIdxOut,
        expertTokensCountOrCumsumOutOptional,
        expandedScaleOutOptional,
        &workspaceSize,
        &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeInitRoutingV3GetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnMoeInitRoutingV3.
    ret = aclnnMoeInitRoutingV3(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeInitRoutingV3 failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
    auto expandedXSize = GetShapeSize(expandedXOutShape);
    std::vector<int8_t> expandedXData(expandedXSize, 0);
    ret = aclrtMemcpy(expandedXData.data(),
        expandedXData.size() * sizeof(expandedXData[0]),
        expandedXOutDeviceAddr,
        expandedXSize * sizeof(int8_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expandedXSize; i++) {
        LOG_PRINT("expandedXData[%ld] is: %d\n", i, expandedXData[i]);
    }
    auto expandedRowIdxSize = GetShapeSize(expandedRowIdxOutShape);
    std::vector<int> expandedRowIdxData(expandedRowIdxSize, 0);
    ret = aclrtMemcpy(expandedRowIdxData.data(),
        expandedRowIdxData.size() * sizeof(expandedRowIdxData[0]),
        expandedRowIdxOutDeviceAddr,
        expandedRowIdxSize * sizeof(int32_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expandedRowIdxSize; i++) {
        LOG_PRINT("expandedRowIdxData[%ld] is: %d\n", i, expandedRowIdxData[i]);
    }
    auto expertTokensBeforeCapacitySize = GetShapeSize(expertTokensCountOrCumsumOutOptionalShape);
    std::vector<int> expertTokenIdxData(expertTokensBeforeCapacitySize, 0);
    ret = aclrtMemcpy(expertTokenIdxData.data(),
        expertTokenIdxData.size() * sizeof(expertTokenIdxData[0]),
        expertTokensCountOrCumsumOutOptionalDeviceAddr,
        expertTokensBeforeCapacitySize * sizeof(int32_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expertTokensBeforeCapacitySize; i++) {
        LOG_PRINT("expertTokenIdxData[%ld] is: %d\n", i, expertTokenIdxData[i]);
    }

    auto dynamicQuantScaleSize = GetShapeSize(expandedScaleOutOptionalShape);
    std::vector<float> dynamicQuantScaleData(dynamicQuantScaleSize, 0);
    ret = aclrtMemcpy(dynamicQuantScaleData.data(),
        dynamicQuantScaleData.size() * sizeof(dynamicQuantScaleData[0]),
        expandedScaleOutOptionalDeviceAddr,
        dynamicQuantScaleSize * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
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
    aclDestroyTensor(expertTokensCountOrCumsumOutOptional);
    aclDestroyTensor(expandedScaleOutOptional);

    // 7. Release device resources. Modify the code based on the API definition.
    aclrtFree(xDeviceAddr);
    aclrtFree(expertIdxDeviceAddr);
    aclrtFree(scaleDeviceAddr);
    aclrtFree(offsetDeviceAddr);
    aclrtFree(expandedXOutDeviceAddr);
    aclrtFree(expandedRowIdxOutDeviceAddr);
    aclrtFree(expertTokensCountOrCumsumOutOptionalDeviceAddr);
    aclrtFree(expandedScaleOutOptionalDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
