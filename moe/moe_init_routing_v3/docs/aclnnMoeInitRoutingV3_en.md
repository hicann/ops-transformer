# aclnnMoeInitRoutingV3

[📄 View Source Code](https://gitcode.com/cann/ops-transformer/tree/master/moe/moe_init_routing_v3)

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  Ascend 950PR/Ascend 950DT  |     √    |
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference products</term>   |     ×    |
|  <term>Atlas training products</term>   |     ×    |

## Function

- Description: Performs mixture of experts (MoE) routing based on the computation results of [aclnnMoeGatingTopKSoftmaxV2](../../moe_gating_top_k_softmax_v2/docs/aclnnMoeGatingTopKSoftmaxV2_en.md). Non-quantization, static quantization, and dynamic quantization configurations are supported. This API has the following function changes based on the V2 API [aclnnMoeInitRoutingV2](../../moe_init_routing_v2/docs/aclnnMoeInitRoutingV2.md). Select a proper API based on your actual requirements.

  1. The dynamic and static quantization functions are added, and the int8 quantization mode of expendX is supported.

  2. The output expertTokensBeforeCapacityOut is deleted, and the output expertTokensCountOrCumsumOut is added.

  3. The original output mode of V2 is compatible, and the key_value output format is added. The original attributes expertTokensBeforeCapacityFlag(bool) and expertTokensCountOrCumsumFlag(int) are redefined as expertsTokensNumFlag(bool) and expertTokensNumType(int), respectively. The following table describes the mapping between the output formats.

  <table align="center">
    <tr>
      <th>DropPadMode</th>
      <th>expertsTokensNumFlag</th>
      <th>expertTokensNumType</th>
      <th style="text-align: center;">Output format description</th>
    </tr>
    <tr align="center">
      <td>0</td>
      <td>true</td>
      <td>0</td>
      <td align="left">In comsum mode, expertTokensCountOrCumsumOut indicates the prefix sum and histogram of tokens processed by each expert after sorting.</td>
    </tr>
    <tr align="center">
      <td>0</td>
      <td>true</td>
      <td>1</td>
      <td align="left">In count mode, expertTokensCountOrCumsumOut indicates the histogram of the number of tokens processed by each expert after sorting.</td>
    </tr>
    <tr align="center">
      <td>0</td>
      <td>true</td>
      <td>2</td>
      <td align="left">In key_value mode, the output shape is [expert_num, 2], indicating the cumulative number of non-zero tokens processed by each expert.</td>
    </tr>
    <tr align="center">
      <td>1</td>
      <td>true</td>
      <td>1</td>
      <td align="left">The output mode is count.</td>
    </tr>
    <tr align="center">
      <td>Disabled</td>
      <td>false</td>
      <td>Disabled</td>
      <td align="left">expertTokensCountOrCumsumOut is not output.</td>
    </tr>
  </table>
  
- Formula: 

  1. Sort the input `expertIdx` to obtain the sorted result `sortedExpertIdx` and the corresponding index `sortedRowIdx`.

      $$
      sortedExpertIdx, sortedRowIdx=keyValueSort(expertIdx,rowIdx)
      $$

  2. Use `sortedRowIdx` for location mapping to obtain `expandedRowIdxOut`.
      - When `rowIdxType` is `1`, the API outputs scatter indices.

        $$
        expandedRowIdxOut[i]=sortedRowIdx[i]
        $$

      - When `rowIdxType` is `0`, the API outputs gather indices.

        $$
        expandedRowIdxOut[sortedRowIdx[i]]=i
        $$
      
  3. Compute the histogram of `sortedExpertIdx` for each expert to obtain `expertTokensCountOrCumsumOutOptional`.

      $$
      expertTokensCountOrCumsumOutOptional[i]=Histogram(sortedExpertIdx)
      $$

  4. If quantMode is not equal to -1, calculate the quantization result.
      - Static quantization:

        $$
        quantResult=round((x∗scaleOptional)+offsetOptional)
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
  
  5. Tokens are rearranged using scatter indices when the active expert range covers all experts. In other configurations, tokens are rearranged using gather indices. When `dropPadMode` is set to `1`, the number of tokens processed by each expert is padded to `expertCapacity`. Tokens exceeding `expertCapacity` are dropped, and insufficient tokens are padded with zeros. The resulting output `expandedXOut` is obtained as follows:
      - Non-quantization scenarios:
        - Rearrangement using scatter indices:

          $$
          expandedXOut[i]=x[scatterRowIdx[i] // K]
          $$

        - Rearrangement using gather indices:

          $$
          expandedXOut[gatherRowIdx[i]]=x[i // K]
          $$

      - Quantization scenarios:
        - Rearrangement using scatter indices:

        $$
        expandedXOut[i]=quantResult[scatterRowIdx[i] // K]
        $$

        - Rearrangement using gather indices:

        $$
        expandedXOut[gatherRowIdx[i]]=quantResult[i // K]
        $$

  6. The valid element count of `expandedRowIdxOut`, denoted by `availableIdxNum`, is calculated as the number of elements in `expertIdx` that fall within the range specified by `activeExpertRangeOptional`.

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
  <col style="width: 400px">
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x (aclTensor)</td>
      <td>Input</td>
      <td>MoE input, that is, the token feature input.</td>
      <td>The shape is (NUM_ROWS, H).</td>
      <td>FLOAT16, BFLOAT16, FLOAT32, INT8</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertIdx (aclTensor)</td>
      <td>Input</td>
      <td> Each row of features corresponds to K processing experts. The number of expert IDs cannot exceed that of experts.</td>
      <td>The shape is (NUM_ROWS, K).</td>
      <td>INT32</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleOptional (aclTensor)</td>
      <td>Input</td>
      <td>Parameter used to calculate the quantization result.</td>
      <td><ul>
        <li>If no value is entered, the scale is not used for calculation.</li>
        <li>This input is optional in non-quantization scenarios. If it is specified, the input must be a 1D tensor with the shape of (NUM_ROWS,).</li>
        <li>This input is required in static quantization scenarios. The input must be a 1D tensor with the shape of [1, ].</li>
        <li>This input is optional in dynamic quantization scenarios. If it is specified, the input must be a 2D tensor with the shape of (expertEnd – expertStart, H).</li>
        <li>This input is not required in the MXFP8 quantization scenarios (when quantMode is set to 2 or 3).</li>
        </ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>offsetOptional (aclTensor)</td>
      <td>Input</td>
      <td>Offset used to compute the quantization result.</td>
      <td><ul>
        <li>This input is not required in non-quantization scenarios.</li><li>This input is required in static quantization scenarios. The input must be a 1D tensor with the shape of [1, ].</li>
        <li>This input is not required in dynamic quantization and MXFP8 quantization scenarios.</li>  </ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activeNum (int64_t) </td>
      <td>Input</td>
      <td>Total maximum number of rows that can be processed, that is, the maximum number of rows that are valid in the output <code>expandedXOut</code>.</td>
      <td>The input value must be greater than or equal to 0. The value 0 indicates the dropless scenario, and a value greater than 0 indicates the active scenario where all experts are required to process the total number of tokens.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertCapacity (int64_t)</td>
      <td>Input</td>
      <td>Number of tokens that can be processed by each expert.</td>
      <td>The input parameter must be greater than 0 and less than NUM_ROWS.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertNum (int64_t)</td>
      <td>Input</td>
      <td>Number of experts.</td>
      <td>When <code>expertTokensNumType</code> is set to <code>key_value</code>, the value range is [0, 5120]. In other modes, the value range is [0, 10240].</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dropPadMode (int64_t)</td>
      <td>Input</td>
      <td>Indicates whether the scenario is in drop/pad mode.</td>
      <td>The values are <code>0</code> and <code>1</code>.
        <br>0: indicates the dropless scenario, where <code>expertCapacity</code> is not verified.
        <br>1: the DropPad scenario
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertTokensNumType (int64_t)</td>
      <td>Input</td>
      <td>Indicates the mode of the histogram.</td>
      <td>The value can be <code>0</code>, <code>1</code>, or <code>2</code>.
        <br>0: comsum mode
        <br><code>1</code>: count mode.
        <br>2: key_value mode
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertTokensNumFlag (bool) </td>
      <td>Input</td>
      <td>Indicates whether to output <code>expertTokensCountOrCumsumOut</code>.</td>
      <td>The value can be <code>false</code> or <code>true</code>.</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode (int64_t)</td>
      <td>Input</td>
      <td>Different quantization scenarios.</td>
      <td>The value can be 0, 1, -1, 2, or 3 (The value range varies according to the product. For details, see the description below the table.)
        <br>0: static quantization scenario.
        <br>1: dynamic quantization scenario.
        <br>-1: non-quantization scenario
        <br>2: MXFP8 quantization scenario, where expandedXOut is quantized to FLOAT8_E5M2
        <br>3: MXFP8 quantization scenario, where expandedXOut is quantized to FLOAT8_E4M3FN
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activeExpertRangeOptional (aclIntArray)</td>
      <td>Input</td>
      <td>Active expert range.</td>
      <td>The length is 2. The value in the array is [expertStart, expertEnd], which is left-closed and right-open. The value must be greater than or equal to 0, and expertEnd must be less than or equal to expertNum. In the Drop/Pad scenario, expertStart is 0 and expertEnd is expertNum.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rowIdxType (int64_t)</td>
      <td>Input</td>
      <td>Index type used by <code>expandedRowIdxOut</code>.</td>
      <td>The value can be 0 or 1.
        <br>0: gather index.
        <br>1: scatter index.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expandedXOut (aclTensor)</td>
      <td>Output</td>
      <td>Extended feature based on expertIdx.</td>
      <td><ul>
        <li>In the dropless scenario, the shape is [NUM_ROWS * K, H].</li>
        <li>In the active scenario, the shape is [min(activeNum, NUM_ROWS * K), H].</li>
        <li>In the Drop/Pad scenario, the value is a 3D tensor with the shape of [expertNum, expertCapacity, H].</li>
        <li>In non-quantization scenarios, the data type is the same as that of x. In quantization scenarios, if quantMode is set to 0 or 1, the data type is supported as INT8. If quantMode is set to 2 or 3, the data type is supported as FLOAT8_E5M2 or FLOAT8_E4M3FN, respectively.</li>
      </ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT32, INT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expandedRowIdxOut (aclTensor)</td>
      <td>Output</td>
      <td>Mapping between the indexes of <code>expandedXOut</code> and <code>x</code>.</td>
      <td>The output shape is (NUM_ROWS*K,). The first availableIdxNum elements are valid data, and the rest invalid data is determined by rowIdxType.
        <ul><li>When rowIdxType is set to 0, the invalid data is filled with -1.</li>
        <li>When rowIdxType is set to 1, the invalid data is not initialized.</li></ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertTokensCountOrCumsumOut (aclTensor)</td>
      <td>Output</td>
      <td>Statistics result or accumulated value of the number of tokens processed by each expert</td>
      <td><ul>
        <li>When expertTokensNumType is set to 0, the value indicates the prefix sum of the total number of tokens processed by experts in the activeExpertRangeOptional range after sorting.</li>
        <li>When expertTokensNumType is set to 1, this field indicates the total number of tokens processed by experts in the activeExpertRangeOptional range.</li>
        <li>When expertTokensNumType is set to 2, this field indicates the total number of tokens processed by experts whose token count is not 0 in the activeExpertRangeOptional range.</li>
      </ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1-2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expandedScaleOut (aclTensor)</td>
      <td>Output</td>
      <td>Intermediate value of <code>scaleOptional</code> in different quantization processes.</td>
      <td>The output is the product of all dimensions except the last dimension of the expandedXOut shape.
        <ul style="list-style-type: circle;">
        <li>In non-quantization scenarios, when scaleOptional is input, the first availableIdxNum elements are valid.</li>
        <li>In dynamic quantization scenarios, when scaleOptional is input, the first availableIdxNum elements are valid.</li>
        <li>This output is not provided in static quantization scenarios.</li>
        <li>In the MXFP8 quantization scenario, the output is of the FLOAT8_E8M0 type, and the shape is [NUM_ROWS*K, M], where M = CeilAlign(CeilDiv(H,32),2). The first availableIdxNum rows of NUM_ROWS*K are valid.</li></ul>
      </td>
      <td>FLOAT32, FLOAT8_E8M0</td>
      <td>ND</td>
      <td>1-2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t)</td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
      <td>-</td>
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
  <tbody>
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
  </tbody></table>

- **Differences in Support for Different Products**
  - The support for quantMode varies as follows:
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: -1, 0, and 1 are supported.
    - Ascend 950PR/Ascend 950DT: -1, 1, 2, and 3 are supported.
  - Ascend 950PR/Ascend 950DT: Only the following values are supported:
    - activeNum can only be set to NUM_ROWS x K.
    - expertCapacity is verified but not used. That is, the number of tokens that each expert can process is not limited.
    - dropPadMode can only be set to 0.
    - expertTokensNumType can only be set to 1 or 2.
    - expertTokensNumFlag can only be set to true.
    
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

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnMoeInitRoutingV3` defaults to deterministic implementation.

- This operator supports three performance profiles on the following product models. The admission conditions must be met for each profile. Otherwise, the general profile is used.
  - Products that support performance profiles:
    - <term>Atlas A2 training products/Atlas A2 inference products</term>
    - <term>Atlas A3 training products/Atlas A3 inference products</term>
  - The admission conditions of the performance profiles are as follows:
    <table>
      <tr align="center">
        <th style="text-align: center;">Performance Profile Type</th>
        <th style="text-align: center;">Admission Conditions</th>
      </tr>
      <tr>
        <td align="center">Low-latency performance profile</td>
        <td>The following conditions must be met: <ul><li>The input shapes of x, expertIdx, and scaleOptional must be (1, 7168), (1, 8), and (256, 7168), respectively. </li><li>The x data type must be BFLOAT16.</li><li>. The attribute requirements are as follows: activeExpertRangeOptional=[0, 256], quantMode=1, expertTokensNumType=2, and expertNum=256</li></ul></td>.
      </tr>
      <tr>
        <td align="center">Large-batch performance profile</td>
        <td>The following conditions must be met: <ul><li>NUM_ROWS ranges from [384, 8192], and K is 8. </li><li>The attribute requirements are as follows: expertNum=256, expertEnd-expertStart<=32, quantMode=-1, rowIdxType=1, and expertTokensNumType=1.</li></ul></td>
      </tr>
      <tr>
        <td align="center"><br>Full-load performance profile</td>
        <td>When the operator input shape is small, the multi-core synchronization time between operations accounts for a large proportion, which becomes a performance bottleneck. Therefore, a performance profile is added for this scenario. In this template, data movement, sorting, and computation are all completed within a single kernel. The following conditions must be met: <ul><li>Attribute requirements: dropPadMode=0</li></ul></td>
      </tr>
    </table>

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
