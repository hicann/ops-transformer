# aclnnMoeGatingTopKV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:09:40.406Z -->

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

- Description: In MoE computation, Sigmoid, SoftMax, or SqrtSoftplus is computed on input x, grouped sorting is performed on the computation results, and the top k experts are selected based on the grouped sorting result. Two modes are supported:
  - **TopK mode**: Selects experts by performing TopK sorting on normValue
  - **Hash mode**: Obtains pre-computed expert indices from the tid2eid mapping table based on inputIds, skips the sorting step, and outputs them directly (newly supported by aclnnMoeGatingTopKV2)

- Formula:

    **TopK mode:**

    **Step 1: Normalization**

    Perform normalization on input x according to normType:

    $$
    normOut =
    \begin{cases}
        \text{SoftMax}(x),      & normType = 0 \\
        \text{Sigmoid}(x),      & normType = 1 \\
        \sqrt{\text{Softplus}(x)},     & normType = 2
    \end{cases}
    $$

    **Step 2: Add bias**

    If bias is not empty, perform bias addition to obtain the value used for selection:

    $$
    normValue = normOut + bias
    $$

    Otherwise, $normValue = normOut$.

    **Step 3: group filtering** (performed only when groupCount > 1)

    Divide normValue into groupCount groups and compute the score of each group based on groupSelectMode:

    $$
    groupedValue = Reshape(normValue,\ [batch,\ groupCount,\ -1])
    $$

    $$
    groupScore = \begin{cases}
      ReduceMax(groupedValue,\ dim=-1), & groupSelectMode = 0 \\
      ReduceSum(TopK(groupedValue,\ k=2,\ dim=-1),\ dim=-1), & groupSelectMode = 1
    \end{cases}
    $$

    Select the kGroup groups with the highest scores and set the corresponding positions of the unselected groups to $-\infty$:

    $$
    groupIdx = TopK(groupScore,\ k=kGroup).indices
    $$

    $$
    normValue = Mask(groupedValue,\ groupIdx,\ fillValue=-\infty)
    $$

    **Step 4: Top-K expert selection**

    Apply Top-K to normValue to obtain the expert index. Only expertIdxOut is needed here:

    $$
    y, expertIdxOut = TopK(normValue[groupIdx, :],\ k=k)
    $$

    **Step 5: Renorm and scaling**

    Based on expertIdxOut, retrieve the corresponding k expert scores from normOut:

    $$
    gathered = normOut[\text{expertIdxOut}]
    $$

    When normType=1 or normType=2, normalization is performed; when normType=0, the renorm parameter effect applies, and renorm is performed when renorm=1:

    $$
    if\ (normType = 1\ or\ normType = 2)\ or\ (normType = 0\ and\ renorm = 1):
    $$

    $$
    \quad yOut = \frac{gathered}{ReduceSum(gathered,\ dim=-1) + eps}
    $$

    Otherwise $yOut = gathered$

    Final output:

    $$
    yOut = yOut \times routedScalingFactor
    $$

    **Step 6: Optional output**

    If outFlag is True, the third output is normOut; otherwise, it is empty.

    **Hash mode:**

    Hash mode is enabled when inputIds and tid2eid are provided:

    **Step 1: Normalization**

    Normalize the input x according to normType (same as in TopK mode):

    $$
    normOut =
    \begin{cases}
        SoftMax(x),      & normType = 0 \\
        Sigmoid(x),      & normType = 1 \\
        \sqrt{Softplus(x)},     & normType = 2
    \end{cases}
    $$

    **Step 2: Hash index lookup**

    Obtain expert indexes from the tid2eid mapping table based on inputIds:

    $$
    expertIdxOut = tid2eid[inputIds, :]
    $$

    The shape of tid2eid is [numKeys, k], and the shape of inputIds is [batch]. Each inputIds value corresponds to a row of k expert indexes.

    **Step 3: Gather and scaling**

    Obtain the corresponding k expert scores from normOut based on expertIdxOut:

    $$
    gathered = normOut[expertIdxOut]
    $$

    Normalization is performed when normType=1 or normType=2; when normType=0, the renorm parameter effect applies, and renorm is performed when renorm=1:

    $$
    if\ (normType = 1\ or\ normType = 2)\ or\ (normType = 0\ and\ renorm = 1):
    $$

    $$
    \quad yOut = \frac{gathered}{ReduceSum(gathered,\ dim=-1) + eps}
    $$

    Otherwise $yOut = gathered$

    Final output:

    $$
    yOut = yOut \times routedScalingFactor
    $$

## Function Prototype

Each operator consists of a two-phase API. Call the "aclnnMoeGatingTopKV2GetWorkspaceSizeV2" API first to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call the "aclnnMoeGatingTopKV2" API to perform the computation.

```Cpp
aclnnStatus aclnnMoeGatingTopKV2GetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *biasOptional,
  const aclTensor *inputIdsOptional,
  const aclTensor *tid2eidOptional,
  int64_t          k,
  int64_t          kGroup,
  int64_t          groupCount,
  int64_t          groupSelectMode,
  int64_t          renorm,
  int64_t          normType,
  bool             outFlag,
  double           routedScalingFactor,
  double           eps,
  const aclTensor *yOut,
  const aclTensor *expertIdxOut,
  const aclTensor *output,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnMoeGatingTopKV2(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnMoeGatingTopKV2GetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1494px"><colgroup>
  <col style="width: 146px">
  <col style="width: 110px">
  <col style="width: 301px">
  <col style="width: 219px">
  <col style="width: 328px">
  <col style="width: 101px">
  <col style="width: 143px">
  <col style="width: 146px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Input parameter to be computed, corresponding to x in the formula.</td>
      <td>None</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>biasOptional</td>
      <td>Input</td>
      <td>Bias value used in computation with input x, corresponding to bias in the formula.</td>
      <td>Data type must be the same as that of x, and the shape value is equal to the last dimension of x.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>inputIdsOptional</td>
      <td>Input</td>
      <td>Input index in Hash mode, used to look up the expert index from tid2eid. It is inputIds in the corresponding formula.</td>
      <td>The shape is [batch], equal to the first dimension of x. Required only in Hash mode.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>tid2eidOptional</td>
      <td>Input</td>
      <td>Hash mapping table, storing precomputed expert indexes. Corresponding formula: tid2eid.</td>
      <td>Shape is [numKeys, k], where numKeys is the number of rows in the mapping table, and k is equal to parameter k. Required only in Hash mode.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>k</td>
      <td>Input</td>
      <td>Value of k for topk, which is k in the corresponding formula.</td>
      <td>None</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>kGroup</td>
      <td>Input</td>
      <td>Number of groups selected after grouped sorting, which is kGroup in the corresponding formula.</td>
      <td>None</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupCount</td>
      <td>Input</td>
      <td>Total number of groups, which is groupCount in the corresponding formula.</td>
      <td>None</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupSelectMode</td>
      <td>Input</td>
      <td>Grouped sorting mode.</td>
      <td>None</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>renorm</td>
      <td>Input</td>
      <td>renorm flag.</td>
      <td>None</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>normType</td>
      <td>Input</td>
      <td>Type of the norm function.</td>
      <td>None</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outFlag</td>
      <td>Input</td>
      <td>Specifies whether to output the result of the norm operation.</td>
      <td>None</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>routedScalingFactor</td>
      <td>Input</td>
      <td>The routedScalingFactor coefficient used to compute yOut, which is the routedScalingFactor in the corresponding formula.</td>
      <td>None</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>Input</td>
      <td>Eps coefficient used to compute yOut, corresponding to eps in the formula.</td>
      <td>None</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut</td>
      <td>Output</td>
      <td>Result of performing norm and grouped sorting topk on x, corresponding to yOut in the formula.</td>
      <td>Data type must be the same as that of x.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertIdxOut</td>
      <td>Output</td>
      <td>Index of x after norm, grouped sorting, and topk, corresponding to expertIdxOut in the formula.</td>
      <td>The shape must be the same as that of yOut.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output</td>
      <td>Output</td>
      <td>Output of the norm computation, corresponding to normOut in the formula.</td>
      <td>The shape must be consistent with that of x.</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor containing the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
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
      <td>Compute input and output are null pointers.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161002</td>
      <td>The data types of input and output are not in the supported range.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_NULLPTR</td>
      <td>561103</td>
      <td>      The shape of x does not meet requirements.<br>
      The shapes of x and biasOptional mismatch.<br>
      The value of k is not in the range 1 to x_shape[-1] / groupCount * kGroup.<br>
      The value of kGroup is not in the range 1 to groupCount.<br>
      After the number of experts in each group is aligned to 32,<br>
      the values of compute input parameters do not meet requirements.<br>
      In hash mode, k exceeds 64.<br>
      In hash mode, the shape of tid2eid mismatches.<br>
      In hash mode, the shape of inputIds mismatches.<br>
      In hash mode, the non-simplified path is not supported.<br>
      </td>
    </tr>
  </tbody></table>

## aclnnMoeGatingTopKV2

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device side, obtained by the first-phase API aclnnMoeGatingTopKV2GetWorkspaceSizeV2.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnMoeGatingTopKV2 defaults to a deterministic implementation.

- Input shape constraints:
    - The last dimension of x (i.e., the number of experts) must not exceed 2048.
- Input value range restrictions:
    - Requires 1 &lt;= k &lt;= x_shape[-1] / groupCount * kGroup.
    - Requires 1 &lt;= kGroup &lt;= groupCount, and the value of kGroup * x_shape[-1] / groupCount must be greater than or equal to k.
    - Requires groupCount &gt; 0, x_shape[-1] must be divisible by groupCount and the quotient must be greater than groupSelectMode, and the result of the quotient aligned to 32 elements multiplied by groupCount must not exceed 2048.
- Other constraints:
    - groupSelectMode takes values 0 and 1. 0 indicates sorting groups by the maximum value, and 1 indicates sorting groups by the sum of topk2 values.
    - The value of normType is 0, 1, or 2. 0 indicates the Softmax function, 1 indicates the Sigmoid function, and 2 indicates the SqrtSoftplus function.
    <!-- npu="A3,910b" id7 -->
    - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: groupCount must be 1 when the value of parameter normType is 2
    <!-- end id7 -->
    - When the value of normType is 1 or 2, the renorm parameter effect does not apply; when the value of normType is 0, the renorm parameter effect applies. The value of renorm is 0 or 1, where 0 indicates no renorm and 1 indicates renorm.
    - outFlag takes values true and false, where true indicates output and false indicates no output.
- **Hash mode constraints**:
    - The Hash mode requires both inputIdsOptional and tid2eidOptional to be provided; otherwise, the TopK mode is used.
    - The Hash mode supports only the simplified path (kGroup == groupCount or groupCount == expertCount).
    - In the Hash mode, k must not be greater than 64.
    - The shape of tid2eid must be [numKeys, k], where numKeys is the total number of rows in the mapping table and k is equal to parameter k.
    - The shape of inputIds must be [batch], which is equal to the first dimension of x.
    - Each value in inputIds must be in the range [0, numKeys-1].
    - Each value in tid2eid must be in the range [0, x_shape[-1]-1], that is, a valid expert index.
    - The data types of inputIdsOptional and tid2eidOptional support the combination of INT32 and INT64.

## Example

The sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_gating_top_k_v2.h"
#include <iostream>
#include <vector>
#include <random>

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

std::vector<float> GenerateRandomFloats(int64_t count) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 10.0f);

    std::vector<float> result(count);
    for (auto& num : result) {
        num = dist(gen);
    }
    return result;
}
int Init(int32_t deviceId, aclrtStream* stream) {
  // Boilerplate: resource initialization
  auto  ret = aclInit(nullptr);
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
  // Call aclrtMalloc to allocate device-side memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host-side data to device-side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
  // Calculate strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  // Call aclCreateTensor to create aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) device/stream initialization. See the AscendCL external API list.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as required.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs. Custom construction is required based on the API definition.
  std::vector<int64_t> inputShape = {3, 256};
  std::vector<int64_t> biasShape = {256};
  std::vector<int64_t> inputIdsShape = {3};
  std::vector<int64_t> tid2eidShape = {3, 2};
  std::vector<int64_t> outShape = {3, 2};
  std::vector<int64_t> expertIdOutShape = {3, 2};
  std::vector<int64_t> normOutShape = {3, 256};

  void* inputAddr = nullptr;
  void* biasAddr = nullptr;
  void* inputIdsAddr = nullptr;
  void* tid2eidAddr = nullptr;
  void* outAddr = nullptr;
  void* expertIdOutAddr = nullptr;
  void* normOutAddr = nullptr;

  aclTensor* input = nullptr;
  aclTensor* bias = nullptr;
  aclTensor* inputIds = nullptr;
  aclTensor* tid2eid = nullptr;
  aclTensor* out = nullptr;
  aclTensor* expertIdOut = nullptr;
  aclTensor* normOut = nullptr;

  std::vector<float> inputHostData = GenerateRandomFloats(GetShapeSize(inputShape));
  std::vector<float> biasHostData = GenerateRandomFloats(GetShapeSize(biasShape));
  std::vector<int32_t> inputIdsHostData = {0, 2, 1};
  std::vector<int32_t> tid2eidHostData = {
    10, 42,
    33, 127,
    5,  88
  };
  std::vector<float> outHostData(GetShapeSize(outShape));
  std::vector<int32_t> expertIdOutHostData(GetShapeSize(expertIdOutShape));
  std::vector<float> normOutHostData(GetShapeSize(normOutShape));

  // Create input aclTensor.
  ret = CreateAclTensor(inputHostData, inputShape, &inputAddr, aclDataType::ACL_FLOAT, &input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create bias aclTensor.
  ret = CreateAclTensor(biasHostData, biasShape, &biasAddr, aclDataType::ACL_FLOAT, &bias);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create inputIds aclTensor.
  ret = CreateAclTensor(inputIdsHostData, inputIdsShape, &inputIdsAddr, aclDataType::ACL_INT32, &inputIds);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create tid2eid aclTensor.
  ret = CreateAclTensor(tid2eidHostData, tid2eidShape, &tid2eidAddr, aclDataType::ACL_INT32, &tid2eid);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create expertIdOut aclTensor.
  ret = CreateAclTensor(expertIdOutHostData, expertIdOutShape, &expertIdOutAddr, aclDataType::ACL_INT32, &expertIdOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create normOut aclTensor.
  ret = CreateAclTensor(normOutHostData, normOutShape, &normOutAddr, aclDataType::ACL_FLOAT, &normOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, and change it to the specific operator API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnMoeGatingTopKV2.
  ret = aclnnMoeGatingTopKV2GetWorkspaceSize(input, bias, inputIds, tid2eid, 2, 1, 1, 1, 0, 1, false, 1.0, 1e-20, out, expertIdOut, normOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeGatingTopKV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnMoeGatingTopKV2.
  ret = aclnnMoeGatingTopKV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeGatingTopKV2 failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output values and copy the result from the Device memory to the Host side; modify the code based on the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0.0f);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                    outAddr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar; modify the code based on the specific API definition.
  aclDestroyTensor(input);
  aclDestroyTensor(bias);
  aclDestroyTensor(inputIds);
  aclDestroyTensor(tid2eid);
  aclDestroyTensor(out);
  aclDestroyTensor(expertIdOut);
  aclDestroyTensor(normOut);

  // 7. Release Device resources; modify the code based on the specific API definition.
  aclrtFree(inputAddr);
  aclrtFree(biasAddr);
  aclrtFree(inputIdsAddr);
  aclrtFree(tid2eidAddr);
  aclrtFree(outAddr);
  aclrtFree(normOutAddr);
  aclrtFree(expertIdOutAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
