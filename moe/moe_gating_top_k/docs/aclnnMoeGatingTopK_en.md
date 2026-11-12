# aclnnMoeGatingTopK

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:09:40.152Z -->

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

- Description: In MoE computation, Sigmoid, SoftMax, or SqrtSoftplus is applied to input x. Grouped sorting is performed on the computation results. Then, the top k experts are selected based on the grouped sorting result.
- Formula:

  **Step 1: Normalization**

  The input x is normalized based on normType:

$$
normOut =
\begin{cases}
    \text{SoftMax}(x),      & normType = 0 \\
    \text{Sigmoid}(x),      & normType = 1 \\
    \sqrt{\text{Softplus}(x)},     & normType = 2
\end{cases}
$$

  **Step 2: Add bias**

  If bias is not empty, add bias to obtain the value used for selection:

  $$
  normValue = normOut + bias
  $$

  Otherwise, $normValue = normOut$.

  **Step 3: Group Selection** (performed only when groupCount > 1)

  Group normValue by groupCount and calculate the score of each group based on groupSelectMode:

  $$
  groupedValue = Reshape(normValue,\ [batch,\ groupCount,\ -1])
  $$

  $$
  groupScore = \begin{cases}
      ReduceMax(groupedValue,\ dim=-1), & groupSelectMode = 0 \\
      ReduceSum(TopK(groupedValue,\ k=2,\ dim=-1),\ dim=-1), & groupSelectMode = 1
  \end{cases}
  $$

  Select the kGroup groups with the highest scores, and set the corresponding positions of unselected groups to $-\infty$:

  $$
  groupIdx = TopK(groupScore,\ k=kGroup).indices
  $$

  $$
  normValue = Mask(groupedValue,\ groupIdx,\ fillValue=-\infty)
  $$

  **Step 4: Top-K Expert Selection**

  Perform Top-K on normValue to obtain expert indices. Here, only expertIdxOut is needed:

  $$
  y, expertIdxOut = TopK(normValue[groupIdx, :],\ k=k)
  $$

  **Step 5: Renorm and Scaling**

  Retrieve the corresponding k expert scores from normOut based on expertIdxOut:

  $$
  gathered = normOut[\text{expertIdxOut}]
  $$

  When normType=1 or normType=2, normalization is performed. When normType=0, the renorm parameter takes effect, and renorm is performed when renorm=1:

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

## Function Prototype

Each operator is split into a two-phase API. You must first call the "aclnnMoeGatingTopKGetWorkspaceSize" API to obtain the workspace size required for calculation and the executor that includes calculation processes of the operator, and then call the "aclnnMoeGatingTopK" API to perform the calculation.

```Cpp
aclnnStatus aclnnMoeGatingTopKGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *biasOptional,
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
aclnnStatus aclnnMoeGatingTopK(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnMoeGatingTopKGetWorkspaceSize

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
      <th>Shape</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Input parameter to be calculated, corresponding to x in the formula.</td>
      <td>None</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>biasOptional</td>
      <td>Input</td>
      <td>Bias value calculated with input x, corresponding to `bias` in the formula.</td>
      <td>The shape value is equal to the last dimension of x.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>k</td>
      <td>Input</td>
      <td>The k value of topk, corresponding to `k` in the formula.</td>
      <td>None</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>kGroup</td>
      <td>Input</td>
      <td>Number of groups taken after grouped sorting. The corresponding formula uses `kGroup`.</td>
      <td>None</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupCount</td>
      <td>Input</td>
      <td>Total number of groups, `groupCount` in the corresponding formula.</td>
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
      <td>Indicates whether to output the norm operation result.</td>
      <td>None</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>routedScalingFactor</td>
      <td>Input</td>
      <td>The routedScalingFactor coefficient used to calculate yOut, which is `routedScalingFactor` in the corresponding formula.</td>
      <td>None</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>Input</td>
      <td>The eps coefficient used to calculate yOut, corresponding to `eps` in the corresponding formula.</td>
      <td>None</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut</td>
      <td>Output</td>
      <td>The result calculated by performing norm and grouped sorting topk on x, corresponding to `yOut` in the corresponding formula.</td>
      <td>The data type must be consistent with that of x.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertIdxOut</td>
      <td>Output</td>
      <td>Indices of x after norm and grouped sorting top-k, corresponding to `expertIdxOut` in the formula.</td>
      <td>The shape must be consistent with that of yOut.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output</td>
      <td>Output</td>
      <td>Output of the norm computation, corresponding to `normOut` in the formula.</td>
      <td>The shape must be the same as that of x.</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be side-applied on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, which includes the calculation process of the operator.</td>
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
      <td>The calculation input or calculation output is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161002</td>
      <td>The data type of the input or output is out of the supported range.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_NULLPTR</td>
      <td>561103</td>
      <td>      The shape of x does not meet requirements.<br>
      The shapes of x and biasOptional do not match.<br>
      The value of k is not in the range of 1 to x_shape[-1] / groupCount * kGroup.<br>
      The value of kGroup is not in the range of 1 to groupCount.<br>
      The total number of experts per group exceeds 2048 after being aligned to 32.<br>
      The value of the calculation input parameter does not meet requirements.<br>
      </td>
    </tr>
  </tbody></table>

## aclnnMoeGatingTopK

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
      <td>Device side-applied workspace memory address.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the Device side-applied workspace, obtained by the first-phase API aclnnMoeGatingTopKGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, includes calculation process of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnMoeGatingTopK defaults to a deterministic implementation.

- Input shape constraints:
    - The last dimension of x (i.e., the number of experts) must not exceed 2048.
- Input value range restrictions:
    - Requires 1 &lt;= k &lt;= x_shape[-1] / groupCount * kGroup.
    - Requires 1 &lt;= kGroup &lt;= groupCount, and the value of kGroup * x_shape[-1] / groupCount must be greater than or equal to k.
    - Requires groupCount &gt; 0, x_shape[-1] must be divisible by groupCount and the quotient must be greater than groupSelectMode, and the result of the quotient aligned to 32 elements multiplied by groupCount must not exceed 2048.
- Other constraints:
    - The value of groupSelectMode is 0 or 1. 0 indicates use of the maximum value for sorting groups. 1 indicates use of the sum value of topk2 for sorting groups.
    - The value of normType is 0, 1, or 2. 0 indicates use of the Softmax function. 1 indicates use of the Sigmoid function. 2 indicates use of the SqrtSoftplus function.
    <!-- npu="A3,910b" id7 -->
    - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: When normType value is 2, groupCount must be 1
    <!-- end id7 -->
    - When normType value is 1 or 2, the renorm parameter is invalid; when normType value is 0, the renorm parameter takes effect. The value of renorm is 0 or 1, where 0 indicates renorm is not performed and 1 indicates renorm is performed.
    - outFlag takes values true and false, where true indicates output and false indicates no output.
    - Empty tensors and scalar tensors are not supported for input.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_gating_top_k.h"
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
  // Boilerplate: resource initialization.
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
  // Calculate the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  // Call the aclCreateTensor API to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize the device/stream. For details, see the AscendCL public API list.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as needed.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs, which need to be custom constructed based on the API.
  std::vector<int64_t> inputShape = {3, 256};
  std::vector<int64_t> biasShape = {256};
  std::vector<int64_t> outShape = {3, 8};
  std::vector<int64_t> expertIdOutShape = {3, 8};
  std::vector<int64_t> normOutShape = {3, 256};

  void* inputAddr = nullptr;
  void* biasAddr = nullptr;
  void* outAddr = nullptr;
  void* expertIdOutAddr = nullptr;
  void* normOutAddr = nullptr;

  aclTensor* input = nullptr;
  aclTensor* bias = nullptr;
  aclTensor* out = nullptr;
  aclTensor* expertIdOut = nullptr;
  aclTensor* normOut = nullptr;

  std::vector<float> inputHostData = GenerateRandomFloats(GetShapeSize(inputShape));
  std::vector<float> biasHostData = GenerateRandomFloats(GetShapeSize(biasShape));
  std::vector<float> outHostData(GetShapeSize(outShape));
  std::vector<int32_t> expertIdOutHostData(GetShapeSize(expertIdOutShape));
  std::vector<float> normOutHostData(GetShapeSize(normOutShape));

  // Create the expandedPermutedRows aclTensor.
  ret = CreateAclTensor(inputHostData, inputShape, &inputAddr, aclDataType::ACL_FLOAT, &input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the expandedPermutedRows aclTensor.
  ret = CreateAclTensor(biasHostData, biasShape, &biasAddr, aclDataType::ACL_FLOAT, &bias);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the expandedSrcToDstRow aclTensor.
  ret = CreateAclTensor(expertIdOutHostData, expertIdOutShape, &expertIdOutAddr, aclDataType::ACL_INT32, &expertIdOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the expandedSrcToDstRow aclTensor.
  ret = CreateAclTensor(normOutHostData, normOutShape, &normOutAddr, aclDataType::ACL_FLOAT, &normOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Modify it to the specific operator API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnMoeGatingTopK.
  ret = aclnnMoeGatingTopKGetWorkspaceSize(input, bias, 8, 4, 8, 1, 0, 1, false, 1, 1, out, expertIdOut, normOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeGatingTopKGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API aclnnMoeGatingTopK.
  ret = aclnnMoeGatingTopK(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeGatingTopK failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output values, and copy the result from device memory to host memory. The code must be modified based on the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0.0f);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                    outAddr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Destroy the aclTensor and aclScalar. The code must be modified based on the specific API definition.
  aclDestroyTensor(input);
  aclDestroyTensor(bias);
  aclDestroyTensor(out);
  aclDestroyTensor(expertIdOut);
  aclDestroyTensor(normOut);

  // 7. Release device resources. The code must be modified based on the specific API definition.
  aclrtFree(inputAddr);
  aclrtFree(biasAddr);
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
