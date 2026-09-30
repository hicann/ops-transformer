# aclnnMoeGatingTopK

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference products</term>   |     ×    |
|  <term>Atlas training products</term>   |     ×    |

## Function

- Description: Applies sigmoid or softmax to input `x` during MoE computation, sorts the results by group, and selects the top-K experts based on the group sorting results.
- Formula:
   
  **Step 1: Normalization**                                                                                                                
                
  Normalize the input x based on `normType`:

  $$
  normOut = \begin{cases}
      SoftMax(x), & normType = 0 \\
      Sigmoid(x), & normType = 1
  \end{cases}
  $$

  **Step 2: Add a bias.**

  If bias is not empty, add the bias to obtain the value used for selection:

  $$
  normValue = normOut + bias
  $$

  Otherwise, $normValue = normOut$.

  **Step 3: Group filtering** (performed only when `groupCount` > 1)

  Group `normValue` by `groupCount` and calculate the score of each group based on `groupSelectMode`:

  $$
  groupedValue = Reshape(normValue,\ [batch,\ groupCount,\ -1])
  $$

  $$
  groupScore = \begin{cases}
      ReduceMax(groupedValue,\ dim=-1), & groupSelectMode = 0 \\
      ReduceSum(TopK(groupedValue,\ k=2,\ dim=-1),\ dim=-1), & groupSelectMode = 1
  \end{cases}
  $$

  Select the `kGroup` groups with the highest scores and set the corresponding positions of the unselected groups to $-\infty$:

  $$
  groupIdx = TopK(groupScore,\ k=kGroup).indices
  $$

  $$
  normValue = Mask(groupedValue,\ groupIdx,\ fillValue=-\infty)
  $$

  **Step 4: Top-K expert selection**

  Obtain the expert index by taking the top K of `normValue`. Here, only `expertIdxOut` is required.

  $$
  y, expertIdxOut = TopK(normValue[groupIdx, :],\ k=k)
  $$

  **Step 5: Renormalization and Scaling**

  When `normType` is 1, normalization is performed. When `normType` is 0, the `renorm` parameter takes effect. When `renorm` is 1, renorm is performed.

  $$
  if\ (normType = 1)\ or\ (normType = 0\ and\ renorm = 1):
  $$

  $$
  \quad yOut = \frac{normOut}{ReduceSum(normOut,\ dim=-1) + eps}
  $$

  Final output:

  $$
  yOut = yOut \times routedScalingFactor
  $$

  **Step 6: Optional output**

  If `outFlag` is `True`, the third output is `normOut`. Otherwise, the output is empty.

## Prototype

Each operator consists of [two-phase APIs](../../../docs/en/context/two_phase_api.md). You must first call the `aclnnMoeGatingTopKGetWorkspaceSize` API to obtain the required workspace size and the executor that contains the operator computation flow, and then call the `aclnnMoeGatingTopK` API to execute the computation.

```cpp
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
  const aclTensor *outOut, 
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
        <td>Input parameter for computation, corresponding to x in the formula.</td>
        <td>None</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>biasOptional</td>
        <td>Input</td>
        <td>Bias value for calculating with the input x, corresponding to bias in the formula.</td>
        <td>The shape value is the same as the last dimension of x.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>k</td>
        <td>Input</td>
        <td>K value of topk, corresponding to k in the formula.</td>
        <td>None</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>kGroup</td>
        <td>Input</td>
        <td>Number of groups obtained after grouping and sorting, corresponding to kGroup in the formula.</td>
        <td>None</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>groupCount</td>
        <td>Input</td>
        <td>Total number of groups, corresponding to groupCount in the formula.</td>
        <td>None</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>groupSelectMode</td>
        <td>Input</td>
        <td>Group sorting mode.</td>
        <td>None</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>renorm</td>
        <td>Input</td>
        <td>Renorm flag.</td>
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
        <td>Whether to output the norm operation result.</td>
        <td>None</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>routedScalingFactor</td>
        <td>Input</td>
        <td>routedScalingFactor coefficient used for calculating yOut, corresponding to routedScalingFactor in the formula.</td>
        <td>None</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>eps</td>
        <td>Input</td>
        <td>eps coefficient used for calculating yOut, corresponding to eps in the formula.</td>
        <td>None</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>yOut</td>
        <td>Output</td>
        <td>Result of the calculation after norm and top-K sorting by group are performed on x, corresponding to yOut in the formula.</td>
        <td>The data type must be the same as that of x.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>0-8</td>
        <td>-</td>
      </tr>
      <tr>
        <td>expertIdxOut</td>
        <td>Output</td>
        <td>Index of the result after norm and topK sorting by group are performed on x, corresponding to expertIdxOut in the formula.</td>
        <td>The shape must be the same as that of yOut.</td>
        <td>INT32</td>
        <td>ND</td>
        <td>0-8</td>
        <td>-</td>
      </tr>
      <tr>
        <td>outOut</td>
        <td>Output</td>
        <td>Output of the norm calculation, corresponding to normOut in the formula.</td>
        <td>The shape must be the same as that of x.</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>0-8</td>
        <td>-</td>
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

- **Returns**:

    `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

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
        <td>The input and output pointers are null.</td></td>
      </tr>
      <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td>The input and output data types are not supported.</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_TILING_ERROR</td>
        <td>561002</td>
        <td>
        The shape of x does not meet the requirements.<br>
        The shapes of x and biasOptional do not match.<br>
        The value of k is not between 1 and x_shape[-1] / groupCount × kGroup.<br>
        The value of kGroup is not between 1 and groupCount.<br>
        After the number of experts in each group is aligned by 32,<br>
        the value of the input parameter does not meet the requirements.<br>
        </td>
      </tr>
    </table>  

## aclnnMoeGatingTopK

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
        <td>Size of the workspace to be allocated on the device, obtained by calling the first-phase API aclnnMoeGatingTopKGetWorkspaceSize.</td>
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
  - `aclnnMoeGatingTopK` defaults to a deterministic implementation.

* Input shape restrictions:
    * The last dimension (i.e., the number of experts) of `x` must be less than or equal to 2048.
* Input value range restrictions:
    * 1 ≤ `k` ≤ `x_shape[-1] / groupCount * kGroup`.
    * 1 ≤ `kGroup` ≤ `groupCount`, and the value of `kGroup` * `x_shape[-1] / groupCount` must be greater than or equal to k.
    * `groupCount` > 0, `x_shape[-1]` can be exactly divided by `groupCount`, and the result is greater than `groupSelectMode`. In addition, the result of multiplying the result of 32-number alignment by `groupCount` is less than or equal to 2048.
    * `renorm` supports only 0, indicating that the norm operation is performed before the `topK` operation.
* Other restrictions:
    * The value of `groupSelectMode` can be 0 or 1. The value 0 indicates that the groups are sorted based on the maximum value, and the value 1 indicates that the groups are sorted based on the sum value of `topK2`.
    * The value of `normType` can be 0 or 1. The value 0 indicates that the softmax function is used, and the value 1 indicates that the sigmoid function is used.
    * The value of `outFlag` can be `true` or `false`. The value `true` indicates that the output is enabled, and the value `false` indicates that the output is disabled.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

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
  // (Boilerplate) Initialize resources.
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
  // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as required.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
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

  // Create an expandedPermutedRows aclTensor.
  ret = CreateAclTensor(inputHostData, inputShape, &inputAddr, aclDataType::ACL_FLOAT, &input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an expandedPermutedRows aclTensor.
  ret = CreateAclTensor(biasHostData, biasShape, &biasAddr, aclDataType::ACL_FLOAT, &bias);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an expertForSourceRow aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an expandedSrcToDstRow aclTensor.
  ret = CreateAclTensor(expertIdOutHostData, expertIdOutShape, &expertIdOutAddr, aclDataType::ACL_INT32, &expertIdOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an expandedSrcToDstRow aclTensor.
  ret = CreateAclTensor(normOutHostData, normOutShape, &normOutAddr, aclDataType::ACL_FLOAT, &normOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with a specific operator API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnMoeGatingTopK.
  ret = aclnnMoeGatingTopKGetWorkspaceSize(input, bias, 8, 4, 8, 1, 0, 1, false, 1, 1, out, expertIdOut, normOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeGatingTopKGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnMoeGatingTopK.
  ret = aclnnMoeGatingTopK(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeGatingTopK failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0.0f);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                    outAddr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Destroy aclTensor objects. Modify the code based on the API definition.
  aclDestroyTensor(input);
  aclDestroyTensor(bias);
  aclDestroyTensor(out);
  aclDestroyTensor(expertIdOut);
  aclDestroyTensor(normOut);

  // 7. Free device resources. Modify the configuration based on the API definition.
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
