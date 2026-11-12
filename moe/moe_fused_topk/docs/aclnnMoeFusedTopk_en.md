# aclnnMoeFusedTopk

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:07:04.351Z -->

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

- Description: In MoE computation, performs Sigmoid computation on the input x, sorts the computation results by group, and finally selects the top k experts based on the grouped sorting results.
- Formula:

    Apply sigmoid to the input:

    $$
    sigmoidRes=sigmoid(x)
    $$

    Add addNum:

    $$
    normOut = sigmoidRes + addNum
    $$

    Group the computation results by groupNum, sort the groups within each group by the sum of topN values, and take the top groupTopk groups:

    $$
    groupOut, groupId = TopK(ReduceSum(TopK(Split(normOut, groupCount), k=2, dim=-1), dim=-1),k=kGroup)
    $$

    Based on the groupId from the previous step, retrieve the corresponding elements from normOut, then perform TopK on the data again to obtain the indices result:

    $$
    normY,indices=TopK(normOut[groupId, :],k=k)
    $$

    Select y from sigmoidRes based on indices:

    $$
    y = gather(sigmoidRes, indices)
    $$

    If isNorm is true, compute y based on the input scale parameter to obtain the result of y:

    $$
    y = y / (ReduceSum(y, dim=-1))*scale
    $$

    If enableExpertMapping is true, the physical experts in indices are then mapped to logical experts based on the input mappingNum and mappingTable to obtain the output indices.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnMoeFusedTopkGetWorkspaceSize" API to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the "aclnnMoeFusedTopk" API to perform the computation.

```Cpp
aclnnStatus aclnnMoeFusedTopkGetWorkspaceSize(
  const aclTensor* x,
  const aclTensor* addNum,
  const aclTensor* mappingNum,
  const aclTensor* mappingTable,
  uint32_t         groupNum,
  uint32_t         groupTopk,
  uint32_t         topN,
  uint32_t         topK,
  uint32_t         activateType,
  bool             isNorm,
  float            scale,
  bool             enableExpertMapping,
  aclTensor*       y,
  aclTensor*       indices,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnMoeFusedTopk(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnMoeFusedTopkGetWorkspaceSize

- **Parameters**:

  <table style="undefined;table-layout: fixed; width: 1517px"><colgroup>
  <col style="width: 216px">
  <col style="width: 134px">
  <col style="width: 321px">
  <col style="width: 204px">
  <col style="width: 165px">
  <col style="width: 130px">
  <col style="width: 196px">
  <col style="width: 151px">
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
      <td>Score of each token for each expert.</td>
      <td>-</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>(numToken, expertNum)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>addNum</td>
      <td>Input</td>
      <td>Bias value to be computed with input x.</td>
      <td>-</td>
      <td>Same as x</td>
      <td>ND</td>
      <td>(expertNum)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mappingNum</td>
      <td>Input</td>
      <td>Number of logical experts that each physical expert is actually mapped to.</td>
      <td>Not enabled when enableExpertMapping is false.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(expertNum)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mappingTable</td>
      <td>Input</td>
      <td>Mapping table of each physical expert/logical expert.</td>
      <td>Not enabled when enableExpertMapping is false.<br>Less than or equal to 128.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(expertNum, maxMappingNum)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>groupNum</td>
      <td>Input</td>
      <td>Number of groups. Must be greater than 0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupTopk</td>
      <td>Input</td>
      <td>Number of selected groups, must be greater than 0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>topN</td>
      <td>Input</td>
      <td>Number of experts selected in a group for summation, must be greater than 0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>topK</td>
      <td>Input</td>
      <td>Number of experts finally selected, must be greater than 0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activateType</td>
      <td>Input</td>
      <td>Activation type. Currently, only 0 (ACTIVATION_SIGMOID) is supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>isNorm</td>
      <td>Input</td>
      <td>Whether to normalize the output.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Input</td>
      <td>Coefficient multiplication after normalization.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>enableExpertMapping</td>
      <td>Input</td>
      <td>Whether to enable the mapping from physical experts to logical experts.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>The aclTensor on the device.</td>
      <td>-</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(numToken, topK)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Output</td>
      <td>aclTensor on the device.</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(numToken, topK)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be applied for on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **Return value**:

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 287px">
  <col style="width: 119px">
  <col style="width: 743px">
  </colgroup>
  <thead>
    <tr>
      <th>return value</th>
      <th>Error code</th>
      <th>description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="3">161001</td>
      <td>Input x or addNum is a null pointer.</td>
    </tr>
    <tr>
      <td>Output y or indices is a null pointer.</td>
    </tr>
    <tr>
      <td>If enableExpertMapping is true, input mappingNum or mappingTable is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type or data format of the input or output is out of the supported range.</td>
    </tr>
    <tr>
      <td>The input parameters do not meet the constraints.</td>
    </tr>
    <tr>
      <td>The input or output shape does not meet the constraints.</td>
    </tr>
  </tbody>
  </table>

## aclnnMoeFusedTopk

- **Parameters**:

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
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
      <td>Memory address of the workspace allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the device, obtained by the first-phase API aclnnMoeFusedTopkGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return value, which is a status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnMoeFusedTopk defaults to a deterministic implementation.

- expertNum must be an integer multiple of groupNum.
- groupTopk is less than or equal to groupNum.
- maxMappingNum must be less than or equal to 128.
- TopK must be less than or equal to expertNum.
- TopN must be less than or equal to expertNum / groupNum.
- expertNum must be less than or equal to 1024.
- groupNum must be less than or equal to 256.

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_fused_topk.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
  do {                                     \
      if (!(cond)) {                       \
          Finalize(deviceId, stream);      \
          return_expr;                     \
      }                                    \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // Boilerplate, initialization.
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
  // Call aclrtMalloc to allocate device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host data to device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the stride of the contiguous tensor.
  std::vector<int64_t> stride(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    stride[i] = shape[i + 1] * stride[i + 1];
  }

  // Call the aclCreateTensor API to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, stride.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
}

int aclnnMoeFusedTopkTest(int32_t deviceId, aclrtStream& stream) {
  auto ret = Init(deviceId, &stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs. Custom construction is required based on the API.

  int64_t num_token = 16;
  int64_t expert_num = 32;
  int64_t max_mapping_num = 16;

  uint32_t groupNum = 2;
  uint32_t groupTopk = 2;
  uint32_t topN = 2;
  uint32_t topK = 4;
  uint32_t activateType = 0;
  bool isNorm = false;
  float scale = 1.0;
  bool enableExpertMapping = true;

  std::vector<int64_t> xShape = {num_token, expert_num};
  std::vector<int64_t> addNumShape = {expert_num};
  std::vector<int64_t> mappingNumShape = {expert_num};
  std::vector<int64_t> mappingTableShape = {expert_num, max_mapping_num};

  std::vector<int64_t> yShape = {num_token, topK};
  std::vector<int64_t> indicesShape = {num_token, topK};

  void* xDeviceAddr = nullptr;
  void* addNumDeviceAddr = nullptr;
  void* mappingNumDeviceAddr = nullptr;
  void* mappingTableDeviceAddr = nullptr;

  void* yDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;

  aclTensor* x = nullptr;
  aclTensor* addNum = nullptr;
  aclTensor* mappingNum = nullptr;
  aclTensor* mappingTable = nullptr;

  aclTensor* y = nullptr;
  aclTensor* indices = nullptr;

  std::vector<float> xHostData(GetShapeSize(xShape), 1);
  std::vector<float> addNumHostData(GetShapeSize(addNumShape), 1);
  std::vector<int32_t> mappingNumHostData(GetShapeSize(mappingNumShape), 1);
  std::vector<int32_t> mappingTableHostData(GetShapeSize(mappingTableShape), 1);

  std::vector<float> yHostData(GetShapeSize(yShape), 0);
  std::vector<int32_t> indicesHostData(GetShapeSize(indicesShape), 0);

  // Create the x aclTensor.
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> xTensorPtr(x, aclDestroyTensor);

  // Create the addNum aclTensor.
  ret = CreateAclTensor(addNumHostData, addNumShape, &addNumDeviceAddr, aclDataType::ACL_FLOAT, &addNum);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> addNumTensorPtr(addNum, aclDestroyTensor);

  // Create the mappingNum aclTensor.
  ret = CreateAclTensor(mappingNumHostData, mappingNumShape, &mappingNumDeviceAddr, aclDataType::ACL_INT32, &mappingNum);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> mappingNumTensorPtr(mappingNum, aclDestroyTensor);

  // Create the mappingTable aclTensor.
  ret = CreateAclTensor(mappingTableHostData, mappingTableShape, &mappingTableDeviceAddr, aclDataType::ACL_INT32, &mappingTable);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> mappingTableTensorPtr(mappingTable, aclDestroyTensor);

  // Create the y aclTensor.
  ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> yTensorPtr(y, aclDestroyTensor);

  // Create the indices aclTensor.
  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT32, &indices);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> indicesTensorPtr(indices, aclDestroyTensor);

  // 3. Call the CANN operator library API. Replace Api with the actual API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the aclnnMoeFusedTopk first-phase API.
  ret = aclnnMoeFusedTopkGetWorkspaceSize(x,
                                       addNum,
                                       mappingNum,
                                       mappingTable,
                                       groupNum,
                                       groupTopk,
                                       topN,
                                       topK,
                                       activateType,
                                       isNorm,
                                       scale,
                                       enableExpertMapping,
                                       y,
                                       indices,
                                       &workspaceSize,
                                       &executor);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeFusedTopkGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    workspaceAddrPtr.reset(workspaceAddr);
  }
  // Call the aclnnMoeFusedTopk second-phase API.
  ret = aclnnMoeFusedTopk(workspaceAddr, workspaceSize, executor, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeFusedTopk failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from device memory to host memory. Modify it according to the specific API definition.
  auto size = GetShapeSize(yShape);
  std::vector<float> yData(size, 0);
  ret = aclrtMemcpy(yData.data(), yData.size() * sizeof(yData[0]), yDeviceAddr,
                    size * sizeof(yData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("out result[%ld] is: %f\n", i, yData[i]);
  }

  return ACL_SUCCESS;
}

int main() {
  // 1. (boilerplate) Initialize device/stream. See the external API list.
  // Fill in deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = aclnnMoeFusedTopkTest(deviceId, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeFusedTopkTest failed. ERROR: %d\n", ret); return ret);

  Finalize(deviceId, stream);
  return 0;
}
```
