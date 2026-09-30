# aclnnMoeFinalizeRouting

## Supported Products

| Product                                                        |  Supported  |
| :----------------------------------------------------------- |:-------:|
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference products</term>                            |    ×    |
| <term>Atlas training products</term>                             |    ×    |

## Function

- Processes and combines the MoE FFN output during MoE computation.
- Formula:

    $$
    expertid=expandedExpertIdx[i,k]
    $$
    
    $$
    out(i,j)=x1_{i,j}+x2Optional_{i,j}+\sum_{k=0}^{K}(scales_{i,k}*(expandedX_{expandedRowIdx_{i+k*num_rows},j}+bias_{expertid,j}))
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnMoeFinalizeRoutingGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnMoeFinalizeRouting` is called to perform computation.

```cpp
aclnnStatus aclnnMoeFinalizeRoutingGetWorkspaceSize(
  const aclTensor* expandedX, 
  const aclTensor* x1, 
  const aclTensor* x2Optional, 
  const aclTensor* bias, 
  const aclTensor* scales, 
  const aclTensor* expandedRowIdx, 
  const aclTensor* expandedExpertIdx, 
  const aclTensor* out, 
  uint64_t*        workspaceSize, 
  aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnMoeFinalizeRouting(
  void*          workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor* executor, 
  aclrtStream    stream)
```

## aclnnMoeFinalizeRoutingGetWorkspaceSize

- **Parameters:**
    <table style="undefined;table-layout: fixed; width: 1546px"><colgroup>
    <col style="width: 179px">
    <col style="width: 122px">
    <col style="width: 271px">
    <col style="width: 250px">
    <col style="width: 207px">
    <col style="width: 123px">
    <col style="width: 241px">
    <col style="width: 153px">
    </colgroup>
    <thead>
      <tr>
        <th>Name</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Usage Notes</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Shape</th>
        <th>Non-contiguous Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>expandedX</td>
        <td>Input</td>
        <td>expandedX in the formula, which is the FFN output of MoE.</td>
        <td>The value must be a 2D tensor.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>(NUM_ROWS × K, H)<br>NUM_ROWS: number of rows<br>K: number of experts selected from the total experts E<br>H: number of columns </td>
        <td>-</td>
      </tr>
      <tr>
        <td>x1</td>
        <td>Input</td>
        <td> x1 in the formula.</td>
        <td>The value must be a 2D tensor.</td>
        <td>Same as expandedX.</td>
        <td>-</td>
        <td>Same as out.</td>
        <td>-</td>
      </tr>
      <tr>
        <td>x2Optional</td>
        <td>Input</td>
        <td> x2Optional in the formula.</td>
        <td>The value must be a 2D tensor.</td>
        <td>Same as expandedX.</td>
        <td>-</td>
        <td>Same as out.</td>
        <td>-</td>
      </tr>
      <tr>
        <td>bias</td>
        <td>Input</td>
        <td>Bias in the formula.</td>
        <td>The value must be a 2D tensor.</td>
        <td>Same as expandedX.</td>
        <td>-</td>
        <td>(E, H)<br>E: total number of experts, H: number of columns </td>
        <td>-</td>
      </tr>
      <tr>
        <td>scales</td>
        <td>Input</td>
        <td> scales in the formula.</td>
        <td>The value must be a 2D tensor.</td>
        <td>Same as expandedX.</td>
        <td>-</td>
        <td>(NUM_ROWS, K)</td>
        <td>-</td>
      </tr>
      <tr>
        <td>expandedRowIdx</td>
        <td>Input</td>
        <td>expandedRowIdx in the formula.</td>
        <td>The value must be a 1D tensor.<br>The value range of the tensor is [0, NUM_ROWS × K – 1].</td>
        <td>INT32</td>
        <td>-</td>
        <td>(NUM_ROWS × K)</td>
        <td>-</td>
      </tr>
      <tr>
        <td>expandedExpertIdx</td>
        <td>Input</td>
        <td>expandedExpertIdx in the formula.</td>
        <td>The value must be a 2D tensor.<br>The value range of the tensor is [0, E – 1], where E indicates the total number of experts.</td>
        <td>INT32</td>
        <td>-</td>
        <td>(NUM_ROWS, K)</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out</td>
        <td>Output</td>
        <td>Output in the formula.</td>
        <td>The value must be a 2D tensor.</td>
        <td>Same as expandedX.</td>
        <td></td>
        <td>(NUM_ROWS, H)</td>
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

  The first-phase API implements input parameter verification. The following errors may be thrown:

  <table style="undefined;table-layout: fixed; width: 1202px"><colgroup>
  <col style="width: 301px">
  <col style="width: 121px">
  <col style="width: 780px">
  </colgroup>
  <thead>
    <tr>
      <th>Return</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>Returned when the passed `expandedX`, `x1`, `x2Optional`, `bias`, `scales`, `expandedRowIdx`, or `expandedExpertIdx` is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data type of `expandedX`, `x1`, `x2Optional`, `bias`, `scales`, `expandedRowIdx`, or `expandedExpertIdx` is not within the supported range.</td>
    </tr>
    <tr>
      <td>The shape of `expandedX`, `x1`, `x2Optional`, `bias`, `scales`, `expandedRowIdx`, or `expandedExpertIdx` is not within the supported range.</td>
    </tr>
  </tbody>
  </table>

## aclnnMoeFinalizeRouting

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1154px"><colgroup>
  <col style="width: 153px">
  <col style="width: 121px">
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
      <td>Size of the workspace to be allocated on the Device side, obtained by calling the first-phase API `aclnnMoeFinalizeRoutingGetWorkspaceSize`.</td>
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
  - `aclnnMoeFinalizeRouting` defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_finalize_routing.h"
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
  std::vector<int64_t> expandedXShape = {3 * 2, 4};
  std::vector<int64_t> x1Shape = {3, 4};
  std::vector<int64_t> x2OptionalShape = {3, 4};
  std::vector<int64_t> biasShape = {2, 4};
  std::vector<int64_t> scalesShape = {3, 2};
  std::vector<int64_t> expandedExpertIdxShape = {3, 2};
  std::vector<int64_t> expandedRowIdxShape = {3 * 2};
  std::vector<int64_t> outShape = {3, 4};
  void* expandedXAddr = nullptr;
  void* x1Addr = nullptr;
  void* x2OptionalAddr = nullptr;
  void* biasAddr = nullptr;
  void* scalesDeviceAddr = nullptr;
  void* expandedExpertIdxAddr = nullptr;
  void* expandedRowIdxAddr = nullptr;
  void* outDeviceAddr = nullptr;
  
  aclTensor* expandedX = nullptr;
  aclTensor* x1 = nullptr;
  aclTensor* x2Optional = nullptr;
  aclTensor* bias = nullptr;
  aclTensor* scales = nullptr;
  aclTensor* expandedExpertIdx = nullptr;
  aclTensor* expandedRowIdx = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> expandedXHostData = {0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1,
                                                     0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1};
  std::vector<float> x1HostData = {0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 10.2, 11.2};
  std::vector<float> x2OptionalHostData = {0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 10.2, 11.2};
  std::vector<float> biasHostData = {0.2, 0.4, 0.2, 0.4, 0.2, 0.4, 0.2, 0.4};
  std::vector<float> scalesHostData = {1.3, 1.6, 1.2, 1.8, 1.2, 2.3};
  std::vector<int32_t> expandedExpertIdxHostData = {0, 1, 0, 1, 0, 1};
  std::vector<int32_t> expandedRowIdxHostData = {2, 1, 4, 3, 0, 5};
  std::vector<float> outHostData(12, 0.0f);
  // Create an expandedX aclTensor.
  ret = CreateAclTensor(expandedXHostData, expandedXShape, &expandedXAddr,
                        aclDataType::ACL_FLOAT, &expandedX);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an x1 aclTensor.
  ret = CreateAclTensor(x1HostData, x1Shape, &x1Addr, aclDataType::ACL_FLOAT, &x1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an x2Optional aclTensor.
  ret = CreateAclTensor(x2OptionalHostData, x2OptionalShape, &x2OptionalAddr, aclDataType::ACL_FLOAT, &x2Optional);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a bias aclTensor.
  ret = CreateAclTensor(biasHostData, biasShape, &biasAddr, aclDataType::ACL_FLOAT, &bias);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a totalWeightOut aclTensor.
  ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_FLOAT, &scales);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  // Create an expandedExpertIdx aclTensor.
  ret = CreateAclTensor(expandedExpertIdxHostData, expandedExpertIdxShape, &expandedExpertIdxAddr,
                        aclDataType::ACL_INT32, &expandedExpertIdx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an expandedRowIdx aclTensor.
  ret = CreateAclTensor(expandedRowIdxHostData, expandedRowIdxShape, &expandedRowIdxAddr,
                        aclDataType::ACL_INT32, &expandedRowIdx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  //Create an Out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with a specific operator API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnMoeFinalizeRouting.
  ret = aclnnMoeFinalizeRoutingGetWorkspaceSize(expandedX, x1, x2Optional, bias, scales,
                                                expandedRowIdx, expandedExpertIdx, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeFinalizeRoutingGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the computed workspaceSize.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnMoeFinalizeRouting.
  ret = aclnnMoeFinalizeRouting(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeFinalizeRouting failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0.0f);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                    outDeviceAddr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
      LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Destroy aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(expandedX);
  aclDestroyTensor(x1);
  aclDestroyTensor(x2Optional);
  aclDestroyTensor(bias);
  aclDestroyTensor(scales);
  aclDestroyTensor(expandedExpertIdx);
  aclDestroyTensor(expandedRowIdx);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the code based on the API definition.
  aclrtFree(expandedXAddr);
  aclrtFree(x1Addr);
  aclrtFree(x2OptionalAddr);
  aclrtFree(biasAddr);
  aclrtFree(scalesDeviceAddr);
  aclrtFree(expandedExpertIdxAddr);
  aclrtFree(expandedRowIdxAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
