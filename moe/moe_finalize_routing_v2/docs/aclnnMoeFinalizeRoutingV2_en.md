# aclnnMoeFinalizeRoutingV2

[📄 View Source Code](https://gitcode.com/cann/ops-transformer/tree/master/moe/moe_finalize_routing_v2)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                     |     √    |
| <term>Atlas A3 training products/Atlas A3 inference products</term>     |    √    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>     |    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- **Description**: Processes and combines the MoE FFN output during MoE computation.
- **Formula**:

  $$
  expertid=expertIdx[i,k]
  $$
  
  $$
  out(i,j)=x1_{i,j}+x2_{i,j}+\sum_{k=0}^{K}(scales_{i,k}*(expandedX_{expandedRowIdx_{i+k*num\_rows},j}+bias_{expertid,j}))
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnMoeFinalizeRoutingV2GetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnMoeFinalizeRoutingV2` is called to perform computation.

```c++
aclnnStatus aclnnMoeFinalizeRoutingV2GetWorkspaceSize(
    const aclTensor *expandedX,
    const aclTensor *expandedRowIdx,
    const aclTensor *x1Optional,
    const aclTensor *x2Optional,
    const aclTensor *biasOptional,
    const aclTensor *scalesOptional,
    const aclTensor *expertIdxOptional,
    int64_t          dropPadMode,
    const aclTensor *out,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```c++
aclnnStatus aclnnMoeFinalizeRoutingV2(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnMoeFinalizeRoutingV2GetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1522px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 315px">
  <col style="width: 231px">
  <col style="width: 210px">
  <col style="width: 121px">
  <col style="width: 210px">
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
    <th>Shape</th>
    <th>Non-contiguous Tensor</th>
  </tr></thead>
  <tbody>
  <tr>
        <td>expandedX</td>
        <td>Input</td>
        <td>`expandedX` in the formula, which is the FFN output of MoE.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>Drop less scenario: (NUM_ROWS × K, H),<br>Drop pad scenario: (E, C, H).</td>
        <td>√</td>
  </tr>
  <tr>
    <td>expandedRowIdx</td>
    <td>Input</td>
    <td>`expandedRowIdx` in the formula.</td>
    <td>-</td>
    <td>INT32</td>
    <td>ND</td>
    <td>(NUM_ROWS × K)</td>
    <td>√</td>
  </tr>
  <tr>
    <td>x1Optional</td>
    <td>Input</td>
    <td>`x1` in the formula, indicating the first shared expert.</td>
    <td>-</td>
    <td>Same as `expandedX`.</td>
    <td>ND</td>
    <td>Same as `out`.</td>
    <td>√</td>
  </tr>
  <tr>
    <td>x2Optional</td>
    <td>Input</td>
    <td>`x2` in the formula, indicating the second shared expert.</td>
    <td>-</td>
    <td>Same as `expandedX`.</td>
    <td>ND</td>
    <td>Same as `out`.</td>
    <td>√</td>
  </tr>
  <tr>
    <td>biasOptional</td>
    <td>Input</td>
    <td>Bias in the formula, indicating the bias value.</td>
    <td>-</td>
    <td>Same as `expandedX`.</td>
    <td>ND</td>
    <td>(E, H)</td>
    <td>√</td>
  </tr>
  <tr>
    <td>scalesOptional</td>
    <td>Input</td>
    <td>`scales` in the formula.</td>
    <td>-</td>
    <td>FLOAT16, BFLOAT16, FLOAT32</td>
    <td>ND</td>
    <td>(NUM_ROWS, K)</td>
    <td>√</td>
  </tr>
  <tr>
    <td>expertIdxOptional</td>
    <td>Input</td>
    <td>`expertIdx` in the formula.</td>
    <td>The value range of the tensor is [0, E-1].</td>
    <td>INT32</td>
    <td>ND</td>
    <td>(NUM_ROWS, K)</td>
    <td>√</td>
  </tr>
  <tr>
    <td>dropPadMode</td>
    <td>Input</td>
    <td>Whether the discard mode is supported, and the arrangement mode of `expandedRowIdx`.</td>
    <td>The value range is [0, 3].</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>out</td>
    <td>Output</td>
    <td>Output in the formula.</td>
    <td>-</td>
    <td>Same as `expandedX`.</td>
    <td>ND</td>
    <td>(NUM_ROWS, H)</td>
    <td>×</td>
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
  </tbody></table>

  - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>:
    - `expandedX` must be a 2D or 3D tensor. The supported data types are FLOAT16, BFLOAT16, and FLOAT32. The drop less and drop pad scenarios are supported.
    - `scalesOptional`: In mixed precision mode, if `expandedX` is BFLOAT16, `scalesOptional` can be FLOAT32. In non-mixed precision mode, the data type must be the same as that of `expandedX`.
  - Ascend 950PR/Ascend 950DT:
    - `expandedX` must be a 2D or 3D tensor. The supported data types are FLOAT16, BFLOAT16, and FLOAT32. The drop less and drop pad scenarios are supported.
    - The data type of scalesOptional can be different from that of expandedX.
  - <term>Atlas inference products</term>:
    - `expandedX` must be a 2D tensor of type FLOAT16 or FLOAT32. The shape of expandedX must be 32-pixel aligned along the last axis (H).
    - `x1Optional`, x2Optional`,`biasOptional`, and`expertIdxOptional` support only null pointers.
    - Only 2 can be passed to `dropPadMode`.
    - The data type of `scalesOptional` can be FLOAT16 or FLOAT32, and it must be the same as that of `expandedX`.

- **Returns**:

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown:

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 253px">
  <col style="width: 140px">
  <col style="width: 762px">
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
    <td> ACLNN_ERR_PARAM_NULLPTR </td>
    <td> 161001 </td>
    <td>The required input, output, or attribute is passed as a null pointer.</td>
    </tr>
    <tr>
    <td> ACLNN_ERR_PARAM_INVALID </td>
    <td> 161002 </td>
    <td>The input and output data types and formats are not supported.</td>
    </tr>
    <tr>
    <td rowspan="2"> ACLNN_ERR_INNER_TILING_ERROR </td>
    <td rowspan="2"> 561002 </td>
    <td>The shapes of multiple input tensors do not match.</td>
    </tr>
    <tr>
    <td>The shape of the input attribute does not match that of the input tensor.</td>
    </tr>
  </tbody></table>

## aclnnMoeFinalizeRoutingV2

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
    <td>Size of the workspace to be allocated on the device, which is obtained by calling the first API `aclnnMoeFinalizeRoutingV2GetWorkspaceSize`.</td>
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

- **Returns**
  
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

1. Deterministic computing:
    - `aclnnMoeFinalizeRoutingV2` defaults to a deterministic implementation.

2. `NUM_ROWS`: number of rows. 
    - `K`: number of experts selected from the total experts E 
    - `H`: hidden size, that is, the length of each token sequence, which is the number of columns. 
    - `E`: expert num, indicating the number of experts. E must be greater than or equal to `K`. 
    - `C`: expert capacity, that is, the threshold of the number of tokens that can be processed by an expert. 

3. `expandedRowIdx`: When `dropPadMode` is set to 0 or 2, the value range of the tensor is [0, NUM_ROWS × K – 1]. When `dropPadMode` is set to 1 or 3, the value range of the tensor is [–1, E × C – 1].

4. `x1Optional` must be specified before `x2Optional` is configured.

5. If `scalesOptional` does not exist, `K` is 1.

6. If `biasOptional` exists, `expertIdxOptional` must also exist.

7. The values and meanings of `dropPadMode` are as follows:
    - 0: In the dropless scenario, `expandedRowIdx` is arranged by column (corresponding to the output format of [aclnnMoeInitRouting](../../moe_init_routing/docs/aclnnMoeInitRouting_en.md)).
    - 1: In the drop and pad scenario, `expandedRowIdx` is arranged by column (corresponding to the output format of [aclnnMoeInitRouting](../../moe_init_routing/docs/aclnnMoeInitRouting_en.md)).
    - 2: In the dropless scenario, `expandedRowIdx` is arranged by row (corresponding to the output format of [aclnnMoeInitRoutingV2](../../moe_init_routing_v2/docs/aclnnMoeInitRoutingV2_en.md)).
    - 3: In the drop and pad scenario, `expandedRowIdx` is arranged by row (corresponding to the output format of [aclnnMoeInitRoutingV2](../../moe_init_routing_v2/docs/aclnnMoeInitRoutingV2_en.md)).

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_finalize_routing_v2.h"
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
  int64_t dropPadMode = 0;
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

  // Call the first-phase API of aclnnMoeFinalizeRoutingV2.
  ret = aclnnMoeFinalizeRoutingV2GetWorkspaceSize(expandedX, expandedRowIdx, x1, x2Optional, bias, scales,
                                                  expandedExpertIdx, dropPadMode, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeFinalizeRoutingV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the computed workspaceSize.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnMoeFinalizeRoutingV2.
  ret = aclnnMoeFinalizeRoutingV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeFinalizeRoutingV2 failed. ERROR: %d\n", ret); return ret);

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

  // 7. Free device resources. Modify the configuration based on the API definition.
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
