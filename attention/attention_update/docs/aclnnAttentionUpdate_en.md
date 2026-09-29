# aclnnAttentionUpdate

## Supported Products

|Product     | Supported|
|:----------------------------|:-----------:|
|Ascend 950PR/Ascend 950DT|      √     |
|<term>Atlas A3 training products/Atlas A3 inference products</term>|      √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Description

- API function: Updates the local variables `lse` and `localOut` (intermediate results output by PA operators in each SP domain) to global results.
- Formula:

$$
lse_{max} = \text{max}lse_i
$$

$$
lse = \sum_i \text{exp}(lse_i - lse_{max})
$$

$$
lse_m = lse_{max} + \text{log}(lse)
$$

$$
O = \sum_i O_i \cdot \text{exp}(lse_i - lse_m)
$$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnAttentionUpdateGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnAttentionUpdate` is called to perform computation.

```c++
aclnnStatus aclnnAttentionUpdateGetWorkspaceSize(
   const aclTensorList  *lse, 
   const aclTensorList  *localOut, 
   int64_t              updateType, 
   aclTensor            *out, 
   aclTensor            *lseOut, 
   uint64_t             *workspaceSize, 
   aclOpExecutor        **executor)
```

```c++
aclnnStatus aclnnAttentionUpdate(
   void          *workspace,
   uint64_t       workspaceSize,
   aclOpExecutor *executor,
   aclrtStream    stream)
```

## aclnnAttentionUpdateGetWorkspaceSize

- **Parameters**
  
  <table style="undefined; table-layout: fixed; width: 1567px">
    <colgroup>
      <col style="width: 170px"><!-- Name -->
      <col style="width: 120px"><!-- Input/Output -->
      <col style="width: 300px"><!-- Description -->
      <col style="width: 330px"><!-- Usage Notes -->
      <col style="width: 212px"><!-- Data Type -->
      <col style="width: 100px"><!-- Data Format -->
      <col style="width: 190px"><!-- Dimension (Shape) -->
      <col style="width: 145px"><!-- Non-contiguous Tensor -->
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
        <td>lse</td>
        <td>Input</td>
        <td>Local lse of each SP domain.</td>
        <td>The length of the tensorList is sp.</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>[batch * seqLen * headNum]</td>
        <td>x</td>
      </tr>
      <tr>
        <td>localOut</td>
        <td>Input</td>
        <td>Local attentionout of each SP domain.</td>
        <td>The length of the tensorList is sp.</td>
        <td>FLOAT32, FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>[batch * seqLen * headNum, headDim]</td>
        <td>x</td>
      </tr>
      <tr>
        <td>updateType</td>
        <td>Input</td>
        <td>Whether lseOut is output.</td>
        <td>The value can be 0 (lseOut is not output) or 1 (lseOut is output).</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out</td>
        <td>Output</td>
        <td>Output tensor.</td>
        <td>
          - 
        </td>
        <td>Same as localOut</td>
        <td>ND</td>
        <td>[batch * seqLen * headNum, headDim]</td>
        <td>x</td>
      </tr>
      <tr>
        <td>lseOut</td>
        <td>Optional output</td>
        <td>Optional output as lse_m.</td>
        <td>nullptr can be passed if lseOut is not output.</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>[batch * seqLen * headNum]</td>
        <td>x</td>
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
  <ul>
    <li><term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: support localOut and out of the FLOAT32, FLOAT16, and BFLOAT16 types.</li>
    <li>Ascend 950PR/Ascend 950DT: supports the localOut and out of FLOAT32, FLOAT16, and BFLOAT16.</li>
  </ul>

- **Returns**
  
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
  
  The first-phase API implements input parameter verification. The following errors may be thrown.
  
  <div style="overflow-x: auto;">
    <table style="table-layout: fixed; width: 1100px">
      <colgroup>
        <col style="width: 250px">
        <col style="width: 130px">
        <col style="width: 720px">
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
          <td>The passed lse, localOut, or out is a null pointer.</td>
        </tr>
        <tr>
          <!-- Add the merged-cell class to the merged cell to keep the structure consistent with the template structure. -->
          <td class="merged-cell" rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
          <td class="merged-cell" rowspan="5">161002</td>
          <td>The data type or format of the passed lse, localOut, or out is not supported.</td>
        </tr>
        <tr>
          <td>The value of the passed updateType or sp is not supported.</td>
        </tr>
        <tr>
          <td>The shape of the passed lse, localOut, or out does not meet the requirements.</td>
        </tr>
        <tr>
          <td>When updateType is 0, the passed lseOut is not nullptr.</td>
        </tr>
        <tr>
          <td>When updateType is 1, the passed lseOut is nullptr.</td>
        </tr>
      </tbody>
    </table>
  </div>

## aclnnAttentionUpdate

- **Parameters**
  
  <table><thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnAttentionUpdateGetWorkspaceSize.</td>
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

- Deterministic computing:
  - `aclnnAttentionUpdate` defaults to a deterministic implementation.
- The value range of the parallel degree `sp` for sequence parallelism is [1, 16].
- The value range of `headDim` is [8, 512] and must be a multiple of 8.
- Empty tensors are supported.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_attention_update.h"

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
  //(Fixed writing) Perform initialization.
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

  // Compute the stride of the contiguous tensor.
  std::vector<int64_t> stride(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    stride[i] = shape[i + 1] * stride[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor.
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

aclnnStatus aclnnAttentionUpdateTest(int32_t deviceId, aclrtStream& stream) {
  auto ret = Init(deviceId, &stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API.

  std::vector<int64_t> lseShape = {256};
  std::vector<int64_t> localOutShape = {256, 128};
  std::vector<int64_t> outShape = {256, 128};

  int64_t updateType = 0;

  void* lseDeviceAddr[2] = {nullptr, nullptr};
  void* localOutDeviceAddr[2] = {nullptr, nullptr};
  void* outDeviceAddr = nullptr;

  std::vector<aclTensor*> lse = {nullptr, nullptr};
  std::vector<aclTensor*> localOut = {nullptr, nullptr};
  aclTensor* out = nullptr;

  std::vector<float> lse1HostData(GetShapeSize(lseShape), 1);
  std::vector<float> lse2HostData(GetShapeSize(lseShape), 1);
  std::vector<float> localOut1HostData(GetShapeSize(localOutShape), 1);
  std::vector<float> localOut2HostData(GetShapeSize(localOutShape), 1);
  std::vector<float> outHostData(GetShapeSize(outShape), 0);

  // Create an lse aclTensorList.
  ret = CreateAclTensor(lse1HostData, lseShape, &(lseDeviceAddr[0]), aclDataType::ACL_FLOAT, &(lse[0]));
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> lse1TensorPtr(lse[0], aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> lse1DeviceAddrPtr(lseDeviceAddr[0], aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(lse2HostData, lseShape, &(lseDeviceAddr[1]), aclDataType::ACL_FLOAT, &(lse[1]));
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> lse2TensorPtr(lse[1], aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> lse2DeviceAddrPtr(lseDeviceAddr[1], aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  aclTensorList *lseList = aclCreateTensorList(lse.data(), lse.size());

  // Create a localOut aclTensorList.
  ret = CreateAclTensor(localOut1HostData, localOutShape, &(localOutDeviceAddr[0]), aclDataType::ACL_FLOAT, &(localOut[0]));
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> localOut1TensorPtr(localOut[0], aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> localOut1DeviceAddrPtr(localOutDeviceAddr[0], aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(localOut2HostData, localOutShape, &(localOutDeviceAddr[1]), aclDataType::ACL_FLOAT, &(localOut[1]));
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> localOut2TensorPtr(localOut[1], aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> localOut2DeviceAddrPtr(localOutDeviceAddr[1], aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  aclTensorList *localOutList = aclCreateTensorList(localOut.data(), localOut.size());

  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnAttentionUpdate.
  ret = aclnnAttentionUpdateGetWorkspaceSize(lseList,
                                       localOutList,
                                       updateType,
                                       out,
                                       nullptr,
                                       &workspaceSize,
                                       &executor);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAttentionUpdateGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the computed workspaceSize.
  void* workspaceAddr = nullptr;
  std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    workspaceAddrPtr.reset(workspaceAddr);
  }
  // Call the second-phase API of aclnnAttentionUpdate.
  ret = aclnnAttentionUpdate(workspaceAddr, workspaceSize, executor, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAttentionUpdate failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> outData(size, 0);
  ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(outData[0]), outDeviceAddr,
                    size * sizeof(outData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("out result[%ld] is: %f\n", i, outData[i]);
  }

  return ACL_SUCCESS;
}

int main() {
  // 1. (Fixed format) Initialize the device and stream. For details, see the external API list.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = aclnnAttentionUpdateTest(deviceId, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAttentionUpdateTest failed. ERROR: %d\n", ret); return ret);

  Finalize(deviceId, stream);
  return 0;
}
```
