
# aclnnFusedFloydAttention

## Supported Products

|Product     | Supported|
|:----------------------------|:-----------:|
|Ascend 950PR/Ascend 950DT|      ×     |
|<term>Atlas A3 training products/Atlas A3 inference products</term>|     √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>|     √     |
|<term>Atlas 200I/500 A2 inference products</term>|     ×      |
|<term>Atlas inference products</term>|     ×     |
|<term>Atlas training products</term>|     ×     |

## Function

- Description: Uses the FloydAttention algorithm to perform multidimensional self-attention computation in training scenarios.

- Formulas:

    The forward propagation formula for attention is as follows:

    $$
    weights = Softmax(attenMask + scale*(einsum(query, key1^T) + einsum(query, key2^T)))
    $$

    $$
    attention\_out = einsum(weights, value1) + einsum(weights, value2)
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnFusedFloydAttentionGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnFusedFloydAttention` is called to perform computation.

```Cpp
aclnnStatus aclnnFusedFloydAttentionGetWorkspaceSize(
    const aclTensor *query, 
    const aclTensor *key1, 
    const aclTensor *value1, 
    const aclTensor *key2, 
    const aclTensor *value2, 
    const aclTensor *attenMaskOptional, 
    double           scaleValueOptional, 
    const aclTensor *softmaxMaxOut, 
    const aclTensor *softmaxSumOut, 
    const aclTensor *attentionOutOut, 
    uint64_t        *workspaceSize, 
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnFusedFloydAttention(
    void             *workspace, 
    uint64_t          workspaceSize, 
    aclOpExecutor    *executor, 
    aclrtStream       stream)
```

## aclnnFusedFloydAttentionGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1565px"><colgroup>
      <col style="width: 146px">
      <col style="width: 135px">
      <col style="width: 326px">
      <col style="width: 246px">
      <col style="width: 275px">
      <col style="width: 120px">
      <col style="width: 171px">
      <col style="width: 146px">
      </colgroup>
      <thead>
        <tr>
          <th>Name</th>
          <th>Input/Output</th>
          <th>Description</th>
          <th>Usage</th>
          <th>Data Type</th>
          <th>Data Format</th>
          <th>Dimension (Shape)</th>
          <th>Non-contiguous Tensor</th>
        </tr></thead>
      <tbody>
        <tr>
          <td>query</td>
          <td>Input</td>
          <td><code>query</code> in the formulas.</td>
          <td>The data type is the same as that of key1/value1/key2/value2.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,N,M,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>key1</td>
          <td>Input</td>
          <td>key1 in the formula.</td>
          <td>The data type is the same as that of query/value1/key2/value2.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,N,K,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>value1</td>
          <td>Input</td>
          <td>value1 in the formula.</td>
          <td>The data type is the same as that of query/key1/key2/value2.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,N,K,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>key2</td>
          <td>Input</td>
          <td>key2 in the formula.</td>
          <td>The data type is the same as that of query/key1/value1/value2.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,K,M,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>value2</td>
          <td>Input</td>
          <td>value2 in the formula.</td>
          <td>The data type is the same as that of query/key1/value1/key2.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,K,M,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>attenMaskOptional</td>
          <td>Input</td>
          <td>attenMask in the formula.</td>
          <td>A value of 1 indicates that the position does not participate in the calculation, while a value of 0 indicates that it does.</td>
          <td>BOOL or UINT8</td>
          <td>ND</td>
          <td>[B,1,N,1,K]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>scaleValueOptional</td>
          <td>Input</td>
          <td><code>scale</code> in the formula, indicating the scaling coefficient.</td>
          <td>-</td>
          <td>DOUBLE</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>softmaxMaxOut</td>
          <td>Output</td>
          <td>Intermediate output of the forward attention calculation.</td>
          <td>The output shape is [B,H,N,M,8].</td>
          <td>FLOAT</td>
          <td>ND</td>
          <td>[B,H,N,M,8]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>softmaxSumOut</td>
          <td>Output</td>
          <td>Intermediate output of the forward attention calculation.</td>
          <td>The output shape is [B,H,N,M,8].</td>
          <td>FLOAT</td>
          <td>ND</td>
          <td>[B,H,N,M,8]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>attentionOutOut</td>
          <td>Output</td>
          <td>Final output of the formula.</td>
          <td>The data type must be the same as that of query.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,N,M,D]</td>
          <td>√</td>
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
      </tbody>
    </table>

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.

    <table style="undefined;table-layout: fixed; width: 1146px"><colgroup>
    <col style="width: 283px">
    <col style="width: 120px">
    <col style="width: 743px">
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
        <td>The required input, output, or attribute is passed as a null pointer.</td>
    </tr>
    <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td>query, key1, value1, key2, value2, attenMaskOptional, softmaxMaxOut, softmaxSumOut and attentionOutOut are not supported in terms of data type and format.</td>
    </tr>
     <tr>
      <td rowspan="1">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="1">561002</td>
      <td>An exception occurs during tiling. The query, key1, value1, key2, value2, and attenMaskOptional parameters do not meet the constraints.</td>
    </tr>
    </tbody>
    </table>

## aclnnFusedFloydAttention

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 168px">
    <col style="width: 128px">
    <col style="width: 854px">
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
        <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnFusedFloydAttentionGetWorkspaceSize.</td>
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

## Constraints<a name="1"></a>

- When this API is used together with PyTorch, ensure that the CANN package versions match the PyTorch package versions.
- Shape constraints:
  - **B**: The value ranges from 1 to 2K.
  - `H`: The value ranges from 1 to 256.
  - `N`: The value ranges from 16 to 1M and must be a multiple of 16.
  - `M`: The value ranges from 128 to 1M and must be a multiple of 128.
  - `K`: The value ranges from 128 to 1M and must be a multiple of 128.
  - `D`: The value can be 32, 64, or 128.

- The axis 0, 2, or 4 of query must be the same as that of key1.
- The shapes of key1 and value1 must be the same.
- The shapes of key2 and value2 must be the same.
- The shapes of softmaxMax and softmaxSum must be the same.
- `D`: Only `32`, `64`, and `128` are supported.
- Due to the restriction of underlying instructions, when M x D >= 65536 or K x D >= 65536, the performance deteriorates significantly. In this case, you are advised to use small operators to replace the implementation.

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).
  
```c++
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_fused_floyd_attention.h"

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
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  size = size > 1000? 1000 : size;  // The number of printed data records is less than or equal to 1000.
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // (Fixed writing) Initialize AscendCL.
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
  // 1. (Boilerplate) Initialize the device and stream. For details, see the list of external AscendCL APIs.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
  int64_t B = 1;
  int64_t H = 32;
  int64_t N = 128;
  int64_t M = 128;
  int64_t K = 128;
  int64_t D = 32;
  double scaleValue = 1.0;

  int64_t q_size = B * H * N * M * D;
  int64_t kv_size = B * H * N * K * D;
  int64_t k1v1_size = B * H * K * M * D;
  int64_t atten_mask_size = B * H * N * M * K;

  std::vector<int64_t> qShape = {B, H, N, M, D};
  std::vector<int64_t> kShape = {B, H, N, K, D};
  std::vector<int64_t> k1Shape = {B, H, K, M, D};
  std::vector<int64_t> vShape = {B, H, N, K, D};
  std::vector<int64_t> v1Shape = {B, H, K, M, D};
  std::vector<int64_t> attenmaskShape = {B, H, N, M, K};
  std::vector<int64_t> attentionOutShape = {B, H, N, M, D};
  std::vector<int64_t> softmaxMaxShape = {B, H, N, M, 8};
  std::vector<int64_t> softmaxSumShape = {B, H, N, M, 8};

  void *qDeviceAddr = nullptr;
  void *kDeviceAddr = nullptr;
  void *vDeviceAddr = nullptr;
  void *k1DeviceAddr = nullptr;
  void *v1DeviceAddr = nullptr;
  void *attenmaskDeviceAddr = nullptr;
  void *attentionOutDeviceAddr = nullptr;
  void *softmaxMaxDeviceAddr = nullptr;
  void *softmaxSumDeviceAddr = nullptr;

  aclTensor *q = nullptr;
  aclTensor *k = nullptr;
  aclTensor *v = nullptr;
  aclTensor *k1 = nullptr;
  aclTensor *v1 = nullptr;
  aclTensor *attenMask = nullptr;
  aclTensor *softmaxMax = nullptr;
  aclTensor *softmaxSum = nullptr;
  aclTensor *attentionOut = nullptr;

  std::vector<float> qHostData(q_size, 1.0);
  std::vector<float> kHostData(kv_size, 1.0);
  std::vector<float> vHostData(kv_size, 1.0);
  std::vector<float> k1HostData(k1v1_size, 1.0);
  std::vector<float> v1HostData(k1v1_size, 1.0);
  std::vector<uint8_t> attenmaskHostData(atten_mask_size, 0);
  std::vector<float> attentionOutHostData(B*H*N*M*D, 0.0);
  std::vector<float> softmaxMaxHostData(B*H*N*M*8, 0.0);
  std::vector<float> softmaxSumHostData(B*H*N*M*8, 0.0);

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT16, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(k1HostData, k1Shape, &k1DeviceAddr, aclDataType::ACL_FLOAT16, &k1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(v1HostData, v1Shape, &v1DeviceAddr, aclDataType::ACL_FLOAT16, &v1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attenmaskHostData, attenmaskShape, &attenmaskDeviceAddr, aclDataType::ACL_UINT8, &attenMask);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attentionOutHostData, attentionOutShape , &attentionOutDeviceAddr, aclDataType::ACL_FLOAT16, &attentionOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  
  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  
  // Call the first-phase API of aclnnFusedFloydAttention.
  ret = aclnnFusedFloydAttentionGetWorkspaceSize(
      q, k, v, k1, v1, attenMask, scaleValue, softmaxMax, softmaxSum, attentionOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedFloydAttentionGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  
  // Call the second-phase API of aclnnFusedFloydAttention.
  ret = aclnnFusedFloydAttention(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedFloydAttention failed. ERROR: %d\n", ret); return ret);
  
  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  
  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  PrintOutResult(attentionOutShape, &attentionOutDeviceAddr);
  PrintOutResult(softmaxMaxShape, &softmaxMaxDeviceAddr);
  PrintOutResult(softmaxSumShape, &softmaxSumDeviceAddr);
  
  // 6. Release aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(q);
  aclDestroyTensor(k);
  aclDestroyTensor(v);
  aclDestroyTensor(k1);
  aclDestroyTensor(v1);
  aclDestroyTensor(attenMask);
  aclDestroyTensor(attentionOut);
  aclDestroyTensor(softmaxMax);
  aclDestroyTensor(softmaxSum);
  
  // 7. Free device resources.
  aclrtFree(qDeviceAddr);
  aclrtFree(kDeviceAddr);
  aclrtFree(vDeviceAddr);
  aclrtFree(k1DeviceAddr);
  aclrtFree(v1DeviceAddr);
  aclrtFree(attenmaskDeviceAddr);
  aclrtFree(attentionOutDeviceAddr);
  aclrtFree(softmaxMaxDeviceAddr);
  aclrtFree(softmaxSumDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  
  return 0;
}
```
