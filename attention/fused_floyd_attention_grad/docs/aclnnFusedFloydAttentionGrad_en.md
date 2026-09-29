# aclnnFusedFloydAttentionGrad

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

- Description: In training scenarios, FloydAttn differs from traditional FlashAttention by treating the sequence dimension (`seq`) as an additional batch axis during QK/PV attention computation, thereby converting the attention computation into batch matrix multiplication (`batchMatmul`).
- **Formula**:

    The forward propagation formula for attention is as follows:

    $$
    S=Mask(scale*(Q*K_1^T + Q*K_2^T), atten\_mask) \\
    P=Softmax(S) \\
    Y=(P*V_1+P*V_2)
    $$

    Then the backward propagation formula for attention is as follows:

    $$
    dV_1=P^TdY
    $$

    $$
    dV_2=P^TdY
    $$

    $$
    dQ=\frac{((dS)*K_1)}{\sqrt{d}}+\frac{((dS)*K_2)}{\sqrt{d}}
    $$

    $$
    dK_1=\frac{((dS)^T*Q)}{\sqrt{d}}
    $$

    $$
    dK_2=\frac{((dS)^T*Q)}{\sqrt{d}}
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnFusedFloydAttentionGradGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnFusedFloydAttentionGrad` is called to perform computation.

```c++
aclnnStatus aclnnFusedFloydAttentionGradGetWorkspaceSize(
  const aclTensor   *query, 
  const aclTensor   *key1, 
  const aclTensor   *value1, 
  const aclTensor   *key2, 
  const aclTensor   *value2, 
  const aclTensor   *dy, 
  const aclTensor   *attenMaskOptional, 
  const aclTensor   *softmaxMax, 
  const aclTensor   *softmaxSum, 
  const aclTensor   *attentionIn, 
  double             scaleValue, 
  const aclTensor   *dqOut, 
  const aclTensor   *dk1Out, 
  const aclTensor   *dv1Out, 
  const aclTensor   *dk2Out, 
  const aclTensor   *dv2Out, 
  uint64_t          *workspaceSize, 
  aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnFusedFloydAttentionGrad(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```

## aclnnFusedFloydAttentionGradGetWorkspaceSize

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
          <td>query (aclTensor)</td>
          <td>Input</td>
          <td>Q in the formula.</td>
          <td>The data type is the same as that of key1/value1/key2/value2.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,N,M,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>key1 (aclTensor)</td>
          <td>Input</td>
          <td>K1 in the formula.</td>
          <td>The data type is the same as that of query/value1/key2/value2.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,N,K,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>value1 (aclTensor)</td>
          <td>Input</td>
          <td>V1 in the formula.</td>
          <td>The data type is the same as that of query/key1/key2/value2.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,N,K,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>key2 (aclTensor)</td>
          <td>Input</td>
          <td>K2 in the formula.</td>
          <td>The data type is the same as that of query/key1/value1/value2.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,K,M,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>value2 (aclTensor)</td>
          <td>Input</td>
          <td>V2 in the formula.</td>
          <td>The data type is the same as that of query/key1/value1/key2.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,K,M,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>dy (aclTensor)</td>
          <td>Input</td>
          <td>dY in the formula.</td>
          <td>-</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,N,M,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>attenMaskOptional (aclTensor)</td>
          <td>Input</td>
          <td><code>atten_mask</code> in the formula.</td>
          <td>A value of 1 indicates that the position does not participate in the calculation, while a value of 0 indicates that it does.</td>
          <td>BOOL or UINT8</td>
          <td>ND</td>
          <td>[B,1,N,1,K]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>softmaxMax (aclTensor)</td>
          <td>Input</td>
          <td>Intermediate output of the forward attention calculation.</td>
          <td>The output shape type is [B,H,N,M,8].</td>
          <td>FLOAT</td>
          <td>ND</td>
          <td>[B,H,N,M,8]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>softmaxSum (aclTensor)</td>
          <td>Input</td>
          <td>Intermediate output of the forward attention calculation.</td>
          <td>The output shape type is [B,H,N,M,8].</td>
          <td>FLOAT</td>
          <td>ND</td>
          <td>[B,H,N,M,8]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>attentionIn (aclTensor)</td>
          <td>Input</td>
          <td>Final output of the forward attention calculation.</td>
          <td>The data type and shape must be the same as those of query.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,N,M,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>dqOut (aclTensor)</td>
          <td>Output</td>
          <td>dQ in the formula, indicating the gradient of query.</td>
          <td>-</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,N,M,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>dk1Out (aclTensor)</td>
          <td>Output</td>
          <td>dK1 in the formula indicates the gradient of key 1.</td>
          <td>-</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,N,K,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>dv1Out (aclTensor)</td>
          <td>Output</td>
          <td>dV1 in the formula indicates the gradient of value 1.</td>
          <td>-</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,N,K,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>dk2Out (aclTensor)</td>
          <td>Output</td>
          <td>dK2 in the formula indicates the gradient of key 2.</td>
          <td>-</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,K,M,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>dv2Out (aclTensor)</td>
          <td>Output</td>
          <td>dV2 in the formula indicates the gradient of value2.</td>
          <td>-</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,K,M,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>scaleValue (double)</td>
          <td>Input</td>
          <td><code>scale</code> in the formula, indicating the scaling coefficient.</td>
          <td>-</td>
          <td>DOUBLE</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>workspaceSize (uint64_t*) </td>
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
      </tbody>
    </table>
- **Returns**
  
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.
  <table style="undefined;table-layout: fixed;width: 1151px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 833px">
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
      <td>The required input, output, or attribute is passed as a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="1">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="1">161002</td>
      <td>query, key1, value1, key2, value2, dy, attenMaskOptional, softmaxMax, softmaxSum, attentionIn, dqOut, dk1Out, dv1Out, dk2Out or dv2Out data type or format is not supported.</td>
    </tr>
    <tr>
      <td rowspan="1">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="1">561002</td>
      <td>An exception occurs during tiling. The query, key1, value1, key2, value2, dy, attenMaskOptional, softmaxMax, softmaxSum, and attentionIn parameters do not meet the constraints.</td>
    </tr>
  </tbody>
  </table>

## aclnnFusedFloydAttentionGrad

- **Parameters:**
  
  <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 833px">
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
      <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnFusedFloydAttentionGradGetWorkspaceSize.</td>
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

- This API does not support determinism.
- When this API is used together with PyTorch, ensure that the CANN package versions match the PyTorch package versions.
- Shape constraints:
  - **B**: The value ranges from 1 to 2K.
  - `H`: The value ranges from 1 to 256.
  - `N`: The value ranges from 16 to 1M and must be a multiple of 16.
  - `M`: The value ranges from 128 to 1M and must be a multiple of 128.
  - `K`: The value ranges from 128 to 1M and must be a multiple of 128.
  - D: The value can be 32, 64, or 128.

- The 0th, 2nd, and 4th axes of query and key1 must be the same.
- The shapes of key1 and value1 must be the same.
- The shapes of key2 and value2 must be the same.
- The shapes of query and dy/attentionIn must be the same.
- The shapes of softmaxMax and softmaxSum must be the same.
- `D`: Only `32`, `64`, and `128` are supported.
- Due to the restriction of underlying instructions, when M x D >= 65536 or K x D >= 65536, the performance deteriorates significantly. In this case, you are advised to use small operators to implement the function.

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_fused_floyd_attention_grad.h"

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
  int64_t N = 1;
  int64_t S1 = 256;
  int64_t S2 = 256;
  int64_t S3 = 256;
  int64_t D = 128;

  int64_t q_size = B * N * S1 * S2 * D;
  int64_t k1_v1_size = B * N * S1 * S3 * D;
  int64_t k2_v2_size = B * N * S3 * S2 * D;
  int64_t atten_mask_size = B * S1 * S3;
  int64_t softmax_size = B * N * S1 * S2 * 8;

  std::vector<int64_t> qShape = {B, N, S1, S2, D};
  std::vector<int64_t> k1v1Shape = {B, N, S1, S3, D};
  std::vector<int64_t> k2v2Shape = {B, N, S3, S2, D};
  std::vector<int64_t> attenmaskShape = {B, 1, S1, 1, S3};
  std::vector<int64_t> softmaxMaxShape = {B, N, S1, S2, 8};
  std::vector<int64_t> softmaxSumShape = {B, N, S1, S2, 8};
  std::vector<int64_t> attentionInShape = {B, N, S1, S2, D};

  std::vector<int64_t> dqShape = {B, N, S1, S2, D};
  std::vector<int64_t> dk1dv1Shape = {B, N, S1, S3, D};
  std::vector<int64_t> dk2dv2Shape = {B, N, S3, S2, D};

  void* qDeviceAddr = nullptr;
  void* k1DeviceAddr = nullptr;
  void* v1DeviceAddr = nullptr;
  void* k2DeviceAddr = nullptr;
  void* v2DeviceAddr = nullptr;
  void* dxDeviceAddr = nullptr;
  void* attenmaskDeviceAddr = nullptr;
  void* softmaxMaxDeviceAddr = nullptr;
  void* softmaxSumDeviceAddr = nullptr;
  void* attentionInDeviceAddr = nullptr;
  void* dqDeviceAddr = nullptr;
  void* dk1DeviceAddr = nullptr;
  void* dv1DeviceAddr = nullptr;
  void* dk2DeviceAddr = nullptr;
  void* dv2DeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* k1 = nullptr;
  aclTensor* v1 = nullptr;
  aclTensor* k2 = nullptr;
  aclTensor* v2 = nullptr;
  aclTensor* dx = nullptr;
  aclTensor* attenmask = nullptr;
  aclTensor* softmaxMax = nullptr;
  aclTensor* softmaxSum = nullptr;
  aclTensor* attentionIn = nullptr;
  aclTensor* dq = nullptr;
  aclTensor* dk1 = nullptr;
  aclTensor* dv1 = nullptr;
  aclTensor* dk2 = nullptr;
  aclTensor* dv2 = nullptr;

  std::vector<float> qHostData(q_size, 1.0);
  std::vector<float> k1HostData(k1_v1_size, 1.0);
  std::vector<float> v1HostData(k1_v1_size, 1.0);
  std::vector<float> k2HostData(k2_v2_size, 1.0);
  std::vector<float> v2HostData(k2_v2_size, 1.0);
  std::vector<float> dxHostData(q_size, 1.0);
  std::vector<uint8_t> attenmaskHostData(atten_mask_size, 0);
  std::vector<float> softmaxMaxHostData(softmax_size, 3.0);
  std::vector<float> softmaxSumHostData(softmax_size, 3.0);
  std::vector<float> attentionInHostData(q_size, 1.0);
  std::vector<float> dqHostData(q_size, 0);
  std::vector<float> dk1HostData(k1_v1_size, 0);
  std::vector<float> dv1HostData(k1_v1_size, 0);
  std::vector<float> dk2HostData(k2_v2_size, 0);
  std::vector<float> dv2HostData(k2_v2_size, 0);

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(k1HostData, k1v1Shape, &k1DeviceAddr, aclDataType::ACL_FLOAT16, &k1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(v1HostData, k1v1Shape, &v1DeviceAddr, aclDataType::ACL_FLOAT16, &v1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(k2HostData, k2v2Shape, &k2DeviceAddr, aclDataType::ACL_FLOAT16, &k2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(v2HostData, k2v2Shape, &v2DeviceAddr, aclDataType::ACL_FLOAT16, &v2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dxHostData, qShape, &dxDeviceAddr, aclDataType::ACL_FLOAT16, &dx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attenmaskHostData, attenmaskShape, &attenmaskDeviceAddr, aclDataType::ACL_UINT8, &attenmask);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attentionInHostData, attentionInShape, &attentionInDeviceAddr, aclDataType::ACL_FLOAT16, &attentionIn);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dqHostData, dqShape, &dqDeviceAddr, aclDataType::ACL_FLOAT16, &dq);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dk1HostData, dk1dv1Shape, &dk1DeviceAddr, aclDataType::ACL_FLOAT16, &dk1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dv1HostData, dk1dv1Shape, &dv1DeviceAddr, aclDataType::ACL_FLOAT16, &dv1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dk2HostData, dk2dv2Shape, &dk2DeviceAddr, aclDataType::ACL_FLOAT16, &dk2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dv2HostData, dk2dv2Shape, &dv2DeviceAddr, aclDataType::ACL_FLOAT16, &dv2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  double scaleValue = 1.0/sqrt(128);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnFusedFloydAttentionGrad.
  ret = aclnnFusedFloydAttentionGradGetWorkspaceSize(q, k1, v1, k2, v2, dx, attenmask, softmaxMax, softmaxSum, 
        attentionIn, scaleValue, dq, dk1, dv1, dk2, dv2, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedFloydAttentionGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnFusedFloydAttentionGrad.
  ret = aclnnFusedFloydAttentionGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedFloydAttentionGrad failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  PrintOutResult(dqShape, &dqDeviceAddr);
  // PrintOutResult(dkShape, &dkDeviceAddr);
  // PrintOutResult(dvShape, &dvDeviceAddr);

  // 6. Release aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(q);
  aclDestroyTensor(k1);
  aclDestroyTensor(v1);
  aclDestroyTensor(k2);
  aclDestroyTensor(v2);
  aclDestroyTensor(dx);
  aclDestroyTensor(attenmask);
  aclDestroyTensor(softmaxMax);
  aclDestroyTensor(softmaxSum);
  aclDestroyTensor(attentionIn);
  aclDestroyTensor(dq);
  aclDestroyTensor(dk1);
  aclDestroyTensor(dv1);
  aclDestroyTensor(dk2);
  aclDestroyTensor(dv2);

  // 7. Free device resources.
  aclrtFree(qDeviceAddr);
  aclrtFree(k1DeviceAddr);
  aclrtFree(v1DeviceAddr);
  aclrtFree(k2DeviceAddr);
  aclrtFree(v2DeviceAddr);
  aclrtFree(dxDeviceAddr);
  aclrtFree(attenmaskDeviceAddr);
  aclrtFree(softmaxMaxDeviceAddr);
  aclrtFree(softmaxSumDeviceAddr);
  aclrtFree(attentionInDeviceAddr);
  aclrtFree(dqDeviceAddr);
  aclrtFree(dk1DeviceAddr);
  aclrtFree(dv1DeviceAddr);
  aclrtFree(dk2DeviceAddr);
  aclrtFree(dv2DeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
