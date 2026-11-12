# aclnnFusedFloydAttentionGrad

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T09:30:23.011Z -->

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

- Description: Computes the backward output of Floyd attention for training. Compared with traditional FA, FloydAttn additionally uses seq as the batch axis when computing qk/pv attention, thereby converting the computation to batchMatmul.
- Formula:

    The forward attention formula is:

    $$
    S=Mask(scale*(Q*K_1^T + Q*K_2^T), atten\_mask) \\
    P=Softmax(S) \\
    Y=(P*V_1+P*V_2)
    $$

    The backward propagation formula for attention is:

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

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnFusedFloydAttentionGradGetWorkspaceSize" API first to obtain the workspace size required for the computation and the executor that includes computation of the operator, and then call the "aclnnFusedFloydAttentionGrad" API to perform the computation.

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
      <col style="width: 101px">
      <col style="width: 190px">
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
          <td>query (aclTensor)</td>
          <td>Input</td>
          <td>Q in the formula.</td>
          <td>The data type must be the same as the data type of key1/value1/key2/value2.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,N,M,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>key1 (aclTensor)</td>
          <td>Input</td>
          <td>K1 in the formula.</td>
          <td>The data type must be the same as the data type of query/value1/key2/value2.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,N,K,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>value1 (aclTensor)</td>
          <td>Input</td>
          <td>V1 in the formula.</td>
          <td>The data type must be the same as the data type of query/key1/key2/value2.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,N,K,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>key2(aclTensor)</td>
          <td>Input</td>
          <td>K2 in the formula.</td>
          <td>The data type must be the same as the data type of query/key1/value1/value2.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,K,M,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>value2(aclTensor)</td>
          <td>Input</td>
          <td>V2 in the formula.</td>
          <td>Data type must be the same as the data type of query/key1/value1/key2.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,K,M,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>dy(aclTensor)</td>
          <td>Input</td>
          <td>Input dY in the formula.</td>
          <td>-</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,N,M,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>attenMaskOptional (aclTensor)</td>
          <td>Input</td>
          <td>atten_mask in the formula.</td>
          <td>A value of 1 indicates that the bit does not participate in the computation, and 0 indicates that it does participate in the computation.</td>
          <td>BOOL, UINT8</td>
          <td>ND</td>
          <td>[B,1,N,1,K]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>softmaxMax (aclTensor)</td>
          <td>Input</td>
          <td>Intermediate output of the attention forward direction computation.</td>
          <td>The shape of the output is [B,H,N,M,8].</td>
          <td>FLOAT</td>
          <td>ND</td>
          <td>[B,H,N,M,8]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>softmaxSum (aclTensor)</td>
          <td>Input</td>
          <td>Intermediate output of the attention forward direction computation.</td>
          <td>The output shape is [B,H,N,M,8].</td>
          <td>FLOAT</td>
          <td>ND</td>
          <td>[B,H,N,M,8]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>attentionIn (aclTensor)</td>
          <td>Input</td>
          <td>Final output of the attention forward direction computation.</td>
          <td>Data type and shape type must be the same as those of query.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,N,M,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>dqOut (aclTensor)</td>
          <td>Output</td>
          <td>Indicates the gradient of query, corresponding to dQ in the formula.</td>
          <td>-</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,N,M,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>dk1Out (aclTensor)</td>
          <td>Output</td>
          <td>Indicates the gradient of key1, corresponding to dK1 in the formula.</td>
          <td>-</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,N,K,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>dv1Out (aclTensor)</td>
          <td>Output</td>
          <td>dV1 in the formula, which indicates the gradient of value1.</td>
          <td>-</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,N,K,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>dk2Out (aclTensor)</td>
          <td>Output</td>
          <td>Indicates dK2 in the formula, the gradient of key2.</td>
          <td>-</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,K,M,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>dv2Out (aclTensor)</td>
          <td>Output</td>
          <td>dV2 in the formula, indicates the gradient of value2.</td>
          <td>-</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,K,M,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>scaleValue (double)</td>
          <td>Input</td>
          <td>scale in the formula indicates the scaling factor.</td>
          <td>-</td>
          <td>DOUBLE</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>workspaceSize (uint64_t*)</td>
          <td>Output</td>
          <td>Returns the workspace size required by the Device side application.</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>executor (aclOpExecutor)</td>
          <td>Output</td>
          <td>Returns the op executor, which includes computation flow of the operator.</td>
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
      <td>The input parameter is a required input, output, or required attribute, and is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="1">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="1">161002</td>
      <td>The data type or data format of query, key1, value1, key2, value2, dy, attenMaskOptional, softmaxMax, softmaxSum, attentionIn, dqOut, dk1Out, dv1Out, dk2Out, or dv2Out is out of the supported range.</td>
    </tr>
    <tr>
      <td rowspan="1">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="1">561002</td>
      <td>An exception occurs during tiling. query, key1, value1, key2, value2, dy, attenMaskOptional, softmaxMax, softmaxSum, or attentionIn does not comply with the constraints.</td>
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
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Workspace memory address applied on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size applied on the Device side, obtained by the first-phase API aclnnFusedFloydAttentionGradGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, includes computation flow of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- This API does not support deterministic computation.
- This API does not support non-contiguous output.
- When using this API with PyTorch, ensure that the versions of CANN-related packages and PyTorch-related packages are compatible.
- Constraints on data shape:
  - B: value range is 1 to 2K.
  - H: value range is 1 to 256.
  - N: value range is 16 to 1M, and N%16==0.
  - M: value range is 128 to 1M, and M%128==0.
  - K: value range is 128 to 1M, and K%128==0.
  - D: value range is 32/64/128.

- Axis 0/2/4 of query and key1 must be the same.
- key1 and value1 must have the same shape.
- key2 and value2 must have the same shape.
- The shapes of query and dy/attentionIn must be the same.
- softmaxMax and softmaxSum must have the same shape.
- D only supports 32, 64, or 128.
- Due to underlying instruction limitations, when M*D >= 65536 or K*D >= 65536, significant performance degradation occurs. In this case, use small operator concatenation as a replacement.

## Example

Example code is as follows, for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

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
  // Boilerplate: initialize AscendCL.
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
  // Call aclrtMalloc to allocate device-side memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host-side data to the device-side memory.
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
  // 1. (Boilerplate) Initialize the device/stream. For details, see the AscendCL external API list.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs. Custom construction is required based on the API.
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

  // 3. Call the CANN operator library API. Replace it with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnFusedFloydAttentionGrad.
  ret = aclnnFusedFloydAttentionGradGetWorkspaceSize(q, k1, v1, k2, v2, dx, attenmask, softmaxMax, softmaxSum,
        attentionIn, scaleValue, dq, dk1, dv1, dk2, dv2, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedFloydAttentionGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnFusedFloydAttentionGrad.
  ret = aclnnFusedFloydAttentionGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedFloydAttentionGrad failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait for task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output values, and copy the results from device memory to the host side. Modify the code as required by the specific API definition.
  PrintOutResult(dqShape, &dqDeviceAddr);
  // PrintOutResult(dkShape, &dkDeviceAddr);
  // PrintOutResult(dvShape, &dvDeviceAddr);

  // 6. Destroy the aclTensor and aclScalar. Modify the code as required by the specific API definition.
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

  // 7. Release device resources.
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
