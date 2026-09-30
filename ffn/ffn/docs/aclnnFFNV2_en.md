# aclnnFFNV2

## Supported Products

|Product     | Supported|
|:----------------------------|:-----------:|
|<term>Atlas A3 training products/Atlas A3 inference products</term>|      ×     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference accelerator cards</term>|      √     |
|<term>Atlas training products</term>|      ×     |

## Description

- **API function**: Computes MoeFFN and FFN. This operator is a feed-forward network (FFN) operator when there is no expert group (`expertTokens` being null) and an MoeFFN operator when there is an expert group. Both operators are variants of FFN and use the Mixture-of-Experts (MoE) architecture. MoE is a technology used to train models with trillions of parameters. MoE divides a prediction modeling task into several subtasks, trains an expert model on each subtask, and develops a gating model. The model assigns one or more experts based on the input data, and finally combines the computation results of multiple experts as the prediction result. In the MoE model, the input data is allocated to one or more most relevant experts, and the final result is determined based on the computation results of all involved experts. Compared with the [`FFN`](./aclnnFFN.md) API, **this API supports inputs of `expertTokens` indices, which are distinguished by `tokensIndexFlag`**.
- Formula:

  - **Non-quantization scenario:**

    $$
    y=activation(x * W1 + b1) * W2 + b2
    $$

  - **Quantization scenario:**

    $$
    y=((activation((x * W1 + b1) * deqScale1) * scale + offset) * W2 + b2) * deqScale2
    $$

  - **Pseudo-quantization scenario:**

    $$
    y=activation(x * ((W1 + antiquantOffset1) * antiquantScale1) + b1) * ((W2 + antiquantOffset2) * antiquantScale2) + b2
    $$

  **Note**:
  Whether FFN has performance benefits in the scenario without experts or in the scenario with a single expert depends on the actual test situation. When the vector time of the small operator corresponding to the FFN structure on the entire network is more than 30 μs and accounts for more than 10% of the FFN structure, try to use this fusion operator. If the actual test performance deteriorates, do not use this function.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnFFNV2GetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnFFNV2` is called to perform computation.

```Cpp
aclnnStatus aclnnFFNV2GetWorkspaceSize(
  const aclTensor*   x, 
  const aclTensor*   weight1, 
  const aclTensor*   weight2, 
  const aclIntArray* expertTokens, 
  const aclTensor*   bias1, 
  const aclTensor*   bias2, 
  const aclTensor*   scale, 
  const aclTensor*   offset, 
  const aclTensor*   deqScale1, 
  const aclTensor*   deqScale2, 
  const aclTensor*   antiquantScale1, 
  const aclTensor*   antiquantScale2, 
  const aclTensor*   antiquantOffset1, 
  const aclTensor*   antiquantOffset2, 
  const char*        activation, 
  int64_t            innerPrecise, 
  bool               tokensIndexFlag, 
  const aclTensor*   y, 
  uint64_t*          workspaceSize, 
  aclOpExecutor**    executor)
```

```Cpp
aclnnStatus aclnnFFNV2(
  void*          workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor* executor, 
  aclrtStream    stream)
```

## aclnnFFNV2GetWorkspaceSize

    The common variables involved in the parameter description are described as follows:
    M: number of tokens, corresponding to BS (B: batch size of input samples; S: sequence length of input samples) in Transformer.
    K1: input channel count of the first MatMul, corresponding to H (head-size, hidden layer size) in Transformer.
    N1: output channel count of the first MatMul.
    K2: input channel count of the second MatMul.
    N2: output channel count of the second MatMul, corresponding to H in Transformer.
    <term>Atlas A2 training products/Atlas A2 inference products</term>: E indicates the number of experts in the expert scenario. G indicates the number of antiquantOffset and antiquantScale groups in the pseudo-quantization per-group scenario.

- **Parameters**

  - `x` (aclTensor\*, compute input): required parameter, aclTensor on the device, x in the formula. The [data format](../../../docs/en/context/data_format.md) can be ND.
      - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be FLOAT16, BFLOAT16, or INT8. The supported input dimensions range from two dimensions [M, K1] to eight dimensions.
      - <term>Atlas inference accelerator card products</term>: The data type can be FLOAT16. The supported input dimensions are two dimensions [M, K1].
  - `weight1` (aclTensor\*, compute input): required parameter, aclTensor on the device, expert weight data, W1 in the formula. The [data format](../../../docs/en/context/data_format.md) can be ND.
      - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be FLOAT16, BFLOAT16, INT8, or INT4. The input is [E, K1, N1] with experts, or [K1, N1] without experts.
      - <term>Atlas inference accelerator card products</term>: The data type can be FLOAT16. The supported input dimensions are two dimensions [K1, N1].
  - `weight2` (aclTensor\*, compute input): required parameter, aclTensor on the device, expert weight data, W2 in the formula. The [data format](../../../docs/en/context/data_format.md) can be ND.
      - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be FLOAT16, BFLOAT16, INT8, or INT4. The input is [E, K2, N2] with experts, or [K2, N2] without experts.
      - <term>Atlas inference accelerator card products</term>: The data type can be FLOAT16. The supported input dimensions are two dimensions [K2, N2].
  - `expertTokens` (aclIntArray\*, compute input): optional parameter, aclIntArray on the host, representing the number of tokens for each expert. The data type can be INT64. The [data format](../../../docs/en/context/data_format.md) can be ND.
      - <term>Atlas A2 training products/Atlas A2 inference products</term>: The maximum length is 256 when the parameter is not null.
      - <term>Atlas inference accelerator cards</term>: Only null pointers can be passed.
  - `bias1` (aclTensor\*, compute input): optional parameter, aclTensor on the device, weight data correction value, b1 in the formula. The [data format](../../../docs/en/context/data_format.md) can be ND.
      - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be FLOAT16, FLOAT32, or INT32. The input is [E, N1] with experts, or [N1] without experts.
      - <term>Atlas inference accelerator card products</term>: The data type can be FLOAT16. The supported input dimension is one dimension [N1].
  - `bias2` (aclTensor\*, compute input): optional parameter, aclTensor on the device, weight data correction value, b2 in the formula. The [data format](../../../docs/en/context/data_format.md) can be ND.
      - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be FLOAT16, FLOAT32, or INT32. The input is [E, N2] with experts, or [N2] without experts.
      - <term>Atlas inference accelerator card products</term>: The data type can be FLOAT16. The supported input dimension is one dimension [N2].
  - `scale` (aclTensor\*, compute input): optional parameter, aclTensor on the device, quantization parameter, quantization scaling factor. The [data format](../../../docs/en/context/data_format.md) can be ND.
      - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be FLOAT32. In per-tensor mode, the input is [E] or [1], with or without experts, respectively. In per-channel mode, the input is [E, N1] or [N1], with or without experts, respectively.
      - <term>Atlas inference accelerator cards</term>: Only null pointers can be passed.
  - `offset` (aclTensor\*, compute input): optional parameter, aclTensor on the device, quantization parameter, quantization offset. The [data format](../../../docs/en/context/data_format.md) can be ND.
      - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be a one-dimensional FLOAT32 vector. The number of input elements is [E] with experts or [1] without experts.
      - <term>Atlas inference accelerator cards</term>: Only null pointers can be passed.
  - `deqScale1` (aclTensor\*, compute input): optional parameter, aclTensor on the device, quantization parameter, dequantization scaling factor for the first MatMul. The [data format](../../../docs/en/context/data_format.md) can be ND.
      - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be UINT64, INT64, FLOAT32, or BFLOAT16. The input is [E, N1] with experts, or [N1] without experts.
      - <term>Atlas inference accelerator cards</term>: Only null pointers can be passed.
  - `deqScale2` (aclTensor\*, compute input): optional parameter, aclTensor on the device, quantization parameter, dequantization scaling factor for the second MatMul. The [data format](../../../docs/en/context/data_format.md) can be ND.
      - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be UINT64, INT64, FLOAT32, or BFLOAT16. The input is [E, N2] with experts, or [N2] without experts.
      - <term>Atlas inference accelerator cards</term>: Only null pointers can be passed.
  - `antiquantScale1` (aclTensor\*, compute input): optional parameter, aclTensor on the device, pseudo-quantization parameter, scaling factor for the first MatMul. The [data format](../../../docs/en/context/data_format.md) can be ND.
      - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be FLOAT16 or BFLOAT16. In per-channel mode, the input is [E, N1] or [N1], with or without experts, respectively. In per-group mode, the input is [E, G, N1] or [G, N1], with or without experts, respectively.
      - <term>Atlas inference accelerator cards</term>: Only null pointers can be passed.
  - `antiquantScale2` (aclTensor\*, compute input): optional parameter, aclTensor on the device, pseudo-quantization parameter, scaling factor for the second MatMul. The [data format](../../../docs/en/context/data_format.md) can be ND.
      - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be FLOAT16 or BFLOAT16. In per-channel mode, the input is [E, N2] or [N2], with or without experts, respectively. In per-group mode, the input is [E, G, N2] or [G, N2], with or without experts, respectively.
      - <term>Atlas inference accelerator cards</term>: Only null pointers can be passed.
  - `antiquantOffset1` (aclTensor\*, compute input): optional parameter, aclTensor on the device, pseudo-quantization parameter, offset for the first MatMul. The [data format](../../../docs/en/context/data_format.md) can be ND.
      - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be FLOAT16 or BFLOAT16. In per-channel mode, the input is [E, N1] or [N1], with or without experts, respectively. In per-group mode, the input is [E, G, N1] or [G, N1], with or without experts, respectively.
      - <term>Atlas inference accelerator cards</term>: Only null pointers can be passed.
  - `antiquantOffset2` (aclTensor\*, compute input): optional parameter, aclTensor on the device, pseudo-quantization parameter, offset for the second MatMul. The [data format](../../../docs/en/context/data_format.md) can be ND.
      - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be FLOAT16 or BFLOAT16. In per-channel mode, the input is [E, N2] or [N2], with or without experts, respectively. In per-group mode, the input is [E, G, N2] or [G, N2], with or without experts, respectively.
      - <term>Atlas inference accelerator cards</term>: Only null pointers can be passed.

  - `activation` (char*, compute input): required parameter, attribute value on the host, representing the activation function used, activation in the formula.
      - <term>Atlas A2 training products/Atlas A2 inference products</term>: fastgelu, gelu, relu, silu,geglu, swiglu, and reglu are supported.
      - <term>Atlas inference accelerator cards</term>: fastgelu, gelu, relu, and silu are supported.
  - `innerPrecise` (int64_t, compute input): optional parameter, int on the host, indicating the high-precision or high-performance mode. The data type is INT64.
      - If `innerPrecise` is set to 0, the high-precision mode is enabled. In non-quantization scenarios where all mandatory parameters are FLOAT16, the input and output of the activation layer within the operator are calculated using the FLOAT32 data type.
      - If `innerPrecise` is set to 1, the high-performance mode is used.
      - <term>Atlas A2 training products/Atlas A2 inference products</term>: This parameter only takes effect when all mandatory parameters are FLOAT16 in non-quantization scenarios. High-precision and high-performance modes are not distinguished in other scenarios.
      - <term>Atlas inference accelerator cards</term>: The value can only be 1.
  - `tokensIndexFlag` (bool, compute input): optional parameter, bool on the host, indicating whether expertTokens is an index value. The data type can be bool.

    - If `tokensIndexFlag` is set to `true`, `expertTokens` is an index value.
    - If `tokensIndexFlag` is set to `false`, `expertTokens` is the number of tokens of each expert.
  - `y` (aclTensor\*, compute output): aclTensor on the device, output y in the formula. The [data format](../../../docs/en/context/data_format.md) can be ND. The output shape is the same as that of `x`.
      - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be FLOAT16 or BFLOAT16.
      - <term>Atlas inference accelerator cards</term>: The data type can be FLOAT16.
  - workspaceSize (uint64\_t\*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor\*\*, output): operator executor, containing the operator computation process.

- **Returns**

  aclnnStatus status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API call completes input parameter verification. The possible error codes and causes are as follows:

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 286px">
  <col style="width: 118px">
  <col style="width: 746px">
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
      <td>The input parameter is a required input, output, or attribute, and is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data type or data format of x, weight1, weight2, activation, expertTokens, bias1, bias2, or y is not supported.</td>
    </tr>
  </tbody>
  </table>

## aclnnFFNV2

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1049px"><colgroup>
  <col style="width: 167px">
  <col style="width: 118px">
  <col style="width: 764px">
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
      <td>Address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnFFNV2GetWorkspaceSize.</td>
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

  aclnnStatus status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computing:
  - `aclnnFFNV2` defaults to a non-deterministic implementation. You can call `aclrtCtxSetSysParamOpt` to enable deterministic computing.
- In all scenarios, K1=N2, K1<65536, K2<65536. The M axis must be less than the maximum value of INT32 after 32-byte alignment.
- <term>Atlas A2 training products/Atlas A2 inference products</term>:
  - If there are experts, the total number of experts must be the same as M of `x`.
  - When the activation layer is geglu, swiglu, or reglu, only the FLOAT16 high-performance scenario without expert groups is supported, and N1 = 2 × K2. (The FLOAT16 scenario refers to a scenario where the data type of all required aclTensor parameters is FLOAT16.)
  - When the activation layer is gelu, fastgelu, relu, or silu, FLOAT16 high-precision and high-performance scenarios, BFLOAT16 scenarios, quantization scenarios, and fake-quantization scenarios with experts or without expert groups are supported, and N1=K2.
  - In non-quantization scenarios, do not input quantization or pseudo-quantization parameters. In quantization scenarios, do not input pseudo-quantization parameters. In pseudo-quantization scenarios, do not input quantization parameters.
  - Parameter types in quantization scenarios: The data type of `x` is INT8, that of `weight` is INT8, that of `bias` is INT32, that of `scale` is FLOAT32, and that of `offset` is FLOAT32. Other parameter types are divided into two cases according to the type of `y`.
    - When the data type of `y` is FLOAT16, `deqScale` supports the following data types: UINT64, INT64, and FLOAT32.
    - When the data type of `y` is BFLOAT16, `deqScale` supports only the BFLOAT16 data type.
    - The data type of `deqScale1` must be the same as that of `deqScale2`.
  - Parameter types in quantization scenarios where the per-channel mode of `scale` is supported: The data type of `x` is INT8, that of `weight` is INT8, that of `bias` is INT32, that of `scale` is FLOAT32, and that of `offset` is FLOAT32. Other parameter types are categorized into two cases according to the type of `y`.
    - When the data type of `y` is FLOAT16, `deqScale` supports the following data types: UINT64 and INT64.
    - When the data type of `y` is BFLOAT16, `deqScale` supports only the BFLOAT16 data type.
    - The data type of `deqScale1` must be the same as that of `deqScale2`.
  - The pseudo-quantization scenario supports two parameter types:
    - The data type of `y` is FLOAT16, that of `x` is FLOAT16, that of `bias` is FLOAT16, that of `antiquantScale` is FLOAT16, and that of `antiquantOffset` is FLOAT16. `weight` supports data types INT8 and INT4.
    - The data type of `y` is BFLOAT16, that of `x` is BFLOAT16, that of `bias` is FLOAT32, that of `antiquantScale` is BFLOAT16, and that of `antiquantOffset` is BFLOAT16. `weight` supports data types INT8 and INT4.
  - When the data type of `weight1` or `weight2` is INT4, the last dimension of the shape must be an even number.
  - In pseudo-quantization scenarios, under per-group mode, G (number of groups) in `antiquantScale1` and `antiquantOffset1` must be exactly divided by K1, and G (number of groups) in `antiquantScale2` and `antiquantOffset2` must be exactly divided by K2.
  - In the BFLOAT16 non-quantization scenario, `innerPrecise` can only be set to `0`. In the FLOAT16 non-quantization scenario, `innerPrecise` can be set to `0` or `1`. In quantization or fake-quantization scenarios, `innerPrecise` can be set to `0` or `1`, but the setting does not take effect.
  - If `tokensIndexFlag` is set to `true` and there are experts (`expertTokens` not being null), the value of `expertTokens` must meet the following requirements: If both `i` and `j` are valid array indexes in `expertTokens`, and `j` is greater than `i`, then the value of a *j*th element in `expertTokens` is greater than or equal to the value of an *i*th element in `expertTokens`.

- <term>Atlas inference accelerator cards</term>:
  - Only the non-expert scenario is supported.
  - N1 must be equal to K2.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_ffn_v2.h"
#include "aclnn/opdev/fp16_t.h"

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

int Init(int32_t deviceId, aclrtStream* stream) {
  // (Fixed writing) Initialize resources.
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
  auto size = GetShapeSize(shape) * aclDataTypeSize(dataType);
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
  // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as required.
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  std::vector<int64_t> weight1Shape = {2, 2};
  std::vector<int64_t> weight2Shape = {2, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* weight1DeviceAddr = nullptr;
  void* weight2DeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclTensor* weight1 = nullptr;
  aclTensor* weight2 = nullptr;
  std::vector<op::fp16_t> selfHostData = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  std::vector<op::fp16_t> outHostData = {0, 0, 0, 0};
  std::vector<op::fp16_t> weight1HostData = {0.1, 0.2, 0.3, 0.4};
  std::vector<op::fp16_t> weight2HostData = {0.4, 0.3, 0.2, 0.1};
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT16, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a weight1 aclTensor.
  ret = CreateAclTensor(weight1HostData, weight1Shape, &weight1DeviceAddr, aclDataType::ACL_FLOAT16, &weight1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a weight2 aclTensor.
  ret = CreateAclTensor(weight2HostData, weight2Shape, &weight2DeviceAddr, aclDataType::ACL_FLOAT16, &weight2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnFFNV2 API call example
  LOG_PRINT("test aclnnFFNV2\n");

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  // Call the first-phase API of aclnnFFNV2.
  ret = aclnnFFNV2GetWorkspaceSize(self, weight1, weight2, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,    "relu", 1, false, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFFNV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the computed workspaceSize.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnFFNV2.
  ret = aclnnFFNV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFFNV2 failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<op::fp16_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    std::cout << "index: " << i << ": " << static_cast<float>(resultData[i]) << std::endl;
  }

  // 6. Release aclTensor. Modify the configuration based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclDestroyTensor(weight1);
  aclDestroyTensor(weight2);

  // 7. Release device resources.
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(weight1DeviceAddr);
  aclrtFree(weight2DeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
