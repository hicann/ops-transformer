# aclnnFFN

[📄 View Source Code](https://gitcode.com/cann/ops-transformer/tree/master/ffn/ffn)

## Supported Products

|Product     | Supported|
|:----------------------------|:-----------:|
|Ascend 950PR/Ascend 950DT|      ×     |
|<term>Atlas A3 training products/Atlas A3 inference products</term>|      ×     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Description

- API function: Computes MoeFFN and FFN. This operator is a feed-forward network (FFN) operator when there is no expert group (`expertTokens` being null) and an MoeFFN operator when there is an expert group. Both operators are variants of FFN and use the Mixture-of-Experts (MoE) architecture. MoE is a technology used to train models with trillions of parameters. MoE divides a prediction modeling task into several subtasks, trains an expert model on each subtask, and develops a gating model. The model assigns one or more experts based on the input data, and finally combines the computation results of multiple experts as the prediction result. In the MoE model, the input data is allocated to one or more most relevant experts, and the final result is determined based on the computation results of all involved experts.
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

**Note:**

Whether FFN has performance benefits in the scenario without experts or in the scenario with a single expert depends on the actual test situation. When the vector time of the small operator corresponding to the FFN structure on the entire network is more than 30 μs and accounts for more than 10% of the FFN structure, try to use this fusion operator. If the actual test performance deteriorates, do not use this function.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnFFNGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnFFN` is called to perform computation.

```Cpp
aclnnStatus aclnnFFNGetWorkspaceSize(
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
  const aclTensor*   y, 
  uint64_t*          workspaceSize, 
  aclOpExecutor**    executor)
```

```Cpp
aclnnStatus aclnnFFN(
  void*          workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor* executor, 
  aclrtStream    stream)
```

## aclnnFFNGetWorkspaceSize

**Note:** 
M indicates the number of tokens, which corresponds to the batch size (BS) in Transformer. B indicates the batch size, and S indicates the sequence length. 
K1 indicates the number of input channels of the first matmul, which corresponds to the head size (H) in Transformer. 
N1 indicates the number of output channels of the first matmul, and K2 indicates the number of input channels of the second matmul. 
N2 indicates the number of output channels of the second matmul, which corresponds to the head size (H) in Transformer. 
E indicates the number of experts in the expert scenario. 
G indicates the number of groups of antiquantOffset and antiquantScale in the fake-quantization per-group scenario.

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1650px">
  <colgroup>
  <col style="width: 200px"> <!-- Parameter name -->
  <col style="width: 120px"> <!-- Input/Output -->
  <col style="width: 280px"> <!-- Description -->
  <col style="width: 300px"> <!-- Usage Description -->
  <col style="width: 250px"> <!-- Data Type -->
  <col style="width: 120px"> <!-- Data Format -->
  <col style="width: 240px"> <!-- Dimension (Shape) -->
  <col style="width: 140px"> <!-- Non-contiguous Tensor -->
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
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td>Input x in the formula.</td>
      <td>
        <ul>
          <li>Empty tensors are not supported.</li>
          <li>The parameter does not have the Optional suffix, and null pointer cannot be passed.</li>
        </ul>
      </td>
      <td>FLOAT16, BFLOAT16, INT8</td>
      <td>ND</td>
      <td>2D to 8D [M, K1]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight1 (aclTensor*)</td>
      <td>Input</td>
      <td>Expert weight data, corresponding to W1 in the formula.</td>
      <td>
        <ul>
          <li>Empty tensors are not supported.</li>
          <li>The parameter does not have the Optional suffix, and null pointer cannot be passed.</li>
        </ul>
      </td>
      <td>FLOAT16, BFLOAT16, INT8, INT4</td>
      <td>ND</td>
      <td>With expert [E, K1, N1]<br>Without expert [K1,N1]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight2 (aclTensor*)</td>
      <td>Input</td>
      <td>Expert weight data, corresponding to W2 in the formula.</td>
      <td>
        <ul>
          <li>Empty tensors are not supported.</li>
          <li>The parameter does not have the Optional suffix, and null pointer cannot be transferred.</li>
        </ul>
      </td>
      <td>FLOAT16, BFLOAT16, INT8, INT4</td>
      <td>ND</td>
      <td>With experts [E, K2, N2]<br>Without experts [K2,N2]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>expertTokens (aclIntArray*)</td>
      <td>Optional input</td>
      <td>Number of tokens of each expert.</td>
      <td>
        <ul>
          <li>A null pointer (null tensor) can be transferred.</li>
          <li>The parameter does not have the Optional suffix. There is no token quantity restriction when a null pointer is transferred.</li>
          <li>If the value is not empty, the maximum length is 256.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>ND</td>
      <td>1D, with a maximum length of 256</td>
      <td>-</td>
    </tr>
    <tr>
      <td>bias1 (aclTensor*)</td>
      <td>Optional input</td>
      <td>Weight data correction value, corresponding to b1 in the formula.</td>
      <td>
        <ul>
          <li>A null tensor can be transferred, and a null pointer can be transferred.</li>
          <li>The parameter does not have the Optional suffix. There is no offset restriction when a null pointer is transferred.</li>
        </ul>
      </td>
      <td>FLOAT16, FLOAT32, INT32</td>
      <td>ND</td>
      <td>With experts [E, N1]<br>Without experts [N1]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>bias2 (aclTensor*)</td>
      <td>Optional input</td>
      <td>Weight data correction value, corresponding to b2 in the formula.</td>
      <td>
        <ul>
          <li>An empty tensor is supported. A null pointer can be passed.</li>
          <li>If there is no suffix Optional, passing a null pointer does not have offset constraints.</li>
        </ul>
      </td>
      <td>FLOAT16, FLOAT32, INT32</td>
      <td>ND</td>
      <td>With expert [E, N2]<br>Without expert [N2]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale (aclTensor*)</td>
      <td>Optional input</td>
      <td>Quantization parameter: quantization scale factor.</td>
      <td>
        <ul>
          <li>An empty tensor is supported. A null pointer can be passed.</li>
          <li>If there is no suffix Optional, passing a null pointer does not have scaling constraints.</li>
        </ul>
      </td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>Per-tensor 1D, with expert [E], without expert [1]<br>Per-channel, with expert [E, N1], without expert [N1]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>offset (aclTensor*)</td>
      <td>Optional input</td>
      <td>Quantization parameter: quantization offset.</td>
      <td>
        <ul>
          <li>An empty tensor is supported. A null pointer can be passed.</li>
          <li>If there is no suffix Optional, passing a null pointer does not have offset constraints.</li>
        </ul>
      </td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1D, expert available [E]<br>Expert not available [1]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>deqScale1 (aclTensor*)</td>
      <td>Optional input</td>
      <td>Quantization parameter: dequantization scale factor for the first MatMul.</td>
      <td>
        <ul>
          <li>An empty tensor is supported. A null pointer can be passed.</li>
          <li>If there is no suffix "Optional", passing a null pointer does not impose any dequantization constraints.</li>
        </ul>
      </td>
      <td>UINT64, INT64, FLOAT32, BFLOAT16</td>
      <td>ND</td>
      <td>Expert available [E, N1]<br>Expert not available [N1]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>deqScale2 (aclTensor*)</td>
      <td>Optional input</td>
      <td>Quantization parameter: dequantization scale factor for the second MatMul.</td>
      <td>
        <ul>
          <li>An empty tensor is supported. A null pointer can be passed.</li>
          <li>If there is no suffix "Optional", passing a null pointer does not impose any dequantization constraints.</li>
        </ul>
      </td>
      <td>UINT64, INT64, FLOAT32, BFLOAT16</td>
      <td>ND</td>
      <td>Expert available [E, N2]<br>Expert not available [N2]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>antiquantScale1 (aclTensor*)</td>
      <td>Optional input</td>
      <td>Pseudo-quantization parameter: scale factor for the first MatMul.</td>
      <td>
        <ul>
          <li>An empty tensor is supported. You can pass a null pointer.</li>
          <li>If the suffix Optional is not added, passing a null pointer does not impose any fake-quantization constraints.</li>
        </ul>
      </td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>Per-channel: with expert [E, N1], without expert [N1]<br>Per-group: with expert [E, G, N1], without expert [[G,N1]</td>]
      <td>√</td>
    </tr>
    <tr>
      <td>antiquantScale2 (aclTensor*)</td>
      <td>Optional input</td>
      <td>Pseudo-quantization parameter: scale factor for the second MatMul.</td>
      <td>
        <ul>
          <li>An empty tensor is supported. You can pass a null pointer.</li>
          <li>If the suffix Optional is not added, passing a null pointer does not impose any fake-quantization constraints.</li>
        </ul>
      </td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>Per-channel: with expert [E, N2], without expert [N2]<br>Per-group: with expert [E, G, N2], without expert [[G,N2]</td>]
      <td>√</td>
    </tr>
    <tr>
      <td>antiquantOffset1 (aclTensor*)</td>
      <td>Optional input</td>
      <td>Pseudo-quantization parameter: offset for the first MatMul.</td>
      <td>
        <ul>
          <li>An empty tensor is supported. You can pass a null pointer.</li>
          <li>If the suffix Optional is not added, passing a null pointer does not impose any fake-quantization constraints.</li>
        </ul>
      </td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>per-channel: with experts [E, N1], without experts [N1]<br>per-group: with experts [E, G, N1], without experts [[G,N1]</td>]
      <td>-</td>
    </tr>
    <tr>
      <td>antiquantOffset2 (aclTensor*)</td>
      <td>Optional input</td>
      <td>Pseudo-quantization parameter: offset for the second MatMul.</td>
      <td>
        <ul>
          <li>An empty tensor is supported. A null pointer can be passed.</li>
          <li>There is no suffix Optional. If a null pointer is passed, there is no fake-quantization restriction.</li>
        </ul>
      </td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>per-channel: with experts [E, N2], without experts [N2]<br>per-group: with experts [E, G, N2], without experts [[G,N2]</td>]
      <td>-</td>
    </tr>
    <tr>
      <td>activation (char*)</td>
      <td>Input</td>
      <td>Activation function used in the formula.</td>
      <td>
        <ul>
          <li>The concept of empty tensor is not supported. A null pointer cannot be passed.</li>
          <li>There is no suffix Optional. A valid value must be passed.</li>
          <li>The value can be fastgelu, gelu, relu, silu, geglu, swiglu, or reglu.</li>
        </ul>
      </td>
      <td>CHAR</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>innerPrecise (int64_t)</td>
      <td>Optional input</td>
      <td>High precision or high performance.</td>
      <td>
        <ul>
          <li>The concept of empty tensor is not supported. The default value can be transferred.</li>
          <li>The suffix Optional is not supported. This parameter is valid only for FLOAT16.</li>
          <li>0: high precision (FLOAT32 computing); 1: high performance.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y (aclTensor*)</td>
      <td>Output</td>
      <td>Output of the computation, that is, the output y in the formula.</td>
      <td>
        <ul>
          <li>Empty tensors are not supported.</li>
          <li>The suffix Optional is not supported. Null pointers cannot be transferred.</li>
        </ul>
      </td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>Same as the x dimension</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
      <td>
        <ul>
          <li>The concept of empty tensor is not supported. Null pointers cannot be transferred.</li>
          <li>The suffix Optional is not supported. A non-negative integer is returned.</li>
        </ul>
      </td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Operator executor, containing the operator computation process.</td>
      <td>
        <ul>
          <li>The concept of empty tensor is not supported. Therefore, null pointers cannot be passed.</li>
          <li>The returned executor does not have the Optional suffix. Resources of the returned executor must be released.</li>
        </ul>
      </td>
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

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 287px">
  <col style="width: 119px">
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
      <td>The input parameter is a required input, output, or attribute, and is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data type or data format of x, weight1, weight2, activation, expertTokens, bias1, bias2, or y is not supported.</td>
    </tr>
  </tbody>
  </table>

## aclnnFFN

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
      <td>Address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnFFNGetWorkspaceSize.</td>
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

## Constraints

- Deterministic computing:
  - `aclnnFFN` defaults to a non-deterministic implementation. You can call `aclrtCtxSetSysParamOpt` to enable deterministic computing.
- If there are experts, the total number of experts must be the same as M of `x`.
- When the activation layer is geglu, swiglu, or reglu, only the FLOAT16 high-performance scenario without expert groups is supported, and N1 = 2 × K2. (The FLOAT16 scenario refers to a scenario where the data type of all required aclTensor parameters is FLOAT16.)
- When the activation layer is gelu, fastgelu, relu, or silu, FLOAT16 high-precision and high-performance scenarios, BFLOAT16 scenarios, quantization scenarios, and fake-quantization scenarios with experts or without expert groups are supported, and N1=K2.
- In all scenarios, K1=N2, K1<65536, K2<65536. The M axis must be less than the maximum value of INT32 after 32-byte alignment.
- In non-quantization scenarios, do not input quantization or pseudo-quantization parameters. In quantization scenarios, do not input pseudo-quantization parameters. In pseudo-quantization scenarios, do not input quantization parameters.
- Parameter types in quantization scenarios: The data type of `x` is INT8, that of `weight` is INT8, that of `bias` is INT32, that of `scale` is FLOAT32, and that of `offset` is FLOAT32. Other parameter types are divided into two cases according to the type of `y`.
  - When the data type of `y` is FLOAT16, `deqScale` supports the following data types: UINT64, INT64, and FLOAT32.
  - When the data type of `y` is BFLOAT16, `deqScale` supports the BFLOAT16 data type.
  - The data types of `deqScale1` and `deqScale2` must be consistent.
- Parameter types in quantization scenarios where the per-channel mode of `scale` is supported: The data type of `x` is INT8, that of `weight` is INT8, that of `bias` is INT32, that of `scale` is FLOAT32, and that of `offset` is FLOAT32. Other parameter types are categorized into two cases according to the type of `y`.
  - When the data type of `y` is FLOAT16, `deqScale` supports the following data types: UINT64 and INT64.
  - When the data type of `y` is BFLOAT16, `deqScale` supports the BFLOAT16 data type.
  - The data types of `deqScale1` and `deqScale2` must be consistent.
- The pseudo-quantization scenario supports two parameter types:
  - The data type of `y` is FLOAT16, that of `x` is FLOAT16, that of `bias` is FLOAT16, that of `antiquantScale` is FLOAT16, and that of `antiquantOffset` is FLOAT16. `weight` supports data types INT8 and INT4.
  - The data type of `y` is BFLOAT16, that of `x` is BFLOAT16, that of `bias` is FLOAT32, that of `antiquantScale` is BFLOAT16, and that of `antiquantOffset` is BFLOAT16. `weight` supports data types INT8 and INT4.
- When the data type of `weight1` or `weight2` is INT4, the last dimension of the shape must be an even number.
- In pseudo-quantization scenarios, under per-group mode, K1 in `antiquantScale1` and `antiquantOffset1` must be exactly divided by G (number of groups), and K2 in `antiquantScale2` and `antiquantOffset2` must be exactly divided by G (number of groups).
- In pseudo-quantization scenarios, under per-group mode, the data type of `weight` can only be INT4.
- In the BFLOAT16 non-quantization scenario, `innerPrecise` can only be set to `0`. In the FLOAT16 non-quantization scenario, `innerPrecise` can be set to `0` or `1`. In quantization or fake-quantization scenarios, `innerPrecise` can be set to `0` or `1`, but the setting does not take effect.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_ffn.h"
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
  // 1. (Boilerplate) Initialize the device and stream. For details, see the list of external AscendCL APIs.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as required.
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API.
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

  // aclnnFFN API call example
  LOG_PRINT("test aclnnFFN\n");

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  // Call the first-phase API of aclnnFFN.
  ret = aclnnFFNGetWorkspaceSize(self, weight1, weight2, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, "relu", 1, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFFNGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the computed workspaceSize.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnFFN.
  ret = aclnnFFN(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFFN failed. ERROR: %d\n", ret); return ret);

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
