# aclnnSwinTransformerLnQkvQuant

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    ×     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    ×     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- API function: The Swin Transformer network model completes the calculation of Q, K, and V. 
- Formula: 

  q/k/v = (Quant(Layernorm(x).transpose)  * weight).dequant.transpose.split
  **weight** is a concatenation of weights of three matrices Q, K, and V.

## Function Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnSwinTransformerLnQkvQuantGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnSwinTransformerLnQkvQuant` is called to perform computation.

```cpp
aclnnStatus aclnnSwinTransformerLnQkvQuantGetWorkspaceSize(
  const aclTensor *x, 
  const aclTensor *gamma, 
  const aclTensor *beta, 
  const aclTensor *weight, 
  const aclTensor *bias, 
  const aclTensor *quantScale, 
  const aclTensor *quantOffset, 
  const aclTensor *dequantScale, 
  int64_t          headNum, 
  int64_t          seqLength, 
  double           epsilon, 
  int64_t          oriHeight, 
  int64_t          oriWeight, 
  int64_t          hWinSize, 
  int64_t          wWinSize, 
  bool             weightTranspose, 
  const aclTensor *queryOutputOut, 
  const aclTensor *keyOutputOut, 
  const aclTensor *valueOutputOut, 
  uint64_t        *workspaceSize, 
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnSwinTransformerLnQkvQuant(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```

## aclnnSwinTransformerLnQkvQuantGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1524px"><colgroup>
  <col style="width: 166px">
  <col style="width: 121px">
  <col style="width: 336px">
  <col style="width: 250px">
  <col style="width: 149px">
  <col style="width: 128px">
  <col style="width: 230px">
  <col style="width: 144px">
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
      <td>x (aclTensor *) </td>
      <td>Input</td>
      <td>Indicates the target tensor for normalization computation.</td>
      <td>-</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>Only the dimension [B, S, H] is supported, where B is the batch size and must be [1, 32], S is the product of the length and width of the original image, and H is the product of the sequence length and the number of channels and is less than or equal to 1024</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gamma (aclTensor *) </td>
      <td>Input</td>
      <td>Indicates the size of the scale in layer normalization computation.</td>
      <td>-</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>Only 1-dimensional [H]is supported.</td>
      <td>-</td>
    </tr>
    <tr>
      <td>beta (aclTensor *) </td>
      <td>Input</td>
      <td>Indicates the size of the bias in layer normalization computation.</td>
      <td>-</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>Only 1-dimensional [H]is supported.</td>
      <td>-</td>
    </tr>
    <tr>
      <td>weight (aclTensor *)</td>
      <td>Input</td>
      <td>Weight matrix used for converting the target tensor.</td>
      <td>-</td>
      <td>INT8</td>
      <td>ND</td>
      <td>Only 2D tensors with dimension [H, 3 * H] are supported.</td>
      <td>-</td>
    </tr>
    <tr>
      <td>bias (aclTensor *)</td>
      <td>Input</td>
      <td>Offset matrix used for converting the target tensor.</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>Only 1D tensors with dimension [3 * H] are  supported.</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantScale (aclTensor *)</td>
      <td>Input</td>
      <td>Scaling parameter used for quantizing the target tensor.</td>
      <td>-</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>Only 1D tensors with dimension [H] are  supported.</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantOffset (aclTensor *)</td>
      <td>Input</td>
      <td>Offset parameter used for quantizing the target tensor.</td>
      <td>-</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>Only 1D tensors with dimension [H] are  supported.</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dequantScale (aclTensor *)</td>
      <td>Input</td>
      <td>Indicates the scaling parameter used for dequantization after the target tensor is multiplied by the weight matrix.</td>
      <td>-</td>
      <td>UINT64</td>
      <td>ND</td>
      <td>Only one-dimensional tensors with dimension [3 * H] are  supported.</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headNum (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of channels used for conversion.</td>
      <td>The value range is [1, 32].</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>seqLength (int64_t)</td>
      <td>Input</td>
      <td>Indicates the channel depth used for conversion.</td>
      <td>The value can be 32 or 64.</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon (double)</td>
      <td>Input</td>
      <td>layernorm: divide-by-zero protection value.</td>
      <td>To ensure accuracy, it is recommended that the value be less than or equal to 1e-4.</td>
      <td>float</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriHeight (int64_t)</td>
      <td>Input</td>
      <td>Dimension of the transpose operation on the S axis in layernorm. The value of oriHeight x oriWeight must be equal to the size of the second dimension S of the input x and must be an integer multiple of hWinSize.</td>
      <td>-</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriWeight (int64_t)</td>
      <td>Input</td>
      <td>Dimension of the transpose operation on the S axis in layernorm. The value of oriHeight x oriWeight must be equal to the size of the second dimension S of the input x and must be an integer multiple of wWinSize.</td>
      <td>-</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>hWinSize (int64_t)</td>
      <td>Input</td>
      <td>Height of the feature window.</td>
      <td>The value range is [7, 32].</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>wWinSize (int64_t)</td>
      <td>Input</td>
      <td>Width of the feature window.</td>
      <td>The value range is [7, 32].</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>weightTranspose (bool)</td>
      <td>Input</td>
      <td>Whether the weight matrix is transposed.</td>
      <td>Currently, the value False is not supported.</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>queryOutputOut (aclTensor *)</td>
      <td>Output</td>
      <td>Tensor after conversion, which is Q in the formula.</td>
      <td>-</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>keyOutputOut (aclTensor *)</td>
      <td>Output</td>
      <td>Tensor after conversion, which is K in the formula.</td>
      <td>-</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>valueOutputOut (aclTensor *)</td>
      <td>Output</td>
      <td>Tensor after conversion, which is V in the formula.</td>
      <td>-</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t)</td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor **)</td>
      <td>Output</td>
      <td>Operator executor, containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>  

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.
  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 291px">
  <col style="width: 135px">
  <col style="width: 723px">
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
      <td>The input tensor is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data type or format of the input or output parameter is not supported.</td>
    </tr>
  </tbody>
  </table>

## aclnnSwinTransformerLnQkvQuant

- **Parameters**

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
      <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnSwinTransformerLnQkvQuantGetWorkspaceSize.</td>
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

- Deterministic computation:
  - `aclnnSwinTransformerLnQkvQuant` defaults to deterministic implementation.
- The value of **seqLength** can only be 32 or 64.
- **oriHeight** × **oriWeight** = Second dimension of the input **x** tensor (**oriHeight** is an integer multiple of **hWinSize** and **oriWeight** is an integer multiple of **wWinSize**.)
- Both values of **hWinSize** and **wWinSize** range from 7 to 32.
- The first dimension B of the input **x** tensor ranges from 1 to 32.
- **weight** needs to be transposed.

## Calling Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_swin_transformer_ln_qkv_quant.h"

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
  // () Initialize resources.
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
  // Call aclrtMemcpy to copy host data to the device memory.
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
  // Set the device ID based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs based on API definitions.
  std::vector<int64_t> selfShape = {1, 49, 32};
  std::vector<int64_t> gammaShape = {32};
  std::vector<int64_t> weightShape = {32*3, 32};
  std::vector<int64_t> biasShape = {3 * 32};
  std::vector<int64_t> outShape = {1,1,49, 32};
  void* xDeviceAddr = nullptr;
  void* gammaDeviceAddr = nullptr;
  void* betaDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* biasDeviceAddr = nullptr;
  void* scaleDeviceAddr = nullptr;
  void* offsetDeviceAddr = nullptr;
  void* dequantDeviceAddr = nullptr;

  void* outqDeviceAddr = nullptr;
  void* outkDeviceAddr = nullptr;
  void* outvDeviceAddr = nullptr;
  aclTensor* x = nullptr;
  aclTensor* gamma = nullptr;
  aclTensor* beta = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* bias = nullptr;
  aclTensor* quantScale = nullptr;
  aclTensor* quantOffset = nullptr;
  aclTensor* dequantScale = nullptr;
  aclTensor* queryOutput = nullptr;
  aclTensor* keyOutput = nullptr;
  aclTensor* valueOutput = nullptr;

  std::vector<uint16_t> selfHostData(49*32, 0x1);
  std::vector<int32_t> biasHostData(3*32, 0x1);
  std::vector<uint16_t> gammaHostData(32, 0x1);
  std::vector<uint16_t> betaHostData(32, 0x1);
  std::vector<int8_t> weightHostData(3*32*32, 0x1);
  std::vector<uint16_t> scaleHostData(32, 0x1);
  std::vector<uint16_t> offsetHostData(32, 0x1);
  std::vector<uint64_t> dequantHostData(3*32, 0x1);

  std::vector<uint16_t> outqHostData(49*32, 0x0);
  std::vector<uint16_t> outkHostData(49*32, 0x0);
  std::vector<uint16_t> outvHostData(49*32, 0x0);

  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(betaHostData, gammaShape, &betaDeviceAddr, aclDataType::ACL_FLOAT16, &beta);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_INT8, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_INT32, &bias);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(scaleHostData, gammaShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT16, &quantScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(offsetHostData, gammaShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT16, &quantOffset);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(dequantHostData, biasShape, &dequantDeviceAddr, aclDataType::ACL_UINT64, &dequantScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outqHostData, outShape, &outqDeviceAddr, aclDataType::ACL_FLOAT16, &queryOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(outkHostData, outShape, &outkDeviceAddr, aclDataType::ACL_FLOAT16, &keyOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outvHostData, outShape, &outvDeviceAddr, aclDataType::ACL_FLOAT16, &valueOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. Call the CANN operator library API. Change the API name to the actual one.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  float epsilon = 0.0001;
  int64_t oriHeight = 7;
  int64_t oriWeight = 7;
  int64_t hWinSize = 7;
  int64_t wWinSize = 7;
  int64_t headNum = 1;
  int64_t seqLength = 32;
  bool weightTranspose = true;

  // Call the first-phase API of aclnnSwinTransformerLnQkvQuant.
  ret = aclnnSwinTransformerLnQkvQuantGetWorkspaceSize(x,gamma,beta,weight, bias, quantScale, quantOffset, dequantScale, headNum, seqLength, epsilon, oriHeight, oriWeight, hWinSize, wWinSize, weightTranspose, queryOutput, keyOutput, valueOutput, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwinTransformerLnQkvQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnSwinTransformerLnQkvQuant.
  ret = aclnnSwinTransformerLnQkvQuant(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwinTransformerLnQkvQuant failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Synchronize the stream and wait for task completion.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<uint16_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outqDeviceAddr,size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  // 6. Destroy aclTensors and aclScalars. Modify the code based on the API definition.
  aclDestroyTensor(x);
  aclDestroyTensor(queryOutput);
  // 7. Release device resources. Modify the code based on the API definition.
  aclrtFree(xDeviceAddr);
  aclrtFree(outqDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
