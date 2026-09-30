# aclnnQuantGroupedMatmulDequant

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     ×    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     ×    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference products</term>   |     √    |
|  <term>Atlas training products</term>   |     ×    |

## Description

- Description: Quantizes the input x, performs grouped matrix multiplication, and dequantizes the result.
- Formula: 
  1. If smoothScaleOptional is input, then:
  
      $$
          x = x\cdot scale_{smooth}
      $$

  2. If xScaleOptional is not input, dynamic quantization is performed, and the x quantization coefficient needs to be calculated.
  
      $$
          scale_{x}=row\_max(abs(x))/max_{quantDataType}
      $$

  3. Quantization
  
      $$
          x_{quantized}=round(x/scale_{x})
      $$

  4. Grouped matrix multiplication + dequantization
    - 4.1 If the data type of the input $scale_{weight}$ is FLOAT32, then:

  $$
    \begin{aligned}
      x^{*}_{quantized} &= x_{quantized}[group[i-1]:group[i]]\\
      out^{*}_{quantized} &= out_{quantized}[group[i-1]:group[i]]\\
      scale^{*}_{x} &= \begin{cases}
                      scale_{x}[group[i-1]:group[i]] & pertoken \\
                      scale_{x} & pertensor \\
                      \end{cases} \\
      out^{*}_{quantized} &= (x^{*}_{quantized}@weight_{quantized}[i] + bias) * scale_{weight}[i] * scale^{*}_{x}
    \end{aligned}
  $$

  - 4.2 If the data type of the input $scale_{weight}$ is INT64, then:

    $$
    x^{*}_{quantized} = x_{quantized}[group[i-1]:group[i]] \\
    out^{*}_{quantized} = out_{quantized}[group[i-1]:group[i]] \\
    scale_{weight} = torch.tensor(np.frombuffer(scale_{weight}.numpy().astype(np.int32).\\tobytes(), dtype=np.float32)).reshape(scale_{weight}) \\
    out^{*}_{quantized} = (x^{*}_{quantized}@weight_{quantized}[i] + bias) * scale_{weight}[i]
    $$

    Note: In the scenario described in 4.2, the matrix multiplication operation has been performed on $scale_{weight}$ and $scale_{x}$ before the input of $scale_{weight}$. Therefore, this step is omitted during internal operator computation. This requires that the scenario must be per-tensor static quantization. That is, before input, $scale_{weight}$ must be processed as follows to obtain data of the INT64 type:

    $$
    scale_{weight} = scale_{weight} * scale_{x} \\
    scale_{weight} = torch.tensor(np.frombuffer(scale_{weight}.numpy().astype(np.float32). \\tobytes(), dtype=np.int32).astype(np.int64)).reshape(scale_{weight})
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnQuantGroupedMatmulDequantGetWorkspaceSize` is called to obtain the input parameters and compute the required workspace size based on the process. Then, `aclnnQuantGroupedMatmulDequant` is called to perform computation.

```Cpp
aclnnStatus aclnnQuantGroupedMatmulDequantGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *weight,
  const aclTensor *weightScale,
  const aclTensor *groupList,
  const aclTensor *biasOptional,
  const aclTensor *xScaleOptional,
  const aclTensor *xOffsetOptional,
  const aclTensor *smoothScaleOptional,
  char            *xQuantMode,
  bool             transposeWeight,
  const aclTensor *out,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnQuantGroupedMatmulDequant(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnQuantGroupedMatmulDequantGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1300px"><colgroup>
  <col style="width: 101px">
  <col style="width: 115px">
  <col style="width: 220px">
  <col style="width: 200px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 238px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Precaution</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
   <tbody>
      <tr>
      <td>x</td>
      <td>Input</td>
      <td>Left matrix of the input, which is mandatory and corresponds to x in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The shape supports two dimensions (m, k).</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
     <tr>
      <td>weight</td>
      <td>Input</td>
      <td>Right matrix of the input, which is mandatory and corresponds to weight_{quantized} in the formula.</td>
      <td>Empty tensors are not supported.</td>
      <td>INT8</td>
      <td>FRACTAL_NZ, ND</td>
      <td>3, 5</td>
      <td>√</td>
    </tr>
      <tr>
      <td>weightScale</td>
      <td>Input</td>
      <td>Quantization scale of the weight, which is mandatory and corresponds to scale_{weight} in the formula.</td>
      <td><ul><li>The shape supports two dimension (g, n), where g and n are the same as those of weight. </li><li>When the data type of this parameter is INT64, the data type of xScaleOptional must be FLOAT16 and the value of xQuantMode must be pertensor.</li></ul></td>
      <td>FLOAT32, INT64</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
      <tr>
      <td>biasOptional</td>
      <td>Input</td>
      <td>Bias of the computation, which is optional and corresponds to bias in the formula.</td>
      <td>Currently, only a null pointer can be passed.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>xScaleOptional</td>
      <td>Input</td>
      <td>Quantization scale of x, which is optional and corresponds to scale_{x} in the formula.</td>
      <td><ul><li>When xQuantMode is pertensor, the shape is one dimension (1,); when xQuantMode is pertoken, the shape is one dimension (m,), where m is the same as that of the input x. If this parameter is set to null, dynamic quantization is used. </li><li>When the data type of this parameter is FLOAT16, the data type of weightScale must be INT64 and the value of xQuantMode must be pertensor. </li><li>Empty tensors are supported.</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
     <tr>
      <td>xOffsetOptional</td>
      <td>Input</td>
      <td>Offset of x, which is optional.</td>
      <td>Currently, only a null pointer can be passed.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>smoothScaleOptional</td>
      <td>Input</td>
      <td>Smoothing coefficient of x, which is optional and corresponds to scale_{smooth} in the formula.</td>
      <td><ul><li>The shape supports one dimension (k,), where k is the same as that of x. </li><li>Empty tensors are supported.</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
       <tr>
      <td>xQuantMode</td>
      <td>Input</td>
      <td>Quantization mode of the input x.</td>
      <td>The value can be pertoken or pertensor. Only pertoken is supported in dynamic quantization. pertoken indicates that each token (row) has its own quantization parameters. pertensor indicates that the entire tensor uses the same quantization parameters.</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
       <tr>
      <td>transposeWeight</td>
      <td>Input</td>
      <td>Whether to transpose the input weight.</td>
      <td>Currently, only true is supported.</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>out</td>
      <td>Output</td>
      <td>Compute result, which is mandatory and corresponds to out in the formula.</td>
      <td>The shape supports two dimensions (m, n). m is the same as that of x, and n is the same as that of weight.</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
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

    - In ND format, the shape of `weight` can be three dimensions.
      - When `transposeWeight` is set to `true`, the dimensions are (g, n, k).
      - When `transposeWeight` is set to `false`, the dimensions are (g, k, n).
    - In FRACTAL_NZ format, the shape of `weight` can be five dimensions.
      - When `transposeWeight` is set to true, the dimensions are (g, k1, n1, n0, k0), where k0 = 32, n0 = 16, and k1 and k of `x` must satisfy ceilDiv(k, 32) = k1.
      - When `transposeWeight` is set to `false`, the dimensions are (g, n1, k1, k0, n0), where k0 = 16, n0 = 32, and k1 and k of `x` must meet the following relationship: ceilDiv(k, 16) = k1.
      - `aclnnCalculateMatmulWeightSizeV2` and `aclnnTransMatmulWeight` can be used to convert the input format from ND to FRACTAL_NZ.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown:

  <table style="undefined;table-layout: fixed; width: 1048px"><colgroup>
  <col style="width: 319px">
  <col style="width: 108px">
  <col style="width: 621px">
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
      <td><li>The input parameter type is aclTensor and its data type is not supported. </li><li>The value of n or k in the shape of weight cannot be exactly divided by 16.</li></td>
    </tr>
    <tr>
      <td rowspan="3">aclnnAdvanceStepGetWorkspaceSize failed</td>
      <td rowspan="3">561002</td>
      <td>The input parameter type is aclTensor and its shape does not comply with the preceding parameter description.</td>
    </tr>
  </tbody>
  </table>

## aclnnFatreluMul

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnQuantGroupedMatmulDequantGetWorkspaceSize.</td>
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
  - `aclnnQuantGroupedMatmulDequant` defaults to a deterministic implementation.

- n and k must be integer multiples of 16.
- If the data type of `weightScale` is INT64, the data type of `xScaleOptional` must be FLOAT16 and the value of `xQuantMode` must be `pertensor`. If the data type of `xScaleOptional` is FLOAT16, the data type of `weightScale` must be INT64 and the value of `xQuantMode` must be `pertensor`.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_grouped_matmul_dequant.h"

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
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy the data from the host to the memory on the device.
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
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
  int G = 4;
  int M = 64;
  int K = 256;
  int N = 512;

  char quantMode[16] = "pertoken";
  bool transposeWeight = true;

  std::vector<int64_t> xShape = {M,K};
  std::vector<int64_t> weightShape = {G,N,K};
  std::vector<int64_t> weightScaleShape = {G,N};
  std::vector<int64_t> xScaleShape = {M};
  std::vector<int64_t> smoothScaleShape = {K};
  std::vector<int64_t> outShape = {M,N};
  std::vector<int64_t> groupListShape = {G};

  void* xDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* weightScaleDeviceAddr = nullptr;
  void* groupListDeviceAddr = nullptr;
  void* xScaleDeviceAddr = nullptr;
  void* smoothScaleDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;

  aclTensor* x = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* weightScale = nullptr;
  aclTensor* groupList = nullptr;
  aclTensor* bias = nullptr;
  aclTensor* xScale = nullptr;
  aclTensor* xOffset = nullptr;
  aclTensor* smoothScale = nullptr;
  aclTensor* out = nullptr;

  std::vector<uint16_t> xHostData(GetShapeSize(xShape));
  std::vector<uint8_t> weightHostData(GetShapeSize(weightShape));
  std::vector<float> weightScaleHostData(GetShapeSize(weightScaleShape));
  std::vector<int64_t> groupListHostData(GetShapeSize(groupListShape));
  groupListHostData[0] = 7;
  groupListHostData[0] = 32;
  groupListHostData[0] = 40;
  groupListHostData[0] = 64;
  std::vector<float> xScaleHostData(GetShapeSize(xScaleShape));
  std::vector<uint16_t> smoothScaleHostData(GetShapeSize(smoothScaleShape));
  std::vector<uint16_t> outHostData(GetShapeSize(outShape));

  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_INT8, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(weightScaleHostData, weightScaleShape, &weightScaleDeviceAddr, aclDataType::ACL_FLOAT, &weightScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupList);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(xScaleHostData, xScaleShape, &xScaleDeviceAddr, aclDataType::ACL_FLOAT, &xScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(smoothScaleHostData, smoothScaleShape, &smoothScaleDeviceAddr, aclDataType::ACL_FLOAT16, &smoothScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnQuantGroupedMatmulDequant.
  ret = aclnnQuantGroupedMatmulDequantGetWorkspaceSize(x, weight, weightScale, groupList,
                                                bias, xScale, xOffset, smoothScale,
                                                quantMode, transposeWeight, out, 
                                                &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantGroupedMatmulDequantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnQuantGroupedMatmulDequant.
  ret = aclnnQuantGroupedMatmulDequant(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantGroupedMatmulDequant failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  PrintOutResult(outShape, &outDeviceAddr);

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(x);
  aclDestroyTensor(weight);
  aclDestroyTensor(weightScale);
  aclDestroyTensor(groupList);
  aclDestroyTensor(xScale);
  aclDestroyTensor(smoothScale);
  aclDestroyTensor(out);

  // 7. Release device resources.
  aclrtFree(xDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(weightScaleDeviceAddr);
  aclrtFree(groupListDeviceAddr);
  aclrtFree(xScaleDeviceAddr);
  aclrtFree(smoothScaleDeviceAddr);
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
