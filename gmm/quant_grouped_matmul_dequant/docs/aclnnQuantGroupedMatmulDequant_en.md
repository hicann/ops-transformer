# aclnnQuantGroupedMatmulDequant

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T12:36:46.907Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Not supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: performs quantization, grouped matrix multiplication, and dequantization on the input x.
- Formula:
  1.If smoothScaleOptional is input, then

  $$
      x = x\cdot scale_{smooth}
  $$

  2\. If xScaleOptional is not input, dynamic quantization is used. The quantization coefficients of x need to be calculated.

  $$
      scale_{x}=row\_max(abs(x))/max_{quantDataType}
  $$

  3\. Quantization

  $$
      x_{quantized}=round(x/scale_{x})
  $$

  4\. Grouped matrix multiplication + dequantization
    - 4.1 If the data type of the input $scale_{weight}$ is FLOAT32:

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

  - 4.2 If the data type of the input $scale_{weight}$ is INT64:

    $$
    x^{*}_{quantized} = x_{quantized}[group[i-1]:group[i]] \\
    out^{*}_{quantized} = out_{quantized}[group[i-1]:group[i]] \\
    scale_{weight} = torch.tensor(np.frombuffer(scale_{weight}.numpy().astype(np.int32).\\tobytes(), dtype=np.float32)).reshape(scale_{weight}) \\
    out^{*}_{quantized} = (x^{*}_{quantized}@weight_{quantized}[i] + bias) * scale_{weight}[i]
    $$

    Note: In use case 4.2 above, $scale_{weight}$ has already been multiplied by $scale_{x}$ before being input, so the operator omits this step in the internal computation. This requires the pertensor static quantization use case. That is, $scale_{weight}$ must be processed as follows before being input to obtain INT64 data:

    $$
    scale_{weight} = scale_{weight} * scale_{x} \\
    scale_{weight} = torch.tensor(np.frombuffer(scale_{weight}.numpy().astype(np.float32). \\tobytes(), dtype=np.int32).astype(np.int64)).reshape(scale_{weight})
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnQuantGroupedMatmulDequantGetWorkspaceSize" API first to obtain the input parameters and calculate the required workspace size according to the process, and then call the "aclnnQuantGroupedMatmulDequant" API to perform the computation.

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

- **Parameters**

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
      <td>x</td>
      <td>Input</td>
      <td>Left matrix of the input. Required parameter. x in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The shape supports 2 dimensions, representing (m, k) respectively.</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
     <tr>
      <td>weight</td>
      <td>Input</td>
      <td>Right matrix of the input. Required parameter. weight_{quantized} in the formula.</td>
      <td>Empty tensors are not supported.</td>
      <td>INT8</td>
      <td>FRACTAL_NZ, ND</td>
      <td>3, 5</td>
      <td>√</td>
    </tr>
      <tr>
      <td>weightScale</td>
      <td>Input</td>
      <td>Quantization coefficient of weight. This is a required parameter, corresponding to scale_{weight} in the formula.</td>
      <td><ul><li>The shape is 2D (g, n), where g and n are consistent with those of weight.</li><li>If the data type is INT64, the data type of xScaleOptional must be FLOAT16 and the value of xQuantMode must be pertensor.</li></ul></td>
      <td>FLOAT32, INT64</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
      <tr>
      <td>groupList</td>
      <td>Input</td>
      <td>The group in the formula. Represents the prefix sum result of grouping the input tensor x in MoE use cases, with a 1-dimensional shape of [g].</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
      <td>biasOptional</td>
      <td>Input</td>
      <td>Indicates the offset for the computation, an optional parameter, corresponding to bias in the formula.</td>
      <td>Currently, only a null pointer is supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>xScaleOptional</td>
      <td>Input</td>
      <td>Quantization coefficient of x. This parameter is optional and corresponds to scale_{x} in the formula.</td>
      <td><ul><li>When xQuantMode is pertensor, the shape is 1-D (1,); when xQuantMode is pertoken, the shape is 1-D (m,), where m is the same as the m of input x. If it is empty, dynamic quantization is performed.</li><li>When the data type is FLOAT16, the data type of weightScale must be INT64, and the value of xQuantMode must be pertensor.</li><li>Empty tensors are supported.</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
     <tr>
      <td>xOffsetOptional</td>
      <td>Input</td>
      <td>Offset of x. This parameter is optional.</td>
      <td>Currently, only a null pointer can be passed in.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>smoothScaleOptional</td>
      <td>Input</td>
      <td>Smoothing factor of x, optional, which is scale_{smooth} in the formula.</td>
      <td><ul><li>Shape is 1D (k,), where k is the same as the k of x.</li><li>Empty tensors are supported.</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
       <tr>
      <td>xQuantMode</td>
      <td>Input</td>
      <td>Specifies the quantization mode of input x.</td>
      <td>Supported values: pertoken/pertensor. Only pertoken is supported for dynamic quantization. pertoken indicates that each token (row) has its own quantization parameters; pertensor indicates that the entire tensor uses unified quantization parameters.</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
       <tr>
      <td>transposeWeight</td>
      <td>Input</td>
      <td>Indicates whether the input weight is transposed.</td>
      <td>Currently, only true is supported.</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>out</td>
      <td>Output</td>
      <td>Computation result, a required parameter, corresponding to out in the formula.</td>
      <td>The shape supports 2 dimensions, where each dimension indicates: (m, n). m is the same as m of x, and n is the same as n of weight.</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
      <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

    - In ND format, the shape of weight supports 3 dimensions.
      - When transposeWeight is true, the dimensions are: (g, n, k).
      - When transposeWeight is false, the dimensions are: (g, k, n).
    - In FRACTAL_NZ format, the shape of weight supports 5 dimensions.
      - When transposeWeight is true, the dimensions are: (g, k1, n1, n0, k0), where k0 = 32, n0 = 16, and k1 and the k dimension of x must satisfy the following relationship: ceilDiv(k, 32) = k1.
      - When transposeWeight is false, the dimensions are: (g, n1, k1, k0, n0), where k0 = 16, n0 = 32, and k1 and the k dimension of x must satisfy the following relationship: ceilDiv(k, 16) = k1.
      - Use the aclnnCalculateMatmulWeightSizeV2 API and the aclnnTransMatmulWeight API to convert the input Format from ND to FRACTAL_NZ.

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1048px"><colgroup>
  <col style="width: 319px">
  <col style="width: 108px">
  <col style="width: 621px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td><li>A required input or output is a null pointer.</li><li>biasOptional or xOffsetOptional is not a null pointer, which is not supported.</li></td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>Examples include:<li>The checked tensor data type combination is unsupported, out and x have different data types, or the device is unsupported.</li><li>transposeWeight is false.</li><li>For non-empty inputs, weight or out is not contiguous.</li><li>For non-empty inputs, weight is three-dimensional ND/NCL and N or K is not divisible by 16; or weight is not three-dimensional ND/NCL and either its storageShape is not five-dimensional or its storage format is not FRACTAL_NZ.</li></td>
    </tr>
    <tr>
      <td rowspan="1">aclnnQuantGroupedMatmulDequantGetWorkspaceSize failed</td>
      <td rowspan="1">561002</td>
      <td>If this error code is returned, check the error logs for the specific cause.</td>
    </tr>
  </tbody>
  </table>

## aclnnQuantGroupedMatmulDequant

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
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
      <td>Workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device side, obtained by the first-phase API aclnnQuantGroupedMatmulDequantGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- deterministic computation:
  - aclnnQuantGroupedMatmulDequant defaults to a deterministic implementation.

- Both n and k must be multiples of 16.
- When the data type of weightScale is INT64, the data type of xScaleOptional must be FLOAT16, and the value of xQuantMode must be pertensor; when the data type of xScaleOptional is FLOAT16, the data type of weightScale must be INT64, and the value of xQuantMode must be pertensor.

## Example

The sample code is as follows and is for reference only. For the specific compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

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
  // Boilerplate for resource initialization.
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
  // Call aclrtMalloc to allocate memory on the device side.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy data from the Host side to the memory on the device side.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
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
  // 1. (Boilerplate) Initialize device/stream. For details, see the acl API documentation.
  // Fill in deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Custom construction is required based on the API.
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
  groupListHostData[1] = 32;
  groupListHostData[2] = 40;
  groupListHostData[3] = 64;
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

  // 3. Call the CANN operator library API, and replace it with the specific API name
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnQuantGroupedMatmulDequant
  ret = aclnnQuantGroupedMatmulDequantGetWorkspaceSize(x, weight, weightScale, groupList,
                                                bias, xScale, xOffset, smoothScale,
                                                quantMode, transposeWeight, out,
                                                &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantGroupedMatmulDequantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on the workspaceSize calculated by the first-phase API
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnQuantGroupedMatmulDequant
  ret = aclnnQuantGroupedMatmulDequant(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantGroupedMatmulDequant failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Wait synchronously for the task execution to complete
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value, copy the result on the device side to the host side, and modify it based on the API definition of the specific API
  PrintOutResult(outShape, &outDeviceAddr);

  // 6. Release aclTensor and aclScalar, and modify them based on the API definition of the specific API
  aclDestroyTensor(x);
  aclDestroyTensor(weight);
  aclDestroyTensor(weightScale);
  aclDestroyTensor(groupList);
  aclDestroyTensor(xScale);
  aclDestroyTensor(smoothScale);
  aclDestroyTensor(out);

  // 7. Release device resources
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
