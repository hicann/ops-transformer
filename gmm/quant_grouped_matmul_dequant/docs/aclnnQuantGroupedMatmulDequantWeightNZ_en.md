# aclnnQuantGroupedMatmulDequantWeightNZ

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T12:40:24.159Z -->

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

- Description: Performs quantization, grouped matrix multiplication, and dequantization on input x. This API is a weightNZ-specialized version of [aclnnQuantGroupedMatmulDequant](./aclnnQuantGroupedMatmulDequant_en.md).
- Formula:
  1\. If smoothScaleOptional is provided, then

  $$
      x = x\cdot scale_{smooth}
  $$

  2.If xScaleOptional is not input, dynamic quantization is applied, and the quantization coefficient of x must be calculated.

  $$
      scale_{x}=row\_max(abs(x))/max_{quantDataType}
  $$

  3.Quantization

  $$
      x_{quantized}=round(x/scale_{x})
  $$

  4.Grouped matrix multiplication + dequantization
    - 4.1 If the data type of $scale_{weight}$ is FLOAT32:

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

  - 4.2 If the data type of $scale_{weight}$ is INT64:

    $$
    x^{*}_{quantized} = x_{quantized}[group[i-1]:group[i]] \\
    out^{*}_{quantized} = out_{quantized}[group[i-1]:group[i]] \\
    scale_{weight} = torch.tensor(np.frombuffer(scale_{weight}.numpy().astype(np.int32).\\tobytes(), dtype=np.float32)).reshape(scale_{weight}) \\
    out^{*}_{quantized} = (x^{*}_{quantized}@weight_{quantized}[i] + bias) * scale_{weight}[i]
    $$

    Special note: In use case 4.2 described above, $scale_{weight}$ has already undergone matrix multiplication with $scale_{x}$ before being passed in. Therefore, the operator omits this step during internal computation, which requires pertensor static quantization. That is, $scale_{weight}$ must be processed as follows before input to obtain data of the INT64 type:

    $$
    scale_{weight} = scale_{weight} * scale_{x} \\
    scale_{weight} = torch.tensor(np.frombuffer(scale_{weight}.numpy().astype(np.float32). \\tobytes(), dtype=np.int32).astype(np.int64)).reshape(scale_{weight})
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call the "aclnnQuantGroupedMatmulDequantWeightNZGetWorkspaceSize" API to obtain the input parameters and calculate the required workspace size according to the process, and then call the "aclnnQuantGroupedMatmulDequantWeightNZ" API to perform the computation.

```Cpp
aclnnStatus aclnnQuantGroupedMatmulDequantWeightNZGetWorkspaceSize(
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
  aclTensor *out,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnQuantGroupedMatmulDequantWeightNZ(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnQuantGroupedMatmulDequantWeightNZGetWorkspaceSize

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
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
   <tbody>
      <tr>
      <td>x</td>
      <td>Input</td>
      <td>Left input matrix, mandatory parameter, corresponding to x in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The shape supports 2 dimensions, which are (m, k).</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
     <tr>
      <td>weight</td>
      <td>Input</td>
      <td>Right input matrix, mandatory parameter, corresponding to weight_{quantized} in the formula.</td>
      <td>Empty tensors are not supported. Note that this API ignores the data format of weight and forcibly treats it as the FRACTAL_NZ format.</td>
      <td>INT8</td>
      <td>FRACTAL_NZ</td>
      <td>5</td>
      <td>-</td>
    </tr>
      <tr>
      <td>weightScale</td>
      <td>Input</td>
      <td>Represents the quantization scale of weight. Mandatory parameter. Corresponds to scale_{weight} in the formula.</td>
      <td><ul><li>The shape is 2D (g, n), where g and n must be the same as those of weight.</li><li>If the data type is INT64, the data type of xScaleOptional must be FLOAT16 and the value of xQuantMode must be pertensor.</li></ul></td>
      <td>FLOAT32, INT64</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
      <tr>
      <td>groupList</td>
      <td>Input</td>
      <td>group in the formula. Denotes the prefix sum result of grouping input tensor x in the MoE use case. The shape is 1D [g].</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
      <td>biasOptional</td>
      <td>Input</td>
      <td>Represents the offset for the computation. Optional parameter. Corresponds to the bias in the formula.</td>
      <td>Currently, only a null pointer can be passed.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>xScaleOptional</td>
      <td>Input</td>
      <td>Represents the quantization scale of x, optional parameter, scale_{x} in the formula.</td>
      <td><ul><li>When xQuantMode is pertensor, the shape is 1D (1,); when xQuantMode is pertoken, the shape is 1D (m,), where m is the same as that of input x. If empty, dynamic quantization is applied.</li><li>When the data type is FLOAT16, the data type of weightScale must be INT64, and xQuantMode must be pertensor.</li><li>Empty tensors are supported.</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
     <tr>
      <td>xOffsetOptional</td>
      <td>Input</td>
      <td>Represents the offset of x, optional parameter.</td>
      <td>Currently, only a null pointer can be passed in.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>smoothScaleOptional</td>
      <td>Input</td>
      <td>Smoothing coefficient of x, an optional parameter, corresponding to scale_{smooth} in the formula.</td>
      <td><ul><li>The shape is one-dimensional (k,), where k is the same as the k of x.</li><li>Empty tensors are supported.</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
       <tr>
      <td>xQuantMode</td>
      <td>Input</td>
      <td>Specifies the quantization mode of input x.</td>
      <td>Supported values are pertoken/pertensor. Only pertoken is supported for dynamic quantization. pertoken indicates that each token (each row) has its own quantization parameters; pertensor indicates that the entire tensor uses the same quantization parameters.</td>
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
      <td>Calculation result, mandatory parameter, out in the formula.</td>
      <td>Supports a 2D shape, with dimensions (m, n). m is the same as the m of x, and n is the same as the n of weight.</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
      <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be requested on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

    - In FRACTAL_NZ format, the weight shape supports 5 dimensions.
      - If transposeWeight is true, the dimensions are (g, k1, n1, n0, k0), where k0 = 32, n0 = 16, and k1 and k of x must meet the following relationship: ceilDiv(k, 32) = k1.
      - If transposeWeight is false, the dimensions are (g, n1, k1, k0, n0), where k0 = 16, n0 = 32, and k1 and k of x must meet the following relationship: ceilDiv(k, 16) = k1.

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
      <th>Return Values</th>
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
      <td>Examples include:<li>The checked tensor data type combination is unsupported, out and x have different data types, or the device is unsupported.</li><li>transposeWeight is false.</li><li>For non-empty inputs, weight or out is not contiguous.</li><li>The storageShape of weight does not have five dimensions.</li></td>
    </tr>
    <tr>
      <td rowspan="1">aclnnQuantGroupedMatmulDequantWeightNZGetWorkspaceSize failed</td>
      <td rowspan="1">561002</td>
      <td>If this error code is returned, check the error logs for the specific cause.</td>
    </tr>
  </tbody>
  </table>

## aclnnQuantGroupedMatmulDequantWeightNZ

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
      <td>Memory address of the workspace requested on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace requested on the Device, obtained by the first-phase API aclnnQuantGroupedMatmulDequantWeightNZGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, which contains the computation flow of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnQuantGroupedMatmulDequantWeightNZ defaults to a deterministic implementation.

- n and k must both be integer multiples of 16.
- If the data type of weightScale is INT64, the data type of xScaleOptional must be FLOAT16 and the value of xQuantMode must be pertensor; if the data type of xScaleOptional is FLOAT16, the data type of weightScale must be INT64 and the value of xQuantMode must be pertensor.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_grouped_matmul_dequant_weight_nz.h"

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
                    aclDataType dataType, aclFormat formatType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, formatType,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  int G = 4;
  int M = 64;
  int K = 256;
  int N = 512;

  char quantMode[16] = "pertoken";
  bool transposeWeight = true;

  std::vector<int64_t> xShape = {M, K};
  std::vector<int64_t> weightShape = {G, K / 32, N / 16, 16, 32};
  std::vector<int64_t> weightScaleShape = {G, N};
  std::vector<int64_t> xScaleShape = {M};
  std::vector<int64_t> smoothScaleShape = {K};
  std::vector<int64_t> outShape = {M, N};
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

  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_INT8, aclFormat::ACL_FORMAT_FRACTAL_NZ, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(weightScaleHostData, weightScaleShape, &weightScaleDeviceAddr, aclDataType::ACL_FLOAT, aclFormat::ACL_FORMAT_ND, &weightScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, aclFormat::ACL_FORMAT_ND, &groupList);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(xScaleHostData, xScaleShape, &xScaleDeviceAddr, aclDataType::ACL_FLOAT, aclFormat::ACL_FORMAT_ND, &xScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(smoothScaleHostData, smoothScaleShape, &smoothScaleDeviceAddr, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &smoothScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  ret = aclnnQuantGroupedMatmulDequantWeightNZGetWorkspaceSize(x, weight, weightScale, groupList,
                                                bias, xScale, xOffset, smoothScale,
                                                quantMode, transposeWeight, out,
                                                &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantGroupedMatmulDequantWeightNZGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnQuantGroupedMatmulDequantWeightNZ(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantGroupedMatmulDequantWeightNZ failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  PrintOutResult(outShape, &outDeviceAddr);

  aclDestroyTensor(x);
  aclDestroyTensor(weight);
  aclDestroyTensor(weightScale);
  aclDestroyTensor(groupList);
  aclDestroyTensor(xScale);
  aclDestroyTensor(smoothScale);
  aclDestroyTensor(out);

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
