# aclnnGroupedMatmulFinalizeRoutingWeightNz

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T12:27:16.375Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Not supported
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

A fused operator of GroupedMatmul and MoeFinalizeRouting, where the output after GroupedMatmul computation is combined according to indices. Supports w in the AI processor-friendly data layout format (NZ).

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnGroupedMatmulFinalizeRoutingWeightNzGetWorkspaceSize" first to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call "aclnnGroupedMatmulFinalizeRoutingWeightNz" to perform the computation.

```cpp
aclnnStatus aclnnGroupedMatmulFinalizeRoutingWeightNzGetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *scale,
    const aclTensor *bias,
    const aclTensor *pertokenScaleOptional,
    const aclTensor *groupList,
    const aclTensor *sharedInput,
    const aclTensor *logit,
    const aclTensor *rowIndex,
    int64_t          dtype,
    float            sharedInputWeight,
    int64_t          sharedInputOffset,
    bool             transposeX1,
    bool             transposeX2,
    int64_t          groupListType,
    aclTensor       *out,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnGroupedMatmulFinalizeRoutingWeightNz(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnGroupedMatmulFinalizeRoutingWeightNzGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1494px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 400px">
  <col style="width: 230px">
  <col style="width: 212px">
  <col style="width: 100px">
  <col style="width: 190px">
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
      <td>x1</td>
      <td>Input</td>
      <td>Input x (left matrix).</td>
      <td>-</td>
      <td>INT8</td>
      <td>ND</td>
      <td>The shape supports two dimensions, and the dimension is (m, k). The data type supports INT8. The value range of dimension m is [1,16\*1024\*8]; k supports 256, 512, 1024, 1408, 2048.</td>
      <td>-</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Input weight (right matrix).</td>
      <td>-</td>
      <td>INT8</td>
      <td>NZ</td>
      <td>The shape supports 5 dimensions. The dimension is (e, n1, k1, k0, n0), where k0 = 16 and n0 = 32. The k in the x1 shape and the k1 in the x2 shape must meet the following relationship: ceilDiv(k,16) = k1. The aclnnCalculateMatmulWeightSizeV2 and aclnnTransMatmulWeight APIs can be used to convert the input format from ND to the AI processor-friendly data layout (NZ). The value range of e is [1, 256].</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Input</td>
      <td>Scale factor in quantization parameters, per-channel quantization parameter.</td>
      <td>-</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>The shape is two-dimensional (e, n), where n = n1 \* n0, e is consistent with the e of w, and n supports 2048, 7168, and 7680.</td>
      <td>-</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>Input</td>
      <td>Bias of the matrix. This parameter is currently reserved and does not take effect. Pass a null pointer.</td>
      <td>-</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <td>pertokenScaleOptional</td>
      <td>Input</td>
      <td>Dequantization parameter for matrix computation.</td>
      <td></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>Shape supports one-dimensional, the dimension is (m), and m is consistent with the m of x1.</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupList</td>
      <td>Input</td>
      <td>Distribution of matmul sizes along the grouping axis of input and output.</td>
      <td></td>
      <td>INT64</td>
      <td>ND</td>
      <td>The shape supports one-dimensional, and the dimension is (e). e is consistent with the e of x2.</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sharedInput</td>
      <td>Input</td>
      <td>Output of the shared expert in the moe computation, which needs to be combined with the output of moe experts.</td>
      <td></td>
      <td>BF16</td>
      <td>ND</td>
      <td>Supports two dimensions, the dimensions are (bsdp,n), bsdp must be less than or equal to batchSize/e, and n is consistent with the n of x2.</td>
      <td>-</td>
    </tr>
    <tr>
      <td>logit</td>
      <td>Input</td>
      <td>Logit size of each token for MoE experts.</td>
      <td></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>Shape supports one-dimensional, the dimension is (m), and m is consistent with the m of x1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rowIndex</td>
      <td>Input</td>
      <td>MoE expert outputs are combined based on this rowIndex, and the value in it is the index for scatter add in combine.</td>
      <td></td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>shape supports one-dimensional, and the dimension is (m), consistent with the m of x1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>Input</td>
      <td>Output data type of the computation: 0: FLOAT32; 1: FLOAT16; 2: BFLOAT16. Currently, only 0 is supported.</td>
      <td></td>
      <td>INT64</td>
      <td></td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <td>sharedInputWeight</td>
      <td>Input</td>
      <td>Coefficient used when the shared expert is combined with the MoE experts. sharedInput is first multiplied by this parameter and then accumulated with the MoE expert result.</td>
      <td></td>
      <td>FLOAT32</td>
      <td></td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <td>sharedInputOffset</td>
      <td>Input</td>
      <td>Offset of the shared expert output in the total output.</td>
      <td></td>
      <td>INT64</td>
      <td></td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <td>transposeX1</td>
      <td>Input</td>
      <td>Whether the left matrix is transposed. Only false is supported.</td>
      <td></td>
      <td>BOOL</td>
      <td></td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <td>transposeX2</td>
      <td>Input</td>
      <td>Whether the right matrix is transposed. Only false is supported.</td>
      <td></td>
      <td>BOOL</td>
      <td></td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupListType</td>
      <td>Input</td>
      <td>Grouping mode. If set to 0, the cumsum mode (prefix sum) is used; if set to 1, the count mode is used.</td>
      <td></td>
      <td>INT64</td>
      <td></td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output result.</td>
      <td>The shape is the same as that of self.</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>0-8</td>
      <td>-</td>
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
      <td>Returns the op executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 700px">
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
      <td>x1, x2, scale, bias, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data type and data format of x1, x2, scale, bias, pertokenScaleOptional, groupList, sharedInput, logit, rowIndex, sharedInputWeight, sharedInputOffset, transposeX1, transposeX2, or out are not in the supported range.</td>
    </tr>
    <tr>
      <td>The shape of x1, x2, scale, bias, pertokenScaleOptional, groupList, sharedInput, logit, rowIndex, or out does not meet the check requirements.</td>
    </tr>
    <tr>
      <td>x1, x2, scale, bias, pertokenScaleOptional, groupList, sharedInput, logit, rowIndex, or out is an empty tensor.</td>
    </tr>
    <tr>
      <td>The value ranges of dtype, sharedInputOffset, transposeX1, transposeX2, and groupListType do not meet the requirements.</td>
    </tr>
  </tbody></table>

## aclnnGroupedMatmulFinalizeRoutingWeightNz

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
        <td>Workspace memory address allocated on the Device.</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Workspace size allocated on the Device, obtained by calling the first-phase API aclnnGroupedMatmulFinalizeRoutingWeightNzGetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>op executor, which contains the operator computation flow.</td>
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

- deterministic computation:

  - aclnnGroupedMatmulFinalizeRoutingWeightNz defaults to a non-deterministic implementation. Deterministic computation can be enabled through aclrtSetSysParamOpt.
- The input and output support the following data type combinations:

  | x1   | x2   | scale   | bias | pertokenScaleOptional | groupList | sharedInput | logit   | rowIndex | out   |
  | ---- | ---- | ------- | ---- | --------------------- | --------- | ----------- | ------- | -------- | ----- |
  | INT8 | INT8 | FLOAT32 | null | FLOAT32               | INT64     | BFLOAT16    | FLOAT32 | INT64    | FLOAT |
  | INT8 | INT8 | FLOAT32 | null | FLOAT32               | INT64     | null        | null    | INT64    | FLOAT |

## Example

The following sample code is for reference only. For details about the compilation and execution procedure, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_permute.h"
  #include "aclnnop/aclnn_grouped_matmul_finalize_routing_weight_nz.h"
  #include "aclnnop/aclnn_trans_matmul_weight.h"

  #define CHECK_RET(cond, return_expr) \
      do {                             \
          if (!(cond)) {               \
              return_expr;             \
          }                            \
      } while (0)

  #define CHECK_FREE_RET(cond, return_expr) \
      do {                                  \
          if (!(cond)) {                    \
              Finalize(deviceId, stream);   \
              return_expr;                  \
          }                                 \
      } while (0)

  #define LOG_PRINT(message, ...)         \
      do {                                \
          printf(message, ##__VA_ARGS__); \
      } while (0)

  int64_t GetShapeSize(const std::vector<int64_t> &shape)
  {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  int Init(int32_t deviceId, aclrtStream *stream)
  {
      // Boilerplate code for resource initialization.
      auto ret = aclInit(nullptr);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
      ret = aclrtSetDevice(deviceId);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
      ret = aclrtCreateStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
      return 0;
  }

  template <typename T>
  int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                      aclDataType dataType, aclTensor **tensor)
  {
      auto size = GetShapeSize(shape) * sizeof(T);
      // Call aclrtMalloc to allocate device memory.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy host data to device memory.
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

  template <typename T>
  int CreateAclTensorWeight(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                        aclDataType dataType, aclTensor **tensor)
  {
      auto size = static_cast<uint64_t>(GetShapeSize(shape));

      const aclIntArray *mat2Size = aclCreateIntArray(shape.data(), shape.size());
      auto ret = aclnnCalculateMatmulWeightSizeV2(mat2Size, dataType, &size);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCalculateMatmulWeightSizeV2 failed. ERROR: %d\n", ret);
                return ret);
      size *= sizeof(T);

      // Call aclrtMalloc to allocate device memory.
      ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy host data to device memory.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Calculate the strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      std::vector<int64_t> storageShape;
      storageShape.push_back(GetShapeSize(shape));

      // Call the aclCreateTensor API to create an aclTensor.
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                storageShape.data(), storageShape.size(), *deviceAddr);
      return 0;
  }

    int main() {
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init stream failed. ERROR: %d\n", ret); return ret);

      // 2. Construct inputs and outputs. Customize the construction based on the API.
      int64_t m = 192;
      int64_t k = 2048;
      int64_t n = 7168;
      int64_t e = 4;
      int64_t batch = 24;
      int64_t bsdp = 8;
      int64_t dtype = 0;
      float sharedInputWeight = 1.0;
      int64_t sharedInputOffset = 0;
      bool transposeX = false;
      bool transposeW = false;
      int64_t groupListType = 1;

      std::vector<int64_t> xShape = {m, k};
      std::vector<int64_t> wShape = {e, k, n};
      std::vector<int64_t> scaleShape = {e, n};
      std::vector<int64_t> pertokenScaleShape = {m};
      std::vector<int64_t> groupListShape = {e};
      std::vector<int64_t> sharedInputShape = {bsdp, n};
      std::vector<int64_t> logitShape = {m};
      std::vector<int64_t> rowIndexShape = {m};
      std::vector<int64_t> outShape = {batch, n};

      void *xDeviceAddr = nullptr;
      void *wDeviceAddr = nullptr;
      void *scaleDeviceAddr = nullptr;
      void *pertokenScaleDeviceAddr = nullptr;
      void *groupListDeviceAddr = nullptr;
      void *sharedInputDeviceAddr = nullptr;
      void *logitDeviceAddr = nullptr;
      void *rowIndexDeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;

      aclTensor* x = nullptr;
      aclTensor* w = nullptr;
      aclTensor* bias = nullptr;
      aclTensor* groupList = nullptr;
      aclTensor* scale = nullptr;
      aclTensor* pertokenScale = nullptr;
      aclTensor* sharedInput = nullptr;
      aclTensor* logit = nullptr;
      aclTensor* rowIndex = nullptr;
      aclTensor* out = nullptr;

      std::vector<int8_t> xHostData(GetShapeSize(xShape));
      std::vector<int8_t> wHostData(GetShapeSize(wShape));
      std::vector<float> scaleHostData(GetShapeSize(scaleShape));
      std::vector<float> pertokenScaleHostData(GetShapeSize(pertokenScaleShape));
      std::vector<int64_t> groupListHostData(GetShapeSize(groupListShape));
      groupListHostData[0] = 7;
      groupListHostData[1] = 32;
      groupListHostData[2] = 40;
      groupListHostData[3] = 64;

      std::vector<uint16_t> sharedInputHostData(GetShapeSize(sharedInputShape));
      std::vector<float> logitHostData(GetShapeSize(logitShape));
      std::vector<int64_t> rowIndexHostData(GetShapeSize(rowIndexShape));
      std::vector<float> outHostData(GetShapeSize(outShape));

      // Create the x aclTensor.
      ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_INT8, &x);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> xTensorPtr(x, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the w aclTensor with the AI processor-friendly data layout.
      ret = CreateAclTensorWeight(wHostData, wShape, &wDeviceAddr, aclDataType::ACL_INT8, &w);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> wTensorPtr(w, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> wDeviceAddrPtr(wDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the scale aclTensor.
      ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the pertokenScale aclTensor.
      ret = CreateAclTensor(pertokenScaleHostData, pertokenScaleShape, &pertokenScaleDeviceAddr, aclDataType::ACL_FLOAT, &pertokenScale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> pertokenScaleTensorPtr(pertokenScale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> pertokenScaleDeviceAddrPtr(pertokenScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the groupList aclTensor.
      ret = CreateAclTensor(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupList);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> groupListTensorPtr(groupList, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> groupListDeviceAddrPtr(groupListDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the sharedInput aclTensor.
      ret = CreateAclTensor(sharedInputHostData, sharedInputShape, &sharedInputDeviceAddr, aclDataType::ACL_BF16, &sharedInput);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> sharedInputTensorPtr(sharedInput, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> sharedInputDeviceAddrPtr(sharedInputDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the logit aclTensor.
      ret = CreateAclTensor(logitHostData, logitShape, &logitDeviceAddr, aclDataType::ACL_FLOAT, &logit);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> logitTensorPtr(logit, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> logitDeviceAddrPtr(logitDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the rowIndex aclTensor.
      ret = CreateAclTensor(rowIndexHostData, rowIndexShape, &rowIndexDeviceAddr, aclDataType::ACL_INT64, &rowIndex);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> rowIndexTensorPtr(rowIndex, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> rowIndexDeviceAddrPtr(rowIndexDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the out aclTensor.
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3. Call the CANN operator library API. Replace Api with the actual API name.
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor;
      void *workspaceAddr = nullptr;

      // x2 must be converted from the ND format to the NZ format required by the operator. Therefore, call aclnnTransMatmulWeight first to complete the weight format conversion.
      // Call the aclnnTransMatmulWeight first-phase API
      ret = aclnnTransMatmulWeightGetWorkspaceSize(w, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeightGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Request device memory based on the workspaceSize calculated by the first-phase API
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // Call the aclnnTransMatmulWeight second-phase API
      ret = aclnnTransMatmulWeight(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeight failed. ERROR: %d\n", ret); return ret);

      // Call the aclnnGroupedMatmulFinalizeRoutingWeightNz first-phase API
      workspaceSize = 0;
      ret = aclnnGroupedMatmulFinalizeRoutingWeightNzGetWorkspaceSize(x, w, scale, nullptr, pertokenScale, groupList, sharedInput, logit, rowIndex, dtype, sharedInputWeight, sharedInputOffset, transposeX, transposeW, groupListType, out, &workspaceSize, &executor);

      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulFinalizeRoutingWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Release the workspace requested during the weight format conversion phase to prevent the second allocation from overwriting the pointer and causing a memory leak.
      if (workspaceAddr != nullptr) {
          aclrtFree(workspaceAddr);
          workspaceAddr = nullptr;
      }
      // Request device memory based on the workspaceSize calculated by the first-phase API

      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // Call the aclnnGroupedMatmulFinalizeRoutingWeightNz second-phase API
      ret = aclnnGroupedMatmulFinalizeRoutingWeightNz(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulFinalizeRoutingWeightNz failed. ERROR: %d\n", ret); return ret);

      // 4. (Boilerplate) Synchronize and wait for the task execution to complete
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value, copy the result from device memory to host memory, and modify this step based on the specific API definition
      auto size = GetShapeSize(outShape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%lld] is: %f\n", i, resultData[i]);
      }

      // 6. Release the aclTensor resources. Modify them according to the specific API definition.
      aclDestroyTensor(x);
      aclDestroyTensor(w);
      aclDestroyTensor(scale);
      aclDestroyTensor(pertokenScale);
      aclDestroyTensor(groupList);
      aclDestroyTensor(sharedInput);
      aclDestroyTensor(logit);
      aclDestroyTensor(rowIndex);
      aclDestroyTensor(out);

      // 7.Release the device resources. Modify them according to the specific API definition.
      aclrtFree(xDeviceAddr);
      aclrtFree(wDeviceAddr);
      aclrtFree(scaleDeviceAddr);
      aclrtFree(pertokenScaleDeviceAddr);
      aclrtFree(groupListDeviceAddr);
      aclrtFree(sharedInputDeviceAddr);
      aclrtFree(logitDeviceAddr);
      aclrtFree(rowIndexDeviceAddr);
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
