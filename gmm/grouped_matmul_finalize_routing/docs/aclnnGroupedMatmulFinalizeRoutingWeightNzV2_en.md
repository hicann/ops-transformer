# aclnnGroupedMatmulFinalizeRoutingWeightNzV2

## Supported Products

| Product                                                               | Supported|
|:------------------------------------------------------------------|:----:|
| <term>Atlas A3 training products/Atlas A3 inference products</term>                     |  √   |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|  √   |
| <term>Atlas 200I/500 A2 inference products</term>                              |  ×   |
| <term>Atlas inference products</term>                                       |  ×   |
| <term>Atlas training products</term>                                        |  ×   |

## Function

A fused operator of `GroupedMatmul` and `MoeFinalizeRouting`. It performs a "combine" operation on the output of the `GroupedMatmul` computation based on specified indices. `w` supports the AI processor-affinity format (NZ).
Compared with [aclnnGroupedMatmulFinalizeRoutingWeightNz](./aclnnGroupedMatmulFinalizeRoutingWeightNz_en.md), this API supports the INT4 weight matrix and introduces the input parameters `offsetOptional`, `antiquantScaleOptional`, `antiquantOffsetOptional`, and `tuningConfigOptional`. The first three parameters are reserved and do not take effect currently. Pass a null pointer for them. `tuningConfigOptional` is used for performance tuning. The first value in the array indicates the expected number of tokens to be processed by each expert. Operator tiling is performed based on this expected value to achieve higher performance. Select the appropriate API as required.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnGroupedMatmulFinalizeRoutingWeightNzV2` is called to perform computation.

```cpp
aclnnStatus aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize(
    const aclTensor   *x1,
    const aclTensor   *x2,
    const aclTensor   *scale,
    const aclTensor   *bias,
    const aclTensor   *offsetOptional,
    const aclTensor   *antiquantScaleOptional,
    const aclTensor   *antiquantOffsetOptional,
    const aclTensor   *pertokenScaleOptional,
    const aclTensor   *groupList,
    const aclTensor   *sharedInput,
    const aclTensor   *logit,
    const aclTensor   *rowIndex,
    int64_t            dtype,
    float              sharedInputWeight,
    int64_t            sharedInputOffset,
    bool               transposeX1,
    bool               transposeX2,
    int64_t            groupListType,
    const aclIntArray *tuningConfigOptional,
    aclTensor         *out,
    uint64_t          *workspaceSize,
    aclOpExecutor     **executor)
```

```cpp
aclnnStatus aclnnGroupedMatmulFinalizeRoutingWeightNzV2(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize

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
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage Notes</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x1</td>
      <td>Input</td>
      <td>Input x (left matrix)</td>
      <td>None</td>
      <td>INT8</td>
      <td>ND</td>
      <td>(m, k). The value range of dimension m is [1, 16*1024*8], and the value of k can be 2048.</td>
      <td>×</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Input weight (right matrix)</td>
      <td>None</td>
      <td>INT4, INT8</td>
      <td>NZ</td>
      <td>The shape can be 3D. When the input type is INT32, the shape is (e, k, n/8). When the input is converted to the INT4 type, the shape is (e, k, n). The value range of e is [1, 256], the value of k can be 2048, and the value of n can be 7168.</td>
      <td>×</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Input</td>
      <td>Scale factor for quantization parameters, used for per-channel quantization</td>
      <td>None</td>
      <td>INT64, FLOAT32, BF16</td>
      <td>ND</td>
      <td>The shape can be three-dimensional (e, 1, n). The values of e and n are the same as those of e and n for w.</td>
      <td>×</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>Input</td>
      <td>Matrix offset</td>
      <td>None</td>
      <td>BF16, FLOAT32</td>
      <td>ND</td>
      <td>The shape can be two-dimensional (e, n). The values of e and n are the same as those of e and n for w.</td>
      <td>×</td>
    </tr>
    <tr>
      <td>offsetOptional</td>
      <td>Input</td>
      <td>Asymmetric quantization offset</td>
      <td>None</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>The shape can be three-dimensional (e, 1, n). The values of e and n are the same as those of e and n for w.</td>
      <td>×</td>
    </tr>
    <tr>
      <td>antiquantScaleOptional</td>
      <td>Input</td>
      <td>Fake-quantization scale factor</td>
      <td>Not available currently</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td></td>
      <td>×</td>
    </tr>
    <tr>
      <td>antiquantOffsetOptional</td>
      <td>Input</td>
      <td>Fake-quantization offset</td>
      <td>Not available currently</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td></td>
      <td>×</td>
    </tr>
    <tr>
      <td>pertokenScaleOptional</td>
      <td>Input</td>
      <td>Dequantization parameter for matrix computation</td>
      <td></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>The shape can be one-dimensional (m). The value of m is the same as that of m for x.</td>
      <td>×</td>
    </tr>
    <tr>
      <td>groupList</td>
      <td>Input</td>
      <td>Matmul size distribution along the grouping axis for inputs and outputs</td>
      <td></td>
      <td>INT64</td>
      <td>ND</td>
      <td>The shape can be one-dimensional (e). The value of e is the same as that of e for w.</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sharedInput</td>
      <td>Input</td>
      <td>Output of the shared expert in MoE computation, which needs to be combined with the MoE experts' output</td>
      <td></td>
      <td>BF16</td>
      <td>ND</td>
      <td>The shape can be one-dimensional (e). The value of e is the same as that of e for w.</td>
      <td>×</td>
    </tr>
    <tr>
      <td>logit</td>
      <td>Input</td>
      <td>Logit size of each MoE expert for each token</td>
      <td></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>The shape can be one-dimensional (m). The value of m is the same as that of m for x.</td>
      <td>×</td>
    </tr>
    <tr>
      <td>rowIndex</td>
      <td>Input</td>
      <td>rowIndex used for combining MoE experts' output. Its values serve as the indices for the scatter-add operation during the combination process.</td>
      <td></td>
      <td>INT64, INT32</td>
      <td>ND</td>
      <td>The shape can be one-dimensional (m). The value of m is the same as that of m for x.</td>
      <td>×</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>Input</td>
      <td>Computation output type. 0: FLOAT32; 1: FLOAT16; 2: BFLOAT16 Currently, only 0 is supported.</td>
      <td></td>
      <td>INT64</td>
      <td></td>
      <td></td>
      <td>×</td>
    </tr>
    <tr>
      <td>sharedInputWeight</td>
      <td>Input</td>
      <td>Combination coefficient. sharedInput is multiplied by this parameter first, and then the result is accumulated with the MoE experts' results.</td>
      <td></td>
      <td>FLOAT32</td>
      <td></td>
      <td></td>
      <td>×</td>
    </tr>
    <tr>
      <td>sharedInputOffset</td>
      <td>Input</td>
      <td>Offset of the shared expert output in the total output.</td>
      <td></td>
      <td>INT64</td>
      <td></td>
      <td></td>
      <td>×</td>
    </tr>
    <tr>
      <td>transposeX</td>
      <td>Input</td>
      <td>Whether to transpose the left matrix. The value can only be false.</td>
      <td></td>
      <td>BOOL</td>
      <td></td>
      <td></td>
      <td>×</td>
    </tr>
    <tr>
      <td>transposeX2</td>
      <td>Input</td>
      <td>Whether to transpose the right matrix. The value can only be false.</td>
      <td></td>
      <td>BOOL</td>
      <td></td>
      <td></td>
      <td>×</td>
    </tr>
    <tr>
      <td>groupListType</td>
      <td>Input</td>
      <td>Grouping mode. 0: cumsum mode (prefix sum); 1: count mode</td>
      <td></td>
      <td>INT64</td>
      <td></td>
      <td></td>
      <td>×</td>
    </tr>
    <tr>
      <td>tuningConfigOptional</td>
      <td>Input</td>
      <td>The first element in the array indicates the expected number of tokens to be processed by each expert. Operator tiling is performed based on the first element to achieve higher performance. If the second element in the array is set to 1, the operator will attempt to use a more suitable algorithm based on the actual input during tiling. When k <= 2048, the performance may be better. The third and subsequent elements are reserved. You do not need to set them. It will be extended in the future. It is compatible with earlier versions. If this parameter is not used, do not pass it (that is, pass nullptr).</td>
      <td></td>
      <td>INT64</td>
      <td></td>
      <td></td>
      <td>×</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output result.</td>
      <td>Its shape is the same as that of self.</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(batch, n)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Size of the workspace required to be allocated on the device.</td>
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

- **Return**

  `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
  
  The first-phase API implements input parameter validation. The following errors may be thrown:
  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 700px">
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
      <td>The passed x1, x2, scale, bias, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>The data type or data format of x1, x2, scale, bias, offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, pertokenScaleOptional, groupList, sharedInputOptional, logit, rowIndex, sharedInputWeight, sharedInputOffset, transposeX, transposeX2, or out is not supported.</td>
    </tr>
    <tr>
      <td>The shape of x1, x2, scale, bias, offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, pertokenScaleOptional, groupList, sharedInputOptional, logit, rowIndex, or out does not meet the validation conditions.</td>
    </tr>
    <tr>
      <td>x1, x2, scale, bias, offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, pertokenScaleOptional, groupList, sharedInputOptional, logit, rowIndex, or out is an empty tensor.</td>
    </tr>
    <tr>
      <td>x1, x2, scale, bias, offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, pertokenScaleOptional, groupList, sharedInputOptional, logit, rowIndex, or out is an empty tensor.</td>
    </tr>
  </tbody></table>

## aclnnGroupedMatmulFinalizeRoutingWeightNzV2

- **Parameters**
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
        <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize.</td>
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

- **Return**

  `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnGroupedMatmulFinalizeRoutingWeightNzV2` defaults to non-deterministic implementation. You can call `aclrtCtxSetSysParamOpt` to enable deterministic computation.

- The input and output support the following data type combinations.

  | x1    | x2    | scale   | bias    | offsetOptional  | antiquantScaleOptional | antiquantOffsetOptional | pertokenScaleOptional| groupList | sharedInput | logit   | rowIndex | out   | tuningConfigOptional |
  |------|------|---------|---------|---------|----------------|-----------------|---------------|-----------|-------------|---------|----------|-------|----------------------|
  | INT8 | INT8 | FLOAT32 | null    | null    | null           | null            | FLOAT32       | INT64     | BFLOAT16    | FLOAT32 | INT64    | FLOAT | IntArray             |
  | INT8 | INT8 | FLOAT32 | null    | null    | null           | null            | FLOAT32       | INT64     | BFLOAT16    | FLOAT32 | INT64    | FLOAT | IntArray             |
  | INT8 | INT4 | INT64   | FLOAT32 | FLOAT32 | null           | null            | FLOAT32       | INT64     | BFLOAT16    | FLOAT32 | INT64    | FLOAT | IntArray             |
  | INT8 | INT4 | INT64   | FLOAT32 | null    | null           | null            | FLOAT32       | INT64     | BFLOAT16    | FLOAT32 | INT64    | FLOAT | IntArray             |

## Calling Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_permute.h"
  #include "aclnnop/aclnn_grouped_matmul_finalize_routing_weight_nz_v2.h"
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
      // (Boilerplate) Initialize resources.
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
      // Call aclrtMemcpy to copy host data to the device memory.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Compute the strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      std::vector<int64_t> storageShape;
      storageShape.push_back(GetShapeSize(shape));

      // Call aclCreateTensor to create an aclTensor.
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                storageShape.data(), storageShape.size(), *deviceAddr);
      return 0;
  }

    int main() {
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init stream failed. ERROR: %d\n", ret); return ret);

      // 2. Construct inputs and outputs based on API definitions.
      int64_t m = 192;
      int64_t k = 2048;
      int64_t n = 7168;
      int64_t e = 4;
      int64_t batch = 24;
      int64_t bsdp = 8;
      int64_t dtype = 0;
      float shareInputWeight = 1.0;
      int64_t shareInputOffset = 0;
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
      std::vector<int64_t> tuningConfigVal = {1}; 

      void *xDeviceAddr = nullptr;
      void *wDeviceAddr = nullptr;
      void *biasDeviceAddr = nullptr;
      void *scaleDeviceAddr = nullptr;
      void *pertokenScaleDeviceAddr = nullptr;
      void *groupListDeviceAddr = nullptr;
      void *sharedInputDeviceAddr = nullptr;
      void *logitDeviceAddr = nullptr;
      void *rowIndexDeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;
      void *tuningConfigDeviceAddr = nullptr;

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
      groupListHostData[0] = 32;
      groupListHostData[0] = 40;
      groupListHostData[0] = 64;

      std::vector<uint16_t> sharedInputHostData(GetShapeSize(sharedInputShape));
      std::vector<int64_t> logitHostData(GetShapeSize(logitShape));
      std::vector<float> rowIndexHostData(GetShapeSize(rowIndexShape));
      std::vector<float> outHostData(GetShapeSize(outShape));  // Half-precision FLOAT16

      // Create an x aclTensor.
      ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_INT8, &x);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> xTensorPtr(x, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a w aclTensor in AI processor-affinity format.
      ret = CreateAclTensorWeight(wHostData, wShape, &wDeviceAddr, aclDataType::ACL_INT8, &w);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> wTensorPtr(w, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> wDeviceAddrPtr(wDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a scale aclTensor.
      ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a pertokenScale aclTensor.
      ret = CreateAclTensor(pertokenScaleHostData, pertokenScaleShape, &pertokenScaleDeviceAddr, aclDataType::ACL_FLOAT, &pertokenScale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> pertokenScaleTensorPtr(pertokenScale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> pertokenScaleDeviceAddrPtr(pertokenScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a groupList aclTensor.
      ret = CreateAclTensor(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupList);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> groupListTensorPtr(groupList, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> groupListDeviceAddrPtr(groupListDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a sharedInput aclTensor.
      ret = CreateAclTensor(sharedInputHostData, sharedInputShape, &sharedInputDeviceAddr, aclDataType::ACL_BF16, &sharedInput);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> sharedInputTensorPtr(sharedInput, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> sharedInputDeviceAddrPtr(sharedInputDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a logit aclTensor.
      ret = CreateAclTensor(logitHostData, logitShape, &logitDeviceAddr, aclDataType::ACL_FLOAT, &logit);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> logitTensorPtr(logit, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> logitDeviceAddrPtr(logitDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a rowIndex aclTensor.
      ret = CreateAclTensor(rowIndexHostData, rowIndexShape, &rowIndexDeviceAddr, aclDataType::ACL_INT64, &rowIndex);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> rowIndexTensorPtr(rowIndex, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> rowIndexDeviceAddrPtr(rowIndexDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an out aclTensor.
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a tuningConfig aclIntArray.
      aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
      std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> tuningConfigIntArrayPtr(tuningConfig, aclDestroyIntArray);
      std::unique_ptr<void, aclError (*)(void *)> tuningConfigDeviceAddrPtr(tuningConfigDeviceAddr, aclrtFree);
      CHECK_RET(tuningConfig == nullptr, -1);
      // 3. Call the CANN operator library API. Change the API name to the actual one.
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor;
      void *workspaceAddr = nullptr;

      // Call the first-phase API of aclnnTransMatmulWeight.
      ret = aclnnTransMatmulWeightGetWorkspaceSize(w, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeightGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // Call the second-phase API of aclnnTransMatmulWeight.
      ret = aclnnTransMatmulWeight(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeight failed. ERROR: %d\n", ret); return ret);

      // Call the first-phase API of aclnnGroupedMatmulFinalizeRoutingWeightNzV2.
      workspaceSize = 0;                                               
      ret = aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize(x, w, scale, nullptr, nullptr, nullptr, nullptr, pertokenScale, groupList, sharedInput, logit, rowIndex, dtype, shareInputWeight, shareInputOffset, transposeX, transposeW, groupListType, tuningConfig, out, &workspaceSize, &executor);

      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.

      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // Call the second-phase API of aclnnGroupedMatmulFinalizeRoutingWeightNzV2.
      ret = aclnnGroupedMatmulFinalizeRoutingWeightNzV2(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulFinalizeRoutingWeightNzV2 failed. ERROR: %d\n", ret); return ret);

      // 4. (Boilerplate) Wait until the task execution is complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
      auto size = GetShapeSize(outShape);
      std::vector<uint16_t> resultData(
          size, 0);  // fp16 data cannot be printed directly in C. It must be read as uint16 and then manually converted to fp16 via binary representation.
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %u\n", i, resultData[i]);
      }

      // 6. Free aclTensor resources. Modify the code based on the API definition.
      aclDestroyTensor(x);
      aclDestroyTensor(w);
      aclDestroyTensor(scale);
      aclDestroyTensor(pertokenScale);
      aclDestroyTensor(groupList);
      aclDestroyTensor(sharedInput);
      aclDestroyTensor(logit);
      aclDestroyTensor(rowIndex);
      aclDestroyTensor(out);

      // 7. Free device resources. Modify the code based on the API definition.
      aclrtFree(xDeviceAddr);
      aclrtFree(wDeviceAddr);
      aclrtFree(scaleDeviceAddr);
      aclrtFree(pertokenScaleDeviceAddr);
      aclrtFree(groupListDeviceAddr);
      aclrtFree(sharedInputDeviceAddr);
      aclrtFree(logitDeviceAddr);
      aclrtFree(rowIndexDeviceAddr);
      aclrtFree(outDeviceAddr);
      aclDestroyIntArray(tuningConfig);

      if (workspaceSize > 0) {
          aclrtFree(workspaceAddr);
      }
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
      return 0;
  }
  ```
