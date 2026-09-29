# aclnnWeightQuantMatmulAllReduce

[📄 View Source Code](https://gitcode.com/cann/ops-transformer/tree/master/mc2/matmul_all_reduce)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    ×     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

**Note**: When using this API, ensure that the driver firmware package and CANN package are in the 8.0.RC2 version or later. Otherwise, an error, such as BUS ERROR, will be reported.

## Function Description

- **API function**: performs fake-quantization on the input x2 and then performs Matmul and AllReduce computation. The perTensor, perChannel, and perGroup quantization modes are supported.

- **Formula**:

  $$
  output = AllReduce(x1 @ ((x2 + antiquantOffset) * antiquantScale) + bias + x3)
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnWeightQuantMatmulAllReduceGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnWeightQuantMatmulAllReduce` is called to perform computation.

```cpp
aclnnStatus aclnnWeightQuantMatmulAllReduceGetWorkspaceSize(
    const aclTensor  *x1,
    const aclTensor  *x2,
    const aclTensor  *bias,
    const aclTensor  *antiquantScale,
    const aclTensor  *antiquantOffset,
    const aclTensor  *x3,
    const char       *group,
    const char       *reduceOp,
    int64_t          commTurn,
    int64_t          streamMode,
    int64_t          antiquantGroupSize,
    const aclTensor *output,
    uint64_t        *workspaceSize,
    aclOpExecutor **executor)
```

```cpp
aclnnStatus aclnnWeightQuantMatmulAllReduce(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnWeightQuantMatmulAllReduceGetWorkspaceSize

- **Parameters**
    <table style="undefined;table-layout: fixed; width: 1567px"><colgroup>
      <col style="width: 170px">
      <col style="width: 120px">
      <col style="width: 300px">
      <col style="width: 330px">
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
          <td>Left matrix of MatMul computation, that is, x1 in the formula.</td>
          <td><ul><li>The current version supports only two-dimensional or three-dimensional inputs. </li><li>The non-transpose scenario is supported.</li></ul></td>
          <td>BFLOAT16, FLOAT16</td>
          <td>See <a href="#constraints">Constraints</a>.</td>
          <td>2-3</td>
          <td>×</td>
        </tr>
        <tr>
          <td>x2</td>
          <td>Input</td>
          <td>Right matrix of MatMul computation, that is, x2 in the formula.</td>
          <td><ul><li>The current version supports only two-dimensional inputs. </li><li> The transpose and non-transpose scenarios are supported. </li><li>In ND format, only non-contiguous tensors with the last two axes transposed are supported.</li></ul></td>
          <td> See <a href="#constraints">Constraints</a>.</td>
          <td>ND, FRACTAL_NZ</td>
          <td>2</td>
          <td>×</td>
        </tr>
        <tr>
          <td>bias</td>
          <td>Input</td>
          <td>Bias in the calculation formula.</td>
          <td><ul><li>A null pointer can be passed. If the pointer is not null, only one-dimensional input is supported in the current version.</li></ul></td>
          <td> See <a href="#constraints">Constraints</a>.</td>
          <td>ND</td>
          <td>1</td>
          <td>√</td>
        </tr>
        <tr>
          <td>antiquantScale</td>
          <td>Input</td>
          <td> Corresponds to antiquantScale in the formula.</td>
          <td><ul><li>In the pertensor scenario, the shape is (1). </li><li>In the perchannel scenario, the shape is (n)/(1,n), where n is the size of the last dimension of x2. </li><li>In the pergroup scenario, the shape is (ceil(k,antiquantGroupSize),n).</li></ul></td>
          <td>BFLOAT16, FLOAT16</td>
          <td>ND</td>
          <td>1-2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>antiquantOffset</td>
          <td>Input</td>
          <td>Offset parameter for fake-quantization computation on <code>x2</code>, that is, <code>antiquantOffset</code> in the formula.</td>
          <td><ul><li>A null pointer can be passed. If the pointer is not null, the shape is the same as that of antiquantScale. </li><li>This parameter is not supported when the data format of x2 is FLOAT8_E4M3FN or HIFLOAT8. In this case, leave the pointer empty.</li></ul></td>
          <td>BFLOAT16, FLOAT16</td>
          <td>ND</td>
          <td>1-2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>x3</td>
          <td>Input</td>
          <td>Addition after MatMul computation, that is, <code>x3</code> in the formula.</td>
          <td><ul><li>A null pointer can be passed. If the pointer is not null, the shape is the same as that after the MM computation.</li></ul></td>
          <td>See <a href="#constraints">Constraints</a>.</td>
          <td>ND</td>
          <td>2-3</td>
          <td>√</td>
        </tr>
        <tr>
          <td>group</td>
          <td>Input</td>
          <td>Communication domain name.</td>
          <td><ul><li>The value can be obtained by calling the Hccl API extern HcclResult HcclGetCommName(HcclComm comm, char* commName);, where commName is the group.</li></ul></td>
          <td>String</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>reduceOp</td>
          <td>Input</td>
          <td>Reduce operation type.</td>
          <td><ul><li>In the current version, only "sum" is supported.</li></ul></td>
          <td>String</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>commTurn</td>
          <td>Input</td>
          <td>Number of communication data splits, that is, the total data volume divided by single communication volume.</td>
          <td><ul><li> In the current version, only 0 is supported.</li></ul></td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>streamMode</td>
          <td>Input</td>
          <td>Stream mode enumeration.</td>
          <td><ul><li> The current version supports only the enumerated value 1.</li></ul></td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>antiquantGroupSize</td>
          <td>Input</td>
          <td>Group size input for dequantizing x2 in fake-quantization pergroup mode.</td>
          This parameter is required in the <td><ul><li>pergroup quantization scenario. The value range is [32,min (k-1,INT_MAX)] and must be a multiple of 32. The value range of k is the same as that of [mm API], that is, [1,65535]. </li><li> In non-pergroup quantization scenarios, only 0 is supported.</li></ul></td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>output</td>
          <td>Output</td>
          <td>Result of MatMul computation and AllReduce communication, that is, the output in the computation formula.</td>
          <td><ul><li>The dimensions of the output are the same as those of x1.</li></ul></td>
          <td>-</td>
          <td>ND</td>
          <td>2-3</td>
          <td>√</td>
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
          <td>Operator executor, covering the operator computation process.</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
      </tbody>
    </table>

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 282px">
  <col style="width: 120px">
  <col style="width: 747px">
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
      <td>The passed x1, x2, antiquantScale, or output is a null pointer.</td>
  </tr>
  <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type of x1, x2, bias, antiquantScale, antiquantOffset, x3, or output does not meet the requirements.</td>
  </tr>
  <tr>
      <td>The values of reduceOp, streamMode, and antiquantGroupSize are invalid.</td>
  </tr>
  <tr>
      <td>The shape of x1, x2, bias, antiquantScale, antiquantOffset, x3, output, and antiquantGroupSize does not meet the constraint requirements.</td>
  </tr>
  </tbody>
  </table>

## aclnnWeightQuantMatmulAllReduce

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
        <td>Size of the workspace allocated on the device, which is obtained by the first API call <code>aclnnWeightQuantMatmulAllReduceGetWorkspaceSize</code>.</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Operator executor, covering the operator computation process.</td>
    </tr>
    <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream for executing the task.</td>
    </tr>
    </tbody></table>

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - <term>Atlas A2 training products/Atlas A2 inference products</term>: The aclnnWeightQuantMatmulAllReduce function is implemented in non-deterministic mode by default. You can enable deterministic computing by setting the HCCL_DETERMINISTIC environment variable to true.
  - Ascend 950PR/Ascend 950DT: The aclnnWeightQuantMatmulAllReduce function is implemented in deterministic mode by default.
- MC2 is disabled in incremental scenarios but enabled in full scenarios.
- The input `x1` can be two-dimensional or three-dimensional, with shape (b, s, k) or (m, k), respectively.
- `x2` must be 2D The shape is (k, n). The k axis meets the input parameter requirements of the MM operator, and the k axis is equal. The value range of m is [1, 2147483647], and the value range of k and n is [1, 65535].
- The passed x1, x2, antiquantScale, or output cannot be a null pointer.
- When the shape of the input `x1` is (b, s, k), the shape of `x3` (non-null scenario) and the shape of `output` are (b, s, n). When the shape of the input `x1` is (m, k), the shape of `x3` (non-null scenario) and the shape of `output` are (m, n).
- If **bias** is not empty, the shape size is the same as the last dimension of **output**. In the pertensor scenario, the shape of antiquantScale is (1). In the perchannel scenario, the shape of antiquantScale is (1, n)/(n). In the pergroup scenario, the shape of antiquantScale is (ceil(k,antiquantGroupSize), n). If antiquantOffset is not empty, its shape is the same as that of antiquantScale.
- The data types and data formats of x1, x2, x3 (non-empty), antiquantScale, antiquantOffset (non-empty), output, and bias (non-empty) must be supported.
- The data types of x1, antiquantScale, antiquantOffset (non-null scenario), x3 (non-null scenario), bias (non-null scenario), and output are the same. The value of antiquantGroupSize must be a multiple of 32 and within the value range.
- In the pergroup scenario, when x2 is transposed, antiquantScale and antiquantOffset must be transposed together to ensure continuity.
- In the long sequence scenario, as **b/s** or **m** increases, out of memory or computation timeout may occur.
- Only the all-mesh networking of HCCS links is supported.
  - <term>Atlas A2 training products/Atlas A2 inference products</term>: 1, 2, 4, and 8 ranks are supported.
  - Ascend 950PR/Ascend 950DT: 1, 2, 4, 8, 16, 32, or 64 cards are supported.
- <term>Atlas A2 training products/Atlas A2 inference products</term>:
  - The merged compute and communication (MC2) operators in a model support only the same communicator.
  - The data format of x2 can be ND (only 2D input is supported in the current version) or FRACTAL_NZ (only 4D input is supported in the current version). When the data format of x2 is FRACTAL_NZ, aclnnCalculateMatmulWeightSizeV2 and aclnnTransMatmulWeight are used together to convert the input from ND to NZ. Only the transpose scenario is supported for non-contiguous tensors.
- Ascend 950PR/Ascend 950DT:
  - The data format of x2 can be ND (only 2D input is supported). In the current version, if the data type is INT8, N and K must be 32-byte aligned. If the data type is INT4, N and K must be 64-byte aligned.
- Support for empty tensors:
  - Only the scenario where k is 0 is supported. The output is bias + x3. Empty tensor input with bs/m/n being 0 is not supported.

The following table describes the supported input and output data type combinations.

- <term>Atlas A2 training products/Atlas A2 inference products</term>:

  <table style="undefined;table-layout: fixed; width: 600px">
    <col style="width: 90px">
    <col style="width: 125px">
    <col style="width: 125px">
    <col style="width: 125px">
    <col style="width: 125px">
    <col style="width: 125px">
    <col style="width: 100px">
    <col style="width: 100px">
    <thead>
      <tr>
        <th>x1</th>
        <th>x2</th>
        <th>bias</th>
         <th>antiquantScale</th>
        <th>antiquantOffset</th>       
        <th>x3</th>
        <th>output</th>
        <th>Limit</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>BFLOAT16</td>
        <td>INT8, INT4</td>
        <td>null, BFLOAT16</td>
        <td>BFLOAT16</td>
        <td>null, BFLOAT16</td>
        <td>null, BFLOAT16</td>
        <td>BFLOAT16</td>
        <td>-</td>
      </tr>
      <tr>
        <td>FLOAT16</td>
        <td>INT8, INT4</td>
        <td>null, FLOAT16</td>
        <td>FLOAT16</td>
        <td>null, FLOAT16</td>
        <td>null, FLOAT16</td>
        <td>FLOAT16</td>
        <td>-</td>
      </tr>
    </tbody>
  </table>
  
- Ascend 950PR/Ascend 950DT
  <table style="undefined;table-layout: fixed; width: 600px">
    <col style="width: 90px">
    <col style="width: 125px">
    <col style="width: 125px">
    <col style="width: 125px">
    <col style="width: 125px">
    <col style="width: 125px">
    <col style="width: 100px">
    <col style="width: 100px">
    <thead>
      <tr>
        <th>x1</th>
        <th>x2</th>
        <th>bias</th>
        <th>antiquantScale</th>
        <th>antiquantOffset</th>
        <th>x3</th>
        <th>output</th>
        <th>Restriction</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>BFLOAT16</td>
        <td>INT8, INT4</td>
        <td>null, BFLOAT16</td>
        <td>BFLOAT16</td>
        <td>null, BFLOAT16</td>
        <td>null, BFLOAT16</td>
        <td>BFLOAT16</td>
        <td>Quantization in pertensor, perchannel, and pergroup modes is supported.</td>
      </tr>
      <tr>
        <td>BFLOAT16</td>
        <td>FLOAT8_E4M3FN, HIFLOAT8</td>
        <td>null, BFLOAT16</td>
        <td>BFLOAT16</td>
        <td>null, BFLOAT16</td>
        <td>null, BFLOAT16</td>
        <td>BFLOAT16</td>
        <td>Only per-channel quantization is supported.</td>
      </tr>
      <tr>
        <td>FLOAT16</td>
        <td>INT8, INT4</td>
        <td>null, FLOAT16</td>
        <td>FLOAT16</td>
        <td>null, FLOAT16</td>
        <td>null, FLOAT16</td>
        <td>FLOAT16</td>
        <td>Per-tensor, per-channel, and per-group quantization are supported.</td>
      </tr>
      <tr>
        <td>FLOAT16</td>
        <td>FLOAT8_E4M3FN, HIFLOAT8</td>
        <td>null, FLOAT16</td>
        <td>FLOAT16</td>
        <td>null, FLOAT16</td>
        <td>null, FLOAT16</td>
        <td>FLOAT16</td>
        <td>Only per-channel quantization is supported.</td>
      </tr>
    </tbody>
  </table>

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

Note: This sample code calls some HCCL collective communication library APIs, including HcclGetCommName, HcclCommInitAll, and HcclCommDestroy. For details, see [<<HCCL API (C)>>](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/hcclapiref/hcclcpp_07_0001.html).

- <term>Atlas A2 training products/Atlas A2 inference products</term> and Ascend 950 PR/Ascend 950 DT:

  ```Cpp
  #include <iostream>
  #include <vector>
  #include <thread>
  #include <string.h>
  #include "hccl/hccl.h"
  #include "aclnn/opdev/fp16_t.h"
  #include "aclnnop/aclnn_weight_quant_matmul_all_reduce.h"

  int ndev = 2;

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

  int64_t GetShapeSize(const std::vector<int64_t> &shape) {
      int64_t shapeSize = 1;
      for (auto i: shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  template<typename T>
  int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                      aclDataType dataType, aclTensor **tensor) {
      auto size = GetShapeSize(shape) * sizeof(T);
      // Call aclrtMalloc to allocate memory on the device.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy the data from the host to the device.
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

  struct Args {
      uint32_t rankId;
      HcclComm hcclComm;
      aclrtStream stream;
      aclrtContext context;
  };

  int launchOneThreadweightQuantmatmulAllReduce(Args &args) {
      int ret;
      ret = aclrtSetCurrentContext(args.context);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
      char hcom_name[128];
      ret = HcclGetCommName(args.hcclComm, hcom_name);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d \n", ret); return -1);
      LOG_PRINT("[INFO] rank %d hcom: %s stream: %p, context : %p\n", args.rankId, hcom_name, args.stream,
              args.context);

      std::vector<int64_t> x1Shape = {32, 64};
      std::vector<int64_t> x2Shape = {64, 128};
      std::vector<int64_t> biasShape = {128};
      std::vector<int64_t> antiquantScaleShape = {128};
      std::vector<int64_t> antiquantOffsetShape = {128};
      std::vector<int64_t> x3Shape = {32, 128};
      std::vector<int64_t> outShape = {32, 128};
      void *x1DeviceAddr = nullptr;
      void *x2DeviceAddr = nullptr;
      void *biasDeviceAddr = nullptr;
      void *antiquantScaleDeviceAddr = nullptr;
      void *antiquantOffsetDeviceAddr = nullptr;
      void *x3DeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;
      aclTensor *x1 = nullptr;
      aclTensor *x2 = nullptr;
      aclTensor *bias = nullptr;
      aclTensor *antiquantScale = nullptr;
      aclTensor *antiquantOffset = nullptr;
      aclTensor *x3 = nullptr;
      aclTensor *out = nullptr;

      int64_t commTurn = 0;
      int64_t streamMode = 1;
      int64_t antiquantGroupSize = 0;
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor;
      void *workspaceAddr = nullptr;

      long long x1ShapeSize = GetShapeSize(x1Shape);
      long long x2ShapeSize = GetShapeSize(x2Shape);
      long long biasShapeSize = GetShapeSize(biasShape);
      long long antiquantScaleShapeSize = GetShapeSize(antiquantScaleShape);
      long long antiquantOffsetShapeSize = GetShapeSize(antiquantOffsetShape);
      long long x3ShapeSize = GetShapeSize(x3Shape);
      long long outShapeSize = GetShapeSize(outShape);
      std::vector<op::fp16_t> x1HostData(x1ShapeSize, 1);
      std::vector<int8_t> x2HostData(x2ShapeSize, 1);
      std::vector<op::fp16_t> biasHostData(biasShapeSize, 1);
      std::vector<op::fp16_t> antiquantScaleHostData(antiquantScaleShapeSize, 1);
      std::vector<op::fp16_t> antiquantOffsetHostData(antiquantOffsetShapeSize, 1);
      std::vector<op::fp16_t> x3HostData(x3ShapeSize, 1);
      std::vector<op::fp16_t> outHostData(outShapeSize, 0);
      // Create a tensor.
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT16, &bias);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(antiquantScaleHostData, antiquantScaleShape, &antiquantScaleDeviceAddr,
                          aclDataType::ACL_FLOAT16, &antiquantScale);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(antiquantOffsetHostData, antiquantOffsetShape, &antiquantOffsetDeviceAddr,
                          aclDataType::ACL_FLOAT16, &antiquantOffset);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(x3HostData, x3Shape, &x3DeviceAddr, aclDataType::ACL_FLOAT16, &x3);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Call the first-phase API.
      ret = aclnnWeightQuantMatmulAllReduceGetWorkspaceSize(x1, x2, bias, antiquantScale, antiquantOffset, x3, hcom_name,
                                                          "sum", commTurn, streamMode, antiquantGroupSize, out,
                                                          &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnWeightQuantMatmulAllReduceGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API.
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // Call the second-phase API.
      ret = aclnnWeightQuantMatmulAllReduce(workspaceAddr, workspaceSize, executor, args.stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantMatmulAllReduce failed. ERROR: %d\n", ret); return     ret);
      // (Fixed writing) Synchronize the stream and wait for the task to complete.
      ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
      LOG_PRINT("device%d aclnnWeightQuantMatmulAllReduce execute success \n", args.rankId);
      // Release device resources. Modify the code based on the API definition.
      if (x1 != nullptr) {
          aclDestroyTensor(x1);
      }
      if (x2 != nullptr) {
          aclDestroyTensor(x2);
      }
      if (bias != nullptr) {
          aclDestroyTensor(bias);
      }
      if (antiquantScale != nullptr) {
          aclDestroyTensor(antiquantScale);
      }
      if (antiquantOffset != nullptr) {
          aclDestroyTensor(antiquantOffset);
      }
      if (x3 != nullptr) {
          aclDestroyTensor(x3);
      }
      if (out != nullptr) {
          aclDestroyTensor(out);
      }
      if (x1DeviceAddr != nullptr) {
          aclrtFree(x1DeviceAddr);
      }
      if (x2DeviceAddr != nullptr) {
          aclrtFree(x2DeviceAddr);
      }
      if (biasDeviceAddr != nullptr) {
          aclrtFree(biasDeviceAddr);
      }
      if (antiquantScaleDeviceAddr != nullptr) {
          aclrtFree(antiquantScaleDeviceAddr);
      }
      if (antiquantOffsetDeviceAddr != nullptr) {
          aclrtFree(antiquantOffsetDeviceAddr);
      }
      if (x3DeviceAddr != nullptr) {
          aclrtFree(x3DeviceAddr);
      }
      if (outDeviceAddr != nullptr) {
          aclrtFree(outDeviceAddr);
      }
      if (workspaceSize > 0) {
          aclrtFree(workspaceAddr);
      }
      aclrtDestroyStream(args.stream);
      HcclCommDestroy(args.hcclComm);
      aclrtDestroyContext(args.context);
      aclrtResetDevice(args.rankId);
      return 0;
  }

  int main(int argc, char *argv[]) {
      int ret;
      int32_t devices[ndev];
      for (int i = 0; i < ndev; i++) {
          devices[i] = i;
      }
      HcclComm comms[128];
      ret = aclInit(nullptr);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
      // Initialize the collective communicator.
      for (int i = 0; i < ndev; i++) {
          ret = aclrtSetDevice(devices[i]);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
      }
      ret = HcclCommInitAll(ndev, devices, comms);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("HcclCommInitAll failed. ERROR: %d\n", ret); return ret);
      Args args[ndev];
      aclrtStream stream[ndev];
      aclrtContext context[ndev];
      for (uint32_t rankId = 0; rankId < ndev; rankId++) {
          ret = aclrtSetDevice(rankId);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
          ret = aclrtCreateContext(&context[rankId], rankId);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
          ret = aclrtCreateStream(&stream[rankId]);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
      }
      // Start multiple threads.
      std::vector<std::unique_ptr<std::thread>> threads(ndev);
      for (uint32_t rankId = 0; rankId < ndev; rankId++) {
          args[rankId].rankId = rankId;
          args[rankId].hcclComm = comms[rankId];
          args[rankId].stream = stream[rankId];
          args[rankId].context = context[rankId];
          threads[rankId].reset(
                  new(std::nothrow) std::thread(&launchOneThreadweightQuantmatmulAllReduce, std::ref(args[rankId])));
      }
      for (uint32_t rankId = 0; rankId < ndev; rankId++) {
          threads[rankId]->join();
      }
      aclFinalize();
      return 0;
  }
  ```
