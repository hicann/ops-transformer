# aclnnQuantMatmulAllReduceV2

## Supported Products

| Product                                                        |  Supported  |
| :----------------------------------------------------------- |:-------:|
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    ×    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference products</term>                            |    ×    |
| <term>Atlas training products</term>                             |    ×    |

**Note**: When using this API, ensure that the driver firmware package and CANN package are in the 8.0.RC2 version or later. Otherwise, an error, such as BUS ERROR, will be reported.

## Description

- **API function**: Extends the functions of the `aclnnQuantMatmulAllReduce` API and supports the per-tensor quantization mode. It supports per-token, per-channel, and per-token [quantization methods](../../../docs/en/context/quant_mode_introduction.md).

- **Formula**:

    This API handles the following scenarios:

    - Scenario 1: Performs MatMul computation on quantized input parameters `x1` and `x2`, followed by dequantization, then performs addition with `x3`, and finally executes AllReduce computation.

  $$
  output= AllReduce(dequantScale*(x1_{int8}@x2_{int8} + bias_{int32}) + x3)
  $$

    - Scenario 2: Performs MatMul computation on quantized input parameters `x1` and `x2`, followed by dequantization and per-token scaling, then performs addition with `x3`, and finally executes AllReduce computation.

  $$
  output= AllReduce(dequantScale * pertokenScaleOptional * (x1_{int8}@x2_{int8} + biasOptional_{int32}) + x3Optional)
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnQuantMatmulAllReduceV2GetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnQuantMatmulAllReduceV2` is called to perform computation.

```cpp
aclnnStatus aclnnQuantMatmulAllReduceV2GetWorkspaceSize(
    const aclTensor  *x1,
    const aclTensor  *x2,
    const aclTensor  *biasOptional,
    const aclTensor  *x3Optional,
    const aclTensor  *dequantScale,
    const aclTensor  *pertokenScaleOptional,
    const char       *group,
    const char       *reduceOp,
    int64_t           commTurn,
    int64_t           streamMode,
    const aclTensor  *output,
    uint64_t         *workspaceSize,
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnQuantMatmulAllReduceV2(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream  stream)
```

## aclnnQuantMatmulAllReduceV2GetWorkspaceSize

- **Parameters:**
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
          <th>Precaution</th>
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
          <td><ul><li>The current version supports only 2D or 3D inputs. </li><li>The non-transpose scenario is supported.</li></ul></td>
          <td>INT8</td>
          <td>ND</td>
          <td>2-3</td>
          <td>×</td>
        </tr>
        <tr>
          <td>x2</td>
          <td>Input</td>
          <td>Right matrix of MatMul computation, that is, x2 in the formula.</td>
          <td><ul><li>The current version supports only 2D inputs. </li><li> The transpose and non-transpose scenarios are supported.</li></ul></td>
          <td>INT8</td>
          <td>ND, FRACTAL_NZ</td>
          <td>2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>biasOptional</td>
          <td>Input</td>
          <td> biasOptional in the formula.</td>
          <td>The current version supports only 1D inputs.</td>
          <td>INT32</td>
          <td>ND</td>
          <td>0-1</td>
          <td>√</td>
        </tr>
        <tr>
          <td>x3Optional</td>
          <td>Input</td>
          <td>Addition after MatMul computation, that is, x3Optional in the formula.</td>
          <td>The shape is the same as that after MatMul computation.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>dequantScale</td>
          <td>Input</td>
          <td>Dequantization coefficient after MatMul computation, that is, dequantScale in the formula.</td>
          <td><ul><li>The shape is (1) in the per-tensor scenario and (n) or (1, n) in the per-channel scenario.</li><li>When the output is of BFLOAT16 type, pass dequantScale of BFLOAT16 type directly to this API. </li><li>When the output is of FLOAT16 type, if pertokenScaleOptional is not empty, you can pass dequantScale of FLOAT32 type directly to this API. However, if pertokenScaleOptional is empty, you must first call the aclnn interface of the TransQuantParamV2 operator to convert dequantScale into INT64 or UINT64 type.</li></ul></td>
          <td>INT64, UINT64, FLOAT32, BFLOAT16</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>pertokenScaleOptional</td>
          <td>Input</td>
          <td>Dequantization coefficient after MatMul computation in the pre-token scale, that is, pertokenScaleOptional in the formula.</td>
          <td>When x1 is (b, s, k), the shape is (b × s). When x1 is (m, k), the shape is (m).</td>
          <td>FLOAT32</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>group</td>
          <td>Input</td>
          <td>Communication domain name.</td>
          <td>It is obtained through the Hccl API extern HcclResult HcclGetCommName(HcclComm comm, char* commName);. commName corresponds to group.</td>
          <td>String</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>reduceOp</td>
          <td>Input</td>
          <td>Reduce operation type.</td>
          <td>The current version supports only "sum."</td>
          <td>String</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>commTurn</td>
          <td>Input</td>
          <td>Number of communication data splits, that is, the total data size divided by the communication size at a time.</td>
          <td>The current version supports only 0.</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>streamMode</td>
          <td>Input</td>
          <td>Enumeration of stream modes.</td>
          <td>The current version supports only enumerated value 1.</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>output</td>
          <td>Output</td>
          <td>Result of MatMul computation and AllReduce communication, that is, output in the formula.</td>
          <td>The number of dimensions of output is the same as that of x1.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>2-3</td>
          <td>√</td>
        </tr>
        <tr>
          <td>workspaceSize</td>
          <td>Output</td>
          <td>Size of the workspace required to be allocated on the device</td>
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

    - <term>Atlas A2 training products/Atlas A2 inference products</term>: The `x2` input can be in ND (2D input only) or FRACTAL_NZ (4D input only) format. When the format of `x2` is FRACTAL_NZ, `aclnnCalculateMatmulWeightSizeV2` and `aclnnTransMatmulWeight` are used to convert the data format from ND into NZ. non-contiguous tensor support only the transpose scenario.

- **Returns:**

    aclnnStatus status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API implements input parameter verification. The following errors may be thrown:

    <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
    <col style="width: 281px">
    <col style="width: 119px">
    <col style="width: 749px">
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
        <td>The input x1, x2, dequantScale, or output is a null pointer.</td>
    </tr>
    <tr>
        <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="3">161002</td>
        <td>The data type of x1, x2, biasOptional, dequantScale, pertokenScaleOptional, x3Optional, or output is not supported.</td>
    </tr>
    <tr>
        <td>The value of streamMode is not within the valid range.</td>
    </tr>
    <tr>
        <td>The shape of x1, x2, biasOptional, dequantScale, pertokenScaleOptional, x3Optional, or output does not meet the requirements.</td>
    </tr>
    </tbody>
    </table>

## aclnnQuantMatmulAllReduceV2

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1150px">
    <col style="width: 168px">
    <col style="width: 128px">
    <col style="width: 854px">
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
        <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnQuantMatmulAllReduceV2GetWorkspaceSize.</td>
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
    </tbody></table>
    
- **Returns:**

    aclnnStatus status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computing:
  - `aclnnQuantMatmulAllReduceV2` defaults to a non-deterministic implementation. You can call `aclrtCtxSetSysParamOpt` to enable deterministic compute.

- MC2 is disabled in incremental generation scenarios but enabled in full generation scenarios.
- `x1` can be 2D (m, k) or 3D (b, s, k) and cannot be empty. `x2` must be 2D and cannot be empty. (k, n), where the k axis meets the input parameter requirements of the MatMul operator. The k axes of `x1` and `x2` must be equal.
- The value of m cannot exceed 2147483647. The size of the last dimension of `x1` (k) and `x2` (k when transposed and n when not transposed) cannot exceed 65535.
- If `bias` is not empty, its shape is (n). If `x3` is not empty, its shape is the same as that of `output`.
- When the shape of input `x1` is (b, s, k), `output` shape is (b, s, n). When the shape of input `x1` is (m, k), `output` shape is (m, n).
- The passed `x1`, `x2`, `dequantScale`, or `output` cannot be a null pointer.
- The data types and formats of `x1`, `x2`, `dequantScale`, `output`, `bias` (when not empty), and `x3` (when not empty) must be supported.
- If `output` is of FLOAT16 type, the type of `dequantScale` is INT64 or UINT64 if `pertokenScaleOptional` is empty, or FLOAT32 if `pertokenScaleOptional` is not empty.
- If `output` is of BFLOAT16 type, the type of `dequantScale` is BFLOAT16.
- If the shape of `x1` is (b, s, k), the shape of `pertokenScaleOptional` is (b × s). If the shape of `x1` is (m, k), the shape of `pertokenScaleOptional` is (m).
- Only the all-mesh networking of HCCS links is supported.
    - <term>Atlas A2 training products/Atlas A2 inference products</term>: 1, 2, 4, or 8 ranks are supported.
- <term>Atlas A2 training products and Atlas A2 inference products</term>: supports only one communication domain for MC2 operators within a model.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

- <term>Atlas A2 training products/Atlas A2 inference products</term>:

    ```Cpp
    #include <iostream>
    #include <vector>
    #include <thread>
    #include "hccl/hccl.h"
    #include "aclnnop/aclnn_trans_matmul_weight.h"
    #include "aclnnop/aclnn_quant_matmul_all_reduce_v2.h"

    int ndev = 8;

    #define ACL_CHECK(ret)                                                                                     \
        do {                                                                                                   \
            auto retcode = ret;                                                                                \
            if (retcode != ACL_SUCCESS) {                                                                      \
                printf("[ERROR] acl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, retcode); \
                return retcode;                                                                                \
            }                                                                                                  \
        } while (0)

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

    struct Args {
        uint32_t rankId;
        HcclComm hcclComm;
        aclrtStream stream;
        aclrtContext context;
        std::string format;
    };

    template<typename T>
    int CreateWeightNzAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                                aclDataType dataType, aclTensor **tensor, Args &args) {
        auto size = GetShapeSize(shape) * sizeof(T);
        const aclIntArray *mat2Size = aclCreateIntArray(shape.data(), shape.size());
        auto ret = aclnnCalculateMatmulWeightSizeV2(mat2Size, ACL_INT8, &size);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCalculateMatmulWeightSizeV2 failed. ERROR: %d\n", ret); return    ret);
        auto tensorSize = size * sizeof(T);

        // Call aclrtMalloc to allocate memory on the device.
        ret = aclrtMalloc(deviceAddr, tensorSize, ACL_MEM_MALLOC_HUGE_FIRST);
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

        uint64_t transWorkspaceSize;
        aclOpExecutor *executor;
        void *transWorkspaceAddr = nullptr;
        ret = aclnnTransMatmulWeightGetWorkspaceSize(*tensor, &transWorkspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS && transWorkspaceSize > 0,
                printf("[ERROR] aclnnTransMatmulWeightGetWorkspaceSize failed. ret = %d \n", ret); return ret);
        ACL_CHECK(aclrtMalloc(&transWorkspaceAddr, transWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
        ret = aclnnTransMatmulWeight(transWorkspaceAddr, transWorkspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnTransMatmulWeight failed. ret = %d \n", ret);return ret);
        ACL_CHECK(aclrtSynchronizeStreamWithTimeout(args.stream, 20000));

        return 0;
    }

    template<typename T>
    int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                        aclDataType dataType, aclTensor **tensor) {
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

    int launchOneThreadQuantMatmulAllReduce(Args &args) {
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
        std::vector<int64_t> dequantScaleShape = {128};
        std::vector<int64_t> pertokenScaleShape = {32};
        std::vector<int64_t> x3Shape = {32, 128};
        std::vector<int64_t> outShape = {32, 128};
        void *x1DeviceAddr = nullptr;
        void *x2DeviceAddr = nullptr;
        void *biasDeviceAddr = nullptr;
        void *dequantScaleDeviceAddr = nullptr;
        void *pertokenScaleDeviceAddr = nullptr;
        void *x3DeviceAddr = nullptr;
        void *outDeviceAddr = nullptr;
        aclTensor *x1 = nullptr;
        aclTensor *x2 = nullptr;
        aclTensor *bias = nullptr;
        aclTensor *dequantScale = nullptr;
        aclTensor *pertokenScale = nullptr;
        aclTensor *x3 = nullptr;
        aclTensor *out = nullptr;

        int64_t commTurn = 0;
        int64_t streamMode = 1;
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;
        void *workspaceAddr = nullptr;

        long long x1ShapeSize = GetShapeSize(x1Shape);
        long long x2ShapeSize = GetShapeSize(x2Shape);
        long long biasShapeSize = GetShapeSize(biasShape);
        long long dequantScaleShapeSize = GetShapeSize(dequantScaleShape);
        long long pertokenScaleShapeSize = GetShapeSize(pertokenScaleShape);
        long long x3ShapeSize = GetShapeSize(x3Shape);
        long long outShapeSize = GetShapeSize(outShape);

        std::vector<int8_t> x1HostData(x1ShapeSize, 1);
        std::vector<int8_t> x2HostData(x2ShapeSize, 1);
        std::vector<int32_t> biasHostData(biasShapeSize, 1);
        std::vector<float> dequantScaleHostData(dequantScaleShapeSize, 1);
        std::vector<float> pertokenScaleHostData(pertokenScaleShapeSize, 1);
        std::vector<int16_t> x3HostData(x3ShapeSize, 1);
        std::vector<int16_t> outHostData(outShapeSize, 0);
        // Create a tensor.
        ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        if (args.format == "NZ") {
            ret = CreateWeightNzAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2, args);
        } else {
            ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
        }
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_INT32, &bias);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(dequantScaleHostData, dequantScaleShape, &dequantScaleDeviceAddr,
                            aclDataType::ACL_FLOAT, &dequantScale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(pertokenScaleHostData, pertokenScaleShape, &pertokenScaleDeviceAddr,
                            aclDataType::ACL_FLOAT, &pertokenScale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x3HostData, x3Shape, &x3DeviceAddr, aclDataType::ACL_FLOAT16, &x3);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Call the first-phase API.
        ret = aclnnQuantMatmulAllReduceV2GetWorkspaceSize(x1, x2, bias, x3, dequantScale, pertokenScale,
                                                        hcom_name, "sum", commTurn, streamMode, out,
                                                        &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclnnQuantMatmulAllReduceV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
        // Allocate device memory based on workspaceSize computed by the first-phase API.
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // Call the second-phase API.
        ret = aclnnQuantMatmulAllReduceV2(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulAllReduceV2 failed. ERROR: %d\n", ret); return ret);
        // (Fixed writing) Wait until the task execution is complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
        LOG_PRINT("device%d aclnnQuantMatmulAllReduceV2 execute success \n", args.rankId);
        // Release device resources. Modify the configuration based on the API definition.
        if (x1 != nullptr) {
            aclDestroyTensor(x1);
        }
        if (x2 != nullptr) {
            aclDestroyTensor(x2);
        }
        if (bias != nullptr) {
            aclDestroyTensor(bias);
        }
        if (dequantScale != nullptr) {
            aclDestroyTensor(dequantScale);
        }
        if (pertokenScale != nullptr) {
            aclDestroyTensor(pertokenScale);
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
        if (dequantScaleDeviceAddr != nullptr) {
            aclrtFree(dequantScaleDeviceAddr);
        }
        if (pertokenScaleDeviceAddr != nullptr) {
            aclrtFree(pertokenScaleDeviceAddr);
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
                    new(std::nothrow) std::thread(&launchOneThreadQuantMatmulAllReduce, std::ref(args[rankId])));
        }
        for (uint32_t rankId = 0; rankId < ndev; rankId++) {
            threads[rankId]->join();
        }
        aclFinalize();
        return 0;
    }
    ```
