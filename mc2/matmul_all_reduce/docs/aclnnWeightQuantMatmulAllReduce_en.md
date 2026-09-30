# aclnnWeightQuantMatmulAllReduce

## Supported Products

| Product                                                      | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>     |    ×     |
| <term>Atlas A2 training products/Atlas A2 inference products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

**Note:** When using this API, ensure that the driver firmware package and CANN package are in the 8.0.RC2 version or later. Otherwise, an error, such as BUS ERROR, will be reported.

## Function

- **Interface Function**: Perform pseudo-quantization computation on the input parameter x2, then complete the Matmul and AllReduce computation. Supports pertensor, perchannel, and pergroup quantization methods.

- **Formula**:

  $$
  output = allreduce(x1 @ ((x2 + antiquantOffset) *antiquantScale) + bias+ x3) 
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First call `aclnnWeightQuantMatmulAllReduceGetWorkspaceSize` to obtain the required workspace size for computation and the executor that includes the operator's computation process. Then, call `aclnnWeightQuantMatmulAllReduce` to perform the computation.

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
          <td>The left matrix for MatMul computation, that is, x1 in the calculation formula.</td>
          <td><ul><li>The current version only supports two-dimensional or three-dimensional input.</li><li>Supports non-transposed scenarios.</li></ul></td>
          <td>BFLOAT16, FLOAT16</td>
          <td>ND</td>
          <td>2-3</td>
          <td>×</td>
        </tr>
        <tr>
          <td>x2</td>
          <td>Input</td>
          <td>The right matrix for MatMul computation, that is, x2 in the calculation formula.</td>
          <td><ul><li>The current version only supports two-dimensional input.</li><li>Supports both transposed and non-transposed scenarios.</li></ul></td>
          <td>-</td>
          <td>ND, FRACTAL_NZ</td>
          <td>2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>bias</td>
          <td>Input</td>
          <td>Corresponds to the bias offset in the calculation formula, that is, biasOptional in the formula.</td>
          <td>Supports passing null pointers. When non-null, the current version only supports one-dimensional input.</td>
          <td>-</td>
          <td>ND</td>
          <td>1</td>
          <td>√</td>
        </tr>
        <tr>
          <td>antiquantScale</td>
          <td>Input</td>
          <td>This is the antiquantScale in the calculation formula.</td>
          <td>In the pertensor scenario, the shape is (1); in the perchannel scenario, the shape is (n)/(1,n), where n is the size of the last dimension of x2; in the pergroup scenario, the shape is (ceil(k,antiquantGroupSize),n).</td>
          <td>BFLOAT16, FLOAT16</td>
          <td>ND</td>
          <td>1-2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>antiquantOffset</td>
          <td>Input</td>
          <td>The offset parameter for pseudo-quantization calculation of x2, i.e., x1ScaleOptional in the computation formula.</td>
          <td>Supports passing a null pointer; when non-null, the shape should match that of antiquantScale. When x1 is FLOAT16 or BFLOAT16 and weight is FLOAT8_E5M2, FLOAT8_E4M3FN, or HIFLOAT8, this parameter is not supported, and a null pointer should be passed.</td>
          <td>BFLOAT16, FLOAT16</td>
          <td>ND</td>
          <td>1-2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>x3</td>
          <td>Input</td>
          <td>The add operation after MatMul computation, that is, x3Optional in the formula.</td>
          <td>Supports passing in a null pointer; when non-null, the shape is the same as the shape after mm computation.</td>
          <td>-</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>group</td>
          <td>Input</td>
          <td>Communication domain name.</td>
          <td>Obtained through the HCCL interface "extern HcclResult HcclGetCommName(HcclComm comm, char* commName);", where commName is the group.</td>
          <td>String</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>reduceOp</td>
          <td>Input</td>
          <td>Type of reduce operation.</td>
          <td>The current version only supports inputting sum.</td>
          <td>String</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>commTurn</td>
          <td>Input</td>
          <td>Number of communication data splits, that is, total data volume/single communication volume.</td>
          <td>The current version only supports inputting 0.</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>streamMode</td>
          <td>Input</td>
          <td>Enumeration of stream modes.</td>
          <td>The current version only supports enumeration value 1.</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>antiquantGroupSize</td>
          <td>Input</td>
          <td>Group size input for dequantization calculation of x2 in per-group pseudo-quantization mode.</td>
          <td>When pergroup is not supported, pass in 0; when supported, the value range is [32, min(k-1, INT_MAX)], and it must be a multiple of 32; the range of k is consistent with the mm interface<a href="./aclnnMatmulAllReduce.md".</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>commQuantMode</td>
          <td>Input</td>
          <td>Flags for static quantization and dynamic quantization.</td>
          <td>The value is 0 and 1.</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>output</td>
          <td>Output</td>
          <td>Calculate the result of MatMul computation and AllReduce communication, that is, the output in the formula.</td>
          <td>The dimension of the output is consistent with x1.</td>
          <td>-</td>
          <td>ND</td>
          <td>2-3</td>
          <td>√</td>
        </tr>
        <tr>
          <td>workspaceSize</td>
          <td>Output</td>
          <td>Returns the workspace size that needs to be allocated on the device side.</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>executor</td>
          <td>Output</td>
          <td>Returns the op executor, which includes the operator computation process.</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
      </tbody>
    </table>

    - <term>Atlas A2 training products/Atlas A2 inference products</term>:
      - The data types supported for input x2 are INT8 and INT4, and the data formats supported are ND (currently only 2D input is supported) and FRACTAL_NZ (currently only 4D input is supported). When the data format of x2 is FRACTAL_NZ, use `aclnnCalculateMatmulWeightSizeV2` and `aclnnTransMatmulWeight` to complete the conversion from ND to NZ input. non-contiguous tensor are only supported in transpose scenarios.
      - The data type of the input bias should be consistent with that of x1.
      - The data type of input x3 supports BFLOAT16 and FLOAT16.
      - The data type of the output supports BFLOAT16 and FLOAT16.

- **Returns:**

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

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
        <td>The passed x1, x2, antiquantScale, or output is a null pointer.</td>
    </tr>
    <tr>
        <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="3">161002</td>
        <td>The data types of x1, x2, bias, antiquantScale, antiquantOffset, x3, or output do not meet the requirements.</td>
    </tr>
    <tr>
        <td>reduceOp, streamMode, antiQuantGroupSize are not within the valid range.</td>
    </tr>
    <tr>
        <td>The shapes of x1, x2, bias, antiquantScale, antiquantOffset, x3, output, and antiquantGroupSize do not meet the constraint requirements.</td>
    </tr>
    </tbody>
    </table>

## aclnnWeightQuantMatmulAllReduce

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
        <td>The memory address of the workspace applied for on the device side.</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>The size of the workspace allocated on the device side, obtained by the first-phase of aclnnWeightQuantMatmulAllReduceGetWorkspaceSize.</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Operator executor, which includes the computation process of the operator.</td>
    </tr>
    <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Specifies the stream for task execution.</td>
    </tr>
    </tbody></table>
    
- **Returns:**

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnWeightQuantMatmulAllReduce` defaults to a non-deterministic implementation. You can call `aclrtCtxSetSysParamOpt` to enable deterministic computation.

- MC2 is disabled in incremental scenarios but enabled in full scenarios.
- The input x1 can be two-dimensional or three-dimensional, with a shape of (b, s, k) or (m, k).
- x2 must be two-dimensional. Its shape is (k, n), where the k-axis meets the input requirements of the mm operator, the k-axis is equal, the range of m is [1, 2147483647], and the ranges of k and n are [1, 65535].
- The passed x1, x2, antiQuantScale, or output is not a null pointer.
- When the shape of input x1 is (b, s, k), the shape of x3 (non-empty scenario) and the output is (b, s, n); when the shape of input x1 is (m, k), the shape of x3 (non-empty scenario) and the output is (m, n).
- If bias is not empty, its shape size is equal to the size of the last dimension of the output. In the pertensor scenario, the shape of antiQuantScale is (1); in the perchannel scenario, the shape is (1,n)/(n); in the pergroup scenario, the shape is (ceil(k,antiQuantGroupSize), n). If antiQuantOffset is not empty, its shape is consistent with antiQuantScale.
- The data types and formats of x1, x2, x3 (non-empty scenarios), antiquantScale, antiquantOffset (non-empty scenarios), output, and bias (non-empty scenarios) must be within the supported range.
- x1, antiquantScale, antiquantOffset (non-empty scenario), x3 (non-empty scenario), bias (non-empty scenario) have the same data type for output. The value of antiquantGroupSize satisfies the range and is a multiple of 32.
- In the per-group scenario, when transposing x2, both the antiquantScale and antiquantOffset need to be transposed together to maintain continuity.
- In long sequence scenarios, as b/s or m increases, OOM or computation timeout may occur.
- Only supports all mesh networking for hccs links.
    - <term>Atlas A2 training products/Atlas A2 inference products</term>: Ranks 1, 2, 4, and 8 are supported.
- <term>Atlas A2 training products/Atlas A2 inference products</term>: Only the same communication domain for MC2 operators within a model is supported.

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

- <term>Atlas A2 training products/Atlas A2 inference products</term>:

    ```Cpp
    #include <iostream>
    #include <vector>
    #include <thread>
    #include <string.h>
    #include "hccl/hccl.h"
    #include "aclnnop/aclnn_weight_quant_matmul_all_reduce.h"

    int ndev = 8;

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
        // Call aclrtMalloc to allocate memory on the device side.
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
        // Call aclrtMemcpy to copy data from the host side to the device side memory.
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
        // Calculate the strides of a contiguous tensor.
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = shape.size() - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
        // Call the aclCreateTensor interface to create an aclTensor.
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
        std::vector<int16_t> x1HostData(x1ShapeSize, 1);
        std::vector<int8_t> x2HostData(x2ShapeSize, 1);
        std::vector<int16_t> biasHostData(biasShapeSize, 1);
        std::vector<int16_t> antiquantScaleHostData(antiquantScaleShapeSize, 1);
        std::vector<int16_t> antiquantOffsetHostData(antiquantOffsetShapeSize, 1);
        std::vector<int16_t> x3HostData(x3ShapeSize, 1);
        std::vector<int16_t> outHostData(outShapeSize, 0);
        // Create tensor.
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
        ret = aclnnWeightQuantMatmulAllReduceGetWorkspaceSize(x1, x2, bias, antiquantScale, antiquantOffset, x3,    hcom_name,
                                                            "sum", commTurn, streamMode, antiquantGroupSize, out,
                                                            &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclnnWeightQuantMatmulAllReduceGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
        // Allocate device memory based on the workspaceSize calculated from the first-phase API.
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
        // Initialize the communication domain.
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
        // Start multithreading.
        std::vector<std::unique_ptr<std::thread>> threads(ndev);
        for (uint32_t rankId = 0; rankId < ndev; rankId++) {
            args[rankId].rankId = rankId;
            args[rankId].hcclComm = comms[rankId];
            args[rankId].stream = stream[rankId];
            args[rankId].context = context[rankId];
            threads[rankId].reset(
                    new(std::nothrow) std::thread(&launchOneThreadweightQuantmatmulAllReduce, std::ref(args [rankId])));
        }
        for (uint32_t rankId = 0; rankId < ndev; rankId++) {
            threads[rankId]->join();
        }
        aclFinalize();
        return 0;
    }
    ```
