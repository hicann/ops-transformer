# aclnnQuantMatmulAllReduceAddRmsNorm

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

- **API function**: Performs MatMul, AllReduce, addition, and RMSNorm operations in sequence.
- **Formula**:

  $$
  mm_out = allReduce(dequantScale * (x1_{int8}@x2_{int8} + bias_{int32}))
  $$

  $$
  y = mm_out + residual
  $$

  $$
  normOut = \frac{y}{RMS(y)} * gamma, RMS(y) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} y_{i}^{2} + epsilon}
  $$

## Prototype

- `aclnnQuantMatmulAllReduceAddRmsNorm` and `aclnnInplaceQuantMatmulAllReduceAddRmsNorm` implement the same function in different ways. Select a proper operator based on your requirements.

  - `aclnnQuantMatmulAllReduceAddRmsNorm`: Output tensor objects `normOut` and `y` need to be created to store the computation result.
  - `aclnnInplaceQuantMatmulAllReduceAddRmsNorm`: Output tensor object `normOut` needs to be created. The results that would have been stored in the output tensor `y` in the non-inplace scenario are directly written to the memory of the input tensor `residual`.

- Each operator has two-phase API calls. First, `aclnnQuantMatmulAllReduceAddRmsNormGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnQuantMatmulAllReduceAddRmsNorm` is called to perform computation.

```cpp
aclnnStatus aclnnQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *bias,
    const aclTensor *dequantScale,
    const aclTensor *residual,
    const aclTensor *gamma,
    double           epsilon,
    const char      *group,
    const char      *reduceOp,
    int64_t          commTurn,
    int64_t          streamMode,
    const aclTensor *y,
    const aclTensor *normOut,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnQuantMatmulAllReduceAddRmsNorm(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnQuantMatmulAllReduceAddRmsNormGetWorkspaceSize

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
          <td><ul><li>Empty tensors are supported. </li><li>The data type must be the same as that of x2. </li><li>The current version supports only 2D or 3D inputs.</li></ul></td>
          <td>INT8</td>
          <td>ND</td>
          <td>2-3</td>
          <td>×</td>
        </tr>
        <tr>
          <td>x2</td>
          <td>Input</td>
          <td>Right matrix of MatMul computation, that is, x2 in the formula.</td>
          <td><ul><li>Empty tensors are supported. </li><li>The data type must be the same as that of x1. </li><li>The current version only supports two-dimensional input shapes and both transposed and non-transposed scenarios. </li><li>non-contiguous tensor in the transpose scenario are supported.</li></ul></td>
          <td>INT8</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>bias</td>
          <td>Input</td>
          <td> Corresponds to bias in the formula.</td>
          <td><ul><li>Null pointers are supported. </li><li>The current version supports only 1D inputs.</li></ul></td>
          <td>INT32</td>
          <td>ND</td>
          <td>1</td>
          <td>√</td>
        </tr>
        <tr>
          <td>dequantScale</td>
          <td>Input</td>
          <td>Dequantization coefficient after MatMul computation, that is, dequantScale in the formula.</td>
          <td>The shape is (1) in the per-tensor scenario and (n) or (1,n) in the per-channel scenario.</td>
          <td>UINT64, INT64, BFLOAT16</td>
          <td>ND</td>
          <td>1-2</td>
          <td>×</td>
        </tr>
        <tr>
          <td>residual</td>
          <td>Input</td>
          <td>Residual input of the AddRmsNorm fusion operator, that is, residual in the formula.</td>
          <td>In the inplace scenario, residual is used as the output address of y. The current version supports only 3D inputs.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>3</td>
          <td>×</td>
        </tr>
        <tr>
          <td>gamma</td>
          <td>Input</td>
          <td>RmsNorm input of the AddRmsNorm fusion operator, that is, gamma in the formula.</td>
          <td>The current version supports only 1D inputs.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>1</td>
          <td>×</td>
        </tr>
        <tr>
          <td>epsilon</td>
          <td>Input</td>
          <td>Double-precision floating-point value on the host, used to prevent division-by-zero errors, corresponding to epsilon in the formula.</td>
          <td>The value of epsilon must be within the range (0, 1).</td>
          <td>Double</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>group</td>
          <td>Input</td>
          <td>String on the host identifying the communication domain, that is, the name of the communication domain.</td>
          <td>It is obtained through the Hccl API extern HcclResult HcclGetCommName(HcclComm comm, char* commName);. commName corresponds to group.</td>
          <td>String</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>reduceOp</td>
          <td>Input</td>
          <td>String on the host identifying the operation type, that is, the reduce operation type.</td>
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
          <td>The current version supports only 1.</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>y</td>
          <td>Output</td>
          <td>Result of MatMul, AllReduce, and addition operations in sequence, that is, y in the formula.</td>
          <td><ul><li>Empty tensors are not supported. </li><li>The data type is the same as that of the residual input.</li></ul></td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>3</td>
          <td>√</td>
        </tr>
        <tr>
          <td>normOut</td>
          <td>Output</td>
          <td>Result of MatMul, AllReduce, addition, and RMSNorm operations in sequence, that is, normOut in the formula.</td>
          <td><ul><li>Empty tensors are not supported. </li><li>The data type is the same as that of the residual input.</li></ul></td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
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

- **Returns:**

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API performs input parameter validation. The following error codes may be returned:

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
        <td>The input x1, x2, residual, gamma, or dequantScale is a null pointer.</td>
    </tr>
    <tr>
        <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="3">161002</td>
        <td>The shape of x1, x2, or residual does not meet the requirements.</td>
    </tr>
    <tr>
        <td>The value of streamMode is not within the valid range.</td>
    </tr>
    <tr>
        <td>The k value of the last dimension of x1 is 0.</td>
    </tr>
    </tbody>
    </table>

## aclnnQuantMatmulAllReduceAddRmsNorm

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1150px">
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
        <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnQuantMatmulAllReduceAddRmsNormGetWorkspaceSize.</td>
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
  - `aclnnQuantMatmulAllReduceAddRmsNorm` defaults to a non-deterministic implementation. You can call `aclrtCtxSetSysParamOpt` to enable deterministic compute.

- The application scenario is the same as that of aclnnQuantMatmulAllReduce. MC2 is disabled in incremental generation scenarios but enabled in full generation scenarios
- `x1` can be 2D (m, k) or 3D (b, s, k). `x2` must be 2D (k, n), where the axes meet the input parameter requirements of the MatMul operator. The k axes of `x1` and `x2` must be equal. If `bias` is not empty, it must be 1D with shape (n).
- The value of m cannot exceed 2147483647. The size of the last dimension of `x1` (k) and `x2` (k when transposed and n when not transposed) cannot exceed 65535.
- The input `residual` must be 3D (b, s, n). When `x1` is 2D, (b × s) of `residual` equals m of `x1`. The shape of the input `gamma` is (n).
- The dimensions and data types of the outputs `y` and `normOut` are the same as those of `residual`, with shape (b, s, n).
- If the output `residual` is of FLOAT16 type, the type of `dequantScale` is UINT64 or INT64. If the output `residual` is of BFLOAT16 type, the type of `dequantScale` is BFLOAT16.
- The data type of `x1` and `x2` is INT8, and the data type of `bias` is INT32. The data types of `residual`, `gamma`, `y`, and `normOut` must be the same.
- The `x1` matrix cannot be transposed. The `x2` matrix can be transposed or not transposed.
- 1, 2, 4, and 8 ranks are supported, and only the all-mesh networking of HCCS links is supported.
- Empty tensors with (b × s) and n of 0 are supported. Empty tensors with k of 0 are not supported.
- <term>Atlas A2 training products and Atlas A2 inference products</term>: supports only one communication domain for MC2 operators within a model.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include <thread>
#include "hccl/hccl.h"
#include "aclnnop/aclnn_quant_matmul_all_reduce_add_rms_norm.h"

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

struct Args {
    uint32_t rankId;
    HcclComm hcclComm;
    aclrtStream stream;
    aclrtContext context;
};

int launchOneThreadQuantMatmulAllReduceAddRmsNorm(Args &args) {
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
    std::vector<int64_t> residualShape = {1, 32, 128};
    std::vector<int64_t> gammaShape = {128};
    std::vector<int64_t> yShape = {1, 32, 128};
    std::vector<int64_t> normOutShape = {1, 32, 128};
    void *x1DeviceAddr = nullptr;
    void *x2DeviceAddr = nullptr;
    void *biasDeviceAddr = nullptr;
    void *dequantScaleDeviceAddr = nullptr;
    void *residualDeviceAddr = nullptr;
    void *gammaDeviceAddr = nullptr;
    void *yDeviceAddr = nullptr;
    void *normOutDeviceAddr = nullptr;
    aclTensor *x1 = nullptr;
    aclTensor *x2 = nullptr;
    aclTensor *bias = nullptr;
    aclTensor *dequantScale = nullptr;
    aclTensor *residual = nullptr;
    aclTensor *gamma = nullptr;
    aclTensor *y = nullptr;
    aclTensor *normOut = nullptr;

    int64_t commTurn = 0;
    int64_t streamMode = 1;
    double  epsilon = 0.000001;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    void *workspaceAddr = nullptr;

    long long x1ShapeSize = GetShapeSize(x1Shape);
    long long x2ShapeSize = GetShapeSize(x2Shape);
    long long biasShapeSize = GetShapeSize(biasShape);
    long long dequantScaleShapeSize = GetShapeSize(dequantScaleShape);
    long long residualShapeSize = GetShapeSize(residualShape);
    long long gammaShapeSize = GetShapeSize(gammaShape);
    long long yShapeSize = GetShapeSize(yShape);
    long long normOutShapeSize = GetShapeSize(normOutShape);

    std::vector<int8_t> x1HostData(x1ShapeSize, 1);
    std::vector<int8_t> x2HostData(x2ShapeSize, 1);
    std::vector<int32_t> biasHostData(biasShapeSize, 1);
    std::vector<uint64_t> dequantScaleHostData(dequantScaleShapeSize, 1);
    std::vector<int16_t> residualHostData(residualShapeSize, 1);
    std::vector<int16_t> gammaHostData(gammaShapeSize, 1);
    std::vector<int16_t> yHostData(yShapeSize, 0);
    std::vector<int16_t> normOutHostData(normOutShapeSize, 0);
    // Create a tensor.
    ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_INT32, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dequantScaleHostData, dequantScaleShape, &dequantScaleDeviceAddr,
                        aclDataType::ACL_UINT64, &dequantScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(residualHostData, residualShape, &residualDeviceAddr, aclDataType::ACL_FLOAT16, &residual);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT16, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(normOutHostData, normOutShape, &normOutDeviceAddr, aclDataType::ACL_FLOAT16, &normOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // aclnnQuantMatmulAllReduceAddRmsNorm API call example
    // Call the first-phase API.
    ret = aclnnQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(x1, x2, bias, dequantScale, residual, gamma, epsilon,
                                        hcom_name, "sum", commTurn, streamMode, y, normOut,
                                        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnQuantMatmulAllReduceAddRmsNormGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API.
    ret = aclnnQuantMatmulAllReduceAddRmsNorm(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulAllReduceAddRmsNorm failed. ERROR: %d\n", ret); return ret);
    // (Fixed writing) Wait until the task execution is complete.
    ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("device%d aclnnQuantMatmulAllReduceAddRmsNorm execute success \n", args.rankId);
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
    if (residual != nullptr) {
        aclDestroyTensor(residual);
    }
    if (gamma != nullptr) {
        aclDestroyTensor(gamma);
    }
    if (y != nullptr) {
        aclDestroyTensor(y);
    }
    if (normOut != nullptr) {
        aclDestroyTensor(normOut);
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
    if (residualDeviceAddr != nullptr) {
        aclrtFree(residualDeviceAddr);
    }
    if (gammaDeviceAddr != nullptr) {
        aclrtFree(gammaDeviceAddr);
    }
    if (yDeviceAddr != nullptr) {
        aclrtFree(yDeviceAddr);
    }
    if (normOutDeviceAddr != nullptr) {
        aclrtFree(normOutDeviceAddr);
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
                new(std::nothrow) std::thread(&launchOneThreadQuantMatmulAllReduceAddRmsNorm, std::ref(args[rankId])));
    }
    for (uint32_t rankId = 0; rankId < ndev; rankId++) {
        threads[rankId]->join();
    }
    aclFinalize();
    return 0;
}
```
