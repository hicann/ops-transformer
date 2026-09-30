# aclnnWeightQuantMatmulAllReduceAddRmsNorm

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

- **Description**: Performs mm + all_reduce + add + rms_norm computation.
- **Formula**:

  $$
  mm\_out = allReduce(x1 @ (x2*antiquantScale + antiquantOffset) + bias)
  $$

  $$
  y = mm\_out + residual
  $$

  $$
  normOut = \frac{y}{RMS(y)} * gamma, RMS(y) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} y_{i}^{2} + epsilon}
  $$

## Prototype

- `aclnnWeightQuantMatmulAllReduceAddRmsNorm` and `aclnnInplaceWeightQuantMatmulAllReduceAddRmsNorm` implement the same function in different ways. Select a proper operator based on your requirements.

  - `aclnnWeightQuantMatmulAllReduceAddRmsNorm`: Output tensor objects `normOut` and `y` need to be created to store the computation result.
  - `aclnnInplaceWeightQuantMatmulAllReduceAddRmsNorm`: Output tensor `normOut` needs to be created. The results that would have been stored in the output tensor `y` in the original non-inplace scenario are directly written to the memory of the input tensor `residual`.

- Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First call `aclnnWeightQuantMatmulAllReduceAddRmsNormGetWorkspaceSize` to obtain the required workspace size for computation and the executor that includes the operator's computation process. Then, call `aclnnWeightQuantMatmulAllReduceAddRmsNorm` to execute the computation.

```cpp
aclnnStatus aclnnWeightQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *bias,
    const aclTensor *antiquantScale,
    const aclTensor *antiquantOffset,
    const aclTensor *residual,
    const aclTensor *gamma,
    double           epsilon,
    const char      *group,
    const char      *reduceOp,
    int64_t          commTurn,
    int64_t          streamMode,
    int64_t          antiquantGroupSize,
    const aclTensor *y,
    const aclTensor *normOut,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnWeightQuantMatmulAllReduceAddRmsNorm(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnWeightQuantMatmulAllReduceAddRmsNormGetWorkspaceSize

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
          <td><ul><li>Supports empty Tensors.</li><li>The current version only supports two-dimensional or three-dimensional inputs.</li></ul></td>
          <td>BFLOAT16, FLOAT16</td>
          <td>ND</td>
          <td>2-3</td>
          <td>×</td>
        </tr>
        <tr>
          <td>x2</td>
          <td>Input</td>
          <td>The right matrix for MatMul computation, that is, x2 in the computation formula.</td>
          <td><ul><li>Supports empty Tensor.</li><li>The current version only supports 2D input, with scenarios of transposed/non-transposed.</li><li>Supports non-contiguous tensor in transposed scenarios.</li></ul></td>
          <td>INT8, INT4</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>bias</td>
          <td>Input</td>
          <td>Represents the bias in the calculation formula.</td>
          <td><ul><li>Supports scenarios with null pointers.</li><li>The current version only supports one-dimensional input.</li></ul></td>
          <td>BFLOAT16, FLOAT16</td>
          <td>ND</td>
          <td>1</td>
          <td>√</td>
        </tr>
        <tr>
          <td>antiquantScale</td>
          <td>Input</td>
          <td>It is the antiquantScale in the calculation formula.</td>
          <td>In the pertensor scenario, the shape is (1); in the PerChannel scenario, the shape is (n)/(1,n), where n is the size of the last dimension of x2; in the pergroup scenario, the shape is (ceil(k,antiquantGroupSize),n).</td>
          <td>BFLOAT16, FLOAT16</td>
          <td>ND</td>
          <td>1-2</td>
          <td>×</td>
        </tr>
        <tr>
          <td>antiquantOffset</td>
          <td>Input</td>
          <td>The offset parameter for pseudo-quantization calculation of x2, that is, the anti-quantOffset in the calculation formula.</td>
          <td>Optional, can be empty, when not empty, the shape should match the antiquantScale.</td>
          <td>BFLOAT16, FLOAT16</td>
          <td>ND</td>
          <td>1-2</td>
          <td>×</td>
        </tr>
        <tr>
          <td>residual</td>
          <td>Input</td>
          <td>The residual input of the AddRmsNorm fusion operator, which is the residual in the calculation formula.</td>
          <td>The current version only supports three-dimensional input.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>3</td>
          <td>×</td>
        </tr>
        <tr>
          <td>gamma</td>
          <td>Input</td>
          <td>The input for the RmsNorm calculation in the AddRmsNorm fusion operator, which is the gamma in the formula.</td>
          <td>The current version only supports one-dimensional input.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>1</td>
          <td>×</td>
        </tr>
        <tr>
          <td>epsilon</td>
          <td>Input</td>
          <td>Double precision on the host side, used to prevent division by zero errors, that is, epsilon in the calculation formula.</td>
          <td>The value of epsilon should be within the range (0,1).</td>
          <td>Double</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>group</td>
          <td>Input</td>
          <td>A string on the host side that identifies the communication domain, the name of the communication domain.</td>
          <td>Obtained through the HCCL interface "extern HcclResult HcclGetCommName(HcclComm comm, char* commName);", where commName is the group.</td>
          <td>String</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>reduceOp</td>
          <td>Input</td>
          <td>A string on the Host side indicating the operation type, specifically the reduce operation type.</td>
          <td>Currently, only "sum" is supported as input.</td>
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
          <td>Currently, only enumeration value 1 is supported.</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>antiquantGroupSize</td>
          <td>Input</td>
          <td>Group size input for dequantization calculation of x2 in per-group pseudo-quantization mode.</td>
          <td>When pergroup is not supported, pass in 0. When supported, the value range is [32, min(k-1,INT_MAX)], and it must be a multiple of 32. The range of k is consistent with the mm interface.</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>y</td>
          <td>Output</td>
          <td>The result of mm + all_reduce + add, which is y in the computation formula.</td>
          <td><ul><li>Empty Tensors are not supported.</li><li>Data type is the same as the residual input.</li></ul></td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>3</td>
          <td>√</td>
        </tr>
        <tr>
          <td>normOut</td>
          <td>Output</td>
          <td>The result of mm + all_reduce + add + rms_norm, which is the normOut in the computation formula.</td>
          <td><ul><li>Empty Tensors are not supported.</li><li>Data type is the same as the residual input.</li></ul></td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>2</td>
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
        <td>The passed x1, x2, residual, antiquantScale, or output is a null pointer.</td>
    </tr>
    <tr>
        <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="2">161002</td>
        <td>The shape of x1, x2, residual, or output does not meet the constraint requirements.</td>
    </tr>
    <tr>
        <td>streamMode is not within the valid range.</td>
    </tr>
    </tbody>
    </table>

## aclnnWeightQuantMatmulAllReduceAddRmsNorm

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
        <td>The memory address of the workspace allocated on the device side.</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>The size of the workspace allocated on the device side, obtained by the first-phase API of aclnnWeightQuantMatmulAllReduceAddRmsNormGetWorkspaceSize.</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Operator executor, which contains the computation process of the operator.</td>
    </tr>
    <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Specifies the stream for task execution.</td>
    </tr>
    </tbody></table>
    
- **Returns:**

    `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnWeightQuantMatmulAllReduceAddRmsNorm` defaults to a non-deterministic implementation. You can call `aclrtCtxSetSysParamOpt` to enable deterministic computation.

- The application scenario is the same as that of `aclnnWeightQuantMatmulAllReduce`. MC2 is disabled in incremental scenarios but enabled in full scenarios.
- The input `x1` can be two-dimensional or three-dimensional, with shape (b, s, k) or (s, k), respectively. `x2` must be two-dimensional, with a shape of (k, n), and the axes must meet the requirements of the mm operator input parameters, with the k axis being equal. The range of b\*s and s is [1, 2147483647], and the range of k and n is [1, 65535]. If bias is not empty, bias is one-dimensional, with a shape of (n).
- The input `residual` must be three-dimensional, with a shape of (b, s, n). When `x1` is two-dimensional, the (b*s) of `residual` equals the `s` of `x1`. The input `gamma` must be one-dimensional, with a shape of (n).
- The `antiquantScale` satisfies the following scenarios: for `pertensor`, the shape is `(1)`; for `perchannel`, the shape is `(1,n)` or `(n)`; for `pergroup`, the shape is `(ceil(k,antiquantGroupSize),n)`. If `antiquantOffset` is not empty, its shape is consistent with `antiquantScale`.
- The dimensions and data types of the output `y` and `normOut` are the same as `residual`. If `bias` is not empty, its shape size is equal to the last dimension of `normOut`.
- The data type of `x2` must be INT8 or INT4, and the data types of `x1`, `bias`, `residual`, `gamma`, `y`, and `normOut` computation inputs must be consistent.
- Only supports `x2` matrix transpose/non-transpose, `x1` matrix supports non-transpose scenarios.
- The value of antiquantGroupSize should satisfy the range [32, min(k-1, INT_MAX)] and be a multiple of 32.
- Supports ranks 1, 2, 4, and 8, and only supports full-mesh networking with HCCS links.
- Supports empty tensors with (b*s) and n as 0, but does not support empty tensors with k as 0.
- <term>Atlas A2 training products/Atlas A2 inference products</term>: Only the same communication domain for MC2 operators within a model is supported.

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include <thread>
#include "hccl/hccl.h"
#include "aclnnop/aclnn_weight_quant_matmul_all_reduce_add_rms_norm.h"

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

int launchOneThreadweightQuantmatmulAllReduceAddRmsNorm(Args &args) {
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
    std::vector<int64_t> residualShape = {1, 32, 128};
    std::vector<int64_t> gammaShape = {128};
    std::vector<int64_t> yShape = {1, 32, 128};
    std::vector<int64_t> normOutShape = {1, 32, 128};
    void *x1DeviceAddr = nullptr;
    void *x2DeviceAddr = nullptr;
    void *biasDeviceAddr = nullptr;
    void *antiquantScaleDeviceAddr = nullptr;
    void *antiquantOffsetDeviceAddr = nullptr;
    void *residualDeviceAddr = nullptr;
    void *gammaDeviceAddr = nullptr;
    void *yDeviceAddr = nullptr;
    void *normOutDeviceAddr = nullptr;
    aclTensor *x1 = nullptr;
    aclTensor *x2 = nullptr;
    aclTensor *bias = nullptr;
    aclTensor *antiquantScale = nullptr;
    aclTensor *antiquantOffset = nullptr;
    aclTensor *residual = nullptr;
    aclTensor *gamma = nullptr;
    aclTensor *y = nullptr;
    aclTensor *normOut = nullptr;

    int64_t commTurn = 0;
    int64_t streamMode = 1;
    double  epsilon = 0.000001;
    int64_t antiquantGroupSize = 0;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    void *workspaceAddr = nullptr;

    long long x1ShapeSize = GetShapeSize(x1Shape);
    long long x2ShapeSize = GetShapeSize(x2Shape);
    long long biasShapeSize = GetShapeSize(biasShape);
    long long antiquantScaleShapeSize = GetShapeSize(antiquantScaleShape);
    long long antiquantOffsetShapeSize = GetShapeSize(antiquantOffsetShape);
    long long residualShapeSize = GetShapeSize(residualShape);
    long long gammaShapeSize = GetShapeSize(gammaShape);
    long long yShapeSize = GetShapeSize(yShape);
    long long normOutShapeSize = GetShapeSize(normOutShape);
    std::vector<int16_t> x1HostData(x1ShapeSize, 1);
    std::vector<int8_t> x2HostData(x2ShapeSize, 1);
    std::vector<int16_t> biasHostData(biasShapeSize, 1);
    std::vector<int16_t> antiquantScaleHostData(antiquantScaleShapeSize, 1);
    std::vector<int16_t> antiquantOffsetHostData(antiquantOffsetShapeSize, 1);
    std::vector<int16_t> residualHostData(residualShapeSize, 1);
    std::vector<int16_t> gammaHostData(gammaShapeSize, 1);
    std::vector<int16_t> yHostData(yShapeSize, 0);
    std::vector<int16_t> normOutHostData(normOutShapeSize, 0);
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
    ret = CreateAclTensor(residualHostData, residualShape, &residualDeviceAddr, aclDataType::ACL_FLOAT16, &residual);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT16, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(normOutHostData, normOutShape, &normOutDeviceAddr, aclDataType::ACL_FLOAT16, &normOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Example of calling aclnnWeightQuantMatmulAllReduceAddRmsNorm.
    // Call the first-phase API.
    ret = aclnnWeightQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(x1, x2, bias, antiquantScale, antiquantOffset,
                                                        residual, gamma, epsilon, hcom_name,
                                                        "sum", commTurn, streamMode, antiquantGroupSize, y, normOut,
                                                        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnWeightQuantMatmulAllReduceAddRmsNormGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated from the first-phase API.
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API.
    ret = aclnnWeightQuantMatmulAllReduceAddRmsNorm(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantMatmulAllReduceAddRmsNorm failed. ERROR: %d\n", ret); return ret);
    // (Fixed writing) Synchronize the stream and wait for the task to complete.
    ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("device%d aclnnWeightQuantMatmulAllReduceAddRmsNorm execute success \n", args.rankId);
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
    if (antiquantScaleDeviceAddr != nullptr) {
        aclrtFree(antiquantScaleDeviceAddr);
    }
    if (antiquantOffsetDeviceAddr != nullptr) {
        aclrtFree(antiquantOffsetDeviceAddr);
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
                new(std::nothrow) std::thread(&launchOneThreadweightQuantmatmulAllReduceAddRmsNorm, std::ref(args[rankId])));
    }
    for (uint32_t rankId = 0; rankId < ndev; rankId++) {
        threads[rankId]->join();
    }
    aclFinalize();
    return 0;
}
```
