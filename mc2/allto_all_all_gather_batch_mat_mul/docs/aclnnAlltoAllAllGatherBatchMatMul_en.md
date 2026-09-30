# aclnnAlltoAllAllGatherBatchMatMul

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Description

- **Operator function**: Implements the fusion and parallelization of AllToAll and AllGather collective communications with BatchMatMul computation.

- **Formula**:

The computing logic is as follows, where y1, y2, and y3 are the outputs.

$$
x1 = AllToAll(x)
$$

$$
y2 = AllGather(x1)
$$

$$
y3 = BatchMatMul(y2, weight, bias)
$$

$$
y1 = Activation(y3)
$$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md)calls. First, `aclnnAlltoAllAllGatherBatchMatMulGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnAlltoAllAllGatherBatchMatMul` is called to perform computation.

```cpp
aclnnStatus aclnnAlltoAllAllGatherBatchMatMulGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* weight,
    const aclTensor* biasOptional,
    const char*      groupEp,
    const char*      groupTp,
    int64_t          epWorldSize,
    int64_t          tpWorldSize,
    int64_t          xShardType,
    int64_t          actType,
    aclTensor*       y1Out,
    aclTensor*       y2OutOptional,
    aclTensor*       y3OutOptional,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnAlltoAllAllGatherBatchMatMul(
    void*           workspace,
    uint64_t        workspaceSize,
    aclOpExecutor*  executor,
    aclrtStream     stream)
```

## aclnnAlltoAllAllGatherBatchMatMulGetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1123px"><colgroup>
    <col style="width: 160px">
    <col style="width: 111px">
    <col style="width: 539px">
    <col style="width: 188px">
    <col style="width: 125px">
    </colgroup>
    <thead>
    <tr>
    <th>Name</th>
    <th>Input/Output</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Data Format</th>
    </tr></thead>
    <tbody>
    <tr>
    <td>x</td>
    <td>Input</td>
    <td>The result of the communication operation serves as the left matrix for BatchMatMul computation. This input is used for AllToAll and AllGather collective communication and must be three-dimensional.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>weight</td>
    <td>Input</td>
    <td>Right matrix for BatchMatMul computation. The data type must be three-dimensional and the same as that of x.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>biasOptional</td>
    <td>Input</td>
    <td>Bias for BatchMatMul computation. If x is FLOAT16, biasOptional must be FLOAT16. If x is BFLOAT16, biasOptional must be FLOAT32. The shape can be two- or three-dimensional. Passing a null pointer is supported.</td>
    <td>FLOAT16, FLOAT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>groupEp</td>
    <td>Input</td>
    <td>Name of the expert parallelism (EP) communication domain. The string length must be greater than 0 and less than 128.</td>
    <td>STRING</td>
    <td>-</td>
    </tr>
    <tr>
    <td>groupTp</td>
    <td>Input</td>
    <td>Name of the tensor parallelism (TP) communication domain. The string length must be greater than 0 and less than 128.</td>
    <td>STRING</td>
    <td>-</td>
    </tr>
    <tr>
    <td>epWorldSize</td>
    <td>Input</td>
    <td>Size of the EP communication domain. The value can be 2, 4, 8, 16, or 32.</td>
    <td>INT64</td>
    <td>-</td>
    </tr>
    <tr>
    <td>tpWorldSize</td>
    <td>Input</td>
    <td>Size of the TP communication domain. The value can be 2, 4, 8, 16, or 32.</td>
    <td>INT64</td>
    <td>-</td>
    </tr>
    <tr>
    <td>xShardType</td>
    <td>Input</td>
    <td>0 indicates that AllGather is performed on the H dimension (the second dimension of x, assuming that x is a 3D tensor with dimensions 0, 1, and 2) within the TP domain. 1 indicates that AllGather is performed on the C dimension (the first dimension of x) within the TP domain.</td>
    <td>INT64</td>
    <td>-</td>
    </tr>
    <tr>
    <td>actType</td>
    <td>Input</td>
    <td>Type of activation function. The value can be 0, 1, 2, 3, or 4. 0 indicates no activation function. The mapping is [0: None, 1: GELU, 2: Silu, 3: Relu, 4: FastGELU].</td>
    <td>INT64</td>
    <td>-</td>
    </tr>
    <tr>
    <td>y1Out</td>
    <td>Output</td>
    <td>The final computation result. If there is an activation function, this is the activation function output. Otherwise, this is the BatchMatMul output. The shape can be three-dimensional . The data type is the same as that of the input x.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>y2OutOptional</td>
    <td>Output</td>
    <td>Optional output representing the result of AllGather, which may be required for the backward pass. The shape can be three-dimensional . The data type is the same as that of the input x. An empty pointer indicates that this output is not required.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>y3OutOptional</td>
    <td>Output</td>
    <td>Optional output representing the result of BatchMatMul when an activation function is used. The shape can be three-dimensional . The data type is the same as that of the input x. An empty pointer indicates that this output is not required.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>Output</td>
    <td>Size of the workspace required to be allocated on the device.</td>
    <td>UINT64</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>Output</td>
    <td>Operator executor, containing the operator computation process.</td>
    <td>aclOpExecutor*</td>
    <td>-</td>
    </tr>
    </tbody></table>

- **Returns**

    aclnnStatus: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
    
    The first-phase API implements input parameter verification. The following errors may be thrown.

    <table style="undefined;table-layout: fixed; width: 1147px"><colgroup>
    <col style="width: 286px">
    <col style="width: 118px">
    <col style="width: 743px">
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
        <td>The input x, weight, groupEp, groupTp, or y1Out is a null pointer.</td>
    </tr>
    <tr>
        <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="6">161002</td>
        <td>The length of the groupEp or groupTp string is invalid.</td>
    </tr>
    <tr>
        <td>The input data type is not supported.</td>
    </tr>
    <tr>
        <td>The attribute value is invalid.</td>
    </tr>
    <tr>
        <td>The aclTensor dimensions are invalid.</td>
    </tr>
    <tr>
        <td>The aclTensor shape is invalid.</td>
    </tr>
    <tr>
        <td>The scenario for enabling optional output is invalid. When the optional output y3OutOptional is required, actType must not be 0.</td>
    </tr>
    </tbody>
    </table>

## aclnnAlltoAllAllGatherBatchMatMul

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
    <td>Size of the workspace to be allocated on the device, which is obtained by calling <code>aclnnAlltoAllAllGatherBatchMatMulGetWorkspaceSize</code>.</td>
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

- **Returns**

    aclnnStatus status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computing:
  - `aclnnAlltoAllAllGatherBatchMatMul` defaults to a deterministic implementation.

Due to the requirements of collective communication and BatchMatMul computation, the input and output shapes must meet the following mathematical relationship (ep = `epWorldSize` and tp = `tpWorldSize`):

AllGather by the H axis (`xShardType` = 0):

  - `x`: (E, C, H/tp)
  - `weight`: (E/ep, H, M/tp)
  - `biasOptional`: (E/ep, 1, M/tp) for 3D and (E/ep, M/tp) for 2D when the pointer is not null.
  - `y1Out`: (E/ep, ep\*C, M/tp)
  - `y2OutOptional`: (E/ep, ep\*C, H)
  - `y3OutOptional`: (E/ep, ep\*C, M/tp)

AllGather by the C axis (xShardType = 1):

  - `x`: (E, C/tp, H)
  - `weight`: (E/ep, H, M/tp)
  - `biasOptional`: (E/ep, 1, M/tp) for 3D and (E/ep, M/tp) for 2D when the pointer is not null.
  - `y1Out`: (E/ep, ep*tp\*C/tp, M/tp)
  - `y2OutOptional`: (E/ep, ep*tp\*C/tp, H)
  - `y3OutOptional`: (E/ep, ep*tp\*C/tp, M/tp)

Data relationship description:

  - For example, if `x.size(0)` is equal to E and weight.size(0) is equal to E/ep, it indicates that `x.size(0)` = `ep*weight.size(0)`, where `x.size(0)` is an integer multiple of ep. Other relationships are similar to this.
  - The value range of E is [2, 512], and E is an integer multiple of ep.
  - The value range of H is [1, 65535]. When xShardType is set to 0, H is an integer multiple of tp.
  - The value range of M/tp is [1, 65535].
  - The value range of E/ep is [1, 32].
  - ep and tp can only be 2, 4, 8, 16, or 32.
  - `groupEp` and `groupTp` cannot have the same name.
  - C must be greater than 0 and less than or equal to the upper limit of the operator device memory. When `xShardType` is set to 1, C is an integer multiple of tp.
  - MC2 operators cannot be called concurrently, nor can different MC2 operators.
  - Cross-supernode operations are not supported.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

- <term>Atlas A3 training products/Atlas A3 inference products</term>:

```Cpp
#include <thread>
#include <iostream>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "aclnnop/aclnn_all_to_all_all_gather_batch_matmul.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while(0)

constexpr int EP_WORLD_SIZE = 4;
constexpr int TP_WORLD_SIZE = 2;
constexpr int DEV_NUM = EP_WORLD_SIZE * TP_WORLD_SIZE;

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

template<typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
        shape.data(), shape.size(), *deviceAddr);
    return 0;
}

struct Args {
    int rankId;
    HcclComm hcclEpComm;
    HcclComm hcclTpComm;
    aclrtStream stream;
    aclrtContext context;
};

int LaunchOneThreadAlltoAllAllGatherBmm(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret: %d\n", ret); return ret);
    char hcomEpName[128] = {0};
    ret = HcclGetCommName(args.hcclEpComm, hcomEpName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetEpCommName failed. ret: %d\n", ret); return -1);
    char hcomTpName[128] = {0};
    ret = HcclGetCommName(args.hcclTpComm, hcomTpName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetTpCommName failed. ret: %d\n", ret); return -1);
    LOG_PRINT("[INFO] rank = %d, hcomEpName = %s, hcomTpName = %s, stream = %p, context = %p\n", args.rankId,
    hcomEpName, hcomTpName, args.stream, args.context);
    
    int64_t E = 2 * EP_WORLD_SIZE;
    int64_t C = 2 * TP_WORLD_SIZE;
    int64_t H = 6 * TP_WORLD_SIZE;
    int64_t M = 6 * TP_WORLD_SIZE;
    int64_t xShardType = 1; // Can be set to 0 to enable the gather H-axis scenario.
    int64_t actType = 1;
    
    std::vector<int64_t> xShape;
    std::vector<int64_t> weightShape;
    std::vector<int64_t> biasShape;
    std::vector<int64_t> y1OutShape;
    std::vector<int64_t> y2OutShape;
    std::vector<int64_t> y3OutShape;
    
    if (xShardType == 1) {
        xShape = {E, C / TP_WORLD_SIZE, H};
        weightShape = {E / EP_WORLD_SIZE, H, M / TP_WORLD_SIZE};
        biasShape = {E / EP_WORLD_SIZE, 1, M / TP_WORLD_SIZE};
        y1OutShape = {E / EP_WORLD_SIZE, EP_WORLD_SIZE * TP_WORLD_SIZE * C / TP_WORLD_SIZE, M / TP_WORLD_SIZE};
        y2OutShape = {E / EP_WORLD_SIZE, EP_WORLD_SIZE * TP_WORLD_SIZE * C / TP_WORLD_SIZE, H};
        y3OutShape = {E / EP_WORLD_SIZE, EP_WORLD_SIZE * TP_WORLD_SIZE * C / TP_WORLD_SIZE, M / TP_WORLD_SIZE};
    } else if (xShardType == 0) {
        xShape = {E, C, H / TP_WORLD_SIZE};
        weightShape = {E / EP_WORLD_SIZE, H, M / TP_WORLD_SIZE};
        biasShape = {E / EP_WORLD_SIZE, 1, M / TP_WORLD_SIZE};
        y1OutShape = {E / EP_WORLD_SIZE, EP_WORLD_SIZE * C, M / TP_WORLD_SIZE};
        y2OutShape = {E / EP_WORLD_SIZE, EP_WORLD_SIZE * C, H};
        y3OutShape = {E / EP_WORLD_SIZE, EP_WORLD_SIZE * C, M / TP_WORLD_SIZE};
    } else {
        LOG_PRINT("[ERROR] unsupported xShardType = %ld.\n", xShardType);
        return -1;
    }

    void *xDeviceAddr = nullptr;
    void *weightDeviceAddr = nullptr;
    void *biasDeviceAddr = nullptr;
    void *y1OutDeviceAddr = nullptr;
    void *y2OutDeviceAddr = nullptr;
    void *y3OutDeviceAddr = nullptr;
    aclTensor *x = nullptr;
    aclTensor *weight = nullptr;
    aclTensor *bias = nullptr;
    aclTensor *y1Out = nullptr;
    aclTensor *y2Out = nullptr;
    aclTensor *y3Out = nullptr;

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;

    long long xShapeSize = GetShapeSize(xShape);
    long long weightShapeSize = GetShapeSize(weightShape);
    long long biasShapeSize = GetShapeSize(biasShape);
    long long y1OutShapeSize = GetShapeSize(y1OutShape);
    long long y2OutShapeSize = GetShapeSize(y2OutShape);
    long long y3OutShapeSize = GetShapeSize(y3OutShape);
    
    std::vector<int16_t> xHostData(xShapeSize, 1);
    std::vector<int16_t> weightHostData(weightShapeSize, 2);
    std::vector<int16_t> biasHostData(biasShapeSize, 3);
    std::vector<int16_t> y1OutHostData(y1OutShapeSize, 0);
    std::vector<int16_t> y2OutHostData(y2OutShapeSize, 0);
    std::vector<int16_t> y3OutHostData(y3OutShapeSize, 0);

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(y1OutHostData, y1OutShape, &y1OutDeviceAddr, aclDataType::ACL_FLOAT16, &y1Out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(y2OutHostData, y2OutShape, &y2OutDeviceAddr, aclDataType::ACL_FLOAT16, &y2Out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(y3OutHostData, y3OutShape, &y3OutDeviceAddr, aclDataType::ACL_FLOAT16, &y3Out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Call the first-phase API.
    ret = aclnnAlltoAllAllGatherBatchMatMulGetWorkspaceSize(x, weight, bias, hcomEpName, hcomTpName, EP_WORLD_SIZE,
        TP_WORLD_SIZE, xShardType, actType, y1Out, y2Out, y3Out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] aclnnAlltoAllAllGatherBatchMatMulGetWorkspaceSize failed. ret = %d \n", ret); return ret);
    // Allocate device memory based on the computed workspaceSize.
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
    }
    // Call the second-phase API.
    ret = aclnnAlltoAllAllGatherBatchMatMul(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnAlltoAllAllGatherBatchMatMul failed. ret = %d \n", ret);
        return ret);
    // (Fixed writing) Wait until the task execution is complete.
    ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
        return ret);
    LOG_PRINT("[INFO] device_%d aclnnAlltoAllAllGatherBatchMatMul execute successfully.\n", args.rankId);
    HcclCommDestroy(args.hcclEpComm);
    HcclCommDestroy(args.hcclTpComm);
    // Release device resources. Modify the configuration based on the API definition.
    if (x != nullptr) {
        aclDestroyTensor(x);
    }
    if (weight != nullptr) {
        aclDestroyTensor(weight);
    }
    if (bias != nullptr) {
        aclDestroyTensor(bias);
    }
    if (y1Out != nullptr) {
        aclDestroyTensor(y1Out);
    }
    if (y2Out != nullptr) {
        aclDestroyTensor(y2Out);
    }
    if (y3Out != nullptr) {
        aclDestroyTensor(y3Out);
    }
    if (xDeviceAddr != nullptr) {
        aclrtFree(xDeviceAddr);
    }
    if (weightDeviceAddr != nullptr) {
        aclrtFree(weightDeviceAddr);
    }
    if (biasDeviceAddr != nullptr) {
        aclrtFree(biasDeviceAddr);
    }
    if (y1OutDeviceAddr != nullptr) {
        aclrtFree(y1OutDeviceAddr);
    }
    if (y2OutDeviceAddr != nullptr) {
        aclrtFree(y2OutDeviceAddr);
    }
    if (y3OutDeviceAddr != nullptr) {
        aclrtFree(y3OutDeviceAddr);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    ret = aclrtDestroyStream(args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtDestroyStream failed. ret = %d \n", ret); return ret);
    ret = aclrtResetDevice(args.rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtResetDevice failed. ret = %d \n", ret); return ret);
    return 0;
}

int main(int argc, char *argv[])
{
    // This sample is based on and can only run on Atlas A3.
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d \n", ret); return ret);
    aclrtStream stream[DEV_NUM];
    aclrtContext context[DEV_NUM];
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
    ret = aclrtSetDevice(rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);
    ret = aclrtCreateContext(&context[rankId], rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d \n", ret); return ret);
    ret = aclrtCreateStream(&stream[rankId]);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d \n", ret); return ret);
    }

    int32_t devicesEp[DEV_NUM];
    int32_t devicesTp[DEV_NUM];

    //Initialize the EP domain. ep = 4 {0,2,4,6} {1,3,5,7}
    HcclComm commsEp[DEV_NUM];
    for (int i = 0; i < TP_WORLD_SIZE; i++) {
        for (int j =0; j < EP_WORLD_SIZE; j++) {
            devicesEp[j + i * EP_WORLD_SIZE] = i + j * TP_WORLD_SIZE;
        }
        ret = HcclCommInitAll(EP_WORLD_SIZE, &devicesEp[i * EP_WORLD_SIZE], &commsEp[i * EP_WORLD_SIZE]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll ep world %d failed. ret = %d \n", i, ret);
            return ret);
    }

    //Initialize the TP domain. tp = 4 {0,1},{2,3},{4,5},{6,7}
    HcclComm commsTp[DEV_NUM];
    for (int i = 0; i < EP_WORLD_SIZE; i++) {
        for (int j =0; j < TP_WORLD_SIZE; j++) {
            devicesTp[j + i * TP_WORLD_SIZE] = j + i * TP_WORLD_SIZE;
        }
        ret = HcclCommInitAll(TP_WORLD_SIZE, &devicesTp[i * TP_WORLD_SIZE], &commsTp[i * TP_WORLD_SIZE]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll tp world %d failed. ret = %d \n", i, ret);
            return ret);
    }

    Args args[DEV_NUM];
    // Start multiple threads.
    std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].hcclEpComm = commsEp[rankId % TP_WORLD_SIZE * EP_WORLD_SIZE + rankId / TP_WORLD_SIZE];
        // TP initialization in sequence. The device ID pairs are [0,4], [1,5], ..., [3,7].
        args[rankId].hcclTpComm = commsTp[rankId];
        std::cout << "test devices id " << rankId << " = " << devicesTp[rankId] << std::endl;
        if (rankId == (DEV_NUM - 1)) {
            args[rankId].hcclTpComm = commsTp[rankId];
        }
        args[rankId].stream = stream[rankId];
        args[rankId].context = context[rankId];
        threads[rankId].reset(new std::thread(&LaunchOneThreadAlltoAllAllGatherBmm, std::ref(args[rankId])));
    }
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        threads[rankId]->join();
    }

    aclFinalize();
    return 0;
}
```
