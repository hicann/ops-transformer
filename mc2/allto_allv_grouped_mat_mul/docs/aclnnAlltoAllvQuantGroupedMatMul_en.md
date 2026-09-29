# aclnnAlltoAllvQuantGroupedMatMul

## Supported Products

| Product                                       | Supported|
| :------------------------------------------ | :------: |
| Ascend 950PR/Ascend 950DT series              |    √     |
| Atlas A3 training products/Atlas A3 inference products|    ×     |
| Atlas A2 training products/Atlas A2 inference products|    ×     |
| Atlas 200I/500 A2 inference products                 |    ×     |
| Atlas inference products                         |    ×     |
| Atlas training products                         |    ×     |

## Function

- **Operator function**: The AlltoAllv operator of the routing expert is fused with the GroupedMatMul operator, and the parallel fusion with the MatMul operator of the shared expert is implemented. **Communication is performed before computation**.

- **Formula**:
  - Routed experts:

    ```text
    permuteOut = AlltoAllv(gmmX)
    quantedPermuteOut = Quant(permuteOut, gmmXScale)
    quantedGmmWeight = Quant(gmmWeight, gmmWeightScale)
    gmmY = quantedPermuteOut @ quantedGmmWeight
    ```

  - Shared experts:

    ```text
    quantedMmX = Quant(mmX, mmXScale)
    quantedMmWeight = Quant(mmWeight, mmWeightScale)
    mmY = quantedMmX @ quantedMmWeight
    ```

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize` is called to obtain the input parameters and compute the required workspace size based on the process. Then, `aclnnAlltoAllvQuantGroupedMatMul` is called to perform computation.

```cpp
aclnnStatus aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize(
    const aclTensor*   gmmX,
    const aclTensor*   gmmWeight,
    const aclTensor*   gmmXScale,
    const aclTensor*   gmmWeightScale,
    const aclTensor*   gmmXOffsetOptional,
    const aclTensor*   gmmWeightOffsetOptional,
    const aclTensor*   sendCountsTensorOptional,
    const aclTensor*   recvCountsTensorOptional,
    const aclTensor*   mmXOptional,
    const aclTensor*   mmWeightOptional,
    const aclTensor*   mmXScaleOptional,
    const aclTensor*   mmWeightScaleOptional,
    const aclTensor*   mmXOffsetOptional,
    const aclTensor*   mmWeightOffsetOptional,
    int64_t            gmmXQuantMode,
    int64_t            gmmWeightQuantMode,
    int64_t            mmXQuantMode,
    int64_t            mmWeightQuantMode,
    const char*        group,
    int64_t            epWorldSize,
    const aclIntArray* sendCounts,
    const aclIntArray* recvCounts,
    bool               transGmmWeight,
    bool               transMmWeight,
    int64_t            groupSize,
    bool               permuteOutFlag,
    aclTensor*         gmmY,
    aclTensor*         mmYOptional,
    aclTensor*         permuteOutOptional,
    uint64_t*          workspaceSize,
    aclOpExecutor**    executor)
```

```cpp
aclnnStatus aclnnAlltoAllvQuantGroupedMatMul(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
```

## aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1392px"><colgroup>
    <col style="width: 120px">
    <col style="width: 120px">
    <col style="width: 160px">
    <col style="width: 150px">
    <col style="width: 80px">
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
    <td>gmmX</td>
    <td>Input</td>
    <td>The result of AlltoAllv communication is used as the left matrix for GroupedMatMul computation. The matrix supports two dimensions, and the shape is (BSK, H1).</td>
    <td>HIFLOAT8</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>gmmWeight</td>
    <td>Input</td>
    <td>The right matrix for GroupedMatMul computation. The data type is HIFLOAT8, and the matrix supports three dimensions, and the shape is (e, H1, N1).</td>
    <td>HIFLOAT8</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>gmmXScale</td>
    <td>Input</td>
    <td>Quantization coefficient of gmmX. When gmmXQuantMode is set to 1, the coefficient supports 1D and the shape is (1).</td>
    <td>FLOAT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>gmmWeightScale</td>
    <td>Input</td>
    <td>Quantization coefficient of gmmWeight. This parameter is mandatory when gmmWeightQuantMode is set to 1. The coefficient supports 1D and the shape is (1).</td>
    <td>FLOAT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>gmmXOffsetOptional</td>
    <td>Input</td>
    <td>Reserved parameter. In the current version, only nullptr can be passed.</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>gmmWeightOffsetOptional</td>
    <td>Input</td>
    <td>Reserved parameter. In the current version, only nullptr can be passed.</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sendCountsTensorOptional</td>
    <td>Input</td>
    <td>Reserved parameter. In the current version, only nullptr can be passed.</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>recvCountsTensorOptional</td>
    <td>Input</td>
    <td>Reserved parameter. In the current version, only nullptr can be passed.</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>mmXOptional</td>
    <td>Input</td>
    <td>Optional input, which is the left matrix shared by the expert MatMul computation. It must be passed together with mmWeightOptional or both of them are nullptr. The matrix supports 2D, and the shape is (BS, H2).</td>
    <td>HIFLOAT8</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>mmWeightOptional</td>
    <td>Input</td>
    <td>Optional input, which is the right matrix shared by the expert MatMul computation. It must be passed together with mmXOptional or both of them are nullptr. The matrix supports 2D, and the shape is (H2, N2).</td>
    <td>HIFLOAT8</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>mmXScaleOptional</td>
    <td>Input</td>
    <td>Quantization coefficient of mmX. This parameter is mandatory when mmXQuantMode is set to 1. It is 1D and the shape is (1).</td>
    <td>FLOAT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>mmWeightScaleOptional</td>
    <td>Input</td>
    <td>Quantization coefficient of mmWeight. This parameter is mandatory when mmWeightQuantMode is set to 1. It is 1D and the shape is (1).</td>
    <td>FLOAT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>mmXOffsetOptional</td>
    <td>Input</td>
    <td>This parameter is not supported in the current version. Set it to nullptr.</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>mmWeightOffsetOptional</td>
    <td>Input</td>
    <td>This parameter is not supported in the current version. Set it to nullptr.</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>gmmXQuantMode</td>
    <td>Input</td>
    <td>Quantization mode of gmmX. Only 1 is supported in the current version.</td>
    <td>INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>gmmWeightQuantMode</td>
    <td>Input</td>
    <td>Quantization mode of gmmWeight. Only 1 is supported in the current version.</td>
    <td>INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>mmXQuantMode</td>
    <td>Input</td>
    <td>Quantization mode of mmX. Only 1 is supported in the current version.</td>
    <td>INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>mmWeightQuantMode</td>
    <td>Input</td>
    <td>Quantization mode of mmWeight. Only 1 is supported in the current version.</td>
    <td>INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>group</td>
    <td>Input</td>
    <td>Communication domain name for expert parallelism (EP). The value is a string of (0, 128) characters.</td>
    <td>STRING</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>epWorldSize</td>
    <td>Input</td>
    <td>Size of the EP communication domain. Ascend 950PR/Ascend 950DT supports 2, 4, 8, 16, 32, and 64.</td>
    <td>INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>sendCounts</td>
    <td>Input</td>
    <td>Number of tokens sent to other cards. The data type is INT64, the length is e x epWorldSize, and the maximum value is 256. The input type must be list.</td>
    <td>aclIntArray* (with INT64 elements)</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>recvCounts</td>
    <td>Input</td>
    <td>Number of tokens received from other cards. The data type is INT64, the length is e x epWorldSize, and the maximum value is 256. The input type must be list.</td>
    <td>aclIntArray* (with INT64 elements)</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>transGmmWeight</td>
    <td>Input</td>
    <td>Whether to transpose the right matrix for GroupedMatMul. Set true to transpose, false to not transpose.</td>
    <td>BOOL</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>transMmWeight</td>
    <td>Input</td>
    <td>Whether to transpose the right matrix for shared expert MatMul. Set true to transpose, false to not transpose.</td>
    <td>BOOL</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>groupSize</td>
    <td>Input</td>
    <td>This parameter is not supported in the current version. Pass nullptr.</td>
    <td>INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>permuteOutFlag</td>
    <td>Input</td>
    <td>Whether permuteOutOptional should be generated. Set true to generate output, and false to not generate output.</td>
    <td>BOOL</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>gmmY</td>
    <td>Output</td>
    <td>Final computation result. The data type is the same as that of the input gmmX. The shape can be two-dimensional (A, N1).</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>mmYOptional</td>
    <td>Output</td>
    <td>Output of the shared expert MatMul. The data type is the same as that of mmXOptional. The shape can be two-dimensional (BS, N2). The output is generated only when both mmXOptional and mmWeightOptional are provided.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>permuteOutOptional</td>
    <td>Output</td>
    <td>Output after the Permute operation. The data type of this parameter is the same as that of gmmX. The output is generated only when permuteOutFlag is set to true.</td>
    <td>HIFLOAT8</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>Output</td>
    <td>Size of the workspace to be allocated on the device.</td>
    <td>UINT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>Output</td>
    <td>Operator executor, containing the operator computation process.</td>
    <td>aclOpExecutor*</td>
    <td>ND</td>
    </tr>
    </tbody></table>

- The enumerated values of gmmXQuantMode, gmmWeightQuantMode, mmXQuantMode, and mmWeightQuantMode are related to the quantization modes as follows:
  - 0: no quantization
  - 1: pertensor
  - 2: perchannel
  - 3: pertoken
  - 4: pergroup
  - 5: perblock
  - 6: mx quantization
  - 7: per-token dynamic quantization

- **Returns**
    
    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md). The first-phase API implements input parameter verification. The following errors may be thrown.

    <table style="undefined;table-layout: fixed; width: 1180px"><colgroup>
    <col style="width: 250px">
    <col style="width: 130px">
    <col style="width: 800px">
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
    <td>A null pointer is passed in instead of the required input, output, or attribute.</td>
    </tr>
    <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
    The data type, format, or dimension of <td>gmmX, gmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional, group, epWorldSize, sendCounts or recvCounts is not supported.</td>
    </tr>
    </tbody></table>

## aclnnAlltoAllvQuantGroupedMatMul

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1180px"><colgroup>
    <col style="width: 250px">
    <col style="width: 130px">
    <col style="width: 800px">
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
    <td>Memory address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>Input</td>
    <td>Workspace size allocated on the device, which is obtained by the first API call `aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize`.</td>
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

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - The default deterministic implementation is used in `aclnnAlltoAllvQuantGroupedMatMul`.

- Shape variables used in parameter descriptions:
  - BSK: Number of tokens sent by the local device, which is the sum of the sendCounts parameters. The value range is (0, 52428800).
  - `H1`: Hidden layer size of the routed experts. The value range is (0, 65536).
  - `H2`: Hidden layer size of the shared experts. The value range is (0, 12288].
  - e: indicates the number of experts on a single device. The value range is (0, 32]. The maximum value of e x epWorldSize is 256.
  - `N1`: `head_num` for routed experts. The value range is (0, 65536).
  - `N2`: `head_num` for shared experts. The value range is (0, 65536).
  - `BS`: Batch sequence size.
  - `K`: Number of experts selected via Top-K. The value range is [2, 8].
  - A: Number of tokens received by the local device, which is the sum of the recvCounts parameters.
  - The sum of the `A` parameters across all devices in the EP communication domain equals the sum of the `BSK` parameters across all devices.

- Quantization parameter constraints:
  - The current version supports only per-tensor quantization.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

Note: The quantization API supports only Ascend 950PR/Ascend 950DT series. The following example is implemented based on this series.

```cpp
#include <thread>
#include <iostream>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "aclnnop/aclnn_allto_allv_quant_grouped_mat_mul.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)           \
    do {                                  \
        printf(message, ##__VA_ARGS__);   \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

template <typename T>
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
    *tensor = aclCreateTensor(shape.data(),
        shape.size(),
        dataType,
        strides.data(),
        0,
        aclFormat::ACL_FORMAT_ND,
        shape.data(),
        shape.size(),
        *deviceAddr);
    return 0;
}

struct Args {
    int rankId;
    HcclComm hcclComm;
    aclrtStream stream;
    aclrtContext context;
};

// Basic shape information.
constexpr int64_t EP_WORLD_SIZE = 8;
constexpr int64_t BS = 4096;
constexpr int64_t K = 2;
constexpr int64_t H = 7168;
constexpr int64_t e = 4;
constexpr int64_t N1 = 4096;
constexpr int64_t N2 = 4096;
constexpr int64_t A = BS * K;

int LaunchOneThreadAlltoAllvQuantGroupedMatMul(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret: %d\n", ret); return ret);
    char hcomName[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetEpCommName failed. ret: %d\n", ret); return -1);

    std::vector<int64_t> gmmXShape = {BS * K, H};
    std::vector<int64_t> gmmWShape = {e, H, N1};
    std::vector<int64_t> gmmYShape = {A, N1};
    std::vector<int64_t> permuteShape = {A, H};
    std::vector<int64_t> mmXShape = {BS, H};
    std::vector<int64_t> mmWShape = {H, N2};
    std::vector<int64_t> mmYShape = {BS, N2};
    std::vector<int64_t> scaleShape = {1}; // scaling factor shape.

    std::vector<int64_t> sendCountsList(EP_WORLD_SIZE * e, BS * K / (EP_WORLD_SIZE * e));
    std::vector<int64_t> recvCountsList(EP_WORLD_SIZE * e, BS * K / (EP_WORLD_SIZE * e));

    void *gmmXDeviceAddr = nullptr;
    void *gmmWDeviceAddr = nullptr;
    void *gmmYDeviceAddr = nullptr;
    void *permuteDeviceAddr = nullptr;
    void *mmXDeviceAddr = nullptr;
    void *mmWDeviceAddr = nullptr;
    void *mmYDeviceAddr = nullptr;
    void *gmmXScaleDeviceAddr = nullptr;
    void *gmmWScaleDeviceAddr = nullptr;
    void *mmXScaleDeviceAddr = nullptr;
    void *mmWScaleDeviceAddr = nullptr;

    aclTensor *gmmX = nullptr;
    aclTensor *gmmW = nullptr;
    aclTensor *gmmY = nullptr;
    aclTensor *mmX = nullptr;
    aclTensor *mmW = nullptr;
    aclTensor *mmY = nullptr;
    aclTensor *permute = nullptr;
    aclTensor *gmmXScale = nullptr;
    aclTensor *gmmWScale = nullptr;
    aclTensor *mmXScale = nullptr;
    aclTensor *mmWScale = nullptr;

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;

    long long gmmXShapeSize = GetShapeSize(gmmXShape);
    long long gmmWShapeSize = GetShapeSize(gmmWShape);
    long long gmmYShapeSize = GetShapeSize(gmmYShape);
    long long permuteShapeSize = GetShapeSize(permuteShape);
    long long mmXShapeSize = GetShapeSize(mmXShape);
    long long mmWShapeSize = GetShapeSize(mmWShape);
    long long mmYShapeSize = GetShapeSize(mmYShape);

    // HIFLOAT8 data (simulated using uint8_t)
    std::vector<uint8_t> gmmXHostData(gmmXShapeSize, (args.rankId + 1) * 10);
    std::vector<uint8_t> gmmWHostData(gmmWShapeSize, (args.rankId + 1) * 5);
    std::vector<uint8_t> mmXHostData(mmXShapeSize, (args.rankId + 1) * 10);
    std::vector<uint8_t> mmWHostData(mmWShapeSize, (args.rankId + 1) * 5);
    
    // Output data (FLOAT16/BFLOAT16)
    std::vector<uint16_t> gmmYHostData(gmmYShapeSize, 65535);
    std::vector<uint16_t> mmYHostData(mmYShapeSize, 0);
    std::vector<uint8_t> permuteHostData(permuteShapeSize, 255);
    
    // Scaling factor data (FLOAT32)
    std::vector<float> gmmXScaleHostData(1, 1.0f);
    std::vector<float> gmmWScaleHostData(1, 1.0f);
    std::vector<float> mmXScaleHostData(1, 1.0f);
    std::vector<float> mmWScaleHostData(1, 1.0f);

    // Create a tensor.
    ret = CreateAclTensor(gmmXHostData, gmmXShape, &gmmXDeviceAddr, ACL_HIFLOAT8, &gmmX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gmmWHostData, gmmWShape, &gmmWDeviceAddr, ACL_HIFLOAT8, &gmmW);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gmmYHostData, gmmYShape, &gmmYDeviceAddr, ACL_FLOAT16, &gmmY);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(mmXHostData, mmXShape, &mmXDeviceAddr, ACL_HIFLOAT8, &mmX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(mmWHostData, mmWShape, &mmWDeviceAddr, ACL_HIFLOAT8, &mmW);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(mmYHostData, mmYShape, &mmYDeviceAddr, ACL_FLOAT16, &mmY);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(permuteHostData, permuteShape, &permuteDeviceAddr, ACL_HIFLOAT8, &permute);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gmmXScaleHostData, scaleShape, &gmmXScaleDeviceAddr, ACL_FLOAT32, &gmmXScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gmmWScaleHostData, scaleShape, &gmmWScaleDeviceAddr, ACL_FLOAT32, &gmmWScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(mmXScaleHostData, scaleShape, &mmXScaleDeviceAddr, ACL_FLOAT32, &mmXScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(mmWScaleHostData, scaleShape, &mmWScaleDeviceAddr, ACL_FLOAT32, &mmWScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());

    // Call the first-phase API.
    ret = aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize(gmmX,
        gmmW,
        gmmXScale,
        gmmWScale,
        nullptr, // gmmXOffsetOptional
        nullptr, // gmmWeightOffsetOptional
        nullptr, // sendCountsTensorOptional
        nullptr, // recvCountsTensorOptional
        mmX,
        mmW,
        mmXScale,
        mmWScale,
        nullptr, // mmXOffsetOptional
        nullptr, // mmWeightOffsetOptional
        1, // gmmXQuantMode
        1, // gmmWeightQuantMode
        1, // mmXQuantMode
        1, // mmWeightQuantMode
        hcomName,
        EP_WORLD_SIZE,
        sendCounts,
        recvCounts,
        false, // transGmmWeight
        false, // transMmWeight
        true,  // permuteOutFlag
        gmmY,
        mmY,
        permute,
        &workspaceSize,
        &executor);
    CHECK_RET(
        ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize failed. ret = %d \n", ret);
        return ret);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
    }

    // Call the second-phase API.
    ret = aclnnAlltoAllvQuantGroupedMatMul(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnAlltoAllvQuantGroupedMatMul failed. ret = %d \n", ret);
            return ret);
    
    // Wait until the task execution is complete.
    ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000000);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret); 
            return ret);

    // Free device resources.
    if (gmmX != nullptr) aclDestroyTensor(gmmX);
    if (gmmW != nullptr) aclDestroyTensor(gmmW);
    if (gmmY != nullptr) aclDestroyTensor(gmmY);
    if (mmX != nullptr) aclDestroyTensor(mmX);
    if (mmW != nullptr) aclDestroyTensor(mmW);
    if (mmY != nullptr) aclDestroyTensor(mmY);
    if (permute != nullptr) aclDestroyTensor(permute);
    if (gmmXScale != nullptr) aclDestroyTensor(gmmXScale);
    if (gmmWScale != nullptr) aclDestroyTensor(gmmWScale);
    if (mmXScale != nullptr) aclDestroyTensor(mmXScale);
    if (mmWScale != nullptr) aclDestroyTensor(mmWScale);
    
    if (gmmXDeviceAddr != nullptr) aclrtFree(gmmXDeviceAddr);
    if (gmmWDeviceAddr != nullptr) aclrtFree(gmmWDeviceAddr);
    if (gmmYDeviceAddr != nullptr) aclrtFree(gmmYDeviceAddr);
    if (mmXDeviceAddr != nullptr) aclrtFree(mmXDeviceAddr);
    if (mmWDeviceAddr != nullptr) aclrtFree(mmWDeviceAddr);
    if (mmYDeviceAddr != nullptr) aclrtFree(mmYDeviceAddr);
    if (permuteDeviceAddr != nullptr) aclrtFree(permuteDeviceAddr);
    if (gmmXScaleDeviceAddr != nullptr) aclrtFree(gmmXScaleDeviceAddr);
    if (gmmWScaleDeviceAddr != nullptr) aclrtFree(gmmWScaleDeviceAddr);
    if (mmXScaleDeviceAddr != nullptr) aclrtFree(mmXScaleDeviceAddr);
    if (mmWScaleDeviceAddr != nullptr) aclrtFree(mmWScaleDeviceAddr);
    if (workspaceSize > 0) aclrtFree(workspaceAddr);
    
    HcclCommDestroy(args.hcclComm);
    aclrtDestroyStream(args.stream);
    aclrtDestroyContext(args.context);
    aclrtResetDevice(args.rankId);
    return 0;
}

int main(int argc, char *argv[])
{
    // This example is implemented based on Ascend 950PR/Ascend 950DT.
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d \n", ret); return ret);
    
    constexpr uint32_t WORLD_SIZE = EP_WORLD_SIZE;
    aclrtStream stream[WORLD_SIZE];
    aclrtContext context[WORLD_SIZE];
    
    for (uint32_t rankId = 0; rankId < WORLD_SIZE; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);
        ret = aclrtCreateContext(&context[rankId], rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d \n", ret); return ret);
        ret = aclrtCreateStream(&stream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d \n", ret); return ret);
    }

    int32_t devices[WORLD_SIZE];
    for (int i = 0; i < WORLD_SIZE; i++) {
        devices[i] = i;
    }
    
    // Initialize the collective communication domain.
    HcclComm comms[WORLD_SIZE];
    ret = HcclCommInitAll(WORLD_SIZE, devices, comms);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll failed. ret = %d \n", ret); return ret);

    Args args[WORLD_SIZE];
    // Enable multi-threading.
    std::vector<std::unique_ptr<std::thread>> threads(WORLD_SIZE);
    for (uint32_t rankId = 0; rankId < WORLD_SIZE; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].hcclComm = comms[rankId];
        args[rankId].stream = stream[rankId];
        args[rankId].context = context[rankId];
        threads[rankId].reset(new std::thread(&LaunchOneThreadAlltoAllvQuantGroupedMatMul, std::ref(args[rankId])));
    }
    
    for (uint32_t rankId = 0; rankId < WORLD_SIZE; rankId++) {
        threads[rankId]->join();
    }
    
    aclFinalize();
    return 0;
}
```
