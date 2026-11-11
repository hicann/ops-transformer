# aclnnAlltoAllvGroupedMatMulV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T13:04:01.252Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
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

- Description: Completes fusion of routing expert AlltoAllv, Permute, and GroupedMatMul, and implements parallel fusion with shared expert MatMul. **Communication before computation**.

- Formula:
    - Routed experts:

    $$
    ataOut = AlltoAllv(gmmX) \\
    permuteOut = Permute(ataOut) \\
    gmmY = permuteOut \times gmmWeight
    $$

    - Shared expert:

    $$
    mmY = mmX \times mmWeight
    $$

Compared with the `aclnnAlltoAllvGroupedMatMul` API, this API has the following changes:

- The `commMode` parameter is added. Use it to specify the communication engine used by the chip.

  <!-- npu="910b" id12 -->
  - <term>Atlas A2 products</term>: `aiv` is supported.
  <!-- end id12 -->
  <!-- npu="A3" id13 -->
  - <term>Atlas A3 products</term>: `aiv` and `ai_cpu` are supported.
  <!-- end id13 -->
  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>: `ai_cpu` and `ccu` are supported.

  <!-- end id7 -->

## Function Prototype

Each operator is provided as a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the `aclnnAlltoAllvGroupedMatMulV2GetWorkspaceSize` API to obtain the input parameters and calculate the required workspace size based on the computation process, and then call the `aclnnAlltoAllvGroupedMatMulV2` API to perform the computation.

```cpp
aclnnStatus aclnnAlltoAllvGroupedMatMulV2GetWorkspaceSize(
    const aclTensor*   gmmX,
    const aclTensor*   gmmWeight,
    const aclTensor*   sendCountsTensorOptional,
    const aclTensor*   recvCountsTensorOptional,
    const aclTensor*   mmXOptional,
    const aclTensor*   mmWeightOptional,
    const char*        group,
    const char*        commMode,
    int64_t            epWorldSize,
    const aclIntArray* sendCounts,
    const aclIntArray* recvCounts,
    bool               transGmmWeight,
    bool               transMmWeight,
    bool               permuteOutFlag,
    aclTensor*         gmmY,
    aclTensor*         mmYOptional,
    aclTensor*         permuteOutOptional,
    uint64_t*          workspaceSize,
    aclOpExecutor**    executor)
```

```cpp
aclnnStatus aclnnAlltoAllvGroupedMatMulV2(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
```

## aclnnAlltoAllvGroupedMatMulV2GetWorkspaceSize

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
        <td>gmmX(aclTensor*)</td>
        <td>Input.</td>
        <td>This input undergoes AlltoAllv communication and Permute operations, and the result serves as the left matrix of the GroupedMatMul computation.</td>
        <td>Supports 2 dimensions with shape (BSK, H1).</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>x</td>
    </tr>
    <tr>
        <td>gmmWeight(aclTensor*)</td>
        <td>Input.</td>
        <td>Right matrix of the GroupedMatMul computation.</td>
        <td>Supports 3 dimensions, with shape (e, H1, N1).</td>
        <td>Keep consistent with gmmX</td>
        <td>ND</td>
        <td>3</td>
        <td>√ (applicable only to the transpose use case)</td>
    </tr>
    <tr>
        <td>sendCountsTensorOptional(aclTensor*)</td>
        <td>Input</td>
        <td>Reserved parameter. Only nullptr is supported in the current version.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>recvCountsTensorOptional(aclTensor*)</td>
        <td>Input</td>
        <td>Reserved parameter. Only nullptr is supported in the current version.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>mmXOptional(aclTensor*)</td>
        <td>Input</td>
        <td>Optional input, left matrix in the shared expert MatMul computation.</td>
        <td>Supports 2 dimensions with shape (BS, H2). Must be passed together with mmWeightOptional or both set to nullptr.</td>
        <td>Keep consistent with gmmX</td>
        <td>ND</td>
        <td>2</td>
        <td>x</td>
    </tr>
    <tr>
        <td>mmWeightOptional(aclTensor*)</td>
        <td>Input</td>
        <td>Optional input, the right matrix in the shared expert MatMul computation.</td>
        <td>Supports 2 dimensions, with shape (H2, N2). Must be passed in together with mmXOptional or both set to nullptr.</td>
        <td>Keep consistent with gmmX</td>
        <td>ND</td>
        <td>2</td>
        <td>√ (transpose use case only)</td>
    </tr>
    <tr>
        <td>group(char*)</td>
        <td>Input</td>
        <td>Communication domain name for expert parallelism, with a string length in the range (0, 128).</td>
        <td>Obtained through the Hccl API "extern HcclResult HcclGetCommName(HcclComm comm, char* commName);", where commName is the group.</td>
        <td>STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commMode(char*)</td>
        <td>Input</td>
        <td>Specifies the current communication type.</td>
        <td><term>Atlas A2 products</term> support input of "aiv";<br><term>Atlas A3 products</term> support input of "aiv" and "ai_cpu";<br><term>Ascend 950PR&950DT products</term> supports input of "ai_cpu" and "ccu".</td>
        <td>STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>epWorldSize (int64_t)</td>
        <td>Input</td>
        <td>Size of the EP communicator.</td>
        <td><term>Atlas A2 products</term> support 2, 4, and 8 when commMode is "aiv";<br><term>Atlas A3 products</term> support 2, 4, 8, 16, 32, 64, and 128;<br><term>Ascend 950PR&950DT products</term> supports 2, 4, 8, 16, 32, and 64.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>sendCounts(aclIntArray*)</td>
        <td>Input</td>
        <td>Indicates the number of tokens sent to other devices.</td>
        <td>Supports the INT64 data type. The length is e * epWorldSize. The maximum is 1024 when commMode is "aiv" and 256 in other modes. The input type must be a list.</td>
        <td>aclIntArray* (element type INT64)</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>recvCounts(aclIntArray*)</td>
        <td>Input</td>
        <td>Indicates the number of tokens received from other devices.</td>
        <td>The supported data type is INT64, with a length of e * epWorldSize. The maximum value is 1024 when commMode is "aiv", and 256 for other modes. The input type must be list.</td>
        <td>aclIntArray* (element type INT64)</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>transGmmWeight(bool)</td>
        <td>Input</td>
        <td>Whether the right matrix of GroupedMatMul needs to be transposed.</td>
        <td>true indicates that transpose is required, and false indicates no transpose.</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>transMmWeight(bool)</td>
        <td>Input</td>
        <td>Whether the right matrix of the shared expert MatMul needs to be transposed.</td>
        <td>true indicates that transpose is required, and false indicates no transpose.</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>permuteOutFlag(bool)</td>
        <td>Input</td>
        <td>Whether output to permuteOutOptional is required.</td>
        <td>true indicates that output is required, and false indicates that output is not required.</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>gmmY(aclTensor*)</td>
        <td>Output</td>
        <td>Output computed by the routed experts.</td>
        <td>Supports 2 dimensions, shape (A, N1).</td>
        <td>Keep consistent with gmmX</td>
        <td>ND</td>
        <td>2</td>
        <td>x</td>
    </tr>
    <tr>
        <td>mmYOptional(aclTensor*)</td>
        <td>Output</td>
        <td>Output of the shared expert computation.</td>
        <td>Supports 2 dimensions, with shape (BS, N2). Output is generated only when mmXOptional and mmWeightOptional are passed.</td>
        <td>Keep consistent with mmXOptional</td>
        <td>ND</td>
        <td>2</td>
        <td>x</td>
    </tr>
    <tr>
        <td>permuteOutOptional(aclTensor*)</td>
        <td>Output</td>
        <td>Output after permute.</td>
        <td>Supports 2 dimensions, shape (A, H1). Output only when permuteOutFlag is true.</td>
        <td>Keep consistent with gmmX</td>
        <td>ND</td>
        <td>2</td>
        <td>x</td>
    </tr>
    <tr>
        <td>workspaceSize (uint64_t*)</td>
        <td>Output</td>
        <td>Returns the size of the workspace to be side-applied on the Device.</td>
        <td>-</td>
        <td>UINT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>executor(aclOpExecutor**)</td>
        <td>Output</td>
        <td>Returns the op executor, which contains the computation process of the operator.</td>
        <td>-</td>
        <td>aclOpExecutor*</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    </tbody></table>

- **Return Value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API performs input parameter validation and returns an error in the following scenarios:

    <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
    <col style="width: 282px">
    <col style="width: 120px">
    <col style="width: 747px">
    </colgroup>
    <thead>
    <tr>
    <th>Return Value</th>
    <th>Error Code</th>
    <th>Description</th>
    </tr></thead>
    <tbody>
    <tr>
    <td>ACLNN_ERR_PARAM_NULLPTR</td>
    <td>161001</td>
    <td>The parameter is a required input, output, or attribute, but a null pointer is passed.</td>
    </tr>
    <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
    <td>The data type, data format, or dimension of gmmX, gmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional, group, epWorldSize, sendCounts, or recvCounts is not supported.</td>
    </tr>
    </tbody></table>

## aclnnAlltoAllvGroupedMatMulV2

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 168px">
    <col style="width: 128px">
    <col style="width: 854px">
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
    <td>Workspace memory address side-applied on the Device.</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>Input</td>
    <td>Workspace size side-applied on the Device, obtained by the first-phase API <code>aclnnAlltoAllvGroupedMatMulV2GetWorkspaceSize</code>.</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>Input</td>
    <td>op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
    <td>stream</td>
    <td>Input</td>
    <td>Stream used for task execution.</td>
    </tr>
    </tbody></table>

- **Return Value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Communication engine constraints:

  <!-- npu="910b" id14 -->
  - <term>Atlas A2 products</term>: Supports AIV communication; in AIV mode, epWorldSize supports 2, 4, and 8.
  <!-- end id14 -->
  <!-- npu="A3" id8 -->
  - <term>Atlas A3 products</term>: Supports AIV and AICPU communication.
  <!-- end id8 -->
  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT products</term>: Supports CCU communication and AICPU communication. CCU supports only interconnection within the UB domain of a single node, and AI_CPU supports cross-node interconnection within the UB domain.

  <!-- end id9 -->

- Deterministic computation:
  - aclnnAlltoAllvGroupedMatMulV2 defaults to a deterministic implementation.

- Variables used in shape in the parameter description:
  - BSK: the number of tokens sent by the current card, which is the cumulative sum of the sendCounts parameter. Value range: (0, 52428800).
  - H1: Indicates the hidden size of the routed expert. Value range: (0, 65536).
  - H2: Indicates the hidden size of the shared expert. Value range: (0, 12288].
  - e: number of experts per card, e ≤ 48, and e * epWorldSize supports up to 384.
  - N1: Indicates the head_num of the routed expert. Value range: (0, 65536).
  - N2: Indicates the head_num of the shared expert. Value range: (0, 65536).
  - BS: batch sequence size.
  - K: indicates the number of TopK experts selected. K ranges from [2, 8].
  - A: the number of tokens received by the current card, which is the cumulative sum of the recvCounts parameter.
  - The total number of received tokens in the ep communicator (the sum of A across all ranks) must equal the total number of sent tokens (the sum of BSK across all ranks). In addition, each rank's `sendCounts` and its peers' `recvCounts` must be transposes of each other along the source and destination rank dimensions, with the expert dimension unchanged.

- Additional constraints and memory layout when commMode is `"aiv"`:
  - For Atlas A2, epWorldSize supports 2, 4, and 8; for Atlas A3, epWorldSize supports 2, 4, 8, 16, 32, 64, and 128. gmmX, gmmWeight, and gmmY must be of the same FLOAT16 or BFLOAT16 data type and ND format, and their shapes are 2D, 3D, and 2D respectively.
  - In AIV mode, the value range of BSK/A is [1, 5000000] and that of H1/N1 is [1, 65535]. The number of local experts e does not exceed 512, and `epWorldSize * e` does not exceed 1024.
  - The sendCounts and recvCounts API attributes are INT64 arrays of direct counts, with the layout of `[rank][localExpert]`. The host side verifies that each item is non-negative and does not exceed the corresponding local input/output M, that the sum of sendCounts equals BSK, and that the sum of recvCounts equals A.

<!-- npu="A3" id10 -->
- <term>Atlas A3 products</term>  : Performance degradation may occur when the communication volume of a single device is less than 2MB.

<!-- end id10 -->

## Example

The sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

Description:

- The sample code calls some HCCL collective communication library APIs, such as HcclGetCommName, HcclCommInitAll, and HcclCommDestroy. For details, see [<<HCCL API (C)>>](https://hiascend.com/en/document/redirect/CannCommunityHcclCppApi).
- The sample code uses 8 devices as an example. Modify `EP_WORLD_SIZE` based on the actual number of devices.

<!-- npu="950" id11 -->
- <term>Ascend 950PR&950DT products</term>:

    ```cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <cstring>
    #include <vector>
    #include "acl/acl.h"
    #include "hccl/hccl.h"
    #include "aclnnop/aclnn_allto_allv_grouped_mat_mul_v2.h"

    #define CHECK_RET(cond, return_expr)                                                                                   \
        do {                                                                                                               \
            if (!(cond)) {                                                                                                 \
                return_expr;                                                                                               \
            }                                                                                                              \
        } while (0)

    #define LOG_PRINT(message, ...)                                                                                        \
        do {                                                                                                               \
            printf(message, ##__VA_ARGS__);                                                                                \
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
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
        return 0;
    }

    struct Args {
        int rankId;
        HcclComm hcclComm;
        aclrtStream stream;
        aclrtContext context;
    };

    // Basic shape information
    constexpr int64_t EP_WORLD_SIZE = 2;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    constexpr int64_t H = 7168;
    constexpr int64_t e = 4;
    constexpr int64_t N1 = 4096;
    constexpr int64_t N2 = 4096;
    constexpr int64_t A = BS * K;

    std::vector<int16_t> pPermuteData(A *H, 0);
    std::vector<int16_t> pGmmYData(A *N1, 0);
    std::vector<int16_t> pmmXData(BS *H, 0);
    std::vector<int16_t> pmmWData(H *N2, 0);
    std::vector<int16_t> pmmYData(BS *N2, 0);

    int LaunchOneThreadAlltoAllvGmm(Args &args)
    {
        int ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret: %d\n", ret); return ret);
        char hcomName[128] = {0};
        ret = HcclGetCommName(args.hcclComm, hcomName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret: %d\n", ret); return -1);

        std::vector<int64_t> gmmXShape = {BS * K, H};
        std::vector<int64_t> gmmWShape = {e, H, N1};
        std::vector<int64_t> gmmYShape = {A, N1};
        std::vector<int64_t> permuteShape = {A, H};
        std::vector<int64_t> mmXShape = {BS, H};
        std::vector<int64_t> mmWShape = {H, N2};
        std::vector<int64_t> mmYShape = {BS, N2};
        std::vector<int64_t> sendCountsShape = {EP_WORLD_SIZE * e};
        std::vector<int64_t> recvCountsShape = {EP_WORLD_SIZE * e};

        std::vector<int64_t> sendCountsList(EP_WORLD_SIZE * e, BS * K / (EP_WORLD_SIZE * e));
        std::vector<int64_t> recvCountsList(EP_WORLD_SIZE * e, BS * K / (EP_WORLD_SIZE * e));

        void *gmmXDeviceAddr = nullptr;
        void *gmmWDeviceAddr = nullptr;
        void *gmmYDeviceAddr = nullptr;
        void *permuteDeviceAddr = nullptr;
        void *mmXDeviceAddr = nullptr;
        void *mmWDeviceAddr = nullptr;
        void *mmYDeviceAddr = nullptr;

        aclTensor *gmmX = nullptr;
        aclTensor *gmmW = nullptr;
        aclTensor *gmmY = nullptr;
        aclTensor *mmX = nullptr;
        aclTensor *mmW = nullptr;
        aclTensor *mmY = nullptr;
        aclTensor *permute = nullptr;
        aclTensor *sendCountsTensor = nullptr;
        aclTensor *recvCountsTensor = nullptr;

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

        std::vector<uint16_t> gmmXHostData(gmmXShapeSize, (args.rankId + 1) * 1024); // BF16, FP16
        std::vector<uint16_t> gmmWHostData(gmmWShapeSize, (args.rankId + 1) * 512);
        std::vector<uint16_t> gmmYHostData(gmmYShapeSize, 65535);
        std::vector<uint16_t> permuteHostData(permuteShapeSize, 65535);
        std::vector<uint16_t> mmXHostData(mmXShapeSize, (args.rankId + 1) * 1024); // BF16, FP16
        std::vector<uint16_t> mmWHostData(mmWShapeSize, (args.rankId + 1) * 512);
        std::vector<uint16_t> mmYHostData(mmYShapeSize, 0);

        ret = CreateAclTensor(gmmXHostData, gmmXShape, &gmmXDeviceAddr, aclDataType::ACL_FLOAT16, &gmmX);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gmmWHostData, gmmWShape, &gmmWDeviceAddr, aclDataType::ACL_FLOAT16, &gmmW);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gmmYHostData, gmmYShape, &gmmYDeviceAddr, aclDataType::ACL_FLOAT16, &gmmY);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(mmXHostData, mmXShape, &mmXDeviceAddr, aclDataType::ACL_FLOAT16, &mmX);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(mmWHostData, mmWShape, &mmWDeviceAddr, aclDataType::ACL_FLOAT16, &mmW);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(mmYHostData, mmYShape, &mmYDeviceAddr, aclDataType::ACL_FLOAT16, &mmY);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        ret = CreateAclTensor(permuteHostData, permuteShape, &permuteDeviceAddr, aclDataType::ACL_FLOAT16, &permute);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
        aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());

        // Call the first-phase API.
        ret = aclnnAlltoAllvGroupedMatMulV2GetWorkspaceSize(gmmX, gmmW, sendCountsTensor, recvCountsTensor, mmX, mmW,
                                                            hcomName, "ai_cpu", EP_WORLD_SIZE, sendCounts, recvCounts, false, false,
                                                            true, gmmY, mmY, permute, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("[ERROR] aclnnAlltoAllvGroupedMatMulV2GetWorkspaceSize failed. ret = %d \n", ret);
                return ret);

        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }

        // Call the second-phase API.
        ret = aclnnAlltoAllvGroupedMatMulV2(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnAlltoAllvGroupedMatMulV2 failed. ret = %d \n", ret);
                return ret);
        // (Boilerplate) Synchronously wait for the task to complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
                return ret);
        LOG_PRINT("[INFO] device_%d aclnnAlltoAllvGroupedMatMulV2 execute successfully.\n", args.rankId);
        // Release device resources. Modify based on the specific API definition.
        if (args.rankId == 0) {
            size_t size = A * H * sizeof(int16_t);
            aclrtMemcpy(pPermuteData.data(), size, permuteDeviceAddr, size, ACL_MEMCPY_DEVICE_TO_HOST);
        }
        if (args.rankId == 0) {
            size_t size = A * N1 * sizeof(int16_t);
            aclrtMemcpy(pGmmYData.data(), size, gmmYDeviceAddr, size, ACL_MEMCPY_DEVICE_TO_HOST);
        }
        if (gmmX != nullptr) {
            aclDestroyTensor(gmmX);
        }
        if (gmmW != nullptr) {
            aclDestroyTensor(gmmW);
        }
        if (gmmY != nullptr) {
            aclDestroyTensor(gmmY);
        }
        if (mmX != nullptr) {
            aclDestroyTensor(mmX);
        }
        if (mmW != nullptr) {
            aclDestroyTensor(mmW);
        }
        if (mmY != nullptr) {
            aclDestroyTensor(mmY);
        }
        if (permute != nullptr) {
            aclDestroyTensor(permute);
        }
        if (gmmXDeviceAddr != nullptr) {
            aclrtFree(gmmXDeviceAddr);
        }
        if (gmmWDeviceAddr != nullptr) {
            aclrtFree(gmmWDeviceAddr);
        }
        if (gmmYDeviceAddr != nullptr) {
            aclrtFree(gmmYDeviceAddr);
        }
        if (mmXDeviceAddr != nullptr) {
            aclrtFree(mmXDeviceAddr);
        }
        if (mmWDeviceAddr != nullptr) {
            aclrtFree(mmWDeviceAddr);
        }
        if (mmYDeviceAddr != nullptr) {
            aclrtFree(mmYDeviceAddr);
        }
        if (permuteDeviceAddr != nullptr) {
            aclrtFree(permuteDeviceAddr);
        }
        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
        HcclCommDestroy(args.hcclComm);
        aclrtDestroyStream(args.stream);
        aclrtDestroyContext(args.context);
        aclrtResetDevice(args.rankId);
        return 0;
    }

    int main(int argc, char *argv[])
    {
        int ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d \n", ret); return ret);
        aclrtStream stream[EP_WORLD_SIZE];
        aclrtContext context[EP_WORLD_SIZE];
        for (uint32_t rankId = 0; rankId < EP_WORLD_SIZE; rankId++) {
            ret = aclrtSetDevice(rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);
            ret = aclrtCreateContext(&context[rankId], rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d \n", ret); return ret);
            ret = aclrtCreateStream(&stream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d \n", ret); return ret);
        }

        int32_t devices[EP_WORLD_SIZE];
        for (int i = 0; i < EP_WORLD_SIZE; i++) {
            devices[i] = i;
        }
        // Initialize the communicator.
        HcclComm comms[EP_WORLD_SIZE];
        ret = HcclCommInitAll(EP_WORLD_SIZE, devices, comms);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll failed. ret = %d \n", ret); return ret);

        Args args[EP_WORLD_SIZE];
        // Start multiple threads.
        std::vector<std::unique_ptr<std::thread>> threads(EP_WORLD_SIZE);
        for (uint32_t rankId = 0; rankId < EP_WORLD_SIZE; rankId++) {
            args[rankId].rankId = rankId;
            args[rankId].hcclComm = comms[rankId];
            args[rankId].stream = stream[rankId];
            args[rankId].context = context[rankId];
            threads[rankId].reset(new std::thread(&LaunchOneThreadAlltoAllvGmm, std::ref(args[rankId])));
        }
        for (uint32_t rankId = 0; rankId < EP_WORLD_SIZE; rankId++) {
            threads[rankId]->join();
        }
        aclFinalize();
        return 0;
    }
    ```

<!-- end id11 -->
