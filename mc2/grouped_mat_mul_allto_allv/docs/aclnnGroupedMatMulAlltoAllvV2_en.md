# aclnnGroupedMatMulAlltoAllvV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-10T10:05:51.427Z -->

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

- Description: Fuses routed expert GroupedMatMul, Unpermute, and AlltoAllv, and performs parallel fusion with the shared expert MatMul, **computing before communication**.

- The formula is as follows:

    - Routed experts:

    $$
    gmmY = gmmX \times gmmWeight \\
    unpermuteOut = Unpermute(gmmY) \\
    y = AlltoAllv(unpermuteOut)
    $$

    - Shared experts:

    $$
    mmY = mmX \times mmWeight
    $$

Compared with the `aclnnGroupedMatMulAlltoAllv` API, the changes to this API are as follows:

- Added the `commMode` parameter for users to specify the communication engine used by the chip.

  <!-- npu="A3" id7 -->
  - <term>Atlas A3 products</term>: supports `ai_cpu` and `aiv`.

  <!-- end id7 -->

  <!-- npu="910b" id8 -->
  - <term>Atlas A2 products</term>: only supports `aiv`; `ai_cpu` and `ccu` are not supported.

  <!-- end id8 -->

  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT products</term>: supports `ai_cpu` and `ccu`.

  <!-- end id9 -->

## Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call `aclnnGroupedMatMulAlltoAllvV2GetWorkspaceSize` to obtain the input parameters and calculate the required workspace size based on the computing flow, and then call `aclnnGroupedMatMulAlltoAllvV2` to execute the computation.

```cpp
aclnnStatus aclnnGroupedMatMulAlltoAllvV2GetWorkspaceSize(
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
    aclTensor*         y,
    aclTensor*         mmYOptional,
    uint64_t*          workspaceSize,
    aclOpExecutor**    executor)
```

```cpp
aclnnStatus aclnnGroupedMatMulAlltoAllvV2(
    void*           workspace,
    uint64_t        workspaceSize,
    aclOpExecutor*  executor,
    aclrtStream     stream)
```

## aclnnGroupedMatMulAlltoAllvV2

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1440px"> <colgroup>
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 200px">
    <col style="width: 250px">
    <col style="width: 150px">
    <col style="width: 180px">
    <col style="width: 180px">
    <col style="width: 180px">
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
    <td>Input</td>
    <td>Left matrix for the GroupedMatMul computation.</td>
    <td><ul><li>The data type must keep consistent with gmmWeight.</li><li>Supports two dimensions with shape (A, H1).</li></ul></td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
    </tr>
    <tr>
    <td>gmmWeight(aclTensor*)</td>
    <td>Input</td>
    <td>Right matrix for the GroupedMatMul computation.</td>
    <td><ul><li>Data type: keep consistent with gmmX.</li><li>Supports 3 dimensions, with a shape of (e, H1, N1).</li></ul></td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>3</td>
    <td>×</td>
    </tr>
    <tr>
    <td>sendCountsTensorOptional (aclTensor*)</td>
    <td>optional input</td>
    <td>Shape: (e * epWorldSize,).</td>
    <td>Not supported in the current version; pass nullptr.</td>
    <td>INT32, INT64</td>
    <td>ND</td>
    <td>1</td>
    <td>×</td>
    </tr>
    <tr>
    <td>recvCountsTensorOptional (aclTensor*)</td>
    <td>optional input</td>
    <td>The shape is (e * epWorldSize,).</td>
    <td>Not supported in the current version. Pass nullptr.</td>
    <td>INT32, INT64</td>
    <td>ND</td>
    <td>1</td>
    <td>×</td>
    </tr>
    <tr>
    <td>mmXOptional(aclTensor*)</td>
    <td>optional input</td>
    <td>Left matrix in the shared expert MatMul computation.</td>
    <td><ul><li>Must be passed in together with mmWeightOptional and mmYOptional, or both set to nullptr.</li><li>The data type must keep consistent with gmmX.</li><li>Supports 2 dimensions, with shape (BS, H2).</li></ul></td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
    </tr>
    <tr>
    <td>mmWeightOptional(aclTensor*)</td>
    <td>Optional input</td>
    <td>Right matrix in the shared expert MatMul computation.</td>
    <td><ul><li>Must be passed together with mmXOptional and mmYOptional, or all set to nullptr.</li><li>The data type must be consistent with that of gmmX.</li><li>Supports 2 dimensions, with the shape (H2, N2).</li></ul></td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
    </tr>
    <tr>
    <td>group(char*)</td>
    <td>Input</td>
    <td>Communication domain name for expert parallelism.</td>
    <td>The string length must be greater than 0 and less than 128.</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>commMode(char*)</td>
    <td>Input</td>
    <td>Specifies the current communication type.</td>
    <td><term>Atlas A2 products</term> supports only "aiv"; <term>Atlas A3 products</term> supports "ai_cpu" and "aiv"; <term>Ascend 950PR&950DT products</term> supports "ai_cpu" and "ccu".</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>epWorldSize (int64_t)</td>
    <td>Input</td>
    <td>Size of the ep communicator; -1 uses the actual size of group, while an explicit value must equal that size, which remains subject to the product constraints.</td>
    <td><ul><li><term>Atlas A2 products</term> supports 2, 4, and 8.</li><li><term>Atlas A3 products</term> supports 8, 16, 32, 64, and 128.</li><li><term>Ascend 950PR&950DT products</term> supports 2, 4, 8, 16, 32, and 64.</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sendCounts(aclIntArray*)</td>
    <td>Input</td>
    <td>Indicates the number of tokens sent to other devices.</td>
    <td><ul><li>The input type must be aclIntArray*, and the element type must be INT64.</li><li>The list size is e * epWorldSize, with a maximum of 1024 for AIV communication and 256 for other communication engines.</li></ul></td>
    <td>aclIntArray* (element type INT64)</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>recvCounts(aclIntArray*)</td>
    <td>Input</td>
    <td>Indicates the number of tokens received from other devices.</td>
    <td><ul><li>The input type must be aclIntArray*, and the element type must be INT64.</li><li>The list size is e * epWorldSize. The maximum is 1024 for AIV communication and 256 for other communication engines.</li></ul></td>
    <td>aclIntArray* (element type INT64)</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>transGmmWeight(bool)</td>
    <td>Input</td>
    <td>Whether gmmWeight needs to be transposed.</td>
    <td>true indicates transpose is required; false indicates transpose is not required.</td>
    <td>BOOL</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>transMmWeight(bool)</td>
    <td>Input</td>
    <td>Indicates whether the shared expert mmWeightOptional needs to be transposed.</td>
    <td>true indicates transpose is required; false indicates transpose is not required.</td>
    <td>BOOL</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>y(aclTensor*)</td>
    <td>Output</td>
    <td>Final computation result.</td>
    <td><ul><li>The data type must keep consistent with input gmmX.</li><li>Supports 2 dimensions with shape (BSK, N1).</li></ul></td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
    </tr>
    <tr>
    <td>mmYOptional(aclTensor*)</td>
    <td>Output</td>
    <td>Output of the shared expert MatMul.</td>
    <td><ul><li>The data type must keep consistent with that of mmXOptional.</li><li>Supports 2 dimensions, with shape (BS, N2).</li><li>Output only when both mmXOptional and mmWeightOptional are provided; otherwise, pass nullptr.</li></ul></td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
    </tr>
    <tr>
    <td>workspaceSize (uint64_t*)</td>
    <td>Output</td>
    <td>Returns the size of the workspace that needs to be allocated on the Device side.</td>
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

    Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

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
    <td>The passed-in parameter must be a required input, output, or required attribute, but a null pointer is actually passed in.</td>
    </tr>
    <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
    <td>The data type, data format, or dimension of gmmX, gmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional, group, epWorldSize, sendCounts, or recvCounts is not within the supported range.</td>
    </tr>
    </tbody></table>

## aclnnGroupedMatMulAlltoAllvV2

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
    <td>Device side-applied workspace memory address.</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>Input</td>
    <td>Device side-applied workspace size, obtained by the first-phase API <code>aclnnGroupedMatMulAlltoAllvV2GetWorkspaceSize</code>.</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>Input</td>
    <td>op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
    <td>stream</td>
    <td>Input</td>
    <td>Stream for task execution.</td>
    </tr>
    </tbody></table>

- **Return Value**

    Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnGroupedMatMulAlltoAllvV2 defaults to a deterministic implementation.
- Communication engine constraints:

  <!-- npu="A3" id10 -->
  - Atlas A3 products: AI_CPU communication and AIV communication are supported.
  <!-- end id10 -->
  <!-- npu="910b" id11 -->
  - Atlas A2 products: Only AIV communication is supported.
  <!-- end id11 -->
  <!-- npu="950" id12 -->
  - Ascend 950PR&950DT products: CCU communication and AI_CPU communication are supported. CCU supports only interconnection within the UB domain of a single node, while AI_CPU supports interconnection across nodes within the UB domain.

  <!-- end id12 -->

- Variables used in shape in the parameter description:
  - BSK: The number of tokens received by the current card, which is the cumulative sum of the recvCounts parameter. Value range: (0, 52428800].
  - H1: Indicates the hidden size of the routed expert. Value range: (0, 65536).
  - H2: Indicates the hidden size of the shared expert. Value range: (0, 12288].
  - e: Number of experts on a single device. AIV communication requires e > 0 and supports a maximum e * epWorldSize of 1024. Other communication engines require e<=32 and support a maximum e * epWorldSize of 256.
  - N1: Indicates the head_num of the routed expert. Value range: (0, 65536).
  - N2: Indicates the head_num of the shared expert. Value range: (0, 65536).
  - BS: batch sequence size.
  - K: Number of selected TopK experts. AIV communication of Atlas A3 products supports [2, 16], and other use cases support [2, 8].
  - When shared experts are enabled, BS is dimension 0 of mmXOptional and mmYOptional, BSK is dimension 0 of y, and `BSK=BS*K` must hold with K in the range above.
  - A: the number of tokens sent by the current card, which is the cumulative sum of the sendCounts parameter.
  - The sum of the A parameter of all devices in the ep communicator equals the sum of the BSK parameter of all devices.

<!-- npu="910b" id13 -->
- <term>Atlas A2 products</term>:
  - Both A and BSK must be in the range [1, 5000000]; N1 must not exceed 32768.
  - `sendCounts` and `recvCounts` are both INT64 direct count arrays, flattened in the order of `[rank][localExpert]`. The length of each must be equal to e * epWorldSize. Elements are non-negative, do not exceed A and BSK respectively, and their sums equal A and BSK respectively.
  - `HCCL_BUFFSIZE` on each device in the communicator must be set based on the maximum send volume, satisfying `HCCL_BUFFSIZE >= max(200, ceil(A * N1 * 2 / 1048576) + 21)`, in MiB. Each FLOAT16 and BFLOAT16 element occupies 2 bytes, and 21 MiB is reserved for the control region.

<!-- end id13 -->

<!-- npu="A3" id14 -->
- <term>Atlas A3 products</term>: Performance degradation may occur when the communication volume on a single device is below 2 MB.

<!-- end id14 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

NOTE: This sample code calls some APIs of the HCCL collective communication library: HcclGetCommName, HcclCommInitAll, and HcclCommDestroy. For details, see [*HCCL API (C)*](https://hiascend.com/en/document/redirect/CannCommunityHcclCppApi).

<!-- npu="950,A3" id15 -->
- <term>Atlas A3 products</term> and <term>Ascend 950PR&950DT products</term>:

    ```Cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <cstring>
    #include <vector>
    #include "acl/acl.h"
    #include "hccl/hccl.h"
    #include "aclnnop/aclnn_grouped_mat_mul_allto_allv_v2.h"

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
    constexpr int64_t EP_WORLD_SIZE = 8;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    constexpr int64_t H = 7168;
    constexpr int64_t e = 4;
    constexpr int64_t N1 = 4096;
    constexpr int64_t N2 = 4096;
    constexpr int64_t A = BS * K;

    std::vector<int16_t> pGmmYData(BS *K *N1, 0);
    std::vector<int16_t> pmmYData(BS *N2, 0);

    int LaunchOneThreadAlltoAllvGmm(Args &args)
    {
        int ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret: %d\n", ret); return ret);
        char hcomName[128] = {0};
        ret = HcclGetCommName(args.hcclComm, hcomName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret: %d\n", ret); return -1);

        std::vector<int64_t> gmmXShape = {A, H};
        std::vector<int64_t> gmmWShape = {e, H, N1};
        std::vector<int64_t> gmmYShape = {BS * K, N1};
        std::vector<int64_t> permuteShape = {A, H};
        std::vector<int64_t> mmXShape = {BS, H};
        std::vector<int64_t> mmWShape = {H, N2};
        std::vector<int64_t> mmYShape = {BS, N2};

        std::vector<int64_t> sendCountsShape = {EP_WORLD_SIZE * e};
        std::vector<int64_t> recvCountsShape = {EP_WORLD_SIZE * e};

        std::vector<int64_t> sendCountsList(EP_WORLD_SIZE * e, A / (EP_WORLD_SIZE * e));
        std::vector<int64_t> recvCountsList(EP_WORLD_SIZE * e, A / (EP_WORLD_SIZE * e));

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

        aclTensor *sendCountsTensor = nullptr;
        aclTensor *recvCountsTensor = nullptr;

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        void *workspaceAddr = nullptr;

        long long gmmXShapeSize = GetShapeSize(gmmXShape);
        long long gmmWShapeSize = GetShapeSize(gmmWShape);
        long long gmmYShapeSize = GetShapeSize(gmmYShape);

        long long mmXShapeSize = GetShapeSize(mmXShape);
        long long mmWShapeSize = GetShapeSize(mmWShape);
        long long mmYShapeSize = GetShapeSize(mmYShape);

        std::vector<uint16_t> gmmXHostData(gmmXShapeSize, (args.rankId + 1) * 1024); // BF16, FP16
        std::vector<uint16_t> gmmWHostData(gmmWShapeSize, (args.rankId + 1) * 512);
        std::vector<uint16_t> gmmYHostData(gmmYShapeSize, 65535);

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

        aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
        aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());

        // Call the first-phase API.
        ret = aclnnGroupedMatMulAlltoAllvV2GetWorkspaceSize(gmmX, gmmW, sendCountsTensor, recvCountsTensor, mmX, mmW,
                                                            hcomName, "ai_cpu", EP_WORLD_SIZE, sendCounts, recvCounts, false, false,
                                                            gmmY, mmY, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("[ERROR] aclnnGroupedMatMulAlltoAllvV2GetWorkspaceSize failed. ret = %d \n", ret);
                return ret);

        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        // Call the second-phase API.
        ret = aclnnGroupedMatMulAlltoAllvV2(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnGroupedMatMulAlltoAllvV2 failed. ret = %d \n", ret);
                return ret);
        // (boilerplate) Synchronously wait for the task to complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
                return ret);
        LOG_PRINT("[INFO] device_%d aclnnGroupedMatMulAlltoAllvV2 execute successfully.\n", args.rankId);
        // Release device resources. Modify this based on the specific API definition.
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

<!-- end id15 -->
