# aclnnGroupedMatMulAlltoAllv

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- Description: Implements the fusion of GroupedMatMul, Unpermute, and AlltoAllv for routed experts, and achieves parallel fusion with the MatMul for shared experts. Computation is performed before communication.

- Formulas:

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

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnGroupedMatMulAlltoAllvGetWorkspaceSize` is called to obtain the input parameters and compute the required workspace size based on the process. Then, `aclnnGroupedMatMulAlltoAllv` is called to perform computation.

```cpp
aclnnStatus aclnnGroupedMatMulAlltoAllvGetWorkspaceSize(
    const aclTensor*   gmmX,
    const aclTensor*   gmmWeight,
    const aclTensor*   sendCountsTensorOptional,
    const aclTensor*   recvCountsTensorOptional,
    const aclTensor*   mmXOptional,            
    const aclTensor*   mmWeightOptional,
    const char*        group,
    int64_t            epWorldSize,
    const aclIntArray* sendCounts,
    const aclIntArray* recvCounts,
    bool               transGmmWeight,
    bool               transMmWeight,
    aclTensor*         y,
    aclTensor*         mmYOptional,
    uint64_t*          workspaceSize,
    aclOpExecutor**       executor)
```

```cpp
aclnnStatus aclnnGroupedMatMulAlltoAllv(
    void*           workspace,
    uint64_t        workspaceSize,
    aclOpExecutor*  executor,
    aclrtStream     stream)
```

## aclnnGroupedMatMulAlltoAllvGetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1237px"><colgroup>
    <col style="width: 252px">
    <col style="width: 121px">
    <col style="width: 523px">
    <col style="width: 216px">
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
    <td>gmmX</td>
    <td>Input</td>
    <td>This input undergoes AlltoAllv communication, and the communication result serves as the left matrix for GroupedMatMul computation. The shape can be two-dimensional (A, H1).</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>gmmWeight</td>
    <td>Input</td>
    <td>Right matrix for GroupedMatMul computation. The data type is the same as that of gmmX. The shape can be three-dimensional (e, H1, N1).</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>sendCountsTensorOptional</td>
    <td>Input</td>
    <td> Optional input with shape (e * epWorldSize,). This parameter is not supported in the current version, and nullptr is used.</td>
    <td>INT32, INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>recvCountsTensorOptional</td>
    <td>Input</td>
    <td> Optional input with shape (e * epWorldSize,). This parameter is not supported in the current version, and nullptr is used.</td>
    <td>INT32, INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>mmXOptional</td>
    <td>Input</td>
    <td>Optional input serving as the left matrix for shared expert MatMul computation. This parameter must be transferred together with mmWeightOptional or set to nullptr. The data type is the same as that of gmmX. The shape can be two-dimensional (BS, H2).</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>mmWeightOptional</td>
    <td>Input</td>
    <td>Optional input serving as the right matrix for shared expert MatMul computation. This parameter must be transferred together with mmXOptional or set to nullptr. The data type is the same as that of gmmX. The shape can be two-dimensional (H2, N2).</td>
    <td>FLOAT16, BFLOAT16</td>
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
    <td>EP communication domain size.<br><term>Atlas A3 training products/Atlas A3 inference products</term>: The value can be 8, 16, 32, 64, or 128.</td>
    <td>INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>sendCounts</td>
    <td>Input</td>
    <td>Number of tokens to send to other devices. The data type is INT64. The value is e * epWorldSize, and the maximum value is 256. The input type must be list.</td>
    <td>aclIntArray* (with INT64 elements)</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>recvCounts</td>
    <td>Input</td>
    <td>Number of tokens to receive from other devices. The data type is INT64. The value is e * epWorldSize, and the maximum value is 256. The input type must be list.</td>
    <td>aclIntArray* (with INT64 elements)</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>transGmmWeight</td>
    <td>Input</td>
    <td>Whether to transpose gmmWeight. Set true to transpose, or false to not transpose.</td>
    <td>BOOL</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>transMmWeight</td>
    <td>Input</td>
    <td>Whether to transpose mmWeightOptional. Set true to transpose, or false to not transpose.</td>
    <td>BOOL</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>y</td>
    <td>Output</td>
    <td>Final computation result. The data type is the same as that of the input gmmX. The shape can be two-dimensional (BSK, N1).</td>
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
    <td>workspaceSize</td>
    <td>Output</td>
    <td>Size of the workspace required to be allocated on the device.</td>
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

- **Return**

    `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
    
    The first-phase API implements input parameter validation. The following errors may be thrown:

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
    <td>1. A null pointer is passed in instead of the required input, output, or attribute.</td>
    </tr>
    <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
    <td>1. The data type, data format, or dimensions of gmmX, gmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional, group, epWorldSize, sendCounts, or recvCounts are not supported.</td>
    </tr>
    </tbody></table>

## aclnnGroupedMatMulAlltoAllv

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
    <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API <code>aclnnGroupedMatMulAlltoAllvGetWorkspaceSize</code>.</td>
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

- **Return**

    `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnGroupedMatMulAlltoAllv` defaults to deterministic implementation.

- Shape variables used in parameter descriptions:
  - `BSK`: Number of tokens received by the local device, which is the sum of all elements in the `recvCounts` parameter. The value range is (0, 52428800).
  - `H1`: Hidden layer size of the routed experts. The value range is (0, 65536).
  - `H2`: Hidden layer size of the shared experts. The value range is (0, 12288].
  - `e`: Number of experts on a single device. `e` <= 32. The maximum supported value for `e * epWorldSize` is 256.
  - `N1`: `head_num` for routed experts. The value range is (0, 65536).
  - `N2`: `head_num` for shared experts. The value range is (0, 65536).
  - `BS`: Batch sequence size.
  - `K`: Number of experts selected via Top-K. The value range is [2, 8].
  - `A`: Number of tokens sent by the local device, which is the sum of all elements in the `sendCounts` parameter.
  - The sum of the `A` parameters across all devices in the EP communication domain equals the sum of the `BSK` parameters across all devices.

- <term>Atlas A3 training products/Atlas A3 inference products</term>: The communication volume per device must be greater than or equal to 2 MB.

## Calling Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

- <term>Atlas A3 training products/Atlas A3 inference products</term>:

    ```Cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <vector>
    #include "acl/acl.h"
    #include "hccl/hccl.h"
    #include "aclnnop/aclnn_grouped_mat_mul_allto_allv.h"

    #define CHECK_RET(cond, return_expr) \
        do {                             \
            if (!(cond)) {               \
                return_expr;             \
            }                            \
        } while (0)

    #define LOG_PRINT(message, ...)          \
        do {                                 \
            printf(message, ##__VA_ARGS__);  \
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

    std::vector<int16_t> pGmmyData(BS *K *N1, 0);
    std::vector<int16_t> pmmYData(BS *N2, 0);

    int LaunchOneThreadAlltoAllvGmm(Args &args)
    {
        int ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret: %d\n", ret); return ret);
        char hcomName[128] = {0};
        ret = HcclGetCommName(args.hcclComm, hcomName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetEpCommName failed. ret: %d\n", ret); return -1);

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

        std::vector<uint16_t> gmmXHostData(gmmXShapeSize, (args.rankId + 1) * 1024);  // BF16, FP16
        std::vector<uint16_t> gmmWHostData(gmmWShapeSize, (args.rankId + 1) * 512);
        std::vector<uint16_t> gmmYHostData(gmmYShapeSize, 65535);

        std::vector<uint16_t> mmXHostData(mmXShapeSize, (args.rankId + 1) * 1024);  // BF16, FP16
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
        ret = aclnnGroupedMatMulAlltoAllvGetWorkspaceSize(gmmX,
            gmmW,
            sendCountsTensor,
            recvCountsTensor,
            mmX,
            mmW,
            hcomName,
            EP_WORLD_SIZE,
            sendCounts,
            recvCounts,
            false,
            false,
            gmmY,
            mmY,
            &workspaceSize,
            &executor);
        CHECK_RET(
            ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnGroupedMatMulAlltoAllvGetWorkspaceSize failed. ret = %d \n", ret);
            return ret);

        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        // Call the second-phase API.
        ret = aclnnGroupedMatMulAlltoAllv(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnGroupedMatMulAlltoAllv failed. ret = %d \n", ret);
                return ret);
        // (Boilerplate) Wait until the task execution is complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret); 
                return ret);
        // Free device resources. Modify the code based on the API definition.
        if (args.rankId == 0) {
            size_t size = A * N1 * sizeof(int16_t);
            aclrtMemcpy(pGmmyData.data(), size, gmmYDeviceAddr, size, ACL_MEMCPY_DEVICE_TO_HOST);
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
        // This example is implemented based on Atlas A3 and can only run on Atlas A3.
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
        // Initialize the collective communication domain.
        HcclComm comms[EP_WORLD_SIZE];
        ret = HcclCommInitAll(EP_WORLD_SIZE, devices, comms);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll failed. ret = %d \n", ret); return ret);

        Args args[EP_WORLD_SIZE];
        // Enable multi-threading.
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
