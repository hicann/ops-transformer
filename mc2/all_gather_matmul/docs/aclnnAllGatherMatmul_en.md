# aclnnAllGatherMatmul

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T12:47:43.182Z -->

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

**NOTE**: When using this API, ensure that the driver firmware package and the CANN package are both of the matching 8.0.RC2 version or a matching later version. Otherwise, errors such as BUS ERROR will occur.

## Function

- **Description**: Fuses all-gather communication with matrix multiplication computation.
- **Formula**:

    $$
    output=AllGather(x1)@x2+bias
    $$

    $$
    gatherOut=AllGather(x1)
    $$

## Function Prototype

Each operator consists of a two-phase API. Call the "aclnnAllGatherMatmulGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor that contains the operator computation flow, and then call the "aclnnAllGatherMatmul" API to perform the computation.

```cpp
aclnnStatus aclnnAllGatherMatmulGetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *bias,
    const char      *group,
    int64_t         gatherIndex,
    int64_t         commTurn,
    int64_t         streamMode,
    const aclTensor *output,
    const aclTensor *gatherOut,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnAllGatherMatmul(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnAllGatherMatmulGetWorkspaceSize

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
        <td>x1</td>
        <td>Input</td>
        <td>Corresponds to x1 in the formula.</td>
        <td><ul><li>Supports empty tensors.</li><li>Must have the same data type as x2.</li><li>The current version supports only 2D shape inputs and only non-transposition use cases.</li></ul></td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
    </tr>
    <tr>
        <td>x2</td>
        <td>Input</td>
        <td>Corresponds to x2 in the formula.</td>
        <td><ul><li>Supports empty tensors.</li><li>The data type must be the same as x1.</li><li>The current version supports only 2D inputs, in both transposed and non-transposed use cases.</li><li>Only non-contiguous tensors with two axes transposed are supported; <a href="../../../docs/en/context/non_contiguous_tensor.md">[non-contiguous tensors]</a> in other use cases are not supported.</li></ul></td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>√ (only for the transposed use case)</td>
    </tr>
    <tr>
        <td>bias</td>
        <td>Input</td>
        <td>The bias in the formula.</td>
        <td><ul><li>Supports passing in a null pointer.</li><li>The current version supports only 1D inputs.</li></ul></td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>1</td>
        <td>×</td>
    </tr>
    <tr>
        <td>group</td>
        <td>Input</td>
        <td>Communicator name.</td>
        <td>Obtained through the Hccl API "extern HcclResult HcclGetCommName(HcclComm comm, char* commName);", where commName is the group.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>gatherIndex</td>
        <td>Input</td>
        <td>Identifies the Gather target.</td>
        <td><ul><li>0 indicates that the target is x1, and 1 indicates that the target is x2.</li><li>In the current version, only input 0 is supported.</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commTurn</td>
        <td>Input</td>
        <td>Number of data splits for communication, that is, total data volume/data volume per communication.</td>
        <td>In the current version, only input 0 is supported.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>streamMode</td>
        <td>Input</td>
        <td>Stream mode enumeration.</td>
        <td>Currently, only the enumerated value 1 is supported.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>Output</td>
        <td>Output</td>
        <td>Result of AllGather communication and MatMul computation, that is, output in the Formula.</td>
        <td><ul><li>Supports empty tensors.</li><li>The data type must be the same as that of x1.</li></ul></td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
    </tr>
    <tr>
        <td>gatherOut</td>
        <td>Output</td>
        <td>Outputs only the result of the AllGather communication, that is, gatherOut in the Formula.</td>
        <td><ul><li>Supports empty tensors.</li><li>The data type must be the same as that of x1.</li></ul></td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size to be allocated on the Device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns the op executor, which contains the operator computation flow.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    </tbody></table>

    <!-- npu="A3,910b" id7 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
        - bias: Currently does not support non-zero input scenarios.
    <!-- end id7 -->
    <!-- npu="950" id8 -->
    - <term>Ascend 950PR&950DT products</term>:
        - bias: Supports non-zero input scenarios.

    <!-- end id8 -->

- **Return value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API performs input parameter validation and returns an error in the following scenarios:

    <table style="undefined;table-layout: fixed; width: 1166px"> <colgroup>
    <col style="width: 267px">
    <col style="width: 124px">
    <col style="width: 775px">
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
        <td>x1, x2, or output is a null pointer.</td>
    </tr>
    <tr>
        <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="3">161002</td>
        <td>The data types of x1, x2, bias, or output are not in the supported range.</td>
    </tr>
    <tr>
        <td>streamMode is not in the valid range.</td>
    </tr>
    <tr>
        <td>The k-axis of x1 is 0.</td>
    </tr>
    </tbody></table>

## aclnnAllGatherMatmul

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1166px"> <colgroup>
    <col style="width: 173px">
    <col style="width: 133px">
    <col style="width: 860px">
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
        <td>The workspace memory address allocated on the device.</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>The workspace size allocated on the device, obtained by calling the first-phase API <code>aclnnAllGatherMatmulGetWorkspaceSize</code>.</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Input</td>
        <td>op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream for executing tasks.</td>
    </tr>
    </tbody></table>

- **Return value**

    Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnAllGatherMatmul defaults to a deterministic implementation.

- The input x1 is 2-dimensional with shape (m, k). x2 must be 2-dimensional with shape (k, n), where the axes meet the input requirements of the MatMul operator, the k axes are equal, and the k axis value range is [256, 65535).
- For the empty tensor scenario supported by x1/x2, m and n can be empty, k cannot be empty, and the following conditions must be met:
    - m is null, k is not null, and n is not null.
    - m is not null, k is not null, and n is null.
    - m is null, k is not null, and n is null.
- The output is 2-dimensional, with a shape of (m*rank_size, n), where rank_size is the number of cards.

<!-- npu="910b" id9 -->
- <term>Atlas A2 products</term>: Supports 2, 4, and 8 cards, and only supports HCCS link full mesh networking.
<!-- end id9 -->
<!-- npu="A3" id10 -->
- <term>Atlas A3 products</term>: supports 2, 4, 8, 16, and 32 devices, and only supports double ring networking over HCCS links.
<!-- end id10 -->
<!-- npu="950" id11 -->
- <term>Ascend 950PR&950DT products</term>:
  - Supports 2, 4, 8, 16, 32, and 64 cards, and only supports HCCS link full mesh networking.
<!-- end id11 -->
<!-- npu="910b" id12 -->
- <term>Atlas A2 products</term>: The communication-computation fusion MC2 operators in a model only support the same communication domain.

<!-- end id12 -->

## Example

The following is sample code for reference only. For details about compilation and execution, see Compile and Run Samples.

NOTE: This sample code calls some APIs of the HCCL collective communication library: HcclGetCommName, HcclCommInitAll, and HcclCommDestroy. For details, see [*HCCL API (C)*](https://hiascend.com/en/document/redirect/CannCommunityHcclCppApi).

<!-- npu="950,A3,910b" id13 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>:

    ```Cpp
    #include <thread>
    #include <iostream>
    #include <vector>
    #include <memory>
    #include <cstdio>
    #include "hccl/hccl.h"
    #include "aclnn/opdev/fp16_t.h"
    #include "aclnnop/aclnn_all_gather_matmul.h"

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

    constexpr int DEV_NUM = 2;

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
            strides[i] = shape[i +1] * strides[i + 1];
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

    int launchOneThread_AllGatherMm(Args &args)
    {
        int ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret = %d \n", ret); return ret);

        char hcomName[128] = {0};
        ret = HcclGetCommName(args.hcclComm, hcomName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret: %d\n", ret); return -1);
        LOG_PRINT("[INFO] rank = %d, hcomName = %s, stream = %p\n", args.rankId, hcomName, args.stream);
        std::vector<int64_t> x1Shape = {128, 256};
        std::vector<int64_t> x2Shape = {256, 512};
        std::vector<int64_t> biasShape = {512};
        std::vector<int64_t> outShape = {128 * DEV_NUM, 512};
        std::vector<int64_t> gatherOutShape = {128 * DEV_NUM, 256};
        void *x1DeviceAddr = nullptr;
        void *x2DeviceAddr = nullptr;
        void *biasDeviceAddr = nullptr;
        void *outDeviceAddr = nullptr;
        void *gatherOutDeviceAddr = nullptr;
        aclTensor *x1 = nullptr;
        aclTensor *x2 = nullptr;
        aclTensor *bias = nullptr;
        aclTensor *out = nullptr;
        aclTensor *gatherOut = nullptr;

        int64_t gatherIndex = 0;
        int64_t commTurn = 0;
        int64_t streamMode = 1;
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        void *workspaceAddr = nullptr;

        long long x1ShapeSize = GetShapeSize(x1Shape);
        long long x2ShapeSize = GetShapeSize(x2Shape);
        long long biasShapeSize = GetShapeSize(biasShape);
        long long outShapeSize = GetShapeSize(outShape);
        long long gatherOutShapeSize = GetShapeSize(gatherOutShape);

        std::vector<op::fp16_t> x1HostData(x1ShapeSize, 0);
        std::vector<op::fp16_t> x2HostData(x2ShapeSize, 0);
        std::vector<op::fp16_t> biasHostData(biasShapeSize, 0);
        std::vector<op::fp16_t> outHostData(outShapeSize, 0);
        std::vector<op::fp16_t> gatherOutHostData(gatherOutShapeSize, 0);

        ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT16, &x2);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gatherOutHostData, gatherOutShape, &gatherOutDeviceAddr,
            aclDataType::ACL_FLOAT16, &gatherOut);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        // Call the first-phase API.
        ret = aclnnAllGatherMatmulGetWorkspaceSize(
            x1, x2, bias, hcomName, gatherIndex, commTurn, streamMode, out, gatherOut, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnAllGatherMatmulGetWorkspaceSize failed. ret = %d \n", ret); return ret);
        // Allocate device memory based on the workspaceSize calculated by the first-phase API.
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        // Call the second-phase API.
        ret = aclnnAllGatherMatmul(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnAllGatherMatmul failed. ret = %d \n", ret); return ret);
        // (Boilerplate) Synchronize and wait for the task execution to complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
            return ret);
        LOG_PRINT("[INFO] device_%d aclnnAllGatherMatmul execute successfully.\n", args.rankId);
        // Release device resources; modify them based on the specific API definition.
        if (x1 != nullptr) {
            aclDestroyTensor(x1);
        }
        if (x2 != nullptr) {
            aclDestroyTensor(x2);
        }
        if (bias != nullptr) {
            aclDestroyTensor(bias);
        }
        if (out != nullptr) {
            aclDestroyTensor(out);
        }
        if (gatherOut != nullptr) {
            aclDestroyTensor(gatherOut);
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
        if (outDeviceAddr != nullptr) {
            aclrtFree(outDeviceAddr);
        }
        if (gatherOutDeviceAddr != nullptr) {
            aclrtFree(gatherOutDeviceAddr);
        }
        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
        ret = HcclCommDestroy(args.hcclComm);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommDestroy failed. ret = %d \n", ret); return ret);
        ret = aclrtDestroyStream(args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtDestroyStream failed. ret = %d \n", ret); return ret);
        ret = aclrtResetDevice(args.rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtResetDevice failed. ret = %d \n", ret); return ret);
        ret = aclrtDestroyContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtDestroyContext failed. ret = %d \n", ret); return ret);
        return 0;
    }

    int main(int argc, char *argv[])
    {
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
        int32_t devices[DEV_NUM];
        for (int i = 0; i < DEV_NUM; i++) {
            devices[i] = i;
        }
        // Initialize the communicator.
        HcclComm comms[DEV_NUM];
        ret = HcclCommInitAll(DEV_NUM, devices, comms);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll failed. ret = %d \n", ret); return ret);

        Args args[DEV_NUM];
        // Start multiple threads.
        std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            args[rankId].rankId = rankId;
            args[rankId].hcclComm = comms[rankId];
            args[rankId].context = context[rankId];
            args[rankId].stream = stream[rankId];
            threads[rankId].reset(new(std::nothrow) std::thread(&launchOneThread_AllGatherMm, std::ref(args[rankId])));
        }
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            threads[rankId]->join();
        }
        aclFinalize();
        return 0;
    }
    ```

<!-- end id13 -->
