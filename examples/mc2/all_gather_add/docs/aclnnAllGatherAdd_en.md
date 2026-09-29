# aclnnAllGatherAdd

[📄 View Source Code](https://gitcode.com/cann/ops-transformer/tree/master/examples/mc2/all_gather_add)

## Supported Products

| Product                                                                           | Supported|
| :------------------------------------------------------------------------------ | :------: |
| Ascend 950PR/Ascend 950DT                                               | ×       |
| <term>Atlas A3 training products/Atlas A3 inference products</term>                       | √       |
| <term>Atlas A2 training products/Atlas A2 inference products</term>| √       |
| <term>Atlas 200I/500 A2 inference products</term>                                        | ×       |
| <term>Atlas inference products</term>                                               | ×       |
| <term>Atlas training products</term>                                                | ×       |

## Function Description

- API function: integrates [AllGather](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0873.html) communication and [Add](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0035.html).
- **Formula**:

    $$
    gatherOut=AllGather(a0, a1)
    $$

    $$
    c[i]=gatherOut[i] + b[i]
    $$

## Function Prototype

Each operator has [two-phase API](../../../../docs/en/context/two_phase_api.md) calls. First, `aclnnAllGatherAddGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnAllGatherAdd` is called to perform computation.

```cpp
aclnnStatus aclnnAllGatherAddGetWorkspaceSize(
    const aclTensor *a,
    const aclTensor *b,
    char            *group,
    int64_t          rankSize,
    const aclTensor *cOut,
    const aclTensor *gatherOutOut,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnAllGatherAdd(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnAllGatherAddGetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1567px"><colgroup>
    <col style="width: 220px">
    <col style="width: 120px">
    <col style="width: 300px">
    <col style="width: 330px">
    <col style="width: 192px">
    <col style="width: 120px">
    <col style="width: 160px">
    <col style="width: 125px">
    </colgroup>
    <thead>
    <tr>
        <th>Name</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Usage</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Dimension (Shape)</th>
        <th>Non-contiguous Tensor</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>a (aclTensor*)</td>
        <td>Input</td>
        <td>a0 or a1 in the calculation formula.</td>
        <td>In the current version, only two-dimensional shape input is supported, and only the non-transposition scenario is supported.</td>
        <td>FLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
    </tr>
    <tr>
        <td>b (aclTensor*)</td>
        <td>Input</td>
        <td>b in the calculation formula.</td>
        <td>In the current version, only two-dimensional shape input is supported, and only the non-transposition scenario is supported.</td>
        <td>FLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
    </tr>
    <tr>
        <td>group (char*)</td>
        <td>Input</td>
        <td>Communication domain name.</td>
        <td>It is obtained through the <code>extern HcclResult HcclGetCommName(HcclComm comm, char* commName);</code> API provided by HCCL, where <code>commName</code> is the same as <code>group</code>.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>rankSize (int64_t)</td>
        <td>Input</td>
        <td>Number of NPUs involved in AllGather communication in the communicator.</td>
        <td>In the current version, only 2 can be entered.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>cOut (aclTensor*)</td>
        <td>Output</td>
        <td>Result of AllGather communication and Add computation, that is, c in the formula.</td>
        <td>Same as the data type of the input.</td>
        <td>FLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
    </tr>
    <tr>
        <td>gatherOutOut (aclTensor*)</td>
        <td>Output</td>
        <td>Result of AllGather communication, that is, gatherOut in the formula.</td>
        <td>Same as the data type of the input.</td>
        <td>FLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
    </tr>
    <tr>
        <td>workspaceSize (uint64_t*)</td>
        <td>Output</td>
        <td>Size of the workspace to be allocated on the device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>executor (aclOpExecutor**)</td>
        <td>Output</td>
        <td>Operator executor, containing the operator computation process.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    </tbody></table>

- **Returns**

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../../docs/en/context/aclnn_return_code.md).
    
    The first-phase API implements input parameter validation. The following error codes may be returned.

    <table style="undefined;table-layout: fixed; width: 1166px"> <colgroup>
    <col style="width: 267px">
    <col style="width: 124px">
    <col style="width: 775px">
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
        <td>The input a, b, or cOut is a null pointer.</td>
    </tr>
    <tr>
        <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="3">161002</td>
        <td>The data type of a, b, gatherout, or output is not supported, or the shape is invalid.</td>
    </tr>
    </tbody></table>

## aclnnAllGatherAdd

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1166px"> <colgroup>
    <col style="width: 173px">
    <col style="width: 133px">
    <col style="width: 860px">
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
        <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnAllGatherAddGetWorkspaceSize`.</td>
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

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../../docs/en/context/aclnn_return_code.md).

## Restrictions

- Deterministic computing: The allGatherAdd operator defaults to deterministic implementation.
- Currently, this example operator supports only the static shape: a(240, 256), b(240 * 2, 256), and fixed rank_size = 2.
- All inputs do not support empty tensors. The value range is [-5, 5].

## Calling Example

The following is the sample code, which is for reference only. For details about the compilation and execution processes, see [Compile and Run Example](../../../../docs/en/context/compile_and_run_sample.md) in the README file of this operator.

Note: This sample code calls some HCCL collective communication library APIs, including HcclGetCommName, HcclCommInitAll, and HcclCommDestroy. For details, see [<<HCCL API (C)>>](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/hcclapiref/hcclcpp_07_0001.html).

- <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>:

    ```Cpp
    #include <thread>
    #include <future>
    #include <iostream>
    #include <algorithm>
    #include <cmath>
    #include <random>
    #include <chrono>
    #include <iomanip>
    #include <string>
    #include <vector>
    #include "hccl/hccl.h"
    #include "aclnn/opdev/fp16_t.h"
    #include "../op_host/op_api/aclnn_all_gather_add.h"

    #define CHECK_RET(cond, return_expr) \
        do {                             \
            if (!(cond)) {               \
                LOG_PRINT("Example failed.\n"); \
                return_expr;             \
            }                            \
        } while (0)

    #define LOG_PRINT(message, ...)         \
        do {                                \
            printf(message, ##__VA_ARGS__); \
        } while(0)

    constexpr int RANK_DIM = 2;
    typedef struct {
        std::vector<op::fp16_t> rank0_a;
        std::vector<op::fp16_t> rank0_b;
        std::vector<op::fp16_t> rank1_a;
        std::vector<op::fp16_t> rank1_b;

        std::vector<op::fp16_t> gatherOut;
        std::vector<op::fp16_t> rank0_c;
        std::vector<op::fp16_t> rank1_c;
    } TestData;

    int64_t GetShapeSize(const std::vector<int64_t> &shape)
    {
        int64_t shapeSize = 1;
        for (auto i : shape) {
            shapeSize *= i;
        }
        return shapeSize;
    }

    // Fixed shape in this example.
    const std::vector<int64_t> aShape = {240, 256};
    const std::vector<int64_t> bShape = {240 * RANK_DIM, 256};

    const long long aShapeSize = GetShapeSize(aShape);
    const long long bShapeSize = GetShapeSize(bShape);

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

    int CompareVector(std::vector<op::fp16_t> &vec1, const std::vector<op::fp16_t> &vec2)
    {
        const float tolerance = 0.001f; // One thousandth.
        for (size_t i = 0; i < vec1.size(); ++i) {
            float a = static_cast<float>(vec1[i]);
            float b = static_cast<float>(vec2[i]);

            float diff = std::fabs(a - b);
            float maxAbs = std::max(std::fabs(a), std::fabs(b));
            if (maxAbs > 1e-6f) {
                float relativeError = diff / maxAbs;
                if (relativeError > tolerance) {
                    return 1;
                }
            } else {
                if (diff > tolerance) {
                    return 1;
                }
            }
        }
        return 0;
    }

    struct Args {
        int rankId;
        HcclComm hcclComm;
        aclrtStream stream;
    };

    int LaunchOneThreadAllGatherAdd(Args &args, const TestData &testData)
    {
        int ret = aclrtSetDevice(args.rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);

        char hcomName[128] = {0};
        ret = HcclGetCommName(args.hcclComm, hcomName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret: %d\n", ret); return -1);
        LOG_PRINT("[INFO] rank = %d, hcomName = %s, stream = %p\n", args.rankId, hcomName, args.stream);

        void *aDeviceAddr = nullptr;
        void *bDeviceAddr = nullptr;
        void *outDeviceAddr = nullptr;
        void *gatherOutDeviceAddr = nullptr;
        aclTensor *a = nullptr;
        aclTensor *b = nullptr;
        aclTensor *out = nullptr;
        aclTensor *gatherOut = nullptr;

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        void *workspaceAddr = nullptr;

        std::vector<op::fp16_t> aHostData(aShapeSize, 0);
        std::vector<op::fp16_t> bHostData(bShapeSize, 0);

        // Fill the host-side input with randomly generated test data.
        if (args.rankId == 0) {
            std::copy(testData.rank0_a.begin(), testData.rank0_a.end(), aHostData.begin());
            std::copy(testData.rank0_b.begin(), testData.rank0_b.end(), bHostData.begin());
        } else {
            std::copy(testData.rank1_a.begin(), testData.rank1_a.end(), aHostData.begin());
            std::copy(testData.rank1_b.begin(), testData.rank1_b.end(), bHostData.begin());
        }

        std::vector<op::fp16_t> outHostData(bShapeSize, 0);
        std::vector<op::fp16_t> gatherOutHostData(bShapeSize, 0);

        ret = CreateAclTensor(aHostData, aShape, &aDeviceAddr, aclDataType::ACL_FLOAT16, &a);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(bHostData, bShape, &bDeviceAddr, aclDataType::ACL_FLOAT16, &b);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(outHostData, bShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gatherOutHostData, bShape, &gatherOutDeviceAddr,
            aclDataType::ACL_FLOAT16, &gatherOut);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        // Call the first-phase API.
        ret = aclnnAllGatherAddGetWorkspaceSize(
            a, b, hcomName, RANK_DIM, out, gatherOut, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnAllGatherAddGetWorkspaceSize failed. ret = %d \n", ret); return ret);
        // Allocate device memory based on the computed workspaceSize.
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret);  return ret);
        }
        // Call the second-phase API.
        ret = aclnnAllGatherAdd(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnAllGatherAdd failed. ret = %d \n", ret); return ret);
        // Wait until the task execution is complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
            return ret);
        LOG_PRINT("[INFO] device_%d aclnnAllGatherAdd execute successfully.\n", args.rankId);

        // Compare the operator computation result with the golden data.
        std::vector<op::fp16_t> gatherOutData(bShapeSize, 0);
        // Copy the computation result from the device to the host.
        ret = aclrtMemcpy(gatherOutData.data(), bShapeSize * sizeof(gatherOutData[0]), gatherOutDeviceAddr,
                        bShapeSize * sizeof(gatherOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        // Compare the AllGather result.
        ret = CompareVector(gatherOutData, testData.gatherOut);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] gatherOut compare failed. ret = %d \n", ret); return ret);

        std::vector<op::fp16_t> outputData(bShapeSize, 0);
        ret = aclrtMemcpy(outputData.data(), bShapeSize * sizeof(outputData[0]), outDeviceAddr,
                        bShapeSize * sizeof(outputData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        // Compare the AllGatherAdd results.
        if (args.rankId == 0) {
            ret = CompareVector(outputData, testData.rank0_c);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] output compare failed. ret = %d \n", ret); return ret);
        } else {
            ret = CompareVector(outputData, testData.rank1_c);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] output compare failed. ret = %d \n", ret); return ret);
        }
        LOG_PRINT("[INFO] device_%d aclnnAllGatherAdd golden compare successfully.\n", args.rankId);

        auto hcclRet = HcclCommDestroy(args.hcclComm);
        CHECK_RET(hcclRet == HCCL_SUCCESS, LOG_PRINT("[ERROR] HcclCommDestroy failed. ret = %d \n", hcclRet));
        // Release device resources. Modify the code based on the API definition.
        if (a != nullptr) {
            aclDestroyTensor(a);
        }
        if (b != nullptr) {
            aclDestroyTensor(b);
        }
        if (out != nullptr) {
            aclDestroyTensor(out);
        }
        if (gatherOut != nullptr) {
            aclDestroyTensor(gatherOut);
        }
        if (aDeviceAddr != nullptr) {
            aclrtFree(aDeviceAddr);
        }
        if (bDeviceAddr != nullptr) {
            aclrtFree(bDeviceAddr);
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
        ret = aclrtDestroyStream(args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtDestroyStream failed. ret = %d \n", ret); return ret);
        ret = aclrtResetDevice(args.rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtResetDevice failed. ret = %d \n", ret); return ret);
        return 0;
    }

    // Randomly generate data in the range of [-5, 5] to fill the vector.
    int RandomVectorGenerator(std::vector<op::fp16_t> &vec, long long size)
    {
        unsigned seed = static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());
        std::mt19937 generator(seed);
        std::uniform_real_distribution<float> distribution(-5.0f, 5.0f);
        for (auto& elem : vec) {
            elem = static_cast<op::fp16_t>(distribution(generator));
        }
        return 0;
    }

    // Concatenate two vectors of the same size.
    int GatherVectors(std::vector<op::fp16_t> &vec1, std::vector<op::fp16_t> &vec2, std::vector<op::fp16_t> &vec3)
    {
        vec3.clear();
        vec3.reserve(vec1.size() + vec2.size());
        vec3.insert(vec3.end(), vec1.begin(), vec1.end());
        vec3.insert(vec3.end(), vec2.begin(), vec2.end());
        return 0;
    }

    // Perform addition on two vectors of the same size.
    int AddVectors(std::vector<op::fp16_t> &vec1, std::vector<op::fp16_t> &vec2, std::vector<op::fp16_t> &vec3)
    {
        vec3.clear();
        vec3.resize(vec1.size());
        for (size_t i = 0; i < vec1.size(); ++i) {
            vec3[i] = vec1[i] + vec2[i];
        }
        return 0;
    }

    int GenerateTestData(TestData &testData)
    {
        // Randomly generate the input.
        int ret = RandomVectorGenerator(testData.rank0_a, testData.rank0_a.size());
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] RandomVectorGenerate rank0_a failed. ret = %d \n", ret);  return ret);
        ret = RandomVectorGenerator(testData.rank0_b, testData.rank0_b.size());
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] RandomVectorGenerate rank0_b failed. ret = %d \n", ret);  return ret);
        ret = RandomVectorGenerator(testData.rank1_a, testData.rank1_a.size());
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] RandomVectorGenerate rank1_a failed. ret = %d \n", ret);  return ret);
        ret = RandomVectorGenerator(testData.rank1_b, testData.rank1_b.size());
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] RandomVectorGenerate rank1_b failed. ret = %d \n", ret);  return ret);

        // Calculate the golden data.
        ret = GatherVectors(testData.rank0_a, testData.rank1_a, testData.gatherOut);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] Generate gatherOut failed. ret = %d \n", ret);  return ret);
        ret = AddVectors(testData.gatherOut, testData.rank0_b, testData.rank0_c);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] Generate rank0 output failed. ret = %d \n", ret);  return ret);
        ret = AddVectors(testData.gatherOut, testData.rank1_b, testData.rank1_c);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] Generate rank1 output failed. ret = %d \n", ret);  return ret);

        return 0;
    }

    int main(int argc, char *argv[])
    {
        // Generate test data.
        TestData testData = {
            std::vector<op::fp16_t>(aShapeSize, 0.0f), // rank0_a
            std::vector<op::fp16_t>(bShapeSize, 0.0f), // rank1_a
            std::vector<op::fp16_t>(aShapeSize, 0.0f), // rank0_b
            std::vector<op::fp16_t>(bShapeSize, 0.0f), // rank1_b

            std::vector<op::fp16_t>(bShapeSize, 0.0f), // gatherOut
            std::vector<op::fp16_t>(bShapeSize, 0.0f), // rank0_c
            std::vector<op::fp16_t>(bShapeSize, 0.0f) // rank1_c
        };
        int ret = GenerateTestData(testData);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] GenerateTestData failed. ret = %d \n", ret);  return ret);

        // Initialize AscendCL.
        ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d \n", ret); return ret);
        aclrtStream stream[RANK_DIM];
        for (uint32_t rankId = 0; rankId < RANK_DIM; rankId++) {
            ret = aclrtSetDevice(rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);
            ret = aclrtCreateStream(&stream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d \n", ret); return ret);
        }
        int32_t devices[RANK_DIM];
        for (int i = 0; i < RANK_DIM; i++) {
            devices[i] = i;
        }
        // Initialize the collective communication domain.
        HcclComm comms[RANK_DIM];
        ret = HcclCommInitAll(RANK_DIM, devices, comms);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll failed. ret = %d \n", ret); return ret);

        Args args[RANK_DIM];
        // Enable multi-threading.
        std::vector<std::future<int>> futures;
        for (uint32_t rankId = 0; rankId < RANK_DIM; rankId++) {
            args[rankId].rankId = rankId;
            args[rankId].hcclComm = comms[rankId];
            args[rankId].stream = stream[rankId];
            futures.push_back(std::async(std::launch::async, &LaunchOneThreadAllGatherAdd,
                                        std::ref(args[rankId]), std::ref(testData)));
        }

        int finalRet = 0;
        for (auto& future : futures) {
            int ret = future.get();
            if (ret != 0) {
                finalRet = ret;
            }
        }

        aclFinalize();
        return finalRet;
    }
    ```
