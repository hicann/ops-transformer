# aclnnAllGatherAdd

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/examples/mc2/all_gather_add)

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：不支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- **接口功能**：完成[AllGather](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/ascendcopapi/atlasascendc_api_07_0873.html)通信和[Add](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/ascendcopapi/atlasascendc_api_07_0035.html)加法的融合。
- **计算公式**：

    $$
    gatherOut=AllGather(a0, a1)
    $$

    $$
    c[i]=gatherOut[i] + b[i]
    $$

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnAllGatherAddGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnAllGatherAdd”接口执行计算。

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

- **参数说明：**

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
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
        <th>使用说明</th>
        <th>数据类型</th>
        <th>数据格式</th>
        <th>维度(shape)</th>
        <th>非连续Tensor</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>a (aclTensor*)</td>
        <td>输入</td>
        <td>计算公式中的a0或a1。</td>
        <td>当前版本仅支持二维shape输入，且仅支持不转置场景。</td>
        <td>FLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
    </tr>
    <tr>
        <td>b (aclTensor*)</td>
        <td>输入</td>
        <td>计算公式中的b。</td>
        <td>当前版本仅支持二维shape输入，且仅支持不转置场景。</td>
        <td>FLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
    </tr>
    <tr>
        <td>group (char*)</td>
        <td>输入</td>
        <td>通信域名称。</td>
        <td>通过Hccl提供的接口“extern HcclResult HcclGetCommName(HcclComm comm, char* commName);”获取，其中commName即为group。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>rankSize (int64_t)</td>
        <td>输入</td>
        <td>标识通信域内参与AllGather通信的npu卡数。</td>
        <td>当前版本仅支持输入2。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>cOut (aclTensor*)</td>
        <td>输出</td>
        <td>AllGather通信与Add计算的结果，即计算公式中的c。</td>
        <td>与输入的数据类型保持一致。</td>
        <td>FLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
    </tr>
    <tr>
        <td>gatherOutOut (aclTensor*)</td>
        <td>输出</td>
        <td>仅输出AllGather通信后的结果，即计算公式中的gatherOut。</td>
        <td>与输入的数据类型保持一致。</td>
        <td>FLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
    </tr>
    <tr>
        <td>workspaceSize (uint64_t*)</td>
        <td>输出</td>
        <td>返回需要在Device侧申请的workspace大小。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>executor (aclOpExecutor**)</td>
        <td>输出</td>
        <td>返回op执行器，包含了算子计算流程。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    </tbody></table>

- **返回值：**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

    第一段接口完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed; width: 1166px"> <colgroup>
    <col style="width: 267px">
    <col style="width: 124px">
    <col style="width: 775px">
    </colgroup>
    <thead>
    <tr>
        <th>返回值</th>
        <th>错误码</th>
        <th>描述</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>传入的a、b或cOut是空指针。</td>
    </tr>
    <tr>
        <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="3">161002</td>
        <td>a, b, gatherout或output的数据类型不在支持的范围之内，或shape不合法。</td>
    </tr>
    </tbody></table>

## aclnnAllGatherAdd

- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1166px"> <colgroup>
    <col style="width: 173px">
    <col style="width: 133px">
    <col style="width: 860px">
    </colgroup>
    <thead>
    <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>workspace</td>
        <td>输入</td>
        <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>输入</td>
        <td>在Device侧申请的workspace大小，由第一段接口<code>aclnnAllGatherAddGetWorkspaceSize</code>获取。</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>输入</td>
        <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
        <td>stream</td>
        <td>输入</td>
        <td>指定执行任务的stream。</td>
    </tr>
    </tbody></table>

- **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)

## 约束说明

- 确定性计算：allGatherAdd算子默认确定性实现。
- 当前该示例算子仅支持固定shape：a(240, 256)，b(240 * 2, 256)，和固定rank_size = 2。
- 所有输入不支持空tensor场景，取值范围在[-5,5]之间。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考本算子README文件调用说明中的[编译与运行样例](../README.md)。

说明：本示例代码调用了部分HCCL集合通信库接口：HcclGetCommName、HcclCommInitAll、HcclCommDestroy,请参考[ <<HCCL API (C)>>](https://hiascend.com/document/redirect/CannCommunityHcclCppApi)。

<!-- npu="A3,910b" id7 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：

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

    // 本示例固定shape
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
        const float tolerance = 0.001f; // 千分之一
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

        // 根据随机生成的测试数据填充host侧输入
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

        // 调用第一阶段接口
        ret = aclnnAllGatherAddGetWorkspaceSize(
            a, b, hcomName, RANK_DIM, out, gatherOut, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnAllGatherAddGetWorkspaceSize failed. ret = %d \n", ret); return ret);
        // 根据第一阶段接口计算出的workspaceSize申请device内存
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret);  return ret);
        }
        // 调用第二阶段接口
        ret = aclnnAllGatherAdd(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnAllGatherAdd failed. ret = %d \n", ret); return ret);
        // 同步等待任务执行结束
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
            return ret);
        LOG_PRINT("[INFO] device_%d aclnnAllGatherAdd execute successfully.\n", args.rankId);

        // 算子计算结果与golden数据进行对比
        std::vector<op::fp16_t> gatherOutData(bShapeSize, 0);
        // 将计算结果从device侧拷贝到host侧
        ret = aclrtMemcpy(gatherOutData.data(), bShapeSize * sizeof(gatherOutData[0]), gatherOutDeviceAddr,
                        bShapeSize * sizeof(gatherOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        // 比较AllGather结果
        ret = CompareVector(gatherOutData, testData.gatherOut);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] gatherOut compare failed. ret = %d \n", ret); return ret);

        std::vector<op::fp16_t> outputData(bShapeSize, 0);
        ret = aclrtMemcpy(outputData.data(), bShapeSize * sizeof(outputData[0]), outDeviceAddr,
                        bShapeSize * sizeof(outputData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        // 比较AllGatherAdd结果
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
        // 释放device资源，需要根据具体API的接口定义修改
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

    // 随机生成[-5, 5]之间的数据填充vector
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

    // 拼接两个大小相同的vector
    int GatherVectors(std::vector<op::fp16_t> &vec1, std::vector<op::fp16_t> &vec2, std::vector<op::fp16_t> &vec3)
    {
        vec3.clear();
        vec3.reserve(vec1.size() + vec2.size());
        vec3.insert(vec3.end(), vec1.begin(), vec1.end());
        vec3.insert(vec3.end(), vec2.begin(), vec2.end());
        return 0;
    }

    // 对两个大小相同的vector执行加法
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
        // 随机生成输入
        int ret = RandomVectorGenerator(testData.rank0_a, testData.rank0_a.size());
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] RandomVectorGenerate rank0_a failed. ret = %d \n", ret);  return ret);
        ret = RandomVectorGenerator(testData.rank0_b, testData.rank0_b.size());
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] RandomVectorGenerate rank0_b failed. ret = %d \n", ret);  return ret);
        ret = RandomVectorGenerator(testData.rank1_a, testData.rank1_a.size());
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] RandomVectorGenerate rank1_a failed. ret = %d \n", ret);  return ret);
        ret = RandomVectorGenerator(testData.rank1_b, testData.rank1_b.size());
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] RandomVectorGenerate rank1_b failed. ret = %d \n", ret);  return ret);

        // 计算golden数据
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
        // 生成测试数据
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

        // AscendCL初始化
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
        // 初始化集合通信域
        HcclComm comms[RANK_DIM];
        ret = HcclCommInitAll(RANK_DIM, devices, comms);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll failed. ret = %d \n", ret); return ret);

        Args args[RANK_DIM];
        // 启动多线程
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

<!-- end id7 -->
