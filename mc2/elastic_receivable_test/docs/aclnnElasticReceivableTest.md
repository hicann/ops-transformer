# aclnnElasticReceivableTest

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |

## 功能说明

算子功能：对一个通信域内的所有卡发送数据并写状态位，用于检测通信链路是否正常。

## 函数原型

每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用 “aclnnElasticReceivableTestGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnElasticReceivableTest”接口执行计算。

```cpp
aclnnStatus aclnnElasticReceivableTestGetWorkspaceSize(
    const aclTensor *dstRank,
    const char      *group,
    int64_t          worldSize,
    int64_t          rankNum,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor);
```

```cpp
aclnnStatus aclnnElasticReceivableTest(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnElasticReceivableTestGetWorkspaceSize

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1576px">
    <colgroup>
    <col style="width: 150px">
    <col style="width: 100px">
    <col style="width: 250px">
    <col style="width: 200px">
    <col style="width: 180px">
    <col style="width: 80px">
    <col style="width: 100px">
    <col style="width: 100px">
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
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>dstRank</td>
    <td>输入</td>
    <td>表示同一个通信域内目的server内的通信卡。</td>
    <td>shape为(rankNum,)，表示在同一个server内的卡号</td>
    <td>INT32</td>
    <td>ND</td>
    <td>1</td>
    <td>√</td>
    </tr>
    <tr>
    <td>group</td>
    <td>输入</td>
    <td>ep通信域名称，专家并行的通信域。</td>
    <td>字符串长度范围为(0, 128)</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>worldSize</td>
    <td>输入</td>
    <td>通信域大小。</td>
    <td>取值支持[16, 128]内16整数倍的数值</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>rankNum</td>
    <td>输入</td>
    <td>本端需要发送的目的server内的卡数。</td>
    <td>当前只支持16卡</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>输出</td>
    <td>返回需要在Device侧申请的workspace大小。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>输出</td>
    <td>返回op执行器，包含了算子的计算流程。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    </tbody>
    </table>

- **返回值**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

    第一段接口完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed; width: 1576px"> <colgroup>
    <col style="width: 170px">
    <col style="width: 170px">
    <col style="width: 400px">
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
    <td>传入的dstRank或group是空指针。</td>
    </tr>
    <tr>
    <td rowspan="3" align="left">ACLNN_ERR_PARAM_INVALID</td>
    <td rowspan="3" align="left">161002</td>
    <td align="left">传入的dstRank的数据类型不在支持的范围内。</td>
    </tr>
    <tr><td align="left">传入的dstRank的数据格式不在支持的范围内。</td></tr>
    <tr><td align="left">传入的dstRank的shape不匹配。</td></tr>
    </tbody></table>

## aclnnElasticReceivableTest

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1576px">
    <colgroup>
    <col style="width: 170px">
    <col style="width: 170px">
    <col style="width: 800px">
    </colgroup>
    <thead>
    <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>workspace</td>
    <td>输入</td>
    <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>输入</td>
    <td>在Device侧申请的workspace大小，由第一段接口aclnnElasticReceivableTestGetWorkspaceSize获取。</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>输入</td>
    <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
    <td>stream</td>
    <td>输入</td>
    <td>指定执行任务的Stream。</td>
    </tr>
    </tbody>
    </table>

- **返回值**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明

无

## 调用示例

以<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>单机为例，调起aclnnElasticReceivableTest。

- 文件准备：

  1.新建testDemo目录，按照下方指导在testDemo下新建aclnnTestDemo.cpp，buildTest.sh文件并参考如下代码修改。

  2.安装cann包，并根据下方指导编译运行testDemo。

-  编译脚本
    ```bash
    #!/bin/bash
    cann_path="/path/to/cann_env" # 更改cann包环境的路径
    g++ "aclnnTestDemo.cpp" -o testDemo -I"$cann_path/latest/include/" -I"$cann_path/latest/include/aclnnop/" \
                        -L="$cann_path/latest/lib64/" -lascendcl -lnnopbase -lopapi -lop_common -lpthread -lhccl
    ```
- 编译与运行：

    ```bash
    # source cann环境
    source /path/to/cann_env/latest/bin/setenv.bash

    # 编译aclnnTestDemo.cpp
    bash buildTest.sh

    ./testDemo
    ```

- 示例代码如下，仅供参考
    ```Cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <vector>
    #include <cstring>
    #include <unordered_set>
    #include <bits/stdc++.h>
    #include "acl/acl.h"
    #include "hccl/hccl.h"
    #include "aclnnop/aclnn_elastic_receivable_test.h"

    #define CHECK_RET(cond, return_expr) \
        do {                             \
            if (!(cond)) {               \
                return_expr;             \
            }                            \
        } while (0)

    #define LOG_PRINT(message, ...)         \
        do {                                \
            printf(message, ##__VA_ARGS__); \
        } while (0)

    constexpr int DIE_PER_SERVER = 16;
    constexpr uint32_t SERVER_NUM = 1;
    constexpr uint32_t WORLD_SIZE = 16;
    constexpr uint32_t EP_WORLD_SIZE = WORLD_SIZE * SERVER_NUM;
    constexpr uint32_t TP_WORLD_SIZE = 1;
    constexpr uint32_t TIME_OUT = 10000;

    constexpr uint32_t DEV_NUM = DIE_PER_SERVER * SERVER_NUM;

    struct Args {
        uint32_t rankId;
        uint32_t epRankId;
        uint32_t tpRankId;
        HcclComm hcclTestComm;
        aclrtStream testStream;
        aclrtContext context;
    };

    int64_t GetShapeSize(const std::vector<int64_t> &shape)
    {
        int64_t shape_size = 1;
        for (auto i : shape) {
            shape_size *= i;
        }
        return shape_size;
    }

    void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
        auto size = GetShapeSize(shape);
        std::vector<int> resultData(size, 0);
        auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                                *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
        LOG_PRINT("[");
        for (int64_t i = 0; i < size; i++) {
            LOG_PRINT("%d,", resultData[i]);
        }
        LOG_PRINT("]\n");
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
        const int64_t stride = 1; 
        *tensor = aclCreateTensor(
            shape.data(), shape.size(), dataType, &stride, 0, 
            aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
        return 0;
    }

    int DetectionThreadFun(Args &args)
    {
        int ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] Set current context failed, ret: %d\n", ret); return ret);

        char hcomTestName[128] = {0};
        ret = HcclGetCommName(args.hcclTestComm, hcomTestName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetTpCommName failed, ret %d\n", ret); return -1);

        /**************************************** 每个server调用test ********************************************/
        for (int32_t server = 0; server < EP_WORLD_SIZE / DIE_PER_SERVER; ++server) {
            void *destRankDeviceAddr = nullptr;
            aclTensor *destRank = nullptr;
            std::vector<int32_t> destRankHostData(DIE_PER_SERVER);
            std::iota(destRankHostData.begin(), destRankHostData.end(), server * DIE_PER_SERVER);
            
            std::vector<int64_t> destRankShape{(int64_t)DIE_PER_SERVER};
            ret = CreateAclTensor(destRankHostData, destRankShape, &destRankDeviceAddr, ACL_INT32, &destRank);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] Failed to create tensor\n"); return ret);

            uint64_t workspaceSize = 0;
            aclOpExecutor *executor = nullptr;
            void *workspaceAddr = nullptr;

            ret = aclnnElasticReceivableTestGetWorkspaceSize(destRank, hcomTestName, EP_WORLD_SIZE, DIE_PER_SERVER, &workspaceSize, &executor);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnTestGetWorkspaceSize failed. ret = %d \n", ret); return ret);

            if (workspaceSize > 0) {
                ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
            }
            // 调用第二阶段接口
            ret = aclnnElasticReceivableTest(workspaceAddr, workspaceSize, executor, args.testStream);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnTest failed. ret = %d \n", ret);  \
                return ret);
            ret = aclrtSynchronizeStreamWithTimeout(args.testStream, TIME_OUT);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
                return ret);
            PrintOutResult(destRankShape, &destRankDeviceAddr);
            // 释放device资源
            if (workspaceSize > 0) {
                aclrtFree(workspaceAddr);
            }
            if (destRank != nullptr) {
                aclDestroyTensor(destRank);
            }
            if (destRankDeviceAddr != nullptr) {
                aclrtFree(destRankDeviceAddr);
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(TIME_OUT));
        
        HcclCommDestroy(args.hcclTestComm);
        aclrtDestroyStream(args.testStream);
        aclrtDestroyContext(args.context);
        aclrtResetDevice(args.rankId);

        LOG_PRINT("[INFO] Device %u completed all steps\n", args.rankId);

        return 0;
    }

    int main(int argc, char *argv[])
    {
        int ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed, ret: %d\n", ret); return ret);

        aclrtStream testStream[DEV_NUM];
        aclrtContext context[DEV_NUM];

        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            ret = aclrtSetDevice(rankId % DIE_PER_SERVER);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] SetDevice failed, ret: %d\n", ret); return ret);

            ret = aclrtCreateContext(&context[rankId % DIE_PER_SERVER], rankId % DIE_PER_SERVER);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] CreateContext failed, ret: %d\n", ret); return ret);

            ret = aclrtCreateStream(&testStream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] Create test stream failed, ret: %d\n", ret); return ret);
        }

        int32_t devicesTest[TP_WORLD_SIZE][EP_WORLD_SIZE];
        for (int32_t tpId = 0; tpId < TP_WORLD_SIZE; tpId++) {
            for (int32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
                devicesTest[tpId][epId] = epId * TP_WORLD_SIZE + tpId;
            }
        }

        HcclComm commsTest[TP_WORLD_SIZE][EP_WORLD_SIZE];
        for (int32_t tpId = 0; tpId < TP_WORLD_SIZE; tpId++) {
            ret = HcclCommInitAll(EP_WORLD_SIZE, devicesTest[tpId], commsTest[tpId]);
            CHECK_RET(ret == ACL_SUCCESS,
                        LOG_PRINT("[ERROR] HcclCommInitAll ep %d failed, ret %d\n", tpId, ret); return ret);
        }

        Args args[DEV_NUM];
        std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);

        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            uint32_t epRankId = rankId / TP_WORLD_SIZE;
            uint32_t tpRankId = rankId % TP_WORLD_SIZE;
            LOG_PRINT("[INFO] RankId %d prepare args.\n", rankId);

            args[rankId].rankId = rankId;
            args[rankId].epRankId = epRankId;
            args[rankId].tpRankId = tpRankId;
            args[rankId].hcclTestComm = commsTest[tpRankId][epRankId];
            args[rankId].testStream = testStream[rankId];
            args[rankId].context = context[rankId % DIE_PER_SERVER];

            threads[rankId].reset(new(std::nothrow) std::thread(DetectionThreadFun, std::ref(args[rankId])));
            CHECK_RET(threads[rankId] != nullptr, LOG_PRINT("[ERROR] Thread creation failed.\n"); return -1);
        }

        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            if (threads[rankId] && threads[rankId]->joinable()) {
                threads[rankId]->join();
            }
        }

        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            HcclCommDestroy(args[rankId].hcclTestComm);
            aclrtDestroyStream(args[rankId].testStream);
            aclrtDestroyContext(args[rankId % DIE_PER_SERVER].context);
            aclrtResetDevice(rankId);
        }

        aclFinalize();
        LOG_PRINT("[INFO] Program finalized successfully.\n");

        return 0;
    }
    ```