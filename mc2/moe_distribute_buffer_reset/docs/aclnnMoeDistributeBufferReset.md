# aclnnMoeDistributeBufferReset

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

算子功能：对EP通信域做数据区与状态区的清理。若当前机器为未被隔离机器，则对其进行通信域的重置操作，对有效的die进行数据区和状态区的清0，确保后续使用时通信域不会存在已被隔离机器的数据或状态信息。

## 函数原型

每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用“aclnnMoeDistributeBufferResetGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeDistributeBufferReset”接口执行计算。

```cpp
aclnnStatus aclnnMoeDistributeBufferResetGetWorkspaceSize(
    const aclTensor *elasticInfo,
    const char      *groupEp,
    int32_t          epWorldSize,
    int32_t          needSync,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor);
```

```cpp
aclnnStatus aclnnMoeDistributeBufferReset(
    void           *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor  *executor,
    aclrtStream     stream);
```

## aclnnMoeDistributeBufferResetGetWorkspaceSize

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
    <td>elasticInfo</td>
    <td>输入</td>
    <td>Device侧的aclTensor，有效rank掩码表，标识有效rank的tensor，其中0标识本卡与对应rank链路不通，1为联通</td>
    <td>shape为(epWorldSize,)</td>
    <td>INT32</td>
    <td>ND</td>
    <td>1</td>
    <td>√</td>
    </tr>
    <tr>
    <td>groupEp</td>
    <td>输入</td>
    <td>ep通信域名称，专家并行的通信域。</td>
    <td>字符串长度范围为(0, 128)</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>epWorldSize</td>
    <td>输入</td>
    <td>通信域大小。</td>
    <td>取值支持[16, 128]内16整数倍的数值</td>
    <td>INT32</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>needSync</td>
    <td>输入</td>
    <td>是否需要全卡同步。</td>
    <td>取值支持0或1，0表示不需要，1表示需要。</td>
    <td>INT32</td>
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
    <td>传入的elasticInfo或groupEp是空指针。</td>
    </tr>
    <tr>
    <td rowspan="3" align="left">ACLNN_ERR_PARAM_INVALID</td>
    <td rowspan="3" align="left">161002</td>
    <td align="left">传入的elasticInfo的数据类型不在支持的范围内。</td>
    </tr>
    <tr><td align="left">传入的elasticInfo的数据格式不在支持的范围内。</td></tr>
    <tr><td align="left">传入的elasticInfo的shape不匹配。</td></tr>
    </tbody></table>

## aclnnMoeDistributeBufferReset

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
    <td>在Device侧申请的workspace大小，由第一段接口aclnnMoeDistributeBufferResetGetWorkspaceSize获取。</td>
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

以<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>单机为例，调起aclnnMoeDistributeBufferReset。

- 文件准备：

  1.新建resetDemo目录，按照下方指导在resetDemo下新建aclnnResetDemo.cpp，buildReset.sh文件并参考如下代码修改。

  2.安装cann包，并根据下方指导编译运行resetDemo。

-  编译脚本
    ```bash
    #!/bin/bash
    cann_path="/path/to/cann_env" # 更改cann包环境的路径
    g++ "aclnnResetDemo.cpp" -o resetDemo -I"$cann_path/latest/include/" -I"$cann_path/latest/include/aclnnop/" \
                        -L="$cann_path/latest/lib64/" -lascendcl -lnnopbase -lopapi -lop_common -lpthread -lhccl
    ```
- 编译与运行：

    ```bash
    # source cann环境
    source /path/to/cann_env/latest/bin/setenv.bash

    # 编译aclnnResetDemo.cpp
    bash buildReset.sh

    ./resetDemo
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
    #include "aclnnop/aclnn_moe_distribute_buffer_reset.h"

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
    constexpr uint32_t TIME_OUT = 1000;
    constexpr uint32_t NEED_SYNC = 0;

    constexpr uint32_t DEV_NUM = DIE_PER_SERVER * SERVER_NUM;

    struct Args {
        uint32_t rankId;
        uint32_t epRankId;
        uint32_t tpRankId;
        HcclComm hcclEpComm;
        aclrtStream resetStream;
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

    int ResetThreadFun(Args &args)
    {
        int ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] Set current context failed, ret: %d\n", ret); return ret);

        char hcomEpName[128] = {0};
        ret = HcclGetCommName(args.hcclEpComm, hcomEpName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetEpCommName failed, ret %d\n", ret); return -1);

        void *resetDeviceAddr = nullptr;
        aclTensor *resetTensor = nullptr;
        std::vector<int> resetHostData(EP_WORLD_SIZE, 1);
        std::vector<int64_t> resetShape{(int64_t)EP_WORLD_SIZE};

        ret = CreateAclTensor(resetHostData, resetShape, &resetDeviceAddr, ACL_INT32, &resetTensor);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] Failed to create reset tensor.\n"); return ret);

        uint64_t resetWorkspaceSize = 0;
        aclOpExecutor *resetExecutor = nullptr;
        void *resetWorkspaceAddr = nullptr;

        ret = aclnnMoeDistributeBufferResetGetWorkspaceSize(resetTensor, hcomEpName, EP_WORLD_SIZE, NEED_SYNC, &resetWorkspaceSize, &resetExecutor);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeBufferResetGetWorkspaceSize failed. ret = %d \n", ret); return ret);

        if (resetWorkspaceSize > 0) {
            ret = aclrtMalloc(&resetWorkspaceAddr, resetWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        ret = aclnnMoeDistributeBufferReset(resetWorkspaceAddr, resetWorkspaceSize, resetExecutor, args.resetStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnBufferReset failed. ret = %d \n", ret); return ret);
        ret = aclrtSynchronizeStreamWithTimeout(args.resetStream, TIME_OUT);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout reset failed. ret = %d \n", ret);
            return ret);

        if (resetWorkspaceSize > 0) {
            aclrtFree(resetWorkspaceAddr);
        }
        if (resetTensor != nullptr) {
            aclDestroyTensor(resetTensor);
        }
        if (resetDeviceAddr != nullptr) {
            aclrtFree(resetDeviceAddr);
        }
        LOG_PRINT("[INFO] Device %u finished reset\n", args.rankId);

        HcclCommDestroy(args.hcclEpComm);
        aclrtDestroyStream(args.resetStream);
        aclrtDestroyContext(args.context);
        aclrtResetDevice(args.rankId);

        return 0;
    }

    int main(int argc, char *argv[])
    {
        int ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed, ret: %d\n", ret); return ret);

        aclrtStream resetStream[DEV_NUM];
        aclrtContext context[DEV_NUM];

        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            ret = aclrtSetDevice(rankId % DIE_PER_SERVER);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] SetDevice failed, ret: %d\n", ret); return ret);

            ret = aclrtCreateContext(&context[rankId % DIE_PER_SERVER], rankId % DIE_PER_SERVER);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] CreateContext failed, ret: %d\n", ret); return ret);

            ret = aclrtCreateStream(&resetStream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] Create collect stream failed, ret: %d\n", ret); return ret);
        }

        int32_t devicesEp[TP_WORLD_SIZE][EP_WORLD_SIZE];
        for (int32_t tpId = 0; tpId < TP_WORLD_SIZE; tpId++) {
            for (int32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
                devicesEp[tpId][epId] = epId * TP_WORLD_SIZE + tpId;
            }
        }

        HcclComm commsEp[TP_WORLD_SIZE][EP_WORLD_SIZE];
        for (int32_t tpId = 0; tpId < TP_WORLD_SIZE; tpId++) {
            ret = HcclCommInitAll(EP_WORLD_SIZE, devicesEp[tpId], commsEp[tpId]);
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
            args[rankId].hcclEpComm = commsEp[tpRankId][epRankId];
            args[rankId].resetStream = resetStream[rankId];
            args[rankId].context = context[rankId % DIE_PER_SERVER];

            threads[rankId].reset(new(std::nothrow) std::thread(ResetThreadFun, std::ref(args[rankId])));
            CHECK_RET(threads[rankId] != nullptr, LOG_PRINT("[ERROR] Thread creation failed.\n"); return -1);
        }

        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            if (threads[rankId] && threads[rankId]->joinable()) {
                threads[rankId]->join();
            }
        }

        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            HcclCommDestroy(args[rankId].hcclEpComm);
            aclrtDestroyStream(args[rankId].resetStream);
            aclrtDestroyContext(args[rankId % DIE_PER_SERVER].context);
            aclrtResetDevice(rankId);
        }

        aclFinalize();
        LOG_PRINT("[INFO] Program finalized successfully.\n");

        return 0;
    }
    ```