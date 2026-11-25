/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_all_gather_add.cpp
 * \brief AllGatherAdd 测试入口：按平台/开关选择用例。
 *        本文件仅保留 BASIC（随机矩阵单次调用）用例，其余用例在独立 header 中：
 *        CONSECUTIVE → test_all_gather_add_consecutive.h
 *        TIMEOUT     → test_all_gather_add_timeout.h
 *        UB_BOUNDARY → test_all_gather_add_ub_boundary.h
 *        共享 helper → test_all_gather_add_common.h / test_all_gather_add_small_shape.h
 */

#include "test_all_gather_add_common.h"
#include "test_all_gather_add_small_shape.h"
#include "test_all_gather_add_consecutive.h"
#include "test_all_gather_add_timeout.h"
#include "test_all_gather_add_ub_boundary.h"
#include "test_all_gather_add_align_reject.h"

// ==================== 测试用例开关 ====================
// 按运行平台自动选择用例（也可手动把 kTestMode 改成其它值强制跑某个）：
//   A5(950)      → CONSECUTIVE：连续两次调用，验证 MTE 0/1 双缓冲
//   A2/A3(910b)  → BASIC：单次调用（高层 Hccl 路径）
//   TIMEOUT      → 对端缺失 READY → WaitPeers 超时 trap（仅 A5 有意义，跑得慢）
//   UB_BOUNDARY  → 大 shape 越过 MTE UB 容量 → 封顶分块后正确出结果（仅 A5 MTE 路径）
//   ALIGN_REJECT → 每核数据量非 240 对齐 → tiling 拒绝（仅 A5 MTE 路径）
enum class TestMode {
    AUTO = 0,
    BASIC,
    CONSECUTIVE,
    TIMEOUT,
    UB_BOUNDARY,
    ALIGN_REJECT,
};
static const TestMode kTestMode = TestMode::AUTO;

static bool IsA5Platform()
{
    const char* socName = aclrtGetSocName();
    return socName != nullptr && std::string(socName).find("950") != std::string::npos;
}

static TestMode ResolveTestMode()
{
    if (kTestMode != TestMode::AUTO) {
        return kTestMode;
    }
    return IsA5Platform() ? TestMode::CONSECUTIVE : TestMode::BASIC;
}

// BASIC 用例：随机矩阵单次调用（两平台通用，A2/A3 高层 Hccl / A5 MTE 均走单次调用）
static int LaunchBasicTest(Args& args, const TestData& round1)
{
    int ret = aclrtSetDevice(args.rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);

    // 创建 tensor（真实输入由 RunOneRound 填充）
    TensorSet ts;
    ret = CreateTensorSet(ts, aShape, bShape);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] rank %d create tensors failed. ret = %d\n", args.rankId, ret);
              return ret);

    // 单次调用
    ret = RunOneRound(args, round1, "round1", ts.a, ts.b, ts.out, ts.gatherOut, ts.aDeviceAddr, ts.bDeviceAddr,
                      ts.outDeviceAddr, ts.gatherOutDeviceAddr, aShapeSize, bShapeSize);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] round1 failed on rank %d. ret = %d\n", args.rankId, ret);
              return ret);

    auto hcclRet = HcclCommDestroy(args.hcclComm);
    CHECK_RET(hcclRet == HCCL_SUCCESS, LOG_PRINT("[ERROR] HcclCommDestroy failed. ret = %d \n", hcclRet));
    DestroyTensorSet(ts);
    ret = aclrtDestroyStream(args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtDestroyStream failed. ret = %d \n", ret); return ret);
    ret = aclrtResetDevice(args.rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtResetDevice failed. ret = %d \n", ret); return ret);
    return 0;
}

int main(int argc, char* argv[])
{
    // 生成测试数据：第 1 轮随机（BASIC/TIMEOUT 用），第 2 轮确定性（CONSECUTIVE 用）
    TestData round1 = {
        std::vector<op::fp16_t>(aShapeSize, 0.0f), // rank0_a
        std::vector<op::fp16_t>(bShapeSize, 0.0f), // rank0_b
        std::vector<op::fp16_t>(aShapeSize, 0.0f), // rank1_a
        std::vector<op::fp16_t>(bShapeSize, 0.0f), // rank1_b

        std::vector<op::fp16_t>(bShapeSize, 0.0f), // gatherOut
        std::vector<op::fp16_t>(bShapeSize, 0.0f), // rank0_c
        std::vector<op::fp16_t>(bShapeSize, 0.0f)  // rank1_c
    };
    TestData round2 = {
        std::vector<op::fp16_t>(aShapeSize, 0.0f), // rank0_a
        std::vector<op::fp16_t>(bShapeSize, 0.0f), // rank0_b
        std::vector<op::fp16_t>(aShapeSize, 0.0f), // rank1_a
        std::vector<op::fp16_t>(bShapeSize, 0.0f), // rank1_b

        std::vector<op::fp16_t>(bShapeSize, 0.0f), // gatherOut
        std::vector<op::fp16_t>(bShapeSize, 0.0f), // rank0_c
        std::vector<op::fp16_t>(bShapeSize, 0.0f)  // rank1_c
    };
    int ret = GenerateTestData(round1);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] GenerateTestData failed. ret = %d \n", ret); return ret);
    ret = GenerateConstTestData(round2, 3.0f, 4.0f, 1.0f, 2.0f);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] GenerateConstTestData failed. ret = %d \n", ret); return ret);

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

    TestMode mode = ResolveTestMode();
    LOG_PRINT("[INFO] test mode = %d (1=BASIC, 2=CONSECUTIVE, 3=TIMEOUT, 4=UB_BOUNDARY, 5=ALIGN_REJECT)\n",
              static_cast<int>(mode));

    Args args[RANK_DIM];
    std::atomic<bool> rank0Done{false}; // 超时用例：rank1 等 rank0 记录结果后再统一清理
    // 启动多线程
    std::vector<std::future<int>> futures;
    for (uint32_t rankId = 0; rankId < RANK_DIM; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].hcclComm = comms[rankId];
        args[rankId].stream = stream[rankId];
        if (mode == TestMode::BASIC) {
            futures.push_back(
                std::async(std::launch::async, &LaunchBasicTest, std::ref(args[rankId]), std::ref(round1)));
        } else if (mode == TestMode::CONSECUTIVE) {
            futures.push_back(std::async(std::launch::async, &LaunchConsecutiveTest, std::ref(args[rankId]),
                                         std::ref(round1), std::ref(round2)));
        } else if (mode == TestMode::TIMEOUT) {
            // rank0 跑算子、rank1 跳过（缺失 READY）；rank1 等 rank0 记录结果后再清理
            futures.push_back(std::async(std::launch::async, &RunTimeoutTest, std::ref(args[rankId]), std::ref(round1),
                                         std::ref(rank0Done)));
        } else if (mode == TestMode::UB_BOUNDARY) {
            // UB 容量边界：大 shape 越过 MTE UB，封顶分块后正确出结果
            futures.push_back(std::async(std::launch::async, &RunUbBoundaryTest, std::ref(args[rankId])));
        } else {
            // 每核 240 对齐：每核数据量非 240 对齐 → tiling 拒绝
            futures.push_back(std::async(std::launch::async, &RunAlignRejectTest, std::ref(args[rankId])));
        }
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
