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
 * \file test_all_gather_add_timeout.h
 * \brief AllGatherAdd WaitPeers 超时失败用例（仅 A5 MTE 路径）。
 *        rank0 跑算子、rank1 只 GetWorkspaceSize 但不 launch（缺失 READY），
 *        预期 rank0 WaitPeers 超时后 trap，host 侧 sync 报错并确认错误码。
 */

#ifndef TEST_ALL_GATHER_ADD_TIMEOUT_H
#define TEST_ALL_GATHER_ADD_TIMEOUT_H

#include "test_all_gather_add_small_shape.h"

// 与 kernel 侧一致：WaitPeers 超时写进 gatherOut 的唯一错误码
constexpr uint32_t MTE_TIMEOUT_ERR_CODE = 0xDEAD0001U;

static int RunTimeoutTest(Args& args, const TestData& testData, std::atomic<bool>& rank0Done)
{
    int ret = aclrtSetDevice(args.rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);

    TensorSet ts;
    ret = CreateTensorSet(ts, aShape, bShape);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] rank %d create tensors failed. ret = %d\n", args.rankId, ret);
              return ret);

    // 两 rank 都做 GetWorkspaceSize，保证对端窗口已分配
    char hcomName[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret: %d\n", ret); return -1);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAllGatherAddGetWorkspaceSize(ts.a, ts.b, hcomName, RANK_DIM, ts.out, ts.gatherOut, &workspaceSize,
                                            &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] GetWorkspaceSize failed. ret = %d \n", ret); return ret);

    int finalRet = 0;
    if (args.rankId == 0) {
        // RAII：rank0 无论后续走哪条路径（含 CHECK_RET 提前 return），析构时都会设置 rank0Done，
        // 避免 rank1 因 rank0 提前失败（如 launch/workspace 失败）而永远死等。
        struct Rank0DoneGuard {
            std::atomic<bool>& flag;
            explicit Rank0DoneGuard(std::atomic<bool>& f)
                : flag(f)
            {}
            ~Rank0DoneGuard()
            {
                flag.store(true, std::memory_order_release);
            }
        } doneGuard{rank0Done};

        // rank0 launch，预期 trap
        void* workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret);
                      return ret);
        }
        ret = aclnnAllGatherAdd(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnAllGatherAdd failed. ret = %d \n", ret); return ret);
        ret = aclrtSynchronizeStream(args.stream); // 阻塞等 kernel（预期 trap 报错）

        if (ret == ACL_SUCCESS) {
            // 判据 1：sync 必须报错
            LOG_PRINT("[ERROR] timeout test failed: expected trap but sync succeeded.\n");
            finalRet = -1;
        } else {
            // 判据 2：读回 gatherOut 首个 uint32，必须等于 MTE_TIMEOUT_ERR_CODE 才确认是 WaitPeers 超时
            uint32_t errCode = 0;
            int memcpyRet = aclrtMemcpy(&errCode, sizeof(errCode), ts.gatherOutDeviceAddr, sizeof(errCode),
                                        ACL_MEMCPY_DEVICE_TO_HOST);
            if (memcpyRet == ACL_SUCCESS && errCode == MTE_TIMEOUT_ERR_CODE) {
                LOG_PRINT("[INFO] timeout test passed: sync error %d + errCode 0x%x confirmed.\n", ret, errCode);
                finalRet = 0;
            } else {
                LOG_PRINT("[ERROR] timeout test failed: sync error %d but errCode 0x%x (memcpy ret %d), "
                          "not confirmed as WaitPeers timeout.\n",
                          ret, errCode, memcpyRet);
                finalRet = -1;
            }
        }
        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
    } else {
        // rank1 跳过 launch（缺失 READY），但必须等 rank0 记录结果后再统一清理
        while (!rank0Done.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    auto hcclRet = HcclCommDestroy(args.hcclComm);
    CHECK_RET(hcclRet == HCCL_SUCCESS, LOG_PRINT("[ERROR] HcclCommDestroy failed. ret = %d \n", hcclRet));
    DestroyTensorSet(ts);
    ret = aclrtDestroyStream(args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtDestroyStream failed. ret = %d \n", ret); return ret);
    ret = aclrtResetDevice(args.rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtResetDevice failed. ret = %d \n", ret); return ret);
    return finalRet;
}

#endif // TEST_ALL_GATHER_ADD_TIMEOUT_H
