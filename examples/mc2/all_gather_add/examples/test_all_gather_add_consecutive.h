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
 * \file test_all_gather_add_consecutive.h
 * \brief AllGatherAdd 连续两次调用用例（仅 A5 MTE 路径）。
 *        同一通信域连续调用、两轮输入不同并错峰，验证 0/1 双缓冲：快 rank 覆盖新半区不踩慢 rank 旧半区。
 */

#ifndef TEST_ALL_GATHER_ADD_CONSECUTIVE_H
#define TEST_ALL_GATHER_ADD_CONSECUTIVE_H

#include "test_all_gather_add_small_shape.h"

static int LaunchConsecutiveTest(Args& args, const TestData& round1, const TestData& round2)
{
    int ret = aclrtSetDevice(args.rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);

    // 创建 tensor（两轮复用）
    TensorSet ts;
    ret = CreateTensorSet(ts, aShape, bShape);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] rank %d create tensors failed. ret = %d\n", args.rankId, ret);
              return ret);

    // 第 1 轮：随机数据
    ret = RunOneRound(args, round1, "round1", ts.a, ts.b, ts.out, ts.gatherOut, ts.aDeviceAddr, ts.bDeviceAddr,
                      ts.outDeviceAddr, ts.gatherOutDeviceAddr, aShapeSize, bShapeSize);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] round1 failed on rank %d. ret = %d\n", args.rankId, ret);
              return ret);

    // 第 2 轮：同一通信域连续调用、输入不同；rank1 略延后制造两卡错峰（rank0 先进入第 2 轮）。
    if (args.rankId == 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    ret = RunOneRound(args, round2, "round2", ts.a, ts.b, ts.out, ts.gatherOut, ts.aDeviceAddr, ts.bDeviceAddr,
                      ts.outDeviceAddr, ts.gatherOutDeviceAddr, aShapeSize, bShapeSize);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] round2 failed on rank %d. ret = %d\n", args.rankId, ret);
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

#endif // TEST_ALL_GATHER_ADD_CONSECUTIVE_H
