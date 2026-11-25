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
 * \file test_all_gather_add_align_reject.h
 * \brief AllGatherAdd 每核 240 对齐拒绝用例（仅 A5 MTE 路径）。
 *        用每核数据量非 240 对齐的 shape，验证 tiling 正确拒绝。
 */

#ifndef TEST_ALL_GATHER_ADD_ALIGN_REJECT_H
#define TEST_ALL_GATHER_ADD_ALIGN_REJECT_H

#include "test_all_gather_add_common.h"

static int RunAlignRejectTest(Args& args)
{
    int ret = aclrtSetDevice(args.rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);

    // 每核 240 未对齐的 shape：a=[64,30], b=[128,30], rankSize=2。
    // totalElemNum=3840，aElemsPerRank=1920（240 对齐），blockElemNumTotal=3840/64=60（非 240 对齐）。
    const std::vector<int64_t> shapeA = {64, 30};
    const std::vector<int64_t> shapeB = {128, 30};

    TensorSet ts;
    ret = CreateTensorSet(ts, shapeA, shapeB);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] rank %d create tensors failed. ret = %d\n", args.rankId, ret);
              return ret);

    // 预期 tiling 因「每核数据量非 240 对齐」拒绝（GRAPH_FAILED → GetWorkspaceSize 报错）
    char hcomName[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret: %d\n", ret); return -1);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAllGatherAddGetWorkspaceSize(ts.a, ts.b, hcomName, RANK_DIM, ts.out, ts.gatherOut, &workspaceSize,
                                            &executor);

    int finalRet = 0;
    if (ret == ACL_SUCCESS) {
        LOG_PRINT("[ERROR] align reject test failed: expected tiling rejection but GetWorkspaceSize succeeded.\n");
        finalRet = -1;
    } else {
        // 未对齐输入被 tiling 拒绝：明确提示用户原因，并停止继续执行（不再 launch）
        LOG_PRINT("[INFO] input rejected: per-core data (blockElemNumTotal) is not divisible by 240.\n");
        LOG_PRINT("[INFO] align reject test passed: GetWorkspaceSize rejected with ret %d as expected.\n", ret);
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

#endif // TEST_ALL_GATHER_ADD_ALIGN_REJECT_H
