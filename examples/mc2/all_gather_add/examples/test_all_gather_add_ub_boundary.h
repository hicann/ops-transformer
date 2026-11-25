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
 * \file test_all_gather_add_ub_boundary.h
 * \brief AllGatherAdd UB 容量边界用例（仅 A5 MTE 路径有意义）。
 *        用大 shape 越过 MTE kernel 的 UB 容量，验证 tiling 封顶分块后仍能正确跑出结果
 *        （而非 UB 越界崩溃或被拒绝）。
 */

#ifndef TEST_ALL_GATHER_ADD_UB_BOUNDARY_H
#define TEST_ALL_GATHER_ADD_UB_BOUNDARY_H

#include "test_all_gather_add_small_shape.h"

static int RunUbBoundaryTest(Args& args)
{
    int ret = aclrtSetDevice(args.rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);

    // 大 shape：totalElemNum = 245760×16 = 3932160，blockDim=64 时每核 61440 half，
    // 原始 2 tile（30720 half/tile）会 UB 越界；封顶分块后切成 3 个 tile（21120+21120+19200）。
    const std::vector<int64_t> largeAShape = {122880, 16};
    const std::vector<int64_t> largeBShape = {245760, 16};
    int64_t largeAShapeSize = GetShapeSize(largeAShape);
    int64_t largeBShapeSize = GetShapeSize(largeBShape);

    // 生成大 shape 的 golden（确定性输入，避免随机开销）
    TestData largeData = {
        std::vector<op::fp16_t>(largeAShapeSize, 0.0f), // rank0_a
        std::vector<op::fp16_t>(largeBShapeSize, 0.0f), // rank0_b
        std::vector<op::fp16_t>(largeAShapeSize, 0.0f), // rank1_a
        std::vector<op::fp16_t>(largeBShapeSize, 0.0f), // rank1_b

        std::vector<op::fp16_t>(largeBShapeSize, 0.0f), // gatherOut
        std::vector<op::fp16_t>(largeBShapeSize, 0.0f), // rank0_c
        std::vector<op::fp16_t>(largeBShapeSize, 0.0f)  // rank1_c
    };
    ret = GenerateConstTestData(largeData, 1.0f, 2.0f, 3.0f, 4.0f);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] GenerateConstTestData failed. ret = %d \n", ret); return ret);

    // 创建大 shape tensor
    TensorSet ts;
    ret = CreateTensorSet(ts, largeAShape, largeBShape);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] rank %d create tensors failed. ret = %d\n", args.rankId, ret);
              return ret);

    // 执行 + 校验（预期正确跑出结果，而非被拒绝）
    ret = RunOneRound(args, largeData, "ub_boundary", ts.a, ts.b, ts.out, ts.gatherOut, ts.aDeviceAddr, ts.bDeviceAddr,
                      ts.outDeviceAddr, ts.gatherOutDeviceAddr, largeAShapeSize, largeBShapeSize);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[ERROR] UB boundary test failed on rank %d. ret = %d\n", args.rankId, ret);
    }

    auto hcclRet = HcclCommDestroy(args.hcclComm);
    CHECK_RET(hcclRet == HCCL_SUCCESS, LOG_PRINT("[ERROR] HcclCommDestroy failed. ret = %d \n", hcclRet));
    DestroyTensorSet(ts);
    ret = aclrtDestroyStream(args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtDestroyStream failed. ret = %d \n", ret); return ret);
    ret = aclrtResetDevice(args.rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtResetDevice failed. ret = %d \n", ret); return ret);
    return ret;
}

#endif // TEST_ALL_GATHER_ADD_UB_BOUNDARY_H
