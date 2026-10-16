/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "../../../../op_host/op_tiling/moe_distribute_check_win_size.h"

namespace {
Mc2Tiling::CheckWinSizeData MakeWindowData()
{
    Mc2Tiling::CheckWinSizeData data{};
    data.epWorldSize = 64;
    data.bs = 512;
    data.globalBs = 64 * 512;
    data.h = 8192;
    data.k = 16;
    data.moeExpertNum = 1024;
    data.localMoeExpertNum = 16;
    data.tpWorldSize = 1;
    return data;
}

TEST(MoeDistributeWindowSize, ExistingA3BufferRemainsSufficient)
{
    auto data = MakeWindowData();
    uint64_t dispatchStride = 0;
    uint64_t combineStride = 0;
    const uint64_t needed = Mc2Tiling::CalcMinWinSize(data, dispatchStride, combineStride);
    // The public estimator returns HCCL_BUFFSIZE=8577 MiB for this specification.
    // Its available data window is 17986224128 B; the regression required 18010341376 B.
    EXPECT_EQ(dispatchStride, 16896UL);
    EXPECT_EQ(combineStride, 16384UL);
    EXPECT_EQ(needed, 17985175552UL);
    EXPECT_LE(needed, 17986224128UL);
}

TEST(MoeDistributeWindowSize, FullMeshV2PacksDispatchButKeepsRawCombine)
{
    auto data = MakeWindowData();
    data.bs = 256;
    data.globalBs = 64 * 256;
    data.k = 8;
    data.h = 3584;
    data.moeExpertNum = 256;
    data.localMoeExpertNum = 4;
    data.isSetFullMeshV2 = true;
    uint64_t dispatchStride = 0;
    uint64_t combineStride = 0;
    EXPECT_EQ(Mc2Tiling::CalcMinWinSize(data, dispatchStride, combineStride), 1103101952UL);
    EXPECT_EQ(dispatchStride, 8192UL);
    EXPECT_EQ(combineStride, 7168UL);
}

TEST(MoeDistributeWindowSize, SharedExpertReservation)
{
    auto data = MakeWindowData();
    data.epWorldSize = 16;
    data.bs = 32;
    data.globalBs = 16 * 32;
    data.h = 3584;
    data.k = 8;
    data.moeExpertNum = 64;
    data.localMoeExpertNum = 8; // Eight routed-expert ranks and eight shared-expert ranks.
    data.sharedExpertNum = 1;
    data.isSetFullMeshV2 = true;
    uint64_t dispatchStride = 0;
    uint64_t combineStride = 0;
    EXPECT_EQ(Mc2Tiling::CalcMinWinSize(data, dispatchStride, combineStride), 71237632UL);
    EXPECT_EQ(combineStride, 7168UL);
}

TEST(MoeDistributeWindowSize, VariableBatchUsesGlobalMaximum)
{
    auto data = MakeWindowData();
    data.bs = 1;
    uint64_t dispatchStride = 0;
    uint64_t combineStride = 0;
    EXPECT_EQ(Mc2Tiling::CalcMinWinSize(data, dispatchStride, combineStride), 17985175552UL);
}

TEST(MoeDistributeWindowSize, A5RetainsPackedCombine)
{
    auto data = MakeWindowData();
    uint64_t dispatchStride = 0;
    uint64_t combineStride = 0;
    EXPECT_EQ(Mc2Tiling::CalcMinWinSize(data, dispatchStride, combineStride, 480), 18010341376UL);
    EXPECT_EQ(combineStride, 17920UL);
}

TEST(MoeDistributeWindowSize, HierarchyReservationIsUnchanged)
{
    auto data = MakeWindowData();
    data.bs = 256;
    data.globalBs = 64 * 256;
    data.isLayered = true;
    uint64_t dispatchStride = 0;
    uint64_t combineStride = 0;
    EXPECT_EQ(Mc2Tiling::CalcMinWinSize(data, dispatchStride, combineStride), 4802478080UL);
}
} // namespace
