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
#include <initializer_list>
#include <limits>
#include "../../../op_kernel/arch35/qgmm_mx_ub_output_policy.h"

namespace {
using namespace QGmmMxUbPolicy;

TEST(QGmmMxUbOutputPolicy, NetworkShapesAndConservativeShapeEdges)
{
    for (int64_t m : {1792, 2048}) {
        for (int64_t n : {4864, 5120}) {
            EXPECT_TRUE(IsShapeCandidate(m, n, 256, 256, 6, 32));
        }
    }
    for (int64_t m : {4864, 5120}) {
        for (int64_t n : {3584, 3840}) {
            EXPECT_TRUE(IsShapeCandidate(m, n, 256, 256, 6, 32));
        }
    }
    EXPECT_FALSE(IsShapeCandidate(4864, 4096, 256, 256, 6, 32));
    EXPECT_FALSE(IsShapeCandidate(1792, 3584, 256, 256, 6, 32));
    EXPECT_FALSE(IsShapeCandidate(3072, 4352, 256, 256, 6, 32));
    for (int64_t m : {0, 512, 1536, 1791, 1793, 5119, 5121, 5376}) {
        EXPECT_FALSE(IsShapeCandidate(m, 5120, 256, 256, 6, 32));
    }
    for (int64_t n : {0, 512, 3328, 3583, 3585, 5119, 5121, 5376}) {
        EXPECT_FALSE(IsShapeCandidate(5120, n, 256, 256, 6, 32));
    }
}

TEST(QGmmMxUbOutputPolicy, ActualGroupKBoundsAllowKTails)
{
    for (int64_t k : {2048, 2049, 2111, 2112, 3584, 4095, 4096, 4097, 4128, 4160, 8192, 16320, 16383, 16384}) {
        EXPECT_TRUE(GroupKPolicy::UseUb(k));
    }
    for (int64_t k : {-1, 0, 64, 1024, 1792, 1984, 2047, 16385, 16448, 32768}) {
        EXPECT_FALSE(GroupKPolicy::UseUb(k));
    }
    EXPECT_FALSE(GroupKPolicy::UseUb(std::numeric_limits<int64_t>::max()));
}

TEST(QGmmMxUbOutputPolicy, SameTotalKCanContainBothPaths)
{
    EXPECT_TRUE(IsTotalKCandidate(24576, 6));
    EXPECT_FALSE(GroupKPolicy::UseUb(1792));
    EXPECT_TRUE(GroupKPolicy::UseUb(6400));
    EXPECT_TRUE(GroupKPolicy::UseUb(4096));
}

TEST(QGmmMxUbOutputPolicy, AcceptVariableGroupCountAndPositiveCoreCounts)
{
    for (uint64_t groups : {1U, 4U, 6U, 8U, 32U}) {
        for (uint64_t cores : {1U, 16U, 28U, 32U, 64U}) {
            EXPECT_TRUE(IsShapeCandidate(5120, 3584, 256, 256, groups, cores));
            EXPECT_TRUE(IsShapeCandidate(1792, 5120, 256, 256, groups, cores));
        }
    }
    EXPECT_FALSE(IsShapeCandidate(5120, 3584, 256, 256, 0, 32));
    EXPECT_FALSE(IsShapeCandidate(5120, 3584, 256, 256, 6, 0));
}

TEST(QGmmMxUbOutputPolicy, RejectUnsupportedTiles)
{
    EXPECT_FALSE(IsShapeCandidate(5120, 3584, 128, 256, 6, 32));
    EXPECT_FALSE(IsShapeCandidate(5120, 3584, 256, 128, 6, 32));
    EXPECT_FALSE(IsShapeCandidate(5120, 3584, 0, 256, 6, 32));
    EXPECT_FALSE(IsShapeCandidate(5120, 3584, 256, 0, 6, 32));
}

TEST(QGmmMxUbOutputPolicy, TotalKIsOnlyANecessaryCondition)
{
    for (int64_t groups : {1, 4, 6, 8, 32}) {
        EXPECT_TRUE(IsTotalKCandidate(groups * 2048, groups));
        EXPECT_FALSE(IsTotalKCandidate(groups * 2048 - 1, groups));
        EXPECT_TRUE(IsTotalKCandidate(groups * 16384, groups));
        EXPECT_TRUE(IsTotalKCandidate(groups * 2048 + 1, groups));
        EXPECT_FALSE(IsTotalKCandidate(groups * 16384 + 1, groups));
    }
    EXPECT_FALSE(IsTotalKCandidate(0, 6));
    EXPECT_FALSE(IsTotalKCandidate(-1, 6));
    EXPECT_FALSE(IsTotalKCandidate(24576, 0));
    EXPECT_FALSE(IsTotalKCandidate(24576, std::numeric_limits<uint64_t>::max()));
    EXPECT_FALSE(IsTotalKCandidate(std::numeric_limits<int64_t>::max(), std::numeric_limits<uint64_t>::max()));
    EXPECT_FALSE(IsTotalKCandidate(std::numeric_limits<int64_t>::max(), 6));
}

TEST(QGmmMxUbOutputPolicy, RejectInvalidShape)
{
    EXPECT_FALSE(IsShapeCandidate(std::numeric_limits<int64_t>::max(), 5120, 256, 256, 6, 32));
    EXPECT_FALSE(IsShapeCandidate(5120, std::numeric_limits<int64_t>::min(), 256, 256, 6, 32));
}
} // namespace
