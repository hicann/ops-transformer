/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include "../../../../op_host/op_api/aclnn_mega_mhc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/op_errno.h"

using namespace op;

namespace {
struct Case {
    bool mid = false;
    bool missingGamma = false;
    bool invalidGamma = false;
    bool externalMix = false;
    bool missingPreOutput = false;
    int64_t preOutputLast = 1;
    aclDataType preOutputType = ACL_FLOAT;
    bool sequence = false;
    bool wrongMixRank = false;
    int64_t mixTokens = 2;
    int64_t mixLast = 1;
    aclDataType mixType = ACL_FLOAT;
    bool reservedOutput = false;
    bool wrongOutputShape = false;
    aclDataType type = ACL_FLOAT16;
    int64_t d = 2560;
    int64_t mode = 0;
    int64_t outputMode = 0;
    int64_t heads = 4;
    int64_t iterations = 0;
    double epsilon = 1e-6;
    double preNormEpsilon = 1e-6;
    double gateEpsilon = 1e-6;
    double sinkhornEpsilon = 1e-6;
    double rmsScale = 1.0;
};

aclnnStatus Workspace(const Case& c)
{
    const std::vector<int64_t> leading = c.sequence ? std::vector<int64_t>{1, 2} : std::vector<int64_t>{2};
    auto shape = [&](std::initializer_list<int64_t> tail) {
        auto dims = leading;
        dims.insert(dims.end(), tail);
        return dims;
    };
    auto x = TensorDesc(shape({c.d}), c.type, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto residual = TensorDesc(shape({4, c.d}), c.type, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto post = TensorDesc(shape({4, 1}), ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto fn = TensorDesc({24, 4 * c.d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-0.01, 0.01);
    auto scales = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 0.2);
    auto bases = TensorDesc({24}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-0.1, 0.1);
    auto gamma = TensorDesc({c.d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 1);
    auto invalidGamma = TensorDesc({c.d + 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 1);
    auto gammaMid = TensorDesc({4 * c.d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 1);
    auto comb = TensorDesc(shape({4, 4}), ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.25, 0.25);
    auto mixShape = shape({4, c.mixLast});
    mixShape[c.sequence ? 1U : 0U] = c.mixTokens;
    if (c.wrongMixRank) {
        mixShape.pop_back();
    }
    auto previous = TensorDesc(mixShape, c.mixType, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto x2 = TensorDesc(shape({4, c.d}), c.type, ACL_FORMAT_ND);
    auto postOut = TensorDesc(shape({4, 1}), ACL_FLOAT, ACL_FORMAT_ND);
    auto combOut = TensorDesc(shape({4, 4}), ACL_FLOAT, ACL_FORMAT_ND);
    auto preOut = TensorDesc(shape({4, c.preOutputLast}), c.preOutputType, ACL_FORMAT_ND);
    auto y = TensorDesc(shape({c.d + (c.wrongOutputShape ? 16 : 0)}), c.type, ACL_FORMAT_ND);
    auto run = [&](auto g0, auto g2, auto prevIn, auto prevOut) {
        auto ut = OP_API_UT(
            aclnnMegaMhc,
            INPUT(x, residual, post, fn, scales, bases, gamma, comb, prevIn, g0, g2, c.epsilon, c.preNormEpsilon,
                  c.gateEpsilon, c.mode, c.heads, 2.0, c.sinkhornEpsilon, c.iterations, c.rmsScale, c.outputMode),
            OUTPUT(x2, postOut, combOut, prevOut, y, nullptr, nullptr));
        uint64_t size = 0;
        return ut.TestGetWorkspaceSize(&size);
    };
    auto runMix = [&](auto g0, auto g2) {
        if (c.externalMix) {
            if (c.missingPreOutput) {
                return run(g0, g2, previous, nullptr);
            }
            return run(g0, g2, previous, preOut);
        }
        return run(g0, g2, nullptr, nullptr);
    };
    if (c.missingGamma) {
        return c.mid ? runMix(nullptr, gammaMid) : runMix(nullptr, nullptr);
    }
    if (c.invalidGamma) {
        return runMix(invalidGamma, nullptr);
    }
    if (c.reservedOutput) {
        return run(gamma, nullptr, nullptr, previous);
    }
    return c.mid ? runMix(gamma, gammaMid) : runMix(gamma, nullptr);
}
} // namespace

TEST(MegaMhcOpApi, NoMidAndWithMidBothTypesAndModes)
{
    for (auto type : {ACL_FLOAT16, ACL_BF16}) {
        for (bool mid : {false, true}) {
            for (int64_t mode : {0, 1}) {
                Case c;
                c.type = type;
                c.mid = mid;
                c.mode = mode;
                EXPECT_EQ(Workspace(c), ACL_SUCCESS);
            }
        }
    }
}

TEST(MegaMhcOpApi, RejectNonpositiveAndNonfiniteEpsilons)
{
    for (double value :
         {0.0, -1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        Case c;
        c.epsilon = value;
        EXPECT_EQ(Workspace(c), ACLNN_ERR_PARAM_INVALID);
        c = Case{};
        c.preNormEpsilon = value;
        EXPECT_EQ(Workspace(c), ACLNN_ERR_PARAM_INVALID);
        c = Case{};
        c.gateEpsilon = value;
        EXPECT_EQ(Workspace(c), ACLNN_ERR_PARAM_INVALID);
    }
}

TEST(MegaMhcOpApi, RejectUnsupportedModesAndReservedAttributes)
{
    Case c;
    c.mode = 2;
    EXPECT_EQ(Workspace(c), ACLNN_ERR_PARAM_INVALID);
    c = Case{};
    c.heads = 8;
    EXPECT_EQ(Workspace(c), ACLNN_ERR_PARAM_INVALID);
    c = Case{};
    c.iterations = 1;
    EXPECT_EQ(Workspace(c), ACLNN_ERR_PARAM_INVALID);
    c = Case{};
    c.sinkhornEpsilon = 1e-5;
    EXPECT_EQ(Workspace(c), ACLNN_ERR_PARAM_INVALID);
    c = Case{};
    c.rmsScale = 0.5;
    EXPECT_EQ(Workspace(c), ACLNN_ERR_PARAM_INVALID);
}

TEST(MegaMhcOpApi, RejectReservedAndInvalidOutputModes)
{
    for (int64_t value : {int64_t{-1}, int64_t{1}, int64_t{2}, int64_t{3}, std::numeric_limits<int64_t>::min(),
                          std::numeric_limits<int64_t>::max()}) {
        Case c;
        c.outputMode = value;
        EXPECT_EQ(Workspace(c), ACLNN_ERR_PARAM_INVALID);
    }
}

TEST(MegaMhcOpApi, ReservedSinkhornEpsilonUsesFloatAttributePrecision)
{
    Case c;
    c.sinkhornEpsilon = static_cast<double>(1.0e-6f);
    EXPECT_EQ(Workspace(c), ACLNN_SUCCESS);
    for (double value : {static_cast<double>(std::nextafter(1.0e-6f, 0.0f)), std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
        c.sinkhornEpsilon = value;
        EXPECT_EQ(Workspace(c), ACLNN_ERR_PARAM_INVALID);
    }
}

TEST(MegaMhcOpApi, OptionalRms0WithBothTypesModesAndMidStates)
{
    for (auto type : {ACL_FLOAT16, ACL_BF16}) {
        for (bool mid : {false, true}) {
            for (int64_t mode : {0, 1}) {
                Case c;
                c.type = type;
                c.mid = mid;
                c.mode = mode;
                c.missingGamma = true;
                EXPECT_EQ(Workspace(c), ACLNN_SUCCESS);
            }
        }
    }
}

TEST(MegaMhcOpApi, RejectInvalidWeightAndReservedOutputs)
{
    Case c;
    c.invalidGamma = true;
    EXPECT_EQ(Workspace(c), ACLNN_ERR_PARAM_INVALID);

    c = Case{};
    c.reservedOutput = true;
    EXPECT_EQ(Workspace(c), ACLNN_ERR_PARAM_INVALID);
}

TEST(MegaMhcOpApi, RejectInvalidDimensionsTypesAndOutputs)
{
    for (int64_t d : {15, 16385}) {
        Case c;
        c.d = d;
        EXPECT_EQ(Workspace(c), ACLNN_ERR_PARAM_INVALID);
    }
    Case c;
    c.type = ACL_FLOAT;
    EXPECT_EQ(Workspace(c), ACLNN_ERR_PARAM_INVALID);
    c = Case{};
    c.wrongOutputShape = true;
    EXPECT_EQ(Workspace(c), ACLNN_ERR_PARAM_INVALID);
}

TEST(MegaMhcOpApi, ExternalMixBothRanksTypesModesAndRmsStates)
{
    for (auto type : {ACL_FLOAT16, ACL_BF16}) {
        for (bool sequence : {false, true}) {
            for (bool missingGamma : {false, true}) {
                for (bool mid : {false, true}) {
                    for (int64_t mode : {0, 1}) {
                        Case c;
                        c.externalMix = true;
                        c.type = type;
                        c.sequence = sequence;
                        c.missingGamma = missingGamma;
                        c.mid = mid;
                        c.mode = mode;
                        EXPECT_EQ(Workspace(c), ACLNN_SUCCESS);
                    }
                }
            }
        }
    }
}

TEST(MegaMhcOpApi, RejectInvalidExternalMixShapeAndType)
{
    for (bool sequence : {false, true}) {
        Case c;
        c.externalMix = true;
        c.sequence = sequence;
        c.mixLast = 2;
        EXPECT_EQ(Workspace(c), ACLNN_ERR_PARAM_INVALID);
        c.mixLast = 1;
        c.mixTokens = 3;
        EXPECT_EQ(Workspace(c), ACLNN_ERR_PARAM_INVALID);
        c.mixTokens = 2;
        c.wrongMixRank = true;
        EXPECT_EQ(Workspace(c), ACLNN_ERR_PARAM_INVALID);
        c.wrongMixRank = false;
        c.mixType = ACL_FLOAT16;
        EXPECT_EQ(Workspace(c), ACLNN_ERR_PARAM_INVALID);
    }
}

TEST(MegaMhcOpApi, ExternalMixRequiresMatchingCurrentPreOutput)
{
    for (bool sequence : {false, true}) {
        Case c;
        c.externalMix = true;
        c.sequence = sequence;
        c.missingPreOutput = true;
        EXPECT_EQ(Workspace(c), ACLNN_ERR_PARAM_NULLPTR);
        c.missingPreOutput = false;
        c.preOutputLast = 2;
        EXPECT_EQ(Workspace(c), ACLNN_ERR_PARAM_INVALID);
        c.preOutputLast = 1;
        c.preOutputType = ACL_FLOAT16;
        EXPECT_EQ(Workspace(c), ACLNN_ERR_PARAM_INVALID);
        c.preOutputType = ACL_FLOAT;
        EXPECT_EQ(Workspace(c), ACL_SUCCESS);
    }
}
