/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <vector>
#include "../../../op_host/op_api/aclnn_mhc_pre_backward_v2.h"
#include "opdev/platform.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

namespace {

class MhcPreBackwardA5CoverageApiTest : public testing::Test {
protected:
    void SetUp() override
    {
        originalSocVersion_ = op::GetCurrentPlatformInfo().GetSocVersion();
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    }

    void TearDown() override
    {
        op::SetPlatformSocVersion(originalSocVersion_);
    }

    op::SocVersion originalSocVersion_;
};

aclnnStatus RunMhcPreBackwardTnd(int64_t innerPrecise, int field = -1, int defect = 0, bool bsnd = false,
                                 int dimension = 0)
{
    constexpr int64_t t = 10;
    constexpr int64_t n = 4;
    constexpr int64_t d = 64;
    constexpr int64_t fusionSize = n * n + 2 * n;
    constexpr int64_t nD = n * d;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({fusionSize, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, fusionSize}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({fusionSize, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({fusionSize}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto post = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND);
    std::vector<TensorDesc*> tensors{&x,     &phi,   &alpha, &gradHIn, &gradHPost, &gradHRes, &invRms,    &hMix, &hPre,
                                     &hPost, &gamma, &gradX, &gradPhi, &gradAlpha, &gradBias, &gradGamma, &post};
    std::vector<std::vector<int64_t>> shapes{
        {t, n, d}, {fusionSize, nD}, {3},       {t, d},           {t, n}, {t, n, n},    {t},    {t, fusionSize}, {t, n},
        {t, n},    {n, d},           {t, n, d}, {fusionSize, nD}, {3},    {fusionSize}, {n, d}, {t, n, d}};
    if (bsnd) {
        for (int i : {0, 3, 4, 5, 6, 7, 8, 9, 11, 16})
            shapes[i].insert(shapes[i].begin(), 1);
        for (size_t i = 0; i < tensors.size(); ++i)
            tensors[i]->ViewDims(shapes[i]);
    }
    // 每次只破坏一个参数，覆盖实际接口校验路径。
    if (field >= 0) {
        if (defect == 0)
            *tensors[field] = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
        if (defect == 1)
            *tensors[field] = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
        if (defect == 2)
            *tensors[field] = TensorDesc({1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
        if (defect == 3) {
            auto shape = shapes[field];
            shape[dimension] += 1;
            tensors[field]->ViewDims(shape);
        }
        if (defect == 4)
            *tensors[field] = TensorDesc(shapes[field], ACL_INT32, ACL_FORMAT_ND);
        if (defect == 5)
            tensors[field]->Format(ACL_FORMAT_FRACTAL_NZ);
    }
    auto ut = OP_API_UT(
        aclnnMhcPreBackwardV2,
        INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, post, 1e-6f, innerPrecise),
        OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    return ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
}

TEST_F(MhcPreBackwardA5CoverageApiTest, TndFp32Mode)
{
    EXPECT_EQ(RunMhcPreBackwardTnd(0), ACLNN_SUCCESS);
}

TEST_F(MhcPreBackwardA5CoverageApiTest, TndHf32Mode)
{
    EXPECT_EQ(RunMhcPreBackwardTnd(1), ACLNN_SUCCESS);
}

TEST_F(MhcPreBackwardA5CoverageApiTest, RejectInvalidInnerPrecise)
{
    EXPECT_NE(RunMhcPreBackwardTnd(-1), ACLNN_SUCCESS);
    EXPECT_NE(RunMhcPreBackwardTnd(2), ACLNN_SUCCESS);
}

TEST_F(MhcPreBackwardA5CoverageApiTest, RejectFactorialResidualLayout)
{
    constexpr int64_t t = 10;
    constexpr int64_t n = 4;
    constexpr int64_t d = 64;
    constexpr int64_t fusionSize = n * n + 2 * n;
    constexpr int64_t nD = n * d;
    constexpr int64_t factorialSize = 24;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND);
    auto phi = TensorDesc({fusionSize, nD}, ACL_FLOAT, ACL_FORMAT_ND);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradHRes = TensorDesc({t, factorialSize}, ACL_FLOAT, ACL_FORMAT_ND);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND);
    auto hMix = TensorDesc({t, fusionSize}, ACL_FLOAT, ACL_FORMAT_ND);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND);
    auto gradPhi = TensorDesc({fusionSize, nD}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBias = TensorDesc({fusionSize}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMhcPreBackwardV2,
        INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, 1e-6f, 0),
        OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_NE(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_SUCCESS);
}

TEST_F(MhcPreBackwardA5CoverageApiTest, RejectEmptyAndInvalidRank)
{
    for (int field = 0; field < 16; ++field) {
        for (int defect = 0; defect < 3; ++defect) {
            SCOPED_TRACE(::testing::Message() << field << "/" << defect);
            EXPECT_NE(RunMhcPreBackwardTnd(0, field, defect), ACLNN_SUCCESS);
        }
    }
}

TEST_F(MhcPreBackwardA5CoverageApiTest, BsndAndParameterValidation)
{
    for (bool bsnd : {false, true}) {
        EXPECT_EQ(RunMhcPreBackwardTnd(0, -1, 0, bsnd), ACLNN_SUCCESS);
        EXPECT_EQ(RunMhcPreBackwardTnd(1, -1, 0, bsnd), ACLNN_SUCCESS);
        for (int field = 0; field < 17; ++field) {
            SCOPED_TRACE(::testing::Message() << bsnd << "/" << field);
            EXPECT_NE(RunMhcPreBackwardTnd(0, field, 4, bsnd), ACLNN_SUCCESS);
            if (field < 11 || field == 16)
                EXPECT_NE(RunMhcPreBackwardTnd(0, field, 5, bsnd), ACLNN_SUCCESS);
            for (int dim = 0; dim < 2; ++dim) {
                if ((field == 2 || field == 13 || field == 14 || (!bsnd && field == 6)) && dim == 1)
                    continue;
                EXPECT_NE(RunMhcPreBackwardTnd(0, field, 3, bsnd, dim), ACLNN_SUCCESS);
            }
        }
    }
}

} // namespace
