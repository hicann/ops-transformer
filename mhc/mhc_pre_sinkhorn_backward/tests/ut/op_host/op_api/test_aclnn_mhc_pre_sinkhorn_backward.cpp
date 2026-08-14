/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <array>
#include <gtest/gtest.h>
#include "../../../../op_host/op_api/aclnn_mhc_pre_sinkhorn_backward.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class MhcPreSinkhornBackwardOpapiUt : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "MhcPreSinkhornBackwardOpapiUt SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "MhcPreSinkhornBackwardOpapiUt TearDown" << endl;
    }
};

TEST_F(MhcPreSinkhornBackwardOpapiUt, aclnn_mhc_pre_sinkhorn_backward_basic_4d_bf16)
{
    int64_t B = 2;
    int64_t S = 128;
    int64_t N = 4;
    int64_t C = 256;
    int64_t skIterCount = 20;
    int64_t hcMix = N * N + 2 * N;
    float hcEps = 1e-6f;

    auto gradHin = TensorDesc({B, S, C}, ACL_BF16, ACL_FORMAT_ND);
    auto gradHPost = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradHRes = TensorDesc({B, S, N, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x = TensorDesc({B, S, N, C}, ACL_BF16, ACL_FORMAT_ND);
    auto phi = TensorDesc({hcMix, N * C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto bias = TensorDesc({hcMix}, ACL_FLOAT, ACL_FORMAT_ND);
    auto hPre = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto hcBeforeNorm = TensorDesc({B, S, hcMix}, ACL_FLOAT, ACL_FORMAT_ND);
    auto invRms = TensorDesc({B, S, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumOut = TensorDesc({skIterCount * 2, B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto normOut = TensorDesc({skIterCount * 2, B, S, N, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto eps = 1e-6;

    auto gradX = TensorDesc({B, S, N, C}, ACL_BF16, ACL_FORMAT_ND);
    auto gradPhi = TensorDesc({hcMix, N * C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBias = TensorDesc({hcMix}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMhcPreSinkhornBackward,
        INPUT(gradHin, gradHPost, gradHRes, x, phi, alpha, bias, hPre, hcBeforeNorm, invRms, sumOut, normOut, eps),
        OUTPUT(gradX, gradPhi, gradAlpha, gradBias));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(MhcPreSinkhornBackwardOpapiUt, aclnn_mhc_pre_sinkhorn_backward_basic_3d_bf16)
{
    int64_t T = 256;
    int64_t N = 4;
    int64_t C = 256;
    int64_t skIterCount = 20;
    int64_t hcMix = N * N + 2 * N;
    float hcEps = 1e-6f;

    auto gradHin = TensorDesc({T, C}, ACL_BF16, ACL_FORMAT_ND);
    auto gradHPost = TensorDesc({T, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradHRes = TensorDesc({T, N, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x = TensorDesc({T, N, C}, ACL_BF16, ACL_FORMAT_ND);
    auto phi = TensorDesc({hcMix, N * C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto bias = TensorDesc({hcMix}, ACL_FLOAT, ACL_FORMAT_ND);
    auto hPre = TensorDesc({T, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto hcBeforeNorm = TensorDesc({T, hcMix}, ACL_FLOAT, ACL_FORMAT_ND);
    auto invRms = TensorDesc({T, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumOut = TensorDesc({skIterCount * 2, T, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto normOut = TensorDesc({skIterCount * 2, T, N, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto eps = 1e-6;

    auto gradX = TensorDesc({T, N, C}, ACL_BF16, ACL_FORMAT_ND);
    auto gradPhi = TensorDesc({hcMix, N * C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBias = TensorDesc({hcMix}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMhcPreSinkhornBackward,
        INPUT(gradHin, gradHPost, gradHRes, x, phi, alpha, bias, hPre, hcBeforeNorm, invRms, sumOut, normOut, eps),
        OUTPUT(gradX, gradPhi, gradAlpha, gradBias));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(MhcPreSinkhornBackwardOpapiUt, aclnn_mhc_pre_sinkhorn_backward_empty_tensor_4d_B0)
{
    int64_t B = 0;
    int64_t S = 128;
    int64_t N = 4;
    int64_t C = 256;
    int64_t skIterCount = 20;
    int64_t hcMix = N * N + 2 * N;
    float hcEps = 1e-6f;

    auto gradHin = TensorDesc({B, S, C}, ACL_BF16, ACL_FORMAT_ND);
    auto gradHPost = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradHRes = TensorDesc({B, S, N, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x = TensorDesc({B, S, N, C}, ACL_BF16, ACL_FORMAT_ND);
    auto phi = TensorDesc({hcMix, N * C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto bias = TensorDesc({hcMix}, ACL_FLOAT, ACL_FORMAT_ND);
    auto hPre = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto hcBeforeNorm = TensorDesc({B, S, hcMix}, ACL_FLOAT, ACL_FORMAT_ND);
    auto invRms = TensorDesc({B, S, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumOut = TensorDesc({skIterCount * 2, B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto normOut = TensorDesc({skIterCount * 2, B, S, N, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto eps = 1e-6;

    auto gradX = TensorDesc({B, S, N, C}, ACL_BF16, ACL_FORMAT_ND);
    auto gradPhi = TensorDesc({hcMix, N * C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBias = TensorDesc({hcMix}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMhcPreSinkhornBackward,
        INPUT(gradHin, gradHPost, gradHRes, x, phi, alpha, bias, hPre, hcBeforeNorm, invRms, sumOut, normOut, eps),
        OUTPUT(gradX, gradPhi, gradAlpha, gradBias));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    EXPECT_EQ(workspaceSize, 0);
}

TEST_F(MhcPreSinkhornBackwardOpapiUt, aclnn_mhc_pre_sinkhorn_backward_empty_tensor_4d_S0)
{
    int64_t B = 2;
    int64_t S = 0;
    int64_t N = 4;
    int64_t C = 256;
    int64_t skIterCount = 20;
    int64_t hcMix = N * N + 2 * N;
    float hcEps = 1e-6f;

    auto gradHin = TensorDesc({B, S, C}, ACL_BF16, ACL_FORMAT_ND);
    auto gradHPost = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradHRes = TensorDesc({B, S, N, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x = TensorDesc({B, S, N, C}, ACL_BF16, ACL_FORMAT_ND);
    auto phi = TensorDesc({hcMix, N * C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto bias = TensorDesc({hcMix}, ACL_FLOAT, ACL_FORMAT_ND);
    auto hPre = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto hcBeforeNorm = TensorDesc({B, S, hcMix}, ACL_FLOAT, ACL_FORMAT_ND);
    auto invRms = TensorDesc({B, S, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumOut = TensorDesc({skIterCount * 2, B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto normOut = TensorDesc({skIterCount * 2, B, S, N, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto eps = 1e-6;

    auto gradX = TensorDesc({B, S, N, C}, ACL_BF16, ACL_FORMAT_ND);
    auto gradPhi = TensorDesc({hcMix, N * C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBias = TensorDesc({hcMix}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMhcPreSinkhornBackward,
        INPUT(gradHin, gradHPost, gradHRes, x, phi, alpha, bias, hPre, hcBeforeNorm, invRms, sumOut, normOut, eps),
        OUTPUT(gradX, gradPhi, gradAlpha, gradBias));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    EXPECT_EQ(workspaceSize, 0);
}

TEST_F(MhcPreSinkhornBackwardOpapiUt, aclnn_mhc_pre_sinkhorn_backward_empty_tensor_3d_T0)
{
    int64_t T = 0;
    int64_t N = 4;
    int64_t C = 256;
    int64_t skIterCount = 20;
    int64_t hcMix = N * N + 2 * N;
    float hcEps = 1e-6f;

    auto gradHin = TensorDesc({T, C}, ACL_BF16, ACL_FORMAT_ND);
    auto gradHPost = TensorDesc({T, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradHRes = TensorDesc({T, N, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x = TensorDesc({T, N, C}, ACL_BF16, ACL_FORMAT_ND);
    auto phi = TensorDesc({hcMix, N * C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto bias = TensorDesc({hcMix}, ACL_FLOAT, ACL_FORMAT_ND);
    auto hPre = TensorDesc({T, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto hcBeforeNorm = TensorDesc({T, hcMix}, ACL_FLOAT, ACL_FORMAT_ND);
    auto invRms = TensorDesc({T, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumOut = TensorDesc({skIterCount * 2, T, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto normOut = TensorDesc({skIterCount * 2, T, N, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto eps = 1e-6;

    auto gradX = TensorDesc({T, N, C}, ACL_BF16, ACL_FORMAT_ND);
    auto gradPhi = TensorDesc({hcMix, N * C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBias = TensorDesc({hcMix}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMhcPreSinkhornBackward,
        INPUT(gradHin, gradHPost, gradHRes, x, phi, alpha, bias, hPre, hcBeforeNorm, invRms, sumOut, normOut, eps),
        OUTPUT(gradX, gradPhi, gradAlpha, gradBias));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    EXPECT_EQ(workspaceSize, 0);
}
