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
#include <float.h>
#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_mhc_pre_backward.h"
#include "opdev/platform.h"
#include "op_api_ut_common/array_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_ai_mhc_pre_backward_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_ai_mhc_pre_backward_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_ai_mhc_pre_backward_test TearDown" << endl;
    }
};

// ==================== 正常测试用例（成功路径） ====================

// TND格式 BF16测试
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_tnd_bf16_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// BSND格式 FP16测试
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_bsnd_fp16_0)
{
    int64_t b = 2;
    int64_t s = 2;
    int64_t n = 4;
    int64_t d = 8;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({b, s, n, d}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({b, s, d}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({b, s, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({b, s, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({b, s}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({b, s, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({b, s, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({b, s, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({b, s, n, d}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// TND格式 无gamma测试
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_tnd_no_gamma_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, nullptr, nullptr, hcEps),
        OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// BSND格式 无gamma测试
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_bsnd_no_gamma_0)
{
    int64_t b = 2;
    int64_t s = 2;
    int64_t n = 4;
    int64_t d = 8;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({b, s, n, d}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({b, s, d}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({b, s, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({b, s, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({b, s}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({b, s, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({b, s, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({b, s, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({b, s, n, d}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, nullptr, nullptr, hcEps),
        OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// ==================== 空指针检查用例（输入参数10个） ====================

// 异常用例: x为nullptr
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_x_nullptr_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(aclnnMhcPreBackward,
                        INPUT((const aclTensor *)nullptr, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre,
                              hPost, gamma, nullptr, hcEps),
                        OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: phi为nullptr
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_phi_nullptr_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(aclnnMhcPreBackward,
                        INPUT(x, (const aclTensor *)nullptr, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre,
                              hPost, gamma, nullptr, hcEps),
                        OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: alpha为nullptr
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_alpha_nullptr_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(aclnnMhcPreBackward,
                        INPUT(x, phi, (const aclTensor *)nullptr, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre,
                              hPost, gamma, nullptr, hcEps),
                        OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: gradHIn为nullptr
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_h_in_grad_nullptr_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(aclnnMhcPreBackward,
                        INPUT(x, phi, alpha, (const aclTensor *)nullptr, gradHPost, gradHRes, invRms, hMix, hPre, hPost,
                              gamma, nullptr, hcEps),
                        OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: gradHPost为nullptr
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_h_post_grad_nullptr_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(aclnnMhcPreBackward,
                        INPUT(x, phi, alpha, gradHIn, (const aclTensor *)nullptr, gradHRes, invRms, hMix, hPre, hPost,
                              gamma, nullptr, hcEps),
                        OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: gradHRes为nullptr
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_h_res_grad_nullptr_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(aclnnMhcPreBackward,
                        INPUT(x, phi, alpha, gradHIn, gradHPost, (const aclTensor *)nullptr, invRms, hMix, hPre, hPost,
                              gamma, nullptr, hcEps),
                        OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: invRms为nullptr
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_inv_rms_nullptr_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(aclnnMhcPreBackward,
                        INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, (const aclTensor *)nullptr, hMix, hPre,
                              hPost, gamma, nullptr, hcEps),
                        OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: hMix为nullptr
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_h_mix_nullptr_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(aclnnMhcPreBackward,
                        INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, (const aclTensor *)nullptr, hPre,
                              hPost, gamma, nullptr, hcEps),
                        OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: hPre为nullptr
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_h_pre_nullptr_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(aclnnMhcPreBackward,
                        INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, (const aclTensor *)nullptr,
                              hPost, gamma, nullptr, hcEps),
                        OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: hPost为nullptr
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_h_post_nullptr_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(aclnnMhcPreBackward,
                        INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre,
                              (const aclTensor *)nullptr, gamma, nullptr, hcEps),
                        OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// ==================== 空指针检查用例（输出参数5个） ====================

// 异常用例: gradX为nullptr
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_grad_x_nullptr_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT((const aclTensor *)nullptr, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: gradPhi为nullptr
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_grad_phi_nullptr_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, (const aclTensor *)nullptr, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: gradAlpha为nullptr
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_grad_alpha_nullptr_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, gradPhi, (const aclTensor *)nullptr, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: gradBias为nullptr
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_grad_bias_nullptr_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, gradPhi, gradAlpha, (const aclTensor *)nullptr, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: gradGamma为nullptr
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_grad_gamma_nullptr_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, gradPhi, gradAlpha, gradBias, (const aclTensor *)nullptr));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// ==================== 空tensor检查用例（输入参数10个） ====================

// 异常用例: x为空tensor
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_x_empty_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({0}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: phi为空tensor
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_phi_empty_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: alpha为空tensor
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_alpha_empty_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: gradHIn为空tensor
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_h_in_grad_empty_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({0}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: gradHPost为空tensor
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_h_post_grad_empty_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: gradHRes为空tensor
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_h_res_grad_empty_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: invRms为空tensor
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_inv_rms_empty_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: hMix为空tensor
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_h_mix_empty_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: hPre为空tensor
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_h_pre_empty_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: hPost为空tensor
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_h_post_empty_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// ==================== 维度检查用例 ====================

// 异常用例: x维度错误
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_x_dim_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: gradX维度错误-TND格式下
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_grad_x_dim_invalid_tnd_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: gradX维度错误-BSND格式下
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_grad_x_dim_invalid_bsnd_0)
{
    int64_t b = 2;
    int64_t s = 2;
    int64_t n = 4;
    int64_t d = 8;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({b, s, n, d}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({b, s, d}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({b, s, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({b, s, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({b, s}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({b, s, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({b, s, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({b, s, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({b, s, d}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// ==================== Dtype检查用例 ====================

// 异常用例: x dtype错误（FP32不支持）
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_x_dtype_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: gradX dtype不匹配
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_grad_x_dtype_mismatch_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// ==================== Format检查用例 ====================

// 异常用例: x格式错误（使用FRACTAL_NZ格式）
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_x_format_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// ==================== Shape检查用例 ====================

// 异常用例: alpha形状错误
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_alpha_shape_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: gradPhi形状错误
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_grad_phi_shape_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n + 1, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// ==================== Dtype检查用例（补充） ====================

// 异常用例: phi dtype错误（BF16不支持）
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_phi_dtype_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: gradHIn dtype错误（FP32不支持）
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_h_in_grad_dtype_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: gradPhi dtype错误（BF16不支持）
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_grad_phi_dtype_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: gradAlpha dtype错误（BF16不支持）
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_grad_alpha_dtype_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: gradBias dtype错误（BF16不支持）
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_grad_bias_dtype_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: gradGamma dtype错误（BF16不支持）
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_grad_gamma_dtype_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// ==================== 维度检查用例（补充） ====================

// 异常用例: phi维度错误
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_phi_dim_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({4 * 4 + 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, 4 * 4 + 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({4 * 4 + 8, n * d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({4 * 4 + 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: gamma维度错误
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_gamma_dim_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// ==================== invRms维度检查用例 ====================

// 异常用例: invRms维度错误-BSND格式下（invRms应为2D [B,S]，传入3D [B,S,1]）
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_inv_rms_dim_invalid_bsnd_0)
{
    int64_t b = 2;
    int64_t s = 2;
    int64_t n = 4;
    int64_t d = 8;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({b, s, n, d}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({b, s, d}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({b, s, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({b, s, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({b, s, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({b, s, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({b, s, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({b, s, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({b, s, n, d}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: invRms维度错误-TND格式下（invRms应为1D [T]，传入2D [T,1]）
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_inv_rms_dim_invalid_tnd_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// ==================== 可选参数gradXPost测试用例 ====================

// TND格式 BF16测试 - 有gradXPost
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_tnd_bf16_with_gradxpost_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto grad_x_post = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, grad_x_post, hcEps),
        OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// BSND格式 FP16测试 - 有gradXPost
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_bsnd_fp16_with_gradxpost_0)
{
    int64_t b = 2;
    int64_t s = 2;
    int64_t n = 4;
    int64_t d = 8;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({b, s, n, d}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({b, s, d}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({b, s, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({b, s, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({b, s}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({b, s, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({b, s, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({b, s, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto grad_x_post = TensorDesc({b, s, n, d}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({b, s, n, d}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, grad_x_post, hcEps),
        OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// TND格式 无gamma - 有gradXPost
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_tnd_no_gamma_with_gradxpost_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto grad_x_post = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, nullptr, grad_x_post, hcEps),
        OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// BSND格式 无gamma - 有gradXPost
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_bsnd_no_gamma_with_gradxpost_0)
{
    int64_t b = 2;
    int64_t s = 2;
    int64_t n = 4;
    int64_t d = 8;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({b, s, n, d}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({b, s, d}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({b, s, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({b, s, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({b, s}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({b, s, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({b, s, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({b, s, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto grad_x_post = TensorDesc({b, s, n, d}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({b, s, n, d}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut = OP_API_UT(
        aclnnMhcPreBackward,
        INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, nullptr, grad_x_post, hcEps),
        OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: gradHPost维度错误
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_h_post_grad_dim_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: gradHRes维度错误
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_h_res_grad_dim_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: hMix维度错误
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_h_mix_dim_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: hPre维度错误
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_h_pre_dim_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: hPost维度错误
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_h_post_dim_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: gradPhi维度错误
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_grad_phi_dim_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: gradAlpha维度错误
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_grad_alpha_dim_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: gradBias维度错误
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_grad_bias_dim_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常用例: gradGamma维度错误
TEST_F(l2_ai_mhc_pre_backward_test, mhc_pre_backward_grad_gamma_dim_invalid_0)
{
    int64_t n = 4;
    int64_t d = 8;
    int64_t t = 10;
    int64_t nD = n * d;
    int64_t n2_plus_2n = n * n + 2 * n;

    auto x = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto phi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 1.5);
    auto gradHIn = TensorDesc({t, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradHRes = TensorDesc({t, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invRms = TensorDesc({t}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto hMix = TensorDesc({t, n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPre = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hPost = TensorDesc({t, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gamma = TensorDesc({n, d}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    double hcEps = 1e-6;

    auto gradX = TensorDesc({t, n, d}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradPhi = TensorDesc({n2_plus_2n, nD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradBias = TensorDesc({n2_plus_2n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);
    auto gradGamma = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 0);

    auto ut =
        OP_API_UT(aclnnMhcPreBackward,
                  INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, nullptr, hcEps),
                  OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}
