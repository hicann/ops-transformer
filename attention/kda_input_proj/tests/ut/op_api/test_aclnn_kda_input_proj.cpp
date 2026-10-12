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

#include "gtest/gtest.h"
#include "../../../op_api/aclnn_kda_input_proj.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

namespace {
constexpr int64_t kT = 8;
constexpr int64_t kHidden = 256;
constexpr int64_t kQkv = 128;
constexpr int64_t kBeta = 16;
constexpr int64_t kGate = 64;
constexpr int64_t kG = 64;
constexpr int64_t kMxBlock = 64;

TensorDesc ContiguousWeight(int64_t kSize, int64_t nSize, aclDataType dtype)
{
    return TensorDesc({kSize, nSize}, dtype, ACL_FORMAT_ND).ValueRange(-1, 1);
}

// 存储 [N, K]，公开 view [K, N] 且 stride[-2]==1、stride[-1]==K → trans=true
TensorDesc TransposedWeight(int64_t kSize, int64_t nSize, aclDataType dtype)
{
    return TensorDesc({kSize, nSize}, dtype, ACL_FORMAT_ND, {1, kSize}, 0, {nSize, kSize}).ValueRange(-1, 1);
}

TensorDesc ScaleKn(int64_t kSize, int64_t nSize)
{
    const int64_t mxK = kSize / kMxBlock;
    return TensorDesc({mxK, nSize, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2);
}

struct CaseTensors {
    TensorDesc x;
    TensorDesc wQkv;
    TensorDesc wBeta;
    TensorDesc wGate;
    TensorDesc wG;
    TensorDesc scale;
    TensorDesc qkv;
    TensorDesc beta;
    TensorDesc gate;
    TensorDesc gOut;
};

CaseTensors MakeContiguousCase()
{
    CaseTensors c;
    c.x = TensorDesc({kT, kHidden}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    c.wQkv = ContiguousWeight(kHidden, kQkv, ACL_FLOAT8_E4M3FN);
    c.wBeta = ContiguousWeight(kHidden, kBeta, ACL_BF16);
    c.wGate = ContiguousWeight(kHidden, kGate, ACL_BF16);
    c.wG = ContiguousWeight(kHidden, kG, ACL_BF16);
    c.scale = ScaleKn(kHidden, kQkv);
    c.qkv = TensorDesc({kT, kQkv}, ACL_BF16, ACL_FORMAT_ND);
    c.beta = TensorDesc({kT, kBeta}, ACL_FLOAT, ACL_FORMAT_ND);
    c.gate = TensorDesc({kT, kGate}, ACL_BF16, ACL_FORMAT_ND);
    c.gOut = TensorDesc({kT, kG}, ACL_BF16, ACL_FORMAT_ND);
    return c;
}

CaseTensors MakeTransposedCase()
{
    CaseTensors c = MakeContiguousCase();
    c.wQkv = TransposedWeight(kHidden, kQkv, ACL_FLOAT8_E4M3FN);
    c.wBeta = TransposedWeight(kHidden, kBeta, ACL_BF16);
    c.wGate = TransposedWeight(kHidden, kGate, ACL_BF16);
    c.wG = TransposedWeight(kHidden, kG, ACL_BF16);
    return c;
}

aclnnStatus RunCase(const CaseTensors& c)
{
    auto ut = OP_API_UT(aclnnKdaInputProj, INPUT(c.x, c.wQkv, c.wBeta, c.wGate, c.wG, c.scale),
                        OUTPUT(c.qkv, c.beta, c.gate, c.gOut));
    uint64_t workspaceSize = 0;
    return ut.TestGetWorkspaceSize(&workspaceSize);
}

// CheckInputParams 通过后进入 nnopbase Inner。L2 失败固定走 ACLNN_ERR_PARAM_INVALID(161002)。
// Inner 在 UT 桩/流水线上可能返回 SUCCESS、PARAM_NULLPTR、RUNTIME_ERROR 等，不作为 L2 失败。
void ExpectL2Accepted(aclnnStatus ret)
{
    EXPECT_NE(ret, ACLNN_ERR_PARAM_INVALID) << "L2 rejected a valid case, ret=" << ret;
}
} // namespace

class aclnnKdaInputProj_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
        cout << "aclnnKdaInputProj_test SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "aclnnKdaInputProj_test TearDown" << endl;
    }
};

TEST_F(aclnnKdaInputProj_test, get_workspace_contiguous_kn)
{
    ExpectL2Accepted(RunCase(MakeContiguousCase()));
}

TEST_F(aclnnKdaInputProj_test, get_workspace_transposed_nk_storage)
{
    ExpectL2Accepted(RunCase(MakeTransposedCase()));
}

TEST_F(aclnnKdaInputProj_test, mixed_trans_qkv_only)
{
    auto c = MakeContiguousCase();
    c.wQkv = TransposedWeight(kHidden, kQkv, ACL_FLOAT8_E4M3FN);
    ExpectL2Accepted(RunCase(c));
}

TEST_F(aclnnKdaInputProj_test, null_x)
{
    auto c = MakeContiguousCase();
    auto ut = OP_API_UT(aclnnKdaInputProj, INPUT(nullptr, c.wQkv, c.wBeta, c.wGate, c.wG, c.scale),
                        OUTPUT(c.qkv, c.beta, c.gate, c.gOut));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(aclnnKdaInputProj_test, null_weight_qkv)
{
    auto c = MakeContiguousCase();
    auto ut = OP_API_UT(aclnnKdaInputProj, INPUT(c.x, nullptr, c.wBeta, c.wGate, c.wG, c.scale),
                        OUTPUT(c.qkv, c.beta, c.gate, c.gOut));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(aclnnKdaInputProj_test, null_weight_beta)
{
    auto c = MakeContiguousCase();
    auto ut = OP_API_UT(aclnnKdaInputProj, INPUT(c.x, c.wQkv, nullptr, c.wGate, c.wG, c.scale),
                        OUTPUT(c.qkv, c.beta, c.gate, c.gOut));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(aclnnKdaInputProj_test, null_weight_gate)
{
    auto c = MakeContiguousCase();
    auto ut = OP_API_UT(aclnnKdaInputProj, INPUT(c.x, c.wQkv, c.wBeta, nullptr, c.wG, c.scale),
                        OUTPUT(c.qkv, c.beta, c.gate, c.gOut));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(aclnnKdaInputProj_test, null_weight_g)
{
    auto c = MakeContiguousCase();
    auto ut = OP_API_UT(aclnnKdaInputProj, INPUT(c.x, c.wQkv, c.wBeta, c.wGate, nullptr, c.scale),
                        OUTPUT(c.qkv, c.beta, c.gate, c.gOut));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(aclnnKdaInputProj_test, null_scale)
{
    auto c = MakeContiguousCase();
    auto ut = OP_API_UT(aclnnKdaInputProj, INPUT(c.x, c.wQkv, c.wBeta, c.wGate, c.wG, nullptr),
                        OUTPUT(c.qkv, c.beta, c.gate, c.gOut));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(aclnnKdaInputProj_test, null_qkv_out)
{
    auto c = MakeContiguousCase();
    auto ut = OP_API_UT(aclnnKdaInputProj, INPUT(c.x, c.wQkv, c.wBeta, c.wGate, c.wG, c.scale),
                        OUTPUT(nullptr, c.beta, c.gate, c.gOut));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(aclnnKdaInputProj_test, null_beta_out)
{
    auto c = MakeContiguousCase();
    auto ut = OP_API_UT(aclnnKdaInputProj, INPUT(c.x, c.wQkv, c.wBeta, c.wGate, c.wG, c.scale),
                        OUTPUT(c.qkv, nullptr, c.gate, c.gOut));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(aclnnKdaInputProj_test, null_gate_out)
{
    auto c = MakeContiguousCase();
    auto ut = OP_API_UT(aclnnKdaInputProj, INPUT(c.x, c.wQkv, c.wBeta, c.wGate, c.wG, c.scale),
                        OUTPUT(c.qkv, c.beta, nullptr, c.gOut));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(aclnnKdaInputProj_test, null_g_out)
{
    auto c = MakeContiguousCase();
    auto ut = OP_API_UT(aclnnKdaInputProj, INPUT(c.x, c.wQkv, c.wBeta, c.wGate, c.wG, c.scale),
                        OUTPUT(c.qkv, c.beta, c.gate, nullptr));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(aclnnKdaInputProj_test, dtype_x_fp16)
{
    auto c = MakeContiguousCase();
    c.x = TensorDesc({kT, kHidden}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    EXPECT_EQ(RunCase(c), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(aclnnKdaInputProj_test, dtype_weight_qkv_bf16)
{
    auto c = MakeContiguousCase();
    c.wQkv = ContiguousWeight(kHidden, kQkv, ACL_BF16);
    EXPECT_EQ(RunCase(c), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(aclnnKdaInputProj_test, dtype_scale_fp32)
{
    auto c = MakeContiguousCase();
    c.scale = TensorDesc({kHidden / kMxBlock, kQkv, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    EXPECT_EQ(RunCase(c), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(aclnnKdaInputProj_test, dtype_beta_out_bf16)
{
    auto c = MakeContiguousCase();
    c.beta = TensorDesc({kT, kBeta}, ACL_BF16, ACL_FORMAT_ND);
    EXPECT_EQ(RunCase(c), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(aclnnKdaInputProj_test, format_fractal_nz)
{
    auto c = MakeContiguousCase();
    c.x = TensorDesc({kT, kHidden}, ACL_BF16, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    EXPECT_EQ(RunCase(c), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(aclnnKdaInputProj_test, format_ncl_nd_like_pass)
{
    auto c = MakeContiguousCase();
    c.x = TensorDesc({kT, kHidden}, ACL_BF16, ACL_FORMAT_NCL).ValueRange(-1, 1);
    ExpectL2Accepted(RunCase(c));
}

TEST_F(aclnnKdaInputProj_test, x_rank3)
{
    auto c = MakeContiguousCase();
    c.x = TensorDesc({1, kT, kHidden}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    EXPECT_EQ(RunCase(c), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(aclnnKdaInputProj_test, weight_k_mismatch)
{
    auto c = MakeContiguousCase();
    c.wQkv = ContiguousWeight(kHidden + 32, kQkv, ACL_FLOAT8_E4M3FN);
    EXPECT_EQ(RunCase(c), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(aclnnKdaInputProj_test, qkv_out_shape_mismatch)
{
    auto c = MakeContiguousCase();
    c.qkv = TensorDesc({kT, kQkv + 1}, ACL_BF16, ACL_FORMAT_ND);
    EXPECT_EQ(RunCase(c), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(aclnnKdaInputProj_test, beta_out_shape_mismatch)
{
    auto c = MakeContiguousCase();
    c.beta = TensorDesc({kT + 1, kBeta}, ACL_FLOAT, ACL_FORMAT_ND);
    EXPECT_EQ(RunCase(c), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(aclnnKdaInputProj_test, dtype_weight_beta_fp32)
{
    auto c = MakeContiguousCase();
    c.wBeta = ContiguousWeight(kHidden, kBeta, ACL_FLOAT);
    EXPECT_EQ(RunCase(c), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(aclnnKdaInputProj_test, dtype_weight_gate_fp16)
{
    auto c = MakeContiguousCase();
    c.wGate = ContiguousWeight(kHidden, kGate, ACL_FLOAT16);
    EXPECT_EQ(RunCase(c), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(aclnnKdaInputProj_test, dtype_weight_g_fp32)
{
    auto c = MakeContiguousCase();
    c.wG = ContiguousWeight(kHidden, kG, ACL_FLOAT);
    EXPECT_EQ(RunCase(c), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(aclnnKdaInputProj_test, dtype_qkv_out_fp32)
{
    auto c = MakeContiguousCase();
    c.qkv = TensorDesc({kT, kQkv}, ACL_FLOAT, ACL_FORMAT_ND);
    EXPECT_EQ(RunCase(c), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(aclnnKdaInputProj_test, dtype_gate_out_fp32)
{
    auto c = MakeContiguousCase();
    c.gate = TensorDesc({kT, kGate}, ACL_FLOAT, ACL_FORMAT_ND);
    EXPECT_EQ(RunCase(c), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(aclnnKdaInputProj_test, dtype_g_out_fp16)
{
    auto c = MakeContiguousCase();
    c.gOut = TensorDesc({kT, kG}, ACL_FLOAT16, ACL_FORMAT_ND);
    EXPECT_EQ(RunCase(c), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(aclnnKdaInputProj_test, format_nchw_nd_like_pass)
{
    auto c = MakeContiguousCase();
    c.x = TensorDesc({kT, kHidden}, ACL_BF16, ACL_FORMAT_NCHW).ValueRange(-1, 1);
    ExpectL2Accepted(RunCase(c));
}

TEST_F(aclnnKdaInputProj_test, mixed_trans_beta_gate_g)
{
    auto c = MakeContiguousCase();
    c.wBeta = TransposedWeight(kHidden, kBeta, ACL_BF16);
    c.wGate = TransposedWeight(kHidden, kGate, ACL_BF16);
    c.wG = TransposedWeight(kHidden, kG, ACL_BF16);
    ExpectL2Accepted(RunCase(c));
}

TEST_F(aclnnKdaInputProj_test, weight_beta_k_mismatch)
{
    auto c = MakeContiguousCase();
    c.wBeta = ContiguousWeight(kHidden + 32, kBeta, ACL_BF16);
    EXPECT_EQ(RunCase(c), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(aclnnKdaInputProj_test, gate_out_shape_mismatch)
{
    auto c = MakeContiguousCase();
    c.gate = TensorDesc({kT, kGate + 1}, ACL_BF16, ACL_FORMAT_ND);
    EXPECT_EQ(RunCase(c), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(aclnnKdaInputProj_test, g_out_shape_mismatch)
{
    auto c = MakeContiguousCase();
    c.gOut = TensorDesc({kT, kG + 1}, ACL_BF16, ACL_FORMAT_ND);
    EXPECT_EQ(RunCase(c), ACLNN_ERR_PARAM_INVALID);
}
