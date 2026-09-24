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
#include <gmock/gmock.h>
#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_moe_token_permute_grad.h"
#include "opdev/platform.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_moe_token_permute_grad_test : public testing::Test {
protected:
    void SetUp() override
    {
        cout << "l2_moe_token_permute_grad_test SetUp" << endl;
        originalSocVersion_ = op::GetCurrentPlatformInfo().GetSocVersion();
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    }

    void TearDown() override
    {
        cout << "l2_moe_token_permute_grad_test TearDown" << endl;
        op::SetPlatformSocVersion(originalSocVersion_);
    }

private:
    op::SocVersion originalSocVersion_ = op::SocVersion::ASCEND910B;
};

TEST_F(l2_moe_token_permute_grad_test, Ascend950_moe_token_permute_grad_fp16)
{
    auto permutedOutputGrad = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t numTopk = 3;
    bool paddedMode = false;

    auto ut =
        OP_API_UT(aclnnMoeTokenPermuteGrad, INPUT(permutedOutputGrad, sortedIndices, numTopk, paddedMode), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_moe_token_permute_grad_test, Ascend950_moe_token_permute_grad_fp32)
{
    auto permutedOutputGrad = TensorDesc({6, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t numTopk = 3;
    bool paddedMode = false;

    auto ut =
        OP_API_UT(aclnnMoeTokenPermuteGrad, INPUT(permutedOutputGrad, sortedIndices, numTopk, paddedMode), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_moe_token_permute_grad_test, Ascend950_moe_token_permute_grad_bf16)
{
    auto permutedOutputGrad = TensorDesc({6, 5}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({2, 5}, ACL_BF16, ACL_FORMAT_ND);
    int64_t numTopk = 3;
    bool paddedMode = false;

    auto ut =
        OP_API_UT(aclnnMoeTokenPermuteGrad, INPUT(permutedOutputGrad, sortedIndices, numTopk, paddedMode), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_moe_token_permute_grad_test, Ascend950_moe_token_permute_grad_permuted_output_grad_nullptr)
{
    auto sortedIndices = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t numTopk = 3;
    bool paddedMode = false;

    auto ut = OP_API_UT(aclnnMoeTokenPermuteGrad, INPUT(nullptr, sortedIndices, numTopk, paddedMode), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_moe_token_permute_grad_test, Ascend950_moe_token_permute_grad_sorted_indices_nullptr)
{
    auto permutedOutputGrad = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto out = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t numTopk = 3;
    bool paddedMode = false;

    auto ut = OP_API_UT(aclnnMoeTokenPermuteGrad, INPUT(permutedOutputGrad, nullptr, numTopk, paddedMode), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_moe_token_permute_grad_test, Ascend950_moe_token_permute_grad_out_nullptr)
{
    auto permutedOutputGrad = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    int64_t numTopk = 3;
    bool paddedMode = false;

    auto ut = OP_API_UT(aclnnMoeTokenPermuteGrad, INPUT(permutedOutputGrad, sortedIndices, numTopk, paddedMode),
                        OUTPUT(nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_moe_token_permute_grad_test, Ascend950_moe_token_permute_grad_padded_mode_invalid)
{
    auto permutedOutputGrad = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t numTopk = 3;
    bool paddedMode = true;

    auto ut =
        OP_API_UT(aclnnMoeTokenPermuteGrad, INPUT(permutedOutputGrad, sortedIndices, numTopk, paddedMode), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_moe_token_permute_grad_test, Ascend950_moe_token_permute_grad_permuted_output_grad_dtype_invalid)
{
    auto permutedOutputGrad = TensorDesc({6, 5}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({2, 5}, ACL_INT8, ACL_FORMAT_ND);
    int64_t numTopk = 3;
    bool paddedMode = false;

    auto ut =
        OP_API_UT(aclnnMoeTokenPermuteGrad, INPUT(permutedOutputGrad, sortedIndices, numTopk, paddedMode), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_moe_token_permute_grad_test, Ascend950_moe_token_permute_grad_sorted_indices_dtype_invalid)
{
    auto permutedOutputGrad = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices = TensorDesc({6}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t numTopk = 3;
    bool paddedMode = false;

    auto ut =
        OP_API_UT(aclnnMoeTokenPermuteGrad, INPUT(permutedOutputGrad, sortedIndices, numTopk, paddedMode), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_moe_token_permute_grad_test, Ascend950_moe_token_permute_grad_out_dtype_invalid)
{
    auto permutedOutputGrad = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({2, 5}, ACL_INT8, ACL_FORMAT_ND);
    int64_t numTopk = 3;
    bool paddedMode = false;

    auto ut =
        OP_API_UT(aclnnMoeTokenPermuteGrad, INPUT(permutedOutputGrad, sortedIndices, numTopk, paddedMode), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_moe_token_permute_grad_test, Ascend950_moe_token_permute_grad_dtype_diff)
{
    auto permutedOutputGrad = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t numTopk = 3;
    bool paddedMode = false;

    auto ut =
        OP_API_UT(aclnnMoeTokenPermuteGrad, INPUT(permutedOutputGrad, sortedIndices, numTopk, paddedMode), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_moe_token_permute_grad_test, Ascend950_moe_token_permute_grad_execute)
{
    auto permutedOutputGrad = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t numTopk = 3;
    bool paddedMode = false;

    auto ut =
        OP_API_UT(aclnnMoeTokenPermuteGrad, INPUT(permutedOutputGrad, sortedIndices, numTopk, paddedMode), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

    aclnnStatus execRet = aclnnMoeTokenPermuteGrad(nullptr, workspaceSize, executor, nullptr);
    EXPECT_THAT(execRet, testing::AnyOf(testing::Eq(ACLNN_SUCCESS), testing::Eq(ACLNN_ERR_PARAM_NULLPTR),
                                        testing::Eq(ACLNN_ERR_PARAM_INVALID)));
}
