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
#include "../../../../op_host/op_api/aclnn_moe_token_unpermute_grad.h"
#include "opdev/platform.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_moe_token_unpermute_grad_test : public testing::Test {
protected:
    void SetUp() override
    {
        cout << "l2_moe_token_unpermute_grad_test SetUp" << endl;
        originalSocVersion_ = op::GetCurrentPlatformInfo().GetSocVersion();
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    }

    void TearDown() override
    {
        cout << "l2_moe_token_unpermute_grad_test TearDown" << endl;
        op::SetPlatformSocVersion(originalSocVersion_);
    }

private:
    op::SocVersion originalSocVersion_ = op::SocVersion::ASCEND910B;
};

TEST_F(l2_moe_token_unpermute_grad_test, Ascend950_moe_token_unpermute_grad_fp16)
{
    auto permuteTokens = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto unpermutedTokensGrad = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto permutedTokensGradOut = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto probsGradOut = TensorDesc({6}, ACL_FLOAT16, ACL_FORMAT_ND);
    bool paddedMode = false;
    std::vector<int64_t> restoreShape = {2, 5};
    auto restoreShapeOptional = aclCreateIntArray(restoreShape.data(), restoreShape.size());

    auto ut =
        OP_API_UT(aclnnMoeTokenUnpermuteGrad,
                  INPUT(permuteTokens, unpermutedTokensGrad, sortedIndices, nullptr, paddedMode, restoreShapeOptional),
                  OUTPUT(permutedTokensGradOut, probsGradOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_moe_token_unpermute_grad_test, Ascend950_moe_token_unpermute_grad_fp32)
{
    auto permuteTokens = TensorDesc({6, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto unpermutedTokensGrad = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto permutedTokensGradOut = TensorDesc({6, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto probsGradOut = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND);
    bool paddedMode = false;
    std::vector<int64_t> restoreShape = {2, 5};
    auto restoreShapeOptional = aclCreateIntArray(restoreShape.data(), restoreShape.size());

    auto ut =
        OP_API_UT(aclnnMoeTokenUnpermuteGrad,
                  INPUT(permuteTokens, unpermutedTokensGrad, sortedIndices, nullptr, paddedMode, restoreShapeOptional),
                  OUTPUT(permutedTokensGradOut, probsGradOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_moe_token_unpermute_grad_test, Ascend950_moe_token_unpermute_grad_bf16)
{
    auto permuteTokens = TensorDesc({6, 5}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto unpermutedTokensGrad = TensorDesc({2, 5}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto permutedTokensGradOut = TensorDesc({6, 5}, ACL_BF16, ACL_FORMAT_ND);
    auto probsGradOut = TensorDesc({6}, ACL_BF16, ACL_FORMAT_ND);
    bool paddedMode = false;
    std::vector<int64_t> restoreShape = {2, 5};
    auto restoreShapeOptional = aclCreateIntArray(restoreShape.data(), restoreShape.size());

    auto ut =
        OP_API_UT(aclnnMoeTokenUnpermuteGrad,
                  INPUT(permuteTokens, unpermutedTokensGrad, sortedIndices, nullptr, paddedMode, restoreShapeOptional),
                  OUTPUT(permutedTokensGradOut, probsGradOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_moe_token_unpermute_grad_test, Ascend950_moe_token_unpermute_grad_with_probs_fp16)
{
    auto permuteTokens = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto unpermutedTokensGrad = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto probsOptional = TensorDesc({6}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto permutedTokensGradOut = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto probsGradOut = TensorDesc({6}, ACL_FLOAT16, ACL_FORMAT_ND);
    bool paddedMode = false;
    std::vector<int64_t> restoreShape = {2, 5};
    auto restoreShapeOptional = aclCreateIntArray(restoreShape.data(), restoreShape.size());

    auto ut = OP_API_UT(
        aclnnMoeTokenUnpermuteGrad,
        INPUT(permuteTokens, unpermutedTokensGrad, sortedIndices, probsOptional, paddedMode, restoreShapeOptional),
        OUTPUT(permutedTokensGradOut, probsGradOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_moe_token_unpermute_grad_test, Ascend950_moe_token_unpermute_grad_permute_tokens_nullptr)
{
    auto unpermutedTokensGrad = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto permutedTokensGradOut = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto probsGradOut = TensorDesc({6}, ACL_FLOAT16, ACL_FORMAT_ND);
    bool paddedMode = false;
    std::vector<int64_t> restoreShape = {2, 5};
    auto restoreShapeOptional = aclCreateIntArray(restoreShape.data(), restoreShape.size());

    auto ut = OP_API_UT(aclnnMoeTokenUnpermuteGrad,
                        INPUT(nullptr, unpermutedTokensGrad, sortedIndices, nullptr, paddedMode, restoreShapeOptional),
                        OUTPUT(permutedTokensGradOut, probsGradOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_moe_token_unpermute_grad_test, Ascend950_moe_token_unpermute_grad_unpermuted_tokens_grad_nullptr)
{
    auto permuteTokens = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto permutedTokensGradOut = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto probsGradOut = TensorDesc({6}, ACL_FLOAT16, ACL_FORMAT_ND);
    bool paddedMode = false;
    std::vector<int64_t> restoreShape = {2, 5};
    auto restoreShapeOptional = aclCreateIntArray(restoreShape.data(), restoreShape.size());

    auto ut = OP_API_UT(aclnnMoeTokenUnpermuteGrad,
                        INPUT(permuteTokens, nullptr, sortedIndices, nullptr, paddedMode, restoreShapeOptional),
                        OUTPUT(permutedTokensGradOut, probsGradOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_moe_token_unpermute_grad_test, Ascend950_moe_token_unpermute_grad_sorted_indices_nullptr)
{
    auto permuteTokens = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto unpermutedTokensGrad = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto permutedTokensGradOut = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto probsGradOut = TensorDesc({6}, ACL_FLOAT16, ACL_FORMAT_ND);
    bool paddedMode = false;
    std::vector<int64_t> restoreShape = {2, 5};
    auto restoreShapeOptional = aclCreateIntArray(restoreShape.data(), restoreShape.size());

    auto ut = OP_API_UT(aclnnMoeTokenUnpermuteGrad,
                        INPUT(permuteTokens, unpermutedTokensGrad, nullptr, nullptr, paddedMode, restoreShapeOptional),
                        OUTPUT(permutedTokensGradOut, probsGradOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_moe_token_unpermute_grad_test, Ascend950_moe_token_unpermute_grad_permuted_tokens_grad_out_nullptr)
{
    auto permuteTokens = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto unpermutedTokensGrad = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto probsGradOut = TensorDesc({6}, ACL_FLOAT16, ACL_FORMAT_ND);
    bool paddedMode = false;
    std::vector<int64_t> restoreShape = {2, 5};
    auto restoreShapeOptional = aclCreateIntArray(restoreShape.data(), restoreShape.size());

    auto ut =
        OP_API_UT(aclnnMoeTokenUnpermuteGrad,
                  INPUT(permuteTokens, unpermutedTokensGrad, sortedIndices, nullptr, paddedMode, restoreShapeOptional),
                  OUTPUT(nullptr, probsGradOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_moe_token_unpermute_grad_test, Ascend950_moe_token_unpermute_grad_probs_grad_out_nullptr)
{
    auto permuteTokens = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto unpermutedTokensGrad = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto permutedTokensGradOut = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    bool paddedMode = false;
    std::vector<int64_t> restoreShape = {2, 5};
    auto restoreShapeOptional = aclCreateIntArray(restoreShape.data(), restoreShape.size());

    auto ut =
        OP_API_UT(aclnnMoeTokenUnpermuteGrad,
                  INPUT(permuteTokens, unpermutedTokensGrad, sortedIndices, nullptr, paddedMode, restoreShapeOptional),
                  OUTPUT(permutedTokensGradOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_moe_token_unpermute_grad_test, Ascend950_moe_token_unpermute_grad_padded_mode_invalid)
{
    auto permuteTokens = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto unpermutedTokensGrad = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto permutedTokensGradOut = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto probsGradOut = TensorDesc({6}, ACL_FLOAT16, ACL_FORMAT_ND);
    bool paddedMode = true;
    std::vector<int64_t> restoreShape = {2, 5};
    auto restoreShapeOptional = aclCreateIntArray(restoreShape.data(), restoreShape.size());

    auto ut =
        OP_API_UT(aclnnMoeTokenUnpermuteGrad,
                  INPUT(permuteTokens, unpermutedTokensGrad, sortedIndices, nullptr, paddedMode, restoreShapeOptional),
                  OUTPUT(permutedTokensGradOut, probsGradOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_moe_token_unpermute_grad_test, Ascend950_moe_token_unpermute_grad_permute_tokens_dtype_invalid)
{
    auto permuteTokens = TensorDesc({6, 5}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto unpermutedTokensGrad = TensorDesc({2, 5}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto permutedTokensGradOut = TensorDesc({6, 5}, ACL_INT8, ACL_FORMAT_ND);
    auto probsGradOut = TensorDesc({6}, ACL_INT8, ACL_FORMAT_ND);
    bool paddedMode = false;
    std::vector<int64_t> restoreShape = {2, 5};
    auto restoreShapeOptional = aclCreateIntArray(restoreShape.data(), restoreShape.size());

    auto ut =
        OP_API_UT(aclnnMoeTokenUnpermuteGrad,
                  INPUT(permuteTokens, unpermutedTokensGrad, sortedIndices, nullptr, paddedMode, restoreShapeOptional),
                  OUTPUT(permutedTokensGradOut, probsGradOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_moe_token_unpermute_grad_test, Ascend950_moe_token_unpermute_grad_sorted_indices_dtype_invalid)
{
    auto permuteTokens = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto unpermutedTokensGrad = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices = TensorDesc({6}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto permutedTokensGradOut = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto probsGradOut = TensorDesc({6}, ACL_FLOAT16, ACL_FORMAT_ND);
    bool paddedMode = false;
    std::vector<int64_t> restoreShape = {2, 5};
    auto restoreShapeOptional = aclCreateIntArray(restoreShape.data(), restoreShape.size());

    auto ut =
        OP_API_UT(aclnnMoeTokenUnpermuteGrad,
                  INPUT(permuteTokens, unpermutedTokensGrad, sortedIndices, nullptr, paddedMode, restoreShapeOptional),
                  OUTPUT(permutedTokensGradOut, probsGradOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_moe_token_unpermute_grad_test, Ascend950_moe_token_unpermute_grad_probs_optional_dtype_invalid)
{
    auto permuteTokens = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto unpermutedTokensGrad = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto probsOptional = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
    auto permutedTokensGradOut = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto probsGradOut = TensorDesc({6}, ACL_FLOAT16, ACL_FORMAT_ND);
    bool paddedMode = false;
    std::vector<int64_t> restoreShape = {2, 5};
    auto restoreShapeOptional = aclCreateIntArray(restoreShape.data(), restoreShape.size());

    auto ut = OP_API_UT(
        aclnnMoeTokenUnpermuteGrad,
        INPUT(permuteTokens, unpermutedTokensGrad, sortedIndices, probsOptional, paddedMode, restoreShapeOptional),
        OUTPUT(permutedTokensGradOut, probsGradOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_moe_token_unpermute_grad_test, Ascend950_moe_token_unpermute_grad_permuted_tokens_grad_out_dtype_invalid)
{
    auto permuteTokens = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto unpermutedTokensGrad = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto permutedTokensGradOut = TensorDesc({6, 5}, ACL_INT8, ACL_FORMAT_ND);
    auto probsGradOut = TensorDesc({6}, ACL_FLOAT16, ACL_FORMAT_ND);
    bool paddedMode = false;
    std::vector<int64_t> restoreShape = {2, 5};
    auto restoreShapeOptional = aclCreateIntArray(restoreShape.data(), restoreShape.size());

    auto ut =
        OP_API_UT(aclnnMoeTokenUnpermuteGrad,
                  INPUT(permuteTokens, unpermutedTokensGrad, sortedIndices, nullptr, paddedMode, restoreShapeOptional),
                  OUTPUT(permutedTokensGradOut, probsGradOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_moe_token_unpermute_grad_test, Ascend950_moe_token_unpermute_grad_probs_grad_out_dtype_invalid)
{
    auto permuteTokens = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto unpermutedTokensGrad = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto permutedTokensGradOut = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto probsGradOut = TensorDesc({6}, ACL_INT8, ACL_FORMAT_ND);
    bool paddedMode = false;
    std::vector<int64_t> restoreShape = {2, 5};
    auto restoreShapeOptional = aclCreateIntArray(restoreShape.data(), restoreShape.size());

    auto ut =
        OP_API_UT(aclnnMoeTokenUnpermuteGrad,
                  INPUT(permuteTokens, unpermutedTokensGrad, sortedIndices, nullptr, paddedMode, restoreShapeOptional),
                  OUTPUT(permutedTokensGradOut, probsGradOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_moe_token_unpermute_grad_test, Ascend950_moe_token_unpermute_grad_dtype_diff)
{
    auto permuteTokens = TensorDesc({6, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto unpermutedTokensGrad = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto permutedTokensGradOut = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto probsGradOut = TensorDesc({6}, ACL_FLOAT16, ACL_FORMAT_ND);
    bool paddedMode = false;
    std::vector<int64_t> restoreShape = {2, 5};
    auto restoreShapeOptional = aclCreateIntArray(restoreShape.data(), restoreShape.size());

    auto ut =
        OP_API_UT(aclnnMoeTokenUnpermuteGrad,
                  INPUT(permuteTokens, unpermutedTokensGrad, sortedIndices, nullptr, paddedMode, restoreShapeOptional),
                  OUTPUT(permutedTokensGradOut, probsGradOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_moe_token_unpermute_grad_test, Ascend950_moe_token_unpermute_grad_execute)
{
    auto permuteTokens = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto unpermutedTokensGrad = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto permutedTokensGradOut = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto probsGradOut = TensorDesc({6}, ACL_FLOAT16, ACL_FORMAT_ND);
    bool paddedMode = false;
    std::vector<int64_t> restoreShape = {2, 5};
    auto restoreShapeOptional = aclCreateIntArray(restoreShape.data(), restoreShape.size());

    auto ut =
        OP_API_UT(aclnnMoeTokenUnpermuteGrad,
                  INPUT(permuteTokens, unpermutedTokensGrad, sortedIndices, nullptr, paddedMode, restoreShapeOptional),
                  OUTPUT(permutedTokensGradOut, probsGradOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

    aclnnStatus execRet = aclnnMoeTokenUnpermuteGrad(nullptr, workspaceSize, executor, nullptr);
    EXPECT_THAT(execRet, testing::AnyOf(testing::Eq(ACLNN_SUCCESS), testing::Eq(ACLNN_ERR_PARAM_NULLPTR),
                                        testing::Eq(ACLNN_ERR_PARAM_INVALID)));
}
