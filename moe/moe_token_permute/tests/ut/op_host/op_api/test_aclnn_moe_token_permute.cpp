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
#include "../../../../op_host/op_api/aclnn_moe_token_permute.h"
#include "../../../../op_host/op_api/aclnn_moe_token_permute_v2.h"
#include "opdev/platform.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_moe_token_permute_regbase_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_moe_token_permute_regbase_test SetUp" << endl;
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    }

    static void TearDownTestCase()
    {
        cout << "l2_moe_token_permute_regbase_test TearDown" << endl;
    }
};

TEST_F(l2_moe_token_permute_regbase_test, Ascend950_moe_token_permute_fp16_int32)
{
    auto tokens = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND);
    int64_t numOutTokens = 6;
    bool paddedMode = false;

    auto ut = OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, numOutTokens, paddedMode),
                        OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_moe_token_permute_regbase_test, Ascend950_moe_token_permute_fp32_int32)
{
    auto tokens = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({6, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND);
    int64_t numOutTokens = 6;
    bool paddedMode = false;

    auto ut = OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, numOutTokens, paddedMode),
                        OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_moe_token_permute_regbase_test, Ascend950_moe_token_permute_bf16_int32)
{
    auto tokens = TensorDesc({2, 5}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({6, 5}, ACL_BF16, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND);
    int64_t numOutTokens = 6;
    bool paddedMode = false;

    auto ut = OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, numOutTokens, paddedMode),
                        OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_moe_token_permute_regbase_test, Ascend950_moe_token_permute_fp16_int64)
{
    auto tokens = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND);
    int64_t numOutTokens = 6;
    bool paddedMode = false;

    auto ut = OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, numOutTokens, paddedMode),
                        OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_moe_token_permute_regbase_test, Ascend950_moe_token_permute_tokens_nullptr)
{
    auto indices = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND);
    int64_t numOutTokens = 6;
    bool paddedMode = false;

    auto ut = OP_API_UT(aclnnMoeTokenPermute, INPUT(nullptr, indices, numOutTokens, paddedMode),
                        OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_moe_token_permute_regbase_test, Ascend950_moe_token_permute_indices_nullptr)
{
    auto tokens = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto permuteTokensOut = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND);
    int64_t numOutTokens = 6;
    bool paddedMode = false;

    auto ut = OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, nullptr, numOutTokens, paddedMode),
                        OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_moe_token_permute_regbase_test, Ascend950_moe_token_permute_permute_tokens_out_nullptr)
{
    auto tokens = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto sortedIndicesOut = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND);
    int64_t numOutTokens = 6;
    bool paddedMode = false;

    auto ut = OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, numOutTokens, paddedMode),
                        OUTPUT(nullptr, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_moe_token_permute_regbase_test, Ascend950_moe_token_permute_sorted_indices_out_nullptr)
{
    auto tokens = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t numOutTokens = 6;
    bool paddedMode = false;

    auto ut = OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, numOutTokens, paddedMode),
                        OUTPUT(permuteTokensOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_moe_token_permute_regbase_test, Ascend950_moe_token_permute_padded_mode_invalid)
{
    auto tokens = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND);
    int64_t numOutTokens = 6;
    bool paddedMode = true;

    auto ut = OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, numOutTokens, paddedMode),
                        OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_moe_token_permute_regbase_test, Ascend950_moe_token_permute_indices_1d)
{
    auto tokens = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND);
    int64_t numOutTokens = 6;
    bool paddedMode = false;

    auto ut = OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, numOutTokens, paddedMode),
                        OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_moe_token_permute_regbase_test, Ascend950_moe_token_permute_num_out_tokens_diff)
{
    auto tokens = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({4, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);
    int64_t numOutTokens = 4;
    bool paddedMode = false;

    auto ut = OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, numOutTokens, paddedMode),
                        OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_moe_token_permute_regbase_test, Ascend950_moe_token_permute_int8_int32)
{
    auto tokens = TensorDesc({2, 5}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({6, 5}, ACL_INT8, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND);
    int64_t numOutTokens = 6;
    bool paddedMode = false;

    auto ut = OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, numOutTokens, paddedMode),
                        OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_moe_token_permute_regbase_test, Ascend950_moe_token_permute_int8_int64_1d_slice)
{
    auto tokens = TensorDesc({6, 5}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({6}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({4, 5}, ACL_INT8, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);
    int64_t numOutTokens = 4;
    bool paddedMode = false;

    auto ut = OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, numOutTokens, paddedMode),
                        OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_moe_token_permute_regbase_test, Ascend950_moe_token_permute_int8_output_dtype_invalid)
{
    auto tokens = TensorDesc({2, 5}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND);
    int64_t numOutTokens = 6;
    bool paddedMode = false;

    auto ut = OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, numOutTokens, paddedMode),
                        OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_moe_token_permute_regbase_test, Ascend950_moe_token_permute_indices_dtype_invalid)
{
    auto tokens = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND);
    int64_t numOutTokens = 6;
    bool paddedMode = false;

    auto ut = OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, numOutTokens, paddedMode),
                        OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_moe_token_permute_regbase_test, Ascend950_moe_token_permute_sorted_indices_dtype_invalid)
{
    auto tokens = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({6}, ACL_INT64, ACL_FORMAT_ND);
    int64_t numOutTokens = 6;
    bool paddedMode = false;

    auto ut = OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, numOutTokens, paddedMode),
                        OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_moe_token_permute_regbase_test, Ascend950_moe_token_permute_permute_tokens_dtype_diff)
{
    auto tokens = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({6, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND);
    int64_t numOutTokens = 6;
    bool paddedMode = false;

    auto ut = OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, numOutTokens, paddedMode),
                        OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_moe_token_permute_regbase_test, Ascend950_moe_token_permute_v2_mxfp8_e5m2)
{
    auto tokens = TensorDesc({2, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({6, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND);
    auto expandedScaleOut = TensorDesc({6, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeTokenPermuteV2, INPUT(tokens, indices, 6, false, 2),
                        OUTPUT(permuteTokensOut, sortedIndicesOut, expandedScaleOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_SUCCESS);
}

TEST_F(l2_moe_token_permute_regbase_test, Ascend950_moe_token_permute_v2_int8_unquant)
{
    auto tokens = TensorDesc({2, 64}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({6, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND);
    auto expandedScaleOut = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeTokenPermuteV2, INPUT(tokens, indices, 6, false, -1),
                        OUTPUT(permuteTokensOut, sortedIndicesOut, expandedScaleOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_SUCCESS);
}

TEST_F(l2_moe_token_permute_regbase_test, Ascend950_moe_token_permute_v2_tokens_rank_invalid)
{
    auto tokens = TensorDesc({2, 64, 1}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({6, 64, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND);
    auto expandedScaleOut = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeTokenPermuteV2, INPUT(tokens, indices, 6, false, -1),
                        OUTPUT(permuteTokensOut, sortedIndicesOut, expandedScaleOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_moe_token_permute_regbase_test, Ascend950_moe_token_permute_v2_unquant_slice)
{
    auto tokens = TensorDesc({2, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({2, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 100);
    auto permuteTokensOut = TensorDesc({10, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND);
    auto expandedScaleOut = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeTokenPermuteV2, INPUT(tokens, indices, 10, false, -1),
                        OUTPUT(permuteTokensOut, sortedIndicesOut, expandedScaleOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_SUCCESS);
}

TEST_F(l2_moe_token_permute_regbase_test, Ascend950_moe_token_permute_v2_mxfp8_e4m3fn)
{
    auto tokens = TensorDesc({2, 64}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({6, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND);
    auto expandedScaleOut = TensorDesc({6, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeTokenPermuteV2, INPUT(tokens, indices, 6, false, 3),
                        OUTPUT(permuteTokensOut, sortedIndicesOut, expandedScaleOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_SUCCESS);
}

TEST_F(l2_moe_token_permute_regbase_test, Ascend950_moe_token_permute_v2_mxfp4_e2m1)
{
    auto tokens = TensorDesc({2, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({6, 64}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND);
    auto expandedScaleOut = TensorDesc({6, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeTokenPermuteV2, INPUT(tokens, indices, 6, false, 9),
                        OUTPUT(permuteTokensOut, sortedIndicesOut, expandedScaleOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_SUCCESS);
}

TEST_F(l2_moe_token_permute_regbase_test, Ascend950_moe_token_permute_execute)
{
    auto tokens = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND);
    int64_t numOutTokens = 6;
    bool paddedMode = false;

    auto ut = OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, numOutTokens, paddedMode),
                        OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

    aclnnStatus execRet = aclnnMoeTokenPermute(nullptr, workspaceSize, executor, nullptr);
    EXPECT_THAT(execRet, testing::AnyOf(testing::Eq(ACLNN_SUCCESS), testing::Eq(ACLNN_ERR_PARAM_NULLPTR),
                                        testing::Eq(ACLNN_ERR_PARAM_INVALID)));
}
