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
#include <gmock/gmock.h>
#include "gtest/gtest.h"
#include "../../../../op_api/aclnn_causal_conv1d_fn.h"
#include "opdev/platform.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_causal_conv1d_fn_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    }
};

TEST_F(l2_causal_conv1d_fn_test, Ascend950_fn_fp16_basic)
{
    auto x = TensorDesc({4, 16, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto weight = TensorDesc({4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto convStates = TensorDesc({4, 4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto bias = TensorDesc({512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto y = TensorDesc({4, 16, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    const char *activation = "silu";
    auto ut = OP_API_UT(aclnnCausalConv1dFn,
                        INPUT(x, weight, convStates, bias, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, activation, (int64_t)0, (int64_t)0),
                        OUTPUT(y));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
}

TEST_F(l2_causal_conv1d_fn_test, Ascend950_fn_bf16_basic)
{
    auto x = TensorDesc({4, 16, 512}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto weight = TensorDesc({4, 512}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto convStates = TensorDesc({4, 4, 512}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto y = TensorDesc({4, 16, 512}, ACL_BF16, ACL_FORMAT_ND);
    const char *activation = "silu";
    auto ut = OP_API_UT(aclnnCausalConv1dFn,
                        INPUT(x, weight, convStates, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, activation, (int64_t)0, (int64_t)0),
                        OUTPUT(y));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
}

TEST_F(l2_causal_conv1d_fn_test, Ascend950_fn_varlen_basic)
{
    auto x = TensorDesc({64, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto weight = TensorDesc({4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto convStates = TensorDesc({4, 4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto queryStartLoc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 64);
    auto y = TensorDesc({64, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    const char *activation = "silu";
    auto ut = OP_API_UT(aclnnCausalConv1dFn,
                        INPUT(x, weight, convStates, nullptr, queryStartLoc, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, activation, (int64_t)0, (int64_t)0),
                        OUTPUT(y));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
}

TEST_F(l2_causal_conv1d_fn_test, Ascend950_fn_x_nullptr)
{
    auto weight = TensorDesc({4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto convStates = TensorDesc({4, 4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto y = TensorDesc({4, 16, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    const char *activation = "silu";
    auto ut = OP_API_UT(aclnnCausalConv1dFn,
                        INPUT(nullptr, weight, convStates, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, activation, (int64_t)0, (int64_t)0),
                        OUTPUT(y));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_causal_conv1d_fn_test, Ascend950_fn_weight_nullptr)
{
    auto x = TensorDesc({4, 16, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto convStates = TensorDesc({4, 4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto y = TensorDesc({4, 16, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    const char *activation = "silu";
    auto ut = OP_API_UT(aclnnCausalConv1dFn,
                        INPUT(x, nullptr, convStates, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, activation, (int64_t)0, (int64_t)0),
                        OUTPUT(y));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_causal_conv1d_fn_test, Ascend950_fn_convStates_nullptr)
{
    auto x = TensorDesc({4, 16, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto weight = TensorDesc({4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto y = TensorDesc({4, 16, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    const char *activation = "silu";
    auto ut = OP_API_UT(aclnnCausalConv1dFn,
                        INPUT(x, weight, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, activation, (int64_t)0, (int64_t)0),
                        OUTPUT(y));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_causal_conv1d_fn_test, Ascend950_fn_y_nullptr)
{
    auto x = TensorDesc({4, 16, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto weight = TensorDesc({4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto convStates = TensorDesc({4, 4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    const char *activation = "silu";
    auto ut = OP_API_UT(aclnnCausalConv1dFn,
                        INPUT(x, weight, convStates, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, activation, (int64_t)0, (int64_t)0),
                        OUTPUT(nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_causal_conv1d_fn_test, Ascend950_fn_x_dtype_invalid)
{
    auto x = TensorDesc({4, 16, 512}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto weight = TensorDesc({4, 512}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto convStates = TensorDesc({4, 4, 512}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto y = TensorDesc({4, 16, 512}, ACL_INT8, ACL_FORMAT_ND);
    const char *activation = "silu";
    auto ut = OP_API_UT(aclnnCausalConv1dFn,
                        INPUT(x, weight, convStates, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, activation, (int64_t)0, (int64_t)0),
                        OUTPUT(y));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_causal_conv1d_fn_test, Ascend950_fn_weight_dtype_mismatch)
{
    auto x = TensorDesc({4, 16, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto weight = TensorDesc({4, 512}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto convStates = TensorDesc({4, 4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto y = TensorDesc({4, 16, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    const char *activation = "silu";
    auto ut = OP_API_UT(aclnnCausalConv1dFn,
                        INPUT(x, weight, convStates, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, activation, (int64_t)0, (int64_t)0),
                        OUTPUT(y));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_causal_conv1d_fn_test, Ascend950_fn_y_dtype_mismatch)
{
    auto x = TensorDesc({4, 16, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto weight = TensorDesc({4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto convStates = TensorDesc({4, 4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto y = TensorDesc({4, 16, 512}, ACL_BF16, ACL_FORMAT_ND);
    const char *activation = "silu";
    auto ut = OP_API_UT(aclnnCausalConv1dFn,
                        INPUT(x, weight, convStates, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, activation, (int64_t)0, (int64_t)0),
                        OUTPUT(y));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_causal_conv1d_fn_test, Ascend950_fn_bias_dtype_mismatch)
{
    auto x = TensorDesc({4, 16, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto weight = TensorDesc({4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto convStates = TensorDesc({4, 4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto bias = TensorDesc({512}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto y = TensorDesc({4, 16, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    const char *activation = "silu";
    auto ut = OP_API_UT(aclnnCausalConv1dFn,
                        INPUT(x, weight, convStates, bias, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, activation, (int64_t)0, (int64_t)0),
                        OUTPUT(y));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_causal_conv1d_fn_test, Ascend950_fn_queryStartLoc_dtype_invalid)
{
    auto x = TensorDesc({4, 16, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto weight = TensorDesc({4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto convStates = TensorDesc({4, 4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto queryStartLoc = TensorDesc({5}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    auto y = TensorDesc({4, 16, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    const char *activation = "silu";
    auto ut = OP_API_UT(aclnnCausalConv1dFn,
                        INPUT(x, weight, convStates, nullptr, queryStartLoc, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, activation, (int64_t)0, (int64_t)0),
                        OUTPUT(y));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_causal_conv1d_fn_test, Ascend950_fn_cacheIndices_dtype_invalid)
{
    auto x = TensorDesc({4, 16, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto weight = TensorDesc({4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto convStates = TensorDesc({4, 4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto cacheIndices = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 4);
    auto y = TensorDesc({4, 16, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    const char *activation = "silu";
    auto ut = OP_API_UT(aclnnCausalConv1dFn,
                        INPUT(x, weight, convStates, nullptr, nullptr, cacheIndices, nullptr, nullptr, nullptr,
                              nullptr, nullptr, activation, (int64_t)0, (int64_t)0),
                        OUTPUT(y));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_causal_conv1d_fn_test, Ascend950_fn_initialStateMode_dtype_invalid)
{
    auto x = TensorDesc({4, 16, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto weight = TensorDesc({4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto convStates = TensorDesc({4, 4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto initialStateMode = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 1);
    auto y = TensorDesc({4, 16, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    const char *activation = "silu";
    auto ut = OP_API_UT(aclnnCausalConv1dFn,
                        INPUT(x, weight, convStates, nullptr, nullptr, nullptr, initialStateMode, nullptr, nullptr,
                              nullptr, nullptr, activation, (int64_t)0, (int64_t)0),
                        OUTPUT(y));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_causal_conv1d_fn_test, Ascend950_fn_x_dimnum_invalid)
{
    auto x = TensorDesc({4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto weight = TensorDesc({4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto convStates = TensorDesc({4, 4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto y = TensorDesc({4, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    const char *activation = "silu";
    auto ut = OP_API_UT(aclnnCausalConv1dFn,
                        INPUT(x, weight, convStates, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, activation, (int64_t)0, (int64_t)0),
                        OUTPUT(y));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_causal_conv1d_fn_test, Ascend950_fn_weight_dimnum_invalid)
{
    auto x = TensorDesc({4, 16, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto weight = TensorDesc({4, 4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto convStates = TensorDesc({4, 4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto y = TensorDesc({4, 16, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    const char *activation = "silu";
    auto ut = OP_API_UT(aclnnCausalConv1dFn,
                        INPUT(x, weight, convStates, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, activation, (int64_t)0, (int64_t)0),
                        OUTPUT(y));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_causal_conv1d_fn_test, Ascend950_fn_kernel_width_invalid)
{
    auto x = TensorDesc({4, 16, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto weight = TensorDesc({1, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto convStates = TensorDesc({4, 4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto y = TensorDesc({4, 16, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    const char *activation = "silu";
    auto ut = OP_API_UT(aclnnCausalConv1dFn,
                        INPUT(x, weight, convStates, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, activation, (int64_t)0, (int64_t)0),
                        OUTPUT(y));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_causal_conv1d_fn_test, Ascend950_fn_convStates_dimnum_invalid)
{
    auto x = TensorDesc({4, 16, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto weight = TensorDesc({4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto convStates = TensorDesc({4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto y = TensorDesc({4, 16, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    const char *activation = "silu";
    auto ut = OP_API_UT(aclnnCausalConv1dFn,
                        INPUT(x, weight, convStates, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, activation, (int64_t)0, (int64_t)0),
                        OUTPUT(y));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_causal_conv1d_fn_test, Ascend950_fn_dim_mismatch_weight)
{
    auto x = TensorDesc({4, 16, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto weight = TensorDesc({4, 256}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto convStates = TensorDesc({4, 4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto y = TensorDesc({4, 16, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    const char *activation = "silu";
    auto ut = OP_API_UT(aclnnCausalConv1dFn,
                        INPUT(x, weight, convStates, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, activation, (int64_t)0, (int64_t)0),
                        OUTPUT(y));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_causal_conv1d_fn_test, Ascend950_fn_stateLen_too_small)
{
    auto x = TensorDesc({4, 16, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto weight = TensorDesc({4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto convStates = TensorDesc({4, 2, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto y = TensorDesc({4, 16, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    const char *activation = "silu";
    auto ut = OP_API_UT(aclnnCausalConv1dFn,
                        INPUT(x, weight, convStates, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, activation, (int64_t)0, (int64_t)0),
                        OUTPUT(y));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_causal_conv1d_fn_test, Ascend950_fn_dim_too_small)
{
    auto x = TensorDesc({4, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto weight = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto convStates = TensorDesc({4, 4, 32}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto y = TensorDesc({4, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    const char *activation = "silu";
    auto ut = OP_API_UT(aclnnCausalConv1dFn,
                        INPUT(x, weight, convStates, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, activation, (int64_t)0, (int64_t)0),
                        OUTPUT(y));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_causal_conv1d_fn_test, Ascend950_fn_dim_not_aligned)
{
    auto x = TensorDesc({4, 16, 72}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto weight = TensorDesc({4, 72}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto convStates = TensorDesc({4, 4, 72}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto y = TensorDesc({4, 16, 72}, ACL_FLOAT16, ACL_FORMAT_ND);
    const char *activation = "silu";
    auto ut = OP_API_UT(aclnnCausalConv1dFn,
                        INPUT(x, weight, convStates, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, activation, (int64_t)0, (int64_t)0),
                        OUTPUT(y));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_causal_conv1d_fn_test, Ascend950_fn_batch_exceed)
{
    auto x = TensorDesc({2048, 16, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto weight = TensorDesc({4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto convStates = TensorDesc({2048, 4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto y = TensorDesc({2048, 16, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    const char *activation = "silu";
    auto ut = OP_API_UT(aclnnCausalConv1dFn,
                        INPUT(x, weight, convStates, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, activation, (int64_t)0, (int64_t)0),
                        OUTPUT(y));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_causal_conv1d_fn_test, Ascend950_fn_numCacheLines_less_than_batch)
{
    auto x = TensorDesc({4, 16, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto weight = TensorDesc({4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto convStates = TensorDesc({2, 4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto y = TensorDesc({4, 16, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    const char *activation = "silu";
    auto ut = OP_API_UT(aclnnCausalConv1dFn,
                        INPUT(x, weight, convStates, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, activation, (int64_t)0, (int64_t)0),
                        OUTPUT(y));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_causal_conv1d_fn_test, Ascend950_fn_y_dimnum_mismatch)
{
    auto x = TensorDesc({4, 16, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto weight = TensorDesc({4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto convStates = TensorDesc({4, 4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto y = TensorDesc({64, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    const char *activation = "silu";
    auto ut = OP_API_UT(aclnnCausalConv1dFn,
                        INPUT(x, weight, convStates, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, activation, (int64_t)0, (int64_t)0),
                        OUTPUT(y));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_causal_conv1d_fn_test, Ascend950_fn_3d_qsl_size_mismatch)
{
    auto x = TensorDesc({4, 16, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto weight = TensorDesc({4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto convStates = TensorDesc({4, 4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto queryStartLoc = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 64);
    auto y = TensorDesc({4, 16, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    const char *activation = "silu";
    auto ut = OP_API_UT(aclnnCausalConv1dFn,
                        INPUT(x, weight, convStates, nullptr, queryStartLoc, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, activation, (int64_t)0, (int64_t)0),
                        OUTPUT(y));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_causal_conv1d_fn_test, Ascend950_fn_3d_qsl_consistent)
{
    auto x = TensorDesc({4, 16, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto weight = TensorDesc({4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto convStates = TensorDesc({4, 4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto queryStartLoc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 64);
    auto y = TensorDesc({4, 16, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    const char *activation = "silu";
    auto ut = OP_API_UT(aclnnCausalConv1dFn,
                        INPUT(x, weight, convStates, nullptr, queryStartLoc, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, activation, (int64_t)0, (int64_t)0),
                        OUTPUT(y));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
}

TEST_F(l2_causal_conv1d_fn_test, Ascend950_fn_2d_varlen_batch_exceed)
{
    auto x = TensorDesc({2049, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto weight = TensorDesc({4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto convStates = TensorDesc({2049, 4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto queryStartLoc = TensorDesc({2050}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2049);
    auto y = TensorDesc({2049, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    const char *activation = "silu";
    auto ut = OP_API_UT(aclnnCausalConv1dFn,
                        INPUT(x, weight, convStates, nullptr, queryStartLoc, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, activation, (int64_t)0, (int64_t)0),
                        OUTPUT(y));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_causal_conv1d_fn_test, Ascend950_fn_2d_varlen_numCacheLines_less_than_batch)
{
    auto x = TensorDesc({64, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto weight = TensorDesc({4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto convStates = TensorDesc({2, 4, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto queryStartLoc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 64);
    auto y = TensorDesc({64, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    const char *activation = "silu";
    auto ut = OP_API_UT(aclnnCausalConv1dFn,
                        INPUT(x, weight, convStates, nullptr, queryStartLoc, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, activation, (int64_t)0, (int64_t)0),
                        OUTPUT(y));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}
