/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "../../../../op_host/op_api/aclnn_moe_finalize_routing_v2_grad.h"
#include "opdev/platform.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_moe_finalize_routing_v2_grad_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_moe_finalize_routing_v2_grad_test SetUp" << endl;
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    }

    static void TearDownTestCase()
    {
        cout << "l2_moe_finalize_routing_v2_grad_test TearDown" << endl;
    }
};

TEST_F(l2_moe_finalize_routing_v2_grad_test, Ascend910B2_moe_finalize_routing_v2_grad_fp32)
{
    auto gradY = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10);
    auto expandedRowIdx = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 10);
    auto expandedXOptional = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10);
    auto scalesOptional = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10);
    auto gradExpandedXOut = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradScalesOut = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeFinalizeRoutingV2Grad,
                        INPUT(gradY, expandedRowIdx, expandedXOptional, scalesOptional, (aclTensor*)nullptr,
                              (aclTensor*)nullptr, 0, 0, 0, 0),
                        OUTPUT(gradExpandedXOut, gradScalesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_moe_finalize_routing_v2_grad_test, Ascend910B2_moe_finalize_routing_v2_grad_fp16)
{
    auto gradY = TensorDesc({1, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 10);
    auto expandedRowIdx = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 10);
    auto expandedXOptional = TensorDesc({1, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 10);
    auto scalesOptional = TensorDesc({1, 1}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 10);
    auto gradExpandedXOut = TensorDesc({1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gradScalesOut = TensorDesc({1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeFinalizeRoutingV2Grad,
                        INPUT(gradY, expandedRowIdx, expandedXOptional, scalesOptional, (aclTensor*)nullptr,
                              (aclTensor*)nullptr, 0, 0, 0, 0),
                        OUTPUT(gradExpandedXOut, gradScalesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_moe_finalize_routing_v2_grad_test, Ascend910B2_moe_finalize_routing_v2_grad_bf16)
{
    auto gradY = TensorDesc({1, 64}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 10);
    auto expandedRowIdx = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 10);
    auto expandedXOptional = TensorDesc({1, 64}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 10);
    auto scalesOptional = TensorDesc({1, 1}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 10);
    auto gradExpandedXOut = TensorDesc({1, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto gradScalesOut = TensorDesc({1, 1}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeFinalizeRoutingV2Grad,
                        INPUT(gradY, expandedRowIdx, expandedXOptional, scalesOptional, (aclTensor*)nullptr,
                              (aclTensor*)nullptr, 0, 0, 0, 0),
                        OUTPUT(gradExpandedXOut, gradScalesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_moe_finalize_routing_v2_grad_test, Ascend910B2_moe_finalize_routing_v2_grad_with_expert_idx_and_bias)
{
    auto gradY = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10);
    auto expandedRowIdx = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 10);
    auto expandedXOptional = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10);
    auto scalesOptional = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10);
    auto expertIdxOptional = TensorDesc({1, 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 10);
    auto biasOptional = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10);
    auto gradExpandedXOut = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradScalesOut = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnMoeFinalizeRoutingV2Grad,
        INPUT(gradY, expandedRowIdx, expandedXOptional, scalesOptional, expertIdxOptional, biasOptional, 0, 0, 0, 0),
        OUTPUT(gradExpandedXOut, gradScalesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_moe_finalize_routing_v2_grad_test, Ascend910B2_moe_finalize_routing_v2_grad_grad_y_nullptr)
{
    auto expandedRowIdx = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 10);
    auto gradExpandedXOut = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradScalesOut = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeFinalizeRoutingV2Grad,
                        INPUT(nullptr, expandedRowIdx, (aclTensor*)nullptr, (aclTensor*)nullptr, (aclTensor*)nullptr,
                              (aclTensor*)nullptr, 0, 0, 0, 0),
                        OUTPUT(gradExpandedXOut, gradScalesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_moe_finalize_routing_v2_grad_test, Ascend910B2_moe_finalize_routing_v2_grad_expanded_row_idx_nullptr)
{
    auto gradY = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10);
    auto gradExpandedXOut = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradScalesOut = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeFinalizeRoutingV2Grad,
                        INPUT(gradY, nullptr, (aclTensor*)nullptr, (aclTensor*)nullptr, (aclTensor*)nullptr,
                              (aclTensor*)nullptr, 0, 0, 0, 0),
                        OUTPUT(gradExpandedXOut, gradScalesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_moe_finalize_routing_v2_grad_test, Ascend910B2_moe_finalize_routing_v2_grad_grad_expanded_x_out_nullptr)
{
    auto gradY = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10);
    auto expandedRowIdx = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 10);
    auto gradScalesOut = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeFinalizeRoutingV2Grad,
                        INPUT(gradY, expandedRowIdx, (aclTensor*)nullptr, (aclTensor*)nullptr, (aclTensor*)nullptr,
                              (aclTensor*)nullptr, 0, 0, 0, 0),
                        OUTPUT(nullptr, gradScalesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_moe_finalize_routing_v2_grad_test, Ascend910B2_moe_finalize_routing_v2_grad_grad_scales_out_nullptr)
{
    auto gradY = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10);
    auto expandedRowIdx = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 10);
    auto gradExpandedXOut = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeFinalizeRoutingV2Grad,
                        INPUT(gradY, expandedRowIdx, (aclTensor*)nullptr, (aclTensor*)nullptr, (aclTensor*)nullptr,
                              (aclTensor*)nullptr, 0, 0, 0, 0),
                        OUTPUT(gradExpandedXOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_moe_finalize_routing_v2_grad_test, Ascend910B2_moe_finalize_routing_v2_grad_grad_y_dtype_invalid)
{
    auto gradY = TensorDesc({1, 64}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 10);
    auto expandedRowIdx = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 10);
    auto gradExpandedXOut = TensorDesc({1, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto gradScalesOut = TensorDesc({1, 1}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeFinalizeRoutingV2Grad,
                        INPUT(gradY, expandedRowIdx, (aclTensor*)nullptr, (aclTensor*)nullptr, (aclTensor*)nullptr,
                              (aclTensor*)nullptr, 0, 0, 0, 0),
                        OUTPUT(gradExpandedXOut, gradScalesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_moe_finalize_routing_v2_grad_test, Ascend910B2_moe_finalize_routing_v2_grad_expanded_row_idx_dtype_invalid)
{
    auto gradY = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10);
    auto expandedRowIdx = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 10);
    auto gradExpandedXOut = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradScalesOut = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeFinalizeRoutingV2Grad,
                        INPUT(gradY, expandedRowIdx, (aclTensor*)nullptr, (aclTensor*)nullptr, (aclTensor*)nullptr,
                              (aclTensor*)nullptr, 0, 0, 0, 0),
                        OUTPUT(gradExpandedXOut, gradScalesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_moe_finalize_routing_v2_grad_test, Ascend910B2_moe_finalize_routing_v2_grad_expanded_x_dtype_diff)
{
    auto gradY = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10);
    auto expandedRowIdx = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 10);
    auto expandedXOptional = TensorDesc({1, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 10);
    auto scalesOptional = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10);
    auto gradExpandedXOut = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradScalesOut = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeFinalizeRoutingV2Grad,
                        INPUT(gradY, expandedRowIdx, expandedXOptional, scalesOptional, (aclTensor*)nullptr,
                              (aclTensor*)nullptr, 0, 0, 0, 0),
                        OUTPUT(gradExpandedXOut, gradScalesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_moe_finalize_routing_v2_grad_test, Ascend910B2_moe_finalize_routing_v2_grad_expert_idx_dtype_invalid)
{
    auto gradY = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10);
    auto expandedRowIdx = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 10);
    auto expandedXOptional = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10);
    auto scalesOptional = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10);
    auto expertIdxOptional = TensorDesc({1, 1}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 10);
    auto gradExpandedXOut = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradScalesOut = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeFinalizeRoutingV2Grad,
                        INPUT(gradY, expandedRowIdx, expandedXOptional, scalesOptional, expertIdxOptional,
                              (aclTensor*)nullptr, 0, 0, 0, 0),
                        OUTPUT(gradExpandedXOut, gradScalesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_moe_finalize_routing_v2_grad_test, Ascend910B2_moe_finalize_routing_v2_grad_bias_dtype_diff)
{
    auto gradY = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10);
    auto expandedRowIdx = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 10);
    auto expandedXOptional = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10);
    auto scalesOptional = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10);
    auto expertIdxOptional = TensorDesc({1, 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 10);
    auto biasOptional = TensorDesc({1, 64}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 10);
    auto gradExpandedXOut = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradScalesOut = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnMoeFinalizeRoutingV2Grad,
        INPUT(gradY, expandedRowIdx, expandedXOptional, scalesOptional, expertIdxOptional, biasOptional, 0, 0, 0, 0),
        OUTPUT(gradExpandedXOut, gradScalesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_moe_finalize_routing_v2_grad_test, Ascend910B2_moe_finalize_routing_v2_grad_grad_expanded_x_out_dtype_diff)
{
    auto gradY = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10);
    auto expandedRowIdx = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 10);
    auto gradExpandedXOut = TensorDesc({1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gradScalesOut = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeFinalizeRoutingV2Grad,
                        INPUT(gradY, expandedRowIdx, (aclTensor*)nullptr, (aclTensor*)nullptr, (aclTensor*)nullptr,
                              (aclTensor*)nullptr, 0, 0, 0, 0),
                        OUTPUT(gradExpandedXOut, gradScalesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_moe_finalize_routing_v2_grad_test, Ascend910B2_moe_finalize_routing_v2_grad_execute)
{
    auto gradY = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10);
    auto expandedRowIdx = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 10);
    auto expandedXOptional = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10);
    auto scalesOptional = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10);
    auto gradExpandedXOut = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradScalesOut = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeFinalizeRoutingV2Grad,
                        INPUT(gradY, expandedRowIdx, expandedXOptional, scalesOptional, (aclTensor*)nullptr,
                              (aclTensor*)nullptr, 0, 0, 0, 0),
                        OUTPUT(gradExpandedXOut, gradScalesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

    aclnnStatus execRet = aclnnMoeFinalizeRoutingV2Grad(nullptr, workspaceSize, executor, nullptr);
    EXPECT_THAT(execRet, testing::AnyOf(testing::Eq(ACLNN_SUCCESS), testing::Eq(ACLNN_ERR_PARAM_NULLPTR),
                                        testing::Eq(ACLNN_ERR_PARAM_INVALID)));
}
