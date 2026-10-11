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
#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_moe_token_permute_with_routing_map_grad.h"
#include "../../../../op_host/op_api/moe_token_permute_with_routing_map_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/shape_utils.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_moe_token_permute_with_routing_map_grad_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_moe_token_permute_with_routing_map_grad_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_moe_token_permute_with_routing_map_grad_test TearDown" << endl;
    }
};

TEST_F(l2_moe_token_permute_with_routing_map_grad_test,
       Ascend910B2_moe_token_permute_with_routing_map_grad_droppad_true_fp32)
{
    auto permutedTokenOutPutGrad = TensorDesc({1024, 7168}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto permutedProbsOutPutGradOptional = TensorDesc({1024}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 0);
    auto routingMapOptional = TensorDesc({512, 512}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto tokensGradOut = TensorDesc({512, 7168}, ACL_FLOAT, ACL_FORMAT_ND);
    auto probsGradOutOptional = TensorDesc({512, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t numExperts = 512;
    int64_t tokensNum = 512;
    bool paddedNum = true;
    auto ut = OP_API_UT(aclnnMoeTokenPermuteWithRoutingMapGrad,
                        INPUT(permutedTokenOutPutGrad, permutedProbsOutPutGradOptional, sortedIndices,
                              routingMapOptional, numExperts, tokensNum, paddedNum),
                        OUTPUT(tokensGradOut, probsGradOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 0);
}

TEST_F(l2_moe_token_permute_with_routing_map_grad_test,
       Ascend910B2_moe_token_permute_with_routing_map_grad_droppad_false_fp32)
{
    auto permutedTokenOutPutGrad = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto permutedProbsOutPutGradOptional = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 7);
    auto routingMapOptional = TensorDesc({8, 2}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 1);
    auto tokensGradOut = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto probsGradOutOptional = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t numExperts = 2;
    int64_t tokensNum = 8;
    bool paddedNum = false;
    auto ut = OP_API_UT(aclnnMoeTokenPermuteWithRoutingMapGrad,
                        INPUT(permutedTokenOutPutGrad, permutedProbsOutPutGradOptional, sortedIndices,
                              routingMapOptional, numExperts, tokensNum, paddedNum),
                        OUTPUT(tokensGradOut, probsGradOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 0);
}

TEST_F(l2_moe_token_permute_with_routing_map_grad_test,
       Ascend910B2_moe_token_permute_with_routing_map_grad_droppad_false_fp32_no_prob)
{
    auto permutedTokenOutPutGrad = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 7);
    auto routingMapOptional = TensorDesc({8, 2}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 1);
    auto tokensGradOut = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t numExperts = 2;
    int64_t tokensNum = 8;
    bool paddedNum = false;
    auto ut = OP_API_UT(
        aclnnMoeTokenPermuteWithRoutingMapGrad,
        INPUT(permutedTokenOutPutGrad, nullptr, sortedIndices, routingMapOptional, numExperts, tokensNum, paddedNum),
        OUTPUT(tokensGradOut, nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 0);
}

TEST_F(l2_moe_token_permute_with_routing_map_grad_test,
       Ascend910B2_moe_token_permute_with_routing_map_grad_droppad_true_fp16)
{
    auto permutedTokenOutPutGrad = TensorDesc({1024, 7168}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto permutedProbsOutPutGradOptional = TensorDesc({1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 0);
    auto routingMapOptional = TensorDesc({512, 512}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto tokensGradOut = TensorDesc({512, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto probsGradOutOptional = TensorDesc({512, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t numExperts = 512;
    int64_t tokensNum = 512;
    bool paddedNum = true;
    auto ut = OP_API_UT(aclnnMoeTokenPermuteWithRoutingMapGrad,
                        INPUT(permutedTokenOutPutGrad, permutedProbsOutPutGradOptional, sortedIndices,
                              routingMapOptional, numExperts, tokensNum, paddedNum),
                        OUTPUT(tokensGradOut, probsGradOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 0);
}

TEST_F(l2_moe_token_permute_with_routing_map_grad_test,
       Ascend910B2_moe_token_permute_with_routing_map_grad_droppad_true_bf16_mix)
{
    auto permutedTokenOutPutGrad = TensorDesc({1024, 7168}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto permutedProbsOutPutGradOptional = TensorDesc({1024}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 0);
    auto routingMapOptional = TensorDesc({512, 512}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto tokensGradOut = TensorDesc({512, 7168}, ACL_BF16, ACL_FORMAT_ND);
    auto probsGradOutOptional = TensorDesc({512, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t numExperts = 512;
    int64_t tokensNum = 512;
    bool paddedNum = true;
    auto ut = OP_API_UT(aclnnMoeTokenPermuteWithRoutingMapGrad,
                        INPUT(permutedTokenOutPutGrad, permutedProbsOutPutGradOptional, sortedIndices,
                              routingMapOptional, numExperts, tokensNum, paddedNum),
                        OUTPUT(tokensGradOut, probsGradOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 0);
}

TEST_F(l2_moe_token_permute_with_routing_map_grad_test,
       Ascend910B2_moe_token_permute_with_routing_map_grad_droppad_false_bf16_mix)
{
    auto permutedTokenOutPutGrad = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto permutedProbsOutPutGradOptional = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 7);
    auto routingMapOptional = TensorDesc({8, 2}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 1);
    auto tokensGradOut = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto probsGradOutOptional = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t numExperts = 2;
    int64_t tokensNum = 8;
    bool paddedNum = false;
    auto ut = OP_API_UT(aclnnMoeTokenPermuteWithRoutingMapGrad,
                        INPUT(permutedTokenOutPutGrad, permutedProbsOutPutGradOptional, sortedIndices,
                              routingMapOptional, numExperts, tokensNum, paddedNum),
                        OUTPUT(tokensGradOut, probsGradOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 0);
}

TEST_F(l2_moe_token_permute_with_routing_map_grad_test,
       Ascend910B2_moe_token_permute_with_routing_map_grad_empty_tensor)
{
    auto permutedTokenOutPutGrad = TensorDesc({1024, 0}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto permutedProbsOutPutGradOptional = TensorDesc({1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 0);
    auto routingMapOptional = TensorDesc({512, 512}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto tokensGradOut = TensorDesc({512, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto probsGradOutOptional = TensorDesc({512, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t numExperts = 512;
    int64_t tokensNum = 512;
    bool paddedNum = true;
    auto ut = OP_API_UT(aclnnMoeTokenPermuteWithRoutingMapGrad,
                        INPUT(permutedTokenOutPutGrad, permutedProbsOutPutGradOptional, sortedIndices,
                              routingMapOptional, numExperts, tokensNum, paddedNum),
                        OUTPUT(tokensGradOut, probsGradOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 0);
}

TEST_F(l2_moe_token_permute_with_routing_map_grad_test,
       Ascend910B2_moe_token_permute_with_routing_map_grad_prob_without_routing_map)
{
    auto permutedTokenOutPutGrad = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto permutedProbsOutPutGradOptional = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 7);
    auto tokensGradOut = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t numExperts = 2;
    int64_t tokensNum = 8;
    bool paddedNum = false;
    auto ut = OP_API_UT(aclnnMoeTokenPermuteWithRoutingMapGrad,
                        INPUT(permutedTokenOutPutGrad, permutedProbsOutPutGradOptional, sortedIndices, nullptr,
                              numExperts, tokensNum, paddedNum),
                        OUTPUT(tokensGradOut, nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_moe_token_permute_with_routing_map_grad_test,
       Ascend910B2_moe_token_permute_with_routing_map_grad_invalid_tokens_num)
{
    auto permutedTokenOutPutGrad = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 7);
    auto routingMapOptional = TensorDesc({8, 2}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 1);
    auto tokensGradOut = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t numExperts = 2;
    int64_t tokensNum = 0;
    bool paddedNum = true;
    auto ut = OP_API_UT(
        aclnnMoeTokenPermuteWithRoutingMapGrad,
        INPUT(permutedTokenOutPutGrad, nullptr, sortedIndices, routingMapOptional, numExperts, tokensNum, paddedNum),
        OUTPUT(tokensGradOut, nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_moe_token_permute_with_routing_map_grad_test,
       Ascend910B2_moe_token_permute_with_routing_map_grad_invalid_attr)
{
    auto permutedTokenOutPutGrad = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 7);
    auto routingMapOptional = TensorDesc({8, 2}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 1);
    auto tokensGradOut = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t numExperts = 0;
    int64_t tokensNum = 0;
    bool paddedNum = true;
    auto ut = OP_API_UT(
        aclnnMoeTokenPermuteWithRoutingMapGrad,
        INPUT(permutedTokenOutPutGrad, nullptr, sortedIndices, routingMapOptional, numExperts, tokensNum, paddedNum),
        OUTPUT(tokensGradOut, nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_moe_token_permute_with_routing_map_grad_test,
       Ascend910B2_moe_token_permute_with_routing_map_grad_private_format_warning)
{
    auto permutedTokenOutPutGrad = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ).ValueRange(-10, 10);
    auto permutedProbsOutPutGradOptional = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ).ValueRange(-10, 10);
    auto sortedIndices = TensorDesc({8}, ACL_INT32, ACL_FORMAT_FRACTAL_NZ).ValueRange(0, 7);
    auto routingMapOptional = TensorDesc({8, 2}, ACL_BOOL, ACL_FORMAT_FRACTAL_NZ).ValueRange(0, 1);
    auto tokensGradOut = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto probsGradOutOptional = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t numExperts = 2;
    int64_t tokensNum = 8;
    bool paddedNum = false;
    auto ut = OP_API_UT(aclnnMoeTokenPermuteWithRoutingMapGrad,
                        INPUT(permutedTokenOutPutGrad, permutedProbsOutPutGradOptional, sortedIndices,
                              routingMapOptional, numExperts, tokensNum, paddedNum),
                        OUTPUT(tokensGradOut, probsGradOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 0);
}

namespace {
aclTensor* CreateL0AclTensor(const std::vector<int64_t>& shape, aclDataType dtype)
{
    static int64_t data[4096] = {0};
    return aclCreateTensor(shape.data(), shape.size(), dtype, nullptr, 0, ACL_FORMAT_ND, shape.data(), shape.size(),
                           data);
}
} // namespace

class l2_moe_token_permute_with_routing_map_grad_l0_test : public testing::Test {
protected:
    l2_moe_token_permute_with_routing_map_grad_l0_test()
        : exe(nullptr)
    {}

    void SetUp() override
    {
        auto executor = &exe;
        auto uniqueExecutor = CREATE_EXECUTOR();
        uniqueExecutor.ReleaseTo(executor);
    }

    void TearDown() override
    {
        delete exe;
        exe = nullptr;
    }

    aclOpExecutor* exe;
};

TEST_F(l2_moe_token_permute_with_routing_map_grad_l0_test, MoeTokenPermuteWithRoutingMapGrad_l0_drop_and_pad_false)
{
    auto permutedTokenGrad = CreateL0AclTensor({8, 64}, ACL_FLOAT);
    auto sortedIndices = CreateL0AclTensor({8}, ACL_INT32);
    auto routingMap = CreateL0AclTensor({3, 8}, ACL_BOOL);
    auto tokensGradOut = CreateL0AclTensor({8, 64}, ACL_FLOAT);
    auto probsGradOut = CreateL0AclTensor({8}, ACL_FLOAT);
    auto result = l0op::MoeTokenPermuteWithRoutingMapGrad(permutedTokenGrad, nullptr, sortedIndices, routingMap, 3, 8,
                                                          false, tokensGradOut, probsGradOut, exe);
    EXPECT_NE(result[0], nullptr);
    EXPECT_NE(result[1], nullptr);
}

TEST_F(l2_moe_token_permute_with_routing_map_grad_l0_test,
       MoeTokenPermuteWithRoutingMapGrad_l0_drop_and_pad_true_without_prob)
{
    auto permutedTokenGrad = CreateL0AclTensor({8, 64}, ACL_FLOAT);
    auto sortedIndices = CreateL0AclTensor({8}, ACL_INT32);
    auto routingMap = CreateL0AclTensor({3, 8}, ACL_BOOL);
    auto probsGradOut = CreateL0AclTensor({8}, ACL_FLOAT);
    auto result = l0op::MoeTokenPermuteWithRoutingMapGrad(permutedTokenGrad, nullptr, sortedIndices, routingMap, 3, 8,
                                                          true, nullptr, probsGradOut, exe);
    EXPECT_NE(result[0], nullptr);
    EXPECT_NE(result[1], nullptr);
}

TEST_F(l2_moe_token_permute_with_routing_map_grad_l0_test,
       MoeTokenPermuteWithRoutingMapGrad_l0_drop_and_pad_false_with_prob)
{
    auto permutedTokenGrad = CreateL0AclTensor({8, 64}, ACL_FLOAT);
    auto permutedProbsGrad = CreateL0AclTensor({8}, ACL_FLOAT);
    auto sortedIndices = CreateL0AclTensor({8}, ACL_INT32);
    auto routingMap = CreateL0AclTensor({3, 8}, ACL_BOOL);
    auto tokensGradOut = CreateL0AclTensor({8, 64}, ACL_FLOAT);
    auto probsGradOut = CreateL0AclTensor({8}, ACL_FLOAT);
    auto result = l0op::MoeTokenPermuteWithRoutingMapGrad(permutedTokenGrad, permutedProbsGrad, sortedIndices,
                                                          routingMap, 3, 8, false, tokensGradOut, probsGradOut, exe);
    EXPECT_NE(result[0], nullptr);
    EXPECT_NE(result[1], nullptr);
}

TEST(l2_moe_token_permute_with_routing_map_grad_phase2_test, phase2_null_executor_path)
{
    aclnnStatus ret = aclnnMoeTokenPermuteWithRoutingMapGrad(nullptr, 0, nullptr, nullptr);
    EXPECT_NE(ret, ACLNN_SUCCESS);
}
