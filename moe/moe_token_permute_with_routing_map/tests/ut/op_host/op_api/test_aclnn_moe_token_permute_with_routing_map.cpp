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
#include "../../../../op_host/op_api/aclnn_moe_token_permute_with_routing_map.h"
#include "../../../../op_host/op_api/moe_token_permute_with_routing_map.h"
#include "opdev/platform.h"
#include "opdev/make_op_executor.h"
#include "opdev/shape_utils.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_moe_token_permute_with_routing_map_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_moe_token_permute_with_routing_map_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_moe_token_permute_with_routing_map_test TearDown" << endl;
    }
};

// dtype fp32
TEST_F(l2_moe_token_permute_with_routing_map_test, Ascend910B2_moe_token_permute_with_routing_map)
{
    auto tokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({1, 8}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeTokenPermuteWithRoutingMap, INPUT(tokens, indices, nullptr, 8, false),
                        OUTPUT(permuteTokensOut, nullptr, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
}

TEST_F(l2_moe_token_permute_with_routing_map_test, Ascend910B2_moe_token_permute_with_routing_map_prob)
{
    auto tokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({1, 8}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 8);
    auto prob = TensorDesc({1, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 8);
    auto probOut = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 8);

    auto permuteTokensOut = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeTokenPermuteWithRoutingMap, INPUT(tokens, indices, prob, 8, false),
                        OUTPUT(permuteTokensOut, probOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
}

TEST_F(l2_moe_token_permute_with_routing_map_test, Ascend910B2_moe_token_permute_with_routing_map_prob_error_1)
{
    auto tokens = TensorDesc({1, 64}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({1, 8}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 8);
    auto prob = TensorDesc({1, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 8);
    auto probOut = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 8);

    auto permuteTokensOut = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeTokenPermuteWithRoutingMap, INPUT(tokens, indices, prob, 8, false),
                        OUTPUT(permuteTokensOut, probOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
}

TEST_F(l2_moe_token_permute_with_routing_map_test, Ascend910B2_moe_token_permute_with_routing_map_prob_error_2)
{
    auto tokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({1, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto prob = TensorDesc({1, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 8);
    auto probOut = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 8);

    auto permuteTokensOut = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeTokenPermuteWithRoutingMap, INPUT(tokens, indices, prob, 8, false),
                        OUTPUT(permuteTokensOut, probOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
}

TEST_F(l2_moe_token_permute_with_routing_map_test, Ascend910B2_moe_token_permute_with_routing_map_prob_error_3)
{
    auto tokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({1, 8}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 8);
    auto prob = TensorDesc({1, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto probOut = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 8);

    auto permuteTokensOut = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeTokenPermuteWithRoutingMap, INPUT(tokens, indices, prob, 8, false),
                        OUTPUT(permuteTokensOut, probOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
}

TEST_F(l2_moe_token_permute_with_routing_map_test, Ascend910B2_moe_token_permute_with_routing_map_prob_error_4)
{
    auto tokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({1, 8}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 8);
    auto prob = TensorDesc({1, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 8);
    auto probOut = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);

    auto permuteTokensOut = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeTokenPermuteWithRoutingMap, INPUT(tokens, indices, prob, 8, false),
                        OUTPUT(permuteTokensOut, probOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
}

TEST_F(l2_moe_token_permute_with_routing_map_test, Ascend910B2_moe_token_permute_with_routing_map_prob_error_5)
{
    auto tokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({1, 8}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 8);
    auto prob = TensorDesc({1, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 8);
    auto probOut = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 8);

    auto permuteTokensOut = TensorDesc({8, 64}, ACL_INT32, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeTokenPermuteWithRoutingMap, INPUT(tokens, indices, prob, 8, false),
                        OUTPUT(permuteTokensOut, probOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
}

TEST_F(l2_moe_token_permute_with_routing_map_test, Ascend910B2_moe_token_permute_with_routing_map_prob_error_6)
{
    auto tokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({1, 8}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 8);
    auto prob = TensorDesc({1, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 8);
    auto probOut = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 8);

    auto permuteTokensOut = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeTokenPermuteWithRoutingMap, INPUT(tokens, indices, prob, 8, false),
                        OUTPUT(permuteTokensOut, probOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
}

TEST_F(l2_moe_token_permute_with_routing_map_test, Ascend910B2_moe_token_permute_with_routing_map_prob_error_7)
{
    auto tokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({1, 2, 8}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 8);
    auto prob = TensorDesc({1, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 8);
    auto probOut = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 8);

    auto permuteTokensOut = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeTokenPermuteWithRoutingMap, INPUT(tokens, indices, prob, 8, false),
                        OUTPUT(permuteTokensOut, probOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
}

TEST_F(l2_moe_token_permute_with_routing_map_test, Ascend910B2_moe_token_permute_with_routing_map_prob_error_8)
{
    auto tokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({1, 8}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 8);
    auto prob = TensorDesc({1, 2, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 8);
    auto probOut = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 8);

    auto permuteTokensOut = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeTokenPermuteWithRoutingMap, INPUT(tokens, indices, prob, 8, false),
                        OUTPUT(permuteTokensOut, probOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
}

TEST_F(l2_moe_token_permute_with_routing_map_test, Ascend910B2_moe_token_permute_with_routing_map_pad)
{
    auto tokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({1, 8}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeTokenPermuteWithRoutingMap, INPUT(tokens, indices, nullptr, 8, true),
                        OUTPUT(permuteTokensOut, nullptr, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
}

TEST_F(l2_moe_token_permute_with_routing_map_test, Ascend910B2_moe_token_permute_with_routing_map_pad_prob)
{
    auto tokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({1, 8}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto prob = TensorDesc({1, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 8);
    auto probOut = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 8);
    auto ut = OP_API_UT(aclnnMoeTokenPermuteWithRoutingMap, INPUT(tokens, indices, prob, 8, true),
                        OUTPUT(permuteTokensOut, probOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
}

TEST_F(l2_moe_token_permute_with_routing_map_test, Ascend910B2_moe_token_permute_with_routing_map_pad_mix)
{
    auto tokens = TensorDesc({1, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({1, 8}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto prob = TensorDesc({1, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 8);
    auto probOut = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 8);
    auto ut = OP_API_UT(aclnnMoeTokenPermuteWithRoutingMap, INPUT(tokens, indices, prob, 8, true),
                        OUTPUT(permuteTokensOut, probOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
}

TEST_F(l2_moe_token_permute_with_routing_map_test, Ascend910B2_moe_token_permute_with_routing_map_prob_error_prob_shape)
{
    auto tokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({1, 8}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 8);
    auto prob = TensorDesc({1, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 8);
    auto probOut = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 8);

    auto permuteTokensOut = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeTokenPermuteWithRoutingMap, INPUT(tokens, indices, prob, 8, false),
                        OUTPUT(permuteTokensOut, probOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

namespace {
aclTensor* CreateL0AclTensor(const std::vector<int64_t>& shape, aclDataType dtype)
{
    static int64_t data[4096] = {0};
    return aclCreateTensor(shape.data(), shape.size(), dtype, nullptr, 0, ACL_FORMAT_ND, shape.data(), shape.size(),
                           data);
}

class l2_moe_token_permute_with_routing_map_l0_test : public testing::Test {
protected:
    l2_moe_token_permute_with_routing_map_l0_test()
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

TEST_F(l2_moe_token_permute_with_routing_map_l0_test, MoeTokenPermuteWithRoutingMap_l0_success_without_prob)
{
    auto tokens = CreateL0AclTensor({1, 64}, ACL_FLOAT);
    auto routingMap = CreateL0AclTensor({8, 1}, ACL_BOOL);
    auto result = l0op::MoeTokenPermuteWithRoutingMap(tokens, routingMap, nullptr, 8, false, exe);
    EXPECT_NE(result[0], nullptr);
    EXPECT_NE(result[1], nullptr);
    EXPECT_NE(result[2], nullptr);
}

TEST_F(l2_moe_token_permute_with_routing_map_l0_test, MoeTokenPermuteWithRoutingMap_l0_success_with_prob)
{
    auto tokens = CreateL0AclTensor({1, 64}, ACL_FLOAT);
    auto routingMap = CreateL0AclTensor({8, 1}, ACL_BOOL);
    auto probs = CreateL0AclTensor({8, 1}, ACL_FLOAT);
    auto result = l0op::MoeTokenPermuteWithRoutingMap(tokens, routingMap, probs, 8, false, exe);
    EXPECT_NE(result[0], nullptr);
    EXPECT_NE(result[1], nullptr);
    EXPECT_NE(result[2], nullptr);
}

TEST_F(l2_moe_token_permute_with_routing_map_l0_test, MoeTokenPermuteWithRoutingMap_l0_drop_and_pad_non_regbase)
{
    auto tokens = CreateL0AclTensor({1, 64}, ACL_FLOAT);
    auto routingMap = CreateL0AclTensor({8, 1}, ACL_BOOL);
    auto result = l0op::MoeTokenPermuteWithRoutingMap(tokens, routingMap, nullptr, 8, true, exe);
    EXPECT_NE(result[0], nullptr);
    EXPECT_NE(result[1], nullptr);
    EXPECT_NE(result[2], nullptr);
}

TEST_F(l2_moe_token_permute_with_routing_map_l0_test, MoeTokenPermuteWithRoutingMap_l0_align_num_zero)
{
    auto tokens = CreateL0AclTensor({0, 64}, ACL_FLOAT);
    auto routingMap = CreateL0AclTensor({8, 0}, ACL_BOOL);
    auto result = l0op::MoeTokenPermuteWithRoutingMap(tokens, routingMap, nullptr, 0, false, exe);
    EXPECT_NE(result[0], nullptr);
    EXPECT_NE(result[1], nullptr);
    EXPECT_NE(result[2], nullptr);
}
} // namespace
