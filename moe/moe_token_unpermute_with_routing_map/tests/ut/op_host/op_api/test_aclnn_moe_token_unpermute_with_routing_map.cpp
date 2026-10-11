/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file test_aclnn_moe_token_unpermute_with_routing_map.cpp
 * \brief 测试 aclnnMoeTokenUnpermuteWithRoutingMap 算子
 */

#include <vector>
#include <array>
#include <float.h>
#include <gmock/gmock.h>
#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_moe_token_unpermute_with_routing_map.h"
#include "opdev/platform.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_moe_token_unpermute_with_routing_map_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_moe_token_unpermute_with_routing_map_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_moe_token_unpermute_with_routing_map_test TearDown" << endl;
    }
};

// 基础功能测试：有probs，paddedMode = false
TEST_F(l2_moe_token_unpermute_with_routing_map_test, Ascend910B2_moe_token_unpermute_with_routingmap_probs_no_padded)
{
    // 张量形状定义
    const int64_t tokens_num = 4;                                // 令牌数量
    const int64_t experts_num = 2;                               // 专家数量
    const int64_t topK_num = 2;                                  // 每个令牌选择的专家数
    const int64_t hidden_size = 64;                              // 隐藏层大小
    const int64_t total_permuted_tokens = tokens_num * topK_num; // 置换后的令牌总数

    // 输入张量定义
    auto permutedTokens =
        TensorDesc({total_permuted_tokens, hidden_size}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices =
        TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND).ValueRange(int64_t(0), total_permuted_tokens - 1);

    // routingMapOptional: 形状 (tokens_num, experts_num), INT8 类型
    auto routingMapOptional = TensorDesc({tokens_num, experts_num}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1);

    // probsOptional: 形状与 routingMapOptional 相同
    auto probsOptional = TensorDesc({tokens_num, experts_num}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);

    // 输出张量
    auto unpermutedTokens = TensorDesc({tokens_num, hidden_size}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outIndex = TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND);
    auto permuteTokenId = TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND);
    auto permuteProbs = TensorDesc({total_permuted_tokens}, ACL_FLOAT, ACL_FORMAT_ND);

    // 创建形状数据
    std::vector<int64_t> shape_data = {tokens_num, hidden_size};
    // 使用工厂函数创建 aclIntArray
    aclIntArray* restoreShape = aclCreateIntArray(shape_data.data(), shape_data.size());

    // paddedMode 设置为 false
    auto ut = OP_API_UT(aclnnMoeTokenUnpermuteWithRoutingMap,
                        INPUT(permutedTokens, sortedIndices, routingMapOptional, probsOptional, false, restoreShape),
                        OUTPUT(unpermutedTokens, outIndex, permuteTokenId, permuteProbs));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// routingMapOptional 为 bool 类型的测试
TEST_F(l2_moe_token_unpermute_with_routing_map_test, Ascend910B2_moe_token_unpermute_with_routingmap_bool_type)
{
    const int64_t tokens_num = 2;
    const int64_t experts_num = 2;
    const int64_t topK_num = 2;
    const int64_t hidden_size = 128;
    const int64_t total_permuted_tokens = tokens_num * topK_num;

    auto permutedTokens = TensorDesc({total_permuted_tokens, hidden_size}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto sortedIndices =
        TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND).ValueRange(int64_t(0), total_permuted_tokens - 1);

    // routingMapOptional: bool 类型
    auto routingMapOptional = TensorDesc({tokens_num, experts_num}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 1);
    auto probsOptional = TensorDesc({tokens_num, experts_num}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 0.9);

    auto unpermutedTokens = TensorDesc({tokens_num, hidden_size}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outIndex = TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND);
    auto permuteTokenId = TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND);
    auto permuteProbs = TensorDesc({total_permuted_tokens}, ACL_FLOAT, ACL_FORMAT_ND);

    // 创建形状数据
    std::vector<int64_t> shape_data = {tokens_num, hidden_size};
    // 使用工厂函数创建 aclIntArray
    aclIntArray* restoreShape = aclCreateIntArray(shape_data.data(), shape_data.size());

    auto ut = OP_API_UT(aclnnMoeTokenUnpermuteWithRoutingMap,
                        INPUT(permutedTokens, sortedIndices, routingMapOptional, probsOptional, false, restoreShape),
                        OUTPUT(unpermutedTokens, outIndex, permuteTokenId, permuteProbs));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 边界情况测试：routingMap 全为 0
TEST_F(l2_moe_token_unpermute_with_routing_map_test, Ascend910B2_moe_token_unpermute_with_routingmap_all_zero)
{
    const int64_t tokens_num = 2;
    const int64_t experts_num = 2;
    const int64_t topK_num = 1;
    const int64_t hidden_size = 32;
    const int64_t total_permuted_tokens = tokens_num * topK_num;

    auto permutedTokens = TensorDesc({total_permuted_tokens, hidden_size}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sortedIndices =
        TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND).ValueRange(int64_t(0), total_permuted_tokens - 1);

    // routingMap 全为 0
    auto routingMapOptional = TensorDesc({tokens_num, experts_num}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 0);
    auto probsOptional = TensorDesc({tokens_num, experts_num}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(double(0), 0.5);

    auto unpermutedTokens = TensorDesc({tokens_num, hidden_size}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outIndex = TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND);
    auto permuteTokenId = TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND);
    auto permuteProbs = TensorDesc({total_permuted_tokens}, ACL_FLOAT, ACL_FORMAT_ND);

    // 创建形状数据
    std::vector<int64_t> shape_data = {tokens_num, hidden_size};
    // 使用工厂函数创建 aclIntArray
    aclIntArray* restoreShape = aclCreateIntArray(shape_data.data(), shape_data.size());

    auto ut = OP_API_UT(aclnnMoeTokenUnpermuteWithRoutingMap,
                        INPUT(permutedTokens, sortedIndices, routingMapOptional, probsOptional, false, restoreShape),
                        OUTPUT(unpermutedTokens, outIndex, permuteTokenId, permuteProbs));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 测试不同数据类型的permutedTokens（FP16）
TEST_F(l2_moe_token_unpermute_with_routing_map_test, Ascend910B2_moe_token_unpermute_with_routingmap_fp16)
{
    const int64_t tokens_num = 2;
    const int64_t experts_num = 2;
    const int64_t topK_num = 2;
    const int64_t hidden_size = 64;
    const int64_t total_permuted_tokens = tokens_num * topK_num;

    // 使用FP16数据类型
    auto permutedTokens =
        TensorDesc({total_permuted_tokens, hidden_size}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices =
        TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND).ValueRange(int64_t(0), total_permuted_tokens - 1);

    auto routingMapOptional = TensorDesc({tokens_num, experts_num}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1);
    auto probsOptional = TensorDesc({tokens_num, experts_num}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);

    auto unpermutedTokens = TensorDesc({tokens_num, hidden_size}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto outIndex = TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND);
    auto permuteTokenId = TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND);
    auto permuteProbs = TensorDesc({total_permuted_tokens}, ACL_FLOAT16, ACL_FORMAT_ND);

    // 创建形状数据
    std::vector<int64_t> shape_data = {tokens_num, hidden_size};
    // 使用工厂函数创建 aclIntArray
    aclIntArray* restoreShape = aclCreateIntArray(shape_data.data(), shape_data.size());

    auto ut = OP_API_UT(aclnnMoeTokenUnpermuteWithRoutingMap,
                        INPUT(permutedTokens, sortedIndices, routingMapOptional, probsOptional, false, restoreShape),
                        OUTPUT(unpermutedTokens, outIndex, permuteTokenId, permuteProbs));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 大规模令牌测试
TEST_F(l2_moe_token_unpermute_with_routing_map_test, Ascend910B2_moe_token_unpermute_with_routingmap_large_scale)
{
    const int64_t tokens_num = 16; // 更多令牌
    const int64_t experts_num = 8; // 更多专家
    const int64_t topK_num = 2;
    const int64_t hidden_size = 512; // 更大隐藏层
    const int64_t total_permuted_tokens = tokens_num * topK_num;

    auto permutedTokens =
        TensorDesc({total_permuted_tokens, hidden_size}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices =
        TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND).ValueRange(int64_t(0), total_permuted_tokens - 1);

    auto routingMapOptional = TensorDesc({tokens_num, experts_num}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1);
    auto probsOptional = TensorDesc({tokens_num, experts_num}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);

    auto unpermutedTokens = TensorDesc({tokens_num, hidden_size}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outIndex = TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND);
    auto permuteTokenId = TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND);
    auto permuteProbs = TensorDesc({total_permuted_tokens}, ACL_FLOAT, ACL_FORMAT_ND);

    // 创建形状数据
    std::vector<int64_t> shape_data = {tokens_num, hidden_size};
    // 使用工厂函数创建 aclIntArray
    aclIntArray* restoreShape = aclCreateIntArray(shape_data.data(), shape_data.size());

    auto ut = OP_API_UT(aclnnMoeTokenUnpermuteWithRoutingMap,
                        INPUT(permutedTokens, sortedIndices, routingMapOptional, probsOptional, false, restoreShape),
                        OUTPUT(unpermutedTokens, outIndex, permuteTokenId, permuteProbs));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 测试 restoreShapeOptional 为 nullptr 的情况
TEST_F(l2_moe_token_unpermute_with_routing_map_test, Ascend910B2_moe_token_unpermute_with_routingmap_null_restore_shape)
{
    const int64_t tokens_num = 2;
    const int64_t experts_num = 2;
    const int64_t topK_num = 2;
    const int64_t hidden_size = 64;
    const int64_t total_permuted_tokens = tokens_num * topK_num;

    auto permutedTokens = TensorDesc({total_permuted_tokens, hidden_size}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto sortedIndices =
        TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND).ValueRange(int64_t(0), total_permuted_tokens - 1);

    auto routingMapOptional = TensorDesc({tokens_num, experts_num}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1);
    auto probsOptional = TensorDesc({tokens_num, experts_num}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);

    auto unpermutedTokens = TensorDesc({tokens_num, hidden_size}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outIndex = TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND);
    auto permuteTokenId = TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND);
    auto permuteProbs = TensorDesc({total_permuted_tokens}, ACL_FLOAT, ACL_FORMAT_ND);

    // restoreShapeOptional 传 nullptr
    auto ut =
        OP_API_UT(aclnnMoeTokenUnpermuteWithRoutingMap,
                  INPUT(permutedTokens, sortedIndices, routingMapOptional, probsOptional, false, (aclIntArray*)nullptr),
                  OUTPUT(unpermutedTokens, outIndex, permuteTokenId, permuteProbs));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_moe_token_unpermute_with_routing_map_test,
       Ascend910B2_moe_token_unpermute_with_routingmap_probs_restore_both_null)
{
    const int64_t tokens_num = 2;
    const int64_t experts_num = 2;
    const int64_t topK_num = 2;
    const int64_t hidden_size = 64;
    const int64_t total_permuted_tokens = tokens_num * topK_num;

    auto permutedTokens = TensorDesc({total_permuted_tokens, hidden_size}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sortedIndices = TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND);
    auto routingMapOptional = TensorDesc({tokens_num, experts_num}, ACL_INT8, ACL_FORMAT_ND);
    auto unpermutedTokens = TensorDesc({tokens_num, hidden_size}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outIndex = TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND);
    auto permuteTokenId = TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND);
    auto permuteProbs = TensorDesc({total_permuted_tokens}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMoeTokenUnpermuteWithRoutingMap,
        INPUT(permutedTokens, sortedIndices, routingMapOptional, (aclTensor*)nullptr, false, (aclIntArray*)nullptr),
        OUTPUT(unpermutedTokens, outIndex, permuteTokenId, permuteProbs));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_moe_token_unpermute_with_routing_map_test,
       Ascend910B2_moe_token_unpermute_with_routingmap_probs_null_restore_only)
{
    const int64_t tokens_num = 2;
    const int64_t experts_num = 2;
    const int64_t topK_num = 2;
    const int64_t hidden_size = 64;
    const int64_t total_permuted_tokens = tokens_num * topK_num;

    auto permutedTokens = TensorDesc({total_permuted_tokens, hidden_size}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto sortedIndices =
        TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND).ValueRange(int64_t(0), total_permuted_tokens - 1);
    auto routingMapOptional = TensorDesc({tokens_num, experts_num}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 1);
    auto unpermutedTokens = TensorDesc({tokens_num, hidden_size}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outIndex = TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND);
    auto permuteTokenId = TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND);
    auto permuteProbs = TensorDesc({total_permuted_tokens}, ACL_FLOAT, ACL_FORMAT_ND);

    std::vector<int64_t> shape_data = {tokens_num, hidden_size};
    aclIntArray* restoreShape = aclCreateIntArray(shape_data.data(), shape_data.size());

    auto ut =
        OP_API_UT(aclnnMoeTokenUnpermuteWithRoutingMap,
                  INPUT(permutedTokens, sortedIndices, routingMapOptional, (aclTensor*)nullptr, false, restoreShape),
                  OUTPUT(unpermutedTokens, outIndex, permuteTokenId, permuteProbs));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_moe_token_unpermute_with_routing_map_test,
       Ascend910B2_moe_token_unpermute_with_routingmap_empty_sorted_indices)
{
    const int64_t tokens_num = 2;
    const int64_t hidden_size = 64;

    auto permutedTokens = TensorDesc({0, hidden_size}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sortedIndices = TensorDesc({0}, ACL_INT32, ACL_FORMAT_ND);
    auto routingMapOptional = TensorDesc({tokens_num, 2}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1);
    auto probsOptional = TensorDesc({tokens_num, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto unpermutedTokens = TensorDesc({tokens_num, hidden_size}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outIndex = TensorDesc({0}, ACL_INT32, ACL_FORMAT_ND);
    auto permuteTokenId = TensorDesc({0}, ACL_INT32, ACL_FORMAT_ND);
    auto permuteProbs = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);

    std::vector<int64_t> shape_data = {tokens_num, hidden_size};
    aclIntArray* restoreShape = aclCreateIntArray(shape_data.data(), shape_data.size());

    auto ut = OP_API_UT(aclnnMoeTokenUnpermuteWithRoutingMap,
                        INPUT(permutedTokens, sortedIndices, routingMapOptional, probsOptional, false, restoreShape),
                        OUTPUT(unpermutedTokens, outIndex, permuteTokenId, permuteProbs));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_moe_token_unpermute_with_routing_map_test, Ascend910B2_moe_token_unpermute_with_routingmap_padded_mode)
{
    const int64_t tokens_num = 4;
    const int64_t experts_num = 2;
    const int64_t topK_num = 2;
    const int64_t hidden_size = 64;
    const int64_t total_permuted_tokens = tokens_num * topK_num;

    auto permutedTokens = TensorDesc({total_permuted_tokens, hidden_size}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto sortedIndices =
        TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND).ValueRange(int64_t(0), total_permuted_tokens - 1);
    auto routingMapOptional = TensorDesc({tokens_num, experts_num}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1);
    auto probsOptional = TensorDesc({tokens_num, experts_num}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto unpermutedTokens = TensorDesc({tokens_num, hidden_size}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outIndex = TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND);
    auto permuteTokenId = TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND);
    auto permuteProbs = TensorDesc({total_permuted_tokens}, ACL_FLOAT, ACL_FORMAT_ND);

    std::vector<int64_t> shape_data = {tokens_num, hidden_size};
    aclIntArray* restoreShape = aclCreateIntArray(shape_data.data(), shape_data.size());

    auto ut = OP_API_UT(aclnnMoeTokenUnpermuteWithRoutingMap,
                        INPUT(permutedTokens, sortedIndices, routingMapOptional, probsOptional, true, restoreShape),
                        OUTPUT(unpermutedTokens, outIndex, permuteTokenId, permuteProbs));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_moe_token_unpermute_with_routing_map_test, Ascend910B2_moe_token_unpermute_with_routingmap_bf16_mixed_probs)
{
    const int64_t tokens_num = 2;
    const int64_t experts_num = 2;
    const int64_t topK_num = 2;
    const int64_t hidden_size = 64;
    const int64_t total_permuted_tokens = tokens_num * topK_num;

    auto permutedTokens = TensorDesc({total_permuted_tokens, hidden_size}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto sortedIndices =
        TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND).ValueRange(int64_t(0), total_permuted_tokens - 1);
    auto routingMapOptional = TensorDesc({tokens_num, experts_num}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1);
    auto probsOptional = TensorDesc({tokens_num, experts_num}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto unpermutedTokens = TensorDesc({tokens_num, hidden_size}, ACL_BF16, ACL_FORMAT_ND);
    auto outIndex = TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND);
    auto permuteTokenId = TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND);
    auto permuteProbs = TensorDesc({total_permuted_tokens}, ACL_FLOAT, ACL_FORMAT_ND);

    std::vector<int64_t> shape_data = {tokens_num, hidden_size};
    aclIntArray* restoreShape = aclCreateIntArray(shape_data.data(), shape_data.size());

    auto ut = OP_API_UT(aclnnMoeTokenUnpermuteWithRoutingMap,
                        INPUT(permutedTokens, sortedIndices, routingMapOptional, probsOptional, true, restoreShape),
                        OUTPUT(unpermutedTokens, outIndex, permuteTokenId, permuteProbs));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_moe_token_unpermute_with_routing_map_test,
       Ascend910B2_moe_token_unpermute_with_routingmap_invalid_routing_map_dim)
{
    const int64_t tokens_num = 2;
    const int64_t experts_num = 2;
    const int64_t topK_num = 2;
    const int64_t hidden_size = 64;
    const int64_t total_permuted_tokens = tokens_num * topK_num;

    auto permutedTokens = TensorDesc({total_permuted_tokens, hidden_size}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sortedIndices = TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND);
    auto routingMapOptional = TensorDesc({tokens_num}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1);
    auto probsOptional = TensorDesc({tokens_num, experts_num}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto unpermutedTokens = TensorDesc({tokens_num, hidden_size}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outIndex = TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND);
    auto permuteTokenId = TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND);
    auto permuteProbs = TensorDesc({total_permuted_tokens}, ACL_FLOAT, ACL_FORMAT_ND);

    std::vector<int64_t> shape_data = {tokens_num, hidden_size};
    aclIntArray* restoreShape = aclCreateIntArray(shape_data.data(), shape_data.size());

    auto ut = OP_API_UT(aclnnMoeTokenUnpermuteWithRoutingMap,
                        INPUT(permutedTokens, sortedIndices, routingMapOptional, probsOptional, false, restoreShape),
                        OUTPUT(unpermutedTokens, outIndex, permuteTokenId, permuteProbs));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_moe_token_unpermute_with_routing_map_test,
       Ascend910B2_moe_token_unpermute_with_routingmap_invalid_routing_map_dtype)
{
    const int64_t tokens_num = 2;
    const int64_t experts_num = 2;
    const int64_t topK_num = 2;
    const int64_t hidden_size = 64;
    const int64_t total_permuted_tokens = tokens_num * topK_num;

    auto permutedTokens = TensorDesc({total_permuted_tokens, hidden_size}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sortedIndices = TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND);
    auto routingMapOptional = TensorDesc({tokens_num, experts_num}, ACL_INT32, ACL_FORMAT_ND);
    auto probsOptional = TensorDesc({tokens_num, experts_num}, ACL_FLOAT, ACL_FORMAT_ND);
    auto unpermutedTokens = TensorDesc({tokens_num, hidden_size}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outIndex = TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND);
    auto permuteTokenId = TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND);
    auto permuteProbs = TensorDesc({total_permuted_tokens}, ACL_FLOAT, ACL_FORMAT_ND);

    std::vector<int64_t> shape_data = {tokens_num, hidden_size};
    aclIntArray* restoreShape = aclCreateIntArray(shape_data.data(), shape_data.size());

    auto ut = OP_API_UT(aclnnMoeTokenUnpermuteWithRoutingMap,
                        INPUT(permutedTokens, sortedIndices, routingMapOptional, probsOptional, false, restoreShape),
                        OUTPUT(unpermutedTokens, outIndex, permuteTokenId, permuteProbs));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_moe_token_unpermute_with_routing_map_test,
       Ascend910B2_moe_token_unpermute_with_routingmap_padded_empty_hidden)
{
    const int64_t tokens_num = 2;
    const int64_t experts_num = 2;
    const int64_t topK_num = 2;
    const int64_t total_permuted_tokens = tokens_num * topK_num;

    auto permutedTokens = TensorDesc({total_permuted_tokens, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sortedIndices =
        TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND).ValueRange(int64_t(0), total_permuted_tokens - 1);
    auto routingMapOptional = TensorDesc({tokens_num, experts_num}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1);
    auto probsOptional = TensorDesc({tokens_num, experts_num}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto unpermutedTokens = TensorDesc({tokens_num, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outIndex = TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND);
    auto permuteTokenId = TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND);
    auto permuteProbs = TensorDesc({total_permuted_tokens}, ACL_FLOAT, ACL_FORMAT_ND);

    std::vector<int64_t> shape_data = {tokens_num, 0};
    aclIntArray* restoreShape = aclCreateIntArray(shape_data.data(), shape_data.size());

    auto ut = OP_API_UT(aclnnMoeTokenUnpermuteWithRoutingMap,
                        INPUT(permutedTokens, sortedIndices, routingMapOptional, probsOptional, true, restoreShape),
                        OUTPUT(unpermutedTokens, outIndex, permuteTokenId, permuteProbs));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_moe_token_unpermute_with_routing_map_test,
       Ascend910B2_moe_token_unpermute_with_routingmap_restore_shape_invalid_size)
{
    const int64_t tokens_num = 2;
    const int64_t experts_num = 2;
    const int64_t topK_num = 2;
    const int64_t hidden_size = 64;
    const int64_t total_permuted_tokens = tokens_num * topK_num;

    auto permutedTokens = TensorDesc({total_permuted_tokens, hidden_size}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sortedIndices =
        TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND).ValueRange(int64_t(0), total_permuted_tokens - 1);
    auto routingMapOptional = TensorDesc({tokens_num, experts_num}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1);
    auto unpermutedTokens = TensorDesc({tokens_num, hidden_size}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outIndex = TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND);
    auto permuteTokenId = TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND);
    auto permuteProbs = TensorDesc({total_permuted_tokens}, ACL_FLOAT, ACL_FORMAT_ND);

    std::vector<int64_t> shape_data = {tokens_num};
    aclIntArray* restoreShape = aclCreateIntArray(shape_data.data(), shape_data.size());

    auto ut =
        OP_API_UT(aclnnMoeTokenUnpermuteWithRoutingMap,
                  INPUT(permutedTokens, sortedIndices, routingMapOptional, (aclTensor*)nullptr, false, restoreShape),
                  OUTPUT(unpermutedTokens, outIndex, permuteTokenId, permuteProbs));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_INNER_NULLPTR);
}

TEST_F(l2_moe_token_unpermute_with_routing_map_test,
       Ascend910B2_moe_token_unpermute_with_routingmap_private_format_warning)
{
    const int64_t tokens_num = 2;
    const int64_t experts_num = 2;
    const int64_t topK_num = 2;
    const int64_t hidden_size = 64;
    const int64_t total_permuted_tokens = tokens_num * topK_num;

    auto permutedTokens =
        TensorDesc({total_permuted_tokens, hidden_size}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ).ValueRange(-5, 5);
    auto sortedIndices = TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_FRACTAL_NZ)
                             .ValueRange(int64_t(0), total_permuted_tokens - 1);
    auto routingMapOptional = TensorDesc({tokens_num, experts_num}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(0, 1);
    auto probsOptional = TensorDesc({tokens_num, experts_num}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ).ValueRange(0, 1);
    auto unpermutedTokens = TensorDesc({tokens_num, hidden_size}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outIndex = TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND);
    auto permuteTokenId = TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND);
    auto permuteProbs = TensorDesc({total_permuted_tokens}, ACL_FLOAT, ACL_FORMAT_ND);

    std::vector<int64_t> shape_data = {tokens_num, hidden_size};
    aclIntArray* restoreShape = aclCreateIntArray(shape_data.data(), shape_data.size());

    auto ut = OP_API_UT(aclnnMoeTokenUnpermuteWithRoutingMap,
                        INPUT(permutedTokens, sortedIndices, routingMapOptional, probsOptional, false, restoreShape),
                        OUTPUT(unpermutedTokens, outIndex, permuteTokenId, permuteProbs));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

namespace {
struct PlatformGuard {
    explicit PlatformGuard(op::SocVersion target)
    {
        op::SetPlatformSocVersion(target);
    }
    ~PlatformGuard()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    }
};
} // namespace

TEST_F(l2_moe_token_unpermute_with_routing_map_test, Ascend910_93_moe_token_unpermute_with_routingmap_padded_fp16)
{
    PlatformGuard guard(op::SocVersion::ASCEND910_93);

    const int64_t tokens_num = 4;
    const int64_t experts_num = 2;
    const int64_t topK_num = 2;
    const int64_t hidden_size = 64;
    const int64_t total_permuted_tokens = tokens_num * topK_num;

    auto permutedTokens =
        TensorDesc({total_permuted_tokens, hidden_size}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto sortedIndices =
        TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND).ValueRange(int64_t(0), total_permuted_tokens - 1);
    auto routingMapOptional = TensorDesc({tokens_num, experts_num}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1);
    auto probsOptional = TensorDesc({tokens_num, experts_num}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto unpermutedTokens = TensorDesc({tokens_num, hidden_size}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto outIndex = TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND);
    auto permuteTokenId = TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND);
    auto permuteProbs = TensorDesc({total_permuted_tokens}, ACL_FLOAT16, ACL_FORMAT_ND);

    std::vector<int64_t> shape_data = {tokens_num, hidden_size};
    aclIntArray* restoreShape = aclCreateIntArray(shape_data.data(), shape_data.size());

    auto ut = OP_API_UT(aclnnMoeTokenUnpermuteWithRoutingMap,
                        INPUT(permutedTokens, sortedIndices, routingMapOptional, probsOptional, true, restoreShape),
                        OUTPUT(unpermutedTokens, outIndex, permuteTokenId, permuteProbs));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_moe_token_unpermute_with_routing_map_test, Ascend910B2_moe_token_unpermute_with_routingmap_execute)
{
    const int64_t tokens_num = 2;
    const int64_t experts_num = 2;
    const int64_t topK_num = 2;
    const int64_t hidden_size = 64;
    const int64_t total_permuted_tokens = tokens_num * topK_num;

    auto permutedTokens = TensorDesc({total_permuted_tokens, hidden_size}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto sortedIndices =
        TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND).ValueRange(int64_t(0), total_permuted_tokens - 1);
    auto routingMapOptional = TensorDesc({tokens_num, experts_num}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1);
    auto probsOptional = TensorDesc({tokens_num, experts_num}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto unpermutedTokens = TensorDesc({tokens_num, hidden_size}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outIndex = TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND);
    auto permuteTokenId = TensorDesc({total_permuted_tokens}, ACL_INT32, ACL_FORMAT_ND);
    auto permuteProbs = TensorDesc({total_permuted_tokens}, ACL_FLOAT, ACL_FORMAT_ND);

    std::vector<int64_t> shape_data = {tokens_num, hidden_size};
    aclIntArray* restoreShape = aclCreateIntArray(shape_data.data(), shape_data.size());

    auto ut = OP_API_UT(aclnnMoeTokenUnpermuteWithRoutingMap,
                        INPUT(permutedTokens, sortedIndices, routingMapOptional, probsOptional, false, restoreShape),
                        OUTPUT(unpermutedTokens, outIndex, permuteTokenId, permuteProbs));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

    aclnnStatus execRet = aclnnMoeTokenUnpermuteWithRoutingMap(nullptr, workspaceSize, executor, nullptr);
    EXPECT_THAT(execRet, testing::AnyOf(testing::Eq(ACLNN_SUCCESS), testing::Eq(ACLNN_ERR_PARAM_NULLPTR),
                                        testing::Eq(ACLNN_ERR_PARAM_INVALID)));
}
