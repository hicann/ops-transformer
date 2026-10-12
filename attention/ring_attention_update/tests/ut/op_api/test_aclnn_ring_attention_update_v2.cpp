/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "../../../op_host/op_api/aclnn_ring_attention_update.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class RingAttentionUpdateApiTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "RingAttentionUpdateApiTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RingAttentionUpdateApiTest TearDown" << std::endl;
    }
};

TEST_F(RingAttentionUpdateApiTest, aclnn_ring_attention_update_v2_sbh_fp32)
{
    auto prev_attn_out = TensorDesc({1024, 2, 1536}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto prev_softmax_max = TensorDesc({2, 12, 1024, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto prev_softmax_sum = TensorDesc({2, 12, 1024, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto cur_attn_out = TensorDesc({1024, 2, 1536}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto cur_softmax_max = TensorDesc({2, 12, 1024, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto cur_softmax_sum = TensorDesc({2, 12, 1024, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    char input_layout[] = "SBH";
    char input_softmax_layout[] = "SBH";

    auto attn_out = TensorDesc({1024, 2, 1536}, ACL_FLOAT, ACL_FORMAT_ND);
    auto softmax_max = TensorDesc({2, 12, 1024, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto softmax_sum = TensorDesc({2, 12, 1024, 8}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRingAttentionUpdateV2,
                        INPUT(prev_attn_out, prev_softmax_max, prev_softmax_sum, cur_attn_out, cur_softmax_max,
                              cur_softmax_sum, nullptr, input_layout, input_softmax_layout),
                        OUTPUT(attn_out, softmax_max, softmax_sum));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(RingAttentionUpdateApiTest, aclnn_ring_attention_update_v2_sbh_fp16)
{
    auto prev_attn_out = TensorDesc({1024, 2, 1536}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto prev_softmax_max = TensorDesc({2, 12, 1024, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto prev_softmax_sum = TensorDesc({2, 12, 1024, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto cur_attn_out = TensorDesc({1024, 2, 1536}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto cur_softmax_max = TensorDesc({2, 12, 1024, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto cur_softmax_sum = TensorDesc({2, 12, 1024, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    char input_layout[] = "SBH";
    char input_softmax_layout[] = "SBH";

    auto attn_out = TensorDesc({1024, 2, 1536}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto softmax_max = TensorDesc({2, 12, 1024, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto softmax_sum = TensorDesc({2, 12, 1024, 8}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRingAttentionUpdateV2,
                        INPUT(prev_attn_out, prev_softmax_max, prev_softmax_sum, cur_attn_out, cur_softmax_max,
                              cur_softmax_sum, nullptr, input_layout, input_softmax_layout),
                        OUTPUT(attn_out, softmax_max, softmax_sum));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
