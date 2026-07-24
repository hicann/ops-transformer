/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN " AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"
#include "aclnn_mega_moe.h"

using namespace op;
using namespace std;

class AclnnMegaMoeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
        std::cout << "MegaMoe AclnnMegaMoeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
        std::cout << "MegaMoe AclnnMegaMoeTest TearDown" << std::endl;
    }
};

TEST_F(AclnnMegaMoeTest, ascend950_nullptr_context)
{
    auto x_desc = TensorDesc({128, 4096}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto topk_ids_desc = TensorDesc({128, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 3);
    auto topk_weights_desc = TensorDesc({128, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);

    TensorListDesc weight1_descs(4, TensorDesc({2048, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1));
    TensorListDesc weight2_descs(4, TensorDesc({4096, 1024}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1));
    TensorListDesc weight_scales1_descs(4, TensorDesc({2048}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2));
    TensorListDesc weight_scales2_descs(4, TensorDesc({4096}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2));

    auto y_desc = TensorDesc({128, 4096}, ACL_BF16, ACL_FORMAT_ND);
    auto expert_token_nums_desc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMegaMoe,
                        INPUT(nullptr, x_desc, topk_ids_desc, topk_weights_desc, weight1_descs, weight2_descs,
                              weight_scales1_descs, weight_scales2_descs, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr, 16, 4, 2097152, 0, 0, 0, 0, "", 0, "swiglu",
                              std::numeric_limits<float>::max(), 0, 2, 0),
                        OUTPUT(y_desc, expert_token_nums_desc));

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AclnnMegaMoeTest, ascend950_nullptr_x)
{
    auto context_desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
    auto topk_ids_desc = TensorDesc({128, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 3);
    auto topk_weights_desc = TensorDesc({128, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);

    TensorListDesc weight1_descs(4, TensorDesc({2048, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1));
    TensorListDesc weight2_descs(4, TensorDesc({4096, 1024}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1));
    TensorListDesc weight_scales1_descs(4, TensorDesc({2048}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2));
    TensorListDesc weight_scales2_descs(4, TensorDesc({4096}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2));

    auto y_desc = TensorDesc({128, 4096}, ACL_BF16, ACL_FORMAT_ND);
    auto expert_token_nums_desc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMegaMoe,
                        INPUT(context_desc, nullptr, topk_ids_desc, topk_weights_desc, weight1_descs, weight2_descs,
                              weight_scales1_descs, weight_scales2_descs, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr, 16, 4, 2097152, 0, 0, 0, 0, "", 0, "swiglu",
                              std::numeric_limits<float>::max(), 0, 2, 0),
                        OUTPUT(y_desc, expert_token_nums_desc));

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AclnnMegaMoeTest, ascend950_nullptr_topk_ids)
{
    auto context_desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
    auto x_desc = TensorDesc({128, 4096}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto topk_weights_desc = TensorDesc({128, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);

    TensorListDesc weight1_descs(4, TensorDesc({2048, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1));
    TensorListDesc weight2_descs(4, TensorDesc({4096, 1024}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1));
    TensorListDesc weight_scales1_descs(4, TensorDesc({2048}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2));
    TensorListDesc weight_scales2_descs(4, TensorDesc({4096}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2));

    auto y_desc = TensorDesc({128, 4096}, ACL_BF16, ACL_FORMAT_ND);
    auto expert_token_nums_desc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMegaMoe,
                        INPUT(context_desc, x_desc, nullptr, topk_weights_desc, weight1_descs, weight2_descs,
                              weight_scales1_descs, weight_scales2_descs, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr, 16, 4, 2097152, 0, 0, 0, 0, "", 0, "swiglu",
                              std::numeric_limits<float>::max(), 0, 2, 0),
                        OUTPUT(y_desc, expert_token_nums_desc));

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AclnnMegaMoeTest, ascend950_nullptr_topk_weights)
{
    auto context_desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
    auto x_desc = TensorDesc({128, 4096}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto topk_ids_desc = TensorDesc({128, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 3);

    TensorListDesc weight1_descs(4, TensorDesc({2048, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1));
    TensorListDesc weight2_descs(4, TensorDesc({4096, 1024}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1));
    TensorListDesc weight_scales1_descs(4, TensorDesc({2048}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2));
    TensorListDesc weight_scales2_descs(4, TensorDesc({4096}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2));

    auto y_desc = TensorDesc({128, 4096}, ACL_BF16, ACL_FORMAT_ND);
    auto expert_token_nums_desc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMegaMoe,
                        INPUT(context_desc, x_desc, topk_ids_desc, nullptr, weight1_descs, weight2_descs,
                              weight_scales1_descs, weight_scales2_descs, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr, 16, 4, 2097152, 0, 0, 0, 0, "", 0, "swiglu",
                              std::numeric_limits<float>::max(), 0, 2, 0),
                        OUTPUT(y_desc, expert_token_nums_desc));

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AclnnMegaMoeTest, ascend950_nullptr_weight1)
{
    auto context_desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
    auto x_desc = TensorDesc({128, 4096}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto topk_ids_desc = TensorDesc({128, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 3);
    auto topk_weights_desc = TensorDesc({128, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);

    TensorListDesc weight2_descs(4, TensorDesc({4096, 1024}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1));
    TensorListDesc weight_scales1_descs(4, TensorDesc({2048}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2));
    TensorListDesc weight_scales2_descs(4, TensorDesc({4096}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2));

    auto y_desc = TensorDesc({128, 4096}, ACL_BF16, ACL_FORMAT_ND);
    auto expert_token_nums_desc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMegaMoe,
                        INPUT(context_desc, x_desc, topk_ids_desc, topk_weights_desc, nullptr, weight2_descs,
                              weight_scales1_descs, weight_scales2_descs, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr, 16, 4, 2097152, 0, 0, 0, 0, "", 0, "swiglu",
                              std::numeric_limits<float>::max(), 0, 2, 0),
                        OUTPUT(y_desc, expert_token_nums_desc));

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AclnnMegaMoeTest, ascend950_nullptr_weight2)
{
    auto context_desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
    auto x_desc = TensorDesc({128, 4096}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto topk_ids_desc = TensorDesc({128, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 3);
    auto topk_weights_desc = TensorDesc({128, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);

    TensorListDesc weight1_descs(4, TensorDesc({2048, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1));
    TensorListDesc weight_scales1_descs(4, TensorDesc({2048}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2));
    TensorListDesc weight_scales2_descs(4, TensorDesc({4096}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2));

    auto y_desc = TensorDesc({128, 4096}, ACL_BF16, ACL_FORMAT_ND);
    auto expert_token_nums_desc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMegaMoe,
                        INPUT(context_desc, x_desc, topk_ids_desc, topk_weights_desc, weight1_descs, nullptr,
                              weight_scales1_descs, weight_scales2_descs, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr, 16, 4, 2097152, 0, 0, 0, 0, "", 0, "swiglu",
                              std::numeric_limits<float>::max(), 0, 2, 0),
                        OUTPUT(y_desc, expert_token_nums_desc));

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AclnnMegaMoeTest, ascend950_nullptr_y_out)
{
    auto context_desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
    auto x_desc = TensorDesc({128, 4096}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto topk_ids_desc = TensorDesc({128, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 3);
    auto topk_weights_desc = TensorDesc({128, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);

    TensorListDesc weight1_descs(4, TensorDesc({2048, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1));
    TensorListDesc weight2_descs(4, TensorDesc({4096, 1024}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1));
    TensorListDesc weight_scales1_descs(4, TensorDesc({2048}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2));
    TensorListDesc weight_scales2_descs(4, TensorDesc({4096}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2));

    auto expert_token_nums_desc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMegaMoe,
                        INPUT(context_desc, x_desc, topk_ids_desc, topk_weights_desc, weight1_descs, weight2_descs,
                              weight_scales1_descs, weight_scales2_descs, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr, 16, 4, 2097152, 0, 0, 0, 0, "", 0, "swiglu",
                              std::numeric_limits<float>::max(), 0, 2, 0),
                        OUTPUT(nullptr, expert_token_nums_desc));

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AclnnMegaMoeTest, ascend950_nullptr_expert_token_nums_out)
{
    auto context_desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
    auto x_desc = TensorDesc({128, 4096}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto topk_ids_desc = TensorDesc({128, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 3);
    auto topk_weights_desc = TensorDesc({128, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);

    TensorListDesc weight1_descs(4, TensorDesc({2048, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1));
    TensorListDesc weight2_descs(4, TensorDesc({4096, 1024}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1));
    TensorListDesc weight_scales1_descs(4, TensorDesc({2048}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2));
    TensorListDesc weight_scales2_descs(4, TensorDesc({4096}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2));

    auto y_desc = TensorDesc({128, 4096}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMegaMoe,
                        INPUT(context_desc, x_desc, topk_ids_desc, topk_weights_desc, weight1_descs, weight2_descs,
                              weight_scales1_descs, weight_scales2_descs, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr, 16, 4, 2097152, 0, 0, 0, 0, "", 0, "swiglu",
                              std::numeric_limits<float>::max(), 0, 2, 0),
                        OUTPUT(y_desc, nullptr));

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}
