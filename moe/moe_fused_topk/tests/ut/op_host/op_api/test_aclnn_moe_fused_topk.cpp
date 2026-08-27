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
#include "../../../../op_host/op_api/aclnn_moe_fused_topk.h"
#include "opdev/platform.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace op;
using namespace std;

class l2_moe_fused_topk_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "moe_fused_topk_test Setup" << std::endl;
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    }
    static void TearDownTestCase()
    {
        std::cout << "moe_fused_topk_test TearDown" << std::endl;
    }
};

class l2_moe_fused_topk_ascend950_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "moe_fused_topk_ascend950_test Setup" << std::endl;
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    }
    static void TearDownTestCase()
    {
        std::cout << "moe_fused_topk_ascend950_test TearDown" << std::endl;
    }
};

TEST_F(l2_moe_fused_topk_ascend950_test, Ascend950_moe_fused_topk_fp16_with_mapping)
{
    auto x = TensorDesc({41, 136}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto addNum = TensorDesc({136}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto mappingNum = TensorDesc({136}, ACL_INT32, ACL_FORMAT_ND).ValueRange(1L, 64L);
    auto mappingTable = TensorDesc({136, 64}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0L, 1023L);
    auto y = TensorDesc({41, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto indices = TensorDesc({41, 8}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeFusedTopk, INPUT(x, addNum, mappingNum, mappingTable, 8, 4, 2, 8, 0, true, 1.0f, true),
                        OUTPUT(y, indices));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
    // OP_API_UT uses executor stubs, so workspace accounting is not materialized
    // here.  Successful graph construction is the host-API assertion; device
    // workspace and execution are covered by the Ascend950 ACLNN smoke test.
}

TEST_F(l2_moe_fused_topk_test, ascend910B2_test_moe_fused_topk_1)
{
    int64_t num_token = 256;
    int64_t expert_num = 256;
    int64_t max_mapping_num = 129;

    uint32_t group_num = 8;
    uint32_t group_topk = 2;
    uint32_t topk = 2;
    uint32_t topn = 2;
    uint32_t activate_type = 0;
    bool is_norm = true;
    float scale = 1.0;
    bool enable_expert_mapping = true;

    vector<int64_t> x_dims = {num_token, expert_num};
    vector<int64_t> add_num_dims = {expert_num};
    vector<int64_t> mapping_num_dims = {expert_num};
    vector<int64_t> mapping_table_dims = {expert_num, max_mapping_num};
    vector<int64_t> y_dims = {num_token, topk};
    vector<int64_t> indices_dims = {num_token, topk};

    auto x_desc = TensorDesc(x_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto add_num_desc = TensorDesc(add_num_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto mapping_num_desc = TensorDesc(mapping_num_dims, ACL_INT32, ACL_FORMAT_ND).ValueRange(1L, max_mapping_num);
    auto mapping_table_desc = TensorDesc(mapping_table_dims, ACL_INT32, ACL_FORMAT_ND).ValueRange(1L, max_mapping_num);

    auto y_desc = TensorDesc(y_dims, ACL_INT32, ACL_FORMAT_ND);
    auto indices_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeFusedTopk, // host api第二段接口名称
                        INPUT(x_desc, add_num_desc, mapping_num_desc, mapping_table_desc, group_num, group_topk, topn,
                              topk, activate_type, is_norm, scale, enable_expert_mapping), // host api输入
                        OUTPUT(y_desc, indices_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_moe_fused_topk_test, ascend910B2_test_moe_fused_topk_2)
{
    int64_t num_token = 256;
    int64_t expert_num = 256;
    int64_t max_mapping_num = 128;

    uint32_t group_num = 8;
    uint32_t group_topk = 2;
    uint32_t topk = 2;
    uint32_t topn = 2;
    uint32_t activate_type = 0;
    bool is_norm = true;
    float scale = 1.0;
    bool enable_expert_mapping = true;

    vector<int64_t> x_dims = {num_token, expert_num};
    vector<int64_t> add_num_dims = {expert_num};
    vector<int64_t> mapping_num_dims = {expert_num};
    vector<int64_t> mapping_table_dims = {expert_num + 1, max_mapping_num};
    vector<int64_t> y_dims = {num_token, topk};
    vector<int64_t> indices_dims = {num_token, topk};

    auto x_desc = TensorDesc(x_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto add_num_desc = TensorDesc(add_num_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto mapping_num_desc = TensorDesc(mapping_num_dims, ACL_INT32, ACL_FORMAT_ND).ValueRange(1L, max_mapping_num);
    auto mapping_table_desc = TensorDesc(mapping_table_dims, ACL_INT32, ACL_FORMAT_ND).ValueRange(1L, max_mapping_num);

    auto y_desc = TensorDesc(y_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto indices_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeFusedTopk, // host api第二段接口名称
                        INPUT(x_desc, add_num_desc, mapping_num_desc, mapping_table_desc, group_num, group_topk, topn,
                              topk, activate_type, is_norm, scale, enable_expert_mapping), // host api输入
                        OUTPUT(y_desc, indices_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_moe_fused_topk_test, ascend910B2_test_moe_fused_topk_3)
{
    int64_t num_token = 256;
    int64_t expert_num = 256;
    int64_t max_mapping_num = 128;

    uint32_t group_num = 0;
    uint32_t group_topk = 2;
    uint32_t topk = 2;
    uint32_t topn = 2;
    uint32_t activate_type = 0;
    bool is_norm = true;
    float scale = 1.0;
    bool enable_expert_mapping = true;

    vector<int64_t> x_dims = {num_token, expert_num};
    vector<int64_t> add_num_dims = {expert_num};
    vector<int64_t> mapping_num_dims = {expert_num};
    vector<int64_t> mapping_table_dims = {expert_num, max_mapping_num};
    vector<int64_t> y_dims = {num_token, topk};
    vector<int64_t> indices_dims = {num_token, topk};

    auto x_desc = TensorDesc(x_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto add_num_desc = TensorDesc(add_num_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto mapping_num_desc = TensorDesc(mapping_num_dims, ACL_INT32, ACL_FORMAT_ND).ValueRange(1L, max_mapping_num);
    auto mapping_table_desc = TensorDesc(mapping_table_dims, ACL_INT32, ACL_FORMAT_ND).ValueRange(1L, max_mapping_num);

    auto y_desc = TensorDesc(y_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto indices_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeFusedTopk, // host api第二段接口名称
                        INPUT(x_desc, add_num_desc, mapping_num_desc, mapping_table_desc, group_num, group_topk, topn,
                              topk, activate_type, is_norm, scale, enable_expert_mapping), // host api输入
                        OUTPUT(y_desc, indices_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_moe_fused_topk_test, ascend910B2_test_moe_fused_topk_4)
{
    int64_t num_token = 256;
    int64_t expert_num = 256;
    int64_t max_mapping_num = 128;

    uint32_t group_num = 4;
    uint32_t group_topk = 2;
    uint32_t topk = 2;
    uint32_t topn = 2;
    uint32_t activate_type = 0;
    bool is_norm = true;
    float scale = 1.0;
    bool enable_expert_mapping = true;

    vector<int64_t> x_dims = {num_token, expert_num};
    vector<int64_t> add_num_dims = {expert_num};
    vector<int64_t> mapping_num_dims = {expert_num};
    vector<int64_t> mapping_table_dims = {expert_num + 1, max_mapping_num};
    vector<int64_t> y_dims = {num_token, topk};
    vector<int64_t> indices_dims = {num_token, topk};

    auto x_desc = TensorDesc(x_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto add_num_desc = TensorDesc(add_num_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto mapping_num_desc = TensorDesc(mapping_num_dims, ACL_INT32, ACL_FORMAT_ND).ValueRange(1L, max_mapping_num);
    auto mapping_table_desc = TensorDesc(mapping_table_dims, ACL_INT32, ACL_FORMAT_ND).ValueRange(1L, max_mapping_num);

    auto y_desc = TensorDesc(y_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto indices_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeFusedTopk, // host api第二段接口名称
                        INPUT(x_desc, add_num_desc, mapping_num_desc, mapping_table_desc, group_num, group_topk, topn,
                              topk, activate_type, is_norm, scale, enable_expert_mapping), // host api输入
                        OUTPUT(y_desc, indices_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_moe_fused_topk_test, ascend910B2_test_moe_fused_topk_5)
{
    int64_t num_token = 256;
    int64_t expert_num = 256;
    int64_t max_mapping_num = 128;

    uint32_t group_num = 4;
    uint32_t group_topk = 2;
    uint32_t topk = 2;
    uint32_t topn = 2;
    uint32_t activate_type = 0;
    bool is_norm = true;
    float scale = 1.0;
    bool enable_expert_mapping = true;

    vector<int64_t> x_dims = {num_token, expert_num};
    vector<int64_t> add_num_dims = {expert_num};
    vector<int64_t> mapping_num_dims = {expert_num};
    vector<int64_t> mapping_table_dims = {expert_num, max_mapping_num};
    vector<int64_t> y_dims = {num_token, topk};
    vector<int64_t> indices_dims = {num_token, topk};

    auto x_desc = TensorDesc(x_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto add_num_desc = TensorDesc(add_num_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto mapping_num_desc = TensorDesc(mapping_num_dims, ACL_INT32, ACL_FORMAT_ND).ValueRange(1L, max_mapping_num);
    auto mapping_table_desc = TensorDesc(mapping_table_dims, ACL_INT32, ACL_FORMAT_ND).ValueRange(1L, max_mapping_num);

    auto y_desc = TensorDesc(y_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto indices_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeFusedTopk, // host api第二段接口名称
                        INPUT(x_desc, add_num_desc, mapping_num_desc, mapping_table_desc, group_num, group_topk, topn,
                              topk, activate_type, is_norm, scale, enable_expert_mapping), // host api输入
                        OUTPUT(y_desc, indices_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_moe_fused_topk_test, ascend910B2_test_moe_fused_topk_6)
{
    int64_t num_token = 256;
    int64_t expert_num = 256;
    int64_t max_mapping_num = 128;

    uint32_t group_num = 4;
    uint32_t group_topk = 2;
    uint32_t topk = 2;
    uint32_t topn = 2;
    uint32_t activate_type = 0;
    bool is_norm = true;
    float scale = 1.0;
    bool enable_expert_mapping = false;

    vector<int64_t> x_dims = {num_token, expert_num};
    vector<int64_t> add_num_dims = {expert_num};
    vector<int64_t> mapping_num_dims = {expert_num};
    vector<int64_t> mapping_table_dims = {expert_num, max_mapping_num};
    vector<int64_t> y_dims = {num_token, topk};
    vector<int64_t> indices_dims = {num_token, topk};

    auto x_desc = TensorDesc(x_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto add_num_desc = TensorDesc(add_num_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto mapping_num_desc = TensorDesc(mapping_num_dims, ACL_INT32, ACL_FORMAT_ND).ValueRange(1L, max_mapping_num);
    auto mapping_table_desc = TensorDesc(mapping_table_dims, ACL_INT32, ACL_FORMAT_ND).ValueRange(1L, max_mapping_num);

    auto y_desc = TensorDesc(y_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto indices_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeFusedTopk, // host api第二段接口名称
                        INPUT(x_desc, add_num_desc, mapping_num_desc, mapping_table_desc, group_num, group_topk, topn,
                              topk, activate_type, is_norm, scale, enable_expert_mapping), // host api输入
                        OUTPUT(y_desc, indices_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_moe_fused_topk_test, ascend910B2_test_moe_fused_topk_7)
{
    int64_t num_token = 256;
    int64_t expert_num = 256;
    int64_t max_mapping_num = 128;

    uint32_t group_num = 4;
    uint32_t group_topk = 2;
    uint32_t topk = 2;
    uint32_t topn = 2;
    uint32_t activate_type = 0;
    bool is_norm = true;
    float scale = 1.0;
    bool enable_expert_mapping = false;

    vector<int64_t> x_dims = {num_token, expert_num};
    vector<int64_t> add_num_dims = {expert_num};
    vector<int64_t> mapping_num_dims = {expert_num};
    vector<int64_t> mapping_table_dims = {expert_num, max_mapping_num};
    vector<int64_t> y_dims = {num_token, topk};
    vector<int64_t> indices_dims = {num_token, topk};

    auto x_desc = TensorDesc(x_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto add_num_desc = TensorDesc(add_num_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto y_desc = TensorDesc(y_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto indices_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeFusedTopk, // host api第二段接口名称
                        INPUT(x_desc, add_num_desc, nullptr, nullptr, group_num, group_topk, topn, topk, activate_type,
                              is_norm, scale, enable_expert_mapping), // host api输入
                        OUTPUT(y_desc, indices_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_moe_fused_topk_test, ascend910B2_test_moe_fused_topk_empty_x)
{
    int64_t num_token = 0;
    int64_t expert_num = 128;
    uint32_t group_num = 4;
    uint32_t group_topk = 2;
    uint32_t topk = 2;
    uint32_t topn = 2;
    uint32_t activate_type = 0;
    bool is_norm = true;
    float scale = 1.0f;
    bool enable_expert_mapping = false;

    vector<int64_t> x_dims = {num_token, expert_num};
    vector<int64_t> add_num_dims = {expert_num};
    vector<int64_t> y_dims = {num_token, topk};
    vector<int64_t> indices_dims = {num_token, topk};

    auto x_desc = TensorDesc(x_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto add_num_desc = TensorDesc(add_num_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto y_desc = TensorDesc(y_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto indices_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeFusedTopk,
                        INPUT(x_desc, add_num_desc, nullptr, nullptr, group_num, group_topk, topn, topk, activate_type,
                              is_norm, scale, enable_expert_mapping),
                        OUTPUT(y_desc, indices_desc));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    EXPECT_EQ(workspace_size, 0U);
}
