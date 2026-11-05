/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>

#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_attention_update.h"
#include "op_api/op_api_def.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_attention_update_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "attention_update_test Setup" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "attention_update_test TearDown" << std::endl;
    }
};

TEST_F(l2_attention_update_test, ascend910B2_test_attention_update_1)
{
    vector<int64_t> lse_dims = {128};
    vector<int64_t> localOut_dims = {128, 256};
    vector<int64_t> out_dims = {128, 256};
    int64_t updateType = 2;

    auto lse1_desc = TensorDesc(lse_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc(localOut_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc});
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, updateType),
                        OUTPUT(out_desc, nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_attention_update_test, ascend910B2_test_attention_update_2)
{
    vector<int64_t> lse_dims = {128};
    vector<int64_t> localOut_dims = {128, 256};
    vector<int64_t> out_dims = {128, 128};
    int64_t updateType = 0;

    auto lse1_desc = TensorDesc(lse_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse2_desc = TensorDesc(lse_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc(localOut_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut2_desc = TensorDesc(localOut_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc, lse2_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc, localOut2_desc});
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, updateType),
                        OUTPUT(out_desc, nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_attention_update_test, ascend910B2_test_attention_update_3)
{
    vector<int64_t> lse_dims = {128};
    vector<int64_t> localOut_dims = {128, 256};
    vector<int64_t> out_dims = {128, 256};
    int64_t updateType = 0;

    auto lse1_desc = TensorDesc(lse_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse2_desc = TensorDesc(lse_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc(localOut_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut2_desc = TensorDesc(localOut_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc, lse2_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc, localOut2_desc});
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto outLse_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, updateType),
                        OUTPUT(out_desc, outLse_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_attention_update_test, ascend910B2_test_attention_update_4)
{
    vector<int64_t> lse_dims = {128};
    vector<int64_t> localOut_dims = {128, 256};
    vector<int64_t> out_dims = {128, 256};
    int64_t updateType = 0;

    auto lse1_desc = TensorDesc(lse_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse2_desc = TensorDesc(lse_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc(localOut_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut2_desc = TensorDesc(localOut_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc, lse2_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc, localOut2_desc});
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, updateType),
                        OUTPUT(out_desc, nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_attention_update_test, ascend910B2_test_attention_update_5)
{
    vector<int64_t> lse_dims = {128};
    vector<int64_t> localOut_dims = {128, 256};
    vector<int64_t> out_dims = {128, 256};
    vector<int64_t> lseout_dims = {128};
    int64_t updateType = 1;

    auto lse1_desc = TensorDesc(lse_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse2_desc = TensorDesc(lse_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc(localOut_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut2_desc = TensorDesc(localOut_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc, lse2_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc, localOut2_desc});
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto lseout_desc = TensorDesc(lseout_dims, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, updateType),
                        OUTPUT(out_desc, lseout_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 测试sp大于16的场景aclnn接口能否正常运行
TEST_F(l2_attention_update_test, ascend910B2_test_attention_update_sp17)
{
    vector<int64_t> lse_dims = {128};
    vector<int64_t> localOut_dims = {128, 256};
    vector<int64_t> out_dims = {128, 256};
    int64_t updateType = 0;
    int64_t sp = 120;

    vector<TensorDesc> lse_list;
    vector<TensorDesc> localOut_list;
    for (int64_t i = 0; i < sp; ++i) {
        lse_list.emplace_back(TensorDesc(lse_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1));
        localOut_list.emplace_back(TensorDesc(localOut_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1));
    }
    auto lse_list_desc = TensorListDesc(lse_list);
    auto localOut_list_desc = TensorListDesc(localOut_list);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, updateType),
                        OUTPUT(out_desc, nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 测试sp大于128的场景aclnn接口能否正常拦截
TEST_F(l2_attention_update_test, ascend910B2_test_attention_update_sp129)
{
    vector<int64_t> lse_dims = {128};
    vector<int64_t> localOut_dims = {128, 256};
    vector<int64_t> out_dims = {128, 256};
    int64_t updateType = 0;
    int64_t sp = 129;

    vector<TensorDesc> lse_list;
    vector<TensorDesc> localOut_list;
    for (int64_t i = 0; i < sp; ++i) {
        lse_list.emplace_back(TensorDesc(lse_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1));
        localOut_list.emplace_back(TensorDesc(localOut_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1));
    }
    auto lse_list_desc = TensorListDesc(lse_list);
    auto localOut_list_desc = TensorListDesc(localOut_list);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, updateType),
                        OUTPUT(out_desc, nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_attention_update_test, Ascend910_9589_test_attention_update_1)
{
    vector<int64_t> lse_dims = {128};
    vector<int64_t> localOut_dims = {128, 256};
    vector<int64_t> out_dims = {128, 256};
    int64_t updateType = 0;

    auto lse1_desc = TensorDesc(lse_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse2_desc = TensorDesc(lse_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc(localOut_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut2_desc = TensorDesc(localOut_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc, lse2_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc, localOut2_desc});
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, updateType),
                        OUTPUT(out_desc, nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_attention_update_test, Ascend910_9589_test_attention_update_2)
{
    vector<int64_t> lse_dims = {128};
    vector<int64_t> localOut_dims = {128, 256};
    vector<int64_t> out_dims = {128, 256};
    int64_t updateType = 0;

    auto lse1_desc = TensorDesc(lse_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse2_desc = TensorDesc(lse_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc(localOut_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut2_desc = TensorDesc(localOut_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc, lse2_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc, localOut2_desc});
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto outLse_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, updateType),
                        OUTPUT(out_desc, outLse_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_attention_update_test, Ascend910_9589_test_attention_update_3)
{
    vector<int64_t> lse_dims = {128};
    vector<int64_t> localOut_dims = {128, 256};
    vector<int64_t> out_dims = {128, 256};
    int64_t updateType = 1;

    auto lse1_desc = TensorDesc(lse_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse2_desc = TensorDesc(lse_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc(localOut_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut2_desc = TensorDesc(localOut_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc, lse2_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc, localOut2_desc});
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto go_desc = TensorDesc(lse_dims, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, updateType),
                        OUTPUT(out_desc, go_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// ==================== OP_LOGE 分支覆盖用例（错误码整改配套，Ascend910B路径） ====================

// 分支: CheckUpdateTypeAndLseOut - updateType非法 (OP_LOGE "updateType should be 0 or 1")
TEST_F(l2_attention_update_test, ascend910B2_error_updateTypeIllegal)
{
    auto lse1_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc});
    auto out_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, (int64_t)2),
                        OUTPUT(out_desc, nullptr));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// 分支: CheckUpdateTypeAndLseOut - updateType=0时lseOut必须为nullptr (OP_LOGE "lseOut should be nullptr")
TEST_F(l2_attention_update_test, ascend910B2_error_lseOutNotNullWithType0)
{
    auto lse1_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc});
    auto out_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND);
    auto lseout_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, (int64_t)0),
                        OUTPUT(out_desc, lseout_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// 分支: CheckUpdateTypeAndLseOut - updateType=1时lseOut不能为nullptr (OP_LOGE "lseOut should not be nullptr")
TEST_F(l2_attention_update_test, ascend910B2_error_lseOutNullWithType1)
{
    auto lse1_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc});
    auto out_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, (int64_t)1),
                        OUTPUT(out_desc, nullptr));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// 分支: CheckNotNull - lse为空指针 (OP_CHECK_NULL隐式OP_LOGE)
TEST_F(l2_attention_update_test, ascend910B2_error_nullLse)
{
    auto localOut1_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut_list_desc = TensorListDesc({localOut1_desc});
    auto out_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnAttentionUpdate, INPUT(nullptr, localOut_list_desc, (int64_t)0), OUTPUT(out_desc, nullptr));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_NULLPTR);
}

// 分支: CheckNotNull - out为空指针 (OP_CHECK_NULL隐式OP_LOGE)
TEST_F(l2_attention_update_test, ascend910B2_error_nullOut)
{
    auto lse1_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc});

    auto ut =
        OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, (int64_t)0), OUTPUT(nullptr, nullptr));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_NULLPTR);
}

// 分支: CheckDtypeValid - lse仅支持FLOAT (OP_CHECK_DTYPE_NOT_SUPPORT隐式OP_LOGE)
TEST_F(l2_attention_update_test, ascend910B2_error_lseDtypeFp16)
{
    auto lse1_desc = TensorDesc({128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc({128, 256}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc});
    auto out_desc = TensorDesc({128, 256}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, (int64_t)0),
                        OUTPUT(out_desc, nullptr));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// 分支: CheckDtypeValid - localOut仅支持FLOAT/FLOAT16/BF16 (OP_CHECK_DTYPE_NOT_SUPPORT隐式OP_LOGE)
TEST_F(l2_attention_update_test, ascend910B2_error_localOutDtypeInt32)
{
    auto lse1_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc({128, 256}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc});
    auto out_desc = TensorDesc({128, 256}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, (int64_t)0),
                        OUTPUT(out_desc, nullptr));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// 分支: CheckDtypeValid - out仅支持FLOAT/FLOAT16/BF16 (OP_CHECK_DTYPE_NOT_SUPPORT隐式OP_LOGE)
TEST_F(l2_attention_update_test, ascend910B2_error_outDtypeInt32)
{
    auto lse1_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc});
    auto out_desc = TensorDesc({128, 256}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, (int64_t)0),
                        OUTPUT(out_desc, nullptr));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// 分支: CheckSp - lse与localOut的size必须相等 (OP_LOGE "Size of lse must equal to size of localOut")
TEST_F(l2_attention_update_test, ascend910B2_error_lseSizeNeLocalOutSize)
{
    auto lse1_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse2_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc, lse2_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc});
    auto out_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, (int64_t)0),
                        OUTPUT(out_desc, nullptr));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// 分支: CheckShape - lse维度数必须为1 (OP_LOGE "Dim num of lse should be 1")
TEST_F(l2_attention_update_test, ascend910B2_error_lseDimNumNot1)
{
    auto lse1_desc = TensorDesc({128, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc});
    auto out_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, (int64_t)0),
                        OUTPUT(out_desc, nullptr));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// 分支: CheckShape - localOut维度数必须为2 (OP_LOGE "Dim num of localOut should be 2")
TEST_F(l2_attention_update_test, ascend910B2_error_localOutDimNumNot2)
{
    auto lse1_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc});
    auto out_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, (int64_t)0),
                        OUTPUT(out_desc, nullptr));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// 分支: CheckShape - lse各tensor shape必须相同 (OP_LOGE "shape of every tensor of lse must be same")
TEST_F(l2_attention_update_test, ascend910B2_error_lseTensorsShapeDiff)
{
    auto lse1_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse2_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut2_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc, lse2_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc, localOut2_desc});
    auto out_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, (int64_t)0),
                        OUTPUT(out_desc, nullptr));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// 分支: CheckShape - localOut各tensor shape必须相同 (OP_LOGE "shape of every tensor of localOut must be same")
TEST_F(l2_attention_update_test, ascend910B2_error_localOutTensorsShapeDiff)
{
    auto lse1_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse2_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut2_desc = TensorDesc({64, 256}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc, lse2_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc, localOut2_desc});
    auto out_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, (int64_t)0),
                        OUTPUT(out_desc, nullptr));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// 分支: CheckShape - localOut第1维必须是8的倍数且不超过512 (OP_LOGE "Size of localOut[1] must be a multiple of 8")
TEST_F(l2_attention_update_test, ascend910B2_error_goHdNotMultipleOf8)
{
    auto lse1_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc({128, 10}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc});
    auto out_desc = TensorDesc({128, 10}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, (int64_t)0),
                        OUTPUT(out_desc, nullptr));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// 分支: CheckShape - lse[0]与localOut[0]的第0维必须相等 (OP_LOGE "Size of lse[0] must equal to size of localOut[0]")
TEST_F(l2_attention_update_test, ascend910B2_error_lseBshcNeLocalOutBshc)
{
    auto lse1_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc});
    auto out_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, (int64_t)0),
                        OUTPUT(out_desc, nullptr));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// 分支: CheckShape - out维度数必须为2 (OP_LOGE "Dim num of out should be 2")
TEST_F(l2_attention_update_test, ascend910B2_error_outDimNumNot2)
{
    auto lse1_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc});
    auto out_desc = TensorDesc({128, 256, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, (int64_t)0),
                        OUTPUT(out_desc, nullptr));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// 分支: CheckShape - lse[0]与out[0]的第0维必须相等 (OP_LOGE "Size of lse[0] must equal to size of out[0]")
TEST_F(l2_attention_update_test, ascend910B2_error_lseBshcNeOutBshc)
{
    auto lse1_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc});
    auto out_desc = TensorDesc({64, 256}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, (int64_t)0),
                        OUTPUT(out_desc, nullptr));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// 分支: CheckShape - localOut[1]与out[1]必须相等 (OP_LOGE "Size of localOut[1] must equal to size of out[1]")
TEST_F(l2_attention_update_test, ascend910B2_error_goHdNeOutHd)
{
    auto lse1_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc});
    auto out_desc = TensorDesc({128, 128}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, (int64_t)0),
                        OUTPUT(out_desc, nullptr));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// 分支: CheckShape - updateType=1时lseOut维度数必须为1 (OP_LOGE "Dim num of lseOut should be 1")
TEST_F(l2_attention_update_test, ascend910B2_error_lseOutDimNumNot1)
{
    auto lse1_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc});
    auto out_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND);
    auto lseout_desc = TensorDesc({128, 8}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, (int64_t)1),
                        OUTPUT(out_desc, lseout_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// 分支: CheckShape - lse[0]与lseOut[0]的第0维必须相等 (OP_LOGE "Size of lse[0] must equal to size of lseOut[0]")
TEST_F(l2_attention_update_test, ascend910B2_error_lseBshcNeLseOutBshc)
{
    auto lse1_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc});
    auto out_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND);
    auto lseout_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, (int64_t)1),
                        OUTPUT(out_desc, lseout_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// ==================== OP_LOGE 分支覆盖用例（Ascend950 regbase路径） ====================

class l2_attention_update_950_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    }
    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    }
};

// 分支: CheckSp_95 - 950平台sp最大为16 (OP_LOGE "Size of lse must be greater than zero and less than 16")
TEST_F(l2_attention_update_950_test, ascend950_error_spOutOfRange)
{
    vector<TensorDesc> lse_list;
    vector<TensorDesc> localOut_list;
    for (int64_t i = 0; i < 17; ++i) {
        lse_list.emplace_back(TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1));
        localOut_list.emplace_back(TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1));
    }
    auto lse_list_desc = TensorListDesc(lse_list);
    auto localOut_list_desc = TensorListDesc(localOut_list);
    auto out_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, (int64_t)0),
                        OUTPUT(out_desc, nullptr));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// 分支: CheckSp_95 - lse与localOut的size必须相等 (OP_LOGE "Size of lse must equal to size of localOut")
TEST_F(l2_attention_update_950_test, ascend950_error_lseSizeNeLocalOutSize)
{
    auto lse1_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse2_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc, lse2_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc});
    auto out_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, (int64_t)0),
                        OUTPUT(out_desc, nullptr));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// 分支: CheckDtypeValid_95 - 950平台lse仅支持FLOAT (OP_CHECK_DTYPE_NOT_SUPPORT隐式OP_LOGE)
TEST_F(l2_attention_update_950_test, ascend950_error_lseDtypeFp16)
{
    auto lse1_desc = TensorDesc({128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc({128, 256}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc});
    auto out_desc = TensorDesc({128, 256}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, (int64_t)0),
                        OUTPUT(out_desc, nullptr));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// 分支: CheckDtypeValid_95 - localOut各tensor dtype必须一致 (OP_CHECK_DTYPE_NOT_SAME隐式OP_LOGE)
TEST_F(l2_attention_update_950_test, ascend950_error_localOutDtypeNotSame)
{
    auto lse1_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse2_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut2_desc = TensorDesc({128, 256}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc, lse2_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc, localOut2_desc});
    auto out_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, (int64_t)0),
                        OUTPUT(out_desc, nullptr));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// 分支: CheckDtypeValid_95 - out与localOut[0] dtype必须一致 (OP_CHECK_DTYPE_NOT_SAME隐式OP_LOGE)
TEST_F(l2_attention_update_950_test, ascend950_error_outDtypeNotSame)
{
    auto lse1_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc});
    auto out_desc = TensorDesc({128, 256}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, (int64_t)0),
                        OUTPUT(out_desc, nullptr));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// 分支: CheckDtypeValid_95 - updateType=1时lseOut仅支持FLOAT (OP_CHECK_DTYPE_NOT_SUPPORT隐式OP_LOGE)
TEST_F(l2_attention_update_950_test, ascend950_error_lseOutDtypeFp16)
{
    auto lse1_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc});
    auto out_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND);
    auto lseout_desc = TensorDesc({128}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, (int64_t)1),
                        OUTPUT(out_desc, lseout_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// 分支: CheckShape_95 - lse维度数必须为1 (OP_LOGE "Dim num of lse should be 1")
TEST_F(l2_attention_update_950_test, ascend950_error_lseDimNumNot1)
{
    auto lse1_desc = TensorDesc({128, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc});
    auto out_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, (int64_t)0),
                        OUTPUT(out_desc, nullptr));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// 分支: CheckShape_95 - localOut维度数必须为2 (OP_LOGE "Dim num of localOut should be 2")
TEST_F(l2_attention_update_950_test, ascend950_error_localOutDimNumNot2)
{
    auto lse1_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc});
    auto out_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, (int64_t)0),
                        OUTPUT(out_desc, nullptr));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// 分支: CheckShape_95 - localOut第1维必须是8的倍数且不超过512 (OP_LOGE "Size of localOut[1] must be a multiple of 8")
TEST_F(l2_attention_update_950_test, ascend950_error_goHdNotMultipleOf8)
{
    auto lse1_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc({128, 10}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc});
    auto out_desc = TensorDesc({128, 10}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, (int64_t)0),
                        OUTPUT(out_desc, nullptr));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// 分支: CheckShape_95 - lse[0]与localOut[0]的第0维必须相等 (OP_LOGE "Size of lse[0] must equal to size of localOut[0]")
TEST_F(l2_attention_update_950_test, ascend950_error_lseBshcNeLocalOutBshc)
{
    auto lse1_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc});
    auto out_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, (int64_t)0),
                        OUTPUT(out_desc, nullptr));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// 分支: CheckShape_95 - lse各tensor shape必须相同 (OP_LOGE "shape of every tensor of lse must be same")
TEST_F(l2_attention_update_950_test, ascend950_error_lseTensorsShapeDiff)
{
    auto lse1_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse2_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut2_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc, lse2_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc, localOut2_desc});
    auto out_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, (int64_t)0),
                        OUTPUT(out_desc, nullptr));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// 分支: CheckShape_95 - localOut各tensor shape必须相同 (OP_LOGE "shape of every tensor of localOut must be same")
TEST_F(l2_attention_update_950_test, ascend950_error_localOutTensorsShapeDiff)
{
    auto lse1_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse2_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut2_desc = TensorDesc({64, 256}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc, lse2_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc, localOut2_desc});
    auto out_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, (int64_t)0),
                        OUTPUT(out_desc, nullptr));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// 分支: CheckShape_95 - out维度数必须为2 (OP_LOGE "Dim num of out should be 2")
TEST_F(l2_attention_update_950_test, ascend950_error_outDimNumNot2)
{
    auto lse1_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc});
    auto out_desc = TensorDesc({128, 256, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, (int64_t)0),
                        OUTPUT(out_desc, nullptr));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// 分支: CheckShape_95 - lse[0]与out[0]的第0维必须相等 (OP_LOGE "Size of lse[0] must equal to size of out[0]")
TEST_F(l2_attention_update_950_test, ascend950_error_lseBshcNeOutBshc)
{
    auto lse1_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc});
    auto out_desc = TensorDesc({64, 256}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, (int64_t)0),
                        OUTPUT(out_desc, nullptr));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// 分支: CheckShape_95 - localOut[1]与out[1]必须相等 (OP_LOGE "Size of localOut[1] must equal to size of out[1]")
TEST_F(l2_attention_update_950_test, ascend950_error_goHdNeOutHd)
{
    auto lse1_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc});
    auto out_desc = TensorDesc({128, 128}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, (int64_t)0),
                        OUTPUT(out_desc, nullptr));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// 分支: CheckShape_95 - updateType=1时lseOut维度数必须为1 (OP_LOGE "Dim num of lseOut should be 1")
TEST_F(l2_attention_update_950_test, ascend950_error_lseOutDimNumNot1)
{
    auto lse1_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc});
    auto out_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND);
    auto lseout_desc = TensorDesc({128, 8}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, (int64_t)1),
                        OUTPUT(out_desc, lseout_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// 分支: CheckShape_95 - lse[0]与lseOut[0]的第0维必须相等 (OP_LOGE "Size of lse[0] must equal to size of lseOut[0]")
TEST_F(l2_attention_update_950_test, ascend950_error_lseBshcNeLseOutBshc)
{
    auto lse1_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse_list_desc = TensorListDesc({lse1_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc});
    auto out_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND);
    auto lseout_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse_list_desc, localOut_list_desc, (int64_t)1),
                        OUTPUT(out_desc, lseout_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

namespace op {
extern SocVersion g_socVersion;
}
namespace {
class AttentionUpdateSocGuard {
public:
    explicit AttentionUpdateSocGuard(op::SocVersion soc)
        : saved_(op::g_socVersion)
    {
        op::g_socVersion = soc;
    }
    ~AttentionUpdateSocGuard()
    {
        op::g_socVersion = saved_;
    }

private:
    op::SocVersion saved_;
};

aclnnStatus CheckPackedAttentionUpdate(op::SocVersion soc, int64_t update, bool mismatchedRows = false)
{
    AttentionUpdateSocGuard guard(soc);
    auto lse = TensorListDesc({TensorDesc({5, mismatchedRows ? 34 : 33}, ACL_FLOAT, ACL_FORMAT_ND)});
    auto localOut =
        TensorListDesc({TensorDesc({5, 33, 72}, ACL_BF16, ACL_FORMAT_ND, {33 * 76, 76, 1}, 3, {5 * 33 * 76 + 3})});
    auto out = TensorDesc({33, 72}, ACL_BF16, ACL_FORMAT_ND);
    uint64_t workspace = 0;
    if (update == 1) {
        auto lseOut = TensorDesc({33}, ACL_FLOAT, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse, localOut, update), OUTPUT(out, lseOut));
        return ut.TestGetWorkspaceSize(&workspace);
    }
    auto ut = OP_API_UT(aclnnAttentionUpdate, INPUT(lse, localOut, update), OUTPUT(out, nullptr));
    return ut.TestGetWorkspaceSize(&workspace);
}
} // namespace

TEST_F(l2_attention_update_test, PackedRowStridedA3WithoutLseOut)
{
    EXPECT_EQ(CheckPackedAttentionUpdate(op::SocVersion::ASCEND910_93, 0), ACLNN_SUCCESS);
}

TEST_F(l2_attention_update_test, PackedRowStridedA5WithLseOut)
{
    EXPECT_EQ(CheckPackedAttentionUpdate(op::SocVersion::ASCEND950, 1), ACLNN_SUCCESS);
}

TEST_F(l2_attention_update_test, PackedMismatchedRowsRejected)
{
    EXPECT_EQ(CheckPackedAttentionUpdate(op::SocVersion::ASCEND950, 1, true), ACLNN_ERR_PARAM_INVALID);
}
