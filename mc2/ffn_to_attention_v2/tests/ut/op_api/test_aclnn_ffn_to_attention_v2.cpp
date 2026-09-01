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
#include <gtest/gtest.h>
#include "test_aclnn_ffn_to_attention_v2_helper.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"
#include "platform/platform_info.h"

using namespace op;
using namespace std;

namespace FFNToAttentionV2UT {

class AclnnFfnToAttentionV2Test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
        cout << "AclnnFfnToAttentionV2Test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
        cout << "AclnnFfnToAttentionV2Test TearDown" << endl;
    }
};

TEST_F(AclnnFfnToAttentionV2Test, TestFfnToAttentionV2AttnRankTable)
{
    TensorDesc context = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc x = TensorDesc({1584, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc sessionIds = TensorDesc({1584}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc microBatchIds = TensorDesc({1584}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc tokenIds = TensorDesc({1584}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expertOffsets = TensorDesc({1584}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc actualTokenNum = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc attnRankTable = TensorDesc({11}, ACL_INT32, ACL_FORMAT_ND);
    const char *group = "test_ffn_to_attention_v2_group";
    int64_t worldSize = 16;
    std::vector<int64_t> tokenInfoTable = {1, 16, 9};
    std::vector<int64_t> tokenData = {1, 16, 9, 7168};
    aclIntArray *tokenInfoTableShape = aclCreateIntArray(tokenInfoTable.data(), tokenInfoTable.size());
    aclIntArray *tokenDataShape = aclCreateIntArray(tokenData.data(), tokenData.size());
    int64_t cclBufferSize = 1LL << 30;

    auto ut = OP_API_UT(aclnnFFNToAttentionV2,
                        INPUT(context, x, sessionIds, microBatchIds, tokenIds, expertOffsets, actualTokenNum,
                              attnRankTable, group, worldSize, tokenInfoTableShape, tokenDataShape, cclBufferSize),
                        OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AclnnFfnToAttentionV2Test, TestFfnToAttentionV2NoAttnRankTable)
{
    TensorDesc context = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc x = TensorDesc({1584, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc sessionIds = TensorDesc({1584}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc microBatchIds = TensorDesc({1584}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc tokenIds = TensorDesc({1584}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expertOffsets = TensorDesc({1584}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc actualTokenNum = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    aclTensor *attnRankTable = nullptr;
    const char *group = "test_ffn_to_attention_v2_group";
    int64_t worldSize = 16;
    std::vector<int64_t> tokenInfoTable = {1, 16, 9};
    std::vector<int64_t> tokenData = {1, 16, 9, 7168};
    aclIntArray *tokenInfoTableShape = aclCreateIntArray(tokenInfoTable.data(), tokenInfoTable.size());
    aclIntArray *tokenDataShape = aclCreateIntArray(tokenData.data(), tokenData.size());
    int64_t cclBufferSize = 1LL << 30;

    auto ut = OP_API_UT(aclnnFFNToAttentionV2,
                        INPUT(context, x, sessionIds, microBatchIds, tokenIds, expertOffsets, actualTokenNum,
                              attnRankTable, group, worldSize, tokenInfoTableShape, tokenDataShape, cclBufferSize),
                        OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

} // namespace FFNToAttentionV2UT
