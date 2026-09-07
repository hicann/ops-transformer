/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <float.h>
#include <array>
#include <vector>
#include "gtest/gtest.h"
#include <gmock/gmock.h>
#include "../../../op_api/aclnn_moe_distribute_combine.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"
using namespace op;
using namespace std;

namespace MoeDistributeCombine {
class L2MoeDistributeCombineTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "L2MoeDistributeCombineTest SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "L2MoeDistributeCombineTest TearDown" << endl;
    }
};

TEST_F(L2MoeDistributeCombineTest, TestMoeDistributeCombineFirstApi)
{
    TensorDesc expandX = TensorDesc({32, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc expertIds = TensorDesc({32, 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expandIdx = TensorDesc({32 * 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc epSendCounts = TensorDesc({288}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expertScales = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc tpSendCounts = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc xActiveMask = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND);
    TensorDesc activationScale = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc weightScale = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList = TensorDesc({288}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc expandScales = TensorDesc({32}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t epWorldSize = 288;
    int64_t tpWorldSize = 1;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t sharedExpertRankNum = 32;
    int64_t moeExpertNum = 256;
    int64_t globalBs = 0;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;

    TensorDesc x = TensorDesc({32, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMoeDistributeCombine,
        INPUT(expandX, expertIds, expandIdx, epSendCounts, expertScales, tpSendCounts, xActiveMask, activationScale,
              weightScale, groupList, expandScales, "test_moe_distribute_combine_ep", epWorldSize, epRankId,
              moeExpertNum, "test_moe_distribute_combine_tp", tpWorldSize, tpRankId, expertShardType, sharedExpertNum,
              sharedExpertRankNum, globalBs, outDtype, commQuantMode, groupListType),
        OUTPUT(x));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MoeDistributeCombineTest, Ascend910B2TestMoeDistributeCombineTpNotEmpty)
{
    TensorDesc expandX = TensorDesc({32, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc expertIds = TensorDesc({32, 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expandIdx = TensorDesc({32 * 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc epSendCounts = TensorDesc({256}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc tpSendCounts = TensorDesc({0}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc xActiveMask = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND);
    TensorDesc activationScale = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc weightScale = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList = TensorDesc({256}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc expandScales = TensorDesc({16384}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc expertScales = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t epWorldSize = 256;
    int64_t tpWorldSize = 0;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 0;
    int64_t sharedExpertRankNum = 0;
    int64_t moeExpertNum = 256;
    int64_t globalBs = 0;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;

    TensorDesc x = TensorDesc({32, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeDistributeCombine,
                        INPUT(expandX, expertIds, expandIdx, epSendCounts, expertScales, tpSendCounts, xActiveMask,
                              activationScale, weightScale, groupList, expandScales, "", epWorldSize, epRankId,
                              moeExpertNum, "test_moe_distribute_combine_tp", tpWorldSize, tpRankId, expertShardType,
                              sharedExpertNum, sharedExpertRankNum, globalBs, outDtype, commQuantMode, groupListType),
                        OUTPUT(x));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}
// 测试非 ASCEND910B 平台分支 - aclnnMoeDistributeCombineGetWorkspaceSize 中的 else 分支 (line 119-123)
TEST_F(L2MoeDistributeCombineTest, Ascend910B2TestMoeDistributeCombineNon910BPlatform)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910_93);

    TensorDesc expandX = TensorDesc({32, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc expertIds = TensorDesc({32, 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expandIdx = TensorDesc({32 * 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc epSendCounts = TensorDesc({288}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expertScales = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc tpSendCounts = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc xActiveMask = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND);
    TensorDesc activationScale = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc weightScale = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList = TensorDesc({288}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc expandScales = TensorDesc({32}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t epWorldSize = 288;
    int64_t tpWorldSize = 1;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t sharedExpertRankNum = 32;
    int64_t moeExpertNum = 256;
    int64_t globalBs = 0;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;

    TensorDesc x = TensorDesc({32, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMoeDistributeCombine,
        INPUT(expandX, expertIds, expandIdx, epSendCounts, expertScales, tpSendCounts, xActiveMask, activationScale,
              weightScale, groupList, expandScales, "test_moe_distribute_combine_ep", epWorldSize, epRankId,
              moeExpertNum, "test_moe_distribute_combine_tp", tpWorldSize, tpRankId, expertShardType, sharedExpertNum,
              sharedExpertRankNum, globalBs, outDtype, commQuantMode, groupListType),
        OUTPUT(x));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);

    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
}

// 测试 groupEp 超过最大长度 - 覆盖 CheckParams 中 groupEp 长度检查
TEST_F(L2MoeDistributeCombineTest, GroupEpExceedsMaxLength)
{
    std::string longGroupEp(130, 'a');

    TensorDesc expandX = TensorDesc({32, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc expertIds = TensorDesc({32, 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expandIdx = TensorDesc({32 * 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc epSendCounts = TensorDesc({288}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expertScales = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc tpSendCounts = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc xActiveMask = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND);
    TensorDesc activationScale = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc weightScale = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList = TensorDesc({288}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc expandScales = TensorDesc({32}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t epWorldSize = 288;
    int64_t tpWorldSize = 1;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t sharedExpertRankNum = 32;
    int64_t moeExpertNum = 256;
    int64_t globalBs = 0;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;

    TensorDesc x = TensorDesc({32, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMoeDistributeCombine,
        INPUT(expandX, expertIds, expandIdx, epSendCounts, expertScales, tpSendCounts, xActiveMask, activationScale,
              weightScale, groupList, expandScales, longGroupEp.c_str(), epWorldSize, epRankId, moeExpertNum,
              "test_moe_distribute_combine_tp", tpWorldSize, tpRankId, expertShardType, sharedExpertNum,
              sharedExpertRankNum, globalBs, outDtype, commQuantMode, groupListType),
        OUTPUT(x));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试 groupEp 为空字符串 - 覆盖 CheckParams 中 groupEp 长度检查
TEST_F(L2MoeDistributeCombineTest, GroupEpEmptyString)
{
    TensorDesc expandX = TensorDesc({32, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc expertIds = TensorDesc({32, 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expandIdx = TensorDesc({32 * 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc epSendCounts = TensorDesc({288}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expertScales = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc tpSendCounts = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc xActiveMask = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND);
    TensorDesc activationScale = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc weightScale = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList = TensorDesc({288}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc expandScales = TensorDesc({32}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t epWorldSize = 288;
    int64_t tpWorldSize = 1;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t sharedExpertRankNum = 32;
    int64_t moeExpertNum = 256;
    int64_t globalBs = 0;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;

    TensorDesc x = TensorDesc({32, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeDistributeCombine,
                        INPUT(expandX, expertIds, expandIdx, epSendCounts, expertScales, tpSendCounts, xActiveMask,
                              activationScale, weightScale, groupList, expandScales, "", epWorldSize, epRankId,
                              moeExpertNum, "test_moe_distribute_combine_tp", tpWorldSize, tpRankId, expertShardType,
                              sharedExpertNum, sharedExpertRankNum, globalBs, outDtype, commQuantMode, groupListType),
                        OUTPUT(x));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// 测试非910B平台下正常调用 - 覆盖 else 分支 (line 119-123 中 groupTp 直接透传)
TEST_F(L2MoeDistributeCombineTest, Non910BPlatformWithTpGroup)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910_93);

    TensorDesc expandX = TensorDesc({32, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc expertIds = TensorDesc({32, 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expandIdx = TensorDesc({32 * 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc epSendCounts = TensorDesc({288}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expertScales = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc tpSendCounts = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc xActiveMask = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND);
    TensorDesc activationScale = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc weightScale = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList = TensorDesc({288}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc expandScales = TensorDesc({32}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t epWorldSize = 288;
    int64_t tpWorldSize = 1;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t sharedExpertRankNum = 32;
    int64_t moeExpertNum = 256;
    int64_t globalBs = 0;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;

    TensorDesc x = TensorDesc({32, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMoeDistributeCombine,
        INPUT(expandX, expertIds, expandIdx, epSendCounts, expertScales, tpSendCounts, xActiveMask, activationScale,
              weightScale, groupList, expandScales, "test_moe_distribute_combine_ep", epWorldSize, epRankId,
              moeExpertNum, "test_moe_distribute_combine_tp", tpWorldSize, tpRankId, expertShardType, sharedExpertNum,
              sharedExpertRankNum, globalBs, outDtype, commQuantMode, groupListType),
        OUTPUT(x));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);

    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
}
} // namespace MoeDistributeCombine
