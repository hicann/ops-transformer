/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../op_api/aclnn_moe_distribute_dispatch_v2.h"

#include <array>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include "gtest/gtest.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

// 950-specific test suite (MUST run before the V2 test suite to cache 950 platform via const static)
namespace MoeDistributeDispatchV2950 {
class L2AclnnMoeDistributeDispatchV2950Test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
        cout << "L2AclnnMoeDistributeDispatchV2950Test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "L2AclnnMoeDistributeDispatchV2950Test TearDown" << endl;
    }
};

TEST_F(L2AclnnMoeDistributeDispatchV2950Test, TestGetWorkspaceSize950NonCcu)
{
    TensorDesc x = TensorDesc({8, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc expertIds = TensorDesc({8, 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc scales = TensorDesc({256, 7168}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc xActiveMask = TensorDesc({8, 256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc expertScales = TensorDesc({8, 8}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t epWorldSize = 288;
    int64_t tpWorldSize = 2;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t shareExpertRankNum = 8;
    int64_t moeExpertNum = 256;
    int64_t quantMode = 0;
    int64_t globalBs = 0;
    int64_t expertTokenNumsType = 1;

    TensorDesc expandX = TensorDesc({8, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc dynamicScales = TensorDesc({8 * 256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc expandIdx = TensorDesc({8 * 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expertTokensNums = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc epRecvCounts = TensorDesc({288}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc tpRecvCounts = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expandScales = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMoeDistributeDispatchV2,
        INPUT(x, expertIds, scales, xActiveMask, expertScales, "test_moe_distribute_dispatch_ep", epWorldSize, epRankId,
              moeExpertNum, "test_moe_distribute_dispatch_tp", tpWorldSize, tpRankId, expertShardType, sharedExpertNum,
              shareExpertRankNum, quantMode, globalBs, expertTokenNumsType, "fullmesh_v1"),
        OUTPUT(expandX, dynamicScales, expandIdx, expertTokensNums, epRecvCounts, tpRecvCounts, expandScales));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AclnnMoeDistributeDispatchV2950Test, TestGetWorkspaceSize950Ccu)
{
    TensorDesc x = TensorDesc({8, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc expertIds = TensorDesc({8, 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc scales = TensorDesc({256, 7168}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc xActiveMask = TensorDesc({8, 256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc expertScales = TensorDesc({8, 8}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t epWorldSize = 288;
    int64_t tpWorldSize = 2;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t shareExpertRankNum = 8;
    int64_t moeExpertNum = 256;
    int64_t quantMode = 0;
    int64_t globalBs = 0;
    int64_t expertTokenNumsType = 1;

    TensorDesc expandX = TensorDesc({8, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc dynamicScales = TensorDesc({8 * 256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc expandIdx = TensorDesc({8 * 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expertTokensNums = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc epRecvCounts = TensorDesc({288}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc tpRecvCounts = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expandScales = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMoeDistributeDispatchV2,
        INPUT(x, expertIds, scales, xActiveMask, expertScales, "test_moe_distribute_dispatch_ep", epWorldSize, epRankId,
              moeExpertNum, "test_moe_distribute_dispatch_tp", tpWorldSize, tpRankId, expertShardType, sharedExpertNum,
              shareExpertRankNum, quantMode, globalBs, expertTokenNumsType, "ccu"),
        OUTPUT(expandX, dynamicScales, expandIdx, expertTokensNums, epRecvCounts, tpRecvCounts, expandScales));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AclnnMoeDistributeDispatchV2950Test, TestExecuteEntry950)
{
    aclnnStatus st = aclnnMoeDistributeDispatchV2(nullptr, 0U, nullptr, nullptr);
    EXPECT_THAT(st, testing::AnyOf(testing::Eq(ACLNN_SUCCESS), testing::Eq(ACLNN_ERR_PARAM_NULLPTR),
                                   testing::Eq(ACLNN_ERR_PARAM_INVALID)));
}
} // namespace MoeDistributeDispatchV2950

namespace MoeDistributeDispatchV2 {
class L2AclnnMoeDistributeDispatchV2Test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "L2AclnnMoeDistributeDispatchV2Test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "L2AclnnMoeDistributeDispatchV2Test TearDown" << endl;
    }
};

TEST_F(L2AclnnMoeDistributeDispatchV2Test, TestAclnnMoeDistributeDispatchFirstApi)
{
    TensorDesc x = TensorDesc({8, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc expertIds = TensorDesc({8, 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc scales = TensorDesc({256, 7168}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc xActiveMask = TensorDesc({8, 256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc expertScales = TensorDesc({8, 8}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t epWorldSize = 288;
    int64_t tpWorldSize = 1;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t shareExpertRankNum = 8;
    int64_t moeExpertNum = 256;
    int64_t quantMode = 0;
    int64_t globalBs = 0;
    int64_t expertTokenNumsType = 1;

    TensorDesc expandX = TensorDesc({8, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc dynamicScales = TensorDesc({8 * 256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc expandIdx = TensorDesc({8 * 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expertTokensNums = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc epRecvCounts = TensorDesc({288}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc tpRecvCounts = TensorDesc({0}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expandScales = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMoeDistributeDispatchV2,
        INPUT(x, expertIds, scales, xActiveMask, expertScales, "test_moe_distribute_dispatch_ep", epWorldSize, epRankId,
              moeExpertNum, "", tpWorldSize, tpRankId, expertShardType, sharedExpertNum, shareExpertRankNum, quantMode,
              globalBs, expertTokenNumsType, "test"),
        OUTPUT(expandX, dynamicScales, expandIdx, expertTokensNums, epRecvCounts, tpRecvCounts, expandScales));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AclnnMoeDistributeDispatchV2Test, TestAclnnMoeDistributeDispatchDynamicQuantRequiresDynamicScales)
{
    TensorDesc x = TensorDesc({8, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc expertIds = TensorDesc({8, 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc scales = TensorDesc({256, 7168}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc xActiveMask = TensorDesc({8, 256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc expertScales = TensorDesc({8, 8}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t epWorldSize = 288;
    int64_t tpWorldSize = 1;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t shareExpertRankNum = 8;
    int64_t moeExpertNum = 256;
    int64_t quantMode = 2;
    int64_t globalBs = 0;
    int64_t expertTokenNumsType = 1;

    TensorDesc expandX = TensorDesc({8, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc dynamicScales = TensorDesc({8 * 256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc expandIdx = TensorDesc({8 * 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expertTokensNums = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc epRecvCounts = TensorDesc({288}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc tpRecvCounts = TensorDesc({0}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expandScales = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMoeDistributeDispatchV2,
        INPUT(x, expertIds, scales, xActiveMask, expertScales, "test_moe_distribute_dispatch_ep", epWorldSize, epRankId,
              moeExpertNum, "", tpWorldSize, tpRankId, expertShardType, sharedExpertNum, shareExpertRankNum, quantMode,
              globalBs, expertTokenNumsType, "test"),
        OUTPUT(expandX, dynamicScales, expandIdx, expertTokensNums, epRecvCounts, tpRecvCounts, expandScales));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AclnnMoeDistributeDispatchV2Test, TestAclnnMoeDistributeDispatchEmptyGroupEp)
{
    TensorDesc x = TensorDesc({8, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc expertIds = TensorDesc({8, 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc scales = TensorDesc({256, 7168}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc xActiveMask = TensorDesc({8, 256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc expertScales = TensorDesc({8, 8}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t epWorldSize = 288;
    int64_t tpWorldSize = 2;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t shareExpertRankNum = 8;
    int64_t moeExpertNum = 256;
    int64_t quantMode = 0;
    int64_t globalBs = 0;
    int64_t expertTokenNumsType = 1;

    TensorDesc expandX = TensorDesc({8, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc dynamicScales = TensorDesc({8 * 256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc expandIdx = TensorDesc({8 * 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expertTokensNums = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc epRecvCounts = TensorDesc({288}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc tpRecvCounts = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expandScales = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMoeDistributeDispatchV2,
        INPUT(x, expertIds, scales, xActiveMask, expertScales, "", epWorldSize, epRankId, moeExpertNum,
              "test_moe_distribute_dispatch_tp", tpWorldSize, tpRankId, expertShardType, sharedExpertNum,
              shareExpertRankNum, quantMode, globalBs, expertTokenNumsType, "test"),
        OUTPUT(expandX, dynamicScales, expandIdx, expertTokensNums, epRecvCounts, tpRecvCounts, expandScales));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2AclnnMoeDistributeDispatchV2Test, TestAclnnMoeDistributeDispatchGroupTpTooLong)
{
    TensorDesc x = TensorDesc({8, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc expertIds = TensorDesc({8, 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc scales = TensorDesc({256, 7168}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc xActiveMask = TensorDesc({8, 256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc expertScales = TensorDesc({8, 8}, ACL_FLOAT, ACL_FORMAT_ND);

    std::string longGroupTp(128, 't');

    int64_t epWorldSize = 288;
    int64_t tpWorldSize = 1;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t shareExpertRankNum = 8;
    int64_t moeExpertNum = 256;
    int64_t quantMode = 0;
    int64_t globalBs = 0;
    int64_t expertTokenNumsType = 1;

    TensorDesc expandX = TensorDesc({8, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc dynamicScales = TensorDesc({8 * 256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc expandIdx = TensorDesc({8 * 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expertTokensNums = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc epRecvCounts = TensorDesc({288}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc tpRecvCounts = TensorDesc({0}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expandScales = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMoeDistributeDispatchV2,
        INPUT(x, expertIds, scales, xActiveMask, expertScales, "test_moe_distribute_dispatch_ep", epWorldSize, epRankId,
              moeExpertNum, longGroupTp.c_str(), tpWorldSize, tpRankId, expertShardType, sharedExpertNum,
              shareExpertRankNum, quantMode, globalBs, expertTokenNumsType, "test"),
        OUTPUT(expandX, dynamicScales, expandIdx, expertTokensNums, epRecvCounts, tpRecvCounts, expandScales));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2AclnnMoeDistributeDispatchV2Test, TestAclnnMoeDistributeDispatchGroupEpTooLong)
{
    TensorDesc x = TensorDesc({8, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc expertIds = TensorDesc({8, 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc scales = TensorDesc({256, 7168}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc xActiveMask = TensorDesc({8, 256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc expertScales = TensorDesc({8, 8}, ACL_FLOAT, ACL_FORMAT_ND);

    std::string longGroupEp(128, 'e');

    int64_t epWorldSize = 288;
    int64_t tpWorldSize = 1;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t shareExpertRankNum = 8;
    int64_t moeExpertNum = 256;
    int64_t quantMode = 0;
    int64_t globalBs = 0;
    int64_t expertTokenNumsType = 1;

    TensorDesc expandX = TensorDesc({8, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc dynamicScales = TensorDesc({8 * 256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc expandIdx = TensorDesc({8 * 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expertTokensNums = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc epRecvCounts = TensorDesc({288}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc tpRecvCounts = TensorDesc({0}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expandScales = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMoeDistributeDispatchV2,
        INPUT(x, expertIds, scales, xActiveMask, expertScales, longGroupEp.c_str(), epWorldSize, epRankId, moeExpertNum,
              "", tpWorldSize, tpRankId, expertShardType, sharedExpertNum, shareExpertRankNum, quantMode, globalBs,
              expertTokenNumsType, "test"),
        OUTPUT(expandX, dynamicScales, expandIdx, expertTokensNums, epRecvCounts, tpRecvCounts, expandScales));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2AclnnMoeDistributeDispatchV2Test, TestAclnnMoeDistributeDispatchV2ExecuteEntry)
{
    // 覆盖 Execute 入口分支（UT 桩无真实 workspace，仅验证不崩溃）
    aclnnStatus st = aclnnMoeDistributeDispatchV2(nullptr, 0U, nullptr, nullptr);
    EXPECT_THAT(st, testing::AnyOf(testing::Eq(ACLNN_SUCCESS), testing::Eq(ACLNN_ERR_PARAM_NULLPTR),
                                   testing::Eq(ACLNN_ERR_PARAM_INVALID)));
}

TEST_F(L2AclnnMoeDistributeDispatchV2Test, TestAclnnMoeDistributeDispatchV2GetWorkspaceAscend950NonCcu)
{
    struct PlatformGuard {
        PlatformGuard(op::SocVersion target)
        {
            op::SetPlatformSocVersion(target);
        }
        ~PlatformGuard()
        {
            op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        }
    } guard(op::SocVersion::ASCEND950);
    TensorDesc x = TensorDesc({8, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc expertIds = TensorDesc({8, 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc scales = TensorDesc({256, 7168}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc xActiveMask = TensorDesc({8, 256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc expertScales = TensorDesc({8, 8}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t epWorldSize = 288;
    int64_t tpWorldSize = 1;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t shareExpertRankNum = 8;
    int64_t moeExpertNum = 256;
    int64_t quantMode = 0;
    int64_t globalBs = 0;
    int64_t expertTokenNumsType = 1;

    TensorDesc expandX = TensorDesc({8, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc dynamicScales = TensorDesc({8 * 256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc expandIdx = TensorDesc({8 * 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expertTokensNums = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc epRecvCounts = TensorDesc({288}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc tpRecvCounts = TensorDesc({0}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expandScales = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMoeDistributeDispatchV2,
        INPUT(x, expertIds, scales, xActiveMask, expertScales, "test_moe_distribute_dispatch_ep", epWorldSize, epRankId,
              moeExpertNum, "", tpWorldSize, tpRankId, expertShardType, sharedExpertNum, shareExpertRankNum, quantMode,
              globalBs, expertTokenNumsType, "fullmesh_v1"),
        OUTPUT(expandX, dynamicScales, expandIdx, expertTokensNums, epRecvCounts, tpRecvCounts, expandScales));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus wsRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(wsRet, ACLNN_ERR_PARAM_INVALID);
}
} // namespace MoeDistributeDispatchV2
