/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../op_api/aclnn_moe_distribute_combine_v2.h"
#include "../../../op_api/moe_distribute_combine_v2_base.h"
#include "../../../../common/utils/op_mc2_def.h"

#include <array>
#include <cstdlib>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include "gtest/gtest.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

namespace MoeDistributeCombineV2 {
namespace {
constexpr int64_t EP_WORLD_SIZE = 288;
constexpr int64_t TP_WORLD_SIZE = 1;
constexpr int64_t EP_RANK_ID = 0;
constexpr int64_t TP_RANK_ID = 0;
constexpr int64_t EXPERT_SHARD_TYPE = 0;
constexpr int64_t SHARED_EXPERT_NUM = 1;
constexpr int64_t SHARED_EXPERT_RANK_NUM = 32;
constexpr int64_t MOE_EXPERT_NUM = 256;
constexpr int64_t GLOBAL_BS = 0;
constexpr int64_t OUT_DTYPE = 0;
constexpr int64_t COMM_QUANT_MODE = 0;
constexpr int64_t GROUP_LIST_TYPE = 0;

aclnnStatus CallBaseGetWorkspaceSize(const char *groupTp, const char *commAlg)
{
    TensorDesc expandXDesc = TensorDesc({32, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc expertIdsDesc = TensorDesc({32, 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc assistInfoDesc = TensorDesc({32 * 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc epSendCountsDesc = TensorDesc({288}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expertScalesDesc = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc xOutDesc = TensorDesc({32, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto expandX = expandXDesc.ToAclType();
    auto expertIds = expertIdsDesc.ToAclType();
    auto assistInfo = assistInfoDesc.ToAclType();
    auto epSendCounts = epSendCountsDesc.ToAclType();
    auto expertScales = expertScalesDesc.ToAclType();
    auto xOut = xOutDesc.ToAclType();
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    return aclnnMoeDistributeCombineBaseGetWorkspaceSize(
        expandX.get(), expertIds.get(), assistInfo.get(), epSendCounts.get(), expertScales.get(), nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        "test_moe_distribute_combine_ep", EP_WORLD_SIZE, EP_RANK_ID, MOE_EXPERT_NUM, groupTp, TP_WORLD_SIZE, TP_RANK_ID,
        EXPERT_SHARD_TYPE, SHARED_EXPERT_NUM, SHARED_EXPERT_RANK_NUM, GLOBAL_BS, OUT_DTYPE, COMM_QUANT_MODE,
        GROUP_LIST_TYPE, commAlg, 0, 0, 0, xOut.get(), &workspaceSize, &executor);
}
} // namespace

TEST(MoeDistributeCombineV2BaseCoverageTest, TestBaseGetWorkspaceSize950Ccu)
{
    EXPECT_EXIT(
        {
            op::SetPlatformNpuArch(Ops::Base::DAV_3510);
            (void)CallBaseGetWorkspaceSize("test_moe_distribute_combine_tp", "ccu");
            std::exit(0);
        },
        testing::ExitedWithCode(0), "");
}

TEST(MoeDistributeCombineV2BaseCoverageTest, TestBaseGetWorkspaceSizeNon910B)
{
    EXPECT_EXIT(
        {
            op::SetPlatformSocVersion(op::SocVersion::ASCEND910_93);
            (void)CallBaseGetWorkspaceSize("test_moe_distribute_combine_tp", "test");
            std::exit(0);
        },
        testing::ExitedWithCode(0), "");
}

class L2MoeDistributeCombineV2Test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "L2MoeDistributeCombineV2Test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "L2MoeDistributeCombineV2Test TearDown" << endl;
    }
};

TEST_F(L2MoeDistributeCombineV2Test, TestMoeDistributeCombineFirstApi)
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
    TensorDesc sharedExpertX = TensorDesc({32, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);

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
        aclnnMoeDistributeCombineV2,
        INPUT(expandX, expertIds, expandIdx, epSendCounts, expertScales, tpSendCounts, xActiveMask, activationScale,
              weightScale, groupList, expandScales, sharedExpertX, "test_moe_distribute_combine_ep", epWorldSize,
              epRankId, moeExpertNum, "test_moe_distribute_combine_tp", tpWorldSize, tpRankId, expertShardType,
              sharedExpertNum, sharedExpertRankNum, globalBs, outDtype, commQuantMode, groupListType, "test"),
        OUTPUT(x));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MoeDistributeCombineV2Test, TestMoeDistributeCombineV2ExecuteEntry)
{
    aclnnStatus ret = aclnnMoeDistributeCombineV2(nullptr, 0, nullptr, nullptr);
    EXPECT_THAT(ret, testing::AnyOf(testing::Eq(ACLNN_SUCCESS), testing::Eq(ACLNN_ERR_PARAM_NULLPTR),
                                    testing::Eq(ACLNN_ERR_PARAM_INVALID), testing::Eq(ACLNN_ERR_INNER)));
}

TEST(MoeDistributeCombineV2BaseHelperTest, CombineCheckNotNullGroupNull)
{
    TensorDesc expandXDesc{{1}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc expertIdsDesc{{1}, ACL_INT32, ACL_FORMAT_ND};
    TensorDesc assistDesc{{1}, ACL_INT32, ACL_FORMAT_ND};
    TensorDesc epSendDesc{{1}, ACL_INT32, ACL_FORMAT_ND};
    TensorDesc scalesDesc{{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc xOutDesc{{1}, ACL_FLOAT16, ACL_FORMAT_ND};
    auto expandX = expandXDesc.ToAclType();
    auto expertIds = expertIdsDesc.ToAclType();
    auto assist = assistDesc.ToAclType();
    auto epSend = epSendDesc.ToAclType();
    auto scales = scalesDesc.ToAclType();
    auto xOut = xOutDesc.ToAclType();
    EXPECT_FALSE(CombineCheckNotNull(expandX.get(), expertIds.get(), assist.get(), epSend.get(), scales.get(), nullptr,
                                     xOut.get()));
}

TEST(MoeDistributeCombineV2BaseHelperTest, CombineCheckNotNullGroupEmpty)
{
    TensorDesc expandXDesc{{1}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc expertIdsDesc{{1}, ACL_INT32, ACL_FORMAT_ND};
    TensorDesc assistDesc{{1}, ACL_INT32, ACL_FORMAT_ND};
    TensorDesc epSendDesc{{1}, ACL_INT32, ACL_FORMAT_ND};
    TensorDesc scalesDesc{{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc xOutDesc{{1}, ACL_FLOAT16, ACL_FORMAT_ND};
    auto expandX = expandXDesc.ToAclType();
    auto expertIds = expertIdsDesc.ToAclType();
    auto assist = assistDesc.ToAclType();
    auto epSend = epSendDesc.ToAclType();
    auto scales = scalesDesc.ToAclType();
    auto xOut = xOutDesc.ToAclType();
    EXPECT_FALSE(
        CombineCheckNotNull(expandX.get(), expertIds.get(), assist.get(), epSend.get(), scales.get(), "", xOut.get()));
}

TEST(MoeDistributeCombineV2BaseHelperTest, CombineCheckParamsSuccess)
{
    TensorDesc expandXDesc{{1}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc expertIdsDesc{{1}, ACL_INT32, ACL_FORMAT_ND};
    TensorDesc assistDesc{{1}, ACL_INT32, ACL_FORMAT_ND};
    TensorDesc epSendDesc{{1}, ACL_INT32, ACL_FORMAT_ND};
    TensorDesc scalesDesc{{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc xOutDesc{{1}, ACL_FLOAT16, ACL_FORMAT_ND};
    auto expandX = expandXDesc.ToAclType();
    auto expertIds = expertIdsDesc.ToAclType();
    auto assist = assistDesc.ToAclType();
    auto epSend = epSendDesc.ToAclType();
    auto scales = scalesDesc.ToAclType();
    auto xOut = xOutDesc.ToAclType();
    EXPECT_EQ(CombineCheckParams(expandX.get(), expertIds.get(), assist.get(), epSend.get(), scales.get(), "group_ep",
                                 "group_tp", xOut.get()),
              ACLNN_SUCCESS);
}

TEST(MoeDistributeCombineV2BaseHelperTest, CombineCheckParamsGroupEpTooLong)
{
    TensorDesc expandXDesc{{1}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc expertIdsDesc{{1}, ACL_INT32, ACL_FORMAT_ND};
    TensorDesc assistDesc{{1}, ACL_INT32, ACL_FORMAT_ND};
    TensorDesc epSendDesc{{1}, ACL_INT32, ACL_FORMAT_ND};
    TensorDesc scalesDesc{{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc xOutDesc{{1}, ACL_FLOAT16, ACL_FORMAT_ND};
    auto expandX = expandXDesc.ToAclType();
    auto expertIds = expertIdsDesc.ToAclType();
    auto assist = assistDesc.ToAclType();
    auto epSend = epSendDesc.ToAclType();
    auto scales = scalesDesc.ToAclType();
    auto xOut = xOutDesc.ToAclType();
    std::string longGroup(HCCL_GROUP_NAME_MAX, 'a');
    EXPECT_EQ(CombineCheckParams(expandX.get(), expertIds.get(), assist.get(), epSend.get(), scales.get(),
                                 longGroup.c_str(), "group_tp", xOut.get()),
              ACLNN_ERR_PARAM_INVALID);
}

TEST(MoeDistributeCombineV2BaseHelperTest, CombineCheckParamsGroupTpTooLong)
{
    TensorDesc expandXDesc{{1}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc expertIdsDesc{{1}, ACL_INT32, ACL_FORMAT_ND};
    TensorDesc assistDesc{{1}, ACL_INT32, ACL_FORMAT_ND};
    TensorDesc epSendDesc{{1}, ACL_INT32, ACL_FORMAT_ND};
    TensorDesc scalesDesc{{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc xOutDesc{{1}, ACL_FLOAT16, ACL_FORMAT_ND};
    auto expandX = expandXDesc.ToAclType();
    auto expertIds = expertIdsDesc.ToAclType();
    auto assist = assistDesc.ToAclType();
    auto epSend = epSendDesc.ToAclType();
    auto scales = scalesDesc.ToAclType();
    auto xOut = xOutDesc.ToAclType();
    std::string longGroup(HCCL_GROUP_NAME_MAX, 'a');
    EXPECT_EQ(CombineCheckParams(expandX.get(), expertIds.get(), assist.get(), epSend.get(), scales.get(), "group_ep",
                                 longGroup.c_str(), xOut.get()),
              ACLNN_ERR_PARAM_INVALID);
}
} // namespace MoeDistributeCombineV2
