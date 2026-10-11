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
#include "../../../op_api/aclnn_moe_update_expert.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

namespace MoeUpdateExpert {
class L2AclnnMoeUpdateExpertTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910_93);
        cout << "L2AclnnMoeUpdateExpertTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "L2AclnnMoeUpdateExpertTest TearDown" << endl;
    }
};

TEST_F(L2AclnnMoeUpdateExpertTest, TestMoeUpdateExpertNoTailor)
{
    TensorDesc expertIds = TensorDesc({50, 4}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc eplbTable = TensorDesc({256, 5}, ACL_INT32, ACL_FORMAT_ND);

    int64_t localRankId = 0;
    int64_t worldSize = 8;
    int64_t balanceMode = 0;

    TensorDesc balancedExpertIds = TensorDesc({50, 4}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc balancedActiveMask = TensorDesc({50, 4}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeUpdateExpert,
                        INPUT(expertIds, eplbTable, nullptr, nullptr, nullptr, localRankId, worldSize, balanceMode),
                        OUTPUT(balancedExpertIds, balancedActiveMask));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AclnnMoeUpdateExpertTest, TestMoeUpdateExpertExpertTailor)
{
    TensorDesc expertIds = TensorDesc({50, 4}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc eplbTable = TensorDesc({256, 5}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expertScales = TensorDesc({50, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc pruningThreshold = TensorDesc(
        {
            4,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc activeMask = TensorDesc(
        {
            50,
        },
        ACL_BOOL, ACL_FORMAT_ND);

    int64_t localRankId = 0;
    int64_t worldSize = 8;
    int64_t balanceMode = 0;

    TensorDesc balancedExpertIds = TensorDesc({50, 4}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc balancedActiveMask = TensorDesc({50, 4}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMoeUpdateExpert,
        INPUT(expertIds, eplbTable, expertScales, pruningThreshold, activeMask, localRankId, worldSize, balanceMode),
        OUTPUT(balancedExpertIds, balancedActiveMask));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AclnnMoeUpdateExpertTest, TestMoeUpdateExpertExecuteEntry)
{
    aclnnStatus ret = aclnnMoeUpdateExpert(nullptr, 0, nullptr, nullptr);
    EXPECT_THAT(ret, testing::AnyOf(testing::Eq(ACLNN_SUCCESS), testing::Eq(ACLNN_ERR_PARAM_NULLPTR),
                                    testing::Eq(ACLNN_ERR_PARAM_INVALID), testing::Eq(ACLNN_ERR_INNER)));
}

TEST_F(L2AclnnMoeUpdateExpertTest, TestMoeUpdateExpertNullExpertIds)
{
    TensorDesc eplbTable = TensorDesc({256, 5}, ACL_INT32, ACL_FORMAT_ND);
    int64_t localRankId = 0;
    int64_t worldSize = 8;
    int64_t balanceMode = 0;
    TensorDesc balancedExpertIds = TensorDesc({50, 4}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc balancedActiveMask = TensorDesc({50, 4}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeUpdateExpert,
                        INPUT(nullptr, eplbTable, nullptr, nullptr, nullptr, localRankId, worldSize, balanceMode),
                        OUTPUT(balancedExpertIds, balancedActiveMask));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2AclnnMoeUpdateExpertTest, TestMoeUpdateExpertNullEplbTable)
{
    TensorDesc expertIds = TensorDesc({50, 4}, ACL_INT32, ACL_FORMAT_ND);
    int64_t localRankId = 0;
    int64_t worldSize = 8;
    int64_t balanceMode = 0;
    TensorDesc balancedExpertIds = TensorDesc({50, 4}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc balancedActiveMask = TensorDesc({50, 4}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeUpdateExpert,
                        INPUT(expertIds, nullptr, nullptr, nullptr, nullptr, localRankId, worldSize, balanceMode),
                        OUTPUT(balancedExpertIds, balancedActiveMask));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2AclnnMoeUpdateExpertTest, TestMoeUpdateExpertNullBalancedExpertIds)
{
    TensorDesc expertIds = TensorDesc({50, 4}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc eplbTable = TensorDesc({256, 5}, ACL_INT32, ACL_FORMAT_ND);
    int64_t localRankId = 0;
    int64_t worldSize = 8;
    int64_t balanceMode = 0;
    TensorDesc balancedActiveMask = TensorDesc({50, 4}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeUpdateExpert,
                        INPUT(expertIds, eplbTable, nullptr, nullptr, nullptr, localRankId, worldSize, balanceMode),
                        OUTPUT(nullptr, balancedActiveMask));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2AclnnMoeUpdateExpertTest, TestMoeUpdateExpertNullBalancedActiveMask)
{
    TensorDesc expertIds = TensorDesc({50, 4}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc eplbTable = TensorDesc({256, 5}, ACL_INT32, ACL_FORMAT_ND);
    int64_t localRankId = 0;
    int64_t worldSize = 8;
    int64_t balanceMode = 0;
    TensorDesc balancedExpertIds = TensorDesc({50, 4}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeUpdateExpert,
                        INPUT(expertIds, eplbTable, nullptr, nullptr, nullptr, localRankId, worldSize, balanceMode),
                        OUTPUT(balancedExpertIds, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}
} // namespace MoeUpdateExpert
