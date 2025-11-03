/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <vector>

#include <gmock/gmock.h>
#include "gtest/gtest.h"
#include "../../../../../../mc2/moe_distribute_combine_add_rms_norm/op_api/aclnn_moe_distribute_combine_add_rms_norm.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

namespace MoeDistributeCombineAddRmsNorm {
class l2_moe_distribute_combine_add_rms_norm_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "l2_moe_distribute_combine_add_rms_norm_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_moe_distribute_combine_add_rms_norm_test TearDown" << endl;
    }
};

TEST_F(l2_moe_distribute_combine_add_rms_norm_test, test_moe_distribute_combine_add_rms_norm_1)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910_93);
    TensorDesc expandX = TensorDesc({32, 7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc expertIds = TensorDesc({32, 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expandIdx = TensorDesc({32 * 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc epSendCounts = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expertScales = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc residualX = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gamma = TensorDesc({7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc tpSendCounts = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc xActiveMask = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND);
    TensorDesc activationScale = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc weightScale = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList = TensorDesc({8}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc expandScales = TensorDesc({32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc sharedExpertX = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);

    constexpr size_t MOE_GROUP_EP_LONG_STR_SIZE = 128;
    constexpr size_t MOE_GROUP_TP_LONG_STR_SIZE = 128;

    char groupEp[] = "test_moe_distribute_combine_add_rms_norm_ep";
    int64_t epWorldSize = 8;
    int64_t epRankId = 0;
    int64_t moeExpertNum = 256;
    char groupTp[] = "test_moe_distribute_combine_add_rms_norm_tp";
    int64_t tpWorldSize = 1;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t sharedExpertRankNum = 0;
    int64_t globalBs = 0;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupList_type = 0;
    float normEps = 1e-6;
    char commAlg[] = "";

    TensorDesc y = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rstdOut = TensorDesc({32, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);

    auto ut1 = OP_API_UT(
        aclnnMoeDistributeCombineAddRmsNorm,
        INPUT(expandX, expertIds, expandIdx, epSendCounts, expertScales, residualX, gamma, tpSendCounts, xActiveMask, activationScale,
              weightScale, groupList, expandScales, sharedExpertX, groupEp, epWorldSize,
              epRankId, moeExpertNum, groupTp, tpWorldSize, tpRankId, expertShardType,
              sharedExpertNum, sharedExpertRankNum, globalBs, outDtype, commQuantMode, groupList_type, commAlg, normEps),
        OUTPUT(y, rstdOut, x));
    uint64_t workspace_size1 = 0;
    aclOpExecutor* executor1 = nullptr;
    aclnnStatus aclRet1 = ut1.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size1, executor1);
    EXPECT_EQ(aclRet1, ACLNN_SUCCESS);

    auto ut2 = OP_API_UT(
        aclnnMoeDistributeCombineAddRmsNorm,
        INPUT(expandX, expertIds, expandIdx, epSendCounts, expertScales, residualX, gamma, tpSendCounts, xActiveMask, activationScale,
              weightScale, groupList, expandScales, sharedExpertX, nullptr, epWorldSize,
              epRankId, moeExpertNum, groupTp, tpWorldSize, tpRankId, expertShardType,
              sharedExpertNum, sharedExpertRankNum, globalBs, outDtype, commQuantMode, groupList_type, commAlg, normEps),
        OUTPUT(y, rstdOut, x));
    uint64_t workspace_size2 = 0;
    aclOpExecutor* executor2 = nullptr;
    aclnnStatus aclRet2 = ut2.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size2, executor2);
    EXPECT_EQ(aclRet2, ACLNN_ERR_PARAM_NULLPTR);

    std::string groupEpLongStr(MOE_GROUP_EP_LONG_STR_SIZE, 'a');
    auto ut3 = OP_API_UT(
        aclnnMoeDistributeCombineAddRmsNorm,
        INPUT(expandX, expertIds, expandIdx, epSendCounts, expertScales, residualX, gamma, tpSendCounts, xActiveMask, activationScale,
              weightScale, groupList, expandScales, sharedExpertX, groupEpLongStr.c_str(), epWorldSize,
              epRankId, moeExpertNum, groupTp, tpWorldSize, tpRankId, expertShardType,
              sharedExpertNum, sharedExpertRankNum, globalBs, outDtype, commQuantMode, groupList_type, commAlg, normEps),
        OUTPUT(y, rstdOut, x));
    uint64_t workspace_size3 = 0;
    aclOpExecutor* executor3 = nullptr;
    aclnnStatus aclRet3 = ut3.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size3, executor3);
    EXPECT_EQ(aclRet3, ACLNN_ERR_PARAM_INVALID);

    std::string groupTpLongStr(MOE_GROUP_TP_LONG_STR_SIZE, 'b');
    auto ut4 = OP_API_UT(
        aclnnMoeDistributeCombineAddRmsNorm,
        INPUT(expandX, expertIds, expandIdx, epSendCounts, expertScales, residualX, gamma, tpSendCounts, xActiveMask, activationScale,
              weightScale, groupList, expandScales, sharedExpertX, groupEp, epWorldSize,
              epRankId, moeExpertNum, groupTpLongStr.c_str(), tpWorldSize, tpRankId, expertShardType,
              sharedExpertNum, sharedExpertRankNum, globalBs, outDtype, commQuantMode, groupList_type, commAlg, normEps),
        OUTPUT(y, rstdOut, x));
    uint64_t workspace_size4 = 0;
    aclOpExecutor* executor4 = nullptr;
    aclnnStatus aclRet4 = ut4.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size4, executor4);
    EXPECT_EQ(aclRet4, ACLNN_ERR_PARAM_INVALID);
}
} // MoeDistributeCombineAddRmsNorm