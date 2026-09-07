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

#include <gmock/gmock.h>
#include "gtest/gtest.h"
#include "../../../op_api/aclnn_moe_distribute_combine_add_rms_norm.h"
#undef OP_API_INC_MOE_DISTRIBUTE_ADD_RMS_NORM_COMBINE
#include "../../../op_api/aclnn_moe_distribute_combine_add_rms_norm_v2.h"
#include "../../../op_api/moe_distribute_combine_add_rms_norm_base.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

namespace MoeDistributeCombineAddRmsNorm {
class L2MoeDistributeCombineAddRmsNormTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910_93);
        cout << "L2MoeDistributeCombineAddRmsNormTest SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "L2MoeDistributeCombineAddRmsNormTest TearDown" << endl;
    }
};

TEST_F(L2MoeDistributeCombineAddRmsNormTest, TestMoeDistributeCombineAddRmsNorm1)
{
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
    int64_t groupListType = 0;
    float normEps = 1e-6;
    char commAlg[] = "";

    TensorDesc y = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rstdOut = TensorDesc({32, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMoeDistributeCombineAddRmsNorm,
        INPUT(expandX, expertIds, expandIdx, epSendCounts, expertScales, residualX, gamma, tpSendCounts, xActiveMask,
              activationScale, weightScale, groupList, expandScales, sharedExpertX, groupEp, epWorldSize, epRankId,
              moeExpertNum, groupTp, tpWorldSize, tpRankId, expertShardType, sharedExpertNum, sharedExpertRankNum,
              globalBs, outDtype, commQuantMode, groupListType, commAlg, normEps),
        OUTPUT(y, rstdOut, x));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MoeDistributeCombineAddRmsNormTest, TestMoeDistributeCombineAddRmsNorm2)
{
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
    int64_t groupListType = 0;
    float normEps = 1e-6;
    char commAlg[] = "";

    TensorDesc y = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rstdOut = TensorDesc({32, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMoeDistributeCombineAddRmsNorm,
        INPUT(expandX, expertIds, expandIdx, epSendCounts, expertScales, residualX, gamma, tpSendCounts, xActiveMask,
              activationScale, weightScale, groupList, expandScales, sharedExpertX, nullptr, epWorldSize, epRankId,
              moeExpertNum, groupTp, tpWorldSize, tpRankId, expertShardType, sharedExpertNum, sharedExpertRankNum,
              globalBs, outDtype, commQuantMode, groupListType, commAlg, normEps),
        OUTPUT(y, rstdOut, x));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2MoeDistributeCombineAddRmsNormTest, TestMoeDistributeCombineAddRmsNorm3)
{
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
    std::string groupEpLongStr(MOE_GROUP_EP_LONG_STR_SIZE, 'a');
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
    int64_t groupListType = 0;
    float normEps = 1e-6;
    char commAlg[] = "";

    TensorDesc y = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rstdOut = TensorDesc({32, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMoeDistributeCombineAddRmsNorm,
        INPUT(expandX, expertIds, expandIdx, epSendCounts, expertScales, residualX, gamma, tpSendCounts, xActiveMask,
              activationScale, weightScale, groupList, expandScales, sharedExpertX, groupEpLongStr.c_str(), epWorldSize,
              epRankId, moeExpertNum, groupTp, tpWorldSize, tpRankId, expertShardType, sharedExpertNum,
              sharedExpertRankNum, globalBs, outDtype, commQuantMode, groupListType, commAlg, normEps),
        OUTPUT(y, rstdOut, x));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MoeDistributeCombineAddRmsNormTest, TestMoeDistributeCombineAddRmsNorm4)
{
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

    constexpr size_t MOE_GROUP_TP_LONG_STR_SIZE = 128;
    char groupEp[] = "test_moe_distribute_combine_add_rms_norm_ep";
    int64_t epWorldSize = 8;
    int64_t epRankId = 0;
    int64_t moeExpertNum = 256;
    std::string groupTpLongStr(MOE_GROUP_TP_LONG_STR_SIZE, 'b');
    int64_t tpWorldSize = 1;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t sharedExpertRankNum = 0;
    int64_t globalBs = 0;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;
    float normEps = 1e-6;
    char commAlg[] = "";

    TensorDesc y = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rstdOut = TensorDesc({32, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMoeDistributeCombineAddRmsNorm,
        INPUT(expandX, expertIds, expandIdx, epSendCounts, expertScales, residualX, gamma, tpSendCounts, xActiveMask,
              activationScale, weightScale, groupList, expandScales, sharedExpertX, groupEp, epWorldSize, epRankId,
              moeExpertNum, groupTpLongStr.c_str(), tpWorldSize, tpRankId, expertShardType, sharedExpertNum,
              sharedExpertRankNum, globalBs, outDtype, commQuantMode, groupListType, commAlg, normEps),
        OUTPUT(y, rstdOut, x));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MoeDistributeCombineAddRmsNormTest, TestMoeDistributeCombineAddRmsNormV2_1)
{
    TensorDesc expandX = TensorDesc({32, 7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc expertIds = TensorDesc({32, 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expandIdx = TensorDesc({32 * 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc epSendCounts = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expertScales = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc residualX = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gamma = TensorDesc({7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc tpSendCounts = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc sharedExpertX = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);

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
    int64_t groupListType = 0;
    float normEps = 1e-6;
    char commAlg[] = "";
    int64_t zeroExpertNum = 0;
    int64_t copyExpertNum = 0;
    int64_t constExpertNum = 0;

    TensorDesc y = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rstdOut = TensorDesc({32, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc xOut = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnMoeDistributeCombineAddRmsNormV2,
                  INPUT(expandX, expertIds, expandIdx, epSendCounts, expertScales, residualX, gamma, tpSendCounts,
                        nullptr, nullptr, nullptr, nullptr, nullptr, sharedExpertX, nullptr, nullptr, nullptr, nullptr,
                        nullptr, groupEp, epWorldSize, epRankId, moeExpertNum, groupTp, tpWorldSize, tpRankId,
                        expertShardType, sharedExpertNum, sharedExpertRankNum, globalBs, outDtype, commQuantMode,
                        groupListType, commAlg, normEps, zeroExpertNum, copyExpertNum, constExpertNum),
                  OUTPUT(y, rstdOut, xOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MoeDistributeCombineAddRmsNormTest, TestMoeDistributeCombineAddRmsNormV2_NullGroupEp)
{
    TensorDesc expandX = TensorDesc({32, 7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc expertIds = TensorDesc({32, 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expandIdx = TensorDesc({32 * 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc epSendCounts = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expertScales = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc residualX = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gamma = TensorDesc({7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc tpSendCounts = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc sharedExpertX = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);

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
    int64_t groupListType = 0;
    float normEps = 1e-6;
    char commAlg[] = "";
    int64_t zeroExpertNum = 0;
    int64_t copyExpertNum = 0;
    int64_t constExpertNum = 0;

    TensorDesc y = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rstdOut = TensorDesc({32, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc xOut = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnMoeDistributeCombineAddRmsNormV2,
                  INPUT(expandX, expertIds, expandIdx, epSendCounts, expertScales, residualX, gamma, tpSendCounts,
                        nullptr, nullptr, nullptr, nullptr, nullptr, sharedExpertX, nullptr, nullptr, nullptr, nullptr,
                        nullptr, nullptr, epWorldSize, epRankId, moeExpertNum, groupTp, tpWorldSize, tpRankId,
                        expertShardType, sharedExpertNum, sharedExpertRankNum, globalBs, outDtype, commQuantMode,
                        groupListType, commAlg, normEps, zeroExpertNum, copyExpertNum, constExpertNum),
                  OUTPUT(y, rstdOut, xOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2MoeDistributeCombineAddRmsNormTest, TestMoeDistributeCombineAddRmsNormV2_LongGroupEp)
{
    TensorDesc expandX = TensorDesc({32, 7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc expertIds = TensorDesc({32, 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expandIdx = TensorDesc({32 * 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc epSendCounts = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expertScales = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc residualX = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gamma = TensorDesc({7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc tpSendCounts = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc sharedExpertX = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);

    char groupTp[] = "test_moe_distribute_combine_add_rms_norm_tp";
    int64_t epWorldSize = 8;
    int64_t epRankId = 0;
    int64_t moeExpertNum = 256;
    int64_t tpWorldSize = 1;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t sharedExpertRankNum = 0;
    int64_t globalBs = 0;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;
    float normEps = 1e-6;
    char commAlg[] = "";
    int64_t zeroExpertNum = 0;
    int64_t copyExpertNum = 0;
    int64_t constExpertNum = 0;

    TensorDesc y = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rstdOut = TensorDesc({32, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc xOut = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);

    constexpr size_t MOE_GROUP_EP_LONG_STR_SIZE = 128;
    std::string groupEpLongStr(MOE_GROUP_EP_LONG_STR_SIZE, 'a');

    auto ut =
        OP_API_UT(aclnnMoeDistributeCombineAddRmsNormV2,
                  INPUT(expandX, expertIds, expandIdx, epSendCounts, expertScales, residualX, gamma, tpSendCounts,
                        nullptr, nullptr, nullptr, nullptr, nullptr, sharedExpertX, nullptr, nullptr, nullptr, nullptr,
                        nullptr, groupEpLongStr.c_str(), epWorldSize, epRankId, moeExpertNum, groupTp, tpWorldSize,
                        tpRankId, expertShardType, sharedExpertNum, sharedExpertRankNum, globalBs, outDtype,
                        commQuantMode, groupListType, commAlg, normEps, zeroExpertNum, copyExpertNum, constExpertNum),
                  OUTPUT(y, rstdOut, xOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MoeDistributeCombineAddRmsNormTest, TestMoeDistributeCombineAddRmsNormV2_LongGroupTp)
{
    TensorDesc expandX = TensorDesc({32, 7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc expertIds = TensorDesc({32, 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expandIdx = TensorDesc({32 * 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc epSendCounts = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expertScales = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc residualX = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gamma = TensorDesc({7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc tpSendCounts = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc sharedExpertX = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);

    char groupEp[] = "test_moe_distribute_combine_add_rms_norm_ep";
    int64_t epWorldSize = 8;
    int64_t epRankId = 0;
    int64_t moeExpertNum = 256;
    int64_t tpWorldSize = 1;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t sharedExpertRankNum = 0;
    int64_t globalBs = 0;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;
    float normEps = 1e-6;
    char commAlg[] = "";
    int64_t zeroExpertNum = 0;
    int64_t copyExpertNum = 0;
    int64_t constExpertNum = 0;

    TensorDesc y = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rstdOut = TensorDesc({32, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc xOut = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);

    constexpr size_t MOE_GROUP_TP_LONG_STR_SIZE = 128;
    std::string groupTpLongStr(MOE_GROUP_TP_LONG_STR_SIZE, 'b');

    auto ut =
        OP_API_UT(aclnnMoeDistributeCombineAddRmsNormV2,
                  INPUT(expandX, expertIds, expandIdx, epSendCounts, expertScales, residualX, gamma, tpSendCounts,
                        nullptr, nullptr, nullptr, nullptr, nullptr, sharedExpertX, nullptr, nullptr, nullptr, nullptr,
                        nullptr, groupEp, epWorldSize, epRankId, moeExpertNum, groupTpLongStr.c_str(), tpWorldSize,
                        tpRankId, expertShardType, sharedExpertNum, sharedExpertRankNum, globalBs, outDtype,
                        commQuantMode, groupListType, commAlg, normEps, zeroExpertNum, copyExpertNum, constExpertNum),
                  OUTPUT(y, rstdOut, xOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MoeDistributeCombineAddRmsNormTest, TestMoeDistributeCombineAddRmsNorm_NullExpandX)
{
    TensorDesc expertIds = TensorDesc({32, 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expandIdx = TensorDesc({32 * 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc epSendCounts = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expertScales = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc residualX = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gamma = TensorDesc({7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc tpSendCounts = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc sharedExpertX = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);

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
    int64_t groupListType = 0;
    float normEps = 1e-6;
    char commAlg[] = "";

    TensorDesc y = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rstdOut = TensorDesc({32, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc xOut = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeDistributeCombineAddRmsNorm,
                        INPUT(nullptr, expertIds, expandIdx, epSendCounts, expertScales, residualX, gamma, tpSendCounts,
                              nullptr, nullptr, nullptr, nullptr, nullptr, sharedExpertX, groupEp, epWorldSize,
                              epRankId, moeExpertNum, groupTp, tpWorldSize, tpRankId, expertShardType, sharedExpertNum,
                              sharedExpertRankNum, globalBs, outDtype, commQuantMode, groupListType, commAlg, normEps),
                        OUTPUT(y, rstdOut, xOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2MoeDistributeCombineAddRmsNormTest, TestMoeDistributeCombineAddRmsNorm_NullExpertIds)
{
    TensorDesc expandX = TensorDesc({32, 7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc expandIdx = TensorDesc({32 * 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc epSendCounts = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expertScales = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc residualX = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gamma = TensorDesc({7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc tpSendCounts = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc sharedExpertX = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);

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
    int64_t groupListType = 0;
    float normEps = 1e-6;
    char commAlg[] = "";

    TensorDesc y = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rstdOut = TensorDesc({32, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc xOut = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeDistributeCombineAddRmsNorm,
                        INPUT(expandX, nullptr, expandIdx, epSendCounts, expertScales, residualX, gamma, tpSendCounts,
                              nullptr, nullptr, nullptr, nullptr, nullptr, sharedExpertX, groupEp, epWorldSize,
                              epRankId, moeExpertNum, groupTp, tpWorldSize, tpRankId, expertShardType, sharedExpertNum,
                              sharedExpertRankNum, globalBs, outDtype, commQuantMode, groupListType, commAlg, normEps),
                        OUTPUT(y, rstdOut, xOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2MoeDistributeCombineAddRmsNormTest, TestMoeDistributeCombineAddRmsNorm_NullAssistInfoForCombine)
{
    TensorDesc expandX = TensorDesc({32, 7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc expertIds = TensorDesc({32, 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc epSendCounts = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expertScales = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc residualX = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gamma = TensorDesc({7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc tpSendCounts = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc sharedExpertX = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);

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
    int64_t groupListType = 0;
    float normEps = 1e-6;
    char commAlg[] = "";

    TensorDesc y = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rstdOut = TensorDesc({32, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc xOut = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeDistributeCombineAddRmsNorm,
                        INPUT(expandX, expertIds, nullptr, epSendCounts, expertScales, residualX, gamma, tpSendCounts,
                              nullptr, nullptr, nullptr, nullptr, nullptr, sharedExpertX, groupEp, epWorldSize,
                              epRankId, moeExpertNum, groupTp, tpWorldSize, tpRankId, expertShardType, sharedExpertNum,
                              sharedExpertRankNum, globalBs, outDtype, commQuantMode, groupListType, commAlg, normEps),
                        OUTPUT(y, rstdOut, xOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2MoeDistributeCombineAddRmsNormTest, TestMoeDistributeCombineAddRmsNorm_NullEpSendCounts)
{
    TensorDesc expandX = TensorDesc({32, 7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc expertIds = TensorDesc({32, 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expandIdx = TensorDesc({32 * 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expertScales = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc residualX = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gamma = TensorDesc({7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc tpSendCounts = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc sharedExpertX = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);

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
    int64_t groupListType = 0;
    float normEps = 1e-6;
    char commAlg[] = "";

    TensorDesc y = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rstdOut = TensorDesc({32, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc xOut = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeDistributeCombineAddRmsNorm,
                        INPUT(expandX, expertIds, expandIdx, nullptr, expertScales, residualX, gamma, tpSendCounts,
                              nullptr, nullptr, nullptr, nullptr, nullptr, sharedExpertX, groupEp, epWorldSize,
                              epRankId, moeExpertNum, groupTp, tpWorldSize, tpRankId, expertShardType, sharedExpertNum,
                              sharedExpertRankNum, globalBs, outDtype, commQuantMode, groupListType, commAlg, normEps),
                        OUTPUT(y, rstdOut, xOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2MoeDistributeCombineAddRmsNormTest, TestMoeDistributeCombineAddRmsNorm_NullExpertScales)
{
    TensorDesc expandX = TensorDesc({32, 7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc expertIds = TensorDesc({32, 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expandIdx = TensorDesc({32 * 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc epSendCounts = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc residualX = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gamma = TensorDesc({7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc tpSendCounts = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc sharedExpertX = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);

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
    int64_t groupListType = 0;
    float normEps = 1e-6;
    char commAlg[] = "";

    TensorDesc y = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rstdOut = TensorDesc({32, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc xOut = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeDistributeCombineAddRmsNorm,
                        INPUT(expandX, expertIds, expandIdx, epSendCounts, nullptr, residualX, gamma, tpSendCounts,
                              nullptr, nullptr, nullptr, nullptr, nullptr, sharedExpertX, groupEp, epWorldSize,
                              epRankId, moeExpertNum, groupTp, tpWorldSize, tpRankId, expertShardType, sharedExpertNum,
                              sharedExpertRankNum, globalBs, outDtype, commQuantMode, groupListType, commAlg, normEps),
                        OUTPUT(y, rstdOut, xOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2MoeDistributeCombineAddRmsNormTest, TestMoeDistributeCombineAddRmsNorm_NullXOut)
{
    TensorDesc expandX = TensorDesc({32, 7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc expertIds = TensorDesc({32, 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expandIdx = TensorDesc({32 * 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc epSendCounts = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expertScales = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc residualX = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gamma = TensorDesc({7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc tpSendCounts = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc sharedExpertX = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);

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
    int64_t groupListType = 0;
    float normEps = 1e-6;
    char commAlg[] = "";

    TensorDesc y = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rstdOut = TensorDesc({32, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeDistributeCombineAddRmsNorm,
                        INPUT(expandX, expertIds, expandIdx, epSendCounts, expertScales, residualX, gamma, tpSendCounts,
                              nullptr, nullptr, nullptr, nullptr, nullptr, sharedExpertX, groupEp, epWorldSize,
                              epRankId, moeExpertNum, groupTp, tpWorldSize, tpRankId, expertShardType, sharedExpertNum,
                              sharedExpertRankNum, globalBs, outDtype, commQuantMode, groupListType, commAlg, normEps),
                        OUTPUT(y, rstdOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2MoeDistributeCombineAddRmsNormTest, TestMoeDistributeCombineAddRmsNorm_EmptyGroupEp)
{
    TensorDesc expandX = TensorDesc({32, 7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc expertIds = TensorDesc({32, 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expandIdx = TensorDesc({32 * 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc epSendCounts = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expertScales = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc residualX = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gamma = TensorDesc({7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc tpSendCounts = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc sharedExpertX = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);

    char groupEp[] = "";
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
    int64_t groupListType = 0;
    float normEps = 1e-6;
    char commAlg[] = "";

    TensorDesc y = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rstdOut = TensorDesc({32, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc xOut = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeDistributeCombineAddRmsNorm,
                        INPUT(expandX, expertIds, expandIdx, epSendCounts, expertScales, residualX, gamma, tpSendCounts,
                              nullptr, nullptr, nullptr, nullptr, nullptr, sharedExpertX, groupEp, epWorldSize,
                              epRankId, moeExpertNum, groupTp, tpWorldSize, tpRankId, expertShardType, sharedExpertNum,
                              sharedExpertRankNum, globalBs, outDtype, commQuantMode, groupListType, commAlg, normEps),
                        OUTPUT(y, rstdOut, xOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}
// 测试执行函数覆盖和 is910B 分支
class L2MoeDistributeCombineAddRmsNorm910BTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "L2MoeDistributeCombineAddRmsNorm910BTest SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "L2MoeDistributeCombineAddRmsNorm910BTest TearDown" << endl;
    }
};

TEST_F(L2MoeDistributeCombineAddRmsNorm910BTest, TestExecuteV1_910B)
{
    aclnnStatus aclRet = aclnnMoeDistributeCombineAddRmsNorm(nullptr, 0, nullptr, nullptr);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MoeDistributeCombineAddRmsNorm910BTest, TestExecuteV2_910B)
{
    aclnnStatus aclRet = aclnnMoeDistributeCombineAddRmsNormV2(nullptr, 0, nullptr, nullptr);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MoeDistributeCombineAddRmsNorm910BTest, TestCombineArnCheckParams_910B)
{
    TensorDesc expandX = TensorDesc({32, 7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc expertIds = TensorDesc({32, 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expandIdx = TensorDesc({32 * 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc epSendCounts = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expertScales = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    char groupEp[] = "test_moe_distribute_combine_add_rms_norm_ep";
    char groupTp[] = "test_moe_distribute_combine_add_rms_norm_tp";
    TensorDesc xOut = TensorDesc({32, 1, 7168}, ACL_BF16, ACL_FORMAT_ND);

    aclnnStatus aclRet =
        CombineArnCheckParams(DescToAclContainer(expandX), DescToAclContainer(expertIds), DescToAclContainer(expandIdx),
                              DescToAclContainer(epSendCounts), nullptr, DescToAclContainer(expertScales), groupEp,
                              groupTp, DescToAclContainer(xOut), true);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

} // namespace MoeDistributeCombineAddRmsNorm
