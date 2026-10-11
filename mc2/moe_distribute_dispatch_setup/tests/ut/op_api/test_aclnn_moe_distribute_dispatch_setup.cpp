/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <float.h>
#include <array>
#include <string>
#include <vector>
#include <gmock/gmock.h>
#include "gtest/gtest.h"
#include "../../../op_api/aclnn_moe_distribute_dispatch_setup.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

namespace MoeDistributeDispatchSetup {

static aclTensor* CreateAclTensor(const std::vector<int64_t> shape, aclDataType dataType, aclFormat format)
{
    void* storage_data = nullptr;
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format, shape.data(),
                                        shape.size(), storage_data);
    assert(tensor != nullptr);
    return tensor;
}

static aclTensor* CreateAclTensorOrNull(const std::vector<int64_t> shape, aclDataType dataType, aclFormat format)
{
    if (shape.empty()) {
        return nullptr;
    }
    return CreateAclTensor(shape, dataType, format);
}

class L2MoeDistributeDispatchSetupTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "L2MoeDistributeDispatchSetupTest SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "L2MoeDistributeDispatchSetupTest TearDown" << endl;
    }
};

TEST_F(L2MoeDistributeDispatchSetupTest, TestTeardownCalcOutputSizeNormal)
{
    aclTensor* x = CreateAclTensor({8, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    aclTensor* expertIds = CreateAclTensor({8, 8}, ACL_INT32, ACL_FORMAT_ND);
    aclTensor* scalesOptional = nullptr;
    aclTensor* xActiveMaskOptional = nullptr;

    int64_t epWorldSize = 2;
    int64_t epRankId = 0;
    int64_t moeExpertNum = 32;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 0;
    int64_t sharedExpertRankNum = 0;
    int64_t quantMode = 0;
    int64_t globalBs = 0;
    int64_t expertTokenNumsType = 1;
    int64_t commType = 2;

    uint64_t tokenMsgSize = 0;
    uint64_t expandIdxOutSize = 0;
    uint64_t assistInfoForCombineOutSize = 0;
    uint64_t commCmdInfoOutSize = 0;

    aclnnStatus aclRet = aclnnMoeDistributeDispatchSetupTeardownCalcOutputSize(
        x, expertIds, scalesOptional, xActiveMaskOptional, "group_ep", epWorldSize, epRankId, moeExpertNum,
        expertShardType, sharedExpertNum, sharedExpertRankNum, quantMode, globalBs, expertTokenNumsType, commType, "",
        tokenMsgSize, expandIdxOutSize, assistInfoForCombineOutSize, commCmdInfoOutSize);

    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    EXPECT_GT(tokenMsgSize, 0);
    EXPECT_GT(expandIdxOutSize, 0);
    EXPECT_GT(assistInfoForCombineOutSize, 0);
    EXPECT_GT(commCmdInfoOutSize, 0);
}

TEST_F(L2MoeDistributeDispatchSetupTest, TestTeardownCalcOutputSizeQuantMode)
{
    aclTensor* x = CreateAclTensor({8, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    aclTensor* expertIds = CreateAclTensor({8, 8}, ACL_INT32, ACL_FORMAT_ND);
    aclTensor* scalesOptional = nullptr;
    aclTensor* xActiveMaskOptional = nullptr;

    uint64_t tokenMsgSize = 0;
    uint64_t expandIdxOutSize = 0;
    uint64_t assistInfoForCombineOutSize = 0;
    uint64_t commCmdInfoOutSize = 0;

    aclnnStatus aclRet = aclnnMoeDistributeDispatchSetupTeardownCalcOutputSize(
        x, expertIds, scalesOptional, xActiveMaskOptional, "group_ep", 2, 0, 32, 0, 0, 0, 2, 0, 1, 2, "", tokenMsgSize,
        expandIdxOutSize, assistInfoForCombineOutSize, commCmdInfoOutSize);

    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    EXPECT_GT(tokenMsgSize, 0);
}

TEST_F(L2MoeDistributeDispatchSetupTest, TestTeardownCalcOutputSizeNullX)
{
    aclTensor* x = nullptr;
    aclTensor* expertIds = CreateAclTensor({8, 8}, ACL_INT32, ACL_FORMAT_ND);
    aclTensor* scalesOptional = nullptr;
    aclTensor* xActiveMaskOptional = nullptr;

    uint64_t tokenMsgSize = 0;
    uint64_t expandIdxOutSize = 0;
    uint64_t assistInfoForCombineOutSize = 0;
    uint64_t commCmdInfoOutSize = 0;

    aclnnStatus aclRet = aclnnMoeDistributeDispatchSetupTeardownCalcOutputSize(
        x, expertIds, scalesOptional, xActiveMaskOptional, "group_ep", 2, 0, 32, 0, 0, 0, 0, 0, 1, 2, "", tokenMsgSize,
        expandIdxOutSize, assistInfoForCombineOutSize, commCmdInfoOutSize);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2MoeDistributeDispatchSetupTest, TestTeardownCalcOutputSizeNullExpertIds)
{
    aclTensor* x = CreateAclTensor({8, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    aclTensor* expertIds = nullptr;
    aclTensor* scalesOptional = nullptr;
    aclTensor* xActiveMaskOptional = nullptr;

    uint64_t tokenMsgSize = 0;
    uint64_t expandIdxOutSize = 0;
    uint64_t assistInfoForCombineOutSize = 0;
    uint64_t commCmdInfoOutSize = 0;

    aclnnStatus aclRet = aclnnMoeDistributeDispatchSetupTeardownCalcOutputSize(
        x, expertIds, scalesOptional, xActiveMaskOptional, "group_ep", 2, 0, 32, 0, 0, 0, 0, 0, 1, 2, "", tokenMsgSize,
        expandIdxOutSize, assistInfoForCombineOutSize, commCmdInfoOutSize);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2MoeDistributeDispatchSetupTest, TestTeardownCalcOutputSizeInvalidXDim)
{
    aclTensor* x = CreateAclTensor({8}, ACL_FLOAT16, ACL_FORMAT_ND);
    aclTensor* expertIds = CreateAclTensor({8, 8}, ACL_INT32, ACL_FORMAT_ND);
    aclTensor* scalesOptional = nullptr;
    aclTensor* xActiveMaskOptional = nullptr;

    uint64_t tokenMsgSize = 0;
    uint64_t expandIdxOutSize = 0;
    uint64_t assistInfoForCombineOutSize = 0;
    uint64_t commCmdInfoOutSize = 0;

    aclnnStatus aclRet = aclnnMoeDistributeDispatchSetupTeardownCalcOutputSize(
        x, expertIds, scalesOptional, xActiveMaskOptional, "group_ep", 2, 0, 32, 0, 0, 0, 0, 0, 1, 2, "", tokenMsgSize,
        expandIdxOutSize, assistInfoForCombineOutSize, commCmdInfoOutSize);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MoeDistributeDispatchSetupTest, TestTeardownCalcOutputSizeInvalidExpertIdsDim)
{
    aclTensor* x = CreateAclTensor({8, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    aclTensor* expertIds = CreateAclTensor({8}, ACL_INT32, ACL_FORMAT_ND);
    aclTensor* scalesOptional = nullptr;
    aclTensor* xActiveMaskOptional = nullptr;

    uint64_t tokenMsgSize = 0;
    uint64_t expandIdxOutSize = 0;
    uint64_t assistInfoForCombineOutSize = 0;
    uint64_t commCmdInfoOutSize = 0;

    aclnnStatus aclRet = aclnnMoeDistributeDispatchSetupTeardownCalcOutputSize(
        x, expertIds, scalesOptional, xActiveMaskOptional, "group_ep", 2, 0, 32, 0, 0, 0, 0, 0, 1, 2, "", tokenMsgSize,
        expandIdxOutSize, assistInfoForCombineOutSize, commCmdInfoOutSize);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MoeDistributeDispatchSetupTest, TestTeardownCalcOutputSizeInvalidSharedExpertRankNum)
{
    aclTensor* x = CreateAclTensor({8, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    aclTensor* expertIds = CreateAclTensor({8, 8}, ACL_INT32, ACL_FORMAT_ND);
    aclTensor* scalesOptional = nullptr;
    aclTensor* xActiveMaskOptional = nullptr;

    uint64_t tokenMsgSize = 0;
    uint64_t expandIdxOutSize = 0;
    uint64_t assistInfoForCombineOutSize = 0;
    uint64_t commCmdInfoOutSize = 0;

    aclnnStatus aclRet = aclnnMoeDistributeDispatchSetupTeardownCalcOutputSize(
        x, expertIds, scalesOptional, xActiveMaskOptional, "group_ep", 2, 0, 32, 0, 0, 3, 0, 0, 1, 2, "", tokenMsgSize,
        expandIdxOutSize, assistInfoForCombineOutSize, commCmdInfoOutSize);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MoeDistributeDispatchSetupTest, TestTeardownCalcOutputSizeInvalidEpRankIdNegative)
{
    aclTensor* x = CreateAclTensor({8, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    aclTensor* expertIds = CreateAclTensor({8, 8}, ACL_INT32, ACL_FORMAT_ND);
    aclTensor* scalesOptional = nullptr;
    aclTensor* xActiveMaskOptional = nullptr;

    uint64_t tokenMsgSize = 0;
    uint64_t expandIdxOutSize = 0;
    uint64_t assistInfoForCombineOutSize = 0;
    uint64_t commCmdInfoOutSize = 0;

    aclnnStatus aclRet = aclnnMoeDistributeDispatchSetupTeardownCalcOutputSize(
        x, expertIds, scalesOptional, xActiveMaskOptional, "group_ep", 2, -1, 32, 0, 0, 0, 0, 0, 1, 2, "", tokenMsgSize,
        expandIdxOutSize, assistInfoForCombineOutSize, commCmdInfoOutSize);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MoeDistributeDispatchSetupTest, TestGetWorkspaceSizeNullX)
{
    aclTensor* x = nullptr;
    aclTensor* expertIds = CreateAclTensor({8, 8}, ACL_INT32, ACL_FORMAT_ND);
    aclTensor* scalesOptional = nullptr;
    aclTensor* xActiveMaskOptional = nullptr;
    aclTensor* yOut = CreateAclTensor({64, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    aclTensor* expandIdxOut = CreateAclTensor({64}, ACL_INT32, ACL_FORMAT_ND);
    aclTensor* commCmdInfoOut = CreateAclTensor({1312}, ACL_INT32, ACL_FORMAT_ND);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclRet = aclnnMoeDistributeDispatchSetupGetWorkspaceSize(
        x, expertIds, scalesOptional, xActiveMaskOptional, "group_ep", 2, 0, 32, 0, 0, 0, 0, 0, 2, "", yOut,
        expandIdxOut, commCmdInfoOut, &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2MoeDistributeDispatchSetupTest, TestGetWorkspaceSizeNullExpertIds)
{
    aclTensor* x = CreateAclTensor({8, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    aclTensor* expertIds = nullptr;
    aclTensor* scalesOptional = nullptr;
    aclTensor* xActiveMaskOptional = nullptr;
    aclTensor* yOut = CreateAclTensor({64, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    aclTensor* expandIdxOut = CreateAclTensor({64}, ACL_INT32, ACL_FORMAT_ND);
    aclTensor* commCmdInfoOut = CreateAclTensor({1312}, ACL_INT32, ACL_FORMAT_ND);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclRet = aclnnMoeDistributeDispatchSetupGetWorkspaceSize(
        x, expertIds, scalesOptional, xActiveMaskOptional, "group_ep", 2, 0, 32, 0, 0, 0, 0, 0, 2, "", yOut,
        expandIdxOut, commCmdInfoOut, &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2MoeDistributeDispatchSetupTest, TestGetWorkspaceSizeNullYOut)
{
    aclTensor* x = CreateAclTensor({8, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    aclTensor* expertIds = CreateAclTensor({8, 8}, ACL_INT32, ACL_FORMAT_ND);
    aclTensor* scalesOptional = nullptr;
    aclTensor* xActiveMaskOptional = nullptr;
    aclTensor* yOut = nullptr;
    aclTensor* expandIdxOut = CreateAclTensor({64}, ACL_INT32, ACL_FORMAT_ND);
    aclTensor* commCmdInfoOut = CreateAclTensor({1312}, ACL_INT32, ACL_FORMAT_ND);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclRet = aclnnMoeDistributeDispatchSetupGetWorkspaceSize(
        x, expertIds, scalesOptional, xActiveMaskOptional, "group_ep", 2, 0, 32, 0, 0, 0, 0, 0, 2, "", yOut,
        expandIdxOut, commCmdInfoOut, &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2MoeDistributeDispatchSetupTest, TestGetWorkspaceSizeNullExpandIdxOut)
{
    aclTensor* x = CreateAclTensor({8, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    aclTensor* expertIds = CreateAclTensor({8, 8}, ACL_INT32, ACL_FORMAT_ND);
    aclTensor* scalesOptional = nullptr;
    aclTensor* xActiveMaskOptional = nullptr;
    aclTensor* yOut = CreateAclTensor({64, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    aclTensor* expandIdxOut = nullptr;
    aclTensor* commCmdInfoOut = CreateAclTensor({1312}, ACL_INT32, ACL_FORMAT_ND);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclRet = aclnnMoeDistributeDispatchSetupGetWorkspaceSize(
        x, expertIds, scalesOptional, xActiveMaskOptional, "group_ep", 2, 0, 32, 0, 0, 0, 0, 0, 2, "", yOut,
        expandIdxOut, commCmdInfoOut, &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2MoeDistributeDispatchSetupTest, TestGetWorkspaceSizeNullCommCmdInfoOut)
{
    aclTensor* x = CreateAclTensor({8, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    aclTensor* expertIds = CreateAclTensor({8, 8}, ACL_INT32, ACL_FORMAT_ND);
    aclTensor* scalesOptional = nullptr;
    aclTensor* xActiveMaskOptional = nullptr;
    aclTensor* yOut = CreateAclTensor({64, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    aclTensor* expandIdxOut = CreateAclTensor({64}, ACL_INT32, ACL_FORMAT_ND);
    aclTensor* commCmdInfoOut = nullptr;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclRet = aclnnMoeDistributeDispatchSetupGetWorkspaceSize(
        x, expertIds, scalesOptional, xActiveMaskOptional, "group_ep", 2, 0, 32, 0, 0, 0, 0, 0, 2, "", yOut,
        expandIdxOut, commCmdInfoOut, &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2MoeDistributeDispatchSetupTest, TestGetWorkspaceSizeNullGroupEp)
{
    aclTensor* x = CreateAclTensor({8, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    aclTensor* expertIds = CreateAclTensor({8, 8}, ACL_INT32, ACL_FORMAT_ND);
    aclTensor* scalesOptional = nullptr;
    aclTensor* xActiveMaskOptional = nullptr;
    aclTensor* yOut = CreateAclTensor({64, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    aclTensor* expandIdxOut = CreateAclTensor({64}, ACL_INT32, ACL_FORMAT_ND);
    aclTensor* commCmdInfoOut = CreateAclTensor({1312}, ACL_INT32, ACL_FORMAT_ND);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclRet = aclnnMoeDistributeDispatchSetupGetWorkspaceSize(
        x, expertIds, scalesOptional, xActiveMaskOptional, nullptr, 2, 0, 32, 0, 0, 0, 0, 0, 2, "", yOut, expandIdxOut,
        commCmdInfoOut, &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2MoeDistributeDispatchSetupTest, TestGetWorkspaceSizeEmptyGroupEp)
{
    aclTensor* x = CreateAclTensor({8, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    aclTensor* expertIds = CreateAclTensor({8, 8}, ACL_INT32, ACL_FORMAT_ND);
    aclTensor* scalesOptional = nullptr;
    aclTensor* xActiveMaskOptional = nullptr;
    aclTensor* yOut = CreateAclTensor({64, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    aclTensor* expandIdxOut = CreateAclTensor({64}, ACL_INT32, ACL_FORMAT_ND);
    aclTensor* commCmdInfoOut = CreateAclTensor({1312}, ACL_INT32, ACL_FORMAT_ND);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclRet = aclnnMoeDistributeDispatchSetupGetWorkspaceSize(
        x, expertIds, scalesOptional, xActiveMaskOptional, "", 2, 0, 32, 0, 0, 0, 0, 0, 2, "", yOut, expandIdxOut,
        commCmdInfoOut, &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}
TEST_F(L2MoeDistributeDispatchSetupTest, TestTeardownCalcOutputSizeSharedExpert)
{
    aclTensor* x = CreateAclTensor({8, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    aclTensor* expertIds = CreateAclTensor({8, 8}, ACL_INT32, ACL_FORMAT_ND);
    aclTensor* scalesOptional = nullptr;
    aclTensor* xActiveMaskOptional = nullptr;

    uint64_t tokenMsgSize = 0;
    uint64_t expandIdxOutSize = 0;
    uint64_t assistInfoForCombineOutSize = 0;
    uint64_t commCmdInfoOutSize = 0;

    aclnnStatus aclRet = aclnnMoeDistributeDispatchSetupTeardownCalcOutputSize(
        x, expertIds, scalesOptional, xActiveMaskOptional, "group_ep", 8, 0, 32, 0, 4, 2, 0, 0, 1, 2, "", tokenMsgSize,
        expandIdxOutSize, assistInfoForCombineOutSize, commCmdInfoOutSize);

    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    EXPECT_GT(tokenMsgSize, 0);
    EXPECT_GT(expandIdxOutSize, 0);
}

TEST_F(L2MoeDistributeDispatchSetupTest, TestTeardownCalcOutputSizeDifferentBatchSize)
{
    aclTensor* x = CreateAclTensor({32, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    aclTensor* expertIds = CreateAclTensor({32, 4}, ACL_INT32, ACL_FORMAT_ND);
    aclTensor* scalesOptional = nullptr;
    aclTensor* xActiveMaskOptional = nullptr;

    uint64_t tokenMsgSize = 0;
    uint64_t expandIdxOutSize = 0;
    uint64_t assistInfoForCombineOutSize = 0;
    uint64_t commCmdInfoOutSize = 0;
    uint64_t idx = 128;

    aclnnStatus aclRet = aclnnMoeDistributeDispatchSetupTeardownCalcOutputSize(
        x, expertIds, scalesOptional, xActiveMaskOptional, "group_ep", 4, 1, 16, 0, 0, 0, 0, 128, 1, 2, "",
        tokenMsgSize, expandIdxOutSize, assistInfoForCombineOutSize, commCmdInfoOutSize);

    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    EXPECT_GT(tokenMsgSize, 0);
    EXPECT_EQ(expandIdxOutSize, idx);
}

TEST_F(L2MoeDistributeDispatchSetupTest, TestTeardownCalcOutputSizeBF16)
{
    aclTensor* x = CreateAclTensor({8, 7168}, ACL_BF16, ACL_FORMAT_ND);
    aclTensor* expertIds = CreateAclTensor({8, 8}, ACL_INT32, ACL_FORMAT_ND);
    aclTensor* scalesOptional = nullptr;
    aclTensor* xActiveMaskOptional = nullptr;

    uint64_t tokenMsgSize = 0;
    uint64_t expandIdxOutSize = 0;
    uint64_t assistInfoForCombineOutSize = 0;
    uint64_t commCmdInfoOutSize = 0;

    aclnnStatus aclRet = aclnnMoeDistributeDispatchSetupTeardownCalcOutputSize(
        x, expertIds, scalesOptional, xActiveMaskOptional, "group_ep", 2, 0, 32, 0, 0, 0, 0, 0, 1, 2, "", tokenMsgSize,
        expandIdxOutSize, assistInfoForCombineOutSize, commCmdInfoOutSize);

    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    EXPECT_GT(tokenMsgSize, 0);
}

TEST_F(L2MoeDistributeDispatchSetupTest, TestTeardownCalcOutputSizeQuantMode1)
{
    aclTensor* x = CreateAclTensor({8, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    aclTensor* expertIds = CreateAclTensor({8, 8}, ACL_INT32, ACL_FORMAT_ND);
    aclTensor* scalesOptional = nullptr;
    aclTensor* xActiveMaskOptional = nullptr;

    uint64_t tokenMsgSize = 0;
    uint64_t expandIdxOutSize = 0;
    uint64_t assistInfoForCombineOutSize = 0;
    uint64_t commCmdInfoOutSize = 0;

    aclnnStatus aclRet = aclnnMoeDistributeDispatchSetupTeardownCalcOutputSize(
        x, expertIds, scalesOptional, xActiveMaskOptional, "group_ep", 2, 0, 32, 0, 0, 0, 1, 0, 1, 2, "", tokenMsgSize,
        expandIdxOutSize, assistInfoForCombineOutSize, commCmdInfoOutSize);

    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    EXPECT_GT(tokenMsgSize, 0);
}

TEST_F(L2MoeDistributeDispatchSetupTest, TestTeardownCalcOutputSizeQuantMode3)
{
    aclTensor* x = CreateAclTensor({8, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    aclTensor* expertIds = CreateAclTensor({8, 8}, ACL_INT32, ACL_FORMAT_ND);
    aclTensor* scalesOptional = nullptr;
    aclTensor* xActiveMaskOptional = nullptr;

    uint64_t tokenMsgSize = 0;
    uint64_t expandIdxOutSize = 0;
    uint64_t assistInfoForCombineOutSize = 0;
    uint64_t commCmdInfoOutSize = 0;

    aclnnStatus aclRet = aclnnMoeDistributeDispatchSetupTeardownCalcOutputSize(
        x, expertIds, scalesOptional, xActiveMaskOptional, "group_ep", 2, 0, 32, 0, 0, 0, 3, 0, 1, 2, "", tokenMsgSize,
        expandIdxOutSize, assistInfoForCombineOutSize, commCmdInfoOutSize);

    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    EXPECT_GT(tokenMsgSize, 0);
}

TEST_F(L2MoeDistributeDispatchSetupTest, TestTeardownCalcOutputSizeQuantMode4)
{
    aclTensor* x = CreateAclTensor({8, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    aclTensor* expertIds = CreateAclTensor({8, 8}, ACL_INT32, ACL_FORMAT_ND);
    aclTensor* scalesOptional = nullptr;
    aclTensor* xActiveMaskOptional = nullptr;

    uint64_t tokenMsgSize = 0;
    uint64_t expandIdxOutSize = 0;
    uint64_t assistInfoForCombineOutSize = 0;
    uint64_t commCmdInfoOutSize = 0;

    aclnnStatus aclRet = aclnnMoeDistributeDispatchSetupTeardownCalcOutputSize(
        x, expertIds, scalesOptional, xActiveMaskOptional, "group_ep", 2, 0, 32, 0, 0, 0, 4, 0, 1, 2, "", tokenMsgSize,
        expandIdxOutSize, assistInfoForCombineOutSize, commCmdInfoOutSize);

    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    EXPECT_GT(tokenMsgSize, 0);
}

TEST_F(L2MoeDistributeDispatchSetupTest, TestTeardownCalcOutputSizeEp8)
{
    aclTensor* x = CreateAclTensor({8, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    aclTensor* expertIds = CreateAclTensor({8, 8}, ACL_INT32, ACL_FORMAT_ND);
    aclTensor* scalesOptional = nullptr;
    aclTensor* xActiveMaskOptional = nullptr;

    uint64_t tokenMsgSize = 0;
    uint64_t expandIdxOutSize = 0;
    uint64_t assistInfoForCombineOutSize = 0;
    uint64_t commCmdInfoOutSize = 0;

    aclnnStatus aclRet = aclnnMoeDistributeDispatchSetupTeardownCalcOutputSize(
        x, expertIds, scalesOptional, xActiveMaskOptional, "group_ep", 8, 0, 32, 0, 0, 0, 0, 0, 1, 2, "", tokenMsgSize,
        expandIdxOutSize, assistInfoForCombineOutSize, commCmdInfoOutSize);

    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    EXPECT_GT(tokenMsgSize, 0);
}

TEST_F(L2MoeDistributeDispatchSetupTest, TestTeardownCalcOutputSizeEpRankIdShared)
{
    aclTensor* x = CreateAclTensor({8, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    aclTensor* expertIds = CreateAclTensor({8, 8}, ACL_INT32, ACL_FORMAT_ND);
    aclTensor* scalesOptional = nullptr;
    aclTensor* xActiveMaskOptional = nullptr;

    uint64_t tokenMsgSize = 0;
    uint64_t expandIdxOutSize = 0;
    uint64_t assistInfoForCombineOutSize = 0;
    uint64_t commCmdInfoOutSize = 0;

    aclnnStatus aclRet = aclnnMoeDistributeDispatchSetupTeardownCalcOutputSize(
        x, expertIds, scalesOptional, xActiveMaskOptional, "group_ep", 8, 1, 32, 0, 4, 2, 0, 0, 1, 2, "", tokenMsgSize,
        expandIdxOutSize, assistInfoForCombineOutSize, commCmdInfoOutSize);

    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    EXPECT_GT(tokenMsgSize, 0);
}

TEST_F(L2MoeDistributeDispatchSetupTest, TestGetWorkspaceSizeGroupEpTooLong)
{
    aclTensor* x = CreateAclTensor({8, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    aclTensor* expertIds = CreateAclTensor({8, 8}, ACL_INT32, ACL_FORMAT_ND);
    aclTensor* yOut = CreateAclTensor({64, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    aclTensor* expandIdxOut = CreateAclTensor({64}, ACL_INT32, ACL_FORMAT_ND);
    aclTensor* commCmdInfoOut = CreateAclTensor({1312}, ACL_INT32, ACL_FORMAT_ND);

    int num = 128;
    std::string longGroupEp(num, 'g');

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclRet = aclnnMoeDistributeDispatchSetupGetWorkspaceSize(
        x, expertIds, nullptr, nullptr, longGroupEp.c_str(), 2, 0, 32, 0, 0, 0, 0, 0, 2, "", yOut, expandIdxOut,
        commCmdInfoOut, &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MoeDistributeDispatchSetupTest, TestTeardownCalcOutputSizeEpRankLessThanSharedRankNum)
{
    aclTensor* x = CreateAclTensor({8, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    aclTensor* expertIds = CreateAclTensor({8, 8}, ACL_INT32, ACL_FORMAT_ND);

    uint64_t tokenMsgSize = 0;
    uint64_t expandIdxOutSize = 0;
    uint64_t assistInfoForCombineOutSize = 0;
    uint64_t commCmdInfoOutSize = 0;

    aclnnStatus aclRet = aclnnMoeDistributeDispatchSetupTeardownCalcOutputSize(
        x, expertIds, nullptr, nullptr, "group_ep", 8, 0, 32, 0, 4, 4, 0, 0, 1, 2, "", tokenMsgSize, expandIdxOutSize,
        assistInfoForCombineOutSize, commCmdInfoOutSize);

    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    EXPECT_GT(assistInfoForCombineOutSize, 0);
}

TEST_F(L2MoeDistributeDispatchSetupTest, TestSetupExecuteEntry)
{
    // 覆盖 aclnnMoeDistributeDispatchSetup 入口与 NnopbaseSetHcclServerType 弱符号分支（UT 桩无真实 workspace）
    aclnnStatus execRet = aclnnMoeDistributeDispatchSetup(nullptr, 0, nullptr, nullptr);
    EXPECT_THAT(execRet, testing::AnyOf(testing::Eq(ACLNN_SUCCESS), testing::Eq(ACLNN_ERR_PARAM_NULLPTR),
                                        testing::Eq(ACLNN_ERR_PARAM_INVALID)));
}

} // namespace MoeDistributeDispatchSetup
