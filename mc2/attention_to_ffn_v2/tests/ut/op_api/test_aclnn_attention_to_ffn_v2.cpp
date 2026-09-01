/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN " AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "test_aclnn_attention_to_ffn_v2_helper.h"

#include <cfloat>

#include <array>
#include <vector>

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"
#include "platform/platform_info.h"

using namespace op;
using namespace std;

namespace AttentionToFfnV2UT {
class AclnnAttentionToFfnV2Test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
        cout << "AclnnAttentionToFfnV2Test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
        cout << "AclnnAttentionToFfnV2Test TearDown" << endl;
    }
};

TEST_F(AclnnAttentionToFfnV2Test, TestAttentionToFfnV2NoQuant)
{
    TensorDesc context = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc x = TensorDesc({1, 16, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc sessionId = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc microBatchId = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc layerId = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expertIds = TensorDesc({1, 16, 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expertRankTable = TensorDesc({1, 9, 4}, ACL_INT32, ACL_FORMAT_ND);
    // TensorDesc scales = TensorDesc({9, 7168}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc activeMask = TensorDesc({1, 16}, ACL_BOOL, ACL_FORMAT_ND);
    const char *group = "test_attention_to_ffn_group";
    int64_t worldSize = 16;
    std::vector<int64_t> ffnTokenInfoTable = {11, 1, 146};
    std::vector<int64_t> ffnTokenData = {11, 1, 16, 9, 7168};
    std::vector<int64_t> attnTokenInfoTable = {1, 16, 9};
    aclIntArray *ffnTokenInfoTableShape = aclCreateIntArray(ffnTokenInfoTable.data(), ffnTokenInfoTable.size());
    aclIntArray *ffnTokenDataShape = aclCreateIntArray(ffnTokenData.data(), ffnTokenData.size());
    aclIntArray *attnTokenInfoTableShape = aclCreateIntArray(attnTokenInfoTable.data(), attnTokenInfoTable.size());
    int64_t moeExpertNum = 8;
    int64_t quantMode = 0;
    int64_t syncFlag = 0;
    int64_t ffnStartRankId = 0;
    int64_t cclBufferSize = 1LL << 30;

    auto ut =
        OP_API_UT(aclnnAttentionToFfnV2,
                  INPUT(context, x, sessionId, microBatchId, layerId, expertIds, expertRankTable, nullptr, activeMask,
                        group, worldSize, ffnTokenInfoTableShape, ffnTokenDataShape, attnTokenInfoTableShape,
                        moeExpertNum, quantMode, syncFlag, ffnStartRankId, cclBufferSize),
                  OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AclnnAttentionToFfnV2Test, TestAttentionToFfnV2Quant)
{
    TensorDesc context = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc x = TensorDesc({1, 16, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc sessionId = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc microBatchId = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc layerId = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expertIds = TensorDesc({1, 16, 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expertRankTable = TensorDesc({1, 9, 4}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc scales = TensorDesc({9, 7168}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc activeMask = TensorDesc({1, 16}, ACL_BOOL, ACL_FORMAT_ND);
    const char *group = "test_attention_to_ffn_group";
    int64_t worldSize = 16;
    std::vector<int64_t> ffnTokenInfoTable = {11, 1, 146};
    std::vector<int64_t> ffnTokenData = {11, 1, 16, 9, 7296};
    std::vector<int64_t> attnTokenInfoTable = {1, 16, 9};
    aclIntArray *ffnTokenInfoTableShape = aclCreateIntArray(ffnTokenInfoTable.data(), ffnTokenInfoTable.size());
    aclIntArray *ffnTokenDataShape = aclCreateIntArray(ffnTokenData.data(), ffnTokenData.size());
    aclIntArray *attnTokenInfoTableShape = aclCreateIntArray(attnTokenInfoTable.data(), attnTokenInfoTable.size());
    int64_t moeExpertNum = 8;
    int64_t quantMode = 2;
    int64_t syncFlag = 0;
    int64_t ffnStartRankId = 0;
    int64_t cclBufferSize = 1LL << 30;

    auto ut =
        OP_API_UT(aclnnAttentionToFfnV2,
                  INPUT(context, x, sessionId, microBatchId, layerId, expertIds, expertRankTable, scales, activeMask,
                        group, worldSize, ffnTokenInfoTableShape, ffnTokenDataShape, attnTokenInfoTableShape,
                        moeExpertNum, quantMode, syncFlag, ffnStartRankId, cclBufferSize),
                  OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

} // namespace AttentionToFfnV2UT
