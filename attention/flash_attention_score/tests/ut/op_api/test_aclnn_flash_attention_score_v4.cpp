/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"
#include "../../../op_api/aclnn_flash_attention_score.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class flash_attention_score_v4_opapi_ut : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        // cout << "flash_attention_score_v4_opapi_ut SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        // cout << "flash_attention_score_v4_opapi_ut TearDown" << endl;
    }
};

TEST_F(flash_attention_score_v4_opapi_ut, flash_attention_score_aclnn_0)
{
    auto tensorQ = TensorDesc({256, 1, 128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorK = TensorDesc({256, 1, 128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorV = TensorDesc({256, 1, 128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorAttenMask = TensorDesc({256, 256}, ACL_UINT8, ACL_FORMAT_ND).Value(vector<uint8_t>{0});

    const double scaleValue = 0.088388;
    const double keepProb = 1.0;
    const int64_t preTokens = 65536;
    const int64_t nextTokens = 65536;
    const int64_t headNum = 1;
    char layout[] = "SBH";
    const int64_t innerPrecise = 0;
    const int64_t sparseMod = 0;
    const int64_t outDtype = 0;
    const int64_t pseType = 1;
    const int64_t seed = 0;
    const int64_t offset = 0;
    char softmaxOutLayout[] = "";

    // 输出
    auto tensorAttentionOutDesc = TensorDesc({256, 1, 128}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorSoftmaxOutDesc = TensorDesc({0, 0, 0, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorSoftmaxMax = TensorDesc({1, 1, 256, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorSoftmaxSum = TensorDesc({1, 1, 256, 8}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFlashAttentionScoreV4,
                        INPUT(tensorQ, tensorK, tensorV,
                              nullptr, // pse
                              nullptr, // dropMask
                              nullptr, // padding
                              tensorAttenMask,
                              nullptr, // queryRope
                              nullptr, // keyRope
                              nullptr, // dScaleQ
                              nullptr, // dScaleK
                              nullptr, // dScaleV
                              nullptr, // sink
                              nullptr, // Prefix
                              nullptr, // ActualSeqQLen
                              nullptr, // ActualSeqKVLen
                              nullptr, // QStartIdx
                              nullptr, // KVStartIdx
                              scaleValue, keepProb, preTokens, nextTokens, headNum, layout, innerPrecise, sparseMod,
                              outDtype, pseType, softmaxOutLayout, seed, offset),
                        OUTPUT(tensorSoftmaxMax, tensorSoftmaxSum, tensorSoftmaxOutDesc, tensorAttentionOutDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
