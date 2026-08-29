/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcpp"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <vector>
#include <cstdint>
#include "gtest/gtest.h"
#include "../../../op_api/aclnn_prompt_flash_attention_v2.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"
#pragma GCC diagnostic pop
using namespace std;
using namespace op;

class prompt_flash_attention_v2_opapi_ut : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        // cout << "prompt_flash_attention_v2_opapi_ut SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        // cout << "prompt_flash_attention_v2_opapi_ut TearDown" << endl;
    }
};

// dav3510 reject
TEST_F(prompt_flash_attention_v2_opapi_ut, prompt_flash_attention_v2_aclnn_950_unsupported)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    char inputLayout[] = "BNSD";
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnPromptFlashAttentionV2GetWorkspaceSize(
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 8,
        0.08838834764831843, INT64_MAX, 0, inputLayout, 8, 0, nullptr, &workspaceSize, &executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_RUNTIME_ERROR);
    EXPECT_EQ(executor, nullptr);

    aclRet = aclnnPromptFlashAttentionV2(nullptr, 0, nullptr, nullptr);
    EXPECT_EQ(aclRet, ACLNN_ERR_RUNTIME_ERROR);
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
}

// BNSD layout, FP16, no quantization, standard MHA (N_q == N_kv)
TEST_F(prompt_flash_attention_v2_opapi_ut, prompt_flash_attention_v2_aclnn_0)
{
    const int64_t numHeads = 8;
    const double scaleValue = 0.08838834764831843; // 1 / sqrt(128)
    const int64_t preTokens = INT64_MAX;
    const int64_t nextTokens = 0;
    char inputLayout[] = "BNSD";
    const int64_t numKeyValueHeads = 8;
    const int64_t sparseMode = 0;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    auto ut = OP_API_UT(aclnnPromptFlashAttentionV2,
                        INPUT(TensorDesc({2, 8, 64, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1), // query
                              TensorDesc({2, 8, 64, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1), // key
                              TensorDesc({2, 8, 64, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1), // value
                              nullptr,                                                                   // pseShift
                              nullptr,                                                                   // attenMask
                              nullptr, // actualSeqLengths
                              nullptr, // actualSeqLengthsKv
                              nullptr, // deqScale1
                              nullptr, // quantScale1
                              nullptr, // deqScale2
                              nullptr, // quantScale2
                              nullptr, // quantOffset2
                              numHeads, scaleValue, preTokens, nextTokens, inputLayout, numKeyValueHeads, sparseMode),
                        OUTPUT(TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND) // attentionOut
                               ));
#pragma GCC diagnostic pop

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// BSH layout, FP16, GQA (N_q=8, N_kv=1)
TEST_F(prompt_flash_attention_v2_opapi_ut, prompt_flash_attention_v2_aclnn_1)
{
    const int64_t numHeads = 8;
    const double scaleValue = 0.08838834764831843; // 1 / sqrt(128)
    const int64_t preTokens = INT64_MAX;
    const int64_t nextTokens = 0;
    char inputLayout[] = "BSH";
    const int64_t numKeyValueHeads = 1;
    const int64_t sparseMode = 0;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    auto ut = OP_API_UT(
        aclnnPromptFlashAttentionV2,
        INPUT(TensorDesc({2, 64, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1), // query: B,S,H_q=N_q*D=8*128
              TensorDesc({2, 64, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1),  // key:   B,S,H_kv=N_kv*D=1*128
              TensorDesc({2, 64, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1),  // value: B,S,H_kv
              nullptr,                                                                 // pseShift
              nullptr,                                                                 // attenMask
              nullptr,                                                                 // actualSeqLengths
              nullptr,                                                                 // actualSeqLengthsKv
              nullptr,                                                                 // deqScale1
              nullptr,                                                                 // quantScale1
              nullptr,                                                                 // deqScale2
              nullptr,                                                                 // quantScale2
              nullptr,                                                                 // quantOffset2
              numHeads, scaleValue, preTokens, nextTokens, inputLayout, numKeyValueHeads, sparseMode),
        OUTPUT(TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND) // attentionOut: B,S,H_q
               ));
#pragma GCC diagnostic pop

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
