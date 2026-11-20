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
#include <tuple>
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

TEST_F(flash_attention_score_v4_opapi_ut, flash_attention_score_aclnn_warns_but_accepts_invalid_softmax_out_layout)
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
    char softmaxOutLayout[] = "SAME_AS_INPUT";

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
    // An undocumented value falls back to the default BNS8 layout with a warning, so the call itself
    // must still succeed: rejecting it here would break callers that already rely on that fallback.
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

static aclnnStatus RunFlashAttentionV4(const char* softmaxOutLayout)
{
    auto tensorQ = TensorDesc({256, 1, 128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorK = TensorDesc({256, 1, 128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorV = TensorDesc({256, 1, 128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorAttenMask = TensorDesc({256, 256}, ACL_UINT8, ACL_FORMAT_ND).Value(vector<uint8_t>{0});
    char layout[] = "SBH";
    char* softmaxLayout = const_cast<char*>(softmaxOutLayout);

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
                              0.088388, 1.0, 65536, 65536, 1, layout, 0, 0, 0, 1, softmaxLayout, 0, 0),
                        OUTPUT(tensorSoftmaxMax, tensorSoftmaxSum, tensorSoftmaxOutDesc, tensorAttentionOutDesc));
    uint64_t workspaceSize = 0;
    return ut.TestGetWorkspaceSize(&workspaceSize);
}

static aclnnStatus RunFlashAttentionVarLenV4(const char* softmaxOutLayout)
{
    auto tensorQ = TensorDesc({2, 4, 16}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorK = TensorDesc({2, 4, 16}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorV = TensorDesc({2, 4, 16}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto prefix = IntArrayDesc(vector<int64_t>{2, 2});
    auto actualSeq = IntArrayDesc(vector<int64_t>{1, 2});
    char layout[] = "TND";
    char* softmaxLayout = const_cast<char*>(softmaxOutLayout);

    auto softmaxMax = TensorDesc({2, 4, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto softmaxSum = TensorDesc({2, 4, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto softmaxOut = TensorDesc({0}, ACL_BF16, ACL_FORMAT_ND);
    auto attentionOut = TensorDesc({2, 4, 16}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFlashAttentionVarLenScoreV4,
                        INPUT(tensorQ, tensorK, tensorV, nullptr, nullptr, nullptr, nullptr, prefix, actualSeq,
                              actualSeq, 0.25, 1.0, INT32_MAX, INT32_MAX, 4, layout, 0, 0, softmaxLayout),
                        OUTPUT(softmaxMax, softmaxSum, softmaxOut, attentionOut));
    uint64_t workspaceSize = 0;
    return ut.TestGetWorkspaceSize(&workspaceSize);
}

static aclnnStatus RunFlashAttentionVarLenV5(const char* softmaxOutLayout)
{
    auto tensorQ = TensorDesc({2, 4, 16}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorK = TensorDesc({2, 4, 16}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorV = TensorDesc({2, 4, 16}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto prefix = IntArrayDesc(vector<int64_t>{2, 2});
    auto actualSeq = IntArrayDesc(vector<int64_t>{1, 2});
    auto startIdx = IntArrayDesc(vector<int64_t>{0});
    char layout[] = "TND";
    char* softmaxLayout = const_cast<char*>(softmaxOutLayout);

    auto softmaxMax = TensorDesc({2, 4, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto softmaxSum = TensorDesc({2, 4, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto softmaxOut = TensorDesc({0}, ACL_BF16, ACL_FORMAT_ND);
    auto attentionOut = TensorDesc({2, 4, 16}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFlashAttentionVarLenScoreV5,
                        INPUT(tensorQ, nullptr, tensorK, nullptr, tensorV, nullptr, nullptr, nullptr, nullptr, nullptr,
                              prefix, actualSeq, actualSeq, startIdx, startIdx, 0.25, 1.0, INT32_MAX, INT32_MAX, 4,
                              layout, 0, 0, 0, softmaxLayout),
                        OUTPUT(softmaxMax, softmaxSum, softmaxOut, attentionOut));
    uint64_t workspaceSize = 0;
    return ut.TestGetWorkspaceSize(&workspaceSize);
}

using SoftmaxLayoutCase = std::tuple<const char*, aclnnStatus>;

class flash_attention_score_softmax_layout_ut : public testing::TestWithParam<SoftmaxLayoutCase> {};

TEST_P(flash_attention_score_softmax_layout_ut, v4_layout_validation)
{
    EXPECT_EQ(RunFlashAttentionV4(std::get<0>(GetParam())), std::get<1>(GetParam()));
}

TEST_P(flash_attention_score_softmax_layout_ut, varlen_v4_layout_validation)
{
    EXPECT_EQ(RunFlashAttentionVarLenV4(std::get<0>(GetParam())), std::get<1>(GetParam()));
}

TEST_P(flash_attention_score_softmax_layout_ut, varlen_v5_layout_validation)
{
    EXPECT_EQ(RunFlashAttentionVarLenV5(std::get<0>(GetParam())), std::get<1>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(softmax_layout_values, flash_attention_score_softmax_layout_ut,
                         testing::Values(SoftmaxLayoutCase{nullptr, ACLNN_SUCCESS},
                                         SoftmaxLayoutCase{"", ACLNN_SUCCESS},
                                         SoftmaxLayoutCase{"same_as_input", ACLNN_SUCCESS},
                                         SoftmaxLayoutCase{"SAME_AS_INPUT", ACLNN_SUCCESS},
                                         SoftmaxLayoutCase{"same_as_input ", ACLNN_SUCCESS}));

// Invalid dtypes must be rejected by the V4 API before the L0 operator is created.
TEST_F(flash_attention_score_v4_opapi_ut, rejects_invalid_qkv_dtype)
{
    const op::ScopedPlatformNpuArch platformGuard(NpuArch::DAV_3510);
    for (auto dtype : {ACL_DOUBLE, ACL_INT32, ACL_INT8}) {
        auto q = TensorDesc({1, 4, 64, 64}, dtype, ACL_FORMAT_ND);
        auto k = TensorDesc({1, 4, 64, 64}, dtype, ACL_FORMAT_ND);
        auto v = TensorDesc({1, 4, 64, 64}, dtype, ACL_FORMAT_ND);
        auto out = TensorDesc({1, 4, 64, 64}, dtype, ACL_FORMAT_ND);
        auto mx = TensorDesc({1, 4, 64, 8}, ACL_FLOAT, ACL_FORMAT_ND);
        auto sum = TensorDesc({1, 4, 64, 8}, ACL_FLOAT, ACL_FORMAT_ND);
        char layout[] = "BNSD";
        char softmaxLayout[] = "";
        auto ut = OP_API_UT(aclnnFlashAttentionScoreV4,
                            INPUT(q, k, v, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                  nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0.125, 1.0, INT32_MAX,
                                  INT32_MAX, 4, layout, 0, 0, 0, 1, softmaxLayout, 0, 0),
                            OUTPUT(mx, sum, nullptr, out));
        uint64_t workspaceSize = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(flash_attention_score_v4_opapi_ut, rejects_invalid_mask_dtype)
{
    const op::ScopedPlatformNpuArch platformGuard(NpuArch::DAV_3510);
    auto q = TensorDesc({1, 4, 64, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto k = TensorDesc({1, 4, 64, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto v = TensorDesc({1, 4, 64, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto mask = TensorDesc({1, 1, 64, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({1, 4, 64, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto mx = TensorDesc({1, 4, 64, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sum = TensorDesc({1, 4, 64, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    char layout[] = "BNSD";
    char softmaxLayout[] = "";
    auto ut = OP_API_UT(aclnnFlashAttentionScoreV4,
                        INPUT(q, k, v, nullptr, nullptr, nullptr, mask, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0.125, 1.0, INT32_MAX, INT32_MAX, 4,
                              layout, 0, 0, 0, 1, softmaxLayout, 0, 0),
                        OUTPUT(mx, sum, nullptr, out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

static aclnnStatus GetWorkspaceForV4Attributes(int64_t innerPrecise, int64_t sparseMode, char* layout)
{
    auto q = TensorDesc({1, 4, 64, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto k = TensorDesc({1, 4, 64, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto v = TensorDesc({1, 4, 64, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({1, 4, 64, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto mx = TensorDesc({1, 4, 64, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sum = TensorDesc({1, 4, 64, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    char softmaxLayout[] = "";
    auto ut = OP_API_UT(aclnnFlashAttentionScoreV4,
                        INPUT(q, k, v, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0.125, 1.0, INT32_MAX, INT32_MAX, 4,
                              layout, innerPrecise, sparseMode, 0, 1, softmaxLayout, 0, 0),
                        OUTPUT(mx, sum, nullptr, out));
    uint64_t workspaceSize = 0;
    return ut.TestGetWorkspaceSize(&workspaceSize);
}

// Undocumented innerPrecise values keep the historical silent-default handling for compatibility.
TEST_F(flash_attention_score_v4_opapi_ut, keeps_silent_default_for_undocumented_inner_precise)
{
    const op::ScopedPlatformNpuArch platformGuard(NpuArch::DAV_3510);
    char layout[] = "BNSD";
    for (auto value : {-1, 3, 255, 256, 258}) {
        EXPECT_EQ(GetWorkspaceForV4Attributes(value, 0, layout), ACLNN_SUCCESS) << value;
    }
}

// Negative sparse modes keep the historical silent-default handling for compatibility.
// The layout string is case-insensitive in AnalysisAxis, so "tnd" must keep the TND sparse modes.
static aclnnStatus GetWorkspaceForV4TndAttributes(int64_t innerPrecise, int64_t sparseMode, char* layout)
{
    auto q = TensorDesc({64, 4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto k = TensorDesc({64, 4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto v = TensorDesc({64, 4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({64, 4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto mx = TensorDesc({64, 4, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sum = TensorDesc({64, 4, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    char softmaxLayout[] = "";
    auto ut = OP_API_UT(aclnnFlashAttentionScoreV4,
                        INPUT(q, k, v, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0.125, 1.0, INT32_MAX, INT32_MAX, 4,
                              layout, innerPrecise, sparseMode, 0, 1, softmaxLayout, 0, 0),
                        OUTPUT(mx, sum, nullptr, out));
    uint64_t workspaceSize = 0;
    return ut.TestGetWorkspaceSize(&workspaceSize);
}

TEST_F(flash_attention_score_v4_opapi_ut, rejects_out_of_range_sparse_mode)
{
    const op::ScopedPlatformNpuArch platformGuard(NpuArch::DAV_3510);
    char bnsd[] = "BNSD";
    for (auto value : {-2, -1}) {
        EXPECT_EQ(GetWorkspaceForV4Attributes(0, value, bnsd), ACLNN_SUCCESS) << value;
    }
    for (auto value : {7, 8, 9}) {
        EXPECT_EQ(GetWorkspaceForV4Attributes(0, value, bnsd), ACLNN_ERR_PARAM_INVALID) << value;
    }
    char tnd[] = "TND";
    EXPECT_EQ(GetWorkspaceForV4TndAttributes(0, -1, tnd), ACLNN_SUCCESS);
    EXPECT_EQ(GetWorkspaceForV4TndAttributes(0, 9, tnd), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(flash_attention_score_v4_opapi_ut, empty_outputs_keep_fast_path)
{
    auto q = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto k = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto v = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto mx = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sum = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    char layout[] = "BNSD";
    char softmaxLayout[] = "";
    auto ut = OP_API_UT(aclnnFlashAttentionScoreV4,
                        INPUT(q, k, v, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0.125, 1.0, INT32_MAX, INT32_MAX, 4,
                              layout, 3, -1, 0, 1, softmaxLayout, 0, 0),
                        OUTPUT(mx, sum, nullptr, out));
    uint64_t workspaceSize = 1;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
    EXPECT_EQ(workspaceSize, 0U);
}

TEST_F(flash_attention_score_v4_opapi_ut, lowercase_tnd_keeps_tnd_sparse_modes)
{
    const op::ScopedPlatformNpuArch platformGuard(NpuArch::DAV_3510);
    char tndLower[] = "tnd";
    for (auto value : {0, 7, 8}) {
        EXPECT_EQ(GetWorkspaceForV4TndAttributes(0, value, tndLower), ACLNN_SUCCESS) << value;
    }
}

TEST_F(flash_attention_score_v4_opapi_ut, accepts_documented_attribute_values)
{
    const op::ScopedPlatformNpuArch platformGuard(NpuArch::DAV_3510);
    char bnsd[] = "BNSD";
    for (auto innerPrecise : {0, 1, 2}) {
        for (auto sparseMode : {0, 1, 2, 3, 4, 5, 6}) {
            EXPECT_EQ(GetWorkspaceForV4Attributes(innerPrecise, sparseMode, bnsd), ACLNN_SUCCESS)
                << "innerPrecise=" << innerPrecise << " sparseMode=" << sparseMode;
        }
    }
    char tndUpper[] = "TND";
    for (auto sparseMode : {0, 1, 2, 3, 4, 6, 7, 8}) {
        EXPECT_EQ(GetWorkspaceForV4TndAttributes(0, sparseMode, tndUpper), ACLNN_SUCCESS) << sparseMode;
    }
}

TEST_F(flash_attention_score_v4_opapi_ut, arch950_keeps_attribute_rejections)
{
    const op::ScopedPlatformNpuArch platformGuard(NpuArch::DAV_3510);
    char bnsd[] = "BNSD";
    EXPECT_EQ(GetWorkspaceForV4Attributes(0, -1, bnsd), ACLNN_SUCCESS);
    EXPECT_EQ(GetWorkspaceForV4Attributes(3, 0, bnsd), ACLNN_SUCCESS);
    EXPECT_EQ(GetWorkspaceForV4Attributes(0, 7, bnsd), ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(GetWorkspaceForV4Attributes(0, 0, bnsd), ACLNN_SUCCESS);
}

// Arch92 shares the arch35 regbase checks (V121 overrides only block sizes), so it rejects like the 950.
TEST_F(flash_attention_score_v4_opapi_ut, arch92_matches_arch35_rejections)
{
    const op::ScopedPlatformNpuArch platformGuard(NpuArch::DAV_9201);
    char bnsd[] = "BNSD";
    EXPECT_EQ(GetWorkspaceForV4Attributes(0, -1, bnsd), ACLNN_SUCCESS);
    EXPECT_EQ(GetWorkspaceForV4Attributes(3, 0, bnsd), ACLNN_SUCCESS);
    EXPECT_EQ(GetWorkspaceForV4Attributes(0, 7, bnsd), ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(GetWorkspaceForV4Attributes(0, 0, bnsd), ACLNN_SUCCESS);
}

// Arch22 tiling accepts any sparseMode without an attenMask, so the wrapper checks stay 950-only.
TEST_F(flash_attention_score_v4_opapi_ut, arch22_skips_v4_wrapper_checks)
{
    const op::ScopedPlatformNpuArch platformGuard(NpuArch::DAV_2201);
    auto q = TensorDesc({1, 4, 64, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto k = TensorDesc({1, 4, 64, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto v = TensorDesc({1, 4, 64, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({1, 4, 64, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto mx = TensorDesc({1, 4, 64, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sum = TensorDesc({1, 4, 64, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    char layout[] = "BNSD";
    char softmaxLayout[] = "";
    auto ut = OP_API_UT(aclnnFlashAttentionScoreV4,
                        INPUT(q, k, v, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0.125, 1.0, INT32_MAX, INT32_MAX, 4,
                              layout, 0, 7, 0, 1, softmaxLayout, 0, 0),
                        OUTPUT(mx, sum, nullptr, out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}
