/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_case_executor.h"

class SwinAttentionScoreQuantTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SwinAttentionScoreQuantTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SwinAttentionScoreQuantTiling TearDown" << std::endl;
    }
};

namespace {

struct SwinAttentionScoreQuantCompileInfo {
    uint32_t coreNum = 8;
    uint64_t ubSize = 196608;
    uint64_t l1Size = 524288;
    uint64_t l2Size = 33554432;
};

gert::TilingContextPara BuildTilingPara(int64_t b, int64_t n, int64_t s, int64_t h, bool mask1_present = true,
                                        bool bias_present = true, int64_t key_s = -1, int64_t value_n = -1,
                                        int64_t mask1_last_dim = -1)
{
    static SwinAttentionScoreQuantCompileInfo compileInfo;
    if (key_s < 0) {
        key_s = s;
    }
    if (value_n < 0) {
        value_n = n;
    }
    if (mask1_last_dim < 0) {
        mask1_last_dim = s;
    }
    return gert::TilingContextPara(
        "SwinAttentionScoreQuant",
        {
            {{{b, n, s, h}, {b, n, s, h}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{b, n, key_s, h}, {b, n, key_s, h}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{b, value_n, s, h}, {b, value_n, s, h}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1, s}, {1, s}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, s}, {1, s}}, ge::DT_UINT64, ge::FORMAT_ND},
            {{{1, h}, {1, h}}, ge::DT_UINT64, ge::FORMAT_ND},
            bias_present ? gert::TilingContextPara::TensorDescription({{1, s}, {1, s}}, ge::DT_FLOAT16, ge::FORMAT_ND) :
                           gert::TilingContextPara::TensorDescription({{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND),
            bias_present ? gert::TilingContextPara::TensorDescription({{1, s}, {1, s}}, ge::DT_INT32, ge::FORMAT_ND) :
                           gert::TilingContextPara::TensorDescription({{}, {}}, ge::DT_INT32, ge::FORMAT_ND),
            bias_present ? gert::TilingContextPara::TensorDescription({{1, h}, {1, h}}, ge::DT_INT32, ge::FORMAT_ND) :
                           gert::TilingContextPara::TensorDescription({{}, {}}, ge::DT_INT32, ge::FORMAT_ND),
            mask1_present ? gert::TilingContextPara::TensorDescription(
                                {{1, n, s, mask1_last_dim}, {1, n, s, mask1_last_dim}}, ge::DT_FLOAT16, ge::FORMAT_ND) :
                            gert::TilingContextPara::TensorDescription({{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND),
            {{{1, n, s, s}, {1, n, s, s}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{{{b, n, s, h}, {b, n, s, h}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"query_transpose", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_transpose", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"value_transpose", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"softmax_axes", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        },
        &compileInfo);
}

void ExpectTilingResult(const gert::TilingContextPara& para, bool expect_success)
{
    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(para, tilingInfo);
    EXPECT_EQ(ok, expect_success);
    if (expect_success) {
        EXPECT_NE(tilingInfo.tilingKey, -1);
        EXPECT_FALSE(tilingInfo.workspaceSizes.empty());
    }
}

} // namespace

TEST_F(SwinAttentionScoreQuantTiling, swin_attention_score_quant_tiling_with_mask_001)
{
    ExpectTilingResult(BuildTilingPara(2048, 3, 49, 32), true);
}

TEST_F(SwinAttentionScoreQuantTiling, swin_attention_score_quant_tiling_without_mask_002)
{
    ExpectTilingResult(BuildTilingPara(2048, 3, 49, 32, false), true);
}

TEST_F(SwinAttentionScoreQuantTiling, swin_attention_score_quant_tiling_null_input_003)
{
    ExpectTilingResult(BuildTilingPara(2048, 3, 49, 32, true, false), false);
}

TEST_F(SwinAttentionScoreQuantTiling, swin_attention_score_quant_tiling_wrong_s_004)
{
    ExpectTilingResult(BuildTilingPara(2048, 3, 1025, 32), false);
}

TEST_F(SwinAttentionScoreQuantTiling, swin_attention_score_quant_tiling_wrong_h_005)
{
    ExpectTilingResult(BuildTilingPara(2048, 3, 49, 33), false);
}

TEST_F(SwinAttentionScoreQuantTiling, swin_attention_score_quant_tiling_wrong_key_006)
{
    ExpectTilingResult(BuildTilingPara(2048, 3, 49, 32, true, true, 50), false);
}

TEST_F(SwinAttentionScoreQuantTiling, swin_attention_score_quant_tiling_wrong_value_007)
{
    ExpectTilingResult(BuildTilingPara(2048, 3, 49, 32, true, true, 49, 4), false);
}

TEST_F(SwinAttentionScoreQuantTiling, swin_attention_score_quant_tiling_wrong_mask_008)
{
    ExpectTilingResult(BuildTilingPara(2048, 3, 49, 32, true, true, 49, 3, 50), false);
}
