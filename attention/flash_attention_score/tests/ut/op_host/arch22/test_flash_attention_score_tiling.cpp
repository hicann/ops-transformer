/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "../flash_attention_score_param.h"
#include "tiling_case_executor.h"

namespace FlashAttentionScoreUT {

class FlashAttentionScoreArch22TilingTest : public testing::TestWithParam<FlashAttentionScoreTilingUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FlashAttentionScore Arch22 TilingTest SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "FlashAttentionScore Arch22 TilingTest TearDown" << std::endl;
    }
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(FlashAttentionScoreArch22TilingTest);

TEST_P(FlashAttentionScoreArch22TilingTest, tiling)
{
    auto param = GetParam();
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        24, 48, 196608, 524288, 131072, 201326592, platform_ascendc::SocVersion::ASCEND910B, NpuArch::DAV_2201};

    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {param.query, param.key, param.value, param.real_shift, param.drop_mask, param.padding_mask, param.atten_mask,
         param.prefix, param.actual_seq_qlen, param.actual_seq_kvlen, param.q_start_idx, param.kv_start_idx,
         param.dScaleQ, param.dScaleK, param.dScaleV, param.queryRope, param.keyRope, param.sink},
        {param.softmaxMax, param.softmaxSum, param.softmaxOut, param.attentionOut},
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(param.scale_value)},
            {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(param.keep_prob)},
            {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.pre_tockens)},
            {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.next_tockens)},
            {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.head_num)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.input_layout)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.inner_precise)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.sparse_mode)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.pse_type)},
            {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.seed)},
            {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.offset)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.out_dtype)},
            {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.softmax_out_layout)},
        },
        &compileInfo, "Ascend910B", 64, 262144, 8192);

    ExecuteTestCase(tilingContextPara, param.expectResult, param.expectTilingKey, param.expectTilingData,
                    param.expectWorkspaces);
}

INSTANTIATE_TEST_SUITE_P(
    FlashAttentionScore, FlashAttentionScoreArch22TilingTest,
    testing::ValuesIn(GetCasesFromCsv<FlashAttentionScoreTilingUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<FlashAttentionScoreTilingUtParam>);

} // namespace FlashAttentionScoreUT
