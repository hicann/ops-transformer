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
#include "../mixed_quant_flash_attn_param.h"
#include "tiling_case_executor.h"

namespace MixedQuantFlashAttnUT {

class MixedQuantFlashAttnArch35TilingTest : public testing::TestWithParam<MixedQuantFlashAttnTilingUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MixedQuantFlashAttn Arch35 TilingTest SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "MixedQuantFlashAttn Arch35 TilingTest TearDown" << std::endl;
    }
};

TEST_P(MixedQuantFlashAttnArch35TilingTest, tiling)
{
    auto param = GetParam();

    gert::TilingContextPara tilingContextPara(
        "MixedQuantFlashAttn",
        {
            param.q,
            param.k,
            param.v,
            param.k_descale,
            param.v_descale,
            param.block_table,
            param.cu_seqlens_q,
            param.seqused_q,
            param.seqused_kv,
            param.sinks,
            param.attn_mask,
            param.metadata,
        },
        {
            param.attn_out,
            param.softmax_lse,
        },
        {
            {"quant_compute_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.quant_compute_mode)},
            {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(param.softmax_scale)},
            {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.mask_mode)},
            {"win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.win_left)},
            {"win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.win_right)},
            {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.max_seqlen_q)},
            {"max_seqlen_kv", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.max_seqlen_kv)},
            {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.layout_q)},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.layout_kv)},
            {"layout_attn_out", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.lay_atten)},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(param.return_softmax_lse)},
        },
        param.inputInstance, param.outputInstance, &param.compileInfo, "Ascend950", 64, 262144, 8192);

    ExecuteTestCase(tilingContextPara, param.expectResult, param.expectTilingKey, param.expectTilingDataHash, {}, 0,
                    true);
}

INSTANTIATE_TEST_SUITE_P(
    MixedQuantFlashAttn, MixedQuantFlashAttnArch35TilingTest,
    testing::ValuesIn(GetCasesFromCsv<MixedQuantFlashAttnTilingUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<MixedQuantFlashAttnTilingUtParam>);

} // namespace MixedQuantFlashAttnUT
