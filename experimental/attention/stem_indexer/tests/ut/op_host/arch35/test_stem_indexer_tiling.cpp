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
#include <string>
#include "../stem_indexer_param.h"
#include "tiling_case_executor.h"

namespace StemIndexerUT {

struct StemIndexerCompileInfo {};

static const std::string STEM_INDEXER_SOC_INFO =
    "{\n"
    "  \"hardware_info\": {\n"
    "    \"BT_SIZE\": 0,\n"
    "    \"load3d_constraints\": \"1\",\n"
    "    \"Intrinsic_fix_pipe_l0c2out\": false,\n"
    "    \"Intrinsic_data_move_l12ub\": true,\n"
    "    \"Intrinsic_data_move_l0c2ub\": true,\n"
    "    \"Intrinsic_data_move_out2l1_nd2nz\": false,\n"
    "    \"UB_SIZE\": 262144,\n"
    "    \"L2_SIZE\": 33554432,\n"
    "    \"L1_SIZE\": 524288,\n"
    "    \"L0A_SIZE\": 65536,\n"
    "    \"L0B_SIZE\": 65536,\n"
    "    \"L0C_SIZE\": 131072,\n"
    "    \"CORE_NUM\": 64,\n"
    "    \"cube_core_cnt\": 32,\n"
    "    \"vector_core_cnt\": 64,\n"
    "    \"socVersion\": \"Ascend950\"\n"
    "  }\n"
    "}";

class StemIndexerArch35TilingTest : public testing::TestWithParam<StemIndexerTilingUtParam> {
protected:
    static void SetUpTestCase() { std::cout << "StemIndexer Arch35 TilingTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "StemIndexer Arch35 TilingTest TearDown" << std::endl; }
};

TEST_P(StemIndexerArch35TilingTest, param)
{
    auto param = GetParam();
    StemIndexerCompileInfo compileInfo;

    gert::TilingContextPara tilingContextPara(
        "StemIndexer",
        {param.qflat, param.kflat, param.vbias, param.q_seq_lens, param.kv_seq_lens, param.num_prompt_tokens,
         param.metadata},
        {param.sparse_indices, param.sparse_seq_len},
        {
            {"causal", Ops::Transformer::AnyValue::CreateFrom<bool>(param.causal)},
            {"stem_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.stem_block_size)},
            {"stem_stride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.stem_stride)},
            {"alpha", Ops::Transformer::AnyValue::CreateFrom<float>(param.alpha)},
            {"initial_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.initial_blocks)},
            {"window_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.window_size)},
            {"k_block_num_rate_medium", Ops::Transformer::AnyValue::CreateFrom<float>(param.k_block_num_rate_medium)},
            {"k_block_num_bias_medium", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.k_block_num_bias_medium)},
            {"k_block_num_rate_large", Ops::Transformer::AnyValue::CreateFrom<float>(param.k_block_num_rate_large)},
            {"k_block_num_bias_large", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.k_block_num_bias_large)},
            {"topk_score_precision", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.topk_score_precision)},
        },
        &compileInfo, "Ascend950", STEM_INDEXER_SOC_INFO, 16384);

    ExecuteTestCase(tilingContextPara, param.expectResult, param.expectTilingKey, param.expectTilingDataHash, {}, 0,
                    true);
}

INSTANTIATE_TEST_SUITE_P(
    StemIndexer, StemIndexerArch35TilingTest,
    testing::ValuesIn(GetCasesFromCsv<StemIndexerTilingUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<StemIndexerTilingUtParam>);

} // namespace StemIndexerUT
