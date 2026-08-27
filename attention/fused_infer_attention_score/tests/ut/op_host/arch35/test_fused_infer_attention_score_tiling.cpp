/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "../fused_infer_attention_score_param.h"
#include "../../../../op_host/fused_infer_attention_score_tiling_compile_info.h"
#include "tiling_case_executor.h"

namespace FusedInferAttentionScoreUT {

class FusedInferAttentionScoreArch35TilingTest : public testing::TestWithParam<FusedInferAttentionTilingUtParam> {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(FusedInferAttentionScoreArch35TilingTest);

TEST_P(FusedInferAttentionScoreArch35TilingTest, param)
{
    auto param = GetParam();
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {};

    const std::string A5SocInfo = "{\n"
                                  "  \"hardware_info\": {\n"
                                  "    \"BT_SIZE\": 0,\n"
                                  "    \"load3d_constraints\": \"1\",\n"
                                  "    \"Intrinsic_fix_pipe_l0c2out\": false,\n"
                                  "    \"Intrinsic_data_move_l12ub\": true,\n"
                                  "    \"Intrinsic_data_move_l0c2ub\": true,\n"
                                  "    \"Intrinsic_data_move_out2l1_nd2nz\": false,\n"
                                  "    \"UB_SIZE\": 196608,\n"
                                  "    \"L2_SIZE\": 117440512,\n"
                                  "    \"L1_SIZE\": 524288,\n"
                                  "    \"L0A_SIZE\": 65536,\n"
                                  "    \"L0B_SIZE\": 65536,\n"
                                  "    \"L0C_SIZE\": 65536,\n"
                                  "    \"vector_core_cnt\": 64,\n"
                                  "    \"cube_core_cnt\": 32,\n"
                                  "    \"socVersion\": \"Ascend950\"\n"
                                  "  }\n"
                                  "}";

    if (!param.isTensorList) {
        gert::TilingContextPara tilingContextPara(
            "FusedInferAttentionScore",
            {param.query,
             param.key,
             param.value,
             param.pse_shift,
             param.atten_mask,
             param.actual_seq_lengths,
             param.actual_seq_lengths_kv,
             param.dequant_scale1,
             param.quant_scale1,
             param.dequant_scale2,
             param.quant_scale2,
             param.quant_offset2,
             param.antiquant_scale,
             param.antiquant_offset,
             param.block_table,
             param.query_padding_size,
             param.kv_padding_size,
             param.key_antiquant_scale,
             param.key_antiquant_offset,
             param.value_antiquant_scale,
             param.value_antiquant_offset,
             param.key_shared_prefix,
             param.value_shared_prefix,
             param.actual_shared_prefix_len,
             param.query_rope,
             param.key_rope,
             param.key_rope_antiquant_scale,
             param.dequant_scale_query,
             param.learnable_sink,
             param.q_start_idx,
             param.kv_start_idx},
            {param.attention_out, param.softmax_lse},
            {
                {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.num_heads)},
                {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(param.scale)},
                {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.pre_tokens)},
                {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.next_tokens)},
                {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.input_layout)},
                {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.num_key_value_heads)},
                {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.sparse_mode)},
                {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.inner_precise)},
                {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.block_size)},
                {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.antiquant_mode)},
                {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(param.softmax_lse_flag)},
                {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.key_antiquant_mode)},
                {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.value_antiquant_mode)},
                {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.query_quant_mode)},
                {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.pse_type)},
                {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.out_dtype)},
            },
            &compileInfo, "Ascend950", 64, 196608, 16384, A5SocInfo);
        ExecuteTestCase(tilingContextPara, param.expectResult, param.expectTilingKey, param.expectTilingDataHash, {}, 0,
                        true);
        return;
    }

    std::vector<gert::TilingContextPara::TensorDescription> allInputs = {param.query,
                                                                         param.key,
                                                                         param.value,
                                                                         param.pse_shift,
                                                                         param.atten_mask,
                                                                         param.actual_seq_lengths,
                                                                         param.actual_seq_lengths_kv,
                                                                         param.dequant_scale1,
                                                                         param.quant_scale1,
                                                                         param.dequant_scale2,
                                                                         param.quant_scale2,
                                                                         param.quant_offset2,
                                                                         param.antiquant_scale,
                                                                         param.antiquant_offset,
                                                                         param.block_table,
                                                                         param.query_padding_size,
                                                                         param.kv_padding_size,
                                                                         param.key_antiquant_scale,
                                                                         param.key_antiquant_offset,
                                                                         param.value_antiquant_scale,
                                                                         param.value_antiquant_offset,
                                                                         param.key_shared_prefix,
                                                                         param.value_shared_prefix,
                                                                         param.actual_shared_prefix_len,
                                                                         param.query_rope,
                                                                         param.key_rope,
                                                                         param.key_rope_antiquant_scale,
                                                                         param.dequant_scale_query,
                                                                         param.learnable_sink,
                                                                         param.q_start_idx,
                                                                         param.kv_start_idx};

    std::vector<gert::TilingContextPara::TensorDescription> inputTensors;
    std::vector<uint32_t> inputInstanceNum;
    inputTensors.emplace_back(allInputs[0]);
    inputInstanceNum.emplace_back(1);
    for (auto &t : param.keyList)
        inputTensors.emplace_back(t);
    inputInstanceNum.emplace_back(static_cast<uint32_t>(param.keyList.size()));
    for (auto &t : param.valueList)
        inputTensors.emplace_back(t);
    inputInstanceNum.emplace_back(static_cast<uint32_t>(param.valueList.size()));
    for (size_t i = 3; i < allInputs.size(); i++) {
        if (param.inputInstance[i] != 0) {
            inputTensors.emplace_back(allInputs[i]);
        }
        inputInstanceNum.emplace_back(param.inputInstance[i]);
    }
    std::vector<uint32_t> outputInstanceNum = param.outputInstance;

    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore", inputTensors, {param.attention_out, param.softmax_lse},
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.num_heads)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(param.scale)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.pre_tokens)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.next_tokens)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.input_layout)},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.num_key_value_heads)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.sparse_mode)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.inner_precise)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.block_size)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.antiquant_mode)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(param.softmax_lse_flag)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.key_antiquant_mode)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.value_antiquant_mode)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.query_quant_mode)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.pse_type)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.out_dtype)},
        },
        inputInstanceNum, outputInstanceNum, &compileInfo, "Ascend950", 64, 196608, 16384, A5SocInfo);

    ExecuteTestCase(tilingContextPara, param.expectResult, param.expectTilingKey, param.expectTilingDataHash, {}, 0,
                    true);
}

INSTANTIATE_TEST_SUITE_P(
    FusedInferAttentionScore, FusedInferAttentionScoreArch35TilingTest,
    testing::ValuesIn(GetCasesFromCsv<FusedInferAttentionTilingUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<FusedInferAttentionTilingUtParam>);

} // namespace FusedInferAttentionScoreUT
