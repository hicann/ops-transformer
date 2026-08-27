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
#include "../fused_infer_attention_score_param.h"
#include "../../../../op_host/fused_infer_attention_score_tiling.h"
#include "tiling_case_executor.h"
#include "softmax_tiling_mocker.h"

namespace FusedInferAttentionScoreUT {
namespace {

const char *A2_SOC_INFO = "{\n"
                          "  \"hardware_info\": {\n"
                          "    \"BT_SIZE\": 0,\n"
                          "    \"load3d_constraints\": \"1\",\n"
                          "    \"Intrinsic_fix_pipe_l0c2out\": false,\n"
                          "    \"Intrinsic_data_move_l12ub\": true,\n"
                          "    \"Intrinsic_data_move_l0c2ub\": true,\n"
                          "    \"Intrinsic_data_move_out2l1_nd2nz\": false,\n"
                          "    \"UB_SIZE\": 196608,\n"
                          "    \"L2_SIZE\": 201326592,\n"
                          "    \"L1_SIZE\": 524288,\n"
                          "    \"L0A_SIZE\": 65536,\n"
                          "    \"L0B_SIZE\": 65536,\n"
                          "    \"L0C_SIZE\": 131072,\n"
                          "    \"vector_core_cnt\": 40,\n"
                          "    \"cube_core_cnt\": 20,\n"
                          "    \"socVersion\": \"Ascend910_B3\"\n"
                          "  }\n"
                          "}";

struct EmptyCompileInfo {};
static EmptyCompileInfo g_compileInfo;

class FusedInferAttentionScorePfaTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        SoftmaxTilingMocker::GetInstance().SetSocVersion("Ascend910B");
    }

    static void TearDownTestCase()
    {
        SoftmaxTilingMocker::GetInstance().Reset();
    }
};

} // namespace

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_0)
{
    int64_t actualSeqKv[] = {256};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query
            {{{1, 16, 1024}, {1, 16, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 2: value
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 3: pse_shift
            TD_DEFAULT,
            // 4: atten_mask
            {{{2048, 2048}, {2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},
            // 5: actual_seq_lengths
            TD_DEFAULT,
            // 6: actual_seq_lengths_kv
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-13: quant and antiquant common optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 14: block_table
            {{{1, 4}, {1, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            // 15: query_padding_size
            TD_DEFAULT,
            // 16: kv_padding_size
            TD_DEFAULT,
            // 17: key_antiquant_scale, use per-channel split KV antiquant
            {{{1, 1024}, {1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 18: key_antiquant_offset
            TD_DEFAULT,
            // 19: value_antiquant_scale
            {{{1, 1024}, {1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 20-31: remaining optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
        },
        {
            // 0: attention_out
            {{{1, 16, 1024}, {1, 16, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // Head and scale config.
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            // Layout / KV head config.
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            // Probe the PA per-channel antiquant support branch with valid split KV antiquant shapes.
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            // Quant / output config.
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo, "Ascend910B", 40, 196608, 4096, A2_SOC_INFO);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_1)
{
    int64_t actualSeqQ[] = {128, 256, 384, 512};
    int64_t actualSeqKv[] = {256, 512, 768, 1024};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, TND layout, keep BF16 so the path stays on PFA instead of FIA v3.
            {{{512, 8, 128}, {512, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: key, BNBD paged-cache layout
            {{{20, 8, 128, 128}, {20, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{20, 8, 128, 128}, {20, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3: pse_shift
            TD_DEFAULT,
            // 4: atten_mask, sparse_mode=0 keeps it optional
            TD_DEFAULT,
            // 5: actual_seq_lengths_q
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqQ},
            // 6: actual_seq_lengths_kv
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-13: quant and antiquant common optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 14: block_table, keep valid so PFA PA helper can continue past block-size validation
            {{{4, 8}, {4, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            // 15-31: remaining optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
        },
        {
            // 0: attention_out
            {{{512, 8, 128}, {512, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo, "Ascend910B", 40, 196608, 4096, A2_SOC_INFO);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_2)
{
    int64_t actualSeqQ[] = {6, 12};
    int64_t actualSeqKv[] = {6, 12};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, keep BF16 so the rope case stays on old PFA instead of RouteToFia()
            {{{12, 64, 128}, {12, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: key
            {{{12, 64, 128}, {12, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{12, 64, 128}, {12, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3: pse_shift
            TD_DEFAULT,
            // 4: atten_mask
            TD_DEFAULT,
            // 5: actual_seq_lengths_q
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqQ},
            // 6: actual_seq_lengths_kv
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-23: unrelated optional inputs remain empty
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 24: query_rope, intentionally use D=32 to probe PFA rope D-axis rejection
            {{{12, 64, 32}, {12, 64, 32}}, ge::DT_BF16, ge::FORMAT_ND},
            // 25: key_rope
            {{{12, 64, 32}, {12, 64, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 26-31: remaining optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
        },
        {
            // 0: attention_out
            {{{12, 64, 128}, {12, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo, "Ascend910B", 40, 196608, 4096, A2_SOC_INFO);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_3)
{
    int64_t actualSharedPrefixLen[] = {1};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, keep BF16 so RouteToFia() stays false and queryS=2 avoids the IFA path.
            {{{1, 2, 80}, {1, 2, 80}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: key
            {{{1, 2, 80}, {1, 2, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{1, 2, 80}, {1, 2, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3-20: optional inputs unrelated to prefix remain empty
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 21: key_shared_prefix, intentionally use mismatched H to hit old-PFA BSH prefix shape check
            {{{1, 1, 64}, {1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 22: value_shared_prefix
            {{{1, 1, 64}, {1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 23: actual_shared_prefix_len
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSharedPrefixLen},
            // 24-31: remaining optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
        },
        {
            // 0: attention_out
            {{{1, 2, 80}, {1, 2, 80}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo, "Ascend910B", 40, 196608, 4096, A2_SOC_INFO);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_4)
{
    int64_t actualSharedPrefixLen[] = {1};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, keep BF16 so RouteToFia() stays false and queryS=2 avoids the IFA path.
            {{{1, 1, 2, 80}, {1, 1, 2, 80}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: key
            {{{1, 1, 2, 80}, {1, 1, 2, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{1, 1, 2, 80}, {1, 1, 2, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3-20: optional inputs unrelated to prefix remain empty
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 21: key_shared_prefix, intentionally mismatch D to hit old-PFA BNSD prefix N/D check
            {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 22: value_shared_prefix
            {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 23: actual_shared_prefix_len
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSharedPrefixLen},
            // 24-31: remaining optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
        },
        {
            // 0: attention_out
            {{{1, 1, 2, 80}, {1, 1, 2, 80}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo, "Ascend910B", 40, 196608, 4096, A2_SOC_INFO);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_5)
{
    int64_t actualSeqQ[] = {2};
    int64_t actualSeqKv[] = {4};
    int64_t queryPaddingSize[] = {1};
    int64_t kvPaddingSize[] = {1};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, keep BF16 so RouteToFia() stays false and queryS=2 avoids the IFA path.
            {{{1, 2, 80}, {1, 2, 80}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: key
            {{{1, 4, 80}, {1, 4, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{1, 4, 80}, {1, 4, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3: pse_shift
            TD_DEFAULT,
            // 4: atten_mask
            TD_DEFAULT,
            // 5: actual_seq_lengths_q, required once left padding is enabled
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqQ},
            // 6: actual_seq_lengths_kv
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-13: quant / antiquant common optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 14: block_table, keep null so the case stays on the non-PA old PFA path
            TD_DEFAULT,
            // 15: query_padding_size
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, queryPaddingSize},
            // 16: kv_padding_size
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, kvPaddingSize},
            // 17-31: remaining optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
        },
        {
            // 0: attention_out
            {{{1, 2, 80}, {1, 2, 80}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo, "Ascend910B", 40, 196608, 4096, A2_SOC_INFO);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_6)
{
    std::vector<uint32_t> inputInstanceNum = {
        1, // 0: query
        2, // 1: key tensorlist
        2, // 2: value tensorlist
        0, // 3: pse_shift
        0, // 4: atten_mask
        0, // 5: actual_seq_lengths_q
        0, // 6: actual_seq_lengths_kv
        0, // 7: dequant_scale1
        0, // 8: quant_scale1
        0, // 9: dequant_scale2
        0, // 10: quant_scale2
        0, // 11: quant_offset2
        0, // 12: antiquant_scale
        0, // 13: antiquant_offset
        0, // 14: block_table
        0, // 15: query_padding_size
        0, // 16: kv_padding_size
        0, // 17: key_antiquant_scale
        0, // 18: key_antiquant_offset
        0, // 19: value_antiquant_scale
        0, // 20: value_antiquant_offset
        0, // 21: key_shared_prefix
        0, // 22: value_shared_prefix
        0, // 23: actual_shared_prefix_len
        0, // 24: query_rope
        0, // 25: key_rope
        0, // 26: latent_cache
        0, // 27: block_table_offset
        0, // 28: query_padding_size_as
        0, // 29: kv_padding_size_as
        0, // 30: query_rope_offset
        0, // 31: key_rope_offset
    };
    std::vector<uint32_t> outputInstanceNum = {1, 0};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, use BF16 so RouteToFia() stays false; queryS=2 avoids the IFA route.
            {{{2, 2, 80}, {2, 2, 80}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1-2: key tensorlist, BSH tensorlist makes old PFA read seq_len from dim 1.
            {{{1, 3, 80}, {1, 3, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 4, 80}, {1, 4, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3-4: value tensorlist
            {{{1, 3, 80}, {1, 3, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 4, 80}, {1, 4, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // 0: attention_out
            {{{2, 2, 80}, {2, 2, 80}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        inputInstanceNum, outputInstanceNum, &g_compileInfo, "Ascend910B", 40, 196608, 4096, A2_SOC_INFO);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_7)
{
    std::vector<uint32_t> inputInstanceNum = {
        1, // 0: query
        2, // 1: key tensorlist
        2, // 2: value tensorlist
        0, // 3: pse_shift
        0, // 4: atten_mask
        0, // 5: actual_seq_lengths_q
        0, // 6: actual_seq_lengths_kv
        0, // 7: dequant_scale1
        0, // 8: quant_scale1
        0, // 9: dequant_scale2
        0, // 10: quant_scale2
        0, // 11: quant_offset2
        0, // 12: antiquant_scale
        0, // 13: antiquant_offset
        0, // 14: block_table
        0, // 15: query_padding_size
        0, // 16: kv_padding_size
        0, // 17: key_antiquant_scale
        0, // 18: key_antiquant_offset
        0, // 19: value_antiquant_scale
        0, // 20: value_antiquant_offset
        0, // 21: key_shared_prefix
        0, // 22: value_shared_prefix
        0, // 23: actual_shared_prefix_len
        0, // 24: query_rope
        0, // 25: key_rope
        0, // 26: latent_cache
        0, // 27: block_table_offset
        0, // 28: query_padding_size_as
        0, // 29: kv_padding_size_as
        0, // 30: query_rope_offset
        0, // 31: key_rope_offset
    };
    std::vector<uint32_t> outputInstanceNum = {1, 0};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, BNSD still stays on old PFA because q dtype != k/v dtype and queryS=2.
            {{{2, 1, 2, 80}, {2, 1, 2, 80}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1-2: key tensorlist, BNSD tensorlist makes old PFA read seq_len from dim 2.
            {{{1, 1, 3, 80}, {1, 1, 3, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 4, 80}, {1, 1, 4, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3-4: value tensorlist
            {{{1, 1, 3, 80}, {1, 1, 3, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 4, 80}, {1, 1, 4, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // 0: attention_out
            {{{2, 1, 2, 80}, {2, 1, 2, 80}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        inputInstanceNum, outputInstanceNum, &g_compileInfo, "Ascend910B", 40, 196608, 4096, A2_SOC_INFO);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_8)
{
    int64_t actualSeqKv[] = {256};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query
            {{{1, 128, 8, 128}, {1, 128, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 2: value
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 3-6
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-13
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 14: block_table
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
            // 15-16
            TD_DEFAULT,
            TD_DEFAULT,
            // 17: key_antiquant_scale
            {{{1, 8, 128}, {1, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 18: key_antiquant_offset, keep 3 dims but break D to hit old-PFA offset shape body.
            {{{1, 8, 64}, {1, 8, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            // 19: value_antiquant_scale
            {{{1, 8, 128}, {1, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 20: value_antiquant_offset
            {{{1, 8, 64}, {1, 8, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            // 21-31
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
        },
        {
            {{{1, 128, 8, 128}, {1, 128, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo, "Ascend910B", 40, 196608, 4096, A2_SOC_INFO);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_9)
{
    int64_t actualSeqKv[] = {256};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query
            {{{1, 128, 1024}, {1, 128, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 2: value
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 3-6
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-11
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 12: antiquant_scale, valid per-tensor first dim.
            {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 13: antiquant_offset, break dim[0] to hit old-PFA per-tensor offset reject.
            {{{3}, {3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 14: block_table
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
            // 15-31
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
        },
        {
            {{{1, 128, 1024}, {1, 128, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo, "Ascend910B", 40, 196608, 4096, A2_SOC_INFO);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_10)
{
    int64_t actualSeqKv[] = {256};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, keep the old-PFA PA antiquant route on simple BSH inputs.
            {{{1, 128, 1024}, {1, 128, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key, paged-cache KV in int8 enables PA per-channel antiquant validation.
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 2: value
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 3: pse_shift, use a fully valid 4D shape so the case can reach old-PFA's "not support PSE" branch.
            {{{1, 8, 128, 256}, {1, 8, 128, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 4: atten_mask, keep a valid sparse_mode=3 mask so the case is not intercepted by earlier mask-null
            // checks.
            {{{1, 1, 2048, 2048}, {1, 1, 2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},
            // 5: actual_seq_lengths
            TD_DEFAULT,
            // 6: actual_seq_lengths_kv
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-13: generic quant / antiquant optional inputs remain empty.
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 14: block_table
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
            // 15: query_padding_size
            TD_DEFAULT,
            // 16: kv_padding_size
            TD_DEFAULT,
            // 17: key_antiquant_scale
            {{{1, 1024}, {1, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
            // 18: key_antiquant_offset
            TD_DEFAULT,
            // 19: value_antiquant_scale
            {{{1, 1024}, {1, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
            // 20-31: remaining optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
        },
        {
            // 0: attention_out
            {{{1, 128, 1024}, {1, 128, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo, "Ascend910B", 40, 196608, 4096, A2_SOC_INFO);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_11)
{
    int64_t actualSeqKv[] = {256};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, BNSD keeps the shared old-PFA paged antiquant route but fails the PA layout whitelist.
            {{{1, 8, 128, 128}, {1, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key, paged-cache KV in int8.
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 2: value
            {{{2, 128, 1024}, {2, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 3: pse_shift
            TD_DEFAULT,
            // 4: atten_mask, keep the same valid sparse_mode=3 mask used by the working PA antiquant route.
            {{{1, 1, 2048, 2048}, {1, 1, 2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},
            // 5: actual_seq_lengths_q
            TD_DEFAULT,
            // 6: actual_seq_lengths_kv
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-13: generic quant / antiquant optional inputs remain empty.
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 14: block_table
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
            // 15: query_padding_size
            TD_DEFAULT,
            // 16: kv_padding_size
            TD_DEFAULT,
            // 17: key_antiquant_scale, valid BNSD per-channel antiquant shape.
            {{{1, 8, 1, 128}, {1, 8, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 18: key_antiquant_offset, keep it non-null so old PFA can safely print dim nums in the fallback error
            // branch
            {{{1, 8, 1, 128}, {1, 8, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 19: value_antiquant_scale
            {{{1, 8, 1, 128}, {1, 8, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 20: value_antiquant_offset
            {{{1, 8, 1, 128}, {1, 8, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 21-31: remaining optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
        },
        {
            // 0: attention_out
            {{{1, 8, 128, 128}, {1, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo, "Ascend910B", 40, 196608, 4096, A2_SOC_INFO);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_12)
{
    int64_t actualSeqQ[] = {6, 12};
    int64_t actualSeqKv[] = {6, 12};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, keep BF16 so RouteToFia() stays false and the case falls into old PFA.
            {{{12, 64, 128}, {12, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: key
            {{{12, 64, 128}, {12, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{12, 64, 128}, {12, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3: pse_shift
            TD_DEFAULT,
            // 4: atten_mask
            TD_DEFAULT,
            // 5: actual_seq_lengths_q
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqQ},
            // 6: actual_seq_lengths_kv
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-23: unrelated optional inputs remain empty
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 24: query_rope, keep the tensor non-null but empty to hit old-PFA rope empty rejection
            {{{0, 64, 128}, {0, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 25: key_rope, keep this one non-empty so FIA checker lets the pair through
            {{{12, 64, 128}, {12, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 26-31: remaining optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
        },
        {
            // 0: attention_out
            {{{12, 64, 128}, {12, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo, "Ascend910B", 40, 196608, 4096, A2_SOC_INFO);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_13)
{
    int64_t actualSeqQ[] = {6, 12};
    int64_t actualSeqKv[] = {6, 12};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, keep BF16 so RouteToFia() stays false and the case falls into old PFA.
            {{{12, 64, 128}, {12, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: key
            {{{12, 64, 128}, {12, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: value
            {{{12, 64, 128}, {12, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 3: pse_shift
            TD_DEFAULT,
            // 4: atten_mask
            TD_DEFAULT,
            // 5: actual_seq_lengths_q
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqQ},
            // 6: actual_seq_lengths_kv
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            // 7-23: unrelated optional inputs remain empty
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 24: query_rope, keep this one non-empty so the old-PFA failure lands on key_rope
            {{{12, 64, 128}, {12, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 25: key_rope, keep the tensor non-null but empty to hit old-PFA rope empty rejection
            {{{0, 64, 128}, {0, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 26-31: remaining optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
        },
        {
            // 0: attention_out
            {{{12, 64, 128}, {12, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo, "Ascend910B", 40, 196608, 4096, A2_SOC_INFO);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_14)
{
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, keep FP16 + INT8 KV and small S so old PFA stays in MSD mode instead of switching back to kv
            // antiquant.
            {{{2, 16, 1024}, {2, 16, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key
            {{{2, 16, 1024}, {2, 16, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 2: value
            {{{2, 16, 1024}, {2, 16, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 3-16: optional inputs not needed for this probe
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 17: key_antiquant_scale, valid for FIA per-token mode but old PFA MSD later requires dimNum == 2
            {{{1, 2, 16}, {1, 2, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // 18: key_antiquant_offset
            TD_DEFAULT,
            // 19: value_antiquant_scale
            {{{1, 2, 16}, {1, 2, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // 20-31: remaining optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
        },
        {
            // 0: attention_out
            {{{2, 16, 1024}, {2, 16, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo, "Ascend910B", 40, 196608, 4096, A2_SOC_INFO);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScorePfaTiling, FusedInferAttentionScore_PFA_tiling_15)
{
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            // 0: query, keep BF16 + INT8 KV and small S so old PFA stays in MSD mode.
            {{{1, 16, 1024}, {1, 16, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: key
            {{{1, 16, 1024}, {1, 16, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 2: value
            {{{1, 16, 1024}, {1, 16, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            // 3-16: optional inputs not needed for this probe
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            // 17: key_antiquant_scale, valid FIA/old-PFA per-channel [H] shape
            {{{1024}, {1024}}, ge::DT_BF16, ge::FORMAT_ND},
            // 18: key_antiquant_offset
            TD_DEFAULT,
            // 19: value_antiquant_scale
            {{{1024}, {1024}}, ge::DT_BF16, ge::FORMAT_ND},
            // 20-31: remaining optional inputs
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
        },
        {
            // 0: attention_out
            {{{1, 16, 1024}, {1, 16, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: softmax_lse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &g_compileInfo, "Ascend910B", 40, 196608, 4096, A2_SOC_INFO);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

} // namespace FusedInferAttentionScoreUT
