/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../../op_host/prompt_flash_attention_tiling_compile_info.h"
#include "../../../../op_host/prompt_flash_attention_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "softmax_tiling_mocker.h"
using namespace std;

class PromptFlashAttentionTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        // std::cout << "PromptFlashAttentionTiling SetUp" << std::endl;
        SoftmaxTilingMocker::GetInstance().SetSocVersion("Ascend910B");
    }

    static void TearDownTestCase()
    {
        // std::cout << "PromptFlashAttentionTiling TearDown" << std::endl;
        SoftmaxTilingMocker::GetInstance().Reset();
    }
};

// BNSD
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_0)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
    int64_t actual_seq_qlist[] = {556, 732, 637, 573, 149, 158, 278, 1011, 623, 683,  680, 449,
                                  538, 920, 396, 322, 268, 153, 452, 458,  821, 1001, 744};
    int64_t actual_seq_kvlist[] = {556, 732, 637, 573, 149, 158, 278, 1011, 623, 683,  680, 449,
                                   538, 920, 396, 322, 268, 153, 452, 458,  821, 1001, 744};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {
            {{{23, 40, 1024, 128}, {23, 40, 1024, 128}}, ge::DT_INT8, ge::FORMAT_ND}, // query input0
            {{{23, 40, 9088, 128}, {23, 40, 9088, 128}}, ge::DT_INT8, ge::FORMAT_ND}, // key input1
            {{{23, 40, 9088, 128}, {23, 40, 9088, 128}}, ge::DT_INT8, ge::FORMAT_ND}, // value input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                // pse_shift input3
            {{{23, 1024, 9088}, {23, 1024, 9088}}, ge::DT_BOOL, ge::FORMAT_ND},       // atten_mask input4
            {{{23}, {23}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},      // actual_seq_lengths_q
            {{{23}, {23}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},     // actual_seq_lengths_kv
            {{{1}, {1}}, ge::DT_UINT64, ge::FORMAT_ND},                               // deq_scale1 input5
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                // quant_scale1 input6
            {{{1}, {1}}, ge::DT_UINT64, ge::FORMAT_ND},                               // deq_scale2 input7
            {{{40, 128}, {40, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},                    // quant_scale2 input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                   // quant_offset2 input9
        },
        {{{{23, 40, 1024, 128}, {23, 40, 1024, 128}}, ge::DT_INT8, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(488)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-127)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    int64_t expectTilingKey = 1000000000000021217;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// BSH
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_1)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        // 硬件参数
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
    int64_t *actual_seq_qlist = nullptr;
    int64_t *actual_seq_kvlist = nullptr;
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {
            {{{256, 14, 5120}, {256, 14, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query input0
            {{{256, 14, 5120}, {256, 14, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key input1
            {{{256, 14, 5120}, {256, 14, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // pse_shift input3
            {{{2048, 2048}, {2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},          // atten_mask input4
            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},   // actual_seq_lengths_q
            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},  // actual_seq_lengths_kv
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                            // deq_scale1 input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // quant_scale1 input6
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                            // deq_scale2 input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // quant_scale2 input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                              // quant_offset2 input9
        },
        {{{{256, 14, 5120}, {256, 14, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0884f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    int64_t expectTilingKey = 1000000000000101612;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// BSND
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_2)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        // 硬件参数
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
    int64_t actual_seq_qlist[] = {2048, 2048, 1024, 1024, 2048, 2028, 2048, 1024};
    int64_t actual_seq_kvlist[] = {2048, 1024, 2048, 2048, 2048, 1024, 1024, 1024};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {
            {{{8, 2048, 40, 128}, {8, 2048, 40, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // query input0
            {{{8, 2048, 40, 128}, {8, 2048, 40, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // key input1
            {{{8, 2048, 40, 128}, {8, 2048, 40, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // value input2
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                 // pse_shift input3
            {{{8, 1, 2048, 2048}, {8, 1, 2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND}, // atten_mask input4
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},      // actual_seq_lengths_q
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},     // actual_seq_lengths_kv
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                               // deq_scale1 input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                // quant_scale1 input6
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                               // deq_scale2 input7
            {{{1, 1, 40, 128}, {1, 1, 40, 128}}, ge::DT_BF16, ge::FORMAT_ND},       // quant_scale2 input8
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}                                  // quant_offset2 input9
        },
        {{{{8, 2048, 40, 128}, {8, 2048, 40, 128}}, ge::DT_INT8, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0884f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1000)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    int64_t expectTilingKey = 1000000000000121112;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// TND
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_3)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        // 硬件参数
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};

    int64_t actual_seq_qlist[] = {409,   818,   1227,  1636,  2045,  2454,  2863,  3272,  3681,  4090,
                                  4499,  4908,  5317,  5726,  6135,  6544,  6953,  7362,  7771,  8180,
                                  8589,  8998,  9407,  9816,  10225, 10634, 11043, 11452, 11861, 12270,
                                  12679, 13088, 13497, 13906, 14315, 14724, 15133, 15542, 15951, 16384};
    int64_t actual_seq_kvlist[] = {409,   818,   1227,  1636,  2045,  2454,  2863,  3272,  3681,  4090,
                                   4499,  4908,  5317,  5726,  6135,  6544,  6953,  7362,  7771,  8180,
                                   8589,  8998,  9407,  9816,  10225, 10634, 11043, 11452, 11861, 12270,
                                   12679, 13088, 13497, 13906, 14315, 14724, 15133, 15542, 15951, 16384};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {
            {{{16384, 64, 192}, {16384, 64, 192}}, ge::DT_BF16, ge::FORMAT_ND},   // query input0
            {{{16384, 64, 192}, {16384, 64, 192}}, ge::DT_BF16, ge::FORMAT_ND},   // key input1
            {{{16384, 64, 192}, {16384, 64, 192}}, ge::DT_BF16, ge::FORMAT_ND},   // value input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                            // pse_shift input3
            {{{1, 2048, 2048}, {1, 2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},     // atten_mask input4
            {{{40}, {40}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},  // actual_seq_lengths_q
            {{{40}, {40}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist}, // actual_seq_lengths_kv
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                             // deq_scale1 input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // quant_scale1 input6
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                             // deq_scale2 input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // quant_scale2 input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                               // quant_offset2 input9
        },
        {{{{16384, 64, 192}, {16384, 64, 192}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.07216878364870323f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16384)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16384)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    int64_t expectTilingKey = 4000000000000000000;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// k = 0/v = 0/out = 0
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_4)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        // 硬件参数
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
    int64_t actual_seq_qlist[] = {2048};
    int64_t actual_seq_kvlist[] = {2048};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {
            {{{0, 5, 2048, 128}, {0, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query input0
            {{{0, 5, 2048, 128}, {0, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key input1
            {{{0, 5, 2048, 128}, {0, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value input2
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                  // pse_shift input3
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},                                 // atten_mask input4
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},       // actual_seq_lengths_q
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},      // actual_seq_lengths_kv
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                // deq_scale1 input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale1 input6
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                // deq_scale2 input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale2 input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                  // quant_offset2 input9
        },
        {{{{0, 5, 2048, 128}, {0, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-100)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-100)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    int64_t expectTilingKey = 1000000000000000020;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// num_heads < 0 || num_key_value_heads < 0
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_5)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        // 硬件参数
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
    int64_t actual_seq_qlist[] = {2048};
    int64_t actual_seq_kvlist[] = {2048};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query input0
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key input1
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value input2
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                  // pse_shift input3
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},                                 // atten_mask input4
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},       // actual_seq_lengths_q
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},      // actual_seq_lengths_kv
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                // deq_scale1 input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale1 input6
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                // deq_scale2 input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale2 input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                  // quant_offset2 input9
        },
        {{{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-100)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-100)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    int64_t expectTilingKey = 132383488;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

//  enableIFA = true
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_6)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        // 硬件参数
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
    int64_t *actual_seq_qlist = nullptr;
    int64_t *actual_seq_kvlist = nullptr;
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {
            {{{1, 5, 1, 128}, {1, 5, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // query input0
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key input1
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                               // pse_shift input3
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                               // atten_mask input4
            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},       // actual_seq_lengths_q
            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},      // actual_seq_lengths_kv
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                // deq_scale1 input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale1 input6
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                // deq_scale2 input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale2 input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                  // quant_offset2 input9
        },
        {{{{1, 5, 1, 128}, {1, 5, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    int64_t expectTilingKey = 1000000000000001012;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_7)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
    int64_t *actual_seq_qlist = nullptr;
    int64_t *actual_seq_kvlist = nullptr;
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {
            {{{1, 4096, 640}, {1, 4096, 640}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query input0
            {{{1, 4096, 640}, {1, 4096, 640}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key input1
            {{{1, 4096, 640}, {1, 4096, 640}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                         // pse_shift input3
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // atten_mask input4 (FP16 mask, not supported with BF16)
            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},  // actual_seq_lengths_q
            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist}, // actual_seq_lengths_kv
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                           // deq_scale1 input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // quant_scale1 input6
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                           // deq_scale2 input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // quant_scale2 input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                             // quant_offset2 input9
        },
        {{{{1, 4096, 640}, {1, 4096, 640}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(10)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(10)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// BNSD layout with mismatched D dimensions between query and key/value
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_8)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
    int64_t actual_seq_qlist[] = {2048};
    int64_t actual_seq_kvlist[] = {2048};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query input0 D=128
            {{{1, 5, 2048, 64}, {1, 5, 2048, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // key input1 D=64 mismatch
            {{{1, 5, 2048, 64}, {1, 5, 2048, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // value input2 D=64 mismatch
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                               // pse_shift input3
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},                                 // atten_mask input4
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},       // actual_seq_lengths_q
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},      // actual_seq_lengths_kv
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                // deq_scale1 input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale1 input6
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                // deq_scale2 input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale2 input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                  // quant_offset2 input9
        },
        {{{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// key dtype and value dtype must be consistent
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_9)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
    int64_t actual_seq_qlist[] = {2048};
    int64_t actual_seq_kvlist[] = {2048};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query input0
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_INT8, ge::FORMAT_ND},    // key input1 (INT8)
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}},
             ge::DT_FLOAT16,
             ge::FORMAT_ND},                                                    // value input2 (FP16) dtype mismatch
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // pse_shift input3
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},                            // atten_mask input4
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},  // actual_seq_lengths_q
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist}, // actual_seq_lengths_kv
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                           // deq_scale1 input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // quant_scale1 input6
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                           // deq_scale2 input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // quant_scale2 input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                             // quant_offset2 input9
        },
        {{{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// num_heads / num_key_value_heads ratio exceeds the maximum allowed value of 64
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_10)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
    int64_t actual_seq_qlist[] = {64};
    int64_t actual_seq_kvlist[] = {64};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {
            {{{1, 128, 64, 128}, {1, 128, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query input0 N=128
            {{{1, 1, 64, 128}, {1, 1, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // key input1 N=1
            {{{1, 1, 64, 128}, {1, 1, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // value input2 N=1
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                               // pse_shift input3
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},                                 // atten_mask input4
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},       // actual_seq_lengths_q
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},      // actual_seq_lengths_kv
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                // deq_scale1
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale1
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                // deq_scale2
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale2
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                  // quant_offset2
        },
        {{{{1, 128, 64, 128}, {1, 128, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // ratio=128/1=128 > 64
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// TND BF16 D=192, sparse_mode=0 (no attenMask) - BENCHMARK_TILING_KEY_1 path
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_11)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
    int64_t actual_seq_qlist[] = {409,   818,   1227,  1636,  2045,  2454,  2863,  3272,  3681,  4090,
                                  4499,  4908,  5317,  5726,  6135,  6544,  6953,  7362,  7771,  8180,
                                  8589,  8998,  9407,  9816,  10225, 10634, 11043, 11452, 11861, 12270,
                                  12679, 13088, 13497, 13906, 14315, 14724, 15133, 15542, 15951, 16384};
    int64_t actual_seq_kvlist[] = {409,   818,   1227,  1636,  2045,  2454,  2863,  3272,  3681,  4090,
                                   4499,  4908,  5317,  5726,  6135,  6544,  6953,  7362,  7771,  8180,
                                   8589,  8998,  9407,  9816,  10225, 10634, 11043, 11452, 11861, 12270,
                                   12679, 13088, 13497, 13906, 14315, 14724, 15133, 15542, 15951, 16384};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {
            {{{16384, 64, 192}, {16384, 64, 192}}, ge::DT_BF16, ge::FORMAT_ND}, // query input0
            {{{16384, 64, 192}, {16384, 64, 192}}, ge::DT_BF16, ge::FORMAT_ND}, // key input1
            {{{16384, 64, 192}, {16384, 64, 192}}, ge::DT_BF16, ge::FORMAT_ND}, // value input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // pse_shift input3
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND}, // atten_mask input4 (null, sparse_mode=0 requires no mask)
            {{{40}, {40}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},  // actual_seq_lengths_q
            {{{40}, {40}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist}, // actual_seq_lengths_kv
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                             // deq_scale1 input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // quant_scale1 input6
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                             // deq_scale2 input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // quant_scale2 input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                               // quant_offset2 input9
        },
        {{{{16384, 64, 192}, {16384, 64, 192}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.07216878364870323f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16384)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16384)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    int64_t expectTilingKey = 4000000000000000001;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// TND BF16 D=128, sparse_mode=0 (no attenMask) - BENCHMARK_TILING_KEY_3 path
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_12)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
    int64_t actual_seq_qlist[] = {409,   818,   1227,  1636,  2045,  2454,  2863,  3272,  3681,  4090,
                                  4499,  4908,  5317,  5726,  6135,  6544,  6953,  7362,  7771,  8180,
                                  8589,  8998,  9407,  9816,  10225, 10634, 11043, 11452, 11861, 12270,
                                  12679, 13088, 13497, 13906, 14315, 14724, 15133, 15542, 15951, 16384};
    int64_t actual_seq_kvlist[] = {409,   818,   1227,  1636,  2045,  2454,  2863,  3272,  3681,  4090,
                                   4499,  4908,  5317,  5726,  6135,  6544,  6953,  7362,  7771,  8180,
                                   8589,  8998,  9407,  9816,  10225, 10634, 11043, 11452, 11861, 12270,
                                   12679, 13088, 13497, 13906, 14315, 14724, 15133, 15542, 15951, 16384};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {
            {{{16384, 64, 128}, {16384, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // query input0
            {{{16384, 64, 128}, {16384, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // key input1
            {{{16384, 64, 128}, {16384, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // value input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // pse_shift input3
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND}, // atten_mask input4 (null, sparse_mode=0 requires no mask)
            {{{40}, {40}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},  // actual_seq_lengths_q
            {{{40}, {40}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist}, // actual_seq_lengths_kv
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                             // deq_scale1 input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // quant_scale1 input6
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                             // deq_scale2 input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // quant_scale2 input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                               // quant_offset2 input9
        },
        {{{{16384, 64, 128}, {16384, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16384)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16384)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    int64_t expectTilingKey = 4000000000000000003;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// BSH FP16, GQA ratio=4 (N_q=8, N_kv=2), with actualSeqLengths → GRAPH_SUCCESS
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_13)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
    int64_t actual_seq_qlist[] = {512, 1024};
    int64_t actual_seq_kvlist[] = {512, 1024};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {{{{2, 1024, 1024}, {2, 1024, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query B=2,S=1024,H_q=8*128=1024
         {{{2, 1024, 256}, {2, 1024, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // key   H_kv=2*128=256
         {{{2, 1024, 256}, {2, 1024, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // value
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2, 1024, 1024}, {2, 1024, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}, // GQA ratio=4
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    int64_t expectTilingKey = 1000000000000101012;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// BSH FP16, sparse_mode=22，ALIBI mask 走 Base API 成功路径
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_14)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
    int64_t actual_seq_qlist[] = {2048};
    int64_t actual_seq_kvlist[] = {2048};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {{{{1, 2048, 4096}, {1, 2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 2048, 4096}, {1, 2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 2048, 4096}, {1, 2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{32, 2048, 2048}, {32, 2048, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 2048, 4096}, {1, 2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(22)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// DT_FLOAT input/output is not supported
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_15)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {{{{1, 8, 1024, 128}, {1, 8, 1024, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1, 8, 1024, 128}, {1, 8, 1024, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1, 8, 1024, 128}, {1, 8, 1024, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 8, 1024, 128}, {1, 8, 1024, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// SH FP16, sparse_mode=22, alibi mask via pse_shift -> Base API alibi path
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_16)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
    int64_t actual_seq_qlist[] = {2048};
    int64_t actual_seq_kvlist[] = {2048};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {{{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{32, 2048, 2048}, {32, 2048, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SH")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(22)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 2000000000004001012);
}

// SH BF16, sparse_mode=20, no mask -> Base API BF16 success path
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_17)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
    int64_t actual_seq_qlist[] = {2048};
    int64_t actual_seq_kvlist[] = {2048};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {{{{2048, 4096}, {2048, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{2048, 4096}, {2048, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{2048, 4096}, {2048, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2048, 4096}, {2048, 4096}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SH")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 2000000002004010112);
}

// SH FP16, sparse_mode=20, actual seq tensors absent -> Base API seq null branch
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_18)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {{{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         {{}, ge::DT_UNDEFINED, ge::FORMAT_NULL},
         {{}, ge::DT_UNDEFINED, ge::FORMAT_NULL},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SH")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// SH FP16, sparse_mode=21, 3D norm mask -> Base API 3D mask stride path
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_19)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
    int64_t actual_seq_qlist[] = {2048};
    int64_t actual_seq_kvlist[] = {2048};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {{{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 2048, 2048}, {1, 2048, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SH")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(21)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 2000000000004001012);
}

// SH FP16, sparse_mode=21, long-seq 2D norm mask -> Base API long-seq mask branch
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_20)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
    int64_t actual_seq_qlist[] = {8192};
    int64_t actual_seq_kvlist[] = {8192};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {{{{8192, 4096}, {8192, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{8192, 4096}, {8192, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{8192, 4096}, {8192, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{8192, 8192}, {8192, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{8192, 4096}, {8192, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SH")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(21)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 2000000000004001012);
}

// SH FP16, sparse_mode=22, 4D alibi mask -> Base API 4D alibi stride path
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_21)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
    int64_t actual_seq_qlist[] = {2048};
    int64_t actual_seq_kvlist[] = {2048};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {{{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 32, 2048, 2048}, {1, 32, 2048, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SH")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(22)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 2000000000004001012);
}

// BSND BF16, sparse_mode=20 -> current arch22 host path still rejects this Base API layout combination
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_22)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
    int64_t actual_seq_qlist[] = {2048};
    int64_t actual_seq_kvlist[] = {2048};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {{{{1, 2048, 32, 128}, {1, 2048, 32, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 2048, 32, 128}, {1, 2048, 32, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 2048, 32, 128}, {1, 2048, 32, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 2048, 32, 128}, {1, 2048, 32, 128}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// BNSD BF16, sparse_mode=20 -> current arch22 host path still rejects this Base API layout combination
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_23)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
    int64_t actual_seq_qlist[] = {2048};
    int64_t actual_seq_kvlist[] = {2048};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {{{{1, 32, 2048, 128}, {1, 32, 2048, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 32, 2048, 128}, {1, 32, 2048, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 32, 2048, 128}, {1, 32, 2048, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 32, 2048, 128}, {1, 32, 2048, 128}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// SH INT8->FP16, q head size=1, invalid deq_scale2 shape -> Base API deqScale2 validation branch
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_24)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
    int64_t actual_seq_qlist[] = {128};
    int64_t actual_seq_kvlist[] = {128};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {{{{128, 32}, {128, 32}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{128, 32}, {128, 32}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{128, 32}, {128, 32}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
         {{{1}, {1}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{128, 32}, {128, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SH")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// SH FP16, sparse_mode=22, 3D alibi mask with long-seq compress pattern -> CheckBaseApiAlibiMask shape rejection branch
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_25)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
    int64_t actual_seq_qlist[] = {128};
    int64_t actual_seq_kvlist[] = {128};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {{{{128, 4096}, {128, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{128, 4096}, {128, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{128, 4096}, {128, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{32, 64, 128}, {32, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{128, 4096}, {128, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SH")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(22)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// SH FP16, sparse_mode=22, 2D alibi long-seq mask with maxSeqLen=128 -> SetBaseApiAlibiMaskInfo length rejection
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_26)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
    int64_t actual_seq_qlist[] = {128};
    int64_t actual_seq_kvlist[] = {128};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {{{{128, 4096}, {128, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{128, 4096}, {128, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{128, 4096}, {128, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{256, 256}, {256, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{128, 4096}, {128, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SH")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(22)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// SH FP16, sparse_mode=21, 4D mask with last dim=128 -> confirm current Base API path still accepts this shape
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_27)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
    int64_t actual_seq_qlist[] = {128};
    int64_t actual_seq_kvlist[] = {128};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {{{{128, 4096}, {128, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{128, 4096}, {128, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{128, 4096}, {128, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 32, 64, 128}, {1, 32, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{128, 4096}, {128, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SH")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(21)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 2000000000004001012);
}

// TND BF16, sparse_mode=4, first batch zero-length -> CheckVarLenPreNextToken continue branch
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_28)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
    int64_t actual_seq_qlist[] = {0, 40, 80, 120};
    int64_t actual_seq_kvlist[] = {0, 40, 80, 120};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {{{{120, 64, 128}, {120, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{120, 64, 128}, {120, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{120, 64, 128}, {120, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 2048, 2048}, {1, 2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},
         {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
         {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{120, 64, 128}, {120, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(10)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-10)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// BNSD FP16, valid 4D pse_shift -> non-Base PFA pse path
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_29)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {{{{1, 8, 512, 128}, {1, 8, 512, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 8, 512, 128}, {1, 8, 512, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 8, 512, 128}, {1, 8, 512, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 8, 512, 512}, {1, 8, 512, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 8, 512, 128}, {1, 8, 512, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// BSND FP16 -> INT8, quant offset present and actual_seq_kv omitted -> post-quant + kv-length fallback path
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_30)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
    int64_t actual_seq_qlist[] = {512};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {{{{1, 512, 8, 128}, {1, 512, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 512, 8, 128}, {1, 512, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 512, 8, 128}, {1, 512, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{1, 512, 8, 128}, {1, 512, 8, 128}}, ge::DT_INT8, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// BNSD FP16, sparse_mode=4 without actual_seq inputs -> band loop fallback path
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_31)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {{{{1, 8, 2048, 128}, {1, 8, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 8, 2048, 128}, {1, 8, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 8, 2048, 128}, {1, 8, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 1, 2048, 2048}, {1, 1, 2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 8, 2048, 128}, {1, 8, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// BNSD FP16, inner_precise=5 -> warning branch followed by normal HIGH_PERFORMANCE fallback
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_32)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
    int64_t actual_seq_qlist[] = {512};
    int64_t actual_seq_kvlist[] = {512};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {{{{1, 8, 2048, 128}, {1, 8, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 8, 2048, 128}, {1, 8, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 8, 2048, 128}, {1, 8, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 1, 2048, 2048}, {1, 1, 2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 8, 2048, 128}, {1, 8, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// NSD FP16, 3D query/key/value -> cover NSD head-num success path
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_33)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {{{{8, 512, 128}, {8, 512, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{8, 512, 128}, {8, 512, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{8, 512, 128}, {8, 512, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{8, 512, 128}, {8, 512, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("NSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// NSD FP16, 4D query/key/value -> cover NSD dim-num reject branch in CheckInputDimAndHeadNum
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_34)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {{{{1, 8, 512, 128}, {1, 8, 512, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 8, 512, 128}, {1, 8, 512, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 8, 512, 128}, {1, 8, 512, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 8, 512, 128}, {1, 8, 512, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("NSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// BSH FP16, query/key/value head size mismatch -> cover old-path BSH D-consistency rejection
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_35)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {{{{1, 128, 1024}, {1, 128, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 128, 512}, {1, 128, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 128, 512}, {1, 128, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 128, 1024}, {1, 128, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// TND BF16, sparse_mode=3, actual_seq_q > actual_seq_kv -> cover old-path var-len right-down reject
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_36)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
    int64_t actual_seq_qlist[] = {96, 128};
    int64_t actual_seq_kvlist[] = {64, 128};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {{{{128, 64, 128}, {128, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{128, 64, 128}, {128, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{128, 64, 128}, {128, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 2048, 2048}, {1, 2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{128, 64, 128}, {128, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// SH FP16, actual_seq_lengths_q omitted but actual_seq_lengths_kv present ->
// cover SH warning + 2D SH basic-shape branch with b=1 fallback.
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_37)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
    int64_t *actual_seq_qlist = nullptr;
    int64_t actual_seq_kvlist[] = {64};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {{{{64, 4096}, {64, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 4096}, {64, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 4096}, {64, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{64, 4096}, {64, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SH")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Long-seq BSH FP16 -> probe the OneN split path in old PFA.
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_38)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
    int64_t *actual_seq_qlist = nullptr;
    int64_t *actual_seq_kvlist = nullptr;
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {{{{1, 16384, 2048}, {1, 16384, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 16384, 2048}, {1, 16384, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 16384, 2048}, {1, 16384, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
         {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 16384, 2048}, {1, 16384, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}
