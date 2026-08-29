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

class PromptFlashAttentionTiling310P : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        // std::cout << "PromptFlashAttentionTiling310P SetUp" << std::endl;
        SoftmaxTilingMocker::GetInstance().SetSocVersion("Ascend310P3");
    }

    static void TearDownTestCase()
    {
        // std::cout << "PromptFlashAttentionTiling310P TearDown" << std::endl;
        SoftmaxTilingMocker::GetInstance().Reset();
    }
};

// 310P allows APPROXIMATE_COMPUTATION and reaches the fp16 softmax datatype adjustment path
TEST_F(PromptFlashAttentionTiling310P, PromptFlashAttention_310p_tiling_0)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        8, 8, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND310P};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {{{{1, 8, 1024, 128}, {1, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 8, 1024, 128}, {1, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 8, 1024, 128}, {1, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 8, 1024, 128}, {1, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)}},
        &compileInfo, "Ascend310P", 8, 262144, 16384);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// SH FP16 on 310P, omit num_key_value_heads -> current host path first falls into head-ratio validation
TEST_F(PromptFlashAttentionTiling310P, PromptFlashAttention_310p_tiling_1)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        8, 8, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND310P};
    int64_t actual_seq_qlist[] = {16};
    int64_t actual_seq_kvlist[] = {16};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {{{{1, 1, 16, 4096}, {1, 1, 16, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 1, 16, 4096}, {1, 1, 16, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 1, 16, 4096}, {1, 1, 16, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 1, 16, 4096}, {1, 1, 16, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SH")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend310P", 8, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// SH FP16 on 310P, omit num_heads -> keep probing 310P Base API null-attr handling
TEST_F(PromptFlashAttentionTiling310P, PromptFlashAttention_310p_tiling_2)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        8, 8, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND310P};
    int64_t actual_seq_qlist[] = {16};
    int64_t actual_seq_kvlist[] = {16};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {{{{1, 1, 16, 4096}, {1, 1, 16, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 1, 16, 4096}, {1, 1, 16, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 1, 16, 4096}, {1, 1, 16, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 1, 16, 4096}, {1, 1, 16, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SH")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend310P", 8, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 310P BNSD Base API, GQA group count larger than 64 -> cover headNum / kvHeadNum > 64 validation
TEST_F(PromptFlashAttentionTiling310P, PromptFlashAttention_310p_tiling_3)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        8, 8, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND310P};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {{{{1, 8, 16, 128}, {1, 8, 16, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 8, 16, 128}, {1, 8, 16, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 8, 16, 128}, {1, 8, 16, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 8, 16, 128}, {1, 8, 16, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(130)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend310P", 8, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 310P BNSD old path, INT8 attention mask -> cover maskElemSize=INT8 and bool-only rejection
TEST_F(PromptFlashAttentionTiling310P, PromptFlashAttention_310p_tiling_4)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        8, 8, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND310P};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {{{{1, 8, 1024, 128}, {1, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 8, 1024, 128}, {1, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 8, 1024, 128}, {1, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 1, 2048, 2048}, {1, 1, 2048, 2048}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 8, 1024, 128}, {1, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend310P", 8, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 310P BNSD old path, UINT8 attention mask -> cover maskElemSize=UINT8 and bool-only rejection
TEST_F(PromptFlashAttentionTiling310P, PromptFlashAttention_310p_tiling_5)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        8, 8, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND310P};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {{{{1, 8, 1024, 128}, {1, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 8, 1024, 128}, {1, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 8, 1024, 128}, {1, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 1, 2048, 2048}, {1, 1, 2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 8, 1024, 128}, {1, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend310P", 8, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
