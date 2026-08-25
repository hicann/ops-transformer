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
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

class KvQuantSparseFlashAttentionTilingArch35 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "KvQuantSparseFlashAttentionTilingArch35 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "KvQuantSparseFlashAttentionTilingArch35 TearDown" << std::endl;
    }
};

// 0
TEST_F(KvQuantSparseFlashAttentionTilingArch35, KvQuantSparseFlashAttention_950_tiling_0)
{
    struct KvQuantSparseFlashAttentionCompileInfo {
    } compileInfo;
    int32_t actualSeqQuery[] = {32, 64};
    int32_t actualSeqKv[] = {256, 512};
    gert::TilingContextPara tilingContextPara(
        "KvQuantSparseFlashAttention",
        {{{{2, 64, 8, 576}, {2, 64, 8, 576}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{32, 16, 1, 656}, {32, 16, 1, 656}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{32, 16, 1, 656}, {32, 16, 1, 656}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 32}, {2, 32}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqQuery},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqKv}},
        {{{{2, 64, 8, 512}, {2, 64, 8, 512}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.044194173f)},
         {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"value_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"rope_head_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)}},
        &compileInfo, "Ascend950", 64, 262144, 16384);

    int64_t expectTilingKey = 33922;
    std::string expectTilingData = "2199023255554 64 16 8589934624 4410436851802832898 8 3 1 128 656 0 "
                                   "10496 40532396646334464 262144 64 316659348865024 2199023255560 ";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// 1
TEST_F(KvQuantSparseFlashAttentionTilingArch35, KvQuantSparseFlashAttention_950_tiling_1)
{
    struct KvQuantSparseFlashAttentionCompileInfo {
    } compileInfo;
    int32_t actualSeqQuery[] = {48, 96};
    int32_t actualSeqKv[] = {128, 256};
    gert::TilingContextPara tilingContextPara(
        "KvQuantSparseFlashAttention",
        {{{{96, 16, 576}, {96, 16, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{256, 1, 656}, {256, 1, 656}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
         {{{256, 1, 656}, {256, 1, 656}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
         {{{96, 1, 2048}, {96, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqQuery},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqKv}},
        {{{{96, 16, 512}, {96, 16, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.044194173f)},
         {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"value_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"rope_head_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)}},
        &compileInfo, "Ascend950", 64, 262144, 16384);

    int64_t expectTilingKey = 1092;
    std::string expectTilingData = "1099511627778 96 0 8589934592 4410436851802832898 4294967312 0 1 2048 "
                                   "656 0 0 40532396646334464 262144 64 316659348832256 1099511627792 ";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// sparse block size must be 1, but got 2
TEST_F(KvQuantSparseFlashAttentionTilingArch35, KvQuantSparseFlashAttention_950_tiling_2)
{
    struct KvQuantSparseFlashAttentionCompileInfo {
    } compileInfo;
    int32_t actualSeqQuery[] = {32, 64};
    int32_t actualSeqKv[] = {256, 512};
    gert::TilingContextPara tilingContextPara(
        "KvQuantSparseFlashAttention",
        {{{{2, 64, 8, 576}, {2, 64, 8, 576}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{32, 16, 1, 656}, {32, 16, 1, 656}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{32, 16, 1, 656}, {32, 16, 1, 656}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 32}, {2, 32}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqQuery},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqKv}},
        {{{{2, 64, 8, 512}, {2, 64, 8, 512}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.044194173f)},
         {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"value_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"rope_head_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)}},
        &compileInfo, "Ascend950", 64, 262144, 16384);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
