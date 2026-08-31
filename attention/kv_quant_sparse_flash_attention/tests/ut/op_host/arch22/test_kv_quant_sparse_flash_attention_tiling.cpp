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

class KvQuantSparseFlashAttentionTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "KvQuantSparseFlashAttentionTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "KvQuantSparseFlashAttentionTiling TearDown" << std::endl;
    }
};

TEST_F(KvQuantSparseFlashAttentionTiling, KvQuantSparseFlashAttention_910b_0)
{
    struct KvQuantSparseFlashAttentionCompileInfo {
    } compileInfo;
    int32_t actualSeqQuery[] = {32, 64};
    int32_t actualSeqKv[] = {256, 512};
    gert::TilingContextPara tilingContextPara(
        "KvQuantSparseFlashAttention",
        {{{{2, 64, 8, 576}, {2, 64, 8, 576}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{32, 16, 1, 656}, {32, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{32, 16, 1, 656}, {32, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
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
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"rope_head_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    int64_t expectTilingKey = 1154;
    std::string expectTilingData = "2199023255554 64 16 8589934624 4410436851802832898 8 3 16 128 656 0 "
                                   "0 40532396646334464 262144 64 316659348865024 2199023255560 ";
    std::vector<size_t> expectWorkspaces = {293928960};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(KvQuantSparseFlashAttentionTiling, KvQuantSparseFlashAttention_910b_1)
{
    struct KvQuantSparseFlashAttentionCompileInfo {
    } compileInfo;
    int32_t actualSeqQuery[] = {48, 96};
    int32_t actualSeqKv[] = {128, 256};
    gert::TilingContextPara tilingContextPara(
        "KvQuantSparseFlashAttention",
        {{{{96, 16, 576}, {96, 16, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{256, 1, 656}, {256, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{256, 1, 656}, {256, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{96, 1, 64}, {96, 1, 64}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqQuery},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqKv}},
        {{{{96, 16, 512}, {96, 16, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.044194173f)},
         {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"value_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"rope_head_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    int64_t expectTilingKey = 1092;
    std::string expectTilingData = "1099511627778 96 0 8589934592 4410436851802832898 4294967312 0 8 64 "
                                   "656 0 0 40532396646334464 262144 64 316659348832256 1099511627792 ";
    std::vector<size_t> expectWorkspaces = {268763136};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// attention_mode should be 2(MLA-absorb), but got 0
TEST_F(KvQuantSparseFlashAttentionTiling, KvQuantSparseFlashAttention_910b_2)
{
    struct KvQuantSparseFlashAttentionCompileInfo {
    } compileInfo;
    int32_t actualSeqQuery[] = {32, 64};
    int32_t actualSeqKv[] = {256, 512};
    gert::TilingContextPara tilingContextPara(
        "KvQuantSparseFlashAttention",
        {
            {{{2, 64, 8, 576}, {2, 64, 8, 576}}, ge::DT_BF16, ge::FORMAT_ND},   // query
            {{{32, 16, 1, 656}, {32, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND}, // key
            {{{32, 16, 1, 656}, {32, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND}, // value
            {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_INT32, ge::FORMAT_ND},  // sparse_indices
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // key_dequant_scale
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // value_dequant_scale
            {{{2, 32}, {2, 32}}, ge::DT_INT32, ge::FORMAT_ND},                  // block_table
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqQuery},    // actual_seq_lengths_query
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqKv}        // actual_seq_lengths_kv
        },
        {{{{2, 64, 8, 512}, {2, 64, 8, 512}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.044194173f)},
         {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"value_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"rope_head_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// sparseBlockSize should be in range [1, 16] and be a power of 2, but got: 3.
TEST_F(KvQuantSparseFlashAttentionTiling, KvQuantSparseFlashAttention_910b_3)
{
    struct KvQuantSparseFlashAttentionCompileInfo {
    } compileInfo;
    int32_t actualSeqQuery[] = {32, 64};
    int32_t actualSeqKv[] = {256, 512};
    gert::TilingContextPara tilingContextPara(
        "KvQuantSparseFlashAttention",
        {{{{2, 64, 8, 576}, {2, 64, 8, 576}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{32, 16, 1, 656}, {32, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{32, 16, 1, 656}, {32, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
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
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"rope_head_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// attention_out layout is BSND, shape is [2, 64, 8, 576], expected shape is [2, 64, 8, 512].
TEST_F(KvQuantSparseFlashAttentionTiling, KvQuantSparseFlashAttention_910b_4)
{
    struct KvQuantSparseFlashAttentionCompileInfo {
    } compileInfo;
    int32_t actualSeqQuery[] = {32, 64};
    int32_t actualSeqKv[] = {256, 512};
    gert::TilingContextPara tilingContextPara(
        "KvQuantSparseFlashAttention",
        {{{{2, 64, 8, 576}, {2, 64, 8, 576}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{32, 16, 1, 656}, {32, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{32, 16, 1, 656}, {32, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 32}, {2, 32}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqQuery},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqKv}},
        {{{{2, 64, 8, 576}, {2, 64, 8, 576}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.044194173f)},
         {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"value_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"rope_head_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// key_quant_mode should be 2(per-tile), but got 1
TEST_F(KvQuantSparseFlashAttentionTiling, KvQuantSparseFlashAttention_910b_5)
{
    struct KvQuantSparseFlashAttentionCompileInfo {
    } compileInfo;
    int32_t actualSeqQuery[] = {32, 64};
    int32_t actualSeqKv[] = {256, 512};
    gert::TilingContextPara tilingContextPara(
        "KvQuantSparseFlashAttention",
        {{{{2, 64, 8, 576}, {2, 64, 8, 576}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{32, 16, 1, 656}, {32, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{32, 16, 1, 656}, {32, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 32}, {2, 32}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqQuery},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqKv}},
        {{{{2, 64, 8, 512}, {2, 64, 8, 512}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.044194173f)},
         {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"value_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"rope_head_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// quant_scale_repo_mode should be 1(combine), but got 0
TEST_F(KvQuantSparseFlashAttentionTiling, KvQuantSparseFlashAttention_910b_6)
{
    struct KvQuantSparseFlashAttentionCompileInfo {
    } compileInfo;
    int32_t actualSeqQuery[] = {32, 64};
    int32_t actualSeqKv[] = {256, 512};
    gert::TilingContextPara tilingContextPara(
        "KvQuantSparseFlashAttention",
        {{{{2, 64, 8, 576}, {2, 64, 8, 576}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{32, 16, 1, 656}, {32, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{32, 16, 1, 656}, {32, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
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
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"rope_head_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// rope_head_dim should be 64, but got 32
TEST_F(KvQuantSparseFlashAttentionTiling, KvQuantSparseFlashAttention_910b_7)
{
    struct KvQuantSparseFlashAttentionCompileInfo {
    } compileInfo;
    int32_t actualSeqQuery[] = {32, 64};
    int32_t actualSeqKv[] = {256, 512};
    gert::TilingContextPara tilingContextPara(
        "KvQuantSparseFlashAttention",
        {{{{2, 64, 8, 576}, {2, 64, 8, 576}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{32, 16, 1, 656}, {32, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{32, 16, 1, 656}, {32, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
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
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"rope_head_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// when layout_kv is TND, actualSeqLengthsKv must not be null
TEST_F(KvQuantSparseFlashAttentionTiling, KvQuantSparseFlashAttention_910b_8)
{
    struct KvQuantSparseFlashAttentionCompileInfo {
    } compileInfo;
    int32_t actualSeqQuery[] = {48, 96};
    gert::TilingContextPara tilingContextPara(
        "KvQuantSparseFlashAttention",
        {{{{96, 16, 576}, {96, 16, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{256, 1, 656}, {256, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{256, 1, 656}, {256, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{96, 1, 64}, {96, 1, 64}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqQuery},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{96, 16, 512}, {96, 16, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.044194173f)},
         {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"value_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"rope_head_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// when layout_kv is PA_BSND, actualSeqLengthsKv must not be null
TEST_F(KvQuantSparseFlashAttentionTiling, KvQuantSparseFlashAttention_910b_9)
{
    struct KvQuantSparseFlashAttentionCompileInfo {
    } compileInfo;
    int32_t actualSeqQuery[] = {32, 64};
    gert::TilingContextPara tilingContextPara(
        "KvQuantSparseFlashAttention",
        {{{{2, 64, 8, 576}, {2, 64, 8, 576}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{32, 16, 1, 656}, {32, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{32, 16, 1, 656}, {32, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 32}, {2, 32}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqQuery},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{2, 64, 8, 512}, {2, 64, 8, 512}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.044194173f)},
         {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"value_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"rope_head_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// the layout_kv is PA_BSND, blockTable must be provided.
TEST_F(KvQuantSparseFlashAttentionTiling, KvQuantSparseFlashAttention_910b_10)
{
    struct KvQuantSparseFlashAttentionCompileInfo {
    } compileInfo;
    int32_t actualSeqQuery[] = {32, 64};
    int32_t actualSeqKv[] = {256, 512};
    gert::TilingContextPara tilingContextPara(
        "KvQuantSparseFlashAttention",
        {{{{2, 64, 8, 576}, {2, 64, 8, 576}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{32, 16, 1, 656}, {32, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{32, 16, 1, 656}, {32, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqQuery},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqKv}},
        {{{{2, 64, 8, 512}, {2, 64, 8, 512}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.044194173f)},
         {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"value_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"rope_head_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// layoutKV is SBH, it is unsupported.
TEST_F(KvQuantSparseFlashAttentionTiling, KvQuantSparseFlashAttention_910b_11)
{
    struct KvQuantSparseFlashAttentionCompileInfo {
    } compileInfo;
    int32_t actualSeqQuery[] = {32, 64};
    int32_t actualSeqKv[] = {256, 512};
    gert::TilingContextPara tilingContextPara(
        "KvQuantSparseFlashAttention",
        {{{{2, 64, 8, 576}, {2, 64, 8, 576}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{32, 16, 1, 656}, {32, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{32, 16, 1, 656}, {32, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
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
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("SBH")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"rope_head_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// When layoutKV is not PA_BSND, layoutKV must be the same as layoutQ.
TEST_F(KvQuantSparseFlashAttentionTiling, KvQuantSparseFlashAttention_910b_12)
{
    struct KvQuantSparseFlashAttentionCompileInfo {
    } compileInfo;
    int32_t actualSeqQuery[] = {48, 96};
    int32_t actualSeqKv[] = {128, 256};
    gert::TilingContextPara tilingContextPara(
        "KvQuantSparseFlashAttention",
        {{{{96, 16, 576}, {96, 16, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{256, 1, 656}, {256, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{256, 1, 656}, {256, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{96, 1, 64}, {96, 1, 64}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqQuery},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqKv}},
        {{{{96, 16, 512}, {96, 16, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.044194173f)},
         {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"value_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"rope_head_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// layoutQuery is SBH, it is unsupported.
TEST_F(KvQuantSparseFlashAttentionTiling, KvQuantSparseFlashAttention_910b_13)
{
    struct KvQuantSparseFlashAttentionCompileInfo {
    } compileInfo;
    int32_t actualSeqQuery[] = {32, 64};
    int32_t actualSeqKv[] = {256, 512};
    gert::TilingContextPara tilingContextPara(
        "KvQuantSparseFlashAttention",
        {{{{2, 64, 8, 576}, {2, 64, 8, 576}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{32, 16, 1, 656}, {32, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{32, 16, 1, 656}, {32, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
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
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("SBH")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"rope_head_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// the dim num of block_table is 3, it should be 2.
TEST_F(KvQuantSparseFlashAttentionTiling, KvQuantSparseFlashAttention_910b_14)
{
    struct KvQuantSparseFlashAttentionCompileInfo {
    } compileInfo;
    int32_t actualSeqQuery[] = {32, 64};
    int32_t actualSeqKv[] = {256, 512};
    gert::TilingContextPara tilingContextPara(
        "KvQuantSparseFlashAttention",
        {
            {{{2, 64, 8, 576}, {2, 64, 8, 576}}, ge::DT_BF16, ge::FORMAT_ND},   // query
            {{{32, 16, 1, 656}, {32, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND}, // key
            {{{32, 16, 1, 656}, {32, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND}, // value
            {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_INT32, ge::FORMAT_ND},  // sparse_indices
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // key_dequant_scale
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // value_dequant_scale
            {{{2, 32, 1}, {2, 32, 1}}, ge::DT_INT32, ge::FORMAT_ND},            // block_table
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqQuery},    // actual_seq_lengths_query
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqKv}        // actual_seq_lengths_kv
        },
        {{{{2, 64, 8, 512}, {2, 64, 8, 512}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.044194173f)},
         {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"value_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"rope_head_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// block_table's second dimension(0) should be greater than 0
TEST_F(KvQuantSparseFlashAttentionTiling, KvQuantSparseFlashAttention_910b_15)
{
    struct KvQuantSparseFlashAttentionCompileInfo {
    } compileInfo;
    int32_t actualSeqQuery[] = {32, 64};
    int32_t actualSeqKv[] = {256, 512};
    gert::TilingContextPara tilingContextPara(
        "KvQuantSparseFlashAttention",
        {
            {{{2, 64, 8, 576}, {2, 64, 8, 576}}, ge::DT_BF16, ge::FORMAT_ND},   // query
            {{{32, 16, 1, 656}, {32, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND}, // key
            {{{32, 16, 1, 656}, {32, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND}, // value
            {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_INT32, ge::FORMAT_ND},  // sparse_indices
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // key_dequant_scale
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // value_dequant_scale
            {{{2, 0}, {2, 0}}, ge::DT_INT32, ge::FORMAT_ND},                    // block_table
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqQuery},    // actual_seq_lengths_query
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqKv}        // actual_seq_lengths_kv
        },
        {{{{2, 64, 8, 512}, {2, 64, 8, 512}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.044194173f)},
         {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"value_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"rope_head_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// actualSeqLengthsQ's dtype is DT_FLOAT, it should be DT_INT32.
TEST_F(KvQuantSparseFlashAttentionTiling, KvQuantSparseFlashAttention_910b_16)
{
    struct KvQuantSparseFlashAttentionCompileInfo {
    } compileInfo;
    int32_t actualSeqQuery[] = {32, 64};
    int32_t actualSeqKv[] = {256, 512};
    gert::TilingContextPara tilingContextPara(
        "KvQuantSparseFlashAttention",
        {
            {{{2, 64, 8, 576}, {2, 64, 8, 576}}, ge::DT_BF16, ge::FORMAT_ND},   // query
            {{{32, 16, 1, 656}, {32, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND}, // key
            {{{32, 16, 1, 656}, {32, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND}, // value
            {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_INT32, ge::FORMAT_ND},  // sparse_indices
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // key_dequant_scale
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // value_dequant_scale
            {{{2, 32}, {2, 32}}, ge::DT_INT32, ge::FORMAT_ND},                  // block_table
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND, true, actualSeqQuery},    // actual_seq_lengths_query
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND, true, actualSeqKv}        // actual_seq_lengths_kv
        },
        {{{{2, 64, 8, 512}, {2, 64, 8, 512}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.044194173f)},
         {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"value_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"rope_head_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// query dimension only supports 3, 4, but got 5
TEST_F(KvQuantSparseFlashAttentionTiling, KvQuantSparseFlashAttention_910b_17)
{
    struct KvQuantSparseFlashAttentionCompileInfo {
    } compileInfo;
    int32_t actualSeqQuery[] = {32, 64};
    int32_t actualSeqKv[] = {256, 512};
    gert::TilingContextPara tilingContextPara(
        "KvQuantSparseFlashAttention",
        {{{{2, 64, 8, 576, 5}, {2, 64, 8, 576, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{32, 16, 1, 656}, {32, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{32, 16, 1, 656}, {32, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
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
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"rope_head_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// query dimension only supports 3, 4, but got 5
TEST_F(KvQuantSparseFlashAttentionTiling, KvQuantSparseFlashAttention_910b_18)
{
    struct KvQuantSparseFlashAttentionCompileInfo {
    } compileInfo;
    int32_t actualSeqQuery[] = {32, 64};
    int32_t actualSeqKv[] = {256, 512};
    gert::TilingContextPara tilingContextPara(
        "KvQuantSparseFlashAttention",
        {{{{2, 64, 8, 576}, {2, 64, 8, 576}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{32, 16, 1, 656}, {32, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{32, 16, 1, 656}, {32, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
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
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"rope_head_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
