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
#include <string>

#include "../../../op_host/bsa_select_block_mask_tiling_base.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class BSASelectBlockMaskTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "--- BSASelectBlockMaskTiling UT SetUp ---" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "--- BSASelectBlockMaskTiling UT TearDown ---" << std::endl;
    }
};

// ============================================================================
// 测试用例 1: BNSD Layout 基本场景
// ============================================================================
TEST_F(BSASelectBlockMaskTilingTest, tiling_bnsd_case0)
{
    optiling::BSASelectBlockMaskCompileInfo compileInfo;

    int64_t b = 1, n = 4, s = 256, s_kv = 256, d = 128;
    int64_t blockX = 128, blockY = 128;
    int64_t ceilQ = (s + blockX - 1) / blockX;
    int64_t ceilKv = (s_kv + blockY - 1) / blockY;

    int64_t blockShapeData[2] = {blockX, blockY};
    int64_t actualSeqData[1] = {s};
    int64_t actualSeqKvData[1] = {s_kv};

    gert::TilingContextPara tilingContextPara(
        "BSASelectBlockMask",
        {// --- Input Info ---
         // 0: query [B, N, S, D]
         {{{b, n, s, d}, {b, n, s, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1: key [B, N, S_kv, D]
         {{{b, n, s_kv, d}, {b, n, s_kv, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 2: block_shape (OPTIONAL, has value)
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, (void *)blockShapeData},
         // 3: post_block_shape (OPTIONAL, not provided)
         {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND},
         // 4: actual_seq_lengths (OPTIONAL, has value)
         {{{b}, {b}}, ge::DT_INT64, ge::FORMAT_ND, true, (void *)actualSeqData},
         // 5: actual_seq_lengths_kv (OPTIONAL, has value)
         {{{b}, {b}}, ge::DT_INT64, ge::FORMAT_ND, true, (void *)actualSeqKvData},
         // 6: actual_block_len_query (OPTIONAL, not provided)
         {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND},
         // 7: actual_block_len_key (OPTIONAL, not provided)
         {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND}},
        {// --- Output Info ---
         // 0: block_sparse_mask_out [B, N, ceilQ, ceilKv]
         {{{b, n, ceilQ, ceilKv}, {b, n, ceilQ, ceilKv}}, ge::DT_INT8, ge::FORMAT_ND}},
        {// --- Attr Info (对齐 OpDef 里的 5 个属性顺序) ---
         {"q_input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"kv_input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(n)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"sparsity", Ops::Transformer::AnyValue::CreateFrom<float>(0.5f)}},
        &compileInfo);

    uint64_t expectTilingKey = 17UL;

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// ============================================================================
// 测试用例 2: 不同 block shape (64x64)
// ============================================================================
TEST_F(BSASelectBlockMaskTilingTest, tiling_bnsd_block64)
{
    optiling::BSASelectBlockMaskCompileInfo compileInfo;

    int64_t b = 1, n = 2, s = 512, s_kv = 512, d = 128;
    int64_t blockX = 64, blockY = 64;
    int64_t ceilQ = (s + blockX - 1) / blockX;
    int64_t ceilKv = (s_kv + blockY - 1) / blockY;

    int64_t blockShapeData[2] = {blockX, blockY};
    int64_t actualSeqData[1] = {s};
    int64_t actualSeqKvData[1] = {s_kv};

    gert::TilingContextPara tilingContextPara(
        "BSASelectBlockMask",
        {// 0: query
         {{{b, n, s, d}, {b, n, s, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1: key
         {{{b, n, s_kv, d}, {b, n, s_kv, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 2: block_shape
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, (void *)blockShapeData},
         // 3: post_block_shape
         {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND},
         // 4: actual_seq_lengths
         {{{b}, {b}}, ge::DT_INT64, ge::FORMAT_ND, true, (void *)actualSeqData},
         // 5: actual_seq_lengths_kv
         {{{b}, {b}}, ge::DT_INT64, ge::FORMAT_ND, true, (void *)actualSeqKvData},
         // 6: actual_block_len_query
         {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND},
         // 7: actual_block_len_key
         {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND}},
        {// 0: block_sparse_mask_out
         {{{b, n, ceilQ, ceilKv}, {b, n, ceilQ, ceilKv}}, ge::DT_INT8, ge::FORMAT_ND}},
        {{"q_input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"kv_input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(n)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"sparsity", Ops::Transformer::AnyValue::CreateFrom<float>(0.25f)}},
        &compileInfo);

    uint64_t expectTilingKey = 17UL;

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// ============================================================================
// 测试用例 3: BF16 数据类型
// ============================================================================
TEST_F(BSASelectBlockMaskTilingTest, tiling_bf16_case)
{
    optiling::BSASelectBlockMaskCompileInfo compileInfo;

    int64_t b = 1, n = 1, s = 256, s_kv = 256, d = 128;
    int64_t blockX = 128, blockY = 128;
    int64_t ceilQ = (s + blockX - 1) / blockX;
    int64_t ceilKv = (s_kv + blockY - 1) / blockY;

    int64_t blockShapeData[2] = {blockX, blockY};
    int64_t actualSeqData[1] = {s};
    int64_t actualSeqKvData[1] = {s_kv};

    gert::TilingContextPara tilingContextPara(
        "BSASelectBlockMask",
        {// 0: query (BF16)
         {{{b, n, s, d}, {b, n, s, d}}, ge::DT_BF16, ge::FORMAT_ND},
         // 1: key (BF16)
         {{{b, n, s_kv, d}, {b, n, s_kv, d}}, ge::DT_BF16, ge::FORMAT_ND},
         // 2: block_shape
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, (void *)blockShapeData},
         // 3: post_block_shape
         {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND},
         // 4: actual_seq_lengths
         {{{b}, {b}}, ge::DT_INT64, ge::FORMAT_ND, true, (void *)actualSeqData},
         // 5: actual_seq_lengths_kv
         {{{b}, {b}}, ge::DT_INT64, ge::FORMAT_ND, true, (void *)actualSeqKvData},
         // 6: actual_block_len_query
         {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND},
         // 7: actual_block_len_key
         {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND}},
        {// 0: block_sparse_mask_out
         {{{b, n, ceilQ, ceilKv}, {b, n, ceilQ, ceilKv}}, ge::DT_INT8, ge::FORMAT_ND}},
        {{"q_input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"kv_input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(n)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"sparsity", Ops::Transformer::AnyValue::CreateFrom<float>(0.5f)}},
        &compileInfo);

    uint64_t expectTilingKey = 17UL;

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// ============================================================================
// 测试用例 4: batch > 1 场景
// ============================================================================
TEST_F(BSASelectBlockMaskTilingTest, tiling_batch2_case)
{
    optiling::BSASelectBlockMaskCompileInfo compileInfo;

    int64_t b = 2, n = 2, s = 256, s_kv = 256, d = 128;
    int64_t blockX = 128, blockY = 128;
    int64_t ceilQ = (s + blockX - 1) / blockX;
    int64_t ceilKv = (s_kv + blockY - 1) / blockY;

    int64_t blockShapeData[2] = {blockX, blockY};
    int64_t actualSeqData[2] = {s, s};
    int64_t actualSeqKvData[2] = {s_kv, s_kv};

    gert::TilingContextPara tilingContextPara(
        "BSASelectBlockMask",
        {// 0: query
         {{{b, n, s, d}, {b, n, s, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1: key
         {{{b, n, s_kv, d}, {b, n, s_kv, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 2: block_shape
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, (void *)blockShapeData},
         // 3: post_block_shape
         {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND},
         // 4: actual_seq_lengths
         {{{b}, {b}}, ge::DT_INT64, ge::FORMAT_ND, true, (void *)actualSeqData},
         // 5: actual_seq_lengths_kv
         {{{b}, {b}}, ge::DT_INT64, ge::FORMAT_ND, true, (void *)actualSeqKvData},
         // 6: actual_block_len_query
         {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND},
         // 7: actual_block_len_key
         {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND}},
        {// 0: block_sparse_mask_out
         {{{b, n, ceilQ, ceilKv}, {b, n, ceilQ, ceilKv}}, ge::DT_INT8, ge::FORMAT_ND}},
        {{"q_input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"kv_input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(n)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"sparsity", Ops::Transformer::AnyValue::CreateFrom<float>(0.75f)}},
        &compileInfo);

    uint64_t expectTilingKey = 17UL;

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// ============================================================================
// 测试用例 5: TND Layout 一致场景（Q/KV 均为 TND）
// tilingKey 编码: LAYOUT_Q 占低 4 位, LAYOUT_KV 占高 4 位（TND=0, BNSD=1）
// ============================================================================
TEST_F(BSASelectBlockMaskTilingTest, tiling_tnd_case)
{
    optiling::BSASelectBlockMaskCompileInfo compileInfo;

    // TND: query [totalQ, N, D], key [totalKv, N, D]；batch/maxSeqLen 由 lens 常量推导
    int64_t totalQ = 1624, totalKv = 3584, n = 8, d = 128;
    int64_t blockX = 256, blockY = 256;
    int64_t maxQ = 1024, maxKv = 2048; // max(600, 1024) / max(2048, 1536)
    int64_t ceilQ = (maxQ + blockX - 1) / blockX;
    int64_t ceilKv = (maxKv + blockY - 1) / blockY;

    int64_t blockShapeData[2] = {blockX, blockY};
    int64_t actualSeqData[2] = {600, 1024};
    int64_t actualSeqKvData[2] = {2048, 1536};

    gert::TilingContextPara tilingContextPara(
        "BSASelectBlockMask",
        {// --- Input Info ---
         // 0: query [totalQ, N, D]
         {{{totalQ, n, d}, {totalQ, n, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1: key [totalKv, N, D]
         {{{totalKv, n, d}, {totalKv, n, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 2: block_shape (OPTIONAL, has value)
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, (void *)blockShapeData},
         // 3: post_block_shape (OPTIONAL, not provided)
         {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND},
         // 4: actual_seq_lengths (OPTIONAL, has value)
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, (void *)actualSeqData},
         // 5: actual_seq_lengths_kv (OPTIONAL, has value)
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, (void *)actualSeqKvData},
         // 6: actual_block_len_query (OPTIONAL, not provided)
         {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND},
         // 7: actual_block_len_key (OPTIONAL, not provided)
         {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND}},
        {// --- Output Info ---
         // 0: block_sparse_mask_out [B, N, ceilQ, ceilKv]
         {{{2, n, ceilQ, ceilKv}, {2, n, ceilQ, ceilKv}}, ge::DT_INT8, ge::FORMAT_ND}},
        {// --- Attr Info (对齐 OpDef 里的 5 个属性顺序) ---
         {"q_input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kv_input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(n)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"sparsity", Ops::Transformer::AnyValue::CreateFrom<float>(0.5f)}},
        &compileInfo);

    // Q/KV 均为 TND: LAYOUT_Q=0(低 4 位), LAYOUT_KV=0(高 4 位) -> tilingKey=0
    uint64_t expectTilingKey = 0UL;

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}
