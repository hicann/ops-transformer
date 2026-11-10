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

namespace {
gert::StorageShape MakeStorageShape(const gert::Shape& shape)
{
    gert::StorageShape result;
    result.MutableOriginShape() = shape;
    result.MutableStorageShape() = shape;
    return result;
}
void CheckArch22Tiling(int64_t heads, const std::string& soc, bool statistics, bool tnd, ge::graphStatus expected)
{
    struct CompileInfo {
    } compileInfo;
    int32_t actualQ[] = {2};
    int32_t actualKV[] = {32};
    auto qShape = tnd ? gert::Shape({2, heads, 576}) : gert::Shape({1, 2, heads, 576});
    auto sparseShape = tnd ? gert::Shape({2, 1, 32}) : gert::Shape({1, 2, 1, 32});
    auto outShape = tnd ? gert::Shape({2, heads, 512}) : gert::Shape({1, 2, heads, 512});
    auto statShape = statistics ? (tnd ? gert::Shape({1, 2, heads}) : gert::Shape({1, 1, 2, heads})) : gert::Shape({0});
    gert::TilingContextPara context(
        "KvQuantSparseFlashAttentionV2",
        {{MakeStorageShape(qShape), ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 16, 1, 656}, {2, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2, 16, 1, 656}, {2, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
         {MakeStorageShape(sparseShape), ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actualQ},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actualKV}},
        {{MakeStorageShape(outShape), ge::DT_FLOAT16, ge::FORMAT_ND},
         {MakeStorageShape(statShape), ge::DT_FLOAT, ge::FORMAT_ND},
         {MakeStorageShape(statShape), ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.5f)},
         {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"value_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>(tnd ? "TND" : "BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"rope_head_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(statistics)}},
        &compileInfo, soc, 64, 262144, 16384);
    uint64_t expectedKey = 1154U + (tnd ? 4U : 0U) + (heads > 64 ? 16384U : 0U);
    ExecuteTestCase(context, expected, expectedKey);
}
} // namespace

TEST(KvQuantSparseFlashAttentionV2Arch22, SupportedHeadCounts)
{
    for (const std::string soc : {"Ascend910B", "Ascend910_93"}) {
        for (int64_t heads = 1; heads <= 128; ++heads) {
            SCOPED_TRACE(soc + ":" + std::to_string(heads));
            CheckArch22Tiling(heads, soc, true, false, ge::GRAPH_SUCCESS);
        }
        CheckArch22Tiling(129, soc, true, false, ge::GRAPH_FAILED);
    }
}

TEST(KvQuantSparseFlashAttentionV2Arch22, A5HeadCounts)
{
    CheckArch22Tiling(95, "Ascend950", true, false, ge::GRAPH_SUCCESS);
    CheckArch22Tiling(96, "Ascend950", true, false, ge::GRAPH_SUCCESS);
}

TEST(KvQuantSparseFlashAttentionV2Arch22, TndAndDisabledStatistics)
{
    CheckArch22Tiling(96, "Ascend910B", true, true, ge::GRAPH_SUCCESS);
    CheckArch22Tiling(96, "Ascend910B", false, false, ge::GRAPH_SUCCESS);
}

TEST(KvQuantSparseFlashAttentionV2Arch22, InvalidHeadCounts)
{
    CheckArch22Tiling(0, "Ascend910B", true, false, ge::GRAPH_FAILED);
    CheckArch22Tiling(129, "Ascend910B", true, false, ge::GRAPH_FAILED);
}
