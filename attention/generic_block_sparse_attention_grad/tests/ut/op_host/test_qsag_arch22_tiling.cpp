/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_qsag_arch22_tiling.cpp
 * \brief QSAG on DAV_2201: the tiling template is selected and emits keys 200 (BF16) / 201 (FP16).
 */

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "tiling_case_executor.h"
#include "tiling_context_faker.h"
#include "attention/generic_block_sparse_attention_grad/op_host/generic_block_sparse_attention_grad_tiling.h"

namespace {
constexpr int64_t kBatch = 1;
constexpr int64_t kNq = 8;
constexpr int64_t kNk = 2;
constexpr int64_t kHeadDim = 64;
constexpr int64_t kSq = 16;
constexpr int64_t kSk = 32;
constexpr int64_t kTopK = 4;
constexpr int64_t kQcap = kSq;
constexpr float kScale = 0.125f;

// QSAG tiling keys: BF16 -> 200, FP16 -> 201.
constexpr uint64_t kTilingKeyBf16 = 200UL;
constexpr uint64_t kTilingKeyFp16 = 201UL;

using TensorDescription = gert::TilingContextPara::TensorDescription;
using OpAttr = gert::TilingContextPara::OpAttr;

TensorDescription Absent()
{
    return TensorDescription({}, ge::DT_UNDEFINED, ge::FORMAT_ND);
}

gert::StorageShape MakeShape(const std::vector<int64_t>& shape)
{
    gert::StorageShape storage;
    for (int64_t dim : shape) {
        storage.MutableOriginShape().AppendDim(dim);
        storage.MutableStorageShape().AppendDim(dim);
    }
    return storage;
}

std::vector<TensorDescription> MakeInputs(const std::string& layout, ge::DataType dtype, int64_t nq = kNq,
                                          int64_t nk = kNk, int64_t headDim = kHeadDim)
{
    std::vector<int64_t> query;
    std::vector<int64_t> key;
    std::vector<int64_t> lse;
    if (layout == "TND") {
        query = {kSq, nq, headDim};
        key = {kSk, nk, headDim};
        lse = {kSq, nq, 1};
    } else if (layout == "BNSD") {
        query = {kBatch, nq, kSq, headDim};
        key = {kBatch, nk, kSk, headDim};
        lse = {kBatch, nq, kSq, 1};
    } else {
        query = {kBatch, kSq, nq, headDim};
        key = {kBatch, kSk, nk, headDim};
        lse = {kBatch, nq, kSq, 1};
    }
    const std::vector<int64_t> indices = {kBatch, nk, kQcap, kTopK};
    const std::vector<int64_t> counts = {kBatch, nk, kQcap};
    std::vector<TensorDescription> inputs;
    inputs.emplace_back(MakeShape(query), dtype, ge::FORMAT_ND);          // query
    inputs.emplace_back(MakeShape(key), dtype, ge::FORMAT_ND);            // key
    inputs.emplace_back(MakeShape(key), dtype, ge::FORMAT_ND);            // value
    inputs.emplace_back(MakeShape(query), dtype, ge::FORMAT_ND);          // dout
    inputs.emplace_back(MakeShape(query), dtype, ge::FORMAT_ND);          // out
    inputs.emplace_back(MakeShape(lse), ge::DT_FLOAT, ge::FORMAT_ND);     // lse
    inputs.emplace_back(MakeShape(indices), ge::DT_INT32, ge::FORMAT_ND); // sparse indices
    inputs.emplace_back(MakeShape(counts), ge::DT_INT32, ge::FORMAT_ND);  // sparse counts
    inputs.emplace_back(Absent());                                        // metadata: QSAG requires null
    inputs.emplace_back(Absent());                                        // atten_mask: QSAG requires null
    inputs.emplace_back(Absent());                                        // cu_seqlens_q
    inputs.emplace_back(Absent());                                        // cu_seqlens_kv
    inputs.emplace_back(Absent());                                        // seqused_q
    inputs.emplace_back(Absent());                                        // seqused_kv
    if (layout == "TND") {
        const std::vector<int64_t> lengths = {kBatch + 1};
        inputs[10] = TensorDescription(MakeShape(lengths), ge::DT_INT64, ge::FORMAT_ND);
        inputs[11] = TensorDescription(MakeShape(lengths), ge::DT_INT64, ge::FORMAT_ND);
    }
    return inputs;
}

std::vector<TensorDescription> MakeOutputs(const std::string& layout, ge::DataType dtype, int64_t nq = kNq,
                                           int64_t nk = kNk, int64_t headDim = kHeadDim)
{
    std::vector<int64_t> qShape;
    std::vector<int64_t> kShape;
    if (layout == "TND") {
        qShape = {kSq, nq, headDim};
        kShape = {kSk, nk, headDim};
    } else if (layout == "BNSD") {
        qShape = {kBatch, nq, kSq, headDim};
        kShape = {kBatch, nk, kSk, headDim};
    } else {
        qShape = {kBatch, kSq, nq, headDim};
        kShape = {kBatch, kSk, nk, headDim};
    }
    return {
        TensorDescription(MakeShape(qShape), dtype, ge::FORMAT_ND),
        TensorDescription(MakeShape(kShape), dtype, ge::FORMAT_ND),
        TensorDescription(MakeShape(kShape), dtype, ge::FORMAT_ND),
    };
}

std::vector<OpAttr> MakeAttrs(const std::string& layout, const std::vector<int64_t>& blockShape = {1, 64})
{
    return {
        {"block_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(blockShape)},
        {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>(layout)},
        {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>(layout)},
        {"layout_sparse_pattern", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(kScale)},
        {"mask_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"softmax_precision", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"residual_block_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"is_consistent_topk", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
    };
}

void RunTilingCase(const std::string& layout, ge::DataType dtype, uint64_t expectedTilingKey, int64_t nq = kNq,
                   int64_t nk = kNk, const std::vector<int64_t>& blockShape = {1, 64}, int64_t headDim = kHeadDim)
{
    optiling::GenericBlockSparseAttentionGradCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttentionGrad",
                                              MakeInputs(layout, dtype, nq, nk, headDim),
                                              MakeOutputs(layout, dtype, nq, nk, headDim),
                                              MakeAttrs(layout, blockShape), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey);
}
} // namespace

class QsagArch22TilingTest : public testing::Test {};

TEST_F(QsagArch22TilingTest, tnd_fp16_maps_to_key_201)
{
    RunTilingCase("TND", ge::DT_FLOAT16, kTilingKeyFp16);
}

TEST_F(QsagArch22TilingTest, tnd_bf16_maps_to_key_200)
{
    RunTilingCase("TND", ge::DT_BF16, kTilingKeyBf16);
}

TEST_F(QsagArch22TilingTest, bnsd_fp16_maps_to_key_201)
{
    RunTilingCase("BNSD", ge::DT_FLOAT16, kTilingKeyFp16);
}

TEST_F(QsagArch22TilingTest, bsnd_bf16_maps_to_key_200)
{
    RunTilingCase("BSND", ge::DT_BF16, kTilingKeyBf16);
}

TEST_F(QsagArch22TilingTest, tnd_fp16_block_shape_equals_ktile)
{
    // G=32 keeps base Ktile=128; Bs=128 gives one sparse block per tile.
    RunTilingCase("TND", ge::DT_FLOAT16, kTilingKeyFp16, 32, 1, {1, 128});
}

TEST_F(QsagArch22TilingTest, tnd_fp16_block_y_8_and_head_dim_192)
{
    RunTilingCase("TND", ge::DT_FLOAT16, kTilingKeyFp16, 8, 1, {1, 8}, 192);
}

TEST_F(QsagArch22TilingTest, bnsd_bf16_block_y_8_and_head_dim_192)
{
    RunTilingCase("BNSD", ge::DT_BF16, kTilingKeyBf16, 8, 1, {1, 8}, 192);
}

TEST_F(QsagArch22TilingTest, accepts_consistent_topk_true)
{
    optiling::GenericBlockSparseAttentionGradCompileInfo compileInfo;
    auto attrs = MakeAttrs("TND");
    attrs.back() = {"is_consistent_topk", Ops::Transformer::AnyValue::CreateFrom<bool>(true)};
    gert::TilingContextPara context("GenericBlockSparseAttentionGrad", MakeInputs("TND", ge::DT_FLOAT16),
                                    MakeOutputs("TND", ge::DT_FLOAT16), attrs, &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, kTilingKeyFp16);
}

TEST_F(QsagArch22TilingTest, rejects_architecture_limits)
{
    optiling::GenericBlockSparseAttentionGradCompileInfo compileInfo;
    for (const auto& spec :
         std::vector<std::vector<int64_t>>{{514, 2, 64, 32}, {8, 2, 272, 64}, {8, 2, 64, 129}, {129, 1, 64, 65}}) {
        gert::TilingContextPara context("GenericBlockSparseAttentionGrad",
                                        MakeInputs("TND", ge::DT_FLOAT16, spec[0], spec[1], spec[2]),
                                        MakeOutputs("TND", ge::DT_FLOAT16, spec[0], spec[1], spec[2]),
                                        MakeAttrs("TND", {1, spec[3]}), &compileInfo, "Ascend910B", 40, 196608);
        ExecuteTestCase(context, ge::GRAPH_FAILED);
    }
}

TEST_F(QsagArch22TilingTest, rejects_tnd_query_capacity_too_small)
{
    optiling::GenericBlockSparseAttentionGradCompileInfo compileInfo;
    auto inputs = MakeInputs("TND", ge::DT_FLOAT16);
    inputs[6] = TensorDescription(MakeShape({kBatch, kNk, kSq / 2, kTopK}), ge::DT_INT32, ge::FORMAT_ND);
    inputs[7] = TensorDescription(MakeShape({kBatch, kNk, kSq / 2}), ge::DT_INT32, ge::FORMAT_ND);
    gert::TilingContextPara context("GenericBlockSparseAttentionGrad", inputs, MakeOutputs("TND", ge::DT_FLOAT16),
                                    MakeAttrs("TND"), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(QsagArch22TilingTest, legacy_layout_one_falls_back_to_gbsag)
{
    optiling::GenericBlockSparseAttentionGradCompileInfo compileInfo;
    std::vector<int32_t> metadata(198, 0);
    for (const auto& layout : {std::string("BNSD"), std::string("BSND")}) {
        for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16}) {
            auto inputs = MakeInputs(layout, dtype, kNq, kNk, 128);
            inputs[6] = TensorDescription(MakeShape({kBatch, kNk, 1, kSq}), ge::DT_INT32, ge::FORMAT_ND);
            inputs[7] = TensorDescription(MakeShape({kBatch, kNk, 1}), ge::DT_INT32, ge::FORMAT_ND);
            inputs[8] = TensorDescription(MakeShape({198}), ge::DT_INT32, ge::FORMAT_ND, true, metadata.data());
            auto attrs = MakeAttrs(layout, {1, 128});
            attrs[3] = {"layout_sparse_pattern", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)};
            attrs[9] = {"residual_block_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)};
            gert::TilingContextPara context("GenericBlockSparseAttentionGrad", inputs,
                                            MakeOutputs(layout, dtype, kNq, kNk, 128), attrs, &compileInfo,
                                            "Ascend910B", 40, 196608);
            const uint64_t expectedKey = (dtype == ge::DT_BF16 ? 110 : 100) + (layout == "BNSD" ? 1 : 2);
            ExecuteTestCase(context, ge::GRAPH_SUCCESS, expectedKey);
        }
    }
}
