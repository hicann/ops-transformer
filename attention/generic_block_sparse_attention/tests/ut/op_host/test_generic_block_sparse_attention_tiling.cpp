/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "attention/generic_block_sparse_attention/op_host/generic_block_sparse_attention_tiling.h"
// UT-only cross-check against device Union UB packing.
#include "attention/generic_block_sparse_attention/op_kernel/arch35/gbsa_layout_utils.hpp"

using namespace std;
using namespace ge;
using namespace optiling;

class generic_block_sparse_attention_tiling_ut : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "generic_block_sparse_attention_tiling_ut SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "generic_block_sparse_attention_tiling_ut TearDown" << endl;
    }
};

namespace {
constexpr int64_t kBatch = 1;
constexpr int64_t kS1 = 4;
constexpr int64_t kS2 = 256;
constexpr int64_t kN1 = 4;
constexpr int64_t kN2 = 1;
constexpr int64_t kFp8HighPrecisionHeads = 16 * kN2;
constexpr int64_t kD = 128;
constexpr int64_t kTopK = 2;
constexpr int64_t kBlockSize = 128;
constexpr int64_t kBlockShapeX = 1;
constexpr int64_t kUnionBlockShapeX = 8; // unionSize == blockShapeX; only 8 supported
constexpr uint64_t kUnionLseKeyOffset = 100000000ULL;
constexpr int64_t kT = kBatch * kS1;
constexpr int64_t kMaxBlocks = (kS2 + kBlockSize - 1) / kBlockSize;
constexpr int64_t kTotalQBlocks = kT;
constexpr float kScale = 1.0f / sqrt(static_cast<float>(kD));

// Expected host tiling keys (BSA-style bitfields; must match GenerateTilingKey).
constexpr uint64_t kFp16Tiling910B = 9200000071001002ULL;
constexpr uint64_t kFp16TndTndTiling910B = 9200000030001002ULL;
constexpr uint64_t kBf16Tiling910B = 9200000071023222ULL;
constexpr uint64_t kFp16HalfSmTiling910B = 9200000071101002ULL;
constexpr uint64_t kFp16Tiling950 = 9250000071101002ULL;
constexpr uint64_t kBf16Tiling950 = 9250000071123222ULL;
constexpr uint64_t kFp8Fp16Tiling950 = 9250000071101012ULL;
constexpr uint64_t kFp8Bf16Tiling950 = 9250000071101022ULL;
constexpr uint64_t kFp16BnbdTiling950 = 9250000081101002ULL;
constexpr uint64_t kBf16BnbdTiling950 = 9250000081123222ULL;
constexpr uint64_t kFp16BnbdNoMaskTiling950 = 9250000081100002ULL;

std::vector<int64_t> kBlockShape = {kBlockShapeX, kBlockSize};
std::vector<int64_t> kUnionBlockShape = {kUnionBlockShapeX, kBlockSize};

std::vector<gert::TilingContextPara::TensorDescription> MakeInputs(ge::DataType qkvDtype, int64_t headDim = kD,
                                                                   int64_t numHeads = kN1, int64_t kvHeads = kN2)
{
    return {
        {{{kT, numHeads, headDim}, {kT, numHeads, headDim}}, qkvDtype, ge::FORMAT_ND},
        {{{kMaxBlocks, kBlockSize, kvHeads, headDim}, {kMaxBlocks, kBlockSize, kvHeads, headDim}},
         qkvDtype,
         ge::FORMAT_ND},
        {{{kMaxBlocks, kBlockSize, kvHeads, headDim}, {kMaxBlocks, kBlockSize, kvHeads, headDim}},
         qkvDtype,
         ge::FORMAT_ND},
        {{{kvHeads, kTotalQBlocks, kTopK}, {kvHeads, kTotalQBlocks, kTopK}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{kvHeads, kTotalQBlocks}, {kvHeads, kTotalQBlocks}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
        {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
        {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
        {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
        {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
        {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
        {{{kBatch + 1}, {kBatch + 1}}, ge::DT_INT64, ge::FORMAT_ND},
        {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
        {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
        {{{kBatch}, {kBatch}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{kBatch, kMaxBlocks}, {kBatch, kMaxBlocks}}, ge::DT_INT32, ge::FORMAT_ND},
    };
}

std::vector<gert::TilingContextPara::TensorDescription> MakeDenseInputs(ge::DataType qkvDtype, int64_t headDim = kD,
                                                                        int64_t topK = kTopK)
{
    auto inputs = MakeInputs(qkvDtype, headDim);
    inputs[1] = {{{kS2, kN2, headDim}, {kS2, kN2, headDim}}, qkvDtype, ge::FORMAT_ND};
    inputs[2] = inputs[1];
    inputs[3] = {{{kN2, kTotalQBlocks, topK}, {kN2, kTotalQBlocks, topK}}, ge::DT_INT32, ge::FORMAT_ND};
    inputs[12] = {{{kBatch + 1}, {kBatch + 1}}, ge::DT_INT64, ge::FORMAT_ND};
    inputs[15] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
    return inputs;
}

uint32_t ReadTilingU32(const TilingInfo& tilingInfo, size_t byteOffset)
{
    uint32_t value = 0;
    std::memcpy(&value, tilingInfo.tilingData.get() + byteOffset, sizeof(uint32_t));
    return value;
}

std::vector<gert::TilingContextPara::TensorDescription> MakeOutputs(ge::DataType outDtype, bool returnLse,
                                                                    int64_t headDim = kD, int64_t numHeads = kN1)
{
    if (returnLse) {
        return {
            {{{kT, numHeads, headDim}, {kT, numHeads, headDim}}, outDtype, ge::FORMAT_ND},
            {{{kT, numHeads, 1}, {kT, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        };
    }
    return {
        {{{kT, numHeads, headDim}, {kT, numHeads, headDim}}, outDtype, ge::FORMAT_ND},
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
    };
}

std::vector<gert::TilingContextPara::OpAttr> MakeAttrs(
    int64_t maskType, int64_t quantType, int64_t softmaxPrec, int64_t returnLse, float dstTypeMax = 0.0f,
    int64_t winLeft = -1, int64_t winRight = -1, int64_t layoutSparsePattern = 4, const std::string& layoutQ = "TND",
    const std::string& layoutKv = "PA_BBND", int64_t residualBlockMode = 0, bool isConsistentTopk = false,
    const std::vector<int64_t>& blockShape = kBlockShape, float scaleValue = kScale)
{
    return {
        {"block_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(blockShape)},
        {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>(layoutQ)},
        {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>(layoutKv)},
        {"layout_sparse_pattern", Ops::Transformer::AnyValue::CreateFrom<int64_t>(layoutSparsePattern)},
        {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
        {"mask_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(maskType)},
        {"quant_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantType)},
        {"dst_type_max", Ops::Transformer::AnyValue::CreateFrom<float>(dstTypeMax)},
        {"softmax_precision", Ops::Transformer::AnyValue::CreateFrom<int64_t>(softmaxPrec)},
        {"win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(winLeft)},
        {"win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(winRight)},
        {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<int64_t>(returnLse)},
        {"residual_block_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(residualBlockMode)},
        {"is_consistent_topk", Ops::Transformer::AnyValue::CreateFrom<bool>(isConsistentTopk)},
    };
}

gert::TilingContextPara::TensorDescription MakeTensorDescription(const gert::Shape& shape, ge::DataType dtype)
{
    gert::StorageShape storageShape;
    storageShape.MutableOriginShape() = shape;
    storageShape.MutableStorageShape() = shape;
    return {storageShape, dtype, ge::FORMAT_ND};
}

void CheckA8C4Tiling(const std::string& layoutQ, const std::string& layoutKv, int64_t qSeq, int64_t kvHeads,
                     ge::graphStatus expectedStatus, uint64_t expectedKey = 0)
{
    SCOPED_TRACE(testing::Message() << "Q layout: " << layoutQ << ", KV layout: " << layoutKv << ", Qs: " << qSeq
                                    << ", KV heads: " << kvHeads);
    const int64_t qHeads = kvHeads * kN1;
    const bool paged = layoutKv == "PA_BBND" || layoutKv == "PA_BNBD";
    const bool tnd = layoutQ == "TND";
    const bool bnsd = layoutQ == "BNSD";
    const auto qShape = tnd  ? gert::Shape({qSeq, qHeads, kD}) :
                        bnsd ? gert::Shape({kBatch, qHeads, qSeq, kD}) :
                               gert::Shape({kBatch, qSeq, qHeads, kD});
    gert::Shape kvShape;
    if (paged) {
        kvShape = layoutKv == "PA_BNBD" ? gert::Shape({kMaxBlocks, kvHeads, kBlockSize, kD}) :
                                          gert::Shape({kMaxBlocks, kBlockSize, kvHeads, kD});
    } else if (tnd) {
        kvShape = gert::Shape({kS2, kvHeads, kD});
    } else {
        kvShape = bnsd ? gert::Shape({kBatch, kvHeads, kS2, kD}) : gert::Shape({kBatch, kS2, kvHeads, kD});
    }
    auto scaleShape = kvShape;
    scaleShape.SetDim(scaleShape.GetDimNum() - 1U, kD / 16);
    auto inputs = MakeInputs(ge::DT_FLOAT8_E4M3FN, kD, qHeads, kvHeads);
    inputs[0] = MakeTensorDescription(qShape, ge::DT_FLOAT8_E4M3FN);
    inputs[1] = MakeTensorDescription(kvShape, ge::DT_FLOAT4_E2M1);
    inputs[2] = inputs[1];
    const int64_t qBlocks = kBatch * qSeq;
    inputs[3] = {{{kvHeads, qBlocks, kTopK}, {kvHeads, qBlocks, kTopK}}, ge::DT_INT32, ge::FORMAT_ND};
    inputs[4] = {{{kvHeads, qBlocks}, {kvHeads, qBlocks}}, ge::DT_INT32, ge::FORMAT_ND};
    inputs[8] = MakeTensorDescription(scaleShape, ge::DT_BF16);
    inputs[9] = inputs[8];
    if (!tnd) {
        inputs[11] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
        inputs[13] = {{{kBatch}, {kBatch}}, ge::DT_INT32, ge::FORMAT_ND};
    } else if (!paged) {
        inputs[12] = inputs[11];
    }
    if (!paged) {
        inputs[15] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
    }
    auto outputs = MakeOutputs(ge::DT_BF16, false, kD, qHeads);
    outputs[0] = MakeTensorDescription(qShape, ge::DT_BF16);

    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara ctx("GenericBlockSparseAttention", inputs, outputs,
                                MakeAttrs(0, 8, 0, 0, 0.0f, -1, -1, 4, layoutQ, layoutKv), &compileInfo, "Ascend950",
                                32, 262144);
    if (expectedStatus == ge::GRAPH_SUCCESS) {
        ExecuteTestCase(ctx, expectedStatus, expectedKey);
    } else {
        ExecuteTestCase(ctx, expectedStatus);
    }
}
} // namespace

TEST_F(generic_block_sparse_attention_tiling_ut, a8c4_batched_q_sequence_boundary)
{
    CheckA8C4Tiling("BSND", "BSND", 16, 1, ge::GRAPH_SUCCESS, 9251000020000024ULL);
    CheckA8C4Tiling("BSND", "BSND", 17, 1, ge::GRAPH_FAILED);
    CheckA8C4Tiling("BNSD", "BNSD", 16, 1, ge::GRAPH_SUCCESS, 9251000010000023ULL);
    CheckA8C4Tiling("BNSD", "BNSD", 17, 1, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, a8c4_interleaved_kv_head_boundary)
{
    struct LayoutCase {
        const char* q;
        const char* kv;
        uint64_t key;
    };
    for (const auto& layout :
         {LayoutCase{"TND", "TND", 9251000030000022ULL}, LayoutCase{"BSND", "BSND", 9251000020000024ULL},
          LayoutCase{"TND", "PA_BBND", 9251000071000022ULL}}) {
        CheckA8C4Tiling(layout.q, layout.kv, 16, 1, ge::GRAPH_SUCCESS, layout.key);
        CheckA8C4Tiling(layout.q, layout.kv, 16, 5, ge::GRAPH_SUCCESS, layout.key);
        CheckA8C4Tiling(layout.q, layout.kv, 16, 6, ge::GRAPH_FAILED);
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, a8c4_contiguous_kv_allows_six_heads)
{
    CheckA8C4Tiling("BNSD", "BNSD", 16, 1, ge::GRAPH_SUCCESS, 9251000010000023ULL);
    CheckA8C4Tiling("BNSD", "BNSD", 16, 6, ge::GRAPH_SUCCESS, 9251000010000023ULL);
    CheckA8C4Tiling("TND", "PA_BNBD", 16, 1, ge::GRAPH_SUCCESS, 9251000081000022ULL);
    CheckA8C4Tiling("TND", "PA_BNBD", 16, 6, ge::GRAPH_SUCCESS, 9251000081000022ULL);
}

TEST_F(generic_block_sparse_attention_tiling_ut, arch22_small_m_kv_tile_after_quantization_removal)
{
    // Serialized baseTileInfo.kvBaseTile; guard the original small-M boundary.
    constexpr size_t kKvBaseTileOffset = 180U;
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16}) {
        for (int64_t dim : {128, 192, 256}) {
            for (int64_t heads : {1, 16, 17}) {
                GenericBlockSparseAttentionCompileInfo compileInfo;
                gert::TilingContextPara context("GenericBlockSparseAttention", MakeInputs(dtype, dim, heads, 1),
                                                MakeOutputs(dtype, false, dim, heads), MakeAttrs(1, 0, 0, 0),
                                                &compileInfo, "Ascend910B", 40, 196608);
                TilingInfo info;
                ASSERT_TRUE(ExecuteTiling(context, info));
                EXPECT_EQ(ReadTilingU32(info, kKvBaseTileOffset), heads <= 16 ? 512U : 256U)
                    << "dtype=" << dtype << ", D=" << dim << ", G=" << heads;
            }
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, generalized_y_arch35)
{
    for (int64_t y : {1, 3, 16, 17, 63, 64, 65, 95, 127, 128}) {
        for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16}) {
            for (bool paged : {false, true}) {
                auto inputs = MakeInputs(dtype);
                if (!paged) {
                    inputs[1] = {{{kS2, kN2, kD}, {kS2, kN2, kD}}, dtype, ge::FORMAT_ND};
                    inputs[2] = inputs[1];
                    inputs[12] = {{{kBatch + 1}, {kBatch + 1}}, ge::DT_INT64, ge::FORMAT_ND};
                    inputs[15] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
                }
                GenericBlockSparseAttentionCompileInfo compileInfo;
                const uint64_t key = paged ? (dtype == ge::DT_FLOAT16 ? kFp16Tiling950 : kBf16Tiling950) :
                                             (dtype == ge::DT_FLOAT16 ? 9250000030101002ULL : 9250000030123222ULL);
                gert::TilingContextPara ctx(
                    "GenericBlockSparseAttention", inputs, MakeOutputs(dtype, false),
                    MakeAttrs(1, 0, 1, 0, 0.0f, -1, -1, 4, "TND", paged ? "PA_BBND" : "TND", 0, false, {1, y}),
                    &compileInfo, "Ascend950", 32, 262144);
                ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, key);
            }
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, generalized_y_rejects_out_of_range)
{
    for (int64_t y : {int64_t(0), int64_t(-1), int64_t(129), int64_t(256), int64_t(257), int64_t(4294967297LL)}) {
        GenericBlockSparseAttentionCompileInfo compileInfo;
        gert::TilingContextPara ctx("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16),
                                    MakeOutputs(ge::DT_FLOAT16, false),
                                    MakeAttrs(1, 0, 1, 0, 0.0f, -1, -1, 4, "TND", "PA_BBND", 0, false, {1, y}),
                                    &compileInfo, "Ascend950", 32, 262144);
        ExecuteTestCase(ctx, ge::GRAPH_FAILED);
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, tnd_paged_fp16_910b)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16),
                                              MakeOutputs(ge::DT_FLOAT16, false), MakeAttrs(1, 0, 0, 0), &compileInfo,
                                              "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kFp16Tiling910B);
}

namespace {
auto MakeInt8Inputs(ge::DataType scaleDtype = ge::DT_FLOAT, int64_t dim = 256)
{
    auto inputs = MakeInputs(ge::DT_INT8, dim);
    inputs[7] = {{{kT, kN1}, {kT, kN1}}, scaleDtype, ge::FORMAT_ND};
    inputs[8] = {{{kMaxBlocks, kN2, kBlockSize}, {kMaxBlocks, kN2, kBlockSize}}, scaleDtype, ge::FORMAT_ND};
    inputs[9] = {{{kN2, dim}, {kN2, dim}}, scaleDtype, ge::FORMAT_ND};
    inputs[10] = {{{1}, {1}}, scaleDtype, ge::FORMAT_ND};
    return inputs;
}
} // namespace

TEST_F(generic_block_sparse_attention_tiling_ut, int8_external_scales_910b)
{
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16}) {
        GenericBlockSparseAttentionCompileInfo compileInfo;
        gert::TilingContextPara context("GenericBlockSparseAttention", MakeInt8Inputs(), MakeOutputs(dtype, false, 256),
                                        MakeAttrs(1, 7, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
        ExecuteTestCase(context, ge::GRAPH_SUCCESS,
                        dtype == ge::DT_FLOAT16 ? 9200000071001032ULL : 9200000071001042ULL);
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, int8_tnd_seqused_q)
{
    auto inputs = MakeInt8Inputs();
    inputs[13] = {{{kBatch}, {kBatch}}, ge::DT_INT32, ge::FORMAT_ND};
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara context("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false, 256),
                                    MakeAttrs(1, 7, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, 9200000071001032ULL);
}

TEST_F(generic_block_sparse_attention_tiling_ut, int8_rejects_token_head_scale_layout)
{
    auto inputs = MakeInt8Inputs();
    inputs[8] = {{{kMaxBlocks, kBlockSize, kN2}, {kMaxBlocks, kBlockSize, kN2}}, ge::DT_FLOAT, ge::FORMAT_ND};
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara context("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false, 256),
                                    MakeAttrs(1, 7, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, int8_dense_tnd_head_token_scales)
{
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16}) {
        for (bool usedKv : {false, true}) {
            auto inputs = MakeDenseInputs(ge::DT_INT8, 256);
            auto scales = MakeInt8Inputs();
            for (uint32_t i : {7U, 9U, 10U}) {
                inputs[i] = scales[i];
            }
            inputs[8] = {{{kN2, kS2}, {kN2, kS2}}, ge::DT_FLOAT, ge::FORMAT_ND};
            if (!usedKv) {
                inputs[14] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
            }
            GenericBlockSparseAttentionCompileInfo compileInfo;
            gert::TilingContextPara context("GenericBlockSparseAttention", inputs, MakeOutputs(dtype, false, 256),
                                            MakeAttrs(1, 7, 0, 0, 0.0f, -1, -1, 4, "TND", "TND"), &compileInfo,
                                            "Ascend910B", 40, 196608);
            ExecuteTestCase(context, ge::GRAPH_SUCCESS,
                            dtype == ge::DT_FLOAT16 ? 9200000071001032ULL : 9200000071001042ULL);
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, int8_dense_rejects_wrong_scale_or_missing_cu)
{
    for (uint32_t option = 0; option < 3; ++option) {
        auto inputs = MakeDenseInputs(ge::DT_INT8, 256);
        auto scales = MakeInt8Inputs();
        for (uint32_t i : {7U, 9U, 10U}) {
            inputs[i] = scales[i];
        }
        inputs[8] = {{{kN2, kS2}, {kN2, kS2}}, ge::DT_FLOAT, ge::FORMAT_ND};
        if (option == 0) {
            inputs[8] = {{{kS2, kN2}, {kS2, kN2}}, ge::DT_FLOAT, ge::FORMAT_ND};
        } else if (option == 1) {
            inputs[8] = scales[8];
        } else {
            inputs[12] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
        }
        GenericBlockSparseAttentionCompileInfo compileInfo;
        gert::TilingContextPara context("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false, 256),
                                        MakeAttrs(1, 7, 0, 0, 0.0f, -1, -1, 4, "TND", "TND"), &compileInfo,
                                        "Ascend910B", 40, 196608);
        ExecuteTestCase(context, ge::GRAPH_FAILED);
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, int8_power_of_two_block_y)
{
    for (int64_t residual : {0, 1}) {
        for (int64_t y : {4, 8, 16, 32, 64, 128}) {
            for (bool paged : {false, true}) {
                auto inputs = paged ? MakeInt8Inputs() : MakeDenseInputs(ge::DT_INT8, 256);
                if (!paged) {
                    auto scales = MakeInt8Inputs();
                    for (uint32_t i : {7U, 9U, 10U}) {
                        inputs[i] = scales[i];
                    }
                    inputs[8] = {{{kN2, kS2}, {kN2, kS2}}, ge::DT_FLOAT, ge::FORMAT_ND};
                }
                GenericBlockSparseAttentionCompileInfo compileInfo;
                gert::TilingContextPara context(
                    "GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false, 256),
                    MakeAttrs(1, 7, 0, 0, 0.0f, -1, -1, 4, "TND", paged ? "PA_BBND" : "TND", residual, false, {1, y}),
                    &compileInfo, "Ascend910B", 40, 196608);
                ExecuteTestCase(context, ge::GRAPH_SUCCESS, 9200000071001032ULL);
            }
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, int8_lse_power_of_two_block_y)
{
    for (int64_t residual : {0, 1}) {
        for (int64_t y : {4, 8, 16, 32, 64, 128}) {
            for (bool paged : {false, true}) {
                auto inputs = paged ? MakeInt8Inputs() : MakeDenseInputs(ge::DT_INT8, 256);
                if (!paged) {
                    auto scales = MakeInt8Inputs();
                    for (uint32_t i : {7U, 9U, 10U}) {
                        inputs[i] = scales[i];
                    }
                    inputs[8] = {{{kN2, kS2}, {kN2, kS2}}, ge::DT_FLOAT, ge::FORMAT_ND};
                }
                GenericBlockSparseAttentionCompileInfo compileInfo;
                gert::TilingContextPara context(
                    "GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, true, 256),
                    MakeAttrs(1, 7, 0, 1, 0.0f, -1, -1, 4, "TND", paged ? "PA_BBND" : "TND", residual, false, {1, y}),
                    &compileInfo, "Ascend910B", 40, 196608);
                ExecuteTestCase(context, ge::GRAPH_SUCCESS, 9200000171001032ULL);
            }
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, int8_generalized_head_dimension)
{
    for (int64_t dim : {128, 256}) {
        for (bool paged : {false, true}) {
            for (int64_t residual : {0, 1}) {
                auto scales = MakeInt8Inputs(ge::DT_FLOAT, dim);
                auto inputs = paged ? scales : MakeDenseInputs(ge::DT_INT8, dim);
                if (!paged) {
                    for (uint32_t i : {7U, 9U, 10U}) {
                        inputs[i] = scales[i];
                    }
                    inputs[8] = {{{kN2, kS2}, {kN2, kS2}}, ge::DT_FLOAT, ge::FORMAT_ND};
                }
                GenericBlockSparseAttentionCompileInfo compileInfo;
                gert::TilingContextPara context(
                    "GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false, dim),
                    MakeAttrs(1, 7, 0, 0, 0.0f, -1, -1, 4, "TND", paged ? "PA_BBND" : "TND", residual, false, {1, 4}),
                    &compileInfo, "Ascend910B", 40, 196608);
                ExecuteTestCase(context, ge::GRAPH_SUCCESS, 9200000071001032ULL);
            }
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, int8_requires_all_four_scales)
{
    for (uint32_t input = 7; input <= 10; ++input) {
        auto inputs = MakeInt8Inputs();
        inputs[input] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
        GenericBlockSparseAttentionCompileInfo compileInfo;
        gert::TilingContextPara context("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false, 256),
                                        MakeAttrs(1, 7, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
        ExecuteTestCase(context, ge::GRAPH_FAILED);
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, int8_no_mask_910b)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara context("GenericBlockSparseAttention", MakeInt8Inputs(),
                                    MakeOutputs(ge::DT_FLOAT16, false, 256), MakeAttrs(0, 7, 0, 0), &compileInfo,
                                    "Ascend910B", 40, 196608);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, 9200000071001032ULL);
}

TEST_F(generic_block_sparse_attention_tiling_ut, int8_generalized_group_size)
{
    for (int64_t group : {1, 12, 16, 129}) {
        auto inputs = MakeInt8Inputs();
        inputs[0] = MakeInputs(ge::DT_INT8, 256, group)[0];
        inputs[7] = {{{kT, group}, {kT, group}}, ge::DT_FLOAT, ge::FORMAT_ND};
        GenericBlockSparseAttentionCompileInfo compileInfo;
        gert::TilingContextPara context("GenericBlockSparseAttention", inputs,
                                        MakeOutputs(ge::DT_FLOAT16, false, 256, group), MakeAttrs(1, 7, 0, 0),
                                        &compileInfo, "Ascend910B", 40, 196608);
        ExecuteTestCase(context, group <= 16 ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED,
                        group <= 16 ? 9200000071001032ULL : 0ULL);
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, int8_rejects_unsupported_shapes)
{
    for (int64_t dim : {257, 512}) {
        auto inputs = MakeInt8Inputs();
        auto qkv = MakeInputs(ge::DT_INT8, dim);
        for (uint32_t i = 0; i < 3; ++i) {
            inputs[i] = qkv[i];
        }
        inputs[9] = {{{kN2, dim}, {kN2, dim}}, ge::DT_FLOAT, ge::FORMAT_ND};
        GenericBlockSparseAttentionCompileInfo compileInfo;
        gert::TilingContextPara context("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false, dim),
                                        MakeAttrs(1, 7, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
        ExecuteTestCase(context, ge::GRAPH_FAILED);
    }
    for (int64_t y : {0, 3, 7, 127, 256}) {
        GenericBlockSparseAttentionCompileInfo compileInfo;
        gert::TilingContextPara context("GenericBlockSparseAttention", MakeInt8Inputs(),
                                        MakeOutputs(ge::DT_FLOAT16, false, 256),
                                        MakeAttrs(1, 7, 0, 0, 0.0f, -1, -1, 4, "TND", "PA_BBND", 0, false, {1, y}),
                                        &compileInfo, "Ascend910B", 40, 196608);
        ExecuteTestCase(context, ge::GRAPH_FAILED);
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, int8_rejects_unimplemented_options)
{
    for (uint32_t option : {0U, 2U}) {
        auto inputs = MakeInt8Inputs();
        GenericBlockSparseAttentionCompileInfo compileInfo;
        gert::TilingContextPara context(
            "GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false, 256),
            MakeAttrs(1, option == 0 ? 0 : 7, 0, 0, 0.0f, -1, -1, 4, "TND", "PA_BBND", option == 2 ? 2 : 0),
            &compileInfo, "Ascend910B", 40, 196608);
        ExecuteTestCase(context, ge::GRAPH_FAILED);
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, int8_rejects_incorrect_scale_dtype_and_shape)
{
    for (uint32_t input = 7; input <= 10; ++input) {
        for (bool wrongDtype : {false, true}) {
            auto inputs = MakeInt8Inputs();
            if (wrongDtype) {
                inputs[input] = MakeInt8Inputs(ge::DT_FLOAT16)[input];
            } else {
                inputs[input] = {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND};
            }
            GenericBlockSparseAttentionCompileInfo compileInfo;
            gert::TilingContextPara context("GenericBlockSparseAttention", inputs,
                                            MakeOutputs(ge::DT_FLOAT16, false, 256), MakeAttrs(1, 7, 0, 0),
                                            &compileInfo, "Ascend910B", 40, 196608);
            ExecuteTestCase(context, ge::GRAPH_FAILED);
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, int8_rejects_unsupported_precision_and_soc)
{
    for (bool arch35 : {false, true}) {
        GenericBlockSparseAttentionCompileInfo compileInfo;
        gert::TilingContextPara context("GenericBlockSparseAttention", MakeInt8Inputs(),
                                        MakeOutputs(ge::DT_FLOAT16, false, 256), MakeAttrs(1, 7, arch35 ? 0 : 1, 0),
                                        &compileInfo, arch35 ? "Ascend950" : "Ascend910B", 40, 196608);
        ExecuteTestCase(context, ge::GRAPH_FAILED);
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, tnd_paged_fp16_halfsm_910b)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16),
                                              MakeOutputs(ge::DT_FLOAT16, false), MakeAttrs(1, 0, 1, 0), &compileInfo,
                                              "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kFp16HalfSmTiling910B);
}

TEST_F(generic_block_sparse_attention_tiling_ut, tnd_paged_bf16_910b)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_BF16),
                                              MakeOutputs(ge::DT_BF16, false), MakeAttrs(1, 0, 0, 0), &compileInfo,
                                              "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kBf16Tiling910B);
}

TEST_F(generic_block_sparse_attention_tiling_ut, return_softmax_lse_sets_tiling_key)
{
    constexpr uint64_t kFp16Tiling910BLse = kFp16Tiling910B + 100000000ULL;
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16),
                                              MakeOutputs(ge::DT_FLOAT16, true), MakeAttrs(1, 0, 0, 1), &compileInfo,
                                              "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kFp16Tiling910BLse);
}

TEST_F(generic_block_sparse_attention_tiling_ut, tnd_paged_fp16_950)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16),
                                              MakeOutputs(ge::DT_FLOAT16, false), MakeAttrs(1, 0, 1, 0), &compileInfo,
                                              "Ascend950", 56, 262144);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kFp16Tiling950);
}

TEST_F(generic_block_sparse_attention_tiling_ut, nonquant_950_rejects_head_dim_over_256)
{
    for (int64_t headDim : {257, 384, 511, 512}) {
        for (int64_t precision : {0, 1}) {
            GenericBlockSparseAttentionCompileInfo compileInfo;
            gert::TilingContextPara tilingContextPara(
                "GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16, headDim),
                MakeOutputs(ge::DT_FLOAT16, false, headDim), MakeAttrs(1, 0, precision, 0), &compileInfo, "Ascend950",
                56, 262144);
            ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, tnd_paged_bf16_950)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_BF16),
                                              MakeOutputs(ge::DT_BF16, false), MakeAttrs(1, 0, 1, 0), &compileInfo,
                                              "Ascend950", 56, 262144);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kBf16Tiling950);
}

TEST_F(generic_block_sparse_attention_tiling_ut, tnd_paged_fp8_out_fp16_950)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT8_E4M3FN),
                                              MakeOutputs(ge::DT_FLOAT16, false), MakeAttrs(1, 5, 1, 0), &compileInfo,
                                              "Ascend950", 56, 262144);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kFp8Fp16Tiling950);
}

TEST_F(generic_block_sparse_attention_tiling_ut, tnd_paged_fp8_out_bf16_950)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT8_E4M3FN),
                                              MakeOutputs(ge::DT_BF16, false), MakeAttrs(1, 5, 1, 0), &compileInfo,
                                              "Ascend950", 56, 262144);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kFp8Bf16Tiling950);
}

TEST_F(generic_block_sparse_attention_tiling_ut, fp8_gather_reserves_descriptor_workspace)
{
    for (int64_t numHeads : {16, 128}) {
        for (auto outDtype : {ge::DT_FLOAT16, ge::DT_BF16}) {
            GenericBlockSparseAttentionCompileInfo compileInfo;
            gert::TilingContextPara nonQuantCtx("GenericBlockSparseAttention", MakeInputs(outDtype, kD, numHeads),
                                                MakeOutputs(outDtype, false, kD, numHeads), MakeAttrs(1, 0, 1, 0),
                                                &compileInfo, "Ascend950", 32, 262144);
            gert::TilingContextPara quantCtx("GenericBlockSparseAttention",
                                             MakeInputs(ge::DT_FLOAT8_E4M3FN, kD, numHeads),
                                             MakeOutputs(outDtype, false, kD, numHeads), MakeAttrs(1, 5, 1, 0),
                                             &compileInfo, "Ascend950", 32, 262144);
            TilingInfo nonQuantInfo;
            TilingInfo quantInfo;
            ASSERT_TRUE(ExecuteTiling(nonQuantCtx, nonQuantInfo));
            ASSERT_TRUE(ExecuteTiling(quantCtx, quantInfo));
            ASSERT_EQ(quantInfo.workspaceSizes.size(), 1U);
            EXPECT_EQ(quantInfo.workspaceSizes, nonQuantInfo.workspaceSizes);
            EXPECT_EQ(quantInfo.tilingKey,
                      static_cast<int64_t>(outDtype == ge::DT_FLOAT16 ? kFp8Fp16Tiling950 : kFp8Bf16Tiling950));
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, fp8_d512_reserves_lo_gm_workspace)
{
    // D=512 + groupSize>64 enables LO GM transit (emb>256 && qBaseTile>64). Non-quant cannot
    // reach this path on 950 (headDim capped at 256), so assert an absolute workspace floor.
    constexpr int64_t kEmb512 = 512;
    constexpr int64_t kNumHeads = 128; // groupSize = 128 / 1 > 64 → qBaseTile = 128
    constexpr uint64_t kBlockDim = 32;
    constexpr uint64_t kQBaseTile = 128;
    constexpr uint64_t kMm1L1TileN = 128;
    constexpr uint64_t kKvBaseTile = 128;
    constexpr uint64_t kFdAlign = 512;
    constexpr uint64_t kBlockShapeY = 128;
    auto alignUp = [](uint64_t value, uint64_t alignment) { return (value + alignment - 1U) / alignment * alignment; };
    const uint64_t loBytes = kBlockDim * 2U * kQBaseTile * alignUp(static_cast<uint64_t>(kEmb512), 16U) * sizeof(float);
    const uint64_t sparseBlocks = (kMm1L1TileN + 2ULL * kBlockShapeY - 2U) / kBlockShapeY;
    const uint64_t copyBlocks = std::min(kMm1L1TileN, sparseBlocks * 2U); // paged
    const uint64_t groupBytes = alignUp(64U + copyBlocks * 4U * sizeof(uint64_t), 64U);
    const uint64_t tileWords = ((kKvBaseTile + kMm1L1TileN - 1U) / kMm1L1TileN) * groupBytes / sizeof(uint64_t);
    const uint64_t kvOffsetBytes = kBlockDim * 4U * tileWords * sizeof(uint64_t);
    const int64_t workspaceFloor = static_cast<int64_t>(alignUp(loBytes, kFdAlign) + kvOffsetBytes);

    const float scale512 = 1.0f / sqrt(static_cast<float>(kEmb512));
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara quantCtx(
        "GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT8_E4M3FN, kEmb512, kNumHeads),
        MakeOutputs(ge::DT_FLOAT16, false, kEmb512, kNumHeads),
        MakeAttrs(1, 5, 1, 0, 0.0f, -1, -1, 4, "TND", "PA_BBND", 0, false, kBlockShape, scale512), &compileInfo,
        "Ascend950", kBlockDim, 262144);
    TilingInfo quantInfo;
    ASSERT_TRUE(ExecuteTiling(quantCtx, quantInfo));
    ASSERT_EQ(quantInfo.workspaceSizes.size(), 1U);
    EXPECT_GE(quantInfo.workspaceSizes[0], workspaceFloor);
}

std::vector<gert::TilingContextPara::TensorDescription> MakeBnbdInputs(ge::DataType qkvDtype, int64_t numHeads = kN1)
{
    auto inputs = MakeInputs(qkvDtype, kD, numHeads);
    // PA_BNBD KV: [numBlocks, Nkv, blockSize, D] (middle two axes swapped vs PA_BBND).
    inputs[1] = {{{kMaxBlocks, kN2, kBlockSize, kD}, {kMaxBlocks, kN2, kBlockSize, kD}}, qkvDtype, ge::FORMAT_ND};
    inputs[2] = inputs[1];
    return inputs;
}

TEST_F(generic_block_sparse_attention_tiling_ut, tnd_paged_bnbd_fp16_950)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "GenericBlockSparseAttention", MakeBnbdInputs(ge::DT_FLOAT16), MakeOutputs(ge::DT_FLOAT16, false),
        MakeAttrs(1, 0, 1, 0, 0.0f, -1, -1, 4, "TND", "PA_BNBD"), &compileInfo, "Ascend950", 56, 262144);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kFp16BnbdTiling950);
}

TEST_F(generic_block_sparse_attention_tiling_ut, tnd_paged_bnbd_bf16_950)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "GenericBlockSparseAttention", MakeBnbdInputs(ge::DT_BF16), MakeOutputs(ge::DT_BF16, false),
        MakeAttrs(1, 0, 1, 0, 0.0f, -1, -1, 4, "TND", "PA_BNBD"), &compileInfo, "Ascend950", 56, 262144);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kBf16BnbdTiling950);
}

TEST_F(generic_block_sparse_attention_tiling_ut, tnd_paged_bnbd_fp8_rejected_950)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "GenericBlockSparseAttention", MakeBnbdInputs(ge::DT_FLOAT8_E4M3FN), MakeOutputs(ge::DT_FLOAT16, false),
        MakeAttrs(1, 5, 1, 0, 0.0f, -1, -1, 4, "TND", "PA_BNBD"), &compileInfo, "Ascend950", 56, 262144);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, tnd_paged_bnbd_fp8_high_precision_950)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "GenericBlockSparseAttention", MakeBnbdInputs(ge::DT_FLOAT8_E4M3FN, kFp8HighPrecisionHeads),
        MakeOutputs(ge::DT_FLOAT16, false, kD, kFp8HighPrecisionHeads),
        MakeAttrs(1, 5, 0, 0, 0.0f, -1, -1, 4, "TND", "PA_BNBD"), &compileInfo, "Ascend950", 56, 262144);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 9250000081001012ULL);
}

TEST_F(generic_block_sparse_attention_tiling_ut, tnd_paged_bnbd_fp16_mask0_950)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "GenericBlockSparseAttention", MakeBnbdInputs(ge::DT_FLOAT16), MakeOutputs(ge::DT_FLOAT16, false),
        MakeAttrs(0, 0, 1, 0, 0.0f, -1, -1, 4, "TND", "PA_BNBD"), &compileInfo, "Ascend950", 56, 262144);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kFp16BnbdNoMaskTiling950);
}

TEST_F(generic_block_sparse_attention_tiling_ut, unsupported_layout)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16), MakeOutputs(ge::DT_FLOAT16, false),
        MakeAttrs(1, 0, 0, 0, 0.0f, -1, -1, 4, "TND", "TND"), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, unsupported_layout_sparse_pattern)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16), MakeOutputs(ge::DT_FLOAT16, false),
        MakeAttrs(1, 0, 0, 0, 0.0f, -1, -1, 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, unsupported_residual_block_mode)
{
    for (int64_t residualMode : {-1, 2}) {
        SCOPED_TRACE(residualMode);
        GenericBlockSparseAttentionCompileInfo compileInfo;
        gert::TilingContextPara tilingContextPara(
            "GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16), MakeOutputs(ge::DT_FLOAT16, false),
            MakeAttrs(1, 0, 0, 0, 0.0f, -1, -1, 4, "TND", "PA_BBND", residualMode), &compileInfo, "Ascend910B", 40,
            196608);
        ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, residual_block_mode_arch35_non_quant)
{
    for (bool paged : {false, true}) {
        for (bool returnLse : {false, true}) {
            auto inputs = MakeInputs(ge::DT_FLOAT16);
            if (!paged) {
                inputs[1] = {{{kS2, kN2, kD}, {kS2, kN2, kD}}, ge::DT_FLOAT16, ge::FORMAT_ND};
                inputs[2] = inputs[1];
                inputs[12] = {{{kBatch + 1}, {kBatch + 1}}, ge::DT_INT64, ge::FORMAT_ND};
                inputs[15] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
            }
            const uint64_t baseKey = paged ? kFp16Tiling950 : 9250000030101002ULL;
            const uint64_t lseOffset = returnLse ? 100000000ULL : 0ULL;
            constexpr uint64_t residualOffset = 1000000000ULL;
            GenericBlockSparseAttentionCompileInfo compileInfo;
            gert::TilingContextPara ctx(
                "GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, returnLse),
                MakeAttrs(1, 0, 1, returnLse, 0.0f, -1, -1, 4, "TND", paged ? "PA_BBND" : "TND", 1), &compileInfo,
                "Ascend950", 56, 262144);
            ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, baseKey + lseOffset + residualOffset);
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, residual_block_mode_rejects_arch35_quant)
{
    // residual_block_mode=1 only supports non-quant.
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara ctx(
        "GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT8_E4M3FN), MakeOutputs(ge::DT_FLOAT16, false),
        MakeAttrs(1, 5, 1, 0, 0.0f, -1, -1, 4, "TND", "PA_BBND", 1), &compileInfo, "Ascend950", 56, 262144);
    ExecuteTestCase(ctx, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, residual_none_and_consistent_topk_910b)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16), MakeOutputs(ge::DT_FLOAT16, false),
        MakeAttrs(1, 0, 0, 0, 0.0f, -1, -1, 4, "TND", "PA_BBND", 0, true), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kFp16Tiling910B);
}

TEST_F(generic_block_sparse_attention_tiling_ut, unsupported_mask_type)
{
    for (int64_t maskType : {-1, 2}) {
        SCOPED_TRACE(maskType);
        GenericBlockSparseAttentionCompileInfo compileInfo;
        gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16),
                                                  MakeOutputs(ge::DT_FLOAT16, false), MakeAttrs(maskType, 0, 0, 0),
                                                  &compileInfo, "Ascend910B", 40, 196608);
        ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, unsupported_block_shape_x)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    std::vector<int64_t> badBlockShape = {128, 128};
    gert::TilingContextPara tilingContextPara(
        "GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16), MakeOutputs(ge::DT_FLOAT16, false),
        MakeAttrs(1, 0, 0, 0, 0.0f, -1, -1, 4, "TND", "PA_BBND", 0, false, badBlockShape), &compileInfo, "Ascend910B",
        40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, unsupported_win_left)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16),
                                              MakeOutputs(ge::DT_FLOAT16, false), MakeAttrs(1, 0, 0, 0, 0.0f, 127, -1),
                                              &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, unsupported_bf16_halfsm_910b)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_BF16),
                                              MakeOutputs(ge::DT_BF16, false), MakeAttrs(1, 0, 1, 0), &compileInfo,
                                              "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, unsupported_fp8_on_910b)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT8_E4M3FN),
                                              MakeOutputs(ge::DT_FLOAT16, false), MakeAttrs(1, 5, 0, 0), &compileInfo,
                                              "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, quant_type_dtype_mismatch)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16),
                                              MakeOutputs(ge::DT_FLOAT16, false), MakeAttrs(1, 5, 0, 0), &compileInfo,
                                              "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, missing_metadata)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    inputs[5] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, missing_cu_seq_lengths_q)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    inputs[11] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, missing_seqused_kv)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    inputs[14] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, pa_rejects_cu_seq_lengths_kv)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    inputs[12] = {{{kBatch + 1}, {kBatch + 1}}, ge::DT_INT64, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, scale_value_zero_defaults_to_rsqrt_d)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16), MakeOutputs(ge::DT_FLOAT16, false),
        MakeAttrs(1, 0, 0, 0, 0.0f, -1, -1, 4, "TND", "PA_BBND", 0, false, kBlockShape, 0.0f), &compileInfo,
        "Ascend910B", 40, 196608);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<int64_t>(kFp16Tiling910B));
    ASSERT_NE(tilingInfo.tilingData, nullptr);
    // scaleValue follows the 10 uint32 fields in the POD tiling layout.
    constexpr size_t kScaleValueOffset = 10U * sizeof(uint32_t);
    ASSERT_GT(tilingInfo.tilingDataSize, kScaleValueOffset + sizeof(float));
    float scaleValue = 0.0f;
    std::memcpy(&scaleValue, tilingInfo.tilingData.get() + kScaleValueOffset, sizeof(float));
    EXPECT_FLOAT_EQ(scaleValue, kScale);
}

TEST_F(generic_block_sparse_attention_tiling_ut, high_precision_softmax_950)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16),
                                              MakeOutputs(ge::DT_FLOAT16, false), MakeAttrs(1, 0, 0, 0), &compileInfo,
                                              "Ascend950", 56, 262144);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 9250000071001002ULL);
}

TEST_F(generic_block_sparse_attention_tiling_ut, missing_block_table)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    inputs[15] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, invalid_metadata_size)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    inputs[5] = {{{512}, {512}}, ge::DT_INT32, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, unsupported_fp8_with_lse_950)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT8_E4M3FN),
                                              MakeOutputs(ge::DT_FLOAT16, true), MakeAttrs(1, 5, 1, 1), &compileInfo,
                                              "Ascend950", 56, 262144);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, fp8_high_precision_lse_out_950)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention",
                                              MakeInputs(ge::DT_FLOAT8_E4M3FN, kD, kFp8HighPrecisionHeads),
                                              MakeOutputs(ge::DT_FLOAT16, true, kD, kFp8HighPrecisionHeads),
                                              MakeAttrs(1, 5, 0, 1), &compileInfo, "Ascend950", 56, 262144);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 9250000171001012ULL);
}

TEST_F(generic_block_sparse_attention_tiling_ut, fp8_high_precision_rejects_non128_head_dim)
{
    for (const int64_t headDim : {64, 256}) {
        GenericBlockSparseAttentionCompileInfo compileInfo;
        gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention",
                                                  MakeInputs(ge::DT_FLOAT8_E4M3FN, headDim, kFp8HighPrecisionHeads),
                                                  MakeOutputs(ge::DT_FLOAT16, false, headDim, kFp8HighPrecisionHeads),
                                                  MakeAttrs(1, 5, 0, 0), &compileInfo, "Ascend950", 56, 262144);
        ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, fp8_high_precision_rejects_unsupported_group_sizes)
{
    for (const int64_t group : {17, 32}) {
        GenericBlockSparseAttentionCompileInfo compileInfo;
        gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention",
                                                  MakeInputs(ge::DT_FLOAT8_E4M3FN, kD, group * kN2),
                                                  MakeOutputs(ge::DT_FLOAT16, false, kD, group * kN2),
                                                  MakeAttrs(1, 5, 0, 0), &compileInfo, "Ascend950", 56, 262144);
        ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, fp8_high_precision_supports_groups_1_to_16)
{
    for (int64_t group = 1; group <= 16; ++group) {
        SCOPED_TRACE(group);
        GenericBlockSparseAttentionCompileInfo compileInfo;
        gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention",
                                                  MakeInputs(ge::DT_FLOAT8_E4M3FN, kD, group * kN2),
                                                  MakeOutputs(ge::DT_FLOAT16, false, kD, group * kN2),
                                                  MakeAttrs(1, 5, 0, 0), &compileInfo, "Ascend950", 56, 262144);
        ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 9250000071001012ULL);
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, fp8_high_precision_950)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention",
                                              MakeInputs(ge::DT_FLOAT8_E4M3FN, kD, kFp8HighPrecisionHeads),
                                              MakeOutputs(ge::DT_FLOAT16, false, kD, kFp8HighPrecisionHeads),
                                              MakeAttrs(1, 5, 0, 0), &compileInfo, "Ascend950", 56, 262144);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 9250000071001012ULL);
}

TEST_F(generic_block_sparse_attention_tiling_ut, invalid_block_table_dims)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    inputs[15] = {{{kBatch, kMaxBlocks, 2}, {kBatch, kMaxBlocks, 2}}, ge::DT_INT32, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, invalid_block_table_dtype)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    inputs[15] = {{{kBatch, kMaxBlocks}, {kBatch, kMaxBlocks}}, ge::DT_INT64, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, invalid_cu_seq_lengths_q_dtype)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    inputs[11] = {{{kBatch + 1}, {kBatch + 1}}, ge::DT_INT32, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, invalid_cu_seq_lengths_q_size)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    // Missing the trailing +1 prefix-sum entry (size B instead of B+1) must be rejected.
    inputs[11] = {{{kBatch}, {kBatch}}, ge::DT_INT64, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, invalid_seqused_kv_dtype)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    inputs[14] = {{{kBatch}, {kBatch}}, ge::DT_INT64, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, invalid_seqused_kv_size)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    inputs[14] = {{{kBatch + 1}, {kBatch + 1}}, ge::DT_INT32, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, key_kv_heads_mismatch)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    // key Nkv=2 while sparseBlockIdx dim0 (N_kv)=1 must be rejected.
    inputs[1] = {
        {{kMaxBlocks, kBlockSize, kN2 + 1, kD}, {kMaxBlocks, kBlockSize, kN2 + 1, kD}}, ge::DT_FLOAT16, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, invalid_block_table_batch_zero)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    // blockTable with batch=0 must be rejected instead of silently passing batch_+1 size checks.
    inputs[15] = {{{0, kMaxBlocks}, {0, kMaxBlocks}}, ge::DT_INT32, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, invalid_cu_seq_lengths_q_dims)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    // 2D cuSeqLengthsQ with the same element count must still be rejected: only 1D is valid.
    inputs[11] = {{{1, kBatch + 1}, {1, kBatch + 1}}, ge::DT_INT64, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, invalid_seqused_kv_dims)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    // 2D sequsedKv with the same element count must still be rejected: only 1D is valid.
    inputs[14] = {{{1, kBatch}, {1, kBatch}}, ge::DT_INT32, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, invalid_sparse_idx_dtype)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    // Kernel reads sparseBlockIdx as int32; INT64 indices would be misread as garbage block ids.
    inputs[3] = {{{kN2, kTotalQBlocks, kTopK}, {kN2, kTotalQBlocks, kTopK}}, ge::DT_INT64, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, invalid_sparse_count_dtype)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    // Kernel reads sparseBlockCount as int32; INT64 counts would be misread as garbage block counts.
    inputs[4] = {{{kN2, kTotalQBlocks}, {kN2, kTotalQBlocks}}, ge::DT_INT64, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, invalid_seqused_q_dtype)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    // sequsedQ is optional, but when present the kernel reads it as int32 [batch].
    inputs[13] = {{{kBatch}, {kBatch}}, ge::DT_INT64, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, invalid_seqused_q_size)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    // sequsedQ present with wrong element count must be rejected.
    inputs[13] = {{{kBatch + 1}, {kBatch + 1}}, ge::DT_INT32, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, invalid_block_table_batch_negative)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    // Dynamic-shape -1 must be rejected on int64_t: after uint32_t narrowing it wraps to 4294967295.
    inputs[15] = {{{-1, kMaxBlocks}, {-1, kMaxBlocks}}, ge::DT_INT32, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, seqused_q_present_valid_910b)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    // sequsedQ optional-present happy path: int32 1D [batch] must keep tiling successful.
    inputs[13] = {{{kBatch}, {kBatch}}, ge::DT_INT32, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kFp16Tiling910B);
}

TEST_F(generic_block_sparse_attention_tiling_ut, batched_layouts_arch35)
{
    for (const std::string layout : {"BNSD", "BSND"}) {
        for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16}) {
            for (int64_t flags = 0; flags < 48; ++flags) {
                const bool paged = flags >= 16;
                const bool bnbd = flags >= 32;
                const int64_t mask = flags & 1;
                const int64_t precision = (flags >> 1) & 1;
                const int64_t lse = (flags >> 2) & 1;
                const int64_t residual = (flags >> 3) & 1;
                auto inputs = MakeInputs(dtype);
                if (layout == "BSND") {
                    inputs[0] = {{{2, 3, 6, 65}, {2, 3, 6, 65}}, dtype, ge::FORMAT_ND};
                    inputs[1] = {{{2, 257, 2, 65}, {2, 257, 2, 65}}, dtype, ge::FORMAT_ND};
                } else {
                    inputs[0] = {{{2, 6, 3, 65}, {2, 6, 3, 65}}, dtype, ge::FORMAT_ND};
                    inputs[1] = {{{2, 2, 257, 65}, {2, 2, 257, 65}}, dtype, ge::FORMAT_ND};
                }
                if (paged) {
                    inputs[1] = {{{8, 128, 2, 65}, {8, 128, 2, 65}}, dtype, ge::FORMAT_ND};
                }
                if (bnbd) {
                    inputs[1] = {{{8, 2, 128, 65}, {8, 2, 128, 65}}, dtype, ge::FORMAT_ND};
                }
                inputs[2] = inputs[1];
                inputs[3] = {{{2, 6, 512}, {2, 6, 512}}, ge::DT_INT32, ge::FORMAT_ND};
                inputs[4] = {{{2, 6}, {2, 6}}, ge::DT_INT32, ge::FORMAT_ND};
                for (int index = 11; index < 16; ++index) {
                    inputs[index] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
                }
                inputs[13] = {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND};
                inputs[14] = inputs[13];
                if (paged) {
                    inputs[15] = {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND};
                }
                // Optional cu inputs never replace the required seqused inputs.
                if (flags & 1) {
                    inputs[11] = {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND};
                    if (!paged) {
                        inputs[12] = inputs[11];
                    }
                }
                auto outputs = MakeOutputs(dtype, lse != 0);
                outputs[0] = inputs[0];
                if (lse != 0) {
                    outputs[1] = {{{2, 6, 3, 1}, {2, 6, 3, 1}}, ge::DT_FLOAT, ge::FORMAT_ND};
                }
                const uint64_t key =
                    9250000000000000ULL +
                    (paged ? (bnbd ? 81000000ULL : 71000000ULL) : (layout == "BNSD" ? 10000000ULL : 20000000ULL)) +
                    (layout == "BNSD" ? 3ULL : 4ULL) + (dtype == ge::DT_BF16 ? 22220ULL : 0ULL) + mask * 1000ULL +
                    precision * 100000ULL + lse * 100000000ULL + residual * 1000000000ULL;
                GenericBlockSparseAttentionCompileInfo compileInfo;
                gert::TilingContextPara ctx(
                    "GenericBlockSparseAttention", inputs, outputs,
                    MakeAttrs(mask, 0, precision, lse, 0.0f, -1, -1, 4, layout,
                              paged ? (bnbd ? "PA_BNBD" : "PA_BBND") : layout, residual, false, {1, 17}),
                    &compileInfo, "Ascend950", 32, 262144);
                ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, key);
            }
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, layout_matrix_arch22)
{
    for (const std::string layout : {"TND", "BSND", "BNSD"}) {
        for (int storage = 0; storage < 3; ++storage) {
            for (int precisionMode = 0; precisionMode < 3; ++precisionMode) {
                const auto dtype = precisionMode == 2 ? ge::DT_BF16 : ge::DT_FLOAT16;
                const int64_t precision = precisionMode == 1 ? 1 : 0;
                for (int flags = 0; flags < 16; ++flags) {
                    SCOPED_TRACE(layout + "/" + std::to_string(storage) + "/" + std::to_string(precisionMode) + "/" +
                                 std::to_string(flags));
                    const bool paged = storage != 0;
                    const int64_t mask = flags & 1;
                    const int64_t lse = (flags >> 1) & 1;
                    const int64_t residual = (flags >> 2) & 1;
                    const int64_t y = (flags & 8) ? 16 : 128;
                    auto inputs = MakeInputs(dtype, 65, 6, 2);
                    inputs[0] =
                        layout == "TND" ?
                            gert::TilingContextPara::TensorDescription{{{6, 6, 65}, {6, 6, 65}}, dtype, ge::FORMAT_ND} :
                        layout == "BSND" ?
                            gert::TilingContextPara::TensorDescription{
                                {{2, 3, 6, 65}, {2, 3, 6, 65}}, dtype, ge::FORMAT_ND} :
                            gert::TilingContextPara::TensorDescription{
                                {{2, 6, 3, 65}, {2, 6, 3, 65}}, dtype, ge::FORMAT_ND};
                    inputs[1] = layout == "TND" ?
                                    gert::TilingContextPara::TensorDescription{
                                        {{514, 2, 65}, {514, 2, 65}}, dtype, ge::FORMAT_ND} :
                                layout == "BSND" ?
                                    gert::TilingContextPara::TensorDescription{
                                        {{2, 257, 2, 65}, {2, 257, 2, 65}}, dtype, ge::FORMAT_ND} :
                                    gert::TilingContextPara::TensorDescription{
                                        {{2, 2, 257, 65}, {2, 2, 257, 65}}, dtype, ge::FORMAT_ND};
                    if (paged) {
                        inputs[1] = storage == 1 ?
                                        gert::TilingContextPara::TensorDescription{
                                            {{8, 128, 2, 65}, {8, 128, 2, 65}}, dtype, ge::FORMAT_ND} :
                                        gert::TilingContextPara::TensorDescription{
                                            {{8, 2, 128, 65}, {8, 2, 128, 65}}, dtype, ge::FORMAT_ND};
                    }
                    inputs[2] = inputs[1];
                    inputs[3] = {{{2, 6, 12}, {2, 6, 12}}, ge::DT_INT32, ge::FORMAT_ND};
                    inputs[4] = {{{2, 6}, {2, 6}}, ge::DT_INT32, ge::FORMAT_ND};
                    for (int index = 11; index < 16; ++index) {
                        inputs[index] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
                    }
                    if (layout == "TND") {
                        inputs[11] = {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND};
                        if (!paged) {
                            inputs[12] = inputs[11];
                        }
                    }
                    inputs[13] = {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND};
                    inputs[14] = inputs[13];
                    if (paged) {
                        inputs[15] = {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND};
                    }
                    auto outputs = MakeOutputs(dtype, lse != 0);
                    outputs[0] = inputs[0];
                    if (lse != 0) {
                        outputs[1] = layout == "TND" ?
                                         gert::TilingContextPara::TensorDescription{
                                             {{6, 6, 1}, {6, 6, 1}}, ge::DT_FLOAT, ge::FORMAT_ND} :
                                         gert::TilingContextPara::TensorDescription{
                                             {{2, 6, 3, 1}, {2, 6, 3, 1}}, ge::DT_FLOAT, ge::FORMAT_ND};
                    }
                    const uint64_t key = 9200000000000000ULL + (paged ? 71000000ULL : 30000000ULL) + 2ULL +
                                         (dtype == ge::DT_BF16 ? 22220ULL : 0ULL) + mask * 1000ULL +
                                         precision * 100000ULL + lse * 100000000ULL + residual * 1000000000ULL;
                    GenericBlockSparseAttentionCompileInfo compileInfo;
                    gert::TilingContextPara ctx(
                        "GenericBlockSparseAttention", inputs, outputs,
                        MakeAttrs(mask, 0, precision, lse, 0.0f, -1, -1, 4, layout,
                                  paged ? (storage == 2 ? "PA_BNBD" : "PA_BBND") : layout, residual, false, {1, y}),
                        &compileInfo, "Ascend910B", 40, 196608);
                    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, key);
                }
            }
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, batched_layout_rejects_invalid_shapes_and_mixed_layouts)
{
    for (int invalid = 0; invalid < 7; ++invalid) {
        auto inputs = MakeInputs(ge::DT_FLOAT16);
        inputs[0] = {{{2, 6, 3, 65}, {2, 6, 3, 65}}, ge::DT_FLOAT16, ge::FORMAT_ND};
        inputs[1] = {{{2, 2, 257, 65}, {2, 2, 257, 65}}, ge::DT_FLOAT16, ge::FORMAT_ND};
        inputs[2] = inputs[1];
        inputs[3] = {{{2, 6, 4}, {2, 6, 4}}, ge::DT_INT32, ge::FORMAT_ND};
        inputs[4] = {{{2, 6}, {2, 6}}, ge::DT_INT32, ge::FORMAT_ND};
        for (int index = 11; index < 16; ++index) {
            inputs[index] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
        }
        inputs[11] = {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND};
        inputs[12] = inputs[11];
        inputs[13] = {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND};
        inputs[14] = inputs[13];
        if (invalid == 5 || invalid == 6) {
            inputs[invalid == 5 ? 13 : 14] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
        }
        if (invalid == 0) {
            inputs[1] = {{{3, 2, 257, 65}, {3, 2, 257, 65}}, ge::DT_FLOAT16, ge::FORMAT_ND};
            inputs[2] = inputs[1];
        } else if (invalid == 1) {
            inputs[13] = {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND};
        } else if (invalid == 2) {
            inputs[3] = {{{2, 3, 4}, {2, 3, 4}}, ge::DT_INT32, ge::FORMAT_ND};
        } else if (invalid == 4) {
            // Arch22 now supports BNSD; retain negative coverage with an invalid KV length vector.
            inputs[14] = {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND};
        }
        GenericBlockSparseAttentionCompileInfo compileInfo;
        gert::TilingContextPara ctx("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                    MakeAttrs(1, 0, 0, 0, 0.0f, -1, -1, 4, "BNSD", invalid == 3 ? "BSND" : "BNSD"),
                                    &compileInfo, invalid == 4 ? "Ascend910B" : "Ascend950", 32, 262144);
        ExecuteTestCase(ctx, ge::GRAPH_FAILED);
    }
}

// ============================================================================
// Incremental coverage for error-reporting refactor: attr validation branches
// ============================================================================

TEST_F(generic_block_sparse_attention_tiling_ut, zero_aicore_num_rejected)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    // Platform reporting 0 AIC cores must fail tiling instead of launching 0 blocks.
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16),
                                              MakeOutputs(ge::DT_FLOAT16, false), MakeAttrs(1, 0, 0, 0), &compileInfo,
                                              "Ascend910B", 0, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, invalid_block_shape_size)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    // blockShape with a single element must be rejected: exactly [x, y] is required.
    std::vector<int64_t> badBlockShape = {kBlockShapeX};
    gert::TilingContextPara tilingContextPara(
        "GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16), MakeOutputs(ge::DT_FLOAT16, false),
        MakeAttrs(1, 0, 0, 0, 0.0f, -1, -1, 4, "TND", "PA_BBND", 0, false, badBlockShape), &compileInfo, "Ascend910B",
        40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, invalid_block_shape_value)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    // Non-positive blockShape elements must be rejected before narrowing to uint32_t.
    std::vector<int64_t> badBlockShape = {0, kBlockSize};
    gert::TilingContextPara tilingContextPara(
        "GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16), MakeOutputs(ge::DT_FLOAT16, false),
        MakeAttrs(1, 0, 0, 0, 0.0f, -1, -1, 4, "TND", "PA_BBND", 0, false, badBlockShape), &compileInfo, "Ascend910B",
        40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, unsupported_block_shape_y)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    // blockShapeY must be within [1, 128]; 256 must be rejected.
    std::vector<int64_t> badBlockShape = {kBlockShapeX, 256};
    gert::TilingContextPara tilingContextPara(
        "GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16), MakeOutputs(ge::DT_FLOAT16, false),
        MakeAttrs(1, 0, 0, 0, 0.0f, -1, -1, 4, "TND", "PA_BBND", 0, false, badBlockShape), &compileInfo, "Ascend910B",
        40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, unsupported_softmax_precision_value)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    // softmaxPrecision only accepts 0 (fp32 SM) or 1 (half/low SM).
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16),
                                              MakeOutputs(ge::DT_FLOAT16, false), MakeAttrs(1, 0, 2, 0), &compileInfo,
                                              "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, unsupported_quant_type_value)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    // quantType accepts 0 (none), 5 (FP8), 7 (INT8), or 8 (A8C4).
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16),
                                              MakeOutputs(ge::DT_FLOAT16, false), MakeAttrs(1, 3, 0, 0), &compileInfo,
                                              "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, unsupported_dst_type_max)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    // dst_type_max is reserved and must stay disabled (0).
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16),
                                              MakeOutputs(ge::DT_FLOAT16, false), MakeAttrs(1, 0, 0, 0, 1.0f),
                                              &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, unsupported_win_right)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    // Windowed attention is not wired yet: win_right must stay -1.
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16),
                                              MakeOutputs(ge::DT_FLOAT16, false), MakeAttrs(1, 0, 0, 0, 0.0f, -1, 127),
                                              &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, unsupported_layout_q)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16), MakeOutputs(ge::DT_FLOAT16, false),
        MakeAttrs(1, 0, 0, 0, 0.0f, -1, -1, 4, "BNSD", "PA_BBND"), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, fp8_invalid_out_dtype_950)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    // FP8 Q/K/V requires attentionOut to be float16/bfloat16; float32 output must be rejected.
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT8_E4M3FN),
                                              MakeOutputs(ge::DT_FLOAT, false), MakeAttrs(1, 5, 1, 0), &compileInfo,
                                              "Ascend950", 56, 262144);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// Incremental coverage: PA_BBND KV stride validation branches
// ============================================================================

TEST_F(generic_block_sparse_attention_tiling_ut, kv_dim0_stride_valid_910b)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    // Dim0-only holes are legal for PA_BBND: stride0 >= pageElems and aligned to Nkv*D.
    inputs[1].stride_ = gert::Stride({kMaxBlocks * 2 * kBlockSize * kN2 * kD, kD, kD, 1});
    inputs[1].hasStride_ = true;
    inputs[2].stride_ = gert::Stride({kMaxBlocks * 2 * kBlockSize * kN2 * kD, kD, kD, 1});
    inputs[2].hasStride_ = true;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kFp16Tiling910B);
}

TEST_F(generic_block_sparse_attention_tiling_ut, kv_non_dim0_stride_rejected)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    // Non-contiguous dim1 (blockSize axis) must be rejected: only dim0 may be strided.
    inputs[1].stride_ = gert::Stride({kMaxBlocks * kBlockSize * kN2 * kD, kD * 2, kD, 1});
    inputs[1].hasStride_ = true;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, kv_dim0_stride_below_page_rejected)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    // key stride0 smaller than one page (blockSize*Nkv*D) must be rejected.
    inputs[1].stride_ = gert::Stride({kD, kD, kD, 1});
    inputs[1].hasStride_ = true;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, kv_dim0_stride_unaligned_rejected)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    // key stride0 >= pageElems but not aligned to Nkv*D must be rejected.
    inputs[1].stride_ = gert::Stride({kMaxBlocks * kBlockSize * kN2 * kD + 1, kD, kD, 1});
    inputs[1].hasStride_ = true;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, value_dim0_stride_below_page_rejected)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    // Valid key stride but value stride0 below one page must be rejected on the value branch.
    inputs[1].stride_ = gert::Stride({kMaxBlocks * 2 * kBlockSize * kN2 * kD, kD, kD, 1});
    inputs[1].hasStride_ = true;
    inputs[2].stride_ = gert::Stride({kD, kD, kD, 1});
    inputs[2].hasStride_ = true;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// Incremental coverage: query/key/sparse shape validation branches
// ============================================================================

TEST_F(generic_block_sparse_attention_tiling_ut, unsupported_head_dim)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    // headDim above the 512 upper limit must be rejected on Ascend910B.
    auto inputs = MakeInputs(ge::DT_FLOAT16, 576);
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs,
                                              MakeOutputs(ge::DT_FLOAT16, false, 576), MakeAttrs(1, 0, 0, 0),
                                              &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, key_block_size_mismatch)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    // KV page blockSize must equal blockShapeY (128); 256 must be rejected.
    inputs[1] = {{{kMaxBlocks, 256, kN2, kD}, {kMaxBlocks, 256, kN2, kD}}, ge::DT_FLOAT16, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, sparse_idx_not_3d)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    // sparseBlockIdx must be 3D [N_kv, totalQBlocks, topK].
    inputs[3] = {{{kN2, kTotalQBlocks}, {kN2, kTotalQBlocks}}, ge::DT_INT32, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, kv_heads_not_divisible)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    // numHeads=4 is not divisible by N_kv=3: the GQA group split is impossible.
    inputs[1] = {{{kMaxBlocks, kBlockSize, 3, kD}, {kMaxBlocks, kBlockSize, 3, kD}}, ge::DT_FLOAT16, ge::FORMAT_ND};
    inputs[2] = {{{kMaxBlocks, kBlockSize, 3, kD}, {kMaxBlocks, kBlockSize, 3, kD}}, ge::DT_FLOAT16, ge::FORMAT_ND};
    inputs[3] = {{{3, kTotalQBlocks, kTopK}, {3, kTotalQBlocks, kTopK}}, ge::DT_INT32, ge::FORMAT_ND};
    inputs[4] = {{{3, kTotalQBlocks}, {3, kTotalQBlocks}}, ge::DT_INT32, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, zero_kv_heads_rejected)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    // N_kv=0 must be rejected before the modulo division.
    inputs[3] = {{{0, kTotalQBlocks, kTopK}, {0, kTotalQBlocks, kTopK}}, ge::DT_INT32, ge::FORMAT_ND};
    inputs[4] = {{{0, kTotalQBlocks}, {0, kTotalQBlocks}}, ge::DT_INT32, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, group_size_exceeds_max)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    // groupSize = numHeads/N_kv = 256 exceeds the L0C/UB-safe bound (128).
    inputs[0] = {{{kT, 256, kD}, {kT, 256, kD}}, ge::DT_FLOAT16, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, arch22_nonquant_large_topk)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    for (int64_t topK : {512LL, 513LL, 1024LL, 2147483647LL}) {
        auto inputs = MakeInputs(ge::DT_FLOAT16);
        inputs[3] = {{{kN2, kTotalQBlocks, topK}, {kN2, kTotalQBlocks, topK}}, ge::DT_INT32, ge::FORMAT_ND};
        gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs,
                                                  MakeOutputs(ge::DT_FLOAT16, false), MakeAttrs(1, 0, 0, 0),
                                                  &compileInfo, "Ascend910B", 40, 196608);
        ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kFp16Tiling910B);
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, topk_exceeds_int32_max)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    constexpr int64_t topK = 2147483648LL;
    inputs[3] = {{{kN2, kTotalQBlocks, topK}, {kN2, kTotalQBlocks, topK}}, ge::DT_INT32, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, sparse_count_not_2d)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    // sparseBlockCount must be 2D [N_kv, totalQBlocks].
    inputs[4] = {{{kN2, kTotalQBlocks, 2}, {kN2, kTotalQBlocks, 2}}, ge::DT_INT32, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, sparse_count_shape_mismatch)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    // sparseBlockCount must match sparseBlockIdx [N_kv, totalQBlocks].
    inputs[4] = {{{kN2, kTotalQBlocks + 1}, {kN2, kTotalQBlocks + 1}}, ge::DT_INT32, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, qkv_dtype_mismatch)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    // key dtype (bf16) diverging from query (fp16) must be rejected.
    inputs[1] = {{{kMaxBlocks, kBlockSize, kN2, kD}, {kMaxBlocks, kBlockSize, kN2, kD}}, ge::DT_BF16, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, unsupported_qkv_dtype)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    // fp32 Q/K/V is not supported: only fp16/bf16/fp8_e4m3fn.
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT),
                                              MakeOutputs(ge::DT_FLOAT, false), MakeAttrs(1, 0, 0, 0), &compileInfo,
                                              "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// Incremental coverage: metadata / reserved optional input branches
// ============================================================================

TEST_F(generic_block_sparse_attention_tiling_ut, metadata_not_1d)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    // metadata is a fixed-size 1D protocol buffer; 2D must be rejected.
    inputs[5] = {{{2, 512}, {2, 512}}, ge::DT_INT32, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, invalid_metadata_dtype)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    // Kernel reads metadata as int32; int64 must be rejected.
    inputs[5] = {{{1024}, {1024}}, ge::DT_INT64, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, atten_mask_not_supported)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    // atten_mask is reserved and must stay absent.
    inputs[6] = {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, q_dequant_scale_not_supported)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    // q/k/v_dequant_scale are reserved and must stay absent.
    inputs[7] = {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, p_quant_scale_not_supported)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    // p_quant_scale is reserved and must stay absent.
    inputs[10] = {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, seqused_q_not_1d)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    // sequsedQ is optional, but when present it must be 1D [batch].
    inputs[13] = {{{1, kBatch}, {1, kBatch}}, ge::DT_INT32, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, supported_dense_head_dim_910b)
{
    constexpr size_t kEmbeddingSizeOffset = 3U * sizeof(uint32_t);
    constexpr size_t kFdStaticEnabledOffset = 120U;
    for (int64_t headDim : {1, 17, 64, 128, 240, 256}) {
        SCOPED_TRACE(headDim);
        GenericBlockSparseAttentionCompileInfo compileInfo;
        gert::TilingContextPara tilingContextPara(
            "GenericBlockSparseAttention", MakeDenseInputs(ge::DT_FLOAT16, headDim),
            MakeOutputs(ge::DT_FLOAT16, false, headDim), MakeAttrs(1, 0, 0, 0, 0.0f, -1, -1, 4, "TND", "TND"),
            &compileInfo, "Ascend910B", 40, 196608);
        TilingInfo tilingInfo;
        ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
        EXPECT_EQ(tilingInfo.tilingKey, static_cast<int64_t>(kFp16TndTndTiling910B));
        ASSERT_NE(tilingInfo.tilingData, nullptr);
        ASSERT_GT(tilingInfo.tilingDataSize, kFdStaticEnabledOffset + sizeof(uint32_t));
        EXPECT_EQ(ReadTilingU32(tilingInfo, kEmbeddingSizeOffset), static_cast<uint32_t>(headDim));
        EXPECT_EQ(ReadTilingU32(tilingInfo, kFdStaticEnabledOffset), 0U);
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, fd_enabled_dense_head_dim_17_910b)
{
    constexpr int64_t headDim = 17;
    constexpr int64_t topK = 12;
    constexpr size_t kEmbeddingSizeOffset = 3U * sizeof(uint32_t);
    constexpr size_t kFdStaticEnabledOffset = 120U;
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "GenericBlockSparseAttention", MakeDenseInputs(ge::DT_FLOAT16, headDim, topK),
        MakeOutputs(ge::DT_FLOAT16, false, headDim), MakeAttrs(1, 0, 0, 0, 0.0f, -1, -1, 4, "TND", "TND"), &compileInfo,
        "Ascend910B", 40, 196608);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<int64_t>(kFp16TndTndTiling910B));
    ASSERT_NE(tilingInfo.tilingData, nullptr);
    ASSERT_GT(tilingInfo.tilingDataSize, kFdStaticEnabledOffset + sizeof(uint32_t));
    EXPECT_EQ(ReadTilingU32(tilingInfo, kEmbeddingSizeOffset), static_cast<uint32_t>(headDim));
    EXPECT_EQ(ReadTilingU32(tilingInfo, kFdStaticEnabledOffset), 1U);
}

TEST_F(generic_block_sparse_attention_tiling_ut, unsupported_dense_head_dim_910b)
{
    for (int64_t headDim : {int64_t(0), int64_t(257)}) {
        SCOPED_TRACE(headDim);
        GenericBlockSparseAttentionCompileInfo compileInfo;
        gert::TilingContextPara tilingContextPara(
            "GenericBlockSparseAttention", MakeDenseInputs(ge::DT_FLOAT16, headDim),
            MakeOutputs(ge::DT_FLOAT16, false, headDim), MakeAttrs(1, 0, 0, 0, 0.0f, -1, -1, 4, "TND", "TND"),
            &compileInfo, "Ascend910B", 40, 196608);
        ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, lse_all_arch22_layouts_and_residual_modes)
{
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16}) {
        for (bool paged : {false, true}) {
            for (int64_t residual : {0, 1}) {
                for (int64_t mask : {0, 1}) {
                    for (int64_t precision : {0, 1}) {
                        if (dtype == ge::DT_BF16 && precision == 1) {
                            continue;
                        }
                        GenericBlockSparseAttentionCompileInfo compileInfo;
                        auto inputs = paged ? MakeInputs(dtype) : MakeDenseInputs(dtype);
                        gert::TilingContextPara ctx("GenericBlockSparseAttention", inputs, MakeOutputs(dtype, true),
                                                    MakeAttrs(mask, 0, precision, 1, 0.0f, -1, -1, 4, "TND",
                                                              paged ? "PA_BBND" : "TND", residual),
                                                    &compileInfo, "Ascend910B", 40, 196608);
                        const uint64_t key = 9200000000000000ULL + (paged ? 71000000ULL : 30000000ULL) +
                                             (dtype == ge::DT_BF16 ? 22220ULL : 0ULL) + precision * 100000ULL +
                                             mask * 1000ULL + 2ULL + 100000000ULL + residual * 1000000000ULL;
                        ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, key);
                    }
                }
            }
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, union_mode_requires_exact_mask_910b)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    constexpr int64_t kUnionG = 2;
    constexpr int64_t kUnionU = 2;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    inputs[3] = {{{kN2, kUnionG, kUnionU}, {kN2, kUnionG, kUnionU}}, ge::DT_INT32, ge::FORMAT_ND};
    inputs[4] = {{{kN2, kUnionG}, {kN2, kUnionG}}, ge::DT_INT32, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara(
        "GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
        MakeAttrs(3, 0, 0, 0, 0.0f, -1, -1, 4, "TND", "PA_BBND", 0, false, kUnionBlockShape), &compileInfo,
        "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, union_mode_nonquant_910b)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    constexpr int64_t kUnionG = 2;
    constexpr int64_t kUnionU = 2;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    inputs[3] = {{{kN2, kUnionG, kUnionU}, {kN2, kUnionG, kUnionU}}, ge::DT_INT32, ge::FORMAT_ND};
    inputs[4] = {{{kN2, kUnionG}, {kN2, kUnionG}}, ge::DT_INT32, ge::FORMAT_ND};
    inputs[6] = {
        {{kN2, kUnionG, kUnionU * kBlockSize}, {kN2, kUnionG, kUnionU * kBlockSize}}, ge::DT_INT32, ge::FORMAT_ND};
    constexpr uint64_t kFp16UnionTiling910B = 9200002071000002ULL;
    gert::TilingContextPara tilingContextPara(
        "GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
        MakeAttrs(3, 0, 0, 0, 0.0f, -1, -1, 4, "TND", "PA_BBND", 0, false, kUnionBlockShape), &compileInfo,
        "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kFp16UnionTiling910B);
}

TEST_F(generic_block_sparse_attention_tiling_ut, union_mode_rejects_block_shape_x_not_8)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    constexpr int64_t kUnionG = 2;
    constexpr int64_t kUnionU = 2;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    inputs[3] = {{{kN2, kUnionG, kUnionU}, {kN2, kUnionG, kUnionU}}, ge::DT_INT32, ge::FORMAT_ND};
    inputs[4] = {{{kN2, kUnionG}, {kN2, kUnionG}}, ge::DT_INT32, ge::FORMAT_ND};
    inputs[6] = {
        {{kN2, kUnionG, kUnionU * kBlockSize}, {kN2, kUnionG, kUnionU * kBlockSize}}, ge::DT_INT32, ge::FORMAT_ND};
    // Legacy [1, Y] is no longer accepted in union mode; unionSize must come from X=8.
    gert::TilingContextPara tilingContextPara(
        "GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
        MakeAttrs(3, 0, 0, 0, 0.0f, -1, -1, 4, "TND", "PA_BBND", 0, false, kBlockShape), &compileInfo, "Ascend910B", 40,
        196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, union_mode_requires_exact_mask_950)
{
    // maskType=3 without atten_mask must fail.
    GenericBlockSparseAttentionCompileInfo compileInfo;
    constexpr int64_t kUnionG = 2;
    constexpr int64_t kUnionU = 2;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    inputs[3] = {{{kN2, kUnionG, kUnionU}, {kN2, kUnionG, kUnionU}}, ge::DT_INT32, ge::FORMAT_ND};
    inputs[4] = {{{kN2, kUnionG}, {kN2, kUnionG}}, ge::DT_INT32, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara(
        "GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
        MakeAttrs(3, 0, 0, 0, 0.0f, -1, -1, 4, "TND", "PA_BBND", 0, false, kUnionBlockShape), &compileInfo, "Ascend950",
        56, 262144);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, union_mode_rejects_block_shape_x_1)
{
    // Union reads unionSize from blockShapeX; X=1 must be rejected.
    GenericBlockSparseAttentionCompileInfo compileInfo;
    constexpr int64_t kUnionG = 2;
    constexpr int64_t kUnionU = 2;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    inputs[3] = {{{kN2, kUnionG, kUnionU}, {kN2, kUnionG, kUnionU}}, ge::DT_INT32, ge::FORMAT_ND};
    inputs[4] = {{{kN2, kUnionG}, {kN2, kUnionG}}, ge::DT_INT32, ge::FORMAT_ND};
    inputs[6] = {
        {{kN2, kUnionG, kUnionU * kBlockSize}, {kN2, kUnionG, kUnionU * kBlockSize}}, ge::DT_INT32, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara(
        "GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
        MakeAttrs(3, 0, 0, 0, 0.0f, -1, -1, 4, "TND", "PA_BBND", 0, false, kBlockShape), &compileInfo, "Ascend950", 56,
        262144);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, union_mode_nonquant_950)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    constexpr int64_t kUnionG = 2;
    constexpr int64_t kUnionU = 2;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    inputs[3] = {{{kN2, kUnionG, kUnionU}, {kN2, kUnionG, kUnionU}}, ge::DT_INT32, ge::FORMAT_ND};
    inputs[4] = {{{kN2, kUnionG}, {kN2, kUnionG}}, ge::DT_INT32, ge::FORMAT_ND};
    // exactMask INT32 [kvHeads, G, U * blockShapeY]
    inputs[6] = {
        {{kN2, kUnionG, kUnionU * kBlockSize}, {kN2, kUnionG, kUnionU * kBlockSize}}, ge::DT_INT32, ge::FORMAT_ND};
    // SoftmaxPrecision=0 (FP32 SoftmaxUnion); expected key matches tilingkey.h UNION macro.
    constexpr uint64_t kFp16UnionTiling950 = 9250002071000002ULL;
    gert::TilingContextPara tilingContextPara(
        "GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
        MakeAttrs(3, 0, 0, 0, 0.0f, -1, -1, 4, "TND", "PA_BBND", 0, false, kUnionBlockShape), &compileInfo, "Ascend950",
        56, 262144);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kFp16UnionTiling950);
}

TEST_F(generic_block_sparse_attention_tiling_ut, union_mode_pa_bnbd_fp8_950)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    constexpr int64_t kUnionG = 2;
    constexpr int64_t kUnionU = 2;
    auto inputs = MakeInputs(ge::DT_FLOAT8_E4M3FN);
    // PA_BNBD KV axes.
    inputs[1] = {
        {{kMaxBlocks, kN2, kBlockSize, kD}, {kMaxBlocks, kN2, kBlockSize, kD}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND};
    inputs[2] = inputs[1];
    inputs[3] = {{{kN2, kUnionG, kUnionU}, {kN2, kUnionG, kUnionU}}, ge::DT_INT32, ge::FORMAT_ND};
    inputs[4] = {{{kN2, kUnionG}, {kN2, kUnionG}}, ge::DT_INT32, ge::FORMAT_ND};
    inputs[6] = {
        {{kN2, kUnionG, kUnionU * kBlockSize}, {kN2, kUnionG, kUnionU * kBlockSize}}, ge::DT_INT32, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara(
        "GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
        MakeAttrs(3, 5, 0, 0, 0.0f, -1, -1, 4, "TND", "PA_BNBD", 0, false, kUnionBlockShape), &compileInfo, "Ascend950",
        56, 262144);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 9250002081000012ULL);
}

namespace {
// Byte offsets in the serialized GBSA tiling protocol (not TilingDef getter caches).
auto MakeUnionInputs(ge::DataType dtype, int64_t dim, int64_t heads, int64_t blockY)
{
    auto inputs = MakeInputs(dtype, dim, heads);
    inputs[3] = {{{1, 1, 2}, {1, 1, 2}}, ge::DT_INT32, ge::FORMAT_ND};
    inputs[4] = {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND};
    inputs[6] = {{{1, 1, 2 * blockY}, {1, 1, 2 * blockY}}, ge::DT_INT32, ge::FORMAT_ND};
    return inputs;
}
} // namespace

TEST_F(generic_block_sparse_attention_tiling_ut, union_fp8_paged_rejects_y_not_dividing_page)
{
    // FP8+PA skips actualKvLen clip; Y must divide pageSize 128.
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara ctx(
        "GenericBlockSparseAttention", MakeUnionInputs(ge::DT_FLOAT8_E4M3FN, 128, 16, 100),
        MakeOutputs(ge::DT_FLOAT16, false, 128, 16),
        MakeAttrs(3, 5, 0, 0, 0.0f, -1, -1, 4, "TND", "PA_BBND", 0, false, {kUnionBlockShapeX, 100}), &compileInfo,
        "Ascend950", 56, 262144);
    ExecuteTestCase(ctx, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, union_ub_budget_includes_exact_mask)
{
    const auto unsafe = GbsaKernelArch35::MakeUbLayout(32, 512, 256, 128, 1, true, false);
    EXPECT_GT(unsafe.end, 256U * 1024U);
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16}) {
        GenericBlockSparseAttentionCompileInfo compileInfo;
        gert::TilingContextPara ctx(
            "GenericBlockSparseAttention", MakeUnionInputs(dtype, 256, 4, 1), MakeOutputs(dtype, false, 256, 4),
            MakeAttrs(3, 0, 0, 0, 0.0f, -1, -1, 4, "TND", "PA_BBND", 0, false, {kUnionBlockShapeX, 1}), &compileInfo,
            "Ascend950", 56, 262144);
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(ctx, info));
        EXPECT_EQ(ReadTilingU32(info, 176U), 32U);
        EXPECT_EQ(ReadTilingU32(info, 180U), 256U);
        EXPECT_EQ(ReadTilingU32(info, 120U), 0U);
        const auto ub = GbsaKernelArch35::MakeUbLayout(32, 256, 256, ReadTilingU32(info, 188U), 1, true, false);
        EXPECT_LE(ub.end, GbsaKernelArch35::USABLE_UB_BYTES);
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, union_fp8_l1_and_lse_all_output_types)
{
    for (int64_t dim : {128}) {
        for (int64_t heads : {1, 4, 8, 16}) {
            for (auto out : {ge::DT_FLOAT16, ge::DT_BF16}) {
                for (bool lse : {false, true}) {
                    SCOPED_TRACE(std::to_string(dim) + "/" + std::to_string(heads));
                    GenericBlockSparseAttentionCompileInfo compileInfo;
                    gert::TilingContextPara ctx(
                        "GenericBlockSparseAttention", MakeUnionInputs(ge::DT_FLOAT8_E4M3FN, dim, heads, 128),
                        MakeOutputs(out, lse, dim, heads),
                        MakeAttrs(3, 5, 0, lse, 0.0f, -1, -1, 4, "TND", "PA_BBND", 0, false, kUnionBlockShape),
                        &compileInfo, "Ascend950", 56, 262144);
                    TilingInfo info;
                    ASSERT_TRUE(ExecuteTiling(ctx, info));
                    const uint32_t qK = ReadTilingU32(info, 192U);
                    EXPECT_GE(qK, static_cast<uint32_t>(dim));
                    const uint32_t qBytes = ReadTilingU32(info, 184U) * qK * ReadTilingU32(info, 216U);
                    const uint32_t kBytes =
                        ReadTilingU32(info, 188U) * ReadTilingU32(info, 196U) * ReadTilingU32(info, 220U);
                    const uint32_t pBytes =
                        ReadTilingU32(info, 200U) * ReadTilingU32(info, 208U) * ReadTilingU32(info, 228U);
                    const uint32_t vBytes =
                        ReadTilingU32(info, 204U) * ReadTilingU32(info, 212U) * ReadTilingU32(info, 224U);
                    EXPECT_GE(qBytes, ReadTilingU32(info, 184U) * GbsaKernelArch35::AlignUp(dim, 32U));
                    EXPECT_LE(qBytes + kBytes + pBytes + vBytes, 512U * 1024U);
                    EXPECT_EQ(ReadTilingU32(info, 120U), 0U);
                    EXPECT_EQ(info.tilingKey,
                              static_cast<int64_t>((out == ge::DT_FLOAT16 ? 9250002071000012ULL : 9250002071000022ULL) +
                                                   (lse ? 100000000ULL : 0ULL)));
                    const auto ub = GbsaKernelArch35::MakeUbLayout(ReadTilingU32(info, 176U), ReadTilingU32(info, 180U),
                                                                   dim, ReadTilingU32(info, 188U), 128, true, true);
                    EXPECT_LE(ub.end, GbsaKernelArch35::USABLE_UB_BYTES);
                }
            }
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, union_fp8_rejects_unsupported_head_dim)
{
    for (int64_t dim : {1, 64, 127, 129, 192, 256, 512}) {
        for (bool lse : {false, true}) {
            GenericBlockSparseAttentionCompileInfo compileInfo;
            gert::TilingContextPara ctx(
                "GenericBlockSparseAttention", MakeUnionInputs(ge::DT_FLOAT8_E4M3FN, dim, 16, 128),
                MakeOutputs(ge::DT_FLOAT16, lse, dim, 16),
                MakeAttrs(3, 5, 0, lse, 0.0f, -1, -1, 4, "TND", "PA_BBND", 0, false, kUnionBlockShape), &compileInfo,
                "Ascend950", 56, 262144);
            ExecuteTestCase(ctx, ge::GRAPH_FAILED);
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, union_fp8_rejects_softmax_precision_1)
{
    // SoftmaxUnion runs FP32 S for FP8 as well; only softmaxPrecision=0 is valid.
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara ctx("GenericBlockSparseAttention", MakeUnionInputs(ge::DT_FLOAT8_E4M3FN, 128, 16, 128),
                                MakeOutputs(ge::DT_FLOAT16, false, 128, 16),
                                MakeAttrs(3, 5, 1, 0, 0.0f, -1, -1, 4, "TND", "PA_BBND", 0, false, kUnionBlockShape),
                                &compileInfo, "Ascend950", 56, 262144);
    ExecuteTestCase(ctx, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, union_mode_rejects_mxfp8)
{
    // Union mode only supports quantType=0 or 5; MXFP8 (quantType=6) must fail.
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara ctx("GenericBlockSparseAttention", MakeUnionInputs(ge::DT_FLOAT8_E4M3FN, 128, 16, 128),
                                MakeOutputs(ge::DT_FLOAT16, false, 128, 16),
                                MakeAttrs(3, 6, 0, 0, 0.0f, -1, -1, 4, "TND", "PA_BBND", 0, false, kUnionBlockShape),
                                &compileInfo, "Ascend950", 56, 262144);
    ExecuteTestCase(ctx, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, union_batched_layout_dispatch)
{
    for (const std::string layout : {"BSND", "BNSD"}) {
        for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN}) {
            const bool fp8 = dtype == ge::DT_FLOAT8_E4M3FN;
            for (const std::string& kvLayout : {layout, std::string("PA_BBND"), std::string("PA_BNBD")}) {
                const bool paged = kvLayout != layout;
                auto inputs = MakeUnionInputs(dtype, 128, 4, 128);
                if (layout == "BSND") {
                    inputs[0] = {{{2, 17, 4, 128}, {2, 17, 4, 128}}, dtype, ge::FORMAT_ND};
                    inputs[1] = {{{2, 129, 1, 128}, {2, 129, 1, 128}}, dtype, ge::FORMAT_ND};
                } else {
                    inputs[0] = {{{2, 4, 17, 128}, {2, 4, 17, 128}}, dtype, ge::FORMAT_ND};
                    inputs[1] = {{{2, 1, 129, 128}, {2, 1, 129, 128}}, dtype, ge::FORMAT_ND};
                }
                if (kvLayout == "PA_BBND") {
                    inputs[1] = {{{4, 128, 1, 128}, {4, 128, 1, 128}}, dtype, ge::FORMAT_ND};
                } else if (kvLayout == "PA_BNBD") {
                    inputs[1] = {{{4, 1, 128, 128}, {4, 1, 128, 128}}, dtype, ge::FORMAT_ND};
                }
                inputs[2] = inputs[1];
                inputs[11] = inputs[12] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
                inputs[13] = inputs[14] = {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND};
                inputs[15] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
                if (paged) {
                    inputs[15] = {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND};
                }
                const auto outDtype = fp8 ? ge::DT_FLOAT16 : dtype;
                const bool bnsd = layout == "BNSD";
                const uint64_t kvField = kvLayout == "PA_BNBD" ? 81000000ULL :
                                         paged                 ? 71000000ULL :
                                         bnsd                  ? 10000000ULL :
                                                                 20000000ULL;
                const uint64_t dtypeField = fp8 ? 10ULL : dtype == ge::DT_BF16 ? 22220ULL : 0ULL;
                for (bool arch35 : {false, true}) {
                    if (!arch35 && fp8) {
                        continue;
                    }
                    const uint64_t base = arch35 ? 9250002000000000ULL : 9200002000000000ULL;
                    const uint64_t baseKey = base + kvField + dtypeField + (bnsd ? 3ULL : 4ULL);
                    for (bool returnLse : {false, true}) {
                        auto outputs = MakeOutputs(outDtype, returnLse);
                        if (bnsd) {
                            outputs[0] = {{{2, 4, 17, 128}, {2, 4, 17, 128}}, outDtype, ge::FORMAT_ND};
                        } else {
                            outputs[0] = {{{2, 17, 4, 128}, {2, 17, 4, 128}}, outDtype, ge::FORMAT_ND};
                        }
                        if (returnLse) {
                            outputs[1] = {{{2, 4, 17, 1}, {2, 4, 17, 1}}, ge::DT_FLOAT, ge::FORMAT_ND};
                        }
                        const uint64_t key = baseKey + (returnLse ? kUnionLseKeyOffset : 0ULL);
                        GenericBlockSparseAttentionCompileInfo compileInfo;
                        gert::TilingContextPara ctx("GenericBlockSparseAttention", inputs, outputs,
                                                    MakeAttrs(3, fp8 ? 5 : 0, 0, returnLse, 0.0f, -1, -1, 4, layout,
                                                              kvLayout, 0, false, kUnionBlockShape),
                                                    &compileInfo, arch35 ? "Ascend950" : "Ascend910B", arch35 ? 56 : 40,
                                                    arch35 ? 262144 : 196608);
                        ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, key);
                    }
                }
            }
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, union_mode_tnd_pa_bnbd)
{
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16}) {
        auto inputs = MakeUnionInputs(dtype, 128, 4, 128);
        // PA_BNBD: [numBlocks, Nkv, blockSize, D]
        inputs[1] = {{{kMaxBlocks, 1, kBlockSize, 128}, {kMaxBlocks, 1, kBlockSize, 128}}, dtype, ge::FORMAT_ND};
        inputs[2] = inputs[1];
        for (bool arch35 : {false, true}) {
            const uint64_t base = arch35 ? 9250002081000002ULL : 9200002081000002ULL;
            const uint64_t baseKey = base + (dtype == ge::DT_BF16 ? 22220ULL : 0ULL);
            for (bool returnLse : {false, true}) {
                const uint64_t key = baseKey + (returnLse ? kUnionLseKeyOffset : 0ULL);
                GenericBlockSparseAttentionCompileInfo compileInfo;
                gert::TilingContextPara ctx(
                    "GenericBlockSparseAttention", inputs, MakeOutputs(dtype, returnLse, 128, 4),
                    MakeAttrs(3, 0, 0, returnLse, 0.0f, -1, -1, 4, "TND", "PA_BNBD", 0, false, kUnionBlockShape),
                    &compileInfo, arch35 ? "Ascend950" : "Ascend910B", arch35 ? 56 : 40, arch35 ? 262144 : 196608);
                ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, key);
            }
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, union_mode_accepts_multi_kv_heads)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    constexpr int64_t kQHeads = 8;
    constexpr int64_t kKvHeads = 2;
    auto inputs = MakeInputs(ge::DT_FLOAT16, 128, kQHeads, kKvHeads);
    inputs[3] = {{{kKvHeads, 1, 2}, {kKvHeads, 1, 2}}, ge::DT_INT32, ge::FORMAT_ND};
    inputs[4] = {{{kKvHeads, 1}, {kKvHeads, 1}}, ge::DT_INT32, ge::FORMAT_ND};
    inputs[6] = {{{kKvHeads, 1, 2 * 128}, {kKvHeads, 1, 2 * 128}}, ge::DT_INT32, ge::FORMAT_ND};
    gert::TilingContextPara ctx("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false, 128, kQHeads),
                                MakeAttrs(3, 0, 0, 0, 0.0f, -1, -1, 4, "TND", "PA_BBND", 0, false, kUnionBlockShape),
                                &compileInfo, "Ascend950", 56, 262144);
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, 9250002071000002ULL);
}

TEST_F(generic_block_sparse_attention_tiling_ut, quant_type_dtype_and_soc_matrix)
{
    for (const std::string soc : {"Ascend910B", "Ascend950"}) {
        for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_INT8}) {
            for (int64_t quantType : {0, 3, 5, 6, 7}) {
                const bool arch35 = soc == "Ascend950";
                const bool supported = (quantType == 0 && (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16)) ||
                                       (quantType == 5 && arch35 && dtype == ge::DT_FLOAT8_E4M3FN) ||
                                       (quantType == 7 && !arch35 && dtype == ge::DT_INT8);
                auto inputs = dtype == ge::DT_INT8 ? MakeInt8Inputs(ge::DT_FLOAT, 128) : MakeInputs(dtype, 128);
                GenericBlockSparseAttentionCompileInfo compileInfo;
                gert::TilingContextPara context(
                    "GenericBlockSparseAttention", inputs,
                    MakeOutputs(dtype == ge::DT_BF16 ? ge::DT_BF16 : ge::DT_FLOAT16, false, 128),
                    MakeAttrs(1, quantType, arch35 && dtype == ge::DT_FLOAT8_E4M3FN ? 1 : 0, 0), &compileInfo,
                    soc.c_str(), 40, 262144);
                TilingInfo info;
                EXPECT_EQ(ExecuteTiling(context, info), supported)
                    << "soc=" << soc << ", dtype=" << dtype << ", quantType=" << quantType;
                if (supported && dtype == ge::DT_INT8) {
                    EXPECT_EQ(info.tilingKey, 9200000071001032ULL);
                }
            }
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, int8_rejects_legacy_quant_type_six)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara context("GenericBlockSparseAttention", MakeInt8Inputs(),
                                    MakeOutputs(ge::DT_FLOAT16, false, 256), MakeAttrs(1, 6, 0, 0), &compileInfo,
                                    "Ascend910B", 40, 196608);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

namespace {}

namespace {
std::vector<gert::TilingContextPara::TensorDescription> MakeMxfp8Inputs(int64_t headDim = kD, bool withPScale = false)
{
    auto inputs = MakeInputs(ge::DT_FLOAT8_E4M3FN, headDim);
    inputs[7] = {
        {{kT, kN1, (headDim + 63) / 64, 2}, {kT, kN1, (headDim + 63) / 64, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND};
    inputs[8] = {
        {{kMaxBlocks, kBlockSize, kN2, (headDim + 63) / 64, 2}, {kMaxBlocks, kBlockSize, kN2, (headDim + 63) / 64, 2}},
        ge::DT_FLOAT8_E8M0,
        ge::FORMAT_ND};
    inputs[9] = {{{kMaxBlocks, kBlockSize / 64, kN2, headDim, 2}, {kMaxBlocks, kBlockSize / 64, kN2, headDim, 2}},
                 ge::DT_FLOAT8_E8M0,
                 ge::FORMAT_ND};
    if (withPScale) {
        inputs[10] = {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND};
    }
    return inputs;
}

bool RunMxfp8Validation(const std::vector<gert::TilingContextPara::TensorDescription>& inputs,
                        std::vector<gert::TilingContextPara::TensorDescription> outputs = MakeOutputs(ge::DT_BF16,
                                                                                                      false, 256),
                        bool returnLse = false)
{
    outputs[0].shape_ = inputs[0].shape_;
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara context("GenericBlockSparseAttention", inputs, outputs,
                                    MakeAttrs(0, 6, 0, returnLse, 0.0f, -1, -1, 4, "TND", "PA_BBND", 0, false, {1, 4}),
                                    &compileInfo, "Ascend950", 28, 262144);
    TilingInfo info;
    return ExecuteTiling(context, info);
}
} // namespace

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_scale_validation_errors)
{
    ASSERT_TRUE(RunMxfp8Validation(MakeMxfp8Inputs(256)));
    const auto validInputs = MakeMxfp8Inputs(256, true);
    ASSERT_TRUE(RunMxfp8Validation(validInputs));
    for (int index : {7, 8, 9, 10}) {
        for (int error : {0, 1, 2, 3}) {
            if (index == 10 && error == 0) {
                continue; // P scale is optional.
            }
            SCOPED_TRACE(testing::Message() << "scale index=" << index << ", error=" << error);
            auto inputs = validInputs;
            if (error == 0) {
                inputs[index] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
            } else if (error == 1) {
                inputs[index].dtype_ = ge::DT_UINT8;
            } else if (error == 2) {
                auto& shape = inputs[index].shape_.MutableStorageShape();
                shape.SetDimNum(2);
                shape.SetDim(1, 1);
            } else {
                auto& shape = inputs[index].shape_.MutableStorageShape();
                const size_t axis = shape.GetDimNum() - 1U;
                shape.SetDim(axis, shape.GetDim(axis) + 1);
            }
            EXPECT_FALSE(RunMxfp8Validation(inputs));
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_rejects_noncontiguous_paged_kv_stride)
{
    constexpr int64_t pageElements = kBlockSize * kN2 * 256;
    ASSERT_TRUE(RunMxfp8Validation(MakeMxfp8Inputs(256)));
    for (int index : {1, 2}) {
        SCOPED_TRACE(index);
        auto inputs = MakeMxfp8Inputs(256);
        inputs[index].hasStride_ = true;
        inputs[index].stride_ = gert::Stride({2 * pageElements, kN2 * 256, 256, 1});
        EXPECT_FALSE(RunMxfp8Validation(inputs));
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_group_size_validation_boundary)
{
    for (int64_t group : {64, 65}) {
        SCOPED_TRACE(group);
        auto inputs = MakeMxfp8Inputs(256);
        inputs[0] = {{{kT, group, 256}, {kT, group, 256}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND};
        inputs[7] = {{{kT, group, 4, 2}, {kT, group, 4, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND};
        EXPECT_EQ(RunMxfp8Validation(inputs), group <= 64);
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_kv512_runtime_batch_and_descriptor_workspace)
{
    for (int64_t batch : {0, 1, 2, 3, 4, 8, 16}) {
        for (int64_t d : {192, 256}) {
            for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16}) {
                const int64_t q = 129, kv = 4608, nq = d == 192 ? 64 : 24, nkv = d == 192 ? 4 : 2;
                const int64_t y = d == 192 ? 8 : 4;
                auto inputs = MakeMxfp8Inputs(d);
                inputs[0] = {{{batch, q, nq, d}, {batch, q, nq, d}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND};
                inputs[1] = {{{batch, kv, nkv, d}, {batch, kv, nkv, d}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND};
                inputs[2] = inputs[1];
                inputs[3] = {{{nkv, batch * q, 513}, {nkv, batch * q, 513}}, ge::DT_INT32, ge::FORMAT_ND};
                inputs[4] = {{{nkv, batch * q}, {nkv, batch * q}}, ge::DT_INT32, ge::FORMAT_ND};
                inputs[7] = {{{batch, q, nq, d / 64, 2}, {batch, q, nq, d / 64, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND};
                inputs[8] = {
                    {{batch, kv, nkv, d / 64, 2}, {batch, kv, nkv, d / 64, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND};
                inputs[9] = {{{batch, nkv, d, 2}, {batch, nkv, d, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND};
                for (int index : {11, 12, 15})
                    inputs[index] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
                inputs[13] = {{{batch}, {batch}}, ge::DT_INT32, ge::FORMAT_ND};
                inputs[14] = inputs[13];
                auto outputs = MakeOutputs(dtype, false, d, nq);
                outputs[0] = inputs[0];
                outputs[0].dtype_ = dtype;
                GenericBlockSparseAttentionCompileInfo compileInfo;
                gert::TilingContextPara ctx("GenericBlockSparseAttention", inputs, outputs,
                                            MakeAttrs(1, 6, 0, 0, 0.0f, -1, -1, 4, "BSND", "BSND", 0, false, {1, y}),
                                            &compileInfo, "Ascend950", 28, 262144);
                if (batch == 0) {
                    ExecuteTestCase(ctx, ge::GRAPH_FAILED);
                    continue;
                }
                TilingInfo info;
                ASSERT_TRUE(ExecuteTiling(ctx, info));
                const uint64_t expectedBase = 9250002900000020ULL;
                EXPECT_EQ(info.tilingKey, expectedBase + (dtype == ge::DT_FLOAT16 ? 0U : 1U));
                const uint64_t tile = 512U;
                // Descriptor rings precede the independently aligned FP32 FD
                // partials. Host reserves them even if metadata selects prefill.
                uint64_t bytes = 512U + 28U * 12U * 8U * Mxfp8Tiling::DescriptorTileWords(256U, y, false);
                constexpr uint64_t partialCapacity = 8U + 28U - 1U;
                bytes = (bytes + 511U) / 512U * 512U;
                bytes += partialCapacity * 2U * 8U * sizeof(float);
                bytes = (bytes + 511U) / 512U * 512U;
                bytes += partialCapacity * (nq / nkv) * d * sizeof(float);
                ASSERT_EQ(info.workspaceSizes.size(), 1U);
                EXPECT_EQ(info.workspaceSizes[0], bytes + 16U * 1024U * 1024U);
                // Descriptor rings start at the source-compatible 64B offset and carry a Host-selected per-tile stride.
                constexpr size_t cacheOffset = 232;
                uint64_t cacheBegin = 0U;
                uint32_t tensorWords = 0U;
                ASSERT_GE(info.tilingDataSize, cacheOffset + sizeof(cacheBegin) + sizeof(tensorWords));
                std::memcpy(&cacheBegin, info.tilingData.get() + cacheOffset, sizeof(cacheBegin));
                std::memcpy(&tensorWords, info.tilingData.get() + cacheOffset + sizeof(cacheBegin),
                            sizeof(tensorWords));
                EXPECT_EQ(cacheBegin, 512U);
                EXPECT_EQ(tensorWords, Mxfp8Tiling::DescriptorTileWords(256U, y, false));
            }
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_implicit_residual_mode_dispatch)
{
    for (int64_t dim : {192, 256}) {
        for (int64_t mode : {0, 1, 2}) {
            GenericBlockSparseAttentionCompileInfo compileInfo;
            gert::TilingContextPara context(
                "GenericBlockSparseAttention", MakeMxfp8Inputs(dim), MakeOutputs(ge::DT_BF16, false, dim),
                MakeAttrs(1, 6, 0, 0, 0.0f, -1, -1, 4, "TND", "PA_BBND", mode, false, {1, 4}), &compileInfo,
                "Ascend950", 28, 262144);
            ExecuteTestCase(context, mode <= 1 ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED,
                            mode == 1 ? 9250002900000153ULL : 9250002900000053ULL);
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_tnd_paged_grouped_shape_boundary)
{
    for (int64_t headDim : {1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 192, 255, 256}) {
        for (int64_t blockY : {0, 1, 2, 4, 8, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 256}) {
            auto inputs = MakeMxfp8Inputs(headDim);
            inputs[3] = {{{kN2, kTotalQBlocks, 512}, {kN2, kTotalQBlocks, 512}}, ge::DT_INT32, ge::FORMAT_ND};
            GenericBlockSparseAttentionCompileInfo compileInfo;
            gert::TilingContextPara ctx("GenericBlockSparseAttention", inputs,
                                        MakeOutputs(ge::DT_FLOAT16, false, headDim),
                                        MakeAttrs(1, 6, 0, 0, 0.0f, -1, -1, 4, "TND", "PA_BBND", 0, false, {1, blockY}),
                                        &compileInfo, "Ascend950", 56, 262144);
            const bool grouped = (headDim == 192 || headDim == 256) && (blockY == 4 || blockY == 8 || blockY == 16 ||
                                                                        blockY == 32 || blockY == 64 || blockY == 128);
            ExecuteTestCase(ctx, grouped ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED, 9250002900000052ULL);
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_tnd_layout_dispatch_and_lse_boundary)
{
    constexpr int64_t dim = 256;
    for (const std::string layout : {"TND", "PA_BBND", "PA_BNBD"}) {
        for (auto outputType : {ge::DT_FLOAT16, ge::DT_BF16}) {
            for (bool lse : {false, true}) {
                auto inputs = MakeMxfp8Inputs(dim);
                if (layout == "TND") {
                    inputs[1] = {{{kS2, kN2, dim}, {kS2, kN2, dim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND};
                    inputs[2] = inputs[1];
                    inputs[8] = {{{kS2, kN2, dim / 64, 2}, {kS2, kN2, dim / 64, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND};
                    inputs[9] = {{{kBatch, kN2, dim, 2}, {kBatch, kN2, dim, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND};
                    inputs[12] = inputs[11];
                    inputs[14] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
                    inputs[15] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
                } else if (layout == "PA_BNBD") {
                    inputs[1] = {{{kMaxBlocks, kN2, kBlockSize, dim}, {kMaxBlocks, kN2, kBlockSize, dim}},
                                 ge::DT_FLOAT8_E4M3FN,
                                 ge::FORMAT_ND};
                    inputs[2] = inputs[1];
                    inputs[8] = {
                        {{kMaxBlocks, kN2, kBlockSize, dim / 64, 2}, {kMaxBlocks, kN2, kBlockSize, dim / 64, 2}},
                        ge::DT_FLOAT8_E8M0,
                        ge::FORMAT_ND};
                    inputs[9] = {
                        {{kMaxBlocks, kN2, kBlockSize / 64, dim, 2}, {kMaxBlocks, kN2, kBlockSize / 64, dim, 2}},
                        ge::DT_FLOAT8_E8M0,
                        ge::FORMAT_ND};
                }
                GenericBlockSparseAttentionCompileInfo compileInfo;
                gert::TilingContextPara ctx("GenericBlockSparseAttention", inputs, MakeOutputs(outputType, lse, dim),
                                            MakeAttrs(0, 6, 0, lse, 0.0f, -1, -1, 4, "TND", layout, 0, false, {1, 4}),
                                            &compileInfo, "Ascend950", 56, 262144);
                TilingInfo info;
                ASSERT_TRUE(ExecuteTiling(ctx, info));
                const uint64_t base = layout == "TND"     ? 9250002900000050ULL :
                                      layout == "PA_BBND" ? 9250002900000052ULL :
                                                            9250002900000054ULL;
                EXPECT_EQ(info.tilingKey, base + (outputType == ge::DT_FLOAT16 ? 0U : 1U));
                // The physical key stays shared; bit 8 of the decimal logical
                // key in tiling data selects the optional output specialization.
                constexpr size_t logicalKeyOffset = 88U;
                uint64_t logicalKey = 0U;
                ASSERT_GE(info.tilingDataSize, logicalKeyOffset + sizeof(logicalKey));
                std::memcpy(&logicalKey, info.tilingData.get() + logicalKeyOffset, sizeof(logicalKey));
                EXPECT_EQ((logicalKey / 100000000ULL) % 10ULL, static_cast<uint64_t>(lse));
            }
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_pipeline_bounded_scale_reservation)
{
    EXPECT_EQ(Mxfp8Tiling::RAW_STAGES, 3U);
    EXPECT_LE(Mxfp8Tiling::UbBytes(), Mxfp8Tiling::UB_CAPACITY);
    EXPECT_EQ(Mxfp8Tiling::DescriptorRows(192, true), 128U);
    EXPECT_EQ(Mxfp8Tiling::DescriptorRows(192, false), 256U);
    EXPECT_EQ(Mxfp8Tiling::DescriptorRows(256, true), 256U);
}

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_fp32_scores_precision_contract)
{
    // MXFP8 accepts only FP32 precision 0 for either dense layout and output type.
    for (const std::string layout : {"BSND", "BNSD"}) {
        for (auto outputType : {ge::DT_FLOAT16, ge::DT_BF16}) {
            for (int64_t precision : {0LL, 1LL, -1LL, 2LL, 4294967296LL}) {
                auto inputs = MakeMxfp8Inputs(256);
                const bool headMajor = layout == "BNSD";
                SCOPED_TRACE(layout + "/" + std::to_string(outputType) + "/" + std::to_string(precision));
                const int64_t q1 = headMajor ? 24 : 4096;
                const int64_t q2 = headMajor ? 4096 : 24;
                const int64_t k1 = headMajor ? 2 : 16384;
                const int64_t k2 = headMajor ? 16384 : 2;
                inputs[0] = {{{1, q1, q2, 256}, {1, q1, q2, 256}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND};
                inputs[1] = {{{1, k1, k2, 256}, {1, k1, k2, 256}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND};
                inputs[2] = inputs[1];
                inputs[3] = {{{2, 4096, 512}, {2, 4096, 512}}, ge::DT_INT32, ge::FORMAT_ND};
                inputs[4] = {{{2, 4096}, {2, 4096}}, ge::DT_INT32, ge::FORMAT_ND};
                inputs[7] = {{{1, q1, q2, 4, 2}, {1, q1, q2, 4, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND};
                inputs[8] = {{{1, k1, k2, 4, 2}, {1, k1, k2, 4, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND};
                inputs[9] = {{{1, 2, 256, 2}, {1, 2, 256, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND};
                for (int index : {11, 12, 15}) {
                    inputs[index] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
                }
                inputs[13] = {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND};
                inputs[14] = inputs[13];
                auto outputs = MakeOutputs(outputType, false, 256, 24);
                outputs[0] = inputs[0];
                outputs[0].dtype_ = outputType;
                GenericBlockSparseAttentionCompileInfo compileInfo;
                gert::TilingContextPara ctx(
                    "GenericBlockSparseAttention", inputs, outputs,
                    MakeAttrs(0, 6, precision, 0, 0.0f, -1, -1, 4, layout, layout, 0, false, {1, 4}), &compileInfo,
                    "Ascend950", 56, 262144);
                if (precision != 0) {
                    ExecuteTestCase(ctx, ge::GRAPH_FAILED);
                    continue;
                }
                TilingInfo info;
                ASSERT_TRUE(ExecuteTiling(ctx, info));
                EXPECT_EQ(info.tilingKey, (headMajor ? 9250002900000032ULL : 9250002900000020ULL) +
                                              (outputType == ge::DT_BF16 ? 1U : 0U));
            }
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_grouped_raw_ring_capacity)
{
    // Per-AIV Raw three tiles retain 24 KiB; total includes the source's local descriptor capacity.
    EXPECT_EQ(Mxfp8Tiling::RAW_STAGES * 256U * 32U, 24576U);
    EXPECT_EQ(Mxfp8Tiling::UbBytes(), 171136U);
    EXPECT_EQ(Mxfp8Tiling::L1Bytes(), 463104U);
}

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_full_tile_group_gate_and_l1_contract)
{
    for (int64_t dim : {192, 256}) {
        for (int64_t group : {1, 12, 16, 17, 31, 32, 33, 48, 64, 65, 128}) {
            auto inputs = MakeMxfp8Inputs(dim);
            inputs[0] = {{{kT, group, dim}, {kT, group, dim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND};
            inputs[7] = {{{kT, group, dim / 64, 2}, {kT, group, dim / 64, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND};
            GenericBlockSparseAttentionCompileInfo compileInfo;
            gert::TilingContextPara context("GenericBlockSparseAttention", inputs,
                                            MakeOutputs(ge::DT_BF16, false, dim, group),
                                            MakeAttrs(0, 6, 0, 0, 0.0f, -1, -1, 4, "TND", "PA_BBND", 0, false, {1, 4}),
                                            &compileInfo, "Ascend950", 28, 262144);
            if (group > 64) {
                ExecuteTestCase(context, ge::GRAPH_FAILED);
                continue;
            }
            TilingInfo info;
            ASSERT_TRUE(ExecuteTiling(context, info));
            // Host selects complete groups: Q32/KV512 through G32, then Q64/KV256.
            const uint32_t q = group <= 32 ? 32U : 64U;
            const uint32_t kv = group <= 32 ? 512U : 256U;
            EXPECT_EQ(ReadTilingU32(info, 176U), q);
            EXPECT_EQ(ReadTilingU32(info, 180U), kv);
            const uint32_t expected[] = {q, kv, 256, 256, q, 256, kv, kv, 1, 1, 2, 3};
            for (size_t i = 0; i < 12; ++i) {
                EXPECT_EQ(ReadTilingU32(info, 184U + i * sizeof(uint32_t)), expected[i]);
            }
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_q64_capacity_and_descriptor_contract)
{
    EXPECT_EQ(Mxfp8Tiling::UbBytes(64), 220800U);
    EXPECT_EQ(Mxfp8Tiling::L1Bytes(64), 268800U);
    EXPECT_LE(Mxfp8Tiling::UbBytes(64), Mxfp8Tiling::UB_CAPACITY);
    EXPECT_LE(Mxfp8Tiling::L1Bytes(64), Mxfp8Tiling::L1_CAPACITY);
    for (uint32_t dim : {192U, 256U}) {
        for (bool paged : {false, true}) {
            EXPECT_EQ(Mxfp8Tiling::DescriptorRows(dim, paged, 256), 128U);
            for (uint32_t y : {4U, 8U, 16U, 32U, 64U, 128U}) {
                EXPECT_EQ(Mxfp8Tiling::DescriptorTileWords(128, y, paged, 256),
                          2U * GbsaKernelArch35::Mxfp8Descriptor::GroupWords(128, y, paged));
            }
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_block_y16_preserves_capacity_tiers)
{
    for (int64_t dim : {192, 256}) {
        for (int64_t group : {1, 12, 16, 32, 33, 64}) {
            for (int64_t residual : {0, 1}) {
                auto inputs = MakeMxfp8Inputs(dim);
                inputs[0] = {{{kT, group, dim}, {kT, group, dim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND};
                inputs[7] = {{{kT, group, dim / 64, 2}, {kT, group, dim / 64, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND};
                GenericBlockSparseAttentionCompileInfo compileInfo;
                gert::TilingContextPara context(
                    "GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_BF16, false, dim, group),
                    MakeAttrs(0, 6, 0, 0, 0.0f, -1, -1, 4, "TND", "PA_BBND", residual, false, {1, 16}), &compileInfo,
                    "Ascend950", 28, 262144);
                TilingInfo info;
                ASSERT_TRUE(ExecuteTiling(context, info));
                EXPECT_EQ(info.tilingKey, 9250002900000053ULL + (group > 32 ? 200U : 0U) + residual * 100U);
                EXPECT_EQ(ReadTilingU32(info, 176U), group <= 32 ? 32U : 64U);
                EXPECT_EQ(ReadTilingU32(info, 180U), group <= 32 ? 512U : 256U);
                // The Host-selected descriptor tile stride is smaller for Y16;
                // the Q/K/V/P capacities and buffer counts remain unchanged.
                const uint32_t tileWords = group > 32 ? 160U : (dim == 192 ? 320U : 288U);
                EXPECT_EQ(ReadTilingU32(info, 240U), tileWords);
            }
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_block_y32_preserves_capacity_tiers)
{
    for (int64_t dim : {192, 256}) {
        for (int64_t group : {1, 12, 16, 32, 33, 64}) {
            for (int64_t residual : {0, 1}) {
                auto inputs = MakeMxfp8Inputs(dim);
                inputs[0] = {{{kT, group, dim}, {kT, group, dim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND};
                inputs[7] = {{{kT, group, dim / 64, 2}, {kT, group, dim / 64, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND};
                GenericBlockSparseAttentionCompileInfo compileInfo;
                gert::TilingContextPara context(
                    "GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_BF16, false, dim, group),
                    MakeAttrs(0, 6, 0, 0, 0.0f, -1, -1, 4, "TND", "PA_BBND", residual, false, {1, 32}), &compileInfo,
                    "Ascend950", 28, 262144);
                TilingInfo info;
                ASSERT_TRUE(ExecuteTiling(context, info));
                EXPECT_EQ(info.tilingKey, 9250002900000053ULL + (group > 32 ? 200U : 0U) + residual * 100U);
                EXPECT_EQ(ReadTilingU32(info, 176U), group <= 32 ? 32U : 64U);
                EXPECT_EQ(ReadTilingU32(info, 180U), group <= 32 ? 512U : 256U);
                // The Host-selected descriptor tile stride is smaller for Y32;
                // the Q/K/V/P capacities and buffer counts remain unchanged.
                const uint32_t tileWords = group > 32 ? 96U : (dim == 192 ? 192U : 160U);
                EXPECT_EQ(ReadTilingU32(info, 240U), tileWords);
            }
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_large_sparse_blocks_preserve_capacity_tiers)
{
    for (int64_t blockY : {64, 128}) {
        for (int64_t dim : {192, 256}) {
            for (int64_t group : {1, 12, 16, 32, 33, 64}) {
                for (int64_t residual : {0, 1}) {
                    auto inputs = MakeMxfp8Inputs(dim);
                    inputs[0] = {{{kT, group, dim}, {kT, group, dim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND};
                    inputs[7] = {
                        {{kT, group, dim / 64, 2}, {kT, group, dim / 64, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND};
                    GenericBlockSparseAttentionCompileInfo compileInfo;
                    gert::TilingContextPara context(
                        "GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_BF16, false, dim, group),
                        MakeAttrs(0, 6, 0, 0, 0.0f, -1, -1, 4, "TND", "PA_BBND", residual, false, {1, blockY}),
                        &compileInfo, "Ascend950", 28, 262144);
                    TilingInfo info;
                    ASSERT_TRUE(ExecuteTiling(context, info));
                    EXPECT_EQ(info.tilingKey, 9250002900000053ULL + (group > 32 ? 200U : 0U) + residual * 100U);
                    const uint32_t q = group <= 32 ? 32U : 64U;
                    const uint32_t kv = group <= 32 ? 512U : 256U;
                    EXPECT_EQ(ReadTilingU32(info, 176U), q);
                    EXPECT_EQ(ReadTilingU32(info, 180U), kv);
                    const uint32_t expected[] = {q, kv, 256, 256, q, 256, kv, kv, 1, 1, 2, 3};
                    for (size_t i = 0; i < 12; ++i) {
                        EXPECT_EQ(ReadTilingU32(info, 184U + i * sizeof(uint32_t)), expected[i]);
                    }
                    // Larger sparse blocks reduce descriptor records, not the data/scale buffer capacities.
                    const uint32_t tileWords = blockY == 64 ? (group > 32 ? 64U : (dim == 192 ? 128U : 96U)) :
                                                              (group > 32 ? 48U : (dim == 192 ? 96U : 64U));
                    EXPECT_EQ(ReadTilingU32(info, 240U), tileWords);
                }
            }
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_rejects_invalid_lse_output)
{
    const auto inputs = MakeMxfp8Inputs(256);
    const auto validOutputs = MakeOutputs(ge::DT_BF16, true, 256);
    ASSERT_TRUE(RunMxfp8Validation(inputs, validOutputs, true));
    for (int error : {0, 1, 2}) {
        SCOPED_TRACE(error);
        auto outputs = validOutputs;
        if (error == 0) {
            outputs[1].dtype_ = ge::DT_BF16;
        } else if (error == 1) {
            outputs[1] = {{{kT, kN1, 2}, {kT, kN1, 2}}, ge::DT_FLOAT, ge::FORMAT_ND};
        } else {
            outputs[1] = {{{kT, kN1, 1, 1}, {kT, kN1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND};
        }
        EXPECT_FALSE(RunMxfp8Validation(inputs, outputs, true));
    }
}
