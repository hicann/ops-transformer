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
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "attention/generic_block_sparse_attention/op_host/generic_block_sparse_attention_tiling.h"

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
constexpr int64_t kD = 128;
constexpr int64_t kTopK = 2;
constexpr int64_t kBlockSize = 128;
constexpr int64_t kBlockShapeX = 1;
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
} // namespace

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
} // namespace

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_rejects_tnd_paged_shapes_and_outputs)
{
    for (int64_t headDim : {128, 256}) {
        for (auto outputType : {ge::DT_FLOAT16, ge::DT_BF16}) {
            for (bool withPScale : {false, true}) {
                GenericBlockSparseAttentionCompileInfo compileInfo;
                gert::TilingContextPara ctx("GenericBlockSparseAttention", MakeMxfp8Inputs(headDim, withPScale),
                                            MakeOutputs(outputType, false, headDim), MakeAttrs(1, 6, 1, 0),
                                            &compileInfo, "Ascend950", 56, 262144);
                ExecuteTestCase(ctx, ge::GRAPH_FAILED);
            }
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_requires_all_e8m0_scales)
{
    for (int index : {7, 8, 9}) {
        for (auto invalidType : {ge::DT_UNDEFINED, ge::DT_FLOAT, ge::DT_UINT8}) {
            auto inputs = MakeMxfp8Inputs();
            // An absent scale, a scalar scale and raw uint8 must all be rejected.
            inputs[index] = {{{1}, {1}}, invalidType, ge::FORMAT_ND};
            if (invalidType == ge::DT_UNDEFINED) {
                inputs[index] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
            }
            GenericBlockSparseAttentionCompileInfo compileInfo;
            gert::TilingContextPara ctx("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                        MakeAttrs(1, 6, 1, 0), &compileInfo, "Ascend950", 56, 262144);
            ExecuteTestCase(ctx, ge::GRAPH_FAILED);
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_rejects_paged_block_y128)
{
    constexpr int64_t headDim = 256;
    constexpr int64_t queryHeads = 128;
    auto inputs = MakeMxfp8Inputs(headDim);
    inputs[0] = {{{kT, queryHeads, headDim}, {kT, queryHeads, headDim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND};
    inputs[7] = {
        {{kT, queryHeads, headDim / 64, 2}, {kT, queryHeads, headDim / 64, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND};
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara ctx("GenericBlockSparseAttention", inputs,
                                MakeOutputs(ge::DT_FLOAT16, false, headDim, queryHeads), MakeAttrs(1, 6, 1, 0),
                                &compileInfo, "Ascend950", 56, 262144);
    ExecuteTestCase(ctx, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_rejects_paged_layout_for_all_groups)
{
    for (int64_t queryHeads : {4, 64, 128}) {
        auto inputs = MakeMxfp8Inputs();
        inputs[0] = {{{kT, queryHeads, kD}, {kT, queryHeads, kD}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND};
        inputs[7] = {{{kT, queryHeads, kD / 64, 2}, {kT, queryHeads, kD / 64, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND};
        GenericBlockSparseAttentionCompileInfo compileInfo;
        gert::TilingContextPara ctx("GenericBlockSparseAttention", inputs,
                                    MakeOutputs(ge::DT_FLOAT16, false, kD, queryHeads), MakeAttrs(1, 6, 1, 0),
                                    &compileInfo, "Ascend950", 56, 262144);
        ExecuteTestCase(ctx, ge::GRAPH_FAILED);
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_direct_kv_workspace_independent_of_sequence)
{
    struct Geometry {
        int64_t d, q, kv, nq, nkv;
    };
    for (const auto shape :
         {Geometry{192, 128, 4096, 64, 4}, Geometry{192, 128, 6229, 16, 1}, Geometry{192, 128, 6230, 16, 1},
          Geometry{192, 127, 4096, 16, 1}, Geometry{192, 128, 4096, 48, 3}, Geometry{256, 128, 4608, 24, 2},
          Geometry{256, 128, 4672, 24, 2}, Geometry{256, 128, 4673, 24, 2}, Geometry{256, 127, 4096, 24, 2},
          Geometry{256, 128, 4096, 12, 3}}) {
        const int64_t y = shape.d == 192 ? 8 : 4;
        auto inputs = MakeMxfp8Inputs(shape.d);
        inputs[0] = {
            {{1, shape.q, shape.nq, shape.d}, {1, shape.q, shape.nq, shape.d}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND};
        inputs[1] = {{{1, shape.kv, shape.nkv, shape.d}, {1, shape.kv, shape.nkv, shape.d}},
                     ge::DT_FLOAT8_E4M3FN,
                     ge::FORMAT_ND};
        inputs[2] = inputs[1];
        inputs[3] = {{{shape.nkv, shape.q, 513}, {shape.nkv, shape.q, 513}}, ge::DT_INT32, ge::FORMAT_ND};
        inputs[4] = {{{shape.nkv, shape.q}, {shape.nkv, shape.q}}, ge::DT_INT32, ge::FORMAT_ND};
        inputs[7] = {{{1, shape.q, shape.nq, shape.d / 64, 2}, {1, shape.q, shape.nq, shape.d / 64, 2}},
                     ge::DT_FLOAT8_E8M0,
                     ge::FORMAT_ND};
        inputs[8] = {{{1, shape.kv, shape.nkv, shape.d / 64, 2}, {1, shape.kv, shape.nkv, shape.d / 64, 2}},
                     ge::DT_FLOAT8_E8M0,
                     ge::FORMAT_ND};
        inputs[9] = {{{1, shape.nkv, shape.d, 2}, {1, shape.nkv, shape.d, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND};
        for (int index : {11, 12, 15})
            inputs[index] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
        inputs[13] = {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND};
        inputs[14] = inputs[13];
        auto outputs = MakeOutputs(ge::DT_BF16, false, shape.d, shape.nq);
        outputs[0] = inputs[0];
        outputs[0].dtype_ = ge::DT_BF16;
        GenericBlockSparseAttentionCompileInfo compileInfo;
        gert::TilingContextPara ctx("GenericBlockSparseAttention", inputs, outputs,
                                    MakeAttrs(1, 6, 1, 0, 0.0f, -1, -1, 4, "BSND", "BSND", 0, false, {1, y}),
                                    &compileInfo, "Ascend950", 28, 262144);
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(ctx, info));
        ASSERT_EQ(info.tilingKey, 9250002900000021ULL);
        ASSERT_EQ(info.blockNum, 28U);
        ASSERT_EQ(info.workspaceSizes.size(), 1U);
        // The task header was removed. Descriptor rings are followed by the
        // aligned FD partial buffers, even when metadata selects prefill.
        uint64_t bytes = 28U * 4U * 8U * Mxfp8Tiling::DescriptorTileWords(512U);
        constexpr uint64_t partialCapacity = 8U + 28U - 1U;
        bytes = (bytes + 511U) / 512U * 512U;
        bytes += partialCapacity * 2U * 8U * sizeof(float);
        bytes = (bytes + 511U) / 512U * 512U;
        bytes += partialCapacity * (shape.nq / shape.nkv) * shape.d * sizeof(float);
        EXPECT_EQ(info.workspaceSizes[0], bytes + 16U * 1024U * 1024U);
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_packed_runtime_batch_dispatch_and_descriptor_workspace)
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
                                            MakeAttrs(1, 6, 1, 0, 0.0f, -1, -1, 4, "BSND", "BSND", 0, false, {1, y}),
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
                uint64_t bytes = 28U * 4U * 8U * Mxfp8Tiling::DescriptorTileWords(tile);
                constexpr uint64_t partialCapacity = 8U + 28U - 1U;
                bytes = (bytes + 511U) / 512U * 512U;
                bytes += partialCapacity * 2U * 8U * sizeof(float);
                bytes = (bytes + 511U) / 512U * 512U;
                bytes += partialCapacity * (nq / nkv) * d * sizeof(float);
                ASSERT_EQ(info.workspaceSizes.size(), 1U);
                EXPECT_EQ(info.workspaceSizes[0], bytes + 16U * 1024U * 1024U);
                // The retired K/V cache fields must remain disabled.
                constexpr size_t cacheOffset = 232;
                uint64_t cacheBegin = 0U;
                uint32_t tensorWords = 0U;
                ASSERT_GE(info.tilingDataSize, cacheOffset + sizeof(cacheBegin) + sizeof(tensorWords));
                std::memcpy(&cacheBegin, info.tilingData.get() + cacheOffset, sizeof(cacheBegin));
                std::memcpy(&tensorWords, info.tilingData.get() + cacheOffset + sizeof(cacheBegin),
                            sizeof(tensorWords));
                EXPECT_EQ(cacheBegin, 0U);
                EXPECT_EQ(tensorWords, 0U);
            }
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_scale_pipeline_explicit_dispatch)
{
    for (int64_t batch : {0, 1, 2, 3, 4, 8, 16}) {
        for (int64_t d : {192, 256}) {
            for (int64_t capacity : {513, 514, 8193}) {
                for (int64_t kvLength : {4608, 16384, 21846, 65537}) {
                    for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16}) {
                        const int64_t q = 129, kv = kvLength, nq = d == 192 ? 64 : 24, nkv = d == 192 ? 4 : 2;
                        const int64_t y = d == 192 ? 8 : 4;
                        auto inputs = MakeMxfp8Inputs(d);
                        inputs[0] = {{{batch, q, nq, d}, {batch, q, nq, d}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND};
                        inputs[1] = {{{batch, kv, nkv, d}, {batch, kv, nkv, d}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND};
                        inputs[2] = inputs[1];
                        inputs[3] = {
                            {{nkv, batch * q, capacity}, {nkv, batch * q, capacity}}, ge::DT_INT32, ge::FORMAT_ND};
                        inputs[4] = {{{nkv, batch * q}, {nkv, batch * q}}, ge::DT_INT32, ge::FORMAT_ND};
                        inputs[7] = {
                            {{batch, q, nq, d / 64, 2}, {batch, q, nq, d / 64, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND};
                        inputs[8] = {{{batch, kv, nkv, d / 64, 2}, {batch, kv, nkv, d / 64, 2}},
                                     ge::DT_FLOAT8_E8M0,
                                     ge::FORMAT_ND};
                        inputs[9] = {{{batch, nkv, d, 2}, {batch, nkv, d, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND};
                        for (int index : {11, 12, 15})
                            inputs[index] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
                        inputs[13] = {{{batch}, {batch}}, ge::DT_INT32, ge::FORMAT_ND};
                        inputs[14] = inputs[13];
                        auto outputs = MakeOutputs(dtype, false, d, nq);
                        outputs[0] = inputs[0];
                        outputs[0].dtype_ = dtype;
                        GenericBlockSparseAttentionCompileInfo compileInfo;
                        gert::TilingContextPara ctx(
                            "GenericBlockSparseAttention", inputs, outputs,
                            MakeAttrs(1, 6, 1, 0, 0.0f, -1, -1, 4, "BSND", "BSND", 0, false, {1, y}), &compileInfo,
                            "Ascend950", 28, 262144);
                        if (batch == 0) {
                            ExecuteTestCase(ctx, ge::GRAPH_FAILED);
                        } else {
                            TilingInfo info;
                            ASSERT_TRUE(ExecuteTiling(ctx, info));
                            EXPECT_EQ(info.tilingKey,
                                      dtype == ge::DT_FLOAT16 ? 9250002900000020ULL : 9250002900000021ULL);
                            EXPECT_LE(Mxfp8Tiling::PipelineUbEnd(d, 512), Mxfp8Tiling::UB_CAPACITY);
                        }
                    }
                }
            }
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_coalesced_pipeline_prefill_boundaries)
{
    for (int64_t batch : {1, 3}) {
        for (int64_t mask : {0, 1}) {
            for (int64_t capacity : {513, 514}) {
                for (int64_t kvLength :
                     {127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 4096, 4320, 4608, 8192, 8193}) {
                    for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16}) {
                        const int64_t d = 256, q = kvLength, kv = kvLength, nq = 24, nkv = 2;
                        const int64_t y = 4;
                        auto inputs = MakeMxfp8Inputs(d);
                        inputs[0] = {{{batch, q, nq, d}, {batch, q, nq, d}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND};
                        inputs[1] = {{{batch, kv, nkv, d}, {batch, kv, nkv, d}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND};
                        inputs[2] = inputs[1];
                        inputs[3] = {
                            {{nkv, batch * q, capacity}, {nkv, batch * q, capacity}}, ge::DT_INT32, ge::FORMAT_ND};
                        inputs[4] = {{{nkv, batch * q}, {nkv, batch * q}}, ge::DT_INT32, ge::FORMAT_ND};
                        inputs[7] = {
                            {{batch, q, nq, d / 64, 2}, {batch, q, nq, d / 64, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND};
                        inputs[8] = {{{batch, kv, nkv, d / 64, 2}, {batch, kv, nkv, d / 64, 2}},
                                     ge::DT_FLOAT8_E8M0,
                                     ge::FORMAT_ND};
                        inputs[9] = {{{batch, nkv, d, 2}, {batch, nkv, d, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND};
                        for (int index : {11, 12, 15})
                            inputs[index] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
                        inputs[13] = {{{batch}, {batch}}, ge::DT_INT32, ge::FORMAT_ND};
                        inputs[14] = inputs[13];
                        auto outputs = MakeOutputs(dtype, false, d, nq);
                        outputs[0] = inputs[0];
                        outputs[0].dtype_ = dtype;
                        GenericBlockSparseAttentionCompileInfo compileInfo;
                        gert::TilingContextPara ctx(
                            "GenericBlockSparseAttention", inputs, outputs,
                            MakeAttrs(mask, 6, 1, 0, 0.0f, -1, -1, 4, "BSND", "BSND", 0, false, {1, y}), &compileInfo,
                            "Ascend950", 28, 262144);
                        TilingInfo info;
                        ASSERT_TRUE(ExecuteTiling(ctx, info));
                        const uint64_t keyBase =
                            mask == 1 && kv >= 1024 && kv <= 8192 ? 9250002900000022ULL : 9250002900000020ULL;
                        EXPECT_EQ(info.tilingKey, keyBase + (dtype == ge::DT_FLOAT16 ? 0ULL : 1ULL));
                        EXPECT_LE(Mxfp8Tiling::PipelineUbEnd(d, 512), Mxfp8Tiling::UB_CAPACITY);
                    }
                }
            }
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_rejects_removed_d128_specialization)
{
    // These BF16 causal shapes previously selected the D128 specialization.
    // Reject both dense layouts, including zero batch and the FP16 variants.
    constexpr int64_t d = 128, q = 129, kv = 4608, nq = 64, nkv = 4;
    for (int64_t batch : {0, 1, 2, 8}) {
        for (const std::string layout : {"BSND", "BNSD"}) {
            for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16}) {
                for (int64_t precision : {0, 1}) {
                    auto inputs = MakeMxfp8Inputs(d);
                    const bool headMajor = layout == "BNSD";
                    const int64_t qAxis1 = headMajor ? nq : q, qAxis2 = headMajor ? q : nq;
                    const int64_t kvAxis1 = headMajor ? nkv : kv, kvAxis2 = headMajor ? kv : nkv;
                    inputs[0] = {
                        {{batch, qAxis1, qAxis2, d}, {batch, qAxis1, qAxis2, d}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND};
                    inputs[1] = {{{batch, kvAxis1, kvAxis2, d}, {batch, kvAxis1, kvAxis2, d}},
                                 ge::DT_FLOAT8_E4M3FN,
                                 ge::FORMAT_ND};
                    inputs[2] = inputs[1];
                    inputs[3] = {{{nkv, batch * q, 514}, {nkv, batch * q, 514}}, ge::DT_INT32, ge::FORMAT_ND};
                    inputs[4] = {{{nkv, batch * q}, {nkv, batch * q}}, ge::DT_INT32, ge::FORMAT_ND};
                    inputs[7] = {{{batch, qAxis1, qAxis2, d / 64, 2}, {batch, qAxis1, qAxis2, d / 64, 2}},
                                 ge::DT_FLOAT8_E8M0,
                                 ge::FORMAT_ND};
                    inputs[8] = {{{batch, kvAxis1, kvAxis2, d / 64, 2}, {batch, kvAxis1, kvAxis2, d / 64, 2}},
                                 ge::DT_FLOAT8_E8M0,
                                 ge::FORMAT_ND};
                    inputs[9] = {{{batch, nkv, d, 2}, {batch, nkv, d, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND};
                    for (int index : {11, 12, 15}) {
                        inputs[index] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
                    }
                    inputs[13] = {{{batch}, {batch}}, ge::DT_INT32, ge::FORMAT_ND};
                    inputs[14] = inputs[13];
                    auto outputs = MakeOutputs(dtype, false, d, nq);
                    outputs[0] = inputs[0];
                    outputs[0].dtype_ = dtype;
                    GenericBlockSparseAttentionCompileInfo compileInfo;
                    gert::TilingContextPara ctx(
                        "GenericBlockSparseAttention", inputs, outputs,
                        MakeAttrs(1, 6, precision, 0, 0.0f, -1, -1, 4, layout, layout, 0, false, {1, 128}),
                        &compileInfo, "Ascend950", 28, 262144);
                    ExecuteTestCase(ctx, ge::GRAPH_FAILED);
                }
            }
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_rejects_invalid_packed_shapes)
{
    for (int index : {7, 8, 9}) {
        auto inputs = MakeMxfp8Inputs();
        if (index == 7) {
            inputs[index] = {{{kT, kN1, kD / 64, 1}, {kT, kN1, kD / 64, 1}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND};
        } else if (index == 8) {
            inputs[index] = {{{kMaxBlocks, kBlockSize, kN2, kD / 64, 1}, {kMaxBlocks, kBlockSize, kN2, kD / 64, 1}},
                             ge::DT_FLOAT8_E8M0,
                             ge::FORMAT_ND};
        } else {
            // Unrepeated per-channel scales do not have the required packed shape.
            inputs[index] = {{{kN2, kD}, {kN2, kD}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND};
        }
        GenericBlockSparseAttentionCompileInfo compileInfo;
        gert::TilingContextPara ctx("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                    MakeAttrs(1, 6, 1, 0), &compileInfo, "Ascend950", 56, 262144);
        ExecuteTestCase(ctx, ge::GRAPH_FAILED);
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_requires_fp32_scalar_p_scale)
{
    for (int invalidCase : {0, 1, 2}) {
        auto inputs = MakeMxfp8Inputs();
        inputs[10] = invalidCase == 0 ?
                         gert::TilingContextPara::TensorDescription{{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND} :
                     invalidCase == 1 ?
                         gert::TilingContextPara::TensorDescription{{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND} :
                         gert::TilingContextPara::TensorDescription{{{1, 1}, {1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND};
        GenericBlockSparseAttentionCompileInfo compileInfo;
        gert::TilingContextPara ctx("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                    MakeAttrs(1, 6, 1, 0), &compileInfo, "Ascend950", 56, 262144);
        ExecuteTestCase(ctx, ge::GRAPH_FAILED);
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_rejects_unsupported_configuration)
{
    for (int invalidCase = 0; invalidCase < 4; ++invalidCase) {
        const int64_t headDim = invalidCase == 0 ? 257 : 128;
        auto inputs = invalidCase == 1 ? MakeInputs(ge::DT_FLOAT16) : MakeMxfp8Inputs(headDim);
        GenericBlockSparseAttentionCompileInfo compileInfo;
        gert::TilingContextPara ctx("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false, headDim),
                                    MakeAttrs(1, 6, 1, 0, 0.0f, -1, -1, 4, "TND", "PA_BBND", invalidCase == 2 ? 1 : 0),
                                    &compileInfo, invalidCase == 3 ? "Ascend910B" : "Ascend950", 56, 262144);
        ExecuteTestCase(ctx, ge::GRAPH_FAILED);
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_rejects_removed_generic_dispatch)
{
    for (auto outputType : {ge::DT_FLOAT16, ge::DT_BF16}) {
        for (int64_t precision : {0, 1}) {
            for (int64_t lse : {0, 1}) {
                GenericBlockSparseAttentionCompileInfo compileInfo;
                gert::TilingContextPara ctx("GenericBlockSparseAttention", MakeMxfp8Inputs(192),
                                            MakeOutputs(outputType, lse != 0, 192), MakeAttrs(0, 6, precision, lse),
                                            &compileInfo, "Ascend950", 56, 262144);
                ExecuteTestCase(ctx, ge::GRAPH_FAILED);
            }
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_tnd_paged_grouped_shape_boundary)
{
    for (int64_t headDim : {1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 192, 255, 256}) {
        for (int64_t blockY : {1, 4, 8, 17, 127, 128}) {
            auto inputs = MakeMxfp8Inputs(headDim);
            inputs[3] = {{{kN2, kTotalQBlocks, 512}, {kN2, kTotalQBlocks, 512}}, ge::DT_INT32, ge::FORMAT_ND};
            GenericBlockSparseAttentionCompileInfo compileInfo;
            gert::TilingContextPara ctx("GenericBlockSparseAttention", inputs,
                                        MakeOutputs(ge::DT_FLOAT16, false, headDim),
                                        MakeAttrs(1, 6, 1, 0, 0.0f, -1, -1, 4, "TND", "PA_BBND", 0, false, {1, blockY}),
                                        &compileInfo, "Ascend950", 56, 262144);
            const bool grouped = (headDim == 192 || headDim == 256) && (blockY == 4 || blockY == 8);
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
                if (lse) {
                    ExecuteTestCase(ctx, ge::GRAPH_FAILED);
                    continue;
                }
                TilingInfo info;
                ASSERT_TRUE(ExecuteTiling(ctx, info));
                const uint64_t base = layout == "TND"     ? 9250002900000050ULL :
                                      layout == "PA_BBND" ? 9250002900000052ULL :
                                                            9250002900000054ULL;
                EXPECT_EQ(info.tilingKey, base + (outputType == ge::DT_FLOAT16 ? 0U : 1U));
            }
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_packed_scale_cache_boundary)
{
    // Crossing the old all-head cache capacity must retain packed execution.
    // Zero KV capacity is rejected after removal of the generic dispatch.
    struct Geometry {
        int64_t d;
        int64_t nq;
        int64_t nkv;
        int64_t y;
        int64_t maxKv;
    };
    for (const auto shape : {Geometry{192, 64, 4, 8, 5461}, Geometry{256, 24, 2, 4, 8192}}) {
        std::vector<int64_t> packedWorkspace[2];
        size_t packedBlockNum[2] = {};
        for (int64_t kv : {shape.maxKv, shape.maxKv + 1, int64_t(0)}) {
            for (auto outputType : {ge::DT_FLOAT16, ge::DT_BF16}) {
                const bool packed = kv != 0;
                const size_t dtypeIndex = outputType == ge::DT_FLOAT16 ? 0U : 1U;
                constexpr uint32_t packedTile = 512U;
                auto inputs = MakeMxfp8Inputs(shape.d);
                inputs[0] = {
                    {{1, kS1, shape.nq, shape.d}, {1, kS1, shape.nq, shape.d}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND};
                inputs[1] = {
                    {{1, kv, shape.nkv, shape.d}, {1, kv, shape.nkv, shape.d}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND};
                inputs[2] = inputs[1];
                inputs[3] = {{{shape.nkv, kS1, 513}, {shape.nkv, kS1, 513}}, ge::DT_INT32, ge::FORMAT_ND};
                inputs[4] = {{{shape.nkv, kS1}, {shape.nkv, kS1}}, ge::DT_INT32, ge::FORMAT_ND};
                inputs[7] = {{{1, kS1, shape.nq, shape.d / 64, 2}, {1, kS1, shape.nq, shape.d / 64, 2}},
                             ge::DT_FLOAT8_E8M0,
                             ge::FORMAT_ND};
                inputs[8] = {{{1, kv, shape.nkv, shape.d / 64, 2}, {1, kv, shape.nkv, shape.d / 64, 2}},
                             ge::DT_FLOAT8_E8M0,
                             ge::FORMAT_ND};
                inputs[9] = {
                    {{1, shape.nkv, shape.d, 2}, {1, shape.nkv, shape.d, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND};
                for (int index : {11, 12, 15}) {
                    inputs[index] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
                }
                inputs[13] = {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND};
                inputs[14] = inputs[13];
                auto outputs = MakeOutputs(outputType, false, shape.d, shape.nq);
                outputs[0] = inputs[0];
                outputs[0].dtype_ = outputType;
                GenericBlockSparseAttentionCompileInfo compileInfo;
                gert::TilingContextPara ctx(
                    "GenericBlockSparseAttention", inputs, outputs,
                    MakeAttrs(1, 6, 1, 0, 0.0f, -1, -1, 4, "BSND", "BSND", 0, false, {1, shape.y}), &compileInfo,
                    "Ascend950", 56, 262144);
                if (!packed) {
                    ExecuteTestCase(ctx, ge::GRAPH_FAILED);
                    continue;
                }
                TilingInfo info;
                ASSERT_TRUE(ExecuteTiling(ctx, info));
                const uint64_t expectedBase = 9250002900000020ULL;
                EXPECT_EQ(info.tilingKey, expectedBase + (outputType == ge::DT_FLOAT16 ? 0U : 1U));
                constexpr size_t l1TilingOffset = 184;
                uint32_t l1[12] = {};
                ASSERT_GE(info.tilingDataSize, l1TilingOffset + sizeof(l1));
                std::memcpy(l1, info.tilingData.get() + l1TilingOffset, sizeof(l1));
                EXPECT_EQ(l1[1], 128U);
                EXPECT_EQ(l1[2], 256U);
                EXPECT_EQ(l1[3], 256U);
                EXPECT_EQ(l1[5], 256U);
                ASSERT_EQ(info.workspaceSizes.size(), 1U);
            }
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_pipeline_fixed_tile_and_workspace)
{
    // Logical QK/softmax/PV extent is independent of the physical K L1 group.
    // Both output dtypes stream K through three 128-token slots.
    struct Geometry {
        int64_t d;
        int64_t nq;
        int64_t nkv;
        int64_t y;
        int64_t max512Kv;
    };
    for (const auto shape : {Geometry{192, 64, 4, 8, 21589}, Geometry{256, 24, 2, 4, 7264},
                             Geometry{192, 64, 4, 4, 21418}, Geometry{256, 24, 2, 8, 7328}}) {
        for (int64_t kv : {int64_t(4096), int64_t(4320), int64_t(4608), shape.max512Kv, shape.max512Kv + 1}) {
            int64_t fp16Workspace = 0;
            size_t fp16BlockNum = 0;
            for (auto outputType : {ge::DT_FLOAT16, ge::DT_BF16}) {
                constexpr uint32_t expectedTile = 512U;
                auto inputs = MakeMxfp8Inputs(shape.d);
                inputs[0] = {
                    {{1, kS1, shape.nq, shape.d}, {1, kS1, shape.nq, shape.d}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND};
                inputs[1] = {
                    {{1, kv, shape.nkv, shape.d}, {1, kv, shape.nkv, shape.d}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND};
                inputs[2] = inputs[1];
                inputs[3] = {{{shape.nkv, kS1, 513}, {shape.nkv, kS1, 513}}, ge::DT_INT32, ge::FORMAT_ND};
                inputs[4] = {{{shape.nkv, kS1}, {shape.nkv, kS1}}, ge::DT_INT32, ge::FORMAT_ND};
                inputs[7] = {{{1, kS1, shape.nq, shape.d / 64, 2}, {1, kS1, shape.nq, shape.d / 64, 2}},
                             ge::DT_FLOAT8_E8M0,
                             ge::FORMAT_ND};
                inputs[8] = {{{1, kv, shape.nkv, shape.d / 64, 2}, {1, kv, shape.nkv, shape.d / 64, 2}},
                             ge::DT_FLOAT8_E8M0,
                             ge::FORMAT_ND};
                inputs[9] = {
                    {{1, shape.nkv, shape.d, 2}, {1, shape.nkv, shape.d, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND};
                for (int index : {11, 12, 15}) {
                    inputs[index] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
                }
                inputs[13] = {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND};
                inputs[14] = inputs[13];
                auto outputs = MakeOutputs(outputType, false, shape.d, shape.nq);
                outputs[0] = inputs[0];
                outputs[0].dtype_ = outputType;
                GenericBlockSparseAttentionCompileInfo compileInfo;
                gert::TilingContextPara ctx(
                    "GenericBlockSparseAttention", inputs, outputs,
                    MakeAttrs(1, 6, 1, 0, 0.0f, -1, -1, 4, "BSND", "BSND", 0, false, {1, shape.y}), &compileInfo,
                    "Ascend950", 56, 262144);
                TilingInfo info;
                ASSERT_TRUE(ExecuteTiling(ctx, info));
                const bool perHead =
                    shape.d == 192 && (kv > 5461 || (outputType == ge::DT_BF16 && kv > (shape.y == 8 ? 5397 : 5354)));
                const uint64_t expectedBase = perHead ? 9250002900000020ULL : 9250002900000020ULL;
                EXPECT_EQ(info.tilingKey, expectedBase + (outputType == ge::DT_FLOAT16 ? 0U : 1U));
                constexpr size_t l1TilingOffset = 184;
                uint32_t l1[12] = {};
                ASSERT_GE(info.tilingDataSize, l1TilingOffset + sizeof(l1));
                std::memcpy(l1, info.tilingData.get() + l1TilingOffset, sizeof(l1));
                EXPECT_EQ(l1[0], 16U);
                EXPECT_EQ(l1[1], 128U);
                EXPECT_EQ(l1[2], 256U);
                EXPECT_EQ(l1[3], 256U);
                EXPECT_EQ(l1[5], 256U);
                EXPECT_EQ(l1[6], expectedTile);
                EXPECT_EQ(l1[7], expectedTile);
                EXPECT_EQ(l1[9], 3U);
                EXPECT_EQ(l1[10], 2U);
                ASSERT_EQ(info.workspaceSizes.size(), 1U);
                ASSERT_GT(info.blockNum, 0U);
                if (outputType == ge::DT_FLOAT16) {
                    fp16Workspace = info.workspaceSizes[0];
                    fp16BlockNum = info.blockNum;
                } else {
                    ASSERT_EQ(info.blockNum, fp16BlockNum);
                    // Four descriptor groups per tile. A repeated partial
                    // block can generate one record per token, so reserve
                    // 128 records in each group, independent of index capacity.
                    EXPECT_EQ(info.workspaceSizes[0], fp16Workspace);
                    const uint64_t dataL1 = 16U * 256U + 3U * 128U * 256U + 3U * 16U * 512U + 2U * 512U * 256U;
                    EXPECT_LT(dataL1, 512U * 1024U - 16U * 1024U - 3U * 4096U);
                    EXPECT_LE(Mxfp8Tiling::PipelineUbEnd(shape.d, 512U), Mxfp8Tiling::UB_CAPACITY);
                }
            }
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_pipeline_bounded_scale_reservation)
{
    using namespace Mxfp8Tiling;
    for (uint64_t dim : {128U, 192U, 256U}) {
        for (uint64_t heads : {1U, 2U, 4U, 13U, 64U, 100U}) {
            EXPECT_TRUE(StreamFits(heads, dim, 256U));
        }
        EXPECT_LE(PipelineUbEnd(dim, 512U), UB_CAPACITY);
        EXPECT_EQ(StreamBytes(dim, 256U), 8192U);
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_fp32_scores_precision_compatibility)
{
    // Precision 0 is the FP32 interface; legacy value 1 must select the same
    // kernel and workspace for both dense layouts and final output types.
    for (const std::string layout : {"BSND", "BNSD"}) {
        for (auto outputType : {ge::DT_FLOAT16, ge::DT_BF16}) {
            TilingInfo previous;
            for (int64_t precision : {0, 1, 2}) {
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
                if (precision == 2) {
                    ExecuteTestCase(ctx, ge::GRAPH_FAILED);
                    continue;
                }
                TilingInfo info;
                ASSERT_TRUE(ExecuteTiling(ctx, info));
                EXPECT_EQ(info.tilingKey, (headMajor ? 9250002900000032ULL : 9250002900000020ULL) +
                                              (outputType == ge::DT_BF16 ? 1U : 0U));
                if (precision == 0) {
                    previous.tilingKey = info.tilingKey;
                    previous.workspaceSizes = info.workspaceSizes;
                } else {
                    EXPECT_EQ(info.tilingKey, previous.tilingKey);
                    EXPECT_EQ(info.workspaceSizes, previous.workspaceSizes);
                }
            }
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_grouped_raw_ring_capacity)
{
    using namespace Mxfp8Tiling;
    // Both layouts and all KV heads share the same current-head Raw capacity.
    EXPECT_EQ(PipelineRawStages(256U, 512U), 3U);
    EXPECT_EQ(PipelineUbEnd(256U, 512U), 170496U);
    EXPECT_EQ(PipelineRawStages(192U, 512U), 3U);
    EXPECT_EQ(PipelineUbEnd(192U, 512U), 170496U);
    EXPECT_GT(PipelineUbEnd(256U, 256U), UB_CAPACITY);
}

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_head_group_tiers)
{
    // Independent tier expectations: cover both sides of every M/KV boundary,
    // odd split rows, both sparse block sizes/output types, and rejection at 129.
    constexpr int64_t nqKv = 2;
    constexpr int64_t kv = 4099;
    for (int64_t d : {192, 256}) {
        for (int64_t group : {1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129}) {
            for (int64_t y : {4, 8}) {
                for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16}) {
                    const int64_t nq = nqKv * group;
                    auto inputs = MakeMxfp8Inputs(d);
                    inputs[0] = {{{1, kS1, nq, d}, {1, kS1, nq, d}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND};
                    inputs[1] = {{{1, kv, nqKv, d}, {1, kv, nqKv, d}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND};
                    inputs[2] = inputs[1];
                    inputs[3] = {{{nqKv, kS1, 513}, {nqKv, kS1, 513}}, ge::DT_INT32, ge::FORMAT_ND};
                    inputs[4] = {{{nqKv, kS1}, {nqKv, kS1}}, ge::DT_INT32, ge::FORMAT_ND};
                    inputs[7] = {{{1, kS1, nq, d / 64, 2}, {1, kS1, nq, d / 64, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND};
                    inputs[8] = {
                        {{1, kv, nqKv, d / 64, 2}, {1, kv, nqKv, d / 64, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND};
                    inputs[9] = {{{1, nqKv, d, 2}, {1, nqKv, d, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND};
                    for (int index : {11, 12, 15}) {
                        inputs[index] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
                    }
                    inputs[13] = {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND};
                    inputs[14] = inputs[13];
                    auto outputs = MakeOutputs(dtype, false, d, nq);
                    outputs[0] = inputs[0];
                    outputs[0].dtype_ = dtype;
                    GenericBlockSparseAttentionCompileInfo compileInfo;
                    gert::TilingContextPara ctx(
                        "GenericBlockSparseAttention", inputs, outputs,
                        MakeAttrs(0, 6, 0, 0, 0.0f, -1, -1, 4, "BSND", "BSND", 0, false, {1, y}), &compileInfo,
                        "Ascend950", 56, 262144);
                    if (group > 128) {
                        ExecuteTestCase(ctx, ge::GRAPH_FAILED);
                        continue;
                    }
                    TilingInfo info;
                    ASSERT_TRUE(ExecuteTiling(ctx, info)) << "D=" << d << " G=" << group;
                    // The original input width is not padded in the runtime shape.
                    EXPECT_EQ(ReadTilingU32(info, 3U * sizeof(uint32_t)), d);
                    const uint32_t m = group <= 16 ? 16U : group <= 32 ? 32U : group <= 64 ? 64U : 128U;
                    const uint32_t tokens = group <= 32 ? 512U : group <= 64 ? 256U : 128U;
                    constexpr size_t l1Offset = 184;
                    EXPECT_EQ(ReadTilingU32(info, l1Offset), m);
                    EXPECT_EQ(ReadTilingU32(info, l1Offset + 4U), 128U);
                    // L1 reserves the D256 capacity for both supported D values.
                    EXPECT_EQ(ReadTilingU32(info, l1Offset + 2U * 4U), 256U);
                    EXPECT_EQ(ReadTilingU32(info, l1Offset + 3U * 4U), 256U);
                    EXPECT_EQ(ReadTilingU32(info, l1Offset + 4U * 4U), m);
                    EXPECT_EQ(ReadTilingU32(info, l1Offset + 5U * 4U), 256U);
                    EXPECT_EQ(ReadTilingU32(info, l1Offset + 6U * 4U), tokens);
                    EXPECT_EQ(ReadTilingU32(info, l1Offset + 7U * 4U), tokens);
                    EXPECT_EQ(ReadTilingU32(info, l1Offset + 9U * 4U), 3U);
                    EXPECT_EQ(ReadTilingU32(info, l1Offset + 10U * 4U), 2U);
                    EXPECT_LT(Mxfp8Tiling::PipelineL1End(d, m, tokens), Mxfp8Tiling::L1_DATA_CAPACITY);
                    EXPECT_LE(Mxfp8Tiling::PipelineUbEnd(d, tokens, m), Mxfp8Tiling::UB_CAPACITY);
                }
            }
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_large_head_group_buffer_capacity)
{
    using namespace Mxfp8Tiling;
    // Exact footprints include Raw, NZ, statistics and both descriptor scratch records.
    EXPECT_EQ(PipelineUbEnd(256U, 512U, 32U), 170496U);
    EXPECT_EQ(PipelineUbEnd(256U, 256U, 64U), 203136U);
    EXPECT_EQ(PipelineUbEnd(256U, 128U, 128U), 235904U);
    EXPECT_EQ(PipelineUbEnd(192U, 512U, 32U), 170496U);
    EXPECT_EQ(PipelineUbEnd(192U, 256U, 64U), 203136U);
    EXPECT_EQ(PipelineUbEnd(192U, 128U, 128U), 235904U);
    EXPECT_EQ(PvUbStages(128U), 1U);
    EXPECT_EQ(PvUbStages(64U), 2U);
    EXPECT_EQ(DescriptorTileWords(128U), 528U); // includes AIV1's empty notification
    EXPECT_EQ(DescriptorTileWords(256U), 528U);
    EXPECT_EQ(DescriptorTileWords(512U), 1056U);
    EXPECT_GT(PipelineUbEnd(256U, 512U, 128U), UB_CAPACITY);
    EXPECT_GT(PipelineUbEnd(256U, 256U, 128U), UB_CAPACITY);
}

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_grouped_indices_independent_of_ub_cache)
{
    using namespace Mxfp8Tiling;
    EXPECT_EQ(PipelineUbEnd(256U, 512U), 170496U);
    EXPECT_EQ(PipelineAuxBytes(256U), 2048U);
    // D192 uses the same D256 buffer capacity while retaining actual-D strides.
    EXPECT_EQ(PipelineAuxBytes(192U), 2048U);
}

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_pipeline_workspace_independent_of_index_capacity)
{
    for (int64_t batch : {1, 2, 3, 8}) {
        for (int64_t d : {192, 256}) {
            const int64_t q = 129, nq = d == 192 ? 64 : 24, nkv = d == 192 ? 4 : 2;
            const int64_t y = d == 192 ? 8 : 4;
            for (int64_t kv : {21846, 32600, 32707, 32708}) {
                int64_t workspace = -1;
                for (int64_t capacity : {512, 513, 514, 1024, 8192, 8193}) {
                    auto inputs = MakeMxfp8Inputs(d);
                    inputs[0] = {{{batch, q, nq, d}, {batch, q, nq, d}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND};
                    inputs[1] = {{{batch, kv, nkv, d}, {batch, kv, nkv, d}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND};
                    inputs[2] = inputs[1];
                    inputs[3] = {{{nkv, batch * q, capacity}, {nkv, batch * q, capacity}}, ge::DT_INT32, ge::FORMAT_ND};
                    inputs[4] = {{{nkv, batch * q}, {nkv, batch * q}}, ge::DT_INT32, ge::FORMAT_ND};
                    inputs[7] = {
                        {{batch, q, nq, d / 64, 2}, {batch, q, nq, d / 64, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND};
                    inputs[8] = {
                        {{batch, kv, nkv, d / 64, 2}, {batch, kv, nkv, d / 64, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND};
                    inputs[9] = {{{batch, nkv, d, 2}, {batch, nkv, d, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND};
                    for (int index : {11, 12, 15})
                        inputs[index] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
                    inputs[13] = {{{batch}, {batch}}, ge::DT_INT32, ge::FORMAT_ND};
                    inputs[14] = inputs[13];
                    auto outputs = MakeOutputs(ge::DT_BF16, false, d, nq);
                    outputs[0] = inputs[0];
                    outputs[0].dtype_ = ge::DT_BF16;
                    GenericBlockSparseAttentionCompileInfo compileInfo;
                    gert::TilingContextPara ctx(
                        "GenericBlockSparseAttention", inputs, outputs,
                        MakeAttrs(1, 6, 1, 0, 0.0f, -1, -1, 4, "BSND", "BSND", 0, false, {1, y}), &compileInfo,
                        "Ascend950", 28, 262144);
                    TilingInfo info;
                    ASSERT_TRUE(ExecuteTiling(ctx, info));
                    ASSERT_EQ(info.tilingKey, 9250002900000021ULL);
                    ASSERT_EQ(info.workspaceSizes.size(), 1U);
                    if (workspace < 0)
                        workspace = info.workspaceSizes[0];
                    EXPECT_EQ(info.workspaceSizes[0], workspace);
                }
            }
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_pipeline_preserves_old_capacity_boundaries)
{
    enum class CacheMode {
        Dense,
        PerHead,
        Stream,
        Hybrid
    };
    struct Shape {
        int64_t d;
        int64_t nq;
        int64_t nkv;
        int64_t y;
        int64_t kv;
        bool packed;
        uint32_t tile;
        CacheMode mode = CacheMode::PerHead;
    };
    // D128 specialization was removed; keep those shapes as rejection checks.
    for (const auto s : {Shape{192, 64, 4, 8, 5397, true, 512, CacheMode::Dense},
                         Shape{192, 64, 4, 8, 5398, true, 512},
                         Shape{192, 64, 4, 8, 5462, true, 512},
                         Shape{192, 64, 4, 8, 21845, true, 256},
                         Shape{192, 64, 4, 8, 21846, true, 384, CacheMode::Hybrid},
                         Shape{256, 24, 2, 4, 8193, true, 256},
                         Shape{256, 24, 2, 4, 16384, true, 256},
                         Shape{256, 24, 2, 4, 16385, true, 256, CacheMode::Hybrid},
                         Shape{256, 24, 2, 4, 21846, true, 256, CacheMode::Hybrid},
                         Shape{256, 24, 2, 4, 32768, true, 256, CacheMode::Hybrid},
                         Shape{256, 24, 2, 4, 32769, true, 256, CacheMode::Stream},
                         Shape{192, 64, 4, 8, 42496, true, 384, CacheMode::Hybrid},
                         Shape{192, 64, 4, 8, 42497, true, 512, CacheMode::Stream},
                         Shape{192, 48, 3, 8, 8193, true, 512, CacheMode::Stream},
                         Shape{192, 64, 4, 8, 65537, true, 512, CacheMode::Stream},
                         Shape{256, 24, 2, 4, 65537, true, 256, CacheMode::Stream},
                         Shape{128, 64, 4, 128, 4096, false, 512, CacheMode::Dense},
                         Shape{128, 64, 4, 128, 8192, false, 512, CacheMode::Dense},
                         Shape{128, 64, 4, 128, 8193, false, 512},
                         Shape{128, 64, 4, 128, 32768, false, 512},
                         Shape{128, 64, 4, 128, 32769, false, 512, CacheMode::Stream},
                         Shape{128, 64, 4, 128, 65536, false, 512, CacheMode::Stream},
                         Shape{128, 64, 4, 128, 65537, false, 512, CacheMode::Stream},
                         Shape{128, 48, 4, 128, 4096, false, 128},
                         Shape{128, 32, 2, 128, 4096, false, 128},
                         Shape{192, 1600, 100, 8, 1000, true, 512},
                         Shape{192, 64, 4, 8, 5592406, false, 128}}) {
        auto inputs = MakeMxfp8Inputs(s.d);
        inputs[0] = {{{2, 129, s.nq, s.d}, {2, 129, s.nq, s.d}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND};
        inputs[1] = {{{2, s.kv, s.nkv, s.d}, {2, s.kv, s.nkv, s.d}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND};
        inputs[2] = inputs[1];
        inputs[3] = {{{s.nkv, 258, 513}, {s.nkv, 258, 513}}, ge::DT_INT32, ge::FORMAT_ND};
        inputs[4] = {{{s.nkv, 258}, {s.nkv, 258}}, ge::DT_INT32, ge::FORMAT_ND};
        inputs[7] = {{{2, 129, s.nq, s.d / 64, 2}, {2, 129, s.nq, s.d / 64, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND};
        inputs[8] = {{{2, s.kv, s.nkv, s.d / 64, 2}, {2, s.kv, s.nkv, s.d / 64, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND};
        inputs[9] = {{{2, s.nkv, s.d, 2}, {2, s.nkv, s.d, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND};
        for (int index : {11, 12, 15})
            inputs[index] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
        inputs[13] = {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND};
        inputs[14] = inputs[13];
        auto outputs = MakeOutputs(ge::DT_BF16, false, s.d, s.nq);
        outputs[0] = inputs[0];
        outputs[0].dtype_ = ge::DT_BF16;
        GenericBlockSparseAttentionCompileInfo compileInfo;
        gert::TilingContextPara ctx("GenericBlockSparseAttention", inputs, outputs,
                                    MakeAttrs(1, 6, 1, 0, 0.0f, -1, -1, 4, "BSND", "BSND", 0, false, {1, s.y}),
                                    &compileInfo, "Ascend950", 56, 262144);
        if (!s.packed) {
            ExecuteTestCase(ctx, ge::GRAPH_FAILED);
            continue;
        }
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(ctx, info));
        // The former cache-mode boundaries now share one packed pipeline key.
        constexpr uint64_t packedKey = 9250002900000021ULL;
        EXPECT_EQ(info.tilingKey, packedKey);
        uint32_t l1[12] = {};
        ASSERT_GE(info.tilingDataSize, 184U + sizeof(l1));
        std::memcpy(l1, info.tilingData.get() + 184U, sizeof(l1));
        EXPECT_EQ(l1[1], 128U);
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_rejects_generic_capacity_inputs)
{
    std::vector<int64_t> referenceWorkspace;
    for (int64_t capacity : {257LL, 512LL, 2147483647LL, 2147483648LL}) {
        auto inputs = MakeMxfp8Inputs(65);
        inputs[3] = {{{kN2, kTotalQBlocks, capacity}, {kN2, kTotalQBlocks, capacity}}, ge::DT_INT32, ge::FORMAT_ND};
        GenericBlockSparseAttentionCompileInfo compileInfo;
        gert::TilingContextPara ctx("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false, 65),
                                    MakeAttrs(1, 6, 0, 0), &compileInfo, "Ascend950", 56, 262144);
        if (capacity > 2147483647LL) {
            ExecuteTestCase(ctx, ge::GRAPH_FAILED);
            continue;
        }
        ExecuteTestCase(ctx, ge::GRAPH_FAILED);
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_rejects_nonpacked_layout_precision_lse)
{
    constexpr int64_t d = 65;
    for (const std::string qLayout : {"TND", "BSND", "BNSD"}) {
        for (int kvLayout = 0; kvLayout < 3; ++kvLayout) {
            const bool paged = kvLayout != 0;
            const bool bnbd = kvLayout == 2;
            auto inputs = MakeMxfp8Inputs(d);
            if (qLayout == "TND") {
                inputs[0] = {{{6, 6, d}, {6, 6, d}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND};
                inputs[1] = {{{514, 3, d}, {514, 3, d}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND};
            } else if (qLayout == "BSND") {
                inputs[0] = {{{2, 3, 6, d}, {2, 3, 6, d}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND};
                inputs[1] = {{{2, 257, 3, d}, {2, 257, 3, d}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND};
            } else {
                inputs[0] = {{{2, 6, 3, d}, {2, 6, 3, d}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND};
                inputs[1] = {{{2, 3, 257, d}, {2, 3, 257, d}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND};
            }
            if (paged) {
                inputs[1] = bnbd ?
                                gert::TilingContextPara::TensorDescription{
                                    {{8, 3, 128, d}, {8, 3, 128, d}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND} :
                                gert::TilingContextPara::TensorDescription{
                                    {{8, 128, 3, d}, {8, 128, 3, d}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND};
            }
            inputs[2] = inputs[1];
            inputs[3] = {{{3, 6, 512}, {3, 6, 512}}, ge::DT_INT32, ge::FORMAT_ND};
            inputs[4] = {{{3, 6}, {3, 6}}, ge::DT_INT32, ge::FORMAT_ND};
            for (int index : {0, 1}) {
                auto scaleShape = inputs[index].shape_.GetStorageShape();
                const size_t rank = scaleShape.GetDimNum();
                scaleShape.SetDimNum(rank + 1U);
                scaleShape.SetDim(rank - 1U, (d + 63) / 64);
                scaleShape.SetDim(rank, 2);
                inputs[7 + index].shape_.MutableOriginShape() = scaleShape;
                inputs[7 + index].shape_.MutableStorageShape() = scaleShape;
                inputs[7 + index].dtype_ = ge::DT_FLOAT8_E8M0;
            }
            inputs[9] = !paged ?
                            gert::TilingContextPara::TensorDescription{
                                {{2, 3, d, 2}, {2, 3, d, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND} :
                            (bnbd ?
                                 gert::TilingContextPara::TensorDescription{
                                     {{8, 3, 2, d, 2}, {8, 3, 2, d, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND} :
                                 gert::TilingContextPara::TensorDescription{
                                     {{8, 2, 3, d, 2}, {8, 2, 3, d, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND});
            for (int index = 11; index < 16; ++index) {
                inputs[index] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
            }
            if (qLayout == "TND") {
                inputs[11] = {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND};
                if (!paged) {
                    inputs[12] = inputs[11];
                }
            } else {
                inputs[13] = {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND};
            }
            inputs[14] = {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND};
            if (paged) {
                inputs[15] = {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND};
            }
            auto outputs = MakeOutputs(ge::DT_FLOAT16, true, d);
            outputs[0] = inputs[0];
            outputs[0].dtype_ = ge::DT_FLOAT16;
            outputs[1] =
                qLayout == "TND" ?
                    gert::TilingContextPara::TensorDescription{{{6, 6, 1}, {6, 6, 1}}, ge::DT_FLOAT, ge::FORMAT_ND} :
                    gert::TilingContextPara::TensorDescription{
                        {{2, 6, 3, 1}, {2, 6, 3, 1}}, ge::DT_FLOAT, ge::FORMAT_ND};
            const std::string layoutKv = paged ? (bnbd ? "PA_BNBD" : "PA_BBND") : qLayout;
            for (bool invalidVScale : {false, true}) {
                if (invalidVScale) {
                    inputs[9] = {{{2, d, 2}, {2, d, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND};
                }
                GenericBlockSparseAttentionCompileInfo compileInfo;
                gert::TilingContextPara ctx("GenericBlockSparseAttention", inputs, outputs,
                                            MakeAttrs(0, 6, 0, 1, 0.0f, -1, -1, 4, qLayout, layoutKv, 0, false, {1, 4}),
                                            &compileInfo, "Ascend950", 56, 262144);
                ExecuteTestCase(ctx, ge::GRAPH_FAILED);
            }
        }
    }
}

std::vector<gert::TilingContextPara::TensorDescription> MakeBnbdInputs(ge::DataType qkvDtype)
{
    auto inputs = MakeInputs(qkvDtype);
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

TEST_F(generic_block_sparse_attention_tiling_ut, fp8_rejects_softmax_precision_0_950)
{
    // FP8 full-quant only supports softmaxPrecision=1.
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT8_E4M3FN),
                                              MakeOutputs(ge::DT_FLOAT16, false), MakeAttrs(1, 5, 0, 0), &compileInfo,
                                              "Ascend950", 56, 262144);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
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
    // quantType only accepts 0 (none) or 5 (full-quant).
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

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_pipeline_and_nonquant_dispatch)
{
    for (int64_t quant : {0, 6}) {
        for (int64_t precision : {0, 1}) {
            auto inputs = quant == 6 ? MakeMxfp8Inputs() : MakeInputs(ge::DT_FLOAT16);
            GenericBlockSparseAttentionCompileInfo compileInfo;
            gert::TilingContextPara ctx(
                "GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                MakeAttrs(1, quant, precision, 0, 0.0f, -1, -1, 4, "TND", "PA_BBND", 0, false, kBlockShape),
                &compileInfo, "Ascend950", 28, 262144);
            if (quant == 6) {
                ExecuteTestCase(ctx, ge::GRAPH_FAILED);
            } else {
                const uint64_t expectedKey = kFp16Tiling950 - (precision == 0 ? 100000U : 0U);
                ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, expectedKey);
            }
        }
    }
    EXPECT_EQ(Mxfp8Tiling::PipelineAuxBytes(256), 2048U);
    EXPECT_LE(Mxfp8Tiling::PipelineUbEnd(256, 512), Mxfp8Tiling::UB_CAPACITY);
}

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_rejects_nonpacked_without_optional_switch)
{
    for (int64_t d : {1, 31, 64, 128, 192, 256}) {
        for (int64_t sm : {0, 1}) {
            for (int64_t lse : {0, 1}) {
                const auto attrs = MakeAttrs(1, 6, sm, lse, 0.0f, -1, -1, 4, "TND", "PA_BBND");
                ASSERT_EQ(attrs.size(), 14U);
                GenericBlockSparseAttentionCompileInfo compileInfo;
                gert::TilingContextPara ctx("GenericBlockSparseAttention", MakeMxfp8Inputs(d),
                                            MakeOutputs(ge::DT_FLOAT16, lse != 0, d), attrs, &compileInfo, "Ascend950",
                                            28, 262144);
                ExecuteTestCase(ctx, ge::GRAPH_FAILED);
            }
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, mxfp8_fd_workspace_capacity_boundary)
{
    for (int64_t dim : {192, 256}) {
        for (int64_t capacity : {11, 12, 512, 8193}) {
            auto inputs = MakeMxfp8Inputs(dim);
            inputs[3] = {{{kN2, kTotalQBlocks, capacity}, {kN2, kTotalQBlocks, capacity}}, ge::DT_INT32, ge::FORMAT_ND};
            GenericBlockSparseAttentionCompileInfo compileInfo;
            gert::TilingContextPara ctx("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_BF16, false, dim),
                                        MakeAttrs(0, 6, 0, 0, 0.0f, -1, -1, 4, "TND", "PA_BBND", 0, false, {1, 4}),
                                        &compileInfo, "Ascend950", 28, 262144);
            TilingInfo info;
            ASSERT_TRUE(ExecuteTiling(ctx, info));
            ASSERT_GT(info.tilingDataSize, 152U);
            EXPECT_EQ(ReadTilingU32(info, 120U), capacity >= 12 ? 1U : 0U);
            const uint32_t partialCapacity = ReadTilingU32(info, 128U);
            if (capacity < 12) {
                EXPECT_EQ(partialCapacity, 0U);
                continue;
            }
            EXPECT_EQ(partialCapacity, 35U);
            uint64_t lseOffset = 0U, outputOffset = 0U;
            std::memcpy(&lseOffset, info.tilingData.get() + 136U, sizeof(uint64_t));
            std::memcpy(&outputOffset, info.tilingData.get() + 144U, sizeof(uint64_t));
            EXPECT_EQ(lseOffset % 512U, 0U);
            EXPECT_EQ(outputOffset % 512U, 0U);
            const uint64_t descriptors = info.blockNum * 4U * 8U * Mxfp8Tiling::DescriptorTileWords(512U);
            EXPECT_GE(lseOffset, descriptors);
            EXPECT_GE(outputOffset, lseOffset + partialCapacity * 2U * 8U * sizeof(float));
            ASSERT_EQ(info.workspaceSizes.size(), 1U);
            EXPECT_EQ(info.workspaceSizes[0],
                      16U * 1024U * 1024U + outputOffset + partialCapacity * (kN1 / kN2) * dim * sizeof(float));
        }
    }
}
