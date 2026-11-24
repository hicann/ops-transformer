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
 * \file test_msa_index_score_tiling.cpp
 * \brief Atlas A2/A3（arch22）tiling UT。平台信息在用例内 mock。
 */

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_kernel/msa_index_score_common.h"

using namespace std;

namespace {
constexpr uint64_t MSA_KEY_BF16 = 0;
constexpr uint64_t MSA_KEY_FP16 = 1;
constexpr uint64_t MSA_KEY_FP16_INT8 = 3;
constexpr uint64_t MSA_KEY_HIFLOAT8 = 4;
constexpr uint64_t MSA_KEY_FP8_E5M2 = 5;
constexpr uint64_t MSA_KEY_FP8_E4M3FN = 6;
constexpr uint64_t MSA_UB_950 = 262144;
constexpr uint64_t MSA_UB_A2 = 196608;
constexpr uint64_t MSA_TILING_CAP = 16384;

struct MsaSocParam {
    const char* name;
    const char* socVersion;
    uint64_t coreNum;
    uint64_t ubSize;
    bool is950;
};

const MsaSocParam kSoc910B{"Ascend910B", "Ascend910B", 24, MSA_UB_A2, false};
const MsaSocParam kSoc91093{"Ascend910_93", "Ascend910_93", 24, MSA_UB_A2, false};
const MsaSocParam kSoc950{"Ascend950", "Ascend950", 56, MSA_UB_950, true};

gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    switch (dims.size()) {
        case 1:
            return gert::StorageShape({dims[0]}, {dims[0]});
        case 2:
            return gert::StorageShape({dims[0], dims[1]}, {dims[0], dims[1]});
        case 3:
            return gert::StorageShape({dims[0], dims[1], dims[2]}, {dims[0], dims[1], dims[2]});
        case 4:
            return gert::StorageShape({dims[0], dims[1], dims[2], dims[3]}, {dims[0], dims[1], dims[2], dims[3]});
        default:
            return gert::StorageShape({}, {});
    }
}

gert::TilingContextPara::TensorDescription Present(const std::vector<int64_t>& shape, ge::DataType dtype)
{
    return gert::TilingContextPara::TensorDescription(MakeStorageShape(shape), dtype, ge::FORMAT_ND);
}

gert::TilingContextPara::TensorDescription Absent(ge::DataType dtype)
{
    return gert::TilingContextPara::TensorDescription(gert::StorageShape({}, {}), dtype, ge::FORMAT_ND);
}

struct MsaTilingSpec {
    std::vector<int64_t> queryShape{32, 8, 128};
    std::vector<int64_t> keyShape{2, 128, 1, 128};
    bool hasBlockTable{true};
    std::vector<int64_t> blockTableShape{1, 2};
    bool hasScale{false};
    std::vector<int64_t> scaleShape{};
    bool hasQueryScale{false};
    std::vector<int64_t> queryScaleShape{32, 8};
    ge::DataType queryScaleDtype{ge::DT_FLOAT};
    ge::DataType scaleDtype{ge::DT_FLOAT};
    bool hasMask{false};
    std::vector<int64_t> qlenShape{2};
    std::vector<int64_t> klenShape{1};
    std::vector<int32_t> klenData{};
    std::vector<int64_t> startLocShape{1};
    std::vector<int64_t> scoreShape{8, 32, 16};
    ge::DataType queryDtype{ge::DT_FLOAT16};
    ge::DataType keyDtype{ge::DT_FLOAT16};
    std::string layout{"BBND"};
    std::string layoutQuery{"TND"};
    int64_t poolBlockSize{128};
    int64_t sparseMode{0};
    int64_t initBlocks{0};
    int64_t localBlocks{1};
    int64_t returnChoice{0};
    int64_t topk{16};
    bool hasScore{true};
    bool hasIndices{false};
    bool hasValues{false};
    std::vector<int64_t> indicesShape{};
    std::vector<int64_t> valuesShape{};
};

gert::TilingContextPara MakeTilingPara(const MsaTilingSpec& spec, const MsaSocParam& soc, void* compileInfo)
{
    std::vector<gert::TilingContextPara::TensorDescription> inputs;
    inputs.push_back(Present(spec.queryShape, spec.queryDtype));
    inputs.push_back(Present(spec.keyShape, spec.keyDtype));
    inputs.push_back(spec.startLocShape.empty() ? Absent(ge::DT_INT32) : Present(spec.startLocShape, ge::DT_INT32));
    inputs.push_back(spec.hasBlockTable ? Present(spec.blockTableShape, ge::DT_INT32) : Absent(ge::DT_INT32));
    inputs.push_back(spec.hasQueryScale ? Present(spec.queryScaleShape, spec.queryScaleDtype) : Absent(ge::DT_FLOAT));
    inputs.push_back(spec.hasScale ? Present(spec.scaleShape, spec.scaleDtype) : Absent(ge::DT_FLOAT));
    inputs.push_back(spec.hasMask ? Present({2048, 2048}, ge::DT_INT8) : Absent(ge::DT_INT8));
    inputs.push_back(spec.qlenShape.empty() ? Absent(ge::DT_INT32) : Present(spec.qlenShape, ge::DT_INT32));
    auto klenDesc = spec.klenShape.empty() ? Absent(ge::DT_INT32) : Present(spec.klenShape, ge::DT_INT32);
    if (!spec.klenData.empty()) {
        klenDesc =
            gert::TilingContextPara::TensorDescription(MakeStorageShape(spec.klenShape), ge::DT_INT32, ge::FORMAT_ND,
                                                       true, const_cast<int32_t*>(spec.klenData.data()));
    }
    inputs.push_back(klenDesc);

    std::vector<gert::TilingContextPara::TensorDescription> outputs;
    if (spec.hasIndices || spec.hasValues || !spec.hasScore) {
        outputs.push_back(spec.hasScore ? Present(spec.scoreShape, ge::DT_FLOAT) : Absent(ge::DT_FLOAT));
        outputs.push_back(spec.hasIndices ? Present(spec.indicesShape, ge::DT_INT32) : Absent(ge::DT_INT32));
        outputs.push_back(spec.hasValues ? Present(spec.valuesShape, ge::DT_FLOAT) : Absent(ge::DT_FLOAT));
    } else {
        outputs.push_back(Present(spec.scoreShape, ge::DT_FLOAT));
    }

    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>(spec.layout)},
        {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(spec.sparseMode)},
        {"init_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(spec.initBlocks)},
        {"local_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(spec.localBlocks)},
        {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>(spec.layoutQuery)},
        {"pool_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(spec.poolBlockSize)},
        {"return_choice", Ops::Transformer::AnyValue::CreateFrom<int64_t>(spec.returnChoice)},
        {"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(spec.topk)}};

    return gert::TilingContextPara("MsaIndexScore", inputs, outputs, attrs, compileInfo, soc.socVersion, soc.coreNum,
                                   soc.ubSize, MSA_TILING_CAP);
}

void RunTiling(const MsaTilingSpec& spec, const MsaSocParam& soc, ge::graphStatus expect, uint64_t expectKey = 0)
{
    struct MsaCompileInfo {
    } compileInfo;
    ExecuteTestCase(MakeTilingPara(spec, soc, &compileInfo), expect, expectKey);
}

bool RunTilingForInfo(const MsaTilingSpec& spec, const MsaSocParam& soc, TilingInfo& info)
{
    struct MsaCompileInfo {
    } compileInfo;
    return ExecuteTiling(MakeTilingPara(spec, soc, &compileInfo), info);
}

MsaTilingSpec A8C8Spec()
{
    MsaTilingSpec spec;
    spec.queryDtype = ge::DT_INT8;
    spec.keyDtype = ge::DT_INT8;
    spec.hasScale = true;
    spec.scaleShape = {2, 1, 128};
    spec.hasQueryScale = true;
    return spec;
}

MsaTilingSpec PagedA8C8ScoreSpec(int64_t tokens, int64_t heads, int64_t pages, int64_t dim = 128, int64_t batch = 1)
{
    auto spec = A8C8Spec();
    spec.queryShape = {tokens, heads, dim};
    spec.queryScaleShape = {tokens, heads};
    spec.keyShape = {batch * pages, 128, 1, dim};
    spec.scaleShape = {batch * pages, 1, 128};
    spec.blockTableShape = {batch, pages};
    spec.qlenShape = {batch + 1};
    spec.klenShape = {batch};
    spec.startLocShape = {batch};
    spec.scoreShape = {heads, tokens, (pages + 15) / 16 * 16};
    spec.sparseMode = 3;
    spec.hasMask = true;
    spec.initBlocks = 0;
    spec.localBlocks = 0;
    return spec;
}

// tiling data 按 uint32 字段顺序打包。用 (batch, totalQ, numQHeads, numKvHeads, gqa, headDim) 定位起点。
bool FindTilingBase(const uint32_t* words, size_t n, uint32_t batch, uint32_t totalQ, uint32_t numQHeads,
                    uint32_t numKvHeads, uint32_t headDim, size_t& base)
{
    const uint32_t gqa = numQHeads / numKvHeads;
    for (size_t i = 0; i + 29 < n; ++i) {
        if (words[i] == batch && words[i + 1] == totalQ && words[i + 2] == numQHeads && words[i + 3] == numKvHeads &&
            words[i + 4] == gqa && words[i + 5] == headDim) {
            base = i;
            return true;
        }
    }
    return false;
}
} // namespace

class MsaIndexScoreTilingArch22 : public testing::TestWithParam<MsaSocParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MsaIndexScoreTilingArch22 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MsaIndexScoreTilingArch22 TearDown" << std::endl;
    }
};

TEST_P(MsaIndexScoreTilingArch22, bbnd_fp16_nonquant_success)
{
    RunTiling(MsaTilingSpec{}, GetParam(), ge::GRAPH_SUCCESS, MSA_KEY_FP16);
}

TEST_P(MsaIndexScoreTilingArch22, bbnd_bf16_causal_success)
{
    MsaTilingSpec spec;
    spec.queryDtype = ge::DT_BF16;
    spec.keyDtype = ge::DT_BF16;
    spec.sparseMode = 3;
    spec.hasMask = true;
    RunTiling(spec, GetParam(), ge::GRAPH_SUCCESS, MSA_KEY_BF16);
}

TEST_P(MsaIndexScoreTilingArch22, bbnd_fp16_int8_rejected)
{
    MsaTilingSpec spec;
    spec.keyDtype = ge::DT_INT8;
    spec.hasScale = true;
    spec.scaleShape = {2, 1, 128};
    RunTiling(spec, GetParam(), ge::GRAPH_FAILED);
}

TEST_P(MsaIndexScoreTilingArch22, bnbd_fp16_success)
{
    MsaTilingSpec spec;
    spec.layout = "BNBD";
    spec.keyShape = {2, 1, 128, 128};
    RunTiling(spec, GetParam(), ge::GRAPH_SUCCESS, MSA_KEY_FP16);
}

TEST_P(MsaIndexScoreTilingArch22, tnd_fp16_success)
{
    MsaTilingSpec spec;
    spec.layout = "TND";
    spec.keyShape = {256, 1, 128};
    spec.hasBlockTable = false;
    spec.klenShape = {2};
    RunTiling(spec, GetParam(), ge::GRAPH_SUCCESS, MSA_KEY_FP16);
}

TEST_P(MsaIndexScoreTilingArch22, empty_query_success)
{
    MsaTilingSpec spec;
    spec.queryShape = {0, 8, 128};
    spec.scoreShape = {8, 0, 16};
    RunTiling(spec, GetParam(), ge::GRAPH_SUCCESS, MSA_KEY_FP16);
}

TEST_P(MsaIndexScoreTilingArch22, missing_qlen_fail)
{
    MsaTilingSpec spec;
    spec.qlenShape = {};
    RunTiling(spec, GetParam(), ge::GRAPH_FAILED);
}

TEST_P(MsaIndexScoreTilingArch22, layout_invalid_fail)
{
    MsaTilingSpec spec;
    spec.layout = "BSH";
    RunTiling(spec, GetParam(), ge::GRAPH_FAILED);
}

TEST_P(MsaIndexScoreTilingArch22, bsnd_pool128_success)
{
    MsaTilingSpec spec;
    spec.layout = "BSND";
    spec.keyShape = {1, 256, 1, 128};
    spec.hasBlockTable = false;
    spec.klenShape = {2};
    RunTiling(spec, GetParam(), ge::GRAPH_SUCCESS, MSA_KEY_FP16);
}

TEST_P(MsaIndexScoreTilingArch22, bnsd_pool128_success)
{
    MsaTilingSpec spec;
    spec.layout = "BNSD";
    spec.keyShape = {1, 1, 256, 128};
    spec.hasBlockTable = false;
    spec.klenShape = {2};
    RunTiling(spec, GetParam(), ge::GRAPH_SUCCESS, MSA_KEY_FP16);
}

// 非 V2 顺序布局的 klen 同样为 [B+1] 前缀和，两个请求的长度分别为 1024、4096。
// 正确 maxTok=4096 -> 32 blocks -> 4 stiles，kvChunks=4、launchAic=16。
TEST_P(MsaIndexScoreTilingArch22, batch_seq_pool128_host_klen)
{
    for (const char* layout : {"BSND", "BNSD"}) {
        SCOPED_TRACE(layout);
        MsaTilingSpec spec;
        spec.layout = layout;
        spec.keyShape =
            (spec.layout == "BSND") ? std::vector<int64_t>{2, 4096, 1, 128} : std::vector<int64_t>{2, 1, 4096, 128};
        spec.hasBlockTable = false;
        spec.qlenShape = {3};
        spec.klenShape = {3};
        spec.klenData = {0, 1024, 5120};
        spec.startLocShape = {2};
        spec.scoreShape = {8, 32, 32};

        TilingInfo tilingInfo;
        ASSERT_TRUE(RunTilingForInfo(spec, GetParam(), tilingInfo));
        EXPECT_EQ(tilingInfo.tilingKey, MSA_KEY_FP16);
        ASSERT_GE(tilingInfo.tilingDataSize, 29U * sizeof(uint32_t));
        const auto* words = reinterpret_cast<const uint32_t*>(tilingInfo.tilingData.get());
        const size_t n = tilingInfo.tilingDataSize / sizeof(uint32_t);
        size_t base = 0;
        ASSERT_TRUE(FindTilingBase(words, n, 2, 32, 8, 1, 128, base));
        EXPECT_EQ(words[base + 11], 16U);
        EXPECT_EQ(words[base + 28], 4U);
    }
}

TEST_P(MsaIndexScoreTilingArch22, bsnd_pool16_rejected_on_a2)
{
    MsaTilingSpec spec;
    spec.layout = "BSND";
    spec.layoutQuery = "BSND";
    spec.poolBlockSize = 16;
    spec.queryShape = {1, 32, 8, 128};
    spec.keyShape = {1, 64, 1, 128};
    spec.hasBlockTable = false;
    spec.klenShape = {2};
    spec.scoreShape = {8, 32, 16};
    RunTiling(spec, GetParam(), ge::GRAPH_FAILED);
}

TEST_P(MsaIndexScoreTilingArch22, causal_without_mask_fail)
{
    MsaTilingSpec spec;
    spec.sparseMode = 3;
    spec.hasMask = false;
    RunTiling(spec, GetParam(), ge::GRAPH_FAILED);
}

TEST_P(MsaIndexScoreTilingArch22, default_mask_with_atten_mask_fail)
{
    MsaTilingSpec spec;
    spec.sparseMode = 0;
    spec.hasMask = true;
    RunTiling(spec, GetParam(), ge::GRAPH_FAILED);
}

TEST_P(MsaIndexScoreTilingArch22, head_dim_not_aligned_fail)
{
    MsaTilingSpec spec;
    spec.queryShape = {32, 8, 8};
    spec.keyShape = {2, 128, 1, 8};
    RunTiling(spec, GetParam(), ge::GRAPH_FAILED);
}

TEST_P(MsaIndexScoreTilingArch22, block_size_invalid_fail)
{
    MsaTilingSpec spec;
    spec.keyShape = {2, 64, 1, 128};
    RunTiling(spec, GetParam(), ge::GRAPH_FAILED);
}

TEST_P(MsaIndexScoreTilingArch22, non_quant_with_scale_fail)
{
    MsaTilingSpec spec;
    spec.hasScale = true;
    spec.scaleShape = {2, 1, 128};
    RunTiling(spec, GetParam(), ge::GRAPH_FAILED);
}

TEST_P(MsaIndexScoreTilingArch22, tnd_with_block_table_fail)
{
    MsaTilingSpec spec;
    spec.layout = "TND";
    spec.keyShape = {256, 1, 128};
    spec.hasBlockTable = true;
    spec.klenShape = {2};
    RunTiling(spec, GetParam(), ge::GRAPH_FAILED);
}

TEST_P(MsaIndexScoreTilingArch22, init_blocks_too_large_fail)
{
    MsaTilingSpec spec;
    spec.initBlocks = 8; // maxBlocksPerBatch = 2
    RunTiling(spec, GetParam(), ge::GRAPH_FAILED);
}

TEST_P(MsaIndexScoreTilingArch22, int8_without_scale_fail)
{
    MsaTilingSpec spec;
    spec.keyDtype = ge::DT_INT8;
    spec.hasScale = false;
    RunTiling(spec, GetParam(), ge::GRAPH_FAILED);
}

TEST_P(MsaIndexScoreTilingArch22, pa_missing_block_table_fail)
{
    MsaTilingSpec spec;
    spec.hasBlockTable = false;
    RunTiling(spec, GetParam(), ge::GRAPH_FAILED);
}

// packedM=2，可见 block=32 → 4 个 S-tile。AIC=24 时 gcd(2,24)=2，
// chunks 被 stile 数封顶为 4，launchAic=packedM*chunks=8。
TEST_P(MsaIndexScoreTilingArch22, short_m_long_s_kv_chunks)
{
    const MsaSocParam& soc = GetParam();
    MsaTilingSpec spec;
    spec.keyShape = {32, 128, 1, 128};
    spec.blockTableShape = {1, 32};
    spec.scoreShape = {8, 32, 32};
    TilingInfo tilingInfo;
    ASSERT_TRUE(RunTilingForInfo(spec, soc, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, MSA_KEY_FP16);
    ASSERT_GE(tilingInfo.tilingDataSize, 27U * sizeof(uint32_t));
    const auto* words = reinterpret_cast<const uint32_t*>(tilingInfo.tilingData.get());
    const size_t n = tilingInfo.tilingDataSize / sizeof(uint32_t);
    size_t base = 0;
    ASSERT_TRUE(FindTilingBase(words, n, 1, 32, 8, 1, 128, base));
    const uint32_t expectChunks = 4U;
    const uint32_t expectLaunch = 8U;
    // usedCoreNum 在 pool/page/table 三个字段之后；kvChunks 在 keyLayout、totalK 之后。
    EXPECT_EQ(words[base + 11], expectLaunch);
    EXPECT_EQ(words[base + 28], expectChunks);
}

INSTANTIATE_TEST_SUITE_P(Soc, MsaIndexScoreTilingArch22, testing::Values(kSoc910B, kSoc91093),
                         [](const testing::TestParamInfo<MsaSocParam>& info) { return info.param.name; });

class MsaIndexScoreTilingFp8Arch22 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MsaIndexScoreTilingFp8Arch22 SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "MsaIndexScoreTilingFp8Arch22 TearDown" << std::endl;
    }
};

TEST_F(MsaIndexScoreTilingFp8Arch22, a2_a3_fp8_rejected)
{
    MsaTilingSpec spec;
    spec.queryDtype = ge::DT_FLOAT8_E4M3FN;
    spec.keyDtype = ge::DT_FLOAT8_E4M3FN;
    RunTiling(spec, kSoc910B, ge::GRAPH_FAILED);
    RunTiling(spec, kSoc91093, ge::GRAPH_FAILED);
    spec.queryDtype = ge::DT_FLOAT8_E5M2;
    spec.keyDtype = ge::DT_FLOAT8_E5M2;
    RunTiling(spec, kSoc910B, ge::GRAPH_FAILED);
    spec.queryDtype = ge::DT_HIFLOAT8;
    spec.keyDtype = ge::DT_HIFLOAT8;
    RunTiling(spec, kSoc910B, ge::GRAPH_FAILED);
}

TEST_F(MsaIndexScoreTilingFp8Arch22, a2_a3_fp16_topk_indices_and_values)
{
    MsaTilingSpec spec;
    spec.returnChoice = 2;
    spec.topk = 16;
    spec.hasScore = false;
    spec.hasIndices = true;
    spec.hasValues = true;
    spec.indicesShape = {8, 32, 16};
    spec.valuesShape = {8, 32, 16};
    RunTiling(spec, kSoc910B, ge::GRAPH_SUCCESS, 14);
    RunTiling(spec, kSoc91093, ge::GRAPH_SUCCESS, 14);
}

TEST_F(MsaIndexScoreTilingFp8Arch22, a2_a3_bf16_topk_indices)
{
    MsaTilingSpec spec;
    spec.queryDtype = ge::DT_BF16;
    spec.keyDtype = ge::DT_BF16;
    spec.returnChoice = 1;
    spec.topk = 32;
    spec.hasScore = false;
    spec.hasIndices = true;
    spec.indicesShape = {8, 32, 32};
    RunTiling(spec, kSoc910B, ge::GRAPH_SUCCESS, 13);
    RunTiling(spec, kSoc91093, ge::GRAPH_SUCCESS, 13);
}

TEST_F(MsaIndexScoreTilingFp8Arch22, a2_a3_topk_pool16_rejected)
{
    MsaTilingSpec spec;
    spec.returnChoice = 1;
    spec.poolBlockSize = 16;
    spec.hasScore = false;
    spec.hasIndices = true;
    spec.indicesShape = {8, 32, 16};
    RunTiling(spec, kSoc910B, ge::GRAPH_FAILED);
    RunTiling(spec, kSoc91093, ge::GRAPH_FAILED);
}

TEST_P(MsaIndexScoreTilingArch22, a8c8_supported_layouts_and_dimensions)
{
    auto spec = A8C8Spec();
    RunTiling(spec, GetParam(), ge::GRAPH_SUCCESS, MSA_TILING_KEY_INT8_INT8_FIXED_PAIRED);
    spec.layout = "BNBD";
    spec.keyShape = {2, 1, 128, 128};
    RunTiling(spec, GetParam(), ge::GRAPH_SUCCESS, MSA_TILING_KEY_INT8_INT8_FIXED_PAIRED);
    spec.layout = "TND";
    spec.hasBlockTable = false;
    spec.klenShape = {2};
    spec.keyShape = {256, 1, 128};
    for (const auto& shape : std::vector<std::vector<int64_t>>{{256}, {256, 1}}) {
        spec.scaleShape = shape;
        RunTiling(spec, GetParam(), ge::GRAPH_SUCCESS, MSA_TILING_KEY_INT8_INT8_FIXED_PAIRED);
    }
    for (int64_t dim : {16, 48, 96, 129}) {
        spec = PagedA8C8ScoreSpec(1, 4, 64, dim);
        RunTiling(spec, GetParam(), dim > 128 ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS, MSA_TILING_KEY_INT8_INT8_FIXED);
    }
    RunTiling(A8C8Spec(), kSoc950, ge::GRAPH_FAILED);
}

TEST_P(MsaIndexScoreTilingArch22, a8c8_key_routes_and_boundaries)
{
    struct Route {
        int64_t tokens, heads, pages, dim, batch;
        uint64_t key;
    };
    for (const auto& route : std::vector<Route>{{1, 4, 64, 64, 1, MSA_TILING_KEY_INT8_INT8_FIXED},
                                                {1, 4, 64, 128, 1, MSA_TILING_KEY_INT8_INT8_FIXED_PAIRED},
                                                {15, 8, 64, 128, 1, MSA_TILING_KEY_INT8_INT8_FIXED_PAIRED},
                                                {16, 8, 64, 64, 1, MSA_TILING_KEY_INT8_INT8_FIXED_GROUP_MASK},
                                                {16, 8, 64, 128, 1, MSA_TILING_KEY_INT8_INT8_FIXED_PAIRED_GROUP_MASK},
                                                {255, 4, 80, 96, 2, MSA_TILING_KEY_INT8_INT8_FIXED},
                                                {256, 4, 80, 96, 2, MSA_TILING_KEY_INT8_INT8_FIXED_WIDE_SCORE},
                                                {64, 1, 65, 64, 2, MSA_TILING_KEY_INT8_INT8_FIXED_MIX_ONE},
                                                {65, 1, 65, 64, 2, MSA_TILING_KEY_INT8_INT8_FIXED},
                                                {80, 1, 65, 96, 2, MSA_TILING_KEY_INT8_INT8_FIXED_MIX_ONE},
                                                {81, 1, 65, 96, 2, MSA_TILING_KEY_INT8_INT8_FIXED},
                                                {1, 1, 64, 128, 1, MSA_TILING_KEY_INT8_INT8_FIXED_PAIRED},
                                                {1, 1, 65, 128, 1, MSA_TILING_KEY_INT8_INT8_FIXED_MIX_ONE},
                                                {1, 4, 255, 64, 1, MSA_TILING_KEY_INT8_INT8_FIXED},
                                                {1, 4, 256, 64, 1, MSA_TILING_KEY_INT8_INT8_FIXED_MIX_ONE},
                                                {1, 4, 257, 64, 1, MSA_TILING_KEY_INT8_INT8_FIXED}}) {
        SCOPED_TRACE("tokens=" + std::to_string(route.tokens) + " D=" + std::to_string(route.dim));
        RunTiling(PagedA8C8ScoreSpec(route.tokens, route.heads, route.pages, route.dim, route.batch), GetParam(),
                  ge::GRAPH_SUCCESS, route.key);
    }
}

TEST_P(MsaIndexScoreTilingArch22, a8c8_topk_preserves_score_routes_and_uses_two_aivs)
{
    struct Route {
        int64_t tokens, heads, pages, dim, batch;
        uint64_t scoreKey, topkScoreKey;
    };
    for (const auto& route : std::vector<Route>{
             {1, 4, 64, 64, 1, MSA_TILING_KEY_INT8_INT8_FIXED, MSA_TILING_KEY_INT8_INT8_FIXED},
             {1, 4, 64, 128, 1, MSA_TILING_KEY_INT8_INT8_FIXED_PAIRED, MSA_TILING_KEY_INT8_INT8_FIXED_PAIRED},
             {16, 8, 64, 64, 1, MSA_TILING_KEY_INT8_INT8_FIXED_GROUP_MASK, MSA_TILING_KEY_INT8_INT8_FIXED_GROUP_MASK},
             {16, 8, 64, 128, 1, MSA_TILING_KEY_INT8_INT8_FIXED_PAIRED_GROUP_MASK,
              MSA_TILING_KEY_INT8_INT8_FIXED_PAIRED_GROUP_MASK},
             {256, 4, 80, 96, 2, MSA_TILING_KEY_INT8_INT8_FIXED_WIDE_SCORE, MSA_TILING_KEY_INT8_INT8_FIXED_WIDE_SCORE},
             {1, 1, 65, 128, 1, MSA_TILING_KEY_INT8_INT8_FIXED_MIX_ONE, MSA_TILING_KEY_INT8_INT8_FIXED_PAIRED},
             {1, 4, 256, 64, 1, MSA_TILING_KEY_INT8_INT8_FIXED_MIX_ONE, MSA_TILING_KEY_INT8_INT8_FIXED}}) {
        SCOPED_TRACE("tokens=" + std::to_string(route.tokens) + " D=" + std::to_string(route.dim));
        auto spec = PagedA8C8ScoreSpec(route.tokens, route.heads, route.pages, route.dim, route.batch);
        RunTiling(spec, GetParam(), ge::GRAPH_SUCCESS, route.scoreKey);
        const auto topkKey = route.topkScoreKey + MSA_TILING_KEY_INT8_INT8_FIXED_TOPK - MSA_TILING_KEY_INT8_INT8_FIXED;
        spec.returnChoice = 1;
        spec.hasScore = false;
        spec.hasIndices = true;
        spec.indicesShape = {route.heads, route.tokens, spec.topk};
        RunTiling(spec, GetParam(), ge::GRAPH_SUCCESS, topkKey);
        spec.returnChoice = 2;
        spec.hasValues = true;
        spec.valuesShape = spec.indicesShape;
        RunTiling(spec, GetParam(), ge::GRAPH_SUCCESS, topkKey);
        spec.hasValues = false;
        RunTiling(spec, GetParam(), ge::GRAPH_FAILED);
    }
}

TEST_P(MsaIndexScoreTilingArch22, a8c8_wide_score_has_safe_fallbacks)
{
    auto spec = PagedA8C8ScoreSpec(256, 4, 80, 96, 2);
    spec.layout = "TND";
    spec.hasBlockTable = false;
    spec.klenShape = {3};
    spec.keyShape = {20480, 1, 96};
    spec.scaleShape = {20480, 1};
    RunTiling(spec, GetParam(), ge::GRAPH_SUCCESS, MSA_TILING_KEY_INT8_INT8_FIXED_WIDE_SCORE);
    spec = PagedA8C8ScoreSpec(256, 4, 64, 96, 2);
    RunTiling(spec, GetParam(), ge::GRAPH_SUCCESS, MSA_TILING_KEY_INT8_INT8_FIXED);
    spec = PagedA8C8ScoreSpec(256, 4, 80, 96, 2);
    spec.layout = "BNBD";
    spec.keyShape = {160, 1, 128, 96};
    RunTiling(spec, GetParam(), ge::GRAPH_SUCCESS, MSA_TILING_KEY_INT8_INT8_FIXED);
}

TEST_P(MsaIndexScoreTilingArch22, a8c8_memory_padding_and_empty_query)
{
    auto spec = A8C8Spec();
    TilingInfo aligned, padded;
    ASSERT_TRUE(RunTilingForInfo(spec, GetParam(), aligned));
    spec.queryShape[2] = spec.keyShape[3] = 48;
    ASSERT_TRUE(RunTilingForInfo(spec, GetParam(), padded));
    ASSERT_EQ(aligned.workspaceSizes.size(), 1U);
    ASSERT_EQ(padded.workspaceSizes.size(), 1U);
    EXPECT_EQ(padded.workspaceSizes[0] - aligned.workspaceSizes[0],
              static_cast<int64_t>(GetParam().coreNum * 128U * 64U));
    auto soc = GetParam();
    soc.ubSize = MsaIndexScoreNs::MSA_FIXED_SCALE_UB_BYTES + 255U;
    RunTiling(spec, soc, ge::GRAPH_FAILED);
    ++soc.ubSize;
    RunTiling(spec, soc, ge::GRAPH_SUCCESS, MSA_TILING_KEY_INT8_INT8_FIXED);
    spec = PagedA8C8ScoreSpec(0, 4, 64);
    RunTiling(spec, GetParam(), ge::GRAPH_SUCCESS, MSA_TILING_KEY_INT8_INT8_FIXED_PAIRED);
}

TEST_P(MsaIndexScoreTilingArch22, a8c8_owner_count_and_pool_exclusion)
{
    auto spec = PagedA8C8ScoreSpec(1, 4, 256, 64);
    TilingInfo info;
    ASSERT_TRUE(RunTilingForInfo(spec, GetParam(), info));
    const auto* words = reinterpret_cast<const uint32_t*>(info.tilingData.get());
    size_t base = 0;
    ASSERT_TRUE(FindTilingBase(words, info.tilingDataSize / sizeof(uint32_t), 1, 1, 4, 1, 64, base));
    EXPECT_EQ(words[base + 11], 16U);
    EXPECT_EQ(words[base + 28], 16U);
    spec.keyShape[0] = spec.scaleShape[0] = 255;
    RunTiling(spec, GetParam(), ge::GRAPH_SUCCESS, MSA_TILING_KEY_INT8_INT8_FIXED);
    spec = PagedA8C8ScoreSpec(1, 4, 256, 64);
    spec.localBlocks = 1;
    RunTiling(spec, GetParam(), ge::GRAPH_SUCCESS, MSA_TILING_KEY_INT8_INT8_FIXED);
}

TEST_P(MsaIndexScoreTilingArch22, a8c8_short_chunks_cover_unequal_queries)
{
    auto spec = PagedA8C8ScoreSpec(1024, 1, 64, 128, 2);
    TilingInfo info;
    ASSERT_TRUE(RunTilingForInfo(spec, GetParam(), info));
    EXPECT_EQ(info.tilingKey, MSA_TILING_KEY_INT8_INT8_FIXED_PAIRED_GROUP_MASK);
    const auto* words = reinterpret_cast<const uint32_t*>(info.tilingData.get());
    size_t base = 0;
    ASSERT_TRUE(FindTilingBase(words, info.tilingDataSize / sizeof(uint32_t), 2, 1024, 1, 1, 128, base));
    EXPECT_EQ(words[base + 11], 18U);
    EXPECT_EQ(words[base + 28], 2U);
}

TEST_P(MsaIndexScoreTilingArch22, a8c8_scale_contract_rejects_invalid_inputs)
{
    auto spec = A8C8Spec();
    spec.hasQueryScale = false;
    RunTiling(spec, GetParam(), ge::GRAPH_FAILED);
    spec = A8C8Spec();
    spec.hasScale = false;
    RunTiling(spec, GetParam(), ge::GRAPH_FAILED);
    for (const auto& shape : std::vector<std::vector<int64_t>>{{256}, {32, 1}, {32, 8, 1}}) {
        spec = A8C8Spec();
        spec.queryScaleShape = shape;
        RunTiling(spec, GetParam(), ge::GRAPH_FAILED);
    }
    spec = A8C8Spec();
    spec.scaleShape = {2, 1, 127};
    RunTiling(spec, GetParam(), ge::GRAPH_FAILED);
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16}) {
        spec = A8C8Spec();
        spec.queryScaleDtype = dtype;
        RunTiling(spec, GetParam(), ge::GRAPH_FAILED);
        spec = A8C8Spec();
        spec.scaleDtype = dtype;
        RunTiling(spec, GetParam(), ge::GRAPH_FAILED);
    }
}

TEST_P(MsaIndexScoreTilingArch22, a8c8_dtype_layout_and_pool_restrictions)
{
    auto spec = A8C8Spec();
    spec.keyDtype = ge::DT_FLOAT16;
    RunTiling(spec, GetParam(), ge::GRAPH_FAILED);
    spec = A8C8Spec();
    spec.layoutQuery = "BSND";
    RunTiling(spec, GetParam(), ge::GRAPH_FAILED);
    spec = A8C8Spec();
    spec.poolBlockSize = 16;
    RunTiling(spec, GetParam(), ge::GRAPH_FAILED);
    spec = A8C8Spec();
    spec.keyShape[2] = 2;
    spec.scaleShape = {2, 2, 128};
    RunTiling(spec, GetParam(), ge::GRAPH_FAILED);
    spec = A8C8Spec();
    spec.queryDtype = ge::DT_FLOAT16;
    RunTiling(spec, GetParam(), ge::GRAPH_FAILED);
    spec = MsaTilingSpec{};
    spec.hasQueryScale = true;
    RunTiling(spec, GetParam(), ge::GRAPH_FAILED);
}

TEST(MsaWideScoreOwner, borrowed_tail_is_written_by_last_covering_tile)
{
    using namespace MsaIndexScoreNs;
    for (uint32_t heads : {1U, 4U}) {
        for (uint32_t q : {0U, 1U, 31U, 32U, 33U, 127U, 128U, 129U, 143U, 144U, 257U}) {
            const uint32_t rows = q * heads;
            std::vector<uint32_t> last(rows, ~0U), owner(rows, ~0U), writes(rows, 0U);
            for (uint32_t tile = 0; tile * MSA_ROW_TILE_M < rows; ++tile) {
                uint32_t start = tile * MSA_ROW_TILE_M;
                uint32_t count = std::min(MSA_ROW_TILE_M, rows - start);
                if (count < MSA_M_ALIGN && start >= MSA_M_ALIGN) {
                    start -= MSA_M_ALIGN - count;
                    count = MSA_M_ALIGN;
                }
                const uint32_t end = MsaWideScoreOwnerEnd(q, heads, start, count);
                ASSERT_GE(end, start);
                ASSERT_LE(end, start + count);
                for (uint32_t r = start; r < start + count; ++r) {
                    last[r] = tile;
                    if (r < end) {
                        owner[r] = tile;
                        ++writes[r];
                    }
                }
            }
            for (uint32_t r = 0; r < rows; ++r) {
                ASSERT_EQ(writes[r], 1U) << "q=" << q << " heads=" << heads << " row=" << r;
                ASSERT_EQ(owner[r], last[r]) << "q=" << q << " heads=" << heads << " row=" << r;
            }
        }
    }
}
