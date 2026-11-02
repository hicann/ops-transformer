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
 * \brief Ascend 950（arch35）tiling UT。平台信息在用例内 mock。
 */

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

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
    bool hasMask{false};
    std::vector<int64_t> qlenShape{2};
    std::vector<int64_t> klenShape{1};
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
};

gert::TilingContextPara MakeTilingPara(const MsaTilingSpec& spec, const MsaSocParam& soc, void* compileInfo)
{
    std::vector<gert::TilingContextPara::TensorDescription> inputs;
    inputs.push_back(Present(spec.queryShape, spec.queryDtype));
    inputs.push_back(Present(spec.keyShape, spec.keyDtype));
    inputs.push_back(spec.hasBlockTable ? Present(spec.blockTableShape, ge::DT_INT32) : Absent(ge::DT_INT32));
    inputs.push_back(spec.hasScale ? Present(spec.scaleShape, ge::DT_FLOAT) : Absent(ge::DT_FLOAT));
    inputs.push_back(spec.hasMask ? Present({2048, 2048}, ge::DT_INT8) : Absent(ge::DT_INT8));
    inputs.push_back(spec.qlenShape.empty() ? Absent(ge::DT_INT32) : Present(spec.qlenShape, ge::DT_INT32));
    inputs.push_back(spec.klenShape.empty() ? Absent(ge::DT_INT32) : Present(spec.klenShape, ge::DT_INT32));
    inputs.push_back(spec.startLocShape.empty() ? Absent(ge::DT_INT32) : Present(spec.startLocShape, ge::DT_INT32));

    std::vector<gert::TilingContextPara::TensorDescription> outputs;
    outputs.push_back(Present(spec.scoreShape, ge::DT_FLOAT));

    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>(spec.layout)},
        {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(spec.sparseMode)},
        {"init_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(spec.initBlocks)},
        {"local_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(spec.localBlocks)},
        {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>(spec.layoutQuery)},
        {"pool_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(spec.poolBlockSize)}};

    return gert::TilingContextPara("MsaIndexScore", inputs, outputs, attrs, compileInfo, soc.socVersion, soc.coreNum,
                                   soc.ubSize, MSA_TILING_CAP);
}

void RunTiling(const MsaTilingSpec& spec, const MsaSocParam& soc, ge::graphStatus expect, uint64_t expectKey = 0)
{
    struct MsaCompileInfo {
    } compileInfo;
    ExecuteTestCase(MakeTilingPara(spec, soc, &compileInfo), expect, expectKey);
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

class MsaIndexScoreTilingArch35 : public testing::TestWithParam<MsaSocParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MsaIndexScoreTilingArch35 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MsaIndexScoreTilingArch35 TearDown" << std::endl;
    }
};

TEST_P(MsaIndexScoreTilingArch35, bbnd_fp16_nonquant_success)
{
    RunTiling(MsaTilingSpec{}, GetParam(), ge::GRAPH_SUCCESS, MSA_KEY_FP16);
}

TEST_P(MsaIndexScoreTilingArch35, bbnd_bf16_causal_success)
{
    MsaTilingSpec spec;
    spec.queryDtype = ge::DT_BF16;
    spec.keyDtype = ge::DT_BF16;
    spec.sparseMode = 3;
    spec.hasMask = true;
    RunTiling(spec, GetParam(), ge::GRAPH_SUCCESS, MSA_KEY_BF16);
}

TEST_P(MsaIndexScoreTilingArch35, bbnd_fp16_int8_rejected)
{
    MsaTilingSpec spec;
    spec.keyDtype = ge::DT_INT8;
    spec.hasScale = true;
    spec.scaleShape = {2, 1, 128};
    RunTiling(spec, GetParam(), ge::GRAPH_FAILED);
}

TEST_P(MsaIndexScoreTilingArch35, bnbd_fp16_success)
{
    MsaTilingSpec spec;
    spec.layout = "BNBD";
    spec.keyShape = {2, 1, 128, 128};
    RunTiling(spec, GetParam(), ge::GRAPH_SUCCESS, MSA_KEY_FP16);
}

TEST_P(MsaIndexScoreTilingArch35, tnd_fp16_success)
{
    MsaTilingSpec spec;
    spec.layout = "TND";
    spec.keyShape = {256, 1, 128};
    spec.hasBlockTable = false;
    spec.klenShape = {2};
    RunTiling(spec, GetParam(), ge::GRAPH_SUCCESS, MSA_KEY_FP16);
}

TEST_P(MsaIndexScoreTilingArch35, empty_query_success)
{
    MsaTilingSpec spec;
    spec.queryShape = {0, 8, 128};
    spec.scoreShape = {8, 0, 16};
    RunTiling(spec, GetParam(), ge::GRAPH_SUCCESS, MSA_KEY_FP16);
}

TEST_P(MsaIndexScoreTilingArch35, missing_qlen_fail)
{
    MsaTilingSpec spec;
    spec.qlenShape = {};
    RunTiling(spec, GetParam(), ge::GRAPH_FAILED);
}

TEST_P(MsaIndexScoreTilingArch35, layout_invalid_fail)
{
    MsaTilingSpec spec;
    spec.layout = "BSH";
    RunTiling(spec, GetParam(), ge::GRAPH_FAILED);
}

TEST_P(MsaIndexScoreTilingArch35, bsnd_pool128_success)
{
    MsaTilingSpec spec;
    spec.layout = "BSND";
    spec.keyShape = {1, 256, 1, 128};
    spec.hasBlockTable = false;
    spec.klenShape = {1};
    RunTiling(spec, GetParam(), ge::GRAPH_SUCCESS, MSA_KEY_FP16);
}

TEST_P(MsaIndexScoreTilingArch35, bnsd_pool128_success)
{
    MsaTilingSpec spec;
    spec.layout = "BNSD";
    spec.keyShape = {1, 1, 256, 128};
    spec.hasBlockTable = false;
    spec.klenShape = {1};
    RunTiling(spec, GetParam(), ge::GRAPH_SUCCESS, MSA_KEY_FP16);
}

TEST_P(MsaIndexScoreTilingArch35, bsnd_with_block_table_fail)
{
    MsaTilingSpec spec;
    spec.layout = "BSND";
    spec.keyShape = {1, 256, 1, 128};
    spec.hasBlockTable = true;
    spec.klenShape = {1};
    RunTiling(spec, GetParam(), ge::GRAPH_FAILED);
}

TEST_P(MsaIndexScoreTilingArch35, bsnd_pool16_query_tnd_fail)
{
    MsaTilingSpec spec;
    spec.layout = "BSND";
    spec.poolBlockSize = 16;
    spec.keyShape = {1, 64, 1, 128};
    spec.hasBlockTable = false;
    spec.klenShape = {2};
    RunTiling(spec, GetParam(), ge::GRAPH_FAILED);
}

TEST_P(MsaIndexScoreTilingArch35, bsnd_pool16_v2_success)
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
    RunTiling(spec, GetParam(), ge::GRAPH_SUCCESS, 8);
}

TEST_P(MsaIndexScoreTilingArch35, causal_without_mask_fail)
{
    MsaTilingSpec spec;
    spec.sparseMode = 3;
    spec.hasMask = false;
    RunTiling(spec, GetParam(), ge::GRAPH_FAILED);
}

TEST_P(MsaIndexScoreTilingArch35, default_mask_with_atten_mask_fail)
{
    MsaTilingSpec spec;
    spec.sparseMode = 0;
    spec.hasMask = true;
    RunTiling(spec, GetParam(), ge::GRAPH_FAILED);
}

TEST_P(MsaIndexScoreTilingArch35, head_dim_not_aligned_fail)
{
    MsaTilingSpec spec;
    spec.queryShape = {32, 8, 8};
    spec.keyShape = {2, 128, 1, 8};
    RunTiling(spec, GetParam(), ge::GRAPH_FAILED);
}

TEST_P(MsaIndexScoreTilingArch35, block_size_invalid_fail)
{
    MsaTilingSpec spec;
    spec.keyShape = {2, 64, 1, 128};
    RunTiling(spec, GetParam(), ge::GRAPH_FAILED);
}

TEST_P(MsaIndexScoreTilingArch35, non_quant_with_scale_fail)
{
    MsaTilingSpec spec;
    spec.hasScale = true;
    spec.scaleShape = {2, 1, 128};
    RunTiling(spec, GetParam(), ge::GRAPH_FAILED);
}

TEST_P(MsaIndexScoreTilingArch35, tnd_with_block_table_fail)
{
    MsaTilingSpec spec;
    spec.layout = "TND";
    spec.keyShape = {256, 1, 128};
    spec.hasBlockTable = true;
    spec.klenShape = {2};
    RunTiling(spec, GetParam(), ge::GRAPH_FAILED);
}

TEST_P(MsaIndexScoreTilingArch35, init_blocks_too_large_fail)
{
    MsaTilingSpec spec;
    spec.initBlocks = 8; // maxBlocksPerBatch = 2
    RunTiling(spec, GetParam(), ge::GRAPH_FAILED);
}

TEST_P(MsaIndexScoreTilingArch35, int8_without_scale_fail)
{
    MsaTilingSpec spec;
    spec.keyDtype = ge::DT_INT8;
    spec.hasScale = false;
    RunTiling(spec, GetParam(), ge::GRAPH_FAILED);
}

TEST_P(MsaIndexScoreTilingArch35, pa_missing_block_table_fail)
{
    MsaTilingSpec spec;
    spec.hasBlockTable = false;
    RunTiling(spec, GetParam(), ge::GRAPH_FAILED);
}

// packedM=2，可见 block=32 → 4 个 S-tile。950 切 kvChunks=4、launchAic=8。
TEST_P(MsaIndexScoreTilingArch35, short_m_long_s_kv_chunks)
{
    const MsaSocParam& soc = GetParam();
    MsaTilingSpec spec;
    spec.keyShape = {32, 128, 1, 128};
    spec.blockTableShape = {1, 32};
    spec.scoreShape = {8, 32, 32};
    struct MsaCompileInfo {
    } compileInfo;
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(MakeTilingPara(spec, soc, &compileInfo), tilingInfo));
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

INSTANTIATE_TEST_SUITE_P(Soc, MsaIndexScoreTilingArch35, testing::Values(kSoc950),
                         [](const testing::TestParamInfo<MsaSocParam>& info) { return info.param.name; });

class MsaIndexScoreTilingFp8Arch35 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MsaIndexScoreTilingFp8Arch35 SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "MsaIndexScoreTilingFp8Arch35 TearDown" << std::endl;
    }
};

TEST_F(MsaIndexScoreTilingFp8Arch35, ascend950_fp8_e4m3fn_success)
{
    MsaTilingSpec spec;
    spec.queryDtype = ge::DT_FLOAT8_E4M3FN;
    spec.keyDtype = ge::DT_FLOAT8_E4M3FN;
    RunTiling(spec, kSoc950, ge::GRAPH_SUCCESS, MSA_KEY_FP8_E4M3FN);
}

TEST_F(MsaIndexScoreTilingFp8Arch35, ascend950_fp8_e5m2_success)
{
    MsaTilingSpec spec;
    spec.queryDtype = ge::DT_FLOAT8_E5M2;
    spec.keyDtype = ge::DT_FLOAT8_E5M2;
    RunTiling(spec, kSoc950, ge::GRAPH_SUCCESS, MSA_KEY_FP8_E5M2);
}

TEST_F(MsaIndexScoreTilingFp8Arch35, ascend950_hifloat8_rejected)
{
    MsaTilingSpec spec;
    spec.queryDtype = ge::DT_HIFLOAT8;
    spec.keyDtype = ge::DT_HIFLOAT8;
    RunTiling(spec, kSoc950, ge::GRAPH_FAILED);
}
