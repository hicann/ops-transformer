/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file test_norm_rope_concat_tiling.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include <map>
#include <string>
#include "err_mgr.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_host/norm_rope_concat_tiling.h"

using namespace std;

class NormRopeConcatTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "NormRopeConcatTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "NormRopeConcatTiling TearDown" << std::endl;
    }
};

static string TilingData2Str(const gert::TilingData* tiling_data)
{
    auto data = tiling_data->GetData();
    string result;
    for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int64_t)) {
        result += std::to_string((reinterpret_cast<const int64_t*>(tiling_data->GetData())[i / sizeof(int64_t)]));
        result += " ";
    }

    return result;
}

TEST_F(NormRopeConcatTiling, norm_rope_concat_tiling_0)
{
    optiling::NormRopeConcatCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "NormRopeConcat",
        {
            {{{1, 8, 8, 64}, {1, 8, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 8, 8, 64}, {1, 8, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 8, 8, 64}, {1, 8, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 8, 8, 64}, {1, 8, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 8, 8, 64}, {1, 8, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 8, 8, 64}, {1, 8, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 64}, {16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 64}, {16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_added_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rope_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"concat_order", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
            {"is_training", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "1 8 8 8 8 8 8 16 16 16 16 1 8 8 64 64 8 64 8 4359484440192628669 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

namespace {
constexpr int64_t BATCH = 1;
constexpr int64_t SEQ = 8;
constexpr int64_t HEAD = 8;
constexpr int64_t DIM = 64;
constexpr int64_t TOTAL_SEQ = 2 * SEQ;

gert::TilingContextPara::TensorDescription MakeTensor(const std::vector<int64_t>& dims, ge::DataType dtype)
{
    gert::Shape shape;
    shape.SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        shape.SetDim(i, dims[i]);
    }
    gert::StorageShape storage;
    storage.MutableOriginShape() = shape;
    storage.MutableStorageShape() = shape;
    return {storage, dtype, ge::FORMAT_ND};
}

// 默认构造一组合法输出，用例仅改动需要校验的字段；空shape表示可选输出未提供
struct OutputShapes {
    std::vector<int64_t> query{BATCH, HEAD, TOTAL_SEQ, DIM};
    std::vector<int64_t> key{BATCH, HEAD, TOTAL_SEQ, DIM};
    std::vector<int64_t> value{BATCH, HEAD, TOTAL_SEQ, DIM};
    std::vector<int64_t> queryMean{};
    std::vector<int64_t> queryRstd{};
    std::vector<int64_t> keyMean{};
    std::vector<int64_t> keyRstd{};
    std::vector<int64_t> addedQueryMean{};
    std::vector<int64_t> addedQueryRstd{};
    std::vector<int64_t> addedKeyMean{};
    std::vector<int64_t> addedKeyRstd{};
};

gert::TilingContextPara MakeTilingPara(const OutputShapes& out, bool hasEncoder, int64_t normType,
                                       int64_t normAddedType, bool isTraining,
                                       optiling::NormRopeConcatCompileInfo* compileInfo, int64_t ropeType = 1,
                                       int64_t concatOrder = 0, bool hasRope = true)
{
    const std::vector<int64_t> xShape{BATCH, SEQ, HEAD, DIM};
    const std::vector<int64_t> encoderShape = hasEncoder ? xShape : std::vector<int64_t>{};
    const std::vector<int64_t> normShape{DIM};
    const int64_t ropeSeq = hasEncoder ? TOTAL_SEQ : SEQ;
    const std::vector<int64_t> ropeShape = hasRope ? std::vector<int64_t>{ropeSeq, DIM} : std::vector<int64_t>{};
    std::vector<gert::TilingContextPara::TensorDescription> inputs;
    for (size_t i = 0; i < 3; ++i) {
        inputs.emplace_back(MakeTensor(xShape, ge::DT_FLOAT16));
    }
    for (size_t i = 0; i < 3; ++i) {
        inputs.emplace_back(MakeTensor(encoderShape, ge::DT_FLOAT16));
    }
    for (size_t i = 0; i < 8; ++i) {
        inputs.emplace_back(MakeTensor(normShape, ge::DT_FLOAT16));
    }
    inputs.emplace_back(MakeTensor(ropeShape, ge::DT_FLOAT16));
    inputs.emplace_back(MakeTensor(ropeShape, ge::DT_FLOAT16));
    std::vector<gert::TilingContextPara::TensorDescription> outputs;
    outputs.emplace_back(MakeTensor(out.query, ge::DT_FLOAT16));
    outputs.emplace_back(MakeTensor(out.key, ge::DT_FLOAT16));
    outputs.emplace_back(MakeTensor(out.value, ge::DT_FLOAT16));
    outputs.emplace_back(MakeTensor(out.queryMean, ge::DT_FLOAT));
    outputs.emplace_back(MakeTensor(out.queryRstd, ge::DT_FLOAT));
    outputs.emplace_back(MakeTensor(out.keyMean, ge::DT_FLOAT));
    outputs.emplace_back(MakeTensor(out.keyRstd, ge::DT_FLOAT));
    outputs.emplace_back(MakeTensor(out.addedQueryMean, ge::DT_FLOAT));
    outputs.emplace_back(MakeTensor(out.addedQueryRstd, ge::DT_FLOAT));
    outputs.emplace_back(MakeTensor(out.addedKeyMean, ge::DT_FLOAT));
    outputs.emplace_back(MakeTensor(out.addedKeyRstd, ge::DT_FLOAT));
    return gert::TilingContextPara("NormRopeConcat", inputs, outputs,
                                   {{"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(normType)},
                                    {"norm_added_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(normAddedType)},
                                    {"rope_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ropeType)},
                                    {"concat_order", Ops::Transformer::AnyValue::CreateFrom<int64_t>(concatOrder)},
                                    {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
                                    {"is_training", Ops::Transformer::AnyValue::CreateFrom<bool>(isTraining)}},
                                   compileInfo);
}

void SetValidStats(OutputShapes& out)
{
    out.queryMean = {BATCH, SEQ, HEAD, 1};
    out.queryRstd = {BATCH, SEQ, HEAD, 1};
    out.keyMean = {BATCH, SEQ, HEAD, 1};
    out.keyRstd = {BATCH, SEQ, HEAD, 1};
    out.addedQueryMean = {BATCH, SEQ, HEAD, 1};
    out.addedQueryRstd = {BATCH, SEQ, HEAD, 1};
    out.addedKeyMean = {BATCH, SEQ, HEAD, 1};
    out.addedKeyRstd = {BATCH, SEQ, HEAD, 1};
}
} // namespace

// 训练模式LayerNorm的合法输出（含均值/标准差倒数）正常tiling
TEST_F(NormRopeConcatTiling, norm_rope_concat_tiling_training_valid)
{
    optiling::NormRopeConcatCompileInfo compileInfo = {};
    OutputShapes out;
    SetValidStats(out);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(MakeTilingPara(out, true, 1, 1, true, &compileInfo), tilingInfo));
}

// 无encoder输入时输出seq等于输入seq
TEST_F(NormRopeConcatTiling, norm_rope_concat_tiling_output_valid_without_encoder)
{
    optiling::NormRopeConcatCompileInfo compileInfo = {};
    OutputShapes out;
    out.query = {BATCH, HEAD, SEQ, DIM};
    out.key = {BATCH, HEAD, SEQ, DIM};
    out.value = {BATCH, HEAD, SEQ, DIM};
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(MakeTilingPara(out, false, 0, 0, false, &compileInfo), tilingInfo));
}

// queryOutput末维偏小时kernel会按headDim写出声明缓冲区，tiling仅告警不阻断
TEST_F(NormRopeConcatTiling, norm_rope_concat_tiling_output_head_dim_invalid)
{
    optiling::NormRopeConcatCompileInfo compileInfo = {};
    OutputShapes out;
    out.query = {BATCH, HEAD, TOTAL_SEQ, DIM / 2};
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(MakeTilingPara(out, true, 0, 0, false, &compileInfo), tilingInfo));
}

// 输出的seq维、head数维与推导结果不一致时仅告警
TEST_F(NormRopeConcatTiling, norm_rope_concat_tiling_output_seq_head_invalid)
{
    optiling::NormRopeConcatCompileInfo compileInfo = {};
    OutputShapes out;
    out.query = {BATCH, HEAD, SEQ, DIM};
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(MakeTilingPara(out, true, 0, 0, false, &compileInfo), tilingInfo));
    out = OutputShapes();
    out.query = {BATCH, 2 * HEAD, TOTAL_SEQ, DIM};
    EXPECT_TRUE(ExecuteTiling(MakeTilingPara(out, true, 0, 0, false, &compileInfo), tilingInfo));
}

// keyOutput/valueOutput同样校验并告警
TEST_F(NormRopeConcatTiling, norm_rope_concat_tiling_key_value_output_invalid)
{
    optiling::NormRopeConcatCompileInfo compileInfo = {};
    OutputShapes out;
    out.key = {BATCH, HEAD, SEQ, DIM};
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(MakeTilingPara(out, true, 0, 0, false, &compileInfo), tilingInfo));
    out = OutputShapes();
    out.value = {BATCH, HEAD, TOTAL_SEQ, DIM / 2};
    EXPECT_TRUE(ExecuteTiling(MakeTilingPara(out, true, 0, 0, false, &compileInfo), tilingInfo));
}

// 训练模式LayerNorm的均值/标准差倒数输出shape不合法时仅告警
TEST_F(NormRopeConcatTiling, norm_rope_concat_tiling_norm_stats_invalid)
{
    optiling::NormRopeConcatCompileInfo compileInfo = {};
    OutputShapes out;
    SetValidStats(out);
    out.queryMean = {BATCH, SEQ, HEAD, 2};
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(MakeTilingPara(out, true, 1, 1, true, &compileInfo), tilingInfo));
    out = OutputShapes();
    SetValidStats(out);
    out.addedKeyRstd = {BATCH, SEQ, HEAD, 1, 1};
    EXPECT_TRUE(ExecuteTiling(MakeTilingPara(out, true, 1, 1, true, &compileInfo), tilingInfo));
}

// 非连续输出视图：storage大于origin时不应被误杀（kernel按origin shape对应的连续布局写出）
TEST_F(NormRopeConcatTiling, norm_rope_concat_tiling_output_storage_larger_valid)
{
    optiling::NormRopeConcatCompileInfo compileInfo = {};
    OutputShapes out;
    auto para = MakeTilingPara(out, true, 0, 0, false, &compileInfo);
    gert::Shape storage = para.outputTensorDesc_[0].shape_.GetStorageShape();
    storage.SetDim(2, TOTAL_SEQ + 1);
    para.outputTensorDesc_[0].shape_.MutableStorageShape() = storage;
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(para, tilingInfo));
}

// 非训练模式下kernel不写均值/标准差倒数，其shape不参与校验，保持原有调用方式可用
TEST_F(NormRopeConcatTiling, norm_rope_concat_tiling_norm_stats_ignored_when_not_training)
{
    optiling::NormRopeConcatCompileInfo compileInfo = {};
    OutputShapes out;
    out.queryMean = {BATCH, SEQ, HEAD, 2};
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(MakeTilingPara(out, true, 1, 1, false, &compileInfo), tilingInfo));
}

// ropeType越界应在第一段tiling即失败
TEST_F(NormRopeConcatTiling, norm_rope_concat_tiling_rope_type_out_of_range)
{
    optiling::NormRopeConcatCompileInfo compileInfo = {};
    OutputShapes out;
    for (const int64_t ropeType : {3, 4, -1}) {
        ExecuteTestCase(MakeTilingPara(out, true, 0, 0, false, &compileInfo, ropeType), ge::GRAPH_FAILED);
    }
}

// concatOrder越界应在第一段tiling即失败
TEST_F(NormRopeConcatTiling, norm_rope_concat_tiling_concat_order_out_of_range)
{
    optiling::NormRopeConcatCompileInfo compileInfo = {};
    OutputShapes out;
    for (const int64_t concatOrder : {2, -1}) {
        ExecuteTestCase(MakeTilingPara(out, true, 0, 0, false, &compileInfo, 1, concatOrder), ge::GRAPH_FAILED);
    }
}

// ropeType=0且不传ropeSin/ropeCos时应正常tiling，ropeDim按headDim补齐，避免value通路搬运0字节
TEST_F(NormRopeConcatTiling, norm_rope_concat_tiling_rope_none_without_rope_inputs)
{
    optiling::NormRopeConcatCompileInfo compileInfo = {};
    OutputShapes out;
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(
        MakeTilingPara(out, true, 0, 0, false, &compileInfo, /*ropeType*/ 0, 0, /*hasRope*/ false), tilingInfo));
    auto* tilingData = reinterpret_cast<const int64_t*>(tilingInfo.tilingData.get());
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData[15], DIM); // ropeDim
    EXPECT_EQ(tilingData[10], 0);   // ropeActualSeq
}

TEST_F(NormRopeConcatTiling, norm_rope_concat_tiling_ub_failure_reports_ez1008)
{
    optiling::NormRopeConcatCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "NormRopeConcat",
        {
            {{{1, 8, 8, 64}, {1, 8, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 8, 8, 64}, {1, 8, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 8, 8, 64}, {1, 8, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 8, 8, 64}, {1, 8, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 8, 8, 64}, {1, 8, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 8, 8, 64}, {1, 8, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 64}, {16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 64}, {16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_added_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rope_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"concat_order", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
            {"is_training", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "1 8 8 8 8 8 8 16 16 16 16 1 8 8 64 64 8 64 8 4359484440192628669 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    // PlatformAscendC subtracts 256 bytes from configured UB; 257 yields actual UB=1.
    tilingContextPara.ubSize_ = 257;
    ASSERT_EQ(error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE), 0);
    error_message::GetErrMgrRawErrorMessages();
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
    auto reports = error_message::GetErrMgrRawErrorMessages();
    ASSERT_EQ(reports.size(), 1U);
    ASSERT_NE(reports[0].error_id.get(), nullptr);
    ASSERT_EQ(std::string(reports[0].error_id.get()), "EZ1008");
    ASSERT_EQ(reports[0].args_key.size(), 2U);
    ASSERT_EQ(reports[0].args_value.size(), 2U);
    std::map<std::string, std::string> args;
    for (size_t i = 0; i < reports[0].args_key.size(); ++i) {
        ASSERT_NE(reports[0].args_key[i].get(), nullptr);
        ASSERT_NE(reports[0].args_value[i].get(), nullptr);
        args.emplace(reports[0].args_key[i].get(), reports[0].args_value[i].get());
    }
    EXPECT_EQ(args["opName"], "NormRopeConcat");
    EXPECT_EQ(args["reason"], "the input dim is too large to be tiled, headDim=64");
}
