/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <array>
#include <gtest/gtest.h>
#include "../../../op_host/dequant_rope_quant_kvcache_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "error_code_test_utils.h"
#include "log/log.h"

using namespace std;

struct DequantRopeQuantKvcacheCompileInfo {};

struct DequantRopeQuantKvcacheTilingDataView {
    int64_t qHeadNum;
    int64_t kvHeadNum;
    int64_t hiddenSize;
    int64_t hiddenSizeFp32Align;
    int64_t hiddenSizeFp16Align;
    int64_t hiddenSizeInt8Align;
    int64_t onceUBMaxS;
    int64_t cacheSeqlen;
    int64_t seqlen;
    int64_t qHiddenSize;
    int64_t kHiddenSize;
    int64_t vHiddenSize;
    int64_t realCoreNum;
    int64_t frontCoreNum;
    int64_t blockFactor;
    int64_t tailCoreBlockFactor;
    int64_t hasOffsetK;
    int64_t hasOffsetV;
    int64_t ifKVout;
    int64_t isPA;
    int64_t hasBias;
    int64_t hasAS;
    int64_t batch;
};
static_assert(sizeof(DequantRopeQuantKvcacheTilingDataView) == 23 * sizeof(int64_t));

class DequantRopeQuantKvcacheTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "DequantRopeQuantKvcacheTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "DequantRopeQuantKvcacheTiling TearDown" << std::endl;
    }
};

static gert::TilingContextPara MakeValidTilingContext(void* compileInfo)
{
    return gert::TilingContextPara(
        "DequantRopeQuantKvcache",
        {
            {{{1, 320, 1280}, {1, 320, 1280}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 320, 1, 128}, {1, 320, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 320, 1, 128}, {1, 320, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{{{1, 320, 8, 128}, {1, 320, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND}},
        {{"size_splits", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({8 * 128, 1 * 128, 1 * 128})},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("static")},
         {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"kv_output", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("contiguous")}},
        compileInfo);
}

TEST_F(DequantRopeQuantKvcacheTiling, DequantRopeQuantKvcachetiling_01)
{
    DequantRopeQuantKvcacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "DequantRopeQuantKvcache",
        {
            {{{1, 320, 1280}, {1, 320, 1280}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 320, 1, 128}, {1, 320, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 320, 1, 128}, {1, 320, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{{{1, 320, 8, 128}, {1, 320, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND}},
        {{"size_splits", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({8 * 128, 1 * 128, 1 * 128})},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("static")},
         {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"kv_output", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);

    uint64_t expectTilingKey = 0;
    string expectTilingData = "8 1 128 128 256 512 44 1024 320 1024 128 128 64 64 5 5 1 1 0 0 0 0 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    drqk_test::ClearErrorMessages();
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
    drqk_test::ExpectNoError();
}

TEST_F(DequantRopeQuantKvcacheTiling, DequantRopeQuantKvcachetiling_unsupported_string_attrs_keep_compatibility)
{
    DequantRopeQuantKvcacheCompileInfo compileInfo = {};
    auto para = MakeValidTilingContext(&compileInfo);
    para.attrs_[1].attr_ = Ops::Transformer::AnyValue::CreateFrom<std::string>("");
    para.attrs_[2].attr_ = Ops::Transformer::AnyValue::CreateFrom<std::string>("TND");
    para.attrs_[4].attr_ = Ops::Transformer::AnyValue::CreateFrom<std::string>("XXXX");

    uint64_t expectTilingKey = 0;
    string expectTilingData = "8 1 128 128 256 512 44 1024 320 1024 128 128 64 64 5 5 1 1 0 0 0 0 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    drqk_test::ClearErrorMessages();
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
    drqk_test::ExpectNoError();
}

TEST_F(DequantRopeQuantKvcacheTiling, DequantRopeQuantKvcachetiling_unsupported_string_attr_warnings)
{
    std::vector<std::string> warnings;
    auto emitWarning = [&warnings](const std::string& warning) { warnings.push_back(warning); };
    optiling::detail::EmitUnsupportedStringAttrWarnings("", "TND", "XXXX", emitWarning);

    ASSERT_EQ(warnings.size(), 3U);
    EXPECT_NE(warnings[0].find("quant_mode=\"\" is not supported"), std::string::npos);
    EXPECT_NE(warnings[1].find("layout=\"TND\" is not supported"), std::string::npos);
    EXPECT_NE(warnings[2].find("cache_mode=\"XXXX\" is not supported"), std::string::npos);

    warnings.clear();
    optiling::detail::EmitUnsupportedStringAttrWarnings("static", "BSND", "Page", emitWarning);
    EXPECT_TRUE(warnings.empty());
}

TEST_F(DequantRopeQuantKvcacheTiling, DequantRopeQuantKvcachetiling_supported_cache_case_keeps_compatibility)
{
    DequantRopeQuantKvcacheCompileInfo compileInfo = {};
    auto para = MakeValidTilingContext(&compileInfo);
    para.inputTensorDesc_[5] =
        gert::TilingContextPara::TensorDescription({{{320}, {320}}, ge::DT_INT32, ge::FORMAT_ND});
    para.attrs_[4].attr_ = Ops::Transformer::AnyValue::CreateFrom<std::string>("Page");

    ExecuteTestCase(para, ge::GRAPH_SUCCESS);
}

TEST_F(DequantRopeQuantKvcacheTiling, DequantRopeQuantKvcachetiling_invalid_x_rank)
{
    DequantRopeQuantKvcacheCompileInfo compileInfo = {};
    auto para = MakeValidTilingContext(&compileInfo);
    para.inputTensorDesc_[0] = gert::TilingContextPara::TensorDescription(
        {{{1, 1, 320, 1280}, {1, 1, 320, 1280}}, ge::DT_FLOAT16, ge::FORMAT_ND});
    drqk_test::ClearErrorMessages();
    ExecuteTestCase(para, ge::GRAPH_FAILED);
    drqk_test::ExpectSingleError("EZ0011", {{"param_name", "x"},
                                            {"op_name", "DequantRopeQuantKvcache"},
                                            {"incorrect_dim", "4"},
                                            {"correct_dim", "2D or 3D"}});
}

TEST_F(DequantRopeQuantKvcacheTiling, DequantRopeQuantKvcachetiling_invalid_size_splits_length)
{
    DequantRopeQuantKvcacheCompileInfo compileInfo = {};
    auto para = MakeValidTilingContext(&compileInfo);
    para.attrs_[0].attr_ = Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1024, 128});
    drqk_test::ClearErrorMessages();
    ExecuteTestCase(para, ge::GRAPH_FAILED);
    drqk_test::ExpectSingleError("EZ0025", {{"param_name", "size_splits"},
                                            {"op_name", "DequantRopeQuantKvcache"},
                                            {"incorrect_size", "2"},
                                            {"correct_size", "3"}});
}

TEST_F(DequantRopeQuantKvcacheTiling, DequantRopeQuantKvcachetiling_invalid_optional_input_dtype)
{
    DequantRopeQuantKvcacheCompileInfo compileInfo = {};
    auto para = MakeValidTilingContext(&compileInfo);
    para.inputTensorDesc_.emplace_back(gert::StorageShape({128}, {128}), ge::DT_FLOAT, ge::FORMAT_ND);
    para.inputTensorDesc_.emplace_back(gert::StorageShape({1}, {1}), ge::DT_FLOAT16, ge::FORMAT_ND);
    drqk_test::ClearErrorMessages();
    ExecuteTestCase(para, ge::GRAPH_FAILED);
    drqk_test::ExpectSingleError("EZ0019", {{"param_name", "activation_scale"},
                                            {"op_name", "DequantRopeQuantKvcache"},
                                            {"incorrect_dtype", "DT_FLOAT16"},
                                            {"correct_dtype", "FLOAT"}});
}

TEST_F(DequantRopeQuantKvcacheTiling, DequantRopeQuantKvcachetiling_invalid_scale_shape)
{
    DequantRopeQuantKvcacheCompileInfo compileInfo = {};
    auto para = MakeValidTilingContext(&compileInfo);
    para.inputTensorDesc_[6] =
        gert::TilingContextPara::TensorDescription({{{127}, {127}}, ge::DT_FLOAT, ge::FORMAT_ND});
    drqk_test::ClearErrorMessages();
    ExecuteTestCase(para, ge::GRAPH_FAILED);
    drqk_test::ExpectSingleError("EZ0016",
                                 {{"param_names", "scale_k, scale_v"},
                                  {"op_name", "DequantRopeQuantKvcache"},
                                  {"incorrect_sizes", "127, 128"},
                                  {"reason", "both sizes must equal k_cache.shape[2] * k_cache.shape[3] 128"}});
}

TEST_F(DequantRopeQuantKvcacheTiling, DequantRopeQuantKvcachetiling_invalid_cache_shape_relation)
{
    DequantRopeQuantKvcacheCompileInfo compileInfo = {};
    auto para = MakeValidTilingContext(&compileInfo);
    para.inputTensorDesc_[4] = gert::TilingContextPara::TensorDescription(
        {{{1, 1024, 2, 128}, {1, 1024, 2, 128}}, ge::DT_INT8, ge::FORMAT_ND});
    drqk_test::ClearErrorMessages();
    ExecuteTestCase(para, ge::GRAPH_FAILED);
    drqk_test::ExpectSingleError("EZ0010", {{"param_names", "k_cache, v_cache"},
                                            {"op_name", "DequantRopeQuantKvcache"},
                                            {"incorrect_shapes", "[1, 1024, 1, 128], [1, 1024, 2, 128]"},
                                            {"reason", "k_cache and v_cache must have the same shape"}});
}

TEST_F(DequantRopeQuantKvcacheTiling, DequantRopeQuantKvcachetiling_invalid_size_splits_value)
{
    DequantRopeQuantKvcacheCompileInfo compileInfo = {};
    auto para = MakeValidTilingContext(&compileInfo);
    para.attrs_[0].attr_ = Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({8 * 128, 64, 1 * 128});
    drqk_test::ClearErrorMessages();
    ExecuteTestCase(para, ge::GRAPH_FAILED);
    drqk_test::ExpectSingleError("EZ0026", {{"param_name", "size_splits"},
                                            {"op_name", "DequantRopeQuantKvcache"},
                                            {"incorrect_value", "[1024, 64, 128]"},
                                            {"reason", "size_splits[1] and size_splits[2] must be equal"}});
}

TEST_F(DequantRopeQuantKvcacheTiling, DequantRopeQuantKvcachetiling_null_context)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto tilingFunc = spaceRegistry->GetOpImpl("DequantRopeQuantKvcache")->tiling;
    drqk_test::ClearErrorMessages();
    EXPECT_EQ(tilingFunc(nullptr), ge::GRAPH_FAILED);
    drqk_test::ExpectSingleError("EZ1008",
                                 {{"opName", "DequantRopeQuantKvcache"}, {"reason", "Tiling context is unavailable"}});
}

TEST_F(DequantRopeQuantKvcacheTiling, DequantRopeQuantKvcachetiling_missing_weight_scale)
{
    DequantRopeQuantKvcacheCompileInfo compileInfo = {};
    auto para = MakeValidTilingContext(&compileInfo);
    para.inputTensorDesc_[0].dtype_ = ge::DT_INT32;
    drqk_test::ClearErrorMessages();
    ExecuteTestCase(para, ge::GRAPH_FAILED);
    drqk_test::ExpectSingleError("EZ0004", {{"param_name", "weight_scale"}, {"op_name", "DequantRopeQuantKvcache"}});
}

TEST_F(DequantRopeQuantKvcacheTiling, DequantRopeQuantKvcachetiling_02)
{
    DequantRopeQuantKvcacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "DequantRopeQuantKvcache",
        {
            {{{1, 320, 1280}, {1, 320, 1280}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 320, 1, 128}, {1, 320, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 320, 1, 128}, {1, 320, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{{{1, 320, 8, 128}, {1, 320, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 320, 1, 128}, {1, 320, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 320, 1, 128}, {1, 320, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND}},
        {{"size_splits", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({8 * 128, 1 * 128, 1 * 128})},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("static")},
         {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"kv_output", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
        &compileInfo);

    uint64_t expectTilingKey = 0;
    string expectTilingData = "8 1 128 128 256 512 44 1024 320 1024 128 128 64 64 5 5 1 1 1 0 0 0 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(DequantRopeQuantKvcacheTiling, DequantRopeQuantKvcachetiling_03)
{
    DequantRopeQuantKvcacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "DequantRopeQuantKvcache",
        {
            {{{1, 320, 1280}, {1, 320, 1280}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 320, 1, 128}, {1, 320, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 320, 1, 128}, {1, 320, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{{{1, 320, 8, 128}, {1, 320, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 320, 1, 128}, {1, 320, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 320, 1, 128}, {1, 320, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND}},
        {{"size_splits", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({8 * 128, 1 * 128, 1 * 128})},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("static")},
         {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"kv_output", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
        &compileInfo);

    uint64_t expectTilingKey = 0;
    string expectTilingData = "8 1 128 128 256 512 44 1024 320 1024 128 128 64 64 5 5 0 0 1 0 0 0 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(DequantRopeQuantKvcacheTiling, DequantRopeQuantKvcachetiling_04)
{
    DequantRopeQuantKvcacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "DequantRopeQuantKvcache",
        {
            {{{1, 320, 1280}, {1, 320, 1280}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 320, 1, 128}, {1, 320, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 320, 1, 128}, {1, 320, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{{{1, 320, 8, 128}, {1, 320, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND}},
        {{"size_splits", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({8 * 128, 1 * 128, 1 * 128})},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("static")},
         {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"kv_output", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);

    uint64_t expectTilingKey = 0;
    string expectTilingData = "8 1 128 128 256 512 44 1024 320 1024 128 128 64 64 5 5 1 1 0 0 0 0 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(DequantRopeQuantKvcacheTiling, DequantRopeQuantKvcachetiling_05)
{
    DequantRopeQuantKvcacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "DequantRopeQuantKvcache",
        {
            {{{1, 320, 1280}, {1, 320, 1280}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 320, 1, 128}, {1, 320, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 320, 1, 128}, {1, 320, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{{{1, 320, 8, 128}, {1, 320, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND}},
        {{"size_splits", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({8 * 128, 1 * 128, 1 * 128})},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("static")},
         {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"kv_output", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);

    uint64_t expectTilingKey = 0;
    string expectTilingData = "8 1 128 128 256 512 44 1024 320 1024 128 128 64 64 5 5 0 0 0 0 0 0 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

namespace {
struct OutputShapeCase {
    std::string name;
    std::array<std::vector<int64_t>, 3> outputs = {{{2, 4, 8, 64}, {2, 4, 1, 64}, {2, 4, 1, 64}}};
    bool twoDim = false;
    bool page = false;
    bool kvOutput = true;
    ge::graphStatus expected = ge::GRAPH_FAILED;
};

OutputShapeCase InvalidOutput(const char* name, size_t index, std::initializer_list<int64_t> shape)
{
    OutputShapeCase result;
    result.name = name;
    result.outputs[index] = shape;
    return result;
}

OutputShapeCase ValidOutputs(const char* name, bool twoDim, bool page, bool kvOutput)
{
    OutputShapeCase result;
    result.name = name;
    result.twoDim = twoDim;
    result.page = page;
    result.kvOutput = kvOutput;
    result.expected = ge::GRAPH_SUCCESS;
    if (twoDim) {
        result.outputs = {{{2, 8, 64}, {2, 1, 64}, {2, 1, 64}}};
    }
    if (!kvOutput) {
        result.outputs[1] = {0};
        result.outputs[2] = {0};
    }
    return result;
}

OutputShapeCase NonemptyKvOutput(const char* name, size_t index)
{
    auto result = ValidOutputs(name, false, false, false);
    result.outputs[index] = {2, 4, 1, 64};
    result.expected = ge::GRAPH_FAILED;
    return result;
}

gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape result;
    for (auto dim : dims) {
        result.MutableOriginShape().AppendDim(dim);
        result.MutableStorageShape().AppendDim(dim);
    }
    return result;
}

class DequantRopeQuantKvcacheOutputShapeTiling : public testing::TestWithParam<OutputShapeCase> {};

TEST_P(DequantRopeQuantKvcacheOutputShapeTiling, ValidateOutputs)
{
    const auto& param = GetParam();
    DequantRopeQuantKvcacheCompileInfo compileInfo = {};
    const gert::StorageShape xShape =
        param.twoDim ? gert::StorageShape{{2, 640}, {2, 640}} : gert::StorageShape{{2, 4, 640}, {2, 4, 640}};
    const gert::StorageShape cosShape =
        param.twoDim ? gert::StorageShape{{2, 64}, {2, 64}} : gert::StorageShape{{2, 4, 1, 64}, {2, 4, 1, 64}};
    const gert::StorageShape cacheShape{{2, 16, 1, 64}, {2, 16, 1, 64}};
    const gert::StorageShape indicesShape =
        param.page && !param.twoDim ? gert::StorageShape{{8}, {8}} : gert::StorageShape{{2}, {2}};
    gert::TilingContextPara context(
        "DequantRopeQuantKvcache",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {cosShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {cosShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {cacheShape, ge::DT_INT8, ge::FORMAT_ND},
         {cacheShape, ge::DT_INT8, ge::FORMAT_ND},
         {indicesShape, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 64}, {1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1, 64}, {1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{MakeStorageShape(param.outputs[0]), ge::DT_FLOAT16, ge::FORMAT_ND},
         {MakeStorageShape(param.outputs[1]), ge::DT_FLOAT16, ge::FORMAT_ND},
         {MakeStorageShape(param.outputs[2]), ge::DT_FLOAT16, ge::FORMAT_ND},
         {cacheShape, ge::DT_INT8, ge::FORMAT_ND},
         {cacheShape, ge::DT_INT8, ge::FORMAT_ND}},
        {{"size_splits", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({512, 64, 64})},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("static")},
         {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"kv_output", Ops::Transformer::AnyValue::CreateFrom<bool>(param.kvOutput)},
         {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.page ? "page" : "contiguous")}},
        &compileInfo, "Ascend950");
    ExecuteTestCase(context, param.expected, 0, "", {16777216});
}

INSTANTIATE_TEST_SUITE_P(
    OutputShapes, DequantRopeQuantKvcacheOutputShapeTiling,
    testing::Values(
        InvalidOutput("Case83QRank5", 0, {1, 1, 1, 1, 1}), InvalidOutput("Case84QRank2", 0, {1, 1}),
        InvalidOutput("Case89KRank5", 1, {1, 1, 1, 1, 1}), InvalidOutput("Case94VRank5", 2, {1, 1, 1, 1, 1}),
        InvalidOutput("QRank5SameNumel", 0, {2, 4, 1, 8, 64}), InvalidOutput("QRank2SameNumel", 0, {8, 512}),
        InvalidOutput("QWrongAxesSameNumel", 0, {2, 4, 4, 128}), InvalidOutput("KWrongAxesSameNumel", 1, {2, 4, 2, 32}),
        InvalidOutput("VWrongAxesSameNumel", 2, {2, 4, 2, 32}), NonemptyKvOutput("NonemptyKWhenDisabled", 1),
        NonemptyKvOutput("NonemptyVWhenDisabled", 2), ValidOutputs("Input3DContiguous", false, false, true),
        ValidOutputs("Input2DContiguous", true, false, true), ValidOutputs("Input3DPage", false, true, true),
        ValidOutputs("Input2DPage", true, true, true), ValidOutputs("Input3DKvDisabled", false, false, false),
        ValidOutputs("Input2DKvDisabled", true, false, false)),
    [](const testing::TestParamInfo<OutputShapeCase>& info) { return info.param.name; });
} // namespace

TEST_F(DequantRopeQuantKvcacheTiling, RejectInconsistentInputDtypes)
{
    struct DtypeCase {
        ge::DataType x;
        ge::DataType cos;
        ge::DataType sin;
    };
    const std::array<DtypeCase, 7> cases = {{{ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16},
                                             {ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16},
                                             {ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16},
                                             {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16},
                                             {ge::DT_INT32, ge::DT_FLOAT16, ge::DT_BF16},
                                             {ge::DT_INT32, ge::DT_BF16, ge::DT_FLOAT16},
                                             {ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT}}};
    for (const auto& item : cases) {
        SCOPED_TRACE(Ops::Base::ToString(item.x) + ", " + Ops::Base::ToString(item.cos) + ", " +
                     Ops::Base::ToString(item.sin));
        DequantRopeQuantKvcacheCompileInfo compileInfo = {};
        auto para = MakeValidTilingContext(&compileInfo);
        para.inputTensorDesc_[0].dtype_ = item.x;
        para.inputTensorDesc_[1].dtype_ = item.cos;
        para.inputTensorDesc_[2].dtype_ = item.sin;
        drqk_test::ClearErrorMessages();
        ExecuteTestCase(para, ge::GRAPH_FAILED);
        const std::string constraint =
            item.x == ge::DT_INT32 ?
                "cos.dtype and sin.dtype must match and be FLOAT16 or BFLOAT16 when x.dtype is INT32" :
                "x.dtype, cos.dtype and sin.dtype must be consistent when x.dtype is FLOAT16 or BFLOAT16";
        const std::string reason = constraint + ". Actual dtypes: x.dtype=" + Ops::Base::ToString(item.x) +
                                   ", cos.dtype=" + Ops::Base::ToString(item.cos) +
                                   ", sin.dtype=" + Ops::Base::ToString(item.sin);
        drqk_test::ExpectSingleError(
            "EZ0037", {{"param_name", "x, cos, sin"}, {"op_name", "DequantRopeQuantKvcache"}, {"reason", reason}});
    }
}

TEST_F(DequantRopeQuantKvcacheTiling, AcceptSupportedInputDtypeCombinations)
{
    for (auto xType : {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32}) {
        for (auto ropeType : {ge::DT_FLOAT16, ge::DT_BF16}) {
            if (xType != ge::DT_INT32 && xType != ropeType) {
                continue;
            }
            SCOPED_TRACE(Ops::Base::ToString(xType) + ", " + Ops::Base::ToString(ropeType));
            DequantRopeQuantKvcacheCompileInfo compileInfo = {};
            auto para = MakeValidTilingContext(&compileInfo);
            para.inputTensorDesc_[0].dtype_ = xType;
            para.inputTensorDesc_[1].dtype_ = ropeType;
            para.inputTensorDesc_[2].dtype_ = ropeType;
            for (size_t i = 0; i < 3; ++i) {
                para.outputTensorDesc_[i].dtype_ = ropeType;
            }
            if (xType == ge::DT_INT32) {
                para.inputTensorDesc_.emplace_back(gert::StorageShape({1280}, {1280}), ge::DT_FLOAT, ge::FORMAT_ND);
            }
            drqk_test::ClearErrorMessages();
            TilingInfo tilingInfo;
            EXPECT_TRUE(ExecuteTiling(para, tilingInfo));
            drqk_test::ExpectNoError();
        }
    }
}

TEST_F(DequantRopeQuantKvcacheTiling, RejectUnsupportedInputXDtype)
{
    DequantRopeQuantKvcacheCompileInfo compileInfo = {};
    auto para = MakeValidTilingContext(&compileInfo);
    para.inputTensorDesc_[0].dtype_ = ge::DT_FLOAT;
    drqk_test::ClearErrorMessages();
    ExecuteTestCase(para, ge::GRAPH_FAILED);
    drqk_test::ExpectSingleError("EZ0019", {{"param_name", "x"},
                                            {"op_name", "DequantRopeQuantKvcache"},
                                            {"incorrect_dtype", "DT_FLOAT"},
                                            {"correct_dtype", "FLOAT16, BFLOAT16 or INT32"}});
}

TEST_F(DequantRopeQuantKvcacheTiling, DequantRopeQuantKvcachetiling_empty)
{
    DequantRopeQuantKvcacheCompileInfo compileInfo = {};
    const gert::TilingContextPara::TensorDescription optionalInput = {{{}, {}}, ge::DT_UNDEFINED, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara(
        "DequantRopeQuantKvcache",
        {
            {{{0, 768}, {0, 768}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{0, 64}, {0, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{0, 64}, {0, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 1, 4, 64}, {4, 1, 4, 64}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{4, 1, 4, 64}, {4, 1, 4, 64}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            optionalInput,
            optionalInput,
            {{{768}, {768}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{{{0, 4, 64}, {0, 4, 64}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{0, 4, 64}, {0, 4, 64}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{0, 4, 64}, {0, 4, 64}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{4, 1, 4, 64}, {4, 1, 4, 64}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{4, 1, 4, 64}, {4, 1, 4, 64}}, ge::DT_INT8, ge::FORMAT_ND}},
        {{"size_splits", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({256, 256, 256})},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("static")},
         {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"kv_output", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
         {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("contiguous")}},
        &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 0);
    EXPECT_EQ(tilingInfo.blockNum, 1);
    ASSERT_FALSE(tilingInfo.workspaceSizes.empty());
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0U);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(DequantRopeQuantKvcacheTilingDataView));
    const auto* emptyTiling =
        reinterpret_cast<const DequantRopeQuantKvcacheTilingDataView*>(tilingInfo.tilingData.get());
    EXPECT_EQ(emptyTiling->batch * emptyTiling->seqlen, 0);
}

TEST_F(DequantRopeQuantKvcacheTiling, DequantRopeQuantKvcachetiling_empty_s_zero)
{
    DequantRopeQuantKvcacheCompileInfo compileInfo = {};
    const gert::TilingContextPara::TensorDescription optionalInput = {{{}, {}}, ge::DT_UNDEFINED, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara(
        "DequantRopeQuantKvcache",
        {
            {{{8, 0, 768}, {8, 0, 768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 0, 1, 128}, {8, 0, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 0, 1, 128}, {8, 0, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{3, 56, 2, 128}, {3, 56, 2, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{3, 56, 2, 128}, {3, 56, 2, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            optionalInput,
            optionalInput,
        },
        {{{{8, 0, 2, 128}, {8, 0, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{8, 0, 2, 128}, {8, 0, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{8, 0, 2, 128}, {8, 0, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{3, 56, 2, 128}, {3, 56, 2, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{3, 56, 2, 128}, {3, 56, 2, 128}}, ge::DT_INT8, ge::FORMAT_ND}},
        {{"size_splits", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({256, 256, 256})},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("static")},
         {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"kv_output", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
         {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("page")}},
        &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 0);
    EXPECT_EQ(tilingInfo.blockNum, 1);
    ASSERT_FALSE(tilingInfo.workspaceSizes.empty());
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0U);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(DequantRopeQuantKvcacheTilingDataView));
    const auto* emptyTiling =
        reinterpret_cast<const DequantRopeQuantKvcacheTilingDataView*>(tilingInfo.tilingData.get());
    EXPECT_EQ(emptyTiling->batch * emptyTiling->seqlen, 0);
}
