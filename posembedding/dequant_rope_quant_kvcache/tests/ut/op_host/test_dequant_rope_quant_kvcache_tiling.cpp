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
#include <gtest/gtest.h>
#include "../../../op_host/dequant_rope_quant_kvcache_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "error_code_test_utils.h"

using namespace std;

struct DequantRopeQuantKvcacheCompileInfo {};

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
         {{{1, 320, 1, 128}, {1, 320, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 320, 1, 128}, {1, 320, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
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
         {{{1, 320, 1, 128}, {1, 320, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 320, 1, 128}, {1, 320, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
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
         {{{1, 320, 1, 128}, {1, 320, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 320, 1, 128}, {1, 320, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
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
         {{{1, 320, 1, 128}, {1, 320, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 320, 1, 128}, {1, 320, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
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
         {{{1, 320, 1, 128}, {1, 320, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 320, 1, 128}, {1, 320, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
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
         {{{1, 320, 1, 128}, {1, 320, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 320, 1, 128}, {1, 320, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
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
