/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file test_scatter_pa_kv_cache_tiling.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_host/scatter_pa_kv_cache_tiling.h"

using namespace std;

class ScatterPaKvCacheTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ScatterPaKvCacheTilingTest SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "ScatterPaKvCacheTilingTest TearDown" << std::endl;
    }
};

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_key_wrong_dim)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 128}, {102, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("None")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_key_cache_wrong_dim)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 128}, {306, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("None")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_slot_mapping_wrong_dim)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102, 1}, {102, 1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("None")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_dtype_mismatch)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("None")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_unsupported_dtype)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
        },
        {
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("None")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_value_wrong_dim)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102, 128}, {102, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("None")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_value_cache_wrong_dim)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 128}, {306, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("None")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_key_dim_num_5d)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 1, 1, 128}, {102, 1, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("None")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_key_value_dim0_mismatch)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64, 1, 128}, {64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("None")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_key_value_dim1_mismatch)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102, 2, 128}, {102, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("None")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_key_value_cache_dim0_mismatch)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{200, 128, 1, 128}, {200, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("None")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_key_value_cache_dim2_mismatch)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 2, 128}, {306, 128, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("None")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_value_dtype_mismatch)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("None")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_nz_key_cache_3d)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 128}, {306, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("None")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_nz_k_head_size_not_aligned)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 100}, {102, 1, 100}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 100}, {306, 128, 1, 100}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102, 1, 100}, {102, 1, 100}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 100}, {306, 128, 1, 100}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{306, 128, 1, 100}, {306, 128, 1, 100}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 100}, {306, 128, 1, 100}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("None")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_key_value_dim_num_mismatch)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102, 1, 1, 128}, {102, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("None")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_norm_vhead_size_mismatch)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 64}, {306, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 64}, {306, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("None")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_norm_kc_vc_dim1_mismatch)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 64, 1, 128}, {306, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 64, 1, 128}, {306, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("None")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_nhsd_khead_not_aligned)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 1, 128, 100}, {306, 1, 128, 100}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 1, 128, 100}, {306, 1, 128, 100}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{306, 1, 128, 100}, {306, 1, 128, 100}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 1, 128, 100}, {306, 1, 128, 100}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("NHSD")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_nhsd_numblocks_too_small)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1, 16, 128}, {2, 1, 16, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1, 16, 128}, {2, 1, 16, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 1, 16, 128}, {2, 1, 16, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1, 16, 128}, {2, 1, 16, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("NHSD")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_nhsd_numhead_mismatch)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 2, 128, 128}, {306, 2, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 2, 128, 128}, {306, 2, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{306, 2, 128, 128}, {306, 2, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 2, 128, 128}, {306, 2, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("NHSD")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_nhsd_kc_vc_dim2_mismatch)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 1, 128, 128}, {306, 1, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 1, 64, 128}, {306, 1, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{306, 1, 128, 128}, {306, 1, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 1, 64, 128}, {306, 1, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("NHSD")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_nhsd_kv_dim0_mismatch)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 1, 128, 128}, {306, 1, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64, 1, 128}, {64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 1, 128, 128}, {306, 1, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{306, 1, 128, 128}, {306, 1, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 1, 128, 128}, {306, 1, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("NHSD")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_nhsd_kv_dim1_mismatch)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 1, 128, 128}, {306, 1, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102, 2, 128}, {102, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 1, 128, 128}, {306, 1, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{306, 1, 128, 128}, {306, 1, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 1, 128, 128}, {306, 1, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("NHSD")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_nhsd_khead_kc_dim3_mismatch)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 1, 128, 64}, {306, 1, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 1, 128, 64}, {306, 1, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{306, 1, 128, 64}, {306, 1, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 1, 128, 64}, {306, 1, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("NHSD")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_nhsd_vhead_vc_dim3_mismatch)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 1, 128, 128}, {306, 1, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 1, 128, 64}, {306, 1, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{306, 1, 128, 128}, {306, 1, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 1, 128, 64}, {306, 1, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("NHSD")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_nhsd_kc_vc_dim0_mismatch)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 1, 128, 128}, {306, 1, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{200, 1, 128, 128}, {200, 1, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{306, 1, 128, 128}, {306, 1, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{200, 1, 128, 128}, {200, 1, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("NHSD")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_compress_kc_dim2_not_1)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 2, 128}, {306, 128, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 2, 128}, {306, 128, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{306, 128, 2, 128}, {306, 128, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 2, 128}, {306, 128, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Alibi")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_compress_lens_2d)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{10, 10}, {10, 10}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Alibi")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_seq_lens_2d)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{10, 10}, {10, 10}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Alibi")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_siso_khead_kc_dim3_mismatch)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 64}, {306, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{306, 128, 1, 64}, {306, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("None")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_siso_numblocks_too_small)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 8, 1, 128}, {2, 8, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 8, 1, 128}, {2, 8, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("None")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_nz_kv_dim0_mismatch_dual)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 8, 128, 16}, {306, 8, 128, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64, 1, 128}, {64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 8, 128, 16}, {306, 8, 128, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{306, 8, 128, 16}, {306, 8, 128, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 8, 128, 16}, {306, 8, 128, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("None")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_nz_kv_dim1_mismatch)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 8, 128, 16}, {306, 8, 128, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102, 2, 128}, {102, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 8, 128, 16}, {306, 8, 128, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{306, 8, 128, 16}, {306, 8, 128, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 8, 128, 16}, {306, 8, 128, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("None")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_nz_kc_vc_dim1_mismatch)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 8, 128, 16}, {306, 8, 128, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 4, 128, 16}, {306, 4, 128, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{306, 8, 128, 16}, {306, 8, 128, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 4, 128, 16}, {306, 4, 128, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("None")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_nct_strides_size_lt_2)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Nct")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_nct_offsets_size_lt_2)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Nct")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_nct_valid_strides_offsets)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Nct")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({128, 128})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    // TEMPLATE_NORM_NCT=7, numTokens<SMALL_TOKEN -> tilingKey = 1000*7+1
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 7001);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_siso_nct_strides_size_lt_2)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Nct")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_nct_khead_kc_dim3_mismatch)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 64}, {306, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 64}, {306, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{306, 128, 1, 64}, {306, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 64}, {306, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Nct")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_nct_numblocks_too_small)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 16, 1, 128}, {2, 16, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 16, 1, 128}, {2, 16, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 16, 1, 128}, {2, 16, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 16, 1, 128}, {2, 16, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Nct")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_nct_numhead_mismatch)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 2, 128}, {306, 128, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 2, 128}, {306, 128, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{306, 128, 2, 128}, {306, 128, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 2, 128}, {306, 128, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Nct")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_nct_kc_vc_dim1_mismatch)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 64, 1, 128}, {306, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 64, 1, 128}, {306, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Nct")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_nct_kv_dim0_mismatch)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64, 1, 128}, {64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Nct")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_nct_kv_dim1_mismatch)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102, 2, 128}, {102, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Nct")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_nct_vhead_vc_dim3_mismatch)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 64}, {306, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 64}, {306, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Nct")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ScatterPaKvCacheTilingTest, tiling_error_nct_kc_vc_dim0_mismatch)
{
    optiling::ScatterPaKvCacheCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{200, 128, 1, 128}, {200, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{200, 128, 1, 128}, {200, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Nct")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
