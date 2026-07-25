/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_host/kv_rms_norm_rope_cache_tiling.h"

using namespace std;

namespace {
const gert::TilingContextPara::TensorDescription kOptionalInput = {{{}, {}}, ge::DT_UNDEFINED, ge::FORMAT_ND};
}

class KvRmsNormRopeCacheTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "KvRmsNormRopeCacheTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "KvRmsNormRopeCacheTiling TearDown" << std::endl;
    }
};

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_5011A)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{64, 1, 1, 576}, {64, 1, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         kOptionalInput},
        {{{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 512}, {64, 1, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BNSD")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend910B", 48, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 5011, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_5011B)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{32, 1, 15, 192}, {32, 1, 15, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{32, 1, 15, 64}, {32, 1, 15, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{32, 1, 15, 64}, {32, 1, 15, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{480}, {480}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{125, 10, 1, 192}, {125, 10, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{125, 10, 1, 128}, {125, 10, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         kOptionalInput,
         {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         {{{32, 1, 15, 128}, {32, 1, 15, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{125, 10, 1, 192}, {125, 10, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{125, 10, 1, 128}, {125, 10, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{32, 1, 15, 192}, {32, 1, 15, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{32, 1, 15, 128}, {32, 1, 15, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BNSD")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend910B", 48, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 5011, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_5010A)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{64, 1, 1, 576}, {64, 1, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         kOptionalInput},
        {{{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 512}, {64, 1, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BLK_BNSD")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend910B", 48, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 5010, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_5010B)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{9, 1, 14, 192}, {9, 1, 14, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{9, 1, 14, 64}, {9, 1, 14, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{9, 1, 14, 64}, {9, 1, 14, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{9}, {9}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{21, 16, 1, 192}, {21, 16, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{21, 16, 1, 128}, {21, 16, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         kOptionalInput,
         {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         {{{9, 1, 14, 128}, {9, 1, 14, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{21, 16, 1, 192}, {21, 16, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{21, 16, 1, 128}, {21, 16, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{9, 1, 14, 192}, {9, 1, 14, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{9, 1, 14, 128}, {9, 1, 14, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BLK_BNSD")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend910B", 48, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 5010, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_5001A)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{64, 1, 1, 576}, {64, 1, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput},
        {{{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 512}, {64, 1, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BNSD")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo, "Ascend910B", 48, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 5001, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_5001B)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{30, 1, 10, 192}, {30, 1, 10, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{30, 1, 10, 64}, {30, 1, 10, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{30, 1, 10, 64}, {30, 1, 10, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{300}, {300}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{42, 12, 1, 192}, {42, 12, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{42, 12, 1, 128}, {42, 12, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         {{{30, 1, 10, 128}, {30, 1, 10, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{42, 12, 1, 192}, {42, 12, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{42, 12, 1, 128}, {42, 12, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{30, 1, 10, 192}, {30, 1, 10, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{30, 1, 10, 128}, {30, 1, 10, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BNSD")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo, "Ascend910B", 48, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 5001, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_5000A)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{38, 1, 3809, 576}, {38, 1, 3809, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{38, 1, 3809, 64}, {38, 1, 3809, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{38, 1, 3809, 64}, {38, 1, 3809, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{228}, {228}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{230, 745, 1, 64}, {230, 745, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{230, 745, 1, 512}, {230, 745, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput},
        {{{{230, 745, 1, 64}, {230, 745, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{230, 745, 1, 512}, {230, 745, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{38, 1, 3809, 64}, {38, 1, 3809, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{38, 1, 3809, 512}, {38, 1, 3809, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BLK_BNSD")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend910B", 48, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 5000, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_5000B)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{9, 1, 14, 192}, {9, 1, 14, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{9, 1, 14, 64}, {9, 1, 14, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{9, 1, 14, 64}, {9, 1, 14, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{9}, {9}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{21, 16, 1, 192}, {21, 16, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{21, 16, 1, 128}, {21, 16, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         {{{9, 1, 14, 128}, {9, 1, 14, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{21, 16, 1, 192}, {21, 16, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{21, 16, 1, 128}, {21, 16, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{9, 1, 14, 192}, {9, 1, 14, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{9, 1, 14, 128}, {9, 1, 14, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BLK_BNSD")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend910B", 48, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 5000, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_4011A)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{64, 1, 1, 576}, {64, 1, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         kOptionalInput},
        {{{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 512}, {64, 1, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend910B", 48, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 4011, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_4011B)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{64, 1, 1, 192}, {64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192}, {192}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 192}, {64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend910B", 48, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 4011, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_4010A)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{64, 1, 1, 576}, {64, 1, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         kOptionalInput},
        {{{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 512}, {64, 1, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BLK_NZ")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend910B", 48, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 4010, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_4010B)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{64, 1, 1, 192}, {64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192}, {192}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 192}, {64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BLK_NZ")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend910B", 48, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 4010, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_4001A)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{64, 1, 1, 576}, {64, 1, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput},
        {{{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 512}, {64, 1, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend910B", 48, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 4001, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_4001B)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{64, 1, 1, 192}, {64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 192}, {64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend910B", 48, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 4001, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_4000A)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{38, 1, 3809, 576}, {38, 1, 3809, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{38, 1, 3809, 64}, {38, 1, 3809, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{38, 1, 3809, 64}, {38, 1, 3809, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{228}, {228}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{230, 745, 1, 64}, {230, 745, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{230, 745, 1, 512}, {230, 745, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput},
        {{{{230, 745, 1, 64}, {230, 745, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{230, 745, 1, 512}, {230, 745, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{38, 1, 3809, 64}, {38, 1, 3809, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{38, 1, 3809, 512}, {38, 1, 3809, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BLK_NZ")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend910B", 48, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 4000, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_4000B)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{38, 1, 3809, 192}, {38, 1, 3809, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{38, 1, 3809, 64}, {38, 1, 3809, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{38, 1, 3809, 64}, {38, 1, 3809, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{228}, {228}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{230, 745, 1, 192}, {230, 745, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{230, 745, 1, 128}, {230, 745, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         {{{38, 1, 3809, 128}, {38, 1, 3809, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{230, 745, 1, 192}, {230, 745, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{230, 745, 1, 128}, {230, 745, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{38, 1, 3809, 192}, {38, 1, 3809, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{38, 1, 3809, 128}, {38, 1, 3809, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BLK_NZ")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend910B", 48, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 4000, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_3010_SYMMETRIC_QUANT)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{64, 1, 7, 192}, {64, 1, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 7, 64}, {64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 7, 64}, {64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 7}, {64, 7}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         kOptionalInput,
         {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         {{{64, 1, 7, 128}, {64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 7, 192}, {64, 1, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 7, 128}, {64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo, "Ascend910B", 48, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 3010, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_3010_ASYMMETRIC_QUANT)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{64, 1, 7, 192}, {64, 1, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 7, 64}, {64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 7, 64}, {64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 7}, {64, 7}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         kOptionalInput,
         {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
         kOptionalInput,
         {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{64, 1, 7, 128}, {64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 7, 192}, {64, 1, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 7, 128}, {64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo, "Ascend910B", 48, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 3010, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_3001A)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{64, 1, 7, 576}, {64, 1, 7, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 7, 64}, {64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 7, 64}, {64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 7}, {64, 7}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput},
        {{{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 7, 64}, {64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 7, 512}, {64, 1, 7, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BNSD")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend910B", 48, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 3001, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_3001B)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{64, 1, 7, 192}, {64, 1, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 7, 64}, {64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 7, 64}, {64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 7}, {64, 7}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         {{{64, 1, 7, 128}, {64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 7, 192}, {64, 1, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 7, 128}, {64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BNSD")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend910B", 48, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 3001, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_3000A)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{64, 1, 7, 576}, {64, 1, 7, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 7, 64}, {64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 7, 64}, {64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 7}, {64, 7}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput},
        {{{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 7, 64}, {64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 7, 512}, {64, 1, 7, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BNSD")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo, "Ascend910B", 48, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 5001, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_3000B)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{64, 1, 7, 192}, {64, 1, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 7, 64}, {64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 7, 64}, {64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 7}, {64, 7}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         {{{64, 1, 7, 128}, {64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 7, 192}, {64, 1, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 7, 128}, {64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BNSD")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo, "Ascend910B", 48, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 5001, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_2001A)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{64, 1, 7, 576}, {64, 1, 7, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 7}, {64, 7}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput},
        {{{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 7, 64}, {64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 7, 512}, {64, 1, 7, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend910B", 48, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 2001, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_2001B)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{64, 1, 7, 192}, {64, 1, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 7}, {64, 7}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         {{{64, 1, 7, 128}, {64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 192}, {64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 7, 128}, {64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend910B", 48, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 2001, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_2000A)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{64, 1, 1, 576}, {64, 1, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1}, {64, 1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{192, 128, 1, 64}, {192, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192, 128, 1, 512}, {192, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput},
        {{{{192, 128, 1, 64}, {192, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192, 128, 1, 512}, {192, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 512}, {64, 1, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend910B", 48, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 2000, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_2000B)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{64, 1, 1, 192}, {64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1}, {64, 1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 192}, {64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend910B", 48, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 2000, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_1011B)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{64, 1, 7, 192}, {64, 1, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 7}, {64, 7}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         kOptionalInput,
         {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         {{{64, 1, 7, 128}, {64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 7, 192}, {64, 1, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 7, 128}, {64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BNSD")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo, "Ascend910B", 48, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 5011, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_1010B)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{64, 1, 1, 192}, {64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1}, {64, 1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         kOptionalInput,
         {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 192}, {64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BNSD")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo, "Ascend910B", 48, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 5011, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_1000A)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{64, 1, 1, 576}, {64, 1, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1}, {64, 1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput},
        {{{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 512}, {64, 1, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BNSD")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo, "Ascend910B", 48, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 5001, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_1000B)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{64, 1, 1, 192}, {64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1}, {64, 1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BNSD")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo, "Ascend910B", 48, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 5001, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_1001B)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{64, 1, 7, 192}, {64, 1, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 7}, {64, 7}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{192, 128, 1, 192}, {192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         {{{64, 1, 7, 128}, {64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{192, 128, 7, 192}, {192, 128, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 7, 128}, {64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BNSD")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo, "Ascend910B", 48, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 5001, "");
}


TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_d_full_load_is_output_kv_tilingB)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{64, 1, 1, 192}, {64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{128}, {128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1}, {64, 1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         kOptionalInput,
         {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{64, 1, 1, 192}, {64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo, "Ascend950", 64, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10000, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_d_full_load_is_output_kv_with_PA_tilingA)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{64, 1, 1, 576}, {64, 1, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1}, {64, 1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
         kOptionalInput},
        {{{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 512}, {64, 1, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo, "Ascend950", 64, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10000, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_d_full_load_is_output_kv_with_PA_tilingB)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{64, 1, 1, 192}, {64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{128}, {128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1}, {64, 1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{192, 128, 1, 64}, {192, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         kOptionalInput,
         {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
         kOptionalInput,
         {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{192, 128, 1, 64}, {192, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{192, 128, 1, 128}, {192, 128, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{64, 1, 1, 192}, {64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo, "Ascend950", 64, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10000, "");
}


TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_norm_cache_bn_match_tiling)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{4, 1, 1, 192}, {4, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{128}, {128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 64}, {4, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 64}, {4, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1}, {4, 1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{4, 1, 128, 64}, {4, 1, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 128, 128}, {4, 1, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput},
        {{{{4, 1, 128, 64}, {4, 1, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 128, 128}, {4, 1, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 64}, {4, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 128}, {4, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend950", 64, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10000, "");
}

// The B and N axes of the cache may be split differently from kv, as long as their product still covers B * N of kv:
// [1, 4, S, D] and [4, 1, S, D] are the same linear buffer to the kernel.
TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_norm_cache_bn_merged_tiling)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{4, 1, 1, 192}, {4, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{128}, {128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 64}, {4, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 64}, {4, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1}, {4, 1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{1, 4, 128, 64}, {1, 4, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 4, 128, 128}, {1, 4, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput},
        {{{{1, 4, 128, 64}, {1, 4, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 4, 128, 128}, {1, 4, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 64}, {4, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 128}, {4, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend950", 64, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10000, "");
}

// B * N of the cache is smaller than B * N of kv: the kernel would scatter past the end of the cache, so tiling has to
// reject the shape instead of letting the kernel write out of bounds.
TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_norm_cache_bn_too_small_tiling)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{4, 1, 1, 192}, {4, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{128}, {128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 64}, {4, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 64}, {4, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1}, {4, 1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{1, 1, 128, 64}, {1, 1, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 1, 128, 128}, {1, 1, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput},
        {{{{1, 1, 128, 64}, {1, 1, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 1, 128, 128}, {1, 1, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 64}, {4, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 128}, {4, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend950", 64, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Recompute 模板（dv 大到 D 无法全载，走 TilingKey 20000）+ PA_NZ：dk / 2 是 dk0(fp16 为 16) 的整数倍，
// RoPE 的前后半区都落在分形块边界上 → 放行
TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_recompute_nz_dk_aligned_tiling)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{4, 1, 1, 32832}, {4, 1, 1, 32832}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{32768}, {32768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 64}, {4, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 64}, {4, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1}, {4, 1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{8, 128, 1, 64}, {8, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{8, 128, 1, 32768}, {8, 128, 1, 32768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput},
        {{{{8, 128, 1, 64}, {8, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{8, 128, 1, 32768}, {8, 128, 1, 32768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 64}, {4, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 32768}, {4, 1, 1, 32768}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend950", 64, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 20000, "");
}

// 同上，但 dk = 16：dk1 = dk / dk0 = 1 为奇数，满足 base 的 dk % dk0 == 0，却不满足 dk % (2 * dk0) == 0。
// recompute 模板改为逐 dk0 子块写出（后半区按错位 r = dk0 / 2 拆两半分别落 GM）之后，
// RoPE 后半区不再需要落在 2 * dk0 边界上，CheckNzHalfDkAligned 的约束已去掉
// → tiling 放行并走 recompute（TilingKey 20000）
TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_recompute_nz_odd_dk1_tiling)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{4, 1, 1, 32784}, {4, 1, 1, 32784}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{32768}, {32768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 16}, {4, 1, 1, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 16}, {4, 1, 1, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1}, {4, 1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{8, 128, 1, 16}, {8, 128, 1, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{8, 128, 1, 32768}, {8, 128, 1, 32768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput},
        {{{{8, 128, 1, 16}, {8, 128, 1, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{8, 128, 1, 32768}, {8, 128, 1, 32768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 16}, {4, 1, 1, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 32768}, {4, 1, 1, 32768}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend950", 64, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 20000,
                    "* * * * * * * * * * * * * 16 * * 3968 * * 16 1 * * * * ");
}

// dk = 5200 > ubFactor：dk 轴要切成多个 tile（ubFactorDkLoopCountCeil > 1）。
// 原实现只在单 tile 下把后半区拼成连续 dk1 分形块整行写出，跨 tile 会静默出错，
// 这里直接钉住 dk / ubFactorDkTail / ubFactorDkLoopCountCeil 三个字段，其余字段通配。
TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_recompute_nz_multi_tile_tiling)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{4, 1, 1, 37968}, {4, 1, 1, 37968}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{32768}, {32768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 5200}, {4, 1, 1, 5200}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 5200}, {4, 1, 1, 5200}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1}, {4, 1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{8, 128, 1, 5200}, {8, 128, 1, 5200}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{8, 128, 1, 32768}, {8, 128, 1, 32768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput},
        {{{{8, 128, 1, 5200}, {8, 128, 1, 5200}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{8, 128, 1, 32768}, {8, 128, 1, 32768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 5200}, {4, 1, 1, 5200}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 32768}, {4, 1, 1, 32768}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend950", 64, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 20000,
                    "* * * * * * * * * * * * * 5200 * * 3968 * * 1232 2 * * * * ");
}

// dk = 13008：dk 轴切成 4 个 tile（3 个满 tile 3968 + 尾 1104），中间存在完整的"非首非尾" tile，
// 是首尾特判之外唯一能暴露跨 tile 携带状态的形状
TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_recompute_nz_multi_tile_full_middle_tiling)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{4, 1, 1, 45776}, {4, 1, 1, 45776}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{32768}, {32768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 13008}, {4, 1, 1, 13008}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 13008}, {4, 1, 1, 13008}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1}, {4, 1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{8, 128, 1, 13008}, {8, 128, 1, 13008}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{8, 128, 1, 32768}, {8, 128, 1, 32768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput},
        {{{{8, 128, 1, 13008}, {8, 128, 1, 13008}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{8, 128, 1, 32768}, {8, 128, 1, 32768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 13008}, {4, 1, 1, 13008}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 32768}, {4, 1, 1, 32768}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend950", 64, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 20000,
                    "* * * * * * * * * * * * * 13008 * * 3968 * * 1104 4 * * * * ");
}

// PA_BLK_NZ 是 kernel 里与 PA_NZ 并列的另一个 scatter 分支，本 PR 两个分支一起改了，
// tiling 侧的放行口径也应一并覆盖
TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_recompute_blk_nz_odd_dk1_tiling)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{4, 1, 1, 32784}, {4, 1, 1, 32784}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{32768}, {32768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 16}, {4, 1, 1, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 16}, {4, 1, 1, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1}, {4, 1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{8, 128, 1, 16}, {8, 128, 1, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{8, 128, 1, 32768}, {8, 128, 1, 32768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput},
        {{{{8, 128, 1, 16}, {8, 128, 1, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{8, 128, 1, 32768}, {8, 128, 1, 32768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 16}, {4, 1, 1, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 32768}, {4, 1, 1, 32768}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BLK_NZ")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend950", 64, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 20000, "");
}

// bf16：recompute 分支此前在 tiling UT 里只有 fp16，dk0 与 fp16 同为 16，奇 dk1 口径应一致
TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_recompute_nz_odd_dk1_bf16_tiling)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{4, 1, 1, 32784}, {4, 1, 1, 32784}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{32768}, {32768}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{4, 1, 1, 16}, {4, 1, 1, 16}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{4, 1, 1, 16}, {4, 1, 1, 16}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{4, 1}, {4, 1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{8, 128, 1, 16}, {8, 128, 1, 16}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{8, 128, 1, 32768}, {8, 128, 1, 32768}}, ge::DT_BF16, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput},
        {{{{8, 128, 1, 16}, {8, 128, 1, 16}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{8, 128, 1, 32768}, {8, 128, 1, 32768}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{4, 1, 1, 16}, {4, 1, 1, 16}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{4, 1, 1, 32768}, {4, 1, 1, 32768}}, ge::DT_BF16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend950", 64, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 20000, "");
}

// int8 量化 cache（dk0 = 32B / 1B = 32，故取 dk = 96 使 dk1 = 3 为奇数）：kernel 里 scatter 按 dtype
// 用 if constexpr 分派到 kQuantLocal 分支，
// 与非量化分支是两条独立路径，tiling 放行口径同样要覆盖
TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_recompute_nz_odd_dk1_int8_tiling)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{4, 1, 1, 32864}, {4, 1, 1, 32864}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{32768}, {32768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 96}, {4, 1, 1, 96}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 96}, {4, 1, 1, 96}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1}, {4, 1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{8, 128, 1, 96}, {8, 128, 1, 96}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{8, 128, 1, 32768}, {8, 128, 1, 32768}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{96}, {96}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{32768}, {32768}}, ge::DT_FLOAT, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         kOptionalInput},
        {{{{8, 128, 1, 96}, {8, 128, 1, 96}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{8, 128, 1, 32768}, {8, 128, 1, 32768}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{4, 1, 1, 96}, {4, 1, 1, 96}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 32768}, {4, 1, 1, 32768}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend950", 64, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 20000, "");
}

// dk = 48（dk1 = 3，奇数）：dk 是 dk0(fp16 为 16)的整数倍，odd-dk1 路径成立 -> 放行。
// 与下面 dk = 40 的用例只差 dk 一个变量，用来把"拦不拦"的原因锁死在 dk 是否整除 dk0 上
TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_recompute_nz_odd_dk1_dk48_tiling)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{4, 1, 1, 32816}, {4, 1, 1, 32816}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{32768}, {32768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 48}, {4, 1, 1, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 48}, {4, 1, 1, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1}, {4, 1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{8, 128, 1, 48}, {8, 128, 1, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{8, 128, 1, 32768}, {8, 128, 1, 32768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput},
        {{{{8, 128, 1, 48}, {8, 128, 1, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{8, 128, 1, 32768}, {8, 128, 1, 32768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 48}, {4, 1, 1, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 32768}, {4, 1, 1, 32768}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend950", 64, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 20000, "");
}

// dk = 40：不是 dk0(fp16 为 16)的整数倍。odd-dk1 的 r = dk0 / 2、m = halfDk / dk0 依赖
// halfDk = m * dk0 + r，只有 dk % dk0 == 0 时才成立，因此这类 shape 必须在 tiling 就拦掉。
// 拦截点在 CheckCosSinValid：NZ 模式下要求 sin 的 D 维（== dk）整除 FP16_BLOCK_ALIGN_NUM(16)
TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_recompute_nz_dk_not_multiple_of_dk0_tiling)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{4, 1, 1, 32808}, {4, 1, 1, 32808}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{32768}, {32768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 40}, {4, 1, 1, 40}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 40}, {4, 1, 1, 40}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1}, {4, 1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{8, 128, 1, 40}, {8, 128, 1, 40}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{8, 128, 1, 32768}, {8, 128, 1, 32768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput},
        {{{{8, 128, 1, 40}, {8, 128, 1, 40}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{8, 128, 1, 32768}, {8, 128, 1, 32768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 40}, {4, 1, 1, 40}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 32768}, {4, 1, 1, 32768}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend950", 64, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// int8 cache 的 dk0 是 32（32B / 1B），dk = 48 对 fp16 合法、对 int8 非法。
// 与上面 fp16 dk = 48 放行的用例形成对照，证明这条约束是按 cache dtype 取 dk0 的，不是巧合
TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_recompute_nz_int8_dk_not_multiple_of_dk0_tiling)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{4, 1, 1, 32816}, {4, 1, 1, 32816}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{32768}, {32768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 48}, {4, 1, 1, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 48}, {4, 1, 1, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1}, {4, 1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{8, 128, 1, 48}, {8, 128, 1, 48}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{8, 128, 1, 32768}, {8, 128, 1, 32768}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{48}, {48}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{32768}, {32768}}, ge::DT_FLOAT, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         kOptionalInput},
        {{{{8, 128, 1, 48}, {8, 128, 1, 48}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{8, 128, 1, 32768}, {8, 128, 1, 32768}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{4, 1, 1, 48}, {4, 1, 1, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 32768}, {4, 1, 1, 32768}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend950", 64, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// dv = 32760 不是 C0(fp16 为 16)的整数倍。NZ 下 ckv_cache 的尾轴同样要求按 C0 分形对齐，
// 校验点在 CheckGammaValid：gamma 的 0 轴(== dv)必须整除 FP16_BLOCK_ALIGN_NUM
TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_recompute_nz_dv_not_multiple_of_c0_tiling)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{4, 1, 1, 32824}, {4, 1, 1, 32824}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{32760}, {32760}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 64}, {4, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 64}, {4, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1}, {4, 1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{8, 128, 1, 64}, {8, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{8, 128, 1, 32760}, {8, 128, 1, 32760}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput},
        {{{{8, 128, 1, 64}, {8, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{8, 128, 1, 32760}, {8, 128, 1, 32760}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 64}, {4, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 32760}, {4, 1, 1, 32760}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend950", 64, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// cacheRowLimit is the sole host-side source of the kernel's index upper bound
// (`if (seqIndex >= 0 && seqIndex < cacheRowLimit)`), so its value is asserted directly here:
// GetCacheRowLimit gives cacheLength for Norm and blockNum * blockSize for the PA family.
TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_norm_cache_row_limit_tiling)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{4, 1, 1, 192}, {4, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{128}, {128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 64}, {4, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 64}, {4, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1}, {4, 1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{4, 1, 128, 64}, {4, 1, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 128, 128}, {4, 1, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput},
        {{{{4, 1, 128, 64}, {4, 1, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 128, 128}, {4, 1, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 64}, {4, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 128}, {4, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend950", 64, 196608);

    // Norm: cacheRowLimit == cacheLength (the S axis of k_cache) == 128
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10000,
                    "* * * * * * * * * * * * * * * * * * * * * * * * * 128 * ");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_pa_cache_row_limit_tiling)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{4, 1, 1, 192}, {4, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{128}, {128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 64}, {4, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 64}, {4, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{8, 16, 1, 64}, {8, 16, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{8, 16, 1, 128}, {8, 16, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput,
         kOptionalInput},
        {{{{8, 16, 1, 64}, {8, 16, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{8, 16, 1, 128}, {8, 16, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 64}, {4, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 128}, {4, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend950", 64, 196608);

    // PA: cacheRowLimit == blockNum * blockSize == 8 * 16 == 128
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10000,
                    "* * * * * * * * * * * * * * * * * * * * * * * * * 128 * ");
}

// outQueue 的 UB 记账必须按 cache 的 dtype 是否量化来判，而不是按可选输入 scale 是否存在：
// 量化 cache 一定带 scale，但带了 scale 的 cache 未必是量化 dtype（CheckInputDtype 只约束前一个方向）。
// 本用例即后者——ckv_cache 是 fp16 却传了 c_kv_scale，此时 kernel 走 RmsNormVF 按 T_KV(2B/元素) 写
// outQueue，故 outUbSize 必须是 dvAlign * sizeof(T_KV) = 128 * 2 = 256，而非量化路径的 dvB8Align * 1B = 128。
TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_non_quant_cache_with_scale_out_ub_tiling)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {{{{4, 1, 1, 192}, {4, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{128}, {128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 64}, {4, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 64}, {4, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1}, {4, 1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{4, 1, 128, 64}, {4, 1, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 128, 128}, {4, 1, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         kOptionalInput,
         {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
         kOptionalInput,
         kOptionalInput,
         kOptionalInput},
        {{{{4, 1, 128, 64}, {4, 1, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 128, 128}, {4, 1, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 64}, {4, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 128}, {4, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend950", 64, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10000,
                    "* * * * * * * * * * * * * * * * * * * * * * * 256 * * * ");
}
