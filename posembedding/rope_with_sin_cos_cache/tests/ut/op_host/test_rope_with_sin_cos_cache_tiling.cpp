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
 * \file test_cross_entropy_loss.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_host/rope_with_sin_cos_cache_tiling.h"

using namespace std;

struct RopeWithSinCosCacheCompileInfo {};
class RopeWithSinCosCacheTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "RopeWithSinCosCacheTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RopeWithSinCosCacheTiling TearDown" << std::endl;
    }
};

TEST_F(RopeWithSinCosCacheTiling, test_tiling_bf16_true)
{
    RopeWithSinCosCacheCompileInfo compile_info = {};
    vector<int64_t> mropeParams{0, 0, 0};
    gert::TilingContextPara tilingContextPara(
        "RopeWithSinCosCache",
        {
            {{{48, 48}, {48, 48}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{48, 256}, {48, 256}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{48, 512}, {48, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{48, 128}, {48, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{48, 256}, {48, 256}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{48, 512}, {48, 512}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"numQHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"numKHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"headSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"mropeSection", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(mropeParams)},
         {"qstride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"kstride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
         {"isNeoxStyle", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"cacheMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compile_info);
    uint64_t expectTilingKey = 20;
    string expectTilingData = "48 48 2 4 128 128 0 0 0 0 256 512 1 0 48 0 1 0 1 0 1 0 0 0 0 1 1 2 0 4 0 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RopeWithSinCosCacheTiling, test_tiling_fp16_true)
{
    RopeWithSinCosCacheCompileInfo compile_info = {};
    vector<int64_t> mropeParams{0, 0, 0};
    gert::TilingContextPara tilingContextPara(
        "RopeWithSinCosCache",
        {
            {{{48, 48}, {48, 48}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{48, 256}, {48, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{48, 512}, {48, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{48, 128}, {48, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{48, 256}, {48, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{48, 512}, {48, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"numQHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"numKHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"headSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"mropeSection", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(mropeParams)},
         {"qstride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"kstride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
         {"isNeoxStyle", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"cacheMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compile_info);
    uint64_t expectTilingKey = 21;
    string expectTilingData = "48 48 2 4 128 128 0 0 0 0 256 512 1 0 48 0 1 0 1 0 1 0 0 0 0 1 1 2 0 4 0 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RopeWithSinCosCacheTiling, test_tiling_fp32_true)
{
    RopeWithSinCosCacheCompileInfo compile_info = {};
    vector<int64_t> mropeParams{0, 0, 0};
    gert::TilingContextPara tilingContextPara(
        "RopeWithSinCosCache",
        {
            {{{48, 48}, {48, 48}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{48, 256}, {48, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{48, 512}, {48, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{48, 128}, {48, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{48, 256}, {48, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{48, 512}, {48, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"numQHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"numKHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"headSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"mropeSection", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(mropeParams)},
         {"qstride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"kstride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
         {"isNeoxStyle", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"cacheMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compile_info);
    uint64_t expectTilingKey = 22;
    string expectTilingData = "48 48 2 4 128 128 0 0 0 0 256 512 1 0 48 0 1 0 1 0 1 0 0 0 0 1 1 2 0 4 0 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RopeWithSinCosCacheTiling, test_tiling_bf16_false)
{
    RopeWithSinCosCacheCompileInfo compile_info = {};
    vector<int64_t> mropeParams{0, 0, 0};
    gert::TilingContextPara tilingContextPara(
        "RopeWithSinCosCache",
        {
            {{{48, 48}, {48, 48}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{48, 256}, {48, 256}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{48, 512}, {48, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{48, 128}, {48, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{48, 256}, {48, 256}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{48, 512}, {48, 512}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"numQHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"numKHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"headSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"mropeSection", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(mropeParams)},
         {"qstride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"kstride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
         {"isNeoxStyle", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"cacheMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compile_info);
    uint64_t expectTilingKey = 20;
    string expectTilingData = "48 48 2 4 128 128 0 0 0 0 256 512 0 0 48 0 1 0 1 0 1 0 0 0 0 1 1 2 0 4 0 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RopeWithSinCosCacheTiling, test_tiling_fp16_false)
{
    RopeWithSinCosCacheCompileInfo compile_info = {};
    vector<int64_t> mropeParams{0, 0, 0};
    gert::TilingContextPara tilingContextPara(
        "RopeWithSinCosCache",
        {
            {{{48, 48}, {48, 48}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{48, 256}, {48, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{48, 512}, {48, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{48, 128}, {48, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{48, 256}, {48, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{48, 512}, {48, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"numQHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"numKHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"headSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"mropeSection", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(mropeParams)},
         {"qstride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"kstride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
         {"isNeoxStyle", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"cacheMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compile_info);
    uint64_t expectTilingKey = 21;
    string expectTilingData = "48 48 2 4 128 128 0 0 0 0 256 512 0 0 48 0 1 0 1 0 1 0 0 0 0 1 1 2 0 4 0 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RopeWithSinCosCacheTiling, test_tiling_fp32_false)
{
    RopeWithSinCosCacheCompileInfo compile_info = {};
    vector<int64_t> mropeParams{0, 0, 0};
    gert::TilingContextPara tilingContextPara(
        "RopeWithSinCosCache",
        {
            {{{48, 48}, {48, 48}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{48, 256}, {48, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{48, 512}, {48, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{48, 128}, {48, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{48, 256}, {48, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{48, 512}, {48, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"numQHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"numKHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"headSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"mropeSection", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(mropeParams)},
         {"qstride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"kstride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
         {"isNeoxStyle", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"cacheMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compile_info);
    uint64_t expectTilingKey = 21;
    string expectTilingData = "48 48 2 4 128 128 0 0 0 0 256 512 0 0 48 0 1 0 1 0 1 0 0 0 0 1 1 2 0 4 0 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RopeWithSinCosCacheTiling, test_v2_tiling_fp32_false_cache1)
{
    RopeWithSinCosCacheCompileInfo compile_info = {};
    vector<int64_t> mropeParams{16, 24, 24};
    gert::TilingContextPara tilingContextPara(
        "RopeWithSinCosCache",
        {
            {{{48, 48}, {48, 48}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{48, 256}, {48, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{48, 512}, {48, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{48, 128}, {48, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{48, 256}, {48, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{48, 512}, {48, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"numQHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"numKHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"headSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"mropeSection", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(mropeParams)},
         {"qstride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"kstride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
         {"isNeoxStyle", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"cacheMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compile_info);
    uint64_t expectTilingKey = 21;
    string expectTilingData = "48 48 2 4 128 128 16 24 24 0 256 512 0 1 48 0 1 0 1 0 1 0 0 0 0 1 1 2 0 4 0 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RopeWithSinCosCacheTiling, test_v2_tiling_fp32_false_cache0_sectionsize4)
{
    RopeWithSinCosCacheCompileInfo compile_info = {};
    vector<int64_t> mropeParams{16, 16, 16, 16};
    gert::TilingContextPara tilingContextPara(
        "RopeWithSinCosCache",
        {
            {{{48, 48}, {48, 48}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{48, 256}, {48, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{48, 512}, {48, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{48, 128}, {48, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{48, 256}, {48, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{48, 512}, {48, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"numQHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"numKHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"headSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"mropeSection", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(mropeParams)},
         {"qstride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"kstride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
         {"isNeoxStyle", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"cacheMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compile_info);
    uint64_t expectTilingKey = 21;
    string expectTilingData = "48 48 2 4 128 128 16 16 16 16 256 512 0 0 48 0 1 0 1 0 1 0 0 0 0 1 1 2 0 4 0 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
