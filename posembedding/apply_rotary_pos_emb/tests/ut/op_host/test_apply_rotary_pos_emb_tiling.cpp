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
#include "../../../op_host/apply_rotary_pos_emb_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class ApplyRotaryPosEmbTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ApplyRotaryPosEmbTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ApplyRotaryPosEmbTiling TearDown" << std::endl;
    }
};

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp16_samll_right)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {// attr
                                               {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                               {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")}},
                                              &compileInfo, "Ascend910B", 40, 196608);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "24 128 64 0 0 0 0 0 0 0 3072 3072 256 256 0 0 0 0 0 1408 128 128 1408 128 128 12 1536 1 "
                              "8 4 0 0 128 11 1 128 128 8 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_bf16_samll_right)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {// attr
                                               {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                               {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")}},
                                              &compileInfo, "Ascend910B", 40, 196608);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "24 128 64 0 0 0 0 0 0 0 6144 6144 256 512 0 0 0 0 0 1408 128 128 1408 128 128 12 1536 2 "
                              "16 8 0 0 64 11 1 128 128 8 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp32_samll_right)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {// attr
                                               {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                               {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")}},
                                              &compileInfo, "Ascend910B", 40, 196608);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "24 128 64 0 0 0 0 0 0 0 6144 6144 512 512 0 0 0 0 0 1408 128 128 1408 128 128 12 1536 2 "
                              "16 8 0 0 64 11 1 128 128 16 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp16_compute_ab_right)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "ApplyRotaryPosEmb",
        {
            // input info
            {{{640, 1, 15, 128}, {640, 1, 15, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{640, 1, 15, 128}, {640, 1, 15, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{640, 1, 1, 128}, {640, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{640, 1, 1, 128}, {640, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // output info
            {{{640, 1, 15, 128}, {640, 1, 15, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{640, 1, 15, 128}, {640, 1, 15, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {// attr
         {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")}},
        &compileInfo, "Ascend910B", 40, 196608);
    uint64_t expectTilingKey = 3;
    string expectTilingData = "40 128 64 5 1 1 4 1 1 1 38400 30720 1280 1024 3 3 1 0 0 30720 30720 2048 1920 1920 128 "
                              "30 240 960 8 120 120 4 128 15 15 128 128 8 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp32_compute_ab_right)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{640, 1, 18, 128}, {640, 1, 18, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{640, 1, 2, 128}, {640, 1, 2, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{640, 1, 1, 128}, {640, 1, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{640, 1, 1, 128}, {640, 1, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{640, 1, 18, 128}, {640, 1, 18, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{640, 1, 2, 128}, {640, 1, 2, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {// attr
                                               {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                               {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")}},
                                              &compileInfo, "Ascend910B", 40, 196608);
    uint64_t expectTilingKey = 3;
    string expectTilingData = "40 128 64 4 4 4 3 1 1 1 40960 30720 2048 1536 3 3 1 1 1 36864 4096 2048 2304 256 128 20 "
                              "320 1152 16 288 32 8 64 18 2 128 128 16 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp16_compute_ab_cast_right)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{64, 1, 15, 128}, {64, 1, 15, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{64, 1, 15, 128}, {64, 1, 15, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{64, 1, 15, 128}, {64, 1, 15, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{64, 1, 15, 128}, {64, 1, 15, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {// attr
                                               {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                               {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")}},
                                              &compileInfo, "Ascend910B", 40, 196608);
    uint64_t expectTilingKey = 4;
    string expectTilingData = "32 128 64 2 2 2 2 2 2 2 15360 15360 512 512 0 0 0 0 0 3840 3840 256 1920 1920 128 30 "
                              "240 960 8 120 120 4 128 15 15 128 128 8 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp32_compute_ab_cast_right)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{64, 1, 18, 128}, {64, 1, 18, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{64, 1, 2, 128}, {64, 1, 2, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{64, 1, 18, 128}, {64, 1, 18, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{64, 1, 2, 128}, {64, 1, 2, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {// attr
                                               {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                               {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")}},
                                              &compileInfo, "Ascend910B", 40, 196608);
    uint64_t expectTilingKey = 4;
    string expectTilingData = "32 128 64 2 2 2 2 2 2 2 20480 20480 1024 1024 0 0 0 0 0 4608 512 256 2304 256 128 20 "
                              "320 1152 16 288 32 8 64 18 2 128 128 16 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_bf16_compute_ab_cast_right)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{64, 1, 18, 128}, {64, 1, 18, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{64, 1, 3, 128}, {64, 1, 3, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{64, 1, 18, 128}, {64, 1, 18, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{64, 1, 3, 128}, {64, 1, 3, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {// attr
                                               {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                               {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")}},
                                              &compileInfo, "Ascend910B", 40, 196608);
    uint64_t expectTilingKey = 4;
    string expectTilingData = "32 128 64 2 2 2 2 2 2 2 21504 21504 512 1024 0 0 0 0 0 4608 768 256 2304 384 128 21 336 "
                              "1152 16 144 24 8 64 18 3 128 128 8 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_bf16_samll_TND)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{24, 11, 128}, {24, 11, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{24, 1, 128}, {24, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{24, 1, 128}, {24, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{24, 1, 128}, {24, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{24, 11, 128}, {24, 11, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{24, 1, 128}, {24, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                              },
                                              &compileInfo, "Ascend910B", 40, 196608);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "24 128 64 0 0 0 0 0 0 0 6144 6144 256 512 0 0 0 0 0 1408 128 128 1408 128 128 12 1536 2 "
                              "16 8 0 0 64 11 1 128 128 8 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp32_compute_ab_TND)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{1024, 20, 128}, {1024, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1024, 20, 128}, {1024, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1024, 1, 128}, {1024, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1024, 1, 128}, {1024, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{1024, 20, 128}, {1024, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1024, 20, 128}, {1024, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                              },
                                              &compileInfo, "Ascend910B", 40, 196608);
    uint64_t expectTilingKey = 3;
    string expectTilingData = "40 128 64 2 2 2 1 1 1 1 40960 20480 1024 512 12 4 1 1 1 66560 66560 3328 2560 2560 128 "
                              "40 640 1280 16 320 320 8 64 20 20 128 128 16 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp16_compute_ab_TND)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{4096, 1, 128}, {4096, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{4096, 1, 128}, {4096, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                              },
                                              &compileInfo, "Ascend910B", 40, 196608);
    uint64_t expectTilingKey = 3;
    string expectTilingData = "40 128 64 4 3 3 3 1 3 3 40960 30720 1024 768 25 19 1 0 0 263680 263680 13184 2560 2560 "
                              "128 40 320 1280 8 160 160 4 128 20 20 128 128 8 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_bf16_compute_ab_cast_TND)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{4096, 16, 128}, {4096, 16, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4096, 16, 128}, {4096, 16, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4096, 1, 128}, {4096, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4096, 1, 128}, {4096, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{4096, 16, 128}, {4096, 16, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4096, 16, 128}, {4096, 16, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                              },
                                              &compileInfo, "Ascend910B", 40, 196608);
    uint64_t expectTilingKey = 4;
    string expectTilingData = "40 128 64 2 1 1 2 2 1 1 32768 32768 512 1024 51 39 0 0 0 210944 210944 13184 2048 2048 "
                              "128 32 512 1024 16 128 128 8 64 16 16 128 128 8 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp16_small_D64)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{24, 1, 11, 64}, {24, 1, 11, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 64}, {24, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 64}, {24, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 64}, {24, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{24, 1, 11, 64}, {24, 1, 11, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 64}, {24, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {// attr
                                               {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                               {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")}},
                                              &compileInfo, "Ascend910B", 40, 196608);
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "24 64 32 0 0 0 0 0 0 0 1536 1536 128 128 0 0 0 0 0 704 64 64 704 64 64 12 768 1 4 2 0 0 64 11 1 64 64 4 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp32_small_D64)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{24, 1, 11, 64}, {24, 1, 11, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 64}, {24, 1, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 64}, {24, 1, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 64}, {24, 1, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{24, 1, 11, 64}, {24, 1, 11, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 64}, {24, 1, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {// attr
                                               {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                               {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")}},
                                              &compileInfo, "Ascend910B", 40, 196608);
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "24 64 32 0 0 0 0 0 0 0 3072 3072 256 256 0 0 0 0 0 704 64 64 704 64 64 12 768 1 8 4 0 0 64 11 1 64 64 8 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_bf16_small_D64)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{24, 1, 11, 64}, {24, 1, 11, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 64}, {24, 1, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 64}, {24, 1, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 64}, {24, 1, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{24, 1, 11, 64}, {24, 1, 11, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 64}, {24, 1, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {// attr
                                               {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                               {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")}},
                                              &compileInfo, "Ascend910B", 40, 196608);
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "24 64 32 0 0 0 0 0 0 0 3072 3072 128 256 0 0 0 0 0 704 64 64 704 64 64 12 768 1 8 4 0 0 64 11 1 64 64 4 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp16_compute_ab_D64)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "ApplyRotaryPosEmb",
        {
            // input info
            {{{4096, 60, 17, 64}, {4096, 60, 17, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 60, 11, 64}, {4096, 60, 11, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 60, 1, 64}, {4096, 60, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 60, 1, 64}, {4096, 60, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // output info
            {{{4096, 60, 17, 64}, {4096, 60, 17, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 60, 11, 64}, {4096, 60, 11, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {// attr
         {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")}},
        &compileInfo, "Ascend910B", 40, 196608);
    uint64_t expectTilingKey = 3;
    string expectTilingData = "40 64 32 11 6 6 10 1 6 6 39424 35840 1408 1280 558 558 1 0 0 6684672 4325376 393216 "
                              "1088 704 64 28 112 544 4 68 44 2 64 17 11 64 64 4 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp32_compute_ab_D64)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "ApplyRotaryPosEmb",
        {
            // input info
            {{{4096, 60, 40, 64}, {4096, 60, 40, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4096, 60, 1, 64}, {4096, 60, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4096, 60, 1, 64}, {4096, 60, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4096, 60, 1, 64}, {4096, 60, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{4096, 60, 40, 64}, {4096, 60, 40, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4096, 60, 1, 64}, {4096, 60, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {// attr
         {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")}},
        &compileInfo, "Ascend910B", 40, 196608);
    uint64_t expectTilingKey = 3;
    string expectTilingData = "40 64 32 4 4 4 3 1 1 1 41984 31488 1024 768 1535 1535 1 1 1 15728640 393216 393216 2560 "
                              "64 64 41 328 1280 8 320 8 4 64 40 1 64 64 8 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp16_compute_ab_cast_D64)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{24, 60, 11, 64}, {24, 60, 11, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{24, 60, 1, 64}, {24, 60, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{24, 60, 1, 64}, {24, 60, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{24, 60, 1, 64}, {24, 60, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{24, 60, 11, 64}, {24, 60, 11, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{24, 60, 1, 64}, {24, 60, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {// attr
                                               {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                               {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")}},
                                              &compileInfo, "Ascend910B", 40, 196608);
    uint64_t expectTilingKey = 4;
    string expectTilingData = "40 64 32 24 12 12 24 24 12 12 36864 36864 3072 3072 1 1 0 0 0 25344 2304 2304 704 64 64 "
                              "12 48 352 4 44 4 2 64 11 1 64 64 4 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp32_compute_ab_cast_D64)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{24, 60, 11, 64}, {24, 60, 11, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{24, 60, 1, 64}, {24, 60, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{24, 60, 1, 64}, {24, 60, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{24, 60, 1, 64}, {24, 60, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{24, 60, 11, 64}, {24, 60, 11, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{24, 60, 1, 64}, {24, 60, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {// attr
                                               {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                               {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")}},
                                              &compileInfo, "Ascend910B", 40, 196608);
    uint64_t expectTilingKey = 4;
    string expectTilingData = "40 64 32 12 12 12 12 12 12 12 36864 36864 3072 3072 2 2 0 0 0 25344 2304 2304 704 64 64 "
                              "12 96 352 8 88 8 4 64 11 1 64 64 8 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_bf16_compute_ab_cast_D64)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{24, 60, 11, 64}, {24, 60, 11, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{24, 60, 1, 64}, {24, 60, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{24, 60, 1, 64}, {24, 60, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{24, 60, 1, 64}, {24, 60, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{24, 60, 11, 64}, {24, 60, 11, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{24, 60, 1, 64}, {24, 60, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {// attr
                                               {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                               {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")}},
                                              &compileInfo, "Ascend910B", 40, 196608);
    uint64_t expectTilingKey = 4;
    string expectTilingData = "40 64 32 12 12 12 12 12 12 12 36864 36864 1536 3072 2 2 0 0 0 25344 2304 2304 704 64 64 "
                              "12 96 352 8 44 4 4 64 11 1 64 64 4 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_bf16_small_D32)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{24, 1, 11, 32}, {24, 1, 11, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 32}, {24, 1, 1, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 32}, {24, 1, 1, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 32}, {24, 1, 1, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{24, 1, 11, 32}, {24, 1, 11, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 32}, {24, 1, 1, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {// attr
                                               {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                               {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")}},
                                              &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_bf16_small_D96)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{24, 1, 11, 96}, {24, 1, 11, 96}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 96}, {24, 1, 1, 96}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 96}, {24, 1, 1, 96}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 96}, {24, 1, 1, 96}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{24, 1, 11, 96}, {24, 1, 11, 96}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 96}, {24, 1, 1, 96}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {// attr
                                               {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                               {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")}},
                                              &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_bf16_small_D256)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{24, 1, 11, 256}, {24, 1, 11, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 256}, {24, 1, 1, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 256}, {24, 1, 1, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 256}, {24, 1, 1, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{24, 1, 11, 256}, {24, 1, 11, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 256}, {24, 1, 1, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {// attr
                                               {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                               {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")}},
                                              &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp16_small_wrong_largesize)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{1, 8, 96, 128}, {1, 8, 96, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 8, 96, 128}, {1, 8, 96, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 8, 1, 128}, {1, 8, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 8, 1, 128}, {1, 8, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{1, 8, 96, 128}, {1, 8, 96, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 8, 96, 128}, {1, 8, 96, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                              },
                                              &compileInfo, "Ascend910B", 40, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp32_small_wrong_largesize)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{1, 8, 48, 128}, {1, 8, 48, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 8, 48, 128}, {1, 8, 48, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 8, 1, 128}, {1, 8, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 8, 1, 128}, {1, 8, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{1, 8, 48, 128}, {1, 8, 48, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 8, 48, 128}, {1, 8, 48, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                              },
                                              &compileInfo, "Ascend910B", 40, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_bf16_small_wrong_largesize)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{1, 8, 48, 128}, {1, 8, 48, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 8, 48, 128}, {1, 8, 48, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 8, 1, 128}, {1, 8, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 8, 1, 128}, {1, 8, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{1, 8, 48, 128}, {1, 8, 48, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 8, 48, 128}, {1, 8, 48, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                              },
                                              &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_bf16_small_wrong_largesize_D64)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{1, 8, 96, 64}, {1, 8, 96, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 8, 96, 64}, {1, 8, 96, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 8, 1, 64}, {1, 8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 8, 1, 64}, {1, 8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{1, 8, 96, 64}, {1, 8, 96, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 8, 96, 64}, {1, 8, 96, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                              },
                                              &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp16_small_wrong_largesize_D64)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{1, 8, 196, 64}, {1, 8, 196, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 8, 196, 64}, {1, 8, 196, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 8, 1, 64}, {1, 8, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 8, 1, 64}, {1, 8, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{1, 8, 196, 64}, {1, 8, 196, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 8, 196, 64}, {1, 8, 196, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                              },
                                              &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_bf16_compute_wrong_largesize)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "ApplyRotaryPosEmb",
        {
            // input info
            {{{4, 1024, 38, 128}, {4, 1024, 38, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 1024, 39, 128}, {4, 1024, 39, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 1024, 1, 128}, {4, 1024, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 1024, 1, 128}, {4, 1024, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            // output info
            {{{4, 1024, 38, 128}, {4, 1024, 38, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 1024, 39, 128}, {4, 1024, 39, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            // attr
            {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileInfo, "Ascend910B", 40, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp32_compute_wrong_largesize)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "ApplyRotaryPosEmb",
        {
            // input info
            {{{4, 1024, 38, 128}, {4, 1024, 38, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 1024, 39, 128}, {4, 1024, 39, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 1024, 1, 128}, {4, 1024, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 1024, 1, 128}, {4, 1024, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{4, 1024, 38, 128}, {4, 1024, 38, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 1024, 39, 128}, {4, 1024, 39, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // attr
            {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileInfo, "Ascend910B", 40, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp16_compute_wrong_largesize)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "ApplyRotaryPosEmb",
        {
            // input info
            {{{4, 1024, 76, 128}, {4, 1024, 76, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 1024, 77, 128}, {4, 1024, 77, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 1024, 1, 128}, {4, 1024, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 1024, 1, 128}, {4, 1024, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // output info
            {{{4, 1024, 76, 128}, {4, 1024, 76, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 1024, 77, 128}, {4, 1024, 77, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileInfo, "Ascend910B", 40, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_bf16_compute_wrong_largesize_D64)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{4, 1024, 76, 64}, {4, 1024, 76, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4, 1024, 78, 64}, {4, 1024, 78, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4, 1024, 1, 64}, {4, 1024, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4, 1024, 1, 64}, {4, 1024, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{4, 1024, 76, 64}, {4, 1024, 76, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4, 1024, 78, 64}, {4, 1024, 78, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                              },
                                              &compileInfo, "Ascend910B", 40, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp32_compute_wrong_largesize_D64)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{4, 1024, 76, 64}, {4, 1024, 76, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4, 1024, 78, 64}, {4, 1024, 78, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4, 1024, 1, 64}, {4, 1024, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4, 1024, 1, 64}, {4, 1024, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{4, 1024, 76, 64}, {4, 1024, 76, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4, 1024, 78, 64}, {4, 1024, 78, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                              },
                                              &compileInfo, "Ascend910B", 40, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp16_compute_wrong_largesize_D64)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "ApplyRotaryPosEmb",
        {
            // input info
            {{{4, 1024, 150, 64}, {4, 1024, 150, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 1024, 160, 64}, {4, 1024, 160, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 1024, 1, 64}, {4, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 1024, 1, 64}, {4, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // output info
            {{{4, 1024, 150, 64}, {4, 1024, 150, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 1024, 160, 64}, {4, 1024, 160, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileInfo, "Ascend910B", 40, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_bf16_small_wrong_largesize_TND)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{8, 48, 128}, {8, 48, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{8, 48, 128}, {8, 48, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{8, 1, 128}, {8, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{8, 1, 128}, {8, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{8, 48, 128}, {8, 48, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{8, 48, 128}, {8, 48, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                              },
                                              &compileInfo, "Ascend910B", 40, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp32_small_wrong_largesize_TND)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{8, 48, 128}, {8, 48, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{8, 48, 128}, {8, 48, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{8, 1, 128}, {8, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{8, 1, 128}, {8, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{8, 48, 128}, {8, 48, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{8, 48, 128}, {8, 48, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                              },
                                              &compileInfo, "Ascend910B", 40, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp16_small_wrong_largesize_TND)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{8, 96, 128}, {8, 96, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{8, 96, 128}, {8, 96, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{8, 1, 128}, {8, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{8, 1, 128}, {8, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{8, 96, 128}, {8, 96, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{8, 96, 128}, {8, 96, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                              },
                                              &compileInfo, "Ascend910B", 40, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_bf16_small_wrong_largesize_TND_D64)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{8, 96, 64}, {8, 96, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{8, 96, 64}, {8, 96, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{8, 96, 64}, {8, 96, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{8, 96, 64}, {8, 96, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                              },
                                              &compileInfo, "Ascend910B", 40, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp32_small_wrong_largesize_TND_D64)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{8, 96, 64}, {8, 96, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{8, 96, 64}, {8, 96, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{8, 1, 64}, {8, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{8, 1, 64}, {8, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{8, 96, 64}, {8, 96, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{8, 96, 64}, {8, 96, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                              },
                                              &compileInfo, "Ascend910B", 40, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp16_small_wrong_largesize_TND_D64)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{8, 196, 64}, {8, 196, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{8, 196, 64}, {8, 196, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{8, 1, 64}, {8, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{8, 1, 64}, {8, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{8, 196, 64}, {8, 196, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{8, 196, 64}, {8, 196, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                              },
                                              &compileInfo, "Ascend910B", 40, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_bf16_compute_wrong_largesize_TND)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{4096, 38, 128}, {4096, 38, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4096, 39, 128}, {4096, 39, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4096, 1, 128}, {4096, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4096, 1, 128}, {4096, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{4096, 38, 128}, {4096, 38, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4096, 39, 128}, {4096, 39, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                              },
                                              &compileInfo, "Ascend910B", 40, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp32_compute_wrong_largesize_TND)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{4096, 38, 128}, {4096, 38, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 39, 128}, {4096, 39, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 1, 128}, {4096, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 1, 128}, {4096, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{4096, 38, 128}, {4096, 38, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 39, 128}, {4096, 39, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                              },
                                              &compileInfo, "Ascend910B", 40, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp16_compute_wrong_largesize_TND)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{4096, 76, 128}, {4096, 76, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{4096, 77, 128}, {4096, 77, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{4096, 1, 128}, {4096, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{4096, 1, 128}, {4096, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{4096, 76, 128}, {4096, 76, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{4096, 77, 128}, {4096, 77, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                              },
                                              &compileInfo, "Ascend910B", 40, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_bf16_compute_wrong_largesize_TND_D64)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{4096, 76, 64}, {4096, 76, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4096, 80, 64}, {4096, 80, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4096, 1, 64}, {4096, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4096, 1, 64}, {4096, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{4096, 76, 64}, {4096, 76, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4096, 80, 64}, {4096, 80, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                              },
                                              &compileInfo, "Ascend910B", 40, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp32_compute_wrong_largesize_TND_D64)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{4096, 76, 64}, {4096, 76, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 80, 64}, {4096, 80, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 1, 64}, {4096, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 1, 64}, {4096, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{4096, 76, 64}, {4096, 76, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 80, 64}, {4096, 80, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                              },
                                              &compileInfo, "Ascend910B", 40, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp16_compute_wrong_largesize_TND_D64)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{4096, 152, 64}, {4096, 152, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{4096, 160, 64}, {4096, 160, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{4096, 1, 64}, {4096, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{4096, 1, 64}, {4096, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{4096, 152, 64}, {4096, 152, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{4096, 160, 64}, {4096, 160, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                              },
                                              &compileInfo, "Ascend910B", 40, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_wrong_shape_is_2)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{20, 128}, {20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 1, 128}, {4096, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 1, 128}, {4096, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{20, 128}, {20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                              },
                                              &compileInfo, "Ascend910B", 40, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_wrong_shape_dim_num_is_diff)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "ApplyRotaryPosEmb",
        {
            // input info
            {{{4, 1024, 20, 128}, {4, 1024, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4096, 1, 128}, {4096, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4096, 1, 128}, {4096, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{4, 1024, 20, 128}, {4, 1024, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // attr
            {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
        },
        &compileInfo, "Ascend910B", 40, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_wrong_shape_TBS_is_diff)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{2048, 20, 128}, {2048, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 1, 128}, {4096, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 1, 128}, {4096, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{2048, 20, 128}, {2048, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                              },
                                              &compileInfo, "Ascend910B", 40, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_wrong_shape_cos_N_isnot_1)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 1, 128}, {4096, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                              },
                                              &compileInfo, "Ascend910B", 40, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_wrong_shape_input_is_null)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},

                                              },
                                              {
                                                  // output info
                                                  {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                              },
                                              &compileInfo, "Ascend910B", 40, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_wrong_layout_is_3)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 1, 128}, {4096, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 1, 128}, {4096, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                              },
                                              &compileInfo, "Ascend910B", 40, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_wrong_layout_isnot_inputDim)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 1, 128}, {4096, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 1, 128}, {4096, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                              },
                                              &compileInfo, "Ascend910B", 40, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_wrong_Dtype_is_int32)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{4096, 1, 128}, {4096, 1, 128}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{4096, 1, 128}, {4096, 1, 128}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                              },
                                              &compileInfo, "Ascend910B", 40, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_wrong_Dtype_is_diff)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 1, 128}, {4096, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{4096, 1, 128}, {4096, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                              },
                                              &compileInfo, "Ascend910B", 40, 196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp16_small_Ascend950)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {// attr
                                               {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                               {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")}},
                                              &compileInfo, "Ascend950", 40, 196608);
    uint64_t expectTilingKey = 20030;
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp16_BNSD_BA_Ascend950)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 1, 11, 128}, {1, 1, 11, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 1, 11, 128}, {1, 1, 11, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {// attr
                                               {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                               {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")}},
                                              &compileInfo, "Ascend950", 40, 196608);
    uint64_t expectTilingKey = 20011;
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp32_BNSD_BA_B40_Ascend950)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  {{{40, 1, 11, 128}, {40, 1, 11, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{40, 1, 11, 128}, {40, 1, 11, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 1, 11, 128}, {1, 1, 11, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 1, 11, 128}, {1, 1, 11, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{40, 1, 11, 128}, {40, 1, 11, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{40, 1, 11, 128}, {40, 1, 11, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {// attr
                                               {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                               {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")}},
                                              &compileInfo, "Ascend950", 40, 196608);
    uint64_t expectTilingKey = 20011;
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp16_BNSD_ABA_S40_Ascend950)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  {{{2, 1, 40, 128}, {2, 1, 40, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{2, 1, 40, 128}, {2, 1, 40, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{2, 1, 40, 128}, {2, 1, 40, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{2, 1, 40, 128}, {2, 1, 40, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2, 1, 40, 128}, {2, 1, 40, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{2, 1, 40, 128}, {2, 1, 40, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {// attr
                                               {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                               {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")}},
                                              &compileInfo, "Ascend950", 40, 196608);
    uint64_t expectTilingKey = 20010;
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_bf16_BNSD_BA_Ascend950)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 1, 11, 128}, {1, 1, 11, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 1, 11, 128}, {1, 1, 11, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {// attr
                                               {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                               {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")}},
                                              &compileInfo, "Ascend950", 40, 196608);
    uint64_t expectTilingKey = 20011;
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp16_BNSD_ABA_LargeD_Ascend950)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  {{{8, 1, 100, 512}, {8, 1, 100, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{8, 1, 100, 512}, {8, 1, 100, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{8, 1, 100, 512}, {8, 1, 100, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{8, 1, 100, 512}, {8, 1, 100, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 1, 100, 512}, {8, 1, 100, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{8, 1, 100, 512}, {8, 1, 100, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {// attr
                                               {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                               {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")}},
                                              &compileInfo, "Ascend950", 40, 196608);
    uint64_t expectTilingKey = 20010;
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp32_BNSD_BA_LargeN_Ascend950)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  {{{8, 32, 1, 256}, {8, 32, 1, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{8, 32, 1, 256}, {8, 32, 1, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 1, 1, 256}, {1, 1, 1, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 1, 1, 256}, {1, 1, 1, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 32, 1, 256}, {8, 32, 1, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{8, 32, 1, 256}, {8, 32, 1, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {// attr
                                               {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                               {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")}},
                                              &compileInfo, "Ascend950", 40, 196608);
    uint64_t expectTilingKey = 20011;
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp16_BNSD_BA_interleave_Ascend950)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "ApplyRotaryPosEmb",
        {
            {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 11, 128}, {1, 1, 11, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 11, 128}, {1, 1, 11, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {// attr
         {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("interleave")}},
        &compileInfo, "Ascend950", 40, 196608);
    uint64_t expectTilingKey = 20011;
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp16_BNSD_ABA_quarter_Ascend950)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "ApplyRotaryPosEmb",
        {
            {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {// attr
         {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("quarter")}},
        &compileInfo, "Ascend950", 40, 196608);
    uint64_t expectTilingKey = 20010;
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp32_BNSD_ABA_Ascend950)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {// attr
                                               {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                               {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")}},
                                              &compileInfo, "Ascend950", 40, 196608);
    uint64_t expectTilingKey = 20010;
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_bf16_BNSD_ABA_Ascend950)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {// attr
                                               {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                               {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")}},
                                              &compileInfo, "Ascend950", 40, 196608);
    uint64_t expectTilingKey = 20010;
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp16_BNSD_BA_D64_Ascend950)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  {{{24, 1, 11, 64}, {24, 1, 11, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{24, 1, 11, 64}, {24, 1, 11, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 1, 11, 64}, {1, 1, 11, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 1, 11, 64}, {1, 1, 11, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{24, 1, 11, 64}, {24, 1, 11, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{24, 1, 11, 64}, {24, 1, 11, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {// attr
                                               {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                               {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")}},
                                              &compileInfo, "Ascend950", 40, 196608);
    uint64_t expectTilingKey = 20011;
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp16_BNSD_ABA_NonDiv_Ascend950)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  {{{3, 1, 200, 128}, {3, 1, 200, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{3, 1, 200, 128}, {3, 1, 200, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{3, 1, 200, 128}, {3, 1, 200, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{3, 1, 200, 128}, {3, 1, 200, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{3, 1, 200, 128}, {3, 1, 200, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{3, 1, 200, 128}, {3, 1, 200, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {// attr
                                               {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                               {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")}},
                                              &compileInfo, "Ascend950", 40, 196608);
    uint64_t expectTilingKey = 20010;
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp16_compute_ab_Ascend950)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "ApplyRotaryPosEmb",
        {
            // input info
            {{{640, 1, 15, 128}, {640, 1, 15, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{640, 1, 15, 128}, {640, 1, 15, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{640, 1, 1, 128}, {640, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{640, 1, 1, 128}, {640, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // output info
            {{{640, 1, 15, 128}, {640, 1, 15, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{640, 1, 15, 128}, {640, 1, 15, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {// attr
         {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")}},
        &compileInfo, "Ascend950", 40, 196608);
    uint64_t expectTilingKey = 20030;
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp16_small_BNSD_Ascend950)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{24, 11, 1, 128}, {24, 11, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{24, 11, 1, 128}, {24, 11, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{24, 11, 1, 128}, {24, 11, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{24, 11, 1, 128}, {24, 11, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {// attr
                                               {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                               {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")}},
                                              &compileInfo, "Ascend950", 40, 196608);
    uint64_t expectTilingKey = 20010;
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp32_compute_ab_BNSD_Ascend950)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{1024, 20, 128}, {1024, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1024, 20, 128}, {1024, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1024, 1, 128}, {1024, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1024, 1, 128}, {1024, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{1024, 20, 128}, {1024, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1024, 20, 128}, {1024, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {// attr
                                               {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                               {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")}},
                                              &compileInfo, "Ascend950", 40, 196608);
    uint64_t expectTilingKey = 20010;
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_bf16_compute_ab_cast_BNSD_Ascend950)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{4096, 16, 128}, {4096, 16, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4096, 16, 128}, {4096, 16, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4096, 1, 128}, {4096, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4096, 1, 128}, {4096, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{4096, 16, 128}, {4096, 16, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4096, 16, 128}, {4096, 16, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {// attr
                                               {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                               {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")}},
                                              &compileInfo, "Ascend950", 40, 196608);
    uint64_t expectTilingKey = 20010;
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp16_small_BNSD_Ascend950_D64)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{24, 11, 1, 64}, {24, 11, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{24, 11, 1, 64}, {24, 11, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 64}, {24, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 64}, {24, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{24, 11, 1, 64}, {24, 11, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{24, 11, 1, 64}, {24, 11, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {// attr
                                               {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                               {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")}},
                                              &compileInfo, "Ascend950", 40, 196608);
    uint64_t expectTilingKey = 20010;
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}
