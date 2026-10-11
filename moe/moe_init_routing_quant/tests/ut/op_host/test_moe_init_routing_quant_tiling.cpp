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
 * \file test_moe_init_routing_quant_tiling.h
 * \brief
 */
#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/moe_init_routing_quant_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "op_tiling_parse_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

using namespace std;

class MoeInitRoutingQuantTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeInitRoutingQuantTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeInitRoutingQuantTiling TearDown" << std::endl;
    }
};

TEST_F(MoeInitRoutingQuantTiling, MoeInitRoutingQuant_tiling_float)
{
    optiling::MoeInitRoutingQuantCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingQuant",
                                              {
                                                  {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{6, 5}, {6, 5}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
                                                  {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
                                                  {"offset", Ops::Transformer::AnyValue::CreateFrom<float>(0.0)},
                                              },
                                              &compileInfo);
    int64_t expectTilingKey = 1;
    string expectTilingData = "1 2 5 3 1065353216 1 6 1 6 6 6 1 6 6 8160 0 1024 1 0 6 0 0 0 6 6 6 0 0 0 6 6 0 0 1 6 2 "
                              "3 3 3 2 2 2 3 3 3 2 2 5 0 ";
    std::vector<size_t> expectWorkspaces = {16777448};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingQuantTiling, MoeInitRoutingQuant_tiling_multi_core)
{
    optiling::MoeInitRoutingQuantCompileInfo compileInfo = {};
    int64_t tokenNum = 16384;
    int64_t topk = 3;
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingQuant",
        {
            {{{tokenNum, 5}, {tokenNum, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{tokenNum, topk}, {tokenNum, topk}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{tokenNum, topk}, {tokenNum, topk}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{tokenNum * topk, 5}, {tokenNum * topk, 5}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tokenNum * topk)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
            {"offset", Ops::Transformer::AnyValue::CreateFrom<float>(0.0)},
        },
        &compileInfo);
    int64_t expectTilingKey = 2;
    string expectTilingData = "64 16384 5 3 1065353216 16 3072 1 3072 3072 3072 1 3072 3072 8096 4 1024 64 0 768 0 0 0 "
                              "768 768 768 0 0 0 768 768 0 0 64 49152 256 3 3 3 256 256 256 3 3 3 256 256 5 0 ";
    std::vector<size_t> expectWorkspaces = {18157568};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingQuantTiling, MoeInitRoutingQuant_tiling_long_h)
{
    optiling::MoeInitRoutingQuantCompileInfo compileInfo = {};
    int64_t tokenNum = 32768;
    int64_t topk = 3;
    int64_t h = 16384;

    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingQuant",
        {
            {{{tokenNum, h}, {tokenNum, h}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{tokenNum, topk}, {tokenNum, topk}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{tokenNum, topk}, {tokenNum, topk}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{tokenNum * topk, h}, {tokenNum * topk, h}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tokenNum * topk)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
            {"offset", Ops::Transformer::AnyValue::CreateFrom<float>(0.0)},
        },
        &compileInfo);
    int64_t expectTilingKey = 2;
    string expectTilingData = "64 32768 16384 3 1065353216 16 6144 1 6144 6144 6144 1 6144 6144 8096 4 1024 64 0 1536 "
                              "0 0 0 1536 1536 1536 0 0 0 1536 1536 0 0 64 98304 512 3 3 3 2 2 512 3 3 3 2 2 4094 0 ";
    std::vector<size_t> expectWorkspaces = {19533824};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingQuantTiling, MoeInitRoutingQuant_tiling_error)
{
    optiling::MoeInitRoutingQuantCompileInfo compileInfo = {};
    int64_t tokenNum = 32768;
    int64_t topk = 3;
    int64_t h = 16384;

    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingQuant",
        {
            {{{tokenNum, h}, {tokenNum, h}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, topk}, {3, topk}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{tokenNum, topk}, {tokenNum, topk}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{tokenNum * topk, h}, {tokenNum * topk, h}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tokenNum * topk)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
            {"offset", Ops::Transformer::AnyValue::CreateFrom<float>(0.0)},
        },
        &compileInfo);
    int64_t expectTilingKey = 2;
    string expectTilingData = "64 16384 5 3 1065353216 16 3072 1 3072 3072 3072 1 3072 3072 8096 4 1024 64 0 768 0 0 0 "
                              "768 768 768 0 0 0 768 768 0 0 64 49152 256 3 3 3 256 256 256 3 3 3 256 256 5 0 ";
    std::vector<size_t> expectWorkspaces = {18157568};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Split-K gather path: n < aivNum && n < k (case2: 1 x 48 x 1024, active=1)
TEST_F(MoeInitRoutingQuantTiling, MoeInitRoutingQuant_tiling_split_k)
{
    optiling::MoeInitRoutingQuantCompileInfo compileInfo = {};
    int64_t n = 1;
    int64_t k = 48;
    int64_t h = 1024;
    int64_t activeNum = 1;
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingQuant",
        {
            {{{n, h}, {n, h}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{n, k}, {n, k}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{n, k}, {n, k}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{activeNum * k, h}, {activeNum * k, h}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{n * k}, {n * k}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{n * k}, {n * k}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"offset", Ops::Transformer::AnyValue::CreateFrom<float>(0.0f)},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1, "", {});
}

// Gather row path: n/2 > active_num -> tilingKey += 2 (case1: 4096 x 40 x 8, active=8)
TEST_F(MoeInitRoutingQuantTiling, MoeInitRoutingQuant_tiling_gather_row)
{
    optiling::MoeInitRoutingQuantCompileInfo compileInfo = {};
    int64_t n = 4096;
    int64_t k = 40;
    int64_t h = 8;
    int64_t activeNum = 8;
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingQuant",
        {
            {{{n, h}, {n, h}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{n, k}, {n, k}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{n, k}, {n, k}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{activeNum * k, h}, {activeNum * k, h}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{n * k}, {n * k}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{n * k}, {n * k}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"offset", Ops::Transformer::AnyValue::CreateFrom<float>(0.0f)},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 4, "", {});
}

// Split-N with large H (case3: 20 x 20 x 44321, active=20)
TEST_F(MoeInitRoutingQuantTiling, MoeInitRoutingQuant_tiling_split_n_large_h)
{
    optiling::MoeInitRoutingQuantCompileInfo compileInfo = {};
    int64_t n = 20;
    int64_t k = 20;
    int64_t h = 44321;
    int64_t activeNum = 20;
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingQuant",
        {
            {{{n, h}, {n, h}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{n, k}, {n, k}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{n, k}, {n, k}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{activeNum * k, h}, {activeNum * k, h}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{n * k}, {n * k}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{n * k}, {n * k}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"offset", Ops::Transformer::AnyValue::CreateFrom<float>(0.0f)},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1, "", {});
}

// SrcToDst per-loop split: row gather + multi-core VBS (n/2 > active_num, totalLength > sortLoopMax)
TEST_F(MoeInitRoutingQuantTiling, MoeInitRoutingQuant_tiling_src_to_dst_split)
{
    optiling::MoeInitRoutingQuantCompileInfo compileInfo = {};
    int64_t n = 512;
    int64_t k = 40;
    int64_t h = 128;
    int64_t activeNum = 100;
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingQuant",
        {
            {{{n, h}, {n, h}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{n, k}, {n, k}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{n, k}, {n, k}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{activeNum * k, h}, {activeNum * k, h}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{n * k}, {n * k}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{n * k}, {n * k}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"offset", Ops::Transformer::AnyValue::CreateFrom<float>(0.0f)},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 4, "", {});
}

TEST_F(MoeInitRoutingQuantTiling, MoeInitRoutingQuant_tiling_fail_active_num_negative)
{
    optiling::MoeInitRoutingQuantCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingQuant",
                                              {
                                                  {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{6, 5}, {6, 5}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                  {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                  {"offset", Ops::Transformer::AnyValue::CreateFrom<float>(0.0f)},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(MoeInitRoutingQuantTiling, MoeInitRoutingQuant_tiling_fail_expanded_x_dim)
{
    optiling::MoeInitRoutingQuantCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingQuant",
                                              {
                                                  {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{6, 5, 1}, {6, 5, 1}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
                                                  {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                  {"offset", Ops::Transformer::AnyValue::CreateFrom<float>(0.0f)},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(MoeInitRoutingQuantTiling, MoeInitRoutingQuant_tiling_fail_row_idx_dim)
{
    optiling::MoeInitRoutingQuantCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingQuant",
                                              {
                                                  {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{6, 5}, {6, 5}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
                                                  {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                  {"offset", Ops::Transformer::AnyValue::CreateFrom<float>(0.0f)},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(MoeInitRoutingQuantTiling, MoeInitRoutingQuant_tiling_fail_expanded_row_idx_dim)
{
    optiling::MoeInitRoutingQuantCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingQuant",
                                              {
                                                  {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{6, 5}, {6, 5}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
                                                  {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                  {"offset", Ops::Transformer::AnyValue::CreateFrom<float>(0.0f)},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(MoeInitRoutingQuantTiling, MoeInitRoutingQuant_tiling_fail_expanded_idx_mismatch)
{
    optiling::MoeInitRoutingQuantCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingQuant",
                                              {
                                                  {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{6, 5}, {6, 5}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
                                                  {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                  {"offset", Ops::Transformer::AnyValue::CreateFrom<float>(0.0f)},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(MoeInitRoutingQuantTiling, MoeInitRoutingQuant_tiling_fail_expanded_x_row)
{
    optiling::MoeInitRoutingQuantCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingQuant",
                                              {
                                                  {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{5, 5}, {5, 5}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
                                                  {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                  {"offset", Ops::Transformer::AnyValue::CreateFrom<float>(0.0f)},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(MoeInitRoutingQuantTiling, MoeInitRoutingQuant_tiling_fail_expanded_x_col)
{
    optiling::MoeInitRoutingQuantCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingQuant",
                                              {
                                                  {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{6, 6}, {6, 6}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
                                                  {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                  {"offset", Ops::Transformer::AnyValue::CreateFrom<float>(0.0f)},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(MoeInitRoutingQuantTiling, MoeInitRoutingQuant_tiling_fail_expanded_row_idx_len)
{
    optiling::MoeInitRoutingQuantCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingQuant",
                                              {
                                                  {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{6, 5}, {6, 5}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
                                                  {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                  {"offset", Ops::Transformer::AnyValue::CreateFrom<float>(0.0f)},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(MoeInitRoutingQuantTiling, MoeInitRoutingQuant_tiling_fail_x_dim)
{
    optiling::MoeInitRoutingQuantCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingQuant",
                                              {
                                                  {{{6}, {6}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{6, 5}, {6, 5}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
                                                  {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                  {"offset", Ops::Transformer::AnyValue::CreateFrom<float>(0.0f)},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(MoeInitRoutingQuantTiling, MoeInitRoutingQuant_tiling_fail_input_rows)
{
    optiling::MoeInitRoutingQuantCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingQuant",
                                              {
                                                  {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{3, 3}, {3, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{3, 3}, {3, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{6, 5}, {6, 5}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
                                                  {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                  {"offset", Ops::Transformer::AnyValue::CreateFrom<float>(0.0f)},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(MoeInitRoutingQuantTiling, MoeInitRoutingQuant_tiling_parse_succ)
{
    gert::OpTilingParseContextBuilder builder;
    auto holder = builder.Build();
    auto parseContext = holder.GetContext();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MoeInitRoutingQuant");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling_parse, nullptr);

    EXPECT_EQ(opImpl->tiling_parse(reinterpret_cast<gert::KernelContext*>(parseContext)), ge::GRAPH_SUCCESS);
}

TEST_F(MoeInitRoutingQuantTiling, MoeInitRoutingQuant_tiling_split_k_active_zero)
{
    optiling::MoeInitRoutingQuantCompileInfo compileInfo = {};
    int64_t n = 1;
    int64_t k = 48;
    int64_t h = 1024;
    int64_t activeNum = 0;
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingQuant",
        {
            {{{n, h}, {n, h}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{n, k}, {n, k}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{n, k}, {n, k}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{0, h}, {0, h}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{n * k}, {n * k}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{n * k}, {n * k}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"offset", Ops::Transformer::AnyValue::CreateFrom<float>(0.0f)},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1, "", {});
}

TEST_F(MoeInitRoutingQuantTiling, MoeInitRoutingQuant_tiling_gather_row_active_zero)
{
    optiling::MoeInitRoutingQuantCompileInfo compileInfo = {};
    int64_t n = 4096;
    int64_t k = 40;
    int64_t h = 8;
    int64_t activeNum = 0;
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingQuant",
        {
            {{{n, h}, {n, h}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{n, k}, {n, k}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{n, k}, {n, k}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{0, h}, {0, h}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{n * k}, {n * k}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{n * k}, {n * k}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"offset", Ops::Transformer::AnyValue::CreateFrom<float>(0.0f)},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 4, "", {});
}

TEST_F(MoeInitRoutingQuantTiling, MoeInitRoutingQuant_tiling_vbs_align_branch)
{
    optiling::MoeInitRoutingQuantCompileInfo compileInfo = {};
    int64_t n = 12500;
    int64_t k = 4;
    int64_t h = 128;
    int64_t activeNum = n * k;
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingQuant",
        {
            {{{n, h}, {n, h}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{n, k}, {n, k}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{n, k}, {n, k}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{activeNum, h}, {activeNum, h}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{activeNum}, {activeNum}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{activeNum}, {activeNum}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"offset", Ops::Transformer::AnyValue::CreateFrom<float>(0.0f)},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 2, "", {});
}
