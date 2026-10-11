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
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/moe_finalize_routing_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class MoeFinalizeRoutingTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeFinalizeRoutingTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeFinalizeRoutingTiling TearDown" << std::endl;
    }
};

// ----------------------------------------------------------------------------------------------------------

TEST_F(MoeFinalizeRoutingTiling, MoeFinalizeRouting_tiling_float)
{
    optiling::MoeFinalizeRoutingCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRouting",
                                              {
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16, 1}, {16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{16, 1}, {16, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    int64_t expectTilingKey = 20015;
    string expectTilingData = "64 16 0 16 16 16 16 16 0 0 0 0 1 1 1 1 1 1 1 1 1 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTiling, MoeFinalizeRouting_tiling_bfloat16)
{
    optiling::MoeFinalizeRoutingCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRouting",
                                              {
                                                  {{{16, 16}, {16, 16}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{16, 16}, {16, 16}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{16, 16}, {16, 16}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{16, 16}, {16, 16}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{16, 1}, {16, 1}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{16, 1}, {16, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{16, 16}, {16, 16}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    int64_t expectTilingKey = 20017;
    string expectTilingData = "64 16 0 16 16 16 16 16 0 0 0 0 1 1 1 1 1 1 1 1 1 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTiling, MoeFinalizeRouting_tiling_float16)
{
    optiling::MoeFinalizeRoutingCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRouting",
                                              {
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{16, 1}, {16, 1}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{16, 1}, {16, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    int64_t expectTilingKey = 20016;
    string expectTilingData = "64 16 0 16 16 16 16 16 0 0 0 0 1 1 1 1 1 1 1 1 1 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTiling, MoeFinalizeRouting_tiling_cut_h_float)
{
    optiling::MoeFinalizeRoutingCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRouting",
                                              {
                                                  {{{4096 * 4, 8934}, {4096 * 4, 8934}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 8934}, {4096, 8934}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 8934}, {4096, 8934}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{8, 8934}, {8, 8934}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 4}, {4096, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096 * 4}, {4096 * 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{4096, 4}, {4096, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{4096, 8934}, {4096, 8934}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    int64_t expectTilingKey = 20009;
    string expectTilingData = "64 64 0 8 4096 8934 5448 3486 1 0 0 0 4 64 128 1 1 64 128 1 1 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTiling, MoeFinalizeRouting_tiling_cut_h_float16)
{
    optiling::MoeFinalizeRoutingCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeFinalizeRouting",
        {
            {{{4096 * 2, 15080}, {4096 * 2, 15080}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 15080}, {4096, 15080}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 15080}, {4096, 15080}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 15080}, {16, 15080}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 2}, {4096, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096 * 2}, {4096 * 2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4096, 2}, {4096, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{4096, 15080}, {4096, 15080}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    int64_t expectTilingKey = 20016;
    string expectTilingData = "64 64 0 16 4096 15080 15080 15080 0 0 0 0 2 64 64 1 1 64 64 1 1 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTiling, MoeFinalizeRouting_tiling_cut_hk_float)
{
    optiling::MoeFinalizeRoutingCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeFinalizeRouting",
        {
            {{{1024 * 258, 8936}, {1024 * 258, 8936}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024, 8936}, {1024, 8936}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024, 8936}, {1024, 8936}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{340, 8936}, {340, 8936}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024, 258}, {1024, 258}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024 * 258}, {1024 * 258}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1024, 258}, {1024, 258}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1024, 8936}, {1024, 8936}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    int64_t expectTilingKey = 20000;
    string expectTilingData = "64 64 0 340 1024 8936 8120 816 1 256 2 2 258 16 32 1 1 16 32 1 1 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTiling, MoeFinalizeRouting_tiling_cut_k_float)
{
    optiling::MoeFinalizeRoutingCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeFinalizeRouting",
        {
            {{{1024 * 258, 2560}, {1024 * 258, 2560}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024, 2560}, {1024, 2560}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024, 2560}, {1024, 2560}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{340, 2560}, {340, 2560}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024, 258}, {1024, 258}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024 * 258}, {1024 * 258}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1024, 258}, {1024, 258}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1024, 2560}, {1024, 2560}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    int64_t expectTilingKey = 20000;
    string expectTilingData = "64 64 0 340 1024 2560 2560 2560 0 256 2 2 258 16 4 5 1 16 4 5 1 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTiling, MoeFinalizeRouting_tiling_float_newwork)
{
    optiling::MoeFinalizeRoutingCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeFinalizeRouting",
        {
            {{{1024 * 258, 2560}, {1024 * 258, 2560}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024, 2560}, {1024, 2560}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024, 2560}, {1024, 2560}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{340, 2560}, {340, 2560}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024, 258}, {1024, 258}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024 * 258}, {1024 * 258}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1024, 258}, {1024, 258}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1024, 2560}, {1024, 2560}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    int64_t expectTilingKey = 20000;
    string expectTilingData = "64 64 0 340 1024 2560 2560 2560 0 256 2 2 258 16 4 5 1 16 4 5 1 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// H=5120 network path (OptimizedCutH), float, k=4
TEST_F(MoeFinalizeRoutingTiling, MoeFinalizeRouting_tiling_network_float)
{
    optiling::MoeFinalizeRoutingCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRouting",
                                              {
                                                  {{{4096 * 4, 5120}, {4096 * 4, 5120}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 5120}, {4096, 5120}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 5120}, {4096, 5120}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{8, 5120}, {8, 5120}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 4}, {4096, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096 * 4}, {4096 * 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{4096, 4}, {4096, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{4096, 5120}, {4096, 5120}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    constexpr uint64_t skipTilingKey = UINT64_MAX;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, skipTilingKey, "", expectWorkspaces);
}

// CutH + BF16 (k=4)
TEST_F(MoeFinalizeRoutingTiling, MoeFinalizeRouting_tiling_cut_h_bfloat16)
{
    optiling::MoeFinalizeRoutingCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRouting",
                                              {
                                                  {{{4096 * 4, 8934}, {4096 * 4, 8934}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4096, 8934}, {4096, 8934}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4096, 8934}, {4096, 8934}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{8, 8934}, {8, 8934}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4096, 4}, {4096, 4}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4096 * 4}, {4096 * 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{4096, 4}, {4096, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{4096, 8934}, {4096, 8934}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    int64_t expectTilingKey = 20011;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// CutH + FP16 (k=4); H large enough to miss all-bias path (8934 still loads all bias for FP16)
TEST_F(MoeFinalizeRoutingTiling, MoeFinalizeRouting_tiling_cut_h_fp16_k4)
{
    optiling::MoeFinalizeRoutingCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeFinalizeRouting",
        {
            {{{4096 * 4, 32768}, {4096 * 4, 32768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 32768}, {4096, 32768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 32768}, {4096, 32768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 32768}, {8, 32768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 4}, {4096, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096 * 4}, {4096 * 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4096, 4}, {4096, 4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{4096, 32768}, {4096, 32768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    int64_t expectTilingKey = 20010;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// Big K + FP16
TEST_F(MoeFinalizeRoutingTiling, MoeFinalizeRouting_tiling_cut_hk_fp16)
{
    optiling::MoeFinalizeRoutingCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeFinalizeRouting",
        {
            {{{1024 * 258, 8936}, {1024 * 258, 8936}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1024, 8936}, {1024, 8936}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1024, 8936}, {1024, 8936}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{340, 8936}, {340, 8936}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1024, 258}, {1024, 258}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1024 * 258}, {1024 * 258}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1024, 258}, {1024, 258}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1024, 8936}, {1024, 8936}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    int64_t expectTilingKey = 20001;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// Big K + BF16
TEST_F(MoeFinalizeRoutingTiling, MoeFinalizeRouting_tiling_cut_hk_bf16)
{
    optiling::MoeFinalizeRoutingCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeFinalizeRouting",
        {
            {{{1024 * 258, 8936}, {1024 * 258, 8936}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 8936}, {1024, 8936}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 8936}, {1024, 8936}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{340, 8936}, {340, 8936}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 258}, {1024, 258}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024 * 258}, {1024 * 258}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1024, 258}, {1024, 258}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1024, 8936}, {1024, 8936}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        &compileInfo);
    int64_t expectTilingKey = 20002;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// CutH + float, k=2
TEST_F(MoeFinalizeRoutingTiling, MoeFinalizeRouting_tiling_cut_h_float_k2)
{
    optiling::MoeFinalizeRoutingCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRouting",
                                              {
                                                  {{{4096 * 2, 8934}, {4096 * 2, 8934}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 8934}, {4096, 8934}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 8934}, {4096, 8934}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{8, 8934}, {8, 8934}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 2}, {4096, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096 * 2}, {4096 * 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{4096, 2}, {4096, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{4096, 8934}, {4096, 8934}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    int64_t expectTilingKey = 20006;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// FP16 default CutH (k=8, not k2/k4)
TEST_F(MoeFinalizeRoutingTiling, MoeFinalizeRouting_tiling_cut_h_fp16_k8)
{
    optiling::MoeFinalizeRoutingCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeFinalizeRouting",
        {
            {{{4096 * 4, 32768}, {4096 * 4, 32768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 32768}, {4096, 32768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 32768}, {4096, 32768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 32768}, {8, 32768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 8}, {4096, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096 * 4}, {4096 * 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4096, 8}, {4096, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{4096, 32768}, {4096, 32768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    int64_t expectTilingKey = 20013;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTiling, MoeFinalizeRouting_tiling_network_float_key)
{
    optiling::MoeFinalizeRoutingCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRouting",
                                              {
                                                  {{{4096 * 4, 5120}, {4096 * 4, 5120}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 5120}, {4096, 5120}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 5120}, {4096, 5120}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{8, 5120}, {8, 5120}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096, 4}, {4096, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096 * 4}, {4096 * 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{4096, 4}, {4096, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{4096, 5120}, {4096, 5120}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    int64_t expectTilingKey = 20018;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTiling, MoeFinalizeRouting_tiling_check_shape_dim_fail)
{
    optiling::MoeFinalizeRoutingCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRouting",
                                              {
                                                  {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16, 1}, {16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{16, 1}, {16, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(MoeFinalizeRoutingTiling, MoeFinalizeRouting_tiling_check_k_zero_fail)
{
    optiling::MoeFinalizeRoutingCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRouting",
                                              {
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16, 0}, {16, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{16, 0}, {16, 0}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(MoeFinalizeRoutingTiling, MoeFinalizeRouting_tiling_data_field_accessors)
{
    optiling::MoeFinalizeRoutingTilingData tilingData;
    tilingData.set_totalCoreNum(64);
    tilingData.set_usedCoreNum(40);
    tilingData.set_skip2IsNull(1);
    tilingData.set_biasRowNum(8);
    tilingData.set_totalRowNum(4096);
    tilingData.set_H(5120);
    tilingData.set_normalH(2560);
    tilingData.set_unnormalH(2560);
    tilingData.set_hSliceNum(1);
    tilingData.set_normalK(256);
    tilingData.set_unnormalK(2);
    tilingData.set_kSliceNum(2);
    tilingData.set_K(258);
    tilingData.set_normalCoreHandleNum(128);
    tilingData.set_normalCoreLoopNum(4);
    tilingData.set_normalCoreHandleNumPerLoop(32);
    tilingData.set_normalCoreHandleNumTailLoop(32);
    tilingData.set_tailCoreHandleNum(64);
    tilingData.set_tailCoreLoopNum(2);
    tilingData.set_tailCoreHandleNumPerLoop(32);
    tilingData.set_tailCoreHandleNumTailLoop(32);
    tilingData.set_tilingKey(optiling::DTYPE_FLOAT_CUTH_NETWORK);

    EXPECT_EQ(tilingData.get_totalCoreNum(), 64);
    EXPECT_EQ(tilingData.get_usedCoreNum(), 40);
    EXPECT_EQ(tilingData.get_skip2IsNull(), 1);
    EXPECT_EQ(tilingData.get_biasRowNum(), 8);
    EXPECT_EQ(tilingData.get_totalRowNum(), 4096);
    EXPECT_EQ(tilingData.get_H(), 5120);
    EXPECT_EQ(tilingData.get_normalH(), 2560);
    EXPECT_EQ(tilingData.get_unnormalH(), 2560);
    EXPECT_EQ(tilingData.get_hSliceNum(), 1);
    EXPECT_EQ(tilingData.get_normalK(), 256);
    EXPECT_EQ(tilingData.get_unnormalK(), 2);
    EXPECT_EQ(tilingData.get_kSliceNum(), 2);
    EXPECT_EQ(tilingData.get_K(), 258);
    EXPECT_EQ(tilingData.get_normalCoreHandleNum(), 128);
    EXPECT_EQ(tilingData.get_normalCoreLoopNum(), 4);
    EXPECT_EQ(tilingData.get_normalCoreHandleNumPerLoop(), 32);
    EXPECT_EQ(tilingData.get_normalCoreHandleNumTailLoop(), 32);
    EXPECT_EQ(tilingData.get_tailCoreHandleNum(), 64);
    EXPECT_EQ(tilingData.get_tailCoreLoopNum(), 2);
    EXPECT_EQ(tilingData.get_tailCoreHandleNumPerLoop(), 32);
    EXPECT_EQ(tilingData.get_tailCoreHandleNumTailLoop(), 32);
    EXPECT_EQ(tilingData.get_tilingKey(), optiling::DTYPE_FLOAT_CUTH_NETWORK);

    const size_t dataSize = tilingData.GetDataSize();
    ASSERT_GT(dataSize, 0U);
    std::vector<uint8_t> buf(dataSize);
    tilingData.SaveToBuffer(buf.data(), buf.size());

    optiling::MoeFinalizeRoutingCompileInfo compileInfo;
    compileInfo.totalCoreNum = 40;
    compileInfo.ubSize = 65536;
    EXPECT_EQ(compileInfo.totalCoreNum, 40);
    EXPECT_EQ(compileInfo.ubSize, 65536U);
}
