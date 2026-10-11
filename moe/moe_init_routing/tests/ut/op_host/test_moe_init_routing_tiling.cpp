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
#include "../../../op_host/moe_init_routing_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class MoeInitRoutingTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeInitRoutingTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeInitRoutingTiling TearDown" << std::endl;
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

TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_01)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting", // op_name
                                              {
                                                  // input info
                                                  {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)}},

                                              },

                                              &compileInfo);
    int64_t expectTilingKey = 0; // tilngkey
    string expectTilingData = "64 2 5 3 1 6 1 6 6 6 1 6 6 8096 0 1024 6 0 1 0 0 0 1 1 1 0 0 0 1 1 0 0 3 6 2 1 1 1 2 2 "
                              "2 1 1 1 2 2 5 1 ";      // tilingData
    std::vector<size_t> expectWorkspaces = {16781480}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_02)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting", // op_name
                                              {
                                                  // input info
                                                  {{{32, 1024}, {32, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{32, 1234}, {32, 1234}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{32, 1234}, {32, 1234}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{7404, 1024}, {7404, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{39488}, {39488}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{39488}, {39488}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)}},

                                              },

                                              &compileInfo);
    int64_t expectTilingKey = 4; // tilngkey
    string expectTilingData = "64 32 1024 1234 16 2496 1 2496 2496 2048 1 2048 2048 8096 4 1024 64 0 617 0 0 0 617 617 "
                              "617 0 0 0 617 617 0 0 64 7404 116 0 0 0 116 116 96 0 0 0 96 96 1024 2 "; // tilingData
    std::vector<size_t> expectWorkspaces = {17886976};                                                  // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_03)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting", // op_name
                                              {
                                                  // input info
                                                  {{{32, 1024}, {32, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{32, 256}, {32, 256}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{32, 256}, {32, 256}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{1536, 1024}, {1536, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{8192}, {8192}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{8192}, {8192}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)}},

                                              },

                                              &compileInfo);
    int64_t expectTilingKey = 4; // tilngkey
    string expectTilingData = "64 32 1024 256 4 2048 1 2048 2048 2048 1 2048 2048 8096 0 1024 64 0 128 0 0 0 128 128 "
                              "128 0 0 0 128 128 0 0 64 1536 24 0 0 0 24 24 24 0 0 0 24 24 1024 2 "; // tilingData
    std::vector<size_t> expectWorkspaces = {17010688};                                               // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_04)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting", // op_name
                                              {
                                                  // input info
                                                  {{{32, 1024}, {32, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{32, 256}, {32, 256}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{32, 256}, {32, 256}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{1536, 1024}, {1536, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{8192}, {8192}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{8192}, {8192}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)}},

                                              },

                                              &compileInfo);
    int64_t expectTilingKey = 4; // tilngkey
    string expectTilingData = "64 32 1024 256 4 2048 1 2048 2048 2048 1 2048 2048 8096 0 1024 64 0 128 0 0 0 128 128 "
                              "128 0 0 0 128 128 0 0 64 1536 24 0 0 0 24 24 24 0 0 0 24 24 1024 2 "; // tilingData
    std::vector<size_t> expectWorkspaces = {17010688};                                               // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_05)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting", // op_name
                                              {
                                                  // input info
                                                  {{{32, 1024}, {32, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{32, 256}, {32, 256}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{32, 256}, {32, 256}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{1536, 1024}, {1536, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{8192}, {8192}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{8192}, {8192}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)}},

                                              },

                                              &compileInfo);
    int64_t expectTilingKey = 4; // tilngkey
    string expectTilingData = "64 32 1024 256 4 2048 1 2048 2048 2048 1 2048 2048 8096 0 1024 64 0 128 0 0 0 128 128 "
                              "128 0 0 0 128 128 0 0 64 1536 24 0 0 0 24 24 24 0 0 0 24 24 1024 2 "; // tilingData
    std::vector<size_t> expectWorkspaces = {17010688};                                               // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_06)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting", // op_name
                                              {
                                                  // input info
                                                  {{{32, 1024}, {32, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{32, 256}, {32, 256}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{32, 256}, {32, 256}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{8192, 1024}, {8192, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{8192}, {8192}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{8192}, {8192}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)}},

                                              },

                                              &compileInfo);
    int64_t expectTilingKey = 2; // tilngkey
    string expectTilingData = "64 32 1024 256 4 2048 1 2048 2048 2048 1 2048 2048 8096 0 1024 64 0 128 0 0 0 128 128 "
                              "128 0 0 0 128 128 0 0 64 8192 32 4 4 4 32 32 32 4 4 4 32 32 1024 1 "; // tilingData
    std::vector<size_t> expectWorkspaces = {17010688};                                               // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_07)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting", // op_name
                                              {
                                                  // input info
                                                  {{{40, 5}, {40, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{40, 3000}, {40, 3000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{40, 3000}, {40, 3000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{120000, 5}, {120000, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{120000}, {120000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{120000}, {120000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)}},

                                              },

                                              &compileInfo);
    int64_t expectTilingKey = 2; // tilngkey
    string expectTilingData =
        "64 40 5 3000 16 7520 1 7520 7520 7200 1 7200 7200 8096 4 1024 64 0 1875 0 0 0 1875 1875 1875 0 0 0 1875 1875 "
        "0 0 64 120000 40 47 47 47 40 40 40 39 47 47 40 40 5 1 "; // tilingData
    std::vector<size_t> expectWorkspaces = {20141312};            // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_08)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting", // op_name
                                              {
                                                  // input info
                                                  {{{40, 5000}, {40, 5000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{40, 3000}, {40, 3000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{40, 3000}, {40, 3000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{30000, 5000}, {30000, 5000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{120000}, {120000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{120000}, {120000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(10)}},

                                              },

                                              &compileInfo);
    int64_t expectTilingKey = 4; // tilngkey
    string expectTilingData =
        "64 40 5000 3000 16 7520 1 7520 7520 7200 1 7200 7200 8096 4 1024 64 0 1875 0 0 0 1875 1875 1875 0 0 0 1875 "
        "1875 0 0 64 30000 469 0 0 0 469 469 453 0 0 0 453 453 5000 2 "; // tilingData
    std::vector<size_t> expectWorkspaces = {20141312};                   // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_09)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting", // op_name
                                              {
                                                  // input info
                                                  {{{39, 500}, {39, 500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{39, 3000}, {39, 3000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{39, 3000}, {39, 3000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{117000, 500}, {117000, 500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{117000}, {117000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{117000}, {117000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(39)}},

                                              },

                                              &compileInfo);
    int64_t expectTilingKey = 2; // tilngkey
    string expectTilingData =
        "64 39 500 3000 16 7328 1 7328 7328 7080 1 7104 7080 8096 4 1024 64 0 1829 0 0 0 1829 1829 1773 0 0 0 1773 "
        "1773 0 0 64 117000 39 47 47 47 39 39 39 39 47 47 39 39 500 1 "; // tilingData
    std::vector<size_t> expectWorkspaces = {20057312};                   // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_10)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting", // op_name
                                              {
                                                  // input info
                                                  {{{39, 5000}, {39, 5000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{39, 3000}, {39, 3000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{39, 3000}, {39, 3000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{117000}, {117000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{117000}, {117000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{117000}, {117000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(39)}},

                                              },

                                              &compileInfo);
    int64_t expectTilingKey = 0; // tilngkey
    string expectTilingData = "64 2 5 3 1 6 1 6 6 6 1 6 6 6848 0 2048 6 0 1 0 0 0 1 1 1 0 0 0 1 1 0 0 3 6 2 1 1 1 2 2 "
                              "2 1 1 1 2 2 5 1 ";      // tilingData
    std::vector<size_t> expectWorkspaces = {16781480}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_11)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting", // op_name
                                              {
                                                  // input info
                                                  {{{39, 5000}, {39, 5000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{39, 3000}, {39, 3000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{39, 3000}, {39, 3000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{117000, 5000}, {117000, 5000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{117000, 2}, {117000, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{117000}, {117000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(39)}},

                                              },

                                              &compileInfo);
    int64_t expectTilingKey = 0; // tilngkey
    string expectTilingData = "64 2 5 3 1 6 1 6 6 6 1 6 6 6848 0 2048 6 0 1 0 0 0 1 1 1 0 0 0 1 1 0 0 3 6 2 1 1 1 2 2 "
                              "2 1 1 1 2 2 5 1 ";      // tilingData
    std::vector<size_t> expectWorkspaces = {16781480}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_12)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting", // op_name
                                              {
                                                  // input info
                                                  {{{39, 5000}, {39, 5000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{39, 3000}, {39, 3000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{39, 3000}, {39, 3000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{117000, 5000}, {117000, 5000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{117000}, {117000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{117}, {117}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(39)}},

                                              },

                                              &compileInfo);
    int64_t expectTilingKey = 0; // tilngkey
    string expectTilingData = "64 2 5 3 1 6 1 6 6 6 1 6 6 6848 0 2048 6 0 1 0 0 0 1 1 1 0 0 0 1 1 0 0 3 6 2 1 1 1 2 2 "
                              "2 1 1 1 2 2 5 1 ";      // tilingData
    std::vector<size_t> expectWorkspaces = {16781480}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_13)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting", // op_name
                                              {
                                                  // input info
                                                  {{{39, 5000}, {39, 5000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{39, 3000}, {39, 3000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{39, 3000}, {39, 3000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{117000, 5000}, {117000, 5000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{117000}, {117000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{117}, {117}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(39)}},

                                              },

                                              &compileInfo);
    int64_t expectTilingKey = 0; // tilngkey
    string expectTilingData = "64 2 5 3 1 6 1 6 6 6 1 6 6 6848 0 2048 6 0 1 0 0 0 1 1 1 0 0 0 1 1 0 0 3 6 2 1 1 1 2 2 "
                              "2 1 1 1 2 2 5 1 ";      // tilingData
    std::vector<size_t> expectWorkspaces = {16781480}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_14)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting", // op_name
                                              {
                                                  // input info
                                                  {{{39, 5000}, {39, 5000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{39, 3000}, {39, 3000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{39, 3000}, {39, 3000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{117, 5000}, {117, 5000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{117000}, {117000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{117000}, {117000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(39)}},

                                              },

                                              &compileInfo);
    int64_t expectTilingKey = 0; // tilngkey
    string expectTilingData = "64 2 5 3 1 6 1 6 6 6 1 6 6 6848 0 2048 6 0 1 0 0 0 1 1 1 0 0 0 1 1 0 0 3 6 2 1 1 1 2 2 "
                              "2 1 1 1 2 2 5 1 ";      // tilingData
    std::vector<size_t> expectWorkspaces = {16781480}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_15)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting", // op_name
                                              {
                                                  // input info
                                                  {{{39, 5000}, {39, 5000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{39, 3000}, {39, 3000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{39, 3000}, {39, 3000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{117000, 5}, {117000, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{117000}, {117000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{117000}, {117000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(39)}},

                                              },

                                              &compileInfo);
    int64_t expectTilingKey = 0; // tilngkey
    string expectTilingData = "64 2 5 3 1 6 1 6 6 6 1 6 6 6848 0 2048 6 0 1 0 0 0 1 1 1 0 0 0 1 1 0 0 3 6 2 1 1 1 2 2 "
                              "2 1 1 1 2 2 5 1 ";      // tilingData
    std::vector<size_t> expectWorkspaces = {16781480}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_16)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting", // op_name
                                              {
                                                  // input info
                                                  {{{1, 577999}, {1, 577999}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 1000000}, {1, 1000000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1, 1000000}, {1, 1000000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{0, 577999}, {0, 577999}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1000000}, {1000000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1000000}, {1000000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},

                                              },

                                              &compileInfo);
    int64_t expectTilingKey = 2; // tilngkey
    string expectTilingData = "64 1 577999 1000000 64 15648 2 8096 7552 14176 2 7104 7072 8096 16 1024 64 0 15625 0 0 "
                              "0 2022 1471 15625 0 0 0 2022 1471 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 "; // tilingData
    std::vector<size_t> expectWorkspaces = {44781312};                                                  // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_17)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting", // op_name
                                              {
                                                  // input info
                                                  {{{4, 2}, {4, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{4, 2}, {4, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{4, 2}, {4, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{8, 2}, {8, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)}},

                                              },

                                              &compileInfo);
    int64_t expectTilingKey = 0; // tilngkey
    string expectTilingData = "64 4 2 2 1 8 1 8 8 8 1 8 8 8096 0 1024 8 0 1 0 0 0 1 1 1 0 0 0 1 1 0 0 4 8 1 2 2 2 1 1 "
                              "1 2 2 2 1 1 2 0 ";      // tilingData
    std::vector<size_t> expectWorkspaces = {16781536}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_18)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting", // op_name
                                              {
                                                  // input info
                                                  {{{1, 577999}, {1, 577999}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 1000000}, {1, 1000000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1, 1000000}, {1, 1000000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{0, 577999}, {0, 577999}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1000000}, {1000000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1000000}, {1000000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},

                                              },

                                              &compileInfo);
    int64_t expectTilingKey = 2; // tilngkey
    string expectTilingData = "64 1 577999 1000000 64 15648 2 8096 7552 14176 2 7104 7072 8096 16 1024 64 0 15625 0 0 "
                              "0 2022 1471 15625 0 0 0 2022 1471 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 "; // tilingData
    std::vector<size_t> expectWorkspaces = {44781312};                                                  // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// x not 2D → GetShapeAttrsInfo L216-219
TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_19)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting",
                                              {
                                                  {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)}},
                                              },
                                              &compileInfo);
    int64_t expectTilingKey = 0;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

// row_idx not 2D → GetShapeAttrsInfo L223-226
TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_20)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting",
                                              {
                                                  {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)}},
                                              },
                                              &compileInfo);
    int64_t expectTilingKey = 0;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

// row_idx shape != expert_idx shape → GetShapeAttrsInfo L229-234
TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_21)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting",
                                              {
                                                  {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{3, 3}, {3, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)}},
                                              },
                                              &compileInfo);
    int64_t expectTilingKey = 0;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

// x dim(0) != expert_idx dim(0) → GetShapeAttrsInfo L237-241
TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_22)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting",
                                              {
                                                  {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)}},
                                              },
                                              &compileInfo);
    int64_t expectTilingKey = 0;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

// activateNum < 0 → GetShapeAttrsInfo L244-248
TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_23)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting",
                                              {
                                                  {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}},
                                              },
                                              &compileInfo);
    int64_t expectTilingKey = 0;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

// expanded_row_idx dim(0) != totalLength → CheckOutShape L187-194
TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_24)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting",
                                              {
                                                  {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{7}, {7}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{7}, {7}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)}},
                                              },
                                              &compileInfo);
    int64_t expectTilingKey = 0;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Ascend950 SOC → IsRegbaseSocVersion=true → covers L128-129
TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_25)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting",
                                              {
                                                  {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)}},
                                              },
                                              &compileInfo, "Ascend950");
    int64_t expectTilingKey = 0;
    string expectTilingData = "64 2 5 3 1 6 1 6 6 6 1 6 6 6848 0 2048 6 0 1 0 0 0 1 1 1 0 0 0 1 1 0 0 3 6 2 1 1 1 2 2 "
                              "2 1 1 1 2 2 5 1 ";      // tilingData
    std::vector<size_t> expectWorkspaces = {16781480}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// n >= aivNum, n >= k, n/2 <= activateNum → SplitN path (L711)
TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_26)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting",
                                              {
                                                  {{{64, 128}, {64, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{64, 200}, {64, 200}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{64, 200}, {64, 200}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8000, 128}, {8000, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{12800}, {12800}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{12800}, {12800}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)}},
                                              },
                                              &compileInfo);
    int64_t expectTilingKey = 2;
    string expectTilingData = "64 64 128 200 4 3200 1 3200 3200 3200 1 3200 3200 8096 0 1024 64 0 200 0 0 0 200 200 "
                              "200 0 0 0 200 200 0 0 64 8000 1 200 200 200 1 1 1 200 200 200 1 1 128 0 "; // tilingData
    std::vector<size_t> expectWorkspaces = {17139712};                                                    // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// cols > MAX_COLS_ONE_LOOP (16376) → IsFullLoad return false at L664
TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_27)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting",
                                              {
                                                  {{{2, 20000}, {2, 20000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{6, 20000}, {6, 20000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)}},
                                              },
                                              &compileInfo);
    int64_t expectTilingKey = 1;
    string expectTilingData = "64 2 20000 3 1 6 1 6 6 6 1 6 6 8096 0 1024 6 0 1 0 0 0 1 1 1 0 0 0 1 1 0 0 3 6 2 1 1 1 "
                              "1 1 2 1 1 1 1 1 16376 1 "; // tilingData
    std::vector<size_t> expectWorkspaces = {16781480};    // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
