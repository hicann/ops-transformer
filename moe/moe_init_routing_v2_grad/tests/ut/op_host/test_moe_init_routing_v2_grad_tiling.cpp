/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/moe_init_routing_v2_grad_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class MoeInitRoutingV2GradTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MoeInitRoutingV2GradTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MoeInitRoutingV2GradTiling TearDown" << std::endl;
  }
};

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_01) {
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                            {
                                              {{{480, 8}, {480, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              {{{480}, {480}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                              {{{80, 8}, {80, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            },
                                            {
                                              {"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
                                              {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 1000;
    string expectTilingData = "64 80 0 0 8 6 0 40 2 2 1 8 8 2 1 1 32 2 1 2 1 2 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_02) {
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                            {
                                              {{{16, 5120}, {16, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                              {{{8, 5120}, {8, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                            },
                                            {
                                              {"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                              {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 1001;
    string expectTilingData = "64 8 0 0 5120 2 0 8 1 1 1 5120 5120 1 1 0 20480 1 1 1 1 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_03) {
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                            {
                                              {{{16, 5120}, {16, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
                                              {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                              {{{8, 5120}, {8, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
                                            },
                                            {
                                              {"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                              {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 1002;
    string expectTilingData = "64 8 0 0 5120 2 0 8 1 1 1 5120 5120 1 1 0 20480 1 1 1 1 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_04) {
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                            {
                                              {{{40,512},{40,512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              {{{640},{640}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                              {{{10,512},{10,512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            },
                                            {
                                              {"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                                              {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)}
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 1010;
    string expectTilingData = "64 10 0 0 512 64 40 10 1 1 1 512 512 32 1 5 2048 1 1 1 1 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_05) {
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                            {
                                              {{{40,512},{40,512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{80},{80}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                              {{{80,512},{80,512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                            },
                                            {
                                              {"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                              {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)}
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 1011;
    string expectTilingData = "64 80 0 0 512 1 40 40 2 2 1 512 512 1 1 0 2048 2 1 2 1 2 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_06) {
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                            {
                                              {{{40,512},{40,512}}, ge::DT_BF16, ge::FORMAT_ND},
                                              {{{640},{640}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                              {{{10,512},{10,512}}, ge::DT_BF16, ge::FORMAT_ND},
                                            },
                                            {
                                              {"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                                              {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)}
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 1012;
    string expectTilingData = "64 10 0 0 512 64 40 10 1 1 1 512 512 32 1 5 2048 1 1 1 1 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_07) {
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                            {
                                              {{{10,8,512},{10,8,512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              {{{640},{640}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                              {{{80,512},{80,512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            },
                                            {
                                              {"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                              {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                              {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)}
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 1100;
    string expectTilingData = "64 80 10 8 512 8 40 40 2 2 1 512 512 4 1 2 2048 2 1 2 1 2 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_08) {
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                            {
                                              {{{10,8,512},{10,8,512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{640},{640}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                              {{{80,512},{80,512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                            },
                                            {
                                              {"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                              {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                              {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)}
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 1101;
    string expectTilingData = "64 80 10 8 512 8 40 40 2 2 1 512 512 4 1 2 2048 2 1 2 1 2 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_09) {
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                            {
                                              {{{10,8,512},{10,8,512}}, ge::DT_BF16, ge::FORMAT_ND},
                                              {{{640},{640}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                              {{{80,512},{80,512}}, ge::DT_BF16, ge::FORMAT_ND},
                                            },
                                            {
                                              {"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                              {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                              {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)}
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 1102;
    string expectTilingData = "64 80 10 8 512 8 40 40 2 2 1 512 512 4 1 2 2048 2 1 2 1 2 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_10) {
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                            {
                                              {{{10,8,512},{10,8,512}}, ge::DT_BF16, ge::FORMAT_ND},
                                              {{{640},{640}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                              {{{80,512},{80,512}}, ge::DT_BF16, ge::FORMAT_ND},
                                            },
                                            {
                                              {"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                              {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                              {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)}
                                            },
                                            &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_11) {
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                            {
                                              {{{40,512},{40,512}}, ge::DT_BF16, ge::FORMAT_ND},
                                              {{{640},{640}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                              {{{10,512},{10,512}}, ge::DT_BF16, ge::FORMAT_ND},
                                            },
                                            {
                                              {"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                                              {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}
                                            },
                                            &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_12) {
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                            {
                                              {{{40,512,1024,1024},{40,512,1024,1024}}, ge::DT_BF16, ge::FORMAT_ND},
                                              {{{640},{640}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                              {{{10,512},{10,512}}, ge::DT_BF16, ge::FORMAT_ND},
                                            },
                                            {
                                              {"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                                              {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)}
                                            },
                                            &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_13) {
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                            {
                                              {{{10,512},{10,512}}, ge::DT_BF16, ge::FORMAT_ND},
                                              {{{640},{640}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                              {{{80,512},{80,512}}, ge::DT_BF16, ge::FORMAT_ND},
                                            },
                                            {
                                              {"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                              {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                              {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)}
                                            },
                                            &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_14) {
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                            {
                                              {{{196608,8},{196608,8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              {{{196607},{196607}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                              {{{4096,8},{4096,8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            },
                                            {
                                              {"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
                                              {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
                                            },
                                            &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_15) {
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                            {
                                              {{{40,512},{40,512}}, ge::DT_BF16, ge::FORMAT_ND},
                                              {{{640},{640}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                              {{{10,512},{10,512}}, ge::DT_BF16, ge::FORMAT_ND},
                                            },
                                            {
                                              {"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                                              {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(30)}
                                            },
                                            &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_16) {
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                            {
                                              {{{40,512},{40,512}}, ge::DT_BF16, ge::FORMAT_ND},
                                              {{{640},{640}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                              {{{10,510},{10,510}}, ge::DT_BF16, ge::FORMAT_ND},
                                            },
                                            {
                                              {"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                                              {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)}
                                            },
                                            &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV2GradTiling, moe_init_routing_v2_grad_tiling_17) {
    optiling::MoeInitRoutingV2GradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV2Grad",
                                            {
                                              {{{40,512},{40,512}}, ge::DT_BF16, ge::FORMAT_ND},
                                              {{{640},{640}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                              {{{10,512},{10,512}}, ge::DT_BF16, ge::FORMAT_ND},
                                            },
                                            {
                                              {"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(10)},
                                              {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)}
                                            },
                                            &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}