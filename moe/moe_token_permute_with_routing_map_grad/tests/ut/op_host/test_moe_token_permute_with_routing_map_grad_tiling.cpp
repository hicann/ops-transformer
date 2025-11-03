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
#include <gtest/gtest.h>
#include "../../../op_host/moe_token_permute_with_routing_map_grad_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class MoeTokenPermuteWithRoutingMapGradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeTokenPermuteWithRoutingMapGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeTokenPermuteWithRoutingMapGradTiling TearDown" << std::endl;
    }
};

TEST_F(MoeTokenPermuteWithRoutingMapGradTiling, test_tiling_fp32)
{
    optiling::MoeTokenPermuteWithRoutingMapGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenPermuteWithRoutingMapGrad", // op_name
                                              {
                                                  // input info
                                                  // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                                  {{{1024, 7168}, {1024, 7168}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{512, 512}, {512, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{512, 7168}, {512, 7168}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{512, 512}, {512, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {{{"num_expert", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
                                                {"tokens_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
                                                {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}}},
                                              &compileInfo);
    int64_t expectTilingKey = 1000; // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 2 512 512 16 16 "; 
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenPermuteWithRoutingMapGradTiling, test_tiling_fp16)
{
    optiling::MoeTokenPermuteWithRoutingMapGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenPermuteWithRoutingMapGrad", // op_name
                                              {
                                                  // input info
                                                  // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                                  {{{1024, 7168}, {1024, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{512, 512}, {512, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{512, 7168}, {512, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{512, 512}, {512, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {{{"num_expert", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
                                                {"tokens_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
                                                {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}}},
                                              &compileInfo);
    int64_t expectTilingKey = 1000; // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 2 512 512 16 16 "; 
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenPermuteWithRoutingMapGradTiling, test_tiling_bf16)
{
    optiling::MoeTokenPermuteWithRoutingMapGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenPermuteWithRoutingMapGrad", // op_name
                                              {
                                                  // input info
                                                  // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                                  {{{1024, 7168}, {1024, 7168}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{512, 512}, {512, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{512, 7168}, {512, 7168}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{512, 512}, {512, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {{{"num_expert", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
                                                {"tokens_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
                                                {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}}},
                                              &compileInfo);
    int64_t expectTilingKey = 1000; // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 2 512 512 16 16 "; 
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}