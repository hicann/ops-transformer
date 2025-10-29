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
#include "../../../op_host/moe_compute_expert_tokens_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class MoeComputeExpertTokensTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeComputeExpertTokensTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeComputeExpertTokensTiling TearDown" << std::endl;
    }
};

static string TilingData2Str(const gert::TilingData *tiling_data)
{
    auto data = tiling_data->GetData();
    string result;
    for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int64_t)) {
        result += std::to_string((reinterpret_cast<const int64_t *>(tiling_data->GetData())[i / sizeof(int64_t)]));
        result += " ";
    }

    return result;
}

TEST_F(MoeComputeExpertTokensTiling, MoeComputeExpertTokens_tiling_int32_1)
{
    optiling::MoeComputeExpertTokensCompileInfo compileInfo = {
        40, 65536}; // 根据tiling头文件中的compileInfo填入对应值
    gert::TilingContextPara tilingContextPara("MoeComputeExpertTokens", // op_name
                                              {
                                                  // input info
                                                  {{{100}, {100}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{98}, {98}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {{{"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(98)}}},
                                              &compileInfo);
    int64_t expectTilingKey = 1001; // tilngkey
    string expectTilingData = "64 50 1 49 261888 24 100 2 1 2 2 2 1 2 2 98 2 1 2 2 2 1 2 2 320 100 1 1 320 100 1 98 98 1 100 0 0 16802304 1001 "; // tilingData
    std::vector<size_t> expectWorkspaces = {16802304};    // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeComputeExpertTokensTiling, MoeComputeExpertTokens_tiling_int32_2)
{
    optiling::MoeComputeExpertTokensCompileInfo compileInfo = {
        40, 65536}; // 根据tiling头文件中的compileInfo填入对应值
    gert::TilingContextPara tilingContextPara("MoeComputeExpertTokens", // op_name
                                              {
                                                  // input info
                                                  {{{20000}, {20000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{100}, {100}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {{{"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(98)}}},
                                              &compileInfo);
    int64_t expectTilingKey = 1002; // tilngkey
    string expectTilingData = "64 64 63 49 261888 32 20000 2 1 2 2 2 1 2 2 98 313 1 313 313 281 1 281 281 320 160 1 1 320 160 1 98 98 1 160 0 0 16802304 1002 "; // tilingData
    std::vector<size_t> expectWorkspaces = {16802304};    // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeComputeExpertTokensTiling, MoeComputeExpertTokens_tiling_int32_3)
{
    optiling::MoeComputeExpertTokensCompileInfo compileInfo = {
        40, 65536}; // 根据tiling头文件中的compileInfo填入对应值
    gert::TilingContextPara tilingContextPara("MoeComputeExpertTokens", // op_name
                                              {
                                                  // input info
                                                  {{{65536}, {65536}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {{{"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(98)}}},
                                              &compileInfo);
    int64_t expectTilingKey = 1002; // tilngkey
    string expectTilingData = "64 64 52 49 261888 48 65536 2 1 2 2 2 1 2 2 98 1024 4 320 64 1024 4 320 64 1280 256 4 1 320 256 1 98 98 1 256 0 0 16802304 1002 "; // tilingData
    std::vector<size_t> expectWorkspaces = {16802304};    // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}