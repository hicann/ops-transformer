/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_moe_token_unpermute_with_ep_tiling.cpp
 * \brief
 */
#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "log/log.h"

#include "../../../op_host/moe_token_unpermute_with_ep_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace std;
using namespace ge;
struct MoeTokenUnpermuteWithEpCompileInfo {};
class MoeTokenUnpermuteWithEpTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeTokenUnpermuteWithEpTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeTokenUnpermuteWithEpTiling TearDown" << std::endl;
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

TEST_F(MoeTokenUnpermuteWithEpTiling, test_tiling_prob_none_bf16)
{
    MoeTokenUnpermuteWithEpCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteWithEp",
		                              {
                                              {{{49152, 5120}, {49152, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
                                              {{{49152}, {49152}}, ge::DT_INT32, ge::FORMAT_ND},
                                              {{{6144, 8}, {6144, 8}}, ge::DT_BF16, ge::FORMAT_ND},
					                  },
                                      {
                                        {{{6144, 5120}, {6144, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
                                      },
                                      {
                                        {"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                        {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 6144})},
                                        {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                        {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({})}
                                      },
                                             &compileInfo);
    uint64_t expectTilingKey = 3;
    string expectTilingData = "5120 0 8 1 6144 49152 5120 1 0 96 0 96 1 0 4 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS,expectTilingKey,expectTilingData,expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteWithEpTiling, test_tiling_prob_none_bf16_2)
{
    MoeTokenUnpermuteWithEpCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteWithEp",
		                              {
                                              {{{6143, 20480}, {6143, 20480}}, ge::DT_BF16, ge::FORMAT_ND},
                                              {{{6144}, {6144}}, ge::DT_INT32, ge::FORMAT_ND},
                                              //{{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},
					                  },
                                      {
                                        {{{768, 20480}, {768, 20480}}, ge::DT_BF16, ge::FORMAT_ND},
                                      },
                                      {
                                        {"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                        {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 6144})},
                                        {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                        {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({})}
                                      },
                                             &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "20480 1 8 1 6144 6143 17920 1 2560 12 0 12 1 0 2 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS,expectTilingKey,expectTilingData,expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteWithEpTiling, test_tiling_prob_not_none_bf16)
{
    MoeTokenUnpermuteWithEpCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteWithEp",
		                              {
                                              {{{49150, 5120}, {49150, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
                                              {{{49152}, {49152}}, ge::DT_INT32, ge::FORMAT_ND},
                                              {{{6144, 8}, {6144, 8}}, ge::DT_BF16, ge::FORMAT_ND},
					                  },
                                      {
                                        {{{6144, 5120}, {6144, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
                                      },
                                      {
                                        {"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                        {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 49151})},
                                        {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                        {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({})}
                                      },
                                             &compileInfo);
    uint64_t expectTilingKey = 3;
    string expectTilingData = "5120 0 8 1 49151 49150 5120 1 0 96 0 96 1 0 4 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS,expectTilingKey,expectTilingData,expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteWithEpTiling, test_tiling_prob_none_fp16)
{
    MoeTokenUnpermuteWithEpCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteWithEp",
		                              {
                                              {{{6143, 20480}, {6143, 20480}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{6144}, {6144}}, ge::DT_INT32, ge::FORMAT_ND},
                                              //{{{6144, 8}, {6144, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
					                  },
                                      {
                                        {{{768, 20480}, {768, 20480}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                      },
                                      {
                                        {"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                        {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 6144})},
                                        {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                        {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({})}
                                      },
                                             &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "20480 1 8 1 6144 6143 17920 1 2560 12 0 12 1 0 2 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS,expectTilingKey,expectTilingData,expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteWithEpTiling, test_tiling_prob_not_none_fp16)
{
    MoeTokenUnpermuteWithEpCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteWithEp",
		                              {
                                              {{{49150, 5120}, {49150, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{49152}, {49152}}, ge::DT_INT32, ge::FORMAT_ND},
                                              {{{6144, 8}, {6144, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
					                  },
                                      {
                                        {{{6144, 5120}, {6144, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                      },
                                      {
                                        {"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                        {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 49151})},
                                        {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                        {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({})}
                                      },
                                             &compileInfo);
    uint64_t expectTilingKey = 2;
    string expectTilingData = "5120 0 8 1 49151 49150 5120 1 0 96 0 96 1 0 4 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS,expectTilingKey,expectTilingData,expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteWithEpTiling, test_tiling_prob_none_fp32)
{
    MoeTokenUnpermuteWithEpCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteWithEp",
		                              {
                                              {{{6143, 20480}, {6143, 20480}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              {{{6144}, {6144}}, ge::DT_INT32, ge::FORMAT_ND},
                                              //{{{6144, 8}, {6144, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
					                  },
                                      {
                                        {{{768, 20480}, {768, 20480}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                      },
                                      {
                                        {"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                        {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 49151})},
                                        {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                        {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({})}
                                      },
                                             &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "20480 1 8 1 49151 6143 20480 1 0 12 0 12 1 0 2 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS,expectTilingKey,expectTilingData,expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteWithEpTiling, test_tiling_prob_not_none_fp32)
{
    MoeTokenUnpermuteWithEpCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteWithEp",
		                              {
                                              {{{49150, 5120}, {49150, 5120}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              {{{49152}, {49152}}, ge::DT_INT32, ge::FORMAT_ND},
                                              {{{6144, 8}, {6144, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
					                  },
                                      {
                                        {{{6144, 5120}, {6144, 5120}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                      },
                                      {
                                        {"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                        {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 49151})},
                                        {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                        {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({})}
                                      },
                                             &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "5120 0 8 1 49151 49150 5120 1 0 96 0 96 1 0 4 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS,expectTilingKey,expectTilingData,expectWorkspaces);
}