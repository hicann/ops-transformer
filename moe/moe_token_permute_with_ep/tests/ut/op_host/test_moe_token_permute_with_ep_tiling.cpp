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
#include "../../../op_host/moe_token_permute_with_ep_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

using namespace std;

class MoeTokenPermuteWithEpTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeTokenPermuteWithEpTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeTokenPermuteWithEpTiling TearDown" << std::endl;
    }
};

TEST_F(MoeTokenPermuteWithEpTiling, moe_token_permute_with_ep_tiling_01)
{
    optiling::MoeTokenPermuteWithEpCompileInfo compileInfo =
        {}; // 根据tiling头文件中的compileInfo填入对应值，一般原先的用例里能找到
    std::vector<int64_t> range({1, 5});
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEp", // op_name
        {                        // input info
                                 // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
         {{{2, 5}, {2, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_BF16, ge::FORMAT_ND}},
        // output info
        {{{{4, 5}, {4, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{
               6,
           },
           {
               6,
           }},
          ge::DT_INT32,
          ge::FORMAT_ND},
         {{{
               4,
           },
           {
               4,
           }},
          ge::DT_INT32,
          ge::FORMAT_ND}},
        // attr
        {{"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    int64_t expectTilingKey = 5; // tilngkey
    string expectTilingData = "1 2 5 16 3 1 6 1 6 6 6 1 6 6 8160 0 0 1024 2 2 0 1 0 10 2 2 2337 4674 14022 1 1 1 1 1 0 "
                              "4674 2 2337 1 1 4 1 5 74784 56096 56096 ";
    std::vector<size_t> expectWorkspaces = {16777376}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenPermuteWithEpTiling, moe_token_permute_with_ep_tiling_02)
{
    optiling::MoeTokenPermuteWithEpCompileInfo compileInfo =
        {}; // 根据tiling头文件中的compileInfo填入对应值，一般原先的用例里能找到
    std::vector<int64_t> range({0, 4096});
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEp", // op_name
        {                        // input info
                                 // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
         {{{4096, 5}, {4096, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{4096, 3}, {4096, 3}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{4096, 3}, {4096, 3}}, ge::DT_BF16, ge::FORMAT_ND}},
        // output info
        {{{{4096, 5}, {4096, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{4096 * 3}, {4096 * 3}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{4096}, {4096}}, ge::DT_INT32, ge::FORMAT_ND}},
        // attr
        {{"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    int64_t expectTilingKey = 6; // tilngkey
    string expectTilingData = "64 4096 5 16 3 4 3072 1 3072 3072 3072 1 3072 3072 8096 0 0 1024 64 64 0 64 64 10 2 2 "
                              "2337 4674 14022 1 1 1 1 64 1 64 1 64 1 64 4096 0 4096 74784 56096 56096 ";
    std::vector<size_t> expectWorkspaces = {16977920}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenPermuteWithEpTiling, moe_token_permute_with_ep_tiling_03)
{
    optiling::MoeTokenPermuteWithEpCompileInfo compileInfo =
        {}; // 根据tiling头文件中的compileInfo填入对应值，一般原先的用例里能找到
    std::vector<int64_t> range({0, 4096});
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEp", // op_name
        {                        // input info
                                 // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
         {{{4096, 5}, {4096, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{4096, 3}, {4096, 3}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{4096, 3}, {4096, 3}}, ge::DT_BF16, ge::FORMAT_ND}},
        // output info
        {{{{4096, 5}, {4096, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{4096 * 3}, {4096 * 3}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{4096}, {4096}}, ge::DT_INT32, ge::FORMAT_ND}},
        // attr
        {{"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    int64_t expectTilingKey = 6; // tilngkey
    string expectTilingData = "64 4096 5 16 3 4 3072 1 3072 3072 3072 1 3072 3072 8096 0 0 1024 64 64 0 64 64 10 2 2 "
                              "2337 4674 14022 1 1 1 1 64 1 64 1 64 1 64 4096 0 4096 74784 56096 56096 ";
    std::vector<size_t> expectWorkspaces = {16977920}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenPermuteWithEpTiling, moe_token_permute_with_ep_tiling_multi_core)
{
    optiling::MoeTokenPermuteWithEpCompileInfo compileInfo =
        {}; // 根据tiling头文件中的compileInfo填入对应值，一般原先的用例里能找到
    int64_t tokenNum = 16384;
    int64_t topk = 3;
    std::vector<int64_t> range({0, 4096});
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEp", // op_name
        {                        // input info
                                 // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
         {{{tokenNum, 5}, {tokenNum, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{tokenNum, 3}, {tokenNum, 3}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{tokenNum, 3}, {tokenNum, 3}}, ge::DT_BF16, ge::FORMAT_ND}},
        // output info
        {{{{4096, 5}, {4096, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{4096}, {4096}}, ge::DT_INT32, ge::FORMAT_ND}},
        // attr
        {{"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    int64_t expectTilingKey = 6; // tilngkey
    string expectTilingData = "64 16384 5 16 3 16 3072 1 3072 3072 3072 1 3072 3072 8096 0 4 1024 64 64 0 256 256 10 2 "
                              "2 2337 4674 14022 1 1 1 1 256 1 256 1 256 1 256 4096 0 4096 74784 56096 56096 ";
    std::vector<size_t> expectWorkspaces = {17567744}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenPermuteWithEpTiling, moe_token_permute_with_ep_tiling_long_h)
{
    optiling::MoeTokenPermuteWithEpCompileInfo compileInfo =
        {}; // 根据tiling头文件中的compileInfo填入对应值，一般原先的用例里能找到
    int64_t tokenNum = 16384;
    int64_t topk = 3;
    int64_t h = 32768;

    std::vector<int64_t> range({0, 4096});
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEp", // op_name
        {                        // input info
                                 // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
         {{{tokenNum, h}, {tokenNum, h}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{tokenNum, 3}, {tokenNum, 3}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{tokenNum, 3}, {tokenNum, 3}}, ge::DT_BF16, ge::FORMAT_ND}},
        // output info
        {{{{4096, h}, {4096, h}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{4096}, {4096}}, ge::DT_INT32, ge::FORMAT_ND}},
        // attr
        {{"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    int64_t expectTilingKey = 6; // tilngkey
    string expectTilingData =
        "64 16384 32768 32768 3 16 3072 1 3072 3072 3072 1 3072 3072 8096 0 4 1024 64 64 0 256 256 65536 2 5446 1 5446 "
        "16338 1 1 1 1 256 1 256 256 1 256 1 4096 0 4096 65536 65376 65376 ";
    std::vector<size_t> expectWorkspaces = {17567744}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenPermuteWithEpTiling, moe_token_permute_with_ep_tiling_error)
{
    optiling::MoeTokenPermuteWithEpCompileInfo compileInfo =
        {}; // 根据tiling头文件中的compileInfo填入对应值，一般原先的用例里能找到
    int64_t tokenNum = 16384;
    int64_t topk = 3;
    int64_t h = 32768;

    std::vector<int64_t> range({0, 4096});
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEp", // op_name
        {                        // input info
                                 // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
         {{{tokenNum, h}, {tokenNum, h}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{tokenNum, 3}, {tokenNum, 3}}, ge::DT_BF16, ge::FORMAT_ND}},
        // output info
        {{{{4096, h}, {4096, h}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{4096}, {4096}}, ge::DT_INT32, ge::FORMAT_ND}},
        // attr
        {{"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    int64_t expectTilingKey = 6; // tilngkey
    string expectTilingData = "64 16384 5 16 3 16 3072 1 3072 3072 3072 1 3072 3072 8096 0 4 1024 64 64 0 256 256 10 2 "
                              "2 2338 4676 14028 1 1 1 1 256 1 256 1 256 1 256 4096 0 4096 74816 14048 14048 ";
    std::vector<size_t> expectWorkspaces = {17567744}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenPermuteWithEpTiling, moe_token_permute_with_ep_tiling_padded_mode_true)
{
    optiling::MoeTokenPermuteWithEpCompileInfo compileInfo = {};
    std::vector<int64_t> range({0, 6});
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEp",
        {{{{2, 5}, {2, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{6, 5}, {6, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenPermuteWithEpTiling, moe_token_permute_with_ep_tiling_range_size_invalid)
{
    optiling::MoeTokenPermuteWithEpCompileInfo compileInfo = {};
    std::vector<int64_t> range({0, 6, 1});
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEp",
        {{{{2, 5}, {2, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{6, 5}, {6, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenPermuteWithEpTiling, moe_token_permute_with_ep_tiling_indices_1d)
{
    optiling::MoeTokenPermuteWithEpCompileInfo compileInfo = {};
    std::vector<int64_t> range({0, 4});
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEp",
        {{{{4, 5}, {4, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{4}, {4}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{4, 5}, {4, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    int64_t expectTilingKey = 1;
    string expectTilingData = "1 4 5 16 1 1 4 1 4 4 4 1 4 4 8160 0 0 1024 4 4 0 1 0 10 2 2 3272 6544 6544 1 1 1 1 1 0 "
                              "6544 2 3272 1 1 4 0 4 104704 26176 26176 ";
    std::vector<size_t> expectWorkspaces = {16777344};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenPermuteWithEpTiling, moe_token_permute_with_ep_tiling_indices_dim_invalid)
{
    optiling::MoeTokenPermuteWithEpCompileInfo compileInfo = {};
    std::vector<int64_t> range({0, 6});
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEp",
        {{{{2, 5}, {2, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{2, 3, 1}, {2, 3, 1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{6, 5}, {6, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenPermuteWithEpTiling, moe_token_permute_with_ep_tiling_topk_out_of_range)
{
    optiling::MoeTokenPermuteWithEpCompileInfo compileInfo = {};
    std::vector<int64_t> range({0, 6});
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEp",
        {{{{2, 5}, {2, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{2, 513}, {2, 513}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{2, 513}, {2, 513}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{6, 5}, {6, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenPermuteWithEpTiling, moe_token_permute_with_ep_tiling_range_out_of_bounds)
{
    optiling::MoeTokenPermuteWithEpCompileInfo compileInfo = {};
    std::vector<int64_t> range({0, 100});
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEp",
        {{{{2, 5}, {2, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{6, 5}, {6, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenPermuteWithEpTiling, moe_token_permute_with_ep_tiling_output_indices_dim_invalid)
{
    optiling::MoeTokenPermuteWithEpCompileInfo compileInfo = {};
    std::vector<int64_t> range({0, 6});
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEp",
        {{{{2, 5}, {2, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{6, 5}, {6, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenPermuteWithEpTiling, moe_token_permute_with_ep_tiling_output_token_dim_invalid)
{
    optiling::MoeTokenPermuteWithEpCompileInfo compileInfo = {};
    std::vector<int64_t> range({0, 6});
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEp",
        {{{{2, 5}, {2, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{6}, {6}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenPermuteWithEpTiling, moe_token_permute_with_ep_tiling_probs_dim_mismatch)
{
    optiling::MoeTokenPermuteWithEpCompileInfo compileInfo = {};
    std::vector<int64_t> range({0, 6});
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEp",
        {{{{2, 5}, {2, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{2, 5}, {2, 5}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{6, 5}, {6, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenPermuteWithEpTiling, moe_token_permute_with_ep_tiling_split_d_small_ub)
{
    optiling::MoeTokenPermuteWithEpCompileInfo compileInfo = {};
    int64_t tokenNum = 16384;
    int64_t topk = 3;
    int64_t h = 32768;
    std::vector<int64_t> range({0, 4096});
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEp",
        {{{{tokenNum, h}, {tokenNum, h}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{tokenNum, 3}, {tokenNum, 3}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{tokenNum, 3}, {tokenNum, 3}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{4096, h}, {4096, h}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{4096}, {4096}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    tilingContextPara.ubSize_ = 98304;
    int64_t expectTilingKey = 8;
    string expectTilingData = "64 16384 32768 32768 3 64 768 1 768 768 768 1 768 768 2976 0 16 1024 64 64 0 256 256 "
                              "65536 2 1 1 170 510 9472 23296 2 2 86 2 86 86 1 86 1 4096 0 4096 46592 2048 2048 ";
    std::vector<size_t> expectWorkspaces = {17567744};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenPermuteWithEpTiling, moe_token_permute_with_ep_tiling_output_indices_size_invalid)
{
    optiling::MoeTokenPermuteWithEpCompileInfo compileInfo = {};
    std::vector<int64_t> range({0, 6});
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEp",
        {{{{2, 5}, {2, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{6, 5}, {6, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenPermuteWithEpTiling, moe_token_permute_with_ep_tiling_output_hidden_size_invalid)
{
    optiling::MoeTokenPermuteWithEpCompileInfo compileInfo = {};
    std::vector<int64_t> range({0, 6});
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEp",
        {{{{2, 5}, {2, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{6, 8}, {6, 8}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenPermuteWithEpTiling, moe_token_permute_with_ep_tiling_output_token_num_invalid)
{
    optiling::MoeTokenPermuteWithEpCompileInfo compileInfo = {};
    std::vector<int64_t> range({0, 6});
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEp",
        {{{{2, 5}, {2, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{4, 5}, {4, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenPermuteWithEpTiling, moe_token_permute_with_ep_tiling_probs_dim_num_mismatch)
{
    optiling::MoeTokenPermuteWithEpCompileInfo compileInfo = {};
    std::vector<int64_t> range({0, 6});
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEp",
        {{{{2, 5}, {2, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{6}, {6}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{6, 5}, {6, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenPermuteWithEpTiling, moe_token_permute_with_ep_tiling_probs_token_dim_mismatch)
{
    optiling::MoeTokenPermuteWithEpCompileInfo compileInfo = {};
    std::vector<int64_t> range({0, 6});
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEp",
        {{{{2, 5}, {2, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{4, 3}, {4, 3}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{6, 5}, {6, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenPermuteWithEpTiling, moe_token_permute_with_ep_tiling_output_probs_dim_invalid)
{
    optiling::MoeTokenPermuteWithEpCompileInfo compileInfo = {};
    std::vector<int64_t> range({0, 6});
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEp",
        {{{{2, 5}, {2, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{6, 5}, {6, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{6, 1}, {6, 1}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenPermuteWithEpTiling, moe_token_permute_with_ep_tiling_data_accessor)
{
    optiling::IndexMixCopyComputeTilingData indexData;
    indexData.set_needCoreNum(1);
    indexData.set_frontCoreNum(2);
    indexData.set_tailCoreNum(3);
    indexData.set_coreCalcNum(4);
    indexData.set_coreCalcTail(5);
    indexData.set_oneTokenBtypeSize(6);
    indexData.set_oneProbBtypeSize(7);
    indexData.set_onceIndicesTokenMoveTimes(8);
    indexData.set_onceUbTokenNums(9);
    indexData.set_onceIndicesTokenNums(10);
    indexData.set_onceIndices(11);
    indexData.set_oneTokenlastMove(12);
    indexData.set_oneTokenOnceMove(13);
    indexData.set_oneTokenMoveTimes(14);
    indexData.set_frontCoreLoop(15);
    indexData.set_frontCoreLastTokenNums(16);
    indexData.set_tailCoreLoop(17);
    indexData.set_tailCoreLastTokenNums(18);
    indexData.set_tailLastonceIndicesTokenMoveTimes(19);
    indexData.set_tailLastIndicesLastTokenNums(20);
    indexData.set_frontLastonceIndicesTokenMoveTimes(21);
    indexData.set_frontLastIndicesLastTokenNums(22);
    indexData.set_numOutTokens(23);
    indexData.set_start(24);
    indexData.set_end(25);
    indexData.set_tokenUB(26);
    indexData.set_indicesUB(27);
    indexData.set_probsUB(28);

    EXPECT_EQ(indexData.get_needCoreNum(), 1);
    EXPECT_EQ(indexData.get_frontCoreNum(), 2);
    EXPECT_EQ(indexData.get_tailCoreNum(), 3);
    EXPECT_EQ(indexData.get_coreCalcNum(), 4);
    EXPECT_EQ(indexData.get_coreCalcTail(), 5);
    EXPECT_EQ(indexData.get_oneTokenBtypeSize(), 6);
    EXPECT_EQ(indexData.get_oneProbBtypeSize(), 7);
    EXPECT_EQ(indexData.get_onceIndicesTokenMoveTimes(), 8);
    EXPECT_EQ(indexData.get_onceUbTokenNums(), 9);
    EXPECT_EQ(indexData.get_onceIndicesTokenNums(), 10);
    EXPECT_EQ(indexData.get_onceIndices(), 11);
    EXPECT_EQ(indexData.get_oneTokenlastMove(), 12);
    EXPECT_EQ(indexData.get_oneTokenOnceMove(), 13);
    EXPECT_EQ(indexData.get_oneTokenMoveTimes(), 14);
    EXPECT_EQ(indexData.get_frontCoreLoop(), 15);
    EXPECT_EQ(indexData.get_frontCoreLastTokenNums(), 16);
    EXPECT_EQ(indexData.get_tailCoreLoop(), 17);
    EXPECT_EQ(indexData.get_tailCoreLastTokenNums(), 18);
    EXPECT_EQ(indexData.get_tailLastonceIndicesTokenMoveTimes(), 19);
    EXPECT_EQ(indexData.get_tailLastIndicesLastTokenNums(), 20);
    EXPECT_EQ(indexData.get_frontLastonceIndicesTokenMoveTimes(), 21);
    EXPECT_EQ(indexData.get_frontLastIndicesLastTokenNums(), 22);
    EXPECT_EQ(indexData.get_numOutTokens(), 23);
    EXPECT_EQ(indexData.get_start(), 24);
    EXPECT_EQ(indexData.get_end(), 25);
    EXPECT_EQ(indexData.get_tokenUB(), 26);
    EXPECT_EQ(indexData.get_indicesUB(), 27);
    EXPECT_EQ(indexData.get_probsUB(), 28);

    optiling::MoeTokenPermuteWithEpTilingData tilingData;
    tilingData.set_coreNum(64);
    tilingData.set_n(100);
    tilingData.set_cols(200);
    tilingData.set_colsAlign(256);
    tilingData.set_topK(8);
    tilingData.indexCopyComputeParamsOp.set_needCoreNum(indexData.get_needCoreNum());
    tilingData.indexCopyComputeParamsOp.set_probsUB(indexData.get_probsUB());

    EXPECT_EQ(tilingData.get_coreNum(), 64);
    EXPECT_EQ(tilingData.get_n(), 100);
    EXPECT_EQ(tilingData.get_cols(), 200);
    EXPECT_EQ(tilingData.get_colsAlign(), 256);
    EXPECT_EQ(tilingData.get_topK(), 8);
    EXPECT_EQ(tilingData.indexCopyComputeParamsOp.get_needCoreNum(), 1);
    EXPECT_EQ(tilingData.indexCopyComputeParamsOp.get_probsUB(), 28);
    EXPECT_GT(tilingData.GetDataSize(), 0);
}

TEST_F(MoeTokenPermuteWithEpTiling, moe_token_permute_tiling_base_data_accessor)
{
    optiling::PermuteVBSComputeTilingEPData vbsData;
    vbsData.set_needCoreNum(1);
    vbsData.set_perCoreElements(2);
    vbsData.set_perCoreLoops(3);
    vbsData.set_perCorePerLoopElements(4);
    vbsData.set_perCoreLastLoopElements(5);
    vbsData.set_lastCoreElements(6);
    vbsData.set_lastCoreLoops(7);
    vbsData.set_lastCorePerLoopElements(8);
    vbsData.set_lastCoreLastLoopElements(9);
    vbsData.set_oneLoopMaxElements(10);
    vbsData.set_lastCoreWSindex(11);

    EXPECT_EQ(vbsData.get_needCoreNum(), 1);
    EXPECT_EQ(vbsData.get_perCoreElements(), 2);
    EXPECT_EQ(vbsData.get_perCoreLoops(), 3);
    EXPECT_EQ(vbsData.get_perCorePerLoopElements(), 4);
    EXPECT_EQ(vbsData.get_perCoreLastLoopElements(), 5);
    EXPECT_EQ(vbsData.get_lastCoreElements(), 6);
    EXPECT_EQ(vbsData.get_lastCoreLoops(), 7);
    EXPECT_EQ(vbsData.get_lastCorePerLoopElements(), 8);
    EXPECT_EQ(vbsData.get_lastCoreLastLoopElements(), 9);
    EXPECT_EQ(vbsData.get_oneLoopMaxElements(), 10);
    EXPECT_EQ(vbsData.get_lastCoreWSindex(), 11);

    optiling::PermuteVMSMiddleComputeTilingEPData vmsData;
    vmsData.set_needCoreNum(12);
    EXPECT_EQ(vmsData.get_needCoreNum(), 12);

    optiling::PermuteSortOutComputeTilingEPData sortOutData;
    sortOutData.set_oneLoopMaxElements(13);
    EXPECT_EQ(sortOutData.get_oneLoopMaxElements(), 13);

    optiling::IndexCopyComputeTilingEPData indexCopyEpData;
    indexCopyEpData.set_needCoreNum(14);
    indexCopyEpData.set_frontCoreNum(15);
    indexCopyEpData.set_tailCoreNum(16);
    indexCopyEpData.set_coreCalcNum(17);
    indexCopyEpData.set_coreCalcTail(18);
    indexCopyEpData.set_oneTokenBtypeSize(19);
    indexCopyEpData.set_onceIndicesTokenMoveTimes(20);
    indexCopyEpData.set_onceUbTokenNums(21);
    indexCopyEpData.set_onceIndicesTokenNums(22);
    indexCopyEpData.set_onceIndices(23);
    indexCopyEpData.set_oneTokenlastMove(24);
    indexCopyEpData.set_oneTokenOnceMove(25);
    indexCopyEpData.set_oneTokenMoveTimes(26);
    indexCopyEpData.set_frontCoreLoop(27);
    indexCopyEpData.set_frontCoreLastTokenNums(28);
    indexCopyEpData.set_tailCoreLoop(29);
    indexCopyEpData.set_tailCoreLastTokenNums(30);
    indexCopyEpData.set_tailLastonceIndicesTokenMoveTimes(31);
    indexCopyEpData.set_tailLastIndicesLastTokenNums(32);
    indexCopyEpData.set_frontLastonceIndicesTokenMoveTimes(33);
    indexCopyEpData.set_frontLastIndicesLastTokenNums(34);
    indexCopyEpData.set_numOutTokens(35);
    indexCopyEpData.set_tokenUB(36);
    indexCopyEpData.set_indicesUB(37);

    EXPECT_EQ(indexCopyEpData.get_needCoreNum(), 14);
    EXPECT_EQ(indexCopyEpData.get_indicesUB(), 37);
}

TEST_F(MoeTokenPermuteWithEpTiling, moe_token_permute_with_ep_tiling_parse)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MoeTokenPermuteWithEp");
    ASSERT_NE(opImpl, nullptr);
    auto tilingParseFunc = opImpl->tiling_parse;
    ASSERT_NE(tilingParseFunc, nullptr);
    EXPECT_EQ(tilingParseFunc(nullptr), ge::GRAPH_SUCCESS);
}

TEST_F(MoeTokenPermuteWithEpTiling, moe_token_permute_with_ep_tiling_output_probs_size_mismatch)
{
    optiling::MoeTokenPermuteWithEpCompileInfo compileInfo = {};
    std::vector<int64_t> range({0, 6});
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEp",
        {{{{2, 5}, {2, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{6, 5}, {6, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenPermuteWithEpTiling, moe_token_permute_with_ep_tiling_probs_3d)
{
    optiling::MoeTokenPermuteWithEpCompileInfo compileInfo = {};
    std::vector<int64_t> range({0, 6});
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEp",
        {{{{2, 5}, {2, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{2, 3, 1}, {2, 3, 1}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{6, 5}, {6, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenPermuteWithEpTiling, moe_token_permute_with_ep_tiling_sort_limit_exceeded)
{
    optiling::MoeTokenPermuteWithEpCompileInfo compileInfo = {};
    int64_t tokenNum = 32768;
    int64_t topk = 512;
    std::vector<int64_t> range({0, tokenNum * topk});
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEp",
        {{{{tokenNum, 5}, {tokenNum, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{tokenNum, topk}, {tokenNum, topk}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{tokenNum, topk}, {tokenNum, topk}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{tokenNum * topk, 5}, {tokenNum * topk, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tokenNum * topk)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenPermuteWithEpTiling, moe_token_permute_with_ep_tiling_vbs_ceil_align_branch)
{
    optiling::MoeTokenPermuteWithEpCompileInfo compileInfo = {};
    int64_t tokenNum = 20000;
    int64_t topk = 5;
    std::vector<int64_t> range({0, 4096});
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEp",
        {{{{tokenNum, 5}, {tokenNum, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{tokenNum, topk}, {tokenNum, topk}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{tokenNum, topk}, {tokenNum, topk}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{4096, 5}, {4096, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{4096}, {4096}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 6);
}
