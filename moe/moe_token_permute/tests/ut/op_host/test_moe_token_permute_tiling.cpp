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
#include "../../../op_host/moe_token_permute_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class MoeTokenPermuteTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeTokenPermuteTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeTokenPermuteTiling TearDown" << std::endl;
    }
};

TEST_F(MoeTokenPermuteTiling, MoeTokenPermute_tiling_float)
{
    optiling::MoeTokenPermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermute",
        {
            {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);
    int64_t expectTilingKey = 1;
    string expectTilingData = "1 2 5 8 3 1 6 1 6 6 6 1 6 6 8160 0 0 1024 2 2 0 1 0 20 2 2976 5952 17856 1 1 1 1 1 0 "
                              "5952 2 2976 1 1 6 95232 71424 ";
    std::vector<size_t> expectWorkspaces = {16777424};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenPermuteTiling, MoeTokenPermute_tiling_float16)
{
    optiling::MoeTokenPermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermute",
        {
            {{{2, 5}, {2, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{6, 5}, {6, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);
    int64_t expectTilingKey = 1;
    string expectTilingData = "1 2 5 16 3 1 6 1 6 6 6 1 6 6 8160 0 0 1024 2 2 0 1 0 10 2 2976 5952 17856 1 1 1 1 1 0 "
                              "5952 2 2976 1 1 6 95232 71424 ";
    std::vector<size_t> expectWorkspaces = {16777424};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenPermuteTiling, MoeTokenPermute_tiling_float16_multi_core)
{
    optiling::MoeTokenPermuteCompileInfo compileInfo = {};
    int64_t tokenNum = 16384;
    int64_t topk = 3;

    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermute",
        {
            {{{tokenNum, 5}, {tokenNum, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{tokenNum, topk}, {tokenNum, topk}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{tokenNum * topk, 5}, {tokenNum * topk, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tokenNum * topk)},
            {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);
    int64_t expectTilingKey = 2;
    string expectTilingData = "64 16384 5 16 3 16 3072 1 3072 3072 3072 1 3072 3072 8096 0 4 1024 64 64 0 256 256 10 2 "
                              "2976 5952 17856 1 1 1 1 256 1 256 1 256 1 256 49152 95232 71424 ";
    std::vector<size_t> expectWorkspaces = {17960960};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenPermuteTiling, MoeTokenPermute_tiling_float16_long)
{
    optiling::MoeTokenPermuteCompileInfo compileInfo = {};
    int64_t tokenNum = 16384;
    int64_t topk = 3;
    int64_t h = 32768;

    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermute",
        {
            {{{tokenNum, h}, {tokenNum, h}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{tokenNum, topk}, {tokenNum, topk}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{tokenNum * topk, h}, {tokenNum * topk, h}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tokenNum * topk)},
            {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);
    int64_t expectTilingKey = 2;
    string expectTilingData = "64 16384 32768 32768 3 16 3072 1 3072 3072 3072 1 3072 3072 8096 0 4 1024 64 64 0 256 "
                              "256 65536 10901 1 10901 32703 1 1 1 1 256 1 256 256 1 256 1 49152 65536 130816 ";
    std::vector<size_t> expectWorkspaces = {17960960};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenPermuteTiling, MoeTokenPermute_tiling_float16_error)
{
    optiling::MoeTokenPermuteCompileInfo compileInfo = {};
    int64_t tokenNum = 16384;
    int64_t topk = 3;
    int64_t h = 32768;

    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermute",
        {
            {{{tokenNum, h}, {tokenNum, h}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{tokenNum, 5}, {tokenNum, 5}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{tokenNum * topk, h}, {tokenNum * topk, h}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tokenNum * topk)},
            {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);
    int64_t expectTilingKey = 2;
    string expectTilingData = "64 16384 5 16 3 16 3072 1 3072 3072 3072 1 3072 3072 8096 0 4 1024 64 64 0 256 256 10 2 "
                              "2976 5952 17856 1 1 1 1 256 1 256 1 256 1 256 49152 95232 17856 ";
    std::vector<size_t> expectWorkspaces = {17960960};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenPermuteTiling, MoeTokenPermute_tiling_padded_mode_error)
{
    optiling::MoeTokenPermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermute",
        {
            {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(MoeTokenPermuteTiling, MoeTokenPermute_tiling_token_indices_dim_mismatch)
{
    optiling::MoeTokenPermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermute",
        {
            {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 3}, {3, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(MoeTokenPermuteTiling, MoeTokenPermute_tiling_indices_dim_invalid)
{
    optiling::MoeTokenPermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermute",
        {
            {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 3, 1}, {2, 3, 1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(MoeTokenPermuteTiling, MoeTokenPermute_tiling_topk_exceed_max)
{
    optiling::MoeTokenPermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermute",
        {
            {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 513}, {2, 513}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1026, 5}, {1026, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1026}, {1026}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1026)},
            {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(MoeTokenPermuteTiling, MoeTokenPermute_tiling_output_token_dim_error)
{
    optiling::MoeTokenPermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermute",
        {
            {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(MoeTokenPermuteTiling, MoeTokenPermute_tiling_output_indices_dim_error)
{
    optiling::MoeTokenPermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermute",
        {
            {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{6, 1}, {6, 1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(MoeTokenPermuteTiling, MoeTokenPermute_tiling_output_cols_mismatch)
{
    optiling::MoeTokenPermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermute",
        {
            {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{6, 10}, {6, 10}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(MoeTokenPermuteTiling, MoeTokenPermute_tiling_output_num_out_tokens_mismatch)
{
    optiling::MoeTokenPermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermute",
        {
            {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{5, 5}, {5, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(MoeTokenPermuteTiling, MoeTokenPermute_tiling_indices_1d)
{
    optiling::MoeTokenPermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermute",
        {
            {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);
    int64_t expectTilingKey = 1;
    string expectTilingData = "1 2 5 8 1 1 2 1 2 2 2 1 2 2 8160 0 0 1024 2 2 0 1 0 20 2 3637 7274 7274 1 1 1 1 1 0 "
                              "7274 2 3637 1 1 2 116384 29120 ";
    std::vector<size_t> expectWorkspaces = {16777328};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenPermuteTiling, MoeTokenPermute_tiling_partial_num_out_tokens)
{
    optiling::MoeTokenPermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermute",
        {
            {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);
    int64_t expectTilingKey = 5;
    string expectTilingData = "1 2 5 8 3 1 6 1 6 6 6 1 6 6 8160 0 0 1024 2 2 0 1 0 20 2 2976 5952 17856 1 1 1 1 1 0 "
                              "5952 2 2976 1 1 3 95232 71424 ";
    std::vector<size_t> expectWorkspaces = {16777424};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenPermuteTiling, MoeTokenPermute_tiling_split_d_small_ub)
{
    optiling::MoeTokenPermuteCompileInfo compileInfo = {};
    int64_t tokenNum = 2;
    int64_t topk = 3;
    int64_t h = 32768;
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermute",
        {
            {{{tokenNum, h}, {tokenNum, h}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{tokenNum, topk}, {tokenNum, topk}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{tokenNum * topk, h}, {tokenNum * topk, h}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tokenNum * topk)},
            {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend910B", 64, 65536);
    int64_t expectTilingKey = 3;
    string expectTilingData = "1 2 32768 32768 3 1 6 1 6 6 6 1 6 6 2016 0 0 1024 2 2 0 1 0 65536 1 1 170 510 1536 "
                              "15616 3 1 1 0 170 170 1 1 1 6 31232 2048 ";
    std::vector<size_t> expectWorkspaces = {16777424};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenPermuteTiling, MoeTokenPermute_tiling_data_field_access)
{
    optiling::PermuteVBSComputeTilingData vbsData;
    vbsData.set_lastCorePerLoopElements(64);
    vbsData.set_lastCoreLastLoopElements(32);
    vbsData.set_oneLoopMaxElements(1024);
    vbsData.set_lastCoreWSindex(1);
    EXPECT_EQ(vbsData.get_lastCorePerLoopElements(), 64);
    EXPECT_EQ(vbsData.get_lastCoreLastLoopElements(), 32);
    EXPECT_EQ(vbsData.get_oneLoopMaxElements(), 1024);
    EXPECT_EQ(vbsData.get_lastCoreWSindex(), 1);

    optiling::PermuteVMSMiddleComputeTilingData vmsData;
    vmsData.set_needCoreNum(4);
    EXPECT_EQ(vmsData.get_needCoreNum(), 4);

    optiling::PermuteSortOutComputeTilingData sortOutData;
    sortOutData.set_oneLoopMaxElements(1024);
    EXPECT_EQ(sortOutData.get_oneLoopMaxElements(), 1024);

    optiling::MoeTokenPermuteTilingData tilingData;
    tilingData.set_coreNum(64);
    EXPECT_EQ(tilingData.get_coreNum(), 64);
}
