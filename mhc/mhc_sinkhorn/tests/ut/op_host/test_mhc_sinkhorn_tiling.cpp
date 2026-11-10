/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/mhc_sinkhorn_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class MhcSinkhornTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MhcSinkhornTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MhcSinkhornTiling TearDown" << std::endl;
    }
};

struct MhcSinkhornCompileInfo {
    int64_t coreNum;
    int64_t ubSize;
};

// ==================== 950 用例 ====================

TEST_F(MhcSinkhornTiling, test_mhc_sinkhorn_4d_fp16_950_success)
{
    MhcSinkhornCompileInfo compileInfo = {64, 64};
    gert::TilingContextPara tilingContextPara("MhcSinkhorn",
                                              {
                                                  {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 128, 4, 4}, {1024, 4, 5120}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
                                                  {"num_iters", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                  {"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              },
                                              &compileInfo, "Ascend950");
    uint64_t expectTilingKey = 1;
    string expectTilingDataStr = "4 5120 64 16 16 1 1024 1 5120 1 5120 5120 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}

TEST_F(MhcSinkhornTiling, test_mhc_sinkhorn_3d_fp16_950_success)
{
    MhcSinkhornCompileInfo compileInfo = {64, 64};
    gert::TilingContextPara tilingContextPara("MhcSinkhorn",
                                              {
                                                  {{{1024, 6, 6}, {1024, 6, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1024, 6, 6}, {1024, 6, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
                                                  {"num_iters", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                  {"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              },
                                              &compileInfo, "Ascend950");
    uint64_t expectTilingKey = 1;
    string expectTilingDataStr = "4 5120 64 16 16 1 1024 1 5120 1 5120 5120 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}

TEST_F(MhcSinkhornTiling, test_mhc_sinkhorn_4d_fp32_950_success)
{
    MhcSinkhornCompileInfo compileInfo = {64, 64};
    gert::TilingContextPara tilingContextPara("MhcSinkhorn",
                                              {
                                                  {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 128, 4, 4}, {1024, 4, 5120}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
                                                  {"num_iters", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                  {"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              },
                                              &compileInfo, "Ascend950");
    uint64_t expectTilingKey = 1;
    string expectTilingDataStr = "4 5120 64 16 16 1 1024 1 5120 1 5120 5120 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}

// ==================== 960PR 用例 ====================

TEST_F(MhcSinkhornTiling, test_mhc_sinkhorn_4d_fp16_960pr_success)
{
    MhcSinkhornCompileInfo compileInfo = {64, 64};
    gert::TilingContextPara tilingContextPara("MhcSinkhorn",
                                              {
                                                  {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 128, 4, 4}, {1024, 4, 5120}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
                                                  {"num_iters", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                  {"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              },
                                              &compileInfo, "Ascend960PR");
    uint64_t expectTilingKey = 1;
    string expectTilingDataStr = "4 5120 64 16 16 1 1024 1 5120 1 5120 5120 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}

TEST_F(MhcSinkhornTiling, test_mhc_sinkhorn_3d_fp16_960pr_success)
{
    MhcSinkhornCompileInfo compileInfo = {64, 64};
    gert::TilingContextPara tilingContextPara("MhcSinkhorn",
                                              {
                                                  {{{1024, 6, 6}, {1024, 6, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1024, 6, 6}, {1024, 6, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
                                                  {"num_iters", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                  {"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              },
                                              &compileInfo, "Ascend960PR");
    uint64_t expectTilingKey = 1;
    string expectTilingDataStr = "4 5120 64 16 16 1 1024 1 5120 1 5120 5120 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}

TEST_F(MhcSinkhornTiling, test_mhc_sinkhorn_4d_fp32_960pr_success)
{
    MhcSinkhornCompileInfo compileInfo = {64, 64};
    gert::TilingContextPara tilingContextPara("MhcSinkhorn",
                                              {
                                                  {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 128, 4, 4}, {1024, 4, 5120}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
                                                  {"num_iters", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                  {"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              },
                                              &compileInfo, "Ascend960PR");
    uint64_t expectTilingKey = 1;
    string expectTilingDataStr = "4 5120 64 16 16 1 1024 1 5120 1 5120 5120 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}

// ==================== 950 异常/边界用例 ====================

TEST_F(MhcSinkhornTiling, test_mhc_sinkhorn_null_input_950_failed)
{
    MhcSinkhornCompileInfo compileInfo = {64, 64};
    gert::TilingContextPara tilingContextPara("MhcSinkhorn",
                                              {
                                                  // 空输入
                                              },
                                              {
                                                  {{{1, 128, 4, 4}, {1024, 4, 5120}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
                                                  {"num_iters", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                  {"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              },
                                              &compileInfo, "Ascend950");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ==================== 960DT 用例 ====================

TEST_F(MhcSinkhornTiling, test_mhc_sinkhorn_4d_fp16_960dt_success)
{
    MhcSinkhornCompileInfo compileInfo = {64, 64};
    gert::TilingContextPara tilingContextPara("MhcSinkhorn",
                                              {
                                                  {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 128, 4, 4}, {1024, 4, 5120}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
                                                  {"num_iters", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                  {"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              },
                                              &compileInfo, "Ascend960DT");
    uint64_t expectTilingKey = 1;
    string expectTilingDataStr = "4 5120 64 16 16 1 1024 1 5120 1 5120 5120 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}

TEST_F(MhcSinkhornTiling, test_mhc_sinkhorn_3d_fp16_960dt_success)
{
    MhcSinkhornCompileInfo compileInfo = {64, 64};
    gert::TilingContextPara tilingContextPara("MhcSinkhorn",
                                              {
                                                  {{{1024, 6, 6}, {1024, 6, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1024, 6, 6}, {1024, 6, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
                                                  {"num_iters", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                  {"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              },
                                              &compileInfo, "Ascend960DT");
    uint64_t expectTilingKey = 1;
    string expectTilingDataStr = "4 5120 64 16 16 1 1024 1 5120 1 5120 5120 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}

TEST_F(MhcSinkhornTiling, test_mhc_sinkhorn_4d_fp32_960dt_success)
{
    MhcSinkhornCompileInfo compileInfo = {64, 64};
    gert::TilingContextPara tilingContextPara("MhcSinkhorn",
                                              {
                                                  {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 128, 4, 4}, {1024, 4, 5120}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
                                                  {"num_iters", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                  {"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              },
                                              &compileInfo, "Ascend960DT");
    uint64_t expectTilingKey = 1;
    string expectTilingDataStr = "4 5120 64 16 16 1 1024 1 5120 1 5120 5120 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}
