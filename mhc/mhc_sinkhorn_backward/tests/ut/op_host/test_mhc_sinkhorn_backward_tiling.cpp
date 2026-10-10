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
#include "../../../op_host/mhc_sinkhorn_backward_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class MhcSinkhornBackwardTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MhcSinkhornBackwardTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MhcSinkhornBackwardTiling TearDown" << std::endl;
    }
};

struct MhcSinkhornBackwardCompileInfo {
    int64_t coreNum;
    int64_t ubSize;
};

TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_4d_fp32_success)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
                                              {
                                                  // input info: grad_y, norm, sum
                                                  {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{81920}, {81920}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{20480}, {20480}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info: grad_input
                                                  {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}

TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_3d_fp32_success)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
                                              {
                                                  // input info: grad_y, norm, sum
                                                  {{{1024, 6, 6}, {1024, 6, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1474560}, {1474560}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{245760}, {245760}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info: grad_input
                                                  {{{1024, 6, 6}, {1024, 6, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}

TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_n8_4d_success)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
                                              {
                                                  // input info: grad_y, norm, sum
                                                  {{{2, 256, 8, 8}, {2, 256, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1638400}, {1638400}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{204800}, {204800}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info: grad_input
                                                  {{{2, 256, 8, 8}, {2, 256, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}

TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_invalid_dtype)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
                                              {
                                                  // input info: grad_y, norm, sum
                                                  {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{163840}, {163840}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{40960}, {40960}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info: grad_input
                                                  {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}

TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_invalid_num_iters)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
                                              {
                                                  // input info: grad_y, norm, sum
                                                  {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{829440}, {829440}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{207360}, {207360}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info: grad_input
                                                  {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}

TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_invalid_odd_num_iters)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
                                              {
                                                  // input info: grad_y, norm, sum
                                                  {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{81921}, {81921}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{20480}, {20480}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info: grad_input
                                                  {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}

TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_invalid_grad_y_shape)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
                                              {
                                                  // input info: grad_y, norm, sum (invalid 5D shape)
                                                  {{{1, 2, 128, 4, 4}, {1, 2, 128, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{163840}, {163840}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{40960}, {40960}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info: grad_input
                                                  {{{1, 2, 128, 4, 4}, {1, 2, 128, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}

TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_invalid_output_shape)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
                                              {
                                                  // input info: grad_y, norm, sum
                                                  {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{81920}, {81920}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{20480}, {20480}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info: grad_input (invalid shape)
                                                  {{{1, 128, 6, 6}, {1, 128, 6, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}

TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_invalid_norm_shape)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
                                              {
                                                  // input info: grad_y, norm, sum
                                                  {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{81919}, {81919}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{20480}, {20480}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info: grad_input
                                                  {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}

TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_invalid_sum_shape)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
                                              {
                                                  // input info: grad_y, norm, sum
                                                  {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{81920}, {81920}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{20479}, {20479}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info: grad_input
                                                  {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}

TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_4d_large_batch)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
                                              {
                                                  // input info: grad_y, norm, sum
                                                  {{{4, 256, 4, 4}, {4, 256, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{655360}, {655360}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{163840}, {163840}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info: grad_input
                                                  {{{4, 256, 4, 4}, {4, 256, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}

TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_3d_large_t)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
                                              {
                                                  // input info: grad_y, norm, sum
                                                  {{{4096, 8, 8}, {4096, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{13107200}, {13107200}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1638400}, {1638400}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info: grad_input
                                                  {{{4096, 8, 8}, {4096, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}

TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_min_num_iters)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
                                              {
                                                  // input info: grad_y, norm, sum
                                                  {{{1, 64, 4, 4}, {1, 64, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096}, {4096}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info: grad_input
                                                  {{{1, 64, 4, 4}, {1, 64, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}

TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_max_num_iters)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
                                              {
                                                  // input info: grad_y, norm, sum
                                                  {{{1, 64, 4, 4}, {1, 64, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{409600}, {409600}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{102400}, {102400}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info: grad_input
                                                  {{{1, 64, 4, 4}, {1, 64, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}

TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_3d_n4)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
                                              {
                                                  // input info: grad_y, norm, sum
                                                  {{{512, 4, 4}, {512, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{327680}, {327680}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{81920}, {81920}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info: grad_input
                                                  {{{512, 4, 4}, {512, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}

TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_4d_n6)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
                                              {
                                                  // input info: grad_y, norm, sum
                                                  {{{2, 256, 6, 6}, {2, 256, 6, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{737280}, {737280}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{122880}, {122880}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info: grad_input
                                                  {{{2, 256, 6, 6}, {2, 256, 6, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}
