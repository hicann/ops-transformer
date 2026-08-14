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
#include "../../../op_host/rotary_position_embedding_grad_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class RotaryPositionEmbeddingGradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "RotaryPositionEmbeddingGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RotaryPositionEmbeddingGradTiling TearDown" << std::endl;
    }
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend910B"}};
    std::map<std::string, std::string> short_soc_version = {{"Short_SoC_version", "Ascend950"}};

    string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_fix_pipe_l0c2out": false,
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "Intrinsic_data_move_out2l1_nd2nz": false,
                                                       "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                       "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                       "CORE_NUM": 48}
                                    })";
};

TEST_F(RotaryPositionEmbeddingGradTiling, RotaryPositionEmbeddingGradTiling_fp16_001)
{
    optiling::RotaryPositionEmbeddingGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("RotaryPositionEmbeddingGrad",
                                              {
                                                  // input info
                                                  {{{2, 64, 4, 10}, {2, 64, 4, 10}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 64, 1, 10}, {1, 64, 1, 10}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 64, 1, 10}, {1, 64, 1, 10}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{2, 64, 4, 10}, {2, 64, 4, 10}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{2, 64, 4, 10}, {2, 64, 4, 10}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 64, 1, 10}, {1, 64, 1, 10}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 64, 1, 10}, {1, 64, 1, 10}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 21000;
    string expectTilingData = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 64 4 10 "
                              "16 6 64 0 1 1 1 1 0 1 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RotaryPositionEmbeddingGradTiling, RotaryPositionEmbeddingGradTiling_bf16_001)
{
    optiling::RotaryPositionEmbeddingGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("RotaryPositionEmbeddingGrad",
                                              {
                                                  // input info
                                                  {{{2, 64, 4, 10}, {2, 64, 4, 10}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 64, 1, 10}, {1, 64, 1, 10}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 64, 1, 10}, {1, 64, 1, 10}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{2, 64, 4, 10}, {2, 64, 4, 10}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{2, 64, 4, 10}, {2, 64, 4, 10}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 64, 1, 10}, {1, 64, 1, 10}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 64, 1, 10}, {1, 64, 1, 10}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 21010;
    string expectTilingData = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 64 4 10 "
                              "16 6 64 0 1 1 1 1 0 1 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RotaryPositionEmbeddingGradTiling, RotaryPositionEmbeddingGradTiling_fp32_001)
{
    optiling::RotaryPositionEmbeddingGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("RotaryPositionEmbeddingGrad",
                                              {
                                                  // input info
                                                  {{{2, 64, 4, 10}, {2, 64, 4, 10}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 64, 1, 10}, {1, 64, 1, 10}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 64, 1, 10}, {1, 64, 1, 10}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 64, 4, 10}, {2, 64, 4, 10}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{2, 64, 4, 10}, {2, 64, 4, 10}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 64, 1, 10}, {1, 64, 1, 10}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 64, 1, 10}, {1, 64, 1, 10}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 21020;
    string expectTilingData = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 64 4 10 "
                              "16 6 64 0 1 1 1 1 0 1 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RotaryPositionEmbeddingGradTiling, RotaryPositionEmbeddingGradTiling_fp16_3d_bsd_broadcast_a5)
{
    optiling::RotaryPositionEmbeddingGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("RotaryPositionEmbeddingGrad",
                                              {
                                                  // input info
                                                  {{{8, 4096, 128}, {8, 4096, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 4096, 128}, {1, 4096, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 4096, 128}, {1, 4096, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{8, 4096, 128}, {8, 4096, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{8, 4096, 128}, {8, 4096, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 4096, 128}, {1, 4096, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 4096, 128}, {1, 4096, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              },
                                              &compileInfo, "Ascend950", 40, 196608);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_EQ(tilingInfo.tilingKey, 17314107473);
    ASSERT_EQ(tilingInfo.blockNum, 40);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    ASSERT_EQ(tilingInfo.workspaceSizes[0], 25165824);

    auto tilingData = reinterpret_cast<const int64_t *>(tilingInfo.tilingData.get());
    // ropeGradParams 起始下标 = reduceTiling 占用的 66 个 int64 之后
    EXPECT_EQ(tilingData[66], 8);    // B = T
    EXPECT_EQ(tilingData[67], 4096); // S = N
    EXPECT_EQ(tilingData[68], 128);  // D
    EXPECT_EQ(tilingData[69], 1);    // N = 1, TND 广播场景重映射为 BS1D
    EXPECT_EQ(tilingData[70], 8);    // blockNumB
    EXPECT_EQ(tilingData[71], 1);    // blockFactorB
    EXPECT_EQ(tilingData[72], 5);    // blockNumS
    EXPECT_EQ(tilingData[73], 820);  // blockFactorS
    EXPECT_EQ(tilingData[79], 40);   // usedCoreNum = blockNumB * blockNumS，40 核用满
    EXPECT_EQ(tilingData[80], 0);    // rotaryMode
    // rotaryXParams: b/s/n 同样为重映射后的值
    EXPECT_EQ(tilingData[97], 8);       // rotaryX B = T
    EXPECT_EQ(tilingData[98], 4096);    // rotaryX S = N
    EXPECT_EQ(tilingData[100], 1);      // rotaryX N = 1
    EXPECT_EQ(tilingData[101], 40);     // rotaryX usedCoreNum
    EXPECT_EQ(tilingData[106], 1);      // dCosFlag = 1, isEmptyDy = 0
    EXPECT_EQ(tilingData[107], 524288); // cosShapeSize = 4096 * 128
}

TEST_F(RotaryPositionEmbeddingGradTiling, RotaryPositionEmbeddingGradTiling_fp32_TND)
{
    optiling::RotaryPositionEmbeddingGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("RotaryPositionEmbeddingGrad",
                                              {
                                                  // input info
                                                  {{{64, 4, 10}, {64, 4, 10}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{64, 1, 10}, {64, 1, 10}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{64, 1, 10}, {64, 1, 10}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{64, 4, 10}, {64, 4, 10}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{64, 4, 10}, {64, 4, 10}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{64, 1, 10}, {64, 1, 10}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{64, 1, 10}, {64, 1, 10}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 21020;
    string expectTilingData = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 64 4 10 "
                              "16 6 64 0 1 1 1 1 0 1 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
