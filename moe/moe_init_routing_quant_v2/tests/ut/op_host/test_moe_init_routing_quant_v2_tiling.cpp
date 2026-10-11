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
#include <gtest/gtest.h>
#include "../../../op_host/moe_init_routing_quant_v2_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

struct MoeInitRoutingQuantV2CompileInfo {};

class MoeInitRoutingQuantV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeInitRoutingQuantV2Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeInitRoutingQuantV2Tiling TearDown" << std::endl;
    }
};

gert::TilingContextPara RunNormalCase(int64_t N, int64_t H, int64_t K, int64_t activeNum, int64_t C, int64_t E,
                                      int64_t dropPadMode, int64_t countFlag, bool tokenFlag, int64_t quantMode,
                                      int64_t dqFlag, ge::DataType optionalDt, int64_t optionalDtypePosi)
{
    MoeInitRoutingQuantV2CompileInfo compileInfo;
    // 根据 optionalDtypePosi 确定数据类型
    ge::DataType dtScale = optionalDtypePosi == 0 ? optionalDt : ge::DT_FLOAT;
    ge::DataType dtOffset = optionalDtypePosi == 1 ? optionalDt : ge::DT_FLOAT;
    ge::DataType dtDynamic = optionalDtypePosi == 2 ? optionalDt : ge::DT_FLOAT;
    ge::DataType dtOut = optionalDtypePosi == 3 ? optionalDt : ge::DT_INT8;

    // 计算输入形状
    std::vector<gert::TilingContextPara::TensorDescription> inputs;
    // x: [N, H]
    inputs.emplace_back(gert::StorageShape({N, H}, {N, H}), ge::DT_FLOAT16, ge::FORMAT_ND);
    // expert_idx: [N, K]
    inputs.emplace_back(gert::StorageShape({N, K}, {N, K}), ge::DT_INT32, ge::FORMAT_ND);

    // scale 形状根据 quantMode 变化
    if (quantMode == 0) {
        // 静态量化: [1]
        inputs.emplace_back(gert::StorageShape({1}, {1}), dtScale, ge::FORMAT_ND);
    } else {
        if (dqFlag == 0) {
            // 动态量化: [E, H]
            inputs.emplace_back(gert::StorageShape({E, H}, {E, H}), dtScale, ge::FORMAT_ND);
        } else {
            // 动态量化: [1, H]
            inputs.emplace_back(gert::StorageShape({1, H}, {1, H}), dtScale, ge::FORMAT_ND);
        }
    }

    // offset 形状根据 quantMode 变化
    if (quantMode == 0) {
        // 静态量化: [1]
        inputs.emplace_back(gert::StorageShape({1}, {1}), dtOffset, ge::FORMAT_ND);
    } else {
        if (dqFlag == 0) {
            // 动态量化: [E, H]
            inputs.emplace_back(gert::StorageShape({E, H}, {E, H}), dtOffset, ge::FORMAT_ND);
        } else {
            // 动态量化: [1, H]
            inputs.emplace_back(gert::StorageShape({1, H}, {1, H}), dtOffset, ge::FORMAT_ND);
        }
    }

    // 计算输出形状
    std::vector<gert::TilingContextPara::TensorDescription> outputs;

    // expanded_x (形状根据 dropPadMode 变化)
    if (dropPadMode == 0) {
        int64_t first_dim = N * K;
        if (activeNum > 0 && activeNum < first_dim) {
            first_dim = activeNum;
        }
        // [first_dim, H]
        outputs.emplace_back(gert::StorageShape({first_dim, H}, {first_dim, H}), dtOut, ge::FORMAT_ND);
    } else {
        // [E, C, H]
        outputs.emplace_back(gert::StorageShape({E, C, H}, {E, C, H}), dtOut, ge::FORMAT_ND);
    }
    // expanded_row_idx [N*K]
    outputs.emplace_back(gert::StorageShape({N * K}, {N * K}), ge::DT_INT32, ge::FORMAT_ND);
    // expert_tokens_count_or_cumsum [E]
    outputs.emplace_back(gert::StorageShape({E}, {E}), ge::DT_INT32, ge::FORMAT_ND);

    // expert_tokens_before_capacity [E]
    outputs.emplace_back(gert::StorageShape({E}, {E}), ge::DT_INT32, ge::FORMAT_ND);
    // dynamic_quant_scale (形状根据 dropPadMode 变化)
    if (dropPadMode == 0) {
        int64_t first_dim = N * K;
        if (activeNum > 0 && activeNum < first_dim) {
            first_dim = activeNum;
        }
        // [first_dim]
        outputs.emplace_back(gert::StorageShape({first_dim}, {first_dim}), dtDynamic, ge::FORMAT_ND);
    } else {
        // [E * C]
        outputs.emplace_back(gert::StorageShape({E * C}, {E * C}), dtDynamic, ge::FORMAT_ND);
    }
    // 设置属性
    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
        {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(C)},
        {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(E)},
        {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
        {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(countFlag)},
        {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tokenFlag)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)}};

    // 创建 TilingContextPara
    return gert::TilingContextPara("MoeInitRoutingQuantV2", inputs, outputs, attrs, &compileInfo);
}
// 单核+静态quant+drop  10100
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_01)
{
    auto tilingContextPara = RunNormalCase(8, 30, 6, 0, 6, 8, 1, 0, true, 0, 0, ge::DT_FLOAT, 0);
    uint64_t expectTilingKey = 10100;
    string expectTilingData = "64 8 30 6 6 8 1 0 1 0 0 0 1 48 1 48 48 48 1 48 48 8160 0 2040 48 0 1 1 1 1 1 1 0 0 0 0 "
                              "0 48 0 1 1 1 1 1 1 1 1 30 30 1 48 48 1 1 1 1 1 1 1 1 30 30 1 ";
    std::vector<size_t> expectWorkspaces = {16779296};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 单核+静态quant+dropless  20000
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_02)
{
    auto tilingContextPara = RunNormalCase(
        /*N=*/80, /*H=*/3000, /*K=*/60,
        /*activeNum=*/0, /*C=*/6, /*E=*/8,
        /*dropPadMode=*/0, /*countFlag=*/1, /*tokenFlag=*/false,
        /*quantMode=*/0, /*dqFlag=*/0,
        /*optionalDt=*/ge::DT_FLOAT, /*optionalDtypePosi=*/0);

    uint64_t expectTilingKey = 20000;
    string expectTilingData =
        "64 80 3000 60 6 8 0 1 0 0 0 0 1 4800 1 4800 4800 4800 1 4800 4800 8160 0 2040 64 0 75 75 75 75 75 75 0 0 0 0 "
        "0 64 0 75 75 75 75 75 75 1 1 3000 3000 1 64 4800 75 75 75 75 75 75 1 1 3000 3000 1 ";
    std::vector<size_t> expectWorkspaces = {16931360};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 单核+动态quant+drop  11100
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_03)
{
    auto tilingContextPara = RunNormalCase(
        /*N=*/8, /*H=*/30, /*K=*/6,
        /*activeNum=*/0, /*C=*/6, /*E=*/8,
        /*dropPadMode=*/1, /*countFlag=*/0, /*tokenFlag=*/true,
        /*quantMode=*/1, /*dqFlag=*/0,
        /*optionalDt=*/ge::DT_FLOAT, /*optionalDtypePosi=*/0);

    uint64_t expectTilingKey = 11100;
    string expectTilingData = "64 8 30 6 6 8 1 0 1 2 0 0 1 48 1 48 48 48 1 48 48 8160 0 2040 48 0 1 1 1 1 1 1 0 0 0 0 "
                              "0 48 0 1 1 1 1 1 1 1 1 30 30 1 48 48 1 1 1 1 1 1 1 1 30 30 1 ";
    std::vector<size_t> expectWorkspaces = {16779296};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 单核+动态quant+dropless  11000
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_04)
{
    auto tilingContextPara = RunNormalCase(
        /*N=*/80, /*H=*/3000, /*K=*/60,
        /*activeNum=*/0, /*C=*/6, /*E=*/8,
        /*dropPadMode=*/0, /*countFlag=*/1, /*tokenFlag=*/false,
        /*quantMode=*/1, /*dqFlag=*/0,
        /*optionalDt=*/ge::DT_FLOAT, /*optionalDtypePosi=*/0);

    uint64_t expectTilingKey = 21000;
    string expectTilingData =
        "64 80 3000 60 6 8 0 1 0 2 0 0 1 4800 1 4800 4800 4800 1 4800 4800 8160 0 2040 64 0 75 75 75 75 75 75 0 0 0 0 "
        "0 64 0 75 75 75 75 75 75 1 1 3000 3000 1 64 4800 75 75 75 75 75 75 1 1 3000 3000 1 ";
    std::vector<size_t> expectWorkspaces = {16931360};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 多核+静态quant+drop  10110
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_05)
{
    auto tilingContextPara = RunNormalCase(
        /*N=*/320, /*H=*/3000, /*K=*/56,
        /*activeNum=*/0, /*C=*/200, /*E=*/32,
        /*dropPadMode=*/1, /*countFlag=*/0, /*tokenFlag=*/true,
        /*quantMode=*/0, /*dqFlag=*/0,
        /*optionalDt=*/ge::DT_FLOAT, /*optionalDtypePosi=*/0);

    uint64_t expectTilingKey = 10110;
    string expectTilingData =
        "64 320 3000 56 200 32 1 0 1 0 0 0 4 4480 1 4480 4480 4480 1 4480 4480 8160 0 2040 64 0 280 280 280 280 280 "
        "280 0 0 0 0 0 64 0 280 280 280 280 280 280 1 1 3000 3000 1 64 17920 280 280 280 280 280 280 1 1 3000 3000 1 ";
    std::vector<size_t> expectWorkspaces = {17351296};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 多核+静态quant+drop + 切H  10110
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_06)
{
    auto tilingContextPara = RunNormalCase(
        /*N=*/320, /*H=*/30000, /*K=*/56,
        /*activeNum=*/0, /*C=*/200, /*E=*/32,
        /*dropPadMode=*/1, /*countFlag=*/0, /*tokenFlag=*/true,
        /*quantMode=*/0, /*dqFlag=*/0,
        /*optionalDt=*/ge::DT_FLOAT, /*optionalDtypePosi=*/0);

    uint64_t expectTilingKey = 10110;
    string expectTilingData = "64 320 30000 56 200 32 1 0 1 0 0 0 4 4480 1 4480 4480 4480 1 4480 4480 8160 0 2040 64 0 "
                              "280 280 280 280 280 280 0 0 0 0 0 64 0 280 280 280 280 280 280 1 1 30000 30000 1 64 "
                              "17920 280 280 280 280 280 280 1 1 21632 8368 2 ";
    std::vector<size_t> expectWorkspaces = {17351296};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 多核+静态quant+dropless  10010
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_07)
{
    auto tilingContextPara = RunNormalCase(
        /*N=*/320, /*H=*/3000, /*K=*/56,
        /*activeNum=*/0, /*C=*/200, /*E=*/32,
        /*dropPadMode=*/0, /*countFlag=*/1, /*tokenFlag=*/false,
        /*quantMode=*/0, /*dqFlag=*/0,
        /*optionalDt=*/ge::DT_FLOAT, /*optionalDtypePosi=*/0);

    uint64_t expectTilingKey = 10010;
    string expectTilingData =
        "64 320 3000 56 200 32 0 1 0 0 0 0 4 4480 1 4480 4480 4480 1 4480 4480 8160 0 2040 64 0 280 280 280 280 280 "
        "280 0 0 0 0 0 64 0 280 280 280 280 280 280 1 1 3000 3000 1 64 17920 280 280 280 280 280 280 1 1 3000 3000 1 ";
    std::vector<size_t> expectWorkspaces = {17351296};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 多核+动态quant+drop  11110
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_08)
{
    auto tilingContextPara = RunNormalCase(
        /*N=*/320, /*H=*/3000, /*K=*/56,
        /*activeNum=*/0, /*C=*/200, /*E=*/32,
        /*dropPadMode=*/1, /*countFlag=*/0, /*tokenFlag=*/true,
        /*quantMode=*/1, /*dqFlag=*/0,
        /*optionalDt=*/ge::DT_FLOAT, /*optionalDtypePosi=*/0);

    uint64_t expectTilingKey = 11110;
    string expectTilingData =
        "64 320 3000 56 200 32 1 0 1 2 0 0 4 4480 1 4480 4480 4480 1 4480 4480 8160 0 2040 64 0 280 280 280 280 280 "
        "280 0 0 0 0 0 64 0 280 280 280 280 280 280 1 1 3000 3000 1 64 17920 280 280 280 280 280 280 1 1 3000 3000 1 ";
    std::vector<size_t> expectWorkspaces = {17351296};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 多核+动态quant+drop + 切H 11110
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_09)
{
    auto tilingContextPara = RunNormalCase(
        /*N=*/320, /*H=*/30000, /*K=*/56,
        /*activeNum=*/0, /*C=*/200, /*E=*/32,
        /*dropPadMode=*/1, /*countFlag=*/0, /*tokenFlag=*/true,
        /*quantMode=*/1, /*dqFlag=*/0,
        /*optionalDt=*/ge::DT_FLOAT, /*optionalDtypePosi=*/0);

    uint64_t expectTilingKey = 11110;
    string expectTilingData = "64 320 30000 56 200 32 1 0 1 2 0 0 4 4480 1 4480 4480 4480 1 4480 4480 8160 0 2040 64 0 "
                              "280 280 280 280 280 280 0 0 0 0 0 64 0 280 280 280 280 280 280 1 1 17156 12844 2 64 "
                              "17920 280 280 280 280 280 280 1 1 12224 5552 3 ";
    std::vector<size_t> expectWorkspaces = {25031296};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 多核+动态quant+dropless  11010
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_10)
{
    auto tilingContextPara = RunNormalCase(
        /*N=*/320, /*H=*/3000, /*K=*/56,
        /*activeNum=*/0, /*C=*/200, /*E=*/32,
        /*dropPadMode=*/0, /*countFlag=*/1, /*tokenFlag=*/false,
        /*quantMode=*/1, /*dqFlag=*/0,
        /*optionalDt=*/ge::DT_FLOAT, /*optionalDtypePosi=*/0);

    uint64_t expectTilingKey = 11010;
    string expectTilingData =
        "64 320 3000 56 200 32 0 1 0 2 0 0 4 4480 1 4480 4480 4480 1 4480 4480 8160 0 2040 64 0 280 280 280 280 280 "
        "280 0 0 0 0 0 64 0 280 280 280 280 280 280 1 1 3000 3000 1 64 17920 280 280 280 280 280 280 1 1 3000 3000 1 ";
    std::vector<size_t> expectWorkspaces = {17351296};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 性能模板+quant
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_11)
{
    auto tilingContextPara = RunNormalCase(
        /*N=*/8, /*H=*/30, /*K=*/6,
        /*activeNum=*/32, /*C=*/0, /*E=*/8,
        /*dropPadMode=*/0, /*countFlag=*/1, /*tokenFlag=*/false,
        /*quantMode=*/0, /*dqFlag=*/0,
        /*optionalDt=*/ge::DT_FLOAT, /*optionalDtypePosi=*/0);

    uint64_t expectTilingKey = 20000;
    string expectTilingData = "64 8 30 6 0 8 0 1 0 0 0 0 1 48 1 48 48 48 1 48 48 8160 0 2040 48 0 1 1 1 1 1 1 0 0 0 0 "
                              "0 48 0 1 1 1 1 1 1 1 1 30 30 1 48 32 1 1 1 1 1 1 1 1 30 30 1 ";
    std::vector<size_t> expectWorkspaces = {16779296};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 性能模板+dynamic quant
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_12)
{
    auto tilingContextPara = RunNormalCase(
        /*N=*/8, /*H=*/30, /*K=*/6,
        /*activeNum=*/32, /*C=*/0, /*E=*/8,
        /*dropPadMode=*/0, /*countFlag=*/1, /*tokenFlag=*/false,
        /*quantMode=*/1, /*dqFlag=*/0,
        /*optionalDt=*/ge::DT_FLOAT, /*optionalDtypePosi=*/0);

    uint64_t expectTilingKey = 21000;
    string expectTilingData = "64 8 30 6 0 8 0 1 0 2 0 0 1 48 1 48 48 48 1 48 48 8160 0 2040 48 0 1 1 1 1 1 1 0 0 0 0 "
                              "0 48 0 1 1 1 1 1 1 1 1 30 30 1 48 32 1 1 1 1 1 1 1 1 30 30 1 ";
    std::vector<size_t> expectWorkspaces = {16779296};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// failed: quant mode != 0 or 1
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_13)
{
    auto tilingContextPara = RunNormalCase(
        /*N=*/320, /*H=*/3000, /*K=*/56,
        /*activeNum=*/0, /*C=*/200, /*E=*/32,
        /*dropPadMode=*/0, /*countFlag=*/1, /*tokenFlag=*/false,
        /*quantMode=*/100, /*dqFlag=*/0,
        /*optionalDt=*/ge::DT_FLOAT, /*optionalDtypePosi=*/0);

    // 对于错误用例，不需要检查 tilingKey 和 tilingData
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// 性能模板+dynamic quant+int4
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_int4_fullload)
{
    auto tilingContextPara = RunNormalCase(
        /*N=*/8, /*H=*/30, /*K=*/6,
        /*activeNum=*/32, /*C=*/0, /*E=*/8,
        /*dropPadMode=*/0, /*countFlag=*/1, /*tokenFlag=*/false,
        /*quantMode=*/1, /*dqFlag=*/1,
        /*optionalDt=*/ge::DT_INT4, /*optionalDtypePosi=*/3);

    uint64_t expectTilingKey = 21000;
    string expectTilingData = "64 8 30 6 0 8 0 1 0 1 0 0 1 48 1 48 48 48 1 48 48 8160 0 2040 48 0 1 1 1 1 1 1 0 0 0 0 "
                              "0 48 0 1 1 1 1 1 1 1 1 30 30 1 48 32 1 1 1 1 1 1 1 1 30 30 1 ";
    std::vector<size_t> expectWorkspaces = {16779296};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 性能模板+dynamic quant+int4
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_int4_fullload_scale_error)
{
    auto tilingContextPara = RunNormalCase(
        /*N=*/8, /*H=*/30, /*K=*/6,
        /*activeNum=*/32, /*C=*/0, /*E=*/8,
        /*dropPadMode=*/0, /*countFlag=*/1, /*tokenFlag=*/false,
        /*quantMode=*/1, /*dqFlag=*/0,
        /*optionalDt=*/ge::DT_INT4, /*optionalDtypePosi=*/3);

    uint64_t expectTilingKey = 21000;
    string expectTilingData = "64 8 30 6 0 8 0 1 0 2 0 0 1 48 1 48 48 48 1 48 48 8160 0 2040 48 0 1 1 1 1 1 1 0 0 0 0 "
                              "0 48 0 1 1 1 1 1 1 1 1 30 30 1 48 32 1 1 1 1 1 1 1 1 30 30 1 ";
    std::vector<size_t> expectWorkspaces = {16779296};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// 性能模板+dynamic quant+int4 pad
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_int4_fullload_pad_error)
{
    auto tilingContextPara = RunNormalCase(
        /*N=*/8, /*H=*/30, /*K=*/6,
        /*activeNum=*/32, /*C=*/0, /*E=*/8,
        /*dropPadMode=*/1, /*countFlag=*/1, /*tokenFlag=*/false,
        /*quantMode=*/1, /*dqFlag=*/1,
        /*optionalDt=*/ge::DT_INT4, /*optionalDtypePosi=*/3);

    uint64_t expectTilingKey = 21000;
    string expectTilingData = "64 8 30 6 0 8 0 1 0 2 0 0 1 48 1 48 48 48 1 48 48 8160 0 2040 48 0 1 1 1 1 1 1 0 0 0 0 "
                              "0 48 0 1 1 1 1 1 1 1 1 30 30 1 48 32 1 1 1 1 1 1 1 1 30 30 1 ";
    std::vector<size_t> expectWorkspaces = {16779296};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// 性能模板+quant+int4 quant
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_int4_fullload_quant_error)
{
    auto tilingContextPara = RunNormalCase(
        /*N=*/8, /*H=*/30, /*K=*/6,
        /*activeNum=*/32, /*C=*/0, /*E=*/8,
        /*dropPadMode=*/0, /*countFlag=*/1, /*tokenFlag=*/false,
        /*quantMode=*/0, /*dqFlag=*/1,
        /*optionalDt=*/ge::DT_INT4, /*optionalDtypePosi=*/3);

    uint64_t expectTilingKey = 21000;
    string expectTilingData = "64 8 30 6 0 8 0 1 0 2 0 0 1 48 1 48 48 48 1 48 48 8160 0 2040 48 0 1 1 1 1 1 1 0 0 0 0 "
                              "0 48 0 1 1 1 1 1 1 1 1 30 30 1 48 32 1 1 1 1 1 1 1 1 30 30 1 ";
    std::vector<size_t> expectWorkspaces = {16779296};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_14)
{
    // 基础参数
    int64_t N = 320;
    int64_t H = 3000;
    int64_t K = 56;
    int64_t activeNum = 1000;
    int64_t C = 0;
    int64_t E = 64;
    int64_t dropPadMode = 0;
    int64_t countFlag = 1;
    bool tokenFlag = false;
    int64_t quantMode = 0;
    int64_t dqFlag = 0;
    ge::DataType optionalDt = ge::DT_FLOAT;
    int64_t optionalDtypePosi = 0;

    // 临时值
    uint64_t expectTilingKey = 0;
    string expectTilingData = "error_tiling_data";
    std::vector<size_t> expectWorkspaces = {1024};

    // 场景1: scale shape dim != 1 when quantMode = 0
    {
        auto tilingContextPara = RunNormalCase(N, H, K, activeNum, C, E, dropPadMode, countFlag, tokenFlag, quantMode,
                                               dqFlag, optionalDt, optionalDtypePosi);

        // 设置错误的 scale 形状
        tilingContextPara.inputTensorDesc_[2].shape_ = gert::StorageShape({1, 2}, {1, 2});

        ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
    }

    // 场景2: scale shape first dim != 1 when quantMode = 0
    {
        auto tilingContextPara = RunNormalCase(N, H, K, activeNum, C, E, dropPadMode, countFlag, tokenFlag, quantMode,
                                               dqFlag, optionalDt, optionalDtypePosi);

        // 设置错误的 scale 形状
        tilingContextPara.inputTensorDesc_[2].shape_ = gert::StorageShape({2}, {2});

        ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
    }

    // 场景3: offset shape dim != 1 when quantMode = 0
    {
        auto tilingContextPara = RunNormalCase(N, H, K, activeNum, C, E, dropPadMode, countFlag, tokenFlag, quantMode,
                                               dqFlag, optionalDt, optionalDtypePosi);

        // 设置正确的 scale 形状
        tilingContextPara.inputTensorDesc_[2].shape_ = gert::StorageShape({1}, {1});
        // 设置错误的 offset 形状
        tilingContextPara.inputTensorDesc_[3].shape_ = gert::StorageShape({1, 100}, {1, 100});

        ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
    }

    // 场景4: offset shape first dim != 1 when quantMode = 0
    {
        auto tilingContextPara = RunNormalCase(N, H, K, activeNum, C, E, dropPadMode, countFlag, tokenFlag, quantMode,
                                               dqFlag, optionalDt, optionalDtypePosi);

        // 设置正确的 scale 形状
        tilingContextPara.inputTensorDesc_[2].shape_ = gert::StorageShape({1}, {1});
        // 设置错误的 offset 形状
        tilingContextPara.inputTensorDesc_[3].shape_ = gert::StorageShape({2}, {2});

        ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
    }

    // 场景5: scale shape dim != 2 when quantMode = 1
    {
        quantMode = 1; // 切换到动态量化模式
        auto tilingContextPara = RunNormalCase(N, H, K, activeNum, C, E, dropPadMode, countFlag, tokenFlag, quantMode,
                                               dqFlag, optionalDt, optionalDtypePosi);

        // 设置错误的 scale 形状
        tilingContextPara.inputTensorDesc_[2].shape_ = gert::StorageShape({2}, {2});

        ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
        quantMode = 0; // 恢复默认
    }

    // 场景6: dynamic quant scale shape is not (E, H) or (1, H) when quantMode = 1
    {
        quantMode = 1; // 切换到动态量化模式
        auto tilingContextPara = RunNormalCase(N, H, K, activeNum, C, E, dropPadMode, countFlag, tokenFlag, quantMode,
                                               dqFlag, optionalDt, optionalDtypePosi);

        // 设置错误的 scale 形状
        tilingContextPara.inputTensorDesc_[2].shape_ = gert::StorageShape({100, 3000}, {100, 3000});

        ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
        quantMode = 0; // 恢复默认
    }

    // 场景7: dynamic quant scale shape is not (E, H) or (1, H) when quantMode = 1
    {
        quantMode = 1; // 切换到动态量化模式
        auto tilingContextPara = RunNormalCase(N, H, K, activeNum, C, E, dropPadMode, countFlag, tokenFlag, quantMode,
                                               dqFlag, optionalDt, optionalDtypePosi);

        // 设置错误的 scale 形状
        tilingContextPara.inputTensorDesc_[2].shape_ = gert::StorageShape({1, 3456}, {1, 3456});

        ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
        quantMode = 0; // 恢复默认
    }

    // 场景8: dynamic quant scale shape dim != 1
    {
        quantMode = 1; // 切换到动态量化模式
        auto tilingContextPara = RunNormalCase(N, H, K, activeNum, C, E, dropPadMode, countFlag, tokenFlag, quantMode,
                                               dqFlag, optionalDt, optionalDtypePosi);

        // 设置正确的 scale 形状
        tilingContextPara.inputTensorDesc_[2].shape_ = gert::StorageShape({1, 3000}, {1, 3000});
        // 设置错误的动态量化输出形状
        tilingContextPara.outputTensorDesc_[4].shape_ = gert::StorageShape({2, 2}, {2, 2});

        ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
        quantMode = 0; // 恢复默认
    }

    // 场景9: dynamic quant scale shape is not (min(n*k, activeNum),) when drop pad mode = 0
    {
        quantMode = 1; // 切换到动态量化模式
        auto tilingContextPara = RunNormalCase(N, H, K, activeNum, C, E, dropPadMode, countFlag, tokenFlag, quantMode,
                                               dqFlag, optionalDt, optionalDtypePosi);

        // 设置错误的动态量化输出形状
        tilingContextPara.outputTensorDesc_[4].shape_ = gert::StorageShape({10000}, {10000});

        ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
        quantMode = 0; // 恢复默认
    }

    // 场景10: dynamic quant scale shape is not (E * C,) when drop pad mode = 1
    {
        quantMode = 1;   // 切换到动态量化模式
        dropPadMode = 1; // 切换到dropless模式
        auto tilingContextPara = RunNormalCase(N, H, K, activeNum, C, E, dropPadMode, countFlag, tokenFlag, quantMode,
                                               dqFlag, optionalDt, optionalDtypePosi);

        // 设置错误的动态量化输出形状
        tilingContextPara.outputTensorDesc_[4].shape_ = gert::StorageShape({10000}, {10000});

        ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
        quantMode = 0;   // 恢复默认
        dropPadMode = 0; // 恢复默认
    }
}

// scale dtype = int32
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_15)
{
    auto tilingContextPara = RunNormalCase(
        /*N=*/320, /*H=*/3000, /*K=*/56,
        /*activeNum=*/0, /*C=*/200, /*E=*/32,
        /*dropPadMode=*/0, /*countFlag=*/1, /*tokenFlag=*/false,
        /*quantMode=*/0, /*dqFlag=*/0,
        /*optionalDt=*/ge::DT_INT32, /*optionalDtypePosi=*/0);

    // 对于错误用例，不需要检查 tilingKey 和 tilingData
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// scale dtype = int32 & quantMode = 1
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_16)
{
    auto tilingContextPara = RunNormalCase(
        /*N=*/320, /*H=*/3000, /*K=*/56,
        /*activeNum=*/0, /*C=*/200, /*E=*/32,
        /*dropPadMode=*/0, /*countFlag=*/1, /*tokenFlag=*/false,
        /*quantMode=*/1, /*dqFlag=*/0,
        /*optionalDt=*/ge::DT_INT32, /*optionalDtypePosi=*/0);

    // 对于错误用例，不需要检查 tilingKey 和 tilingData
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// offset dtype = int32
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_17)
{
    auto tilingContextPara = RunNormalCase(
        /*N=*/320, /*H=*/3000, /*K=*/56,
        /*activeNum=*/0, /*C=*/200, /*E=*/32,
        /*dropPadMode=*/0, /*countFlag=*/1, /*tokenFlag=*/false,
        /*quantMode=*/0, /*dqFlag=*/0,
        /*optionalDt=*/ge::DT_INT32, /*optionalDtypePosi=*/1);

    // 对于错误用例，不需要检查 tilingKey 和 tilingData
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// Large totalLength + large cols: triggers SrcToDst else, VMSMiddle else,
// VBS lastCoreElement>alineCeil, CapacityCompute else (cols >= MAX_COLS_ONE_LOOP)
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_19)
{
    auto tilingContextPara = RunNormalCase(
        /*N=*/3000, /*H=*/130000, /*K=*/50,
        /*activeNum=*/0, /*C=*/0, /*E=*/8,
        /*dropPadMode=*/0, /*countFlag=*/1, /*tokenFlag=*/false,
        /*quantMode=*/0, /*dqFlag=*/0,
        /*optionalDt=*/ge::DT_FLOAT, /*optionalDtypePosi=*/0);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, (uint64_t)-1, "", {});
}

// Large totalLength + moderate cols: triggers CapacityCompute else (cols < MAX_COLS_ONE_LOOP path)
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_20)
{
    auto tilingContextPara = RunNormalCase(
        /*N=*/40000, /*H=*/10000, /*K=*/50,
        /*activeNum=*/0, /*C=*/0, /*E=*/8,
        /*dropPadMode=*/0, /*countFlag=*/1, /*tokenFlag=*/false,
        /*quantMode=*/0, /*dqFlag=*/0,
        /*optionalDt=*/ge::DT_FLOAT, /*optionalDtypePosi=*/0);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, (uint64_t)-1, "", {});
}

// regBase SOC (NpuArch 3510) path in GetPlatformInfo
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_21_regbase)
{
    auto tilingContextPara = RunNormalCase(
        /*N=*/8, /*H=*/30, /*K=*/6,
        /*activeNum=*/0, /*C=*/6, /*E=*/8,
        /*dropPadMode=*/1, /*countFlag=*/0, /*tokenFlag=*/true,
        /*quantMode=*/0, /*dqFlag=*/0,
        /*optionalDt=*/ge::DT_FLOAT, /*optionalDtypePosi=*/0);
    // Use Ascend950 so UT platform JSON parses CORE_NUM; NpuArch still maps to 3510 (regbase).
    tilingContextPara.socVersion_ = "Ascend950";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, (uint64_t)-1, "", {});
}

// Large params, quantMode=1, dropPadMode=1: triggers SrcToDstCapacityCompute else (cols<6144)
// and GatherDynamicQuant else (cols<6144)
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_22_dynquant_large)
{
    auto tilingContextPara = RunNormalCase(
        /*N=*/20000, /*H=*/5000, /*K=*/40,
        /*activeNum=*/0, /*C=*/6, /*E=*/8,
        /*dropPadMode=*/1, /*countFlag=*/0, /*tokenFlag=*/true,
        /*quantMode=*/1, /*dqFlag=*/0,
        /*optionalDt=*/ge::DT_FLOAT, /*optionalDtypePosi=*/0);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, (uint64_t)-1, "", {});
}

// Large params, quantMode=0, dropPadMode=1: triggers GatherQuant else (cols<MAX_COLS_ONE_LOOP)
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_23_quant_large)
{
    auto tilingContextPara = RunNormalCase(
        /*N=*/20000, /*H=*/4000, /*K=*/40,
        /*activeNum=*/0, /*C=*/6, /*E=*/8,
        /*dropPadMode=*/1, /*countFlag=*/0, /*tokenFlag=*/true,
        /*quantMode=*/0, /*dqFlag=*/0,
        /*optionalDt=*/ge::DT_FLOAT, /*optionalDtypePosi=*/0);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, (uint64_t)-1, "", {});
}

// quantMode=1 with no scale input: triggers smoothType=SMOOTH_NONE path
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_24_no_scale)
{
    int64_t N = 8, H = 30, K = 6, C = 6, E = 8;
    std::vector<gert::TilingContextPara::TensorDescription> inputs;
    inputs.emplace_back(gert::StorageShape({N, H}, {N, H}), ge::DT_FLOAT16, ge::FORMAT_ND);
    inputs.emplace_back(gert::StorageShape({N, K}, {N, K}), ge::DT_INT32, ge::FORMAT_ND);
    // No scale/offset inputs - use empty shapes to make GetOptionalInputShape return null
    inputs.emplace_back(gert::StorageShape({}, {}), ge::DT_FLOAT, ge::FORMAT_ND);
    inputs.emplace_back(gert::StorageShape({}, {}), ge::DT_FLOAT, ge::FORMAT_ND);

    std::vector<gert::TilingContextPara::TensorDescription> outputs;
    outputs.emplace_back(gert::StorageShape({E, C, H}, {E, C, H}), ge::DT_INT8, ge::FORMAT_ND);
    outputs.emplace_back(gert::StorageShape({N * K}, {N * K}), ge::DT_INT32, ge::FORMAT_ND);
    outputs.emplace_back(gert::StorageShape({E}, {E}), ge::DT_INT32, ge::FORMAT_ND);
    outputs.emplace_back(gert::StorageShape({E}, {E}), ge::DT_INT32, ge::FORMAT_ND);
    outputs.emplace_back(gert::StorageShape({E * C}, {E * C}), ge::DT_FLOAT, ge::FORMAT_ND);

    MoeInitRoutingQuantV2CompileInfo compileInfo;
    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(C)},
        {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(E)},
        {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}};

    auto tilingContextPara = gert::TilingContextPara("MoeInitRoutingQuantV2", inputs, outputs, attrs, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, (uint64_t)-1, "", {});
}

// dynamic quant scale dtype = int32
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_18)
{
    auto tilingContextPara = RunNormalCase(
        /*N=*/320, /*H=*/3000, /*K=*/56,
        /*activeNum=*/0, /*C=*/200, /*E=*/32,
        /*dropPadMode=*/0, /*countFlag=*/1, /*tokenFlag=*/false,
        /*quantMode=*/1, /*dqFlag=*/0,
        /*optionalDt=*/ge::DT_INT32, /*optionalDtypePosi=*/2);

    // 对于错误用例，不需要检查 tilingKey 和 tilingData
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}
