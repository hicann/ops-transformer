/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <cstdlib>
#include <vector>
#include "../matmul_all_reduce_host_ut_param.h"
#include "mc2_tiling_case_executor.h"

namespace MatmulAllReduceUT {

namespace {

constexpr uint64_t SKIP_TILING_KEY_VALIDATION = UINT64_MAX;

gert::TilingContextPara::TensorDescription MakeTensorDesc(const std::string& shapeStr, ge::DataType dtype)
{
    if (shapeStr.empty()) {
        return TD_DEFAULT;
    }
    return gert::TilingContextPara::TensorDescription(GetStorageShape(shapeStr), dtype, ge::FORMAT_ND);
}

std::vector<gert::TilingContextPara::OpAttr> MakeDefaultAttrs(const std::string& reduceOp, bool isTransB = false)
{
    return {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>(reduceOp)},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(isTransB)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")}};
}

void RunMatmulAllReduceTilingCase(const gert::TilingContextPara& tilingContextPara, uint64_t ranksize,
                                  ge::graphStatus expectResult = ge::GRAPH_SUCCESS)
{
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", ranksize}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, expectResult, SKIP_TILING_KEY_VALIDATION, "", {},
                       MC2_TILING_DATA_RESERVED_LEN, false);
}

struct MatmulAllReduceCompileInfo {
} g_extraTestCompileInfo;

gert::TilingContextPara BuildFloat16TilingContext(const std::string& reduceOp, uint64_t coreNum, uint64_t ubsize,
                                                  uint64_t ranksize)
{
    (void)ranksize;
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {MakeTensorDesc("8192 1536", ge::DT_FLOAT16),
                                                                      MakeTensorDesc("1536 12288", ge::DT_FLOAT16),
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT};
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {MakeTensorDesc("8192 12288", ge::DT_FLOAT16)};
    return gert::TilingContextPara("MatmulAllReduce", inputs, outputs, MakeDefaultAttrs(reduceOp),
                                   &g_extraTestCompileInfo, "Ascend910B", coreNum, ubsize);
}

} // namespace

class MatmulAllReduceArch22TilingTest : public testing::TestWithParam<MatmulAllReduceTilingUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatmulAllReduceArch22TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MatmulAllReduceArch22TilingTest TearDown" << std::endl;
    }
};

TEST_P(MatmulAllReduceArch22TilingTest, param)
{
    auto param = GetParam();
    struct MatmulAllReduceCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulAllReduce",
        {param.x1, param.x2, param.bias, param.x3, param.antiquant_scale, param.antiquant_offset, param.dequant_scale,
         param.pertoken_scale, param.comm_quant_scale_1, param.comm_quant_scale_2},
        {param.y},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.group)},
         {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.reduce_op)},
         {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_trans_a)},
         {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_trans_b)},
         {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.comm_turn)},
         {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.antiquant_group_size)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.group_size)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.y_dtype)},
         {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.comm_quant_mode)},
         {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.comm_mode)}},
        param.inputInstance, param.outputInstance, &compileInfo, param.soc, param.coreNum, param.ubsize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", param.ranksize}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, param.expectResult, param.expectTilingKey,
                       param.expectTilingDataHash, {}, MC2_TILING_DATA_RESERVED_LEN, true);
}

INSTANTIATE_TEST_SUITE_P(
    MatmulAllReduce, MatmulAllReduceArch22TilingTest,
    testing::ValuesIn(GetCasesFromCsv<MatmulAllReduceTilingUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<MatmulAllReduceTilingUtParam>);

class MatmulAllReduceArch22TilingExtraTest : public testing::Test {};

TEST_F(MatmulAllReduceArch22TilingExtraTest, DebugModeAicpuBufferType)
{
    const char* envValue = getenv("ASCEND_MC2_DEBUG_MODE");
    std::string originalEnv = envValue != nullptr ? envValue : "";
    setenv("ASCEND_MC2_DEBUG_MODE", "4", 1);
    auto tilingContextPara = BuildFloat16TilingContext("sum", 24, 196608, 8);
    RunMatmulAllReduceTilingCase(tilingContextPara, 8);
    if (originalEnv.empty()) {
        unsetenv("ASCEND_MC2_DEBUG_MODE");
    } else {
        setenv("ASCEND_MC2_DEBUG_MODE", originalEnv.c_str(), 1);
    }
}

TEST_F(MatmulAllReduceArch22TilingExtraTest, HcclBuffsizeValid)
{
    const char* envValue = getenv("HCCL_BUFFSIZE");
    std::string originalEnv = envValue != nullptr ? envValue : "";
    setenv("HCCL_BUFFSIZE", "50", 1);
    auto tilingContextPara = BuildFloat16TilingContext("sum", 24, 196608, 8);
    RunMatmulAllReduceTilingCase(tilingContextPara, 8);
    if (originalEnv.empty()) {
        unsetenv("HCCL_BUFFSIZE");
    } else {
        setenv("HCCL_BUFFSIZE", originalEnv.c_str(), 1);
    }
}

TEST_F(MatmulAllReduceArch22TilingExtraTest, HcclBuffsizeInvalid)
{
    const char* envValue = getenv("HCCL_BUFFSIZE");
    std::string originalEnv = envValue != nullptr ? envValue : "";
    setenv("HCCL_BUFFSIZE", "invalid", 1);
    auto tilingContextPara = BuildFloat16TilingContext("sum", 24, 196608, 8);
    RunMatmulAllReduceTilingCase(tilingContextPara, 8);
    if (originalEnv.empty()) {
        unsetenv("HCCL_BUFFSIZE");
    } else {
        setenv("HCCL_BUFFSIZE", originalEnv.c_str(), 1);
    }
}

TEST_F(MatmulAllReduceArch22TilingExtraTest, L2CacheSmallNTile)
{
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {MakeTensorDesc("128 512", ge::DT_FLOAT16),
                                                                      MakeTensorDesc("512 64", ge::DT_FLOAT16),
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT};
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {MakeTensorDesc("128 64", ge::DT_FLOAT16)};
    gert::TilingContextPara tilingContextPara("MatmulAllReduce", inputs, outputs, MakeDefaultAttrs("sum"),
                                              &g_extraTestCompileInfo, "Ascend910B", 32, 262144);
    RunMatmulAllReduceTilingCase(tilingContextPara, 8);
}

TEST_F(MatmulAllReduceArch22TilingExtraTest, A16W8WeightQuantTailM)
{
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {MakeTensorDesc("1025 1024", ge::DT_FLOAT16),
                                                                      MakeTensorDesc("1024 4096", ge::DT_INT8),
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      MakeTensorDesc("4096", ge::DT_FLOAT16),
                                                                      MakeTensorDesc("4096", ge::DT_FLOAT16),
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT};
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {MakeTensorDesc("1025 4096", ge::DT_FLOAT16)};
    gert::TilingContextPara tilingContextPara("MatmulAllReduce", inputs, outputs, MakeDefaultAttrs("sum", true),
                                              &g_extraTestCompileInfo, "Ascend910B", 24, 196608);
    RunMatmulAllReduceTilingCase(tilingContextPara, 8);
}

TEST_F(MatmulAllReduceArch22TilingExtraTest, A16W8WeightQuantLargeTailM)
{
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {MakeTensorDesc("16384 1536", ge::DT_FLOAT16),
                                                                      MakeTensorDesc("1536 8192", ge::DT_INT8),
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      MakeTensorDesc("8192", ge::DT_FLOAT16),
                                                                      MakeTensorDesc("8192", ge::DT_FLOAT16),
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT};
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {MakeTensorDesc("16384 8192", ge::DT_FLOAT16)};
    gert::TilingContextPara tilingContextPara("MatmulAllReduce", inputs, outputs, MakeDefaultAttrs("sum", true),
                                              &g_extraTestCompileInfo, "Ascend910B", 24, 196608);
    RunMatmulAllReduceTilingCase(tilingContextPara, 8);
}

TEST_F(MatmulAllReduceArch22TilingExtraTest, CommQuantFrontBalancingSmallK)
{
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {MakeTensorDesc("1800 512", ge::DT_INT8),
                                                                      MakeTensorDesc("512 8192", ge::DT_INT8),
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      MakeTensorDesc("8192", ge::DT_UINT64),
                                                                      TD_DEFAULT,
                                                                      MakeTensorDesc("8192", ge::DT_FLOAT16),
                                                                      MakeTensorDesc("8192", ge::DT_FLOAT16)};
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {MakeTensorDesc("1800 8192", ge::DT_FLOAT16)};
    gert::TilingContextPara tilingContextPara("MatmulAllReduce", inputs, outputs, MakeDefaultAttrs("sum"),
                                              &g_extraTestCompileInfo, "Ascend910B", 24, 196608);
    RunMatmulAllReduceTilingCase(tilingContextPara, 8);
}

TEST_F(MatmulAllReduceArch22TilingExtraTest, CommQuantUniformCutMidM)
{
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {MakeTensorDesc("1792 640", ge::DT_INT8),
                                                                      MakeTensorDesc("640 4096", ge::DT_INT8),
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      MakeTensorDesc("4096", ge::DT_UINT64),
                                                                      TD_DEFAULT,
                                                                      MakeTensorDesc("4096", ge::DT_FLOAT16),
                                                                      MakeTensorDesc("4096", ge::DT_FLOAT16)};
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {MakeTensorDesc("1792 4096", ge::DT_FLOAT16)};
    gert::TilingContextPara tilingContextPara("MatmulAllReduce", inputs, outputs, MakeDefaultAttrs("sum"),
                                              &g_extraTestCompileInfo, "Ascend910B", 24, 196608);
    RunMatmulAllReduceTilingCase(tilingContextPara, 8);
}

TEST_F(MatmulAllReduceArch22TilingExtraTest, Float16SmallMLargeNK)
{
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {MakeTensorDesc("512 8192", ge::DT_FLOAT16),
                                                                      MakeTensorDesc("8192 8192", ge::DT_FLOAT16),
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT};
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {MakeTensorDesc("512 8192", ge::DT_FLOAT16)};
    gert::TilingContextPara tilingContextPara("MatmulAllReduce", inputs, outputs, MakeDefaultAttrs("sum"),
                                              &g_extraTestCompileInfo, "Ascend910B", 24, 196608);
    RunMatmulAllReduceTilingCase(tilingContextPara, 8);
}

TEST_F(MatmulAllReduceArch22TilingExtraTest, CommQuantSmallShortCheckOddM)
{
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {MakeTensorDesc("3333 512", ge::DT_INT8),
                                                                      MakeTensorDesc("512 8192", ge::DT_INT8),
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      MakeTensorDesc("8192", ge::DT_UINT64),
                                                                      TD_DEFAULT,
                                                                      MakeTensorDesc("8192", ge::DT_FLOAT16),
                                                                      MakeTensorDesc("8192", ge::DT_FLOAT16)};
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {MakeTensorDesc("3333 8192", ge::DT_FLOAT16)};
    gert::TilingContextPara tilingContextPara("MatmulAllReduce", inputs, outputs, MakeDefaultAttrs("sum"),
                                              &g_extraTestCompileInfo, "Ascend910B", 24, 196608);
    RunMatmulAllReduceTilingCase(tilingContextPara, 8);
}

TEST_F(MatmulAllReduceArch22TilingExtraTest, Int8QuantTailMNoCommQuant)
{
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {MakeTensorDesc("1025 512", ge::DT_INT8),
                                                                      MakeTensorDesc("512 8192", ge::DT_INT8),
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      MakeTensorDesc("8192", ge::DT_UINT64),
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT};
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {MakeTensorDesc("1025 8192", ge::DT_FLOAT16)};
    gert::TilingContextPara tilingContextPara("MatmulAllReduce", inputs, outputs, MakeDefaultAttrs("sum"),
                                              &g_extraTestCompileInfo, "Ascend910B", 24, 196608);
    RunMatmulAllReduceTilingCase(tilingContextPara, 8);
}

} // namespace MatmulAllReduceUT
