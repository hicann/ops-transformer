/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <cstdlib>
#include "../matmul_all_reduce_host_ut_param.h"
#include "mc2_tiling_case_executor.h"
#include "op_tiling_parse_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace MatmulAllReduceUT {

namespace {

constexpr uint64_t SKIP_TILING_KEY_VALIDATION = UINT64_MAX;

struct MatmulAllReduceCompileInfo {
} g_arch35ExtraTestCompileInfo;

gert::TilingContextPara::TensorDescription MakeArch35TensorDesc(const std::string &shapeStr, ge::DataType dtype)
{
    if (shapeStr.empty()) {
        return TD_DEFAULT;
    }
    return gert::TilingContextPara::TensorDescription(GetStorageShape(shapeStr), dtype, ge::FORMAT_ND);
}

std::vector<gert::TilingContextPara::OpAttr> MakeArch35DefaultAttrs(bool isTransB = false, int64_t groupSize = 0)
{
    return {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(isTransB)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupSize)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")}};
}

void RunArch35MatmulAllReduceTilingCase(const gert::TilingContextPara &tilingContextPara, uint64_t ranksize)
{
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", ranksize}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, SKIP_TILING_KEY_VALIDATION, "", {},
                       0, false);
}

} // namespace

class MatmulAllReduceArch35TilingTest : public testing::TestWithParam<MatmulAllReduceTilingUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatmulAllReduceArch35TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MatmulAllReduceArch35TilingTest TearDown" << std::endl;
    }
};

TEST_P(MatmulAllReduceArch35TilingTest, param)
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
                       param.expectTilingDataHash, {}, 0, true);
}

INSTANTIATE_TEST_SUITE_P(
    MatmulAllReduce, MatmulAllReduceArch35TilingTest,
    testing::ValuesIn(GetCasesFromCsv<MatmulAllReduceTilingUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<MatmulAllReduceTilingUtParam>);

class MatmulAllReduceArch35TilingExtraTest : public testing::Test {};

constexpr int64_t MXFP_GROUP_SIZE_ATTR = 4295032864;

TEST_F(MatmulAllReduceArch35TilingExtraTest, Mxfp8QuantTiling)
{
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {
        MakeArch35TensorDesc("256 1024", ge::DT_FLOAT8_E4M3FN),
        MakeArch35TensorDesc("1024 8192", ge::DT_FLOAT8_E4M3FN),
        TD_DEFAULT,
        TD_DEFAULT,
        TD_DEFAULT,
        TD_DEFAULT,
        MakeArch35TensorDesc("8192 16 2", ge::DT_FLOAT8_E8M0),
        MakeArch35TensorDesc("256 16 2", ge::DT_FLOAT8_E8M0),
        TD_DEFAULT,
        TD_DEFAULT};
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        MakeArch35TensorDesc("256 8192", ge::DT_FLOAT16)};
    gert::TilingContextPara tilingContextPara("MatmulAllReduce", inputs, outputs,
                                              MakeArch35DefaultAttrs(true, MXFP_GROUP_SIZE_ATTR),
                                              &g_arch35ExtraTestCompileInfo, "3510", 32, 262144);
    RunArch35MatmulAllReduceTilingCase(tilingContextPara, 8);
}

TEST_F(MatmulAllReduceArch35TilingExtraTest, AicpuCommModeFloat16)
{
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {MakeArch35TensorDesc("256 512", ge::DT_FLOAT16),
                                                                      MakeArch35TensorDesc("512 8192", ge::DT_FLOAT16),
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT};
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        MakeArch35TensorDesc("256 8192", ge::DT_FLOAT16)};
    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
        {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
        {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
        {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")}};
    gert::TilingContextPara tilingContextPara("MatmulAllReduce", inputs, outputs, attrs, &g_arch35ExtraTestCompileInfo,
                                              "3510", 32, 262144);
    RunArch35MatmulAllReduceTilingCase(tilingContextPara, 8);
}

TEST_F(MatmulAllReduceArch35TilingExtraTest, AicpuCommModeRank2AllReduce)
{
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {MakeArch35TensorDesc("256 512", ge::DT_FLOAT16),
                                                                      MakeArch35TensorDesc("512 8192", ge::DT_FLOAT16),
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT};
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        MakeArch35TensorDesc("256 8192", ge::DT_FLOAT16)};
    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
        {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
        {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
        {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")}};
    gert::TilingContextPara tilingContextPara("MatmulAllReduce", inputs, outputs, attrs, &g_arch35ExtraTestCompileInfo,
                                              "3510", 32, 262144);
    RunArch35MatmulAllReduceTilingCase(tilingContextPara, 2);
}

TEST_F(MatmulAllReduceArch35TilingExtraTest, Mxfp4BatchTiling)
{
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {
        MakeArch35TensorDesc("2 256 1024", ge::DT_FLOAT4_E2M1),
        MakeArch35TensorDesc("1024 8192", ge::DT_FLOAT4_E2M1),
        TD_DEFAULT,
        TD_DEFAULT,
        TD_DEFAULT,
        TD_DEFAULT,
        MakeArch35TensorDesc("8192 16 2", ge::DT_FLOAT8_E8M0),
        MakeArch35TensorDesc("256 16 2", ge::DT_FLOAT8_E8M0),
        TD_DEFAULT,
        TD_DEFAULT};
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        MakeArch35TensorDesc("2 256 8192", ge::DT_FLOAT16)};
    gert::TilingContextPara tilingContextPara("MatmulAllReduce", inputs, outputs,
                                              MakeArch35DefaultAttrs(true, MXFP_GROUP_SIZE_ATTR),
                                              &g_arch35ExtraTestCompileInfo, "3510", 32, 262144);
    RunArch35MatmulAllReduceTilingCase(tilingContextPara, 8);
}

TEST_F(MatmulAllReduceArch35TilingExtraTest, Fp8Hif8PerblockTiling)
{
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {
        MakeArch35TensorDesc("256 1024", ge::DT_HIFLOAT8),
        MakeArch35TensorDesc("1024 8192", ge::DT_HIFLOAT8),
        TD_DEFAULT,
        TD_DEFAULT,
        TD_DEFAULT,
        TD_DEFAULT,
        MakeArch35TensorDesc("8 64", ge::DT_FLOAT),
        MakeArch35TensorDesc("2 8", ge::DT_FLOAT),
        TD_DEFAULT,
        TD_DEFAULT};
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        MakeArch35TensorDesc("256 8192", ge::DT_FLOAT16)};
    gert::TilingContextPara tilingContextPara("MatmulAllReduce", inputs, outputs, MakeArch35DefaultAttrs(true),
                                              &g_arch35ExtraTestCompileInfo, "3510", 32, 262144);
    RunArch35MatmulAllReduceTilingCase(tilingContextPara, 8);
}

TEST_F(MatmulAllReduceArch35TilingExtraTest, A16W8PerGroupTailM)
{
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {MakeArch35TensorDesc("1025 1536", ge::DT_FLOAT16),
                                                                      MakeArch35TensorDesc("1536 8192", ge::DT_INT8),
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      MakeArch35TensorDesc("8192", ge::DT_FLOAT16),
                                                                      MakeArch35TensorDesc("8192", ge::DT_FLOAT16),
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT};
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        MakeArch35TensorDesc("1025 8192", ge::DT_FLOAT16)};
    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
        {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
        {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
        {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
        {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")}};
    gert::TilingContextPara tilingContextPara("MatmulAllReduce", inputs, outputs, attrs, &g_arch35ExtraTestCompileInfo,
                                              "3510", 32, 262144);
    RunArch35MatmulAllReduceTilingCase(tilingContextPara, 8);
}

TEST_F(MatmulAllReduceArch35TilingExtraTest, A16W8PerGroupTailRank2Aicpu)
{
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {MakeArch35TensorDesc("1025 1024", ge::DT_FLOAT16),
                                                                      MakeArch35TensorDesc("1024 4096", ge::DT_INT8),
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      MakeArch35TensorDesc("4096", ge::DT_FLOAT16),
                                                                      MakeArch35TensorDesc("4096", ge::DT_FLOAT16),
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT};
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        MakeArch35TensorDesc("1025 4096", ge::DT_FLOAT16)};
    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
        {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
        {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
        {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
        {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")}};
    gert::TilingContextPara tilingContextPara("MatmulAllReduce", inputs, outputs, attrs, &g_arch35ExtraTestCompileInfo,
                                              "3510", 32, 262144);
    RunArch35MatmulAllReduceTilingCase(tilingContextPara, 2);
}

TEST_F(MatmulAllReduceArch35TilingExtraTest, A16W8PerChannelTail4097)
{
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {MakeArch35TensorDesc("4097 1536", ge::DT_FLOAT16),
                                                                      MakeArch35TensorDesc("1536 8192", ge::DT_INT8),
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      MakeArch35TensorDesc("8192", ge::DT_FLOAT16),
                                                                      MakeArch35TensorDesc("8192", ge::DT_FLOAT16),
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT};
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        MakeArch35TensorDesc("4097 8192", ge::DT_FLOAT16)};
    gert::TilingContextPara tilingContextPara("MatmulAllReduce", inputs, outputs, MakeArch35DefaultAttrs(true),
                                              &g_arch35ExtraTestCompileInfo, "3510", 32, 262144);
    RunArch35MatmulAllReduceTilingCase(tilingContextPara, 8);
}

TEST_F(MatmulAllReduceArch35TilingExtraTest, Int8QuantTail4097)
{
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {MakeArch35TensorDesc("4097 512", ge::DT_INT8),
                                                                      MakeArch35TensorDesc("512 8192", ge::DT_INT8),
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      MakeArch35TensorDesc("8192", ge::DT_UINT64),
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT,
                                                                      TD_DEFAULT};
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        MakeArch35TensorDesc("4097 8192", ge::DT_FLOAT16)};
    gert::TilingContextPara tilingContextPara("MatmulAllReduce", inputs, outputs, MakeArch35DefaultAttrs(),
                                              &g_arch35ExtraTestCompileInfo, "3510", 32, 262144);
    RunArch35MatmulAllReduceTilingCase(tilingContextPara, 8);
}

TEST_F(MatmulAllReduceArch35TilingExtraTest, Float16Rank4FitBalanceLargeM)
{
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {
        MakeArch35TensorDesc("12290 15360", ge::DT_FLOAT16),
        MakeArch35TensorDesc("15360 12288", ge::DT_FLOAT16),
        TD_DEFAULT,
        TD_DEFAULT,
        TD_DEFAULT,
        TD_DEFAULT,
        TD_DEFAULT,
        TD_DEFAULT,
        TD_DEFAULT,
        TD_DEFAULT};
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        MakeArch35TensorDesc("12290 12288", ge::DT_FLOAT16)};
    gert::TilingContextPara tilingContextPara("MatmulAllReduce", inputs, outputs, MakeArch35DefaultAttrs(),
                                              &g_arch35ExtraTestCompileInfo, "3510", 32, 262144);
    RunArch35MatmulAllReduceTilingCase(tilingContextPara, 4);
}

TEST_F(MatmulAllReduceArch35TilingExtraTest, TilingParseSuccess)
{
    gert::OpTilingParseContextBuilder builder;
    auto holder = builder.Build();
    auto parseContext = holder.GetContext();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MatmulAllReduce");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling_parse, nullptr);

    auto ret = opImpl->tiling_parse(reinterpret_cast<gert::KernelContext *>(parseContext));
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

} // namespace MatmulAllReduceUT
