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
#include "../matmul_all_reduce_host_ut_param.h"
#include "mc2_tiling_case_executor.h"

namespace MatmulAllReduceUT {

class MatmulAllReduceArch20TilingTest : public testing::TestWithParam<MatmulAllReduceTilingUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatmulAllReduceArch20TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MatmulAllReduceArch20TilingTest TearDown" << std::endl;
    }
};

TEST_P(MatmulAllReduceArch20TilingTest, param)
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
    MatmulAllReduce, MatmulAllReduceArch20TilingTest,
    testing::ValuesIn(GetCasesFromCsv<MatmulAllReduceTilingUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<MatmulAllReduceTilingUtParam>);

class MatmulAllReduceArch20TilingExtraTest : public testing::Test {};

TEST_F(MatmulAllReduceArch20TilingExtraTest, Float16TailM1025)
{
    struct MatmulAllReduceCompileInfo {
    } compileInfo;
    gert::TilingContextPara::TensorDescription x1 = {GetStorageShape("1025 2048"), ge::DT_FLOAT16, ge::FORMAT_ND};
    gert::TilingContextPara::TensorDescription x2 = {GetStorageShape("2048 4096"), ge::DT_FLOAT16, ge::FORMAT_ND};
    gert::TilingContextPara::TensorDescription y = {GetStorageShape("1025 4096"), ge::DT_FLOAT16, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara(
        "MatmulAllReduce",
        {x1, x2, TD_DEFAULT, TD_DEFAULT, TD_DEFAULT, TD_DEFAULT, TD_DEFAULT, TD_DEFAULT, TD_DEFAULT, TD_DEFAULT}, {y},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
         {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
         {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        {1, 1, 0, 0, 0, 0, 0, 0, 0, 0}, {1}, &compileInfo, "Ascend310P", 8, 196608);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, UINT64_MAX, "", {}, 0, false);
}

TEST_F(MatmulAllReduceArch20TilingExtraTest, Float16EmptyK)
{
    struct MatmulAllReduceCompileInfo {
    } compileInfo;
    gert::TilingContextPara::TensorDescription x1 = {GetStorageShape("256 0"), ge::DT_FLOAT16, ge::FORMAT_ND};
    gert::TilingContextPara::TensorDescription x2 = {GetStorageShape("0 8192"), ge::DT_FLOAT16, ge::FORMAT_ND};
    gert::TilingContextPara::TensorDescription y = {GetStorageShape("256 8192"), ge::DT_FLOAT16, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara(
        "MatmulAllReduce",
        {x1, x2, TD_DEFAULT, TD_DEFAULT, TD_DEFAULT, TD_DEFAULT, TD_DEFAULT, TD_DEFAULT, TD_DEFAULT, TD_DEFAULT}, {y},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
         {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
         {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")}},
        {1, 1, 0, 0, 0, 0, 0, 0, 0, 0}, {1}, &compileInfo, "Ascend310P", 8, 196608);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, UINT64_MAX, "", {}, 0, false);
}

} // namespace MatmulAllReduceUT
