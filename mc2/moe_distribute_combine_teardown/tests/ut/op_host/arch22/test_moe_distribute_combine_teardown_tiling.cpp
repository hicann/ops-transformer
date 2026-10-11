/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_moe_distribute_combine_teardown_tiling.cpp
 * \brief host侧tiling ut (arch22/A3, ascend910_93)
 */
#include <iostream>
#include <gtest/gtest.h>
#include "mc2_tiling_case_executor.h"

namespace MoeDistributeCombineTeardownArch22UT {

const std::string OP_NAME = "MoeDistributeCombineTeardown";

struct MoeDistributeCombineTeardownTestParam {
    std::string caseName;
    std::initializer_list<int64_t> expandXShape;
    ge::DataType expandXDtype;
    ge::Format expandXFormat;
    std::initializer_list<int64_t> quantExpandXShape;
    ge::DataType quantExpandXDtype;
    ge::Format quantExpandXFormat;
    std::initializer_list<int64_t> expertIdsShape;
    ge::DataType expertIdsDtype;
    ge::Format expertIdsFormat;
    std::initializer_list<int64_t> expandIdxShape;
    ge::DataType expandIdxDtype;
    ge::Format expandIdxFormat;
    std::initializer_list<int64_t> expertScalesShape;
    ge::DataType expertScalesDtype;
    ge::Format expertScalesFormat;
    std::initializer_list<int64_t> commCmdInfoShape;
    ge::DataType commCmdInfoDtype;
    ge::Format commCmdInfoFormat;
    std::initializer_list<int64_t> xOutShape;
    ge::DataType xOutDtype;
    ge::Format xOutFormat;
    std::string groupEpAttr;
    int64_t epWorldSizeAttr;
    int64_t epRankIdAttr;
    int64_t moeExpertNumAttr;
    int64_t expertShardTypeAttr;
    int64_t sharedExpertNumAttr;
    int64_t sharedExpertRankNumAttr;
    int64_t globalBsAttr;
    int64_t commQuantModeAttr;
    int64_t commTypeAttr;
    std::string commAlgAttr;
    ge::graphStatus status;
};

inline std::ostream& operator<<(std::ostream& os, const MoeDistributeCombineTeardownTestParam& param)
{
    return os << param.caseName;
}

static MoeDistributeCombineTeardownTestParam g_testCases[] = {
    {"teardown_arch22_fp16_ep2_normal",
     {96, 4096},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     {96, 6144},
     ge::DT_INT8,
     ge::FORMAT_ND,
     {8, 6},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {48},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {8, 6},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {1568},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {8, 4096},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     "hccl_world_group",
     2,
     0,
     32,
     0,
     0,
     0,
     0,
     0,
     2,
     "",
     ge::GRAPH_FAILED},
    {"teardown_arch22_bf16_ep2_normal",
     {128, 7168},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {128, 10752},
     ge::DT_INT8,
     ge::FORMAT_ND,
     {8, 8},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {64},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {8, 8},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {2080},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {8, 7168},
     ge::DT_BF16,
     ge::FORMAT_ND,
     "hccl_world_group",
     2,
     0,
     32,
     0,
     0,
     0,
     16,
     0,
     2,
     "",
     ge::GRAPH_FAILED},
    {"teardown_arch22_fp16_ep4_normal",
     {192, 4096},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     {192, 6144},
     ge::DT_INT8,
     ge::FORMAT_ND,
     {8, 6},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {48},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {8, 6},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {3136},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {8, 4096},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     "hccl_world_group",
     4,
     0,
     32,
     0,
     0,
     0,
     0,
     0,
     2,
     "",
     ge::GRAPH_FAILED},
    {"teardown_arch22_bf16_ep8_normal",
     {256, 7168},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {256, 10752},
     ge::DT_INT8,
     ge::FORMAT_ND,
     {8, 8},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {64},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {8, 8},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {4224},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {8, 7168},
     ge::DT_BF16,
     ge::FORMAT_ND,
     "hccl_world_group",
     8,
     0,
     32,
     0,
     0,
     0,
     64,
     0,
     2,
     "",
     ge::GRAPH_FAILED},
    {"teardown_arch22_bf16_ep2_globalBs",
     {128, 7168},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {128, 10752},
     ge::DT_INT8,
     ge::FORMAT_ND,
     {8, 8},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {64},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {8, 8},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {2080},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {8, 7168},
     ge::DT_BF16,
     ge::FORMAT_ND,
     "hccl_world_group",
     2,
     0,
     32,
     0,
     0,
     0,
     16,
     0,
     2,
     "",
     ge::GRAPH_FAILED},
    {"teardown_arch22_invalid_dtype_error",
     {128, 7168},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {128, 10752},
     ge::DT_INT8,
     ge::FORMAT_ND,
     {8, 8},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {64},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {8, 8},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {2080},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {8, 7168},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     "hccl_world_group",
     2,
     0,
     32,
     0,
     0,
     0,
     16,
     0,
     2,
     "",
     ge::GRAPH_FAILED},
};

class MoeDistributeCombineTeardownTilingArch22Test
    : public testing::TestWithParam<MoeDistributeCombineTeardownTestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeDistributeCombineTeardownTilingArch22Test SetUp." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeDistributeCombineTeardownTilingArch22Test TearDown." << std::endl;
    }
};

static struct MoeDistributeCombineTeardownCompileInfo {
} compileInfo;

static gert::TilingContextPara BuildTilingContextPara(const MoeDistributeCombineTeardownTestParam& param)
{
    std::cout << "[TEST_CASE] " << param.caseName << std::endl;
    gert::StorageShape expandXShape = {param.expandXShape, param.expandXShape};
    gert::StorageShape quantExpandXShape = {param.quantExpandXShape, param.quantExpandXShape};
    gert::StorageShape expertIdsShape = {param.expertIdsShape, param.expertIdsShape};
    gert::StorageShape expandIdxShape = {param.expandIdxShape, param.expandIdxShape};
    gert::StorageShape expertScalesShape = {param.expertScalesShape, param.expertScalesShape};
    gert::StorageShape commCmdInfoShape = {param.commCmdInfoShape, param.commCmdInfoShape};
    gert::StorageShape xOutShape = {param.xOutShape, param.xOutShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{expandXShape, param.expandXDtype, param.expandXFormat},
         {quantExpandXShape, param.quantExpandXDtype, param.quantExpandXFormat},
         {expertIdsShape, param.expertIdsDtype, param.expertIdsFormat},
         {expandIdxShape, param.expandIdxDtype, param.expandIdxFormat},
         {expertScalesShape, param.expertScalesDtype, param.expertScalesFormat},
         {commCmdInfoShape, param.commCmdInfoDtype, param.commCmdInfoFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{xOutShape, param.xOutDtype, param.xOutFormat}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_(
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.groupEpAttr)},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.epWorldSizeAttr)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.epRankIdAttr)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.moeExpertNumAttr)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.expertShardTypeAttr)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.sharedExpertNumAttr)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.sharedExpertRankNumAttr)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.globalBsAttr)},
         {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.commQuantModeAttr)},
         {"comm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.commTypeAttr)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.commAlgAttr)}});
    return gert::TilingContextPara(OP_NAME, inputTensorDesc_, outputTensorDesc_, attrs_, &compileInfo, "Ascend910_93");
}

TEST_P(MoeDistributeCombineTeardownTilingArch22Test, Arch22CasesTest)
{
    auto param = GetParam();
    auto tilingContextPara = BuildTilingContextPara(param);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", param.epWorldSizeAttr}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, param.status);
}

INSTANTIATE_TEST_CASE_P(MoeDistributeCombineTeardownTilingArch22UT, MoeDistributeCombineTeardownTilingArch22Test,
                        testing::ValuesIn(g_testCases));

} // namespace MoeDistributeCombineTeardownArch22UT
