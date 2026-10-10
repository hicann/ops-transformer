/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_moe_distribute_dispatch_setup_tiling_arch22.cpp
 * \brief arch22 tiling UT
 */

#include <iostream>
#include <fstream>
#include <thread>
#include <vector>
#include <string>
#include <gtest/gtest.h>
#include "mc2_tiling_case_executor.h"

namespace MoeDistributeDispatchSetupArch22UT {

static const std::string OP_NAME = "MoeDistributeDispatchSetup";

struct MoeDistributeDispatchSetupArch22TestParam {
    std::string caseName;
    std::initializer_list<int64_t> xShape;
    ge::DataType xDtype;
    ge::Format xFormat;

    std::initializer_list<int64_t> expertIdsShape;
    ge::DataType expertIdsDtype;
    ge::Format expertIdsFormat;

    std::initializer_list<int64_t> scalesShape;
    ge::DataType scalesDtype;
    ge::Format scalesFormat;

    std::initializer_list<int64_t> xActiveMaskShape;
    ge::DataType xActiveMaskDtype;
    ge::Format xActiveMaskFormat;

    std::initializer_list<int64_t> yOutShape;
    ge::DataType yOutDtype;
    ge::Format yOutFormat;

    std::initializer_list<int64_t> expandIdxOutShape;
    ge::DataType expandIdxOutDtype;
    ge::Format expandIdxOutFormat;

    std::initializer_list<int64_t> commCmdInfoOutShape;
    ge::DataType commCmdInfoOutDtype;
    ge::Format commCmdInfoOutFormat;

    std::string groupEp;
    int64_t epWorldSize;
    int64_t epRankId;
    int64_t moeExpertNum;
    int64_t expertShardType;
    int64_t sharedExpertNum;
    int64_t sharedExpertRankNum;
    int64_t quantMode;
    int64_t globalBs;
    int64_t commType;
    std::string commAlg;

    std::string socVersion;
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
    uint64_t mc2TilingDataReservedLen;
};

inline std::ostream& operator<<(std::ostream& os, const MoeDistributeDispatchSetupArch22TestParam& param)
{
    return os << param.caseName;
}

static MoeDistributeDispatchSetupArch22TestParam g_arch22TestCases[] = {
    {"test_moe_distribute_dispatch_setup_arch22_normal_fp16_bs8_h7168_k8",
     {8, 7168},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     {8, 8},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {},
     ge::DT_BOOL,
     ge::FORMAT_ND,
     {64, 7168},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     {64},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {1536},
     ge::DT_INT32,
     ge::FORMAT_ND,
     "group_ep",
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
     "Ascend910B",
     ge::GRAPH_FAILED,
     0UL,
     "",
     {},
     0},
    {"test_moe_distribute_dispatch_setup_arch22_normal_bf16_bs8_h7168_k8",
     {8, 7168},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {8, 8},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {},
     ge::DT_BOOL,
     ge::FORMAT_ND,
     {64, 7168},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {64},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {1536},
     ge::DT_INT32,
     ge::FORMAT_ND,
     "group_ep",
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
     "Ascend910B",
     ge::GRAPH_FAILED,
     0UL,
     "",
     {},
     0},
    {"test_moe_distribute_dispatch_setup_arch22_with_scales",
     {8, 7168},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     {8, 8},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {32, 7168},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {},
     ge::DT_BOOL,
     ge::FORMAT_ND,
     {64, 7168},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     {64},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {1536},
     ge::DT_INT32,
     ge::FORMAT_ND,
     "group_ep",
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
     "Ascend910B",
     ge::GRAPH_FAILED,
     0UL,
     "",
     {},
     0},
    {"test_moe_distribute_dispatch_setup_arch22_ep4_bs8",
     {8, 7168},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     {8, 8},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {},
     ge::DT_BOOL,
     ge::FORMAT_ND,
     {64, 7168},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     {64},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {1536},
     ge::DT_INT32,
     ge::FORMAT_ND,
     "group_ep",
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
     "Ascend910B",
     ge::GRAPH_FAILED,
     0UL,
     "",
     {},
     0},
};

class MoeDistributeDispatchSetupArch22TilingTest
    : public testing::TestWithParam<MoeDistributeDispatchSetupArch22TestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeDistributeDispatchSetupArch22TilingTest SetUp." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeDistributeDispatchSetupArch22TilingTest TearDown." << std::endl;
    }
};

gert::StorageShape MakeShape(const std::initializer_list<int64_t>& input_shape)
{
    if (input_shape.size() == 0) {
        return gert::StorageShape{};
    }
    return gert::StorageShape{input_shape, input_shape};
}

static struct MoeDistributeDispatchSetupCompileInfo {
} compileInfo;

static gert::TilingContextPara BuildTilingContextPara(const MoeDistributeDispatchSetupArch22TestParam& param)
{
    std::cout << "[TEST_CASE] " << param.caseName << std::endl;

    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_;
    inputTensorDesc_.push_back({MakeShape(param.xShape), param.xDtype, param.xFormat});
    inputTensorDesc_.push_back({MakeShape(param.expertIdsShape), param.expertIdsDtype, param.expertIdsFormat});
    if (param.scalesShape.size() > 0) {
        inputTensorDesc_.push_back({MakeShape(param.scalesShape), param.scalesDtype, param.scalesFormat});
    }
    if (param.xActiveMaskShape.size() > 0) {
        inputTensorDesc_.push_back(
            {MakeShape(param.xActiveMaskShape), param.xActiveMaskDtype, param.xActiveMaskFormat});
    }

    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_;
    outputTensorDesc_.push_back({MakeShape(param.yOutShape), param.yOutDtype, param.yOutFormat});
    outputTensorDesc_.push_back(
        {MakeShape(param.expandIdxOutShape), param.expandIdxOutDtype, param.expandIdxOutFormat});
    outputTensorDesc_.push_back(
        {MakeShape(param.commCmdInfoOutShape), param.commCmdInfoOutDtype, param.commCmdInfoOutFormat});

    std::vector<gert::TilingContextPara::OpAttr> attrs_;
    attrs_.push_back({"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.groupEp)});
    attrs_.push_back({"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.epWorldSize)});
    attrs_.push_back({"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.epRankId)});
    attrs_.push_back({"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.moeExpertNum)});
    attrs_.push_back({"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.expertShardType)});
    attrs_.push_back({"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.sharedExpertNum)});
    attrs_.push_back(
        {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.sharedExpertRankNum)});
    attrs_.push_back({"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.quantMode)});
    attrs_.push_back({"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.globalBs)});
    attrs_.push_back({"comm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.commType)});
    attrs_.push_back({"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.commAlg)});

    return gert::TilingContextPara(OP_NAME, inputTensorDesc_, outputTensorDesc_, attrs_, &compileInfo,
                                   param.socVersion);
}

TEST_P(MoeDistributeDispatchSetupArch22TilingTest, GeneralCases)
{
    auto param = GetParam();
    auto tilingContextPara = BuildTilingContextPara(param);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", param.epWorldSize},
                                               {"HCCL_BUFFSIZE", 12000UL * 1024UL * 1024UL}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, param.status, param.expectTilingKey,
                       param.expectTilingData, param.expectWorkspaces, param.mc2TilingDataReservedLen);
}

INSTANTIATE_TEST_CASE_P(MoeDistributeDispatchSetupArch22TilingUT, MoeDistributeDispatchSetupArch22TilingTest,
                        testing::ValuesIn(g_arch22TestCases));

} // namespace MoeDistributeDispatchSetupArch22UT
