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
#include <limits>
#include <map>
#include <gtest/gtest.h>
#include "../../../op_host/moe_init_routing_v3_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "platform/platform_infos_def.h"
#include "op_tiling_parse_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

using namespace std;

namespace {
constexpr uint32_t ASCEND910B_CORE_NUM = 64;
constexpr uint32_t ASCEND910B_L2_SIZE = 33554432;
constexpr uint32_t ASCEND910B_UB_SIZE = 262144;
constexpr uint32_t ASCEND910B_L1_SIZE = 524288;

void InitParsePlatformInfo(fe::PlatFormInfos &platformInfo)
{
    platformInfo.Init();
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    socInfos["ai_core_cnt"] = to_string(ASCEND910B_CORE_NUM);
    socInfos["l2_size"] = to_string(ASCEND910B_L2_SIZE);
    socInfos["core_type_list"] = "AICore";
    aicoreSpec["ub_size"] = to_string(ASCEND910B_UB_SIZE);
    aicoreSpec["l1_size"] = to_string(ASCEND910B_L1_SIZE);
    map<string, string> versions = {{"NpuArch", "2201"}, {"Short_SoC_version", "Ascend910B"}};
    platformInfo.SetPlatformRes("SoCInfo", socInfos);
    platformInfo.SetPlatformRes("AICoreSpec", aicoreSpec);
    platformInfo.SetCoreNumByCoreType("AICore");
    platformInfo.SetPlatformRes("version", versions);
}
} // namespace

constexpr int64_t QUANT_MODE_NONE = -1;
constexpr int64_t QUANT_MODE_STATIC = 0;
constexpr int64_t QUANT_MODE_DYNAMIC = 1;
constexpr int64_t ROW_IDX_TYPE_DROPPAD = 0;
constexpr int64_t ROW_IDX_TYPE_DROPLESS = 1;
constexpr int64_t EXPERT_NUM = 256;
constexpr uint64_t SKIP_TILING_KEY_VALIDATION = std::numeric_limits<uint64_t>::max();
constexpr uint64_t EMPTY_TENSOR_TILINGKEY = 3000000ULL;
constexpr size_t EMPTY_TENSOR_WORKSPACE = 16ULL * 1024ULL * 1024ULL;
constexpr int64_t EXPERT_TOKENS_TYPE_COUNT = 1;
constexpr int64_t EXPERT_TOKENS_TYPE_KEY_VALUE = 2;

gert::StorageShape MakeShape(const std::vector<int64_t> &dims)
{
    switch (dims.size()) {
        case 0:
            return gert::StorageShape({}, {});
        case 1:
            return gert::StorageShape({dims[0]}, {dims[0]});
        case 2:
            return gert::StorageShape({dims[0], dims[1]}, {dims[0], dims[1]});
        case 3:
            return gert::StorageShape({dims[0], dims[1], dims[2]}, {dims[0], dims[1], dims[2]});
        default:
            return gert::StorageShape({}, {});
    }
}

gert::TilingContextPara MakeExtendedTilingContextPara(
    const std::vector<gert::TilingContextPara::TensorDescription> &inputs,
    const std::vector<gert::TilingContextPara::TensorDescription> &outputs,
    const std::vector<gert::TilingContextPara::OpAttr> &attrs,
    optiling::MoeInitRoutingV3CompileInfo *compileInfo)
{
    return gert::TilingContextPara("MoeInitRoutingV3", inputs, outputs, attrs, compileInfo);
}

void RunExtendedTilingCase(const gert::TilingContextPara &tilingContextPara,
                           ge::graphStatus expectResult = ge::GRAPH_SUCCESS,
                           uint64_t expectTilingKey = SKIP_TILING_KEY_VALIDATION,
                           const std::vector<size_t> &expectWorkspaces = {})
{
    ExecuteTestCase(tilingContextPara, expectResult, expectTilingKey, "", expectWorkspaces);
}

ge::DataType GetLargeColsExpandedXDtype(int64_t quantMode, ge::DataType xDataType)
{
    if (quantMode == QUANT_MODE_NONE) {
        return xDataType;
    }
    return ge::DT_INT8;
}

std::vector<int64_t> GetLargeColsExpertTokensShape(int64_t expertTokensNumType, int64_t expertNum, int64_t expertRange)
{
    if (expertTokensNumType == EXPERT_TOKENS_TYPE_KEY_VALUE) {
        return {expertNum, 2};
    }
    return {expertRange};
}

std::vector<gert::TilingContextPara::TensorDescription> BuildLargeColsInputs(
    int64_t n, int64_t h, int64_t k, ge::DataType xDataType,
    const std::vector<gert::TilingContextPara::TensorDescription> &extraInputs)
{
    std::vector<gert::TilingContextPara::TensorDescription> inputs;
    inputs.emplace_back(gert::StorageShape({n, h}, {n, h}), xDataType, ge::FORMAT_ND);
    inputs.emplace_back(gert::StorageShape({n, k}, {n, k}), ge::DT_INT32, ge::FORMAT_ND);
    inputs.insert(inputs.end(), extraInputs.begin(), extraInputs.end());
    return inputs;
}

std::vector<gert::TilingContextPara::TensorDescription> BuildLargeColsOutputs(
    int64_t totalLength, int64_t h, ge::DataType expandedXDtype, const std::vector<int64_t> &expertTokensShape,
    const std::vector<gert::TilingContextPara::TensorDescription> &extraOutputs)
{
    std::vector<gert::TilingContextPara::TensorDescription> outputs;
    outputs.emplace_back(gert::StorageShape({totalLength, h}, {totalLength, h}), expandedXDtype, ge::FORMAT_ND);
    outputs.emplace_back(gert::StorageShape({totalLength}, {totalLength}), ge::DT_INT32, ge::FORMAT_ND);
    outputs.emplace_back(MakeShape(expertTokensShape), ge::DT_INT64, ge::FORMAT_ND);
    outputs.emplace_back(gert::StorageShape({totalLength}, {totalLength}), ge::DT_FLOAT, ge::FORMAT_ND);
    outputs.insert(outputs.end(), extraOutputs.begin(), extraOutputs.end());
    return outputs;
}

std::vector<gert::TilingContextPara::OpAttr> BuildLargeColsAttrs(
    int64_t totalLength, int64_t dropPadMode, int64_t expertTokensNumType, bool expertTokensNumFlag, int64_t quantMode,
    const std::vector<int64_t> &aciveExpertRange, int64_t rowIdxType)
{
    return {
        {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(totalLength)},
        {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0LL)},
        {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(EXPERT_NUM)},
        {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
        {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertTokensNumType)},
        {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(expertTokensNumFlag)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
        {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(aciveExpertRange)},
        {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rowIdxType)},
    };
}

void RunLargeColsCase(int64_t N, int64_t H, int64_t K, int64_t quantMode, int64_t dropPadMode,
                      int64_t expertTokensNumType, bool expertTokensNumFlag, ge::DataType xDataType,
                      std::vector<int64_t> aciveExpertRange, int64_t rowIdxType, uint64_t ubSize,
                      const std::vector<gert::TilingContextPara::TensorDescription> &extraInputs,
                      const std::vector<gert::TilingContextPara::TensorDescription> &extraOutputs)
{
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, ubSize};
    int64_t expertNum = EXPERT_NUM;
    int64_t expertRange = aciveExpertRange.empty() ? expertNum : (aciveExpertRange[1] - aciveExpertRange[0]);
    int64_t totalLength = N * K;
    ge::DataType expandedXDtype = GetLargeColsExpandedXDtype(quantMode, xDataType);
    std::vector<int64_t> expertTokensShape =
        GetLargeColsExpertTokensShape(expertTokensNumType, expertNum, expertRange);
    auto inputs = BuildLargeColsInputs(N, H, K, xDataType, extraInputs);
    auto outputs = BuildLargeColsOutputs(totalLength, H, expandedXDtype, expertTokensShape, extraOutputs);
    auto attrs = BuildLargeColsAttrs(totalLength, dropPadMode, expertTokensNumType, expertTokensNumFlag, quantMode,
                                     aciveExpertRange, rowIdxType);
    RunExtendedTilingCase(MakeExtendedTilingContextPara(inputs, outputs, attrs, &compileInfo));
}

class MoeInitRoutingV3Tiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeInitRoutingV3Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeInitRoutingV3Tiling TearDown" << std::endl;
    }
};

void RunNormalCase(int64_t N, int64_t H, int64_t K, int64_t C, int64_t activeNum, int64_t dropPadMode, int64_t countFlag, 
                   bool tokenFlag, int64_t quantMode, int64_t scaleFlag, ge::DataType xDataType, std::vector<int64_t> aciveExpertRange,
                   int64_t rowIdxType, ge::graphStatus result, int64_t expectTilingKey, string expectTilingData, vector<size_t> expectWorkspaces)
{   
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t expert_num = EXPERT_NUM;
    int64_t E = aciveExpertRange[1] - aciveExpertRange[0];
    ge::DataType expandedx_type;
    if (quantMode == 0) {
        expandedx_type = xDataType;
    }
    else {
        expandedx_type = ge::DT_INT8;
    }
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3",
                                        {
                                            {{{N, H}, {N, H}}, xDataType, ge::FORMAT_ND},
                                            {{{N, K}, {N, K}}, ge::DT_INT32, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{(N * K > activeNum) ? activeNum : N * K, H}, {(N * K > activeNum) ? activeNum : N * K, H}}, expandedx_type, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{(N * K > activeNum) ? activeNum : N * K}, {(N * K > activeNum) ? activeNum : N * K}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                        },
                                        {
                                            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
                                            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(C)},
                                            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
                                            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
                                            {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(countFlag)},
                                            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tokenFlag)},
                                            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
                                            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(aciveExpertRange)},
                                            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rowIdxType)},
                                        },
                                        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

void RunNormalCaseDropless(int64_t N, int64_t H, int64_t K, int64_t C, int64_t activeNum, int64_t dropPadMode, int64_t countFlag, 
                   bool tokenFlag, int64_t quantMode, int64_t scaleFlag, ge::DataType xDataType, std::vector<int64_t> aciveExpertRange,
                   int64_t rowIdxType, ge::graphStatus result, int64_t expectTilingKey, string expectTilingData, vector<size_t> expectWorkspaces)
{   
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t expert_num = EXPERT_NUM;
    int64_t E = aciveExpertRange[1] - aciveExpertRange[0];
    ge::DataType expandedx_type;
    if (quantMode == 0) {
        expandedx_type = xDataType;
    }
    else {
        expandedx_type = ge::DT_INT8;
    }
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3",
                                        {
                                            {{{N, H}, {N, H}}, xDataType, ge::FORMAT_ND},
                                            {{{N, K}, {N, K}}, ge::DT_INT32, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{N * K, H}, {N * K, H}}, expandedx_type, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                        },
                                        {
                                            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
                                            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(C)},
                                            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
                                            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
                                            {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(countFlag)},
                                            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tokenFlag)},
                                            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
                                            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(aciveExpertRange)},
                                            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rowIdxType)},
                                        },
                                        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

void RunNormalCaseNoQuant(int64_t N, int64_t H, int64_t K, int64_t C, int64_t activeNum, int64_t dropPadMode, int64_t countFlag, 
                   bool tokenFlag, int64_t quantMode, int64_t scaleFlag, ge::DataType xDataType, std::vector<int64_t> aciveExpertRange,
                   int64_t rowIdxType, ge::graphStatus result, int64_t expectTilingKey, string expectTilingData, vector<size_t> expectWorkspaces)
{   
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t expert_num = EXPERT_NUM;
    int64_t E = aciveExpertRange[1] - aciveExpertRange[0];
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3",
                                        {
                                            {{{N, H}, {N, H}}, xDataType, ge::FORMAT_ND},
                                            {{{N, K}, {N, K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{N}, {N}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{(N * K > activeNum) ? activeNum : N * K, H}, {(N * K > activeNum) ? activeNum : N * K, H}}, xDataType, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{(N * K > activeNum) ? activeNum : N * K}, {(N * K > activeNum) ? activeNum : N * K}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                        },
                                        {
                                            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
                                            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(C)},
                                            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
                                            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
                                            {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(countFlag)},
                                            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tokenFlag)},
                                            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
                                            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(aciveExpertRange)},
                                            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rowIdxType)},
                                        },
                                        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

void RunNormalCaseNoQuantDroppad(int64_t N, int64_t H, int64_t K, int64_t C, int64_t activeNum, int64_t dropPadMode, int64_t countFlag, 
                   bool tokenFlag, int64_t quantMode, int64_t scaleFlag, ge::DataType xDataType, std::vector<int64_t> aciveExpertRange,
                   int64_t rowIdxType, ge::graphStatus result, int64_t expectTilingKey, string expectTilingData, vector<size_t> expectWorkspaces)
{   
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t expert_num = EXPERT_NUM;
    int64_t E = EXPERT_NUM;
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3",
                                        {
                                            {{{N, H}, {N, H}}, xDataType, ge::FORMAT_ND},
                                            {{{N, K}, {N, K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{N}, {N}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{expert_num, C, H}, {expert_num, C, H}}, xDataType, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{expert_num * C}, {expert_num * C}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                        },
                                        {
                                            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
                                            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(C)},
                                            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
                                            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
                                            {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(countFlag)},
                                            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tokenFlag)},
                                            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
                                            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(aciveExpertRange)},
                                            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rowIdxType)},
                                        },
                                        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

void RunNormalCaseKeyValue(int64_t N, int64_t H, int64_t K, int64_t C, int64_t activeNum, int64_t dropPadMode, int64_t countFlag, 
                   bool tokenFlag, int64_t quantMode, int64_t scaleFlag, ge::DataType xDataType, std::vector<int64_t> aciveExpertRange,
                   int64_t rowIdxType, ge::graphStatus result, int64_t expectTilingKey, string expectTilingData, vector<size_t> expectWorkspaces)
{   
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t expert_num = EXPERT_NUM;
    int64_t E = aciveExpertRange[1] - aciveExpertRange[0];
    ge::DataType expandedx_type;
    if (quantMode == 0) {
        expandedx_type = xDataType;
    }
    else {
        expandedx_type = ge::DT_INT8;
    }
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3",
                                        {
                                            {{{N, H}, {N, H}}, xDataType, ge::FORMAT_ND},
                                            {{{N, K}, {N, K}}, ge::DT_INT32, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{N * K, H}, {N * K, H}}, expandedx_type, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{expert_num, 2}, {expert_num, 2}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                        },
                                        {
                                            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
                                            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(C)},
                                            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
                                            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
                                            {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(countFlag)},
                                            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tokenFlag)},
                                            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
                                            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(aciveExpertRange)},
                                            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rowIdxType)},
                                        },
                                        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

void RunNormalCaseFullload(int64_t N, int64_t H, int64_t K, int64_t C, int64_t activeNum, int64_t dropPadMode, int64_t countFlag, 
                   bool tokenFlag, int64_t quantMode, int64_t scaleFlag, ge::DataType xDataType, std::vector<int64_t> aciveExpertRange,
                   int64_t rowIdxType, ge::graphStatus result, int64_t expectTilingKey, string expectTilingData, vector<size_t> expectWorkspaces)
{   
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t expert_num = EXPERT_NUM;
    int64_t E = aciveExpertRange[1] - aciveExpertRange[0];
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3",
                                        {
                                            {{{N, H}, {N, H}}, xDataType, ge::FORMAT_ND},
                                            {{{N, K}, {N, K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{E, H}, {E, H}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{N * K, H}, {N * K, H}}, ge::DT_INT8, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{expert_num, 2}, {expert_num, 2}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                        },
                                        {
                                            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
                                            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(C)},
                                            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
                                            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
                                            {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(countFlag)},
                                            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tokenFlag)},
                                            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
                                            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(aciveExpertRange)},
                                            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rowIdxType)},
                                        },
                                        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

void RunNormalCaseStaticQuant(int64_t N, int64_t H, int64_t K, int64_t C, int64_t activeNum, int64_t dropPadMode, int64_t countFlag, 
                   bool tokenFlag, int64_t quantMode, int64_t scaleFlag, ge::DataType xDataType, std::vector<int64_t> aciveExpertRange,
                   int64_t rowIdxType, ge::graphStatus result, int64_t expectTilingKey, string expectTilingData, vector<size_t> expectWorkspaces)
{   
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t expert_num = EXPERT_NUM;
    int64_t E = aciveExpertRange[1] - aciveExpertRange[0];
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3",
                                        {
                                            {{{N, H}, {N, H}}, xDataType, ge::FORMAT_ND},
                                            {{{N, K}, {N, K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{(N * K > activeNum) ? activeNum : N * K, H}, {(N * K > activeNum) ? activeNum : N * K, H}}, ge::DT_INT8, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{(N * K > activeNum) ? activeNum : N * K}, {(N * K > activeNum) ? activeNum : N * K}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                        },
                                        {
                                            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
                                            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(C)},
                                            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
                                            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
                                            {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(countFlag)},
                                            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tokenFlag)},
                                            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
                                            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(aciveExpertRange)},
                                            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rowIdxType)},
                                        },
                                        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

void RunNormalCaseStaticQuantDroppad(int64_t N, int64_t H, int64_t K, int64_t C, int64_t activeNum, int64_t dropPadMode, int64_t countFlag, 
                   bool tokenFlag, int64_t quantMode, int64_t scaleFlag, ge::DataType xDataType, std::vector<int64_t> aciveExpertRange,
                   int64_t rowIdxType, ge::graphStatus result, int64_t expectTilingKey, string expectTilingData, vector<size_t> expectWorkspaces)
{   
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t expert_num = EXPERT_NUM;
    int64_t E = EXPERT_NUM;
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3",
                                        {
                                            {{{N, H}, {N, H}}, xDataType, ge::FORMAT_ND},
                                            {{{N, K}, {N, K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{expert_num, C, H}, {expert_num, C, H}}, ge::DT_INT8, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{expert_num * C}, {expert_num * C}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                        },
                                        {
                                            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
                                            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(C)},
                                            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
                                            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
                                            {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(countFlag)},
                                            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tokenFlag)},
                                            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
                                            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(aciveExpertRange)},
                                            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rowIdxType)},
                                        },
                                        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

void RunNormalCaseDynamicQuant1H(int64_t N, int64_t H, int64_t K, int64_t C, int64_t activeNum, int64_t dropPadMode, int64_t countFlag, 
                   bool tokenFlag, int64_t quantMode, int64_t scaleFlag, ge::DataType xDataType, std::vector<int64_t> aciveExpertRange,
                   int64_t rowIdxType, ge::graphStatus result, int64_t expectTilingKey, string expectTilingData, vector<size_t> expectWorkspaces)
{   
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t expert_num = EXPERT_NUM;
    int64_t E = aciveExpertRange[1] - aciveExpertRange[0];
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3",
                                        {
                                            {{{N, H}, {N, H}}, xDataType, ge::FORMAT_ND},
                                            {{{N, K}, {N, K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1, H}, {1, H}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{(N * K > activeNum) ? activeNum : N * K, H}, {(N * K > activeNum) ? activeNum : N * K, H}}, ge::DT_INT8, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{(N * K > activeNum) ? activeNum : N * K}, {(N * K > activeNum) ? activeNum : N * K}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                        },
                                        {
                                            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
                                            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(C)},
                                            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
                                            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
                                            {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(countFlag)},
                                            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tokenFlag)},
                                            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
                                            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(aciveExpertRange)},
                                            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rowIdxType)},
                                        },
                                        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

void RunNormalCaseDynamicQuantEH(int64_t N, int64_t H, int64_t K, int64_t C, int64_t activeNum, int64_t dropPadMode, int64_t countFlag, 
                   bool tokenFlag, int64_t quantMode, int64_t scaleFlag, ge::DataType xDataType, std::vector<int64_t> aciveExpertRange,
                   int64_t rowIdxType, ge::graphStatus result, int64_t expectTilingKey, string expectTilingData, vector<size_t> expectWorkspaces)
{   
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t expert_num = EXPERT_NUM;
    int64_t E = aciveExpertRange[1] - aciveExpertRange[0];
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3",
                                        {
                                            {{{N, H}, {N, H}}, xDataType, ge::FORMAT_ND},
                                            {{{N, K}, {N, K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{E, H}, {E, H}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{(N * K > activeNum) ? activeNum : N * K, H}, {(N * K > activeNum) ? activeNum : N * K, H}}, ge::DT_INT8, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{(N * K > activeNum) ? activeNum : N * K}, {(N * K > activeNum) ? activeNum : N * K}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                        },
                                        {
                                            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
                                            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(C)},
                                            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
                                            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
                                            {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(countFlag)},
                                            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tokenFlag)},
                                            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
                                            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(aciveExpertRange)},
                                            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rowIdxType)},
                                        },
                                        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

void RunNormalCaseDynamicQuantDroppad(int64_t N, int64_t H, int64_t K, int64_t C, int64_t activeNum, int64_t dropPadMode, int64_t countFlag, 
                   bool tokenFlag, int64_t quantMode, int64_t scaleFlag, ge::DataType xDataType, std::vector<int64_t> aciveExpertRange,
                   int64_t rowIdxType, ge::graphStatus result, int64_t expectTilingKey, string expectTilingData, vector<size_t> expectWorkspaces)
{   
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t expert_num = EXPERT_NUM;
    int64_t E = EXPERT_NUM;
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3",
                                        {
                                            {{{N, H}, {N, H}}, xDataType, ge::FORMAT_ND},
                                            {{{N, K}, {N, K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{E, H}, {E, H}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{expert_num, C, H}, {expert_num, C, H}}, ge::DT_INT8, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{expert_num * C}, {expert_num * C}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                        },
                                        {
                                            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
                                            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(C)},
                                            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
                                            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
                                            {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(countFlag)},
                                            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tokenFlag)},
                                            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
                                            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(aciveExpertRange)},
                                            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rowIdxType)},
                                        },
                                        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 通用模板
// 单核 + not quant + active + gather + scale not None float32  1000000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_01)
{   
    string expectTilingData = "40 200 2880 1 0 256 256 -1 0 1 0 256 1 0 0 0 100 0 0 0 0 1 200 1 200 200 200 1 200 200 1984 0 1504 40 5 5 1 5 5 1 5 5 40 5 5 1 5 5 1 5 5 1 2880 2880 0 40 5 5 5 5 5 5 1 1 2880 2880 1 0 0 0 0 0 0 0 0 0 0 0 0 40 5 5 5 5 5 5 1 1 ";
    std::vector<size_t> expectWorkspaces = {16786432};
    RunNormalCaseNoQuant(200, 2880, 1, 0, 100, 0, 1, false, QUANT_MODE_NONE, 1, ge::DT_FLOAT, {0, 256}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 1000000, expectTilingData, expectWorkspaces);
}

// 单核 + not quant + dropless + scatter + consum + scale None bfloat16  1001000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_02)
{
    string expectTilingData = "40 255 3584 1 0 256 256 -1 1 0 0 256 0 1 0 0 255 0 0 0 0 1 255 1 255 255 255 1 255 255 1984 0 1504 37 7 3 1 7 7 1 3 3 37 7 3 1 7 7 1 3 3 1 3584 3584 0 37 7 7 7 3 3 3 1 1 3584 3584 1 0 0 0 0 0 0 0 0 0 0 0 0 37 7 7 7 3 3 3 1 1 ";
    std::vector<size_t> expectWorkspaces = {16787972};
    RunNormalCaseDropless(255, 3584, 1, 0, 0, 0, 0, true, QUANT_MODE_NONE, 0, ge::DT_BF16, {0, 256}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1001000, expectTilingData, expectWorkspaces);
}

// 多核 + not quant + dropless + gather + keyvalue + scale None int8  1100000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_03)
{
    string expectTilingData = "40 160 96 1450 0 256 256 -1 0 0 0 256 2 1 0 0 232000 0 0 0 0 40 5824 3 1984 1856 4864 3 1632 1600 1984 10 1504 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 1 5800 5800 1 5800 5800 1 96 96 0 40 5800 5800 5800 5800 5800 5800 1 1 96 96 1 0 0 0 0 0 0 0 0 0 0 0 0 40 5800 1450 1450 5800 1450 1450 4 4 ";
    std::vector<size_t> expectWorkspaces = {23276832};
    RunNormalCaseKeyValue(160, 96, 1450, 0, 0, 0, 2, true, QUANT_MODE_NONE, 0, ge::DT_INT8, {0, 256}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 1100000, expectTilingData, expectWorkspaces);
}

// 多核 + not quant + active + scatter + count + scale None float16  1101000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_04)
{
    string expectTilingData = "40 160 96 1450 180 192 12 -1 1 0 0 256 1 1 0 1 230000 0 0 0 0 40 5824 3 1984 1856 4864 3 1632 1600 1984 10 1504 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 1 5800 5800 1 5800 5800 1 96 96 0 40 5800 5800 5800 5800 5800 5800 1 1 96 96 1 0 0 0 0 0 0 0 0 0 0 0 0 40 5800 1450 1450 5800 1450 1450 4 4 ";
    std::vector<size_t> expectWorkspaces = {23275856};
    RunNormalCase(160, 96, 1450, 0, 230000, 0, 1, true, QUANT_MODE_NONE, 0, ge::DT_FLOAT16, {180, 192}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1101000, expectTilingData, expectWorkspaces);
}

// 单核 + not quant + droppad + gather + scale not None float32  1000100
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_05)
{   
    string expectTilingData = "40 89 2880 8 0 256 256 -1 0 1 0 256 1 0 0 0 712 1 0 0 13 1 712 1 712 712 712 1 712 712 1984 0 1504 40 18 10 1 18 18 1 10 10 40 18 10 1 18 18 1 10 10 1 2880 2880 0 40 18 18 18 10 10 10 1 1 2880 2880 1 0 0 0 0 0 0 0 0 0 0 0 0 40 18 18 18 10 10 10 1 1 ";
    std::vector<size_t> expectWorkspaces = {16801088};
    RunNormalCaseNoQuantDroppad(89, 2880, 8, 13, 712, 1, 1, false, QUANT_MODE_NONE, 1, ge::DT_FLOAT, {0, 256}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 1000100, expectTilingData, expectWorkspaces);
}

// 多核 + not quant + droppad + gather + count + scale not None int8  1100100
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_06)
{
    string expectTilingData = "40 250 2880 8 0 256 256 -1 0 1 0 256 1 1 0 0 2000 1 0 0 78 4 512 1 512 512 464 1 480 464 1984 0 1504 40 50 50 1 50 50 1 50 50 40 50 50 1 50 50 1 50 50 1 2880 2880 0 40 50 50 50 50 50 50 1 1 2880 2880 1 0 0 0 0 0 0 0 0 0 0 0 0 40 50 50 50 50 50 50 1 1 ";
    std::vector<size_t> expectWorkspaces = {16837152};
    RunNormalCaseNoQuantDroppad(250, 2880, 8, 78, 7408, 1, 1, true, QUANT_MODE_NONE, 1, ge::DT_INT8, {0, 256}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 1100100, expectTilingData, expectWorkspaces);
}

// 单核 + static quant + active + gather float32  1010000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_07)
{
    string expectTilingData = "40 200 2880 1 0 256 256 0 0 1 1 256 1 0 0 0 100 0 0 0 0 1 200 1 200 200 200 1 200 200 1984 0 1504 40 5 5 1 5 5 1 5 5 40 5 5 1 5 5 1 5 5 1 2880 2880 0 40 5 5 5 5 5 5 1 1 2880 2880 1 0 0 0 0 0 0 0 0 0 0 0 0 40 5 5 5 5 5 5 1 1 ";
    std::vector<size_t> expectWorkspaces = {16786432};
    RunNormalCaseStaticQuant(200, 2880, 1, 0, 100, 0, 1, false, QUANT_MODE_STATIC, 1, ge::DT_FLOAT, {0, 256}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 1010000, expectTilingData, expectWorkspaces);
}

// 单核 + static quant + active + scatter + count float16  1011000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_08)
{
    string expectTilingData = "40 255 4096 1 180 192 12 0 1 1 1 256 1 1 1 1 255 0 0 0 0 1 255 1 255 255 255 1 255 255 1984 0 1504 37 7 3 1 7 7 1 3 3 37 7 3 1 7 7 1 3 3 1 4096 4096 0 37 7 7 7 3 3 3 1 1 4096 4096 1 0 0 0 0 0 0 0 0 0 0 0 0 37 7 7 7 3 3 3 1 1 ";
    std::vector<size_t> expectWorkspaces = {16786996};
    RunNormalCaseStaticQuant(255, 4096, 1, 0, 500, 0, 1, true, QUANT_MODE_STATIC, 1, ge::DT_FLOAT16, {180, 192}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1011000, expectTilingData, expectWorkspaces);
}

// 多核 + static quant + dropless + gather + consum bfloat16  1110000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_09)
{
    string expectTilingData = "40 250 2880 8 0 256 256 0 0 1 1 256 0 1 0 0 2000 0 0 0 0 4 512 1 512 512 464 1 480 464 1984 0 1504 40 50 50 1 50 50 1 50 50 40 50 50 1 50 50 1 50 50 1 2880 2880 0 40 50 50 50 50 50 50 1 1 2880 2880 1 0 0 0 0 0 0 0 0 0 0 0 0 40 50 50 50 50 50 50 1 1 ";
    std::vector<size_t> expectWorkspaces = {16836832};
    RunNormalCaseStaticQuant(250, 2880, 8, 0, 2000, 0, 0, true, QUANT_MODE_STATIC, 1, ge::DT_BF16, {0, 256}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 1110000, expectTilingData, expectWorkspaces);
}

// 多核 + static quant + active + scatter + count float32  1111000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_10)
{
    string expectTilingData = "40 250 2880 8 180 192 12 0 1 1 1 256 1 1 0 1 2000 0 0 0 0 4 512 1 512 512 464 1 480 464 1984 0 1504 40 50 50 1 50 50 1 50 50 40 50 50 1 50 50 1 50 50 1 2880 2880 0 40 50 50 50 50 50 50 1 1 2880 2880 1 0 0 0 0 0 0 0 0 0 0 0 0 40 50 50 50 50 50 50 1 1 ";
    std::vector<size_t> expectWorkspaces = {16835856};
    RunNormalCaseStaticQuant(250, 2880, 8, 0, 2000, 0, 1, true, QUANT_MODE_STATIC, 1, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1111000, expectTilingData, expectWorkspaces);
}

// 单核 + static quant + droppad + gather bfloat16  1010100
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_11)
{
    string expectTilingData = "40 76 2880 8 0 256 256 0 0 1 1 256 1 0 0 0 608 1 0 0 9 1 608 1 608 608 608 1 608 608 1984 0 1504 38 16 16 1 16 16 1 16 16 38 16 16 1 16 16 1 16 16 1 2880 2880 0 38 16 16 16 16 16 16 1 1 2880 2880 1 0 0 0 0 0 0 0 0 0 0 0 0 38 16 16 16 16 16 16 1 1 ";
    std::vector<size_t> expectWorkspaces = {16798176};
    RunNormalCaseStaticQuantDroppad(76, 2880, 8, 9, 608, 1, 1, false, QUANT_MODE_STATIC, 1, ge::DT_BF16, {0, 256}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 1010100, expectTilingData, expectWorkspaces);
}

// 多核 + static quant + droppad + gather + bfloat16 1110100
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_12)
{
    string expectTilingData = "40 250 2880 8 0 256 256 0 0 1 1 256 1 1 0 0 2000 1 0 0 28 4 512 1 512 512 464 1 480 464 1984 0 1504 40 50 50 1 50 50 1 50 50 40 50 50 1 50 50 1 50 50 1 2880 2880 0 40 50 50 50 50 50 50 1 1 2880 2880 1 0 0 0 0 0 0 0 0 0 0 0 0 40 50 50 50 50 50 50 1 1 ";
    std::vector<size_t> expectWorkspaces = {16837152};
    RunNormalCaseStaticQuantDroppad(250, 2880, 8, 28, 6168, 1, 1, true, QUANT_MODE_STATIC, 1, ge::DT_FLOAT, {0, 256}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 1110100, expectTilingData, expectWorkspaces);
}

// 单核 + dynamci quant + active + gather + count + scale not None 1H float32  1020000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_13)
{
    string expectTilingData = "40 255 4096 1 0 256 256 1 0 1 0 256 1 1 0 0 255 0 1 0 0 1 255 1 255 255 255 1 255 255 1984 0 1504 37 7 3 1 7 7 1 3 3 37 7 3 1 7 7 1 3 3 2 2048 2048 0 37 7 7 7 3 3 3 1 1 4096 4096 1 0 0 0 0 0 0 0 0 0 0 0 0 37 7 7 7 3 3 3 1 1 ";
    std::vector<size_t> expectWorkspaces = {17443332};
    RunNormalCaseDynamicQuant1H(255, 4096, 1, 0, 500, 0, 1, true, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT, {0, 256}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 1020000, expectTilingData, expectWorkspaces);
}

// 单核 + dynamci quant + active + gather + scale None bfloat16  1020000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_14)
{
    string expectTilingData = "40 255 4096 1 180 192 12 1 0 0 0 256 1 0 1 1 255 0 0 0 0 1 255 1 255 255 255 1 255 255 1984 0 1504 37 7 3 1 7 7 1 3 3 37 7 3 1 7 7 1 3 3 2 2048 2048 0 37 7 7 7 3 3 3 1 1 4096 4096 1 0 0 0 0 0 0 0 0 0 0 0 0 37 7 7 7 3 3 3 1 1 ";
    std::vector<size_t> expectWorkspaces = {17442356};
    RunNormalCase(255, 4096, 1, 0, 500, 0, 1, false, QUANT_MODE_DYNAMIC, 0, ge::DT_BF16, {180, 192}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 1020000, expectTilingData, expectWorkspaces);
}

// 单核 + dynamci quant + active + scatter + cosum + scale not None EH  1021000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_15)
{
    string expectTilingData = "40 255 4096 1 0 100 100 1 1 1 0 256 0 1 0 1 255 0 2 0 0 1 255 1 255 255 255 1 255 255 1984 0 1504 37 7 3 1 7 7 1 3 3 37 7 3 1 7 7 1 3 3 2 2048 2048 0 37 7 7 7 3 3 3 1 1 4096 4096 1 0 0 0 0 0 0 0 0 0 0 0 0 37 7 7 7 3 3 3 1 1 ";
    std::vector<size_t> expectWorkspaces = {17442708};
    RunNormalCaseDynamicQuantEH(255, 4096, 1, 0, 500, 0, 0, true, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT, {0, 100}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1021000, expectTilingData, expectWorkspaces);
}

// 单核 + dynamci quant + dropless + scatter + count + scale None float16  1021000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_16)
{
    string expectTilingData = "40 255 4096 1 0 100 100 1 1 0 0 256 1 1 0 1 255 0 0 0 0 1 255 1 255 255 255 1 255 255 1984 0 1504 37 7 3 1 7 7 1 3 3 37 7 3 1 7 7 1 3 3 2 2048 2048 0 37 7 7 7 3 3 3 1 1 4096 4096 1 0 0 0 0 0 0 0 0 0 0 0 0 37 7 7 7 3 3 3 1 1 ";
    std::vector<size_t> expectWorkspaces = {17442708};
    RunNormalCaseDropless(255, 4096, 1, 0, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_FLOAT16, {0, 100}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1021000, expectTilingData, expectWorkspaces);
}

// 多核 + dynamci quant + dropless + gather + scale not None EH  1120000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_17)
{
    string expectTilingData = "40 160 96 1450 0 256 256 1 0 1 0 256 1 1 0 0 202000 0 2 0 0 40 5824 3 1984 1856 4864 3 1632 1600 1984 10 1504 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 2 3966 1834 2 3966 1834 1 96 96 0 40 5800 5800 5800 5800 5800 5800 1 1 96 96 1 0 0 0 0 0 0 0 0 0 0 0 0 40 5800 1450 1450 5800 1450 1450 4 4 ";
    std::vector<size_t> expectWorkspaces = {23292192};
    RunNormalCaseDynamicQuantEH(160, 96, 1450, 0, 202000, 0, 1, true, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT, {0, 256}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 1120000, expectTilingData, expectWorkspaces);
}

// 多核 + dynamci quant + droppless + gather + keyvalue + scale None  1120000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_18)
{
    string expectTilingData = "40 160 96 1450 180 192 12 1 0 0 0 256 2 1 0 1 232000 0 0 0 0 40 5824 3 1984 1856 4864 3 1632 1600 1984 10 1504 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 2 3966 1834 2 3966 1834 1 96 96 0 40 5800 5800 5800 5800 5800 5800 1 1 96 96 1 0 0 0 0 0 0 0 0 0 0 0 0 40 5800 1450 1450 5800 1450 1450 4 4 ";
    std::vector<size_t> expectWorkspaces = {23291216};
    RunNormalCaseKeyValue(160, 96, 1450, 0, 0, 0, 2, true, QUANT_MODE_DYNAMIC, 0, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 1120000, expectTilingData, expectWorkspaces);
}

// 多核 + dynamci quant + active + scatter + count + scale not None 1H  1121000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_19)
{
    string expectTilingData = "40 160 96 1450 0 100 100 1 1 1 0 256 1 1 0 1 202000 0 1 0 0 40 5824 3 1984 1856 4864 3 1632 1600 1984 10 1504 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 2 3966 1834 2 3966 1834 1 96 96 0 40 5800 5800 5800 5800 5800 5800 1 1 96 96 1 0 0 0 0 0 0 0 0 0 0 0 0 40 5800 1450 1450 5800 1450 1450 4 4 ";
    std::vector<size_t> expectWorkspaces = {23291568};
    RunNormalCaseDynamicQuant1H(160, 96, 1450, 0, 202000, 0, 1, true, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT, {0, 100}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1121000, expectTilingData, expectWorkspaces);
}

// 多核 + dynamci quant + active + scatter + count + scale None  1121000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_20)
{
    string expectTilingData = "40 160 96 1450 0 100 100 1 1 0 0 256 1 1 0 1 202000 0 0 0 0 40 5824 3 1984 1856 4864 3 1632 1600 1984 10 1504 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 2 3966 1834 2 3966 1834 1 96 96 0 40 5800 5800 5800 5800 5800 5800 1 1 96 96 1 0 0 0 0 0 0 0 0 0 0 0 0 40 5800 1450 1450 5800 1450 1450 4 4 ";
    std::vector<size_t> expectWorkspaces = {23291568};
    RunNormalCase(160, 96, 1450, 0, 202000, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_FLOAT, {0, 100}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1121000, expectTilingData, expectWorkspaces);
}

// 多核 + dynamci quant + active + scatter + scale None bfloat16  1121000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_21)
{
    string expectTilingData = "40 160 96 1450 0 100 100 1 1 0 0 256 1 1 0 1 202000 0 0 0 0 40 5824 3 1984 1856 4864 3 1632 1600 1984 10 1504 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 2 3966 1834 2 3966 1834 1 96 96 0 40 5800 5800 5800 5800 5800 5800 1 1 96 96 1 0 0 0 0 0 0 0 0 0 0 0 0 40 5800 1450 1450 5800 1450 1450 4 4 ";
    std::vector<size_t> expectWorkspaces = {23291568};
    RunNormalCase(160, 96, 1450, 0, 202000, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_BF16, {0, 100}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1121000, expectTilingData, expectWorkspaces);
}

// 多核 + dynamci quant + dropless + scatter + scale None float16  1121000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_22)
{
    string expectTilingData = "40 160 96 1450 0 100 100 1 1 0 0 256 1 1 0 1 232000 0 0 0 0 40 5824 3 1984 1856 4864 3 1632 1600 1984 10 1504 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 2 3966 1834 2 3966 1834 1 96 96 0 40 5800 5800 5800 5800 5800 5800 1 1 96 96 1 0 0 0 0 0 0 0 0 0 0 0 0 40 5800 1450 1450 5800 1450 1450 4 4 ";
    std::vector<size_t> expectWorkspaces = {23291568};
    RunNormalCaseDropless(160, 96, 1450, 0, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_FLOAT16, {0, 100}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1121000, expectTilingData, expectWorkspaces);
}

// 单核 + dynamci quant + droppad + gather + scale not None EH  1020100
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_23)
{
    string expectTilingData = "40 55 2880 8 0 256 256 1 0 1 0 256 1 0 0 0 440 1 2 0 8 1 440 1 440 440 440 1 440 440 1984 0 1504 40 11 11 1 11 11 1 11 11 40 11 11 1 11 11 1 11 11 1 2880 2880 0 0 0 0 0 0 0 0 0 0 0 0 0 40 11 11 11 11 11 11 1 1 2880 2880 1 40 11 11 11 11 11 11 1 1 ";
    std::vector<size_t> expectWorkspaces = {17254272};
    RunNormalCaseDynamicQuantDroppad(55, 2880, 8, 8, 440, 1, 1, false, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT, {0, 256}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 1020100, expectTilingData, expectWorkspaces);
}

// 多核 + dynamci quant + droppad + gather + count + scale None float16  1120100
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_24)
{
    string expectTilingData = "40 250 880 8 0 256 256 1 0 1 0 256 1 1 0 0 2000 1 2 0 191 4 512 1 512 512 464 1 480 464 1984 0 1504 40 50 50 1 50 50 1 50 50 40 50 50 1 50 50 1 50 50 1 880 880 0 0 0 0 0 0 0 0 0 0 0 0 0 40 50 50 50 50 50 50 1 1 880 880 1 40 50 50 50 50 50 50 1 1 ";
    std::vector<size_t> expectWorkspaces = {16977952};
    RunNormalCaseDynamicQuantDroppad(250, 880, 8, 191, 6128, 1, 1, true, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT16, {0, 256}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 1120100, expectTilingData, expectWorkspaces);
}

// full load
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_25)
{
    string expectTilingData = "40 1 7168 8 0 256 256 1 1 1 0 256 2 1 0 0 8 0 2 0 0 1 8 1 8 8 8 1 8 8 2016 0 1504 8 1 1 1 1 1 1 1 1 8 1 1 1 1 1 1 1 1 4 1792 1792 0 8 1 1 1 1 1 1 1 1 7168 7168 1 0 0 0 0 0 0 0 0 0 0 0 0 8 1 1 1 1 1 1 1 1 ";
    std::vector<size_t> expectWorkspaces = {17010432};
    RunNormalCaseFullload(1, 7168, 8, 0, 8, 0, 2, true, QUANT_MODE_DYNAMIC, 1, ge::DT_BF16, {0, 256}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 2000000, expectTilingData, expectWorkspaces);
}

// performance 单核gather
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_26)
{
    string expectTilingData = "40 384 7168 8 0 8 8 -1 1 0 0 256 1 1 0 1 3072 0 0 0 0 4 768 1 768 768 768 1 768 768 1984 0 1504 40 77 69 1 77 77 1 69 69 40 77 69 1 77 77 1 69 69 1 7168 7168 0 40 77 77 77 69 69 69 1 1 7168 7168 1 0 0 0 0 0 0 0 0 0 0 0 0 40 77 77 77 69 69 69 1 1 ";
    std::vector<size_t> expectWorkspaces = {16865856};
    RunNormalCase(384, 7168, 8, 0, 3072, 0, 1, true, QUANT_MODE_NONE, 1, ge::DT_FLOAT, {0, 8}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1201000, expectTilingData, expectWorkspaces);
}

// performance 多核gather
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_27)
{
    string expectTilingData = "40 4608 7168 8 0 8 8 -1 1 0 0 256 1 1 0 1 36864 0 0 0 0 40 928 1 928 928 672 1 672 672 1984 10 1504 40 922 906 1 922 922 1 906 906 40 116 906 1 922 922 1 906 906 3 2928 1312 0 40 922 922 922 906 906 906 1 1 7168 7168 1 0 0 0 0 0 0 0 0 0 0 0 0 40 922 922 922 906 906 906 1 1 ";
    std::vector<size_t> expectWorkspaces = {17812032};
    RunNormalCase(4608, 7168, 8, 0, 36864, 0, 1, true, QUANT_MODE_NONE, 1, ge::DT_FLOAT, {0, 8}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1301000, expectTilingData, expectWorkspaces);
}

// 多核排序
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_28)
{
    string expectTilingData = "40 250 7168 10 0 8 8 -1 1 0 0 256 1 1 0 1 2500 0 0 0 0 4 640 1 640 640 580 1 608 580 1984 0 1504 40 63 43 1 63 63 1 43 43 40 63 43 1 63 63 1 43 43 1 7168 7168 0 40 63 63 63 43 43 43 1 1 7168 7168 1 0 0 0 0 0 0 0 0 0 0 0 0 40 63 63 63 43 43 43 1 1 ";
    std::vector<size_t> expectWorkspaces = {16849840};
    RunNormalCase(250, 7168, 10, 0, 2500, 0, 1, true, QUANT_MODE_NONE, 1, ge::DT_FLOAT, {0, 8}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1101000, expectTilingData, expectWorkspaces);
}

// v3性能收编新增UT
// 非量化 fullload + scale None + count  2100000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_29)
{
    string expectTilingData = "40 1 83 27 180 192 12 -1 0 0 0 256 1 1 0 1 27 0 0 0 0 1 27 1 27 27 27 1 27 27 1984 0 1504 27 1 1 1 1 1 1 1 1 27 1 1 1 1 1 1 1 1 1 83 83 0 27 1 1 1 1 1 1 1 1 83 83 1 0 0 0 0 0 0 0 0 0 0 0 0 27 1 1 1 1 1 1 1 1 ";
    std::vector<size_t> expectWorkspaces = {16780612};
    RunNormalCase(1, 83, 27, 0, 27, 0, 1, true, QUANT_MODE_NONE, 0, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 2100000, expectTilingData, expectWorkspaces);
}

// 非量化 fullload + scale not None + consum  2100000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_30)
{
    string expectTilingData = "40 1 83 27 180 192 12 -1 0 1 0 256 0 1 0 1 27 0 0 0 0 1 27 1 27 27 27 1 27 27 1984 0 1504 27 1 1 1 1 1 1 1 1 27 1 1 1 1 1 1 1 1 1 83 83 0 27 1 1 1 1 1 1 1 1 83 83 1 0 0 0 0 0 0 0 0 0 0 0 0 27 1 1 1 1 1 1 1 1 ";
    std::vector<size_t> expectWorkspaces = {16780612};
    RunNormalCaseNoQuant(1, 83, 27, 0, 27, 0, 0, true, QUANT_MODE_NONE, 1, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 2100000, expectTilingData, expectWorkspaces);
}

// 非量化 fullload + scale None + key_value  2100000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_31)
{
    string expectTilingData = "40 1 83 27 10 22 12 -1 0 0 0 256 2 1 0 1 27 0 0 0 0 1 27 1 27 27 27 1 27 27 1984 0 1504 27 1 1 1 1 1 1 1 1 27 1 1 1 1 1 1 1 1 1 83 83 0 27 1 1 1 1 1 1 1 1 83 83 1 0 0 0 0 0 0 0 0 0 0 0 0 27 1 1 1 1 1 1 1 1 ";
    std::vector<size_t> expectWorkspaces = {16780612};
    RunNormalCaseKeyValue(1, 83, 27, 0, 27, 0, 2, true, QUANT_MODE_NONE, 0, ge::DT_FLOAT, {10, 22}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 2100000, expectTilingData, expectWorkspaces);
}

// 非量化 fullload + scale None + GatherFirst  2100000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_32)
{
    string expectTilingData = "40 100 2880 4 180 192 12 -1 0 0 0 256 1 1 1 1 400 0 0 0 0 1 400 1 400 400 400 1 400 400 1984 0 1504 40 10 10 1 10 10 1 10 10 40 10 10 1 10 10 1 10 10 1 2880 2880 0 40 10 10 10 10 10 10 1 1 2880 2880 1 0 0 0 0 0 0 0 0 0 0 0 0 40 10 10 10 10 10 10 1 1 ";
    std::vector<size_t> expectWorkspaces = {16791056};
    RunNormalCase(100, 2880, 4, 0, 400, 0, 1, true, QUANT_MODE_NONE, 0, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 2100000, expectTilingData, expectWorkspaces);
}

// 非量化 fullload + scale None + scatter  2100000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_33)
{
    string expectTilingData = "40 1 83 27 180 192 12 -1 1 0 0 256 1 1 0 1 27 0 0 0 0 1 27 1 27 27 27 1 27 27 1984 0 1504 27 1 1 1 1 1 1 1 1 27 1 1 1 1 1 1 1 1 1 83 83 0 27 1 1 1 1 1 1 1 1 83 83 1 0 0 0 0 0 0 0 0 0 0 0 0 27 1 1 1 1 1 1 1 1 ";
    std::vector<size_t> expectWorkspaces = {16780612};
    RunNormalCase(1, 83, 27, 0, 27, 0, 1, true, QUANT_MODE_NONE, 0, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 2100000, expectTilingData, expectWorkspaces);
}

// 静态量化 fullload + gather + count  2200000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_34)
{
    string expectTilingData = "40 1 83 27 180 192 12 0 0 1 1 256 1 1 0 1 27 0 0 0 0 1 27 1 27 27 27 1 27 27 1984 0 1504 27 1 1 1 1 1 1 1 1 27 1 1 1 1 1 1 1 1 1 83 83 0 27 1 1 1 1 1 1 1 1 83 83 1 0 0 0 0 0 0 0 0 0 0 0 0 27 1 1 1 1 1 1 1 1 ";
    std::vector<size_t> expectWorkspaces = {16780612};
    RunNormalCaseStaticQuant(1, 83, 27, 0, 27, 0, 1, true, QUANT_MODE_STATIC, 1, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 2200000, expectTilingData, expectWorkspaces);
}

// 静态量化 fullload + scatter + consum  2200000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_35)
{
    string expectTilingData = "40 1 83 27 180 192 12 0 1 1 1 256 0 1 0 1 27 0 0 0 0 1 27 1 27 27 27 1 27 27 1984 0 1504 27 1 1 1 1 1 1 1 1 27 1 1 1 1 1 1 1 1 1 83 83 0 27 1 1 1 1 1 1 1 1 83 83 1 0 0 0 0 0 0 0 0 0 0 0 0 27 1 1 1 1 1 1 1 1 ";
    std::vector<size_t> expectWorkspaces = {16780612};
    RunNormalCaseStaticQuant(1, 83, 27, 0, 27, 0, 0, true, QUANT_MODE_STATIC, 1, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 2200000, expectTilingData, expectWorkspaces);
}

// 动态量化 fullload + gather + consum + scale None  2300000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_36)
{
    string expectTilingData = "40 77 1024 4 0 256 256 1 0 0 0 256 0 1 0 0 308 0 0 0 0 1 308 1 308 308 308 1 308 308 1984 0 1504 39 8 4 1 8 8 1 4 4 39 8 4 1 8 8 1 4 4 1 1024 1024 0 39 8 8 8 4 4 4 1 1 1024 1024 1 0 0 0 0 0 0 0 0 0 0 0 0 39 8 8 8 4 4 4 1 1 ";
    std::vector<size_t> expectWorkspaces = {16953296};
    RunNormalCase(77, 1024, 4, 0, 308, 0, 0, true, QUANT_MODE_DYNAMIC, 0, ge::DT_FLOAT, {0, 256}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 2300000, expectTilingData, expectWorkspaces);
}

// 动态量化 EPFULLLOAD fullload + scatter + count + scale None  2310000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_37)
{
    string expectTilingData = "40 120 1024 4 180 192 12 1 1 0 0 256 1 1 1 1 480 0 0 0 0 1 480 1 480 480 480 1 480 480 1984 0 1504 40 12 12 1 12 12 1 12 12 40 12 12 1 12 12 1 12 12 1 1024 1024 0 40 12 12 12 12 12 12 1 1 1024 1024 1 0 0 0 0 0 0 0 0 0 0 0 0 40 12 12 12 12 12 12 1 1 ";
    std::vector<size_t> expectWorkspaces = {16957136};
    RunNormalCase(120, 1024, 4, 0, 480, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 2310000, expectTilingData, expectWorkspaces);
}

// 动态量化 SMOOTHTYPE 1H fullload + gather + consum  2301000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_38)
{
    string expectTilingData = "40 77 1024 4 0 256 256 1 0 1 0 256 0 1 0 0 308 0 1 0 0 1 308 1 308 308 308 1 308 308 1984 0 1504 39 8 4 1 8 8 1 4 4 39 8 4 1 8 8 1 4 4 1 1024 1024 0 39 8 8 8 4 4 4 1 1 1024 1024 1 0 0 0 0 0 0 0 0 0 0 0 0 39 8 8 8 4 4 4 1 1 ";
    std::vector<size_t> expectWorkspaces = {16953296};
    RunNormalCaseDynamicQuant1H(77, 1024, 4, 0, 308, 0, 0, true, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT, {0, 256}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 2301000, expectTilingData, expectWorkspaces);
}

// 动态量化 SMOOTHTYPE EH fullload + gather + consum  2302000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_39)
{
    string expectTilingData = "40 77 1024 4 0 256 256 1 0 1 0 256 0 1 0 0 308 0 2 0 0 1 308 1 308 308 308 1 308 308 1984 0 1504 39 8 4 1 8 8 1 4 4 39 8 4 1 8 8 1 4 4 1 1024 1024 0 39 8 8 8 4 4 4 1 1 1024 1024 1 0 0 0 0 0 0 0 0 0 0 0 0 39 8 8 8 4 4 4 1 1 ";
    std::vector<size_t> expectWorkspaces = {16953296};
    RunNormalCaseDynamicQuantEH(77, 1024, 4, 0, 308, 0, 0, true, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT, {0, 256}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 2302000, expectTilingData, expectWorkspaces);
}

// 动态量化 EPFULLLOAD  + SMOOTHTYPE 1H fullload + scatter + count  2311000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_40)
{
    string expectTilingData = "40 120 1024 4 180 192 12 1 1 1 0 256 1 1 1 1 480 0 1 0 0 1 480 1 480 480 480 1 480 480 1984 0 1504 40 12 12 1 12 12 1 12 12 40 12 12 1 12 12 1 12 12 1 1024 1024 0 40 12 12 12 12 12 12 1 1 1024 1024 1 0 0 0 0 0 0 0 0 0 0 0 0 40 12 12 12 12 12 12 1 1 ";
    std::vector<size_t> expectWorkspaces = {16957136};
    RunNormalCaseDynamicQuant1H(120, 1024, 4, 0, 480, 0, 1, true, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 2311000, expectTilingData, expectWorkspaces);
}

// 动态量化 EPFULLLOAD  + SMOOTHTYPE EH fullload + scatter + count  2312000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_41)
{
    string expectTilingData = "40 120 1024 4 180 192 12 1 1 1 0 256 1 1 1 1 480 0 2 0 0 1 480 1 480 480 480 1 480 480 1984 0 1504 40 12 12 1 12 12 1 12 12 40 12 12 1 12 12 1 12 12 1 1024 1024 0 40 12 12 12 12 12 12 1 1 1024 1024 1 0 0 0 0 0 0 0 0 0 0 0 0 40 12 12 12 12 12 12 1 1 ";
    std::vector<size_t> expectWorkspaces = {16957136};
    RunNormalCaseDynamicQuantEH(120, 1024, 4, 0, 480, 0, 1, true, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 2312000, expectTilingData, expectWorkspaces);
}

// 空张量路径 3000000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_42)
{
string expectTilingData = "40 0 256 1 0 256 256 -1 1 0 0 256 1 1 0 0 0 0 0 256 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
std::vector<size_t> expectWorkspaces = {16777216};
RunNormalCaseDropless(0, 256, 1, 0, 0, 0, 1, true, QUANT_MODE_NONE, 0, ge::DT_FLOAT, {0, 256}, ROW_IDX_TYPE_DROPLESS,
ge::GRAPH_SUCCESS, 3000000, expectTilingData, expectWorkspaces);
}

// counting sort 多核gather非fullload 1300000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_43)
{
    string expectTilingData = "40 1000 2880 4 0 256 256 -1 0 1 0 256 1 0 0 0 4000 0 0 0 0 4 992 1 992 992 1024 1 1024 1024 1984 0 1504 40 100 100 1 100 100 1 100 100 40 100 100 1 100 100 1 100 100 1 2880 2880 0 40 100 100 100 100 100 100 1 1 2880 2880 1 0 0 0 0 0 0 0 0 0 0 0 0 40 100 100 100 100 100 100 1 1 ";
    std::vector<size_t> expectWorkspaces = {16892832};
    RunNormalCaseNoQuant(1000, 2880, 4, 0, 4000, 0, 1, false, QUANT_MODE_NONE, 1, ge::DT_FLOAT, {0, 256}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 1100000, expectTilingData, expectWorkspaces);
}

// ========== ComputeCountingSortTiling UT测试用例 ==========

// 计数排序场景测试辅助函数
// 特点: ep_=1(设置了expert_range), quantMode=-1(非量化), dropPadMode=0(DropLess),
// expertTokensNumType=1(COUNT), K=8, 有scale输入
void RunNormalCaseCountingSort(
    int64_t N, int64_t H, int64_t K, int64_t C, int64_t activeNum, int64_t dropPadMode, int64_t countFlag,
    bool tokenFlag, int64_t quantMode, ge::DataType xDataType, std::vector<int64_t> activeExpertRange,
    int64_t rowIdxType, ge::graphStatus result, int64_t expectTilingKey, string expectTilingData,
    vector<size_t> expectWorkspaces)
{
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t expert_num = EXPERT_NUM;
    int64_t E = activeExpertRange[1] - activeExpertRange[0]; // 活跃专家数

    // 计数排序场景: 输出expertTokensCount shape为[E]
    // expandedX shape为[N*K, H] (DropLess模式)
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV3",
        {
            {{{N, H}, {N, H}}, xDataType, ge::FORMAT_ND},
            {{{N, K}, {N, K}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{N}, {N}}, ge::DT_FLOAT, ge::FORMAT_ND}, // scale输入
        },
        {{{{N * K, H}, {N * K, H}}, xDataType, ge::FORMAT_ND},
        {{{N * K}, {N * K}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{N * K}, {N * K}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(C)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
            {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(countFlag)},
            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tokenFlag)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(activeExpertRange)},
            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rowIdxType)},
        },
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);

}

// 计数排序 - FullLoad路径
// 场景: T=1536, D=2560, K=8, E=256, LE=16
// drop_pad_mode=0, expert_tokens_num_type=1, expert_tokens_num_flag=true
// quant_mode=-1, row_idx_type=0, dtype=int8
// 特点: actualExpertNum=16 <= 128, UB足够容纳所有数据, countingSortFullLoad=1
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_counting_sort_fullload)
{
    // FullLoad路径: UB足够容纳所有数据
    // N=1536, H=2560, K=8, expert_range=[0,16]
    // 满足计数排序条件: N>=256, ep_=1, quantMode=-1, dropPadMode=0,
    // expertTokensNumType=1, actualExpertNum=16<=128, K=8
    // expectTilingKey和expectTilingData需要根据实际Tiling计算确定
    // FullLoad workspace = filterNeedCoreNum * expertCountStride * 4B
    // filterNeedCoreNum = CeilDiv(1536, CeilDiv(1536, 40)) = 40
    // expertCountStride = CeilAlign(16, 8) = 16
    // workspace = 40 * 16 * 4 = 2560 bytes (约2.5KB)
    string expectTilingData = ""; // 运行后获取实际值
    std::vector<size_t> expectWorkspaces = {17123936};
    RunNormalCaseCountingSort(
        1536, 2560, 8, 0, 0, 0, 1, true, QUANT_MODE_NONE, ge::DT_INT8, {0, 16}, ROW_IDX_TYPE_DROPPAD, ge::GRAPH_SUCCESS,
        1300000, expectTilingData, expectWorkspaces);
}

// 计数排序 - CutOriginT路径 (非全载)
// 场景: T=2560, D=5120, K=8, E=256, LE=3
// drop_pad_mode=0, expert_tokens_num_type=1, expert_tokens_num_flag=true
// quant_mode=-1, row_idx_type=0, dtype=int8
// 特点: actualExpertNum=3 <= 128, 但UB不足以一次性容纳所有数据, countingSortFullLoad=0
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_counting_sort_cutorigint)
{
    // CutOriginT路径: UB不足，需要分块处理
    // N=2560, H=5120, K=8, expert_range=[0,3]
    // 满足计数排序条件: N>=256, ep_=1, quantMode=-1, dropPadMode=0,
    // expertTokensNumType=1, actualExpertNum=3<=128, K=8
    // 但H=5120较大，UB可能不足以FullLoad
    // CutOriginT workspace = pairsWorkspace + expertCountWorkspace
    // pairsWorkspace = filterNeedCoreNum * pairsPerCore * 4B
    // expertCountWorkspace = filterNeedCoreNum * expertCountStride * 4B
    string expectTilingData = ""; // 运行后获取实际值
    std::vector<size_t> expectWorkspaces = {17353260};
    RunNormalCaseCountingSort(
        2560, 5120, 8, 0, 0, 0, 1, true, QUANT_MODE_NONE, ge::DT_INT8, {0, 3}, ROW_IDX_TYPE_DROPPAD, ge::GRAPH_SUCCESS,
        1300000, expectTilingData, expectWorkspaces);
}

// 计数排序异常场景测试辅助函数
// 用于测试期望返回GRAPH_FAILED的异常场景
void RunExceptionCaseCountingSort(
int64_t N, int64_t H, int64_t K, int64_t C, int64_t activeNum, int64_t dropPadMode, int64_t countFlag,
bool tokenFlag, int64_t quantMode, ge::DataType xDataType, std::vector<int64_t> activeExpertRange,
int64_t rowIdxType, int64_t expertNum, ge::graphStatus expectedResult)
{
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t E = activeExpertRange.size() >= 2 ? (activeExpertRange[1] - activeExpertRange[0]) : expertNum;

    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV3",
        {
            {{{N, H}, {N, H}}, xDataType, ge::FORMAT_ND},
            {{{N, K}, {N, K}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{N}, {N}}, ge::DT_FLOAT, ge::FORMAT_ND}, // scale输入
        },
        {{{{N * K, H}, {N * K, H}}, xDataType, ge::FORMAT_ND},
        {{{N * K}, {N * K}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{N * K}, {N * K}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(C)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertNum)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
            {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(countFlag)},
            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tokenFlag)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(activeExpertRange)},
            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rowIdxType)},
        },
        &compileInfo);

    ExecuteTestCase(tilingContextPara, expectedResult, 0, "", {});

}

// ========== ComputeCountingSortTiling 异常场景UT测试用例 ==========

// 异常场景1: 空Tensor场景 (n_=0)
// 处理方式: 校验isEmptyTensor_标志, 设置workspaceSize为固定小值, BlockDim为1, 返回GRAPH_SUCCESS
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_counting_sort_empty_tensor_n_zero)
{
    // N=0时, 应返回GRAPH_SUCCESS, 设置isEmptyTensor_=true
    // workspaceSize设置为固定小值(16MB对齐块), BlockDim设置为1
    RunNormalCaseCountingSort(
        0, 2560, 8, 0, 0, 0, 1, true, QUANT_MODE_NONE, ge::DT_INT8, {0, 16}, ROW_IDX_TYPE_DROPPAD, ge::GRAPH_SUCCESS,
        3000000, "", {});
}

// 异常场景2: Input X维度非法 - xShape非2维
// 处理方式: 校验xShape的DimNum是否为2, 若不满足返回GRAPH_FAILED
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_counting_sort_input_x_dim_not_2d)
{
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};

    // xShape为1维, 非法
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV3",
        {
            {{{2560}, {2560}}, ge::DT_INT8, ge::FORMAT_ND}, // 1维,非法
            {{{1536, 8}, {1536, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1536}, {1536}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{{{12288, 2560}, {12288, 2560}}, ge::DT_INT8, ge::FORMAT_ND},
        {{{12288}, {12288}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{16}, {16}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{12288}, {12288}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(EXPERT_NUM)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(QUANT_MODE_NONE)},
            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 16})},
            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ROW_IDX_TYPE_DROPPAD)},
        },
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});

}

// 异常场景3: Input X维度非法 - 行数不匹配
// 处理方式: 校验xShape[0]是否等于expertIdxShape[0], 若不满足返回GRAPH_FAILED
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_counting_sort_input_rows_mismatch)
{
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};

    // xShape[0]=100, expertIdxShape[0]=200, 行数不匹配
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV3",
        {
            {{{100, 2560}, {100, 2560}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{200, 8}, {200, 8}}, ge::DT_INT32, ge::FORMAT_ND}, // 行数200!=100,非法
            {{{100}, {100}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{{{800, 2560}, {800, 2560}}, ge::DT_INT8, ge::FORMAT_ND},
        {{{800}, {800}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{16}, {16}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{800}, {800}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(EXPERT_NUM)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(QUANT_MODE_NONE)},
            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 16})},
            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ROW_IDX_TYPE_DROPPAD)},
        },
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});

}

// 异常场景4: Expert Num非法 - expert_num <= 0
// 处理方式: 校验expertNum_ > 0, 若不满足返回GRAPH_FAILED
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_counting_sort_expert_num_zero)
{
    RunExceptionCaseCountingSort(
        1536, 2560, 8, 0, 0, 0, 1, true, QUANT_MODE_NONE, ge::DT_INT8, {0, 16}, ROW_IDX_TYPE_DROPPAD, 0,
        ge::GRAPH_FAILED);
}

// 异常场景5: Expert Num非法 - expert_range无效 (expertStart >= expertEnd)
// 处理方式: 校验expertStart_ < expertEnd_, 若不满足返回GRAPH_FAILED
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_counting_sort_expert_range_invalid)
{
    RunExceptionCaseCountingSort(
        1536, 2560, 8, 0, 0, 0, 1, true, QUANT_MODE_NONE, ge::DT_INT8, {100, 50}, // expertStart=100 > expertEnd=50,非法
        ROW_IDX_TYPE_DROPPAD, EXPERT_NUM, ge::GRAPH_FAILED);
}

// 异常场景6: Expert Num非法 - expert_range Size非法 (非0或2)
// 处理方式: 校验expert_range的Size是否为0或2, 若不满足返回GRAPH_FAILED
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_counting_sort_expert_range_size_invalid)
{
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};

    // expert_range Size=3, 非法
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV3",
        {
            {{{1536, 2560}, {1536, 2560}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1536, 8}, {1536, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1536}, {1536}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{{{12288, 2560}, {12288, 2560}}, ge::DT_INT8, ge::FORMAT_ND},
        {{{12288}, {12288}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{16}, {16}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{12288}, {12288}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(EXPERT_NUM)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(QUANT_MODE_NONE)},
            {"acive_expert_range",
            Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 16, 32})}, // Size=3,非法
            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ROW_IDX_TYPE_DROPPAD)},
        },
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});

}

// 异常场景7: DropPad模式配置冲突 - DropPad下RowIdxType非GATHER
// 处理方式: 当dropPadMode_==1时, 强制要求rowIdxTytpe_==0(GATHER), 若不满足返回GRAPH_FAILED
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_counting_sort_droppad_rowidxtype_scatter)
{
    // dropPadMode=1, rowIdxType=1(SCATTER), 非法
    RunExceptionCaseCountingSort(
        1536, 2560, 8, 50, 0, 1, 1, true, QUANT_MODE_NONE, ge::DT_INT8, {0, EXPERT_NUM}, ROW_IDX_TYPE_DROPLESS,
        EXPERT_NUM, ge::GRAPH_FAILED);
}

// 异常场景8: DropPad模式配置冲突 - DropPad下expert_range非全量
// 处理方式: 当dropPadMode_==1时, 强制要求expertStart_==0且expertEnd_==expertNum_, 若不满足返回GRAPH_FAILED
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_counting_sort_droppad_range_not_full)
{
    // dropPadMode=1, expert_range=[0,100]非全量, 非法
    RunExceptionCaseCountingSort(
        1536, 2560, 8, 50, 0, 1, 1, true, QUANT_MODE_NONE, ge::DT_INT8, {0, 100}, ROW_IDX_TYPE_DROPPAD, EXPERT_NUM,
        ge::GRAPH_FAILED);
}

// 异常场景9: Expert Capacity非法 - DropPad模式下Capacity <= 0
// 处理方式: 当dropPadMode_==1时, 校验0 < expertCapacity_ <= n_, 若不满足返回GRAPH_FAILED
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_counting_sort_droppad_capacity_zero)
{
    // dropPadMode=1, expertCapacity=0, 非法
    RunExceptionCaseCountingSort(
        1536, 2560, 8, 0, 0, 1, 1, true, QUANT_MODE_NONE, ge::DT_INT8, {0, EXPERT_NUM}, ROW_IDX_TYPE_DROPPAD,
        EXPERT_NUM, ge::GRAPH_FAILED);
}

// 异常场景10: Expert Capacity非法 - DropPad模式下Capacity > N
// 处理方式: 当dropPadMode_==1时, 校验expertCapacity_ <= n_, 若不满足返回GRAPH_FAILED
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_counting_sort_droppad_capacity_exceed_n)
{
// dropPadMode=1, expertCapacity=2000 > N=1536, 非法
    RunExceptionCaseCountingSort(
        1536, 2560, 8, 2000, 0, 1, 1, true, QUANT_MODE_NONE, ge::DT_INT8, {0, EXPERT_NUM}, ROW_IDX_TYPE_DROPPAD,
        EXPERT_NUM, ge::GRAPH_FAILED);
}

// 异常场景11: Quant Mode数据格式不匹配 - INT8输入不支持量化
// 处理方式: 校验input_X为INT8时quant_mode必须为UN_QUANT, 若不满足返回GRAPH_FAILED
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_counting_sort_int8_with_quant)
{
    // input_X为INT8, quant_mode=STATIC_QUANT, 非法
    RunExceptionCaseCountingSort(
        1536, 2560, 8, 0, 0, 0, 1, true, QUANT_MODE_STATIC, ge::DT_INT8, {0, 16}, ROW_IDX_TYPE_DROPPAD, EXPERT_NUM,
        ge::GRAPH_FAILED);
}

// 异常场景12: Quant Mode数据格式不匹配 - 静态量化缺Scale/Offset
// 处理方式: 校验quant_mode==STATIC_QUANT时Scale和Offset输入不能为空, 若不满足返回GRAPH_FAILED
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_counting_sort_static_quant_missing_scale)
{
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};

    // STATIC_QUANT模式但缺少scale和offset输入
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV3",
        {
            {{{1536, 2560}, {1536, 2560}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1536, 8}, {1536, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            // 缺少scale和offset输入
        },
        {{{{12288, 2560}, {12288, 2560}}, ge::DT_INT8, ge::FORMAT_ND},
        {{{12288}, {12288}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{16}, {16}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{12288}, {12288}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(EXPERT_NUM)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"quant_mode",
            Ops::Transformer::AnyValue::CreateFrom<int64_t>(QUANT_MODE_STATIC)}, // STATIC_QUANT但缺少scale,非法
            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 16})},
            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ROW_IDX_TYPE_DROPPAD)},
        },
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});

}

// 异常场景13: Output Shape校验失败 - expandedX维度不匹配
// 处理方式: 校验expandedX的DimNum和Dim0/Dim1是否符合预期, 若不满足返回GRAPH_FAILED
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_counting_sort_output_expandedx_dim_mismatch)
{
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};

    // expandedX为3维, 但DropLess模式期望2维, 非法
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV3",
        {
            {{{1536, 2560}, {1536, 2560}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1536, 8}, {1536, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1536}, {1536}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{{{EXPERT_NUM, 50, 2560}, {EXPERT_NUM, 50, 2560}}, ge::DT_INT8, ge::FORMAT_ND}, // 3维,DropLess模式期望2维,非法
        {{{12288}, {12288}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{16}, {16}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{EXPERT_NUM * 50}, {EXPERT_NUM * 50}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(50)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(EXPERT_NUM)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, // DropLess模式
            {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(QUANT_MODE_NONE)},
            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 16})},
            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ROW_IDX_TYPE_DROPPAD)},
        },
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});

}

// 异常场景14: Output Shape校验失败 - expertTokensCount维度不匹配
// 处理方式: 校验expertTokensCount的维度是否符合expertTokensNumType, 若不满足返回GRAPH_FAILED
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_counting_sort_output_experttokenscount_dim_mismatch)
{
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};

    // expertTokensNumType=KEY_VALUE(2), 但expertTokensCount为1维, 非法
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV3",
        {
            {{{1536, 2560}, {1536, 2560}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1536, 8}, {1536, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1536}, {1536}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{{{12288, 2560}, {12288, 2560}}, ge::DT_INT8, ge::FORMAT_ND},
        {{{12288}, {12288}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{16}, {16}}, ge::DT_INT64, ge::FORMAT_ND}, // 1维,KEY_VALUE模式期望2维,非法
        {{{12288}, {12288}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(EXPERT_NUM)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}, // KEY_VALUE模式
            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(QUANT_MODE_NONE)},
            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 16})},
            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ROW_IDX_TYPE_DROPPAD)},
        },
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});

}

// ========== 计数排序泛化功能测试用例 ==========

// 用例1: 计数排序 - FullLoad路径（DropLess模式, COUNT类型）
// actualExpertNum=16 <= 128, UB足够容纳所有数据, countingSortFullLoad=1
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_counting_sort_fullload_dropless)
{
    RunNormalCaseCountingSort(
        1536, 2560, 8, 0, 0, 0, 1, true, QUANT_MODE_NONE, ge::DT_INT8, {0, 16}, ROW_IDX_TYPE_DROPPAD, ge::GRAPH_SUCCESS,
        1300000, "", {});
}

// 用例2: 计数排序 - FullLoad路径（DropPad模式）
// dropPadMode=1, expertCapacity=50, UB足够容纳, countingSortFullLoad=1
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_counting_sort_fullload_droppad)
{
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t expert_num = EXPERT_NUM;

    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV3",
        {
            {{{1536, 2560}, {1536, 2560}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1536, 8}, {1536, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1536}, {1536}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{{{expert_num, 50, 2560}, {expert_num, 50, 2560}}, ge::DT_INT8, ge::FORMAT_ND},
        {{{12288}, {12288}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{expert_num}, {expert_num}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{expert_num * 50}, {expert_num * 50}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(50)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, // DropPad模式
            {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(QUANT_MODE_NONE)},
            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, (int64_t)expert_num})},
            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ROW_IDX_TYPE_DROPPAD)},
        },
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1300000, "", {});

}


// 用例3: 触发 ep_==0 路径（expert_range=[0, expertNum] 全量专家）
// 覆盖 FilterAndCountChunked 中 ep_==0 的快速路径（不过滤）
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_counting_sort_ep_zero)
{
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t expert_num = EXPERT_NUM;

    // expert_range=[0, 256] 触发 ep_=0
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV3",
        {
            {{{1536, 2560}, {1536, 2560}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1536, 8}, {1536, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1536}, {1536}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{{{1536 * 8, 2560}, {1536 * 8, 2560}}, ge::DT_INT8, ge::FORMAT_ND},
        {{{1536 * 8}, {1536 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{expert_num}, {expert_num}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{1536 * 8}, {1536 * 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(QUANT_MODE_NONE)},
            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, (int64_t)expert_num})},
            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ROW_IDX_TYPE_DROPPAD)},
        },
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1100000, "", {});

}

// 用例4: 触发 shortPath_ 路径（coreEntries_ <= 2048）
// coreEntries_ = filterPerCoreTokens * k_ <= 2048 时走 shortPath
// 选择 T=128, K=8, filterPerCoreTokens = ceil(128/40) = 4, coreEntries_ = 32 < 2048
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_counting_sort_short_path)
{
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t expert_num = EXPERT_NUM;

    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV3",
        {
            {{{128, 2560}, {128, 2560}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{{{128 * 8, 2560}, {128 * 8, 2560}}, ge::DT_INT8, ge::FORMAT_ND},
        {{{128 * 8}, {128 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{16}, {16}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{128 * 8}, {128 * 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(QUANT_MODE_NONE)},
            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 16})},
            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ROW_IDX_TYPE_DROPPAD)},
        },
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 2100000, "", {});

}

// 用例5: 触发 countingSortFullLoad_==0 (cutOriginT路径) - 大batch_size触发
// 当 T 大且 H 大时，UB不足以 FullLoad，走 cutOriginT 路径
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_counting_sort_full_load_zero)
{
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t expert_num = EXPERT_NUM;

    // T=4096, H=8192, K=8, LE=16
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV3",
        {
            {{{4096, 8192}, {4096, 8192}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{4096, 8}, {4096, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4096}, {4096}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{{{4096 * 8, 8192}, {4096 * 8, 8192}}, ge::DT_INT8, ge::FORMAT_ND},
        {{{4096 * 8}, {4096 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{16}, {16}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{4096 * 8}, {4096 * 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(QUANT_MODE_NONE)},
            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 16})},
            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ROW_IDX_TYPE_DROPPAD)},
        },
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1300000, "", {});

}

// 用例6: 触发 isInputScale_ 路径（非量化 + 输入 scale，cutOriginT路径）
// 覆盖 GatherAndWriteChunked 中 isInputScale_ 分支（unquant + inputScale scaleBuf）
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_counting_sort_input_scale)
{
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t expert_num = EXPERT_NUM;

    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV3",
        {
            {{{4096, 8192}, {4096, 8192}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4096, 8}, {4096, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4096}, {4096}}, ge::DT_FLOAT, ge::FORMAT_ND}, // 输入scale
        },
        {{{{4096 * 8, 8192}, {4096 * 8, 8192}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{4096 * 8}, {4096 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{16}, {16}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{4096 * 8}, {4096 * 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(QUANT_MODE_NONE)},
            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 16})},
            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ROW_IDX_TYPE_DROPPAD)},
        },
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1300000, "", {});

}

// 用例7: 触发 dropPad + cutOriginT 路径（drop_pad=1 + 大H）
// 覆盖 GatherAndWriteDroppad 函数的执行路径
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_counting_sort_droppad_large_h)
{
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t expert_num = EXPERT_NUM;

    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV3",
        {
            {{{4096, 8192}, {4096, 8192}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{4096, 8}, {4096, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4096}, {4096}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{{{expert_num, 50, 8192}, {expert_num, 50, 8192}}, ge::DT_INT8, ge::FORMAT_ND},
        {{{4096 * 8}, {4096 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{expert_num}, {expert_num}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{expert_num * 50}, {expert_num * 50}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(50)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, // DropPad
            {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(QUANT_MODE_NONE)},
            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, (int64_t)expert_num})},
            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ROW_IDX_TYPE_DROPPAD)},
        },
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1300000, "", {});

}

// 用例8: 触发 cutOriginT 路径 - 静态量化 + SCATTER
// 覆盖 StaticQuantCompute + SCATTER分支
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_counting_sort_static_quant_scatter)
{
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t expert_num = EXPERT_NUM;

    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV3",
        {
            {{{4096, 8192}, {4096, 8192}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4096, 8}, {4096, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},  // scale
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},  // offset
        },
        {{{{4096 * 8, 8192}, {4096 * 8, 8192}}, ge::DT_INT8, ge::FORMAT_ND},
        {{{4096 * 8}, {4096 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{16}, {16}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{4096 * 8}, {4096 * 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(QUANT_MODE_STATIC)},
            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 16})},
            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ROW_IDX_TYPE_DROPLESS)}, // SCATTER
        },
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1111000, "", {});

}

// 用例9: 触发 cutOriginT 路径 - 动态量化 + 多核
// 覆盖 DynamicQuantComputeAbsMax + DynamicQuantApplyScale
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_counting_sort_dynamic_quant_multi_core)
{
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t expert_num = EXPERT_NUM;

    // 大batch + 大hidden触发多核cutOriginT
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV3",
        {
            {{{4096, 8192}, {4096, 8192}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4096, 8}, {4096, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16, 8192}, {16, 8192}}, ge::DT_FLOAT, ge::FORMAT_ND}, // EH scale
        },
        {{{{4096 * 8, 8192}, {4096 * 8, 8192}}, ge::DT_INT8, ge::FORMAT_ND},
        {{{4096 * 8}, {4096 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{16}, {16}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{4096 * 8}, {4096 * 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(QUANT_MODE_DYNAMIC)},
            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 16})},
            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ROW_IDX_TYPE_DROPPAD)},
        },
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1120000, "", {});

}

// 用例10: 触发 dropPad + 多核cutOriginT路径 + 动态量化
// 覆盖 GatherAndWriteDroppad + DynamicQuant
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_counting_sort_droppad_dynamic_quant)
{
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t expert_num = EXPERT_NUM;

    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV3",
        {
            {{{4096, 8192}, {4096, 8192}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4096, 8}, {4096, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 8192}, {1, 8192}}, ge::DT_FLOAT, ge::FORMAT_ND}, // EH scale (global)
        },
        {{{{expert_num, 50, 8192}, {expert_num, 50, 8192}}, ge::DT_INT8, ge::FORMAT_ND},
        {{{4096 * 8}, {4096 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{expert_num}, {expert_num}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{expert_num * 50}, {expert_num * 50}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(50)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, // DropPad
            {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(QUANT_MODE_DYNAMIC)},
            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, (int64_t)expert_num})},
            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ROW_IDX_TYPE_DROPPAD)},
        },
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1120100, "", {});

}

// 用例11: 触发 colsLoops_>1 路径 - 大H触发多列循环
// 覆盖 GatherAndWriteChunked 中 colsLoops_>1 的 serial fallback 分支
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_counting_sort_cols_loops_gt_one)
{
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t expert_num = EXPERT_NUM;

    // H=16384 较大，触发 colsLoops_>1
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV3",
        {
            {{{1024, 16384}, {1024, 16384}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1024, 8}, {1024, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{{{1024 * 8, 16384}, {1024 * 8, 16384}}, ge::DT_INT8, ge::FORMAT_ND},
        {{{1024 * 8}, {1024 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{16}, {16}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{1024 * 8}, {1024 * 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(QUANT_MODE_NONE)},
            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 16})},
            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ROW_IDX_TYPE_DROPPAD)},
        },
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1300000, "", {});

}

// 用例12: 计数排序 - FullLoad路径（KeyValue类型）
// expert_tokens_num_type=2, 输出为2维key/value pair, UB足够容纳, countingSortFullLoad=1
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_counting_sort_fullload_keyvalue)
{
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t expert_num = EXPERT_NUM;

    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV3",
        {
            {{{1536, 2560}, {1536, 2560}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1536, 8}, {1536, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1536}, {1536}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{{{12288, 2560}, {12288, 2560}}, ge::DT_INT8, ge::FORMAT_ND},
        {{{12288}, {12288}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{expert_num, 2}, {expert_num, 2}}, ge::DT_INT64, ge::FORMAT_ND}, // 2维KeyValue
        {{{12288}, {12288}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}, // KeyValue类型
            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(QUANT_MODE_NONE)},
            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 16})},
            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ROW_IDX_TYPE_DROPPAD)},
        },
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1300000, "", {});

}

// 用例13: 计数排序 - FullLoad路径（CONSUM类型）
// expert_tokens_num_type=0, UB足够容纳, countingSortFullLoad=1
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_counting_sort_fullload_consum)
{
    RunNormalCaseCountingSort(
        1536, 2560, 8, 0, 0, 0, 0, true, QUANT_MODE_NONE, ge::DT_INT8, {0, 16}, ROW_IDX_TYPE_DROPPAD, ge::GRAPH_SUCCESS,
        1300000, "", {});
}

// 用例14: 计数排序 - FullLoad路径（静态量化）
// quant_mode=0, 包含scale和offset输入, UB足够容纳, countingSortFullLoad=1
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_counting_sort_fullload_static_quant)
{
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};

    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV3",
        {
            {{{1536, 2560}, {1536, 2560}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1536, 8}, {1536, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},   // scale
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},   // offset
        },
        {{{{12288, 2560}, {12288, 2560}}, ge::DT_INT8, ge::FORMAT_ND},
        {{{12288}, {12288}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{16}, {16}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{12288}, {12288}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(EXPERT_NUM)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(QUANT_MODE_STATIC)}, // 静态量化
            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 16})},
            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ROW_IDX_TYPE_DROPPAD)},
        },
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1110000, "", {});

}



// 用例15: 计数排序 - T=256边界值（小batch）
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_counting_sort_t_256)
{
    RunNormalCaseCountingSort(
        256, 1024, 8, 0, 0, 0, 1, true, QUANT_MODE_NONE, ge::DT_INT8, {0, 32}, ROW_IDX_TYPE_DROPPAD, ge::GRAPH_SUCCESS,
        1200000, "", {});
}

// 用例16: 计数排序 - row_idx_type=1（SCATTER索引）
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_counting_sort_scatter)
{
    RunNormalCaseCountingSort(
        1536, 2560, 8, 0, 0, 0, 1, true, QUANT_MODE_NONE, ge::DT_INT8, {0, 16}, ROW_IDX_TYPE_DROPLESS, ge::GRAPH_SUCCESS,
        1201000, "", {});
}

// 用例17: 计数排序 - 动态量化（quant_mode=1）
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_counting_sort_dynamic_quant)
{
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};

    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV3",
        {
            {{{1536, 2560}, {1536, 2560}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1536, 8}, {1536, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16, 2560}, {16, 2560}}, ge::DT_FLOAT, ge::FORMAT_ND}, // EH scale
        },
        {{{{12288, 2560}, {12288, 2560}}, ge::DT_INT8, ge::FORMAT_ND},
        {{{12288}, {12288}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{16}, {16}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{12288}, {12288}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(EXPERT_NUM)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(QUANT_MODE_DYNAMIC)}, // 动态量化
            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 16})},
            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ROW_IDX_TYPE_DROPPAD)},
        },
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1120000, "", {});

}

// 用例18: 计数排序 - active_num>0（有激活token过滤）
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_counting_sort_active_num_positive)
{
    RunNormalCaseCountingSort(
        1536, 2560, 8, 0, 20000, 0, 1, true, QUANT_MODE_NONE, ge::DT_INT8, {0, 16}, ROW_IDX_TYPE_DROPPAD, ge::GRAPH_SUCCESS,
        1300000, "", {});
}

// 用例19: 计数排序 - LE泛化（LE=128，最大限制）
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_counting_sort_le_128)
{
    RunNormalCaseCountingSort(
        1536, 2560, 8, 0, 0, 0, 1, true, QUANT_MODE_NONE, ge::DT_INT8, {0, 128}, ROW_IDX_TYPE_DROPPAD, ge::GRAPH_SUCCESS,
        1300000, "", {});
}

// 用例20: 计数排序 - LE泛化（LE=1，最小专家数）
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_counting_sort_le_1)
{
    RunNormalCaseCountingSort(
        1536, 2560, 8, 0, 0, 0, 1, true, QUANT_MODE_NONE, ge::DT_INT8, {0, 1}, ROW_IDX_TYPE_DROPPAD, ge::GRAPH_SUCCESS,
        1300000, "", {});
}

// 用例21: 计数排序 - LE泛化（LE=64，中间值）
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_counting_sort_le_64)
{
    RunNormalCaseCountingSort(
        1536, 2560, 8, 0, 0, 0, 1, true, QUANT_MODE_NONE, ge::DT_INT8, {64, 128}, ROW_IDX_TYPE_DROPPAD, ge::GRAPH_SUCCESS,
        1300000, "", {});
}

// 用例22: 计数排序 - DropPad模式 + active_num>0
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_counting_sort_droppad_active)
{
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t expert_num = EXPERT_NUM;

    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV3",
        {
            {{{1536, 2560}, {1536, 2560}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1536, 8}, {1536, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1536}, {1536}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{{{expert_num, 100, 2560}, {expert_num, 100, 2560}}, ge::DT_INT8, ge::FORMAT_ND},
        {{{12288}, {12288}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{expert_num}, {expert_num}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{expert_num * 100}, {expert_num * 100}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5000)}, // active_num>0
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(100)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, // DropPad模式
            {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(QUANT_MODE_NONE)},
            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, (int64_t)expert_num})},
            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ROW_IDX_TYPE_DROPPAD)},
        },
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1300000, "", {});

}

// 用例23: 计数排序 - 动态量化 + SCATTER索引
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_counting_sort_dynamic_quant_scatter)
{
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};

    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV3",
        {
            {{{1536, 2560}, {1536, 2560}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1536, 8}, {1536, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16, 2560}, {16, 2560}}, ge::DT_FLOAT, ge::FORMAT_ND}, // EH scale
        },
        {{{{12288, 2560}, {12288, 2560}}, ge::DT_INT8, ge::FORMAT_ND},
        {{{12288}, {12288}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{16}, {16}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{12288}, {12288}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(EXPERT_NUM)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(QUANT_MODE_DYNAMIC)},
            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 16})},
            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ROW_IDX_TYPE_DROPLESS)}, // SCATTER
        },
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1121000, "", {});

}

// 用例24: 计数排序 - CONSUM类型 + DropPad模式
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_counting_sort_consum_droppad)
{
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t expert_num = EXPERT_NUM;

    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV3",
        {
            {{{1536, 2560}, {1536, 2560}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1536, 8}, {1536, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1536}, {1536}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{{{expert_num, 50, 2560}, {expert_num, 50, 2560}}, ge::DT_INT8, ge::FORMAT_ND},
        {{{12288}, {12288}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{expert_num}, {expert_num}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{expert_num * 50}, {expert_num * 50}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(50)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, // DropPad
            {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, // CONSUM
            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(QUANT_MODE_NONE)},
            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, (int64_t)expert_num})},
            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ROW_IDX_TYPE_DROPPAD)},
        },
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1300000, "", {});

}
