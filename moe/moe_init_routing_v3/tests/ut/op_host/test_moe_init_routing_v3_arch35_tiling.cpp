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
#include <vector>
#include <string>
#include <cstdint>
#include <limits>
#include <gtest/gtest.h>
#include "../../../op_host/moe_init_routing_v3_tiling.h"
#include "../../../op_kernel/arch35/moe_init_routing_v3_arch35_tiling_def.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

namespace {
// 固定属性
constexpr int64_t EXPERT_NUM = 256;
// 可选quant_mode
constexpr int64_t QUANT_MODE_UNQUANT = -1;
constexpr int64_t QUANT_MODE_STATIC = 0;
constexpr int64_t QUANT_MODE_DYNAMIC = 1;
constexpr int64_t QUANT_MODE_MXFP8_E5M2 = 2;
constexpr int64_t QUANT_MODE_MXFP8_E4M3FN = 3;
constexpr int64_t QUANT_MODE_FP8_GROUP_E5M2 = 4;
constexpr int64_t QUANT_MODE_FP8_GROUP_E4M3FN = 5;
constexpr int64_t QUANT_MODE_HIF8_CAST = 6;
constexpr int64_t QUANT_MODE_HIF8_PERTENSOR = 7;
constexpr int64_t QUANT_MODE_HIF8_PERTOKEN = 8;
constexpr int64_t QUANT_MODE_MXFP4_E2M1 = 9;
constexpr int64_t QUANT_MODE_FP8_PERBLOCK_E5M2 = 11;
constexpr int64_t QUANT_MODE_FP8_PERBLOCK_E4M3FN = 12;
constexpr int64_t QUANT_MODE_INT4_DYNAMIC = 13;
constexpr int64_t QUANT_MODE_FP8_GROUP_AMAX_E5M2 = 14;
constexpr int64_t QUANT_MODE_FP8_GROUP_AMAX_E4M3FN = 15;
constexpr int64_t QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E5M2 = 16;
constexpr int64_t QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E4M3FN = 17;
// 可选row_idx_type
constexpr int64_t ROW_IDX_TYPE_GATHER = 0;
constexpr int64_t ROW_IDX_TYPE_SCATTER = 1;
constexpr int64_t EXPERT_TOKENS_TYPE_COUNT = 1;
constexpr int64_t EXPERT_TOKENS_TYPE_KEY_VALUE = 2;
constexpr uint64_t SKIP_TILING_KEY_VALIDATION = std::numeric_limits<uint64_t>::max();
constexpr ge::DataType kExpandedXDtypeAuto = static_cast<ge::DataType>(-2);

int64_t CeilDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

int64_t CeilAlign(int64_t a, int64_t align)
{
    return CeilDiv(a, align) * align;
}

ge::DataType GetExpandedXDtype(int64_t quantMode, ge::DataType xDtype, ge::DataType expandedXDtypeOverride)
{
    if (expandedXDtypeOverride != kExpandedXDtypeAuto) {
        return expandedXDtypeOverride;
    }
    switch (quantMode) {
        case QUANT_MODE_UNQUANT:
            return xDtype;
        case QUANT_MODE_STATIC:
        case QUANT_MODE_DYNAMIC:
            return ge::DT_INT8;
        case QUANT_MODE_INT4_DYNAMIC:
            return ge::DT_INT4;
        case QUANT_MODE_MXFP8_E5M2:
        case QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E5M2:
        case QUANT_MODE_FP8_GROUP_E5M2:
        case QUANT_MODE_FP8_PERBLOCK_E5M2:
        case QUANT_MODE_FP8_GROUP_AMAX_E5M2:
            return ge::DT_FLOAT8_E5M2;
        case QUANT_MODE_MXFP8_E4M3FN:
        case QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E4M3FN:
        case QUANT_MODE_FP8_GROUP_E4M3FN:
        case QUANT_MODE_FP8_PERBLOCK_E4M3FN:
        case QUANT_MODE_FP8_GROUP_AMAX_E4M3FN:
            return ge::DT_FLOAT8_E4M3FN;
        case QUANT_MODE_MXFP4_E2M1:
            return ge::DT_FLOAT4_E2M1;
        case QUANT_MODE_HIF8_CAST:
        case QUANT_MODE_HIF8_PERTENSOR:
        case QUANT_MODE_HIF8_PERTOKEN:
            return ge::DT_HIFLOAT8;
        default:
            return ge::DT_INT8;
    }
}

struct ExpandedScaleDesc {
    std::vector<int64_t> shape;
    ge::DataType dtype = ge::DT_FLOAT;
};

bool IsMxfpXNoQuant(ge::DataType xDtype)
{
    return xDtype == ge::DT_FLOAT8_E5M2 || xDtype == ge::DT_FLOAT8_E4M3FN || xDtype == ge::DT_FLOAT4_E2M1;
}

ExpandedScaleDesc MakePerTokenScaleDesc(int64_t totalLength)
{
    ExpandedScaleDesc desc;
    desc.shape = {totalLength};
    return desc;
}

ExpandedScaleDesc MakeMxfp8ScaleDesc(int64_t totalLength, int64_t cols)
{
    ExpandedScaleDesc desc;
    desc.dtype = ge::DT_FLOAT8_E8M0;
    desc.shape = {totalLength, CeilAlign(CeilDiv(cols, 32), 2)};
    return desc;
}

ExpandedScaleDesc MakeMxfp4ScaleDesc(int64_t totalLength, int64_t cols)
{
    ExpandedScaleDesc desc;
    desc.dtype = ge::DT_FLOAT8_E8M0;
    desc.shape = {totalLength, CeilDiv(cols, 64), 2};
    return desc;
}

ExpandedScaleDesc MakeFp8PerBlockScaleDesc(int64_t totalLength, int64_t cols)
{
    ExpandedScaleDesc desc;
    desc.dtype = ge::DT_FLOAT;
    desc.shape = {totalLength, CeilDiv(cols, 256), 2};
    return desc;
}

ExpandedScaleDesc MakeFp8GroupScaleDesc(int64_t totalLength, int64_t cols)
{
    ExpandedScaleDesc desc;
    desc.dtype = ge::DT_FLOAT;
    desc.shape = {totalLength, CeilDiv(cols, 128)};
    return desc;
}

ExpandedScaleDesc MakeUnquantInputScaleDesc(int64_t totalLength, int64_t cols, ge::DataType xDtype)
{
    if (IsMxfpXNoQuant(xDtype)) {
        ExpandedScaleDesc desc;
        desc.dtype = ge::DT_FLOAT8_E8M0;
        desc.shape = {totalLength, CeilDiv(cols, 64), 2};
        return desc;
    }
    return MakePerTokenScaleDesc(totalLength);
}

ExpandedScaleDesc GetExpandedScaleDesc(int64_t quantMode, int64_t totalLength, int64_t cols, int64_t n,
                                       bool isInputScale, ge::DataType xDtype, ge::DataType expandedXDtype)
{
    (void)n;
    (void)expandedXDtype;
    switch (quantMode) {
        case QUANT_MODE_STATIC:
        case QUANT_MODE_HIF8_CAST:
        case QUANT_MODE_HIF8_PERTENSOR:
        case QUANT_MODE_HIF8_PERTOKEN:
        case QUANT_MODE_DYNAMIC:
        case QUANT_MODE_INT4_DYNAMIC:
            return MakePerTokenScaleDesc(totalLength);
        case QUANT_MODE_MXFP8_E5M2:
        case QUANT_MODE_MXFP8_E4M3FN:
        case QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E5M2:
        case QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E4M3FN:
            return MakeMxfp8ScaleDesc(totalLength, cols);
        case QUANT_MODE_MXFP4_E2M1:
            return MakeMxfp4ScaleDesc(totalLength, cols);
        case QUANT_MODE_FP8_PERBLOCK_E5M2:
        case QUANT_MODE_FP8_PERBLOCK_E4M3FN:
            return MakeFp8PerBlockScaleDesc(totalLength, cols);
        case QUANT_MODE_FP8_GROUP_E5M2:
        case QUANT_MODE_FP8_GROUP_E4M3FN:
        case QUANT_MODE_FP8_GROUP_AMAX_E5M2:
        case QUANT_MODE_FP8_GROUP_AMAX_E4M3FN:
            return MakeFp8GroupScaleDesc(totalLength, cols);
        case QUANT_MODE_UNQUANT:
            if (isInputScale) {
                return MakeUnquantInputScaleDesc(totalLength, cols, xDtype);
            }
            return MakePerTokenScaleDesc(totalLength);
        default:
            return MakePerTokenScaleDesc(totalLength);
    }
}

gert::StorageShape MakeStorageShape(const std::vector<int64_t> &dims)
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

void AppendOptionalInput(std::vector<gert::TilingContextPara::TensorDescription> &inputs,
                         const std::vector<int64_t> &shape, ge::DataType dtype)
{
    inputs.emplace_back(MakeStorageShape(shape), dtype, ge::FORMAT_ND);
}

void AppendOptionalOutput(std::vector<gert::TilingContextPara::TensorDescription> &outputs,
                          const std::vector<int64_t> &shape, ge::DataType dtype)
{
    outputs.emplace_back(MakeStorageShape(shape), dtype, ge::FORMAT_ND);
}

static std::string A5SocInfo = "{\n"
                               "  \"hardware_info\": {\n"
                               "    \"BT_SIZE\": 0,\n"
                               "    \"load3d_constraints\": \"1\",\n"
                               "    \"Intrinsic_fix_pipe_l0c2out\": false,\n"
                               "    \"Intrinsic_data_move_l12ub\": true,\n"
                               "    \"Intrinsic_data_move_l0c2ub\": true,\n"
                               "    \"Intrinsic_data_move_out2l1_nd2nz\": false,\n"
                               "    \"UB_SIZE\": 262144,\n"
                               "    \"L2_SIZE\": 134217728,\n"
                               "    \"L1_SIZE\": 524288,\n"
                               "    \"L0A_SIZE\": 65536,\n"
                               "    \"L0B_SIZE\": 65536,\n"
                               "    \"L0C_SIZE\": 262144,\n"
                               "    \"CORE_NUM\": 32,\n"
                               "    \"socVersion\": \"Ascend910_95\"\n"
                               "  }\n"
                               "}";

std::vector<int64_t> GetExpertTokensShape(int64_t expertTokensNumType, int64_t expertNum, int64_t expertRange)
{
    if (expertTokensNumType == EXPERT_TOKENS_TYPE_KEY_VALUE) {
        return {expertNum, 2};
    }
    return {expertRange};
}

std::vector<gert::TilingContextPara::TensorDescription> BuildArch35ExtendedInputs(
    int64_t n, int64_t h, int64_t k, ge::DataType xDataType, const std::vector<int64_t> &scaleShape,
    ge::DataType scaleDtype, const std::vector<int64_t> &offsetShape, ge::DataType offsetDtype)
{
    std::vector<gert::TilingContextPara::TensorDescription> inputDesc;
    inputDesc.emplace_back(gert::StorageShape({n, h}, {n, h}), xDataType, ge::FORMAT_ND);
    inputDesc.emplace_back(gert::StorageShape({n, k}, {n, k}), ge::DT_INT32, ge::FORMAT_ND);
    AppendOptionalInput(inputDesc, scaleShape, scaleDtype);
    AppendOptionalInput(inputDesc, offsetShape, offsetDtype);
    return inputDesc;
}

std::vector<gert::TilingContextPara::TensorDescription> BuildArch35ExtendedOutputs(
    int64_t totalLength, int64_t h, ge::DataType expandedXDtype, const std::vector<int64_t> &expertTokensShape,
    const ExpandedScaleDesc &expandedScale)
{
    std::vector<gert::TilingContextPara::TensorDescription> outputDesc;
    outputDesc.emplace_back(gert::StorageShape({totalLength, h}, {totalLength, h}), expandedXDtype, ge::FORMAT_ND);
    outputDesc.emplace_back(gert::StorageShape({totalLength}, {totalLength}), ge::DT_INT32, ge::FORMAT_ND);
    outputDesc.emplace_back(MakeStorageShape(expertTokensShape), ge::DT_INT64, ge::FORMAT_ND);
    AppendOptionalOutput(outputDesc, expandedScale.shape, expandedScale.dtype);
    return outputDesc;
}

std::vector<gert::TilingContextPara::OpAttr> BuildArch35ExtendedAttrs(
    int64_t activeNum, int64_t expertCapacity, int64_t expertNum, int64_t dropPadMode, int64_t expertTokensNumType,
    bool expertTokensNumFlag, int64_t quantMode, const std::vector<int64_t> &aciveExpertRange, int64_t rowIdxType)
{
    return {
        {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
        {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertCapacity)},
        {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertNum)},
        {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
        {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertTokensNumType)},
        {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(expertTokensNumFlag)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
        {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(aciveExpertRange)},
        {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rowIdxType)},
    };
}

gert::TilingContextPara MakeArch35ExtendedTilingContextPara(
    int64_t n, int64_t h, int64_t k, int64_t expertCapacity, int64_t dropPadMode, int64_t expertTokensNumType,
    bool expertTokensNumFlag, int64_t quantMode, ge::DataType xDataType, const std::vector<int64_t> &aciveExpertRange,
    int64_t rowIdxType, const std::vector<int64_t> &scaleShape, ge::DataType scaleDtype,
    const std::vector<int64_t> &offsetShape, ge::DataType offsetDtype, ge::DataType expandedXDtypeOverride,
    optiling::MoeInitRoutingV3CompileInfo *compileInfo)
{
    int64_t activeNum = n * k;
    int64_t expertNum = EXPERT_NUM;
    int64_t expertRange = aciveExpertRange[1] - aciveExpertRange[0];
    int64_t totalLength = n * k;
    ge::DataType expandedXDtype = GetExpandedXDtype(quantMode, xDataType, expandedXDtypeOverride);
    ExpandedScaleDesc expandedScale =
        GetExpandedScaleDesc(quantMode, totalLength, h, n, !scaleShape.empty(), xDataType, expandedXDtype);
    std::vector<int64_t> expertTokensShape = GetExpertTokensShape(expertTokensNumType, expertNum, expertRange);
    auto inputDesc = BuildArch35ExtendedInputs(n, h, k, xDataType, scaleShape, scaleDtype, offsetShape, offsetDtype);
    auto outputDesc = BuildArch35ExtendedOutputs(totalLength, h, expandedXDtype, expertTokensShape, expandedScale);
    auto attrs = BuildArch35ExtendedAttrs(activeNum, expertCapacity, expertNum, dropPadMode, expertTokensNumType,
                                          expertTokensNumFlag, quantMode, aciveExpertRange, rowIdxType);
    return gert::TilingContextPara("MoeInitRoutingV3", inputDesc, outputDesc, attrs, compileInfo, "Ascend950",
                                   A5SocInfo, 4096);
}

void RunArch35ExtendedTestcase(int64_t N, int64_t H, int64_t K, int64_t expertCapacity, int64_t dropPadMode,
                               int64_t expertTokensNumType, bool expertTokensNumFlag, int64_t quantMode,
                               ge::DataType xDataType, std::vector<int64_t> aciveExpertRange, int64_t rowIdxType,
                               const std::vector<int64_t> &scaleShape, ge::DataType scaleDtype,
                               const std::vector<int64_t> &offsetShape, ge::DataType offsetDtype,
                               ge::DataType expandedXDtypeOverride, ge::graphStatus expectResult,
                               uint64_t expectTilingKey = SKIP_TILING_KEY_VALIDATION)
{
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 262144, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara = MakeArch35ExtendedTilingContextPara(
        N, H, K, expertCapacity, dropPadMode, expertTokensNumType, expertTokensNumFlag, quantMode, xDataType,
        aciveExpertRange, rowIdxType, scaleShape, scaleDtype, offsetShape, offsetDtype, expandedXDtypeOverride,
        &compileInfo);
    ExecuteTestCase(tilingContextPara, expectResult, expectTilingKey, "", {});
}

void ExpectArch35MxFP8NoQuantUseGatherCopy(int64_t n, int64_t h, int64_t k, int64_t expectedUseGatherCopy)
{
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 262144, platform_ascendc::SocVersion::ASCEND950};
    std::vector<int64_t> scaleShape = {n, CeilDiv(h, 64), 2};
    gert::TilingContextPara tilingContextPara = MakeArch35ExtendedTilingContextPara(
        n, h, k, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_UNQUANT, ge::DT_FLOAT8_E5M2, {0, 100},
        ROW_IDX_TYPE_SCATTER, scaleShape, ge::DT_FLOAT8_E8M0, {}, ge::DT_FLOAT, kExpandedXDtypeAuto, &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(MoeInitRoutingV3Arch35TilingData));
    const auto *tilingData = reinterpret_cast<const MoeInitRoutingV3Arch35TilingData *>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->useGatherCopy, expectedUseGatherCopy);
}
} // namespace

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

void RunSuccessTestcase(int64_t N, int64_t H, int64_t K, int64_t expertCapacity, int64_t dropPadMode,
                        int64_t expertTokensNumType, bool expertTokensNumFlag, int64_t quantMode, int64_t isInputScale,
                        ge::DataType xDataType, ge::DataType expandedXDtype, std::vector<int64_t> aciveExpertRange,
                        int64_t rowIdxType, ge::graphStatus result, int64_t expectTilingKey,
                        std::string expectTilingData, std::vector<size_t> expectWorkspaces)
{
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 262144, platform_ascendc::SocVersion::ASCEND950};
    int64_t activeNum = N * K;
    int64_t expert_num = EXPERT_NUM;
    int64_t E = aciveExpertRange[1] - aciveExpertRange[0];
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV3",
        {
            {{{N, H}, {N, H}}, xDataType, ge::FORMAT_ND},
            {{{N, K}, {N, K}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{{{N * K, H}, {N * K, H}}, expandedXDtype, ge::FORMAT_ND},
         {{{N * K}, {N * K}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{N * K}, {N * K}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertCapacity)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
            {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertTokensNumType)},
            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(expertTokensNumFlag)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(aciveExpertRange)},
            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rowIdxType)},
        },
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

void RunFailureTestcase(int64_t N, int64_t H, int64_t K, int64_t expertCapacity, int64_t dropPadMode,
                        int64_t expertTokensNumType, bool expertTokensNumFlag, int64_t quantMode, int64_t isInputScale,
                        ge::DataType xDataType, std::vector<int64_t> aciveExpertRange, int64_t rowIdxType,
                        ge::graphStatus result, int64_t expectTilingKey, std::string expectTilingData,
                        std::vector<size_t> expectWorkspaces)
{
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 196608, platform_ascendc::SocVersion::ASCEND950};
    int64_t activeNum = N * K;
    int64_t expert_num = EXPERT_NUM;
    int64_t E = aciveExpertRange[1] - aciveExpertRange[0];
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV3",
        {
            {{{N, H}, {N, H}}, xDataType, ge::FORMAT_ND},
            {{{N, K}, {N, K}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{{{N * K, H}, {N * K, H}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{N * K}, {N * K}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{N * K}, {N * K}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertCapacity)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
            {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertTokensNumType)},
            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(expertTokensNumFlag)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(aciveExpertRange)},
            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rowIdxType)},
        },
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

void RunDropPadTilingTestcase(int64_t N, int64_t H, int64_t K, int64_t expertCapacity,
                              std::vector<int64_t> activeExpertRange, int64_t rowIdxType, ge::graphStatus result,
                              int64_t expectTilingKey = UINT64_MAX)
{
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 262144, platform_ascendc::SocVersion::ASCEND950};
    int64_t activeNum = N * K;
    int64_t expertNum = EXPERT_NUM;
    int64_t expertRange = activeExpertRange[1] - activeExpertRange[0];
    gert::TilingContextPara tilingContextPara(
        "MoeInitRoutingV3",
        {
            {{{N, H}, {N, H}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{N, K}, {N, K}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{{{expertNum, expertCapacity, H}, {expertNum, expertCapacity, H}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{N * K}, {N * K}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{expertRange}, {expertRange}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{expertNum * expertCapacity}, {expertNum * expertCapacity}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertCapacity)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertNum)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(QUANT_MODE_UNQUANT)},
            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(activeExpertRange)},
            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rowIdxType)},
        },
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, result, expectTilingKey);
}

TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_droppad_non_full_load_multi_row_loop)
{
    RunDropPadTilingTestcase(1000000, 16, 1, 1, {0, EXPERT_NUM}, ROW_IDX_TYPE_GATHER, ge::GRAPH_SUCCESS, 11000100);
}

TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_droppad_non_full_load_single_row_loop)
{
    RunDropPadTilingTestcase(8192, 16, 1, 1, {0, EXPERT_NUM}, ROW_IDX_TYPE_GATHER, ge::GRAPH_SUCCESS, 11000100);
}

TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_droppad_non_full_load_tail_core)
{
    RunDropPadTilingTestcase(8201, 16, 1, 1, {0, EXPERT_NUM}, ROW_IDX_TYPE_GATHER, ge::GRAPH_SUCCESS, 11000100);
}

TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_droppad_non_full_load_multi_k)
{
    RunDropPadTilingTestcase(1024, 16, 8, 1, {0, EXPERT_NUM}, ROW_IDX_TYPE_GATHER, ge::GRAPH_SUCCESS, 11000100);
}

TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_droppad_non_full_load_multi_col_loop)
{
    RunDropPadTilingTestcase(4096, 32769, 2, 1, {0, EXPERT_NUM}, ROW_IDX_TYPE_GATHER, ge::GRAPH_SUCCESS, 11000100);
}

TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_droppad_non_full_load_multi_col_aligned)
{
    RunDropPadTilingTestcase(4096, 65536, 2, 1, {0, EXPERT_NUM}, ROW_IDX_TYPE_GATHER, ge::GRAPH_SUCCESS, 11000100);
}

TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_droppad_reject_scatter_row_idx)
{
    RunDropPadTilingTestcase(4, 16, 2, 2, {0, EXPERT_NUM}, ROW_IDX_TYPE_SCATTER, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_droppad_reject_partial_expert_range)
{
    RunDropPadTilingTestcase(4, 16, 2, 2, {0, 128}, ROW_IDX_TYPE_GATHER, ge::GRAPH_FAILED);
}

// fullload + not quant 200000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_01)
{
    std::string expectTilingData =
        "40 1 83 27 180 192 12 -1 0 0 0 0 256 1 1 0 1 27 0 0 0 0 1 27 1 27 27 27 1 27 27 6144 0 1024 27 1 1 1 1 1 1 "
        "1 1 27 1 1 1 1 1 1 1 1 1 83 83 27 6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16780660};
    RunSuccessTestcase(1, 83, 27, 0, 0, 1, true, QUANT_MODE_UNQUANT, 1, ge::DT_FLOAT, ge::DT_FLOAT, {180, 192},
                       ROW_IDX_TYPE_GATHER, ge::GRAPH_SUCCESS, 200000, expectTilingData, expectWorkspaces);
}

// fullload + not quant 200000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_02)
{
    std::string expectTilingData =
        "40 1 83 27 180 192 12 -1 1 0 0 0 256 1 1 0 1 27 0 0 0 0 1 27 1 27 27 27 1 27 27 6144 0 1024 27 1 1 1 1 1 1 "
        "1 1 27 1 1 1 1 1 1 1 1 1 83 83 27 6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16780660};
    RunSuccessTestcase(1, 83, 27, 0, 0, 1, true, QUANT_MODE_UNQUANT, 0, ge::DT_FLOAT, ge::DT_FLOAT, {180, 192},
                       ROW_IDX_TYPE_SCATTER, ge::GRAPH_SUCCESS, 200000, expectTilingData, expectWorkspaces);
}

// 多核 + not quant + scale None 1100000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_03)
{
    std::string expectTilingData =
        "40 160 96 1450 180 192 12 -1 0 0 0 0 256 1 1 0 1 232000 0 0 0 0 40 5824 1 5824 5824 4864 1 4864 4864 6144 "
        "10 1024 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 8 744 592 8 744 592 1 96 96 232000 2 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23275904};
    RunSuccessTestcase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_UNQUANT, 0, ge::DT_INT8, ge::DT_INT8, {180, 192},
                       ROW_IDX_TYPE_GATHER, ge::GRAPH_SUCCESS, 11000000, expectTilingData, expectWorkspaces);
}

// 多核 + not quant + drop pad + scale None 1101000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_04)
{
    std::string expectTilingData =
        "40 160 96 1450 180 192 12 -1 1 0 0 0 256 1 1 0 1 232000 0 0 0 0 40 5824 1 5824 5824 4864 1 4864 4864 6144 "
        "10 1024 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 25 234 184 25 234 184 1 96 96 232000 2 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23275904};
    RunSuccessTestcase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_UNQUANT, 0, ge::DT_FLOAT, ge::DT_FLOAT, {180, 192},
                       ROW_IDX_TYPE_SCATTER, ge::GRAPH_SUCCESS, 11001000, expectTilingData, expectWorkspaces);
}

// fullload + dynamci quant 220000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_05)
{
    std::string expectTilingData =
        "40 1 83 27 180 192 12 1 0 0 0 0 256 1 1 0 1 27 0 0 0 0 1 27 1 27 27 27 1 27 27 6144 0 1024 27 1 1 1 1 1 1 1 "
        "1 27 1 1 1 1 1 1 1 1 1 83 83 27 6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16793940};
    RunSuccessTestcase(1, 83, 27, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT, ge::DT_INT8, {180, 192},
                       ROW_IDX_TYPE_GATHER, ge::GRAPH_SUCCESS, 220000, expectTilingData, expectWorkspaces);
}

// fullload + dynamci quant 220000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_06)
{
    std::string expectTilingData =
        "40 1 83 27 180 192 12 1 0 0 0 0 256 1 1 0 1 27 0 0 0 0 1 27 1 27 27 27 1 27 27 6144 0 1024 27 1 1 1 1 1 1 1 "
        "1 27 1 1 1 1 1 1 1 1 1 83 83 27 6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16793940};
    RunSuccessTestcase(1, 83, 27, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_FLOAT, ge::DT_INT8, {180, 192},
                       ROW_IDX_TYPE_GATHER, ge::GRAPH_SUCCESS, 220000, expectTilingData, expectWorkspaces);
}

// fullload + dynamci quant 220000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_07)
{
    std::string expectTilingData =
        "40 8 60 32 0 100 100 1 1 0 0 0 256 1 1 0 1 256 0 0 0 0 1 256 1 256 256 256 1 256 256 6144 0 1024 37 7 4 1 7 "
        "7 1 4 4 37 7 4 1 7 7 1 4 4 1 60 60 256 6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16797376};
    RunSuccessTestcase(8, 60, 32, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT, ge::DT_INT8, {0, 100},
                       ROW_IDX_TYPE_SCATTER, ge::GRAPH_SUCCESS, 220000, expectTilingData, expectWorkspaces);
}

// fullload + dynamci quant 220000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_08)
{
    std::string expectTilingData =
        "40 8 60 32 0 100 100 1 1 0 0 0 256 1 1 0 1 256 0 0 0 0 1 256 1 256 256 256 1 256 256 6144 0 1024 37 7 4 1 7 "
        "7 1 4 4 37 7 4 1 7 7 1 4 4 1 60 60 256 6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16797376};
    RunSuccessTestcase(8, 60, 32, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_FLOAT, ge::DT_INT8, {0, 100},
                       ROW_IDX_TYPE_SCATTER, ge::GRAPH_SUCCESS, 220000, expectTilingData, expectWorkspaces);
}

// 多核 + dynamci quant + scale not None  11020000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_09)
{
    std::string expectTilingData =
        "40 160 96 1450 180 192 12 1 0 0 0 0 256 1 1 0 1 232000 0 0 0 0 40 5824 1 5824 5824 4864 1 4864 4864 6144 10 "
        "1024 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 1 5800 5800 1 5800 5800 1 96 96 232000 6 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23291264};
    RunSuccessTestcase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT, ge::DT_INT8, {180, 192},
                       ROW_IDX_TYPE_GATHER, ge::GRAPH_SUCCESS, 11020000, expectTilingData, expectWorkspaces);
}

// 多核 + dynamci quant + scale None 11020000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_10)
{
    std::string expectTilingData =
        "40 160 96 1450 180 192 12 1 0 0 0 0 256 1 1 0 1 232000 0 0 0 0 40 5824 1 5824 5824 4864 1 4864 4864 6144 10 "
        "1024 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 1 5800 5800 1 5800 5800 1 96 96 232000 6 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23291264};
    RunSuccessTestcase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_FLOAT, ge::DT_INT8, {180, 192},
                       ROW_IDX_TYPE_GATHER, ge::GRAPH_SUCCESS, 11020000, expectTilingData, expectWorkspaces);
}

// 多核 + dynamci quant + drop mode + scale not None  11021000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_11)
{
    std::string expectTilingData =
        "40 160 96 1450 0 100 100 1 1 0 0 0 256 1 1 0 1 232000 0 0 0 0 40 5824 1 5824 5824 4864 1 4864 4864 6144 10 "
        "1024 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 1 5800 5800 1 5800 5800 1 96 96 232000 6 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23291968};
    RunSuccessTestcase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT, ge::DT_INT8, {0, 100},
                       ROW_IDX_TYPE_SCATTER, ge::GRAPH_SUCCESS, 11021000, expectTilingData, expectWorkspaces);
}

// 多核 + dynamci quant + drop mode + scale None  11021000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_12)
{
    std::string expectTilingData =
        "40 160 96 1450 0 100 100 1 1 0 0 0 256 1 1 0 1 232000 0 0 0 0 40 5824 1 5824 5824 4864 1 4864 4864 6144 10 "
        "1024 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 1 5800 5800 1 5800 5800 1 96 96 232000 6 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23291968};
    RunSuccessTestcase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_FLOAT, ge::DT_INT8, {0, 100},
                       ROW_IDX_TYPE_SCATTER, ge::GRAPH_SUCCESS, 11021000, expectTilingData, expectWorkspaces);
}

// 多核 + dynamci quant + drop mode + scale None + bfloat16  11021000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_13)
{
    std::string expectTilingData =
        "40 160 96 1450 0 100 100 1 1 0 0 0 256 1 1 0 1 232000 0 0 0 0 40 5824 1 5824 5824 4864 1 4864 4864 6144 10 "
        "1024 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 1 5800 5800 1 5800 5800 1 96 96 232000 6 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23291968};
    RunSuccessTestcase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_BF16, ge::DT_INT8, {0, 100},
                       ROW_IDX_TYPE_SCATTER, ge::GRAPH_SUCCESS, 11021000, expectTilingData, expectWorkspaces);
}

// 多核 + dynamci quant + drop mode + scale None + float16 11021000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_14)
{
    std::string expectTilingData =
        "40 160 96 1450 0 100 100 1 1 0 0 0 256 1 1 0 1 232000 0 0 0 0 40 5824 1 5824 5824 4864 1 4864 4864 6144 10 "
        "1024 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 1 5800 5800 1 5800 5800 1 96 96 232000 6 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23291968};
    RunSuccessTestcase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_FLOAT16, ge::DT_INT8, {0, 100},
                       ROW_IDX_TYPE_SCATTER, ge::GRAPH_SUCCESS, 11021000, expectTilingData, expectWorkspaces);
}

// 单核 + static quant + drop mode + scale not None   1100000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_15)
{
    std::string expectTilingData = "40 1 83 27 180 192 12 -1 0 0 0 256 1 1 27 1 27 27 27 1 27 27 1984 0 1024 27 1 1 1 "
                                   "1 1 1 1 1 27 1 1 1 1 1 1 1 1 1 83 83 ";
    std::vector<size_t> expectWorkspaces = {5329576};
    RunFailureTestcase(1, 83, 27, 0, 0, 1, true, QUANT_MODE_STATIC, 1, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_SCATTER,
                       ge::GRAPH_FAILED, 1020000, expectTilingData, expectWorkspaces);
}

// MXFP8 E5M2 fullload
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_mxfp8_e5m2)
{
    int64_t h = 83;
    RunArch35ExtendedTestcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_MXFP8_E5M2, ge::DT_FLOAT16,
                              {180, 192}, ROW_IDX_TYPE_GATHER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto,
                              ge::GRAPH_SUCCESS);
}

// MXFP8 E4M3FN fullload
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_mxfp8_e4m3)
{
    int64_t h = 83;
    RunArch35ExtendedTestcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_MXFP8_E4M3FN, ge::DT_BF16,
                              {180, 192}, ROW_IDX_TYPE_SCATTER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto,
                              ge::GRAPH_SUCCESS);
}

// MXFP8 RoundScale + Amax E5M2 fullload
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_mxfp8_roundscale_amax_e5m2)
{
    int64_t h = 65;
    RunArch35ExtendedTestcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E5M2,
                              ge::DT_FLOAT16, {180, 192}, ROW_IDX_TYPE_GATHER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT,
                              kExpandedXDtypeAuto, ge::GRAPH_SUCCESS, 10170000);
}

// MXFP8 RoundScale + Amax E4M3FN fullload
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_mxfp8_roundscale_amax_e4m3)
{
    int64_t h = 97;
    RunArch35ExtendedTestcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E4M3FN,
                              ge::DT_BF16, {180, 192}, ROW_IDX_TYPE_SCATTER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT,
                              kExpandedXDtypeAuto, ge::GRAPH_SUCCESS, 10171000);
}

// MXFP4 E2M1 fullload
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_mxfp4)
{
    int64_t h = 83;
    RunArch35ExtendedTestcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_MXFP4_E2M1, ge::DT_FLOAT16,
                              {180, 192}, ROW_IDX_TYPE_GATHER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto,
                              ge::GRAPH_SUCCESS);
}

// FP8 PerBlock E5M2 fullload
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_fp8_perblock_e5m2)
{
    int64_t h = 83;
    RunArch35ExtendedTestcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_FP8_PERBLOCK_E5M2,
                              ge::DT_FLOAT16, {180, 192}, ROW_IDX_TYPE_GATHER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT,
                              kExpandedXDtypeAuto, ge::GRAPH_SUCCESS);
}

// FP8 PerBlock E4M3FN fullload
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_fp8_perblock_e4m3)
{
    int64_t h = 83;
    RunArch35ExtendedTestcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_FP8_PERBLOCK_E4M3FN,
                              ge::DT_BF16, {180, 192}, ROW_IDX_TYPE_SCATTER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT,
                              kExpandedXDtypeAuto, ge::GRAPH_SUCCESS);
}

// FP8 PerGroup E5M2
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_fp8_group_e5m2)
{
    int64_t h = 257;
    RunArch35ExtendedTestcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_FP8_GROUP_E5M2, ge::DT_FLOAT16,
                              {180, 192}, ROW_IDX_TYPE_GATHER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto,
                              ge::GRAPH_SUCCESS);
}

// FP8 PerGroup E4M3FN
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_fp8_group_e4m3)
{
    int64_t h = 257;
    RunArch35ExtendedTestcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_FP8_GROUP_E4M3FN, ge::DT_BF16,
                              {180, 192}, ROW_IDX_TYPE_SCATTER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto,
                              ge::GRAPH_SUCCESS);
}

// FP8 PerGroup E5M2 with amax clamp
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_fp8_group_amax_e5m2)
{
    int64_t h = 257;
    RunArch35ExtendedTestcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_FP8_GROUP_AMAX_E5M2,
                              ge::DT_FLOAT16, {180, 192}, ROW_IDX_TYPE_GATHER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT,
                              kExpandedXDtypeAuto, ge::GRAPH_SUCCESS);
}

// FP8 PerGroup E4M3FN with amax clamp
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_fp8_group_amax_e4m3)
{
    int64_t h = 257;
    RunArch35ExtendedTestcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_FP8_GROUP_AMAX_E4M3FN,
                              ge::DT_BF16, {180, 192}, ROW_IDX_TYPE_SCATTER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT,
                              kExpandedXDtypeAuto, ge::GRAPH_SUCCESS);
}

// UNQUANT + FP8 x + E8M0 scale
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_unquant_fp8_scale)
{
    int64_t h = 83;
    RunArch35ExtendedTestcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_UNQUANT, ge::DT_FLOAT8_E5M2,
                              {180, 192}, ROW_IDX_TYPE_GATHER, {1, 2, 2}, ge::DT_FLOAT8_E8M0, {}, ge::DT_FLOAT,
                              kExpandedXDtypeAuto, ge::GRAPH_SUCCESS);
}

TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_unquant_fp8_gather_copy_min_k)
{
    ExpectArch35MxFP8NoQuantUseGatherCopy(16, 64, 8, 1);
    ExpectArch35MxFP8NoQuantUseGatherCopy(16, 64, 16, 1);
}

TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_unquant_fp8_gather_copy_wide_cols)
{
    ExpectArch35MxFP8NoQuantUseGatherCopy(16, 8193, 16, 1);
}

// dynamic quant + per-expert scale
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_dynamic_with_scale)
{
    int64_t h = 83;
    RunArch35ExtendedTestcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_DYNAMIC, ge::DT_FLOAT,
                              {180, 192}, ROW_IDX_TYPE_GATHER, {12, 83}, ge::DT_FLOAT, {}, ge::DT_FLOAT,
                              kExpandedXDtypeAuto, ge::GRAPH_SUCCESS);
}

// INT4 dynamic quant + smooth scale (1, H)
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_int4_dynamic)
{
    int64_t h = 84;
    RunArch35ExtendedTestcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_INT4_DYNAMIC, ge::DT_FLOAT,
                              {180, 192}, ROW_IDX_TYPE_GATHER, {1, 84}, ge::DT_FLOAT, {}, ge::DT_FLOAT,
                              kExpandedXDtypeAuto, ge::GRAPH_SUCCESS);
}

// static quant success with scale and offset
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_static_success)
{
    int64_t h = 83;
    RunArch35ExtendedTestcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_STATIC, ge::DT_FLOAT,
                              {180, 192}, ROW_IDX_TYPE_GATHER, {1}, ge::DT_FLOAT, {1}, ge::DT_FLOAT,
                              kExpandedXDtypeAuto, ge::GRAPH_SUCCESS);
}

// HIF8 cast fullload
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_hif8_cast)
{
    int64_t h = 83;
    RunArch35ExtendedTestcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_HIF8_CAST, ge::DT_FLOAT16,
                              {180, 192}, ROW_IDX_TYPE_GATHER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto,
                              ge::GRAPH_SUCCESS);
}

// HIF8 pertoken fullload
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_hif8_pertoken)
{
    int64_t h = 83;
    RunArch35ExtendedTestcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_HIF8_PERTOKEN, ge::DT_BF16,
                              {180, 192}, ROW_IDX_TYPE_SCATTER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto,
                              ge::GRAPH_SUCCESS);
}

// MXFP8 multicore
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_mxfp8_multicore)
{
    int64_t h = 60;
    RunArch35ExtendedTestcase(8, 60, 32, 0, 0, EXPERT_TOKENS_TYPE_KEY_VALUE, true, QUANT_MODE_MXFP8_E5M2,
                              ge::DT_FLOAT16, {0, 100}, ROW_IDX_TYPE_SCATTER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT,
                              kExpandedXDtypeAuto, ge::GRAPH_SUCCESS);
}

// static quant missing scale
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_static_missing_scale)
{
    RunArch35ExtendedTestcase(1, 83, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_STATIC, ge::DT_FLOAT,
                              {180, 192}, ROW_IDX_TYPE_GATHER, {}, ge::DT_FLOAT, {1}, ge::DT_FLOAT, kExpandedXDtypeAuto,
                              ge::GRAPH_FAILED);
}

// INT4 dynamic with odd cols
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_int4_odd_cols)
{
    RunArch35ExtendedTestcase(1, 83, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_INT4_DYNAMIC, ge::DT_FLOAT,
                              {180, 192}, ROW_IDX_TYPE_GATHER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto,
                              ge::GRAPH_FAILED);
}

// MXFP4 unsupported x dtype
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_mxfp4_bad_xdtype)
{
    RunArch35ExtendedTestcase(1, 83, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_MXFP4_E2M1, ge::DT_FLOAT,
                              {180, 192}, ROW_IDX_TYPE_GATHER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto,
                              ge::GRAPH_FAILED);
}

// INT4 dynamic with float16 x
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_int4_bad_xdtype)
{
    RunArch35ExtendedTestcase(1, 84, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_INT4_DYNAMIC, ge::DT_FLOAT16,
                              {180, 192}, ROW_IDX_TYPE_GATHER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto,
                              ge::GRAPH_FAILED);
}

// UNQUANT FP8 with wrong scale dtype
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_unquant_fp8_bad_scale_dtype)
{
    RunArch35ExtendedTestcase(1, 83, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_UNQUANT, ge::DT_FLOAT8_E5M2,
                              {180, 192}, ROW_IDX_TYPE_GATHER, {1, 2, 2}, ge::DT_FLOAT, {}, ge::DT_FLOAT,
                              kExpandedXDtypeAuto, ge::GRAPH_FAILED);
}

// 空tensor场景：n_==0
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_empty_tensor_n_zero)
{
    RunArch35ExtendedTestcase(0, 128, 8, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_UNQUANT, ge::DT_FLOAT,
                              {0, EXPERT_NUM}, ROW_IDX_TYPE_GATHER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT,
                              kExpandedXDtypeAuto, ge::GRAPH_SUCCESS);
}

// 空tensor场景：k_==0（expertIdx第1维为0，如[3,0]）
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_empty_tensor_k_zero)
{
    RunArch35ExtendedTestcase(3, 128, 0, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_UNQUANT, ge::DT_FLOAT,
                              {0, EXPERT_NUM}, ROW_IDX_TYPE_GATHER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT,
                              kExpandedXDtypeAuto, ge::GRAPH_SUCCESS);
}

// 空tensor场景：cols_==0（x第1维为0，如[3,0]）
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_empty_tensor_cols_zero)
{
    RunArch35ExtendedTestcase(3, 0, 8, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_UNQUANT, ge::DT_FLOAT,
                              {0, EXPERT_NUM}, ROW_IDX_TYPE_GATHER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT,
                              kExpandedXDtypeAuto, ge::GRAPH_SUCCESS);
}

// 空tensor场景：k_==0 + key_value模式
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_empty_tensor_k_zero_keyvalue)
{
    RunArch35ExtendedTestcase(3, 64, 0, 0, 0, EXPERT_TOKENS_TYPE_KEY_VALUE, true, QUANT_MODE_UNQUANT, ge::DT_FLOAT16,
                              {0, EXPERT_NUM}, ROW_IDX_TYPE_SCATTER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT,
                              kExpandedXDtypeAuto, ge::GRAPH_SUCCESS);
}
