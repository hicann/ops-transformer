/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <string>
#include <cstdint>
#include <limits>
#include <gtest/gtest.h>
#include "../../../op_host/moe_init_routing_v4_tiling_arch35.h"
#include "../../../../moe_init_routing_v3/op_kernel/arch35/moe_init_routing_v3_arch35_tiling_def.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

namespace {
constexpr int64_t EXPERT_NUM = 256;
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

std::vector<gert::TilingContextPara::TensorDescription> BuildV4Inputs(
    int64_t n, int64_t h, int64_t k, ge::DataType xDataType, const std::vector<int64_t> &scaleShape,
    ge::DataType scaleDtype, const std::vector<int64_t> &offsetShape, ge::DataType offsetDtype, bool hasActiveNum,
    int64_t activeNumVal, bool hasTopkWeight, int64_t totalLength)
{
    std::vector<gert::TilingContextPara::TensorDescription> inputDesc;
    inputDesc.emplace_back(gert::StorageShape({n, h}, {n, h}), xDataType, ge::FORMAT_ND);
    inputDesc.emplace_back(gert::StorageShape({n, k}, {n, k}), ge::DT_INT32, ge::FORMAT_ND);
    AppendOptionalInput(inputDesc, scaleShape, scaleDtype);
    AppendOptionalInput(inputDesc, offsetShape, offsetDtype);
    if (hasActiveNum) {
        AppendOptionalInput(inputDesc, {1}, ge::DT_INT64);
    } else {
        AppendOptionalInput(inputDesc, {}, ge::DT_INT64);
    }
    if (hasTopkWeight) {
        AppendOptionalInput(inputDesc, {n, k}, ge::DT_FLOAT);
    } else {
        AppendOptionalInput(inputDesc, {}, ge::DT_FLOAT);
    }
    return inputDesc;
}

std::vector<gert::TilingContextPara::TensorDescription> BuildV4Outputs(
    int64_t totalLength, int64_t h, ge::DataType expandedXDtype, const std::vector<int64_t> &expertTokensShape,
    const ExpandedScaleDesc &expandedScale, bool hasExpandedTopkWeight, int64_t n, int64_t k, int64_t dropPadMode,
    int64_t expertNum, int64_t expertCapacity)
{
    std::vector<gert::TilingContextPara::TensorDescription> outputDesc;
    if (dropPadMode == 1) {
        outputDesc.emplace_back(gert::StorageShape({expertNum, expertCapacity, h}, {expertNum, expertCapacity, h}),
                                expandedXDtype, ge::FORMAT_ND);
    } else {
        outputDesc.emplace_back(gert::StorageShape({totalLength, h}, {totalLength, h}), expandedXDtype, ge::FORMAT_ND);
    }
    outputDesc.emplace_back(gert::StorageShape({totalLength}, {totalLength}), ge::DT_INT32, ge::FORMAT_ND);
    outputDesc.emplace_back(MakeStorageShape(expertTokensShape), ge::DT_INT64, ge::FORMAT_ND);
    AppendOptionalOutput(outputDesc, expandedScale.shape, expandedScale.dtype);
    if (hasExpandedTopkWeight) {
        int64_t topkWeightDim0 = (dropPadMode == 1) ? (expertNum * expertCapacity) : totalLength;
        AppendOptionalOutput(outputDesc, {topkWeightDim0, 1}, ge::DT_FLOAT);
    } else {
        AppendOptionalOutput(outputDesc, {}, ge::DT_FLOAT);
    }
    return outputDesc;
}

std::vector<gert::TilingContextPara::OpAttr> BuildV4Attrs(int64_t expertCapacity, int64_t expertNum,
                                                          int64_t dropPadMode, int64_t expertTokensNumType,
                                                          bool expertTokensNumFlag, int64_t quantMode,
                                                          const std::vector<int64_t> &aciveExpertRange,
                                                          int64_t rowIdxType)
{
    return {
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

gert::TilingContextPara MakeV4TilingContextPara(
    int64_t n, int64_t h, int64_t k, int64_t expertCapacity, int64_t dropPadMode, int64_t expertTokensNumType,
    bool expertTokensNumFlag, int64_t quantMode, ge::DataType xDataType, const std::vector<int64_t> &aciveExpertRange,
    int64_t rowIdxType, const std::vector<int64_t> &scaleShape, ge::DataType scaleDtype,
    const std::vector<int64_t> &offsetShape, ge::DataType offsetDtype, ge::DataType expandedXDtypeOverride,
    bool hasActiveNum, int64_t activeNumVal, bool hasTopkWeight, optiling::MoeInitRoutingV3CompileInfo *compileInfo)
{
    int64_t expertNum = EXPERT_NUM;
    int64_t expertRange = aciveExpertRange[1] - aciveExpertRange[0];
    int64_t totalLength = n * k;
    ge::DataType expandedXDtype = GetExpandedXDtype(quantMode, xDataType, expandedXDtypeOverride);
    ExpandedScaleDesc expandedScale =
        GetExpandedScaleDesc(quantMode, totalLength, h, n, !scaleShape.empty(), xDataType, expandedXDtype);
    std::vector<int64_t> expertTokensShape = GetExpertTokensShape(expertTokensNumType, expertNum, expertRange);
    auto inputDesc = BuildV4Inputs(n, h, k, xDataType, scaleShape, scaleDtype, offsetShape, offsetDtype, hasActiveNum,
                                   activeNumVal, hasTopkWeight, totalLength);
    auto outputDesc = BuildV4Outputs(totalLength, h, expandedXDtype, expertTokensShape, expandedScale, hasTopkWeight, n,
                                     k, dropPadMode, expertNum, expertCapacity);
    auto attrs = BuildV4Attrs(expertCapacity, expertNum, dropPadMode, expertTokensNumType, expertTokensNumFlag,
                              quantMode, aciveExpertRange, rowIdxType);
    return gert::TilingContextPara("MoeInitRoutingV4", inputDesc, outputDesc, attrs, compileInfo, "Ascend950",
                                   A5SocInfo, 4096);
}

void RunV4Testcase(int64_t N, int64_t H, int64_t K, int64_t expertCapacity, int64_t dropPadMode,
                   int64_t expertTokensNumType, bool expertTokensNumFlag, int64_t quantMode, ge::DataType xDataType,
                   std::vector<int64_t> aciveExpertRange, int64_t rowIdxType, const std::vector<int64_t> &scaleShape,
                   ge::DataType scaleDtype, const std::vector<int64_t> &offsetShape, ge::DataType offsetDtype,
                   ge::DataType expandedXDtypeOverride, bool hasActiveNum, int64_t activeNumVal, bool hasTopkWeight,
                   ge::graphStatus expectResult, uint64_t expectTilingKey = SKIP_TILING_KEY_VALIDATION)
{
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 262144, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara = MakeV4TilingContextPara(
        N, H, K, expertCapacity, dropPadMode, expertTokensNumType, expertTokensNumFlag, quantMode, xDataType,
        aciveExpertRange, rowIdxType, scaleShape, scaleDtype, offsetShape, offsetDtype, expandedXDtypeOverride,
        hasActiveNum, activeNumVal, hasTopkWeight, &compileInfo);
    ExecuteTestCase(tilingContextPara, expectResult, expectTilingKey, "", {});
}

struct V4TilingDataCase {
    int64_t n;
    int64_t h;
    int64_t k;
    int64_t expertCapacity;
    int64_t dropPadMode;
    std::vector<int64_t> activeExpertRange;
    int64_t rowIdxType;
    uint64_t expectedTilingKey;
    int64_t expertTokensNumType = EXPERT_TOKENS_TYPE_COUNT;
    int64_t quantMode = QUANT_MODE_UNQUANT;
    ge::DataType xDataType = ge::DT_FLOAT;
    std::vector<int64_t> scaleShape;
    ge::DataType scaleDtype = ge::DT_FLOAT;
    std::vector<int64_t> offsetShape;
    ge::DataType offsetDtype = ge::DT_FLOAT;
    bool hasTopkWeight = true;
    int64_t expectedUseGatherCopy = -1;
};

void ExpectV4TilingData(const V4TilingDataCase &testCase)
{
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 262144, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara = MakeV4TilingContextPara(
        testCase.n, testCase.h, testCase.k, testCase.expertCapacity, testCase.dropPadMode, testCase.expertTokensNumType,
        true, testCase.quantMode, testCase.xDataType, testCase.activeExpertRange, testCase.rowIdxType,
        testCase.scaleShape, testCase.scaleDtype, testCase.offsetShape, testCase.offsetDtype, kExpandedXDtypeAuto,
        false, 0, testCase.hasTopkWeight, &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(MoeInitRoutingV3Arch35TilingData));
    const auto *tilingData = reinterpret_cast<const MoeInitRoutingV3Arch35TilingData *>(tilingInfo.tilingData.get());

    EXPECT_EQ(tilingInfo.tilingKey, testCase.expectedTilingKey);
    EXPECT_EQ(tilingData->coreNum, 40);
    EXPECT_EQ(tilingData->n, testCase.n);
    EXPECT_EQ(tilingData->cols, testCase.h);
    EXPECT_EQ(tilingData->k, testCase.k);
    EXPECT_EQ(tilingData->expertStart, testCase.activeExpertRange[0]);
    EXPECT_EQ(tilingData->expertEnd, testCase.activeExpertRange[1]);
    EXPECT_EQ(tilingData->actualExpertNum, testCase.activeExpertRange[1] - testCase.activeExpertRange[0]);
    EXPECT_EQ(tilingData->quantMode, testCase.quantMode);
    EXPECT_EQ(tilingData->rowIdxType, testCase.rowIdxType);
    if (testCase.expectedUseGatherCopy >= 0) {
        EXPECT_EQ(tilingData->useGatherCopy, testCase.expectedUseGatherCopy);
    }
    EXPECT_EQ(tilingData->isInputScale, testCase.scaleShape.empty() ? 0 : 1);
    EXPECT_EQ(tilingData->isInputOffset, testCase.offsetShape.empty() ? 0 : 1);
    EXPECT_EQ(tilingData->expertNum, EXPERT_NUM);
    EXPECT_EQ(tilingData->expertTokensNumType, testCase.expertTokensNumType);
    EXPECT_EQ(tilingData->expertTokensNumFlag, 1);
    EXPECT_EQ(tilingData->activeNum, testCase.n * testCase.k);
    EXPECT_EQ(tilingData->dropPadMode, testCase.dropPadMode);
    EXPECT_EQ(tilingData->expertCapacity, testCase.expertCapacity);
    EXPECT_EQ(tilingData->isInputTopkWeight, testCase.hasTopkWeight ? 1 : 0);

    EXPECT_GT(tilingData->vbsComputeParamsOp.needCoreNum, 0);
    if (testCase.dropPadMode == 1) {
        EXPECT_GT(tilingData->srcToDstDropPadParamsOp.needCoreNum, 0);
    } else if (testCase.quantMode == QUANT_MODE_UNQUANT &&
               (testCase.xDataType == ge::DT_FLOAT8_E5M2 || testCase.xDataType == ge::DT_FLOAT8_E4M3FN)) {
        EXPECT_GT(tilingData->gatherOutComputeParamsOp.colsLoops, 0);
        EXPECT_GT(tilingData->gatherOutComputeParamsOp.perCorePerLoopIndicesElements, 0);
    } else {
        EXPECT_GT(tilingData->gatherOutComputeParamsOp.needCoreNum, 0);
    }
}
} // namespace

class MoeInitRoutingV4Tiling : public testing::Test {
protected:
};

TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_data_fullload_gather_topkweight)
{
    ExpectV4TilingData({1, 83, 27, 0, 0, {180, 192}, ROW_IDX_TYPE_GATHER, 200000});
}

TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_data_fullload_scatter_topkweight)
{
    ExpectV4TilingData({1, 83, 27, 0, 0, {180, 192}, ROW_IDX_TYPE_SCATTER, 200000});
}

TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_data_nonfull_gather_topkweight)
{
    ExpectV4TilingData({160, 96, 1450, 0, 0, {180, 192}, ROW_IDX_TYPE_GATHER, 11000000});
}

TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_data_nonfull_scatter_topkweight)
{
    ExpectV4TilingData({160, 96, 1450, 0, 0, {180, 192}, ROW_IDX_TYPE_SCATTER, 11001000});
}

TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_data_droppad_topkweight)
{
    ExpectV4TilingData({1000000, 16, 1, 1, 1, {0, EXPERT_NUM}, ROW_IDX_TYPE_GATHER, 11000100});
}

TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_data_fullload_dynamic_scale_topkweight)
{
    V4TilingDataCase testCase{1, 83, 27, 0, 0, {180, 192}, ROW_IDX_TYPE_GATHER, 220000};
    testCase.quantMode = QUANT_MODE_DYNAMIC;
    testCase.scaleShape = {12, 83};
    ExpectV4TilingData(testCase);
}

TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_data_nonfull_dynamic_gather_scale_topkweight)
{
    V4TilingDataCase testCase{160, 96, 1450, 0, 0, {180, 192}, ROW_IDX_TYPE_GATHER, 11020000};
    testCase.quantMode = QUANT_MODE_DYNAMIC;
    testCase.scaleShape = {12, 96};
    ExpectV4TilingData(testCase);
}

TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_data_nonfull_dynamic_scatter_bf16_topkweight)
{
    V4TilingDataCase testCase{160, 96, 1450, 0, 0, {0, 100}, ROW_IDX_TYPE_SCATTER, 11021000};
    testCase.quantMode = QUANT_MODE_DYNAMIC;
    testCase.xDataType = ge::DT_BF16;
    ExpectV4TilingData(testCase);
}

TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_data_fullload_static_scale_offset_topkweight)
{
    V4TilingDataCase testCase{1, 83, 27, 0, 0, {180, 192}, ROW_IDX_TYPE_GATHER, 210000};
    testCase.quantMode = QUANT_MODE_STATIC;
    testCase.scaleShape = {1};
    testCase.offsetShape = {1};
    ExpectV4TilingData(testCase);
}

TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_data_fullload_int4_dynamic_topkweight)
{
    V4TilingDataCase testCase{1, 84, 27, 0, 0, {180, 192}, ROW_IDX_TYPE_GATHER, 220000};
    testCase.quantMode = QUANT_MODE_INT4_DYNAMIC;
    testCase.scaleShape = {1, 84};
    ExpectV4TilingData(testCase);
}

TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_data_mxfp8_roundscale_topkweight)
{
    V4TilingDataCase testCase{1, 65, 27, 0, 0, {180, 192}, ROW_IDX_TYPE_GATHER, 10170000};
    testCase.quantMode = QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E5M2;
    testCase.xDataType = ge::DT_FLOAT16;
    ExpectV4TilingData(testCase);
}

TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_data_key_value_scatter_topkweight)
{
    V4TilingDataCase testCase{8, 60, 32, 0, 0, {0, 100}, ROW_IDX_TYPE_SCATTER, 200000};
    testCase.expertTokensNumType = EXPERT_TOKENS_TYPE_KEY_VALUE;
    testCase.xDataType = ge::DT_FLOAT16;
    ExpectV4TilingData(testCase);
}

TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_data_unquant_fp8_gather_copy_topkweight)
{
    V4TilingDataCase testCase{16, 64, 8, 0, 0, {0, 100}, ROW_IDX_TYPE_SCATTER, 10001000};
    testCase.xDataType = ge::DT_FLOAT8_E5M2;
    testCase.scaleShape = {16, 1, 2};
    testCase.scaleDtype = ge::DT_FLOAT8_E8M0;
    testCase.expectedUseGatherCopy = 1;
    ExpectV4TilingData(testCase);
}

// fullload + not quant, no topk_weight
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_fullload_unquant_no_topkweight)
{
    int64_t h = 83;
    RunV4Testcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_UNQUANT, ge::DT_FLOAT16, {180, 192},
                  ROW_IDX_TYPE_GATHER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto, false, 0, false,
                  ge::GRAPH_SUCCESS);
}

// fullload + not quant, with topk_weight
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_fullload_unquant_with_topkweight)
{
    int64_t h = 83;
    RunV4Testcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_UNQUANT, ge::DT_FLOAT16, {180, 192},
                  ROW_IDX_TYPE_GATHER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto, false, 0, true,
                  ge::GRAPH_SUCCESS);
}

// fullload + dynamic quant, no topk_weight
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_fullload_dynamic_no_topkweight)
{
    int64_t h = 83;
    RunV4Testcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_DYNAMIC, ge::DT_FLOAT, {180, 192},
                  ROW_IDX_TYPE_GATHER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto, false, 0, false,
                  ge::GRAPH_SUCCESS);
}

// fullload + dynamic quant, with topk_weight
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_fullload_dynamic_with_topkweight)
{
    int64_t h = 83;
    int64_t expertRange = 192 - 180;
    RunV4Testcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_DYNAMIC, ge::DT_FLOAT, {180, 192},
                  ROW_IDX_TYPE_GATHER, {expertRange, h}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto, false, 0,
                  true, ge::GRAPH_SUCCESS);
}

// multicore + not quant + no topk_weight
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_multicore_unquant_no_topkweight)
{
    int64_t h = 60;
    RunV4Testcase(8, h, 32, 0, 0, EXPERT_TOKENS_TYPE_KEY_VALUE, true, QUANT_MODE_UNQUANT, ge::DT_FLOAT16, {0, 100},
                  ROW_IDX_TYPE_SCATTER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto, false, 0, false,
                  ge::GRAPH_SUCCESS);
}

// multicore + not quant + with topk_weight
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_multicore_unquant_with_topkweight)
{
    int64_t h = 60;
    RunV4Testcase(8, h, 32, 0, 0, EXPERT_TOKENS_TYPE_KEY_VALUE, true, QUANT_MODE_UNQUANT, ge::DT_FLOAT16, {0, 100},
                  ROW_IDX_TYPE_SCATTER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto, false, 0, true,
                  ge::GRAPH_SUCCESS);
}

// multicore + dynamic quant + with topk_weight
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_multicore_dynamic_with_topkweight)
{
    int64_t h = 60;
    RunV4Testcase(8, h, 32, 0, 0, EXPERT_TOKENS_TYPE_KEY_VALUE, true, QUANT_MODE_DYNAMIC, ge::DT_BF16, {0, 100},
                  ROW_IDX_TYPE_SCATTER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto, false, 0, true,
                  ge::GRAPH_SUCCESS);
}

// droppad + not quant + with topk_weight
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_droppad_unquant_with_topkweight)
{
    RunV4Testcase(1000000, 16, 1, 1, 1, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_UNQUANT, ge::DT_FLOAT,
                  {0, EXPERT_NUM}, ROW_IDX_TYPE_GATHER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto, false,
                  0, true, ge::GRAPH_SUCCESS);
}

// droppad reject scatter row_idx
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_droppad_reject_scatter_row_idx)
{
    RunV4Testcase(4, 16, 2, 2, 1, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_UNQUANT, ge::DT_FLOAT, {0, EXPERT_NUM},
                  ROW_IDX_TYPE_SCATTER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto, false, 0, false,
                  ge::GRAPH_FAILED);
}

// MXFP8 E5M2 fullload + no topk_weight
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_mxfp8_e5m2_no_topkweight)
{
    int64_t h = 83;
    RunV4Testcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_MXFP8_E5M2, ge::DT_FLOAT16, {180, 192},
                  ROW_IDX_TYPE_GATHER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto, false, 0, false,
                  ge::GRAPH_SUCCESS);
}

// MXFP8 E4M3FN fullload + with topk_weight
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_mxfp8_e4m3_with_topkweight)
{
    int64_t h = 83;
    RunV4Testcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_MXFP8_E4M3FN, ge::DT_BF16, {180, 192},
                  ROW_IDX_TYPE_SCATTER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto, false, 0, true,
                  ge::GRAPH_SUCCESS);
}

// MXFP8 RoundScale + Amax E5M2 fullload
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_mxfp8_roundscale_amax_e5m2)
{
    int64_t h = 65;
    RunV4Testcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E5M2, ge::DT_FLOAT16,
                  {180, 192}, ROW_IDX_TYPE_GATHER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto, false, 0,
                  false, ge::GRAPH_SUCCESS);
}

// MXFP8 RoundScale + Amax E4M3FN fullload
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_mxfp8_roundscale_amax_e4m3)
{
    int64_t h = 97;
    RunV4Testcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E4M3FN, ge::DT_BF16,
                  {180, 192}, ROW_IDX_TYPE_SCATTER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto, false, 0,
                  false, ge::GRAPH_SUCCESS);
}

// MXFP4 E2M1 fullload
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_mxfp4)
{
    int64_t h = 83;
    RunV4Testcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_MXFP4_E2M1, ge::DT_FLOAT16, {180, 192},
                  ROW_IDX_TYPE_GATHER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto, false, 0, false,
                  ge::GRAPH_SUCCESS);
}

// FP8 PerBlock E5M2 fullload
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_fp8_perblock_e5m2)
{
    int64_t h = 83;
    RunV4Testcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_FP8_PERBLOCK_E5M2, ge::DT_FLOAT16,
                  {180, 192}, ROW_IDX_TYPE_GATHER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto, false, 0,
                  false, ge::GRAPH_SUCCESS);
}

// FP8 PerBlock E4M3FN fullload
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_fp8_perblock_e4m3)
{
    int64_t h = 83;
    RunV4Testcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_FP8_PERBLOCK_E4M3FN, ge::DT_BF16,
                  {180, 192}, ROW_IDX_TYPE_SCATTER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto, false, 0,
                  false, ge::GRAPH_SUCCESS);
}

// FP8 PerGroup E5M2
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_fp8_group_e5m2)
{
    int64_t h = 257;
    RunV4Testcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_FP8_GROUP_E5M2, ge::DT_FLOAT16, {180, 192},
                  ROW_IDX_TYPE_GATHER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto, false, 0, false,
                  ge::GRAPH_SUCCESS);
}

// FP8 PerGroup E4M3FN
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_fp8_group_e4m3)
{
    int64_t h = 257;
    RunV4Testcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_FP8_GROUP_E4M3FN, ge::DT_BF16, {180, 192},
                  ROW_IDX_TYPE_SCATTER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto, false, 0, false,
                  ge::GRAPH_SUCCESS);
}

// FP8 PerGroup with amax clamp E5M2
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_fp8_group_amax_e5m2)
{
    int64_t h = 257;
    RunV4Testcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_FP8_GROUP_AMAX_E5M2, ge::DT_FLOAT16,
                  {180, 192}, ROW_IDX_TYPE_GATHER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto, false, 0,
                  false, ge::GRAPH_SUCCESS);
}

// FP8 PerGroup with amax clamp E4M3FN
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_fp8_group_amax_e4m3)
{
    int64_t h = 257;
    RunV4Testcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_FP8_GROUP_AMAX_E4M3FN, ge::DT_BF16,
                  {180, 192}, ROW_IDX_TYPE_SCATTER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto, false, 0,
                  false, ge::GRAPH_SUCCESS);
}

// UNQUANT + FP8 x + E8M0 scale
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_unquant_fp8_scale)
{
    int64_t h = 83;
    RunV4Testcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_UNQUANT, ge::DT_FLOAT8_E5M2, {180, 192},
                  ROW_IDX_TYPE_GATHER, {1, 2, 2}, ge::DT_FLOAT8_E8M0, {}, ge::DT_FLOAT, kExpandedXDtypeAuto, false, 0,
                  false, ge::GRAPH_SUCCESS);
}

// dynamic quant + per-expert scale
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_dynamic_with_scale)
{
    int64_t h = 83;
    RunV4Testcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_DYNAMIC, ge::DT_FLOAT, {180, 192},
                  ROW_IDX_TYPE_GATHER, {12, h}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto, false, 0, false,
                  ge::GRAPH_SUCCESS);
}

// INT4 dynamic quant + smooth scale (1, H)
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_int4_dynamic)
{
    int64_t h = 84;
    RunV4Testcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_INT4_DYNAMIC, ge::DT_FLOAT, {180, 192},
                  ROW_IDX_TYPE_GATHER, {1, h}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto, false, 0, false,
                  ge::GRAPH_SUCCESS);
}

// static quant success with scale and offset
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_static_success)
{
    int64_t h = 83;
    RunV4Testcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_STATIC, ge::DT_FLOAT, {180, 192},
                  ROW_IDX_TYPE_GATHER, {1}, ge::DT_FLOAT, {1}, ge::DT_FLOAT, kExpandedXDtypeAuto, false, 0, false,
                  ge::GRAPH_SUCCESS);
}

// HIF8 cast fullload
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_hif8_cast)
{
    int64_t h = 83;
    RunV4Testcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_HIF8_CAST, ge::DT_FLOAT16, {180, 192},
                  ROW_IDX_TYPE_GATHER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto, false, 0, false,
                  ge::GRAPH_SUCCESS);
}

// HIF8 pertoken fullload
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_hif8_pertoken)
{
    int64_t h = 83;
    RunV4Testcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_HIF8_PERTOKEN, ge::DT_BF16, {180, 192},
                  ROW_IDX_TYPE_SCATTER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto, false, 0, false,
                  ge::GRAPH_SUCCESS);
}

// MXFP8 multicore
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_mxfp8_multicore)
{
    int64_t h = 60;
    RunV4Testcase(8, h, 32, 0, 0, EXPERT_TOKENS_TYPE_KEY_VALUE, true, QUANT_MODE_MXFP8_E5M2, ge::DT_FLOAT16, {0, 100},
                  ROW_IDX_TYPE_SCATTER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto, false, 0, false,
                  ge::GRAPH_SUCCESS);
}

// static quant missing scale -> fail
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_static_missing_scale)
{
    RunV4Testcase(1, 83, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_STATIC, ge::DT_FLOAT, {180, 192},
                  ROW_IDX_TYPE_GATHER, {}, ge::DT_FLOAT, {1}, ge::DT_FLOAT, kExpandedXDtypeAuto, false, 0, false,
                  ge::GRAPH_FAILED);
}

// INT4 dynamic with odd cols -> fail
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_int4_odd_cols)
{
    RunV4Testcase(1, 83, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_INT4_DYNAMIC, ge::DT_FLOAT, {180, 192},
                  ROW_IDX_TYPE_GATHER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto, false, 0, false,
                  ge::GRAPH_FAILED);
}

// MXFP4 unsupported x dtype -> fail
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_mxfp4_bad_xdtype)
{
    RunV4Testcase(1, 83, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_MXFP4_E2M1, ge::DT_FLOAT, {180, 192},
                  ROW_IDX_TYPE_GATHER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto, false, 0, false,
                  ge::GRAPH_FAILED);
}

// INT4 dynamic with float16 x -> fail
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_int4_bad_xdtype)
{
    RunV4Testcase(1, 84, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_INT4_DYNAMIC, ge::DT_FLOAT16, {180, 192},
                  ROW_IDX_TYPE_GATHER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto, false, 0, false,
                  ge::GRAPH_FAILED);
}

// UNQUANT FP8 with wrong scale dtype -> fail
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_unquant_fp8_bad_scale_dtype)
{
    RunV4Testcase(1, 83, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_UNQUANT, ge::DT_FLOAT8_E5M2, {180, 192},
                  ROW_IDX_TYPE_GATHER, {1, 2, 2}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto, false, 0, false,
                  ge::GRAPH_FAILED);
}

// empty tensor: n==0
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_empty_tensor_n_zero)
{
    RunV4Testcase(0, 128, 8, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_UNQUANT, ge::DT_FLOAT, {0, EXPERT_NUM},
                  ROW_IDX_TYPE_GATHER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto, false, 0, false,
                  ge::GRAPH_SUCCESS);
}

// empty tensor: k==0
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_empty_tensor_k_zero)
{
    RunV4Testcase(3, 128, 0, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_UNQUANT, ge::DT_FLOAT, {0, EXPERT_NUM},
                  ROW_IDX_TYPE_GATHER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto, false, 0, false,
                  ge::GRAPH_SUCCESS);
}

// empty tensor: cols==0
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_empty_tensor_cols_zero)
{
    RunV4Testcase(3, 0, 8, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_UNQUANT, ge::DT_FLOAT, {0, EXPERT_NUM},
                  ROW_IDX_TYPE_GATHER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto, false, 0, false,
                  ge::GRAPH_SUCCESS);
}

// empty tensor: k==0 + key_value mode
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_empty_tensor_k_zero_keyvalue)
{
    RunV4Testcase(3, 64, 0, 0, 0, EXPERT_TOKENS_TYPE_KEY_VALUE, true, QUANT_MODE_UNQUANT, ge::DT_FLOAT16,
                  {0, EXPERT_NUM}, ROW_IDX_TYPE_SCATTER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto, false,
                  0, false, ge::GRAPH_SUCCESS);
}

// V4 specific: active_num as input tensor present
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_active_num_input_present)
{
    int64_t h = 83;
    RunV4Testcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_UNQUANT, ge::DT_FLOAT16, {180, 192},
                  ROW_IDX_TYPE_GATHER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto, true, 27, false,
                  ge::GRAPH_SUCCESS);
}

// V4 specific: active_num as input tensor present + topk_weight present
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_active_num_and_topk_weight)
{
    int64_t h = 83;
    RunV4Testcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_UNQUANT, ge::DT_FLOAT16, {180, 192},
                  ROW_IDX_TYPE_GATHER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto, true, 27, true,
                  ge::GRAPH_SUCCESS);
}

// V4 specific: fullload + not quant + active_num + topk_weight + scatter
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_fullload_scatter_with_active_and_topkweight)
{
    int64_t h = 83;
    RunV4Testcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_UNQUANT, ge::DT_FLOAT, {180, 192},
                  ROW_IDX_TYPE_SCATTER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto, true, 27, true,
                  ge::GRAPH_SUCCESS);
}

// V4 specific: multicore + dynamic quant + active_num input + topk_weight
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_multicore_dynamic_active_topkweight)
{
    int64_t h = 60;
    RunV4Testcase(8, h, 32, 0, 0, EXPERT_TOKENS_TYPE_KEY_VALUE, true, QUANT_MODE_DYNAMIC, ge::DT_BF16, {0, 100},
                  ROW_IDX_TYPE_SCATTER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto, true, 256, true,
                  ge::GRAPH_SUCCESS);
}

// V4 specific: static quant + active_num + topk_weight + scale + offset
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_static_active_topkweight_scale_offset)
{
    int64_t h = 83;
    RunV4Testcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_STATIC, ge::DT_FLOAT, {180, 192},
                  ROW_IDX_TYPE_GATHER, {1}, ge::DT_FLOAT, {1}, ge::DT_FLOAT, kExpandedXDtypeAuto, true, 27, true,
                  ge::GRAPH_SUCCESS);
}

// V4 specific: MXFP8 + active_num + topk_weight
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_mxfp8_active_topkweight)
{
    int64_t h = 83;
    RunV4Testcase(1, h, 27, 0, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_MXFP8_E5M2, ge::DT_FLOAT16, {180, 192},
                  ROW_IDX_TYPE_GATHER, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, kExpandedXDtypeAuto, true, 27, true,
                  ge::GRAPH_SUCCESS);
}

// topk_weight failure cases
TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_topk_weight_invalid_dtype)
{
    int64_t h = 83;
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 262144, platform_ascendc::SocVersion::ASCEND950};
    int64_t n = 1;
    int64_t k = 27;
    int64_t totalLength = n * k;
    auto inputDesc =
        BuildV4Inputs(n, h, k, ge::DT_FLOAT16, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, false, 0, true, totalLength);
    auto outputDesc = BuildV4Outputs(totalLength, h, ge::DT_FLOAT16, {EXPERT_NUM - 180}, {{totalLength}, ge::DT_FLOAT},
                                     true, n, k, 0, EXPERT_NUM, 0);
    auto attrs = BuildV4Attrs(0, EXPERT_NUM, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_UNQUANT, {180, 192},
                              ROW_IDX_TYPE_GATHER);
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV4", inputDesc, outputDesc, attrs, &compileInfo,
                                              "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_topk_weight_1d)
{
    int64_t h = 83;
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 262144, platform_ascendc::SocVersion::ASCEND950};
    int64_t n = 1;
    int64_t k = 27;
    int64_t totalLength = n * k;
    auto inputDesc =
        BuildV4Inputs(n, h, k, ge::DT_FLOAT16, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, false, 0, true, totalLength);
    inputDesc[5] = {gert::StorageShape({totalLength}, {totalLength}), ge::DT_FLOAT, ge::FORMAT_ND};
    auto outputDesc = BuildV4Outputs(totalLength, h, ge::DT_FLOAT16, {EXPERT_NUM - 180}, {{totalLength}, ge::DT_FLOAT},
                                     true, n, k, 0, EXPERT_NUM, 0);
    auto attrs = BuildV4Attrs(0, EXPERT_NUM, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_UNQUANT, {180, 192},
                              ROW_IDX_TYPE_GATHER);
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV4", inputDesc, outputDesc, attrs, &compileInfo,
                                              "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_topk_weight_inconsistent)
{
    int64_t h = 83;
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 262144, platform_ascendc::SocVersion::ASCEND950};
    int64_t n = 1;
    int64_t k = 27;
    int64_t totalLength = n * k;
    auto inputDesc =
        BuildV4Inputs(n, h, k, ge::DT_FLOAT16, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, false, 0, true, totalLength);
    auto outputDesc = BuildV4Outputs(totalLength, h, ge::DT_FLOAT16, {EXPERT_NUM - 180}, {{totalLength}, ge::DT_FLOAT},
                                     false, n, k, 0, EXPERT_NUM, 0);
    auto attrs = BuildV4Attrs(0, EXPERT_NUM, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_UNQUANT, {180, 192},
                              ROW_IDX_TYPE_GATHER);
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV4", inputDesc, outputDesc, attrs, &compileInfo,
                                              "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV4Tiling, moe_init_routing_v4_tiling_active_num_invalid_dtype)
{
    int64_t h = 83;
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 262144, platform_ascendc::SocVersion::ASCEND950};
    int64_t n = 1;
    int64_t k = 27;
    int64_t totalLength = n * k;
    auto inputDesc =
        BuildV4Inputs(n, h, k, ge::DT_FLOAT16, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, false, 0, false, totalLength);
    inputDesc[4] = {gert::StorageShape({1}, {1}), ge::DT_FLOAT, ge::FORMAT_ND};
    auto outputDesc = BuildV4Outputs(totalLength, h, ge::DT_FLOAT16, {EXPERT_NUM - 180}, {{totalLength}, ge::DT_FLOAT},
                                     false, n, k, 0, EXPERT_NUM, 0);
    auto attrs = BuildV4Attrs(0, EXPERT_NUM, 0, EXPERT_TOKENS_TYPE_COUNT, true, QUANT_MODE_UNQUANT, {180, 192},
                              ROW_IDX_TYPE_GATHER);
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV4", inputDesc, outputDesc, attrs, &compileInfo,
                                              "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
