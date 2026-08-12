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

#include "../../../op_host/op_tiling/arch35/grouped_weight_quant_batch_matmul_tiling.h"
#include "../../../op_kernel/arch35/grouped_matmul_tiling_data_apt.h"
#include "../../../op_kernel/arch35/weight_quant_basic_block/weight_quant_tiling_key.h"
#include "gmm_csv_ge_parse_utils.h"
#include "tiling_case_executor.h"

namespace {
using TensorDescription = gert::TilingContextPara::TensorDescription;
using OpAttr = gert::TilingContextPara::OpAttr;
using S8S4TilingData = GroupedMatmulTilingData::GMMS8S4BasicApiTilingData;

constexpr uint64_t SYS_WORKSPACE_SIZE = 16UL * 1024UL * 1024UL;
constexpr uint64_t S8S4_ND_PERGROUP_TILING_KEY = 0x20600300UL;
constexpr uint64_t S8S4_ND_PERCHANNEL_TILING_KEY = 0x20600200UL;
constexpr uint64_t S8S4_NZ_PERGROUP_TILING_KEY = 0x20600301UL;
constexpr uint64_t S8S4_NZ_PERCHANNEL_TILING_KEY = 0x20600201UL;

uint64_t AlignWorkspace(uint64_t value)
{
    constexpr uint64_t align = 512UL;
    return (value + align - 1UL) / align * align;
}

static_assert(sizeof(S8S4TilingData) % sizeof(uint64_t) == 0);

uint64_t GetExpectedS8S4TilingKey(ge::Format weightFormat, uint64_t cQuantType)
{
    if (weightFormat == ge::FORMAT_FRACTAL_NZ) {
        return cQuantType == WQGMM_PER_GROUP ? S8S4_NZ_PERGROUP_TILING_KEY :
                                              S8S4_NZ_PERCHANNEL_TILING_KEY;
    }
    return cQuantType == WQGMM_PER_GROUP ? S8S4_ND_PERGROUP_TILING_KEY :
                                          S8S4_ND_PERCHANNEL_TILING_KEY;
}

struct S8S4Case {
    int64_t m = 512;
    int64_t k = 1024;
    int64_t n = 256;
    int64_t groupNum = 2;
    int64_t scaleGroupNum = 4;
    int64_t groupListTypeAttr = 1;
    ge::DataType weightDtype = ge::DT_INT4;
    ge::DataType outDtype = ge::DT_BF16;
    ge::Format weightFormat = ge::FORMAT_ND;
    bool hasBias = true;
    bool hasOffset = false;
    bool hasAntiquantScale = false;
    bool hasAntiquantOffset = false;
    bool specialWeightFormat = false;
    bool invalidPackedStorageShape = false;
    bool apiNormalizedPackedInt32 = false;
    bool multiBias = false;
    bool multiOffset = false;
    bool multiOutput = false;
    int64_t splitItem = 3;
    std::vector<int64_t> tuningConfig = {0};
};

gert::StorageShape EmptyShape()
{
    return gert::StorageShape();
}

TensorDescription MakeTensor(const std::vector<int64_t> &shape, ge::DataType dtype,
                             ge::Format format = ge::FORMAT_ND)
{
    return {ops::ut::MakeGertStorageShape(shape), dtype, format};
}

TensorDescription MakeTensor(const std::vector<int64_t> &originShape, const std::vector<int64_t> &storageShape,
                             ge::DataType dtype, ge::Format format)
{
    return {ops::ut::MakeGertStorageShape(originShape, storageShape), dtype, format};
}

std::vector<OpAttr> MakeAttrs(const S8S4Case &param)
{
    return {
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.splitItem)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.groupListTypeAttr)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(param.tuningConfig)},
    };
}

optiling::GMMCompileInfo MakeCompileInfo()
{
    return {
        32,                                      // aicNum
        64,                                      // aivNum
        262144,                                  // ubSize
        524288,                                  // l1Size
        134217728,                               // l2Size
        262144,                                  // l0CSize
        65536,                                   // l0ASize
        65536,                                   // l0BSize
        platform_ascendc::SocVersion::ASCEND950, // socVersion
        NpuArch::DAV_3510,
    };
}

gert::TilingContextPara MakeContext(const S8S4Case &param, optiling::GMMCompileInfo *compileInfo)
{
    const std::string socInfoString =
        std::string(R"({"hardware_info":{"BT_SIZE":0,"load3d_constraints":"1",)"
                    R"("Intrinsic_fix_pipe_l0c2out":false,"Intrinsic_data_move_l12ub":true,)"
                    R"("Intrinsic_data_move_l0c2ub":true,"Intrinsic_data_move_out2l1_nd2nz":false,)"
                    R"("UB_SIZE":)") +
        std::to_string(compileInfo->ubSize) + R"(,"L2_SIZE":)" + std::to_string(compileInfo->l2Size) +
        R"(,"L1_SIZE":)" + std::to_string(compileInfo->l1Size) + R"(,"L0A_SIZE":)" +
        std::to_string(compileInfo->l0ASize) + R"(,"L0B_SIZE":)" + std::to_string(compileInfo->l0BSize) +
        R"(,"L0C_SIZE":)" + std::to_string(compileInfo->l0CSize) + R"(,"CORE_NUM":)" +
        std::to_string(compileInfo->aicNum) + R"(,"socVersion":"3510"}})";
    std::vector<int64_t> weightOriginShape =
        param.specialWeightFormat ?
            std::vector<int64_t>{param.groupNum, param.n, param.k} :
            std::vector<int64_t>{param.groupNum, param.k, param.n};
    std::vector<int64_t> weightStorageShape = weightOriginShape;
    if (param.weightFormat == ge::FORMAT_FRACTAL_NZ) {
        weightStorageShape =
            param.specialWeightFormat ?
                std::vector<int64_t>{param.groupNum, (param.k + 31) / 32, (param.n + 15) / 16, 16, 32} :
                std::vector<int64_t>{param.groupNum, (param.n + 31) / 32, (param.k + 15) / 16, 16, 32};
        if (param.weightDtype == ge::DT_INT32) {
            weightStorageShape.back() /= 8;
        }
    } else if (param.weightDtype == ge::DT_INT32 || param.apiNormalizedPackedInt32) {
        weightStorageShape.back() /= 8;
    }
    if (param.invalidPackedStorageShape) {
        ++weightStorageShape.back();
    }

    const gert::StorageShape offsetShape =
        param.hasOffset
            ? ops::ut::MakeGertStorageShape(std::vector<int64_t>{param.groupNum, 1, param.n})
            : EmptyShape();
    const gert::StorageShape biasShape =
        param.hasBias ? ops::ut::MakeGertStorageShape(std::vector<int64_t>{param.groupNum, param.n}) : EmptyShape();
    const gert::StorageShape antiquantScaleShape =
        param.hasAntiquantScale
            ? ops::ut::MakeGertStorageShape(std::vector<int64_t>{param.groupNum, 1, param.n})
            : EmptyShape();
    const gert::StorageShape antiquantOffsetShape =
        param.hasAntiquantOffset
            ? ops::ut::MakeGertStorageShape(std::vector<int64_t>{param.groupNum, 1, param.n})
            : EmptyShape();
    std::vector<TensorDescription> inputs = {
        MakeTensor({param.m, param.k}, ge::DT_INT8), // x
        MakeTensor(weightOriginShape, weightStorageShape, param.weightDtype, param.weightFormat),
        {biasShape, ge::DT_FLOAT, ge::FORMAT_ND}, // bias
    };
    if (param.multiBias) {
        inputs.emplace_back(biasShape, ge::DT_FLOAT, ge::FORMAT_ND);
    }
    inputs.insert(inputs.end(), {
        MakeTensor({param.groupNum, param.scaleGroupNum, param.n}, ge::DT_UINT64), // scale
        {offsetShape, ge::DT_FLOAT, ge::FORMAT_ND},                                 // offset
    });
    if (param.multiOffset) {
        inputs.emplace_back(offsetShape, ge::DT_FLOAT, ge::FORMAT_ND);
    }
    inputs.insert(inputs.end(), {
        {antiquantScaleShape, ge::DT_FLOAT, ge::FORMAT_ND},                        // antiquantScale
        {antiquantOffsetShape, ge::DT_FLOAT, ge::FORMAT_ND},                       // antiquantOffset
        MakeTensor(param.groupListTypeAttr == 2 ? std::vector<int64_t>{param.groupNum, 2} :
                                                  std::vector<int64_t>{param.groupNum},
                   ge::DT_INT64),                                                  // groupList
        MakeTensor({param.m}, ge::DT_FLOAT),                                       // perTokenScale
    });
    std::vector<TensorDescription> outputs = {
        MakeTensor({param.m, param.n}, param.outDtype),
    };
    if (param.multiOutput) {
        outputs.emplace_back(MakeTensor({param.m, param.n}, param.outDtype));
    }
    if (param.multiBias || param.multiOffset || param.multiOutput) {
        std::vector<TensorDescription> flatInputs = {
            MakeTensor({param.m, param.k}, ge::DT_INT8),
            MakeTensor(weightOriginShape, weightStorageShape, param.weightDtype, param.weightFormat),
            {biasShape, ge::DT_FLOAT, ge::FORMAT_ND},
        };
        if (param.multiBias) {
            flatInputs.emplace_back(biasShape, ge::DT_FLOAT, ge::FORMAT_ND);
        }
        flatInputs.emplace_back(MakeTensor({param.groupNum, param.scaleGroupNum, param.n}, ge::DT_UINT64));
        if (param.hasOffset) {
            flatInputs.emplace_back(offsetShape, ge::DT_FLOAT, ge::FORMAT_ND);
            if (param.multiOffset) {
                flatInputs.emplace_back(offsetShape, ge::DT_FLOAT, ge::FORMAT_ND);
            }
        }
        if (param.hasAntiquantScale) {
            flatInputs.emplace_back(antiquantScaleShape, ge::DT_FLOAT, ge::FORMAT_ND);
        }
        if (param.hasAntiquantOffset) {
            flatInputs.emplace_back(antiquantOffsetShape, ge::DT_FLOAT, ge::FORMAT_ND);
        }
        flatInputs.emplace_back(MakeTensor(param.groupListTypeAttr == 2 ?
                                               std::vector<int64_t>{param.groupNum, 2} :
                                               std::vector<int64_t>{param.groupNum},
                                           ge::DT_INT64));
        flatInputs.emplace_back(MakeTensor({param.m}, ge::DT_FLOAT));
        const std::vector<uint32_t> inputInstanceNum = {
            1U, 1U, param.multiBias ? 2U : 1U, 1U,
            param.hasOffset ? (param.multiOffset ? 2U : 1U) : 0U,
            param.hasAntiquantScale ? 1U : 0U, param.hasAntiquantOffset ? 1U : 0U, 1U, 1U};
        const std::vector<uint32_t> outputInstanceNum = {param.multiOutput ? 2U : 1U};
        return gert::TilingContextPara("GroupedMatmul", flatInputs, outputs, MakeAttrs(param), inputInstanceNum,
                                       outputInstanceNum, compileInfo, "3510", 32, 262144, 4096, socInfoString);
    }
    return gert::TilingContextPara("GroupedMatmul", inputs, outputs, MakeAttrs(param), compileInfo, "3510", 32,
                                   262144, 4096, socInfoString);
}

bool ExecuteS8S4Tiling(const S8S4Case &param, TilingInfo &tilingInfo)
{
    auto compileInfo = MakeCompileInfo();
    auto context = MakeContext(param, &compileInfo);
    return ExecuteTiling(context, tilingInfo);
}

bool ExecuteS8S4TilingWithCompileInfo(
    const S8S4Case &param, optiling::GMMCompileInfo &compileInfo, TilingInfo &tilingInfo)
{
    auto context = MakeContext(param, &compileInfo);
    return ExecuteTiling(context, tilingInfo);
}

const S8S4TilingData *GetTilingData(const TilingInfo &tilingInfo)
{
    EXPECT_EQ(tilingInfo.tilingDataSize, sizeof(S8S4TilingData));
    return reinterpret_cast<const S8S4TilingData *>(tilingInfo.tilingData.get());
}

TEST(GroupedS8S4BasicApiArch35Tiling, ExactTilingKeys)
{
    EXPECT_EQ(
        GET_TPL_TILING_KEY(
            WQGMM_ND, WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST, WQGMM_PER_GROUP, WQGMM_NONE,
            WQGMM_NO_TRANS, WQGMM_NO_TRANS, WQGMM_S8S4_FIXED_TEMPLATE, WQGMM_NOT_SINGLE_MULTI_SINGLE,
            WQGMM_VDEFAULT, WQGMM_MULTI_SCALE_DEQUANT),
        S8S4_ND_PERGROUP_TILING_KEY);
    EXPECT_EQ(
        GET_TPL_TILING_KEY(
            WQGMM_ND, WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST, WQGMM_PER_CHANNEL, WQGMM_NONE,
            WQGMM_NO_TRANS, WQGMM_NO_TRANS, WQGMM_S8S4_FIXED_TEMPLATE, WQGMM_NOT_SINGLE_MULTI_SINGLE,
            WQGMM_VDEFAULT, WQGMM_MULTI_SCALE_DEQUANT),
        S8S4_ND_PERCHANNEL_TILING_KEY);
    EXPECT_EQ(
        GET_TPL_TILING_KEY(
            WQGMM_FRACTAL_NZ, WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST, WQGMM_PER_GROUP, WQGMM_NONE,
            WQGMM_NO_TRANS, WQGMM_NO_TRANS, WQGMM_S8S4_FIXED_TEMPLATE, WQGMM_NOT_SINGLE_MULTI_SINGLE,
            WQGMM_VDEFAULT, WQGMM_MULTI_SCALE_DEQUANT),
        S8S4_NZ_PERGROUP_TILING_KEY);
    EXPECT_EQ(
        GET_TPL_TILING_KEY(
            WQGMM_FRACTAL_NZ, WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST, WQGMM_PER_CHANNEL, WQGMM_NONE,
            WQGMM_NO_TRANS, WQGMM_NO_TRANS, WQGMM_S8S4_FIXED_TEMPLATE, WQGMM_NOT_SINGLE_MULTI_SINGLE,
            WQGMM_VDEFAULT, WQGMM_MULTI_SCALE_DEQUANT),
        S8S4_NZ_PERCHANNEL_TILING_KEY);
}

TEST(GroupedS8S4BasicApiArch35Tiling, V5SymmetricPergroupUsesProductionRouter)
{
    S8S4Case param;

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteS8S4Tiling(param, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, S8S4_ND_PERGROUP_TILING_KEY);
    const auto *tilingData = GetTilingData(tilingInfo);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->gmmQuantParams.bQuantMode,
              static_cast<uint32_t>(optiling::QuantMode::PERGROUP_MODE));
}

TEST(GroupedS8S4BasicApiArch35Tiling, MapsA3CompatibleWeightFormatsToNd)
{
    for (const ge::Format weightFormat : {ge::FORMAT_ND, ge::FORMAT_NCL, ge::FORMAT_NCHW}) {
        S8S4Case param;
        param.weightFormat = weightFormat;

        TilingInfo tilingInfo;
        ASSERT_TRUE(ExecuteS8S4Tiling(param, tilingInfo)) << "weightFormat=" << weightFormat;
        EXPECT_EQ(tilingInfo.tilingKey, S8S4_ND_PERGROUP_TILING_KEY);
    }
}

TEST(GroupedS8S4BasicApiArch35Tiling, RejectsWeightFormatOutsideA3Whitelist)
{
    S8S4Case param;
    param.weightFormat = ge::FORMAT_NHWC;

    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteS8S4Tiling(param, tilingInfo));
}

TEST(GroupedS8S4BasicApiArch35Tiling, V5AsymmetricPerchannelUsesProductionRouter)
{
    S8S4Case param;
    param.k = 1001;
    param.scaleGroupNum = 1;
    param.hasOffset = true;
    param.outDtype = ge::DT_FLOAT16;

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteS8S4Tiling(param, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, S8S4_ND_PERCHANNEL_TILING_KEY);
    const auto *tilingData = GetTilingData(tilingInfo);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->gmmQuantParams.bQuantMode,
              static_cast<uint32_t>(optiling::QuantMode::PERCHANNEL_MODE));
    EXPECT_EQ(tilingData->mmTilingData.k, 1001U);
}

TEST(GroupedS8S4BasicApiArch35Tiling, V5ApiNormalizedPackedInt32UsesProductionRouter)
{
    S8S4Case param;
    // This is the descriptor state produced by UnpackB32ToB4 for an ND packed-INT32 input:
    // logical dtype/shape are INT4 [E,K,N], while storage retains [E,K,N/8].
    param.apiNormalizedPackedInt32 = true;

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteS8S4Tiling(param, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, S8S4_ND_PERGROUP_TILING_KEY);
    const auto *tilingData = GetTilingData(tilingInfo);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->s8s4Params.weightPackedInt32, 1U);
}

TEST(GroupedS8S4BasicApiArch35Tiling, SymmetricPergroupUsesOneQuantGroupPerL1Window)
{
    S8S4Case param;
    param.groupListTypeAttr = 1;

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteS8S4Tiling(param, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, GetExpectedS8S4TilingKey(param.weightFormat, WQGMM_PER_GROUP));
    const auto *tilingData = GetTilingData(tilingInfo);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->gmmQuantParams.groupListType, 1);
    EXPECT_EQ(tilingData->gmmQuantParams.bQuantMode,
              static_cast<uint32_t>(optiling::QuantMode::PERGROUP_MODE));
    EXPECT_EQ(tilingData->s8s4Params.quantGroupSize, 256U);
    EXPECT_EQ(tilingData->s8s4Params.quantGroupNum, 4U);
    EXPECT_EQ(tilingData->s8s4Params.dequantMode, 0U);
    EXPECT_EQ(tilingData->mmTilingData.baseM, 256U);
    EXPECT_EQ(tilingData->mmTilingData.baseN, 256U);
    EXPECT_EQ(tilingData->mmTilingData.baseK, 128U);
    EXPECT_EQ(tilingData->mmTilingData.kAL1, 256U);
    EXPECT_EQ(tilingData->mmTilingData.kBL1, 256U);
    EXPECT_EQ(tilingData->mmTilingData.dbL0C, 1U);
    EXPECT_EQ(tilingData->s8s4Params.coreNum, 32U);
    const uint64_t expandedSize = AlignWorkspace(static_cast<uint64_t>(param.groupNum * param.k * param.n));
    const uint64_t tileStride = AlignWorkspace(256UL * 256UL * sizeof(uint16_t));
    const uint64_t tileSize = 32UL * tileStride;
    EXPECT_EQ(tilingData->s8s4Params.enableWeightPreprocess, 1U);
    EXPECT_EQ(tilingData->s8s4Params.expandedWeightOffsetBytes, 0U);
    EXPECT_EQ(tilingData->s8s4Params.expandedWeightSizeBytes, expandedSize);
    EXPECT_EQ(tilingData->s8s4Params.tileWorkspaceOffsetBytes, expandedSize);
    EXPECT_EQ(tilingData->s8s4Params.tileWorkspaceStrideBytes, tileStride);
    EXPECT_EQ(tilingData->s8s4Params.tileWorkspaceSizeBytes, tileSize);
    EXPECT_EQ(tilingData->s8s4Params.rowSumSizeBytes, 0U);
    EXPECT_EQ(tilingData->s8s4Params.userWorkspaceSizeBytes, expandedSize + tileSize);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], SYS_WORKSPACE_SIZE + expandedSize + tileSize);
}

TEST(GroupedS8S4BasicApiArch35Tiling, AcceptsCountGroupListType)
{
    S8S4Case param;
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteS8S4Tiling(param, tilingInfo));
    const auto *tilingData = GetTilingData(tilingInfo);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->gmmQuantParams.groupListType, 1);
}

TEST(GroupedS8S4BasicApiArch35Tiling, RejectsUnsupportedGroupListTypes)
{
    for (int64_t groupListType : {0L, 2L, 99L}) {
        S8S4Case param;
        param.groupListTypeAttr = groupListType;
        TilingInfo tilingInfo;
        EXPECT_FALSE(ExecuteS8S4Tiling(param, tilingInfo)) << "groupListType=" << groupListType;
    }
}

TEST(GroupedS8S4BasicApiArch35Tiling, AsymmetricPerchannelUsesOffsetAndTilePipelineWorkspace)
{
    S8S4Case param;
    param.hasOffset = true;
    param.scaleGroupNum = 1;
    param.outDtype = ge::DT_FLOAT16;

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteS8S4Tiling(param, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, GetExpectedS8S4TilingKey(param.weightFormat, WQGMM_PER_CHANNEL));
    const auto *tilingData = GetTilingData(tilingInfo);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->gmmQuantParams.bQuantMode,
              static_cast<uint32_t>(optiling::QuantMode::PERCHANNEL_MODE));
    EXPECT_EQ(tilingData->s8s4Params.quantGroupNum, 1U);
    EXPECT_EQ(tilingData->s8s4Params.dequantMode, 1U);
    EXPECT_EQ(tilingData->s8s4Params.hasOffset, 1U);
    EXPECT_EQ(tilingData->s8s4Params.enableWeightPreprocess, 1U);
    EXPECT_EQ(tilingData->s8s4Params.expandedWeightStrideBytes,
              static_cast<uint64_t>(param.k * param.n));
    EXPECT_EQ(tilingData->mmTilingData.baseM, 256U);
    EXPECT_EQ(tilingData->mmTilingData.kAL1, 256U);
    EXPECT_EQ(tilingData->mmTilingData.kBL1, 256U);
    EXPECT_EQ(tilingData->mmTilingData.dbL0C, 1U);
    EXPECT_EQ(tilingData->s8s4Params.coreNum, 32U);
    const uint64_t expandedSize = AlignWorkspace(static_cast<uint64_t>(param.groupNum * param.k * param.n));
    const uint64_t tileStride = AlignWorkspace(256UL * 256UL * sizeof(uint16_t));
    const uint64_t tileSize = 32UL * tileStride;
    const uint64_t rowSumSize = AlignWorkspace(static_cast<uint64_t>(param.m) * sizeof(float));
    EXPECT_EQ(tilingData->s8s4Params.expandedWeightOffsetBytes, 0U);
    EXPECT_EQ(tilingData->s8s4Params.expandedWeightSizeBytes, expandedSize);
    EXPECT_EQ(tilingData->s8s4Params.tileWorkspaceOffsetBytes, expandedSize);
    EXPECT_EQ(tilingData->s8s4Params.tileWorkspaceStrideBytes, tileStride);
    EXPECT_EQ(tilingData->s8s4Params.tileWorkspaceSizeBytes, tileSize);
    EXPECT_EQ(tilingData->s8s4Params.rowSumOffsetBytes, expandedSize + tileSize);
    EXPECT_EQ(tilingData->s8s4Params.rowSumSizeBytes, rowSumSize);
    EXPECT_EQ(tilingData->s8s4Params.userWorkspaceSizeBytes, expandedSize + tileSize + rowSumSize);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], SYS_WORKSPACE_SIZE + expandedSize + tileSize + rowSumSize);
}

TEST(GroupedS8S4BasicApiArch35Tiling, FixedTilePhysicalFootprintSeparatesBAndScaleForSmallN)
{
    S8S4Case param;
    param.m = 16;
    param.k = 512;
    param.n = 16;
    param.scaleGroupNum = 1;
    param.hasOffset = true;
    param.outDtype = ge::DT_FLOAT16;

    auto compileInfo = MakeCompileInfo();
    compileInfo.l1Size = 2UL * 12416UL;
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteS8S4TilingWithCompileInfo(param, compileInfo, tilingInfo));

    compileInfo.l1Size--;
    EXPECT_FALSE(ExecuteS8S4TilingWithCompileInfo(param, compileInfo, tilingInfo));
}

TEST(GroupedS8S4BasicApiArch35Tiling, PerchannelNTailsUseLogicalTileAndPhysicalNzFootprint)
{
    for (const int64_t n : {8L, 16L, 24L, 32L, 248L, 250L, 256L}) {
        S8S4Case param;
        param.m = 8;
        param.k = 512;
        param.n = n;
        param.scaleGroupNum = 1;
        param.hasOffset = true;
        param.outDtype = ge::DT_FLOAT16;

        TilingInfo tilingInfo;
        ASSERT_TRUE(ExecuteS8S4Tiling(param, tilingInfo)) << "N=" << n;
        const auto *tilingData = GetTilingData(tilingInfo);
        ASSERT_NE(tilingData, nullptr);
        const uint64_t expectedBaseN =
            GroupedMatmul::CeilAlign(static_cast<uint64_t>(n), optiling::GmmConstant::CUBE_BLOCK);
        EXPECT_EQ(tilingData->mmTilingData.baseM, 16U);
        EXPECT_EQ(tilingData->mmTilingData.baseN, expectedBaseN);
    }
}

TEST(GroupedS8S4BasicApiArch35Tiling, MaximumFixedTileMatchesA5ResourceBudget)
{
    S8S4Case param;
    param.scaleGroupNum = 1;
    param.hasOffset = true;
    param.outDtype = ge::DT_FLOAT16;

    auto compileInfo = MakeCompileInfo();
    compileInfo.l1Size = 260UL * 1024UL;
    compileInfo.l0ASize = 64UL * 1024UL;
    compileInfo.l0BSize = 64UL * 1024UL;
    compileInfo.l0CSize = 256UL * 1024UL;
    compileInfo.ubSize = 130UL * 1024UL;
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteS8S4TilingWithCompileInfo(param, compileInfo, tilingInfo));
}

TEST(GroupedS8S4BasicApiArch35Tiling, PergroupValidatesFixedUbFootprint)
{
    S8S4Case param;

    auto compileInfo = MakeCompileInfo();
    compileInfo.ubSize = 130UL * 1024UL;
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteS8S4TilingWithCompileInfo(param, compileInfo, tilingInfo));

    compileInfo.ubSize--;
    EXPECT_FALSE(ExecuteS8S4TilingWithCompileInfo(param, compileInfo, tilingInfo));
}

TEST(GroupedS8S4BasicApiArch35Tiling, FixedTileRejectsInsufficientOnChipMemory)
{
    S8S4Case param;
    param.hasOffset = true;
    param.scaleGroupNum = 1;
    param.outDtype = ge::DT_FLOAT16;

    for (const int resource : {0, 1, 2, 3, 4}) {
        auto compileInfo = MakeCompileInfo();
        if (resource == 0) {
            compileInfo.l1Size = 260UL * 1024UL - 1UL;
        } else if (resource == 1) {
            compileInfo.l0ASize = 64UL * 1024UL - 1UL;
        } else if (resource == 2) {
            compileInfo.l0BSize = 64UL * 1024UL - 1UL;
        } else if (resource == 3) {
            compileInfo.l0CSize = 256UL * 1024UL - 1UL;
        } else {
            compileInfo.ubSize = 130UL * 1024UL - 1UL;
        }
        TilingInfo tilingInfo;
        EXPECT_FALSE(ExecuteS8S4TilingWithCompileInfo(param, compileInfo, tilingInfo))
            << "resource=" << resource;
    }
}

TEST(GroupedS8S4BasicApiArch35Tiling, AsymmetricPerchannelNzUsesNzTilingKey)
{
    S8S4Case param;
    param.k = 1001;
    param.n = 250;
    param.hasOffset = true;
    param.scaleGroupNum = 1;
    param.outDtype = ge::DT_FLOAT16;
    param.weightFormat = ge::FORMAT_FRACTAL_NZ;

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteS8S4Tiling(param, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, S8S4_NZ_PERCHANNEL_TILING_KEY);
    const auto *tilingData = GetTilingData(tilingInfo);
    ASSERT_NE(tilingData, nullptr);
    constexpr uint64_t expectedStride = 1008UL * 256UL;
    EXPECT_EQ(tilingData->s8s4Params.expandedWeightStrideBytes, expectedStride);
    const uint64_t expandedSize = AlignWorkspace(2UL * expectedStride);
    const uint64_t tileStride = AlignWorkspace(256UL * 256UL * sizeof(uint16_t));
    const uint64_t tileSize = 32UL * tileStride;
    const uint64_t rowSumSize = AlignWorkspace(static_cast<uint64_t>(param.m) * sizeof(float));
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], SYS_WORKSPACE_SIZE + expandedSize + tileSize + rowSumSize);
}

TEST(GroupedS8S4BasicApiArch35Tiling, PerchannelInt4AndPackedInt32ShareWorkspaceAndLogicalTiling)
{
    S8S4Case int4Param;
    int4Param.k = 1001;
    int4Param.scaleGroupNum = 1;
    int4Param.hasOffset = true;
    int4Param.outDtype = ge::DT_FLOAT16;
    S8S4Case packedParam = int4Param;
    packedParam.weightDtype = ge::DT_INT32;

    TilingInfo int4TilingInfo;
    TilingInfo packedTilingInfo;
    ASSERT_TRUE(ExecuteS8S4Tiling(int4Param, int4TilingInfo));
    ASSERT_TRUE(ExecuteS8S4Tiling(packedParam, packedTilingInfo));
    EXPECT_EQ(int4TilingInfo.tilingKey, packedTilingInfo.tilingKey);
    EXPECT_EQ(int4TilingInfo.workspaceSizes, packedTilingInfo.workspaceSizes);
    const auto *int4TilingData = GetTilingData(int4TilingInfo);
    const auto *packedTilingData = GetTilingData(packedTilingInfo);
    ASSERT_NE(int4TilingData, nullptr);
    ASSERT_NE(packedTilingData, nullptr);
    EXPECT_EQ(int4TilingData->s8s4Params.expandedWeightStrideBytes,
              packedTilingData->s8s4Params.expandedWeightStrideBytes);
    EXPECT_EQ(int4TilingData->mmTilingData.k, packedTilingData->mmTilingData.k);
    EXPECT_EQ(int4TilingData->mmTilingData.n, packedTilingData->mmTilingData.n);
}

TEST(GroupedS8S4BasicApiArch35Tiling, Float16WithoutOffsetStillUsesPergroup)
{
    S8S4Case param;
    param.outDtype = ge::DT_FLOAT16;

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteS8S4Tiling(param, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, GetExpectedS8S4TilingKey(param.weightFormat, WQGMM_PER_GROUP));
    const auto *tilingData = GetTilingData(tilingInfo);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->s8s4Params.dequantMode, 0U);
    EXPECT_EQ(tilingData->s8s4Params.quantGroupNum, 4U);
}

TEST(GroupedS8S4BasicApiArch35Tiling, PackedInt32KeepsLogicalOriginShape)
{
    S8S4Case param;
    param.weightDtype = ge::DT_INT32;

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteS8S4Tiling(param, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, GetExpectedS8S4TilingKey(param.weightFormat, WQGMM_PER_GROUP));
    const auto *tilingData = GetTilingData(tilingInfo);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->s8s4Params.weightPackedInt32, 1U);
    EXPECT_EQ(tilingData->mmTilingData.k, static_cast<uint32_t>(param.k));
    EXPECT_EQ(tilingData->mmTilingData.n, static_cast<uint32_t>(param.n));
}

TEST(GroupedS8S4BasicApiArch35Tiling, Int4AndPackedInt32ShareKeyAndLogicalTiling)
{
    S8S4Case int4Param;
    S8S4Case packedParam;
    packedParam.weightDtype = ge::DT_INT32;

    TilingInfo int4TilingInfo;
    TilingInfo packedTilingInfo;
    ASSERT_TRUE(ExecuteS8S4Tiling(int4Param, int4TilingInfo));
    ASSERT_TRUE(ExecuteS8S4Tiling(packedParam, packedTilingInfo));
    EXPECT_EQ(int4TilingInfo.tilingKey, packedTilingInfo.tilingKey);

    const auto *int4TilingData = GetTilingData(int4TilingInfo);
    const auto *packedTilingData = GetTilingData(packedTilingInfo);
    ASSERT_NE(int4TilingData, nullptr);
    ASSERT_NE(packedTilingData, nullptr);
    EXPECT_EQ(int4TilingData->mmTilingData.m, packedTilingData->mmTilingData.m);
    EXPECT_EQ(int4TilingData->mmTilingData.n, packedTilingData->mmTilingData.n);
    EXPECT_EQ(int4TilingData->mmTilingData.k, packedTilingData->mmTilingData.k);
    EXPECT_EQ(int4TilingData->s8s4Params.weightPackedInt32, 0U);
    EXPECT_EQ(packedTilingData->s8s4Params.weightPackedInt32, 1U);
}

TEST(GroupedS8S4BasicApiArch35Tiling, PackedInt32StorageIsEightToOne)
{
    S8S4Case param;
    param.weightDtype = ge::DT_INT32;
    auto compileInfo = MakeCompileInfo();
    const auto context = MakeContext(param, &compileInfo);
    const auto &weightShape = context.inputTensorDesc_[1].shape_;
    EXPECT_EQ(weightShape.GetOriginShape().GetDim(2), param.n);
    EXPECT_EQ(weightShape.GetStorageShape().GetDim(2), param.n / 8);
}

TEST(GroupedS8S4BasicApiArch35Tiling, PackedInt32NzStorageIsEightToOne)
{
    S8S4Case param;
    param.weightDtype = ge::DT_INT32;
    param.weightFormat = ge::FORMAT_FRACTAL_NZ;

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteS8S4Tiling(param, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, S8S4_NZ_PERGROUP_TILING_KEY);
    const auto *tilingData = GetTilingData(tilingInfo);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->s8s4Params.weightPackedInt32, 1U);

    auto compileInfo = MakeCompileInfo();
    const auto context = MakeContext(param, &compileInfo);
    const auto &storageShape = context.inputTensorDesc_[1].shape_.GetStorageShape();
    ASSERT_EQ(storageShape.GetDimNum(), 5U);
    EXPECT_EQ(storageShape.GetDim(1), (param.n + 31) / 32);
    EXPECT_EQ(storageShape.GetDim(2), (param.k + 15) / 16);
    EXPECT_EQ(storageShape.GetDim(3), 16);
    EXPECT_EQ(storageShape.GetDim(4), 4);
}

TEST(GroupedS8S4BasicApiArch35Tiling, RejectsMalformedPackedInt32StorageShape)
{
    S8S4Case param;
    param.weightDtype = ge::DT_INT32;
    param.invalidPackedStorageShape = true;

    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteS8S4Tiling(param, tilingInfo));
}

TEST(GroupedS8S4BasicApiArch35Tiling, RejectsSpecialEnkNzWithoutOffset)
{
    S8S4Case param;
    param.k = 2048;
    param.n = 2048;
    param.scaleGroupNum = param.k / 256;
    param.weightFormat = ge::FORMAT_FRACTAL_NZ;
    param.specialWeightFormat = true;
    param.tuningConfig = {0, 1};

    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteS8S4Tiling(param, tilingInfo));
}

TEST(GroupedS8S4BasicApiArch35Tiling, PerchannelSpecialEnkUsesStandardW8NzWorkspaceStride)
{
    S8S4Case param;
    param.k = 2050;
    param.n = 2040;
    param.scaleGroupNum = 1;
    param.hasOffset = true;
    param.outDtype = ge::DT_FLOAT16;
    param.weightFormat = ge::FORMAT_FRACTAL_NZ;
    param.specialWeightFormat = true;
    param.tuningConfig = {0, 1};

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteS8S4Tiling(param, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, S8S4_NZ_PERCHANNEL_TILING_KEY);
    const auto *tilingData = GetTilingData(tilingInfo);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->s8s4Params.specialWeightFormat, 1U);
    auto compileInfo = MakeCompileInfo();
    const auto context = MakeContext(param, &compileInfo);
    const auto &storageShape = context.inputTensorDesc_[1].shape_.GetStorageShape();
    ASSERT_EQ(storageShape.GetDimNum(), 5U);
    EXPECT_EQ(storageShape.GetDim(1), (param.k + 31) / 32);
    EXPECT_EQ(storageShape.GetDim(2), (param.n + 15) / 16);
    EXPECT_EQ(storageShape.GetDim(3), 16);
    EXPECT_EQ(storageShape.GetDim(4), 32);
    constexpr uint64_t expectedStandardNzStride = 2064UL * 2048UL;
    EXPECT_EQ(tilingData->s8s4Params.expandedWeightStrideBytes, expectedStandardNzStride);
    const uint64_t expandedSize = AlignWorkspace(2UL * expectedStandardNzStride);
    const uint64_t tileStride = AlignWorkspace(256UL * 256UL * sizeof(uint16_t));
    const uint64_t tileSize = 32UL * tileStride;
    const uint64_t rowSumSize = AlignWorkspace(static_cast<uint64_t>(param.m) * sizeof(float));
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], SYS_WORKSPACE_SIZE + expandedSize + tileSize + rowSumSize);
}

TEST(GroupedS8S4BasicApiArch35Tiling, ExpectedTokensKeepRequiredWeightPreprocess)
{
    S8S4Case param;
    param.tuningConfig = {128}; // 128 > N/4(64)

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteS8S4Tiling(param, tilingInfo));
    const auto *tilingData = GetTilingData(tilingInfo);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->s8s4Params.expectedTokenNum, 128U);
    EXPECT_EQ(tilingData->s8s4Params.enableWeightPreprocess, 1U);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_GT(tilingInfo.workspaceSizes[0], SYS_WORKSPACE_SIZE);
}

TEST(GroupedS8S4BasicApiArch35Tiling, RejectsWrongPergroupScaleShape)
{
    S8S4Case param;
    param.scaleGroupNum = 1;
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteS8S4Tiling(param, tilingInfo));
}

TEST(GroupedS8S4BasicApiArch35Tiling, RejectsBf16WithOffset)
{
    S8S4Case param;
    param.hasOffset = true;
    param.scaleGroupNum = 1;
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteS8S4Tiling(param, tilingInfo));
}

TEST(GroupedS8S4BasicApiArch35Tiling, RejectsKNotDivisibleBy256)
{
    S8S4Case param;
    param.k = 1152;
    param.scaleGroupNum = 4;
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteS8S4Tiling(param, tilingInfo));
}

TEST(GroupedS8S4BasicApiArch35Tiling, AsymmetricPerchannelAcceptsUnalignedLogicalK)
{
    S8S4Case param;
    param.k = 1001;
    param.scaleGroupNum = 1;
    param.hasOffset = true;
    param.outDtype = ge::DT_FLOAT16;

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteS8S4Tiling(param, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, S8S4_ND_PERCHANNEL_TILING_KEY);
    const auto *tilingData = GetTilingData(tilingInfo);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->mmTilingData.k, 1001U);
    EXPECT_EQ(tilingData->s8s4Params.quantGroupNum, 1U);
}

TEST(GroupedS8S4BasicApiArch35Tiling, AsymmetricPerchannelPadsSmallKTilingWindow)
{
    S8S4Case param;
    param.k = 127;
    param.scaleGroupNum = 1;
    param.hasOffset = true;
    param.outDtype = ge::DT_FLOAT16;

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteS8S4Tiling(param, tilingInfo));
    const auto *tilingData = GetTilingData(tilingInfo);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->mmTilingData.k, 127U);
    EXPECT_EQ(tilingData->mmTilingData.kAL1, 256U);
    EXPECT_EQ(tilingData->mmTilingData.kBL1, 256U);
}

TEST(GroupedS8S4BasicApiArch35Tiling, RejectsMissingBias)
{
    S8S4Case param;
    param.hasBias = false;
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteS8S4Tiling(param, tilingInfo));
}

TEST(GroupedS8S4BasicApiArch35Tiling, RejectsNonEmptyAntiquantScale)
{
    S8S4Case param;
    param.hasAntiquantScale = true;
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteS8S4Tiling(param, tilingInfo));
}

TEST(GroupedS8S4BasicApiArch35Tiling, RejectsNonEmptyAntiquantOffset)
{
    S8S4Case param;
    param.hasAntiquantOffset = true;
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteS8S4Tiling(param, tilingInfo));
}

TEST(GroupedS8S4BasicApiArch35Tiling, RejectsSpecialWeightWithoutNzConversion)
{
    S8S4Case param;
    param.specialWeightFormat = true;
    param.tuningConfig = {0, 1};
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteS8S4Tiling(param, tilingInfo));
}

TEST(GroupedS8S4BasicApiArch35Tiling, RejectsInvalidSpecialWeightFormatFlag)
{
    for (const int64_t value : {-1L, 2L}) {
        S8S4Case param;
        param.tuningConfig = {0, value};
        TilingInfo tilingInfo;
        EXPECT_FALSE(ExecuteS8S4Tiling(param, tilingInfo)) << "tuningConfig[1]=" << value;
    }
}

TEST(GroupedS8S4BasicApiArch35Tiling, RejectsUnsupportedSplitItem)
{
    for (const int64_t value : {0L, 1L}) {
        S8S4Case param;
        param.splitItem = value;
        TilingInfo tilingInfo;
        EXPECT_FALSE(ExecuteS8S4Tiling(param, tilingInfo)) << "splitItem=" << value;
    }
}

TEST(GroupedS8S4BasicApiArch35Tiling, RejectsMultiBiasTensorList)
{
    S8S4Case param;
    param.multiBias = true;
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteS8S4Tiling(param, tilingInfo));
}

TEST(GroupedS8S4BasicApiArch35Tiling, RejectsMultiOffsetTensorList)
{
    S8S4Case param;
    param.hasOffset = true;
    param.scaleGroupNum = 1;
    param.outDtype = ge::DT_FLOAT16;
    param.multiOffset = true;
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteS8S4Tiling(param, tilingInfo));
}

TEST(GroupedS8S4BasicApiArch35Tiling, RejectsMultiOutputTensorList)
{
    S8S4Case param;
    param.multiOutput = true;
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteS8S4Tiling(param, tilingInfo));
}
} // namespace
