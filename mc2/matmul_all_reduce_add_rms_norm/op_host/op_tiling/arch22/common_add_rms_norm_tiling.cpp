/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file common_add_rms_norm_tiling.cc
 * \brief
 */

#include "common_add_rms_norm_tiling.h"

namespace optiling {
namespace {
constexpr uint32_t DTYPE_KEY_FP16 = 1U;
constexpr uint32_t DTYPE_KEY_FP32 = 2U;
constexpr uint32_t DTYPE_KEY_BF16 = 3U;
constexpr uint32_t UB_FACTOR_B16 = 12288U;
constexpr uint32_t UB_FACTOR_B32 = 10240U;
constexpr uint32_t UB_FACTOR_B16_CUTD = 12096U;
constexpr uint32_t UB_FACTOR_B32_CUTD = 9696U;
constexpr uint32_t BLOCK_ALIGN_NUM = 16U;
constexpr uint32_t VALUE_2 = 2U;
constexpr uint32_t VALUE_16 = 16U;
constexpr uint32_t VALUE_64 = 64U;
constexpr uint32_t VALUE_256 = 256U;
constexpr uint32_t VALUE_1024 = 1024U;
constexpr uint32_t SMALL_REDUCE_NUM = 2000;
constexpr uint32_t COL_ALIGN_SIZE = 260;
constexpr uint32_t NUM_ROW_IDX = 0;
constexpr uint32_t NUM_COL_IDX = 1;
constexpr uint32_t BLOCK_FACTOR_IDX = 2;
constexpr uint32_t ROW_FACTOR_IDX = 3;
constexpr uint32_t UB_FACTOR_IDX = 4;
constexpr uint32_t EPSILON_IDX = 5;
constexpr uint32_t AVG_FACTOR_IDX = 6;

using TilingInfo = std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, float>;
void SetWorkSpaceSize(AddRMSNormTilingOutput &addRmsNormTilingOutput)
{
    size_t sysWorkspaceSize = SYS_WORKSPACE_SIZE;
    size_t usrSize = 256;
    addRmsNormTilingOutput.tilingOut.workSpaceSize = usrSize + sysWorkspaceSize;
}
void SetTilingKey(const uint32_t dtypeKey, uint32_t modeKey, AddRMSNormTilingOutput &addRmsNormTilingOutput)
{
    uint32_t tilingKey = dtypeKey * 10 + modeKey;
    addRmsNormTilingOutput.tilingOut.tilingKey = tilingKey;
}
void SetNumBlocks(const uint32_t numRow, const uint32_t blockFactor, AddRMSNormTilingOutput &addRmsNormTilingOutput)
{
    uint32_t useCoreNum = Ops::Base::CeilDiv(numRow, blockFactor);
    addRmsNormTilingOutput.tilingOut.numBlocks = useCoreNum;
}
void SetTilingData(const TilingInfo &tilingInfo, AddRMSNormTilingOutput &addRmsNormTilingOutput)
{
    auto &&tilingData = addRmsNormTilingOutput.addRmsNormTilingData;
    tilingData.num_row = std::get<NUM_ROW_IDX>(tilingInfo);
    tilingData.num_col = std::get<NUM_COL_IDX>(tilingInfo);
    tilingData.block_factor = std::get<BLOCK_FACTOR_IDX>(tilingInfo);
    tilingData.row_factor = std::get<ROW_FACTOR_IDX>(tilingInfo);
    tilingData.ub_factor = std::get<UB_FACTOR_IDX>(tilingInfo);
    tilingData.epsilon = std::get<EPSILON_IDX>(tilingInfo);
    tilingData.avg_factor = std::get<AVG_FACTOR_IDX>(tilingInfo);
}
ge::graphStatus AssembleX1Shape(const AddRMSNormTilingDepend &addRmsNormTilingDepend, gert::Shape &xShape)
{
    if (!addRmsNormTilingDepend.useMmOutputAsX1Input) {
        MC2_CHECK_NOTNULL_RET(addRmsNormTilingDepend.nodeName, addRmsNormTilingDepend.arnCtxInfo.x1_shape);
        xShape = addRmsNormTilingDepend.arnCtxInfo.x1_shape->GetStorageShape();
    } else {
        xShape = {addRmsNormTilingDepend.addRmsNormTilingInputFromMm.m,
                  addRmsNormTilingDepend.addRmsNormTilingInputFromMm.n};
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AssembleX1Dtype(const AddRMSNormTilingDepend &addRmsNormTilingDepend, ge::DataType &xDtype)
{
    if (!addRmsNormTilingDepend.useMmOutputAsX1Input) {
        MC2_CHECK_NOTNULL_RET(addRmsNormTilingDepend.nodeName, addRmsNormTilingDepend.arnCtxInfo.x1);
        xDtype = addRmsNormTilingDepend.arnCtxInfo.x1->GetDataType();
    } else {
        xDtype = addRmsNormTilingDepend.addRmsNormTilingInputFromMm.x1Dtype;
    }
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus CheckAddRmsNormInputDtype(const gert::TilingContext *context, const ge::DataType &x2Type,
                                          const ge::DataType &gammaType, const ge::DataType &yType,
                                          const ge::DataType &xType)
{
    // residual和gamma数据类型相同
    OP_TILING_CHECK(x2Type != gammaType,
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        context->GetNodeName(), "residual and gamma",
                        (Ops::Base::ToString(x2Type) + " and " + Ops::Base::ToString(gammaType)).c_str(),
                        "The dtypes of residual and gamma must be the same"),
                    return ge::GRAPH_FAILED);
    // residual和normOut数据类型相同
    OP_TILING_CHECK(x2Type != yType,
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        context->GetNodeName(), "residual and normOut",
                        (Ops::Base::ToString(x2Type) + " and " + Ops::Base::ToString(yType)).c_str(),
                        "The dtypes of residual and normOut must be the same"),
                    return ge::GRAPH_FAILED);
    // residual和输出y数据类型相同
    OP_TILING_CHECK(x2Type != xType,
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        context->GetNodeName(), "residual and y",
                        (Ops::Base::ToString(x2Type) + " and " + Ops::Base::ToString(xType)).c_str(),
                        "The dtypes of residual and y must be the same"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckAddRmsNormInputShape(const gert::TilingContext *context, const gert::StorageShape *x2Shape,
                                          const gert::StorageShape *gammaShape, const gert::StorageShape *yShape,
                                          const gert::StorageShape *xShape)
{
    // residual shape
    OP_TILING_CHECK(
        x2Shape->GetStorageShape().GetDimNum() != DIM_THREE,
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "residual",
                                     (std::to_string(x2Shape->GetStorageShape().GetDimNum()) + "D").c_str(), "3D"),
        return ge::GRAPH_FAILED);
    // gamma shape
    OP_TILING_CHECK(
        gammaShape->GetStorageShape().GetDimNum() != DIM_ONE,
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "gamma",
                                     (std::to_string(gammaShape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
        return ge::GRAPH_FAILED);
    // normOut shape
    OP_TILING_CHECK(
        yShape->GetStorageShape().GetDimNum() != DIM_THREE,
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "normOut",
                                     (std::to_string(yShape->GetStorageShape().GetDimNum()) + "D").c_str(), "3D"),
        return ge::GRAPH_FAILED);
    // y shape
    OP_TILING_CHECK(
        xShape->GetStorageShape().GetDimNum() != DIM_THREE,
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "y",
                                     (std::to_string(xShape->GetStorageShape().GetDimNum()) + "D").c_str(), "3D"),
        return ge::GRAPH_FAILED);
    // residual和gamma的n值
    OP_TILING_CHECK(x2Shape->GetStorageShape().GetDim(2) != gammaShape->GetStorageShape().GetDim(0),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context->GetNodeName(), "residual",
                        (std::to_string(x2Shape->GetStorageShape().GetDim(2)) + " vs " +
                         std::to_string(gammaShape->GetStorageShape().GetDim(0)))
                            .c_str(),
                        "The shape of residual n dimension must be the same as the n dimension of gamma"),
                    return ge::GRAPH_FAILED);
    // residual和y，normOut的shape
    OP_TILING_CHECK(x2Shape->GetStorageShape() != yShape->GetStorageShape(),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "residual and normOut",
                                                           (Ops::Base::ToString(x2Shape->GetStorageShape()) + " and " +
                                                            Ops::Base::ToString(yShape->GetStorageShape()))
                                                               .c_str(),
                                                           "The shapes of residual and normOut must be the same"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(x2Shape->GetStorageShape() != xShape->GetStorageShape(),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "residual and y",
                                                           (Ops::Base::ToString(x2Shape->GetStorageShape()) + " and " +
                                                            Ops::Base::ToString(xShape->GetStorageShape()))
                                                               .c_str(),
                                                           "The shapes of residual and y must be the same"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
} // namespace

ge::graphStatus CommonAddResNormTiling::CheckAddRmsNormInput(const gert::TilingContext *context,
                                                             const ARNCtxInfo &arnCtxInfo)
{
    auto x2Type = arnCtxInfo.x2->GetDataType();
    auto gammaType = arnCtxInfo.gamma->GetDataType();
    auto yType = arnCtxInfo.y->GetDataType();
    auto xType = arnCtxInfo.x->GetDataType();
    auto epsilon = arnCtxInfo.epsilon;
    const gert::StorageShape *x2Shape = arnCtxInfo.x2_shape;
    const gert::StorageShape *gammaShape = arnCtxInfo.gamma_shape;
    const gert::StorageShape *yShape = arnCtxInfo.y_shape;
    const gert::StorageShape *xShape = arnCtxInfo.x_shape;
    MC2_CHECK_LOG_RET(context->GetNodeName(), CheckAddRmsNormInputDtype(context, x2Type, gammaType, yType, xType));
    MC2_CHECK_LOG_RET(context->GetNodeName(), CheckAddRmsNormInputShape(context, x2Shape, gammaShape, yShape, xShape));
    OP_TILING_CHECK(
        (epsilon != nullptr && (*epsilon <= 0 || *epsilon >= 1)),
        OP_LOGE_WITH_INVALID_ATTR(context->GetNodeName(), "epsilon", std::to_string(*epsilon).c_str(), "(0, 1)"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonAddResNormTiling::SetAddRmsNormTilingData(const AddRMSNormTilingDepend &addRmsNormTilingDepend,
                                                                const uint32_t numRow, const int64_t numCol,
                                                                const uint32_t blockFactor,
                                                                AddRMSNormTilingOutput &addRmsNormTilingOutput)
{
    ge::DataType dataType;
    MC2_CHECK_LOG_RET(addRmsNormTilingDepend.nodeName, AssembleX1Dtype(addRmsNormTilingDepend, dataType));

    uint32_t dtypeKey = DTYPE_KEY_FP16;
    uint32_t ubFactor = UB_FACTOR_B16;
    if (dataType == ge::DT_FLOAT) {
        dtypeKey = DTYPE_KEY_FP32;
        ubFactor = UB_FACTOR_B32;
    }
    if (dataType == ge::DT_BF16) {
        dtypeKey = DTYPE_KEY_BF16;
    }

    uint32_t modeKey = ModeKey::K_NORMAL; // 0: Normal, 1: SplitD, 2: MergeN, 3: SingleN 4: MultiN
    uint32_t numColAlign = Ops::Base::CeilDiv(numCol, static_cast<int64_t>(BLOCK_ALIGN_NUM)) * BLOCK_ALIGN_NUM;
    uint32_t rowFactor = VALUE_64;
    uint64_t ubSize;
    const auto &ascendcPlatform = platform_ascendc::PlatformAscendC(&addRmsNormTilingDepend.platFormInfos);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (numCol > ubFactor) {
        modeKey = ModeKey::K_SPLIT_D;
        ubFactor = (dataType == ge::DT_FLOAT) ? UB_FACTOR_B32_CUTD : UB_FACTOR_B16_CUTD;
        uint32_t colTileNum = Ops::Base::CeilDiv(numCol, static_cast<int64_t>(ubFactor));
        ubFactor = Ops::Base::CeilDiv(numCol, static_cast<int64_t>(colTileNum * BLOCK_ALIGN_NUM)) * BLOCK_ALIGN_NUM;
    } else if (blockFactor == 1) {
        modeKey = ModeKey::K_SINGLE_N;
    } else if (numColAlign <= SMALL_REDUCE_NUM) {
        modeKey = ModeKey::K_MERGE_N;
        uint64_t numColAlignWeight = (dtypeKey == DTYPE_KEY_FP32) ? 24 : 18;
        rowFactor = ubSize / (numColAlign * numColAlignWeight + COL_ALIGN_SIZE);
        ubFactor = rowFactor * numColAlign;
    } else if (dataType == ge::DT_FLOAT16 && numCol == numColAlign) {
        modeKey = ModeKey::K_MULTI_N;
        MC2_CHECK_TRUE_RET(addRmsNormTilingDepend.nodeName, ubSize >= VALUE_1024 + VALUE_256 + numColAlign * VALUE_2);
        MC2_CHECK_TRUE_RET(addRmsNormTilingDepend.nodeName, numColAlign * VALUE_16 + VALUE_64 != 0);
        // reserve 1024byte
        rowFactor = (ubSize - VALUE_1024 - VALUE_256 - numColAlign * VALUE_2) / (numColAlign * VALUE_16 + VALUE_64);
        ubFactor = rowFactor * numColAlign;
        if (rowFactor == 0) {
            modeKey = ModeKey::K_NORMAL;
            rowFactor = VALUE_64;
            ubFactor = UB_FACTOR_B16;
        }
    }
    MC2_CHECK_NOTNULL_RET(addRmsNormTilingDepend.nodeName, addRmsNormTilingDepend.arnCtxInfo.epsilon);
    const float epsilon = *addRmsNormTilingDepend.arnCtxInfo.epsilon;
    float avgFactor = (numCol == 0) ? 0 : 1.0F / numCol;
    SetTilingData(std::make_tuple(numRow, numCol, blockFactor, rowFactor, ubFactor, epsilon, avgFactor),
                  addRmsNormTilingOutput);
    SetTilingKey(dtypeKey, modeKey, addRmsNormTilingOutput);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonAddResNormTiling::Tiling4AddRmsNorm(const AddRMSNormTilingDepend &addRmsNormTilingDepend,
                                                          AddRMSNormTilingOutput &addRmsNormTilingOutput)
{
    const std::string node = addRmsNormTilingDepend.nodeName;
    OP_LOGD(node, "Enter CommonAddResNormTiling");

    const auto &ascendcPlatform = platform_ascendc::PlatformAscendC(&addRmsNormTilingDepend.platFormInfos);
    uint32_t numCore = ascendcPlatform.GetCoreNumAiv();
    if (addRmsNormTilingDepend.useHalfNumBlocks) {
        numCore /= 2U; // 配比1：1时, 1/2个数
    }
    OP_LOGD(node, "Core Num: %u, use half %d", numCore, addRmsNormTilingDepend.useHalfNumBlocks);
    uint32_t blockFactor = 1U;
    MC2_CHECK_NOTNULL_RET(addRmsNormTilingDepend.nodeName, addRmsNormTilingDepend.arnCtxInfo.gamma_shape);
    const auto &gammaShape = addRmsNormTilingDepend.arnCtxInfo.gamma_shape->GetStorageShape();
    int64_t numCol = gammaShape.GetShapeSize();
    gert::Shape xShape;
    MC2_CHECK_LOG_RET(addRmsNormTilingDepend.nodeName, AssembleX1Shape(addRmsNormTilingDepend, xShape));

    size_t xDimNum = xShape.GetDimNum();
    size_t gammaDimNum = gammaShape.GetDimNum();
    MC2_CHECK_TRUE_RET(addRmsNormTilingDepend.nodeName, xDimNum >= gammaDimNum);
    uint32_t numRow = 1U;
    for (size_t i = 0; i < xDimNum - gammaDimNum; i++) {
        numRow *= xShape.GetDim(i);
    }
    uint32_t tileNum = Ops::Base::CeilDiv(numRow, numCore * blockFactor);
    blockFactor *= tileNum;
    OP_LOGD(node, "tile num: %u, numRow: %u blockFactor: %u", tileNum, numRow, blockFactor);

    // block dim的值小于等于num_core
    SetNumBlocks(numRow, blockFactor, addRmsNormTilingOutput);
    SetWorkSpaceSize(addRmsNormTilingOutput);
    MC2_CHECK_LOG_RET(addRmsNormTilingDepend.nodeName, SetAddRmsNormTilingData(addRmsNormTilingDepend, numRow, numCol,
                                                                               blockFactor, addRmsNormTilingOutput));

    OP_LOGI(node, "Tiling Key: %u", addRmsNormTilingOutput.tilingOut.tilingKey);
    OP_LOGI(node, "Block Dim: %u", addRmsNormTilingOutput.tilingOut.numBlocks);
    OP_LOGI(node, "Workspace: %u", addRmsNormTilingOutput.tilingOut.workSpaceSize);
    OP_LOGI(node, "numRow: %d, numCol: %ld, blockFactor: %d, rowFactor: %d, ubFactor: %d, epsilon: %f, avgFactor: %f",
            numRow, numCol, blockFactor, addRmsNormTilingOutput.addRmsNormTilingData.row_factor,
            addRmsNormTilingOutput.addRmsNormTilingData.ub_factor, addRmsNormTilingOutput.addRmsNormTilingData.epsilon,
            addRmsNormTilingOutput.addRmsNormTilingData.avg_factor);

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
