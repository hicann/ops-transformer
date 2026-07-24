/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file allto_allv_quant_grouped_mat_mul_tiling.cpp
 * \brief
 */

#include "allto_allv_tt_quant_grouped_mat_mul_tiling.h"
#include "mc2_tiling_utils.h"
#include "mc2_comm_utils.h"

using namespace ge;
using namespace AscendC;
using namespace Ops::Transformer::OpTiling;

namespace optiling {
bool AlltoAllvTTQuantGmmTiling::IsCapable()
{
    // hifloat8 quant
    if (gmmXDataType_ != ge::DT_HIFLOAT8) {
        return false;
    }
    if (gmmWeightDataType_ != ge::DT_HIFLOAT8) {
        return false;
    }
    OP_LOGD(context_->GetNodeName(), "AlltoAllvTTQuantGmmTiling is capable.");
    return true;
}

uint64_t AlltoAllvTTQuantGmmTiling::GetTilingKey() const
{
    uint8_t commMode = 0;
    if (AlltoAllvQuantGmmTilingCommon::QuantGetAndConvertCommMode(context_, commMode) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    uint64_t tilingKey = GET_TPL_TILING_KEY(transGmmWeight_, transMmWeight_, commMode);
    return tilingKey;
}

ge::graphStatus AlltoAllvTTQuantGmmTiling::DoGmmTiling(uint64_t gmmxMSzie)
{
    OP_LOGD(context_->GetNodeName(), "start DoGmmTiling.");
    uint32_t expertNum = CalcExpertNum(e_, epWorldSize_, bsk_, n1_);
    uint32_t gmmGroupNum = expertNum * static_cast<uint32_t>(epWorldSize_);
    if (gmmxMSzie != 0) {
        auto &gmmQuantTilingData = tilingData->gmmQuantTilingData;
        SetGMMQuantParams(gmmQuantTilingData, gmmGroupNum);
        SetTilingArray(gmmQuantTilingData, gmmxMSzie, n1_, h1_, gmmGroupNum);
        SetTilingParams(gmmQuantTilingData, gmmxMSzie, n1_, h1_, transGmmWeight_);
        PrintGMMQuantTilingData(gmmQuantTilingData);
    }
    if (bs_ != 0) {
        auto &mmQuantTilingData = tilingData->mmQuantTilingData;
        SetGMMQuantParams(mmQuantTilingData, SINGLE_GROUP_NUM);
        SetTilingArray(mmQuantTilingData, bs_, n2_, h2_, SINGLE_GROUP_NUM);
        SetTilingParams(mmQuantTilingData, bs_, n2_, h2_, transMmWeight_);
        PrintGMMQuantTilingData(mmQuantTilingData);
    }
    GetPermuteOutSize();
    OP_LOGD(context_->GetNodeName(), "end DoGmmTiling.");
    return ge::GRAPH_SUCCESS;
}

void AlltoAllvTTQuantGmmTiling::SetGMMQuantParams(
    Mc2GroupedMatmulTilingData::GMMQuantTilingData &gmmQuantTilingData, uint32_t groupNum) const
{
    gmmQuantTilingData.gmmQuantParams.groupNum = groupNum;
    gmmQuantTilingData.gmmQuantParams.activeType = GMM_ACT_TYPE_NONE;
    gmmQuantTilingData.gmmQuantParams.aQuantMode = PERTENSOR_QUANT_MODE;
    gmmQuantTilingData.gmmQuantParams.bQuantMode = PERTENSOR_QUANT_MODE;
    gmmQuantTilingData.gmmQuantParams.singleX = 1;
    gmmQuantTilingData.gmmQuantParams.singleW = 1;
    gmmQuantTilingData.gmmQuantParams.singleY = 1;
    gmmQuantTilingData.gmmQuantParams.groupType = 0;
    gmmQuantTilingData.gmmQuantParams.groupListType = 1;
    gmmQuantTilingData.gmmQuantParams.hasBias = 0;
    gmmQuantTilingData.gmmQuantParams.reserved = 0;
}

void AlltoAllvTTQuantGmmTiling::SetTilingArray(
    Mc2GroupedMatmulTilingData::GMMQuantTilingData &gmmQuantTilingData,
    uint64_t M, uint64_t N, uint64_t K, uint32_t groupNum) const
{
    constexpr uint32_t MAX_TENSOR_CONT = 128U;
    OP_TILING_CHECK(groupNum > MAX_TENSOR_CONT,
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "groupNum",
            std::to_string(groupNum).c_str(),
            (std::string("<=") + std::to_string(MAX_TENSOR_CONT)).c_str()),
        return);
    for (uint32_t i = 0; i < groupNum; i++) {
        gmmQuantTilingData.gmmArray.mList[i] = static_cast<int32_t>(M);
    }
    gmmQuantTilingData.gmmArray.kList[0] = static_cast<int32_t>(K);
    gmmQuantTilingData.gmmArray.nList[0] = static_cast<int32_t>(N);
}

void AlltoAllvTTQuantGmmTiling::SetTilingParams(Mc2GroupedMatmulTilingData::GMMQuantTilingData &gmmQuantTilingData,
                                                uint64_t M, uint64_t N, uint64_t K, bool transB) const
{
    auto &mm = gmmQuantTilingData.mmTilingData;

    mm.M = M;
    mm.N = N;
    mm.Ka = K;
    mm.Kb = K;
    mm.usedCoreNum = aicCoreNum_;
    mm.isBias = 0;
    mm.dbL0A = DOUBLE_BUFFER;
    mm.dbL0B = DOUBLE_BUFFER;

    mm.baseM = std::min(static_cast<int32_t>(M), static_cast<int32_t>(BASIC_BLOCK_SIZE_256));
    mm.baseM = Ops::Base::CeilAlign(mm.baseM, static_cast<int32_t>(CUBE_BLOCK));
    mm.baseN = std::min(static_cast<int32_t>(N), static_cast<int32_t>(BASIC_BLOCK_SIZE_256));
    mm.baseN =
        Ops::Base::CeilAlign(mm.baseN, transB ? static_cast<int32_t>(CUBE_BLOCK) : static_cast<int32_t>(L1_ALIGN_SIZE));

    mm.baseK = std::min(static_cast<int32_t>(K), static_cast<int32_t>(BASIC_BLOCK_SIZE_128));
    mm.baseK = Ops::Base::CeilAlign(mm.baseK, static_cast<int32_t>(CUBE_REDUCE_BLOCK));

    mm.singleCoreM = std::min(static_cast<int32_t>(M), mm.baseM);
    mm.singleCoreN = std::min(static_cast<int32_t>(N), mm.baseN);
    mm.singleCoreK = K;

    uint64_t l0cRequired = static_cast<uint64_t>(mm.baseM) * mm.baseN * DATA_SIZE_L0C * DB_SIZE;
    mm.dbL0C = (l0cRequired <= l0cSize_) ? DB_SIZE : 1;

    mm.iterateOrder = 0U;

    uint64_t baseASize = static_cast<uint64_t>(mm.baseM) * mm.baseK;
    uint64_t baseBSize = static_cast<uint64_t>(mm.baseN) * mm.baseK;
    uint64_t baseL1Size = baseASize + baseBSize;

    OP_TILING_CHECK(baseL1Size == 0, OP_LOGW(context_->GetNodeName(), "baseL1Size cannot be zero."), return);

    uint64_t leftL1Size = l1Size_;

    uint64_t depthInit = leftL1Size / baseL1Size;
    depthInit = std::max(depthInit, static_cast<uint64_t>(1));

    uint64_t depthScale = depthInit;
    while (depthScale * mm.baseK % BASIC_BLOCK_SIZE_512 != 0 && depthScale > 1) {
        depthScale--;
    }
    depthScale = std::max(depthScale, static_cast<uint64_t>(1));

    mm.depthA1 = depthScale;
    mm.depthB1 = depthScale;

    mm.stepKa = (mm.depthA1 > 1) ? (mm.depthA1 / DB_SIZE) : 1;
    mm.stepKb = (mm.depthB1 > 1) ? (mm.depthB1 / DB_SIZE) : 1;

    OP_TILING_CHECK(mm.baseK == 0, OP_LOGW(context_->GetNodeName(), "baseK cannot be zero."), return);

    if (mm.stepKa * mm.baseK > mm.Ka) {
        mm.stepKa = Ops::Base::CeilDiv(mm.Ka, mm.baseK);
    }
    if (mm.stepKb * mm.baseK > mm.Kb) {
        mm.stepKb = Ops::Base::CeilDiv(mm.Kb, mm.baseK);
    }

    mm.depthA1 = mm.stepKa * DB_SIZE;
    mm.depthB1 = mm.stepKb * DB_SIZE;

    mm.stepM = 1;
    mm.stepN = 1;
}

ge::graphStatus AlltoAllvTTQuantGmmTiling::CheckQuantMode() const
{
    OP_LOGD(context_->GetNodeName(), "start CheckQuantMode.");
    // check gmmXQuantMode null
    OP_TILING_CHECK(gmmXQuantModePtr_ == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "gmmXQuantMode"),
                    return ge::GRAPH_FAILED);
    // check gmmXQuantMode
    int64_t gmmXQuantMode = *gmmXQuantModePtr_;
    OP_TILING_CHECK(
        gmmXQuantMode != static_cast<int64_t>(PERTENSOR_QUANT_MODE),
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "gmmXQuantMode", std::to_string(gmmXQuantMode).c_str(), "1"),
        return ge::GRAPH_FAILED);
    // check gmmWeightQuantMode null
    OP_TILING_CHECK(gmmWeightQuantModePtr_ == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "gmmWeightQuantMode"), return ge::GRAPH_FAILED);
    // check gmmWeightQuantMode
    int64_t gmmWeightQuantMode = *gmmWeightQuantModePtr_;
    OP_TILING_CHECK(gmmWeightQuantMode != static_cast<int64_t>(PERTENSOR_QUANT_MODE),
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "gmmWeightQuantMode",
                                              std::to_string(gmmWeightQuantMode).c_str(), "1"),
                    return ge::GRAPH_FAILED);
    if (hasSharedExpertFlag_) {
        // mmXQuantMode(same as gmmXQuantMode)
        int64_t mmXQuantMode = *mmXQuantModePtr_;
        OP_TILING_CHECK(mmXQuantMode != gmmXQuantMode,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            context_->GetNodeName(), "mmXQuantMode", std::to_string(mmXQuantMode).c_str(),
                            "The value of mmXQuantMode must be the same as that of gmmXQuantMode(1)."),
                        return ge::GRAPH_FAILED);
        // mmWeightQuantMode
        int64_t mmWeightQuantMode = *mmWeightQuantModePtr_;
        OP_TILING_CHECK(mmWeightQuantMode != gmmWeightQuantMode,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            context_->GetNodeName(), "mmWeightQuantMode", std::to_string(mmWeightQuantMode).c_str(),
                            "The value of mmWeightQuantMode must be the same as that of gmmWeightQuantMode(1)."),
                        return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    // check group size
    auto groupSizePtr = context_->GetAttrs()->GetAttrPointer<int64_t>(ATTR_GROUP_SIZE_INDEX);
    OP_TILING_CHECK(groupSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "groupSize"),
                    return ge::GRAPH_FAILED);
    uint64_t groupSize = *groupSizePtr;
    OP_TILING_CHECK(groupSize != 0,
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "groupSize", std::to_string(groupSize), "0"),
                    return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "end CheckQuantMode.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvTTQuantGmmTiling::CheckScaleFormatAndDtype() const
{
    OP_LOGD(context_->GetNodeName(), "start CheckScaleFormatAndDtype.");
    // check gmmXScale null
    auto gmmXScaleDesc = context_->GetRequiredInputDesc(GMM_X_SCALE_INDEX);
    OP_TILING_CHECK(gmmXScaleDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "gmmXScale"),
                    return ge::GRAPH_FAILED);
    // check gmmXScale format
    OP_TILING_CHECK(gmmXScaleDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "gmmXScale",
                                               Ops::Base::ToString(gmmXScaleDesc->GetStorageFormat()), "ND"),
                    return ge::GRAPH_FAILED);
    // check gmmXScale dataType
    OP_TILING_CHECK(gmmXScaleDesc->GetDataType() != ge::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE(
                        context_->GetNodeName(), "gmmXScale",
                        ge::TypeUtils::DataTypeToSerialString(gmmXScaleDesc->GetDataType()).c_str(), "float32"),
                    return ge::GRAPH_FAILED);
    // check gmmWeightScale null
    auto gmmWeightScaleDesc = context_->GetRequiredInputDesc(GMM_WEIGHT_SCALE_INDEX);
    OP_TILING_CHECK(gmmWeightScaleDesc == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "gmmWeightScale"), return ge::GRAPH_FAILED);
    // check gmmWeightScale format
    OP_TILING_CHECK(gmmWeightScaleDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "gmmWeightScale",
                                               Ops::Base::ToString(gmmWeightScaleDesc->GetStorageFormat()), "ND"),
                    return ge::GRAPH_FAILED);
    // check gmmWeightScale dataType
    OP_TILING_CHECK(gmmWeightScaleDesc->GetDataType() != ge::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE(
                        context_->GetNodeName(), "gmmWeightScale",
                        ge::TypeUtils::DataTypeToSerialString(gmmWeightScaleDesc->GetDataType()).c_str(), "float32"),
                    return ge::GRAPH_FAILED);
    if (hasSharedExpertFlag_) {
        // check mmXScale null
        auto mmXScaleDesc = context_->GetOptionalInputDesc(MM_X_SCALE_INDEX);
        OP_TILING_CHECK(mmXScaleDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "mmXScale"),
                        return ge::GRAPH_FAILED);
        // check mmXScale format
        OP_TILING_CHECK(mmXScaleDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
                        OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "mmXScale",
                                                   Ops::Base::ToString(mmXScaleDesc->GetStorageFormat()), "ND"),
                        return ge::GRAPH_FAILED);
        // check mmXScale dataType(same as gmmXScale)
        OP_TILING_CHECK(mmXScaleDesc->GetDataType() != gmmXScaleDesc->GetDataType(),
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            context_->GetNodeName(), "mmXScale",
                            ge::TypeUtils::DataTypeToSerialString(mmXScaleDesc->GetDataType()).c_str(),
                            "The dtype of mmXScale must be float32."),
                        return ge::GRAPH_FAILED);
        // check mmWeightScale null
        auto mmWeightScaleDesc = context_->GetOptionalInputDesc(MM_WEIGHT_SCALE_INDEX);
        OP_TILING_CHECK(mmWeightScaleDesc == nullptr,
                        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "mmWeightScale"), return ge::GRAPH_FAILED);
        // check mmWeightScale format
        OP_TILING_CHECK(mmWeightScaleDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
                        OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "mmWeightScale",
                                                   Ops::Base::ToString(mmWeightScaleDesc->GetStorageFormat()), "ND"),
                        return ge::GRAPH_FAILED);
        // check mmWeightScale dataType(same as gmmWeightScale)
        OP_TILING_CHECK(mmWeightScaleDesc->GetDataType() != gmmWeightScaleDesc->GetDataType(),
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            context_->GetNodeName(), "mmWeightScale",
                            ge::TypeUtils::DataTypeToSerialString(mmWeightScaleDesc->GetDataType()).c_str(),
                            "The dtype of mmWeightScale must be float32."),
                        return ge::GRAPH_FAILED);
    }
    OP_LOGD(context_->GetNodeName(), "end CheckScaleFormatAndDtype.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvTTQuantGmmTiling::CheckInputDtype() const
{
    OP_LOGD(context_->GetNodeName(), "start CheckInputDtype.");
    // check gmmX datatype
    ge::DataType gmmXDataType = context_->GetInputDesc(GMM_X_INDEX)->GetDataType();
    OP_TILING_CHECK(gmmXDataType != ge::DT_HIFLOAT8,
                    OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "gmmX",
                                              ge::TypeUtils::DataTypeToSerialString(gmmXDataType).c_str(), "hifloat8"),
                    return ge::GRAPH_FAILED);
    // check gmmWeight datatype
    ge::DataType gmmWeightDataType = context_->GetInputDesc(GMM_WEIGHT_INDEX)->GetDataType();
    OP_TILING_CHECK(gmmWeightDataType != ge::DT_HIFLOAT8,
                    OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "gmmWeight",
                                              ge::TypeUtils::DataTypeToSerialString(gmmWeightDataType).c_str(),
                                              "hifloat8"),
                    return ge::GRAPH_FAILED);
    // check gmmY dataType
    ge::DataType gmmYDataType = context_->GetOutputDesc(OUTPUT_GMM_Y_INDEX)->GetDataType();
    OP_TILING_CHECK(gmmYDataType != ge::DT_FLOAT16 && gmmYDataType != ge::DT_BF16,
                    OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "gmmY",
                                              ge::TypeUtils::DataTypeToSerialString(gmmYDataType).c_str(),
                                              "float16 or bfloat16"),
                    return ge::GRAPH_FAILED);
    if (permuteOutFlag_) {
        // check permuteOut dtype
        ge::DataType permuteOutDataType = context_->GetOutputDesc(OUTPUT_PERMUTE_OUT_INDEX)->GetDataType();
        OP_TILING_CHECK(
            permuteOutDataType != gmmXDataType,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "permuteOut",
                                                  ge::TypeUtils::DataTypeToSerialString(permuteOutDataType).c_str(),
                                                  "The dtype of permuteOut must be hifloat8."),
            return ge::GRAPH_FAILED);
    }
    if (hasSharedExpertFlag_) {
        // check mmX dataType(same as gmmX)
        ge::DataType mmXDataType = context_->GetOptionalInputDesc(MM_X_INDEX)->GetDataType();
        OP_TILING_CHECK(mmXDataType != gmmXDataType,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            context_->GetNodeName(), "mmX", ge::TypeUtils::DataTypeToSerialString(mmXDataType).c_str(),
                            "The dtype of mmX must be hifloat8."),
                        return ge::GRAPH_FAILED);
        // check mmWeight dataType(same as gmmWeight)
        ge::DataType mmWeightDataType = context_->GetOptionalInputDesc(MM_WEIGHT_INDEX)->GetDataType();
        OP_TILING_CHECK(
            mmWeightDataType != gmmWeightDataType,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "mmWeight",
                                                  ge::TypeUtils::DataTypeToSerialString(mmWeightDataType).c_str(),
                                                  "The dtype of mmWeight must be hifloat8."),
            return ge::GRAPH_FAILED);
        // check mmY dataType(same as gmmY)
        ge::DataType mmYDataType = context_->GetOutputDesc(OUTPUT_MM_Y_INDEX)->GetDataType();
        OP_TILING_CHECK(mmYDataType != gmmYDataType,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            context_->GetNodeName(), "mmY", ge::TypeUtils::DataTypeToSerialString(mmYDataType).c_str(),
                            ("The dtype of mmY must be the same as that of gmmY(" +
                             std::string(ge::TypeUtils::DataTypeToSerialString(gmmYDataType)) + ").")
                                .c_str()),
                        return ge::GRAPH_FAILED);
    }
    OP_LOGD(context_->GetNodeName(), "end CheckInputDtype.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvTTQuantGmmTiling::CheckScaleShape() const
{
    OP_LOGD(context_->GetNodeName(), "start CheckScaleShape.");
    // check gmmXScale dimNum
    size_t gmmXScaleDimNum = context_->GetRequiredInputShape(GMM_X_SCALE_INDEX)->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(
        gmmXScaleDimNum != DIM_ONE,
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "gmmXScale", std::to_string(gmmXScaleDimNum), "1"),
        return ge::GRAPH_FAILED);
    // check gmmXScale shape
    int64_t gmmXScaleShape = context_->GetRequiredInputShape(GMM_X_SCALE_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
    OP_TILING_CHECK(
        gmmXScaleShape != DIM_ONE,
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "gmmXScale dim0", std::to_string(gmmXScaleShape), "1"),
        return ge::GRAPH_FAILED);
    // check gmmWeightScale dimNum
    size_t gmmWeightScaleDimNum =
        context_->GetRequiredInputShape(GMM_WEIGHT_SCALE_INDEX)->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(gmmWeightScaleDimNum != DIM_ONE,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "gmmWeightScale",
                                                 std::to_string(gmmWeightScaleDimNum), "1"),
                    return ge::GRAPH_FAILED);
    // check gmmWeightScale shape
    int64_t gmmWeightScaleShape =
        context_->GetRequiredInputShape(GMM_WEIGHT_SCALE_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
    OP_TILING_CHECK(gmmWeightScaleShape != DIM_ONE,
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "gmmWeightScale dim0",
                                              std::to_string(gmmWeightScaleShape), "1"),
                    return ge::GRAPH_FAILED);
    if (hasSharedExpertFlag_) {
        // check mmXScale dimNum(same as gmmXScale)
        size_t mmXScaleDimNum = context_->GetOptionalInputShape(MM_X_SCALE_INDEX)->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(mmXScaleDimNum != gmmXScaleDimNum,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                            context_->GetNodeName(), "mmXScale", std::to_string(mmXScaleDimNum).c_str(),
                            "The shape dim of mmXScale must be the same as that of gmmXScale(1D)."),
                        return ge::GRAPH_FAILED);
        // check mmXScale shape
        int64_t mmXScaleShape = context_->GetOptionalInputShape(MM_X_SCALE_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
        OP_TILING_CHECK(mmXScaleShape != DIM_ONE,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "mmXScale dim0",
                                                              std::to_string(mmXScaleShape).c_str(),
                                                              "The value of mmXScale dim0 must be 1."),
                        return ge::GRAPH_FAILED);
        // check mmWeightScale dimNum(same as gmmWeightScale)
        size_t mmWeightScaleDimNum =
            context_->GetOptionalInputShape(MM_WEIGHT_SCALE_INDEX)->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(mmWeightScaleDimNum != gmmWeightScaleDimNum,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                            context_->GetNodeName(), "mmWeightScale", std::to_string(mmWeightScaleDimNum).c_str(),
                            "The shape dim of mmWeightScale must be the same as that of gmmWeightScale(1D)."),
                        return ge::GRAPH_FAILED);
        // check mmWeightScale shape
        int64_t mmWeightScaleShape =
            context_->GetOptionalInputShape(MM_WEIGHT_SCALE_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
        OP_TILING_CHECK(mmWeightScaleShape != DIM_ONE,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "mmWeightScale dim0",
                                                              std::to_string(mmWeightScaleShape).c_str(),
                                                              "The value of mmWeightScale dim0 must be 1."),
                        return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGD(context_->GetNodeName(), "end CheckScaleShape.");
    return ge::GRAPH_SUCCESS;
}

void AlltoAllvTTQuantGmmTiling::GetPermuteOutSize()
{
    // commOutSize: 通信输出区域，kernel始终从workspace偏移0分配此空间存放通信结果
    permuteOutSize_ = a_ * h1_ * GetSizeByDataType(gmmXDataType_);
    // 将 permuteOutSize 对齐到 512 字节
    permuteOutSize_ = Ops::Base::CeilAlign(permuteOutSize_, static_cast<uint64_t>(BASIC_BLOCK_SIZE_512));
}

REGISTER_OPS_TILING_TEMPLATE(AlltoAllvQuantGroupedMatMul, AlltoAllvTTQuantGmmTiling, 1);
} // namespace optiling
