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
 * \file allto_allv_mx_quant_grouped_mat_mul_tiling.cpp
 * \brief
 */

#include "allto_allv_mx_quant_grouped_mat_mul_tiling.h"
#include "mc2_tiling_utils.h"
#include "mc2_comm_utils.h"

using namespace ge;
using namespace AscendC;
using namespace Ops::Transformer::OpTiling;

namespace optiling {
bool AlltoAllvMXQuantGmmTiling::IsCapable()
{
    // support fp8_e5m2 or fp8_e4m3
    if (gmmXDataType_ != ge::DT_FLOAT8_E5M2 &&
        gmmXDataType_ != ge::DT_FLOAT8_E4M3FN &&
        gmmXDataType_ != ge::DT_FLOAT4_E2M1) {
        return false;
    }
    if (gmmWeightDataType_ != ge::DT_FLOAT8_E5M2 &&
        gmmWeightDataType_ != ge::DT_FLOAT8_E4M3FN &&
        gmmWeightDataType_ != ge::DT_FLOAT4_E2M1) {
        return false;
    }
    OP_LOGD(context_->GetNodeName(), "AlltoAllvMXQuantGmmTiling is capable.");
    return true;
}

uint64_t AlltoAllvMXQuantGmmTiling::GetTilingKey() const
{
    uint8_t commMode = 0;
    if (AlltoAllvQuantGmmTilingCommon::QuantGetAndConvertCommMode(context_, commMode) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    uint64_t tilingKey = GET_TPL_TILING_KEY(transGmmWeight_, transMmWeight_, commMode);
    return tilingKey;
}

ge::graphStatus AlltoAllvMXQuantGmmTiling::CheckInputDtype() const
{
    OP_LOGD(context_->GetNodeName(), "start CheckInputDtype.");
    // gmm dtype
    ge::DataType gmmXDataType = context_->GetInputDesc(GMM_X_INDEX)->GetDataType();
    ge::DataType gmmWeightDataType = context_->GetInputDesc(GMM_WEIGHT_INDEX)->GetDataType();
    ge::DataType gmmYDataType = context_->GetOutputDesc(OUTPUT_GMM_Y_INDEX)->GetDataType();
    // check gmm input dtype
    ge::graphStatus status = CheckGmmInputDtype(gmmXDataType, gmmWeightDataType, gmmYDataType);
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // has mm
    if (hasSharedExpertFlag_) {
        // check mm input dtype(same as gmm)
        status = CheckMmInputDtype(gmmXDataType, gmmWeightDataType, gmmYDataType);
        if (status != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    // mxfp4 check
    if (gmmXDataType == ge::DT_FLOAT4_E2M1) {
        status = CheckFp4Input(gmmXDataType, gmmWeightDataType);
        if (status != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    OP_LOGD(context_->GetNodeName(), "end CheckInputDtype.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvMXQuantGmmTiling::CheckScaleFormatAndDtype() const
{
    OP_LOGD(context_->GetNodeName(), "start CheckScaleFormatAndDtype.");
    // check gmm scale
    ge::graphStatus status = CheckGmmScale();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // has mm
    if (hasSharedExpertFlag_) {
        // check mm scale
        status = CheckMmScale();
        if (status != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    OP_LOGD(context_->GetNodeName(), "end CheckScaleFormatAndDtype.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvMXQuantGmmTiling::CheckScaleShape() const
{
    OP_LOGD(context_->GetNodeName(), "Start CheckScaleShape.");
    // check gmm scale
    ge::graphStatus status = CheckGmmScaleShape();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (hasSharedExpertFlag_) {
        // check mm scale
        status = CheckMmScaleShape();
        if (status != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    OP_LOGD(context_->GetNodeName(), "End CheckScaleShape.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvMXQuantGmmTiling::DoGmmTiling(uint64_t gmmxMSzie)
{
    OP_LOGD(context_->GetNodeName(), "start DoGmmTiling.");
    uint32_t packFactor = (gmmXDataType_ == ge::DT_FLOAT4_E2M1) ? MXFP4_PACK_FACTOR : 1U;
    uint32_t expertNum = CalcExpertNum(e_, epWorldSize_, bsk_, n1_, packFactor);
    uint32_t gmmGroupNum = expertNum * static_cast<uint32_t>(epWorldSize_);
    // gmm group matmul tiling
    if (gmmxMSzie != 0) {
        AlltoAllvMXQuantGmmTilingHelper gmmHelper(*this);
        gmmHelper.SetGroupNum(gmmGroupNum);
        MC2_CHECK_LOG_RET(context_->GetNodeName(), gmmHelper.SetInputParams(gmmxMSzie, n1_, h1_, transGmmWeight_));
        MC2_CHECK_LOG_RET(context_->GetNodeName(), gmmHelper.Process());
        tilingData->gmmQuantTilingData = gmmHelper.GetAlltoAllvQuantHelperData();
        PrintGMMQuantTilingData(tilingData->gmmQuantTilingData);
    }
    // mm group matmul tiling
    if (bs_ != 0) {
        AlltoAllvMXQuantGmmTilingHelper mmHelper(*this);
        MC2_CHECK_LOG_RET(context_->GetNodeName(), mmHelper.SetInputParams(bs_, n2_, h2_, transMmWeight_));
        MC2_CHECK_LOG_RET(context_->GetNodeName(), mmHelper.Process());
        tilingData->mmQuantTilingData = mmHelper.GetAlltoAllvQuantHelperData();
        PrintGMMQuantTilingData(tilingData->mmQuantTilingData);
    }
    // permute out
    GetPermuteOutSize();
    OP_LOGD(context_->GetNodeName(), "end DoGmmTiling.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvMXQuantGmmTiling::CheckQuantGroupSize() const
{
    OP_LOGD(context_->GetNodeName(), "start CheckQuantGroupSize.");
    auto groupSizePtr = context_->GetAttrs()->GetAttrPointer<int64_t>(ATTR_GROUP_SIZE_INDEX);
    OP_TILING_CHECK(groupSizePtr == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "groupSize"),
        return ge::GRAPH_FAILED);
    uint64_t groupSizeK = static_cast<uint64_t>(*groupSizePtr) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeN = (static_cast<uint64_t>(*groupSizePtr) >> GROUP_N_OFFSET) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeM = (static_cast<uint64_t>(*groupSizePtr) >> GROUP_M_OFFSET) & GROUP_MNK_BIT_SIZE;
    OP_TILING_CHECK(((groupSizeM != MX_GROUP_SIZE_M && groupSizeM != 0) ||
                    (groupSizeN != MX_GROUP_SIZE_N && groupSizeN != 0) ||
                    (groupSizeK != MX_GROUP_SIZE_K && groupSizeK != 0)),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "groupSize",
                (std::string("[M=") + std::to_string(groupSizeM) + ", N=" + std::to_string(groupSizeN) +
                 ", K=" + std::to_string(groupSizeK) + "]").c_str(),
                "The value of groupSize M must be 1 or 0, N must be 1 or 0, K must be 32 or 0."), return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "end CheckQuantGroupSize.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvMXQuantGmmTiling::CheckQuantMode() const
{
    OP_LOGD(context_->GetNodeName(), "start CheckQuantMode.");
    // check gmmXQuantMode null
    OP_TILING_CHECK(gmmXQuantModePtr_ == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "gmmXQuantMode"),
        return ge::GRAPH_FAILED);
    // check gmmXQuantMode
    int64_t gmmXQuantMode = *gmmXQuantModePtr_;
    OP_TILING_CHECK(gmmXQuantMode != static_cast<int64_t>(MX_QUANT_MODE),
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "gmmXQuantMode",
        std::to_string(gmmXQuantMode).c_str(), "6"),
        return ge::GRAPH_FAILED);
    // check gmmWeightQuantMode null
    OP_TILING_CHECK(gmmWeightQuantModePtr_ == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "gmmWeightQuantMode"),
        return ge::GRAPH_FAILED);
    // check gmmWeightQuantMode
    int64_t gmmWeightQuantMode = *gmmWeightQuantModePtr_;
    OP_TILING_CHECK(gmmWeightQuantMode != static_cast<int64_t>(MX_QUANT_MODE),
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "gmmWeightQuantMode",
        std::to_string(gmmWeightQuantMode).c_str(), "6"), return ge::GRAPH_FAILED);
    if (hasSharedExpertFlag_) {
        // check mmXQuantMode null
        OP_TILING_CHECK(mmXQuantModePtr_ == nullptr,
            OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "mmXQuantMode"),
            return ge::GRAPH_FAILED);
        // mmXQuantMode(same as gmmXQuantMode)
        int64_t mmXQuantMode = *mmXQuantModePtr_;
        OP_TILING_CHECK(mmXQuantMode != gmmXQuantMode,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "mmXQuantMode",
                std::to_string(mmXQuantMode).c_str(),
                "The value of mmXQuantMode must be the same as gmmXQuantMode (6)."), return ge::GRAPH_FAILED);
        // check mmWeightQuantMode null
        OP_TILING_CHECK(mmWeightQuantModePtr_ == nullptr,
            OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "mmWeightQuantMode"),
            return ge::GRAPH_FAILED);
        // mmWeightQuantMode(same as gmmWeightQuantMode)
        int64_t mmWeightQuantMode = *mmWeightQuantModePtr_;
        OP_TILING_CHECK(mmWeightQuantMode != gmmWeightQuantMode,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "mmWeightQuantMode",
                std::to_string(mmWeightQuantMode).c_str(),
                "The value of mmWeightQuantMode must be the same as gmmWeightQuantMode (6)."),
            return ge::GRAPH_FAILED);
    }
    ge::graphStatus status = CheckQuantGroupSize();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context_->GetNodeName(), "end CheckQuantMode.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvMXQuantGmmTiling::CheckGmmScale() const
{
    // check gmmXScale null
    auto gmmXScaleDesc = context_->GetRequiredInputDesc(GMM_X_SCALE_INDEX);
    OP_TILING_CHECK(gmmXScaleDesc == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "gmmXScale"),
        return ge::GRAPH_FAILED);
    // check gmmXScale format
    OP_TILING_CHECK(gmmXScaleDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
        OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "gmmXScale",
        Ops::Base::ToString(gmmXScaleDesc->GetStorageFormat()), "ND"),
        return ge::GRAPH_FAILED);
    // check gmmXScale dataType
    OP_TILING_CHECK(gmmXScaleDesc->GetDataType() != ge::DT_FLOAT8_E8M0,
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "gmmXScale",
        ge::TypeUtils::DataTypeToSerialString(gmmXScaleDesc->GetDataType()).c_str(), "fp8_e8m0"),
        return ge::GRAPH_FAILED);
    // check gmmWeightScale null
    auto gmmWeightScaleDesc = context_->GetRequiredInputDesc(GMM_WEIGHT_SCALE_INDEX);
    OP_TILING_CHECK(gmmWeightScaleDesc == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "gmmWeightScale"),
        return ge::GRAPH_FAILED);
    // check gmmWeightScale format
    OP_TILING_CHECK(gmmWeightScaleDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
        OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "gmmWeightScale",
        Ops::Base::ToString(gmmWeightScaleDesc->GetStorageFormat()), "ND"),
        return ge::GRAPH_FAILED);
    // check gmmWeightScale dataType
    OP_TILING_CHECK(gmmWeightScaleDesc->GetDataType() != ge::DT_FLOAT8_E8M0,
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "gmmWeightScale",
        ge::TypeUtils::DataTypeToSerialString(gmmWeightScaleDesc->GetDataType()).c_str(), "fp8_e8m0"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvMXQuantGmmTiling::CheckMmScale() const
{
    // check mmXScale null
    auto mmXScaleDesc = context_->GetOptionalInputDesc(MM_X_SCALE_INDEX);
    auto gmmXScaleDesc = context_->GetRequiredInputDesc(GMM_X_SCALE_INDEX);
    OP_TILING_CHECK(mmXScaleDesc == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "mmXScale"),
        return ge::GRAPH_FAILED);
    // check mmXScale format
    OP_TILING_CHECK(mmXScaleDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
        OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "mmXScale",
        Ops::Base::ToString(mmXScaleDesc->GetStorageFormat()), "ND"),
        return ge::GRAPH_FAILED);
    // check mmXScale dataType(same as gmmXScale)
    OP_TILING_CHECK(mmXScaleDesc->GetDataType() != gmmXScaleDesc->GetDataType(),
        OP_LOGE(context_->GetNodeName(), "When mx quant mode and has shared expert, mmXScale should "
        "be same as gmmXScale(fp8_e8m0), but actual is %s.",
        ge::TypeUtils::DataTypeToSerialString(mmXScaleDesc->GetDataType()).c_str()),
        return ge::GRAPH_FAILED);
    // check mmWeightScale null
    auto mmWeightScaleDesc = context_->GetOptionalInputDesc(MM_WEIGHT_SCALE_INDEX);
    auto gmmWeightScaleDesc = context_->GetRequiredInputDesc(GMM_WEIGHT_SCALE_INDEX);
    OP_TILING_CHECK(mmWeightScaleDesc == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "mmWeightScale"),
        return ge::GRAPH_FAILED);
    // check mmWeightScale format
    OP_TILING_CHECK(mmWeightScaleDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
        OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "mmWeightScale",
        Ops::Base::ToString(mmWeightScaleDesc->GetStorageFormat()), "ND"),
        return ge::GRAPH_FAILED);
    // check mmWeightScale dataType(same as gmmWeightScale)
    OP_TILING_CHECK(mmWeightScaleDesc->GetDataType() != gmmWeightScaleDesc->GetDataType(),
        OP_LOGE(context_->GetNodeName(), "When mx quant mode and has shared expert, mmWeightScale should "
        "be same as gmmWeightScale(fp8_e8m0), but actual is %s.",
        ge::TypeUtils::DataTypeToSerialString(mmWeightScaleDesc->GetDataType()).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvMXQuantGmmTiling::CheckMmInputDtype(ge::DataType gmmXDataType,
    ge::DataType gmmWeightDataType, ge::DataType gmmYDataType) const
{
    // // check mmX dataType(same as gmmX)
    ge::DataType mmXDataType = context_->GetOptionalInputDesc(MM_X_INDEX)->GetDataType();
    OP_TILING_CHECK((mmXDataType != gmmXDataType),
        OP_LOGE(context_->GetNodeName(), "When mx quant mode, mmX should be same as gmmX(%s), "
        "but actual is %s.",
        ge::TypeUtils::DataTypeToSerialString(gmmXDataType).c_str(),
        ge::TypeUtils::DataTypeToSerialString(mmXDataType).c_str()),
        return ge::GRAPH_FAILED);
    // check mmWeight dataType(same as gmmWeight)
    ge::DataType mmWeightDataType = context_->GetOptionalInputDesc(MM_WEIGHT_INDEX)->GetDataType();
    OP_TILING_CHECK((mmWeightDataType != gmmWeightDataType),
        OP_LOGE(context_->GetNodeName(), "When mx quant mode, mmWeight should be same as gmmWeight(%s), "
        "but actual is %s.",
        ge::TypeUtils::DataTypeToSerialString(gmmWeightDataType).c_str(),
        ge::TypeUtils::DataTypeToSerialString(mmWeightDataType).c_str()),
        return ge::GRAPH_FAILED);
    // check mmY dataType(same as gmmY)
    ge::DataType mmYDataType = context_->GetOutputDesc(OUTPUT_MM_Y_INDEX)->GetDataType();
    OP_TILING_CHECK(mmYDataType != gmmYDataType,
        OP_LOGE(context_->GetNodeName(), "When mx quant mode, mmY should be same as gmmY(%s), "
        "but actual is %s.",
        ge::TypeUtils::DataTypeToSerialString(gmmYDataType).c_str(),
        ge::TypeUtils::DataTypeToSerialString(mmYDataType).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvMXQuantGmmTiling::CheckGmmInputDtype(ge::DataType gmmXDataType,
    ge::DataType gmmWeightDataType, ge::DataType gmmYDataType) const
{
    static const std::vector<ge::DataType> legalDtypes = {ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN,
        ge::DT_FLOAT4_E2M1};
    // check gmmX datatype
    OP_TILING_CHECK(std::find(legalDtypes.begin(), legalDtypes.end(), gmmXDataType) == legalDtypes.end(),
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "gmmX",
            ge::TypeUtils::DataTypeToSerialString(gmmXDataType).c_str(), "{fp8_e5m2, fp8_e4m3, fp4_e2m1}"),
            return ge::GRAPH_FAILED);
    // check gmmWeight datatype
    OP_TILING_CHECK(std::find(legalDtypes.begin(), legalDtypes.end(), gmmWeightDataType) == legalDtypes.end(),
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "gmmWeight",
            ge::TypeUtils::DataTypeToSerialString(gmmWeightDataType).c_str(), "{fp8_e5m2, fp8_e4m3, fp4_e2m1}"),
            return ge::GRAPH_FAILED);
    // check gmmY dataType
    OP_TILING_CHECK(gmmYDataType != ge::DT_FLOAT16 && gmmYDataType != ge::DT_BF16,
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "gmmY",
        ge::TypeUtils::DataTypeToSerialString(gmmYDataType).c_str(), "float16 or bfloat16"),
        return ge::GRAPH_FAILED);
    if (permuteOutFlag_) {
        // check permuteOut dtype
        ge::DataType permuteOutDataType = context_->GetOutputDesc(OUTPUT_PERMUTE_OUT_INDEX)->GetDataType();
        OP_TILING_CHECK(permuteOutDataType != gmmXDataType,
            OP_LOGE(context_->GetNodeName(), "When mx quant mode, permuteOut should be "
            "same as gmmX dataType(%s), but actual is %s.",
            ge::TypeUtils::DataTypeToSerialString(gmmXDataType).c_str(),
            ge::TypeUtils::DataTypeToSerialString(permuteOutDataType).c_str()),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvMXQuantGmmTiling::CheckFp4Input(ge::DataType gmmXDataType,
    ge::DataType gmmWeightDataType) const
{
    // mxfp4 gmmweight(same as gmmx)
    OP_TILING_CHECK(gmmXDataType != gmmWeightDataType,
        OP_LOGE(context_->GetNodeName(), "When mx quant mode and gmmx is fp4_e2m1, gmmWeight should "
        "be same as gmmx, but actual is %s.",
        ge::TypeUtils::DataTypeToSerialString(gmmWeightDataType).c_str()),
        return ge::GRAPH_FAILED);
    // mxfp4 h1 is even and h1 not two
    OP_TILING_CHECK(h1_ % 2 != 0 || h1_ == 2,
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "gmmX dim1(H1)",
        std::to_string(h1_), "even and not 2"), return ge::GRAPH_FAILED);
    // mxfp4 gmmweight not trans n1 is even
    OP_TILING_CHECK(!transGmmWeight_ && n1_ % 2 != 0,
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "gmmWeight dim(N1)",
        std::to_string(n1_), "even"), return ge::GRAPH_FAILED);
    // has mm
    if (hasSharedExpertFlag_) {
        // mxfp4 h2 is even and h2 not two
        OP_TILING_CHECK(h2_ % 2 != 0 || h2_ == 2,
            OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "mmX dim1(H2)",
            std::to_string(h2_), "even and not 2"), return ge::GRAPH_FAILED);
        // mxfp4 mmweight not trans n2 is even
        OP_TILING_CHECK(!transMmWeight_ && n2_ % 2 != 0,
            OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "mmWeight dim(N2)",
            std::to_string(n2_), "even"), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvMXQuantGmmTiling::CheckGmmScaleShape() const
{
    // check gmmXScale dimNum
    size_t gmmXScaleDimNum = context_->GetRequiredInputShape(GMM_X_SCALE_INDEX)->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(gmmXScaleDimNum != DIM_THREE,
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "gmmXScale",
        std::to_string(gmmXScaleDimNum), "3"), return ge::GRAPH_FAILED);
    // check gmmXScale shape
    uint64_t gmmExpectedH = Ops::Base::CeilDiv(h1_, MX_BASIC_FACTOR);
    uint64_t gmmXScaleDim0 = context_->GetRequiredInputShape(GMM_X_SCALE_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t gmmXScaleDim1 = context_->GetRequiredInputShape(GMM_X_SCALE_INDEX)->GetStorageShape().GetDim(DIM_ONE);
    uint64_t gmmXScaleDim2 = context_->GetRequiredInputShape(GMM_X_SCALE_INDEX)->GetStorageShape().GetDim(DIM_TWO);
    OP_TILING_CHECK((gmmXScaleDim0 != bsk_) || (gmmXScaleDim1 != gmmExpectedH) || (gmmXScaleDim2 != 2),
        OP_LOGE(context_->GetNodeName(), "When mx quant mode, the expected shape of gmmxscale is "
        "(%lu, %lu, 2) but the actual is (%lu, %lu, %lu)",
        bsk_, gmmExpectedH, gmmXScaleDim0, gmmXScaleDim1, gmmXScaleDim2),
        return ge::GRAPH_FAILED);
    // check gmmWeightScale dimNum
    size_t gmmWeightScaleDimNum =
        context_->GetRequiredInputShape(GMM_WEIGHT_SCALE_INDEX)->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(gmmWeightScaleDimNum != DIM_FOUR,
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "gmmWeightScale",
        std::to_string(gmmWeightScaleDimNum), "4"), return ge::GRAPH_FAILED);
    // check gmmWeightScale shape
    uint64_t gmmWeightScaleDim0 =
        context_->GetRequiredInputShape(GMM_WEIGHT_SCALE_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t gmmWeightScaleDim1 =
        context_->GetRequiredInputShape(GMM_WEIGHT_SCALE_INDEX)->GetStorageShape().GetDim(DIM_ONE);
    uint64_t gmmWeightScaleDim2 =
        context_->GetRequiredInputShape(GMM_WEIGHT_SCALE_INDEX)->GetStorageShape().GetDim(DIM_TWO);
    uint64_t gmmWeightScaleDim3 =
        context_->GetRequiredInputShape(GMM_WEIGHT_SCALE_INDEX)->GetStorageShape().GetDim(DIM_THREE);
    uint64_t gmmWeightDimH = transGmmWeight_ ? gmmWeightScaleDim2 : gmmWeightScaleDim1;
    uint64_t gmmWeightDimN = transGmmWeight_ ? gmmWeightScaleDim1 : gmmWeightScaleDim2;
    if (transGmmWeight_) {
        OP_TILING_CHECK((gmmWeightDimN != n1_) || (gmmWeightDimH != gmmExpectedH) ||
            (gmmWeightScaleDim0 != e_) || (gmmWeightScaleDim3 != 2),
            OP_LOGE(context_->GetNodeName(), "When mx quant mode and trans gmmWeight, the expected "
            "shape of gmmWeightscale is (%lu, %lu, %lu, 2) but the actual is (%lu, %lu, %lu, %lu)",
            e_, n1_, gmmExpectedH, gmmWeightScaleDim0, gmmWeightDimN, gmmWeightDimH, gmmWeightScaleDim3),
            return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK((gmmWeightDimN != n1_) || (gmmWeightDimH != gmmExpectedH) ||
            (gmmWeightScaleDim0 != e_) || (gmmWeightScaleDim3 != 2),
            OP_LOGE(context_->GetNodeName(), "When mx quant mode and not trans gmmWeight, the expected "
            "shape of gmmWeightscale is (%lu, %lu, %lu, 2) but the actual is (%lu, %lu, %lu, %lu)",
            e_, gmmExpectedH, n1_, gmmWeightScaleDim0, gmmWeightDimH, gmmWeightDimN, gmmWeightScaleDim3),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvMXQuantGmmTiling::CheckMmScaleShape() const
{
    // check mmXScale dimNum
    size_t mmXScaleDimNum = context_->GetOptionalInputShape(MM_X_SCALE_INDEX)->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(mmXScaleDimNum != context_->GetRequiredInputShape(GMM_X_SCALE_INDEX)->GetStorageShape().GetDimNum(),
        OP_LOGE(context_->GetNodeName(), "When mx quant mode and has shared expert, mmXScaleDimNum input dimNum "
        "should be 3, but actual dimNum is %lu.", mmXScaleDimNum), return ge::GRAPH_FAILED);
    // check mmXScale shape
    uint64_t mmExpectedH = Ops::Base::CeilDiv(h2_, MX_BASIC_FACTOR);
    uint64_t mmXScaleDim0 = context_->GetOptionalInputShape(MM_X_SCALE_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t mmXScaleDim1 = context_->GetOptionalInputShape(MM_X_SCALE_INDEX)->GetStorageShape().GetDim(DIM_ONE);
    uint64_t mmXScaleDim2 = context_->GetOptionalInputShape(MM_X_SCALE_INDEX)->GetStorageShape().GetDim(DIM_TWO);
    OP_TILING_CHECK((mmXScaleDim0 != bs_) || (mmXScaleDim1 != mmExpectedH) || (mmXScaleDim2 != 2),
        OP_LOGE(context_->GetNodeName(), "When mx quant mode and has shared expert, the expected shape "
        "of mmxscale is (%lu, %lu, 2) but the actual is (%lu, %lu, %lu)",
        bs_, mmExpectedH, mmXScaleDim0, mmXScaleDim1, mmXScaleDim2), return ge::GRAPH_FAILED);
    // check mmWeightScale dimNum
    size_t mmWeightScaleDimNum = context_->GetOptionalInputShape(MM_WEIGHT_SCALE_INDEX)->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(mmWeightScaleDimNum != DIM_THREE,
        OP_LOGE(context_->GetNodeName(), "When mx quant mode and has shared expert, mmWeightScaleDimNum input dimNum "
        "should be 3, but actual is %lu.", mmWeightScaleDimNum), return ge::GRAPH_FAILED);
    // check mmWeightScale shape
    uint64_t mmWeightScaleDim0 =
        context_->GetOptionalInputShape(MM_WEIGHT_SCALE_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t mmWeightScaleDim1 =
        context_->GetOptionalInputShape(MM_WEIGHT_SCALE_INDEX)->GetStorageShape().GetDim(DIM_ONE);
    uint64_t mmWeightScaleDim2 =
        context_->GetOptionalInputShape(MM_WEIGHT_SCALE_INDEX)->GetStorageShape().GetDim(DIM_TWO);
    uint64_t mmWeightDimH = transMmWeight_ ? mmWeightScaleDim1 : mmWeightScaleDim0;
    uint64_t mmWeightDimN = transMmWeight_ ? mmWeightScaleDim0 : mmWeightScaleDim1;
    if (transMmWeight_) {
        OP_TILING_CHECK((mmWeightDimN != n2_) || (mmWeightDimH != mmExpectedH) || (mmWeightScaleDim2 != 2),
            OP_LOGE(context_->GetNodeName(), "When mx quant mode and trans mmWeight, the expected shape of "
            "mmWeightscale is (%lu, %lu, 2) but the actual is (%lu, %lu, %lu)",
            n2_, mmExpectedH, mmWeightDimN, mmWeightDimH, mmWeightScaleDim2),
            return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK((mmWeightDimN != n2_) || (mmWeightDimH != mmExpectedH) || (mmWeightScaleDim2 != 2),
            OP_LOGE(context_->GetNodeName(), "When mx quant mode and not trans mmWeight, "
            "the expected shape of mmWeightscale is (%lu, %lu, 2) but the actual is (%lu, %lu, %lu)",
            mmExpectedH, n2_, mmWeightDimH, mmWeightDimN, mmWeightScaleDim2),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

void AlltoAllvMXQuantGmmTiling::GetPermuteOutSize()
{
    // commOutSize: 通信输出区域，kernel始终从workspace偏移0分配此空间存放通信结果
    permuteOutSize_ = a_ * h1_;
    if (gmmXDataType_ == ge::DT_FLOAT4_E2M1) {
        // mxfp4 场景占半字节，shapesize大小除以二
        permuteOutSize_ = Ops::Base::CeilDiv(permuteOutSize_, MXFP4_PACK_FACTOR);
    }
    // 将 permuteOutSize 对齐到 512 字节
    permuteOutSize_ = Ops::Base::CeilAlign(permuteOutSize_, static_cast<uint64_t>(BASIC_BLOCK_SIZE_512));
    // permutscaleout size
    uint64_t hSize = Ops::Base::CeilDiv(h1_, MX_BASIC_FACTOR);
    permuteScaleOutSize_ =  Ops::Base::CeilAlign((a_ * hSize * 2 * GetSizeByDataType(gmmWeightDataType_)),
        static_cast<uint64_t>(BASIC_BLOCK_SIZE_512));
    // 如果单卡专家数大于等于32时，需要额外申请scale重排空间
    if (e_ >= SCALE_BATCH_THRESHOLD) {
        permuteScaleOutSize_ = permuteScaleOutSize_ * 2;
    }
}

ge::graphStatus AlltoAllvMXQuantGmmTilingHelper::SetInputParams(uint64_t M, uint64_t N, uint64_t K, bool transB)
{
    OP_LOGD(context_->GetNodeName(), "start SetInputParams.");
    GetPlatformInfo();
    inputParams_.opName = context_->GetNodeName();
    inputParams_.kernelType = 0UL;
    // 输出是否切分，0/1代表输出多tensor， 2/3代表输出单tensor
    inputParams_.splitItem = 0;
    inputParams_.actType = GMM_ACT_TYPE_NONE;
    // common
    inputParams_.aFormat = ge::FORMAT_ND;
    inputParams_.bFormat = ge::FORMAT_ND;
    inputParams_.cFormat = ge::FORMAT_ND;
    inputParams_.transA = 0;
    inputParams_.transB = transB;
    inputParams_.hasBias = 0;
    inputParams_.isSingleX = 1;
    inputParams_.isSingleW = 1;
    inputParams_.isSingleY = 1;

    inputParams_.mSize = M;
    inputParams_.kSize = K;
    inputParams_.nSize = N;
    inputParams_.groupNum = gmmGroupNum_;
    inputParams_.aQuantMode = static_cast<Mc2GroupedMatmulTiling::QuantMode>(1U << QUANT_MODE_MAP[MX_MODE]);
    inputParams_.bQuantMode = static_cast<Mc2GroupedMatmulTiling::QuantMode>(1U << QUANT_MODE_MAP[MX_MODE]);
    // 是否做切分
    inputParams_.groupType = optiling::Mc2GroupedMatmul::SPLIT_M;
    inputParams_.groupListType = 1;
    inputParams_.aDtype = context_->GetInputDesc(GMM_X_INDEX)->GetDataType();
    inputParams_.bDtype = context_->GetInputDesc(GMM_WEIGHT_INDEX)->GetDataType();
    inputParams_.cDtype = context_->GetInputDesc(OUTPUT_GMM_Y_INDEX)->GetDataType();
    inputParams_.biasDtype = ge::DT_INT32;
    inputParams_.scaleDtype = ge::DT_FLOAT8_E8M0;
    inputParams_.perTokenScaleDtype = ge::DT_FLOAT8_E8M0;

    OP_TILING_CHECK(gmmGroupNum_ > optiling::Mc2GroupedMatmul::MAX_TENSOR_CONT,
        OP_LOGE_FOR_INVALID_VALUE(inputParams_.opName, "gmmGroupNum_",
            std::to_string(gmmGroupNum_).c_str(),
            (std::string("<=") + std::to_string(optiling::Mc2GroupedMatmul::MAX_TENSOR_CONT)).c_str()),
        return ge::GRAPH_FAILED);
    for (uint32_t i = 0; i < gmmGroupNum_; i++) {
        mList_[i] = static_cast<int32_t>(M);
    }
    kList_[0] = static_cast<int32_t>(K);
    nList_[0] = static_cast<int32_t>(N);
    SetKernelType();
    
    OP_LOGD(context_->GetNodeName(), "end SetInputParams.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvMXQuantGmmTilingHelper::Process()
{
    MC2_CHECK_LOG_RET(context_->GetNodeName(), DoOpTiling());
    MC2_CHECK_LOG_RET(context_->GetNodeName(), DoLibApiTiling());
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(AlltoAllvQuantGroupedMatMul, AlltoAllvMXQuantGmmTiling, 2);
} // namespace optiling