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
 * \file rope_regbase_tiling_base.cc
 * \brief
 */

#include <graph/utils/type_utils.h>
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"
#include "partial_rotary_mul_quant_tiling.h"
#include "log/log.h"

namespace {
constexpr int64_t X_INDEX = 0;
constexpr int64_t COS_INDEX = 1;
constexpr int64_t SIN_INDEX = 2;
constexpr int64_t SCALE_INDEX = 3;
constexpr int64_t Y_INDEX = 0;
constexpr int64_t Y_SCALE_INDEX = 1;
constexpr int64_t DIM_NUM = 4;
constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;
constexpr int64_t HALF_INTERLEAVE_MODE_COEF = 2;
constexpr int64_t QUARTER_MODE_COEF = 4;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t D_LIMIT = 1024;
constexpr int64_t ATTR_IDX_Y_DTYPE = 0;
constexpr int64_t ATTR_IDX_AXIS = -1;
constexpr int32_t ATTR_IDX_ROTARY_MODE = 4;
constexpr int32_t ATTR_IDX_SLICE = 5;
constexpr int32_t HIFLOAT4_QUANT_GROUP = 64;
constexpr int32_t WORKSPACE_SIZE = 16 * 1024 * 1024;
const std::vector<ge::DataType> SUPPORT_DTYPE = {ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16};
} // namespace

namespace optiling {
ge::graphStatus QuantRopeRegBaseTilingClass::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aicoreParams_.blockDim = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        aicoreParams_.ubSize = ubSizePlatForm;
        socVersion_ = ascendcPlatform.GetSocVersion();
    } else {
        auto compileInfoPtr =
            reinterpret_cast<const QuantRotaryPositionEmbeddingCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_->GetNodeName(), "compile info is null"),
                    return ge::GRAPH_FAILED);
        aicoreParams_.blockDim = compileInfoPtr->blockDim;
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
        socVersion_ = compileInfoPtr->socVersion;
    }
    blockSize_ = BLOCK_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantRopeRegBaseTilingClass::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantRopeRegBaseTilingClass::GetWorkspaceSize()
{
    workspaceSize_ = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantRopeRegBaseTilingClass::CheckNullptr()
{
    for (int64_t i = 0; i <= SIN_INDEX; i++) {
        auto desc = context_->GetInputDesc(i);
        OP_CHECK_IF(desc == nullptr, OP_LOGE(context_->GetNodeName(), "input %ld desc is nullptr.", i),
                    return ge::GRAPH_FAILED);
        auto shape = context_->GetInputShape(i);
        OP_CHECK_IF(shape == nullptr, OP_LOGE(context_->GetNodeName(), "input %ld shape is nullptr.", i),
                    return ge::GRAPH_FAILED);
    }
    auto yDesc = context_->GetOutputDesc(Y_INDEX);
    OP_CHECK_IF(yDesc == nullptr, OP_LOGE(context_->GetNodeName(), "output desc is nullptr."), return ge::GRAPH_FAILED);
    auto yShape = context_->GetOutputShape(Y_INDEX);
    OP_CHECK_IF(yShape == nullptr, OP_LOGE(context_->GetNodeName(), "output shape is nullptr."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantRopeRegBaseTilingClass::CheckShapeAllPositive(const int64_t idx) const
{
    auto shape = context_->GetInputShape(idx)->GetStorageShape();
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        OP_CHECK_IF(shape.GetDim(i) <= 0,
                    OP_LOGE(context_->GetNodeName(), "input %ld has non positive shape, dim %lu actual %ld .", idx, i,
                            shape.GetDim(i)),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}
bool QuantRopeRegBaseTilingClass::IsRotaryPosEmbeddingMode(const int32_t mode) const
{
    switch (mode) {
        case static_cast<int32_t>(RotaryPosEmbeddingMode::HALF):
        case static_cast<int32_t>(RotaryPosEmbeddingMode::INTERLEAVE):
        case static_cast<int32_t>(RotaryPosEmbeddingMode::QUARTER):
        case static_cast<int32_t>(RotaryPosEmbeddingMode::DEEPSEEK_INTERLEAVE):
            return true;
        default:
            return false;
    }
}

ge::graphStatus QuantRopeRegBaseTilingClass::CheckShapeAllPositive() const
{
    OP_CHECK_IF(CheckShapeAllPositive(X_INDEX) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "x has non positive shape."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShapeAllPositive(COS_INDEX) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "cos has non positive shape."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShapeAllPositive(SIN_INDEX) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "sin has non positive shape."), return ge::GRAPH_FAILED);
    auto yShape = context_->GetOutputShape(Y_INDEX)->GetStorageShape();
    for (size_t i = 0; i < yShape.GetDimNum(); i++) {
        OP_CHECK_IF(yShape.GetDim(i) <= 0,
                    OP_LOGE(context_->GetNodeName(), "output has non positive shape, dim %lu actual %ld .", i,
                            yShape.GetDim(i)),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantRopeRegBaseTilingClass::JudgeLayoutByShape(const gert::Shape& xShape, const gert::Shape& cosShape)
{
    uint64_t xShape0 = xShape.GetDim(DIM_0);
    uint64_t xShape1 = xShape.GetDim(DIM_1);
    uint64_t xShape2 = xShape.GetDim(DIM_2);
    uint64_t cosShape0 = cosShape.GetDim(DIM_0);
    uint64_t cosShape1 = cosShape.GetDim(DIM_1);
    uint64_t cosShape2 = cosShape.GetDim(DIM_2);
    if (xShape0 == cosShape0 && xShape1 == cosShape1 && xShape2 == cosShape2) { // BSND
        layout_ = RopeLayout::NO_BROADCAST;
    } else if (cosShape0 == 1 && cosShape1 == 1 && cosShape2 == 1) { // (111D)
        layout_ = RopeLayout::BROADCAST_BSN;
    } else if (cosShape2 == 1 && xShape0 == cosShape0 && (cosShape1 == 1 || cosShape1 == xShape1)) { // SBND (S11D,
                                                                                                     // SB1D), BSND
                                                                                                     // (BS1D)
        layout_ = RopeLayout::SBND;
    } else if (cosShape2 == 1 && cosShape0 == 1 && xShape1 == cosShape1) { // BSND (1S1D)
        layout_ = RopeLayout::BSND;
    } else if (cosShape1 == 1 && xShape2 == cosShape2 && (cosShape0 == 1 || cosShape0 == xShape0)) { // BNSD (11SD,
                                                                                                     // B1SD)
        layout_ = RopeLayout::BNSD;
    } else if (cosShape0 == 1 && xShape1 == cosShape1 && xShape2 == cosShape2) { // 1SND
        layout_ = RopeLayout::BNSD;
        is1snd_ = true;
    } else {
        OP_LOGE(context_->GetNodeName(), "the shape of x and sin not satisfy the broadcast.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantRopeRegBaseTilingClass::CheckShape()
{
    auto& xShape = context_->GetInputShape(X_INDEX)->GetStorageShape();
    auto& cosShape = context_->GetInputShape(COS_INDEX)->GetStorageShape();
    auto& sinShape = context_->GetInputShape(SIN_INDEX)->GetStorageShape();
    auto& yShape = context_->GetOutputShape(Y_INDEX)->GetStorageShape();
    OP_CHECK_IF(xShape.GetDimNum() != DIM_NUM,
                OP_LOGE(context_->GetNodeName(), "dim of x expect 4, actual %lu.", xShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(cosShape.GetDimNum() != DIM_NUM,
                OP_LOGE(context_->GetNodeName(), "dim of cos expect 4, actual %lu.", cosShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(sinShape.GetDimNum() != DIM_NUM,
                OP_LOGE(context_->GetNodeName(), "dim of sin expect 4, actual %lu.", sinShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(yShape.GetDimNum() != DIM_NUM,
                OP_LOGE(context_->GetNodeName(), "dim of output expect 4, actual %lu.", yShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    auto& yScaleShape = context_->GetOutputShape(Y_SCALE_INDEX)->GetStorageShape();
    OP_CHECK_IF(
        yScaleShape.GetDimNum() != DIM_NUM + 1,
        OP_LOGE(context_->GetNodeName(), "dim of y_scale output expect 5, actual %lu.", yScaleShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(cosShape != sinShape,
                OP_LOGE(context_->GetNodeName(),
                        "shape of cos and sin should be same, actual cos shape is (%ld, %ld, %ld, %ld), sin shape is "
                        "(%ld, %ld, %ld, %ld). ",
                        cosShape.GetDim(DIM_0), cosShape.GetDim(DIM_1), cosShape.GetDim(DIM_2), cosShape.GetDim(DIM_3),
                        sinShape.GetDim(DIM_0), sinShape.GetDim(DIM_1), sinShape.GetDim(DIM_2), sinShape.GetDim(DIM_3)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(xShape != yShape,
                OP_LOGE(context_->GetNodeName(),
                        "shape of x and output should be2 same, actual x shape is (%ld, "
                        "%ld, %ld, %ld), output shape is (%ld, %ld, %ld, %ld). ",
                        xShape.GetDim(DIM_0), xShape.GetDim(DIM_1), xShape.GetDim(DIM_2), xShape.GetDim(DIM_3),
                        yShape.GetDim(DIM_0), yShape.GetDim(DIM_1), yShape.GetDim(DIM_2), yShape.GetDim(DIM_3)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckRotaryModeShapeRelation(xShape.GetDim(DIM_3)) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "D is invalid for rotary mode."), return ge::GRAPH_FAILED);
    return CheckShapeAllPositive();
}

ge::graphStatus QuantRopeRegBaseTilingClass::CheckDtypeAndAttr()
{
    dtype_ = context_->GetInputDesc(X_INDEX)->GetDataType();
    OP_CHECK_IF(std::find(SUPPORT_DTYPE.begin(), SUPPORT_DTYPE.end(), dtype_) == SUPPORT_DTYPE.end(),
                OP_LOGE(context_->GetNodeName(), "Only support F32, BF16, F16 datetype, actual %s.",
                        ge::TypeUtils::DataTypeToSerialString(dtype_).c_str()),
                return ge::GRAPH_FAILED);
    for (int64_t i = X_INDEX; i <= SIN_INDEX; i++) {
        auto type = context_->GetInputDesc(i)->GetDataType();
        OP_CHECK_IF(type != dtype_,
                    OP_LOGE(context_->GetNodeName(), "input %ld datatype expect %s, actual %s.", i,
                            ge::TypeUtils::DataTypeToSerialString(dtype_).c_str(),
                            ge::TypeUtils::DataTypeToSerialString(type).c_str()),
                    return ge::GRAPH_FAILED);
    }
    // output is always uint8 (hifloat4 on wire)
    auto outputType = context_->GetOutputDesc(Y_INDEX)->GetDataType();
    OP_CHECK_IF(outputType != ge::DT_HIFLOAT4,
                OP_LOGE(context_->GetNodeName(), "output datatype expect hfloat4, actual %s.",
                        ge::TypeUtils::DataTypeToSerialString(outputType).c_str()),
                return ge::GRAPH_FAILED);
    auto yScaleType = context_->GetOutputDesc(Y_SCALE_INDEX)->GetDataType();
    OP_CHECK_IF(yScaleType != ge::DT_HIFLOAT4_SCALE,
                OP_LOGE(context_->GetNodeName(), "y_scale output datatype expect hifloat4_scale, actual %s.",
                        ge::TypeUtils::DataTypeToSerialString(yScaleType).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantRopeRegBaseTilingClass::CheckParam()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context_->GetNodeName(), "platform info is nullptr."),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    if (!IsRegbaseSocVersion()) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(CheckNullptr() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "check nullptr fail."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckDtypeAndAttr() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "check dtype and attr fail."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShape() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "check shape fail."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantRopeRegBaseTilingClass::CheckRotaryModeShapeRelation(const int64_t d)
{
    OP_CHECK_IF(d > D_LIMIT, OP_LOGE(context_->GetNodeName(), "D must be small than %ld, actual %ld.", D_LIMIT, d),
                return ge::GRAPH_FAILED);
    if (rotaryMode_ == RotaryPosEmbeddingMode::HALF || rotaryMode_ == RotaryPosEmbeddingMode::INTERLEAVE ||
        rotaryMode_ == RotaryPosEmbeddingMode::DEEPSEEK_INTERLEAVE) {
        OP_CHECK_IF(d % HALF_INTERLEAVE_MODE_COEF != 0,
                    OP_LOGE(context_->GetNodeName(),
                            "D must be multiples of 2 in half, interleave and interleave-half mode, actual %ld.", d),
                    return ge::GRAPH_FAILED);
    } else if (rotaryMode_ == RotaryPosEmbeddingMode::QUARTER) {
        OP_CHECK_IF(d % QUARTER_MODE_COEF != 0,
                    OP_LOGE(context_->GetNodeName(), "D must be multiples of 4 in quarter mode, actual %ld.", d),
                    return ge::GRAPH_FAILED);
    }
    if (rotaryMode_ == RotaryPosEmbeddingMode::HALF || rotaryMode_ == RotaryPosEmbeddingMode::DEEPSEEK_INTERLEAVE) {
        dSplitCoef_ = HALF_INTERLEAVE_MODE_COEF;
    } else if (rotaryMode_ == RotaryPosEmbeddingMode::QUARTER) {
        dSplitCoef_ = QUARTER_MODE_COEF;
    } else {
        dSplitCoef_ = 1;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantRopeRegBaseTilingClass::JudgeSliceInfo()
{
    OP_LOGI(context_->GetNodeName(), "TEST LOG, sliceStart_ =  %ld. sliceEnd_ =  %ld", sliceStart_, sliceEnd_);
    if (sliceStart_ < 0 || sliceEnd_ < 0 || sliceLength_ <= 0 || sliceEnd_ > d_) {
        OP_LOGE(context_->GetNodeName(), "slice info fail, sliceStart_ =  %ld. sliceEnd_ =  %ld", sliceStart_,
                sliceEnd_);
        return ge::GRAPH_FAILED;
    }
    if (cosd_ != sind_ || cosd_ != sliceLength_) {
        OP_LOGE(context_->GetNodeName(), "slice info fail, sliceLength_ =  %ld. cosd_ =  %ld", sliceLength_, cosd_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantRopeRegBaseTilingClass::CheckHiF4Support()
{
    if (d_ % HIFLOAT4_QUANT_GROUP != 0) {
        OP_LOGE(context_->GetNodeName(), "d should be divided by 64, but get %ld", d_);
        return ge::GRAPH_FAILED;
    }
    if (sliceStart_ % HIFLOAT4_QUANT_GROUP != 0) {
        OP_LOGE(context_->GetNodeName(), "slice start should be divided by 64, but get %ld", sliceStart_);
        return ge::GRAPH_FAILED;
    }
    if (sliceEnd_ != d_) {
        OP_LOGE(context_->GetNodeName(), "slice end should be equals with last dim of d, but get %ld", sliceEnd_);
        return ge::GRAPH_FAILED;
    }
    const gert::RuntimeAttrs* attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "attrs is null"), return ge::GRAPH_FAILED);
    const int32_t* yDtype = attrs->GetAttrPointer<int32_t>(ATTR_IDX_Y_DTYPE);
    int32_t yDtypeValue = (yDtype == nullptr) ? 0 : static_cast<int32_t>(*yDtype);
    if (yDtypeValue != 1) {
        OP_LOGE(context_->GetNodeName(), "y_dtype only support 1(hifloat4), but get %ld", yDtypeValue);
        return ge::GRAPH_FAILED;
    }
    if (rotaryMode_ != RotaryPosEmbeddingMode::HALF && rotaryMode_ != RotaryPosEmbeddingMode::INTERLEAVE) {
        OP_LOGE(context_->GetNodeName(), "rotary_mode only support 0(half) or 1(interleave) now, but get %ld",
                rotaryMode_);
        return ge::GRAPH_FAILED;
    }

    if (layout_ != RopeLayout::NO_BROADCAST && layout_ != RopeLayout::BROADCAST_BSN && layout_ != RopeLayout::SBND) {
        OP_LOGE(context_->GetNodeName(), "input shape only support x(BSND) and cos(BSND/111D/B11D/BS1D) now.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantRopeRegBaseTilingClass::GetShapeAttrsInfo()
{
    const gert::RuntimeAttrs* attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "attrs is null"), return ge::GRAPH_FAILED);
    const int32_t* rotaryMode = attrs->GetAttrPointer<int32_t>(ATTR_IDX_ROTARY_MODE);
    int32_t modeValue = (rotaryMode == nullptr) ? 0 : static_cast<int32_t>(*rotaryMode);
    OP_CHECK_IF(IsRotaryPosEmbeddingMode(modeValue) != true,
                OP_LOGE(context_->GetNodeName(), "rotary_mode only support 0, 1, 2 3, actual %d.", modeValue),
                return ge::GRAPH_FAILED);
    rotaryMode_ = static_cast<RotaryPosEmbeddingMode>(modeValue);

    OP_CHECK_IF(CheckParam() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "check param fail."),
                return ge::GRAPH_FAILED);

    dtype_ = context_->GetInputDesc(X_INDEX)->GetDataType();
    auto& xShape = context_->GetInputShape(X_INDEX)->GetStorageShape();
    auto& cosShape = context_->GetInputShape(COS_INDEX)->GetStorageShape();
    auto& sinShape = context_->GetInputShape(SIN_INDEX)->GetStorageShape();
    OP_CHECK_IF(JudgeLayoutByShape(xShape, cosShape) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "JudgeLayoutByShape fail."), return ge::GRAPH_FAILED);

    d_ = xShape.GetDim(DIM_3);
    cosd_ = cosShape.GetDim(DIM_3);
    sind_ = sinShape.GetDim(DIM_3);
    if (layout_ == RopeLayout::BSND) {
        b_ = xShape.GetDim(DIM_0);
        cosb_ = cosShape.GetDim(DIM_0);
        s_ = xShape.GetDim(DIM_1);
        n_ = xShape.GetDim(DIM_2);
    } else if (layout_ == RopeLayout::BNSD || layout_ == RopeLayout::NO_BROADCAST ||
               layout_ == RopeLayout::BROADCAST_BSN) {
        b_ = xShape.GetDim(DIM_0);
        cosb_ = cosShape.GetDim(DIM_0);
        n_ = xShape.GetDim(DIM_1);
        s_ = xShape.GetDim(DIM_2);
        // 1XXX情况下，reshape成11XX
        if (is1snd_) {
            s_ = s_ * n_;
            n_ = 1;
        }
    } else if (layout_ == RopeLayout::SBND) {
        s_ = xShape.GetDim(DIM_0);
        b_ = xShape.GetDim(DIM_1);
        cosb_ = cosShape.GetDim(DIM_1);
        n_ = xShape.GetDim(DIM_2);
    }

    // 获取slice
    const gert::ContinuousVector* sliceRangeListPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_IDX_SLICE);
    if (sliceRangeListPtr->GetSize() == 0) {
        sliceStart_ = 0;
        sliceEnd_ = d_;
    } else {
        const int64_t* expertRangeList = reinterpret_cast<const int64_t*>(sliceRangeListPtr->GetData());
        sliceStart_ = expertRangeList[0];
        sliceEnd_ = expertRangeList[1];
    }
    sliceLength_ = sliceEnd_ - sliceStart_;
    OP_CHECK_IF(JudgeSliceInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "JudgeSliceInfo fail."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckHiF4Support() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "check hif4 support fail."),
                return ge::GRAPH_FAILED);

    scaleAttr_ = 1.0f;

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
