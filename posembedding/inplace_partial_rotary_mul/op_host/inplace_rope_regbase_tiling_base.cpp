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
 * \file inplace_rope_regbase_tiling_base.cc
 * \brief
 */

#include "tiling/tiling_api.h"
#include "platform/platform_info.h"
#include "inplace_partial_rotary_mul_tiling.h"
#include <graph/utils/type_utils.h>

namespace {
constexpr int64_t X_INDEX = 0;
constexpr int64_t COS_INDEX = 1;
constexpr int64_t SIN_INDEX = 2;
constexpr int64_t Y_INDEX = 0;
constexpr int64_t DIM_NUM = 4;
constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;
constexpr int64_t HALF_INTERLEAVE_MODE_COEF = 2;
constexpr int64_t QUARTER_MODE_COEF = 4;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t D_LIMIT = 1024;
const std::vector<ge::DataType> SUPPORT_DTYPE = {ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16};
} // namespace

namespace optiling {
using namespace Ops::Base;
ge::graphStatus InplacePartialRopeRegBaseTilingClass::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aicoreParams_.blockDim = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        aicoreParams_.ubSize = ubSizePlatForm;
    } else {
        auto compileInfoPtr =
            reinterpret_cast<const InplacePartialRotaryPositionEmbeddingCompileInfo *>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
        aicoreParams_.blockDim = compileInfoPtr->blockDim;
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
    }
    blockSize_ = BLOCK_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRopeRegBaseTilingClass::CheckNullptr()
{
    for (int64_t i = 0; i <= SIN_INDEX; i++) {
        auto desc = context_->GetInputDesc(i);
        OP_CHECK_IF(desc == nullptr, OP_LOGE(context_, "input %ld desc is nullptr.", i), return ge::GRAPH_FAILED);
        auto shape = context_->GetInputShape(i);
        OP_CHECK_IF(shape == nullptr, OP_LOGE(context_, "input %ld shape is nullptr.", i), return ge::GRAPH_FAILED);
    }
    auto yDesc = context_->GetOutputDesc(Y_INDEX);
    OP_CHECK_IF(yDesc == nullptr, OP_LOGE(context_, "output desc is nullptr."), return ge::GRAPH_FAILED);
    auto yShape = context_->GetOutputShape(Y_INDEX);
    OP_CHECK_IF(yShape == nullptr, OP_LOGE(context_, "output shape is nullptr."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRopeRegBaseTilingClass::CheckShapeAllPositive(const int64_t idx) const
{
    auto shape = context_->GetInputShape(idx)->GetStorageShape();
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        if (shape.GetDim(i) <= 0) {
            std::string reasonMsg = "All dimensions of input must be positive, but dim " + std::to_string(i) + " is " +
                                    std::to_string(shape.GetDim(i));
            const char *tensorName = (idx == X_INDEX) ? "x" : (idx == COS_INDEX) ? "cos" : "sin";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), tensorName, ToString(shape).c_str(),
                                                  reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}
bool InplacePartialRopeRegBaseTilingClass::IsInplacePartialRotaryPosEmbeddingMode(const int32_t mode) const
{
    switch (mode) {
        case static_cast<int32_t>(InplacePartialRotaryPosEmbeddingMode::HALF):
        case static_cast<int32_t>(InplacePartialRotaryPosEmbeddingMode::INTERLEAVE):
        case static_cast<int32_t>(InplacePartialRotaryPosEmbeddingMode::QUARTER):
        case static_cast<int32_t>(InplacePartialRotaryPosEmbeddingMode::DEEPSEEK_INTERLEAVE):
            return true;
        default:
            return false;
    }
}

ge::graphStatus InplacePartialRopeRegBaseTilingClass::CheckShapeAllPositive() const
{
    if (CheckShapeAllPositive(X_INDEX) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckShapeAllPositive(COS_INDEX) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckShapeAllPositive(SIN_INDEX) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto yShape = context_->GetOutputShape(Y_INDEX)->GetStorageShape();
    for (size_t i = 0; i < yShape.GetDimNum(); i++) {
        if (yShape.GetDim(i) <= 0) {
            std::string reasonMsg = "All dimensions of output must be positive, but dim " + std::to_string(i) + " is " +
                                    std::to_string(yShape.GetDim(i));
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "output", ToString(yShape).c_str(),
                                                  reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRopeRegBaseTilingClass::JudgeLayoutByShape(const gert::Shape &xShape,
                                                                         const gert::Shape &cosShape)
{
    uint64_t xShape0 = xShape.GetDim(DIM_0);
    uint64_t xShape1 = xShape.GetDim(DIM_1);
    uint64_t xShape2 = xShape.GetDim(DIM_2);
    uint64_t cosShape0 = cosShape.GetDim(DIM_0);
    uint64_t cosShape1 = cosShape.GetDim(DIM_1);
    uint64_t cosShape2 = cosShape.GetDim(DIM_2);
    if (xShape0 == cosShape0 && xShape1 == cosShape1 && xShape2 == cosShape2) { // BSND
        layout_ = InplacePartialRopeLayout::NO_BROADCAST;
    } else if (cosShape0 == 1 && cosShape1 == 1 && cosShape2 == 1) { // (111D)
        layout_ = InplacePartialRopeLayout::BROADCAST_BSN;
    } else if (cosShape2 == 1 && cosShape0 == 1 && xShape1 == cosShape1) { // BSND (1S1D)
        layout_ = InplacePartialRopeLayout::BSND;
    } else if (cosShape2 == 1 && xShape0 == cosShape0 && (cosShape1 == 1 || cosShape1 == xShape1)) { // SBND (S11D,
                                                                                                     // SB1D), BSND
                                                                                                     // (BS1D)
        layout_ = InplacePartialRopeLayout::SBND;
    } else if (cosShape1 == 1 && xShape2 == cosShape2 && (cosShape0 == 1 || cosShape0 == xShape0)) { // BNSD (11SD,
                                                                                                     // B1SD)
        layout_ = InplacePartialRopeLayout::BNSD;
    } else if (cosShape0 == 1 && xShape1 == cosShape1 && xShape2 == cosShape2) { // 1SND
        layout_ = InplacePartialRopeLayout::BNSD;
        is1snd_ = true;
    } else {
        std::string shapeMsg = ToString(xShape) + " and " + ToString(cosShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "x and cos", shapeMsg.c_str(),
                                               "The shapes of input x and cos do not satisfy the broadcast rule");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRopeRegBaseTilingClass::CheckShape()
{
    auto &xShape = context_->GetInputShape(X_INDEX)->GetStorageShape();
    auto &cosShape = context_->GetInputShape(COS_INDEX)->GetStorageShape();
    auto &sinShape = context_->GetInputShape(SIN_INDEX)->GetStorageShape();
    auto &yShape = context_->GetOutputShape(Y_INDEX)->GetStorageShape();
    if (xShape.GetDimNum() != DIM_NUM || cosShape.GetDimNum() != DIM_NUM || sinShape.GetDimNum() != DIM_NUM) {
        std::string dimNumMsg = std::to_string(xShape.GetDimNum()) + ", " + std::to_string(cosShape.GetDimNum()) +
                                " and " + std::to_string(sinShape.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context_->GetNodeName(), "x, cos and sin", dimNumMsg.c_str(),
                                                  "The shape dims of input x, cos and sin should all be 4D");
        return ge::GRAPH_FAILED;
    }
    if (yShape.GetDimNum() != DIM_NUM) {
        std::string dimNumMsg = std::to_string(yShape.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context_->GetNodeName(), "output", dimNumMsg.c_str(),
                                                  "The shape dims of output should be 4D");
        return ge::GRAPH_FAILED;
    }
    if (cosShape != sinShape) {
        std::string shapeMsg = ToString(cosShape) + " and " + ToString(sinShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "cos and sin", shapeMsg.c_str(),
                                               "The shapes of input cos and input sin should be the same");
        return ge::GRAPH_FAILED;
    }
    if (xShape != yShape) {
        std::string shapeMsg = ToString(xShape) + " and " + ToString(yShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "x and output", shapeMsg.c_str(),
                                               "The shapes of input x and output should be the same");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRopeRegBaseTilingClass::CheckDtypeAndAttr()
{
    dtype_ = context_->GetInputDesc(X_INDEX)->GetDataType();
    if (std::find(SUPPORT_DTYPE.begin(), SUPPORT_DTYPE.end(), dtype_) == SUPPORT_DTYPE.end()) {
        std::string inputDtypeStr = ToString(dtype_);
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x", inputDtypeStr.c_str(), "FLOAT, BF16, FLOAT16");
        return ge::GRAPH_FAILED;
    }

    auto cosType = context_->GetInputDesc(COS_INDEX)->GetDataType();
    auto sinType = context_->GetInputDesc(SIN_INDEX)->GetDataType();
    // Check cos/sin dtype: same type, and must be F32, BF16, or F16
    if (cosType != sinType) {
        std::string dtypeMsg = ToString(cosType) + " and " + ToString(sinType);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "cos and sin", dtypeMsg.c_str(),
                                               "The dtypes of input cos and input sin should be the same");
        return ge::GRAPH_FAILED;
    }
    if (std::find(SUPPORT_DTYPE.begin(), SUPPORT_DTYPE.end(), cosType) == SUPPORT_DTYPE.end()) {
        std::string cosDtypeStr = ToString(cosType);
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "cos", cosDtypeStr.c_str(), "FLOAT, BF16, FLOAT16");
        return ge::GRAPH_FAILED;
    }

    // Mixed precision: x is BF16/FP16, cos/sin are FP32
    bool isMixedPrecision = (dtype_ == ge::DT_BF16 || dtype_ == ge::DT_FLOAT16) && cosType == ge::DT_FLOAT;
    bool isSamePrecision = (dtype_ == cosType);

    if (!isSamePrecision && !isMixedPrecision) {
        std::string dtypeMsg = ToString(dtype_) + " and " + ToString(cosType);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context_->GetNodeName(), "x and cos", dtypeMsg.c_str(),
            "Unsupported dtype combination: supported combinations are same type, or x=BF16/FP16 with cos/sin=FP32");
        return ge::GRAPH_FAILED;
    }

    auto outputType = context_->GetOutputDesc(Y_INDEX)->GetDataType();
    if (outputType != dtype_) {
        std::string outputDtypeStr = ToString(outputType);
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "output", outputDtypeStr.c_str(), ToString(dtype_).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRopeRegBaseTilingClass::CheckParam()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context_, "platform info is nullptr."), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    if (!IsRegbaseSocVersion()) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(CheckNullptr() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check nullptr fail."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckDtypeAndAttr() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check dtype and attr fail."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShape() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check shape fail."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRopeRegBaseTilingClass::CheckRotaryModeShapeRelation(const int64_t d)
{
    if (d > D_LIMIT) {
        std::string reasonMsg = "The D axis of input x can not be greater than " + std::to_string(D_LIMIT) +
                                ", where D refers to the last dim";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", ("D=" + std::to_string(d)).c_str(),
                                              reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    if (rotaryMode_ == InplacePartialRotaryPosEmbeddingMode::HALF ||
        rotaryMode_ == InplacePartialRotaryPosEmbeddingMode::INTERLEAVE ||
        rotaryMode_ == InplacePartialRotaryPosEmbeddingMode::DEEPSEEK_INTERLEAVE) {
        if (d % HALF_INTERLEAVE_MODE_COEF != 0) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                context_->GetNodeName(), "x", ("D=" + std::to_string(d)).c_str(),
                "D must be multiples of 2 in half, interleave and interleave-half mode, "
                "where D refers to the last dim");
            return ge::GRAPH_FAILED;
        }
    } else if (rotaryMode_ == InplacePartialRotaryPosEmbeddingMode::QUARTER) {
        if (d % QUARTER_MODE_COEF != 0) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                context_->GetNodeName(), "x", ("D=" + std::to_string(d)).c_str(),
                "D must be multiples of 4 in quarter mode, where D refers to the last dim");
            return ge::GRAPH_FAILED;
        }
    }
    if (rotaryMode_ == InplacePartialRotaryPosEmbeddingMode::HALF ||
        rotaryMode_ == InplacePartialRotaryPosEmbeddingMode::DEEPSEEK_INTERLEAVE) {
        dSplitCoef_ = HALF_INTERLEAVE_MODE_COEF;
    } else if (rotaryMode_ == InplacePartialRotaryPosEmbeddingMode::QUARTER) {
        dSplitCoef_ = QUARTER_MODE_COEF;
    } else {
        dSplitCoef_ = 1;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRopeRegBaseTilingClass::JudgeSliceInfo()
{
    OP_LOGI(context_, "TEST LOG, sliceStart_ =  %ld. sliceEnd_ =  %ld", sliceStart_, sliceEnd_);
    isNoOp_ = false;
    if (sliceStart_ < 0 || sliceEnd_ < 0 || sliceLength_ < 0 || sliceEnd_ > d_) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), "x",
            ("slice=[" + std::to_string(sliceStart_) + ", " + std::to_string(sliceEnd_) + "), D=" + std::to_string(d_))
                .c_str(),
            "Slice range is invalid: sliceStart/sliceEnd must be >= 0, "
            "sliceLength must be >= 0, and sliceEnd must be <= D");
        return ge::GRAPH_FAILED;
    }
    if (sliceLength_ == 0 || cosd_ == 0 || sind_ == 0) {
        isNoOp_ = true;
        return ge::GRAPH_SUCCESS;
    }
    if (sliceLength_ > 0 && (cosd_ != sind_ || cosd_ != sliceLength_)) {
        std::string shapesMsg = std::to_string(sliceLength_) + " (slice), " + std::to_string(cosd_) + " (cos_D) and " +
                                std::to_string(sind_) + " (sin_D)";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "slice, cos and sin", shapesMsg.c_str(),
            "When sliceLength > 0, the D axes of cos and sin must equal sliceLength");
        return ge::GRAPH_FAILED;
    }
    if (sliceLength_ > 0) {
        if (rotaryMode_ == InplacePartialRotaryPosEmbeddingMode::HALF ||
            rotaryMode_ == InplacePartialRotaryPosEmbeddingMode::INTERLEAVE ||
            rotaryMode_ == InplacePartialRotaryPosEmbeddingMode::DEEPSEEK_INTERLEAVE) {
            if (sliceLength_ % HALF_INTERLEAVE_MODE_COEF != 0) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->GetNodeName(), "x", ("sliceLength=" + std::to_string(sliceLength_)).c_str(),
                    "sliceLength must be multiples of 2 in half, interleave and interleave-half mode, "
                    "where sliceLength refers to partialSlice[1] - partialSlice[0]");
                return ge::GRAPH_FAILED;
            }
        } else if (rotaryMode_ == InplacePartialRotaryPosEmbeddingMode::QUARTER) {
            if (sliceLength_ % QUARTER_MODE_COEF != 0) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x",
                                                      ("sliceLength=" + std::to_string(sliceLength_)).c_str(),
                                                      "sliceLength must be multiples of 4 in quarter mode, "
                                                      "where sliceLength refers to partialSlice[1] - partialSlice[0]");
                return ge::GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRopeRegBaseTilingClass::GetShapeAttrsInfo()
{
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const int32_t *mode = attrs->GetAttrPointer<int32_t>(0);
    int32_t modeValue = (mode == nullptr) ? 0 : static_cast<int32_t>(*mode);
    OP_CHECK_IF(modeValue != static_cast<int32_t>(InplacePartialRotaryPosEmbeddingMode::INTERLEAVE),
                OP_LOGE(context_->GetNodeName(),
                        "InplacePartialRotaryMul only supports interleave mode (mode=1), actual %d.", modeValue),
                return ge::GRAPH_FAILED);
    rotaryMode_ = static_cast<InplacePartialRotaryPosEmbeddingMode>(modeValue);

    OP_CHECK_IF(CheckParam() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check param fail."), return ge::GRAPH_FAILED);

    dtype_ = context_->GetInputDesc(X_INDEX)->GetDataType();
    auto &xShape = context_->GetInputShape(X_INDEX)->GetStorageShape();
    auto &cosShape = context_->GetInputShape(COS_INDEX)->GetStorageShape();
    auto &sinShape = context_->GetInputShape(SIN_INDEX)->GetStorageShape();

    d_ = xShape.GetDim(DIM_3);
    cosd_ = cosShape.GetDim(DIM_3);
    sind_ = sinShape.GetDim(DIM_3);

    // 获取slice
    const gert::ContinuousVector *sliceRangeListPtr = attrs->GetAttrPointer<gert::ContinuousVector>(1);
    if (sliceRangeListPtr == nullptr || sliceRangeListPtr->GetSize() == 0) {
        sliceStart_ = 0;
        sliceEnd_ = d_;
    } else {
        const int64_t *expertRangeList = reinterpret_cast<const int64_t *>(sliceRangeListPtr->GetData());
        sliceStart_ = expertRangeList[0];
        sliceEnd_ = expertRangeList[1];
    }
    sliceLength_ = sliceEnd_ - sliceStart_;
    OP_CHECK_IF(JudgeSliceInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "JudgeSliceInfo fail."),
                return ge::GRAPH_FAILED);
    if (!IsNoOp()) {
        OP_CHECK_IF(CheckRotaryModeShapeRelation(d_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_, "CheckRotaryModeShapeRelation fail."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(CheckShapeAllPositive() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "CheckShapeAllPositive fail."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(JudgeLayoutByShape(xShape, cosShape) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_, "JudgeLayoutByShape fail."), return ge::GRAPH_FAILED);

        if (layout_ == InplacePartialRopeLayout::BSND) {
            b_ = xShape.GetDim(DIM_0);
            cosb_ = cosShape.GetDim(DIM_0);
            s_ = xShape.GetDim(DIM_1);
            n_ = xShape.GetDim(DIM_2);
        } else if (layout_ == InplacePartialRopeLayout::BNSD || layout_ == InplacePartialRopeLayout::NO_BROADCAST ||
                   layout_ == InplacePartialRopeLayout::BROADCAST_BSN) {
            b_ = xShape.GetDim(DIM_0);
            cosb_ = cosShape.GetDim(DIM_0);
            n_ = xShape.GetDim(DIM_1);
            s_ = xShape.GetDim(DIM_2);
            // 1XXX情况下，reshape成11XX
            if (is1snd_ == true) {
                s_ = s_ * n_;
                n_ = 1;
            }
        } else if (layout_ == InplacePartialRopeLayout::SBND) {
            s_ = xShape.GetDim(DIM_0);
            b_ = xShape.GetDim(DIM_1);
            cosb_ = cosShape.GetDim(DIM_1);
            n_ = xShape.GetDim(DIM_2);
        }
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
