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
 * \file inplace_partial_rotary_mul_grad_tiling_base_arch35.cpp
 * \brief
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <graph/utils/type_utils.h>
#include "tiling/tiling_api.h"
#include "inplace_partial_rotary_mul_grad_regbase_tiling.h"

namespace {
constexpr int64_t DY_INPUT_INDEX = 0;
constexpr int64_t COS_INDEX = 1;
constexpr int64_t SIN_INDEX = 2;
constexpr int64_t DY_OUTPUT_INDEX = 0;
constexpr int64_t DIM_NUM = 4;
constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;
constexpr int64_t HALF_INTERLEAVE_MODE_COEF = 2;
constexpr int64_t QUARTER_MODE_COEF = 4;
constexpr int64_t D_LIMIT = 1024;

const std::vector<ge::DataType> SUPPORT_DTYPE = {ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16};
static const std::vector<std::string> inputNames = {"dy", "cos", "sin"};
static const std::vector<std::string> outputNames = {"dy"};
} // namespace

namespace optiling {
using namespace Ops::Base;
ge::graphStatus InplacePartialRotaryMulGradRegbaseTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context_, "platformInfo cannot be nullptr."), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    aicoreParams_.numBlocks = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    aicoreParams_.ubSize = ubSizePlatForm;
    blockSize_ = 32; // UB block size is always 32 bytes on Ascend
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulGradRegbaseTiling::CheckNullptr()
{
    int64_t checkInputIndexRange = SIN_INDEX;
    int64_t checkOutputIndexRange = DY_OUTPUT_INDEX;

    for (int64_t i = 0; i <= checkInputIndexRange; i++) {
        auto desc = context_->GetInputDesc(i);
        OP_CHECK_IF(desc == nullptr, OP_LOGE(context_, "the input %s desc is nullptr.", inputNames[i].c_str()),
                    return ge::GRAPH_FAILED);
        auto shape = context_->GetInputShape(i);
        OP_CHECK_IF(shape == nullptr, OP_LOGE(context_, "the input %s shape is nullptr.", inputNames[i].c_str()),
                    return ge::GRAPH_FAILED);
    }

    for (int64_t i = 0; i <= checkOutputIndexRange; i++) {
        auto desc = context_->GetOutputDesc(i);
        OP_CHECK_IF(desc == nullptr, OP_LOGE(context_, "the output %s desc is nullptr.", outputNames[i].c_str()),
                    return ge::GRAPH_FAILED);
        auto shape = context_->GetOutputShape(i);
        OP_CHECK_IF(shape == nullptr, OP_LOGE(context_, "the output %s shape is nullptr.", outputNames[i].c_str()),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulGradRegbaseTiling::CheckInPutShapeAllPositive(const int64_t idx) const
{
    auto shape = context_->GetInputShape(idx)->GetStorageShape();
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        if (shape.GetDim(i) <= 0) {
            std::string shapeMsg = ToString(shape);
            std::string reasonMsg = "The shape of the input " + inputNames[idx] +
                                    " can not be an empty tensor or an invalid tensor with a negative dimension";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), inputNames[idx].c_str(), shapeMsg.c_str(),
                                                  reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulGradRegbaseTiling::CheckOutPutShapeAllPositive(const int64_t idx) const
{
    auto shape = context_->GetOutputShape(idx)->GetStorageShape();
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        if (shape.GetDim(i) <= 0) {
            std::string shapeMsg = ToString(shape);
            std::string reasonMsg = "The shape of output " + outputNames[idx] +
                                    " can not be an empty tensor or an invalid tensor with a negative dimension";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), outputNames[idx].c_str(), shapeMsg.c_str(),
                                                  reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

bool InplacePartialRotaryMulGradRegbaseTiling::IsInplacePartialRotaryMulGradMode(const int32_t mode) const
{
    switch (mode) {
        case static_cast<int32_t>(InplacePartialRotaryMulGradMode::HALF):
            return false;
        case static_cast<int32_t>(InplacePartialRotaryMulGradMode::INTERLEAVE):
            return true;
        case static_cast<int32_t>(InplacePartialRotaryMulGradMode::QUARTER):
            return false;
        case static_cast<int32_t>(InplacePartialRotaryMulGradMode::INTERLEAVE_HALF):
            return false;
        default:
            return false;
    }
}

ge::graphStatus InplacePartialRotaryMulGradRegbaseTiling::CheckShapeAllPositive() const
{
    if (isNoOp_) {
        // No-op: empty slice or empty dy/cos/sin, nothing to compute -> positive-shape checks skipped.
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(CheckInPutShapeAllPositive(DY_INPUT_INDEX) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "the dy input has non positive shape."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInPutShapeAllPositive(COS_INDEX) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "the cos input has non positive shape."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInPutShapeAllPositive(SIN_INDEX) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "the sin input has non positive shape."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckOutPutShapeAllPositive(DY_OUTPUT_INDEX) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "the dy output has non positive shape."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulGradRegbaseTiling::JudgeLayoutByShape(const gert::Shape &xShape,
                                                                             const gert::Shape &cosShape)
{
    uint64_t xShape0 = xShape.GetDim(DIM_0);
    uint64_t xShape1 = xShape.GetDim(DIM_1);
    uint64_t xShape2 = xShape.GetDim(DIM_2);
    uint64_t cosShape0 = cosShape.GetDim(DIM_0);
    uint64_t cosShape1 = cosShape.GetDim(DIM_1);
    uint64_t cosShape2 = cosShape.GetDim(DIM_2);
    if (xShape0 == cosShape0 && xShape1 == cosShape1 && xShape2 == cosShape2) { // BSND
        layout_ = InplacePartialRotaryMulGradLayout::NO_BROADCAST;
    } else if (cosShape0 == 1 && cosShape1 == 1 && cosShape2 == 1) { // (111D)
        layout_ = InplacePartialRotaryMulGradLayout::BROADCAST_BSN;
    } else if (cosShape2 == 1 && cosShape0 == 1 && xShape1 == cosShape1) { // BSND (1S1D)
        layout_ = InplacePartialRotaryMulGradLayout::BSND;
    } else if (cosShape2 == 1 && xShape0 == cosShape0 && (cosShape1 == 1 || cosShape1 == xShape1)) { // SBND (S11D,
                                                                                                     // SB1D), BSND
                                                                                                     // (BS1D)
        layout_ = InplacePartialRotaryMulGradLayout::SBND;
    } else if (cosShape1 == 1 && xShape2 == cosShape2 && (cosShape0 == 1 || cosShape0 == xShape0)) { // BNSD (11SD,
                                                                                                     // B1SD)
        layout_ = InplacePartialRotaryMulGradLayout::BNSD;
    } else if (cosShape0 == 1 && xShape1 == cosShape1 && xShape2 == cosShape2) { // 1SND
        layout_ = InplacePartialRotaryMulGradLayout::BNSD;
        is1snd_ = true;
    } else {
        std::string shapeMsg = ToString(cosShape) + " and " + ToString(xShape);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), "cos and dy", shapeMsg.c_str(),
            "Each axis of input cos except the last must be 1 or equal to the same axis of input dy");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulGradRegbaseTiling::CheckShapeDim() const
{
    auto &dyInputShape = context_->GetInputShape(DY_INPUT_INDEX)->GetStorageShape();
    auto &cosShape = context_->GetInputShape(COS_INDEX)->GetStorageShape();
    auto &sinShape = context_->GetInputShape(SIN_INDEX)->GetStorageShape();
    auto &dyOutputShape = context_->GetOutputShape(DY_OUTPUT_INDEX)->GetStorageShape();

    if (dyInputShape.GetDimNum() != DIM_NUM) {
        std::string dimNumStr = std::to_string(dyInputShape.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "dy", dimNumStr.c_str(), "4D");
        return ge::GRAPH_FAILED;
    }
    if (cosShape.GetDimNum() != DIM_NUM) {
        std::string dimNumStr = std::to_string(cosShape.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "cos", dimNumStr.c_str(), "4D");
        return ge::GRAPH_FAILED;
    }
    if (sinShape.GetDimNum() != DIM_NUM) {
        std::string dimNumStr = std::to_string(sinShape.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "sin", dimNumStr.c_str(), "4D");
        return ge::GRAPH_FAILED;
    }
    if (dyOutputShape.GetDimNum() != DIM_NUM) {
        std::string dimNumStr = std::to_string(dyOutputShape.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "dy", dimNumStr.c_str(), "4D");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulGradRegbaseTiling::CheckShapeLimit()
{
    auto &dyInputShape = context_->GetInputShape(DY_INPUT_INDEX)->GetStorageShape();
    auto &cosShape = context_->GetInputShape(COS_INDEX)->GetStorageShape();
    auto &sinShape = context_->GetInputShape(SIN_INDEX)->GetStorageShape();
    auto &dyOutputShape = context_->GetOutputShape(DY_OUTPUT_INDEX)->GetStorageShape();

    if (cosShape != sinShape) {
        std::string shapeMsg = ToString(cosShape) + " and " + ToString(sinShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "cos and sin", shapeMsg.c_str(),
                                               "The shapes of input cos and sin should be the same");
        return ge::GRAPH_FAILED;
    }
    if (dyInputShape != dyOutputShape) {
        std::string shapeMsg = ToString(dyInputShape) + " and " + ToString(dyOutputShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "dy", shapeMsg.c_str(),
                                               "The shapes of input dy and output dy should be the same");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulGradRegbaseTiling::CheckShape()
{
    // Shape checks still run for the no-op case: they enforce 4D and pass for the
    // valid no-op inputs (cos==sin, dy==out, cosD<=dyD all hold for empty shapes too).
    auto &cosShape = context_->GetInputShape(COS_INDEX)->GetStorageShape();
    auto &dyInputShape = context_->GetInputShape(DY_INPUT_INDEX)->GetStorageShape();
    OP_CHECK_IF(CheckShapeDim() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check shape dim fail."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShapeLimit() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "CheckShapeLimit fail."),
                return ge::GRAPH_FAILED);
    int64_t dyLastDim = dyInputShape.GetDim(DIM_3);
    int64_t cosLastDim = cosShape.GetDim(DIM_3);
    if (cosLastDim > dyLastDim) {
        std::string shapeMsg = ToString(dyInputShape) + " and " + ToString(cosShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "dy and cos", shapeMsg.c_str(),
            "The D axis of input cos should not be greater than the D axis of input dy");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulGradRegbaseTiling::CheckDtypeDyGroup()
{
    dtype_ = context_->GetInputDesc(DY_INPUT_INDEX)->GetDataType();
    if (std::find(SUPPORT_DTYPE.begin(), SUPPORT_DTYPE.end(), dtype_) == SUPPORT_DTYPE.end()) {
        std::string dtypeStr = ToString(dtype_);
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "dy", dtypeStr.c_str(), "FLOAT, BF16 or FLOAT16");
        return ge::GRAPH_FAILED;
    }

    auto dxDtype = context_->GetOutputDesc(DY_OUTPUT_INDEX)->GetDataType();
    if (dxDtype != dtype_) {
        std::string paramMsg = outputNames[DY_OUTPUT_INDEX] + " and dy";
        std::string dtypeMsg = ToString(dxDtype) + " and " + ToString(dtype_);
        std::string reasonMsg =
            "The dtypes of output " + outputNames[DY_OUTPUT_INDEX] + " and input dy should be the same";
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), paramMsg.c_str(), dtypeMsg.c_str(),
                                               reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulGradRegbaseTiling::CheckDtypeCosGroup()
{
    auto cosDtype = context_->GetInputDesc(COS_INDEX)->GetDataType();
    if (std::find(SUPPORT_DTYPE.begin(), SUPPORT_DTYPE.end(), cosDtype) == SUPPORT_DTYPE.end()) {
        std::string dtypeStr = ToString(cosDtype);
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "cos", dtypeStr.c_str(), "FLOAT, BF16 or FLOAT16");
        return ge::GRAPH_FAILED;
    }
    cosDtype_ = cosDtype;

    auto sinDtype = context_->GetInputDesc(SIN_INDEX)->GetDataType();
    if (sinDtype != cosDtype) {
        std::string paramMsg = "sin and cos";
        std::string dtypeMsg = ToString(sinDtype) + " and " + ToString(cosDtype);
        std::string reasonMsg = "The dtypes of input sin and input cos should be the same";
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), paramMsg.c_str(), dtypeMsg.c_str(),
                                               reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulGradRegbaseTiling::CheckDtype()
{
    OP_CHECK_IF(CheckDtypeDyGroup() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check dy group dtype fail."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckDtypeCosGroup() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check cos group dtype fail."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulGradRegbaseTiling::CheckRotaryModeShapeRelation(const int64_t sliceLen)
{
    auto &dyShape = context_->GetInputShape(DY_INPUT_INDEX)->GetStorageShape();
    int64_t dyD = dyShape.GetDim(DIM_3);
    if (sliceStart_ < 0 || sliceEnd_ > dyD) {
        std::string sliceStr = "[" + std::to_string(sliceStart_) + ", " + std::to_string(sliceEnd_) + ")";
        std::string reasonMsg = "partial_slice [start, end) must satisfy 0 <= start <= end <= D, "
                                "where D is the last dim of input dy. Got D=" +
                                std::to_string(dyD);
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), sliceStr.c_str(), std::to_string(dyD).c_str(),
                                  reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    if (dyD > D_LIMIT) {
        std::string shapeMsg = ToString(dyShape);
        std::string reasonMsg = "The D axis of input dy can not be greater than " + std::to_string(D_LIMIT) +
                                ", where D refers to the last dim";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "dy", shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    if (rotaryMode_ == InplacePartialRotaryMulGradMode::HALF ||
        rotaryMode_ == InplacePartialRotaryMulGradMode::INTERLEAVE ||
        rotaryMode_ == InplacePartialRotaryMulGradMode::INTERLEAVE_HALF) {
        if (sliceLen % HALF_INTERLEAVE_MODE_COEF != 0) {
            std::string reasonMsg = "The slice length (partial_slice's end - start) should be divisible by " +
                                    std::to_string(HALF_INTERLEAVE_MODE_COEF) +
                                    " when the attr mode is half, interleave or interleave-half";
            OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "partial_slice", std::to_string(sliceLen).c_str(),
                                      reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    } else if (rotaryMode_ == InplacePartialRotaryMulGradMode::QUARTER) {
        if (sliceLen % QUARTER_MODE_COEF != 0) {
            std::string reasonMsg = "The slice length (partial_slice's end - start) should be divisible by " +
                                    std::to_string(QUARTER_MODE_COEF) + " when the attr mode is quarter";
            OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "partial_slice", std::to_string(sliceLen).c_str(),
                                      reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }

    if (rotaryMode_ == InplacePartialRotaryMulGradMode::HALF ||
        rotaryMode_ == InplacePartialRotaryMulGradMode::INTERLEAVE_HALF) {
        dSplitCoef_ = HALF_INTERLEAVE_MODE_COEF;
    } else if (rotaryMode_ == InplacePartialRotaryMulGradMode::QUARTER) {
        dSplitCoef_ = QUARTER_MODE_COEF;
    } else {
        dSplitCoef_ = 1;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulGradRegbaseTiling::CheckAttr()
{
    isNoOp_ = false; // reset across tiling instances
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    const int32_t *mode = attrs->GetAttrPointer<int32_t>(0);
    int32_t modeValue = (mode == nullptr) ? 0 : static_cast<int32_t>(*mode);
    if (IsInplacePartialRotaryMulGradMode(modeValue) != true) {
        std::string modeStr = std::to_string(modeValue);
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "mode", modeStr.c_str(), "0, 1, 2 or 3");
        return ge::GRAPH_FAILED;
    }
    rotaryMode_ = static_cast<InplacePartialRotaryMulGradMode>(modeValue);

    auto partialSlicePtr = attrs->GetListInt(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, partialSlicePtr);
    sliceStart_ = partialSlicePtr->GetData()[0];
    sliceEnd_ = partialSlicePtr->GetData()[1];
    sliceLength_ = sliceEnd_ - sliceStart_;

    // No-op: empty slice (start == end) or any of dy/cos/sin is an empty tensor
    // (some dim is 0). Nothing to compute, the kernel goes through TILING_KEY_EMPTY
    // and returns, dx == dy (in-place). Use total element count so an empty
    // B/S/N dim of cos/sin is caught too, not just an empty D dim.
    auto &dyShape = context_->GetInputShape(DY_INPUT_INDEX)->GetStorageShape();
    auto &cosShape = context_->GetInputShape(COS_INDEX)->GetStorageShape();
    auto &sinShape = context_->GetInputShape(SIN_INDEX)->GetStorageShape();
    int64_t cosD = cosShape.GetDim(DIM_3);
    int64_t dySize = dyShape.GetShapeSize();
    int64_t cosSize = cosShape.GetShapeSize();
    int64_t sinSize = sinShape.GetShapeSize();
    if (sliceLength_ == 0 || dySize == 0 || cosSize == 0 || sinSize == 0) {
        isNoOp_ = true;
        return ge::GRAPH_SUCCESS;
    }
    if (sliceLength_ != cosD) {
        std::string sliceStr = "[" + std::to_string(sliceStart_) + ", " + std::to_string(sliceEnd_) + ")";
        std::string reasonMsg =
            "The D axis of input cos should be equal to the slice length (partial_slice's end - start)";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), sliceStr.c_str(), std::to_string(cosD).c_str(),
                                              reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(CheckRotaryModeShapeRelation(sliceLength_) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "check rotary mode shape relation fail."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulGradRegbaseTiling::CheckParam()
{
    OP_CHECK_IF(CheckNullptr() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check nullptr fail."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckDtype() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check dtype fail."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckAttr() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check attr fail."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShape() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check shape fail."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShapeAllPositive() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check shape positive fail."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulGradRegbaseTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(CheckParam() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check param fail."), return ge::GRAPH_FAILED);
    if (isNoOp_) {
        // No-op: kernel returns at TILING_KEY_EMPTY and ignores the tiling data. Set a
        // valid layout so the template dispatch (TilingRegistry) selects a capable class
        // (A&B handles BROADCAST_BSN), then skip JudgeLayoutByShape — an empty B/S/N dim
        // of cos/sin has no valid broadcast pattern and would fail there.
        layout_ = InplacePartialRotaryMulGradLayout::BROADCAST_BSN;
        return ge::GRAPH_SUCCESS;
    }

    auto &dyInputShape = context_->GetInputShape(DY_INPUT_INDEX)->GetStorageShape();
    auto &cosInputShape = context_->GetInputShape(COS_INDEX)->GetStorageShape();
    dyShape_ = dyInputShape;
    cosShape_ = cosInputShape;
    OP_CHECK_IF(JudgeLayoutByShape(dyInputShape, cosInputShape) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "JudgeLayoutByShape fail."), return ge::GRAPH_FAILED);
    d_ = dyInputShape.GetDim(DIM_3);
    if (layout_ == InplacePartialRotaryMulGradLayout::BSND) {
        b_ = dyShape_.GetDim(DIM_0);
        cosb_ = cosShape_.GetDim(DIM_0);
        s_ = dyShape_.GetDim(DIM_1);
        n_ = dyShape_.GetDim(DIM_2);
    } else if (layout_ == InplacePartialRotaryMulGradLayout::BNSD ||
               layout_ == InplacePartialRotaryMulGradLayout::NO_BROADCAST ||
               layout_ == InplacePartialRotaryMulGradLayout::BROADCAST_BSN) {
        b_ = dyShape_.GetDim(DIM_0);
        cosb_ = cosShape_.GetDim(DIM_0);
        n_ = dyShape_.GetDim(DIM_1);
        s_ = dyShape_.GetDim(DIM_2);
        // 1XXX情况下，reshape成11XX
        if (is1snd_ == true) {
            s_ = s_ * n_;
            n_ = 1;
        }
    } else if (layout_ == InplacePartialRotaryMulGradLayout::SBND) {
        s_ = dyShape_.GetDim(DIM_0);
        b_ = dyShape_.GetDim(DIM_1);
        cosb_ = cosShape_.GetDim(DIM_1);
        n_ = dyShape_.GetDim(DIM_2);
    }

    return ge::GRAPH_SUCCESS;
}

uint64_t InplacePartialRotaryMulGradRegbaseTiling::GetTilingKey() const
{
    return tilingKey_;
}

} // namespace optiling
