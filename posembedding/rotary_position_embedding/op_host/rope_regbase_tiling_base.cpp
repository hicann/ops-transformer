/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
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

#include "op_host/tiling_templates_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "platform/platform_info.h"
#include "rotary_position_embedding_tiling.h"
#include "log/log.h"

namespace {
constexpr int64_t X_INDEX = 0;
constexpr int64_t COS_INDEX = 1;
constexpr int64_t SIN_INDEX = 2;
constexpr int64_t Y_INDEX = 0;
constexpr int64_t DIM_NUM = 4;
constexpr int64_t DIM_NUM_TND = 3;
constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;
constexpr int64_t HALF_INTERLEAVE_MODE_COEF = 2;
constexpr int64_t QUARTER_MODE_COEF = 4;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t D_LIMIT = 1024;
const std::vector<ge::DataType> SUPPORT_DTYPE = {ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16};
static const std::vector<std::string> inputNames = {"x", "cos", "sin", "rotate"};
} // namespace

namespace optiling {
using namespace Ops::Base;
ge::graphStatus RopeRegBaseTilingClass::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aicoreParams_.numBlocks = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        aicoreParams_.ubSize = ubSizePlatForm;
        socVersion_ = ascendcPlatform.GetSocVersion();
    } else {
        auto compileInfoPtr = reinterpret_cast<const RotaryPositionEmbeddingCompileInfo *>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
        aicoreParams_.numBlocks = compileInfoPtr->numBlocks;
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
        socVersion_ = compileInfoPtr->socVersion;
    }
    blockSize_ = BLOCK_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeRegBaseTilingClass::CheckNullptr()
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

ge::graphStatus RopeRegBaseTilingClass::CheckShapeAllPositive(const int64_t idx) const
{
    auto shape = context_->GetInputShape(idx)->GetStorageShape();
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        if (shape.GetDim(i) <= 0) {
            std::string reasonMsg = "The shape of input " + inputNames[idx] +
                                    " can not be an empty tensor or an invalid tensor with a negative dimension";
            std::string shapeStr = ToString(shape);
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), inputNames[idx].c_str(), shapeStr.c_str(),
                                                  reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}
bool RopeRegBaseTilingClass::IsRotaryPosEmbeddingMode(const int32_t mode) const
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

ge::graphStatus RopeRegBaseTilingClass::CheckShapeAllPositive() const
{
    OP_CHECK_IF(CheckShapeAllPositive(X_INDEX) != ge::GRAPH_SUCCESS, OP_LOGE(context_, "x has non positive shape."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShapeAllPositive(COS_INDEX) != ge::GRAPH_SUCCESS, OP_LOGE(context_, "cos has non positive shape."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShapeAllPositive(SIN_INDEX) != ge::GRAPH_SUCCESS, OP_LOGE(context_, "sin has non positive shape."),
                return ge::GRAPH_FAILED);
    auto yShape = context_->GetOutputShape(Y_INDEX)->GetStorageShape();
    for (size_t i = 0; i < yShape.GetDimNum(); i++) {
        if (yShape.GetDim(i) <= 0) {
            std::string reasonMsg = "The shape of output y can not be an empty tensor "
                                    "or an invalid tensor with a negative dimension";
            std::string yShapeStr = ToString(yShape);
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "y", yShapeStr.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

bool RopeRegBaseTilingClass::Is3dBsdBroadcastLayout(const gert::Shape &xShape, const gert::Shape &cosShape) const
{
    // 识别 x（TND）cos（1ND）场景
    return xShape.GetDimNum() == DIM_NUM_TND && cosShape.GetDimNum() == DIM_NUM_TND && xShape.GetDim(DIM_1) != 1 &&
           cosShape.GetDim(DIM_0) == 1 && cosShape.GetDim(DIM_1) == xShape.GetDim(DIM_1) &&
           cosShape.GetDim(DIM_2) == xShape.GetDim(DIM_2);
}

ge::graphStatus RopeRegBaseTilingClass::JudgeLayoutByShape(const gert::Shape &xShape, const gert::Shape &cosShape)
{
    isTndLayout_ = (xShape.GetDimNum() == DIM_NUM_TND);
    is3dBsdLayout_ = false;
    // TND的广播场景需要打上标记后续做分核优化
    if (Is3dBsdBroadcastLayout(xShape, cosShape)) {
        layout_ = RopeLayout::BSND;
        is3dBsdLayout_ = true;
        return ge::GRAPH_SUCCESS;
    }
    uint64_t xShape0 = isTndLayout_ ? 1 : xShape.GetDim(DIM_0);
    uint64_t xShape1 = isTndLayout_ ? xShape.GetDim(DIM_0) : xShape.GetDim(DIM_1);
    uint64_t xShape2 = isTndLayout_ ? xShape.GetDim(DIM_1) : xShape.GetDim(DIM_2);
    uint64_t cosShape0 = isTndLayout_ ? 1 : cosShape.GetDim(DIM_0);
    uint64_t cosShape1 = isTndLayout_ ? cosShape.GetDim(DIM_0) : cosShape.GetDim(DIM_1);
    uint64_t cosShape2 = isTndLayout_ ? cosShape.GetDim(DIM_1) : cosShape.GetDim(DIM_2);
    if (xShape0 == cosShape0 && xShape1 == cosShape1 && xShape2 == cosShape2) { // BSND
        layout_ = RopeLayout::NO_BROADCAST;
    } else if (cosShape0 == 1 && cosShape1 == 1 && cosShape2 == 1) { // (111D)
        layout_ = RopeLayout::BROADCAST_BSN;
    } else if (cosShape2 == 1 && cosShape0 == 1 && xShape1 == cosShape1) { // BSND (1S1D)
        layout_ = RopeLayout::BSND;
    } else if (cosShape2 == 1 && xShape0 == cosShape0 && (cosShape1 == 1 || cosShape1 == xShape1)) { // SBND (S11D,
                                                                                                     // SB1D), BSND
                                                                                                     // (BS1D)
        layout_ = RopeLayout::SBND;
    } else if (cosShape1 == 1 && xShape2 == cosShape2 && (cosShape0 == 1 || cosShape0 == xShape0)) { // BNSD (11SD,
                                                                                                     // B1SD)
        layout_ = RopeLayout::BNSD;
    } else if (cosShape0 == 1 && xShape1 == cosShape1 && xShape2 == cosShape2) { // 1SND
        layout_ = RopeLayout::BNSD;
        is1snd_ = true;
    } else {
        std::string shapeMsg = ToString(cosShape) + " and " + ToString(xShape);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), "cos and x", shapeMsg.c_str(),
            "Each axis of input cos except the last must be 1 or equal to the same axis of input x");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeRegBaseTilingClass::CheckShape()
{
    auto &xShape = context_->GetInputShape(X_INDEX)->GetStorageShape();
    auto &cosShape = context_->GetInputShape(COS_INDEX)->GetStorageShape();
    auto &sinShape = context_->GetInputShape(SIN_INDEX)->GetStorageShape();
    auto &yShape = context_->GetOutputShape(Y_INDEX)->GetStorageShape();
    if (xShape.GetDimNum() != DIM_NUM && xShape.GetDimNum() != DIM_NUM_TND) {
        std::string dimNumStr = std::to_string(xShape.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "x", dimNumStr.c_str(), "3D or 4D");
        return ge::GRAPH_FAILED;
    }
    if (cosShape.GetDimNum() != DIM_NUM && cosShape.GetDimNum() != DIM_NUM_TND) {
        std::string dimNumStr = std::to_string(cosShape.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "cos", dimNumStr.c_str(), "3D or 4D");
        return ge::GRAPH_FAILED;
    }
    if (sinShape.GetDimNum() != DIM_NUM && sinShape.GetDimNum() != DIM_NUM_TND) {
        std::string dimNumStr = std::to_string(sinShape.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "sin", dimNumStr.c_str(), "3D or 4D");
        return ge::GRAPH_FAILED;
    }
    if (yShape.GetDimNum() != DIM_NUM && yShape.GetDimNum() != DIM_NUM_TND) {
        std::string dimNumStr = std::to_string(yShape.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "y", dimNumStr.c_str(), "3D or 4D");
        return ge::GRAPH_FAILED;
    }

    if (cosShape != sinShape) {
        std::string shapeMsg = ToString(cosShape) + " and " + ToString(sinShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "cos and sin", shapeMsg.c_str(),
                                               "The shapes of input cos and sin should be the same");
        return ge::GRAPH_FAILED;
    }
    if (xShape != yShape) {
        std::string shapeMsg = ToString(xShape) + " and " + ToString(yShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "x and y", shapeMsg.c_str(),
                                               "The shapes of input x and output y should be the same");
        return ge::GRAPH_FAILED;
    }
    isTndLayout_ = (xShape.GetDimNum() == DIM_NUM_TND);
    int64_t xLastDim = isTndLayout_ ? xShape.GetDim(DIM_2) : xShape.GetDim(DIM_3);
    int64_t cosLastDim = isTndLayout_ ? cosShape.GetDim(DIM_2) : cosShape.GetDim(DIM_3);
    if (cosLastDim != xLastDim) {
        std::string shapeMsg = ToString(xShape) + " and " + ToString(cosShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "x and cos", shapeMsg.c_str(),
            "The D axes of input x and cos should be the same, where D refers to the last dim");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(CheckRotaryModeShapeRelation(xLastDim) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "D is invalid for rotary mode."), return ge::GRAPH_FAILED);
    return CheckShapeAllPositive();
}

ge::graphStatus RopeRegBaseTilingClass::CheckDtypeAndAttr()
{
    dtype_ = context_->GetInputDesc(X_INDEX)->GetDataType();
    if (std::find(SUPPORT_DTYPE.begin(), SUPPORT_DTYPE.end(), dtype_) == SUPPORT_DTYPE.end()) {
        std::string dtypeStr = ToString(dtype_);
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x", dtypeStr.c_str(), "FLOAT, BF16, FLOAT16");
        return ge::GRAPH_FAILED;
    }
    for (int64_t i = X_INDEX; i <= SIN_INDEX; i++) {
        auto type = context_->GetInputDesc(i)->GetDataType();
        if (type != dtype_) {
            std::string dtypeMsg = ToString(type) + " and " + ToString(dtype_);
            std::string reasonMsg = "The dtypes of input " + inputNames[i] + " and input x should be the same";
            std::string paramMsg = inputNames[i] + " and x";
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), paramMsg.c_str(), dtypeMsg.c_str(),
                                                   reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    auto outputType = context_->GetOutputDesc(Y_INDEX)->GetDataType();
    if (outputType != dtype_) {
        std::string dtypeMsg = ToString(outputType) + " and " + ToString(dtype_);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "y and x", dtypeMsg.c_str(),
                                               "The dtypes of output y and input x should be the same");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeRegBaseTilingClass::CheckParam()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context_, "platform info is nullptr."), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    if (!Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(CheckNullptr() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check nullptr fail."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckDtypeAndAttr() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check dtype and attr fail."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShape() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check shape fail."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeRegBaseTilingClass::CheckRotaryModeShapeRelation(const int64_t d)
{
    auto xShape = context_->GetInputShape(X_INDEX)->GetStorageShape();
    if (d > D_LIMIT) {
        std::string reasonMsg = "The D axis of input x can not be greater than " + std::to_string(D_LIMIT) +
                                ", where D refers to the last dim";
        std::string xShapeStr = ToString(xShape);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", xShapeStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    if (rotaryMode_ == RotaryPosEmbeddingMode::HALF || rotaryMode_ == RotaryPosEmbeddingMode::INTERLEAVE ||
        rotaryMode_ == RotaryPosEmbeddingMode::DEEPSEEK_INTERLEAVE) {
        if (d % HALF_INTERLEAVE_MODE_COEF != 0) {
            std::string reasonMsg = "The D axis of input x should be divisible by " +
                                    std::to_string(HALF_INTERLEAVE_MODE_COEF) +
                                    " when the attribute mode is half, interleave or interleave-half, "
                                    "where D refers to the last dim";
            std::string xShapeStr = ToString(xShape);
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", xShapeStr.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    } else if (rotaryMode_ == RotaryPosEmbeddingMode::QUARTER) {
        if (d % QUARTER_MODE_COEF != 0) {
            std::string reasonMsg = "The D axis of input x should be divisible by " +
                                    std::to_string(QUARTER_MODE_COEF) +
                                    " when the attr mode is quarter, where D refers to the last dim";
            std::string xShapeStr = ToString(xShape);
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", xShapeStr.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
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

void RopeRegBaseTilingClass::Set3dBsdShapeAttrs(const gert::Shape &xShape)
{
    // TND的广播场景需要映射成为BS1D，用于分核优化
    b_ = xShape.GetDim(DIM_0);
    cosb_ = 1; // 实际就为1
    s_ = xShape.GetDim(DIM_1);
    n_ = 1;
}

ge::graphStatus RopeRegBaseTilingClass::GetShapeAttrsInfo()
{
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const int32_t *mode = attrs->GetAttrPointer<int32_t>(0);
    int32_t modeValue = (mode == nullptr) ? 0 : static_cast<int32_t>(*mode);
    if (IsRotaryPosEmbeddingMode(modeValue) != true) {
        std::string modeValueStr = std::to_string(modeValue);
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "mode", modeValueStr.c_str(), "0, 1, 2 or 3");
        return ge::GRAPH_FAILED;
    }
    rotaryMode_ = static_cast<RotaryPosEmbeddingMode>(modeValue);

    OP_CHECK_IF(CheckParam() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check param fail."), return ge::GRAPH_FAILED);

    dtype_ = context_->GetInputDesc(X_INDEX)->GetDataType();
    auto &xShape = context_->GetInputShape(X_INDEX)->GetStorageShape();
    auto &cosShape = context_->GetInputShape(COS_INDEX)->GetStorageShape();
    OP_CHECK_IF(JudgeLayoutByShape(xShape, cosShape) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "JudgeLayoutByShape fail."), return ge::GRAPH_FAILED);

    d_ = isTndLayout_ ? xShape.GetDim(DIM_2) : xShape.GetDim(DIM_3);
    if (is3dBsdLayout_) {
        Set3dBsdShapeAttrs(xShape);
    } else if (isTndLayout_) {
        // TND: (T, N, D) -> b=1, s=T, n=N, d=D, cosb=1
        b_ = 1;
        cosb_ = 1;
        s_ = xShape.GetDim(DIM_0);
        n_ = xShape.GetDim(DIM_1);
    } else if (layout_ == RopeLayout::BSND) {
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
        if (is1snd_ == true) {
            s_ = s_ * n_;
            n_ = 1;
        }
    } else if (layout_ == RopeLayout::SBND) {
        s_ = xShape.GetDim(DIM_0);
        b_ = xShape.GetDim(DIM_1);
        cosb_ = cosShape.GetDim(DIM_1);
        n_ = xShape.GetDim(DIM_2);
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
