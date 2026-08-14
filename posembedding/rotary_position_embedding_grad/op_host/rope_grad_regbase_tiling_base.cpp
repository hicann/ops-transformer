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
 * \file rope_grad_regbase_tiling_base.cc
 * \brief
 */
#include <cmath>
#include <graph/utils/type_utils.h>
#include "tiling/tiling_api.h"
#include "posembedding/rotary_position_embedding_grad/op_kernel/arch35/rotary_position_embedding_grad_dag.h"
#include "posembedding/rotary_position_embedding_grad/op_kernel/arch35/rotary_position_embedding_grad_tiling_key.h"
#include "rotary_position_embedding_grad_tiling.h"

namespace {
constexpr int64_t DY_INDEX = 0;
constexpr int64_t COS_INDEX = 1;
constexpr int64_t SIN_INDEX = 2;
constexpr int64_t X_INDEX = 3;
constexpr int64_t DX_INDEX = 0;
constexpr int64_t DCOS_INDEX = 1;
constexpr int64_t DSIN_INDEX = 2;
constexpr int64_t DIM_NUM = 4;
constexpr int64_t DIM_NUM_TND = 3;
constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;
constexpr int64_t HALF_INTERLEAVE_MODE_COEF = 2;
constexpr int64_t QUARTER_MODE_COEF = 4;
constexpr int64_t D_LIMIT = 1024;
constexpr int64_t CACHE_LINE_SIZE = 512;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t INPUT_OUTPUT_NUM = 2;
constexpr int64_t INPUT_TWO_OUTPUT = 3; // 1个输入 2个输出

const std::vector<ge::DataType> SUPPORT_DTYPE = {ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16};
static const std::vector<std::string> inputNames = {"dy", "cos", "sin", "x"};
static const std::vector<std::string> outputNames = {"dx", "dcos", "dsin"};
} // namespace

namespace optiling {
using namespace Ops::Base;
ge::graphStatus RopeGradRegBaseTilingClass::GetPlatformInfo()
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
        auto compileInfoPtr = context_->GetCompileInfo<RotaryPositionEmbeddingGradCompileInfo>();
        OP_CHECK_IF(
            compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"),
            return ge::GRAPH_FAILED);
        aicoreParams_.numBlocks = compileInfoPtr->numBlocks;
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
        socVersion_ = compileInfoPtr->socVersion;
    }
    blockSize_ = Ops::Base::GetUbBlockSize(context_);
    vLength_ = Ops::Base::GetVRegSize(context_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeGradRegBaseTilingClass::CheckNullptr()
{
    int64_t checkInputIndexRange = X_INDEX;
    int64_t checkOutputIndexRange = DSIN_INDEX;
    if (context_->GetInputShape(X_INDEX) == nullptr) {
        checkInputIndexRange = SIN_INDEX;
        checkOutputIndexRange = DX_INDEX;
    }

    for (int64_t i = 0; i <= checkInputIndexRange; i++) {
        auto desc = context_->GetInputDesc(i);
        OP_CHECK_IF(
            desc == nullptr, OP_LOGE(context_, "input %ld desc is nullptr.", i),
            return ge::GRAPH_FAILED);
        auto shape = context_->GetInputShape(i);
        OP_CHECK_IF(
            shape == nullptr, OP_LOGE(context_, "input %ld shape is nullptr.", i),
            return ge::GRAPH_FAILED);
    }

    for (int64_t i = 0; i <= checkOutputIndexRange; i++) {
        auto desc = context_->GetInputDesc(i);
        OP_CHECK_IF(
            desc == nullptr, OP_LOGE(context_, "output %ld desc is nullptr.", i),
            return ge::GRAPH_FAILED);
        auto shape = context_->GetInputShape(i);
        OP_CHECK_IF(
            shape == nullptr, OP_LOGE(context_, "output %ld shape is nullptr.", i),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeGradRegBaseTilingClass::CheckInPutShapeAllPositive(const int64_t idx) const
{
    auto shape = context_->GetInputShape(idx)->GetStorageShape();
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        // 只拦截负维度；0 维(empty tensor)放行，由空进空出逻辑处理
        if (shape.GetDim(i) < 0) {
            std::string shapeMsg = ToString(shape);
            std::string reasonMsg = "The shape of input " + inputNames[idx] +
                                    " can not be an invalid tensor with a negative dimension";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), inputNames[idx].c_str(),
                                                  shapeMsg.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeGradRegBaseTilingClass::CheckOutPutShapeAllPositive(const int64_t idx) const
{
    auto shape = context_->GetOutputShape(idx)->GetStorageShape();
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        // 只拦截负维度；0 维(empty tensor)放行，由空进空出逻辑处理
        if (shape.GetDim(i) < 0) {
            std::string shapeMsg = ToString(shape);
            std::string reasonMsg = "The shape of output " + outputNames[idx] +
                                    " can not be an invalid tensor with a negative dimension";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), outputNames[idx].c_str(),
                                                  shapeMsg.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

bool RopeGradRegBaseTilingClass::IsRotaryPosEmbeddingMode(const int32_t mode) const
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

ge::graphStatus RopeGradRegBaseTilingClass::CheckShapeAllPositive() const
{
    // 所有输入/输出均放行 0 维（空进空出）；"dy 非空仅 cos/sin 空"由广播约束/D 维校验天然拦截，无需在此拦截
    OP_CHECK_IF(
        CheckInPutShapeAllPositive(DY_INDEX) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_, "dy has negative shape."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckInPutShapeAllPositive(COS_INDEX) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_, "cos has negative shape."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckInPutShapeAllPositive(SIN_INDEX) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_, "sin has negative shape."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckOutPutShapeAllPositive(DX_INDEX) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_, "dx has negative shape."), return ge::GRAPH_FAILED);
    if (context_->GetInputShape(X_INDEX) != nullptr) {
        OP_CHECK_IF(
            CheckInPutShapeAllPositive(X_INDEX) != ge::GRAPH_SUCCESS,
            OP_LOGE(context_, "x has negative shape."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            CheckOutPutShapeAllPositive(DCOS_INDEX) != ge::GRAPH_SUCCESS,
            OP_LOGE(context_, "dcos has negative shape."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            CheckOutPutShapeAllPositive(DSIN_INDEX) != ge::GRAPH_SUCCESS,
            OP_LOGE(context_, "dsin has negative shape."), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

bool RopeGradRegBaseTilingClass::Is3dBsdBroadcastLayout(const gert::Shape &xShape, const gert::Shape &cosShape) const
{
    // 识别 dy（TND）cos（1ND）场景
    return xShape.GetDimNum() == DIM_NUM_TND && cosShape.GetDimNum() == DIM_NUM_TND &&
           xShape.GetDim(DIM_1) != 1 && cosShape.GetDim(DIM_0) == 1 &&
           cosShape.GetDim(DIM_1) == xShape.GetDim(DIM_1) && cosShape.GetDim(DIM_2) == xShape.GetDim(DIM_2);
}

ge::graphStatus RopeGradRegBaseTilingClass::JudgeLayoutByShape(const gert::Shape &xShape, const gert::Shape &cosShape)
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
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "cos and dy", shapeMsg.c_str(),
                                              "Each axis of input cos except the last must be 1 or equal to the same axis of input dy");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeGradRegBaseTilingClass::CheckShapeDim() const
{
    auto &dyShape = context_->GetInputShape(DY_INDEX)->GetStorageShape();
    auto &cosShape = context_->GetInputShape(COS_INDEX)->GetStorageShape();
    auto &sinShape = context_->GetInputShape(SIN_INDEX)->GetStorageShape();
    auto &dxShape = context_->GetOutputShape(DX_INDEX)->GetStorageShape();

    if (dyShape.GetDimNum() != DIM_NUM && dyShape.GetDimNum() != DIM_NUM_TND) {
        std::string dimNumStr = std::to_string(dyShape.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "dy", dimNumStr.c_str(), "3D or 4D");
        return ge::GRAPH_FAILED;
    }
    if (context_->GetInputShape(X_INDEX) != nullptr) {
        auto &xShape = context_->GetInputShape(X_INDEX)->GetStorageShape();
        auto &dcosShape = context_->GetOutputShape(DCOS_INDEX)->GetStorageShape();
        auto &dsinShape = context_->GetOutputShape(DSIN_INDEX)->GetStorageShape();
        if (xShape.GetDimNum() != DIM_NUM && xShape.GetDimNum() != DIM_NUM_TND) {
            std::string dimNumStr = std::to_string(xShape.GetDimNum());
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "x", dimNumStr.c_str(), "3D or 4D");
            return ge::GRAPH_FAILED;
        }
        if (dcosShape.GetDimNum() != DIM_NUM && dcosShape.GetDimNum() != DIM_NUM_TND) {
            std::string dimNumStr = std::to_string(dcosShape.GetDimNum());
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "dcos", dimNumStr.c_str(), "3D or 4D");
            return ge::GRAPH_FAILED;
        }
        if (dsinShape.GetDimNum() != DIM_NUM && dsinShape.GetDimNum() != DIM_NUM_TND) {
            std::string dimNumStr = std::to_string(dsinShape.GetDimNum());
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "dsin", dimNumStr.c_str(), "3D or 4D");
            return ge::GRAPH_FAILED;
        }
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
    if (dxShape.GetDimNum() != DIM_NUM && dxShape.GetDimNum() != DIM_NUM_TND) {
        std::string dimNumStr = std::to_string(dxShape.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "dx", dimNumStr.c_str(), "3D or 4D");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeGradRegBaseTilingClass::CheckShapeLimit()
{
    auto &dyShape = context_->GetInputShape(DY_INDEX)->GetStorageShape();
    auto &cosShape = context_->GetInputShape(COS_INDEX)->GetStorageShape();
    auto &sinShape = context_->GetInputShape(SIN_INDEX)->GetStorageShape();
    auto &dxShape = context_->GetOutputShape(DX_INDEX)->GetStorageShape();

    if (cosShape != sinShape) {
        std::string shapeMsg = ToString(cosShape) + " and " + ToString(sinShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "cos and sin", shapeMsg.c_str(),
                                               "The shapes of input cos and sin should be the same");
        return ge::GRAPH_FAILED;
    }
    if (dyShape != dxShape) {
        std::string shapeMsg = ToString(dyShape) + " and " + ToString(dxShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "dy and dx", shapeMsg.c_str(),
                                               "The shapes of input dy and output dx should be the same");
        return ge::GRAPH_FAILED;
    }

    if (context_->GetInputShape(X_INDEX) != nullptr) {
        dCosFlag_ = 1;
        return CheckOptionalInput();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeGradRegBaseTilingClass::CheckOptionalInput() const
{
    auto &cosShape = context_->GetInputShape(COS_INDEX)->GetStorageShape();
    auto &sinShape = context_->GetInputShape(SIN_INDEX)->GetStorageShape();
    auto &dxShape = context_->GetOutputShape(DX_INDEX)->GetStorageShape();
    auto &xShape = context_->GetInputShape(X_INDEX)->GetStorageShape();
    auto &dcosShape = context_->GetOutputShape(DCOS_INDEX)->GetStorageShape();
    auto &dsinShape = context_->GetOutputShape(DSIN_INDEX)->GetStorageShape();
    if (xShape != dxShape) {
        std::string shapeMsg = ToString(xShape) + " and " + ToString(dxShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "x and dx", shapeMsg.c_str(),
                                               "The shapes of input x and output dx should be the same");
        return ge::GRAPH_FAILED;
    }
    if (cosShape != dcosShape) {
        std::string shapeMsg = ToString(cosShape) + " and " + ToString(dcosShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "cos and dcos", shapeMsg.c_str(),
                                               "The shapes of input cos and output dcos should be the same");
        return ge::GRAPH_FAILED;
    }
    if (sinShape != dsinShape) {
        std::string shapeMsg = ToString(sinShape) + " and " + ToString(dsinShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "sin and dsin", shapeMsg.c_str(),
                                               "The shapes of input sin and output dsin should be the same");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeGradRegBaseTilingClass::CheckShape()
{
    auto &cosShape = context_->GetInputShape(COS_INDEX)->GetStorageShape();
    auto &dyShape = context_->GetInputShape(DY_INDEX)->GetStorageShape();
    OP_CHECK_IF(
        CheckShapeDim() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check shape dim fail."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckShapeLimit() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "CheckShapeLimit fail."),
        return ge::GRAPH_FAILED);
    isTndLayout_ = (dyShape.GetDimNum() == DIM_NUM_TND);
    int64_t dyLastDim = isTndLayout_ ? dyShape.GetDim(DIM_2) : dyShape.GetDim(DIM_3);
    int64_t cosLastDim = isTndLayout_ ? cosShape.GetDim(DIM_2) : cosShape.GetDim(DIM_3);
    if (cosLastDim != dyLastDim) {
        std::string shapeMsg = ToString(dyShape) + " and " + ToString(cosShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "dy and cos", shapeMsg.c_str(),
                                               "The D axis of input dy and input cos should be the same, where D refers to the last dim");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        CheckRotaryModeShapeRelation(dyLastDim) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_, "D is invalid for rotary mode."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeGradRegBaseTilingClass::CheckDtypeAndAttr()
{
    dtype_ = context_->GetInputDesc(DY_INDEX)->GetDataType();
    if (std::find(SUPPORT_DTYPE.begin(), SUPPORT_DTYPE.end(), dtype_) == SUPPORT_DTYPE.end()) {
        std::string dtypeStr = ToString(dtype_);
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "dy", dtypeStr.c_str(), "FLOAT, BF16 or FLOAT16");
        return ge::GRAPH_FAILED;
    }
    int64_t checkInputIndexRange = SIN_INDEX;
    int64_t checkOutputIndexRange = DX_INDEX;
    if (context_->GetInputShape(X_INDEX) != nullptr) {
        checkInputIndexRange = X_INDEX;
        checkOutputIndexRange = DSIN_INDEX;
    }

    for (int64_t i = DY_INDEX; i <= checkInputIndexRange; i++) {
        auto type = context_->GetInputDesc(i)->GetDataType();
        if (type != dtype_) {
            std::string paramMsg = inputNames[i] + " and dy";
            std::string dtypeMsg = ToString(type) + " and " + ToString(dtype_);
            std::string reasonMsg = "The dtypes of input " + inputNames[i] + " and input dy should be the same";
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), paramMsg.c_str(),
                                                   dtypeMsg.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    for (int64_t i = DX_INDEX; i <= checkOutputIndexRange; i++) {
        auto type = context_->GetOutputDesc(i)->GetDataType();
        if (type != dtype_) {
            std::string paramMsg = outputNames[i] + " and dy";
            std::string dtypeMsg = ToString(type) + " and " + ToString(dtype_);
            std::string reasonMsg = "The dtypes of output " + outputNames[i] + " and input dy should be the same";
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), paramMsg.c_str(),
                                                   dtypeMsg.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeGradRegBaseTilingClass::CheckParam()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(
        platformInfo == nullptr, OP_LOGE(context_, "platform info is nullptr."),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    if (!Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(
        CheckNullptr() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check nullptr fail."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckDtypeAndAttr() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_, "check dtype and attr fail."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckShape() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check shape fail."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckShapeAllPositive() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_, "check shape positive fail."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeGradRegBaseTilingClass::CheckRotaryModeShapeRelation(const int64_t d)
{
    auto dyShape = context_->GetInputShape(DY_INDEX)->GetStorageShape();
    if (d > D_LIMIT) {
        std::string shapeMsg = ToString(dyShape);
        std::string reasonMsg = "The D axis of input dy can not be greater than " + std::to_string(D_LIMIT) +
                                ", where D refers to the last dim";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "dy",
                                              shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    if (rotaryMode_ == RotaryPosEmbeddingMode::HALF || rotaryMode_ == RotaryPosEmbeddingMode::INTERLEAVE ||
        rotaryMode_ == RotaryPosEmbeddingMode::DEEPSEEK_INTERLEAVE) {
        if (d % HALF_INTERLEAVE_MODE_COEF != 0) {
            std::string shapeMsg = ToString(dyShape);
            std::string reasonMsg =
                "The D axis of input dy should be divisible by " + std::to_string(HALF_INTERLEAVE_MODE_COEF) +
                " when the attr mode is half, interleave or deepseek_interleave, "
                "where D refers to the last dim";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "dy",
                                                  shapeMsg.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    } else if (rotaryMode_ == RotaryPosEmbeddingMode::QUARTER) {
        if (d % QUARTER_MODE_COEF != 0) {
            std::string shapeMsg = ToString(dyShape);
            std::string reasonMsg =
                "The D axis of input dy should be divisible by " + std::to_string(QUARTER_MODE_COEF) +
                " when the attr mode is quarter, where D refers to the last dim";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "dy",
                                                  shapeMsg.c_str(), reasonMsg.c_str());
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

void RopeGradRegBaseTilingClass::Set3dBsdShapeAttrs(const gert::Shape &xShape)
{
    // TND的广播场景需要映射成为BS1D，用于分核优化
    b_ = xShape.GetDim(DIM_0);
    cosb_ = 1; // 实际就为1
    s_ = xShape.GetDim(DIM_1);
    n_ = 1;
}

ge::graphStatus RopeGradRegBaseTilingClass::GetShapeAttrsInfo()
{
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const int32_t *mode = attrs->GetAttrPointer<int32_t>(0);
    int32_t modeValue = (mode == nullptr) ? 0 : static_cast<int32_t>(*mode);
    if (IsRotaryPosEmbeddingMode(modeValue) != true) {
        std::string modeStr = std::to_string(modeValue);
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "mode",
                                  modeStr.c_str(), "0, 1, 2 or 3");
        return ge::GRAPH_FAILED;
    }
    rotaryMode_ = static_cast<RotaryPosEmbeddingMode>(modeValue);

    OP_CHECK_IF(
        CheckParam() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check param fail."),
        return ge::GRAPH_FAILED);

    dtype_ = context_->GetInputDesc(DY_INDEX)->GetDataType();
    dyShape_ = context_->GetInputShape(DY_INDEX)->GetStorageShape();
    cosShape_ = context_->GetInputShape(COS_INDEX)->GetStorageShape();
    OP_CHECK_IF(
        JudgeLayoutByShape(dyShape_, cosShape_) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_, "JudgeLayoutByShape fail."), return ge::GRAPH_FAILED);

    d_ = isTndLayout_ ? dyShape_.GetDim(DIM_2) : dyShape_.GetDim(DIM_3);
    if (is3dBsdLayout_) {
        Set3dBsdShapeAttrs(dyShape_);
    } else if (isTndLayout_) {
        // TND: (T, N, D) -> b=1, s=T, n=N, d=D, cosb=1
        b_ = 1;
        cosb_ = 1;
        s_ = dyShape_.GetDim(DIM_0);
        n_ = dyShape_.GetDim(DIM_1);
    } else if (layout_ == RopeLayout::BSND) {
        b_ = dyShape_.GetDim(DIM_0);
        cosb_ = cosShape_.GetDim(DIM_0);
        s_ = dyShape_.GetDim(DIM_1);
        n_ = dyShape_.GetDim(DIM_2);
    } else if (
        layout_ == RopeLayout::BNSD || layout_ == RopeLayout::NO_BROADCAST || layout_ == RopeLayout::BROADCAST_BSN) {
        b_ = dyShape_.GetDim(DIM_0);
        cosb_ = cosShape_.GetDim(DIM_0);
        n_ = dyShape_.GetDim(DIM_1);
        s_ = dyShape_.GetDim(DIM_2);
        // 1XXX情况下，reshape成11XX
        if (is1snd_ == true) {
            s_ = s_ * n_;
            n_ = 1;
        }
    } else if (layout_ == RopeLayout::SBND) {
        s_ = dyShape_.GetDim(DIM_0);
        b_ = dyShape_.GetDim(DIM_1);
        cosb_ = cosShape_.GetDim(DIM_1);
        n_ = dyShape_.GetDim(DIM_2);
    }
    // dy/x 为空 tensor 时（空进空出），后续分核/规约的 host 计算按维度 1 处理，
    // kernel 侧依据 isEmptyDy 跳过全部有效计算，仅对 dcos/dsin 清零
    isEmptyDy_ = (dyShape_.GetShapeSize() == 0);
    if (isEmptyDy_) {
        b_ = (b_ <= 0) ? 1 : b_;
        s_ = (s_ <= 0) ? 1 : s_;
        n_ = (n_ <= 0) ? 1 : n_;
        // D=0 时所有输出均为空（cosShapeSize=0），同样清洗避免下游除零；
        // 取 dSplitCoef_（mode 0/3=2、mode 2=4、mode 1=1）保证 dAlign_ 等下游计算结果为正
        d_ = (d_ <= 0) ? dSplitCoef_ : d_;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeGradRegBaseTilingClass::GetInputParam(
    Ops::Base::ReduceOpInputParam &opInput, uint32_t inputIdx, uint32_t axesIdx)
{
    // set opInput dtype
    auto inputDesc = context_->GetInputDesc(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    opInput.inputDtype = inputDesc->GetDataType();
    // set opInput shape
    auto dyInput = context_->GetInputTensor(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dyInput);
    auto dyInputShape = Ops::Transformer::OpTiling::EnsureNotScalar(dyInput->GetStorageShape());
    size_t shapeSize = dyInputShape.GetDimNum();
    opInput.shape.resize(shapeSize);
    for (size_t i = 0; i < shapeSize; i++) {
        opInput.shape[i] = dyInputShape.GetDim(i);
    }
    // set opInput axes
    inputDesc = context_->GetInputDesc(axesIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    auto cosInput = context_->GetInputTensor(axesIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context_, cosInput);
    auto cosShape = Ops::Transformer::OpTiling::EnsureNotScalar(cosInput->GetStorageShape());
    for (size_t i = 0; i < shapeSize; i++) {
        if (cosShape.GetDim(i) == 1 && opInput.shape[i] != 1) {
            opInput.axes.push_back(i);
        }
    }
    if (isEmptyDy_) {
        // 空 tensor 时规约轴维度按 1 处理，使 reduce tiling 按正常配置走通并生成标准 tilingKey；
        // axes 仍按真实 shape 计算，保证 key 与非空等价配置一致
        for (size_t i = 0; i < shapeSize; i++) {
            opInput.shape[i] = (opInput.shape[i] <= 0) ? 1 : opInput.shape[i];
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeGradRegBaseTilingClass::InitTilingData()
{
    if (tilingData_ == nullptr) {
        tilingData_ = context_->GetTilingData<RopeGradTilingData>();
        OP_CHECK_IF(
            tilingData_ == nullptr,
            OP_LOGE(context_->GetNodeName(), "get tilingdata ptr failed"),
            return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(
        (memset_s(tilingData_, sizeof(RopeGradTilingData), 0, sizeof(RopeGradTilingData)) != EOK),
        OP_LOGE(context_->GetNodeName(), "memset tilingdata failed"), return ge::GRAPH_FAILED);
    // 空进空出标记：kernel 据此跳过全部有效计算，仅将 dcos/dsin 清零（cosShapeSize 个元素）
    tilingData_->isEmptyDy = isEmptyDy_ ? 1U : 0U;
    tilingData_->cosShapeSize = cosShape_.GetShapeSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeGradRegBaseTilingClass::GetReduceOpCompileInfo(Ops::Base::ReduceOpCompileInfo *compileInfo)
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->vectorCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->vectorCoreNum == 0),
        OP_LOGE(
            context_->GetNodeName(), "ReduceOp GetHardwareInfo Failed, vectorCoreNum:%lu", compileInfo->vectorCoreNum),
        return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(
        ubSize <= static_cast<uint64_t>(Ops::Base::CACHE_BUF_SIZE),
        OP_LOGE(
            context_->GetNodeName(), "ReduceOp GetHardwareInfo Failed, ubSize:%lu, at least:%ld.", compileInfo->ubSize,
            Ops::Base::CACHE_BUF_SIZE),
        return ge::GRAPH_FAILED);
    compileInfo->ubSize = ubSize;

    compileInfo->cacheLineSize = Ops::Base::GetCacheLineSize(context_);
    OP_CHECK_IF(
        compileInfo->cacheLineSize == 0,
        OP_LOGE(
            context_->GetNodeName(), "ReduceOp GetHardwareInfo Failed, cacheLineSize:%lu.", compileInfo->cacheLineSize),
        return ge::GRAPH_FAILED);

    compileInfo->ubBlockSize = Ops::Base::GetUbBlockSize(context_);
    OP_CHECK_IF(
        compileInfo->ubBlockSize == 0,
        OP_LOGE(
            context_->GetNodeName(), "ReduceOp GetHardwareInfo Failed, ubBlockSize:%lu.", compileInfo->ubBlockSize),
        return ge::GRAPH_FAILED);

    compileInfo->vRegSize = Ops::Base::GetVRegSize(context_);
    OP_CHECK_IF(
        compileInfo->vRegSize == 0,
        OP_LOGE(
            context_->GetNodeName(), "ReduceOp GetHardwareInfo Failed, vRegSize:%lu.", compileInfo->vRegSize),
        return ge::GRAPH_FAILED);

    OP_LOGD(
        context_->GetNodeName(), "GetCoreNum:%lu, ubSize:%lu, cacheLineSize:%lu, ubBlockSize:%lu, vRegSize:%lu",
        compileInfo->vectorCoreNum, compileInfo->ubSize, compileInfo->cacheLineSize, compileInfo->ubBlockSize,
        compileInfo->vRegSize);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeGradRegBaseTilingClass::TilingReduce()
{
    auto compileInfo = context_->GetCompileInfo<RotaryPositionEmbeddingGradCompileInfo>();
    Ops::Base::ReduceOpCompileInfo opInfo;
    if (compileInfo == nullptr) {
        OP_CHECK_IF(
            (GetReduceOpCompileInfo(&opInfo) == ge::GRAPH_FAILED),
            OP_LOGE(context_->GetNodeName(), "GetReduceOpCompileInfo failed"),
            return ge::GRAPH_FAILED);
    } else {
        opInfo = compileInfo->opInfo;
    }
    Ops::Base::ReduceOpInputParam opInput;
    OP_CHECK_IF(
        (GetInputParam(opInput, 0, 1) == ge::GRAPH_FAILED),
        OP_LOGE(context_->GetNodeName(), "ReduceOp get dy input param failed"),
        return ge::GRAPH_FAILED);
    ge::graphStatus status;
    if (ge::GetSizeByDataType(opInput.inputDtype) == sizeof(float)) {
        status = Ops::Base::Tiling4ReduceOp<RotaryPositionEmbeddingGrad::RotaryPositionEmbeddingGradDag<float, float>::OpDag>(
            context_, opInput, key_, &opInfo, &tilingData_->reduceTiling);
    } else {
        status = Ops::Base::Tiling4ReduceOp<RotaryPositionEmbeddingGrad::RotaryPositionEmbeddingGradDag<Ops::Base::half, float>::OpDag>(
            context_, opInput, key_, &opInfo, &tilingData_->reduceTiling);
    }
    OP_CHECK_IF(
        (status == ge::GRAPH_FAILED),
        OP_LOGE(context_->GetNodeName(), "ReduceOp Tiling failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeGradRegBaseTilingClass::SetTilingKeyBlockDim(uint32_t dxTilingKey)
{
    GEN_REDUCE_TILING_KEY(tilingKey_, key_, dxTilingKey, dCosFlag_);
    OP_LOGI(
        context_->GetNodeName(),
        "patternID:%u, loopARCount:%u, loopInnerARCount:%u, dxTilingKey is: %u, tilingKey is:%lu.", key_.patternID,
        key_.loopARCount, key_.loopInnerARCount, dxTilingKey, tilingKey_);
    int64_t reduceBlockNum = context_->GetBlockDim();
    context_->SetBlockDim(std::max(usedCoreNum_, reduceBlockNum));
    OP_LOGD(context_->GetNodeName(), "reduceBlockNum :%ld, usedCoreNum_ = %ld.\n", reduceBlockNum, usedCoreNum_);
    context_->SetTilingKey(tilingKey_);
    auto workspaces = context_->GetWorkspaceSizes(1);
    auto usrWorkSpaceSize = b_ * s_ * n_ * d_ * ge::GetSizeByDataType(dtype_);
    if (rotaryMode_ == RotaryPosEmbeddingMode::DEEPSEEK_INTERLEAVE) {
        usrWorkSpaceSize = usrWorkSpaceSize * INPUT_OUTPUT_NUM; // 需要两块workspace大小
    }
    workspaces[0] = workspaces[0] + usrWorkSpaceSize;
    if (dCosFlag_ == 1) {
        context_->SetScheduleMode(1); // 设置为batch mode模式，所有核同时启动
    }
    OP_LOGD(context_->GetNodeName(), "workspaces[0] :%ld, usrWorkSpaceSize = %ld.\n", workspaces[0], usrWorkSpaceSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeGradRegBaseTilingClass::SetRotaryXTilingData()
{
    int64_t ubFactorN = 1;
    int64_t dSumNum = b_ * s_ * n_;
    int64_t blockFactorN = Ops::Base::CeilDiv(dSumNum, usedCoreNum_);
    int64_t rotaryUsedCoreNum = Ops::Base::CeilDiv(dSumNum, blockFactorN);
    int64_t blockTailFactorN = (dSumNum % blockFactorN == 0) ? blockFactorN : (dSumNum % blockFactorN);
    int64_t dSize = Ops::Base::CeilAlign(d_ * GetSizeByDataType(dtype_) / dSplitCoef_, this->blockSize_) * dSplitCoef_;
    int64_t numOfDAvailable = Ops::Base::FloorDiv(static_cast<int64_t>(aicoreParams_.ubSize), DOUBLE_BUFFER * dSize);
    ubFactorN = numOfDAvailable / INPUT_TWO_OUTPUT; // 1个输入 2个输出
    ubFactorN = std::min(blockFactorN, ubFactorN);
    tilingData_->rotaryXParams.b = b_;
    tilingData_->rotaryXParams.s = s_;
    tilingData_->rotaryXParams.n = n_;
    tilingData_->rotaryXParams.d = d_;
    tilingData_->rotaryXParams.usedCoreNum = rotaryUsedCoreNum;
    tilingData_->rotaryXParams.blockFactorN = blockFactorN;
    tilingData_->rotaryXParams.blockTailFactorN = blockTailFactorN;
    tilingData_->rotaryXParams.ubFactorN = ubFactorN;
    tilingData_->rotaryXParams.rotaryMode = static_cast<int64_t>(rotaryMode_);

    OP_LOGI(
        context_->GetNodeName(),
        "RotaryX tilingData: usedCoreNum = %ld, blockFactorN %ld, blockTailFactorN is %ld, ubFactorN is %ld",
        tilingData_->rotaryXParams.usedCoreNum, tilingData_->rotaryXParams.blockFactorN,
        tilingData_->rotaryXParams.blockTailFactorN, tilingData_->rotaryXParams.ubFactorN);
    return ge::GRAPH_SUCCESS;
}

uint64_t RopeGradRegBaseTilingClass::GetTilingKey() const
{
    return tilingKey_;
}

} // namespace optiling
