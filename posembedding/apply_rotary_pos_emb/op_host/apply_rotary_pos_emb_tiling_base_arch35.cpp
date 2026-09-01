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
 * \file apply_rotary_pos_emb_regbase_tiling_base.cpp
 * \brief
 */
#include <algorithm>
#include "apply_rotary_pos_emb_tiling.h"
#include "log/log.h"
#include <algorithm>

namespace {
constexpr int64_t Q_INDEX = 0;
constexpr int64_t K_INDEX = 1;
constexpr int64_t COS_INDEX = 2;
constexpr int64_t SIN_INDEX = 3;
constexpr int64_t QOUT_INDEX = 0;
constexpr int64_t KOUT_INDEX = 1;
constexpr int64_t DIM_NUM = 4;
constexpr int64_t DIM_NUM_TND = 3;
constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;
constexpr int64_t HALF_INTERLEAVE_MODE_COEF = 2;
constexpr int64_t QUARTER_MODE_COEF = 4;
constexpr int64_t D_LIMIT = 1024;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t V_REG_SIZE = 256;
const std::vector<ge::DataType> SUPPORT_DTYPE = {ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16};
} // namespace

namespace optiling {
using namespace Ops::Base;
ge::graphStatus ApplyRotaryPosEmbRegbaseTilingBaseClass::GetPlatformInfo()
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
        auto compileInfoPtr = reinterpret_cast<const ApplyRotaryPosEmbCompileInfo *>(context_->GetCompileInfo());
        OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfoPtr);
        aicoreParams_.numBlocks = compileInfoPtr->numBlocks;
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
        socVersion_ = compileInfoPtr->socVersion;
    }
    blockSize_ = BLOCK_SIZE;
    vLength_ = V_REG_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbRegbaseTilingBaseClass::CheckNullptr()
{
    auto qDesc = context_->GetInputDesc(Q_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, qDesc);
    qDataType_ = qDesc->GetDataType();
    auto qShapePoint = context_->GetInputShape(Q_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, qShapePoint);
    qShape_ = qShapePoint->GetStorageShape();

    auto kDesc = context_->GetInputDesc(K_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kDesc);
    kDataType_ = kDesc->GetDataType();
    auto kShapePoint = context_->GetInputShape(K_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kShapePoint);
    kShape_ = kShapePoint->GetStorageShape();

    auto cosDesc = context_->GetInputDesc(COS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, cosDesc);
    cosDataType_ = cosDesc->GetDataType();
    auto cosShapePoint = context_->GetInputShape(COS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, cosShapePoint);
    cosShape_ = cosShapePoint->GetStorageShape();

    auto sinDesc = context_->GetInputDesc(SIN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sinDesc);
    sinDataType_ = sinDesc->GetDataType();
    auto sinShapePoint = context_->GetInputShape(SIN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sinShapePoint);
    sinShape_ = sinShapePoint->GetStorageShape();

    auto qOutDesc = context_->GetOutputDesc(QOUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, qOutDesc);
    qOutDataType_ = qOutDesc->GetDataType();
    auto qOutShapePoint = context_->GetOutputShape(QOUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, qOutShapePoint);
    qOutShape_ = qOutShapePoint->GetStorageShape();

    auto kOutDesc = context_->GetOutputDesc(KOUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kOutDesc);
    kOutDataType_ = kOutDesc->GetDataType();
    auto kOutShapePoint = context_->GetOutputShape(KOUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kOutShapePoint);
    kOutShape_ = kOutShapePoint->GetStorageShape();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbRegbaseTilingBaseClass::CheckShapeAllPositive(const int64_t &idx,
                                                                               const gert::Shape &shape)
{
    static const std::vector<std::string> inputNames = {"query", "key", "cos", "sin"};
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        if (shape.GetDim(i) <= 0) {
            std::string reasonMsg = "The shape of input " + inputNames[idx] +
                " can not be an invalid tensor with a negative dimension";
            std::string shapeStr = ToString(shape);
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), inputNames[idx].c_str(),
                shapeStr.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbRegbaseTilingBaseClass::CheckShapeAllPositive()
{
    ge::graphStatus qRes = CheckShapeAllPositive(Q_INDEX, qShape_);
    ge::graphStatus kRes = CheckShapeAllPositive(K_INDEX, kShape_);
    // q, k有一个为空Tensor时，提示用 rotaryPositionEmbedding 算子
    OP_CHECK_IF(qRes != kRes,
                OP_LOGE(context_, "q or k has non positive shape, please use rotaryPositionEmbedding operator"),
                return ge::GRAPH_FAILED);
    // q, k 都为空Tensor，处理空tensor
    OP_CHECK_IF(qRes != ge::GRAPH_SUCCESS, OP_LOGE(context_, "query and key have non positive shape."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckShapeAllPositive(COS_INDEX, cosShape_) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_, "cos has non positive shape."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckShapeAllPositive(SIN_INDEX, sinShape_) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_, "sin has non positive shape."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbRegbaseTilingBaseClass::CheckRotaryModeShapeRelation(const int64_t &d)
{
    if (d > D_LIMIT) {
        std::string reason = "The D axis of input query can not be greater than " + std::to_string(D_LIMIT) +
            ", where D axis refers to the last dim";
        std::string qShapeStr = ToString(qShape_);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "query",
            qShapeStr.c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    if (rotaryModeStr_ == "half" || rotaryModeStr_ == "interleave") {
        if (d % HALF_INTERLEAVE_MODE_COEF != 0) {
            std::string reason = "The D axis of input query should be divisible by " + std::to_string(HALF_INTERLEAVE_MODE_COEF) +
                " when the attr rotary_mode is half and interleave, where D axis refers to the last dim";
            std::string qShapeStr = ToString(qShape_);
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "query",
                qShapeStr.c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
    } else if (rotaryModeStr_ == "quarter") {
        if (d % QUARTER_MODE_COEF != 0) {
            std::string reason = "The D axis of input query should be divisible by " + std::to_string(QUARTER_MODE_COEF) +
                " when the attr rotary_mode is quarter, where D axis refers to the last dim";
            std::string qShapeStr = ToString(qShape_);
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "query",
                qShapeStr.c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbRegbaseTilingBaseClass::CheckShapeRelation()
{
    static const std::map<ApplyRotaryPosEmbLayout, std::string> layoutMap = {
        {ApplyRotaryPosEmbLayout::BSND, "BSND"},
        {ApplyRotaryPosEmbLayout::SBND, "SBND"},
        {ApplyRotaryPosEmbLayout::BNSD, "BNSD"},
        {ApplyRotaryPosEmbLayout::TND, "TND"}
    };
    int64_t bIdx = DIM_0;
    int64_t sIdx = DIM_1;
    int64_t nIdx = DIM_2;
    int64_t dIdx = DIM_3;
    if (layout_ == ApplyRotaryPosEmbLayout::BNSD) {
        sIdx = DIM_2;
        nIdx = DIM_1;
    } else if (layout_ == ApplyRotaryPosEmbLayout::SBND) {
        sIdx = DIM_0;
        bIdx = DIM_1;
        nIdx = DIM_2;
    } else if (layout_ == ApplyRotaryPosEmbLayout::TND) {
        sIdx = DIM_0;
        nIdx = DIM_1;
        dIdx = DIM_2;
    }
    if (cosShape_ != sinShape_) {
        std::string shapeMsg = ToString(cosShape_) + " and " + ToString(sinShape_);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "cos and sin",
            shapeMsg.c_str(), "The shapes of input cos and input sin should be the same");
        return ge::GRAPH_FAILED;
    }
    if (qShape_ != qOutShape_) {
        std::string shapeMsg = ToString(qShape_) + " and " + ToString(qOutShape_);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "query(input) and query(output)",
            shapeMsg.c_str(), "The shapes of input query and output query should be the same");
        return ge::GRAPH_FAILED;
    }
    if (kShape_ != kOutShape_) {
        std::string shapeMsg = ToString(kShape_) + " and " + ToString(kOutShape_);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "key(input) and key(output)",
            shapeMsg.c_str(), "The shapes of input key and output key should be the same");
        return ge::GRAPH_FAILED;
    }
    if (cosShape_.GetDim(nIdx) != 1) {
        std::string reasonMsg = "The N axis of input cos should be 1, "
            "where N refers to the " + std::to_string(nIdx) + "th dim when the attr layout is " + layoutMap.at(layout_);
        std::string cosShapeStr = ToString(cosShape_);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "cos",
            cosShapeStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    if (!(cosShape_.GetDim(sIdx) == qShape_.GetDim(sIdx) && kShape_.GetDim(sIdx) == qShape_.GetDim(sIdx))) {
        std::string shapeMsg = ToString(cosShape_) + ", " + ToString(kShape_) + " and " + ToString(qShape_);
        std::string reason = "The S axes of input query, cos and key should be the same, "
            "where S refers to the " + std::to_string(sIdx) + "th dim when the attr layout is " + layoutMap.at(layout_);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "cos, key and query",
            shapeMsg.c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    if (!(cosShape_.GetDim(dIdx) <= qShape_.GetDim(dIdx) && kShape_.GetDim(dIdx) <= qShape_.GetDim(dIdx))) {
        std::string shapeMsg = ToString(cosShape_) + ", " + ToString(kShape_) + " and " + ToString(qShape_);
        std::string reasonMsg = "The D axes of input cos and key can not be greater than the D axis of input query, "
            "where D refers to the " + std::to_string(dIdx) + "th dim when the attr layout is " + layoutMap.at(layout_);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(),
            "cos, key and query", shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    if (layout_ != ApplyRotaryPosEmbLayout::TND) {
        if (kShape_.GetDim(bIdx) != qShape_.GetDim(bIdx)) {
            std::string shapeMsg = ToString(kShape_) + " and " + ToString(qShape_);
            std::string reasonMsg = "The B axes of input query and key should be the same, "
                "where B refers to the " + std::to_string(bIdx) + "th dim when the attr layout is " + layoutMap.at(layout_);
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "key and query",
                shapeMsg.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
        if (cosShape_.GetDim(bIdx) != qShape_.GetDim(bIdx) && cosShape_.GetDim(bIdx) != 1) {
            std::string shapeMsg = ToString(cosShape_) + " and " + ToString(qShape_);
            std::string reasonMsg = "The B axis of input cos should be 1 or equal to the B axis of input query, "
                "where B refers to the " + std::to_string(bIdx) + "th dim when the attr layout is " + layoutMap.at(layout_);
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "cos and query",
                shapeMsg.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    OP_CHECK_IF(CheckRotaryModeShapeRelation(qShape_.GetDim(dIdx)) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "D is invalid for rotary mode."), return ge::GRAPH_FAILED);
    return CheckShapeAllPositive();
}

ge::graphStatus ApplyRotaryPosEmbRegbaseTilingBaseClass::CheckShape()
{
    if (qShape_.GetDimNum() != DIM_NUM && qShape_.GetDimNum() != DIM_NUM_TND) {
        std::string dimNumStr = std::to_string(qShape_.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "query",
            dimNumStr.c_str(), "3D or 4D");
        return ge::GRAPH_FAILED;
    }
    if (kShape_.GetDimNum() != DIM_NUM && kShape_.GetDimNum() != DIM_NUM_TND) {
        std::string dimNumStr = std::to_string(kShape_.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "key",
            dimNumStr.c_str(), "3D or 4D");
        return ge::GRAPH_FAILED;
    }
    if (cosShape_.GetDimNum() != DIM_NUM && cosShape_.GetDimNum() != DIM_NUM_TND) {
        std::string dimNumStr = std::to_string(cosShape_.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "cos",
            dimNumStr.c_str(), "3D or 4D");
        return ge::GRAPH_FAILED;
    }
    if (sinShape_.GetDimNum() != DIM_NUM && sinShape_.GetDimNum() != DIM_NUM_TND) {
        std::string dimNumStr = std::to_string(sinShape_.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "sin",
            dimNumStr.c_str(), "3D or 4D");
        return ge::GRAPH_FAILED;
    }
    if (qOutShape_.GetDimNum() != DIM_NUM && qOutShape_.GetDimNum() != DIM_NUM_TND) {
        std::string dimNumStr = std::to_string(qOutShape_.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "query(output)",
            dimNumStr.c_str(), "3D or 4D");
        return ge::GRAPH_FAILED;
    }
    if (kOutShape_.GetDimNum() != DIM_NUM && kOutShape_.GetDimNum() != DIM_NUM_TND) {
        std::string dimNumStr = std::to_string(kOutShape_.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "key(output)",
            dimNumStr.c_str(), "3D or 4D");
        return ge::GRAPH_FAILED;
    }

    return CheckShapeRelation();
}

ge::graphStatus ApplyRotaryPosEmbRegbaseTilingBaseClass::CheckDtypeAndAttr()
{
    OP_CHECK_IF(
        (rotaryModeStr_ != "half" && rotaryModeStr_ != "interleave" && rotaryModeStr_ != "quarter"),
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "rotary_mode",
            rotaryModeStr_.c_str(), "half, interleave or quarter"),
        return ge::GRAPH_FAILED);
    if (std::find(SUPPORT_DTYPE.begin(), SUPPORT_DTYPE.end(), qDataType_) == SUPPORT_DTYPE.end()) {
        std::string dtypeStr = ToString(qDataType_);
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "query",
            dtypeStr.c_str(), "FLOAT, BF16 or FLOAT16");
        return ge::GRAPH_FAILED;
    }
    if (kDataType_ != qDataType_) {
        std::string dtypesStr = ToString(kDataType_) + " and " + ToString(qDataType_);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "key and query",
            dtypesStr.c_str(), "The dtypes of input key and input query should be the same");
        return ge::GRAPH_FAILED;
    }
    if (cosDataType_ != qDataType_) {
        std::string dtypesStr = ToString(cosDataType_) + " and " + ToString(qDataType_);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "cos and query",
            dtypesStr.c_str(), "The dtypes of input cos and input query should be the same");
        return ge::GRAPH_FAILED;
    }
    if (sinDataType_ != qDataType_) {
        std::string dtypesStr = ToString(sinDataType_) + " and " + ToString(qDataType_);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "sin and query",
            dtypesStr.c_str(), "The dtypes of input sin and input query should be the same");
        return ge::GRAPH_FAILED;
    }
    if (qOutDataType_ != qDataType_) {
        std::string dtypesStr = ToString(qOutDataType_) + " and " + ToString(qDataType_);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "query(output) and query(input)",
            dtypesStr.c_str(), "The dtypes of output query and input query should be the same");
        return ge::GRAPH_FAILED;
    }
    if (kOutDataType_ != qDataType_) {
        std::string dtypesStr = ToString(kOutDataType_) + " and " + ToString(qDataType_);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "key(output) and query(input)",
            dtypesStr.c_str(), "The dtypes of output key and input query should be the same");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbRegbaseTilingBaseClass::CheckParam()
{
    OP_CHECK_IF(CheckNullptr() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check nullptr fail."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckDtypeAndAttr() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check dtype and attr fail."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShape() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check shape fail."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void ApplyRotaryPosEmbRegbaseTilingBaseClass::ConvertRotaryMode()
{
    if (rotaryModeStr_ == "half") {
        rotaryMode_ = ApplyRotaryPosEmbRotaryMode::HALF;
        dSplitCoef_ = HALF_INTERLEAVE_MODE_COEF;
    } else if (rotaryModeStr_ == "interleave") {
        rotaryMode_ = ApplyRotaryPosEmbRotaryMode::INTERLEAVE;
        dSplitCoef_ = 1;
    } else if (rotaryModeStr_ == "quarter") {
        rotaryMode_ = ApplyRotaryPosEmbRotaryMode::QUARTER;
        dSplitCoef_ = QUARTER_MODE_COEF;
    }
}

void ApplyRotaryPosEmbRegbaseTilingBaseClass::SetBsnd()
{
    if (layout_ == ApplyRotaryPosEmbLayout::BSND) {
        b_ = qShape_.GetDim(DIM_0);
        cosb_ = cosShape_.GetDim(DIM_0);
        s_ = qShape_.GetDim(DIM_1);
        qn_ = qShape_.GetDim(DIM_2);
        kn_ = kShape_.GetDim(DIM_2);
        d_ = qShape_.GetDim(DIM_3);
        reald_ = cosShape_.GetDim(DIM_3);
    } else if (layout_ == ApplyRotaryPosEmbLayout::BNSD) {
        b_ = qShape_.GetDim(DIM_0);
        cosb_ = cosShape_.GetDim(DIM_0);
        qn_ = qShape_.GetDim(DIM_1);
        kn_ = kShape_.GetDim(DIM_1);
        s_ = qShape_.GetDim(DIM_2);
        d_ = qShape_.GetDim(DIM_3);
        reald_ = cosShape_.GetDim(DIM_3);
    } else if (layout_ == ApplyRotaryPosEmbLayout::SBND) {
        s_ = qShape_.GetDim(DIM_0);
        b_ = qShape_.GetDim(DIM_1);
        cosb_ = cosShape_.GetDim(DIM_1);
        qn_ = qShape_.GetDim(DIM_2);
        kn_ = kShape_.GetDim(DIM_2);
        d_ = qShape_.GetDim(DIM_3);
        reald_ = cosShape_.GetDim(DIM_3);
    } else if (layout_ == ApplyRotaryPosEmbLayout::TND) {
        b_ = 1;
        cosb_ = 1;
        s_ = qShape_.GetDim(DIM_0);
        qn_ = qShape_.GetDim(DIM_1);
        kn_ = kShape_.GetDim(DIM_1);
        d_ = qShape_.GetDim(DIM_2);
        reald_ = cosShape_.GetDim(DIM_2);
    }
    isPartialRope_ = (reald_ != d_);
}

ge::graphStatus ApplyRotaryPosEmbRegbaseTilingBaseClass::GetShapeAttrsInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    if (!Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)) {
        return ge::GRAPH_SUCCESS;
    }
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const int64_t *layout = attrs->GetAttrPointer<int64_t>(0);
    int64_t layoutValue = (layout == nullptr) ? static_cast<int64_t>(ApplyRotaryPosEmbLayout::BSND) : (*layout);
    if (static_cast<int64_t>(layoutValue) < static_cast<int64_t>(ApplyRotaryPosEmbLayout::BSND) ||
        static_cast<int64_t>(layoutValue) > static_cast<int64_t>(ApplyRotaryPosEmbLayout::TND)) {
        std::string layoutValStr = std::to_string(static_cast<int64_t>(layoutValue));
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "layout",
            layoutValStr.c_str(), "1, 2, 3 or 4");
        return ge::GRAPH_FAILED;
    }
    layout_ = static_cast<ApplyRotaryPosEmbLayout>(layoutValue);
    const char *rotaryMode = attrs->GetAttrPointer<char>(1);
    rotaryModeStr_ = (rotaryMode == nullptr) ? "half" : rotaryMode;
    OP_CHECK_IF(CheckParam() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check param fail."), return ge::GRAPH_FAILED);
    dtype_ = qDataType_;
    ConvertRotaryMode();
    SetBsnd();
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling