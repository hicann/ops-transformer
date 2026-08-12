/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_gating_top_k_backward_tiling_arch35.cpp
 * \brief
 */

#include "moe_gating_top_k_backward_tiling_arch35.h"
#include "op_host/tiling_util.h"

namespace optiling {

const static int64_t NORM_TYPE_SIGMOID = 1;
const static size_t X_NORM_INPUT_DIMS = 2;
const static size_t GRAD_Y_INPUT_DIMS = 2;
const static size_t EXPERT_IDX_INPUT_DIMS = 2;
const static size_t GRAD_X_OUTPUT_DIMS = 2;
const static int64_t X_NORM_INPUT_INDEX = 0;
const static int64_t GRAD_Y_INPUT_INDEX = 1;
const static int64_t EXPERT_IDX_INPUT_INDEX = 2;
const static int64_t GRAD_X_OUTPUT_INDEX = 0;
const static int64_t RENORM_ATTR_INDEX = 0;
const static int64_t NORM_TYPE_ATTR_INDEX = 1;
const static int64_t ROUTED_SCALING_FACTOR_ATTR_INDEX = 2;
const static int64_t EPS_ATTR_INDEX = 3;
const static int64_t M_DIM_INDEX = 0;
const static int64_t N_DIM_INDEX = 1;
const static int64_t K_DIM_INDEX = 1;
const static int64_t DOUBLE_BUFFER_NUM = 2;
const static int64_t UB_ALIGN_BYTES_MINUS_ONE = 31;
const static int64_t UB_RESERVE_SPACE = 1024;
const static int64_t NUM_TWO = 2;
const static int64_t MAX_EXPERT_COUNT = 2048;
const static int64_t SIZE_OF_FLOAT32 = 4;
const static int64_t SIZE_OF_INT32 = 4;
const static int64_t NUM_BYTES_FOUR = 4;
const static int64_t NUM_BYTES_TWO = 2;
const static int64_t DEFAULT_WORKSPACE_SIZE = 16777216;
const static uint64_t TILING_KEY_REGBASE = 10000;
const static int64_t DIM_ZERO = 0;
const static int64_t DIM_ONE = 1;

inline int64_t AlignBytes_(int64_t x)
{
    return (x + UB_ALIGN_BYTES_MINUS_ONE) & ~UB_ALIGN_BYTES_MINUS_ONE;
}

MoeGatingTopKBackwardTilingArch35::MoeGatingTopKBackwardTilingArch35(gert::TilingContext *context)
    : Ops::Transformer::OpTiling::TilingBaseClass(context)
{
}

bool MoeGatingTopKBackwardTilingArch35::IsCapable()
{
    if (!Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)) {
        return false;
    }
    return true;
}

ge::graphStatus MoeGatingTopKBackwardTilingArch35::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "platform_info"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    aicoreParams_.numBlocks = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    aicoreParams_.ubSize = ubSizePlatForm;
    OP_CHECK_IF(aicoreParams_.numBlocks == 0, OP_LOGE(context_, "coreNum is 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(aicoreParams_.ubSize == 0, OP_LOGE(context_, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTilingArch35::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTilingArch35::DoOpTiling()
{
    auto ret = CheckInputShape();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = CheckOutShape();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = CheckAttr();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = SplitRows();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    DumpTiling();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTilingArch35::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeGatingTopKBackwardTilingArch35::GetTilingKey() const
{
    return TILING_KEY_REGBASE;
}

ge::graphStatus MoeGatingTopKBackwardTilingArch35::GetWorkspaceSize()
{
    workspaceSize_ = DEFAULT_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTilingArch35::PostTiling()
{
    context_->SetBlockDim(static_cast<uint32_t>(needCoreNum_));
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;

    MoeGatingTopKBackwardA5TilingData td;
    td.needCoreNum = needCoreNum_;
    td.perCoreRows = perCoreRows_;
    td.baseRows = baseRows_;
    td.perLoopTimes = perLoopTimes_;
    td.perTailRows = perTailRows_;
    td.lastLoopTimes = lastLoopTimes_;
    td.lastTailRows = lastTailRows_;
    td.expertCount = expertCount_;
    td.k = k_;
    td.gradYDtypeSize = gradYDtypeSize_;
    td.renorm = renorm_;
    td.normType = normType_;
    td.routedScalingFactor = routedScalingFactor_;
    td.eps = eps_;

    auto tilingDataSize = sizeof(MoeGatingTopKBackwardA5TilingData);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(),
                           context_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void *>(&td), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTilingArch35::CheckInputShape()
{
    if (CheckXNorm() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckGradY() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckExpertIdx() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTilingArch35::CheckXNorm()
{
    const gert::StorageShape *xNormShapePtr = context_->GetInputShape(X_NORM_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xNormShapePtr);
    auto xNormDimSize = xNormShapePtr->GetOriginShape().GetDimNum();
    if (xNormDimSize != X_NORM_INPUT_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x_norm",
                                                 std::to_string(xNormDimSize) + "D", "The shape of x_norm must be 2D");
        return ge::GRAPH_FAILED;
    }

    tokenCount_ = xNormShapePtr->GetOriginShape().GetDim(DIM_ZERO);
    expertCount_ = xNormShapePtr->GetOriginShape().GetDim(DIM_ONE);
    if (tokenCount_ < 1) {
        std::string incorrectShape = "[" + std::to_string(tokenCount_) + ", " +
                                     std::to_string(expertCount_) + "]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x_norm",
                                              incorrectShape.c_str(), "x_norm cannot be an empty tensor");
        return ge::GRAPH_FAILED;
    }
    if (expertCount_ < 2 || expertCount_ > MAX_EXPERT_COUNT) {
        std::string incorrectShape = "[" + std::to_string(tokenCount_) + ", " +
                                     std::to_string(expertCount_) + "]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x_norm",
                                              incorrectShape.c_str(),
                                              "Shape [1] of this parameter must be within the range [2, 2048]");
        return ge::GRAPH_FAILED;
    }

    auto xNormDesc = context_->GetInputDesc(X_NORM_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xNormDesc);
    auto xNormDtype = xNormDesc->GetDataType();
    if (xNormDtype != ge::DataType::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "x_norm",
                                              ge::TypeUtils::DataTypeToSerialString(xNormDtype).c_str(),
                                              "The dtype of x_norm must be float32");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTilingArch35::CheckGradY()
{
    const gert::StorageShape *gradYShapePtr = context_->GetInputShape(GRAD_Y_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradYShapePtr);
    auto gradYDimSize = gradYShapePtr->GetOriginShape().GetDimNum();
    if (gradYDimSize != GRAD_Y_INPUT_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "grad_y",
                                                 std::to_string(gradYDimSize) + "D", "The shape of grad_y must be 2D");
        return ge::GRAPH_FAILED;
    }

    if (gradYShapePtr->GetOriginShape().GetDim(DIM_ZERO) != tokenCount_) {
        std::string incorrectShapes = "[" + std::to_string(gradYShapePtr->GetOriginShape().GetDim(DIM_ZERO)) +
                                      ", " + std::to_string(gradYShapePtr->GetOriginShape().GetDim(DIM_ONE)) + "] and [" +
                                      std::to_string(tokenCount_) + ", " + std::to_string(expertCount_) + "]";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "grad_y and x_norm",
                                               incorrectShapes.c_str(),
                                               "Shape [0] of grad_y must be equal to shape [0] of x_norm");
        return ge::GRAPH_FAILED;
    }
    k_ = gradYShapePtr->GetOriginShape().GetDim(DIM_ONE);
    if (k_ < 1 || k_ > expertCount_) {
        std::string incorrectShape = "[" + std::to_string(gradYShapePtr->GetOriginShape().GetDim(DIM_ZERO)) +
                                     ", " + std::to_string(k_) + "]";
        std::string reason = "Shape [1] of this parameter must be within the range [1, " +
                             std::to_string(expertCount_) + "]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "grad_y",
                                              incorrectShape.c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }

    auto gradYDesc = context_->GetInputDesc(GRAD_Y_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradYDesc);
    gradYDtype_ = gradYDesc->GetDataType();
    if (gradYDtype_ != ge::DataType::DT_FLOAT && gradYDtype_ != ge::DataType::DT_FLOAT16 &&
        gradYDtype_ != ge::DataType::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "grad_y",
                                              ge::TypeUtils::DataTypeToSerialString(gradYDtype_).c_str(),
                                              "The dtype of grad_y must be within the range [float32, float16, bfloat16]");
        return ge::GRAPH_FAILED;
    }
    gradYDtypeSize_ = gradYDtype_ == ge::DataType::DT_FLOAT ? NUM_BYTES_FOUR : NUM_BYTES_TWO;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTilingArch35::CheckExpertIdx()
{
    const gert::StorageShape *expertIdxShapePtr = context_->GetInputShape(EXPERT_IDX_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertIdxShapePtr);
    auto expertIdxDimSize = expertIdxShapePtr->GetOriginShape().GetDimNum();
    if (expertIdxDimSize != EXPERT_IDX_INPUT_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "expert_idx",
                                                 std::to_string(expertIdxDimSize) + "D", "The shape of expert_idx must be 2D");
        return ge::GRAPH_FAILED;
    }

    if (expertIdxShapePtr->GetOriginShape().GetDim(DIM_ZERO) != tokenCount_) {
        std::string incorrectShapes = "[" + std::to_string(expertIdxShapePtr->GetOriginShape().GetDim(DIM_ZERO)) +
                                      ", " + std::to_string(expertIdxShapePtr->GetOriginShape().GetDim(DIM_ONE)) + "] and [" +
                                      std::to_string(tokenCount_) + ", " + std::to_string(expertCount_) + "]";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "expert_idx and x_norm",
                                               incorrectShapes.c_str(),
                                               "Shape [0] of expert_idx must be equal to shape [0] of x_norm");
        return ge::GRAPH_FAILED;
    }
    if (expertIdxShapePtr->GetOriginShape().GetDim(DIM_ONE) != k_) {
        std::string incorrectShapes = "[" + std::to_string(expertIdxShapePtr->GetOriginShape().GetDim(DIM_ZERO)) +
                                      ", " + std::to_string(expertIdxShapePtr->GetOriginShape().GetDim(DIM_ONE)) + "] and [" +
                                      std::to_string(tokenCount_) + ", " + std::to_string(k_) + "]";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "expert_idx and grad_y",
                                               incorrectShapes.c_str(),
                                               "Shape [1] of expert_idx must be equal to shape [1] of grad_y");
        return ge::GRAPH_FAILED;
    }

    auto expertIdxDesc = context_->GetInputDesc(EXPERT_IDX_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertIdxDesc);
    auto expertIdxDtype = expertIdxDesc->GetDataType();
    if (expertIdxDtype != ge::DataType::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "expert_idx",
                                              ge::TypeUtils::DataTypeToSerialString(expertIdxDtype).c_str(),
                                              "The dtype of expert_idx must be int32");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTilingArch35::CheckAttr()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    const int64_t *renormPtr = attrs->GetAttrPointer<int64_t>(RENORM_ATTR_INDEX);
    if (renormPtr != nullptr) {
        renorm_ = *renormPtr;
    }
    OP_LOGI(context_, "Attr renorm: %ld.", renorm_);

    const int64_t *normTypePtr = attrs->GetAttrPointer<int64_t>(NORM_TYPE_ATTR_INDEX);
    if (normTypePtr != nullptr) {
        normType_ = *normTypePtr;
    }
    OP_LOGI(context_, "Attr normType: %ld.", normType_);
    if (normType_ != NORM_TYPE_SIGMOID) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "norm_type",
                                              std::to_string(normType_).c_str(),
                                              "The value of norm_type must be 1");
        return ge::GRAPH_FAILED;
    }

    const float *routedScalingFactorPtr = attrs->GetAttrPointer<float>(ROUTED_SCALING_FACTOR_ATTR_INDEX);
    if (routedScalingFactorPtr != nullptr) {
        routedScalingFactor_ = *routedScalingFactorPtr;
    }
    OP_LOGI(context_, "Attr routedScalingFactor: %f.", routedScalingFactor_);

    const float *epsPtr = attrs->GetAttrPointer<float>(EPS_ATTR_INDEX);
    if (epsPtr != nullptr) {
        eps_ = *epsPtr;
    }
    OP_LOGI(context_, "Attr eps: %f.", eps_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTilingArch35::CheckOutShape()
{
    auto gradXShapePtr = context_->GetOutputShape(GRAD_X_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradXShapePtr);

    auto gradXDimSize = gradXShapePtr->GetOriginShape().GetDimNum();
    if (gradXDimSize != GRAD_X_OUTPUT_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "grad_x",
                                                 std::to_string(gradXDimSize) + "D", "The shape of grad_x must be 2D");
        return ge::GRAPH_FAILED;
    }

    if (gradXShapePtr->GetOriginShape().GetDim(DIM_ZERO) != tokenCount_) {
        std::string incorrectShapes = "[" + std::to_string(gradXShapePtr->GetOriginShape().GetDim(DIM_ZERO)) +
                                      ", " + std::to_string(gradXShapePtr->GetOriginShape().GetDim(DIM_ONE)) + "] and [" +
                                      std::to_string(tokenCount_) + ", " + std::to_string(expertCount_) + "]";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "grad_x and x_norm",
                                               incorrectShapes.c_str(),
                                               "Shape [0] of grad_x must be equal to shape [0] of x_norm");
        return ge::GRAPH_FAILED;
    }
    if (gradXShapePtr->GetOriginShape().GetDim(DIM_ONE) != expertCount_) {
        std::string incorrectShapes = "[" + std::to_string(gradXShapePtr->GetOriginShape().GetDim(DIM_ZERO)) +
                                      ", " + std::to_string(gradXShapePtr->GetOriginShape().GetDim(DIM_ONE)) + "] and [" +
                                      std::to_string(tokenCount_) + ", " + std::to_string(expertCount_) + "]";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "grad_x and x_norm",
                                               incorrectShapes.c_str(),
                                               "Shape [1] of grad_x must be equal to shape [1] of x_norm");
        return ge::GRAPH_FAILED;
    }

    auto gradXDesc = context_->GetOutputDesc(GRAD_X_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradXDesc);
    auto gradXDtype = gradXDesc->GetDataType();
    if (gradXDtype != gradYDtype_) {
        std::string incorrectDtypes = std::string(ge::TypeUtils::DataTypeToSerialString(gradYDtype_)) +
                                      " and " + ge::TypeUtils::DataTypeToSerialString(gradXDtype);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "grad_x and grad_y",
                                               incorrectDtypes.c_str(),
                                               "The dtypes of grad_x and grad_y must be the same");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTilingArch35::CalcMaxRows()
{
    int64_t gradYQuePerTokenSpace = k_ * gradYDtypeSize_;
    int64_t indicesQuePerTokenSpace = k_ * SIZE_OF_INT32;
    int64_t xQuePerTokenSpace = expertCount_ * SIZE_OF_FLOAT32;
    int64_t outQuePerTokenSpace = expertCount_ * gradYDtypeSize_;
    int64_t wPrimeCachePerTokenSpace = k_ * SIZE_OF_FLOAT32;
    int64_t bufnPerTokenSpace = expertCount_ * SIZE_OF_FLOAT32;

    int64_t availableSpace = static_cast<int64_t>(aicoreParams_.ubSize) - UB_RESERVE_SPACE;
    int64_t quePerTokenSpace =
        DOUBLE_BUFFER_NUM * (gradYQuePerTokenSpace + indicesQuePerTokenSpace + xQuePerTokenSpace) + outQuePerTokenSpace;

    int64_t bufPerTokenSpace = bufnPerTokenSpace + wPrimeCachePerTokenSpace;
    int64_t maxRows = availableSpace / (quePerTokenSpace + bufPerTokenSpace);

    while (maxRows > 0) {
        int64_t queSpace =
            DOUBLE_BUFFER_NUM * maxRows * AlignBytes_(gradYQuePerTokenSpace) +
            DOUBLE_BUFFER_NUM * maxRows * AlignBytes_(indicesQuePerTokenSpace) +
            DOUBLE_BUFFER_NUM * AlignBytes_(maxRows * xQuePerTokenSpace) +
            AlignBytes_(maxRows * outQuePerTokenSpace);
        int64_t bufSpace = AlignBytes_(maxRows * bufnPerTokenSpace) +
                           maxRows * AlignBytes_(wPrimeCachePerTokenSpace);

        int64_t usedSpace = UB_RESERVE_SPACE + queSpace + bufSpace;
        if (usedSpace <= static_cast<int64_t>(aicoreParams_.ubSize)) {
            break;
        }
        maxRows--;
    }

    baseRows_ = maxRows;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTilingArch35::SplitRows()
{
    perCoreRows_ = Ops::Base::CeilDiv(tokenCount_, static_cast<int64_t>(aicoreParams_.numBlocks));
    needCoreNum_ = Ops::Base::CeilDiv(tokenCount_, perCoreRows_);
    lastCoreRows_ = tokenCount_ % perCoreRows_ == 0 ? perCoreRows_ : tokenCount_ % perCoreRows_;

    auto ret = CalcMaxRows();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    if (baseRows_ <= 0) {
        OP_LOGE(context_->GetNodeName(), "UB space is not enough to fit a single token row, "
                                         "baseRows must be greater than 0, but got %ld. Please reduce N or K.",
                baseRows_);
        return ge::GRAPH_FAILED;
    }

    perLoopTimes_ = (perCoreRows_ + baseRows_ - 1) / baseRows_;
    perTailRows_ = perCoreRows_ - (perLoopTimes_ - 1) * baseRows_;
    lastLoopTimes_ = (lastCoreRows_ + baseRows_ - 1) / baseRows_;
    lastTailRows_ = lastCoreRows_ - (lastLoopTimes_ - 1) * baseRows_;
    return ge::GRAPH_SUCCESS;
}

void MoeGatingTopKBackwardTilingArch35::DumpTiling()
{
    OP_LOGD(context_, "ubSize:  %ld", aicoreParams_.ubSize);
    OP_LOGD(context_, "numBlocks:  %ld", aicoreParams_.numBlocks);
    OP_LOGD(context_, "needCoreNum:  %ld", needCoreNum_);
    OP_LOGD(context_, "perCoreRows:  %ld", perCoreRows_);
    OP_LOGD(context_, "baseRows:  %ld", baseRows_);
    OP_LOGD(context_, "perLoopTimes:  %ld", perLoopTimes_);
    OP_LOGD(context_, "perTailRows:  %ld", perTailRows_);
    OP_LOGD(context_, "lastLoopTimes:  %ld", lastLoopTimes_);
    OP_LOGD(context_, "lastTailRows:  %ld", lastTailRows_);
    OP_LOGD(context_, "expertCount:  %ld", expertCount_);
    OP_LOGD(context_, "k:  %ld", k_);
    OP_LOGD(context_, "gradYDtypeSize:  %ld", gradYDtypeSize_);
    OP_LOGD(context_, "renorm:  %ld", renorm_);
    OP_LOGD(context_, "normType:  %ld", normType_);
    OP_LOGD(context_, "routedScalingFactor:  %f", routedScalingFactor_);
    OP_LOGD(context_, "eps:  %f", eps_);
}

REGISTER_OPS_TILING_TEMPLATE(MoeGatingTopKBackward, MoeGatingTopKBackwardTilingArch35, 1000);
} // namespace optiling
