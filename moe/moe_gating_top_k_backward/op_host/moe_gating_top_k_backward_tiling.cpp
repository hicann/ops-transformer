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
 * \file moe_gating_top_k_backward_tiling.cpp
 * \brief
 */

#include "platform/platform_info.h"
#include "moe_gating_top_k_backward_tiling.h"

namespace optiling {
// Attribute constants
const static int64_t NORM_TYPE_SIGMOID = 1;

// Tensor dimensions
const static size_t X_NORM_INPUT_DIMS = 2;     // [m,n]
const static size_t GRAD_Y_INPUT_DIMS = 2;     // [m,k]
const static size_t EXPERT_IDX_INPUT_DIMS = 2; // [m,k]
const static size_t GRAD_X_OUTPUT_DIMS = 2;    // [m,n]

// Indices
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

// UB related
const static int64_t DOUBLE_BUFFER_NUM = 2;
const static int64_t UB_ALIGN_BYTES_MINUS_ONE = 31;
const static int64_t UB_RESERVE_SPACE = 1024;
const static int64_t NUM_TWO = 2; // 2 buffers with the same size
const static int64_t NUM_SIX = 6; // 6 buffers with the same size

// Tiling keys
const static uint64_t TILING_KEY_RENORM_SIGMOID = 1;

// Data type sizes
const static int64_t SIZE_OF_FLOAT32 = 4;
const static int64_t NUM_BYTES_FOUR = 4;
const static int64_t NUM_BYTES_TWO = 2;

// Miscellaneous
const static int64_t MAX_EXPERT_COUNT = 2048;           // max value for n
const static int64_t MIN_EXPERT_COUNT = 2;              // min value for n
const static int64_t MIN_K = 1;                         // min value for k
const static int64_t DEFAULT_WORKSPACE_SIZE = 16777216; // 预留16M空间

// dim size
const static int64_t DIM_ZERO = 0;
const static int64_t DIM_ONE = 1;

class MoeGatingTopKBackwardTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit MoeGatingTopKBackwardTiling(gert::TilingContext *context)
        : Ops::Transformer::OpTiling::TilingBaseClass(context)
    {
    }
    ~MoeGatingTopKBackwardTiling() override = default;

protected:
    bool IsCapable() override
    {
        return true;
    }
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

private:
    ge::graphStatus CheckInputShape();
    ge::graphStatus CheckXNorm();
    ge::graphStatus CheckGradY();
    ge::graphStatus CheckExpertIdx();
    ge::graphStatus CheckAttr();
    ge::graphStatus CheckOutShape();
    ge::graphStatus CalcMaxRows();
    ge::graphStatus SplitRows();
    void DumpTiling();

    // Shapes
    const gert::Shape *xNormShape_ = nullptr;
    const gert::Shape *gradYShape_ = nullptr;
    const gert::Shape *expertIdxShape_ = nullptr;
    const gert::Shape *outputGradXShape_ = nullptr;

    // Miscellaneous
    int64_t inputDtypeSize_;
    ge::DataType gradYDtype_;

    // Tiling data
    MoeGatingTopKBackwardTilingData MoeGatingTopKBackwardTilingData_;

    // Tiling data parameters
    int64_t needCoreNum_ = 0;
    int64_t perCoreRows_ = 0;
    int64_t lastCoreRows_ = 0;
    int64_t baseRows_ = 0;
    int64_t perLoopTimes_ = 0;
    int64_t perTailRows_ = 0;
    int64_t lastLoopTimes_ = 0;
    int64_t lastTailRows_ = 0;
    int64_t tokenCount_ = 0;
    int64_t expertCount_ = 0;
    int64_t k_ = 0;
    int64_t gradYDtypeSize_ = 0;
    int64_t renorm_ = 0;
    int64_t normType_ = NORM_TYPE_SIGMOID;
    float routedScalingFactor_ = 1.0f;
    float eps_ = 1e-20f;
};

inline int64_t AlignBytes_(int64_t x)
{
    return (x + UB_ALIGN_BYTES_MINUS_ONE) & ~UB_ALIGN_BYTES_MINUS_ONE;
}

ge::graphStatus MoeGatingTopKBackwardTiling::CheckInputShape()
{
    OP_CHECK_IF(CheckXNorm() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Check XNorm failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckGradY() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Check GradY failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckExpertIdx() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Check ExpertIdx failed"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTiling::CheckXNorm()
{
    const gert::StorageShape *xNormShapePtr = context_->GetInputShape(X_NORM_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xNormShapePtr);
    auto xNormDimSize = xNormShapePtr->GetOriginShape().GetDimNum();
    OP_CHECK_IF(xNormDimSize != X_NORM_INPUT_DIMS,
                OP_LOGE(context_, "xNorm: Rank must be 2, but got %lu.", xNormDimSize), return ge::GRAPH_FAILED);

    tokenCount_ = xNormShapePtr->GetOriginShape().GetDim(DIM_ZERO);
    expertCount_ = xNormShapePtr->GetOriginShape().GetDim(DIM_ONE);
    OP_CHECK_IF(
        expertCount_ > MAX_EXPERT_COUNT,
        OP_LOGE(context_, "xNorm: Dimension 1 (N) must be less than or equal to 2048, but got %ld.", expertCount_),
        return ge::GRAPH_FAILED);

    auto xNormDesc = context_->GetInputDesc(X_NORM_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xNormDesc);
    auto xNormDtype = xNormDesc->GetDataType();
    OP_CHECK_IF(xNormDtype != ge::DataType::DT_FLOAT,
                OP_LOGE(context_, "xNorm: Unsupported data type. Expected %s, but got %s.",
                        ge::TypeUtils::DataTypeToSerialString(ge::DataType::DT_FLOAT).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(xNormDtype).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTiling::CheckGradY()
{
    const gert::StorageShape *gradYShapePtr = context_->GetInputShape(GRAD_Y_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradYShapePtr);
    auto gradYDimSize = gradYShapePtr->GetOriginShape().GetDimNum();
    OP_CHECK_IF(gradYDimSize != GRAD_Y_INPUT_DIMS,
                OP_LOGE(context_, "gradY: Rank must be 2, but got %lu.", gradYDimSize), return ge::GRAPH_FAILED);

    OP_CHECK_IF(gradYShapePtr->GetOriginShape().GetDim(DIM_ZERO) != tokenCount_,
                OP_LOGE(context_, "gradY: Must match xNorm on dimension 0, but got %ld vs %ld.",
                        gradYShapePtr->GetOriginShape().GetDim(DIM_ZERO), tokenCount_),
                return ge::GRAPH_FAILED);
    k_ = gradYShapePtr->GetOriginShape().GetDim(DIM_ONE);
    OP_CHECK_IF(k_ > expertCount_ || k_ <= 0,
                OP_LOGE(context_, "gradY: Dimension 1 (K) must be in the range of (0, %ld (N)], but got %ld.",
                        expertCount_, k_),
                return ge::GRAPH_FAILED);

    auto gradYDesc = context_->GetInputDesc(GRAD_Y_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradYDesc);
    gradYDtype_ = gradYDesc->GetDataType();
    OP_CHECK_IF(gradYDtype_ != ge::DataType::DT_FLOAT && gradYDtype_ != ge::DataType::DT_FLOAT16 &&
                    gradYDtype_ != ge::DataType::DT_BF16,
                OP_LOGE(context_, "gradY: Unsupported data type. Expected %s, %s or %s, but got %s.",
                        ge::TypeUtils::DataTypeToSerialString(ge::DataType::DT_FLOAT).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(ge::DataType::DT_FLOAT16).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(ge::DataType::DT_BF16).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(gradYDtype_).c_str()),
                return ge::GRAPH_FAILED);
    gradYDtypeSize_ = gradYDtype_ == ge::DataType::DT_FLOAT ? NUM_BYTES_FOUR : NUM_BYTES_TWO;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTiling::CheckExpertIdx()
{
    const gert::StorageShape *expertIdxShapePtr = context_->GetInputShape(EXPERT_IDX_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertIdxShapePtr);
    auto expertIdxDimSize = expertIdxShapePtr->GetOriginShape().GetDimNum();
    OP_CHECK_IF(expertIdxDimSize != EXPERT_IDX_INPUT_DIMS,
                OP_LOGE(context_, "expertIdx: Rank must be 2, but got %lu.", expertIdxDimSize),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(expertIdxShapePtr->GetOriginShape().GetDim(DIM_ZERO) != tokenCount_,
                OP_LOGE(context_, "expertIdx: Must match xNorm on dimension 0, but got %ld vs %ld.",
                        expertIdxShapePtr->GetOriginShape().GetDim(DIM_ZERO), tokenCount_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(expertIdxShapePtr->GetOriginShape().GetDim(DIM_ONE) != k_,
                OP_LOGE(context_, "expertIdx: Must match gradY on dimension 1, but got %ld vs %ld.",
                        expertIdxShapePtr->GetOriginShape().GetDim(DIM_ONE), k_),
                return ge::GRAPH_FAILED);

    auto expertIdxDesc = context_->GetInputDesc(EXPERT_IDX_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertIdxDesc);
    auto expertIdxDtype = expertIdxDesc->GetDataType();
    OP_CHECK_IF(expertIdxDtype != ge::DataType::DT_INT32,
                OP_LOGE(context_, "expertIdx: Unsupported data type. Expected %s, but got %s.",
                        ge::TypeUtils::DataTypeToSerialString(ge::DataType::DT_INT32).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(expertIdxDtype).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTiling::CheckAttr()
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

    OP_CHECK_IF(normType_ != NORM_TYPE_SIGMOID,
                OP_LOGE(context_, "Attr normType: expected %ld, but got %ld.", NORM_TYPE_SIGMOID, normType_),
                return ge::GRAPH_FAILED);

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

ge::graphStatus MoeGatingTopKBackwardTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<MoeGatingTopKBackwardCompileInfo>();
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compileInfoPtr is null."), return ge::GRAPH_FAILED);
        aicoreParams_.numBlocks = compileInfoPtr->numBlocks;
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aicoreParams_.numBlocks = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        aicoreParams_.ubSize = ubSizePlatForm;
        OP_CHECK_IF(aicoreParams_.numBlocks == 0, OP_LOGE(context_, "coreNum is 0"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(aicoreParams_.ubSize == 0, OP_LOGE(context_, "ubSize is 0"), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTiling::CheckOutShape()
{
    auto gradXShapePtr = context_->GetOutputShape(GRAD_X_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradXShapePtr);

    auto gradXDimSize = gradXShapePtr->GetOriginShape().GetDimNum();
    OP_CHECK_IF(gradXDimSize != GRAD_Y_INPUT_DIMS,
                OP_LOGE(context_, "gradX: Rank must be 2, but got %lu.", gradXDimSize), return ge::GRAPH_FAILED);

    OP_CHECK_IF(gradXShapePtr->GetOriginShape().GetDim(DIM_ZERO) != tokenCount_,
                OP_LOGE(context_, "gradX: Must match xNorm on dimension 0, but got %ld vs %ld.",
                        gradXShapePtr->GetOriginShape().GetDim(DIM_ZERO), tokenCount_),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(gradXShapePtr->GetOriginShape().GetDim(DIM_ONE) != expertCount_,
                OP_LOGE(context_, "gradX: Must match xNorm on dimension 1, but got %ld vs %ld.",
                        gradXShapePtr->GetOriginShape().GetDim(DIM_ONE), expertCount_),
                return ge::GRAPH_FAILED);

    auto gradXDesc = context_->GetOutputDesc(GRAD_X_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradXDesc);
    auto gradXDtype = gradXDesc->GetDataType();

    OP_CHECK_IF(gradXDtype != gradYDtype_,
                OP_LOGE(context_, "expertIdx: Mismatched data type. Expected %s, but got %s.",
                        ge::TypeUtils::DataTypeToSerialString(gradYDtype_).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(gradXDtype).c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTiling::CalcMaxRows()
{
    int64_t gradYQuePerTokenSpace = k_ * gradYDtypeSize_;
    int64_t indicesQuePerTokenSpace = k_ * SIZE_OF_FLOAT32;
    int64_t xQuePerTokenSpace = expertCount_ * SIZE_OF_FLOAT32;
    int64_t outQuePerTokenSpace = expertCount_ * gradYDtypeSize_;

    int64_t bufk0PerTokenSpace = k_ * SIZE_OF_FLOAT32;
    int64_t bufk1PerTokenSpace = k_ * SIZE_OF_FLOAT32;
    int64_t bufn2PerTokenSpace = expertCount_ * SIZE_OF_FLOAT32;
    int64_t bufn3PerTokenSpace = expertCount_ * SIZE_OF_FLOAT32;
    int64_t bufk4AddPerTokenSpace = k_ * SIZE_OF_FLOAT32;
    int64_t bufk4IndexPerTokenSpace = k_ * SIZE_OF_FLOAT32;
    int64_t bufk4RecipSumPerTokenSpace = k_ * SIZE_OF_FLOAT32;
    int64_t bufk4MaskPerTokenSpace = k_ * SIZE_OF_FLOAT32;
    int64_t bufsPerTokenSpace = k_ * SIZE_OF_FLOAT32;

    int64_t available_space = aicoreParams_.ubSize - UB_RESERVE_SPACE;
    int64_t quePerTokenSpace =
        DOUBLE_BUFFER_NUM * (gradYQuePerTokenSpace + indicesQuePerTokenSpace + xQuePerTokenSpace) + outQuePerTokenSpace;
    int64_t bufPerTokenSpace = bufk0PerTokenSpace + bufk1PerTokenSpace + bufn2PerTokenSpace + bufn3PerTokenSpace +
                               bufk4AddPerTokenSpace + bufk4IndexPerTokenSpace + bufk4RecipSumPerTokenSpace +
                               bufk4MaskPerTokenSpace + bufsPerTokenSpace;
    int64_t maxRows = available_space / (quePerTokenSpace + bufPerTokenSpace);
    int64_t kAligned = AlignBytes_(k_ * SIZE_OF_FLOAT32) >> NUM_TWO;
    // queSpace: gradYQueSpace; indicesQueSpace; xQueSpace; outQueSpace
    // bufSpace: bufk0Space = bufk1Space == bufk4AddSpace == bufk4IndexSpace == bufk4RecipSumSpace == bufk4MaskSpace;
    //           bufn2Space == bufn3Space; bufsSpace
    while (maxRows > 0) {
        int64_t queSpace = DOUBLE_BUFFER_NUM * maxRows * AlignBytes_(gradYQuePerTokenSpace) +
                           DOUBLE_BUFFER_NUM * maxRows * AlignBytes_(indicesQuePerTokenSpace) +
                           DOUBLE_BUFFER_NUM * AlignBytes_(maxRows * xQuePerTokenSpace) +
                           AlignBytes_(maxRows * outQuePerTokenSpace);
        int64_t bufSpace = NUM_SIX * maxRows * AlignBytes_(bufk1PerTokenSpace) +
                           NUM_TWO * AlignBytes_(maxRows * bufn2PerTokenSpace) +
                           maxRows * AlignBytes_(bufsPerTokenSpace);

        uint32_t reduceSumTmpBuffSpace = 0;
        uint32_t tmpMaxValue = 0;
        auto shape = ge::Shape({maxRows, kAligned});
        AscendC::GetReduceSumMaxMinTmpSize(shape, ge::DataType::DT_FLOAT, AscendC::ReducePattern::AR, true, false,
                                           tmpMaxValue, reduceSumTmpBuffSpace);

        auto srcShape = ge::Shape({1, kAligned});
        auto platformInfo = context_->GetPlatformInfo();
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        uint32_t broadcastTmpBuffSpace = 0;
        AscendC::GetBroadCastMaxMinTmpSize(ascendcPlatform, srcShape, shape, SIZE_OF_FLOAT32, false, tmpMaxValue,
                                           broadcastTmpBuffSpace);

        int64_t usedSpace = UB_RESERVE_SPACE + queSpace + bufSpace +
                            static_cast<int64_t>(std::max(reduceSumTmpBuffSpace, broadcastTmpBuffSpace));
        OP_LOGD(context_, "queSpace: %ld.\nbufSpace: %ld.\nsharebuf: %ld", queSpace, bufSpace,
                static_cast<int64_t>(std::max(reduceSumTmpBuffSpace, broadcastTmpBuffSpace)));

        if (usedSpace <= static_cast<int64_t>(aicoreParams_.ubSize)) {
            break;
        }
        maxRows--;
    }

    baseRows_ = maxRows;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTiling::SplitRows()
{
    perCoreRows_ = Ops::Base::CeilDiv(tokenCount_, static_cast<int64_t>(aicoreParams_.numBlocks));
    needCoreNum_ = Ops::Base::CeilDiv(tokenCount_, perCoreRows_);
    lastCoreRows_ = tokenCount_ % perCoreRows_ == 0 ? perCoreRows_ : tokenCount_ % perCoreRows_;

    auto ret = CalcMaxRows();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    OP_CHECK_IF(baseRows_ <= 0, OP_LOGE(context_, "baseRows must be > 0, but got %ld.", baseRows_),
                return ge::GRAPH_FAILED);

    perLoopTimes_ = (perCoreRows_ + baseRows_ - 1) / baseRows_;
    perTailRows_ = perCoreRows_ - (perLoopTimes_ - 1) * baseRows_;
    lastLoopTimes_ = (lastCoreRows_ + baseRows_ - 1) / baseRows_;
    lastTailRows_ = lastCoreRows_ - (lastLoopTimes_ - 1) * baseRows_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTiling::DoOpTiling()
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

    MoeGatingTopKBackwardTilingData_.set_needCoreNum(needCoreNum_);
    MoeGatingTopKBackwardTilingData_.set_perCoreRows(perCoreRows_);
    MoeGatingTopKBackwardTilingData_.set_lastCoreRows(lastCoreRows_);
    MoeGatingTopKBackwardTilingData_.set_baseRows(baseRows_);
    MoeGatingTopKBackwardTilingData_.set_perLoopTimes(perLoopTimes_);
    MoeGatingTopKBackwardTilingData_.set_perTailRows(perTailRows_);
    MoeGatingTopKBackwardTilingData_.set_lastLoopTimes(lastLoopTimes_);
    MoeGatingTopKBackwardTilingData_.set_lastTailRows(lastTailRows_);
    MoeGatingTopKBackwardTilingData_.set_tokenCount(tokenCount_);
    MoeGatingTopKBackwardTilingData_.set_expertCount(expertCount_);
    MoeGatingTopKBackwardTilingData_.set_k(k_);
    MoeGatingTopKBackwardTilingData_.set_gradYDtypeSize(gradYDtypeSize_);
    MoeGatingTopKBackwardTilingData_.set_renorm(renorm_);
    MoeGatingTopKBackwardTilingData_.set_normType(normType_);
    MoeGatingTopKBackwardTilingData_.set_routedScalingFactor(routedScalingFactor_);
    MoeGatingTopKBackwardTilingData_.set_eps(eps_);

    DumpTiling();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTiling::GetWorkspaceSize()
{
    workspaceSize_ = DEFAULT_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTiling::PostTiling()
{
    context_->SetBlockDim(MoeGatingTopKBackwardTilingData_.get_needCoreNum());
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    MoeGatingTopKBackwardTilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                                                  context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(MoeGatingTopKBackwardTilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeGatingTopKBackwardTiling::GetTilingKey() const
{
    return TILING_KEY_RENORM_SIGMOID;
}

void MoeGatingTopKBackwardTiling::DumpTiling()
{
    OP_LOGD(context_, "ubSize:  %ld", aicoreParams_.ubSize);
    OP_LOGD(context_, "numBlocks:  %ld", aicoreParams_.numBlocks);
    OP_LOGD(context_, "needCoreNum:  %ld", needCoreNum_);
    OP_LOGD(context_, "perCoreRows:  %ld", perCoreRows_);
    OP_LOGD(context_, "lastCoreRows:  %ld", lastCoreRows_);
    OP_LOGD(context_, "baseRows:  %ld", baseRows_);
    OP_LOGD(context_, "perLoopTimes:  %ld", perLoopTimes_);
    OP_LOGD(context_, "perTailRows:  %ld", perTailRows_);
    OP_LOGD(context_, "lastLoopTimes:  %ld", lastLoopTimes_);
    OP_LOGD(context_, "lastTailRows:  %ld", lastTailRows_);
    OP_LOGD(context_, "tokenCount:  %ld", tokenCount_);
    OP_LOGD(context_, "expertCount:  %ld", expertCount_);
    OP_LOGD(context_, "k:  %ld", k_);
    OP_LOGD(context_, "gradYDtypeSize:  %ld", gradYDtypeSize_);
    OP_LOGD(context_, "renorm:  %ld", renorm_);
    OP_LOGD(context_, "normType:  %ld", normType_);
    OP_LOGD(context_, "routedScalingFactor:  %f", routedScalingFactor_);
    OP_LOGD(context_, "eps:  %f", eps_);
    return;
}

REGISTER_OPS_TILING_TEMPLATE(MoeGatingTopKBackward, MoeGatingTopKBackwardTiling, 2000);
} // namespace optiling
