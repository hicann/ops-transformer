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
 * \file allto_allv_quant_grouped_mat_mul_tiling_base.cpp
 * \brief
 */
#include "allto_allv_quant_grouped_mat_mul_tiling_base.h"
#include <numeric>

using namespace ge;
using namespace AscendC;
using namespace Ops::Transformer::OpTiling;
namespace optiling {
// protected
ge::graphStatus AlltoAllvQuantGmmTilingBase::GetCommonPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, OP_LOGE(context_->GetNodeName(), "can not get platform info."),
                    return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    socVersion_ = ascendcPlatform.GetSocVersion();
    aicCoreNum_ = ascendcPlatform.GetCoreNumAic();
    aivCoreNum_ = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0aSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0bSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size_);
    libApiWorkSpaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvQuantGmmTilingBase::CheckCommonPlatformInfo()
{
    OP_TILING_CHECK((aicCoreNum_ == 0U),
                    OP_LOGE(context_->GetNodeName(), "platform info is invalid, aic num can not be 0."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((aivCoreNum_ == 0U),
                    OP_LOGE(context_->GetNodeName(), "platform info is invalid, aiv num can not be 0."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((l0aSize_ == 0U),
                    OP_LOGE(context_->GetNodeName(), "platform info is invalid, l0a size can not be 0."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((l0bSize_ == 0U),
                    OP_LOGE(context_->GetNodeName(), "platform info is invalid, l0b size can not be 0."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((l0cSize_ == 0U),
                    OP_LOGE(context_->GetNodeName(), "platform info is invalid, l0c size can not be 0."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((l1Size_ == 0U),
                    OP_LOGE(context_->GetNodeName(), "platform info is invalid, l1 size can not be 0."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((ubSize_ == 0U),
                    OP_LOGE(context_->GetNodeName(), "platform info is invalid, ub size can not be 0."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvQuantGmmTilingBase::GetCommonShapeAttrsInfo()
{
    OP_LOGD(context_->GetNodeName(), "start GetCommonShapeAttrsInfo.");
    auto getAttrsInfoStatus = GetAttrsInfo();
    if (getAttrsInfoStatus != ge::GRAPH_SUCCESS) {
        return getAttrsInfoStatus;
    }
    // get shape info depends attr
    auto getShapeInfoStatus = GetShapeInfo();
    if (getShapeInfoStatus != ge::GRAPH_SUCCESS) {
        return getShapeInfoStatus;
    }
    OP_LOGD(context_->GetNodeName(), "end GetCommonShapeAttrsInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvQuantGmmTilingBase::CheckCommonShapeAttrsInfo()
{
    OP_LOGD(context_->GetNodeName(), "start CheckCommonShapeAttrsInfo.");
    auto checkAttrsInfoStatus = CheckAttrsInfo();
    if (checkAttrsInfoStatus != ge::GRAPH_SUCCESS) {
        return checkAttrsInfoStatus;
    }
    auto checkShapeInfoStatus = CheckShapeInfo();
    if (checkShapeInfoStatus != ge::GRAPH_SUCCESS) {
        return checkShapeInfoStatus;
    }
    auto checkFormatStatus = CheckFormat();
    if (checkFormatStatus != ge::GRAPH_SUCCESS) {
        return checkFormatStatus;
    }
    OP_LOGD(context_->GetNodeName(), "end CheckCommonShapeAttrsInfo.");
    return ge::GRAPH_SUCCESS;
}

// private

ge::graphStatus AlltoAllvQuantGmmTilingBase::GetAttrsInfo()
{
    OP_LOGD(context_->GetNodeName(), "start GetAttrsInfo.");
    auto attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "attrs"),
                    return ge::GRAPH_FAILED);
    // group
    groupPtr_ = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    OP_TILING_CHECK(groupPtr_ == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "group"),
                    return ge::GRAPH_FAILED);
    group_ = groupPtr_;
    // epWorldSize
    epWorldSizePtr_ = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    OP_TILING_CHECK(epWorldSizePtr_ == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "epWorldSize"),
                    return ge::GRAPH_FAILED);
    epWorldSize_ = *epWorldSizePtr_;
    // sendCounts
    sendCountsPtr_ = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    OP_TILING_CHECK(sendCountsPtr_ == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "sendCounts"),
                    return ge::GRAPH_FAILED);
    sendCounts = static_cast<const int64_t *>(sendCountsPtr_->GetData());
    // recvCounts
    recvCountsPtr_ = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(recvCountsPtr_ == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "recvCounts"),
                    return ge::GRAPH_FAILED);
    recvCounts = static_cast<const int64_t *>(recvCountsPtr_->GetData());
    // transGmmWeight
    transGmmWeightPtr_ = attrs->GetAttrPointer<bool>(ATTR_TRANS_GMM_WEIGHT_INDEX);
    if (transGmmWeightPtr_ != nullptr) {
        transGmmWeight_ = *transGmmWeightPtr_;
    }
    // transMmWeight
    transMmWeightPtr_ = attrs->GetAttrPointer<bool>(ATTR_TRANS_MM_WEIGHT_INDEX);
    if (transMmWeightPtr_ != nullptr) {
        transMmWeight_ = *transMmWeightPtr_;
    }
    // permuteOutFlag
    permuteOutFlagPtr_ = attrs->GetAttrPointer<bool>(ATTR_PERMUTE_OUT_FLAG_INDEX);
    if (permuteOutFlagPtr_ != nullptr) {
        permuteOutFlag_ = *permuteOutFlagPtr_;
    }
    gmmXQuantModePtr_ = attrs->GetAttrPointer<int64_t>(ATTR_GMM_X_QUANT_MODE_INDEX);
    gmmWeightQuantModePtr_ = attrs->GetAttrPointer<int64_t>(ATTR_GMM_WEIGHT_QUANT_MODE_INDEX);
    mmXQuantModePtr_ = attrs->GetAttrPointer<int64_t>(ATTR_MM_X_QUANT_MODE_INDEX);
    mmWeightQuantModePtr_ = attrs->GetAttrPointer<int64_t>(ATTR_MM_WEIGHT_QUANT_MODE_INDEX);
    OP_LOGD(context_->GetNodeName(), "end GetAttrsInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvQuantGmmTilingBase::CheckAttrsInfo()
{
    OP_LOGD(context_->GetNodeName(), "start CheckAttrsInfo.");
    if (CheckEpWorldSizeValue() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckCommCountsRange() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckCommCountsValue() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context_->GetNodeName(), "end CheckAttrsInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvQuantGmmTilingBase::CheckEpWorldSizeValue()
{
    // check epWorldSize in socVersion
    std::vector<int64_t> epWorldSizeValueList;
    std::string epWorldSizeValueStr = "";
    auto platformInfo = context_->GetPlatformInfo();
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    if (ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510) {
        epWorldSizeValueList = {2,  4,  8,   16,
                                32, 64, 128, 256}; // epWorldSize value only support 2, 4, 8, 16, 32, 64, 128, 256
    } else {
        epWorldSizeValueList = {8, 16, 32, 64, 128}; // epWorldSize value only support 8, 16, 32, 64, 128
    }
    for (size_t i = 0; i < epWorldSizeValueList.size(); i++) {
        if (i != 0) {
            epWorldSizeValueStr += ", ";
        }
        epWorldSizeValueStr += std::to_string(epWorldSizeValueList[i]);
    }
    OP_TILING_CHECK(
        std::find(epWorldSizeValueList.begin(), epWorldSizeValueList.end(), epWorldSize_) == epWorldSizeValueList.end(),
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "epWorldSize", std::to_string(epWorldSize_),
                                  "[" + epWorldSizeValueStr + "]"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvQuantGmmTilingBase::CheckCommCountsRange()
{
    // check sendCounts/recvCounts size
    uint64_t sendCountsSize = sendCountsPtr_->GetSize();
    uint64_t recvCountsSize = recvCountsPtr_->GetSize();
    if (e_ > E_MIN_VALUE && e_ <= E_MAX_VALUE) {
        OP_TILING_CHECK(sendCountsSize != recvCountsSize,
                        OP_LOGE_FOR_INVALID_VALUE(
                            context_->GetNodeName(), "sendCounts and recvCounts",
                            (std::to_string(sendCountsSize) + " and " + std::to_string(recvCountsSize)).c_str(),
                            "sendCounts size should equal recvCounts size"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(
            e_ * epWorldSize_ != sendCountsSize,
            OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "sendCounts", std::to_string(sendCountsSize).c_str(),
                                      "sendCounts size should equal e * epWorldSize"),
            return ge::GRAPH_FAILED);
        OP_TILING_CHECK(
            (e_ * epWorldSize_ <= EXPERT_MIN_VALUE) || (e_ * epWorldSize_ > EXPERT_MAX_VALUE),
            OP_LOGE_FOR_INVALID_VALUE(
                context_->GetNodeName(), "e * epWorldSize", std::to_string(e_ * epWorldSize_).c_str(),
                "should be in (" + std::to_string(EXPERT_MIN_VALUE) + ", " + std::to_string(EXPERT_MAX_VALUE) + "]"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvQuantGmmTilingBase::CheckCommCountsValue()
{
    // check sendCounts range
    for (uint64_t idx = 0U; idx < e_ * epWorldSize_; idx++) {
        OP_TILING_CHECK((sendCounts[idx] < SEND_COUNTS_MIN_VALUE) || (sendCounts[idx] > bsk_),
                        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "sendCounts",
                                                  std::to_string(sendCounts[idx]), "[0, " + std::to_string(bsk_) + "]"),
                        return ge::GRAPH_FAILED);
    }

    // check recvCounts range
    for (uint64_t i = 0U; i < e_ * epWorldSize_; i++) {
        OP_TILING_CHECK((recvCounts[i] < DIM_ZERO) || (recvCounts[i] > a_),
                        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "recvCounts", std::to_string(recvCounts[i]),
                                                  "[0, " + std::to_string(a_) + "]"),
                        return ge::GRAPH_FAILED);
    }

    uint64_t sendCountsSize = sendCountsPtr_->GetSize();
    uint64_t recvCountsSize = recvCountsPtr_->GetSize();
    // check sum(sendCounts) = BSK
    uint64_t sendCountsSum = std::accumulate(sendCounts, sendCounts + sendCountsSize, 0ULL);
    OP_TILING_CHECK(sendCountsSum != bsk_,
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "sum(sendCounts)", std::to_string(sendCountsSum),
                                              std::to_string(bsk_)),
                    return ge::GRAPH_FAILED);

    // check sum(recvCounts) = A
    uint64_t recvCountsSum = std::accumulate(recvCounts, recvCounts + recvCountsSize, 0ULL);
    OP_TILING_CHECK(recvCountsSum != a_,
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "sum(recvCounts)", std::to_string(recvCountsSum),
                                              std::to_string(a_)),
                    return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "end CheckAttrsInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvQuantGmmTilingBase::GetShapeInfo()
{
    OP_LOGD(context_->GetNodeName(), "start GetShapeInfo.");
    // Get GMM input X shape info
    if (GetGmmXShapeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetGmmWeightShapeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // Get counts tensor shape info
    if (GetCountsTensorShapeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // Get MM related shape info
    if (GetMmxShapeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetMmWeightShapeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // Get output shape info
    if (GetGmmYShapeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetMmYShapeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetPermuteOutShapeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context_->GetNodeName(), "end GetShapeInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvQuantGmmTilingBase::CheckShapeInfo()
{
    OP_LOGD(context_->GetNodeName(), "start CheckShapeInfo.");
    // Check GMM input shape info
    if (CheckGmmXShapeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckGmmWeightShapeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // Check counts tensor shape info
    if (CheckCountsTensorShapeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // Check MM related shape info
    if (CheckMmxShapeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckMmWeightShapeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // Check output shape info
    if (CheckGmmYShapeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckMmYShapeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckPermuteOutShapeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context_->GetNodeName(), "end CheckShapeInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvQuantGmmTilingBase::GetGmmXShapeInfo()
{
    OP_LOGD(context_->GetNodeName(), "start GetGmmXShapeInfo.");
    // check gmmX not null
    OP_TILING_CHECK(context_->GetInputShape(GMM_X_INDEX) == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "gmmX"), return ge::GRAPH_FAILED);
    bsk_ = context_->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
    h1_ = context_->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDim(DIM_ONE);
    OP_TILING_CHECK(context_->GetInputDesc(GMM_X_INDEX) == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "gmmX"), return ge::GRAPH_FAILED);
    gmmXDataType_ = context_->GetInputDesc(GMM_X_INDEX)->GetDataType();
    OP_LOGD(context_->GetNodeName(), "end GetGmmXShapeInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvQuantGmmTilingBase::CheckGmmXShapeInfo()
{
    OP_LOGD(context_->GetNodeName(), "start CheckGmmXShapeInfo.");
    // check dim = 2
    if (context_->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDimNum() != DIM_TWO) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            context_->GetNodeName(), "gmmX",
            std::to_string(context_->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDimNum()), "2");
        return ge::GRAPH_FAILED;
    }
    // check bsk range
    OP_TILING_CHECK(
        bsk_ <= BSK_MIN_VALUE || bsk_ >= BSK_MAX_VALUE,
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "gmmX dim0(BSK)", std::to_string(bsk_),
                                  "(" + std::to_string(BSK_MIN_VALUE) + ", " + std::to_string(BSK_MAX_VALUE) + ")"),
        return ge::GRAPH_FAILED);
    // check h1 range
    OP_TILING_CHECK(
        (h1_ <= H1_MIN_VALUE) || (h1_ >= H1_MAX_VALUE),
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "gmmX dim1(H1)", std::to_string(h1_),
                                  "(" + std::to_string(H1_MIN_VALUE) + ", " + std::to_string(H1_MAX_VALUE) + ")"),
        return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "end CheckGmmXShapeInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvQuantGmmTilingBase::GetGmmWeightShapeInfo()
{
    OP_LOGD(context_->GetNodeName(), "start GetGmmWeightShapeInfo.");
    OP_TILING_CHECK(context_->GetInputShape(GMM_WEIGHT_INDEX) == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "gmmWeight"), return ge::GRAPH_FAILED);
    e_ = context_->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t n1DimIndex = transGmmWeight_ ? DIM_ONE : DIM_TWO;
    n1_ = context_->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDim(n1DimIndex);
    OP_TILING_CHECK(context_->GetInputDesc(GMM_WEIGHT_INDEX) == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "gmmWeight"), return ge::GRAPH_FAILED);
    gmmWeightDataType_ = context_->GetInputDesc(GMM_WEIGHT_INDEX)->GetDataType();
    OP_LOGD(context_->GetNodeName(), "end GetGmmWeightShapeInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvQuantGmmTilingBase::CheckGmmWeightShapeInfo()
{
    OP_LOGD(context_->GetNodeName(), "start CheckGmmWeightShapeInfo.");
    // check dim
    if (context_->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDimNum() != DIM_THREE) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            context_->GetNodeName(), "gmmWeight",
            std::to_string(context_->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDimNum()), "3");
        return ge::GRAPH_FAILED;
    }
    // check H1 equal
    uint64_t gmmWeightH1 = transGmmWeight_ ?
                               context_->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDim(DIM_TWO) :
                               context_->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDim(DIM_ONE);
    OP_TILING_CHECK(h1_ != gmmWeightH1,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context_->GetNodeName(), "gmmX H1", std::to_string(h1_).c_str(),
                        ("should equal gmmWeight H1(" + std::to_string(gmmWeightH1) + ")").c_str()),
                    return ge::GRAPH_FAILED);
    // check e range
    OP_TILING_CHECK(
        (e_ <= E_MIN_VALUE) || (e_ > E_MAX_VALUE),
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "gmmWeight dim0(e)", std::to_string(e_),
                                  "(" + std::to_string(E_MIN_VALUE) + ", " + std::to_string(E_MAX_VALUE) + "]"),
        return ge::GRAPH_FAILED);
    // check N1 range
    OP_TILING_CHECK(
        n1_ <= N1_MIN_VALUE || n1_ >= N1_MAX_VALUE,
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "gmmWeight dim(N1)", std::to_string(n1_),
                                  "(" + std::to_string(N1_MIN_VALUE) + ", " + std::to_string(N1_MAX_VALUE) + ")"),
        return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "end CheckGmmWeightShapeInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvQuantGmmTilingBase::GetCountsTensorShapeInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvQuantGmmTilingBase::CheckCountsTensorShapeInfo()
{
    OP_LOGD(context_->GetNodeName(), "start CheckCountsTensorShapeInfo.");
    // sendCountsTensor only support nullptr
    OP_TILING_CHECK(
        context_->GetOptionalInputShape(SEND_COUNTS_TENSOR_INDEX) != nullptr,
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "sendCountsTensor", "non-null", "should be null"),
        return ge::GRAPH_FAILED);
    // recvCountsTensor only support nullptr
    OP_TILING_CHECK(
        context_->GetOptionalInputShape(RECV_COUNTS_TENSOR_INDEX) != nullptr,
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "recvCountsTensor", "non-null", "should be null"),
        return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "end CheckCountsTensorShapeInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvQuantGmmTilingBase::GetMmxShapeInfo()
{
    OP_LOGD(context_->GetNodeName(), "start GetMmxShapeInfo.");
    // optional input mmX
    if (context_->GetOptionalInputShape(MM_X_INDEX) != nullptr) {
        hasSharedExpertFlag_ = true;
        bs_ = context_->GetOptionalInputShape(MM_X_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
        OP_TILING_CHECK(context_->GetOptionalInputDesc(MM_X_INDEX) == nullptr,
                        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "mmX"), return ge::GRAPH_FAILED);
        mmXDataType_ = context_->GetOptionalInputDesc(MM_X_INDEX)->GetDataType();
        h2_ = context_->GetOptionalInputShape(MM_X_INDEX)->GetStorageShape().GetDim(DIM_ONE);
    }
    OP_LOGD(context_->GetNodeName(), "end GetMmxShapeInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvQuantGmmTilingBase::CheckMmxShapeInfo()
{
    OP_LOGD(context_->GetNodeName(), "start CheckMmxShapeInfo.");
    if (!hasSharedExpertFlag_) {
        OP_LOGD(context_->GetNodeName(), "end CheckMmxShapeInfo.");
        return ge::GRAPH_SUCCESS;
    }
    // check dim
    if (context_->GetOptionalInputShape(MM_X_INDEX)->GetStorageShape().GetDimNum() != DIM_TWO) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            context_->GetNodeName(), "mmX",
            std::to_string(context_->GetOptionalInputShape(MM_X_INDEX)->GetStorageShape().GetDimNum()), "2");
        return ge::GRAPH_FAILED;
    }
    // check BS range
    OP_TILING_CHECK(bs_ <= BS_MIN_VALUE,
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "mmX dim0(BS)", std::to_string(bs_),
                                              "> " + std::to_string(BS_MIN_VALUE)),
                    return ge::GRAPH_FAILED);
    // check BSK divisible by BS
    OP_TILING_CHECK(
        (bsk_ % bs_ != 0),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "BSK", std::to_string(bsk_).c_str(),
                                              ("should be divisible by BS(" + std::to_string(bs_) + ")").c_str()),
        return ge::GRAPH_FAILED);
    k_ = bsk_ / bs_;
    // check K range
    OP_TILING_CHECK(
        (k_ < K_MIN_VALUE) || (k_ > K_MAX_VALUE),
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "K(=BSK/BS)", std::to_string(k_),
                                  "(" + std::to_string(K_MIN_VALUE) + ", " + std::to_string(K_MAX_VALUE) + ")"),
        return ge::GRAPH_FAILED);
    // check H2 range
    OP_TILING_CHECK(
        (h2_ <= H2_MIN_VALUE) || (h2_ > H2_MAX_VALUE),
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "mmX dim1(H2)", std::to_string(h2_),
                                  "(" + std::to_string(H2_MIN_VALUE) + ", " + std::to_string(H2_MAX_VALUE) + ")"),
        return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "end CheckMmxShapeInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvQuantGmmTilingBase::GetMmWeightShapeInfo()
{
    OP_LOGD(context_->GetNodeName(), "start GetMmWeightShapeInfo.");
    if (context_->GetOptionalInputShape(MM_WEIGHT_INDEX) != nullptr) {
        n2_ = transMmWeight_ ? context_->GetOptionalInputShape(MM_WEIGHT_INDEX)->GetStorageShape().GetDim(DIM_ZERO) :
                               context_->GetOptionalInputShape(MM_WEIGHT_INDEX)->GetStorageShape().GetDim(DIM_ONE);
        OP_TILING_CHECK(
            n2_ <= N2_MIN_VALUE || n2_ >= N2_MAX_VALUE,
            OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "mmWeight dim(N2)", std::to_string(n2_),
                                      "(" + std::to_string(N2_MIN_VALUE) + ", " + std::to_string(N2_MAX_VALUE) + ")"),
            return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(transMmWeight_ == true,
                        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "transMmWeight", "true",
                                                  "should be false when mmWeight is null"),
                        return ge::GRAPH_FAILED);
    }
    OP_LOGD(context_->GetNodeName(), "end GetMmWeightShapeInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvQuantGmmTilingBase::CheckMmWeightShapeInfo()
{
    OP_LOGD(context_->GetNodeName(), "start CheckMmWeightShapeInfo.");
    if (context_->GetOptionalInputShape(MM_WEIGHT_INDEX) == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    // check h2 equal
    uint64_t mmWeightH2 = transMmWeight_ ?
                              context_->GetOptionalInputShape(MM_WEIGHT_INDEX)->GetStorageShape().GetDim(DIM_ONE) :
                              context_->GetOptionalInputShape(MM_WEIGHT_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
    OP_TILING_CHECK(
        h2_ != mmWeightH2,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "mmX H2", std::to_string(h2_).c_str(),
                                              ("should equal mmWeight H2(" + std::to_string(mmWeightH2) + ")").c_str()),
        return ge::GRAPH_FAILED);
    // check dim
    if (context_->GetOptionalInputShape(MM_WEIGHT_INDEX)->GetStorageShape().GetDimNum() != DIM_TWO) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            context_->GetNodeName(), "mmWeight",
            std::to_string(context_->GetOptionalInputShape(MM_WEIGHT_INDEX)->GetStorageShape().GetDimNum()), "2");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context_->GetNodeName(), "end CheckMmWeightShapeInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvQuantGmmTilingBase::GetGmmYShapeInfo()
{
    OP_LOGD(context_->GetNodeName(), "start GetGmmYShapeInfo.");
    // output gmmY
    OP_TILING_CHECK(context_->GetOutputShape(OUTPUT_GMM_Y_INDEX) == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "gmmY"), return ge::GRAPH_FAILED);
    a_ = context_->GetOutputShape(OUTPUT_GMM_Y_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
    OP_LOGD(context_->GetNodeName(), "end GetGmmYShapeInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvQuantGmmTilingBase::CheckGmmYShapeInfo()
{
    OP_LOGD(context_->GetNodeName(), "start CheckGmmYShapeInfo.");
    // check dim
    if (context_->GetOutputShape(OUTPUT_GMM_Y_INDEX)->GetStorageShape().GetDimNum() != DIM_TWO) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            context_->GetNodeName(), "gmmY",
            std::to_string(context_->GetOutputShape(OUTPUT_GMM_Y_INDEX)->GetStorageShape().GetDimNum()), "2");
        return ge::GRAPH_FAILED;
    }
    uint64_t gmmYdim2 = context_->GetOutputShape(OUTPUT_GMM_Y_INDEX)->GetStorageShape().GetDim(DIM_ONE);
    OP_TILING_CHECK(
        gmmYdim2 != n1_,
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "gmmY dim1", std::to_string(gmmYdim2), std::to_string(n1_)),
        return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "end CheckGmmYShapeInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvQuantGmmTilingBase::GetMmYShapeInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvQuantGmmTilingBase::CheckMmYShapeInfo()
{
    OP_LOGD(context_->GetNodeName(), "start CheckMmYShapeInfo.");
    // check mmX, mmWeight and mmY all be nullptr or all be not nullptr
    const gert::StorageShape *mmXStorageShape = context_->GetOptionalInputShape(MM_X_INDEX);
    const gert::StorageShape *mmWeightStorageShape = context_->GetOptionalInputShape(MM_WEIGHT_INDEX);
    const gert::StorageShape *outputMmYStorageShape = context_->GetOutputShape(OUTPUT_MM_Y_INDEX);
    if (!((mmXStorageShape == nullptr) && (mmWeightStorageShape == nullptr) &&
          (outputMmYStorageShape == nullptr || outputMmYStorageShape->GetStorageShape().GetDimNum() == DIM_ZERO)) &&
        !((mmXStorageShape != nullptr) && (mmWeightStorageShape != nullptr) &&
          (outputMmYStorageShape != nullptr && outputMmYStorageShape->GetStorageShape().GetDimNum() != DIM_ZERO))) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "mmX/mmWeight/mmY", "inconsistent",
                                              "The values of mmX/mmWeight/mmY must be the same.");
        return ge::GRAPH_FAILED;
    }
    if (context_->GetOutputShape(OUTPUT_MM_Y_INDEX) != nullptr &&
        outputMmYStorageShape->GetStorageShape().GetDimNum() != DIM_ZERO) {
        // check dim
        if (context_->GetOutputShape(OUTPUT_MM_Y_INDEX)->GetStorageShape().GetDimNum() != DIM_TWO) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(
                context_->GetNodeName(), "mmY",
                std::to_string(context_->GetOutputShape(OUTPUT_MM_Y_INDEX)->GetStorageShape().GetDimNum()), "2");
            return ge::GRAPH_FAILED;
        }
        // check BS equal
        uint64_t mmYBS = context_->GetOutputShape(OUTPUT_MM_Y_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
        OP_TILING_CHECK(
            bs_ != mmYBS,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "mmY BS", std::to_string(bs_).c_str(),
                                                  ("should equal mmX BS(" + std::to_string(mmYBS) + ")").c_str()),
            return ge::GRAPH_FAILED);
        uint64_t mmYdim2 = context_->GetOutputShape(OUTPUT_MM_Y_INDEX)->GetStorageShape().GetDim(DIM_ONE);
        OP_TILING_CHECK(mmYdim2 != n2_,
                        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "mmY dim1", std::to_string(mmYdim2),
                                                  std::to_string(n2_)),
                        return ge::GRAPH_FAILED);
    }
    OP_LOGD(context_->GetNodeName(), "end CheckMmYShapeInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvQuantGmmTilingBase::GetPermuteOutShapeInfo()
{
    OP_LOGD(context_->GetNodeName(), "start GetPermuteOutShapeInfo.");
    if (!permuteOutFlag_) {
        OP_LOGD(context_->GetNodeName(), "end GetPermuteOutShapeInfo.");
        return ge::GRAPH_SUCCESS;
    }
    OP_TILING_CHECK(context_->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX) == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "permuteOut"), return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "end GetPermuteOutShapeInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvQuantGmmTilingBase::CheckPermuteOutShapeInfo()
{
    OP_LOGD(context_->GetNodeName(), "start CheckPermuteOutShapeInfo.");
    if (!permuteOutFlag_) {
        if (context_->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX) != nullptr &&
            context_->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX)->GetStorageShape().GetDimNum() != DIM_ZERO) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "permuteOut", "present",
                                                  "The value of permuteOut must be nullptr.");
            return ge::GRAPH_FAILED;
        }
        OP_LOGD(context_->GetNodeName(), "end CheckPermuteOutShapeInfo.");
        return ge::GRAPH_SUCCESS;
    }
    // check not null
    if (context_->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX) == nullptr ||
        context_->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX)->GetStorageShape().GetDimNum() == DIM_ZERO) {
        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "permuteOut");
        return ge::GRAPH_FAILED;
    }
    // check dim
    if (context_->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX)->GetStorageShape().GetDimNum() != DIM_TWO) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            context_->GetNodeName(), "permuteOut",
            std::to_string(context_->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX)->GetStorageShape().GetDimNum()), "2");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context_->GetNodeName(), "end CheckPermuteOutShapeInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvQuantGmmTilingBase::CheckFormat()
{
    OP_LOGD(context_->GetNodeName(), "start CheckFormat.");
    auto gmmXDesc = context_->GetInputDesc(GMM_X_INDEX);
    OP_TILING_CHECK(gmmXDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "gmmX"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(gmmXDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "gmmX",
                                               Ops::Base::ToString(gmmXDesc->GetStorageFormat()), "ND"),
                    return ge::GRAPH_FAILED);
    auto gmmWeightDesc = context_->GetInputDesc(GMM_WEIGHT_INDEX);
    OP_TILING_CHECK(gmmWeightDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "gmmWeight"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(gmmWeightDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "gmmWeight",
                                               Ops::Base::ToString(gmmWeightDesc->GetStorageFormat()), "ND"),
                    return ge::GRAPH_FAILED);
    auto mmXDesc = context_->GetOptionalInputDesc(MM_X_INDEX);
    if (mmXDesc != nullptr) {
        OP_TILING_CHECK(mmXDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
                        OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "mmX",
                                                   Ops::Base::ToString(mmXDesc->GetStorageFormat()), "ND"),
                        return ge::GRAPH_FAILED);
    }
    auto mmWeightDesc = context_->GetOptionalInputDesc(MM_WEIGHT_INDEX);
    if (mmWeightDesc != nullptr) {
        OP_TILING_CHECK(mmWeightDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
                        OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "mmWeight",
                                                   Ops::Base::ToString(mmWeightDesc->GetStorageFormat()), "ND"),
                        return ge::GRAPH_FAILED);
    }
    auto gmmYDesc = context_->GetOutputDesc(OUTPUT_GMM_Y_INDEX);
    OP_TILING_CHECK(gmmYDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "gmmY"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(gmmYDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "gmmY",
                                               Ops::Base::ToString(gmmYDesc->GetStorageFormat()), "ND"),
                    return ge::GRAPH_FAILED);
    if (hasSharedExpertFlag_) {
        auto mmYDesc = context_->GetOutputDesc(OUTPUT_MM_Y_INDEX);
        if (mmYDesc != nullptr) {
            OP_TILING_CHECK(mmYDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
                            OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "mmY",
                                                       Ops::Base::ToString(mmYDesc->GetStorageFormat()), "ND"),
                            return ge::GRAPH_FAILED);
        }
    }
    auto permuteOutDesc = context_->GetOutputDesc(OUTPUT_PERMUTE_OUT_INDEX);
    if (permuteOutFlag_) {
        if (permuteOutDesc != nullptr) {
            OP_TILING_CHECK(permuteOutDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
                            OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "permuteOut",
                                                       Ops::Base::ToString(permuteOutDesc->GetStorageFormat()), "ND"),
                            return ge::GRAPH_FAILED);
        }
    }
    OP_LOGD(context_->GetNodeName(), "end CheckFormat.");
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
