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
 * \file masked_causal_conv1d_backward_arch35.cpp
 * \brief MaskedCausalConv1dBackward tiling implementation
 */

#include "masked_causal_conv1d_backward_tiling_arch35.h"
#include <algorithm>
#include "securec.h"

namespace optiling {

using MaskedCausalConv1dBackwardArch35Tiling::MaskedCausalConv1dBackwardTilingDataV35;

bool MaskedCausalConv1dBackwardTilingArch35::IsCapable()
{
    return true;
}

ge::graphStatus MaskedCausalConv1dBackwardTilingArch35::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr =
            reinterpret_cast<const MaskedCausalConv1dBackwardArch35CompileInfo *>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
        totalCoreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        totalCoreNum_ = static_cast<uint64_t>(ascendcPlatform.GetCoreNumAiv());
        OP_CHECK_IF(totalCoreNum_ == 0UL, OP_LOGE(context_->GetNodeName(), "coreNum is 0"), return ge::GRAPH_FAILED);
        uint64_t ubSize = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        OP_CHECK_IF(ubSize == static_cast<uint64_t>(0), OP_LOGE(context_->GetNodeName(), "ubSize is 0"),
                    return ge::GRAPH_FAILED);
        ubSize_ = static_cast<uint64_t>(ubSize);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedCausalConv1dBackwardTilingArch35::ValidateGradYShape()
{
    auto shape = context_->GetInputShape(GRAD_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shape);
    auto origin = shape->GetOriginShape();
    if (origin.GetDimNum() != NUM_3) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "grad_y",
                                                 std::to_string(origin.GetDimNum()) + "D",
                                                 "The shape of grad_y must be 3D");
        return ge::GRAPH_FAILED;
    }

    S_ = static_cast<int64_t>(origin.GetDim(DIM_0));
    B_ = static_cast<int64_t>(origin.GetDim(DIM_1));
    H_ = static_cast<int64_t>(origin.GetDim(DIM_2));

    if (B_ <= 0 || B_ > B_MAX) {
        std::string incorrectShape =
            "[" + std::to_string(S_) + ", " + std::to_string(B_) + ", " + std::to_string(H_) + "]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "grad_y", incorrectShape.c_str(),
                                              "Shape [1] of this parameter must be within the range [1, 32]");
        return ge::GRAPH_FAILED;
    }
    if (B_ * S_ > BS_MAX) {
        std::string incorrectShape =
            "[" + std::to_string(S_) + ", " + std::to_string(B_) + ", " + std::to_string(H_) + "]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "grad_y", incorrectShape.c_str(),
                                              "The following constraint must be met: shape [0] * shape [1] <= 524288");
        return ge::GRAPH_FAILED;
    }
    if (H_ < H_MIN || H_ > H_MAX) {
        std::string incorrectShape =
            "[" + std::to_string(S_) + ", " + std::to_string(B_) + ", " + std::to_string(H_) + "]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "grad_y", incorrectShape.c_str(),
                                              "Shape [2] of this parameter must be within the range [384, 24576]");
        return ge::GRAPH_FAILED;
    }

    // H 必须按 64 对齐以便 H 方向切分
    if ((H_ % DIM_ALIGN_ELEMENT) != 0) {
        std::string incorrectShape =
            "[" + std::to_string(S_) + ", " + std::to_string(B_) + ", " + std::to_string(H_) + "]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "grad_y", incorrectShape.c_str(),
                                              "Shape [2] of this parameter must be exactly divided by hReg (64)");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedCausalConv1dBackwardTilingArch35::ValidateXShape()
{
    auto goShape = context_->GetInputShape(GRAD_Y_INDEX)->GetOriginShape();

    auto inShape = context_->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inShape);
    auto inOrigin = inShape->GetOriginShape();
    if (inOrigin.GetDimNum() != NUM_3) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context_->GetNodeName(), "x", std::to_string(inOrigin.GetDimNum()) + "D", "The shape of x must be 3D");
        return ge::GRAPH_FAILED;
    }

    if (goShape.GetDim(DIM_0) != inOrigin.GetDim(DIM_0) || goShape.GetDim(DIM_1) != inOrigin.GetDim(DIM_1) ||
        goShape.GetDim(DIM_2) != inOrigin.GetDim(DIM_2)) {
        std::string incorrectShapes =
            "[" + std::to_string(inOrigin.GetDim(DIM_0)) + ", " + std::to_string(inOrigin.GetDim(DIM_1)) + ", " +
            std::to_string(inOrigin.GetDim(DIM_2)) + "] and [" + std::to_string(goShape.GetDim(DIM_0)) + ", " +
            std::to_string(goShape.GetDim(DIM_1)) + ", " + std::to_string(goShape.GetDim(DIM_2)) + "]";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "x and grad_y", incorrectShapes.c_str(),
                                               "The shape of x must be equal to the shape of grad_y");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedCausalConv1dBackwardTilingArch35::ValidateWeightShape()
{
    auto wShape = context_->GetInputShape(WEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, wShape);
    auto wOrigin = wShape->GetOriginShape();
    if (wOrigin.GetDimNum() != DIM_2) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "weight",
                                                 std::to_string(wOrigin.GetDimNum()) + "D",
                                                 "The shape of weight must be 2D");
        return ge::GRAPH_FAILED;
    }

    W_ = static_cast<int64_t>(wOrigin.GetDim(DIM_0));
    auto wH = static_cast<int64_t>(wOrigin.GetDim(DIM_1));

    if (W_ != NUM_3) {
        std::string incorrectShape = "[" + std::to_string(W_) + ", " + std::to_string(wH) + "]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "weight", incorrectShape.c_str(),
                                              "Shape [0] of this parameter must be equal to 3");
        return ge::GRAPH_FAILED;
    }

    if (wH != H_) {
        std::string incorrectShapes = "[" + std::to_string(W_) + ", " + std::to_string(wH) + "] and [" +
                                      std::to_string(S_) + ", " + std::to_string(B_) + ", " + std::to_string(H_) + "]";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "weight and grad_y", incorrectShapes.c_str(),
                                               "Shape [1] of weight must be equal to shape [2] of grad_y");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedCausalConv1dBackwardTilingArch35::ValidateMaskShape()
{
    auto maskShape = context_->GetOptionalInputShape(MASK_INDEX);
    if (maskShape == nullptr) {
        hasMask_ = 0;
        return ge::GRAPH_SUCCESS;
    }
    hasMask_ = 1;
    auto mOrigin = maskShape->GetOriginShape();
    if (mOrigin.GetDimNum() != NUM_2) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context_->GetNodeName(), "mask", std::to_string(mOrigin.GetDimNum()) + "D", "The shape of mask must be 2D");
        return ge::GRAPH_FAILED;
    }

    auto mB = static_cast<int64_t>(mOrigin.GetDim(DIM_0));
    auto mS = static_cast<int64_t>(mOrigin.GetDim(DIM_1));

    if (mB != B_) {
        std::string incorrectShapes = "[" + std::to_string(mB) + ", " + std::to_string(mS) + "] and [" +
                                      std::to_string(S_) + ", " + std::to_string(B_) + ", " + std::to_string(H_) + "]";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "mask and grad_y", incorrectShapes.c_str(),
                                               "Shape [0] of mask must be equal to shape [1] of grad_y");
        return ge::GRAPH_FAILED;
    }
    if (mS != S_) {
        std::string incorrectShapes = "[" + std::to_string(mB) + ", " + std::to_string(mS) + "] and [" +
                                      std::to_string(S_) + ", " + std::to_string(B_) + ", " + std::to_string(H_) + "]";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "mask and grad_y", incorrectShapes.c_str(),
                                               "Shape [1] of mask must be equal to shape [0] of grad_y");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedCausalConv1dBackwardTilingArch35::ValidateGradYType()
{
    dataType_ = context_->GetInputDesc(GRAD_Y_INDEX)->GetDataType();
    if (dataType_ != ge::DataType::DT_FLOAT16 && dataType_ != ge::DataType::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "grad_y", Ops::Base::ToString(dataType_).c_str(),
                                              "The dtype of grad_y must be within the range [float16, bfloat16]");
        return ge::GRAPH_FAILED;
    }
    dtypeSize_ = DTYPE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedCausalConv1dBackwardTilingArch35::ValidateXType()
{
    auto t = context_->GetInputDesc(X_INDEX)->GetDataType();
    if (t != dataType_) {
        std::string incorrectDtypes = Ops::Base::ToString(t) + " and " + Ops::Base::ToString(dataType_);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "x and grad_y", incorrectDtypes.c_str(),
                                               "The dtypes of x and grad_y must be the same");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedCausalConv1dBackwardTilingArch35::ValidateWeightType()
{
    auto t = context_->GetInputDesc(WEIGHT_INDEX)->GetDataType();
    if (t != dataType_) {
        std::string incorrectDtypes = Ops::Base::ToString(t) + " and " + Ops::Base::ToString(dataType_);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "weight and grad_y", incorrectDtypes.c_str(),
                                               "The dtypes of weight and grad_y must be the same");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedCausalConv1dBackwardTilingArch35::ValidateMaskType()
{
    auto desc = context_->GetOptionalInputDesc(MASK_INDEX);
    if (desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto t = desc->GetDataType();
    if (t != ge::DataType::DT_BOOL) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "mask", Ops::Base::ToString(t).c_str(),
                                              "The dtype of mask must be BOOL");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedCausalConv1dBackwardTilingArch35::CheckInputParams()
{
    // Shapes
    OP_CHECK_IF(ValidateGradYShape() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "grad_y shape validation failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateXShape() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "x shape validation failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateWeightShape() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "weight shape validation failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateMaskShape() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "mask shape validation failed"), return ge::GRAPH_FAILED);

    // Types
    OP_CHECK_IF(ValidateGradYType() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "grad_y type validation failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateXType() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "x type validation failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateWeightType() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "weight type validation failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateMaskType() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "mask type validation failed"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedCausalConv1dBackwardTilingArch35::GetShapeAttrsInfo()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE("MaskedCausalConv1dBackward", "context is null"), return ge::GRAPH_FAILED);

    // 收集形状和类型，并做完整校验
    OP_CHECK_IF(CheckInputParams() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "MaskedCausalConv1dBackward CheckInputParams FAILED."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedCausalConv1dBackwardTilingArch35::ComputeInterCoreSplit()
{
    // 沿 H 方向按 64-tile 切分，采用基于 tile 的平均分配：
    // 每个核心分到 q 或 (q+1) 个 64-tile，其中 q = tiles / usedCoreNum，r = tiles % usedCoreNum
    int64_t tiles = H_ / DIM_ALIGN_ELEMENT;
    OP_CHECK_IF(tiles <= 0, OP_LOGE(context_->GetNodeName(), "H(%ld) < %ld", H_, DIM_ALIGN_ELEMENT),
                return ge::GRAPH_FAILED);

    int64_t maxCores = static_cast<int64_t>(totalCoreNum_);
    usedCoreNum_ = std::min<int64_t>(tiles, std::max<int64_t>(1, maxCores));

    int64_t q = tiles / usedCoreNum_; // 每核基础 tile 数
    int64_t r = tiles % usedCoreNum_; // 需要多 1 个 tile 的核心数量

    if (r == 0) {
        // 完全均分：只有主核，无尾核；两种块大小相同
        hMainSize_ = q * DIM_ALIGN_ELEMENT;
        hTailSize_ = hMainSize_;
        hMainCoreCnt_ = usedCoreNum_;
        hTailCoreCnt_ = 0;
    } else {
        // 非均分：r 个主核分到 (q+1) 个 tile，其余为尾核（q 个 tile）
        hTailSize_ = q * DIM_ALIGN_ELEMENT;
        hMainSize_ = (q + 1) * DIM_ALIGN_ELEMENT;
        hMainCoreCnt_ = r;
        hTailCoreCnt_ = usedCoreNum_ - r;
    }

    return ge::GRAPH_SUCCESS;
}

// Helper: precise UB buffer size estimation for given h and s_effective (valid output rows)
static inline int64_t CalculateBufferSizeHelper(int64_t h, int64_t s_effective, int64_t S, int64_t B, int64_t W,
                                                int64_t dtypeSize, int hasMask, int64_t alignBytes, int64_t bufferNum)
{
    int64_t currentSLoopCnt = (S + s_effective - 1) / s_effective;
    int64_t sCacheLevels = 1;
    {
        int64_t t = currentSLoopCnt;
        while (t > 1) {
            ++sCacheLevels;
            t >>= 1;
        }
    }

    int64_t s_with_overlap = s_effective + (W - 1);
    int64_t gradYSize = h * s_with_overlap * dtypeSize;
    int64_t xSize = h * s_effective * dtypeSize;
    int64_t weightSize = h * W * dtypeSize;
    int64_t maskSize = 0;
    int64_t maskBufSize = 0;
    if (hasMask) {
        maskSize = ((s_with_overlap + alignBytes - 1) / alignBytes) * alignBytes;
        int64_t vecBytes = ((s_with_overlap * dtypeSize + (alignBytes - 1)) / alignBytes) * alignBytes;
        int64_t planeBytes = s_with_overlap * h * dtypeSize;
        maskBufSize += vecBytes + planeBytes;
    }
    int64_t gradXSize = h * s_effective * dtypeSize;
    int64_t gradWeightSize = h * W * dtypeSize;
    int64_t tempBuffSize = h * W * B * static_cast<int64_t>(sizeof(float));
    int64_t prodBufSize = W * s_effective * h * static_cast<int64_t>(sizeof(float));
    int64_t sumBufSize = h * static_cast<int64_t>(sizeof(float));

    int64_t wsSize = W * sCacheLevels * h * static_cast<int64_t>(sizeof(float));
    int64_t totalSize = bufferNum * (gradYSize + xSize + weightSize + maskSize + gradXSize + gradWeightSize) +
                        tempBuffSize + prodBufSize + sumBufSize + wsSize + maskBufSize;
    return totalSize;
}

ge::graphStatus MaskedCausalConv1dBackwardTilingArch35::ComputeIntraCoreUbTiling()
{
    int64_t availableUbSize = static_cast<int64_t>(ubSize_) - SYSTEM_RESERVED_UB_SIZE;
    OP_LOGI(context_->GetNodeName(), "ubSize is %ld", ubSize_);
    OP_LOGI(context_->GetNodeName(), "availableUbSize is %ld", availableUbSize);
    OP_CHECK_IF(availableUbSize <= 0, OP_LOGE(context_->GetNodeName(), "available UB size <= 0, ubSize=%lu", ubSize_),
                return ge::GRAPH_FAILED);

    // UB先按H=64(128B)，全载核内BS，核内仅切 S, sUB_ 存储的是有效输出行数（不包含重叠部分）
    int64_t minH = DIM_ALIGN_ELEMENT; // 64
    int64_t perCoreH = (hMainCoreCnt_ > 0) ? hMainSize_ : hTailSize_;
    hUB_ = minH;

    // 用精确 Buffer 估算函数，执行二分搜索以找到最大 sUB_
    int64_t lowS = 1;
    int64_t highS = std::max<int64_t>(1, S_);
    while (lowS <= highS) {
        int64_t midS = (lowS + highS) / NUM_2;
        int64_t need = CalculateBufferSizeHelper(hUB_, midS, S_, B_, W_, static_cast<int64_t>(dtypeSize_),
                                                 static_cast<int>(hasMask_), static_cast<int64_t>(ALIGN_BYTES),
                                                 static_cast<int64_t>(BUFFER_NUM));
        if (need <= availableUbSize) {
            sUB_ = midS; // 可行，尝试更大
            lowS = midS + 1;
        } else {
            highS = midS - 1; // 不可行，缩小范围
        }
    }
    // 保护：不超过 S_
    if (sUB_ > S_)
        sUB_ = S_;

    // 循环次数
    hLoopCnt_ = (perCoreH + hUB_ - 1) / hUB_;
    sLoopCnt_ = (S_ + sUB_ - 1) / sUB_;

    // 主块与尾块
    ubMainFactorH_ = hUB_;
    ubTailFactorH_ = (hLoopCnt_ == 1) ? perCoreH : (perCoreH % hUB_ == 0 ? hUB_ : perCoreH % hUB_);

    // 均分 S 维
    if (sLoopCnt_ <= 0) {
        sLoopCnt_ = 1;
    }
    ubMainFactorS_ = (S_ + sLoopCnt_ - 1) / sLoopCnt_;
    ubTailFactorS_ = (S_ / sLoopCnt_);
    ubMainFactorSCount_ = S_ % sLoopCnt_;
    ubTailFactorSCount_ = sLoopCnt_ - ubMainFactorSCount_;

    // 尾核
    if (hTailCoreCnt_ > 0 && hTailSize_ > 0) {
        // 尾核可能需要不同的H循环次数
        int64_t tailCoreH = hTailSize_;
        tailHLoopCnt_ = (tailCoreH + hUB_ - 1) / hUB_;
        tailSLoopCnt_ = sLoopCnt_;

        tailCoreUbMainFactorH_ = hUB_;
        tailCoreUbTailFactorH_ = (tailHLoopCnt_ == 1) ? hUB_ : (tailCoreH % hUB_ == 0 ? hUB_ : tailCoreH % hUB_);

        tailCoreUbMainFactorS_ = ubMainFactorS_; // 有效输出大小，不包含重叠
        tailCoreUbTailFactorS_ = ubTailFactorS_; // 有效输出大小，不包含重叠
    } else {
        // 没有尾核，或尾核参数与主核相同
        tailHLoopCnt_ = hLoopCnt_;
        tailSLoopCnt_ = sLoopCnt_;

        tailCoreUbMainFactorH_ = ubMainFactorH_;
        tailCoreUbTailFactorH_ = ubTailFactorH_;
        tailCoreUbMainFactorS_ = ubMainFactorS_;
        tailCoreUbTailFactorS_ = ubTailFactorS_;
    }

    allUb_ = CalculateBufferSizeHelper(ubMainFactorH_, ubMainFactorS_, S_, B_, W_, static_cast<int64_t>(dtypeSize_),
                                       static_cast<int>(hasMask_), static_cast<int64_t>(ALIGN_BYTES),
                                       static_cast<int64_t>(BUFFER_NUM));
    OP_LOGI(context_->GetNodeName(), "allUb_ = %ld", allUb_);
    return ge::GRAPH_SUCCESS;
}


ge::graphStatus MaskedCausalConv1dBackwardTilingArch35::DoOpTiling()
{
    OP_CHECK_IF(ComputeInterCoreSplit() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "ComputeInterCoreSplit failed"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(ComputeIntraCoreUbTiling() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "ComputeIntraCoreUbTiling failed"), return ge::GRAPH_FAILED);

    tilingData_.hMainCoreCnt = hMainCoreCnt_;
    tilingData_.hTailCoreCnt = hTailCoreCnt_;
    tilingData_.hMainSize = hMainSize_;
    tilingData_.hTailSize = hTailSize_;
    tilingData_.hLoopCnt = hLoopCnt_;
    tilingData_.sLoopCnt = sLoopCnt_;
    tilingData_.ubMainFactorH = ubMainFactorH_;
    tilingData_.ubTailFactorH = ubTailFactorH_;
    tilingData_.ubMainFactorS = ubMainFactorS_;
    tilingData_.ubTailFactorS = ubTailFactorS_;
    tilingData_.tailHLoopCnt = tailHLoopCnt_;
    tilingData_.tailSLoopCnt = tailSLoopCnt_;
    tilingData_.tailCoreUbMainFactorH = tailCoreUbMainFactorH_;
    tilingData_.tailCoreUbTailFactorH = tailCoreUbTailFactorH_;
    tilingData_.ubMainFactorSCount = ubMainFactorSCount_;
    tilingData_.ubTailFactorSCount = ubTailFactorSCount_;
    tilingData_.tailCoreUbMainFactorS = tailCoreUbMainFactorS_;
    tilingData_.tailCoreUbTailFactorS = tailCoreUbTailFactorS_;
    tilingData_.hasMask = hasMask_;
    tilingData_.S = S_;
    tilingData_.B = B_;
    tilingData_.H = H_;
    tilingData_.W = W_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedCausalConv1dBackwardTilingArch35::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MaskedCausalConv1dBackwardTilingArch35::GetTilingKey() const
{
    if (dataType_ == ge::DataType::DT_BF16) {
        return TILING_KEY_MASKED_CAUSAL_CONV1D_BACKWARD_BF16;
    }
    return TILING_KEY_MASKED_CAUSAL_CONV1D_BACKWARD_FP16;
}

ge::graphStatus MaskedCausalConv1dBackwardTilingArch35::GetWorkspaceSize()
{
    auto platformInfo = context_->GetPlatformInfo();
    uint32_t sysWorkspaceSize = 0;
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    }
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = static_cast<size_t>(0UL + sysWorkspaceSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedCausalConv1dBackwardTilingArch35::PostTiling()
{
    context_->SetBlockDim(static_cast<uint32_t>(usedCoreNum_));

    auto *raw = context_->GetRawTilingData();
    auto tilingDataSize = sizeof(MaskedCausalConv1dBackwardTilingDataV35);
    errno_t ret = memcpy_s(raw->GetData(), raw->GetCapacity(), reinterpret_cast<void *>(&tilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    raw->SetDataSize(tilingDataSize);

    context_->SetTilingKey(GetTilingKey());
    return ge::GRAPH_SUCCESS;
}

void MaskedCausalConv1dBackwardTilingArch35::DumpTilingInfo()
{
    OP_LOGI(context_->GetNodeName(), "=== MaskedCausalConv1dBackward DumpTilingInfo ===");
    OP_LOGI(context_->GetNodeName(), "hMainCoreCnt = %ld", hMainCoreCnt_);
    OP_LOGI(context_->GetNodeName(), "hTailCoreCnt = %ld", hTailCoreCnt_);
    OP_LOGI(context_->GetNodeName(), "hMainSize = %ld", hMainSize_);
    OP_LOGI(context_->GetNodeName(), "hTailSize = %ld", hTailSize_);
    OP_LOGI(context_->GetNodeName(), "hLoopCnt = %ld", hLoopCnt_);
    OP_LOGI(context_->GetNodeName(), "sLoopCnt = %ld", sLoopCnt_);
    OP_LOGI(context_->GetNodeName(), "ubMainFactorH = %ld", ubMainFactorH_);
    OP_LOGI(context_->GetNodeName(), "ubTailFactorH = %ld", ubTailFactorH_);
    OP_LOGI(context_->GetNodeName(), "ubMainFactorS = %ld", ubMainFactorS_);
    OP_LOGI(context_->GetNodeName(), "ubTailFactorS = %ld", ubTailFactorS_);
    OP_LOGI(context_->GetNodeName(), "ubMainFactorSCount = %ld", ubMainFactorSCount_);
    OP_LOGI(context_->GetNodeName(), "ubTailFactorSCount = %ld", ubTailFactorSCount_);
    OP_LOGI(context_->GetNodeName(), "tailHLoopCnt = %ld", tailHLoopCnt_);
    OP_LOGI(context_->GetNodeName(), "tailSLoopCnt = %ld", tailSLoopCnt_);
    OP_LOGI(context_->GetNodeName(), "tailCoreUbMainFactorH = %ld", tailCoreUbMainFactorH_);
    OP_LOGI(context_->GetNodeName(), "tailCoreUbTailFactorH = %ld", tailCoreUbTailFactorH_);
    OP_LOGI(context_->GetNodeName(), "tailCoreUbMainFactorS = %ld", tailCoreUbMainFactorS_);
    OP_LOGI(context_->GetNodeName(), "tailCoreUbTailFactorS = %ld", tailCoreUbTailFactorS_);
    OP_LOGI(context_->GetNodeName(), "hasMask = %ld", hasMask_);
    OP_LOGI(context_->GetNodeName(), "S = %ld", S_);
    OP_LOGI(context_->GetNodeName(), "B = %ld", B_);
    OP_LOGI(context_->GetNodeName(), "H = %ld", H_);
    OP_LOGI(context_->GetNodeName(), "W = %ld", W_);
}


} // namespace optiling
