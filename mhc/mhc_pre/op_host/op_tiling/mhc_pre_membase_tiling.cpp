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
 * \file mhc_pre_membase_tiling.cpp
 * \brief MhcPre tiling implementation
 */

#include <sstream>
#include "mhc_pre_membase_tiling.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_util.h"

using namespace ge;
namespace optiling {
namespace {
constexpr uint64_t WORKSPACE_SIZE = 32;
int64_t CeilDiv(int64_t x, int64_t y)
{
    if (y != 0) {
        return (x + y - 1) / y;
    }
    return x;
}
int64_t DownAlign(int64_t x, int64_t y)
{
    if (y == 0) {
        return x;
    }
    return (x / y) * y;
}
int64_t RoundUp(int64_t x, int64_t y)
{
    return CeilDiv(x, y) * y;
}
int64_t Factorial(int64_t n)
{
    int64_t r = 1;
    for (int64_t i = 2; i <= n; ++i) {
        r *= i;
    }
    return r;
}

constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t REPEAT_SIZE = 256;
constexpr int64_t UB_RESEVED_SIZE = 8192;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr uint64_t M_L1_MAX_SIZE = 256;
constexpr uint64_t K_MULIT_CORE_SPLIT_BASE_SIZE = 256;
constexpr uint64_t A_L1_SIZE = 128 * 256;
constexpr uint64_t K_L1_MAX_SIZE = 1024;
constexpr int64_t OUT_FLAG_ATTR_IDX = 0;
constexpr int64_t NORM_EPS_ATTR_IDX = 1;
constexpr int64_t HC_EPS_ATTR_IDX = 2;
constexpr int64_t BS_SPLIT_THRESHOLD = 128;
constexpr int64_t MAX_BS_PER_LOOP = 32;
constexpr int64_t MAX_REDUCE_SIZE = 256;
} // namespace

int8_t CheckOptionalParamsExisting(const gert::StorageShape *checkShape)
{
    OP_CHECK_IF(nullptr == checkShape, OP_LOGD("CheckOptionalParamsExisting", "Get nullptr checkShape"), return 0);
    int64_t checkShapeSize = checkShape->GetOriginShape().GetShapeSize();
    OP_CHECK_IF((checkShapeSize <= 0), OP_LOGD("CheckOptionalParamsExisting", "Get empty checkShapeSize"), return 0);
    return 1;
}

ge::graphStatus MhcPreTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<MhcPreMembaseCompileInfo>();
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
        aivCoreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aivCoreNum_ = ascendcPlatform.GetCoreNumAiv();
        aicCoreNum_ = ascendcPlatform.GetCoreNumAic();

        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize_ = ubSizePlatForm;
        socVersion_ = ascendcPlatform.GetSocVersion();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreTiling::GetAttr()
{
    auto *attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    auto outFlagAttr = attrs->GetAttrPointer<int64_t>(OUT_FLAG_ATTR_IDX);
    needGrad_ = (outFlagAttr == nullptr || *outFlagAttr == 0) ? false : true;

    auto normEpsAttr = attrs->GetAttrPointer<float>(NORM_EPS_ATTR_IDX);
    normEps_ = normEpsAttr == nullptr ? 1e-6f : *normEpsAttr;

    auto hcEpsAttr = attrs->GetAttrPointer<float>(HC_EPS_ATTR_IDX);
    hcEps_ = hcEpsAttr == nullptr ? 1e-6f : *hcEpsAttr;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreTiling::GetShapeAttrsInfoInner(int8_t &gammaExist)
{
    auto xShape = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    if (xDimNum == 3) {
        bs_ = xShape->GetStorageShape().GetDim(0);
        hcMult_ = xShape->GetStorageShape().GetDim(1);
        d_ = xShape->GetStorageShape().GetDim(2);
    } else if (xDimNum == 4) {
        int64_t b = xShape->GetStorageShape().GetDim(0);
        int64_t s = xShape->GetStorageShape().GetDim(1);
        bs_ = b * s;
        hcMult_ = xShape->GetStorageShape().GetDim(2);
        d_ = xShape->GetStorageShape().GetDim(3);
    }

    nn_fac = Factorial(hcMult_) + 2 * hcMult_;
    nn_pow = hcMult_ * hcMult_ + 2 * hcMult_;

    auto shapeAlpha = context_->GetInputShape(2);
    int64_t scaleFirstDim = shapeAlpha->GetStorageShape().GetDim(0);
    OP_CHECK_IF(scaleFirstDim != 2 && scaleFirstDim != 3,
                OP_LOGE(context_->GetNodeName(), "alpha size should be 2 or 3, but is %ld", scaleFirstDim),
                return ge::GRAPH_FAILED);
    hasResi = (scaleFirstDim == 3);

    auto shapePhi = context_->GetInputShape(1);
    hcMix_ = shapePhi->GetStorageShape().GetDim(0);

    auto shapeBias = context_->GetInputShape(3);
    int64_t biasFirstDim = shapeBias->GetStorageShape().GetDim(0);
    int64_t expectedBias = hasResi ? hcMix_ : 2 * hcMult_;

    //
    if (hasResi) {
        OP_CHECK_IF(hcMix_ != nn_fac && hcMix_ != nn_pow,
                    OP_LOGE(context_->GetNodeName(), "Phi dim 0 should be equal with %ld or %ld, but is %ld", nn_fac,
                            nn_pow, hcMix_),
                    return ge::GRAPH_FAILED);
        if (hcMix_ == nn_fac) {
            isFac = true;
        } else {
            isFac = false;
        }
    } else {
        OP_CHECK_IF(
            hcMix_ != 2 * hcMult_,
            OP_LOGE(context_->GetNodeName(), "Phi dim 0 should be equal with %ld, but is %ld", 2 * hcMult_, hcMix_),
            return ge::GRAPH_FAILED);
        isFac = false;
    }

    OP_CHECK_IF(shapePhi->GetStorageShape().GetDim(1) != d_ * hcMult_,
                OP_LOGE(context_->GetNodeName(), "Phi dim 1 should be equal with d_ * hcMult_  %ld, but is %ld",
                        d_ * hcMult_, shapePhi->GetStorageShape().GetDim(1)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        biasFirstDim != expectedBias,
        OP_LOGE(context_->GetNodeName(), "Bias dim 0 should be equal with %ld, but is %ld", expectedBias, biasFirstDim),
        return ge::GRAPH_FAILED);

    auto shapeGamma = context_->GetOptionalInputShape(4);
    gammaExist = CheckOptionalParamsExisting(shapeGamma);
    if (gammaExist == 1) {
        int64_t gammaFirstDim = shapeGamma->GetStorageShape().GetDim(0);
        int64_t gammaSecondDim = shapeGamma->GetStorageShape().GetDim(1);
        OP_CHECK_IF(gammaFirstDim != hcMult_ || gammaSecondDim != d_,
                    OP_LOGE(context_->GetNodeName(), "gamma shape should be equal with [%ld, %ld], but is [%ld, %ld]",
                            hcMult_, d_, gammaFirstDim, gammaSecondDim),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(GetAttr() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "get attr failed."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreTiling::CalcMKSplitCoreMembasePart2Tiling()
{
    rowOfFormerBlock_ = CeilDiv(bs_, static_cast<int64_t>(aivCoreNum_));
    usedAivCoreNums_ = std::min(CeilDiv(bs_, rowOfFormerBlock_), static_cast<int64_t>(aivCoreNum_));
    rowOfTailBlock_ = bs_ - (usedAivCoreNums_ - 1) * rowOfFormerBlock_;

    int64_t minRowPerCore = 1;
    int64_t rowOnceLoop = std::min(rowOfFormerBlock_, minRowPerCore);
    int64_t kBlockNum = tilingData_.get_cubeBlockDimK();

    hcMultAlign_ = RoundUp(hcMult_, BLOCK_SIZE / sizeof(float));
    int64_t mix0Size = rowOnceLoop * hcMultAlign_ * sizeof(float);
    int64_t mix1Size = rowOnceLoop * hcMultAlign_ * sizeof(float);
    int64_t mix2Size = rowOnceLoop * hcMult_ * hcMultAlign_ * sizeof(float);
    int64_t squareSumSize = kBlockNum * RoundUp(rowOnceLoop * 16, 16) * sizeof(float) * DOUBLE_BUFFER;
    int64_t rsqrtSize = RoundUp(rowOnceLoop, BLOCK_SIZE / sizeof(float)) * sizeof(float) * DOUBLE_BUFFER;
    int64_t xSize = rowOnceLoop * hcMult_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER;
    int64_t ySize = rowOnceLoop * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER;
    int64_t postSize = rowOnceLoop * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
    int64_t combFragSize = rowOnceLoop * hcMult_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
    int64_t combFragBufSize = rowOnceLoop * hcMult_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER * 2;
    int64_t base0Size = hcMultAlign_ * sizeof(float);
    int64_t base1Size = hcMultAlign_ * sizeof(float);
    int64_t base2Size = hcMult_ * hcMultAlign_ * sizeof(float);
    int64_t xCastSize = rowOnceLoop * hcMult_ * RoundUp(d_, 8) * sizeof(float);
    int64_t yCastSize = rowOnceLoop * RoundUp(d_, 8) * sizeof(float);
    int64_t rowBrcb0Size = RoundUp(rowOnceLoop, 8) * BLOCK_SIZE;
    int64_t hcBrcb1Size = RoundUp(rowOnceLoop * hcMultAlign_, 8) * 2 * BLOCK_SIZE;
    int64_t reduceBufSize = rowOnceLoop * hcMultAlign_ * sizeof(float);
    int64_t maskPatternSize = BLOCK_SIZE * 16;

    int64_t totalSize = mix0Size + mix1Size + mix2Size + squareSumSize + rsqrtSize + xSize + ySize + postSize +
                        combFragSize + combFragBufSize + base0Size + base1Size + base2Size + xCastSize + yCastSize +
                        rowBrcb0Size + hcBrcb1Size + reduceBufSize + maskPatternSize;
    rowFactor_ = rowOnceLoop;
    if (totalSize <= ubSize_) {
        dLoop_ = 1;
        dFactor_ = d_;
        tailDFactor_ = dFactor_;
    } else {
        int64_t usedUbSize = mix0Size + mix1Size + mix2Size + squareSumSize + rsqrtSize + postSize + combFragSize +
                             base0Size + base1Size + base2Size + rowBrcb0Size + hcBrcb1Size + reduceBufSize +
                             maskPatternSize;
        int64_t ubRemain = ubSize_ - usedUbSize;
        dFactor_ = d_;
        int64_t base = 2;
        while (1) {
            dFactor_ = CeilDiv(d_, base);
            xSize = rowOnceLoop * hcMult_ * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
            ySize = rowOnceLoop * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
            xCastSize = rowOnceLoop * hcMult_ * RoundUp(dFactor_, 8) * sizeof(float);
            yCastSize = rowOnceLoop * RoundUp(dFactor_, 8) * sizeof(float);
            int64_t targetSize = xSize + ySize + xCastSize + yCastSize;
            if (targetSize <= ubRemain) {
                break;
            }
            base++;
        }
        if (dFactor_ > 32) {
            dFactor_ = DownAlign(dFactor_, 32);
        }
        dLoop_ = CeilDiv(d_, dFactor_);
        tailDFactor_ = d_ % dFactor_ == 0 ? dFactor_ : d_ % dFactor_;
    }

    if (dFactor_ == d_) {
        rowFactor_ = rowFactor_ > rowOfFormerBlock_ ? rowFactor_ - 1 : rowFactor_;
    }
    rowLoopOfFormerBlock_ = CeilDiv(rowOfFormerBlock_, rowFactor_);
    rowLoopOfTailBlock_ = CeilDiv(rowOfTailBlock_, rowFactor_);
    tailRowFactorOfFormerBlock_ = rowOfFormerBlock_ % rowFactor_ == 0 ? rowFactor_ : rowOfFormerBlock_ % rowFactor_;
    tailRowFactorOfTailBlock_ = rowOfTailBlock_ % rowFactor_ == 0 ? rowFactor_ : rowOfTailBlock_ % rowFactor_;

    tilingData_.set_bs(bs_);
    tilingData_.set_hcMix(hcMix_);
    tilingData_.set_hcMult(hcMult_);
    tilingData_.set_d(d_);
    tilingData_.set_hcMultAlign(hcMultAlign_);
    tilingData_.set_rowOfFormerBlock(rowOfFormerBlock_);
    tilingData_.set_rowLoopOfFormerBlock(rowLoopOfFormerBlock_);
    tilingData_.set_rowLoopOfTailBlock(rowLoopOfTailBlock_);
    tilingData_.set_stage2RowFactor(rowFactor_);
    tilingData_.set_secondUsedCoreNum(usedAivCoreNums_);
    tilingData_.set_tailRowFactorOfFormerBlock(tailRowFactorOfFormerBlock_);
    tilingData_.set_tailRowFactorOfTailBlock(tailRowFactorOfTailBlock_);
    tilingData_.set_dLoop(dLoop_);
    tilingData_.set_dFactor(dFactor_);
    tilingData_.set_tailDFactor(tailDFactor_);
    tilingData_.set_hcEps(hcEps_);
    tilingData_.set_normEps(normEps_);
    tilingData_.set_needGrad(needGrad_ ? 1 : 0);
    return ge::GRAPH_SUCCESS;
}

// 预留功能，切K情况尾核单独计算
ge::graphStatus MhcPreTiling::CalcBsSplit()
{
    tailBs_ = bs_;
    if (bs_ > BS_SPLIT_THRESHOLD) {
        tailBs_ = bs_ % BS_SPLIT_THRESHOLD;
        if (tailBs_ == 0) {
            tailBs_ = BS_SPLIT_THRESHOLD;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreTiling::CalcTailBsTiling()
{
    if (tailBs_ == BS_SPLIT_THRESHOLD || tailBs_ == bs_) {
        tilingData_.set_tailBsRowOfFormerBlock(tilingData_.get_rowOfFormerBlock());
        tilingData_.set_tailBsRowLoopOfFormerBlock(tilingData_.get_rowLoopOfFormerBlock());
        tilingData_.set_tailBsRowLoopOfTailBlock(tilingData_.get_rowLoopOfTailBlock());
        tilingData_.set_tailBsUsedCoreNum(tilingData_.get_secondUsedCoreNum());
        tilingData_.set_tailBsRowFactor(tilingData_.get_stage2RowFactor());
        tilingData_.set_tailBsTailRowFactorOfFormerBlock(tilingData_.get_tailRowFactorOfFormerBlock());
        tilingData_.set_tailBsTailRowFactorOfTailBlock(tilingData_.get_tailRowFactorOfTailBlock());
        tilingData_.set_tailBsML1Size(tilingData_.get_mL1Size());
        return ge::GRAPH_SUCCESS;
    }

    uint64_t tailMDimNum = std::min(aicCoreNum_, static_cast<uint64_t>(CeilDiv(tailBs_, M_L1_MAX_SIZE)));
    uint64_t tailSingleCoreM = RoundUp(CeilDiv(tailBs_, tailMDimNum), AscendC::BLOCK_CUBE);

    tailBsML1Size_ = std::min(M_L1_MAX_SIZE, tailSingleCoreM);

    tailBsRowOfFormerBlock_ = CeilDiv(tailBs_, static_cast<int64_t>(aivCoreNum_));
    tailBsUsedCoreNum_ = std::min(CeilDiv(tailBs_, tailBsRowOfFormerBlock_), static_cast<int64_t>(aivCoreNum_));
    tailBsRowOfTailBlock_ = tailBs_ - (tailBsUsedCoreNum_ - 1) * tailBsRowOfFormerBlock_;

    int64_t minRowPerCore = 1;
    tailBsRowFactor_ = std::min(tailBsRowOfFormerBlock_, minRowPerCore);

    tailBsRowLoopOfFormerBlock_ = CeilDiv(tailBsRowOfFormerBlock_, tailBsRowFactor_);
    tailBsRowLoopOfTailBlock_ = CeilDiv(tailBsRowOfTailBlock_, tailBsRowFactor_);
    tailBsTailRowFactorOfFormerBlock_ =
        tailBsRowOfFormerBlock_ % tailBsRowFactor_ == 0 ? tailBsRowFactor_ : tailBsRowOfFormerBlock_ % tailBsRowFactor_;
    tailBsTailRowFactorOfTailBlock_ =
        tailBsRowOfTailBlock_ % tailBsRowFactor_ == 0 ? tailBsRowFactor_ : tailBsRowOfTailBlock_ % tailBsRowFactor_;

    tilingData_.set_tailBsRowOfFormerBlock(tailBsRowOfFormerBlock_);
    tilingData_.set_tailBsRowLoopOfFormerBlock(tailBsRowLoopOfFormerBlock_);
    tilingData_.set_tailBsRowLoopOfTailBlock(tailBsRowLoopOfTailBlock_);
    tilingData_.set_tailBsUsedCoreNum(tailBsUsedCoreNum_);
    tilingData_.set_tailBsRowFactor(tailBsRowFactor_);
    tilingData_.set_tailBsTailRowFactorOfFormerBlock(tailBsTailRowFactorOfFormerBlock_);
    tilingData_.set_tailBsTailRowFactorOfTailBlock(tailBsTailRowFactorOfTailBlock_);
    tilingData_.set_tailBsML1Size(tailBsML1Size_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreTiling::CalcOpTiling()
{
    if (CalcBsSplit() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    uint64_t kSize = hcMult_ * d_;
    // 切K计算stage1
    uint64_t mDimNum = std::min(aicCoreNum_, static_cast<uint64_t>(CeilDiv(bs_, M_L1_MAX_SIZE)));
    uint64_t singleCoreM = RoundUp(CeilDiv(bs_, mDimNum), AscendC::BLOCK_CUBE);

    uint64_t kDimNum = aicCoreNum_ / mDimNum;
    uint64_t splitKSize = RoundUp(CeilDiv(kSize, kDimNum), K_MULIT_CORE_SPLIT_BASE_SIZE);

    tilingData_.set_cubeBlockDimK(CeilDiv(kSize, splitKSize));
    tilingData_.set_mL1Size(std::min(M_L1_MAX_SIZE, singleCoreM));

    tilingData_.set_cvLoopKSize(1024);

    tilingData_.set_cubeCoreNum(static_cast<int64_t>(aicCoreNum_));
    return CalcMKSplitCoreMembasePart2Tiling();
}

ge::graphStatus MhcPreTiling::DoOpTiling()
{
    if (GetPlatformInfo() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    int8_t gammaExist = 0;
    if (GetShapeAttrsInfoInner(gammaExist) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    tilingData_.set_hasGamma(gammaExist);
    // 预留参数计算（切K）与stage2计算
    if (CalcOpTiling() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    // 预留参数计算（切K）
    if (CalcTailBsTiling() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    // 目前stage1计算在用分支
    if (CalcStage1Tiling() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (GetWorkspaceSize() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (PostTiling() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static void PrintTilingData(const gert::TilingContext *context, MhcPreMembaseTilingData &tiling)
{
    OP_LOGD(context->GetNodeName(), ">>>>>>>>>> Start to print MhcPre tiling data <<<<<<<<<<");
    OP_LOGD(context->GetNodeName(), "MhcPre_tiling: bs is %ld.", tiling.get_bs());
    OP_LOGD(context->GetNodeName(), "MhcPre_tiling: hcMix is %ld.", tiling.get_hcMix());
    OP_LOGD(context->GetNodeName(), "MhcPre_tiling: hcMult is %ld.", tiling.get_hcMult());
    OP_LOGD(context->GetNodeName(), "MhcPre_tiling: d is %ld.", tiling.get_d());
    OP_LOGD(context->GetNodeName(), "MhcPre_tiling: hcMultAlign is %ld.", tiling.get_hcMultAlign());
    OP_LOGD(context->GetNodeName(), "MhcPre_tiling: rowOfFormerBlock is %ld.", tiling.get_rowOfFormerBlock());
    OP_LOGD(context->GetNodeName(), "MhcPre_tiling: rowLoopOfFormerBlock is %ld.", tiling.get_rowLoopOfFormerBlock());
    OP_LOGD(context->GetNodeName(), "MhcPre_tiling: rowLoopOfTailBlock is %ld.", tiling.get_rowLoopOfTailBlock());
    OP_LOGD(context->GetNodeName(), "MhcPre_tiling: tailRowFactorOfFormerBlock is %ld.",
            tiling.get_tailRowFactorOfFormerBlock());
    OP_LOGD(context->GetNodeName(), "MhcPre_tiling: tailRowFactorOfTailBlock is %ld.",
            tiling.get_tailRowFactorOfTailBlock());
    OP_LOGD(context->GetNodeName(), "MhcPre_tiling: dLoop is %ld.", tiling.get_dLoop());
    OP_LOGD(context->GetNodeName(), "MhcPre_tiling: dFactor is %ld.", tiling.get_dFactor());
    OP_LOGD(context->GetNodeName(), "MhcPre_tiling: tailDFactor is %ld.", tiling.get_tailDFactor());
    OP_LOGD(context->GetNodeName(), "MhcPre_tiling: hcEps is %f.", tiling.get_hcEps());
    OP_LOGD(context->GetNodeName(), "MhcPre_tiling: normEps is %f.", tiling.get_normEps());
    OP_LOGD(context->GetNodeName(), "MhcPre_tiling: stage2RowFactor is %ld.", tiling.get_stage2RowFactor());
    OP_LOGD(context->GetNodeName(), "MhcPre_tiling: mL1Size is %ld.", tiling.get_mL1Size());
    OP_LOGD(context->GetNodeName(), "MhcPre_tiling: cubeBlockDimK is %ld.", tiling.get_cubeBlockDimK());
    OP_LOGD(context->GetNodeName(), "MhcPre_tiling: cvLoopKSize is %ld.", tiling.get_cvLoopKSize());
    OP_LOGD(context->GetNodeName(), "MhcPre_tiling: secondUsedCoreNum is %ld.", tiling.get_secondUsedCoreNum());
    OP_LOGD(context->GetNodeName(), "MhcPre_tiling: cubeCoreNum is %ld.", tiling.get_cubeCoreNum());
    OP_LOGD(context->GetNodeName(), "MhcPre_tiling: needGrad is %ld.", tiling.get_needGrad());
    OP_LOGD(context->GetNodeName(), "MhcPre_tiling: tailBsRowOfFormerBlock is %ld.",
            tiling.get_tailBsRowOfFormerBlock());
    OP_LOGD(context->GetNodeName(), "MhcPre_tiling: tailBsRowLoopOfFormerBlock is %ld.",
            tiling.get_tailBsRowLoopOfFormerBlock());
    OP_LOGD(context->GetNodeName(), "MhcPre_tiling: tailBsRowLoopOfTailBlock is %ld.",
            tiling.get_tailBsRowLoopOfTailBlock());
    OP_LOGD(context->GetNodeName(), "MhcPre_tiling: tailBsUsedCoreNum is %ld.", tiling.get_tailBsUsedCoreNum());
    OP_LOGD(context->GetNodeName(), "MhcPre_tiling: tailBsRowFactor is %ld.", tiling.get_tailBsRowFactor());
    OP_LOGD(context->GetNodeName(), "MhcPre_tiling: tailBsTailRowFactorOfFormerBlock is %ld.",
            tiling.get_tailBsTailRowFactorOfFormerBlock());
    OP_LOGD(context->GetNodeName(), "MhcPre_tiling: tailBsTailRowFactorOfTailBlock is %ld.",
            tiling.get_tailBsTailRowFactorOfTailBlock());
    OP_LOGD(context->GetNodeName(), "MhcPre_tiling: tailBsML1Size is %ld.", tiling.get_tailBsML1Size());
    OP_LOGD(context->GetNodeName(), "MhcPre_tiling: stage1VecCoreNum is %ld.", tiling.get_stage1VecCoreNum());
    OP_LOGD(context->GetNodeName(), "MhcPre_tiling: stage1CubeCoreNum is %ld.", tiling.get_stage1CubeCoreNum());
    OP_LOGD(context->GetNodeName(), "MhcPre_tiling: stage1BsFactor is %ld.", tiling.get_stage1BsFactor());
    OP_LOGD(context->GetNodeName(), "MhcPre_tiling: stage1NcLoop is %ld.", tiling.get_stage1NcLoop());
    OP_LOGD(context->GetNodeName(), "MhcPre_tiling: stage1NcFactor is %ld.", tiling.get_stage1NcFactor());
    OP_LOGD(context->GetNodeName(), "MhcPre_tiling: stage1TailNcFactor is %ld.", tiling.get_stage1TailNcFactor());
    OP_LOGD(context->GetNodeName(), "MhcPre_tiling: stage1XCastWsSize is %ld.", tiling.get_stage1XCastWsSize());
}

ge::graphStatus MhcPreTiling::CalcStage1Tiling()
{
    // 1. 计算核心数量
    int64_t vecCoreNum = static_cast<int64_t>(aivCoreNum_);
    int64_t cubeCoreNum = static_cast<int64_t>(aicCoreNum_);

    tilingData_.set_stage1VecCoreNum(vecCoreNum);
    tilingData_.set_stage1CubeCoreNum(cubeCoreNum);

    int64_t bsFactor = MAX_BS_PER_LOOP;

    tilingData_.set_stage1BsFactor(bsFactor);

    // 4. 计算 ReduceSum 切分（每次最多 256 个元素）
    int64_t ncSize = hcMult_ * d_;
    int64_t ncLoop = CeilDiv(ncSize, MAX_REDUCE_SIZE);
    int64_t ncFactor = MAX_REDUCE_SIZE;
    int64_t tailNcFactor = ncSize % MAX_REDUCE_SIZE;
    if (tailNcFactor == 0) {
        tailNcFactor = MAX_REDUCE_SIZE;
    }

    tilingData_.set_stage1NcLoop(ncLoop);
    tilingData_.set_stage1NcFactor(ncFactor);
    tilingData_.set_stage1TailNcFactor(tailNcFactor);

    int64_t mm1M = bsFactor * 2;
    int64_t mm1N = isFac ? nn_fac : nn_pow;
    if (!hasResi) {
        mm1N = 2 * hcMult_;
    }
    int64_t mm1K = hcMult_ * d_;

    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE("TilingForMhcPre", "Tiling platformInfo is null"),
                return ge::GRAPH_FAILED);
    auto ascendPlatformInfo = platform_ascendc::PlatformAscendC(platformInfo);

    auto featureDataType = matmul_tiling::DataType::DT_FLOAT;
    matmul_tiling::MatmulApiTiling mm1Tiling(ascendPlatformInfo);

    mm1Tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, featureDataType);
    mm1Tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, featureDataType, true);
    mm1Tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, featureDataType);
    mm1Tiling.SetOrgShape(mm1M, mm1N, mm1K);
    mm1Tiling.SetShape(mm1M, mm1N, mm1K);
    mm1Tiling.SetBias(false);
    mm1Tiling.SetBufferSpace(-1, -1, -1);
    if (mm1Tiling.GetTiling(tilingData_.mm1TilingData) == -1) {
        OP_LOGE(context_, "mm1Tiling.GetTiling failed, M=%ld, N=%ld, K=%ld", mm1M, mm1N, mm1K);
        return ge::GRAPH_FAILED;
    }

    // 5. 计算 Workspace 大小
    // X_cast: cubeCoreNum * maxBsPerCore * n * c * sizeof(float)
    int64_t xCastWsSize = bsFactor * vecCoreNum * 2 * ncSize * sizeof(float);
    workspaceSize_ += xCastWsSize;
    tilingData_.set_stage1XCastWsSize(xCastWsSize);

    // 6. 计算额外的 workspace 大小（needGrad=false 时使用）
    if (!needGrad_) {
        // invRms: bs * sizeof(float)
        int64_t invRmsWsSize = bs_ * sizeof(float);
        // hcBeforeNorm: bs * hcMix * sizeof(float)
        int64_t hcMix = isFac ? nn_fac : nn_pow;
        if (!hasResi) {
            hcMix = 2 * hcMult_;
        }
        int64_t hcBeforeNormWsSize = bs_ * hcMix * sizeof(float);

        workspaceSize_ += invRmsWsSize + hcBeforeNormWsSize;
    }

    int64_t kBlockNum = 1;
    tilingData_.set_cubeBlockDimK(kBlockNum);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreTiling::GetWorkspaceSize()
{
    workspaceSize_ += 16 * 1024 * 1024 + 128 * 1024 * 1024;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreTiling::PostTiling()
{
    int64_t tilingKey = 0;
    if (!isFac && hasResi) {
        tilingKey = 10;
    } else if (!isFac && !hasResi) {
        tilingKey = 11;
    } else {
        tilingKey = 0;
    }
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(aicCoreNum_);
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;

    PrintTilingData(context_, tilingData_);

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForMhcPre(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForMhcPre(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("MhcPre", "Tiling context is null"), return ge::GRAPH_FAILED);

    if (Ops::Transformer::OpTiling::IsRegbaseSocVersion(context)) {
        OP_LOGD(context->GetNodeName(), "Using arch35 tiling for regbase soc");
        return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
    }

    MhcPreTiling membaseTiling(context);
    return membaseTiling.DoOpTiling();
}

IMPL_OP_OPTILING(MhcPre).Tiling(TilingForMhcPre).TilingParse<MhcPreMembaseCompileInfo>(TilingPrepareForMhcPre);

} // namespace optiling
