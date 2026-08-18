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
 * \file mhc_pre_sinkhorn_tiling_arch35.cpp
 * \brief
 */

#include <sstream>
#include "mhc_pre_sinkhorn_tiling.h"

using namespace ge;
namespace optiling {
namespace {
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

constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t REPEAT_SIZE = 256;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t BF16_BYTES = 2;
constexpr int64_t CUBE_D_ALIGN = 16;
constexpr int64_t D_FACTOR_ALIGN = 32;
constexpr uint64_t M_L1_MAX_SIZE = 256;
constexpr uint64_t K_MULIT_CORE_SPLIT_BASE_SIZE = 256;
constexpr uint64_t A_L1_SIZE = 128 * 256;
constexpr uint64_t K_L1_MAX_SIZE = 1024;
// Ascend950(regbase, A5) only supports these two tail-axis(c) sizes, which differs from A2/A3(membase).
constexpr int64_t C_VALID_VALUE_4096 = 4096;
constexpr int64_t C_VALID_VALUE_7168 = 7168;
constexpr int64_t HCMULT_VALUE = 4;
constexpr int64_t NUM_ITERS_VALUE = 20;
constexpr size_t X_DIM_NUM_3 = 3;
constexpr size_t X_DIM_NUM_4 = 4;
constexpr int32_t BATCH_CONSISTENCY_LEVEL = 3; // batch一致性开关使能的确定性级别
constexpr uint32_t BATCH_MODE = 1;
constexpr uint64_t TILING_KEY_NO_GRAD_K_SPLIT = 1000;
constexpr uint64_t TILING_KEY_NO_GRAD_M_SPLIT = 1001;
constexpr uint64_t TILING_KEY_GRAD_K_SPLIT = 2000;
constexpr uint64_t TILING_KEY_GRAD_M_SPLIT = 2001;

struct CoreRowTiling {
    int64_t rowOfFormerBlock = 0;
    int64_t rowOfTailBlock = 0;
    int64_t usedCoreNum = 0;
    int64_t rowLoopOfFormerBlock = 0;
    int64_t rowLoopOfTailBlock = 0;
    int64_t tailRowFactorOfFormerBlock = 0;
    int64_t tailRowFactorOfTailBlock = 0;
};

struct RowAndDTiling {
    int64_t rowFactor = 0;
    int64_t dLoop = 0;
    int64_t dFactor = 0;
    int64_t tailDFactor = 0;
};

struct UbBufferConfig {
    int64_t hcMix = 0;
    int64_t hcMult = 0;
    int64_t hcMultAlign = 0;
    int64_t iterTimes = 0;
    int64_t kBlockNum = 0;
    int64_t mUbSize = 0;
    bool withGradOut = false;
    bool isKSplit = false;
};

bool CalcCoreRowTiling(int64_t bs, int64_t coreNum, CoreRowTiling &rowTiling)
{
    if (bs <= 0 || coreNum <= 0) {
        return false;
    }
    rowTiling.rowOfFormerBlock = CeilDiv(bs, coreNum);
    rowTiling.usedCoreNum = std::min(CeilDiv(bs, rowTiling.rowOfFormerBlock), coreNum);
    rowTiling.rowOfTailBlock = bs - (rowTiling.usedCoreNum - 1) * rowTiling.rowOfFormerBlock;
    return true;
}

void CompleteCoreRowTiling(CoreRowTiling &rowTiling, int64_t rowFactor)
{
    rowTiling.rowLoopOfFormerBlock = CeilDiv(rowTiling.rowOfFormerBlock, rowFactor);
    rowTiling.rowLoopOfTailBlock = CeilDiv(rowTiling.rowOfTailBlock, rowFactor);
    rowTiling.tailRowFactorOfFormerBlock = rowTiling.rowOfFormerBlock % rowFactor == 0 ?
                                               rowFactor :
                                               rowTiling.rowOfFormerBlock % rowFactor;
    rowTiling.tailRowFactorOfTailBlock = rowTiling.rowOfTailBlock % rowFactor == 0 ?
                                             rowFactor :
                                             rowTiling.rowOfTailBlock % rowFactor;
}

int64_t CalcUbBufferSize(const UbBufferConfig &config, int64_t rowFactor, int64_t dFactor)
{
    const int64_t floatAlign = BLOCK_SIZE / sizeof(float);
    const int64_t hcMixAlign = RoundUp(config.hcMix, floatAlign);
    const int64_t dAlign = RoundUp(dFactor, CUBE_D_ALIGN);

    int64_t totalSize = rowFactor * hcMixAlign * sizeof(float);                                  // mixes
    totalSize += rowFactor * config.hcMult * dAlign * BF16_BYTES * DOUBLE_BUFFER;                // x
    totalSize += rowFactor * dAlign * BF16_BYTES * DOUBLE_BUFFER;                                // y
    totalSize += rowFactor * config.hcMultAlign * sizeof(float) * DOUBLE_BUFFER;                 // post
    totalSize += rowFactor * config.hcMult * config.hcMultAlign * sizeof(float) * DOUBLE_BUFFER; // combFrag

    if (config.isKSplit) {
        totalSize += config.kBlockNum * rowFactor * hcMixAlign * sizeof(float) * DOUBLE_BUFFER;         // mm
        totalSize += config.kBlockNum * RoundUp(rowFactor, floatAlign) * sizeof(float) * DOUBLE_BUFFER; // rms
        totalSize += config.hcMultAlign * sizeof(float) * 2;                                            // base0/base1
        totalSize += config.hcMult * config.hcMultAlign * sizeof(float);                                // base2
    }

    if (config.withGradOut) {
        totalSize += 2 * config.iterTimes * rowFactor * config.hcMult * config.hcMultAlign *
                     sizeof(float) * DOUBLE_BUFFER; // normOut
        totalSize += 2 * config.iterTimes * rowFactor * config.hcMultAlign * sizeof(float) *
                     DOUBLE_BUFFER;                                                  // sumOut
        totalSize += RoundUp(rowFactor, floatAlign) * sizeof(float) * DOUBLE_BUFFER; // invRmsOut
        const int64_t hcBeforeNormRows = config.isKSplit ? rowFactor : config.mUbSize;
        totalSize += hcBeforeNormRows * hcMixAlign * sizeof(float) * DOUBLE_BUFFER;  // hcBeforeNorm
        totalSize += rowFactor * config.hcMultAlign * sizeof(float) * DOUBLE_BUFFER; // hPre
    }
    return totalSize;
}

int64_t CalcMOnlyBufferPool1Size(uint64_t ubSize, int64_t mUbSize, int64_t hcMix, int64_t hcMult,
                                 int64_t hcMultAlign)
{
    const int64_t floatAlign = BLOCK_SIZE / sizeof(float);
    const int64_t mmXBufSize = mUbSize * RoundUp(hcMix, floatAlign) * sizeof(float);
    const int64_t rmsNormBufSize = RoundUp(mUbSize, floatAlign) * sizeof(float);
    const int64_t baseSize = hcMultAlign * sizeof(float) * 2 + hcMult * hcMultAlign * sizeof(float);
    return DownAlign(static_cast<int64_t>(ubSize) - mmXBufSize - rmsNormBufSize - baseSize, BLOCK_SIZE);
}

template <typename FitsInUb>
int64_t FindMaxRowFactor(int64_t maxRowFactor, const FitsInUb &fitsInUb)
{
    int64_t left = 1;
    int64_t right = maxRowFactor;
    int64_t result = 0;
    while (left <= right) {
        const int64_t middle = left + (right - left) / 2;
        if (fitsInUb(middle)) {
            result = middle;
            left = middle + 1;
        } else {
            right = middle - 1;
        }
    }
    return result;
}

template <typename FitsInUb>
int64_t FindFirstDFactorDivisor(int64_t d, const FitsInUb &fitsInUb)
{
    int64_t left = 2;
    int64_t right = d;
    while (left < right) {
        const int64_t middle = left + (right - left) / 2;
        if (fitsInUb(1, CeilDiv(d, middle))) {
            right = middle;
        } else {
            left = middle + 1;
        }
    }
    return left;
}

template <typename FitsInUb>
bool CalcRowAndDTiling(int64_t d, int64_t maxRowFactor, const FitsInUb &fitsInUb, RowAndDTiling &result)
{
    if (d <= 0 || maxRowFactor <= 0 || !fitsInUb(1, 1)) {
        return false;
    }

    result.rowFactor = 1;
    result.dFactor = d;
    if (fitsInUb(1, d)) {
        result.rowFactor = FindMaxRowFactor(maxRowFactor,
                                            [&fitsInUb, d](int64_t rowFactor) { return fitsInUb(rowFactor, d); });
    } else {
        // fitsInUb(1, CeilDiv(d, base)) is monotonic in base, so binary search finds the first feasible base.
        result.dFactor = CeilDiv(d, FindFirstDFactorDivisor(d, fitsInUb));
        if (result.dFactor > D_FACTOR_ALIGN) {
            result.dFactor = DownAlign(result.dFactor, D_FACTOR_ALIGN);
        }
    }

    result.dLoop = CeilDiv(d, result.dFactor);
    result.tailDFactor = d % result.dFactor == 0 ? result.dFactor : d % result.dFactor;
    return result.rowFactor > 0 && result.dFactor > 0;
}

void PrintFinalTilingData(gert::TilingContext *context, MhcPreSinkhornRegbaseTilingData &tilingData, uint64_t tilingKey,
                          uint64_t blockDim, size_t workspaceSize)
{
    OP_LOGD(context->GetNodeName(),
            "MhcPreSinkhorn regbase final tiling summary: tilingKey=%lu, blockDim=%lu, workspaceSize=%zu, "
            "tilingDataSize=%zu",
            tilingKey, blockDim, workspaceSize, tilingData.GetDataSize());
    OP_LOGD(context->GetNodeName(),
            "MhcPreSinkhorn regbase final tiling shape: bs=%ld, hcMix=%ld, hcMult=%ld, d=%ld, hcMultAlign=%ld, "
            "iterTimes=%ld, hcEps=%g, normEps=%g",
            tilingData.get_bs(), tilingData.get_hcMix(), tilingData.get_hcMult(), tilingData.get_d(),
            tilingData.get_hcMultAlign(), tilingData.get_iterTimes(), tilingData.get_hcEps(), tilingData.get_normEps());
    OP_LOGD(context->GetNodeName(),
            "MhcPreSinkhorn regbase final tiling core: rowOfFormerBlock=%ld, "
            "rowLoopOfFormerBlock=%ld, rowLoopOfTailBlock=%ld, stage2RowFactor=%ld, "
            "secondUsedCoreNum=%ld, rowInnerFactor=%ld",
            tilingData.get_rowOfFormerBlock(), tilingData.get_rowLoopOfFormerBlock(),
            tilingData.get_rowLoopOfTailBlock(), tilingData.get_stage2RowFactor(),
            tilingData.get_secondUsedCoreNum(), tilingData.get_rowInnerFactor());
    OP_LOGD(context->GetNodeName(),
            "MhcPreSinkhorn regbase final tiling tail: tailRowFactorOfFormerBlock=%ld, "
            "tailRowFactorOfTailBlock=%ld, dLoop=%ld, dFactor=%ld, tailDFactor=%ld",
            tilingData.get_tailRowFactorOfFormerBlock(), tilingData.get_tailRowFactorOfTailBlock(),
            tilingData.get_dLoop(), tilingData.get_dFactor(), tilingData.get_tailDFactor());
    OP_LOGD(context->GetNodeName(),
            "MhcPreSinkhorn regbase final tiling split: k=%ld, cubeBlockDimM=%ld, cubeBlockDimK=%ld, "
            "kBlockFactor=%ld, multCoreSplitKSize=%ld, mL1Size=%ld, kL1Size=%ld, "
            "mUbSize=%ld, kUbSize=%ld, bufferPool0Size=%ld, bufferPool1Size=%ld",
            tilingData.get_k(), tilingData.get_cubeBlockDimM(), tilingData.get_cubeBlockDimK(),
            tilingData.get_kBlockFactor(), tilingData.get_multCoreSplitKSize(),
            tilingData.get_mL1Size(), tilingData.get_kL1Size(), tilingData.get_mUbSize(), tilingData.get_kUbSize(),
            tilingData.get_bufferPool0Size(), tilingData.get_bufferPool1Size());
}
} // namespace

MhcPreSinkhornTilingRegbase::MhcPreSinkhornTilingRegbase(gert::TilingContext *tilingContext)
    : context_(tilingContext)
{
}

ge::graphStatus MhcPreSinkhornTilingRegbase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<MhcPreSinkhornCompileInfo>();
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
        aivCoreNum_ = compileInfoPtr->coreNum;
        aicCoreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
        sysWorkspaceSize_ = compileInfoPtr->sysWorkspaceSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aivCoreNum_ = ascendcPlatform.GetCoreNumAiv();
        aicCoreNum_ = ascendcPlatform.GetCoreNumAic();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize_ = ubSizePlatForm;
        sysWorkspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
        socVersion_ = ascendcPlatform.GetSocVersion();
    }
    OP_LOGD(context_->GetNodeName(),
            "MhcPreSinkhorn regbase platform: aivCoreNum=%lu, aicCoreNum=%lu, ubSize=%lu, sysWorkspaceSize=%lu",
            aivCoreNum_, aicCoreNum_, ubSize_, sysWorkspaceSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornTilingRegbase::GetAttr()
{
    auto *attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    auto hcMultAttr = attrs->GetAttrPointer<int64_t>(0);
    hcMult_ = hcMultAttr == nullptr ? 4 : *hcMultAttr;

    auto iterTimesAttr = attrs->GetAttrPointer<int64_t>(1);
    iterTimes_ = iterTimesAttr == nullptr ? 20 : *iterTimesAttr;

    auto epsAttr = attrs->GetAttrPointer<float>(2);
    hcEps_ = epsAttr == nullptr ? 1e-6 : *epsAttr;

    auto normEpsAttr = attrs->GetAttrPointer<float>(3);
    normEps_ = normEpsAttr == nullptr ? 1e-6 : *normEpsAttr;

    auto needGradAttr = attrs->GetAttrPointer<bool>(4);
    needGrad_ = needGradAttr == nullptr ? true : *needGradAttr;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornTilingRegbase::GetShapeAttrsInfoInner()
{
    // (b, s, hc_mult, d) or (bs, hc_mult, d)
    auto xShape = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    if (xDimNum == X_DIM_NUM_3) {
        bs_ = xShape->GetStorageShape().GetDim(0);
        hcMult_ = xShape->GetStorageShape().GetDim(1);
        d_ = xShape->GetStorageShape().GetDim(2);
    } else if (xDimNum == X_DIM_NUM_4) {
        int64_t b = xShape->GetStorageShape().GetDim(0);
        int64_t s = xShape->GetStorageShape().GetDim(1);
        bs_ = b * s;
        hcMult_ = xShape->GetStorageShape().GetDim(2);
        d_ = xShape->GetStorageShape().GetDim(3);
    } else {
        OP_LOGE(context_->GetNodeName(), "x dim num should be 3 or 4, but is %ld", static_cast<int64_t>(xDimNum));
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(d_ != C_VALID_VALUE_4096 && d_ != C_VALID_VALUE_7168,
                OP_LOGE(context_->GetNodeName(),
                        "On Ascend950, x last dim c only supports %ld or %ld, but is %ld. "
                        "Please set x last dim c to 4096 or 7168.",
                        C_VALID_VALUE_4096, C_VALID_VALUE_7168, d_),
                return ge::GRAPH_FAILED);

    auto shapeHcFn = context_->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeHcFn);
    hcMix_ = shapeHcFn->GetStorageShape().GetDim(0);
    OP_CHECK_IF(shapeHcFn->GetStorageShape().GetDim(1) != d_ * hcMult_,
                OP_LOGE(context_->GetNodeName(), "HcFn dim 1 should be equal with d_ * hcMult_  %ld, but is %ld",
                        d_ * hcMult_, shapeHcFn->GetStorageShape().GetDim(1)),
                return ge::GRAPH_FAILED);

    auto shapeHcScale = context_->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeHcScale);
    int64_t scaleFirstDim = shapeHcScale->GetStorageShape().GetDim(0);
    OP_CHECK_IF(scaleFirstDim != 3,
                OP_LOGE(context_->GetNodeName(), "hc_scale size should be equal with 3, but is %ld", scaleFirstDim),
                return ge::GRAPH_FAILED);

    auto shapeHcBase = context_->GetInputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeHcBase);
    int64_t baseFirstDim = shapeHcBase->GetStorageShape().GetDim(0);
    OP_CHECK_IF(baseFirstDim != hcMix_,
                OP_LOGE(context_->GetNodeName(), "hc_base size should be equal with mixhc, but is %ld", baseFirstDim),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetAttr() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "get attr failed."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(hcMult_ != HCMULT_VALUE,
                OP_LOGE(context_->GetNodeName(), "hcMult should be %ld, but is %ld", HCMULT_VALUE, hcMult_),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(iterTimes_ != NUM_ITERS_VALUE,
                OP_LOGE(context_->GetNodeName(), "numIters should be %ld, but is %ld", NUM_ITERS_VALUE, iterTimes_),
                return ge::GRAPH_FAILED);

    OP_LOGD(context_->GetNodeName(),
            "MhcPreSinkhorn regbase shape attrs: bs=%ld, hcMix=%ld, hcMult=%ld, d=%ld, iterTimes=%ld, "
            "needGrad=%d, hcEps=%g, normEps=%g",
            bs_, hcMix_, hcMult_, d_, iterTimes_, needGrad_ ? 1 : 0, hcEps_, normEps_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornTilingRegbase::CalcRegbaseCommonTiling(GradOutMode gradOutMode, SplitMode splitMode)
{
    const bool withGradOut = gradOutMode == GradOutMode::ENABLED;
    const bool isKSplit = splitMode == SplitMode::K_SPLIT;
    CoreRowTiling coreRowTiling;
    OP_CHECK_IF(!CalcCoreRowTiling(bs_, static_cast<int64_t>(aivCoreNum_), coreRowTiling),
                OP_LOGE(context_->GetNodeName(), "Invalid row tiling input: bs=%ld, aivCoreNum=%lu", bs_, aivCoreNum_),
                return ge::GRAPH_FAILED);

    const int64_t hcMultAlign = RoundUp(hcMult_, BLOCK_SIZE / sizeof(float));
    // Use half of kL1Size so kUbSize stays below 256 when m reaches its maximum of 256.
    const int64_t kUbSize = tilingData_.get_kL1Size() / DOUBLE_BUFFER;
    const int64_t mUbSize = CeilDiv(tilingData_.get_mL1Size(), 2);
    const int64_t bufferPool0Size = static_cast<int64_t>(ubSize_);
    const int64_t bufferPool1Size =
        isKSplit ? 0 : CalcMOnlyBufferPool1Size(ubSize_, mUbSize, hcMix_, hcMult_, hcMultAlign);
    OP_CHECK_IF(!isKSplit && bufferPool1Size < 0,
                OP_LOGE(context_->GetNodeName(),
                        "M-split fixed buffers exceed UB: ubSize=%lu, bufferPool1Size=%ld, mUbSize=%ld, hcMix=%ld",
                        ubSize_, bufferPool1Size, mUbSize, hcMix_),
                return ge::GRAPH_FAILED);
    const int64_t availableUbSize = isKSplit ? bufferPool0Size : bufferPool1Size;

    UbBufferConfig bufferConfig;
    bufferConfig.hcMix = hcMix_;
    bufferConfig.hcMult = hcMult_;
    bufferConfig.hcMultAlign = hcMultAlign;
    bufferConfig.iterTimes = iterTimes_;
    bufferConfig.kBlockNum = tilingData_.get_cubeBlockDimK();
    bufferConfig.mUbSize = mUbSize;
    bufferConfig.withGradOut = withGradOut;
    bufferConfig.isKSplit = isKSplit;

    const auto fitsInUb = [&bufferConfig, availableUbSize](int64_t rowFactor, int64_t dFactor) {
        return CalcUbBufferSize(bufferConfig, rowFactor, dFactor) <= availableUbSize;
    };
    const int64_t maxRowFactor = isKSplit ? coreRowTiling.rowOfFormerBlock : mUbSize;
    RowAndDTiling rowAndDTiling;
    OP_CHECK_IF(!CalcRowAndDTiling(d_, maxRowFactor, fitsInUb, rowAndDTiling),
                OP_LOGE(context_->GetNodeName(),
                        "UB is insufficient for minimum tile: ubSize=%lu, availableUbSize=%ld, "
                        "withGradOut=%d, isKSplit=%d",
                        ubSize_, availableUbSize, withGradOut ? 1 : 0, isKSplit ? 1 : 0),
                return ge::GRAPH_FAILED);
    CompleteCoreRowTiling(coreRowTiling, rowAndDTiling.rowFactor);

    tilingData_.set_bs(bs_);
    tilingData_.set_hcMix(hcMix_);
    tilingData_.set_hcMult(hcMult_);
    tilingData_.set_d(d_);
    tilingData_.set_hcMultAlign(hcMultAlign);
    tilingData_.set_rowOfFormerBlock(coreRowTiling.rowOfFormerBlock);
    tilingData_.set_rowLoopOfFormerBlock(coreRowTiling.rowLoopOfFormerBlock);
    tilingData_.set_rowLoopOfTailBlock(coreRowTiling.rowLoopOfTailBlock);
    tilingData_.set_tailRowFactorOfFormerBlock(coreRowTiling.tailRowFactorOfFormerBlock);
    tilingData_.set_tailRowFactorOfTailBlock(coreRowTiling.tailRowFactorOfTailBlock);
    tilingData_.set_dLoop(rowAndDTiling.dLoop);
    tilingData_.set_dFactor(rowAndDTiling.dFactor);
    tilingData_.set_tailDFactor(rowAndDTiling.tailDFactor);
    tilingData_.set_iterTimes(iterTimes_);
    tilingData_.set_hcEps(hcEps_);
    tilingData_.set_normEps(normEps_);
    tilingData_.set_kUbSize(kUbSize);
    tilingData_.set_mUbSize(mUbSize);
    tilingData_.set_kBlockFactor(tilingData_.get_cubeBlockDimK());
    if (isKSplit) {
        tilingData_.set_stage2RowFactor(rowAndDTiling.rowFactor);
        tilingData_.set_secondUsedCoreNum(coreRowTiling.usedCoreNum);
    } else {
        tilingData_.set_rowInnerFactor(rowAndDTiling.rowFactor);
        tilingData_.set_bufferPool0Size(bufferPool0Size);
        tilingData_.set_bufferPool1Size(bufferPool1Size);
    }

    OP_LOGD(context_->GetNodeName(),
            "MhcPreSinkhorn regbase UB tiling: withGradOut=%d, isKSplit=%d, availableUbSize=%ld, "
            "maxRowFactor=%ld, rowFactor=%ld, dFactor=%ld, dLoop=%ld, tailDFactor=%ld",
            withGradOut ? 1 : 0, isKSplit ? 1 : 0, availableUbSize, maxRowFactor, rowAndDTiling.rowFactor,
            rowAndDTiling.dFactor, rowAndDTiling.dLoop, rowAndDTiling.tailDFactor);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornTilingRegbase::CalcRegbaseOpTiling()
{
    return CalcRegbaseCommonTiling(GradOutMode::DISABLED, SplitMode::M_SPLIT);
}

ge::graphStatus MhcPreSinkhornTilingRegbase::CalcRegbaseOpGradoutTiling()
{
    return CalcRegbaseCommonTiling(GradOutMode::ENABLED, SplitMode::M_SPLIT);
}

ge::graphStatus MhcPreSinkhornTilingRegbase::CalcMKSplitCorePart2Tiling()
{
    return CalcRegbaseCommonTiling(GradOutMode::DISABLED, SplitMode::K_SPLIT);
}

ge::graphStatus MhcPreSinkhornTilingRegbase::CalcMKSplitCorePart2GradoutTiling()
{
    return CalcRegbaseCommonTiling(GradOutMode::ENABLED, SplitMode::K_SPLIT);
}

ge::graphStatus MhcPreSinkhornTilingRegbase::CalcOpTiling()
{
    uint64_t kSize = hcMult_ * d_;
    tilingData_.set_k(kSize);
    // 计算bs_轴切核
    uint64_t mDimNum = std::min(aicCoreNum_, static_cast<uint64_t>(CeilDiv(bs_, M_L1_MAX_SIZE)));
    uint64_t singleCoreM = RoundUp(CeilDiv(bs_, mDimNum), AscendC::BLOCK_CUBE);
    uint64_t kDimNum = aicCoreNum_ / mDimNum;
    uint64_t originalKDimNum = kDimNum;

    // 如果开启了batch一致性开关，则强制走M分核模板
    int32_t deterministicLevel = context_->GetDeterministicLevel();
    OP_LOGD(context_->GetNodeName(), "ops: MhcPreSinkhorn, deterministic_level=%d", deterministicLevel);
    if (deterministicLevel == BATCH_CONSISTENCY_LEVEL) {
        OP_LOGI(context_->GetNodeName(),
                "MhcPreSinkhorn batch consistency enabled: deterministicLevel=%d, force kDimNum from %lu to 1.",
                deterministicLevel, originalKDimNum);
        kDimNum = 1;
    }

    uint64_t splitKSize = RoundUp(CeilDiv(kSize, kDimNum), K_MULIT_CORE_SPLIT_BASE_SIZE);
    uint64_t actualKBlockNum = CeilDiv(kSize, splitKSize);

    tilingData_.set_cubeBlockDimM(mDimNum);
    tilingData_.set_cubeBlockDimK(actualKBlockNum);
    tilingData_.set_mL1Size(std::min(M_L1_MAX_SIZE, singleCoreM));
    tilingData_.set_multCoreSplitKSize(splitKSize);
    // 开启batch一致性开关时mL1Size固定取M_L1_MAX_SIZE，使kL1Size不随实际bs_切分结果变化，保证在不同batch下切分一致
    uint64_t mSizeForKL1 = (deterministicLevel == BATCH_CONSISTENCY_LEVEL) ? M_L1_MAX_SIZE : tilingData_.get_mL1Size();
    tilingData_.set_kL1Size(std::min(A_L1_SIZE / mSizeForKL1, static_cast<uint64_t>(K_L1_MAX_SIZE)) / 128 * 128);

    bool isKSplit = kDimNum != 1;
    if (isKSplit) {
        tilingKey_ = needGrad_ ? TILING_KEY_GRAD_K_SPLIT : TILING_KEY_NO_GRAD_K_SPLIT;
    } else {
        tilingKey_ = needGrad_ ? TILING_KEY_GRAD_M_SPLIT : TILING_KEY_NO_GRAD_M_SPLIT;
    }
    OP_LOGD(context_->GetNodeName(),
            "MhcPreSinkhorn regbase tiling decision: tilingKey=%lu, needGrad=%d, deterministicLevel=%d, "
            "mDimNum=%lu, singleCoreM=%lu, originalKDimNum=%lu, finalKDimNum=%lu, actualKBlockNum=%lu, "
            "splitKSize=%lu, mL1Size=%ld, kL1Size=%ld",
            tilingKey_, needGrad_ ? 1 : 0, deterministicLevel, mDimNum, singleCoreM, originalKDimNum, kDimNum,
            actualKBlockNum, splitKSize, tilingData_.get_mL1Size(), tilingData_.get_kL1Size());
    if (isKSplit) {
        return needGrad_ ? CalcMKSplitCorePart2GradoutTiling() : CalcMKSplitCorePart2Tiling();
    }
    return needGrad_ ? CalcRegbaseOpGradoutTiling() : CalcRegbaseOpTiling();
}

ge::graphStatus MhcPreSinkhornTilingRegbase::DoOpTiling()
{
    if (GetPlatformInfo() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (GetShapeAttrsInfoInner() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (CalcOpTiling() == ge::GRAPH_FAILED) {
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

ge::graphStatus MhcPreSinkhornTilingRegbase::GetWorkspaceSize()
{
    workspaceSize_ = sysWorkspaceSize_;
    if (tilingKey_ == TILING_KEY_NO_GRAD_K_SPLIT || tilingKey_ == TILING_KEY_GRAD_K_SPLIT) {
        // K-split uses two float buffers after the system workspace.
        workspaceSize_ += tilingData_.get_kBlockFactor() * tilingData_.get_bs() * tilingData_.get_hcMix() *
                              sizeof(float) +
                          tilingData_.get_kBlockFactor() * tilingData_.get_bs() * sizeof(float);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornTilingRegbase::PostTiling()
{
    context_->SetTilingKey(tilingKey_);
    // 按实际参与计算的核数启动，避免多余核空转：
    // - M+K 分核模板(1000/2000)：stage1 需要 cubeBlockDimM * cubeBlockDimK 个 AIC 块；
    //    stage2 需要 secondUsedCoreNum 个 AIV，按 MIX 1 AIC:2 AIV 折算成块参与比较。
    uint64_t blockDim = aicCoreNum_;
    if (tilingKey_ == TILING_KEY_NO_GRAD_K_SPLIT || tilingKey_ == TILING_KEY_GRAD_K_SPLIT) {
        uint64_t stage1BlockNum =
            tilingData_.get_cubeBlockDimM() * tilingData_.get_cubeBlockDimK();
        uint64_t stage2BlockNum = CeilDiv(tilingData_.get_secondUsedCoreNum(), 2);
        blockDim = std::min(std::max(stage1BlockNum, stage2BlockNum), aicCoreNum_);
    }
    context_->SetBlockDim(blockDim);
    context_->SetScheduleMode(BATCH_MODE);
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    PrintFinalTilingData(context_, tilingData_, tilingKey_, blockDim, workspaceSize_);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
