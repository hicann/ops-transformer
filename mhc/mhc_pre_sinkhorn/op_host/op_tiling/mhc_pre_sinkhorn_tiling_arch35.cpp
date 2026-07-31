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

constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t REPEAT_SIZE = 256;
constexpr int64_t DOUBLE_BUFFER = 2;
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
            "MhcPreSinkhorn regbase final tiling core: rowOfFormerBlock=%ld, rowOfTailBlock=%ld, "
            "rowLoopOfFormerBlock=%ld, rowLoopOfTailBlock=%ld, rowFactor=%ld, stage2RowFactor=%ld, "
            "secondUsedCoreNum=%ld, rowInnerFactor=%ld",
            tilingData.get_rowOfFormerBlock(), tilingData.get_rowOfTailBlock(), tilingData.get_rowLoopOfFormerBlock(),
            tilingData.get_rowLoopOfTailBlock(), tilingData.get_rowFactor(), tilingData.get_stage2RowFactor(),
            tilingData.get_secondUsedCoreNum(), tilingData.get_rowInnerFactor());
    OP_LOGD(context->GetNodeName(),
            "MhcPreSinkhorn regbase final tiling tail: tailRowFactorOfFormerBlock=%ld, "
            "tailRowFactorOfTailBlock=%ld, dLoop=%ld, dFactor=%ld, tailDFactor=%ld",
            tilingData.get_tailRowFactorOfFormerBlock(), tilingData.get_tailRowFactorOfTailBlock(),
            tilingData.get_dLoop(), tilingData.get_dFactor(), tilingData.get_tailDFactor());
    OP_LOGD(context->GetNodeName(),
            "MhcPreSinkhorn regbase final tiling split: k=%ld, cubeBlockDimM=%ld, cubeBlockDimK=%ld, "
            "kBlockFactor=%ld, multCoreSplitMSize=%ld, multCoreSplitKSize=%ld, mL1Size=%ld, kL1Size=%ld, "
            "mUbSize=%ld, kUbSize=%ld, cvLoopKSize=%ld, bufferPool0Size=%ld, bufferPool1Size=%ld",
            tilingData.get_k(), tilingData.get_cubeBlockDimM(), tilingData.get_cubeBlockDimK(),
            tilingData.get_kBlockFactor(), tilingData.get_multCoreSplitMSize(), tilingData.get_multCoreSplitKSize(),
            tilingData.get_mL1Size(), tilingData.get_kL1Size(), tilingData.get_mUbSize(), tilingData.get_kUbSize(),
            tilingData.get_cvLoopKSize(), tilingData.get_bufferPool0Size(), tilingData.get_bufferPool1Size());
}
} // namespace

MhcPreSinkhornTilingRegbase::MhcPreSinkhornTilingRegbase(gert::TilingContext *tilingContext) : context_(tilingContext)
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
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aivCoreNum_ = ascendcPlatform.GetCoreNumAiv();
        aicCoreNum_ = ascendcPlatform.GetCoreNumAic();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize_ = ubSizePlatForm;
        socVersion_ = ascendcPlatform.GetSocVersion();
    }
    OP_LOGD(context_->GetNodeName(), "MhcPreSinkhorn regbase platform: aivCoreNum=%lu, aicCoreNum=%lu, ubSize=%lu",
            aivCoreNum_, aicCoreNum_, ubSize_);
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

ge::graphStatus MhcPreSinkhornTilingRegbase::CalcRegbaseOpTiling()
{
    rowOfFormerBlock_ = CeilDiv(bs_, static_cast<int64_t>(aivCoreNum_));
    usedCoreNums_ = std::min(CeilDiv(bs_, rowOfFormerBlock_), static_cast<int64_t>(aivCoreNum_));
    rowOfTailBlock_ = bs_ - (usedCoreNums_ - 1) * rowOfFormerBlock_;

    int64_t minRowPerCore = 1;
    int64_t rowOnceLoop = std::min(rowOfFormerBlock_, minRowPerCore);

    hcMultAlign_ = RoundUp(hcMult_, BLOCK_SIZE / sizeof(float));
    int64_t mixSize = rowOnceLoop * RoundUp(hcMix_, BLOCK_SIZE / sizeof(float)) * sizeof(float);
    int64_t xSize = rowOnceLoop * hcMult_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER; // x是bfloat16_t 类型
    int64_t ySize = rowOnceLoop * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER;
    int64_t postSize = rowOnceLoop * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
    int64_t combFragSize = rowOnceLoop * hcMult_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
    int64_t base0Size = hcMultAlign_ * sizeof(float);
    int64_t base1Size = hcMultAlign_ * sizeof(float);
    int64_t base2Size = hcMult_ * hcMultAlign_ * sizeof(float);

    uint64_t kUbSize = tilingData_.get_kL1Size() / 2; // 先按2倍系数计算，m最大256，需保证kub小于256
    uint64_t mUbSize = CeilDiv(tilingData_.get_mL1Size(), 2);

    int64_t mmXBufSize = mUbSize * RoundUp(hcMix_, BLOCK_SIZE / sizeof(float)) * sizeof(float);
    int64_t rmsNormBufSize = RoundUp(mUbSize, BLOCK_SIZE / sizeof(float)) * sizeof(float);
    int64_t bufferPool0Size = ubSize_;
    int64_t bufferPool1Size =
        DownAlign(bufferPool0Size - mmXBufSize - rmsNormBufSize - base0Size - base1Size - base2Size, BLOCK_SIZE);

    int64_t totalSize = mixSize + xSize + ySize + postSize + combFragSize;
    rowFactor_ = rowOnceLoop;
    if (totalSize <= bufferPool1Size) {
        // row和d均可以在ub内全载
        dLoop_ = 1;
        dFactor_ = d_;
        tailDFactor_ = dFactor_;
    } else {
        int64_t usedUbSize = mixSize + postSize + combFragSize;
        int64_t ubRemain = bufferPool1Size - usedUbSize;
        dFactor_ = d_;
        int64_t base = 2;
        while (1) {
            dFactor_ = CeilDiv(d_, base);
            xSize = rowOnceLoop * hcMult_ * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER; // x是bfloat16_t 类型
            ySize = rowOnceLoop * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
            int64_t targetSize = xSize + ySize;
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

    // d全载,尝试搬入更多的bs
    if (dFactor_ == d_) {
        while (rowFactor_ <= mUbSize) {
            mixSize = rowFactor_ * RoundUp(hcMix_, BLOCK_SIZE / sizeof(float)) * sizeof(float);
            xSize = rowFactor_ * hcMult_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER; // x是bfloat16_t 类型
            ySize = rowFactor_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER;
            postSize = rowFactor_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
            combFragSize = rowFactor_ * hcMult_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
            totalSize = mixSize + xSize + ySize + postSize + combFragSize;
            if (totalSize > bufferPool1Size) {
                rowFactor_ = rowFactor_ - 1;
                break;
            }
            rowFactor_ = rowFactor_ + 1;
        }
        rowFactor_ = rowFactor_ > mUbSize ? rowFactor_ - 1 : rowFactor_;
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
    tilingData_.set_rowOfTailBlock(rowOfTailBlock_);
    tilingData_.set_rowLoopOfFormerBlock(rowLoopOfFormerBlock_);
    tilingData_.set_rowLoopOfTailBlock(rowLoopOfTailBlock_);
    tilingData_.set_rowFactor(rowFactor_);
    tilingData_.set_tailRowFactorOfFormerBlock(tailRowFactorOfFormerBlock_);
    tilingData_.set_tailRowFactorOfTailBlock(tailRowFactorOfTailBlock_);
    tilingData_.set_dLoop(dLoop_);
    tilingData_.set_dFactor(dFactor_);
    tilingData_.set_tailDFactor(tailDFactor_);
    tilingData_.set_iterTimes(iterTimes_);
    tilingData_.set_hcEps(hcEps_);
    tilingData_.set_normEps(normEps_);

    tilingData_.set_bufferPool0Size(bufferPool0Size);
    tilingData_.set_bufferPool1Size(bufferPool1Size);

    tilingData_.set_kUbSize(kUbSize);
    tilingData_.set_mUbSize(mUbSize);

    tilingData_.set_kBlockFactor(tilingData_.get_cubeBlockDimK());

    tilingData_.set_rowInnerFactor(rowFactor_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornTilingRegbase::CalcRegbaseOpGradoutTiling()
{
    rowOfFormerBlock_ = CeilDiv(bs_, static_cast<int64_t>(aivCoreNum_));
    usedCoreNums_ = std::min(CeilDiv(bs_, rowOfFormerBlock_), static_cast<int64_t>(aivCoreNum_));
    rowOfTailBlock_ = bs_ - (usedCoreNums_ - 1) * rowOfFormerBlock_;

    int64_t minRowPerCore = 1;
    int64_t rowOnceLoop = std::min(rowOfFormerBlock_, minRowPerCore);

    hcMultAlign_ = RoundUp(hcMult_, BLOCK_SIZE / sizeof(float));
    int64_t mixSize = rowOnceLoop * RoundUp(hcMix_, BLOCK_SIZE / sizeof(float)) * sizeof(float);
    int64_t xSize = rowOnceLoop * hcMult_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER; // x是bfloat16_t 类型
    int64_t ySize = rowOnceLoop * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER;
    int64_t postSize = rowOnceLoop * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
    int64_t combFragSize = rowOnceLoop * hcMult_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
    // 新增 normOutSize sumOutSize invRmsOutSize
    int64_t normOutSize = 2 * iterTimes_ * rowOnceLoop * hcMult_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
    int64_t sumOutSize = 2 * iterTimes_ * rowOnceLoop * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
    int64_t invRmsOutSize = RoundUp(rowOnceLoop, BLOCK_SIZE / sizeof(float)) * sizeof(float) * DOUBLE_BUFFER;
    int64_t hcBeforeNormOutSize = CeilDiv(tilingData_.get_mL1Size(), 2) * RoundUp(hcMix_, BLOCK_SIZE / sizeof(float)) *
                                  sizeof(float) * DOUBLE_BUFFER;
    int64_t hPreOutSize = rowOnceLoop * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;

    int64_t base0Size = hcMultAlign_ * sizeof(float);
    int64_t base1Size = hcMultAlign_ * sizeof(float);
    int64_t base2Size = hcMult_ * hcMultAlign_ * sizeof(float);

    uint64_t kUbSize = tilingData_.get_kL1Size() / 2; // 先按2倍系数计算，m最大256，需保证kub小于256
    uint64_t mUbSize = CeilDiv(tilingData_.get_mL1Size(), 2);

    int64_t mmXBufSize = mUbSize * RoundUp(hcMix_, BLOCK_SIZE / sizeof(float)) * sizeof(float);
    int64_t rmsNormBufSize = RoundUp(mUbSize, BLOCK_SIZE / sizeof(float)) * sizeof(float);
    int64_t bufferPool0Size = ubSize_;
    int64_t bufferPool1Size =
        DownAlign(bufferPool0Size - mmXBufSize - rmsNormBufSize - base0Size - base1Size - base2Size, BLOCK_SIZE);

    int64_t totalSize = mixSize + xSize + ySize + postSize + combFragSize + normOutSize + sumOutSize + invRmsOutSize +
                        hcBeforeNormOutSize + hPreOutSize;
    rowFactor_ = rowOnceLoop;
    if (totalSize <= bufferPool1Size) {
        // row和d均可以在ub内全载
        dLoop_ = 1;
        dFactor_ = d_;
        tailDFactor_ = dFactor_;
    } else {
        int64_t usedUbSize = mixSize + postSize + combFragSize + normOutSize + sumOutSize + invRmsOutSize +
                             hcBeforeNormOutSize + hPreOutSize;
        int64_t ubRemain = bufferPool1Size - usedUbSize;
        dFactor_ = d_;
        int64_t base = 2;
        while (1) {
            dFactor_ = CeilDiv(d_, base);
            xSize = rowOnceLoop * hcMult_ * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER; // x是bfloat16_t 类型
            ySize = rowOnceLoop * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
            int64_t targetSize = xSize + ySize;
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

    // d全载,尝试搬入更多的bs
    if (dFactor_ == d_) {
        while (rowFactor_ <= mUbSize) {
            mixSize = rowFactor_ * RoundUp(hcMix_, BLOCK_SIZE / sizeof(float)) * sizeof(float);
            xSize = rowFactor_ * hcMult_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER; // x是bfloat16_t 类型
            ySize = rowFactor_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER;
            postSize = rowFactor_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
            combFragSize = rowFactor_ * hcMult_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
            normOutSize = 2 * iterTimes_ * rowFactor_ * hcMult_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
            sumOutSize = 2 * iterTimes_ * rowFactor_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
            invRmsOutSize = RoundUp(rowFactor_, BLOCK_SIZE / sizeof(float)) * sizeof(float) * DOUBLE_BUFFER;
            hcBeforeNormOutSize = mUbSize * RoundUp(hcMix_, BLOCK_SIZE / sizeof(float)) * sizeof(float) * DOUBLE_BUFFER;
            hPreOutSize = rowFactor_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
            totalSize = mixSize + xSize + ySize + postSize + combFragSize + normOutSize + sumOutSize + invRmsOutSize +
                        hcBeforeNormOutSize + hPreOutSize;
            if (totalSize > bufferPool1Size) {
                rowFactor_ = rowFactor_ - 1;
                break;
            }
            rowFactor_ = rowFactor_ + 1;
        }
        rowFactor_ = rowFactor_ > mUbSize ? rowFactor_ - 1 : rowFactor_;
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
    tilingData_.set_rowOfTailBlock(rowOfTailBlock_);
    tilingData_.set_rowLoopOfFormerBlock(rowLoopOfFormerBlock_);
    tilingData_.set_rowLoopOfTailBlock(rowLoopOfTailBlock_);
    tilingData_.set_rowFactor(rowFactor_);
    tilingData_.set_tailRowFactorOfFormerBlock(tailRowFactorOfFormerBlock_);
    tilingData_.set_tailRowFactorOfTailBlock(tailRowFactorOfTailBlock_);
    tilingData_.set_dLoop(dLoop_);
    tilingData_.set_dFactor(dFactor_);
    tilingData_.set_tailDFactor(tailDFactor_);
    tilingData_.set_iterTimes(iterTimes_);
    tilingData_.set_hcEps(hcEps_);
    tilingData_.set_normEps(normEps_);

    tilingData_.set_bufferPool0Size(bufferPool0Size);
    tilingData_.set_bufferPool1Size(bufferPool1Size);

    tilingData_.set_kUbSize(kUbSize);
    tilingData_.set_mUbSize(mUbSize);

    tilingData_.set_kBlockFactor(tilingData_.get_cubeBlockDimK());

    tilingData_.set_rowInnerFactor(rowFactor_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornTilingRegbase::CalcMKSplitCorePart2Tiling()
{
    uint64_t kUbSize = tilingData_.get_kL1Size() / 2; // 先按2倍系数计算，m最大256，需保证kub小于256
    uint64_t mUbSize = CeilDiv(tilingData_.get_mL1Size(), 2);

    rowOfFormerBlock_ = CeilDiv(bs_, static_cast<int64_t>(aivCoreNum_));
    usedAivCoreNums_ = std::min(CeilDiv(bs_, rowOfFormerBlock_), static_cast<int64_t>(aivCoreNum_));
    rowOfTailBlock_ = bs_ - (usedAivCoreNums_ - 1) * rowOfFormerBlock_;

    int64_t minRowPerCore = 1;
    int64_t rowOnceLoop = std::min(rowOfFormerBlock_, minRowPerCore);
    int64_t kBlockNum = tilingData_.get_cubeBlockDimK();

    hcMultAlign_ = RoundUp(hcMult_, BLOCK_SIZE / sizeof(float));
    uint64_t hcMixAlign = RoundUp(hcMix_, BLOCK_SIZE / sizeof(float));
    int64_t mmSize = kBlockNum * rowOnceLoop * hcMixAlign * sizeof(float) * DOUBLE_BUFFER;
    int64_t mixSize = rowOnceLoop * hcMixAlign * sizeof(float);
    int64_t rmsSize = kBlockNum * RoundUp(rowOnceLoop, BLOCK_SIZE / sizeof(float)) * sizeof(float) * DOUBLE_BUFFER;
    int64_t xSize = rowOnceLoop * hcMult_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER; // x是bfloat16_t 类型
    int64_t ySize = rowOnceLoop * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER;
    int64_t postSize = rowOnceLoop * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
    int64_t combFragSize = rowOnceLoop * hcMult_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
    int64_t base0Size = hcMultAlign_ * sizeof(float);
    int64_t base1Size = hcMultAlign_ * sizeof(float);
    int64_t base2Size = hcMult_ * hcMultAlign_ * sizeof(float);

    int64_t totalSize =
        mmSize + mixSize + rmsSize + xSize + ySize + postSize + combFragSize + base0Size + base1Size + base2Size;
    rowFactor_ = rowOnceLoop;
    if (totalSize <= ubSize_) {
        // row和d均可以在ub内全载
        dLoop_ = 1;
        dFactor_ = d_;
        tailDFactor_ = dFactor_;
    } else {
        int64_t usedUbSize = mmSize + mixSize + rmsSize + postSize + combFragSize + base0Size + base1Size + base2Size;
        int64_t ubRemain = ubSize_ - usedUbSize;
        dFactor_ = d_;
        int64_t base = 2;
        while (1) {
            dFactor_ = CeilDiv(d_, base);
            xSize = rowOnceLoop * hcMult_ * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER; // x是bfloat16_t 类型
            ySize = rowOnceLoop * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
            int64_t targetSize = xSize + ySize;
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

    // d全载,尝试搬入更多的bs
    if (dFactor_ == d_) {
        while (rowFactor_ <= rowOfFormerBlock_) {
            mmSize = kBlockNum * rowFactor_ * hcMixAlign * sizeof(float) * DOUBLE_BUFFER;
            mixSize = rowFactor_ * hcMixAlign * sizeof(float);
            rmsSize = kBlockNum * RoundUp(rowFactor_, BLOCK_SIZE / sizeof(float)) * sizeof(float) * DOUBLE_BUFFER;
            xSize = rowFactor_ * hcMult_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER; // x是bfloat16_t 类型
            ySize = rowFactor_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER;
            postSize = rowFactor_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
            combFragSize = rowFactor_ * hcMult_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
            base0Size = hcMultAlign_ * sizeof(float);
            base1Size = hcMultAlign_ * sizeof(float);
            base2Size = hcMult_ * hcMultAlign_ * sizeof(float);

            totalSize = mmSize + mixSize + rmsSize + xSize + ySize + postSize + combFragSize + base0Size + base1Size +
                        base2Size;
            if (totalSize > ubSize_) {
                rowFactor_ = rowFactor_ - 1;
                break;
            }
            rowFactor_ = rowFactor_ + 1;
        }
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
    tilingData_.set_rowOfTailBlock(rowOfTailBlock_);
    tilingData_.set_rowLoopOfFormerBlock(rowLoopOfFormerBlock_);
    tilingData_.set_rowLoopOfTailBlock(rowLoopOfTailBlock_);
    tilingData_.set_stage2RowFactor(rowFactor_);
    tilingData_.set_secondUsedCoreNum(usedAivCoreNums_);
    tilingData_.set_tailRowFactorOfFormerBlock(tailRowFactorOfFormerBlock_);
    tilingData_.set_tailRowFactorOfTailBlock(tailRowFactorOfTailBlock_);
    tilingData_.set_dLoop(dLoop_);
    tilingData_.set_dFactor(dFactor_);
    tilingData_.set_tailDFactor(tailDFactor_);
    tilingData_.set_iterTimes(iterTimes_);
    tilingData_.set_hcEps(hcEps_);
    tilingData_.set_normEps(normEps_);
    tilingData_.set_kUbSize(kUbSize);
    tilingData_.set_mUbSize(mUbSize);
    tilingData_.set_kBlockFactor(tilingData_.get_cubeBlockDimK());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornTilingRegbase::CalcMKSplitCorePart2GradoutTiling()
{
    uint64_t kUbSize = tilingData_.get_kL1Size() / 2; // 先按2倍系数计算，m最大256，需保证kub小于256
    uint64_t mUbSize = CeilDiv(tilingData_.get_mL1Size(), 2);

    rowOfFormerBlock_ = CeilDiv(bs_, static_cast<int64_t>(aivCoreNum_));
    usedAivCoreNums_ = std::min(CeilDiv(bs_, rowOfFormerBlock_), static_cast<int64_t>(aivCoreNum_));
    rowOfTailBlock_ = bs_ - (usedAivCoreNums_ - 1) * rowOfFormerBlock_;

    int64_t minRowPerCore = 1;
    int64_t rowOnceLoop = std::min(rowOfFormerBlock_, minRowPerCore);
    int64_t kBlockNum = tilingData_.get_cubeBlockDimK();

    hcMultAlign_ = RoundUp(hcMult_, BLOCK_SIZE / sizeof(float));
    uint64_t hcMixAlign = RoundUp(hcMix_, BLOCK_SIZE / sizeof(float));
    int64_t mmSize = kBlockNum * rowOnceLoop * hcMixAlign * sizeof(float) * DOUBLE_BUFFER;
    int64_t mixSize = rowOnceLoop * hcMixAlign * sizeof(float);
    int64_t rmsSize = kBlockNum * RoundUp(rowOnceLoop, BLOCK_SIZE / sizeof(float)) * sizeof(float) * DOUBLE_BUFFER;
    int64_t xSize = rowOnceLoop * hcMult_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER; // x是bfloat16_t 类型
    int64_t ySize = rowOnceLoop * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER;
    int64_t postSize = rowOnceLoop * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
    int64_t combFragSize = rowOnceLoop * hcMult_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
    int64_t base0Size = hcMultAlign_ * sizeof(float);
    int64_t base1Size = hcMultAlign_ * sizeof(float);
    int64_t base2Size = hcMult_ * hcMultAlign_ * sizeof(float);
    // 新增 normOutSize sumOutSize invRmsOutSize hcBeforeNormSize hPreSize
    int64_t normOutSize = 2 * iterTimes_ * rowOnceLoop * hcMult_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
    int64_t sumOutSize = 2 * iterTimes_ * rowOnceLoop * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
    int64_t invRmsOutSize = RoundUp(rowOnceLoop, BLOCK_SIZE / sizeof(float)) * sizeof(float) * DOUBLE_BUFFER;
    int64_t hcBeforeNormSize = rowOnceLoop * hcMixAlign * sizeof(float) * DOUBLE_BUFFER;
    int64_t hPreSize = rowOnceLoop * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;

    int64_t totalSize = mmSize + mixSize + rmsSize + xSize + ySize + postSize + combFragSize + base0Size + base1Size +
                        base2Size + normOutSize + sumOutSize + invRmsOutSize + hcBeforeNormSize + hPreSize;
    rowFactor_ = rowOnceLoop;
    if (totalSize <= ubSize_) {
        // row和d均可以在ub内全载
        dLoop_ = 1;
        dFactor_ = d_;
        tailDFactor_ = dFactor_;
    } else {
        int64_t usedUbSize = mmSize + mixSize + rmsSize + postSize + combFragSize + base0Size + base1Size + base2Size +
                             normOutSize + sumOutSize + invRmsOutSize + hcBeforeNormSize + hPreSize;
        int64_t ubRemain = ubSize_ - usedUbSize;
        dFactor_ = d_;
        int64_t base = 2;
        while (1) {
            dFactor_ = CeilDiv(d_, base);
            xSize = rowOnceLoop * hcMult_ * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER; // x是bfloat16_t 类型
            ySize = rowOnceLoop * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
            int64_t targetSize = xSize + ySize;
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

    // d全载,尝试搬入更多的bs
    if (dFactor_ == d_) {
        while (rowFactor_ <= rowOfFormerBlock_) {
            mmSize = kBlockNum * rowFactor_ * hcMixAlign * sizeof(float) * DOUBLE_BUFFER;
            mixSize = rowFactor_ * hcMixAlign * sizeof(float);
            rmsSize = kBlockNum * RoundUp(rowFactor_, BLOCK_SIZE / sizeof(float)) * sizeof(float) * DOUBLE_BUFFER;
            xSize = rowFactor_ * hcMult_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER; // x是bfloat16_t 类型
            ySize = rowFactor_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER;
            postSize = rowFactor_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
            combFragSize = rowFactor_ * hcMult_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
            base0Size = hcMultAlign_ * sizeof(float);
            base1Size = hcMultAlign_ * sizeof(float);
            base2Size = hcMult_ * hcMultAlign_ * sizeof(float);
            // 新增 normOutSize sumOutSize invRmsOutSize hcBeforeNormSize
            normOutSize = 2 * iterTimes_ * rowFactor_ * hcMult_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
            sumOutSize = 2 * iterTimes_ * rowFactor_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
            invRmsOutSize = RoundUp(rowFactor_, BLOCK_SIZE / sizeof(float)) * sizeof(float) * DOUBLE_BUFFER;
            hcBeforeNormSize = rowFactor_ * hcMixAlign * sizeof(float) * DOUBLE_BUFFER;
            hPreSize = rowFactor_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;

            totalSize = mmSize + mixSize + rmsSize + xSize + ySize + postSize + combFragSize + base0Size + base1Size +
                        base2Size + normOutSize + sumOutSize + invRmsOutSize + hcBeforeNormSize + hPreSize;
            if (totalSize > ubSize_) {
                rowFactor_ = rowFactor_ - 1;
                break;
            }
            rowFactor_ = rowFactor_ + 1;
        }
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
    tilingData_.set_rowOfTailBlock(rowOfTailBlock_);
    tilingData_.set_rowLoopOfFormerBlock(rowLoopOfFormerBlock_);
    tilingData_.set_rowLoopOfTailBlock(rowLoopOfTailBlock_);
    tilingData_.set_stage2RowFactor(rowFactor_);
    tilingData_.set_secondUsedCoreNum(usedAivCoreNums_);
    tilingData_.set_tailRowFactorOfFormerBlock(tailRowFactorOfFormerBlock_);
    tilingData_.set_tailRowFactorOfTailBlock(tailRowFactorOfTailBlock_);
    tilingData_.set_dLoop(dLoop_);
    tilingData_.set_dFactor(dFactor_);
    tilingData_.set_tailDFactor(tailDFactor_);
    tilingData_.set_iterTimes(iterTimes_);
    tilingData_.set_hcEps(hcEps_);
    tilingData_.set_normEps(normEps_);
    tilingData_.set_kUbSize(kUbSize);
    tilingData_.set_mUbSize(mUbSize);
    tilingData_.set_kBlockFactor(tilingData_.get_cubeBlockDimK());
    return ge::GRAPH_SUCCESS;
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
    tilingData_.set_multCoreSplitMSize(singleCoreM); // todo: 这个 tiling 根本没有使用，是否需要删掉
    tilingData_.set_mL1Size(std::min(M_L1_MAX_SIZE, singleCoreM));
    tilingData_.set_multCoreSplitKSize(splitKSize);
    tilingData_.set_kL1Size(std::min(A_L1_SIZE / tilingData_.get_mL1Size(), static_cast<uint64_t>(K_L1_MAX_SIZE)) /
                            128 * 128);

    tilingData_.set_cvLoopKSize(1024);
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
    if (tilingKey_ == TILING_KEY_NO_GRAD_K_SPLIT || tilingKey_ == TILING_KEY_GRAD_K_SPLIT) {
        // K分核模板需要预留Workspace大小
        workspaceSize_ = tilingData_.get_kBlockFactor() * tilingData_.get_bs() * tilingData_.get_hcMix() * 4 +
                         tilingData_.get_kBlockFactor() * tilingData_.get_bs() * 4 + 16 * 1024 * 1024;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornTilingRegbase::PostTiling()
{
    context_->SetTilingKey(tilingKey_);
    context_->SetBlockDim(aicCoreNum_);
    context_->SetScheduleMode(BATCH_MODE);
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    PrintFinalTilingData(context_, tilingData_, tilingKey_, aicCoreNum_, workspaceSize_);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
