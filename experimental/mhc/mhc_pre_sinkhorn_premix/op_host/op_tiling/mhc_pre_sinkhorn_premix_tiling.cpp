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
 * \file mmhc_pre_sinkhorn_premix_sinkhorn_tiling.cpp
 * \brief MhcPreSinkhornPremix tiling implementation
 */

#include <functional>
#include <sstream>
#include "mhc_pre_sinkhorn_premix_tiling.h"
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

// 在 [lo, hi] 内二分查找使 sizeFn(x) <= limit 成立的最大 x; 无可行值时返回 lo - 1
int64_t FindMaxFit(int64_t lo, int64_t hi, int64_t limit, const std::function<int64_t(int64_t)>& sizeFn)
{
    int64_t best = lo - 1;
    while (lo <= hi) {
        int64_t mid = lo + (hi - lo) / 2;
        if (sizeFn(mid) <= limit) {
            best = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return best;
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
constexpr int64_t UB_RESEVED_SIZE = 8192;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr uint64_t M_L1_MAX_SIZE = 256;
constexpr uint64_t K_MULIT_CORE_SPLIT_BASE_SIZE = 256;
constexpr uint64_t A_L1_SIZE = 128 * 256;
constexpr uint64_t K_L1_MAX_SIZE = 1024;
constexpr int64_t HC_MULT_ATTR_IDX = 0;
constexpr int64_t ITER_TIMES_ATTR_IDX = 1;
constexpr int64_t HC_EPS_ATTR_IDX = 2;
constexpr int64_t NORM_EPS_ATTR_IDX = 3;
constexpr int64_t NEED_BACKWARD_ATTR_IDX = 4;
constexpr int64_t HCMULT_VALUE = 4;
constexpr int64_t DEFAULT_ITER_TIMES = 20;
constexpr int64_t MAX_BS_PER_LOOP = 32;
constexpr int64_t MAX_REDUCE_SIZE = 256;
constexpr uint64_t CV_LOOP_K_SIZE = 1024;
constexpr int64_t K_L1_ALIGN_SIZE = 128;
constexpr int64_t SQUARE_SUM_SIZE = 16;
constexpr int64_t MASK_PATTERN_DIM_SIZE = 16;
constexpr int64_t PRE_POST_ROW_WORK_SCALE = 3;
constexpr int64_t SINKHORN_ROW_WORK_SCALE = 64;
constexpr int64_t SINKHORN_SETUP_ITER_EQUIVALENT = 4;
constexpr int64_t SMALL_BS_THRESHOLD = 32;
constexpr uint64_t MPMD_TILING_KEY = 0;
constexpr uint64_t ORIGINAL_TILING_KEY = 1;
// 切M路径：B矩阵走 MatmulImpl，不受 nd2nz srcDValue 上限约束
constexpr uint64_t MSPLIT_TILING_KEY = 2;
// Nd2NzParams.srcDValue 为 uint16，取值范围 [1, 65535]；K(=hcMult*d) 超限时
// 切K路径的 B1 搬运行距溢出，强制走切M分支
constexpr int64_t MAX_ND2NZ_SRC_DVALUE = 65535;
} // namespace

ge::graphStatus MhcPreSinkhornPremixTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<MhcPreSinkhornPremixCompileInfo>();
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
        aivCoreNum_ = compileInfoPtr->coreNum;
        aicCoreNum_ = compileInfoPtr->aicCoreNum;
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
    OP_CHECK_IF(aicCoreNum_ == 0, OP_LOGE(context_, "aicCoreNum is 0, cannot do tiling"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(aivCoreNum_ == 0, OP_LOGE(context_, "aivCoreNum_ is 0, cannot do tiling"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornPremixTiling::GetAttr()
{
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    auto hcMultAttr = attrs->GetAttrPointer<int64_t>(HC_MULT_ATTR_IDX);
    hcMult_ = hcMultAttr == nullptr ? 4 : *hcMultAttr;

    auto iterTimesAttr = attrs->GetAttrPointer<int64_t>(ITER_TIMES_ATTR_IDX);
    iterTimes_ = iterTimesAttr == nullptr ? DEFAULT_ITER_TIMES : *iterTimesAttr;

    auto hcEpsAttr = attrs->GetAttrPointer<float>(HC_EPS_ATTR_IDX);
    hcEps_ = hcEpsAttr == nullptr ? 1e-6f : *hcEpsAttr;

    auto normEpsAttr = attrs->GetAttrPointer<float>(NORM_EPS_ATTR_IDX);
    normEps_ = normEpsAttr == nullptr ? 1e-6f : *normEpsAttr;

    auto needBackwardAttr = attrs->GetAttrPointer<bool>(NEED_BACKWARD_ATTR_IDX);
    needGrad_ = needBackwardAttr == nullptr ? true : *needBackwardAttr;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornPremixTiling::GetShapeAttrsInfoInner()
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

    auto shapePhi = context_->GetInputShape(1);
    hcMix_ = shapePhi->GetStorageShape().GetDim(0);
    OP_CHECK_IF(shapePhi->GetStorageShape().GetDim(1) != d_ * hcMult_,
                OP_LOGE(context_->GetNodeName(), "Phi dim 1 should be equal with d_ * hcMult_  %ld, but is %ld",
                        d_ * hcMult_, shapePhi->GetStorageShape().GetDim(1)),
                return ge::GRAPH_FAILED);

    auto shapeAlpha = context_->GetInputShape(2);
    int64_t scaleFirstDim = shapeAlpha->GetStorageShape().GetDim(0);
    OP_CHECK_IF(scaleFirstDim != 3,
                OP_LOGE(context_->GetNodeName(), "alpha size should be equal with 3, but is %ld", scaleFirstDim),
                return ge::GRAPH_FAILED);

    auto shapeBias = context_->GetInputShape(3);
    int64_t baseFirstDim = shapeBias->GetStorageShape().GetDim(0);
    OP_CHECK_IF(baseFirstDim != hcMix_,
                OP_LOGE(context_->GetNodeName(), "bias size should be equal with mixhc, but is %ld", baseFirstDim),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetAttr() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "get attr failed."),
                return ge::GRAPH_FAILED);
    // hcMult only support 4
    OP_CHECK_IF(hcMult_ != HCMULT_VALUE,
                OP_LOGE(context_->GetNodeName(), "hcMult should be %ld, but is %ld", HCMULT_VALUE, hcMult_),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void MhcPreSinkhornPremixTiling::CalcVectorCoreNum()
{
    int64_t totalCoreNum = static_cast<int64_t>(aivCoreNum_);
    int64_t effectiveIterTimes = std::max(iterTimes_, static_cast<int64_t>(1));

    // Profiling gives a 2:1 row cost at d=4096, hcMult=4 and 20 iterations.
    // Enumerating every split also accounts for row-rounding cliffs at different batch sizes.
    uint64_t prePostRowWork = static_cast<uint64_t>(PRE_POST_ROW_WORK_SCALE) * d_;
    uint64_t sinkhornRowWork = static_cast<uint64_t>(SINKHORN_ROW_WORK_SCALE) * hcMult_ *
                               (effectiveIterTimes + SINKHORN_SETUP_ITER_EQUIVALENT);
    uint64_t bestLatency = 0;
    uint64_t bestImbalance = 0;
    bool hasCandidate = false;

    for (int64_t candidatePrePostCoreNum = 1; candidatePrePostCoreNum < totalCoreNum; candidatePrePostCoreNum++) {
        int64_t candidateSinkhornCoreNum = totalCoreNum - candidatePrePostCoreNum;
        uint64_t prePostLatency = static_cast<uint64_t>(CeilDiv(bs_, candidatePrePostCoreNum)) * prePostRowWork;
        uint64_t sinkhornLatency = static_cast<uint64_t>(CeilDiv(bs_, candidateSinkhornCoreNum)) * sinkhornRowWork;
        uint64_t latency = std::max(prePostLatency, sinkhornLatency);
        uint64_t imbalance =
            prePostLatency > sinkhornLatency ? prePostLatency - sinkhornLatency : sinkhornLatency - prePostLatency;
        if (!hasCandidate || latency < bestLatency || (latency == bestLatency && imbalance < bestImbalance)) {
            hasCandidate = true;
            bestLatency = latency;
            bestImbalance = imbalance;
            prePostCoreNum_ = candidatePrePostCoreNum;
            sinkhornCoreNum_ = candidateSinkhornCoreNum;
        }
    }
}

ge::graphStatus MhcPreSinkhornPremixTiling::CalcMKSplitCoreMembasePart2Tiling()
{
    // needGrad 输出(sumOut/normOut/hcBeforeNorm)依赖 original 路径的中间量,
    // MPMD 的 Compact Sinkhorn 归约中间态为 element-major 布局, 无法低成本落盘,
    // 因此 needGrad 场景强制走 original(KEY(1))。
    // K(=hcMult*d) 超出 nd2nz srcDValue 上限(65535)时, 切K路径 CopyInB1 的搬运行距
    // 溢出 uint16, 强制走切M(KEY(2)); Stage2 沿用 original 的分核/分片字段。
    bool kExceedNd2nzLimit = static_cast<int64_t>(hcMult_) * d_ > MAX_ND2NZ_SRC_DVALUE;
    bool useOriginal = kExceedNd2nzLimit || bs_ <= SMALL_BS_THRESHOLD || needGrad_;
    if (useOriginal) {
        tilingKey_ = kExceedNd2nzLimit ? MSPLIT_TILING_KEY : ORIGINAL_TILING_KEY;
        rowOfFormerBlock_ = CeilDiv(bs_, static_cast<int64_t>(aivCoreNum_));
    } else {
        tilingKey_ = MPMD_TILING_KEY;
        OP_CHECK_IF(aivCoreNum_ < 2,
                    OP_LOGE(context_->GetNodeName(),
                            "MhcPreSinkhornPremix MPMD requires at least two vector cores, but got %lu", aivCoreNum_),
                    return ge::GRAPH_FAILED);
        CalcVectorCoreNum();
        rowOfFormerBlock_ = CeilDiv(bs_, prePostCoreNum_);
    }

    int64_t stage2CoreNum = useOriginal ? static_cast<int64_t>(aivCoreNum_) : prePostCoreNum_;
    usedAivCoreNums_ = std::min(CeilDiv(bs_, rowOfFormerBlock_), stage2CoreNum);
    rowOfTailBlock_ = bs_ - (usedAivCoreNums_ - 1) * rowOfFormerBlock_;

    int64_t minRowPerCore = 1;
    int64_t rowOnceLoop = std::min(rowOfFormerBlock_, minRowPerCore);
    int64_t kBlockNum = tilingData_.get_cubeBlockDimK();

    hcMultAlign_ = RoundUp(hcMult_, BLOCK_SIZE / sizeof(float));
    int64_t mix0OriginSize = kBlockNum * rowOnceLoop * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
    int64_t mix1OriginSize = kBlockNum * rowOnceLoop * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
    int64_t mix0Size = rowOnceLoop * hcMultAlign_ * sizeof(float);
    int64_t mix1Size = rowOnceLoop * hcMultAlign_ * sizeof(float);
    int64_t squareSumSize = kBlockNum * RoundUp(rowOnceLoop * 16, 16) * sizeof(float) * DOUBLE_BUFFER;
    int64_t squareReduceSize = rowOnceLoop * SQUARE_SUM_SIZE * sizeof(float);
    int64_t rsqrtSize = RoundUp(rowOnceLoop, BLOCK_SIZE / sizeof(float)) * sizeof(float);
    int64_t xSize = rowOnceLoop * hcMult_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER; // x是bfloat16_t 类型
    int64_t ySize = rowOnceLoop * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER;
    int64_t postSize = rowOnceLoop * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
    int64_t base0Size = hcMultAlign_ * sizeof(float);
    int64_t base1Size = hcMultAlign_ * sizeof(float);
    int64_t xCastSize = rowOnceLoop * hcMult_ * RoundUp(d_, 8) * sizeof(float);
    int64_t yCastSize = rowOnceLoop * RoundUp(d_, 8) * sizeof(float);
    int64_t rowBrcb0Size = RoundUp(rowOnceLoop, 8) * BLOCK_SIZE;
    int64_t hcBrcb1Size = RoundUp(rowOnceLoop * hcMultAlign_, 8) * BLOCK_SIZE;
    int64_t maskPatternSize = (rowOnceLoop + MASK_PATTERN_DIM_SIZE - 1) * BLOCK_SIZE;
    auto calcOriginalExtraSize = [&](int64_t rowFactor) {
        if (!useOriginal) {
            return static_cast<int64_t>(0);
        }
        int64_t mix2OriginSize = kBlockNum * rowFactor * hcMult_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
        int64_t combFragSize = rowFactor * hcMult_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
        int64_t base2Size = hcMult_ * hcMultAlign_ * sizeof(float);
        int64_t reduceBufSize = rowFactor * hcMultAlign_ * sizeof(float);
        int64_t mix2Size = rowFactor * hcMult_ * hcMultAlign_ * sizeof(float);
        return mix2OriginSize + combFragSize + base2Size + reduceBufSize + mix2Size;
    };
    int64_t originalExtraSize = calcOriginalExtraSize(rowOnceLoop);

    int64_t totalSize = mix0OriginSize + mix1OriginSize + mix0Size + mix1Size + squareSumSize + squareReduceSize +
                        rsqrtSize + xSize + ySize + postSize + base0Size + base1Size + xCastSize + yCastSize +
                        rowBrcb0Size + hcBrcb1Size + maskPatternSize + originalExtraSize;
    rowFactor_ = rowOnceLoop;
    if (totalSize <= ubSize_) {
        // row和d均可以在ub内全载
        dLoop_ = 1;
        dFactor_ = d_;
        tailDFactor_ = dFactor_;
    } else {
        int64_t usedUbSize = mix0OriginSize + mix1OriginSize + mix0Size + mix1Size + squareSumSize + squareReduceSize +
                             rsqrtSize + postSize + base0Size + base1Size + rowBrcb0Size + hcBrcb1Size +
                             maskPatternSize + originalExtraSize;
        int64_t ubRemain = ubSize_ - usedUbSize;
        // 二分查找最小的 base 使 x/y/xCast/yCast 缓冲不超过 ubRemain (等价原首个可行 base, dFactor 尽量大)
        auto calcShrinkSize = [&](int64_t b) {
            int64_t df = CeilDiv(d_, b);
            int64_t xs = rowOnceLoop * hcMult_ * RoundUp(df, 16) * 2 * DOUBLE_BUFFER; // x是bfloat16_t 类型
            int64_t ys = rowOnceLoop * RoundUp(df, 16) * 2 * DOUBLE_BUFFER;
            int64_t xcs = rowOnceLoop * hcMult_ * RoundUp(df, 8) * sizeof(float);
            int64_t ycs = rowOnceLoop * RoundUp(df, 8) * sizeof(float);
            return xs + ys + xcs + ycs;
        };
        int64_t lo = 2, hi = d_, base = d_ + 1;
        while (lo <= hi) {
            int64_t mid = lo + (hi - lo) / 2;
            if (calcShrinkSize(mid) <= ubRemain) {
                base = mid;
                hi = mid - 1;
            } else {
                lo = mid + 1;
            }
        }
        dFactor_ = CeilDiv(d_, base);
        if (dFactor_ > 32) {
            dFactor_ = DownAlign(dFactor_, 32);
        }
        dLoop_ = CeilDiv(d_, dFactor_);
        tailDFactor_ = d_ % dFactor_ == 0 ? dFactor_ : d_ % dFactor_;
    }

    // d全载,尝试搬入更多的bs: 二分查找 [1, rowOfFormerBlock_] 内使总缓冲不超过 ubSize_ 的最大 rowFactor
    if (dFactor_ == d_) {
        auto updateRowSizes = [&](int64_t rf) {
            mix0OriginSize = kBlockNum * rf * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
            mix1OriginSize = kBlockNum * rf * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
            mix0Size = rf * hcMultAlign_ * sizeof(float);
            mix1Size = rf * hcMultAlign_ * sizeof(float);
            squareSumSize = kBlockNum * RoundUp(rf * 16, 16) * sizeof(float) * DOUBLE_BUFFER;
            squareReduceSize = rf * SQUARE_SUM_SIZE * sizeof(float);
            rsqrtSize = RoundUp(rf, BLOCK_SIZE / sizeof(float)) * sizeof(float);
            xSize = rf * hcMult_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER; // x是bfloat16_t 类型
            ySize = rf * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER;
            postSize = rf * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
            xCastSize = rf * hcMult_ * RoundUp(d_, 8) * sizeof(float);
            yCastSize = rf * RoundUp(d_, 8) * sizeof(float);
            rowBrcb0Size = RoundUp(rf, 8) * BLOCK_SIZE;
            hcBrcb1Size = RoundUp(rf * hcMultAlign_, 8) * BLOCK_SIZE;
            maskPatternSize = (rf + MASK_PATTERN_DIM_SIZE - 1) * BLOCK_SIZE;
            originalExtraSize = calcOriginalExtraSize(rf);
            return mix0OriginSize + mix1OriginSize + mix0Size + mix1Size + squareSumSize + squareReduceSize +
                   rsqrtSize + xSize + ySize + postSize + base0Size + base1Size + xCastSize + yCastSize + rowBrcb0Size +
                   hcBrcb1Size + maskPatternSize + originalExtraSize;
        };
        rowFactor_ = FindMaxFit(1, rowOfFormerBlock_, ubSize_, updateRowSizes);
    }

    // original 路径(ProcessOriginal)的 ProcessY 系数行距/softmax/sumOut 链路仅在
    // rf==1 的逐行布局下验证过(hc_pre 蓝本 original 仅运行于 bs<=32 即 rfb==1),
    // 强制收敛避免 rf>1 的 UB 布局错位; MPMD 的 prePost/sinkhornRowFactor 独立枚举不受影响
    if (useOriginal) {
        rowFactor_ = 1;
    }

    // The pre/post transforms only need row-sized state; y still uses the
    // full-D buffers above. Give pre/post its own row tile so a full-D shape
    // does not force sigmoid and RMSNorm setup to run once per row.
    int64_t prePostRowFactor = rowFactor_;
    if (!useOriginal) {
        auto calcPrePostSize = [&](int64_t candidateRow) {
            int64_t originSize = kBlockNum * candidateRow * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
            int64_t reduceSize = candidateRow * hcMultAlign_ * sizeof(float);
            int64_t squareSize = kBlockNum * RoundUp(candidateRow * 16, 16) * sizeof(float) * DOUBLE_BUFFER;
            int64_t squareReduce = candidateRow * SQUARE_SUM_SIZE * sizeof(float);
            int64_t rsqrt = RoundUp(candidateRow, BLOCK_SIZE / sizeof(float)) * sizeof(float);
            int64_t post = rowFactor_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
            int64_t rowBrcb = RoundUp(candidateRow, BLOCK_SIZE / sizeof(float)) * BLOCK_SIZE;
            int64_t hcBrcb = RoundUp(candidateRow * hcMultAlign_, BLOCK_SIZE / sizeof(float)) * BLOCK_SIZE;
            int64_t mask = (candidateRow + MASK_PATTERN_DIM_SIZE - 1) * BLOCK_SIZE;
            int64_t yX = rowFactor_ * hcMult_ * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
            int64_t yY = rowFactor_ * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
            int64_t yXCast = rowFactor_ * hcMult_ * RoundUp(dFactor_, 8) * sizeof(float);
            int64_t yYCast = rowFactor_ * RoundUp(dFactor_, 8) * sizeof(float);
            return originSize * 2 + reduceSize * 2 + squareSize + squareReduce + rsqrt + post + base0Size + base1Size +
                   yX + yY + yXCast + yYCast + rowBrcb + hcBrcb + mask;
        };
        prePostRowFactor = 1;
        for (int64_t candidateRow = 1; candidateRow <= rowOfFormerBlock_; candidateRow++) {
            if (calcPrePostSize(candidateRow) > static_cast<int64_t>(ubSize_)) {
                break;
            }
            prePostRowFactor = candidateRow;
        }
        // Keep the batched row path conservative until the target UB layout
        // is fully characterized; four rows already amortize the row setup.
        if (prePostRowFactor > 4) {
            prePostRowFactor = 4;
        }
    }
    int64_t prePostRowLoopOfFormerBlock = CeilDiv(rowOfFormerBlock_, prePostRowFactor);
    int64_t prePostRowLoopOfTailBlock = CeilDiv(rowOfTailBlock_, prePostRowFactor);
    int64_t prePostTailRowFactorOfFormerBlock =
        rowOfFormerBlock_ % prePostRowFactor == 0 ? prePostRowFactor : rowOfFormerBlock_ % prePostRowFactor;
    int64_t prePostTailRowFactorOfTailBlock =
        rowOfTailBlock_ % prePostRowFactor == 0 ? prePostRowFactor : rowOfTailBlock_ % prePostRowFactor;
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
    tilingData_.set_prePostCoreNum(prePostCoreNum_);
    tilingData_.set_sinkhornCoreNum(sinkhornCoreNum_);
    tilingData_.set_prePostRowOfFormerBlock(rowOfFormerBlock_);
    tilingData_.set_prePostRowOfTailBlock(rowOfTailBlock_);
    tilingData_.set_prePostRowLoopOfFormerBlock(prePostRowLoopOfFormerBlock);
    tilingData_.set_prePostRowLoopOfTailBlock(prePostRowLoopOfTailBlock);
    tilingData_.set_prePostRowFactor(prePostRowFactor);
    tilingData_.set_prePostTailRowFactorOfFormerBlock(prePostTailRowFactorOfFormerBlock);
    tilingData_.set_prePostTailRowFactorOfTailBlock(prePostTailRowFactorOfTailBlock);
    tilingData_.set_needGrad(needGrad_ ? 1 : 0);
    // 可选输入 premix: [bs, hcMult] fp32，premix 存在时 hin 加权源由内部 hPre 切换为 premix
    tilingData_.set_hasPremix(context_->GetOptionalInputShape(4) != nullptr ? 1 : 0);
    if (useOriginal) {
        return ge::GRAPH_SUCCESS;
    }
    return CalcSinkhornTiling();
}

ge::graphStatus MhcPreSinkhornPremixTiling::CalcSinkhornTiling()
{
    sinkhornRowOfFormerBlock_ = CeilDiv(bs_, sinkhornCoreNum_);
    sinkhornUsedCoreNum_ = std::min(CeilDiv(bs_, sinkhornRowOfFormerBlock_), sinkhornCoreNum_);
    sinkhornRowOfTailBlock_ = bs_ - (sinkhornUsedCoreNum_ - 1) * sinkhornRowOfFormerBlock_;

    int64_t kBlockNum = tilingData_.get_cubeBlockDimK();
    auto calcTotalSize = [&](int64_t rowFactor) {
        int64_t rowFactorAlign = RoundUp(rowFactor, BLOCK_SIZE / sizeof(float));
        int64_t matrixSize = hcMult_ * hcMult_;
        int64_t mix2OriginSize = kBlockNum * rowFactor * matrixSize * sizeof(float) * DOUBLE_BUFFER;
        int64_t combFragSize = rowFactor * matrixSize * sizeof(float) * DOUBLE_BUFFER;
        int64_t squareSumSize =
            kBlockNum * RoundUp(rowFactor * SQUARE_SUM_SIZE, SQUARE_SUM_SIZE) * sizeof(float) * DOUBLE_BUFFER;
        int64_t rsqrtSize = RoundUp(rowFactor, BLOCK_SIZE / sizeof(float)) * sizeof(float);
        int64_t squareReduceSize = rowFactor * SQUARE_SUM_SIZE * sizeof(float);
        int64_t mixes02ReduceSize = matrixSize * rowFactorAlign * sizeof(float);
        int64_t reduceBufSize = hcMult_ * rowFactorAlign * sizeof(float);
        int64_t rowBrcb0Size = RoundUp(rowFactor, BLOCK_SIZE / sizeof(float)) * BLOCK_SIZE;
        int64_t hcBrcb1Size = matrixSize * BLOCK_SIZE;
        int64_t maskPatternSize = (rowFactor + MASK_PATTERN_DIM_SIZE - 1) * BLOCK_SIZE;
        return mix2OriginSize + combFragSize + squareSumSize + rsqrtSize + squareReduceSize + mixes02ReduceSize +
               reduceBufSize + rowBrcb0Size + hcBrcb1Size + maskPatternSize;
    };

    int64_t base2Size = hcMult_ * hcMult_ * sizeof(float);

    int64_t maxSinkhornRowFactor = std::min(sinkhornRowOfFormerBlock_, static_cast<int64_t>(64));
    // Keep double buffering for generalized batch sizes. The row-factor search
    // accounts for both queue slots so the allocation stays in UB.
    // 二分查找 [1, maxSinkhornRowFactor] 内使 sinkhorn 缓冲(含 base2)不超过 ubSize_ 的最大行因子
    sinkhornRowFactor_ = FindMaxFit(1, maxSinkhornRowFactor, static_cast<int64_t>(ubSize_) - base2Size, calcTotalSize);
    sinkhornRowFactor_ = std::min(sinkhornRowFactor_, maxSinkhornRowFactor);
    OP_CHECK_IF(sinkhornRowFactor_ < 1,
                OP_LOGE(context_->GetNodeName(), "MhcPreSinkhornPremix sinkhorn buffers exceed UB size %lu", ubSize_),
                return ge::GRAPH_FAILED);

    sinkhornRowLoopOfFormerBlock_ = CeilDiv(sinkhornRowOfFormerBlock_, sinkhornRowFactor_);
    sinkhornRowLoopOfTailBlock_ = CeilDiv(sinkhornRowOfTailBlock_, sinkhornRowFactor_);
    sinkhornTailRowFactorOfFormerBlock_ = sinkhornRowOfFormerBlock_ % sinkhornRowFactor_ == 0 ?
                                              sinkhornRowFactor_ :
                                              sinkhornRowOfFormerBlock_ % sinkhornRowFactor_;
    sinkhornTailRowFactorOfTailBlock_ = sinkhornRowOfTailBlock_ % sinkhornRowFactor_ == 0 ?
                                            sinkhornRowFactor_ :
                                            sinkhornRowOfTailBlock_ % sinkhornRowFactor_;

    tilingData_.set_sinkhornUsedCoreNum(sinkhornUsedCoreNum_);
    tilingData_.set_sinkhornRowOfFormerBlock(sinkhornRowOfFormerBlock_);
    tilingData_.set_sinkhornRowOfTailBlock(sinkhornRowOfTailBlock_);
    tilingData_.set_sinkhornRowLoopOfFormerBlock(sinkhornRowLoopOfFormerBlock_);
    tilingData_.set_sinkhornRowLoopOfTailBlock(sinkhornRowLoopOfTailBlock_);
    tilingData_.set_sinkhornRowFactor(sinkhornRowFactor_);
    tilingData_.set_sinkhornTailRowFactorOfFormerBlock(sinkhornTailRowFactorOfFormerBlock_);
    tilingData_.set_sinkhornTailRowFactorOfTailBlock(sinkhornTailRowFactorOfTailBlock_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornPremixTiling::CalcOpTiling()
{
    uint64_t kSize = hcMult_ * d_;
    tilingData_.set_k(kSize);
    // 计算bs_轴切核
    uint64_t mDimNum = std::min(aicCoreNum_, static_cast<uint64_t>(CeilDiv(bs_, M_L1_MAX_SIZE)));
    uint64_t kDimNum = aicCoreNum_ / mDimNum;
    if (kDimNum == 1) {
        // K cannot use the remaining AICs, so distribute M across all cores.
        mDimNum = aicCoreNum_;
    } else if (aicCoreNum_ % mDimNum != 0) {
        // Align mDimNum up to the nearest factor of aicCoreNum_ so that
        // mDimNum * kDimNum == aicCoreNum_ holds exactly. Without this the
        // integer division truncates kDimNum and the trailing AIC cores idle
        // (e.g. bs=2024: mDimNum=8, kDimNum=2, only 16 of 20 AICs run).
        for (uint64_t factor = mDimNum; factor <= aicCoreNum_; ++factor) {
            if (aicCoreNum_ % factor == 0) {
                mDimNum = factor;
                break;
            }
        }
    }
    uint64_t singleCoreM = RoundUp(CeilDiv(bs_, static_cast<int64_t>(mDimNum)), AscendC::BLOCK_CUBE);
    kDimNum = aicCoreNum_ / mDimNum;
    uint64_t splitKSize = RoundUp(CeilDiv(kSize, kDimNum), K_MULIT_CORE_SPLIT_BASE_SIZE);

    tilingData_.set_cubeBlockDimM(mDimNum);
    tilingData_.set_cubeBlockDimK(CeilDiv(kSize, splitKSize));
    tilingData_.set_multCoreSplitMSize(singleCoreM);
    tilingData_.set_mL1Size(std::min(M_L1_MAX_SIZE, singleCoreM));
    tilingData_.set_multCoreSplitKSize(splitKSize);
    tilingData_.set_kL1Size(std::min(A_L1_SIZE / tilingData_.get_mL1Size(), static_cast<uint64_t>(K_L1_MAX_SIZE)) /
                            K_L1_ALIGN_SIZE * K_L1_ALIGN_SIZE);

    tilingData_.set_cvLoopKSize(CV_LOOP_K_SIZE);

    // vector stage1 tiling
    tilingData_.set_cubeCoreNum(static_cast<int64_t>(aicCoreNum_));
    // x type bfloat16, y type float32 and double
    // exit node 1 b16 input Queue and 1 b32 output Queue
    int64_t lineByteSize = (sizeof(int16_t) + sizeof(int32_t)) * DOUBLE_BUFFER * tilingData_.get_cvLoopKSize();
    int64_t stage1MFactorValue = ubSize_ / lineByteSize;
    tilingData_.set_stage1MFactor(stage1MFactorValue);
    return CalcMKSplitCoreMembasePart2Tiling();
}

ge::graphStatus MhcPreSinkhornPremixTiling::DoOpTiling()
{
    if (GetPlatformInfo() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (GetShapeAttrsInfoInner() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    // 预留参数计算（切K）与stage2计算
    if (CalcOpTiling() == ge::GRAPH_FAILED) {
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

static void PrintTilingData(const gert::TilingContext* context, MhcPreSinkhornPremixTilingData& tiling)
{
    OP_LOGD(context->GetNodeName(), ">>>>>>>>>> Start to print MhcPreSinkhornPremix tiling data <<<<<<<<<<");
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: bs is %ld.", tiling.get_bs());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: hcMix is %ld.", tiling.get_hcMix());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: hcMult is %ld.", tiling.get_hcMult());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: d is %ld.", tiling.get_d());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: hcMultAlign is %ld.", tiling.get_hcMultAlign());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: rowOfFormerBlock is %ld.",
            tiling.get_rowOfFormerBlock());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: rowOfTailBlock is %ld.", tiling.get_rowOfTailBlock());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: rowLoopOfFormerBlock is %ld.",
            tiling.get_rowLoopOfFormerBlock());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: rowLoopOfTailBlock is %ld.",
            tiling.get_rowLoopOfTailBlock());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: rowFactor is %ld.", tiling.get_rowFactor());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: tailRowFactorOfFormerBlock is %ld.",
            tiling.get_tailRowFactorOfFormerBlock());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: tailRowFactorOfTailBlock is %ld.",
            tiling.get_tailRowFactorOfTailBlock());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: dLoop is %ld.", tiling.get_dLoop());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: dFactor is %ld.", tiling.get_dFactor());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: tailDFactor is %ld.", tiling.get_tailDFactor());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: iterTimes is %ld.", tiling.get_iterTimes());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: hcEps is %f.", tiling.get_hcEps());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: normEps is %f.", tiling.get_normEps());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: kBlockFactor is %ld.", tiling.get_kBlockFactor());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: kFactor is %ld.", tiling.get_kFactor());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: tailKFactor is %ld.", tiling.get_tailKFactor());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: kLoop is %ld.", tiling.get_kLoop());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: stage2RowFactor is %ld.",
            tiling.get_stage2RowFactor());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: k is %ld.", tiling.get_k());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: kLoopOfFormerBlock is %ld.",
            tiling.get_kLoopOfFormerBlock());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: kLoopOfTailBlock is %ld.",
            tiling.get_kLoopOfTailBlock());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: stage1KFactor is %ld.", tiling.get_stage1KFactor());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: kFactorOfFormerBlock is %ld.",
            tiling.get_kFactorOfFormerBlock());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: kL1Size is %ld.", tiling.get_kL1Size());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: mL1Size is %ld.", tiling.get_mL1Size());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: cubeBlockDimM is %ld.", tiling.get_cubeBlockDimM());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: cubeBlockDimK is %ld.", tiling.get_cubeBlockDimK());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: kUbSize is %ld.", tiling.get_kUbSize());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: cvLoopKSize is %ld.", tiling.get_cvLoopKSize());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: multCoreSplitKSize is %ld.",
            tiling.get_multCoreSplitKSize());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: multCoreSplitMSize is %ld.",
            tiling.get_multCoreSplitMSize());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: tailKSizeOfFormerBlock is %ld.",
            tiling.get_tailKSizeOfFormerBlock());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: tailKSizeOfTailBlock is %ld.",
            tiling.get_tailKSizeOfTailBlock());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: mLoopOfFormerBlock is %ld.",
            tiling.get_mLoopOfFormerBlock());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: mLoopOfTailBlock is %ld.",
            tiling.get_mLoopOfTailBlock());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: formerMSize is %ld.", tiling.get_formerMSize());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: tailMSizeOfFormerBlock is %ld.",
            tiling.get_tailMSizeOfFormerBlock());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: tailMSizeOfTailBlock is %ld.",
            tiling.get_tailMSizeOfTailBlock());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: firstUsedCoreNum is %ld.",
            tiling.get_firstUsedCoreNum());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: secondUsedCoreNum is %ld.",
            tiling.get_secondUsedCoreNum());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: rowInnerFactor is %ld.", tiling.get_rowInnerFactor());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: cubeCoreNum is %ld.", tiling.get_cubeCoreNum());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: stage1MFactor is %ld.", tiling.get_stage1MFactor());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: bufferPool0Size is %ld.",
            tiling.get_bufferPool0Size());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: bufferPool1Size is %ld.",
            tiling.get_bufferPool1Size());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: mUbSize is %ld.", tiling.get_mUbSize());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: needGrad is %ld.", tiling.get_needGrad());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: bsLoop is %ld.", tiling.get_bsLoop());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: tailBs is %ld.", tiling.get_tailBs());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: bsSplitThreshold is %ld.",
            tiling.get_bsSplitThreshold());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: curBsSplit is %ld.", tiling.get_curBsSplit());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: tailBsRowOfFormerBlock is %ld.",
            tiling.get_tailBsRowOfFormerBlock());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: tailBsRowOfTailBlock is %ld.",
            tiling.get_tailBsRowOfTailBlock());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: tailBsRowLoopOfFormerBlock is %ld.",
            tiling.get_tailBsRowLoopOfFormerBlock());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: tailBsRowLoopOfTailBlock is %ld.",
            tiling.get_tailBsRowLoopOfTailBlock());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: tailBsUsedCoreNum is %ld.",
            tiling.get_tailBsUsedCoreNum());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: tailBsRowFactor is %ld.",
            tiling.get_tailBsRowFactor());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: tailBsTailRowFactorOfFormerBlock is %ld.",
            tiling.get_tailBsTailRowFactorOfFormerBlock());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: tailBsTailRowFactorOfTailBlock is %ld.",
            tiling.get_tailBsTailRowFactorOfTailBlock());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: tailBsML1Size is %ld.", tiling.get_tailBsML1Size());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: tailBsKL1Size is %ld.", tiling.get_tailBsKL1Size());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: tailBsMultCoreSplitMSize is %ld.",
            tiling.get_tailBsMultCoreSplitMSize());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: tailBsCubeBlockDimM is %ld.",
            tiling.get_tailBsCubeBlockDimM());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: stage1VecCoreNum is %ld.",
            tiling.get_stage1VecCoreNum());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: stage1CubeCoreNum is %ld.",
            tiling.get_stage1CubeCoreNum());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: stage1BsPerVecCore is %ld.",
            tiling.get_stage1BsPerVecCore());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: stage1TailBsPerVecCore is %ld.",
            tiling.get_stage1TailBsPerVecCore());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: stage1BsLoop is %ld.", tiling.get_stage1BsLoop());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: stage1BsFactor is %ld.", tiling.get_stage1BsFactor());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: stage1NcLoop is %ld.", tiling.get_stage1NcLoop());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: stage1NcFactor is %ld.", tiling.get_stage1NcFactor());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: stage1TailNcFactor is %ld.",
            tiling.get_stage1TailNcFactor());
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremix_tiling: stage1XCastWsSize is %ld.",
            tiling.get_stage1XCastWsSize());
}

ge::graphStatus MhcPreSinkhornPremixTiling::CalcStage1Tiling()
{
    // 1. 计算核心数量
    int64_t vecCoreNum = static_cast<int64_t>(aivCoreNum_);
    int64_t cubeCoreNum = static_cast<int64_t>(aicCoreNum_);

    tilingData_.set_stage1VecCoreNum(vecCoreNum);
    tilingData_.set_stage1CubeCoreNum(cubeCoreNum);

    // 2. 计算 BS 分核
    int64_t bsPerVecCore = CeilDiv(bs_, vecCoreNum);
    int64_t tailBsPerVecCore = bs_ - (vecCoreNum - 1) * bsPerVecCore;
    if (tailBsPerVecCore <= 0) {
        tailBsPerVecCore = bsPerVecCore;
    }

    tilingData_.set_stage1BsPerVecCore(bsPerVecCore);
    tilingData_.set_stage1TailBsPerVecCore(tailBsPerVecCore);

    // 3. 计算 BS 循环（每轮最多 32 个 BS）
    int64_t bsLoop = CeilDiv(bsPerVecCore, MAX_BS_PER_LOOP);
    int64_t bsFactor = MAX_BS_PER_LOOP;

    tilingData_.set_stage1BsLoop(bsLoop);
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
    int64_t mm1N = hcMult_ * hcMult_ + 2 * hcMult_;
    int64_t mm1K = hcMult_ * d_;

    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE("TilingForMhcPreSinkhornPremix", "Tiling platformInfo is null"),
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
        int64_t hcMix = hcMult_ * hcMult_ + 2 * hcMult_;
        int64_t hcBeforeNormWsSize = bs_ * hcMix * sizeof(float);

        workspaceSize_ += invRmsWsSize + hcBeforeNormWsSize;
    }

    // cubeBlockDimK 以 CalcOpTiling 计算结果为准（K维切核），不再强制覆盖为1
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornPremixTiling::GetWorkspaceSize()
{
    // 对齐 hc_pre 的动态 workspace 公式
    int64_t xCastFp32BufSize = tilingData_.get_mL1Size() * RoundUp(tilingData_.get_cvLoopKSize(), K_L1_ALIGN_SIZE);
    int64_t workspaceSize1 = tilingData_.get_cubeCoreNum() * DOUBLE_BUFFER * xCastFp32BufSize * sizeof(float);

    int64_t mmLastAxisSize = RoundUp(tilingData_.get_hcMix(), 128);
    int64_t workspaceSize2 =
        RoundUp(tilingData_.get_cubeBlockDimK() * tilingData_.get_bs() * mmLastAxisSize * sizeof(float), 512);

    int64_t squareSumSize = RoundUp(tilingData_.get_cubeBlockDimK() * RoundUp(tilingData_.get_bs(), SQUARE_SUM_SIZE) *
                                        SQUARE_SUM_SIZE * sizeof(float),
                                    512);

    int64_t requiredSize = workspaceSize1 + workspaceSize2 + squareSumSize + 16 * 1024 * 1024; // 16MB 预留缓冲
    int64_t defaultSize = 16 * 1024 * 1024 + 192 * 1024 * 1024;                                // 208MB
    // 累加而非覆盖，保留 CalcStage1Tiling 已追加的 stage1 workspace
    workspaceSize_ += requiredSize > defaultSize ? requiredSize : defaultSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornPremixTiling::PostTiling()
{
    context_->SetTilingKey(tilingKey_);
    context_->SetBlockDim(aicCoreNum_);
    if (tilingKey_ == MPMD_TILING_KEY) {
        auto ret = context_->SetScheduleMode(1);
        OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_, "Failed to set schedule mode to 1 for MhcPreSinkhornPremix."),
                    return ge::GRAPH_FAILED);
    }
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;

    PrintTilingData(context_, tilingData_);

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForMhcPreSinkhornPremix(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForMhcPreSinkhornPremix(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("MhcPreSinkhornPremix", "Tiling context is null"), return ge::GRAPH_FAILED);

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE("TilingForMhcPreSinkhornPremix", "Tiling platformInfo is null"),
                return ge::GRAPH_FAILED);
    if (Ops::Transformer::OpTiling::IsRegbaseSocVersion(context)) {
        OP_LOGD(context->GetNodeName(), "Using arch35 tiling for regbase soc");
        MhcPreSinkhornPremixTilingRegbase mhcPreSinkhornTiling(context);
        return mhcPreSinkhornTiling.DoOpTiling();
    }

    MhcPreSinkhornPremixTiling mhcPreSinkhornTiling(context);
    return mhcPreSinkhornTiling.DoOpTiling();
}

IMPL_OP_OPTILING(MhcPreSinkhornPremix)
    .Tiling(TilingForMhcPreSinkhornPremix)
    .TilingParse<MhcPreSinkhornPremixCompileInfo>(TilingPrepareForMhcPreSinkhornPremix);

} // namespace optiling
