/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lightning_indexer_v2_metadata_aicpu.h"
#include <algorithm>
#include <climits>

namespace aicpu {
using namespace optiling;
namespace {
struct LiV2FastSequenceWork {
    uint32_t actS1Size;
    uint32_t actS2Size;
    uint64_t actS2SizeOrig{0U}; // 原始域上下文长度(S2×r+residual)
};

struct CubeFillVectorSplitContext {
    uint32_t s1BaseSize{0U};
    uint32_t coreNum{0U};
    uint64_t totalS1TileNum{0U};
    uint64_t totalWorkBlockNum{0U};
    uint64_t usefulQueryKeyPairs{0U};
    uint32_t minS1Size{UINT32_MAX};
};

struct CubeFillVectorSplitPlan {
    uint32_t coreMetadata[AIC_CORE_MAX_NUM][LI_V2_FAST_CORE_INFO_SIZE]{};
    uint64_t maxCoreWorkBlockNum{0U};
};

bool CheckCubeFillVectorSequenceTensor(const Tensor* tensor, int64_t expectedSize, bool required = false)
{
    if ((tensor == nullptr) || (tensor->GetData() == nullptr)) {
        return !required;
    }
    const auto shape = tensor->GetTensorShape();
    return (shape != nullptr) && (shape->GetDims() == 1) && (tensor->GetDataType() == DT_INT32) &&
           (shape->GetDimSize(0) == expectedSize) &&
           (tensor->GetDataSize() >= static_cast<uint64_t>(expectedSize) * sizeof(int32_t));
}

constexpr uint32_t CUBE_FILL_VECTOR_S2_BASE_SIZE = 512U;
// Preserve signed vector index headroom for padded sorting chunks.
constexpr uint32_t CUBE_FILL_VECTOR_MAX_INDEX_LENGTH = INT32_MAX - 8192U;
// Retain the V7 empirical load bound while using the actual AIC count.
// Count all launched AICs in the average, including cores without a Q tile.
constexpr uint32_t CUBE_FILL_VECTOR_MIN_KEY_LENGTH = 2048U;
constexpr uint32_t CUBE_FILL_VECTOR_MAX_LOAD_PERCENT = 175U;
constexpr uint32_t CUBE_FILL_VECTOR_BALANCED_LOAD_PERCENT = 110U;
constexpr uint32_t CUBE_FILL_VECTOR_MIN_UTILIZATION_PERCENT = 55U;
constexpr uint32_t CUBE_FILL_VECTOR_BALANCED_UTILIZATION_PERCENT = 35U;
constexpr uint32_t CUBE_FILL_VECTOR_SMALL_QUERY_UTILIZATION_PERCENT = 45U;

// maskMode=0: tile 可见全部 S2; maskMode=3: 右下三角可见。
// cmpRatio>1: 因果可见长度在原始域计算后折算压缩域 /r(与 kernel_fast GetS2BaseBlockNumOnMask 一致)。
uint32_t GetS2BaseBlockNumOnMask(uint32_t gS1Idx, uint32_t actS1Size, uint32_t actS2Size, uint32_t s1BaseSize,
                                 bool noMask, uint32_t cmpRatio, uint64_t actS2SizeOrig)
{
    if (noMask) {
        return (actS2Size + CUBE_FILL_VECTOR_S2_BASE_SIZE - 1U) / CUBE_FILL_VECTOR_S2_BASE_SIZE;
    }
    if (cmpRatio == 1U) {
        const uint32_t s2End = std::min(actS2Size, actS2Size - actS1Size + (gS1Idx + 1U) * s1BaseSize);
        return (s2End + CUBE_FILL_VECTOR_S2_BASE_SIZE - 1U) / CUBE_FILL_VECTOR_S2_BASE_SIZE;
    }
    const uint64_t s1Offset = static_cast<uint64_t>(s1BaseSize) * gS1Idx;
    uint64_t validS2Len = (s1Offset + actS2SizeOrig - actS1Size + s1BaseSize) / cmpRatio;
    const uint64_t compressedS2 = actS2SizeOrig / cmpRatio;
    if (validS2Len > compressedS2) {
        validS2Len = compressedS2;
    }
    if (validS2Len < 1U) {
        validS2Len = 1U;
    }
    return static_cast<uint32_t>((validS2Len + CUBE_FILL_VECTOR_S2_BASE_SIZE - 1U) / CUBE_FILL_VECTOR_S2_BASE_SIZE);
}

bool CanCloseCubeFillVectorCore(uint32_t ownerCore, uint32_t coreNum, uint64_t ownerWork, uint64_t remainingWork,
                                uint64_t remainingTiles)
{
    const uint32_t remainingCores = coreNum - ownerCore;
    const uint64_t targetWork = (remainingWork + remainingCores - 1U) / remainingCores;
    const bool mustLeaveOneTilePerCore = remainingTiles == remainingCores - 1U;
    return ((ownerCore + 1U) == coreNum) ? (remainingTiles == 0) :
                                           (mustLeaveOneTilePerCore || (ownerWork >= targetWork));
}

bool PrepareCubeFillVectorSplitContext(const std::vector<LiV2FastSequenceWork>& sequences, uint32_t aicCoreNum,
                                       CubeFillVectorSplitContext& splitContext, bool noMask, uint32_t cmpRatio)
{
    for (uint32_t bN2Idx = 0; bN2Idx < sequences.size(); ++bN2Idx) {
        const uint32_t actS1Size = sequences[bN2Idx].actS1Size;
        const uint32_t actS2Size = sequences[bN2Idx].actS2Size;
        if (actS2Size < CUBE_FILL_VECTOR_MIN_KEY_LENGTH) {
            return false;
        }
        splitContext.minS1Size = std::min(splitContext.minS1Size, actS1Size);
        // maskMode=3: 右下三角有效对; maskMode=0: 全矩形; cmpRatio>1 按原始域逐行求和。
        if (noMask) {
            splitContext.usefulQueryKeyPairs += static_cast<uint64_t>(actS1Size) * actS2Size;
        } else if (cmpRatio == 1U) {
            splitContext.usefulQueryKeyPairs +=
                static_cast<uint64_t>(actS1Size) * (2ULL * actS2Size - actS1Size + 1U) / 2U;
        } else {
            const uint64_t actS2SizeOrig = sequences[bN2Idx].actS2SizeOrig;
            for (uint32_t i = 0; i < actS1Size; ++i) {
                splitContext.usefulQueryKeyPairs += (actS2SizeOrig - actS1Size + i + 1U) / cmpRatio;
            }
        }
        const uint32_t curS1TileNum = (actS1Size + splitContext.s1BaseSize - 1U) / splitContext.s1BaseSize;
        splitContext.totalS1TileNum += curS1TileNum;
        for (uint32_t gS1Idx = 0; gS1Idx < curS1TileNum; ++gS1Idx) {
            splitContext.totalWorkBlockNum +=
                GetS2BaseBlockNumOnMask(gS1Idx, actS1Size, actS2Size, splitContext.s1BaseSize, noMask, cmpRatio,
                                        sequences[bN2Idx].actS2SizeOrig);
        }
    }
    if (splitContext.totalS1TileNum == 0 || splitContext.totalWorkBlockNum == 0) {
        return false;
    }
    splitContext.coreNum = static_cast<uint32_t>(std::min<uint64_t>(aicCoreNum, splitContext.totalS1TileNum));
    return true;
}

// enhance 的加权整 S1 tile 分核：同一 tile 的全部 S2 block 由同一 AIC/AIV group 处理。
// Build the plan once for both load admission and native metadata emission.
void SplitCoreCubeFillVector(const std::vector<LiV2FastSequenceWork>& sequences,
                             const CubeFillVectorSplitContext& splitContext, CubeFillVectorSplitPlan& splitPlan,
                             bool noMask, uint32_t cmpRatio)
{
    uint32_t ownerCore = 0;
    uint64_t ownerWork = 0;
    uint64_t remainingWork = splitContext.totalWorkBlockNum;
    uint64_t remainingTiles = splitContext.totalS1TileNum;
    bool currentCoreStarted = false;
    for (uint32_t bN2Idx = 0; bN2Idx < sequences.size(); ++bN2Idx) {
        const uint32_t actS1Size = sequences[bN2Idx].actS1Size;
        const uint32_t actS2Size = sequences[bN2Idx].actS2Size;
        const uint32_t curS1TileNum = (actS1Size + splitContext.s1BaseSize - 1U) / splitContext.s1BaseSize;
        for (uint32_t gS1Idx = 0; gS1Idx < curS1TileNum; ++gS1Idx) {
            const uint32_t tileWork = GetS2BaseBlockNumOnMask(gS1Idx, actS1Size, actS2Size, splitContext.s1BaseSize,
                                                              noMask, cmpRatio, sequences[bN2Idx].actS2SizeOrig);
            auto& coreMetadata = splitPlan.coreMetadata[ownerCore];
            if (!currentCoreStarted) {
                coreMetadata[LI_V2_FAST_BN2_START_INDEX] = bN2Idx;
                coreMetadata[LI_V2_FAST_M_START_INDEX] = gS1Idx;
                currentCoreStarted = true;
            }
            ownerWork += tileWork;
            --remainingTiles;
            if (CanCloseCubeFillVectorCore(ownerCore, splitContext.coreNum, ownerWork, remainingWork, remainingTiles)) {
                coreMetadata[LI_V2_FAST_BN2_END_INDEX] = bN2Idx;
                coreMetadata[LI_V2_FAST_M_END_INDEX] = gS1Idx;
                splitPlan.maxCoreWorkBlockNum = std::max(splitPlan.maxCoreWorkBlockNum, ownerWork);
                remainingWork -= ownerWork;
                ownerWork = 0;
                ++ownerCore;
                currentCoreStarted = false;
            }
        }
    }
}

bool CheckCubeFillVectorPerformance(const CubeFillVectorSplitPlan& splitPlan,
                                    const CubeFillVectorSplitContext& splitContext, uint32_t aicCoreNum)
{
    constexpr uint64_t PERCENT_BASE = 100U;
    const uint64_t scheduledWork = splitPlan.maxCoreWorkBlockNum * aicCoreNum;
    // 最忙核的工作量超过平均1.75倍，跳过, M/(W/P) >= 1.75
    if (splitContext.totalWorkBlockNum == 0U ||
        scheduledWork * PERCENT_BASE > splitContext.totalWorkBlockNum * CUBE_FILL_VECTOR_MAX_LOAD_PERCENT) {
        return false;
    }
    // All shapes reuse the completed plan for utilization admission, including Q tail tiles.
    // A balanced plan tolerates less Q tile fill;
    // batches containing Q1/Q2 need a higher bound because the ordinary vector path is cheaper.
    uint64_t minUtilization = CUBE_FILL_VECTOR_MIN_UTILIZATION_PERCENT;
    // 最忙核工作量最多超平均工作量10%比较均衡 门槛放宽, M/(W/P) <= 1.1
    if (scheduledWork * PERCENT_BASE <= splitContext.totalWorkBlockNum * CUBE_FILL_VECTOR_BALANCED_LOAD_PERCENT) {
        minUtilization = splitContext.minS1Size >= 3U ? CUBE_FILL_VECTOR_BALANCED_UTILIZATION_PERCENT :
                                                        CUBE_FILL_VECTOR_SMALL_QUERY_UTILIZATION_PERCENT;
    }
    const uint64_t usefulWork = splitContext.usefulQueryKeyPairs /
                                (static_cast<uint64_t>(splitContext.s1BaseSize) * CUBE_FILL_VECTOR_S2_BASE_SIZE);

    const uint64_t requiredWork = scheduledWork / PERCENT_BASE * minUtilization +
                                  (scheduledWork % PERCENT_BASE * minUtilization + PERCENT_BASE - 1U) / PERCENT_BASE;
    return usefulWork >= requiredWork;
}

} // namespace

void LightningIndexerV2MetadataCpuKernel::GenFastMetadataArch22()
{
    // Compute 已清零 Fast 区，准入失败时保持 enable 为 0，由 Kernel 执行普通路径
    // Metadata 接口不提供 weight dtype、returnValue、outputIdxOffset、deterministic level
    // 和 candidate 属性；host 检查这些静态条件。AICPU 使用实际序列长度检查分核与性能准入。
    // TopK 泛化: ≤2048 任意档; >2048 需 1024 倍数且 ≤8192。
    // BSND 泛化: Q/K BSND 放开, 实际长度由 sequsedQ/K 提供。
    if (ProcessSocVersion() != ValidSocVersion::ASCEND910B || (layoutQ_ != "TND" && layoutQ_ != "BSND") ||
        (layoutK_ != "TND" && layoutK_ != "PA_BBND" && layoutK_ != "BSND") ||
        (maskMode_ != SPARSE_MODE_LOWER && maskMode_ != 0) || (numHeadsQ_ < 8 || numHeadsQ_ > 64) ||
        (numHeadsQ_ % 8U != 0U) || (numHeadsK_ != 1) || (headDim_ != HEAD_DIM_LIMIT) ||
        ((topk_ > TOPK_NUM_2048) && ((topk_ % 1024U != 0U) || (topk_ > 8192U))) || (aicCoreNum_ == 0) ||
        (aicCoreNum_ > AIC_CORE_MAX_NUM)) {
        return;
    }
    // cmpRatio 泛化: 放开 [2,128] 压缩域 + cmpResidualK
    // A supplied Q upper bound can rule out all-Q1/Q2 decode without scanning the batches.
    if (maxSeqlenQ_ > 0 && maxSeqlenQ_ <= 2) {
        return;
    }
    // Keep fast-specific storage checks local; do not change ordinary-route validation.
    // BSND: cuSeqlensQ 可选, 长度取 GetValue(bIdx+1); TND: 前缀和必传。
    if (batchSize_ <= 0 ||
        !CheckCubeFillVectorSequenceTensor(cuSeqlensQ_, static_cast<int64_t>(batchSize_) + 1, layoutQ_ != "BSND") ||
        (layoutK_ == "TND" &&
         !CheckCubeFillVectorSequenceTensor(cuSeqlensK_, static_cast<int64_t>(batchSize_) + 1, true)) ||
        !CheckCubeFillVectorSequenceTensor(sequsedQ_, batchSize_) ||
        !CheckCubeFillVectorSequenceTensor(sequsedK_, batchSize_)) {
        return;
    }
    std::vector<LiV2FastSequenceWork> sequences;
    sequences.reserve(static_cast<size_t>(batchSize_));
    uint32_t maxS1Size = 0;
    // BSND: cuSeqlensQ 为 [0, len...] 平铺格式(GetValue(bIdx+1) 即长度), fast 侧本地读取
    const bool isBsndQ = (layoutQ_ == "BSND");
    const int32_t* bsndCuSeqlensQ = nullptr;
    if (isBsndQ && cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
        bsndCuSeqlensQ = static_cast<const int32_t*>(cuSeqlensQ_->GetData());
    }
    for (uint32_t bIdx = 0; bIdx < static_cast<uint32_t>(batchSize_); ++bIdx) {
        const uint32_t actS1Size = isBsndQ ?
                                       (bsndCuSeqlensQ != nullptr ? static_cast<uint32_t>(bsndCuSeqlensQ[bIdx + 1]) :
                                                                    static_cast<uint32_t>(maxSeqlenQ_)) :
                                       GetS1SeqSize(bIdx);
        const uint32_t actS2Size = GetS2SeqSize(bIdx);
        // G1 的 actS2>=actS1 检查用原始域(压缩域 S2 可合法小于 S1)
        const uint64_t actS2SizeOrig = GetRevertS2Size(bIdx);
        if (actS1Size == 0 || actS2SizeOrig < static_cast<uint64_t>(actS1Size) ||
            actS2Size < static_cast<uint32_t>(topk_) || actS2Size > CUBE_FILL_VECTOR_MAX_INDEX_LENGTH) {
            return;
        }
        if (!isBsndQ) {
            const auto* cuSeqlensQ = static_cast<const int32_t*>(cuSeqlensQ_->GetData());
            if (actS1Size != static_cast<uint32_t>(cuSeqlensQ[bIdx + 1] - cuSeqlensQ[bIdx])) {
                return;
            }
        } else if (bsndCuSeqlensQ != nullptr &&
                   static_cast<uint32_t>(bsndCuSeqlensQ[bIdx + 1]) > static_cast<uint32_t>(maxSeqlenQ_)) {
            return; // BSND: 各样本长度不得超过 S1(shape)
        }
        if (layoutK_ == "TND") {
            const auto* cuSeqlensK = static_cast<const int32_t*>(cuSeqlensK_->GetData());
            if (actS2Size != static_cast<uint32_t>(cuSeqlensK[bIdx + 1] - cuSeqlensK[bIdx])) {
                return;
            }
        }
        sequences.push_back({actS1Size, actS2Size, actS2SizeOrig});
        maxS1Size = std::max(maxS1Size, actS1Size);
    }
    // All-Q1/Q2 batches cannot reach even the lowest utilization bound with Q tiles of 6/8.
    // Reject them before constructing a plan; mixed batches still use the full admission check.
    if (maxS1Size <= 2U) {
        return;
    }
    CubeFillVectorSplitContext splitContext;
    constexpr uint32_t FAST_M_TILE_ROWS = 192U;
    constexpr uint32_t FAST_S1_ROWS_FOR_32_HEADS = FAST_M_TILE_ROWS / QUERY_HEAD_32;
    constexpr uint32_t FAST_S1_ROWS_FOR_24_HEADS = FAST_M_TILE_ROWS / QUERY_HEAD_24;
    // N1 泛化: s1BaseSize 自适应(偶数, M = s1Base*N1 ≤ 192)
    auto CalcAdaptiveS1Base = [&FAST_M_TILE_ROWS](uint32_t heads) -> uint32_t {
        uint32_t raw = (FAST_M_TILE_ROWS + heads - 1U) / heads;
        uint32_t s1b = (raw % 2U != 0U) ? raw - 1U : raw;
        s1b = (s1b > 8U) ? 8U : s1b; // cap 8: Vector UB seq≤4 安全区
        while (s1b * heads > FAST_M_TILE_ROWS && s1b > 2U) {
            s1b -= 2U;
        }
        return s1b;
    };

    // 每个 tile 包含192个 token/head 行，32头处理6个token，24头处理8个token
    splitContext.s1BaseSize =
        (numHeadsQ_ == QUERY_HEAD_32) ?
            FAST_S1_ROWS_FOR_32_HEADS :
            ((numHeadsQ_ == QUERY_HEAD_24) ? FAST_S1_ROWS_FOR_24_HEADS : CalcAdaptiveS1Base(numHeadsQ_));
    // TopK 联合 cap: >2048 时缩小 s1Base, 与 host tiling 同步
    constexpr uint32_t FAST_TOPK_S1_CAP_BASE = 8192U;
    const uint32_t topkS1Cap = FAST_TOPK_S1_CAP_BASE / static_cast<uint32_t>(topk_) * 2U;
    if (splitContext.s1BaseSize > topkS1Cap) {
        splitContext.s1BaseSize = topkS1Cap;
    }
    const bool noMask = (maskMode_ == 0);
    if (!PrepareCubeFillVectorSplitContext(sequences, aicCoreNum_, splitContext, noMask,
                                           static_cast<uint32_t>(cmpRatio_))) {
        return;
    }
    CubeFillVectorSplitPlan splitPlan;
    SplitCoreCubeFillVector(sequences, splitContext, splitPlan, noMask, static_cast<uint32_t>(cmpRatio_));
    // The vector pipeline accumulates its consumed K512 block count in int32.
    if (splitPlan.maxCoreWorkBlockNum > static_cast<uint64_t>(INT32_MAX)) {
        return;
    }
    const bool cubeFillVector = CheckCubeFillVectorPerformance(splitPlan, splitContext, aicCoreNum_);
    if (!cubeFillVector) {
        return;
    }
    auto& fastMetadata = static_cast<detail::LiV2MetadataArch22*>(metadata_->GetData())->fastMetadata;
    fastMetadata.coreNum = splitContext.coreNum;
    for (uint32_t coreIdx = 0; coreIdx < splitContext.coreNum; ++coreIdx) {
        std::copy_n(splitPlan.coreMetadata[coreIdx], LI_V2_FAST_CORE_INFO_SIZE, fastMetadata.coreInfo[coreIdx]);
    }
    fastMetadata.enable = 1U;
}

} // namespace aicpu
