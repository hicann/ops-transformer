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
 * \file compressor_tools.h
 * \brief Batch 元数据访问与 Vec 压缩窗切片迭代
 */

#ifndef COMPRESSOR_TOOLS_H
#define COMPRESSOR_TOOLS_H

#include "compressor_comm_arch92.h"

using namespace AscendC;

namespace Compressor {

struct ToolsParams {
    uint32_t seqSize = 0U;
    uint32_t cmpRatio = 0U;
};

template <typename COMP>
class CompressorTools {
public:
    __aicore__ inline CompressorTools() {}

    __aicore__ inline void Init(__gm__ uint8_t* startPos, __gm__ uint8_t* seqUsed, __gm__ uint8_t* cuSeqlens);

    __aicore__ inline uint32_t GetSeqUsed(uint32_t bIdx);
    __aicore__ inline uint32_t GetStartPos(uint32_t bIdx);
    __aicore__ inline uint32_t GetSeqLength(uint32_t bIdx);
    __aicore__ inline uint32_t GetTIdxByBatch(uint32_t bIdx);

public:
    ToolsParams toolParams_{};
    bool isExistSeqUsed_ = false;

private:
    bool isExistStartPos_ = false;
    GlobalTensor<int32_t> cuSeqlensGm_;
    GlobalTensor<int32_t> sequsedGm_;
    GlobalTensor<int32_t> startPosGm_;
};

template <typename COMP>
__aicore__ inline void CompressorTools<COMP>::Init(__gm__ uint8_t* startPos, __gm__ uint8_t* seqUsed,
                                                   __gm__ uint8_t* cuSeqlens)
{
    isExistStartPos_ = (startPos != nullptr);
    if (isExistStartPos_) {
        startPosGm_.SetGlobalBuffer((__gm__ int32_t*)startPos);
    }

    isExistSeqUsed_ = (seqUsed != nullptr);
    if (isExistSeqUsed_) {
        sequsedGm_.SetGlobalBuffer((__gm__ int32_t*)seqUsed);
    }

    if constexpr (COMP::xLayout == X_LAYOUT::TH) {
        cuSeqlensGm_.SetGlobalBuffer((__gm__ int32_t*)cuSeqlens);
    }
}

template <typename COMP>
__aicore__ inline uint32_t CompressorTools<COMP>::GetSeqUsed(uint32_t bIdx)
{
    if (isExistSeqUsed_) {
        return (uint32_t)sequsedGm_.GetValue(bIdx);
    }
    return GetSeqLength(bIdx);
}

template <typename COMP>
__aicore__ inline uint32_t CompressorTools<COMP>::GetStartPos(uint32_t bIdx)
{
    if (isExistStartPos_) {
        return (uint32_t)startPosGm_.GetValue(bIdx);
    } else {
        return 0;
    }
}

template <typename COMP>
__aicore__ inline uint32_t CompressorTools<COMP>::GetSeqLength(uint32_t bIdx)
{
    if constexpr (COMP::xLayout == X_LAYOUT::TH) {
        return cuSeqlensGm_.GetValue(bIdx + 1) - cuSeqlensGm_.GetValue(bIdx);
    } else {
        return toolParams_.seqSize;
    }
}

template <typename COMP>
__aicore__ inline uint32_t CompressorTools<COMP>::GetTIdxByBatch(uint32_t bIdx)
{
    if constexpr (COMP::xLayout == X_LAYOUT::TH) {
        return (uint32_t)(cuSeqlensGm_.GetValue(bIdx));
    } else {
        return (uint32_t)toolParams_.seqSize * bIdx;
    }
}

// iterator
struct Vec1SliceInfo {
    uint32_t bIdx = 0U;
    uint32_t sIdx = 0U;
    uint32_t bSeqUsed = 0U;
    uint64_t bStartPos = 0U;

    uint32_t headHolderSeqCnt = 0U;
    uint32_t validSeqCnt = 0U;
    uint32_t tailHolderSeqCnt = 0U;

    uint32_t dealTcSize = 0U;
    uint32_t compressTcSize = 0U;

    uint32_t dealedSeqCnt = 0U;
    uint32_t bSeqLength = 0U;
    uint32_t compressoredScCnt = 0U;
    bool isFirst = false;
    bool isLast = false;
};

struct StatisticInfo {
    uint32_t actualTcCnt = 0U;
    uint32_t dealSeqCnt = 0U;
    uint32_t compressorScCnt = 0U;
};

template <typename COMP>
class CompressorVec1SliceIterator {
public:
    // Tc 配额覆盖完整 seqLength（含 seqUsed 之后的空洞）；行游标始终对应 workspace 中的真实位置。
    // GetSlice 只生成有效行切片，IteratorSlice 同时消耗有效窗及后续空洞的配额。
    __aicore__ inline CompressorVec1SliceIterator(CompressorTools<COMP>& tools)
        : tools_(tools)
    {}

    __aicore__ inline void Reset(uint32_t bIdx, uint32_t sIdx, uint32_t dealedSeqCnt, uint32_t compressoredScCnt);
    __aicore__ inline void SetMaxBatchSize(uint32_t batchSize);
    __aicore__ inline void SetNeedDealTcSize(uint32_t needDealTcSize);
    __aicore__ inline uint32_t GetNeedDealTcSize();
    __aicore__ inline bool IsEnd();
    __aicore__ inline void IteratorSlice();
    __aicore__ inline Vec1SliceInfo& GetSlice();
    __aicore__ inline StatisticInfo& CollectSliceStats();

private:
    __aicore__ inline bool LoadBatch();
    __aicore__ inline void AdvanceRows(uint32_t rows);
    __aicore__ inline void AdvanceToNextUsedBatch();
    __aicore__ inline bool SkipBatchGap();
    CompressorTools<COMP>& tools_;

    bool isFirst_ = true;
    Vec1SliceInfo sliceInfo_{};
    StatisticInfo statisticInfo_{};
    uint32_t needDealTcSize_ = 0U;
    uint32_t batchSize_ = 0U;
};

template <typename COMP>
__aicore__ inline bool CompressorVec1SliceIterator<COMP>::LoadBatch()
{
    if (sliceInfo_.bIdx >= batchSize_) {
        sliceInfo_.sIdx = 0;
        sliceInfo_.bSeqUsed = 0;
        sliceInfo_.bSeqLength = 0;
        sliceInfo_.bStartPos = 0;
        needDealTcSize_ = 0;
        return false;
    }
    sliceInfo_.bSeqLength = tools_.GetSeqLength(sliceInfo_.bIdx);
    sliceInfo_.bSeqUsed = tools_.isExistSeqUsed_ ? tools_.GetSeqUsed(sliceInfo_.bIdx) : sliceInfo_.bSeqLength;
    sliceInfo_.bStartPos = tools_.GetStartPos(sliceInfo_.bIdx);
    return true;
}

template <typename COMP>
__aicore__ inline void CompressorVec1SliceIterator<COMP>::Reset(uint32_t bIdx, uint32_t sIdx, uint32_t dealedSeqCnt,
                                                                uint32_t compressoredScCnt)
{
    sliceInfo_ = {};
    sliceInfo_.bIdx = bIdx;
    sliceInfo_.sIdx = sIdx;
    sliceInfo_.dealedSeqCnt = dealedSeqCnt;
    sliceInfo_.compressoredScCnt = compressoredScCnt;
    needDealTcSize_ = 0;
    isFirst_ = true;
    // TH 可以有空 batch；只向前查找，到末尾终止，不回绕。
    while (LoadBatch() && sliceInfo_.bSeqLength == 0) {
        ++sliceInfo_.bIdx;
        sliceInfo_.sIdx = 0;
    }
}

template <typename COMP>
__aicore__ inline void CompressorVec1SliceIterator<COMP>::SetMaxBatchSize(uint32_t batchSize)
{
    batchSize_ = batchSize;
}

template <typename COMP>
__aicore__ inline void CompressorVec1SliceIterator<COMP>::SetNeedDealTcSize(uint32_t needDealTcSize)
{
    needDealTcSize_ = sliceInfo_.bIdx < batchSize_ ? needDealTcSize : 0U;
}

template <typename COMP>
__aicore__ inline void CompressorVec1SliceIterator<COMP>::AdvanceRows(uint32_t rows)
{
    sliceInfo_.sIdx += rows;
    sliceInfo_.dealedSeqCnt += rows;
}

template <typename COMP>
__aicore__ inline void CompressorVec1SliceIterator<COMP>::IteratorSlice()
{
    needDealTcSize_ -= sliceInfo_.dealTcSize;
    AdvanceRows(sliceInfo_.validSeqCnt);
    sliceInfo_.compressoredScCnt += sliceInfo_.compressTcSize;
    if (sliceInfo_.sIdx >= sliceInfo_.bSeqUsed) {
        AdvanceToNextUsedBatch();
    }
    isFirst_ = false;
}

template <typename COMP>
__aicore__ inline void CompressorVec1SliceIterator<COMP>::AdvanceToNextUsedBatch()
{
    do {
        if (SkipBatchGap()) {
            return;
        }
        ++sliceInfo_.bIdx;
        sliceInfo_.sIdx = 0;
        if (!LoadBatch()) {
            return;
        }
    } while (sliceInfo_.bSeqUsed == 0);
}

template <typename COMP>
__aicore__ inline bool CompressorVec1SliceIterator<COMP>::SkipBatchGap()
{
    if (sliceInfo_.bSeqUsed >= sliceInfo_.bSeqLength) {
        return false;
    }
    const uint64_t ratio = tools_.toolParams_.cmpRatio;
    const uint64_t startPos = sliceInfo_.bStartPos;
    // 未处理过有效行时，起点窗尚未计入；否则起点所在的窗已经由 slice 消耗。
    const uint64_t firstTc = sliceInfo_.sIdx == 0 ? startPos / ratio : CeilDivT(startPos + sliceInfo_.sIdx, ratio);
    const uint32_t gapTc = static_cast<uint32_t>(CeilDivT(startPos + sliceInfo_.bSeqLength, ratio) - firstTc);
    const uint32_t skipTc = min(needDealTcSize_, gapTc);
    // 由消费后的窗边界直接确定行终点，同时覆盖尾部 padding、空洞和部分跳过。
    // max 保证零配额不会回退；min 保证行终点不越过当前 batch。
    const uint64_t endPos =
        max(startPos + sliceInfo_.sIdx, min(startPos + sliceInfo_.bSeqLength, (firstTc + skipTc) * ratio));
    AdvanceRows(static_cast<uint32_t>(endPos - startPos - sliceInfo_.sIdx));
    needDealTcSize_ -= skipTc;
    return skipTc < gapTc;
}

template <typename COMP>
__aicore__ inline uint32_t CompressorVec1SliceIterator<COMP>::GetNeedDealTcSize()
{
    return needDealTcSize_;
}

template <typename COMP>
__aicore__ inline bool CompressorVec1SliceIterator<COMP>::IsEnd()
{
    return needDealTcSize_ == 0;
}

template <typename COMP>
__aicore__ inline Vec1SliceInfo& CompressorVec1SliceIterator<COMP>::GetSlice()
{
    sliceInfo_.isFirst = isFirst_;
    sliceInfo_.isLast = false;
    if (IsEnd() || sliceInfo_.sIdx >= sliceInfo_.bSeqUsed) {
        sliceInfo_.headHolderSeqCnt = 0;
        sliceInfo_.validSeqCnt = 0;
        sliceInfo_.tailHolderSeqCnt = 0;
        sliceInfo_.dealTcSize = 0;
        sliceInfo_.compressTcSize = 0;
        return sliceInfo_;
    }
    const uint64_t ratio = tools_.toolParams_.cmpRatio;
    const uint32_t remainingRows = sliceInfo_.bSeqUsed - sliceInfo_.sIdx;
    sliceInfo_.headHolderSeqCnt = (sliceInfo_.bStartPos + sliceInfo_.sIdx) % ratio;
    sliceInfo_.validSeqCnt =
        min(static_cast<uint64_t>(remainingRows), needDealTcSize_ * ratio - sliceInfo_.headHolderSeqCnt);
    const uint64_t windowRows = sliceInfo_.headHolderSeqCnt + static_cast<uint64_t>(sliceInfo_.validSeqCnt);
    sliceInfo_.dealTcSize = CeilDivT(windowRows, ratio);
    sliceInfo_.tailHolderSeqCnt = sliceInfo_.dealTcSize * ratio - windowRows;
    sliceInfo_.compressTcSize = windowRows / ratio;
    sliceInfo_.isLast =
        CeilDivT(sliceInfo_.headHolderSeqCnt + static_cast<uint64_t>(remainingRows), ratio) >= needDealTcSize_;
    return sliceInfo_;
}

template <typename COMP>
__aicore__ inline StatisticInfo& CompressorVec1SliceIterator<COMP>::CollectSliceStats()
{
    statisticInfo_ = {};
    const uint32_t seqStart = sliceInfo_.dealedSeqCnt;
    while (!IsEnd()) {
        const Vec1SliceInfo& slice = GetSlice();
        statisticInfo_.actualTcCnt += slice.dealTcSize;
        statisticInfo_.compressorScCnt += slice.compressTcSize;
        IteratorSlice();
    }
    statisticInfo_.dealSeqCnt = sliceInfo_.dealedSeqCnt - seqStart;
    return statisticInfo_;
}

} // namespace Compressor

#endif
