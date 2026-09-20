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
 * \file lightning_indexer_v2_kernel_base_arch35.h
 * \brief Common kernel helpers shared by LightningIndexer variants.
 */

#ifndef LIGHTNING_INDEXER_V2_KERNEL_BASE_ARCH35_H
#define LIGHTNING_INDEXER_V2_KERNEL_BASE_ARCH35_H

#include "lightning_indexer_v2_base_arch35.h"

namespace LIV2Common {

__aicore__ inline uint32_t GetActualSeqLen(uint32_t batchIdx, bool hasCuSeqlens, bool hasSeqused,
                                           AscendC::GlobalTensor<uint32_t> &cuSeqlensGm,
                                           AscendC::GlobalTensor<uint32_t> &sequsedGm, uint32_t defaultSeqLen)
{
    if (hasSeqused) {
        return sequsedGm.GetValue(batchIdx);
    } else if (hasCuSeqlens) {
        return cuSeqlensGm.GetValue(batchIdx + 1) - cuSeqlensGm.GetValue(batchIdx);
    } else {
        return defaultSeqLen;
    }
}

template <typename RatioT>
__aicore__ inline uint32_t GetMaskedS2BaseBlockNum(uint32_t s1gIdx, uint32_t actS1Size, uint32_t actS2SizeOrig,
                                                   uint32_t s1BaseSize, RatioT cmpRatio, uint32_t s2BaseSize,
                                                   uint32_t *validLength = nullptr)
{
    if (actS2SizeOrig / cmpRatio == 0) {
        return 0;
    }
    uint32_t s1Offset = s1BaseSize * s1gIdx;
    // 压缩前的validS2LenBase
    int32_t validS2LenBase = static_cast<int32_t>(actS2SizeOrig) - static_cast<int32_t>(actS1Size);
    int32_t validS2Len = (static_cast<int32_t>(s1Offset) + validS2LenBase + static_cast<int32_t>(s1BaseSize)) /
                         static_cast<int32_t>(cmpRatio);
    validS2Len = Min(validS2Len, static_cast<int32_t>(actS2SizeOrig) / cmpRatio);
    validS2Len = Max(validS2Len, 1);
    if (validLength != nullptr) {
        *validLength = validS2Len;
    }
    return (validS2Len + s2BaseSize - 1) / s2BaseSize;
}

template <typename TempLoopInfoT, typename ConstInfoT>
__aicore__ inline void GetBN2Idx(TempLoopInfoT &tempLoopInfo, const ConstInfoT &constInfo, uint32_t bN2Idx)
{
    tempLoopInfo.bN2Idx = bN2Idx;
    tempLoopInfo.bIdx = bN2Idx / constInfo.kHeadNum;
    tempLoopInfo.n2Idx = bN2Idx % constInfo.kHeadNum;
}

template <bool IS_BSND, typename TempLoopInfoT, typename ConstInfoT, typename SplitCoreInfoT>
__aicore__ inline void CalcGS1LoopParams(TempLoopInfoT &tempLoopInfo, const ConstInfoT &constInfo,
                                         const SplitCoreInfoT &splitCoreInfo, uint32_t bN2LoopIdx)
{
    if ((tempLoopInfo.actS2Size == 0) || (tempLoopInfo.actS1Size == 0)) {
        tempLoopInfo.curActSeqLenIsZero = true;
        return;
    }
    tempLoopInfo.curActSeqLenIsZero = false;
    tempLoopInfo.s2BasicSizeTail = tempLoopInfo.actS2Size % constInfo.s2BaseSize;
    tempLoopInfo.s2BasicSizeTail =
        (tempLoopInfo.s2BasicSizeTail == 0) ? constInfo.s2BaseSize : tempLoopInfo.s2BasicSizeTail;
    tempLoopInfo.mBasicSizeTail = (tempLoopInfo.actS1Size * constInfo.gSize) % constInfo.mBaseSize;
    tempLoopInfo.mBasicSizeTail =
        (tempLoopInfo.mBasicSizeTail == 0) ? constInfo.mBaseSize : tempLoopInfo.mBasicSizeTail;

    uint32_t gS1SplitNum = (tempLoopInfo.actS1Size * constInfo.gSize + constInfo.mBaseSize - 1) / constInfo.mBaseSize;
    tempLoopInfo.gS1LoopEnd = (bN2LoopIdx == splitCoreInfo.bN2End) ? splitCoreInfo.gS1End : gS1SplitNum - 1;
    if constexpr (IS_BSND) {
        if (tempLoopInfo.gS1LoopEnd == gS1SplitNum - 1 && constInfo.qSeqSize > tempLoopInfo.actS1Size) {
            tempLoopInfo.needDealActS1LessThanS1 = true;
        }
    }
}

template <bool UPDATE_LD, typename TempLoopInfoT, typename ConstInfoT, typename SplitCoreInfoT, typename RatioT>
__aicore__ inline void CalcS2LoopParams(TempLoopInfoT &tempLoopInfo, const ConstInfoT &constInfo,
                                        const SplitCoreInfoT &splitCoreInfo, uint32_t bN2LoopIdx, uint32_t gS1LoopIdx,
                                        RatioT cmpRatio, uint32_t *validS2Len = nullptr)
{
    tempLoopInfo.gS1Idx = gS1LoopIdx;
    tempLoopInfo.actMBaseSize = constInfo.mBaseSize;
    uint32_t remainedGS1Size = tempLoopInfo.actS1Size * constInfo.gSize - tempLoopInfo.gS1Idx * constInfo.mBaseSize;
    if (remainedGS1Size <= constInfo.mBaseSize && remainedGS1Size > 0) {
        tempLoopInfo.actMBaseSize = tempLoopInfo.mBasicSizeTail;
    }

    bool isEnd = (bN2LoopIdx == splitCoreInfo.bN2End) && (gS1LoopIdx == splitCoreInfo.gS1End);
    uint32_t s2BlockNum;
    if (constInfo.attenMaskFlag) {
        s2BlockNum = GetMaskedS2BaseBlockNum(gS1LoopIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2SizeOrig,
                                             constInfo.s1BaseSize, cmpRatio, constInfo.s2BaseSize, validS2Len);
    } else {
        s2BlockNum = (tempLoopInfo.actS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    }
    tempLoopInfo.s2LoopEnd = isEnd ? splitCoreInfo.s2End : s2BlockNum - 1;
    if constexpr (UPDATE_LD) {
        if (splitCoreInfo.s2Start > 0 || tempLoopInfo.s2LoopEnd < s2BlockNum - 1) {
            tempLoopInfo.isNeedLD = true;
        } else {
            tempLoopInfo.isNeedLD = false;
        }
    }
}

template <typename RunInfoT, typename TempLoopInfoT, typename ConstInfoT, typename SplitCoreInfoT>
__aicore__ inline bool InitRunInfoBase(uint32_t loop, uint32_t s2LoopIdx, RunInfoT &runInfo,
                                       const TempLoopInfoT &tempLoopInfo, const ConstInfoT &constInfo,
                                       const SplitCoreInfoT &splitCoreInfo)
{
    runInfo.loop = loop;
    runInfo.bIdx = tempLoopInfo.bIdx;
    runInfo.gS1Idx = tempLoopInfo.gS1Idx;
    runInfo.s2Idx = s2LoopIdx;
    runInfo.bN2Idx = tempLoopInfo.bN2Idx;
    runInfo.isValid = s2LoopIdx <= tempLoopInfo.s2LoopEnd;
    if (!runInfo.isValid) {
        return false;
    }

    runInfo.actS1Size = tempLoopInfo.actS1Size;
    runInfo.actS2Size = tempLoopInfo.actS2Size;
    runInfo.actS2SizeOrig = tempLoopInfo.actS2SizeOrig;
    // 计算实际基本块size
    runInfo.actMBaseSize = tempLoopInfo.actMBaseSize;
    runInfo.actualSingleProcessSInnerSize = constInfo.s2BaseSize;
    uint32_t s2SplitNum = (tempLoopInfo.actS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    if (runInfo.s2Idx == s2SplitNum - 1) {
        runInfo.actualSingleProcessSInnerSize = tempLoopInfo.s2BasicSizeTail;
    }
    runInfo.actualSingleProcessSInnerSizeAlign =
        Align((uint32_t)runInfo.actualSingleProcessSInnerSize, ConstInfoT::BUFFER_SIZE_BYTE_32B);

    runInfo.isFirstS2InnerLoop = s2LoopIdx == splitCoreInfo.s2Start;
    runInfo.isLastS2InnerLoop = s2LoopIdx == tempLoopInfo.s2LoopEnd;
    runInfo.isAllLoopEnd = (runInfo.bN2Idx == splitCoreInfo.bN2End) && (runInfo.gS1Idx == splitCoreInfo.gS1End) &&
                           (runInfo.s2Idx == splitCoreInfo.s2End);
    return true;
}

template <typename RunInfoT, typename TempLoopInfoT, typename ConstInfoT, typename SplitCoreInfoT,
          typename LdSplitCoreInfoT>
__aicore__ inline bool InitRunInfo(uint32_t loop, uint32_t s2LoopIdx, RunInfoT &runInfo,
                                   const TempLoopInfoT &tempLoopInfo, const ConstInfoT &constInfo,
                                   const SplitCoreInfoT &splitCoreInfo, LdSplitCoreInfoT &ldInfo,
                                   bool isOutputIdxOffsetValid)
{
    runInfo.isNeedLD = tempLoopInfo.isNeedLD;
    if (runInfo.isNeedLD && s2LoopIdx == tempLoopInfo.s2LoopEnd) {
        runInfo.saveWorkSpaceIdx = ldInfo.saveWorkSpaceIdx;
        ldInfo.saveWorkSpaceIdx++;
    }
    if (!InitRunInfoBase(loop, s2LoopIdx, runInfo, tempLoopInfo, constInfo, splitCoreInfo)) {
        return false;
    }
    runInfo.isOutputIdxOffsetValid = isOutputIdxOffsetValid;
    return true;
}

template <auto TND_LAYOUT, auto BSND_LAYOUT, typename ConstInfoT, typename VectorServiceT>
__aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, uint32_t s1Start, uint32_t tBase,
                                           uint32_t s1Count, uint32_t topk, const ConstInfoT &constInfo,
                                           VectorServiceT &vectorService)
{
    if ASCEND_IS_AIV {
        if (constInfo.outputLayout == TND_LAYOUT) {
            for (uint32_t s1Idx = s1Start; s1Idx < s1Count; s1Idx++) {
                uint64_t indiceOutOffset =
                    (static_cast<uint64_t>(tBase) + s1Idx) * constInfo.kHeadNum * topk + // T轴、s1轴偏移
                    static_cast<uint64_t>(n2Idx) * topk;                                 // N2轴偏移
                vectorService.CleanInvalidOutput(indiceOutOffset);
            }
        } else if (constInfo.outputLayout == BSND_LAYOUT) {
            for (uint32_t s1Idx = s1Start; s1Idx < constInfo.qSeqSize; s1Idx++) {
                // B,S1,N2,K
                uint64_t indiceOutOffset =
                    static_cast<uint64_t>(bIdx) * constInfo.qSeqSize * constInfo.kHeadNum * topk +
                    static_cast<uint64_t>(s1Idx) * constInfo.kHeadNum * topk + static_cast<uint64_t>(n2Idx) * topk;
                vectorService.CleanInvalidOutput(indiceOutOffset);
            }
        }
    }
}

template <typename VectorServiceT, typename MatmulServiceT, typename PipeT>
__aicore__ inline void InitBuffers(VectorServiceT &vectorService, MatmulServiceT &matmulService, PipeT *pipe)
{
    if ASCEND_IS_AIV {
        vectorService.InitBuffers(pipe);
    } else {
        matmulService.InitBuffers(pipe);
    }
}

} // namespace LIV2Common

#endif // LIGHTNING_INDEXER_V2_KERNEL_BASE_ARCH35_H
