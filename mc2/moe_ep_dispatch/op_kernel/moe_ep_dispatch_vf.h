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
 * \file moe_ep_dispatch_vf.h
 * \brief
 */

#ifndef MOE_EP_DISPATCH_VF_H
#define MOE_EP_DISPATCH_VF_H

#include "moe_ep_dispatch_base.h"

namespace MoeEpDispatchVf {

#if defined(ENABLE_MOE_EP_KERNEL)

using namespace AscendC;

// A 256-byte vector register holds 128 B16 elements or 64 B32 elements.
constexpr uint32_t B16_ELEMENTS_PER_VECTOR = 128U;
constexpr uint32_t B32_ELEMENTS_PER_VECTOR = 64U;
// One B16 histogram register therefore holds 128 bins; a BIN0/BIN1 pair covers 256 bins per pass.
constexpr uint32_t HISTOGRAM_BINS_PER_REG = 128U;
constexpr uint32_t HISTOGRAM_BINS_PER_PASS = 2U * HISTOGRAM_BINS_PER_REG;
constexpr uint32_t HISTOGRAM_BINS_PER_BATCH = 2U * HISTOGRAM_BINS_PER_PASS;
constexpr static Reg::CastTrait HIST_U16_TO_U32_EVEN = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                        Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr static Reg::CastTrait HIST_U16_TO_U32_ODD = {Reg::RegLayout::ONE, Reg::SatMode::NO_SAT,
                                                       Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

/** Narrow expert ids to int16 and derive the raw destination rank. */
__simd_callee__ inline void PrepareExpertAndRank(__ubuf__ int32_t* topkIds, __ubuf__ int16_t* expertIds,
                                                 __ubuf__ int16_t* rawRank, uint32_t elementCount,
                                                 uint32_t expertsNumPerRank)
{
    Reg::RegTensor<int32_t> expertS32Reg0;
    Reg::RegTensor<int32_t> expertS32Reg1;
    Reg::RegTensor<int16_t> expertS16Reg;
    Reg::RegTensor<int16_t> tempReg;
    Reg::RegTensor<int16_t> rankReg;
    Reg::RegTensor<int16_t> divisorReg;
    Reg::RegTensor<int16_t> negOneReg;
    Reg::MaskReg validExpertMask;
    Reg::MaskReg allMask = Reg::CreateMask<int16_t, Reg::MaskPattern::ALL>();
    Reg::UnalignRegForLoad loadState;
    Reg::UnalignRegForStore expertStoreState;
    Reg::UnalignRegForStore rankStoreState;

    uint32_t remaining = elementCount;
    uint16_t loopCount = static_cast<uint16_t>(Ceil(elementCount, B16_ELEMENTS_PER_VECTOR));
    __ubuf__ int32_t* topkAddr = topkIds;
    __ubuf__ int16_t* expertAddr = expertIds;
    __ubuf__ int16_t* rankAddr = rawRank;
    Reg::Duplicate(divisorReg, static_cast<int16_t>(expertsNumPerRank), allMask);
    Reg::Duplicate(negOneReg, static_cast<int16_t>(-1), allMask);
    Reg::LoadUnAlignPre(loadState, topkAddr);

    for (uint16_t loop = 0; loop < loopCount; ++loop) {
        uint32_t validElements = remaining > B16_ELEMENTS_PER_VECTOR ? B16_ELEMENTS_PER_VECTOR : remaining;
        // Reg0 and Reg1 are consecutive groups of 64 S32 ids.
        Reg::LoadUnAlign(expertS32Reg0, loadState, topkAddr, B32_ELEMENTS_PER_VECTOR);
        Reg::LoadUnAlign(expertS32Reg1, loadState, topkAddr, B32_ELEMENTS_PER_VECTOR);
        // The input contract limits expert ids to the S16 range.
        Reg::DeInterleave(expertS16Reg, tempReg, reinterpret_cast<Reg::RegTensor<int16_t>&>(expertS32Reg0),
                          reinterpret_cast<Reg::RegTensor<int16_t>&>(expertS32Reg1));
        Reg::CompareScalar<int16_t, CMPMODE::GE>(validExpertMask, expertS16Reg, static_cast<int16_t>(0), allMask);
        Reg::Div(rankReg, expertS16Reg, divisorReg, allMask);
        Reg::Select(rankReg, rankReg, negOneReg, validExpertMask);
        Reg::StoreUnAlign(expertAddr, expertS16Reg, expertStoreState, validElements);
        Reg::StoreUnAlign(rankAddr, rankReg, rankStoreState, validElements);
        remaining -= validElements;
    }
    Reg::StoreUnAlignPost(expertAddr, expertStoreState, 0);
    Reg::StoreUnAlignPost(rankAddr, rankStoreState, 0);
}

/** Add at most 64 uint32 histogram bins to the persistent Direct counter. */
__simd_callee__ inline void StoreHistogramChunk(__ubuf__ uint32_t* dst, Reg::RegTensor<uint32_t>& localCount,
                                                uint32_t validBins)
{
    Reg::RegTensor<uint32_t> totalCount;
    Reg::UnalignRegForLoad loadState;
    Reg::UnalignRegForStore storeState;
    uint32_t maskCount = validBins;
    Reg::MaskReg activeMask = Reg::UpdateMask<uint32_t>(maskCount);
    __ubuf__ uint32_t* readAddr = dst;
    __ubuf__ uint32_t* writeAddr = dst;
    Reg::LoadUnAlignPre(loadState, readAddr);
    Reg::LoadUnAlign(totalCount, loadState, readAddr, validBins);
    Reg::Add(totalCount, totalCount, localCount, activeMask);
    Reg::StoreUnAlign(writeAddr, totalCount, storeState, validBins);
    Reg::StoreUnAlignPost(writeAddr, storeState, 0);
}

/** Export one 128-bin uint16 histogram register to uint32 counters in UB. */
__simd_callee__ inline void StoreHistogramBin(__ubuf__ uint32_t* dst, Reg::RegTensor<uint16_t>& histogram,
                                              uint32_t& remainCount)
{
    Reg::RegTensor<uint32_t> evenCount;
    Reg::RegTensor<uint32_t> oddCount;
    Reg::RegTensor<uint32_t> countReg0;
    Reg::RegTensor<uint32_t> countReg1;
    Reg::MaskReg allMask = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    Reg::Cast<uint32_t, uint16_t, HIST_U16_TO_U32_EVEN>(evenCount, histogram, allMask);
    Reg::Cast<uint32_t, uint16_t, HIST_U16_TO_U32_ODD>(oddCount, histogram, allMask);
    Reg::Interleave<uint32_t>(countReg0, countReg1, evenCount, oddCount);

    uint32_t curBinCount = remainCount > HISTOGRAM_BINS_PER_REG ? HISTOGRAM_BINS_PER_REG : remainCount;
    uint32_t firstRegCount = curBinCount > B32_ELEMENTS_PER_VECTOR ? B32_ELEMENTS_PER_VECTOR : curBinCount;
    StoreHistogramChunk(dst, countReg0, firstRegCount);
    if (curBinCount > B32_ELEMENTS_PER_VECTOR) {
        StoreHistogramChunk(dst + B32_ELEMENTS_PER_VECTOR, countReg1, curBinCount - B32_ELEMENTS_PER_VECTOR);
    }
    remainCount -= curBinCount;
}

/** Export four histogram registers, covering at most 512 bins, to UB. */
__simd_callee__ inline void StoreBatchHistogram(__ubuf__ uint32_t* dst, Reg::RegTensor<uint16_t>& histogramReg0,
                                                Reg::RegTensor<uint16_t>& histogramReg1,
                                                Reg::RegTensor<uint16_t>& histogramReg2,
                                                Reg::RegTensor<uint16_t>& histogramReg3, uint32_t validBinCount)
{
    __ubuf__ uint32_t* dstAddr = dst;
    uint32_t remainCount = validBinCount;
    StoreHistogramBin(dstAddr, histogramReg0, remainCount);
    if (remainCount == 0U) {
        return;
    }
    dstAddr += HISTOGRAM_BINS_PER_REG;
    StoreHistogramBin(dstAddr, histogramReg1, remainCount);
    if (remainCount == 0U) {
        return;
    }
    dstAddr += HISTOGRAM_BINS_PER_REG;
    StoreHistogramBin(dstAddr, histogramReg2, remainCount);
    if (remainCount == 0U) {
        return;
    }
    dstAddr += HISTOGRAM_BINS_PER_REG;
    StoreHistogramBin(dstAddr, histogramReg3, remainCount);
}

/** Clear four histogram registers covering one 512-bin batch. */
__simd_callee__ inline void ResetHistogramBatch(Reg::RegTensor<uint16_t>& histogramReg0,
                                                Reg::RegTensor<uint16_t>& histogramReg1,
                                                Reg::RegTensor<uint16_t>& histogramReg2,
                                                Reg::RegTensor<uint16_t>& histogramReg3, Reg::MaskReg& allMask)
{
    Reg::Duplicate(histogramReg0, static_cast<uint16_t>(0), allMask);
    Reg::Duplicate(histogramReg1, static_cast<uint16_t>(0), allMask);
    Reg::Duplicate(histogramReg2, static_cast<uint16_t>(0), allMask);
    Reg::Duplicate(histogramReg3, static_cast<uint16_t>(0), allMask);
}

/** Count one pair of B16 register vectors into a 256-bin histogram. */
__simd_callee__ inline void AccumulateHistogramBins(Reg::RegTensor<uint16_t>& histogramReg0,
                                                    Reg::RegTensor<uint16_t>& histogramReg1,
                                                    Reg::RegTensor<int16_t>& valueReg0,
                                                    Reg::RegTensor<int16_t>& valueReg1, Reg::MaskReg& activeMask0,
                                                    Reg::MaskReg& activeMask1, uint32_t binBase, uint32_t validBinCount)
{
    Reg::RegTensor<int16_t> relativeReg0;
    Reg::RegTensor<int16_t> relativeReg1;
    Reg::RegTensor<uint8_t> histogramInput;
    Reg::RegTensor<uint8_t> discardedInput;
    Reg::MaskReg inRangeMask0;
    Reg::MaskReg inRangeMask1;
    Reg::MaskReg histogramMask;
    Reg::MaskReg discardedMask;

    Reg::Adds(relativeReg0, valueReg0, -static_cast<int16_t>(binBase), activeMask0);
    Reg::Adds(relativeReg1, valueReg1, -static_cast<int16_t>(binBase), activeMask1);
    // range check, [binBase, binBase + validBinCount)
    Reg::CompareScalar<uint16_t, CMPMODE::LT>(inRangeMask0, reinterpret_cast<Reg::RegTensor<uint16_t>&>(relativeReg0),
                                              static_cast<uint16_t>(validBinCount), activeMask0);
    Reg::CompareScalar<uint16_t, CMPMODE::LT>(inRangeMask1, reinterpret_cast<Reg::RegTensor<uint16_t>&>(relativeReg1),
                                              static_cast<uint16_t>(validBinCount), activeMask1);
    // Histograms consumes the low byte of each B16 id; the range masks prevent truncation aliases.
    Reg::DeInterleave(histogramInput, discardedInput, reinterpret_cast<Reg::RegTensor<uint8_t>&>(relativeReg0),
                      reinterpret_cast<Reg::RegTensor<uint8_t>&>(relativeReg1));
    Reg::MaskDeInterleave<uint8_t>(histogramMask, discardedMask, inRangeMask0, inRangeMask1);
    Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::FREQUENCY>(
        histogramReg0, histogramInput, histogramMask);
    if (validBinCount > HISTOGRAM_BINS_PER_REG) {
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::FREQUENCY>(
            histogramReg1, histogramInput, histogramMask);
    }
}

/** Count one pair of B16 register vectors into a 512-bin histogram batch. */
__simd_callee__ inline void AccumulateHistogramBatch(
    Reg::RegTensor<uint16_t>& histogramReg0, Reg::RegTensor<uint16_t>& histogramReg1,
    Reg::RegTensor<uint16_t>& histogramReg2, Reg::RegTensor<uint16_t>& histogramReg3,
    Reg::RegTensor<int16_t>& valueReg0, Reg::RegTensor<int16_t>& valueReg1, Reg::MaskReg& activeMask0,
    Reg::MaskReg& activeMask1, uint32_t binBase, uint32_t validBinCount)
{
    uint32_t firstPassBinCount = validBinCount > HISTOGRAM_BINS_PER_PASS ? HISTOGRAM_BINS_PER_PASS : validBinCount;
    AccumulateHistogramBins(histogramReg0, histogramReg1, valueReg0, valueReg1, activeMask0, activeMask1, binBase,
                            firstPassBinCount);
    if (validBinCount > HISTOGRAM_BINS_PER_PASS) {
        AccumulateHistogramBins(histogramReg2, histogramReg3, valueReg0, valueReg1, activeMask0, activeMask1,
                                binBase + HISTOGRAM_BINS_PER_PASS, validBinCount - HISTOGRAM_BINS_PER_PASS);
    }
}

/** Scan UB ids once per 512-bin batch and accumulate into persistent uint32 counters. */
__simd_callee__ inline void CalSendCntPerExpert(__ubuf__ int16_t* expertIds, __ubuf__ int32_t* expertCounter,
                                                uint32_t elementCount, uint32_t expertCount)
{
    Reg::RegTensor<int16_t> valueReg0;
    Reg::RegTensor<int16_t> valueReg1;
    Reg::RegTensor<uint16_t> histogram0;
    Reg::RegTensor<uint16_t> histogram1;
    Reg::RegTensor<uint16_t> histogram2;
    Reg::RegTensor<uint16_t> histogram3;
    Reg::MaskReg allMask = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    uint16_t binLoopCount = static_cast<uint16_t>(Ceil(expertCount, HISTOGRAM_BINS_PER_BATCH));
    uint16_t dataLoopCount = static_cast<uint16_t>(Ceil(elementCount, HISTOGRAM_BINS_PER_PASS));

    for (uint16_t binLoop = 0; binLoop < binLoopCount; ++binLoop) {
        uint32_t binBase = static_cast<uint32_t>(binLoop) * HISTOGRAM_BINS_PER_BATCH;
        uint32_t validBinCount =
            (expertCount - binBase) > HISTOGRAM_BINS_PER_BATCH ? HISTOGRAM_BINS_PER_BATCH : (expertCount - binBase);
        uint32_t remaining = elementCount;
        __ubuf__ int16_t* valueAddr = expertIds;
        Reg::UnalignRegForLoad loadState;
        ResetHistogramBatch(histogram0, histogram1, histogram2, histogram3, allMask);
        Reg::LoadUnAlignPre(loadState, valueAddr);

        for (uint16_t dataLoop = 0; dataLoop < dataLoopCount; ++dataLoop) {
            uint32_t validElements = remaining > HISTOGRAM_BINS_PER_PASS ? HISTOGRAM_BINS_PER_PASS : remaining;
            uint32_t activeCount0 = validElements > HISTOGRAM_BINS_PER_REG ? HISTOGRAM_BINS_PER_REG : validElements;
            Reg::MaskReg activeMask0 = Reg::UpdateMask<int16_t>(activeCount0);
            Reg::MaskReg activeMask1 = Reg::CreateMask<int16_t, Reg::MaskPattern::ALLF>();
            if (validElements > HISTOGRAM_BINS_PER_REG) {
                uint32_t activeCount1 = validElements - HISTOGRAM_BINS_PER_REG;
                activeMask1 = Reg::UpdateMask<int16_t>(activeCount1);
            }
            Reg::LoadUnAlign(valueReg0, loadState, valueAddr, HISTOGRAM_BINS_PER_REG);
            Reg::LoadUnAlign(valueReg1, loadState, valueAddr, HISTOGRAM_BINS_PER_REG);
            AccumulateHistogramBatch(histogram0, histogram1, histogram2, histogram3, valueReg0, valueReg1, activeMask0,
                                     activeMask1, binBase, validBinCount);
            remaining -= validElements;
        }

        __ubuf__ uint32_t* counterAddr = reinterpret_cast<__ubuf__ uint32_t*>(expertCounter) + binBase;
        StoreBatchHistogram(counterAddr, histogram0, histogram1, histogram2, histogram3, validBinCount);
    }
}

/** Remove repeated destination ranks in each token row and count small-rank histograms in registers. */
__simd_callee__ inline void DedupAndSendDirect(__ubuf__ int16_t* dstRank, __ubuf__ int16_t* rawRank,
                                               __ubuf__ int32_t* rankCounter, uint32_t tokenCount, uint32_t topK,
                                               uint32_t rankCount)
{
    Reg::RegTensor<int16_t> rankReg;
    Reg::RegTensor<int16_t> dedupReg;
    Reg::RegTensor<int16_t> previousRankReg;
    Reg::RegTensor<int16_t> linearReg;
    Reg::RegTensor<int16_t> tokenReg;
    Reg::RegTensor<int16_t> rowBaseReg;
    Reg::RegTensor<int16_t> laneReg;
    Reg::RegTensor<int16_t> gatherIndexReg;
    Reg::RegTensor<int16_t> topKReg;
    Reg::RegTensor<int16_t> negOneReg;
    Reg::RegTensor<uint16_t> histogram0;
    Reg::RegTensor<uint16_t> histogram1;
    Reg::RegTensor<uint16_t> histogram2;
    Reg::RegTensor<uint16_t> histogram3;
    Reg::RegTensor<uint16_t> histogram4;
    Reg::RegTensor<uint16_t> histogram5;
    Reg::RegTensor<uint16_t> histogram6;
    Reg::RegTensor<uint16_t> histogram7;
    Reg::MaskReg earlierMask;
    Reg::MaskReg equalMask;
    Reg::MaskReg allMask = Reg::CreateMask<int16_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg noMask = Reg::CreateMask<int16_t, Reg::MaskPattern::ALLF>();
    Reg::UnalignRegForLoad loadState;
    Reg::UnalignRegForStore storeState;
    uint32_t tokensPerLoop = B16_ELEMENTS_PER_VECTOR / topK;
    uint32_t elementsPerLoop = tokensPerLoop * topK;
    uint32_t loopCount = Ceil(tokenCount, tokensPerLoop);
    uint32_t remaining = tokenCount * topK;
    __ubuf__ int16_t* rankReadAddr = rawRank;
    __ubuf__ int16_t* rankWriteAddr = dstRank;
    uint32_t firstBatchBinCount = rankCount > HISTOGRAM_BINS_PER_BATCH ? HISTOGRAM_BINS_PER_BATCH : rankCount;
    uint32_t secondBatchBinCount = rankCount > HISTOGRAM_BINS_PER_BATCH ? rankCount - HISTOGRAM_BINS_PER_BATCH : 0U;

    Reg::Arange(linearReg, 0);
    Reg::Duplicate(topKReg, static_cast<int16_t>(topK));
    Reg::Duplicate(negOneReg, static_cast<int16_t>(-1));
    ResetHistogramBatch(histogram0, histogram1, histogram2, histogram3, allMask);
    if (secondBatchBinCount > 0U) {
        ResetHistogramBatch(histogram4, histogram5, histogram6, histogram7, allMask);
    }
    Reg::Div(tokenReg, linearReg, topKReg, allMask);
    Reg::Muls(rowBaseReg, tokenReg, static_cast<int16_t>(topK), allMask);
    Reg::Sub(laneReg, linearReg, rowBaseReg, allMask);
    Reg::LoadUnAlignPre(loadState, rankReadAddr);

    for (uint16_t loop = 0; loop < static_cast<uint16_t>(loopCount); ++loop) {
        uint32_t validElements = remaining > elementsPerLoop ? elementsPerLoop : remaining;
        uint32_t maskCount = validElements;
        Reg::MaskReg activeMask = Reg::UpdateMask<int16_t>(maskCount);
        Reg::MaskReg duplicateMask = Reg::CreateMask<int16_t, Reg::MaskPattern::ALLF>();
        Reg::LoadUnAlign(rankReg, loadState, rankReadAddr, validElements);

        for (uint16_t k = 0; k < static_cast<uint16_t>(topK); ++k) {
            Reg::Adds(gatherIndexReg, rowBaseReg, static_cast<int16_t>(k), activeMask);
            Reg::Gather(previousRankReg, rankReg, reinterpret_cast<Reg::RegTensor<uint16_t>&>(gatherIndexReg));
            Reg::CompareScalar<int16_t, CMPMODE::GT>(earlierMask, laneReg, static_cast<int16_t>(k), activeMask);
            Reg::Compare<int16_t, CMPMODE::EQ>(equalMask, rankReg, previousRankReg, earlierMask);
            Reg::MaskOr(duplicateMask, duplicateMask, equalMask, activeMask);
        }
        Reg::Select(dedupReg, negOneReg, rankReg, duplicateMask);
        Reg::StoreUnAlign(rankWriteAddr, dedupReg, storeState, validElements);
        AccumulateHistogramBatch(histogram0, histogram1, histogram2, histogram3, dedupReg, negOneReg, activeMask,
                                 noMask, 0U, firstBatchBinCount);
        if (secondBatchBinCount > 0U) {
            AccumulateHistogramBatch(histogram4, histogram5, histogram6, histogram7, dedupReg, negOneReg, activeMask,
                                     noMask, HISTOGRAM_BINS_PER_BATCH, secondBatchBinCount);
        }
        remaining -= validElements;
    }

    Reg::StoreUnAlignPost(rankWriteAddr, storeState, 0);
    __ubuf__ uint32_t* rankCounterAddr = reinterpret_cast<__ubuf__ uint32_t*>(rankCounter);
    StoreBatchHistogram(rankCounterAddr, histogram0, histogram1, histogram2, histogram3, firstBatchBinCount);
    if (secondBatchBinCount > 0U) {
        StoreBatchHistogram(rankCounterAddr + HISTOGRAM_BINS_PER_BATCH, histogram4, histogram5, histogram6, histogram7,
                            secondBatchBinCount);
    }
}

/**
 * Direct count pipeline. One VF call performs conversion, both histograms and row-wise rank deduplication.
 */
__simd_vf__ __aicore__ inline void DedupAndCalCnt(__ubuf__ int32_t* topkIds, __ubuf__ int16_t* expertScratch,
                                                  __ubuf__ int16_t* rawRankScratch, __ubuf__ int16_t* dstRank,
                                                  __ubuf__ int32_t* expertCounter, __ubuf__ int32_t* rankCounter,
                                                  uint32_t elementCount, uint32_t tokenCount, uint32_t topK,
                                                  uint32_t expertCount, uint32_t rankCount, uint32_t expertsNumPerRank)
{
    PrepareExpertAndRank(topkIds, expertScratch, rawRankScratch, elementCount, expertsNumPerRank);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    CalSendCntPerExpert(expertScratch, expertCounter, elementCount, expertCount);
    DedupAndSendDirect(dstRank, rawRankScratch, rankCounter, tokenCount, topK, rankCount);
}

#endif

} // namespace MoeEpDispatchVf

#endif // MOE_EP_DISPATCH_VF_H
