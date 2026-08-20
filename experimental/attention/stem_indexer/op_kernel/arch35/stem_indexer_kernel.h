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
 * \file stem_indexer_kernel.h
 * \brief
 */

#ifndef stem_indexer_KERNEL_H
#define stem_indexer_KERNEL_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "stem_indexer_common.h"
#include "stem_indexer_service_vector.h"
#include "stem_indexer_service_cube.h"
#include "../stem_indexer_metadata.h"

namespace SIKernel {
using namespace SICommon;
using namespace matmul;
using namespace optiling;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename SIT>
class SIPreload {
public:
    __aicore__ inline SIPreload(){};
    __aicore__ inline void Init(__gm__ uint8_t *qflat, __gm__ uint8_t *kflat, __gm__ uint8_t *vbias,
                                __gm__ uint8_t *qSeqLens, __gm__ uint8_t *kvSeqLens, __gm__ uint8_t *numPromptTokens,
                                __gm__ uint8_t *metadata, __gm__ uint8_t *sparseIndices, __gm__ uint8_t *sparseSeqLen,
                                __gm__ uint8_t *workspace, const StemIndexerTilingData *__restrict tiling,
                                TPipe *tPipe);
    __aicore__ inline void Process();

    // =================================类型定义区=================================
    using Q_T = typename SIT::queryType;
    using K_T = typename SIT::keyType;
    using OUT_T = typename SIT::outputType;
    static constexpr SI_LAYOUT Q_LAYOUT_T = SIT::layout;

    using SCORE_T = typename SIT::scoreType;

    SIMatmul<SIT> matmulService;
    SIVector<SIT> vectorService;

    // =================================常量区=================================
protected:
    TPipe *pipe = nullptr;

    // offset
    int64_t queryCoreOffset = 0LL;
    int64_t keyCoreOffset = 0LL;
    int64_t vbiasCoreOffset = 0LL;
    int64_t indiceOutCoreOffset = 0LL;
    int64_t indiceLenCoreOffset = 0LL;
    uint32_t cachedBIdx = UINT32_MAX;
    uint32_t cachedActS1Size = 0U;
    uint32_t cachedActS2Size = 0U;
    uint32_t cachedPromptLen = 0U;
    uint32_t s2BlockNum = 0U;
    bool isUsedCoreEqZero = false;
    bool needCleanOutput = false;
    bool useKvSeqLensAsNumPrompt = false;
    // ================================Global Buffer区=================================

    GlobalTensor<uint32_t> faMetaDataGm;
    GlobalTensor<Q_T> queryGm;
    GlobalTensor<K_T> keyGm;
    GlobalTensor<float> vbiasGm;
    GlobalTensor<int32_t> qSeqLensGm;
    GlobalTensor<int32_t> kvSeqLensGm;
    GlobalTensor<int32_t> numPromptTokensGm;
    GlobalTensor<int32_t> sparseIndicesGm;
    GlobalTensor<int32_t> sparseSeqLenGm;

    // ================================类成员变量====================================
    // aic、aiv核信息
    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;
    uint32_t usedCoreNum = 0U;
    uint32_t sectionNum_ = 0U;

    SICommon::ConstInfo constInfo{};
    SICommon::TempLoopInfo tempLoopInfo{};
    SICommon::SplitCoreInfo splitCoreInfo{};

    // ================================Init functions==================================
    __aicore__ inline void InitTilingData(const StemIndexerTilingData *__restrict tilingData);
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengthsK);
    __aicore__ inline bool NeedCleanOutput();
    // ================================Split Core================================
    __aicore__ inline void GetFASectionInfo(uint32_t sectionIdx);
    __aicore__ inline uint32_t GetS2BaseBlockNumOnMask(uint32_t gS1Idx, uint32_t actS1Size, uint32_t actS2Size);
    __aicore__ inline uint32_t CalcS2ValidSize(uint32_t gS1Idx, uint32_t actS1Size, uint32_t actS2Size);
    // ================================Process functions================================
    __aicore__ inline void WaitMm1CrossCoreSlots();
    __aicore__ inline void ProcessMain();
    __aicore__ inline void ProcessBaseBlock(uint32_t loop, uint32_t s2LoopIdx, SICommon::RunInfo &runInfo);
    // ================================Params Calc=====================================
    __aicore__ inline void CalcGS1LoopParams(uint32_t bN2Idx);
    __aicore__ inline void GetBN2Idx(uint32_t bN2Idx);
    __aicore__ inline uint32_t GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims, GlobalTensor<int32_t> &seqLensGm,
                                               uint32_t defaultSeqLen);
    __aicore__ inline void GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t &actS1Size, uint32_t &actS2Size);
    __aicore__ inline void CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx);
    __aicore__ inline void CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx, SICommon::RunInfo &runInfo);
};

template <typename SIT>
__aicore__ inline void SIPreload<SIT>::InitTilingData(const StemIndexerTilingData *__restrict tilingData)
{
    usedCoreNum = tilingData->usedCoreNum;
    useKvSeqLensAsNumPrompt = tilingData->useKvSeqLensAsNumPrompt != 0U;
    constInfo.usedCoreNum = tilingData->usedCoreNum;
    constInfo.batchSize = tilingData->bSize;
    constInfo.qHeadNum = tilingData->qHeadNum;
    constInfo.kvHeadNum = tilingData->kvHeadNum;
    constInfo.gSize = tilingData->gSize;
    constInfo.maxQb = tilingData->maxQb;
    constInfo.maxKb = tilingData->maxKb;
    constInfo.qSeqSize = tilingData->maxQb;
    constInfo.kSeqSize = tilingData->maxKb;
    constInfo.headDim = tilingData->headDim;
    constInfo.sparseCount = tilingData->maxKb;
    constInfo.causal = tilingData->causal;
    constInfo.stemBlockSize = tilingData->stemBlockSize;
    constInfo.stemStride = tilingData->stemStride;
    constInfo.initialBlocks = tilingData->initialBlocks;
    constInfo.windowSize = tilingData->windowSize;
    constInfo.mBaseSize = tilingData->mBaseSize;
    constInfo.s2BaseSize = tilingData->s2BaseSize;
    constInfo.s1BaseSize = CeilDiv(tilingData->mBaseSize, tilingData->gSize);
    constInfo.rSquare = tilingData->rSquare;
    constInfo.alpha = tilingData->alpha;
    constInfo.kBlockNumRateMedium = tilingData->kBlockNumRateMedium;
    constInfo.kBlockNumBiasMedium = tilingData->kBlockNumBiasMedium;
    constInfo.kBlockNumRateLarge = tilingData->kBlockNumRateLarge;
    constInfo.kBlockNumBiasLarge = tilingData->kBlockNumBiasLarge;
    constInfo.outputLayout = Q_LAYOUT_T; // 输出和输入形状一致
    constInfo.attenMaskFlag = tilingData->causal != 0U;
}

template <typename SIT>
__aicore__ inline void SIPreload<SIT>::InitBuffers()
{
    if ASCEND_IS_AIV {
        vectorService.InitBuffers(pipe, needCleanOutput);
    } else {
        matmulService.InitBuffers(pipe);
    }
}

template <typename SIT>
__aicore__ inline void SIPreload<SIT>::InitActualSeqLen(__gm__ uint8_t *qSeqLens, __gm__ uint8_t *kvSeqLens)
{
    if (qSeqLens == nullptr) {
        constInfo.actualLenQDims = 0;
    } else {
        constInfo.actualLenQDims = constInfo.batchSize;
        qSeqLensGm.SetGlobalBuffer((__gm__ int32_t *)qSeqLens, constInfo.actualLenQDims);
    }
    if (kvSeqLens == nullptr) {
        constInfo.actualLenDims = 0;
    } else {
        constInfo.actualLenDims = constInfo.batchSize;
        kvSeqLensGm.SetGlobalBuffer((__gm__ int32_t *)kvSeqLens, constInfo.actualLenDims);
    }
}

template <typename SIT>
__aicore__ inline bool SIPreload<SIT>::NeedCleanOutput()
{
    if (isUsedCoreEqZero) {
        return true;
    }
    for (uint32_t bIdx = 0U; bIdx < constInfo.batchSize; ++bIdx) {
        uint32_t actS1Size = 0U;
        uint32_t actS2Size = 0U;
        GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size);
        if (actS1Size == 0U || actS2Size == 0U) {
            return true;
        }
        if constexpr (Q_LAYOUT_T == SI_LAYOUT::BNSD) {
            if (actS1Size != constInfo.qSeqSize) {
                return true;
            }
        }
    }
    return false;
}

template <typename SIT>
__aicore__ inline uint32_t SIPreload<SIT>::GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims,
                                                           GlobalTensor<int32_t> &seqLensGm, uint32_t defaultSeqLen)
{
    if (actualLenDims == 0) {
        return defaultSeqLen;
    }
    return static_cast<uint32_t>(seqLensGm.GetValue(bIdx));
}

template <typename SIT>
__aicore__ inline void SIPreload<SIT>::GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t &actS1Size, uint32_t &actS2Size)
{
    uint32_t qTokenLen =
        GetActualSeqLen(bIdx, constInfo.actualLenQDims, qSeqLensGm, constInfo.qSeqSize * constInfo.stemBlockSize);
    uint32_t kTokenLen =
        GetActualSeqLen(bIdx, constInfo.actualLenDims, kvSeqLensGm, constInfo.kSeqSize * constInfo.stemBlockSize);
    actS1Size = CeilDiv(qTokenLen, constInfo.stemBlockSize);
    actS2Size = CeilDiv(kTokenLen, constInfo.stemBlockSize);
}

template <typename SIT>
__aicore__ inline uint32_t SIPreload<SIT>::GetS2BaseBlockNumOnMask(uint32_t gS1Idx, uint32_t actS1Size,
                                                                   uint32_t actS2Size)
{
    uint32_t s2ValidSize = CalcS2ValidSize(gS1Idx, actS1Size, actS2Size);
    if (s2ValidSize == 0U) {
        return 0;
    }
    return CeilDiv(s2ValidSize, constInfo.s2BaseSize);
}

template <typename SIT>
__aicore__ inline uint32_t SIPreload<SIT>::CalcS2ValidSize(uint32_t gS1Idx, uint32_t actS1Size, uint32_t actS2Size)
{
    uint32_t validS2SizeWithWindow = 0U;
    if (actS2Size == 0U) {
        return 0U;
    }
    if (!constInfo.attenMaskFlag) {
        validS2SizeWithWindow = actS2Size;
    } else {
        if (actS1Size == 0U || constInfo.gSize == 0U) {
            return 0U;
        }

        uint64_t totalMSize = static_cast<uint64_t>(actS1Size) * constInfo.gSize;
        uint64_t mBlockStart = static_cast<uint64_t>(gS1Idx) * constInfo.mBaseSize;
        if (mBlockStart >= totalMSize) {
            return 0U;
        }
        uint64_t mBlockEnd = Min(mBlockStart + constInfo.mBaseSize, totalMSize);
        uint64_t mBlockLen = mBlockEnd - mBlockStart;
        uint32_t firstS1Idx = static_cast<uint32_t>(mBlockStart % actS1Size);
        uint32_t lastS1Idx = static_cast<uint32_t>((mBlockEnd - 1U) % actS1Size);
        uint32_t maxS1Idx = (mBlockLen >= actS1Size || lastS1Idx < firstS1Idx) ? (actS1Size - 1U) : lastS1Idx;
        uint32_t qBlockEnd = maxS1Idx + 1U;

        // For causal mode, KV may contain prefix/cache blocks before the current Q range.
        // Shift qBlockEnd by the KV-Q block offset to get the visible K block count.
        int64_t validS2Size =
            static_cast<int64_t>(actS2Size) - static_cast<int64_t>(actS1Size) + static_cast<int64_t>(qBlockEnd);
        validS2SizeWithWindow = static_cast<uint32_t>(Max(validS2Size, static_cast<int64_t>(0)));
    }
    return (validS2SizeWithWindow > constInfo.windowSize) ? (validS2SizeWithWindow - constInfo.windowSize) : 0U;
}

template <typename SIT>
__aicore__ inline void SIPreload<SIT>::GetFASectionInfo(uint32_t sectionIdx)
{
    uint32_t sectionBase = SLI_PER_CORE_STRIDE * (AIC_CORE_NUM * sectionIdx + aiCoreIdx);
    uint32_t bN2StartIndex = sectionBase + SLI_SEC_BN2_START_INDEX;
    uint32_t mStartIndex = sectionBase + SLI_SEC_M_START_INDEX;
    uint32_t s2StartIndex = sectionBase + SLI_SEC_S2_START_INDEX;
    uint32_t bN2EndIndex = sectionBase + SLI_SEC_BN2_END_INDEX;
    uint32_t mEndIndex = sectionBase + SLI_SEC_M_END_INDEX;
    uint32_t s2EndIndex = sectionBase + SLI_SEC_S2_END_INDEX;

    uint32_t bN2EndRhs = faMetaDataGm.GetValue(bN2EndIndex);
    uint32_t mEndRhs = faMetaDataGm.GetValue(mEndIndex);
    uint32_t s2EndRhs = faMetaDataGm.GetValue(s2EndIndex);
    if (bN2EndRhs == 0U && mEndRhs == 0U && s2EndRhs == 0U) {
        splitCoreInfo.isCoreEnable = false;
        return;
    }
    splitCoreInfo.isCoreEnable = true;

    splitCoreInfo.bN2Start = faMetaDataGm.GetValue(bN2StartIndex);
    splitCoreInfo.gS1Start = faMetaDataGm.GetValue(mStartIndex);
    splitCoreInfo.s2Start = faMetaDataGm.GetValue(s2StartIndex);
    splitCoreInfo.bN2End = bN2EndRhs;
    splitCoreInfo.gS1End = mEndRhs;
    splitCoreInfo.s2End = s2EndRhs;

    if (splitCoreInfo.s2End != 0) {
        splitCoreInfo.s2End = splitCoreInfo.s2End - 1;
    } else {
        if (splitCoreInfo.gS1End != 0) {
            splitCoreInfo.gS1End = splitCoreInfo.gS1End - 1;
            uint32_t bIdx = splitCoreInfo.bN2End / constInfo.kvHeadNum;
            uint32_t actS1Size, actS2Size;
            GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size);
            uint32_t s2BaseNum = GetS2BaseBlockNumOnMask(splitCoreInfo.gS1End, actS1Size, actS2Size);
            splitCoreInfo.s2End = (s2BaseNum == 0U) ? 0U : s2BaseNum - 1U;
        } else {
            splitCoreInfo.bN2End = splitCoreInfo.bN2End - 1;
            uint32_t bIdx = splitCoreInfo.bN2End / constInfo.kvHeadNum;
            uint32_t actS1Size, actS2Size;
            GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size);
            uint32_t gS1BaseNum = static_cast<uint32_t>(CeilDiv(static_cast<uint64_t>(actS1Size) * constInfo.gSize,
                                                                static_cast<uint64_t>(constInfo.mBaseSize)));
            splitCoreInfo.gS1End = (gS1BaseNum == 0U) ? 0U : gS1BaseNum - 1U;
            uint32_t s2BaseNum = GetS2BaseBlockNumOnMask(splitCoreInfo.gS1End, actS1Size, actS2Size);
            splitCoreInfo.s2End = (s2BaseNum == 0U) ? 0U : s2BaseNum - 1U;
        }
    }
}

template <typename SIT>
__aicore__ inline void SIPreload<SIT>::Init(__gm__ uint8_t *qflat, __gm__ uint8_t *kflat, __gm__ uint8_t *vbias,
                                            __gm__ uint8_t *qSeqLens, __gm__ uint8_t *kvSeqLens,
                                            __gm__ uint8_t *numPromptTokens, __gm__ uint8_t *metadata,
                                            __gm__ uint8_t *sparseIndices, __gm__ uint8_t *sparseSeqLen,
                                            __gm__ uint8_t *workspace, const StemIndexerTilingData *__restrict tiling,
                                            TPipe *tPipe)
{
    if ASCEND_IS_AIV {
        tmpBlockIdx = GetBlockIdx(); // vec:0-47
        aiCoreIdx = tmpBlockIdx >> 1U;
    } else {
        tmpBlockIdx = GetBlockIdx(); // cube:0-23
        aiCoreIdx = tmpBlockIdx;
    }

    InitTilingData(tiling);
    InitActualSeqLen(qSeqLens, kvSeqLens);
    __gm__ uint8_t *effectiveNumPromptTokens = useKvSeqLensAsNumPrompt ? kvSeqLens : numPromptTokens;
    numPromptTokensGm.SetGlobalBuffer((__gm__ int32_t *)effectiveNumPromptTokens, constInfo.batchSize);

    sectionNum_ = ((__gm__ uint32_t *)metadata)[0];
    if (sectionNum_ == 0U) {
        isUsedCoreEqZero = true;
    }
    faMetaDataGm.SetGlobalBuffer((__gm__ uint32_t *)(metadata + SLI_METADATA_HEADER_OFFSET),
                                 AIC_CORE_NUM * SLI_PER_CORE_STRIDE * sectionNum_);

    if ASCEND_IS_AIV {
        needCleanOutput = NeedCleanOutput();
    }

    pipe = tPipe;

    if ASCEND_IS_AIV {
        vbiasGm.SetGlobalBuffer((__gm__ float *)vbias);
        sparseIndicesGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);
        sparseSeqLenGm.SetGlobalBuffer((__gm__ int32_t *)sparseSeqLen);
        vectorService.InitParams(constInfo);
        vectorService.InitVecInputTensor(vbiasGm, sparseIndicesGm, sparseSeqLenGm);
    } else {
        queryGm.SetGlobalBuffer((__gm__ Q_T *)qflat);
        keyGm.SetGlobalBuffer((__gm__ K_T *)kflat);
        queryGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        keyGm.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL);
        matmulService.InitParams(constInfo);
        matmulService.InitMm1GlobalTensor(queryGm, keyGm);
    }

    InitBuffers();
}

template <typename SIT>
__aicore__ inline void SIPreload<SIT>::GetBN2Idx(uint32_t bN2Idx)
{
    tempLoopInfo.bN2Idx = bN2Idx;
    tempLoopInfo.bIdx = bN2Idx / constInfo.kvHeadNum;
    tempLoopInfo.n2Idx = bN2Idx % constInfo.kvHeadNum;
}

template <typename SIT>
__aicore__ inline void SIPreload<SIT>::CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx)
{
    tempLoopInfo.gS1Idx = gS1LoopIdx;
    tempLoopInfo.actMBaseSize = constInfo.mBaseSize;
    uint32_t remainedGS1Size = tempLoopInfo.actS1Size * constInfo.gSize - tempLoopInfo.gS1Idx * constInfo.mBaseSize;
    if (remainedGS1Size <= constInfo.mBaseSize && remainedGS1Size > 0) {
        tempLoopInfo.actMBaseSize = tempLoopInfo.mBasicSizeTail;
    }
    tempLoopInfo.s2ValidSize = CalcS2ValidSize(gS1LoopIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2Size);
    tempLoopInfo.s2BasicSizeTail = tempLoopInfo.s2ValidSize % constInfo.s2BaseSize;
    tempLoopInfo.s2BasicSizeTail =
        (tempLoopInfo.s2BasicSizeTail == 0) ? constInfo.s2BaseSize : tempLoopInfo.s2BasicSizeTail;

    bool isEnd = (bN2LoopIdx == splitCoreInfo.bN2End) && (gS1LoopIdx == splitCoreInfo.gS1End);
    s2BlockNum = CeilDiv(tempLoopInfo.s2ValidSize, constInfo.s2BaseSize);
    if (s2BlockNum == 0U) {
        tempLoopInfo.s2LoopEnd = 0U;
        return;
    }
    uint32_t tileS2LoopEnd = s2BlockNum - 1U;
    tempLoopInfo.s2LoopEnd = isEnd ? splitCoreInfo.s2End : tileS2LoopEnd;
}

template <typename SIT>
__aicore__ inline void SIPreload<SIT>::CalcGS1LoopParams(uint32_t bN2LoopIdx)
{
    GetBN2Idx(bN2LoopIdx);
    tempLoopInfo.s2ValidSize = 0U;
    if (tempLoopInfo.bIdx != cachedBIdx) {
        GetS1S2ActualSeqLen(tempLoopInfo.bIdx, cachedActS1Size, cachedActS2Size);
        uint32_t promptTokenLen =
            GetActualSeqLen(tempLoopInfo.bIdx, static_cast<uint32_t>(constInfo.batchSize), numPromptTokensGm, 0U);
        cachedPromptLen = CeilDiv(promptTokenLen, constInfo.stemBlockSize);
        cachedBIdx = tempLoopInfo.bIdx;
    }
    tempLoopInfo.actS1Size = cachedActS1Size;
    tempLoopInfo.actS2Size = cachedActS2Size;
    tempLoopInfo.promptLen = cachedPromptLen;
    if ((tempLoopInfo.actS2Size == 0) || (tempLoopInfo.actS1Size == 0)) {
        tempLoopInfo.curActSeqLenIsZero = true;
        return;
    }
    tempLoopInfo.curActSeqLenIsZero = false;
    tempLoopInfo.mBasicSizeTail = (tempLoopInfo.actS1Size * constInfo.gSize) % constInfo.mBaseSize;
    tempLoopInfo.mBasicSizeTail =
        (tempLoopInfo.mBasicSizeTail == 0) ? constInfo.mBaseSize : tempLoopInfo.mBasicSizeTail;

    uint32_t gS1SplitNum = (tempLoopInfo.actS1Size * constInfo.gSize + constInfo.mBaseSize - 1) / constInfo.mBaseSize;
    tempLoopInfo.gS1LoopEnd = (bN2LoopIdx == splitCoreInfo.bN2End) ? splitCoreInfo.gS1End : gS1SplitNum - 1;

    const int64_t bIdx = static_cast<int64_t>(tempLoopInfo.bIdx);
    const int64_t n2Idx = static_cast<int64_t>(tempLoopInfo.n2Idx);
    const int64_t qHeadNum = static_cast<int64_t>(constInfo.qHeadNum);
    const int64_t kvHeadNum = static_cast<int64_t>(constInfo.kvHeadNum);
    const int64_t gSize = static_cast<int64_t>(constInfo.gSize);
    const int64_t qSeqSize = static_cast<int64_t>(constInfo.qSeqSize);
    const int64_t kSeqSize = static_cast<int64_t>(constInfo.kSeqSize);
    const int64_t headDim = static_cast<int64_t>(constInfo.headDim);
    queryCoreOffset = bIdx * qHeadNum * qSeqSize * headDim + n2Idx * gSize * qSeqSize * headDim;
    keyCoreOffset = bIdx * kvHeadNum * kSeqSize * headDim + n2Idx * kSeqSize * headDim;
    vbiasCoreOffset = bIdx * kvHeadNum * kSeqSize + n2Idx * kSeqSize;
    indiceOutCoreOffset = bIdx * qHeadNum * qSeqSize * kSeqSize + n2Idx * gSize * qSeqSize * kSeqSize;
    indiceLenCoreOffset = bIdx * qHeadNum * qSeqSize + n2Idx * gSize * qSeqSize;
}

template <typename SIT>
__aicore__ inline void SIPreload<SIT>::CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx, SICommon::RunInfo &runInfo)
{
    runInfo.loop = loop;
    runInfo.bN2Idx = tempLoopInfo.bN2Idx;
    runInfo.bIdx = tempLoopInfo.bIdx;
    runInfo.n2Idx = tempLoopInfo.n2Idx;
    runInfo.gS1Idx = tempLoopInfo.gS1Idx;
    runInfo.s2Idx = s2LoopIdx;
    runInfo.promptLen = tempLoopInfo.promptLen;

    runInfo.s2Start = splitCoreInfo.s2Start;
    runInfo.s2LoopEnd = tempLoopInfo.s2LoopEnd;

    runInfo.actS1Size = tempLoopInfo.actS1Size;
    runInfo.actS2Size = tempLoopInfo.actS2Size;
    // 计算实际基本块size
    runInfo.actMBaseSize = tempLoopInfo.actMBaseSize;
    runInfo.actualSingleProcessSInnerSize = constInfo.s2BaseSize;
    if (runInfo.s2Idx == s2BlockNum - 1U) {
        runInfo.actualSingleProcessSInnerSize = tempLoopInfo.s2BasicSizeTail;
    }

    runInfo.isFirstS2InnerLoop = s2LoopIdx == splitCoreInfo.s2Start;
    runInfo.isLastS2InnerLoop = s2LoopIdx == tempLoopInfo.s2LoopEnd;

    runInfo.tensorQueryOffset = queryCoreOffset;
    runInfo.indiceOutOffset = indiceOutCoreOffset;
    runInfo.indiceLenOffset = indiceLenCoreOffset;

    runInfo.tensorKeyOffset = keyCoreOffset + static_cast<int64_t>(runInfo.s2Idx) *
                                                  static_cast<int64_t>(constInfo.s2BaseSize) *
                                                  static_cast<int64_t>(constInfo.headDim);

    runInfo.tensorVBiasOffset = vbiasCoreOffset;
}

template <typename SIT>
__aicore__ inline void SIPreload<SIT>::WaitMm1CrossCoreSlots()
{
    CrossCoreWaitFlag<SICommon::SI_SYNC_MODE4, PIPE_FIX>(SICommon::CROSS_VC_EVENT + 0U);
    CrossCoreWaitFlag<SICommon::SI_SYNC_MODE4, PIPE_FIX>(SICommon::CROSS_VC_EVENT + 1U);
    CrossCoreWaitFlag<SICommon::SI_SYNC_MODE4, PIPE_FIX>(SICommon::CROSS_VC_EVENT + 2U);
    CrossCoreWaitFlag<SICommon::SI_SYNC_MODE4, PIPE_FIX>(SICommon::CROSS_VC_EVENT + SICommon::AIV0_AIV1_OFFSET + 0U);
    CrossCoreWaitFlag<SICommon::SI_SYNC_MODE4, PIPE_FIX>(SICommon::CROSS_VC_EVENT + SICommon::AIV0_AIV1_OFFSET + 1U);
    CrossCoreWaitFlag<SICommon::SI_SYNC_MODE4, PIPE_FIX>(SICommon::CROSS_VC_EVENT + SICommon::AIV0_AIV1_OFFSET + 2U);
}

template <typename SIT>
__aicore__ inline void SIPreload<SIT>::Process()
{
    if (isUsedCoreEqZero) {
        if ASCEND_IS_AIC {
            WaitMm1CrossCoreSlots();
        }
        return;
    }

    if ASCEND_IS_AIV {
        vectorService.AllocEventID();
    } else {
        matmulService.AllocEventID();
    }

    for (uint32_t sectionIdx = 0; sectionIdx < sectionNum_; sectionIdx++) {
        GetFASectionInfo(sectionIdx);
        if (!splitCoreInfo.isCoreEnable) {
            continue;
        }
        ProcessMain();
    }

    if ASCEND_IS_AIV {
        vectorService.FreeEventID();
    } else {
        matmulService.FreeEventID();
        WaitMm1CrossCoreSlots();
    }
}

template <typename SIT>
__aicore__ inline void SIPreload<SIT>::ProcessMain()
{
    SICommon::RunInfo runInfo{};
    uint32_t gloop = 0;
    for (uint32_t bN2LoopIdx = splitCoreInfo.bN2Start; bN2LoopIdx <= splitCoreInfo.bN2End; bN2LoopIdx++) {
        CalcGS1LoopParams(bN2LoopIdx);
        if (tempLoopInfo.curActSeqLenIsZero) {
            continue;
        }

        for (uint32_t gS1LoopIdx = splitCoreInfo.gS1Start; gS1LoopIdx <= tempLoopInfo.gS1LoopEnd; gS1LoopIdx++) {
            CalcS2LoopParams(bN2LoopIdx, gS1LoopIdx);
            if (tempLoopInfo.s2ValidSize <= constInfo.initialBlocks) {
                vectorService.ProcessDirectOutput(tempLoopInfo, gS1LoopIdx);
                splitCoreInfo.s2Start = 0;
                continue;
            }

            for (uint32_t s2LoopIdx = splitCoreInfo.s2Start; s2LoopIdx <= tempLoopInfo.s2LoopEnd; s2LoopIdx++) {
                ProcessBaseBlock(gloop, s2LoopIdx, runInfo);
                ++gloop;
            }
            splitCoreInfo.s2Start = 0;
        }
        splitCoreInfo.gS1Start = 0;
    }
}

template <typename SIT>
__aicore__ inline void SIPreload<SIT>::ProcessBaseBlock(uint32_t loop, uint32_t s2LoopIdx, SICommon::RunInfo &runInfo)
{
    CalcRunInfo(loop, s2LoopIdx, runInfo);
    if ASCEND_IS_AIC {
        matmulService.ComputeMm1(runInfo);
    } else {
        if (!needCleanOutput && runInfo.isFirstS2InnerLoop) {
            vectorService.InitSparseIndicesToNegOne(runInfo.gS1Idx, runInfo.actMBaseSize, runInfo.actS1Size,
                                                    runInfo.indiceOutOffset);
        }
        vectorService.ProcessVec1(runInfo);
        vectorService.ProcessTopK(runInfo, runInfo.isFirstS2InnerLoop, runInfo.isLastS2InnerLoop);
    }
}

} // namespace SIKernel
#endif // stem_indexer_KERNEL_H
