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
 * \file lightning_indexer_kernel_arch35.h
 * \brief
 */

#ifndef LIGHTNING_INDEXER_KERNEL_ARCH35_H
#define LIGHTNING_INDEXER_KERNEL_ARCH35_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../lightning_indexer_common.h"
#include "common/lightning_indexer_kernel_base_arch35.h"
#include "lightning_indexer_service_vector_arch35.h"
#include "lightning_indexer_service_cube_arch35.h"

namespace LIKernel {
using namespace LICommon;
using namespace matmul;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

// 由于S2循环前，RunInfo还没有赋值，使用TempLoopInfo临时存放B、N、S1轴相关的信息；同时减少重复计算
struct TempLoopInfo {
    uint32_t bIdx = 0U;
    uint32_t bN2Idx = 0;
    uint32_t n2Idx = 0U;
    uint32_t gS1Idx = 0U;
    uint32_t gS1LoopEnd = 0U;    // gS1方向循环的结束Idx
    uint32_t s2LoopEnd = 0U;     // S2方向循环的结束Idx
    uint32_t actS1Size = 1U;     // 当前Batch循环处理的S1轴的实际大小
    uint32_t actS2SizeOrig = 0U; // 压缩前s2
    uint32_t actS2Size = 0U;
    uint32_t actMBaseSize = 0U;    // m轴(gS1)方向实际大小
    uint32_t mBasicSizeTail = 0U;  // gS1方向循环的尾基本块大小
    uint32_t s2BasicSizeTail = 0U; // S2方向循环的尾基本块大小
    bool curActSeqLenIsZero = false;
    bool needDealActS1LessThanS1 = false; // S1的实际长度小于shape的S1长度时，是否需要清理输出
};

template <typename LIT>
class LightningIndexerKernel {
public:
    __aicore__ inline LightningIndexerKernel(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
                                __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengthsK,
                                __gm__ uint8_t *blockTable, __gm__ uint8_t *sparseIndices, __gm__ uint8_t *sparseValues,
                                __gm__ uint8_t *workspace, const LITilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();

    // =================================类型定义区=================================
    static constexpr bool DT_W_FLAG = LIT::weightsTypeFlag;
    using Q_T = typename LIT::queryType;
    using K_T = typename LIT::keyType;
    using OUT_T = typename LIT::outputType;
    using SCORE_T = uint32_t;
    static constexpr bool PAGE_ATTENTION = LIT::pageAttention;
    static constexpr LI_LAYOUT LAYOUT_T = LIT::layout;
    static constexpr LI_LAYOUT K_LAYOUT_T = LIT::keyLayout;
    using W_T =
        typename LightningIndexerTypeTraits<Q_T, typename std::conditional<DT_W_FLAG, float, void>::type>::weightsType;

    LightningIndexerServiceCube<LIT> matmulService;
    LightningIndexerServiceVector<LIT> vectorService;

    // =================================常量区=================================
    static constexpr uint32_t SYNC_C1_V1_FLAG = 4;
    static constexpr uint32_t SYNC_V1_C1_FLAG = 5;

    static constexpr uint32_t M_BASE_SIZE = 256;
    static constexpr uint32_t S1_BASE_SIZE = 4;
    static constexpr uint32_t S1_BASE_SIZE_SMALL = 2;
    static constexpr uint32_t S2_BASE_SIZE = 128;
    static constexpr uint32_t HEAD_DIM = 128;
    static constexpr uint32_t K_HEAD_NUM = 1;
    static constexpr uint32_t GM_ALIGN_BYTES = 512;

    static constexpr int64_t LD_PREFETCH_LEN = 2;

protected:
    TPipe *pipe = nullptr;

    // offset
    uint64_t queryCoreOffset = 0ULL;
    uint64_t keyCoreOffset = 0ULL;
    uint64_t weightsCoreOffset = 0ULL;
    uint64_t indiceOutCoreOffset = 0ULL;
    uint64_t valueOutCoreOffset = 0ULL;
    // ================================Global Buffer区=================================
    GlobalTensor<Q_T> queryGm;
    GlobalTensor<K_T> keyGm;
    GlobalTensor<W_T> weightsGm;

    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<K_T> valueOutGm;
    GlobalTensor<int32_t> blockTableGm;

    GlobalTensor<uint32_t> actualSeqLengthsGmQ;
    GlobalTensor<uint32_t> actualSeqLengthsGmKv;

    // ================================类成员变量====================================
    // aic、aiv核信息
    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;
    uint32_t usedCoreNum = 0U;

    LICommon::ConstInfo constInfo{};
    TempLoopInfo tempLoopInfo{};
    LICommon::SplitCoreInfo splitCoreInfo{};

    // ================================Init functions==================================
    __aicore__ inline void InitTilingData(const LITilingData *__restrict tilingData);
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengthsK);
    // ================================Split Core================================
    __aicore__ inline void SplitCore(uint32_t curCoreIdx, uint32_t &coreNum, LICommon::SplitCoreInfo &info);
    __aicore__ inline uint32_t GetTotalBaseBlockNum();
    __aicore__ inline uint32_t GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size, uint32_t actS2SizeOrig);
    // ================================Process functions================================
    __aicore__ inline void ProcessMain();
    __aicore__ inline void ProcessBaseBlock(uint32_t liLoop, uint64_t s2LoopIdx, LICommon::RunInfo liRunInfo);
    __aicore__ inline void ProcessInvalid();
    // ================================Params Calc=====================================
    __aicore__ inline void CalcGS1LoopParams(uint32_t bN2Idx);
    __aicore__ inline void GetBN2Idx(uint32_t bN2Idx);
    __aicore__ inline uint32_t GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                               GlobalTensor<uint32_t> &actualSeqLengthsGmKv, uint32_t defaultSeqLen);
    __aicore__ inline void GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t &actS1Size, uint32_t &actS2Size,
                                               uint32_t &actS2SizeOrig);
    __aicore__ inline void CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx);
    __aicore__ inline void CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx, LICommon::RunInfo &runInfo);
    __aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, uint32_t s1Start);
};

template <typename LIT>
__aicore__ inline void LightningIndexerKernel<LIT>::InitTilingData(const LITilingData *__restrict tilingData)
{
    usedCoreNum = tilingData->usedCoreNum;
    constInfo.keyStride0 = tilingData->keyStride0;
    constInfo.batchSize = tilingData->bSize;
    constInfo.qHeadNum = constInfo.gSize = tilingData->gSize;
    constInfo.qSeqSize = tilingData->s1Size;
    constInfo.kSeqSize = tilingData->s2Size;
    constInfo.attenMaskFlag = (tilingData->sparseMode == 3);
    constInfo.kCacheBlockSize = tilingData->blockSize;
    constInfo.maxBlockNumPerBatch = tilingData->maxBlockNumPerBatch;
    constInfo.sparseCount = tilingData->sparseCount;
    constInfo.outputLayout = LAYOUT_T; // 输出和输入形状一致
    if constexpr (std::is_same_v<K_T, float16_t>) {
        constInfo.INVALID_VAL = 0xFC00;
    } else {
        constInfo.INVALID_VAL = 0xFF80;
    }
    if constexpr (LAYOUT_T == LI_LAYOUT::TND) {
        constInfo.isAccumSeqS1 = true;
    }
    if constexpr (K_LAYOUT_T == LI_LAYOUT::TND) {
        constInfo.isAccumSeqS2 = true;
    }

    constInfo.kHeadNum = K_HEAD_NUM;
    constInfo.headDim = HEAD_DIM;

    if (constInfo.sparseCount > 2048) {
        constInfo.mBaseSize = S1_BASE_SIZE_SMALL * constInfo.gSize;
        constInfo.s1BaseSize = S1_BASE_SIZE_SMALL;
    } else {
        constInfo.mBaseSize = S1_BASE_SIZE * constInfo.gSize;
        constInfo.s1BaseSize = S1_BASE_SIZE;
    }
    constInfo.s2BaseSize = S2_BASE_SIZE;
    constInfo.returnValueFlag = tilingData->returnValue;
    constInfo.splitMFlag = (constInfo.gSize == 64 && constInfo.sparseCount <= 2048);
}

template <typename LIT>
__aicore__ inline void LightningIndexerKernel<LIT>::InitBuffers()
{
    LIV2Common::InitBuffers(vectorService, matmulService, pipe);
}

template <typename LIT>
__aicore__ inline void LightningIndexerKernel<LIT>::InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ,
                                                                     __gm__ uint8_t *actualSeqLengthsK)
{
    LICommon::InitActualSeqLen(actualSeqLengthsQ, actualSeqLengthsK, constInfo, actualSeqLengthsGmQ,
                               actualSeqLengthsGmKv);
}

template <typename LIT>
__aicore__ inline uint32_t LightningIndexerKernel<LIT>::GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims,
                                                                        bool isAccumSeq,
                                                                        GlobalTensor<uint32_t> &actualSeqLengthsGmKv,
                                                                        uint32_t defaultSeqLen)
{
    return LICommon::GetActualSeqLen(bIdx, actualLenDims, isAccumSeq, actualSeqLengthsGmKv, defaultSeqLen);
}

template <typename LIT>
__aicore__ inline void LightningIndexerKernel<LIT>::GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t &actS1Size,
                                                                        uint32_t &actS2Size, uint32_t &actS2SizeOrig)
{
    actS1Size = GetActualSeqLen(bIdx, constInfo.actualLenQDims, constInfo.isAccumSeqS1, actualSeqLengthsGmQ,
                                constInfo.qSeqSize);
    actS2SizeOrig = GetActualSeqLen(bIdx, constInfo.actualLenDims, constInfo.isAccumSeqS2, actualSeqLengthsGmKv,
                                    constInfo.kSeqSize);
    actS2Size = actS2SizeOrig;
}

template <typename LIT>
__aicore__ inline uint32_t LightningIndexerKernel<LIT>::GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size,
                                                                                uint32_t actS2SizeOrig)
{
    return LIV2Common::GetMaskedS2BaseBlockNum(s1gIdx, actS1Size, actS2SizeOrig, constInfo.s1BaseSize, 1U,
                                               constInfo.s2BaseSize);
}

template <typename LIT>
__aicore__ inline uint32_t LightningIndexerKernel<LIT>::GetTotalBaseBlockNum()
{
    uint32_t liTotalBlockNum = 0;
    uint32_t actS1Size, actS2Size, actS2SizeOrig;
    uint32_t s1GBaseNum, s2BaseNum;
    for (uint32_t bIdx = 0; bIdx < constInfo.batchSize; bIdx++) {
        GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size, actS2SizeOrig);
        s1GBaseNum = CeilDiv(actS1Size, constInfo.s1BaseSize);
        if (!constInfo.attenMaskFlag) {
            s2BaseNum = constInfo.isLDOpen ? CeilDiv(actS2Size, constInfo.s2BaseSize) : (actS2Size > 0 ? 1 : 0);
            liTotalBlockNum += s1GBaseNum * s2BaseNum * constInfo.kHeadNum;
            continue;
        }
        for (uint32_t s1gIdx = 0; s1gIdx < s1GBaseNum; s1gIdx++) {
            s2BaseNum = constInfo.isLDOpen ? GetS2BaseBlockNumOnMask(s1gIdx, actS1Size, actS2SizeOrig) :
                                             (actS2Size > 0 ? 1 : 0);
            liTotalBlockNum += s2BaseNum * constInfo.kHeadNum;
        }
    }
    return liTotalBlockNum;
}

// 多核版本，双闭区间。基本原则：计算每个核最少处理的块数, 剩余的部分前面的核每个核多处理一块
template <typename LIT>
__aicore__ void inline LightningIndexerKernel<LIT>::SplitCore(uint32_t curCoreIdx, uint32_t &coreNum,
                                                              LICommon::SplitCoreInfo &liInfo)
{
    uint32_t liTotalBlockNum = GetTotalBaseBlockNum();
    uint32_t liMinBlockPerCore = liTotalBlockNum / coreNum;
    uint32_t liDeal1MoreBlockCoreNum = liTotalBlockNum % coreNum;
    uint32_t liCoreIdx = 0;
    uint32_t liLastGS1RemainBlockCnt = 0;
    uint32_t liCoreDealBlockCnt = liCoreIdx < liDeal1MoreBlockCoreNum ? liMinBlockPerCore + 1 : liMinBlockPerCore;
    coreNum = liMinBlockPerCore == 0 ? liDeal1MoreBlockCoreNum : coreNum;
    if (curCoreIdx < coreNum) {
        splitCoreInfo.isCoreEnable = true;
    } else {
        splitCoreInfo.isCoreEnable = false;
        return;
    }

    bool liFindLastCoreEnd = true;
    uint32_t liActS1Size, liActS2Size, liActS2SizeOrig;
    uint32_t liS1GBaseNum, liS2BaseNum, liS2Loop;
    for (uint32_t liBN2Idx = 0; liBN2Idx < constInfo.batchSize * constInfo.kHeadNum; liBN2Idx++) {
        uint32_t liBIdx = liBN2Idx / constInfo.kHeadNum;
        if (liBN2Idx % constInfo.kHeadNum == 0) {
            GetS1S2ActualSeqLen(liBIdx, liActS1Size, liActS2Size, liActS2SizeOrig);
            liS1GBaseNum = CeilDiv(liActS1Size, constInfo.s1BaseSize);
            liS2BaseNum = CeilDiv(liActS2Size, constInfo.s2BaseSize);
        }
        if constexpr (LAYOUT_T == LI_LAYOUT::BSND) {
            if (liFindLastCoreEnd && (liS1GBaseNum == 0U || liS2BaseNum == 0U)) {
                liInfo.bN2Start = liBN2Idx;
                liInfo.gS1Start = 0;
                liInfo.s2Start = 0;
                liFindLastCoreEnd = false;
            }
        }
        for (uint32_t liGS1Idx = 0; liGS1Idx < liS1GBaseNum; liGS1Idx++) {
            if (constInfo.attenMaskFlag) {
                liS2BaseNum = GetS2BaseBlockNumOnMask(liGS1Idx, liActS1Size, liActS2SizeOrig);
            }
            if (liFindLastCoreEnd && liS2BaseNum == 0U) {
                liInfo.bN2Start = liBN2Idx;
                liInfo.gS1Start = liGS1Idx;
                liInfo.s2Start = 0;
                liFindLastCoreEnd = false;
            }
            liS2Loop = constInfo.isLDOpen ? liS2BaseNum : (liActS2Size > 0 ? 1 : 0);
            for (uint32_t liS2Idx = 0; liS2Idx < liS2Loop;) {
                if (liFindLastCoreEnd) {
                    liInfo.bN2Start = liBN2Idx;
                    liInfo.gS1Start = liGS1Idx;
                    liInfo.s2Start = liS2Idx;
                    liFindLastCoreEnd = false;
                }
                uint32_t liS2RemainBaseNum = liS2Loop - liS2Idx;
                if (liLastGS1RemainBlockCnt + liS2RemainBaseNum >= liCoreDealBlockCnt) {
                    liInfo.bN2End = liBN2Idx;
                    liInfo.gS1End = liGS1Idx;
                    liInfo.s2End = constInfo.isLDOpen ? liS2Idx + liCoreDealBlockCnt - liLastGS1RemainBlockCnt - 1 :
                                                        liS2BaseNum - 1;

                    if (liCoreIdx == curCoreIdx) {
                        // S2被切N核，那么只有第一个核需要处理LD，其他核不用
                        if (liS2Idx == 0 && liInfo.s2End + 1 < liS2BaseNum) {
                            liInfo.isLD = true;
                        }
                        // 最后一个核处理的不是最后一个Batch，表明后面的Batch为空块(S2=0), 调整终点坐标以便清理输出
                        if (liCoreIdx == coreNum - 1 && liInfo.bN2End != constInfo.batchSize - 1) {
                            liInfo.bN2End = constInfo.batchSize - 1;
                            liInfo.s2End = 0;
                            liInfo.gS1End = 0;
                        }
                        return;
                    }
                    liCoreIdx++;
                    liFindLastCoreEnd = true;
                    liS2Idx = liInfo.s2End + 1;
                    liLastGS1RemainBlockCnt = 0;
                    liCoreDealBlockCnt =
                        liCoreIdx < liDeal1MoreBlockCoreNum ? liMinBlockPerCore + 1 : liMinBlockPerCore;
                } else {
                    liLastGS1RemainBlockCnt += liS2RemainBaseNum;
                    break;
                }
            }
        }
    }
}

template <typename LIT>
__aicore__ inline void LightningIndexerKernel<LIT>::DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, uint32_t s1Start)
{
    uint32_t tBase = 0U;
    if (constInfo.outputLayout == LI_LAYOUT::TND) {
        tBase = bIdx == 0 ? 0 : actualSeqLengthsGmQ.GetValue(bIdx - 1);
    }
    LIV2Common::DealActSeqLenIsZero<LI_LAYOUT::TND, LI_LAYOUT::BSND>(
        bIdx, n2Idx, s1Start, tBase, tempLoopInfo.actS1Size, constInfo.sparseCount, constInfo, vectorService);
}

template <typename LIT>
__aicore__ inline void LightningIndexerKernel<LIT>::Init(__gm__ uint8_t *query, __gm__ uint8_t *key,
                                                         __gm__ uint8_t *weights, __gm__ uint8_t *actualSeqLengthsQ,
                                                         __gm__ uint8_t *actualSeqLengthsK, __gm__ uint8_t *blockTable,
                                                         __gm__ uint8_t *sparseIndices, __gm__ uint8_t *sparseValues,
                                                         __gm__ uint8_t *workspace,
                                                         const LITilingData *__restrict tiling, TPipe *tPipe)
{
    if ASCEND_IS_AIV {
        tmpBlockIdx = GetBlockIdx(); // vec:0-47
        aiCoreIdx = tmpBlockIdx / 2;
    } else {
        tmpBlockIdx = GetBlockIdx(); // cube:0-23
        aiCoreIdx = tmpBlockIdx;
    }

    InitTilingData(tiling);
    InitActualSeqLen(actualSeqLengthsQ, actualSeqLengthsK);

    pipe = tPipe;

    // 获取分核信息
    SplitCore(aiCoreIdx, usedCoreNum, splitCoreInfo);

    uint64_t offset = 0;
    // vec 把整个s2的score存储在GM，大小为s1BaseSize * 16K * 4
    GlobalTensor<SCORE_T> scoreGm; // 存放vec核写出的score
    uint64_t singleCoreScoreSize = constInfo.s1BaseSize *
                                   LICommon::Align((uint64_t)constInfo.kSeqSize, (uint64_t)constInfo.s2BaseSize) *
                                   sizeof(SCORE_T);
    scoreGm.SetGlobalBuffer((__gm__ SCORE_T *)(workspace + aiCoreIdx * singleCoreScoreSize));
    offset += GetBlockNum() * singleCoreScoreSize;

    if ASCEND_IS_AIV {
        vectorService.InitParams(constInfo, tiling);
        indiceOutGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);
        valueOutGm.SetGlobalBuffer((__gm__ K_T *)sparseValues);
        weightsGm.SetGlobalBuffer((__gm__ W_T *)weights);
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        vectorService.InitVecInputTensor(weightsGm, indiceOutGm, valueOutGm, blockTableGm);
        vectorService.InitVecWorkspaceTensor(scoreGm);
    } else {
        matmulService.InitParams(constInfo);
        queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
        keyGm.SetGlobalBuffer((__gm__ K_T *)key);
        if constexpr (PAGE_ATTENTION) {
            blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        }
        matmulService.InitMm1GlobalTensor(blockTableGm, keyGm, queryGm);
    }
    InitBuffers();
}

template <typename LIT>
__aicore__ inline void LightningIndexerKernel<LIT>::GetBN2Idx(uint32_t bN2Idx)
{
    LIV2Common::GetBN2Idx(tempLoopInfo, constInfo, bN2Idx);
}

template <typename LIT>
__aicore__ inline void LightningIndexerKernel<LIT>::CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx)
{
    LIV2Common::CalcS2LoopParams<false>(tempLoopInfo, constInfo, splitCoreInfo, bN2LoopIdx, gS1LoopIdx, 1U);
}

template <typename LIT>
__aicore__ inline void LightningIndexerKernel<LIT>::CalcGS1LoopParams(uint32_t bN2LoopIdx)
{
    GetBN2Idx(bN2LoopIdx);
    GetS1S2ActualSeqLen(tempLoopInfo.bIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2Size, tempLoopInfo.actS2SizeOrig);
    LIV2Common::CalcGS1LoopParams<(LAYOUT_T == LI_LAYOUT::BSND)>(tempLoopInfo, constInfo, splitCoreInfo, bN2LoopIdx);
}

template <typename LIT>
__aicore__ inline void LightningIndexerKernel<LIT>::CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx,
                                                                LICommon::RunInfo &runInfo)
{
    if (!LIV2Common::InitRunInfoBase(loop, s2LoopIdx, runInfo, tempLoopInfo, constInfo, splitCoreInfo)) {
        return;
    }

    if (runInfo.isFirstS2InnerLoop) {
        uint64_t actualSeqQPrefixSum;
        if constexpr (LAYOUT_T == LI_LAYOUT::TND) {
            actualSeqQPrefixSum = (runInfo.bIdx <= 0) ? 0 : actualSeqLengthsGmQ.GetValue(runInfo.bIdx - 1);
        } else { // BSND
            actualSeqQPrefixSum = (runInfo.bIdx <= 0) ? 0 : runInfo.bIdx * constInfo.qSeqSize;
        }
        uint64_t tndBIdxOffset = actualSeqQPrefixSum * constInfo.qHeadNum * constInfo.headDim;
        // B,S1,N1(N2,G),D
        queryCoreOffset = tndBIdxOffset + runInfo.gS1Idx * constInfo.mBaseSize * constInfo.headDim;
        // B,S1,N1(N2,G)/T,N1(N2,G)
        weightsCoreOffset = actualSeqQPrefixSum * constInfo.qHeadNum + runInfo.n2Idx * constInfo.gSize;
        // B,S1,N2,k/T,N2,k
        indiceOutCoreOffset =
            actualSeqQPrefixSum * constInfo.kHeadNum * constInfo.sparseCount + runInfo.n2Idx * constInfo.sparseCount;
        // B,S1,N2,k/T,N2,k
        valueOutCoreOffset =
            actualSeqQPrefixSum * constInfo.kHeadNum * constInfo.sparseCount + runInfo.n2Idx * constInfo.sparseCount;
    }
    uint64_t actualSeqKPrefixSum;
    if constexpr (K_LAYOUT_T == LI_LAYOUT::TND) { // T N2 D
        actualSeqKPrefixSum = (runInfo.bIdx <= 0) ? 0 : actualSeqLengthsGmKv.GetValue(runInfo.bIdx - 1);
    } else {
        actualSeqKPrefixSum = (runInfo.bIdx <= 0) ? 0 : runInfo.bIdx * constInfo.kSeqSize;
    }
    uint64_t liTndBIdxOffsetForK = actualSeqKPrefixSum * constInfo.kHeadNum * constInfo.headDim;
    keyCoreOffset = liTndBIdxOffsetForK + runInfo.s2Idx * constInfo.s2BaseSize * constInfo.kHeadNum * constInfo.headDim;
    runInfo.tensorQueryOffset = queryCoreOffset;
    runInfo.tensorKeyOffset = keyCoreOffset;
    runInfo.tensorWeightsOffset = weightsCoreOffset;
    runInfo.indiceOutOffset = indiceOutCoreOffset;
    runInfo.valueOutOffset = valueOutCoreOffset;
}

template <typename LIT>
__aicore__ inline void LightningIndexerKernel<LIT>::Process()
{
    if (usedCoreNum == 0) {
        // 没有计算任务，直接清理输出
        ProcessInvalid();
        return;
    }

    ProcessMain();
}

template <typename LIT>
__aicore__ inline void LightningIndexerKernel<LIT>::ProcessInvalid()
{
    if ASCEND_IS_AIV {
        uint32_t aivCoreNum = GetBlockNum() * 2; // 2 means c:v = 1:2
        uint64_t totalOutputSize =
            constInfo.batchSize * constInfo.qSeqSize * constInfo.kHeadNum * constInfo.sparseCount;
        uint64_t singleCoreSize =
            LICommon::Align((totalOutputSize + aivCoreNum - 1) / aivCoreNum, GM_ALIGN_BYTES / sizeof(OUT_T));
        uint64_t baseSize = tmpBlockIdx * singleCoreSize;
        if (baseSize < totalOutputSize) {
            uint64_t dealSize =
                (baseSize + singleCoreSize <= totalOutputSize) ? singleCoreSize : totalOutputSize - baseSize;
            GlobalTensor<OUT_T> output = indiceOutGm[baseSize];
            AscendC::InitGlobalMemory(output, dealSize, constInfo.INVALID_IDX);
            if (constInfo.returnValueFlag) {
                GlobalTensor<uint16_t> valueOutGmTmp;
                valueOutGmTmp.SetGlobalBuffer((__gm__ uint16_t *)valueOutGm.GetPhyAddr());
                GlobalTensor<uint16_t> valueOut = valueOutGmTmp[baseSize];
                AscendC::InitGlobalMemory(valueOut, dealSize, constInfo.INVALID_VAL);
            }
        }
    }
}

template <typename LIT>
__aicore__ inline void LightningIndexerKernel<LIT>::ProcessMain()
{
    if (!splitCoreInfo.isCoreEnable) {
        return;
    }

    if ASCEND_IS_AIV {
        vectorService.AllocEventID();
        CrossCoreSetFlag<LICommon::ConstInfo::LI_SYNC_MODE4, PIPE_V>(LICommon::ConstInfo::CROSS_VC_EVENT + 0);
        CrossCoreSetFlag<LICommon::ConstInfo::LI_SYNC_MODE4, PIPE_V>(LICommon::ConstInfo::CROSS_VC_EVENT + 1);
    } else {
        matmulService.AllocEventID();
    }

    LICommon::RunInfo runInfo;
    uint32_t liGLoop = 0;
    for (uint32_t bN2LoopIdx = splitCoreInfo.bN2Start; bN2LoopIdx <= splitCoreInfo.bN2End; bN2LoopIdx++) {
        CalcGS1LoopParams(bN2LoopIdx);
        if (tempLoopInfo.curActSeqLenIsZero) {
            DealActSeqLenIsZero(tempLoopInfo.bIdx, tempLoopInfo.n2Idx, 0U);
            continue;
        }
        for (uint32_t gS1LoopIdx = splitCoreInfo.gS1Start; gS1LoopIdx <= tempLoopInfo.gS1LoopEnd; gS1LoopIdx++) {
            CalcS2LoopParams(bN2LoopIdx, gS1LoopIdx);
            for (int s2LoopIdx = splitCoreInfo.s2Start; s2LoopIdx <= tempLoopInfo.s2LoopEnd; s2LoopIdx++) {
                ProcessBaseBlock(liGLoop, s2LoopIdx, runInfo);
                ++liGLoop;
            }
            splitCoreInfo.s2Start = 0;
        }
        if (tempLoopInfo.needDealActS1LessThanS1) {
            DealActSeqLenIsZero(tempLoopInfo.bIdx, tempLoopInfo.n2Idx, tempLoopInfo.actS1Size);
        }
        splitCoreInfo.gS1Start = 0;
    }

    if ASCEND_IS_AIV {
        vectorService.FreeEventID();
    } else {
        matmulService.FreeEventID();
        CrossCoreWaitFlag<LICommon::ConstInfo::LI_SYNC_MODE4, PIPE_FIX>(LICommon::ConstInfo::CROSS_VC_EVENT + 0);
        CrossCoreWaitFlag<LICommon::ConstInfo::LI_SYNC_MODE4, PIPE_FIX>(LICommon::ConstInfo::CROSS_VC_EVENT + 1);
    }
}

template <typename LIT>
__aicore__ inline void LightningIndexerKernel<LIT>::ProcessBaseBlock(uint32_t liLoop, uint64_t s2LoopIdx,
                                                                     LICommon::RunInfo liRunInfo)
{
    CalcRunInfo(liLoop, s2LoopIdx, liRunInfo);
    if ASCEND_IS_AIC {
        matmulService.ComputeMm1(liRunInfo);
    } else {
        vectorService.ProcessVec1(liRunInfo);
        if (liRunInfo.isLastS2InnerLoop) { // 本核s2last
            vectorService.ProcessTopK(liRunInfo);
        }
    }
}

} // namespace LIKernel
#endif // LIGHTNING_INDEXER_KERNEL_ARCH35_H
