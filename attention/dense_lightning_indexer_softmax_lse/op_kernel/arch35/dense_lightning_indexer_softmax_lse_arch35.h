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
 * \file dense_lightning_indexer_softmax_lse_arch35.h
 * \brief A5架构适配的kernel头文件
 */

#ifndef __DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_H___ARCH35
#define __DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_H___ARCH35

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../dense_lightning_indexer_softmax_lse_common.h"
#include "dense_lightning_indexer_softmax_lse_service_cube_arch35.h"
#include "dense_lightning_indexer_softmax_lse_service_vector_arch35.h"

namespace DenseLISoftmaxLseKernel {
using namespace DenseLISoftmaxLseCommon;
using namespace AscendC;
using namespace matmul;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

// 由于S2循环前，RunInfo还没有赋值，使用TempLoopInfo临时存放B、N、S1轴相关的信息；同时减少重复计算
struct TempLoopInfo {
    uint32_t bN2Idx = 0;
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;
    uint32_t gS1Idx = 0U;
    uint32_t gS1LoopEnd = 0U;  // gS1方向循环的结束Idx
    uint32_t s2LoopEnd = 0U;   // S2方向循环的结束Idx
    uint32_t actS1Size = 1ULL; // 当前Batch循环处理的S1轴的实际大小
    uint32_t actS2Size = 0ULL;
    bool curActSeqLenIsZero = false;
    bool needDealActS1LessThanS1 = false; // S1的实际长度小于shape的S1长度时，是否需要清理输出
    uint32_t actMBaseSize = 0U;           // m轴(gS1)方向实际大小
    uint32_t mBasicSizeTail = 0U;         // gS1方向循环的尾基本块大小
    uint32_t s2BasicSizeTail = 0U;        // S2方向循环的尾基本块大小
};

template <typename T>
class DenseLISoftmaxLse {
public:
    __aicore__ inline DenseLISoftmaxLse(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
                                __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
                                __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, __gm__ uint8_t *workspace,
                                const DenseLISoftmaxLseTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();
    // 类型定义区
    using Q_T = typename T::queryType;
    using K_T = typename T::keyType;
    using W_T = typename T::weightType;
    using OUT_T = typename T::outputType;
    static constexpr LAYOUT LAYOUT_T = T::layout;

    using MM1_OUT_T = float;

    DenseLISoftmaxLseMatmul<T> matmulService;
    DenseLISoftmaxLseVector<T> vectorService;

    // 常量区
    static constexpr uint32_t SYNC_C1_V1_FLAG = 4;
    static constexpr uint32_t SYNC_V1_C1_FLAG = 5;

    static constexpr uint32_t M_BASE_SIZE = 512;
    static constexpr uint32_t S2_BASE_SIZE = 512;
    static constexpr uint32_t HEAD_DIM = 128;
    static constexpr uint32_t K_HEAD_NUM = 1;
    static constexpr uint32_t GM_ALIGN_BYTES = 512;

    static constexpr int64_t LD_PREFETCH_LEN = 2;
    // for workspace double
    static constexpr uint32_t WS_DOBULE = 2;

protected:
    TPipe *pipe = nullptr;

    // offset
    uint64_t queryCoreOffset = 0ULL;
    uint64_t keyCoreOffset = 0ULL;
    uint64_t weightsCoreOffset = 0ULL;
    uint64_t reduceSumCoreOffset = 0ULL;
    uint64_t reduceMaxOutGmOffset = 0ULL;

    // Global Buffer
    GlobalTensor<Q_T> queryGm;
    GlobalTensor<K_T> keyGm;
    GlobalTensor<W_T> weightsGm;
    GlobalTensor<OUT_T> softmaxMaxGm;
    GlobalTensor<OUT_T> softmaxSumGm;
    GlobalTensor<int64_t> actualSeqLengthsGmQ;
    GlobalTensor<int64_t> actualSeqLengthsGm;

    // workspace
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<float> vec1ResGm;

    // aic、aiv核信息
    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;
    uint32_t usedCoreNum = 0U;

    DenseLISoftmaxLseCommon::ConstInfo constInfo{};
    TempLoopInfo tempLoopInfo{};
    DenseLISoftmaxLseCommon::SplitCoreInfo splitCoreInfo{};
    DenseLISoftmaxLseCommon::SplitCoreInfo softmaxSplitCoreInfo{};

    // Init functions
    __aicore__ inline void InitTilingData(const DenseLISoftmaxLseTilingData *__restrict tilingData);
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths);
    __aicore__ inline void InitWorkSpace(__gm__ uint8_t *workspace);
    // Split Core
    __aicore__ inline void SplitCore(uint32_t curCoreIdx, uint32_t &coreNum,
                                     DenseLISoftmaxLseCommon::SplitCoreInfo &info);
    // ================================Process functions================================
    __aicore__ inline void ProcessMain();
    __aicore__ inline void ProcessBaseBlock(DenseLISoftmaxLseCommon::RunInfo &runInfo);
    __aicore__ inline void ProcessSoftmax(DenseLISoftmaxLseCommon::RunInfo &runInfo);
    __aicore__ inline void ResetInvalidOutputZero();
    // ================================Params Calc=====================================
    __aicore__ inline void CalcGS1LoopParams(uint32_t bN2Idx);
    __aicore__ inline void GetBN2Idx(uint32_t bN2Idx);
    __aicore__ inline uint32_t GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                               GlobalTensor<int64_t> &actualSeqLengthsGm, uint32_t defaultSeqLen);
    __aicore__ inline void GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t &actS1Size, uint32_t &actS2Size);
    __aicore__ inline void CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx);
    __aicore__ inline void CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx, DenseLISoftmaxLseCommon::RunInfo &runInfo);
};

template <typename T>
__aicore__ inline void DenseLISoftmaxLse<T>::InitTilingData(const DenseLISoftmaxLseTilingData *__restrict tilingData)
{
    usedCoreNum = tilingData->usedCoreNum;
    constInfo.batchSize = tilingData->bSize;
    constInfo.qHeadNum = constInfo.gSize = tilingData->gSize;
    constInfo.kSeqSize = tilingData->s2Size;
    constInfo.qSeqSize = tilingData->s1Size;
    constInfo.storageS1Size = tilingData->storageS1Size;
    constInfo.storageS2Size = tilingData->storageS2Size;
    constInfo.attenMaskFlag = (tilingData->sparseMode == 3);
    constInfo.preTokens = tilingData->preTokens;
    constInfo.nextTokens = tilingData->nextTokens;
    constInfo.outputLayout = LAYOUT_T; // 输出和输入形状一致
    if (LAYOUT_T == LAYOUT::TND) {
        constInfo.isAccumSeqS1 = true;
        constInfo.isAccumSeqS2 = true;
    }
    constInfo.kHeadNum = K_HEAD_NUM;
    constInfo.headDim = HEAD_DIM;
    constInfo.s2BaseSize = S2_BASE_SIZE;

    constInfo.s1BaseSize = 8;
    constInfo.mBaseSize = constInfo.s1BaseSize * constInfo.gSize;
}

template <typename T>
__aicore__ inline void DenseLISoftmaxLse<T>::InitBuffers()
{
    if ASCEND_IS_AIV {
        vectorService.InitBuffers(pipe);
    } else {
        matmulService.InitBuffers(pipe);
    }
}

template <typename T>
__aicore__ inline void DenseLISoftmaxLse<T>::InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ,
                                                              __gm__ uint8_t *actualSeqLengths)
{
    if (actualSeqLengthsQ == nullptr) {
        constInfo.actualLenQDims = 0;
    } else {
        constInfo.actualLenQDims = constInfo.batchSize;
        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ int64_t *)actualSeqLengthsQ, constInfo.actualLenQDims);
    }
    if (actualSeqLengths == nullptr) {
        constInfo.actualLenDims = 0;
    } else {
        constInfo.actualLenDims = constInfo.batchSize;
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ int64_t *)actualSeqLengths, constInfo.actualLenDims);
    }
}

template <typename T>
__aicore__ inline void DenseLISoftmaxLse<T>::InitWorkSpace(__gm__ uint8_t *workspace)
{
    // workspace 内存排布
    // |mm1ResGm(存S) DB|vec1ResGm(存放reduceSum中间结果) DB
    // |Core0_mm1ResDB0-Core0_mm1ResDB1-Core1_mm1ResDB0....Core23_mm1ResDB0-Core23_mm1ResDB1|
    uint64_t offset = 0;
    uint64_t singleCoreMm1ResSize = WS_DOBULE * constInfo.mBaseSize * constInfo.s2BaseSize * sizeof(MM1_OUT_T);
    mm1ResGm.SetGlobalBuffer((__gm__ MM1_OUT_T *)(workspace + offset + aiCoreIdx * singleCoreMm1ResSize));
    offset += GetBlockNum() * singleCoreMm1ResSize;

    uint64_t reduceSumResSize = constInfo.s1BaseSize *
                                DenseLISoftmaxLseCommon::Align(constInfo.kSeqSize, (uint64_t)constInfo.s2BaseSize) *
                                sizeof(float);
    vec1ResGm.SetGlobalBuffer((__gm__ float *)(workspace + offset + aiCoreIdx * reduceSumResSize));
    offset += GetBlockNum() * reduceSumResSize;
}

template <typename T>
__aicore__ inline uint32_t DenseLISoftmaxLse<T>::GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                                                 GlobalTensor<int64_t> &actualSeqLengthsGm,
                                                                 uint32_t defaultSeqLen)
{
    if (actualLenDims == 0) {
        return defaultSeqLen;
    } else if (isAccumSeq && bIdx > 0) {
        return actualSeqLengthsGm.GetValue(bIdx) - actualSeqLengthsGm.GetValue(bIdx - 1);
    } else {
        return actualSeqLengthsGm.GetValue(bIdx);
    }
}

template <typename T>
__aicore__ inline void DenseLISoftmaxLse<T>::GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t &actS1Size,
                                                                 uint32_t &actS2Size)
{
    actS1Size = GetActualSeqLen(bIdx, constInfo.actualLenQDims, constInfo.isAccumSeqS1, actualSeqLengthsGmQ,
                                constInfo.qSeqSize);
    actS2Size =
        GetActualSeqLen(bIdx, constInfo.actualLenDims, constInfo.isAccumSeqS2, actualSeqLengthsGm, constInfo.kSeqSize);
}

template <typename T>
__aicore__ void inline DenseLISoftmaxLse<T>::SplitCore(uint32_t curCoreIdx, uint32_t &coreNum,
                                                       DenseLISoftmaxLseCommon::SplitCoreInfo &info)
{
    uint32_t totalBlockNum = 0;
    uint32_t s1GBaseNum = 0;
    for (uint32_t bIdx = 0; bIdx < constInfo.batchSize; bIdx++) {
        uint32_t actS1Size = GetActualSeqLen(bIdx, constInfo.actualLenQDims, constInfo.isAccumSeqS1,
                                             actualSeqLengthsGmQ, constInfo.qSeqSize);
        s1GBaseNum = CeilDiv(actS1Size, constInfo.s1BaseSize);
        totalBlockNum += s1GBaseNum;
    }

    uint32_t minSeqS1BlockPerCore = totalBlockNum / coreNum;
    uint32_t deal1MoreS1BlockCoreNum = totalBlockNum % coreNum;
    uint32_t coreDealS1BlockCnt =
        curCoreIdx < deal1MoreS1BlockCoreNum ? minSeqS1BlockPerCore + 1 : minSeqS1BlockPerCore;

    info.dealCnt = coreDealS1BlockCnt;
    if (coreDealS1BlockCnt <= 0) {
        return;
    }

    uint32_t coreDealS1BlockOffset = curCoreIdx < deal1MoreS1BlockCoreNum ?
                                         (minSeqS1BlockPerCore + 1) * curCoreIdx :
                                         (minSeqS1BlockPerCore + 1) * deal1MoreS1BlockCoreNum +
                                             minSeqS1BlockPerCore * (curCoreIdx - deal1MoreS1BlockCoreNum);

    coreDealS1BlockOffset = deal1MoreS1BlockCoreNum == 0 ? (curCoreIdx * coreDealS1BlockCnt) : coreDealS1BlockOffset;

    uint32_t tmpAccumSeqS1Block = 0;
    bool isFindStart = false;
    bool isFindEnd = false;
    uint32_t startBIdx = 0;
    uint32_t endBIdx = 0;
    uint32_t s1StartBlock = 0;
    uint32_t s1EndBlock = 0;

    for (uint32_t bIdx = 1; bIdx <= constInfo.batchSize; bIdx++) {
        uint32_t actS1Size = GetActualSeqLen(bIdx - 1, constInfo.actualLenQDims, constInfo.isAccumSeqS1,
                                             actualSeqLengthsGmQ, constInfo.qSeqSize);
        uint32_t s1GBaseNum = CeilDiv(actS1Size, constInfo.s1BaseSize);
        tmpAccumSeqS1Block += s1GBaseNum;
        if (!isFindStart && tmpAccumSeqS1Block > coreDealS1BlockOffset) {
            startBIdx = bIdx - 1;
            s1StartBlock = coreDealS1BlockOffset - tmpAccumSeqS1Block + s1GBaseNum;
            isFindStart = true;
        }

        if (!isFindEnd && tmpAccumSeqS1Block >= (coreDealS1BlockOffset + coreDealS1BlockCnt)) {
            endBIdx = bIdx - 1;
            s1EndBlock = coreDealS1BlockOffset + coreDealS1BlockCnt - tmpAccumSeqS1Block + s1GBaseNum - 1;
            isFindEnd = true;
        }
    }

    info.bN2Start = startBIdx;
    info.bN2End = endBIdx;

    info.gS1Start = s1StartBlock;
    info.gS1End = s1EndBlock;
}

template <typename T>
__aicore__ inline void DenseLISoftmaxLse<T>::Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
                                                  __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
                                                  __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
                                                  __gm__ uint8_t *workspace,
                                                  const DenseLISoftmaxLseTilingData *__restrict tiling, TPipe *tPipe)
{
    if ASCEND_IS_AIV {
        tmpBlockIdx = GetBlockIdx(); // vec:0-47
        aiCoreIdx = tmpBlockIdx / 2;
    } else {
        tmpBlockIdx = GetBlockIdx(); // cube:0-23
        aiCoreIdx = tmpBlockIdx;
    }

    InitTilingData(tiling);
    InitActualSeqLen(actualSeqLengthsQ, actualSeqLengths);

    SplitCore(aiCoreIdx, usedCoreNum, splitCoreInfo);
    pipe = tPipe;
    InitWorkSpace(workspace);

    if ASCEND_IS_AIV {
        vectorService.InitParams(constInfo, tiling);
        softmaxMaxGm.SetGlobalBuffer((__gm__ OUT_T *)softmaxMax, constInfo.isAccumSeqS1 ?
                                                                     constInfo.storageS1Size :
                                                                     constInfo.batchSize * constInfo.qSeqSize);
        softmaxSumGm.SetGlobalBuffer((__gm__ OUT_T *)softmaxSum, constInfo.isAccumSeqS1 ?
                                                                     constInfo.storageS1Size :
                                                                     constInfo.batchSize * constInfo.qSeqSize);
        weightsGm.SetGlobalBuffer((__gm__ W_T *)weights,
                                  constInfo.isAccumSeqS1 ?
                                      constInfo.storageS1Size * constInfo.qHeadNum :
                                      constInfo.batchSize * constInfo.qSeqSize * constInfo.qHeadNum);
        vectorService.InitVec1GlobalTensor(mm1ResGm, vec1ResGm, weightsGm, softmaxMaxGm, softmaxSumGm);
    } else {
        matmulService.InitParams(constInfo);
        queryGm.SetGlobalBuffer((__gm__ Q_T *)query,
                                constInfo.isAccumSeqS1 ?
                                    constInfo.storageS1Size * constInfo.qHeadNum * constInfo.headDim :
                                    constInfo.batchSize * constInfo.qSeqSize * constInfo.qHeadNum * constInfo.headDim);
        keyGm.SetGlobalBuffer((__gm__ K_T *)key,
                              constInfo.isAccumSeqS2 ?
                                  constInfo.storageS2Size * constInfo.kHeadNum * constInfo.headDim :
                                  constInfo.batchSize * constInfo.kSeqSize * constInfo.kHeadNum * constInfo.headDim);
        matmulService.InitMm1GlobalTensor(keyGm, queryGm, mm1ResGm);
    }
    InitBuffers();
}

template <typename T>
__aicore__ inline void DenseLISoftmaxLse<T>::GetBN2Idx(uint32_t bN2Idx)
{
    tempLoopInfo.bN2Idx = bN2Idx;
    tempLoopInfo.bIdx = bN2Idx / constInfo.kHeadNum;
    tempLoopInfo.n2Idx = bN2Idx % constInfo.kHeadNum;
}

template <typename T>
__aicore__ inline void DenseLISoftmaxLse<T>::CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx)
{
    tempLoopInfo.gS1Idx = gS1LoopIdx;
    tempLoopInfo.actMBaseSize = constInfo.mBaseSize;
    uint32_t remainedGS1Size = tempLoopInfo.actS1Size * constInfo.gSize - tempLoopInfo.gS1Idx * constInfo.mBaseSize;
    if (remainedGS1Size <= constInfo.mBaseSize && remainedGS1Size > 0) {
        tempLoopInfo.actMBaseSize = tempLoopInfo.mBasicSizeTail;
    }

    uint32_t s2BlockNum = (tempLoopInfo.actS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    tempLoopInfo.s2LoopEnd = s2BlockNum - 1;
}

template <typename T>
__aicore__ inline void DenseLISoftmaxLse<T>::CalcGS1LoopParams(uint32_t bN2LoopIdx)
{
    GetBN2Idx(bN2LoopIdx);
    GetS1S2ActualSeqLen(tempLoopInfo.bIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2Size);
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
}

template <typename T>
__aicore__ inline void DenseLISoftmaxLse<T>::CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx,
                                                         DenseLISoftmaxLseCommon::RunInfo &runInfo)
{
    runInfo.loop = loop;
    runInfo.bIdx = tempLoopInfo.bIdx;
    runInfo.gS1Idx = tempLoopInfo.gS1Idx;
    runInfo.s2Idx = s2LoopIdx;
    runInfo.bN2Idx = tempLoopInfo.bN2Idx;

    runInfo.actS1Size = tempLoopInfo.actS1Size;
    runInfo.actS2Size = tempLoopInfo.actS2Size;
    // 计算实际基本块size
    runInfo.actMBaseSize = tempLoopInfo.actMBaseSize;
    runInfo.actualSingleProcessSInnerSize = constInfo.s2BaseSize;
    uint32_t s2SplitNum = (tempLoopInfo.actS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    if (runInfo.s2Idx == s2SplitNum - 1) {
        runInfo.actualSingleProcessSInnerSize = tempLoopInfo.s2BasicSizeTail;
    }
    runInfo.actualSingleProcessSInnerSizeAlign = DenseLISoftmaxLseCommon::Align(
        (uint32_t)runInfo.actualSingleProcessSInnerSize, DenseLISoftmaxLseCommon::ConstInfo::BUFFER_SIZE_BYTE_32B);

    runInfo.isFirstS2InnerLoop = s2LoopIdx == splitCoreInfo.s2Start;
    runInfo.isLastS2InnerLoop = s2LoopIdx == tempLoopInfo.s2LoopEnd;
    runInfo.isAllLoopEnd = (runInfo.bN2Idx == splitCoreInfo.bN2End) && (runInfo.gS1Idx == splitCoreInfo.gS1End) &&
                           (runInfo.s2Idx == splitCoreInfo.s2End);

    if (runInfo.isFirstS2InnerLoop) {
        uint64_t actualSeqQPrefixSum;
        uint64_t actualSeqKPrefixSum;
        if constexpr (LAYOUT_T == LAYOUT::TND) {
            actualSeqQPrefixSum = (runInfo.bIdx <= 0) ? 0 : actualSeqLengthsGmQ.GetValue(runInfo.bIdx - 1);
            actualSeqKPrefixSum = (runInfo.bIdx <= 0) ? 0 : actualSeqLengthsGm.GetValue(runInfo.bIdx - 1);
        } else { // BSND
            actualSeqQPrefixSum = (runInfo.bIdx <= 0) ? 0 : runInfo.bIdx * constInfo.qSeqSize;
            actualSeqKPrefixSum = (runInfo.bIdx <= 0) ? 0 : runInfo.bIdx * constInfo.kSeqSize;
        }
        uint64_t tndBIdxOffset = actualSeqQPrefixSum * constInfo.qHeadNum * constInfo.headDim;
        uint64_t tndKeyBIdxOffset = actualSeqKPrefixSum * constInfo.kHeadNum * constInfo.headDim;
        // B,S1,N1(N2,G),D
        queryCoreOffset = tndBIdxOffset + runInfo.gS1Idx * constInfo.mBaseSize * constInfo.headDim;
        keyCoreOffset = tndKeyBIdxOffset + runInfo.n2Idx * constInfo.headDim;
        // B,S1,N1(N2,G)/T,N1(N2,G)
        weightsCoreOffset = actualSeqQPrefixSum * constInfo.qHeadNum + runInfo.n2Idx * constInfo.gSize;
        reduceMaxOutGmOffset = actualSeqQPrefixSum;
    }
    runInfo.tensorQueryOffset = queryCoreOffset;
    runInfo.tensorKeyOffset =
        keyCoreOffset + runInfo.s2Idx * constInfo.s2BaseSize * constInfo.kHeadNum * constInfo.headDim;
    runInfo.tensorWeightsOffset = weightsCoreOffset;
    runInfo.reduceSumGmOutOffset = runInfo.s2Idx * constInfo.s2BaseSize;

    runInfo.reduceMaxOutGmOffset = reduceMaxOutGmOffset;
}

template <typename T>
__aicore__ inline void DenseLISoftmaxLse<T>::Process()
{
    if (usedCoreNum == 0) {
        return;
    }
    ProcessMain();
    SyncAll();
    ResetInvalidOutputZero();
}

template <typename T>
__aicore__ inline void DenseLISoftmaxLse<T>::ProcessMain()
{
    if (aiCoreIdx >= usedCoreNum) {
        return;
    }

    if (splitCoreInfo.dealCnt <= 0) {
        return;
    }

    if ASCEND_IS_AIV {
        vectorService.AllocEventID();
        CrossCoreSetFlag<DenseLISoftmaxLseCommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV1C1);
    } else {
        matmulService.AllocEventID();
    }

    DenseLISoftmaxLseCommon::RunInfo runInfo;
    uint32_t gloop = 0;

    for (uint32_t bN2LoopIdx = splitCoreInfo.bN2Start; bN2LoopIdx <= splitCoreInfo.bN2End; bN2LoopIdx++) {
        CalcGS1LoopParams(bN2LoopIdx);
        if (tempLoopInfo.curActSeqLenIsZero) {
            splitCoreInfo.gS1Start = 0;
            continue;
        }
        for (uint32_t gS1LoopIdx = splitCoreInfo.gS1Start; gS1LoopIdx <= tempLoopInfo.gS1LoopEnd; gS1LoopIdx++) {
            CalcS2LoopParams(bN2LoopIdx, gS1LoopIdx);
            for (int s2LoopIdx = splitCoreInfo.s2Start; s2LoopIdx <= tempLoopInfo.s2LoopEnd; s2LoopIdx++) {
                CalcRunInfo(gloop, s2LoopIdx, runInfo);
                ProcessBaseBlock(runInfo);
                ++gloop;
            }
            if ASCEND_IS_AIV {
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
                ProcessSoftmax(runInfo);
                CrossCoreSetFlag<DenseLISoftmaxLseCommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV1C1);
            }
            splitCoreInfo.s2Start = 0;
        }
        splitCoreInfo.gS1Start = 0;
    }

    if ASCEND_IS_AIV {
        vectorService.FreeEventID();
    } else {
        matmulService.FreeEventID();
        CrossCoreWaitFlag(constInfo.syncV1C1);
    }
}

template <typename T>
__aicore__ inline void DenseLISoftmaxLse<T>::ProcessBaseBlock(DenseLISoftmaxLseCommon::RunInfo &runInfo)
{
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag(constInfo.syncV1C1);
        matmulService.ComputeMm1(runInfo);
        CrossCoreSetFlag<DenseLISoftmaxLseCommon::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V1);
    } else {
        CrossCoreWaitFlag(constInfo.syncC1V1);
        vectorService.ProcessVec(runInfo);
        if (!runInfo.isLastS2InnerLoop) {
            CrossCoreSetFlag<DenseLISoftmaxLseCommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV1C1);
        }
    }
}

template <typename T>
__aicore__ inline void DenseLISoftmaxLse<T>::ProcessSoftmax(DenseLISoftmaxLseCommon::RunInfo &runInfo)
{
    uint32_t startS1Idx = runInfo.gS1Idx * constInfo.s1BaseSize;
    uint32_t endS1Idx = runInfo.actS1Size < (runInfo.gS1Idx + 1) * constInfo.s1BaseSize ?
                            runInfo.actS1Size - 1 :
                            (runInfo.gS1Idx + 1) * constInfo.s1BaseSize - 1;
    uint32_t s1InnerLoop = 0;
    for (uint32_t s1Idx = startS1Idx; s1Idx <= endS1Idx; s1Idx++) {
        runInfo.s1BlockInnerLoop = s1InnerLoop;
        runInfo.s1Idx = s1Idx;
        vectorService.ProcessSoftmax(runInfo);
        s1InnerLoop += 1;
    }
}

template <typename T>
__aicore__ inline void DenseLISoftmaxLse<T>::ResetInvalidOutputZero()
{
    if (aiCoreIdx >= usedCoreNum) {
        return;
    }

    if ASCEND_IS_AIV {
        if (constInfo.isAccumSeqS1 && constInfo.isAccumSeqS2) {
            uint64_t lastAccumSeqS1 = actualSeqLengthsGmQ.GetValue(constInfo.batchSize - 1);
            uint64_t totalDealSize = constInfo.storageS1Size - lastAccumSeqS1;
            if (totalDealSize <= 0) {
                return;
            }

            uint64_t coreDealSize = totalDealSize / usedCoreNum;
            uint64_t outOffset = lastAccumSeqS1;
            if (coreDealSize == 0 && aiCoreIdx == 0) {
                coreDealSize = totalDealSize;
            }

            uint64_t minDealSize = totalDealSize / usedCoreNum;
            uint32_t moreOneCoreNum = totalDealSize % usedCoreNum;
            if (minDealSize > 0) {
                if (moreOneCoreNum > 0) {
                    coreDealSize = aiCoreIdx < moreOneCoreNum ? minDealSize + 1 : minDealSize;
                    outOffset += aiCoreIdx < moreOneCoreNum ?
                                     aiCoreIdx * (minDealSize + 1) :
                                     moreOneCoreNum * (minDealSize + 1) + (aiCoreIdx - moreOneCoreNum) * minDealSize;
                } else {
                    coreDealSize = minDealSize;
                    outOffset += aiCoreIdx * minDealSize;
                }
            }
            if (coreDealSize <= 0) {
                return;
            }
            vectorService.ProcessOutput(outOffset, coreDealSize);
        }
    }
}
} // namespace DenseLISoftmaxLseKernel
#endif // DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_H
