/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fia_kernel_nonquant_mla.h
 * \brief
 */

#ifndef FIA_KERNEL_NONQUANT_MLA_H
#define FIA_KERNEL_NONQUANT_MLA_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../fia_public_define.h"
#include "../vector_common.h"
#include "fia_block_vec_nonquant_mla.h"
#include "fia_block_cube_nonquant_mla.h"
#include "fia_block_vec_flashdecode.h"

using namespace optiling;
using namespace matmul;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename FIAT>
class FiaKernelNonQuantMla {
public:
    __aicore__ inline FiaKernelNonQuantMla(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *pseShift, __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengthsQ,
                                __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *blockTable,
                                __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
                                __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
                                const FusedInferAttentionScoreTilingData *__restrict tiling, __gm__ uint8_t *gmTiling,
                                TPipe *tPipe, bool isPrefix = false);
    __aicore__ inline void InitQuant(__gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2,
                                     __gm__ uint8_t *quantScale2, __gm__ uint8_t *quantOffset2,
                                     __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
                                     __gm__ uint8_t *keyAntiquantScale, __gm__ uint8_t *keyAntiquantOffset,
                                     __gm__ uint8_t *valueAntiquantScale, __gm__ uint8_t *valueAntiquantOffset,
                                     __gm__ uint8_t *keyRopeAntiquantScale, __gm__ uint8_t *workspace);

    __aicore__ inline void Process();

    // =================================类型定义区=================================
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using Q_T = typename FIAT::queryType;
    using KV_T = typename FIAT::kvType;
    using OUT_T = typename FIAT::outputType;
    using ORIGIN_T = typename FIAT::orginalType;
    static constexpr bool PAGE_ATTENTION = FIAT::pageAttention;
    static constexpr bool FLASH_DECODE = FIAT::flashDecode;
    static constexpr FIA_LAYOUT LAYOUT_T = FIAT::layout;
    static constexpr FIA_LAYOUT KV_LAYOUT_T = FIAT::kvLayout;

    static constexpr bool QUANT = (IsSameType<Q_T, KV_T>::value && IsSameType<KV_T, int8_t>::value);
    static constexpr uint8_t PER_CHANNEL_MODE = 0; // 伪量化: K V per-channel
    static constexpr uint8_t ANTIQUANT_MODE = FIAT::antiquantMode;
    static constexpr bool ANTIQUANT = !IsSameType<Q_T, KV_T>::value;
    static constexpr bool ANTIQUANT_PER_CHANNEL = (ANTIQUANT && (ANTIQUANT_MODE == PER_CHANNEL_MODE));
    using ANTIQ_PARAMS_T = Q_T;
    using Q_ROPE_T = typename AscendC::Conditional<ANTIQUANT, Q_T, ORIGIN_T>::type;
    using K_ROPE_T = typename AscendC::Conditional<ANTIQUANT, KV_T, ORIGIN_T>::type;
    using UPDATE_T = typename AscendC::Conditional<QUANT || ANTIQUANT, half, T>::type;
    using TMP_T = typename AscendC::Conditional<ANTIQUANT, half, T>::type;
    using MM1_OUT_T = typename AscendC::Conditional<QUANT, int32_t, TMP_T>::type;
    using MM2_OUT_T = typename AscendC::Conditional<QUANT, half, TMP_T>::type;

    FiaBlockCubeNonQuantMla<FIAT> matmulService;
    FiaBlockVecNonQuantMla<FIAT> vectorService;
    FiaBlockVecFlashDecode<FIAT> fdService;

    // =================================常量区=================================
    static constexpr uint32_t PRELOAD_NUM = 2;
    static constexpr uint32_t N_BUFFER_M_BASIC_SIZE = 256;
    static constexpr uint32_t FIA_PRELOAD_TASK_CACHE_SIZE = 3;

    static constexpr uint32_t SYNC_V0_C1_FLAG = 6;
    static constexpr uint32_t SYNC_C1_V1_FLAG = 7;
    static constexpr uint32_t SYNC_V1_C2_FLAG = 8;
    static constexpr uint32_t SYNC_C2_V2_FLAG = 9;
    static constexpr uint32_t SYNC_C2_V1_FLAG = 4;
    static constexpr uint32_t SYNC_V1_NUPDATE_C2_FLAG = 5;

    static constexpr uint64_t kvHeadNum = 1ULL;
    static constexpr uint64_t headDim = 512ULL;
    static constexpr uint64_t headDimAlign = 512ULL;
    static constexpr uint64_t headDimRope = 64ULL;
    static constexpr bool batchContinuous = true;
    static constexpr uint32_t msdIterNum = 2U;
    static constexpr int64_t fdPrefetchLen = 2;

protected:
    // 由于S2循环前，RunInfo还没有赋值，使用Bngs1Param临时存放B、N、S1轴相关的信息；同时减少重复计算
    struct TempLoopInfo {
        uint32_t bn2IdxInCurCore = 0;
        uint32_t bIdx = 0U;
        uint32_t n2Idx = 0U;
        uint64_t s2BasicSizeTail = 0U; // S2方向循环的尾基本块大小
        uint32_t s2LoopTimes = 0U; // S2方向循环的总次数，无论TND还是BXXD都是等于实际次数，不用减1
        uint64_t curActualSeqLen = 0ULL;
        bool curActSeqLenIsZero = false;
        uint64_t actS1Size = 1ULL;
        uint32_t tndCoreStartKVSplitPos = 0U;
        bool tndIsS2SplitCore = false;

        uint32_t gS1Idx = 0U;
        uint64_t mBasicSizeTail = 0U; // gS1方向循环的尾基本块大小
        int32_t preTokensPerBatch = 0;
        int32_t nextTokensPerBatch = 0;
    };

    const FusedInferAttentionScoreTilingData *__restrict tilingData = nullptr;
    TPipe *pipe = nullptr;

    // for workspace pingpong
    const uint32_t dbWorkspaceRatio = PRELOAD_NUM;

    // offset
    uint64_t tensorACoreOffset = 0ULL;
    uint64_t tensorBCoreOffset = 0ULL;
    uint64_t tensorARopeCoreOffset = 0ULL;
    uint64_t tensorBRopeCoreOffset = 0ULL;
    uint64_t tensorBOffset = 0ULL;
    uint64_t attenOutOffset = 0ULL;
    uint64_t attenMaskOffset = 0ULL;
    uint64_t attenMaskCoreOffset = 0ULL;

    // ================================Global Buffer区=================================
    GlobalTensor<Q_T> queryGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<KV_T> valueGm;
    GlobalTensor<Q_ROPE_T> qRopeGm;
    GlobalTensor<K_ROPE_T> kRopeGm;

    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<int32_t> blockTableGm;

    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;

    __gm__ uint8_t *key_ = nullptr;
    __gm__ uint8_t *value_ = nullptr;
    // atten mask
    GlobalTensor<bool> attenMaskBoolGm;
    GlobalTensor<uint64_t> actualSeqLengthsGmQ;
    GlobalTensor<uint64_t> actualSeqLengthsGm;
    // workspace
    GlobalTensor<KV_T> queryPreProcessResGm; // 存放Q1, Q2
    GlobalTensor<MM1_OUT_T> mm1ResGm;        // 存放S
    GlobalTensor<KV_T> vec1ResGm;            // 存放A1, A2
    GlobalTensor<MM2_OUT_T> mm2ResGm;        // 存放O

    GlobalTensor<int32_t> mm2ResInt32Gm;
    GlobalTensor<UPDATE_T> vec2ResGm;

    GlobalTensor<T> accumOutGm;
    GlobalTensor<T> lseSumFdGm;
    GlobalTensor<T> lseMaxFdGm;
    // ================================类成员变量====================================
    // aic、aiv核信息
    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;
    uint32_t usedCoreNum = 0U;

    AttentionCommon::ConstInfo constInfo{};
    TempLoopInfo tempLoopInfo{};
    // ================================Util functions==================================
    template <typename T> __aicore__ inline T Align(T num, T rnd)
    {
        return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
    }

    template <typename T1, typename T2> __aicore__ inline T1 Min(T1 a, T2 b)
    {
        return (a > b) ? (b) : (a);
    }
    // ================================Init functions==================================
    __aicore__ inline void InitTilingData();
    __aicore__ inline void InitCalcParamsEach();
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths);    
    __aicore__ inline void InitValueGm(uint32_t bIdx);
    __aicore__ inline void InitKeyGm(uint32_t bIdx);
    __aicore__ inline void InitOutputSingleCore();
    // ================================Process functions================================
    __aicore__ inline void ProcessBalance();
    __aicore__ inline void PreloadPipeline(uint32_t loop, 
                                           AttentionCommon::RunInfo extraInfo[FIA_PRELOAD_TASK_CACHE_SIZE]);
    // ================================Offset Calc=====================================
    __aicore__ inline void GetActualSeqLen(uint32_t bIdx, uint32_t s1Idx = 0);
    __aicore__ inline void UpdateInnerLoopCond();
    __aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx);
    __aicore__ inline void CalcParams(uint32_t loop, uint64_t s2Start, uint32_t s2LoopIdx, AttentionCommon::RunInfo &info);
    __aicore__ inline void GetAxisEndIdx(uint32_t bN2End, uint32_t s1GEnd, uint32_t s2End);
    __aicore__ inline uint64_t GetBalanceActualSeqLengths(GlobalTensor<uint64_t> &actualSeqLengths, uint32_t bIdx);
    __aicore__ inline uint32_t GetActualSeqLenKV(uint32_t bIdx);
    __aicore__ inline uint64_t GetTNDBatchOffset(int bIdx);
    __aicore__ inline void GetBN2Idx(uint32_t bN2Idx, uint32_t &bIdx, uint32_t &n2Idx);
    __aicore__ inline void UpdateInner(uint32_t &s2Start, uint32_t &s2End, uint32_t &curS2Start, uint32_t &curS2End,
                                       uint32_t s1Idx, bool isStart, bool isEnd);
    __aicore__ inline void UpdateInnerNum(uint32_t &s2End, uint32_t actS1Size, uint32_t actS2Size, uint32_t s1Idx);
    __aicore__ inline void GetPreNextTokensLeftUp();
    __aicore__ inline int64_t ClipSInnerToken(int64_t sInnerToken, int64_t minValue, int64_t maxValue);
    // ================================Mm1==============================================
    __aicore__ inline void ComputeMm1(const AttentionCommon::RunInfo &info);
    // ================================Mm2==============================================
    __aicore__ inline void ComputeMm2(const AttentionCommon::RunInfo &info);

    __aicore__ inline void InitAllZeroOutput(uint32_t bIdx, uint32_t n2Idx);
    __aicore__ inline uint64_t SeqLenFromTensorList(uint32_t bIdx);
    __aicore__ inline void FlashDecode();
};

template <typename FIAT> __aicore__ inline void FiaKernelNonQuantMla<FIAT>::InitTilingData()
{
    usedCoreNum = tilingData->baseParams.usedCoreNum;
    constInfo.mmResUbSize = tilingData->workspaceParams.mm1ResSize;
    constInfo.bmm2ResUbSize = tilingData->workspaceParams.mm2ResSize;
    constInfo.vec1ResUbSize = constInfo.mmResUbSize * msdIterNum;

    constInfo.batchSize = tilingData->baseParams.bSize;
    constInfo.qHeadNum = constInfo.gSize = tilingData->baseParams.gSize;
    constInfo.kvSeqSize = tilingData->baseParams.s2Size;
    constInfo.qSeqSize = tilingData->baseParams.s1Size;
    constInfo.attenMaskFlag = (tilingData->maskParams.attenMaskFlag != 0) ? true : false;
    constInfo.attenMaskSize = tilingData->maskParams.attenMaskSize;
    constInfo.maxBlockNumPerBatch = tilingData->pageAttenParams.maxBlockNumPerBatch;
    constInfo.kvCacheBlockSize = tilingData->pageAttenParams.blockSize;
    constInfo.outputLayout = static_cast<FIA_LAYOUT>(tilingData->baseParams.outputLayout);
    constInfo.mBaseSize = tilingData->innerSplitParams.mBaseSize;
    constInfo.s2BaseSize = tilingData->innerSplitParams.s2BaseSize;
    constInfo.needInit = tilingData->baseParams.needInit;
    constInfo.preToken = tilingData->maskParams.preToken;
    constInfo.nextToken = tilingData->maskParams.nextToken;
    constInfo.slidingFlag = tilingData->baseParams.slidingFlag;

    constInfo.kvHeadNum = kvHeadNum;
    constInfo.headDim = headDim;
    constInfo.headDimRope = headDimRope;
    constInfo.headDimAlign = headDimAlign;

    constInfo.preLoadNum = PRELOAD_NUM;
    constInfo.nBufferMBaseSize = N_BUFFER_M_BASIC_SIZE;
    constInfo.syncV0C1 = SYNC_V0_C1_FLAG;
    constInfo.syncC1V1 = SYNC_C1_V1_FLAG;
    constInfo.syncV1C2 = SYNC_V1_C2_FLAG;
    constInfo.syncC2V2 = SYNC_C2_V2_FLAG;
    constInfo.syncC2V1 = SYNC_C2_V1_FLAG;
    constInfo.syncV1NupdateC2 = SYNC_V1_NUPDATE_C2_FLAG;
}

template <typename FIAT> __aicore__ inline void FiaKernelNonQuantMla<FIAT>::InitBuffers()
{
    if ASCEND_IS_AIV {
        vectorService.InitBuffers(pipe);
    } else {
        matmulService.InitBuffers(pipe);
    }
}

template <typename FIAT>
__aicore__ inline void
FiaKernelNonQuantMla<FIAT>::InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ,
                                                                __gm__ uint8_t *actualSeqLengths)
{
    constInfo.actualLenQDims = tilingData->baseParams.actualSeqS1Dims;
    constInfo.actualLenDims = tilingData->baseParams.actualSeqS2Dims;
    if (constInfo.actualLenDims != 0) {
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint64_t *)actualSeqLengths, constInfo.actualLenDims);
    }
    if (constInfo.actualLenQDims != 0) {
        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint64_t *)actualSeqLengthsQ, constInfo.actualLenQDims);
    }
}

template <typename FIAT>
__aicore__ inline void FiaKernelNonQuantMla<FIAT>::InitAllZeroOutput(uint32_t bIdx, uint32_t n2Idx)
{
    if (constInfo.outputLayout == FIA_LAYOUT::TND) {
        uint32_t tSize = actualSeqLengthsGmQ.GetValue(constInfo.batchSize - 1);
        uint32_t tBase = bIdx == 0 ? 0 : actualSeqLengthsGmQ.GetValue(bIdx - 1);
        uint32_t s1Count = tempLoopInfo.actS1Size;

        for (int s1Idx = 0; s1Idx < s1Count; s1Idx++) {
            uint64_t attenOutOffset = (tBase + s1Idx) * kvHeadNum * constInfo.gSize * headDim + // T轴、s1轴偏移
                                      n2Idx * constInfo.gSize * headDim;                        // N2轴偏移
            matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], constInfo.gSize * headDim, 0);
        }
    } else if (constInfo.outputLayout == FIA_LAYOUT::NTD) {
        uint32_t tSize = actualSeqLengthsGmQ.GetValue(constInfo.batchSize - 1);
        uint32_t tBase = bIdx == 0 ? 0 : actualSeqLengthsGmQ.GetValue(bIdx - 1);
        uint32_t s1Count = tempLoopInfo.actS1Size;

        for (int gIdx = 0; gIdx < constInfo.gSize; gIdx++) {
            uint64_t attenOutOffset = n2Idx * constInfo.gSize * tSize * headDim +
                                      gIdx * tSize * headDim + // N2轴偏移，G轴偏移
                                      tBase * headDim;
            matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], s1Count * headDim, 0);
        }
    } else if (constInfo.outputLayout == FIA_LAYOUT::BNSD) {
        uint64_t attenOutOffset = bIdx * kvHeadNum * constInfo.gSize * constInfo.qSeqSize * headDim + // B轴偏移
                                  n2Idx * constInfo.gSize * constInfo.qSeqSize * headDim;             // N2轴偏移
        matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], constInfo.gSize * constInfo.qSeqSize * headDim, 0);
    } else if (constInfo.outputLayout == FIA_LAYOUT::BSND || constInfo.outputLayout == FIA_LAYOUT::BSH) {
        for (int s1Idx = 0; s1Idx < constInfo.qSeqSize; s1Idx++) {
            uint64_t attenOutOffset = bIdx * constInfo.qSeqSize * kvHeadNum * constInfo.gSize * headDim +
                                      s1Idx * kvHeadNum * constInfo.gSize * headDim + // B轴、S1轴偏移
                                      n2Idx * constInfo.gSize * headDim;              // N2轴偏移
            matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], constInfo.gSize * headDim, 0);
        }
    } else if (constInfo.outputLayout == FIA_LAYOUT::NBSD) {
        for (int gIdx = 0; gIdx < constInfo.gSize; gIdx++) {
            uint64_t attenOutOffset =
                n2Idx * constInfo.gSize * constInfo.batchSize * constInfo.qSeqSize * headDim + // N2轴偏移
                gIdx * constInfo.batchSize * constInfo.qSeqSize * headDim +
                bIdx * constInfo.qSeqSize * headDim; // G轴、B轴偏移
            matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], constInfo.qSeqSize * headDim, 0);
        }
    }
}

template <typename FIAT>
__aicore__ inline void FiaKernelNonQuantMla<FIAT>::InitOutputSingleCore()
{
    if (usedCoreNum != 0) {
        uint64_t totalOutputSize = constInfo.batchSize * constInfo.qHeadNum * constInfo.qSeqSize * constInfo.headDim;
        uint64_t singleCoreSize = (totalOutputSize + (2 * usedCoreNum) - 1) / (2 * usedCoreNum); // 2 means c:v = 1:2
        uint64_t tailSize = totalOutputSize - tmpBlockIdx * singleCoreSize;
        uint64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;
        matmul::InitOutput<OUT_T>(attentionOutGm[tmpBlockIdx * singleCoreSize], singleInitOutputSize, 0);
        SyncAll();
    }
}

template <typename FIAT>
__aicore__ inline int64_t FiaKernelNonQuantMla<FIAT>::ClipSInnerToken(int64_t sInnerToken,
                                                                    int64_t minValue, int64_t maxValue)
{
    sInnerToken = sInnerToken > minValue ? sInnerToken : minValue;
    sInnerToken = sInnerToken < maxValue ? sInnerToken : maxValue;
    return sInnerToken;
}

template <typename FIAT>
__aicore__ inline void FiaKernelNonQuantMla<FIAT>::GetActualSeqLen(uint32_t bIdx, uint32_t s1Idx)
{
    tempLoopInfo.curActualSeqLen = GetActualSeqLenKV(bIdx);
    tempLoopInfo.actS1Size = GetBalanceActualSeqLengths(actualSeqLengthsGmQ, bIdx);
}

template <typename FIAT>
__aicore__ inline uint32_t FiaKernelNonQuantMla<FIAT>::GetActualSeqLenKV(uint32_t bIdx)
{
    if (constInfo.actualLenDims == 0) {
        if (!batchContinuous) {
            return SeqLenFromTensorList(bIdx);
        }
        return constInfo.kvSeqSize;
    } else if (constInfo.actualLenDims == 1) {
        return actualSeqLengthsGm.GetValue(0);
    } else {
        return actualSeqLengthsGm.GetValue(bIdx);
    }
}

template <typename FIAT>
__aicore__ inline void FiaKernelNonQuantMla<FIAT>::GetPreNextTokensLeftUp()
{
    if (!constInfo.attenMaskFlag) {
        return;
    }
    if (constInfo.slidingFlag) {
        tempLoopInfo.preTokensPerBatch =
            static_cast<int32_t>(tempLoopInfo.actS1Size) - static_cast<int32_t>(tempLoopInfo.curActualSeqLen) + constInfo.preToken;
        tempLoopInfo.nextTokensPerBatch =
        static_cast<int32_t>(tempLoopInfo.curActualSeqLen) - static_cast<int32_t>(tempLoopInfo.actS1Size) + constInfo.nextToken;
    } else {
        tempLoopInfo.nextTokensPerBatch = static_cast<int32_t>(tempLoopInfo.curActualSeqLen) - static_cast<int32_t>(tempLoopInfo.actS1Size);
    }
}

template <typename FIAT>
__aicore__ inline void FiaKernelNonQuantMla<FIAT>::UpdateInner(uint32_t &s2Start, uint32_t &s2End,
                                                                                  uint32_t &curS2Start, uint32_t &curS2End,
                                                                                  uint32_t s1Idx, bool isStart, bool isEnd)
{
    if (!constInfo.slidingFlag) {
        tempLoopInfo.s2LoopTimes = isEnd ? constInfo.s2End + 1 : curS2End;
        return;
    }
    uint32_t s1BaseSize = 0;
    if constexpr (LAYOUT_T == FIA_LAYOUT::BNSD) {
    } else {
        s1BaseSize = constInfo.mBaseSize / constInfo.gSize;
    }
    int64_t s1Offset = s1BaseSize * s1Idx;
    int64_t s2FirstToken =
        ClipSInnerToken(s1Offset - tempLoopInfo.preTokensPerBatch, 0, tempLoopInfo.curActualSeqLen);
    curS2Start = s2FirstToken / constInfo.s2BaseSize;
    if (!isStart || s2Start == 0) { // 没有切S2或不是初始获得的s2Start
        s2Start = curS2Start;
    }

    int64_t s2LastToken =
        ClipSInnerToken(s1Offset + tempLoopInfo.nextTokensPerBatch + s1BaseSize, 0, tempLoopInfo.curActualSeqLen);
    curS2End = (s2LastToken + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    tempLoopInfo.s2LoopTimes = isEnd ? constInfo.s2End + 1 : curS2End;
}

template <typename FIAT>
__aicore__ inline void FiaKernelNonQuantMla<FIAT>::UpdateInnerNum(uint32_t &s2End, uint32_t actS1Size,
                                                                                     uint32_t actS2Size, uint32_t s1Idx)
{
    if (actS2Size == 0) {
        s2End = 1;
        return;
    }
    if (!constInfo.slidingFlag) {
        s2End = (actS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
        return;
    }
    uint32_t s1BaseSize = 0;
    if constexpr (LAYOUT_T == FIA_LAYOUT::BNSD) {
    } else {
        s1BaseSize = constInfo.mBaseSize / constInfo.gSize;
    }
    int64_t s1Offset = s1BaseSize * s1Idx;
    int64_t nextTokensPerBatch =
        static_cast<int64_t>(actS2Size) - static_cast<int64_t>(actS1Size) + static_cast<int64_t>(constInfo.nextToken);
    int64_t s2LastToken =
        ClipSInnerToken(s1Offset + nextTokensPerBatch + s1BaseSize, 0, actS2Size);
    s2End = (s2LastToken + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
}

template <typename FIAT>
__aicore__ inline void FiaKernelNonQuantMla<FIAT>::DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx)
{
    if ASCEND_IS_AIV {
        InitAllZeroOutput(bIdx, n2Idx);
    }
}

template <typename FIAT> __aicore__ inline void FiaKernelNonQuantMla<FIAT>::UpdateInnerLoopCond()
{
    if ((tempLoopInfo.curActualSeqLen == 0) || (tempLoopInfo.actS1Size == 0)) {
        tempLoopInfo.curActSeqLenIsZero = true;
        return;
    }
    tempLoopInfo.curActSeqLenIsZero = false;
    tempLoopInfo.s2BasicSizeTail = tempLoopInfo.curActualSeqLen % constInfo.s2BaseSize;
    tempLoopInfo.s2BasicSizeTail =
        (tempLoopInfo.s2BasicSizeTail == 0) ? constInfo.s2BaseSize : tempLoopInfo.s2BasicSizeTail;
    tempLoopInfo.mBasicSizeTail = (tempLoopInfo.actS1Size * constInfo.gSize) % constInfo.mBaseSize;
    tempLoopInfo.mBasicSizeTail =
        (tempLoopInfo.mBasicSizeTail == 0) ? constInfo.mBaseSize : tempLoopInfo.mBasicSizeTail;
    tempLoopInfo.s2LoopTimes = 0;
}

template <typename FIAT>
__aicore__ inline uint64_t FiaKernelNonQuantMla<FIAT>::SeqLenFromTensorList(uint32_t bIndex)
{
    uint64_t dimInfo[4]; // this mem is used to set shapeinfo, BSH(3) or BNSD(4)
    AscendC::TensorDesc<__gm__ uint8_t> keyTensorDesc;
    ListTensorDesc keyListTensorDesc((__gm__ void *)keyPtr);
    keyTensorDesc.SetShapeAddr(&dimInfo[0]);
    keyListTensorDesc.GetDesc(keyTensorDesc, bIndex);
    if constexpr (LAYOUT_T == FIA_LAYOUT::BSH || LAYOUT_T == FIA_LAYOUT::BSND) {
        return keyTensorDesc.GetShape(1); // BSH, idx of s is 1
    } else {
        return keyTensorDesc.GetShape(2); // BNSD, idx of s is 2
    }
}

template <typename FIAT>
__aicore__ inline void FiaKernelNonQuantMla<FIAT>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift,
    __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
    const FusedInferAttentionScoreTilingData *__restrict tiling, __gm__ uint8_t *gmTiling, TPipe *tPipe, bool isPrefix)
{
    if ASCEND_IS_AIV {
        tmpBlockIdx = GetBlockIdx(); // vec:0-47
        aiCoreIdx = tmpBlockIdx / 2;
    } else {
        tmpBlockIdx = GetBlockIdx(); // cube:0-23
        aiCoreIdx = tmpBlockIdx;
    }

    // init tiling data
    tilingData = tiling;
    if (aiCoreIdx >= tilingData->baseParams.usedCoreNum) {
        return;
    }

    InitTilingData();
    // 初始化计算参数
    InitActualSeqLen(actualSeqLengthsQ, actualSeqLengths);
    InitCalcParamsEach();

    pipe = tPipe;
    keyPtr = key;
    valuePtr = value;

    // init global buffer
    queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
    qRopeGm.SetGlobalBuffer((__gm__ Q_ROPE_T *)queryRope);
    kRopeGm.SetGlobalBuffer((__gm__ K_ROPE_T *)keyRope);

    attentionOutGm.SetGlobalBuffer((__gm__ OUT_T *)attentionOut);
    if ASCEND_IS_AIV {
        if (constInfo.slidingFlag && constInfo.needInit) {
            InitOutputSingleCore();
        }
    }
    // batch连续时,只需要初始化一次;不连续时,需要在使用时根据batchIdx初始化
    if (batchContinuous) {
        InitKeyGm(0);
        InitValueGm(0);
    }

    if (constInfo.attenMaskFlag) {
        attenMaskBoolGm.SetGlobalBuffer((__gm__ bool *)attenMask);
    }

    if constexpr (PAGE_ATTENTION) {
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
    }

    // workspace 内存排布
    uint64_t offset = 0;

    mm1ResGm.SetGlobalBuffer(
        (__gm__ MM1_OUT_T *)(workspace + offset +
                             aiCoreIdx * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(MM1_OUT_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(MM1_OUT_T);

    vec1ResGm.SetGlobalBuffer(
        (__gm__ KV_T *)(workspace + offset + aiCoreIdx * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(KV_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(KV_T);

    mm2ResGm.SetGlobalBuffer(
        (__gm__ MM2_OUT_T *)(workspace + offset +
                             aiCoreIdx * dbWorkspaceRatio * constInfo.bmm2ResUbSize * sizeof(MM2_OUT_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.bmm2ResUbSize * sizeof(MM2_OUT_T);
    mm2ResInt32Gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(mm2ResGm.GetPhyAddr(0)));

    vec2ResGm.SetGlobalBuffer(
        (__gm__ T *)(workspace + offset + aiCoreIdx * dbWorkspaceRatio * constInfo.bmm2ResUbSize * sizeof(T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.bmm2ResUbSize * sizeof(T);

    if constexpr (FLASH_DECODE) {
        accumOutGm.SetGlobalBuffer((__gm__ float *)(workspace + offset));
        offset = offset + tilingData->workspaceParams.fdAccumOutSize * sizeof(float);
        lseSumFdGm.SetGlobalBuffer((__gm__ float *)(workspace + offset));
        lseMaxFdGm.SetGlobalBuffer((__gm__ float *)(workspace + offset) + tilingData->workspaceParams.fdLogSumExpSize / 2);
        offset = offset + tilingData->workspaceParams.fdLogSumExpSize * sizeof(float);
    }

    if ASCEND_IS_AIV {
        if constexpr (FLASH_DECODE) {
            fdService.InitParams(constInfo);
            fdService.InitGlobalTensor(lseMaxFdGm, lseSumFdGm, accumOutGm, attentionOutGm, actualSeqLengthsGmQ);
        }
        vectorService.InitParams(constInfo, tilingData);
        vectorService.InitMm2ResInt32GmGlobalTensor(mm2ResInt32Gm);
        vectorService.InitVec1GlobalTensor(mm1ResGm, vec1ResGm, attenMaskBoolGm, actualSeqLengthsGmQ,
                                           actualSeqLengthsGm, lseMaxFdGm, lseSumFdGm);
        vectorService.InitVec2GlobalTensor(accumOutGm, vec2ResGm, mm2ResGm, attentionOutGm);
    }

    if ASCEND_IS_AIC {
        matmulService.InitParams(constInfo);
        matmulService.InitMm1GlobalTensor(queryGm, qRopeGm, keyGm, kRopeGm, mm1ResGm);
        matmulService.InitMm2GlobalTensor(vec1ResGm, valueGm, mm2ResGm, attentionOutGm);
        matmulService.InitPageAttentionInfo(blockTableGm, constInfo.kvCacheBlockSize, constInfo.maxBlockNumPerBatch);
    }
    // 要在InitParams之后执行
    if (pipe != nullptr) {
        InitBuffers();
    }
}

template <typename FIAT> __aicore__ inline void FiaKernelNonQuantMla<FIAT>::InitKeyGm(uint32_t bIdx)
{
    ListTensorDesc keyListTensorDesc((__gm__ void *)keyPtr);
    key_ = (__gm__ uint8_t *)keyListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);

    keyGm.SetGlobalBuffer((__gm__ KV_T *)key_);
}

template <typename FIAT>
__aicore__ inline void FiaKernelNonQuantMla<FIAT>::InitValueGm(uint32_t bIdx)
{
    ListTensorDesc valueListTensorDesc((__gm__ void *)valuePtr);
    value_ = (__gm__ uint8_t *)valueListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);

    valueGm.SetGlobalBuffer((__gm__ KV_T *)value_);
}

template <typename FIAT>
__aicore__ inline void FiaKernelNonQuantMla<FIAT>::InitQuant(
    __gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2, __gm__ uint8_t *quantScale2,
    __gm__ uint8_t *quantOffset2, __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
    __gm__ uint8_t *keyAntiquantScale, __gm__ uint8_t *keyAntiquantOffset, __gm__ uint8_t *valueAntiquantScale,
    __gm__ uint8_t *valueAntiquantOffset, __gm__ uint8_t *keyRopeAntiquantScale, __gm__ uint8_t *workspace)
{
}

template <typename FIAT> __aicore__ inline void FiaKernelNonQuantMla<FIAT>::InitCalcParamsEach()
{
    // 这里是编译器优化写法
    // 定义一个局部数组变量coreSidxEnd(存在栈上)，使用copy_data_align64接口拷贝tiling中数组的内容到栈上
    // 其他非数组的tiling成员在不使用时不需要做Tiling拷贝，从而优化Tiling拷贝耗时
#ifdef ASCENDC_CPU_DEBUG
    const uint32_t *bN2End = tilingData->outerSplitParams.bN2End;
    const uint32_t *gS1End = tilingData->outerSplitParams.gS1End;
    const uint32_t *s2End = tilingData->outerSplitParams.s2End;
    const uint32_t *s2SplitStartIdxOfCore = tilingData->fdParams.s2SplitStartIdxOfCore;
#else
    uint32_t bN2End[ARRAY_SIZE(tilingData->outerSplitParams.bN2End)];
    uint32_t gS1End[ARRAY_SIZE(tilingData->outerSplitParams.gS1End)];
    uint32_t s2End[ARRAY_SIZE(tilingData->outerSplitParams.s2End)];
    uint32_t s2SplitStartIdxOfCore[ARRAY_SIZE(tilingData->fdParams.s2SplitStartIdxOfCore)];
    copy_data_align64((uint8_t *)bN2End, (uint8_t *)(tilingData->outerSplitParams.bN2End), sizeof(bN2End));
    copy_data_align64((uint8_t *)gS1End, (uint8_t *)(tilingData->outerSplitParams.gS1End), sizeof(gS1End));
    copy_data_align64((uint8_t *)s2End, (uint8_t *)(tilingData->outerSplitParams.s2End), sizeof(s2End));
    copy_data_align64((uint8_t *)s2SplitStartIdxOfCore,
                      (uint8_t *)(tilingData->fdParams.s2SplitStartIdxOfCore), sizeof(s2SplitStartIdxOfCore));
#endif
    // TND分核信息
    if (aiCoreIdx != 0) {
        constInfo.bN2Start = bN2End[aiCoreIdx - 1];
        constInfo.gS1Start = gS1End[aiCoreIdx - 1];
        constInfo.s2Start = s2End[aiCoreIdx - 1];
    }
    GetAxisEndIdx(bN2End[aiCoreIdx], gS1End[aiCoreIdx], s2End[aiCoreIdx]);

    constInfo.coreStartKVSplitPos = s2SplitStartIdxOfCore[aiCoreIdx];
}

template <typename FIAT>
__aicore__ inline void FiaKernelNonQuantMla<FIAT>::CalcParams(uint32_t loop, uint64_t s2Start,
                                                                                 uint32_t s2LoopIdx, AttentionCommon::RunInfo &info)
{
    info.loop = loop;
    info.bIdx = tempLoopInfo.bIdx;
    info.gS1Idx = tempLoopInfo.gS1Idx;
    info.s2Idx = s2LoopIdx;
    info.curSInnerLoopTimes = tempLoopInfo.s2LoopTimes;

    info.tndIsS2SplitCore = tempLoopInfo.tndIsS2SplitCore;
    info.tndCoreStartKVSplitPos = tempLoopInfo.tndCoreStartKVSplitPos;

    info.actS1Size = tempLoopInfo.actS1Size;
    info.actS2Size = tempLoopInfo.curActualSeqLen;
    //计算实际基本块size
    info.actualSingleProcessSInnerSize = constInfo.s2BaseSize; // info.actS2BaseSize
    int s2SplitNum = (tempLoopInfo.curActualSeqLen + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    if (info.s2Idx == s2SplitNum - 1) {
        info.actualSingleProcessSInnerSize = tempLoopInfo.s2BasicSizeTail; // info.actS2BaseSize
    }
    info.actualSingleProcessSInnerSizeAlign =
        Align((uint32_t)info.actualSingleProcessSInnerSize, (uint32_t)fa_base_vector::BYTE_BLOCK); // info.actS2BaseSizeAlign
    info.actMBaseSize = constInfo.mBaseSize;
    uint32_t remainedGS1Size = tempLoopInfo.actS1Size * constInfo.gSize - tempLoopInfo.gS1Idx;
    if (remainedGS1Size <= constInfo.mBaseSize && remainedGS1Size > 0) {
        info.actMBaseSize = tempLoopInfo.mBasicSizeTail;
    }

    info.isValid = s2LoopIdx < tempLoopInfo.s2LoopTimes;

    if (batchContinuous) {
        info.isChangeBatch = false;
    } else {
        if (loop == 0) {
            info.isChangeBatch = true;
        } else {
            info.isChangeBatch = (tempLoopInfo.n2Idx == 0 && s2LoopIdx == 0);
        }
    }

    info.isFirstSInnerLoop = s2LoopIdx == s2Start; // info.isFirstS2Loop
    if (info.isFirstSInnerLoop) {
        tempLoopInfo.bn2IdxInCurCore++;
    }
    info.isLastS2Loop = s2LoopIdx == tempLoopInfo.s2LoopTimes - 1;
    info.bn2IdxInCurCore = tempLoopInfo.bn2IdxInCurCore - 1;
    info.preTokensPerBatch = tempLoopInfo.preTokensPerBatch;
    info.nextTokensPerBatch = tempLoopInfo.nextTokensPerBatch;

    if (info.isFirstSInnerLoop) {
        if constexpr (LAYOUT_T == FIA_LAYOUT::BNSD) {
            // B,N2,G,S1,D
            tensorACoreOffset = info.bIdx * constInfo.qHeadNum * constInfo.qSeqSize * headDim + info.gS1Idx * headDim;
            tensorARopeCoreOffset =
                info.bIdx * constInfo.qHeadNum * constInfo.qSeqSize * headDimRope + info.gS1Idx * headDimRope;
            // B,N2,S2,D
            tensorBCoreOffset =
                info.bIdx * kvHeadNum * constInfo.kvSeqSize * headDim + info.n2Idx * constInfo.kvSeqSize * headDim;
            tensorBRopeCoreOffset = info.bIdx * kvHeadNum * constInfo.kvSeqSize * headDimRope +
                                    info.n2Idx * constInfo.kvSeqSize * headDimRope;
            if (!batchContinuous) {
                uint64_t seqSize = SeqLenFromTensorList(info.bIdx);
                tensorBCoreOffset = info.n2Idx * seqSize * headDim;
                tensorBRopeCoreOffset = info.n2Idx * seqSize * headDimRope;
            }
        } else {
            uint64_t actualSeqQPrefixSum;
            if constexpr (LAYOUT_T == FIA_LAYOUT::TND) {
                actualSeqQPrefixSum = (info.bIdx <= 0) ? 0 : actualSeqLengthsGmQ.GetValue(info.bIdx - 1);
            } else {
                actualSeqQPrefixSum = (info.bIdx <= 0) ? 0 : info.bIdx * constInfo.qSeqSize;
            }
            info.tndBIdxOffset = actualSeqQPrefixSum * constInfo.qHeadNum * headDim;
            uint64_t tndBIdxRopeOffset = actualSeqQPrefixSum * constInfo.qHeadNum * headDimRope;
            tensorACoreOffset = info.tndBIdxOffset + info.gS1Idx * headDim;
            tensorARopeCoreOffset = tndBIdxRopeOffset + info.gS1Idx * headDimRope;
            tensorBCoreOffset = info.bIdx * constInfo.kvSeqSize * kvHeadNum * headDim + info.n2Idx * headDim;
            tensorBRopeCoreOffset =
                info.bIdx * constInfo.kvSeqSize * kvHeadNum * headDimRope + info.n2Idx * headDimRope;

            if (!batchContinuous) {
                tensorBCoreOffset = info.n2Idx * headDim;
                tensorBRopeCoreOffset = info.n2Idx * headDimRope;
            }
        }
    }
    info.tensorAOffset = tensorACoreOffset;
    info.tensorARopeOffset = tensorARopeCoreOffset;
    if constexpr (LAYOUT_T == FIA_LAYOUT::BNSD) {
        info.tensorBOffset = tensorBCoreOffset + info.s2Idx * constInfo.s2BaseSize * headDim;
        info.tensorBRopeOffset = tensorBRopeCoreOffset + info.s2Idx * constInfo.s2BaseSize * headDimRope;
    } else {
        info.tensorBOffset = tensorBCoreOffset + info.s2Idx * constInfo.s2BaseSize * kvHeadNum * headDim;
        info.tensorBRopeOffset = tensorBRopeCoreOffset + info.s2Idx * constInfo.s2BaseSize * kvHeadNum * headDimRope;
    }
    info.attenOutOffset = tensorACoreOffset;

    uint64_t sInnerOffsetDataSize = info.s2Idx * constInfo.s2BaseSize;

    attenMaskCoreOffset = info.bIdx * constInfo.attenMaskSize;
    info.attenMaskOffset = attenMaskCoreOffset + sInnerOffsetDataSize;

    info.s2BatchOffset = sInnerOffsetDataSize;
}

template <typename FIAT>
__aicore__ inline void FiaKernelNonQuantMla<FIAT>::ComputeMm1(const AttentionCommon::RunInfo &info)
{
    if (info.isChangeBatch) {
        InitKeyGm(info.bIdx);
        matmulService.UpdateKey(keyGm);
    }
    uint32_t nBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t nBufferTail = info.actMBaseSize - (nBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferStartM = i * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? constInfo.nBufferMBaseSize : nBufferTail;
        matmulService.ComputeMm1(info, mSplitInfo);
        CrossCoreSetFlag<AttentionCommon::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V1);
    }
}

template <typename FIAT>
__aicore__ inline void FiaKernelNonQuantMla<FIAT>::ComputeMm2(const AttentionCommon::RunInfo &info)
{
    if (info.isChangeBatch) {
        InitValueGm(info.bIdx);
        matmulService.UpdateValue(valueGm);
    }
    uint32_t nBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t nBufferTail = info.actMBaseSize - (nBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferStartM = i * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? constInfo.nBufferMBaseSize : nBufferTail;
        CrossCoreWaitFlag(constInfo.syncV1C2);
        matmulService.ComputeMm2(info, mSplitInfo);
        CrossCoreSetFlag<AttentionCommon::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC2V2);
        CrossCoreSetFlag<AttentionCommon::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC2V1);
    }
}

template <typename FIAT>
__aicore__ inline void FiaKernelNonQuantMla<FIAT>::FlashDecode()
{
    if (tmpBlockIdx < tilingData->fdParams.usedVecNumOfFd) {
        FDparams fdParams;
        fdService.InitBuffers(pipe);
        AscendC::ICachePreLoad(fdPrefetchLen);
#ifdef ASCENDC_CPU_DEBUG
        const uint32_t *bN2IdxOfFdHead = tilingData->fdParams.bN2IdxOfFdHead;
        const uint32_t *gS1IdxOfFdHead = tilingData->fdParams.gS1IdxOfFdHead;
        const uint32_t *s2SplitNumOfFdHead = tilingData->fdParams.s2SplitNumOfFdHead;
        const uint32_t *gS1IdxEndOfFdHead = tilingData->fdParams.gS1IdxEndOfFdHead;
        const uint32_t *gS1IdxEndOfFdHeadSplit = tilingData->fdParams.gS1IdxEndOfFdHeadSplit;
        const uint32_t *gS1SplitNumOfFdHead = tilingData->fdParams.gS1SplitNumOfFdHead;
        const uint32_t *gS1LastPartSizeOfFdHead = tilingData->fdParams.gS1LastPartSizeOfFdHead;
#else
        uint32_t bN2IdxOfFdHead[ARRAY_SIZE(tilingData->fdParams.bN2IdxOfFdHead)];
        uint32_t gS1IdxOfFdHead[ARRAY_SIZE(tilingData->fdParams.gS1IdxOfFdHead)];
        uint32_t s2SplitNumOfFdHead[ARRAY_SIZE(tilingData->fdParams.s2SplitNumOfFdHead)];
        uint32_t gS1IdxEndOfFdHead[ARRAY_SIZE(tilingData->fdParams.gS1IdxEndOfFdHead)];
        uint32_t gS1IdxEndOfFdHeadSplit[ARRAY_SIZE(tilingData->fdParams.gS1IdxEndOfFdHeadSplit)];
        uint32_t gS1SplitNumOfFdHead[ARRAY_SIZE(tilingData->fdParams.gS1SplitNumOfFdHead)];
        uint32_t gS1LastPartSizeOfFdHead[ARRAY_SIZE(tilingData->fdParams.gS1LastPartSizeOfFdHead)];
        copy_data_align64((uint8_t *)bN2IdxOfFdHead, (uint8_t *)(tilingData->fdParams.bN2IdxOfFdHead),
                    sizeof(bN2IdxOfFdHead));
        copy_data_align64((uint8_t *)gS1IdxOfFdHead, (uint8_t *)(tilingData->fdParams.gS1IdxOfFdHead),
                    sizeof(gS1IdxOfFdHead));
        copy_data_align64((uint8_t *)s2SplitNumOfFdHead, (uint8_t *)(tilingData->fdParams.s2SplitNumOfFdHead),
                    sizeof(s2SplitNumOfFdHead));
        copy_data_align64((uint8_t *)gS1IdxEndOfFdHead, (uint8_t *)(tilingData->fdParams.gS1IdxEndOfFdHead),
                    sizeof(gS1IdxEndOfFdHead));
        copy_data_align64((uint8_t *)gS1IdxEndOfFdHeadSplit,
                    (uint8_t *)(tilingData->fdParams.gS1IdxEndOfFdHeadSplit),
                    sizeof(gS1IdxEndOfFdHeadSplit));
        copy_data_align64((uint8_t *)gS1SplitNumOfFdHead, (uint8_t *)(tilingData->fdParams.gS1SplitNumOfFdHead),
                    sizeof(gS1SplitNumOfFdHead));
        copy_data_align64((uint8_t *)gS1LastPartSizeOfFdHead,
                    (uint8_t *)(tilingData->fdParams.gS1LastPartSizeOfFdHead),
                    sizeof(gS1LastPartSizeOfFdHead));
#endif
        fdParams = {bN2IdxOfFdHead, gS1IdxOfFdHead, s2SplitNumOfFdHead, gS1SplitNumOfFdHead, gS1LastPartSizeOfFdHead,
                gS1IdxEndOfFdHead, gS1IdxEndOfFdHeadSplit, tilingData->fdParams.usedVecNumOfFd,
                tilingData->fdParams.gS1BaseSizeOfFd};

        SyncAll();

        fdService.AllocEventID();
        fdService.InitDecodeParams();
        fdService.FlashDecode(fdParams);
        fdService.FreeEventID();
    } else {
        // superkernel 场景，启动核数大于实际运行核数时，未启动的核仅需要保留 SyncAll
        SyncAll();
    }
}

template <typename FIAT> __aicore__ inline void FiaKernelNonQuantMla<FIAT>::Process()
{
    // usedCoreNum: 使用的总核数
    if (aiCoreIdx < usedCoreNum) {
        if ASCEND_IS_AIV {
            vectorService.AllocEventID();
            vectorService.InitSoftmaxDefaultBuffer();
        } else {
            matmulService.AllocEventID();
        }
        ProcessBalance();

        if ASCEND_IS_AIV {
            vectorService.FreeEventID();
        } else {
            matmulService.FreeEventID();
        }
    }

    if constexpr (FLASH_DECODE) {
        if ASCEND_IS_AIV {
            FlashDecode();
        }
    }
}

template <typename FIAT>
__aicore__ inline void FiaKernelNonQuantMla<FIAT>::GetBN2Idx(uint32_t bN2Idx, uint32_t &bIdx,
                                                                                uint32_t &n2Idx)
{
    bIdx = bN2Idx / kvHeadNum;
    n2Idx = bN2Idx % kvHeadNum;
}

template <typename FIAT> __aicore__ inline void FiaKernelNonQuantMla<FIAT>::ProcessBalance()
{
    AttentionCommon::RunInfo extraInfo[FIA_PRELOAD_TASK_CACHE_SIZE];
    uint32_t gloop = 0;
    int gS1LoopEnd;
    bool globalLoopStart = true;
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<AttentionCommon::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC2V1);
    }
    for (uint32_t bN2LoopIdx = constInfo.bN2Start; bN2LoopIdx <= constInfo.bN2End; bN2LoopIdx++) {
        GetBN2Idx(bN2LoopIdx, tempLoopInfo.bIdx, tempLoopInfo.n2Idx);
        GetActualSeqLen(tempLoopInfo.bIdx);
        GetPreNextTokensLeftUp();
        UpdateInnerLoopCond();

        if (tempLoopInfo.curActSeqLenIsZero) {
            DealActSeqLenIsZero(tempLoopInfo.bIdx, tempLoopInfo.n2Idx);
            continue;
        }
        int gS1SplitNum = (tempLoopInfo.actS1Size * constInfo.gSize + constInfo.mBaseSize - 1) / constInfo.mBaseSize;
        gS1LoopEnd = (bN2LoopIdx == constInfo.bN2End) ? constInfo.gS1End : gS1SplitNum - 1;
        for (uint32_t gS1LoopIdx = constInfo.gS1Start; gS1LoopIdx <= gS1LoopEnd; gS1LoopIdx++) {
            tempLoopInfo.gS1Idx = gS1LoopIdx * constInfo.mBaseSize;

            uint32_t s2SplitNum =
                (tempLoopInfo.curActualSeqLen + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize; // S2切分份数
            bool isEnd = (bN2LoopIdx == constInfo.bN2End) && (gS1LoopIdx == constInfo.gS1End);
            tempLoopInfo.tndCoreStartKVSplitPos = globalLoopStart ? constInfo.coreStartKVSplitPos : 0;
            uint32_t curS2Start = 0;
            UpdateInner(constInfo.s2Start, tempLoopInfo.s2LoopTimes, curS2Start, s2SplitNum,
                        gS1LoopIdx, gS1LoopIdx == constInfo.gS1Start, isEnd);
            // 当前s2是否被切，决定了输出是否要写到attenOut上
            tempLoopInfo.tndIsS2SplitCore =
                ((constInfo.s2Start == curS2Start) && (tempLoopInfo.s2LoopTimes == s2SplitNum)) ? false : true;
            for (int s2LoopIdx = constInfo.s2Start; s2LoopIdx < tempLoopInfo.s2LoopTimes; s2LoopIdx++) {
                CalcParams(gloop, constInfo.s2Start, s2LoopIdx, extraInfo[gloop % FIA_PRELOAD_TASK_CACHE_SIZE]); // 创建本轮任务
                // PreloadPipeline loop初始值要求为 PRELOAD_NUM
                PreloadPipeline(gloop, extraInfo);
                ++gloop;
            }
            globalLoopStart = false;
            constInfo.s2Start = 0;
        }
        constInfo.gS1Start = 0;
    }

    for (int i = 0; i < 2; ++i) {  // 2: extra loop
        PreloadPipeline(gloop, extraInfo);
        ++gloop;
    }

    if ASCEND_IS_AIV {
        CrossCoreWaitFlag(constInfo.syncC2V1);
    }
}

template <typename FIAT>
__aicore__ inline void
FiaKernelNonQuantMla<FIAT>::PreloadPipeline(uint32_t loop,
                                                               AttentionCommon::RunInfo extraInfo[FIA_PRELOAD_TASK_CACHE_SIZE])
{
    AttentionCommon::RunInfo &extraInfo0 = extraInfo[loop % FIA_PRELOAD_TASK_CACHE_SIZE];       // 本轮任务
    AttentionCommon::RunInfo &extraInfo2 = extraInfo[(loop + 2) % FIA_PRELOAD_TASK_CACHE_SIZE]; // 上一轮任务
    AttentionCommon::RunInfo &extraInfo1 = extraInfo[(loop + 1) % FIA_PRELOAD_TASK_CACHE_SIZE]; // 上两轮任务

    if (extraInfo0.isValid) {
        if ASCEND_IS_AIC {
            ComputeMm1(extraInfo0);
        }
    }
    if (extraInfo2.isValid) {
        if ASCEND_IS_AIV {
            vectorService.ProcessVec1L(extraInfo2);
        }
        if ASCEND_IS_AIC {
            ComputeMm2(extraInfo2);
        }
    }
    if (extraInfo1.isValid) {
        if ASCEND_IS_AIV {
            vectorService.ProcessVec2L(extraInfo1);
        }
        extraInfo1.isValid = false;
    }
}

template <typename FIAT>
__aicore__ inline uint64_t
FiaKernelNonQuantMla<FIAT>::GetBalanceActualSeqLengths(GlobalTensor<uint64_t> &actualSeqLengths,
                                                                          uint32_t bIdx)
{
    if constexpr (LAYOUT_T == FIA_LAYOUT::TND) {
        if (bIdx > 0) {
            return actualSeqLengths.GetValue(bIdx) - actualSeqLengths.GetValue(bIdx - 1);
        } else if (bIdx == 0) {
            return actualSeqLengths.GetValue(0);
        } else {
            return 0;
        }
    } else {
        if (constInfo.actualLenQDims == 0) {
            return constInfo.qSeqSize;
        } else if (constInfo.actualLenQDims == 1) {
            return actualSeqLengths.GetValue(0);
        } else {
            return actualSeqLengths.GetValue(bIdx);
        }
    }
}

template <typename FIAT>
__aicore__ inline void FiaKernelNonQuantMla<FIAT>::GetAxisEndIdx(uint32_t bN2End, uint32_t s1GEnd,
                                                                                    uint32_t s2End)
{
    constInfo.bN2End = bN2End;
    if (s1GEnd == 0 && s2End == 0) {
        constInfo.bN2End = bN2End - 1;
    }
    if (s2End > 0) {
        constInfo.gS1End = s1GEnd;  
        constInfo.s2End = s2End - 1;
        return;  
    }

    uint32_t bEnd = constInfo.bN2End / kvHeadNum;
    uint32_t actualSeqQ = GetBalanceActualSeqLengths(actualSeqLengthsGmQ, bEnd);
    uint32_t actualSeqKV = GetActualSeqLenKV(bEnd);
    uint32_t s2BaseNum = 0;
    if (s1GEnd > 0) {
        constInfo.gS1End = s1GEnd - 1;  
    } else {
        uint32_t s1GBaseNum = (actualSeqQ * constInfo.gSize + constInfo.mBaseSize - 1) / constInfo.mBaseSize;
        if (actualSeqQ == 0) {
            s1GBaseNum = 1;
            s2BaseNum = 1;
        }
        constInfo.gS1End = s1GBaseNum - 1;
    } 
    UpdateInnerNum(s2BaseNum, actualSeqQ, actualSeqKV, constInfo.gS1End); 
    constInfo.s2End = s2BaseNum - 1;
}
#endif // FIA_KERNEL_NONQUANT_MLA_H
