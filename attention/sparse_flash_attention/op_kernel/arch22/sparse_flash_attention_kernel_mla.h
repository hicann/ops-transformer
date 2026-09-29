/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_flash_attention_kernel_mla.h
 * \brief
 */

#ifndef SPARSE_FLASH_ATTENTION_KERNEL_MLA_H
#define SPARSE_FLASH_ATTENTION_KERNEL_MLA_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "sparse_flash_attention_common.h"
#include "sparse_flash_attention_service_cube_mla.h"
#include "sparse_flash_attention_service_vector_mla.h"

using namespace matmul;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

// 由于S2循环前，RunInfo还没有赋值，使用Bngs1Param临时存放B、N、S1轴相关的信息；同时减少重复计算
struct TempLoopInfo {
    uint32_t bn2IdxInCurCore = 0;
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;
    uint32_t s2LoopTimes = 0U; // S2方向循环的总次数，无论TND还是BXXD都是等于实际次数，不用减1
    uint64_t s2BasicSizeTail = 0U; // S2方向循环的尾基本块大小
    uint64_t curActualSeqLen = 0ULL;
    uint64_t curActualSeqLenOri = 0ULL;
    uint64_t actS1Size = 1ULL; // TND场景下当前Batch循环处理的S1轴的大小
    int32_t nextTokensPerBatch = 0;
    uint32_t tndCoreStartKVSplitPos;
    uint64_t mBasicSizeTail = 0U; // gS1方向循环的尾基本块大小
    uint32_t gS1Idx = 0U;
    bool curActSeqLenIsZero = false;
    bool tndIsS2SplitCore;
};

template <typename SFAT>
class SparseFlashAttentionMla {
public:
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using Q_T = typename SFAT::queryType;
    using KV_T = typename SFAT::kvType;
    using OUT_T = typename SFAT::outputType;
    using Q_ROPE_T = Q_T;
    using K_ROPE_T = KV_T;
    using UPDATE_T = T;
    using MM1_OUT_T = T;
    using MM2_OUT_T = T;

    __aicore__ inline SparseFlashAttentionMla(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *sparseIndices, __gm__ uint8_t *actualSeqLengthsQ,
                                __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *blockTable, __gm__ uint8_t *queryRope,
                                __gm__ uint8_t *keyRope, __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxMax,
                                __gm__ uint8_t *softmaxSum, __gm__ uint8_t *workspace,
                                const SparseFlashAttentionTilingDataMla *__restrict tiling, __gm__ uint8_t *gmTiling,
                                TPipe *tPipe);

    __aicore__ inline void Process();

private:
    static constexpr bool PAGE_ATTENTION = SFAT::pageAttention;
    static constexpr int TEMPLATE_MODE = SFAT::templateMode;
    static constexpr bool FLASH_DECODE = SFAT::flashDecode;
    static constexpr SFA_LAYOUT LAYOUT_T = SFAT::layout;
    static constexpr SFA_LAYOUT KV_LAYOUT_T = SFAT::kvLayout;

    static constexpr uint32_t PRELOAD_NUM = 2;
    static constexpr uint32_t N_BUFFER_M_BASIC_SIZE = 256;
    static constexpr uint32_t SFA_PRELOAD_TASK_CACHE_SIZE = 3;

    static constexpr uint32_t SYNC_V0_C1_FLAG = 6;
    static constexpr uint32_t SYNC_C1_V1_FLAG = 7;
    static constexpr uint32_t SYNC_V1_C2_FLAG = 8;
    static constexpr uint32_t SYNC_C2_V2_FLAG = 9;
    static constexpr uint32_t SYNC_C2_V1_FLAG = 4;
    static constexpr uint32_t SYNC_V1_NUPDATE_C2_FLAG = 5;

    static constexpr uint64_t SYNC_MM2RES_BUF1_FLAG = 10;
    static constexpr uint64_t SYNC_MM2RES_BUF2_FLAG = 11;
    static constexpr uint64_t SYNC_FDOUTPUT_BUF_FLAG = 12;

    static constexpr uint32_t BLOCK_ELEMENT_NUM = SFAVectorService<SFAT>::BYTE_BLOCK / sizeof(T);

    static constexpr uint64_t kvHeadNum = 1ULL;
    static constexpr uint64_t headDim = 512ULL;
    static constexpr uint64_t headDimAlign = 512ULL;
    static constexpr uint64_t headDimRope = 64ULL;
    static constexpr uint32_t msdIterNum = 2U;

    static constexpr uint32_t dbWorkspaceRatio = PRELOAD_NUM;

    const SparseFlashAttentionTilingDataMla *__restrict tilingData = nullptr;

    TPipe *pipe = nullptr;

    uint64_t mSizeVStart = 0ULL;
    int64_t threshold = 0;
    uint64_t topKBaseOffset = 0ULL;
    uint64_t s2BatchBaseOffset = 0;
    uint64_t tensorACoreOffset = 0ULL;
    uint64_t tensorBCoreOffset = 0ULL;
    uint64_t tensorARopeCoreOffset = 0ULL;
    uint64_t tensorBRopeCoreOffset = 0ULL;
    uint64_t tensorBOffset = 0ULL;
    uint64_t attenOutOffset = 0ULL;

    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;
    uint32_t usedCoreNum = 0U;

    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;

    ConstInfo sfaKernelConstInfo{};
    TempLoopInfo sfaKernelLoopInfo{};

    SFAMatmulService<SFAT> matmulService;
    SFAVectorService<SFAT> sfaKernelVectorService;

    GlobalTensor<Q_T> queryGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<KV_T> valueGm;
    GlobalTensor<Q_ROPE_T> qRopeGm;
    GlobalTensor<K_ROPE_T> kRopeGm;

    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<T> softmaxMaxGm;
    GlobalTensor<T> softmaxSumGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<int32_t> topKGm;

    GlobalTensor<int32_t> actualSeqLengthsQGm;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;

    // workspace
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<KV_T> vec1ResGm;
    GlobalTensor<MM2_OUT_T> mm2ResGm;
    GlobalTensor<KV_T> kvMergeGm_;
    GlobalTensor<int32_t> kvValidSizeGm_;

    GlobalTensor<int32_t> mm2ResInt32Gm;
    GlobalTensor<UPDATE_T> vec2ResGm;

    GlobalTensor<T> accumOutGm;
    GlobalTensor<T> lseSumFdGm;
    GlobalTensor<T> lseMaxFdGm;

    GlobalTensor<T> lseSumFaGm;
    GlobalTensor<T> lseMaxFaGm;

    // ================================Init functions===================================
    __aicore__ inline void InitTilingData();
    __aicore__ inline void InitCalcParamsEach();
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths);
    __aicore__ inline void InitOutputSingleCore();
    // ================================Process functions================================
    __aicore__ inline void ProcessBalance();
    __aicore__ inline void PreloadPipeline(uint32_t loop, uint64_t s2Start, uint64_t s2LoopIdx,
                                           RunInfo extraInfo[SFA_PRELOAD_TASK_CACHE_SIZE], uint32_t &curTopKIdx,
                                           uint64_t &curOffsetInSparseBlock);
    // ================================Offset Calc=====================================
    __aicore__ inline void GetActualSeqLen(uint32_t bIdx, uint32_t s1Idx = 0);
    __aicore__ inline void GetSparseActualSeqLen(uint32_t bIdx, uint32_t s1Idx, uint32_t n2Idx);
    __aicore__ inline void CalcSinnerTopKBegin(RunInfo &sfaKernelRunInfo, uint32_t &curTopKIdx,
                                               uint64_t &curOffsetInSparseBlock);
    __aicore__ inline void UpdateInnerLoopCond();
    __aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t s1Idx, uint32_t n2Idx);
    __aicore__ inline void CalcParams(uint32_t loop, uint64_t s2Start, uint32_t s2LoopIdx, RunInfo &sfaKernelRunInfo);
    __aicore__ inline void GetAxisStartIdx(uint32_t bN2EndPrev, uint32_t gS1EndPrev, uint32_t s2EndPrev);
    __aicore__ inline uint64_t GetBalanceActualSeqLengths(GlobalTensor<int32_t> &actualSeqLengths, uint32_t bIdx);
    __aicore__ inline uint32_t GetActualSeqLenKV(uint32_t bIdx);
    __aicore__ inline void GetBN2Idx(uint32_t bN2Idx, uint32_t &bIdx, uint32_t &n2Idx);
    __aicore__ inline void UpdateInner(uint32_t &s2End, uint32_t &curS2End, uint32_t s1Idx, bool isEnd);
    __aicore__ inline void GetPreNextTokensLeftUp();
    // ================================Mm1==============================================
    __aicore__ inline void ComputeMm1(const RunInfo &sfaKernelRunInfo);
    // ================================Mm2==============================================
    __aicore__ inline void ComputeMm2(const RunInfo &sfaKernelRunInfo);
    __aicore__ inline void Bmm2DataCopyOut(uint64_t attenOutOffset, LocalTensor<OUT_T> &attenOutUb, uint32_t startRow,
                                           uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void InitAllZeroOutput(uint32_t bIdx, uint32_t s1Idx, uint32_t n2Idx);
};

template <typename SFAT>
__aicore__ inline void SparseFlashAttentionMla<SFAT>::InitTilingData()
{
    usedCoreNum = tilingData->singleCoreParams.usedCoreNum;
    sfaKernelConstInfo.splitKVNum = tilingData->splitKVParams.s2;
    sfaKernelConstInfo.mmResUbSize = tilingData->singleCoreTensorSize.mmResUbSize;
    sfaKernelConstInfo.bmm2ResUbSize = tilingData->singleCoreTensorSize.bmm2ResUbSize;
    sfaKernelConstInfo.vec1ResUbSize = sfaKernelConstInfo.mmResUbSize * msdIterNum;

    sfaKernelConstInfo.batchSize = tilingData->baseParams.batchSize;
    sfaKernelConstInfo.qHeadNum = sfaKernelConstInfo.gSize = tilingData->baseParams.nNumOfQInOneGroup;
    sfaKernelConstInfo.kvSeqSize = tilingData->baseParams.seqSize;
    sfaKernelConstInfo.qSeqSize = tilingData->baseParams.qSeqSize;
    sfaKernelConstInfo.maxBlockNumPerBatch = tilingData->baseParams.maxBlockNumPerBatch;
    sfaKernelConstInfo.kvCacheBlockSize = tilingData->baseParams.blockSize;
    sfaKernelConstInfo.outputLayout = static_cast<SFA_LAYOUT>(tilingData->baseParams.outputLayout);
    sfaKernelConstInfo.mBaseSize = tilingData->innerSplitParams.mBaseSize;
    sfaKernelConstInfo.s2BaseSize = tilingData->innerSplitParams.s2BaseSize;
    sfaKernelConstInfo.kvHeadNum = kvHeadNum;
    sfaKernelConstInfo.headDim = headDim;
    sfaKernelConstInfo.headDimRope = headDimRope;
    sfaKernelConstInfo.sparseBlockSize = tilingData->baseParams.sparseBlockSize;
    sfaKernelConstInfo.sparseBlockCount = tilingData->baseParams.sparseBlockCount;
    sfaKernelConstInfo.sparseMode = tilingData->baseParams.sparseMode;
    sfaKernelConstInfo.preTokens = tilingData->baseParams.preTokens;
    sfaKernelConstInfo.nextTokens = tilingData->baseParams.nextTokens;
    sfaKernelConstInfo.attentionMode = tilingData->baseParams.attentionMode;
    sfaKernelConstInfo.returnSoftmaxLse = tilingData->baseParams.returnSoftmaxLse;

    sfaKernelConstInfo.preLoadNum = PRELOAD_NUM;
    sfaKernelConstInfo.nBufferMBaseSize = N_BUFFER_M_BASIC_SIZE;
    sfaKernelConstInfo.syncV0C1 = SYNC_V0_C1_FLAG;
    sfaKernelConstInfo.syncC1V1 = SYNC_C1_V1_FLAG;
    sfaKernelConstInfo.syncV1C2 = SYNC_V1_C2_FLAG;
    sfaKernelConstInfo.syncC2V2 = SYNC_C2_V2_FLAG;
    sfaKernelConstInfo.syncC2V1 = SYNC_C2_V1_FLAG;
    sfaKernelConstInfo.syncV1NupdateC2 = SYNC_V1_NUPDATE_C2_FLAG;
}

template <typename SFAT>
__aicore__ inline void SparseFlashAttentionMla<SFAT>::InitBuffers()
{
    if ASCEND_IS_AIV {
        sfaKernelVectorService.InitBuffers(pipe);
    } else {
        matmulService.InitBuffers(pipe);
    }
}

template <typename SFAT>
__aicore__ inline void SparseFlashAttentionMla<SFAT>::InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ,
                                                                       __gm__ uint8_t *actualSeqLengths)
{
    sfaKernelConstInfo.actualLenDimsQ = tilingData->baseParams.actualLenDimsQ;
    sfaKernelConstInfo.actualLenDimsKV = tilingData->baseParams.actualLenDimsKV;
    if (sfaKernelConstInfo.actualLenDimsKV != 0) {
        actualSeqLengthsKVGm.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengths, sfaKernelConstInfo.actualLenDimsKV);
    }
    if (sfaKernelConstInfo.actualLenDimsQ != 0) {
        actualSeqLengthsQGm.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengthsQ, sfaKernelConstInfo.actualLenDimsQ);
    }
}

template <typename SFAT>
__aicore__ inline void SparseFlashAttentionMla<SFAT>::InitAllZeroOutput(uint32_t bIdx, uint32_t s1Idx, uint32_t n2Idx)
{
    if (sfaKernelConstInfo.outputLayout == SFA_LAYOUT::TND) {
        uint32_t tBase = bIdx == 0 ? 0 : actualSeqLengthsQGm.GetValue(bIdx - 1);
        uint32_t s1Count = sfaKernelLoopInfo.actS1Size;

        uint64_t attenOutOffset = (tBase + s1Idx) * kvHeadNum * sfaKernelConstInfo.gSize * headDim + // T轴、s1轴偏移
                                  n2Idx * sfaKernelConstInfo.gSize * headDim;                        // N2轴偏移
        matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], sfaKernelConstInfo.gSize * headDim, 0);
        if (sfaKernelConstInfo.returnSoftmaxLse) {
            uint64_t softmaxSumOffset =
                n2Idx * actualSeqLengthsQGm.GetValue(sfaKernelConstInfo.batchSize - 1) * sfaKernelConstInfo.gSize +
                (tBase + s1Idx) * sfaKernelConstInfo.gSize;
            uint64_t softmaxMaxOffset = softmaxSumOffset;
            matmul::InitOutput<T>(softmaxSumGm[softmaxSumOffset], sfaKernelConstInfo.gSize, 0);
            matmul::InitOutput<T>(softmaxMaxGm[softmaxMaxOffset], sfaKernelConstInfo.gSize, 0);
        }
    } else if (sfaKernelConstInfo.outputLayout == SFA_LAYOUT::BSND) {
        uint64_t attenOutOffset = bIdx * sfaKernelConstInfo.qSeqSize * kvHeadNum * sfaKernelConstInfo.gSize * headDim +
                                  s1Idx * kvHeadNum * sfaKernelConstInfo.gSize * headDim + // B轴、S1轴偏移
                                  n2Idx * sfaKernelConstInfo.gSize * headDim;              // N2轴偏移
        matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], sfaKernelConstInfo.gSize * headDim, 0);
        if (sfaKernelConstInfo.returnSoftmaxLse) {
            uint64_t softmaxSumOffset = bIdx * kvHeadNum * sfaKernelConstInfo.qSeqSize * sfaKernelConstInfo.gSize +
                                        n2Idx * sfaKernelConstInfo.qSeqSize * sfaKernelConstInfo.gSize +
                                        s1Idx * sfaKernelConstInfo.gSize;
            uint64_t softmaxMaxOffset = softmaxSumOffset;
            matmul::InitOutput<T>(softmaxSumGm[softmaxSumOffset], sfaKernelConstInfo.gSize, 0);
            matmul::InitOutput<T>(softmaxMaxGm[softmaxMaxOffset], sfaKernelConstInfo.gSize, 0);
        }
    }
}

template <typename SFAT>
__aicore__ inline void SparseFlashAttentionMla<SFAT>::InitOutputSingleCore()
{
    uint32_t coreNum = GetBlockNum();
    if (coreNum != 0) {
        uint64_t totalOutputSize = sfaKernelConstInfo.batchSize * sfaKernelConstInfo.qHeadNum *
                                   sfaKernelConstInfo.qSeqSize * sfaKernelConstInfo.headDim;
        uint64_t singleCoreSize = (totalOutputSize + (2 * coreNum) - 1) / (2 * coreNum); // 2 means c:v = 1:2
        uint64_t tailSize = totalOutputSize - tmpBlockIdx * singleCoreSize;
        uint64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;
        if (tmpBlockIdx * singleCoreSize < totalOutputSize && singleInitOutputSize > 0) {
            matmul::InitOutput<OUT_T>(attentionOutGm[tmpBlockIdx * singleCoreSize], singleInitOutputSize, 0);
        }
        if (sfaKernelConstInfo.returnSoftmaxLse) {
            uint64_t totalReturnSoftmaxSize = sfaKernelConstInfo.batchSize * sfaKernelConstInfo.kvHeadNum *
                                              sfaKernelConstInfo.qSeqSize * sfaKernelConstInfo.gSize;
            uint64_t singleCoreReturnSoftmaxSize = (totalReturnSoftmaxSize + (2 * coreNum) - 1) / (2 * coreNum);
            uint64_t tailReturnSoftmaxSize = totalReturnSoftmaxSize - tmpBlockIdx * singleCoreReturnSoftmaxSize;
            uint64_t singleInitReturnSoftmaxSize = tailReturnSoftmaxSize < singleCoreReturnSoftmaxSize ?
                                                       tailReturnSoftmaxSize :
                                                       singleCoreReturnSoftmaxSize;
            if (tmpBlockIdx * singleCoreReturnSoftmaxSize < totalReturnSoftmaxSize && singleInitReturnSoftmaxSize > 0) {
                matmul::InitOutput<T>(softmaxSumGm[tmpBlockIdx * singleCoreReturnSoftmaxSize],
                                      singleInitReturnSoftmaxSize, 0);
                matmul::InitOutput<T>(softmaxMaxGm[tmpBlockIdx * singleCoreReturnSoftmaxSize],
                                      singleInitReturnSoftmaxSize, 0);
            }
        }
        SyncAll();
    }
}

template <typename SFAT>
__aicore__ inline void SparseFlashAttentionMla<SFAT>::GetActualSeqLen(uint32_t bIdx, uint32_t s1Idx)
{
    sfaKernelLoopInfo.curActualSeqLenOri = GetActualSeqLenKV(bIdx);
    sfaKernelLoopInfo.actS1Size = GetBalanceActualSeqLengths(actualSeqLengthsQGm, bIdx);
}

template <typename SFAT>
__aicore__ inline void SparseFlashAttentionMla<SFAT>::GetSparseActualSeqLen(uint32_t bIdx, uint32_t s1Idx,
                                                                            uint32_t n2Idx)
{
    if (sfaKernelLoopInfo.nextTokensPerBatch < 0 && s1Idx < (-sfaKernelLoopInfo.nextTokensPerBatch)) { // 存在行无效
        sfaKernelLoopInfo.curActualSeqLen = 0;
        return;
    }
    int64_t threshold = sfaKernelLoopInfo.curActualSeqLenOri;
    if (sfaKernelConstInfo.sparseMode == 3) {
        threshold = static_cast<int64_t>(sfaKernelLoopInfo.nextTokensPerBatch) + s1Idx + 1;
    }
    sfaKernelLoopInfo.curActualSeqLen =
        (sfaKernelConstInfo.sparseBlockCount * sfaKernelConstInfo.sparseBlockSize > threshold) ?
            threshold :
            (sfaKernelConstInfo.sparseBlockCount * sfaKernelConstInfo.sparseBlockSize);
}

template <typename SFAT>
__aicore__ inline uint32_t SparseFlashAttentionMla<SFAT>::GetActualSeqLenKV(uint32_t bIdx)
{
    if constexpr (KV_LAYOUT_T == SFA_LAYOUT::TND) {
        if (bIdx > 0) {
            return actualSeqLengthsKVGm.GetValue(bIdx) - actualSeqLengthsKVGm.GetValue(bIdx - 1);
        } else if (bIdx == 0) {
            return actualSeqLengthsKVGm.GetValue(0);
        } else {
            return 0;
        }
    } else {
        if (sfaKernelConstInfo.actualLenDimsKV == 0) {
            return sfaKernelConstInfo.kvSeqSize;
        } else if (sfaKernelConstInfo.actualLenDimsKV == 1) {
            return actualSeqLengthsKVGm.GetValue(0);
        } else {
            return actualSeqLengthsKVGm.GetValue(bIdx);
        }
    }
}

template <typename SFAT>
__aicore__ inline void SparseFlashAttentionMla<SFAT>::DealActSeqLenIsZero(uint32_t bIdx, uint32_t s1Idx, uint32_t n2Idx)
{
    if ASCEND_IS_AIV {
        InitAllZeroOutput(bIdx, s1Idx, n2Idx);
    }
}

template <typename SFAT>
__aicore__ inline void SparseFlashAttentionMla<SFAT>::GetPreNextTokensLeftUp()
{
    if (sfaKernelConstInfo.sparseMode == 3) {
        sfaKernelLoopInfo.nextTokensPerBatch = static_cast<int32_t>(sfaKernelLoopInfo.curActualSeqLenOri) -
                                               static_cast<int32_t>(sfaKernelLoopInfo.actS1Size);
    }
}

template <typename SFAT>
__aicore__ inline void SparseFlashAttentionMla<SFAT>::UpdateInnerLoopCond()
{
    if ((sfaKernelLoopInfo.curActualSeqLen == 0) || (sfaKernelLoopInfo.actS1Size == 0)) {
        sfaKernelLoopInfo.curActSeqLenIsZero = true;
        return;
    }
    sfaKernelLoopInfo.curActSeqLenIsZero = false;
    sfaKernelLoopInfo.mBasicSizeTail =
        (sfaKernelLoopInfo.actS1Size * sfaKernelConstInfo.gSize) % sfaKernelConstInfo.mBaseSize;
    sfaKernelLoopInfo.mBasicSizeTail =
        (sfaKernelLoopInfo.mBasicSizeTail == 0) ? sfaKernelConstInfo.mBaseSize : sfaKernelLoopInfo.mBasicSizeTail;
    sfaKernelLoopInfo.s2LoopTimes = 0;
}

template <typename SFAT>
__aicore__ inline void SparseFlashAttentionMla<SFAT>::UpdateInner(uint32_t &s2End, uint32_t &curS2End, uint32_t s1Idx,
                                                                  bool isEnd)
{
    uint32_t s1BaseSize = 1;
    int64_t s1Offset = s1BaseSize * s1Idx;
    int64_t s2LastToken =
        Min(s1Offset + sfaKernelLoopInfo.nextTokensPerBatch + s1BaseSize, sfaKernelLoopInfo.curActualSeqLenOri);
    s2LastToken = Min(sfaKernelConstInfo.sparseBlockSize * sfaKernelConstInfo.sparseBlockCount, s2LastToken);
    curS2End = (s2LastToken + sfaKernelConstInfo.s2BaseSize - 1) / sfaKernelConstInfo.s2BaseSize;
    sfaKernelLoopInfo.s2LoopTimes = isEnd ? sfaKernelConstInfo.s2End + 1 : curS2End;
}

template <typename SFAT>
__aicore__ inline void SparseFlashAttentionMla<SFAT>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *sparseIndices,
    __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *blockTable,
    __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope, __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxMax,
    __gm__ uint8_t *softmaxSum, __gm__ uint8_t *workspace, const SparseFlashAttentionTilingDataMla *__restrict tiling,
    __gm__ uint8_t *gmTiling, TPipe *tPipe)
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

    InitTilingData();
    InitActualSeqLen(actualSeqLengthsQ, actualSeqLengths);

    // 初始化计算参数
    InitCalcParamsEach();
    pipe = tPipe;
    keyPtr = key;
    valuePtr = value;

    // init global buffer
    queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
    keyGm.SetGlobalBuffer((__gm__ KV_T *)keyPtr);
    valueGm.SetGlobalBuffer((__gm__ KV_T *)valuePtr);
    qRopeGm.SetGlobalBuffer((__gm__ Q_ROPE_T *)queryRope);
    kRopeGm.SetGlobalBuffer((__gm__ K_ROPE_T *)keyRope);

    attentionOutGm.SetGlobalBuffer((__gm__ OUT_T *)attentionOut);
    softmaxMaxGm.SetGlobalBuffer((__gm__ T *)softmaxMax);
    softmaxSumGm.SetGlobalBuffer((__gm__ T *)softmaxSum);
    if ASCEND_IS_AIV {
        if (sfaKernelConstInfo.needInit && LAYOUT_T != SFA_LAYOUT::TND) {
            InitOutputSingleCore();
        }
    }

    if constexpr (PAGE_ATTENTION) {
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
    }
    topKGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);

    // workspace 内存排布
    // |Q--|mm1ResGm(存S)|vec1ResGm(存A1,A2)|mm2ResGm(存O)|vec2ResGm
    // |Core0_Q1-Core0_Q2-Core1_Q1-Core1_Q2....Core32_Q1-Core32_Q2|Core0_mmRes
    uint64_t offset = 0;
    mm1ResGm.SetGlobalBuffer(
        (__gm__ MM1_OUT_T *)(workspace + offset +
                             aiCoreIdx * dbWorkspaceRatio * sfaKernelConstInfo.mmResUbSize * sizeof(MM1_OUT_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * sfaKernelConstInfo.mmResUbSize * sizeof(MM1_OUT_T);

    vec1ResGm.SetGlobalBuffer(
        (__gm__ KV_T *)(workspace + offset +
                        aiCoreIdx * dbWorkspaceRatio * sfaKernelConstInfo.mmResUbSize * sizeof(KV_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * sfaKernelConstInfo.mmResUbSize * sizeof(KV_T);

    mm2ResGm.SetGlobalBuffer(
        (__gm__ MM2_OUT_T *)(workspace + offset +
                             aiCoreIdx * dbWorkspaceRatio * sfaKernelConstInfo.bmm2ResUbSize * sizeof(MM2_OUT_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * sfaKernelConstInfo.bmm2ResUbSize * sizeof(MM2_OUT_T);
    mm2ResInt32Gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(mm2ResGm.GetPhyAddr(0)));

    if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
        // s2  d+rope bufNum
        kvMergeGm_.SetGlobalBuffer((__gm__ KV_T *)(workspace + offset + aiCoreIdx * 512 * 576 * 4 * sizeof(KV_T)));
        offset += GetBlockNum() * 512 * 576 * 4 * sizeof(KV_T);

        kvValidSizeGm_.SetGlobalBuffer(
            (__gm__ int32_t *)(workspace + offset + (aiCoreIdx * 2) * 128 * 4 * sizeof(int32_t)));
    }

    if constexpr (FLASH_DECODE) {
        accumOutGm.SetGlobalBuffer((__gm__ float *)(workspace + offset));
        offset = offset + tilingData->splitKVParams.accumOutSize * sizeof(float);
        lseSumFdGm.SetGlobalBuffer((__gm__ float *)(workspace + offset));
        lseMaxFdGm.SetGlobalBuffer((__gm__ float *)(workspace + offset) + tilingData->splitKVParams.logSumExpSize / 2);
        offset = offset + tilingData->splitKVParams.logSumExpSize * sizeof(float);
    }

    if ASCEND_IS_AIV {
        sfaKernelVectorService.InitParams(sfaKernelConstInfo, tilingData);
        sfaKernelVectorService.InitMm2ResInt32GmGlobalTensor(mm2ResInt32Gm);
        if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
            sfaKernelVectorService.InitVec0GlobalTensor(kvValidSizeGm_, kvMergeGm_, kRopeGm, keyGm, blockTableGm);
        }
        sfaKernelVectorService.InitVec1GlobalTensor(mm1ResGm, vec1ResGm, actualSeqLengthsQGm, actualSeqLengthsKVGm,
                                                    lseMaxFdGm, lseSumFdGm, topKGm, softmaxMaxGm, softmaxSumGm);
        sfaKernelVectorService.InitVec2GlobalTensor(accumOutGm, vec2ResGm, mm2ResGm, attentionOutGm);
    }

    if ASCEND_IS_AIC {
        matmulService.InitParams(sfaKernelConstInfo);
        matmulService.InitMm1GlobalTensor(queryGm, qRopeGm, keyGm, kRopeGm, mm1ResGm);
        matmulService.InitMm2GlobalTensor(vec1ResGm, valueGm, mm2ResGm, attentionOutGm);
        matmulService.InitPageAttentionInfo(kvMergeGm_, blockTableGm, topKGm, sfaKernelConstInfo.kvCacheBlockSize,
                                            sfaKernelConstInfo.maxBlockNumPerBatch);
    }
    // 要在InitParams之后执行
    if (pipe != nullptr) {
        InitBuffers();
    }
}

template <typename SFAT>
__aicore__ inline void SparseFlashAttentionMla<SFAT>::InitCalcParamsEach()
{
    // 计算总的基本块
    uint32_t actBatchS2 = 1;
    uint32_t totalBaseNum = 0;
    uint32_t s1GBaseSize = sfaKernelConstInfo.gSize;
    uint32_t coreNum = GetBlockNum();
    uint32_t currCoreIdx = aiCoreIdx;
    uint32_t actBatchS1 = 1;
    for (uint32_t bIdx = 0; bIdx < sfaKernelConstInfo.batchSize; bIdx++) {
        uint32_t actBatchS1 = GetBalanceActualSeqLengths(actualSeqLengthsQGm, bIdx);
        if (actBatchS1 < sfaKernelConstInfo.qSeqSize) {
            sfaKernelConstInfo.needInit = true;
        }
        totalBaseNum += actBatchS1 * actBatchS2;
    }
    uint32_t avgBaseNum = 1;
    if (totalBaseNum > coreNum) {
        avgBaseNum = (totalBaseNum + coreNum - 1) / coreNum;
    } else {
        usedCoreNum = totalBaseNum;
    }
    if (aiCoreIdx >= usedCoreNum) {
        return;
    }
    // 计算当前核的基本块
    uint32_t accumBaseNum = 0; // 当前累积的基本块数
    uint32_t targetBaseNum = 0;
    uint32_t lastValidBIdx = 0;
    uint32_t lastValidactBatchS1 = 0;
    bool setStart = false;
    targetBaseNum = (currCoreIdx + 1) * avgBaseNum; // 计算当前的目标权重
    uint32_t targetStartBaseNum = targetBaseNum - avgBaseNum;
    for (uint32_t bN2Idx = 0; bN2Idx < sfaKernelConstInfo.batchSize * sfaKernelConstInfo.kvHeadNum; bN2Idx++) {
        uint32_t bIdx = bN2Idx / sfaKernelConstInfo.kvHeadNum;
        actBatchS1 = GetBalanceActualSeqLengths(actualSeqLengthsQGm, bIdx);
        for (uint32_t s1GIdx = 0; s1GIdx < actBatchS1; s1GIdx++) {
            accumBaseNum += 1;
            if (!setStart && accumBaseNum >= targetStartBaseNum) {
                sfaKernelConstInfo.bN2Start = bN2Idx;
                sfaKernelConstInfo.gS1Start = s1GIdx;
                setStart = true;
            }
            if (accumBaseNum >= targetBaseNum) {
                // 更新当前核的End分核信息
                sfaKernelConstInfo.bN2End = bN2Idx;
                sfaKernelConstInfo.gS1End = s1GIdx;
                sfaKernelConstInfo.s2End = 0;
                sfaKernelConstInfo.coreStartKVSplitPos = 0;
                if (aiCoreIdx != 0) {
                    GetAxisStartIdx(sfaKernelConstInfo.bN2Start, sfaKernelConstInfo.gS1Start, 0);
                }
                return;
            }
        }
        if ((actBatchS1 > 0) && (actBatchS2 > 0)) {
            lastValidBIdx = bIdx;
            lastValidactBatchS1 = actBatchS1;
        }
    }
    if (!setStart) {
        sfaKernelConstInfo.bN2Start = lastValidBIdx;
        sfaKernelConstInfo.gS1Start = lastValidactBatchS1 - 1;
    }
    if (accumBaseNum < targetBaseNum) {
        // 更新最后一个核的End分核信息
        sfaKernelConstInfo.coreStartKVSplitPos = 0;
        sfaKernelConstInfo.bN2End = lastValidBIdx;
        sfaKernelConstInfo.gS1End = lastValidactBatchS1 - 1;
        sfaKernelConstInfo.s2End = 0;
        if (aiCoreIdx != 0) {
            GetAxisStartIdx(sfaKernelConstInfo.bN2Start, sfaKernelConstInfo.gS1Start, 0);
        }
        return;
    }
}

template <typename SFAT>
__aicore__ inline void SparseFlashAttentionMla<SFAT>::Bmm2DataCopyOut(uint64_t attenOutOffset,
                                                                      LocalTensor<OUT_T> &attenOutUb, uint32_t startRow,
                                                                      uint32_t dealRowCount, uint32_t columnCount,
                                                                      uint32_t actualColumnCount)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(OUT_T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount) / (SFAVectorService<SFAT>::BYTE_BLOCK / sizeof(OUT_T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(attentionOutGm[attenOutOffset + (mSizeVStart + startRow) * actualColumnCount], attenOutUb,
                dataCopyParams);
}

template <typename SFAT>
__aicore__ inline void SparseFlashAttentionMla<SFAT>::CalcParams(uint32_t loop, uint64_t s2Start, uint32_t s2LoopIdx,
                                                                 RunInfo &sfaKernelRunInfo)
{
    sfaKernelRunInfo.bIdx = sfaKernelLoopInfo.bIdx;
    sfaKernelRunInfo.gS1Idx = sfaKernelLoopInfo.gS1Idx;
    sfaKernelRunInfo.loop = loop;
    sfaKernelRunInfo.s2Idx = s2LoopIdx;
    sfaKernelRunInfo.curSInnerLoopTimes = sfaKernelLoopInfo.s2LoopTimes;

    sfaKernelRunInfo.tndIsS2SplitCore = sfaKernelLoopInfo.tndIsS2SplitCore;
    sfaKernelRunInfo.tndCoreStartKVSplitPos = sfaKernelLoopInfo.tndCoreStartKVSplitPos;
    sfaKernelRunInfo.isBmm2Output = false;

    sfaKernelRunInfo.actS1Size = sfaKernelLoopInfo.actS1Size;

    sfaKernelRunInfo.actMBaseSize = sfaKernelConstInfo.mBaseSize;
    uint32_t remainedGS1Size = sfaKernelLoopInfo.actS1Size * sfaKernelConstInfo.gSize - sfaKernelLoopInfo.gS1Idx;
    if (remainedGS1Size <= sfaKernelConstInfo.mBaseSize && remainedGS1Size > 0) {
        sfaKernelRunInfo.actMBaseSize = sfaKernelLoopInfo.mBasicSizeTail;
    }

    sfaKernelRunInfo.isValid = s2LoopIdx < sfaKernelLoopInfo.s2LoopTimes;

    if ASCEND_IS_AIV {
        sfaKernelRunInfo.mSize = sfaKernelRunInfo.actMBaseSize;
        sfaKernelRunInfo.mSizeV = (sfaKernelRunInfo.mSize <= 16) ? sfaKernelRunInfo.mSize :
                                                                   (((sfaKernelRunInfo.mSize + 15) / 16 + 1) / 2 * 16);
        sfaKernelRunInfo.mSizeVStart = 0;
        if (tmpBlockIdx % 2 == 1) {
            sfaKernelRunInfo.mSizeVStart = sfaKernelRunInfo.mSizeV;
            sfaKernelRunInfo.mSizeV = sfaKernelRunInfo.mSize - sfaKernelRunInfo.mSizeV;
        }
    }

    sfaKernelRunInfo.isChangeBatch = false;

    sfaKernelRunInfo.isFirstSInnerLoop = s2LoopIdx == s2Start;
    if (sfaKernelRunInfo.isFirstSInnerLoop) {
        sfaKernelLoopInfo.bn2IdxInCurCore++;
    }
    sfaKernelRunInfo.isLastS2Loop = s2LoopIdx == sfaKernelLoopInfo.s2LoopTimes - 1;
    sfaKernelRunInfo.bn2IdxInCurCore = sfaKernelLoopInfo.bn2IdxInCurCore - 1;
    uint64_t actualSeqQPrefixSum;
    if constexpr (LAYOUT_T == SFA_LAYOUT::TND) {
        actualSeqQPrefixSum =
            (sfaKernelRunInfo.bIdx <= 0) ? 0 : actualSeqLengthsQGm.GetValue(sfaKernelRunInfo.bIdx - 1);
    } else {
        actualSeqQPrefixSum = (sfaKernelRunInfo.bIdx <= 0) ? 0 : sfaKernelRunInfo.bIdx * sfaKernelConstInfo.qSeqSize;
    }
    sfaKernelRunInfo.tndBIdxOffsetForQ = actualSeqQPrefixSum * sfaKernelConstInfo.qHeadNum * headDim;

    uint64_t actualSeqKVPrefixSum;
    if constexpr (KV_LAYOUT_T == SFA_LAYOUT::TND) {
        actualSeqKVPrefixSum =
            (sfaKernelRunInfo.bIdx <= 0) ? 0 : actualSeqLengthsKVGm.GetValue(sfaKernelRunInfo.bIdx - 1);
    } else {
        actualSeqKVPrefixSum = (sfaKernelRunInfo.bIdx <= 0) ? 0 : sfaKernelRunInfo.bIdx * sfaKernelConstInfo.kvSeqSize;
    }
    sfaKernelRunInfo.tndBIdxOffsetForKV = actualSeqKVPrefixSum * sfaKernelConstInfo.kvHeadNum * headDim;

    if (sfaKernelRunInfo.isFirstSInnerLoop) {
        uint64_t tndBIdxRopeOffsetForQ = actualSeqQPrefixSum * sfaKernelConstInfo.qHeadNum * headDimRope;
        tensorACoreOffset = sfaKernelRunInfo.tndBIdxOffsetForQ + sfaKernelRunInfo.gS1Idx * headDim;
        tensorARopeCoreOffset = tndBIdxRopeOffsetForQ + sfaKernelRunInfo.gS1Idx * headDimRope;

        uint64_t tndBIdxRopeOffsetForK = actualSeqKVPrefixSum * sfaKernelConstInfo.kvHeadNum * headDimRope;
        tensorBCoreOffset = sfaKernelRunInfo.tndBIdxOffsetForKV + sfaKernelRunInfo.n2Idx * headDim;
        tensorBRopeCoreOffset = tndBIdxRopeOffsetForK + sfaKernelRunInfo.n2Idx * headDimRope;
        if (sfaKernelConstInfo.sparseMode == 3) {
            threshold = static_cast<int64_t>(sfaKernelLoopInfo.nextTokensPerBatch) +
                        sfaKernelRunInfo.gS1Idx / sfaKernelConstInfo.gSize + 1;
        } else {
            threshold = sfaKernelLoopInfo.curActualSeqLenOri;
        }
        if constexpr (LAYOUT_T == SFA_LAYOUT::BSND) { // B,S1,N2 K
            topKBaseOffset = sfaKernelRunInfo.bIdx * sfaKernelConstInfo.qSeqSize * sfaKernelConstInfo.kvHeadNum *
                                 sfaKernelConstInfo.sparseBlockCount +
                             sfaKernelRunInfo.gS1Idx / sfaKernelConstInfo.gSize * sfaKernelConstInfo.kvHeadNum *
                                 sfaKernelConstInfo.sparseBlockCount +
                             sfaKernelRunInfo.n2Idx * sfaKernelConstInfo.sparseBlockCount;
        } else if (LAYOUT_T == SFA_LAYOUT::TND) { // T N2 K
            topKBaseOffset = sfaKernelRunInfo.tndBIdxOffsetForQ / sfaKernelConstInfo.gSize /
                                 sfaKernelConstInfo.headDim * sfaKernelConstInfo.kvHeadNum *
                                 sfaKernelConstInfo.sparseBlockCount +
                             sfaKernelRunInfo.n2Idx * sfaKernelConstInfo.sparseBlockCount +
                             sfaKernelRunInfo.gS1Idx / sfaKernelConstInfo.gSize * sfaKernelConstInfo.kvHeadNum *
                                 sfaKernelConstInfo.sparseBlockCount;
        } else { // B N2 S1 K
            topKBaseOffset =
                sfaKernelRunInfo.bIdx * sfaKernelConstInfo.kvHeadNum * sfaKernelConstInfo.qSeqSize *
                    sfaKernelConstInfo.sparseBlockCount +
                sfaKernelRunInfo.n2Idx * sfaKernelConstInfo.qSeqSize * sfaKernelConstInfo.sparseBlockCount +
                sfaKernelRunInfo.gS1Idx / sfaKernelConstInfo.gSize * sfaKernelConstInfo.sparseBlockCount;
        }
    }
    sfaKernelRunInfo.threshold = threshold;
    sfaKernelRunInfo.topKBaseOffset = topKBaseOffset;
    sfaKernelRunInfo.tensorAOffset = tensorACoreOffset;
    sfaKernelRunInfo.tensorARopeOffset = tensorARopeCoreOffset;
    sfaKernelRunInfo.attenOutOffset = tensorACoreOffset;
    sfaKernelRunInfo.tensorBOffset = tensorBCoreOffset;
    sfaKernelRunInfo.tensorBRopeOffset = tensorBRopeCoreOffset;

    uint64_t sInnerOffsetDataSize = sfaKernelRunInfo.s2Idx * sfaKernelConstInfo.s2BaseSize;
    sfaKernelRunInfo.s2BatchOffset = s2BatchBaseOffset + sInnerOffsetDataSize;

    sfaKernelRunInfo.curActualSeqLenOri = sfaKernelLoopInfo.curActualSeqLenOri;
    // 计算实际基本块size
    if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
        if (sfaKernelLoopInfo.curActualSeqLen > sInnerOffsetDataSize) {
            sfaKernelRunInfo.actualSingleProcessSInnerSize = sfaKernelLoopInfo.curActualSeqLen - sInnerOffsetDataSize;
            sfaKernelRunInfo.actualSingleProcessSInnerSize =
                sfaKernelRunInfo.actualSingleProcessSInnerSize > sfaKernelConstInfo.s2BaseSize ?
                    sfaKernelConstInfo.s2BaseSize :
                    sfaKernelRunInfo.actualSingleProcessSInnerSize;
        } else {
            sfaKernelRunInfo.actualSingleProcessSInnerSize = 0;
        }
        sfaKernelRunInfo.actualSingleProcessSInnerSizeAlign = SFAAlign(
            (uint32_t)sfaKernelRunInfo.actualSingleProcessSInnerSize, (uint32_t)SFAVectorService<SFAT>::BYTE_BLOCK);
    }
}

template <typename SFAT>
__aicore__ inline void SparseFlashAttentionMla<SFAT>::ComputeMm1(const RunInfo &sfaKernelRunInfo)
{
    uint32_t nBufferLoopTimes =
        (sfaKernelRunInfo.actMBaseSize + sfaKernelConstInfo.nBufferMBaseSize - 1) / sfaKernelConstInfo.nBufferMBaseSize;
    uint32_t nBufferTail = sfaKernelRunInfo.actMBaseSize - (nBufferLoopTimes - 1) * sfaKernelConstInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo sfaKernelSplitInfo;
        sfaKernelSplitInfo.nBufferStartM = i * sfaKernelConstInfo.nBufferMBaseSize;
        sfaKernelSplitInfo.nBufferDealM =
            (i + 1 != nBufferLoopTimes) ? sfaKernelConstInfo.nBufferMBaseSize : nBufferTail;
        matmulService.ComputeMm1(sfaKernelRunInfo, sfaKernelSplitInfo);
        CrossCoreSetFlag<ConstInfo::SFA_SYNC_MODE2, PIPE_FIX>(sfaKernelConstInfo.syncC1V1);
    }
}

template <typename SFAT>
__aicore__ inline void SparseFlashAttentionMla<SFAT>::ComputeMm2(const RunInfo &sfaKernelRunInfo)
{
    uint32_t nBufferLoopTimes =
        (sfaKernelRunInfo.actMBaseSize + sfaKernelConstInfo.nBufferMBaseSize - 1) / sfaKernelConstInfo.nBufferMBaseSize;
    uint32_t nBufferTail = sfaKernelRunInfo.actMBaseSize - (nBufferLoopTimes - 1) * sfaKernelConstInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo sfaKernelSplitInfo;
        sfaKernelSplitInfo.nBufferStartM = i * sfaKernelConstInfo.nBufferMBaseSize;
        sfaKernelSplitInfo.nBufferDealM =
            (i + 1 != nBufferLoopTimes) ? sfaKernelConstInfo.nBufferMBaseSize : nBufferTail;
        CrossCoreWaitFlag(sfaKernelConstInfo.syncV1C2);
        matmulService.ComputeMm2(sfaKernelRunInfo, sfaKernelSplitInfo);
        CrossCoreSetFlag<ConstInfo::SFA_SYNC_MODE2, PIPE_FIX>(sfaKernelConstInfo.syncC2V2);
        CrossCoreSetFlag<ConstInfo::SFA_SYNC_MODE2, PIPE_FIX>(sfaKernelConstInfo.syncC2V1);
    }
}

template <typename SFAT>
__aicore__ inline void SparseFlashAttentionMla<SFAT>::Process()
{
    if (aiCoreIdx < usedCoreNum) {
        if ASCEND_IS_AIV {
            sfaKernelVectorService.AllocEventID();
            sfaKernelVectorService.InitSoftmaxDefaultBuffer();
        } else {
            matmulService.AllocEventID();
        }
        ProcessBalance();

        if ASCEND_IS_AIV {
            sfaKernelVectorService.FreeEventID();
        } else {
            matmulService.FreeEventID();
        }
    }
}

template <typename SFAT>
__aicore__ inline void SparseFlashAttentionMla<SFAT>::GetBN2Idx(uint32_t bN2Idx, uint32_t &bIdx, uint32_t &n2Idx)
{
    bIdx = bN2Idx / kvHeadNum;
    n2Idx = bN2Idx % kvHeadNum;
}

template <typename SFAT>
__aicore__ inline void SparseFlashAttentionMla<SFAT>::ProcessBalance()
{
    RunInfo extraInfo[SFA_PRELOAD_TASK_CACHE_SIZE];
    uint32_t gloop = 0;
    int gS1LoopEnd;
    bool globalLoopStart = true;
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<ConstInfo::SFA_SYNC_MODE2, PIPE_FIX>(sfaKernelConstInfo.syncC2V1);
        if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
            CrossCoreSetFlag<ConstInfo::SFA_SYNC_MODE2, PIPE_MTE2>(3);
            CrossCoreSetFlag<ConstInfo::SFA_SYNC_MODE2, PIPE_MTE2>(3);
            CrossCoreSetFlag<ConstInfo::SFA_SYNC_MODE2, PIPE_MTE2>(3);
            CrossCoreSetFlag<ConstInfo::SFA_SYNC_MODE2, PIPE_MTE2>(3);
        }
    }
    for (uint32_t bN2LoopIdx = sfaKernelConstInfo.bN2Start; bN2LoopIdx <= sfaKernelConstInfo.bN2End; bN2LoopIdx++) {
        GetBN2Idx(bN2LoopIdx, sfaKernelLoopInfo.bIdx, sfaKernelLoopInfo.n2Idx);
        GetActualSeqLen(sfaKernelLoopInfo.bIdx); // 获取actualSeqLength及ActualSeqLengthKV
        GetPreNextTokensLeftUp();
        if (sfaKernelLoopInfo.actS1Size == 0) {
            continue;
        }
        int gS1SplitNum = (sfaKernelLoopInfo.actS1Size * sfaKernelConstInfo.gSize + sfaKernelConstInfo.mBaseSize - 1) /
                          sfaKernelConstInfo.mBaseSize;
        gS1LoopEnd = (bN2LoopIdx == sfaKernelConstInfo.bN2End) ? sfaKernelConstInfo.gS1End : gS1SplitNum - 1;
        for (uint32_t gS1LoopIdx = sfaKernelConstInfo.gS1Start; gS1LoopIdx <= gS1LoopEnd; gS1LoopIdx++) {
            sfaKernelLoopInfo.gS1Idx = gS1LoopIdx * sfaKernelConstInfo.mBaseSize;
            GetSparseActualSeqLen(sfaKernelLoopInfo.bIdx, gS1LoopIdx,
                                  sfaKernelLoopInfo.n2Idx); // TopK值sparse完后的ActualSeqLengthKV
            UpdateInnerLoopCond();

            if (sfaKernelLoopInfo.curActSeqLenIsZero) {
                DealActSeqLenIsZero(sfaKernelLoopInfo.bIdx, gS1LoopIdx, sfaKernelLoopInfo.n2Idx);
            }
            int s2SplitNum = (sfaKernelLoopInfo.curActualSeqLen + sfaKernelConstInfo.s2BaseSize - 1) /
                             sfaKernelConstInfo.s2BaseSize; // S2切分份数
            bool isEnd = (bN2LoopIdx == sfaKernelConstInfo.bN2End) && (gS1LoopIdx == sfaKernelConstInfo.gS1End);
            sfaKernelLoopInfo.s2LoopTimes = s2SplitNum;
            // 分核修改后需要打开
            // 当前s2是否被切，决定了输出是否要写到attenOut上
            sfaKernelLoopInfo.tndIsS2SplitCore =
                ((sfaKernelConstInfo.s2Start == 0) && (sfaKernelLoopInfo.s2LoopTimes == s2SplitNum)) ? false : true;
            sfaKernelLoopInfo.tndCoreStartKVSplitPos = globalLoopStart ? sfaKernelConstInfo.coreStartKVSplitPos : 0;
            uint32_t extraLoop = isEnd ? 2 : 0;

            uint32_t curTopKIdx = 0;
            uint64_t curOffsetInSparseBlock = 0;
            for (int s2LoopIdx = sfaKernelConstInfo.s2Start; s2LoopIdx < (sfaKernelLoopInfo.s2LoopTimes + extraLoop);
                 s2LoopIdx++) {
                // PreloadPipeline loop初始值要求为 PRELOAD_NUM
                PreloadPipeline(gloop, sfaKernelConstInfo.s2Start, s2LoopIdx, extraInfo, curTopKIdx,
                                curOffsetInSparseBlock);
                ++gloop;
            }
            globalLoopStart = false;
            sfaKernelConstInfo.s2Start = 0;
        }
        sfaKernelConstInfo.gS1Start = 0;
    }
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag(sfaKernelConstInfo.syncC2V1);
        if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
            CrossCoreWaitFlag(3);
            CrossCoreWaitFlag(3);
            CrossCoreWaitFlag(3);
            CrossCoreWaitFlag(3);
        }
    }
}

template <typename SFAT>
__aicore__ inline void SparseFlashAttentionMla<SFAT>::PreloadPipeline(uint32_t loop, uint64_t s2Start,
                                                                      uint64_t s2LoopIdx,
                                                                      RunInfo extraInfo[SFA_PRELOAD_TASK_CACHE_SIZE],
                                                                      uint32_t &curTopKIdx,
                                                                      uint64_t &curOffsetInSparseBlock)
{
    RunInfo &extraInfo0 = extraInfo[loop % SFA_PRELOAD_TASK_CACHE_SIZE];       // 本轮任务
    RunInfo &extraInfo2 = extraInfo[(loop + 2) % SFA_PRELOAD_TASK_CACHE_SIZE]; // 上一轮任务
    RunInfo &extraInfo1 = extraInfo[(loop + 1) % SFA_PRELOAD_TASK_CACHE_SIZE]; // 上两轮任务

    CalcParams(loop, s2Start, s2LoopIdx, extraInfo0);
    CalcSinnerTopKBegin(extraInfo0, curTopKIdx, curOffsetInSparseBlock);

    if (extraInfo0.isValid) {
        if ASCEND_IS_AIC {
            if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
                CrossCoreWaitFlag(sfaKernelConstInfo.syncV0C1);
            }
            ComputeMm1(extraInfo0);
        } else {
            if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
                CrossCoreWaitFlag(3);
                sfaKernelVectorService.MergeKv(extraInfo0);
                CrossCoreSetFlag<ConstInfo::SFA_SYNC_MODE2, PIPE_MTE3>(sfaKernelConstInfo.syncV0C1);
            }
        }
    }
    if (extraInfo2.isValid) {
        if ASCEND_IS_AIV {
            sfaKernelVectorService.ProcessVec1L(extraInfo2);
        }
        if ASCEND_IS_AIC {
            ComputeMm2(extraInfo2);
            if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
                CrossCoreSetFlag<ConstInfo::SFA_SYNC_MODE2, PIPE_MTE2>(3);
            }
        }
    }
    if (extraInfo1.isValid) {
        if ASCEND_IS_AIV {
            sfaKernelVectorService.ProcessVec2L(extraInfo1);
        }
        extraInfo1.isValid = false;
    }
}

template <typename SFAT>
__aicore__ inline uint64_t SparseFlashAttentionMla<SFAT>::GetBalanceActualSeqLengths(
    GlobalTensor<int32_t> &actualSeqLengths, uint32_t bIdx)
{
    if constexpr (LAYOUT_T == SFA_LAYOUT::TND) {
        if (bIdx > 0) {
            return actualSeqLengths.GetValue(bIdx) - actualSeqLengths.GetValue(bIdx - 1);
        } else if (bIdx == 0) {
            return actualSeqLengths.GetValue(0);
        } else {
            return 0;
        }
    } else {
        if (sfaKernelConstInfo.actualLenDimsQ == 0) {
            return sfaKernelConstInfo.qSeqSize;
        } else if (sfaKernelConstInfo.actualLenDimsQ == 1) {
            return actualSeqLengths.GetValue(0);
        } else {
            return actualSeqLengths.GetValue(bIdx);
        }
    }
}

template <typename SFAT>
__aicore__ inline void SparseFlashAttentionMla<SFAT>::GetAxisStartIdx(uint32_t bN2EndPrev, uint32_t s1GEndPrev,
                                                                      uint32_t s2EndPrev)
{
    uint32_t bEndPrev = bN2EndPrev / kvHeadNum;
    uint32_t actualSeqQPrev = GetBalanceActualSeqLengths(actualSeqLengthsQGm, bEndPrev);
    uint32_t s1GPrevBaseNum =
        (actualSeqQPrev * sfaKernelConstInfo.gSize + sfaKernelConstInfo.mBaseSize - 1) / sfaKernelConstInfo.mBaseSize;
    sfaKernelConstInfo.bN2Start = bN2EndPrev;
    sfaKernelConstInfo.gS1Start = s1GEndPrev;

    sfaKernelConstInfo.s2Start = 0;
    if (s1GEndPrev >= s1GPrevBaseNum - 1) { // 上个核把S1G处理完了
        sfaKernelConstInfo.gS1Start = 0;
        sfaKernelConstInfo.bN2Start++;
    } else {
        sfaKernelConstInfo.gS1Start++;
    }
}

template <typename SFAT>
__aicore__ inline void SparseFlashAttentionMla<SFAT>::CalcSinnerTopKBegin(RunInfo &sfaKernelRunInfo,
                                                                          uint32_t &curTopKIdx,
                                                                          uint64_t &curOffsetInSparseBlock)

{
    if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
        return;
    }

    uint64_t thresholdSparseCount =
        (sfaKernelRunInfo.threshold + sfaKernelConstInfo.sparseBlockSize - 1) / sfaKernelConstInfo.sparseBlockSize;
    uint64_t validCount = (sfaKernelConstInfo.sparseBlockCount > thresholdSparseCount) ?
                              thresholdSparseCount :
                              sfaKernelConstInfo.sparseBlockCount;

    int32_t sparseIndices = topKGm.GetValue(sfaKernelRunInfo.topKBaseOffset + curTopKIdx);
    if (sparseIndices == -1 || curTopKIdx == validCount) {
        sfaKernelRunInfo.actualSingleProcessSInnerSize = 0;
        sfaKernelRunInfo.actualSingleProcessSInnerSizeAlign = 0;
        sfaKernelLoopInfo.s2BasicSizeTail = 0;
        if (curTopKIdx == 0) {
            DealActSeqLenIsZero(sfaKernelRunInfo.bIdx, sfaKernelRunInfo.gS1Idx / sfaKernelConstInfo.gSize,
                                sfaKernelLoopInfo.n2Idx);
        }
        return;
    }

    uint32_t sparseLen = 0;
    uint64_t blockBegin = sparseIndices * sfaKernelConstInfo.sparseBlockSize;
    uint64_t blockEnd = (blockBegin + sfaKernelConstInfo.sparseBlockSize > sfaKernelRunInfo.threshold) ?
                            sfaKernelRunInfo.threshold :
                            blockBegin + sfaKernelConstInfo.sparseBlockSize;
    int32_t blockLen = blockEnd - blockBegin;
    sparseLen += (blockLen > static_cast<int32_t>(curOffsetInSparseBlock)) ? blockLen - curOffsetInSparseBlock : 0;

    bool firstVaildFlag = false;
    if (curTopKIdx > 0) {
        sfaKernelRunInfo.curTopKIdx = curTopKIdx;
        sfaKernelRunInfo.curOffsetInSparseBlock = curOffsetInSparseBlock;
    } else if (curTopKIdx == 0 && sparseLen > 0) {
        sfaKernelRunInfo.curTopKIdx = curTopKIdx;
        sfaKernelRunInfo.curOffsetInSparseBlock = 0;
        firstVaildFlag = true;
    }

    for (uint64_t topkIdx = curTopKIdx + 1; topkIdx < validCount; topkIdx++) {
        int32_t sparseIndices = topKGm.GetValue(sfaKernelRunInfo.topKBaseOffset + topkIdx);
        if (sparseIndices == -1) {
            curTopKIdx = topkIdx;
            curOffsetInSparseBlock = 0;
            break;
        }
        uint64_t blockBegin = sparseIndices * sfaKernelConstInfo.sparseBlockSize;
        if (blockBegin >= sfaKernelRunInfo.threshold) {
            continue;
        }
        if (firstVaildFlag == false && curTopKIdx == 0) {
            sfaKernelRunInfo.curTopKIdx = topkIdx;
            sfaKernelRunInfo.curOffsetInSparseBlock = 0;
            firstVaildFlag = true;
        }
        uint64_t blockEnd = (blockBegin + sfaKernelConstInfo.sparseBlockSize > sfaKernelRunInfo.threshold) ?
                                sfaKernelRunInfo.threshold :
                                blockBegin + sfaKernelConstInfo.sparseBlockSize;
        uint64_t blockLen = blockEnd - blockBegin;
        sparseLen += blockLen;
        if (sparseLen >= sfaKernelConstInfo.s2BaseSize) {
            curTopKIdx = topkIdx;
            curOffsetInSparseBlock = blockLen - (sparseLen - sfaKernelConstInfo.s2BaseSize);
            sparseLen = sfaKernelConstInfo.s2BaseSize;
            break;
        }

        if (topkIdx == validCount - 1) {
            curTopKIdx = validCount;
            curOffsetInSparseBlock = 0;
        }
    }

    sfaKernelRunInfo.actualSingleProcessSInnerSize = sparseLen;
    sfaKernelRunInfo.actualSingleProcessSInnerSizeAlign = SFAAlign(
        (uint32_t)sfaKernelRunInfo.actualSingleProcessSInnerSize, (uint32_t)SFAVectorService<SFAT>::BYTE_BLOCK);
    sfaKernelLoopInfo.s2BasicSizeTail = (sparseLen == sfaKernelConstInfo.s2BaseSize) ? 0 : sparseLen;
    if (curTopKIdx == 0 && sparseLen == 0) {
        DealActSeqLenIsZero(sfaKernelRunInfo.bIdx, sfaKernelRunInfo.gS1Idx / sfaKernelConstInfo.gSize,
                            sfaKernelLoopInfo.n2Idx);
    }
}

#endif // SPARSE_FLASH_ATTENTION_KERNEL_MLA_H
