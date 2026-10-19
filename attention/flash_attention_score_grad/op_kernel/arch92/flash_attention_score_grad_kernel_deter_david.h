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
 * \file flash_attention_score_grad_kernel_deter_david.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_KERNEL_DETER_DAVID_H
#define FLASH_ATTENTION_SCORE_GRAD_KERNEL_DETER_DAVID_H

#include "flash_attention_score_grad_kernel_base_david.h"
#include "flash_attention_score_grad_common_david.h"
#include "flash_attention_score_grad_tiling_data_david.h"
#include "deter_david.h"

namespace FagBaseApi {

template <typename CubeBlockType, typename VecBlockType>

class FlashAttentionScoreGradKernelDeter
    : public FlashAttentionScoreGradKernelBase<FlashAttentionScoreGradKernelDeter<CubeBlockType, VecBlockType>,
                                               CubeBlockType, VecBlockType> {
public:
    ARGS_TRAITS;
    using BaseClass = FlashAttentionScoreGradKernelBase<FlashAttentionScoreGradKernelDeter<CubeBlockType, VecBlockType>,
                                                        CubeBlockType, VecBlockType>;
    __aicore__ inline void Init(GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR query, GM_ADDR pseShift,
                                GM_ADDR dropMask, GM_ADDR attenMask, GM_ADDR y, GM_ADDR softmaxMax, GM_ADDR softmaxSum,
                                GM_ADDR prefixN, GM_ADDR actualSeqQlen, GM_ADDR actualSeqKvlen, GM_ADDR deqScaleQ,
                                GM_ADDR deqScaleK, GM_ADDR deqScaleV, GM_ADDR deqScaleDy, GM_ADDR deqScaleO,
                                GM_ADDR queryRope, GM_ADDR keyRope, GM_ADDR sink, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
                                GM_ADDR dpse, GM_ADDR dsink, GM_ADDR workspace, FagTilingType ordTilingData,
                                TPipe* pipeIn);
    __aicore__ inline void SetUniqueRunInfo(FagRunInfo& runInfo);
    __aicore__ inline void SetUniqueConstInfo(FagConstInfo& constInfo);
    __aicore__ inline void SetRunInfoDeterForTND(FagRunInfo& runInfo, int64_t taskId, int64_t index,
                                                 CoordinateInfo& coordinateInfo, int64_t nextIndex);
    __aicore__ inline int64_t CalDeterMaxLoopNum();
    __aicore__ inline void CalDeterIndex(uint32_t roundId, uint32_t maxLoopNum, int64_t& nextValidRoundId,
                                         int64_t& nextValidIndex, int64_t taskId, CoordinateInfo& coordinateInfo,
                                         FagRunInfo& runInfo);
    __aicore__ inline int64_t CalDenseDeterIndex(uint32_t roundId, CoordinateInfo& coordinateInfo);
    __aicore__ inline int64_t CalCausalDeterIndex(uint32_t roundId, CoordinateInfo& coordinateInfo);
    __aicore__ inline int64_t CalBandDeterIndex(uint32_t roundId, CoordinateInfo& coordinateInfo);
    __aicore__ inline bool IsValidDeterForTnd(FagRunInfo& runInfo, int64_t index, CoordinateInfo& coordinateInfo);
    __aicore__ inline void Process();
    __aicore__ inline void Process_NEW_DETER();
    __aicore__ inline void Process_OLD_DETER();
    __aicore__ inline void GetIsNeedDeter(int64_t computeLoopIdx);
    __aicore__ inline void DeterSync(int64_t loopIdx);
    __aicore__ inline int64_t SpecialS2Index(int64_t dkvGmOffset);

protected:
    GlobalTensor<float> deterGm;
    uint64_t dkvWorkSpaceOffet{0};
    uint64_t dAlign16 = 0;
    uint64_t dvAlign16 = 0;
    int64_t deterGmOffset = 0;
    typename std::conditional<IS_DETER_NEW(DETER_SPARSE_TYPE), CoordinateInfo[2], std::nullptr_t>::type
        coordinateInfos{};
    typename std::conditional<DETER_SPARSE_TYPE == DETER_BAND, BandInfo, std::nullptr_t>::type bandInfo;
    bool isMm3NeedWait = false;
    typename std::conditional<IS_DETER_OLD(DETER_SPARSE_TYPE), bool[2], std::nullptr_t>::type dqIsNeedDeter{};
    typename std::conditional<IS_DETER_OLD(DETER_SPARSE_TYPE), bool[2], std::nullptr_t>::type dkDvIsNeedDeter{};
    typename std::conditional<IS_DETER_OLD(DETER_SPARSE_TYPE), bool, std::nullptr_t>::type isFirstDeter{};
    int64_t deterLoopMax = 0;
    static constexpr uint32_t BITS_EACH_UINT64 = 64;
    static constexpr uint32_t MAX_BITS_IN_TILING = 32 * BITS_EACH_UINT64;
    static constexpr uint64_t SYNC_MODE = 4;
};

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelDeter<CubeBlockType, VecBlockType>::Init(
    GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR query, GM_ADDR pseShift, GM_ADDR dropMask, GM_ADDR attenMask,
    GM_ADDR y, GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR prefixN, GM_ADDR actualSeqQlen, GM_ADDR actualSeqKvlen,
    GM_ADDR deqScaleQ, GM_ADDR deqScaleK, GM_ADDR deqScaleV, GM_ADDR deqScaleDy, GM_ADDR deqScaleO, GM_ADDR queryRope,
    GM_ADDR keyRope, GM_ADDR sink, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dpse, GM_ADDR dsink, GM_ADDR workspace,
    FagTilingType ordTilingData, TPipe* pipeIn)
{
    BaseClass::Init(key, value, dy, query, pseShift, dropMask, attenMask, y, softmaxMax, softmaxSum, prefixN,
                    actualSeqQlen, actualSeqKvlen, deqScaleQ, deqScaleK, deqScaleV, deqScaleDy, deqScaleO, queryRope,
                    keyRope, sink, dq, dk, dv, dpse, dsink, workspace, ordTilingData, pipeIn);

    dkvWorkSpaceOffet = this->cBlockIdx * this->CUBE_BASEN * this->HEAD_DIM_ALIGN;
    dAlign16 = AlignTo16(this->constInfo.commonConstInfo.dSize);
    dvAlign16 = AlignTo16(this->constInfo.commonConstInfo.dSizeV);

    deterGm.SetGlobalBuffer((__gm__ float*)workspace + this->tilingData->postTilingData.deterGmOffset / sizeof(T2));
    deterGmOffset = this->cBlockIdx * this->CUBE_BASEN * this->HEAD_DIM_ALIGN * NUM_TWO;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelDeter<CubeBlockType, VecBlockType>::SetUniqueRunInfo(
    FagRunInfo& runInfo)
{}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelDeter<CubeBlockType, VecBlockType>::SetUniqueConstInfo(
    FagConstInfo& constInfo)
{}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t FlashAttentionScoreGradKernelDeter<CubeBlockType, VecBlockType>::CalDeterMaxLoopNum()
{
    int64_t b = this->constInfo.bSize * this->constInfo.n2Size;
    int64_t m = this->constInfo.s1Outer;
    int64_t n = this->constInfo.s2Outer;
    int64_t k = static_cast<int64_t>(this->tilingData->s1s2BNGS1S2BaseParams.coreNum);
    int64_t loopMax = 0;
    if constexpr (BaseClass::IS_TND) {
        return this->tilingData->s1s2BNGS1S2SplitCoreParams.deterMaxRound;
    }

    InitCoordinateInfo(this->constInfo.s1Outer, this->constInfo.s2Outer, 0, 0, this->coordinateInfos[0]);
    InitCoordinateInfo(this->constInfo.s1Outer, this->constInfo.s2Outer, 0, 0, this->coordinateInfos[1]);
    if constexpr (BaseClass::DETER_SPARSE_TYPE == DETER_CAUSAL) {
        return this->tilingData->s1s2BNGS1S2SplitCoreParams.deterMaxRound;
    }

    if constexpr (BaseClass::DETER_SPARSE_TYPE == DETER_DENSE) {
        if constexpr (BaseClass::IS_N_EQUAL) {
            return Ceil<int64_t>(n * b, Min(k, m * b)) * m;
        } else {
            return Max(Max(Ceil<int64_t>(b * n * this->constInfo.commonConstInfo.gSize,
                                         (Min(Min(k, b * this->constInfo.commonConstInfo.gSize * m), b * n))),
                           Ceil<int64_t>(n, m)),
                       this->constInfo.commonConstInfo.gSize) *
                   m;
        }
    }

    if constexpr (BaseClass::DETER_SPARSE_TYPE == DETER_BAND) {
        int64_t s1Outer = this->constInfo.s1Outer;
        int64_t s2Outer = this->constInfo.s2Outer;
        if constexpr (BaseClass::DETER_TILING_SPLIT_MODE == 2) {
            s2Outer = Ceil<int64_t>(s2Outer, 2);
            n = s2Outer;
        } else if constexpr (BaseClass::DETER_TILING_SPLIT_MODE == 1) {
            m = Ceil<int64_t>(m, 2);
            s1Outer = m;
        }

        int64_t p = Ceil<int64_t>(this->constInfo.s1Token, this->DETER_CUBE_BASEM) + 1;
        int64_t q = Ceil<int64_t>(this->constInfo.s2Token, this->DETER_CUBE_BASEN) + 1;
        p = p > s1Outer ? s1Outer : p;
        q = q > s2Outer ? s2Outer : q;

        int64_t actualM, actualN, actualP, actualQ;
        int64_t mOffset, nOffset;
        if (p < 0) {
            actualM = m;
            actualN = n + p;
            actualP = 1;
            actualQ = p + q;
            mOffset = 0;
            nOffset = -p;
        } else if (q < 0) {
            actualM = m + q;
            actualN = n;
            actualP = p + q;
            actualQ = 1;
            mOffset = -q;
            nOffset = 0;
        } else {
            actualM = m;
            actualN = n;
            actualP = p;
            actualQ = q;
            mOffset = 0;
            nOffset = 0;
        }

        if constexpr (BaseClass::IS_N_EQUAL) {
            GenBandInfo(k, actualM, actualN, actualP, actualQ, b, this->bandInfo);
            loopMax = Max(this->tilingData->s1s2BNGS1S2SplitCoreParams.deterMaxRound / 2, this->bandInfo.rm2);
        } else {
            k = Min(Min(k, b * this->constInfo.commonConstInfo.gSize * m), b * n);
            int64_t b2 = b % k;
            GenGQABandInfo(k, actualM, actualN, actualP, actualQ, b, this->constInfo.commonConstInfo.gSize,
                           this->bandInfo);
            loopMax = this->bandInfo.rm + this->bandInfo.rm2;
        }
        if constexpr (BaseClass::DETER_TILING_SPLIT_MODE == 1 || BaseClass::DETER_TILING_SPLIT_MODE == 2) {
            loopMax *= 2;
        }
        InitCoordinateInfo(this->constInfo.s1Outer, this->constInfo.s2Outer, mOffset, nOffset,
                           this->coordinateInfos[0]);
        InitCoordinateInfo(this->constInfo.s1Outer, this->constInfo.s2Outer, mOffset, nOffset,
                           this->coordinateInfos[1]);
        return loopMax;
    }
    return loopMax;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelDeter<CubeBlockType, VecBlockType>::CalDeterIndex(
    uint32_t roundId, uint32_t maxLoopNum, int64_t& nextValidRoundId, int64_t& nextValidIndex, int64_t taskId,
    CoordinateInfo& coordinateInfo, FagRunInfo& runInfo)
{
    coordinateInfo.sparseMode = this->constInfo.sparseMode;
    for (uint32_t currentRoundId = roundId; currentRoundId < maxLoopNum; currentRoundId++) {
        if constexpr (BaseClass::DETER_SPARSE_TYPE == DETER_BAND) {
            nextValidIndex = CalBandDeterIndex(currentRoundId, coordinateInfo);
        } else if constexpr (BaseClass::DETER_SPARSE_TYPE == DETER_CAUSAL) {
            nextValidIndex = CalCausalDeterIndex(currentRoundId, coordinateInfo);
        } else {
            nextValidIndex = CalDenseDeterIndex(currentRoundId, coordinateInfo);
        }

        bool isValidBlock = (nextValidIndex >= 0);
        if constexpr (BaseClass::IS_TND) {
            isValidBlock = isValidBlock && IsValidDeterForTnd(runInfo, nextValidIndex, coordinateInfo);
        } else {
            isValidBlock = isValidBlock && this->IsValidForDeter(runInfo, taskId, nextValidIndex);
        }
        if (isValidBlock) {
            nextValidRoundId = currentRoundId;
            return;
        }
    }
    nextValidIndex = -1;
    nextValidRoundId = maxLoopNum;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelDeter<CubeBlockType, VecBlockType>::SetRunInfoDeterForTND(
    FagRunInfo& runInfo, int64_t taskId, int64_t index, CoordinateInfo& coordinateInfo, int64_t nextIndex)
{
    runInfo.taskPingPong = taskId % 2;
    // runInfo.qDxPingPongIdx = taskId & 1;
    int64_t bIdx = coordinateInfo.batchId;
    int64_t actualS1Len = coordinateInfo.actualS1Len;
    int64_t actualS2Len = coordinateInfo.actualS2Len;
    int64_t s1OuterTmp = coordinateInfo.s1Outer;
    int64_t s1CvTailTmp = actualS1Len - (s1OuterTmp - 1) * this->CUBE_BASEM;
    runInfo.commonRunInfo.boIdx = bIdx;

    if (runInfo.lastBatchIdx != bIdx) {
        int64_t seqQLenPrefix = bIdx == 0 ? 0 : ((__gm__ int64_t*)this->actualSeqQlenAddr)[bIdx - 1];
        int64_t seqKvLenPrefix = bIdx == 0 ? 0 : ((__gm__ int64_t*)this->actualSeqKvlenAddr)[bIdx - 1];
        runInfo.lastBatchTotalS1BOffset = seqQLenPrefix * this->constInfo.commonConstInfo.n2GD;
        runInfo.lastBatchTotalS2BOffset = seqKvLenPrefix * this->constInfo.commonConstInfo.n2D;
        runInfo.lastBatchTotalS1BOffsetForDv = seqQLenPrefix * this->constInfo.commonConstInfo.n2GDv;
        runInfo.lastBatchTotalS2BOffsetForDv = seqKvLenPrefix * this->constInfo.commonConstInfo.n2Dv;
        runInfo.lastBatchTotalS1S2SizeAlign = GetPrefixByBidx<true>(this->actualSeqQlenAddr, this->actualSeqKvlenAddr,
                                                                    this->tilingData->deterParam.deterPrefixAlign, bIdx,
                                                                    this->tilingData->deterParam.deterPrefixStep);
        runInfo.lastBatchTotalS1S2Size = GetPrefixByBidx<false>(this->actualSeqQlenAddr, this->actualSeqKvlenAddr,
                                                                this->tilingData->deterParam.deterPrefix, bIdx,
                                                                this->tilingData->deterParam.deterPrefixStep);
        runInfo.lastBatchTotalS2Size = seqKvLenPrefix;
        if constexpr (IS_ROPE) {
            runInfo.lastBatchTotalS1BRopeOffset = seqQLenPrefix * this->constInfo.ropeConstInfo.n2GDr;
            runInfo.lastBatchTotalS2BRopeOffset = seqKvLenPrefix * this->constInfo.ropeConstInfo.n2Dr;
        }
    }

    runInfo.lastBatchIdx = bIdx;
    runInfo.commonRunInfo.n2oIdx = coordinateInfo.n2Idx;
    runInfo.commonRunInfo.goIdx = coordinateInfo.gIdx;
    runInfo.s2oIdx = coordinateInfo.s2Idx;
    runInfo.commonRunInfo.s1oIdx = coordinateInfo.s1Idx;
    runInfo.commonRunInfo.s1RealSize =
        (runInfo.commonRunInfo.s1oIdx == s1OuterTmp - 1) ? s1CvTailTmp : this->CUBE_BASEM;
    runInfo.commonRunInfo.taskId = taskId;
    runInfo.commonRunInfo.taskIdMod2 = taskId & 1;
    runInfo.commonRunInfo.s2RealSize = runInfo.s2CvEnd - runInfo.s2CvBegin; // 真实s2基本块大小
    runInfo.commonRunInfo.halfS1RealSize = runInfo.commonRunInfo.s1RealSize;
    runInfo.commonRunInfo.firstHalfS1RealSize = runInfo.commonRunInfo.halfS1RealSize;

    runInfo.commonRunInfo.s1SizeThisBatch = actualS1Len;
    runInfo.commonRunInfo.s2SizeThisBatch = actualS2Len;
    runInfo.commonRunInfo.s2SizeAcc = runInfo.lastBatchTotalS2Size;
    runInfo.commonRunInfo.b1SSOffsetAlign = runInfo.lastBatchTotalS1S2SizeAlign;
    runInfo.commonRunInfo.b1SSOffset = runInfo.lastBatchTotalS1S2Size;
    runInfo.commonRunInfo.s2StartIdx = runInfo.s2CvBegin;
    runInfo.commonRunInfo.s2AlignedSize = AlignTo16(runInfo.commonRunInfo.s2RealSize);

    runInfo.isS2IdxNoChange =
        (this->lastS2oCvDimIdx == runInfo.s2oIdx && this->lastBdimIdx == runInfo.commonRunInfo.boIdx &&
         this->lastN2dimIdx == runInfo.commonRunInfo.n2oIdx);
    if (!runInfo.isS2IdxNoChange) {
        this->lastS2oCvDimIdx = runInfo.s2oIdx;
        this->lastBdimIdx = runInfo.commonRunInfo.boIdx;
        this->lastN2dimIdx = runInfo.commonRunInfo.n2oIdx;
    }

    runInfo.dxGmOffset = this->GetDxOffset(runInfo);
    runInfo.valueGmOffset = this->GetValueOffset(runInfo);
    runInfo.queryGmOffset = this->GetQueryOffset(runInfo, true);
    runInfo.keyGmOffset = this->GetKeyOffset(runInfo, true);
    if constexpr (IS_ROPE) {
        runInfo.queryRopeGmOffset = this->GetQueryRopeOffset(runInfo);
        runInfo.keyRopeGmOffset = this->GetKeyRopeOffset(runInfo);
    }
    if (unlikely(this->constInfo.isSink)) {
        runInfo.sinkN1Idx =
            runInfo.commonRunInfo.n2oIdx * this->constInfo.commonConstInfo.gSize + runInfo.commonRunInfo.goIdx;
        uint64_t s1oIdxSink = runInfo.commonRunInfo.s1oIdx;
        // [N, B, S1, S2]
        runInfo.dsinkWorkSpaceOffset =
            runInfo.sinkN1Idx * this->constInfo.bSize * this->constInfo.s1SinkOuter * this->constInfo.s2SinkOuter +
            runInfo.commonRunInfo.boIdx * this->constInfo.s1SinkOuter * this->constInfo.s2SinkOuter +
            s1oIdxSink * this->constInfo.s2SinkOuter + runInfo.s2oIdx;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t FlashAttentionScoreGradKernelDeter<CubeBlockType, VecBlockType>::CalCausalDeterIndex(
    uint32_t roundId, CoordinateInfo& coordinateInfo)
{
    int64_t j = this->cBlockIdx + 1;
    int64_t r = roundId + 1;
    int64_t k = static_cast<int64_t>(this->tilingData->s1s2BNGS1S2BaseParams.coreNum);
    int64_t mGap = 0;
    bool useEvenCol = false;

    if constexpr (!BaseClass::IS_TND) {
        int64_t b = this->constInfo.bSize * this->constInfo.n2Size;
        int64_t m = this->constInfo.s1Outer;
        int64_t n = this->constInfo.s2Outer;
        TransTilingSplitMode<BaseClass::DETER_TILING_SPLIT_MODE>(m, n, r, useEvenCol, deterLoopMax / 2);

        if (this->constInfo.sparseMode == RIGHT_DOWN_CAUSAL && m < n) {
            return -1;
        }

        if (this->constInfo.sparseMode == RIGHT_DOWN_CAUSAL && m > n) {
            mGap = (this->constInfo.commonConstInfo.s1Size - this->constInfo.commonConstInfo.s2Size) /
                   BaseClass::DETER_CUBE_BASEM;
            m -= mGap;
        } else if ((this->constInfo.sparseMode == NO_MASK || this->constInfo.sparseMode == LEFT_UP_CAUSAL) && n > m) {
            // 256 * 128 切分
            n = m;
        }

        if constexpr (BaseClass::IS_N_EQUAL) {
            CalCausalIndex(k, m, n, b, j, r, coordinateInfo);
        } else {
            CalGQACausalIndex(k, m, n, b, j, r, this->constInfo.commonConstInfo.gSize, coordinateInfo);
        }
    } else {
        TransDeterRound<BaseClass::DETER_TILING_SPLIT_MODE>(r, useEvenCol, deterLoopMax / 2);
        if constexpr (BaseClass::IS_N_EQUAL) {
            CalTNDCausalIndex<BaseClass::DETER_CUBE_BASEM, BaseClass::DETER_CUBE_BASEN>(
                this->actualSeqQlenAddr, this->actualSeqKvlenAddr, this->tilingData->deterParam.deterPrefix0,
                this->tilingData->deterParam.deterPrefix1, this->tilingData->deterParam.deterPrefix2,
                this->constInfo.bSize, this->constInfo.n2Size, k, j, r, this->tilingData->deterParam.deterPrefixStep,
                coordinateInfo);
        } else {
            CalTNDGQACausalIndex<BaseClass::DETER_CUBE_BASEM, BaseClass::DETER_CUBE_BASEN>(
                this->actualSeqQlenAddr, this->actualSeqKvlenAddr, this->tilingData->deterParam.deterPrefix0,
                this->tilingData->deterParam.deterPrefix1, this->tilingData->deterParam.deterPrefix2,
                this->constInfo.bSize, this->constInfo.commonConstInfo.gSize, this->constInfo.n2Size, k, j, r,
                this->tilingData->deterParam.deterPrefixStep, this->tilingData->deterParam.coreDivide, coordinateInfo);
        }
    }

    int64_t w = coordinateInfo.batchId;
    int64_t n1 = this->constInfo.commonConstInfo.gSize * this->constInfo.n2Size;
    if (!BaseClass::IS_TND || BaseClass::IS_N_EQUAL) {
        coordinateInfo.batchId = Ceil<int64_t>(w, n1) - 1;
        int64_t n1Idx = w - coordinateInfo.batchId * n1 - 1;
        coordinateInfo.n2Idx = n1Idx / this->constInfo.commonConstInfo.gSize;
        coordinateInfo.gIdx = n1Idx % this->constInfo.commonConstInfo.gSize;
    } else {
        coordinateInfo.batchId -= 1;
        coordinateInfo.n2Idx -= 1;
        coordinateInfo.gIdx -= 1;
    }
    coordinateInfo.s1Idx = coordinateInfo.s1Idx + mGap;
    TransTilingSplitModeBack<BaseClass::DETER_TILING_SPLIT_MODE, BaseClass::IS_TND>(coordinateInfo, useEvenCol);
    coordinateInfo.s1Idx -= 1;

    coordinateInfo.s2Idx = coordinateInfo.s2Idx - 1;
    if (!(w > 0 && coordinateInfo.batchId < this->constInfo.bSize && coordinateInfo.n2Idx < this->constInfo.n2Size &&
          coordinateInfo.gIdx < this->constInfo.commonConstInfo.gSize && coordinateInfo.s1Idx >= 0 &&
          coordinateInfo.s1Idx < coordinateInfo.s1Outer && coordinateInfo.s2Idx >= 0 &&
          coordinateInfo.s2Idx < coordinateInfo.s2Outer)) {
        return -1;
    }
    if (BaseClass::IS_TND) {
        return coordinateInfo.batchId;
    } else {
        return (coordinateInfo.batchId * n1 + coordinateInfo.n2Idx * this->constInfo.commonConstInfo.gSize +
                coordinateInfo.gIdx) *
                   this->constInfo.s1Outer * this->constInfo.s2Outer +
               coordinateInfo.s2Idx * this->constInfo.s1Outer + coordinateInfo.s1Idx;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t FlashAttentionScoreGradKernelDeter<CubeBlockType, VecBlockType>::CalDenseDeterIndex(
    uint32_t roundId, CoordinateInfo& coordinateInfo)
{
    int64_t j = this->cBlockIdx + 1;
    int64_t r = roundId + 1;

    if constexpr (BaseClass::IS_TND) {
        int64_t b = this->constInfo.bSize;
        CalTNDDenseIndex<BaseClass::CUBE_BASEM, BaseClass::CUBE_BASEN, BaseClass::DETER_SPARSE_TYPE,
                         BaseClass::IS_N_EQUAL>(
            this->actualSeqQlenAddr, this->actualSeqKvlenAddr, this->tilingData->deterParam.deterPrefix0,
            this->tilingData->s1s2BNGS1S2SplitCoreParams.deterMaxRound, this->constInfo.bSize, this->constInfo.n2Size,
            this->constInfo.commonConstInfo.gSize, j, r, 0, this->tilingData->deterParam.deterPrefixStep,
            coordinateInfo);
    } else {
        int64_t k = static_cast<int64_t>(this->tilingData->s1s2BNGS1S2BaseParams.coreNum);
        int64_t b = this->constInfo.bSize * this->constInfo.n2Size;
        if constexpr (BaseClass::IS_N_EQUAL) {
            CalDenseIndex(k, this->constInfo.s1Outer, this->constInfo.s2Outer, b, j, r, coordinateInfo);
        } else {
            CalGQADenseIndex(k, this->constInfo.s1Outer, this->constInfo.s2Outer, b, j, r,
                             this->constInfo.commonConstInfo.gSize, coordinateInfo);
        }
    }

    int64_t w = coordinateInfo.batchId;
    int64_t n1 = this->constInfo.commonConstInfo.gSize * this->constInfo.n2Size;
    if (!BaseClass::IS_TND || this->constInfo.commonConstInfo.gSize == 1) {
        coordinateInfo.batchId = Ceil<int64_t>(w, n1) - 1;
        int64_t n1Idx = w - coordinateInfo.batchId * n1 - 1;
        coordinateInfo.n2Idx = n1Idx / this->constInfo.commonConstInfo.gSize;
        coordinateInfo.gIdx = n1Idx % this->constInfo.commonConstInfo.gSize;
    } else {
        coordinateInfo.batchId -= 1;
        coordinateInfo.n2Idx -= 1;
        coordinateInfo.gIdx -= 1;
    }
    coordinateInfo.s1Idx -= 1;
    coordinateInfo.s2Idx -= 1;
    if (!(w > 0 && coordinateInfo.batchId < this->constInfo.bSize && coordinateInfo.n2Idx < this->constInfo.n2Size &&
          coordinateInfo.gIdx < this->constInfo.commonConstInfo.gSize && coordinateInfo.s1Idx >= 0 &&
          coordinateInfo.s1Idx < coordinateInfo.s1Outer && coordinateInfo.s2Idx >= 0 &&
          coordinateInfo.s2Idx < coordinateInfo.s2Outer)) {
        return -1;
    }

    return (coordinateInfo.batchId * n1 + coordinateInfo.n2Idx * this->constInfo.commonConstInfo.gSize +
            coordinateInfo.gIdx) *
               this->constInfo.s1Outer * this->constInfo.s2Outer +
           coordinateInfo.s2Idx * this->constInfo.s1Outer + coordinateInfo.s1Idx;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t FlashAttentionScoreGradKernelDeter<CubeBlockType, VecBlockType>::CalBandDeterIndex(
    uint32_t roundId, CoordinateInfo& coordinateInfo)
{
    int64_t j = this->cBlockIdx + 1;
    int64_t r = roundId + 1;
    bool useEvenCol = false;
    TransDeterRound<BaseClass::DETER_TILING_SPLIT_MODE>(r, useEvenCol, deterLoopMax / 2);
    if constexpr (!BaseClass::IS_TND) {
        if constexpr (BaseClass::IS_N_EQUAL) {
            CalBandIndex(this->bandInfo, j, r, coordinateInfo);
        } else {
            CalGQABandIndex(this->bandInfo, j, r, this->constInfo.commonConstInfo.gSize, coordinateInfo);
        }
    } else {
        int64_t k = static_cast<int64_t>(this->tilingData->s1s2BNGS1S2BaseParams.coreNum);
        coordinateInfo.p = this->constInfo.s1Token;
        coordinateInfo.q = this->constInfo.s2Token;
        if constexpr (BaseClass::IS_N_EQUAL) {
            CalTNDBandIndex<BaseClass::DETER_CUBE_BASEM, BaseClass::DETER_CUBE_BASEN>(
                this->actualSeqQlenAddr, this->actualSeqKvlenAddr, this->tilingData->deterParam.deterPrefix0,
                this->tilingData->deterParam.deterPrefix1, this->constInfo.bSize, this->constInfo.n2Size, k, j, r,
                this->tilingData->deterParam.deterPrefixStep, coordinateInfo);
        } else {
            CalTNDGQABandIndex<BaseClass::DETER_CUBE_BASEM, BaseClass::DETER_CUBE_BASEN>(
                this->actualSeqQlenAddr, this->actualSeqKvlenAddr, this->tilingData->deterParam.deterPrefix0,
                this->tilingData->deterParam.deterPrefix1, this->constInfo.bSize, this->constInfo.commonConstInfo.gSize,
                this->constInfo.n2Size, k, j, r, this->tilingData->deterParam.deterPrefixStep, coordinateInfo);
        }
    }
    int64_t convertedMOffset = coordinateInfo.mOffset;
    int64_t convertedNOffset = coordinateInfo.nOffset;
    if constexpr (BaseClass::DETER_TILING_SPLIT_MODE == 1) {
        convertedMOffset = coordinateInfo.mOffset * 2;
    } else if constexpr (BaseClass::DETER_TILING_SPLIT_MODE == 2) {
        convertedNOffset = coordinateInfo.nOffset * 2;
    }
    TransTilingSplitModeBack<BaseClass::DETER_TILING_SPLIT_MODE, BaseClass::IS_TND>(coordinateInfo, useEvenCol);

    int64_t w = coordinateInfo.batchId;
    int64_t n1 = this->constInfo.commonConstInfo.gSize * this->constInfo.n2Size;
    if (!BaseClass::IS_TND || this->constInfo.commonConstInfo.gSize == 1) {
        coordinateInfo.batchId = Ceil<int64_t>(w, n1) - 1;
        int64_t n1Idx = w - coordinateInfo.batchId * n1 - 1;
        coordinateInfo.n2Idx = n1Idx / this->constInfo.commonConstInfo.gSize;
        coordinateInfo.gIdx = n1Idx % this->constInfo.commonConstInfo.gSize;
    } else {
        coordinateInfo.batchId -= 1;
        coordinateInfo.n2Idx -= 1;
        coordinateInfo.gIdx -= 1;
    }
    coordinateInfo.s1Idx = coordinateInfo.s1Idx - 1 + convertedMOffset;
    coordinateInfo.s2Idx = coordinateInfo.s2Idx - 1 + convertedNOffset;
    // TND this->constInfo.s1Outer得换成真实值
    if (!(w > 0 && coordinateInfo.batchId < this->constInfo.bSize && coordinateInfo.n2Idx < this->constInfo.n2Size &&
          coordinateInfo.gIdx < this->constInfo.commonConstInfo.gSize && coordinateInfo.s1Idx >= 0 &&
          coordinateInfo.s1Idx < coordinateInfo.s1Outer && coordinateInfo.s2Idx >= 0 &&
          coordinateInfo.s2Idx < coordinateInfo.s2Outer)) {
        return -1;
    }

    if (BaseClass::IS_TND) {
        return coordinateInfo.batchId;
    } else {
        return (coordinateInfo.batchId * n1 + coordinateInfo.n2Idx * this->constInfo.commonConstInfo.gSize +
                coordinateInfo.gIdx) *
                   this->constInfo.s1Outer * this->constInfo.s2Outer +
               coordinateInfo.s2Idx * this->constInfo.s1Outer + coordinateInfo.s1Idx;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t FlashAttentionScoreGradKernelDeter<CubeBlockType, VecBlockType>::SpecialS2Index(
    int64_t dkvGmOffset)
{
    if constexpr (IS_DETER_NEW(DETER_SPARSE_TYPE)) {
        for (int8_t i = 0; i < MAX_CUBE_CORE_NUM; i++) {
            if (this->tilingData->deterParam.deterPrefix2[i] == dkvGmOffset) {
                return i;
            }
        }
    }
    return -1;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelDeter<CubeBlockType, VecBlockType>::DeterSync(int64_t loopIdx)
{
    if (loopIdx == 0) {
        return;
    }

    // 此处复用dqIsNeedDeter和dkDvIsNeedDeter装载需要同步的轮次范围
    for (int8_t i = 0; i < MAX_CUBE_CORE_NUM; i++) {
        if (this->tilingData->s1s2BNGS1S2SplitCoreParams.dkDvIsNeedDeter[i] == 0) {
            return;
        }
        if (static_cast<uint64_t>(loopIdx) >= this->tilingData->s1s2BNGS1S2SplitCoreParams.dqIsNeedDeter[i] &&
            static_cast<uint64_t>(loopIdx) <= this->tilingData->s1s2BNGS1S2SplitCoreParams.dkDvIsNeedDeter[i]) {
            CrossCoreSetFlag<0, PIPE_FIX>(this->SYNC_DETER_FIX_FLAG);
            CrossCoreWaitFlag<0, PIPE_FIX>(this->SYNC_DETER_FIX_FLAG);
            return;
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline bool FlashAttentionScoreGradKernelDeter<CubeBlockType, VecBlockType>::IsValidDeterForTnd(
    FagRunInfo& runInfo, int64_t index, CoordinateInfo& coordinateInfo)
{
    if (coordinateInfo.batchId < 0 || index < 0) {
        return false;
    }
    int64_t bIdx = coordinateInfo.batchId;
    int64_t actualS1Len = coordinateInfo.actualS1Len;
    int64_t actualS2Len = coordinateInfo.actualS2Len;
    int64_t s1oDimIdx = coordinateInfo.s1Idx;
    int64_t s2oDimIdx = coordinateInfo.s2Idx;
    int64_t s2IdxLeft = s2oDimIdx * this->CUBE_BASEN;
    int64_t s2IdxRight = Min((s2oDimIdx + 1) * this->CUBE_BASEN, actualS2Len);
    if constexpr (IS_ATTEN_MASK) {
        if (this->constInfo.sparseMode == RIGHT_DOWN_CASUAL_BAND && bIdx != this->attenMaskInfo.bandIndex) {
            this->actualCalcS1Token = static_cast<int64_t>(INT32_MAX) + actualS1Len - actualS2Len;
            this->actualCalcS2Token = static_cast<int64_t>(0) - actualS1Len + actualS2Len;
        } else if (this->constInfo.sparseMode == BAND_LEFT_UP_CASUAL && bIdx != this->attenMaskInfo.bandIndex) {
            this->actualCalcS1Token = INT32_MAX;
            this->actualCalcS2Token = 0;
        } else if (this->constInfo.sparseMode == RIGHT_DOWN_CAUSAL || this->constInfo.sparseMode == BAND ||
                   (this->constInfo.sparseMode == RIGHT_DOWN_CASUAL_BAND && bIdx == this->attenMaskInfo.bandIndex) ||
                   (this->constInfo.sparseMode == BAND_LEFT_UP_CASUAL && bIdx == this->attenMaskInfo.bandIndex)) {
            this->actualCalcS1Token = this->constInfo.s1Token + actualS1Len - actualS2Len;
            this->actualCalcS2Token = this->constInfo.s2Token - actualS1Len + actualS2Len;
        }

        int64_t s2SparseLeft = Max(this->CUBE_BASEM * s1oDimIdx - this->actualCalcS1Token, 0);
        s2SparseLeft = s2SparseLeft >> 6 << 6;
        int64_t s2SparseRight = AlignTo64(
            Min(this->CUBE_BASEM * (s1oDimIdx + 1), this->constInfo.commonConstInfo.s1Size) + this->actualCalcS2Token);
        s2SparseRight = Min(s2SparseRight, actualS2Len);
        bool isValid = s2IdxLeft < s2SparseRight && s2IdxRight > s2SparseLeft;
        runInfo.s2CvBegin = s2IdxLeft;
        runInfo.s2CvEnd = runInfo.s2CvBegin + this->CUBE_BASEN; // 非尾块s2按照+CUBE_BASEN处理
        if (s2oDimIdx == coordinateInfo.s2Outer - 1) {          // 默认s2 cv tail相等
            runInfo.s2CvEnd = runInfo.s2CvBegin + actualS2Len - s2oDimIdx * this->CUBE_BASEN;
        }
        return isValid;
    } else {
        runInfo.s2CvBegin = s2IdxLeft;
        runInfo.s2CvEnd = s2IdxRight;
        return true;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelDeter<CubeBlockType, VecBlockType>::Process()
{
    if constexpr (IS_DETER_NEW(DETER_SPARSE_TYPE)) {
        Process_NEW_DETER();
    } else {
        Process_OLD_DETER();
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelDeter<CubeBlockType, VecBlockType>::Process_NEW_DETER()
{
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<0, PIPE_FIX>(this->SYNC_DETER_FIX_FLAG);
    }

    deterLoopMax = CalDeterMaxLoopNum();

    int64_t blockInnerIdx = 0;
    int64_t taskId = 0, preRunId = -1, runId = 0, nextRunId = 1;
    FagRunInfo runInfos[3]; // for ping pong

    int64_t nextValidLoopIdx;
    int64_t nextblockIdx;
    bool isLastLoop;
    int8_t ubBufIdx = 0;
    int64_t innerMm12LoopIdx = 0;
    LocalTensor<T2> mm1ResTensor;
    LocalTensor<T2> mm2ResTensor;

    CalDeterIndex(0, deterLoopMax, nextValidLoopIdx, nextblockIdx, taskId, coordinateInfos[taskId & 1],
                  runInfos[taskId & 1]);
    // Pre-Set
    blockInnerIdx = nextblockIdx;
    if (blockInnerIdx != -1) {
        if constexpr (BaseClass::IS_TND) {
            SetRunInfoDeterForTND(runInfos[runId], taskId, blockInnerIdx, coordinateInfos[taskId & 1], nextblockIdx);
        } else {
            this->SetRunInfo(runInfos[runId], taskId, blockInnerIdx);
        }
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(this->QKDXV_RES_RELEASE[0]);
        }
        if ASCEND_IS_AIV {
            CrossCoreSetFlag<SYNC_MODE, PIPE_V>(this->QKDXV_RES_RELEASE[0]);
        }
        // Pre-Copy
        this->vecBlock.ProcessVec1(runInfos[runId], this->constInfo);
    }

    for (int64_t loopIdx = 0; loopIdx < deterLoopMax + 1; loopIdx++) {
        if (loopIdx >= nextValidLoopIdx) {
            blockInnerIdx = nextblockIdx;
            CalDeterIndex(loopIdx + 1, deterLoopMax, nextValidLoopIdx, nextblockIdx, taskId,
                          coordinateInfos[(taskId + 1) & 1], runInfos[nextRunId]);
        } else {
            blockInnerIdx = -1;
        }
        isLastLoop = (blockInnerIdx == -1 && nextblockIdx == -1);
        bool isValidBlock = (blockInnerIdx >= 0);

        if (isValidBlock) {
            if (nextblockIdx != -1) {
                if constexpr (BaseClass::IS_TND) {
                    SetRunInfoDeterForTND(runInfos[nextRunId], taskId + 1, nextblockIdx,
                                          coordinateInfos[(taskId + 1) & 1], nextblockIdx);
                } else {
                    this->SetRunInfo(runInfos[nextRunId], taskId + 1, nextblockIdx);
                }
            }
            uint32_t innerLoopNum =
                (runInfos[runId].commonRunInfo.s1RealSize + this->VECTOR_BASEM - 1) / this->VECTOR_BASEM;
            for (uint32_t innerLoopIdx = 0; innerLoopIdx < innerLoopNum; innerLoopIdx++) {
                mm1ResTensor = this->mm1ResBuf[ubBufIdx].template Get<T2>();
                mm2ResTensor = this->mm2ResBuf[ubBufIdx].template Get<T2>();
                if ASCEND_IS_AIC {
                    if (innerMm12LoopIdx >= this->UB_BUF_NUM) {
                        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(this->QKDXV_RES_RELEASE[ubBufIdx]);
                    }
                }
                this->cubeBlock.IterateMmQKAndDyV(mm1ResTensor, mm2ResTensor, runInfos[runId], runInfos[nextRunId],
                                                  this->constInfo, innerLoopNum, innerLoopIdx, ubBufIdx);
                if ASCEND_IS_AIC {
                    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(this->QKDXV_RES_READY[ubBufIdx]);
                }
                this->vecBlock.ProcessVec2(runInfos[runId], runInfos[nextRunId], this->constInfo, blockInnerIdx == -1,
                                           mm1ResTensor, mm2ResTensor, innerLoopNum, innerLoopIdx, ubBufIdx);
                ubBufIdx = (ubBufIdx + 1 == this->UB_BUF_NUM) ? 0 : ubBufIdx + 1;
                innerMm12LoopIdx += 1;
            }
            runInfos[runId].completed = false;
        }

        if (preRunId >= 0 && !(runInfos[preRunId].completed)) {
            if ASCEND_IS_AIC {
                runInfos[preRunId].nextS2IdxNoChange = isLastLoop ? false : runInfos[runId].isS2IdxNoChange;
                CrossCoreWaitFlag<0, PIPE_FIX>(this->SYNC_DETER_FIX_FLAG);
                this->cubeBlock.IterateMmDsK(runInfos[preRunId], this->constInfo, this->dqWorkSpaceGm); // bmm3
                CrossCoreSetFlag<0, PIPE_FIX>(this->SYNC_DETER_FIX_FLAG);
                this->cubeBlock.IterateMmDsQ(runInfos[preRunId], this->constInfo, this->dkWorkSpaceGm); // bmm4
                this->cubeBlock.IterateMmPDy(runInfos[preRunId], this->constInfo, this->dvWorkSpaceGm); // bmm5
            }
            runInfos[preRunId].completed = true;
        } else {
            if ASCEND_IS_AIC {
                CrossCoreWaitFlag<0, PIPE_FIX>(this->SYNC_DETER_FIX_FLAG);
                CrossCoreSetFlag<0, PIPE_FIX>(this->SYNC_DETER_FIX_FLAG);
            }
        }

        if (isValidBlock) {
            taskId++;
            preRunId = runId;
            runId = nextRunId;
            nextRunId = nextRunId + 1 == 3 ? 0 : nextRunId + 1;
        }
    }
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag<0, PIPE_FIX>(this->SYNC_DETER_FIX_FLAG);
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(this->ALL_DONE);
    }
    this->vecBlock.FreeAsync(this->constInfo);
    this->cubeBlock.FreeBuffer();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelDeter<CubeBlockType, VecBlockType>::Process_OLD_DETER()
{
    int64_t maxValidBBLen = this->tilingData->s1s2BNGS1S2SplitCoreParams.maxValidBBLen;
    int64_t remainLoopNum = maxValidBBLen;
    int64_t blockInnerIdx = 0;
    int64_t taskId = 0, preRunId = -1, runId = 0, nextRunId = 1;
    int64_t loopIdx = 0;
    FagRunInfo runInfos[3];
    int16_t deterPpFlag = 1;
    isFirstDeter = true;

    bool isLastLoop = false;
    int8_t ubBufIdx = 0;
    int64_t innerMm12LoopIdx = 0;
    int64_t curLoopIdx = 0; // just for continuous split core
    LocalTensor<T2> mm1ResTensor;
    LocalTensor<T2> mm2ResTensor;
    bool needSyncDkMM = false;

    // Pre-Copy
    if (this->tilingData->s1s2BNGS1S2BlockNumList.blockEnds[this->cBlockIdx] != 0) {
        int64_t firstValidIdx =
            this->GetNextValidIdx(runInfos[runId], taskId,
                                  this->tilingData->s1s2BNGS1S2BlockNumList.blockStarts[this->cBlockIdx], curLoopIdx);
        if (firstValidIdx != -1 &&
            firstValidIdx <= this->tilingData->s1s2BNGS1S2BlockNumList.blockEnds[this->cBlockIdx]) {
            this->SetRunInfo(runInfos[runId], 0, firstValidIdx);
            if ASCEND_IS_AIC {
                CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(this->QKDXV_RES_RELEASE[0]);
            }
            if ASCEND_IS_AIV {
                CrossCoreSetFlag<SYNC_MODE, PIPE_V>(this->QKDXV_RES_RELEASE[0]);
            }
            this->vecBlock.ProcessVec1(runInfos[runId], this->constInfo);
        }
    }

    while (remainLoopNum > 0) {
        blockInnerIdx = this->tilingData->s1s2BNGS1S2BlockNumList.blockStarts[this->cBlockIdx] + loopIdx;
        if (this->tilingData->s1s2BNGS1S2BlockNumList.blockEnds[this->cBlockIdx] != 0 &&
            blockInnerIdx <=
                this->tilingData->s1s2BNGS1S2BlockNumList.blockEnds[this->cBlockIdx]) { // 有主流程需要处理的核
            isLastLoop = (blockInnerIdx == this->tilingData->s1s2BNGS1S2BlockNumList.blockEnds[this->cBlockIdx]);

            // 跳过无效块
            if (!isLastLoop && !this->IsValid(runInfos[runId], taskId, blockInnerIdx)) {
                loopIdx++;
                continue;
            }
            if (!isLastLoop) {
                int64_t nextValidIdx = -1;
                nextValidIdx = this->GetNextValidIdx(runInfos[nextRunId], taskId, blockInnerIdx + 1, curLoopIdx);
                if (nextValidIdx != -1) {
                    this->SetRunInfo(runInfos[nextRunId], taskId + 1, nextValidIdx);
                }
                if ASCEND_IS_AIC {
                    if (needSyncDkMM) {
                        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(this->SYNC_DETER_FIX_FLAG);
                    }
                }
                uint32_t innerLoopNum =
                    (runInfos[runId].commonRunInfo.s1RealSize + this->VECTOR_BASEM - 1) / this->VECTOR_BASEM;
                for (uint32_t innerLoopIdx = 0; innerLoopIdx < innerLoopNum; innerLoopIdx++) {
                    mm1ResTensor = this->mm1ResBuf[ubBufIdx].template Get<T2>();
                    mm2ResTensor = this->mm2ResBuf[ubBufIdx].template Get<T2>();
                    if ASCEND_IS_AIC {
                        if (innerMm12LoopIdx >= this->UB_BUF_NUM) {
                            CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(this->QKDXV_RES_RELEASE[ubBufIdx]);
                        }
                    }
                    this->cubeBlock.IterateMmQKAndDyV(mm1ResTensor, mm2ResTensor, runInfos[runId], runInfos[nextRunId],
                                                      this->constInfo, innerLoopNum, innerLoopIdx, ubBufIdx);
                    if ASCEND_IS_AIC {
                        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(this->QKDXV_RES_READY[ubBufIdx]);
                    }
                    this->vecBlock.ProcessVec2(runInfos[runId], runInfos[nextRunId], this->constInfo,
                                               nextValidIdx == -1, mm1ResTensor, mm2ResTensor, innerLoopNum,
                                               innerLoopIdx, ubBufIdx);
                    ubBufIdx = (ubBufIdx + 1 == this->UB_BUF_NUM) ? 0 : ubBufIdx + 1;
                    innerMm12LoopIdx += 1;
                }

                if ASCEND_IS_AIV {
                    this->vecBlock.WriteOffsetToGM(this->constInfo, static_cast<int64_t>(runInfos[runId].queryGmOffset),
                                                   static_cast<int64_t>(runInfos[runId].keyGmOffset),
                                                   static_cast<int64_t>(runInfos[runId].valueGmOffset), taskId,
                                                   maxValidBBLen - taskId);
                }
            }

            // mm345滞后一块
            if (taskId > 0) {
                deterPpFlag = 1 - deterPpFlag;
                GetIsNeedDeter(taskId - 1);
                if ASCEND_IS_AIC {
                    // 这个位置同步需确认
                    if (needSyncDkMM) {
                        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(6);
                    }
                    runInfos[preRunId].nextS2IdxNoChange = isLastLoop ? false : runInfos[runId].isS2IdxNoChange;
                    this->cubeBlock.IterateMmDsK(runInfos[preRunId], this->constInfo, this->dqWorkSpaceGm,
                                                 dqIsNeedDeter[deterPpFlag], this->cBlockIdx, &this->deterGm,
                                                 deterPpFlag);
                    this->cubeBlock.IterateMmDsQ(runInfos[preRunId], this->constInfo, this->dkWorkSpaceGm,
                                                 dkDvIsNeedDeter[deterPpFlag], this->cBlockIdx, &this->deterGm,
                                                 deterPpFlag);
                    this->cubeBlock.IterateMmPDy(runInfos[preRunId], this->constInfo, this->dvWorkSpaceGm,
                                                 dkDvIsNeedDeter[deterPpFlag], this->cBlockIdx, &this->deterGm,
                                                 deterPpFlag);
                    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(this->SYNC_C4_TO_V6_FLAG);
                }
                needSyncDkMM = !this->constInfo.deterConstInfo.noNeedDeter;
                // 确定性计算再滞后一块
                if (taskId > 1) {
                    if ASCEND_IS_AIV {
                        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(this->SYNC_C4_TO_V6_FLAG);
                        mm1ResTensor = this->mm1ResBuf[runInfos[preRunId].commonRunInfo.taskIdMod2].template Get<T2>();
                        mm2ResTensor = this->mm2ResBuf[runInfos[preRunId].commonRunInfo.taskIdMod2].template Get<T2>();
                        int64_t curComputeBlockIdx = maxValidBBLen - remainLoopNum;
                        this->vecBlock.DeterCompute(
                            mm1ResTensor, mm2ResTensor, this->constInfo, dqIsNeedDeter[curComputeBlockIdx & 1],
                            dkDvIsNeedDeter[curComputeBlockIdx & 1],
                            /*isLastSort=*/false, curComputeBlockIdx, remainLoopNum, deterPpFlag);
                    }
                    remainLoopNum--;
                }
                if ASCEND_IS_AIV {
                    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(this->SYNC_DETER_FIX_FLAG);
                    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(6);
                }

                // 主流程末尾补一次确定性计算
                if (isLastLoop) {
                    if ASCEND_IS_AIV {
                        mm1ResTensor = this->mm1ResBuf[runInfos[runId].commonRunInfo.taskIdMod2].template Get<T2>();
                        mm2ResTensor = this->mm2ResBuf[runInfos[runId].commonRunInfo.taskIdMod2].template Get<T2>();
                        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(this->SYNC_C4_TO_V6_FLAG);
                        int64_t curComputeBlockIdx = maxValidBBLen - remainLoopNum;
                        this->vecBlock.DeterCompute(
                            mm1ResTensor, mm2ResTensor, this->constInfo, dqIsNeedDeter[curComputeBlockIdx & 1],
                            dkDvIsNeedDeter[curComputeBlockIdx & 1],
                            /*isLastSort=*/true, curComputeBlockIdx, remainLoopNum, deterPpFlag);
                    }
                    remainLoopNum--;
                }
            }
            taskId++;
            if (preRunId == -1) {
                preRunId = runId;
            } else {
                preRunId = runId;
            }
            runId = nextRunId;
            nextRunId = (nextRunId + 1 == 3) ? 0 : nextRunId + 1;
        } else if (this->tilingData->s1s2BNGS1S2BlockNumList.blockEnds[this->cBlockIdx] !=
                   0) { // 主流程结束，仍要进行确定性计算
            int64_t computeLoopIdx = maxValidBBLen - remainLoopNum;
            GetIsNeedDeter(computeLoopIdx);
            if ASCEND_IS_AIV {
                this->vecBlock.WriteOffsetToGM(this->constInfo, this->OUTINDEX, this->OUTINDEX, this->OUTINDEX,
                                               computeLoopIdx, remainLoopNum);
            }
            deterPpFlag = 1 - deterPpFlag;
            if ASCEND_IS_AIV {
                mm1ResTensor = this->mm1ResBuf[runInfos[runId].commonRunInfo.taskIdMod2].template Get<T2>();
                mm2ResTensor = this->mm2ResBuf[runInfos[runId].commonRunInfo.taskIdMod2].template Get<T2>();
                this->vecBlock.DeterCompute(mm1ResTensor, mm2ResTensor, this->constInfo,
                                            dqIsNeedDeter[computeLoopIdx & 1], dkDvIsNeedDeter[computeLoopIdx & 1],
                                            /*isLastSort=*/true, computeLoopIdx, remainLoopNum, deterPpFlag);
            }
            remainLoopNum--;
        } else { // 没有主流程，仍要参与确定性计算
            int64_t computeLoopIdx = maxValidBBLen - remainLoopNum;
            GetIsNeedDeter(computeLoopIdx);
            deterPpFlag = 1 - deterPpFlag;
            if ASCEND_IS_AIV {
                mm1ResTensor = this->mm1ResBuf[runInfos[runId].commonRunInfo.taskIdMod2].template Get<T2>();
                mm2ResTensor = this->mm2ResBuf[runInfos[runId].commonRunInfo.taskIdMod2].template Get<T2>();
                this->vecBlock.DeterCompute(mm1ResTensor, mm2ResTensor, this->constInfo,
                                            dqIsNeedDeter[computeLoopIdx & 1], dkDvIsNeedDeter[computeLoopIdx & 1],
                                            /*isLastSort=*/true, computeLoopIdx, remainLoopNum, deterPpFlag);
            }
            remainLoopNum--;
        }
        loopIdx++;
    }
    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(this->ALL_DONE);
    this->vecBlock.FreeAsync(this->constInfo);
    this->cubeBlock.FreeBuffer();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelDeter<CubeBlockType, VecBlockType>::GetIsNeedDeter(
    int64_t computeLoopIdx)
{
    if (this->constInfo.deterConstInfo.noNeedDeter) {
        return;
    }
    if (unlikely(computeLoopIdx >= this->MAX_BITS_IN_TILING)) {
        dqIsNeedDeter[computeLoopIdx & 1] = true;
        dkDvIsNeedDeter[computeLoopIdx & 1] = true;
    } else {
        // caculate index and bit position
        int64_t arrayIndex = computeLoopIdx / this->BITS_EACH_UINT64;
        int64_t bitShift = computeLoopIdx % this->BITS_EACH_UINT64;
        uint64_t mask = 1ULL << bitShift;
        dqIsNeedDeter[computeLoopIdx & 1] =
            (this->tilingData->s1s2BNGS1S2SplitCoreParams.dqIsNeedDeter[arrayIndex] & mask) != 0;
        dkDvIsNeedDeter[computeLoopIdx & 1] =
            (this->tilingData->s1s2BNGS1S2SplitCoreParams.dkDvIsNeedDeter[arrayIndex] & mask) != 0;
    }
}

} // namespace FagBaseApi

#endif
