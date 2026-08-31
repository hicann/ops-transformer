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
 * \file flash_attention_score_grad_kernel_deter.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_KERNEL_DETER_H
#define FLASH_ATTENTION_SCORE_GRAD_KERNEL_DETER_H

#include "flash_attention_score_grad_common.h"
#include "flash_attention_score_grad_kernel_base.h"
#include "cube_api/mutex_buffer.h"
#include "flash_attention_score_grad_tiling_data_regbase.h"
#include "deter.h"

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
                                GM_ADDR deqScaleK, GM_ADDR deqScaleV, GM_ADDR deqScaleDy, GM_ADDR queryRope,
                                GM_ADDR keyRope, GM_ADDR sink, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dpse,
                                GM_ADDR dqRope, GM_ADDR dkRope, GM_ADDR dsink, GM_ADDR workspace,
                                FagTilingType ordTilingData, TPipe *pipeIn);
    __aicore__ inline void SetRunInfoDeterForTND(FagRunInfo &runInfo, int64_t taskId, int64_t index,
                                                 CoordinateInfo &coordinateInfo, int64_t nextIndex);
    __aicore__ inline int64_t CalDeterMaxLoopNum();
    __aicore__ inline void CalDeterIndex(uint32_t roundId, uint32_t maxLoopNum, int64_t &nextValidRoundId,
                                         int64_t &nextValidIndex, int64_t taskId, CoordinateInfo &coordinateInfo,
                                         FagRunInfo &runInfo);
    __aicore__ inline int64_t CalDenseDeterIndex(uint32_t roundId, CoordinateInfo &coordinateInfo);
    __aicore__ inline int64_t CalCausalDeterIndex(uint32_t roundId, CoordinateInfo &coordinateInfo);
    __aicore__ inline int64_t CalBandDeterIndex(uint32_t roundId, CoordinateInfo &coordinateInfo);
    __aicore__ inline bool IsValidDeterForTnd(FagRunInfo &runInfo, int64_t index, CoordinateInfo &coordinateInfo);
    __aicore__ inline void GetNextDxAndQueryOffsetTND(FagRunInfo &runInfo, int64_t nextIndex,
                                                      PreloadArgs<IS_ROPE> &preloadArgs);
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
    int64_t specialDkGmOffset = 0;
    int64_t specialDvGmOffset = 0;
    int16_t deterPpFlag = 1;
    int64_t deterGmOffset = 0;
    int64_t specialHalfS2RealSize = 0;
    int64_t specialFirstHalfS2RealSize = 0;
    bool isFirstBlock = false;
    typename std::conditional<IS_DETER_NEW(DETER_SPARSE_TYPE), CoordinateInfo[2], std::nullptr_t>::type
        coordinateInfos{};
    typename std::conditional<DETER_SPARSE_TYPE == DETER_BAND, BandInfo, std::nullptr_t>::type bandInfo;
    bool isMm3NeedWait = false;
    typename std::conditional<IS_DETER_OLD(DETER_SPARSE_TYPE), bool[2], std::nullptr_t>::type dqIsNeedDeter{};
    typename std::conditional<IS_DETER_OLD(DETER_SPARSE_TYPE), bool[2], std::nullptr_t>::type dkDvIsNeedDeter{};
    constexpr static int64_t OUTINDEX = -1;
    int64_t deterLoopMax = 0;
};

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelDeter<CubeBlockType, VecBlockType>::Init(
    GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR query, GM_ADDR pseShift, GM_ADDR dropMask, GM_ADDR attenMask,
    GM_ADDR y, GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR prefixN, GM_ADDR actualSeqQlen, GM_ADDR actualSeqKvlen,
    GM_ADDR deqScaleQ, GM_ADDR deqScaleK, GM_ADDR deqScaleV, GM_ADDR deqScaleDy, GM_ADDR queryRope, GM_ADDR keyRope,
    GM_ADDR sink, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dpse, GM_ADDR dqRope, GM_ADDR dkRope, GM_ADDR dsink,
    GM_ADDR workspace, FagTilingType ordTilingData, TPipe *pipeIn)
{
    BaseClass::Init(key, value, dy, query, pseShift, dropMask, attenMask, y, softmaxMax, softmaxSum, prefixN,
                    actualSeqQlen, actualSeqKvlen, deqScaleQ, deqScaleK, deqScaleV, deqScaleDy, queryRope, keyRope,
                    sink, dq, dk, dv, dpse, dqRope, dkRope, dsink, workspace, ordTilingData, pipeIn);

    dkvWorkSpaceOffet = this->cBlockIdx * this->CUBE_BASEN * this->HEAD_DIM_ALIGN;
    dAlign16 = AlignTo16(this->constInfo.commonConstInfo.dSize);
    dvAlign16 = AlignTo16(this->constInfo.commonConstInfo.dSizeV);

    deterGm.SetGlobalBuffer((__gm__ float *)workspace +
                            this->tilingData->baseDeterParam.deterGmOffset / sizeof(CALC_TYPE));
    deterGmOffset = this->cBlockIdx * this->CUBE_BASEN * this->HEAD_DIM_ALIGN * NUM_TWO;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelDeter<CubeBlockType, VecBlockType>::GetNextDxAndQueryOffsetTND(
    FagRunInfo &runInfo, int64_t nextIndex, PreloadArgs<IS_ROPE> &preloadArgs)
{
    preloadArgs.copyNext = !(nextIndex == -1);
    preloadArgs.copyCurrent = (runInfo.commonRunInfo.taskId == 0);
    if (!preloadArgs.copyNext) {
        runInfo.isNextS2IdxNoChange = false;
        return;
    }
    int64_t nextBoIdx = 0;
    int64_t bDimTail = 0;
    int64_t nextN2oIdx = 0;
    int64_t n2DimTail = 0;
    int64_t nextGoIdx = 0;
    int64_t gDimTail = 0;
    int64_t nextS1oIdx = 0;
    int64_t nextS2oIdx = 0;

    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t gOffset = 0;
    int64_t s1Offset = 0;
    int64_t bOffsetDv = 0;
    int64_t n2OffsetDv = 0;
    int64_t gOffsetDv = 0;
    int64_t s1OffsetDv = 0;

    if constexpr (IS_TND && IS_DETER_NEW(DETER_SPARSE_TYPE)) {
        int64_t actualS1Len = coordinateInfos[nextIndex].actualS1Len;
        int64_t s1OuterTmp = coordinateInfos[nextIndex].s1Outer;
        int64_t bIdx = coordinateInfos[nextIndex].batchId;
        int64_t seqQLenPrefix = bIdx == 0 ? 0 : ((__gm__ int64_t *)this->actualSeqQlenAddr)[bIdx - 1];
        int64_t lastBatchTotalS1BOffset = seqQLenPrefix * this->constInfo.commonConstInfo.n2GD;
        int64_t lastBatchTotalS1BOffsetForDv = seqQLenPrefix * this->constInfo.commonConstInfo.n2GDv;
        int64_t s1CvTail = actualS1Len - (s1OuterTmp - 1) * this->CUBE_BASEM;

        nextS1oIdx = coordinateInfos[nextIndex].s1Idx;
        nextN2oIdx = coordinateInfos[nextIndex].n2Idx;
        nextGoIdx = coordinateInfos[nextIndex].gIdx;
        bOffset = lastBatchTotalS1BOffset;
        s1Offset = nextS1oIdx * this->CUBE_BASEM * this->constInfo.commonConstInfo.n2GD;
        n2Offset = nextN2oIdx * this->constInfo.commonConstInfo.gD;
        gOffset = nextGoIdx * this->constInfo.commonConstInfo.dSize;
        if constexpr (IS_D_NO_EQUAL) {
            bOffsetDv = lastBatchTotalS1BOffsetForDv;
            s1OffsetDv = nextS1oIdx * this->CUBE_BASEM * this->constInfo.commonConstInfo.n2GDv;
            n2OffsetDv = nextN2oIdx * this->constInfo.commonConstInfo.gDv;
            gOffsetDv = nextGoIdx * this->constInfo.commonConstInfo.dSizeV;
        }
        preloadArgs.nextMOrN = (nextS1oIdx == s1OuterTmp - 1) ? s1CvTail : this->CUBE_BASEM;
        nextS2oIdx = coordinateInfos[nextIndex].s2Idx;
        nextBoIdx = coordinateInfos[nextIndex].batchId;
    }
    preloadArgs.nextQueryOffset = bOffset + n2Offset + gOffset + s1Offset;
    if constexpr (IS_ROPE) {
        preloadArgs.nextQueryRopeOffset = preloadArgs.nextQueryOffset / 3;
        preloadArgs.nextQueryOffset = (preloadArgs.nextQueryOffset / 3) << 1;
    }
    if constexpr (IS_D_NO_EQUAL) {
        preloadArgs.nextDyOffset = bOffsetDv + n2OffsetDv + gOffsetDv + s1OffsetDv;
    } else {
        preloadArgs.nextDyOffset = preloadArgs.nextQueryOffset;
    }

    runInfo.isNextS2IdxNoChange = (nextS2oIdx == runInfo.s2oIdx && nextN2oIdx == runInfo.commonRunInfo.n2oIdx &&
                                   nextBoIdx == runInfo.commonRunInfo.boIdx);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline bool FlashAttentionScoreGradKernelDeter<CubeBlockType, VecBlockType>::IsValidDeterForTnd(
    FagRunInfo &runInfo, int64_t index, CoordinateInfo &coordinateInfo)
{
    if (coordinateInfo.batchId < 0 || index < 0) {
        return false;
    }
    int64_t bIdx = coordinateInfo.batchId;            // 0
    int64_t actualS1Len = coordinateInfo.actualS1Len; // 512
    int64_t actualS2Len = coordinateInfo.actualS2Len; // 257
    int64_t s1oDimIdx = coordinateInfo.s1Idx;         // 3
    int64_t s2oDimIdx = coordinateInfo.s2Idx;         // 2
    int64_t s2IdxLeft = s2oDimIdx * this->CUBE_BASEN;
    int64_t s2IdxRight = Min((s2oDimIdx + 1) * this->CUBE_BASEN, actualS2Len);
    if constexpr (BaseClass::IS_ATTEN_MASK) {
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
__aicore__ inline void FlashAttentionScoreGradKernelDeter<CubeBlockType, VecBlockType>::SetRunInfoDeterForTND(
    FagRunInfo &runInfo, int64_t taskId, int64_t index, CoordinateInfo &coordinateInfo, int64_t nextIndex)
{
    int64_t bIdx = coordinateInfo.batchId;
    int64_t actualS1Len = coordinateInfo.actualS1Len;
    int64_t actualS2Len = coordinateInfo.actualS2Len;
    int64_t s1OuterTmp = coordinateInfo.s1Outer;
    int64_t s1CvTailTmp = actualS1Len - (s1OuterTmp - 1) * this->CUBE_BASEM;
    runInfo.commonRunInfo.boIdx = bIdx;

    if (runInfo.lastBatchIdx != bIdx) {
        int64_t seqQLenPrefix = bIdx == 0 ? 0 : ((__gm__ int64_t *)this->actualSeqQlenAddr)[bIdx - 1];
        int64_t seqKvLenPrefix = bIdx == 0 ? 0 : ((__gm__ int64_t *)this->actualSeqKvlenAddr)[bIdx - 1];
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
            runInfo.lastBatchTotalS1BRopeOffset = seqQLenPrefix * this->constInfo.commonConstInfo.n2GDr;
            runInfo.lastBatchTotalS2BRopeOffset = seqKvLenPrefix * this->constInfo.commonConstInfo.n2Dr;
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
    runInfo.halfS2RealSize = (runInfo.commonRunInfo.s2RealSize + 1) >> 1;
    runInfo.firstHalfS2RealSize = runInfo.halfS2RealSize;
    runInfo.commonRunInfo.halfS1RealSize = (runInfo.commonRunInfo.s1RealSize + 1) >> 1;
    runInfo.commonRunInfo.firstHalfS1RealSize = runInfo.commonRunInfo.halfS1RealSize;
    if (this->vSubBlockIdx == 1) {
        runInfo.commonRunInfo.halfS1RealSize = runInfo.commonRunInfo.s1RealSize - runInfo.commonRunInfo.halfS1RealSize;
        runInfo.halfS2RealSize = runInfo.commonRunInfo.s2RealSize - runInfo.halfS2RealSize;
    }

    runInfo.commonRunInfo.actualS1Size = actualS1Len;
    runInfo.commonRunInfo.actualS2Size = actualS2Len;
    runInfo.commonRunInfo.s2SizeAcc = runInfo.lastBatchTotalS2Size;
    runInfo.commonRunInfo.b1SSOffsetAlign = runInfo.lastBatchTotalS1S2SizeAlign;
    runInfo.commonRunInfo.b1SSOffset = runInfo.lastBatchTotalS1S2Size;
    runInfo.commonRunInfo.s2StartIdx = runInfo.s2CvBegin;
    runInfo.commonRunInfo.vecCoreOffset = this->vSubBlockIdx * runInfo.commonRunInfo.firstHalfS1RealSize;
    runInfo.commonRunInfo.s2AlignedSize = AlignTo16(runInfo.commonRunInfo.s2RealSize);
    runInfo.s1AlignedSize = AlignTo16(runInfo.commonRunInfo.s1RealSize);

    runInfo.isS2IdxNoChange =
        (this->lastS2oCvDimIdx == runInfo.s2oIdx && this->lastBdimIdx == runInfo.commonRunInfo.boIdx &&
         this->lastN2dimIdx == runInfo.commonRunInfo.n2oIdx);
    if (!runInfo.isS2IdxNoChange) {
        this->lastS2oCvDimIdx = runInfo.s2oIdx;
        this->lastBdimIdx = runInfo.commonRunInfo.boIdx;
        this->lastN2dimIdx = runInfo.commonRunInfo.n2oIdx;
    }

    //----------------------------------------------PART
    // preload next query and dy offset for l1 preload
    if (unlikely(taskId == 0)) {
        runInfo.commonRunInfo.queryOffset = this->GetQueryOffset(runInfo);
        runInfo.dyOffset = runInfo.commonRunInfo.queryOffset;
        if constexpr (IS_D_NO_EQUAL) {
            runInfo.dyOffset = this->GetDxOffset(runInfo);
        }
    } else {
        runInfo.commonRunInfo.queryOffset = this->preloadArgs.nextQueryOffset;
        runInfo.dyOffset = this->preloadArgs.nextDyOffset;
    }
    GetNextDxAndQueryOffsetTND(runInfo, nextIndex, this->preloadArgs); // get nextQueryOffset, nextDyOffset, nextMorN
    runInfo.commonRunInfo.keyOffset = this->GetKeyOffset(runInfo);
    runInfo.commonRunInfo.valueOffset = runInfo.commonRunInfo.keyOffset;
    if constexpr (IS_D_NO_EQUAL) {
        runInfo.commonRunInfo.valueOffset = this->GetValueOffset(runInfo);
    }
    if ASCEND_IS_AIC {
        runInfo.queryOffsetWithRope = runInfo.commonRunInfo.queryOffset;
        runInfo.keyOffsetWithRope = runInfo.commonRunInfo.keyOffset;
        // Rope场景后面三个mm的GM offset不能和前面两个mm共用，因此需要重新计算
        if constexpr (IS_ROPE) {
            runInfo.queryOffsetWithRope = this->template GetQueryOffset<false>(runInfo);
            runInfo.keyOffsetWithRope = this->template GetKeyOffset<false>(runInfo);
            runInfo.queryOffsetWithRopeForMm12 = this->template GetQueryOffset<true>(runInfo);
            runInfo.keyOffsetWithRopeForMm12 = this->template GetKeyOffset<true>(runInfo);
            runInfo.commonRunInfo.qRopeOffset = this->GetQueryRopeOffset(runInfo);
            runInfo.commonRunInfo.kRopeOffset = this->GetKeyRopeOffset(runInfo);
        }
    }
    //----------------------------------------------PART
    if constexpr (SPLIT_AXIS == BN2S2) {
        runInfo.specialS2Index = SpecialS2Index(runInfo.commonRunInfo.keyOffset);
        runInfo.isFirstBlock = isFirstBlock;
        isFirstBlock = false;
    }

    if (unlikely(this->constInfo.isSink)) {
        runInfo.sinkN1Idx =
            runInfo.commonRunInfo.n2oIdx * this->constInfo.commonConstInfo.gSize + runInfo.commonRunInfo.goIdx;
        uint64_t s1oIdxSink =
            this->vSubBlockIdx == 1 ? runInfo.commonRunInfo.s1oIdx * 2 + 1 : runInfo.commonRunInfo.s1oIdx * 2;
        // [N, B, S1, S2]
        runInfo.dsinkWorkSpaceOffset =
            runInfo.sinkN1Idx * this->constInfo.bSize * this->constInfo.s1SinkOuter * this->constInfo.s2SinkOuter +
            runInfo.commonRunInfo.boIdx * this->constInfo.s1SinkOuter * this->constInfo.s2SinkOuter +
            s1oIdxSink * this->constInfo.s2SinkOuter + runInfo.s2oIdx;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t FlashAttentionScoreGradKernelDeter<CubeBlockType, VecBlockType>::CalCausalDeterIndex(
    uint32_t roundId, CoordinateInfo &coordinateInfo)
{
    int64_t j = this->cBlockIdx + 1;
    int64_t r = roundId + 1;
    int64_t k = static_cast<int64_t>(this->tilingData->s1s2BNGS1S2BaseParams.coreNum / NUM_TWO);
    int64_t mGap = 0;
    bool useEven = false;

    if constexpr (!BaseClass::IS_TND) {
        int64_t b = this->constInfo.bSize * this->constInfo.n2Size;
        int64_t m = this->constInfo.s1Outer;
        int64_t n = this->constInfo.s2Outer;
        TransTilingSplitMode<BaseClass::DETER_TILING_SPLIT_MODE>(m, n, r, useEven, deterLoopMax / NUM_TWO);
        if (this->constInfo.sparseMode == RIGHT_DOWN_CAUSAL && m < n) {
            return -1;
        }

        if (this->constInfo.sparseMode == RIGHT_DOWN_CAUSAL && m > n) {
            mGap = (this->constInfo.commonConstInfo.s1Size - this->constInfo.commonConstInfo.s2Size) /
                   BaseClass::DETER_CUBE_BASEM;
            m -= mGap;
        } else if ((this->constInfo.sparseMode == NO_MASK || this->constInfo.sparseMode == LEFT_UP_CAUSAL) && n > m) {
            n = m;
        }
        if constexpr (BaseClass::IS_N_EQUAL) {
            if (unlikely(this->tilingData->s1s2BNGS1S2BaseParams.isSplitByBlockIdx)) {
                if (this->constInfo.sparseMode == LEFT_UP_CAUSAL || this->constInfo.sparseMode == NO_MASK) {
                    CalLeftUpCausalSwizzleIndex(k, m, n, b, j, r, coordinateInfo);
                } else {
                    CalCausalSwizzleIndex(k, m, n, b, j, r, coordinateInfo);
                }
            } else {
                CalCausalIndex(k, m, n, b, j, r, coordinateInfo);
            }
        } else {
            CalGQACausalIndex(k, m, n, b, j, r, this->constInfo.commonConstInfo.gSize, coordinateInfo);
        }
    } else {
        TransDeterRound<BaseClass::DETER_TILING_SPLIT_MODE>(r, useEven, deterLoopMax / NUM_TWO);
        if constexpr (BaseClass::IS_N_EQUAL) {
            if (this->tilingData->deterParam.tndLineDeter != 0) {
                CalTNDCausalLineIndex<BaseClass::DETER_CUBE_BASEM, BaseClass::DETER_CUBE_BASEN>(
                    this->actualSeqQlenAddr, this->actualSeqKvlenAddr, this->tilingData->deterParam.deterPrefix0,
                    this->constInfo.bSize, this->constInfo.n2Size, k, j, r,
                    this->tilingData->deterParam.deterPrefixStep, this->constInfo.sparseMode, coordinateInfo);
            } else {
                CalTNDCausalIndex<BaseClass::DETER_CUBE_BASEM, BaseClass::DETER_CUBE_BASEN>(
                    this->actualSeqQlenAddr, this->actualSeqKvlenAddr, this->tilingData->deterParam.deterPrefix0,
                    this->tilingData->deterParam.deterPrefix1, this->tilingData->deterParam.deterPrefix2,
                    this->constInfo.bSize, this->constInfo.n2Size, k, j, r,
                    this->tilingData->deterParam.deterPrefixStep, coordinateInfo);
            }
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
    // [64, 128]场景特化
    TransTilingSplitModeBack<BaseClass::DETER_TILING_SPLIT_MODE, BaseClass::IS_TND>(coordinateInfo, useEven);
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
    uint32_t roundId, CoordinateInfo &coordinateInfo)
{
    int64_t j = this->cBlockIdx + 1;
    int64_t r = roundId + 1;

    if constexpr (BaseClass::IS_TND) {
        if constexpr (IS_TND_SWIZZLE) {
            CalTNDDenseSwizzleIndex<BaseClass::CUBE_BASEM, BaseClass::CUBE_BASEN>(
                this->actualSeqQlenAddr, this->actualSeqKvlenAddr,
                this->tilingData->tndSwizzleParam.tndS2BlockPrefixSum, this->constInfo.bSize, this->constInfo.n2Size,
                this->constInfo.commonConstInfo.gSize, j,
                static_cast<int64_t>(this->tilingData->s1s2BNGS1S2BaseParams.coreNum / NUM_TWO), r, coordinateInfo);
        } else {
            int64_t b = this->constInfo.bSize;
            CalTNDDenseIndex<BaseClass::CUBE_BASEM, BaseClass::CUBE_BASEN, BaseClass::DETER_SPARSE_TYPE,
                             BaseClass::IS_N_EQUAL>(
                this->actualSeqQlenAddr, this->actualSeqKvlenAddr, this->tilingData->deterParam.deterPrefix0,
                this->tilingData->baseDeterParam.deterMaxRound, this->constInfo.bSize, this->constInfo.n2Size,
                this->constInfo.commonConstInfo.gSize, j, r, 0, this->tilingData->deterParam.deterPrefixStep,
                coordinateInfo);
        }
    } else {
        int64_t k = static_cast<int64_t>(this->tilingData->s1s2BNGS1S2BaseParams.coreNum / NUM_TWO);
        int64_t b = this->constInfo.bSize * this->constInfo.n2Size;
        if constexpr (BaseClass::IS_N_EQUAL) {
            if (unlikely(this->tilingData->s1s2BNGS1S2BaseParams.isSplitByBlockIdx)) {
                CalDenseSwizzleIndex(k, this->constInfo.s1Outer, this->constInfo.s2Outer, b, j, r, coordinateInfo);
            } else {
                CalDenseIndex(k, this->constInfo.s1Outer, this->constInfo.s2Outer, b, j, r, coordinateInfo);
            }
        } else {
            int64_t denseRound = this->tilingData->baseDeterParam.deterMaxRound / this->constInfo.s1Outer;
            CalGQADenseIndex(k, this->constInfo.s1Outer, this->constInfo.s2Outer, b, j, r,
                             this->constInfo.commonConstInfo.gSize, coordinateInfo, denseRound);
        }
    }

    int64_t w = coordinateInfo.batchId;
    int64_t n1 = this->constInfo.commonConstInfo.gSize * this->constInfo.n2Size;
    if ((!BaseClass::IS_TND || this->constInfo.commonConstInfo.gSize == 1) && !IS_TND_SWIZZLE) {
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
    uint32_t roundId, CoordinateInfo &coordinateInfo)
{
    int64_t j = this->cBlockIdx + 1;
    int64_t r = roundId + 1;
    bool useEven = false;
    TransDeterRound<BaseClass::DETER_TILING_SPLIT_MODE>(r, useEven, deterLoopMax / NUM_TWO);
    if constexpr (!BaseClass::IS_TND) {
        if constexpr (BaseClass::IS_N_EQUAL) {
            int64_t b = this->constInfo.bSize * this->constInfo.n2Size;
            if (unlikely(this->tilingData->s1s2BNGS1S2BaseParams.isSplitByBlockIdx)) {
                int64_t k = static_cast<int64_t>(this->tilingData->s1s2BNGS1S2BaseParams.coreNum / NUM_TWO);
                int64_t m = this->bandInfo.m;
                int64_t n = this->bandInfo.n;
                uint8_t scheduleMode = this->tilingData->s1s2BNGS1S2BaseParams.deterBandScheduleMode;
                if (scheduleMode == static_cast<uint8_t>(optiling::fag::DeterBandScheduleMode::CAUSAL)) {
                    int64_t causalShift = this->bandInfo.q - 1;
                    int64_t causalSize = m + causalShift;
                    int64_t causalPairCount = b / NUM_TWO;
                    int64_t pairedBatch = causalPairCount * NUM_TWO;
                    int64_t causalK = Min(k, causalSize * causalPairCount);
                    int64_t pairRound = causalPairCount > 0 ?
                                            Ceil<int64_t>((causalSize + 1) * causalPairCount, causalK) * causalSize :
                                            0;
                    if ((b & 1) != 0 && r > pairRound) {
                        CalCausalSingleBatchDeterIndex(k, causalSize, causalSize, j, r - pairRound, coordinateInfo);
                        if (coordinateInfo.batchId != -1) {
                            coordinateInfo.batchId = b;
                        }
                    } else {
                        CalCausalSwizzleIndex(k, causalSize, causalSize, pairedBatch, j, r, coordinateInfo);
                    }
                    if (coordinateInfo.batchId != -1) {
                        coordinateInfo.s1Idx -= causalShift;
                        if (coordinateInfo.s1Idx < 1 || coordinateInfo.s1Idx > m || coordinateInfo.s2Idx > n) {
                            coordinateInfo.batchId = -1;
                        }
                    }
                } else if (scheduleMode == static_cast<uint8_t>(optiling::fag::DeterBandScheduleMode::DENSE)) {
                    CalDenseSwizzleIndex(k, m, n, b, j, r, coordinateInfo);
                } else if (scheduleMode == static_cast<uint8_t>(optiling::fag::DeterBandScheduleMode::BAND)) {
                    CalBandHybridIndex(this->bandInfo, j, r, coordinateInfo);
                } else {
                    CalCausalSwizzleIndex(k, m, n, b, j, r, coordinateInfo);
                }
            } else {
                CalBandIndex(this->bandInfo, j, r, coordinateInfo);
            }
        } else {
            CalGQABandIndex(this->bandInfo, j, r, this->constInfo.commonConstInfo.gSize, coordinateInfo);
        }
    } else {
        int64_t k = static_cast<int64_t>(this->tilingData->s1s2BNGS1S2BaseParams.coreNum / NUM_TWO);
        coordinateInfo.p = this->constInfo.s1Token;
        coordinateInfo.q = this->constInfo.s2Token;
        if constexpr (BaseClass::IS_N_EQUAL) {
            if (this->tilingData->deterParam.tndLineDeter != 0) {
                CalTNDBandLineIndex<BaseClass::DETER_CUBE_BASEM, BaseClass::DETER_CUBE_BASEN>(
                    this->actualSeqQlenAddr, this->actualSeqKvlenAddr, this->tilingData->deterParam.deterPrefix0,
                    this->constInfo.bSize, this->constInfo.n2Size, k, j, r,
                    this->tilingData->deterParam.deterPrefixStep, coordinateInfo);
            } else {
                CalTNDBandIndex<BaseClass::DETER_CUBE_BASEM, BaseClass::DETER_CUBE_BASEN>(
                    this->actualSeqQlenAddr, this->actualSeqKvlenAddr, this->tilingData->deterParam.deterPrefix0,
                    this->tilingData->deterParam.deterPrefix1, this->constInfo.bSize, this->constInfo.n2Size, k, j, r,
                    this->tilingData->deterParam.deterPrefixStep, coordinateInfo);
            }
        } else {
            CalTNDGQABandIndex<BaseClass::DETER_CUBE_BASEM, BaseClass::DETER_CUBE_BASEN>(
                this->actualSeqQlenAddr, this->actualSeqKvlenAddr, this->tilingData->deterParam.deterPrefix0,
                this->tilingData->deterParam.deterPrefix1, this->constInfo.bSize, this->constInfo.commonConstInfo.gSize,
                this->constInfo.n2Size, k, j, r, this->tilingData->deterParam.deterPrefixStep, coordinateInfo);
        }
    }
    // [64, 128]场景特化
    int64_t convertedMOffset = coordinateInfo.mOffset;
    int64_t convertedNOffset = coordinateInfo.nOffset;
    if constexpr (BaseClass::DETER_TILING_SPLIT_MODE == 1) {
        convertedMOffset = coordinateInfo.mOffset * NUM_TWO;
    }
    TransTilingSplitModeBack<BaseClass::DETER_TILING_SPLIT_MODE, BaseClass::IS_TND>(coordinateInfo, useEven);
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
__aicore__ inline int64_t FlashAttentionScoreGradKernelDeter<CubeBlockType, VecBlockType>::CalDeterMaxLoopNum()
{
    int64_t b = this->constInfo.bSize * this->constInfo.n2Size;
    int64_t m = this->constInfo.s1Outer;
    int64_t n = this->constInfo.s2Outer;
    int64_t k = static_cast<int64_t>(this->tilingData->s1s2BNGS1S2BaseParams.coreNum / NUM_TWO);
    deterLoopMax = 0;
    if constexpr (BaseClass::IS_TND) {
        if constexpr (IS_TND_SWIZZLE) {
            return this->tilingData->tndSwizzleParam.tndS2BlockPrefixSum[this->constInfo.bSize];
        } else {
            return this->tilingData->baseDeterParam.deterMaxRound;
        }
    }

    InitCoordinateInfo(this->constInfo.s1Outer, this->constInfo.s2Outer, 0, 0, this->coordinateInfos[0]);
    InitCoordinateInfo(this->constInfo.s1Outer, this->constInfo.s2Outer, 0, 0, this->coordinateInfos[1]);
    if constexpr (BaseClass::DETER_SPARSE_TYPE == DETER_CAUSAL) {
        if (unlikely(this->tilingData->s1s2BNGS1S2BaseParams.isSplitByBlockIdx)) {
            if (this->constInfo.sparseMode == LEFT_UP_CAUSAL || this->constInfo.sparseMode == NO_MASK) {
                return CalLeftUpCausalSwizzleMaxRound(k, m, n, b);
            }
            return Max(this->constInfo.s1Outer * Ceil<int64_t>((n + 1) * (b >> 1), k), n + 1);
        }
        return this->tilingData->baseDeterParam.deterMaxRound;
    }

    if constexpr (BaseClass::DETER_SPARSE_TYPE == DETER_DENSE) {
        if constexpr (BaseClass::IS_N_EQUAL) {
            return Ceil<int64_t>(n * b, Min(k, m * b)) * m;
        } else {
            if (this->tilingData->baseDeterParam.deterMaxRound > 0) {
                return this->tilingData->baseDeterParam.deterMaxRound;
            }
            return Max(Max(Ceil<int64_t>(b * n * this->constInfo.commonConstInfo.gSize,
                                         (Min(Min(k, b * this->constInfo.commonConstInfo.gSize * m), b * n))),
                           Ceil<int64_t>(n, m)),
                       this->constInfo.commonConstInfo.gSize) *
                   m;
        }
    }

    if constexpr (BaseClass::DETER_SPARSE_TYPE == DETER_BAND) {
        if constexpr (BaseClass::IS_N_EQUAL) {
            uint8_t scheduleMode = this->tilingData->s1s2BNGS1S2BaseParams.deterBandScheduleMode;
            if (unlikely(this->tilingData->s1s2BNGS1S2BaseParams.isSplitByBlockIdx &&
                         scheduleMode == static_cast<uint8_t>(optiling::fag::DeterBandScheduleMode::DISABLED) &&
                         this->constInfo.sparseMode == RIGHT_DOWN_CAUSAL)) {
                int64_t mNew = 0;
                int64_t nNew = 0;
                if (m < n) {
                    mNew = m;
                    nNew = (n << 1) - m + NUM_THREE;
                    this->bandInfo.m = m;
                    this->bandInfo.n = n;
                } else {
                    mNew = n + 1;
                    nNew = n + NUM_TWO;
                    this->bandInfo.m = mNew;
                    this->bandInfo.n = mNew;
                    InitCoordinateInfo(this->constInfo.s1Outer, this->constInfo.s2Outer, m - n - 1, 0,
                                       this->coordinateInfos[0]);
                    InitCoordinateInfo(this->constInfo.s1Outer, this->constInfo.s2Outer, m - n - 1, 0,
                                       this->coordinateInfos[1]);
                }
                return Max(mNew * Ceil<int64_t>(nNew * (b >> 1), k), nNew);
            }
        }
        int64_t s1Outer = this->constInfo.s1Outer;
        int64_t s2Outer = this->constInfo.s2Outer;
        if constexpr (BaseClass::DETER_TILING_SPLIT_MODE == 1) {
            m = Ceil<int64_t>(m, static_cast<int64_t>(NUM_TWO));
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
            uint8_t scheduleMode = this->tilingData->s1s2BNGS1S2BaseParams.deterBandScheduleMode;
            // The Host sets a non-DISABLED mode only after DETER_BAND normalization. Keep this mode
            // authoritative for both hybrid-info initialization and coordinate dispatch.
            bool isHybridSchedule =
                scheduleMode >= static_cast<uint8_t>(optiling::fag::DeterBandScheduleMode::CAUSAL) &&
                scheduleMode <= static_cast<uint8_t>(optiling::fag::DeterBandScheduleMode::BAND);
            if (unlikely(isHybridSchedule)) {
                GenBandHybridInfo(k, actualM, actualN, actualP, actualQ, b, this->bandInfo);
                InitCoordinateInfo(this->constInfo.s1Outer, this->constInfo.s2Outer, mOffset, nOffset,
                                   this->coordinateInfos[0]);
                InitCoordinateInfo(this->constInfo.s1Outer, this->constInfo.s2Outer, mOffset, nOffset,
                                   this->coordinateInfos[1]);
                if (scheduleMode == static_cast<uint8_t>(optiling::fag::DeterBandScheduleMode::CAUSAL)) {
                    // The odd-batch CAUSAL schedule has a paired section followed by a single-batch
                    // tail. Host serializes the sum of both sections, so use it as the sole bound.
                    deterLoopMax = this->tilingData->baseDeterParam.deterMaxRound;
                } else if (scheduleMode == static_cast<uint8_t>(optiling::fag::DeterBandScheduleMode::DENSE)) {
                    int64_t denseK = Min(k, this->bandInfo.m * b);
                    deterLoopMax = Ceil<int64_t>(this->bandInfo.n * b, denseK) * this->bandInfo.m;
                } else {
                    int64_t colsPerBatch = this->bandInfo.nSeg - this->bandInfo.hybridPairCount;
                    deterLoopMax = Ceil<int64_t>(b * colsPerBatch, k) * (this->bandInfo.p + this->bandInfo.q - 1);
                }
                return deterLoopMax;
            }
            GenBandInfo(k, actualM, actualN, actualP, actualQ, b, this->bandInfo);
            int64_t convertedMaxRound = BaseClass::DETER_TILING_SPLIT_MODE ?
                                            this->tilingData->baseDeterParam.deterMaxRound / NUM_TWO :
                                            this->tilingData->baseDeterParam.deterMaxRound;
            deterLoopMax = Max(convertedMaxRound, this->bandInfo.rm2);
        } else {
            k = Min(Min(k, b * this->constInfo.commonConstInfo.gSize * m), b * n);
            int64_t b2 = b % k;
            GenGQABandInfo(k, actualM, actualN, actualP, actualQ, b, this->constInfo.commonConstInfo.gSize,
                           this->bandInfo);
            deterLoopMax = this->bandInfo.rm + this->bandInfo.rm2;
        }
        InitCoordinateInfo(this->constInfo.s1Outer, this->constInfo.s2Outer, mOffset, nOffset,
                           this->coordinateInfos[0]);
        InitCoordinateInfo(this->constInfo.s1Outer, this->constInfo.s2Outer, mOffset, nOffset,
                           this->coordinateInfos[1]);
        if constexpr (BaseClass::DETER_TILING_SPLIT_MODE) {
            deterLoopMax *= NUM_TWO;
        }
        return deterLoopMax;
    }
    return deterLoopMax;
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
__aicore__ inline void FlashAttentionScoreGradKernelDeter<CubeBlockType, VecBlockType>::CalDeterIndex(
    uint32_t roundId, uint32_t maxLoopNum, int64_t &nextValidRoundId, int64_t &nextValidIndex, int64_t taskId,
    CoordinateInfo &coordinateInfo, FagRunInfo &runInfo)
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
__aicore__ inline void FlashAttentionScoreGradKernelDeter<CubeBlockType, VecBlockType>::DeterSync(int64_t loopIdx)
{
    if (loopIdx == 0) {
        return;
    }

    // 此处复用dqIsNeedDeter和dkDvIsNeedDeter装载需要同步的轮次范围
    for (int8_t i = 0; i < MAX_CUBE_CORE_NUM; i++) {
        if (this->tilingData->baseDeterParam.dkDvIsNeedDeter[i] == 0) {
            return;
        }
        if (static_cast<uint64_t>(loopIdx) >= this->tilingData->baseDeterParam.dqIsNeedDeter[i] &&
            static_cast<uint64_t>(loopIdx) <= this->tilingData->baseDeterParam.dkDvIsNeedDeter[i]) {
            CrossCoreSetFlag<0, PIPE_FIX>(SYNC_DETER_FIX_FLAG);
            CrossCoreWaitFlag<0, PIPE_FIX>(SYNC_DETER_FIX_FLAG);
            return;
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelDeter<CubeBlockType, VecBlockType>::Process_NEW_DETER()
{
    if constexpr (SPLIT_AXIS == BN2S2) {
        if ASCEND_IS_AIV {
            InitOutput<float>(
                this->deterGm[deterGmOffset + this->vSubBlockIdx * this->CUBE_BASEN * this->HEAD_DIM_ALIGN],
                this->CUBE_BASEN * this->HEAD_DIM_ALIGN, 0);
        }
    }
    deterLoopMax = CalDeterMaxLoopNum();
    int64_t blockInnerIdx = 0;
    int64_t taskId = 0;
    int64_t nextValidLoopIdx;
    int64_t nextblockIdx;
    bool needSyncDkMM = false;
    dkvWorkSpaceOffet = this->cBlockIdx * this->CUBE_BASEN * this->HEAD_DIM_ALIGN;
    dAlign16 = AlignTo16(this->constInfo.commonConstInfo.dSize);
    dvAlign16 = AlignTo16(this->constInfo.commonConstInfo.dSizeV);
    LocalTensor<CALC_TYPE> mm1ResTensor;
    LocalTensor<CALC_TYPE> mm2ResTensor;

    FagRunInfo runInfos[2]; // for ping pongs
    CalDeterIndex(0, deterLoopMax, nextValidLoopIdx, nextblockIdx, taskId, this->coordinateInfos[taskId & 1],
                  runInfos[taskId & 1]);

    for (int64_t loopIdx = 0; loopIdx < deterLoopMax + 1; loopIdx++) {
        if (loopIdx >= nextValidLoopIdx) {
            blockInnerIdx = nextblockIdx;
            CalDeterIndex(loopIdx + 1, deterLoopMax, nextValidLoopIdx, nextblockIdx, taskId,
                          this->coordinateInfos[(taskId + 1) & 1], runInfos[(taskId + 1) & 1]);
        } else {
            blockInnerIdx = -1;
        }
        bool isValidBlock = (blockInnerIdx >= 0);
        if constexpr (BaseClass::IS_TND) {
            isValidBlock = isValidBlock && !(this->coordinateInfos[taskId & 1].batchId < 0 || blockInnerIdx < 0);
            nextblockIdx = nextblockIdx < 0 ? nextblockIdx : ((taskId + 1) & 1);
        } else {
            isValidBlock = isValidBlock && this->IsValidForDeter(runInfos[taskId & 1], taskId, blockInnerIdx);
        }

        if (!(runInfos[(taskId + 1) & 1].completed)) {
            if (likely(!this->constInfo.enablePreSfmg)) {
                this->vecBlock.ProcessVec1(this->constInfo, runInfos[(taskId + 1) & 1]); // v1: softmaxGrad
            }
            // wait mm1 and mm2 result
            if ASCEND_IS_AIV {
                CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C1_TO_V2_FLAG[(taskId + 1) & 1]);
                CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C2_TO_V2_FLAG[(taskId + 1) & 1]);
            }
        }
        if (isValidBlock) {
            if constexpr (!BaseClass::IS_TND) {
                runInfos[taskId & 1].s2CvBegin = this->s2CvBegin;
                runInfos[taskId & 1].s2CvEnd = this->s2CvEnd;
            }

            if constexpr (BaseClass::IS_TND) {
                SetRunInfoDeterForTND(runInfos[taskId & 1], taskId, blockInnerIdx, coordinateInfos[taskId & 1],
                                      nextblockIdx);
            } else {
                this->SetRunInfo(runInfos[taskId & 1], runInfos[(taskId + 1) & 1], taskId, blockInnerIdx, nextblockIdx);
            }

            mm1ResTensor = this->mm1ResBuf[runInfos[taskId & 1].commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
            this->cubeBlock.IterateMmDyV(mm1ResTensor, this->constInfo, runInfos[taskId & 1], this->preloadArgs);
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C1_TO_V2_FLAG[taskId & 1]);
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C1_TO_V2_FLAG[taskId & 1]);
            }

            mm2ResTensor = this->mm2ResBuf[runInfos[taskId & 1].commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
            this->cubeBlock.IterateMmQK(mm2ResTensor, this->constInfo, runInfos[taskId & 1], this->preloadArgs);
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C2_TO_V2_FLAG[taskId & 1]);
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C2_TO_V2_FLAG[taskId & 1]);
            }

            this->vecBlock.CopyMaxSum(this->constInfo, runInfos[taskId & 1],
                                      taskId); // copy in max and sum double buffer
            runInfos[taskId & 1].completed = false;
        }

        if (!(runInfos[(taskId + 1) & 1].completed)) {
            mm1ResTensor =
                this->mm1ResBuf[runInfos[(taskId + 1) & 1].commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
            mm2ResTensor =
                this->mm2ResBuf[runInfos[(taskId + 1) & 1].commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();

            this->vecBlock.ProcessVec2(mm2ResTensor, this->constInfo,
                                       runInfos[(taskId + 1) & 1]); // v2: pse + attenMask + simpleSoftmax
            if (unlikely(this->constInfo.isSink && !IS_DROP)) {
                this->vecBlock.ProcessVecSink(mm1ResTensor, mm2ResTensor, this->constInfo, runInfos[(taskId + 1) & 1]);
            }
            if ASCEND_IS_AIV {
                if (needSyncDkMM) {
                    CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_C4_TO_V3_FLAG);
                }
            }

            MutexBuffer<BufferType::L1, SyncType::NO_SYNC> dSL1Buffer = this->dSL1Buf.Get();
            MutexBuffer<BufferType::L1, SyncType::NO_SYNC> pL1Buffer = this->pL1Buf.Get();
            this->vecBlock.ProcessVec3(dSL1Buffer, mm1ResTensor, mm2ResTensor, this->constInfo,
                                       runInfos[(taskId + 1) & 1]); // v3: dropout + cast + nd2nz
            if (unlikely(this->constInfo.isSink && IS_DROP)) {
                this->vecBlock.ProcessVecSink(mm1ResTensor, mm2ResTensor, this->constInfo, runInfos[(taskId + 1) & 1]);
            }

            if ASCEND_IS_AIV {
                if (needSyncDkMM) {
                    CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_C5_TO_V4_FLAG);
                }
            }
            this->vecBlock.ProcessVec4(pL1Buffer, mm2ResTensor, this->constInfo,
                                       runInfos[(taskId + 1) & 1]); // v4: cast + nd2nz

            if ASCEND_IS_AIV {
                CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V3_TO_C3_FLAG); // dqk must wait ds copy completely
                CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V4_TO_C5_FLAG); // dv must wait ds copy completely
            }

            if ASCEND_IS_AIC {
                // wait ds in ub copy to l1
                CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SYNC_V3_TO_C3_FLAG);
                CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_V3_TO_C3_FLAG);
                // wait p in ub copy to l1
                CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SYNC_V4_TO_C5_FLAG);
                CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_V4_TO_C5_FLAG);
            }

            if constexpr (SPLIT_AXIS == BN2S2) {
                this->cubeBlock.template IterateMmDsK<CALC_TYPE, BaseClass::IS_DQ_WRITE_UB>(
                    this->dqWorkSpaceGm, dSL1Buffer, this->constInfo,
                    runInfos[(taskId + 1) & 1]); // c3
                // compute dk
                if (runInfos[(taskId + 1) & 1].specialS2Index != -1) {
                    this->cubeBlock.template IterateMmDsQ<CALC_TYPE, BaseClass::IS_DK_WRITE_UB>(
                        this->deterGm, dSL1Buffer, this->constInfo,
                        runInfos[(taskId + 1) & 1]); // c4
                } else {
                    this->cubeBlock.template IterateMmDsQ<CALC_TYPE, BaseClass::IS_DK_WRITE_UB>(
                        this->dkWorkSpaceGm, dSL1Buffer, this->constInfo,
                        runInfos[(taskId + 1) & 1]); // c4
                }
                if (runInfos[(taskId + 1) & 1].specialS2Index == -1 &&
                    !runInfos[(taskId + 1) & 1].isNextS2IdxNoChange) {
                    if ASCEND_IS_AIC {
                        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C4_TO_V6_FLAG);
                        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C4_TO_V6_FLAG);
                    } else {
                        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SYNC_C4_TO_V6_FLAG);
                    }
                    this->vecBlock.template ProcessMulsAndCast<CALC_TYPE, BaseClass::IS_DK_WRITE_UB, DK_IDX>(
                        this->dkWorkSpaceGm, this->constInfo, runInfos[(taskId + 1) & 1]); // v5: dk muls + cast
                }
                if ASCEND_IS_AIC {
                    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_C4_TO_V3_FLAG);
                    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_C4_TO_V3_FLAG);
                }
                // ------ deter 特殊部分 START ------
                if (this->cBlockIdx == runInfos[(taskId + 1) & 1].specialS2Index) {
                    specialDkGmOffset = runInfos[(taskId + 1) & 1].commonRunInfo.keyOffset;
                    specialHalfS2RealSize = runInfos[(taskId + 1) & 1].halfS2RealSize;
                    specialFirstHalfS2RealSize = runInfos[(taskId + 1) & 1].firstHalfS2RealSize;
                }
                // ------ deter 特殊部分 END ------
                // compute dv
                if (runInfos[(taskId + 1) & 1].specialS2Index != -1) {
                    this->cubeBlock.template IterateMmPDy<CALC_TYPE, BaseClass::IS_DV_WRITE_UB>(
                        this->deterGm, pL1Buffer, this->constInfo, runInfos[(taskId + 1) & 1]); // c5
                } else {
                    this->cubeBlock.template IterateMmPDy<CALC_TYPE, BaseClass::IS_DV_WRITE_UB>(
                        this->dvWorkSpaceGm, pL1Buffer, this->constInfo, runInfos[(taskId + 1) & 1]); // c5
                }

                if (runInfos[(taskId + 1) & 1].specialS2Index == -1 &&
                    !runInfos[(taskId + 1) & 1].isNextS2IdxNoChange) {
                    if ASCEND_IS_AIC {
                        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C3_TO_V5_FLAG);
                        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C3_TO_V5_FLAG);
                    } else {
                        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SYNC_C3_TO_V5_FLAG);
                    }
                    this->vecBlock.template ProcessMulsAndCast<CALC_TYPE, BaseClass::IS_DV_WRITE_UB, DV_IDX>(
                        this->dvWorkSpaceGm, this->constInfo, runInfos[(taskId + 1) & 1]); // v6: dv muls + cast
                }
                if ASCEND_IS_AIC {
                    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_C5_TO_V4_FLAG);
                    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_C5_TO_V4_FLAG);
                }
                needSyncDkMM = true;
                // ------ deter 特殊部分 START ------
                if (static_cast<int64_t>(this->cBlockIdx) ==
                    static_cast<int64_t>(runInfos[(taskId + 1) & 1].specialS2Index)) {
                    specialDvGmOffset = runInfos[(taskId + 1) & 1].commonRunInfo.valueOffset;
                }
                // ------ deter 特殊部分 END ------
            } else {
                if ASCEND_IS_AIC {
                    CrossCoreWaitFlag<0, PIPE_FIX>(SYNC_DETER_FIX_FLAG);
                }
                // compute dq
                this->cubeBlock.template IterateMmDsK<CALC_TYPE, BaseClass::IS_DQ_WRITE_UB>(
                    this->dqWorkSpaceGm, dSL1Buffer, this->constInfo,
                    runInfos[(taskId + 1) & 1]); // c3
                if ASCEND_IS_AIC {
                    CrossCoreSetFlag<0, PIPE_FIX>(SYNC_DETER_FIX_FLAG);
                    CrossCoreWaitFlag<0, PIPE_FIX>(SYNC_DK_DETER_FIX_FLAG);
                }
                // compute dk
                this->cubeBlock.template IterateMmDsQ<CALC_TYPE, BaseClass::IS_DK_WRITE_UB>(
                    this->dkWorkSpaceGm, dSL1Buffer, this->constInfo,
                    runInfos[(taskId + 1) & 1]); // c4
                if ASCEND_IS_AIC {
                    CrossCoreSetFlag<0, PIPE_FIX>(SYNC_DK_DETER_FIX_FLAG);
                    CrossCoreWaitFlag<0, PIPE_FIX>(SYNC_DV_DETER_FIX_FLAG);

                    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_C4_TO_V3_FLAG);
                    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_C4_TO_V3_FLAG);
                }
                // compute dv
                this->cubeBlock.template IterateMmPDy<CALC_TYPE, BaseClass::IS_DV_WRITE_UB>(
                    this->dvWorkSpaceGm, pL1Buffer, this->constInfo, runInfos[(taskId + 1) & 1]); // c5
                if ASCEND_IS_AIC {
                    CrossCoreSetFlag<0, PIPE_FIX>(SYNC_DV_DETER_FIX_FLAG);

                    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_C5_TO_V4_FLAG);
                    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_C5_TO_V4_FLAG);
                }
                needSyncDkMM = true;
            }
            runInfos[(taskId + 1) & 1].completed = true;
        } else {
            if constexpr (SPLIT_AXIS != BN2S2) {
                if ASCEND_IS_AIC {
                    if (loopIdx > 0) {
                        CrossCoreWaitFlag<0, PIPE_FIX>(SYNC_DETER_FIX_FLAG);
                        CrossCoreSetFlag<0, PIPE_FIX>(SYNC_DETER_FIX_FLAG);
                        CrossCoreWaitFlag<0, PIPE_FIX>(SYNC_DK_DETER_FIX_FLAG);
                        CrossCoreSetFlag<0, PIPE_FIX>(SYNC_DK_DETER_FIX_FLAG);
                        CrossCoreWaitFlag<0, PIPE_FIX>(SYNC_DV_DETER_FIX_FLAG);
                        CrossCoreSetFlag<0, PIPE_FIX>(SYNC_DV_DETER_FIX_FLAG);
                    } else {
                        CrossCoreSetFlag<0, PIPE_FIX>(SYNC_DETER_FIX_FLAG);
                        CrossCoreSetFlag<0, PIPE_FIX>(SYNC_DK_DETER_FIX_FLAG);
                        CrossCoreSetFlag<0, PIPE_FIX>(SYNC_DV_DETER_FIX_FLAG);
                    }
                }
            }
        }
        if constexpr (SPLIT_AXIS == BN2S2) {
            if ASCEND_IS_AIC {
                DeterSync(loopIdx);
            }
        }
        if (isValidBlock) {
            taskId++;
        }
    }
    if constexpr (SPLIT_AXIS != BN2S2) {
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag<0, PIPE_FIX>(SYNC_DETER_FIX_FLAG);
            CrossCoreWaitFlag<0, PIPE_FIX>(SYNC_DK_DETER_FIX_FLAG);
            CrossCoreWaitFlag<0, PIPE_FIX>(SYNC_DV_DETER_FIX_FLAG);
        }
    }
    if constexpr (SPLIT_AXIS == BN2S2) {
        SyncAll<false>();
        this->vecBlock.template ProcessPostDeter<true>(this->constInfo, this->deterGm[deterGmOffset], this->dkGm,
                                                       specialHalfS2RealSize, specialFirstHalfS2RealSize, dAlign16,
                                                       dvAlign16, specialDkGmOffset, specialDvGmOffset);
        this->vecBlock.template ProcessPostDeter<false>(
            this->constInfo, this->deterGm[deterGmOffset + this->CUBE_BASEN * this->HEAD_DIM_ALIGN], this->dvGm,
            specialHalfS2RealSize, specialFirstHalfS2RealSize, dAlign16, dvAlign16, specialDkGmOffset,
            specialDvGmOffset);
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
__aicore__ inline void FlashAttentionScoreGradKernelDeter<CubeBlockType, VecBlockType>::Process_OLD_DETER()
{
    // maxValidBBLen表示所有核中处理有效块最多的数量，需要做这么多次确定性计算
    int64_t maxValidBBLen = this->tilingData->s1s2BNGS1S2SplitCoreParams.maxValidBBLen;
    int64_t remainLoopNum = maxValidBBLen;
    int64_t blockInnerIdx = 0;
    int64_t nextValidBlockInnerIdx = 0;
    int64_t taskId = 0;
    int64_t loopIdx = 0;
    FagRunInfo runInfos[2]; // for ping pong
    FagRunInfo prevRunInfo;

    LocalTensor<CALC_TYPE> mm1ResTensor;
    LocalTensor<CALC_TYPE> mm2ResTensor;
    bool needSyncDkMM = false;

    while (remainLoopNum > 0) { // remainLoop每次做完确定性计算减1
        blockInnerIdx = this->tilingData->s1s2BNGS1S2BlockNumList.blockStarts[this->cBlockIdx] + loopIdx;
        if (this->tilingData->s1s2BNGS1S2BlockNumList.blockEnds[this->cBlockIdx] != 0) { // 有主流程需要处理的核
            if (blockInnerIdx < this->tilingData->s1s2BNGS1S2BlockNumList.blockEnds[this->cBlockIdx] +
                                    1) { // 这个1是提前发射mm1mm2，所以需要额外增加一个轮次
                this->isLastLoop =
                    (blockInnerIdx == this->tilingData->s1s2BNGS1S2BlockNumList.blockEnds[this->cBlockIdx]);
                // 无效块跳过
                if (!this->isLastLoop && !this->IsValid(runInfos[taskId & 1], taskId, blockInnerIdx)) {
                    loopIdx++;
                    continue;
                }
                if (taskId > 0) {
                    prevRunInfo = runInfos[(taskId + 1) & 1];
                    deterPpFlag = 1 - deterPpFlag;
                    // softmaxGrad
                    this->vecBlock.ProcessVec1(this->constInfo, runInfos[(taskId + 1) & 1]); // v1: softmaxGrad
                    // wait mm1 and mm2 result
                    if ASCEND_IS_AIV {
                        CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C1_TO_V2_FLAG[(taskId + 1) & 1]);
                        CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C2_TO_V2_FLAG[(taskId + 1) & 1]);
                    }
                }
                if (!this->isLastLoop) {
                    nextValidBlockInnerIdx =
                        this->GetNextValidIdx(runInfos[(taskId + 1) & 1], taskId + 1, blockInnerIdx + 1);
                    this->SetRunInfo(runInfos[taskId & 1], runInfos[(taskId + 1) & 1], taskId, blockInnerIdx,
                                     nextValidBlockInnerIdx);
                    // IterateMm1Mm2
                    if ASCEND_IS_AIC {
                        if (needSyncDkMM) {
                            CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SYNC_DETER_FIX_FLAG);
                            CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_DETER_FIX_FLAG);
                        }
                    }
                    mm1ResTensor =
                        this->mm1ResBuf[runInfos[taskId & 1].commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
                    this->cubeBlock.IterateMmDyV(mm1ResTensor, this->constInfo, runInfos[taskId & 1],
                                                 this->preloadArgs);
                    if ASCEND_IS_AIC {
                        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C1_TO_V2_FLAG[taskId & 1]);
                        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C1_TO_V2_FLAG[taskId & 1]);
                    }
                    mm2ResTensor =
                        this->mm2ResBuf[runInfos[taskId & 1].commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
                    this->cubeBlock.IterateMmQK(mm2ResTensor, this->constInfo, runInfos[taskId & 1], this->preloadArgs);
                    if ASCEND_IS_AIC {
                        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C2_TO_V2_FLAG[taskId & 1]);
                        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C2_TO_V2_FLAG[taskId & 1]);
                    }
                    if constexpr (IS_ROPE) {
                        this->vecBlock.WriteOffsetToGM(this->constInfo, runInfos[taskId & 1].queryOffsetWithRope,
                                                       runInfos[taskId & 1].keyOffsetWithRope,
                                                       runInfos[taskId & 1].commonRunInfo.valueOffset, taskId,
                                                       maxValidBBLen - taskId);
                    } else {
                        this->vecBlock.WriteOffsetToGM(this->constInfo, runInfos[taskId & 1].commonRunInfo.queryOffset,
                                                       runInfos[taskId & 1].commonRunInfo.keyOffset,
                                                       runInfos[taskId & 1].commonRunInfo.valueOffset, taskId,
                                                       maxValidBBLen - taskId);
                    }

                    // CopyInMaxSum
                    this->vecBlock.CopyMaxSum(this->constInfo, runInfos[taskId & 1],
                                              taskId); // copy in max and sum double buffer
                }
                if (taskId > 0) {
                    mm1ResTensor = this->mm1ResBuf[prevRunInfo.commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
                    mm2ResTensor = this->mm2ResBuf[prevRunInfo.commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
                    // ProcessReCompute
                    this->vecBlock.ProcessVec2(mm2ResTensor, this->constInfo,
                                               prevRunInfo); // v2: pse + attenMask + simpleSoftmax
                    if (unlikely(this->constInfo.isSink && !IS_DROP)) {
                        this->vecBlock.ProcessVecSink(mm1ResTensor, mm2ResTensor, this->constInfo,
                                                      runInfos[(taskId + 1) & 1]);
                    }
                    if ASCEND_IS_AIV {
                        if (needSyncDkMM) {
                            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_C4_TO_V3_FLAG);
                        }
                    }
                    MutexBuffer<BufferType::L1, SyncType::NO_SYNC> dSL1Buffer = this->dSL1Buf.Get();
                    MutexBuffer<BufferType::L1, SyncType::NO_SYNC> pL1Buffer = this->pL1Buf.Get();
                    // GetIsNeedDeter, vec3需要知道是否dqneedDeter
                    GetIsNeedDeter(taskId - 1);
                    this->vecBlock.ProcessVec3(dSL1Buffer, mm1ResTensor, mm2ResTensor, this->constInfo, prevRunInfo,
                                               dqIsNeedDeter[deterPpFlag]); // v3: dropout + cast + nd2nz
                    if (unlikely(this->constInfo.isSink && IS_DROP)) {
                        this->vecBlock.ProcessVecSink(mm1ResTensor, mm2ResTensor, this->constInfo,
                                                      runInfos[(taskId + 1) & 1]);
                    }
                    if ASCEND_IS_AIV {
                        if (needSyncDkMM) {
                            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_C5_TO_V4_FLAG);
                        }
                    }
                    this->vecBlock.ProcessVec4(pL1Buffer, mm2ResTensor, this->constInfo,
                                               prevRunInfo); // v4: cast + nd2nz
                    if ASCEND_IS_AIV {
                        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V3_TO_C3_FLAG); // dqk must wait ds copy completely
                        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V4_TO_C5_FLAG); // dv must wait ds copy completely
                    } else {
                        // wait ds in ub copy to l1
                        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SYNC_V3_TO_C3_FLAG);
                        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_V3_TO_C3_FLAG);
                        // wait p in ub copy to l1
                        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SYNC_V4_TO_C5_FLAG);
                        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_V4_TO_C5_FLAG);
                    }

                    // compute dq
                    if (likely(this->constInfo.deterConstInfo.noNeedDeter)) {
                        // c3-normal
                        this->cubeBlock.template IterateMmDsK<CALC_TYPE, BaseClass::IS_DQ_WRITE_UB>(
                            this->dqWorkSpaceGm, dSL1Buffer, this->constInfo, prevRunInfo);
                    } else {
                        if (dqIsNeedDeter[deterPpFlag]) {
                            int64_t offset = this->cBlockIdx * this->BASE_DQ_SIZE +
                                             deterPpFlag * (this->BASE_DQ_SIZE + 2 * this->BASE_DKV_SIZE) *
                                                 this->constInfo.deterConstInfo.usedCubeCoreNum;
                            // c3-deter
                            this->cubeBlock.template IterateMmDsKOlderDeter<CALC_TYPE, BaseClass::IS_DQ_WRITE_UB>(
                                this->deterGm[offset], this->dSL1Buf, this->constInfo, prevRunInfo);
                        } else {
                            // c3-normal
                            this->cubeBlock.template IterateMmDsK<CALC_TYPE, BaseClass::IS_DQ_WRITE_UB>(
                                this->dqWorkSpaceGm, dSL1Buffer, this->constInfo, prevRunInfo);
                        }
                    }
                    // compute dk
                    if (likely(this->constInfo.deterConstInfo.noNeedDeter)) {
                        // c4-normal
                        this->cubeBlock.template IterateMmDsQ<CALC_TYPE, BaseClass::IS_DK_WRITE_UB>(
                            this->dkWorkSpaceGm, dSL1Buffer, this->constInfo, prevRunInfo,
                            dqIsNeedDeter[deterPpFlag]); // c4
                    } else {
                        if (dkDvIsNeedDeter[deterPpFlag]) {
                            int64_t offset = this->cBlockIdx * this->BASE_DKV_SIZE +
                                             this->BASE_DQ_SIZE * this->constInfo.deterConstInfo.usedCubeCoreNum +
                                             deterPpFlag * (this->BASE_DQ_SIZE + 2 * this->BASE_DKV_SIZE) *
                                                 this->constInfo.deterConstInfo.usedCubeCoreNum;
                            // c4-deter
                            this->cubeBlock.template IterateMmDsQOlderDeter<CALC_TYPE, BaseClass::IS_DK_WRITE_UB>(
                                this->deterGm[offset], this->dSL1Buf, this->constInfo, prevRunInfo,
                                dqIsNeedDeter[deterPpFlag]); // c4
                        } else {
                            // c4-normal
                            this->cubeBlock.template IterateMmDsQ<CALC_TYPE, BaseClass::IS_DK_WRITE_UB>(
                                this->dkWorkSpaceGm, dSL1Buffer, this->constInfo, prevRunInfo,
                                dqIsNeedDeter[deterPpFlag]); // c4
                        }
                    }
                    if ASCEND_IS_AIC {
                        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_C4_TO_V3_FLAG);
                        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_C4_TO_V3_FLAG);
                    }
                    // compute dv
                    if (likely(this->constInfo.deterConstInfo.noNeedDeter)) {
                        // c5-normal
                        this->cubeBlock.template IterateMmPDy<CALC_TYPE, BaseClass::IS_DV_WRITE_UB>(
                            this->dvWorkSpaceGm, pL1Buffer, this->constInfo, prevRunInfo); // c5
                    } else {
                        if (dkDvIsNeedDeter[deterPpFlag]) {
                            int64_t offset = this->cBlockIdx * this->BASE_DKV_SIZE +
                                             (this->BASE_DQ_SIZE + this->BASE_DKV_SIZE) *
                                                 this->constInfo.deterConstInfo.usedCubeCoreNum +
                                             deterPpFlag * (this->BASE_DQ_SIZE + 2 * this->BASE_DKV_SIZE) *
                                                 this->constInfo.deterConstInfo.usedCubeCoreNum;
                            // c5-deter
                            this->cubeBlock.template IterateMmPDyOlderDeter<CALC_TYPE, BaseClass::IS_DV_WRITE_UB>(
                                this->deterGm[offset], this->pL1Buf, this->constInfo, prevRunInfo); // c5
                        } else {
                            // c5-normal
                            this->cubeBlock.template IterateMmPDy<CALC_TYPE, BaseClass::IS_DV_WRITE_UB>(
                                this->dvWorkSpaceGm, pL1Buffer, this->constInfo, prevRunInfo); // c5
                        }
                    }
                    if ASCEND_IS_AIC {
                        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_C5_TO_V4_FLAG);
                        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_C5_TO_V4_FLAG);
                    }
                    if ASCEND_IS_AIC {
                        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C4_TO_V6_FLAG);
                        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C4_TO_V6_FLAG);
                    }

                    needSyncDkMM = true;
                    if (taskId > 1) {
                        // 确定性计算的基本块滞后于上面的mm345一次
                        // WaitMm3Result
                        if ASCEND_IS_AIV {
                            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SYNC_C4_TO_V6_FLAG);
                        }
                        // DeterCompute
                        this->vecBlock.DeterCompute(mm1ResTensor, mm2ResTensor, this->constInfo, dqIsNeedDeter,
                                                    dkDvIsNeedDeter, false, maxValidBBLen - remainLoopNum,
                                                    remainLoopNum, &deterPpFlag);
                        remainLoopNum--;
                    }
                    if ASCEND_IS_AIV {
                        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_DETER_FIX_FLAG);
                    }
                }
                if (this->isLastLoop && taskId > 0) {
                    mm1ResTensor =
                        this->mm1ResBuf[runInfos[(taskId + 1) & 1].commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
                    mm2ResTensor =
                        this->mm2ResBuf[runInfos[(taskId + 1) & 1].commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
                    // WaitMm3Result
                    if ASCEND_IS_AIV {
                        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SYNC_C4_TO_V6_FLAG);
                    }
                    // DeterCompute
                    this->vecBlock.DeterCompute(mm1ResTensor, mm2ResTensor, this->constInfo, dqIsNeedDeter,
                                                dkDvIsNeedDeter, true, maxValidBBLen - remainLoopNum, remainLoopNum,
                                                &deterPpFlag);
                    remainLoopNum--;
                }
                taskId++;
            } else { // 主流程做完的核，后面需要参与到确定性计算中
                mm1ResTensor =
                    this->mm1ResBuf[runInfos[(taskId + 1) & 1].commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
                mm2ResTensor =
                    this->mm2ResBuf[runInfos[(taskId + 1) & 1].commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
                int64_t computeLoopIdx = maxValidBBLen - remainLoopNum;
                GetIsNeedDeter(computeLoopIdx);
                this->vecBlock.WriteOffsetToGM(this->constInfo, OUTINDEX, OUTINDEX, OUTINDEX, computeLoopIdx,
                                               remainLoopNum);
                deterPpFlag = 1 - deterPpFlag;
                // DeterCompute
                this->vecBlock.DeterCompute(mm1ResTensor, mm2ResTensor, this->constInfo, dqIsNeedDeter, dkDvIsNeedDeter,
                                            true, computeLoopIdx, remainLoopNum, &deterPpFlag);
                remainLoopNum--;
            }
        } else { // 没有主流程需要处理的核，只参与确定性计算
            mm1ResTensor =
                this->mm1ResBuf[runInfos[(taskId + 1) & 1].commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
            mm2ResTensor =
                this->mm2ResBuf[runInfos[(taskId + 1) & 1].commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
            GetIsNeedDeter(loopIdx);
            deterPpFlag = 1 - deterPpFlag;
            // DeterCompute
            this->vecBlock.DeterCompute(mm1ResTensor, mm2ResTensor, this->constInfo, dqIsNeedDeter, dkDvIsNeedDeter,
                                        true, loopIdx, remainLoopNum, &deterPpFlag);
            remainLoopNum--;
        }
        loopIdx++;
    }
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
        dqIsNeedDeter[computeLoopIdx & 1] = (this->tilingData->baseDeterParam.dqIsNeedDeter[arrayIndex] & mask) != 0;
        dkDvIsNeedDeter[computeLoopIdx & 1] =
            (this->tilingData->baseDeterParam.dkDvIsNeedDeter[arrayIndex] & mask) != 0;
    }
}

} // namespace FagBaseApi

#endif
