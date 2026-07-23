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
 * \file flash_attention_score_grad_kernel_small_sd.h
 * \brief Small-S/Small-D BN2 kernel entry for arch35 regbase.
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_KERNEL_SMALL_SD_H
#define FLASH_ATTENTION_SCORE_GRAD_KERNEL_SMALL_SD_H

#include "flash_attention_score_grad_common.h"
#include "flash_attention_score_grad_block_cube.h"
#include "flash_attention_score_grad_block_vec.h"
#include "flash_attention_score_grad_kernel_base.h"
#include "cube_api/mutex_buffer.h"
#include "flash_attention_score_grad_tiling_data_regbase.h"

namespace FagBaseApi {

// SmallSD fixed DAG sync table:
// - SYNC_C2_TO_V2_FLAG[task&1]: C2(QK score) ready for V2 softmax.
// - SYNC_C1_TO_V2_FLAG[task&1]: C1(dP) ready for V2/V3.
// - SYNC_V3_TO_C3_FLAG: dS copied to L1, C3/C4 may consume it.
// - SYNC_V4_TO_C5_FLAG: P copied to L1, C5 may consume it.
// - SYNC_C3_TO_V5_FLAG / SYNC_C4_TO_V6_FLAG: DQ/DK UB result ready for cast/writeback.
// - SYNC_DETER_FIX_FLAG: previous DK UB writeback completed, next C4 can reuse mm2 result buffer.
// SmallSD is entered only by the arch35 regbase key with:
// FP16/BF16, BN2, G=1, N1=N2, D==Dv, 0<S1/S2<128, 0<D<=128,
// no optional feature, no deterministic path, no NZ output, and no swizzle/remap.
// L0A/L0B double buffering is still provided by the existing CubeBlock implementation;
// this class trims scalar-heavy scheduling, validity search, and generic offset helpers.
template <typename CubeBlockType, typename VecBlockType>
class FlashAttentionScoreGradKernelSmallSD
    : public FlashAttentionScoreGradKernelBase<FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>,
                                               CubeBlockType, VecBlockType> {
public:
    ARGS_TRAITS;
    static_assert(SPLIT_AXIS == BN2, "SmallSD only supports BN2 split axis.");
    static_assert(!IS_ATTEN_MASK && !IS_PSE && !IS_DROP && !IS_ROPE, "SmallSD does not support optional features.");
    static_assert(!IS_BN2_MULTIBLK && !IS_D_NO_EQUAL && !IS_NZ_OUT && !IS_TND_SWIZZLE,
                  "SmallSD expects simple BN2 ownership without multiblock/swizzle/NZ/D mismatch.");
    static_assert(DETER_SPARSE_TYPE == NO_DETER, "SmallSD only supports non-deterministic dense ownership.");
    using BaseClass = FlashAttentionScoreGradKernelBase<FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>,
                                                        CubeBlockType, VecBlockType>;
    static_assert(!BaseClass::IS_FP32_INPUT, "SmallSD only supports FP16/BF16 input.");
    static_assert(BaseClass::CUBE_BASEM == static_cast<uint32_t>(S1TemplateType::Aligned128) &&
                      BaseClass::CUBE_BASEN == static_cast<uint32_t>(S2TemplateType::Aligned128),
                  "SmallSD reuses the existing 128x128 S tile only.");
    static_assert(BaseClass::HEAD_DIM_ALIGN <= static_cast<uint32_t>(DTemplateType::Aligned128),
                  "SmallSD only supports D template <= 128.");
    __aicore__ inline void Init(GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR query, GM_ADDR pseShift,
                                GM_ADDR dropMask, GM_ADDR attenMask, GM_ADDR y, GM_ADDR softmaxMax, GM_ADDR softmaxSum,
                                GM_ADDR prefixN, GM_ADDR actualSeqQlen, GM_ADDR actualSeqKvlen, GM_ADDR deqScaleQ,
                                GM_ADDR deqScaleK, GM_ADDR deqScaleV, GM_ADDR deqScaleDy, GM_ADDR queryRope,
                                GM_ADDR keyRope, GM_ADDR sink, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dpse,
                                GM_ADDR dqRope, GM_ADDR dkRope, GM_ADDR dsink, GM_ADDR workspace,
                                FagTilingType ordTilingData, TPipe *pipeIn);
    __aicore__ inline void Process();
    __aicore__ inline void ReadTndSeqLenSmallSD(int64_t batchIdx, int64_t &actualS1Len, int64_t &actualS2Len);
    __aicore__ inline void AdvanceTndBatchPrefixSmallSD(int64_t actualS1Len, int64_t actualS2Len);
    __aicore__ inline void ComputeDqkvBn2(FagRunInfo &prevRunInfo, int64_t taskId);

private:
    struct SmallSDTaskCursor {
        int64_t bIdx = 0;
        int64_t n2oIdx = 0;
        int64_t actualS1Len = 0;
        int64_t actualS2Len = 0;
        int64_t s2SizeAcc = 0;
        int64_t b1SSOffset = 0;
        int64_t b1SSOffsetAlign = 0;
        int64_t lastBatchTotalBaseIdx = 0;
        int64_t lastBatchTotalS1BOffset = 0;
        int64_t lastBatchTotalS2BOffset = 0;
        int64_t lastBatchTotalS1BOffsetForDv = 0;
        int64_t lastBatchTotalS2BOffsetForDv = 0;
        int64_t lastBatchTotalS1S2SizeAlign = 0;
        int64_t lastBatchTotalS1S2Size = 0;
        int64_t lastBatchTotalS2Size = 0;
        int64_t qOffset = 0;
        int64_t kOffset = 0;
        int64_t qTaskStride = 0;
        int64_t kTaskStride = 0;
        int64_t qBatchGap = 0;
        int64_t kBatchGap = 0;
    };

    struct SmallSDShape {
        int64_t s1 = 0;
        int64_t s2 = 0;
        int64_t s2Align16 = 0;
        int64_t halfS1 = 0;
        int64_t firstHalfS1 = 0;
        int64_t halfS2 = 0;
        int64_t firstHalfS2 = 0;
    };

    struct SmallSDOffsets {
        int64_t q = 0;
        int64_t k = 0;
        int64_t attention = 0;
        int64_t attentionAlign = 0;
        int64_t s2Prefix = 0;
    };

    __aicore__ inline void InitTaskCursorSmallSD(SmallSDTaskCursor &cursor, int64_t index);
    __aicore__ inline void AdvanceTaskCursorSmallSD(SmallSDTaskCursor &cursor);
    __aicore__ inline void SetCursorGmOffsetSmallSD(SmallSDTaskCursor &cursor);
    __aicore__ inline void LoadTndCursorPrefixSmallSD(SmallSDTaskCursor &cursor);
    __aicore__ inline SmallSDShape MakeShapeSmallSD(const SmallSDTaskCursor &cursor) const;
    __aicore__ inline SmallSDOffsets MakeOffsetsSmallSD(const SmallSDTaskCursor &cursor) const;
    __aicore__ inline void SetRunInfoSmallSD(FagRunInfo &runInfo, int64_t taskId, const SmallSDTaskCursor &cursor);
    __aicore__ inline void IssueMm12SmallSD(FagRunInfo &runInfo, int64_t taskId);

    int64_t cachedTndBatchIdx = -1;
    int64_t cachedTndS1Len = 0;
    int64_t cachedTndS2Len = 0;
};

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::Init(
    GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR query, GM_ADDR pseShift, GM_ADDR dropMask, GM_ADDR attenMask,
    GM_ADDR y, GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR prefixN, GM_ADDR actualSeqQlen, GM_ADDR actualSeqKvlen,
    GM_ADDR deqScaleQ, GM_ADDR deqScaleK, GM_ADDR deqScaleV, GM_ADDR deqScaleDy, GM_ADDR queryRope, GM_ADDR keyRope,
    GM_ADDR sink, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dpse, GM_ADDR dqRope, GM_ADDR dkRope, GM_ADDR dsink,
    GM_ADDR workspace, FagTilingType ordTilingData, TPipe *pipeIn)
{
    (void)pseShift;
    (void)dropMask;
    (void)attenMask;
    (void)prefixN;
    (void)queryRope;
    (void)keyRope;
    (void)sink;
    (void)dpse;
    (void)dqRope;
    (void)dkRope;
    (void)dsink;

    if ASCEND_IS_AIV {
        this->vBlockIdx = GetBlockIdx();
        this->cBlockIdx = this->vBlockIdx / CV_CORE_RATIO;
        this->vSubBlockIdx = GetSubBlockIdx();
    } else {
        this->cBlockIdx = GetBlockIdx();
    }
    this->tilingData = ordTilingData;
    this->pipe = pipeIn;

    this->SetConstInfo();
    this->actualCalcS1Token = this->constInfo.s1Token;
    this->actualCalcS2Token = this->constInfo.s2Token;

    this->prefixNAddr = prefixN;
    this->actualSeqQlenAddr = actualSeqQlen;
    this->actualSeqKvlenAddr = actualSeqKvlen;
    this->constInfo.seqS1_addr = actualSeqQlen;
    this->constInfo.seqS2_addr = actualSeqKvlen;

    this->InitCVCommonGlobalBuffer(dq, dk, dv, deqScaleQ, deqScaleK, deqScaleV, deqScaleDy, workspace);
    this->InitCVCommonBuffer();

    this->vecBlock.SetVecBlockParams(pipeIn, this->tilingData, this->vBlockIdx, this->cBlockIdx, this->vSubBlockIdx,
                                     this->attenMaskInfo, this->pseInfo, this->dropInfo);
    this->vecBlock.InitUbBuffer();
    this->vecBlock.InitGlobalBuffer(value, dy, y, pseShift, dropMask, attenMask, softmaxMax, softmaxSum, deqScaleQ,
                                    deqScaleK, deqScaleV, deqScaleDy, dq, dk, dv, dqRope, dkRope, sink, dsink,
                                    workspace);

    this->cubeBlock.SetCubeBlockParams(pipeIn, this->tilingData, &this->l1BufferManager);
    this->cubeBlock.InitCubeBuffer(this->constInfo);
    this->cubeBlock.InitGlobalBuffer(query, key, value, dy, queryRope, keyRope, dq, dk, dv, workspace);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::ReadTndSeqLenSmallSD(
    int64_t batchIdx, int64_t &actualS1Len, int64_t &actualS2Len)
{
    if (batchIdx == cachedTndBatchIdx) {
        actualS1Len = cachedTndS1Len;
        actualS2Len = cachedTndS2Len;
        return;
    }
    __gm__ int64_t *actualSeqQlen = (__gm__ int64_t *)this->actualSeqQlenAddr;
    __gm__ int64_t *actualSeqKvlen = (__gm__ int64_t *)this->actualSeqKvlenAddr;
    if (batchIdx == 0) {
        actualS1Len = actualSeqQlen[0];
        actualS2Len = actualSeqKvlen[0];
    } else {
        actualS1Len = actualSeqQlen[batchIdx] - actualSeqQlen[batchIdx - 1];
        actualS2Len = actualSeqKvlen[batchIdx] - actualSeqKvlen[batchIdx - 1];
    }
    cachedTndBatchIdx = batchIdx;
    cachedTndS1Len = actualS1Len;
    cachedTndS2Len = actualS2Len;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::AdvanceTndBatchPrefixSmallSD(
    int64_t actualS1Len, int64_t actualS2Len)
{
    this->curBatchIdx++;
    this->curBatchTotalBaseIdx += this->constInfo.commonConstInfo.n2G;
    this->curBatchTotalS1BOffset += actualS1Len * this->constInfo.commonConstInfo.n2GD;
    this->curBatchTotalS2BOffset += actualS2Len * this->constInfo.commonConstInfo.n2D;
    this->curBatchTotalS1BOffsetForDv += actualS1Len * this->constInfo.commonConstInfo.n2GDv;
    this->curBatchTotalS2BOffsetForDv += actualS2Len * this->constInfo.commonConstInfo.n2Dv;
    this->curBatchTotalS1S2SizeAlign += actualS1Len * AlignTo16(actualS2Len);
    this->curBatchTotalS1S2Size += actualS1Len * actualS2Len;
    this->curBatchTotalS2Size += actualS2Len;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::SetCursorGmOffsetSmallSD(
    typename FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::SmallSDTaskCursor &cursor)
{
    // The initial SmallSD host route guarantees G == 1; GQA needs a separate cursor/accumulation extension.
    if constexpr (IS_TND) {
        cursor.qOffset = cursor.lastBatchTotalS1BOffset +
                         cursor.n2oIdx * this->constInfo.commonConstInfo.gD;
        cursor.kOffset = cursor.lastBatchTotalS2BOffset +
                         cursor.n2oIdx * this->constInfo.commonConstInfo.dSize;
        cursor.qTaskStride = this->constInfo.commonConstInfo.gD;
        cursor.kTaskStride = this->constInfo.commonConstInfo.dSize;
        cursor.qBatchGap = 0;
        cursor.kBatchGap = 0;
    } else {
        if (this->constInfo.commonConstInfo.layoutType == BNGSD) {
            cursor.qOffset = cursor.bIdx * this->constInfo.commonConstInfo.n2GS1D +
                             cursor.n2oIdx * this->constInfo.commonConstInfo.gS1D;
            cursor.kOffset = cursor.bIdx * this->constInfo.commonConstInfo.n2S2D +
                             cursor.n2oIdx * this->constInfo.commonConstInfo.s2D;
            cursor.qTaskStride = this->constInfo.commonConstInfo.gS1D;
            cursor.kTaskStride = this->constInfo.commonConstInfo.s2D;
            cursor.qBatchGap = 0;
            cursor.kBatchGap = 0;
        } else if (this->constInfo.commonConstInfo.layoutType == SBNGD) {
            cursor.qOffset = cursor.bIdx * this->constInfo.commonConstInfo.n2GD +
                             cursor.n2oIdx * this->constInfo.commonConstInfo.gD;
            cursor.kOffset = cursor.bIdx * this->constInfo.commonConstInfo.n2D +
                             cursor.n2oIdx * this->constInfo.commonConstInfo.dSize;
            cursor.qTaskStride = this->constInfo.commonConstInfo.gD;
            cursor.kTaskStride = this->constInfo.commonConstInfo.dSize;
            cursor.qBatchGap = 0;
            cursor.kBatchGap = 0;
        } else {
            cursor.qOffset = cursor.bIdx * this->constInfo.commonConstInfo.n2GS1D +
                             cursor.n2oIdx * this->constInfo.commonConstInfo.gD;
            cursor.kOffset = cursor.bIdx * this->constInfo.commonConstInfo.n2S2D +
                             cursor.n2oIdx * this->constInfo.commonConstInfo.dSize;
            cursor.qTaskStride = this->constInfo.commonConstInfo.gD;
            cursor.kTaskStride = this->constInfo.commonConstInfo.dSize;
            cursor.qBatchGap = this->constInfo.commonConstInfo.n2GS1D -
                               this->constInfo.commonConstInfo.n2G * this->constInfo.commonConstInfo.gD;
            cursor.kBatchGap = this->constInfo.commonConstInfo.n2S2D -
                               this->constInfo.n2Size * this->constInfo.commonConstInfo.dSize;
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::LoadTndCursorPrefixSmallSD(
    typename FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::SmallSDTaskCursor &cursor)
{
    cursor.lastBatchTotalBaseIdx = this->curBatchTotalBaseIdx;
    cursor.lastBatchTotalS1BOffset = this->curBatchTotalS1BOffset;
    cursor.lastBatchTotalS2BOffset = this->curBatchTotalS2BOffset;
    cursor.lastBatchTotalS1BOffsetForDv = this->curBatchTotalS1BOffsetForDv;
    cursor.lastBatchTotalS2BOffsetForDv = this->curBatchTotalS2BOffsetForDv;
    cursor.lastBatchTotalS1S2SizeAlign = this->curBatchTotalS1S2SizeAlign;
    cursor.lastBatchTotalS1S2Size = this->curBatchTotalS1S2Size;
    cursor.lastBatchTotalS2Size = this->curBatchTotalS2Size;
    cursor.s2SizeAcc = cursor.lastBatchTotalS2Size;
    cursor.b1SSOffsetAlign = cursor.lastBatchTotalS1S2SizeAlign;
    cursor.b1SSOffset = cursor.lastBatchTotalS1S2Size;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::InitTaskCursorSmallSD(
    typename FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::SmallSDTaskCursor &cursor,
    int64_t index)
{
    if constexpr (IS_TND) {
        const auto &tndCore =
            this->tilingData->smallSDTilingData.tndCoreParam[this->cBlockIdx];
        const int64_t bIdx = tndCore.startBatchIdx;
        ReadTndSeqLenSmallSD(bIdx, cursor.actualS1Len, cursor.actualS2Len);
        cursor.bIdx = bIdx;
        // G == 1 in the SmallSD route, so the host-provided startN2 is the local N2 index.
        cursor.n2oIdx = tndCore.startN2Idx;
        LoadTndCursorPrefixSmallSD(cursor);
    } else {
        cursor.bIdx = index / this->constInfo.n2Size;
        cursor.n2oIdx = index - cursor.bIdx * this->constInfo.n2Size;
        cursor.actualS1Len = this->constInfo.commonConstInfo.s1Size;
        cursor.actualS2Len = this->constInfo.commonConstInfo.s2Size;
        cursor.s2SizeAcc = cursor.bIdx * this->constInfo.commonConstInfo.s2Size;
        cursor.b1SSOffset = cursor.bIdx * this->constInfo.commonConstInfo.s1S2;
        cursor.b1SSOffsetAlign =
            cursor.bIdx * this->constInfo.commonConstInfo.s1Size *
            AlignTo16(this->constInfo.commonConstInfo.s2Size);
    }
    SetCursorGmOffsetSmallSD(cursor);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::IssueMm12SmallSD(
    FagRunInfo &runInfo, int64_t taskId)
{
    const int64_t taskMod2 = runInfo.commonRunInfo.taskIdMod2;
    LocalTensor<CALC_TYPE> mm2ResTensor =
        this->mm2ResBuf[taskMod2].template Get<CALC_TYPE>();
    this->cubeBlock.IterateMmQK(mm2ResTensor, this->constInfo, runInfo, this->preloadArgs);
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C2_TO_V2_FLAG[taskMod2]);
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C2_TO_V2_FLAG[taskMod2]);
    }

    LocalTensor<CALC_TYPE> mm1ResTensor =
        this->mm1ResBuf[taskMod2].template Get<CALC_TYPE>();
    this->cubeBlock.IterateMmDyV(mm1ResTensor, this->constInfo, runInfo, this->preloadArgs);
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C1_TO_V2_FLAG[taskMod2]);
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C1_TO_V2_FLAG[taskMod2]);
    }

    this->vecBlock.CopyMaxSum(this->constInfo, runInfo, taskId);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::AdvanceTaskCursorSmallSD(
    typename FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::SmallSDTaskCursor &cursor)
{
    cursor.n2oIdx++;
    if (cursor.n2oIdx < this->constInfo.n2Size) {
        cursor.qOffset += cursor.qTaskStride;
        cursor.kOffset += cursor.kTaskStride;
        return;
    }
    cursor.n2oIdx = 0;
    cursor.bIdx++;

    if constexpr (IS_TND) {
        AdvanceTndBatchPrefixSmallSD(cursor.actualS1Len, cursor.actualS2Len);
        ReadTndSeqLenSmallSD(cursor.bIdx, cursor.actualS1Len, cursor.actualS2Len);
        LoadTndCursorPrefixSmallSD(cursor);
        SetCursorGmOffsetSmallSD(cursor);
    } else {
        cursor.qOffset += cursor.qTaskStride + cursor.qBatchGap;
        cursor.kOffset += cursor.kTaskStride + cursor.kBatchGap;
        cursor.s2SizeAcc += this->constInfo.commonConstInfo.s2Size;
        cursor.b1SSOffset += this->constInfo.commonConstInfo.s1S2;
        cursor.b1SSOffsetAlign += this->constInfo.commonConstInfo.s1Size *
                                  AlignTo16(this->constInfo.commonConstInfo.s2Size);
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline typename FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::SmallSDShape
FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::MakeShapeSmallSD(
    const typename FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::SmallSDTaskCursor &cursor) const
{
    SmallSDShape shape;
    shape.s1 = cursor.actualS1Len;
    shape.s2 = cursor.actualS2Len;
    shape.s2Align16 = AlignTo16(cursor.actualS2Len);
    if ASCEND_IS_AIV {
        shape.halfS1 = (shape.s1 + 1) >> 1;
        shape.firstHalfS1 = shape.halfS1;
        shape.halfS2 = (shape.s2 + 1) >> 1;
        shape.firstHalfS2 = shape.halfS2;
        if (this->vSubBlockIdx == 1) {
            shape.halfS1 = shape.s1 - shape.halfS1;
            shape.halfS2 = shape.s2 - shape.halfS2;
        }
    }
    return shape;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline typename FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::SmallSDOffsets
FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::MakeOffsetsSmallSD(
    const typename FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::SmallSDTaskCursor &cursor) const
{
    SmallSDOffsets offsets;
    offsets.q = cursor.qOffset;
    offsets.k = cursor.kOffset;
    offsets.attention = cursor.b1SSOffset;
    offsets.attentionAlign = cursor.b1SSOffsetAlign;
    offsets.s2Prefix = cursor.s2SizeAcc;
    return offsets;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::SetRunInfoSmallSD(
    FagRunInfo &runInfo, int64_t taskId,
    const typename FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::SmallSDTaskCursor &cursor)
{
    const SmallSDShape shape = MakeShapeSmallSD(cursor);
    const SmallSDOffsets offsets = MakeOffsetsSmallSD(cursor);

    if constexpr (IS_TND) {
        runInfo.lastBatchTotalBaseIdx = cursor.lastBatchTotalBaseIdx;
        runInfo.lastBatchTotalS1BOffset = cursor.lastBatchTotalS1BOffset;
        runInfo.lastBatchTotalS2BOffset = cursor.lastBatchTotalS2BOffset;
        runInfo.lastBatchTotalS1BOffsetForDv = cursor.lastBatchTotalS1BOffsetForDv;
        runInfo.lastBatchTotalS2BOffsetForDv = cursor.lastBatchTotalS2BOffsetForDv;
        runInfo.lastBatchTotalS1S2SizeAlign = cursor.lastBatchTotalS1S2SizeAlign;
        runInfo.lastBatchTotalS1S2Size = cursor.lastBatchTotalS1S2Size;
        runInfo.lastBatchTotalS2Size = cursor.lastBatchTotalS2Size;
        runInfo.lastBatchIdx = cursor.bIdx;
    }

    runInfo.commonRunInfo.boIdx = cursor.bIdx;
    runInfo.commonRunInfo.n2oIdx = cursor.n2oIdx;
    runInfo.commonRunInfo.goIdx = 0;
    runInfo.commonRunInfo.s1oIdx = 0;
    runInfo.s2oIdx = 0;
    runInfo.commonRunInfo.s2SizeAcc = offsets.s2Prefix;
    runInfo.commonRunInfo.b1SSOffsetAlign = offsets.attentionAlign;
    runInfo.commonRunInfo.b1SSOffset = offsets.attention;
    runInfo.commonRunInfo.queryOffset = offsets.q;
    runInfo.dyOffset = offsets.q;
    runInfo.commonRunInfo.keyOffset = offsets.k;
    runInfo.commonRunInfo.valueOffset = offsets.k;

    runInfo.s2CvBegin = 0;
    runInfo.s2CvEnd = shape.s2;
    runInfo.commonRunInfo.taskId = taskId;
    runInfo.commonRunInfo.taskIdMod2 = taskId & 1;
    runInfo.commonRunInfo.actualS1Size = shape.s1;
    runInfo.commonRunInfo.actualS2Size = shape.s2;
    runInfo.commonRunInfo.s1RealSize = shape.s1;
    runInfo.commonRunInfo.s2RealSize = shape.s2;
    runInfo.commonRunInfo.s2StartIdx = 0;
    runInfo.commonRunInfo.s2AlignedSize = shape.s2Align16;
    runInfo.commonRunInfo.b1SSAttenMaskOffset = runInfo.commonRunInfo.b1SSOffset;
    runInfo.isS2IdxNoChange = false;
    runInfo.isNextS2IdxNoChange = false;
    this->preloadArgs.copyCurrent = true;
    this->preloadArgs.copyNext = false;

    if ASCEND_IS_AIV {
        runInfo.halfS2RealSize = shape.halfS2;
        runInfo.firstHalfS2RealSize = shape.firstHalfS2;
        runInfo.commonRunInfo.halfS1RealSize = shape.halfS1;
        runInfo.commonRunInfo.firstHalfS1RealSize = shape.firstHalfS1;
        runInfo.commonRunInfo.vecCoreOffset = this->vSubBlockIdx * shape.firstHalfS1;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::ComputeDqkvBn2(
    FagRunInfo &prevRunInfo, int64_t taskId)
{
    static_assert(BaseClass::IS_DQ_WRITE_UB, "SmallSD expects DQ to be written through UB.");
    static_assert(BaseClass::IS_DK_WRITE_UB, "SmallSD expects DK to be written through UB.");
    const bool needWaitPrevDkv = taskId > 1;
    const int64_t prevTaskMod2 = prevRunInfo.commonRunInfo.taskIdMod2;

    LocalTensor<CALC_TYPE> mm1ResTensor =
        this->mm1ResBuf[prevTaskMod2].template Get<CALC_TYPE>();
    LocalTensor<CALC_TYPE> mm2ResTensor =
        this->mm2ResBuf[prevTaskMod2].template Get<CALC_TYPE>();
    // wait mm2 result
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C2_TO_V2_FLAG[prevTaskMod2]);
    }
    this->vecBlock.ProcessVec2(mm2ResTensor, this->constInfo, prevRunInfo); // v2: simpleSoftmax
    // wait mm1 result
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C1_TO_V2_FLAG[prevTaskMod2]);
    }
    if ASCEND_IS_AIV {
        if (needWaitPrevDkv) {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_C4_TO_V3_FLAG);
        }
    }
    MutexBuffer<BufferType::L1, SyncType::NO_SYNC> dSL1Buffer = this->dSL1Buf.Get();
    MutexBuffer<BufferType::L1, SyncType::NO_SYNC> pL1Buffer = this->pL1Buf.Get();
    this->vecBlock.ProcessVec3(dSL1Buffer, mm1ResTensor, mm2ResTensor, this->constInfo,
                               prevRunInfo); // v3: cast + nd2nz
    if ASCEND_IS_AIV {
        if (needWaitPrevDkv) {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_C5_TO_V4_FLAG);
        }
    }
    this->vecBlock.ProcessVec4(pL1Buffer, mm2ResTensor, this->constInfo, prevRunInfo); // v4: cast + nd2nz
    if ASCEND_IS_AIV {
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V3_TO_C3_FLAG); // dqk must wait ds copy completely
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V4_TO_C5_FLAG); // dv must wait p copy completely
    }

    if ASCEND_IS_AIC {
        // wait ds in ub copy to l1
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SYNC_V3_TO_C3_FLAG);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_V3_TO_C3_FLAG);
        // wait p in ub copy to l1
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SYNC_V4_TO_C5_FLAG);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_V4_TO_C5_FLAG);
    }

    // compute dq
    mm1ResTensor = this->mm1ResBuf[prevTaskMod2].template Get<CALC_TYPE>();
    this->cubeBlock.template IterateMmDsK<CALC_TYPE, BaseClass::IS_DQ_WRITE_UB>(mm1ResTensor, dSL1Buffer,
                                                                                this->constInfo, prevRunInfo); // c3
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C3_TO_V5_FLAG);
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C3_TO_V5_FLAG);
    } else {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C3_TO_V5_FLAG);
    }
    this->vecBlock.template ProcessMulsAndCast<CALC_TYPE, BaseClass::IS_DQ_WRITE_UB, DQ_IDX>(
        mm1ResTensor, this->constInfo, prevRunInfo); // v5: dq muls + cast

    // compute dk
    mm2ResTensor = this->mm2ResBuf[prevTaskMod2].template Get<CALC_TYPE>();
    this->cubeBlock.template IterateMmDsQ<CALC_TYPE, BaseClass::IS_DK_WRITE_UB>(mm2ResTensor, dSL1Buffer,
                                                                                this->constInfo, prevRunInfo); // c4
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C4_TO_V6_FLAG);
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C4_TO_V6_FLAG);
    } else {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C4_TO_V6_FLAG);
    }
    this->vecBlock.template ProcessMulsAndCast<CALC_TYPE, BaseClass::IS_DK_WRITE_UB, DK_IDX>(
        mm2ResTensor, this->constInfo, prevRunInfo); // v6: dk muls + cast
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_C4_TO_V3_FLAG);
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_C4_TO_V3_FLAG);
    }
    if ASCEND_IS_AIV {
        CrossCoreSetFlag<SYNC_MODE, PIPE_V>(SYNC_DETER_FIX_FLAG);
    }

    // compute dv
    this->cubeBlock.template IterateMmPDy<OUTDTYPE, BaseClass::IS_DV_WRITE_UB>(this->dvGm, pL1Buffer,
                                                                               this->constInfo, prevRunInfo); // c5
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_C5_TO_V4_FLAG);
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_C5_TO_V4_FLAG);
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::Process()
{
    const auto &coreTask =
        this->tilingData->smallSDTilingData.coreTaskParam[this->cBlockIdx];
    const int64_t blockStart = coreTask.blockStart;
    const int64_t groupCount = coreTask.groupCount;
    if (groupCount <= 0) {
        return;
    }
    if constexpr (IS_TND) {
        const auto &tndCore =
            this->tilingData->smallSDTilingData.tndCoreParam[this->cBlockIdx];
        this->curBatchIdx = tndCore.startBatchIdx;
        this->curBatchTotalBaseIdx = tndCore.baseTaskPrefix;
        this->curBatchTotalS1BOffset = tndCore.qPrefixOffset;
        this->curBatchTotalS2BOffset = tndCore.kPrefixOffset;
        this->curBatchTotalS1BOffsetForDv = tndCore.qDvPrefixOffset;
        this->curBatchTotalS2BOffsetForDv = tndCore.kDvPrefixOffset;
        this->curBatchTotalS1S2SizeAlign = tndCore.attenAlignPrefixOffset;
        this->curBatchTotalS1S2Size = tndCore.attenPrefixOffset;
        this->curBatchTotalS2Size = tndCore.s2PrefixSize;
        cachedTndBatchIdx = -1;
    }
    int64_t taskId = 0;
    SmallSDTaskCursor cursor;
    InitTaskCursorSmallSD(cursor, blockStart);

    if (groupCount == 1) {
        constexpr int64_t singleTaskId = 0;
        FagRunInfo runInfo;
        SetRunInfoSmallSD(runInfo, singleTaskId, cursor);
        IssueMm12SmallSD(runInfo, singleTaskId);
        this->isLastLoop = true;
        this->vecBlock.ProcessVec1(this->constInfo, runInfo); // v1: softmaxGrad
        ComputeDqkvBn2(runInfo, singleTaskId + 1);
        return;
    }

    FagRunInfo runInfos[2]; // for cv ping pong
    for (; taskId < groupCount; ++taskId) {
        this->isLastLoop = false;
        const int64_t runIdx = taskId & 1;
        const int64_t prevIdx = runIdx ^ 1;
        if (taskId > 0) {
            FagRunInfo &prevRunInfo = runInfos[prevIdx];
            this->vecBlock.ProcessVec1(this->constInfo, prevRunInfo); // v1: softmaxGrad
            ComputeDqkvBn2(prevRunInfo, taskId);
        }

        SetRunInfoSmallSD(runInfos[runIdx], taskId, cursor);

        if constexpr (BaseClass::IS_DK_WRITE_UB) {
            if ASCEND_IS_AIC {
                if (taskId > 1) {
                    CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SYNC_DETER_FIX_FLAG);
                    CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_DETER_FIX_FLAG);
                }
            }
        }

        IssueMm12SmallSD(runInfos[runIdx], taskId);
        if (taskId + 1 < groupCount) {
            AdvanceTaskCursorSmallSD(cursor);
        }
    }

    this->isLastLoop = true;
    const int64_t prevIdx = (taskId + 1) & 1;
    FagRunInfo &prevRunInfo = runInfos[prevIdx];
    this->vecBlock.ProcessVec1(this->constInfo, prevRunInfo); // v1: softmaxGrad
    ComputeDqkvBn2(prevRunInfo, taskId);
}

} // namespace FagBaseApi

#endif
