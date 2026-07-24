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
#include "flash_attention_score_grad_buffer_small_sd.h"
#include "flash_attention_score_grad_event_small_sd.h"
#include "flash_attention_score_grad_kernel_base.h"
#include "cube_api/mutex_buffer.h"
#include "flash_attention_score_grad_tiling_data_regbase.h"

namespace FagBaseApi {

// SmallSD fixed DAG sync table. The numeric ids intentionally mirror the generic
// regbase event allocation, but the SmallSD path uses its own semantic aliases:
// - SMALL_SD_CUBE_QK_READY_FLAG[slot]: Cube QK score ready for Vector softmax.
// - SMALL_SD_CUBE_DYV_READY_FLAG[slot]: Cube dP ready for Vector dS/P conversion.
// - SMALL_SD_DS_L1_READY_FLAG: dS copied to L1, DQ/DK Cube may consume it.
// - SMALL_SD_P_L1_READY_FLAG: P copied to L1, DV Cube may consume it.
// - SMALL_SD_DQ_UB_READY_FLAG / SMALL_SD_DK_UB_READY_FLAG: DQ/DK UB ready for cast/writeback.
// - SMALL_SD_DS_L1_REUSABLE_FLAG / SMALL_SD_P_L1_REUSABLE_FLAG: shared L1 buffers reusable.
// - SMALL_SD_SLOT_REUSE_READY_FLAG: DK UB writeback completed, the two-slot buffer is reusable.
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
    __aicore__ inline void SetSmallSDConstInfo();

private:
    __aicore__ inline const optiling::fag::SmallSDTilingDataRegbase &GetSmallSDTilingData() const;
    __aicore__ inline void InitSmallSDBlocks(GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR query,
                                             GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR dq, GM_ADDR dk,
                                             GM_ADDR dv, GM_ADDR workspace, TPipe *pipeIn);
    __aicore__ inline void InitTaskCursorSmallSD(SmallSDTaskCursor &cursor, int64_t index);
    __aicore__ inline void AdvanceTaskCursor(SmallSDTaskCursor &cursor);
    __aicore__ inline void SetCursorGmOffsetSmallSD(SmallSDTaskCursor &cursor);
    __aicore__ inline void LoadTndCursorPrefixSmallSD(SmallSDTaskCursor &cursor);
    __aicore__ inline SmallSDShape MakeShapeSmallSD(const SmallSDTaskCursor &cursor) const;
    __aicore__ inline SmallSDOffsets MakeOffsetsSmallSD(const SmallSDTaskCursor &cursor) const;
    __aicore__ inline void BuildSmallSDConstInfo();
    __aicore__ inline void PrepareSlot(SmallSDPipelineSlot &slot, const SmallSDTaskCursor &cursor, int64_t taskId);
    __aicore__ inline void IssueCube(SmallSDPipelineSlot &slot);
    __aicore__ inline void ConsumeVector(SmallSDPipelineSlot &slot, bool needWaitL1Reusable);
    __aicore__ inline void WaitSlotReusable(SmallSDPipelineSlot &slot);
    __aicore__ inline void DrainPipeline(SmallSDPipelineSlot (&slots)[2], int64_t lastSlotIdx, int64_t taskCount);
    __aicore__ inline void ProcessSingleTask(SmallSDTaskCursor &cursor);
    __aicore__ inline void ProcessMultipleTasks(SmallSDTaskCursor &cursor, int64_t taskCount);

    SmallSDConstInfo smallSDConstInfo;
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
    (void)y;
    (void)prefixN;
    (void)deqScaleQ;
    (void)deqScaleK;
    (void)deqScaleV;
    (void)deqScaleDy;
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

    SetSmallSDConstInfo();
    this->actualCalcS1Token = 0;
    this->actualCalcS2Token = 0;

    this->prefixNAddr = prefixN;
    this->actualSeqQlenAddr = actualSeqQlen;
    this->actualSeqKvlenAddr = actualSeqKvlen;

    InitSmallSDBlocks(key, value, dy, query, softmaxMax, softmaxSum, dq, dk, dv, workspace, pipeIn);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::InitSmallSDBlocks(
    GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR query, GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR dq,
    GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace, TPipe *pipeIn)
{
    this->cubeBlock.Init(query, key, value, dy, dq, dk, dv, workspace, pipeIn, &smallSDConstInfo);
    this->vecBlock.Init(softmaxMax, softmaxSum, dq, dk, dv, workspace, pipeIn, &smallSDConstInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline const optiling::fag::SmallSDTilingDataRegbase &
FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::GetSmallSDTilingData() const
{
    // Transition seam: today SmallSD payload is embedded in the generic regbase tiling data.
    // When Host/Entry switch to FlashAttentionScoreGradSmallSDTilingDataRegbase, this accessor is the only
    // Kernel-side read boundary that needs to change.
    return this->tilingData->smallSDTilingData;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::BuildSmallSDConstInfo()
{
    const auto &baseParam = GetSmallSDTilingData().baseParam;
    smallSDConstInfo.b = baseParam.bSize;
    smallSDConstInfo.n1 = baseParam.n1Size;
    smallSDConstInfo.n2 = baseParam.n2Size;
    smallSDConstInfo.g = baseParam.gSize;
    smallSDConstInfo.s1 = baseParam.maxS1;
    smallSDConstInfo.s2 = baseParam.maxS2;
    smallSDConstInfo.d = baseParam.actualD;
    smallSDConstInfo.dv = baseParam.actualDv;
    smallSDConstInfo.layout = baseParam.layoutType;
    smallSDConstInfo.inputDtype = baseParam.inputDtype;
    smallSDConstInfo.outputDtype = baseParam.outputDtype;
    smallSDConstInfo.calcTypeSize = baseParam.calcTypeSize;
    smallSDConstInfo.usedCoreNum = baseParam.usedCoreNum;
    smallSDConstInfo.taskCount = baseParam.validTaskCount;
    smallSDConstInfo.isTnd = baseParam.isTnd;
    smallSDConstInfo.isSingleTask = baseParam.isSingleTask;
    smallSDConstInfo.s2Align16 = baseParam.s2Align16;
    smallSDConstInfo.scale = baseParam.scaleValue;
    smallSDConstInfo.workspaceBaseOffset = baseParam.workspaceBaseOffset;
    smallSDConstInfo.workspaceSize = baseParam.workspaceSize;

    const uint64_t gD = static_cast<uint64_t>(smallSDConstInfo.g) * smallSDConstInfo.d;
    const uint64_t gDv = static_cast<uint64_t>(smallSDConstInfo.g) * smallSDConstInfo.dv;
    const uint64_t s1D = static_cast<uint64_t>(smallSDConstInfo.s1) * smallSDConstInfo.d;
    const uint64_t s2D = static_cast<uint64_t>(smallSDConstInfo.s2) * smallSDConstInfo.d;
    const uint64_t s1Dv = static_cast<uint64_t>(smallSDConstInfo.s1) * smallSDConstInfo.dv;
    const uint64_t s2Dv = static_cast<uint64_t>(smallSDConstInfo.s2) * smallSDConstInfo.dv;
    const uint64_t n2G = static_cast<uint64_t>(smallSDConstInfo.n2) * smallSDConstInfo.g;

    if constexpr (IS_TND) {
        smallSDConstInfo.qStrideB = 0;
        smallSDConstInfo.qStrideN2 = gD;
        smallSDConstInfo.kStrideB = 0;
        smallSDConstInfo.kStrideN2 = smallSDConstInfo.d;
        smallSDConstInfo.vStrideB = 0;
        smallSDConstInfo.vStrideN2 = smallSDConstInfo.dv;
    } else if (smallSDConstInfo.layout == BNGSD) {
        smallSDConstInfo.qStrideB = n2G * s1D;
        smallSDConstInfo.qStrideN2 = static_cast<uint64_t>(smallSDConstInfo.g) * s1D;
        smallSDConstInfo.kStrideB = static_cast<uint64_t>(smallSDConstInfo.n2) * s2D;
        smallSDConstInfo.kStrideN2 = s2D;
        smallSDConstInfo.vStrideB = static_cast<uint64_t>(smallSDConstInfo.n2) * s2Dv;
        smallSDConstInfo.vStrideN2 = s2Dv;
    } else if (smallSDConstInfo.layout == SBNGD) {
        smallSDConstInfo.qStrideB = n2G * smallSDConstInfo.d;
        smallSDConstInfo.qStrideN2 = gD;
        smallSDConstInfo.kStrideB = static_cast<uint64_t>(smallSDConstInfo.n2) * smallSDConstInfo.d;
        smallSDConstInfo.kStrideN2 = smallSDConstInfo.d;
        smallSDConstInfo.vStrideB = static_cast<uint64_t>(smallSDConstInfo.n2) * smallSDConstInfo.dv;
        smallSDConstInfo.vStrideN2 = smallSDConstInfo.dv;
    } else {
        smallSDConstInfo.qStrideB = n2G * s1D;
        smallSDConstInfo.qStrideN2 = gD;
        smallSDConstInfo.kStrideB = static_cast<uint64_t>(smallSDConstInfo.n2) * s2D;
        smallSDConstInfo.kStrideN2 = smallSDConstInfo.d;
        smallSDConstInfo.vStrideB = static_cast<uint64_t>(smallSDConstInfo.n2) * s2Dv;
        smallSDConstInfo.vStrideN2 = smallSDConstInfo.dv;
    }
    smallSDConstInfo.dyStrideB = smallSDConstInfo.qStrideB;
    smallSDConstInfo.dyStrideN2 = smallSDConstInfo.qStrideN2;
    smallSDConstInfo.dqStrideB = smallSDConstInfo.qStrideB;
    smallSDConstInfo.dqStrideN2 = smallSDConstInfo.qStrideN2;
    smallSDConstInfo.dkStrideB = smallSDConstInfo.kStrideB;
    smallSDConstInfo.dkStrideN2 = smallSDConstInfo.kStrideN2;
    smallSDConstInfo.dvStrideB = smallSDConstInfo.vStrideB;
    smallSDConstInfo.dvStrideN2 = smallSDConstInfo.vStrideN2;

    smallSDConstInfo.qMatrixBytes = static_cast<uint64_t>(smallSDConstInfo.s1) * smallSDConstInfo.d *
                                    sizeof(INPUT_TYPE);
    smallSDConstInfo.kMatrixBytes = static_cast<uint64_t>(smallSDConstInfo.s2) * smallSDConstInfo.d *
                                    sizeof(INPUT_TYPE);
    smallSDConstInfo.vMatrixBytes = static_cast<uint64_t>(smallSDConstInfo.s2) * smallSDConstInfo.dv *
                                    sizeof(INPUT_TYPE);
    smallSDConstInfo.dyMatrixBytes = static_cast<uint64_t>(smallSDConstInfo.s1) * smallSDConstInfo.d *
                                     sizeof(INPUT_TYPE);
    smallSDConstInfo.dqMatrixBytes = static_cast<uint64_t>(smallSDConstInfo.s1) * smallSDConstInfo.d *
                                     sizeof(OUTDTYPE);
    smallSDConstInfo.dkMatrixBytes = static_cast<uint64_t>(smallSDConstInfo.s2) * smallSDConstInfo.d *
                                     sizeof(OUTDTYPE);
    smallSDConstInfo.dvMatrixBytes = static_cast<uint64_t>(smallSDConstInfo.s2) * smallSDConstInfo.dv *
                                     sizeof(OUTDTYPE);
    smallSDConstInfo.cubeResultBytes = static_cast<uint64_t>(smallSDConstInfo.s1) * smallSDConstInfo.s2 *
                                       sizeof(CALC_TYPE);
    smallSDConstInfo.vectorTempBytes = smallSDConstInfo.cubeResultBytes;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::SetSmallSDConstInfo()
{
    BuildSmallSDConstInfo();
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
    const uint64_t n2 = smallSDConstInfo.n2;
    const uint64_t n2G = n2 * smallSDConstInfo.g;
    const uint64_t n2D = n2 * smallSDConstInfo.d;
    const uint64_t n2Dv = n2 * smallSDConstInfo.dv;
    const uint64_t n2GD = n2G * smallSDConstInfo.d;
    const uint64_t n2GDv = n2G * smallSDConstInfo.dv;
    this->curBatchIdx++;
    this->curBatchTotalBaseIdx += n2G;
    this->curBatchTotalS1BOffset += actualS1Len * n2GD;
    this->curBatchTotalS2BOffset += actualS2Len * n2D;
    this->curBatchTotalS1BOffsetForDv += actualS1Len * n2GDv;
    this->curBatchTotalS2BOffsetForDv += actualS2Len * n2Dv;
    this->curBatchTotalS1S2SizeAlign += actualS1Len * AlignTo16(actualS2Len);
    this->curBatchTotalS1S2Size += actualS1Len * actualS2Len;
    this->curBatchTotalS2Size += actualS2Len;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::SetCursorGmOffsetSmallSD(
    SmallSDTaskCursor &cursor)
{
    // The initial SmallSD host route guarantees G == 1; GQA needs a separate cursor/accumulation extension.
    if constexpr (IS_TND) {
        cursor.qOffset = cursor.lastBatchTotalS1BOffset + cursor.n2oIdx * smallSDConstInfo.qStrideN2;
        cursor.kOffset = cursor.lastBatchTotalS2BOffset + cursor.n2oIdx * smallSDConstInfo.kStrideN2;
        cursor.qTaskStride = smallSDConstInfo.qStrideN2;
        cursor.kTaskStride = smallSDConstInfo.kStrideN2;
        cursor.qBatchGap = 0;
        cursor.kBatchGap = 0;
    } else {
        cursor.qOffset = cursor.bIdx * smallSDConstInfo.qStrideB + cursor.n2oIdx * smallSDConstInfo.qStrideN2;
        cursor.kOffset = cursor.bIdx * smallSDConstInfo.kStrideB + cursor.n2oIdx * smallSDConstInfo.kStrideN2;
        cursor.qTaskStride = smallSDConstInfo.qStrideN2;
        cursor.kTaskStride = smallSDConstInfo.kStrideN2;
        if (smallSDConstInfo.layout == BNGSD || smallSDConstInfo.layout == SBNGD) {
            cursor.qBatchGap = 0;
            cursor.kBatchGap = 0;
        } else {
            cursor.qBatchGap = smallSDConstInfo.qStrideB -
                               static_cast<uint64_t>(smallSDConstInfo.n2) * smallSDConstInfo.qStrideN2;
            cursor.kBatchGap = smallSDConstInfo.kStrideB -
                               static_cast<uint64_t>(smallSDConstInfo.n2) * smallSDConstInfo.kStrideN2;
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::LoadTndCursorPrefixSmallSD(
    SmallSDTaskCursor &cursor)
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
    SmallSDTaskCursor &cursor,
    int64_t index)
{
    if constexpr (IS_TND) {
        const auto &tndCore =
            GetSmallSDTilingData().tndCoreParam[this->cBlockIdx];
        const int64_t bIdx = tndCore.startBatchIdx;
        ReadTndSeqLenSmallSD(bIdx, cursor.actualS1Len, cursor.actualS2Len);
        cursor.bIdx = bIdx;
        // G == 1 in the SmallSD route, so the host-provided startN2 is the local N2 index.
        cursor.n2oIdx = tndCore.startN2Idx;
        LoadTndCursorPrefixSmallSD(cursor);
    } else {
        cursor.bIdx = index / smallSDConstInfo.n2;
        cursor.n2oIdx = index - cursor.bIdx * smallSDConstInfo.n2;
        cursor.actualS1Len = smallSDConstInfo.s1;
        cursor.actualS2Len = smallSDConstInfo.s2;
        cursor.s2SizeAcc = cursor.bIdx * smallSDConstInfo.s2;
        cursor.b1SSOffset = cursor.bIdx * smallSDConstInfo.s1 * smallSDConstInfo.s2;
        cursor.b1SSOffsetAlign =
            cursor.bIdx * smallSDConstInfo.s1 * AlignTo16(smallSDConstInfo.s2);
    }
    SetCursorGmOffsetSmallSD(cursor);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::IssueCube(
    SmallSDPipelineSlot &slot)
{
    if (slot.state != SmallSDSlotState::PREPARED) {
        return;
    }
    slot.state = SmallSDSlotState::CUBE_INFLIGHT;
    this->cubeBlock.IssueQkAndDyV(slot);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::AdvanceTaskCursor(
    SmallSDTaskCursor &cursor)
{
    cursor.n2oIdx++;
    if (cursor.n2oIdx < smallSDConstInfo.n2) {
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
        cursor.s2SizeAcc += smallSDConstInfo.s2;
        cursor.b1SSOffset += smallSDConstInfo.s1 * smallSDConstInfo.s2;
        cursor.b1SSOffsetAlign += smallSDConstInfo.s1 * AlignTo16(smallSDConstInfo.s2);
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline SmallSDShape
FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::MakeShapeSmallSD(
    const SmallSDTaskCursor &cursor) const
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
__aicore__ inline SmallSDOffsets
FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::MakeOffsetsSmallSD(
    const SmallSDTaskCursor &cursor) const
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
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::PrepareSlot(
    SmallSDPipelineSlot &slot,
    const SmallSDTaskCursor &cursor,
    int64_t taskId)
{
    if (slot.state != SmallSDSlotState::EMPTY && slot.state != SmallSDSlotState::REUSABLE) {
        return;
    }
    const SmallSDShape shape = MakeShapeSmallSD(cursor);
    const SmallSDOffsets offsets = MakeOffsetsSmallSD(cursor);

    slot.taskId = taskId;
    slot.taskIdMod2 = taskId & 1;
    slot.bIdx = cursor.bIdx;
    slot.n2oIdx = cursor.n2oIdx;
    slot.actualS1Len = shape.s1;
    slot.actualS2Len = shape.s2;
    slot.s2AlignedSize = shape.s2Align16;
    slot.qOffset = offsets.q;
    slot.kOffset = offsets.k;
    slot.attentionOffset = offsets.attention;
    slot.attentionAlignOffset = offsets.attentionAlign;
    slot.s2Prefix = offsets.s2Prefix;
    if constexpr (IS_TND) {
        slot.lastBatchTotalBaseIdx = cursor.lastBatchTotalBaseIdx;
        slot.lastBatchTotalS1BOffset = cursor.lastBatchTotalS1BOffset;
        slot.lastBatchTotalS2BOffset = cursor.lastBatchTotalS2BOffset;
        slot.lastBatchTotalS1BOffsetForDv = cursor.lastBatchTotalS1BOffsetForDv;
        slot.lastBatchTotalS2BOffsetForDv = cursor.lastBatchTotalS2BOffsetForDv;
        slot.lastBatchTotalS1S2SizeAlign = cursor.lastBatchTotalS1S2SizeAlign;
        slot.lastBatchTotalS1S2Size = cursor.lastBatchTotalS1S2Size;
        slot.lastBatchTotalS2Size = cursor.lastBatchTotalS2Size;
    }
    if ASCEND_IS_AIV {
        slot.halfS1 = shape.halfS1;
        slot.firstHalfS1 = shape.firstHalfS1;
        slot.halfS2 = shape.halfS2;
        slot.firstHalfS2 = shape.firstHalfS2;
        slot.vecCoreOffset = this->vSubBlockIdx * shape.firstHalfS1;
    }
    slot.state = SmallSDSlotState::PREPARED;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::ConsumeVector(
    SmallSDPipelineSlot &slot,
    bool needWaitL1Reusable)
{
    static_assert(BaseClass::IS_DQ_WRITE_UB, "SmallSD expects DQ to be written through UB.");
    static_assert(BaseClass::IS_DK_WRITE_UB, "SmallSD expects DK to be written through UB.");
    if (slot.state != SmallSDSlotState::READY_FOR_VECTOR) {
        return;
    }
    slot.state = SmallSDSlotState::VECTOR_INFLIGHT;
    this->vecBlock.ProduceDsAndP(slot, needWaitL1Reusable);
    this->cubeBlock.IssueDqDkDv(slot);
    this->vecBlock.FinalizeGradOutput(slot);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::WaitSlotReusable(
    SmallSDPipelineSlot &slot)
{
    if (slot.state == SmallSDSlotState::EMPTY || slot.state == SmallSDSlotState::REUSABLE) {
        return;
    }
    if (slot.state != SmallSDSlotState::VECTOR_INFLIGHT) {
        return;
    }
    if constexpr (BaseClass::IS_DK_WRITE_UB) {
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SMALL_SD_SLOT_REUSE_READY_FLAG);
            CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + SMALL_SD_SLOT_REUSE_READY_FLAG);
        }
    }
    slot.state = SmallSDSlotState::REUSABLE;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::DrainPipeline(
    SmallSDPipelineSlot (&slots)[2],
    int64_t lastSlotIdx, int64_t taskCount)
{
    if (taskCount <= 0) {
        return;
    }
    this->isLastLoop = true;
    WaitSlotReusable(slots[lastSlotIdx ^ 1]);
    const bool needWaitL1Reusable = taskCount > 1;
    ConsumeVector(slots[lastSlotIdx], needWaitL1Reusable);
    WaitSlotReusable(slots[lastSlotIdx]);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::ProcessSingleTask(
    SmallSDTaskCursor &cursor)
{
    SmallSDPipelineSlot slot;
    PrepareSlot(slot, cursor, 0);
    IssueCube(slot);
    this->isLastLoop = true;
    ConsumeVector(slot, false);
    WaitSlotReusable(slot);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::ProcessMultipleTasks(
    SmallSDTaskCursor &cursor,
    int64_t taskCount)
{
    SmallSDPipelineSlot slots[2];

    PrepareSlot(slots[0], cursor, 0);
    IssueCube(slots[0]);
    AdvanceTaskCursor(cursor);

    for (int64_t taskId = 1; taskId < taskCount; ++taskId) {
        this->isLastLoop = false;
        const int64_t fillSlotIdx = taskId & 1;
        const int64_t consumeSlotIdx = fillSlotIdx ^ 1;
        WaitSlotReusable(slots[fillSlotIdx]);
        PrepareSlot(slots[fillSlotIdx], cursor, taskId);
        IssueCube(slots[fillSlotIdx]);
        ConsumeVector(slots[consumeSlotIdx], taskId > 1);
        if (taskId + 1 < taskCount) {
            AdvanceTaskCursor(cursor);
        }
    }

    DrainPipeline(slots, (taskCount - 1) & 1, taskCount);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::Process()
{
    const auto &coreTask =
        GetSmallSDTilingData().coreTaskParam[this->cBlockIdx];
    const int64_t blockStart = coreTask.blockStart;
    const int64_t groupCount = coreTask.groupCount;
    if (groupCount <= 0) {
        return;
    }
    if constexpr (IS_TND) {
        const auto &tndCore =
            GetSmallSDTilingData().tndCoreParam[this->cBlockIdx];
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
    SmallSDTaskCursor cursor;
    InitTaskCursorSmallSD(cursor, blockStart);

    if (groupCount == 1) {
        ProcessSingleTask(cursor);
        return;
    }

    ProcessMultipleTasks(cursor, groupCount);
}

} // namespace FagBaseApi

#endif
