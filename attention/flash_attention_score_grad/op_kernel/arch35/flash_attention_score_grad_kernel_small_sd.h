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
#include "flash_attention_score_grad_buffer_small_sd.h"
#include "flash_attention_score_grad_event_small_sd.h"
#include "cube_api/mutex_buffer.h"
#include "cube_api/mutex_buffers_policy.h"
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
// - SMALL_SD_CUBE_OUTPUT_COMMIT_FLAG: Cube-side DV Fixpipe output has been drained before Process returns.
// SmallSD is entered only by the arch35 regbase key with:
// FP16/BF16, BN2, G=1, N1=N2, D==Dv, 0<S1/S2<128, 0<D<128,
// no optional feature, no deterministic path, no NZ output, and no swizzle/remap.
// L0A/L0B double buffering is owned by SmallSDCubeBlock; this class trims scalar-heavy scheduling,
// validity search, and generic offset helpers from the SmallSD runtime path.
template <typename CubeBlockType, typename VecBlockType>
class FlashAttentionScoreGradKernelSmallSD {
public:
    using INPUT_TYPE = typename CubeBlockTraits<CubeBlockType>::INPUT_TYPE_TRAITS;
    using CALC_TYPE = typename CubeBlockTraits<CubeBlockType>::CALC_TYPE_TRAITS;
    using OUTDTYPE = typename CubeBlockTraits<CubeBlockType>::OUTDTYPE_TRAITS;
    static constexpr bool IS_ATTEN_MASK = CubeBlockTraits<CubeBlockType>::IS_ATTEN_MASKTraits;
    static constexpr bool IS_PSE = CubeBlockTraits<CubeBlockType>::IS_PSETraits;
    static constexpr bool IS_DROP = CubeBlockTraits<CubeBlockType>::IS_DROPTraits;
    static constexpr bool IS_TND = CubeBlockTraits<CubeBlockType>::IS_TNDTraits;
    static constexpr bool IS_BN2_MULTIBLK = CubeBlockTraits<CubeBlockType>::IS_BN2_MULTIBLKTraits;
    static constexpr uint8_t DETER_SPARSE_TYPE = CubeBlockTraits<CubeBlockType>::DETER_SPARSE_TYPETraits;
    static constexpr bool IS_N_EQUAL = CubeBlockTraits<CubeBlockType>::IS_N_EQUALTraits;
    static constexpr bool IS_D_NO_EQUAL = CubeBlockTraits<CubeBlockType>::IS_D_NO_EQUALTraits;
    static constexpr bool IS_ROPE = CubeBlockTraits<CubeBlockType>::IS_ROPETraits;
    static constexpr bool IS_NZ_OUT = CubeBlockTraits<CubeBlockType>::IS_NZ_OUTTraits;
    static constexpr bool IS_TND_SWIZZLE = CubeBlockTraits<CubeBlockType>::IS_TND_SWIZZLETraits;
    static constexpr uint8_t SPLIT_AXIS = CubeBlockTraits<CubeBlockType>::SPLIT_AXISTraits;
    static constexpr S1TemplateType s1TemplateType = CubeBlockTraits<CubeBlockType>::s1TemplateTypeTraits;
    static constexpr S2TemplateType s2TemplateType = CubeBlockTraits<CubeBlockType>::s2TemplateTypeTraits;
    static constexpr DTemplateType dTemplateType = CubeBlockTraits<CubeBlockType>::dTemplateTypeTraits;
    static constexpr bool IS_FP32_INPUT = IsSameType<INPUT_TYPE, float>::value;
    static constexpr uint32_t CUBE_BASEM = static_cast<uint32_t>(s1TemplateType);
    static constexpr uint32_t CUBE_BASEN = static_cast<uint32_t>(s2TemplateType);
    static constexpr uint32_t HEAD_DIM_ALIGN = static_cast<uint32_t>(dTemplateType);
    using SmallSDFagTilingType = const __gm__ optiling::fag::SmallSDTilingDataRegbase *__restrict;

    __aicore__ inline void Init(GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR query, GM_ADDR pseShift,
                                GM_ADDR dropMask, GM_ADDR attenMask, GM_ADDR y, GM_ADDR softmaxMax, GM_ADDR softmaxSum,
                                GM_ADDR prefixN, GM_ADDR actualSeqQlen, GM_ADDR actualSeqKvlen, GM_ADDR deqScaleQ,
                                GM_ADDR deqScaleK, GM_ADDR deqScaleV, GM_ADDR deqScaleDy, GM_ADDR queryRope,
                                GM_ADDR keyRope, GM_ADDR sink, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dpse,
                                GM_ADDR dqRope, GM_ADDR dkRope, GM_ADDR dsink, GM_ADDR workspace,
                                SmallSDFagTilingType ordTilingData, TPipe *pipeIn);
    __aicore__ inline void Process();
    __aicore__ inline void ReadTndSeqLenSmallSD(int64_t batchIdx, int64_t &actualS1Len, int64_t &actualS2Len);
    __aicore__ inline void AdvanceTndBatchPrefixSmallSD(int64_t actualS1Len, int64_t actualS2Len);
    __aicore__ inline void SetSmallSDConstInfo();

private:
    __aicore__ inline SmallSDFagTilingType GetSmallSDTilingData() const;
    __aicore__ inline void InitSmallSDBlocks(GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR query,
                                             GM_ADDR y, GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR dq, GM_ADDR dk,
                                             GM_ADDR dv, GM_ADDR workspace, TPipe *pipeIn);
    __aicore__ inline void InitSmallSDSharedBuffers(TPipe *pipeIn);
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
    __aicore__ inline void WaitOutputCommitComplete();
    __aicore__ inline void ProcessSingleTask(SmallSDTaskCursor &cursor);
    __aicore__ inline void ProcessMultipleTasks(SmallSDTaskCursor &cursor, int64_t taskCount);

    SmallSDConstInfo smallSDConstInfo;
    SmallSDFagTilingType tilingData = nullptr;
    GM_ADDR actualSeqQlenAddr = nullptr;
    GM_ADDR actualSeqKvlenAddr = nullptr;
    uint32_t vBlockIdx = 0;
    uint32_t cBlockIdx = 0;
    uint32_t vSubBlockIdx = 0;
    int64_t curBatchIdx = 0;
    int64_t curBaseTaskIndex = 0;
    int64_t curQSeqPrefix = 0;
    int64_t curKvSeqPrefix = 0;
    int64_t curQDyDqElementOffset = 0;
    int64_t curKvDkDvElementOffset = 0;
    int64_t curAttentionElementPrefix = 0;
    int64_t curAlignedAttentionElementPrefix = 0;
    int64_t curSoftmaxRowPrefix = 0;
    CubeBlockType cubeBlock;
    VecBlockType vecBlock;
    TBuf<> smallSDMm1ResBuf[2];
    TBuf<> smallSDMm2ResBuf[2];
    MutexBufferManager<BufferType::L1> smallSDL1BufferManager;
    MutexBuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> smallSDDsL1Buf;
    MutexBuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> smallSDPL1Buf;
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
    GM_ADDR workspace, SmallSDFagTilingType ordTilingData, TPipe *pipeIn)
{
    (void)pseShift;
    (void)dropMask;
    (void)attenMask;
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
        vBlockIdx = GetBlockIdx();
        cBlockIdx = vBlockIdx / CV_CORE_RATIO;
        vSubBlockIdx = GetSubBlockIdx();
    } else {
        cBlockIdx = GetBlockIdx();
    }
    tilingData = ordTilingData;

    SetSmallSDConstInfo();

    actualSeqQlenAddr = actualSeqQlen;
    actualSeqKvlenAddr = actualSeqKvlen;

    InitSmallSDSharedBuffers(pipeIn);
    InitSmallSDBlocks(key, value, dy, query, y, softmaxMax, softmaxSum, dq, dk, dv, workspace, pipeIn);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::InitSmallSDSharedBuffers(
    TPipe *pipeIn)
{
    static constexpr uint32_t SMALL_SD_VECTOR_BASEM = 64;
    static constexpr uint32_t SCORE_UB_BYTES = SMALL_SD_VECTOR_BASEM * 128 * sizeof(CALC_TYPE);
    static constexpr uint32_t GRAD_UB_BYTES = SMALL_SD_VECTOR_BASEM * HEAD_DIM_ALIGN * sizeof(CALC_TYPE);
    static constexpr uint32_t DS_P_L1_BYTES = 128 * 128 * sizeof(INPUT_TYPE);
    smallSDL1BufferManager.Init(pipeIn, L1_MAX_SIZE);
    smallSDDsL1Buf.Init(smallSDL1BufferManager, DS_P_L1_BYTES);
    smallSDPL1Buf.Init(smallSDL1BufferManager, DS_P_L1_BYTES);
    pipeIn->InitBuffer(smallSDMm1ResBuf[0], SCORE_UB_BYTES > GRAD_UB_BYTES ? SCORE_UB_BYTES : GRAD_UB_BYTES);
    pipeIn->InitBuffer(smallSDMm1ResBuf[1], SCORE_UB_BYTES > GRAD_UB_BYTES ? SCORE_UB_BYTES : GRAD_UB_BYTES);
    pipeIn->InitBuffer(smallSDMm2ResBuf[0], SCORE_UB_BYTES > GRAD_UB_BYTES ? SCORE_UB_BYTES : GRAD_UB_BYTES);
    pipeIn->InitBuffer(smallSDMm2ResBuf[1], SCORE_UB_BYTES > GRAD_UB_BYTES ? SCORE_UB_BYTES : GRAD_UB_BYTES);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::InitSmallSDBlocks(
    GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR query, GM_ADDR y, GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR dq,
    GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace, TPipe *pipeIn)
{
    cubeBlock.Init(query, key, value, dy, dq, dk, dv, workspace, pipeIn, &smallSDConstInfo, smallSDMm1ResBuf,
                   smallSDMm2ResBuf, &smallSDL1BufferManager, &smallSDDsL1Buf, &smallSDPL1Buf);
    vecBlock.Init(value, dy, y, softmaxMax, softmaxSum, dq, dk, dv, workspace, pipeIn, &smallSDConstInfo,
                  vSubBlockIdx, smallSDMm1ResBuf, smallSDMm2ResBuf, &smallSDDsL1Buf, &smallSDPL1Buf);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline typename FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::SmallSDFagTilingType
FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::GetSmallSDTilingData() const
{
    return tilingData;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::BuildSmallSDConstInfo()
{
    const auto &baseParam = GetSmallSDTilingData()->baseParam;
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
    smallSDConstInfo.tndMaxSumLayout = baseParam.tndMaxSumLayout;
    smallSDConstInfo.scale = baseParam.scaleValue;
    smallSDConstInfo.workspaceBaseOffset = baseParam.workspaceBaseOffset;
    smallSDConstInfo.workspaceSize = baseParam.workspaceSize;

    const auto &layoutParam = GetSmallSDTilingData()->layoutParam;
    smallSDConstInfo.qStrideB = layoutParam.qStrideB;
    smallSDConstInfo.qStrideN2 = layoutParam.qStrideN2;
    smallSDConstInfo.qStrideS = layoutParam.qStrideS;
    smallSDConstInfo.kStrideB = layoutParam.kStrideB;
    smallSDConstInfo.kStrideN2 = layoutParam.kStrideN2;
    smallSDConstInfo.kStrideS = layoutParam.kStrideS;
    smallSDConstInfo.vStrideB = layoutParam.vStrideB;
    smallSDConstInfo.vStrideN2 = layoutParam.vStrideN2;
    smallSDConstInfo.vStrideS = layoutParam.vStrideS;
    smallSDConstInfo.dyStrideB = layoutParam.dyStrideB;
    smallSDConstInfo.dyStrideN2 = layoutParam.dyStrideN2;
    smallSDConstInfo.dyStrideS = layoutParam.dyStrideS;
    smallSDConstInfo.dqStrideB = layoutParam.dqStrideB;
    smallSDConstInfo.dqStrideN2 = layoutParam.dqStrideN2;
    smallSDConstInfo.dqStrideS = layoutParam.dqStrideS;
    smallSDConstInfo.dkStrideB = layoutParam.dkStrideB;
    smallSDConstInfo.dkStrideN2 = layoutParam.dkStrideN2;
    smallSDConstInfo.dkStrideS = layoutParam.dkStrideS;
    smallSDConstInfo.dvStrideB = layoutParam.dvStrideB;
    smallSDConstInfo.dvStrideN2 = layoutParam.dvStrideN2;
    smallSDConstInfo.dvStrideS = layoutParam.dvStrideS;
    smallSDConstInfo.attentionStrideB = layoutParam.attentionStrideB;
    smallSDConstInfo.attentionStrideN2 = layoutParam.attentionStrideN2;
    smallSDConstInfo.attentionStrideS = layoutParam.attentionStrideS;
    smallSDConstInfo.softmaxStrideB = layoutParam.softmaxStrideB;
    smallSDConstInfo.softmaxStrideN2 = layoutParam.softmaxStrideN2;
    smallSDConstInfo.softmaxStrideS = layoutParam.softmaxStrideS;
    smallSDConstInfo.qMatrixElements = layoutParam.qMatrixElements;
    smallSDConstInfo.kMatrixElements = layoutParam.kMatrixElements;
    smallSDConstInfo.vMatrixElements = layoutParam.vMatrixElements;
    smallSDConstInfo.dyMatrixElements = layoutParam.dyMatrixElements;
    smallSDConstInfo.dqMatrixElements = layoutParam.dqMatrixElements;
    smallSDConstInfo.dkMatrixElements = layoutParam.dkMatrixElements;
    smallSDConstInfo.dvMatrixElements = layoutParam.dvMatrixElements;
    smallSDConstInfo.cubeResultElements = layoutParam.cubeResultElements;
    smallSDConstInfo.vectorTempElements = layoutParam.vectorTempElements;
    smallSDConstInfo.qMatrixBytes = layoutParam.qMatrixBytes;
    smallSDConstInfo.kMatrixBytes = layoutParam.kMatrixBytes;
    smallSDConstInfo.vMatrixBytes = layoutParam.vMatrixBytes;
    smallSDConstInfo.dyMatrixBytes = layoutParam.dyMatrixBytes;
    smallSDConstInfo.dqMatrixBytes = layoutParam.dqMatrixBytes;
    smallSDConstInfo.dkMatrixBytes = layoutParam.dkMatrixBytes;
    smallSDConstInfo.dvMatrixBytes = layoutParam.dvMatrixBytes;
    smallSDConstInfo.cubeResultBytes = layoutParam.cubeResultBytes;
    smallSDConstInfo.vectorTempBytes = layoutParam.vectorTempBytes;
    smallSDConstInfo.dTemplateCapacity = layoutParam.dTemplateCapacity;
    smallSDConstInfo.aivHalfS1 = layoutParam.aivHalfS1;
    smallSDConstInfo.aivFirstHalfS1 = layoutParam.aivFirstHalfS1;
    smallSDConstInfo.aivHalfS2 = layoutParam.aivHalfS2;
    smallSDConstInfo.aivFirstHalfS2 = layoutParam.aivFirstHalfS2;
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
    __gm__ int64_t *actualSeqQCumEnd = (__gm__ int64_t *)actualSeqQlenAddr;
    __gm__ int64_t *actualSeqKvCumEnd = (__gm__ int64_t *)actualSeqKvlenAddr;
    if (batchIdx == 0) {
        actualS1Len = actualSeqQCumEnd[0];
        actualS2Len = actualSeqKvCumEnd[0];
    } else {
        actualS1Len = actualSeqQCumEnd[batchIdx] - actualSeqQCumEnd[batchIdx - 1];
        actualS2Len = actualSeqKvCumEnd[batchIdx] - actualSeqKvCumEnd[batchIdx - 1];
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
    const uint64_t n2GD = n2G * smallSDConstInfo.d;
    curBaseTaskIndex += n2;
    curQSeqPrefix += actualS1Len;
    curKvSeqPrefix += actualS2Len;
    curQDyDqElementOffset += actualS1Len * n2GD;
    curKvDkDvElementOffset += actualS2Len * n2D;
    curAttentionElementPrefix += actualS1Len * actualS2Len;
    curAlignedAttentionElementPrefix += actualS1Len * AlignTo16(actualS2Len);
    curSoftmaxRowPrefix += actualS1Len;
    curBatchIdx++;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::SetCursorGmOffsetSmallSD(
    SmallSDTaskCursor &cursor)
{
    // The initial SmallSD host route guarantees G == 1; GQA needs a separate cursor/accumulation extension.
    if constexpr (IS_TND) {
        cursor.qOffset = cursor.qDyDqElementOffset + cursor.n2oIdx * smallSDConstInfo.qStrideN2;
        cursor.kOffset = cursor.kvDkDvElementOffset + cursor.n2oIdx * smallSDConstInfo.kStrideN2;
        cursor.qTaskStride = smallSDConstInfo.qStrideN2;
        cursor.kTaskStride = smallSDConstInfo.kStrideN2;
        cursor.qBatchGap = 0;
        cursor.kBatchGap = 0;
    } else {
        cursor.qOffset = cursor.bIdx * smallSDConstInfo.qStrideB + cursor.n2oIdx * smallSDConstInfo.qStrideN2;
        cursor.kOffset = cursor.bIdx * smallSDConstInfo.kStrideB + cursor.n2oIdx * smallSDConstInfo.kStrideN2;
        cursor.qTaskStride = smallSDConstInfo.qStrideN2;
        cursor.kTaskStride = smallSDConstInfo.kStrideN2;
        cursor.qBatchGap = smallSDConstInfo.qStrideB -
                           static_cast<uint64_t>(smallSDConstInfo.n2) * smallSDConstInfo.qStrideN2;
        cursor.kBatchGap = smallSDConstInfo.kStrideB -
                           static_cast<uint64_t>(smallSDConstInfo.n2) * smallSDConstInfo.kStrideN2;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::LoadTndCursorPrefixSmallSD(
    SmallSDTaskCursor &cursor)
{
    cursor.baseTaskIndex = curBaseTaskIndex;
    cursor.qSeqPrefix = curQSeqPrefix;
    cursor.kvSeqPrefix = curKvSeqPrefix;
    cursor.qDyDqElementOffset = curQDyDqElementOffset;
    cursor.kvDkDvElementOffset = curKvDkDvElementOffset;
    cursor.softmaxRowPrefix = curSoftmaxRowPrefix;
    cursor.alignedAttentionElementPrefix = curAlignedAttentionElementPrefix;
    cursor.attentionElementPrefix = curAttentionElementPrefix;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::InitTaskCursorSmallSD(
    SmallSDTaskCursor &cursor,
    int64_t index)
{
    if constexpr (IS_TND) {
        const auto &tndCore =
            GetSmallSDTilingData()->tndCoreParam[cBlockIdx];
        const int64_t bIdx = tndCore.startBatchIdx;
        ReadTndSeqLenSmallSD(bIdx, cursor.actualS1Len, cursor.actualS2Len);
        cursor.bIdx = bIdx;
        // G == 1 in the SmallSD route, so the host-provided startN2 is the local N2 index.
        cursor.n2oIdx = tndCore.startN2Idx;
        LoadTndCursorPrefixSmallSD(cursor);
    } else {
        (void)index;
        const auto &coreTask = GetSmallSDTilingData()->coreTaskParam[cBlockIdx];
        cursor.bIdx = coreTask.startBatchIdx;
        cursor.n2oIdx = coreTask.startN2Idx;
        cursor.actualS1Len = smallSDConstInfo.s1;
        cursor.actualS2Len = smallSDConstInfo.s2;
        cursor.softmaxRowPrefix = cursor.bIdx * smallSDConstInfo.s2;
        cursor.attentionElementPrefix = cursor.bIdx * smallSDConstInfo.s1 * smallSDConstInfo.s2;
        cursor.alignedAttentionElementPrefix =
            cursor.bIdx * smallSDConstInfo.s1 * AlignTo16(smallSDConstInfo.s2);
    }
    SetCursorGmOffsetSmallSD(cursor);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::IssueCube(
    SmallSDPipelineSlot &slot)
{
    cubeBlock.IssueQkAndDyV(slot);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::AdvanceTaskCursor(
    SmallSDTaskCursor &cursor)
{
    if (cursor.taskRemaining > 0) {
        cursor.taskRemaining--;
    }
    cursor.n2oIdx++;
    if (cursor.n2oIdx < smallSDConstInfo.n2) {
        cursor.qOffset += cursor.qTaskStride;
        cursor.kOffset += cursor.kTaskStride;
        return;
    }
    cursor.n2oIdx = 0;

    if constexpr (IS_TND) {
        AdvanceTndBatchPrefixSmallSD(cursor.actualS1Len, cursor.actualS2Len);
        cursor.bIdx = curBatchIdx;
        ReadTndSeqLenSmallSD(cursor.bIdx, cursor.actualS1Len, cursor.actualS2Len);
        LoadTndCursorPrefixSmallSD(cursor);
        SetCursorGmOffsetSmallSD(cursor);
    } else {
        cursor.bIdx++;
        cursor.qOffset += cursor.qTaskStride + cursor.qBatchGap;
        cursor.kOffset += cursor.kTaskStride + cursor.kBatchGap;
        cursor.softmaxRowPrefix += smallSDConstInfo.s2;
        cursor.attentionElementPrefix += smallSDConstInfo.s1 * smallSDConstInfo.s2;
        cursor.alignedAttentionElementPrefix += smallSDConstInfo.s1 * AlignTo16(smallSDConstInfo.s2);
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
        if (vSubBlockIdx == 1) {
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
    offsets.attention = cursor.attentionElementPrefix;
    offsets.attentionAlign = cursor.alignedAttentionElementPrefix;
    offsets.softmaxRowPrefix = cursor.softmaxRowPrefix;
    return offsets;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::PrepareSlot(
    SmallSDPipelineSlot &slot,
    const SmallSDTaskCursor &cursor,
    int64_t taskId)
{
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
    slot.vOffset = offsets.k;
    slot.dyOffset = offsets.q;
    slot.dqOffset = offsets.q;
    slot.dkOffset = offsets.k;
    slot.dvOffset = offsets.k;
    slot.attentionOffset = offsets.attention;
    slot.attentionAlignOffset = offsets.attentionAlign;
    slot.softmaxRowPrefix = offsets.softmaxRowPrefix;
    if constexpr (IS_TND) {
        slot.baseTaskIndex = cursor.baseTaskIndex;
        slot.qSeqPrefix = cursor.qSeqPrefix;
        slot.kvSeqPrefix = cursor.kvSeqPrefix;
        slot.qDyDqElementOffset = cursor.qDyDqElementOffset;
        slot.kvDkDvElementOffset = cursor.kvDkDvElementOffset;
    }
    if ASCEND_IS_AIV {
        slot.halfS1 = shape.halfS1;
        slot.firstHalfS1 = shape.firstHalfS1;
        slot.halfS2 = shape.halfS2;
        slot.firstHalfS2 = shape.firstHalfS2;
        slot.vecCoreOffset = vSubBlockIdx * shape.firstHalfS1;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::ConsumeVector(
    SmallSDPipelineSlot &slot,
    bool needWaitL1Reusable)
{
    vecBlock.ProduceDsAndP(slot, needWaitL1Reusable);
    cubeBlock.IssueDqDkDv(slot);
    vecBlock.FinalizeGradOutput(slot);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::WaitSlotReusable(
    SmallSDPipelineSlot &slot)
{
    if (slot.taskId < 0) {
        return;
    }
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SMALL_SD_SLOT_REUSE_READY_FLAG);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SMALL_SD_EVENT_MIRROR_OFFSET + SMALL_SD_SLOT_REUSE_READY_FLAG);
    }
    slot.taskId = -1;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::DrainPipeline(
    SmallSDPipelineSlot (&slots)[2],
    int64_t lastSlotIdx, int64_t taskCount)
{
    if (taskCount <= 0) {
        return;
    }
    WaitSlotReusable(slots[lastSlotIdx ^ 1]);
    const bool needWaitL1Reusable = taskCount > 1;
    ConsumeVector(slots[lastSlotIdx], needWaitL1Reusable);
    WaitSlotReusable(slots[lastSlotIdx]);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::WaitOutputCommitComplete()
{
    cubeBlock.WaitOutputCommitComplete();
    vecBlock.WaitOutputCommitComplete();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::ProcessSingleTask(
    SmallSDTaskCursor &cursor)
{
    SmallSDPipelineSlot slot = {};
    PrepareSlot(slot, cursor, 0);
    IssueCube(slot);
    ConsumeVector(slot, false);
    WaitSlotReusable(slot);
    WaitOutputCommitComplete();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::ProcessMultipleTasks(
    SmallSDTaskCursor &cursor,
    int64_t taskCount)
{
    SmallSDPipelineSlot slots[2] = {};

    PrepareSlot(slots[0], cursor, 0);
    IssueCube(slots[0]);
    AdvanceTaskCursor(cursor);

    for (int64_t taskId = 1; taskId < taskCount; ++taskId) {
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
    WaitOutputCommitComplete();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::Process()
{
    const auto &coreTask =
        GetSmallSDTilingData()->coreTaskParam[cBlockIdx];
    const int64_t blockStart = coreTask.blockStart;
    int64_t groupCount = coreTask.groupCount;
    if (groupCount <= 0) {
        return;
    }
    if constexpr (IS_TND) {
        const auto &tndCore =
            GetSmallSDTilingData()->tndCoreParam[cBlockIdx];
        groupCount = tndCore.taskCount;
        curBatchIdx = tndCore.startBatchIdx;
        curBaseTaskIndex = tndCore.baseTaskIndex;
        curQSeqPrefix = tndCore.qSeqPrefix;
        curKvSeqPrefix = tndCore.kvSeqPrefix;
        curQDyDqElementOffset = tndCore.qDyDqElementOffset;
        curKvDkDvElementOffset = tndCore.kvDkDvElementOffset;
        curAttentionElementPrefix = tndCore.attentionElementPrefix;
        curAlignedAttentionElementPrefix = tndCore.alignedAttentionElementPrefix;
        curSoftmaxRowPrefix = tndCore.softmaxRowPrefix;
        cachedTndBatchIdx = -1;
        if (groupCount <= 0) {
            return;
        }
    }
    SmallSDTaskCursor cursor;
    InitTaskCursorSmallSD(cursor, blockStart);
    cursor.taskRemaining = groupCount;

    if (groupCount == 1) {
        ProcessSingleTask(cursor);
        return;
    }

    ProcessMultipleTasks(cursor, groupCount);
}

} // namespace FagBaseApi

#endif
