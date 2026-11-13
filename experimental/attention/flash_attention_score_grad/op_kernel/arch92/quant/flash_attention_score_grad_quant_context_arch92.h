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
 * \file flash_attention_score_grad_quant_context_arch92.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_QUANT_CONTEXT_ARCH92_H_
#define FLASH_ATTENTION_SCORE_GRAD_QUANT_CONTEXT_ARCH92_H_

// Reuse common control helpers; the arch92 tiling ABI has 64 core entries.
#include "flash_attention_score_grad_common_quant_arch92.h"
#include "flash_attention_score_grad_kernel_base_quant_arch92.h"
#include "flash_attention_score_grad_tiling_data_regbase_arch92.h"
#include "flash_attention_score_grad_deter_quant_arch92.h"

namespace FagBaseApi {
namespace quant {

template <typename CubeBlockType, typename VecBlockType>
class FlashAttentionScoreGradQuantContext
    : public FlashAttentionScoreGradKernelBase<FlashAttentionScoreGradQuantContext<CubeBlockType, VecBlockType>,
                                               CubeBlockType, VecBlockType> {
public:
    ARGS_TRAITS;
    static constexpr uint64_t SYNC_MODE = 4;
    constexpr static uint32_t CUBE_BASEM = static_cast<uint32_t>(s1TemplateType);
    constexpr static uint32_t CUBE_BASEN = static_cast<uint32_t>(s2TemplateType);

    BufferManager<BufferType::L1> l1BufferManager;
    BuffersPolicy4buff<BufferType::L1, SyncType::NO_SYNC> pL1Buf;

    BuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> dsL1Buf[4][4];

    GlobalTensor<OUTDTYPE> dqGm, dkGm, dvGm;
    GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm;
    GlobalTensor<float> deqScaleQGm, deqScaleKGm, deqScaleVGm, deqScaleDyGm, pScaleGm, dsScaleGm;

    FagTilingType tilingData;
    TPipe* pipe;
    using BaseClass =
        FlashAttentionScoreGradKernelBase<FlashAttentionScoreGradQuantContext<CubeBlockType, VecBlockType>,
                                          CubeBlockType, VecBlockType>;

    __aicore__ inline void SetConstInfo();

    __aicore__ inline void InitCVCommonGlobalBuffer(GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR deqScaleQ,
                                                    GM_ADDR deqScaleK, GM_ADDR deqScaleV, GM_ADDR deqScaleDy,
                                                    GM_ADDR dsScale, GM_ADDR pScale, GM_ADDR workspace);
    __aicore__ inline void SetRunInfo(FagRunInfo& runInfo, FagRunInfo& lastRunInfo, int64_t taskId,
                                      CoordinateInfo& coordinateInfo, CoordinateInfo& nextCoordinateInfo);

    __aicore__ inline int64_t GetDeqScaleKOffset(FagRunInfo& runInfo);
    __aicore__ inline int64_t GetDeqScaleQOffset(FagRunInfo& runInfo);
    __aicore__ inline int64_t CalDeterMaxLoopNum();
    __aicore__ inline void CalDeterIndex(uint32_t roundId, uint32_t maxLoopNum, int64_t& nextValidRoundId,
                                         int64_t& nextValidIndex, int64_t taskId, CoordinateInfo& coordinateInfo,
                                         FagRunInfo& runInfo);
    __aicore__ inline int64_t CalDenseDeterIndex(uint32_t roundId, uint32_t maxLoopNum, CoordinateInfo& coordinateInfo);

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
    int64_t lastS2Idx = -1;
    int64_t lastBatchIdx = -1;
    int64_t lastN2Idx = -1;

    int64_t innerN2GD;

    TBuf<> vecQue;
    LocalTensor<CALC_TYPE> commonTensors; // Cached statistics and permutation table.

    CoordinateInfo coordinateInfos[2];
    CoordinateInfo nextCoreFirstBlockCoordinateInfo;
};

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradQuantContext<CubeBlockType, VecBlockType>::SetConstInfo()
{
    // split info
    this->constInfo.s1Outer = tilingData->s1s2BNGS1S2SplitCoreParams.s1Outer;
    this->constInfo.s1CvTail = tilingData->s1s2BNGS1S2SplitCoreParams.s1CvTail;
    this->constInfo.s2Tail = tilingData->s1s2BNGS1S2SplitCoreParams.s2Tail;
    this->constInfo.s2Outer = tilingData->s1s2BNGS1S2SplitCoreParams.s2Outer;

    this->constInfo.commonConstInfo.s1BaseSize = CUBE_BASEM;
    this->constInfo.commonConstInfo.s2BaseSize = CUBE_BASEN;
    this->constInfo.bSize = tilingData->s1s2BNGS1S2BaseParams.b;
    this->constInfo.n2Size = tilingData->s1s2BNGS1S2BaseParams.n2;
    this->constInfo.commonConstInfo.gSize = tilingData->s1s2BNGS1S2BaseParams.g;
    this->constInfo.commonConstInfo.s1Size = tilingData->s1s2BNGS1S2BaseParams.s1;
    this->constInfo.commonConstInfo.s2Size = tilingData->s1s2BNGS1S2BaseParams.s2;
    this->constInfo.commonConstInfo.dSize = tilingData->s1s2BNGS1S2BaseParams.d;
    this->constInfo.commonConstInfo.dSizeV = tilingData->s1s2BNGS1S2BaseParams.d1;
    this->constInfo.commonConstInfo.layoutType = tilingData->s1s2BNGS1S2BaseParams.layout;

    this->constInfo.commonConstInfo.s1D =
        this->constInfo.commonConstInfo.s1Size * this->constInfo.commonConstInfo.dSize;
    this->constInfo.commonConstInfo.gS1D = this->constInfo.commonConstInfo.gSize * this->constInfo.commonConstInfo.s1D;
    this->constInfo.commonConstInfo.n2GS1D = this->constInfo.n2Size * this->constInfo.commonConstInfo.gS1D;
    this->constInfo.commonConstInfo.s2D =
        this->constInfo.commonConstInfo.s2Size * this->constInfo.commonConstInfo.dSize;
    this->constInfo.commonConstInfo.n2S2D = this->constInfo.n2Size * this->constInfo.commonConstInfo.s2D;
    this->constInfo.commonConstInfo.s1S2 =
        this->constInfo.commonConstInfo.s1Size * this->constInfo.commonConstInfo.s2Size;
    this->constInfo.commonConstInfo.gS1 =
        this->constInfo.commonConstInfo.gSize * this->constInfo.commonConstInfo.s1Size;
    this->constInfo.commonConstInfo.gD = this->constInfo.commonConstInfo.gSize * this->constInfo.commonConstInfo.dSize;
    this->constInfo.commonConstInfo.n2D = this->constInfo.n2Size * this->constInfo.commonConstInfo.dSize;
    this->constInfo.commonConstInfo.bN2D = this->constInfo.bSize * this->constInfo.commonConstInfo.n2D;
    this->constInfo.commonConstInfo.n2G = this->constInfo.n2Size * this->constInfo.commonConstInfo.gSize;
    this->constInfo.commonConstInfo.n2GD = this->constInfo.commonConstInfo.n2G * this->constInfo.commonConstInfo.dSize;
    this->constInfo.commonConstInfo.bN2GD = this->constInfo.bSize * this->constInfo.commonConstInfo.n2GD;
    this->constInfo.commonConstInfo.gS2 =
        this->constInfo.commonConstInfo.gSize * this->constInfo.commonConstInfo.s2Size;
    // for D_V
    this->constInfo.commonConstInfo.s1Dv = this->constInfo.commonConstInfo.s1D;
    this->constInfo.commonConstInfo.gS1Dv = this->constInfo.commonConstInfo.gS1D;
    this->constInfo.commonConstInfo.n2GS1Dv = this->constInfo.commonConstInfo.n2GS1D;
    this->constInfo.commonConstInfo.s2Dv = this->constInfo.commonConstInfo.s2D;
    this->constInfo.commonConstInfo.n2S2Dv = this->constInfo.commonConstInfo.n2S2D;
    this->constInfo.commonConstInfo.gDv = this->constInfo.commonConstInfo.gD;
    this->constInfo.commonConstInfo.n2Dv = this->constInfo.commonConstInfo.n2D;
    this->constInfo.commonConstInfo.bN2Dv = this->constInfo.commonConstInfo.bN2D;
    this->constInfo.commonConstInfo.n2GDv = this->constInfo.commonConstInfo.n2GD;
    this->constInfo.commonConstInfo.bN2GDv = this->constInfo.commonConstInfo.bN2GD;

    this->constInfo.scaleValue = tilingData->s1s2BNGS1S2BaseParams.scaleValue;
    this->constInfo.n2GS1oS2o = this->constInfo.commonConstInfo.n2G * this->constInfo.s1Outer * this->constInfo.s2Outer;
    this->constInfo.gS1oS2o = this->constInfo.commonConstInfo.gSize * this->constInfo.s1Outer * this->constInfo.s2Outer;
    this->constInfo.s1oS2o = this->constInfo.s1Outer * this->constInfo.s2Outer;
    if ASCEND_IS_AIC {
        this->constInfo.commonConstInfo.mm1Ka = this->constInfo.commonConstInfo.n2GDv;
        this->constInfo.commonConstInfo.mm1Kb = this->constInfo.commonConstInfo.n2Dv;
        this->constInfo.mm2Ka = this->constInfo.commonConstInfo.n2GD;
        this->constInfo.mm2Kb = this->constInfo.commonConstInfo.n2D;
        this->constInfo.mm3Ka = this->constInfo.mm2Ka;
        this->constInfo.mm4Kb = this->constInfo.mm2Kb;
    }
    this->constInfo.commonConstInfo.subBlockIdx = this->vSubBlockIdx;

    uint32_t tmp = 0xFF7FFFFF;

    uint32_t maxContinuousBlockNum =
        this->constInfo.commonConstInfo.s1Size < MIN_SWIZZLE_S1 ?
            MAX_CONTINUOUS_BLOCK_NUM :
            (this->constInfo.commonConstInfo.s1Size / MIN_SWIZZLE_S1) * BASE_SWIZZLE_BLOCK_NUM;
    this->constInfo.continuousBlockNum = tilingData->s1s2BNGS1S2SplitCoreParams.maxValidBBLen > maxContinuousBlockNum ?
                                             maxContinuousBlockNum :
                                             tilingData->s1s2BNGS1S2SplitCoreParams.maxValidBBLen;
    this->innerN2GD = this->constInfo.commonConstInfo.n2GD * 128;

    if ASCEND_IS_AIV {
        this->constInfo.pScale = pScaleGm.GetValue(0);
        this->constInfo.dsScale = dsScaleGm.GetValue(0);
        this->constInfo.pScaleD = (float)1.0 / this->constInfo.pScale;
        this->constInfo.dsScaleD = (float)1.0 / this->constInfo.dsScale;
        this->constInfo.copyOutDStride =
            (this->constInfo.commonConstInfo.n2GD - FagBaseApi::quant::QUANT_CUBE_BASE) * sizeof(CALC_TYPE);
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradQuantContext<CubeBlockType, VecBlockType>::InitCVCommonGlobalBuffer(
    GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR deqScaleQ, GM_ADDR deqScaleK, GM_ADDR deqScaleV, GM_ADDR deqScaleDy,
    GM_ADDR dsScale, GM_ADDR pScale, GM_ADDR workspace)
{
    dqGm.SetGlobalBuffer((__gm__ OUTDTYPE*)dq);
    dkGm.SetGlobalBuffer((__gm__ OUTDTYPE*)dk);
    dvGm.SetGlobalBuffer((__gm__ OUTDTYPE*)dv);
    deqScaleQGm.SetGlobalBuffer((__gm__ float*)deqScaleQ);
    deqScaleKGm.SetGlobalBuffer((__gm__ float*)deqScaleK);
    deqScaleVGm.SetGlobalBuffer((__gm__ float*)deqScaleV);
    deqScaleDyGm.SetGlobalBuffer((__gm__ float*)deqScaleDy);
    dsScaleGm.SetGlobalBuffer((__gm__ float*)dsScale);
    pScaleGm.SetGlobalBuffer((__gm__ float*)pScale);

    dqWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace +
                                  tilingData->postTilingData.dqWorkSpaceOffset / sizeof(float));
    dkWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace +
                                  tilingData->postTilingData.dkWorkSpaceOffset / sizeof(float));
    dvWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace +
                                  tilingData->postTilingData.dvWorkSpaceOffset / sizeof(float));
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t FlashAttentionScoreGradQuantContext<CubeBlockType, VecBlockType>::CalDenseDeterIndex(
    uint32_t roundId, uint32_t maxLoopNum, CoordinateInfo& coordinateInfo)
{
    int64_t j = this->cBlockIdx + 1;
    int64_t r = roundId + 1;

    // coreNum 是 AIV 数。arch35 下 cube 数 = AIV 数 / 2，arch92 下两者相等。
    int64_t k =
        static_cast<int64_t>(this->tilingData->s1s2BNGS1S2BaseParams.coreNum / FagBaseApi::quant::QUANT_CV_CORE_RATIO);
    if (j > k) {
        return -1;
    }
    int64_t b = this->constInfo.bSize * this->constInfo.n2Size;
    if (this->constInfo.s2Outer == 1) {
        CalDenseIndexForSingleN(k, this->constInfo.s1Outer, b, j, r, maxLoopNum, coordinateInfo);
    } else {
        CalDenseIndex(k, this->constInfo.s1Outer, this->constInfo.s2Outer, b, j, r, coordinateInfo);
    }

    int64_t w = coordinateInfo.batchId;
    int64_t n1 = this->constInfo.commonConstInfo.gSize * this->constInfo.n2Size;
    coordinateInfo.batchId = Ceil<int64_t>(w, n1) - 1;
    int64_t n1Idx = w - coordinateInfo.batchId * n1 - 1;
    coordinateInfo.n2Idx = n1Idx / this->constInfo.commonConstInfo.gSize;
    coordinateInfo.gIdx = n1Idx % this->constInfo.commonConstInfo.gSize;
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
__aicore__ inline int64_t FlashAttentionScoreGradQuantContext<CubeBlockType, VecBlockType>::CalDeterMaxLoopNum()
{
    int64_t b = this->constInfo.bSize * this->constInfo.n2Size;
    int64_t m = this->constInfo.s1Outer;
    int64_t n = this->constInfo.s2Outer;
    // 同 CalDenseDeterIndex：arch92 的 cube 数等于 AIV 数，不再除 2。
    int64_t k =
        static_cast<int64_t>(this->tilingData->s1s2BNGS1S2BaseParams.coreNum / FagBaseApi::quant::QUANT_CV_CORE_RATIO);

    InitCoordinateInfo(this->constInfo.s1Outer, this->constInfo.s2Outer, 0, 0, this->coordinateInfos[0]);
    InitCoordinateInfo(this->constInfo.s1Outer, this->constInfo.s2Outer, 0, 0, this->coordinateInfos[1]);

    if (n == 1) {
        return Max(Ceil<int64_t>(m * b, k), m);
    } else {
        return Ceil<int64_t>(n * b, Min(k, m * b)) * m;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradQuantContext<CubeBlockType, VecBlockType>::CalDeterIndex(
    uint32_t roundId, uint32_t maxLoopNum, int64_t& nextValidRoundId, int64_t& nextValidIndex, int64_t taskId,
    CoordinateInfo& coordinateInfo, FagRunInfo& runInfo)
{
    coordinateInfo.sparseMode = this->constInfo.sparseMode;
    for (uint32_t currentRoundId = roundId; currentRoundId < maxLoopNum; currentRoundId++) {
        nextValidIndex = CalDenseDeterIndex(currentRoundId, maxLoopNum, coordinateInfo);

        bool isValidBlock = (nextValidIndex >= 0);
        if (isValidBlock) {
            nextValidRoundId = currentRoundId;
            return;
        }
    }
    coordinateInfo.batchId = -1;
    nextValidIndex = -1;
    nextValidRoundId = maxLoopNum;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t FlashAttentionScoreGradQuantContext<CubeBlockType, VecBlockType>::GetDeqScaleQOffset(
    FagRunInfo& runInfo)
{
    int64_t scaleOffset = 0;
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t gOffset = 0;
    int64_t s1Offset = 0;
    int64_t scaleNumPerS1 = Ceil<int64_t>(this->constInfo.commonConstInfo.s1Size, CUBE_BASEM);
    bOffset = runInfo.commonRunInfo.boIdx * this->constInfo.commonConstInfo.n2G * scaleNumPerS1;
    n2Offset = runInfo.commonRunInfo.n2oIdx * this->constInfo.commonConstInfo.gSize * scaleNumPerS1;
    gOffset = runInfo.commonRunInfo.goIdx * scaleNumPerS1;
    s1Offset = runInfo.commonRunInfo.s1oIdx;
    scaleOffset = bOffset + n2Offset + gOffset + s1Offset;
    return scaleOffset;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t FlashAttentionScoreGradQuantContext<CubeBlockType, VecBlockType>::GetDeqScaleKOffset(
    FagRunInfo& runInfo)
{
    int64_t scaleOffset = 0;
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s2Offset = 0;
    int64_t scaleNumPerS2 = Ceil<int64_t>(this->constInfo.commonConstInfo.s2Size, CUBE_BASEN);
    bOffset = runInfo.commonRunInfo.boIdx * this->constInfo.n2Size * scaleNumPerS2;
    n2Offset = runInfo.commonRunInfo.n2oIdx * scaleNumPerS2;
    s2Offset = runInfo.s2oIdx;
    scaleOffset = bOffset + n2Offset + s2Offset;
    return scaleOffset;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradQuantContext<CubeBlockType, VecBlockType>::SetRunInfo(
    FagRunInfo& runInfo, FagRunInfo& lastRunInfo, int64_t taskId, CoordinateInfo& coordinateInfo,
    CoordinateInfo& nextCoordinateInfo)
{
    runInfo.isKeyReuse =
        ((nextCoordinateInfo.batchId == coordinateInfo.batchId) && (nextCoordinateInfo.n2Idx == coordinateInfo.n2Idx) &&
         (nextCoordinateInfo.s2Idx == coordinateInfo.s2Idx)) ||
        (nextCoordinateInfo.batchId == -1);
    runInfo.isLastProcessBlock = (nextCoordinateInfo.batchId == -1);
    runInfo.isFirstProcessBlock = taskId == 0;
    lastRunInfo.isNextKeyReuse = runInfo.isKeyReuse;
    runInfo.isValueReuse = (lastS2Idx == coordinateInfo.s2Idx && lastBatchIdx == coordinateInfo.batchId &&
                            lastN2Idx == coordinateInfo.n2Idx);
    lastBatchIdx = coordinateInfo.batchId;
    lastN2Idx = coordinateInfo.n2Idx;
    lastS2Idx = coordinateInfo.s2Idx;

    runInfo.commonRunInfo.boIdx = coordinateInfo.batchId;
    runInfo.commonRunInfo.n2oIdx = coordinateInfo.n2Idx;
    runInfo.commonRunInfo.goIdx = coordinateInfo.gIdx;
    runInfo.s2oIdx = coordinateInfo.s2Idx;
    runInfo.commonRunInfo.s1oIdx = coordinateInfo.s1Idx;
    runInfo.s2CvBegin = runInfo.s2oIdx * CUBE_BASEN;

    runInfo.commonRunInfo.taskId = taskId;
    runInfo.commonRunInfo.taskIdMod2 = taskId & 1;

    runInfo.commonRunInfo.s1RealSize =
        (runInfo.commonRunInfo.s1oIdx == this->constInfo.s1Outer - 1) ? this->constInfo.s1CvTail : CUBE_BASEM;
    runInfo.commonRunInfo.s2RealSize =
        (runInfo.s2oIdx == this->constInfo.s2Outer - 1) ? this->constInfo.s2Tail : CUBE_BASEN;

    runInfo.quantRunInfo.innerS1LoopNum = Ceil<int64_t>(runInfo.commonRunInfo.s1RealSize, 128);
    runInfo.quantRunInfo.innerS2LoopNum = Ceil<int64_t>(runInfo.commonRunInfo.s2RealSize, 128);

    if ASCEND_IS_AIV {
        int64_t deqScaleQGmOffset = GetDeqScaleQOffset(runInfo);
        int64_t deqScaleKGmOffset = GetDeqScaleKOffset(runInfo);
        runInfo.quantScaleInfo.deqScaleQValue = deqScaleQGm.GetValue(deqScaleQGmOffset);
        runInfo.quantScaleInfo.deqScaleKValue = deqScaleKGm.GetValue(deqScaleKGmOffset);
        runInfo.quantScaleInfo.deqScaleVValue = deqScaleVGm.GetValue(deqScaleKGmOffset);
        runInfo.quantScaleInfo.deqScaleDyValue = deqScaleDyGm.GetValue(deqScaleQGmOffset);

        runInfo.quantRunInfo.qkDScale = runInfo.quantScaleInfo.deqScaleQValue * runInfo.quantScaleInfo.deqScaleKValue;
        runInfo.quantRunInfo.vdyDScale = runInfo.quantScaleInfo.deqScaleDyValue * runInfo.quantScaleInfo.deqScaleVValue;

        runInfo.quantRunInfo.dsScaleDMulDeqScaleK = this->constInfo.dsScaleD * runInfo.quantScaleInfo.deqScaleKValue;
        runInfo.quantRunInfo.pScaleDMulDeqScaleDy = this->constInfo.pScaleD * runInfo.quantScaleInfo.deqScaleDyValue;
        runInfo.quantRunInfo.dsScaleDMulDeqScaleQ = this->constInfo.dsScaleD * runInfo.quantScaleInfo.deqScaleQValue;

        runInfo.maxsumOffset = ((runInfo.commonRunInfo.boIdx * this->constInfo.n2Size + runInfo.commonRunInfo.n2oIdx) *
                                    this->constInfo.commonConstInfo.gSize +
                                runInfo.commonRunInfo.goIdx) *
                                   this->constInfo.commonConstInfo.s1Size +
                               runInfo.commonRunInfo.s1oIdx * CUBE_BASEM;
        runInfo.quantRunInfo.kvNeedAtomic =
            (this->constInfo.s2Outer != 1 && runInfo.isValueReuse) ||
            (this->constInfo.s2Outer == 1 &&
             ((!runInfo.isFirstProcessBlock && (((coordinateInfo.batchId == nextCoreFirstBlockCoordinateInfo.batchId) &&
                                                 (coordinateInfo.n2Idx == nextCoreFirstBlockCoordinateInfo.n2Idx)) ||
                                                runInfo.isValueReuse))));
    }

    runInfo.commonRunInfo.queryOffset = this->GetQueryOffset(runInfo);
    runInfo.dyOffset = runInfo.commonRunInfo.queryOffset;
    runInfo.commonRunInfo.keyOffset = this->GetKeyOffset(runInfo);
    runInfo.commonRunInfo.valueOffset = runInfo.commonRunInfo.keyOffset;

    int64_t innerS1TailSize =
        (runInfo.commonRunInfo.s1RealSize % 128) == 0 ? 128 : runInfo.commonRunInfo.s1RealSize % 128;
    int64_t innerS2TailSize =
        (runInfo.commonRunInfo.s2RealSize % 128) == 0 ? 128 : runInfo.commonRunInfo.s2RealSize % 128;
    for (int64_t i = 0; i < 4; i++) {
        runInfo.quantRunInfo.innerS1RealSize[i] =
            (i == runInfo.quantRunInfo.innerS1LoopNum - 1) ? innerS1TailSize : 128;
        runInfo.quantRunInfo.innerS2RealSize[i] =
            (i == runInfo.quantRunInfo.innerS2LoopNum - 1) ? innerS2TailSize : 128;

        runInfo.quantRunInfo.qInnerOffset[i] = runInfo.commonRunInfo.queryOffset + i * this->innerN2GD;
        runInfo.quantRunInfo.kvInnerOffset[i] = runInfo.commonRunInfo.keyOffset + i * this->innerN2GD;
    }
}

} // namespace quant
} // namespace FagBaseApi

#endif
