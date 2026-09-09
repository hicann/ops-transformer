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
 * \file fused_floyd_attention_s1s2_bn2gs1.h
 * \brief
 */

#ifndef FUSED_FLOYD_ATTENTION_S1S2_BN2GS1_SPECIAL_H
#define FUSED_FLOYD_ATTENTION_S1S2_BN2GS1_SPECIAL_H

#include "util.h"
#include "dropmask.h"
#include "fused_floyd_attention_common.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"

using matmul::MatmulType;

struct SplitSpecialExtraInfo {
    int64_t s2LoopCount;
    int64_t s1oIdx;
    int64_t boIdx;
    int64_t n2oIdx;
    int64_t taskId;
    int8_t taskIdMod2;
    int8_t multiCoreInnerIdxMod2;
    bool lastNotPair;
    int32_t s1RealSize;
    int32_t s2RealSize;
    int32_t n2RealSize;
    int32_t s2AlignedSize;
    int32_t vec1S1BaseSize;
    int32_t vec1S1RealSize;
    int32_t vec2S1BaseSize;
    int32_t vec2S1RealSize;
    uint32_t s1RealFp32;
    int32_t realSplitN;
    int32_t s2LoopLimit;
    int64_t multiCoreInnerIdx;
    int64_t qCoreOffset;
    int64_t attenB1SSOffset;
    int64_t attenMaskS2Size;
    int64_t softmaxMaxOffset;
    int64_t k1CoreOffset;
    int64_t vCoreOffset;
};

__aicore__ inline void BoolCopyInBrcd(LocalTensor<uint8_t> &dstTensor, GlobalTensor<uint8_t> &srcTensor,
                                      int64_t srcOffset, uint32_t s1Size, uint32_t s2Size, int64_t totalS2Size)
{
    uint32_t alignedS2Size = CeilDiv(s2Size, blockBytes) * blockBytes;

    DataCopy(dstTensor, srcTensor[srcOffset], alignedS2Size);
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

    uint32_t logStep = 0;
    for (uint32_t tmp = 1; tmp < s1Size; tmp <<= 1)
        ++logStep;

    /* 3. 逐级“自身翻倍”广播 */
    uint32_t bcount = alignedS2Size;
    for (uint32_t step = 0; step < logStep; ++step) {
        AscendC::PipeBarrier<PIPE_V>();
        DataCopy(dstTensor[bcount], dstTensor, bcount);
        bcount <<= 1; // 行数翻倍
    }
}

// INPUT_T - means data type for input
// T       - means data type when calc
template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T = INPUT_T, bool isBasicBlock = false, CubeFormat bmm1Format = CubeFormat::ND,
          bool enableL1Reuse = false>
class FusedFloydAttentionS1s2Bn2gs1Special {
public:
    __aicore__ inline FusedFloydAttentionS1s2Bn2gs1Special(){};

    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *key1,
                                __gm__ uint8_t *value1, __gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxMax,
                                __gm__ uint8_t *softmaxSum, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                                const FusedFloydAttentionGeneralTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();

    // define matmul
    using a1Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, false, LayoutMode::NORMAL>;
    using b1Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, true, LayoutMode::NORMAL>;
    using c1Type = MatmulType<TPosition::GM, CubeFormat::ND, T, false, LayoutMode::NORMAL>;
    using bias1Type = MatmulType<TPosition::GM, CubeFormat::ND, float>;

    constexpr static MatmulConfig MM_CFG = GetNormalConfig(false, false, false, BatchMode::BATCH_LESS_THAN_L1);
    constexpr static MatmulConfig MM_CFG_LL1 = GetNormalConfig(false, false, false, BatchMode::BATCH_LARGE_THAN_L1);

    matmul::Matmul<a1Type, b1Type, c1Type, bias1Type, MM_CFG> bmm1;
    matmul::Matmul<a1Type, a1Type, c1Type, bias1Type, MM_CFG_LL1> bmm2;

    using a3Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, false, LayoutMode::BSNGD>;
    using b3Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, true, LayoutMode::BSNGD>;
    using c3Type = MatmulType<TPosition::GM, CubeFormat::ND, T, false, LayoutMode::BSNGD>;

    constexpr static MatmulConfig MM_CFG2 = GetNormalConfig(true);
    matmul::Matmul<a3Type, b3Type, c3Type, bias1Type, MM_CFG2> bmm1k1;
    matmul::Matmul<a3Type, a3Type, c3Type, bias1Type, MM_CFG2> bmm2v2;

protected:
    __aicore__ inline void InitInput(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                     __gm__ uint8_t *key1, __gm__ uint8_t *value1, __gm__ uint8_t *attenMask,
                                     __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
                                     __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                                     const FusedFloydAttentionGeneralTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void IterateBmm2(SplitSpecialExtraInfo &extraInfo);
    __aicore__ inline void IterateBmm2forValue1(SplitSpecialExtraInfo &extraInfo);
    __aicore__ inline void SetExtraInfo(SplitSpecialExtraInfo &extraInfo, int64_t taskId, int64_t s2LoopCount,
                                        int64_t s2LoopLimit, int64_t multiCoreInnerIdx, bool lastNotPair);
    __aicore__ inline void SetTiling(const FusedFloydAttentionGeneralTilingData *__restrict tilingData);
    __aicore__ inline void InitBuffer();
    __aicore__ inline void ComputeConstexpr();
    __aicore__ inline void ComputeAxisIdx(int64_t multiCoreInnerIdx);
    template <typename T2, const MatmulConfig &MM_CFG>
    __aicore__ inline void IterateBmm1k1(SplitSpecialExtraInfo &extraInfo,
                                         matmul::Matmul<a3Type, b3Type, T2, bias1Type, MM_CFG> &bmm1k1);
    template <typename T2, const MatmulConfig &MM_CFG>
    __aicore__ inline void IterateBmm1(SplitSpecialExtraInfo &extraInfo,
                                       matmul::Matmul<a1Type, b1Type, T2, bias1Type, MM_CFG> &bmm1);
    __aicore__ inline void BmmSetOffset(SplitSpecialExtraInfo &extraInfo);
    template <typename T2, const MatmulConfig &MM_CFG>
    __aicore__ inline void SetBmm1TensorB(SplitSpecialExtraInfo &extraInfo,
                                          matmul::Matmul<a1Type, b1Type, T2, bias1Type, MM_CFG> &bmm1, int64_t loopN2);
    __aicore__ inline void ComputeBmm1Tail(SplitSpecialExtraInfo &extraInfo);
    __aicore__ inline void ProcessVec1(SplitSpecialExtraInfo &extraInfo);
    __aicore__ inline void CopyInAttenMask(SplitSpecialExtraInfo &extraInfo, int64_t loopIdx, int64_t maskOffset,
                                           int64_t loopN2, bool secondTime = false);
    __aicore__ inline int64_t ComputeOffsetForNoCompress(SplitSpecialExtraInfo &extraInfo, int64_t loopIdx,
                                                         int64_t loopN2);
    __aicore__ inline void GetBmm1Result(SplitSpecialExtraInfo &extraInfo, LocalTensor<T> &bmm1ResUb, int64_t loopIdx,
                                         int64_t loopN2);
    __aicore__ inline void ComputeAttenMask(SelectWithBytesMaskShapeInfo &shapeInfo, LocalTensor<T> &bmm1ResUb,
                                            LocalTensor<uint8_t> &maskUb, const uint8_t maskType, event_t vWaitMte2);

    __aicore__ inline void SoftMaxCompute(SplitSpecialExtraInfo &extraInfo, LocalTensor<T> &srcTensor, int64_t loopIdx,
                                          int64_t loopN2);
    __aicore__ inline void ProcessVec2(SplitSpecialExtraInfo &extraInfo);
    __aicore__ inline void Bmm2ResultMul(SplitSpecialExtraInfo &extraInfo, LocalTensor<T> &bmm2ResUb, int64_t s1oIdx);
    __aicore__ inline void Bmm2ResultDiv(SplitSpecialExtraInfo &extraInfo, int64_t s1oIdx);
    __aicore__ inline void Bmm2DataCopyOut(SplitSpecialExtraInfo &extraInfo, int64_t s1oIdx, int64_t mm2ResCalcSize,
                                           int64_t loopN2);
    __aicore__ inline void SoftmaxDataCopyOut(SplitSpecialExtraInfo &extraInfo, int64_t loopN2);
    __aicore__ inline void Stage2BufCopyOut(SplitSpecialExtraInfo &extraInfo, int64_t loopN2);

    __aicore__ inline void CopyInSoftmaxSumUb(SplitSpecialExtraInfo &extraInfo, int64_t loopN2);
    __aicore__ inline void CopyInSoftmaxMaxUb(SplitSpecialExtraInfo &extraInfo, int64_t loopN2);
    __aicore__ inline void CopyInSoftmaxExpUb(SplitSpecialExtraInfo &extraInfo, int64_t loopN2);
    __aicore__ inline void CopyInStage2BufUb(SplitSpecialExtraInfo &extraInfo, int64_t loopN2);

    uint32_t s1BaseSize;
    uint32_t s2BaseSize;
    uint32_t n2BaseSize = 16;
    uint32_t bmm1BatchNum;
    uint32_t bmmAddLoops;
    uint32_t attenMaskShapeType = 3;

    uint32_t dSize;
    int64_t dSizeAlign16;
    int64_t s1Size;
    int64_t s2Size;
    int64_t hSize;
    int64_t s1OuterSize;

    // L1Reuse场景vector核是否是奇数核
    int64_t l1ReuseBlockMod2 = 0;

    // 资源分配
    TBuf<> maskTBufPing;
    TBuf<> pseTBuf;
    TBuf<> stage1PingBuf;
    TBuf<> stage1PongBuf;
    TBuf<> stage2TBuf;
    TBuf<> softmaxSumBuf[2];
    TBuf<> softmaxExpBuf[2];
    TBuf<> softmaxMaxBuf;
    TBuf<> commonTBuf; // common的复用空间

    LocalTensor<T> softmaxExpUb;
    GlobalTensor<T> mm1Res[2];
    GlobalTensor<T> mm2Res[2];
    GlobalTensor<T> vec2Res[2];
    GlobalTensor<INPUT_T> stage1Res[2];
    GlobalTensor<T> softMaxExpGM[2];
    GlobalTensor<T> stage2TBufGM;

    // 轴的乘积
    int64_t gS1o;
    int64_t n2GS1o;
    int64_t s1D;
    int64_t n2GS1D;
    int64_t s2D;
    int64_t n2S2D;
    int64_t s1S2;
    int64_t s1S2D;
    int64_t n2S1;
    int64_t n2S2;
    int64_t n2G;

    // s2base*N之后的长度
    int64_t s2BaseNratioSize;

    int64_t s1S2NratioD;
    int64_t s1BaseD;
    int64_t s1BaseA16D;
    int64_t s2BaseNratioD;

    int64_t mm1Ka;
    int64_t mm1Kb;
    int64_t mm2Kb;
    uint32_t negativeIntScalar = NEGATIVE_MIN_VAULE_FP32;
    T negativeFloatScalar;
    T positiveFloatScalar;

    AttenMaskComputeMode attenMaskComputeMode = AttenMaskComputeMode::NORMAL_MODE;

    int32_t blockIdx;
    const FusedFloydAttentionGeneralTilingData *__restrict tilingData;

    int64_t boIdx;
    int64_t n2oIdx;
    int64_t s1oIdx;
    int64_t attenBoIdx;

    TPipe *pipe;

    GlobalTensor<INPUT_T> queryGm;
    GlobalTensor<INPUT_T> keyGm;
    GlobalTensor<INPUT_T> keyGm1;
    GlobalTensor<INPUT_T> valueGm;
    GlobalTensor<INPUT_T> valueGm1;
    GlobalTensor<INPUT_T> attentionOutGm;
    GlobalTensor<float> softmaxMaxGm;
    GlobalTensor<float> softmaxSumGm;
    GlobalTensor<uint8_t> attenMaskGmInt;
};

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void FusedFloydAttentionS1s2Bn2gs1Special<
    implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock, bmm1Format,
    enableL1Reuse>::Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *key1,
                         __gm__ uint8_t *value1, __gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxMax,
                         __gm__ uint8_t *softmaxSum, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                         const FusedFloydAttentionGeneralTilingData *__restrict tiling, TPipe *tPipe)
{
    this->InitInput(query, key, value, key1, value1, attenMask, softmaxMax, softmaxSum, attentionOut, workspace, tiling,
                    tPipe); // gm设置

    this->ComputeConstexpr();
    this->InitBuffer();
    LocalTensor<T> apiTmpBuffer = this->commonTBuf.template Get<T>();
    DropOutBitModeInit(apiTmpBuffer);
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void FusedFloydAttentionS1s2Bn2gs1Special<
    implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock, bmm1Format,
    enableL1Reuse>::InitInput(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *key1,
                              __gm__ uint8_t *value1, __gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxMax,
                              __gm__ uint8_t *softmaxSum, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                              const FusedFloydAttentionGeneralTilingData *__restrict tiling, TPipe *tPipe)
{
    this->blockIdx = GetBlockIdx();
    this->pipe = tPipe;
    this->SetTiling(tiling);

    // init global buffer
    this->queryGm.SetGlobalBuffer((__gm__ INPUT_T *)query);
    this->keyGm.SetGlobalBuffer((__gm__ INPUT_T *)key);
    this->keyGm1.SetGlobalBuffer((__gm__ INPUT_T *)key1);
    this->valueGm.SetGlobalBuffer((__gm__ INPUT_T *)value);
    this->valueGm1.SetGlobalBuffer((__gm__ INPUT_T *)value1);
    this->attenMaskGmInt.SetGlobalBuffer((__gm__ uint8_t *)attenMask);
    this->softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmaxMax);
    this->softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmaxSum);
    this->attentionOutGm.SetGlobalBuffer((__gm__ INPUT_T *)attentionOut);

    // 补齐到512， 统一按T处理
    int64_t mm1ResultSize = n2BaseSize * s1BaseSize * s2BaseSize;
    int64_t mmNRatioOffset = CeilDiv(mm1ResultSize * this->tilingData->coreParams.nRatio, 128) * 128 * sizeof(T);
    int64_t mm2ResultSize = n2BaseSize * s1BaseSize * dSizeAlign16;
    int64_t mm2Offset = CeilDiv(mm2ResultSize, 128) * 128 * 4;
    int64_t bmm1AndVec1Ratio = GM_DOUBLE_BUFFER;
    int64_t vector1OffsetPing = 0;
    int64_t vector1OffsetPong = mmNRatioOffset;

    // FP32场景，stage1Result不与bmm1Result共用空间，需要占用2倍mmNRatioOffset空间
    if constexpr (IsSameType<INPUT_T, float>::value) {
        vector1OffsetPing = mmNRatioOffset * GM_DOUBLE_BUFFER;
        vector1OffsetPong = vector1OffsetPing + mmNRatioOffset;
        bmm1AndVec1Ratio = GM_DOUBLE_BUFFER + 2;
    }

    int64_t softmaxExpSize = n2BaseSize * s1BaseSize * blockBytes;
    int64_t stage2bufSize = n2BaseSize * s1BaseSize * dSizeAlign16 * 4;
    int64_t totalOffset =
        mmNRatioOffset * bmm1AndVec1Ratio + mm2Offset * GM_DOUBLE_BUFFER + softmaxExpSize * 2 + stage2bufSize;
    if (dSizeAlign16 > 64) {
        totalOffset = mmNRatioOffset * bmm1AndVec1Ratio + mm2Offset * 2 * GM_DOUBLE_BUFFER;
    }

    // bmm1Result，占用2倍mmNRatioOffset空间
    this->mm1Res[0].SetGlobalBuffer((__gm__ T *)(workspace + this->blockIdx * totalOffset));
    this->mm1Res[1].SetGlobalBuffer((__gm__ T *)(workspace + this->blockIdx * totalOffset + mmNRatioOffset));

    // stage1Result，不占用/占用1倍/占用2倍mmNRatioOffset空间
    this->stage1Res[0].SetGlobalBuffer(
        (__gm__ INPUT_T *)(workspace + this->blockIdx * totalOffset + vector1OffsetPing));
    this->stage1Res[1].SetGlobalBuffer(
        (__gm__ INPUT_T *)(workspace + this->blockIdx * totalOffset + vector1OffsetPong));

    int64_t stage2BaseAddr = this->blockIdx * totalOffset + mmNRatioOffset * bmm1AndVec1Ratio;
    // bmm2Result，占用2倍mmOffset空间
    this->mm2Res[0].SetGlobalBuffer((__gm__ T *)(workspace + stage2BaseAddr));
    this->mm2Res[1].SetGlobalBuffer((__gm__ T *)(workspace + stage2BaseAddr + mm2Offset));

    this->softMaxExpGM[0].SetGlobalBuffer((__gm__ T *)(workspace + stage2BaseAddr + mm2Offset * 2));
    this->softMaxExpGM[1].SetGlobalBuffer((__gm__ T *)(workspace + stage2BaseAddr + mm2Offset * 2 + softmaxExpSize));

    this->stage2TBufGM.SetGlobalBuffer((__gm__ T *)(workspace + stage2BaseAddr + mm2Offset * 2 + softmaxExpSize * 2));

    if constexpr (IsSameType<T, half>::value) {
        this->negativeIntScalar = NEGATIVE_MIN_VAULE_FP16;
    }
    GetExtremeValue(this->negativeFloatScalar, this->positiveFloatScalar);
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void FusedFloydAttentionS1s2Bn2gs1Special<
    implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock, bmm1Format,
    enableL1Reuse>::SetTiling(const FusedFloydAttentionGeneralTilingData *__restrict tilingData)
{
    // copy base params
    this->tilingData = tilingData;
    this->s1BaseSize = this->tilingData->coreParams.s1BaseSize;
    this->s2BaseSize = this->tilingData->coreParams.s2BaseSize;
    this->n2BaseSize = this->tilingData->coreParams.n2BaseSize;
    this->bmm1BatchNum = this->tilingData->coreParams.bmm1Num;
    this->bmmAddLoops = this->tilingData->coreParams.bmmv2Num;

    this->dSize = this->tilingData->inputParams.dSize;
    this->dSizeAlign16 = CeilDiv(this->tilingData->inputParams.dSize, 16) * 16;
};

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void FusedFloydAttentionS1s2Bn2gs1Special<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                            isBasicBlock, bmm1Format, enableL1Reuse>::InitBuffer()
{
    uint64_t stage1Size = 8 * 1024;
    uint64_t stage1AttenSize = 9 * 1024;
    uint64_t stage1PongSize = 35 * 1024;
    uint64_t stage2Size = 64 * 128;
    uint64_t maskTBufPongSize = 16 * 1024;

    // 可选输入的buffer空间，保持和stage1处理的size一致
    this->pipe->InitBuffer(this->maskTBufPing, stage1AttenSize); // 可以给attenmask 9k -> 16k
    this->pipe->InitBuffer(this->pseTBuf, 16384);                // pse 16k

    this->pipe->InitBuffer(this->stage1PingBuf, stage2Size * sizeof(T)); // t.a 32k
    this->pipe->InitBuffer(this->stage2TBuf, stage2Size * sizeof(T));    // t.c 32k
    this->pipe->InitBuffer(this->commonTBuf, stage2Size * sizeof(T));    // t.b 32k

    this->pipe->InitBuffer(this->softmaxSumBuf[0], s1BaseSize * blockBytes); // 4k
    this->pipe->InitBuffer(this->softmaxSumBuf[1], s1BaseSize * blockBytes); // 4k
    this->pipe->InitBuffer(this->softmaxMaxBuf, s1BaseSize * blockBytes);    // 4k
    this->pipe->InitBuffer(this->softmaxExpBuf[0], s1BaseSize * blockBytes); // 4k
    this->pipe->InitBuffer(this->softmaxExpBuf[1], s1BaseSize * blockBytes); // 4k

    this->pipe->InitBuffer(this->stage1PongBuf, stage1Size * sizeof(T)); // i.a 32k
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void FusedFloydAttentionS1s2Bn2gs1Special<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                            isBasicBlock, bmm1Format, enableL1Reuse>::ComputeConstexpr()
{
    // 计算轴的乘积
    if constexpr (enableL1Reuse) {
        this->s1OuterSize = this->tilingData->coreParams.s1OuterSize;
    }
    this->s1Size = this->tilingData->inputParams.s1Size;
    this->s2Size = this->tilingData->inputParams.s2Size;

    this->s1D = this->tilingData->inputParams.s1Size * dSize;
    this->s2D = this->tilingData->inputParams.s2Size * dSize;
    this->n2S1 = this->tilingData->inputParams.n2Size * this->tilingData->inputParams.s1Size;
    this->n2S2 = this->tilingData->inputParams.n2Size * this->tilingData->inputParams.s2Size;
    this->s1S2 = this->tilingData->inputParams.s1Size * this->tilingData->inputParams.s2Size;
    this->s1S2D = this->s1S2 * dSize;
    this->n2G = this->tilingData->inputParams.n2Size * this->tilingData->inputParams.gSize;
    this->gS1o = this->tilingData->inputParams.gSize * this->tilingData->coreParams.s1OuterSize;

    this->n2GS1o = this->tilingData->coreParams.n2OuterSize * this->gS1o;
    this->n2S2D = this->tilingData->inputParams.n2Size * this->s2D;

    this->n2GS1D = this->tilingData->inputParams.n2Size * this->s1D;

    // 计算切分轴的乘积
    this->s2BaseNratioSize = this->s2BaseSize * this->tilingData->coreParams.nRatio;
    this->s1S2NratioD = this->s2BaseNratioSize * this->s1D;
    this->s1BaseD = this->s1BaseSize * this->dSize;
    this->s2BaseNratioD = this->s2BaseNratioSize * this->dSize;
    this->s1BaseA16D = this->s1BaseSize * dSizeAlign16;
    this->mm1Ka = this->dSize;
    this->mm1Kb = this->dSize;
    this->mm2Kb = this->dSize;

    this->attenMaskShapeType = this->tilingData->inputParams.attenMaskShapeType;
    this->hSize = this->tilingData->inputParams.bSize / this->tilingData->inputParams.attenbSize;
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void FusedFloydAttentionS1s2Bn2gs1Special<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                            isBasicBlock, bmm1Format, enableL1Reuse>::Process()
{
    // 1. 确定核内切分起点（保持不变）
    int64_t multiCoreInnerOffset = this->blockIdx * this->tilingData->multiCoreParams.splitFactorSize;
    if constexpr (enableL1Reuse) {
        this->l1ReuseBlockMod2 = this->blockIdx % 2;
        multiCoreInnerOffset = this->blockIdx / 2 * this->tilingData->multiCoreParams.splitFactorSize;
    }
    int64_t multiCoreInnerLimit = multiCoreInnerOffset + this->tilingData->multiCoreParams.splitFactorSize;
    if (this->tilingData->multiCoreParams.totalSize < multiCoreInnerLimit) {
        multiCoreInnerLimit = this->tilingData->multiCoreParams.totalSize;
    }

    SplitSpecialExtraInfo currentInfo;
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));

    // 2. 遍历实际的数据范围（不再需要额外的 +2 循环）
    for (int64_t multiCoreInnerIdx = multiCoreInnerOffset; multiCoreInnerIdx < multiCoreInnerLimit;
         multiCoreInnerIdx++) {
        this->ComputeAxisIdx(multiCoreInnerIdx);
        int64_t s2LoopLimit = CeilDiv(this->s2Size, s2BaseNratioSize) - 1;

        for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; s2LoopCount++) {
            // --- 阶段 1: Bmm1 计算 ---
            // 设置当前任务信息

            this->SetExtraInfo(currentInfo, 0, s2LoopCount, s2LoopLimit, multiCoreInnerIdx, false);
            this->BmmSetOffset(currentInfo);

            this->IterateBmm1k1(currentInfo, this->bmm1k1);
            this->IterateBmm1(currentInfo, this->bmm1);

            // --- 阶段 2: Vector 处理 1 ---
            // 单发射模式下，直接处理当前索引的数据
            this->ProcessVec1(currentInfo);

            // 这里同步确保 MTE 搬运或计算完成，视硬件需求而定
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);

            // --- 阶段 3: Bmm2 计算 ---
            this->IterateBmm2(currentInfo);
            this->IterateBmm2forValue1(currentInfo);

            // --- 阶段 4: Vector 处理 2 ---
            this->ProcessVec2(currentInfo);

            // 每一个 s2LoopCount 完成，逻辑闭环
        }
    }
};

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1Special<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock,
                                     bmm1Format, enableL1Reuse>::ComputeAxisIdx(int64_t multiCoreInnerIdx)
{
    // 计算轴的idx
    this->boIdx = multiCoreInnerIdx / this->n2GS1o;
    this->n2oIdx = multiCoreInnerIdx % this->n2GS1o / this->gS1o;
    this->s1oIdx = multiCoreInnerIdx % this->tilingData->coreParams.s1OuterSize;
    this->attenBoIdx = this->boIdx / hSize;
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1Special<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock,
                                     bmm1Format, enableL1Reuse>::SetExtraInfo(SplitSpecialExtraInfo &extraInfo,
                                                                              int64_t taskId, int64_t s2LoopCount,
                                                                              int64_t s2LoopLimit,
                                                                              int64_t multiCoreInnerIdx,
                                                                              bool lastNotPair)
{
    extraInfo.s2LoopCount = s2LoopCount;
    extraInfo.s1oIdx = this->s1oIdx;
    extraInfo.boIdx = this->boIdx;
    extraInfo.n2oIdx = this->n2oIdx;
    extraInfo.taskId = taskId;
    extraInfo.taskIdMod2 = taskId % 2;
    extraInfo.s2LoopLimit = s2LoopLimit;
    extraInfo.multiCoreInnerIdx = multiCoreInnerIdx;
    extraInfo.multiCoreInnerIdxMod2 = multiCoreInnerIdx % 2;
    extraInfo.attenB1SSOffset = extraInfo.boIdx * this->s1S2;
    extraInfo.attenMaskS2Size = this->tilingData->inputParams.attenMaskS2Size;
    if constexpr (enableL1Reuse) {
        extraInfo.lastNotPair = lastNotPair;
    }

    extraInfo.s1RealSize = Min(s1BaseSize, this->s1Size - extraInfo.s1oIdx * s1BaseSize);
    extraInfo.n2RealSize = Min(n2BaseSize, this->tilingData->inputParams.n2Size - this->n2oIdx * n2BaseSize);
    extraInfo.s1RealFp32 = extraInfo.s1RealSize * fp32BaseSize;

    this->ComputeBmm1Tail(extraInfo);
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1Special<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock,
                                     bmm1Format, enableL1Reuse>::ComputeBmm1Tail(SplitSpecialExtraInfo &extraInfo)
{
    if (this->s1Size < (extraInfo.s1oIdx + 1) * this->s1BaseSize) {
        extraInfo.s1RealSize = this->s1Size - extraInfo.s1oIdx * this->s1BaseSize;
    }
    extraInfo.s2RealSize = this->s2BaseNratioSize;
    extraInfo.s2AlignedSize = extraInfo.s2RealSize;
    if ((extraInfo.s2LoopCount + 1) * extraInfo.s2RealSize > this->s2Size) {
        extraInfo.s2RealSize = this->s2Size - extraInfo.s2LoopCount * extraInfo.s2RealSize;
        extraInfo.s2AlignedSize = Align(extraInfo.s2RealSize);
    }

    extraInfo.vec1S1BaseSize = Min(s2BaseNratioSize / extraInfo.s2AlignedSize * 8, extraInfo.s1RealSize);
    extraInfo.realSplitN = CeilDiv(extraInfo.s1RealSize, extraInfo.vec1S1BaseSize);

    if (dSizeAlign16 > 64) {
        extraInfo.vec2S1BaseSize = 64 * 128 / dSizeAlign16;
    } else {
        extraInfo.vec2S1BaseSize = extraInfo.s1RealSize;
    }

    return;
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
template <typename T2, const MatmulConfig &MM_CFG>
__aicore__ inline void FusedFloydAttentionS1s2Bn2gs1Special<
    implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock, bmm1Format,
    enableL1Reuse>::IterateBmm1(SplitSpecialExtraInfo &extraInfo,
                                matmul::Matmul<a1Type, b1Type, T2, bias1Type, MM_CFG> &bmm1)
{
    uint32_t batchNum = this->bmm1BatchNum;
    uint32_t loops = CeilDiv(extraInfo.n2RealSize, batchNum);
    int64_t matrixStrideA = this->s1D;
    int64_t matrixStrideB = 0;
    if (extraInfo.s2LoopLimit > 0) {
        matrixStrideB = this->s2D;
    }

    for (uint32_t idx = 0; idx < loops; ++idx) {
        uint32_t batchSizes = idx * batchNum;
        int64_t batchOffsetD = batchSizes * this->s1D;
        bmm1.SetTensorA(this->queryGm[extraInfo.qCoreOffset + batchOffsetD]);
        SetBmm1TensorB(extraInfo, bmm1, batchSizes);

        int64_t outOffset = batchSizes * extraInfo.s2RealSize * extraInfo.s1RealSize;
        uint32_t realBatchNum = Min(batchNum, extraInfo.n2RealSize - batchSizes);
        bmm1.IterateBatch(this->mm1Res[extraInfo.taskIdMod2][outOffset], realBatchNum, realBatchNum, false,
                          matrixStrideA, matrixStrideB, 0, false, 1);
    }
    bmm1.End();
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
template <typename T2, const MatmulConfig &MM_CFG>
__aicore__ inline void FusedFloydAttentionS1s2Bn2gs1Special<
    implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock, bmm1Format,
    enableL1Reuse>::IterateBmm1k1(SplitSpecialExtraInfo &extraInfo,
                                  matmul::Matmul<a3Type, b3Type, T2, bias1Type, MM_CFG> &bmm1mix)
{
    uint32_t loops = (extraInfo.s1RealSize + bmmAddLoops - 1) / bmmAddLoops;
    uint32_t loop_rem = extraInfo.s1RealSize % bmmAddLoops;
    uint32_t realBatchNum = bmmAddLoops;

    for (uint32_t idx = 0; idx < loops; ++idx) { // 确定this->s1BaseSize 是否tail
        if (idx == loops - 1 && loop_rem != 0) {
            realBatchNum = loop_rem;
        }

        int64_t batchOffsetD = bmmAddLoops * idx * this->dSize;
        bmm1mix.SetTensorA(this->queryGm[extraInfo.qCoreOffset + batchOffsetD]);

        int64_t kCoreOffset = extraInfo.k1CoreOffset + batchOffsetD;
        bmm1mix.SetTensorB(this->keyGm1[kCoreOffset], true);
        bmm1mix.IterateBatch(this->mm1Res[extraInfo.taskIdMod2][bmmAddLoops * idx * extraInfo.s2RealSize], realBatchNum,
                             realBatchNum, false);
    }

    bmm1mix.End();
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1Special<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock,
                                     bmm1Format, enableL1Reuse>::BmmSetOffset(SplitSpecialExtraInfo &extraInfo)
{
    // BNSD
    int64_t bOffset = extraInfo.boIdx * this->n2GS1D;
    int64_t n2Offset = extraInfo.n2oIdx * this->s1D * this->n2BaseSize;
    int64_t s1Offset = extraInfo.s1oIdx * this->s1BaseD;
    extraInfo.qCoreOffset = bOffset + n2Offset + s1Offset;

    // BNSD
    int64_t bs1Offset = extraInfo.boIdx * this->s1S2D;
    int64_t s2Offset = extraInfo.s2LoopCount * this->s1S2NratioD;
    extraInfo.k1CoreOffset = bs1Offset + s2Offset + s1Offset;

    int64_t bOffsetV = extraInfo.boIdx * this->n2S2D;
    int64_t n2OffsetV = extraInfo.n2oIdx * this->s2D * this->n2BaseSize;
    int64_t s2OffsetV = extraInfo.s2LoopCount * s2BaseNratioD;
    extraInfo.vCoreOffset = bOffsetV + n2OffsetV + s2OffsetV;
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
template <typename T2, const MatmulConfig &MM_CFG>
__aicore__ inline void FusedFloydAttentionS1s2Bn2gs1Special<
    implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock, bmm1Format,
    enableL1Reuse>::SetBmm1TensorB(SplitSpecialExtraInfo &extraInfo,
                                   matmul::Matmul<a1Type, b1Type, T2, bias1Type, MM_CFG> &bmm1, int64_t loopN2)
{
    int64_t n2Offset = loopN2 * this->s2D;
    int64_t vCoreOffset = extraInfo.vCoreOffset + n2Offset;
    bmm1.SetTensorB(this->keyGm[vCoreOffset], true);
    bmm1.SetTail(extraInfo.s1RealSize, extraInfo.s2RealSize);
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1Special<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock,
                                     bmm1Format, enableL1Reuse>::CopyInSoftmaxSumUb(SplitSpecialExtraInfo &extraInfo,
                                                                                    int64_t loopN2)
{
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));

    // s1Size 可能没有对齐的情况
    LocalTensor<float> sumUb = softmaxSumBuf[extraInfo.multiCoreInnerIdxMod2].Get<float>();
    int64_t vec2S1N2Offset = loopN2 * s1Size * fp32BaseSize;
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    DataCopy(sumUb, this->softmaxSumGm[extraInfo.softmaxMaxOffset + vec2S1N2Offset], extraInfo.s1RealFp32);
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1Special<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock,
                                     bmm1Format, enableL1Reuse>::CopyInSoftmaxExpUb(SplitSpecialExtraInfo &extraInfo,
                                                                                    int64_t loopN2)
{
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));

    LocalTensor<T> expUb = softmaxExpBuf[extraInfo.taskIdMod2].Get<T>();
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    DataCopy(expUb, this->softMaxExpGM[extraInfo.taskIdMod2][loopN2 * s1BaseSize * fp32BaseSize], extraInfo.s1RealFp32);
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1Special<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock,
                                     bmm1Format, enableL1Reuse>::CopyInStage2BufUb(SplitSpecialExtraInfo &extraInfo,
                                                                                   int64_t loopN2)
{
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));

    LocalTensor<T> bmm2ResUb = this->stage2TBuf.template Get<T>();
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    DataCopy(bmm2ResUb, this->stage2TBufGM[loopN2 * s1BaseA16D], this->s1BaseA16D);
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1Special<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock,
                                     bmm1Format, enableL1Reuse>::CopyInSoftmaxMaxUb(SplitSpecialExtraInfo &extraInfo,
                                                                                    int64_t loopN2)
{
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));

    int64_t vec2S1N2Offset = loopN2 * s1Size * fp32BaseSize;

    LocalTensor<T> maxUb = this->softmaxMaxBuf.template Get<T>();
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    DataCopy(maxUb, this->softmaxMaxGm[extraInfo.softmaxMaxOffset + vec2S1N2Offset], extraInfo.s1RealFp32);
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1Special<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock,
                                     bmm1Format, enableL1Reuse>::ProcessVec1(SplitSpecialExtraInfo &extraInfo)
{
    if constexpr (enableL1Reuse) {
        if (extraInfo.lastNotPair) {
            return;
        }
    }

    extraInfo.softmaxMaxOffset = (extraInfo.boIdx * this->n2S1 + extraInfo.n2oIdx * this->n2BaseSize * s1Size +
                                  extraInfo.s1oIdx * static_cast<int64_t>(s1BaseSize)) *
                                 static_cast<int64_t>(fp32BaseSize);

    LocalTensor<T> stage1PingTensor = this->stage1PingBuf.template Get<T>(); // t.a 32k
    LocalTensor<T> stage1PongTensor = this->stage1PongBuf.template Get<T>(); // i.a 32k
    LocalTensor<T> &actualUseTensor = stage1PongTensor;

    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    event_t eventIdVToMte2A = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    event_t eventIdVToMte2B = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    event_t eventIdVToMte2C = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    event_t eventIdDropMte3ToMte2;
    extraInfo.vec1S1RealSize = extraInfo.vec1S1BaseSize;
    int64_t vecS1S2Align = extraInfo.vec1S1BaseSize * extraInfo.s2AlignedSize;

    for (int32_t loopN2 = 0; loopN2 < extraInfo.n2RealSize; ++loopN2) {
        if (extraInfo.s2LoopCount > 0) {
            CopyInSoftmaxMaxUb(extraInfo, loopN2);
            CopyInSoftmaxSumUb(extraInfo, loopN2);
        }

        for (int32_t loopIdx = 0; loopIdx < extraInfo.realSplitN; loopIdx++) {
            if (loopIdx == extraInfo.realSplitN - 1) {
                extraInfo.vec1S1RealSize = extraInfo.s1RealSize - loopIdx * extraInfo.vec1S1BaseSize;
            }
            if (loopIdx > 0) {
                WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2B);
            } else {
                if constexpr (IsSameType<T, INPUT_T>::value == false && layOutType == LayOutTypeEnum::LAYOUT_TND) {
                    event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                    SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
                    WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
                }
            }

            // FP32场景，需要等待vec1上一轮输出搬完
            if constexpr (IsSameType<INPUT_T, float>::value) {
                event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
                SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
                WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            }
            this->GetBmm1Result(extraInfo, actualUseTensor, loopIdx, loopN2);

            // mul需要等bmm结果搬完
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

            this->CopyInAttenMask(extraInfo, loopIdx, -1, loopN2);

            AscendC::PipeBarrier<PIPE_V>();
            Muls(stage1PingTensor, actualUseTensor, static_cast<T>(this->tilingData->inputParams.scaleValue),
                 extraInfo.vec1S1RealSize * extraInfo.s2AlignedSize);

            if constexpr (hasAtten) {
                SelectWithBytesMaskShapeInfo shapeInfo;
                shapeInfo.firstAxis = extraInfo.vec1S1RealSize;
                shapeInfo.srcLastAxis = extraInfo.s2AlignedSize;
                shapeInfo.maskLastAxis = CeilDiv(extraInfo.s2RealSize, blockBytes) * blockBytes;
                stage1PingTensor.SetSize(extraInfo.vec1S1RealSize * extraInfo.s2AlignedSize);

                uint8_t maskType = 0;
                LocalTensor<uint8_t> attenMaskUb = this->maskTBufPing.template Get<uint8_t>();
                this->ComputeAttenMask(shapeInfo, stage1PingTensor, attenMaskUb, maskType, eventIdMte2ToV);
            }

            if (loopIdx < extraInfo.realSplitN - 1) {
                SetFlag<HardEvent::V_MTE2>(eventIdVToMte2B);
            }

            if (loopIdx > 0) {
                WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2A);
            }

            this->SoftMaxCompute(extraInfo, stage1PingTensor, loopIdx, loopN2);
            if (loopIdx < extraInfo.realSplitN - 1) {
                SetFlag<HardEvent::V_MTE2>(eventIdVToMte2A);
            }

            if (loopIdx > 0) {
                WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
            }
            AscendC::PipeBarrier<PIPE_V>();

            LocalTensor<INPUT_T> stage1CastTensor;
            stage1CastTensor = this->pseTBuf.template Get<INPUT_T>();
            Cast(stage1CastTensor, stage1PingTensor, RoundMode::CAST_ROUND,
                 extraInfo.vec1S1RealSize * extraInfo.s2AlignedSize);
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

            // 注意这里如果S1有尾块
            int64_t n2LoopIOffset = (loopN2 * extraInfo.realSplitN + loopIdx) * vecS1S2Align;
            DataCopy(this->stage1Res[extraInfo.taskIdMod2][n2LoopIOffset], stage1CastTensor,
                     extraInfo.vec1S1RealSize * extraInfo.s2AlignedSize);
            if (loopIdx < extraInfo.realSplitN - 1) {
                SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
            }
        }

        SoftmaxDataCopyOut(extraInfo, loopN2);
    }

    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2A);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2B);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2C);
    return;
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1Special<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock,
                                     bmm1Format, enableL1Reuse>::CopyInAttenMask(SplitSpecialExtraInfo &extraInfo,
                                                                                 int64_t loopIdx, int64_t maskOffset,
                                                                                 int64_t loopN2, bool secondTime)
{
    if constexpr (hasAtten == true) {
        LocalTensor<uint8_t> attenMaskUb;
        attenMaskUb = this->maskTBufPing.template Get<uint8_t>();
        int64_t maskOffset = this->ComputeOffsetForNoCompress(extraInfo, loopIdx, loopN2);
        int64_t s2StrideSize = this->tilingData->inputParams.attenMaskS2Size;

        if (this->attenMaskShapeType == 3) {
            BoolCopyInBrcd(attenMaskUb, this->attenMaskGmInt, maskOffset, extraInfo.vec1S1RealSize,
                           extraInfo.s2RealSize, s2StrideSize);
        } else {
            BoolCopyIn(attenMaskUb, this->attenMaskGmInt, maskOffset, extraInfo.vec1S1RealSize, extraInfo.s2RealSize,
                       s2StrideSize);
        }
        return;
    }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline int64_t FusedFloydAttentionS1s2Bn2gs1Special<
    implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock, bmm1Format,
    enableL1Reuse>::ComputeOffsetForNoCompress(SplitSpecialExtraInfo &extraInfo, int64_t loopIdx, int64_t loopN2)
{
    if constexpr (hasAtten == true) {
        int64_t bOffset = 0;
        int64_t n2Offset = 0;
        int64_t s1Offset = 0;
        int64_t s2Offset = extraInfo.s2LoopCount * s2BaseNratioSize;

        if (this->attenMaskShapeType == 0) {
            bOffset = extraInfo.attenB1SSOffset * this->n2G;
            n2Offset = (extraInfo.n2oIdx * this->n2BaseSize + loopN2) * this->s1S2;
            s1Offset = extraInfo.s1oIdx * this->s1BaseSize * s2Size + loopIdx * extraInfo.vec1S1BaseSize * s2Size;
        } else if (this->attenMaskShapeType == 3) {
            bOffset = this->attenBoIdx * this->n2S2;
            n2Offset = (extraInfo.n2oIdx * this->n2BaseSize + loopN2) * s2Size;
        }

        return bOffset + n2Offset + s1Offset + s2Offset;
    }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1Special<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock,
                                     bmm1Format, enableL1Reuse>::GetBmm1Result(SplitSpecialExtraInfo &extraInfo,
                                                                               LocalTensor<T> &bmm1ResUb,
                                                                               int64_t loopIdx, int64_t loopN2)
{
    int64_t n2BaseOffset = loopN2 * extraInfo.s1RealSize * extraInfo.s2RealSize;

    if (likely(extraInfo.s2AlignedSize == extraInfo.s2RealSize)) {
        DataCopy2D(bmm1ResUb,
                   this->mm1Res[extraInfo.taskIdMod2]
                               [n2BaseOffset + loopIdx * extraInfo.vec1S1BaseSize * extraInfo.s2RealSize],
                   extraInfo.vec1S1RealSize, extraInfo.s2RealSize, extraInfo.s2RealSize);

    } else {
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = extraInfo.vec1S1RealSize;
        dataCopyParams.blockLen = extraInfo.s2RealSize * sizeof(T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPadParams dataCopyPadParams;
        dataCopyPadParams.isPad = true;
        dataCopyPadParams.rightPadding = extraInfo.s2AlignedSize - extraInfo.s2RealSize;
        if (dataCopyPadParams.rightPadding > blockSize) {
            dataCopyPadParams.rightPadding -= blockSize;
            dataCopyParams.dstStride = 1;
            int32_t s2BlockAlignedSize = CeilDiv(extraInfo.s2RealSize, blockSize) * blockSize;
            Duplicate<T>(bmm1ResUb[s2BlockAlignedSize], 0, blockSize, extraInfo.vec1S1RealSize, 0,
                         extraInfo.s2AlignedSize * sizeof(T) / blockBytes);
        }
        dataCopyPadParams.paddingValue = 0;
        DataCopyPad(bmm1ResUb,
                    this->mm1Res[extraInfo.taskIdMod2]
                                [n2BaseOffset + loopIdx * extraInfo.vec1S1BaseSize * extraInfo.s2RealSize],
                    dataCopyParams, dataCopyPadParams);
    }
    uint32_t bmm1ResUbShape[] = {static_cast<uint32_t>(extraInfo.vec1S1RealSize),
                                 static_cast<uint32_t>(extraInfo.s2AlignedSize)};
    bmm1ResUb.SetShapeInfo(ShapeInfo(2, bmm1ResUbShape, DataFormat::ND));
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void FusedFloydAttentionS1s2Bn2gs1Special<
    implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock, bmm1Format,
    enableL1Reuse>::ComputeAttenMask(SelectWithBytesMaskShapeInfo &shapeInfo, LocalTensor<T> &bmm1ResUb,
                                     LocalTensor<uint8_t> &attenMaskUb, const uint8_t maskType, event_t vWaitMte2)
{
    if constexpr (hasAtten == true) {
        LocalTensor<uint8_t> apiTmpBuffer = commonTBuf.template Get<uint8_t>();
        attenMaskUb.SetSize(shapeInfo.firstAxis * shapeInfo.maskLastAxis);
        SetFlag<HardEvent::MTE2_V>(vWaitMte2);
        WaitFlag<HardEvent::MTE2_V>(vWaitMte2);
        event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        if (maskType == 0) {
            SelectWithBytesMask(bmm1ResUb, bmm1ResUb, this->negativeFloatScalar, attenMaskUb, apiTmpBuffer, shapeInfo);
        } else {
            SelectWithBytesMask(bmm1ResUb, this->negativeFloatScalar, bmm1ResUb, attenMaskUb, apiTmpBuffer, shapeInfo);
        }
        return;
    }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1Special<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock,
                                     bmm1Format, enableL1Reuse>::SoftMaxCompute(SplitSpecialExtraInfo &extraInfo,
                                                                                LocalTensor<T> &srcTensor,
                                                                                int64_t loopIdx, int64_t loopN2)
{
    uint32_t bmm1ResUbShape[] = {static_cast<uint32_t>(extraInfo.vec1S1RealSize),
                                 static_cast<uint32_t>(extraInfo.s2AlignedSize)};
    uint32_t bmm1ResUbOrgShape[] = {static_cast<uint32_t>(extraInfo.vec1S1RealSize),
                                    static_cast<uint32_t>(extraInfo.s2RealSize)};
    srcTensor.SetShapeInfo(ShapeInfo(2, bmm1ResUbShape, 2, bmm1ResUbOrgShape, DataFormat::ND));

    uint32_t maxSumShape[] = {static_cast<uint32_t>(extraInfo.vec1S1RealSize), static_cast<uint32_t>(fp32BaseSize)};
    LocalTensor<T> sumUb;
    sumUb = this->softmaxSumBuf[extraInfo.multiCoreInnerIdxMod2]
                .template Get<T>()[loopIdx * extraInfo.vec1S1BaseSize * fp32BaseSize];
    LocalTensor<T> maxUb = this->softmaxMaxBuf.template Get<T>()[loopIdx * extraInfo.vec1S1BaseSize * fp32BaseSize];

    sumUb.SetShapeInfo(ShapeInfo(2, maxSumShape, DataFormat::ND));
    maxUb.SetShapeInfo(ShapeInfo(2, maxSumShape, DataFormat::ND));
    LocalTensor<T> expUb;
    expUb = this->softmaxExpBuf[extraInfo.taskIdMod2]
                .template Get<T>()[loopIdx * extraInfo.vec1S1BaseSize * blockBytes / sizeof(T)];

    expUb.SetShapeInfo(ShapeInfo(2, maxSumShape, DataFormat::ND));
    LocalTensor<uint8_t> apiTmpBuffer = this->commonTBuf.template Get<uint8_t>();
    AscendC::PipeBarrier<PIPE_V>();
    if (unlikely(extraInfo.s2LoopCount == 0)) {
        if (IsBasicBlockInSoftMax(extraInfo.vec1S1RealSize, extraInfo.s2RealSize)) {
            SoftMaxTiling newTiling =
                AscendC::SoftMaxFlashV2TilingFuncImpl(extraInfo.vec1S1RealSize, extraInfo.s2AlignedSize, sizeof(T),
                                                      sizeof(T), apiTmpBuffer.GetSize() / sizeof(T), false, true);
            SoftmaxFlashV2<T, false, true, true, false, SOFTMAX_DEFAULT_CFG>(srcTensor, sumUb, maxUb, srcTensor, expUb,
                                                                             sumUb, maxUb, apiTmpBuffer, newTiling);
        } else {
            SoftMaxTiling newTiling =
                AscendC::SoftMaxFlashV2TilingFuncImpl(extraInfo.vec1S1RealSize, extraInfo.s2AlignedSize, sizeof(T),
                                                      sizeof(T), apiTmpBuffer.GetSize() / sizeof(T), false, false);
            SoftmaxFlashV2<T, false, true, false, false, SOFTMAX_DEFAULT_CFG>(srcTensor, sumUb, maxUb, srcTensor, expUb,
                                                                              sumUb, maxUb, apiTmpBuffer, newTiling);
        }
    } else {
        if (IsBasicBlockInSoftMax(extraInfo.vec1S1RealSize, extraInfo.s2RealSize)) {
            SoftMaxTiling newTiling =
                AscendC::SoftMaxFlashV2TilingFuncImpl(extraInfo.vec1S1RealSize, extraInfo.s2AlignedSize, sizeof(T),
                                                      sizeof(T), apiTmpBuffer.GetSize() / sizeof(T), true, true);
            SoftmaxFlashV2<T, true, true, true, false, SOFTMAX_DEFAULT_CFG>(srcTensor, sumUb, maxUb, srcTensor, expUb,
                                                                            sumUb, maxUb, apiTmpBuffer, newTiling);
        } else {
            SoftMaxTiling newTiling =
                AscendC::SoftMaxFlashV2TilingFuncImpl(extraInfo.vec1S1RealSize, extraInfo.s2AlignedSize, sizeof(T),
                                                      sizeof(T), apiTmpBuffer.GetSize() / sizeof(T), true, false);
            SoftmaxFlashV2<T, true, true, false, false, SOFTMAX_DEFAULT_CFG>(srcTensor, sumUb, maxUb, srcTensor, expUb,
                                                                             sumUb, maxUb, apiTmpBuffer, newTiling);
        }
    }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1Special<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock,
                                     bmm1Format, enableL1Reuse>::IterateBmm2(SplitSpecialExtraInfo &extraInfo)
{
    uint32_t batchNum = this->bmmAddLoops;
    uint32_t loops = CeilDiv(extraInfo.n2RealSize, batchNum);

    int64_t matrixStrideB = 0;
    if (extraInfo.s2LoopLimit > 0) {
        matrixStrideB = dSize * this->s2Size;
    }
    for (uint32_t idx = 0; idx < loops; ++idx) {
        int64_t batchSizes = idx * batchNum;
        this->bmm2.SetTensorA(
            this->stage1Res[extraInfo.taskIdMod2][batchSizes * extraInfo.s1RealSize * this->s2BaseNratioSize]);
        this->bmm2.SetTensorB(this->valueGm[extraInfo.vCoreOffset + batchSizes * this->s2D]);

        int64_t outOffset = batchSizes * extraInfo.s1RealSize * dSize;
        int64_t realBatchNum = Min(batchNum, extraInfo.n2RealSize - batchSizes);
        this->bmm2.template IterateBatch(this->mm2Res[extraInfo.taskIdMod2][outOffset], realBatchNum, realBatchNum,
                                         false, 0, matrixStrideB, 0, false, 0);
    }
    this->bmm2.End();
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1Special<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock,
                                     bmm1Format, enableL1Reuse>::IterateBmm2forValue1(SplitSpecialExtraInfo &extraInfo)
{
    uint32_t loops = (extraInfo.s1RealSize + bmmAddLoops - 1) / bmmAddLoops;
    uint32_t loop_rem = extraInfo.s1RealSize % bmmAddLoops;
    uint32_t realBatchNum = bmmAddLoops;

    for (uint32_t idx = 0; idx < loops; ++idx) {
        if (idx == loops - 1 && loop_rem != 0) {
            realBatchNum = loop_rem;
        }
        int64_t batchOffsetD = bmmAddLoops * idx * this->dSize;
        this->bmm2v2.SetTensorA(this->stage1Res[extraInfo.taskIdMod2][bmmAddLoops * idx * extraInfo.s2AlignedSize]);
        this->bmm2v2.SetTensorB(this->valueGm1[extraInfo.k1CoreOffset + batchOffsetD]);
        this->bmm2v2.IterateBatch(this->mm2Res[extraInfo.taskIdMod2][batchOffsetD], realBatchNum, realBatchNum, false,
                                  0, 0, 0, false, 1);
    }

    this->bmm2v2.End();
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1Special<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock,
                                     bmm1Format, enableL1Reuse>::ProcessVec2(SplitSpecialExtraInfo &extraInfo)
{
    if constexpr (enableL1Reuse) {
        if (extraInfo.lastNotPair) {
            return;
        }
    }

    // 获取缓存bmm2的计算结果
    LocalTensor<T> bmm2ResUb = this->stage2TBuf.template Get<T>();
    LocalTensor<T> stage2BufTensor = this->commonTBuf.template Get<T>();
    int64_t vec2LoopLimit = CeilDiv(extraInfo.s1RealSize, extraInfo.vec2S1BaseSize);

    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    event_t eventIdMte2ToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    extraInfo.vec2S1RealSize = extraInfo.vec2S1BaseSize;

    for (int64_t loopN2 = 0; loopN2 < extraInfo.n2RealSize; ++loopN2) {
        for (int64_t s1oIdx = 0; s1oIdx < vec2LoopLimit; s1oIdx++) {
            if (s1oIdx == vec2LoopLimit - 1) {
                extraInfo.vec2S1RealSize = extraInfo.s1RealSize - s1oIdx * extraInfo.vec2S1BaseSize;
            }
            int64_t mm2ResCalcSize = extraInfo.vec2S1RealSize * dSize;
            int64_t n2Offset = loopN2 * extraInfo.s1RealSize * dSize;
            int64_t mm2ResOffset = s1oIdx * extraInfo.vec2S1BaseSize * dSize;
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            int64_t dAlign8 = (this->dSize + 7) / 8 * 8;

            if (likely(this->dSizeAlign16 == this->dSize)) {
                DataCopy(stage2BufTensor, this->mm2Res[extraInfo.taskIdMod2][n2Offset + mm2ResOffset], mm2ResCalcSize);
            } else {
                DataCopyParams dataCopyParams;
                DataCopyPadParams dataCopyPadParams;
                dataCopyParams.blockCount = extraInfo.vec2S1RealSize;
                dataCopyParams.dstStride = 0;
                dataCopyParams.srcStride = 0;
                dataCopyParams.blockLen = this->dSize * 4;
                dataCopyPadParams.rightPadding = this->dSizeAlign16 - this->dSize;
                dataCopyPadParams.paddingValue = 0;
                if (dataCopyPadParams.rightPadding > blockSize) {
                    // 8对齐场景，内部vector需要16对齐，我们在data copy的时候需要手动补0
                    dataCopyPadParams.rightPadding -= blockSize;
                    dataCopyParams.dstStride = 1;
                    Duplicate<T>(stage2BufTensor[dAlign8], 0, blockSize, extraInfo.vec2S1RealSize, 0,
                                 this->dSizeAlign16 * sizeof(T) / blockBytes);
                }
                DataCopyPad(stage2BufTensor, this->mm2Res[extraInfo.taskIdMod2][n2Offset + mm2ResOffset],
                            dataCopyParams, dataCopyPadParams);
                mm2ResCalcSize = extraInfo.vec2S1RealSize * dSizeAlign16;
                mm2ResOffset = s1oIdx * extraInfo.vec2S1BaseSize * dSizeAlign16;
            }

            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            if (likely(extraInfo.s2LoopCount == 0)) {
                DataCopy(bmm2ResUb, stage2BufTensor, mm2ResCalcSize);
            } else {
                CopyInStage2BufUb(extraInfo, loopN2);

                CopyInSoftmaxExpUb(extraInfo, loopN2);

                this->Bmm2ResultMul(extraInfo, bmm2ResUb, s1oIdx);
                AscendC::PipeBarrier<PIPE_V>();
                Add(bmm2ResUb, bmm2ResUb, stage2BufTensor, mm2ResCalcSize);
            }

            Stage2BufCopyOut(extraInfo, loopN2);
            if (extraInfo.s2LoopCount == extraInfo.s2LoopLimit) {
                CopyInSoftmaxSumUb(extraInfo, loopN2);
                Bmm2ResultDiv(extraInfo, s1oIdx);
                Bmm2DataCopyOut(extraInfo, s1oIdx, mm2ResCalcSize, loopN2);
                event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
                SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
                WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
            }
        }
    }
    return;
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1Special<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock,
                                     bmm1Format, enableL1Reuse>::Bmm2ResultMul(SplitSpecialExtraInfo &extraInfo,
                                                                               LocalTensor<T> &bmm2ResUb,
                                                                               int64_t s1oIdx)
{
    AscendC::PipeBarrier<PIPE_V>();
    LocalTensor<T> expUb;
    expUb = softmaxExpBuf[extraInfo.taskIdMod2].Get<T>();

    BinaryRepeatParams repeatParams;
    repeatParams.src0BlkStride = 0;
    repeatParams.src0RepStride = 1;
    repeatParams.src1RepStride = dSizeAlign16 / blockSize;
    repeatParams.dstRepStride = dSizeAlign16 / blockSize;

    // s1长度可能会超过255限制，修改成双重循环
    // 根据一次最多计算的byte数量，对bmm2Res分组mul
    int32_t loop = dSizeAlign16 / repeatMaxSize;
    int32_t remain = dSizeAlign16 % repeatMaxSize;
    for (int i = 0; i < loop; i++) {
        Mul(bmm2ResUb[i * repeatMaxSize], expUb[s1oIdx * extraInfo.vec2S1BaseSize * 8], bmm2ResUb[i * repeatMaxSize],
            repeatMaxSize, extraInfo.vec2S1RealSize, repeatParams);
    }
    if (likely(remain)) {
        Mul(bmm2ResUb[loop * repeatMaxSize], expUb[s1oIdx * extraInfo.vec2S1BaseSize * 8],
            bmm2ResUb[loop * repeatMaxSize], remain, extraInfo.vec2S1RealSize, repeatParams);
    }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1Special<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock,
                                     bmm1Format, enableL1Reuse>::Bmm2ResultDiv(SplitSpecialExtraInfo &extraInfo,
                                                                               int64_t s1oIdx)
{
    LocalTensor<T> bmm2ResUb = this->stage2TBuf.template Get<T>();

    BinaryRepeatParams repeatParams;
    repeatParams.src0BlkStride = 1;
    repeatParams.src0RepStride = dSizeAlign16 / blockSize;
    repeatParams.src1BlkStride = 0;
    repeatParams.src1RepStride = 1;
    repeatParams.dstRepStride = dSizeAlign16 / blockSize;
    int32_t loop = dSizeAlign16 / repeatMaxSize;
    int32_t remain = dSizeAlign16 % repeatMaxSize;

    LocalTensor<float> sumUb;
    sumUb = softmaxSumBuf[extraInfo.multiCoreInnerIdxMod2].Get<float>();

    int32_t calcSize = sumUb.GetSize();
    // 用optionalInputQueue的queue
    AscendC::PipeBarrier<PIPE_V>();
    if constexpr (IsSameType<T, half>::value) {
        LocalTensor<float> commonBufTensor = this->commonTBuf.template Get<float>();
        Copy(commonBufTensor, sumUb, 64, calcSize / 64, {2, 1, 16, 8});
        Copy(commonBufTensor[8], sumUb, 64, calcSize / 64, {2, 1, 16, 8});
        LocalTensor<T> sumCastTensor = softmaxExpBuf[0].Get<T>(calcSize * 2);

        Cast(sumCastTensor, commonBufTensor, RoundMode::CAST_ROUND, 2 * sumUb.GetSize());
        for (int i = 0; i < loop; i++) {
            Div(bmm2ResUb[i * repeatMaxSize], bmm2ResUb[i * repeatMaxSize], sumCastTensor, repeatMaxSize,
                extraInfo.vec2S1RealSize, repeatParams);
        }
        if (likely(remain)) {
            Div(bmm2ResUb[loop * repeatMaxSize], bmm2ResUb[loop * repeatMaxSize], sumCastTensor, remain,
                extraInfo.vec2S1RealSize, repeatParams);
        }
    } else {
        for (int i = 0; i < loop; i++) {
            Div(bmm2ResUb[i * repeatMaxSize], bmm2ResUb[i * repeatMaxSize],
                sumUb[s1oIdx * extraInfo.vec2S1BaseSize * 8], repeatMaxSize, extraInfo.vec2S1RealSize, repeatParams);
        }
        if (likely(remain)) {
            Div(bmm2ResUb[loop * repeatMaxSize], bmm2ResUb[loop * repeatMaxSize],
                sumUb[s1oIdx * extraInfo.vec2S1BaseSize * 8], remain, extraInfo.vec2S1RealSize, repeatParams);
        }
    }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1Special<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock,
                                     bmm1Format, enableL1Reuse>::Bmm2DataCopyOut(SplitSpecialExtraInfo &extraInfo,
                                                                                 int64_t s1oIdx, int64_t mm2ResCalcSize,
                                                                                 int64_t loopN2)
{
    LocalTensor<T> bmm2ResUb = this->stage2TBuf.template Get<T>();
    LocalTensor<INPUT_T> attenOut = this->stage2TBuf.template Get<INPUT_T>();
    bmm2ResUb.SetSize(mm2ResCalcSize);
    AscendC::PipeBarrier<PIPE_V>();

    if constexpr (!IsSameType<INPUT_T, T>::value) {
        Cast(attenOut, bmm2ResUb, RoundMode::CAST_ROUND, mm2ResCalcSize);
    } else {
        DataCopy(attenOut, bmm2ResUb, mm2ResCalcSize);
    }

    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    DataCopyParams dataCopyParams;
    dataCopyParams.blockLen = this->dSize * sizeof(INPUT_T);
    dataCopyParams.srcStride = 0;
    if constexpr (IsSameType<INPUT_T, float>::value) {
        if (this->dSizeAlign16 - this->dSize >= blockSize) {
            dataCopyParams.srcStride = 1;
        }
    }
    int64_t dstStride = 0;
    int64_t attenOutOffset = this->dSize;
    int64_t datacopyOffset = this->dSize;

    int64_t n2Offset = loopN2 * this->s1D;

    // dataCopyParams.dstStride类型定义uint16_t，65535是其最大值
    if (likely(dstStride <= 65535)) {
        dataCopyParams.blockCount = extraInfo.vec2S1RealSize;
        dataCopyParams.dstStride = static_cast<uint16_t>(dstStride);
        DataCopyPad(
            this->attentionOutGm[extraInfo.qCoreOffset + n2Offset + s1oIdx * extraInfo.vec2S1BaseSize * attenOutOffset],
            attenOut, dataCopyParams);
    } else {
        dataCopyParams.blockCount = 1;
        dataCopyParams.dstStride = 0;

        for (int32_t i = 0; i < extraInfo.vec2S1RealSize; i++) {
            DataCopyPad(this->attentionOutGm[extraInfo.qCoreOffset + n2Offset +
                                             s1oIdx * extraInfo.vec2S1BaseSize * attenOutOffset + i * datacopyOffset],
                        attenOut[i * this->dSizeAlign16], dataCopyParams);
        }
    }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1Special<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock,
                                     bmm1Format, enableL1Reuse>::Stage2BufCopyOut(SplitSpecialExtraInfo &extraInfo,
                                                                                  int64_t loopN2)
{
    LocalTensor<T> bmm2ResUb = this->stage2TBuf.template Get<T>();
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    DataCopy(this->stage2TBufGM[loopN2 * s1BaseA16D], bmm2ResUb, this->s1BaseA16D);

    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1Special<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock,
                                     bmm1Format, enableL1Reuse>::SoftmaxDataCopyOut(SplitSpecialExtraInfo &extraInfo,
                                                                                    int64_t loopN2)
{
    int64_t vec2S1N2Offset = loopN2 * s1Size * fp32BaseSize;

    LocalTensor<float> sumTensor;
    sumTensor = this->softmaxSumBuf[extraInfo.multiCoreInnerIdxMod2].template Get<float>();
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    DataCopy(this->softmaxSumGm[extraInfo.softmaxMaxOffset + vec2S1N2Offset], sumTensor, extraInfo.s1RealFp32);
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);

    LocalTensor<T> expUb;
    expUb = this->softmaxExpBuf[extraInfo.taskIdMod2].template Get<T>();
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    DataCopy(this->softMaxExpGM[extraInfo.taskIdMod2][loopN2 * s1BaseSize * fp32BaseSize], expUb, extraInfo.s1RealFp32);
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);

    LocalTensor<float> maxTensor = this->softmaxMaxBuf.template Get<float>();
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    DataCopy(this->softmaxMaxGm[extraInfo.softmaxMaxOffset + loopN2 * s1Size * fp32BaseSize], maxTensor,
             extraInfo.s1RealFp32);
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
}

#endif // FUSED_FLOYD_ATTENTION_S1S2_BN2GS1_SPECIAL_H
