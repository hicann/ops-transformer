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
 * \file nsa_compress_attention_s1s2_bn2gs1_sab.h
 * \brief
 */

#ifndef NSA_COMPRESS_ATTENTION_S1S2_BN2GS1_SAB_H
#define NSA_COMPRESS_ATTENTION_S1S2_BN2GS1_SAB_H

#include "util.h"
#include "dropmask.h"
#include "nsa_compress_attention_common.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "pse.h"

using matmul::MatmulType;

struct SplitSameABExtraInfo {
    int64_t s1oIdx;
    int64_t boIdx;
    int64_t n2oIdx;
    int64_t goIdx;

    int64_t taskId;
    int8_t taskIdMod2;
    int32_t s1RealSize;
    int32_t cubeS1RealSize;
    int32_t cubeS1gRealSize;
    int32_t s2RealSize;
    int32_t s2AlignedSize;
    int32_t s2RealSizeAlign32 = 0;

    int32_t vec1S1RealSize;
    int32_t vec1MaxG;
    int32_t vec1LoopCountG;
    int64_t multiCoreInnerIdx;
    int64_t qCoreOffset;
    int64_t s1SizeAcc;
    int64_t s2SizeAcc;
    int64_t attenMaskS2Size;
    int64_t s1Size;
    int64_t s2Size;
    int64_t softmaxMaxOffset;
    int64_t vecCoreOffset;
};

enum class MmPolicyType : uint32_t {
    NORMAL = 0
};

constexpr int64_t NSA_BASE_S1G_SIZE = 128;
constexpr int64_t GM_DOUBLE_BUFFER = 2;
constexpr int64_t MAX_ADD_NUM = 64;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t MAX_TIMES = 100;
constexpr AscendC::SoftmaxConfig SOFTMAX_DEFAULT_CFG = {false};
constexpr AscendC::SoftmaxConfig SOFTMAX_REDUCE_CFG = {false, 0, 0, AscendC::SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC};

// L1 extension
template <MmPolicyType mmPolicyType>
struct MatmulPolicySelector {
    template <const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : AscendC::Impl::Detail::MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {};
};

// INPUT_T - means data type for input
// T       - means data type when calc
template <LayOutTypeEnum layOutType, bool hasAtten, bool hasTopkMask, typename INPUT_T, typename T = INPUT_T,
          CubeFormat bmm1Format = CubeFormat::ND, MmPolicyType mmPolicyType = MmPolicyType::NORMAL>
class NsaCompressAttentionS1s2Bn2gs1SameAB {
public:
    __aicore__ inline NsaCompressAttentionS1s2Bn2gs1SameAB(){};

    __aicore__ inline void Process();

    // define matmul
    using a1Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, false, LayoutMode::NONE, true>;
    using b1Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, true, LayoutMode::NONE, true>;
    using bias1Type = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using c1Type = MatmulType<TPosition::GM, CubeFormat::ND, T>;
    matmul::Matmul<a1Type, b1Type, c1Type, bias1Type, CFG_EXCEED, matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>,
                   MatmulPolicySelector<mmPolicyType>::template Result>
        bmm1;

    // define batchmatmul
    using a2Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, false, LayoutMode::NONE, true>;
    using b2Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, false, LayoutMode::NONE, true>;
    using bias2Type = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using c2NzType = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T>;
    matmul::Matmul<a2Type, b2Type, c2NzType, bias2Type, CFG_EXCEED,
                   matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>,
                   MatmulPolicySelector<mmPolicyType>::template Result>
        bmm2;

protected:
    __aicore__ inline void InitInput(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                     __gm__ uint8_t *attenMask, __gm__ uint8_t *topkMask, __gm__ uint8_t *softmaxMax,
                                     __gm__ uint8_t *softmaxSum, __gm__ uint8_t *attentionOut,
                                     __gm__ uint8_t *topkIndicesOut, __gm__ uint8_t *workspace,
                                     const NsaCompressAttentionGeneralTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void WaitBmm1Result(SplitSameABExtraInfo &extraInfo);
    __aicore__ inline void WaitBmm2Result(SplitSameABExtraInfo &extraInfo);
    __aicore__ inline void IterateBmm2(SplitSameABExtraInfo &extraInfo,
                                       matmul::Matmul<a2Type, b2Type, c2NzType, bias2Type, CFG_EXCEED,
                                                      matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>,
                                                      MatmulPolicySelector<mmPolicyType>::template Result> &bmm2);
    __aicore__ inline void SetTiling(const NsaCompressAttentionGeneralTilingData *__restrict tilingData);
    __aicore__ inline void InitBuffer();
    template <typename T2>
    __aicore__ inline void IterateBmm1(
        SplitSameABExtraInfo &extraInfo,
        matmul::Matmul<a1Type, b1Type, T2, bias1Type, CFG_EXCEED, matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>,
                       MatmulPolicySelector<mmPolicyType>::template Result> &bmm1);
    template <typename T2>
    __aicore__ inline void Bmm1SetTensorA(
        SplitSameABExtraInfo &extraInfo,
        matmul::Matmul<a1Type, b1Type, T2, bias1Type, CFG_EXCEED, matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>,
                       MatmulPolicySelector<mmPolicyType>::template Result> &bmm1);
    template <typename T2>
    __aicore__ inline void SetBmm1TensorB(
        SplitSameABExtraInfo &extraInfo,
        matmul::Matmul<a1Type, b1Type, T2, bias1Type, CFG_EXCEED, matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>,
                       MatmulPolicySelector<mmPolicyType>::template Result> &bmm1);
    __aicore__ inline void ComputeBmm1Tail(SplitSameABExtraInfo &extraInfo);
    __aicore__ inline void ProcessVec1(SplitSameABExtraInfo &extraInfo);
    __aicore__ inline void ProcessVec2(SplitSameABExtraInfo &extraInfo);
    __aicore__ inline void GetBmm1Result(SplitSameABExtraInfo &extraInfo, LocalTensor<T> &bmm1ResUb, int64_t s1LoopIdx,
                                         int64_t gLoopIdx);
    __aicore__ inline void CopyInAttenMask(SplitSameABExtraInfo &extraInfo, LocalTensor<uint8_t> &attenMaskUb,
                                           int64_t s1LoopIdx);
    __aicore__ inline void TopkCompute(SplitSameABExtraInfo &extraInfo);
    __aicore__ inline void TransposeTilingCompute(ConfusionTransposeTiling &tiling, uint64_t row, uint64_t col);
    __aicore__ inline void TopkTilingCompute(TopkTiling &tiling, uint64_t inner, uint64_t outter);

    uint32_t cubeS1BaseSize;
    uint32_t vecS1BaseSize;
    uint32_t dSize;
    uint32_t d2Size;
    int64_t s1Size;
    int64_t s2Size;
    int64_t s1OuterSize;
    int64_t gSize;
    int64_t n2G;
    int64_t s2SizeSum;
    int64_t vecMaxG;

    int64_t s1TotalSize;
    int64_t s2TotalSize;
    int64_t qCoreOffset;

    TBuf<> allUbBuffer;
    LocalTensor<uint8_t> allUbLocal;

    LocalTensor<T> softmaxExpUb;
    GlobalTensor<T> mm1Res[2];
    GlobalTensor<T> mm2Res[2];
    GlobalTensor<T> vec2Res[2];
    GlobalTensor<INPUT_T> stage1Res;
    GlobalTensor<T> impScoreRes[2];

    uint32_t negativeIntScalar = NEGATIVE_MIN_VAULE_FP32;
    T negativeFloatScalar;
    T positiveFloatScalar;

    int32_t vecBlockIdx;
    int32_t cubeBlockIdx;
    int32_t cubeSubIdx;
    const NsaCompressAttentionGeneralTilingData *__restrict tilingData;
    SoftMaxTiling softmaxtiling;
    TopkTiling topktiling;

    int64_t boIdx;
    int64_t n2oIdx;
    int64_t goIdx;
    int64_t s1oIdx;

    TPipe *pipe;

    GlobalTensor<INPUT_T> queryGm;
    GlobalTensor<INPUT_T> keyGm;
    GlobalTensor<INPUT_T> valueGm;
    GlobalTensor<INPUT_T> attentionOutGm;
    GlobalTensor<float> softmaxMaxGm;
    GlobalTensor<float> softmaxSumGm;
    GlobalTensor<uint8_t> attenMaskGmInt;
    GlobalTensor<uint8_t> topKMaskGmInt;
    GlobalTensor<int32_t> topkIndicesOutGm;

    // Softmax
    LocalTensor<T> mm1ResultLocal;
    LocalTensor<T> softMaxResultLocal;
    LocalTensor<uint8_t> softmaxTmpLocal;
    LocalTensor<T> softMaxMaxLocal;
    LocalTensor<T> softMaxSumLocal;
    LocalTensor<INPUT_T> castTmpLocal;
    LocalTensor<uint8_t> attenMaskLocal;

    // Importance Score
    LocalTensor<uint8_t> maskTensor;
    LocalTensor<float> softmaxRes;
    LocalTensor<float> scoreRes;
    LocalTensor<float> trans;
    LocalTensor<float> transBack;
    LocalTensor<uint8_t> tmpBuf;
    LocalTensor<uint8_t> sharedBuf;
    int timesArray[MAX_TIMES]; // isInfo.isM + isInfo.isN -1  < 100
    int32_t scoreLoop;
    int32_t innerLoop;
    uint64_t s2Loop;
    uint64_t s2Aligned64B;
    uint64_t maskLenAligned32B;
    uint64_t loopIdx;
    uint64_t remainS2;
    uint64_t s2Length;

    ConfusionTransposeTiling transposeInfoForward;
    ConfusionTransposeTiling transposeInfoBackward;
    // Importance Score use end

    // topK
    LocalTensor<T> sorceResultLocal;
    LocalTensor<int32_t> topKIdexLocal;
    LocalTensor<T> topKValueLocal;
    LocalTensor<uint8_t> topKTmpLocal;
    LocalTensor<bool> srcLocalFinish;
    LocalTensor<int32_t> srcLocalIndex;
};

template <LayOutTypeEnum layOutType, bool hasAtten, bool hasTopkMask, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void NsaCompressAttentionS1s2Bn2gs1SameAB<
    layOutType, hasAtten, hasTopkMask, INPUT_T, T, bmm1Format,
    mmPolicyType>::InitInput(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                             __gm__ uint8_t *attenMask, __gm__ uint8_t *topkMask, __gm__ uint8_t *softmaxMax,
                             __gm__ uint8_t *softmaxSum, __gm__ uint8_t *attentionOut, __gm__ uint8_t *topkIndicesOut,
                             __gm__ uint8_t *workspace, const NsaCompressAttentionGeneralTilingData *__restrict tiling,
                             TPipe *tPipe)
{
    this->vecBlockIdx = GetBlockIdx();
    this->cubeBlockIdx = this->vecBlockIdx / 2;
    this->cubeSubIdx = this->vecBlockIdx % 2;
    this->pipe = tPipe;
    this->SetTiling(tiling);
    this->softmaxtiling = this->tilingData->softmaxTilingData;
    this->topktiling = this->tilingData->topkTilingData;

    // init global buffer
    this->queryGm.SetGlobalBuffer((__gm__ INPUT_T *)query);
    this->keyGm.SetGlobalBuffer((__gm__ INPUT_T *)key);
    this->valueGm.SetGlobalBuffer((__gm__ INPUT_T *)value);
    this->attenMaskGmInt.SetGlobalBuffer((__gm__ uint8_t *)attenMask);
    this->topKMaskGmInt.SetGlobalBuffer((__gm__ uint8_t *)topkMask);
    this->softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmaxMax);
    this->softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmaxSum);
    this->attentionOutGm.SetGlobalBuffer((__gm__ INPUT_T *)attentionOut);
    this->topkIndicesOutGm.SetGlobalBuffer((__gm__ int32_t *)topkIndicesOut);

    int64_t mm1BaseCount = NSA_BASE_S1G_SIZE * this->tilingData->inputParams.alignedS2;
    int64_t mm1BaseSize = mm1BaseCount * sizeof(float); // s2的最大值 //向上16对齐
    int64_t mm1BaseSizeHalf = mm1BaseCount * sizeof(INPUT_T);
    int64_t totalOffset = mm1BaseSize * GM_DOUBLE_BUFFER + mm1BaseSizeHalf;

    // bmm1Result
    this->mm1Res[0].SetGlobalBuffer((__gm__ T *)(workspace + this->cubeBlockIdx * totalOffset));
    this->mm1Res[1].SetGlobalBuffer((__gm__ T *)(workspace + this->cubeBlockIdx * totalOffset + mm1BaseSize));

    // stage1Result
    this->stage1Res.SetGlobalBuffer(
        (__gm__ INPUT_T *)(workspace + this->cubeBlockIdx * totalOffset + mm1BaseSize * GM_DOUBLE_BUFFER));

    // impScore
    int64_t totalAicNum = AscendC::GetBlockNum();
    uint64_t impScoreOffsetU64 = static_cast<uint64_t>(totalOffset) * static_cast<uint64_t>(totalAicNum);
    uint64_t baseS1G =
        static_cast<uint64_t>(NSA_BASE_S1G_SIZE) / static_cast<uint64_t>(this->tilingData->inputParams.gSize);
    uint64_t alignedS2Div = static_cast<uint64_t>(this->tilingData->inputParams.alignedS2) /
                            static_cast<uint64_t>(this->tilingData->importanceScoreParams.isM);
    uint64_t perCoreOffsetU64 = baseS1G * alignedS2Div * sizeof(float) * GM_DOUBLE_BUFFER;
    this->impScoreRes[0].SetGlobalBuffer(
        (__gm__ T *)(workspace + impScoreOffsetU64 + perCoreOffsetU64 * static_cast<uint64_t>(this->cubeBlockIdx)));
    this->impScoreRes[1].SetGlobalBuffer((__gm__ T *)(workspace + impScoreOffsetU64 +
                                                      perCoreOffsetU64 * static_cast<uint64_t>(this->cubeBlockIdx) +
                                                      perCoreOffsetU64 / 2));

    GetExtremeValue(this->negativeFloatScalar, this->positiveFloatScalar);
}

template <LayOutTypeEnum layOutType, bool hasAtten, bool hasTopkMask, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void NsaCompressAttentionS1s2Bn2gs1SameAB<
    layOutType, hasAtten, hasTopkMask, INPUT_T, T, bmm1Format,
    mmPolicyType>::SetTiling(const NsaCompressAttentionGeneralTilingData *__restrict tilingData)
{
    // copy base params
    this->tilingData = tilingData;
    this->cubeS1BaseSize = this->tilingData->coreParams.s1BaseSize;
    this->vecS1BaseSize = this->cubeS1BaseSize / 2;
    this->dSize = this->tilingData->inputParams.dSize;
    this->d2Size = this->tilingData->coreParams.d2Size;
};

template <LayOutTypeEnum layOutType, bool hasAtten, bool hasTopkMask, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void NsaCompressAttentionS1s2Bn2gs1SameAB<layOutType, hasAtten, hasTopkMask, INPUT_T, T, bmm1Format,
                                                            mmPolicyType>::InitBuffer()
{
    int64_t tempBufSize = 32 * 1024;
    int64_t availUbSize = 192 * 1024;
    int64_t attenMaskAlign = CeilDiv(this->tilingData->inputParams.s2Size, 32) * 32;
    this->vecMaxG = (availUbSize - tempBufSize) / (2 * sizeof(float) * this->tilingData->inputParams.alignedS2 +
                                                   1 * sizeof(INPUT_T) * this->tilingData->inputParams.alignedS2 +
                                                   1 * sizeof(uint8_t) * attenMaskAlign + 2 * sizeof(float) * 8);
    this->vecMaxG = Min(this->vecMaxG, this->gSize);
    this->pipe->InitBuffer(this->allUbBuffer, availUbSize);
    allUbLocal = allUbBuffer.Get<uint8_t>();
    auto softmaxLen = this->vecMaxG * this->tilingData->inputParams.alignedS2;
    auto softmaxMaxLen = this->vecMaxG * 8;
    auto attenMaskLen = this->vecMaxG * attenMaskAlign;

    // softmax ub
    int64_t addrOffset = 0;
    mm1ResultLocal = allUbBuffer.GetWithOffset<T>(softmaxLen, addrOffset);
    addrOffset = addrOffset + softmaxLen * sizeof(float);
    softMaxResultLocal = allUbBuffer.GetWithOffset<T>(softmaxLen, addrOffset);
    addrOffset = addrOffset + softmaxLen * sizeof(float);
    castTmpLocal = allUbBuffer.GetWithOffset<INPUT_T>(softmaxLen, addrOffset);
    addrOffset = addrOffset + softmaxLen * sizeof(INPUT_T);
    softmaxTmpLocal = allUbBuffer.GetWithOffset<uint8_t>(tempBufSize / sizeof(uint8_t), addrOffset);
    addrOffset = addrOffset + tempBufSize;
    softMaxMaxLocal = allUbBuffer.GetWithOffset<T>(softmaxMaxLen, addrOffset);
    addrOffset = addrOffset + softmaxMaxLen * sizeof(float);
    softMaxSumLocal = allUbBuffer.GetWithOffset<T>(softmaxMaxLen, addrOffset);
    addrOffset = addrOffset + softmaxMaxLen * sizeof(float);
    attenMaskLocal = allUbBuffer.GetWithOffset<uint8_t>(attenMaskLen, addrOffset);

    // importance score ub
    const auto &isInfo = this->tilingData->importanceScoreParams;
    this->innerLoop = isInfo.innerLoop;
    this->s2Length = isInfo.s2ScoreLoopLen;
    this->scoreLoop = isInfo.outerLoop;
    this->s2Aligned64B = (scoreLoop + 15) / 16 * 16;
    this->maskLenAligned32B = (s2Aligned64B + 31) / 32 * 32;
    this->transposeInfoForward = isInfo.confusionTransposeTilingData;
    this->transposeInfoBackward = isInfo.confusionTransposeTilingData2;

    addrOffset = 0;
    softmaxRes = allUbLocal[addrOffset].template ReinterpretCast<T>();
    addrOffset += this->vecS1BaseSize * this->gSize * isInfo.s2ScoreLoopLen * sizeof(float);
    trans = allUbLocal[addrOffset].template ReinterpretCast<T>();
    addrOffset += this->vecS1BaseSize * this->gSize * isInfo.s2ScoreLoopLen * sizeof(float);
    scoreRes = allUbLocal[addrOffset].template ReinterpretCast<T>();
    addrOffset += this->s2Aligned64B * this->vecS1BaseSize * this->gSize * sizeof(float);
    transBack = allUbBuffer.GetWithOffset<T>(this->vecS1BaseSize * this->gSize * s2Aligned64B, addrOffset);
    if (this->gSize >= 4) {
        tmpBuf = allUbLocal[addrOffset + this->vecS1BaseSize * s2Aligned64B * sizeof(float)];
        addrOffset += this->vecS1BaseSize * this->gSize * s2Aligned64B * sizeof(float);
    } else {
        addrOffset += this->vecS1BaseSize * this->gSize * s2Aligned64B * sizeof(float);
        tmpBuf = allUbLocal[addrOffset];
        addrOffset += 16 * 1024;
    }
    maskTensor = allUbLocal[addrOffset].template ReinterpretCast<uint8_t>();
    sharedBuf = allUbLocal;
    for (int i = 0; i < this->innerLoop; ++i) {
        if (i < this->innerLoop / 2) {
            this->timesArray[i] = Min(i + 1, isInfo.isN);
        } else {
            this->timesArray[i] = Min(this->innerLoop - i, isInfo.isN);
        }
    }
}

template <LayOutTypeEnum layOutType, bool hasAtten, bool hasTopkMask, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void
NsaCompressAttentionS1s2Bn2gs1SameAB<layOutType, hasAtten, hasTopkMask, INPUT_T, T, bmm1Format,
                                     mmPolicyType>::WaitBmm1Result(SplitSameABExtraInfo &extraInfo)
{
    this->bmm1.WaitIterateAll();
    this->bmm1.End();
}

template <LayOutTypeEnum layOutType, bool hasAtten, bool hasTopkMask, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType>
template <typename T2>
__aicore__ inline void
NsaCompressAttentionS1s2Bn2gs1SameAB<layOutType, hasAtten, hasTopkMask, INPUT_T, T, bmm1Format, mmPolicyType>::
    IterateBmm1(
        SplitSameABExtraInfo &extraInfo,
        matmul::Matmul<a1Type, b1Type, T2, bias1Type, CFG_EXCEED, matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>,
                       MatmulPolicySelector<mmPolicyType>::template Result> &bmm1)
{
    bmm1.SetOrgShape(extraInfo.cubeS1gRealSize, extraInfo.s2RealSize, this->dSize,
                     this->tilingData->inputParams.n2Size * this->dSize, extraInfo.s2RealSize); // b,s2,n2,d

    this->Bmm1SetTensorA(extraInfo, bmm1);
    this->SetBmm1TensorB(extraInfo, bmm1);
    bmm1.template IterateAll<false>(this->mm1Res[extraInfo.taskIdMod2], false, false, true);
}

template <LayOutTypeEnum layOutType, bool hasAtten, bool hasTopkMask, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType>
template <typename T2>
__aicore__ inline void
NsaCompressAttentionS1s2Bn2gs1SameAB<layOutType, hasAtten, hasTopkMask, INPUT_T, T, bmm1Format, mmPolicyType>::
    Bmm1SetTensorA(
        SplitSameABExtraInfo &extraInfo,
        matmul::Matmul<a1Type, b1Type, T2, bias1Type, CFG_EXCEED, matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>,
                       MatmulPolicySelector<mmPolicyType>::template Result> &bmm1)
{
    // 计算gm上的offset
    int64_t bOffset = 0;

    // s1需要考虑inner轴的影响
    int64_t s1gOffset = 0;
    int64_t n2Offset = 0;

    if constexpr (layOutType == LayOutTypeEnum::LAYOUT_TND) {
        uint64_t n2OffsetU64 = static_cast<uint64_t>(extraInfo.n2oIdx) * static_cast<uint64_t>(this->s1TotalSize);
        n2OffsetU64 = n2OffsetU64 * static_cast<uint64_t>(this->gSize) * static_cast<uint64_t>(this->dSize);
        uint64_t bOffsetU64 = static_cast<uint64_t>(extraInfo.s1SizeAcc) * static_cast<uint64_t>(this->gSize) *
                              static_cast<uint64_t>(this->dSize);
        uint64_t s1gOffsetU64 =
            static_cast<uint64_t>(extraInfo.s1oIdx) * static_cast<uint64_t>(this->tilingData->coreParams.s1BaseSize);
        s1gOffsetU64 = s1gOffsetU64 * static_cast<uint64_t>(this->gSize) * static_cast<uint64_t>(this->dSize);
        n2Offset = static_cast<int64_t>(n2OffsetU64);
        bOffset = static_cast<int64_t>(bOffsetU64);
        s1gOffset = static_cast<int64_t>(s1gOffsetU64);
    }
    this->qCoreOffset = n2Offset + bOffset + s1gOffset;
    extraInfo.qCoreOffset = this->qCoreOffset;
    bmm1.SetTensorA(this->queryGm[extraInfo.qCoreOffset]);
}

template <LayOutTypeEnum layOutType, bool hasAtten, bool hasTopkMask, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType>
template <typename T2>
__aicore__ inline void
NsaCompressAttentionS1s2Bn2gs1SameAB<layOutType, hasAtten, hasTopkMask, INPUT_T, T, bmm1Format, mmPolicyType>::
    SetBmm1TensorB(
        SplitSameABExtraInfo &extraInfo,
        matmul::Matmul<a1Type, b1Type, T2, bias1Type, CFG_EXCEED, matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>,
                       MatmulPolicySelector<mmPolicyType>::template Result> &bmm1)
{
    // 计算gm上的offset
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    if constexpr (layOutType == LayOutTypeEnum::LAYOUT_TND) {
        // b,s2,n2,d
        bOffset = extraInfo.s2SizeAcc * this->tilingData->inputParams.n2Size * this->dSize;
        n2Offset = extraInfo.n2oIdx * this->dSize;
    }
    int64_t kCoreOffset = n2Offset + bOffset;
    bmm1.SetTensorB(this->keyGm[kCoreOffset], true);
    bmm1.SetTail(extraInfo.cubeS1gRealSize, extraInfo.s2RealSize);
}

template <LayOutTypeEnum layOutType, bool hasAtten, bool hasTopkMask, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void NsaCompressAttentionS1s2Bn2gs1SameAB<layOutType, hasAtten, hasTopkMask, INPUT_T, T, bmm1Format,
                                                            mmPolicyType>::ProcessVec1(SplitSameABExtraInfo &extraInfo)
{
    if (extraInfo.s1RealSize == 0) {
        return;
    }

    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
    event_t eventIdMte3ToV1 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
    event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());

    AscendC::SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    AscendC::SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    AscendC::SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV1);
    // 外循环：[0, s1RealSize]
    // 内循环：[0, Ceil(g, vecMaxG)]
    for (int32_t s1LoopIdx = 0; s1LoopIdx < extraInfo.s1RealSize; ++s1LoopIdx) {
        for (int32_t gLoopIdx = 0; gLoopIdx < extraInfo.vec1LoopCountG; ++gLoopIdx) {
            extraInfo.vec1S1RealSize = extraInfo.vec1MaxG;
            if (gLoopIdx == extraInfo.vec1LoopCountG - 1) {
                extraInfo.vec1S1RealSize = this->gSize - gLoopIdx * extraInfo.vec1MaxG;
            }
            AscendC::WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            this->GetBmm1Result(extraInfo, mm1ResultLocal, s1LoopIdx, gLoopIdx);
            this->CopyInAttenMask(extraInfo, attenMaskLocal, s1LoopIdx);

            // 正向Wait
            AscendC::SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            AscendC::WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

            // scale
            Muls(mm1ResultLocal, mm1ResultLocal, static_cast<T>(this->tilingData->inputParams.scaleValue),
                 extraInfo.vec1S1RealSize * extraInfo.s2RealSizeAlign32);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
            if constexpr (hasAtten == true) {
                // SelectWithBytesMask
                AscendC::SelectWithBytesMaskShapeInfo selectShapeInfo;
                selectShapeInfo.firstAxis = (uint32_t)extraInfo.vec1S1RealSize;
                selectShapeInfo.srcLastAxis = (uint32_t)extraInfo.s2RealSizeAlign32;
                selectShapeInfo.maskLastAxis = (uint32_t)CeilDiv(extraInfo.s2RealSize, 32) * 32;
                AscendC::SelectWithBytesMask(mm1ResultLocal, mm1ResultLocal, this->negativeFloatScalar, attenMaskLocal,
                                             softmaxTmpLocal, selectShapeInfo);
                AscendC::PipeBarrier<PIPE_V>();
            }

            AscendC::WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV1);
            SoftMaxShapeInfo srcShape = {(uint32_t)extraInfo.vec1S1RealSize, (uint32_t)extraInfo.s2RealSizeAlign32,
                                         (uint32_t)extraInfo.vec1S1RealSize, (uint32_t)extraInfo.s2RealSize};
            AscendC::SoftMax<T>(softMaxResultLocal, softMaxSumLocal, softMaxMaxLocal, mm1ResultLocal, softmaxTmpLocal,
                                this->softmaxtiling, srcShape);
            AscendC::SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);

            // 正向Wait
            AscendC::SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            AscendC::WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

            AscendC::PipeBarrier<PIPE_V>();
            DataCopyPad(this->mm1Res[extraInfo.taskIdMod2][(extraInfo.vecCoreOffset + s1LoopIdx * this->gSize +
                                                            gLoopIdx * extraInfo.vec1MaxG) *
                                                           extraInfo.s2RealSize],
                        softMaxResultLocal,
                        {(uint16_t)extraInfo.vec1S1RealSize, (uint32_t)(extraInfo.s2RealSize * sizeof(float)),
                         (uint32_t)(extraInfo.s2RealSizeAlign32 - extraInfo.s2RealSize) / 8, 0, 0});
            DataCopyPad(softmaxSumGm[extraInfo.softmaxMaxOffset +
                                     (gLoopIdx * extraInfo.vec1MaxG * extraInfo.s1Size + s1LoopIdx) * 8],
                        softMaxSumLocal,
                        {(uint16_t)extraInfo.vec1S1RealSize, (uint32_t)(8 * sizeof(float)), 0,
                         (uint32_t)((extraInfo.s1Size - 1) * 8 * sizeof(float)), 0});
            DataCopyPad(softmaxMaxGm[extraInfo.softmaxMaxOffset +
                                     (gLoopIdx * extraInfo.vec1MaxG * extraInfo.s1Size + s1LoopIdx) * 8],
                        softMaxMaxLocal,
                        {(uint16_t)extraInfo.vec1S1RealSize, (uint32_t)(8 * sizeof(float)), 0,
                         (uint32_t)((extraInfo.s1Size - 1) * 8 * sizeof(float)), 0});
            AscendC::SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV1);

            Cast(castTmpLocal, softMaxResultLocal, RoundMode::CAST_RINT,
                 extraInfo.vec1S1RealSize * extraInfo.s2RealSizeAlign32);
            AscendC::PipeBarrier<PIPE_V>();

            // bf16 datacopy --> GM
            AscendC::SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            AscendC::WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            DataCopy(
                this->stage1Res[(extraInfo.vecCoreOffset + s1LoopIdx * this->gSize + gLoopIdx * extraInfo.vec1MaxG) *
                                extraInfo.s2RealSizeAlign32],
                castTmpLocal, extraInfo.vec1S1RealSize * extraInfo.s2RealSizeAlign32);
            AscendC::SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        }
    }
    AscendC::WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    AscendC::WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    AscendC::WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV1);

    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventIdMte3ToV);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventIdMte3ToV1);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMte3);

    return;
}

template <LayOutTypeEnum layOutType, bool hasAtten, bool hasTopkMask, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void NsaCompressAttentionS1s2Bn2gs1SameAB<layOutType, hasAtten, hasTopkMask, INPUT_T, T, bmm1Format,
                                                            mmPolicyType>::ProcessVec2(SplitSameABExtraInfo &extraInfo)
{
    if (extraInfo.s1RealSize == 0) {
        return;
    }

    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
    event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
    event_t eventIdMte3ToVMask = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
    event_t eventIdMte2ToVMask = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());

    const auto &isInfo = this->tilingData->importanceScoreParams;
    uint64_t impScoreNums = CeilDiv(static_cast<uint64_t>(extraInfo.s2RealSize), isInfo.isM);
    this->s2Loop = CeilDiv(impScoreNums, isInfo.outerLoop);
    this->innerLoop = isInfo.innerLoop;
    this->s2Length = isInfo.s2ScoreLoopLen;
    this->scoreLoop = isInfo.outerLoop;
    this->s2Aligned64B = (scoreLoop + 15) / 16 * 16;
    this->maskLenAligned32B = (s2Aligned64B + 31) / 32 * 32;

    TransposeTilingCompute(transposeInfoForward, this->vecS1BaseSize * this->gSize, this->s2Length);
    TransposeTilingCompute(transposeInfoBackward, this->s2Aligned64B, this->vecS1BaseSize * this->gSize);

    AscendC::SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    AscendC::SetFlag<HardEvent::MTE3_V>(eventIdMte3ToVMask);
    AscendC::SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    // [0, s2] loop
    for (int64_t loopIdx = 0; loopIdx < this->s2Loop; ++loopIdx) {
        // s2 last loop adapt
        if (loopIdx == this->s2Loop - 1) { // lastLoop
            scoreLoop = impScoreNums - isInfo.outerLoop * (this->s2Loop - 1);
            s2Aligned64B = (scoreLoop + 15) / 16 * 16;
            s2Length = (scoreLoop - 1) * isInfo.isM + this->innerLoop;
            s2Length = (s2Length + 15) / 16 * 16;
            maskLenAligned32B = (s2Aligned64B + 31) / 32 * 32;
            TransposeTilingCompute(transposeInfoForward, this->vecS1BaseSize * this->gSize, this->s2Length);
            TransposeTilingCompute(transposeInfoBackward, this->s2Aligned64B, this->vecS1BaseSize * this->gSize);
        }

        // CopyInSoftmaxRes
        AscendC::WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        uint64_t s2Offset = isInfo.outerLoop * isInfo.isM * loopIdx;
        if (loopIdx >= 1) {
            s2Offset -= innerLoop - isInfo.isM;
        }
        if (isInfo.isN == 1) {
            s2Offset += 1;
        }
        uint32_t softmaxSrcBlockLen = this->s2Length <= extraInfo.s2RealSize ? this->s2Length : extraInfo.s2RealSize;
        uint32_t softmaxSrcStride = extraInfo.s2RealSize - softmaxSrcBlockLen;
        uint32_t softmaxDstStride = 0;
        if (loopIdx == this->s2Loop - 1) {
            softmaxSrcBlockLen = extraInfo.s2RealSize - s2Offset;
            softmaxSrcStride = s2Offset;
            softmaxDstStride = this->s2Length + s2Offset - extraInfo.s2RealSize;
        }

        AscendC::DataCopyExtParams params = {static_cast<uint16_t>(this->vecS1BaseSize * this->gSize),
                                             static_cast<uint32_t>(softmaxSrcBlockLen * sizeof(float)),
                                             static_cast<uint16_t>(softmaxSrcStride * sizeof(float)),
                                             static_cast<uint32_t>(softmaxDstStride * sizeof(float) / 32), 0};
        AscendC::DataCopyPadExtParams<T> extParams = {true, 0, static_cast<uint8_t>(softmaxDstStride % 8), 0.0};
        AscendC::DataCopyPad(
            softmaxRes, this->mm1Res[extraInfo.taskIdMod2][extraInfo.vecCoreOffset * extraInfo.s2RealSize + s2Offset],
            params, extParams);

        // CopyInTopKMask
        AscendC::SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        if constexpr (hasTopkMask == true) {
            uint64_t maskOffset = isInfo.outerLoop * loopIdx;
            uint64_t s1Offset = static_cast<uint64_t>(extraInfo.s1oIdx) * static_cast<uint64_t>(this->cubeS1BaseSize);
            uint64_t vecOffset = static_cast<uint64_t>(extraInfo.vecCoreOffset) / static_cast<uint64_t>(this->gSize);
            uint64_t totalSeqOffset = s1Offset + vecOffset;
            uint64_t s2Div = CeilDiv(static_cast<uint64_t>(this->tilingData->inputParams.s2Size), isInfo.isM);
            uint64_t maskGmOffset = totalSeqOffset * s2Div;
            uint64_t maskPad = maskLenAligned32B - scoreLoop;
            uint32_t maskSrcStride = static_cast<uint32_t>((this->s2Aligned64B - this->scoreLoop) / 32);
            if (loopIdx == this->s2Loop - 1) {
                maskSrcStride = 0;
            }
            AscendC::DataCopyExtParams params = {
                static_cast<uint16_t>(this->vecS1BaseSize), static_cast<uint32_t>(this->scoreLoop * sizeof(uint8_t)),
                static_cast<uint16_t>((CeilDiv(this->tilingData->inputParams.s2Size, isInfo.isM) - this->scoreLoop) *
                                      sizeof(uint8_t)),
                maskSrcStride, 0};
            AscendC::DataCopyPadExtParams<uint8_t> extParams = {true, 0, static_cast<uint8_t>(maskPad), 0};
            AscendC::DataCopyPad<uint8_t>(maskTensor, topKMaskGmInt[maskGmOffset + maskOffset], params, extParams);
        }

        // CalcImportanceScore
        AscendC::SetFlag<HardEvent::MTE2_V>(eventIdMte2ToVMask);
        AscendC::WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        AscendC::ConfusionTranspose<float>(trans, softmaxRes, sharedBuf, AscendC::TransposeType::TRANSPOSE_ND2ND_ONLY,
                                           transposeInfoForward);
        AscendC::PipeBarrier<PIPE_V>();

        int64_t lineOffset = innerLoop - isInfo.isM;
        if (loopIdx == 0) {
            lineOffset = 0;
        }
        if (isInfo.isN == 1) {
            lineOffset = lineOffset - 1;
        }
        int64_t vS1MulsGsize = this->vecS1BaseSize * this->gSize;
        if (loopIdx == this->s2Loop - 1) {
            int64_t dstOffset = (scoreLoop * isInfo.isM + lineOffset) * vS1MulsGsize;
            uint64_t unalignedLen =
                (s2Offset + scoreLoop * isInfo.isM + lineOffset - extraInfo.s2RealSize) * vS1MulsGsize;
            Duplicate(trans[dstOffset - unalignedLen], (float)0.0, unalignedLen + vS1MulsGsize);
            AscendC::PipeBarrier<PIPE_V>();
        }

        DataCopyParams dataCopyParamsTrans;
        dataCopyParamsTrans.blockCount = scoreLoop;
        dataCopyParamsTrans.blockLen = vS1MulsGsize * sizeof(float) / BLOCK_SIZE;
        dataCopyParamsTrans.srcStride = (isInfo.isM - 1) * vS1MulsGsize * sizeof(float) / BLOCK_SIZE;
        dataCopyParamsTrans.dstStride = 0;
        DataCopy(scoreRes, trans[(isInfo.isM + lineOffset) * vS1MulsGsize], dataCopyParamsTrans);

        int ubOffset = scoreLoop * vS1MulsGsize;
        uint8_t srcStride = isInfo.isM * vS1MulsGsize * sizeof(float) / BLOCK_SIZE;
        AscendC::WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        for (int i = 1; i < innerLoop; ++i) {
            int times = this->timesArray[i];
            int scoreIdx = 0;
            int64_t startOffset = ((scoreIdx + 1) * isInfo.isM + lineOffset) * vS1MulsGsize;
            int64_t transOffset = startOffset - (i * vS1MulsGsize);
            if (unlikely(transOffset < 0)) {
                scoreIdx = 1;
                startOffset = ((scoreIdx + 1) * isInfo.isM + lineOffset) * vS1MulsGsize;
                transOffset = startOffset - (i * vS1MulsGsize);
            }
            Muls(scoreRes[scoreIdx * vS1MulsGsize + ubOffset], trans[transOffset], static_cast<float>(times),
                 static_cast<int32_t>(vS1MulsGsize), scoreLoop - scoreIdx, {1, 1, 8, srcStride});
            AscendC::PipeBarrier<PIPE_V>();
            Add(scoreRes[scoreIdx * vS1MulsGsize], scoreRes[scoreIdx * vS1MulsGsize + ubOffset],
                scoreRes[scoreIdx * vS1MulsGsize], static_cast<int32_t>(vS1MulsGsize), scoreLoop - scoreIdx,
                {1, 1, 1, 8, 8, 8});
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ConfusionTranspose<float>(transBack, scoreRes, sharedBuf, AscendC::TransposeType::TRANSPOSE_ND2ND_ONLY,
                                           transposeInfoBackward);
        AscendC::SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        // reduce g [s1g, outerLoop(s2ScoreLoopLen)]
        if (this->gSize > 1) {
            AscendC::PipeBarrier<PIPE_V>();
            DataCopyParams dataCopyParamsReduce;
            dataCopyParamsReduce.blockCount = this->vecS1BaseSize;
            dataCopyParamsReduce.blockLen = s2Aligned64B * sizeof(float) / BLOCK_SIZE;
            dataCopyParamsReduce.srcStride = s2Aligned64B * sizeof(float) / BLOCK_SIZE * (this->gSize - 1);
            dataCopyParamsReduce.dstStride = 0;
            int addLoopOuter = (s2Aligned64B + MAX_ADD_NUM - 1) / MAX_ADD_NUM;
            for (int addIdex = 0; addIdex < addLoopOuter; addIdex++) {
                int CalcNum = addIdex == addLoopOuter - 1 ? s2Aligned64B - addIdex * MAX_ADD_NUM : MAX_ADD_NUM;
                for (int gIdx = 1; gIdx < this->gSize; gIdx *= 2) { //  this->gSize % 2 == 0
                    uint16_t sgBlock = static_cast<uint16_t>(s2Aligned64B * gIdx * 2 * sizeof(float) / BLOCK_SIZE);
                    if (sgBlock < 256) {
                        Add(transBack[addIdex * MAX_ADD_NUM], transBack[gIdx * s2Aligned64B + addIdex * MAX_ADD_NUM],
                            transBack[addIdex * MAX_ADD_NUM], CalcNum, this->vecS1BaseSize * (this->gSize / (gIdx * 2)),
                            {1, 1, 1, static_cast<uint8_t>(sgBlock), static_cast<uint8_t>(sgBlock),
                             static_cast<uint8_t>(sgBlock)});
                        AscendC::PipeBarrier<PIPE_V>();
                    } else {
                        for (int i = 0; i < this->vecS1BaseSize * (this->gSize / (gIdx * 2)); i++) {
                            Add(transBack[addIdex * MAX_ADD_NUM + i * s2Aligned64B * gIdx * 2],
                                transBack[gIdx * s2Aligned64B + addIdex * MAX_ADD_NUM + i * s2Aligned64B * gIdx * 2],
                                transBack[addIdex * MAX_ADD_NUM + i * s2Aligned64B * gIdx * 2], CalcNum);
                            AscendC::PipeBarrier<PIPE_V>();
                        }
                    }
                }
            }
            DataCopy(transBack, transBack, dataCopyParamsReduce);
        }
        AscendC::WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToVMask);
        AscendC::WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToVMask);
        if constexpr (hasTopkMask == true) {
            AscendC::SelectWithBytesMaskShapeInfo selectShapeInfo;
            selectShapeInfo.firstAxis = this->vecS1BaseSize;
            selectShapeInfo.srcLastAxis = s2Aligned64B;
            selectShapeInfo.maskLastAxis = maskLenAligned32B;
            AscendC::SelectWithBytesMask(transBack, transBack, this->positiveFloatScalar, maskTensor, tmpBuf,
                                         selectShapeInfo);
        }
        AscendC::SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        AscendC::WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

        // CopyOutImportanceScore
        uint64_t s2OutOffset = isInfo.outerLoop * loopIdx;
        uint32_t ImpScoreSrcStride = (this->s2Aligned64B - this->scoreLoop) / 8;
        uint32_t ImpScoreDstStride =
            static_cast<uint32_t>((CeilDiv(extraInfo.s2RealSize, isInfo.isM) - isInfo.outerLoop) * sizeof(float));
        if (loopIdx == this->s2Loop - 1) {
            ImpScoreSrcStride = static_cast<uint32_t>((this->s2Aligned64B - this->scoreLoop) / 8);
            ImpScoreDstStride = static_cast<uint32_t>(isInfo.outerLoop * (this->s2Loop - 1) * sizeof(float));
        }
        AscendC::DataCopyExtParams copyOutParams = {static_cast<uint16_t>(extraInfo.s1RealSize),
                                                    static_cast<uint32_t>(this->scoreLoop * sizeof(float)),
                                                    ImpScoreSrcStride, ImpScoreDstStride, 0};
        DataCopyPad(this->impScoreRes[extraInfo.taskIdMod2][(extraInfo.vecCoreOffset / this->gSize) *
                                                                CeilDiv(extraInfo.s2RealSize, isInfo.isM) +
                                                            s2OutOffset],
                    transBack, copyOutParams);

        AscendC::SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        AscendC::SetFlag<HardEvent::MTE3_V>(eventIdMte3ToVMask);
    }

    AscendC::WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    AscendC::WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    AscendC::WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToVMask);

    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventIdMte3ToV);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMte3);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventIdMte3ToVMask);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToVMask);
    return;
}

template <LayOutTypeEnum layOutType, bool hasAtten, bool hasTopkMask, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void NsaCompressAttentionS1s2Bn2gs1SameAB<layOutType, hasAtten, hasTopkMask, INPUT_T, T, bmm1Format,
                                                            mmPolicyType>::TopkCompute(SplitSameABExtraInfo &extraInfo)
{
    AscendC::TopKInfo topKInfoData;
    uint32_t blockLength = 32;
    uint32_t S1Base = extraInfo.s1RealSize;
    uint32_t S2sizeTopK = CeilDiv(extraInfo.s2RealSize, this->tilingData->importanceScoreParams.isM);
    uint32_t k = this->tilingData->inputParams.selectBlockCount;
    uint32_t k_pad = (k + 8 - 1) / 8 * 8; // float数据类型对齐，暂时不考虑其他
    uint32_t S2sizeAlign32 = (S2sizeTopK + blockLength - 1) / blockLength * blockLength;
    uint32_t S2sizeAlign8 = (S2sizeTopK + 8 - 1) / 8 * 8;
    uint32_t tmpUbSize = this->tilingData->importanceScoreParams.tmpUbSize;
    uint32_t maxTopkBase = (190 * 1024 - tmpUbSize) / (4 * S2sizeAlign32 + 8 * k_pad);
    maxTopkBase = maxTopkBase > 0 ? maxTopkBase : 1;
    uint32_t topkBase = maxTopkBase > S1Base ? S1Base : maxTopkBase;
    AscendC::DataCopyExtParams copyInParamsV1, copyInParamsPad;
    AscendC::DataCopyPadExtParams<float> padParamsFloatV1{false, 0, 0, 0};
    AscendC::DataCopyPadExtParams<float> padParamsFloat{true, 0, static_cast<uint8_t>(S2sizeAlign8 - S2sizeTopK), 0};

    topKInfoData.inner = S2sizeAlign32;
    topKInfoData.n = S2sizeTopK;
    uint32_t topkLoopNum = (S1Base + topkBase - 1) / topkBase;
    uint32_t topkBaseTail = S1Base - (topkLoopNum - 1) * topkBase;

    int64_t addrOffset = 0;
    sorceResultLocal = allUbBuffer.GetWithOffset<T>(topkBase * S2sizeAlign32, addrOffset);
    addrOffset = addrOffset + topkBase * S2sizeAlign32 * sizeof(float);
    topKIdexLocal = allUbBuffer.GetWithOffset<int32_t>(topkBase * k_pad, addrOffset);
    addrOffset = addrOffset + topkBase * k_pad * sizeof(int32_t);
    topKValueLocal = allUbBuffer.GetWithOffset<T>(topkBase * k_pad, addrOffset);
    addrOffset = addrOffset + topkBase * k_pad * sizeof(float);
    topKTmpLocal = allUbBuffer.GetWithOffset<uint8_t>(tmpUbSize, addrOffset);

    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s2Offset = 0;
    int64_t s1gOffset = 0;

    if constexpr (layOutType == LayOutTypeEnum::LAYOUT_TND) {
        uint64_t n2OffsetU64 = static_cast<uint64_t>(extraInfo.n2oIdx) * static_cast<uint64_t>(this->s1TotalSize) *
                               static_cast<uint64_t>(k);
        uint64_t bOffsetU64 = static_cast<uint64_t>(extraInfo.s1SizeAcc) * static_cast<uint64_t>(k);
        uint64_t s1Base =
            static_cast<uint64_t>(extraInfo.s1oIdx) * static_cast<uint64_t>(this->tilingData->coreParams.s1BaseSize);
        uint64_t cubeOffset =
            static_cast<uint64_t>(this->cubeSubIdx) * ((static_cast<uint64_t>(extraInfo.cubeS1RealSize) + 1) / 2);
        uint64_t s1gOffsetU64 = (s1Base + cubeOffset) * static_cast<uint64_t>(k);
        n2Offset = static_cast<int64_t>(n2OffsetU64);
        bOffset = static_cast<int64_t>(bOffsetU64);
        s1gOffset = static_cast<int64_t>(s1gOffsetU64);
    }
    int64_t TopkOutCoreOffset = n2Offset + bOffset + s1gOffset;

    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
    event_t eventIdMte3ToV1 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
    event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());

    AscendC::SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    AscendC::SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);

    AscendC::PipeBarrier<PIPE_ALL>();
    for (uint64_t taskLoop = 0; taskLoop < topkLoopNum; taskLoop++) {
        uint64_t topkOutNum = taskLoop == topkLoopNum - 1 ? topkBaseTail : topkBase;
        copyInParamsV1 = {1, (uint32_t)(topkOutNum * S2sizeTopK * sizeof(float)), 0, 0, 0};
        copyInParamsPad = {(uint16_t)topkOutNum, (uint32_t)(S2sizeTopK * sizeof(float)), 0,
                           (S2sizeAlign32 - S2sizeAlign8) / 8, 0};

        topKInfoData.outter = topkOutNum;
        if (taskLoop == 0 || taskLoop == topkLoopNum - 1) {
            TopkTilingCompute(this->topktiling, topKInfoData.inner, topKInfoData.outter);
        }

        AscendC::PipeBarrier<PIPE_V>();
        uint32_t dstStride = S2sizeAlign32 / 8;
        if (dstStride < 256) {
            Duplicate(sorceResultLocal[S2sizeAlign32 - blockLength], (T)-1, blockLength, topkBase, 1, dstStride);
        } else {
            for (int i = 0; i < topkBase; i++) {
                Duplicate(sorceResultLocal[S2sizeAlign32 * i + S2sizeAlign32 - blockLength], (T)-1, blockLength);
            }
        }

        AscendC::SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        AscendC::WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        if (S2sizeTopK % 32 == 0) {
            DataCopyPad(
                sorceResultLocal,
                this->impScoreRes[extraInfo.taskIdMod2]
                                 [(extraInfo.vecCoreOffset / this->gSize) *
                                      CeilDiv(extraInfo.s2RealSize, this->tilingData->importanceScoreParams.isM) +
                                  taskLoop * topkBase * S2sizeTopK],
                copyInParamsV1, padParamsFloatV1);
        } else {
            DataCopyPad(
                sorceResultLocal,
                this->impScoreRes[extraInfo.taskIdMod2]
                                 [(extraInfo.vecCoreOffset / this->gSize) *
                                      CeilDiv(extraInfo.s2RealSize, this->tilingData->importanceScoreParams.isM) +
                                  taskLoop * topkBase * S2sizeTopK],
                copyInParamsPad, padParamsFloat);
        }

        AscendC::SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        AscendC::WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        AscendC::WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);

        AscendC::TopK<float, false, false, false, AscendC::TopKMode::TOPK_NORMAL>(
            topKValueLocal, topKIdexLocal, sorceResultLocal, srcLocalIndex, srcLocalFinish, topKTmpLocal, k,
            this->topktiling, topKInfoData, true);

        AscendC::SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        AscendC::WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

        if (k % 8 == 0) {
            DataCopyPad(topkIndicesOutGm[TopkOutCoreOffset + taskLoop * topkBase * k], topKIdexLocal,
                        {(uint16_t)1, (uint32_t)(topkOutNum * k * sizeof(int32_t)), 0, 0, 0});
        } else {
            DataCopyPad(topkIndicesOutGm[TopkOutCoreOffset + taskLoop * topkBase * k], topKIdexLocal,
                        {(uint16_t)topkOutNum, (uint32_t)(k * sizeof(int32_t)), 0, 0, 0});
        }
        AscendC::SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    }
    AscendC::WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    AscendC::WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);

    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventIdMte3ToV);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventIdMte3ToV1);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMte3);
}

template <LayOutTypeEnum layOutType, bool hasAtten, bool hasTopkMask, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void
NsaCompressAttentionS1s2Bn2gs1SameAB<layOutType, hasAtten, hasTopkMask, INPUT_T, T, bmm1Format,
                                     mmPolicyType>::TransposeTilingCompute(ConfusionTransposeTiling &tiling,
                                                                           uint64_t row, uint64_t col)
{
    uint32_t blockSizeT = 8;
    uint32_t rowBlock = row / 16;
    uint32_t stride = row * blockSizeT * 4 / 32;
    uint32_t repeat = col / blockSizeT;

    tiling.param0 = blockSizeT;
    tiling.param1 = row;
    tiling.param2 = col;
    tiling.param3 = rowBlock;
    tiling.param4 = stride;
    tiling.param5 = repeat;
}

template <LayOutTypeEnum layOutType, bool hasAtten, bool hasTopkMask, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void NsaCompressAttentionS1s2Bn2gs1SameAB<layOutType, hasAtten, hasTopkMask, INPUT_T, T, bmm1Format,
                                                            mmPolicyType>::TopkTilingCompute(TopkTiling &tiling,
                                                                                             uint64_t inner,
                                                                                             uint64_t outter)
{
    tiling.tmpLocalSize = 4 * inner;
    tiling.innerDataSize = 2 * inner;
    tiling.copyUbToUbBlockCount = static_cast<uint32_t>(inner * 2 * 4 / 32);
    tiling.sortRepeat = static_cast<uint32_t>(inner / 32);
    tiling.mrgSortRepeat = inner / 4;

    const int32_t kAlignFourBytesTmp = ((this->tilingData->inputParams.selectBlockCount + 7) / 8) * 8;
    tiling.maskOffset = outter * kAlignFourBytesTmp;
    tiling.mrgFourQueueTailPara1 = inner * 2;

    tiling.tmpLocalSize = 5 * inner;
    tiling.srcIndexOffset = 4 * inner;
}

template <LayOutTypeEnum layOutType, bool hasAtten, bool hasTopkMask, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void
NsaCompressAttentionS1s2Bn2gs1SameAB<layOutType, hasAtten, hasTopkMask, INPUT_T, T, bmm1Format,
                                     mmPolicyType>::GetBmm1Result(SplitSameABExtraInfo &extraInfo,
                                                                  LocalTensor<T> &bmm1ResUb, int64_t s1LoopIdx,
                                                                  int64_t gLoopIdx)
{
    if (likely(extraInfo.s2RealSizeAlign32 == extraInfo.s2RealSize)) {
        DataCopy2D(bmm1ResUb,
                   this->mm1Res[extraInfo.taskIdMod2]
                               [(extraInfo.vecCoreOffset + s1LoopIdx * this->gSize + gLoopIdx * extraInfo.vec1MaxG) *
                                extraInfo.s2RealSize],
                   extraInfo.vec1S1RealSize, extraInfo.s2RealSize, extraInfo.s2RealSize);
    } else {
        DataCopyParams dataCopyParams;
        DataCopyPadParams dataCopyPadParams;
        dataCopyParams.blockCount = extraInfo.vec1S1RealSize;
        dataCopyParams.blockLen = extraInfo.s2RealSize * sizeof(T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        dataCopyPadParams.isPad = true;
        dataCopyPadParams.rightPadding = extraInfo.s2AlignedSize - extraInfo.s2RealSize;
        if (dataCopyPadParams.rightPadding > blockSize) {
            dataCopyPadParams.rightPadding -= blockSize;
            dataCopyParams.dstStride = 1;
            int32_t s2BlockAlignedSize = CeilDiv(extraInfo.s2RealSize, blockSize) * blockSize;
            auto dstRepeatStride = extraInfo.s2AlignedSize * sizeof(T) / blockBytes;
            if (dstRepeatStride < 256) {
                Duplicate<T>(bmm1ResUb[s2BlockAlignedSize], 0, blockSize, extraInfo.vec1S1RealSize, 0, dstRepeatStride);
            } else {
                for (int i = 0; i < extraInfo.vec1S1RealSize; ++i) {
                    Duplicate<T>(bmm1ResUb[s2BlockAlignedSize + i * extraInfo.s2AlignedSize], 0, blockSize);
                }
            }
        }
        dataCopyPadParams.paddingValue = 0;
        DataCopyPad(bmm1ResUb,
                    this->mm1Res[extraInfo.taskIdMod2]
                                [(extraInfo.vecCoreOffset + s1LoopIdx * this->gSize + gLoopIdx * extraInfo.vec1MaxG) *
                                 extraInfo.s2RealSize],
                    dataCopyParams, dataCopyPadParams);
    }
    uint32_t bmm1ResUbShape[] = {static_cast<uint32_t>(extraInfo.vec1S1RealSize),
                                 static_cast<uint32_t>(extraInfo.s2RealSizeAlign32)};
    bmm1ResUb.SetShapeInfo(ShapeInfo(2, bmm1ResUbShape, DataFormat::ND));
}

template <LayOutTypeEnum layOutType, bool hasAtten, bool hasTopkMask, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void
NsaCompressAttentionS1s2Bn2gs1SameAB<layOutType, hasAtten, hasTopkMask, INPUT_T, T, bmm1Format,
                                     mmPolicyType>::CopyInAttenMask(SplitSameABExtraInfo &extraInfo,
                                                                    LocalTensor<uint8_t> &attenMaskUb,
                                                                    int64_t s1LoopIdx)
{
    if constexpr (hasAtten == true) {
        auto attenMaskS2Align = CeilDiv(extraInfo.s2RealSize, 32) * 32;
        int64_t maskOffset =
            (extraInfo.s1oIdx * this->cubeS1BaseSize + extraInfo.vecCoreOffset / this->gSize + s1LoopIdx) *
            this->tilingData->inputParams
                .s2Size; // attenMaskGm offset: [cubeS1Offset + vecS1Offset + s1LoopIdx] * maxS2
        if (likely(attenMaskS2Align == extraInfo.s2RealSize)) {
            for (int i = 0; i < extraInfo.vec1S1RealSize; ++i) {
                DataCopy(attenMaskUb[i * attenMaskS2Align], this->attenMaskGmInt[maskOffset],
                         extraInfo.attenMaskS2Size);
            }
        } else {
            AscendC::DataCopyExtParams dataCopyParams = {1, (uint32_t)extraInfo.attenMaskS2Size, 0, 0, 0};
            AscendC::DataCopyPadExtParams<uint8_t> dataCopyPadParams = {
                true, 0, (uint8_t)(attenMaskS2Align - extraInfo.attenMaskS2Size), 1};
            for (int i = 0; i < extraInfo.vec1S1RealSize; ++i) {
                DataCopyPad(attenMaskUb[i * attenMaskS2Align], this->attenMaskGmInt[maskOffset], dataCopyParams,
                            dataCopyPadParams);
            }
        }
    }
}

template <LayOutTypeEnum layOutType, bool hasAtten, bool hasTopkMask, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void
NsaCompressAttentionS1s2Bn2gs1SameAB<layOutType, hasAtten, hasTopkMask, INPUT_T, T, bmm1Format,
                                     mmPolicyType>::WaitBmm2Result(SplitSameABExtraInfo &extraInfo)
{
    this->bmm2.WaitIterateAll();
    this->bmm2.End();
}

template <LayOutTypeEnum layOutType, bool hasAtten, bool hasTopkMask, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void NsaCompressAttentionS1s2Bn2gs1SameAB<
    layOutType, hasAtten, hasTopkMask, INPUT_T, T, bmm1Format,
    mmPolicyType>::IterateBmm2(SplitSameABExtraInfo &extraInfo,
                               matmul::Matmul<a2Type, b2Type, c2NzType, bias2Type, CFG_EXCEED,
                                              matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>,
                                              MatmulPolicySelector<mmPolicyType>::template Result> &bmm2)
{
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s2Offset = 0;
    int64_t s1gOffset = 0;

    if constexpr (layOutType == LayOutTypeEnum::LAYOUT_TND) {
        // b,s2,n2,d
        uint64_t bOffsetU64 = static_cast<uint64_t>(extraInfo.s2SizeAcc) *
                              static_cast<uint64_t>(this->tilingData->inputParams.n2Size) *
                              static_cast<uint64_t>(this->d2Size);
        uint64_t n2OffsetU64 = static_cast<uint64_t>(extraInfo.n2oIdx) * static_cast<uint64_t>(this->d2Size);
        bOffset = static_cast<int64_t>(bOffsetU64);
        n2Offset = static_cast<int64_t>(n2OffsetU64);
    }

    int64_t vCoreOffset = n2Offset + bOffset + s2Offset;
    bmm2.SetOrgShape(extraInfo.cubeS1gRealSize, this->tilingData->inputParams.n2Size * this->d2Size,
                     extraInfo.s2AlignedSize, extraInfo.s2AlignedSize, this->d2Size); // b,s2,n2,d

    bmm2.SetTensorA(this->stage1Res);
    bmm2.SetTensorB(this->valueGm[vCoreOffset]);

    if constexpr (layOutType == LayOutTypeEnum::LAYOUT_TND) {
        uint64_t n2OffsetU64 = static_cast<uint64_t>(extraInfo.n2oIdx) * static_cast<uint64_t>(this->s1TotalSize);
        n2OffsetU64 = n2OffsetU64 * static_cast<uint64_t>(this->gSize) * static_cast<uint64_t>(this->d2Size);
        uint64_t bOffsetU64 = static_cast<uint64_t>(extraInfo.s1SizeAcc) * static_cast<uint64_t>(this->gSize) *
                              static_cast<uint64_t>(this->d2Size);
        uint64_t s1gOffsetU64 =
            static_cast<uint64_t>(extraInfo.s1oIdx) * static_cast<uint64_t>(this->tilingData->coreParams.s1BaseSize);
        s1gOffsetU64 = s1gOffsetU64 * static_cast<uint64_t>(this->gSize) * static_cast<uint64_t>(this->d2Size);
        n2Offset = static_cast<int64_t>(n2OffsetU64);
        bOffset = static_cast<int64_t>(bOffsetU64);
        s1gOffset = static_cast<int64_t>(s1gOffsetU64);
    }
    int64_t outCoreOffset = n2Offset + bOffset + s1gOffset;

    bmm2.SetTail(extraInfo.cubeS1gRealSize, this->d2Size, extraInfo.s2RealSize);
    bmm2.template IterateAll<false>(this->attentionOutGm[outCoreOffset], false, false, false);
}

#endif // NSA_COMPRESS_ATTENTION_S1S2_BN2GS1_SAB_H
