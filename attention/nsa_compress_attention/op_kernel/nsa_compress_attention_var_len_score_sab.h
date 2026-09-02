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
 * \file nsa_compress_attention_var_len_score_sab.h
 * \brief
 */

#ifndef NSA_COMPRESS_ATTENTION_VAR_LEN_SCORE_SAB_H
#define NSA_COMPRESS_ATTENTION_VAR_LEN_SCORE_SAB_H

#include "nsa_compress_attention_s1s2_bn2gs1_sab.h"

// INPUT_T - means data type for input
// T       - means data type when calc
template <LayOutTypeEnum layOutType, bool hasAtten, bool hasTopkMask, typename INPUT_T, typename T = INPUT_T,
          CubeFormat bmm1Format = CubeFormat::ND, MmPolicyType mmPolicyType = MmPolicyType::NORMAL>
class NsaCompressAttentionVarLenScoreSameAB
    : public NsaCompressAttentionS1s2Bn2gs1SameAB<layOutType, hasAtten, hasTopkMask, INPUT_T, T, bmm1Format,
                                                  mmPolicyType> {
public:
    __aicore__ inline NsaCompressAttentionVarLenScoreSameAB(){};

    __aicore__ inline void UnpackInit(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                      __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths,
                                      __gm__ uint8_t *actualSeqLengthsCmpKv, __gm__ uint8_t *actualSeqLengthsSelKv,
                                      __gm__ uint8_t *topkMask, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
                                      __gm__ uint8_t *attentionOut, __gm__ uint8_t *topkIndicesOut,
                                      __gm__ uint8_t *workspace,
                                      const NsaCompressAttentionGeneralTilingData *__restrict tiling, TPipe *tPipe);

    __aicore__ inline void Process();

protected:
    __aicore__ inline void ComputeConstexpr();

    __aicore__ inline void ComputeAxisIdx(int64_t multiCoreInnerIdx);

    __aicore__ inline void ComputeBmm1Tail(SplitSameABExtraInfo &extraInfo);

    __aicore__ inline void SetExtraInfo(SplitSameABExtraInfo &extraInfo, int64_t taskId, int64_t multiCoreInnerIdx);

    __aicore__ inline void CalS1OuterSize(int64_t offset);

    __aicore__ inline void GetSeqQlenKvlenByBoidx(int64_t boIdx, int64_t &actualSeqQlen, int64_t &actualSeqKvlen);

    // Unpack 用参数
    GM_ADDR actualSeqQlenAddr;
    GM_ADDR actualSeqKvlenAddr;
    GM_ADDR prefixNAddr;
    uint64_t n2bs1OuterSizeAcc;
    uint64_t s1SizeAcc;
    uint64_t s2SizeAcc;
    GlobalTensor<int64_t> qListGm;
    GlobalTensor<int64_t> kvListGm;
};

template <LayOutTypeEnum layOutType, bool hasAtten, bool hasTopkMask, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void NsaCompressAttentionVarLenScoreSameAB<
    layOutType, hasAtten, hasTopkMask, INPUT_T, T, bmm1Format,
    mmPolicyType>::UnpackInit(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                              __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths,
                              __gm__ uint8_t *actualSeqLengthsCmpKv, __gm__ uint8_t *actualSeqLengthsSelKv,
                              __gm__ uint8_t *topkMask, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
                              __gm__ uint8_t *attentionOut, __gm__ uint8_t *topkIndicesOut, __gm__ uint8_t *workspace,
                              const NsaCompressAttentionGeneralTilingData *__restrict tiling, TPipe *tPipe)
{
    this->InitInput(query, key, value, attenMask, topkMask, softmaxMax, softmaxSum, attentionOut, topkIndicesOut,
                    workspace, tiling, tPipe); // gm设置
    this->ComputeConstexpr();
    qListGm.SetGlobalBuffer((__gm__ int64_t *)actualSeqLengths);
    kvListGm.SetGlobalBuffer((__gm__ int64_t *)actualSeqLengthsCmpKv);
    // 初始化unpack
    actualSeqQlenAddr = actualSeqLengths;
    actualSeqKvlenAddr = actualSeqLengthsCmpKv;
    int64_t actualS1Len;
    int64_t actualS2Len;
    this->InitBuffer();

    this->s2SizeSum = ((__gm__ uint64_t *)actualSeqLengthsCmpKv)[this->tilingData->inputParams.bSize - 1];
    return;
}

template <LayOutTypeEnum layOutType, bool hasAtten, bool hasTopkMask, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void NsaCompressAttentionVarLenScoreSameAB<layOutType, hasAtten, hasTopkMask, INPUT_T, T, bmm1Format,
                                                             mmPolicyType>::ComputeConstexpr()
{
    this->gSize = this->tilingData->inputParams.gSize;
    this->n2G = this->tilingData->inputParams.n2Size * this->gSize;
    this->s1Size = this->tilingData->inputParams.s1Size;
}

template <LayOutTypeEnum layOutType, bool hasAtten, bool hasTopkMask, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void
NsaCompressAttentionVarLenScoreSameAB<layOutType, hasAtten, hasTopkMask, INPUT_T, T, bmm1Format,
                                      mmPolicyType>::GetSeqQlenKvlenByBoidx(int64_t boIdx, int64_t &actualSeqQlen,
                                                                            int64_t &actualSeqKvlen)
{
    if (boIdx == 0) {
        actualSeqQlen = qListGm.GetValue(0);
        actualSeqKvlen = kvListGm.GetValue(0);
    } else {
        actualSeqQlen = qListGm.GetValue(boIdx) - qListGm.GetValue(boIdx - 1);
        actualSeqKvlen = kvListGm.GetValue(boIdx) - kvListGm.GetValue(boIdx - 1);
    }
    return;
}

// 初始化s1方向上的累加值
template <LayOutTypeEnum layOutType, bool hasAtten, bool hasTopkMask, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void NsaCompressAttentionVarLenScoreSameAB<layOutType, hasAtten, hasTopkMask, INPUT_T, T, bmm1Format,
                                                             mmPolicyType>::CalS1OuterSize(int64_t offset)
{
    int64_t actualN2BS1Outersize = 0;
    // 用于取actualS1Len下标
    int64_t actualS1Len;
    int64_t actualS2Len;
    this->n2oIdx = 0;
    this->n2bs1OuterSizeAcc = 0;

    for (int64_t n2 = 0; n2 < this->tilingData->inputParams.n2Size; n2++) {
        this->boIdx = 0;
        this->s1SizeAcc = 0;
        this->s2SizeAcc = 0;
        for (int64_t i = 0; i < this->tilingData->inputParams.bSize; i++) {
            GetSeqQlenKvlenByBoidx(i, actualS1Len, actualS2Len);
            actualN2BS1Outersize += CeilDiv(actualS1Len, this->cubeS1BaseSize); // idx反算回去
            if (offset >= actualN2BS1Outersize) {
                this->n2bs1OuterSizeAcc = actualN2BS1Outersize;
                this->s1SizeAcc += actualS1Len;
                this->s2SizeAcc += actualS2Len;
                this->boIdx++;
            } else {
                return;
            }
        }
        this->n2oIdx++;
        this->s1TotalSize = this->s1SizeAcc;
        this->s2TotalSize = this->s2SizeAcc;
    }
    return;
}

template <LayOutTypeEnum layOutType, bool hasAtten, bool hasTopkMask, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void NsaCompressAttentionVarLenScoreSameAB<layOutType, hasAtten, hasTopkMask, INPUT_T, T, bmm1Format,
                                                             mmPolicyType>::ComputeAxisIdx(int64_t multiCoreInnerIdx)
{
    int64_t actualS1Len;
    int64_t actualS2Len;
    GetSeqQlenKvlenByBoidx(this->boIdx, actualS1Len, actualS2Len);
    int64_t actualN2BS1Outersize = this->n2bs1OuterSizeAcc + CeilDiv(actualS1Len, this->cubeS1BaseSize);
    while (multiCoreInnerIdx >= actualN2BS1Outersize) {
        this->n2bs1OuterSizeAcc = actualN2BS1Outersize;
        this->s1SizeAcc += actualS1Len;
        this->s2SizeAcc += actualS2Len;
        this->boIdx++;
        this->s1oIdx += CeilDiv(actualS1Len, this->cubeS1BaseSize);
        if (this->boIdx == this->tilingData->inputParams.bSize) {
            this->s1TotalSize = this->s1SizeAcc;
            this->s2TotalSize = this->s2SizeAcc;
            this->boIdx = 0;
            this->s1SizeAcc = 0;
            this->s2SizeAcc = 0;
            this->n2oIdx++;
        }
        GetSeqQlenKvlenByBoidx(this->boIdx, actualS1Len, actualS2Len);
        actualN2BS1Outersize = this->n2bs1OuterSizeAcc + CeilDiv(actualS1Len, this->cubeS1BaseSize);
    }
    // 计算轴的idx
    int64_t tmpS1Outersize = CeilDiv(actualS1Len, this->cubeS1BaseSize);
    actualN2BS1Outersize = multiCoreInnerIdx - this->n2bs1OuterSizeAcc;
    this->s1oIdx = actualN2BS1Outersize % tmpS1Outersize;
    GetSeqQlenKvlenByBoidx(this->boIdx, this->s1Size, this->s2Size);
}

template <LayOutTypeEnum layOutType, bool hasAtten, bool hasTopkMask, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void NsaCompressAttentionVarLenScoreSameAB<layOutType, hasAtten, hasTopkMask, INPUT_T, T, bmm1Format,
                                                             mmPolicyType>::Process()
{
    // 确定核内切分起点
    int64_t multiCoreInnerOffset = this->cubeBlockIdx * this->tilingData->multiCoreParams.splitFactorSize;
    int64_t multiCoreInnerLimit = multiCoreInnerOffset + this->tilingData->multiCoreParams.splitFactorSize;
    if (this->tilingData->multiCoreParams.totalSize < multiCoreInnerLimit) {
        multiCoreInnerLimit = this->tilingData->multiCoreParams.totalSize;
    }

    // 初始化AxisIdx
    this->CalS1OuterSize(multiCoreInnerOffset);

    SplitSameABExtraInfo extraInfo[2];
    int64_t taskId = 0;
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    for (int64_t multiCoreInnerIdx = multiCoreInnerOffset; multiCoreInnerIdx < multiCoreInnerLimit;
         ++multiCoreInnerIdx) {
        if (taskId == 0) {
            this->ComputeAxisIdx(multiCoreInnerIdx);
            this->SetExtraInfo(extraInfo[taskId % 2], taskId, multiCoreInnerIdx);
            this->IterateBmm1(extraInfo[taskId % 2], this->bmm1);
        }

        this->WaitBmm1Result(extraInfo[taskId % 2]);

        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        this->ProcessVec1(extraInfo[taskId % 2]);
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);

        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        this->IterateBmm2(extraInfo[taskId % 2], this->bmm2);

        if (multiCoreInnerIdx != multiCoreInnerLimit - 1) {
            this->ComputeAxisIdx(multiCoreInnerIdx + 1);
            this->SetExtraInfo(extraInfo[(taskId + 1) % 2], taskId + 1, multiCoreInnerIdx);
            this->IterateBmm1(extraInfo[(taskId + 1) % 2], this->bmm1);
        }
        this->ProcessVec2(extraInfo[taskId % 2]); // ImpScore
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);

        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        this->TopkCompute(extraInfo[taskId % 2]); // TopK
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);

        taskId++;
    }
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
};

template <LayOutTypeEnum layOutType, bool hasAtten, bool hasTopkMask, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void
NsaCompressAttentionVarLenScoreSameAB<layOutType, hasAtten, hasTopkMask, INPUT_T, T, bmm1Format,
                                      mmPolicyType>::SetExtraInfo(SplitSameABExtraInfo &extraInfo, int64_t taskId,
                                                                  int64_t multiCoreInnerIdx)
{
    extraInfo.s1oIdx = this->s1oIdx;
    extraInfo.boIdx = this->boIdx;
    extraInfo.n2oIdx = this->n2oIdx;
    extraInfo.taskId = taskId;
    extraInfo.taskIdMod2 = taskId % 2;
    extraInfo.multiCoreInnerIdx = multiCoreInnerIdx;
    extraInfo.s1Size = this->tilingData->inputParams.s1Size;
    extraInfo.s2Size = this->tilingData->inputParams.s2Size;
    extraInfo.s1SizeAcc = s1SizeAcc;
    extraInfo.s2SizeAcc = s2SizeAcc;

    GetSeqQlenKvlenByBoidx(extraInfo.boIdx, extraInfo.s1Size, extraInfo.s2Size);
    extraInfo.attenMaskS2Size = extraInfo.s2Size; // s2RealSize

    // s1Base * g
    extraInfo.cubeS1RealSize = Min(this->cubeS1BaseSize, extraInfo.s1Size - extraInfo.s1oIdx * this->cubeS1BaseSize);
    extraInfo.cubeS1gRealSize = extraInfo.cubeS1RealSize * this->gSize;
    this->ComputeBmm1Tail(extraInfo);
}

template <LayOutTypeEnum layOutType, bool hasAtten, bool hasTopkMask, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void
NsaCompressAttentionVarLenScoreSameAB<layOutType, hasAtten, hasTopkMask, INPUT_T, T, bmm1Format,
                                      mmPolicyType>::ComputeBmm1Tail(SplitSameABExtraInfo &extraInfo)
{
    extraInfo.s1RealSize = (extraInfo.cubeS1RealSize + 1) / 2; // s1Base / 2
    extraInfo.vecCoreOffset = this->cubeSubIdx * extraInfo.s1RealSize * this->gSize;
    auto gIndex = extraInfo.vecCoreOffset % this->gSize;
    auto s1Index = extraInfo.vecCoreOffset / this->gSize;
    int64_t softmaxMaxVecOffset = gIndex * this->s1Size * 8 + s1Index * 8;

    if (this->cubeSubIdx == 1) {
        extraInfo.s1RealSize = extraInfo.cubeS1RealSize - extraInfo.s1RealSize; // s1外循环
    }
    extraInfo.s2RealSize = extraInfo.s2Size;
    extraInfo.s2AlignedSize = CeilDiv(extraInfo.s2RealSize, 32 / sizeof(INPUT_T)) * (32 / sizeof(INPUT_T));

    extraInfo.s2RealSizeAlign32 = extraInfo.s2AlignedSize;

    extraInfo.vec1MaxG = this->vecMaxG;
    extraInfo.vec1LoopCountG = CeilDiv(this->gSize, extraInfo.vec1MaxG); // g内循环

    // [b, n2, g, s1, 8]
    // [n2, g, s1[0], 8] + [n2, g, s1[1], 8] + [n2, g, s1[2], 8] ...
    extraInfo.softmaxMaxOffset = this->s1SizeAcc * this->n2G * 8 + this->n2oIdx * this->gSize * this->s1Size * 8 +
                                 this->s1oIdx * this->cubeS1BaseSize * 8 + softmaxMaxVecOffset;
    return;
}

#endif // NSA_COMPRESS_ATTENTION_VAR_LEN_SCORE_SAB_H
