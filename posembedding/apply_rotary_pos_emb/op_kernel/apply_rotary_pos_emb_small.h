/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file apply_rotary_pos_emb_small.h
 * \brief
 */
#ifndef APPLY_ROTARY_POS_EMB_SMALL_H
#define APPLY_ROTARY_POS_EMB_SMALL_H

#include "kernel_operator.h"
#include "apply_rotary_pos_emb_base.h"

namespace ApplyRotaryPosEmb {
using namespace AscendC;

template <typename T1, typename T2>
class ARPESmall : public ApplyRotaryPosEmbBase<T1> {
public:
    __aicore__ inline ARPESmall(){};
    __aicore__ inline void Init(
        GM_ADDR q, GM_ADDR k, GM_ADDR cos, GM_ADDR sin, GM_ADDR qOut, GM_ADDR kOut, GM_ADDR workspace,
        const ApplyRotaryPosEmbTilingData* tilingData);
    __aicore__ inline void Process(const ApplyRotaryPosEmbTilingData* tilingData);

private:
    __aicore__ inline void ProcessPerCore(const ApplyRotaryPosEmbTilingData* tilingData);
    __aicore__ inline void CastCopyIn(const ApplyRotaryPosEmbTilingData* tilingData);
    __aicore__ inline void CopyInQK(const ApplyRotaryPosEmbTilingData* tilingData);
    __aicore__ inline void ComputeTotary(
        LocalTensor<T1>& qSize, LocalTensor<T1>& cosSize, LocalTensor<T1>& sinSize, LocalTensor<T1>& qoutSize,
        const ApplyRotaryPosEmbTilingData* tilingData);
    __aicore__ inline void ComputeBF16(
        LocalTensor<T1>& qSize, LocalTensor<T1>& cosSize, LocalTensor<T1>& sinSize, LocalTensor<T1>& qoutSize,
        const ApplyRotaryPosEmbTilingData* tilingData);

    constexpr static int32_t bufferNum = 1;

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, bufferNum> qInQueue;
    TQue<QuePosition::VECOUT, bufferNum> qOutQueue;
    TQue<QuePosition::VECIN, bufferNum> cosInQueue;
    TQue<QuePosition::VECIN, bufferNum> sinInQueue;
    TBuf<QuePosition::VECCALC> mul1;
    TBuf<QuePosition::VECCALC> mul2;
    TBuf<QuePosition::VECCALC> cosCast;
    TBuf<QuePosition::VECCALC> sinCast;

    GlobalTensor<T1> qGm;
    GlobalTensor<T1> kGm;
    GlobalTensor<T1> cosGm;
    GlobalTensor<T1> sinGm;
    uint64_t blockIdx = 0;
    DataCopyParams copyIn1;
    BinaryRepeatParams repeatParams;
    UnaryRepeatParams mulRepeatP = {1, 1, 0, 0};
};

template <typename T1, typename T2>
__aicore__ inline void ARPESmall<T1, T2>::Init(
    GM_ADDR q, GM_ADDR k, GM_ADDR cos, GM_ADDR sin, GM_ADDR qOut, GM_ADDR kOut, GM_ADDR workspace,
    const ApplyRotaryPosEmbTilingData* tilingData)
{
    blockIdx = GetBlockIdx();
    qGm.SetGlobalBuffer((__gm__ T1*)q);
    kGm.SetGlobalBuffer((__gm__ T1*)k);
    cosGm.SetGlobalBuffer((__gm__ T1*)cos);
    sinGm.SetGlobalBuffer((__gm__ T1*)sin);
    pipe.InitBuffer(cosInQueue, bufferNum, tilingData->cosPart1Ub);
    pipe.InitBuffer(mul1, tilingData->q2q1Part1Ub);
    pipe.InitBuffer(sinInQueue, bufferNum, tilingData->cosPart1Ub);
    pipe.InitBuffer(mul2, tilingData->q2q1Part1Ub);
    pipe.InitBuffer(qInQueue, bufferNum, tilingData->qPart1Ub);
    pipe.InitBuffer(qOutQueue, bufferNum, tilingData->qPart1Ub);

    copyIn1.blockCount = tilingData->qkcNum;
    copyIn1.blockLen = tilingData->blockLenQ;
    copyIn1.srcStride = tilingData->blockLenQ;
    copyIn1.dstStride = tilingData->blockLenQ;

    repeatParams.dstBlkStride = 1;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1BlkStride = 1;
    repeatParams.dstRepStride = tilingData->dstRepSBr;
    repeatParams.src0RepStride = tilingData->dstRepSBr;
    repeatParams.src1RepStride = 0;

#if ORIG_DTYPE_QUERY == DT_BF16
    pipe.InitBuffer(cosCast, tilingData->sin1UbSize);
    pipe.InitBuffer(sinCast, tilingData->sin1UbSize);
#endif
}

template <typename T1, typename T2>
__aicore__ inline void ARPESmall<T1, T2>::CopyInQK(const ApplyRotaryPosEmbTilingData* tilingData)
{
    LocalTensor<T1> qUb = qInQueue.AllocTensor<T1>();
    LocalTensor<T1> cosUb = cosInQueue.AllocTensor<T1>();
    LocalTensor<T1> sinUb = sinInQueue.AllocTensor<T1>();
    DataCopy(cosUb, cosGm[blockIdx * tilingData->cosCoreOffset], tilingData->coscdNum);
    DataCopy(sinUb, sinGm[blockIdx * tilingData->cosCoreOffset], tilingData->coscdNum);
    DataCopy(qUb, qGm[blockIdx * tilingData->qCoreOffset], tilingData->qcdNum);
    DataCopy(qUb[tilingData->qcdNum], kGm[blockIdx * tilingData->kCoreOffset], tilingData->kcdNum);
    qInQueue.EnQue(qUb);
    cosInQueue.EnQue(cosUb);
    sinInQueue.EnQue(sinUb);
}

template <typename T1, typename T2>
__aicore__ inline void ARPESmall<T1, T2>::CastCopyIn(const ApplyRotaryPosEmbTilingData* tilingData)
{
    CopyInQK(tilingData);
    LocalTensor<T1> qSize = qInQueue.DeQue<T1>();
    LocalTensor<T1> cosSize = cosInQueue.DeQue<T1>();
    LocalTensor<T1> sinSize = sinInQueue.DeQue<T1>();
#if ORIG_DTYPE_QUERY == DT_BF16
    LocalTensor<T1> qOutUb = qOutQueue.AllocTensor<T1>();
    ComputeTotary(qSize, cosSize, sinSize, qOutUb, tilingData);

    qInQueue.FreeTensor(qSize);
    cosInQueue.FreeTensor(cosSize);
    sinInQueue.FreeTensor(sinSize);
    qOutQueue.EnQue(qOutUb);
    LocalTensor<T1> qOutUbSize = qOutQueue.DeQue<T1>();

    DataCopy(qGm[blockIdx * tilingData->qCoreOffset], qOutUbSize, tilingData->qcdNum);
    DataCopy(kGm[blockIdx * tilingData->kCoreOffset], qOutUbSize[tilingData->qcdNum], tilingData->kcdNum);
    qOutQueue.FreeTensor(qOutUbSize);
#else
    T1 scalar_data = -1.0;
    Muls(sinSize, sinSize, scalar_data, tilingData->halfNum, 1, mulRepeatP);
#if ORIG_DTYPE_QUERY == DT_FLOAT
    SetMaskNorm();
    SetVectorMask<T1>(64);
#else
    SetMaskNorm();
    SetVectorMask<T1>(128);
#endif
    LocalTensor<T1> qOutUb = qOutQueue.AllocTensor<T1>();
    ComputeTotary(qSize, cosSize, sinSize, qOutUb, tilingData);

    qInQueue.FreeTensor(qSize);
    cosInQueue.FreeTensor(cosSize);
    sinInQueue.FreeTensor(sinSize);
    qOutQueue.EnQue(qOutUb);
    LocalTensor<T1> qOutUbSize = qOutQueue.DeQue<T1>();
    DataCopy(qGm[blockIdx * tilingData->qCoreOffset], qOutUbSize, tilingData->qcdNum);
    DataCopy(kGm[blockIdx * tilingData->kCoreOffset], qOutUbSize[tilingData->qcdNum], tilingData->kcdNum);
    qOutQueue.FreeTensor(qOutUbSize);
#endif
}

template <typename T1, typename T2>
__aicore__ inline void ARPESmall<T1, T2>::ComputeBF16(
    LocalTensor<T1>& qUb, LocalTensor<T1>& cosUb, LocalTensor<T1>& sinUb, LocalTensor<T1>& qoutSize,
    const ApplyRotaryPosEmbTilingData* tilingData)
{
    LocalTensor<float> mul2Ub = mul2.Get<float>();
    LocalTensor<float> mul1Ub = mul1.Get<float>();
    LocalTensor<float> cosC1 = cosCast.Get<float>();
    LocalTensor<float> sinC1 = sinCast.Get<float>();
    Cast(mul2Ub, qUb, RoundMode::CAST_NONE, tilingData->mulNum);
    LocalTensor<float> qoutSizeFP32;
    this->LocalTensor2NewTensor(qoutSizeFP32, qoutSize);
    DataCopy(qoutSizeFP32, mul2Ub[tilingData->halfNum], copyIn1);
    DataCopy(qoutSizeFP32[tilingData->halfNum], mul2Ub, copyIn1);

    Cast(cosC1, cosUb, RoundMode::CAST_NONE, tilingData->lastDim);
    Cast(sinC1, sinUb, RoundMode::CAST_NONE, tilingData->lastDim);
    SetMaskNorm();
    SetVectorMask<float>(64);
    float scalar_data = -1.0f;

    Muls<float, false>(sinC1, sinC1, scalar_data, tilingData->halfNum, 1, mulRepeatP);
    LocalTensor<float> qUbFP32;
    this->LocalTensor2NewTensor(qUbFP32, qUb);
    for (int32_t aa = 0; aa < (tilingData->qcdHalfNum); aa++) {
        Mul<float, false>(
            mul1Ub[aa * tilingData->mask], qoutSizeFP32[aa * tilingData->mask], sinC1[aa * tilingData->mask],
            tilingData->mask, tilingData->qkcNum, repeatParams);
        Mul<float, false>(
            qUbFP32[aa * tilingData->mask], mul2Ub[aa * tilingData->mask], cosC1[aa * tilingData->mask],
            tilingData->mask, tilingData->qkcNum, repeatParams);
    }
    Add(mul2Ub, mul1Ub, qUbFP32, tilingData->mulNum);

    Cast(qoutSize, mul2Ub, RoundMode::CAST_ROUND, tilingData->mulNum);
}

template <typename T1, typename T2>
__aicore__ inline void ARPESmall<T1, T2>::ComputeTotary(
    LocalTensor<T1>& qUb, LocalTensor<T1>& cosUb, LocalTensor<T1>& sinUb, LocalTensor<T1>& qoutSize,
    const ApplyRotaryPosEmbTilingData* tilingData)
{
#if ORIG_DTYPE_QUERY == DT_BF16
    ComputeBF16(qUb, cosUb, sinUb, qoutSize, tilingData);
#else
    LocalTensor<T1> mul2Ub = mul2.Get<T1>();
    LocalTensor<T1> mul1Ub = mul1.Get<T1>();

    DataCopy(qoutSize, qUb[tilingData->halfNum], copyIn1);
    DataCopy(qoutSize[tilingData->halfNum], qUb, copyIn1);

    for (int32_t aa = 0; aa < (tilingData->qcdHalfNum); aa++) {
        Mul<T1, false>(
            mul1Ub[aa * tilingData->mask], qoutSize[aa * tilingData->mask], sinUb[aa * tilingData->mask],
            tilingData->mask, tilingData->qkcNum, repeatParams);
        Mul<T1, false>(
            mul2Ub[aa * tilingData->mask], qUb[aa * tilingData->mask], cosUb[aa * tilingData->mask], tilingData->mask,
            tilingData->qkcNum, repeatParams);
    }
    Add(qoutSize, mul1Ub, mul2Ub, tilingData->mulNum);
#endif
}

template <typename T1, typename T2>
__aicore__ inline void ARPESmall<T1, T2>::ProcessPerCore(const ApplyRotaryPosEmbTilingData* tilingData)

{
    CastCopyIn(tilingData);
}

template <typename T1, typename T2>
__aicore__ inline void ARPESmall<T1, T2>::Process(const ApplyRotaryPosEmbTilingData* tilingData)
{
    if (blockIdx >= tilingData->useCoreNum) {
        return;
    }
    ProcessPerCore(tilingData);
}

} // namespace ApplyRotaryPosEmb
#endif // APPLY_ROTARY_POS_EMB_SMALL_H
