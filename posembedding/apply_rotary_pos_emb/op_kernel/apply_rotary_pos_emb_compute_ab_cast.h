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
 * \file apply_rotary_pos_emb_compute_ab_cast.h
 * \brief
 */
#ifndef APPLY_ROTARY_POS_EMB_COMPUTE_AB_CAST_H
#define APPLY_ROTARY_POS_EMB_COMPUTE_AB_CAST_H

#include "kernel_operator.h"
#include "apply_rotary_pos_emb_base.h"

namespace ApplyRotaryPosEmb {
using namespace AscendC;

template <typename T1, typename T2>
class ARPEComputeABCast : public ApplyRotaryPosEmbBase<T1> {
public:
    __aicore__ inline ARPEComputeABCast(){};
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR cos, GM_ADDR sin, GM_ADDR qOut, GM_ADDR kOut,
                                GM_ADDR workspace, const ApplyRotaryPosEmbTilingData *tilingData);
    __aicore__ inline void Process(const ApplyRotaryPosEmbTilingData *tilingData);

private:
    __aicore__ inline void ProcessPreCore(const ApplyRotaryPosEmbTilingData *tilingData);
    __aicore__ inline void ProcessLastCore(const ApplyRotaryPosEmbTilingData *tilingData);

    __aicore__ inline void CopyIn(const int64_t coreBatchIndex, const int64_t preCBatchB,
                                  const ApplyRotaryPosEmbTilingData *tilingData);
    __aicore__ inline void ComputeBF16(const int64_t coreBatchIndex, const int64_t preCBatchB, LocalTensor<T1> &qSize,
                                       LocalTensor<T1> &cosSize, LocalTensor<T1> &sinSize,
                                       const ApplyRotaryPosEmbTilingData *tilingData);
    __aicore__ inline void ComputeElse(const int64_t coreBatchIndex, const int64_t preCBatchB, LocalTensor<T1> &qSize,
                                       LocalTensor<T1> &cosSize, LocalTensor<T1> &sinSize,
                                       const ApplyRotaryPosEmbTilingData *tilingData);
    __aicore__ inline void CTotaryBF16(const int64_t preCBatchB, LocalTensor<T1> &qSize, LocalTensor<T1> &cosSize,
                                       LocalTensor<T1> &sinSize, LocalTensor<T1> &outUb,
                                       const ApplyRotaryPosEmbTilingData *tilingData);
    __aicore__ inline void CTotaryBF32(const int64_t preCBatchB, LocalTensor<T1> &qSize, LocalTensor<T1> &cosSize,
                                       LocalTensor<T1> &sinSize, LocalTensor<T1> &outUb,
                                       const ApplyRotaryPosEmbTilingData *tilingData);
    __aicore__ inline void LargeQCFP16(const int64_t preCBatchB, LocalTensor<T1> &mul1Ub, LocalTensor<T1> &qoutSizeFP32,
                                       LocalTensor<T1> &sinCUb, LocalTensor<T1> &qUbFP32, LocalTensor<T1> &mul2Ub,
                                       LocalTensor<T1> &cosCUb, const ApplyRotaryPosEmbTilingData *tilingData,
                                       BinaryRepeatParams &repeatParams);
    __aicore__ inline void SmallQCFP16(const int64_t preCBatchB, LocalTensor<T1> &mul1Ub, LocalTensor<T1> &qoutSizeFP32,
                                       LocalTensor<T1> &sinCUb, LocalTensor<T1> &qUbFP32, LocalTensor<T1> &mul2Ub,
                                       LocalTensor<T1> &cosCUb, const ApplyRotaryPosEmbTilingData *tilingData,
                                       BinaryRepeatParams &repeatParams);
    __aicore__ inline void LargeQC(const int64_t preCBatchB, LocalTensor<T2> &mul1Ub, LocalTensor<T2> &qoutSizeFP32,
                                   LocalTensor<T2> &sinCUb, LocalTensor<T2> &qUbFP32, LocalTensor<T2> &mul2Ub,
                                   LocalTensor<T2> &cosCUb, const ApplyRotaryPosEmbTilingData *tilingData,
                                   BinaryRepeatParams &repeatParams);
    __aicore__ inline void SmallQC(const int64_t preCBatchB, LocalTensor<T2> &mul1Ub, LocalTensor<T2> &qoutSizeFP32,
                                   LocalTensor<T2> &sinCUb, LocalTensor<T2> &qUbFP32, LocalTensor<T2> &mul2Ub,
                                   LocalTensor<T2> &cosCUb, const ApplyRotaryPosEmbTilingData *tilingData,
                                   BinaryRepeatParams &repeatParams);
    __aicore__ inline void ComputeTotary(const int64_t preCBatchB, LocalTensor<T1> &qSize, LocalTensor<T1> &cosSize,
                                         LocalTensor<T1> &sinSize, LocalTensor<T1> &outUb,
                                         const ApplyRotaryPosEmbTilingData *tilingData);
    __aicore__ inline void CopyOutPartialRotation(const int64_t coreBatchIndex, const int64_t preCBatchB,
                                                  LocalTensor<T1> &qOutUbSize,
                                                  const ApplyRotaryPosEmbTilingData *tilingData);
    __aicore__ inline void CopyInPartialRotation(const int64_t coreBatchIndex, const int64_t preCBatchB,
                                                 LocalTensor<T1> &qUb, const ApplyRotaryPosEmbTilingData *tilingData);
    constexpr static int32_t bufferNum = 1;
    constexpr static int32_t bufferNumdb = 2;

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, bufferNumdb> qInQueue;
    TQue<QuePosition::VECIN, bufferNumdb> cosInQueue;
    TQue<QuePosition::VECIN, bufferNumdb> sinInQueue;
    TQue<QuePosition::VECOUT, bufferNum> qOutQueue;
    TBuf<QuePosition::VECCALC> mul1;
    TBuf<QuePosition::VECCALC> mul2;
    TBuf<QuePosition::VECCALC> cosC;
    TBuf<QuePosition::VECCALC> sinC;
    GlobalTensor<T1> qGm;
    GlobalTensor<T1> kGm;
    GlobalTensor<T1> cosGm;
    GlobalTensor<T1> sinGm;
    DataCopyParams copyIn1;
    DataCopyParams copyInq2q1;
    DataCopyParams copyIn2;
    DataCopyParams copyOut1;
    DataCopyParams copyOut2;
    UnaryRepeatParams mulRepeatP;
    BinaryRepeatParams repeatParams = {1, 1, 1, 0, 0, 0};
    uint64_t blockIdx = 0;
};

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCast<T1, T2>::Init(GM_ADDR q, GM_ADDR k, GM_ADDR cos, GM_ADDR sin, GM_ADDR qOut,
                                                       GM_ADDR kOut, GM_ADDR workspace,
                                                       const ApplyRotaryPosEmbTilingData *tilingData)
{
    blockIdx = GetBlockIdx();
    qGm.SetGlobalBuffer((__gm__ T1 *)q);
    kGm.SetGlobalBuffer((__gm__ T1 *)k);
    cosGm.SetGlobalBuffer((__gm__ T1 *)cos);
    sinGm.SetGlobalBuffer((__gm__ T1 *)sin);
    pipe.InitBuffer(cosInQueue, bufferNumdb, tilingData->cosPart1Ub);
    pipe.InitBuffer(qInQueue, bufferNumdb, tilingData->qPart1Ub);
    pipe.InitBuffer(sinInQueue, bufferNumdb, tilingData->cosPart1Ub);
    pipe.InitBuffer(mul1, tilingData->q2q1Part1Ub);
    pipe.InitBuffer(qOutQueue, bufferNum, tilingData->qPart1Ub);
    pipe.InitBuffer(mul2, tilingData->q2q1Part1Ub);

#if ORIG_DTYPE_QUERY == DT_BF16
    pipe.InitBuffer(cosC, tilingData->sin1UbSize);
    pipe.InitBuffer(sinC, tilingData->sin1UbSize);
#endif
    copyIn1.blockCount = 1;
    copyIn1.blockLen = tilingData->blockLenQ;
    copyIn1.srcStride = 0;
    copyIn1.dstStride = tilingData->srcStrideK;

    copyIn2.blockCount = 1;
    copyIn2.blockLen = tilingData->srcStrideK;
    copyIn2.srcStride = 0;
    copyIn2.dstStride = tilingData->blockLenQ;

    mulRepeatP.dstBlkStride = 1;
    mulRepeatP.srcBlkStride = 1;
    mulRepeatP.dstRepStride = tilingData->dstRepSBr;
    mulRepeatP.srcRepStride = tilingData->dstRepSBr;

    copyInq2q1.blockCount = 1;
    copyInq2q1.blockLen = tilingData->blockLenq2q1;
    copyInq2q1.srcStride = tilingData->blockLenq2q1;
    copyInq2q1.dstStride = tilingData->blockLenq2q1;

    copyOut1.blockCount = 1;
    copyOut1.blockLen = tilingData->blockLenQ;
    copyOut1.srcStride = tilingData->srcStrideK;
    copyOut1.dstStride = 0;

    copyOut2.blockCount = 1;
    copyOut2.blockLen = tilingData->srcStrideK;
    copyOut2.srcStride = tilingData->blockLenQ;
    copyOut2.dstStride = 0;
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCast<T1, T2>::SmallQC(const int64_t preCBatchB, LocalTensor<T2> &mul1Ub,
                                                          LocalTensor<T2> &qoutSizeFP32, LocalTensor<T2> &sinCUb,
                                                          LocalTensor<T2> &qUbFP32, LocalTensor<T2> &mul2Ub,
                                                          LocalTensor<T2> &cosCUb,
                                                          const ApplyRotaryPosEmbTilingData *tilingData,
                                                          BinaryRepeatParams &repeatParams)
{
    int64_t qcdhalfNum = tilingData->lastDim / tilingData->mask; // 类似于tiling侧小shape情况处理
    for (int64_t ii = 0; ii < tilingData->qkcNum; ii++) {
        for (int32_t j = 0; j < qcdhalfNum; j++) {
            Mul<T2>(mul1Ub[ii * tilingData->lastDim + j * tilingData->mask],
                    qoutSizeFP32[ii * tilingData->lastDim + j * tilingData->mask], sinCUb[j * tilingData->mask],
                    tilingData->mask, preCBatchB, repeatParams);
            Mul<T2>(qUbFP32[ii * tilingData->lastDim + j * tilingData->mask],
                    mul2Ub[ii * tilingData->lastDim + j * tilingData->mask], cosCUb[j * tilingData->mask],
                    tilingData->mask, preCBatchB, repeatParams);
        }
    }
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCast<T1, T2>::SmallQCFP16(const int64_t preCBatchB, LocalTensor<T1> &mul1Ub,
                                                              LocalTensor<T1> &outUb, LocalTensor<T1> &sinSize,
                                                              LocalTensor<T1> &mul2Ub, LocalTensor<T1> &qSize,
                                                              LocalTensor<T1> &cosSize,
                                                              const ApplyRotaryPosEmbTilingData *tilingData,
                                                              BinaryRepeatParams &repeatParams)
{
    for (int64_t ii = 0; ii < tilingData->qkcNum; ii++) {
        Mul<T1>(mul1Ub[ii * tilingData->lastDim], outUb[ii * tilingData->lastDim], sinSize, tilingData->mask,
                preCBatchB, repeatParams);
        Mul<T1>(mul2Ub[ii * tilingData->lastDim], qSize[ii * tilingData->lastDim], cosSize, tilingData->mask,
                preCBatchB, repeatParams);
    }
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCast<T1, T2>::LargeQCFP16(const int64_t preCBatchB, LocalTensor<T1> &mul1Ub,
                                                              LocalTensor<T1> &outUb, LocalTensor<T1> &sinSize,
                                                              LocalTensor<T1> &mul2Ub, LocalTensor<T1> &qSize,
                                                              LocalTensor<T1> &cosSize,
                                                              const ApplyRotaryPosEmbTilingData *tilingData,
                                                              BinaryRepeatParams &repeatParams)
{
    for (int64_t i = 0; i < preCBatchB; i++) {
        Mul<T1>(mul1Ub[i * tilingData->qkcNum * tilingData->lastDim],
                outUb[i * tilingData->qkcNum * tilingData->lastDim], sinSize[i * tilingData->lastDim], tilingData->mask,
                tilingData->qkcNum, repeatParams);
        Mul<T1>(mul2Ub[i * tilingData->qkcNum * tilingData->lastDim],
                qSize[i * tilingData->qkcNum * tilingData->lastDim], cosSize[i * tilingData->lastDim], tilingData->mask,
                tilingData->qkcNum, repeatParams);
    }
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCast<T1, T2>::LargeQC(const int64_t preCBatchB, LocalTensor<T2> &mul1Ub,
                                                          LocalTensor<T2> &qoutSizeFP32, LocalTensor<T2> &sinCUb,
                                                          LocalTensor<T2> &qUbFP32, LocalTensor<T2> &mul2Ub,
                                                          LocalTensor<T2> &cosCUb,
                                                          const ApplyRotaryPosEmbTilingData *tilingData,
                                                          BinaryRepeatParams &repeatParams)
{
    int64_t qcdhalfNum = tilingData->lastDim / tilingData->mask;
    for (int64_t i = 0; i < preCBatchB; i++) {
        for (int32_t j = 0; j < qcdhalfNum; j++) {
            Mul<T2>(mul1Ub[i * tilingData->qkcNum * tilingData->lastDim + j * tilingData->mask],
                    qoutSizeFP32[i * tilingData->qkcNum * tilingData->lastDim + j * tilingData->mask],
                    sinCUb[i * tilingData->lastDim + j * tilingData->mask], tilingData->mask, tilingData->qkcNum,
                    repeatParams);
            Mul<T2>(qUbFP32[i * tilingData->qkcNum * tilingData->lastDim + j * tilingData->mask],
                    mul2Ub[i * tilingData->qkcNum * tilingData->lastDim + j * tilingData->mask],
                    cosCUb[i * tilingData->lastDim + j * tilingData->mask], tilingData->mask, tilingData->qkcNum,
                    repeatParams);
        }
    }
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCast<T1, T2>::CTotaryBF32(const int64_t preCBatchB, LocalTensor<T1> &qSize,
                                                              LocalTensor<T1> &cosSize, LocalTensor<T1> &sinSize,
                                                              LocalTensor<T1> &outUb,
                                                              const ApplyRotaryPosEmbTilingData *tilingData)
{
    LocalTensor<T1> mul2Ub = mul2.Get<T1>();
    LocalTensor<T1> mul1Ub = mul1.Get<T1>();
    if (tilingData->qkcNum >= preCBatchB) {
        repeatParams.dstRepStride = tilingData->dstRepSBr;
        repeatParams.src0RepStride = tilingData->dstRepSBr;
        repeatParams.src1RepStride = 0;
        LargeQC(preCBatchB, mul1Ub, outUb, sinSize, mul2Ub, qSize, cosSize, tilingData, repeatParams);
    } else {
        repeatParams.dstRepStride = tilingData->mulNum;
        repeatParams.src0RepStride = tilingData->mulNum;
        repeatParams.src1RepStride = tilingData->dstRepSBr;
        SmallQC(preCBatchB, mul1Ub, outUb, sinSize, mul2Ub, qSize, cosSize, tilingData, repeatParams);
    }

    Add(outUb, mul1Ub, mul2Ub, preCBatchB * tilingData->qkcNum * tilingData->lastDim);
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCast<T1, T2>::CTotaryBF16(const int64_t preCBatchB, LocalTensor<T1> &qSize,
                                                              LocalTensor<T1> &cosSize, LocalTensor<T1> &sinSize,
                                                              LocalTensor<T1> &outUb,
                                                              const ApplyRotaryPosEmbTilingData *tilingData)
{
    LocalTensor<T2> cosCUb = cosC.Get<T2>();
    LocalTensor<T2> sinCUb = sinC.Get<T2>();
    LocalTensor<T2> mul2Ub = mul2.Get<T2>();
    LocalTensor<T2> mul1Ub = mul1.Get<T2>();
    Cast(mul2Ub, qSize, RoundMode::CAST_NONE, preCBatchB * tilingData->qkcNum * tilingData->lastDim);
    copyInq2q1.blockCount = preCBatchB * tilingData->qkcNum;

    LocalTensor<T2> qoutSizeFP32;
    this->LocalTensor2NewTensor(qoutSizeFP32, outUb);
    DataCopy(qoutSizeFP32, mul2Ub[tilingData->halfNum], copyInq2q1);
    DataCopy(qoutSizeFP32[tilingData->halfNum], mul2Ub, copyInq2q1);
    Cast(cosCUb, cosSize, RoundMode::CAST_NONE, preCBatchB * tilingData->lastDim);
    Cast(sinCUb, sinSize, RoundMode::CAST_NONE, preCBatchB * tilingData->lastDim);

    Muls(sinCUb, sinCUb, T2(-1.0), tilingData->halfNum, preCBatchB, mulRepeatP);
    LocalTensor<T2> qUbFP32;
    this->LocalTensor2NewTensor(qUbFP32, qSize);
    if (tilingData->qkcNum >= preCBatchB) {
        repeatParams.dstRepStride = tilingData->dstRepSBr;
        repeatParams.src0RepStride = tilingData->dstRepSBr;
        repeatParams.src1RepStride = 0;
        LargeQC(preCBatchB, mul1Ub, qoutSizeFP32, sinCUb, qUbFP32, mul2Ub, cosCUb, tilingData, repeatParams);
    } else {
        repeatParams.dstRepStride = tilingData->mulNum;
        repeatParams.src0RepStride = tilingData->mulNum;
        repeatParams.src1RepStride = tilingData->dstRepSBr;
        SmallQC(preCBatchB, mul1Ub, qoutSizeFP32, sinCUb, qUbFP32, mul2Ub, cosCUb, tilingData, repeatParams);
    }

    Add(mul2Ub, mul1Ub, qUbFP32, preCBatchB * tilingData->qkcNum * tilingData->lastDim);
    Cast(outUb, mul2Ub, RoundMode::CAST_ROUND, preCBatchB * tilingData->qkcNum * tilingData->lastDim);
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCast<T1, T2>::ComputeTotary(const int64_t preCBatchB, LocalTensor<T1> &qSize,
                                                                LocalTensor<T1> &cosSize, LocalTensor<T1> &sinSize,
                                                                LocalTensor<T1> &outUb,
                                                                const ApplyRotaryPosEmbTilingData *tilingData)
{
#if ORIG_DTYPE_QUERY == DT_BF16
    CTotaryBF16(preCBatchB, qSize, cosSize, sinSize, outUb, tilingData);
#endif
#if ORIG_DTYPE_QUERY == DT_FLOAT
    CTotaryBF32(preCBatchB, qSize, cosSize, sinSize, outUb, tilingData);
#endif
#if ORIG_DTYPE_QUERY == DT_FLOAT16
    LocalTensor<T1> mul2Ub = mul2.Get<T1>();
    LocalTensor<T1> mul1Ub = mul1.Get<T1>();
    if (tilingData->qkcNum >= preCBatchB) {
        repeatParams.dstRepStride = tilingData->dstRepSBr;
        repeatParams.src0RepStride = tilingData->dstRepSBr;
        repeatParams.src1RepStride = 0;
        LargeQCFP16(preCBatchB, mul1Ub, outUb, sinSize, mul2Ub, qSize, cosSize, tilingData, repeatParams);
    } else {
        repeatParams.dstRepStride = tilingData->mulNum;
        repeatParams.src0RepStride = tilingData->mulNum;
        repeatParams.src1RepStride = tilingData->dstRepSBr;
        SmallQCFP16(preCBatchB, mul1Ub, outUb, sinSize, mul2Ub, qSize, cosSize, tilingData, repeatParams);
    }
    Add(outUb, mul1Ub, mul2Ub, preCBatchB * tilingData->qkcNum * tilingData->lastDim);
#endif
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCast<T1, T2>::CopyOutPartialRotation(const int64_t coreBatchIndex,
                                                                         const int64_t preCBatchB,
                                                                         LocalTensor<T1> &qOutUbSize,
                                                                         const ApplyRotaryPosEmbTilingData *tilingData)
{
    copyOut1.blockCount = tilingData->qcNum;
    copyOut1.blockLen = tilingData->blockMoveQ;
    copyOut1.srcStride = 0;
    copyOut1.dstStride = tilingData->blockMoveQ;
    int64_t qBatchSize = tilingData->qcNum * tilingData->qDim3;    // 每个批次的q元素数
    int64_t kBatchSize = tilingData->kcNum * tilingData->kDim3;    // 每个批次的k元素数
    int64_t batchOffset = coreBatchIndex * tilingData->preCBatchB; // 批次的偏移量

    if ((tilingData->kcNum) == 1) { // kn==1: 特殊情况，k可以非连续搬运，q根据批次循环非连续搬运
        for (int i = 0; i < preCBatchB; i++) {
            DataCopy(qGm[blockIdx * tilingData->qCoreOffset + batchOffset * qBatchSize + i * qBatchSize],
                     qOutUbSize[i * (tilingData->qcdNum + tilingData->kcdNum)], copyOut1);
        }
        copyOut2.dstStride = tilingData->srcStrideK;
        DataCopy(kGm[blockIdx * tilingData->kCoreOffset + batchOffset * kBatchSize], qOutUbSize[tilingData->qcdNum],
                 copyOut2);
    } else { // kn!=1: q和k正常根据批次非连续搬运
        copyOut2.blockCount = tilingData->kcNum;
        copyOut2.blockLen = tilingData->blockMoveQ;
        copyOut2.srcStride = 0;
        copyOut2.dstStride = tilingData->blockMoveQ;
        for (int i = 0; i < preCBatchB; i++) {
            DataCopy(qGm[blockIdx * tilingData->qCoreOffset + batchOffset * qBatchSize + i * qBatchSize],
                     qOutUbSize[i * (tilingData->qcdNum + tilingData->kcdNum)], copyOut1);
            DataCopy(kGm[blockIdx * tilingData->kCoreOffset + batchOffset * kBatchSize + i * kBatchSize],
                     qOutUbSize[tilingData->qcdNum + i * (tilingData->qcdNum + tilingData->kcdNum)], copyOut2);
        }
    }
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCast<T1, T2>::ComputeBF16(const int64_t coreBatchIndex, const int64_t preCBatchB,
                                                              LocalTensor<T1> &qSize, LocalTensor<T1> &cosSize,
                                                              LocalTensor<T1> &sinSize,
                                                              const ApplyRotaryPosEmbTilingData *tilingData)
{
    LocalTensor<T1> qOutUb = qOutQueue.AllocTensor<T1>();
    ComputeTotary(preCBatchB, qSize, cosSize, sinSize, qOutUb, tilingData);
    qInQueue.FreeTensor(qSize);
    cosInQueue.FreeTensor(cosSize);
    sinInQueue.FreeTensor(sinSize);
    qOutQueue.EnQue(qOutUb);

    LocalTensor<T1> qOutUbSize = qOutQueue.DeQue<T1>();
    copyOut1.blockCount = preCBatchB;
    copyOut2.blockCount = preCBatchB;

    if (tilingData->qDim3 > tilingData->lastDim) { // 部分维度旋转处理
        CopyOutPartialRotation(coreBatchIndex, preCBatchB, qOutUbSize, tilingData);
    } else { // 全维度旋转处理
        DataCopy(qGm[blockIdx * tilingData->qCoreOffset + coreBatchIndex * tilingData->preCBatchB * tilingData->qcdNum],
                 qOutUbSize, copyOut1);
        DataCopy(kGm[blockIdx * tilingData->kCoreOffset + coreBatchIndex * tilingData->preCBatchB * tilingData->kcdNum],
                 qOutUbSize[tilingData->qcdNum], copyOut2);
    }
    qOutQueue.FreeTensor(qOutUbSize);
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCast<T1, T2>::ComputeElse(const int64_t coreBatchIndex, const int64_t preCBatchB,
                                                              LocalTensor<T1> &qSize, LocalTensor<T1> &cosSize,
                                                              LocalTensor<T1> &sinSize,
                                                              const ApplyRotaryPosEmbTilingData *tilingData)
{
    Muls(sinSize, sinSize, T1(-1.0), tilingData->halfNum, preCBatchB, mulRepeatP);
    LocalTensor<T1> qOutUb = qOutQueue.AllocTensor<T1>();
    copyInq2q1.blockCount = preCBatchB * tilingData->qkcNum;
    DataCopy(qOutUb, qSize[tilingData->halfNum], copyInq2q1);
    DataCopy(qOutUb[tilingData->halfNum], qSize, copyInq2q1);
    ComputeTotary(preCBatchB, qSize, cosSize, sinSize, qOutUb, tilingData);
    qInQueue.FreeTensor(qSize);
    cosInQueue.FreeTensor(cosSize);
    sinInQueue.FreeTensor(sinSize);
    qOutQueue.EnQue(qOutUb);
    LocalTensor<T1> qOutUbSize = qOutQueue.DeQue<T1>();

    copyOut1.blockCount = preCBatchB;
    copyOut2.blockCount = preCBatchB;
    if (tilingData->qDim3 > tilingData->lastDim) { // 部分维度旋转处理
        CopyOutPartialRotation(coreBatchIndex, preCBatchB, qOutUbSize, tilingData);
    } else { // 全维度旋转处理
        DataCopy(qGm[blockIdx * tilingData->qCoreOffset + coreBatchIndex * tilingData->preCBatchB * tilingData->qcdNum],
                 qOutUbSize, copyOut1);
        DataCopy(kGm[blockIdx * tilingData->kCoreOffset + coreBatchIndex * tilingData->preCBatchB * tilingData->kcdNum],
                 qOutUbSize[tilingData->qcdNum], copyOut2);
    }
    qOutQueue.FreeTensor(qOutUbSize);
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCast<T1, T2>::CopyInPartialRotation(const int64_t coreBatchIndex,
                                                                        const int64_t preCBatchB, LocalTensor<T1> &qUb,
                                                                        const ApplyRotaryPosEmbTilingData *tilingData)
{
    copyIn1.blockCount = tilingData->qcNum;
    copyIn1.blockLen = tilingData->blockMoveQ;
    copyIn1.srcStride = tilingData->blockMoveQ;
    copyIn1.dstStride = 0;
    int64_t qBatchSize = tilingData->qcNum * tilingData->qDim3;    // 每个批次的q元素数
    int64_t kBatchSize = tilingData->kcNum * tilingData->kDim3;    // 每个批次的k元素数
    int64_t batchOffset = coreBatchIndex * tilingData->preCBatchB; // 批次的偏移量

    if ((tilingData->kcNum) == 1) { // kn==1: 特殊情况，k非连续搬运，q根据批次循环非连续搬运
        for (int i = 0; i < preCBatchB; i++) {
            DataCopy(qUb[i * (tilingData->qcdNum + tilingData->kcdNum)],
                     qGm[blockIdx * tilingData->qCoreOffset + batchOffset * qBatchSize + i * qBatchSize], copyIn1);
        }
        copyIn2.srcStride = tilingData->srcStrideK;
        DataCopy(qUb[tilingData->qcdNum], kGm[blockIdx * tilingData->kCoreOffset + batchOffset * kBatchSize], copyIn2);
    } else { // kn!=1: q和k正常根据批次非连续搬运
        copyIn2.blockCount = tilingData->kcNum;
        copyIn2.blockLen = tilingData->blockMoveQ;
        copyIn2.srcStride = tilingData->blockMoveQ;
        copyIn2.dstStride = 0;
        for (int i = 0; i < preCBatchB; i++) {
            DataCopy(qUb[i * (tilingData->qcdNum + tilingData->kcdNum)],
                     qGm[blockIdx * tilingData->qCoreOffset + batchOffset * qBatchSize + i * qBatchSize], copyIn1);
            DataCopy(qUb[tilingData->qcdNum + i * (tilingData->qcdNum + tilingData->kcdNum)],
                     kGm[blockIdx * tilingData->kCoreOffset + batchOffset * kBatchSize + i * kBatchSize], copyIn2);
        }
    }
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCast<T1, T2>::CopyIn(const int64_t coreBatchIndex, const int64_t preCBatchB,
                                                         const ApplyRotaryPosEmbTilingData *tilingData)

{
    copyIn2.blockCount = preCBatchB;
    copyIn1.blockCount = preCBatchB;
    LocalTensor<T1> qUb = qInQueue.AllocTensor<T1>();
    LocalTensor<T1> cosUb = cosInQueue.AllocTensor<T1>();
    LocalTensor<T1> sinUb = sinInQueue.AllocTensor<T1>();
    DataCopy(
        cosUb,
        cosGm[blockIdx * tilingData->cosCoreOffset + coreBatchIndex * tilingData->preCBatchB * tilingData->coscdNum],
        preCBatchB * tilingData->coscdNum);
    DataCopy(
        sinUb,
        sinGm[blockIdx * tilingData->cosCoreOffset + coreBatchIndex * tilingData->preCBatchB * tilingData->coscdNum],
        preCBatchB * tilingData->coscdNum);

    if (tilingData->qDim3 > tilingData->lastDim) { // 部分维度旋转处理
        CopyInPartialRotation(coreBatchIndex, preCBatchB, qUb, tilingData);
    } else { // 全维度旋转处理
        DataCopy(qUb,
                 qGm[blockIdx * tilingData->qCoreOffset + coreBatchIndex * tilingData->preCBatchB * tilingData->qcdNum],
                 copyIn1);
        DataCopy(qUb[tilingData->qcdNum],
                 kGm[blockIdx * tilingData->kCoreOffset + coreBatchIndex * tilingData->preCBatchB * tilingData->kcdNum],
                 copyIn2);
    }
    qInQueue.EnQue(qUb);
    cosInQueue.EnQue(cosUb);
    sinInQueue.EnQue(sinUb);
    LocalTensor<T1> qSize = qInQueue.DeQue<T1>();
    LocalTensor<T1> cosSize = cosInQueue.DeQue<T1>();
    LocalTensor<T1> sinSize = sinInQueue.DeQue<T1>();

#if ORIG_DTYPE_QUERY == DT_BF16
    ComputeBF16(coreBatchIndex, preCBatchB, qSize, cosSize, sinSize, tilingData);
#else
    ComputeElse(coreBatchIndex, preCBatchB, qSize, cosSize, sinSize, tilingData);
#endif
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCast<T1, T2>::ProcessPreCore(const ApplyRotaryPosEmbTilingData *tilingData)

{
    for (int64_t i = 0; i < tilingData->preCLTimes; i++) {
        CopyIn(i, tilingData->preCBatchB, tilingData);
    }
    CopyIn(tilingData->preCLTimes, tilingData->preCBatchL, tilingData);
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCast<T1, T2>::ProcessLastCore(const ApplyRotaryPosEmbTilingData *tilingData)

{
    for (int64_t i = 0; i < tilingData->lastCLTimes; i++) {
        CopyIn(i, tilingData->preCBatchB, tilingData);
    }
    CopyIn(tilingData->lastCLTimes, tilingData->lastCBatchL, tilingData);
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCast<T1, T2>::Process(const ApplyRotaryPosEmbTilingData *tilingData)
{
    if (blockIdx >= tilingData->useCoreNum) {
        return;
    }

    if (blockIdx == tilingData->useCoreNum - 1) {
        ProcessLastCore(tilingData);
    } else {
        ProcessPreCore(tilingData);
    }
}

} // namespace ApplyRotaryPosEmb

#endif // APPLY_ROTARY_POS_EMB_COMPUTE_AB_CAST_H
