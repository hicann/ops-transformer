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
 * \file rotary_position_embedding_reg_a_and_b.h
 * \brief
 */
#ifndef ROTARY_POSITION_EMBEDDING_REG_A_AND_B_H
#define ROTARY_POSITION_EMBEDDING_REG_A_AND_B_H

#include "partial_rotary_mul_quant_hif4_common.h"

namespace PartialRotaryMulQuantOps {
using namespace AscendC::MicroAPI;
using namespace AscendC;

template <typename T, bool IsBoardCast>
class RotaryPositionEmbeddingAAndB {
public:
    __aicore__ inline RotaryPositionEmbeddingAAndB(){};
    __aicore__ inline ~RotaryPositionEmbeddingAAndB(){};
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR cos, GM_ADDR sin, GM_ADDR qOut, GM_ADDR scaleOut, GM_ADDR workspace,
                                const QuantRopeRegbaseTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitAllGlobalBuffer(GM_ADDR q, GM_ADDR cos, GM_ADDR sin, GM_ADDR qOut, GM_ADDR scaleOut);
    __aicore__ inline void InitAllBuffer();
    __aicore__ inline void InitLoopParams();
    __aicore__ inline void ProcessInLoop(LocalTensor<T>& cos, LocalTensor<T>& sin, int64_t bStart, int64_t bLength);
    __aicore__ inline void CopyInCosAndSin(int64_t bStart, int64_t bLength);
    __aicore__ inline void CopyIn(GlobalTensor<T>& source, int64_t bStart, int64_t bLength);
    __aicore__ inline void ComputeAndQuantize(LocalTensor<T>& inUb, LocalTensor<T>& cos, LocalTensor<T>& sin,
                                              LocalTensor<int32_t>& yQuantLocal, LocalTensor<uint16_t>& scaleLocal,
                                              int64_t bLength, int64_t sliceLen);
    __aicore__ inline void CopyOut(int64_t bStart, int64_t bLength, int64_t dCount);

private:
    constexpr static uint32_t COS_DB_BUFFER = IsBoardCast ? 1 : DOUBLE_BUFFER;

    TPipe* pipe_;

    GlobalTensor<T> qGm_;
    GlobalTensor<T> cosGm_;
    GlobalTensor<T> sinGm_;
    GlobalTensor<uint8_t> qOutGm_;
    GlobalTensor<float> scaleGm_;

    TQue<QuePosition::VECIN, COS_DB_BUFFER> cosInQueue_;
    TQue<QuePosition::VECIN, COS_DB_BUFFER> sinInQueue_;

    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    TQue<QuePosition::VECOUT, 1> scaleQueue_;

    TBuf<TPosition::VECCALC> ropeBuf_;
    TBuf<TPosition::VECCALC> eaBuf_;
    TBuf<TPosition::VECCALC> ebBuf_;
    TBuf<TPosition::VECCALC> ecBuf_;

    int64_t blockIdx_ = 0;
    int64_t bBlockStart_ = 0;
    int64_t bBlockLength_ = 0;

    const QuantRopeRegbaseTilingData* tilingData_;
    int64_t ubFactorB_ = 0;
    int64_t D_ = 0;
    int64_t dAlign_ = 0;
    int64_t dAlignNope_ = 0;
    int64_t dAlignH8_ = 0;
    int64_t dAlignH8Nope_ = 0;
    int64_t sliceAlign_ = 0;
    float scale_ = 1.0f;

    uint8_t dSplitCoef_ = 1;
    uint8_t copyInQSplitCoef_ = 1;
    uint64_t ubCopyInStride = 0;
};

template <typename T, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingAAndB<T, IsBoardCast>::Init(GM_ADDR q, GM_ADDR cos, GM_ADDR sin,
                                                                          GM_ADDR qOut, GM_ADDR scaleOut,
                                                                          GM_ADDR workspace,
                                                                          const QuantRopeRegbaseTilingData* tilingData,
                                                                          TPipe* pipe)
{
    this->tilingData_ = tilingData;
    this->pipe_ = pipe;
    this->blockIdx_ = GetBlockIdx();
    this->InitAllGlobalBuffer(q, cos, sin, qOut, scaleOut);
    this->InitAllBuffer();
    this->InitLoopParams();
}

template <typename T, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingAAndB<T, IsBoardCast>::InitAllGlobalBuffer(GM_ADDR q, GM_ADDR cos,
                                                                                         GM_ADDR sin, GM_ADDR qOut,
                                                                                         GM_ADDR scaleOut)
{
    this->qGm_.SetGlobalBuffer((__gm__ T*)q);
    this->cosGm_.SetGlobalBuffer((__gm__ T*)cos);
    this->sinGm_.SetGlobalBuffer((__gm__ T*)sin);
    this->qOutGm_.SetGlobalBuffer((__gm__ uint8_t*)qOut);
    this->scaleGm_.SetGlobalBuffer((__gm__ float*)scaleOut);
}

template <typename T, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingAAndB<T, IsBoardCast>::InitAllBuffer()
{
    this->ubFactorB_ = this->tilingData_->ubFactorB;
    this->D_ = this->tilingData_->D;
    if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::HALF) ||
        tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::DEEPSEEK_INTERLEAVE)) {
        this->dSplitCoef_ = HALF_INTERLEAVE_COEF;
    } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::QUARTER)) {
        this->dSplitCoef_ = QUARTER_MODE_COEF;
    }
    this->copyInQSplitCoef_ = 1;
    this->dAlign_ = ops::CeilAlign<int64_t>(D_, BLOCK_TYPE_SIZE / sizeof(T));
    this->dAlignNope_ = ops::CeilAlign<int64_t>(tilingData_->sliceStart, BLOCK_TYPE_SIZE / sizeof(T));
    this->dAlignH8_ = ops::CeilAlign<int64_t>(tilingData_->sliceLength, BLOCK_TYPE_SIZE / sizeof(uint8_t));
    this->dAlignH8Nope_ = ops::CeilAlign<int64_t>(tilingData_->sliceStart, BLOCK_TYPE_SIZE / sizeof(uint8_t));
    this->sliceAlign_ =
        ops::CeilAlign<int64_t>(tilingData_->sliceLength / dSplitCoef_, BLOCK_TYPE_SIZE / sizeof(T)) * dSplitCoef_;
    this->scale_ = tilingData_->scale;

    this->pipe_->InitBuffer(this->inQueue_, DOUBLE_BUFFER, ubFactorB_ * dAlign_ * sizeof(T));
    if constexpr (IsSameType<T, float>::value) {
        this->pipe_->InitBuffer(this->ropeBuf_, ubFactorB_ * dAlign_ * sizeof(bfloat16_t));
    }
    if constexpr (IsBoardCast) {
        this->pipe_->InitBuffer(this->cosInQueue_, COS_DB_BUFFER, sliceAlign_ * sizeof(T));
        this->pipe_->InitBuffer(this->sinInQueue_, COS_DB_BUFFER, sliceAlign_ * sizeof(T));
    } else {
        this->pipe_->InitBuffer(this->cosInQueue_, COS_DB_BUFFER, ubFactorB_ * sliceAlign_ * sizeof(T));
        this->pipe_->InitBuffer(this->sinInQueue_, COS_DB_BUFFER, ubFactorB_ * sliceAlign_ * sizeof(T));
    }
    pipe_->InitBuffer(eaBuf_, ubFactorB_ * dAlign_ / 64);
    pipe_->InitBuffer(ebBuf_, ubFactorB_ * dAlign_ / 64);
    pipe_->InitBuffer(ecBuf_, ubFactorB_ * dAlign_ / 32);
    this->pipe_->InitBuffer(this->outQueue_, 1, ubFactorB_ * dAlign_ / 2);
    this->pipe_->InitBuffer(this->scaleQueue_, 1, ubFactorB_ * dAlign_ / 64 * 4);
}

template <typename T, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingAAndB<T, IsBoardCast>::InitLoopParams()
{
    this->bBlockLength_ = tilingData_->blockFactorB;
    if (blockIdx_ == tilingData_->blockNumB - 1 && tilingData_->B % tilingData_->blockFactorB != 0) {
        this->bBlockLength_ = tilingData_->B % tilingData_->blockFactorB;
    }
    this->bBlockStart_ = blockIdx_ * tilingData_->blockFactorB;
}

template <typename T, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingAAndB<T, IsBoardCast>::Process()
{
    int64_t ubLoopCount = ops::CeilDiv(bBlockLength_, ubFactorB_);
    if constexpr (IsBoardCast) {
        this->CopyInCosAndSin(0, 1);
        LocalTensor<T> cosUb = this->cosInQueue_.template DeQue<T>();
        LocalTensor<T> sinUb = this->sinInQueue_.template DeQue<T>();
        for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopCount; ubLoopIdx++) {
            this->ProcessInLoop(cosUb, sinUb, bBlockStart_ + ubLoopIdx * ubFactorB_,
                                ubLoopIdx != ubLoopCount - 1 ? ubFactorB_ : bBlockLength_ - ubLoopIdx * ubFactorB_);
        }
        this->cosInQueue_.FreeTensor(cosUb);
        this->sinInQueue_.FreeTensor(sinUb);
    } else {
        for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopCount; ubLoopIdx++) {
            this->CopyInCosAndSin(bBlockStart_ + ubLoopIdx * ubFactorB_,
                                  ubLoopIdx != ubLoopCount - 1 ? ubFactorB_ : bBlockLength_ - ubLoopIdx * ubFactorB_);
            LocalTensor<T> cosUb = this->cosInQueue_.template DeQue<T>();
            LocalTensor<T> sinUb = this->sinInQueue_.template DeQue<T>();
            this->ProcessInLoop(cosUb, sinUb, bBlockStart_ + ubLoopIdx * ubFactorB_,
                                ubLoopIdx != ubLoopCount - 1 ? ubFactorB_ : bBlockLength_ - ubLoopIdx * ubFactorB_);
            this->cosInQueue_.FreeTensor(cosUb);
            this->sinInQueue_.FreeTensor(sinUb);
        }
    }
}

template <typename T, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingAAndB<T, IsBoardCast>::ProcessInLoop(LocalTensor<T>& cos,
                                                                                   LocalTensor<T>& sin,
                                                                                   int64_t bUbStart, int64_t bUbLength)
{
    int64_t sliceStart = tilingData_->sliceStart;
    int64_t sliceLen = tilingData_->sliceLength;
    CopyIn(qGm_, bUbStart, bUbLength);
    LocalTensor<T> inUb = this->inQueue_.template DeQue<T>();

    LocalTensor<int32_t> yQuantLocal = outQueue_.AllocTensor<int32_t>();
    LocalTensor<uint16_t> scaleLocal = scaleQueue_.AllocTensor<uint16_t>();

    ComputeAndQuantize(inUb, cos, sin, yQuantLocal, scaleLocal, bUbLength, sliceLen);
    CopyOut(bUbStart, bUbLength, D_);

    this->inQueue_.FreeTensor(inUb);
}

template <typename T, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingAAndB<T, IsBoardCast>::CopyInCosAndSin(int64_t bStart, int64_t bLength)
{
    LocalTensor<T> cosUb = this->cosInQueue_.template AllocTensor<T>();
    LocalTensor<T> sinUb = this->sinInQueue_.template AllocTensor<T>();
    DataCopyPadExtParams<T> copyPadExtparams;
    copyPadExtparams.isPad = false;
    copyPadExtparams.leftPadding = 0;
    copyPadExtparams.rightPadding = 0;
    copyPadExtparams.paddingValue = 0;
    DataCopyExtParams copyExtParams;
    copyExtParams.blockCount = bLength * dSplitCoef_;
    copyExtParams.blockLen = tilingData_->sliceLength * sizeof(T) / dSplitCoef_;
    copyExtParams.srcStride = 0;
    copyExtParams.dstStride = (sliceAlign_ - tilingData_->sliceLength) * sizeof(T) / dSplitCoef_ / ops::BLOCK_BYTES;
    DataCopyPad(cosUb, this->cosGm_[bStart * tilingData_->sliceLength], copyExtParams, copyPadExtparams);
    DataCopyPad(sinUb, this->sinGm_[bStart * tilingData_->sliceLength], copyExtParams, copyPadExtparams);
    this->cosInQueue_.template EnQue(cosUb);
    this->sinInQueue_.template EnQue(sinUb);
}

template <typename T, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingAAndB<T, IsBoardCast>::CopyIn(GlobalTensor<T>& source, int64_t bStart,
                                                                            int64_t bLength)
{
    LocalTensor<T> target = this->inQueue_.template AllocTensor<T>();

    DataCopyExtParams copyExtParams;
    copyExtParams.blockCount = bLength;
    copyExtParams.blockLen = D_ * sizeof(T);
    copyExtParams.srcStride = 0;
    copyExtParams.dstStride = 0;
    DataCopyPad(target, source[bStart * D_], copyExtParams, {false, 0, 0, static_cast<T>(0)});
    this->inQueue_.template EnQue(target);
}

template <typename T, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingAAndB<T, IsBoardCast>::ComputeAndQuantize(
    LocalTensor<T>& inUb, LocalTensor<T>& cos, LocalTensor<T>& sin, LocalTensor<int32_t>& yQuantLocal,
    LocalTensor<uint16_t>& scaleLocal, int64_t bLength, int64_t sliceLen)
{
    LocalTensor<float> outUb = inUb.template ReinterpretCast<float>();
    if constexpr (IsSameType<T, float>::value) {
        outUb = ropeBuf_.Get<float>();
    }
    __local_mem__ T* inAddr = (__local_mem__ T*)inUb.GetPhyAddr();
    __local_mem__ T* sinAddr = (__local_mem__ T*)sin.GetPhyAddr();
    __local_mem__ T* cosAddr = (__local_mem__ T*)cos.GetPhyAddr();
    __local_mem__ bfloat16_t* outAddr = (__local_mem__ bfloat16_t*)outUb.GetPhyAddr();
    if constexpr (IsBoardCast) {
        if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::HALF)) {
            HalfAlignVF<T>(sinAddr, cosAddr, inAddr, outAddr, sliceLen, dAlign_, sliceAlign_, dAlign_, 1, bLength,
                           tilingData_->sliceStart);
        } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::INTERLEAVE)) {
            InterleaveModeVF<T>(sinAddr, cosAddr, inAddr, outAddr, sliceLen, dAlign_, sliceAlign_, dAlign_, 1, bLength,
                                tilingData_->sliceStart);
        } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::QUARTER)) {
        } else {
        }
    } else {
        if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::HALF)) {
            BatchHalfAlignVF<T, IsBoardCast>(inAddr, cosAddr, sinAddr, outAddr, bLength, 1, 1, sliceLen, dAlign_,
                                             sliceAlign_, dAlign_, ubFactorB_, 1, tilingData_->sliceStart);
        } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::INTERLEAVE)) {
            BatchInterleaveModeVF<T, IsBoardCast>(inAddr, cosAddr, sinAddr, outAddr, bLength, 1, 1, sliceLen, dAlign_,
                                                  sliceAlign_, dAlign_, ubFactorB_, 1, tilingData_->sliceStart);
        } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::QUARTER)) {
        } else {
        }
    }
    LocalTensor<uint8_t> eaLocal = eaBuf_.Get<uint8_t>();
    LocalTensor<uint8_t> ebLocal = ebBuf_.Get<uint8_t>();
    LocalTensor<uint8_t> ecLocal = ecBuf_.Get<uint8_t>();

    VFProcessSwigluHiF4InvScaleAndQuant(yQuantLocal, outUb, eaLocal, ebLocal.ReinterpretCast<uint32_t>(), ecLocal,
                                        bLength * D_);
    VFProcessSwigluGroupQuantScale(scaleLocal, eaLocal, ebLocal, ecLocal.ReinterpretCast<uint16_t>(), bLength * D_);

    outQueue_.template EnQue(yQuantLocal);
    scaleQueue_.template EnQue(scaleLocal);
}

template <typename T, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingAAndB<T, IsBoardCast>::CopyOut(int64_t bStart, int64_t bLength,
                                                                             int64_t dCount)
{
    LocalTensor<uint8_t> yQuantLocal = outQueue_.template DeQue<uint8_t>();
    LocalTensor<float> scaleLocal = scaleQueue_.template DeQue<float>();
    DataCopyExtParams outParams = {1, static_cast<uint32_t>(bLength * dCount * sizeof(uint8_t) / 2), 0, 0, 0};
    DataCopyExtParams scaleParams = {1, static_cast<uint32_t>(bLength * dCount * sizeof(float) / 64), 0, 0, 0};
    DataCopyPad(qOutGm_[(bStart * D_) / 2], yQuantLocal, outParams);
    DataCopyPad(scaleGm_[(bStart * D_) / 64], scaleLocal, scaleParams);

    outQueue_.FreeTensor(yQuantLocal);
    scaleQueue_.FreeTensor(scaleLocal);
}

} // namespace PartialRotaryMulQuantOps

#endif
