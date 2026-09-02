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
 * \file rotary_position_embbeding_grad_bab.h
 * \brief
 */

#ifndef ROTARY_POSITION_EMBEDDING_GRAD_BAB_H
#define ROTARY_POSITION_EMBEDDING_GRAD_BAB_H

#include "rotary_position_embedding_grad_common.h"
#include "rotary_position_embedding_grad_tiling_data.h"

namespace RotaryPositionEmbeddingGrad {
using namespace AscendC;

template <typename T>
class RotaryPositionEmbeddingGradBAB {
public:
    __aicore__ inline RotaryPositionEmbeddingGradBAB(TPipe *pipe, const RopeGradRegbaseParams *tiling)
        : pipe_(pipe),
          tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR cos, GM_ADDR sin, GM_ADDR xGrad);
    __aicore__ inline void Process();

private:
    constexpr static int32_t bufferNum = 2;
    const RopeGradRegbaseParams *tilingData_;
    TPipe *pipe_;
    int64_t blockIdx_ = 0;
    int64_t dSplitCoef_ = 1; // 切分系数初始化为1
    uint32_t dSplitSize_ = 0;
    int64_t dAlign_ = 0;
    int64_t bIdx_ = 0;
    int64_t sIdx_ = 0;
    int64_t bNum_ = 0;
    int64_t sNum_ = 0;
    int64_t ubFactorS_ = 0;
    int64_t ubFactorN_ = 0;
    GlobalTensor<T> gradGm_;
    GlobalTensor<T> cosGm_;
    GlobalTensor<T> sinGm_;
    GlobalTensor<T> xGradOutGm_;

    TQue<QuePosition::VECIN, 1> gradInQue_;
    TQue<QuePosition::VECIN, 1> cosInQue_;
    TQue<QuePosition::VECIN, 1> sinInQue_;
    TQue<QuePosition::VECOUT, 1> xGradOutQue_;

private:
    __aicore__ inline void PrePareParams();
    __aicore__ inline void ProcessNLoop(const uint32_t bIdx, const uint32_t sIdx, const uint32_t currSNum);
    __aicore__ inline void Compute(const LocalTensor<T> &sinTensor, const LocalTensor<T> &cosTensor,
                                   const LocalTensor<T> &inTensor, const LocalTensor<T> &outTensor,
                                   const uint32_t currSNum, const uint32_t currDNum);
    __aicore__ inline void ProcessN(const LocalTensor<T> &sinTensor, const LocalTensor<T> &cosTensor,
                                    const uint32_t bIdx, const uint32_t sIdx, const uint32_t currSNum);
};

template <typename T>
__aicore__ inline void RotaryPositionEmbeddingGradBAB<T>::Init(GM_ADDR grad, GM_ADDR cos, GM_ADDR sin, GM_ADDR xGrad)
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }
    this->blockIdx_ = GetBlockIdx();
    if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::HALF) ||
        tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::DEEPSEEK_INTERLEAVE)) {
        this->dSplitCoef_ = HALF_INTERLEAVE_COEF;
    } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::QUARTER)) {
        this->dSplitCoef_ = QUARTER_MODE_COEF;
    }
    this->dSplitSize_ = tilingData_->d / dSplitCoef_ * sizeof(T);
    this->dAlign_ =
        Ops::Base::CeilAlign<int64_t>(tilingData_->d / dSplitCoef_, BLOCK_TYPE_SIZE / sizeof(T)) * dSplitCoef_;
    ubFactorN_ = tilingData_->ubFactorN;
    ubFactorS_ = tilingData_->ubFactorS;
    this->gradGm_.SetGlobalBuffer((__gm__ T *)grad);
    this->cosGm_.SetGlobalBuffer((__gm__ T *)cos);
    this->sinGm_.SetGlobalBuffer((__gm__ T *)sin);
    this->xGradOutGm_.SetGlobalBuffer((__gm__ T *)xGrad);
    this->pipe_->InitBuffer(gradInQue_, bufferNum, ubFactorS_ * ubFactorN_ * dAlign_ * sizeof(T));
    this->pipe_->InitBuffer(cosInQue_, bufferNum, ubFactorS_ * dAlign_ * sizeof(T));
    this->pipe_->InitBuffer(sinInQue_, bufferNum, ubFactorS_ * dAlign_ * sizeof(T));
    this->pipe_->InitBuffer(xGradOutQue_, bufferNum, ubFactorS_ * ubFactorN_ * dAlign_ * sizeof(T));
}

template <typename T>
__aicore__ inline void RotaryPositionEmbeddingGradBAB<T>::PrePareParams()
{
    bIdx_ = blockIdx_ % tilingData_->blockNumB;
    sIdx_ = blockIdx_ / tilingData_->blockNumB;
    bNum_ = tilingData_->blockFactorB;
    sNum_ = tilingData_->blockFactorS;
    if (bIdx_ == tilingData_->blockNumB - 1 && tilingData_->b % tilingData_->blockFactorB != 0) {
        bNum_ = tilingData_->b % tilingData_->blockFactorB;
    }
    if (sIdx_ == tilingData_->blockNumS - 1 && tilingData_->s % tilingData_->blockFactorS != 0) {
        sNum_ = tilingData_->s % tilingData_->blockFactorS;
    }
}

template <typename T>
__aicore__ inline void RotaryPositionEmbeddingGradBAB<T>::Process()
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }
    PrePareParams();
    uint32_t bIdxStart = bIdx_ * tilingData_->blockFactorB;
    for (uint32_t bIdx = bIdxStart; bIdx < bIdxStart + bNum_; bIdx++) {
        uint32_t sIdxStart = sIdx_ * tilingData_->blockFactorS;
        uint32_t sLoopCnt = Ops::Base::CeilDiv(sNum_, ubFactorS_);
        for (uint32_t loopIdx = 0; loopIdx < sLoopCnt; loopIdx++) {
            uint32_t currSNum = (loopIdx != sLoopCnt - 1) ? ubFactorS_ : sNum_ - loopIdx * ubFactorS_;
            ProcessNLoop(bIdx, sIdxStart + loopIdx * ubFactorS_, currSNum);
        }
    }
}

template <typename T>
__aicore__ inline void RotaryPositionEmbeddingGradBAB<T>::ProcessNLoop(const uint32_t bIdx, const uint32_t sIdx,
                                                                       const uint32_t currSNum)
{
    LocalTensor<T> sinTensor = sinInQue_.AllocTensor<T>();
    LocalTensor<T> cosTensor = cosInQue_.AllocTensor<T>();
    int64_t offset = sIdx * tilingData_->d;
    DataCopyExtParams copyParams{static_cast<uint16_t>(currSNum * dSplitCoef_), dSplitSize_, 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(sinTensor, sinGm_[offset], copyParams, padParams);
    DataCopyPad(cosTensor, cosGm_[offset], copyParams, padParams);
    sinInQue_.EnQue(sinTensor);
    cosInQue_.EnQue(cosTensor);
    sinTensor = sinInQue_.DeQue<T>();
    cosTensor = cosInQue_.DeQue<T>();
    ProcessN(sinTensor, cosTensor, bIdx, sIdx, currSNum);
    sinInQue_.FreeTensor(sinTensor);
    cosInQue_.FreeTensor(cosTensor);
}

template <typename T>
__aicore__ inline void RotaryPositionEmbeddingGradBAB<T>::ProcessN(const LocalTensor<T> &sinTensor,
                                                                   const LocalTensor<T> &cosTensor, const uint32_t bIdx,
                                                                   const uint32_t sIdx, const uint32_t currSNum)
{
    LocalTensor<T> gradInTensor;
    LocalTensor<T> xGradOutTensor;
    int64_t baseOffset = (bIdx * tilingData_->s + sIdx) * tilingData_->n * tilingData_->d;
    for (uint32_t idxN = 0; idxN < tilingData_->ubLoopNumN; idxN++) {
        int64_t currDNum = (idxN == tilingData_->ubLoopNumN - 1) ? tilingData_->ubTailFactorN : ubFactorN_;
        int64_t offset = baseOffset + idxN * ubFactorN_ * tilingData_->d;
        gradInTensor = gradInQue_.AllocTensor<T>();
        DataCopyExtParams copyInParams{static_cast<uint16_t>(currSNum * currDNum * dSplitCoef_), dSplitSize_, 0, 0, 0};
        DataCopyExtParams copyOutParams{static_cast<uint16_t>(currSNum * currDNum * dSplitCoef_), dSplitSize_, 0, 0, 0};
        // deepSeekInterleave 场景，copyIn时按照D/2对齐，copyOut 时，按照 D 对齐
        if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::DEEPSEEK_INTERLEAVE)) {
            copyOutParams = {static_cast<uint16_t>(currSNum * currDNum), tilingData_->d * sizeof(T), 0, 0, 0};
        }
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(gradInTensor, gradGm_[offset], copyInParams, padParams);
        gradInQue_.EnQue(gradInTensor);
        gradInTensor = gradInQue_.DeQue<T>();
        xGradOutTensor = xGradOutQue_.AllocTensor<T>();
        Compute(sinTensor, cosTensor, gradInTensor, xGradOutTensor, currSNum, currDNum);
        gradInQue_.FreeTensor(gradInTensor);
        xGradOutQue_.EnQue(xGradOutTensor);
        xGradOutTensor = xGradOutQue_.DeQue<T>();
        DataCopyPad(xGradOutGm_[offset], xGradOutTensor, copyOutParams);
        xGradOutQue_.FreeTensor(xGradOutTensor);
    }
}

template <typename T>
__aicore__ inline void RotaryPositionEmbeddingGradBAB<T>::Compute(const LocalTensor<T> &sinTensor,
                                                                  const LocalTensor<T> &cosTensor,
                                                                  const LocalTensor<T> &inTensor,
                                                                  const LocalTensor<T> &outTensor,
                                                                  const uint32_t currSNum, const uint32_t currDNum)
{
    if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::HALF)) {
        HalfGradAlignVF<T>(sinTensor, cosTensor, inTensor, outTensor, tilingData_->d, dAlign_, currSNum, currDNum);
    } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::INTERLEAVE)) {
        InterleaveModeGradVF<T>(sinTensor, cosTensor, inTensor, outTensor, tilingData_->d, currSNum, currDNum);
    } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::QUARTER)) {
        QuarterGradAlignVF<T>(sinTensor, cosTensor, inTensor, outTensor, tilingData_->d, dAlign_, currSNum, currDNum);
    } else {
        DeepSeekInterleaveModeGradVF<T>(sinTensor, cosTensor, inTensor, outTensor, tilingData_->d, currSNum, currDNum);
    }
}

} // namespace RotaryPositionEmbeddingGrad
#endif // ROTARY_POSITION_EMBEDDING_GRAD_BAB_H
