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
 * \file rotary_position_embbeding_grad_rotary_x.h
 * \brief
 */

#ifndef __ROTARY_POSITION_EMBEDDING_GRAD_ROTARY_X_H__
#define __ROTARY_POSITION_EMBEDDING_GRAD_ROTARY_X_H__

#include "rotary_position_embedding_grad_rotary_x_vf.h"
#include "rotary_position_embedding_grad_tiling_data.h"

namespace RotaryPositionEmbeddingGrad {
using namespace AscendC;

template <typename T>
class RopeGradRotaryX {
public:
    __aicore__ inline RopeGradRotaryX(TPipe *pipe, const RotaryXParams *tiling)
        : pipe_(pipe),
          tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR workspace);
    __aicore__ inline void Process();

private:
    constexpr static int32_t bufferNum = 2;
    const RotaryXParams *tilingData_;
    TPipe *pipe_;
    int64_t blockIdx_ = 0;
    int64_t dSplitCoef_ = 1; // 切分系数初始化为1
    uint32_t dSplitSize_ = 0;
    int64_t dAlign_ = 0;
    int64_t ubFactorN_ = 0;
    int64_t workSpaceSize_ = 0;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> xRotaryWorkSpace_;    // dcos
    GlobalTensor<T> xRotary2ndWorkSpace_; // dsin
    TQue<QuePosition::VECIN, 1> xInQue_;
    TQue<QuePosition::VECOUT, 1> xRotaryOutQue_;
    TQue<QuePosition::VECOUT, 1> xRotary2ndOutQue_;

private:
    __aicore__ inline void ComputeRotaryX(const LocalTensor<T> &xTensor, const LocalTensor<T> &xRotaryTensor,
                                          const uint32_t currDNum);
};

template <typename T>
__aicore__ inline void RopeGradRotaryX<T>::Init(GM_ADDR x, GM_ADDR workspace)
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }
    blockIdx_ = GetBlockIdx();
    if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::HALF) ||
        tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::DEEPSEEK_INTERLEAVE)) {
        this->dSplitCoef_ = HALF_INTERLEAVE_COEF;
    } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::QUARTER)) {
        this->dSplitCoef_ = QUARTER_MODE_COEF;
    }
    workSpaceSize_ = tilingData_->b * tilingData_->s * tilingData_->n * tilingData_->d;
    dSplitSize_ = tilingData_->d / dSplitCoef_ * sizeof(T);
    dAlign_ = Ops::Base::CeilAlign<int64_t>(tilingData_->d / dSplitCoef_, BLOCK_TYPE_SIZE / sizeof(T)) * dSplitCoef_;
    int64_t gmOffset = blockIdx_ * tilingData_->blockFactorN * tilingData_->d;
    ubFactorN_ = tilingData_->ubFactorN;
    xGm_.SetGlobalBuffer((__gm__ T *)x + gmOffset);
    xRotaryWorkSpace_.SetGlobalBuffer((__gm__ T *)workspace + gmOffset, workSpaceSize_);
    pipe_->InitBuffer(xInQue_, bufferNum, ubFactorN_ * dAlign_ * sizeof(T));
    pipe_->InitBuffer(xRotaryOutQue_, bufferNum, ubFactorN_ * dAlign_ * sizeof(T));
    if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::DEEPSEEK_INTERLEAVE)) {
        xRotary2ndWorkSpace_.SetGlobalBuffer((__gm__ T *)workspace + workSpaceSize_ + gmOffset, workSpaceSize_);
        pipe_->InitBuffer(xRotary2ndOutQue_, bufferNum, ubFactorN_ * dAlign_ * sizeof(T));
    }
}

template <typename T>
__aicore__ inline void RopeGradRotaryX<T>::Process()
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }
    LocalTensor<T> xTensor;
    LocalTensor<T> xRotaryTensor;
    LocalTensor<T> xRotary2ndTensor;
    int64_t currBlockFactorN =
        (blockIdx_ != tilingData_->usedCoreNum - 1) ? tilingData_->blockFactorN : tilingData_->blockTailFactorN;
    uint32_t loopCnt = Ops::Base::CeilDiv(currBlockFactorN, tilingData_->ubFactorN);
    for (uint32_t loopIdx = 0; loopIdx < loopCnt; loopIdx++) {
        uint32_t currDNum =
            (loopIdx != loopCnt - 1) ? tilingData_->ubFactorN : currBlockFactorN - loopIdx * tilingData_->ubFactorN;
        int64_t offset = loopIdx * tilingData_->ubFactorN * tilingData_->d;
        DataCopyExtParams copyInParams{static_cast<uint16_t>(currDNum * dSplitCoef_), dSplitSize_, 0, 0, 0};
        DataCopyExtParams copyOutParams{static_cast<uint16_t>(currDNum * dSplitCoef_), dSplitSize_, 0, 0, 0};
        if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::DEEPSEEK_INTERLEAVE)) {
            // deepSeekInterleave 场景，copyIn时按照D对齐，copyOut 时，按照 d/2 对齐
            copyInParams = {static_cast<uint16_t>(currDNum), tilingData_->d * sizeof(T), 0, 0, 0};
        }
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        xTensor = xInQue_.AllocTensor<T>();
        DataCopyPad(xTensor, xGm_[offset], copyInParams, padParams);
        xInQue_.EnQue(xTensor);
        xTensor = xInQue_.DeQue<T>();
        xRotaryTensor = xRotaryOutQue_.AllocTensor<T>();
        ComputeRotaryX(xTensor, xRotaryTensor, currDNum);
        xRotaryOutQue_.EnQue(xRotaryTensor);
        xRotaryTensor = xRotaryOutQue_.DeQue<T>();
        DataCopyPad(xRotaryWorkSpace_[offset], xRotaryTensor, copyOutParams);
        xRotaryOutQue_.FreeTensor(xRotaryTensor);
        if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::DEEPSEEK_INTERLEAVE)) {
            xRotary2ndTensor = xRotary2ndOutQue_.AllocTensor<T>();
            DSDSinInterleaveHalfVF<T>(xTensor, xRotary2ndTensor, tilingData_->d, currDNum);
            xRotary2ndOutQue_.EnQue(xRotary2ndTensor);
            xRotary2ndTensor = xRotary2ndOutQue_.DeQue<T>();
            DataCopyPad(xRotary2ndWorkSpace_[offset], xRotary2ndTensor, copyOutParams);
            xRotary2ndOutQue_.FreeTensor(xRotary2ndTensor);
        }
        xInQue_.FreeTensor(xTensor);
    }
}

template <typename T>
__aicore__ inline void RopeGradRotaryX<T>::ComputeRotaryX(const LocalTensor<T> &xTensor,
                                                          const LocalTensor<T> &xRotaryTensor, const uint32_t currDNum)
{
    if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::HALF)) {
        HalfRotaryVF<T>(xTensor, xRotaryTensor, tilingData_->d, dAlign_, currDNum);
    } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::INTERLEAVE)) {
        InterleaveRotaryVF<T>(xTensor, xRotaryTensor, tilingData_->d, currDNum);
    } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::QUARTER)) {
        QuarterRotaryVF<T>(xTensor, xRotaryTensor, tilingData_->d, dAlign_, currDNum);
    } else {
        DSCosInterleaveHalfVF<T>(xTensor, xRotaryTensor, tilingData_->d, currDNum);
    }
}

} // namespace RotaryPositionEmbeddingGrad
#endif // __ROTARY_POSITION_EMBEDDING_GRAD_ROTARY_X_H__
