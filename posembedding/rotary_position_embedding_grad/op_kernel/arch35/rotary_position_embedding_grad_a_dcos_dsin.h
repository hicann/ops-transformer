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
 * \file rotary_position_embbeding_grad_a_dcos_dsin.h
 * \brief
 */

#ifndef __ROTARY_POSITION_EMBEDDING_GRAD_A_DCOS_DSIN_H__
#define __ROTARY_POSITION_EMBEDDING_GRAD_A_DCOS_DSIN_H__

#include "rotary_position_embedding_grad_tiling_data.h"

namespace RotaryPositionEmbeddingGrad {
using namespace AscendC;

template <typename T>
class ComputeADcosDsin {
public:
    __aicore__ inline ComputeADcosDsin(TPipe *pipe, const RotaryXParams *tiling)
        : pipe_(pipe),
          tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR xCos, GM_ADDR xSin, GM_ADDR cosGrad, GM_ADDR sinGrad);
    __aicore__ inline void Process();

private:
    constexpr static int32_t bufferNum = 2;
    const RotaryXParams *tilingData_;
    TPipe *pipe_;
    int64_t blockIdx_ = 0;
    int64_t ubFactorN_ = 0;
    GlobalTensor<T> gradGm_;
    GlobalTensor<T> xCosGm_;
    GlobalTensor<T> xSinGm_;
    GlobalTensor<T> cosGradGm_;
    GlobalTensor<T> sinGradGm_;
    TQue<QuePosition::VECIN, 1> gradInQue_;
    TQue<QuePosition::VECIN, 1> xInQue_;
    TQue<QuePosition::VECOUT, 1> cosSinGradOutQue_;

private:
    __aicore__ inline void ComputeDcos(const int64_t offset, const uint32_t currDNum);
    __aicore__ inline void ComputeDsin(const int64_t offset, const uint32_t currDNum);
};

template <typename T>
__aicore__ inline void ComputeADcosDsin<T>::Init(GM_ADDR grad, GM_ADDR xCos, GM_ADDR xSin, GM_ADDR cosGrad,
                                                 GM_ADDR sinGrad)
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }
    blockIdx_ = GetBlockIdx();
    int64_t gmOffset = blockIdx_ * tilingData_->blockFactorN * tilingData_->d;
    ubFactorN_ = tilingData_->ubFactorN;
    gradGm_.SetGlobalBuffer((__gm__ T *)grad + gmOffset);
    xCosGm_.SetGlobalBuffer((__gm__ T *)xCos + gmOffset);
    xSinGm_.SetGlobalBuffer((__gm__ T *)xSin + gmOffset);
    cosGradGm_.SetGlobalBuffer((__gm__ T *)cosGrad + gmOffset);
    sinGradGm_.SetGlobalBuffer((__gm__ T *)sinGrad + gmOffset);
    pipe_->InitBuffer(gradInQue_, bufferNum, ubFactorN_ * tilingData_->d * sizeof(T));
    pipe_->InitBuffer(xInQue_, bufferNum, ubFactorN_ * tilingData_->d * sizeof(T));
    pipe_->InitBuffer(cosSinGradOutQue_, bufferNum, ubFactorN_ * tilingData_->d * sizeof(T));
}

template <typename T>
__aicore__ inline void ComputeADcosDsin<T>::Process()
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }
    int64_t currBlockFactorN =
        (blockIdx_ != tilingData_->usedCoreNum - 1) ? tilingData_->blockFactorN : tilingData_->blockTailFactorN;
    uint32_t loopCnt = Ops::Base::CeilDiv(currBlockFactorN, tilingData_->ubFactorN);
    for (uint32_t loopIdx = 0; loopIdx < loopCnt; loopIdx++) {
        uint32_t currDNum =
            (loopIdx != loopCnt - 1) ? tilingData_->ubFactorN : currBlockFactorN - loopIdx * tilingData_->ubFactorN;
        int64_t offset = loopIdx * tilingData_->ubFactorN * tilingData_->d;
        ComputeDcos(offset, currDNum);
        ComputeDsin(offset, currDNum);
    }
}

template <typename T>
__aicore__ inline void ComputeADcosDsin<T>::ComputeDcos(const int64_t offset, const uint32_t currDNum)
{
    LocalTensor<T> xCosTensor;
    LocalTensor<T> gradTensor;
    LocalTensor<T> cosGradTensor;
    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(currDNum * tilingData_->d * sizeof(T)), 0, 0, 0};
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(currDNum * tilingData_->d * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    gradTensor = gradInQue_.AllocTensor<T>();
    DataCopyPad(gradTensor, gradGm_[offset], copyInParams, padParams);
    gradInQue_.EnQue(gradTensor);
    gradTensor = gradInQue_.DeQue<T>();

    xCosTensor = xInQue_.AllocTensor<T>();
    DataCopyPad(xCosTensor, xCosGm_[offset], copyInParams, padParams);
    xInQue_.EnQue(xCosTensor);
    xCosTensor = xInQue_.DeQue<T>();

    cosGradTensor = cosSinGradOutQue_.AllocTensor<T>();
    Mul(cosGradTensor, xCosTensor, gradTensor, currDNum * tilingData_->d);
    gradInQue_.FreeTensor(gradTensor);
    xInQue_.FreeTensor(xCosTensor);
    cosSinGradOutQue_.EnQue(cosGradTensor);
    cosGradTensor = cosSinGradOutQue_.DeQue<T>();

    DataCopyPad(cosGradGm_[offset], cosGradTensor, copyOutParams);
    cosSinGradOutQue_.FreeTensor(cosGradTensor);
}

template <typename T>
__aicore__ inline void ComputeADcosDsin<T>::ComputeDsin(const int64_t offset, const uint32_t currDNum)
{
    LocalTensor<T> xSinTensor;
    LocalTensor<T> gradTensor;
    LocalTensor<T> sinGradTensor;
    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(currDNum * tilingData_->d * sizeof(T)), 0, 0, 0};
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(currDNum * tilingData_->d * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    gradTensor = gradInQue_.AllocTensor<T>();
    DataCopyPad(gradTensor, gradGm_[offset], copyInParams, padParams);
    gradInQue_.EnQue(gradTensor);
    gradTensor = gradInQue_.DeQue<T>();

    xSinTensor = xInQue_.AllocTensor<T>();
    DataCopyPad(xSinTensor, xSinGm_[offset], copyInParams, padParams);
    xInQue_.EnQue(xSinTensor);
    xSinTensor = xInQue_.DeQue<T>();

    sinGradTensor = cosSinGradOutQue_.AllocTensor<T>();
    Mul(sinGradTensor, xSinTensor, gradTensor, currDNum * tilingData_->d);
    gradInQue_.FreeTensor(gradTensor);
    xInQue_.FreeTensor(xSinTensor);
    cosSinGradOutQue_.EnQue(sinGradTensor);
    sinGradTensor = cosSinGradOutQue_.DeQue<T>();

    DataCopyPad(sinGradGm_[offset], sinGradTensor, copyOutParams);
    cosSinGradOutQue_.FreeTensor(sinGradTensor);
}

} // namespace RotaryPositionEmbeddingGrad
#endif // __ROTARY_POSITION_EMBEDDING_GRAD_A_DCOS_DSIN_H__
