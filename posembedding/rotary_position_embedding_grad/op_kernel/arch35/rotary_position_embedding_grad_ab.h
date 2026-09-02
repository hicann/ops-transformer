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
 * \file rotary_position_embedding_grad_ab.h
 * \brief
 */
#ifndef ROTARY_POSITION_EMBEDDING_GRAD_AB_H
#define ROTARY_POSITION_EMBEDDING_GRAD_AB_H

#include "op_kernel/math_util.h"
#include "op_kernel/load_store_utils.h"
#include "rotary_position_embedding_grad_common.h"
#include "rotary_position_embedding_grad_tiling_data.h"

namespace RotaryPositionEmbeddingGrad {
using namespace AscendC;

template <typename T>
class RotaryPositionEmbeddingGradAB {
public:
    __aicore__ inline RotaryPositionEmbeddingGradAB(TPipe *pipe, const RopeGradRegbaseABParams *tiling)
        : pipe_(pipe),
          tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR cos, GM_ADDR sin, GM_ADDR xGrad);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessLoop(int64_t gradGmOffset, LocalTensor<T> cosBuffer, LocalTensor<T> sinBuffer,
                                       int64_t ubIdx, int64_t bsCount, int64_t nCount);

private:
    TPipe *pipe_;
    TQue<QuePosition::VECIN, 1> gradInQueue_;
    TQue<QuePosition::VECIN, 1> cosInQueue_;
    TQue<QuePosition::VECIN, 1> sinInQueue_;
    TQue<QuePosition::VECOUT, 1> xGradOutQueue_;

    GlobalTensor<T> gradGm_;
    GlobalTensor<T> cosGm_;
    GlobalTensor<T> sinGm_;
    GlobalTensor<T> xGradOutGm_;
    const RopeGradRegbaseABParams *tilingData_;
    DataCopyPadExtParams<T> padParams_ = {false, 0, 0, static_cast<T>(0)};
    uint8_t DB_FLAG = 2;
    uint32_t dSplitSize_ = 0;
    int64_t bsBlockCount_ = 0;
    int64_t nBlockCount_ = 0;
};

template <typename T>
__aicore__ inline void RotaryPositionEmbeddingGradAB<T>::Init(GM_ADDR grad, GM_ADDR cos, GM_ADDR sin, GM_ADDR xGradOut)
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }
    dSplitSize_ = tilingData_->d / tilingData_->dSplitCoef * sizeof(T);
    int64_t blockDimBS = GetBlockIdx() / tilingData_->blockNumN;
    int64_t blockDimN = GetBlockIdx() % tilingData_->blockNumN;
    bsBlockCount_ = (blockDimBS == tilingData_->blockNumBS - 1) ? tilingData_->blockTailBS : tilingData_->blockFactorBS;
    nBlockCount_ = (blockDimN == tilingData_->blockNumN - 1) ? tilingData_->blockTailN : tilingData_->blockFactorN;

    int64_t cosOffset = blockDimBS * tilingData_->blockFactorBS * tilingData_->d;
    int64_t gradOffset = cosOffset * tilingData_->n + blockDimN * tilingData_->blockFactorN * tilingData_->d;
    this->cosGm_.SetGlobalBuffer((__gm__ T *)cos + cosOffset);
    this->sinGm_.SetGlobalBuffer((__gm__ T *)sin + cosOffset);
    this->gradGm_.SetGlobalBuffer((__gm__ T *)grad + gradOffset);
    this->xGradOutGm_.SetGlobalBuffer((__gm__ T *)xGradOut + gradOffset);

    int64_t bufferSize = tilingData_->dAlign * sizeof(T) * tilingData_->ubFactorBS;
    pipe_->InitBuffer(gradInQueue_, DB_FLAG, bufferSize * tilingData_->ubFactorN);
    pipe_->InitBuffer(cosInQueue_, DB_FLAG, bufferSize);
    pipe_->InitBuffer(sinInQueue_, DB_FLAG, bufferSize);
    pipe_->InitBuffer(xGradOutQueue_, DB_FLAG, bufferSize * tilingData_->ubFactorN);
}

template <typename T>
__aicore__ inline void RotaryPositionEmbeddingGradAB<T>::Process()
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }
    uint32_t bsLoopCnt = Ops::Base::CeilDiv(bsBlockCount_, tilingData_->ubFactorBS);
    uint32_t nLoopCnt = Ops::Base::CeilDiv(nBlockCount_, tilingData_->ubFactorN);
    for (uint32_t bsLoopIdx = 0; bsLoopIdx < bsLoopCnt; bsLoopIdx++) {
        int64_t xGmOffset = bsLoopIdx * tilingData_->ubFactorBS * tilingData_->n * tilingData_->d;
        uint32_t currBSNum = (bsLoopIdx != bsLoopCnt - 1) ? tilingData_->ubFactorBS :
                                                            bsBlockCount_ - (bsLoopIdx * tilingData_->ubFactorBS);

        DataCopyExtParams cosParams = {static_cast<uint16_t>(currBSNum * tilingData_->dSplitCoef), dSplitSize_, 0, 0,
                                       0};
        LocalTensor<T> cosBuffer = cosInQueue_.AllocTensor<T>();
        LocalTensor<T> sinBuffer = sinInQueue_.AllocTensor<T>();

        DataCopyPad(cosBuffer, cosGm_[bsLoopIdx * tilingData_->ubFactorBS * tilingData_->d], cosParams, padParams_);
        cosInQueue_.EnQue(cosBuffer);
        cosBuffer = cosInQueue_.DeQue<T>();
        DataCopyPad(sinBuffer, sinGm_[bsLoopIdx * tilingData_->ubFactorBS * tilingData_->d], cosParams, padParams_);
        sinInQueue_.EnQue(sinBuffer);
        sinBuffer = sinInQueue_.DeQue<T>();

        for (int64_t nLoopIdx = 0; nLoopIdx < nLoopCnt; nLoopIdx++) {
            int64_t currNNum = (nLoopIdx != nLoopCnt - 1) ? tilingData_->ubFactorN :
                                                            nBlockCount_ - (nLoopIdx * tilingData_->ubFactorN);
            ProcessLoop(xGmOffset, cosBuffer, sinBuffer, nLoopIdx, currBSNum, currNNum);
        }

        cosInQueue_.FreeTensor(cosBuffer);
        sinInQueue_.FreeTensor(sinBuffer);
    }
}

template <typename T>
__aicore__ inline void RotaryPositionEmbeddingGradAB<T>::ProcessLoop(int64_t gradGmOffset, LocalTensor<T> cosBuffer,
                                                                     LocalTensor<T> sinBuffer, int64_t ubIdx,
                                                                     int64_t bsCount, int64_t nCount)
{
    int64_t totalCount = bsCount * nCount;
    DataCopyExtParams inParams = {static_cast<uint16_t>(totalCount * tilingData_->dSplitCoef), dSplitSize_, 0, 0, 0};
    DataCopyExtParams outParams = {static_cast<uint16_t>(totalCount * tilingData_->dSplitCoef), dSplitSize_, 0, 0, 0};
    if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::DEEPSEEK_INTERLEAVE)) {
        outParams = {static_cast<uint16_t>(totalCount), tilingData_->d * sizeof(T), 0, 0, 0};
    }

    LocalTensor<T> inBuffer = gradInQueue_.AllocTensor<T>();
    LocalTensor<T> outBuffer = xGradOutQueue_.AllocTensor<T>();

    DataCopyPad(inBuffer, gradGm_[gradGmOffset + ubIdx * tilingData_->ubFactorN * tilingData_->d], inParams,
                padParams_);

    gradInQueue_.EnQue(inBuffer);
    inBuffer = gradInQueue_.DeQue<T>();

    if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::HALF)) {
        HalfGradAlignVF(sinBuffer, cosBuffer, inBuffer, outBuffer, tilingData_->d, tilingData_->dAlign, bsCount,
                        nCount);
    } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::INTERLEAVE)) {
        InterleaveModeGradVF(sinBuffer, cosBuffer, inBuffer, outBuffer, tilingData_->d, bsCount, nCount);
    } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::QUARTER)) {
        QuarterGradAlignVF(sinBuffer, cosBuffer, inBuffer, outBuffer, tilingData_->d, tilingData_->dAlign, bsCount,
                           nCount);
    } else {
        DeepSeekInterleaveModeGradVF<T>(sinBuffer, cosBuffer, inBuffer, outBuffer, tilingData_->d, bsCount, nCount);
    }

    xGradOutQueue_.EnQue(outBuffer);
    outBuffer = xGradOutQueue_.DeQue<T>();
    gradInQueue_.FreeTensor(inBuffer);

    DataCopyPad(xGradOutGm_[gradGmOffset + ubIdx * tilingData_->ubFactorN * tilingData_->d], outBuffer, outParams);

    xGradOutQueue_.FreeTensor(outBuffer);
}

} // namespace RotaryPositionEmbeddingGrad

#endif // ROTARY_POSITION_EMBEDDING_GRAD_AB_H
