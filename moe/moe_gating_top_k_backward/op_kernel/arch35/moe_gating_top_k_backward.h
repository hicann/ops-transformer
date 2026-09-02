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
 * \file moe_gating_top_k_backward.h
 * \brief
 */
#ifndef MOE_GATING_TOP_K_BACKWARD_H
#define MOE_GATING_TOP_K_BACKWARD_H

#include <cmath>
#include "moe_gating_top_k_backward_arch35_common.h"
#include "kernel_operator.h"
#include "op_kernel/load_store_utils.h"
#include "moe_gating_top_k_backward_struct.h"
#include "vf/moe_gating_top_k_backward_vf.h"

namespace MoeGatingTopKBackwardNs {
using namespace AscendC;
using MicroAPI::RegTensor;

template <typename T>
class MoeGatingTopKBackward {
public:
    __aicore__ inline MoeGatingTopKBackward(){};
    __aicore__ inline void Init(GM_ADDR xNorm, GM_ADDR gradY, GM_ADDR expertIdx, GM_ADDR gradX,
                                const MoeGatingTopKBackwardA5TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInGradY(int64_t loopIdx);
    __aicore__ inline void CopyInXNorm(int64_t loopIdx);
    __aicore__ inline void CopyInExpertIdx(int64_t loopIdx);
    __aicore__ inline void SigmoidRenormBackward();
    __aicore__ inline void SigmoidGrad();
    __aicore__ inline void CopyOut(int64_t loopIdx);

    TPipe *pipe_;
    TQue<QuePosition::VECIN, 2> gradYQue_;
    TQue<QuePosition::VECIN, 2> indicesQue_;
    TQue<QuePosition::VECIN, 2> xQue_;
    TQue<QuePosition::VECOUT, 1> outQue_;
    TBuf<TPosition::VECCALC> gradNormXBuf_;
    TBuf<TPosition::VECCALC> wPrimeCache_;
    GlobalTensor<float> xNormGm_;
    GlobalTensor<T> gradYGm_;
    GlobalTensor<int32_t> expertIdxGm_;
    GlobalTensor<T> gradXGm_;
    LocalTensor<float> xNormLocal_;
    LocalTensor<T> gradYLocal_;
    LocalTensor<int32_t> expertIdxLocal_;
    LocalTensor<T> gradXOutTensor_;

    int64_t blockIdx_ = 0, loopTimes_ = 0, tailRows_ = 0, curRows_ = 0;
    int64_t elementCountPerLoop_ = 0, curRowsByKAlign_ = 0, kAlign_ = 0, kAlign16_ = 0;
    const MoeGatingTopKBackwardA5TilingData *tilingData_;
};

template <typename T>
__aicore__ inline void MoeGatingTopKBackward<T>::Init(GM_ADDR xNorm, GM_ADDR gradY, GM_ADDR expertIdx, GM_ADDR gradX,
                                                      const MoeGatingTopKBackwardA5TilingData *tilingData, TPipe *tPipe)
{
    tilingData_ = tilingData;
    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();
    kAlign_ = Align(tilingData_->k, SIZE_OF_INT32);
    kAlign16_ = Align(tilingData_->k, 2);
    if (blockIdx_ == tilingData_->needCoreNum - 1) {
        loopTimes_ = tilingData_->lastLoopTimes;
        tailRows_ = tilingData_->lastTailRows;
    } else {
        loopTimes_ = tilingData_->perLoopTimes;
        tailRows_ = tilingData_->perTailRows;
    }
    xNormGm_.SetGlobalBuffer((__gm__ float *)xNorm + tilingData_->perCoreRows * tilingData_->expertCount * blockIdx_);
    gradYGm_.SetGlobalBuffer((__gm__ T *)gradY + tilingData_->perCoreRows * tilingData_->k * blockIdx_);
    expertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expertIdx + tilingData_->perCoreRows * tilingData_->k * blockIdx_);
    gradXGm_.SetGlobalBuffer((__gm__ T *)gradX + tilingData_->perCoreRows * tilingData_->expertCount * blockIdx_);
    pipe_->InitBuffer(gradYQue_, 2, tilingData_->baseRows * AlignBytes(tilingData_->k, tilingData_->gradYDtypeSize));
    pipe_->InitBuffer(indicesQue_, 2, tilingData_->baseRows * AlignBytes(tilingData_->k, SIZE_OF_INT32));
    pipe_->InitBuffer(xQue_, 2, AlignBytes(tilingData_->baseRows * tilingData_->expertCount, SIZE_OF_FLOAT32));
    pipe_->InitBuffer(outQue_, 1,
                      AlignBytes(tilingData_->baseRows * tilingData_->expertCount, tilingData_->gradYDtypeSize));
    pipe_->InitBuffer(gradNormXBuf_, AlignBytes(tilingData_->baseRows * tilingData_->expertCount, SIZE_OF_FLOAT32));
    pipe_->InitBuffer(wPrimeCache_, tilingData_->baseRows * AlignBytes(tilingData_->k, SIZE_OF_FLOAT32));
}

template <typename T>
__aicore__ inline void MoeGatingTopKBackward<T>::Process()
{
    for (int64_t loopIdx = 0; loopIdx < loopTimes_; loopIdx++) {
        curRows_ = loopIdx == loopTimes_ - 1 ? tailRows_ : tilingData_->baseRows;
        elementCountPerLoop_ = curRows_ * tilingData_->expertCount;
        curRowsByKAlign_ = curRows_ * kAlign_;
        CopyInGradY(loopIdx);
        CopyInXNorm(loopIdx);
        CopyInExpertIdx(loopIdx);
        SigmoidRenormBackward();
        SigmoidGrad();
        CopyOut(loopIdx);
    }
}

template <typename T>
__aicore__ inline void MoeGatingTopKBackward<T>::CopyInGradY(int64_t loopIdx)
{
    gradYLocal_ = gradYQue_.AllocTensor<T>();
    LocalTensor<float> wPrimeCache = wPrimeCache_.Get<float>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(curRows_);
    copyParams.blockLen = static_cast<uint32_t>(tilingData_->k) * static_cast<uint32_t>(sizeof(T));
    copyParams.srcStride = static_cast<uint32_t>(0);
    copyParams.dstStride = static_cast<uint32_t>(0);
    int64_t kAlign = tilingData_->gradYDtypeSize == 2 ? kAlign16_ : kAlign_;
    DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(kAlign - tilingData_->k), 0};
    DataCopyPad(gradYLocal_, gradYGm_[loopIdx * tilingData_->baseRows * tilingData_->k], copyParams, padParams);
    gradYQue_.EnQue(gradYLocal_);
    gradYLocal_ = gradYQue_.DeQue<T>();
    PipeBarrier<PIPE_V>();
    if constexpr (IsSameType<T, float>::value) {
        Muls(wPrimeCache, gradYLocal_, tilingData_->routedScalingFactor, curRowsByKAlign_);
    } else {
        if (kAlign16_ == kAlign_) {
            CallCastGradYFlatVF<T>(gradYLocal_, wPrimeCache, static_cast<uint32_t>(curRowsByKAlign_));
        } else {
            CallCastGradYRowsVF<T>(gradYLocal_, wPrimeCache, static_cast<uint16_t>(curRows_),
                                   static_cast<uint16_t>(tilingData_->k), static_cast<uint16_t>(kAlign16_),
                                   static_cast<uint16_t>(kAlign_));
        }
        PipeBarrier<PIPE_V>();
        Muls(wPrimeCache, wPrimeCache, tilingData_->routedScalingFactor, curRowsByKAlign_);
    }
    PipeBarrier<PIPE_V>();
    gradYQue_.FreeTensor(gradYLocal_);
}

template <typename T>
__aicore__ inline void MoeGatingTopKBackward<T>::CopyInXNorm(int64_t loopIdx)
{
    xNormLocal_ = xQue_.AllocTensor<float>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(1);
    copyParams.blockLen = static_cast<uint32_t>(curRows_ * tilingData_->expertCount * SIZE_OF_FLOAT32);
    copyParams.srcStride = static_cast<uint32_t>(0);
    copyParams.dstStride = static_cast<uint32_t>(0);
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(xNormLocal_, xNormGm_[loopIdx * tilingData_->baseRows * tilingData_->expertCount], copyParams,
                padParams);
    xQue_.EnQue(xNormLocal_);
    xNormLocal_ = xQue_.DeQue<float>();
}

template <typename T>
__aicore__ inline void MoeGatingTopKBackward<T>::CopyInExpertIdx(int64_t loopIdx)
{
    expertIdxLocal_ = indicesQue_.AllocTensor<int32_t>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(curRows_);
    copyParams.blockLen = static_cast<uint32_t>(tilingData_->k) * static_cast<uint32_t>(SIZE_OF_INT32);
    copyParams.srcStride = static_cast<uint32_t>(0);
    copyParams.dstStride = static_cast<uint32_t>(0);
    DataCopyPadExtParams<int32_t> padParams{true, 0, static_cast<uint8_t>(kAlign_ - tilingData_->k), 0};
    DataCopyPad(expertIdxLocal_, expertIdxGm_[loopIdx * tilingData_->baseRows * tilingData_->k], copyParams, padParams);
    indicesQue_.EnQue(expertIdxLocal_);
    expertIdxLocal_ = indicesQue_.DeQue<int32_t>();
}

template <typename T>
__aicore__ inline void MoeGatingTopKBackward<T>::SigmoidRenormBackward()
{
    PipeBarrier<PIPE_V>();
    LocalTensor<float> wCache = wPrimeCache_.Get<float>();
    LocalTensor<float> gradNormX = gradNormXBuf_.Get<float>();
    Duplicate<float>(gradNormX, 0.0f, elementCountPerLoop_);
    PipeBarrier<PIPE_V>();
    CallSigmoidRenormBackwardVF(xNormLocal_, expertIdxLocal_, wCache, gradNormX, tilingData_->eps,
                                static_cast<uint16_t>(curRows_), static_cast<uint16_t>(tilingData_->k),
                                static_cast<uint16_t>(kAlign_), static_cast<uint16_t>(tilingData_->expertCount));
    PipeBarrier<PIPE_V>();
    indicesQue_.FreeTensor(expertIdxLocal_);
}

template <typename T>
__aicore__ inline void MoeGatingTopKBackward<T>::SigmoidGrad()
{
    gradXOutTensor_ = outQue_.AllocTensor<T>();
    LocalTensor<float> gradNormX = gradNormXBuf_.Get<float>();
    CallSigmoidGradVF<T>(gradXOutTensor_, xNormLocal_, gradNormX, static_cast<uint32_t>(elementCountPerLoop_));
    outQue_.EnQue(gradXOutTensor_);
    xQue_.FreeTensor(xNormLocal_);
}

template <typename T>
__aicore__ inline void MoeGatingTopKBackward<T>::CopyOut(int64_t loopIdx)
{
    gradXOutTensor_ = outQue_.DeQue<T>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(elementCountPerLoop_ * sizeof(T)), 0, 0, 0};
    DataCopyPad(gradXGm_[loopIdx * tilingData_->baseRows * tilingData_->expertCount], gradXOutTensor_, dataCopyParams);
    outQue_.FreeTensor(gradXOutTensor_);
}

} // namespace MoeGatingTopKBackwardNs
#endif // MOE_GATING_TOP_K_BACKWARD_H
