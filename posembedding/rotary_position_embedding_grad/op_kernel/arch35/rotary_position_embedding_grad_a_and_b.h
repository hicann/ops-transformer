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
 * \file rotary_position_embedding_grad_a_and_b.h
 * \brief
 */
#ifndef ROTARY_POSITION_EMBEDDING_GRAD_A_AND_B_H
#define ROTARY_POSITION_EMBEDDING_GRAD_A_AND_B_H

#include "rotary_position_embedding_grad_common.h"
#include "rotary_position_embedding_grad_tiling_data.h"

namespace RotaryPositionEmbeddingGrad {
using namespace AscendC;

template <typename T, bool IsBoardCast>
class RotaryPositionEmbeddingGradAAndB {
public:
    __aicore__ inline RotaryPositionEmbeddingGradAAndB(TPipe *pipe, const RopeGradRegbaseParams *tiling)
        : pipe_(pipe),
          tilingData_(tiling){};

    __aicore__ inline ~RotaryPositionEmbeddingGradAAndB(){};

    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR cos, GM_ADDR sin, GM_ADDR xGradOut);

    __aicore__ inline void Process();

private:
    // Init过程中使用的内部函数
    __aicore__ inline void InitAllGlobalBuffer(GM_ADDR grad, GM_ADDR cos, GM_ADDR sin, GM_ADDR xGradOut);
    __aicore__ inline void InitAllBuffer();
    __aicore__ inline void InitLoopParams();
    // 各个层级的Process函数
    __aicore__ inline void ProcessInLoop(LocalTensor<T> &cos, LocalTensor<T> &sin, int64_t bStart, int64_t bLength);
    // 拷入拷出函数
    __aicore__ inline void CopyInCosAndSin(int64_t bStart, int64_t bLength);
    __aicore__ inline void CopyIn(GlobalTensor<T> &source, int64_t bStart, int64_t bLength);
    __aicore__ inline void CopyOut(GlobalTensor<T> &target, int64_t bStart, int64_t bLength);

    // 计算函数
    __aicore__ inline void Compute(LocalTensor<T> &cos, LocalTensor<T> &sin, int64_t bLength);

private:
    constexpr static uint32_t COS_DB_BUFFER = IsBoardCast ? 1 : DOUBLE_BUFFER;

    TPipe *pipe_;

    // GlobalMemory
    GlobalTensor<T> gradGm_;
    GlobalTensor<T> cosGm_;
    GlobalTensor<T> sinGm_;
    GlobalTensor<T> xGradOutGm_;

    // UB
    TQue<QuePosition::VECIN, 1> gradInQueue_;
    TQue<QuePosition::VECIN, 1> cosInQueue_;
    TQue<QuePosition::VECIN, 1> sinInQueue_;
    TQue<QuePosition::VECOUT, 1> xGradOutQueue_;

    // Split core info
    int64_t blockIdx_ = 0;
    int64_t bBlockStart_ = 0;
    int64_t bBlockLength_ = 0;

    // TilingData
    const RopeGradRegbaseParams *tilingData_;
    int64_t ubFactorB_ = 0;
    int64_t D_ = 0;
    int64_t dAlign_ = 0;

    // 拷贝参数
    uint8_t dSplitCoef_ = 1;
    uint8_t copyOutSplitCoef_ = 1; // 拷贝出xGrad时使用的splitCoef
    uint64_t ubCopyOutStride = 0;  // 输出在ub中的stride，deepseek_interleave中不为0
};

template <typename T, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingGradAAndB<T, IsBoardCast>::Init(GM_ADDR grad, GM_ADDR cos, GM_ADDR sin,
                                                                              GM_ADDR xGradOut)
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }
    this->blockIdx_ = GetBlockIdx();
    this->InitAllGlobalBuffer(grad, cos, sin, xGradOut);
    this->InitAllBuffer();
    this->InitLoopParams();
}

template <typename T, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingGradAAndB<T, IsBoardCast>::InitAllGlobalBuffer(GM_ADDR grad, GM_ADDR cos,
                                                                                             GM_ADDR sin,
                                                                                             GM_ADDR xGradOut)
{
    this->gradGm_.SetGlobalBuffer((__gm__ T *)grad);
    this->cosGm_.SetGlobalBuffer((__gm__ T *)cos);
    this->sinGm_.SetGlobalBuffer((__gm__ T *)sin);
    this->xGradOutGm_.SetGlobalBuffer((__gm__ T *)xGradOut);
}

template <typename T, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingGradAAndB<T, IsBoardCast>::InitAllBuffer()
{
    this->ubFactorB_ = this->tilingData_->ubFactorB;
    this->D_ = this->tilingData_->d;
    if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::HALF) ||
        tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::DEEPSEEK_INTERLEAVE)) {
        this->dSplitCoef_ = HALF_INTERLEAVE_COEF;
    } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::QUARTER)) {
        this->dSplitCoef_ = QUARTER_MODE_COEF;
    }
    this->copyOutSplitCoef_ = dSplitCoef_;
    this->dAlign_ = Ops::Base::CeilAlign<int64_t>(D_ / dSplitCoef_, BLOCK_TYPE_SIZE / sizeof(T)) * dSplitCoef_;
    if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::DEEPSEEK_INTERLEAVE)) {
        this->copyOutSplitCoef_ = 1;
        // 非boardcast时，使用批量计算API，需要拷贝时添加stride
        if constexpr (!IsBoardCast) {
            this->ubCopyOutStride =
                (this->dAlign_ * sizeof(T) - Ops::Base::CeilAlign<int64_t>(D_ * sizeof(T), BLOCK_TYPE_SIZE)) /
                BLOCK_TYPE_SIZE;
        }
    }

    this->pipe_->InitBuffer(this->gradInQueue_, DOUBLE_BUFFER, ubFactorB_ * dAlign_ * sizeof(T));
    this->pipe_->InitBuffer(this->xGradOutQueue_, DOUBLE_BUFFER, ubFactorB_ * dAlign_ * sizeof(T));
    if constexpr (IsBoardCast) {
        this->pipe_->InitBuffer(this->cosInQueue_, COS_DB_BUFFER, dAlign_ * sizeof(T));
        this->pipe_->InitBuffer(this->sinInQueue_, COS_DB_BUFFER, dAlign_ * sizeof(T));
    } else {
        this->pipe_->InitBuffer(this->cosInQueue_, COS_DB_BUFFER, ubFactorB_ * dAlign_ * sizeof(T));
        this->pipe_->InitBuffer(this->sinInQueue_, COS_DB_BUFFER, ubFactorB_ * dAlign_ * sizeof(T));
    }
}

template <typename T, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingGradAAndB<T, IsBoardCast>::InitLoopParams()
{
    this->bBlockLength_ = tilingData_->blockFactorB;
    if (blockIdx_ == tilingData_->blockNumB - 1 && tilingData_->b % tilingData_->blockFactorB != 0) {
        this->bBlockLength_ = tilingData_->b % tilingData_->blockFactorB;
    }
    this->bBlockStart_ = blockIdx_ * tilingData_->blockFactorB;
}

template <typename T, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingGradAAndB<T, IsBoardCast>::Process()
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }
    // 在B轴进行循环
    int64_t ubLoopCount = Ops::Base::CeilDiv(bBlockLength_, ubFactorB_);
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
__aicore__ inline void RotaryPositionEmbeddingGradAAndB<T, IsBoardCast>::ProcessInLoop(LocalTensor<T> &cos,
                                                                                       LocalTensor<T> &sin,
                                                                                       int64_t bUbStart,
                                                                                       int64_t bUbLength)
{
    CopyIn(gradGm_, bUbStart, bUbLength);
    Compute(cos, sin, bUbLength);
    CopyOut(xGradOutGm_, bUbStart, bUbLength);
}

template <typename T, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingGradAAndB<T, IsBoardCast>::CopyInCosAndSin(int64_t bStart,
                                                                                         int64_t bLength)
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
    copyExtParams.blockLen = D_ * sizeof(T) / dSplitCoef_;
    copyExtParams.srcStride = 0;
    copyExtParams.dstStride = 0;
    DataCopyPad(cosUb, this->cosGm_[bStart * D_], copyExtParams, copyPadExtparams);
    DataCopyPad(sinUb, this->sinGm_[bStart * D_], copyExtParams, copyPadExtparams);
    this->cosInQueue_.template EnQue(cosUb);
    this->sinInQueue_.template EnQue(sinUb);
}

template <typename T, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingGradAAndB<T, IsBoardCast>::CopyIn(GlobalTensor<T> &source, int64_t bStart,
                                                                                int64_t bLength)
{
    LocalTensor<T> target = this->gradInQueue_.template AllocTensor<T>();
    DataCopyExtParams copyExtParams;
    copyExtParams.blockCount = bLength * dSplitCoef_;
    copyExtParams.blockLen = D_ * sizeof(T) / dSplitCoef_;
    copyExtParams.srcStride = 0;
    copyExtParams.dstStride = 0;
    DataCopyPadExtParams<T> copyPadExtparams;
    copyPadExtparams.isPad = false;
    copyPadExtparams.leftPadding = 0;
    copyPadExtparams.rightPadding = 0;
    copyPadExtparams.paddingValue = 0;
    DataCopyPad(target, source[bStart * D_], copyExtParams, copyPadExtparams);
    this->gradInQueue_.template EnQue(target);
}

template <typename T, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingGradAAndB<T, IsBoardCast>::CopyOut(GlobalTensor<T> &target,
                                                                                 int64_t bStart, int64_t bLength)
{
    LocalTensor<T> source = this->xGradOutQueue_.template DeQue<T>();
    DataCopyExtParams copyExtParams;
    copyExtParams.blockCount = bLength * copyOutSplitCoef_;
    copyExtParams.blockLen = D_ * sizeof(T) / copyOutSplitCoef_;
    copyExtParams.srcStride = ubCopyOutStride;
    copyExtParams.dstStride = 0;
    DataCopyPad(target[bStart * D_], source, copyExtParams);
    this->xGradOutQueue_.FreeTensor(source);
}

template <typename T, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingGradAAndB<T, IsBoardCast>::Compute(LocalTensor<T> &cos,
                                                                                 LocalTensor<T> &sin, int64_t bLength)
{
    LocalTensor<T> inUb = this->gradInQueue_.template DeQue<T>();
    LocalTensor<T> outUb = this->xGradOutQueue_.template AllocTensor<T>();
    if constexpr (IsBoardCast) {
        if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::HALF)) {
            HalfGradAlignVF<T>(sin, cos, inUb, outUb, tilingData_->d, dAlign_, 1, bLength);
        } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::INTERLEAVE)) {
            InterleaveModeGradVF<T>(sin, cos, inUb, outUb, tilingData_->d, 1, bLength);
        } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::QUARTER)) {
            QuarterGradAlignVF<T>(sin, cos, inUb, outUb, tilingData_->d, dAlign_, 1, bLength);
        } else {
            DeepSeekInterleaveModeGradVF<T>(sin, cos, inUb, outUb, tilingData_->d, 1, bLength);
        }
    } else {
        if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::HALF)) {
            BatchHalfGradAlignVF<T, IsBoardCast>((__ubuf__ T *)inUb.GetPhyAddr(), (__ubuf__ T *)cos.GetPhyAddr(),
                                                 (__ubuf__ T *)sin.GetPhyAddr(), (__ubuf__ T *)outUb.GetPhyAddr(),
                                                 bLength, 1, 1, D_, dAlign_, ubFactorB_, 1);
        } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::INTERLEAVE)) {
            BatchInterleaveModeGradVF<T, IsBoardCast>((__ubuf__ T *)inUb.GetPhyAddr(), (__ubuf__ T *)cos.GetPhyAddr(),
                                                      (__ubuf__ T *)sin.GetPhyAddr(), (__ubuf__ T *)outUb.GetPhyAddr(),
                                                      bLength, 1, 1, D_, dAlign_, ubFactorB_, 1);
        } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::QUARTER)) {
            BatchQuarterGradAlignVF<T, IsBoardCast>((__ubuf__ T *)inUb.GetPhyAddr(), (__ubuf__ T *)cos.GetPhyAddr(),
                                                    (__ubuf__ T *)sin.GetPhyAddr(), (__ubuf__ T *)outUb.GetPhyAddr(),
                                                    bLength, 1, 1, D_, dAlign_, ubFactorB_, 1);
        } else {
            BatchDeepSeekInterleaveModeGradVF<T, IsBoardCast>(
                (__ubuf__ T *)inUb.GetPhyAddr(), (__ubuf__ T *)cos.GetPhyAddr(), (__ubuf__ T *)sin.GetPhyAddr(),
                (__ubuf__ T *)outUb.GetPhyAddr(), bLength, 1, 1, D_, dAlign_, ubFactorB_, 1);
        }
    }

    this->gradInQueue_.FreeTensor(inUb);
    this->xGradOutQueue_.template EnQue(outUb);
}
} // namespace RotaryPositionEmbeddingGrad

#endif
