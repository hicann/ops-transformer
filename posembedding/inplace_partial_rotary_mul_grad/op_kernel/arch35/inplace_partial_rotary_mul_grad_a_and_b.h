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
 * \file inplace_partial_rotary_mul_grad_a_and_b.h
 * \brief A (NO_BROADCAST) and B (BROADCAST_BSN) kernel for InplacePartialRotaryMulGrad
 */
#ifndef __INPLACE_PARTIAL_ROTARY_MUL_GRAD_A_AND_B_H__
#define __INPLACE_PARTIAL_ROTARY_MUL_GRAD_A_AND_B_H__

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/load_store_utils.h"
#include "inplace_partial_rotary_mul_grad_common.h"
#include <cstdint>

namespace InplacePartialRotaryMulGrad {
using namespace AscendC;

template <typename TDY, typename TCOS, bool IsBroadCast>
class InplacePartialRotaryMulGradAAndB {
public:
    __aicore__ inline InplacePartialRotaryMulGradAAndB(TPipe *pipe,
                                                       const InplacePartialRotaryMulGradRegbaseTilingData *tiling)
        : pipe_(pipe),
          tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR cos, GM_ADDR sin, GM_ADDR dx);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessInLoop(int64_t bUbStart, int64_t bUbLength, LocalTensor<TCOS> &cosUb,
                                         LocalTensor<TCOS> &sinUb);
    __aicore__ inline void CopyInCosAndSin(int64_t bStart, int64_t bLength);
    __aicore__ inline void CopyInDy(int64_t bStart, int64_t bLength);
    __aicore__ inline void CopyOutDx(int64_t bStart, int64_t bLength);
    __aicore__ inline void Compute(LocalTensor<TCOS> &cosUb, LocalTensor<TCOS> &sinUb, int64_t bLength);

    TPipe *pipe_{nullptr};
    int64_t blockIdx_{0};

    GlobalTensor<TDY> dyGm_;
    GlobalTensor<TDY> dxGm_;
    GlobalTensor<TCOS> cosGm_;
    GlobalTensor<TCOS> sinGm_;

    TQue<QuePosition::VECIN, 1> dyInQue_;
    TQue<QuePosition::VECIN, 1> cosInQue_;
    TQue<QuePosition::VECIN, 1> sinInQue_;
    TQue<QuePosition::VECOUT, 1> dxOutQue_;

    const InplacePartialRotaryMulGradRegbaseTilingData *tilingData_{nullptr};

    int64_t bBlockStart_{0};
    int64_t bBlockLength_{0};
    int64_t ubFactorB_{0};
    int64_t D_{0};
    int64_t dAlignDy_{0};
    int64_t dAlignCos_{0};
    uint32_t dSplitSizeDy_{0};
    uint32_t dSplitSizeCos_{0};
    uint16_t dSplitCoef_{0};
};

// ---- Init ----
template <typename TDY, typename TCOS, bool IsBroadCast>
__aicore__ inline void InplacePartialRotaryMulGradAAndB<TDY, TCOS, IsBroadCast>::Init(GM_ADDR dy, GM_ADDR cos,
                                                                                      GM_ADDR sin, GM_ADDR dx)
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }
    this->blockIdx_ = GetBlockIdx();
    this->D_ = tilingData_->d;

    this->dSplitCoef_ = static_cast<uint16_t>(tilingData_->dSplitCoef);

    this->dAlignDy_ = Ops::Base::CeilAlign<int64_t>(tilingData_->sliceLength / dSplitCoef_,
                                                    static_cast<int64_t>(BLOCK_TYPE_SIZE / sizeof(TDY))) *
                      dSplitCoef_;
    this->dAlignCos_ = Ops::Base::CeilAlign<int64_t>(tilingData_->sliceLength / dSplitCoef_,
                                                     static_cast<int64_t>(BLOCK_TYPE_SIZE / sizeof(TCOS))) *
                       dSplitCoef_;
    this->dSplitSizeDy_ = static_cast<uint32_t>(tilingData_->sliceLength * sizeof(TDY) / dSplitCoef_);
    this->dSplitSizeCos_ = static_cast<uint32_t>(tilingData_->sliceLength * sizeof(TCOS) / dSplitCoef_);
    this->ubFactorB_ = tilingData_->ubFactorB;

    this->dyGm_.SetGlobalBuffer((__gm__ TDY *)dy);
    this->dxGm_.SetGlobalBuffer((__gm__ TDY *)dx);
    this->cosGm_.SetGlobalBuffer((__gm__ TCOS *)cos);
    this->sinGm_.SetGlobalBuffer((__gm__ TCOS *)sin);

    this->pipe_->InitBuffer(this->dyInQue_, 2, ubFactorB_ * dAlignDy_ * sizeof(TDY));
    this->pipe_->InitBuffer(this->dxOutQue_, 2, ubFactorB_ * dAlignDy_ * sizeof(TDY));
    if constexpr (IsBroadCast) {
        this->pipe_->InitBuffer(this->cosInQue_, 1, dAlignCos_ * sizeof(TCOS));
        this->pipe_->InitBuffer(this->sinInQue_, 1, dAlignCos_ * sizeof(TCOS));
    } else {
        this->pipe_->InitBuffer(this->cosInQue_, 2, ubFactorB_ * dAlignCos_ * sizeof(TCOS));
        this->pipe_->InitBuffer(this->sinInQue_, 2, ubFactorB_ * dAlignCos_ * sizeof(TCOS));
    }
}

// ---- Process ----
template <typename TDY, typename TCOS, bool IsBroadCast>
__aicore__ inline void InplacePartialRotaryMulGradAAndB<TDY, TCOS, IsBroadCast>::Process()
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }
    bBlockLength_ = tilingData_->blockFactorB;
    if (blockIdx_ == tilingData_->blockNumB - 1 && tilingData_->b % tilingData_->blockFactorB != 0) {
        bBlockLength_ = tilingData_->b % tilingData_->blockFactorB;
    }
    bBlockStart_ = blockIdx_ * tilingData_->blockFactorB;

    int64_t ubLoopCount = Ops::Base::CeilDiv(bBlockLength_, ubFactorB_);
    if constexpr (IsBroadCast) {
        // cos/sin broadcast on B: load once, reuse
        CopyInCosAndSin(0, 1);
        LocalTensor<TCOS> cosUb = cosInQue_.DeQue<TCOS>();
        LocalTensor<TCOS> sinUb = sinInQue_.DeQue<TCOS>();
        for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopCount; ubLoopIdx++) {
            ProcessInLoop(bBlockStart_ + ubLoopIdx * ubFactorB_,
                          (ubLoopIdx != ubLoopCount - 1) ? ubFactorB_ : bBlockLength_ - ubLoopIdx * ubFactorB_, cosUb,
                          sinUb);
        }
        cosInQue_.FreeTensor(cosUb);
        sinInQue_.FreeTensor(sinUb);
    } else {
        // cos/sin vary with B: reload per group
        for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopCount; ubLoopIdx++) {
            int64_t currBLength = (ubLoopIdx != ubLoopCount - 1) ? ubFactorB_ : bBlockLength_ - ubLoopIdx * ubFactorB_;
            CopyInCosAndSin(bBlockStart_ + ubLoopIdx * ubFactorB_, currBLength);
            LocalTensor<TCOS> cosUb = cosInQue_.DeQue<TCOS>();
            LocalTensor<TCOS> sinUb = sinInQue_.DeQue<TCOS>();
            ProcessInLoop(bBlockStart_ + ubLoopIdx * ubFactorB_, currBLength, cosUb, sinUb);
            cosInQue_.FreeTensor(cosUb);
            sinInQue_.FreeTensor(sinUb);
        }
    }
}

template <typename TDY, typename TCOS, bool IsBroadCast>
__aicore__ inline void InplacePartialRotaryMulGradAAndB<TDY, TCOS, IsBroadCast>::ProcessInLoop(int64_t bUbStart,
                                                                                               int64_t bUbLength,
                                                                                               LocalTensor<TCOS> &cosUb,
                                                                                               LocalTensor<TCOS> &sinUb)
{
    CopyInDy(bUbStart, bUbLength);
    Compute(cosUb, sinUb, bUbLength);
    CopyOutDx(bUbStart, bUbLength);
}

// ---- Copy In Cos/Sin ----
template <typename TDY, typename TCOS, bool IsBroadCast>
__aicore__ inline void InplacePartialRotaryMulGradAAndB<TDY, TCOS, IsBroadCast>::CopyInCosAndSin(int64_t bStart,
                                                                                                 int64_t bLength)
{
    LocalTensor<TCOS> cosUb = cosInQue_.AllocTensor<TCOS>();
    LocalTensor<TCOS> sinUb = sinInQue_.AllocTensor<TCOS>();

    int64_t cosSinOffset = bStart * tilingData_->sliceLength;

    DataCopyExtParams copyParams = {static_cast<uint16_t>(bLength * dSplitCoef_), dSplitSizeCos_, 0, 0, 0};
    DataCopyPadExtParams<TCOS> padParams = {false, 0, 0, 0};

    DataCopyPad(cosUb, cosGm_[cosSinOffset], copyParams, padParams);
    DataCopyPad(sinUb, sinGm_[cosSinOffset], copyParams, padParams);

    cosInQue_.EnQue(cosUb);
    sinInQue_.EnQue(sinUb);
}

// ---- Copy In Dy ----
template <typename TDY, typename TCOS, bool IsBroadCast>
__aicore__ inline void InplacePartialRotaryMulGradAAndB<TDY, TCOS, IsBroadCast>::CopyInDy(int64_t bStart,
                                                                                          int64_t bLength)
{
    LocalTensor<TDY> dyUb = dyInQue_.AllocTensor<TDY>();

    // After MergeDim: tensor is effectively [B, D]
    int64_t dyOffset = bStart * D_ + tilingData_->sliceStart;

    DataCopyExtParams copyParams = {static_cast<uint16_t>(bLength * dSplitCoef_), dSplitSizeDy_,
                                    static_cast<uint32_t>((D_ - tilingData_->sliceLength) * sizeof(TDY)), 0, 0};
    DataCopyPadExtParams<TDY> padParams = {false, 0, 0, 0};

    DataCopyPad(dyUb, dyGm_[dyOffset], copyParams, padParams);
    dyInQue_.EnQue(dyUb);
}

// ---- Copy Out Dx ----
template <typename TDY, typename TCOS, bool IsBroadCast>
__aicore__ inline void InplacePartialRotaryMulGradAAndB<TDY, TCOS, IsBroadCast>::CopyOutDx(int64_t bStart,
                                                                                           int64_t bLength)
{
    LocalTensor<TDY> dxUb = dxOutQue_.DeQue<TDY>();

    int64_t dxOffset = bStart * D_ + tilingData_->sliceStart;

    DataCopyExtParams copyParams = {static_cast<uint16_t>(bLength * dSplitCoef_), dSplitSizeDy_, 0,
                                    static_cast<uint32_t>((D_ - tilingData_->sliceLength) * sizeof(TDY)), 0};

    DataCopyPad(dxGm_[dxOffset], dxUb, copyParams);
    dxOutQue_.FreeTensor(dxUb);
}

// ---- Compute ----
template <typename TDY, typename TCOS, bool IsBroadCast>
__aicore__ inline void InplacePartialRotaryMulGradAAndB<TDY, TCOS, IsBroadCast>::Compute(LocalTensor<TCOS> &cosUb,
                                                                                         LocalTensor<TCOS> &sinUb,
                                                                                         int64_t bLength)
{
    LocalTensor<TDY> inUb = dyInQue_.DeQue<TDY>();
    LocalTensor<TDY> outUb = dxOutQue_.AllocTensor<TDY>();

    if constexpr (IsBroadCast) {
        // cos/sin 1 shared row, dy bLength rows: treat B as "N" dimension
        InterleaveModeGradVF<TDY, TCOS>(sinUb, cosUb, inUb, outUb, tilingData_->sliceLength, dSplitCoef_,
                                        static_cast<uint16_t>(1), static_cast<uint16_t>(bLength));
    } else {
        // cos/sin bLength rows, dy bLength rows: treat B as "S" dimension
        BatchInterleaveModeGradVF<TDY, TCOS, IsBroadCast>(
            (__ubuf__ TDY *)inUb.GetPhyAddr(), (__ubuf__ TCOS *)cosUb.GetPhyAddr(), (__ubuf__ TCOS *)sinUb.GetPhyAddr(),
            (__ubuf__ TDY *)outUb.GetPhyAddr(), static_cast<uint16_t>(bLength), static_cast<uint16_t>(1),
            static_cast<uint16_t>(1), tilingData_->sliceLength, dAlignDy_, dAlignCos_, ubFactorB_,
            static_cast<int64_t>(1));
    }

    dyInQue_.FreeTensor(inUb);
    dxOutQue_.EnQue(outUb);
}

} // namespace InplacePartialRotaryMulGrad
#endif
