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
 * \file inplace_partial_rotary_mul_grad_aba_and_ba.h
 * \brief ABA (cosb_!=1) and BA (cosb_==1) kernel for InplacePartialRotaryMulGrad (BNSD layout)
 */
#ifndef __INPLACE_PARTIAL_ROTARY_MUL_GRAD_ABA_AND_BA_H__
#define __INPLACE_PARTIAL_ROTARY_MUL_GRAD_ABA_AND_BA_H__

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/load_store_utils.h"
#include "inplace_partial_rotary_mul_grad_common.h"
#include <cstdint>

namespace InplacePartialRotaryMulGrad {
using namespace AscendC;

template <typename TDY, typename TCOS, bool IsBroadCast>
class InplacePartialRotaryMulGradABAAndBA {
public:
    __aicore__ inline InplacePartialRotaryMulGradABAAndBA(TPipe *pipe,
                                                          const InplacePartialRotaryMulGradRegbaseTilingData *tiling)
        : pipe_(pipe),
          tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR cos, GM_ADDR sin, GM_ADDR dx);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessInSLoop(int64_t sUbStart, int64_t sUbLength);
    __aicore__ inline void ProcessInSBLoop(int64_t sUbStart, int64_t sUbLength, int64_t bUbStart, int64_t bUbLength,
                                           LocalTensor<TCOS> &cosUb, LocalTensor<TCOS> &sinUb);
    __aicore__ inline void ProcessInSBNLoop(int64_t sUbStart, int64_t sUbLength, int64_t bUbStart, int64_t bUbLength,
                                            int64_t nUbStart, int64_t nUbLength, int64_t nTotalSize,
                                            LocalTensor<TCOS> &cosUb, LocalTensor<TCOS> &sinUb);
    __aicore__ inline void CopyInCosAndSin(int64_t sStart, int64_t sLength, int64_t bStart, int64_t bLength);
    __aicore__ inline void CopyInDy(int64_t sStart, int64_t sLength, int64_t bStart, int64_t bLength, int64_t nStart,
                                    int64_t nLength, int64_t nTotalSize);
    __aicore__ inline void CopyOutDx(int64_t sStart, int64_t sLength, int64_t bStart, int64_t bLength, int64_t nStart,
                                     int64_t nLength, int64_t nTotalSize);
    __aicore__ inline void Compute(LocalTensor<TCOS> &cosUb, LocalTensor<TCOS> &sinUb, int64_t sLength, int64_t bLength,
                                   int64_t nLength);

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
    int64_t sBlockStart_{0};
    int64_t sBlockLength_{0};

    int64_t ubFactorB_{0};
    int64_t ubFactorS_{0};
    int64_t ubFactorN_{0};
    int64_t D_{0};
    int64_t dAlignDy_{0};
    int64_t dAlignCos_{0};
    uint32_t dSplitSizeDy_{0};
    uint32_t dSplitSizeCos_{0};
    uint16_t dSplitCoef_{0};
};

// ---- Init ----
template <typename TDY, typename TCOS, bool IsBroadCast>
__aicore__ inline void InplacePartialRotaryMulGradABAAndBA<TDY, TCOS, IsBroadCast>::Init(GM_ADDR dy, GM_ADDR cos,
                                                                                         GM_ADDR sin, GM_ADDR dx)
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }
    this->blockIdx_ = GetBlockIdx();
    this->dSplitCoef_ = static_cast<uint16_t>(tilingData_->dSplitCoef);
    this->D_ = tilingData_->d;

    this->dAlignDy_ = Ops::Base::CeilAlign<int64_t>(tilingData_->sliceLength / dSplitCoef_,
                                                    static_cast<int64_t>(BLOCK_TYPE_SIZE / sizeof(TDY))) *
                      dSplitCoef_;
    this->dAlignCos_ = Ops::Base::CeilAlign<int64_t>(tilingData_->sliceLength / dSplitCoef_,
                                                     static_cast<int64_t>(BLOCK_TYPE_SIZE / sizeof(TCOS))) *
                       dSplitCoef_;
    this->dSplitSizeDy_ = static_cast<uint32_t>(tilingData_->sliceLength * sizeof(TDY) / dSplitCoef_);
    this->dSplitSizeCos_ = static_cast<uint32_t>(tilingData_->sliceLength * sizeof(TCOS) / dSplitCoef_);
    this->ubFactorB_ = tilingData_->ubFactorB;
    this->ubFactorS_ = tilingData_->ubFactorS;
    this->ubFactorN_ = tilingData_->ubFactorN;

    this->dyGm_.SetGlobalBuffer((__gm__ TDY *)dy);
    this->dxGm_.SetGlobalBuffer((__gm__ TDY *)dx);
    this->cosGm_.SetGlobalBuffer((__gm__ TCOS *)cos);
    this->sinGm_.SetGlobalBuffer((__gm__ TCOS *)sin);

    this->pipe_->InitBuffer(this->dyInQue_, 2, ubFactorB_ * ubFactorS_ * ubFactorN_ * dAlignDy_ * sizeof(TDY));
    this->pipe_->InitBuffer(this->dxOutQue_, 2, ubFactorB_ * ubFactorS_ * ubFactorN_ * dAlignDy_ * sizeof(TDY));
    if constexpr (IsBroadCast) {
        this->pipe_->InitBuffer(this->cosInQue_, 2, ubFactorS_ * dAlignCos_ * sizeof(TCOS));
        this->pipe_->InitBuffer(this->sinInQue_, 2, ubFactorS_ * dAlignCos_ * sizeof(TCOS));
    } else {
        this->pipe_->InitBuffer(this->cosInQue_, 2, ubFactorB_ * ubFactorS_ * dAlignCos_ * sizeof(TCOS));
        this->pipe_->InitBuffer(this->sinInQue_, 2, ubFactorB_ * ubFactorS_ * dAlignCos_ * sizeof(TCOS));
    }
}

template <typename TDY, typename TCOS, bool IsBroadCast>
__aicore__ inline void InplacePartialRotaryMulGradABAAndBA<TDY, TCOS, IsBroadCast>::CopyInCosAndSin(int64_t sStart,
                                                                                                    int64_t sLength,
                                                                                                    int64_t bStart,
                                                                                                    int64_t bLength)
{
    LocalTensor<TCOS> cosUb = cosInQue_.AllocTensor<TCOS>();
    LocalTensor<TCOS> sinUb = sinInQue_.AllocTensor<TCOS>();

    LoopModeParams loopParams;
    loopParams.loop2Size = 1;
    loopParams.loop1Size = bLength;
    loopParams.loop2SrcStride = 0;
    loopParams.loop2DstStride = 0;
    loopParams.loop1SrcStride = tilingData_->s * tilingData_->sliceLength * sizeof(TCOS);
    loopParams.loop1DstStride = ubFactorS_ * dAlignCos_ * sizeof(TCOS);
    SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);

    DataCopyExtParams copyParams = {static_cast<uint16_t>(sLength * dSplitCoef_), dSplitSizeCos_, 0, 0, 0};
    DataCopyPadExtParams<TCOS> padParams = {false, 0, 0, 0};

    int64_t cosSinOffset = bStart * tilingData_->s * tilingData_->sliceLength + sStart * tilingData_->sliceLength;
    DataCopyPad(cosUb, cosGm_[cosSinOffset], copyParams, padParams);
    DataCopyPad(sinUb, sinGm_[cosSinOffset], copyParams, padParams);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);

    cosInQue_.EnQue(cosUb);
    sinInQue_.EnQue(sinUb);
}

template <typename TDY, typename TCOS, bool IsBroadCast>
__aicore__ inline void InplacePartialRotaryMulGradABAAndBA<TDY, TCOS, IsBroadCast>::CopyInDy(
    int64_t sStart, int64_t sLength, int64_t bStart, int64_t bLength, int64_t nStart, int64_t nLength,
    int64_t nTotalSize)
{
    LocalTensor<TDY> dyUb = dyInQue_.AllocTensor<TDY>();

    // BNSD: dy[b][n][s][d] — multi-level striding
    LoopModeParams loopParams;
    loopParams.loop2Size = bLength;
    loopParams.loop1Size = nLength;
    loopParams.loop2SrcStride = nTotalSize * tilingData_->s * D_ * sizeof(TDY);
    loopParams.loop2DstStride = ubFactorN_ * ubFactorS_ * dAlignDy_ * sizeof(TDY);
    loopParams.loop1SrcStride = tilingData_->s * D_ * sizeof(TDY);
    loopParams.loop1DstStride = ubFactorS_ * dAlignDy_ * sizeof(TDY);
    SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);

    DataCopyExtParams copyParams = {static_cast<uint16_t>(sLength * dSplitCoef_), dSplitSizeDy_,
                                    static_cast<uint32_t>((D_ - tilingData_->sliceLength) * sizeof(TDY)), 0, 0};
    DataCopyPadExtParams<TDY> padParams = {false, 0, 0, 0};

    int64_t dyOffset = bStart * nTotalSize * tilingData_->s * D_ + nStart * tilingData_->s * D_ + sStart * D_ +
                       tilingData_->sliceStart;
    DataCopyPad(dyUb, dyGm_[dyOffset], copyParams, padParams);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);

    dyInQue_.EnQue(dyUb);
}

template <typename TDY, typename TCOS, bool IsBroadCast>
__aicore__ inline void InplacePartialRotaryMulGradABAAndBA<TDY, TCOS, IsBroadCast>::CopyOutDx(
    int64_t sStart, int64_t sLength, int64_t bStart, int64_t bLength, int64_t nStart, int64_t nLength,
    int64_t nTotalSize)
{
    LocalTensor<TDY> dxUb = dxOutQue_.DeQue<TDY>();

    LoopModeParams loopParams;
    loopParams.loop2Size = bLength;
    loopParams.loop1Size = nLength;
    loopParams.loop2DstStride = nTotalSize * tilingData_->s * D_ * sizeof(TDY);
    loopParams.loop2SrcStride = ubFactorN_ * ubFactorS_ * dAlignDy_ * sizeof(TDY);
    loopParams.loop1DstStride = tilingData_->s * D_ * sizeof(TDY);
    loopParams.loop1SrcStride = ubFactorS_ * dAlignDy_ * sizeof(TDY);
    SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);

    DataCopyExtParams copyParams = {static_cast<uint16_t>(sLength * dSplitCoef_), dSplitSizeDy_, 0,
                                    static_cast<uint32_t>((D_ - tilingData_->sliceLength) * sizeof(TDY)), 0};

    int64_t dxOffset = bStart * nTotalSize * tilingData_->s * D_ + nStart * tilingData_->s * D_ + sStart * D_ +
                       tilingData_->sliceStart;
    DataCopyPad(dxGm_[dxOffset], dxUb, copyParams);
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);

    dxOutQue_.FreeTensor(dxUb);
}

template <typename TDY, typename TCOS, bool IsBroadCast>
__aicore__ inline void InplacePartialRotaryMulGradABAAndBA<TDY, TCOS, IsBroadCast>::Compute(
    LocalTensor<TCOS> &cosUb, LocalTensor<TCOS> &sinUb, int64_t sLength, int64_t bLength, int64_t nLength)
{
    LocalTensor<TDY> inUb = dyInQue_.DeQue<TDY>();
    LocalTensor<TDY> outUb = dxOutQue_.AllocTensor<TDY>();

    BatchInterleaveModeGradVF<TDY, TCOS, IsBroadCast>(
        (__ubuf__ TDY *)inUb.GetPhyAddr(), (__ubuf__ TCOS *)cosUb.GetPhyAddr(), (__ubuf__ TCOS *)sinUb.GetPhyAddr(),
        (__ubuf__ TDY *)outUb.GetPhyAddr(), static_cast<uint16_t>(sLength), static_cast<uint16_t>(bLength),
        static_cast<uint16_t>(nLength), tilingData_->sliceLength, dAlignDy_, dAlignCos_, ubFactorS_, ubFactorN_);

    dyInQue_.FreeTensor(inUb);
    dxOutQue_.EnQue(outUb);
}

// ---- Process ----
template <typename TDY, typename TCOS, bool IsBroadCast>
__aicore__ inline void InplacePartialRotaryMulGradABAAndBA<TDY, TCOS, IsBroadCast>::Process()
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }
    // BNSD: blockIdx maps to (B,S) grid
    int64_t bIdx = blockIdx_ % tilingData_->blockNumB;
    int64_t sIdx = blockIdx_ / tilingData_->blockNumB;
    bBlockLength_ = tilingData_->blockFactorB;
    sBlockLength_ = tilingData_->blockFactorS;
    if (bIdx == tilingData_->blockNumB - 1 && tilingData_->b % tilingData_->blockFactorB != 0) {
        bBlockLength_ = tilingData_->b % tilingData_->blockFactorB;
    }
    if (sIdx == tilingData_->blockNumS - 1 && tilingData_->s % tilingData_->blockFactorS != 0) {
        sBlockLength_ = tilingData_->s % tilingData_->blockFactorS;
    }
    bBlockStart_ = bIdx * tilingData_->blockFactorB;
    sBlockStart_ = sIdx * tilingData_->blockFactorS;

    int64_t ubLoopCount = Ops::Base::CeilDiv(sBlockLength_, ubFactorS_);
    for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopCount; ubLoopIdx++) {
        ProcessInSLoop(sBlockStart_ + ubLoopIdx * ubFactorS_,
                       (ubLoopIdx != ubLoopCount - 1) ? ubFactorS_ : sBlockLength_ - ubLoopIdx * ubFactorS_);
    }
}

template <typename TDY, typename TCOS, bool IsBroadCast>
__aicore__ inline void InplacePartialRotaryMulGradABAAndBA<TDY, TCOS, IsBroadCast>::ProcessInSLoop(int64_t sUbStart,
                                                                                                   int64_t sUbLength)
{
    int64_t ubLoopCount = Ops::Base::CeilDiv(bBlockLength_, ubFactorB_);
    if constexpr (IsBroadCast) {
        CopyInCosAndSin(sUbStart, sUbLength, 0, 1);
        LocalTensor<TCOS> cosUb = cosInQue_.DeQue<TCOS>();
        LocalTensor<TCOS> sinUb = sinInQue_.DeQue<TCOS>();
        for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopCount; ubLoopIdx++) {
            ProcessInSBLoop(sUbStart, sUbLength, bBlockStart_ + ubLoopIdx * ubFactorB_,
                            (ubLoopIdx != ubLoopCount - 1) ? ubFactorB_ : bBlockLength_ - ubLoopIdx * ubFactorB_, cosUb,
                            sinUb);
        }
        cosInQue_.FreeTensor(cosUb);
        sinInQue_.FreeTensor(sinUb);
    } else {
        for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopCount; ubLoopIdx++) {
            CopyInCosAndSin(sUbStart, sUbLength, bBlockStart_ + ubLoopIdx * ubFactorB_,
                            (ubLoopIdx != ubLoopCount - 1) ? ubFactorB_ : bBlockLength_ - ubLoopIdx * ubFactorB_);
            LocalTensor<TCOS> cosUb = cosInQue_.DeQue<TCOS>();
            LocalTensor<TCOS> sinUb = sinInQue_.DeQue<TCOS>();
            ProcessInSBLoop(sUbStart, sUbLength, bBlockStart_ + ubLoopIdx * ubFactorB_,
                            (ubLoopIdx != ubLoopCount - 1) ? ubFactorB_ : bBlockLength_ - ubLoopIdx * ubFactorB_, cosUb,
                            sinUb);
            cosInQue_.FreeTensor(cosUb);
            sinInQue_.FreeTensor(sinUb);
        }
    }
}

template <typename TDY, typename TCOS, bool IsBroadCast>
__aicore__ inline void InplacePartialRotaryMulGradABAAndBA<TDY, TCOS, IsBroadCast>::ProcessInSBLoop(
    int64_t sUbStart, int64_t sUbLength, int64_t bUbStart, int64_t bUbLength, LocalTensor<TCOS> &cosUb,
    LocalTensor<TCOS> &sinUb)
{
    int64_t nUbLoopCount = Ops::Base::CeilDiv(tilingData_->n, ubFactorN_);
    for (int64_t ubLoopIdx = 0; ubLoopIdx < nUbLoopCount; ubLoopIdx++) {
        ProcessInSBNLoop(sUbStart, sUbLength, bUbStart, bUbLength, ubLoopIdx * ubFactorN_,
                         (ubLoopIdx != nUbLoopCount - 1) ? ubFactorN_ : tilingData_->n - ubLoopIdx * ubFactorN_,
                         tilingData_->n, cosUb, sinUb);
    }
}

template <typename TDY, typename TCOS, bool IsBroadCast>
__aicore__ inline void InplacePartialRotaryMulGradABAAndBA<TDY, TCOS, IsBroadCast>::ProcessInSBNLoop(
    int64_t sUbStart, int64_t sUbLength, int64_t bUbStart, int64_t bUbLength, int64_t nUbStart, int64_t nUbLength,
    int64_t nTotalSize, LocalTensor<TCOS> &cosUb, LocalTensor<TCOS> &sinUb)
{
    CopyInDy(sUbStart, sUbLength, bUbStart, bUbLength, nUbStart, nUbLength, nTotalSize);
    Compute(cosUb, sinUb, sUbLength, bUbLength, nUbLength);
    CopyOutDx(sUbStart, sUbLength, bUbStart, bUbLength, nUbStart, nUbLength, nTotalSize);
}

} // namespace InplacePartialRotaryMulGrad
#endif
