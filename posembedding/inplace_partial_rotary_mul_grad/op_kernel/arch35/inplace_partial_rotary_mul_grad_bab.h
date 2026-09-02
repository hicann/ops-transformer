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
 * \file inplace_partial_rotary_mul_grad_bab.h
 * \brief BAB template kernel for InplacePartialRotaryMulGrad (BSND layout, cosb_==1)
 */
#ifndef __INPLACE_PARTIAL_ROTARY_MUL_GRAD_BAB_H__
#define __INPLACE_PARTIAL_ROTARY_MUL_GRAD_BAB_H__

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/load_store_utils.h"
#include "inplace_partial_rotary_mul_grad_common.h"
#include <cstdint>

namespace InplacePartialRotaryMulGrad {
using namespace AscendC;

template <typename TDY, typename TCOS>
class InplacePartialRotaryMulGradBAB {
public:
    __aicore__ inline InplacePartialRotaryMulGradBAB(TPipe *pipe,
                                                     const InplacePartialRotaryMulGradRegbaseTilingData *tiling)
        : pipe_(pipe),
          tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR cos, GM_ADDR sin, GM_ADDR dx);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessNLoop(const uint32_t bIdx, const uint32_t sIdx, const uint32_t currSNum);
    __aicore__ inline void ProcessN(LocalTensor<TCOS> &sinTensor, LocalTensor<TCOS> &cosTensor, const uint32_t bIdx,
                                    const uint32_t sIdx, const uint32_t currSNum);
    __aicore__ inline void Compute(LocalTensor<TCOS> &sinTensor, LocalTensor<TCOS> &cosTensor,
                                   LocalTensor<TDY> &dyTensor, LocalTensor<TDY> &dxTensor, const uint32_t currSNum,
                                   const uint32_t currNNum);

    TPipe *pipe_{nullptr};
    int64_t blockIdx_{0};
    TQue<QuePosition::VECIN, 1> dyInQue_;
    TQue<QuePosition::VECIN, 1> cosInQue_;
    TQue<QuePosition::VECIN, 1> sinInQue_;
    TQue<QuePosition::VECOUT, 1> dxOutQue_;

    GlobalTensor<TDY> dyGm_;
    GlobalTensor<TDY> dxGm_;
    GlobalTensor<TCOS> cosGm_;
    GlobalTensor<TCOS> sinGm_;

    const InplacePartialRotaryMulGradRegbaseTilingData *tilingData_{nullptr};
    uint32_t dSplitSizeDy_{0};
    uint32_t dSplitSizeCos_{0};
    int64_t dAlignLenDy_{0};
    int64_t dAlignLenCos_{0};
    int64_t bIdx_{0};
    int64_t sIdx_{0};
    int64_t bNum_{0};
    int64_t sNum_{0};
    int64_t ubFactorN_{0};
    int64_t ubFactorS_{0};
    uint16_t dSplitCoef_{0};
};

template <typename TDY, typename TCOS>
__aicore__ inline void InplacePartialRotaryMulGradBAB<TDY, TCOS>::Init(GM_ADDR dy, GM_ADDR cos, GM_ADDR sin, GM_ADDR dx)
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }
    this->blockIdx_ = GetBlockIdx();
    this->dSplitCoef_ = static_cast<uint16_t>(tilingData_->dSplitCoef);
    this->dAlignLenDy_ = Ops::Base::CeilAlign<int64_t>(tilingData_->sliceLength / dSplitCoef_,
                                                       static_cast<int64_t>(BLOCK_TYPE_SIZE / sizeof(TDY))) *
                         dSplitCoef_;
    this->dAlignLenCos_ = Ops::Base::CeilAlign<int64_t>(tilingData_->sliceLength / dSplitCoef_,
                                                        static_cast<int64_t>(BLOCK_TYPE_SIZE / sizeof(TCOS))) *
                          dSplitCoef_;
    this->dSplitSizeDy_ = static_cast<uint32_t>(tilingData_->sliceLength * sizeof(TDY) / dSplitCoef_);
    this->dSplitSizeCos_ = static_cast<uint32_t>(tilingData_->sliceLength * sizeof(TCOS) / dSplitCoef_);
    this->ubFactorN_ = tilingData_->ubFactorN;
    this->ubFactorS_ = tilingData_->ubFactorS;
    this->dyGm_.SetGlobalBuffer((__gm__ TDY *)dy);
    this->dxGm_.SetGlobalBuffer((__gm__ TDY *)dx);
    this->cosGm_.SetGlobalBuffer((__gm__ TCOS *)cos);
    this->sinGm_.SetGlobalBuffer((__gm__ TCOS *)sin);

    this->pipe_->InitBuffer(this->dyInQue_, 2, ubFactorS_ * ubFactorN_ * dAlignLenDy_ * sizeof(TDY));
    this->pipe_->InitBuffer(this->dxOutQue_, 2, ubFactorS_ * ubFactorN_ * dAlignLenDy_ * sizeof(TDY));
    this->pipe_->InitBuffer(this->cosInQue_, 2, ubFactorS_ * dAlignLenCos_ * sizeof(TCOS));
    this->pipe_->InitBuffer(this->sinInQue_, 2, ubFactorS_ * dAlignLenCos_ * sizeof(TCOS));
}

template <typename TDY, typename TCOS>
__aicore__ inline void InplacePartialRotaryMulGradBAB<TDY, TCOS>::Process()
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }

    this->bIdx_ = blockIdx_ % tilingData_->blockNumB;
    this->sIdx_ = blockIdx_ / tilingData_->blockNumB;
    this->bNum_ = tilingData_->blockFactorB;
    this->sNum_ = tilingData_->blockFactorS;
    if (bIdx_ == tilingData_->blockNumB - 1 && tilingData_->b % tilingData_->blockFactorB != 0) {
        this->bNum_ = tilingData_->b % tilingData_->blockFactorB;
    }
    if (sIdx_ == tilingData_->blockNumS - 1 && tilingData_->s % tilingData_->blockFactorS != 0) {
        this->sNum_ = tilingData_->s % tilingData_->blockFactorS;
    }

    uint32_t bIdxStart = bIdx_ * tilingData_->blockFactorB;
    for (uint32_t bIdx = bIdxStart; bIdx < bIdxStart + bNum_; bIdx++) {
        uint32_t sIdxStart = sIdx_ * tilingData_->blockFactorS;
        uint32_t sLoopCnt = Ops::Base::CeilDiv(sNum_, tilingData_->ubFactorS);
        for (uint32_t loopIdx = 0; loopIdx < sLoopCnt; loopIdx++) {
            uint32_t currSNum = (loopIdx != sLoopCnt - 1) ? ubFactorS_ : sNum_ - loopIdx * ubFactorS_;
            ProcessNLoop(bIdx, sIdxStart + loopIdx * ubFactorS_, currSNum);
        }
    }
}

template <typename TDY, typename TCOS>
__aicore__ inline void InplacePartialRotaryMulGradBAB<TDY, TCOS>::ProcessNLoop(const uint32_t bIdx, const uint32_t sIdx,
                                                                               const uint32_t currSNum)
{
    LocalTensor<TCOS> cosTensor = cosInQue_.template AllocTensor<TCOS>();
    LocalTensor<TCOS> sinTensor = sinInQue_.template AllocTensor<TCOS>();

    int64_t offset = sIdx * tilingData_->sliceLength;
    DataCopyExtParams copyParams = {static_cast<uint16_t>(currSNum * tilingData_->dSplitCoef), dSplitSizeCos_, 0, 0, 0};
    DataCopyPadExtParams<TCOS> padParams = {false, 0, 0, 0};
    DataCopyPad(sinTensor, sinGm_[offset], copyParams, padParams);
    DataCopyPad(cosTensor, cosGm_[offset], copyParams, padParams);
    sinInQue_.EnQue(sinTensor);
    cosInQue_.EnQue(cosTensor);

    sinTensor = sinInQue_.template DeQue<TCOS>();
    cosTensor = cosInQue_.template DeQue<TCOS>();

    ProcessN(sinTensor, cosTensor, bIdx, sIdx, currSNum);

    cosInQue_.FreeTensor(cosTensor);
    sinInQue_.FreeTensor(sinTensor);
}

template <typename TDY, typename TCOS>
__aicore__ inline void InplacePartialRotaryMulGradBAB<TDY, TCOS>::ProcessN(LocalTensor<TCOS> &sinTensor,
                                                                           LocalTensor<TCOS> &cosTensor,
                                                                           const uint32_t bIdx, const uint32_t sIdx,
                                                                           const uint32_t currSNum)
{
    LocalTensor<TDY> dyTensor, dxTensor;

    int64_t baseOffset = (bIdx * tilingData_->s + sIdx) * tilingData_->n * tilingData_->d + tilingData_->sliceStart;
    for (uint32_t idxN = 0; idxN < tilingData_->ubLoopNumN; idxN++) {
        int64_t currNNum = (idxN == tilingData_->ubLoopNumN - 1) ? tilingData_->ubTailFactorN : ubFactorN_;
        int64_t offset = baseOffset + idxN * ubFactorN_ * tilingData_->d;
        dyTensor = dyInQue_.template AllocTensor<TDY>();
        DataCopyExtParams copyInParams{
            static_cast<uint16_t>(currSNum * currNNum * tilingData_->dSplitCoef), dSplitSizeDy_,
            static_cast<uint32_t>((tilingData_->d - tilingData_->sliceLength) * sizeof(TDY)), 0, 0};
        DataCopyPadExtParams<TDY> padParams{false, 0, 0, 0};
        DataCopyPad(dyTensor, dyGm_[offset], copyInParams, padParams);
        dyInQue_.EnQue(dyTensor);
        dyTensor = dyInQue_.template DeQue<TDY>();
        dxTensor = dxOutQue_.template AllocTensor<TDY>();
        Compute(sinTensor, cosTensor, dyTensor, dxTensor, currSNum, currNNum);
        dyInQue_.FreeTensor(dyTensor);
        dxOutQue_.EnQue(dxTensor);
        dxTensor = dxOutQue_.template DeQue<TDY>();
        DataCopyExtParams copyOutParams{
            static_cast<uint16_t>(currSNum * currNNum * tilingData_->dSplitCoef), dSplitSizeDy_, 0,
            static_cast<uint32_t>((tilingData_->d - tilingData_->sliceLength) * sizeof(TDY)), 0};
        DataCopyPad(dxGm_[offset], dxTensor, copyOutParams);
        dxOutQue_.FreeTensor(dxTensor);
    }
}

template <typename TDY, typename TCOS>
__aicore__ inline void InplacePartialRotaryMulGradBAB<TDY, TCOS>::Compute(
    LocalTensor<TCOS> &sinTensor, LocalTensor<TCOS> &cosTensor, LocalTensor<TDY> &dyTensor, LocalTensor<TDY> &dxTensor,
    const uint32_t currSNum, const uint32_t currNNum)
{
    InterleaveModeGradVF<TDY, TCOS>(sinTensor, cosTensor, dyTensor, dxTensor, tilingData_->sliceLength, dSplitCoef_,
                                    static_cast<uint16_t>(currSNum), static_cast<uint16_t>(currNNum));
}

} // namespace InplacePartialRotaryMulGrad
#endif
