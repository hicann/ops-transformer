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
 * \file inplace_partial_rotary_mul_grad_ab.h
 * \brief AB template kernel for InplacePartialRotaryMulGrad (SBND layout, BS+N blocking)
 */
#ifndef __INPLACE_PARTIAL_ROTARY_MUL_GRAD_AB_H__
#define __INPLACE_PARTIAL_ROTARY_MUL_GRAD_AB_H__

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/load_store_utils.h"
#include "inplace_partial_rotary_mul_grad_common.h"
#include <cstdint>

namespace InplacePartialRotaryMulGrad {
using namespace AscendC;

template <typename TDY, typename TCOS>
class InplacePartialRotaryMulGradAB {
public:
    __aicore__ inline InplacePartialRotaryMulGradAB(TPipe *pipe,
                                                    const InplacePartialRotaryMulGradRegbaseTilingDataAb *tiling)
        : pipe_(pipe),
          tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR cos, GM_ADDR sin, GM_ADDR dx);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessLoop(int64_t dyGmOffset, LocalTensor<TCOS> &cosBuffer, LocalTensor<TCOS> &sinBuffer,
                                       int64_t ubIdx, int64_t bsCount, int64_t nCount);
    __aicore__ inline void Compute(LocalTensor<TCOS> &sinTensor, LocalTensor<TCOS> &cosTensor,
                                   LocalTensor<TDY> &dyTensor, LocalTensor<TDY> &dxTensor, uint32_t bsCount,
                                   uint32_t nCount);

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

    const InplacePartialRotaryMulGradRegbaseTilingDataAb *tilingData_{nullptr};
    uint32_t dSplitSizeDy_{0};
    uint32_t dSplitSizeCos_{0};
    int64_t dAlignLenDy_{0};
    int64_t dAlignLenCos_{0};
    int64_t bsBlockCount_{0};
    int64_t nBlockCount_{0};
    int64_t ubFactorBS_{0};
    int64_t ubFactorN_{0};
    uint16_t dSplitCoef_{0};
};

template <typename TDY, typename TCOS>
__aicore__ inline void InplacePartialRotaryMulGradAB<TDY, TCOS>::Init(GM_ADDR dy, GM_ADDR cos, GM_ADDR sin, GM_ADDR dx)
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
    this->ubFactorBS_ = tilingData_->ubFactorBS;
    this->ubFactorN_ = tilingData_->ubFactorN;

    int64_t blockDimBS = blockIdx_ / tilingData_->blockNumN;
    int64_t blockDimN = blockIdx_ % tilingData_->blockNumN;
    bsBlockCount_ = (blockDimBS == tilingData_->blockNumBS - 1) ? tilingData_->blockTailBS : tilingData_->blockFactorBS;
    nBlockCount_ = (blockDimN == tilingData_->blockNumN - 1) ? tilingData_->blockTailN : tilingData_->blockFactorN;

    // cos/sin are stored by BS dimension contiguously (sliceLength stride per row)
    int64_t cosOffset = blockDimBS * tilingData_->blockFactorBS * tilingData_->sliceLength;
    // dy/dx: [BS, N, D] layout, offset to this block's data with sliceStart
    int64_t dyOffset = blockDimBS * tilingData_->blockFactorBS * tilingData_->n * tilingData_->d +
                       blockDimN * tilingData_->blockFactorN * tilingData_->d + tilingData_->sliceStart;

    this->dyGm_.SetGlobalBuffer((__gm__ TDY *)dy + dyOffset);
    this->dxGm_.SetGlobalBuffer((__gm__ TDY *)dx + dyOffset);
    this->cosGm_.SetGlobalBuffer((__gm__ TCOS *)cos + cosOffset);
    this->sinGm_.SetGlobalBuffer((__gm__ TCOS *)sin + cosOffset);

    this->pipe_->InitBuffer(this->dyInQue_, 2, ubFactorBS_ * ubFactorN_ * dAlignLenDy_ * sizeof(TDY));
    this->pipe_->InitBuffer(this->dxOutQue_, 2, ubFactorBS_ * ubFactorN_ * dAlignLenDy_ * sizeof(TDY));
    this->pipe_->InitBuffer(this->cosInQue_, 2, ubFactorBS_ * dAlignLenCos_ * sizeof(TCOS));
    this->pipe_->InitBuffer(this->sinInQue_, 2, ubFactorBS_ * dAlignLenCos_ * sizeof(TCOS));
}

template <typename TDY, typename TCOS>
__aicore__ inline void InplacePartialRotaryMulGradAB<TDY, TCOS>::Process()
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }

    uint32_t bsLoopCnt = Ops::Base::CeilDiv(bsBlockCount_, ubFactorBS_);
    uint32_t nLoopCnt = Ops::Base::CeilDiv(nBlockCount_, ubFactorN_);
    for (uint32_t bsLoopIdx = 0; bsLoopIdx < bsLoopCnt; bsLoopIdx++) {
        // dyGmOffset accumulates as we advance through BS groups within this block
        int64_t dyGmOffset = bsLoopIdx * ubFactorBS_ * tilingData_->n * tilingData_->d;
        uint32_t currBSNum = (bsLoopIdx != bsLoopCnt - 1) ?
                                 static_cast<uint32_t>(ubFactorBS_) :
                                 static_cast<uint32_t>(bsBlockCount_ - bsLoopIdx * ubFactorBS_);

        // Load cos/sin for current BS chunk (stored contiguously by sliceLength)
        DataCopyExtParams cosParams = {static_cast<uint16_t>(currBSNum * tilingData_->dSplitCoef), dSplitSizeCos_, 0, 0,
                                       0};
        DataCopyPadExtParams<TCOS> padParamsCos = {false, 0, 0, 0};
        LocalTensor<TCOS> cosBuffer = cosInQue_.AllocTensor<TCOS>();
        LocalTensor<TCOS> sinBuffer = sinInQue_.AllocTensor<TCOS>();

        int64_t cosSinGmOffset = bsLoopIdx * ubFactorBS_ * tilingData_->sliceLength;
        DataCopyPad(cosBuffer, cosGm_[cosSinGmOffset], cosParams, padParamsCos);
        cosInQue_.EnQue(cosBuffer);
        cosBuffer = cosInQue_.DeQue<TCOS>();
        DataCopyPad(sinBuffer, sinGm_[cosSinGmOffset], cosParams, padParamsCos);
        sinInQue_.EnQue(sinBuffer);
        sinBuffer = sinInQue_.DeQue<TCOS>();

        for (uint32_t nLoopIdx = 0; nLoopIdx < nLoopCnt; nLoopIdx++) {
            int64_t currNNum = (nLoopIdx != nLoopCnt - 1) ? ubFactorN_ : nBlockCount_ - nLoopIdx * ubFactorN_;
            ProcessLoop(dyGmOffset, cosBuffer, sinBuffer, nLoopIdx, currBSNum, currNNum);
        }

        cosInQue_.FreeTensor(cosBuffer);
        sinInQue_.FreeTensor(sinBuffer);
    }
}

template <typename TDY, typename TCOS>
__aicore__ inline void InplacePartialRotaryMulGradAB<TDY, TCOS>::ProcessLoop(int64_t dyGmOffset,
                                                                             LocalTensor<TCOS> &cosBuffer,
                                                                             LocalTensor<TCOS> &sinBuffer,
                                                                             int64_t ubIdx, int64_t bsCount,
                                                                             int64_t nCount)
{
    uint32_t totalRows = static_cast<uint32_t>(bsCount * nCount);

    // Read dy: stride over (d - sliceLength) between rows to skip elements outside the slice
    DataCopyExtParams copyInParams = {static_cast<uint16_t>(totalRows * tilingData_->dSplitCoef), dSplitSizeDy_,
                                      static_cast<uint32_t>((tilingData_->d - tilingData_->sliceLength) * sizeof(TDY)),
                                      0, 0};
    DataCopyPadExtParams<TDY> padParamsDy = {false, 0, 0, 0};

    LocalTensor<TDY> dyBuffer = dyInQue_.AllocTensor<TDY>();
    int64_t dyGmRowOffset = dyGmOffset + ubIdx * ubFactorN_ * tilingData_->d;
    DataCopyPad(dyBuffer, dyGm_[dyGmRowOffset], copyInParams, padParamsDy);
    dyInQue_.EnQue(dyBuffer);
    dyBuffer = dyInQue_.DeQue<TDY>();

    LocalTensor<TDY> dxBuffer = dxOutQue_.AllocTensor<TDY>();
    Compute(sinBuffer, cosBuffer, dyBuffer, dxBuffer, static_cast<uint32_t>(bsCount), static_cast<uint32_t>(nCount));
    dyInQue_.FreeTensor(dyBuffer);

    dxOutQue_.EnQue(dxBuffer);
    dxBuffer = dxOutQue_.DeQue<TDY>();

    // Write dx: stride over (d - sliceLength) between rows to skip elements outside the slice
    DataCopyExtParams copyOutParams = {static_cast<uint16_t>(totalRows * tilingData_->dSplitCoef), dSplitSizeDy_, 0,
                                       static_cast<uint32_t>((tilingData_->d - tilingData_->sliceLength) * sizeof(TDY)),
                                       0};
    DataCopyPad(dxGm_[dyGmRowOffset], dxBuffer, copyOutParams);
    dxOutQue_.FreeTensor(dxBuffer);
}

template <typename TDY, typename TCOS>
__aicore__ inline void InplacePartialRotaryMulGradAB<TDY, TCOS>::Compute(LocalTensor<TCOS> &sinTensor,
                                                                         LocalTensor<TCOS> &cosTensor,
                                                                         LocalTensor<TDY> &dyTensor,
                                                                         LocalTensor<TDY> &dxTensor, uint32_t bsCount,
                                                                         uint32_t nCount)
{
    InterleaveModeGradVF<TDY, TCOS>(sinTensor, cosTensor, dyTensor, dxTensor, tilingData_->sliceLength, dSplitCoef_,
                                    static_cast<uint16_t>(bsCount), static_cast<uint16_t>(nCount));
}

} // namespace InplacePartialRotaryMulGrad
#endif
