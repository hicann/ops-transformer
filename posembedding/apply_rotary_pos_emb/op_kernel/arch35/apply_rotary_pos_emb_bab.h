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
 * \file apply_rotary_pos_emb_bab.h
 * \brief
 */

#ifndef APPLY_ROTARY_POS_EMB_BAB_H
#define APPLY_ROTARY_POS_EMB_BAB_H

#include "apply_rotary_pos_emb_common.h"

namespace ApplyRotaryPosEmb {
using namespace AscendC;

template <typename T>
class ApplyRotaryPosEmbBAB {
public:
    __aicore__ inline ApplyRotaryPosEmbBAB(TPipe *pipe, const ApplyRotaryPosEmbRegbaseTilingData *tiling)
        : pipe_(pipe),
          tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR cos, GM_ADDR sin, GM_ADDR q_out, GM_ADDR k_out,
                                GM_ADDR workspace);
    __aicore__ inline void Process();

private:
    constexpr static int32_t bufferNum = 2;
    const ApplyRotaryPosEmbRegbaseTilingData *tilingData_;
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
    GlobalTensor<T> qGm_;
    GlobalTensor<T> kGm_;
    GlobalTensor<T> cosGm_;
    GlobalTensor<T> sinGm_;
    GlobalTensor<T> qOutGm_;
    GlobalTensor<T> kOutGm_;
    TQue<QuePosition::VECIN, bufferNum> qkInQue_;
    TQue<QuePosition::VECIN, bufferNum> cosInQue_;
    TQue<QuePosition::VECIN, bufferNum> sinInQue_;
    TQue<QuePosition::VECOUT, bufferNum> qkOutQue_;

private:
    __aicore__ inline void PrePareParams();
    __aicore__ inline void ProcessNLoop(const uint32_t bIdx, const uint32_t sIdx, const uint32_t currSNum);
    __aicore__ inline void Compute(const LocalTensor<T> &sinTensor, const LocalTensor<T> &cosTensor,
                                   const LocalTensor<T> &inTensor, const LocalTensor<T> &outTensor,
                                   const uint32_t currSNum, const uint32_t currDNum);
    __aicore__ inline void ProcessQN(const LocalTensor<T> &sinTensor, const LocalTensor<T> &cosTensor,
                                     const uint32_t bIdx, const uint32_t sIdx, const uint32_t currSNum);
    __aicore__ inline void ProcessKN(const LocalTensor<T> &sinTensor, const LocalTensor<T> &cosTensor,
                                     const uint32_t bIdx, const uint32_t sIdx, const uint32_t currSNum);
    __aicore__ inline void CopyInByMode(LocalTensor<T> &tensor, GlobalTensor<T> &gm, const int64_t offset,
                                        const int64_t count, const uint32_t splitSize,
                                        const DataCopyPadExtParams<T> &padParams);
    __aicore__ inline void CopyOutByMode(LocalTensor<T> &tensor, GlobalTensor<T> &gm, const int64_t offset,
                                         const int64_t count, const uint32_t splitSize);
};

template <typename T>
__aicore__ inline void ApplyRotaryPosEmbBAB<T>::Init(GM_ADDR q, GM_ADDR k, GM_ADDR cos, GM_ADDR sin, GM_ADDR q_out,
                                                     GM_ADDR k_out, GM_ADDR workspace)
{
    this->blockIdx_ = GetBlockIdx();
    if (tilingData_->rotaryMode == static_cast<int64_t>(ApplyRotaryPosEmbRotaryMode::HALF)) {
        this->dSplitCoef_ = HALF_INTERLEAVE_COEF;
    } else if (tilingData_->rotaryMode == static_cast<int64_t>(ApplyRotaryPosEmbRotaryMode::QUARTER)) {
        this->dSplitCoef_ = QUARTER_MODE_COEF;
    }
    this->dSplitSize_ = tilingData_->realDim / dSplitCoef_ * sizeof(T);
    ubFactorS_ = tilingData_->ubFactorS;
    this->dAlign_ =
        Ops::Base::CeilAlign<int64_t>(tilingData_->realDim / dSplitCoef_, BLOCK_TYPE_SIZE / sizeof(T)) * dSplitCoef_;
    ubFactorN_ = tilingData_->ubFactorQN > tilingData_->ubFactorKN ? tilingData_->ubFactorQN : tilingData_->ubFactorKN;
    this->qGm_.SetGlobalBuffer((__gm__ T *)q);
    this->kGm_.SetGlobalBuffer((__gm__ T *)k);
    this->cosGm_.SetGlobalBuffer((__gm__ T *)cos);
    this->sinGm_.SetGlobalBuffer((__gm__ T *)sin);
    this->qOutGm_.SetGlobalBuffer((__gm__ T *)q_out);
    this->kOutGm_.SetGlobalBuffer((__gm__ T *)k_out);
    this->pipe_->InitBuffer(qkInQue_, bufferNum, ubFactorS_ * ubFactorN_ * dAlign_ * sizeof(T));
    this->pipe_->InitBuffer(cosInQue_, bufferNum, ubFactorS_ * dAlign_ * sizeof(T));
    this->pipe_->InitBuffer(sinInQue_, bufferNum, ubFactorS_ * dAlign_ * sizeof(T));
    this->pipe_->InitBuffer(qkOutQue_, bufferNum, ubFactorS_ * ubFactorN_ * dAlign_ * sizeof(T));
}

template <typename T>
__aicore__ inline void ApplyRotaryPosEmbBAB<T>::PrePareParams()
{
    bIdx_ = blockIdx_ % tilingData_->blockNumB;
    sIdx_ = blockIdx_ / tilingData_->blockNumB;
    bNum_ = tilingData_->blockFactorB;
    sNum_ = tilingData_->blockFactorS;
    if (bIdx_ == tilingData_->blockNumB - 1 && tilingData_->B % tilingData_->blockFactorB != 0) {
        bNum_ = tilingData_->B % tilingData_->blockFactorB;
    }
    if (sIdx_ == tilingData_->blockNumS - 1 && tilingData_->S % tilingData_->blockFactorS != 0) {
        sNum_ = tilingData_->S % tilingData_->blockFactorS;
    }
}

template <typename T>
__aicore__ inline void ApplyRotaryPosEmbBAB<T>::Process()
{
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
__aicore__ inline void ApplyRotaryPosEmbBAB<T>::ProcessNLoop(const uint32_t bIdx, const uint32_t sIdx,
                                                             const uint32_t currSNum)
{
    LocalTensor<T> sinTensor = sinInQue_.AllocTensor<T>();
    LocalTensor<T> cosTensor = cosInQue_.AllocTensor<T>();
    int64_t offset = sIdx * tilingData_->realDim;
    DataCopyExtParams copyParams{static_cast<uint16_t>(currSNum * dSplitCoef_), dSplitSize_, 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(sinTensor, sinGm_[offset], copyParams, padParams);
    DataCopyPad(cosTensor, cosGm_[offset], copyParams, padParams);
    sinInQue_.EnQue(sinTensor);
    cosInQue_.EnQue(cosTensor);
    sinTensor = sinInQue_.DeQue<T>();
    cosTensor = cosInQue_.DeQue<T>();
    ProcessQN(sinTensor, cosTensor, bIdx, sIdx, currSNum);
    ProcessKN(sinTensor, cosTensor, bIdx, sIdx, currSNum);
    sinInQue_.FreeTensor(sinTensor);
    cosInQue_.FreeTensor(cosTensor);
}

template <typename T>
__aicore__ inline void ApplyRotaryPosEmbBAB<T>::ProcessQN(const LocalTensor<T> &sinTensor,
                                                          const LocalTensor<T> &cosTensor, const uint32_t bIdx,
                                                          const uint32_t sIdx, const uint32_t currSNum)
{
    LocalTensor<T> qTensor;
    LocalTensor<T> qOutTensor;
    int64_t baseOffset = (bIdx * tilingData_->S + sIdx) * tilingData_->QN * tilingData_->D;
    for (uint32_t idxQn = 0; idxQn < tilingData_->ubLoopNumQN; idxQn++) {
        int64_t currDNum =
            (idxQn == tilingData_->ubLoopNumQN - 1) ? tilingData_->ubTailFactorQN : tilingData_->ubFactorQN;
        int64_t offset = baseOffset + idxQn * tilingData_->ubFactorQN * tilingData_->D;
        qTensor = qkInQue_.AllocTensor<T>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        if (tilingData_->isPartialRope) {
            CopyInByMode(qTensor, qGm_, offset, currSNum * currDNum, dSplitSize_, padParams);
        } else {
            DataCopyExtParams copyParams;
            copyParams.blockCount = currSNum * currDNum * dSplitCoef_;
            copyParams.blockLen = dSplitSize_;
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPad(qTensor, qGm_[offset], copyParams, padParams);
        }

        qkInQue_.EnQue(qTensor);
        qTensor = qkInQue_.DeQue<T>();

        qOutTensor = qkOutQue_.AllocTensor<T>();
        Compute(sinTensor, cosTensor, qTensor, qOutTensor, currSNum, currDNum);
        qkInQue_.FreeTensor(qTensor);
        qkOutQue_.EnQue(qOutTensor);
        qOutTensor = qkOutQue_.DeQue<T>();

        if (tilingData_->isPartialRope) {
            CopyOutByMode(qOutTensor, qOutGm_, offset, currSNum * currDNum, dSplitSize_);
        } else {
            DataCopyExtParams copyOutParams;
            copyOutParams.blockCount = currSNum * currDNum * dSplitCoef_;
            copyOutParams.blockLen = dSplitSize_;
            copyOutParams.srcStride = 0;
            copyOutParams.dstStride = 0;
            DataCopyPad(qOutGm_[offset], qOutTensor, copyOutParams);
        }
        qkOutQue_.FreeTensor(qOutTensor);
    }
}

template <typename T>
__aicore__ inline void ApplyRotaryPosEmbBAB<T>::ProcessKN(const LocalTensor<T> &sinTensor,
                                                          const LocalTensor<T> &cosTensor, const uint32_t bIdx,
                                                          const uint32_t sIdx, const uint32_t currSNum)
{
    LocalTensor<T> kTensor;
    LocalTensor<T> kOutTensor;
    int64_t baseOffset = (bIdx * tilingData_->S + sIdx) * tilingData_->KN * tilingData_->D;
    for (uint32_t idxKn = 0; idxKn < tilingData_->ubLoopNumKN; idxKn++) {
        int64_t currDNum =
            (idxKn == tilingData_->ubLoopNumKN - 1) ? tilingData_->ubTailFactorKN : tilingData_->ubFactorKN;
        int64_t offset = baseOffset + idxKn * tilingData_->ubFactorKN * tilingData_->D;
        kTensor = qkInQue_.AllocTensor<T>();

        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        if (tilingData_->isPartialRope) {
            CopyInByMode(kTensor, kGm_, offset, currSNum * currDNum, dSplitSize_, padParams);
        } else {
            DataCopyExtParams copyParams;
            copyParams.blockCount = currSNum * currDNum * dSplitCoef_;
            copyParams.blockLen = dSplitSize_;
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPad(kTensor, kGm_[offset], copyParams, padParams);
        }

        qkInQue_.EnQue(kTensor);
        kTensor = qkInQue_.DeQue<T>();
        kOutTensor = qkOutQue_.AllocTensor<T>();
        Compute(sinTensor, cosTensor, kTensor, kOutTensor, currSNum, currDNum);
        qkInQue_.FreeTensor(kTensor);
        qkOutQue_.EnQue(kOutTensor);
        kOutTensor = qkOutQue_.DeQue<T>();

        if (tilingData_->isPartialRope) {
            CopyOutByMode(kOutTensor, kOutGm_, offset, currSNum * currDNum, dSplitSize_);
        } else {
            DataCopyExtParams copyOutParams;
            copyOutParams.blockCount = currSNum * currDNum * dSplitCoef_;
            copyOutParams.blockLen = dSplitSize_;
            copyOutParams.srcStride = 0;
            copyOutParams.dstStride = 0;
            DataCopyPad(kOutGm_[offset], kOutTensor, copyOutParams);
        }

        qkOutQue_.FreeTensor(kOutTensor);
    }
}

template <typename T>
__aicore__ inline void ApplyRotaryPosEmbBAB<T>::Compute(const LocalTensor<T> &sinTensor,
                                                        const LocalTensor<T> &cosTensor, const LocalTensor<T> &inTensor,
                                                        const LocalTensor<T> &outTensor, const uint32_t currSNum,
                                                        const uint32_t currDNum)
{
    if (tilingData_->rotaryMode == static_cast<int64_t>(ApplyRotaryPosEmbRotaryMode::HALF)) {
        HalfAlignVF<T>(sinTensor, cosTensor, inTensor, outTensor, tilingData_->realDim, dAlign_, currSNum, currDNum);
    } else if (tilingData_->rotaryMode == static_cast<int64_t>(ApplyRotaryPosEmbRotaryMode::INTERLEAVE)) {
        InterleaveModeVF<T>(sinTensor, cosTensor, inTensor, outTensor, tilingData_->realDim, currSNum, currDNum);
    } else {
        QuarterAlignVF<T>(sinTensor, cosTensor, inTensor, outTensor, tilingData_->realDim, dAlign_, currSNum, currDNum);
    }
}

template <typename T>
__aicore__ inline void ApplyRotaryPosEmbBAB<T>::CopyInByMode(LocalTensor<T> &tensor, GlobalTensor<T> &gm,
                                                             const int64_t offset, const int64_t count,
                                                             const uint32_t splitSize,
                                                             const DataCopyPadExtParams<T> &padParams)
{
    if (tilingData_->rotaryMode == static_cast<int64_t>(ApplyRotaryPosEmbRotaryMode::HALF)) {
        CopyInHalfMode(tensor, gm, offset, count, splitSize, padParams, tilingData_->D, tilingData_->realDim,
                       static_cast<uint32_t>(dSplitCoef_));
    } else if (tilingData_->rotaryMode == static_cast<int64_t>(ApplyRotaryPosEmbRotaryMode::QUARTER)) {
        CopyInQuarterMode(tensor, gm, offset, count, splitSize, padParams, tilingData_->D, tilingData_->realDim,
                          static_cast<uint32_t>(dSplitCoef_));
    } else {
        CopyInInterleaveMode(tensor, gm, offset, count, splitSize, padParams, tilingData_->D, tilingData_->realDim);
    }
}

template <typename T>
__aicore__ inline void ApplyRotaryPosEmbBAB<T>::CopyOutByMode(LocalTensor<T> &tensor, GlobalTensor<T> &gm,
                                                              const int64_t offset, const int64_t count,
                                                              const uint32_t splitSize)
{
    if (tilingData_->rotaryMode == static_cast<int64_t>(ApplyRotaryPosEmbRotaryMode::HALF)) {
        CopyOutHalfMode(tensor, gm, offset, count, splitSize, tilingData_->D, tilingData_->realDim,
                        static_cast<uint32_t>(dSplitCoef_));
    } else if (tilingData_->rotaryMode == static_cast<int64_t>(ApplyRotaryPosEmbRotaryMode::QUARTER)) {
        CopyOutQuarterMode(tensor, gm, offset, count, splitSize, tilingData_->D, tilingData_->realDim,
                           static_cast<uint32_t>(dSplitCoef_));
    } else {
        CopyOutInterleaveMode(tensor, gm, offset, count, splitSize, tilingData_->D, tilingData_->realDim);
    }
}

} // namespace ApplyRotaryPosEmb
#endif // APPLY_ROTARY_POS_EMB_BAB_H
