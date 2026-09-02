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
 * \file apply_rotary_pos_emb_ab.h
 * \brief
 */
#ifndef APPLY_ROTARY_POS_EMB_AB_H
#define APPLY_ROTARY_POS_EMB_AB_H

#include "apply_rotary_pos_emb_common.h"

namespace ApplyRotaryPosEmb {
using namespace AscendC;

template <typename T>
class ApplyRotaryPosEmbAB {
public:
    __aicore__ inline ApplyRotaryPosEmbAB(){};
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR cos, GM_ADDR sin, GM_ADDR q_out, GM_ADDR k_out,
                                GM_ADDR workspace, const ApplyRotaryPosEmbRegbaseABTilingData *tilingData, TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessLoop(int64_t qGmOffset, int64_t kGmOffset, LocalTensor<T> cosBuffer,
                                       LocalTensor<T> sinBuffer, int64_t ubIdx);
    __aicore__ inline void ProcessQKLoop(const int64_t qkGmOffset, LocalTensor<T> cosBuffer, LocalTensor<T> sinBuffer,
                                         GlobalTensor<T> inGm, GlobalTensor<T> outGm, const uint32_t currBSNum,
                                         const int64_t count);
    __aicore__ inline void CopyInByRotaryMode(LocalTensor<T> &inTensor, GlobalTensor<T> &gmTensor, const int64_t offset,
                                              const uint32_t blockCount, const uint32_t blockLen,
                                              const DataCopyPadExtParams<T> &padParams);
    __aicore__ inline void CopyOutByRotaryMode(LocalTensor<T> &outTensor, GlobalTensor<T> &gmTensor,
                                               const int64_t offset, const uint32_t blockCount,
                                               const uint32_t blockLen);

private:
    TPipe *pipe_;
    TQue<QuePosition::VECIN, 1> qkInQueue_;
    TQue<QuePosition::VECIN, 1> cosInQueue_;
    TQue<QuePosition::VECIN, 1> sinInQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;

    GlobalTensor<T> qGm_;
    GlobalTensor<T> kGm_;
    GlobalTensor<T> cosGm_;
    GlobalTensor<T> sinGm_;
    GlobalTensor<T> qOutGm_;
    GlobalTensor<T> kOutGm_;
    const ApplyRotaryPosEmbRegbaseABTilingData *tilingData_;
    DataCopyPadExtParams<T> padParams_ = {false, 0, 0, static_cast<T>(0)};
    uint8_t DB_FLAG = 2;
    uint32_t dSplitSize_ = 0;
    uint32_t ubFactorBS_{0};
    uint32_t dSplitCoef_{1};
};

template <typename T>
__aicore__ inline void ApplyRotaryPosEmbAB<T>::Init(GM_ADDR q, GM_ADDR k, GM_ADDR cos, GM_ADDR sin, GM_ADDR q_out,
                                                    GM_ADDR k_out, GM_ADDR workspace,
                                                    const ApplyRotaryPosEmbRegbaseABTilingData *tilingData, TPipe *pipe)
{
    pipe_ = pipe;
    tilingData_ = tilingData;
    dSplitSize_ = tilingData_->realDim / tilingData_->dSplitCoef * sizeof(T);
    dSplitCoef_ = tilingData_->dSplitCoef;
    int64_t cosOffset = GetBlockIdx() * tilingData_->blockFactorBS * tilingData_->realDim;
    int64_t qOffset = GetBlockIdx() * tilingData_->blockFactorBS * tilingData_->D;
    this->cosGm_.SetGlobalBuffer((__gm__ T *)cos + cosOffset);
    this->sinGm_.SetGlobalBuffer((__gm__ T *)sin + cosOffset);
    this->qGm_.SetGlobalBuffer((__gm__ T *)q + qOffset * tilingData_->QN);
    this->kGm_.SetGlobalBuffer((__gm__ T *)k + qOffset * tilingData_->KN);
    this->qOutGm_.SetGlobalBuffer((__gm__ T *)q_out + qOffset * tilingData_->QN);
    this->kOutGm_.SetGlobalBuffer((__gm__ T *)k_out + qOffset * tilingData_->KN);

    int64_t bufferSize = tilingData_->dAlign * sizeof(T);
    ubFactorBS_ = tilingData_->ubFactorBS;
    pipe_->InitBuffer(qkInQueue_, DB_FLAG, bufferSize * ubFactorBS_ * tilingData_->ubFactorN);
    pipe_->InitBuffer(cosInQueue_, DB_FLAG, ubFactorBS_ * bufferSize);
    pipe_->InitBuffer(sinInQueue_, DB_FLAG, ubFactorBS_ * bufferSize);
    pipe_->InitBuffer(outQueue_, DB_FLAG, bufferSize * ubFactorBS_ * tilingData_->ubFactorN);
}

template <typename T>
__aicore__ inline void ApplyRotaryPosEmbAB<T>::Process()
{
    uint32_t BSNum =
        (GetBlockIdx() == tilingData_->blockNumBS - 1) ? tilingData_->blockTailBS : tilingData_->blockFactorBS;
    uint32_t bsLoopCnt = Ops::Base::CeilDiv(BSNum, ubFactorBS_);
    for (uint32_t loopIdx = 0; loopIdx < bsLoopCnt; loopIdx++) {
        uint32_t currBSNum = (loopIdx != bsLoopCnt - 1) ? ubFactorBS_ : BSNum - loopIdx * ubFactorBS_;
        DataCopyExtParams cosParams = {static_cast<uint16_t>(currBSNum * tilingData_->dSplitCoef), dSplitSize_, 0, 0,
                                       0};
        int64_t qGmOffset = loopIdx * ubFactorBS_ * tilingData_->QN * tilingData_->D;
        int64_t kGmOffset = loopIdx * ubFactorBS_ * tilingData_->KN * tilingData_->D;
        LocalTensor<T> cosBuffer = cosInQueue_.AllocTensor<T>();
        LocalTensor<T> sinBuffer = sinInQueue_.AllocTensor<T>();
        int64_t sinOffset = loopIdx * ubFactorBS_ * tilingData_->realDim;

        DataCopyPad(cosBuffer, cosGm_[sinOffset], cosParams, padParams_);
        cosInQueue_.EnQue(cosBuffer);
        cosBuffer = cosInQueue_.DeQue<T>();
        DataCopyPad(sinBuffer, sinGm_[sinOffset], cosParams, padParams_);
        sinInQueue_.EnQue(sinBuffer);
        sinBuffer = sinInQueue_.DeQue<T>();
        if (currBSNum > 1) {
            ProcessQKLoop(qGmOffset, cosBuffer, sinBuffer, qGm_, qOutGm_, currBSNum, tilingData_->QN);
            ProcessQKLoop(kGmOffset, cosBuffer, sinBuffer, kGm_, kOutGm_, currBSNum, tilingData_->KN);
        } else {
            for (int64_t n = 0; n < tilingData_->ubLoopN; n++) {
                ProcessLoop(qGmOffset, kGmOffset, cosBuffer, sinBuffer, n);
            }
        }
        cosInQueue_.FreeTensor(cosBuffer);
        sinInQueue_.FreeTensor(sinBuffer);
    }
}

template <typename T>
__aicore__ inline void ApplyRotaryPosEmbAB<T>::ProcessLoop(int64_t qGmOffset, int64_t kGmOffset,
                                                           LocalTensor<T> cosBuffer, LocalTensor<T> sinBuffer,
                                                           int64_t ubIdx)
{
    int64_t ubStart = ubIdx * tilingData_->ubFactorN;
    int64_t nCount = (ubIdx == tilingData_->ubLoopN - 1) ? tilingData_->ubTailN : tilingData_->ubFactorN;
    int64_t qCount = (tilingData_->QN - ubStart);

    DataCopyExtParams qkParams = {static_cast<uint16_t>(nCount * tilingData_->dSplitCoef), dSplitSize_, 0, 0, 0};
    DataCopyExtParams qParams = {static_cast<uint16_t>(qCount * tilingData_->dSplitCoef), dSplitSize_, 0, 0, 0};
    DataCopyExtParams kParams = {static_cast<uint16_t>((ubStart + nCount - tilingData_->QN) * tilingData_->dSplitCoef),
                                 dSplitSize_, 0, 0, 0};

    LocalTensor<T> qkBuffer = qkInQueue_.AllocTensor<T>();
    LocalTensor<T> outBuffer = outQueue_.AllocTensor<T>();

    if (tilingData_->isPartialRope) {
        if (ubStart >= tilingData_->QN) { // copy的内容全部是K
            CopyInByRotaryMode(qkBuffer, kGm_, kGmOffset + (ubStart - tilingData_->QN) * tilingData_->D, nCount,
                               dSplitSize_, padParams_);
        } else if (ubStart + nCount <= tilingData_->QN) { // copy的内容全部是Q
            CopyInByRotaryMode(qkBuffer, qGm_, qGmOffset + ubStart * tilingData_->D, nCount, dSplitSize_, padParams_);
        } else { // Q copy一部分， Q copy一部分
            CopyInByRotaryMode(qkBuffer, qGm_, qGmOffset + ubStart * tilingData_->D, qCount, dSplitSize_, padParams_);
            LocalTensor<T> qkTempBuffer = qkBuffer[qCount * tilingData_->dAlign];
            CopyInByRotaryMode(qkTempBuffer, kGm_, kGmOffset, ubStart + nCount - tilingData_->QN, dSplitSize_,
                               padParams_);
        }
    } else {
        if (ubStart >= tilingData_->QN) { // copy的内容全部是K
            DataCopyPad(qkBuffer, kGm_[kGmOffset + (ubStart - tilingData_->QN) * tilingData_->D], qkParams, padParams_);
        } else if (ubStart + nCount <= tilingData_->QN) { // copy的内容全部是Q
            DataCopyPad(qkBuffer, qGm_[qGmOffset + ubStart * tilingData_->D], qkParams, padParams_);
        } else { // Q copy一部分， Q copy一部分
            DataCopyPad(qkBuffer, qGm_[qGmOffset + ubStart * tilingData_->D], qParams, padParams_);
            DataCopyPad(qkBuffer[qCount * tilingData_->dAlign], kGm_[kGmOffset], kParams, padParams_);
        }
    }

    qkInQueue_.EnQue(qkBuffer);
    qkBuffer = qkInQueue_.DeQue<T>();

    if (tilingData_->rotaryMode == static_cast<int64_t>(ApplyRotaryPosEmbRotaryMode::HALF)) {
        HalfAlignVF(sinBuffer, cosBuffer, qkBuffer, outBuffer, tilingData_->realDim, tilingData_->dAlign, 1, nCount);
    } else if (tilingData_->rotaryMode == static_cast<int64_t>(ApplyRotaryPosEmbRotaryMode::QUARTER)) {
        QuarterAlignVF(sinBuffer, cosBuffer, qkBuffer, outBuffer, tilingData_->realDim, tilingData_->dAlign, 1, nCount);
    } else {
        InterleaveModeVF(sinBuffer, cosBuffer, qkBuffer, outBuffer, tilingData_->realDim, 1, nCount);
    }

    outQueue_.EnQue(outBuffer);
    outBuffer = outQueue_.DeQue<T>();
    qkInQueue_.FreeTensor(qkBuffer);

    if (tilingData_->isPartialRope) {
        if (ubStart >= tilingData_->QN) { // copy的内容全部是K
            CopyOutByRotaryMode(outBuffer, kOutGm_, kGmOffset + (ubStart - tilingData_->QN) * tilingData_->D, nCount,
                                dSplitSize_);
        } else if (ubStart + nCount <= tilingData_->QN) { // copy的内容全部是Q
            CopyOutByRotaryMode(outBuffer, qOutGm_, qGmOffset + ubStart * tilingData_->D, nCount, dSplitSize_);
        } else { // Q copy一部分， Q copy一部分
            CopyOutByRotaryMode(outBuffer, qOutGm_, qGmOffset + ubStart * tilingData_->D, qCount, dSplitSize_);
            LocalTensor<T> outTempBuffer = outBuffer[qCount * tilingData_->dAlign];
            CopyOutByRotaryMode(outTempBuffer, kOutGm_, kGmOffset, ubStart + nCount - tilingData_->QN, dSplitSize_);
        }
    } else {
        if (ubStart >= tilingData_->QN) {
            DataCopyPad(kOutGm_[kGmOffset + (ubStart - tilingData_->QN) * tilingData_->D], outBuffer, qkParams);
        } else if (ubStart + nCount <= tilingData_->QN) {
            DataCopyPad(qOutGm_[qGmOffset + ubStart * tilingData_->D], outBuffer, qkParams);
        } else {
            DataCopyPad(qOutGm_[qGmOffset + ubStart * tilingData_->D], outBuffer, qParams);
            DataCopyPad(kOutGm_[kGmOffset], outBuffer[qCount * tilingData_->dAlign], kParams);
        }
    }

    outQueue_.FreeTensor(outBuffer);
}

template <typename T>
__aicore__ inline void ApplyRotaryPosEmbAB<T>::ProcessQKLoop(const int64_t qkGmOffset, LocalTensor<T> cosBuffer,
                                                             LocalTensor<T> sinBuffer, GlobalTensor<T> inGm,
                                                             GlobalTensor<T> outGm, const uint32_t currBSNum,
                                                             const int64_t count)
{
    LocalTensor<T> qkBuffer = qkInQueue_.AllocTensor<T>();
    LocalTensor<T> outBuffer = outQueue_.AllocTensor<T>();
    DataCopyExtParams qkParams = {static_cast<uint16_t>(currBSNum * count * tilingData_->dSplitCoef), dSplitSize_, 0, 0,
                                  0};
    if (tilingData_->isPartialRope) {
        CopyInByRotaryMode(qkBuffer, inGm, qkGmOffset, currBSNum * count, dSplitSize_, padParams_);
    } else {
        DataCopyPad(qkBuffer, inGm[qkGmOffset], qkParams, padParams_);
    }

    qkInQueue_.EnQue(qkBuffer);
    qkBuffer = qkInQueue_.DeQue<T>();
    if (tilingData_->rotaryMode == static_cast<int64_t>(ApplyRotaryPosEmbRotaryMode::HALF)) {
        HalfAlignVF(sinBuffer, cosBuffer, qkBuffer, outBuffer, tilingData_->realDim, tilingData_->dAlign, currBSNum,
                    count);
    } else if (tilingData_->rotaryMode == static_cast<int64_t>(ApplyRotaryPosEmbRotaryMode::QUARTER)) {
        QuarterAlignVF(sinBuffer, cosBuffer, qkBuffer, outBuffer, tilingData_->realDim, tilingData_->dAlign, currBSNum,
                       count);
    } else {
        InterleaveModeVF(sinBuffer, cosBuffer, qkBuffer, outBuffer, tilingData_->realDim, currBSNum, count);
    }
    qkInQueue_.FreeTensor(qkBuffer);
    outQueue_.EnQue(outBuffer);
    outBuffer = outQueue_.DeQue<T>();

    if (tilingData_->isPartialRope) {
        CopyOutByRotaryMode(outBuffer, outGm, qkGmOffset, currBSNum * count, dSplitSize_);
    } else {
        DataCopyPad(outGm[qkGmOffset], outBuffer, qkParams);
    }
    outQueue_.FreeTensor(outBuffer);
}

template <typename T>
__aicore__ inline void ApplyRotaryPosEmbAB<T>::CopyOutByRotaryMode(LocalTensor<T> &outTensor, GlobalTensor<T> &gmTensor,
                                                                   const int64_t offset, const uint32_t blockCount,
                                                                   const uint32_t blockLen)
{
    if (tilingData_->rotaryMode == static_cast<int64_t>(ApplyRotaryPosEmbRotaryMode::HALF)) {
        CopyOutHalfMode(outTensor, gmTensor, offset, blockCount, blockLen, tilingData_->D, tilingData_->realDim,
                        dSplitCoef_);
    } else if (tilingData_->rotaryMode == static_cast<int64_t>(ApplyRotaryPosEmbRotaryMode::QUARTER)) {
        CopyOutQuarterMode(outTensor, gmTensor, offset, blockCount, blockLen, tilingData_->D, tilingData_->realDim,
                           dSplitCoef_);
    } else {
        CopyOutInterleaveMode(outTensor, gmTensor, offset, blockCount, blockLen, tilingData_->D, tilingData_->realDim);
    }
}

template <typename T>
__aicore__ inline void ApplyRotaryPosEmbAB<T>::CopyInByRotaryMode(LocalTensor<T> &inTensor, GlobalTensor<T> &gmTensor,
                                                                  const int64_t offset, const uint32_t blockCount,
                                                                  const uint32_t blockLen,
                                                                  const DataCopyPadExtParams<T> &padParams)
{
    if (tilingData_->rotaryMode == static_cast<int64_t>(ApplyRotaryPosEmbRotaryMode::HALF)) {
        CopyInHalfMode(inTensor, gmTensor, offset, blockCount, blockLen, padParams, tilingData_->D,
                       tilingData_->realDim, dSplitCoef_);
    } else if (tilingData_->rotaryMode == static_cast<int64_t>(ApplyRotaryPosEmbRotaryMode::QUARTER)) {
        CopyInQuarterMode(inTensor, gmTensor, offset, blockCount, blockLen, padParams, tilingData_->D,
                          tilingData_->realDim, dSplitCoef_);
    } else {
        CopyInInterleaveMode(inTensor, gmTensor, offset, blockCount, blockLen, padParams, tilingData_->D,
                             tilingData_->realDim);
    }
}

} // namespace ApplyRotaryPosEmb

#endif // APPLY_ROTARY_POS_EMB_AB_H
