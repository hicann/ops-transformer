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
 * \file apply_rotary_pos_emb_aba_and_ba.h
 * \brief
 */
#ifndef APPLY_ROTARY_POS_EMB_ABA_AND_BA_H
#define APPLY_ROTARY_POS_EMB_ABA_AND_BA_H

#include "op_kernel/math_util.h"
#include "apply_rotary_pos_emb_common.h"

namespace ApplyRotaryPosEmb {
using namespace AscendC;

template <typename T, bool IsBBoardcast>
class ApplyRotaryPosEmbABAAndBA {
public:
    __aicore__ inline ApplyRotaryPosEmbABAAndBA(){};

    __aicore__ inline ~ApplyRotaryPosEmbABAAndBA(){};

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR cos, GM_ADDR sin, GM_ADDR qOut, GM_ADDR kOut,
                                GM_ADDR workspace, const ApplyRotaryPosEmbRegbaseTilingData *tilingData, TPipe *pipe);

    __aicore__ inline void Process();

private:
    // Init过程中使用的内部函数
    __aicore__ inline void InitAllGlobalBuffer(GM_ADDR q, GM_ADDR k, GM_ADDR cos, GM_ADDR sin, GM_ADDR qOut,
                                               GM_ADDR kOut);
    __aicore__ inline void InitAllBuffer();
    __aicore__ inline void InitLoopParams();
    // 各个层级的Process函数
    __aicore__ inline void ProcessInSLoop(int64_t sUbStart,
                                          int64_t sUbLength); // 第一重循环体，给定S范围，沿B轴进行遍历处理
    __aicore__ inline void ProcessInSBLoop(int64_t sUbStart, int64_t sUbLength, int64_t bUbStart, int64_t bUbLength,
                                           LocalTensor<T> &cos,
                                           LocalTensor<T> &sin); // 第二重循环体，给定BS范围，沿Q和K的N轴进行遍历处理
    __aicore__ inline void ProcessInSBNLoop(int64_t sUbStart, int64_t sUbLength, int64_t bUbStart, int64_t bUbLength,
                                            int64_t nUbStart, int64_t nUbLength, int64_t nTotalSize,
                                            LocalTensor<T> &cos, LocalTensor<T> &sin, GlobalTensor<T> &in,
                                            GlobalTensor<T> &out); // 第三重循环体，给定BSN范围，计算其中数据的rope
    // 拷入拷出函数
    __aicore__ inline void CopyInCosAndSin(int64_t sStart, int64_t sLength, int64_t bStart, int64_t bLength);
    __aicore__ inline void CopyInQOrK(GlobalTensor<T> &source, int64_t sStart, int64_t sLength, int64_t bStart,
                                      int64_t bLength, int64_t nStart, int64_t nLength, int64_t nTotalSize);
    __aicore__ inline void CopyOutQOrK(GlobalTensor<T> &target, int64_t sStart, int64_t sLength, int64_t bStart,
                                       int64_t bLength, int64_t nStart, int64_t nLength, int64_t nTotalSize);

    // 计算函数
    __aicore__ inline void Compute(LocalTensor<T> &cos, LocalTensor<T> &sin, int64_t sLength, int64_t bLength,
                                   int64_t nLength);

private:
    TPipe *pipe_;

    // GlobalMemory
    GlobalTensor<T> qGm_;
    GlobalTensor<T> kGm_;
    GlobalTensor<T> cosGm_;
    GlobalTensor<T> sinGm_;
    GlobalTensor<T> qOutGm_;
    GlobalTensor<T> kOutGm_;

    // UB
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> qInQueue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> cosInQueue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> sinInQueue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> qOutQueue_;

    // Split core info
    int64_t blockIdx_ = 0;
    int64_t bBlockStart_ = 0;
    int64_t bBlockLength_ = 0;
    int64_t sBlockStart_ = 0;
    int64_t sBlockLength_ = 0;

    // TilingData
    const ApplyRotaryPosEmbRegbaseTilingData *tilingData_;
    int64_t ubFactorB_ = 0;
    int64_t ubFactorS_ = 0;
    int64_t ubFactorN_ = 0;
    int64_t D_ = 0;
    int64_t dAlign_ = 0;
    uint8_t dSplitCoef_ = 1;
};

template <typename T, bool IsBBoardcast>
__aicore__ inline void ApplyRotaryPosEmbABAAndBA<T, IsBBoardcast>::Init(
    GM_ADDR q, GM_ADDR k, GM_ADDR cos, GM_ADDR sin, GM_ADDR qOut, GM_ADDR kOut, GM_ADDR workspace,
    const ApplyRotaryPosEmbRegbaseTilingData *tilingData, TPipe *pipe)
{
    this->tilingData_ = tilingData;
    this->pipe_ = pipe;
    this->blockIdx_ = GetBlockIdx();
    this->InitAllGlobalBuffer(q, k, cos, sin, qOut, kOut);
    this->InitAllBuffer();
    this->InitLoopParams();
}

template <typename T, bool IsBBoardcast>
__aicore__ inline void ApplyRotaryPosEmbABAAndBA<T, IsBBoardcast>::InitAllGlobalBuffer(GM_ADDR q, GM_ADDR k,
                                                                                       GM_ADDR cos, GM_ADDR sin,
                                                                                       GM_ADDR qOut, GM_ADDR kOut)
{
    this->qGm_.SetGlobalBuffer((__gm__ T *)q);
    this->kGm_.SetGlobalBuffer((__gm__ T *)k);
    this->cosGm_.SetGlobalBuffer((__gm__ T *)cos);
    this->sinGm_.SetGlobalBuffer((__gm__ T *)sin);
    this->qOutGm_.SetGlobalBuffer((__gm__ T *)qOut);
    this->kOutGm_.SetGlobalBuffer((__gm__ T *)kOut);
}

template <typename T, bool IsBBoardcast>
__aicore__ inline void ApplyRotaryPosEmbABAAndBA<T, IsBBoardcast>::InitAllBuffer()
{
    this->ubFactorB_ = this->tilingData_->ubFactorB;
    this->ubFactorS_ = this->tilingData_->ubFactorS;
    this->ubFactorN_ = this->tilingData_->ubFactorQN;
    this->D_ = this->tilingData_->D;
    if (tilingData_->rotaryMode == static_cast<int64_t>(ApplyRotaryPosEmbRotaryMode::HALF)) {
        this->dSplitCoef_ = HALF_INTERLEAVE_COEF;
    } else if (tilingData_->rotaryMode == static_cast<int64_t>(ApplyRotaryPosEmbRotaryMode::QUARTER)) {
        this->dSplitCoef_ = QUARTER_MODE_COEF;
    }
    this->dAlign_ =
        Ops::Base::CeilAlign<int64_t>(this->tilingData_->realDim / dSplitCoef_, BLOCK_TYPE_SIZE / sizeof(T)) *
        dSplitCoef_;

    this->pipe_->InitBuffer(this->qInQueue_, DOUBLE_BUFFER, ubFactorB_ * ubFactorS_ * ubFactorN_ * dAlign_ * sizeof(T));
    this->pipe_->InitBuffer(this->qOutQueue_, DOUBLE_BUFFER,
                            ubFactorB_ * ubFactorS_ * ubFactorN_ * dAlign_ * sizeof(T));
    if constexpr (IsBBoardcast) {
        this->pipe_->InitBuffer(this->cosInQueue_, DOUBLE_BUFFER, ubFactorS_ * dAlign_ * sizeof(T));
        this->pipe_->InitBuffer(this->sinInQueue_, DOUBLE_BUFFER, ubFactorS_ * dAlign_ * sizeof(T));
    } else {
        this->pipe_->InitBuffer(this->cosInQueue_, DOUBLE_BUFFER, ubFactorB_ * ubFactorS_ * dAlign_ * sizeof(T));
        this->pipe_->InitBuffer(this->sinInQueue_, DOUBLE_BUFFER, ubFactorB_ * ubFactorS_ * dAlign_ * sizeof(T));
    }
}

template <typename T, bool IsBBoardcast>
__aicore__ inline void ApplyRotaryPosEmbABAAndBA<T, IsBBoardcast>::InitLoopParams()
{
    int64_t bIdx = blockIdx_ % tilingData_->blockNumB;
    int64_t sIdx = blockIdx_ / tilingData_->blockNumB;
    this->bBlockLength_ = tilingData_->blockFactorB;
    this->sBlockLength_ = tilingData_->blockFactorS;
    if (bIdx == tilingData_->blockNumB - 1 && tilingData_->B % tilingData_->blockFactorB != 0) {
        this->bBlockLength_ = tilingData_->B % tilingData_->blockFactorB;
    }
    if (sIdx == tilingData_->blockNumS - 1 && tilingData_->S % tilingData_->blockFactorS != 0) {
        this->sBlockLength_ = tilingData_->S % tilingData_->blockFactorS;
    }
    this->bBlockStart_ = bIdx * tilingData_->blockFactorB;
    this->sBlockStart_ = sIdx * tilingData_->blockFactorS;
}

template <typename T, bool IsBBoardcast>
__aicore__ inline void ApplyRotaryPosEmbABAAndBA<T, IsBBoardcast>::Process()
{
    // 在S轴进行循环
    int64_t ubLoopCount = Ops::Base::CeilDiv(sBlockLength_, ubFactorS_);
    for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopCount; ubLoopIdx++) {
        this->ProcessInSLoop(sBlockStart_ + ubLoopIdx * ubFactorS_,
                             ubLoopIdx != ubLoopCount - 1 ? ubFactorS_ : sBlockLength_ - ubLoopIdx * ubFactorS_);
    }
}

template <typename T, bool IsBBoardcast>
__aicore__ inline void ApplyRotaryPosEmbABAAndBA<T, IsBBoardcast>::ProcessInSLoop(int64_t sUbStart, int64_t sUbLength)
{
    // 在B轴进行循环
    int64_t ubLoopCount = Ops::Base::CeilDiv(bBlockLength_, ubFactorB_);
    if constexpr (IsBBoardcast) {
        // cos和sin需要在B轴广播的情况
        this->CopyInCosAndSin(sUbStart, sUbLength, 0, 1);
        LocalTensor<T> cosUb = this->cosInQueue_.template DeQue<T>();
        LocalTensor<T> sinUb = this->sinInQueue_.template DeQue<T>();
        for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopCount; ubLoopIdx++) {
            this->ProcessInSBLoop(sUbStart, sUbLength, bBlockStart_ + ubLoopIdx * ubFactorB_,
                                  ubLoopIdx != ubLoopCount - 1 ? ubFactorB_ : bBlockLength_ - ubLoopIdx * ubFactorB_,
                                  cosUb, sinUb);
        }
        this->sinInQueue_.FreeTensor(cosUb);
        this->cosInQueue_.FreeTensor(sinUb);
    } else {
        // sin和cos无需在B轴广播的情况
        for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopCount; ubLoopIdx++) {
            this->CopyInCosAndSin(sUbStart, sUbLength, bBlockStart_ + ubLoopIdx * ubFactorB_,
                                  ubLoopIdx != ubLoopCount - 1 ? ubFactorB_ : bBlockLength_ - ubLoopIdx * ubFactorB_);
            LocalTensor<T> cosUb = this->cosInQueue_.template DeQue<T>();
            LocalTensor<T> sinUb = this->sinInQueue_.template DeQue<T>();
            this->ProcessInSBLoop(sUbStart, sUbLength, bBlockStart_ + ubLoopIdx * ubFactorB_,
                                  ubLoopIdx != ubLoopCount - 1 ? ubFactorB_ : bBlockLength_ - ubLoopIdx * ubFactorB_,
                                  cosUb, sinUb);
            this->cosInQueue_.FreeTensor(cosUb);
            this->sinInQueue_.FreeTensor(sinUb);
        }
    }
}

template <typename T, bool IsBBoardcast>
__aicore__ inline void ApplyRotaryPosEmbABAAndBA<T, IsBBoardcast>::ProcessInSBLoop(int64_t sUbStart, int64_t sUbLength,
                                                                                   int64_t bUbStart, int64_t bUbLength,
                                                                                   LocalTensor<T> &cos,
                                                                                   LocalTensor<T> &sin)
{
    // 循环处理Q
    int64_t qUbLoopCount = Ops::Base::CeilDiv(tilingData_->QN, ubFactorN_);
    for (int64_t ubLoopIdx = 0; ubLoopIdx < qUbLoopCount; ubLoopIdx++) {
        this->ProcessInSBNLoop(sUbStart, sUbLength, bUbStart, bUbLength, ubLoopIdx * ubFactorN_,
                               ubLoopIdx != qUbLoopCount - 1 ? ubFactorN_ : tilingData_->QN - ubLoopIdx * ubFactorN_,
                               tilingData_->QN, cos, sin, qGm_, qOutGm_);
    }
    // 循环处理K
    int64_t kUbLoopCount = Ops::Base::CeilDiv(tilingData_->KN, ubFactorN_);
    for (int64_t ubLoopIdx = 0; ubLoopIdx < kUbLoopCount; ubLoopIdx++) {
        this->ProcessInSBNLoop(sUbStart, sUbLength, bUbStart, bUbLength, ubLoopIdx * ubFactorN_,
                               ubLoopIdx != kUbLoopCount - 1 ? ubFactorN_ : tilingData_->KN - ubLoopIdx * ubFactorN_,
                               tilingData_->KN, cos, sin, kGm_, kOutGm_);
    }
}

template <typename T, bool IsBBoardcast>
__aicore__ inline void ApplyRotaryPosEmbABAAndBA<T, IsBBoardcast>::ProcessInSBNLoop(
    int64_t sUbStart, int64_t sUbLength, int64_t bUbStart, int64_t bUbLength, int64_t nUbStart, int64_t nUbLength,
    int64_t nTotalSize, LocalTensor<T> &cos, LocalTensor<T> &sin, GlobalTensor<T> &in, GlobalTensor<T> &out)
{
    CopyInQOrK(in, sUbStart, sUbLength, bUbStart, bUbLength, nUbStart, nUbLength, nTotalSize);
    Compute(cos, sin, sUbLength, bUbLength, nUbLength);
    CopyOutQOrK(out, sUbStart, sUbLength, bUbStart, bUbLength, nUbStart, nUbLength, nTotalSize);
}

template <typename T, bool IsBBoardcast>
__aicore__ inline void ApplyRotaryPosEmbABAAndBA<T, IsBBoardcast>::CopyInCosAndSin(int64_t sStart, int64_t sLength,
                                                                                   int64_t bStart, int64_t bLength)
{
    LocalTensor<T> cosUb = this->cosInQueue_.template AllocTensor<T>();
    LocalTensor<T> sinUb = this->sinInQueue_.template AllocTensor<T>();
    LoopModeParams loopParams;
    loopParams.loop2Size = 1;
    loopParams.loop1Size = bLength;
    loopParams.loop2SrcStride = 0;
    loopParams.loop2DstStride = 0;
    loopParams.loop1SrcStride = tilingData_->S * this->tilingData_->realDim * sizeof(T);
    loopParams.loop1DstStride = ubFactorS_ * dAlign_ * sizeof(T);
    SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
    DataCopyPadExtParams<T> copyPadExtparams;
    copyPadExtparams.isPad = false;
    copyPadExtparams.leftPadding = 0;
    copyPadExtparams.rightPadding = 0;
    copyPadExtparams.paddingValue = 0;
    DataCopyExtParams copyExtParams;
    copyExtParams.blockCount = sLength * dSplitCoef_;
    copyExtParams.blockLen = this->tilingData_->realDim * sizeof(T) / dSplitCoef_;
    copyExtParams.srcStride = 0;
    copyExtParams.dstStride = 0;
    DataCopyPad(
        cosUb, this->cosGm_[bStart * tilingData_->S * this->tilingData_->realDim + sStart * this->tilingData_->realDim],
        copyExtParams, copyPadExtparams);
    DataCopyPad(
        sinUb, this->sinGm_[bStart * tilingData_->S * this->tilingData_->realDim + sStart * this->tilingData_->realDim],
        copyExtParams, copyPadExtparams);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    this->cosInQueue_.template EnQue(cosUb);
    this->sinInQueue_.template EnQue(sinUb);
}

template <typename T, bool IsBBoardcast>
__aicore__ inline void ApplyRotaryPosEmbABAAndBA<T, IsBBoardcast>::CopyInQOrK(GlobalTensor<T> &source, int64_t sStart,
                                                                              int64_t sLength, int64_t bStart,
                                                                              int64_t bLength, int64_t nStart,
                                                                              int64_t nLength, int64_t nTotalSize)
{
    LocalTensor<T> target = this->qInQueue_.template AllocTensor<T>();
    // 数据格式为BNSD，B->N->S->D
    LoopModeParams loopParams;
    loopParams.loop2Size = bLength;
    loopParams.loop1Size = nLength;
    loopParams.loop2SrcStride = nTotalSize * tilingData_->S * D_ * sizeof(T);
    loopParams.loop2DstStride = ubFactorN_ * ubFactorS_ * dAlign_ * sizeof(T);
    loopParams.loop1SrcStride = tilingData_->S * D_ * sizeof(T);
    loopParams.loop1DstStride = ubFactorS_ * dAlign_ * sizeof(T);
    SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);

    DataCopyPadExtParams<T> copyPadExtparams;
    copyPadExtparams.isPad = false;
    copyPadExtparams.leftPadding = 0;
    copyPadExtparams.rightPadding = 0;
    copyPadExtparams.paddingValue = 0;

    int64_t offset = bStart * nTotalSize * tilingData_->S * D_ + nStart * tilingData_->S * D_ + sStart * D_;
    int64_t blockLen = tilingData_->realDim * sizeof(T) / dSplitCoef_;
    if (tilingData_->isPartialRope) {
        if (tilingData_->rotaryMode == static_cast<int64_t>(ApplyRotaryPosEmbRotaryMode::HALF)) {
            CopyInHalfMode(target, source, offset, sLength, blockLen, copyPadExtparams, D_, tilingData_->realDim,
                           dSplitCoef_);
        } else if (tilingData_->rotaryMode == static_cast<int64_t>(ApplyRotaryPosEmbRotaryMode::QUARTER)) {
            CopyInQuarterMode(target, source, offset, sLength, blockLen, copyPadExtparams, D_, tilingData_->realDim,
                              dSplitCoef_);
        } else {
            CopyInInterleaveMode(target, source, offset, sLength, blockLen, copyPadExtparams, D_, tilingData_->realDim);
        }
    } else {
        DataCopyExtParams copyExtParams;
        copyExtParams.blockCount = sLength * dSplitCoef_;
        copyExtParams.blockLen = D_ * sizeof(T) / dSplitCoef_;
        copyExtParams.srcStride = 0;
        copyExtParams.dstStride = 0;
        DataCopyPad(target, source[offset], copyExtParams, copyPadExtparams);
    }

    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    this->qInQueue_.template EnQue(target);
}

template <typename T, bool IsBBoardcast>
__aicore__ inline void ApplyRotaryPosEmbABAAndBA<T, IsBBoardcast>::CopyOutQOrK(GlobalTensor<T> &target, int64_t sStart,
                                                                               int64_t sLength, int64_t bStart,
                                                                               int64_t bLength, int64_t nStart,
                                                                               int64_t nLength, int64_t nTotalSize)
{
    LocalTensor<T> source = this->qOutQueue_.template DeQue<T>();
    // 数据格式为BNSD，B->N->S->D
    LoopModeParams loopParams;
    loopParams.loop2Size = bLength;
    loopParams.loop1Size = nLength;
    loopParams.loop2DstStride = nTotalSize * tilingData_->S * D_ * sizeof(T);
    loopParams.loop2SrcStride = ubFactorN_ * ubFactorS_ * dAlign_ * sizeof(T);
    loopParams.loop1DstStride = tilingData_->S * D_ * sizeof(T);
    loopParams.loop1SrcStride = ubFactorS_ * dAlign_ * sizeof(T);
    SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);

    int64_t offset = bStart * nTotalSize * tilingData_->S * D_ + nStart * tilingData_->S * D_ + sStart * D_;
    int64_t blockLen = tilingData_->realDim * sizeof(T) / dSplitCoef_;
    if (tilingData_->isPartialRope) {
        if (tilingData_->rotaryMode == static_cast<int64_t>(ApplyRotaryPosEmbRotaryMode::HALF)) {
            CopyOutHalfMode(source, target, offset, sLength, blockLen, D_, tilingData_->realDim, dSplitCoef_);
        } else if (tilingData_->rotaryMode == static_cast<int64_t>(ApplyRotaryPosEmbRotaryMode::QUARTER)) {
            CopyOutQuarterMode(source, target, offset, sLength, blockLen, D_, tilingData_->realDim, dSplitCoef_);
        } else {
            CopyOutInterleaveMode(source, target, offset, sLength, blockLen, D_, tilingData_->realDim);
        }
    } else {
        DataCopyExtParams copyExtParams;
        copyExtParams.blockCount = sLength * dSplitCoef_;
        copyExtParams.blockLen = D_ * sizeof(T) / dSplitCoef_;
        copyExtParams.srcStride = 0;
        copyExtParams.dstStride = 0;
        DataCopyPad(target[bStart * nTotalSize * tilingData_->S * D_ + nStart * tilingData_->S * D_ + sStart * D_],
                    source, copyExtParams);
    }

    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    this->qOutQueue_.FreeTensor(source);
}

template <typename T, bool IsBBoardcast>
__aicore__ inline void ApplyRotaryPosEmbABAAndBA<T, IsBBoardcast>::Compute(LocalTensor<T> &cos, LocalTensor<T> &sin,
                                                                           int64_t sLength, int64_t bLength,
                                                                           int64_t nLength)
{
    LocalTensor<T> inUb = this->qInQueue_.template DeQue<T>();
    LocalTensor<T> outUb = this->qOutQueue_.template AllocTensor<T>();
    if (tilingData_->rotaryMode == static_cast<int64_t>(ApplyRotaryPosEmbRotaryMode::HALF)) {
        BatchHalfAlignVF<T, IsBBoardcast>((__ubuf__ T *)inUb.GetPhyAddr(), (__ubuf__ T *)cos.GetPhyAddr(),
                                          (__ubuf__ T *)sin.GetPhyAddr(), (__ubuf__ T *)outUb.GetPhyAddr(), sLength,
                                          bLength, nLength, tilingData_->realDim, dAlign_, ubFactorS_, ubFactorN_);
    } else if (tilingData_->rotaryMode == static_cast<int64_t>(ApplyRotaryPosEmbRotaryMode::INTERLEAVE)) {
        BatchInterleaveModeVF<T, IsBBoardcast>((__ubuf__ T *)inUb.GetPhyAddr(), (__ubuf__ T *)cos.GetPhyAddr(),
                                               (__ubuf__ T *)sin.GetPhyAddr(), (__ubuf__ T *)outUb.GetPhyAddr(),
                                               sLength, bLength, nLength, tilingData_->realDim, dAlign_, ubFactorS_,
                                               ubFactorN_);
    } else {
        BatchQuarterAlignVF<T, IsBBoardcast>((__ubuf__ T *)inUb.GetPhyAddr(), (__ubuf__ T *)cos.GetPhyAddr(),
                                             (__ubuf__ T *)sin.GetPhyAddr(), (__ubuf__ T *)outUb.GetPhyAddr(), sLength,
                                             bLength, nLength, tilingData_->realDim, dAlign_, ubFactorS_, ubFactorN_);
    }
    this->qInQueue_.FreeTensor(inUb);
    this->qOutQueue_.template EnQue(outUb);
}

} // namespace ApplyRotaryPosEmb

#endif
