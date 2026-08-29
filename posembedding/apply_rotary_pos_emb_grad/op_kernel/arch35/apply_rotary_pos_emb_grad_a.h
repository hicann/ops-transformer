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
 * \file apply_rotary_pos_emb_grad_a.h
 * \brief
 * MergeDim 将 B×N×S 合并为 b_，逐 b_ 分核处理 Q 和 K 的 dx (grad_query/grad_key)
 */
#ifndef APPLY_ROTARY_POS_EMB_GRAD_A_H
#define APPLY_ROTARY_POS_EMB_GRAD_A_H

#include "apply_rotary_pos_emb_grad_common.h"
#include "apply_rotary_pos_emb_grad_tiling_data.h"

namespace ApplyRotaryPosEmbGrad {
using namespace AscendC;

template <typename T>
class ApplyRotaryPosEmbGradA {
public:
    __aicore__ inline ApplyRotaryPosEmbGradA(TPipe *pipe, const ApplyRopeGradRegbaseParams *td)
        : pipe_(pipe),
          tilingData_(td)
    {}
    __aicore__ inline ~ApplyRotaryPosEmbGradA() {}

    __aicore__ inline void Init(GM_ADDR grad_query_embed, GM_ADDR grad_key_embed, GM_ADDR cos, GM_ADDR sin,
                                GM_ADDR grad_query, GM_ADDR grad_key);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitAllGlobalBuffer(GM_ADDR grad_query_embed, GM_ADDR grad_key_embed, GM_ADDR cos,
                                               GM_ADDR sin, GM_ADDR grad_query, GM_ADDR grad_key);
    __aicore__ inline void InitAllBuffer();
    __aicore__ inline void ProcessInLoop(TQue<QuePosition::VECIN, 1> &gInQ, TQue<QuePosition::VECOUT, 1> &oOutQ,
                                         GlobalTensor<T> &gGm, GlobalTensor<T> &oGm, LocalTensor<T> &cU,
                                         LocalTensor<T> &sU, int64_t pos, int64_t len);

    __aicore__ inline void CopyInCosAndSin(int64_t pos, int64_t len);
    __aicore__ inline void CopyIn(TQue<QuePosition::VECIN, 1> &inQ, GlobalTensor<T> &src, int64_t pos, int64_t len);
    __aicore__ inline void CopyOut(TQue<QuePosition::VECOUT, 1> &outQ, GlobalTensor<T> &dst, int64_t pos, int64_t len);
    __aicore__ inline void Compute(TQue<QuePosition::VECIN, 1> &gInQ, TQue<QuePosition::VECOUT, 1> &oOutQ,
                                   LocalTensor<T> &cU, LocalTensor<T> &sU, int64_t bL, int64_t nL);

private:
    TPipe *pipe_;
    const ApplyRopeGradRegbaseParams *tilingData_;
    GlobalTensor<T> gqGm_, gkGm_, cosGm_, sinGm_, oqGm_, okGm_;
    TQue<QuePosition::VECIN, 1> gradQInQ_, gradKInQ_, cosInQ_, sinInQ_;
    TQue<QuePosition::VECOUT, 1> gradQOutQ_, gradKOutQ_;
    int64_t blkIdx_, bStart_, bLen_, ubB_, ubN_, D_, dHalf_, dBuffer_;
    bool dHalfAligned_ = false;
};

template <typename T>
__aicore__ inline void ApplyRotaryPosEmbGradA<T>::Init(GM_ADDR gq, GM_ADDR gk, GM_ADDR cos, GM_ADDR sin, GM_ADDR oq,
                                                       GM_ADDR ok)
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum)
        return;
    InitAllGlobalBuffer(gq, gk, cos, sin, oq, ok);
    InitAllBuffer();
    blkIdx_ = GetBlockIdx();
    bLen_ = tilingData_->blockFactorB;
    bStart_ = blkIdx_ * tilingData_->blockFactorB;
    if (blkIdx_ == static_cast<int64_t>(tilingData_->blockNumB) - 1 && tilingData_->b % tilingData_->blockFactorB != 0)
        bLen_ = tilingData_->b % tilingData_->blockFactorB;
}

template <typename T>
__aicore__ inline void ApplyRotaryPosEmbGradA<T>::InitAllGlobalBuffer(GM_ADDR gq, GM_ADDR gk, GM_ADDR cos, GM_ADDR sin,
                                                                      GM_ADDR oq, GM_ADDR ok)
{
    gqGm_.SetGlobalBuffer((__gm__ T *)gq);
    gkGm_.SetGlobalBuffer((__gm__ T *)gk);
    cosGm_.SetGlobalBuffer((__gm__ T *)cos);
    sinGm_.SetGlobalBuffer((__gm__ T *)sin);
    oqGm_.SetGlobalBuffer((__gm__ T *)oq);
    okGm_.SetGlobalBuffer((__gm__ T *)ok);

    gqGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    gkGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
}

template <typename T>
__aicore__ inline void ApplyRotaryPosEmbGradA<T>::InitAllBuffer()
{
    ubB_ = tilingData_->ubFactorS;
    ubN_ = tilingData_->ubFactorN;
    D_ = tilingData_->d;
    dHalf_ = D_ / HALF_COEF;
    dHalfAligned_ = (dHalf_ % (BLOCK_TYPE_SIZE / sizeof(T))) == 0;
    dBuffer_ = Ops::Base::CeilAlign<int64_t>(dHalf_, BLOCK_TYPE_SIZE / sizeof(T)) * HALF_COEF;
    pipe_->InitBuffer(gradQInQ_, DOUBLE_BUFFER, ubB_ * ubN_ * dBuffer_ * sizeof(T));
    pipe_->InitBuffer(gradKInQ_, DOUBLE_BUFFER, ubB_ * ubN_ * dBuffer_ * sizeof(T));
    pipe_->InitBuffer(cosInQ_, DOUBLE_BUFFER, ubB_ * dBuffer_ * sizeof(T));
    pipe_->InitBuffer(sinInQ_, DOUBLE_BUFFER, ubB_ * dBuffer_ * sizeof(T));
    pipe_->InitBuffer(gradQOutQ_, DOUBLE_BUFFER, ubB_ * ubN_ * dBuffer_ * sizeof(T));
    pipe_->InitBuffer(gradKOutQ_, DOUBLE_BUFFER, ubB_ * ubN_ * dBuffer_ * sizeof(T));
}

template <typename T>
__aicore__ inline void ApplyRotaryPosEmbGradA<T>::Process()
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum)
        return;
    int64_t ulc = Ops::Base::CeilDiv(bLen_, ubB_);
    for (int64_t i = 0; i < ulc; i++) {
        int64_t pos = bStart_ + i * ubB_;
        int64_t len = (i != ulc - 1) ? ubB_ : bLen_ - i * ubB_;

        CopyInCosAndSin(pos, len);
        LocalTensor<T> cU = cosInQ_.DeQue<T>();
        LocalTensor<T> sU = sinInQ_.DeQue<T>();

        ProcessInLoop(gradQInQ_, gradQOutQ_, gqGm_, oqGm_, cU, sU, pos, len);
        ProcessInLoop(gradKInQ_, gradKOutQ_, gkGm_, okGm_, cU, sU, pos, len);

        cosInQ_.FreeTensor(cU);
        sinInQ_.FreeTensor(sU);
    }
}

template <typename T>
__aicore__ inline void ApplyRotaryPosEmbGradA<T>::ProcessInLoop(TQue<QuePosition::VECIN, 1> &gInQ,
                                                                TQue<QuePosition::VECOUT, 1> &oOutQ,
                                                                GlobalTensor<T> &gGm, GlobalTensor<T> &oGm,
                                                                LocalTensor<T> &cU, LocalTensor<T> &sU, int64_t pos,
                                                                int64_t len)
{
    CopyIn(gInQ, gGm, pos, len);
    Compute(gInQ, oOutQ, cU, sU, len, 1);
    CopyOut(oOutQ, oGm, pos, len);
}

template <typename T>
__aicore__ inline void ApplyRotaryPosEmbGradA<T>::CopyInCosAndSin(int64_t pos, int64_t len)
{
    LocalTensor<T> cU = cosInQ_.AllocTensor<T>();
    LocalTensor<T> sU = sinInQ_.AllocTensor<T>();
    DataCopyExtParams ep;
    if (dHalfAligned_) {
        ep = {1, static_cast<uint32_t>(len * D_ * sizeof(T)), 0, 0, 0};
    } else {
        ep = {static_cast<uint16_t>(len * HALF_COEF), static_cast<uint32_t>(dHalf_ * sizeof(T)), 0, 0, 0};
    }
    DataCopyPadExtParams<T> pp{false, 0, 0, 0};
    DataCopyPad(cU, cosGm_[pos * D_], ep, pp);
    DataCopyPad(sU, sinGm_[pos * D_], ep, pp);
    cosInQ_.EnQue(cU);
    sinInQ_.EnQue(sU);
}

template <typename T>
__aicore__ inline void ApplyRotaryPosEmbGradA<T>::CopyIn(TQue<QuePosition::VECIN, 1> &inQ, GlobalTensor<T> &src,
                                                         int64_t pos, int64_t len)
{
    LocalTensor<T> t = inQ.AllocTensor<T>();
    DataCopyExtParams ep;
    if (dHalfAligned_) {
        ep = {1, static_cast<uint32_t>(len * D_ * sizeof(T)), 0, 0, 0};
    } else {
        ep = {static_cast<uint16_t>(len * HALF_COEF), static_cast<uint32_t>(dHalf_ * sizeof(T)), 0, 0, 0};
    }
    DataCopyPadExtParams<T> pp{false, 0, 0, 0};
    DataCopyPad(t, src[pos * D_], ep, pp);
    inQ.EnQue(t);
}

template <typename T>
__aicore__ inline void ApplyRotaryPosEmbGradA<T>::CopyOut(TQue<QuePosition::VECOUT, 1> &outQ, GlobalTensor<T> &dst,
                                                          int64_t pos, int64_t len)
{
    LocalTensor<T> s = outQ.DeQue<T>();
    DataCopyExtParams ep;
    if (dHalfAligned_) {
        ep = {1, static_cast<uint32_t>(len * D_ * sizeof(T)), 0, 0, 0};
    } else {
        ep = {static_cast<uint16_t>(len * HALF_COEF), static_cast<uint32_t>(dHalf_ * sizeof(T)), 0, 0, 0};
    }
    DataCopyPad(dst[pos * D_], s, ep);
    outQ.FreeTensor(s);
}

template <typename T>
__aicore__ inline void ApplyRotaryPosEmbGradA<T>::Compute(TQue<QuePosition::VECIN, 1> &gInQ,
                                                          TQue<QuePosition::VECOUT, 1> &oOutQ, LocalTensor<T> &cU,
                                                          LocalTensor<T> &sU, int64_t bL, int64_t nL)
{
    LocalTensor<T> in = gInQ.DeQue<T>();
    LocalTensor<T> out = oOutQ.AllocTensor<T>();
    BatchHalfGradContiguousVF<T, false>((__ubuf__ T *)in.GetPhyAddr(), (__ubuf__ T *)cU.GetPhyAddr(),
                                        (__ubuf__ T *)sU.GetPhyAddr(), (__ubuf__ T *)out.GetPhyAddr(), bL, 1, nL, D_,
                                        dBuffer_, ubB_, ubN_);
    gInQ.FreeTensor(in);
    oOutQ.EnQue(out);
}

} // namespace ApplyRotaryPosEmbGrad
#endif
