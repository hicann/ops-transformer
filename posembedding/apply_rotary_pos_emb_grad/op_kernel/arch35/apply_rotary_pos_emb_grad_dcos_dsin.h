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
 * \file apply_rotary_pos_emb_grad_dcos_dsin.h
 * \brief A 模板的 grad_cos/grad_sin 计算
 */
#ifndef APPLY_ROTARY_POS_EMB_GRAD_DCOS_DSIN_H
#define APPLY_ROTARY_POS_EMB_GRAD_DCOS_DSIN_H

#include "apply_rotary_pos_emb_grad_common.h"
#include "apply_rotary_pos_emb_grad_tiling_data.h"

namespace ApplyRotaryPosEmbGrad {
using namespace AscendC;

template <typename T>
__aicore__ inline void HalfRotaryVF(const LocalTensor<T> &inTensor, const LocalTensor<T> &rotaryTensor, uint32_t dLen,
                                    uint32_t dAlign, uint16_t currDNum)
{
    __ubuf__ T *inUb = (__ubuf__ T *)inTensor.GetPhyAddr();
    __ubuf__ T *outUb = (__ubuf__ T *)rotaryTensor.GetPhyAddr();
    uint32_t halfD = dLen / HALF_COEF;
    uint32_t halfDAlign = dAlign / HALF_COEF;
    uint16_t repeatTimes = Ops::Base::CeilDiv(halfD, VL_FLOAT32_SIZE);
    __ubuf__ T *currInUb, *currOutUb;

    __VEC_SCOPE__
    {
        Reg::RegTensor<float> vregIn, vregHalfIn;
        Reg::MaskReg preg;
        for (uint16_t idxD = 0; idxD < currDNum; idxD++) {
            currInUb = inUb + idxD * dAlign;
            currOutUb = outUb + idxD * dAlign;
            uint32_t updateCnt = halfD;
            for (uint16_t i = 0; i < repeatTimes; i++) {
                preg = Reg::UpdateMask<float>(updateCnt);
                int32_t offset = i * VL_FLOAT32_SIZE;
                int32_t halfOffset = offset + halfDAlign;
                ops::LoadTwoTensorForDtypeT<T>(currInUb, currInUb, vregIn, vregHalfIn, preg, preg, offset, halfOffset);
                Muls(vregHalfIn, vregHalfIn, -1.0f, preg);
                ops::StoreOneTensorForDtypeT<T>(currOutUb, vregHalfIn, preg, offset);
                ops::StoreOneTensorForDtypeT<T>(currOutUb, vregIn, preg, halfOffset);
            }
        }
    }
}

template <typename T>
class ApplyRotaryXDual {
public:
    __aicore__ inline ApplyRotaryXDual(TPipe *pipe, const ApplyRopeGradRegbaseParams *td)
        : pipe_(pipe),
          tilingData_(td)
    {}
    __aicore__ inline ~ApplyRotaryXDual() {}

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR workspace);
    __aicore__ inline void Process();

private:
    __aicore__ inline void RotateOne(GlobalTensor<T> &xGm, GlobalTensor<T> &wsGm);
    __aicore__ inline void CopyIn(GlobalTensor<T> &src, int64_t pos, int64_t len);
    __aicore__ inline void CopyOut(GlobalTensor<T> &dst, int64_t pos, int64_t len);

private:
    TPipe *pipe_;
    const ApplyRopeGradRegbaseParams *tilingData_;
    GlobalTensor<T> qGm_, kGm_, wsQGm_, wsKGm_;
    TQue<QuePosition::VECIN, 1> xInQ_;
    TQue<QuePosition::VECOUT, 1> rotOutQ_;
    int64_t blkIdx_, bStart_, bLen_, ubB_, D_, dHalf_, dBuffer_, bTotal_;
};

template <typename T>
__aicore__ inline void ApplyRotaryXDual<T>::Init(GM_ADDR query, GM_ADDR key, GM_ADDR workspace)
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum)
        return;
    D_ = tilingData_->d;
    bTotal_ = tilingData_->b;
    dHalf_ = D_ / HALF_COEF;
    dBuffer_ = Ops::Base::CeilAlign<int64_t>(dHalf_, BLOCK_TYPE_SIZE / sizeof(T)) * HALF_COEF;
    ubB_ = tilingData_->ubFactorS;

    qGm_.SetGlobalBuffer((__gm__ T *)query);
    kGm_.SetGlobalBuffer((__gm__ T *)key);
    wsQGm_.SetGlobalBuffer((__gm__ T *)workspace, bTotal_ * D_);
    wsKGm_.SetGlobalBuffer((__gm__ T *)workspace + bTotal_ * D_, bTotal_ * D_);

    pipe_->InitBuffer(xInQ_, DOUBLE_BUFFER, ubB_ * dBuffer_ * sizeof(T));
    pipe_->InitBuffer(rotOutQ_, DOUBLE_BUFFER, ubB_ * dBuffer_ * sizeof(T));

    blkIdx_ = GetBlockIdx();
    bLen_ = tilingData_->blockFactorB;
    bStart_ = blkIdx_ * tilingData_->blockFactorB;
    if (blkIdx_ == tilingData_->blockNumB - 1 && tilingData_->b % tilingData_->blockFactorB != 0) {
        bLen_ = tilingData_->b % tilingData_->blockFactorB;
    }
}

template <typename T>
__aicore__ inline void ApplyRotaryXDual<T>::Process()
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum)
        return;
    RotateOne(qGm_, wsQGm_);
    RotateOne(kGm_, wsKGm_);
}

template <typename T>
__aicore__ inline void ApplyRotaryXDual<T>::RotateOne(GlobalTensor<T> &xGm, GlobalTensor<T> &wsGm)
{
    int64_t ulc = Ops::Base::CeilDiv(bLen_, ubB_);
    for (int64_t i = 0; i < ulc; i++) {
        int64_t pos = bStart_ + i * ubB_;
        int64_t len = (i != ulc - 1) ? ubB_ : bLen_ - i * ubB_;
        CopyIn(xGm, pos, len);
        LocalTensor<T> xU = xInQ_.template DeQue<T>();
        LocalTensor<T> rU = rotOutQ_.template AllocTensor<T>();
        HalfRotaryVF<T>(xU, rU, static_cast<uint32_t>(D_), static_cast<uint32_t>(dBuffer_), static_cast<uint16_t>(len));
        rotOutQ_.EnQue(rU);
        CopyOut(wsGm, pos, len);
        xInQ_.FreeTensor(xU);
    }
}

template <typename T>
__aicore__ inline void ApplyRotaryXDual<T>::CopyIn(GlobalTensor<T> &src, int64_t pos, int64_t len)
{
    LocalTensor<T> t = xInQ_.template AllocTensor<T>();
    DataCopyExtParams ep{static_cast<uint16_t>(len * HALF_COEF), static_cast<uint32_t>(dHalf_ * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> pp{false, 0, 0, 0};
    DataCopyPad(t, src[pos * D_], ep, pp);
    xInQ_.EnQue(t);
}

template <typename T>
__aicore__ inline void ApplyRotaryXDual<T>::CopyOut(GlobalTensor<T> &dst, int64_t pos, int64_t len)
{
    LocalTensor<T> s = rotOutQ_.template DeQue<T>();
    DataCopyExtParams ep{static_cast<uint16_t>(len * HALF_COEF), static_cast<uint32_t>(dHalf_ * sizeof(T)), 0, 0, 0};
    DataCopyPad(dst[pos * D_], s, ep);
    rotOutQ_.FreeTensor(s);
}

template <typename T>
class ApplyDcosDsin {
public:
    __aicore__ inline ApplyDcosDsin(TPipe *pipe, const ApplyRopeGradRegbaseParams *td)
        : pipe_(pipe),
          tilingData_(td)
    {}
    __aicore__ inline ~ApplyDcosDsin() {}

    __aicore__ inline void Init(GM_ADDR gradQ, GM_ADDR gradK, GM_ADDR query, GM_ADDR key, GM_ADDR wsQ, GM_ADDR wsK,
                                GM_ADDR gradCosOut, GM_ADDR gradSinOut);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(TQue<QuePosition::VECIN, 1> &inQ, GlobalTensor<T> &src, int64_t pos, int64_t len);
    __aicore__ inline void CopyOut(TQue<QuePosition::VECOUT, 1> &outQ, GlobalTensor<T> &dst, int64_t pos, int64_t len);
    __aicore__ inline LocalTensor<float> CastInputToFloat(LocalTensor<T> &input, uint32_t elemCnt);
    __aicore__ inline void ProcessTile(int64_t pos, int64_t len);

private:
    TPipe *pipe_;
    const ApplyRopeGradRegbaseParams *tilingData_;
    GlobalTensor<T> gqGm_, gkGm_, qGm_, kGm_, wsQGm_, wsKGm_, ocGm_, osGm_;
    TQue<QuePosition::VECIN, 1> gradInQQ_, xInQQ_, rotInQQ_;
    TQue<QuePosition::VECIN, 1> gradInQK_, xInQK_, rotInQK_;
    TQue<QuePosition::VECOUT, 1> cosOutQ_, sinOutQ_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    int64_t blkIdx_, bStart_, bLen_, ubB_, D_, inputTOffset_;
};

template <typename T>
__aicore__ inline void ApplyDcosDsin<T>::Init(GM_ADDR gradQ, GM_ADDR gradK, GM_ADDR query, GM_ADDR key, GM_ADDR wsQ,
                                              GM_ADDR wsK, GM_ADDR gradCosOut, GM_ADDR gradSinOut)
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum)
        return;
    D_ = tilingData_->d;
    ubB_ = tilingData_->ubFactorS;
    inputTOffset_ = 0;
    if constexpr (sizeof(T) != sizeof(float)) {
        inputTOffset_ = Ops::Base::CeilAlign<int64_t>(ubB_ * D_ * sizeof(float), BLOCK_TYPE_SIZE) / sizeof(T);
    }
    gqGm_.SetGlobalBuffer((__gm__ T *)gradQ);
    gkGm_.SetGlobalBuffer((__gm__ T *)gradK);
    qGm_.SetGlobalBuffer((__gm__ T *)query);
    kGm_.SetGlobalBuffer((__gm__ T *)key);
    wsQGm_.SetGlobalBuffer((__gm__ T *)wsQ);
    wsKGm_.SetGlobalBuffer((__gm__ T *)wsK);
    ocGm_.SetGlobalBuffer((__gm__ T *)gradCosOut);
    osGm_.SetGlobalBuffer((__gm__ T *)gradSinOut);
    int64_t inputBufferSize = ubB_ * D_ * sizeof(T);
    if constexpr (sizeof(T) != sizeof(float)) {
        inputBufferSize =
            Ops::Base::CeilAlign<int64_t>(inputTOffset_ * sizeof(T) + ubB_ * D_ * sizeof(T), BLOCK_TYPE_SIZE);
    }
    pipe_->InitBuffer(gradInQQ_, DOUBLE_BUFFER, inputBufferSize);
    pipe_->InitBuffer(xInQQ_, DOUBLE_BUFFER, inputBufferSize);
    pipe_->InitBuffer(rotInQQ_, DOUBLE_BUFFER, inputBufferSize);
    pipe_->InitBuffer(gradInQK_, DOUBLE_BUFFER, inputBufferSize);
    pipe_->InitBuffer(xInQK_, DOUBLE_BUFFER, inputBufferSize);
    pipe_->InitBuffer(rotInQK_, DOUBLE_BUFFER, inputBufferSize);
    pipe_->InitBuffer(cosOutQ_, DOUBLE_BUFFER, ubB_ * D_ * sizeof(float));
    pipe_->InitBuffer(sinOutQ_, DOUBLE_BUFFER, ubB_ * D_ * sizeof(float));
    pipe_->InitBuffer(tmpBuf_, ubB_ * D_ * sizeof(float));
    blkIdx_ = GetBlockIdx();
    bLen_ = tilingData_->blockFactorB;
    bStart_ = blkIdx_ * tilingData_->blockFactorB;
    if (blkIdx_ == tilingData_->blockNumB - 1 && tilingData_->b % tilingData_->blockFactorB != 0) {
        bLen_ = tilingData_->b % tilingData_->blockFactorB;
    }
}

template <typename T>
__aicore__ inline void ApplyDcosDsin<T>::Process()
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum)
        return;
    int64_t ulc = Ops::Base::CeilDiv(bLen_, ubB_);
    for (int64_t i = 0; i < ulc; i++) {
        int64_t pos = bStart_ + i * ubB_;
        int64_t len = (i != ulc - 1) ? ubB_ : bLen_ - i * ubB_;
        ProcessTile(pos, len);
    }
}

template <typename T>
__aicore__ inline void ApplyDcosDsin<T>::ProcessTile(int64_t pos, int64_t len)
{
    uint32_t elemCnt = static_cast<uint32_t>(len * D_);
    CopyIn(gradInQQ_, gqGm_, pos, len);
    CopyIn(xInQQ_, qGm_, pos, len);
    CopyIn(rotInQQ_, wsQGm_, pos, len);
    LocalTensor<T> gradQUb = gradInQQ_.template DeQue<T>();
    LocalTensor<T> queryUb = xInQQ_.template DeQue<T>();
    LocalTensor<T> rotQUb = rotInQQ_.template DeQue<T>();
    LocalTensor<float> gradQFloat = CastInputToFloat(gradQUb, elemCnt);
    LocalTensor<float> queryFloat = CastInputToFloat(queryUb, elemCnt);
    LocalTensor<float> rotQFloat = CastInputToFloat(rotQUb, elemCnt);
    PipeBarrier<PIPE_V>();
    LocalTensor<T> cosOut = cosOutQ_.template AllocTensor<T>();
    LocalTensor<T> sinOut = sinOutQ_.template AllocTensor<T>();
    LocalTensor<float> cosAcc = cosOut.template ReinterpretCast<float>();
    LocalTensor<float> sinAcc = sinOut.template ReinterpretCast<float>();
    Mul(cosAcc, queryFloat, gradQFloat, elemCnt);
    Mul(sinAcc, rotQFloat, gradQFloat, elemCnt);
    gradInQQ_.FreeTensor(gradQUb);
    xInQQ_.FreeTensor(queryUb);
    rotInQQ_.FreeTensor(rotQUb);

    CopyIn(gradInQK_, gkGm_, pos, len);
    CopyIn(xInQK_, kGm_, pos, len);
    CopyIn(rotInQK_, wsKGm_, pos, len);
    LocalTensor<T> gradKUb = gradInQK_.template DeQue<T>();
    LocalTensor<T> keyUb = xInQK_.template DeQue<T>();
    LocalTensor<T> rotKUb = rotInQK_.template DeQue<T>();
    LocalTensor<float> gradKFloat = CastInputToFloat(gradKUb, elemCnt);
    LocalTensor<float> keyFloat = CastInputToFloat(keyUb, elemCnt);
    LocalTensor<float> rotKFloat = CastInputToFloat(rotKUb, elemCnt);
    PipeBarrier<PIPE_V>();
    LocalTensor<float> tmp = tmpBuf_.Get<float>();
    Mul(tmp, keyFloat, gradKFloat, elemCnt);
    Add(cosAcc, cosAcc, tmp, elemCnt);
    Mul(tmp, rotKFloat, gradKFloat, elemCnt);
    Add(sinAcc, sinAcc, tmp, elemCnt);
    gradInQK_.FreeTensor(gradKUb);
    xInQK_.FreeTensor(keyUb);
    rotInQK_.FreeTensor(rotKUb);

    if constexpr (sizeof(T) != sizeof(float)) {
        Cast(cosOut, cosAcc, RoundMode::CAST_RINT, elemCnt);
        Cast(sinOut, sinAcc, RoundMode::CAST_RINT, elemCnt);
        PipeBarrier<PIPE_V>();
    }
    cosOutQ_.EnQue(cosOut);
    sinOutQ_.EnQue(sinOut);
    CopyOut(cosOutQ_, ocGm_, pos, len);
    CopyOut(sinOutQ_, osGm_, pos, len);
}

template <typename T>
__aicore__ inline void ApplyDcosDsin<T>::CopyIn(TQue<QuePosition::VECIN, 1> &inQ, GlobalTensor<T> &src, int64_t pos,
                                                int64_t len)
{
    LocalTensor<T> t = inQ.template AllocTensor<T>();
    DataCopyExtParams ep{1, static_cast<uint32_t>(len * D_ * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> pp{false, 0, 0, 0};
    if constexpr (sizeof(T) != sizeof(float)) {
        DataCopyPad(t[inputTOffset_], src[pos * D_], ep, pp);
    } else {
        DataCopyPad(t, src[pos * D_], ep, pp);
    }
    inQ.EnQue(t);
}

template <typename T>
__aicore__ inline void ApplyDcosDsin<T>::CopyOut(TQue<QuePosition::VECOUT, 1> &outQ, GlobalTensor<T> &dst, int64_t pos,
                                                 int64_t len)
{
    LocalTensor<T> s = outQ.template DeQue<T>();
    DataCopyExtParams ep{1, static_cast<uint32_t>(len * D_ * sizeof(T)), 0, 0, 0};
    DataCopyPad(dst[pos * D_], s, ep);
    outQ.FreeTensor(s);
}

template <typename T>
__aicore__ inline LocalTensor<float> ApplyDcosDsin<T>::CastInputToFloat(LocalTensor<T> &input, uint32_t elemCnt)
{
    LocalTensor<float> inputFloat = input.template ReinterpretCast<float>();
    if constexpr (sizeof(T) != sizeof(float)) {
        Cast(inputFloat, input[inputTOffset_], RoundMode::CAST_NONE, elemCnt);
    }
    return inputFloat;
}

} // namespace ApplyRotaryPosEmbGrad

#endif // APPLY_ROTARY_POS_EMB_GRAD_DCOS_DSIN_H
