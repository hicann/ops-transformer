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
 * \file moe_gating_top_k_hash_regbase.h
 * \brief Arch35 register-based kernel for MoE Gating TopK hash mode.
 */
#ifndef MOE_GATING_TOP_K_HASH_REGBASE_H
#define MOE_GATING_TOP_K_HASH_REGBASE_H

#include "moe_gating_top_k_regbase.h"

namespace MoeGatingTopK {
using namespace AscendC;
using Reg::RegTensor;

template <typename T, typename U1, typename U2>
class MoeGatingTopKHashRegbase : public MoeGatingTopKRegbase<T> {
public:
    __aicore__ inline MoeGatingTopKHashRegbase(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR inputIds, GM_ADDR tid2eid, GM_ADDR y,
                                GM_ADDR expertIdx, GM_ADDR out, GM_ADDR workspace,
                                const MoeGatingTopKRegbaseTilingData* tilingData, TPipe* tPipe);
    __aicore__ inline void Process();

private:
    using MoeGatingTopKRegbase<T>::CopyInBias;
    using MoeGatingTopKRegbase<T>::CopyInX;
    using MoeGatingTopKRegbase<T>::ComputeX;
    using MoeGatingTopKRegbase<T>::CopyOutXNorm;
    using MoeGatingTopKRegbase<T>::CopyOut;
    using MoeGatingTopKRegbase<T>::expertIdxOutQueue_;
    using MoeGatingTopKRegbase<T>::sortedInGroupBuf_;
    using MoeGatingTopKRegbase<T>::xSigmoidBuf_;
    using MoeGatingTopKRegbase<T>::yOutQueue_;
    using MoeGatingTopKRegbase<T>::k_;
    using MoeGatingTopKRegbase<T>::eps_;
    using MoeGatingTopKRegbase<T>::routedScalingFactor_;
    using MoeGatingTopKRegbase<T>::noNorm_;
    using MoeGatingTopKRegbase<T>::perGroupExpertCount_;
    using MoeGatingTopKRegbase<T>::perGroupExpertCountAlign_;
    using MoeGatingTopKRegbase<T>::curCoreRowCount_;
    using MoeGatingTopKRegbase<T>::tilingData_;
    using MoeGatingTopKRegbase<T>::blockIdx_;

    __aicore__ inline void HashCompute(int64_t row);
    __aicore__ inline void HashGatherWithSmallKAlignE(LocalTensor<float> xSigmoidTensor,
                                                      LocalTensor<int32_t> expertIdxTensor, LocalTensor<T> yTensor,
                                                      uint32_t k, float eps, float routedScalingFactor);
    __aicore__ inline void HashGatherWithSmallKNotAlignE(LocalTensor<float> xSigmoidTensor,
                                                         LocalTensor<int32_t> expertIdxTensor, LocalTensor<T> yTensor,
                                                         uint32_t k, float eps, float routedScalingFactor,
                                                         int32_t expertIdxPad, int32_t perGroupExpertCountAlign);

private:
    GlobalTensor<U1> inputIdsGm_;
    GlobalTensor<U2> tid2eidGm_;
};

template <typename T, typename U1, typename U2>
__aicore__ inline void MoeGatingTopKHashRegbase<T, U1, U2>::HashGatherWithSmallKAlignE(
    LocalTensor<float> xSigmoidTensor, LocalTensor<int32_t> expertIdxTensor, LocalTensor<T> yTensor, uint32_t k,
    float eps, float routedScalingFactor)
{
    __ubuf__ float* inputAddr = (__ubuf__ float*)xSigmoidTensor.GetPhyAddr();
    __ubuf__ uint32_t* expertIdxAddr = (__ubuf__ uint32_t*)expertIdxTensor.GetPhyAddr();
    __ubuf__ T* outputAddr = (__ubuf__ T*)yTensor.GetPhyAddr();

    if (noNorm_) {
        __VEC_SCOPE__
        {
            RegTensor<uint32_t> vreg1;
            RegTensor<float> vreg2;

            Reg::MaskReg preg0 = Reg::UpdateMask<uint32_t>(k);
            Reg::LoadAlign(vreg1, expertIdxAddr);
            Reg::Gather(vreg2, inputAddr, vreg1, preg0);
            Reg::Muls(vreg2, vreg2, routedScalingFactor, preg0);
            ops::StoreOneTensorForDtypeT<T>(outputAddr, vreg2, preg0, 0);
        }
    } else {
        __VEC_SCOPE__
        {
            RegTensor<uint32_t> vreg1;
            RegTensor<float> vreg2;
            RegTensor<float> vreg3;
            RegTensor<float> vreg4;

            Reg::MaskReg preg0 = Reg::UpdateMask<uint32_t>(k);
            Reg::LoadAlign(vreg1, expertIdxAddr);
            Reg::Gather(vreg2, inputAddr, vreg1, preg0);
            Reg::Reduce<Reg::ReduceType::SUM>(vreg3, vreg2, preg0);
            Reg::Adds(vreg3, vreg3, eps, preg0);
            Reg::Duplicate(vreg4, vreg3, preg0);
            Reg::Div(vreg4, vreg2, vreg4, preg0);
            Reg::Muls(vreg4, vreg4, routedScalingFactor, preg0);
            ops::StoreOneTensorForDtypeT<T>(outputAddr, vreg4, preg0, 0);
        }
    }
}

template <typename T, typename U1, typename U2>
__aicore__ inline void MoeGatingTopKHashRegbase<T, U1, U2>::HashGatherWithSmallKNotAlignE(
    LocalTensor<float> xSigmoidTensor, LocalTensor<int32_t> expertIdxTensor, LocalTensor<T> yTensor, uint32_t k,
    float eps, float routedScalingFactor, int32_t expertIdxPad, int32_t perGroupExpertCountAlign)
{
    __ubuf__ float* inputAddr = (__ubuf__ float*)xSigmoidTensor.GetPhyAddr();
    __ubuf__ uint32_t* expertIdxAddr = (__ubuf__ uint32_t*)expertIdxTensor.GetPhyAddr();
    __ubuf__ T* outputAddr = (__ubuf__ T*)yTensor.GetPhyAddr();

    if (noNorm_) {
        __VEC_SCOPE__
        {
            RegTensor<uint32_t> vreg1;
            RegTensor<float> vreg2;
            RegTensor<uint32_t> vregAlign;

            Reg::MaskReg preg0 = Reg::UpdateMask<float>(k);
            Reg::Duplicate(vregAlign, perGroupExpertCountAlign, preg0);

            Reg::LoadAlign(vreg1, expertIdxAddr);
            Reg::Gather(vreg2, inputAddr, vreg1, preg0);
            Reg::Muls(vreg2, vreg2, routedScalingFactor, preg0);

            Reg::Div(vregAlign, vreg1, vregAlign, preg0);
            Reg::Muls(vregAlign, vregAlign, expertIdxPad, preg0);
            Reg::Sub(vreg1, vreg1, vregAlign, preg0);

            ops::StoreOneTensorForDtypeT<T>(outputAddr, vreg2, preg0, 0);
        }
    } else {
        __VEC_SCOPE__
        {
            RegTensor<uint32_t> vreg1;
            RegTensor<float> vreg2;
            RegTensor<float> vreg3;
            RegTensor<float> vreg4;
            RegTensor<uint32_t> vregAlign;

            Reg::MaskReg preg0 = Reg::UpdateMask<float>(k);
            Reg::Duplicate(vregAlign, perGroupExpertCountAlign, preg0);

            Reg::LoadAlign(vreg1, expertIdxAddr);
            Reg::Gather(vreg2, inputAddr, vreg1, preg0);
            Reg::Reduce<Reg::ReduceType::SUM>(vreg3, vreg2, preg0);
            Reg::Adds(vreg3, vreg3, eps, preg0);
            Reg::Duplicate(vreg4, vreg3, preg0);
            Reg::Div(vreg4, vreg2, vreg4, preg0);
            Reg::Muls(vreg4, vreg4, routedScalingFactor, preg0);

            Reg::Div(vregAlign, vreg1, vregAlign, preg0);
            Reg::Muls(vregAlign, vregAlign, expertIdxPad, preg0);
            Reg::Sub(vreg1, vreg1, vregAlign, preg0);

            ops::StoreOneTensorForDtypeT<T>(outputAddr, vreg4, preg0, 0);
        }
    }
}

template <typename T, typename U1, typename U2>
__aicore__ inline void MoeGatingTopKHashRegbase<T, U1, U2>::HashCompute(int64_t row)
{
    LocalTensor<int32_t> expertIdxTensor = expertIdxOutQueue_.template AllocTensor<int32_t>();
    LocalTensor<U2> hashExpertId = sortedInGroupBuf_.template Get<U2>();
    LocalTensor<int32_t> hashExpertIdInt32 = hashExpertId.template ReinterpretCast<int32_t>();

    U1 key = inputIdsGm_.GetValue(row);
    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(k_ * sizeof(U2)), 0, 0, 0};
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<U2>(0)};
    DataCopyPad(hashExpertId, tid2eidGm_[key * k_], dataCopyParams, dataCopyPadParams);
    SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    if constexpr (IsSameType<U2, int32_t>::value) {
        DataCopy(expertIdxTensor, hashExpertId, Align(k_, sizeof(int32_t)));
    } else {
        Cast(hashExpertIdInt32, hashExpertId, RoundMode::CAST_NONE, Align(k_, sizeof(U2)));
        PipeBarrier<PIPE_V>();
        DataCopy(expertIdxTensor, hashExpertIdInt32, Align(k_, sizeof(int32_t)));
    }
    PipeBarrier<PIPE_V>();

    LocalTensor<float> xSigmoidTensor = xSigmoidBuf_.template Get<float>();
    LocalTensor<T> yTensor = yOutQueue_.template AllocTensor<T>();
    int32_t expertIdxPad = perGroupExpertCountAlign_ - perGroupExpertCount_;
    if (expertIdxPad != 0) {
        HashGatherWithSmallKNotAlignE(xSigmoidTensor, expertIdxTensor, yTensor, k_, eps_, routedScalingFactor_,
                                      expertIdxPad, perGroupExpertCountAlign_);
    } else {
        HashGatherWithSmallKAlignE(xSigmoidTensor, expertIdxTensor, yTensor, k_, eps_, routedScalingFactor_);
    }
    yOutQueue_.EnQue(yTensor);
    expertIdxOutQueue_.template EnQue<int32_t>(expertIdxTensor);
}

template <typename T, typename U1, typename U2>
__aicore__ inline void MoeGatingTopKHashRegbase<T, U1, U2>::Init(GM_ADDR x, GM_ADDR bias, GM_ADDR inputIds,
                                                                 GM_ADDR tid2eid, GM_ADDR y, GM_ADDR expertIdx,
                                                                 GM_ADDR out, GM_ADDR workspace,
                                                                 const MoeGatingTopKRegbaseTilingData* tilingData,
                                                                 TPipe* tPipe)
{
    MoeGatingTopKRegbase<T>::Init(x, bias, y, expertIdx, out, workspace, tilingData, tPipe);
    inputIdsGm_.SetGlobalBuffer((__gm__ U1*)inputIds);
    tid2eidGm_.SetGlobalBuffer((__gm__ U2*)tid2eid);
}

template <typename T, typename U1, typename U2>
__aicore__ inline void MoeGatingTopKHashRegbase<T, U1, U2>::Process()
{
    CopyInBias();
    CopyInX(0);
    for (int64_t row = 1; row < curCoreRowCount_; row++) {
        ComputeX();
        CopyOutXNorm(row - 1);
        CopyInX(row);
        HashCompute(row - 1 + tilingData_->perCoreRowCount * blockIdx_);
        CopyOut(row - 1);
    }
    ComputeX();
    CopyOutXNorm(curCoreRowCount_ - 1);
    HashCompute(curCoreRowCount_ - 1 + tilingData_->perCoreRowCount * blockIdx_);
    CopyOut(curCoreRowCount_ - 1);
}
} // namespace MoeGatingTopK
#endif // MOE_GATING_TOP_K_HASH_REGBASE_H
