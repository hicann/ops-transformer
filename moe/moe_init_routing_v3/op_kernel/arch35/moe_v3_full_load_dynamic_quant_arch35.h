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
 * \file moe_v3_full_load_dynamic_quant_arch35.h
 * \brief
 */
#ifndef MOE_V3_FULL_LOAD_DYNAMIC_QUANT_ARCH35_H
#define MOE_V3_FULL_LOAD_DYNAMIC_QUANT_ARCH35_H

#include "moe_v3_full_load_base.h"
#include "moe_v3_common.h"
#include "kernel_operator.h"
#include "op_kernel/load_store_utils.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

constexpr int64_t FULLLOAD_DYNAMIC_QUANT_BUFFER_NUM = 1;
constexpr int64_t FULLLOAD_DYNAMIC_QUANT_INPUT_X_QUEUE_NUM = 3;

template <typename T, typename QuantT = int8_t>
class MoeV3FullLoadDynamicQuant : public MoeV3FullLoadBase<T> {
public:
    __aicore__ inline MoeV3FullLoadDynamicQuant(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR expandedX, GM_ADDR expandedRowIdx,
                                GM_ADDR expertTokensCountOrCumsum, GM_ADDR expandedScale, GM_ADDR topkWeight,
                                GM_ADDR expandedTopkWeight, GM_ADDR workspace,
                                const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    template <bool IS_INPUT_SCALE>
    __aicore__ inline void Compute(LocalTensor<float> &smoothLocal);
    template <bool IS_INPUT_SCALE>
    __aicore__ inline void ComputeInt4SinglePass(LocalTensor<float> &smoothLocal);
    template <bool IS_INPUT_SCALE>
    __aicore__ inline void ComputeMultiPass(LocalTensor<float> &smoothLocal);
    __aicore__ inline void ScatterOutXDynamicQuant();
    __aicore__ inline void GatherOutXDynamicQuant();
    __aicore__ inline void StoreInt4QuantOut(__ubuf__ uint8_t *outUbAddr, MicroAPI::RegTensor<float> &inReg,
                                             MicroAPI::MaskReg &maskRegInLoop, MicroAPI::MaskReg &maskRegHalf);

private:
    TQue<QuePosition::VECIN, FULLLOAD_DYNAMIC_QUANT_INPUT_X_QUEUE_NUM> inputXInQueue_;
    TQue<QuePosition::VECIN, FULLLOAD_DYNAMIC_QUANT_BUFFER_NUM> smoothInQueue_;
    TQue<QuePosition::VECOUT, FULLLOAD_DYNAMIC_QUANT_BUFFER_NUM> inputXOutQueue_;
    TQue<QuePosition::VECOUT, FULLLOAD_DYNAMIC_QUANT_BUFFER_NUM> scaleOutQueue_;

    GlobalTensor<T> xGm_;
    GlobalTensor<int8_t> expandedXGm_;
    GlobalTensor<float> quantSmoothGm_;
    GlobalTensor<float> expandedScaleGm_;

    int64_t colsAlign_;
    int64_t colsAsInt8_;

    constexpr static MicroAPI::CastTrait castTraitF32ToF16 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                              MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};
    constexpr static MicroAPI::CastTrait castTraitF16ToI8 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                             MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
    constexpr static MicroAPI::CastTrait castTraitF16ToI4 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                             MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
};

template <typename T, typename QuantT>
__aicore__ inline void MoeV3FullLoadDynamicQuant<T, QuantT>::Init(
    GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR expandedX, GM_ADDR expandedRowIdx,
    GM_ADDR expertTokensCountOrCumsum, GM_ADDR expandedScale, GM_ADDR topkWeight, GM_ADDR expandedTopkWeight,
    GM_ADDR workspace, const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe)
{
    MoeV3FullLoadBase<T>::Init(expertIdx, expandedRowIdx, expertTokensCountOrCumsum, topkWeight, expandedTopkWeight,
                               workspace, tilingData, tPipe);

    colsAlign_ = Align(this->cols_, sizeof(T));
    if constexpr (IsSameType<QuantT, int4b_t>::value) {
        colsAsInt8_ = Ceil(this->cols_, static_cast<int64_t>(2)) * sizeof(int8_t);
    } else {
        colsAsInt8_ = this->cols_ * sizeof(int8_t);
    }

    xGm_.SetGlobalBuffer((__gm__ T *)x);
    expandedXGm_.SetGlobalBuffer((__gm__ int8_t *)expandedX);

    if (this->isInputScale_) {
        quantSmoothGm_.SetGlobalBuffer((__gm__ float *)scale);
    }
    expandedScaleGm_.SetGlobalBuffer((__gm__ float *)expandedScale);

    if constexpr (IsSameType<T, float>::value) {
        this->pipe_->InitBuffer(inputXInQueue_, FULLLOAD_DYNAMIC_QUANT_BUFFER_NUM,
                                AlignBytes(this->cols_, sizeof(float)));
    } else {
        this->pipe_->InitBuffer(inputXInQueue_, FULLLOAD_DYNAMIC_QUANT_BUFFER_NUM,
                                2 * AlignBytes(this->cols_, sizeof(T)));
    }
    this->pipe_->InitBuffer(smoothInQueue_, FULLLOAD_DYNAMIC_QUANT_BUFFER_NUM, AlignBytes(this->cols_, sizeof(float)));
    this->pipe_->InitBuffer(inputXOutQueue_, FULLLOAD_DYNAMIC_QUANT_BUFFER_NUM,
                            AlignBytes(colsAsInt8_, sizeof(int8_t)));
    this->pipe_->InitBuffer(scaleOutQueue_, FULLLOAD_DYNAMIC_QUANT_BUFFER_NUM, BLOCK_BYTES + BLOCK_BYTES);
}

template <typename T, typename QuantT>
__aicore__ inline void MoeV3FullLoadDynamicQuant<T, QuantT>::StoreInt4QuantOut(__ubuf__ uint8_t *outUbAddr,
                                                                               MicroAPI::RegTensor<float> &inReg,
                                                                               MicroAPI::MaskReg &maskRegInLoop,
                                                                               MicroAPI::MaskReg &maskRegHalf)
{
    MicroAPI::RegTensor<half> outRegF16;
    MicroAPI::RegTensor<uint8_t> outRegI4;
    MicroAPI::RegTensor<uint16_t> outRegPackedHalf;

    MicroAPI::Cast<half, float, castTraitF32ToF16>(outRegF16, inReg, maskRegInLoop);
    MicroAPI::Pack(outRegPackedHalf, (MicroAPI::RegTensor<uint32_t> &)outRegF16);
    MicroAPI::Cast<int4x2_t, half, castTraitF16ToI4>((MicroAPI::RegTensor<int4x2_t> &)outRegI4,
                                                     (MicroAPI::RegTensor<half> &)outRegPackedHalf, maskRegInLoop);
    MicroAPI::StoreAlign<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(outUbAddr, outRegI4, maskRegHalf);
}

template <typename T, typename QuantT>
template <bool IS_INPUT_SCALE>
__aicore__ inline void MoeV3FullLoadDynamicQuant<T, QuantT>::Compute(LocalTensor<float> &smoothLocal)
{
    if constexpr (IsSameType<QuantT, int4b_t>::value) {
        if (this->cols_ <= FLOAT_REG_TENSOR_LENGTH) {
            ComputeInt4SinglePass<IS_INPUT_SCALE>(smoothLocal);
            return;
        }
    }
    ComputeMultiPass<IS_INPUT_SCALE>(smoothLocal);
}

template <typename T, typename QuantT>
template <bool IS_INPUT_SCALE>
__aicore__ inline void MoeV3FullLoadDynamicQuant<T, QuantT>::ComputeInt4SinglePass(LocalTensor<float> &smoothLocal)
{
    LocalTensor<float> inLocal = inputXInQueue_.DeQue<float>();
    LocalTensor<int8_t> outLocal = inputXOutQueue_.AllocTensor<int8_t>();
    LocalTensor<float> scaleLocal = scaleOutQueue_.AllocTensor<float>();

    __ubuf__ float *inUbAddr = (__ubuf__ float *)inLocal.GetPhyAddr();
    __ubuf__ float *scaleUbAddr = (__ubuf__ float *)scaleLocal.GetPhyAddr();
    __ubuf__ int8_t *outUbAddr = (__ubuf__ int8_t *)outLocal.GetPhyAddr();
    __ubuf__ T *inUbAddrCastT;
    if constexpr (!IsSameType<T, float>::value) {
        inUbAddrCastT = (__ubuf__ T *)inLocal.ReinterpretCast<T>().GetPhyAddr() + colsAlign_;
    }
    __ubuf__ float *smoothUbAddr;
    if constexpr (IS_INPUT_SCALE) {
        smoothUbAddr = (__ubuf__ float *)smoothLocal.GetPhyAddr();
    }

    uint32_t sreg = static_cast<uint32_t>(this->cols_);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> inReg, absReg, smoothReg, scaleValueReg, quantFactorReg, zeroReg;
        MicroAPI::Duplicate(scaleValueReg, 0.0f);
        MicroAPI::Duplicate(absReg, 0.0f);

        MicroAPI::MaskReg maskRegInLoop = MicroAPI::UpdateMask<float>(sreg);
        MicroAPI::MaskReg maskRegAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskRegHalf = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::H>();
        MicroAPI::MaskReg maskRegVL1 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
        MicroAPI::MaskReg maskRegScaleZero;

        if constexpr (!IsSameType<T, float>::value) {
            ops::LoadOneTensorForDtypeT<T>(inUbAddrCastT, inReg, maskRegInLoop, 0);
        } else {
            MicroAPI::LoadAlign(inReg, inUbAddr);
        }
        if constexpr (IS_INPUT_SCALE) {
            MicroAPI::LoadAlign(smoothReg, smoothUbAddr);
            MicroAPI::Mul(inReg, inReg, smoothReg, maskRegInLoop);
        }
        MicroAPI::Abs(absReg, inReg, maskRegInLoop);
        MicroAPI::Max(scaleValueReg, scaleValueReg, absReg, maskRegAll);
        Reg::Reduce<Reg::ReduceType::MAX>(scaleValueReg, scaleValueReg, maskRegAll);
        MicroAPI::Duplicate(quantFactorReg, DYNAMIC_QUANT_INT4_SYM_SCALE, maskRegVL1);
        MicroAPI::Div(quantFactorReg, quantFactorReg, scaleValueReg, maskRegVL1);
        MicroAPI::Duplicate(zeroReg, 0.0f, maskRegVL1);
        MicroAPI::Compares<float, CMPMODE::EQ>(maskRegScaleZero, scaleValueReg, 0.0f, maskRegVL1);
        MicroAPI::Select(quantFactorReg, zeroReg, quantFactorReg, maskRegScaleZero);
        MicroAPI::Duplicate(quantFactorReg, quantFactorReg, maskRegAll);

        MicroAPI::Muls(scaleValueReg, scaleValueReg, 1.0f / DYNAMIC_QUANT_INT4_SYM_SCALE, maskRegVL1);
        MicroAPI::Duplicate(scaleValueReg, scaleValueReg, maskRegAll);
        MicroAPI::StoreAlign(scaleUbAddr, scaleValueReg, maskRegVL1);

        MicroAPI::Mul(inReg, inReg, quantFactorReg, maskRegInLoop);
        StoreInt4QuantOut((__ubuf__ uint8_t *)outUbAddr, inReg, maskRegInLoop, maskRegHalf);
    }

    inputXOutQueue_.EnQue(outLocal);
    scaleOutQueue_.EnQue(scaleLocal);
}

template <typename T, typename QuantT>
template <bool IS_INPUT_SCALE>
__aicore__ inline void MoeV3FullLoadDynamicQuant<T, QuantT>::ComputeMultiPass(LocalTensor<float> &smoothLocal)
{
    LocalTensor<float> inLocal = inputXInQueue_.DeQue<float>();
    LocalTensor<int8_t> outLocal = inputXOutQueue_.AllocTensor<int8_t>();
    LocalTensor<float> scaleLocal = scaleOutQueue_.AllocTensor<float>();

    __ubuf__ float *inUbAddr = (__ubuf__ float *)inLocal.GetPhyAddr();
    __ubuf__ float *scaleUbAddr = (__ubuf__ float *)scaleLocal.GetPhyAddr();
    __ubuf__ int8_t *outUbAddr = (__ubuf__ int8_t *)outLocal.GetPhyAddr();
    __ubuf__ T *inUbAddrCastT;
    if constexpr (!IsSameType<T, float>::value) {
        inUbAddrCastT = (__ubuf__ T *)inLocal.ReinterpretCast<T>().GetPhyAddr() + colsAlign_;
    }
    __ubuf__ float *smoothUbAddr;
    if constexpr (IS_INPUT_SCALE) {
        smoothUbAddr = (__ubuf__ float *)smoothLocal.GetPhyAddr();
    }

    uint16_t repeatTimes = Ceil(this->cols_, FLOAT_REG_TENSOR_LENGTH);
    uint32_t maskLength = static_cast<uint32_t>(this->cols_);
    uint32_t sreg;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> inReg, smoothReg, scaleValueReg, quantFactorReg, zeroReg;
        MicroAPI::Duplicate(scaleValueReg, 0.0f);
        MicroAPI::RegTensor<half> outRegF16;
        MicroAPI::RegTensor<int8_t> outRegI8;

        MicroAPI::MaskReg maskRegInLoop;
        MicroAPI::MaskReg maskRegAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskRegHalf = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::H>();
        MicroAPI::MaskReg maskRegVL1 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
        MicroAPI::MaskReg maskRegVL8 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL8>();
        MicroAPI::MaskReg maskRegScaleZero;

        sreg = maskLength;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            maskRegInLoop = MicroAPI::UpdateMask<float>(sreg);
            if constexpr (!IsSameType<T, float>::value) {
                ops::LoadOneTensorForDtypeT<T>(inUbAddrCastT, inReg, maskRegInLoop, i * FLOAT_REG_TENSOR_LENGTH);
            } else {
                MicroAPI::LoadAlign(inReg, inUbAddr + i * FLOAT_REG_TENSOR_LENGTH);
            }
            if constexpr (IS_INPUT_SCALE) {
                MicroAPI::LoadAlign(smoothReg, smoothUbAddr + i * FLOAT_REG_TENSOR_LENGTH);
                MicroAPI::Mul(inReg, inReg, smoothReg, maskRegInLoop);
            }
            if constexpr (!IsSameType<T, float>::value || IS_INPUT_SCALE) {
                MicroAPI::StoreAlign(inUbAddr + i * FLOAT_REG_TENSOR_LENGTH, inReg, maskRegInLoop);
            }
            MicroAPI::Abs(inReg, inReg, maskRegInLoop);
            MicroAPI::Max(scaleValueReg, scaleValueReg, inReg, maskRegAll);
        }
        Reg::Reduce<Reg::ReduceType::MAX>(scaleValueReg, scaleValueReg, maskRegAll);
        if constexpr (!IsSameType<QuantT, int4b_t>::value) {
            MicroAPI::Muls(scaleValueReg, scaleValueReg, 1.0f / 127.0f, maskRegVL1);
        }
        if constexpr (IsSameType<QuantT, int4b_t>::value) {
            MicroAPI::Duplicate(quantFactorReg, DYNAMIC_QUANT_INT4_SYM_SCALE, maskRegVL1);
            MicroAPI::Div(quantFactorReg, quantFactorReg, scaleValueReg, maskRegVL1);
            MicroAPI::Duplicate(zeroReg, 0.0f, maskRegVL1);
            MicroAPI::Compares<float, CMPMODE::EQ>(maskRegScaleZero, scaleValueReg, 0.0f, maskRegVL1);
            MicroAPI::Select(quantFactorReg, zeroReg, quantFactorReg, maskRegScaleZero);
            MicroAPI::Duplicate(quantFactorReg, quantFactorReg, maskRegAll);

            MicroAPI::Muls(scaleValueReg, scaleValueReg, 1.0f / DYNAMIC_QUANT_INT4_SYM_SCALE, maskRegVL1);
        }
        MicroAPI::Duplicate(scaleValueReg, scaleValueReg, maskRegAll);
        MicroAPI::StoreAlign(scaleUbAddr, scaleValueReg, maskRegVL8);

        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

        sreg = maskLength;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            maskRegInLoop = MicroAPI::UpdateMask<float>(sreg);
            MicroAPI::LoadAlign(inReg, inUbAddr + i * FLOAT_REG_TENSOR_LENGTH);
            if constexpr (IsSameType<QuantT, int4b_t>::value) {
                MicroAPI::Mul(inReg, inReg, quantFactorReg, maskRegInLoop);
                StoreInt4QuantOut((__ubuf__ uint8_t *)outUbAddr + i * FLOAT_REG_TENSOR_LENGTH / 2, inReg, maskRegInLoop,
                                  maskRegHalf);
            } else {
                MicroAPI::Div(inReg, inReg, scaleValueReg, maskRegInLoop);
                MicroAPI::Cast<half, float, castTraitF32ToF16>(outRegF16, inReg, maskRegInLoop);
                MicroAPI::Cast<int8_t, half, castTraitF16ToI8>(outRegI8, outRegF16, maskRegInLoop);
                MicroAPI::StoreAlign<int8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(
                    outUbAddr + i * FLOAT_REG_TENSOR_LENGTH, outRegI8, maskRegInLoop);
            }
        }
    }

    inputXOutQueue_.EnQue(outLocal);
    scaleOutQueue_.EnQue(scaleLocal);
}

template <typename T, typename QuantT>
__aicore__ inline void MoeV3FullLoadDynamicQuant<T, QuantT>::ScatterOutXDynamicQuant()
{
    LocalTensor<int32_t> sortedRowIdx = this->sortedRowIdxQueue_.template DeQue<int32_t>();
    LocalTensor<int32_t> sortedExpertIdx = this->sortedExpertIdxQueue_.template DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

    int64_t startRowIdx = this->blockIdx_ * this->perCoreIndicesElements_;
    int64_t endRowIdx = Min(startRowIdx + this->coreIndicesElements_, this->actualExpertIdxNum_);

    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(this->cols_ * sizeof(T)), 0, 0, 0};
    DataCopyExtParams smoothParams{1, static_cast<uint32_t>(this->cols_ * sizeof(float)), 0, 0, 0};
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(colsAsInt8_), 0, 0, 0};

    LocalTensor<float> smoothLocal = smoothInQueue_.AllocTensor<float>();
    int32_t lastExpertIdx = -1;

    for (int64_t i = startRowIdx; i < endRowIdx && i < this->activeNum_; i++) {
        int32_t curExpertId = sortedExpertIdx.GetValue(i);
        int32_t srcIdx = sortedRowIdx.GetValue(i);
        int32_t expertIdx = curExpertId - this->expertStart_;
        LocalTensor<T> inLocal = inputXInQueue_.AllocTensor<T>();
        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(inLocal, xGm_[srcIdx / this->k_ * this->cols_], copyInParams, {false, 0, 0, 0});
        } else {
            DataCopyPad(inLocal[colsAlign_], xGm_[srcIdx / this->k_ * this->cols_], copyInParams, {false, 0, 0, 0});
        }
        inputXInQueue_.EnQue<T>(inLocal);

        if (this->isInputScale_) {
            if constexpr (IsSameType<QuantT, int4b_t>::value) {
                if (lastExpertIdx == -1) {
                    DataCopyPad(smoothLocal, quantSmoothGm_[0], smoothParams, {false, 0, 0, 0});
                    smoothInQueue_.EnQue(smoothLocal);
                    smoothLocal = smoothInQueue_.DeQue<float>();
                    lastExpertIdx = 0;
                }
            } else {
                if (expertIdx != lastExpertIdx) {
                    int64_t smoothOffset = expertIdx * this->cols_;
                    DataCopyPad(smoothLocal, quantSmoothGm_[smoothOffset], smoothParams, {false, 0, 0, 0});
                    smoothInQueue_.EnQue(smoothLocal);
                    smoothLocal = smoothInQueue_.DeQue<float>();
                    lastExpertIdx = expertIdx;
                }
            }
        }

        if (this->isInputScale_) {
            Compute<true>(smoothLocal);
        } else {
            Compute<false>(smoothLocal);
        }
        inputXInQueue_.FreeTensor(inLocal);

        LocalTensor<float> scaleLocal = scaleOutQueue_.DeQue<float>();
        LocalTensor<int8_t> outLocal = inputXOutQueue_.DeQue<int8_t>();
        DataCopyPad(expandedScaleGm_[i], scaleLocal, {1, 4, 0, 0, 0});
        DataCopyPad(expandedXGm_[i * colsAsInt8_], outLocal, copyOutParams);

        inputXOutQueue_.FreeTensor(outLocal);
        scaleOutQueue_.FreeTensor(scaleLocal);
    }

    smoothInQueue_.FreeTensor(smoothLocal);
    this->sortedRowIdxQueue_.template EnQue<int32_t>(sortedRowIdx);
    this->sortedExpertIdxQueue_.template EnQue<int32_t>(sortedExpertIdx);
}

template <typename T, typename QuantT>
__aicore__ inline void MoeV3FullLoadDynamicQuant<T, QuantT>::GatherOutXDynamicQuant()
{
    LocalTensor<int32_t> expandedRowIdx = this->expandedRowIdxQueue_.template DeQue<int32_t>();

    int64_t startRowIdx = this->blockIdx_ * this->perCoreIndicesElements_;

    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(this->cols_ * sizeof(T)), 0, 0, 0};
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(colsAsInt8_), 0, 0, 0};

    int64_t curIndex = startRowIdx;
    int64_t k = 0;
    int64_t outputRows = Min(this->actualExpertIdxNum_, this->activeNum_);

    LocalTensor<float> smoothLocal = smoothInQueue_.AllocTensor<float>();

    for (int64_t i = this->startXRow_; i <= this->endXRow_; i++) {
        LocalTensor<T> inLocal = inputXInQueue_.AllocTensor<T>();
        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(inLocal, xGm_[i * this->cols_], copyInParams, {false, 0, 0, 0});
        } else {
            DataCopyPad(inLocal[colsAlign_], xGm_[i * this->cols_], copyInParams, {false, 0, 0, 0});
        }
        inputXInQueue_.EnQue<T>(inLocal);

        Compute<false>(smoothLocal);

        LocalTensor<int8_t> outLocal = inputXOutQueue_.DeQue<int8_t>();
        LocalTensor<float> scaleLocal = scaleOutQueue_.DeQue<float>();

        for (; k < this->coreIndicesElements_ && curIndex / this->k_ == i; curIndex++, k++) {
            int32_t outIndex = expandedRowIdx.GetValue(curIndex);
            if (outIndex >= 0 && outIndex < outputRows) {
                DataCopyPad(expandedXGm_[outIndex * colsAsInt8_], outLocal, copyOutParams);
                DataCopyPad(expandedScaleGm_[outIndex], scaleLocal, {1, 4, 0, 0, 0});
            }
        }

        inputXOutQueue_.FreeTensor(outLocal);
        scaleOutQueue_.FreeTensor(scaleLocal);
        inputXInQueue_.FreeTensor(inLocal);
    }

    smoothInQueue_.FreeTensor(smoothLocal);
    this->expandedRowIdxQueue_.template EnQue<int32_t>(expandedRowIdx);
}

template <typename T, typename QuantT>
__aicore__ inline void MoeV3FullLoadDynamicQuant<T, QuantT>::Process()
{
    if (this->blockIdx_ < this->needCoreNum_) {
        this->CopyIn();
        this->SortCompute();

        if (this->blockIdx_ == 0) {
            this->CopyOutRowIdx();
        }

        if (this->blockIdx_ == this->needCoreNum_ - 1 && this->expertTokensNumFlag_ == 1) {
            this->ComputeExpertTokenCount();
            this->CopyExpertCountToOutput();
        }

        if (this->epFullload_ || this->isInputScale_) {
            if (this->isInputTopkWeight_) {
                this->TopkWeightScatterOut();
            }
            ScatterOutXDynamicQuant();
        } else {
            if (this->isInputTopkWeight_) {
                this->TopkWeightGatherOut();
            }
            GatherOutXDynamicQuant();
        }

        this->FreeLocalTensor();
    }
}

} // namespace MoeInitRoutingV3
#endif // MOE_V3_FULL_LOAD_DYNAMIC_QUANT_ARCH35_H
