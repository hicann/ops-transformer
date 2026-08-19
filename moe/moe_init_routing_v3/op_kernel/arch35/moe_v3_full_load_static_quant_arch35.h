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
 * \file moe_v3_full_load_static_quant_arch35.h
 * \brief
 */
#ifndef MOE_V3_FULL_LOAD_STATIC_QUANT_ARCH35_H
#define MOE_V3_FULL_LOAD_STATIC_QUANT_ARCH35_H

#include "moe_v3_full_load_base.h"
#include "moe_v3_common.h"
#include "kernel_operator.h"
#include "op_kernel/load_store_utils.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

constexpr int64_t FULLLOAD_STATIC_QUANT_BUFFER_NUM = 1;

template <typename T>
class MoeV3FullLoadStaticQuant : public MoeV3FullLoadBase<T> {
public:
    __aicore__ inline MoeV3FullLoadStaticQuant(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR expandedX, GM_ADDR expandedRowIdx,
                                GM_ADDR expertTokensCountOrCumsum, GM_ADDR expandedScale, GM_ADDR topkWeight,
                                GM_ADDR expandedTopkWeight, GM_ADDR workspace, GM_ADDR offset,
                                const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void Compute(int64_t rowLength);
    __aicore__ inline void ScatterOutXStaticQuant();
    __aicore__ inline void GatherOutXStaticQuant();

private:
    TQue<QuePosition::VECIN, 1> inputXInQueue_;
    TQue<QuePosition::VECOUT, 1> inputXOutQueue_;

    GlobalTensor<T> xGm_;
    GlobalTensor<int8_t> expandedXGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<float> offsetGm_;

    int64_t colsAlign_;
    int64_t inFactor_;

    float scale_;
    float offset_;

    constexpr static MicroAPI::CastTrait castTraitF32ToF16 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                              MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
    constexpr static MicroAPI::CastTrait castTraitF16ToI8 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                             MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
};

template <typename T>
__aicore__ inline void MoeV3FullLoadStaticQuant<T>::Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR expandedX,
                                                         GM_ADDR expandedRowIdx, GM_ADDR expertTokensCountOrCumsum,
                                                         GM_ADDR expandedScale, GM_ADDR topkWeight,
                                                         GM_ADDR expandedTopkWeight, GM_ADDR workspace, GM_ADDR offset,
                                                         const MoeInitRoutingV3Arch35TilingData *tilingData,
                                                         TPipe *tPipe)
{
    MoeV3FullLoadBase<T>::Init(expertIdx, expandedRowIdx, expertTokensCountOrCumsum, topkWeight, expandedTopkWeight,
                               workspace, tilingData, tPipe);

    colsAlign_ = Align(this->cols_, sizeof(T));
    inFactor_ = Align(this->cols_, sizeof(int8_t));

    xGm_.SetGlobalBuffer((__gm__ T *)x);
    expandedXGm_.SetGlobalBuffer((__gm__ int8_t *)expandedX);

    scaleGm_.SetGlobalBuffer((__gm__ float *)scale, 1);
    offsetGm_.SetGlobalBuffer((__gm__ float *)offset, 1);
    scale_ = scaleGm_.GetValue(0);
    offset_ = offsetGm_.GetValue(0);

    int64_t rowLength = this->endXRow_ - this->startXRow_ + 1;

    int64_t colsAlignBytes = AlignBytes(this->cols_ * rowLength, sizeof(T));
    colsAlignBytes = static_cast<int64_t>(colsAlignBytes * sizeof(float) / sizeof(T));
    this->pipe_->InitBuffer(inputXInQueue_, FULLLOAD_STATIC_QUANT_BUFFER_NUM, colsAlignBytes);
    this->pipe_->InitBuffer(inputXOutQueue_, FULLLOAD_STATIC_QUANT_BUFFER_NUM,
                            AlignBytes(this->cols_ * rowLength, sizeof(int8_t)));
}

template <typename T>
__aicore__ inline void MoeV3FullLoadStaticQuant<T>::Compute(int64_t rowLength)
{
    LocalTensor<float> inLocal = inputXInQueue_.DeQue<float>();
    LocalTensor<int8_t> outLocal = inputXOutQueue_.AllocTensor<int8_t>();

    __ubuf__ float *inUbAddr = (__ubuf__ float *)inLocal.GetPhyAddr();
    __ubuf__ int8_t *outUbAddr = (__ubuf__ int8_t *)outLocal.GetPhyAddr();
    __ubuf__ T *inUbAddrCastT;
    if constexpr (!IsSameType<T, float>::value) {
        inUbAddrCastT = (__ubuf__ T *)inLocal.ReinterpretCast<T>().GetPhyAddr() + colsAlign_;
    }

    uint16_t repeatTimes = Ceil(this->cols_ * rowLength, FLOAT_REG_TENSOR_LENGTH);
    uint32_t sreg = static_cast<uint32_t>(this->cols_ * rowLength);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> inReg;
        MicroAPI::RegTensor<half> outRegF16;
        MicroAPI::RegTensor<int8_t> outRegI8;

        MicroAPI::MaskReg maskRegInLoop;

        for (uint16_t i = 0; i < repeatTimes; i++) {
            maskRegInLoop = MicroAPI::UpdateMask<float>(sreg);
            if constexpr (!IsSameType<T, float>::value) {
                ops::LoadOneTensorForDtypeT<T>(inUbAddrCastT, inReg, maskRegInLoop, i * FLOAT_REG_TENSOR_LENGTH);
            } else {
                MicroAPI::LoadAlign(inReg, inUbAddr + i * FLOAT_REG_TENSOR_LENGTH);
            }
            MicroAPI::Cast<half, float, castTraitF32ToF16>(outRegF16, inReg, maskRegInLoop);
            MicroAPI::Muls(outRegF16, outRegF16, static_cast<half>(scale_), maskRegInLoop);
            MicroAPI::Adds(outRegF16, outRegF16, static_cast<half>(offset_), maskRegInLoop);
            MicroAPI::Cast<int8_t, half, castTraitF16ToI8>(outRegI8, outRegF16, maskRegInLoop);
            MicroAPI::StoreAlign<int8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(outUbAddr + i * FLOAT_REG_TENSOR_LENGTH,
                                                                              outRegI8, maskRegInLoop);
        }
    }

    inputXOutQueue_.EnQue(outLocal);
}

template <typename T>
__aicore__ inline void MoeV3FullLoadStaticQuant<T>::ScatterOutXStaticQuant()
{
    LocalTensor<int32_t> sortedRowIdx = this->sortedRowIdxQueue_.template DeQue<int32_t>();
    LocalTensor<int32_t> sortedExpertIdx = this->sortedExpertIdxQueue_.template DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

    int64_t startRowIdx = this->blockIdx_ * this->perCoreIndicesElements_;
    int64_t endRowIdx = startRowIdx + this->coreIndicesElements_;
    int64_t outputRows = Min(this->actualExpertIdxNum_, this->activeNum_);

    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(this->cols_ * sizeof(T)), 0, 0, 0};
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(this->cols_ * sizeof(int8_t)), 0, 0, 0};

    for (int64_t i = startRowIdx; i < endRowIdx && i < outputRows; i++) {
        int32_t curExpertId = sortedExpertIdx.GetValue(i);
        if (curExpertId < this->expertStart_ || curExpertId >= this->expertEnd_) {
            break;
        }

        int32_t srcIdx = sortedRowIdx.GetValue(i);
        int64_t xSrcOffset = srcIdx / this->k_ * this->cols_;
        int64_t xDstOffset = i * this->cols_;

        LocalTensor<T> inLocal = inputXInQueue_.AllocTensor<T>();
        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(inLocal, xGm_[xSrcOffset], copyInParams, {false, 0, 0, 0});
        } else {
            DataCopyPad(inLocal[colsAlign_], xGm_[xSrcOffset], copyInParams, {false, 0, 0, 0});
        }
        inputXInQueue_.EnQue<T>(inLocal);

        Compute(1);

        LocalTensor<int8_t> outLocal = inputXOutQueue_.DeQue<int8_t>();
        DataCopyPad(expandedXGm_[xDstOffset], outLocal, copyOutParams);
        inputXOutQueue_.FreeTensor(outLocal);
    }

    this->sortedRowIdxQueue_.template EnQue<int32_t>(sortedRowIdx);
    this->sortedExpertIdxQueue_.template EnQue<int32_t>(sortedExpertIdx);
}

template <typename T>
__aicore__ inline void MoeV3FullLoadStaticQuant<T>::GatherOutXStaticQuant()
{
    LocalTensor<int32_t> expandedRowIdx = this->expandedRowIdxQueue_.template DeQue<int32_t>();

    int64_t startRowIdx = this->blockIdx_ * this->perCoreIndicesElements_;
    int64_t rowLength = this->endXRow_ - this->startXRow_ + 1;
    int64_t outputRows = Min(this->actualExpertIdxNum_, this->activeNum_);

    uint32_t dstStride = (inFactor_ * sizeof(T) - AlignBytes(this->cols_, sizeof(T))) / BLOCK_BYTES;
    DataCopyExtParams dataXCopyParams{static_cast<uint16_t>(rowLength), static_cast<uint32_t>(this->cols_ * sizeof(T)),
                                      0, dstStride, 0};
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(this->cols_ * sizeof(int8_t)), 0, 0, 0};

    LocalTensor<T> xLocal = inputXInQueue_.AllocTensor<T>();
    if constexpr (IsSameType<T, float>::value) {
        DataCopyPad(xLocal, xGm_[this->startXRow_ * this->cols_], dataXCopyParams, {false, 0, 0, 0});
    } else {
        DataCopyPad(xLocal[colsAlign_], xGm_[this->startXRow_ * this->cols_], dataXCopyParams, {false, 0, 0, 0});
    }

    inputXInQueue_.EnQue<T>(xLocal);
    SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);

    Compute(rowLength);

    LocalTensor<int8_t> outLocal = inputXOutQueue_.DeQue<int8_t>();
    int64_t curIndex = startRowIdx;
    int64_t k = 0;

    for (int64_t i = this->startXRow_; i <= this->endXRow_; i++) {
        for (; k < this->coreIndicesElements_ && curIndex / this->k_ == i; curIndex++, k++) {
            int32_t outIndex = expandedRowIdx.GetValue(curIndex);
            if (outIndex >= 0 && outIndex < outputRows) {
                DataCopyPad(expandedXGm_[outIndex * this->cols_], outLocal[(i - this->startXRow_) * inFactor_],
                            copyOutParams);
            }
        }
    }

    inputXOutQueue_.FreeTensor(outLocal);
    this->expandedRowIdxQueue_.template EnQue<int32_t>(expandedRowIdx);
}

template <typename T>
__aicore__ inline void MoeV3FullLoadStaticQuant<T>::Process()
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

        if (this->epFullload_) {
            if (this->isInputTopkWeight_) {
                this->TopkWeightScatterOut();
            }
            ScatterOutXStaticQuant();
        } else {
            if (this->isInputTopkWeight_) {
                this->TopkWeightGatherOut();
            }
            GatherOutXStaticQuant();
        }

        this->FreeLocalTensor();
    }
}

} // namespace MoeInitRoutingV3
#endif // MOE_V3_FULL_LOAD_STATIC_QUANT_ARCH35_H
