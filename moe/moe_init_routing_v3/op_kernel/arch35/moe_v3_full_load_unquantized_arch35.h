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
 * \file moe_v3_full_load_unquantized_arch35.h
 * \brief
 */
#ifndef MOE_V3_FULL_LOAD_UNQUANTIZED_ARCH35_H
#define MOE_V3_FULL_LOAD_UNQUANTIZED_ARCH35_H

#include "moe_v3_full_load_base.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

template <typename T>
class MoeV3FullLoadUnquantized : public MoeV3FullLoadBase<T> {
public:
    __aicore__ inline MoeV3FullLoadUnquantized(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR expandedX, GM_ADDR expandedRowIdx,
                                GM_ADDR expertTokensCountOrCumsum, GM_ADDR expandedScale, GM_ADDR topkWeight,
                                GM_ADDR expandedTopkWeight, GM_ADDR workspace,
                                const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void ScatterOutX();
    __aicore__ inline void GatherOutX();
    __aicore__ inline void ProcessDropPadMode();
    __aicore__ inline void GatherOutXScatterLoop(const LocalTensor<int32_t> &expandedRowIdx,
                                                 const LocalTensor<T> &xLocal, int64_t startRowIdx, int64_t outputRows,
                                                 int64_t inFactor);
    __aicore__ inline void ScatterOutScale();
    __aicore__ inline void GatherOutScale();
    __aicore__ inline void ZeroOutX();
    __aicore__ inline void ZeroOutScale();

private:
    TQue<QuePosition::VECIN, 1> xCopyInQueue_;
    TQue<QuePosition::VECIN, 1> scaleCopyInQueue_;

    GlobalTensor<T> xGm_;
    GlobalTensor<uint8_t> xUint8tGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<T> expandedXGm_;
    GlobalTensor<float> expandedScaleGm_;
};

template <typename T>
__aicore__ inline void MoeV3FullLoadUnquantized<T>::Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR expandedX,
                                                         GM_ADDR expandedRowIdx, GM_ADDR expertTokensCountOrCumsum,
                                                         GM_ADDR expandedScale, GM_ADDR topkWeight,
                                                         GM_ADDR expandedTopkWeight, GM_ADDR workspace,
                                                         const MoeInitRoutingV3Arch35TilingData *tilingData,
                                                         TPipe *tPipe)
{
    MoeV3FullLoadBase<T>::Init(expertIdx, expandedRowIdx, expertTokensCountOrCumsum, topkWeight, expandedTopkWeight,
                               workspace, tilingData, tPipe);

    if constexpr (IsSameType<T, hifloat8_t>::value) {
        xUint8tGm_.SetGlobalBuffer((__gm__ uint8_t *)x);
    } else {
        xGm_.SetGlobalBuffer((__gm__ T *)x);
    }
    if (this->isInputScale_) {
        scaleGm_.SetGlobalBuffer((__gm__ float *)scale);
        expandedScaleGm_.SetGlobalBuffer((__gm__ float *)expandedScale);
    }

    expandedXGm_.SetGlobalBuffer((__gm__ T *)expandedX);

    int64_t rowLength = this->endXRow_ - this->startXRow_ + 1;

    if (this->epFullload_) {
        if constexpr (IsSameType<T, hifloat8_t>::value) {
            this->pipe_->InitBuffer(xCopyInQueue_, this->bufferNum_, AlignBytes(this->cols_, sizeof(uint8_t)));
        } else {
            this->pipe_->InitBuffer(xCopyInQueue_, this->bufferNum_, AlignBytes(this->cols_, sizeof(T)));
        }
    } else {
        if constexpr (IsSameType<T, hifloat8_t>::value) {
            this->pipe_->InitBuffer(xCopyInQueue_, this->bufferNum_,
                                    AlignBytes(this->cols_ * rowLength, sizeof(uint8_t)));
        } else {
            this->pipe_->InitBuffer(xCopyInQueue_, this->bufferNum_, AlignBytes(this->cols_ * rowLength, sizeof(T)));
        }
    }
    this->pipe_->InitBuffer(scaleCopyInQueue_, 1, AlignBytes(1, sizeof(float)));
}

template <typename T>
__aicore__ inline void MoeV3FullLoadUnquantized<T>::Process()
{
    if (this->dropPadMode_ == DROP_PAD_MODE) {
        ProcessDropPadMode();
        return;
    }

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
            this->ScatterOutX();
            if (this->isInputScale_) {
                this->ScatterOutScale();
            }
        } else {
            if (this->isInputTopkWeight_) {
                this->TopkWeightGatherOut();
            }
            this->GatherOutX();
            if (this->isInputScale_) {
                this->GatherOutScale();
            }
        }

        this->FreeLocalTensor();
    }
}

template <typename T>
__aicore__ inline void MoeV3FullLoadUnquantized<T>::ProcessDropPadMode()
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

        this->ZeroOutX();
        if (this->isInputScale_) {
            this->ZeroOutScale();
        }
        if (this->isInputTopkWeight_) {
            this->TopkWeightZeroOut();
        }
    }
#ifndef __CCE_KT_TEST__
    AscendC::SyncAll();
#endif
    if (this->blockIdx_ < this->needCoreNum_) {
        if (this->isInputTopkWeight_) {
            this->TopkWeightGatherOut();
        }
        this->GatherOutX();
        if (this->isInputScale_) {
            this->GatherOutScale();
        }
        this->FreeLocalTensor();
    }
}

template <typename T>
__aicore__ inline void MoeV3FullLoadUnquantized<T>::ScatterOutX()
{
    LocalTensor<int32_t> sortedRowIdx = this->sortedRowIdxQueue_.template DeQue<int32_t>();

    LocalTensor<int32_t> sortedExpertIdx = this->sortedExpertIdxQueue_.template DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

    int64_t startRowIdx = this->blockIdx_ * this->perCoreIndicesElements_;
    int64_t endRowIdx = startRowIdx + this->coreIndicesElements_;

    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(this->cols_ * sizeof(T)), 0, 0, 0};

    if constexpr (IsSameType<T, hifloat8_t>::value) {
        DataCopyExtParams hif8CopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(this->cols_ * sizeof(uint8_t)),
                                         0, 0, 0};
        DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
        LocalTensor<uint8_t> xLocal = xCopyInQueue_.AllocTensor<uint8_t>();
        for (int64_t i = startRowIdx; i < endRowIdx && i < this->activeNum_; i++) {
            int32_t curExpertId = sortedExpertIdx.GetValue(i);
            if (curExpertId < this->expertStart_ || curExpertId >= this->expertEnd_) {
                break;
            }
            int64_t rowIdx = sortedRowIdx.GetValue(i);
            int64_t srcOffset = rowIdx / this->k_ * this->cols_;
            int64_t dstOffset = i * this->cols_;
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
            DataCopyPad(xLocal, xUint8tGm_[srcOffset], hif8CopyParams, padParams);
            xCopyInQueue_.EnQue(xLocal);
            LocalTensor<T> xOutLocal = xCopyInQueue_.DeQue<T>();
            SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
            DataCopyPad(expandedXGm_[dstOffset], xOutLocal, copyParams);
            xCopyInQueue_.FreeTensor(xOutLocal);
        }
    } else {
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        LocalTensor<T> xLocal = xCopyInQueue_.AllocTensor<T>();
        for (int64_t i = startRowIdx; i < endRowIdx && i < this->activeNum_; i++) {
            int32_t curExpertId = sortedExpertIdx.GetValue(i);
            if (curExpertId < this->expertStart_ || curExpertId >= this->expertEnd_) {
                break;
            }
            int64_t rowIdx = sortedRowIdx.GetValue(i);
            int64_t srcOffset = rowIdx / this->k_ * this->cols_;
            int64_t dstOffset = i * this->cols_;
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
            DataCopyPad(xLocal, xGm_[srcOffset], copyParams, padParams);
            SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
            DataCopyPad(expandedXGm_[dstOffset], xLocal, copyParams);
        }
        xCopyInQueue_.FreeTensor(xLocal);
    }

    this->sortedRowIdxQueue_.template EnQue<int32_t>(sortedRowIdx);
    this->sortedExpertIdxQueue_.template EnQue<int32_t>(sortedExpertIdx);
}

template <typename T>
__aicore__ inline void MoeV3FullLoadUnquantized<T>::GatherOutX()
{
    LocalTensor<int32_t> expandedRowIdx = this->expandedRowIdxQueue_.template DeQue<int32_t>();

    int64_t startRowIdx = this->blockIdx_ * this->perCoreIndicesElements_;
    int64_t rowLength = this->endXRow_ - this->startXRow_ + 1;
    int64_t inFactor = Align(this->cols_, sizeof(T));
    int64_t outputRows =
        this->dropPadMode_ == DROP_PAD_MODE ? this->outputRows_ : Min(this->actualExpertIdxNum_, this->activeNum_);

    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(this->cols_ * sizeof(T)), 0, 0, 0};

    if constexpr (IsSameType<T, hifloat8_t>::value) {
        DataCopyExtParams hif8CopyParams{static_cast<uint16_t>(rowLength),
                                         static_cast<uint32_t>(this->cols_ * sizeof(uint8_t)), 0, 0, 0};
        DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
        LocalTensor<uint8_t> xLocal = xCopyInQueue_.AllocTensor<uint8_t>();
        DataCopyPad(xLocal, xUint8tGm_[this->startXRow_ * this->cols_], hif8CopyParams, padParams);
        SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
        xCopyInQueue_.EnQue(xLocal);
        LocalTensor<T> xOutLocal = xCopyInQueue_.DeQue<T>();
        GatherOutXScatterLoop(expandedRowIdx, xOutLocal, startRowIdx, outputRows, inFactor);
        xCopyInQueue_.FreeTensor(xOutLocal);
    } else {
        DataCopyExtParams dataXCopyParams{static_cast<uint16_t>(rowLength),
                                          static_cast<uint32_t>(this->cols_ * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        LocalTensor<T> xLocal = xCopyInQueue_.AllocTensor<T>();
        DataCopyPad(xLocal, xGm_[this->startXRow_ * this->cols_], dataXCopyParams, padParams);
        SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
        GatherOutXScatterLoop(expandedRowIdx, xLocal, startRowIdx, outputRows, inFactor);
        xCopyInQueue_.FreeTensor(xLocal);
    }

    this->expandedRowIdxQueue_.template EnQue<int32_t>(expandedRowIdx);
}

template <typename T>
__aicore__ inline void MoeV3FullLoadUnquantized<T>::GatherOutXScatterLoop(const LocalTensor<int32_t> &expandedRowIdx,
                                                                          const LocalTensor<T> &xLocal,
                                                                          int64_t startRowIdx, int64_t outputRows,
                                                                          int64_t inFactor)
{
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(this->cols_ * sizeof(T)), 0, 0, 0};
    int64_t curIndex = startRowIdx;
    int64_t k = 0;
    for (int64_t i = this->startXRow_; i <= this->endXRow_; i++) {
        for (; k < this->coreIndicesElements_ && curIndex / this->k_ == i; curIndex++, k++) {
            int32_t outIndex = expandedRowIdx.GetValue(curIndex);
            if (outIndex >= 0 && outIndex < outputRows) {
                DataCopyPad(expandedXGm_[outIndex * this->cols_], xLocal[(i - this->startXRow_) * inFactor],
                            copyParams);
            }
        }
    }
}

template <typename T>
__aicore__ inline void MoeV3FullLoadUnquantized<T>::ScatterOutScale()
{
    LocalTensor<float> scaleLocal = scaleCopyInQueue_.AllocTensor<float>();
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};

    LocalTensor<int32_t> sortedRowIdx = this->sortedRowIdxQueue_.template DeQue<int32_t>();
    LocalTensor<int32_t> sortedExpertIdx = this->sortedExpertIdxQueue_.template DeQue<int32_t>();

    int64_t startRowIdx = this->blockIdx_ * this->perCoreIndicesElements_;
    int64_t endRowIdx = startRowIdx + this->coreIndicesElements_;

    for (int64_t i = startRowIdx; i < endRowIdx && i < this->activeNum_; i++) {
        int32_t curExpertId = sortedExpertIdx.GetValue(i);
        if (curExpertId < this->expertStart_ || curExpertId >= this->expertEnd_) {
            break;
        }
        int64_t rowIdx = sortedRowIdx.GetValue(i);
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        DataCopyPad(scaleLocal, scaleGm_[rowIdx / this->k_], copyParams, padParams);
        SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
        DataCopyPad(expandedScaleGm_[i], scaleLocal, copyParams);
    }

    this->sortedRowIdxQueue_.template EnQue<int32_t>(sortedRowIdx);
    this->sortedExpertIdxQueue_.template EnQue<int32_t>(sortedExpertIdx);
    scaleCopyInQueue_.FreeTensor(scaleLocal);
}

template <typename T>
__aicore__ inline void MoeV3FullLoadUnquantized<T>::GatherOutScale()
{
    LocalTensor<float> scaleLocal = scaleCopyInQueue_.AllocTensor<float>();
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};

    LocalTensor<int32_t> expandedRowIdx = this->expandedRowIdxQueue_.template DeQue<int32_t>();

    int64_t curIndex = this->blockIdx_ * this->perCoreIndicesElements_;
    int64_t k = 0;
    int64_t outputRows =
        this->dropPadMode_ == DROP_PAD_MODE ? this->outputRows_ : Min(this->actualExpertIdxNum_, this->activeNum_);

    for (int64_t i = this->startXRow_; i <= this->endXRow_; i++) {
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        DataCopyPad(scaleLocal, scaleGm_[i], copyParams, padParams);
        SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
        for (; k < this->coreIndicesElements_ && curIndex / this->k_ == i; curIndex++, k++) {
            int32_t outIndex = expandedRowIdx.GetValue(curIndex);
            if (outIndex >= 0 && outIndex < outputRows) {
                DataCopyPad(expandedScaleGm_[outIndex], scaleLocal, copyParams);
            }
        }
    }

    this->expandedRowIdxQueue_.template EnQue<int32_t>(expandedRowIdx);
    scaleCopyInQueue_.FreeTensor(scaleLocal);
}

template <typename T>
__aicore__ inline void MoeV3FullLoadUnquantized<T>::ZeroOutX()
{
    int64_t perCoreRows = Ceil(this->outputRows_, this->needCoreNum_);
    int64_t startRow = this->blockIdx_ * perCoreRows;
    int64_t endRow = Min(startRow + perCoreRows, this->outputRows_);
    if (startRow >= endRow) {
        return;
    }
    int64_t rowCount = endRow - startRow;

    if constexpr (IsSameType<T, hifloat8_t>::value) {
        GlobalTensor<uint8_t> zeroGm;
        zeroGm.SetGlobalBuffer((__gm__ uint8_t *)expandedXGm_.GetPhyAddr() + startRow * this->cols_,
                               rowCount * this->cols_);
        InitGlobalMemory(zeroGm, rowCount * this->cols_, static_cast<uint8_t>(0));
    } else {
        GlobalTensor<T> zeroGm;
        zeroGm.SetGlobalBuffer((__gm__ T *)expandedXGm_.GetPhyAddr() + startRow * this->cols_, rowCount * this->cols_);
        InitGlobalMemory(zeroGm, rowCount * this->cols_, static_cast<T>(0));
    }
}

template <typename T>
__aicore__ inline void MoeV3FullLoadUnquantized<T>::ZeroOutScale()
{
    int64_t perCoreRows = Ceil(this->outputRows_, this->needCoreNum_);
    int64_t startRow = this->blockIdx_ * perCoreRows;
    int64_t endRow = Min(startRow + perCoreRows, this->outputRows_);
    if (startRow >= endRow) {
        return;
    }
    int64_t rowCount = endRow - startRow;
    GlobalTensor<float> scaleZeroGm;
    scaleZeroGm.SetGlobalBuffer((__gm__ float *)expandedScaleGm_.GetPhyAddr() + startRow, rowCount);
    InitGlobalMemory(scaleZeroGm, rowCount, 0.0f);
}

} // namespace MoeInitRoutingV3
#endif // MOE_V3_FULL_LOAD_UNQUANTIZED_ARCH35_H
