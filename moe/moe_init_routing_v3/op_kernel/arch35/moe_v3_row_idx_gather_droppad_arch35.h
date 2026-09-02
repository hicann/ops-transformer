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
 * \file moe_v3_row_idx_gather_droppad_arch35.h
 * \brief
 */
#ifndef MOE_V3_ROW_IDX_GATHER_DROPPAD_H_REGBASE_ARCH35
#define MOE_V3_ROW_IDX_GATHER_DROPPAD_H_REGBASE_ARCH35

#include "moe_v3_common.h"
#include "kernel_operator.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

template <typename T>
class MoeV3RowIdxGatherDropPad {
public:
    __aicore__ inline MoeV3RowIdxGatherDropPad(){};
    __aicore__ inline void Init(GM_ADDR expandedRowIdx, GM_ADDR expandedX, GM_ADDR expandedScale, GM_ADDR workspace,
                                const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress);
    __aicore__ inline void CopyOut(int64_t progress);
    __aicore__ inline void SyncAll();
    __aicore__ inline void AssistInit();
    __aicore__ inline void ZeroOutAllOutput();
    __aicore__ inline void InitBasicParams(const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe);

private:
    TPipe *pipe_;
    TQue<QuePosition::VECIN, 1> copyInQueue_;
    TQue<QuePosition::VECOUT, 1> copyOutQueue_;

    GlobalTensor<int32_t> expandDstToSrcRowGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<int32_t> outputToSrcRowGm_;
    GlobalTensor<int32_t> expertIdxValueGm_;
    GlobalTensor<int32_t> expandedExpertIdxGm_;
    GlobalTensor<T> expandedXGm_;
    GlobalTensor<uint8_t> expandedXUint8Gm_;
    GlobalTensor<float> expandedScaleGm_;

    const MoeV3Arch35SrcToDstCapacityComputeTilingData *srcToDstTilingData_;
    int64_t coreNum_;
    int64_t blockIdx_;
    int64_t totalLength_;
    int64_t currentLoopRows_;
    int64_t coreRows_;
    int64_t perLoopRows_;
    int64_t lastLoopRows_;
    int64_t rowLoops_;
    int64_t expertCapacity_;
    int64_t expertNum_;
    int64_t outputRows_;
    int64_t cols_;
    int64_t k_;
    int64_t perLoopCols_;
    int64_t lastLoopCols_;
    int64_t colLoops_;
    int64_t isInputScale_;
    int64_t quantMode_;
    int64_t useCompactOutputRows_ = 0;

    int64_t tokenCount_ = 0;
    int32_t lastExpertId_ = -1;
    int32_t lastCoreExpertId_ = 0;
    int32_t lastCoreExpertIdNum_ = 0;
};

template <typename T>
__aicore__ inline void MoeV3RowIdxGatherDropPad<T>::AssistInit()
{
    if (this->blockIdx_ != 0) {
        this->lastCoreExpertId_ = expertIdxValueGm_.GetValue((this->blockIdx_ - 1) * 2);
        this->lastCoreExpertIdNum_ = expertIdxValueGm_.GetValue((this->blockIdx_ - 1) * 2 + 1);
        for (int64_t i = this->blockIdx_ - 2; i >= 0; i--) {
            int32_t lastExpertIdx = expertIdxValueGm_.GetValue(i * 2);
            if (lastExpertIdx < this->lastCoreExpertId_) {
                break;
            }
            int32_t lastExpertNum = expertIdxValueGm_.GetValue(i * 2 + 1);
            this->lastCoreExpertIdNum_ += lastExpertNum;
        }
    }
}

template <typename T>
__aicore__ inline void MoeV3RowIdxGatherDropPad<T>::ZeroOutAllOutput()
{
    int64_t perCoreRows = Ceil(this->outputRows_, this->coreNum_);
    int64_t startIndex = this->blockIdx_ * perCoreRows;
    int64_t endIndex = Min(startIndex + perCoreRows, this->outputRows_);
    int64_t rowCount = endIndex - startIndex;
    if (rowCount <= 0) {
        return;
    }

    // DropPad padding rows may never be touched by gather_out. Pre-clear the whole output once, then
    // synchronize before valid rows are written, avoiding many small expert-gap clears.
    if constexpr (IsSameType<T, hifloat8_t>::value) {
        GlobalTensor<uint8_t> zeroGm;
        zeroGm.SetGlobalBuffer((__gm__ uint8_t *)expandedXUint8Gm_.GetPhyAddr() + startIndex * this->cols_,
                               rowCount * this->cols_);
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        InitGlobalMemory(zeroGm, rowCount * this->cols_, static_cast<uint8_t>(0));
    } else {
        GlobalTensor<T> zeroGm;
        zeroGm.SetGlobalBuffer((__gm__ T *)expandedXGm_.GetPhyAddr() + startIndex * this->cols_,
                               rowCount * this->cols_);
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        InitGlobalMemory(zeroGm, rowCount * this->cols_, static_cast<T>(0));
    }
    SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);

    if (this->isInputScale_ == 1) {
        GlobalTensor<float> scaleZeroGm;
        scaleZeroGm.SetGlobalBuffer((__gm__ float *)expandedScaleGm_.GetPhyAddr() + startIndex, rowCount);
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        InitGlobalMemory(scaleZeroGm, rowCount, static_cast<float>(0));
        SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    }

    if (this->useCompactOutputRows_ == 1) {
        GlobalTensor<int32_t> sourceRowGm;
        sourceRowGm.SetGlobalBuffer((__gm__ int32_t *)outputToSrcRowGm_.GetPhyAddr() + startIndex, rowCount);
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        InitGlobalMemory(sourceRowGm, rowCount, static_cast<int32_t>(-1));
        SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    }
}

template <typename T>
__aicore__ inline void MoeV3RowIdxGatherDropPad<T>::CopyIn(int64_t progress)
{
    LocalTensor<int32_t> inLocal = copyInQueue_.AllocTensor<int32_t>();
    int64_t length = Align(currentLoopRows_, sizeof(int32_t));
    DataCopy(inLocal, expandDstToSrcRowGm_[progress * perLoopRows_], length);
    DataCopy(inLocal[length], expandedExpertIdxGm_[progress * perLoopRows_], length);
    copyInQueue_.EnQue<int32_t>(inLocal);
}

template <typename T>
__aicore__ inline void MoeV3RowIdxGatherDropPad<T>::CopyOut(int64_t progress)
{
    LocalTensor<int32_t> inLocal = copyInQueue_.DeQue<int32_t>();
    LocalTensor<int32_t> outLocal = copyOutQueue_.AllocTensor<int32_t>();
    int64_t length = Align(currentLoopRows_, sizeof(int32_t));
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};

    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    if (this->lastExpertId_ == -1) {
        this->lastExpertId_ = this->lastCoreExpertId_;
        this->tokenCount_ = this->lastCoreExpertIdNum_;
    }
    for (int64_t idx = 0; idx < currentLoopRows_; idx++) {
        int32_t expertIdx = inLocal[length].GetValue(idx);
        if (expertIdx < 0 || expertIdx >= this->expertNum_) {
            continue;
        }
        int32_t index = 0;
        if (this->lastExpertId_ < 0) {
            this->lastExpertId_ = 0;
            this->tokenCount_ = 0;
        }
        if (this->tokenCount_ > this->expertCapacity_) {
            this->tokenCount_ = this->expertCapacity_;
        }
        if (this->lastExpertId_ < expertIdx) {
            this->lastExpertId_ = expertIdx;
            this->tokenCount_ = 0;
        }

        if (this->tokenCount_ < this->expertCapacity_) {
            int32_t outOffset = inLocal.GetValue(idx);
            index = expertIdx * this->expertCapacity_ + this->tokenCount_;
            outLocal.SetValue(0, index);
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopyPad(expandedRowIdxGm_[outOffset], outLocal, copyParams);

            if (this->useCompactOutputRows_ == 1) {
                SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
                outLocal.SetValue(0, outOffset / this->k_);
                SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                DataCopyPad(outputToSrcRowGm_[index], outLocal, copyParams);
                SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
            }
            this->tokenCount_++;
        }
    }
    copyInQueue_.FreeTensor(inLocal);
    copyOutQueue_.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void MoeV3RowIdxGatherDropPad<T>::SyncAll()
{
    if (coreNum_ == 1) {
        return;
    }
#ifndef __CCE_KT_TEST__
    AscendC::SyncAll();
#endif
}

template <typename T>
__aicore__ inline void MoeV3RowIdxGatherDropPad<T>::InitBasicParams(const MoeInitRoutingV3Arch35TilingData *tilingData,
                                                                    TPipe *tPipe)
{
    pipe_ = tPipe;
    this->blockIdx_ = GetBlockIdx();

    this->coreNum_ = tilingData->coreNum;
    this->totalLength_ = tilingData->n * tilingData->k;
    this->srcToDstTilingData_ = &(tilingData->srcToDstDropPadParamsOp);
    this->expertNum_ = tilingData->expertNum;
    this->expertCapacity_ = tilingData->expertCapacity;
    this->outputRows_ = this->expertNum_ * this->expertCapacity_;
    this->cols_ = tilingData->cols;
    this->k_ = tilingData->k;
    this->quantMode_ = tilingData->quantMode;
    this->isInputScale_ = tilingData->isInputScale;
    this->useCompactOutputRows_ = tilingData->gatherOutComputeParamsOp.useCompactGatherOutDropPad;

    if (this->blockIdx_ == this->srcToDstTilingData_->needCoreNum - 1) {
        this->coreRows_ = this->srcToDstTilingData_->lastCoreRows;
        this->perLoopRows_ = this->srcToDstTilingData_->lastCorePerLoopRows;
        this->lastLoopRows_ = this->srcToDstTilingData_->lastCoreLastLoopRows;
        this->rowLoops_ = this->srcToDstTilingData_->lastCoreLoops;
    } else {
        this->coreRows_ = this->srcToDstTilingData_->perCoreRows;
        this->perLoopRows_ = this->srcToDstTilingData_->perCorePerLoopRows;
        this->lastLoopRows_ = this->srcToDstTilingData_->perCoreLastLoopRows;
        this->rowLoops_ = this->srcToDstTilingData_->perCoreLoops;
    }
    this->perLoopCols_ = this->srcToDstTilingData_->perLoopCols;
    this->lastLoopCols_ = this->srcToDstTilingData_->lastLoopCols;
    this->colLoops_ = this->srcToDstTilingData_->colLoops;
}

template <typename T>
__aicore__ inline void MoeV3RowIdxGatherDropPad<T>::Init(GM_ADDR expandedRowIdx, GM_ADDR expandedX,
                                                         GM_ADDR expandedScale, GM_ADDR workspace,
                                                         const MoeInitRoutingV3Arch35TilingData *tilingData,
                                                         TPipe *tPipe)
{
    InitBasicParams(tilingData, tPipe);

    int64_t length = Align(this->totalLength_, sizeof(int32_t));
    expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx, length);

    if constexpr (IsSameType<T, hifloat8_t>::value) {
        expandedXUint8Gm_.SetGlobalBuffer((__gm__ uint8_t *)expandedX,
                                          this->expertNum_ * this->expertCapacity_ * this->cols_);
    } else {
        expandedXGm_.SetGlobalBuffer((__gm__ T *)expandedX, this->expertNum_ * this->expertCapacity_ * this->cols_);
    }

    if (this->isInputScale_ == 1) {
        expandedScaleGm_.SetGlobalBuffer((__gm__ float *)expandedScale, this->expertNum_ * this->expertCapacity_);
    }

    expandedExpertIdxGm_.SetGlobalBuffer(
        (__gm__ int32_t *)workspace + this->blockIdx_ * this->srcToDstTilingData_->perCoreRows,
        Align(this->coreRows_, sizeof(int32_t)));
    expandDstToSrcRowGm_.SetGlobalBuffer(
        (__gm__ int32_t *)workspace + length + this->blockIdx_ * this->srcToDstTilingData_->perCoreRows,
        Align(this->coreRows_, sizeof(int32_t)));
    // expertIdxValueGm偏移地址必须与expert_tokens_count.h保持一致
    // expert_tokens_count.h偏移: Align(n*k)*2 + Align(actualExpertNum) (DropPad模式不保留expertTotalCountGm空间)
    int64_t actualExpertNumOffset = Align(tilingData->actualExpertNum, sizeof(int32_t));
    expertIdxValueGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + length * 2 + actualExpertNumOffset,
                                      this->coreNum_ * 2);
    if (this->useCompactOutputRows_ == 1) {
        outputToSrcRowGm_.SetGlobalBuffer(
            (__gm__ int32_t *)workspace + length * 2 + actualExpertNumOffset + this->coreNum_ * 2, this->outputRows_);
    }

    pipe_->InitBuffer(copyInQueue_, 1, AlignBytes(this->perLoopRows_, sizeof(int32_t)) * 2);
    pipe_->InitBuffer(copyOutQueue_, 1, AlignBytes(INT32_ONE_BLOCK_NUM, sizeof(int32_t)));
}

template <typename T>
__aicore__ inline void MoeV3RowIdxGatherDropPad<T>::Process()
{
    ZeroOutAllOutput();
    this->SyncAll();

    if (this->blockIdx_ < this->srcToDstTilingData_->needCoreNum) {
        AssistInit();
        currentLoopRows_ = perLoopRows_;
        for (int64_t loop = 0; loop < this->rowLoops_; loop++) {
            if (loop == this->rowLoops_ - 1) {
                currentLoopRows_ = lastLoopRows_;
            }
            CopyIn(loop);
            CopyOut(loop);
        }
    }
    this->SyncAll();
}
} // namespace MoeInitRoutingV3
#endif // MOE_V3_ROW_IDX_GATHER_DROPPAD_H_REGBASE_ARCH35
