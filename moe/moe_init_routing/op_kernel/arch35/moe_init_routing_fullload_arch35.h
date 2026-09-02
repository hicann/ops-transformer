/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_init_routing_fullload_arch35.h
 * \brief
 */
#ifndef MOE_FULL_LOAD_ARCH35_H
#define MOE_FULL_LOAD_ARCH35_H

#include "moe_mrgsort_arch35.h"

namespace MoeInitRouting {
using namespace AscendC;

constexpr int32_t BLOCK_B32_SIZE = 8;
constexpr int32_t REPEAT_B32_SIZE = 64;
constexpr int32_t CONSTANT_TWO = 2;
constexpr int32_t CONSTANT_THREE = 3;
constexpr int32_t CONSTANT_FOUR = 4;
constexpr int32_t CONSTANT_FIVE = 5;
constexpr int32_t CONSTANT_SIX = 6;
constexpr int32_t CONSTANT_SEVEN = 7;

template <typename T>
class MoeFullLoad : public MoeSortBase {
public:
    __aicore__ inline MoeFullLoad(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR rowIdx, GM_ADDR expertIdx, GM_ADDR expandedX, GM_ADDR expandedRowIdx,
                                GM_ADDR expandedExpertIdx, GM_ADDR workspace,
                                const MoeInitRoutingTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn();
    __aicore__ inline void SortCompute();
    __aicore__ inline void CopyOutIdx();
    __aicore__ inline void CopyOutEmpty();
    __aicore__ inline void CopyOutX();
    __aicore__ inline void ArithProgressionPerf(const LocalTensor<int32_t> &dst, const int32_t firstValue,
                                                const int32_t diffValue, int32_t countAlign);

private:
    int64_t sortNum_;
    const GatherOutComputeTilingData *gatherOutTilingData_;
    int64_t blockIdx_;
    int64_t needCoreNum_;
    int64_t coreRows_;
    int64_t perCoreRows_;
    int64_t k_;
    int64_t n_;
    int64_t cols_;
    int64_t activateRows_;
    int64_t coreK_;
    int64_t perCoreK_;
    int64_t splitFlag_;

    TQue<QuePosition::VECIN, 1> xCopyInQueue_;
    TQue<QuePosition::VECOUT, 1> expandedRowIdxCopyOutQueue_;
    TQue<QuePosition::VECOUT, 1> expandedExpertIdxCopyOutQueue_;
    TQue<QuePosition::VECOUT, 1> expandDstToSrcRowQueue_;

    GlobalTensor<T> xGm_;
    GlobalTensor<int32_t> rowIdxGm_;
    GlobalTensor<int32_t> expertIdxGm_;

    GlobalTensor<T> expandedXGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<int32_t> expandedExpertIdxGm_;
};

template <typename T>
__aicore__ inline void MoeFullLoad<T>::ArithProgressionPerf(const LocalTensor<int32_t> &dst, const int32_t firstValue,
                                                            const int32_t diffValue, int32_t countAlign)
{
    // countAlign must be eight aligned
    countAlign = (countAlign + CONSTANT_SEVEN) / BLOCK_B32_SIZE * BLOCK_B32_SIZE;
    for (int32_t index = 0; index <= CONSTANT_SEVEN; index++) {
        dst.SetValue(index, firstValue + diffValue * index);
    }
    auto eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventID);
    WaitFlag<HardEvent::S_V>(eventID);
    if (countAlign > BLOCK_B32_SIZE) {
        int32_t offset;
        if (countAlign > REPEAT_B32_SIZE) {
            for (int32_t i = 1; i < BLOCK_B32_SIZE; i++) {
                offset = i * BLOCK_B32_SIZE;
                Adds(dst[offset], dst, diffValue * offset, BLOCK_B32_SIZE, 1, {1, 1, 1, 1});
            }
            int32_t loopTimes = (countAlign + REPEAT_B32_SIZE - 1) / REPEAT_B32_SIZE;
            for (int32_t i = 1; i < loopTimes - 1; i++) {
                offset = i * REPEAT_B32_SIZE;
                Adds(dst[offset], dst, diffValue * offset, REPEAT_B32_SIZE, 1, {1, 1, 1, 1});
            }
            offset = (loopTimes - 1) * REPEAT_B32_SIZE;
            Adds(dst[offset], dst, diffValue * offset, countAlign - offset, 1, {1, 1, 1, 1});
        } else {
            for (int32_t i = 1; i < countAlign / BLOCK_B32_SIZE; i++) {
                offset = i * BLOCK_B32_SIZE;
                Adds(dst[offset], dst, diffValue * offset, BLOCK_B32_SIZE, 1, {1, 1, 1, 1});
            }
        }
    }
}

template <typename T>
__aicore__ inline void MoeFullLoad<T>::CopyIn()
{
    LocalTensor<T> xLocal = xCopyInQueue_.AllocTensor<T>();
    LocalTensor<int32_t> inLocal = sortDataCopyInQueue.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1),
                                     static_cast<uint32_t>(this->totalLength * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(inLocal[0], expertIdxGm_, dataCopyParams, dataCopyPadParams);
    DataCopyPad(inLocal[this->sortNum_], rowIdxGm_, dataCopyParams, dataCopyPadParams);

    DataCopyExtParams dataXCopyParams{static_cast<uint16_t>(this->coreRows_),
                                      static_cast<uint32_t>(this->cols_ * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> dataXCopyPadParams{false, 0, 0, static_cast<T>(0)};
    int64_t outIndex = 0;
    if (this->splitFlag_ == SPLIT_N) {
        outIndex = this->blockIdx_ * this->perCoreRows_ * this->cols_;
    }
    DataCopyPad(xLocal, xGm_[outIndex], dataXCopyParams, dataXCopyPadParams);
    xCopyInQueue_.EnQue(xLocal);
    sortDataCopyInQueue.EnQue(inLocal);
}

template <typename T>
__aicore__ inline void MoeFullLoad<T>::SortCompute()
{
    LocalTensor<int32_t> inLocal = sortDataCopyInQueue.DeQue<int32_t>();
    LocalTensor<int32_t> expertIdxLocal = inLocal[0];
    LocalTensor<uint32_t> rowIdxLocal = inLocal[this->sortNum_].template ReinterpretCast<uint32_t>();
    LocalTensor<float> expertIdxLocalFp32 = expertIdxLocal.ReinterpretCast<float>();
    Cast(expertIdxLocalFp32, expertIdxLocal, RoundMode::CAST_ROUND, this->totalLength);
    Muls(expertIdxLocalFp32, expertIdxLocalFp32, (float)-1, this->totalLength);
    int64_t duplicateNum = this->totalLength % ONE_REPEAT_SORT_NUM;
    if (duplicateNum > 0) {
        int duplicateIndex = this->totalLength - duplicateNum;
        uint64_t mask0 = (UINT64_MAX << duplicateNum) & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
        uint64_t mask[2] = {mask0, 0};
        Duplicate(expertIdxLocalFp32[duplicateIndex], MIN_FP32, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);
    }
    LocalTensor<float> concatLocal;
    LocalTensor<float> tempTensor = tempBuffer.Get<float>(GetSortLen<float>(this->sortNum_));
    Concat(concatLocal, expertIdxLocalFp32, tempTensor, this->sortNum_ / ONE_REPEAT_SORT_NUM);
    LocalTensor<float> sortedLocal = sortedBuffer.Get<float>(GetSortLen<float>(this->sortNum_));
    Sort<float, true>(sortedLocal, concatLocal, rowIdxLocal, tempTensor, this->sortNum_ / ONE_REPEAT_SORT_NUM);
    LocalTensor<float> expandedExpertIdxLocal = expandedExpertIdxCopyOutQueue_.AllocTensor<float>();
    LocalTensor<uint32_t> expandDstToSrcRowLocal = expandDstToSrcRowQueue_.AllocTensor<uint32_t>();
    LocalTensor<float> expandDstToSrcRowLocalFp32 = expandDstToSrcRowLocal.ReinterpretCast<float>();
    Extract(expandedExpertIdxLocal, expandDstToSrcRowLocal, sortedLocal, this->sortNum_ / ONE_REPEAT_SORT_NUM);
    Cast(expandDstToSrcRowLocalFp32, expandDstToSrcRowLocal.ReinterpretCast<int32_t>(), RoundMode::CAST_ROUND,
         this->totalLength);
    Muls(expandedExpertIdxLocal, expandedExpertIdxLocal, (float)-1, this->totalLength);
    LocalTensor<int32_t> expandedExpertIdxLocalInt32;
    expandedExpertIdxLocalInt32 = expandedExpertIdxLocal.ReinterpretCast<int32_t>();
    Cast(expandedExpertIdxLocalInt32, expandedExpertIdxLocal, RoundMode::CAST_ROUND, this->totalLength);
    expandedExpertIdxCopyOutQueue_.EnQue<int32_t>(expandedExpertIdxLocalInt32);

    LocalTensor<uint32_t> expandedRowIdx = expandedRowIdxCopyOutQueue_.AllocTensor<uint32_t>();
    LocalTensor<uint32_t> expandedRowIdxU32 = expandedRowIdx.ReinterpretCast<uint32_t>();
    Muls(expandDstToSrcRowLocalFp32, expandDstToSrcRowLocalFp32, (float)-1, this->totalLength);
    ArithProgressionPerf(inLocal[this->sortNum_], 0, 1, this->totalLength);
    if (duplicateNum > 0) {
        int duplicateIndex = this->totalLength - duplicateNum;
        uint64_t mask0 = (UINT64_MAX << duplicateNum) & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
        uint64_t mask[2] = {mask0, 0};
        Duplicate(expandDstToSrcRowLocalFp32[duplicateIndex], MIN_FP32, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);
    }
    Concat(concatLocal, expandDstToSrcRowLocalFp32, tempTensor, this->sortNum_ / ONE_REPEAT_SORT_NUM);
    Sort<float, true>(sortedLocal, concatLocal, rowIdxLocal, tempTensor, this->sortNum_ / ONE_REPEAT_SORT_NUM);
    Extract(tempTensor, expandedRowIdxU32, sortedLocal, this->sortNum_ / ONE_REPEAT_SORT_NUM);
    expandedRowIdxCopyOutQueue_.EnQue<uint32_t>(expandedRowIdx);
    sortDataCopyInQueue.FreeTensor(inLocal);

    expandDstToSrcRowQueue_.FreeTensor(expandDstToSrcRowLocal);
}

template <typename T>
__aicore__ inline void MoeFullLoad<T>::CopyOutIdx()
{
    LocalTensor<int32_t> expandedExpertIdx = expandedExpertIdxCopyOutQueue_.DeQue<int32_t>();
    LocalTensor<int32_t> expandedRowIdx = expandedRowIdxCopyOutQueue_.DeQue<int32_t>();
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = this->totalLength * sizeof(int32_t);
    DataCopyPad(expandedExpertIdxGm_, expandedExpertIdx, intriParams);
    DataCopyPad(expandedRowIdxGm_, expandedRowIdx, intriParams);
    expandedExpertIdxCopyOutQueue_.FreeTensor(expandedExpertIdx);
    expandedRowIdxCopyOutQueue_.EnQue(expandedRowIdx);
}

template <typename T>
__aicore__ inline void MoeFullLoad<T>::CopyOutX()
{
    LocalTensor<int32_t> expendRowIdx = expandedRowIdxCopyOutQueue_.DeQue<int32_t>();
    LocalTensor<T> x = xCopyInQueue_.DeQue<T>();
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = this->cols_ * sizeof(T);
    int64_t inFactor = Align(this->cols_, sizeof(T));
    int64_t idxStart;
    if (this->splitFlag_ == SPLIT_N) {
        idxStart = this->blockIdx_ * this->perCoreRows_;
    } else {
        idxStart = this->blockIdx_ * this->perCoreRows_ * this->perCoreK_;
    }
    int outIndex;
    for (int64_t i = 0; i < this->coreRows_; i++) {
        for (int64_t k = 0; k < this->coreK_; k++) {
            outIndex = expendRowIdx.GetValue(idxStart + this->n_ * k + i);
            if (outIndex < this->activateRows_) {
                event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
                SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
                WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
                DataCopyPad(expandedXGm_[outIndex * this->cols_], x[i * inFactor], intriParams);
            }
        }
    }
    expandedRowIdxCopyOutQueue_.FreeTensor(expendRowIdx);
    xCopyInQueue_.FreeTensor(x);
}

template <typename T>
__aicore__ inline void MoeFullLoad<T>::CopyOutEmpty()
{
    LocalTensor<int32_t> outLocal = expandedExpertIdxCopyOutQueue_.DeQue<int32_t>();
    expandedExpertIdxCopyOutQueue_.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void MoeFullLoad<T>::Init(GM_ADDR x, GM_ADDR rowIdx, GM_ADDR expertIdx, GM_ADDR expandedX,
                                            GM_ADDR expandedRowIdx, GM_ADDR expandedExpertIdx, GM_ADDR workspace,
                                            const MoeInitRoutingTilingData *tilingData, TPipe *tPipe)
{
    this->gatherOutTilingData_ = &(tilingData->gatherOutComputeParamsOp);
    this->blockIdx_ = GetBlockIdx();
    splitFlag_ = gatherOutTilingData_->splitFlag;
    this->k_ = tilingData->k;
    this->n_ = tilingData->n;
    this->cols_ = tilingData->cols;
    this->needCoreNum_ = this->gatherOutTilingData_->needCoreNum;
    this->perCoreRows_ = this->gatherOutTilingData_->perCoreRows;
    this->perCoreK_ = this->gatherOutTilingData_->perCoreK;
    this->activateRows_ = this->gatherOutTilingData_->activateRows;
    if (this->blockIdx_ == this->gatherOutTilingData_->needCoreNum - 1) {
        this->coreRows_ = this->gatherOutTilingData_->lastCoreRows;
        this->coreK_ = this->gatherOutTilingData_->lastCoreK;
    } else {
        this->coreRows_ = this->gatherOutTilingData_->perCoreRows;
        this->coreK_ = this->gatherOutTilingData_->perCoreK;
    }

    this->tileLength = Align(tilingData->vbsComputeParamsOp.lastCorePerLoopElements, sizeof(int32_t));
    this->sortNum_ = Ceil(this->tileLength, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
    this->totalLength = tilingData->n * tilingData->k;
    this->pipe = tPipe;

    xGm_.SetGlobalBuffer((__gm__ T *)x);
    rowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)rowIdx, this->tileLength);
    expertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expertIdx, this->tileLength);

    expandedXGm_.SetGlobalBuffer((__gm__ T *)expandedX);
    expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx, this->tileLength);
    expandedExpertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedExpertIdx, this->tileLength);

    int64_t kvFactor = 2;
    int64_t buffSize = this->sortNum_ * sizeof(int32_t);

    pipe->InitBuffer(xCopyInQueue_, 1, AlignBytes(this->cols_, sizeof(T)) * this->coreRows_);
    pipe->InitBuffer(expandedRowIdxCopyOutQueue_, bufferNum, buffSize);
    pipe->InitBuffer(expandedExpertIdxCopyOutQueue_, bufferNum, buffSize);
    pipe->InitBuffer(expandDstToSrcRowQueue_, bufferNum, buffSize);
    pipe->InitBuffer(sortDataCopyInQueue, bufferNum, buffSize * kvFactor);
    pipe->InitBuffer(tempBuffer, buffSize * kvFactor);
    pipe->InitBuffer(sortedBuffer, buffSize * kvFactor);
}

template <typename T>
__aicore__ inline void MoeFullLoad<T>::Process()
{
    if (this->totalLength <= 0) {
        return;
    }
    if (this->blockIdx_ == 0) {
        CopyIn();
        SortCompute();
        CopyOutIdx();
        CopyOutX();
    } else if (this->blockIdx_ < this->needCoreNum_) {
        CopyIn();
        SortCompute();
        CopyOutEmpty();
        CopyOutX();
    }
}
} // namespace MoeInitRouting
#endif // MOE_FULL_LOAD_ARCH35_H
