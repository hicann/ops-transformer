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
 * \file moe_v3_full_load_base.h
 * \brief
 */
#ifndef MOE_V3_FULL_LOAD_BASE_H
#define MOE_V3_FULL_LOAD_BASE_H

#include "moe_v3_common.h"
#include "simt_api/asc_simt.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void FullLoadComputeExpertFirstIndexSimt(
    int32_t elementNum, int32_t expertStart, int32_t expertEnd, __ubuf__ int32_t *sortedExpertIdLocalAddr,
    __ubuf__ int32_t *expertFirstIndexLocalAddr)
{
    for (auto i = static_cast<int32_t>(threadIdx.x); i < elementNum; i += static_cast<int32_t>(blockDim.x)) {
        auto currExpertId = sortedExpertIdLocalAddr[i];
        if (currExpertId < expertStart) {
            continue;
        }
        if (currExpertId >= expertEnd) {
            break;
        }
        auto prevExpertId = (i == 0 ? -1 : sortedExpertIdLocalAddr[i - 1]);
        if (currExpertId != prevExpertId) {
            expertFirstIndexLocalAddr[currExpertId - expertStart] = i;
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void FullLoadComputeExpertCountOutSimt(
    int32_t elementNum, int32_t expertStart, int32_t expertEnd, __ubuf__ int32_t *sortedExpertIdLocalAddr,
    __ubuf__ int32_t *expertFirstIndexLocalAddr, __ubuf__ int32_t *expertCountOutLocalAddr)
{
    for (auto i = static_cast<int32_t>(threadIdx.x); i < elementNum; i += static_cast<int32_t>(blockDim.x)) {
        auto currExpertId = sortedExpertIdLocalAddr[i];
        if (currExpertId < expertStart) {
            continue;
        }
        if (currExpertId >= expertEnd) {
            break;
        }
        if (i == elementNum - 1 || currExpertId != sortedExpertIdLocalAddr[i + 1]) {
            expertCountOutLocalAddr[currExpertId - expertStart] =
                i + 1 - expertFirstIndexLocalAddr[currExpertId - expertStart];
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void FullLoadComputeActualIdxNumSimt(
    int32_t elementNum, int32_t expertStart, int32_t expertEnd, __ubuf__ int32_t *sortedExpertIdLocalAddr,
    __ubuf__ int32_t *actualIdxNumAddr)
{
    int32_t localCount = 0;
    for (auto i = static_cast<int32_t>(threadIdx.x); i < elementNum; i += static_cast<int32_t>(blockDim.x)) {
        auto currExpertId = sortedExpertIdLocalAddr[i];
        if (currExpertId >= expertStart && currExpertId < expertEnd) {
            localCount++;
        }
    }
    asc_atomic_add(actualIdxNumAddr, localCount);
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void FullLoadTopkWeightGatherSimt(
    int64_t elements, int64_t indexBase, int64_t totalLength, int64_t expertStart, int64_t expertEnd,
    __ubuf__ int32_t *sortedRowIdx, __ubuf__ int32_t *sortedExpertIdx, __gm__ float *topkWeight,
    __gm__ volatile float *expandedTopkWeight)
{
    for (int64_t index = static_cast<int64_t>(threadIdx.x); index < elements;
         index += static_cast<int64_t>(blockDim.x)) {
        int64_t dstIndex = indexBase + index;
        int32_t expertId = sortedExpertIdx[dstIndex];
        int32_t srcIndex = sortedRowIdx[dstIndex];
        if (expertId >= expertStart && expertId < expertEnd && srcIndex >= 0 && srcIndex < totalLength) {
            expandedTopkWeight[dstIndex] = topkWeight[srcIndex];
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void FullLoadTopkWeightScatterSimt(
    int64_t elements, int64_t indexBase, int64_t outputRows, __ubuf__ int32_t *srcToDstRow, __gm__ float *topkWeight,
    __gm__ volatile float *expandedTopkWeight)
{
    for (int64_t index = static_cast<int64_t>(threadIdx.x); index < elements;
         index += static_cast<int64_t>(blockDim.x)) {
        int64_t srcIndex = indexBase + index;
        int32_t dstIndex = srcToDstRow[srcIndex];
        if (dstIndex >= 0 && dstIndex < outputRows) {
            expandedTopkWeight[dstIndex] = topkWeight[srcIndex];
        }
    }
}

template <typename T>
class MoeV3FullLoadBase {
public:
    __aicore__ inline MoeV3FullLoadBase(){};
    __aicore__ inline void Init(GM_ADDR expertIdx, GM_ADDR expandedRowIdx, GM_ADDR expertTokensCountOrCumsum,
                                GM_ADDR topkWeight, GM_ADDR expandedTopkWeight, GM_ADDR workspace,
                                const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe);

protected:
    __aicore__ inline void CopyIn();
    __aicore__ inline void InitGlobalBuffers(GM_ADDR expertIdx, GM_ADDR expandedRowIdx,
                                             GM_ADDR expertTokensCountOrCumsum, GM_ADDR topkWeight,
                                             GM_ADDR expandedTopkWeight, GM_ADDR workspace);
    __aicore__ inline void InitQueBuffers();
    __aicore__ inline void SortCompute();
    __aicore__ inline void FilterExpertIdx(LocalTensor<int32_t> &expertIdxLocal,
                                           LocalTensor<float> &expertIdxLocalFp32);
    __aicore__ inline void ComputeGatherIdx(LocalTensor<int32_t> &inLocal);
    __aicore__ inline void ComputeGatherIdxWithCapacity();
    __aicore__ inline void CopyOutRowIdx();
    __aicore__ inline void ComputeExpertTokenCount();
    __aicore__ inline void CopyExpertCountToOutput();
    __aicore__ inline void FreeLocalTensor();
    __aicore__ inline void TopkWeightScatterOut();
    __aicore__ inline void TopkWeightGatherOut();
    __aicore__ inline void TopkWeightZeroOut();

protected:
    int64_t sortNum_;
    const MoeV3Arch35GatherOutComputeTilingData *gatherOutTilingData_;
    int64_t blockIdx_;
    int64_t needCoreNum_;
    int64_t coreIndicesElements_;
    int64_t perCoreIndicesElements_;
    int64_t k_;
    int64_t n_;
    int64_t cols_;
    int64_t activeNum_;
    int64_t dropPadMode_ = 0;
    int64_t expertCapacity_ = 0;
    int64_t outputRows_ = 0;
    int64_t expertNum_;
    int64_t expertStart_ = 0;
    int64_t expertEnd_ = 0;
    int64_t bufferNum_ = 1;
    int64_t kvFactor_ = 2;
    int64_t totalLength_;
    int64_t tileLength_;
    int64_t expertTokensNumType_ = 0;
    int64_t expertTokensNumFlag_ = 0;
    int64_t rowIdxType_ = 0;
    int64_t actualExpertNum_ = 0;
    int64_t actualExpertIdxNum_ = 0;
    int64_t expertCountElements_ = 0;
    int64_t actualExpertTotalNum_ = 0;
    int64_t curIndexStart_;
    int64_t startXRow_;
    int64_t endXRow_;
    int64_t isInputScale_ = 0;
    int64_t epFullload_ = 0;
    int64_t isInputTopkWeight_ = 0;

    static constexpr int64_t DST_BLK_STRIDE = 1;
    static constexpr int64_t DST_REP_STRIDE = 8;
    static constexpr int64_t MASK_STRIDE = 64;

    TQue<QuePosition::VECOUT, 1> sortDataCopyInQueue_;
    TQue<QuePosition::VECOUT, 1> sortedExpertIdxQueue_;
    TQue<QuePosition::VECOUT, 1> sortedRowIdxQueue_;
    TQue<QuePosition::VECOUT, 1> expandedRowIdxQueue_;
    TQue<QuePosition::VECOUT, 1> expertTokensCopyOutQueue_;

    TBuf<TPosition::VECCALC> tempBuffer_;
    TBuf<TPosition::VECCALC> sortedBuffer_;
    TBuf<TPosition::VECCALC> expertCountBuf_;

    GlobalTensor<int32_t> expertIdxGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<int64_t> expertTokensCountOrCumsumGm_;
    GlobalTensor<int32_t> expertTotalCountGm_;

    GlobalTensor<float> topkWeightGm_;
    GlobalTensor<float> expandedTopkWeightGm_;

    TPipe *pipe_;
};

template <typename T>
__aicore__ inline void MoeV3FullLoadBase<T>::Init(GM_ADDR expertIdx, GM_ADDR expandedRowIdx,
                                                  GM_ADDR expertTokensCountOrCumsum, GM_ADDR topkWeight,
                                                  GM_ADDR expandedTopkWeight, GM_ADDR workspace,
                                                  const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe)
{
    this->gatherOutTilingData_ = &(tilingData->gatherOutComputeParamsOp);
    this->blockIdx_ = GetBlockIdx();
    this->n_ = tilingData->n;
    this->k_ = tilingData->k;
    this->cols_ = tilingData->cols;
    this->expertStart_ = tilingData->expertStart;
    this->expertEnd_ = tilingData->expertEnd;
    this->needCoreNum_ = this->gatherOutTilingData_->needCoreNum;

    this->perCoreIndicesElements_ = this->gatherOutTilingData_->perCoreIndicesElements;
    this->activeNum_ = tilingData->activeNum;
    this->dropPadMode_ = tilingData->dropPadMode;
    this->expertCapacity_ = tilingData->expertCapacity;
    this->outputRows_ =
        this->dropPadMode_ == DROP_PAD_MODE ? tilingData->expertNum * tilingData->expertCapacity : this->activeNum_;
    if (this->blockIdx_ == this->gatherOutTilingData_->needCoreNum - 1) {
        this->coreIndicesElements_ = this->gatherOutTilingData_->lastCoreIndicesElements;
    } else {
        this->coreIndicesElements_ = this->gatherOutTilingData_->perCoreIndicesElements;
    }
    this->expertTokensNumType_ = tilingData->expertTokensNumType;
    this->expertTokensNumFlag_ = tilingData->expertTokensNumFlag;
    this->expertNum_ = tilingData->expertNum;
    this->totalLength_ = tilingData->n * tilingData->k;
    this->isInputScale_ = tilingData->isInputScale;
    this->tileLength_ = Align(tilingData->vbsComputeParamsOp.lastCorePerLoopElements, sizeof(int32_t));
    this->sortNum_ = Ceil(this->tileLength_, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
    this->rowIdxType_ = tilingData->rowIdxType;
    this->actualExpertNum_ = tilingData->actualExpertNum;
    this->epFullload_ = tilingData->epFullload;
    this->pipe_ = tPipe;
    isInputTopkWeight_ = tilingData->isInputTopkWeight;

    if (expertTokensNumType_ == EXERPT_TOKENS_KEY_VALUE) {
        expertCountElements_ = ((actualExpertNum_ + 1) < expertNum_) ?
                                   (actualExpertNum_ + 1) * EXERPT_TOKENS_KEY_VALUE :
                                   expertNum_ * EXERPT_TOKENS_KEY_VALUE;
    } else {
        expertCountElements_ = actualExpertNum_;
    }

    int64_t buffSize = this->sortNum_ * sizeof(int32_t);

    curIndexStart_ = this->blockIdx_ * this->perCoreIndicesElements_;
    startXRow_ = curIndexStart_ / this->k_;
    endXRow_ = (curIndexStart_ + this->coreIndicesElements_ - 1) / this->k_;

    InitGlobalBuffers(expertIdx, expandedRowIdx, expertTokensCountOrCumsum, topkWeight, expandedTopkWeight, workspace);
    InitQueBuffers();
}

template <typename T>
__aicore__ inline void MoeV3FullLoadBase<T>::InitGlobalBuffers(GM_ADDR expertIdx, GM_ADDR expandedRowIdx,
                                                               GM_ADDR expertTokensCountOrCumsum, GM_ADDR topkWeight,
                                                               GM_ADDR expandedTopkWeight, GM_ADDR workspace)
{
    expertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expertIdx, this->tileLength_);
    expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx, this->tileLength_);

    if (this->expertTokensNumFlag_ > 0) {
        expertTokensCountOrCumsumGm_.SetGlobalBuffer((__gm__ int64_t *)expertTokensCountOrCumsum);
    }

    expertTotalCountGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + Align(this->totalLength_, sizeof(int32_t)) * 2 +
                                            Align(this->actualExpertNum_, sizeof(int32_t)),
                                        1);

    if (isInputTopkWeight_ == 1) {
        topkWeightGm_.SetGlobalBuffer((__gm__ float *)topkWeight, totalLength_);
        expandedTopkWeightGm_.SetGlobalBuffer((__gm__ float *)expandedTopkWeight,
                                              (dropPadMode_ == DROP_PAD_MODE) ? outputRows_ : totalLength_);
    }
}

template <typename T>
__aicore__ inline void MoeV3FullLoadBase<T>::InitQueBuffers()
{
    int64_t buffSize = this->sortNum_ * sizeof(int32_t);
    pipe_->InitBuffer(sortDataCopyInQueue_, bufferNum_, buffSize * kvFactor_);
    pipe_->InitBuffer(sortedExpertIdxQueue_, bufferNum_, buffSize);
    pipe_->InitBuffer(sortedRowIdxQueue_, bufferNum_, buffSize);
    pipe_->InitBuffer(expandedRowIdxQueue_, bufferNum_, buffSize);
    pipe_->InitBuffer(expertTokensCopyOutQueue_, bufferNum_, AlignBytes(expertCountElements_, sizeof(int64_t)));
    pipe_->InitBuffer(tempBuffer_, buffSize * kvFactor_);
    pipe_->InitBuffer(sortedBuffer_, buffSize * kvFactor_);
    pipe_->InitBuffer(expertCountBuf_, AlignBytes(actualExpertNum_, sizeof(int32_t)));
}

template <typename T>
__aicore__ inline void MoeV3FullLoadBase<T>::CopyIn()
{
    LocalTensor<int32_t> inLocal = sortDataCopyInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(totalLength_ * sizeof(int32_t)), 0,
                                     0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(inLocal[0], expertIdxGm_, dataCopyParams, dataCopyPadParams);
    ArithProgression<int32_t>(inLocal[this->sortNum_], 0, 1, this->sortNum_);
    sortDataCopyInQueue_.EnQue(inLocal);
}

template <typename T>
__aicore__ inline void MoeV3FullLoadBase<T>::SortCompute()
{
    LocalTensor<int32_t> inLocal = sortDataCopyInQueue_.DeQue<int32_t>();
    LocalTensor<int32_t> expertIdxLocal = inLocal[0];
    LocalTensor<float> expertIdxLocalFp32 = expertIdxLocal.ReinterpretCast<float>();
    FilterExpertIdx(expertIdxLocal, expertIdxLocalFp32);

    int64_t duplicateNum = totalLength_ % ONE_REPEAT_SORT_NUM;
    if (duplicateNum > 0) {
        int duplicateIndex = totalLength_ - duplicateNum;
        uint64_t mask0 = (UINT64_MAX << duplicateNum) & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
        uint64_t mask[2] = {mask0, 0};
        Duplicate(expertIdxLocalFp32[duplicateIndex], MIN_FP32, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);
    }

    LocalTensor<float> concatLocal;
    LocalTensor<float> tempTensor = tempBuffer_.Get<float>(GetSortLen<float>(this->sortNum_));
    Concat(concatLocal, expertIdxLocalFp32, tempTensor, this->sortNum_ / ONE_REPEAT_SORT_NUM);

    LocalTensor<uint32_t> rowIdxLocal = inLocal[this->sortNum_].template ReinterpretCast<uint32_t>();
    LocalTensor<float> sortedLocal = sortedBuffer_.Get<float>(GetSortLen<float>(this->sortNum_));
    Sort<float, true>(sortedLocal, concatLocal, rowIdxLocal, tempTensor, this->sortNum_ / ONE_REPEAT_SORT_NUM);

    LocalTensor<float> sortedExpertForSourceRowLocal = sortedExpertIdxQueue_.AllocTensor<float>();
    LocalTensor<uint32_t> expandDstToSrcRowLocal = sortedRowIdxQueue_.AllocTensor<uint32_t>();
    Extract(sortedExpertForSourceRowLocal, expandDstToSrcRowLocal, sortedLocal, this->sortNum_ / ONE_REPEAT_SORT_NUM);
    Muls(sortedExpertForSourceRowLocal, sortedExpertForSourceRowLocal, static_cast<float>(-1), this->tileLength_);

    LocalTensor<int32_t> sortedExpertIdxLocal = sortedExpertForSourceRowLocal.ReinterpretCast<int32_t>();
    Cast(sortedExpertIdxLocal, sortedExpertForSourceRowLocal, RoundMode::CAST_ROUND, this->tileLength_);
    PipeBarrier<PIPE_V>();

    LocalTensor<int32_t> expertCountLocal = expertCountBuf_.Get<int32_t>(actualExpertNum_);
    Duplicate(expertCountLocal, static_cast<int32_t>(0), static_cast<int32_t>(1));
    __ubuf__ int32_t *sortedExpertIdxLocalAddr = (__ubuf__ int32_t *)sortedExpertIdxLocal.GetPhyAddr();
    __ubuf__ int32_t *actualIdxNumAddr = (__ubuf__ int32_t *)expertCountLocal.GetPhyAddr();
    asc_vf_call<FullLoadComputeActualIdxNumSimt>(dim3{SIMT_THREAD_NUM, 1, 1}, static_cast<int32_t>(totalLength_),
                                                 static_cast<int32_t>(expertStart_), static_cast<int32_t>(expertEnd_),
                                                 sortedExpertIdxLocalAddr, actualIdxNumAddr);
    actualExpertIdxNum_ = static_cast<int64_t>(expertCountLocal.GetValue(0));

    sortedExpertIdxQueue_.EnQue<int32_t>(sortedExpertIdxLocal);
    sortedRowIdxQueue_.EnQue<uint32_t>(expandDstToSrcRowLocal);

    if (this->dropPadMode_ == DROP_PAD_MODE) {
        ComputeGatherIdxWithCapacity();
        sortDataCopyInQueue_.FreeTensor(inLocal);
        return;
    }

    if (actualExpertIdxNum_ < 1) {
        sortDataCopyInQueue_.FreeTensor(inLocal);
        return;
    }

    ComputeGatherIdx(inLocal);

    sortDataCopyInQueue_.FreeTensor(inLocal);
}

template <typename T>
__aicore__ inline void MoeV3FullLoadBase<T>::FilterExpertIdx(LocalTensor<int32_t> &expertIdxLocal,
                                                             LocalTensor<float> &expertIdxLocalFp32)
{
    Cast(expertIdxLocalFp32, expertIdxLocal, RoundMode::CAST_ROUND, this->tileLength_);

    uint16_t repeatTimes = Ceil(this->tileLength_, FLOAT_REG_TENSOR_LENGTH);
    uint32_t sreg = static_cast<uint32_t>(this->tileLength_);
    __ubuf__ float *inUbAddr = (__ubuf__ float *)expertIdxLocalFp32.GetPhyAddr();
    float cmpScalar = static_cast<float>(expertStart_);
    float negOne = static_cast<float>(-1);

    __VEC_SCOPE__
    {
        MicroAPI::MaskReg maskRegLoop, cmpMaskReg;
        MicroAPI::MaskReg pregMain = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

        MicroAPI::RegTensor<float> inRegToFloat, infFloat, vDstReg0;
        Duplicate(infFloat, static_cast<float>(MIN_FP32), pregMain);

        for (uint16_t i = 0; i < repeatTimes; i++) {
            maskRegLoop = MicroAPI::UpdateMask<float>(sreg);
            MicroAPI::LoadAlign(inRegToFloat, inUbAddr + i * FLOAT_REG_TENSOR_LENGTH);
            MicroAPI::Compares<float, CMPMODE::LT>(cmpMaskReg, inRegToFloat, cmpScalar, maskRegLoop);
            MicroAPI::Muls(inRegToFloat, inRegToFloat, negOne, maskRegLoop);
            MicroAPI::Select(vDstReg0, infFloat, inRegToFloat, cmpMaskReg);
            MicroAPI::StoreAlign(inUbAddr + i * FLOAT_REG_TENSOR_LENGTH, vDstReg0, maskRegLoop);
        }
    }
}

template <typename T>
__aicore__ inline void MoeV3FullLoadBase<T>::ComputeGatherIdx(LocalTensor<int32_t> &inLocal)
{
    LocalTensor<uint32_t> expandDstToSrcRowLocal = sortedRowIdxQueue_.DeQue<uint32_t>();
    LocalTensor<float> expandDstToSrcRowLocalFp32 = expandDstToSrcRowLocal.ReinterpretCast<float>();

    Cast(expandDstToSrcRowLocalFp32, expandDstToSrcRowLocal.ReinterpretCast<int32_t>(), RoundMode::CAST_ROUND,
         totalLength_);
    PipeBarrier<PIPE_V>();
    Muls(expandDstToSrcRowLocalFp32, expandDstToSrcRowLocalFp32, static_cast<float>(-1), totalLength_);
    PipeBarrier<PIPE_V>();

    ArithProgression<int32_t>(inLocal[this->sortNum_], 0, 1, totalLength_);
    PipeBarrier<PIPE_V>();

    int64_t duplicateNum = totalLength_ % ONE_REPEAT_SORT_NUM;
    if (duplicateNum > 0) {
        int duplicateIndex = totalLength_ - duplicateNum;
        uint64_t mask0 = (UINT64_MAX << duplicateNum) & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
        uint64_t mask[2] = {mask0, 0};
        Duplicate(expandDstToSrcRowLocalFp32[duplicateIndex], MIN_FP32, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);
        PipeBarrier<PIPE_V>();
    }

    LocalTensor<float> concatLocal2 = inLocal[0].ReinterpretCast<float>();
    LocalTensor<float> tempTensor = tempBuffer_.Get<float>(GetSortLen<float>(this->sortNum_));
    Concat(concatLocal2, expandDstToSrcRowLocalFp32, tempTensor, this->sortNum_ / ONE_REPEAT_SORT_NUM);
    PipeBarrier<PIPE_V>();

    LocalTensor<uint32_t> rowIdxLocal = inLocal[this->sortNum_].template ReinterpretCast<uint32_t>();
    LocalTensor<float> sortedLocal = sortedBuffer_.Get<float>(GetSortLen<float>(this->sortNum_));
    Sort<float, true>(sortedLocal, concatLocal2, rowIdxLocal, tempTensor, this->sortNum_ / ONE_REPEAT_SORT_NUM);
    PipeBarrier<PIPE_V>();

    LocalTensor<uint32_t> expandedRowIdxLocal = expandedRowIdxQueue_.AllocTensor<uint32_t>();
    Extract(tempTensor, expandedRowIdxLocal, sortedLocal, this->sortNum_ / ONE_REPEAT_SORT_NUM);
    PipeBarrier<PIPE_V>();

    Muls(expandDstToSrcRowLocalFp32, expandDstToSrcRowLocalFp32, static_cast<float>(-1), totalLength_);
    PipeBarrier<PIPE_V>();
    Cast(expandDstToSrcRowLocal.ReinterpretCast<int32_t>(), expandDstToSrcRowLocalFp32, RoundMode::CAST_RINT,
         totalLength_);
    PipeBarrier<PIPE_V>();

    expandedRowIdxQueue_.EnQue<uint32_t>(expandedRowIdxLocal);
    sortedRowIdxQueue_.EnQue<uint32_t>(expandDstToSrcRowLocal);
}

template <typename T>
__aicore__ inline void MoeV3FullLoadBase<T>::ComputeGatherIdxWithCapacity()
{
    LocalTensor<int32_t> sortedRowIdx = sortedRowIdxQueue_.DeQue<int32_t>();
    LocalTensor<int32_t> sortedExpertIdx = sortedExpertIdxQueue_.DeQue<int32_t>();
    LocalTensor<int32_t> expandedRowIdx = expandedRowIdxQueue_.AllocTensor<int32_t>();
    Duplicate(expandedRowIdx, static_cast<int32_t>(-1), static_cast<int32_t>(this->totalLength_));
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

    int32_t lastExpertId = -1;
    int64_t tokenCount = 0;
    for (int64_t i = 0; i < this->actualExpertIdxNum_; i++) {
        int32_t curExpertId = sortedExpertIdx.GetValue(i);
        if (curExpertId < this->expertStart_) {
            continue;
        }
        if (curExpertId >= this->expertEnd_) {
            break;
        }
        if (curExpertId != lastExpertId) {
            lastExpertId = curExpertId;
            tokenCount = 0;
        }
        if (tokenCount < this->expertCapacity_) {
            int32_t srcIndex = sortedRowIdx.GetValue(i);
            int32_t dstIndex = static_cast<int32_t>(curExpertId * this->expertCapacity_ + tokenCount);
            expandedRowIdx.SetValue(srcIndex, dstIndex);
        }
        tokenCount++;
    }

    expandedRowIdxQueue_.EnQue<int32_t>(expandedRowIdx);
    sortedRowIdxQueue_.EnQue<int32_t>(sortedRowIdx);
    sortedExpertIdxQueue_.EnQue<int32_t>(sortedExpertIdx);
}

template <typename T>
__aicore__ inline void MoeV3FullLoadBase<T>::CopyOutRowIdx()
{
    LocalTensor<int32_t> sortedRowIdx = sortedRowIdxQueue_.DeQue<int32_t>();
    LocalTensor<int32_t> expandedExpertIdx = sortedExpertIdxQueue_.DeQue<int32_t>();
    if (this->rowIdxType_ == SCATTER) {
        DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(totalLength_ * sizeof(int32_t)), 0,
                                     0, 0};
        DataCopyPad(expandedRowIdxGm_, sortedRowIdx, copyParams);
    } else {
        if (this->epFullload_) {
            LocalTensor<int32_t> expandedRowIdx = expandedRowIdxQueue_.AllocTensor<int32_t>();
            Duplicate(expandedRowIdx, static_cast<int32_t>(-1), static_cast<int32_t>(totalLength_));
            SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
            for (int64_t i = 0; i < actualExpertIdxNum_; i++) {
                int32_t curExpertId = expandedExpertIdx.GetValue(i);
                if (curExpertId < expertStart_ || curExpertId >= expertEnd_) {
                    break;
                }
                int64_t outIndices = sortedRowIdx.GetValue(i);
                expandedRowIdx.SetValue(outIndices, i);
            }
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopyExtParams copyParams{static_cast<uint16_t>(1),
                                         static_cast<uint32_t>(totalLength_ * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(expandedRowIdxGm_, expandedRowIdx, copyParams);
            expandedRowIdxQueue_.FreeTensor(expandedRowIdx);
        } else {
            LocalTensor<int32_t> expandedRowIdx = expandedRowIdxQueue_.DeQue<int32_t>();
            for (int64_t i = actualExpertIdxNum_; i < totalLength_; i++) {
                int32_t invalidRouteIdx = sortedRowIdx.GetValue(i);
                expandedRowIdx.SetValue(invalidRouteIdx, -1);
            }
            DataCopyExtParams copyParams{static_cast<uint16_t>(1),
                                         static_cast<uint32_t>(totalLength_ * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(expandedRowIdxGm_, expandedRowIdx, copyParams);
            expandedRowIdxQueue_.EnQue<int32_t>(expandedRowIdx);
        }
    }
    sortedRowIdxQueue_.EnQue<int32_t>(sortedRowIdx);
    sortedExpertIdxQueue_.EnQue<int32_t>(expandedExpertIdx);
}

template <typename T>
__aicore__ inline void MoeV3FullLoadBase<T>::ComputeExpertTokenCount()
{
    LocalTensor<int32_t> sortedExpertIdx = sortedExpertIdxQueue_.DeQue<int32_t>();
    LocalTensor<int32_t> expertCountLocal = expertCountBuf_.Get<int32_t>(actualExpertNum_);
    Duplicate(expertCountLocal, static_cast<int32_t>(0), actualExpertNum_);

    __ubuf__ int32_t *sortedExpertIdxLocalAddr = (__ubuf__ int32_t *)sortedExpertIdx.GetPhyAddr();
    __ubuf__ int32_t *expertCountOutLocalAddr = (__ubuf__ int32_t *)expertCountLocal.GetPhyAddr();

    asc_vf_call<FullLoadComputeExpertFirstIndexSimt>(
        dim3{SIMT_THREAD_NUM, 1, 1}, static_cast<int32_t>(totalLength_), static_cast<int32_t>(expertStart_),
        static_cast<int32_t>(expertEnd_), sortedExpertIdxLocalAddr, expertCountOutLocalAddr);
    asc_vf_call<FullLoadComputeExpertCountOutSimt>(
        dim3{SIMT_THREAD_NUM, 1, 1}, static_cast<int32_t>(totalLength_), static_cast<int32_t>(expertStart_),
        static_cast<int32_t>(expertEnd_), sortedExpertIdxLocalAddr, expertCountOutLocalAddr, expertCountOutLocalAddr);

    sortedExpertIdxQueue_.EnQue<int32_t>(sortedExpertIdx);
}

template <typename T>
__aicore__ inline void MoeV3FullLoadBase<T>::CopyExpertCountToOutput()
{
    PipeBarrier<PIPE_ALL>();
    LocalTensor<int32_t> expertCountLocal = expertCountBuf_.Get<int32_t>(actualExpertNum_);
    LocalTensor<int64_t> expertTokensOutLocal = expertTokensCopyOutQueue_.AllocTensor<int64_t>();

    if (expertTokensNumType_ == EXERPT_TOKENS_KEY_VALUE) {
        int64_t expertOffset = 0;
        Duplicate(expertTokensOutLocal.ReinterpretCast<int32_t>(), static_cast<int32_t>(0),
                  static_cast<int32_t>(expertCountElements_ * KEY_VALUE_MODE));
        event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIDVToS);
        WaitFlag<HardEvent::V_S>(eventIDVToS);
        for (int64_t i = 0; i < actualExpertNum_; i++) {
            int64_t expertCount = static_cast<int64_t>(expertCountLocal.GetValue(i));
            if (expertCount != 0) {
                expertTokensOutLocal.SetValue(expertOffset * EXERPT_TOKENS_KEY_VALUE, i + expertStart_);
                expertTokensOutLocal.SetValue(expertOffset * EXERPT_TOKENS_KEY_VALUE + 1, expertCount);
                expertOffset++;
                actualExpertTotalNum_ += expertCount;
            }
        }
        expertCountElements_ = Min(expertCountElements_, static_cast<int64_t>((expertOffset + 1) * KEY_VALUE_MODE));
    } else if (expertTokensNumType_ == EXERPT_TOKENS_COUNT) {
        for (int64_t i = 0; i < actualExpertNum_; i++) {
            int64_t expertCount = static_cast<int64_t>(expertCountLocal.GetValue(i));
            expertTokensOutLocal.SetValue(i, expertCount);
            actualExpertTotalNum_ += expertCount;
        }
    } else {
        int64_t cumsumCount = 0;
        for (int64_t i = 0; i < actualExpertNum_; i++) {
            int64_t expertCount = static_cast<int64_t>(expertCountLocal.GetValue(i));
            cumsumCount += expertCount;
            expertTokensOutLocal.SetValue(i, cumsumCount);
            actualExpertTotalNum_ += expertCount;
        }
    }
    event_t eventIDSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIDSToMte3);
    WaitFlag<HardEvent::S_MTE3>(eventIDSToMte3);
    DataCopyExtParams copyParams{static_cast<uint16_t>(1),
                                 static_cast<uint32_t>(expertCountElements_ * sizeof(int64_t)), 0, 0, 0};
    DataCopyPad(expertTokensCountOrCumsumGm_, expertTokensOutLocal, copyParams);
    expertTokensCopyOutQueue_.FreeTensor(expertTokensOutLocal);

    expertTotalCountGm_.SetValue(0, static_cast<int32_t>(actualExpertTotalNum_));
    AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        expertTotalCountGm_);
}

template <typename T>
__aicore__ inline void MoeV3FullLoadBase<T>::FreeLocalTensor()
{
    LocalTensor<int32_t> sortedExpertIdx = sortedExpertIdxQueue_.DeQue<int32_t>();
    LocalTensor<uint32_t> sortedRowIdx = sortedRowIdxQueue_.DeQue<uint32_t>();
    sortedExpertIdxQueue_.FreeTensor(sortedExpertIdx);
    sortedRowIdxQueue_.FreeTensor(sortedRowIdx);

    LocalTensor<int32_t> expandedRowIdx = expandedRowIdxQueue_.DeQue<int32_t>();
    expandedRowIdxQueue_.FreeTensor(expandedRowIdx);
}

template <typename T>
__aicore__ inline void MoeV3FullLoadBase<T>::TopkWeightScatterOut()
{
    LocalTensor<int32_t> sortedRowIdx = sortedRowIdxQueue_.DeQue<int32_t>();
    LocalTensor<int32_t> sortedExpertIdx = sortedExpertIdxQueue_.DeQue<int32_t>();

    int64_t startRowIdx = blockIdx_ * perCoreIndicesElements_;
    int64_t endRowIdx = Min(startRowIdx + coreIndicesElements_, activeNum_);
    int64_t elements = endRowIdx - startRowIdx;
    if (elements > 0) {
        uint32_t threadNum = static_cast<uint32_t>(Min(elements, SIMT_THREAD_NUM));
        asc_vf_call<FullLoadTopkWeightGatherSimt>(
            dim3{threadNum, 1, 1}, elements, startRowIdx, totalLength_, expertStart_, expertEnd_,
            (__ubuf__ int32_t *)sortedRowIdx.GetPhyAddr(), (__ubuf__ int32_t *)sortedExpertIdx.GetPhyAddr(),
            (__gm__ float *)topkWeightGm_.GetPhyAddr(), (__gm__ volatile float *)expandedTopkWeightGm_.GetPhyAddr());
    }

    sortedRowIdxQueue_.EnQue<int32_t>(sortedRowIdx);
    sortedExpertIdxQueue_.EnQue<int32_t>(sortedExpertIdx);
}

template <typename T>
__aicore__ inline void MoeV3FullLoadBase<T>::TopkWeightGatherOut()
{
    LocalTensor<int32_t> expandedRowIdx = expandedRowIdxQueue_.DeQue<int32_t>();

    int64_t startRowIdx = blockIdx_ * perCoreIndicesElements_;
    int64_t endRowIdx = startRowIdx + coreIndicesElements_;
    int64_t outputRows = (dropPadMode_ == DROP_PAD_MODE) ? outputRows_ : Min(actualExpertIdxNum_, activeNum_);
    int64_t elements = endRowIdx - startRowIdx;
    if (elements > 0) {
        uint32_t threadNum = static_cast<uint32_t>(Min(elements, SIMT_THREAD_NUM));
        asc_vf_call<FullLoadTopkWeightScatterSimt>(
            dim3{threadNum, 1, 1}, elements, startRowIdx, outputRows, (__ubuf__ int32_t *)expandedRowIdx.GetPhyAddr(),
            (__gm__ float *)topkWeightGm_.GetPhyAddr(), (__gm__ volatile float *)expandedTopkWeightGm_.GetPhyAddr());
    }

    expandedRowIdxQueue_.EnQue<int32_t>(expandedRowIdx);
}

template <typename T>
__aicore__ inline void MoeV3FullLoadBase<T>::TopkWeightZeroOut()
{
    int64_t perCoreRows = Ceil(outputRows_, needCoreNum_);
    int64_t startRow = blockIdx_ * perCoreRows;
    int64_t endRow = Min(startRow + perCoreRows, outputRows_);
    if (startRow >= endRow) {
        return;
    }
    int64_t rowCount = endRow - startRow;
    GlobalTensor<float> topkWeightZeroGm;
    topkWeightZeroGm.SetGlobalBuffer((__gm__ float *)expandedTopkWeightGm_.GetPhyAddr() + startRow, rowCount);
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    InitGlobalMemory(topkWeightZeroGm, rowCount, 0.0f);
    SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
}

} // namespace MoeInitRoutingV3
#endif // MOE_V3_FULL_LOAD_BASE_H
