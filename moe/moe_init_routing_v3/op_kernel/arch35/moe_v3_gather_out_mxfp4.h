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
 * \file moe_v3_gather_out_mxfp4.h
 * \brief
 */
#ifndef MOE_V3_GATHER_OUT_MXFP4_H_REGBASE
#define MOE_V3_GATHER_OUT_MXFP4_H_REGBASE

#include "moe_v3_common.h"
#include "kernel_operator.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

// 1Byte存储2个fp4数据，索引需要除以2
constexpr int64_t NUM_TWO = 2LL;

template <typename T>
class MoeV3GatherOutMxFp4 {
public:
    __aicore__ inline MoeV3GatherOutMxFp4(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR workspace, GM_ADDR expandedRowIdx, GM_ADDR expandedX,
                                GM_ADDR expandedScale, const MoeInitRoutingV3Arch35TilingData *tilingData,
                                TPipe *tPipe);
    __aicore__ inline void Process();
    __aicore__ inline void CopyExpertIn(int64_t curExpertLoopOffset, int64_t curLoopElements);
    __aicore__ inline void CopyXIn(int64_t xSrcOffset, int64_t curLoopCols);
    __aicore__ inline void CopyXOut(int64_t xDstOffset, int64_t curLoopCols);
    __aicore__ inline void CopyScaleIn(int64_t scaleSrcOffset, int64_t curLoopCols);
    __aicore__ inline void CopyScaleOut(int64_t scaleDstOffset, int64_t curLoopCols);

private:
    __aicore__ inline void InitBasicParams(GM_ADDR workspace, const MoeInitRoutingV3Arch35TilingData *tilingData,
                                           TPipe *tPipe);

    TPipe *pipe_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, GATHER_OUT_BUFFER_NUM> xCopyInQueue_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, GATHER_OUT_BUFFER_NUM> scaleCopyInQueue_;
    TQue<QuePosition::VECIN, GATHER_OUT_BUFFER_NUM> expandedRowIdxCopyInQueue_;

    GlobalTensor<uint8_t> xUint8tGm_;
    GlobalTensor<uint8_t> xGscaleGm_;
    GlobalTensor<int32_t> sortedExpertIdxGm_;
    GlobalTensor<uint8_t> expandedXGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<uint8_t> expandedScaleGm_;
    GlobalTensor<int32_t> expertTotalCountGm_;

    int64_t blockIdx_;
    int64_t cols_;
    int64_t n_;
    int64_t k_;

    int64_t colsLoops_;
    int64_t perLoopCols_;
    int64_t lastLoopCols_;

    int64_t indicesLoops_;

    int64_t perCoreIndicesElements_;
    int64_t lastCoreIndicesElements_;
    int64_t perCorePerLoopIndicesElements_;
    int64_t lastCorePerLoopIndicesElements_;
    int64_t curCorePerLoopIndicesElements_;
    int64_t curCoreLastLoopIndicesElements_;
    int64_t needCoreNum_;
    int64_t curCoreIndicesElements_;

    int64_t actualExpertNum_;
    int64_t expertTotalCount_;

    int64_t rowIdxType_ = 0;
    int64_t isInputScale_ = 0;
};

template <typename T>
__aicore__ inline void MoeV3GatherOutMxFp4<T>::InitBasicParams(GM_ADDR workspace,
                                                               const MoeInitRoutingV3Arch35TilingData *tilingData,
                                                               TPipe *tPipe)
{
    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();

    cols_ = tilingData->cols;
    n_ = tilingData->n;
    k_ = tilingData->k;

    isInputScale_ = tilingData->isInputScale;
    rowIdxType_ = tilingData->rowIdxType;

    colsLoops_ = tilingData->gatherOutComputeParamsOp.colsLoops;
    perLoopCols_ = tilingData->gatherOutComputeParamsOp.perLoopCols;
    lastLoopCols_ = tilingData->gatherOutComputeParamsOp.lastLoopCols;

    actualExpertNum_ = tilingData->actualExpertNum;
    expertTotalCountGm_.SetGlobalBuffer(
        (__gm__ int32_t *)workspace + Align(n_ * k_, sizeof(int32_t)) * 2 + Align(actualExpertNum_, sizeof(int32_t)),
        1);
    AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        expertTotalCountGm_);
    expertTotalCount_ = expertTotalCountGm_.GetValue(0);

    perCorePerLoopIndicesElements_ = tilingData->gatherOutComputeParamsOp.perCorePerLoopIndicesElements;
    lastCorePerLoopIndicesElements_ = tilingData->gatherOutComputeParamsOp.lastCorePerLoopIndicesElements;

    perCoreIndicesElements_ = Ceil(expertTotalCount_, tilingData->coreNum);
    needCoreNum_ = Ceil(expertTotalCount_, perCoreIndicesElements_);
    lastCoreIndicesElements_ = expertTotalCount_ - (needCoreNum_ - 1) * perCoreIndicesElements_;

    if (blockIdx_ == needCoreNum_ - 1) {
        curCoreIndicesElements_ = lastCoreIndicesElements_;
        curCorePerLoopIndicesElements_ = Min(lastCorePerLoopIndicesElements_, curCoreIndicesElements_);
    } else {
        curCoreIndicesElements_ = perCoreIndicesElements_;
        curCorePerLoopIndicesElements_ = Min(perCorePerLoopIndicesElements_, curCoreIndicesElements_);
    }
    indicesLoops_ = Ceil(curCoreIndicesElements_, curCorePerLoopIndicesElements_);
    curCoreLastLoopIndicesElements_ = curCoreIndicesElements_ - (indicesLoops_ - 1) * curCorePerLoopIndicesElements_;
}

template <typename T>
__aicore__ inline void MoeV3GatherOutMxFp4<T>::Init(GM_ADDR x, GM_ADDR scale, GM_ADDR workspace, GM_ADDR expandedRowIdx,
                                                    GM_ADDR expandedX, GM_ADDR expandedScale,
                                                    const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe)
{
    InitBasicParams(workspace, tilingData, tPipe);
    xUint8tGm_.SetGlobalBuffer((__gm__ uint8_t *)x, n_ * cols_ / NUM_TWO);
    xGscaleGm_.SetGlobalBuffer((__gm__ uint8_t *)scale, n_ * cols_ / SCALE_FACTOR_WITH_X);
    expandedXGm_.SetGlobalBuffer((__gm__ uint8_t *)expandedX + blockIdx_ * perCoreIndicesElements_ * cols_ / NUM_TWO,
                                 curCoreIndicesElements_ * cols_ / NUM_TWO);
    expandedScaleGm_.SetGlobalBuffer(
        (__gm__ uint8_t *)expandedScale + blockIdx_ * perCoreIndicesElements_ * cols_ / SCALE_FACTOR_WITH_X,
        curCoreIndicesElements_ * cols_ / SCALE_FACTOR_WITH_X);

    pipe_->InitBuffer(expandedRowIdxCopyInQueue_, GATHER_OUT_BUFFER_NUM,
                      AlignBytes(curCorePerLoopIndicesElements_, sizeof(int32_t)));
    pipe_->InitBuffer(xCopyInQueue_, GATHER_OUT_BUFFER_NUM, AlignBytes(perLoopCols_, sizeof(uint8_t)));
    pipe_->InitBuffer(scaleCopyInQueue_, GATHER_OUT_BUFFER_NUM,
                      AlignBytes(perLoopCols_ / SCALE_FACTOR_WITH_X, sizeof(uint8_t)));

    sortedExpertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + blockIdx_ * perCoreIndicesElements_,
                                       Align(curCoreIndicesElements_, sizeof(int32_t)));

    if (rowIdxType_ == SCATTER) {
        expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx + blockIdx_ * perCoreIndicesElements_,
                                          Align(curCoreIndicesElements_, sizeof(int32_t)));
    } else {
        expandedRowIdxGm_.SetGlobalBuffer(
            (__gm__ int32_t *)workspace + Align(n_ * k_, sizeof(int32_t)) + blockIdx_ * perCoreIndicesElements_,
            Align(curCoreIndicesElements_, sizeof(int32_t)));
    }
}

template <typename T>
__aicore__ inline void MoeV3GatherOutMxFp4<T>::CopyExpertIn(int64_t curExpertLoopOffset, int64_t curLoopElements)
{
    LocalTensor<int32_t> subRowIdxLocal = expandedRowIdxCopyInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLoopElements * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    DataCopyPad(subRowIdxLocal, expandedRowIdxGm_[curExpertLoopOffset], copyParams, padParams);
    expandedRowIdxCopyInQueue_.EnQue(subRowIdxLocal);
}

template <typename T>
__aicore__ inline void MoeV3GatherOutMxFp4<T>::CopyXIn(int64_t xSrcOffset, int64_t curLoopCols)
{
    LocalTensor<uint8_t> xLocal = xCopyInQueue_.AllocTensor<uint8_t>();
    DataCopyExtParams copyParams0{static_cast<uint16_t>(1), static_cast<uint32_t>(curLoopCols * sizeof(uint8_t)), 0, 0,
                                  0};
    DataCopyPadExtParams<uint8_t> padParams0{false, 0, 0, 0};
    DataCopyPad(xLocal, xUint8tGm_[xSrcOffset], copyParams0, padParams0);
    xCopyInQueue_.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void MoeV3GatherOutMxFp4<T>::CopyXOut(int64_t xDstOffset, int64_t curLoopCols)
{
    LocalTensor<uint8_t> xLocal = xCopyInQueue_.DeQue<uint8_t>();
    DataCopyExtParams copyParams2{1, static_cast<uint32_t>(curLoopCols * sizeof(uint8_t)), 0, 0, 0};
    DataCopyPad(expandedXGm_[xDstOffset], xLocal, copyParams2);
    xCopyInQueue_.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void MoeV3GatherOutMxFp4<T>::CopyScaleIn(int64_t scaleSrcOffset, int64_t curLoopCols)
{
    LocalTensor<uint8_t> scaleLocal = scaleCopyInQueue_.AllocTensor<uint8_t>();
    DataCopyExtParams copyParams1{static_cast<uint16_t>(1), static_cast<uint32_t>(curLoopCols * sizeof(uint8_t)), 0, 0,
                                  0};
    DataCopyPadExtParams<uint8_t> padParams1{false, 0, 0, 0};
    DataCopyPad(scaleLocal, xGscaleGm_[scaleSrcOffset], copyParams1, padParams1);
    scaleCopyInQueue_.EnQue(scaleLocal);
}

template <typename T>
__aicore__ inline void MoeV3GatherOutMxFp4<T>::CopyScaleOut(int64_t scaleDstOffset, int64_t curLoopCols)
{
    LocalTensor<uint8_t> scaleLocal = scaleCopyInQueue_.DeQue<uint8_t>();
    DataCopyExtParams copyParams3{1, static_cast<uint32_t>(curLoopCols * sizeof(uint8_t)), 0, 0, 0};
    DataCopyPad(expandedScaleGm_[scaleDstOffset], scaleLocal, copyParams3);
    scaleCopyInQueue_.FreeTensor(scaleLocal);
}

template <typename T>
__aicore__ inline void MoeV3GatherOutMxFp4<T>::Process()
{
    if (blockIdx_ < needCoreNum_) {
        int64_t curLoopElements = curCorePerLoopIndicesElements_;
        for (int64_t indicesLoop = 0; indicesLoop < indicesLoops_; indicesLoop++) {
            if (indicesLoop == indicesLoops_ - 1) {
                curLoopElements = curCoreLastLoopIndicesElements_;
            }
            int64_t curExpertLoopOffset = indicesLoop * curCorePerLoopIndicesElements_;
            SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            CopyExpertIn(curExpertLoopOffset, curLoopElements);

            LocalTensor<int32_t> subRowIdxLocal = expandedRowIdxCopyInQueue_.DeQue<int32_t>();
            SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
            for (int64_t indicesIndex = 0; indicesIndex < curLoopElements; indicesIndex++) {
                int64_t rowIdx = subRowIdxLocal.GetValue(indicesIndex);
                int64_t xSrcOffset = rowIdx / k_ * cols_ / NUM_TWO;
                int64_t scaleSrcOffset = rowIdx / k_ * cols_ / SCALE_FACTOR_WITH_X;
                int64_t scaleDstOffset = (curExpertLoopOffset + indicesIndex) * cols_ / SCALE_FACTOR_WITH_X;
                int64_t xDstOffset = (curExpertLoopOffset + indicesIndex) * cols_ / NUM_TWO;
                SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
                int64_t curLoopCols = perLoopCols_;
                for (int64_t colsLoop = 0; colsLoop < colsLoops_; colsLoop++) {
                    if (colsLoop == colsLoops_ - 1) {
                        curLoopCols = lastLoopCols_;
                    }
                    if (isInputScale_ == 1) {
                        CopyScaleIn(scaleSrcOffset + colsLoop * perLoopCols_ / SCALE_FACTOR_WITH_X,
                                    curLoopCols / SCALE_FACTOR_WITH_X);
                        CopyScaleOut(scaleDstOffset + colsLoop * perLoopCols_ / SCALE_FACTOR_WITH_X,
                                     curLoopCols / SCALE_FACTOR_WITH_X);
                    }
                    int64_t colsLoopOffset = colsLoop * perLoopCols_ / NUM_TWO;
                    CopyXIn(xSrcOffset + colsLoopOffset, curLoopCols / NUM_TWO);
                    CopyXOut(xDstOffset + colsLoopOffset, curLoopCols / NUM_TWO);
                }
            }
            expandedRowIdxCopyInQueue_.FreeTensor(subRowIdxLocal);
        }
    }
}
} // namespace MoeInitRoutingV3
#endif // MOE_V3_GATHER_OUT_MXFP4_H_REGBASE
