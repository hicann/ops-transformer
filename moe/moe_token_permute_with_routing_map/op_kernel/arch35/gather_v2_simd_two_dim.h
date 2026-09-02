/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file gather_v2_simd_tow_dim.h
 * \brief
 */
#ifndef GATHER_V2_SIMD_TWO_DIM_H
#define GATHER_V2_SIMD_TWO_DIM_H

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"

namespace gatherv2 {
using namespace AscendC;

constexpr int32_t BUFFER_NUM_SIMD_TWO_DIM = 2;

template <typename INDICES_T>
class Gatherv2SimdTwoDim {
public:
    __aicore__ inline Gatherv2SimdTwoDim(TPipe *pipe)
        : pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR y,
                                const MoeTokenPermuteWithRoutingMapTilingData *tilingData);
    __aicore__ inline void Process();
    template <typename T>
    __aicore__ inline void CopyIn(LocalTensor<T> xLocal, GlobalTensor<T> xGm, int64_t offset, uint32_t nBurst,
                                  uint32_t copyLen);
    __aicore__ inline INDICES_T GetIndex(int64_t idx, int64_t endIdx);
    __aicore__ inline void CopyOut(int64_t offset, uint32_t nBurst, uint32_t copyLen);
    __aicore__ inline void CalculateCoreRange(int64_t &yStart, int64_t &yEnd);
    __aicore__ inline void NoSplitColProcess(int64_t colsAlign);
    __aicore__ inline void SplitColProcess(int64_t colsAlign);

private:
    GlobalTensor<int8_t> xGm_;
    GlobalTensor<INDICES_T> indicesGm_;
    GlobalTensor<int8_t> yGm_;
    TPipe *pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM_SIMD_TWO_DIM> inQueue_;
    TBuf<QuePosition::VECCALC> indexBuf_;
    const IndexCopyComputeRMTilingData *indexCopyTilingData;

    int64_t indicesOffsetBase_ = 0;
    int64_t curIndexSize_ = 0;
};

template <typename INDICES_T>
__aicore__ inline void Gatherv2SimdTwoDim<INDICES_T>::Init(GM_ADDR x, GM_ADDR indices, GM_ADDR y,
                                                           const MoeTokenPermuteWithRoutingMapTilingData *tilingData)
{
    this->indexCopyTilingData = &(tilingData->indexCopyComputeParamsOp);

    xGm_.SetGlobalBuffer((__gm__ int8_t *)x);

    indicesGm_.SetGlobalBuffer((__gm__ INDICES_T *)indices);
    yGm_.SetGlobalBuffer((__gm__ int8_t *)y);
    pipe_->InitBuffer(inQueue_, BUFFER_NUM_SIMD_TWO_DIM,
                      indexCopyTilingData->tokenUB * indexCopyTilingData->oneTokenBtypeSize);
    pipe_->InitBuffer(indexBuf_, indexCopyTilingData->onceIndices * sizeof(INDICES_T));
}

template <typename INDICES_T>
template <typename T>
__aicore__ inline void Gatherv2SimdTwoDim<INDICES_T>::CopyIn(LocalTensor<T> xLocal, GlobalTensor<T> xGm, int64_t offset,
                                                             uint32_t nBurst, uint32_t copyLen)
{
    DataCopyPadExtParams<T> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;

    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = nBurst;
    dataCoptExtParams.blockLen = copyLen * sizeof(T);
    dataCoptExtParams.srcStride = 0;
    dataCoptExtParams.dstStride = 0;
    DataCopyPad(xLocal, xGm[offset], dataCoptExtParams, dataCopyPadExtParams);
}

template <typename INDICES_T>
__aicore__ inline void Gatherv2SimdTwoDim<INDICES_T>::CopyOut(int64_t offset, uint32_t nBurst, uint32_t copyLen)
{
    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = nBurst;
    dataCoptExtParams.blockLen = copyLen;
    dataCoptExtParams.srcStride = 0;
    dataCoptExtParams.dstStride = 0;
    LocalTensor<int8_t> yLocal = inQueue_.DeQue<int8_t>();
    event_t eventIdVtoMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVtoMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVtoMTE3);
    DataCopyPad(yGm_[offset], yLocal, dataCoptExtParams);
    inQueue_.FreeTensor(yLocal);
}

template <typename INDICES_T>
__aicore__ inline INDICES_T Gatherv2SimdTwoDim<INDICES_T>::GetIndex(int64_t idx, int64_t endIdx)
{
    if (idx >= indicesOffsetBase_ + curIndexSize_) {
        int64_t copyLen =
            (endIdx - idx) > indexCopyTilingData->onceIndices ? indexCopyTilingData->onceIndices : endIdx - idx;
        indicesOffsetBase_ = idx;
        curIndexSize_ = copyLen;
        LocalTensor<INDICES_T> tmpLocal = indexBuf_.Get<INDICES_T>();
        CopyIn(tmpLocal, indicesGm_, indicesOffsetBase_, 1, copyLen);

        event_t eventIdMTE2toS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIdMTE2toS);
        WaitFlag<HardEvent::MTE2_S>(eventIdMTE2toS);
    }
    LocalTensor<INDICES_T> indexLocal = indexBuf_.Get<INDICES_T>();
    return indexLocal.GetValue(idx - indicesOffsetBase_);
}

template <typename INDICES_T>
__aicore__ inline void Gatherv2SimdTwoDim<INDICES_T>::CalculateCoreRange(int64_t &yStart, int64_t &yEnd)
{
    if (GetBlockIdx() < indexCopyTilingData->tailCoreNum) {
        yStart = (indexCopyTilingData->coreCalcNum + 1) * GetBlockIdx();
        yEnd = yStart + indexCopyTilingData->coreCalcNum + 1;
    } else {
        yStart = indexCopyTilingData->coreCalcNum * GetBlockIdx() + indexCopyTilingData->tailCoreNum;
        yEnd = yStart + indexCopyTilingData->coreCalcNum;
    }
}

template <typename INDICES_T>
__aicore__ inline void Gatherv2SimdTwoDim<INDICES_T>::NoSplitColProcess(int64_t colsAlign)
{
    int64_t yStart = 0, yEnd = 0;
    CalculateCoreRange(yStart, yEnd);
    int64_t currentCoreElements = yEnd - yStart;
    int64_t onceMaxRows = indexCopyTilingData->tokenUB * indexCopyTilingData->oneTokenBtypeSize / colsAlign;
    int64_t loopNum = (currentCoreElements + onceMaxRows - 1) / onceMaxRows;

    int64_t indiceEndIdx = yEnd;
    for (int64_t i = 0; i < loopNum; i++) {
        int64_t rows = (i == loopNum - 1) ? (currentCoreElements - i * onceMaxRows) : onceMaxRows;
        int64_t cols = indexCopyTilingData->onceUbTokenNums;
        LocalTensor<int8_t> xLocal = inQueue_.AllocTensor<int8_t>();
        event_t eventIdMTE3toV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventIdMTE3toV);
        WaitFlag<HardEvent::MTE3_V>(eventIdMTE3toV);
        int64_t preIdx0 = -1, preIdx1 = -1, preIdx2 = -1;
        int32_t backOffset0 = 0, backOffset1 = 0, backOffset2 = 0;
        for (int64_t j = 0; j < rows; j++) {
            int64_t yIdx = yStart + i * onceMaxRows + j;
            INDICES_T index = GetIndex(yIdx, indiceEndIdx);
            int64_t xIndex = index * indexCopyTilingData->onceUbTokenNums;
            int64_t offset = xIndex * indexCopyTilingData->oneTokenBtypeSize;
            if (unlikely(xIndex == preIdx0 || xIndex == preIdx1 || xIndex == preIdx2)) {
                event_t eventIdMTE2toV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(eventIdMTE2toV);
                WaitFlag<HardEvent::MTE2_V>(eventIdMTE2toV);
                int32_t backStep = (xIndex == preIdx0) ? backOffset0 : (xIndex == preIdx1) ? backOffset1 : backOffset2;
                Copy(xLocal[j * colsAlign], xLocal[backStep],
                     indexCopyTilingData->onceUbTokenNums * indexCopyTilingData->oneTokenBtypeSize);
            } else {
                CopyIn(xLocal[j * colsAlign], xGm_, offset, 1,
                       indexCopyTilingData->onceUbTokenNums * indexCopyTilingData->oneTokenBtypeSize);
                preIdx2 = preIdx1;
                preIdx1 = preIdx0;
                preIdx0 = xIndex;
                backOffset2 = backOffset1;
                backOffset1 = backOffset0;
                backOffset0 = j * colsAlign;
            }
        }
        inQueue_.EnQue<int8_t>(xLocal);
        int64_t yOffset =
            (yStart + i * onceMaxRows) * indexCopyTilingData->onceUbTokenNums * indexCopyTilingData->oneTokenBtypeSize;
        CopyOut(yOffset, rows, cols * indexCopyTilingData->oneTokenBtypeSize);
    }
}

template <typename INDICES_T>
__aicore__ inline void Gatherv2SimdTwoDim<INDICES_T>::SplitColProcess(int64_t colsAlign)
{
    // indice -> g
    // output -> g, i
    int64_t yStart = 0, yEnd = 0;
    CalculateCoreRange(yStart, yEnd);
    int64_t indiceEndIdx = yEnd;
    int64_t loopSize =
        (indexCopyTilingData->onceUbTokenNums + indexCopyTilingData->tokenUB - 1) / indexCopyTilingData->tokenUB;
    for (int64_t i = yStart; i < yEnd; i++) {
        INDICES_T index = GetIndex(i, indiceEndIdx);
        int64_t xIndex = index * indexCopyTilingData->onceUbTokenNums;
        int64_t yIndex = i * indexCopyTilingData->onceUbTokenNums;
        for (int64_t j = 0; j < loopSize; j++) {
            int64_t cols = (j == loopSize - 1) ?
                               (indexCopyTilingData->onceUbTokenNums - j * indexCopyTilingData->tokenUB) :
                               indexCopyTilingData->tokenUB;
            LocalTensor<int8_t> xLocal = inQueue_.AllocTensor<int8_t>();
            int64_t offset = (xIndex + j * indexCopyTilingData->tokenUB) * indexCopyTilingData->oneTokenBtypeSize;
            CopyIn(xLocal[0], xGm_, offset, 1, cols * indexCopyTilingData->oneTokenBtypeSize);
            inQueue_.EnQue<int8_t>(xLocal);
            int64_t yOffset = (yIndex + j * indexCopyTilingData->tokenUB) * indexCopyTilingData->oneTokenBtypeSize;
            CopyOut(yOffset, 1, cols * indexCopyTilingData->oneTokenBtypeSize);
        }
    }
}

template <typename INDICES_T>
__aicore__ inline void Gatherv2SimdTwoDim<INDICES_T>::Process()
{
    if (static_cast<int32_t>(GetBlockIdx()) >= this->indexCopyTilingData->needCoreNum) {
        return;
    }
    int64_t ubBlockSize = static_cast<int64_t>(Ops::Base::GetUbBlockSize());
    int64_t colsAlign =
        (indexCopyTilingData->onceUbTokenNums * indexCopyTilingData->oneTokenBtypeSize + ubBlockSize - 1) /
        ubBlockSize * ubBlockSize;

    if (colsAlign <= indexCopyTilingData->tokenUB * indexCopyTilingData->oneTokenBtypeSize) {
        NoSplitColProcess(colsAlign);
    } else {
        SplitColProcess(colsAlign);
    }
}
} // namespace gatherv2
#endif // GATHER_V2_SIMD_TWO_DIM_H
