/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_index_copy_with_ep.h
 * \brief
 */
#ifndef MOE_INDEX_COPY_WITH_EP_H
#define MOE_INDEX_COPY_WITH_EP_H

#include "moe_token_permute_with_ep_common.h"

namespace MoeTokenPermuteWithEp {
using namespace AscendC;

template <typename T, typename probsT, bool ifNumOutTokens>
class MoeindexCopyOp {
public:
    __aicore__ inline MoeindexCopyOp(){};
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR srcProbs, GM_ADDR indices, GM_ADDR dst, GM_ADDR dstProbs,
                                const MoeTokenPermuteWithEpTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t Offset, DataCopyExtParams &dataCopyExtParams,
                                  DataCopyPadExtParams<T> &dataCopyPadExtParams);
    __aicore__ inline void CopyOut(int64_t innerLoop, int64_t outTokenNum, DataCopyExtParams &dataCopyExtParams,
                                   DataCopyPadExtParams<T> &dataCopyPadExtParams);
    __aicore__ inline void CopyInIndices(int64_t progress, DataCopyExtParams &dataCopyExtParams,
                                         DataCopyPadExtParams<int32_t> &dataCopyPadExtParams);
    __aicore__ inline void CopyInProbs(int64_t Offset, DataCopyExtParams &dataCopyExtParams,
                                       DataCopyPadExtParams<probsT> &dataCopyPadExtParams);
    __aicore__ inline void CopyOutProbs(int64_t innerLoop, int64_t outTokenNum, DataCopyExtParams &dataCopyExtParams,
                                        DataCopyPadExtParams<probsT> &dataCopyPadExtParams);
    __aicore__ inline void SyncAll();

private:
    TPipe *pipe;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> copyInQueue;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> copyProbsInQueue;
    TQue<QuePosition::VECOUT, 1> indicesBuffer;
    TQue<QuePosition::VECOUT, 1> probsTmpBuffer;

    GlobalTensor<T> srcGm;
    GlobalTensor<T> dstGm;

    GlobalTensor<probsT> srcProbsGm;
    GlobalTensor<probsT> dstProbsGm;
    LocalTensor<probsT> probsLocal;
    LocalTensor<probsT> probsTmpLocal;

    GlobalTensor<int32_t> indicesGm;
    LocalTensor<int32_t> indicesLocal;

    const IndexMixCopyComputeTilingData *indexCopyTilingData;

    event_t indicesMte2ToS;
    event_t indicesSToMte2;
    event_t probsMte3ToS;
    event_t probsSToMte3;
    int64_t cols;
    int64_t colsAlign;
    int64_t topK;
    int64_t needCoreNum;
    int64_t frontCoreNum;
    int64_t tailCoreNum;
    int64_t coreCalcNum;
    int64_t oneTokenBtypeSize;
    int64_t oneProbBtypeSize;
    int64_t onceIndicesTokenMoveTimes;
    int64_t onceUbTokenNums;
    int64_t onceIndicesTokenNums;
    int64_t onceIndices;
    int64_t oneTokenlastMove;
    int64_t oneTokenOnceMove;
    int64_t oneTokenMoveTimes;
    int64_t CoreLoop;
    int64_t CoreLastTokenNums;

    int64_t LastonceIndicesTokenMoveTimes;
    int64_t LastIndicesLastTokenNums;
    int64_t numOutTokens;
    int64_t start;
    int64_t end;

    int64_t onceIndicesTokenOffset;
    int64_t onceUbTokenCols;
    int64_t onceUbIndicesCols;
    int64_t onceUbProbsCols;
    int64_t coreNum;
    int64_t blockIdx;
    int64_t totalLength;

    int64_t onceIndicesNums;
    int64_t onceProbsNums;
    DataCopyExtParams indicesCopyParams;
    DataCopyExtParams indicesCopyLastParams;
    DataCopyExtParams tokenCopyParams;
    DataCopyExtParams tokenCopyLastParams;
    DataCopyExtParams tokenCopyOutParams;
    DataCopyExtParams probsCopyParams;
    DataCopyExtParams probsCopyLastParams;
    DataCopyExtParams probsCopyOutParams;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPadExtParams<int32_t> indicesPadParams{false, 0, 0, 0};
    DataCopyPadExtParams<probsT> probsPadParams{false, 0, 0, 0};
};

template <typename T, typename probsT, bool ifNumOutTokens>
__aicore__ inline void MoeindexCopyOp<T, probsT, ifNumOutTokens>::CopyIn(int64_t Offset,
                                                                         DataCopyExtParams &dataCopyExtParams,
                                                                         DataCopyPadExtParams<T> &dataCopyPadExtParams)
{
    LocalTensor<T> inLocal = copyInQueue.AllocTensor<T>();
    DataCopyPad(inLocal, srcGm[Offset], dataCopyExtParams, dataCopyPadExtParams);
    copyInQueue.EnQue<T>(inLocal);
}

template <typename T, typename probsT, bool ifNumOutTokens>
__aicore__ inline void MoeindexCopyOp<T, probsT, ifNumOutTokens>::CopyInProbs(
    int64_t progress, DataCopyExtParams &dataCopyExtParams, DataCopyPadExtParams<probsT> &dataCopyPadExtParams)
{
    probsLocal = copyProbsInQueue.AllocTensor<probsT>();
    if (srcProbsGm.GetPhyAddr() == nullptr) {
        return;
    }
    DataCopyPad(probsLocal, srcProbsGm[progress * onceProbsNums], dataCopyExtParams, dataCopyPadExtParams);
    copyProbsInQueue.EnQue<probsT>(probsLocal);
    copyProbsInQueue.DeQue<probsT>();
}

template <typename T, typename probsT, bool ifNumOutTokens>
__aicore__ inline void MoeindexCopyOp<T, probsT, ifNumOutTokens>::CopyInIndices(
    int64_t progress, DataCopyExtParams &dataCopyExtParams, DataCopyPadExtParams<int32_t> &dataCopyPadExtParams)
{
    indicesLocal = indicesBuffer.AllocTensor<int32_t>();
    DataCopyPad(indicesLocal, indicesGm[progress * onceIndicesNums], dataCopyExtParams, dataCopyPadExtParams);
}

template <typename T, typename probsT, bool ifNumOutTokens>
__aicore__ inline void MoeindexCopyOp<T, probsT, ifNumOutTokens>::CopyOut(int64_t innerLoop, int64_t outTokenNum,
                                                                          DataCopyExtParams &dataCopyExtParams,
                                                                          DataCopyPadExtParams<T> &dataCopyPadExtParams)
{
    LocalTensor<T> inLocal = copyInQueue.DeQue<T>();
    int64_t offset = innerLoop * onceUbIndicesCols;
    for (int64_t tokensId = 0; tokensId < outTokenNum; tokensId++) {
        for (int64_t topKId = 0; topKId < topK; topKId++) {
            if constexpr (ifNumOutTokens == true) {
                auto indicesValue = indicesLocal.GetValue(offset);
                if ((indicesValue >= start) && (indicesValue < start + numOutTokens)) {
                    DataCopyPad(dstGm[(indicesValue - start) * cols], inLocal[tokensId * colsAlign], dataCopyExtParams);
                }
            } else {
                DataCopyPad(dstGm[indicesLocal.GetValue(offset) * cols], inLocal[tokensId * colsAlign],
                            dataCopyExtParams);
            }
            offset++;
        }
    }
    copyInQueue.FreeTensor(inLocal);
}

template <typename T, typename probsT, bool ifNumOutTokens>
__aicore__ inline void MoeindexCopyOp<T, probsT, ifNumOutTokens>::CopyOutProbs(
    int64_t innerLoop, int64_t outTokenNum, DataCopyExtParams &dataCopyExtParams,
    DataCopyPadExtParams<probsT> &dataCopyPadExtParams)
{
    if (srcProbsGm.GetPhyAddr() == nullptr) {
        return;
    }
    probsTmpLocal = probsTmpBuffer.AllocTensor<probsT>();
    int64_t offset = innerLoop * onceUbIndicesCols;
    for (int64_t tokensId = 0; tokensId < outTokenNum; tokensId++) {
        for (int64_t topKId = 0; topKId < topK; topKId++) {
            auto indicesValue = indicesLocal.GetValue(offset);
            if constexpr (ifNumOutTokens == true) {
                if ((indicesValue >= start) && (indicesValue < start + numOutTokens)) {
                    SetFlag<HardEvent::MTE3_S>(probsMte3ToS);
                    WaitFlag<HardEvent::MTE3_S>(probsMte3ToS);
                    probsTmpLocal.SetValue(0, probsLocal.GetValue(offset));
                    SetFlag<HardEvent::S_MTE3>(probsSToMte3);
                    WaitFlag<HardEvent::S_MTE3>(probsSToMte3);
                    DataCopyPad(dstProbsGm[indicesValue - start], probsTmpLocal, dataCopyExtParams);
                }
            } else {
                SetFlag<HardEvent::MTE3_S>(probsMte3ToS);
                WaitFlag<HardEvent::MTE3_S>(probsMte3ToS);
                probsTmpLocal.SetValue(0, probsLocal.GetValue(offset));
                SetFlag<HardEvent::S_MTE3>(probsSToMte3);
                WaitFlag<HardEvent::S_MTE3>(probsSToMte3);
                DataCopyPad(dstProbsGm[indicesValue], probsTmpLocal, dataCopyExtParams);
            }
            offset++;
        }
    }
    probsTmpBuffer.FreeTensor(probsTmpLocal);
}

template <typename T, typename probsT, bool ifNumOutTokens>
__aicore__ inline void MoeindexCopyOp<T, probsT, ifNumOutTokens>::SyncAll()
{
    if (coreNum == 1) {
        return;
    }
    AscendC::SyncAll();
}

template <typename T, typename probsT, bool ifNumOutTokens>
__aicore__ inline void MoeindexCopyOp<T, probsT, ifNumOutTokens>::Init(
    GM_ADDR src, GM_ADDR srcProbs, GM_ADDR indices, GM_ADDR dst, GM_ADDR dstProbs,
    const MoeTokenPermuteWithEpTilingData *tilingData, TPipe *tPipe)
{
    int64_t blockNum = GetBlockNum();
    this->pipe = tPipe;
    this->blockIdx = GetBlockIdx();
    this->cols = tilingData->cols;
    this->colsAlign = tilingData->colsAlign;

    this->topK = tilingData->topK;

    this->indexCopyTilingData = &(tilingData->indexCopyComputeParamsOp);
    this->coreNum = this->indexCopyTilingData->needCoreNum;

    this->totalLength = this->indexCopyTilingData->numOutTokens * tilingData->cols;
    this->indexCopyTilingData = &(tilingData->indexCopyComputeParamsOp);
    this->needCoreNum = this->indexCopyTilingData->needCoreNum;
    this->frontCoreNum = this->indexCopyTilingData->frontCoreNum;
    this->tailCoreNum = this->indexCopyTilingData->tailCoreNum;
    this->coreCalcNum = this->indexCopyTilingData->coreCalcNum;
    this->oneTokenBtypeSize = this->indexCopyTilingData->oneTokenBtypeSize;
    this->oneProbBtypeSize = this->indexCopyTilingData->oneProbBtypeSize;
    this->onceIndicesTokenMoveTimes = this->indexCopyTilingData->onceIndicesTokenMoveTimes;
    this->onceUbTokenNums = this->indexCopyTilingData->onceUbTokenNums;
    this->onceIndicesTokenNums = this->indexCopyTilingData->onceIndicesTokenNums;
    this->onceIndices = this->indexCopyTilingData->onceIndices;
    this->oneTokenlastMove = this->indexCopyTilingData->oneTokenlastMove;
    this->oneTokenOnceMove = this->indexCopyTilingData->oneTokenOnceMove;
    this->oneTokenMoveTimes = this->indexCopyTilingData->oneTokenMoveTimes;
    this->numOutTokens = this->indexCopyTilingData->numOutTokens;
    this->start = this->indexCopyTilingData->start;
    this->end = this->indexCopyTilingData->end;
    int64_t offset;
    if (this->blockIdx > this->frontCoreNum - 1) {
        this->coreCalcNum = this->indexCopyTilingData->coreCalcTail;
        this->LastonceIndicesTokenMoveTimes = this->indexCopyTilingData->tailLastonceIndicesTokenMoveTimes;
        this->LastIndicesLastTokenNums = this->indexCopyTilingData->tailLastIndicesLastTokenNums;
        this->CoreLoop = this->indexCopyTilingData->tailCoreLoop;
        this->CoreLastTokenNums = this->indexCopyTilingData->tailCoreLastTokenNums;
        offset = this->indexCopyTilingData->coreCalcNum * this->frontCoreNum +
                 (this->blockIdx - this->frontCoreNum) * this->indexCopyTilingData->coreCalcTail;
    } else {
        this->coreCalcNum = this->indexCopyTilingData->coreCalcNum;
        this->LastonceIndicesTokenMoveTimes = this->indexCopyTilingData->frontLastonceIndicesTokenMoveTimes;
        this->LastIndicesLastTokenNums = this->indexCopyTilingData->frontLastIndicesLastTokenNums;
        this->CoreLoop = this->indexCopyTilingData->frontCoreLoop;
        this->CoreLastTokenNums = this->indexCopyTilingData->frontCoreLastTokenNums;
        offset = this->coreCalcNum * this->blockIdx;
    }
    int64_t srcoffset = offset * cols;
    int64_t indicesoffset = offset * topK;
    int64_t probsoffset = indicesoffset;
    onceUbTokenCols = this->onceUbTokenNums * cols;
    onceUbIndicesCols = this->onceUbTokenNums * topK;
    onceUbProbsCols = onceUbIndicesCols;

    srcGm.SetGlobalBuffer((__gm__ T *)src + srcoffset, this->coreCalcNum * cols);
    indicesGm.SetGlobalBuffer((__gm__ int32_t *)indices + indicesoffset, this->coreCalcNum * topK);
    dstGm.SetGlobalBuffer((__gm__ T *)dst, this->numOutTokens * cols);
    if (srcProbs != nullptr) {
        srcProbsGm.SetGlobalBuffer((__gm__ probsT *)srcProbs + probsoffset, this->coreCalcNum * topK);
    } else {
        srcProbsGm.SetGlobalBuffer(nullptr);
    }
    dstProbsGm.SetGlobalBuffer((__gm__ probsT *)dstProbs, this->numOutTokens);
    pipe->InitBuffer(copyInQueue, 2, this->indexCopyTilingData->tokenUB);
    pipe->InitBuffer(indicesBuffer, 1, this->indexCopyTilingData->indicesUB);
    pipe->InitBuffer(copyProbsInQueue, 1, this->indexCopyTilingData->probsUB);
    pipe->InitBuffer(probsTmpBuffer, 1, 32);
    onceIndicesNums = onceIndicesTokenNums * topK;
    onceProbsNums = onceIndicesNums;
    indicesCopyParams = {(uint16_t)1, (uint32_t)(onceIndicesNums * sizeof(int32_t)), 0, 0, 0};
    indicesCopyLastParams = {(uint16_t)1, (uint32_t)(CoreLastTokenNums * topK * sizeof(int32_t)), 0, 0, 0};
    probsCopyParams = {(uint16_t)1, (uint32_t)(onceProbsNums * oneProbBtypeSize), 0, 0, 0};
    probsCopyLastParams = {(uint16_t)1, (uint32_t)(CoreLastTokenNums * topK * oneProbBtypeSize), 0, 0, 0};
    probsCopyOutParams = {(uint16_t)1, (uint32_t)(oneProbBtypeSize), 0, 0, 0};
    tokenCopyParams = {(uint16_t)onceUbTokenNums, (uint32_t)(oneTokenBtypeSize), 0, 0, 0};
    tokenCopyLastParams = {(uint16_t)LastIndicesLastTokenNums, (uint32_t)(oneTokenBtypeSize), 0, 0, 0};
    tokenCopyOutParams = {(uint16_t)1, (uint32_t)(oneTokenBtypeSize), 0, 0, 0};
    indicesMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    indicesSToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
    probsMte3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    probsSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
}

template <typename T, typename probsT, bool ifNumOutTokens>
__aicore__ inline void MoeindexCopyOp<T, probsT, ifNumOutTokens>::Process()
{
    int64_t outoffset = 0;
    int64_t outProbsOffset = 0;
    onceIndicesTokenOffset = onceIndicesTokenNums * cols;
    if (this->blockIdx < this->indexCopyTilingData->needCoreNum) {
        for (int64_t outLoop = 0; outLoop < CoreLoop - 1; outLoop++) {
            CopyInIndices(outLoop, indicesCopyParams, indicesPadParams);
            CopyInProbs(outLoop, probsCopyParams, probsPadParams);
            SetFlag<HardEvent::MTE2_S>(indicesMte2ToS);
            WaitFlag<HardEvent::MTE2_S>(indicesMte2ToS);
            for (int64_t innerLoop = 0; innerLoop < onceIndicesTokenMoveTimes; innerLoop++) {
                CopyIn(outoffset, tokenCopyParams, padParams);
                CopyOut(innerLoop, onceUbTokenNums, tokenCopyOutParams, padParams);
                CopyOutProbs(innerLoop, onceUbTokenNums, probsCopyOutParams, probsPadParams);
                outoffset += onceUbTokenCols;
                outProbsOffset += onceUbProbsCols;
            }
            copyProbsInQueue.FreeTensor(probsLocal);
            indicesBuffer.FreeTensor(indicesLocal);
            SetFlag<HardEvent::S_MTE2>(indicesSToMte2);
            WaitFlag<HardEvent::S_MTE2>(indicesSToMte2);
        }
        CopyInIndices(CoreLoop - 1, indicesCopyLastParams, indicesPadParams);
        CopyInProbs(CoreLoop - 1, probsCopyLastParams, probsPadParams);
        SetFlag<HardEvent::MTE2_S>(indicesMte2ToS);
        WaitFlag<HardEvent::MTE2_S>(indicesMte2ToS);
        for (int64_t innerLoop = 0; innerLoop < LastonceIndicesTokenMoveTimes - 1; innerLoop++) {
            CopyIn(outoffset, tokenCopyParams, padParams);
            CopyOut(innerLoop, onceUbTokenNums, tokenCopyOutParams, padParams);
            CopyOutProbs(innerLoop, onceUbTokenNums, probsCopyOutParams, probsPadParams);
            outoffset += onceUbTokenCols;
            outProbsOffset += onceUbProbsCols;
        }
        CopyIn(outoffset, tokenCopyLastParams, padParams);
        CopyOut(LastonceIndicesTokenMoveTimes - 1, LastIndicesLastTokenNums, tokenCopyOutParams, padParams);
        CopyOutProbs(LastonceIndicesTokenMoveTimes - 1, LastIndicesLastTokenNums, probsCopyOutParams, probsPadParams);
        copyProbsInQueue.FreeTensor(probsLocal);
        indicesBuffer.FreeTensor(indicesLocal);
    }
}

} // namespace MoeTokenPermuteWithEp
#endif // MOE_INDEX_COPY_WITH_EP_H
