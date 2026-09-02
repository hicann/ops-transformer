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
 * \file moe_init_routing_v2_grad_base.h
 * \brief
 */
#ifndef MOE_INIT_ROUTING_V2_GRAD_BASE_H
#define MOE_INIT_ROUTING_V2_GRAD_BASE_H

#include "kernel_operator.h"

namespace MoeInitRoutingV2Grad {
using namespace AscendC;

constexpr int64_t ALIGN_SIZE = 32;

template <typename T>
class MoeInitRoutingV2GradBase {
public:
    __aicore__ inline MoeInitRoutingV2GradBase(){};
    __aicore__ inline void ParseTilingData(const MoeInitRoutingV2GradTilingData *tilingData);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void CopyIn(int64_t xRow, LocalTensor<T> &xLocal, int64_t colOffset, int64_t cpyCols);
    __aicore__ inline void CopyInWithCastFloat(int64_t xRow, LocalTensor<float> &xLocal, LocalTensor<T> &xLocalT,
                                               int64_t colOffset, int64_t cpyCols);
    __aicore__ inline void GradAdd(LocalTensor<float> &dstLocal, LocalTensor<float> srcLocal, int64_t cpyCols);
    __aicore__ inline void BinaryAddWithMovIn(int64_t xRow, LocalTensor<float> xLocal, LocalTensor<float> inLocal,
                                              LocalTensor<T> inLocalT, int64_t colOffset, int64_t cpyCols);
    __aicore__ inline void CopyOut(LocalTensor<float> xLocal, LocalTensor<T> tmpLocalT, int64_t cpyCols,
                                   int64_t outOffset);

protected:
    const MoeV2GradComputeTilingData *gradTilingData;

    TBuf<QuePosition::VECIN> binBuff;
    TBuf<QuePosition::VECIN> tmpBuff;

    GlobalTensor<T> gradExpandedXGm;
    GlobalTensor<T> gradXGm;

    int64_t needCoreNum;
    int64_t blockIdx;
    int64_t cols;
    int64_t n;
    int64_t k;
    int64_t e;
    int64_t c;
    int64_t activeNum;
    int64_t cpyLoops;
    int64_t perCpyCols;
    int64_t tailCpyCols;
    int64_t perCoreElements;
    int64_t coreElements;
    int64_t tokensLoop;
    int64_t tokensFormer;
    int64_t lastTokensFormer;

    int64_t perBuffSize;
    int64_t binBufferNum;
    int64_t tmpBufferNum;
    int64_t expNum;
    int64_t cpyOffset = 0;

    bool multiTmpBuff = false;

    int64_t baseStride;
};

template <typename T>
__aicore__ inline void MoeInitRoutingV2GradBase<T>::ParseTilingData(const MoeInitRoutingV2GradTilingData *tilingData)
{
    this->gradTilingData = &(tilingData->MoeV2GradComputeParamsOp);

    this->needCoreNum = this->gradTilingData->needCoreNum;
    this->cols = tilingData->cols;
    this->n = tilingData->n;
    this->k = tilingData->k;
    this->e = tilingData->e;
    this->c = tilingData->c;
    this->activeNum = tilingData->activeNum;

    this->baseStride = 2; // 2: 间隔搬运

    this->cpyLoops = this->gradTilingData->elementCopyLoops;
    this->perCpyCols = this->gradTilingData->elementPerCopyCols;
    this->tailCpyCols = this->gradTilingData->elementLastCopyCols;

    this->blockIdx = GetBlockIdx();
    this->perCoreElements = this->gradTilingData->perCoreElements;
    this->tokensFormer = this->gradTilingData->tokensFormer;
    if (this->blockIdx == this->gradTilingData->needCoreNum - 1) {
        this->coreElements = this->gradTilingData->lastCoreElements;
        this->tokensLoop = this->gradTilingData->lastCoreTokensLoop;
        this->lastTokensFormer = this->gradTilingData->lastCoreTailTokensFormer;
    } else {
        this->coreElements = this->gradTilingData->perCoreElements;
        this->tokensLoop = this->gradTilingData->perCoreTokensLoop;
        this->lastTokensFormer = this->gradTilingData->perCoreTailTokensFormer;
    }

    this->perBuffSize = this->gradTilingData->copyBufferSize;
    this->binBufferNum = this->gradTilingData->binaryAddBufferNum;
    this->tmpBufferNum = this->gradTilingData->tmpBufferNum;
    this->expNum = this->gradTilingData->exponentOfBinary;
    if constexpr (!IsSameType<T, float>::value) {
        this->cpyOffset = this->perBuffSize / 2; // 2: 从buff的后半部分开始拷贝，为了能在同一块空间里面做cast
    }

    this->multiTmpBuff = (this->tmpBufferNum == this->binBufferNum) ? true : false;
}

template <typename T>
__aicore__ inline void MoeInitRoutingV2GradBase<T>::CopyIn(int64_t xRow, LocalTensor<T> &xLocal, int64_t colOffset,
                                                           int64_t cpyCols)
{
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(cpyCols * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<T>(0)};
    DataCopyPad(xLocal, gradExpandedXGm[xRow * this->cols + colOffset], dataCopyParams, dataCopyPadParams);
}

template <typename T>
__aicore__ inline void MoeInitRoutingV2GradBase<T>::CopyInWithCastFloat(int64_t xRow, LocalTensor<float> &xLocal,
                                                                        LocalTensor<T> &xLocalT, int64_t colOffset,
                                                                        int64_t cpyCols)
{
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(cpyCols * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<T>(0)};
    DataCopyPad(xLocalT, gradExpandedXGm[xRow * this->cols + colOffset], dataCopyParams, dataCopyPadParams);

    if constexpr (!IsSameType<T, float>::value) {
        event_t eventMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMte2V);
        WaitFlag<HardEvent::MTE2_V>(eventMte2V);
        Cast(xLocal, xLocalT, RoundMode::CAST_NONE, cpyCols);
    }
}

template <typename T>
__aicore__ inline void MoeInitRoutingV2GradBase<T>::GradAdd(LocalTensor<float> &dstLocal, LocalTensor<float> srcLocal,
                                                            int64_t cpyCols)
{
    Add(dstLocal, dstLocal, srcLocal, cpyCols);
}

template <typename T>
__aicore__ inline void MoeInitRoutingV2GradBase<T>::BinaryAddWithMovIn(int64_t xRow, LocalTensor<float> xLocal,
                                                                       LocalTensor<float> inLocal,
                                                                       LocalTensor<T> inLocalT, int64_t colOffset,
                                                                       int64_t cpyCols)
{
    event_t eventMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    event_t eventVMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));

    SetFlag<HardEvent::V_MTE2>(eventVMte2);
    WaitFlag<HardEvent::V_MTE2>(eventVMte2);

    CopyIn(xRow, inLocalT, colOffset, cpyCols);

    SetFlag<HardEvent::MTE2_V>(eventMte2V);
    WaitFlag<HardEvent::MTE2_V>(eventMte2V);

    if constexpr (!IsSameType<T, float>::value) {
        Cast(inLocal, inLocalT, RoundMode::CAST_NONE, cpyCols);
        PipeBarrier<PIPE_V>();
    }

    GradAdd(xLocal, inLocal, cpyCols);
}

template <typename T>
__aicore__ inline void MoeInitRoutingV2GradBase<T>::CopyOut(LocalTensor<float> xLocal, LocalTensor<T> tmpLocalT,
                                                            int64_t cpyCols, int64_t outOffset)
{
    event_t eventVMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    event_t eventMte3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    event_t eventMte3S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    event_t eventMte3Mte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));

    if constexpr (!IsSameType<T, float>::value) {
        PipeBarrier<PIPE_V>();
        Cast(tmpLocalT, xLocal, RoundMode::CAST_RINT, cpyCols);
    }
    SetFlag<HardEvent::V_MTE3>(eventVMte3);
    WaitFlag<HardEvent::V_MTE3>(eventVMte3);

    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(cpyCols * sizeof(T)), 0, 0, 0};

    if constexpr (IsSameType<T, float>::value) {
        DataCopyPad(gradXGm[outOffset], xLocal, dataCopyParams);
    } else {
        DataCopyPad(gradXGm[outOffset], tmpLocalT, dataCopyParams);
    }
    SetFlag<HardEvent::MTE3_MTE2>(eventMte3Mte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventMte3Mte2);

    SetFlag<HardEvent::MTE3_V>(eventMte3V);
    WaitFlag<HardEvent::MTE3_V>(eventMte3V);

    SetFlag<HardEvent::MTE3_S>(eventMte3S);
    WaitFlag<HardEvent::MTE3_S>(eventMte3S);
}

} // namespace MoeInitRoutingV2Grad
#endif // MOE_INIT_ROUTING_V2_GRAD_BASE_H
