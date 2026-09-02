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
 * \file moe_init_routing_v2_grad_with_pos_drop_and_pad_zero.h
 * \brief
 */
#ifndef MOE_INIT_ROUTING_V2_GRAD_POS_PAD0_COMPUTE_H
#define MOE_INIT_ROUTING_V2_GRAD_POS_PAD0_COMPUTE_H

#include "moe_init_routing_v2_grad_base.h"

namespace MoeInitRoutingV2Grad {
using namespace AscendC;

template <typename T>
class MoeInitRoutingV2GradPositionPad0Compute : public MoeInitRoutingV2GradBase<T> {
public:
    __aicore__ inline MoeInitRoutingV2GradPositionPad0Compute(){};
    __aicore__ inline void Init(GM_ADDR gradExpandedX, GM_ADDR expandedRowIdx, GM_ADDR gradX,
                                const MoeInitRoutingV2GradTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void TokenLoopProcess(int64_t elementIdx, int64_t tokenBinBuffSizeOffset,
                                            int64_t tokenTmpBuffSizeOffset);
    __aicore__ inline void GradProcess(int64_t elementIdx, int64_t tokenBinBuffSizeOffset,
                                       int64_t tokenTmpBuffSizeOffset, int64_t cpyCols, int64_t colOffset,
                                       int64_t outOffset);
    __aicore__ inline void GradProcessTokenAccumulate(int64_t tokenIdxStart, int64_t tokenIdxEnd,
                                                      int64_t tokenBinBuffSizeOffset, int64_t tokenTmpBuffSizeOffset,
                                                      int64_t colOffset, int64_t cpyCols);

private:
    TPipe *pipe;

    int64_t perTokenUseBinBuffSize;
    int64_t perTokenUseTmpBuffSize;
    int64_t binBufferSize;
    int64_t tmpBufferSize;

    GM_ADDR expandedRowIdxAddr;
};

template <typename T>
__aicore__ inline void MoeInitRoutingV2GradPositionPad0Compute<T>::Init(
    GM_ADDR gradExpandedX, GM_ADDR expandedRowIdx, GM_ADDR gradX, const MoeInitRoutingV2GradTilingData *tilingData,
    TPipe *tPipe)
{
    this->pipe = tPipe;
    this->ParseTilingData(tilingData);

    this->gradExpandedXGm.SetGlobalBuffer((__gm__ T *)gradExpandedX,
                                          this->e * this->c * this->cols);        // input: {E, C, H}
    this->gradXGm.SetGlobalBuffer((__gm__ T *)gradX, tilingData->n * this->cols); // output: {B*S, H}
    expandedRowIdxAddr = expandedRowIdx;

    this->perTokenUseBinBuffSize = this->binBufferNum * this->perBuffSize;
    this->perTokenUseTmpBuffSize = this->tmpBufferNum * this->perBuffSize;
    this->binBufferSize = this->tokensFormer * this->perTokenUseBinBuffSize;
    this->tmpBufferSize = this->tokensFormer * this->perTokenUseTmpBuffSize;

    pipe->InitBuffer(this->binBuff, this->binBufferSize);
    pipe->InitBuffer(this->tmpBuff, this->tmpBufferSize);
}

template <typename T>
__aicore__ inline void MoeInitRoutingV2GradPositionPad0Compute<T>::GradProcessTokenAccumulate(
    int64_t tokenIdxStart, int64_t tokenIdxEnd, int64_t tokenBinBuffSizeOffset, int64_t tokenTmpBuffSizeOffset,
    int64_t colOffset, int64_t cpyCols)
{
    int64_t tokenIdx = tokenIdxStart;
    for (; tokenIdx < tokenIdxEnd; tokenIdx += this->baseStride) {
        int32_t xRow = ((__gm__ int32_t *)expandedRowIdxAddr)[tokenIdx];
        if (xRow == -1) {
            continue;
        }
        int64_t binIdx = ((tokenIdx - tokenIdxStart) / this->baseStride) % this->binBufferNum;
        int64_t binBuffOffset = tokenBinBuffSizeOffset + this->perBuffSize * binIdx;
        int64_t tmpBuffOffset =
            (this->multiTmpBuff) ? (tokenTmpBuffSizeOffset + this->perBuffSize * binIdx) : tokenTmpBuffSizeOffset;
        LocalTensor<float> xLocal = this->binBuff.template GetWithOffset<float>(this->perCpyCols, binBuffOffset);
        LocalTensor<float> tmpLocal = this->tmpBuff.template GetWithOffset<float>(this->perCpyCols, tmpBuffOffset);
        LocalTensor<T> tmpLocalT =
            this->tmpBuff.template GetWithOffset<T>(this->perCpyCols, tmpBuffOffset + this->cpyOffset);
        this->BinaryAddWithMovIn((int64_t)xRow, xLocal, tmpLocal, tmpLocalT, colOffset, cpyCols);
    }
}

template <typename T>
__aicore__ inline void MoeInitRoutingV2GradPositionPad0Compute<T>::GradProcess(int64_t elementIdx,
                                                                               int64_t tokenBinBuffSizeOffset,
                                                                               int64_t tokenTmpBuffSizeOffset,
                                                                               int64_t cpyCols, int64_t colOffset,
                                                                               int64_t outOffset)
{
    event_t eventMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    event_t eventVMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));

    // S1: 先拷贝一定数量的被加数到buffer中
    int64_t tokenIdxStart = elementIdx * this->k;
    int64_t tokenIdxEnd = tokenIdxStart + this->k;
    int64_t tokenIdx = tokenIdxStart;
    int64_t binIdx = 0;
    for (; tokenIdx < tokenIdxEnd && binIdx < this->binBufferNum; tokenIdx += this->baseStride, binIdx++) {
        int32_t xRow = ((__gm__ int32_t *)expandedRowIdxAddr)[tokenIdx];
        int64_t binBuffOffset = tokenBinBuffSizeOffset + this->perBuffSize * binIdx;
        LocalTensor<float> xLocal = this->binBuff.template GetWithOffset<float>(this->perCpyCols, binBuffOffset);
        LocalTensor<T> xLocalT =
            this->binBuff.template GetWithOffset<T>(this->perCpyCols, binBuffOffset + this->cpyOffset);
        if (xRow == -1) {
            Duplicate<float>(xLocal, 0.0f, this->perBuffSize / sizeof(float));
            SetFlag<HardEvent::V_MTE2>(eventVMte2);
            WaitFlag<HardEvent::V_MTE2>(eventVMte2);
            PipeBarrier<PIPE_V>();
            continue;
        }
        this->CopyInWithCastFloat((int64_t)xRow, xLocal, xLocalT, colOffset, cpyCols);
        SetFlag<HardEvent::MTE2_V>(eventMte2V);
        WaitFlag<HardEvent::MTE2_V>(eventMte2V);
    }

    // S2：K超过累加buffer的情况下，多出的部分，被加数位置，直接累加到对应的buffer中
    GradProcessTokenAccumulate(tokenIdx, tokenIdxEnd, tokenBinBuffSizeOffset, tokenTmpBuffSizeOffset, colOffset,
                               cpyCols);

    // S3：拷贝加数，累加到对应的被加数buffer中
    GradProcessTokenAccumulate(tokenIdxStart + 1, tokenIdxEnd, tokenBinBuffSizeOffset, tokenTmpBuffSizeOffset,
                               colOffset, cpyCols);
    SetFlag<HardEvent::V_S>(eventVS);
    WaitFlag<HardEvent::V_S>(eventVS);

    // S4：递归完成二分累加
    int64_t stride = 1;
    for (int64_t expIdx = 0; expIdx < this->expNum; expIdx++) {
        // stride标识累加位置和被累加位置的间隔，间隔值成指数趋势
        // interval标识相邻两个存放累加结果内存之间的位置间隔，其正好是stride的2倍
        stride *= (expIdx > 0) ? this->baseStride : 1;
        int64_t interval = stride * this->baseStride;
        for (int64_t j = stride; j < this->binBufferNum; j += interval) {
            int64_t aBuffOffset = tokenBinBuffSizeOffset + this->perBuffSize * (j - stride);
            int64_t bBuffOffset = tokenBinBuffSizeOffset + this->perBuffSize * j;
            LocalTensor<float> addALocal = this->binBuff.template GetWithOffset<float>(this->perCpyCols, aBuffOffset);
            LocalTensor<float> addBLocal = this->binBuff.template GetWithOffset<float>(this->perCpyCols, bBuffOffset);
            this->GradAdd(addALocal, addBLocal, cpyCols);
        }
        PipeBarrier<PIPE_V>();
    }

    LocalTensor<float> xLocal = this->binBuff.template GetWithOffset<float>(this->perCpyCols, tokenBinBuffSizeOffset);
    LocalTensor<T> tmpLocalT = this->tmpBuff.template GetWithOffset<T>(this->perCpyCols, tokenTmpBuffSizeOffset);
    this->CopyOut(xLocal, tmpLocalT, cpyCols, outOffset);
}

template <typename T>
__aicore__ inline void MoeInitRoutingV2GradPositionPad0Compute<T>::TokenLoopProcess(int64_t elementIdx,
                                                                                    int64_t tokenBinBuffSizeOffset,
                                                                                    int64_t tokenTmpBuffSizeOffset)
{
    /*
      expanded_x: [6, 4] expanded_row_idx:[6] grad_x:[3, 4] K:2
      expanded_row_idx: 4 1 5 3 0 2
                        -     -        for token 0, stride N
                          -     -      for token 1, stride N
                            -     -    for token 2, stride N
    */

    for (int64_t tokenLoop = 0; tokenLoop < this->cpyLoops; tokenLoop++) {
        int64_t cpyCols = (tokenLoop == this->cpyLoops - 1) ? this->tailCpyCols : this->perCpyCols;
        int64_t colOffset = tokenLoop * this->perCpyCols;
        int64_t outOffset = elementIdx * this->cols + colOffset;
        // element split loop
        GradProcess(elementIdx, tokenBinBuffSizeOffset, tokenTmpBuffSizeOffset, cpyCols, colOffset, outOffset);
    }
}

template <typename T>
__aicore__ inline void MoeInitRoutingV2GradPositionPad0Compute<T>::Process()
{
    if (this->blockIdx >= this->needCoreNum) {
        return;
    }

    int64_t elementStartIdx = this->blockIdx * this->perCoreElements;
    LocalTensor<float> xLocalGroup = this->binBuff.template Get<float>();
    LocalTensor<float> tmpLocalGroup = this->tmpBuff.template Get<float>();

    // token group loops
    for (int64_t tokenGroupIdx = 0; tokenGroupIdx < this->tokensLoop; tokenGroupIdx++) {
        int64_t tokenGroupSize = (tokenGroupIdx == this->tokensLoop - 1) ? this->lastTokensFormer : this->tokensFormer;

        // empty buffer space
        Duplicate<float>(xLocalGroup, 0.0f, this->binBufferSize / sizeof(float));
        Duplicate<float>(tmpLocalGroup, 0.0f, this->tmpBufferSize / sizeof(float));
        PipeBarrier<PIPE_V>();

        // token elements loops
        for (int64_t elementLoop = 0; elementLoop < tokenGroupSize; elementLoop++) {
            int64_t elementIdx = elementStartIdx + tokenGroupIdx * this->tokensFormer + elementLoop; // 绝对位置
            int64_t tokenBinBuffSizeOffset =
                elementLoop *
                this->perTokenUseBinBuffSize; // 当前处理token使用的二分buffer偏移地址，用于获取对应二分buffer
            int64_t tokenTmpBuffSizeOffset =
                elementLoop *
                this->perTokenUseTmpBuffSize; // 当前处理token使用的临时buffer偏移地址，用于获取对应临时buffer
            TokenLoopProcess(elementIdx, tokenBinBuffSizeOffset, tokenTmpBuffSizeOffset);
        }
    }
}

} // namespace MoeInitRoutingV2Grad
#endif // MOE_INIT_ROUTING_V2_GRAD_POS_PAD0_COMPUTE_H
