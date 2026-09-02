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
 * \file ring_attention_update_tnd.h
 * \brief
 */
#ifndef _RING_ATTENTION_UPDATE_TND_H_
#define _RING_ATTENTION_UPDATE_TND_H_
#include "kernel_operator.h"

using namespace AscendC;

template <typename T>
class KernelRingAttentionUpdateTND {
public:
    __aicore__ inline KernelRingAttentionUpdateTND() {}
    __aicore__ inline void Init(GM_ADDR prevAttnOut, GM_ADDR prevSoftmaxMax, GM_ADDR prevSoftmaxSum, GM_ADDR curAttnOut,
                                GM_ADDR curSoftmaxMax, GM_ADDR curSoftmaxSum, GM_ADDR actualSeqQlen, GM_ADDR attnOut,
                                GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR workspace,
                                const RingAttentionUpdateTilingData *__restrict tiling, TPipe *tPipe)
    {
        // init input global gm buffer
        InitComputeInfo(tiling);
        prevAttnOutGm.SetGlobalBuffer((__gm__ T *)prevAttnOut);
        prevSoftmaxMaxGM.SetGlobalBuffer((__gm__ float *)prevSoftmaxMax);
        prevSoftmaxSumGm.SetGlobalBuffer((__gm__ float *)prevSoftmaxSum);
        curAttnOutGm.SetGlobalBuffer((__gm__ T *)curAttnOut);
        curSoftmaxMaxGm.SetGlobalBuffer((__gm__ float *)curSoftmaxMax);
        curSoftmaxSumGm.SetGlobalBuffer((__gm__ float *)curSoftmaxSum);
        actualSeqQlenGm.SetGlobalBuffer((__gm__ int64_t *)actualSeqQlen);
        // init output global gm buffer
        attnOutGM.SetGlobalBuffer((__gm__ T *)attnOut);
        softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmaxMax);
        softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmaxSum);

        // init input queue
        uint32_t bufferNumInQueue = 2;
        tPipe->InitBuffer(prevCurAttnOutQueue, BUFFER_NUM, attnEleNumLoop * bufferNumInQueue * inputDataSize);
        tPipe->InitBuffer(prevCurSoftmaxMaxQueue, BUFFER_NUM, softmaxEleNumLoop * bufferNumInQueue * floatDataSize);
        tPipe->InitBuffer(prevCurSoftmaxSumQueue, BUFFER_NUM, softmaxEleNumLoop * bufferNumInQueue * floatDataSize);

        // init output queue
        tPipe->InitBuffer(attnOutQueue, BUFFER_NUM, attnEleNumLoop * inputDataSize);
        tPipe->InitBuffer(softmaxMaxQueue, BUFFER_NUM, softmaxEleNumLoop * floatDataSize);
        tPipe->InitBuffer(softmaxSumQueue, BUFFER_NUM, softmaxEleNumLoop * floatDataSize);

        // init temp buffer
        uint32_t attnBufferNum = 2;
        uint32_t softmaxBufferNum = 2;
        tPipe->InitBuffer(tempFp32Buf,
                          (attnEleNumLoop * attnBufferNum + softmaxEleNumLoop * softmaxBufferNum) * floatDataSize);
        InitTempBuffer();
    }
    __aicore__ inline void Process()
    {
        if (tndSoftmaxLayout == 1 || headNum == 1) {
            ProcessSoftmaxTnd();
        } else {
            ProcessTnd();
        }
    }

    __aicore__ inline void ProcessTnd()
    {
        for (int64_t batchSizeIndex = 0; batchSizeIndex < batchSize; batchSizeIndex++) {
            seqNumBatchEndIndex = actualSeqQlenGm.GetValue(batchSizeIndex + 1);
            if (dimTIndexCore < seqNumBatchEndIndex) {
                curBatchIndex = batchSizeIndex;
                seqNumBatchStartIndex = actualSeqQlenGm.GetValue(batchSizeIndex);
                seqNumBatch = seqNumBatchEndIndex - seqNumBatchStartIndex;
                seqNumBatchTail = dimTIndexCore - seqNumBatchStartIndex;
                startTCount = 0 - seqNumBatchTail;
                break;
            }
        }
        softmaxGmOffsetLoop = seqNumBatchStartIndex * headNum * softmaxTailSize;
        attnGmOffsetLoop = dimTIndexCore * headNum * headDim;

        for (int64_t seqNumLoopIndex = 0; seqNumLoopIndex < dimTCore; seqNumLoopIndex++) {
            while (seqNumBatchTail == seqNumBatch) {
                startTCount += seqNumBatch;
                curBatchIndex += 1;
                seqNumBatchStartIndex = actualSeqQlenGm.GetValue(curBatchIndex);
                seqNumBatchEndIndex = actualSeqQlenGm.GetValue(curBatchIndex + 1);
                seqNumBatch = seqNumBatchEndIndex - seqNumBatchStartIndex;
                seqNumBatchTail = 0;
                softmaxGmOffsetLoop = seqNumBatchStartIndex * headNum * softmaxTailSize;
            }
            softmaxGmOffset = softmaxGmOffsetLoop + (seqNumLoopIndex - startTCount) * softmaxTailSize;
            attnGmOffset = attnGmOffsetLoop + seqNumLoopIndex * headNum * headDim;

            for (int64_t headNumLoopIndex = 0; headNumLoopIndex < headNumLoopTimes; headNumLoopIndex++) {
                softmaxGmStride = (seqNumBatch - 1) * softmaxTailSize * floatDataSize;
                softmaxBlockCount = headNumLoopEach;
                attnBlockLen = headNumLoopEach * headDim * inputDataSize;

                if (headNumLoopIndex != 0 && headNumLoopIndex == headNumLoopTimes - 1) {
                    softmaxBlockCount = headNumLoopTail;
                    attnBlockLen = headNumLoopTail * headDim * inputDataSize;
                }

                SoftmaxDataMoveIn();
                AttnDataMoveIn();
                SoftmaxCompute();
                AttnCompute();
                SoftmaxDataMoveOut();
                AttnDataMoveOut();

                softmaxGmOffset += headNumLoopEach * seqNumBatch * softmaxTailSize;
                attnGmOffset += headNumLoopEach * headDim;
            }
            seqNumBatchTail += 1;
        }
    }

    __aicore__ inline void ProcessSoftmaxTnd()
    {
        softmaxGmOffsetLoop = dimTIndexCore * headNum * softmaxTailSize;
        attnGmOffsetLoop = dimTIndexCore * headNum * headDim;
        softmaxGmStride = 0;
        softmaxBlockCount = 1;
        softmaxBlockLen = headNumLoopEach * softmaxTailSize * floatDataSize;
        attnBlockLen = headNumLoopEach * headDim * inputDataSize;

        for (int64_t seqNumLoopIndex = 0; seqNumLoopIndex < allLoopTimes; seqNumLoopIndex++) {
            softmaxGmOffset = softmaxGmOffsetLoop + seqNumLoopIndex * headNumLoopEach * softmaxTailSize;
            attnGmOffset = attnGmOffsetLoop + seqNumLoopIndex * headNumLoopEach * headDim;

            if (seqNumLoopIndex != 0 && seqNumLoopIndex == allLoopTimes - 1) {
                softmaxBlockLen = allLoopTail * softmaxTailSize * floatDataSize;
                attnBlockLen = allLoopTail * headDim * inputDataSize;
            }

            SoftmaxDataMoveIn();
            AttnDataMoveIn();
            SoftmaxCompute();
            AttnCompute();
            SoftmaxDataMoveOut();
            AttnDataMoveOut();

            softmaxGmOffset += headNumLoopEach * softmaxTailSize;
            attnGmOffset += headNumLoopEach * headDim;
        }
    }

private:
    __aicore__ inline void InitComputeInfo(const RingAttentionUpdateTilingData *__restrict tiling)
    {
        curBlockIdx = GetBlockIdx();
        batchSize = tiling->batchSize;
        headNum = tiling->headNum;
        headDim = tiling->headDim;
        softmaxTailSize = tiling->softmaxTailSize;
        tndSoftmaxLayout = tiling->tndSoftmaxLayout;

        inputDataSize = sizeof(T);
        floatDataSize = sizeof(float);

        blockNumInput = BLOCK_SIZE / inputDataSize;
        blockNumB32 = BLOCK_SIZE / floatDataSize;
        repeatNumB32 = REPEAT_SIZE / floatDataSize;

        coreNum = tiling->coreNum;
        if (curBlockIdx == (coreNum - 1)) {
            dimTCore = tiling->dimTCoreTail;
        } else {
            dimTCore = tiling->dimTCoreEach;
        }
        dimTIndexCore = tiling->dimTCoreEach * curBlockIdx;

        seqNumLoopEach = tiling->seqNumLoopEach;

        headNumLoopEach = tiling->headNumLoopEach;
        headNumLoopTimes = (headNum + headNumLoopEach - 1) / headNumLoopEach;
        headNumLoopTail = headNum - (headNumLoopTimes - 1) * headNumLoopEach;

        allLoopTimes = (headNum * dimTCore + headNumLoopEach - 1) / headNumLoopEach;
        if (allLoopTimes == 1) {
            headNumLoopEach = headNum * dimTCore;
        }
        allLoopTail = headNum * dimTCore - (allLoopTimes - 1) * headNumLoopEach;

        // num loop align repeat
        softmaxEleNumLoop = (headNumLoopEach * softmaxTailSize + repeatNumB32 - 1) / repeatNumB32 * repeatNumB32;
        attnEleNumLoop = (headNumLoopEach * headDim + repeatNumB32 - 1) / repeatNumB32 * repeatNumB32;

        softmaxBlockLen = softmaxTailSize * floatDataSize;
    }

    __aicore__ inline void InitTempBuffer()
    {
        uint32_t softmaxTempBuf0Offset = 0;
        uint32_t softmaxTempBuf1Offset = softmaxTempBuf0Offset + softmaxEleNumLoop * floatDataSize;
        uint32_t attnTempBuf0Offset = softmaxTempBuf1Offset + softmaxEleNumLoop * floatDataSize;
        uint32_t attnTempBuf1Offset = attnTempBuf0Offset + attnEleNumLoop * floatDataSize;
        softmaxTempBuf0 = tempFp32Buf.GetWithOffset<float>(softmaxEleNumLoop, softmaxTempBuf0Offset);
        softmaxTempBuf1 = tempFp32Buf.GetWithOffset<float>(softmaxEleNumLoop, softmaxTempBuf1Offset);
        attnTempBuf0 = tempFp32Buf.GetWithOffset<float>(attnEleNumLoop, attnTempBuf0Offset);
        attnTempBuf1 = tempFp32Buf.GetWithOffset<float>(attnEleNumLoop, attnTempBuf1Offset);
    }

    __aicore__ inline void SoftmaxDataMoveIn()
    {
        // softmax max move in
        DataCopyExtParams softmaxCopyParams{(uint16_t)softmaxBlockCount, softmaxBlockLen, softmaxGmStride, 0, 0};
        DataCopyPadExtParams<float> softmaxPadParams{false, 0, 0, 0};

        prevCurSoftmaxMaxLocal = prevCurSoftmaxMaxQueue.AllocTensor<float>();
        DataCopyPad(prevCurSoftmaxMaxLocal, prevSoftmaxMaxGM[softmaxGmOffset], softmaxCopyParams, softmaxPadParams);
        DataCopyPad(prevCurSoftmaxMaxLocal[softmaxEleNumLoop], curSoftmaxMaxGm[softmaxGmOffset], softmaxCopyParams,
                    softmaxPadParams);
        prevCurSoftmaxMaxQueue.EnQue<float>(prevCurSoftmaxMaxLocal);

        // softmax sum move in
        prevCurSoftmaxSumLocal = prevCurSoftmaxSumQueue.AllocTensor<float>();
        DataCopyPad(prevCurSoftmaxSumLocal, prevSoftmaxSumGm[softmaxGmOffset], softmaxCopyParams, softmaxPadParams);
        DataCopyPad(prevCurSoftmaxSumLocal[softmaxEleNumLoop], curSoftmaxSumGm[softmaxGmOffset], softmaxCopyParams,
                    softmaxPadParams);
        prevCurSoftmaxSumQueue.EnQue<float>(prevCurSoftmaxSumLocal);
    }

    __aicore__ inline void SoftmaxCompute()
    {
        // softmax_max = torch.maximum(prev_softmax_max, cur_softmax_max)
        uint8_t softmaxRepeatTimes = softmaxEleNumLoop / repeatNumB32;

        prevCurSoftmaxMaxLocal = prevCurSoftmaxMaxQueue.DeQue<float>();
        softmaxMaxLocal = softmaxMaxQueue.AllocTensor<float>();
        Max(softmaxMaxLocal, prevCurSoftmaxMaxLocal, prevCurSoftmaxMaxLocal[softmaxEleNumLoop], mask,
            softmaxRepeatTimes, {1, 1, 1, 8, 8, 8});
        PipeBarrier<PIPE_V>();
        // prev_scale = torch.exp(prev_softmax_max - softmax)
        // cur_scale = torch.exp(cur_softmax_max - softmax)
        Sub(softmaxTempBuf0, prevCurSoftmaxMaxLocal, softmaxMaxLocal, mask, softmaxRepeatTimes, {1, 1, 1, 8, 8, 8});
        Sub(softmaxTempBuf1, prevCurSoftmaxMaxLocal[softmaxEleNumLoop], softmaxMaxLocal, mask, softmaxRepeatTimes,
            {1, 1, 1, 8, 8, 8});
        PipeBarrier<PIPE_V>();
        prevCurSoftmaxMaxQueue.FreeTensor<float>(prevCurSoftmaxMaxLocal);
        softmaxMaxQueue.EnQue<float>(softmaxMaxLocal);

        Exp(softmaxTempBuf0, softmaxTempBuf0, mask, softmaxRepeatTimes, {1, 1, 8, 8});
        Exp(softmaxTempBuf1, softmaxTempBuf1, mask, softmaxRepeatTimes, {1, 1, 8, 8});
        PipeBarrier<PIPE_V>();
        // prev_softmax_sum_scaled = prev_softmax_sum * prev_scale
        // cur_softmax_sum_scaled = cur_softmax_sum * prev_scale
        prevCurSoftmaxSumLocal = prevCurSoftmaxSumQueue.DeQue<float>();
        Mul(softmaxTempBuf0, prevCurSoftmaxSumLocal, softmaxTempBuf0, mask, softmaxRepeatTimes, {1, 1, 1, 8, 8, 8});
        Mul(softmaxTempBuf1, prevCurSoftmaxSumLocal[softmaxEleNumLoop], softmaxTempBuf1, mask, softmaxRepeatTimes,
            {1, 1, 1, 8, 8, 8});
        PipeBarrier<PIPE_V>();
        prevCurSoftmaxSumQueue.FreeTensor<float>(prevCurSoftmaxSumLocal);

        // softmax_sum = prev_softmax_sum_scaled + cur_softmax_sum_scaled
        softmaxSumLocal = softmaxSumQueue.AllocTensor<float>();
        Add(softmaxSumLocal, softmaxTempBuf0, softmaxTempBuf1, mask, softmaxRepeatTimes, {1, 1, 1, 8, 8, 8});
        PipeBarrier<PIPE_V>();
        // prev_out_scale = prev_softmax_sum_scaled / softmax_sum
        // cur_out_scale = cur_softmax_sum_scaled / softmax_sum
        Div(softmaxTempBuf0, softmaxTempBuf0, softmaxSumLocal, mask, softmaxRepeatTimes, {1, 1, 1, 8, 8, 8});
        Div(softmaxTempBuf1, softmaxTempBuf1, softmaxSumLocal, mask, softmaxRepeatTimes, {1, 1, 1, 8, 8, 8});
        PipeBarrier<PIPE_V>();
        softmaxSumQueue.EnQue<float>(softmaxSumLocal);
    }

    __aicore__ inline void SoftmaxDataMoveOut()
    {
        // softmax max move out
        DataCopyExtParams copy_params{(uint16_t)softmaxBlockCount, softmaxBlockLen, 0, softmaxGmStride, 0};
        softmaxMaxLocal = softmaxMaxQueue.DeQue<float>();
        DataCopyPad(softmaxMaxGm[softmaxGmOffset], softmaxMaxLocal, copy_params);
        softmaxMaxQueue.FreeTensor<float>(softmaxMaxLocal);

        // softmax sum move out
        softmaxSumLocal = softmaxSumQueue.DeQue<float>();
        DataCopyPad(softmaxSumGm[softmaxGmOffset], softmaxSumLocal, copy_params);
        softmaxSumQueue.FreeTensor<float>(softmaxSumLocal);
    }

    __aicore__ inline void AttnDataMoveIn()
    {
        // attn move in
        DataCopyExtParams attnCopyParams{1, attnBlockLen, 0, 0, 0};
        DataCopyPadExtParams<T> attnPadParams{false, 0, 0, 0};
        prevCurAttnOutLocal = prevCurAttnOutQueue.AllocTensor<T>();
        DataCopyPad(prevCurAttnOutLocal, prevAttnOutGm[attnGmOffset], attnCopyParams, attnPadParams);
        DataCopyPad(prevCurAttnOutLocal[attnEleNumLoop], curAttnOutGm[attnGmOffset], attnCopyParams, attnPadParams);
        prevCurAttnOutQueue.EnQue<T>(prevCurAttnOutLocal);
    }

    __aicore__ inline void AttnCompute()
    {
        uint8_t attnRepeatTimes = attnEleNumLoop / repeatNumB32;
        // attn_out = prev_attn_out * prev_out_scale + cur_attn_out * cur_out_scale
        prevCurAttnOutLocal = prevCurAttnOutQueue.DeQue<T>();
        if constexpr (std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value) {
            Cast(attnTempBuf0, prevCurAttnOutLocal, RoundMode::CAST_NONE, mask, attnRepeatTimes, {1, 1, 8, 4});
            Cast(attnTempBuf1, prevCurAttnOutLocal[attnEleNumLoop], RoundMode::CAST_NONE, mask, attnRepeatTimes,
                 {1, 1, 8, 4});
        } else if constexpr (std::is_same<T, float>::value) {
            Copy(attnTempBuf0, prevCurAttnOutLocal, mask, attnRepeatTimes, {1, 1, 8, 8});
            Copy(attnTempBuf1, prevCurAttnOutLocal[attnEleNumLoop], mask, attnRepeatTimes, {1, 1, 8, 8});
        }
        PipeBarrier<PIPE_V>();
        prevCurAttnOutQueue.FreeTensor<T>(prevCurAttnOutLocal);

        BinaryRepeatParams repeatParamsAttnSoftmax = {1,
                                                      1,
                                                      0,
                                                      (uint8_t)(headDim / blockNumB32),
                                                      (uint8_t)(headDim / blockNumB32),
                                                      (uint8_t)(softmaxTailSize / blockNumB32)};
        for (int64_t attnLoopIndex = 0; attnLoopIndex < headDim / repeatNumB32; attnLoopIndex++) {
            Mul(attnTempBuf0[attnLoopIndex * repeatNumB32], attnTempBuf0[attnLoopIndex * repeatNumB32], softmaxTempBuf0,
                mask, headNumLoopEach, repeatParamsAttnSoftmax);
            Mul(attnTempBuf1[attnLoopIndex * repeatNumB32], attnTempBuf1[attnLoopIndex * repeatNumB32], softmaxTempBuf1,
                mask, headNumLoopEach, repeatParamsAttnSoftmax);
        }
        PipeBarrier<PIPE_V>();
        Add(attnTempBuf0, attnTempBuf0, attnTempBuf1, mask, attnRepeatTimes, {1, 1, 1, 8, 8, 8});
        PipeBarrier<PIPE_V>();
        attnOutLocal = attnOutQueue.AllocTensor<T>();
        if constexpr (std::is_same<T, half>::value) {
            Cast(attnOutLocal, attnTempBuf0, RoundMode::CAST_NONE, mask, attnRepeatTimes, {1, 1, 4, 8});
        } else if constexpr (std::is_same<T, bfloat16_t>::value) {
            Cast(attnOutLocal, attnTempBuf0, RoundMode::CAST_RINT, mask, attnRepeatTimes, {1, 1, 4, 8});
        } else if constexpr (std::is_same<T, float>::value) {
            Copy(attnOutLocal, attnTempBuf0, mask, attnRepeatTimes, {1, 1, 8, 8});
        }
        PipeBarrier<PIPE_V>();
        attnOutQueue.EnQue<T>(attnOutLocal);
    }

    __aicore__ inline void AttnDataMoveOut()
    {
        // attn move out
        DataCopyExtParams attnCopyParams{1, attnBlockLen, 0, 0, 0};
        attnOutLocal = attnOutQueue.DeQue<T>();
        DataCopyPad(attnOutGM[attnGmOffset], attnOutLocal, attnCopyParams);
        attnOutQueue.FreeTensor<T>(attnOutLocal);
    }

    constexpr static uint32_t BLOCK_SIZE = 32;
    constexpr static uint32_t REPEAT_SIZE = 256;
    // buffer num: 1 or 2
    constexpr static int32_t BUFFER_NUM = 2;
    // define input global input gm buffer
    GlobalTensor<T> prevAttnOutGm;
    GlobalTensor<float> prevSoftmaxMaxGM;
    GlobalTensor<float> prevSoftmaxSumGm;
    GlobalTensor<T> curAttnOutGm;
    GlobalTensor<float> curSoftmaxMaxGm;
    GlobalTensor<float> curSoftmaxSumGm;
    GlobalTensor<int64_t> actualSeqQlenGm;
    // define input global input gm buffer
    GlobalTensor<T> attnOutGM;
    GlobalTensor<float> softmaxMaxGm;
    GlobalTensor<float> softmaxSumGm;
    // define input queue
    TQue<QuePosition::VECIN, BUFFER_NUM> prevCurAttnOutQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> prevCurSoftmaxMaxQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> prevCurSoftmaxSumQueue;
    // define output queue
    TQue<QuePosition::VECOUT, BUFFER_NUM> attnOutQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> softmaxMaxQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> softmaxSumQueue;
    // define temp buffer
    TBuf<TPosition::VECCALC> tempFp32Buf;
    // define input ub tensor buffer
    LocalTensor<T> prevCurAttnOutLocal;
    LocalTensor<float> prevCurSoftmaxMaxLocal;
    LocalTensor<float> prevCurSoftmaxSumLocal;
    // define output ub tensor buffer
    LocalTensor<T> attnOutLocal;
    LocalTensor<float> softmaxMaxLocal;
    LocalTensor<float> softmaxSumLocal;
    // define temp ub tensor buffer
    LocalTensor<float> softmaxTempBuf0;
    LocalTensor<float> softmaxTempBuf1;
    LocalTensor<float> attnTempBuf0;
    LocalTensor<float> attnTempBuf1;
    // core info
    int64_t curBlockIdx;
    int64_t coreNum;
    // shape info
    int64_t batchSize;
    int64_t headNum;
    int64_t headDim;
    int64_t softmaxTailSize;

    // define loop data
    int64_t batchSizeIndexLoop;
    int64_t batchSizeIndexCore;
    int64_t dimTIndexCore;

    int64_t seqNumBatchStartIndex;
    int64_t seqNumBatchEndIndex;
    int64_t seqNumBatchTail;
    int64_t seqNumBatch;
    int64_t curBatchIndex;
    int64_t startTCount;

    int64_t bnNumGroup;
    int64_t bnGmIndexCore;
    int64_t bnGmIndexLoop;

    int64_t seqNumGmIndexCore;
    int64_t seqNumGmIndexLoop;
    int64_t seqNumCore;
    int64_t seqNumLoopTimes;
    int64_t seqNumLoopEach;
    int64_t seqNumLoopTail;
    int64_t seqNumLoop;

    int64_t headDimLoopTimes;
    int64_t headDimLoopEach;
    int64_t headDimLoopTail;
    int64_t headDimLoop;
    int64_t headDimLoopAlign;

    int64_t headNumLoopTimes;
    int64_t headNumLoopEach;
    int64_t headNumLoopTail;
    int64_t headNumLoop;
    int64_t headNumLoopAlign;
    int64_t allLoopTimes;
    int64_t allLoopTail;

    // gm offset
    int64_t softmaxBnGmOffset;
    int64_t softmaxSeqNumGmOffset;
    int64_t softmaxGmOffset;

    int64_t attnBnGmOffset;
    int64_t attnSeqNumGmOffset;
    int64_t attnGmOffset;

    uint32_t attnEleNumLoop;
    uint32_t softmaxEleNumLoop;
    // core compute
    uint32_t softmaxBlockCount;
    uint32_t softmaxBlockLen;
    uint16_t attnBlockCount;
    uint32_t attnBlockLen;
    uint32_t attnGmStride;
    uint32_t attnUbStride;
    uint32_t softmaxGmStride;
    //
    int64_t batchSizeCore;
    int64_t dimTCore;
    int64_t attnGmOffsetLoop;
    int64_t softmaxGmOffsetLoop;

    // compute info
    uint64_t mask[2] = {UINT64_MAX, 0};
    uint32_t inputDataSize;
    uint32_t floatDataSize;
    uint32_t blockNumInput;
    uint32_t blockNumB32;
    uint32_t repeatNumB32;
    uint8_t tndSoftmaxLayout;
};
#endif // _RING_ATTENTION_UPDATE_TND_H_
