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
 * \file ring_attention_update_regbase_softmax_tnd.h
 * \brief
 */
#ifndef _RING_ATTENTION_UPDATE_REGBASE_SOFTMAX_TND_H_
#define _RING_ATTENTION_UPDATE_REGBASE_SOFTMAX_TND_H_
#include "kernel_operator.h"
#include "ring_attention_update_regbase_common.h"

namespace RingAttentionUpdateRegbase {

using namespace AscendC;

template <typename T>
class KernelRingAttentionUpdateRegbaseSoftmaxTND {
public:
    __aicore__ inline KernelRingAttentionUpdateRegbaseSoftmaxTND() {}
    __aicore__ inline void Init(GM_ADDR prevAttnOut, GM_ADDR prevSoftmaxMax, GM_ADDR prevSoftmaxSum, GM_ADDR curAttnOut,
                                GM_ADDR curSoftmaxMax, GM_ADDR curSoftmaxSum, GM_ADDR actualSeqQlen, GM_ADDR attnOut,
                                GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR workspace,
                                const RingAttentionUpdateRegbaseSoftmaxTNDTilingData *__restrict tiling, TPipe *tPipe)
    {
        // init input global gm buffer
        InitComputeInfo(tiling);
        prevAttnOutGm.SetGlobalBuffer((__gm__ T *)prevAttnOut);
        prevSoftmaxSumGm.SetGlobalBuffer((__gm__ float *)prevSoftmaxSum);
        prevSoftmaxMaxGm.SetGlobalBuffer((__gm__ float *)prevSoftmaxMax);
        curAttnOutGm.SetGlobalBuffer((__gm__ T *)curAttnOut);
        curSoftmaxSumGm.SetGlobalBuffer((__gm__ float *)curSoftmaxSum);
        curSoftmaxMaxGm.SetGlobalBuffer((__gm__ float *)curSoftmaxMax);
        actualSeqQlenGm.SetGlobalBuffer((__gm__ int64_t *)actualSeqQlen);
        // init output global gm buffer
        attnOutGM.SetGlobalBuffer((__gm__ T *)attnOut);
        softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmaxMax);
        softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmaxSum);

        // init input queue
        uint32_t bufferNumInQueue = 2; // cur和prev共用一个flagid
        tPipe->InitBuffer(prevCurAttnOutInQueue, BUFFER_NUM, attnEleNumLoop * bufferNumInQueue * inputDataSize);
        tPipe->InitBuffer(prevCurSoftmaxSumInQueue, BUFFER_NUM, softmaxEleNumLoop * bufferNumInQueue * floatDataSize);
        tPipe->InitBuffer(prevCurSoftmaxMaxInQueue, BUFFER_NUM, softmaxEleNumLoop * bufferNumInQueue * floatDataSize);

        // init output queue
        tPipe->InitBuffer(attnOutOutQueue, BUFFER_NUM, attnEleNumLoop * inputDataSize);
        tPipe->InitBuffer(softmaxSumOutQueue, BUFFER_NUM, softmaxEleNumLoop * floatDataSize);
        tPipe->InitBuffer(softmaxMaxOutQueue, BUFFER_NUM, softmaxEleNumLoop * floatDataSize);

        // init temp buffer
        tPipe->InitBuffer(tempFp32Buf1, softmaxEleNumLoop * floatDataSize);
        tPipe->InitBuffer(tempFp32Buf2, softmaxEleNumLoop * floatDataSize);

        tempFp32Buf1Local = tempFp32Buf1.Get<float>();
        tempFp32Buf2Local = tempFp32Buf2.Get<float>();
    }

    __aicore__ inline void Process()
    {
        if (curBlockIdx >= usedVectorCoreNum) {
            return;
        }
        ProcessSoftmaxTND();
    }

    __aicore__ inline void ProcessSoftmaxTND()
    {
        tnNumLoopTimes = CeilDiv(tnNumCore, tnNumLoopEach);
        tnNumLoopTail = tnNumCore - tnNumLoopEach * (tnNumLoopTimes - 1);
        for (int64_t tnLoopIdx = 0; tnLoopIdx < tnNumLoopTimes; tnLoopIdx++) {
            tnNumLoop = tnLoopIdx == tnNumLoopTimes - 1 ? tnNumLoopTail : tnNumLoopEach; // 当前循环TN方向计算行数
            softmaxGmOffset = (tnStartIdxCore + tnLoopIdx * tnNumLoopEach) * softmaxTailSize;
            softmaxBlockLen = tnNumLoop * softmaxTailSize * floatDataSize;

            SoftmaxDataMoveInSoftmaxTND();
            SoftmaxComputeSoftmaxTND();
            SoftmaxDataMoveOutSoftmaxTND();

            attnBlockCount = tnNumLoop;
            for (int64_t headDimLoopIdx = 0; headDimLoopIdx < headDimLoopTimes; headDimLoopIdx++) {
                headDimLoop = headDimLoopIdx == headDimLoopTimes - 1 ? headDimLoopTail : headDimLoopEach;
                attnGmOffset = (tnStartIdxCore + tnLoopIdx * tnNumLoopEach) * dimD + headDimLoopIdx * headDimLoopEach;
                attnBlockLen = headDimLoop * inputDataSize;
                attnGmStride = (dimD - headDimLoop) * inputDataSize;
                attnUbStride = (headDimLoopEach - headDimLoop) / blockNumInput; // 32B为单位，UB补齐到headDimLoopEach

                AttnDataMoveInSoftmaxTND();
                PipeBarrier<PIPE_V>();
                if constexpr (IsSameType<T, float>::value) {
                    AttnComputeSoftmaxTNDFp32();
                } else {
                    AttnComputeSoftmaxTNDBf16Fp16();
                }
                AttnDataMoveOutSoftmaxTND();
            }
        }
    }

private:
    __aicore__ inline void InitComputeInfo(const RingAttentionUpdateRegbaseSoftmaxTNDTilingData *__restrict tiling)
    {
        curBlockIdx = GetBlockIdx();
        dimT = tiling->dimT;
        dimN = tiling->dimN;
        dimD = tiling->dimD;
        softmaxTailSize = tiling->softmaxTailSize;
        batchSize = tiling->batchSize;

        inputDataSize = sizeof(T);
        floatDataSize = sizeof(float);

        blockNumInput = BLOCK_SIZE / inputDataSize;

        tnNumCoreEach = tiling->tnNumCoreEach;         // 每核处理的TN方向行数
        tnNumCoreTail = tiling->tnNumCoreTail;         // 尾核处理的TN方向行数
        usedVectorCoreNum = tiling->usedVectorCoreNum; // 使用的vector核数

        tnNumLoopEach = tiling->tnNumLoopEach;     // 核内单次根据ub空间计算出的最大存放TN方向行数
        headDimLoopEach = tiling->headDimLoopEach; // dimD方向单次循环长度
        headDimLoopTimes = (dimD + headDimLoopEach - 1) / headDimLoopEach; // dimD方向循环次数
        headDimLoopTail = dimD - (headDimLoopTimes - 1) * headDimLoopEach; // dimD方向尾次循环长度

        // 本核处理
        tnNumCore = curBlockIdx == usedVectorCoreNum - 1 ? tnNumCoreTail : tnNumCoreEach; // 本核处理的TN方向行数
        tnStartIdxCore = curBlockIdx * tnNumCoreEach; // 本核处理的TN起始行号(包)
        tnEndIdxCore = tnStartIdxCore + tnNumCore;    // 本核处理的TN结束行号(不包)

        softmaxEleNumLoop = tnNumLoopEach * softmaxTailSize;
        attnEleNumLoop = tnNumLoopEach * headDimLoopEach;

        softmaxBlockLen = softmaxTailSize * floatDataSize;
    }

    __aicore__ inline void SoftmaxDataMoveInSoftmaxTND()
    {
        prevCurSoftmaxMaxLocal = prevCurSoftmaxMaxInQueue.AllocTensor<float>();
        prevCurSoftmaxSumLocal = prevCurSoftmaxSumInQueue.AllocTensor<float>();

        DataCopyExtParams softmaxCopyParams{1, softmaxBlockLen, 0, 0, 0};
        DataCopyPadExtParams<float> softmaxPadParams{false, 0, 0, 0};

        // softmax max move in
        DataCopyPad(prevCurSoftmaxMaxLocal, prevSoftmaxMaxGm[softmaxGmOffset], softmaxCopyParams, softmaxPadParams);
        DataCopyPad(prevCurSoftmaxMaxLocal[softmaxEleNumLoop], curSoftmaxMaxGm[softmaxGmOffset], softmaxCopyParams,
                    softmaxPadParams);

        // softmax sum move in
        DataCopyPad(prevCurSoftmaxSumLocal, prevSoftmaxSumGm[softmaxGmOffset], softmaxCopyParams, softmaxPadParams);
        DataCopyPad(prevCurSoftmaxSumLocal[softmaxEleNumLoop], curSoftmaxSumGm[softmaxGmOffset], softmaxCopyParams,
                    softmaxPadParams);

        prevCurSoftmaxMaxInQueue.EnQue<float>(prevCurSoftmaxMaxLocal);
        prevCurSoftmaxSumInQueue.EnQue<float>(prevCurSoftmaxSumLocal);
    }

    __aicore__ inline void SoftmaxComputeSoftmaxTND()
    {
        prevCurSoftmaxMaxLocal = prevCurSoftmaxMaxInQueue.DeQue<float>();
        prevCurSoftmaxSumLocal = prevCurSoftmaxSumInQueue.DeQue<float>();
        softmaxMaxLocal = softmaxMaxOutQueue.AllocTensor<float>();
        softmaxSumLocal = softmaxSumOutQueue.AllocTensor<float>();

        SoftmaxCompute(prevCurSoftmaxMaxLocal, prevCurSoftmaxSumLocal, softmaxMaxLocal, softmaxSumLocal,
                       tempFp32Buf1Local, tempFp32Buf2Local, softmaxEleNumLoop, tnNumLoop * softmaxTailSize);

        prevCurSoftmaxMaxInQueue.FreeTensor<float>(prevCurSoftmaxMaxLocal);
        prevCurSoftmaxSumInQueue.FreeTensor<float>(prevCurSoftmaxSumLocal);
        softmaxMaxOutQueue.EnQue<float>(softmaxMaxLocal);
        softmaxSumOutQueue.EnQue<float>(softmaxSumLocal);
    }

    __aicore__ inline void SoftmaxDataMoveOutSoftmaxTND()
    {
        softmaxMaxLocal = softmaxMaxOutQueue.DeQue<float>();
        softmaxSumLocal = softmaxSumOutQueue.DeQue<float>();

        DataCopyExtParams copy_params{1, softmaxBlockLen, 0, 0, 0};

        // softmax max move out
        DataCopyPad(softmaxMaxGm[softmaxGmOffset], softmaxMaxLocal, copy_params);
        // softmax sum move out
        DataCopyPad(softmaxSumGm[softmaxGmOffset], softmaxSumLocal, copy_params);

        softmaxMaxOutQueue.FreeTensor<float>(softmaxMaxLocal);
        softmaxSumOutQueue.FreeTensor<float>(softmaxSumLocal);
    }

    __aicore__ inline void AttnDataMoveInSoftmaxTND()
    {
        DataCopyExtParams attnCopyParams{attnBlockCount, attnBlockLen, attnGmStride, attnUbStride, 0};
        DataCopyPadExtParams<T> attnPadParams{false, 0, 0, 0};
        // attn move in
        prevCurAttnOutLocal = prevCurAttnOutInQueue.AllocTensor<T>();
        DataCopyPad(prevCurAttnOutLocal, prevAttnOutGm[attnGmOffset], attnCopyParams, attnPadParams);
        DataCopyPad(prevCurAttnOutLocal[attnEleNumLoop], curAttnOutGm[attnGmOffset], attnCopyParams, attnPadParams);
        prevCurAttnOutInQueue.EnQue<T>(prevCurAttnOutLocal);
    }

    __aicore__ inline void AttnComputeSoftmaxTNDBf16Fp16()
    {
        attnOutLocal = attnOutOutQueue.AllocTensor<T>();
        prevCurAttnOutLocal = prevCurAttnOutInQueue.DeQue<T>();

        AttnComputeBf16Fp16<T>(prevCurAttnOutLocal, attnOutLocal, tempFp32Buf1Local, tempFp32Buf2Local, attnEleNumLoop,
                               tnNumLoop, headDimLoop, headDimLoopEach, softmaxTailSize);

        prevCurAttnOutInQueue.FreeTensor<T>(prevCurAttnOutLocal);
        attnOutOutQueue.EnQue<T>(attnOutLocal);
    }

    __aicore__ inline void AttnComputeSoftmaxTNDFp32()
    {
        prevCurAttnOutLocal = prevCurAttnOutInQueue.DeQue<T>();
        attnOutLocal = attnOutOutQueue.AllocTensor<T>();

        AttnComputeFp32<T>(prevCurAttnOutLocal, attnOutLocal, tempFp32Buf1Local, tempFp32Buf2Local, attnEleNumLoop,
                           tnNumLoop, headDimLoop, headDimLoopEach, softmaxTailSize);

        prevCurAttnOutInQueue.FreeTensor<T>(prevCurAttnOutLocal);
        attnOutOutQueue.EnQue<T>(attnOutLocal);
    }

    __aicore__ inline void AttnDataMoveOutSoftmaxTND()
    {
        // attn move out
        DataCopyExtParams attnCopyParams{attnBlockCount, attnBlockLen, attnUbStride, attnGmStride, 0};
        attnOutLocal = attnOutOutQueue.DeQue<T>();
        DataCopyPad(attnOutGM[attnGmOffset], attnOutLocal, attnCopyParams);
        attnOutOutQueue.FreeTensor<T>(attnOutLocal);
    }

    // buffer num: 1 or 2
    constexpr static int32_t BUFFER_NUM = 2;
    constexpr static uint32_t BLOCK_SIZE = 32;
    // define input global input gm buffer
    GlobalTensor<T> prevAttnOutGm;
    GlobalTensor<float> prevSoftmaxMaxGm;
    GlobalTensor<float> prevSoftmaxSumGm;
    GlobalTensor<T> curAttnOutGm;
    GlobalTensor<float> curSoftmaxMaxGm;
    GlobalTensor<float> curSoftmaxSumGm;
    GlobalTensor<int64_t> actualSeqQlenGm;
    // define output global output gm buffer
    GlobalTensor<T> attnOutGM;
    GlobalTensor<float> softmaxMaxGm;
    GlobalTensor<float> softmaxSumGm;
    // define input queue
    TQue<QuePosition::VECIN, BUFFER_NUM> prevCurAttnOutInQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> prevCurSoftmaxMaxInQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> prevCurSoftmaxSumInQueue;
    // define output queue
    TQue<QuePosition::VECOUT, BUFFER_NUM> attnOutOutQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> softmaxMaxOutQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> softmaxSumOutQueue;
    // define temp buffer
    TBuf<TPosition::VECCALC> tempFp32Buf1;
    TBuf<TPosition::VECCALC> tempFp32Buf2;
    // define input ub tensor buffer
    LocalTensor<T> prevCurAttnOutLocal;
    LocalTensor<float> prevCurSoftmaxMaxLocal;
    LocalTensor<float> prevCurSoftmaxSumLocal;
    // define output ub tensor buffer
    LocalTensor<T> attnOutLocal;
    LocalTensor<float> softmaxMaxLocal;
    LocalTensor<float> softmaxSumLocal;
    // define temp ub tensor buffer
    LocalTensor<float> tempFp32Buf1Local;
    LocalTensor<float> tempFp32Buf2Local;
    // core info
    int64_t curBlockIdx;
    // shape info
    int64_t dimT;
    int64_t dimN;
    int64_t dimD;
    int64_t softmaxTailSize;
    int64_t batchSize;

    // define loop data
    int64_t tnNumCoreEach;
    int64_t tnNumCoreTail;
    int64_t usedVectorCoreNum;

    int64_t tnNumLoopEach;
    int64_t headDimLoopEach;
    int64_t headDimLoopTimes;
    int64_t headDimLoopTail;
    int64_t headDimLoop;

    // core compute
    int64_t tnNumCore;
    int64_t tnStartIdxCore;
    int64_t tnEndIdxCore;

    // gm offset
    int64_t attnGmOffset;
    int64_t softmaxGmOffset;

    int64_t attnEleNumLoop;
    int64_t softmaxEleNumLoop;
    uint32_t softmaxBlockLen;
    uint16_t attnBlockCount;
    uint32_t attnBlockLen;
    uint32_t attnGmStride;
    uint32_t attnUbStride;

    int64_t tnNumLoopTimes;
    int64_t tnNumLoopTail;
    int64_t tnNumLoop;

    // compute info
    uint32_t inputDataSize;
    uint32_t floatDataSize;
    uint32_t blockNumInput;
};
} // namespace RingAttentionUpdateRegbase
#endif // _RING_ATTENTION_UPDATE_REGBASE_SOFTMAX_TND_H_
