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
 * \file ring_attention_update_regbase_sbh.h
 * \brief
 */
#ifndef _RING_ATTENTION_UPDATE_REGBASE_SBH_H_
#define _RING_ATTENTION_UPDATE_REGBASE_SBH_H_
#include "kernel_operator.h"
#include "ring_attention_update_regbase_common.h"

namespace RingAttentionUpdateRegbase {

using namespace AscendC;

template <typename T>
class KernelRingAttentionUpdateRegbaseSBH {
public:
    __aicore__ inline KernelRingAttentionUpdateRegbaseSBH() {}
    __aicore__ inline void Init(GM_ADDR prevAttnOut, GM_ADDR prevSoftmaxMax, GM_ADDR prevSoftmaxSum, GM_ADDR curAttnOut,
                                GM_ADDR curSoftmaxMax, GM_ADDR curSoftmaxSum, GM_ADDR actualSeqQlen, GM_ADDR attnOut,
                                GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR workspace,
                                const RingAttentionUpdateRegbaseSBHTilingData *__restrict tiling, TPipe *tPipe)
    {
        // init input global gm buffer
        InitComputeInfo(tiling);
        prevAttnOutGm.SetGlobalBuffer((__gm__ T *)prevAttnOut);
        prevSoftmaxMaxGm.SetGlobalBuffer((__gm__ float *)prevSoftmaxMax);
        prevSoftmaxSumGm.SetGlobalBuffer((__gm__ float *)prevSoftmaxSum);
        curAttnOutGm.SetGlobalBuffer((__gm__ T *)curAttnOut);
        curSoftmaxMaxGm.SetGlobalBuffer((__gm__ float *)curSoftmaxMax);
        curSoftmaxSumGm.SetGlobalBuffer((__gm__ float *)curSoftmaxSum);
        // init output global gm buffer
        attnOutGM.SetGlobalBuffer((__gm__ T *)attnOut);
        softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmaxMax);
        softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmaxSum);

        // init input queue
        uint32_t bufferNumInQueue = 2; // cur和prev共用一个flagid
        tPipe->InitBuffer(prevCurSoftmaxMaxInQueue, BUFFER_NUM, softmaxEleNumLoop * bufferNumInQueue * floatDataSize);
        tPipe->InitBuffer(prevCurSoftmaxSumInQueue, BUFFER_NUM, softmaxEleNumLoop * bufferNumInQueue * floatDataSize);
        tPipe->InitBuffer(prevCurAttnOutInQueue, BUFFER_NUM, attnEleNumLoop * bufferNumInQueue * inputDataSize);

        // init output queue
        tPipe->InitBuffer(softmaxMaxOutQueue, BUFFER_NUM, softmaxEleNumLoop * floatDataSize);
        tPipe->InitBuffer(softmaxSumOutQueue, BUFFER_NUM, softmaxEleNumLoop * floatDataSize);
        tPipe->InitBuffer(attnOutOutQueue, BUFFER_NUM, attnEleNumLoop * inputDataSize);

        // init temp buffer
        tPipe->InitBuffer(tempFp32Buf1, softmaxEleNumLoop * floatDataSize);
        tPipe->InitBuffer(tempFp32Buf2, softmaxEleNumLoop * floatDataSize);

        tempFp32Buf2Local = tempFp32Buf2.Get<float>();
        tempFp32Buf1Local = tempFp32Buf1.Get<float>();
    }

    __aicore__ inline void Process()
    {
        if (curBlockIdx >= usedVectorCoreNum) {
            return;
        }

        for (int64_t bnLoopIndex = 0; bnLoopIndex < bnNumGroup; bnLoopIndex++) {
            bnGmIndexLoop = bnGmIndexCore + bnLoopIndex; // BN方向偏移
            for (int64_t seqNumLoopIndex = 0; seqNumLoopIndex < seqNumLoopTimes; seqNumLoopIndex++) {
                seqNumGmIndexLoop = seqNumGmIndexCore + seqNumLoopIndex * seqNumLoopEach; // S方向偏移
                SoftmaxComputeLoop(seqNumLoopIndex, seqNumGmIndexLoop);
                for (int64_t headDimLoopIndex = 0; headDimLoopIndex < headDimLoopTimes; headDimLoopIndex++) {
                    AttnComputeLoop(headDimLoopIndex, seqNumGmIndexLoop);
                }
            }
        }
    }

private:
    __aicore__ inline void InitComputeInfo(const RingAttentionUpdateRegbaseSBHTilingData *__restrict tiling)
    {
        curBlockIdx = GetBlockIdx();
        dimB = tiling->dimB;
        dimS = tiling->dimS;
        dimN = tiling->dimN;
        dimD = tiling->dimD;
        softmaxTailSize = tiling->softmaxTailSize;
        bnNum = dimB * dimN;
        usedVectorCoreNum = tiling->usedVectorCoreNum; // 使用的vector核数

        inputDataSize = sizeof(T);
        floatDataSize = sizeof(float);

        blockNumInput = BLOCK_SIZE / inputDataSize;

        groupIndex = curBlockIdx / tiling->coreNumGroup;     // 当前核所在组号
        coreIndexGroup = curBlockIdx % tiling->coreNumGroup; // 当前核在当前组核序号
        if (coreIndexGroup == (tiling->coreNumGroup - 1)) {
            seqNumCore = tiling->seqNumCoreTail;
        } else {
            seqNumCore = tiling->seqNumCoreEach;
        }

        seqNumLoopEach = tiling->seqNumLoopEach;
        seqNumLoopTimes = (seqNumCore + seqNumLoopEach - 1) / seqNumLoopEach;
        seqNumLoopTail = seqNumCore - (seqNumLoopTimes - 1) * seqNumLoopEach;
        seqNumGmIndexCore = coreIndexGroup * tiling->seqNumCoreEach; // 当前核在当前Seq Gm的偏移

        headDimLoopEach = tiling->headDimLoopEach;
        headDimLoopTimes = (dimD + headDimLoopEach - 1) / headDimLoopEach;
        headDimLoopTail = dimD - (headDimLoopTimes - 1) * headDimLoopEach;

        bnNumGroup = tiling->bnNumGroup;
        bnGmIndexCore = groupIndex * bnNumGroup;

        softmaxBnGmOffset = dimS * softmaxTailSize; // softmax在BN方向的维度每+1的实际GM偏移
        softmaxSeqNumGmOffset = softmaxTailSize;    // softmax在S方向的维度每+1的实际GM偏移

        attnBnGmOffset = dimD;             // attn在BN方向的维度每+1的实际GM偏移
        attnSeqNumGmOffset = bnNum * dimD; // attn在S方向的维度每+1的实际GM偏移

        softmaxEleNumLoop = seqNumLoopEach * softmaxTailSize; // prev和cur共用flag
        attnEleNumLoop = seqNumLoopEach * headDimLoopEach;    // prev和cur共用flag
    }

    __aicore__ inline void SoftmaxComputeLoop(int64_t seqNumLoopIndex, int64_t seqNumGmIndexLoop)
    {
        if (seqNumLoopIndex == (seqNumLoopTimes - 1)) {
            seqNumLoop = seqNumLoopTail;
            softmaxBlockLen = seqNumLoopTail * softmaxTailSize * floatDataSize;
        } else {
            seqNumLoop = seqNumLoopEach;
            softmaxBlockLen = seqNumLoopEach * softmaxTailSize * floatDataSize;
        }
        attnBlockCount = seqNumLoop;
        softmaxGmOffset = bnGmIndexLoop * softmaxBnGmOffset + seqNumGmIndexLoop * softmaxSeqNumGmOffset;
        SoftmaxDataMoveIn();
        SoftmaxComputeSBH();
        SoftmaxDataMoveOut();
    }

    __aicore__ inline void AttnComputeLoop(int64_t headDimLoopIndex, int64_t seqNumGmIndexLoop)
    {
        attnGmOffset = bnGmIndexLoop * attnBnGmOffset + seqNumGmIndexLoop * attnSeqNumGmOffset +
                       headDimLoopIndex * headDimLoopEach;
        if (headDimLoopIndex == (headDimLoopTimes - 1)) {
            headDimLoop = headDimLoopTail;
        } else {
            headDimLoop = headDimLoopEach;
        }

        attnBlockLen = headDimLoop * inputDataSize;
        attnGmStride = (bnNum * dimD - headDimLoop) * inputDataSize;
        attnUbStride = (headDimLoopEach - headDimLoop) / blockNumInput; // 32B为单位，UB补齐到headDimLoopEach
        AttnDataMoveIn();
        PipeBarrier<PIPE_V>();
        if constexpr (IsSameType<T, float>::value) {
            AttnComputeSBHFp32();
        } else {
            AttnComputeSBHBf16Fp16();
        }
        AttnDataMoveOut();
    }

    __aicore__ inline void SoftmaxDataMoveIn()
    {
        DataCopyExtParams softmaxCopyParams{1, softmaxBlockLen, 0, 0, 0};
        DataCopyPadExtParams<float> softmaxPadParams{false, 0, 0, 0};

        // softmax max move in
        prevCurSoftmaxMaxLocal = prevCurSoftmaxMaxInQueue.AllocTensor<float>();
        DataCopyPad(prevCurSoftmaxMaxLocal, prevSoftmaxMaxGm[softmaxGmOffset], softmaxCopyParams, softmaxPadParams);
        DataCopyPad(prevCurSoftmaxMaxLocal[softmaxEleNumLoop], curSoftmaxMaxGm[softmaxGmOffset], softmaxCopyParams,
                    softmaxPadParams);
        prevCurSoftmaxMaxInQueue.EnQue<float>(prevCurSoftmaxMaxLocal);

        // softmax sum move in
        prevCurSoftmaxSumLocal = prevCurSoftmaxSumInQueue.AllocTensor<float>();
        DataCopyPad(prevCurSoftmaxSumLocal[softmaxEleNumLoop], curSoftmaxSumGm[softmaxGmOffset], softmaxCopyParams,
                    softmaxPadParams);
        DataCopyPad(prevCurSoftmaxSumLocal, prevSoftmaxSumGm[softmaxGmOffset], softmaxCopyParams, softmaxPadParams);
        prevCurSoftmaxSumInQueue.EnQue<float>(prevCurSoftmaxSumLocal);
    }

    __aicore__ inline void SoftmaxComputeSBH()
    {
        prevCurSoftmaxMaxLocal = prevCurSoftmaxMaxInQueue.DeQue<float>();
        prevCurSoftmaxSumLocal = prevCurSoftmaxSumInQueue.DeQue<float>();
        softmaxMaxLocal = softmaxMaxOutQueue.AllocTensor<float>();
        softmaxSumLocal = softmaxSumOutQueue.AllocTensor<float>();

        SoftmaxCompute(prevCurSoftmaxMaxLocal, prevCurSoftmaxSumLocal, softmaxMaxLocal, softmaxSumLocal,
                       tempFp32Buf1Local, tempFp32Buf2Local, softmaxEleNumLoop, seqNumLoop * softmaxTailSize);

        prevCurSoftmaxMaxInQueue.FreeTensor<float>(prevCurSoftmaxMaxLocal);
        prevCurSoftmaxSumInQueue.FreeTensor<float>(prevCurSoftmaxSumLocal);
        softmaxMaxOutQueue.EnQue<float>(softmaxMaxLocal);
        softmaxSumOutQueue.EnQue<float>(softmaxSumLocal);
    }

    __aicore__ inline void SoftmaxDataMoveOut()
    {
        DataCopyExtParams copy_params{1, softmaxBlockLen, 0, 0, 0};
        // softmax max move out
        softmaxMaxLocal = softmaxMaxOutQueue.DeQue<float>();
        DataCopyPad(softmaxMaxGm[softmaxGmOffset], softmaxMaxLocal, copy_params);
        softmaxMaxOutQueue.FreeTensor<float>(softmaxMaxLocal);

        // softmax sum move out
        softmaxSumLocal = softmaxSumOutQueue.DeQue<float>();
        DataCopyPad(softmaxSumGm[softmaxGmOffset], softmaxSumLocal, copy_params);
        softmaxSumOutQueue.FreeTensor<float>(softmaxSumLocal);
    }

    __aicore__ inline void AttnDataMoveIn()
    {
        // attn move in
        DataCopyExtParams attnCopyParams{attnBlockCount, attnBlockLen, attnGmStride, attnUbStride, 0};
        DataCopyPadExtParams<T> attnPadParams{false, 0, 0, 0};
        prevCurAttnOutLocal = prevCurAttnOutInQueue.AllocTensor<T>();
        DataCopyPad(prevCurAttnOutLocal, prevAttnOutGm[attnGmOffset], attnCopyParams, attnPadParams);
        DataCopyPad(prevCurAttnOutLocal[attnEleNumLoop], curAttnOutGm[attnGmOffset], attnCopyParams, attnPadParams);
        prevCurAttnOutInQueue.EnQue<T>(prevCurAttnOutLocal);
    }

    __aicore__ inline void AttnComputeSBHBf16Fp16()
    {
        prevCurAttnOutLocal = prevCurAttnOutInQueue.DeQue<T>();
        attnOutLocal = attnOutOutQueue.AllocTensor<T>();

        AttnComputeBf16Fp16<T>(prevCurAttnOutLocal, attnOutLocal, tempFp32Buf1Local, tempFp32Buf2Local, attnEleNumLoop,
                               seqNumLoop, headDimLoop, headDimLoopEach, softmaxTailSize);

        prevCurAttnOutInQueue.FreeTensor<T>(prevCurAttnOutLocal);
        attnOutOutQueue.EnQue<T>(attnOutLocal);
    }

    __aicore__ inline void AttnComputeSBHFp32()
    {
        prevCurAttnOutLocal = prevCurAttnOutInQueue.DeQue<T>();
        attnOutLocal = attnOutOutQueue.AllocTensor<T>();

        AttnComputeFp32<T>(prevCurAttnOutLocal, attnOutLocal, tempFp32Buf1Local, tempFp32Buf2Local, attnEleNumLoop,
                           seqNumLoop, headDimLoop, headDimLoopEach, softmaxTailSize);

        prevCurAttnOutInQueue.FreeTensor<T>(prevCurAttnOutLocal);
        attnOutOutQueue.EnQue<T>(attnOutLocal);
    }

    __aicore__ inline void AttnDataMoveOut()
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
    GlobalTensor<int64_t> actualSeqQlenGm;
    GlobalTensor<T> prevAttnOutGm;
    GlobalTensor<T> curAttnOutGm;
    GlobalTensor<float> prevSoftmaxSumGm;
    GlobalTensor<float> prevSoftmaxMaxGm;
    GlobalTensor<float> curSoftmaxSumGm;
    GlobalTensor<float> curSoftmaxMaxGm;
    // define input global input gm buffer
    GlobalTensor<T> attnOutGM;
    GlobalTensor<float> softmaxSumGm;
    GlobalTensor<float> softmaxMaxGm;
    // define input queue
    TQue<QuePosition::VECIN, BUFFER_NUM> prevCurAttnOutInQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> prevCurSoftmaxMaxInQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> prevCurSoftmaxSumInQueue;
    // define temp buffer
    TBuf<TPosition::VECCALC> tempFp32Buf1;
    TBuf<TPosition::VECCALC> tempFp32Buf2;
    // define output queue
    TQue<QuePosition::VECOUT, BUFFER_NUM> attnOutOutQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> softmaxMaxOutQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> softmaxSumOutQueue;
    // define input ub tensor buffer
    LocalTensor<T> prevCurAttnOutLocal;
    LocalTensor<float> prevCurSoftmaxMaxLocal;
    LocalTensor<float> prevCurSoftmaxSumLocal;
    // define temp ub tensor buffer
    LocalTensor<float> tempFp32Buf1Local;
    LocalTensor<float> tempFp32Buf2Local;
    // define output ub tensor buffer
    LocalTensor<T> attnOutLocal;
    LocalTensor<float> softmaxMaxLocal;
    LocalTensor<float> softmaxSumLocal;
    // core info
    int64_t curBlockIdx;
    // shape info
    int64_t dimB;
    int64_t dimS;
    int64_t dimN;
    int64_t dimD;
    int64_t softmaxTailSize;
    int64_t bnNum;
    int64_t usedVectorCoreNum;

    // define loop data
    int64_t groupIndex;
    int64_t coreIndexGroup;

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
    uint32_t softmaxBlockLen;
    uint16_t attnBlockCount;
    uint32_t attnBlockLen;
    uint32_t attnGmStride;
    uint32_t attnUbStride;

    // compute info
    uint32_t inputDataSize;
    uint32_t floatDataSize;
    uint32_t blockNumInput;
};
} // namespace RingAttentionUpdateRegbase
#endif // _RING_ATTENTION_UPDATE_REGBASE_SBH_H_
