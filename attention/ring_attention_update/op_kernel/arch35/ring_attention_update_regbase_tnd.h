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
 * \file ring_attention_update_regbase_tnd.h
 * \brief
 */
#ifndef _RING_ATTENTION_UPDATE_REGBASE_TND_H_
#define _RING_ATTENTION_UPDATE_REGBASE_TND_H_
#include "kernel_operator.h"
#include "ring_attention_update_regbase_common.h"

namespace RingAttentionUpdateRegbase {

using namespace AscendC;

template <typename T>
class KernelRingAttentionUpdateRegbaseTND {
public:
    __aicore__ inline KernelRingAttentionUpdateRegbaseTND() {}
    __aicore__ inline void Init(GM_ADDR prevAttnOut, GM_ADDR prevSoftmaxMax, GM_ADDR prevSoftmaxSum, GM_ADDR curAttnOut,
                                GM_ADDR curSoftmaxMax, GM_ADDR curSoftmaxSum, GM_ADDR actualSeqQlen, GM_ADDR attnOut,
                                GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR workspace,
                                const RingAttentionUpdateRegbaseTNDTilingData *__restrict tiling, TPipe *tPipe)
    {
        // init input global gm buffer
        InitComputeInfo(tiling);
        prevAttnOutGm.SetGlobalBuffer((__gm__ T *)prevAttnOut);
        prevSoftmaxMaxGm.SetGlobalBuffer((__gm__ float *)prevSoftmaxMax);
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
        uint32_t bufferNumInQueue = 2; // cur和prev共用一个flagid
        tPipe->InitBuffer(prevCurAttnOutInQueue, BUFFER_NUM, attnEleNumLoop * bufferNumInQueue * inputDataSize);
        tPipe->InitBuffer(prevCurSoftmaxMaxInQueue, BUFFER_NUM, softmaxEleNumLoop * bufferNumInQueue * floatDataSize);
        tPipe->InitBuffer(prevCurSoftmaxSumInQueue, BUFFER_NUM, softmaxEleNumLoop * bufferNumInQueue * floatDataSize);

        // init output queue
        tPipe->InitBuffer(attnOutOutQueue, BUFFER_NUM, attnEleNumLoop * inputDataSize);
        tPipe->InitBuffer(softmaxMaxOutQueue, BUFFER_NUM, softmaxEleNumLoop * floatDataSize);
        tPipe->InitBuffer(softmaxSumOutQueue, BUFFER_NUM, softmaxEleNumLoop * floatDataSize);

        // init temp buffer
        tPipe->InitBuffer(tempFp32Buf1, softmaxEleNumLoop * floatDataSize);
        tPipe->InitBuffer(tempFp32Buf2, softmaxEleNumLoop * floatDataSize);

        tempFp32Buf1Local = tempFp32Buf1.Get<float>();
        tempFp32Buf2Local = tempFp32Buf2.Get<float>();

        if (seqNumLoopEach > 1) { // 搬入多个Seq，需要有额外UB空间做重排
            tPipe->InitBuffer(tempFp32Buf3, softmaxEleNumLoop * floatDataSize);
            tPipe->InitBuffer(tempFp32Buf4, softmaxEleNumLoop * floatDataSize);

            tempFp32Buf3Local = tempFp32Buf3.Get<float>();
            tempFp32Buf4Local = tempFp32Buf4.Get<float>();
        }
    }

    __aicore__ inline void Process()
    {
        if (curBlockIdx >= usedVectorCoreNum) {
            return;
        }
        if (seqNumLoopEach > 1) {
            ProcessTndMultiSeq();
        } else {
            ProcessTndOneSeq();
        }
    }

    // 一次搬多个Seq进UB，ND UB全载，需要做重排
    __aicore__ inline void ProcessTndMultiSeq()
    {
        for (int64_t batchSizeIndex = 0; batchSizeIndex < batchSize;
             batchSizeIndex++) { // 找到本核处理的dimT首次交集的batch
            seqNumBatchEndIdx = actualSeqQlenGm.GetValue(batchSizeIndex + 1);
            if (dimTStartIdxCore < seqNumBatchEndIdx) {
                curBatchIndex = batchSizeIndex; // 当前batch索引
                seqNumBatchStartIdx = actualSeqQlenGm.GetValue(batchSizeIndex);
                seqNumBatch = seqNumBatchEndIdx - seqNumBatchStartIdx; // 当前batch的Seq长度
                seqNumBatchTail =
                    seqNumBatch -
                    (dimTStartIdxCore - seqNumBatchStartIdx); // 当前batch剩余未处理的Seq，用于更新batch的Seq长度
                break;
            }
        }
        softmaxGmOffsetLoop = seqNumBatchStartIdx * dimN * softmaxTailSize; // 当前处理batch的起始位置在softmaxGM的偏移
        attnGmOffsetLoop = dimTStartIdxCore * attnTGmOffset; // 本核处理的起始位置在attnGM的偏移

        int64_t seqNumLoopIndex = 0;
        while (seqNumLoopIndex < dimTCore) { // Seq循环
            while (seqNumBatchTail == 0) {   // 是否更新batch的Seq，包含SeqNum=0的处理
                curBatchIndex = curBatchIndex + 1;
                seqNumBatchStartIdx = actualSeqQlenGm.GetValue(curBatchIndex);
                seqNumBatchEndIdx = actualSeqQlenGm.GetValue(curBatchIndex + 1);
                seqNumBatch = seqNumBatchEndIdx - seqNumBatchStartIdx;
                seqNumBatchTail = seqNumBatch;
                softmaxGmOffsetLoop =
                    seqNumBatchStartIdx * dimN * softmaxTailSize; // 更新当前batch之前的在softmaxGM的偏移
            }

            seqNumLoop =
                Min(seqNumBatchTail,
                    Min(seqNumLoopEach,
                        dimTCore - seqNumLoopIndex)); // 当前循环处理的Seq长度, 注意尾次长度和不超当前batch否则要更新Seq

            softmaxGmOffset =
                softmaxGmOffsetLoop + (dimTStartIdxCore - seqNumBatchStartIdx + seqNumLoopIndex) * softmaxTailSize;
            attnGmOffset = attnGmOffsetLoop + seqNumLoopIndex * attnTGmOffset;

            SoftmaxComputeLoopMultiSeq(seqNumBatch, seqNumLoop);
            AttnComputeLoopMultiSeq();

            seqNumBatchTail -= seqNumLoop; // 当前batch剩余未处理的Seq
            seqNumLoopIndex += seqNumLoop; // Seq每次偏移seqNumLoop长度
        }
    }

    // 一次只搬1个Seq进UB，ND做循环，无需做重排
    __aicore__ inline void ProcessTndOneSeq()
    {
        for (int64_t batchSizeIndex = 0; batchSizeIndex < batchSize;
             batchSizeIndex++) { // 找到本核处理的dimT首次交集的batch
            seqNumBatchEndIdx = actualSeqQlenGm.GetValue(batchSizeIndex + 1);
            if (dimTStartIdxCore < seqNumBatchEndIdx) {
                curBatchIndex = batchSizeIndex; // 当前batch索引
                seqNumBatchStartIdx = actualSeqQlenGm.GetValue(batchSizeIndex);
                seqNumBatch = seqNumBatchEndIdx - seqNumBatchStartIdx;   // 当前batch的Seq长度
                seqNumBatchPre = dimTStartIdxCore - seqNumBatchStartIdx; // 当前batch已经处理的Seq
                break;
            }
        }
        softmaxGmOffsetLoop = seqNumBatchStartIdx * dimN * softmaxTailSize; // 当前处理batch的起始位置在softmaxGM的偏移
        attnGmOffsetLoop = dimTStartIdxCore * attnTGmOffset; // 本核处理的起始位置在attnGM的偏移

        seqNumLoop = 1;                                                                    // 单次搬入1个Seq
        for (int64_t seqNumLoopIndex = 0; seqNumLoopIndex < dimTCore; seqNumLoopIndex++) { // 每个循环处理1个Seq
            while (seqNumBatchPre == seqNumBatch) { // 是否更新batch的Seq，包含SeqNum=0的边界处理
                curBatchIndex = curBatchIndex + 1;
                seqNumBatchStartIdx = actualSeqQlenGm.GetValue(curBatchIndex);
                seqNumBatchEndIdx = actualSeqQlenGm.GetValue(curBatchIndex + 1);
                seqNumBatch = seqNumBatchEndIdx - seqNumBatchStartIdx; // 当前batch的Seq长度
                seqNumBatchPre = 0;                                    // 当前batch已经处理的Seq
                softmaxGmOffsetLoop = seqNumBatchStartIdx * dimN * softmaxTailSize;
            }

            softmaxGmOffset =
                softmaxGmOffsetLoop + (dimTStartIdxCore - seqNumBatchStartIdx + seqNumLoopIndex) * softmaxTailSize;
            attnGmOffsetHeadNumLoop = attnGmOffsetLoop + seqNumLoopIndex * attnTGmOffset;

            for (int64_t headNumLoopIndex = 0; headNumLoopIndex < headNumLoopTimes; headNumLoopIndex++) {
                SoftmaxComputeLoopOneSeq(seqNumBatch, headNumLoopIndex);
                for (int64_t headDimLoopIndex = 0; headDimLoopIndex < headDimLoopTimes; headDimLoopIndex++) {
                    attnGmOffset = attnGmOffsetHeadNumLoop + headDimLoopIndex * headDimLoopEach;
                    AttnComputeLoopOneSeq(headDimLoopIndex);
                }

                softmaxGmOffset += headNumLoopEach * seqNumBatch * softmaxTailSize;
                attnGmOffsetHeadNumLoop += headNumLoopEach * dimD;
            }
            seqNumBatchPre += 1; // 当前batch已经处理的Seq + 1
        }
    }

private:
    __aicore__ inline int64_t Min(int64_t a, int64_t b)
    {
        return a < b ? a : b;
    }

    __aicore__ inline void InitComputeInfo(const RingAttentionUpdateRegbaseTNDTilingData *__restrict tiling)
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

        dimTCoreEach = tiling->dimTCoreEach;           // 每核处理的T方向行数
        dimTCoreTail = tiling->dimTCoreTail;           // 尾核处理的T方向行数
        usedVectorCoreNum = tiling->usedVectorCoreNum; // 使用的vector核数

        seqNumLoopEach = tiling->seqNumLoopEach; // 核内单次Seq方向搬入1长度

        headNumLoopEach = tiling->headNumLoopEach;                         // 单次UB循环headNum长度
        headNumLoopTimes = (dimN + headNumLoopEach - 1) / headNumLoopEach; // dimN方向循环次数
        headNumLoopTail = dimN - (headNumLoopTimes - 1) * headNumLoopEach; // dimN方向尾次循环长度

        headDimLoopEach = tiling->headDimLoopEach;                         // 单次UB循环headDim长度
        headDimLoopTimes = (dimD + headDimLoopEach - 1) / headDimLoopEach; // dimD方向循环次数
        headDimLoopTail = dimD - (headDimLoopTimes - 1) * headDimLoopEach; // dimD方向尾次循环长度

        softmaxSeqNumGmOffset = softmaxTailSize;

        attnTGmOffset = dimN * dimD;
        attnHeadNumGmOffset = dimD;

        // 本核处理
        dimTCore = curBlockIdx == usedVectorCoreNum - 1 ? dimTCoreTail : dimTCoreEach; // 本核处理的T方向行数
        dimTStartIdxCore = curBlockIdx * dimTCoreEach; // 本核处理的T起始行号(包)

        if (seqNumLoopEach > 1) {
            softmaxEleNumLoop = seqNumLoopEach * dimN * softmaxTailSize;
            attnEleNumLoop = seqNumLoopEach * dimN * headDimLoopEach;
        } else {
            softmaxEleNumLoop = headNumLoopEach * softmaxTailSize;
            attnEleNumLoop = headNumLoopEach * headDimLoopEach;
        }
    }

    __aicore__ inline void SoftmaxComputeLoopMultiSeq(int64_t seqNumBatch, int64_t seqNumLoop)
    {
        softmaxGmStride = (seqNumBatch - seqNumLoop) * softmaxTailSize * floatDataSize;
        softmaxBlockLen = seqNumLoop * softmaxTailSize * floatDataSize;
        headNumLoop = dimN;
        softmaxBlockCount = headNumLoop;
        attnBlockCount = seqNumLoop * headNumLoop;

        SoftmaxDataMoveIn();
        SoftmaxComputeTND();
        SoftmaxDataMoveOut();
    }

    __aicore__ inline void AttnComputeLoopMultiSeq()
    {
        headDimLoop = dimD;
        attnBlockLen = headDimLoop * inputDataSize;
        attnUbStride = (headDimLoopEach - headDimLoop) / blockNumInput; // 32B为单位，UB补齐到headDimLoopEach
        attnGmStride = 0;

        AttnDataMoveIn();
        PipeBarrier<PIPE_V>();
        if constexpr (IsSameType<T, float>::value) {
            AttnComputeTNDFp32();
        } else {
            AttnComputeTNDBf16Fp16();
        }
        AttnDataMoveOut();
    }

    __aicore__ inline void SoftmaxComputeLoopOneSeq(int64_t seqNumBatch, int64_t headNumLoopIndex)
    {
        softmaxGmStride = (seqNumBatch - 1) * softmaxTailSize * floatDataSize;
        headNumLoop = headNumLoopIndex == headNumLoopTimes - 1 ? headNumLoopTail : headNumLoopEach;
        softmaxBlockCount = headNumLoop;
        attnBlockCount = headNumLoop;
        softmaxBlockLen = softmaxTailSize * floatDataSize;

        SoftmaxDataMoveIn();
        SoftmaxComputeTND();
        SoftmaxDataMoveOut();
    }

    __aicore__ inline void AttnComputeLoopOneSeq(int64_t headDimLoopIndex)
    {
        headDimLoop = headDimLoopIndex == headDimLoopTimes - 1 ? headDimLoopTail : headDimLoopEach;
        attnBlockLen = headDimLoop * inputDataSize;
        attnGmStride = (dimD - headDimLoop) * inputDataSize;
        attnUbStride = (headDimLoopEach - headDimLoop) / blockNumInput; // 32B为单位，UB补齐到headDimLoopEach

        AttnDataMoveIn();
        PipeBarrier<PIPE_V>();
        if constexpr (IsSameType<T, float>::value) {
            AttnComputeTNDFp32();
        } else {
            AttnComputeTNDBf16Fp16();
        }
        AttnDataMoveOut();
    }

    __aicore__ inline void SoftmaxDataMoveIn()
    {
        DataCopyExtParams softmaxCopyParams{(uint16_t)softmaxBlockCount, softmaxBlockLen, softmaxGmStride, 0, 0};
        DataCopyPadExtParams<float> softmaxPadParams{false, 0, 0, 0};

        // softmax max move in
        prevCurSoftmaxMaxLocal = prevCurSoftmaxMaxInQueue.AllocTensor<float>();
        DataCopyPad(prevCurSoftmaxMaxLocal, prevSoftmaxMaxGm[softmaxGmOffset], softmaxCopyParams, softmaxPadParams);
        DataCopyPad(prevCurSoftmaxMaxLocal[softmaxEleNumLoop], curSoftmaxMaxGm[softmaxGmOffset], softmaxCopyParams,
                    softmaxPadParams);
        prevCurSoftmaxMaxInQueue.EnQue<float>(prevCurSoftmaxMaxLocal);

        // softmax sum move in
        prevCurSoftmaxSumLocal = prevCurSoftmaxSumInQueue.AllocTensor<float>();
        DataCopyPad(prevCurSoftmaxSumLocal, prevSoftmaxSumGm[softmaxGmOffset], softmaxCopyParams, softmaxPadParams);
        DataCopyPad(prevCurSoftmaxSumLocal[softmaxEleNumLoop], curSoftmaxSumGm[softmaxGmOffset], softmaxCopyParams,
                    softmaxPadParams);
        prevCurSoftmaxSumInQueue.EnQue<float>(prevCurSoftmaxSumLocal);
    }

    __aicore__ inline void AttnDataMoveIn()
    {
        // attn move in
        DataCopyExtParams attnCopyParams{(uint16_t)attnBlockCount, attnBlockLen, attnGmStride, attnUbStride, 0};
        DataCopyPadExtParams<T> attnPadParams{false, 0, 0, 0};
        prevCurAttnOutLocal = prevCurAttnOutInQueue.AllocTensor<T>();
        DataCopyPad(prevCurAttnOutLocal, prevAttnOutGm[attnGmOffset], attnCopyParams, attnPadParams);
        DataCopyPad(prevCurAttnOutLocal[attnEleNumLoop], curAttnOutGm[attnGmOffset], attnCopyParams, attnPadParams);
        prevCurAttnOutInQueue.EnQue<T>(prevCurAttnOutLocal);
    }

    __aicore__ inline void SoftmaxComputeTND()
    {
        prevCurSoftmaxMaxLocal = prevCurSoftmaxMaxInQueue.DeQue<float>();
        prevCurSoftmaxSumLocal = prevCurSoftmaxSumInQueue.DeQue<float>();
        softmaxMaxLocal = softmaxMaxOutQueue.AllocTensor<float>();
        softmaxSumLocal = softmaxSumOutQueue.AllocTensor<float>();

        SoftmaxCompute(prevCurSoftmaxMaxLocal, prevCurSoftmaxSumLocal, softmaxMaxLocal, softmaxSumLocal,
                       tempFp32Buf1Local, tempFp32Buf2Local, softmaxEleNumLoop,
                       seqNumLoop * headNumLoop * softmaxTailSize);

        if (seqNumLoopEach > 1) { // 搬入多个Seq，需要有额外UB空间做重排
            DataCopyParams copyParams{(uint16_t)headNumLoop, 1, (uint16_t)(seqNumLoop - 1), 0};
            for (int64_t seqLoop = 0; seqLoop < seqNumLoop; seqLoop++) {
                DataCopy(tempFp32Buf3Local[seqLoop * headNumLoop * softmaxTailSize],
                         tempFp32Buf1Local[seqLoop * softmaxTailSize], copyParams);
                DataCopy(tempFp32Buf4Local[seqLoop * headNumLoop * softmaxTailSize],
                         tempFp32Buf2Local[seqLoop * softmaxTailSize], copyParams);
            }
        }

        prevCurSoftmaxMaxInQueue.FreeTensor<float>(prevCurSoftmaxMaxLocal);
        prevCurSoftmaxSumInQueue.FreeTensor<float>(prevCurSoftmaxSumLocal);
        softmaxMaxOutQueue.EnQue<float>(softmaxMaxLocal);
        softmaxSumOutQueue.EnQue<float>(softmaxSumLocal);
    }

    __aicore__ inline void AttnComputeTNDBf16Fp16()
    {
        prevCurAttnOutLocal = prevCurAttnOutInQueue.DeQue<T>();
        attnOutLocal = attnOutOutQueue.AllocTensor<T>();

        if (seqNumLoopEach > 1) { // 搬入多个Seq，需要有额外UB空间做重排
            AttnComputeBf16Fp16<T>(prevCurAttnOutLocal, attnOutLocal, tempFp32Buf3Local, tempFp32Buf4Local,
                                   attnEleNumLoop, seqNumLoop * headNumLoop, headDimLoop, headDimLoopEach,
                                   softmaxTailSize);
        } else {
            AttnComputeBf16Fp16<T>(prevCurAttnOutLocal, attnOutLocal, tempFp32Buf1Local, tempFp32Buf2Local,
                                   attnEleNumLoop, seqNumLoop * headNumLoop, headDimLoop, headDimLoopEach,
                                   softmaxTailSize);
        }

        prevCurAttnOutInQueue.FreeTensor<T>(prevCurAttnOutLocal);
        attnOutOutQueue.EnQue<T>(attnOutLocal);
    }

    __aicore__ inline void AttnComputeTNDFp32()
    {
        prevCurAttnOutLocal = prevCurAttnOutInQueue.DeQue<T>();
        attnOutLocal = attnOutOutQueue.AllocTensor<T>();

        if (seqNumLoopEach > 1) { // 搬入多个Seq，需要有额外UB空间做重排
            AttnComputeFp32<T>(prevCurAttnOutLocal, attnOutLocal, tempFp32Buf3Local, tempFp32Buf4Local, attnEleNumLoop,
                               seqNumLoop * headNumLoop, headDimLoop, headDimLoopEach, softmaxTailSize);
        } else {
            AttnComputeFp32<T>(prevCurAttnOutLocal, attnOutLocal, tempFp32Buf1Local, tempFp32Buf2Local, attnEleNumLoop,
                               seqNumLoop * headNumLoop, headDimLoop, headDimLoopEach, softmaxTailSize);
        }

        prevCurAttnOutInQueue.FreeTensor<T>(prevCurAttnOutLocal);
        attnOutOutQueue.EnQue<T>(attnOutLocal);
    }

    __aicore__ inline void SoftmaxDataMoveOut()
    {
        // softmax max move out
        DataCopyExtParams copy_params{(uint16_t)softmaxBlockCount, softmaxBlockLen, 0, softmaxGmStride, 0};
        softmaxMaxLocal = softmaxMaxOutQueue.DeQue<float>();
        DataCopyPad(softmaxMaxGm[softmaxGmOffset], softmaxMaxLocal, copy_params);
        softmaxMaxOutQueue.FreeTensor<float>(softmaxMaxLocal);

        // softmax sum move out
        softmaxSumLocal = softmaxSumOutQueue.DeQue<float>();
        DataCopyPad(softmaxSumGm[softmaxGmOffset], softmaxSumLocal, copy_params);
        softmaxSumOutQueue.FreeTensor<float>(softmaxSumLocal);
    }

    __aicore__ inline void AttnDataMoveOut()
    {
        // attn move out
        DataCopyExtParams attnCopyParams{(uint16_t)attnBlockCount, attnBlockLen, attnUbStride, attnGmStride, 0};
        attnOutLocal = attnOutOutQueue.DeQue<T>();
        DataCopyPad(attnOutGM[attnGmOffset], attnOutLocal, attnCopyParams);
        attnOutOutQueue.FreeTensor<T>(attnOutLocal);
    }

    constexpr static uint32_t BLOCK_SIZE = 32;
    // buffer num: 1 or 2
    constexpr static int32_t BUFFER_NUM = 2;
    // define input global input gm buffer
    GlobalTensor<T> prevAttnOutGm;
    GlobalTensor<T> curAttnOutGm;
    GlobalTensor<float> prevSoftmaxMaxGm;
    GlobalTensor<float> prevSoftmaxSumGm;
    GlobalTensor<float> curSoftmaxMaxGm;
    GlobalTensor<float> curSoftmaxSumGm;
    GlobalTensor<int64_t> actualSeqQlenGm;
    // define output global output gm buffer
    GlobalTensor<T> attnOutGM;
    GlobalTensor<float> softmaxMaxGm;
    GlobalTensor<float> softmaxSumGm;
    // define input queue
    TQue<QuePosition::VECIN, BUFFER_NUM> prevCurAttnOutInQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> prevCurSoftmaxSumInQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> prevCurSoftmaxMaxInQueue;
    // define output queue
    TQue<QuePosition::VECOUT, BUFFER_NUM> attnOutOutQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> softmaxSumOutQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> softmaxMaxOutQueue;
    // define temp buffer
    TBuf<TPosition::VECCALC> tempFp32Buf1;
    TBuf<TPosition::VECCALC> tempFp32Buf2;
    TBuf<TPosition::VECCALC> tempFp32Buf3;
    TBuf<TPosition::VECCALC> tempFp32Buf4;
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
    LocalTensor<float> tempFp32Buf3Local;
    LocalTensor<float> tempFp32Buf4Local;
    // core info
    int64_t curBlockIdx;
    // shape info
    int64_t dimT;
    int64_t dimN;
    int64_t dimD;
    int64_t softmaxTailSize;
    int64_t batchSize;

    // define loop data
    int64_t dimTCoreEach;
    int64_t dimTCoreTail;
    int64_t usedVectorCoreNum;

    int64_t seqNumLoopEach;
    int64_t headNumLoopEach;
    int64_t headNumLoopTimes;
    int64_t headNumLoopTail;
    int64_t headNumLoop;

    int64_t headDimLoopEach;
    int64_t headDimLoopTimes;
    int64_t headDimLoopTail;
    int64_t headDimLoop;

    // core compute
    int64_t attnEleNumLoop;
    int64_t softmaxEleNumLoop;

    int64_t dimTCore;
    int64_t dimTStartIdxCore;
    int64_t curBatchIndex;
    int64_t seqNumBatchStartIdx;
    int64_t seqNumBatchEndIdx;
    int64_t seqNumBatch;
    int64_t seqNumBatchTail;
    int64_t seqNumBatchPre;
    int64_t seqNumLoop;

    uint16_t attnBlockCount;
    uint32_t attnBlockLen;
    uint32_t attnGmStride;
    uint32_t attnUbStride;
    uint16_t softmaxBlockCount;
    uint32_t softmaxBlockLen;
    uint32_t softmaxGmStride;

    // gm offset
    int64_t attnGmOffsetLoop;
    int64_t softmaxGmOffsetLoop;
    int64_t softmaxGmOffset;
    int64_t attnGmOffset;
    int64_t attnGmOffsetHeadNumLoop;
    int64_t attnTGmOffset;
    int64_t attnHeadNumGmOffset;
    int64_t softmaxSeqNumGmOffset;

    // compute info
    uint32_t inputDataSize;
    uint32_t floatDataSize;
    uint32_t blockNumInput;
};
} // namespace RingAttentionUpdateRegbase
#endif // _RING_ATTENTION_UPDATE_REGBASE_TND_H_
