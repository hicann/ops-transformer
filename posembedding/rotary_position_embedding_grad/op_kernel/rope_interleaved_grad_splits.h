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
 * \file rope_interleaved_grad_splits.h
 * \brief
 */
#ifndef ROPE_INTERLEAVED_GRAD_SPLITS_H
#define ROPE_INTERLEAVED_GRAD_SPLITS_H
#include "rope_interleaved_grad_common.h"

using namespace AscendC;

template <typename T, bool LARGE, bool NEEDBACKWARD>
class RopeInterleavedGrad {
public:
    __aicore__ inline RopeInterleavedGrad(){};
    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR cos, GM_ADDR sin, GM_ADDR x, GM_ADDR xGrad, GM_ADDR cosGrad,
                                GM_ADDR sinGrad, const RotaryPositionEmbeddingGradTilingData &tiling, TPipe *pipe);
    __aicore__ inline void InitTilingData(const RotaryPositionEmbeddingGradTilingData &tiling);
    __aicore__ inline void TensorMul(LocalTensor<float> &ansTensor, LocalTensor<float> &tempTensor,
                                     LocalTensor<float> &mulTensor, uint32_t dataNum);
    __aicore__ inline void Process();

    __aicore__ inline void SmallProcess();
    __aicore__ inline void SmallCopyIn(uint64_t loopIdx);
    __aicore__ inline void SmallCompute();
    __aicore__ inline void SmallCopyOut(uint64_t loopIdx);

    __aicore__ inline void CalculateDx();
    __aicore__ inline void CalculateDcos(const LocalTensor<T> &outCos, uint32_t loop_num);
    __aicore__ inline void CalculateDsin(const LocalTensor<T> &outSin, uint32_t loop_num);

    __aicore__ inline void LargeProcess();
    __aicore__ inline void LargeCosSinInit();
    __aicore__ inline void LargeCopyIn(uint64_t gmOffset, uint64_t sIndex);
    __aicore__ inline void LargeCompute(uint64_t innerLoopIndex, uint64_t outerLoopIndex);
    __aicore__ inline void LargeXGradCopyOut(uint64_t gmOffset);
    __aicore__ inline void LargeCosSinCopyOut(uint64_t sIndex);

    __aicore__ inline void BroadCastToBsnd(LocalTensor<T> &src, LocalTensor<T> &dst, uint32_t calcLen);
    __aicore__ inline void ReduceToBsnd(LocalTensor<float> &x, LocalTensor<float> &y, uint32_t calcLen);
    __aicore__ inline void BroadCastToSbnd(LocalTensor<T> &src, LocalTensor<T> &dst, uint32_t calcLen);
    __aicore__ inline void ReduceToSbnd(LocalTensor<float> &x, LocalTensor<float> &tri, uint32_t calcLen);
    __aicore__ inline void BroadCastToBnsd(LocalTensor<T> &src, uint32_t calcLen);

protected:
    TPipe pipe;
    GlobalTensor<T> gradGm, cosGm, sinGm, xGm, xGradGm, cosGradGm, sinGradGm;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueGrad, inQueCosSin, inQueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueXGrad, outQueueCosSinGrad;
    TBuf<TPosition::VECCALC> calcBuf, calcBuf2, forwardMulBuf, backwardMulBuf, gatherBuf;
    TBuf<TPosition::VECCALC> inputGradFloatBuf, inputCosSinFloatBuf, inputXFloatBuf, outDxTensorFloatBuf,
        outCosSinFloatBuf, outCosSinGradBuf, broadCastBuf;

    // tilingdata
    uint64_t batchSize;
    uint64_t seqLen;
    uint64_t numHeads;
    uint64_t headDim;
    uint64_t alignHeadDim;
    uint64_t padHeadDim;

    // core split
    uint64_t frontCoreNum;
    uint64_t tailCoreNum;
    uint64_t seqFrontLen;
    uint64_t seqTailLen;
    uint64_t seqCoreLen;

    // front seq split
    uint64_t seqFrontCalcNum;
    uint64_t seqFrontCalcLoop;
    uint64_t seqFrontCalcTail;

    // tail seq spilt
    uint64_t seqTailCalcNum;
    uint64_t seqTailCalcLoop;
    uint64_t seqTailCalcTail;

    // split numHeads
    uint64_t numHeadsLength;
    uint64_t numHeadsLoop;
    uint64_t numHeadsTail;

    // split batchNumHeads
    uint64_t batchNumHeadsLength;
    uint64_t batchNumHeadsLoop;
    uint64_t batchNumHeadsTail;

    uint64_t innerLength;
    uint64_t innerLoop;
    uint64_t innerTail;
    uint64_t outerLoop;

    uint64_t blockIdx;
    uint64_t layout;

    // core inner arguments
    uint64_t coreInnerseqCalcNum;
    uint64_t coreInnerseqCalcLoop;
    uint64_t coreInnerseqCalcTail;

    uint64_t bsndSize;
    uint64_t sdSize;
    uint64_t bufferXSize;
    uint64_t buffercosSize;

    // absolute offset
    uint64_t xOffset;
    uint64_t cosOffset;
    uint64_t seqOffset;

    // pad params
    DataCopyPadExtParams<T> padParams;

    // LocalTensor defination
    LocalTensor<T> inputGrad;
    LocalTensor<T> inputCosSin;
    LocalTensor<T> inputCos;
    LocalTensor<T> inputSin;
    LocalTensor<T> inputX;
    // bsnd
    LocalTensor<T> broadCastTmp;
    LocalTensor<float> broadCastFloatTmp;

    // float calculate tensor
    LocalTensor<float> inputGradFloat;
    LocalTensor<float> inputCosSinFloat;
    LocalTensor<float> inputCosFloat;
    LocalTensor<float> inputSinFloat;
    LocalTensor<float> inputXFloat;
    LocalTensor<float> outputCosSinFloat;
    LocalTensor<float> outputCosFloat;
    LocalTensor<float> outputSinFloat;
    LocalTensor<float> outDxTensorFloat;

    LocalTensor<float> outCosSinTensorFloat;
    LocalTensor<float> outCosFloat;
    LocalTensor<float> outSinFloat;

    // temp calc tensor
    LocalTensor<float> calcTensor;
    LocalTensor<float> calcTensor2;

    // gather forward/backward tensor
    LocalTensor<float> forwardMulTensor;
    LocalTensor<float> backwardMulTensor;
    LocalTensor<int32_t> gatherTensor;

    // layout
    uint32_t layoutBSND = 0;
    uint32_t layoutBNSD = 1;
    uint32_t layoutSBND = 2;

    uint64_t dataNum;
    uint64_t calcSeq;
    uint64_t calcBn;

    uint64_t inputSinOffset;
    uint64_t outputSinOffset;
    RoundMode roundMode;

    // split bn fp16/bf16 condition
    LocalTensor<T> outCosSinGradTemp;
    LocalTensor<T> outCosGradTemp;
    LocalTensor<T> outSinGradTemp;
};

template <typename T, bool LARGE, bool NEEDBACKWARD>
__aicore__ inline void RopeInterleavedGrad<T, LARGE, NEEDBACKWARD>::BroadCastToBnsd(LocalTensor<T> &src,
                                                                                    uint32_t calcLen)
{
    // broadcast sd -> bnsd
    DataCopyParams bnsdParams;
    bnsdParams.blockCount = static_cast<uint16_t>(1);
    bnsdParams.blockLen = static_cast<uint16_t>(calcLen * alignHeadDim * sizeof(T) / BLOCK_SIZE);
    bnsdParams.srcStride = 0;
    bnsdParams.dstStride = 0;
    for (uint32_t idx = 0; idx < batchSize * numHeads; idx++) {
        DataCopy(src[idx * calcLen * alignHeadDim], src, bnsdParams);
    }
}

template <typename T, bool LARGE, bool NEEDBACKWARD>
__aicore__ inline void RopeInterleavedGrad<T, LARGE, NEEDBACKWARD>::BroadCastToSbnd(LocalTensor<T> &src,
                                                                                    LocalTensor<T> &dst,
                                                                                    uint32_t calcLen)
{
    // broadcast sd -> sbnd
    DataCopyParams intriParams;
    intriParams.blockCount = static_cast<uint16_t>(calcLen);
    intriParams.blockLen = static_cast<uint16_t>(alignHeadDim * sizeof(T) / BLOCK_SIZE);
    intriParams.srcStride = 0;
    intriParams.dstStride = static_cast<uint16_t>((batchSize * numHeads - 1) * alignHeadDim * sizeof(T) / BLOCK_SIZE);
    for (uint32_t bn_idx = 0; bn_idx < batchSize * numHeads; bn_idx++) {
        DataCopy(dst[bn_idx * alignHeadDim], src, intriParams);
        PipeBarrier<PIPE_V>();
    }
    DataCopy(src, dst, uint32_t(calcLen * batchSize * numHeads * alignHeadDim));
}

template <typename T, bool LARGE, bool NEEDBACKWARD>
__aicore__ inline void RopeInterleavedGrad<T, LARGE, NEEDBACKWARD>::BroadCastToBsnd(LocalTensor<T> &src,
                                                                                    LocalTensor<T> &dst,
                                                                                    uint32_t calcLen)
{
    DataCopyParams intriParams;
    intriParams.blockCount = static_cast<uint16_t>(calcLen);
    intriParams.blockLen = static_cast<uint16_t>(alignHeadDim * sizeof(T) / BLOCK_SIZE);
    intriParams.srcStride = 0;
    intriParams.dstStride = static_cast<uint16_t>((numHeads - 1) * alignHeadDim * sizeof(T) / BLOCK_SIZE);
    for (uint32_t numHeadsIdx = 0; numHeadsIdx < numHeads; numHeadsIdx++) {
        DataCopy(dst[numHeadsIdx * alignHeadDim], src, intriParams);
        PipeBarrier<PIPE_V>();
    }
    intriParams.blockCount = 1;
    intriParams.blockLen = static_cast<uint16_t>(calcLen * numHeads * alignHeadDim * sizeof(T) / BLOCK_SIZE);
    intriParams.srcStride = 0;
    intriParams.dstStride = 0;
    for (uint32_t batchIdx = 0; batchIdx < batchSize; batchIdx++) {
        DataCopy(src[batchIdx * calcLen * numHeads * alignHeadDim], dst, intriParams);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T, bool LARGE, bool NEEDBACKWARD>
__aicore__ inline void RopeInterleavedGrad<T, LARGE, NEEDBACKWARD>::ReduceToSbnd(LocalTensor<float> &x,
                                                                                 LocalTensor<float> &y,
                                                                                 uint32_t calcLen)
{
    // reduce sbnd -> sd
    for (uint32_t sIndex = 0; sIndex < calcLen; sIndex++) {
        for (uint32_t bnIndex = 0; bnIndex < batchSize * numHeads; bnIndex++) {
            uint32_t offset = sIndex * batchSize * numHeads * alignHeadDim + bnIndex * alignHeadDim;
            if (bnIndex == 0) {
                DataCopy(y[sIndex * alignHeadDim], x[offset], uint32_t(alignHeadDim));
            } else {
                Add(y[sIndex * alignHeadDim], y[sIndex * alignHeadDim], x[offset], uint32_t(alignHeadDim));
            }
            PipeBarrier<PIPE_V>();
        }
    }
    DataCopy(x, y, uint32_t(calcLen * alignHeadDim));
}

template <typename T, bool LARGE, bool NEEDBACKWARD>
__aicore__ inline void RopeInterleavedGrad<T, LARGE, NEEDBACKWARD>::ReduceToBsnd(LocalTensor<float> &x,
                                                                                 LocalTensor<float> &y,
                                                                                 uint32_t calcLen)
{
    for (uint32_t batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        if (batchIndex == 0) {
            DataCopy(y, x, uint32_t(calcLen * numHeads * alignHeadDim));
        } else {
            uint32_t offset = batchIndex * calcLen * numHeads * alignHeadDim;
            Add(y, y, x[offset], uint32_t(calcLen * numHeads * alignHeadDim));
        }
        PipeBarrier<PIPE_V>();
    }
    for (uint32_t sIndex = 0; sIndex < calcLen; sIndex++) {
        for (uint32_t numIndex = 0; numIndex < numHeads; numIndex++) {
            uint32_t offset = sIndex * numHeads * alignHeadDim + numIndex * alignHeadDim;
            if (numIndex == 0) {
                DataCopy(x[sIndex * alignHeadDim], y[offset], uint32_t(alignHeadDim));
            } else {
                Add(x[sIndex * alignHeadDim], x[sIndex * alignHeadDim], y[offset], uint32_t(alignHeadDim));
            }
            PipeBarrier<PIPE_V>();
        }
    }
}

template <typename T, bool LARGE, bool NEEDBACKWARD>
__aicore__ inline void RopeInterleavedGrad<T, LARGE, NEEDBACKWARD>::TensorMul(LocalTensor<float> &ansTensor,
                                                                              LocalTensor<float> &tempTensor,
                                                                              LocalTensor<float> &mulTensor,
                                                                              uint32_t dataNum)
{
    if (dataNum % MASK_FP32 != 0) {
        uint8_t repeatTimes = dataNum / MASK_FP32;
        Mul(ansTensor, tempTensor, mulTensor, uint64_t(MASK_FP32), repeatTimes, {1, 1, 0, 8, 8, 0});
        PipeBarrier<PIPE_V>();
        uint8_t left_num = (dataNum % MASK_FP32) / 8;
        Mul(ansTensor[repeatTimes * MASK_FP32], tempTensor[repeatTimes * MASK_FP32], mulTensor, uint64_t(8), left_num,
            {1, 1, 0, 1, 1, 0});
    } else {
        Mul(ansTensor, tempTensor, mulTensor, uint64_t(MASK_FP32), uint8_t(dataNum / 64), {1, 1, 0, 8, 8, 0});
    }
}

template <typename T, bool LARGE, bool NEEDBACKWARD>
__aicore__ inline void RopeInterleavedGrad<T, LARGE, NEEDBACKWARD>::Init(
    GM_ADDR grad, GM_ADDR cos, GM_ADDR sin, GM_ADDR x, GM_ADDR xGrad, GM_ADDR cosGrad, GM_ADDR sinGrad,
    const RotaryPositionEmbeddingGradTilingData &tiling, TPipe *pipe)
{
    InitTilingData(tiling);
    blockIdx = GetBlockIdx();

    if (blockIdx < frontCoreNum) {
        coreInnerseqCalcNum = seqFrontCalcNum;
        coreInnerseqCalcLoop = seqFrontCalcLoop;
        coreInnerseqCalcTail = seqFrontCalcTail;
        if (layout == layoutBSND) {
            xOffset = blockIdx * seqFrontLen * numHeads * headDim;
        } else if (layout == layoutBNSD) {
            xOffset = blockIdx * seqFrontLen * headDim;
        } else if (layout == layoutSBND) {
            xOffset = blockIdx * batchSize * seqFrontLen * numHeads * headDim;
        }
        cosOffset = blockIdx * seqFrontLen * headDim;
        seqCoreLen = seqFrontLen;
        seqOffset = blockIdx * seqFrontLen;
    } else {
        coreInnerseqCalcNum = seqTailCalcNum;
        coreInnerseqCalcLoop = seqTailCalcLoop;
        coreInnerseqCalcTail = seqTailCalcTail;
        if (layout == layoutBSND) {
            xOffset = frontCoreNum * seqFrontLen * numHeads * headDim +
                      (blockIdx - frontCoreNum) * seqTailLen * numHeads * headDim;
        } else if (layout == layoutBNSD) {
            xOffset = frontCoreNum * seqFrontLen * headDim + (blockIdx - frontCoreNum) * seqTailLen * headDim;
        } else if (layout == layoutSBND) {
            xOffset = frontCoreNum * seqFrontLen * batchSize * numHeads * headDim +
                      (blockIdx - frontCoreNum) * seqTailLen * batchSize * numHeads * headDim;
        }
        cosOffset = frontCoreNum * seqFrontLen * headDim + (blockIdx - frontCoreNum) * seqTailLen * headDim;
        seqCoreLen = seqTailLen;
        seqOffset = frontCoreNum * seqFrontLen + (blockIdx - frontCoreNum) * seqTailLen;
    }
    bsndSize = batchSize * seqLen * numHeads * headDim;
    sdSize = seqLen * headDim;
    if (numHeadsLoop == 0 && batchNumHeadsLoop == 0) {
        gradGm.SetGlobalBuffer((__gm__ T *)grad + xOffset, bsndSize);
        cosGm.SetGlobalBuffer((__gm__ T *)cos + cosOffset, sdSize);
        sinGm.SetGlobalBuffer((__gm__ T *)sin + cosOffset, sdSize);
        xGm.SetGlobalBuffer((__gm__ T *)x + xOffset, bsndSize);
        xGradGm.SetGlobalBuffer((__gm__ T *)xGrad + xOffset, bsndSize);
        cosGradGm.SetGlobalBuffer((__gm__ T *)cosGrad + cosOffset, sdSize);
        sinGradGm.SetGlobalBuffer((__gm__ T *)sinGrad + cosOffset, sdSize);
    } else {
        gradGm.SetGlobalBuffer((__gm__ T *)grad, bsndSize);
        cosGm.SetGlobalBuffer((__gm__ T *)cos, sdSize);
        sinGm.SetGlobalBuffer((__gm__ T *)sin, sdSize);
        xGm.SetGlobalBuffer((__gm__ T *)x, bsndSize);
        xGradGm.SetGlobalBuffer((__gm__ T *)xGrad, bsndSize);
        cosGradGm.SetGlobalBuffer((__gm__ T *)cosGrad, sdSize);
        sinGradGm.SetGlobalBuffer((__gm__ T *)sinGrad, sdSize);
    }

    // calc compute buffer inputSinOffset
    if (numHeadsLoop == 0 && batchNumHeadsLoop == 0) {
        bufferXSize = coreInnerseqCalcNum * batchSize * numHeads * alignHeadDim * sizeof(T);
        buffercosSize = coreInnerseqCalcNum * alignHeadDim * sizeof(T);
    } else if (numHeadsLoop > 0) {
        bufferXSize = numHeadsLength * alignHeadDim * sizeof(T);
        buffercosSize = alignHeadDim * sizeof(T);
    } else if (batchNumHeadsLoop > 0) {
        bufferXSize = batchNumHeadsLength * alignHeadDim * sizeof(T);
        buffercosSize = alignHeadDim * sizeof(T);
    }

    // total buffer length
    pipe->InitBuffer(inQueGrad, BUFFER_NUM, bufferXSize);
    pipe->InitBuffer(inQueCosSin, BUFFER_NUM, 2 * bufferXSize);
    pipe->InitBuffer(inQueX, BUFFER_NUM, bufferXSize);
    pipe->InitBuffer(outQueueXGrad, BUFFER_NUM, bufferXSize);
    if constexpr (LARGE && (std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value)) {
        pipe->InitBuffer(outQueueCosSinGrad, BUFFER_NUM, 2 * buffercosSize * 2);
        pipe->InitBuffer(outCosSinGradBuf, 2 * buffercosSize);
        outCosSinGradTemp = outCosSinGradBuf.Get<T>();
        outCosGradTemp = outCosSinGradTemp[0];
        outSinGradTemp = outCosSinGradTemp[buffercosSize / sizeof(T)];
    } else {
        pipe->InitBuffer(outQueueCosSinGrad, BUFFER_NUM, 2 * buffercosSize);
    }

    pipe->InitBuffer(forwardMulBuf, BLOCK_SIZE);
    pipe->InitBuffer(backwardMulBuf, BLOCK_SIZE);

    // temp buffer
    // fp16 special buf
    if constexpr (std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value) {
        pipe->InitBuffer(gatherBuf, bufferXSize * 2);
        pipe->InitBuffer(calcBuf, bufferXSize * 2);
        pipe->InitBuffer(calcBuf2, bufferXSize * 2);

        pipe->InitBuffer(inputGradFloatBuf, bufferXSize * 2);
        // combine
        pipe->InitBuffer(inputCosSinFloatBuf, bufferXSize * 2 * 2);
        pipe->InitBuffer(inputXFloatBuf, bufferXSize * 2);
        pipe->InitBuffer(outDxTensorFloatBuf, bufferXSize * 2);
        pipe->InitBuffer(outCosSinFloatBuf, buffercosSize * 2 * 2);

        inputGradFloat = inputGradFloatBuf.Get<float>();
        inputCosSinFloat = inputCosSinFloatBuf.Get<float>();
        inputCosFloat = inputCosSinFloat[0];
        inputSinFloat = inputCosSinFloat[bufferXSize / sizeof(T)];
        inputXFloat = inputXFloatBuf.Get<float>();

        outputCosSinFloat = outCosSinFloatBuf.Get<float>();
        outputCosFloat = outputCosSinFloat[0];
        outputSinFloat = outputCosSinFloat[buffercosSize / sizeof(T)];

        outDxTensorFloat = outDxTensorFloatBuf.Get<float>();
    } else {
        pipe->InitBuffer(gatherBuf, bufferXSize);
        pipe->InitBuffer(calcBuf, bufferXSize);
        pipe->InitBuffer(calcBuf2, bufferXSize);
    }

    if (layout == layoutBSND || layout == layoutSBND) {
        if constexpr (std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value) {
            pipe->InitBuffer(broadCastBuf, bufferXSize * 2);
        } else {
            pipe->InitBuffer(broadCastBuf, bufferXSize);
        }
        broadCastTmp = broadCastBuf.Get<T>();
        broadCastFloatTmp = broadCastBuf.Get<float>();
    }

    if constexpr (std::is_same<T, half>::value) {
        roundMode = RoundMode::CAST_NONE;
    } else if constexpr (std::is_same<T, bfloat16_t>::value) {
        roundMode = RoundMode::CAST_RINT;
    }
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.rightPadding = alignHeadDim - headDim;
    padParams.paddingValue = 0;

    // gatherbuf generate
    calcTensor = calcBuf.Get<float>();
    calcTensor2 = calcBuf2.Get<float>();
    forwardMulTensor = forwardMulBuf.Get<float>();
    backwardMulTensor = backwardMulBuf.Get<float>();
    gatherTensor = gatherBuf.Get<int32_t>();

    // three split conditon
    if (numHeadsLoop == 0 && batchNumHeadsLoop == 0) {
        dataNum = batchSize * coreInnerseqCalcNum * numHeads * alignHeadDim;
        calcSeq = coreInnerseqCalcNum;
    } else if (batchNumHeadsLoop > 0) {
        dataNum = batchNumHeadsLength * alignHeadDim;
        calcBn = batchNumHeadsLength;
        innerLength = batchNumHeadsLength;
        innerLoop = batchNumHeadsLoop;
        innerTail = batchNumHeadsTail;
        outerLoop = 1;
        calcSeq = 1;
    } else if (numHeadsLoop > 0) {
        dataNum = numHeadsLength * alignHeadDim;
        innerLength = numHeadsLength;
        innerLoop = numHeadsLoop;
        innerTail = numHeadsTail;
        outerLoop = batchSize;
        calcSeq = 1;
    }

    inputSinOffset = bufferXSize / sizeof(T);
    outputSinOffset = buffercosSize / sizeof(T);

    for (uint32_t idx = 0; idx < BLOCK_FP32_NUM; idx++) {
        uint32_t pos = idx % 2 == 0 ? (idx + 1) : (idx - 1);
        gatherTensor.SetValue(idx, pos * 4);
        backwardMulTensor.SetValue(idx, idx % 2 == 0 ? 1 : -1);
        forwardMulTensor.SetValue(idx, idx % 2 == 0 ? -1 : 1);
    }
    PipeBarrier<PIPE_V>();

    uint64_t gatherLength = bufferXSize / sizeof(T);
    uint32_t loop = 0;
    uint32_t binaryNum = BLOCK_FP32_NUM;
    for (loop = 0; 2 * binaryNum <= gatherLength; loop++, binaryNum *= 2) {
        uint64_t mask = 1 << (LOG_BLOCK_FP32_NUM + loop);
        if (mask > MASK_FP32)
            mask = MASK_FP32;
        uint32_t beginPos = 1 << (LOG_BLOCK_FP32_NUM + loop);
        uint32_t scalar = 1 << (LOG_FP32_SIZE + LOG_BLOCK_FP32_NUM + loop);
        uint32_t calcNum = beginPos;
        uint32_t repeatTimes = (calcNum > MASK_FP32) ? (calcNum / MASK_FP32) : 1;
        Adds(gatherTensor[beginPos], gatherTensor, static_cast<int32_t>(scalar), static_cast<uint64_t>(mask),
             static_cast<int32_t>(repeatTimes), {1, 1, 8, 8});
        PipeBarrier<PIPE_V>();
    }

    uint32_t leftLength = gatherLength - binaryNum;
    if (leftLength > 0) {
        uint32_t beginPos = 1 << (LOG_BLOCK_FP32_NUM + loop);
        uint32_t scalar = 1 << (LOG_FP32_SIZE + LOG_BLOCK_FP32_NUM + loop);
        uint32_t repeatTimes = leftLength / MASK_FP32;
        Adds(gatherTensor[beginPos], gatherTensor, static_cast<int32_t>(scalar), static_cast<uint64_t>(MASK_FP32),
             static_cast<int32_t>(repeatTimes), {1, 1, 8, 8});
        leftLength -= repeatTimes * MASK_FP32;
        PipeBarrier<PIPE_V>();
    }
    if (leftLength <= 64 && leftLength % 8 == 0) {
        Adds(gatherTensor[gatherLength - leftLength], gatherTensor[gatherLength - 2 * leftLength],
             static_cast<int32_t>(leftLength * sizeof(float)), static_cast<uint64_t>(leftLength), 1, {1, 1, 0, 0});
        PipeBarrier<PIPE_V>();
    }
}

template <typename T, bool LARGE, bool NEEDBACKWARD>
__aicore__ inline void RopeInterleavedGrad<T, LARGE, NEEDBACKWARD>::InitTilingData(
    const RotaryPositionEmbeddingGradTilingData &tiling)
{
    const RopeInterleavedGradParams &ropeInterleavedGradTiling = tiling.ropeInterleavedGradParams;
    batchSize = ropeInterleavedGradTiling.batchSize;
    seqLen = ropeInterleavedGradTiling.seqLen;
    numHeads = ropeInterleavedGradTiling.numHeads;
    headDim = ropeInterleavedGradTiling.headDim;
    alignHeadDim = ropeInterleavedGradTiling.alignHeadDim;
    padHeadDim = ropeInterleavedGradTiling.padHeadDim;

    // split whole s
    frontCoreNum = ropeInterleavedGradTiling.frontCoreNum;
    tailCoreNum = ropeInterleavedGradTiling.tailCoreNum;
    seqFrontLen = ropeInterleavedGradTiling.seqFrontLen;
    seqTailLen = ropeInterleavedGradTiling.seqTailLen;

    // split front s
    seqFrontCalcNum = ropeInterleavedGradTiling.seqFrontCalcNum;
    seqFrontCalcLoop = ropeInterleavedGradTiling.seqFrontCalcLoop;
    seqFrontCalcTail = ropeInterleavedGradTiling.seqFrontCalcTail;

    // split tail s
    seqTailCalcNum = ropeInterleavedGradTiling.seqTailCalcNum;
    seqTailCalcLoop = ropeInterleavedGradTiling.seqTailCalcLoop;
    seqTailCalcTail = ropeInterleavedGradTiling.seqTailCalcTail;

    // split numHeads
    numHeadsLength = ropeInterleavedGradTiling.numHeadsLength;
    numHeadsLoop = ropeInterleavedGradTiling.numHeadsLoop;
    numHeadsTail = ropeInterleavedGradTiling.numHeadsTail;

    // split batchNumHeads
    batchNumHeadsLength = ropeInterleavedGradTiling.batchNumHeadsLength;
    batchNumHeadsLoop = ropeInterleavedGradTiling.batchNumHeadsLoop;
    batchNumHeadsTail = ropeInterleavedGradTiling.batchNumHeadsTail;

    layout = ropeInterleavedGradTiling.layout;
}

template <typename T, bool LARGE, bool NEEDBACKWARD>
__aicore__ inline void RopeInterleavedGrad<T, LARGE, NEEDBACKWARD>::SmallCopyIn(uint64_t loopIdx)
{
    LocalTensor<T> grad = inQueGrad.AllocTensor<T>();
    LocalTensor<T> cosSin = inQueCosSin.AllocTensor<T>();
    LocalTensor<T> cos = cosSin[0];
    LocalTensor<T> sin = cosSin[inputSinOffset];

    DataCopyExtParams dataCopyCosParams;
    dataCopyCosParams.blockCount = calcSeq;
    dataCopyCosParams.blockLen = headDim * sizeof(T);
    dataCopyCosParams.srcStride = 0;
    dataCopyCosParams.dstStride = 0;
    DataCopyPad(cos, cosGm[loopIdx * coreInnerseqCalcNum * headDim], dataCopyCosParams, padParams);
    DataCopyPad(sin, sinGm[loopIdx * coreInnerseqCalcNum * headDim], dataCopyCosParams, padParams);
    event_t eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);

    // copy in grad
    DataCopyExtParams dataCopyXParams;
    if (layout == layoutBSND) {
        dataCopyXParams.blockCount = calcSeq * numHeads;
        dataCopyXParams.blockLen = headDim * sizeof(T);
        dataCopyXParams.srcStride = 0;
        dataCopyXParams.dstStride = 0;
    } else if (layout == layoutBNSD) {
        dataCopyXParams.blockCount = calcSeq;
        dataCopyXParams.blockLen = headDim * sizeof(T);
        dataCopyXParams.srcStride = 0;
        dataCopyXParams.dstStride = 0;
    } else if (layout == layoutSBND) {
        dataCopyXParams.blockCount = calcSeq * batchSize * numHeads;
        dataCopyXParams.blockLen = headDim * sizeof(T);
        dataCopyXParams.srcStride = 0;
        dataCopyXParams.dstStride = 0;
    }
    if (layout == layoutBNSD) {
        for (uint32_t idx = 0; idx < batchSize * numHeads; idx++) {
            uint32_t gradOffset = idx * calcSeq * alignHeadDim;
            uint32_t gradGmOffset = idx * seqLen * headDim + loopIdx * coreInnerseqCalcNum * headDim;
            DataCopyPad(grad[gradOffset], gradGm[gradGmOffset], dataCopyXParams, padParams);
        }
    } else if (layout == layoutBSND) {
        for (uint32_t idx = 0; idx < batchSize; idx++) {
            uint32_t gradOffset = idx * calcSeq * numHeads * alignHeadDim;
            uint32_t gradGmOffset =
                idx * seqLen * numHeads * headDim + loopIdx * coreInnerseqCalcNum * numHeads * headDim;
            DataCopyPad(grad[gradOffset], gradGm[gradGmOffset], dataCopyXParams, padParams);
        }
    } else if (layout == layoutSBND) {
        DataCopyPad(grad, gradGm[loopIdx * coreInnerseqCalcNum * batchSize * numHeads * headDim], dataCopyXParams,
                    padParams);
    }
    inQueGrad.EnQue(grad);

    // copy in x
    if constexpr (NEEDBACKWARD) {
        LocalTensor<T> x = inQueX.AllocTensor<T>();
        if (layout == layoutBNSD) {
            for (uint32_t idx = 0; idx < batchSize * numHeads; idx++) {
                uint32_t xOffset = idx * calcSeq * alignHeadDim;
                uint32_t xGmOffset = idx * seqLen * headDim + loopIdx * coreInnerseqCalcNum * headDim;
                DataCopyPad(x[xOffset], xGm[xGmOffset], dataCopyXParams, padParams);
            }
        } else if (layout == layoutBSND) {
            for (uint32_t idx = 0; idx < batchSize; idx++) {
                uint32_t xOffset = idx * calcSeq * numHeads * alignHeadDim;
                uint32_t xGmOffset =
                    idx * seqLen * numHeads * headDim + loopIdx * coreInnerseqCalcNum * numHeads * headDim;
                DataCopyPad(x[xOffset], xGm[xGmOffset], dataCopyXParams, padParams);
            }
        } else if (layout == layoutSBND) {
            DataCopyPad(x, xGm[loopIdx * coreInnerseqCalcNum * batchSize * numHeads * headDim], dataCopyXParams,
                        padParams);
        }
        inQueX.EnQue(x);
    }

    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);

    if (layout == layoutBNSD) {
        // vector bnsd broadcast
        BroadCastToBnsd(cos, calcSeq);
        BroadCastToBnsd(sin, calcSeq);
    } else if (layout == layoutBSND) {
        // vector bsnd broadcast
        BroadCastToBsnd(cos, broadCastTmp, calcSeq);
        BroadCastToBsnd(sin, broadCastTmp, calcSeq);
    } else if (layout == layoutSBND) {
        // vector sbnd broadcast
        BroadCastToSbnd(cos, broadCastTmp, calcSeq);
        BroadCastToSbnd(sin, broadCastTmp, calcSeq);
    }
    inQueCosSin.EnQue(cosSin);
}

template <typename T, bool LARGE, bool NEEDBACKWARD>
__aicore__ inline void RopeInterleavedGrad<T, LARGE, NEEDBACKWARD>::CalculateDx()
{
    // use inputGrad, inputCos, inputSin to calc xGrad
    LocalTensor<T> outDxTensor = outQueueXGrad.AllocTensor<T>();
    if constexpr (std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value) {
        Cast(inputCosFloat, inputCos, RoundMode::CAST_NONE, dataNum);
        Cast(inputSinFloat, inputSin, RoundMode::CAST_NONE, dataNum);
        Cast(inputGradFloat, inputGrad, RoundMode::CAST_NONE, dataNum);
        PipeBarrier<PIPE_V>();
    } else {
        outDxTensorFloat = outDxTensor;
        inputSinFloat = inputSin;
        inputCosFloat = inputCos;
        inputGradFloat = inputGrad;
    }
    Mul(outDxTensorFloat, inputSinFloat, inputGradFloat, dataNum);
    Mul(calcTensor, inputCosFloat, inputGradFloat, dataNum);
    PipeBarrier<PIPE_V>();
    Gather(calcTensor2, outDxTensorFloat, gatherBuf.Get<uint32_t>(), 0, dataNum);
    PipeBarrier<PIPE_V>();
    inQueCosSin.FreeTensor(inputCosSin);
    TensorMul(outDxTensorFloat, calcTensor2, backwardMulTensor, dataNum);
    PipeBarrier<PIPE_V>();
    Add(outDxTensorFloat, outDxTensorFloat, calcTensor, dataNum);

    if constexpr (std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value) {
        Cast(outDxTensor, outDxTensorFloat, roundMode, dataNum);
    }
    outQueueXGrad.EnQue(outDxTensor);
}

template <typename T, bool LARGE, bool NEEDBACKWARD>
__aicore__ inline void RopeInterleavedGrad<T, LARGE, NEEDBACKWARD>::CalculateDcos(const LocalTensor<T> &outCos,
                                                                                  uint32_t loop_num)
{
    if constexpr (std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value) {
        Cast(inputXFloat, inputX, RoundMode::CAST_NONE, dataNum);
        PipeBarrier<PIPE_V>();
    } else {
        inputXFloat = inputX;
        inputGradFloat = inputGrad;
        outputCosFloat = outCos;
    }
    if constexpr ((std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value) && LARGE) {
        outputCosFloat = outCos.template ReinterpretCast<float>();
    }

    Mul(calcTensor, inputXFloat, inputGradFloat, dataNum);
    PipeBarrier<PIPE_V>();
    // vector bnsd reduce
    if constexpr (!LARGE) {
        Duplicate<float>(outputCosFloat, static_cast<float>(0), uint32_t(buffercosSize / sizeof(T)));
    }
    PipeBarrier<PIPE_V>();
    if (layout == layoutBNSD) {
        for (uint32_t idx = 0; idx < loop_num; idx++) {
            Add(outputCosFloat, outputCosFloat, calcTensor[idx * calcSeq * alignHeadDim],
                uint32_t(calcSeq * alignHeadDim));
            PipeBarrier<PIPE_V>();
        }
    } else if (layout == layoutBSND) {
        if constexpr (LARGE) {
            for (uint32_t idx = 0; idx < loop_num; idx++) {
                Add(outputCosFloat, outputCosFloat, calcTensor[idx * calcSeq * alignHeadDim],
                    uint32_t(calcSeq * alignHeadDim));
                PipeBarrier<PIPE_V>();
            }
        } else {
            ReduceToBsnd(calcTensor, broadCastFloatTmp, calcSeq);
            DataCopy(outputCosFloat, calcTensor, uint32_t(calcSeq * alignHeadDim));
        }
    } else if (layout == layoutSBND) {
        if constexpr (LARGE) {
            for (uint32_t idx = 0; idx < loop_num; idx++) {
                Add(outputCosFloat, outputCosFloat, calcTensor[idx * calcSeq * alignHeadDim],
                    uint32_t(calcSeq * alignHeadDim));
                PipeBarrier<PIPE_V>();
            }
        } else {
            ReduceToSbnd(calcTensor, broadCastFloatTmp, calcSeq);
            DataCopy(outputCosFloat, calcTensor, uint32_t(calcSeq * alignHeadDim));
        }
    }

    if constexpr ((std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value) && !LARGE) {
        Cast(outCos, outputCosFloat, roundMode, calcSeq * alignHeadDim);
    }
}

template <typename T, bool LARGE, bool NEEDBACKWARD>
__aicore__ inline void RopeInterleavedGrad<T, LARGE, NEEDBACKWARD>::CalculateDsin(const LocalTensor<T> &outSin,
                                                                                  uint32_t loop_num)
{
    if constexpr (std::is_same<T, float>::value) {
        outputSinFloat = outSin;
    }
    if constexpr ((std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value) && LARGE) {
        outputSinFloat = outSin.template ReinterpretCast<float>();
    }
    Gather(calcTensor2, inputXFloat, gatherBuf.Get<uint32_t>(), (uint32_t)0, dataNum);
    PipeBarrier<PIPE_V>();
    TensorMul(calcTensor, calcTensor2, forwardMulTensor, dataNum);
    PipeBarrier<PIPE_V>();
    Mul(calcTensor, calcTensor, inputGradFloat, dataNum);
    PipeBarrier<PIPE_V>();

    if constexpr (!LARGE) {
        Duplicate<float>(outputSinFloat, static_cast<float>(0), uint32_t(buffercosSize / sizeof(T)));
    }
    PipeBarrier<PIPE_V>();
    if (layout == layoutBNSD) {
        for (uint32_t idx = 0; idx < loop_num; idx++) {
            Add(outputSinFloat, outputSinFloat, calcTensor[idx * calcSeq * alignHeadDim],
                uint32_t(calcSeq * alignHeadDim));
            PipeBarrier<PIPE_V>();
        }
    } else if (layout == layoutBSND) {
        if constexpr (LARGE) {
            for (uint32_t idx = 0; idx < loop_num; idx++) {
                Add(outputSinFloat, outputSinFloat, calcTensor[idx * calcSeq * alignHeadDim],
                    uint32_t(calcSeq * alignHeadDim));
                PipeBarrier<PIPE_V>();
            }
        } else {
            ReduceToBsnd(calcTensor, broadCastFloatTmp, calcSeq);
            DataCopy(outputSinFloat, calcTensor, uint32_t(calcSeq * alignHeadDim));
        }
    } else if (layout == layoutSBND) {
        if constexpr (LARGE) {
            for (uint32_t idx = 0; idx < loop_num; idx++) {
                Add(outputSinFloat, outputSinFloat, calcTensor[idx * calcSeq * alignHeadDim],
                    uint32_t(calcSeq * alignHeadDim));
                PipeBarrier<PIPE_V>();
            }
        } else {
            ReduceToSbnd(calcTensor, broadCastFloatTmp, calcSeq);
            DataCopy(outputSinFloat, calcTensor, uint32_t(calcSeq * alignHeadDim));
        }
    }
    if constexpr ((std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value) && !LARGE) {
        Cast(outSin, outputSinFloat, roundMode, calcSeq * alignHeadDim);
    }
}

template <typename T, bool LARGE, bool NEEDBACKWARD>
__aicore__ inline void RopeInterleavedGrad<T, LARGE, NEEDBACKWARD>::SmallCompute()
{
    inputCosSin = inQueCosSin.DeQue<T>();
    inputGrad = inQueGrad.DeQue<T>();
    inputCos = inputCosSin[0];
    inputSin = inputCosSin[inputSinOffset];

    CalculateDx();

    if constexpr (NEEDBACKWARD) {
        inputX = inQueX.DeQue<T>();
        LocalTensor<T> outCosSinTensor = outQueueCosSinGrad.AllocTensor<T>();
        LocalTensor<T> outCos = outCosSinTensor[0];
        LocalTensor<T> outSin = outCosSinTensor[outputSinOffset];
        CalculateDcos(outCos, uint32_t(batchSize * numHeads));
        CalculateDsin(outSin, uint32_t(batchSize * numHeads));
        outQueueCosSinGrad.EnQue(outCosSinTensor);
    }

    inQueGrad.FreeTensor(inputGrad);
    if constexpr (NEEDBACKWARD) {
        inQueX.FreeTensor(inputX);
    }
}

template <typename T, bool LARGE, bool NEEDBACKWARD>
__aicore__ inline void RopeInterleavedGrad<T, LARGE, NEEDBACKWARD>::SmallCopyOut(uint64_t loopIdx)
{
    // copy out xGrad
    DataCopyExtParams dataCopyOutParams;
    if (layout == layoutBNSD) {
        dataCopyOutParams.blockCount = calcSeq;
        dataCopyOutParams.blockLen = headDim * sizeof(T);
        dataCopyOutParams.srcStride = 0;
        dataCopyOutParams.dstStride = 0;
    } else if (layout == layoutBSND) {
        dataCopyOutParams.blockCount = calcSeq * numHeads;
        dataCopyOutParams.blockLen = headDim * sizeof(T);
        dataCopyOutParams.srcStride = 0;
        dataCopyOutParams.dstStride = 0;
    } else if (layout == layoutSBND) {
        dataCopyOutParams.blockCount = calcSeq * batchSize * numHeads;
        dataCopyOutParams.blockLen = headDim * sizeof(T);
        dataCopyOutParams.srcStride = 0;
        dataCopyOutParams.dstStride = 0;
    }
    LocalTensor<T> outXGradTensor = outQueueXGrad.DeQue<T>();
    if (layout == layoutBNSD) {
        for (uint32_t idx = 0; idx < batchSize * numHeads; idx++) {
            uint32_t outXGradTensorOffset = idx * calcSeq * alignHeadDim;
            uint32_t xGradGmOffset = idx * seqLen * headDim + loopIdx * coreInnerseqCalcNum * headDim;
            DataCopyPad(xGradGm[xGradGmOffset], outXGradTensor[outXGradTensorOffset], dataCopyOutParams);
        }
    } else if (layout == layoutBSND) {
        for (uint32_t idx = 0; idx < batchSize; idx++) {
            uint32_t outXGradTensorOffset = idx * calcSeq * numHeads * alignHeadDim;
            uint32_t xGradGmOffset =
                idx * seqLen * numHeads * headDim + loopIdx * coreInnerseqCalcNum * numHeads * headDim;
            DataCopyPad(xGradGm[xGradGmOffset], outXGradTensor[outXGradTensorOffset], dataCopyOutParams);
        }
    } else if (layout == layoutSBND) {
        DataCopyPad(xGradGm[loopIdx * coreInnerseqCalcNum * batchSize * numHeads * headDim], outXGradTensor,
                    dataCopyOutParams);
    }
    outQueueXGrad.FreeTensor(outXGradTensor);

    if constexpr (NEEDBACKWARD) {
        DataCopyExtParams dataCopyOutCosParams;
        dataCopyOutCosParams.blockCount = calcSeq;
        dataCopyOutCosParams.blockLen = headDim * sizeof(T);
        dataCopyOutCosParams.srcStride = 0;
        dataCopyOutCosParams.dstStride = 0;
        // copy out dcos/dsin
        LocalTensor<T> outCosSinGradTensor = outQueueCosSinGrad.DeQue<T>();
        LocalTensor<T> outCosGradTensor = outCosSinGradTensor[0];
        LocalTensor<T> outSinGradTensor = outCosSinGradTensor[outputSinOffset];
        DataCopyPad(cosGradGm[loopIdx * coreInnerseqCalcNum * headDim], outCosGradTensor, dataCopyOutCosParams);
        DataCopyPad(sinGradGm[loopIdx * coreInnerseqCalcNum * headDim], outSinGradTensor, dataCopyOutCosParams);
        outQueueCosSinGrad.FreeTensor(outCosSinGradTensor);
    }
}

template <typename T, bool LARGE, bool NEEDBACKWARD>
__aicore__ inline void RopeInterleavedGrad<T, LARGE, NEEDBACKWARD>::SmallProcess()
{
    for (uint64_t loopIdx = 0; loopIdx < coreInnerseqCalcLoop; loopIdx++) {
        if ((loopIdx == coreInnerseqCalcLoop - 1) && coreInnerseqCalcTail > 0) {
            dataNum = batchSize * coreInnerseqCalcTail * numHeads * alignHeadDim;
            calcSeq = coreInnerseqCalcTail;
        }
        SmallCopyIn(loopIdx);
        SmallCompute();
        SmallCopyOut(loopIdx);
    }
}

template <typename T, bool LARGE, bool NEEDBACKWARD>
__aicore__ inline void RopeInterleavedGrad<T, LARGE, NEEDBACKWARD>::LargeCosSinInit()
{
    if constexpr (NEEDBACKWARD) {
        outCosSinTensorFloat = outQueueCosSinGrad.AllocTensor<float>();
        Duplicate<float>(outCosSinTensorFloat, static_cast<float>(0), 2 * alignHeadDim);
        PipeBarrier<PIPE_V>();
        outCosFloat = outCosSinTensorFloat[0];
        outSinFloat = outCosSinTensorFloat[outputSinOffset];
    }
}

template <typename T, bool LARGE, bool NEEDBACKWARD>
__aicore__ inline void RopeInterleavedGrad<T, LARGE, NEEDBACKWARD>::LargeCosSinCopyOut(uint64_t sIndex)
{
    if constexpr (NEEDBACKWARD) {
        LocalTensor<float> outCosSinGradTensor = outQueueCosSinGrad.DeQue<float>();
        LocalTensor<float> outCosGradTensor = outCosSinGradTensor[0];
        LocalTensor<float> outSinGradTensor = outCosSinGradTensor[outputSinOffset];

        if constexpr (std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value) {
            Cast(outCosGradTemp, outCosGradTensor, roundMode, alignHeadDim);
            event_t eventIdCosVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventIdCosVToMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventIdCosVToMTE3);
        } else {
            outCosGradTemp = outCosGradTensor;
        }

        if constexpr (std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value) {
            Cast(outSinGradTemp, outSinGradTensor, roundMode, alignHeadDim);
            event_t eventIdSinVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventIdSinVToMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventIdSinVToMTE3);
        } else {
            outSinGradTemp = outSinGradTensor;
        }

        DataCopyExtParams dataCopyCosParams;
        dataCopyCosParams.blockCount = 1;
        dataCopyCosParams.blockLen = headDim * sizeof(T);
        dataCopyCosParams.srcStride = 0;
        dataCopyCosParams.dstStride = 0;
        DataCopyPad(cosGradGm[(sIndex + seqOffset) * headDim], outCosGradTemp, dataCopyCosParams);
        DataCopyPad(sinGradGm[(sIndex + seqOffset) * headDim], outSinGradTemp, dataCopyCosParams);
        outQueueCosSinGrad.FreeTensor(outCosSinGradTensor);
    }
}

template <typename T, bool LARGE, bool NEEDBACKWARD>
__aicore__ inline void RopeInterleavedGrad<T, LARGE, NEEDBACKWARD>::LargeCopyIn(uint64_t gmOffset, uint64_t sIndex)
{
    LocalTensor<T> grad = inQueGrad.AllocTensor<T>();
    LocalTensor<T> cosSin = inQueCosSin.AllocTensor<T>();
    LocalTensor<T> cos = cosSin[0];
    LocalTensor<T> sin = cosSin[inputSinOffset];

    // cos/sin move in datacopyparams
    DataCopyExtParams dataCopyCosParams;
    dataCopyCosParams.blockCount = 1;
    dataCopyCosParams.blockLen = headDim * sizeof(T);
    dataCopyCosParams.srcStride = 0;
    dataCopyCosParams.dstStride = 0;

    // copy in cos/sin
    DataCopyPad(cos, cosGm[(sIndex + seqOffset) * headDim], dataCopyCosParams, padParams);
    DataCopyPad(sin, sinGm[(sIndex + seqOffset) * headDim], dataCopyCosParams, padParams);
    event_t eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);

    // copy in grad
    DataCopyExtParams dataCopyXParams;
    if (layout == layoutBNSD) {
        dataCopyXParams.blockCount = calcBn;
        dataCopyXParams.blockLen = headDim * sizeof(T);
        dataCopyXParams.srcStride = (seqLen - 1) * headDim * sizeof(T);
        dataCopyXParams.dstStride = 0;
    } else if (layout == layoutBSND) {
        dataCopyXParams.blockCount = calcBn;
        dataCopyXParams.blockLen = headDim * sizeof(T);
        dataCopyXParams.srcStride = 0;
        dataCopyXParams.dstStride = 0;
    } else if (layout == layoutSBND) {
        dataCopyXParams.blockCount = calcBn;
        dataCopyXParams.blockLen = headDim * sizeof(T);
        dataCopyXParams.srcStride = 0;
        dataCopyXParams.dstStride = 0;
    }
    DataCopyPad(grad, gradGm[gmOffset], dataCopyXParams, padParams);
    inQueGrad.EnQue(grad);

    // copy in x
    if constexpr (NEEDBACKWARD) {
        LocalTensor<T> x = inQueX.AllocTensor<T>();
        DataCopyPad(x, xGm[gmOffset], dataCopyXParams, padParams);
        inQueX.EnQue(x);
    }

    // vector large broadcast
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
    for (uint64_t idx = 0; idx < calcBn; idx++) {
        DataCopy(cos[idx * alignHeadDim], cos, alignHeadDim);
    }
    for (uint64_t idx = 0; idx < calcBn; idx++) {
        DataCopy(sin[idx * alignHeadDim], sin, alignHeadDim);
    }
    inQueCosSin.EnQue(cosSin);
}

template <typename T, bool LARGE, bool NEEDBACKWARD>
__aicore__ inline void RopeInterleavedGrad<T, LARGE, NEEDBACKWARD>::LargeCompute(uint64_t innerLoopIndex,
                                                                                 uint64_t outerLoopIndex)
{
    inputCosSin = inQueCosSin.DeQue<T>();
    inputGrad = inQueGrad.DeQue<T>();
    inputCos = inputCosSin[0];
    inputSin = inputCosSin[inputSinOffset];
    CalculateDx();
    if constexpr (NEEDBACKWARD) {
        inputX = inQueX.DeQue<T>();
        CalculateDcos(outCosFloat.template ReinterpretCast<T>(), calcBn);
        CalculateDsin(outSinFloat.template ReinterpretCast<T>(), calcBn);
    }
    inQueGrad.FreeTensor(inputGrad);
    if constexpr (NEEDBACKWARD) {
        inQueX.FreeTensor(inputX);
        if ((innerLoopIndex == (innerLoop - 1)) && (outerLoopIndex == (outerLoop - 1))) {
            outQueueCosSinGrad.EnQue(outCosSinTensorFloat);
        }
    }
}

template <typename T, bool LARGE, bool NEEDBACKWARD>
__aicore__ inline void RopeInterleavedGrad<T, LARGE, NEEDBACKWARD>::LargeXGradCopyOut(uint64_t gmOffset)
{
    DataCopyExtParams dataCopyOutParams;
    if (layout == layoutBNSD) {
        dataCopyOutParams.blockCount = calcBn;
        dataCopyOutParams.blockLen = headDim * sizeof(T);
        dataCopyOutParams.srcStride = 0;
        dataCopyOutParams.dstStride = (seqLen - 1) * headDim * sizeof(T);
    } else if (layout == layoutBSND) {
        dataCopyOutParams.blockCount = calcBn;
        dataCopyOutParams.blockLen = headDim * sizeof(T);
        dataCopyOutParams.srcStride = 0;
        dataCopyOutParams.dstStride = 0;
    } else if (layout == layoutSBND) {
        dataCopyOutParams.blockCount = calcBn;
        dataCopyOutParams.blockLen = headDim * sizeof(T);
        dataCopyOutParams.srcStride = 0;
        dataCopyOutParams.dstStride = 0;
    }
    LocalTensor<T> outXGradTensor = outQueueXGrad.DeQue<T>();
    DataCopyPad(xGradGm[gmOffset], outXGradTensor, dataCopyOutParams);
    outQueueXGrad.FreeTensor(outXGradTensor);
}

template <typename T, bool LARGE, bool NEEDBACKWARD>
__aicore__ inline void RopeInterleavedGrad<T, LARGE, NEEDBACKWARD>::LargeProcess()
{
    for (uint64_t sIndex = 0; sIndex < seqCoreLen; sIndex++) {
        LargeCosSinInit();
        for (uint64_t outerLoopIndex = 0; outerLoopIndex < outerLoop; outerLoopIndex++) {
            for (uint64_t innerLoopIndex = 0; innerLoopIndex < innerLoop; innerLoopIndex++) {
                uint64_t gmOffset = 0;
                if (layout == layoutBNSD) {
                    gmOffset = innerLoopIndex * innerLength * seqLen * headDim + (sIndex + seqOffset) * headDim;
                } else if (layout == layoutBSND) {
                    gmOffset = outerLoopIndex * seqLen * numHeads * headDim +
                               (sIndex + seqOffset) * numHeads * headDim + innerLoopIndex * innerLength * headDim;
                } else if (layout == layoutSBND) {
                    gmOffset =
                        (sIndex + seqOffset) * batchSize * numHeads * headDim + innerLoopIndex * innerLength * headDim;
                }
                if ((innerLoopIndex == (innerLoop - 1)) && innerTail > 0) {
                    dataNum = innerTail * alignHeadDim;
                    calcBn = innerTail;
                } else {
                    dataNum = innerLength * alignHeadDim;
                    calcBn = innerLength;
                }
                LargeCopyIn(gmOffset, sIndex);
                LargeCompute(innerLoopIndex, outerLoopIndex);
                LargeXGradCopyOut(gmOffset);
            }
        }
        LargeCosSinCopyOut(sIndex);
    }
}

template <typename T, bool LARGE, bool NEEDBACKWARD>
__aicore__ inline void RopeInterleavedGrad<T, LARGE, NEEDBACKWARD>::Process()
{
    if constexpr (!LARGE) {
        SmallProcess();
    } else {
        LargeProcess();
    }
}

#endif // ROPE_INTERLEAVED_GRAD_SPLITS_H
