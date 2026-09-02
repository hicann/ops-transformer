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
 * \file rotate_half_grad.h
 * \brief
 */
#ifndef _ROTATE_HALF_GRAD_H_
#define _ROTATE_HALF_GRAD_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

template <typename T, int LAYOUT, bool IF_NEED_BACKWARD, bool IF_CAST, bool IF_ALIGN>
class RotateHalfGrad {
public:
    __aicore__ inline RotateHalfGrad(){};
    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR cos, GM_ADDR sin, GM_ADDR x, GM_ADDR xGrad, GM_ADDR cosGrad,
                                GM_ADDR sinGrad, GM_ADDR workspace,
                                const RotaryPositionEmbeddingGradTilingData &tiling);
    __aicore__ inline void Process();
    __aicore__ inline void LoopProcess(uint64_t &loopIdx, uint64_t &elementNum);
    __aicore__ inline void Compute(uint64_t &outerOffset, uint64_t &innerOffset, uint64_t &elementNum,
                                   uint64_t &ubPerReserveNum, uint64_t &blockLenInner);
    __aicore__ inline void ComputeInner(const LocalTensor<T> &cosLocal, const LocalTensor<T> &sinLocal,
                                        uint64_t &offset, uint64_t &elementNum, uint64_t &ubPerReserveNum,
                                        uint64_t &blockLenInner);
    __aicore__ inline void CopyInOuter(uint64_t &offset, uint64_t &elementNum);
    __aicore__ inline void CopyInInner(uint64_t &offset, uint16_t blockCount, uint32_t blockLen, uint32_t srcStride,
                                       uint32_t dstStride);
    __aicore__ inline void CopyOutInner(uint64_t &offset, uint16_t blockCount, uint32_t blockLen, uint32_t srcStride,
                                        uint32_t dstStride);
    __aicore__ inline void CopyOutOuter(uint64_t &offset, uint64_t &elementNum);
    __aicore__ inline void CopyInPadOuter(uint64_t &offset, uint16_t blockCount, uint32_t blockLen, uint32_t srcStride,
                                          uint32_t dstStride);
    __aicore__ inline void CopyInPadInner(uint64_t &offset, uint16_t blockCount, uint32_t blockLen, uint32_t srcStride,
                                          uint32_t dstStride);
    __aicore__ inline void CopyOutPadInner(uint64_t &offset, uint16_t blockCount, uint32_t blockLen, uint32_t srcStride,
                                           uint32_t dstStride);
    __aicore__ inline void CopyOutPadOuter(uint64_t &offset, uint16_t blockCount, uint32_t blockLen, uint32_t srcStride,
                                           uint32_t dstStride);
    __aicore__ inline void DataChunkCat(const LocalTensor<float> &dstLocal, const LocalTensor<float> &src1Local,
                                        const LocalTensor<float> &src2Local, uint64_t offset, uint16_t blockCount,
                                        uint32_t blockLen, uint32_t srcStride, uint32_t dstStride);
    __aicore__ inline void InitCastInOuter(const LocalTensor<T> &cosLocal, const LocalTensor<T> &sinLocal,
                                           uint64_t &elementNum);
    __aicore__ inline void CastOutOuter(const LocalTensor<T> &cosGradLocal, const LocalTensor<T> &sinGradLocal,
                                        uint64_t &elementNum);

protected:
    // define const value
    static constexpr uint64_t BLOCK_SIZE = 32;
    static constexpr uint64_t MASK_NUM_FP32 = 64;
    static constexpr uint64_t BUFFER_NUM = 1;
    static constexpr uint64_t LAYOUT_BSND = 0;
    static constexpr uint64_t LAYOUT_BNSD = 1;
    static constexpr uint64_t LAYOUT_SBND = 2;

    // init const
    uint64_t outerOffset;
    uint64_t innerOffset;
    uint64_t firstReduceOffset;
    uint64_t secondReduceOffset;
    uint64_t firstReduceOuterOffset;
    uint64_t dataEachBlock;
    uint64_t dataEachBlockFP32;
    uint64_t loopNum;
    uint64_t tailNum;
    uint64_t ubPerReserveNum;
    uint64_t elementNumPad;

    // init tiling data
    uint64_t xShapeSize;
    uint64_t cosShapeSize;
    uint64_t dimB;
    uint64_t dimS;
    uint64_t dimN;
    uint64_t dimD;
    uint64_t cosDimB;
    uint64_t cosDimN;
    uint64_t halfDimDAlignNum;

    uint64_t coreData;
    uint64_t coreLast;
    uint64_t copyLoop;
    uint64_t copyTail;
    uint64_t lastCopyLoop;
    uint64_t lastCopyTail;
    uint64_t alignUbSize;
    uint64_t calcUbSize;
    uint64_t coreUsed;
    uint64_t coreNum;

    uint64_t firstReduce;
    uint64_t secondReduce;
    uint64_t ubLoopGap;
    uint64_t blockLenInner;
    uint64_t strideInner;
    uint64_t blockLenPadInner;
    uint64_t stridePadInner;

    // init gm tensor
    GlobalTensor<T> xGm, cosGm, sinGm, gradGm, xGradGm, cosGradGm, sinGradGm;

    // init pipe
    TPipe pipe;

    // init mte queues
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueCos, inQueueSin, inQueueGrad;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueXGrad, outQueueCosGrad, outQueueSinGrad;

    // init vector buffer
    TBuf<TPosition::VECCALC> tmpCalcBufOutCosGrad, tmpCalcBufOutSinGrad, tmpCalcBufA, tmpCalcBufB, tmpCalcBufC,
        tmpCalcBufCastRes, tmpCalcBufCastInCos, tmpCalcBufCastInSin, tmpCalcBufCastInGrad;
    LocalTensor<float> tmpOutCosGrad;
    LocalTensor<float> tmpOutSinGrad;
    LocalTensor<float> tmpCastRes;
    LocalTensor<float> cosCast;
    LocalTensor<float> sinCast;
    LocalTensor<float> gradCast;
};

template <typename T, int LAYOUT, bool IF_NEED_BACKWARD, bool IF_CAST, bool IF_ALIGN>
__aicore__ inline void RotateHalfGrad<T, LAYOUT, IF_NEED_BACKWARD, IF_CAST, IF_ALIGN>::Init(
    GM_ADDR grad, GM_ADDR cos, GM_ADDR sin, GM_ADDR x, GM_ADDR xGrad, GM_ADDR cosGrad, GM_ADDR sinGrad,
    GM_ADDR workspace, const RotaryPositionEmbeddingGradTilingData &tiling)
{
    // set const
    outerOffset = 0;
    innerOffset = 0;
    firstReduceOffset = 0;
    secondReduceOffset = 0;
    firstReduceOuterOffset = 0;
    dataEachBlock = BLOCK_SIZE / sizeof(T);
    dataEachBlockFP32 = BLOCK_SIZE / sizeof(float);
    ubPerReserveNum = 0;
    elementNumPad = 0;
    loopNum = 0;
    tailNum = 0;

    // set tiling data
    const RopeHalfGradParams &rotateHalfGradTiling = tiling.ropeHalfGradParams;
    xShapeSize = rotateHalfGradTiling.xShapeSize;
    cosShapeSize = rotateHalfGradTiling.cosShapeSize;
    dimB = rotateHalfGradTiling.dimB;
    dimS = rotateHalfGradTiling.dimS;
    dimN = rotateHalfGradTiling.dimN;
    dimD = rotateHalfGradTiling.dimD;
    cosDimB = rotateHalfGradTiling.cosDimB;
    cosDimN = rotateHalfGradTiling.cosDimN;
    halfDimDAlignNum = rotateHalfGradTiling.halfDimDAlignNum;

    coreData = rotateHalfGradTiling.coreData;
    coreLast = rotateHalfGradTiling.coreLast;
    copyLoop = rotateHalfGradTiling.copyLoop;
    copyTail = rotateHalfGradTiling.copyTail;
    lastCopyLoop = rotateHalfGradTiling.lastCopyLoop;
    lastCopyTail = rotateHalfGradTiling.lastCopyTail;
    alignUbSize = rotateHalfGradTiling.alignUbSize;
    calcUbSize = rotateHalfGradTiling.calcUbSize;
    coreUsed = rotateHalfGradTiling.coreUsed;
    coreNum = rotateHalfGradTiling.coreNum;

    firstReduce = rotateHalfGradTiling.firstReduce;
    secondReduce = rotateHalfGradTiling.secondReduce;
    ubLoopGap = rotateHalfGradTiling.ubLoopGap;
    blockLenInner = rotateHalfGradTiling.blockLenInner;
    strideInner = rotateHalfGradTiling.strideInner;
    blockLenPadInner = rotateHalfGradTiling.blockLenPadInner;
    stridePadInner = rotateHalfGradTiling.stridePadInner;

    // set gm tensor
    gradGm.SetGlobalBuffer((__gm__ T *)grad, xShapeSize);
    cosGm.SetGlobalBuffer((__gm__ T *)cos, cosShapeSize);
    sinGm.SetGlobalBuffer((__gm__ T *)sin, cosShapeSize);
    xGm.SetGlobalBuffer((__gm__ T *)x, xShapeSize);

    xGradGm.SetGlobalBuffer((__gm__ T *)xGrad, xShapeSize);
    cosGradGm.SetGlobalBuffer((__gm__ T *)cosGrad, cosShapeSize);
    sinGradGm.SetGlobalBuffer((__gm__ T *)sinGrad, cosShapeSize);

    if (GetBlockIdx() != coreUsed - 1) {
        loopNum = copyLoop;
        tailNum = copyTail;
    } else {
        loopNum = lastCopyLoop;
        tailNum = lastCopyTail;
    }

    // init buffer
    pipe.InitBuffer(inQueueGrad, BUFFER_NUM, alignUbSize * sizeof(T));
    pipe.InitBuffer(inQueueCos, BUFFER_NUM, alignUbSize * sizeof(T));
    pipe.InitBuffer(inQueueSin, BUFFER_NUM, alignUbSize * sizeof(T));
    pipe.InitBuffer(outQueueXGrad, BUFFER_NUM, alignUbSize * sizeof(T));

    pipe.InitBuffer(tmpCalcBufA, alignUbSize * sizeof(float));
    pipe.InitBuffer(tmpCalcBufB, alignUbSize * sizeof(float));
    pipe.InitBuffer(tmpCalcBufC, alignUbSize * sizeof(float));

    if constexpr (IF_NEED_BACKWARD) {
        pipe.InitBuffer(inQueueX, BUFFER_NUM, alignUbSize * sizeof(T));
        pipe.InitBuffer(outQueueCosGrad, BUFFER_NUM, alignUbSize * sizeof(T));
        pipe.InitBuffer(outQueueSinGrad, BUFFER_NUM, alignUbSize * sizeof(T));

        pipe.InitBuffer(tmpCalcBufOutCosGrad, alignUbSize * sizeof(float));
        pipe.InitBuffer(tmpCalcBufOutSinGrad, alignUbSize * sizeof(float));
    }

    if constexpr (IF_CAST) {
        pipe.InitBuffer(tmpCalcBufCastInGrad, alignUbSize * sizeof(float));
        pipe.InitBuffer(tmpCalcBufCastInCos, alignUbSize * sizeof(float));
        pipe.InitBuffer(tmpCalcBufCastInSin, alignUbSize * sizeof(float));
        pipe.InitBuffer(tmpCalcBufCastRes, alignUbSize * sizeof(float));
    }

    // init tensor
    if constexpr (IF_NEED_BACKWARD) {
        tmpOutCosGrad = tmpCalcBufOutCosGrad.Get<float>(alignUbSize);
        tmpOutSinGrad = tmpCalcBufOutSinGrad.Get<float>(alignUbSize);
    }

    if constexpr (IF_CAST) {
        gradCast = tmpCalcBufCastInGrad.Get<float>(alignUbSize);
        cosCast = tmpCalcBufCastInCos.Get<float>(alignUbSize);
        sinCast = tmpCalcBufCastInSin.Get<float>(alignUbSize);
        tmpCastRes = tmpCalcBufCastRes.Get<float>(alignUbSize);
    }
}

template <typename T, int LAYOUT, bool IF_NEED_BACKWARD, bool IF_CAST, bool IF_ALIGN>
__aicore__ inline void RotateHalfGrad<T, LAYOUT, IF_NEED_BACKWARD, IF_CAST, IF_ALIGN>::Process()
{
    for (uint64_t loopIdx = 0; loopIdx < loopNum; ++loopIdx) {
        LoopProcess(loopIdx, calcUbSize);
    }
    if (tailNum > 0) {
        LoopProcess(loopNum, tailNum);
    }
}

template <typename T, int LAYOUT, bool IF_NEED_BACKWARD, bool IF_CAST, bool IF_ALIGN>
__aicore__ inline void RotateHalfGrad<T, LAYOUT, IF_NEED_BACKWARD, IF_CAST, IF_ALIGN>::LoopProcess(uint64_t &loopIdx,
                                                                                                   uint64_t &elementNum)
{
    outerOffset = GetBlockIdx() * coreData + loopIdx * calcUbSize;
    innerOffset = GetBlockIdx() * coreData * secondReduce + loopIdx * calcUbSize * secondReduce;
    ubPerReserveNum = elementNum / dimD;
    if constexpr (IF_ALIGN) {
        if constexpr (LAYOUT == LAYOUT_BNSD) {
            blockLenInner = (ubPerReserveNum * dimD + dataEachBlock - 1) / dataEachBlock;
        }
        Compute(outerOffset, innerOffset, elementNum, ubPerReserveNum, blockLenInner);
    } else {
        elementNumPad = ubPerReserveNum * halfDimDAlignNum * 2;
        Compute(outerOffset, innerOffset, elementNumPad, ubPerReserveNum, blockLenInner);
    }
}

template <typename T, int LAYOUT, bool IF_NEED_BACKWARD, bool IF_CAST, bool IF_ALIGN>
__aicore__ inline void RotateHalfGrad<T, LAYOUT, IF_NEED_BACKWARD, IF_CAST, IF_ALIGN>::Compute(
    uint64_t &outerOffset, uint64_t &innerOffset, uint64_t &elementNum, uint64_t &ubPerReserveNum,
    uint64_t &blockLenInner)
{
    if (LAYOUT == LAYOUT_BNSD && cosDimB != 1 && cosDimN == 1) {
        for (uint64_t firstReduceIdx = 0; firstReduceIdx < cosDimB; ++firstReduceIdx) {
            firstReduceOuterOffset = outerOffset + firstReduceIdx * dimS * dimD;
            firstReduceOffset = innerOffset + firstReduceIdx * dimN * dimS * dimD;
            if constexpr (IF_ALIGN) {
                CopyInOuter(firstReduceOuterOffset, elementNum);
            } else {
                CopyInPadOuter(firstReduceOuterOffset, static_cast<uint16_t>(ubPerReserveNum * 2),
                               static_cast<uint32_t>(blockLenPadInner), 0, 0);
            }
            LocalTensor<T> cosLocal = inQueueCos.DeQue<T>();
            LocalTensor<T> sinLocal = inQueueSin.DeQue<T>();
            InitCastInOuter(cosLocal, sinLocal, elementNum);
            for (uint64_t secondReduceIdx = 0; secondReduceIdx < dimN; ++secondReduceIdx) {
                secondReduceOffset = firstReduceOffset + secondReduceIdx * dimS * dimD;
                if constexpr (IF_ALIGN) {
                    CopyInInner(secondReduceOffset, static_cast<uint16_t>(ubPerReserveNum),
                                static_cast<uint32_t>(blockLenInner), static_cast<uint32_t>(strideInner), 0);
                } else {
                    CopyInPadInner(secondReduceOffset, static_cast<uint16_t>(ubPerReserveNum),
                                   static_cast<uint32_t>(blockLenPadInner), static_cast<uint32_t>(stridePadInner),
                                   static_cast<uint32_t>(halfDimDAlignNum / dataEachBlock));
                }
                ComputeInner(cosLocal, sinLocal, secondReduceOffset, elementNum, ubPerReserveNum, blockLenInner);
            }
            inQueueCos.FreeTensor(cosLocal);
            inQueueSin.FreeTensor(sinLocal);

            if constexpr (IF_NEED_BACKWARD) {
                LocalTensor<T> cosGradLocal = outQueueCosGrad.AllocTensor<T>();
                LocalTensor<T> sinGradLocal = outQueueSinGrad.AllocTensor<T>();
                CastOutOuter(cosGradLocal, sinGradLocal, elementNum);
                outQueueCosGrad.EnQue<T>(cosGradLocal);
                outQueueSinGrad.EnQue<T>(sinGradLocal);
                if constexpr (IF_ALIGN) {
                    CopyOutOuter(firstReduceOuterOffset, elementNum);
                } else {
                    CopyOutPadOuter(firstReduceOuterOffset, static_cast<uint16_t>(ubPerReserveNum * 2),
                                    static_cast<uint32_t>(blockLenPadInner), 0, 0);
                }
            }
        }
    } else {
        if constexpr (IF_ALIGN) {
            CopyInOuter(outerOffset, elementNum);
        } else {
            CopyInPadOuter(outerOffset, static_cast<uint16_t>(ubPerReserveNum * 2),
                           static_cast<uint32_t>(blockLenPadInner), 0, 0);
        }
        LocalTensor<T> cosLocal = inQueueCos.DeQue<T>();
        LocalTensor<T> sinLocal = inQueueSin.DeQue<T>();
        InitCastInOuter(cosLocal, sinLocal, elementNum);
        for (uint64_t firstReduceIdx = 0; firstReduceIdx < firstReduce; ++firstReduceIdx) {
            firstReduceOffset = innerOffset + firstReduceIdx * ubLoopGap;
            for (uint64_t secondReduceIdx = 0; secondReduceIdx < secondReduce; ++secondReduceIdx) {
                secondReduceOffset = firstReduceOffset + secondReduceIdx * dimD;
                if constexpr (IF_ALIGN) {
                    CopyInInner(secondReduceOffset, static_cast<uint16_t>(ubPerReserveNum),
                                static_cast<uint32_t>(blockLenInner), static_cast<uint32_t>(strideInner), 0);
                } else {
                    CopyInPadInner(secondReduceOffset, static_cast<uint16_t>(ubPerReserveNum),
                                   static_cast<uint32_t>(blockLenPadInner), static_cast<uint32_t>(stridePadInner),
                                   static_cast<uint32_t>(halfDimDAlignNum / dataEachBlock));
                }
                ComputeInner(cosLocal, sinLocal, secondReduceOffset, elementNum, ubPerReserveNum, blockLenInner);
            }
        }
        inQueueCos.FreeTensor(cosLocal);
        inQueueSin.FreeTensor(sinLocal);

        if constexpr (IF_NEED_BACKWARD) {
            LocalTensor<T> cosGradLocal = outQueueCosGrad.AllocTensor<T>();
            LocalTensor<T> sinGradLocal = outQueueSinGrad.AllocTensor<T>();
            CastOutOuter(cosGradLocal, sinGradLocal, elementNum);
            outQueueCosGrad.EnQue<T>(cosGradLocal);
            outQueueSinGrad.EnQue<T>(sinGradLocal);
            if constexpr (IF_ALIGN) {
                CopyOutOuter(outerOffset, elementNum);
            } else {
                CopyOutPadOuter(outerOffset, static_cast<uint16_t>(ubPerReserveNum * 2),
                                static_cast<uint32_t>(blockLenPadInner), 0, 0);
            }
        }
    }
}

template <typename T, int LAYOUT, bool IF_NEED_BACKWARD, bool IF_CAST, bool IF_ALIGN>
__aicore__ inline void RotateHalfGrad<T, LAYOUT, IF_NEED_BACKWARD, IF_CAST, IF_ALIGN>::ComputeInner(
    const LocalTensor<T> &cosLocal, const LocalTensor<T> &sinLocal, uint64_t &offset, uint64_t &elementNum,
    uint64_t &ubPerReserveNum, uint64_t &blockLenInner)
{
    // compute xGrad, cosGrad, sinGrad

    // xGrad
    LocalTensor<T> gradLocal = inQueueGrad.DeQue<T>();
    LocalTensor<T> xGradLocal = outQueueXGrad.AllocTensor<T>();
    LocalTensor<float> tmpLocalA = tmpCalcBufA.Get<float>(alignUbSize);
    LocalTensor<float> tmpLocalB = tmpCalcBufB.Get<float>(alignUbSize);
    LocalTensor<float> tmpLocalC = tmpCalcBufC.Get<float>(alignUbSize);
    if constexpr (IF_CAST) {
        Cast(gradCast, gradLocal, AscendC::RoundMode::CAST_NONE, elementNum);
        inQueueGrad.FreeTensor(gradLocal);
        PipeBarrier<PIPE_V>();
        Mul(tmpCastRes, cosCast, gradCast, elementNum);
    } else {
        Mul(xGradLocal, cosLocal, gradLocal, elementNum);
    }

    if constexpr (IF_CAST) {
        Mul(tmpLocalB, sinCast, gradCast, elementNum);
    } else {
        Mul(tmpLocalB, sinLocal, gradLocal, elementNum);
        if constexpr (!IF_NEED_BACKWARD) {
            inQueueGrad.FreeTensor(gradLocal);
        }
    }
    PipeBarrier<PIPE_V>();
    Muls(tmpLocalA, tmpLocalB, (float)-1.0, elementNum);
    PipeBarrier<PIPE_V>();
    DataChunkCat(tmpLocalC, tmpLocalB, tmpLocalA, halfDimDAlignNum, static_cast<uint16_t>(ubPerReserveNum),
                 static_cast<uint32_t>(halfDimDAlignNum / dataEachBlockFP32),
                 static_cast<uint32_t>(halfDimDAlignNum / dataEachBlockFP32),
                 static_cast<uint32_t>(halfDimDAlignNum / dataEachBlockFP32));
    PipeBarrier<PIPE_V>();
    if constexpr (IF_CAST) {
        Add(tmpCastRes, tmpLocalC, tmpCastRes, elementNum);
        PipeBarrier<PIPE_V>();
        Cast(xGradLocal, tmpCastRes, AscendC::RoundMode::CAST_RINT, elementNum);
    } else {
        Add(xGradLocal, tmpLocalC, xGradLocal, elementNum);
    }
    outQueueXGrad.EnQue<T>(xGradLocal);
    if constexpr (IF_ALIGN) {
        CopyOutInner(offset, static_cast<uint16_t>(ubPerReserveNum), static_cast<uint32_t>(blockLenInner), 0,
                     static_cast<uint32_t>(strideInner));
    } else {
        CopyOutPadInner(offset, static_cast<uint16_t>(ubPerReserveNum), static_cast<uint32_t>(blockLenPadInner),
                        static_cast<uint32_t>(halfDimDAlignNum / dataEachBlock), static_cast<uint32_t>(stridePadInner));
    }

    if constexpr (IF_NEED_BACKWARD) {
        // cosGrad
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        if constexpr (IF_CAST) {
            Cast(tmpLocalC, xLocal, AscendC::RoundMode::CAST_NONE, elementNum);
            inQueueX.FreeTensor(xLocal);
            PipeBarrier<PIPE_V>();
            Mul(tmpCastRes, tmpLocalC, gradCast, elementNum);
            PipeBarrier<PIPE_V>();
            Add(tmpOutCosGrad, tmpCastRes, tmpOutCosGrad, elementNum);
        } else {
            PipeBarrier<PIPE_V>();
            Mul(tmpLocalB, xLocal, gradLocal, elementNum);
            PipeBarrier<PIPE_V>();
            Add(tmpOutCosGrad, tmpLocalB, tmpOutCosGrad, elementNum);
        }

        // sinGrad
        PipeBarrier<PIPE_V>();
        if constexpr (IF_CAST) {
            Muls(tmpLocalA, tmpLocalC, (float)-1.0, elementNum);
            PipeBarrier<PIPE_V>();
            DataChunkCat(tmpLocalB, tmpLocalA, tmpLocalC, halfDimDAlignNum, static_cast<uint16_t>(ubPerReserveNum),
                         static_cast<uint32_t>(halfDimDAlignNum / dataEachBlockFP32),
                         static_cast<uint32_t>(halfDimDAlignNum / dataEachBlockFP32),
                         static_cast<uint32_t>(halfDimDAlignNum / dataEachBlockFP32));
            PipeBarrier<PIPE_V>();
            Mul(tmpCastRes, tmpLocalB, gradCast, elementNum);
            PipeBarrier<PIPE_V>();
            Add(tmpOutSinGrad, tmpCastRes, tmpOutSinGrad, elementNum);
        } else {
            Muls(tmpLocalA, xLocal, (float)-1.0, elementNum);
            PipeBarrier<PIPE_V>();
            DataChunkCat(tmpLocalB, tmpLocalA, xLocal, halfDimDAlignNum, static_cast<uint16_t>(ubPerReserveNum),
                         static_cast<uint32_t>(halfDimDAlignNum / dataEachBlockFP32),
                         static_cast<uint32_t>(halfDimDAlignNum / dataEachBlockFP32),
                         static_cast<uint32_t>(halfDimDAlignNum / dataEachBlockFP32));
            inQueueX.FreeTensor(xLocal);
            PipeBarrier<PIPE_V>();
            Mul(tmpLocalA, tmpLocalB, gradLocal, elementNum);
            inQueueGrad.FreeTensor(gradLocal);
            PipeBarrier<PIPE_V>();
            Add(tmpOutSinGrad, tmpLocalA, tmpOutSinGrad, elementNum);
        }
    }
}

template <typename T, int LAYOUT, bool IF_NEED_BACKWARD, bool IF_CAST, bool IF_ALIGN>
__aicore__ inline void RotateHalfGrad<T, LAYOUT, IF_NEED_BACKWARD, IF_CAST, IF_ALIGN>::CopyInOuter(uint64_t &offset,
                                                                                                   uint64_t &elementNum)
{
    // copy in cos, sin

    LocalTensor<T> cosLocal = inQueueCos.AllocTensor<T>();
    LocalTensor<T> sinLocal = inQueueSin.AllocTensor<T>();
    DataCopy(cosLocal, cosGm[offset], elementNum);
    DataCopy(sinLocal, sinGm[offset], elementNum);
    inQueueCos.EnQue(cosLocal);
    inQueueSin.EnQue(sinLocal);
}

template <typename T, int LAYOUT, bool IF_NEED_BACKWARD, bool IF_CAST, bool IF_ALIGN>
__aicore__ inline void RotateHalfGrad<T, LAYOUT, IF_NEED_BACKWARD, IF_CAST, IF_ALIGN>::CopyInInner(
    uint64_t &offset, uint16_t blockCount, uint32_t blockLen, uint32_t srcStride, uint32_t dstStride)
{
    // copy in x, grad

    DataCopyParams copyParams;
    if constexpr (LAYOUT == LAYOUT_BNSD) {
        copyParams.blockCount = 1;
    } else {
        copyParams.blockCount = blockCount;
    }
    copyParams.blockLen = blockLen;
    copyParams.srcStride = srcStride;
    copyParams.dstStride = dstStride;

    LocalTensor<T> gradLocal = inQueueGrad.AllocTensor<T>();
    DataCopy(gradLocal, gradGm[offset], copyParams);
    inQueueGrad.EnQue(gradLocal);
    if constexpr (IF_NEED_BACKWARD) {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        DataCopy(xLocal, xGm[offset], copyParams);
        inQueueX.EnQue(xLocal);
    }
}

template <typename T, int LAYOUT, bool IF_NEED_BACKWARD, bool IF_CAST, bool IF_ALIGN>
__aicore__ inline void RotateHalfGrad<T, LAYOUT, IF_NEED_BACKWARD, IF_CAST, IF_ALIGN>::CopyOutInner(
    uint64_t &offset, uint16_t blockCount, uint32_t blockLen, uint32_t srcStride, uint32_t dstStride)
{
    // copy out xGrad

    DataCopyParams copyParams;
    if constexpr (LAYOUT == LAYOUT_BNSD) {
        copyParams.blockCount = 1;
    } else {
        copyParams.blockCount = blockCount;
    }
    copyParams.blockLen = blockLen;
    copyParams.srcStride = srcStride;
    copyParams.dstStride = dstStride;

    LocalTensor<T> xGradLocal = outQueueXGrad.DeQue<T>();
    DataCopy(xGradGm[offset], xGradLocal, copyParams);
    outQueueXGrad.FreeTensor(xGradLocal);
}

template <typename T, int LAYOUT, bool IF_NEED_BACKWARD, bool IF_CAST, bool IF_ALIGN>
__aicore__ inline void RotateHalfGrad<T, LAYOUT, IF_NEED_BACKWARD, IF_CAST, IF_ALIGN>::CopyOutOuter(
    uint64_t &offset, uint64_t &elementNum)
{
    // copy out cos_grad, sin_grad

    LocalTensor<T> cosGradLocal = outQueueCosGrad.DeQue<T>();
    LocalTensor<T> sinGradLocal = outQueueSinGrad.DeQue<T>();
    DataCopy(cosGradGm[offset], cosGradLocal, elementNum);
    DataCopy(sinGradGm[offset], sinGradLocal, elementNum);
    outQueueCosGrad.FreeTensor(cosGradLocal);
    outQueueSinGrad.FreeTensor(sinGradLocal);
}

template <typename T, int LAYOUT, bool IF_NEED_BACKWARD, bool IF_CAST, bool IF_ALIGN>
__aicore__ inline void RotateHalfGrad<T, LAYOUT, IF_NEED_BACKWARD, IF_CAST, IF_ALIGN>::CopyInPadOuter(
    uint64_t &offset, uint16_t blockCount, uint32_t blockLen, uint32_t srcStride, uint32_t dstStride)
{
    // copy pad in cos, sin

    DataCopyPadExtParams<T> copyPadParams{false, 0, 0, 0};

    DataCopyExtParams copyParams;
    copyParams.blockCount = blockCount;
    copyParams.blockLen = blockLen;
    copyParams.srcStride = srcStride;
    copyParams.dstStride = dstStride;

    LocalTensor<T> cosLocal = inQueueCos.AllocTensor<T>();
    LocalTensor<T> sinLocal = inQueueSin.AllocTensor<T>();
    DataCopyPad(cosLocal, cosGm[offset], copyParams, copyPadParams);
    DataCopyPad(sinLocal, sinGm[offset], copyParams, copyPadParams);
    inQueueCos.EnQue(cosLocal);
    inQueueSin.EnQue(sinLocal);
}

template <typename T, int LAYOUT, bool IF_NEED_BACKWARD, bool IF_CAST, bool IF_ALIGN>
__aicore__ inline void RotateHalfGrad<T, LAYOUT, IF_NEED_BACKWARD, IF_CAST, IF_ALIGN>::CopyInPadInner(
    uint64_t &offset, uint16_t blockCount, uint32_t blockLen, uint32_t srcStride, uint32_t dstStride)
{
    // copy pad in x, grad

    DataCopyPadExtParams<T> copyPadParams{false, 0, 0, 0};

    DataCopyExtParams copyParams;
    copyParams.blockCount = blockCount;
    copyParams.blockLen = blockLen;
    copyParams.srcStride = srcStride;
    copyParams.dstStride = dstStride;

    LocalTensor<T> gradLocal = inQueueGrad.AllocTensor<T>();
    DataCopyPad(gradLocal, gradGm[offset], copyParams, copyPadParams);
    DataCopyPad(gradLocal[halfDimDAlignNum], gradGm[offset + dimD / 2], copyParams, copyPadParams);
    inQueueGrad.EnQue(gradLocal);
    if constexpr (IF_NEED_BACKWARD) {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        DataCopyPad(xLocal, xGm[offset], copyParams, copyPadParams);
        DataCopyPad(xLocal[halfDimDAlignNum], xGm[offset + dimD / 2], copyParams, copyPadParams);
        inQueueX.EnQue(xLocal);
    }
}

template <typename T, int LAYOUT, bool IF_NEED_BACKWARD, bool IF_CAST, bool IF_ALIGN>
__aicore__ inline void RotateHalfGrad<T, LAYOUT, IF_NEED_BACKWARD, IF_CAST, IF_ALIGN>::CopyOutPadInner(
    uint64_t &offset, uint16_t blockCount, uint32_t blockLen, uint32_t srcStride, uint32_t dstStride)
{
    // copy pad out xGrad

    DataCopyExtParams copyParams;
    copyParams.blockCount = blockCount;
    copyParams.blockLen = blockLen;
    copyParams.srcStride = srcStride;
    copyParams.dstStride = dstStride;

    LocalTensor<T> xGradLocal = outQueueXGrad.DeQue<T>();
    DataCopyPad(xGradGm[offset], xGradLocal, copyParams);
    DataCopyPad(xGradGm[offset + dimD / 2], xGradLocal[halfDimDAlignNum], copyParams);
    outQueueXGrad.FreeTensor(xGradLocal);
}

template <typename T, int LAYOUT, bool IF_NEED_BACKWARD, bool IF_CAST, bool IF_ALIGN>
__aicore__ inline void RotateHalfGrad<T, LAYOUT, IF_NEED_BACKWARD, IF_CAST, IF_ALIGN>::CopyOutPadOuter(
    uint64_t &offset, uint16_t blockCount, uint32_t blockLen, uint32_t srcStride, uint32_t dstStride)
{
    // copy pad out cos, sin

    DataCopyExtParams copyParams;
    copyParams.blockCount = blockCount;
    copyParams.blockLen = blockLen;
    copyParams.srcStride = srcStride;
    copyParams.dstStride = dstStride;

    LocalTensor<T> cosGradLocal = outQueueCosGrad.DeQue<T>();
    LocalTensor<T> sinGradLocal = outQueueSinGrad.DeQue<T>();
    DataCopyPad(cosGradGm[offset], cosGradLocal, copyParams);
    DataCopyPad(sinGradGm[offset], sinGradLocal, copyParams);
    outQueueCosGrad.FreeTensor(cosGradLocal);
    outQueueSinGrad.FreeTensor(sinGradLocal);
}

template <typename T, int LAYOUT, bool IF_NEED_BACKWARD, bool IF_CAST, bool IF_ALIGN>
__aicore__ inline void RotateHalfGrad<T, LAYOUT, IF_NEED_BACKWARD, IF_CAST, IF_ALIGN>::DataChunkCat(
    const LocalTensor<float> &dstLocal, const LocalTensor<float> &src1Local, const LocalTensor<float> &src2Local,
    uint64_t offset, uint16_t blockCount, uint32_t blockLen, uint32_t srcStride, uint32_t dstStride)
{
    // chunk and concatenate

    DataCopyParams copyParams;
    copyParams.blockCount = blockCount;
    copyParams.blockLen = blockLen;
    copyParams.srcStride = srcStride;
    copyParams.dstStride = dstStride;

    DataCopy(dstLocal, src1Local[offset], copyParams);
    DataCopy(dstLocal[offset], src2Local, copyParams);
}

template <typename T, int LAYOUT, bool IF_NEED_BACKWARD, bool IF_CAST, bool IF_ALIGN>
__aicore__ inline void RotateHalfGrad<T, LAYOUT, IF_NEED_BACKWARD, IF_CAST, IF_ALIGN>::InitCastInOuter(
    const LocalTensor<T> &cosLocal, const LocalTensor<T> &sinLocal, uint64_t &elementNum)
{
    // init tmp tensor as all zero and cast in cos, sin

    if constexpr (IF_NEED_BACKWARD) {
        Duplicate(tmpOutCosGrad, (float)0.0, elementNum);
        Duplicate(tmpOutSinGrad, (float)0.0, elementNum);
    }
    if constexpr (IF_CAST) {
        Cast(cosCast, cosLocal, AscendC::RoundMode::CAST_NONE, elementNum);
        Cast(sinCast, sinLocal, AscendC::RoundMode::CAST_NONE, elementNum);
    }
}

template <typename T, int LAYOUT, bool IF_NEED_BACKWARD, bool IF_CAST, bool IF_ALIGN>
__aicore__ inline void RotateHalfGrad<T, LAYOUT, IF_NEED_BACKWARD, IF_CAST, IF_ALIGN>::CastOutOuter(
    const LocalTensor<T> &cosGradLocal, const LocalTensor<T> &sinGradLocal, uint64_t &elementNum)
{
    // cast out cos, sin

    PipeBarrier<PIPE_V>();
    if constexpr (IF_CAST) {
        Cast(cosGradLocal, tmpOutCosGrad, AscendC::RoundMode::CAST_RINT, elementNum);
        Cast(sinGradLocal, tmpOutSinGrad, AscendC::RoundMode::CAST_RINT, elementNum);
    } else {
        DataCopy(cosGradLocal, tmpOutCosGrad, elementNum);
        DataCopy(sinGradLocal, tmpOutSinGrad, elementNum);
    }
}

#endif // _ROTATE_HALF_GRAD_H_
