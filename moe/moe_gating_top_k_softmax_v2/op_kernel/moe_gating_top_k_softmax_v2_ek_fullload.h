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
 * \file moe_gating_top_k_softmax_v2_ek_fullload.h
 * \brief
 */
#ifndef MOE_GATING_TOP_K_SOFTMAX_V2_EK_FULLOAD
#define MOE_GATING_TOP_K_SOFTMAX_V2_EK_FULLOAD

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace MoeGatingTopKSoftmaxV2 {
using namespace AscendC;

constexpr int32_t ALIGNMENT_SIZE = 32;
constexpr int64_t REPEAT_MAX = 255;
constexpr int64_t COUNTINUOUS_DATA = 256;
constexpr int64_t FLOAT_BYTES = 4;
constexpr int64_t FLOAT_MASK = 64;
constexpr int64_t ALIGN_FACTOR = 32;

template <typename T, int32_t bufferNum, int32_t renorm>
class MoeGatingTopKSoftmaxV2EKFullLoad {
public:
    __aicore__ inline MoeGatingTopKSoftmaxV2EKFullLoad(){};
    __aicore__ inline void Init(GM_ADDR gating, GM_ADDR finished, GM_ADDR out, GM_ADDR indicesOut, GM_ADDR softmaxOut,
                                GM_ADDR workspace,
                                const MoeGatingTopKSoftmaxV2EKFullLoadTilingData *__restrict tilingData)
    {
        ParesTiling(tilingData);
        //   计算核块大小，获取当前核的起始索引
        int64_t formerblockLength = blockFormer * col;
        int64_t blockLength = (GetBlockIdx() != blockNum - 1) ? formerblockLength : blockTail * col;
        gatingTensorGM.SetGlobalBuffer((__gm__ T *)gating + formerblockLength * GetBlockIdx(), blockLength);
        if (finished != nullptr) {
            exitFinished = true;
            int64_t blockLengthFinished = (GetBlockIdx() != blockNum - 1) ? blockFormer : blockTail;
            finishedTensorGM.SetGlobalBuffer((__gm__ bool *)finished + blockFormer * GetBlockIdx(),
                                             blockLengthFinished);
        }

        int64_t outFormerBlockLength = blockFormer * k;
        int64_t outBlockLength = (GetBlockIdx() != blockNum - 1) ? outFormerBlockLength : blockTail * k;
        outTensorGM.SetGlobalBuffer((__gm__ T *)out + outFormerBlockLength * GetBlockIdx(), outBlockLength);
        indicesOutTensorGM.SetGlobalBuffer((__gm__ int32_t *)indicesOut + outFormerBlockLength * GetBlockIdx(),
                                           outBlockLength);
        softmaxOutTensorGM.SetGlobalBuffer((__gm__ float *)softmaxOut + formerblockLength * GetBlockIdx(), blockLength);

        if constexpr (renorm == 0 || IsSameType<T, bfloat16_t>::value) {
            pipe.InitBuffer(gatingQueue, bufferNum, ubFormer * colAlign * sizeof(float));
        } else {
            pipe.InitBuffer(gatingQueue, bufferNum, ubFormer * colAlign * sizeof(T));
        }

        pipe.InitBuffer(finishedQueue, bufferNum,
                        (ubFormer * sizeof(bool) + ALIGNMENT_SIZE - 1) / ALIGNMENT_SIZE * ALIGNMENT_SIZE);
        pipe.InitBuffer(outQueue, bufferNum, ubFormer * tilingData->kAlignT);
        pipe.InitBuffer(indicesOutQueue, bufferNum, ubFormer * kAlignB32);
        pipe.InitBuffer(softmaxOutQueue, bufferNum, ubFormer * colAlign * sizeof(float));
        pipe.InitBuffer(oneDimTensorUb, colAlign * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        int32_t curRowNum = ubFormer;
        SoftMaxTiling *softmaxTilingData = &formerSoftmaxTilingData;
        TopkTiling *topKTilingData = &formerTopkTilingData;
        int64_t ubLoopCount = (GetBlockIdx() == blockNum - 1) ? ubLoopOfTailBlock : ubLoopOfFormerBlock;

        for (int64_t i = 0; i < ubLoopCount - 1; i++) {
            CopyIn(i, ubFormer);
            if constexpr (renorm == 0) {
                if constexpr (IsSameType<T, float>::value) {
                    Compute(i, ubFormer, softmaxTilingData, topKTilingData);
                } else {
                    ComputeCast(i, ubFormer, softmaxTilingData, topKTilingData);
                }
            } else {
                if constexpr (IsSameType<T, bfloat16_t>::value) {
                    ComputeRenormCast(i, ubFormer, softmaxTilingData, topKTilingData);
                } else {
                    ComputeRenorm(i, ubFormer, softmaxTilingData, topKTilingData);
                }
            }
            CopyOut(i, ubFormer);
        }

        if (GetBlockIdx() < blockNum - 1) {
            curRowNum = ubTailOfFormerBlock;
            softmaxTilingData = &formerBlockTailSoftmaxTilingData;
            topKTilingData = &formerBlockTailTopkTilingData;
        } else {
            curRowNum = ubTailOfTailBlock;
            softmaxTilingData = &tailBlockTailSoftmaxTilingData;
            topKTilingData = &tailBlockTailTopkTilingData;
        }
        CopyIn(ubLoopCount - 1, curRowNum);
        if constexpr (renorm == 0) {
            if constexpr (IsSameType<T, float>::value) {
                Compute(ubLoopCount - 1, curRowNum, softmaxTilingData, topKTilingData);
            } else {
                ComputeCast(ubLoopCount - 1, curRowNum, softmaxTilingData, topKTilingData);
            }
        } else {
            if constexpr (IsSameType<T, bfloat16_t>::value) {
                ComputeRenormCast(ubLoopCount - 1, curRowNum, softmaxTilingData, topKTilingData);
            } else {
                ComputeRenorm(ubLoopCount - 1, curRowNum, softmaxTilingData, topKTilingData);
            }
        }
        CopyOut(ubLoopCount - 1, curRowNum);
    }

private:
    template <typename U>
    __aicore__ inline void CastComputeAlignmentBefore(LocalTensor<float> &dst, LocalTensor<U> &src,
                                                      const int32_t rowCount, const int32_t colCount)
    {
        int32_t repeatTimesOneRow = (colCount * sizeof(float) + COUNTINUOUS_DATA - 1) / COUNTINUOUS_DATA;
        int32_t colCountAlign = (colCount + ALIGNMENT_SIZE - 1) / ALIGNMENT_SIZE * ALIGNMENT_SIZE;
        UnaryRepeatParams repeatParams;
        if (rowCount > repeatTimesOneRow - 1 ||
            (colCount + ALIGNMENT_SIZE - 1) / ALIGNMENT_SIZE * sizeof(float) > REPEAT_MAX) {
            for (int32_t i = 0; i < rowCount; i++) {
                repeatParams.dstBlkStride = 1;
                repeatParams.srcBlkStride = 1;
                repeatParams.dstRepStride = COUNTINUOUS_DATA / ALIGNMENT_SIZE;
                repeatParams.srcRepStride = (COUNTINUOUS_DATA / FLOAT_BYTES) * sizeof(U) / ALIGNMENT_SIZE;
                Cast(dst[i * ((colCount + ALIGNMENT_SIZE - 1) / ALIGNMENT_SIZE * ALIGNMENT_SIZE)],
                     src[i * ((colCount + ALIGNMENT_SIZE - 1) / ALIGNMENT_SIZE * ALIGNMENT_SIZE)], RoundMode::CAST_NONE,
                     FLOAT_MASK, repeatTimesOneRow - 1, repeatParams);
            }

            repeatParams.dstBlkStride = 1;
            repeatParams.srcBlkStride = 1;
            repeatParams.dstRepStride = (colCount + ALIGNMENT_SIZE - 1) / ALIGNMENT_SIZE * sizeof(float);
            repeatParams.srcRepStride = (colCount + ALIGNMENT_SIZE - 1) / ALIGNMENT_SIZE * sizeof(U);
            int32_t mask = colCount - FLOAT_MASK * (repeatTimesOneRow - 1);
            if ((colCount + ALIGNMENT_SIZE - 1) / ALIGNMENT_SIZE * sizeof(float) <= REPEAT_MAX) {
                Cast(dst[(repeatTimesOneRow - 1) * COUNTINUOUS_DATA / FLOAT_BYTES],
                     src[(repeatTimesOneRow - 1) * COUNTINUOUS_DATA / FLOAT_BYTES], RoundMode::CAST_NONE, mask,
                     rowCount, repeatParams);
            } else {
                for (int32_t i = 0; i < rowCount; i++) {
                    Cast(dst[(repeatTimesOneRow - 1) * COUNTINUOUS_DATA / FLOAT_BYTES + i * colCountAlign],
                         src[(repeatTimesOneRow - 1) * COUNTINUOUS_DATA / FLOAT_BYTES + i * colCountAlign],
                         RoundMode::CAST_NONE, mask, 1, {1, 1, 1, 1});
                }
            }
        } else {
            for (int32_t i = 0; i < repeatTimesOneRow - 1; i++) {
                repeatParams.dstBlkStride = 1;
                repeatParams.srcBlkStride = 1;
                repeatParams.dstRepStride = (colCount + ALIGNMENT_SIZE - 1) / ALIGNMENT_SIZE * sizeof(float);
                repeatParams.srcRepStride = (colCount + ALIGNMENT_SIZE - 1) / ALIGNMENT_SIZE * sizeof(U);
                Cast(dst[i * (COUNTINUOUS_DATA / FLOAT_BYTES)], src[i * (COUNTINUOUS_DATA / FLOAT_BYTES)],
                     RoundMode::CAST_NONE, FLOAT_MASK, rowCount, repeatParams);
            }

            int32_t mask = colCount - FLOAT_MASK * (repeatTimesOneRow - 1);
            Cast(dst[(repeatTimesOneRow - 1) * (COUNTINUOUS_DATA / FLOAT_BYTES)],
                 src[(repeatTimesOneRow - 1) * (COUNTINUOUS_DATA / FLOAT_BYTES)], RoundMode::CAST_NONE, mask, rowCount,
                 repeatParams);
        }
    }

    template <typename U>
    __aicore__ inline void CastComputeAlignmentAfter(LocalTensor<U> &dst, LocalTensor<float> &src,
                                                     const int32_t rowCount, const int32_t colCount)
    {
        int32_t repeatTimesOneRow = (colCount * sizeof(float) + COUNTINUOUS_DATA - 1) / COUNTINUOUS_DATA;

        UnaryRepeatParams repeatParams;
        if (rowCount > repeatTimesOneRow - 1) {
            for (int32_t i = 0; i < rowCount; i++) {
                repeatParams.dstBlkStride = 1;
                repeatParams.srcBlkStride = 1;
                repeatParams.dstRepStride = (COUNTINUOUS_DATA / FLOAT_BYTES) * sizeof(U) / ALIGNMENT_SIZE;
                repeatParams.srcRepStride = COUNTINUOUS_DATA / ALIGNMENT_SIZE;
                Cast(dst[i * ((colCount * sizeof(U) + ALIGNMENT_SIZE - 1) / ALIGNMENT_SIZE * ALIGNMENT_SIZE) /
                         sizeof(U)],
                     src[i * ((colCount * sizeof(float) + ALIGNMENT_SIZE - 1) / ALIGNMENT_SIZE * ALIGNMENT_SIZE) /
                         sizeof(float)],
                     RoundMode::CAST_ROUND, FLOAT_MASK, repeatTimesOneRow - 1, repeatParams);
            }

            repeatParams.dstBlkStride = 1;
            repeatParams.srcBlkStride = 1;
            repeatParams.dstRepStride = (colCount * sizeof(U) + ALIGNMENT_SIZE - 1) / ALIGNMENT_SIZE;
            repeatParams.srcRepStride = (colCount * sizeof(float) + ALIGNMENT_SIZE - 1) / ALIGNMENT_SIZE;
            int32_t mask =
                FLOAT_MASK - (((repeatTimesOneRow * COUNTINUOUS_DATA) - (colCount * FLOAT_BYTES)) / FLOAT_BYTES);
            Cast(dst[(repeatTimesOneRow - 1) * (COUNTINUOUS_DATA / FLOAT_BYTES)],
                 src[(repeatTimesOneRow - 1) * (COUNTINUOUS_DATA / FLOAT_BYTES)], RoundMode::CAST_ROUND, mask, rowCount,
                 repeatParams);
        } else {
            for (int32_t i = 0; i < repeatTimesOneRow - 1; i++) {
                repeatParams.dstBlkStride = 1;
                repeatParams.srcBlkStride = 1;
                repeatParams.dstRepStride = (colCount * sizeof(U) + ALIGNMENT_SIZE - 1) / ALIGNMENT_SIZE;
                repeatParams.srcRepStride = (colCount * sizeof(float) + ALIGNMENT_SIZE - 1) / ALIGNMENT_SIZE;
                Cast(dst[i * (COUNTINUOUS_DATA / FLOAT_BYTES)], src[i * (COUNTINUOUS_DATA / FLOAT_BYTES)],
                     RoundMode::CAST_ROUND, FLOAT_MASK, rowCount, repeatParams);
            }

            int32_t mask =
                FLOAT_MASK - (((repeatTimesOneRow * COUNTINUOUS_DATA) - (colCount * FLOAT_BYTES)) / FLOAT_BYTES);
            Cast(dst[(repeatTimesOneRow - 1) * (COUNTINUOUS_DATA / FLOAT_BYTES)],
                 src[(repeatTimesOneRow - 1) * (COUNTINUOUS_DATA / FLOAT_BYTES)], RoundMode::CAST_ROUND, mask, rowCount,
                 repeatParams);
        }
    }

    template <typename U>
    __aicore__ inline void VectorDup(LocalTensor<U> &dst, const int32_t rowCount, const int32_t colCount)
    {
        if (colAlign - colCount != 0) {
            U scalar;
            if constexpr (IsSameType<U, half>::value) {
                uint16_t tmp = 0xFC00; // -inf
                scalar = *((half *)&tmp);
            } else if constexpr (IsSameType<U, bfloat16_t>::value) {
                uint16_t tmp = 0xFF80; // -inf
                scalar = *((bfloat16_t *)&tmp);
            } else {
                uint32_t tmp = 0xFF800000; // -inf
                scalar = *((float *)&tmp);
            }
            // 当对齐后大小与实际大小不一致，需要将 colCount到colAlign之间的数据掩成-1
            uint64_t mask[2] = {(((uint64_t)1 << (colAlign - colCount)) - 1) << (ALIGN_FACTOR - (colAlign - colCount)),
                                0};
            if (colAlign * sizeof(U) / ALIGNMENT_SIZE <= REPEAT_MAX) {
                Duplicate(dst[colAlign - ALIGN_FACTOR], scalar, mask, rowCount, 1,
                          colAlign * sizeof(U) / ALIGNMENT_SIZE);
            } else {
                for (int32_t i = 0; i < rowCount; i++) {
                    Duplicate(dst[i * colAlign + colAlign - ALIGN_FACTOR], scalar, mask, 1, 1, 1);
                }
            }
        }
    }

    __aicore__ inline void CopyIn(int32_t progress, int32_t curRowsNum)
    {
        LocalTensor<T> gatingLocal = gatingQueue.template AllocTensor<T>();

        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyParams intriParams;
        intriParams.blockCount = curRowsNum;
        intriParams.blockLen = col * sizeof(T);
        intriParams.srcStride = 0;
        intriParams.dstStride =
            (colAlign * sizeof(T) - (((col * sizeof(T) + ALIGNMENT_SIZE - 1) / ALIGNMENT_SIZE) * ALIGNMENT_SIZE)) /
            ALIGNMENT_SIZE;

        DataCopyPad(gatingLocal, gatingTensorGM[ubFormer * col * progress], intriParams, padParams);
        gatingQueue.EnQue(gatingLocal);

        if (exitFinished) {
            LocalTensor<bool> finishedLocal = finishedQueue.template AllocTensor<bool>();
            DataCopyParams intriParamsFinished{1, static_cast<uint16_t>(curRowsNum * sizeof(bool)), 0, 0};
            DataCopyPad(finishedLocal, finishedTensorGM[ubFormer * progress], intriParamsFinished, padParams);
            finishedQueue.EnQue(finishedLocal);
        }
    }

    __aicore__ inline void Compute(int32_t progress, int32_t curRowsNum, const SoftMaxTiling *softmaxTilingData,
                                   const TopkTiling *topKTilingData)
    {
        SoftMaxShapeInfo softmaxShapeInfoData{(uint32_t)curRowsNum, colAlign, (uint32_t)curRowsNum, col};
        TopKInfo topKInfoData{curRowsNum, (int32_t)colAlign, (int32_t)col};

        LocalTensor<T> gatingLocal = gatingQueue.template DeQue<T>();
        LocalTensor<bool> finishedLocal;
        if (exitFinished) {
            finishedLocal = finishedQueue.template DeQue<bool>();
        }

        LocalTensor<T> outLocal = outQueue.template AllocTensor<T>();
        LocalTensor<int32_t> indicesOutLocal = indicesOutQueue.template AllocTensor<int32_t>();
        LocalTensor<int32_t> oneDimTensor = oneDimTensorUb.template Get<int32_t>();

        LocalTensor<T> softmaxOutBuffer = softmaxOutQueue.template AllocTensor<T>();
        SoftMax<T, true, false>(softmaxOutBuffer, gatingLocal, *softmaxTilingData, softmaxShapeInfoData);
        gatingQueue.FreeTensor(gatingLocal);
        PipeBarrier<PIPE_V>();

        if (softmaxFlag == 1) {
            DataCopyParams intriParams;
            intriParams.blockCount = curRowsNum;
            intriParams.blockLen = col * sizeof(float);
            intriParams.srcStride =
                (colAlign * sizeof(T) - (((col * sizeof(T) + ALIGNMENT_SIZE - 1) / ALIGNMENT_SIZE) * ALIGNMENT_SIZE)) /
                ALIGNMENT_SIZE;
            intriParams.dstStride = 0;
            softmaxOutQueue.EnQue(softmaxOutBuffer);
            softmaxOutBuffer = softmaxOutQueue.template DeQue<float>();
            DataCopyPad(softmaxOutTensorGM[ubFormer * col * progress], softmaxOutBuffer, intriParams);
        }

        VectorDup<T>(softmaxOutBuffer, curRowsNum, col);
        PipeBarrier<PIPE_V>();

        ArithProgression(oneDimTensor, 0, 1, colAlign);
        PipeBarrier<PIPE_V>();

        if (exitFinished) {
            TopK<T, true, true, true, TopKMode::TOPK_NORMAL>(outLocal, indicesOutLocal, softmaxOutBuffer, oneDimTensor,
                                                             finishedLocal, k, *topKTilingData, topKInfoData, true);
        } else {
            TopK<T, true, false, true, TopKMode::TOPK_NORMAL>(outLocal, indicesOutLocal, softmaxOutBuffer, oneDimTensor,
                                                              finishedLocal, k, *topKTilingData, topKInfoData, true);
        }
        PipeBarrier<PIPE_V>();
        softmaxOutQueue.FreeTensor(softmaxOutBuffer);

        outQueue.EnQue(outLocal);
        indicesOutQueue.EnQue(indicesOutLocal);

        if (exitFinished) {
            finishedQueue.FreeTensor(finishedLocal);
        }
    }

    __aicore__ inline void ComputeCast(int32_t progress, int32_t curRowsNum, const SoftMaxTiling *softmaxTilingData,
                                       const TopkTiling *topKTilingData)
    {
        SoftMaxShapeInfo softmaxShapeInfoData{(uint32_t)curRowsNum, colAlign, (uint32_t)curRowsNum, col};
        TopKInfo topKInfoData{curRowsNum, (int32_t)colAlign, (int32_t)col};

        LocalTensor<T> gatingLocal = gatingQueue.template DeQue<T>();
        LocalTensor<bool> finishedLocal;
        if (exitFinished) {
            finishedLocal = finishedQueue.template DeQue<bool>();
        }

        LocalTensor<T> outLocal = outQueue.template AllocTensor<T>();
        LocalTensor<int32_t> indicesOutLocal = indicesOutQueue.template AllocTensor<int32_t>();
        LocalTensor<int32_t> oneDimTensor = oneDimTensorUb.template Get<int32_t>();

        LocalTensor<float> castBuffer = softmaxOutQueue.template AllocTensor<float>();
        CastComputeAlignmentBefore(castBuffer, gatingLocal, curRowsNum, col);
        gatingQueue.FreeTensor(gatingLocal);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> softmaxOutBuffer = gatingQueue.template AllocTensor<float>();
        SoftMax<float, true, false>(softmaxOutBuffer, castBuffer, *softmaxTilingData, softmaxShapeInfoData);
        PipeBarrier<PIPE_V>();

        if (softmaxFlag == 1) {
            DataCopyParams intriParams{static_cast<uint16_t>(curRowsNum), static_cast<uint16_t>(col * sizeof(float)), 0,
                                       0};
            intriParams.srcStride = (colAlign * sizeof(float) -
                                     (((col * sizeof(float) + ALIGNMENT_SIZE - 1) / ALIGNMENT_SIZE) * ALIGNMENT_SIZE)) /
                                    ALIGNMENT_SIZE;
            auto eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventID);
            WaitFlag<HardEvent::V_MTE3>(eventID);
            DataCopyPad(softmaxOutTensorGM[ubFormer * col * progress], softmaxOutBuffer, intriParams);
            auto eventID2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(eventID2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventID2);
        }

        VectorDup<float>(softmaxOutBuffer, curRowsNum, col);
        PipeBarrier<PIPE_V>();

        ArithProgression(oneDimTensor, 0, 1, colAlign);
        PipeBarrier<PIPE_V>();

        if (exitFinished) {
            TopK<float, true, true, true, TopKMode::TOPK_NORMAL>(castBuffer, indicesOutLocal, softmaxOutBuffer,
                                                                 oneDimTensor, finishedLocal, k, *topKTilingData,
                                                                 topKInfoData, true);
        } else {
            TopK<float, true, false, true, TopKMode::TOPK_NORMAL>(castBuffer, indicesOutLocal, softmaxOutBuffer,
                                                                  oneDimTensor, finishedLocal, k, *topKTilingData,
                                                                  topKInfoData, true);
        }
        PipeBarrier<PIPE_V>();

        CastComputeAlignmentAfter(outLocal, castBuffer, curRowsNum, k);
        softmaxOutQueue.FreeTensor(castBuffer);
        gatingQueue.FreeTensor(softmaxOutBuffer);

        outQueue.EnQue(outLocal);
        indicesOutQueue.EnQue(indicesOutLocal);

        if (exitFinished) {
            finishedQueue.FreeTensor(finishedLocal);
        }
    }

    __aicore__ inline void ComputeRenorm(int32_t progress, int32_t curRowsNum, const SoftMaxTiling *softmaxTilingData,
                                         const TopkTiling *topKTilingData)
    {
        SoftMaxShapeInfo softmaxShapeInfoData;
        if constexpr (IsSameType<T, half>::value) {
            softmaxShapeInfoData.srcK = kAlignB16 / sizeof(half);
        } else {
            softmaxShapeInfoData.srcK = kAlignB32 / sizeof(float);
        }
        softmaxShapeInfoData.srcM = curRowsNum;
        softmaxShapeInfoData.oriSrcK = k;
        softmaxShapeInfoData.oriSrcM = curRowsNum;
        TopKInfo topKInfoData;
        topKInfoData.outter = curRowsNum;
        topKInfoData.inner = colAlign;
        topKInfoData.n = col;

        LocalTensor<T> gatingLocal = gatingQueue.template DeQue<T>();
        LocalTensor<bool> finishedLocal;
        if (exitFinished) {
            finishedLocal = finishedQueue.template DeQue<bool>();
        }

        LocalTensor<T> outLocal = outQueue.template AllocTensor<T>();
        LocalTensor<int32_t> indicesOutLocal = indicesOutQueue.template AllocTensor<int32_t>();
        LocalTensor<int32_t> oneDimTensor = oneDimTensorUb.template Get<int32_t>();

        VectorDup<T>(gatingLocal, curRowsNum, col);
        PipeBarrier<PIPE_V>();

        ArithProgression(oneDimTensor, 0, 1, colAlign);
        PipeBarrier<PIPE_V>();

        LocalTensor<T> topkBuffer = softmaxOutQueue.template AllocTensor<T>();
        if (exitFinished) {
            TopK<T, true, true, true, TopKMode::TOPK_NORMAL>(topkBuffer, indicesOutLocal, gatingLocal, oneDimTensor,
                                                             finishedLocal, k, *topKTilingData, topKInfoData, true);
        } else {
            TopK<T, true, false, true, TopKMode::TOPK_NORMAL>(topkBuffer, indicesOutLocal, gatingLocal, oneDimTensor,
                                                              finishedLocal, k, *topKTilingData, topKInfoData, true);
        }
        PipeBarrier<PIPE_V>();

        SoftMax<T, true, false>(outLocal, topkBuffer, *softmaxTilingData, softmaxShapeInfoData);

        softmaxOutQueue.FreeTensor(topkBuffer);
        gatingQueue.FreeTensor(gatingLocal);

        outQueue.EnQue(outLocal);
        indicesOutQueue.EnQue(indicesOutLocal);

        if (exitFinished) {
            finishedQueue.FreeTensor(finishedLocal);
        }
    }

    __aicore__ inline void ComputeRenormCast(int32_t progress, int32_t curRowsNum,
                                             const SoftMaxTiling *softmaxTilingData, const TopkTiling *topKTilingData)
    {
        SoftMaxShapeInfo softmaxShapeInfoData;
        if constexpr (IsSameType<T, half>::value) {
            softmaxShapeInfoData.srcK = kAlignB16 / sizeof(half);
        } else {
            softmaxShapeInfoData.srcK = kAlignB32 / sizeof(float);
        }
        softmaxShapeInfoData.srcM = curRowsNum;
        softmaxShapeInfoData.oriSrcK = k;
        softmaxShapeInfoData.oriSrcM = curRowsNum;
        TopKInfo topKInfoData;
        topKInfoData.outter = curRowsNum;
        topKInfoData.inner = colAlign;
        topKInfoData.n = col;

        LocalTensor<T> gatingLocal = gatingQueue.template DeQue<T>();
        LocalTensor<bool> finishedLocal;
        if (exitFinished) {
            finishedLocal = finishedQueue.template DeQue<bool>();
        }

        LocalTensor<T> outLocal = outQueue.template AllocTensor<T>();
        LocalTensor<int32_t> indicesOutLocal = indicesOutQueue.template AllocTensor<int32_t>();
        LocalTensor<int32_t> oneDimTensor = oneDimTensorUb.template Get<int32_t>();

        LocalTensor<float> castBuffer = softmaxOutQueue.template AllocTensor<float>();
        CastComputeAlignmentBefore(castBuffer, gatingLocal, curRowsNum, col);
        gatingQueue.FreeTensor(gatingLocal);
        PipeBarrier<PIPE_V>();

        VectorDup<float>(castBuffer, curRowsNum, col);
        PipeBarrier<PIPE_V>();

        ArithProgression(oneDimTensor, 0, 1, colAlign);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> topkBuffer = gatingQueue.template AllocTensor<float>();
        if (exitFinished) {
            TopK<float, true, true, true, TopKMode::TOPK_NORMAL>(topkBuffer, indicesOutLocal, castBuffer, oneDimTensor,
                                                                 finishedLocal, k, *topKTilingData, topKInfoData, true);
        } else {
            TopK<float, true, false, true, TopKMode::TOPK_NORMAL>(topkBuffer, indicesOutLocal, castBuffer, oneDimTensor,
                                                                  finishedLocal, k, *topKTilingData, topKInfoData,
                                                                  true);
        }
        PipeBarrier<PIPE_V>();

        SoftMax<float, true, false>(castBuffer, topkBuffer, *softmaxTilingData, softmaxShapeInfoData);
        PipeBarrier<PIPE_V>();

        CastComputeAlignmentAfter(outLocal, castBuffer, curRowsNum, k);
        softmaxOutQueue.FreeTensor(castBuffer);
        gatingQueue.FreeTensor(topkBuffer);

        outQueue.EnQue(outLocal);
        indicesOutQueue.EnQue(indicesOutLocal);

        if (exitFinished) {
            finishedQueue.FreeTensor(finishedLocal);
        }
    }

    __aicore__ inline void CopyOut(int32_t progress, int32_t curRowsNum)
    {
        LocalTensor<T> newOutLocal = outQueue.template DeQue<T>();
        LocalTensor<int32_t> newIndicesOutLocal = indicesOutQueue.template DeQue<int32_t>();

        int64_t gmIndex = ubFormer * k * progress;
        DataCopyParams intriParams{static_cast<uint16_t>(curRowsNum), static_cast<uint16_t>(k * sizeof(T)), 0, 0};
        DataCopyPad(outTensorGM[gmIndex], newOutLocal, intriParams);
        intriParams.blockLen = k * sizeof(int32_t);
        DataCopyPad(indicesOutTensorGM[gmIndex], newIndicesOutLocal, intriParams);
        outQueue.FreeTensor(newOutLocal);
        indicesOutQueue.FreeTensor(newIndicesOutLocal);
    }

    __aicore__ inline void ParesTiling(const MoeGatingTopKSoftmaxV2EKFullLoadTilingData *__restrict tilingData)
    {
        row = tilingData->row;
        col = tilingData->col;
        colAlign = tilingData->colAlign;
        k = tilingData->k;
        kAlignB16 = tilingData->kAlignB16;
        kAlignB32 = tilingData->kAlignB32;
        blockNum = tilingData->blockNum;
        blockFormer = tilingData->blockFormer;
        blockTail = tilingData->blockTail;
        ubLoopOfFormerBlock = tilingData->ubLoopOfFormerBlock;
        ubLoopOfTailBlock = tilingData->ubLoopOfTailBlock;
        ubFormer = tilingData->ubFormer;
        ubTailOfFormerBlock = tilingData->ubTailOfFormerBlock;
        ubTailOfTailBlock = tilingData->ubTailOfTailBlock;
        formerSoftmaxTilingData = tilingData->formerSoftmaxTilingData;
        formerBlockTailSoftmaxTilingData = tilingData->formerBlockTailSoftmaxTilingData;
        tailBlockTailSoftmaxTilingData = tilingData->tailBlockTailSoftmaxTilingData;
        formerTopkTilingData = tilingData->formerTopkTilingData;
        formerBlockTailTopkTilingData = tilingData->formerBlockTailTopkTilingData;
        tailBlockTailTopkTilingData = tilingData->tailBlockTailTopkTilingData;
        softmaxFlag = tilingData->softmaxFlag;
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, bufferNum> gatingQueue, finishedQueue;
    TQue<QuePosition::VECOUT, bufferNum> outQueue, indicesOutQueue, softmaxOutQueue;

    TBuf<> oneDimTensorUb;

    GlobalTensor<T> gatingTensorGM;
    GlobalTensor<bool> finishedTensorGM;
    GlobalTensor<T> outTensorGM;
    GlobalTensor<int32_t> indicesOutTensorGM;
    GlobalTensor<float> softmaxOutTensorGM;

    uint32_t row;
    uint32_t col;
    uint32_t colAlign;
    uint32_t k;
    uint32_t kAlignB16;
    uint32_t kAlignB32;
    uint32_t blockNum;
    uint32_t blockFormer;
    uint32_t blockTail;
    uint32_t ubLoopOfFormerBlock;
    uint32_t ubLoopOfTailBlock;
    uint32_t ubFormer;
    uint32_t ubTailOfFormerBlock;
    uint32_t ubTailOfTailBlock;
    uint32_t softmaxFlag;
    SoftMaxTiling formerSoftmaxTilingData;
    SoftMaxTiling formerBlockTailSoftmaxTilingData;
    SoftMaxTiling tailBlockTailSoftmaxTilingData;
    TopkTiling formerTopkTilingData;
    TopkTiling formerBlockTailTopkTilingData;
    TopkTiling tailBlockTailTopkTilingData;

    bool exitFinished{false};
};
} // namespace MoeGatingTopKSoftmaxV2
#endif
