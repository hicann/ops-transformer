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
 * \file moe_gating_top_k_softmax_v2_k_renorm.h
 * \brief
 */
#ifndef MOE_GATING_TOP_K_SOFTMAX_V2_K_RENORM
#define MOE_GATING_TOP_K_SOFTMAX_V2_K_RENORM

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace MoeGatingTopKSoftmaxV2 {
using namespace AscendC;
constexpr int32_t DB_RENORM = 2;

template <typename T, int32_t renorm>
class MoeGatingTopKSoftmaxV2KRenorm {
public:
    __aicore__ inline MoeGatingTopKSoftmaxV2KRenorm(){};
    __aicore__ inline void Init(GM_ADDR gating, GM_ADDR finished, GM_ADDR out, GM_ADDR indicesOut, GM_ADDR softmaxOut,
                                GM_ADDR workspace,
                                const MoeGatingTopKSoftmaxV2KFullLoadTilingData *__restrict tilingData)
    {
        ParesTiling(tilingData);
        //  计算核块大小，获取当前核的起始索引
        int64_t formerblockLength = blockFormer * col;
        int64_t blockLength = (GetBlockIdx() != blockNum - 1) ? formerblockLength : blockTail * col;
        gatingTensorGM.SetGlobalBuffer((__gm__ T *)gating + formerblockLength * GetBlockIdx(), blockLength);
        if (finished != nullptr) {
            int64_t blockLengthFinished = (GetBlockIdx() != blockNum - 1) ? blockFormer : blockTail;
            finishedTensorGM.SetGlobalBuffer((__gm__ bool *)finished + blockFormer * GetBlockIdx(),
                                             blockLengthFinished);
            exitFinished = true;
        }

        int64_t outFormerBlockLength = blockFormer * k;
        int64_t outBlockLength = (GetBlockIdx() != blockNum - 1) ? outFormerBlockLength : blockTail * k;
        outTensorGM.SetGlobalBuffer((__gm__ T *)out + outFormerBlockLength * GetBlockIdx(), outBlockLength);
        indicesOutTensorGM.SetGlobalBuffer((__gm__ int32_t *)indicesOut + outFormerBlockLength * GetBlockIdx(),
                                           outBlockLength);

        pipe.InitBuffer(gatingQueue, DB_RENORM, (kAlign + ubFormerAlign) * sizeof(float));
        pipe.InitBuffer(finishedQueue, DB_RENORM, ALIGNMENT_SIZE);
        pipe.InitBuffer(indicesOutQueue, DB_RENORM, (kAlign + ubFormerAlign) * sizeof(int32_t));
        pipe.InitBuffer(outQueue, DB_RENORM, kAlign * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int64_t rowCount = (GetBlockIdx() < blockNum - 1) ? blockFormer : blockTail;

        for (int64_t rowIdx = 0; rowIdx < rowCount; rowIdx++) {
            LocalTensor<bool> finishedLocal;
            float softmaxSum = 0.0f;
            if (exitFinished) {
                CopyInFinished(rowIdx);
                finishedLocal = finishedQueue.template DeQue<bool>();
            }

            LocalTensor<int32_t> indicesOutLocal = indicesOutQueue.template AllocTensor<int32_t>();

            if constexpr (IsSameType<T, bfloat16_t>::value) {
                LocalTensor<float> gatingLocal = gatingQueue.template AllocTensor<float>();

                for (int64_t ubIdx = 0; ubIdx < ubLoop; ubIdx++) {
                    int32_t curCol = (ubIdx < ubLoop - 1) ? ubFormer : ubTail;
                    CopyInGating<float>(rowIdx, ubIdx, curCol, gatingLocal);
                    ComputeTopK<float>(rowIdx, ubIdx, curCol, indicesOutLocal, finishedLocal);
                }
                ComputeSoftmax<float>(gatingLocal, indicesOutLocal);
            } else {
                LocalTensor<T> gatingLocal = gatingQueue.template AllocTensor<T>();

                for (int64_t ubIdx = 0; ubIdx < ubLoop; ubIdx++) {
                    int32_t curCol = (ubIdx < ubLoop - 1) ? ubFormer : ubTail;
                    CopyInGating<T>(rowIdx, ubIdx, curCol, gatingLocal);
                    ComputeTopK<T>(rowIdx, ubIdx, curCol, indicesOutLocal, finishedLocal);
                }
                ComputeSoftmax<T>(gatingLocal, indicesOutLocal);
            }
            if (exitFinished) {
                if (finishedLocal.GetValue(0)) {
                    Duplicate(indicesOutLocal, static_cast<int32_t>(col), k);
                }
                finishedQueue.FreeTensor(finishedLocal);
            }
            CopyOut(rowIdx);
        }
    }

private:
    __aicore__ inline int64_t Align(const int64_t elementNum)
    {
        return (elementNum + ALIGN_FACTOR - 1) / ALIGN_FACTOR * ALIGN_FACTOR;
    }

    template <typename U>
    __aicore__ inline U VectorDup(LocalTensor<U> &dst, const int32_t colCount, const int32_t colCountAlign)
    {
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
        if (colCountAlign - colCount != 0) {
            // 当对齐后大小与实际大小不一致，需要将 colCount到colAlign之间的数据掩成-1
            uint64_t mask[2] = {
                (((uint64_t)1 << (colCountAlign - colCount)) - 1) << (ALIGN_FACTOR - (colCountAlign - colCount)), 0};
            Duplicate(dst[(kAlign + colCountAlign) - ALIGN_FACTOR], scalar, mask, 1, 1, 1);
        }
        return scalar;
    }

    __aicore__ inline void CopyInFinished(int32_t progress)
    {
        LocalTensor<bool> finishedLocal = finishedQueue.template AllocTensor<bool>();

        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyParams intriParams{1, static_cast<uint16_t>(sizeof(bool)), 0, 0};
        DataCopyPad(finishedLocal, finishedTensorGM[progress], intriParams, padParams);

        finishedQueue.EnQue(finishedLocal);
    }

    template <typename U>
    __aicore__ inline void CopyInGating(int32_t progress, int32_t ubIdx, int32_t curCol, LocalTensor<U> &gatingLocal)
    {
        DataCopyPadExtParams<T> padParams{false, 0, 0, (T)0};
        DataCopyExtParams intriParams{1, static_cast<uint32_t>(curCol * sizeof(T)), 0, 0, 0};

        if constexpr (IsSameType<T, bfloat16_t>::value) {
            DataCopyPad(gatingLocal.template ReinterpretCast<bfloat16_t>()[kAlign * 2 + ubFormerAlign],
                        gatingTensorGM[col * progress + ubFormer * ubIdx], intriParams, padParams);
        } else {
            DataCopyPad(gatingLocal[kAlign], gatingTensorGM[col * progress + ubFormer * ubIdx], intriParams, padParams);
        }
        gatingQueue.EnQue(gatingLocal);
    }

    template <typename U>
    __aicore__ inline void ComputeTopK(int32_t progress, int32_t ubIdx, int32_t curCol,
                                       LocalTensor<int32_t> &indicesOutLocal, LocalTensor<bool> &finishedLocal)
    {
        TopKInfo topKInfoData;
        topKInfoData.outter = 1;
        topKInfoData.inner = (ubIdx < ubLoop - 1) ? kAlign + ubFormerAlign : kAlign + ubTailAlign;
        topKInfoData.n = (ubIdx < ubLoop - 1) ? kAlign + ubFormer : kAlign + ubTail;
        TopkTiling *topKTilingData = (ubIdx < ubLoop - 1) ? &topkFormerTilingData : &topkTailTilingData;

        LocalTensor<U> gatingLocal = gatingQueue.template DeQue<U>();

        if constexpr (IsSameType<T, bfloat16_t>::value) {
            Cast(gatingLocal[kAlign], gatingLocal.template ReinterpretCast<T>()[kAlign * 2 + ubFormerAlign],
                 RoundMode::CAST_NONE, curCol);
        }

        int32_t curColAlign = (ubIdx < ubLoop - 1) ? ubFormerAlign : ubTailAlign;
        U scalar = VectorDup<U>(gatingLocal, curCol, curColAlign);
        if (ubIdx == 0) {
            Duplicate(gatingLocal, scalar, kAlign);
        }
        PipeBarrier<PIPE_V>();

        int32_t firstValue = ubIdx * ubFormer;
        ArithProgression(indicesOutLocal[kAlign], firstValue, 1, curColAlign);
        PipeBarrier<PIPE_V>();

        TopK<U, true, false, false, TopKMode::TOPK_NORMAL>(gatingLocal, indicesOutLocal, gatingLocal, indicesOutLocal,
                                                           finishedLocal, kAlign, *topKTilingData, topKInfoData, true);
        PipeBarrier<PIPE_V>();
    }

    template <typename U>
    __aicore__ inline void ComputeSoftmax(LocalTensor<U> &gatingLocal, LocalTensor<int32_t> &indicesOutLocal)
    {
        LocalTensor<U> outLocal = outQueue.template AllocTensor<U>();
        SoftMaxShapeInfo softmaxShapeInfoData;
        softmaxShapeInfoData.srcK = kAlign;
        softmaxShapeInfoData.srcM = 1;
        softmaxShapeInfoData.oriSrcK = k;
        softmaxShapeInfoData.oriSrcM = 1;

        SoftMax<U, true, false>(outLocal, gatingLocal, ubFormerSoftmaxTilingData, softmaxShapeInfoData);

        if constexpr (IsSameType<T, bfloat16_t>::value) {
            PipeBarrier<PIPE_V>();
            Cast(outLocal.template ReinterpretCast<bfloat16_t>(), outLocal, RoundMode::CAST_ROUND, k);
        }

        gatingQueue.FreeTensor(gatingLocal);
        outQueue.EnQue(outLocal);
        indicesOutQueue.EnQue(indicesOutLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<T> newOutLocal = outQueue.template DeQue<T>();
        LocalTensor<int32_t> newIndicesOutLocal = indicesOutQueue.template DeQue<int32_t>();

        int64_t gmIndex = k * progress;
        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = k * sizeof(T);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;

        DataCopyPad(outTensorGM[gmIndex], newOutLocal, intriParams);
        intriParams.blockLen = k * sizeof(int32_t);
        DataCopyPad(indicesOutTensorGM[gmIndex], newIndicesOutLocal, intriParams);
        outQueue.FreeTensor(newOutLocal);
        indicesOutQueue.FreeTensor(newIndicesOutLocal);
    }

    __aicore__ inline void ParesTiling(const MoeGatingTopKSoftmaxV2KFullLoadTilingData *__restrict tilingData)
    {
        row = tilingData->row;
        col = tilingData->col;
        k = tilingData->k;
        kAlign = tilingData->kAlign;
        blockNum = tilingData->blockNum;
        blockFormer = tilingData->blockFormer;
        blockTail = tilingData->blockTail;
        ubLoop = tilingData->ubLoop;
        ubFormer = tilingData->ubFormer;
        ubFormerAlign = tilingData->ubFormerAlign;
        ubTail = tilingData->ubTail;
        ubTailAlign = tilingData->ubTailAlign;
        ubFormerSoftmaxTilingData = tilingData->ubFormerSoftmaxTilingData;
        topkFormerTilingData = tilingData->topkFormerTilingData;
        topkTailTilingData = tilingData->topkTailTilingData;
        softmaxFlag = tilingData->softmaxFlag;
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, DB_RENORM> gatingQueue, finishedQueue;
    TQue<QuePosition::VECOUT, DB_RENORM> outQueue, indicesOutQueue;

    GlobalTensor<T> gatingTensorGM;
    GlobalTensor<bool> finishedTensorGM;
    GlobalTensor<T> outTensorGM;
    GlobalTensor<int32_t> indicesOutTensorGM;

    uint32_t row;
    uint32_t col;
    uint32_t k;
    uint32_t kAlign;
    uint32_t blockNum;
    uint32_t blockFormer;
    uint32_t blockTail;
    uint32_t ubLoop;
    uint32_t ubFormer;
    uint32_t ubFormerAlign;
    uint32_t ubTail;
    uint32_t ubTailAlign;
    uint32_t softmaxFlag;
    SoftMaxTiling ubFormerSoftmaxTilingData;
    TopkTiling topkFormerTilingData;
    TopkTiling topkTailTilingData;

    bool exitFinished{false};
};
} // namespace MoeGatingTopKSoftmaxV2
#endif
