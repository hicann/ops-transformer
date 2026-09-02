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
 * \file moe_gating_top_k_softmax_v2_k_fullload_arch35.h
 * \brief
 */
#ifndef MOE_GATING_TOP_K_SOFTMAX_V2_K_FULLLOAD_V35
#define MOE_GATING_TOP_K_SOFTMAX_V2_K_FULLLOAD_V35

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace MoeGatingTopKSoftmaxV2 {
using namespace AscendC;
constexpr int32_t DB_KFULL_LOAD = 2;

template <typename T, int32_t renorm>
class MoeGatingTopKSoftmaxV2KFullLoad {
public:
    __aicore__ inline MoeGatingTopKSoftmaxV2KFullLoad(){};
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

        pipe.InitBuffer(gatingQueue, DB_KFULL_LOAD, ubFormerAlign * sizeof(float));
        pipe.InitBuffer(outQueue, DB_KFULL_LOAD, (ubFormerAlign + kAlign) * sizeof(float));

        pipe.InitBuffer(softmaxMaxTensorUb, ALIGNMENT_SIZE);
        pipe.InitBuffer(softmaxSumTensorUb, ALIGNMENT_SIZE);
        pipe.InitBuffer(expMaxTensorUb, ALIGNMENT_SIZE);
        pipe.InitBuffer(finishedQueue, DB_KFULL_LOAD, ALIGNMENT_SIZE);
        pipe.InitBuffer(indicesOutQueue, DB_KFULL_LOAD, (ubFormerAlign + kAlign) * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        int32_t curRowNum = ubFormer;
        int64_t rowCount = (GetBlockIdx() < blockNum - 1) ? blockFormer : blockTail;
        int32_t curCol;

        for (int64_t rowIdx = 0; rowIdx < rowCount; rowIdx++) {
            LocalTensor<int32_t> indicesOutLocal = indicesOutQueue.template AllocTensor<int32_t>();
            LocalTensor<bool> finishedLocal;
            float softmaxSum = 0.0f;
            if (exitFinished) {
                CopyInFinished(rowIdx);
                finishedLocal = finishedQueue.template DeQue<bool>();
            }

            LocalTensor<float> softmaxSumTensor = softmaxSumTensorUb.template Get<float>();
            LocalTensor<float> softmaxMaxTensor = softmaxMaxTensorUb.template Get<float>();
            LocalTensor<float> outLocal = outQueue.template AllocTensor<float>();

            for (int64_t ubIdx = 0; ubIdx < ubLoop; ubIdx++) {
                curCol = (ubIdx < ubLoop - 1) ? ubFormer : ubTail;
                CopyInGating(rowIdx, ubIdx, curCol);
                Compute(rowIdx, ubIdx, curCol, 1, outLocal, indicesOutLocal, softmaxSumTensor, softmaxMaxTensor,
                        finishedLocal);
            }
            softmaxSum = ComputeOut(outLocal, indicesOutLocal, softmaxSumTensor, finishedLocal, rowIdx);

            CopyOut(rowIdx);
            if (softmaxFlag == 1) {
                float softmaxMax = static_cast<float>(softmaxMaxTensor.GetValue(0));
                UpdateSoftmax(rowIdx, softmaxSum, softmaxMax);
            }
        }
    }

private:
    __aicore__ inline int64_t Align(const int64_t elementNum)
    {
        return (elementNum + ALIGN_FACTOR - 1) / ALIGN_FACTOR * ALIGN_FACTOR;
    }

    template <typename U>
    __aicore__ inline void VectorDup(LocalTensor<U> &dst, const int32_t colCount, const int32_t colCountAlign)
    {
        if (colCountAlign - colCount != 0) {
            U scalar;
            if constexpr (IsSameType<U, half>::value) {
                uint16_t localScalar = 0xFC00; // -inf
                scalar = *((half *)&localScalar);
            } else if constexpr (IsSameType<U, bfloat16_t>::value) {
                uint16_t localScalar = 0xFF80; // -inf
                scalar = *((bfloat16_t *)&localScalar);
            } else {
                uint32_t localScalar = 0xFF800000; // -inf
                scalar = *((float *)&localScalar);
            }
            // 当对齐后大小与实际大小不一致，需要将 colCount到colAlign之间的数据掩成-1
            uint64_t mask[2] = {
                (((uint64_t)1 << (colCountAlign - colCount)) - 1) << (ALIGN_FACTOR - (colCountAlign - colCount)), 0};
            Duplicate(dst[(kAlign + colCountAlign) - ALIGN_FACTOR], scalar, mask, 1, 1, 1);
        }
    }

    __aicore__ inline void CopyInFinished(int32_t progress)
    {
        LocalTensor<bool> finishedLocal = finishedQueue.template AllocTensor<bool>();

        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyParams intriParams{1, static_cast<uint16_t>(sizeof(bool)), 0, 0};
        DataCopyPad(finishedLocal, finishedTensorGM[progress], intriParams, padParams);

        finishedQueue.EnQue(finishedLocal);
    }

    __aicore__ inline void CopyInGating(int32_t progress, int32_t ubIdx, int32_t curCol)
    {
        LocalTensor<T> gatingLocal = gatingQueue.template AllocTensor<T>();

        DataCopyPadExtParams<T> padParams{false, 0, 0, (T)0};
        DataCopyExtParams intriParams{1, static_cast<uint32_t>(curCol * sizeof(T)), 0, 0, 0};

        uint32_t ubOffset = 0;
        if constexpr (!IsSameType<T, float>::value) {
            ubOffset = ubFormerAlign;
        }
        DataCopyPad(gatingLocal[ubOffset], gatingTensorGM[col * progress + ubFormer * ubIdx], intriParams, padParams);
        gatingQueue.EnQue(gatingLocal);
    }

    __aicore__ inline void Compute(int32_t progress, int32_t ubIdx, int32_t curCol, int32_t curRowsNum,
                                   LocalTensor<float> &outLocal, LocalTensor<int32_t> &indicesOutLocal,
                                   LocalTensor<float> &softmaxSumTensor, LocalTensor<float> &softmaxMaxTensor,
                                   LocalTensor<bool> &finishedLocal)
    {
        SoftMaxShapeInfo softmaxShapeInfoData;
        softmaxShapeInfoData.srcK = Align(curCol);
        softmaxShapeInfoData.srcM = curRowsNum;
        softmaxShapeInfoData.oriSrcK = curCol;
        softmaxShapeInfoData.oriSrcM = curRowsNum;
        TopKInfo topKInfoData;
        topKInfoData.outter = curRowsNum;
        topKInfoData.inner = (ubIdx < ubLoop - 1) ? kAlign + ubFormerAlign : kAlign + ubTailAlign;
        topKInfoData.n = (ubIdx < ubLoop - 1) ? kAlign + ubFormer : kAlign + ubTail;

        LocalTensor<float> gatingLocal = gatingQueue.template DeQue<float>();

        SoftMaxTiling *softmaxTilingData = (ubIdx < ubLoop - 1) ? &ubFormerSoftmaxTilingData : &ubTailSoftmaxTilingData;
        TopkTiling *topKTilingData = (ubIdx < ubLoop - 1) ? &topkFormerTilingData : &topkTailTilingData;

        // curcol haven't dirty data
        if constexpr (!IsSameType<T, float>::value) {
            Cast(gatingLocal, gatingLocal.ReinterpretCast<T>()[ubFormerAlign], RoundMode::CAST_NONE, curCol);
        }

        LocalTensor<float> expMaxTensor = expMaxTensorUb.template Get<float>();
        LocalTensor<float> inExpSumTensor;
        LocalTensor<float> inMaxTensor;

        if (ubIdx == 0) {
            Duplicate(outLocal, static_cast<float>(-1), kAlign);
            SoftmaxFlashV2<float, false, false, false, false>(outLocal[kAlign], softmaxSumTensor, softmaxMaxTensor,
                                                              gatingLocal, expMaxTensor, inExpSumTensor, inMaxTensor,
                                                              *softmaxTilingData, softmaxShapeInfoData);
        } else {
            SoftmaxFlashV2<float, true, false, false, false>(
                outLocal[kAlign], softmaxSumTensor, softmaxMaxTensor, gatingLocal, expMaxTensor, softmaxSumTensor,
                softmaxMaxTensor, *softmaxTilingData, softmaxShapeInfoData);
            auto eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventID);
            WaitFlag<HardEvent::V_S>(eventID);
            Muls<float>(outLocal, outLocal, expMaxTensor.GetValue(0), kAlign);
        }

        gatingQueue.FreeTensor(gatingLocal);

        int32_t curColAlign = (ubIdx < ubLoop - 1) ? ubFormerAlign : ubTailAlign;
        VectorDup<float>(outLocal, curCol, curColAlign);
        int32_t firstValue = ubIdx * ubFormer;
        ArithProgression(indicesOutLocal[kAlign], firstValue, 1, curColAlign);

        TopK<float, true, false, false, TopKMode::TOPK_NORMAL>(outLocal, indicesOutLocal, outLocal, indicesOutLocal,
                                                               finishedLocal, kAlign, *topKTilingData, topKInfoData,
                                                               true);
    }

    __aicore__ inline float ComputeOut(LocalTensor<float> &outLocal, LocalTensor<int32_t> &indicesOutLocal,
                                       LocalTensor<float> &softmaxSumTensor, LocalTensor<bool> &finishedLocal,
                                       int32_t rowIdx)
    {
        auto eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventID);
        WaitFlag<HardEvent::V_S>(eventID);
        // if softmax reslut is nan，indicesOutLocal is 0 - (k-1)
        if (((softmaxSumTensor.template ReinterpretCast<int32_t>().GetValue(0) & 0x7f800000) == 0x7f800000) &&
            (softmaxSumTensor.template ReinterpretCast<int32_t>().GetValue(0) & 0x7fffff) > 0) {
            ArithProgression(indicesOutLocal, 0, 1, k);
        }

        if (exitFinished) {
            if (finishedLocal.GetValue(0)) {
                Duplicate(indicesOutLocal, static_cast<int32_t>(col), k);
            }
            finishedQueue.FreeTensor(finishedLocal);
        }

        float softmaxSum = 1.0f / static_cast<float>(softmaxSumTensor.GetValue(0));
        Muls<float>(outLocal, outLocal, softmaxSum, k);

        if constexpr (IsSameType<T, bfloat16_t>::value) {
            Cast(outLocal.template ReinterpretCast<bfloat16_t>(), outLocal, RoundMode::CAST_ROUND, k);
        } else if constexpr (IsSameType<T, half>::value) {
            Cast(outLocal.template ReinterpretCast<half>(), outLocal, RoundMode::CAST_RINT, k);
        }

        outQueue.EnQue(outLocal);
        indicesOutQueue.EnQue(indicesOutLocal);
        return softmaxSum;
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<T> newOutLocal = outQueue.template DeQue<T>();
        LocalTensor<int32_t> newIndicesOutLocal = indicesOutQueue.template DeQue<int32_t>();

        int64_t gmIndex = k * progress;
        DataCopyParams intriParams{1, static_cast<uint16_t>(k * sizeof(T)), 0, 0};

        DataCopyPad(outTensorGM[gmIndex], newOutLocal, intriParams);
        intriParams.blockLen = k * sizeof(int32_t);
        DataCopyPad(indicesOutTensorGM[gmIndex], newIndicesOutLocal, intriParams);
        indicesOutQueue.FreeTensor(newIndicesOutLocal);
        outQueue.FreeTensor(newOutLocal);
    }

    __aicore__ inline void UpdateSoftmax(int32_t progress, float softmaxSum, float softmaxMax)
    {
        for (int64_t ubIdx = 0; ubIdx < ubLoop; ubIdx++) {
            int32_t curCol = (ubIdx < ubLoop - 1) ? ubFormer : ubTail;
            CopyInGating(progress, ubIdx, curCol);
            LocalTensor<float> gatingLocal = gatingQueue.template DeQue<float>();
            if constexpr (!IsSameType<T, float>::value) {
                Cast(gatingLocal, gatingLocal.ReinterpretCast<T>()[ubFormerAlign], RoundMode::CAST_NONE, curCol);
            }
            Adds(gatingLocal, gatingLocal, -softmaxMax, curCol);

            LocalTensor<float> outLocal = outQueue.template AllocTensor<float>();
            Exp(outLocal, gatingLocal, curCol);
            gatingQueue.FreeTensor(gatingLocal);

            Muls(outLocal, outLocal, softmaxSum, curCol);
            outQueue.EnQue(outLocal);
            outQueue.template DeQue<float>();

            DataCopyExtParams intriParams{1, static_cast<uint32_t>(curCol * sizeof(float)), 0, 0, 0};
            DataCopyPad(softmaxOutTensorGM[col * progress + ubFormer * ubIdx], outLocal, intriParams);
            outQueue.FreeTensor(outLocal);
        }
    }

    __aicore__ inline void ParesTiling(const MoeGatingTopKSoftmaxV2KFullLoadTilingData *__restrict tilingData)
    {
        col = tilingData->col;
        row = tilingData->row;
        k = tilingData->k;
        kAlign = tilingData->kAlign;
        blockNum = tilingData->blockNum;
        blockTail = tilingData->blockTail;
        blockFormer = tilingData->blockFormer;
        ubLoop = tilingData->ubLoop;
        ubFormer = tilingData->ubFormer;
        ubFormerAlign = tilingData->ubFormerAlign;
        ubTail = tilingData->ubTail;
        ubTailAlign = tilingData->ubTailAlign;
        softmaxFlag = tilingData->softmaxFlag;
        ubTailSoftmaxTilingData = tilingData->ubTailSoftmaxTilingData;
        topkFormerTilingData = tilingData->topkFormerTilingData;
        topkTailTilingData = tilingData->topkTailTilingData;
        ubFormerSoftmaxTilingData = tilingData->ubFormerSoftmaxTilingData;
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, DB_KFULL_LOAD> gatingQueue, finishedQueue;
    TQue<QuePosition::VECOUT, DB_KFULL_LOAD> outQueue, indicesOutQueue, softmaxOutQueue;

    TBuf<> softmaxMaxTensorUb;
    TBuf<> softmaxSumTensorUb;
    TBuf<> expMaxTensorUb;

    GlobalTensor<T> gatingTensorGM;
    GlobalTensor<bool> finishedTensorGM;
    GlobalTensor<T> outTensorGM;
    GlobalTensor<int32_t> indicesOutTensorGM;
    GlobalTensor<float> softmaxOutTensorGM;

    uint32_t k;
    uint32_t row;
    uint32_t col;
    uint32_t kAlign;
    uint32_t blockNum;
    uint32_t blockFormer;
    uint32_t blockTail;
    uint32_t ubTail;
    uint32_t ubLoop;
    uint32_t ubFormer;
    uint32_t ubFormerAlign;
    uint32_t ubTailAlign;
    uint32_t softmaxFlag;
    SoftMaxTiling ubFormerSoftmaxTilingData;
    SoftMaxTiling ubTailSoftmaxTilingData;
    TopkTiling topkTailTilingData;
    TopkTiling topkFormerTilingData;

    bool exitFinished{false};
};
} // namespace MoeGatingTopKSoftmaxV2
#endif
