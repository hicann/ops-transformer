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
 * \file decode_update.h
 * \brief
 */

#ifndef ASCENDC_ATTENTION_UPDATE_DECODE_UPDATE_H_
#define ASCENDC_ATTENTION_UPDATE_DECODE_UPDATE_H_

#include <limits>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;

namespace AttentionUpdate {
static constexpr uint32_t BUFFER_NUM = 2;
static constexpr uint32_t NUM0 = 0;
static constexpr uint32_t NUM1 = 1;
static constexpr uint32_t NUM2 = 2;
static constexpr uint32_t NUM7 = 7;
static constexpr uint32_t NUM8 = 8;
static constexpr uint32_t NUM16 = 16;
static constexpr uint32_t NUM32 = 32;
static constexpr uint32_t NUM64 = 64;
static constexpr uint32_t NUM255 = 255;
static constexpr uint32_t NUM256 = 256;
static constexpr uint32_t MAX_EFFECTIVE_SP = 16;
static constexpr float POS_INF = std::numeric_limits<float>::infinity();
static constexpr float NEG_INF = -std::numeric_limits<float>::infinity();
static constexpr float DUMMY_LSE = -1e38f;
static constexpr uint32_t MAX_UB_SIZE = 188 * 1024; //  double buffer, 每块94KB共188KB
static const uint16_t ALIGNED_TO_8 = 8;
static const int32_t ALIGNED_TO_2 = 2;
static const uint32_t SPLIT_TO_2 = 2;
static const uint32_t ELEM_PER_256B = NUM256 / sizeof(float);

template <typename lseType, typename outType>
class DecodeUpdate {
public:
    __aicore__ inline DecodeUpdate() {}
    __aicore__ inline void Init(GM_ADDR lse, GM_ADDR in, GM_ADDR out, GM_ADDR lesout,
                                const DecodeUpdateTilingData *tdata)
    {
        this->lsePtr = GetTensorPtr(lse);
        this->inPtr = GetTensorPtr(in);
        ParseTilingData(tdata);
        CalcTileLength();
        // totalLength 表示 Tiling 约束下的元素个数，保证可落在 uint32_t 内；
        // 参与 GM 地址计算（如 totalLength * hDim）时再显式宽化到 uint64_t。
        // 设置全局变量的起始地址与总长度: BLOCK_LENGTH; sp, B*s*hc in1 sp; B*s*hc, hd in2
        outGm.SetGlobalBuffer((__gm__ outType *)out, static_cast<uint64_t>(totalLength) * hDim);
        lseoutGm.SetGlobalBuffer((__gm__ lseType *)lesout, totalLength);
        InitBuffers();
    }
    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->loopCount; i++) {
            if (lastLength && i == this->loopCount - 1) {
                curLength = lastLength;
            }
            if (spLoopCount == 1) {
                CopyInSp(i, 0, spFirstLoop, spFirstLoop);
                Compute(spFirstLoop);
                CopyOut(i);
            } else {
                LocalTensor<float> outAccLocal = lseexpBroadcastBuffer.Get<float>();
                for (uint32_t spLoop = 0; spLoop < spLoopCount; spLoop++) {
                    uint32_t currentSp;
                    uint32_t effectiveSp;
                    if (spLoop == 0) {
                        currentSp = spFirstLoop;
                        effectiveSp = 1 + currentSp;
                    } else {
                        uint32_t remainingSp = sp - spFirstLoop;
                        uint32_t processedNewSp = (spLoop - 1) * spNextLoop;
                        uint32_t remainingNewSp = remainingSp > processedNewSp ? remainingSp - processedNewSp : 0;
                        currentSp = remainingNewSp < spNextLoop ? remainingNewSp : spNextLoop;
                        effectiveSp = 1 + currentSp;
                    }
                    CopyInSp(i, spLoop, currentSp, effectiveSp);
                    Compute(effectiveSp);
                    SaveAcc(outAccLocal);
                }
                WriteAcc(outAccLocal);
                CopyOut(i);
            }
        }
    }

private:
    __aicore__ inline __gm__ uint64_t *GetTensorPtr(GM_ADDR gmAddr)
    {
        __gm__ uint64_t *dataAddr = reinterpret_cast<__gm__ uint64_t *>(gmAddr);
        uint64_t tensorPtrOffset = *dataAddr; // The offset of the data address from the first address.
        // Moving 3 bits to the right means dividing by sizeof(uint64 t).
        __gm__ uint64_t *tensorPtr = dataAddr + (tensorPtrOffset >> 3);
        return tensorPtr;
    }

    __aicore__ inline void ParseTilingData(const DecodeUpdateTilingData *tdata)
    {
        this->hDim = tdata->hDim;
        this->sp = tdata->sp;
        this->totalLength = tdata->totalLength;
        this->updateType = tdata->updateType;
        if (GetBlockIdx() < tdata->formerNum) {
            blockLength = tdata->formerLength;
            this->gmStartOffset = static_cast<uint64_t>(GetBlockIdx()) * tdata->formerLength;
        } else {
            blockLength = tdata->tailLength;
            this->gmStartOffset =
                static_cast<uint64_t>(tdata->formerNum) * tdata->formerLength +
                static_cast<uint64_t>(GetBlockIdx() - tdata->formerNum) * tdata->tailLength; // tail block
        }

        if (sp <= MAX_EFFECTIVE_SP) {
            this->spLoopCount = 1;
            this->spFirstLoop = sp;
            this->spNextLoop = 0;
            this->spAlloc = sp;
        } else {
            this->spFirstLoop = MAX_EFFECTIVE_SP - 1;
            this->spNextLoop = MAX_EFFECTIVE_SP - 1;
            this->spLoopCount = 1 + (sp - spFirstLoop + spNextLoop - 1) / spNextLoop;
            this->spAlloc = MAX_EFFECTIVE_SP;
        }
    }

    __aicore__ inline void CalcTileLength()
    {
        uint32_t spAligned = (spAlloc + NUM7) / NUM8 * NUM8;
        // 用94K的UB大小推算出 tileLength 最大能设置到多少
        uint32_t maxTileLength =
            (MAX_UB_SIZE - NUM8 * sizeof(uint32_t) - (ELEM_PER_256B - 1) * (sizeof(float) + sizeof(uint8_t))) /
            (sizeof(float) * spAligned * BUFFER_NUM * (NUM2 * (NUM1 + hDim) + hDim / spAligned) +
             hDim * spAligned * sizeof(float) * NUM2 + sizeof(float) * BUFFER_NUM + sizeof(float) * NUM2 +
             sizeof(uint8_t) * spAligned);
        if constexpr (!std::is_same<outType, float>::value) {
            maxTileLength =
                (MAX_UB_SIZE - NUM8 * sizeof(uint32_t) - (ELEM_PER_256B - 1) * (sizeof(float) + sizeof(uint8_t))) /
                (sizeof(float) * spAligned * BUFFER_NUM * (NUM2 * (NUM1 + hDim) + hDim / spAligned) +
                 hDim * spAligned * sizeof(float) * NUM2 + sizeof(float) * BUFFER_NUM + sizeof(float) * NUM2 +
                 (spAlloc + NUM1) * NUM16 * BUFFER_NUM + sizeof(uint8_t) * spAligned);
        }
        if (sp >= NUM8) {
            maxTileLength = maxTileLength / SPLIT_TO_2;
        }
        maxTileLength = maxTileLength < NUM1 ? NUM1 : maxTileLength;
        this->tileLength = maxTileLength < blockLength ? maxTileLength : blockLength;
        this->curLength = this->tileLength;
        this->lastLength = blockLength % tileLength;
        this->loopCount = blockLength / tileLength + (lastLength == NUM0 ? NUM0 : NUM1);
        this->tileLengthAlig = ((tileLength + NUM7) / NUM8) * NUM8;
        this->lastLengthAlig = ((lastLength + NUM7) / NUM8) * NUM8;
    }

    __aicore__ inline void InitBuffers()
    {
        uint32_t inQueueLseLengthAlign = (tileLengthAlig * spAlloc * sizeof(float) + NUM255) / NUM256 * NUM256;
        pipe.InitBuffer(inQueueLse, BUFFER_NUM, inQueueLseLengthAlign);
        if constexpr (std::is_same<outType, float>::value) {
            pipe.InitBuffer(inQueueIn, BUFFER_NUM, tileLength * hDim * spAlloc * sizeof(float));
        } else {
            pipe.InitBuffer(inQueueIn, BUFFER_NUM,
                            tileLength * hDim * spAlloc * sizeof(float) + (spAlloc + NUM1) * NUM16);
        }

        pipe.InitBuffer(outQueueOut, BUFFER_NUM, tileLength * hDim * sizeof(float));
        pipe.InitBuffer(outQueueLse, BUFFER_NUM, tileLengthAlig * sizeof(float));

        pipe.InitBuffer(lsemaxBuffer, tileLengthAlig * sizeof(float));
        pipe.InitBuffer(lseexpsumBuffer, tileLengthAlig * sizeof(float));
        pipe.InitBuffer(lseexpBuffer, tileLengthAlig * spAlloc * sizeof(float));
        if (hDim > NUM256 && spAlloc >= NUM8) {
            pipe.InitBuffer(lseexpBroadcastBuffer, tileLengthAlig * spAlloc * NUM64 * sizeof(float));
        } else {
            pipe.InitBuffer(lseexpBroadcastBuffer, tileLengthAlig * spAlloc * hDim * sizeof(float));
        }
        pipe.InitBuffer(selMaskBuffer, inQueueLseLengthAlign * sizeof(uint8_t));
    }

    __aicore__ inline void CopyInSpAccData(LocalTensor<float> &lseLocal, LocalTensor<float> &inLocal, int32_t spLoopIdx,
                                           uint32_t curLengthPad)
    {
        LocalTensor<float> lseAccLocal = lseexpsumBuffer.Get<float>();
        LocalTensor<float> outAccLocal = lseexpBroadcastBuffer.Get<float>();
        DataCopy(lseLocal[0], lseAccLocal, curLengthPad);
        PipeBarrier<PIPE_V>();
        DataCopy(inLocal[0], outAccLocal, curLength * hDim);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyInLsePerSp(LocalTensor<float> &lseLocal, uint32_t posIdx, uint32_t gmSpIdx,
                                          int32_t progress, uint32_t curLengthPad)
    {
        lseGm.SetGlobalBuffer(reinterpret_cast<__gm__ lseType *>(*(lsePtr + gmSpIdx)), totalLength);
        if (curLength % ALIGNED_TO_8 == 0) {
            DataCopy(lseLocal[posIdx * curLengthPad],
                     lseGm[static_cast<uint64_t>(progress) * tileLength + gmStartOffset], curLength);
        } else {
            DataCopyPad(lseLocal[posIdx * curLengthPad],
                        lseGm[static_cast<uint64_t>(progress) * tileLength + gmStartOffset],
                        {static_cast<uint16_t>(1), static_cast<uint32_t>(curLength * sizeof(lseType)), 0, 0, 0},
                        {true, 0, static_cast<uint8_t>(NUM8 - curLength % NUM8), 0});
        }
    }

    __aicore__ inline void CopyInOPerSp(LocalTensor<float> &inLocal, LocalTensor<outType> &inLocalFp16, uint32_t posIdx,
                                        uint32_t gmSpIdx, int32_t progress)
    {
        inGm.SetGlobalBuffer(reinterpret_cast<__gm__ outType *>(*(inPtr + gmSpIdx)),
                             static_cast<uint64_t>(totalLength) * hDim);
        uint32_t inOffset = posIdx * curLength * hDim;
        uint64_t gmOffset = static_cast<uint64_t>(progress) * tileLength * hDim + gmStartOffset * hDim;
        if constexpr (std::is_same<outType, float>::value) {
            DataCopy(inLocal[inOffset], inGm[gmOffset], curLength * hDim);
        } else {
            if (curLength % ALIGNED_TO_8 == 0) {
                DataCopy(inLocalFp16[inOffset], inGm[gmOffset], curLength * hDim);
            } else {
                uint64_t fp16Offset = static_cast<uint64_t>(posIdx) * curLength * hDim * NUM2;
                if (fp16Offset % NUM32 == NUM16) {
                    fp16Offset += NUM16;
                }
                DataCopyPad(
                    inLocalFp16[fp16Offset / NUM2], inGm[gmOffset],
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(curLength * hDim * sizeof(outType)), 0, 0, 0},
                    {true, 0, static_cast<uint8_t>((NUM32 - curLength * hDim * sizeof(outType) % NUM32) / NUM2), 0});
                PipeBarrier<PIPE_ALL>();
                Cast(inLocal[inOffset], inLocalFp16[fp16Offset / NUM2], RoundMode::CAST_NONE, curLength * hDim);
            }
        }
    }

    __aicore__ inline void CopyInSp(int32_t progress, int32_t spLoopIdx, uint32_t currentSp, uint32_t effectiveSp)
    {
        //  按照逻辑队列初始化的buffer数量和每块大小，分配对应大小的内存
        LocalTensor<float> lseLocal = inQueueLse.AllocTensor<float>();
        LocalTensor<float> inLocal = inQueueIn.AllocTensor<float>();

        uint32_t curLengthPad = ((curLength + NUM7) / NUM8) * NUM8;
        uint32_t posOffset = (spLoopIdx == 0 && sp > MAX_EFFECTIVE_SP) ? 1 : (spLoopIdx > 0 ? 1 : 0);
        uint32_t gmSpBase = (spLoopIdx == 0) ? 0 : (spFirstLoop + (spLoopIdx - 1) * spNextLoop);

        if (spLoopIdx == 0 && sp > MAX_EFFECTIVE_SP) {
            Duplicate<float>(lseLocal, DUMMY_LSE, static_cast<int32_t>(curLengthPad));
            PipeBarrier<PIPE_V>();
            Duplicate<float>(inLocal, 0.0f, static_cast<int32_t>(curLength * hDim));
            PipeBarrier<PIPE_V>();
        } else if (spLoopIdx > 0) {
            CopyInSpAccData(lseLocal, inLocal, spLoopIdx, curLengthPad);
        }

        uint64_t inLocalFp16Offset = static_cast<uint64_t>(tileLength) * hDim * spAlloc;
        if ((inLocalFp16Offset * NUM2) % NUM32 == NUM16) {
            inLocalFp16Offset += NUM8;
        }
        LocalTensor<outType> inLocalFp16 = inLocal.template ReinterpretCast<outType>()[inLocalFp16Offset];

        for (uint32_t i = 0; i < currentSp; i++) {
            uint32_t posIdx = posOffset + i;
            uint32_t gmSpIdx = gmSpBase + i;
            CopyInLsePerSp(lseLocal, posIdx, gmSpIdx, progress, curLengthPad);
            CopyInOPerSp(inLocal, inLocalFp16, posIdx, gmSpIdx, progress);
        }

        if constexpr (!std::is_same<outType, float>::value) {
            if (curLength % ALIGNED_TO_8 == 0) {
                inQueueIn.EnQue(inLocal);
                inLocal = inQueueIn.DeQue<float>();
                inLocalFp16 = inLocal.template ReinterpretCast<outType>()[inLocalFp16Offset];
                Cast(inLocal[posOffset * curLength * hDim], inLocalFp16[posOffset * curLength * hDim],
                     RoundMode::CAST_NONE, curLength * hDim * currentSp);
            }
        }

        inQueueLse.EnQue(lseLocal);
        inQueueIn.EnQue(inLocal);
    }

    __aicore__ inline void SaveAcc(LocalTensor<float> &outAccLocal)
    {
        LocalTensor<float> lseoutLocal = outQueueLse.DeQue<float>();
        LocalTensor<float> outLocal = outQueueOut.DeQue<float>();

        LocalTensor<float> lseAccLocal = lseexpsumBuffer.Get<float>();
        DataCopy(lseAccLocal, lseoutLocal, ((curLength + NUM7) / NUM8) * NUM8);
        PipeBarrier<PIPE_V>();
        DataCopy(outAccLocal, outLocal, curLength * hDim);
        PipeBarrier<PIPE_V>();

        outQueueLse.FreeTensor(lseoutLocal);
        outQueueOut.FreeTensor(outLocal);
    }

    __aicore__ inline void WriteAcc(LocalTensor<float> &outAccLocal)
    {
        LocalTensor<float> lseAccLocal = lseexpsumBuffer.Get<float>();
        LocalTensor<float> outLocal = outQueueOut.AllocTensor<float>();
        LocalTensor<float> lseoutLocal = outQueueLse.AllocTensor<float>();

        uint32_t curLengthPad = ((curLength + NUM7) / NUM8) * NUM8;
        DataCopy(outLocal, outAccLocal, curLength * hDim);
        PipeBarrier<PIPE_V>();
        DataCopy(lseoutLocal, lseAccLocal, curLengthPad);
        PipeBarrier<PIPE_V>();

        outQueueOut.EnQue(outLocal);
        outQueueLse.EnQue(lseoutLocal);
    }

    __aicore__ inline void ProcessLseInfReplacement(LocalTensor<float> &lseLocal, uint32_t effectiveSp)
    {
        LocalTensor<uint8_t> selMask = selMaskBuffer.Get<uint8_t>();
        uint32_t curLengthPad = ((curLength + NUM7) / NUM8) * NUM8;
        uint32_t alignedCount = ((curLengthPad * effectiveSp + ELEM_PER_256B - 1) / ELEM_PER_256B) * ELEM_PER_256B;

        CompareScalar(selMask, lseLocal, POS_INF, CMPMODE::EQ, alignedCount);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> negInfTensor = lseexpBuffer.Get<float>();
        Duplicate<float>(negInfTensor, NEG_INF, static_cast<int32_t>(curLengthPad * effectiveSp));
        PipeBarrier<PIPE_V>();

        Select<float, uint8_t>(lseLocal, selMask, negInfTensor, lseLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
                               static_cast<uint32_t>(curLengthPad * effectiveSp));
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Compute(uint32_t effectiveSp)
    {
        LocalTensor<float> lseLocal = inQueueLse.DeQue<float>();
        LocalTensor<float> inLocal = inQueueIn.DeQue<float>();
        LocalTensor<float> lseexpLocal = lseexpBuffer.Get<float>();
        LocalTensor<float> outLocal = outQueueOut.AllocTensor<float>();
        LocalTensor<float> lsemaxLocal = lsemaxBuffer.Get<float>();
        LocalTensor<float> lseexpsumLocal = lseexpsumBuffer.Get<float>();
        LocalTensor<float> lseexpBroadcastLocal = lseexpBroadcastBuffer.Get<float>();
        LocalTensor<float> lseoutLocal = outQueueLse.AllocTensor<float>();

        ProcessLseInfReplacement(lseLocal, effectiveSp);

        // broad to 8
        uint32_t curLengthPad = ((curLength + NUM7) / NUM8) * NUM8;
        const uint32_t srcShape[2] = {effectiveSp * curLengthPad, 1};
        uint32_t dstShape[2] = {effectiveSp * curLengthPad, hDim};
        if (hDim > NUM256 && spAlloc >= NUM8) {
            dstShape[1] = NUM64;
        }

        DataCopy(lsemaxLocal, lseLocal, curLengthPad);
        PipeBarrier<PIPE_V>();

        for (uint32_t i = 1; i < effectiveSp; i++) {
            Max(lsemaxLocal, lsemaxLocal, lseLocal[i * curLengthPad], curLengthPad);
            PipeBarrier<PIPE_V>();
        }

        PipeBarrier<PIPE_V>();
        for (uint32_t i = 0; i < effectiveSp; i++) {
            Sub(lseexpLocal[i * curLengthPad], lseLocal[i * curLengthPad], lsemaxLocal, curLengthPad);
            PipeBarrier<PIPE_V>();
        }

        Exp(lseexpLocal, lseexpLocal, curLengthPad * effectiveSp);
        PipeBarrier<PIPE_V>();

        DataCopy(lseexpsumLocal, lseexpLocal, curLengthPad);
        PipeBarrier<PIPE_V>();
        for (uint32_t i = 1; i < effectiveSp; i++) {
            Add(lseexpsumLocal, lseexpsumLocal, lseexpLocal[i * curLengthPad], curLengthPad);
            PipeBarrier<PIPE_V>();
        }

        Log(lseexpsumLocal, lseexpsumLocal, curLengthPad);
        PipeBarrier<PIPE_V>();

        Add(lseoutLocal, lsemaxLocal, lseexpsumLocal, curLengthPad);
        PipeBarrier<PIPE_V>();

        for (uint32_t i = 0; i < effectiveSp; i++) {
            Sub(lseexpLocal[i * curLengthPad], lseLocal[i * curLengthPad], lseoutLocal, curLengthPad);
            PipeBarrier<PIPE_V>();
        }

        Exp(lseexpLocal, lseexpLocal, curLengthPad * effectiveSp);
        PipeBarrier<PIPE_V>();
        BroadCast<float, ALIGNED_TO_2, 1>(lseexpBroadcastLocal, lseexpLocal, dstShape, srcShape);
        PipeBarrier<PIPE_V>();
        if (hDim > NUM256 && spAlloc >= NUM8) {
            int64_t tmpTailLength = hDim % NUM64;
            int64_t tmpTailStart = hDim - tmpTailLength;
            for (uint32_t i = 0; i < effectiveSp; i++) {
                for (uint32_t k = 0; k < curLength; k++) {
                    Mul(inLocal[i * curLength * hDim + k * hDim], inLocal[i * curLength * hDim + k * hDim],
                        lseexpBroadcastLocal[i * curLengthPad * NUM64 + k * NUM64], NUM64, hDim / NUM64,
                        {1, 1, 1, 8, 8, 0});
                    if (tmpTailLength > 0) {
                        Mul(inLocal[i * curLength * hDim + k * hDim + tmpTailStart],
                            inLocal[i * curLength * hDim + k * hDim + tmpTailStart],
                            lseexpBroadcastLocal[i * curLengthPad * NUM64 + k * NUM64], tmpTailLength, 1,
                            {1, 1, 1, 8, 8, 0});
                    }
                }
            }
        } else {
            for (uint32_t i = 0; i < effectiveSp; i++) {
                Mul(inLocal[i * curLength * hDim], inLocal[i * curLength * hDim],
                    lseexpBroadcastLocal[i * curLengthPad * hDim], curLength * hDim);
            }
        }
        PipeBarrier<PIPE_V>();

        DataCopy(outLocal, inLocal, curLength * hDim);
        PipeBarrier<PIPE_V>();
        for (uint32_t i = 1; i < effectiveSp; i++) {
            Add(outLocal, outLocal, inLocal[i * curLength * hDim], curLength * hDim);
            PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_V>();

        outQueueLse.EnQue<float>(lseoutLocal);
        outQueueOut.EnQue<float>(outLocal);
        inQueueLse.FreeTensor(lseLocal);
        inQueueIn.FreeTensor(inLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<float> outLocal = outQueueOut.DeQue<float>();
        if constexpr (std::is_same<outType, float>::value) { // fp32直接搬运
            DataCopy(outGm[gmStartOffset * hDim + static_cast<uint64_t>(progress) * tileLength * hDim], outLocal,
                     curLength * hDim);
        } else if constexpr (std::is_same<outType, bfloat16_t>::value) { // 先转fp32，再搬运
            LocalTensor<outType> outLocal16 = outLocal.template ReinterpretCast<outType>();
            Cast(outLocal16, outLocal, RoundMode::CAST_RINT, curLength * hDim);
            PipeBarrier<PIPE_V>();
            DataCopyPad(outGm[gmStartOffset * hDim + static_cast<uint64_t>(progress) * tileLength * hDim], outLocal16,
                        {static_cast<uint16_t>(1), static_cast<uint32_t>(curLength * hDim * sizeof(outType)), 0, 0, 0});
        } else {
            LocalTensor<outType> outLocal16 = outLocal.template ReinterpretCast<outType>();
            Cast(outLocal16, outLocal, RoundMode::CAST_NONE, curLength * hDim);
            PipeBarrier<PIPE_V>();
            DataCopyPad(outGm[gmStartOffset * hDim + static_cast<uint64_t>(progress) * tileLength * hDim], outLocal16,
                        {static_cast<uint16_t>(1), static_cast<uint32_t>(curLength * hDim * sizeof(outType)), 0, 0, 0});
        }
        outQueueOut.FreeTensor(outLocal);
        LocalTensor<float> lseoutLocal = outQueueLse.DeQue<float>();
        if (updateType == 1) {
            uint64_t lseoutGmStart =
                gmStartOffset + static_cast<uint64_t>(progress) * tileLength; // lseout无hDim维度，直接按长度偏移
            uint32_t validBytes = static_cast<uint32_t>(curLength * sizeof(float));
            DataCopyExtParams copyParams{1, validBytes, 0, 0, 0};
            DataCopyPad(lseoutGm[lseoutGmStart], lseoutLocal, copyParams);
        }
        outQueueLse.FreeTensor(lseoutLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueLse;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueIn;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueOut;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueLse;
    TBuf<QuePosition::VECCALC> lseexpBuffer;
    TBuf<QuePosition::VECCALC> lsemaxBuffer;
    TBuf<QuePosition::VECCALC> lseexpsumBuffer;
    TBuf<QuePosition::VECCALC> lseexpBroadcastBuffer;
    TBuf<QuePosition::VECCALC> selMaskBuffer;

    GlobalTensor<lseType> lseGm;
    GlobalTensor<outType> inGm;
    GlobalTensor<outType> outGm;
    GlobalTensor<lseType> lseoutGm;

    __gm__ uint64_t *lsePtr;
    __gm__ uint64_t *inPtr;

    uint32_t blockLength; //  单核上数据总长度
    uint16_t tileLength;  //  单核循环的非最后一轮数据长度
    uint16_t curLength; //  Tiling 循环数据长度为 curLength，最后一轮为 lastLength，内存申请使用 tileLength，数据量用
                        //  curLength
    uint16_t lastLength; //  单核循环的最后一轮数据长度
    uint32_t loopCount;  //  单核循环次数
    uint32_t hDim;       //  head dimension
    uint32_t sp;
    uint32_t tileLengthAlig;
    uint32_t lastLengthAlig;
    uint32_t totalLength; // 由 Tiling 保证元素个数不超过 uint32_t；地址计算（如 totalLength * hDim）使用 uint64_t。
    uint64_t gmStartOffset;
    uint32_t updateType;
    uint32_t spLoopCount;
    uint32_t spFirstLoop;
    uint32_t spNextLoop;
    uint32_t spAlloc;
};
} // namespace AttentionUpdate

#endif // ASCENDC_ATTENTION_UPDATE_DECODE_UPDATE_H_
