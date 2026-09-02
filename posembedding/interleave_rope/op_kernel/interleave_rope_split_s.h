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
 * \file interleave_rope_split_s.h
 * \brief
 */
#ifndef _INTERLEAVE_ROPE_SPLIT_S_H_
#define _INTERLEAVE_ROPE_SPLIT_S_H_
#include "platform/platform_infos_def.h"
#include "kernel_operator.h"

namespace InterleaveRope {
using namespace AscendC;

template <typename T>
class KernelInterleaveRopeSplitS {
public:
    __aicore__ inline KernelInterleaveRopeSplitS(TPipe *pipe, const InterleaveRopeTilingData *tiling)
        : pipe_(pipe),
          tilingData_(tiling)
    {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR y)
    {
        /*
         * For each block, process
         * x:   [B, N, S, 64]
         * cos: [B, 1, S, 64]
         * sin: [B, 1, S, 64]
         * y:   [B, N, S, 64]
         */
        // tilingData
        int64_t blockFactor = (tilingData_->seqLength + tilingData_->numBlocks - 1) / tilingData_->numBlocks;
        int64_t currentBlockFactor = blockFactor;
        if (GetBlockIdx() == tilingData_->numBlocks - 1) {
            currentBlockFactor = tilingData_->seqLength - GetBlockIdx() * blockFactor;
        }
        ubFactor = 32; // ubfactor: 32
        ubLoop = currentBlockFactor / ubFactor;
        ubTail = currentBlockFactor - ubFactor * ubLoop;

        xGm.SetGlobalBuffer((__gm__ T *)x + GetBlockIdx() * blockFactor * ROPE_LENGTH);
        yGm.SetGlobalBuffer((__gm__ T *)y + GetBlockIdx() * blockFactor * ROPE_LENGTH);
        cosGm.SetGlobalBuffer((__gm__ T *)cos + GetBlockIdx() * blockFactor * ROPE_LENGTH);
        sinGm.SetGlobalBuffer((__gm__ T *)sin + GetBlockIdx() * blockFactor * ROPE_LENGTH);

        // init pipe
        pipe_->InitBuffer(inQueueX, 2, ubFactor * ROPE_LENGTH * sizeof(T));          // 2*ubFactor*64*2=256*ubFactor
        pipe_->InitBuffer(inQueuePosEmb, 2, 2 * ubFactor * ROPE_LENGTH * sizeof(T)); // 2*2*ubFactor*64*2=512*ubFactor
        pipe_->InitBuffer(outQueue, 2, ubFactor * ROPE_LENGTH * sizeof(T));          // 2*ubFactor*64*2=256*ubFactor
        pipe_->InitBuffer(wsBuffer, 8 * ubFactor * ROPE_LENGTH * sizeof(float));     // 8*ubFactor*64*4=2048*ubFactor
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> workspaceBuffer = wsBuffer.Get<float>();

        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams copyParamsContinguous;
        copyParamsContinguous.blockCount = 1;
        int64_t xOffset = 0;
        int64_t freqOffset = 0;
        for (int64_t batchId = 0; batchId < tilingData_->batchSize; batchId++) {
            copyParamsContinguous.blockLen = ubFactor * ROPE_LENGTH * sizeof(T);
            for (int64_t loopIdx = 0; loopIdx < ubLoop; ++loopIdx) {
                freqOffset = batchId * tilingData_->seqLength * ROPE_LENGTH + loopIdx * ubFactor * ROPE_LENGTH;
                /*
                 * cosLocal = [batchId, 0, :ubFactor:, 64]
                 * sinLocal = [batchId, 0, :ubFactor:, 64]
                 */
                LocalTensor<T> cosLocal = inQueuePosEmb.AllocTensor<T>();
                LocalTensor<T> sinLocal = cosLocal[ubFactor * ROPE_LENGTH];
                DataCopyPad(cosLocal, cosGm[freqOffset], copyParamsContinguous, padParams);
                DataCopyPad(sinLocal, sinGm[freqOffset], copyParamsContinguous, padParams);
                inQueuePosEmb.EnQue(cosLocal);
                cosLocal = inQueuePosEmb.DeQue<T>();
                sinLocal = cosLocal[ubFactor * ROPE_LENGTH];

                for (int64_t headId = 0; headId < tilingData_->numHead; ++headId) {
                    xOffset = batchId * tilingData_->numHead * tilingData_->seqLength * ROPE_LENGTH +
                              loopIdx * ubFactor * ROPE_LENGTH + headId * tilingData_->seqLength * ROPE_LENGTH;
                    /*
                     * xLocal = [batchId, headId, seqId: seqId+ubFactor, 64]
                     * yLocal = [batchId, headId, seqId: seqId+ubFactor, 64]
                     */
                    LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
                    DataCopyPad(xLocal, xGm[xOffset], copyParamsContinguous, padParams);
                    inQueueX.EnQue(xLocal);
                    xLocal = inQueueX.DeQue<T>();
                    LocalTensor<T> yLocal = outQueue.AllocTensor<T>();
                    RoPE<T, true, ROPE_LENGTH>(yLocal, xLocal, cosLocal, sinLocal, workspaceBuffer, ubFactor);
                    inQueueX.FreeTensor(xLocal);
                    outQueue.EnQue(yLocal);
                    yLocal = outQueue.DeQue<T>();
                    DataCopyPad(yGm[xOffset], yLocal, copyParamsContinguous);
                    outQueue.FreeTensor(yLocal);
                }
                inQueuePosEmb.FreeTensor(cosLocal);
            }
            copyParamsContinguous.blockLen = ubTail * ROPE_LENGTH * sizeof(T);
            if (ubTail > 0) {
                freqOffset = batchId * tilingData_->seqLength * ROPE_LENGTH + ubLoop * ubFactor * ROPE_LENGTH;
                /*
                 * cosLocal = [batchId, 0, :ubTail:, 64]
                 * sinLocal = [batchId, 0, :ubTail:, 64]
                 */
                LocalTensor<T> cosLocal = inQueuePosEmb.AllocTensor<T>();
                LocalTensor<T> sinLocal = cosLocal[ubFactor * ROPE_LENGTH];
                DataCopyPad(cosLocal, cosGm[freqOffset], copyParamsContinguous, padParams);
                DataCopyPad(sinLocal, sinGm[freqOffset], copyParamsContinguous, padParams);
                inQueuePosEmb.EnQue(cosLocal);
                cosLocal = inQueuePosEmb.DeQue<T>();
                sinLocal = cosLocal[ubFactor * ROPE_LENGTH];

                for (int64_t headId = 0; headId < tilingData_->numHead; ++headId) {
                    xOffset = batchId * tilingData_->numHead * tilingData_->seqLength * ROPE_LENGTH +
                              ubLoop * ubFactor * ROPE_LENGTH + headId * tilingData_->seqLength * ROPE_LENGTH;
                    /*
                     * xLocal = [batchId, headId, seqId: seqId+ubFactor, 64]
                     * yLocal = [batchId, headId, seqId: seqId+ubFactor, 64]
                     */
                    LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
                    DataCopyPad(xLocal, xGm[xOffset], copyParamsContinguous, padParams);
                    inQueueX.EnQue(xLocal);
                    xLocal = inQueueX.DeQue<T>();
                    LocalTensor<T> yLocal = outQueue.AllocTensor<T>();
                    RoPE<T, true, ROPE_LENGTH>(yLocal, xLocal, cosLocal, sinLocal, workspaceBuffer, ubTail);
                    inQueueX.FreeTensor(xLocal);
                    outQueue.EnQue(yLocal);
                    yLocal = outQueue.DeQue<T>();
                    DataCopyPad(yGm[xOffset], yLocal, copyParamsContinguous);
                    outQueue.FreeTensor(yLocal);
                }
                inQueuePosEmb.FreeTensor(cosLocal);
            }
        }
    }

    template <typename T1, bool isElementWise = true, int64_t headSize>
    __aicore__ inline void RoPE(const LocalTensor<T1> &outLocal, const LocalTensor<T1> &xLocal,
                                const LocalTensor<T1> &cosLocal, const LocalTensor<T1> &sinLocal,
                                const LocalTensor<float> &wsLocal, int64_t rows)
    {
        constexpr static int64_t NUM_ONE = 1;
        constexpr static int64_t NUM_TWO = 2;
        constexpr static int64_t NUM_THREE = 3;
        constexpr static int64_t NUM_FOUR = 4;
        constexpr static int64_t NUM_EIGHT = 8;
        if constexpr (isElementWise) {
            /**
             * cosLocalFp32 : [rows * headSize * 0, rows * headSize * 1]
             * sinLocalFp32 : [rows * headSize * 1, rows * headSize * 2]
             * y0           : [rows * headSize * 2, rows * headSize * 3]
             * y1           : [rows * headSize * 3, rows * headSize * 4]
             * realLocalFp32: [rows * headSize * 4, rows * headSize * 5]
             * imagLocalFp32: [rows * headSize * 5, rows * headSize * 6]
             * realLocal    : [rows * headSize * 6, rows * headSize * 7]
             * imagLocal    : [rows * headSize * 7, rows * headSize * 8]
             */
            int64_t cosLocalFp32Offset = rows * headSize * 0;
            int64_t sinLocalFp32Offset = rows * headSize * 1;
            int64_t y0Offset = rows * headSize * 2;
            int64_t y1Offset = rows * headSize * 3;
            int64_t realLocalFp32Offset = rows * headSize * 4;
            int64_t imagLocalFp32Offset = rows * headSize * 5;
            int64_t realLocalOffset = rows * headSize * 6;
            int64_t imagLocalOffset = rows * headSize * 7;

            LocalTensor<float> cosLocalFp32 = wsLocal[cosLocalFp32Offset];
            LocalTensor<float> sinLocalFp32 = wsLocal[sinLocalFp32Offset];
            LocalTensor<float> y0 = wsLocal[y0Offset];
            LocalTensor<float> y1 = wsLocal[y1Offset];
            LocalTensor<float> realLocalFp32 = wsLocal[realLocalFp32Offset];
            LocalTensor<float> imagLocalFp32 = wsLocal[imagLocalFp32Offset];
            LocalTensor<T1> realLocal = wsLocal[realLocalOffset].template ReinterpretCast<T1>();
            LocalTensor<T1> imagLocal = wsLocal[imagLocalOffset].template ReinterpretCast<T1>();

            // step #1: cast cosLocal and sinLocal to fp32
            Cast(cosLocalFp32, cosLocal, RoundMode::CAST_NONE, rows * headSize);
            Cast(sinLocalFp32, sinLocal, RoundMode::CAST_NONE, rows * headSize);
            PipeBarrier<PIPE_V>();

            // step #2: Gather out real and imag
            uint64_t rsvdCnt = 0;
            GatherMask(realLocal, xLocal, NUM_ONE, true, rows * headSize, {1, 1, NUM_EIGHT, 0}, rsvdCnt);
            GatherMask(imagLocal, xLocal, NUM_TWO, true, rows * headSize, {1, 1, NUM_EIGHT, 0}, rsvdCnt);
            PipeBarrier<PIPE_V>();

            // step #3: Cast realLocal and imagLocal to Fp32
            Cast(realLocalFp32, realLocal, RoundMode::CAST_NONE, rows * (headSize >> 1));
            Cast(imagLocalFp32, imagLocal, RoundMode::CAST_NONE, rows * (headSize >> 1));
            PipeBarrier<PIPE_V>();

            // step #4: y0 = (realLocalFp32, imagLocalFp32) * cosLocalFp32
            Mul(y0, realLocalFp32, cosLocalFp32, /* mask */ (headSize >> 1), /* repeat */ rows,
                {1, 1, 1, NUM_EIGHT, NUM_FOUR, NUM_EIGHT});
            Mul(y0[(headSize >> 1)], imagLocalFp32, cosLocalFp32[(headSize >> 1)], /* mask */ (headSize >> 1),
                /* repeat */ rows, {1, 1, 1, NUM_EIGHT, NUM_FOUR, NUM_EIGHT});
            PipeBarrier<PIPE_V>();

            // step #5: y1 = (-imagLocalFp32, realLocalFp32) * sinLocalFp32
            Muls<float>(imagLocalFp32, imagLocalFp32, -1.0f, rows * (headSize >> 1));
            PipeBarrier<PIPE_V>();
            Mul(y1, imagLocalFp32, sinLocalFp32, /* mask */ (headSize >> 1), /* repeat */ rows,
                {1, 1, 1, NUM_EIGHT, NUM_FOUR, NUM_EIGHT});
            Mul(y1[(headSize >> 1)], realLocalFp32, sinLocalFp32[(headSize >> 1)], /* mask */ (headSize >> 1),
                /* repeat */ rows, {1, 1, 1, NUM_EIGHT, NUM_FOUR, NUM_EIGHT});
            PipeBarrier<PIPE_V>();

            // step #6: y0 = y0 + y1
            Add(y0, y0, y1, rows * headSize);
            PipeBarrier<PIPE_V>();

            // step #7: outLocal = Cast(y0, T1)
            if constexpr (std::is_same<T1, bfloat16_t>::value) {
                Cast(outLocal, y0, RoundMode::CAST_RINT, rows * headSize);
            } else if constexpr (std::is_same<T1, half>::value) {
                Cast(outLocal, y0, RoundMode::CAST_NONE, rows * headSize);
            }
        }
    }

private:
    TPipe *pipe_ = nullptr;
    const InterleaveRopeTilingData *tilingData_;
    constexpr static int64_t ROPE_LENGTH = 64;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<T> cosGm;
    GlobalTensor<T> sinGm;

    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECIN, 1> inQueuePosEmb;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TBuf<TPosition::VECCALC> wsBuffer;
    int64_t ubFactor = 1;
    int64_t ubTail = 0;
    int64_t ubLoop = 0;
};
} // namespace InterleaveRope

#endif // _INTERLEAVE_ROPE_SPLIT_S_H_
