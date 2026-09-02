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
 * \file interleave_rope_b1sd.h
 * \brief
 */
#ifndef _INTERLEAVE_ROPE_B1SD_H_
#define _INTERLEAVE_ROPE_B1SD_H_
#include "platform/platform_infos_def.h"
#include "kernel_operator.h"

namespace InterleaveRope {
using namespace AscendC;

template <typename T>
class KernelInterleaveRopeB1SD {
public:
    __aicore__ inline KernelInterleaveRopeB1SD(TPipe *pipe, const InterleaveRopeTilingData *tiling)
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

        numHead_ = tilingData_->numHead;
        seqLength_ = tilingData_->seqLength;
        NS_ = numHead_ * seqLength_;

        batchsPerBlock_ = tilingData_->batchsPerBlock;
        curBlockBatchs_ = tilingData_->batchsPerBlock;
        batchLoops_ = tilingData_->batchLoops;
        batchPerLoop_ = tilingData_->batchPerLoop;
        batchLastLoop_ = tilingData_->batchLastLoop;
        hiddenDimLoops_ = tilingData_->hiddenDimLoopsPerBlock;
        hiddenDimCountPerLoop_ = tilingData_->hiddenDimCountPerLoopPerBlock;
        hiddenDimCountLastLoop_ = tilingData_->hiddenDimCountLastLoopPerBlock;
        if (GetBlockIdx() == GetBlockNum() - 1) {
            curBlockBatchs_ = tilingData_->batchsLastBlock;
            hiddenDimLoops_ = tilingData_->hiddenDimLoopsLastBlock;
            hiddenDimCountPerLoop_ = tilingData_->hiddenDimCountPerLoopLastBlock;
            hiddenDimCountLastLoop_ = tilingData_->hiddenDimCountLastLoopLastBlock;
        }

        xGm.SetGlobalBuffer((__gm__ T *)x);
        yGm.SetGlobalBuffer((__gm__ T *)y);
        cosGm.SetGlobalBuffer((__gm__ T *)cos);
        sinGm.SetGlobalBuffer((__gm__ T *)sin);

        // init pipe
        pipe_->InitBuffer(inQueueX, 1, hiddenDimCountPerLoop_ * hiddenDim * sizeof(T));
        pipe_->InitBuffer(outQueueY, 1, hiddenDimCountPerLoop_ * hiddenDim * sizeof(float) * numTwo);
        pipe_->InitBuffer(inQueueCos, 1, hiddenDimCountPerLoop_ * hiddenDim * sizeof(float) * numTwo);
        pipe_->InitBuffer(inQueueSin, 1, hiddenDimCountPerLoop_ * hiddenDim * sizeof(float) * numTwo);

        pipe_->InitBuffer(bufferReal, hiddenDimCountPerLoop_ * hiddenDimHalf * sizeof(float) * numTwo);
        pipe_->InitBuffer(bufferImag, hiddenDimCountPerLoop_ * hiddenDimHalf * sizeof(float) * numTwo);
        pipe_->InitBuffer(buffer_, hiddenDimCountPerLoop_ * hiddenDim * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (int64_t i = 0; i < curBlockBatchs_; i++) {
            Rope(i);
        }
    }

    __aicore__ inline void Rope(int64_t idx)
    {
        int64_t hiddenDimCount = hiddenDimCountPerLoop_;
        for (int64_t hiddenDimLoop = 0; hiddenDimLoop < hiddenDimLoops_; hiddenDimLoop++) {
            if (hiddenDimLoop == hiddenDimLoops_ - 1) {
                hiddenDimCount = hiddenDimCountLastLoop_;
            }
            int64_t sinCosIdxOffset = ((GetBlockIdx() * tilingData_->batchsPerBlock + idx) * seqLength_ +
                                       hiddenDimLoop * hiddenDimCountPerLoop_) *
                                      hiddenDim;
            LocalTensor<float> cosLocal = inQueueCos.AllocTensor<float>();
            DataCopy(cosLocal[hiddenDimCount * hiddenDim].template ReinterpretCast<T>(), cosGm[sinCosIdxOffset],
                     hiddenDimCount * hiddenDim);

            inQueueCos.EnQue(cosLocal);
            cosLocal = inQueueCos.DeQue<float>();
            Cast(cosLocal, cosLocal[hiddenDimCount * hiddenDim].template ReinterpretCast<T>(), RoundMode::CAST_NONE,
                 hiddenDimCount * hiddenDim);

            LocalTensor<float> sinLocal = inQueueSin.AllocTensor<float>();
            DataCopy(sinLocal[hiddenDimCount * hiddenDim].template ReinterpretCast<T>(), sinGm[sinCosIdxOffset],
                     hiddenDimCount * hiddenDim);
            inQueueSin.EnQue(sinLocal);
            sinLocal = inQueueSin.DeQue<float>();
            Cast(sinLocal, sinLocal[hiddenDimCount * hiddenDim].template ReinterpretCast<T>(), RoundMode::CAST_NONE,
                 hiddenDimCount * hiddenDim);

            int64_t batchOffset = (GetBlockIdx() * tilingData_->batchsPerBlock + idx) * numHead_ * seqLength_;
            for (int64_t nIdx = 0; nIdx < numHead_; nIdx++) {
                int64_t xOffset =
                    (batchOffset + nIdx * seqLength_ + hiddenDimLoop * hiddenDimCountPerLoop_) * hiddenDim;
                // load xLocal
                LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
                DataCopy(xLocal, xGm[xOffset], hiddenDimCount * hiddenDim);
                inQueueX.EnQue(xLocal);
                xLocal = inQueueX.DeQue<T>();

                // split Real and Imag
                LocalTensor<float> realLocal = bufferReal.Get<float>();
                LocalTensor<float> imagLocal = bufferImag.Get<float>();
                LocalTensor<float> buf_ = buffer_.Get<float>();
                uint64_t rsvdCnt = 0;
                GatherMask(realLocal[hiddenDimCount * hiddenDimHalf].template ReinterpretCast<T>(), xLocal, 1, true,
                           hiddenDimCount * hiddenDim, {1, 1, 8, 0}, rsvdCnt);
                GatherMask(imagLocal[hiddenDimCount * hiddenDimHalf].template ReinterpretCast<T>(), xLocal, numTwo,
                           true, hiddenDimCount * hiddenDim, {1, 1, 8, 0}, rsvdCnt);
                Cast(realLocal, realLocal[hiddenDimCount * hiddenDimHalf].template ReinterpretCast<T>(),
                     RoundMode::CAST_NONE, hiddenDimCount * hiddenDimHalf);
                Cast(imagLocal, imagLocal[hiddenDimCount * hiddenDimHalf].template ReinterpretCast<T>(),
                     RoundMode::CAST_NONE, hiddenDimCount * hiddenDimHalf);
                inQueueX.FreeTensor(xLocal);

                uint64_t mask[numTwo] = {0xffffffff, 0}; // mask hiddenDimHalf Elements
                LocalTensor<float> outLocal = outQueueY.AllocTensor<float>();
                Mul(outLocal, realLocal, cosLocal, mask, hiddenDimCount, {1, 1, 1, 8, 4, 8});
                Mul(outLocal[hiddenDimHalf], imagLocal, cosLocal[hiddenDimHalf], mask, hiddenDimCount,
                    {1, 1, 1, 8, 4, 8});
                PipeBarrier<PIPE_V>();
                Muls<float>(imagLocal, imagLocal, -1.0f, hiddenDimCount * hiddenDimHalf);
                PipeBarrier<PIPE_V>();
                Mul(buf_, imagLocal, sinLocal, mask, hiddenDimCount, {1, 1, 1, 8, 4, 8});
                Mul(buf_[hiddenDimHalf], realLocal, sinLocal[hiddenDimHalf], mask, hiddenDimCount, {1, 1, 1, 8, 4, 8});
                PipeBarrier<PIPE_V>();
                Add(outLocal, outLocal, buf_, hiddenDimCount * hiddenDim);
                PipeBarrier<PIPE_V>();
                Cast(outLocal[hiddenDimCount * hiddenDim].template ReinterpretCast<T>(), outLocal, RoundMode::CAST_RINT,
                     hiddenDimCount * hiddenDim);
                PipeBarrier<PIPE_V>();
                outQueueY.EnQue(outLocal);
                outLocal = outQueueY.DeQue<float>();
                DataCopy(yGm[xOffset], outLocal[hiddenDimCount * hiddenDim].template ReinterpretCast<T>(),
                         hiddenDimCount * hiddenDim);
                outQueueY.FreeTensor(outLocal);
            }
            inQueueCos.FreeTensor(cosLocal);
            inQueueSin.FreeTensor(sinLocal);
        }
    }

private:
    TPipe *pipe_ = nullptr;
    const InterleaveRopeTilingData *tilingData_;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<T> cosGm;
    GlobalTensor<T> sinGm;

    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECIN, 1> inQueueCos;
    TQue<QuePosition::VECIN, 1> inQueueSin;
    TQue<QuePosition::VECOUT, 1> outQueueY;
    TBuf<TPosition::VECCALC> bufferReal;
    TBuf<TPosition::VECCALC> bufferImag;
    TBuf<TPosition::VECCALC> buffer_;
    int64_t numHead_ = 0;
    int64_t seqLength_ = 0;
    int64_t NS_ = 0;

    constexpr static int64_t hiddenDim = 64;
    constexpr static int64_t hiddenDimHalf = 32;
    constexpr static int64_t numTwo = 2;
    int64_t batchLoops_ = 0;
    int64_t batchPerLoop_ = 0;
    int64_t batchLastLoop_ = 0;

    int64_t batchsPerBlock_ = 0;
    int64_t curBlockBatchs_ = 0;
    int64_t hiddenDimLoops_ = 0;
    int64_t hiddenDimCountPerLoop_ = 0;
    int64_t hiddenDimCountLastLoop_ = 0;
};
} // namespace InterleaveRope

#endif // _INTERLEAVE_ROPE_B1SD_H_
