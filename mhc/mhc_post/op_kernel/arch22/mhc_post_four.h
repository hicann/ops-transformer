/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mhc_post_four.h
 * \brief Four-head broadcast kernel for aligned D tiles.
 */

#ifndef ASCENDC_MHC_POST_FOUR_H
#define ASCENDC_MHC_POST_FOUR_H

#include "kernel_operator.h"
#include "mhc_post_four_constants.h"
#include "mhc_post_tiling_data.h"

namespace MhcPost {
template <typename T>
class MhcPostFour {
public:
    __aicore__ inline void Init(TPipe* pipe, const MhcPostTilingData* tiling, GM_ADDR x, GM_ADDR hRes, GM_ADDR hOut,
                                GM_ADDR hPost, GM_ADDR output)
    {
        tiling_ = tiling;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        hOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(hOut));
        hResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(hRes));
        hPostGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(hPost));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output));
        pipe->InitBuffer(hPostQueue_, 1, DATA_BLOCK_BYTES);
        pipe->InitBuffer(hResQueue_, 1, HEADS * DATA_BLOCK_BYTES);
        pipe->InitBuffer(inputQueue_, BUFFER_COUNT, HEADS * tiling_->dInner * sizeof(T));
        pipe->InitBuffer(outputQueue_, BUFFER_COUNT, HEADS * tiling_->dInner * sizeof(T));
        pipe->InitBuffer(outBuffer_, HEADS * tiling_->dInner * sizeof(float) + BANK_OFFSET);
        pipe->InitBuffer(inputBuffer_, HEADS * tiling_->dInner * sizeof(float));
        pipe->InitBuffer(postBroadcast_, REPEAT_ELEMENTS * sizeof(float));
        pipe->InitBuffer(resBroadcast_, HEADS * REPEAT_ELEMENTS * sizeof(float));
        pipe->InitBuffer(sumBuffer_, HEADS * tiling_->dInner * sizeof(float) + BANK_OFFSET);
    }

    __aicore__ inline void Process()
    {
        int64_t block = GetBlockIdx();
        if (block >= tiling_->usedCoreNum) {
            return;
        }
        int64_t count = block + 1 == tiling_->usedCoreNum ? tiling_->tailCoreProcessNum : tiling_->normalCoreProcessNum;
        int64_t first = block * tiling_->normalCoreProcessNum;
        int64_t previousBatch = -1;
        for (int64_t item = first; item < first + count; ++item) {
            int64_t batch = item / tiling_->dOuter;
            int64_t tile = item % tiling_->dOuter;
            if (batch != previousBatch) {
                LoadWeights(batch);
                previousBatch = batch;
            }
            int64_t length = tile + 1 == tiling_->dOuter ? tiling_->dTail : tiling_->dInner;
            Compute(batch, tile, length);
        }
    }

private:
    static constexpr int64_t HEADS = FOUR_HEAD_COUNT;
    static constexpr uint32_t DATA_BLOCK_BYTES = 32;
    static constexpr int64_t BLOCK_ELEMENTS = DATA_BLOCK_BYTES / sizeof(float);
    static constexpr int64_t REPEAT_ELEMENTS = 64;
    static constexpr int64_t BANK_OFFSET = 256;
    static constexpr uint8_t REPEAT_BLOCK_STRIDE = 8;
    static constexpr uint8_t BUFFER_COUNT = 2;

    __aicore__ inline void LoadWeights(int64_t batch)
    {
        auto post = hPostQueue_.AllocTensor<float>();
        auto res = hResQueue_.AllocTensor<float>();
        DataCopyExtParams postParams = {1, HEADS * sizeof(float), 0, 0, 0};
        DataCopyExtParams resParams = {HEADS, HEADS * sizeof(float), 0, 0, 0};
        DataCopyPad(post, hPostGm_[batch * HEADS], postParams, {false, 0, 0, 0});
        DataCopyPad(res, hResGm_[batch * HEADS * HEADS], resParams, {false, 0, 0, 0});
        hPostQueue_.EnQue(post);
        hResQueue_.EnQue(res);
        post = hPostQueue_.DeQue<float>();
        res = hResQueue_.DeQue<float>();
        auto postBrcb = postBroadcast_.Get<float>();
        auto resBrcb = resBroadcast_.Get<float>();
        Brcb(postBrcb, post, 1, {1, REPEAT_BLOCK_STRIDE});
        for (int64_t head = 0; head < HEADS; ++head) {
            Brcb(resBrcb[head * REPEAT_ELEMENTS], res[head * BLOCK_ELEMENTS], 1, {1, REPEAT_BLOCK_STRIDE});
        }
        PipeBarrier<PIPE_V>();
        hPostQueue_.FreeTensor(post);
        hResQueue_.FreeTensor(res);
    }

    __aicore__ inline void BroadcastMul(LocalTensor<float> dst, LocalTensor<float> src, LocalTensor<float> weights,
                                        int64_t length)
    {
        BinaryRepeatParams params = {1, 1, 0, REPEAT_BLOCK_STRIDE, REPEAT_BLOCK_STRIDE, 0};
        for (int64_t head = 0; head < HEADS; ++head) {
            Mul(dst[head * length], src, weights[head * BLOCK_ELEMENTS], REPEAT_ELEMENTS, length / REPEAT_ELEMENTS,
                params);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Compute(int64_t batch, int64_t tile, int64_t length)
    {
        int64_t dStart = tile * tiling_->dInner;
        auto input = inputQueue_.AllocTensor<T>();
        DataCopyExtParams hOutParams = {1, static_cast<uint32_t>(length * sizeof(T)), 0, 0, 0};
        DataCopyPad(input, hOutGm_[batch * tiling_->d + dStart], hOutParams, {false, 0, 0, 0});
        inputQueue_.EnQue(input);
        input = inputQueue_.DeQue<T>();
        auto inputF32 = inputBuffer_.Get<float>();
        auto outF32 = outBuffer_.Get<float>();
        // Keep the Add operands in different UB bank groups.
        auto sumF32 = sumBuffer_.Get<float>()[BANK_OFFSET / sizeof(float)];
        Cast(inputF32, input, RoundMode::CAST_NONE, length);
        PipeBarrier<PIPE_V>();
        BroadcastMul(outF32, inputF32, postBroadcast_.Get<float>(), length);
        inputQueue_.FreeTensor(input);

        input = inputQueue_.AllocTensor<T>();
        DataCopyExtParams xParams = {HEADS, static_cast<uint32_t>(length * sizeof(T)),
                                     static_cast<uint32_t>((tiling_->d - length) * sizeof(T)), 0, 0};
        DataCopyPad(input, xGm_[batch * HEADS * tiling_->d + dStart], xParams, {false, 0, 0, 0});
        inputQueue_.EnQue(input);
        input = inputQueue_.DeQue<T>();
        Cast(inputF32, input, RoundMode::CAST_NONE, HEADS * length);
        PipeBarrier<PIPE_V>();
        auto weights = resBroadcast_.Get<float>();
        for (int64_t head = 0; head < HEADS; ++head) {
            BroadcastMul(sumF32, inputF32[head * length], weights[head * REPEAT_ELEMENTS], length);
            Add(outF32, outF32, sumF32, HEADS * length);
            PipeBarrier<PIPE_V>();
        }
        inputQueue_.FreeTensor(input);
        auto output = outputQueue_.AllocTensor<T>();
        Cast(output, outF32, RoundMode::CAST_RINT, HEADS * length);
        outputQueue_.EnQue(output);
        output = outputQueue_.DeQue<T>();
        DataCopyExtParams outParams = {HEADS, static_cast<uint32_t>(length * sizeof(T)), 0,
                                       static_cast<uint32_t>((tiling_->d - length) * sizeof(T)), 0};
        DataCopyPad(outputGm_[batch * HEADS * tiling_->d + dStart], output, outParams);
        outputQueue_.FreeTensor(output);
    }

    const MhcPostTilingData* tiling_;
    TQue<QuePosition::VECIN, 1> hPostQueue_;
    TQue<QuePosition::VECIN, 1> hResQueue_;
    TQue<QuePosition::VECIN, 1> inputQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;
    TBuf<QuePosition::VECCALC> outBuffer_;
    TBuf<QuePosition::VECCALC> inputBuffer_;
    TBuf<QuePosition::VECCALC> sumBuffer_;
    TBuf<QuePosition::VECCALC> postBroadcast_;
    TBuf<QuePosition::VECCALC> resBroadcast_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> hOutGm_;
    GlobalTensor<T> outputGm_;
    GlobalTensor<float> hResGm_;
    GlobalTensor<float> hPostGm_;
};
} // namespace MhcPost
#endif // ASCENDC_MHC_POST_FOUR_H
