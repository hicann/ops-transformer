/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file norm_rope_concat_grad_base_kernel.h
 * \brief
 */

#ifndef NORM_ROPE_CONCAT_GRAD_BASE_KERNEL_H
#define NORM_ROPE_CONCAT_GRAD_BASE_KERNEL_H

#include "kernel_operator.h"

namespace nrcg {
using namespace AscendC;

constexpr int32_t SINGLE_BUFFER = 1;
constexpr int32_t DOUBLE_BUFFER = 1;
constexpr uint32_t BUF_ID_SIZE = 16;
constexpr size_t TOTAL_NORM_NUM = 8;
constexpr int32_t NUM_TWO = 2;
constexpr int32_t NUM_THREE = 3;
constexpr size_t MIN_SHARE_BUFFER = 256;
constexpr uint32_t MIN_UB_AVGHEADS_NUM = 2;
constexpr uint32_t MAX_AFFINE_BLOCK_NUM = 40;

enum class NormType : uint8_t {
    NONE = 0,
    LAYER_NORM,
    LAYER_NORM_AFFINE,
    LAYER_NORM_ACROSS_HEADS,
    LAYER_NORM_AFFINE_ACROSS_HEADS,
    RMS_NORM,
    RMS_NORM_AFFINE,
    RMS_NORM_ACROSS_HEADS,
    RMS_NORM_AFFINE_ACROSS_HEADS,
    L2_NORM,
};

// read_store
enum class RopeType : uint8_t {
    NONE = 0,
    INTERLEAVE,
    HALF,
};

enum class ConcatOrder : uint8_t {
    BEFORE_ENCODER = 0,
    AFTER_ENCODER,
};

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a > b ? b : a;
}

template <typename T>
__aicore__ inline T Max(T a, T b)
{
    return a > b ? a : b;
}

template <typename T>
__aicore__ inline T CeilDiv(T a, T b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

template <typename T>
__aicore__ inline T CeilAlign(T a, T b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b * b;
}

__aicore__ inline void VToMTE3Sync()
{
    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
}
__aicore__ inline void MTE3ToMTE2Sync()
{
    event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
}

__aicore__ inline void MTE2ToVSync()
{
    event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
}

template <RopeType ropeType, bool forward>
class RopeOperation {
public:
    __aicore__ inline RopeOperation(TBufPool<TPosition::VECCALC, BUF_ID_SIZE> &bufPool, GM_ADDR sin, GM_ADDR cos,
                                    GM_ADDR y, uint32_t actualSeq, uint32_t curSeq, uint32_t ropeDim,
                                    uint32_t alignedRopeDim)
        : ropeDim_(ropeDim),
          alignedRopeDim_(alignedRopeDim),
          isActive_(curSeq < actualSeq),
          actualSeq_(actualSeq)
    {
        yGm_.SetGlobalBuffer((__gm__ DTYPE_QUERY *)y);
        if constexpr (ropeType == RopeType::NONE) {
            return;
        }

        if (!isActive_) {
            return;
        }
        sinGm_.SetGlobalBuffer((__gm__ DTYPE_ROPE_SIN *)sin);
        cosGm_.SetGlobalBuffer((__gm__ DTYPE_ROPE_SIN *)cos);
        bufPool.InitBuffer(ropeQueue_, SINGLE_BUFFER, alignedRopeDim * NUM_TWO * sizeof(DTYPE_ROPE_SIN));
        bufPool.InitBuffer(buf_, (alignedRopeDim * sizeof(int32_t) + alignedRopeDim * NUM_TWO * sizeof(float)));
        sin_ = buf_.GetWithOffset<float>(alignedRopeDim * NUM_TWO, 0);
        cos_ = buf_.GetWithOffset<float>(alignedRopeDim, alignedRopeDim * sizeof(float));
        mask_ = buf_.GetWithOffset<uint32_t>(alignedRopeDim, alignedRopeDim * sizeof(float) * NUM_TWO);

        sinRepeatTimes_ =
            static_cast<uint8_t>(CeilDiv(alignedRopeDim, static_cast<uint32_t>(B32_DATA_NUM_PER_REPEAT))) - 1;
        uint32_t tailNum = alignedRopeDim - sinRepeatTimes_ * B32_DATA_NUM_PER_REPEAT;
        sinTailMask_[0] =
            tailNum == B32_DATA_NUM_PER_REPEAT ? 0x5555555555555555 : 0x5555555555555555 & ((1UL << tailNum) - 1);
    }

    __aicore__ inline void Prepare(uint32_t curSeq)
    {
        if constexpr (ropeType == RopeType::NONE) {
            return;
        }
        if (!isActive_) {
            return;
        }
        uint32_t halfDim = ropeDim_ / NUM_TWO;
        Duplicate(mask_, 0U, alignedRopeDim_);
        if constexpr (ropeType == RopeType::INTERLEAVE) {
            // 0, 1, 2, 3, 4, 5, 6, 7 -> 1, 0, 3, 2, 5, 4, 7, 6
            for (uint32_t i = 0; i < halfDim; ++i) {
                mask_.SetValue(NUM_TWO * i, (NUM_TWO * i + 1) * sizeof(float));
                mask_.SetValue(NUM_TWO * i + 1, NUM_TWO * i * sizeof(float));
            }
        } else if constexpr (ropeType == RopeType::HALF) {
            // 0, 1, 2, 3, 4, 5, 6, 7 -> 4, 5, 6, 7, 0, 1, 2, 3
            for (uint32_t i = 0; i < halfDim; ++i) {
                if constexpr (forward) {
                    mask_.SetValue(i, (halfDim + i) * sizeof(float));
                    mask_.SetValue(halfDim + i, i * sizeof(float));
                } else {
                    mask_.SetValue(halfDim + i, i * sizeof(float));
                    mask_.SetValue(i, (halfDim + i) * sizeof(float));
                }
            }
        }
    }

    __aicore__ inline void PreProcess(uint32_t curSeq)
    {
        if constexpr (ropeType == RopeType::NONE) {
            return;
        }
        isActive_ = curSeq < actualSeq_;
        if (!isActive_) {
            return;
        }
        LocalTensor<DTYPE_ROPE_SIN> rope = ropeQueue_.AllocTensor<DTYPE_ROPE_SIN>();
        auto sinLen = ropeDim_ * actualSeq_;
        auto curSeqropeDimLen = curSeq * ropeDim_;
        auto leftLen = sinLen - curSeqropeDimLen;
        if (leftLen >= alignedRopeDim_) {
            DataCopyExtParams params{1, static_cast<uint32_t>(alignedRopeDim_ * sizeof(DTYPE_ROPE_SIN)), 0, 0, 0};
            DataCopyPadExtParams<DTYPE_ROPE_SIN> padParams{false, 0, 0, 0};
            DataCopyPad(rope, sinGm_[curSeq * ropeDim_], params, padParams);
            DataCopyPad(rope[alignedRopeDim_], cosGm_[curSeq * ropeDim_], params, padParams);
        } else {
            DataCopyExtParams params{1, static_cast<uint32_t>(leftLen * sizeof(DTYPE_ROPE_SIN)), 0, 0, 0};
            DataCopyPadExtParams<DTYPE_ROPE_SIN> padParams{true, 0, static_cast<uint8_t>(alignedRopeDim_ - leftLen), 0};
            DataCopyPad(rope, sinGm_[curSeq * ropeDim_], params, padParams);
            DataCopyPad(rope[alignedRopeDim_], cosGm_[curSeq * ropeDim_], params, padParams);
        }

        ropeQueue_.EnQue(rope);
        LocalTensor<DTYPE_ROPE_SIN> deQue = ropeQueue_.DeQue<DTYPE_ROPE_SIN>();
        if (sizeof(DTYPE_ROPE_SIN) < sizeof(float)) {
            Cast(sin_, deQue, RoundMode::CAST_NONE, NUM_TWO * alignedRopeDim_);
        } else {
            Adds(sin_, deQue.ReinterpretCast<float>(), 0.f, NUM_TWO * alignedRopeDim_);
        }
        ropeQueue_.FreeTensor(deQue);
        PipeBarrier<PIPE_V>();
        if constexpr (ropeType == RopeType::INTERLEAVE) {
            const UnaryRepeatParams params;
            if (sinRepeatTimes_ > 0) {
                Muls(sin_, sin_, -1.f, sinRepeatMask_, sinRepeatTimes_, params);
            }
            Muls(sin_[sinRepeatTimes_ * B32_DATA_NUM_PER_REPEAT], sin_[sinRepeatTimes_ * B32_DATA_NUM_PER_REPEAT], -1.f,
                 sinTailMask_, 1, params);
        } else {
            Muls(sin_, sin_, -1.f, ropeDim_ / NUM_TWO);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Rotate(const LocalTensor<float> &x, const LocalTensor<float> &rotatedX)
    {
        if constexpr (forward) {
            Gather(rotatedX, x, mask_, 0, alignedRopeDim_);
        } else {
            Mul(rotatedX[alignedRopeDim_], x, sin_, alignedRopeDim_);
            PipeBarrier<PIPE_V>();
            Gather(rotatedX, rotatedX[alignedRopeDim_], mask_, 0, alignedRopeDim_);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Collect(const LocalTensor<float> &x, const LocalTensor<float> &rotatedX)
    {
        if constexpr (forward) {
            Mul(x, x, cos_, alignedRopeDim_);
            PipeBarrier<PIPE_V>();
            MulAddDst(x, rotatedX, sin_, alignedRopeDim_);
        } else {
            Mul(x, x, cos_, alignedRopeDim_);
            PipeBarrier<PIPE_V>();
            Add(x, x, rotatedX, alignedRopeDim_);
        }
    }

protected:
    GlobalTensor<DTYPE_ROPE_SIN> sinGm_;
    GlobalTensor<DTYPE_ROPE_SIN> cosGm_;
    GlobalTensor<DTYPE_QUERY> yGm_;
    TQue<TPosition::VECIN, SINGLE_BUFFER> ropeQueue_;
    TBuf<TPosition::VECCALC> buf_;
    LocalTensor<uint32_t> mask_;
    LocalTensor<float> sin_, cos_;
    uint32_t ropeDim_;
    uint32_t alignedRopeDim_;
    bool isActive_;
    uint32_t actualSeq_;
    uint64_t sinRepeatMask_[1] = {0x5555555555555555};
    uint64_t sinTailMask_[1] = {0};
    uint8_t sinRepeatTimes_;
};

template <NormType normType>
class NormOperation {
public:
    __aicore__ inline NormOperation(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR mean, GM_ADDR rstd, float eps,
                                    float scale, uint32_t normDim, uint32_t normNum, uint32_t alignedNormDim,
                                    uint32_t alignedNormNum)
        : eps_(eps),
          scale_(scale),
          normDim_(normDim),
          normNum_(normNum),
          alignedNormDim_(alignedNormDim),
          alignedNormNum_(alignedNormNum)
    {
        xGm_.SetGlobalBuffer((__gm__ DTYPE_QUERY *)x);
        weightGm_.SetGlobalBuffer((__gm__ DTYPE_QUERY *)weight);
        biasGm_.SetGlobalBuffer((__gm__ DTYPE_QUERY *)bias);
        meanGm_.SetGlobalBuffer((__gm__ float *)mean);
        rstdGm_.SetGlobalBuffer((__gm__ float *)rstd);
    }

protected:
    GlobalTensor<DTYPE_QUERY> xGm_;
    GlobalTensor<DTYPE_QUERY> weightGm_; // affine
    GlobalTensor<DTYPE_QUERY> biasGm_;   // affine
    GlobalTensor<float> meanGm_;
    GlobalTensor<float> rstdGm_;
    float eps_;
    float scale_;
    uint32_t normDim_;
    uint32_t normNum_;
    uint32_t alignedNormDim_;
    uint32_t alignedNormNum_;
};
} // namespace nrcg

#endif
