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
 * \file norm_rope_concat_base_kernel.h
 * \brief
 */

#ifndef _NORM_ROPE_CONCAT_BASE_KERNEL_H_
#define _NORM_ROPE_CONCAT_BASE_KERNEL_H_

#include "kernel_operator.h"

namespace nrc {
using namespace AscendC;

constexpr int32_t SINGLE_BUFFER = 1;
constexpr int32_t DOUBLE_BUFFER = 1;
constexpr uint32_t BUF_ID_SIZE = 16;
constexpr size_t TOTAL_NORM_NUM = 8;
constexpr int32_t NUM_TWO = 2;
constexpr int32_t NUM_THREE = 3;
constexpr int32_t NUM_FOUR = 4;
constexpr int32_t SIZE_OF_FLOAT = 4;
constexpr size_t MIN_SHARE_BUFFER = 256;

enum class NormType : uint8_t {
    NONE = 0,
    LAYER_NORM,
    LAYER_NORM_AFFINE,
    RMS_NORM,
    RMS_NORM_AFFINE,
    LAYER_NORM_ACROSS_HEADS,
    LAYER_NORM_AFFINE_ACROSS_HEADS,
    RMS_NORM_ACROSS_HEADS,
    RMS_NORM_AFFINE_ACROSS_HEADS,
    L2_NORM,
};

__aicore__ inline constexpr bool IsLayerNorm(NormType normType)
{
    return (normType == NormType::LAYER_NORM || normType == NormType::LAYER_NORM_AFFINE ||
            normType == NormType::LAYER_NORM_ACROSS_HEADS || normType == NormType::LAYER_NORM_AFFINE_ACROSS_HEADS);
}

__aicore__ inline constexpr bool IsRMSNorm(NormType normType)
{
    return (normType == NormType::RMS_NORM || normType == NormType::RMS_NORM_AFFINE);
}

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

template <RopeType ropeType>
class RopeOperation {
public:
    __aicore__ inline RopeOperation(TPipe *pipe, GM_ADDR sin, GM_ADDR cos, uint32_t actualSeq, uint32_t ropeDim,
                                    uint32_t ropeNum, uint32_t alignedRopeDim)
        : ropeDim_(ropeDim),
          ropeNum_(ropeNum),
          alignedRopeDim_(alignedRopeDim),
          totalRopeDim_(ropeNum * alignedRopeDim),
          actualSeq_(actualSeq)
    {
        if constexpr (ropeType == RopeType::NONE) {
            return;
        }

        sinGm_.SetGlobalBuffer((__gm__ DTYPE_ROPE_SIN *)sin);
        cosGm_.SetGlobalBuffer((__gm__ DTYPE_ROPE_SIN *)cos);
        pipe->InitBuffer(ropeQueue_, SINGLE_BUFFER, totalRopeDim_ * NUM_TWO * sizeof(DTYPE_ROPE_SIN));
        pipe->InitBuffer(buf_, (totalRopeDim_ * sizeof(int32_t) + totalRopeDim_ * NUM_TWO * sizeof(float)));
        sin_ = buf_.GetWithOffset<float>(totalRopeDim_, 0);
        cos_ = buf_.GetWithOffset<float>(totalRopeDim_, totalRopeDim_ * sizeof(float));
        mask_ = buf_.GetWithOffset<uint32_t>(totalRopeDim_, totalRopeDim_ * sizeof(float) * NUM_TWO);

        sinRepeatTimes_ =
            static_cast<uint8_t>(CeilDiv(alignedRopeDim, static_cast<uint32_t>(B32_DATA_NUM_PER_REPEAT))) - 1;
        uint32_t tailNum = alignedRopeDim - sinRepeatTimes_ * B32_DATA_NUM_PER_REPEAT;
        sinTailMask_[0] =
            tailNum == B32_DATA_NUM_PER_REPEAT ? 0x5555555555555555 : 0x5555555555555555 & ((1UL << tailNum) - 1);
        uint32_t halfDim = ropeDim_ / NUM_TWO;
        Duplicate(mask_, 0U, alignedRopeDim_);
        if constexpr (ropeType == RopeType::INTERLEAVE) {
            // 0, 1, 2, 3, 4, 5, 6, 7 -> 1, 0, 3, 2, 5, 4, 7, 6
            for (uint32_t i = 0; i < halfDim; ++i) {
                mask_.SetValue(NUM_TWO * i, (NUM_TWO * i + 1) * SIZE_OF_FLOAT);
                mask_.SetValue(NUM_TWO * i + 1, NUM_TWO * i * SIZE_OF_FLOAT);
            }
        } else if constexpr (ropeType == RopeType::HALF) {
            // 0, 1, 2, 3, 4, 5, 6, 7 -> 4, 5, 6, 7, 0, 1, 2, 3
            for (uint32_t i = 0; i < halfDim; ++i) {
                mask_.SetValue(i, (halfDim + i) * SIZE_OF_FLOAT);
                mask_.SetValue(halfDim + i, i * SIZE_OF_FLOAT);
            }
        }
        for (int32_t i = 1; i < ropeNum_; ++i) {
            Adds(mask_[i * alignedRopeDim_].ReinterpretCast<int32_t>(), mask_.ReinterpretCast<int32_t>(),
                 static_cast<int32_t>(i * alignedRopeDim_ * NUM_FOUR), alignedRopeDim_);
        }
    }

    template <RopeType actualRopeType>
    __aicore__ inline void PreProcess(uint32_t curSeq)
    {
        if constexpr (actualRopeType == RopeType::NONE) {
            return;
        }
        isActive_ = curSeq < actualSeq_;
        if (!isActive_) {
            return;
        }
        LocalTensor<DTYPE_ROPE_SIN> rope = ropeQueue_.AllocTensor<DTYPE_ROPE_SIN>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(ropeDim_ * sizeof(DTYPE_QUERY)), 0, 0, 0};
        DataCopyPadExtParams<DTYPE_QUERY> padParams{false, 0, 0, 0};
        DataCopyPad(rope, sinGm_[curSeq * ropeDim_], copyParams, padParams);
        DataCopyPad(rope[totalRopeDim_], cosGm_[curSeq * ropeDim_], copyParams, padParams);
        ropeQueue_.EnQue(rope);
        LocalTensor<DTYPE_ROPE_SIN> deQue = ropeQueue_.DeQue<DTYPE_ROPE_SIN>();
        if (sizeof(DTYPE_ROPE_SIN) < sizeof(float)) {
            Cast(sin_, deQue, RoundMode::CAST_NONE, totalRopeDim_);
            Cast(cos_, deQue[totalRopeDim_], RoundMode::CAST_NONE, totalRopeDim_);
        } else {
            Adds(sin_, deQue.ReinterpretCast<float>(), 0.f, totalRopeDim_);
            Adds(cos_, deQue[totalRopeDim_].ReinterpretCast<float>(), 0.f, totalRopeDim_);
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
            Muls(sin_, sin_, -1.f, ropeDim_ / 2);
        }
        PipeBarrier<PIPE_V>();
        for (int32_t i = 1; i < ropeNum_; ++i) {
            Adds(cos_[i * alignedRopeDim_], cos_, 0.f, alignedRopeDim_);
            Adds(sin_[i * alignedRopeDim_], sin_, 0.f, alignedRopeDim_);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Rotate(const LocalTensor<float> &x)
    {
        Gather(x[totalRopeDim_], x, mask_, 0, totalRopeDim_);
    }

    __aicore__ inline void Collect(const LocalTensor<float> &x)
    {
        Mul(x, x, cos_, totalRopeDim_);
        PipeBarrier<PIPE_V>();
        MulAddDst(x, x[totalRopeDim_], sin_, totalRopeDim_);
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
    uint32_t ropeNum_;
    uint32_t alignedRopeDim_;
    uint32_t totalRopeDim_;
    bool isActive_;
    uint32_t actualSeq_;
    uint64_t sinRepeatMask_[1] = {0x5555555555555555};
    uint64_t sinTailMask_[1] = {0};
    uint8_t sinRepeatTimes_;
};

class NormOperation {
public:
    __aicore__ inline NormOperation(float eps, float scale, uint32_t normDim, uint32_t normNum, uint32_t alignedNormDim,
                                    uint32_t alignedNormNum)
        : eps_(eps),
          scale_(scale),
          normDim_(normDim),
          normNum_(normNum),
          alignedNormDim_(alignedNormDim),
          alignedNormNum_(alignedNormNum)
    {}

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
} // namespace nrc

#endif // _NORM_ROPE_CONCAT_BASE_H_
