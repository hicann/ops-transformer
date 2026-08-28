/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SCALED_COSINE_ATTENTION_SCORE_IMPL_HPP_
#define SCALED_COSINE_ATTENTION_SCORE_IMPL_HPP_

#include "kernel_operator.h"
#include "scaled_cosine_attention_score_tiling_def.h"

namespace NsScaledCosineAttentionScore {
using namespace AscendC;

// Every LocalTensor passed to a vector instruction must start at a 32-byte
// aligned UB address. A float scalar therefore occupies one eight-float slot.
constexpr uint32_t SCALAR_SLOT_FLOATS = 8;
constexpr uint32_t SCALAR_COUNT = 12 * SCALAR_SLOT_FLOATS;

constexpr uint32_t Q_SUM = 0 * SCALAR_SLOT_FLOATS;
constexpr uint32_t Q_SUM_EPS = 1 * SCALAR_SLOT_FLOATS;
constexpr uint32_t Q_NORM = 2 * SCALAR_SLOT_FLOATS;
constexpr uint32_t Q_INV_NORM = 3 * SCALAR_SLOT_FLOATS;
constexpr uint32_t SCALE_CLAMPED = 4 * SCALAR_SLOT_FLOATS;
constexpr uint32_t LOGIT_SCALE = 5 * SCALAR_SLOT_FLOATS;
constexpr uint32_t K_SUM = 6 * SCALAR_SLOT_FLOATS;
constexpr uint32_t K_SUM_EPS = 7 * SCALAR_SLOT_FLOATS;
constexpr uint32_t K_NORM = 8 * SCALAR_SLOT_FLOATS;
constexpr uint32_t K_INV_NORM = 9 * SCALAR_SLOT_FLOATS;
constexpr uint32_t DOT_SUM = 10 * SCALAR_SLOT_FLOATS;
constexpr uint32_t SCORE = 11 * SCALAR_SLOT_FLOATS;

__aicore__ inline uint32_t AlignUpBytes(uint32_t bytes)
{
    return (bytes + 31U) / 32U * 32U;
}

template <typename T>
struct Convert;

template <>
struct Convert<half> {
    __aicore__ static inline void ToFloat(const LocalTensor<float> &dst, const LocalTensor<half> &src, uint32_t count)
    {
        Cast(dst, src, RoundMode::CAST_NONE, count);
    }
    __aicore__ static inline void FromFloat(const LocalTensor<half> &dst, const LocalTensor<float> &src, uint32_t count)
    {
        Cast(dst, src, RoundMode::CAST_NONE, count);
    }
};

template <>
struct Convert<bfloat16_t> {
    __aicore__ static inline void ToFloat(const LocalTensor<float> &dst, const LocalTensor<bfloat16_t> &src,
                                          uint32_t count)
    {
        Cast(dst, src, RoundMode::CAST_NONE, count);
    }
    __aicore__ static inline void FromFloat(const LocalTensor<bfloat16_t> &dst, const LocalTensor<float> &src,
                                            uint32_t count)
    {
        Cast(dst, src, RoundMode::CAST_RINT, count);
    }
};

template <>
struct Convert<float> {
    __aicore__ static inline void ToFloat(const LocalTensor<float> &dst, const LocalTensor<float> &src, uint32_t count)
    {
        Adds(dst, src, 0.0F, count);
    }
    __aicore__ static inline void FromFloat(const LocalTensor<float> &dst, const LocalTensor<float> &src,
                                            uint32_t count)
    {
        Adds(dst, src, 0.0F, count);
    }
};

template <typename T>
class ScaledCosineAttentionScoreImpl {
public:
    __aicore__ inline ScaledCosineAttentionScoreImpl() = default;

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR scale, GM_ADDR output,
                                const optiling::ScaledCosineAttentionScoreTilingData *tiling)
    {
        heads_ = tiling->heads;
        seqLen_ = tiling->seqLen;
        headDim_ = tiling->headDim;
        alignedHeadDim_ = tiling->alignedHeadDim;
        keyTileRows_ = tiling->keyTileRows;
        usedCoreNum_ = tiling->usedCoreNum;
        totalQueryRows_ = tiling->totalQueryRows;
        clampMax_ = tiling->clampMax;
        eps_ = tiling->eps;

        const uint64_t inputElements = static_cast<uint64_t>(tiling->batch) * heads_ * seqLen_ * headDim_;
        const uint64_t outputElements = static_cast<uint64_t>(tiling->batch) * heads_ * seqLen_ * seqLen_;
        queryGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(query), inputElements);
        keyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(key), inputElements);
        scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scale), heads_);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(output), outputElements);

        pipe_.InitBuffer(queryQueue_, 1, AlignUpBytes(alignedHeadDim_ * sizeof(T)));
        pipe_.InitBuffer(keyQueue_, 1, AlignUpBytes(keyTileRows_ * alignedHeadDim_ * sizeof(T)));
        pipe_.InitBuffer(scaleQueue_, 1, 32U);
        pipe_.InitBuffer(outputQueue_, 1, AlignUpBytes(keyTileRows_ * sizeof(T)));
        pipe_.InitBuffer(queryFloatBuf_, AlignUpBytes(alignedHeadDim_ * sizeof(float)));
        pipe_.InitBuffer(keyFloatBuf_, AlignUpBytes(keyTileRows_ * alignedHeadDim_ * sizeof(float)));
        pipe_.InitBuffer(tmp0Buf_, AlignUpBytes(alignedHeadDim_ * sizeof(float)));
        pipe_.InitBuffer(tmp1Buf_, AlignUpBytes(alignedHeadDim_ * sizeof(float)));
        pipe_.InitBuffer(reduceWorkBuf_, AlignUpBytes(alignedHeadDim_ * sizeof(float)));
        pipe_.InitBuffer(outputFloatBuf_, AlignUpBytes(keyTileRows_ * sizeof(float)));
        pipe_.InitBuffer(scalarBuf_, AlignUpBytes(SCALAR_COUNT * sizeof(float)));
    }

    __aicore__ inline void Process()
    {
        const uint64_t core = GetBlockIdx();
        if (core >= usedCoreNum_) {
            return;
        }
        for (uint64_t queryRow = core; queryRow < totalQueryRows_; queryRow += usedCoreNum_) {
            ProcessQueryRow(queryRow);
        }
    }

private:
    __aicore__ inline void CopyOneInputRow(const LocalTensor<T> &dst, const GlobalTensor<T> &src)
    {
        DataCopyExtParams copy{1, static_cast<uint32_t>(headDim_ * sizeof(T)), 0, 0, 0};
        const uint8_t rightPad = static_cast<uint8_t>(alignedHeadDim_ - headDim_);
        DataCopyPadExtParams<T> pad{true, 0, rightPad, 0};
        DataCopyPad(dst, src, copy, pad);
    }

    __aicore__ inline void CopyKeyTile(const LocalTensor<T> &dst, const GlobalTensor<T> &src, uint32_t rows)
    {
        DataCopyExtParams copy{static_cast<uint16_t>(rows), static_cast<uint32_t>(headDim_ * sizeof(T)), 0, 0, 0};
        const uint8_t rightPad = static_cast<uint8_t>(alignedHeadDim_ - headDim_);
        DataCopyPadExtParams<T> pad{true, 0, rightPad, 0};
        DataCopyPad(dst, src, copy, pad);
    }

    __aicore__ inline void CopyScale(uint32_t head)
    {
        LocalTensor<float> scaleLocal = scaleQueue_.AllocTensor<float>();
        DataCopyExtParams copy{1, sizeof(float), 0, 0, 0};
        DataCopyPadExtParams<float> pad{true, 0, 7, 0};
        DataCopyPad(scaleLocal, scaleGm_[head], copy, pad);
        scaleQueue_.EnQue(scaleLocal);
    }

    __aicore__ inline void CopyOutput(uint64_t offset, const LocalTensor<T> &src, uint32_t count)
    {
        DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGm_[offset], src, copy);
    }

    __aicore__ inline void ComputeInverseNorm(const LocalTensor<float> &src, const LocalTensor<float> &tmp,
                                              const LocalTensor<float> &work, const LocalTensor<float> &scalar,
                                              uint32_t sumOffset, uint32_t sumEpsOffset, uint32_t normOffset,
                                              uint32_t invNormOffset)
    {
        // Each vector op depends on the previous UB write; the device vector
        // pipe does not auto-handle RAW, so PipeBarrier<PIPE_V>() is required
        // (no-op under CPU mock, so the UT is unaffected).
        Mul(tmp, src, src, alignedHeadDim_);
        PipeBarrier<PIPE_V>();
        ReduceSum<float>(scalar[sumOffset], tmp, work, alignedHeadDim_);
        Adds(scalar[sumEpsOffset], scalar[sumOffset], eps_, 1);
        PipeBarrier<PIPE_V>();
        Sqrt(scalar[normOffset], scalar[sumEpsOffset], 1);
        PipeBarrier<PIPE_V>();
        Duplicate(scalar[invNormOffset], 1.0F, 1);
        PipeBarrier<PIPE_V>();
        Div(scalar[invNormOffset], scalar[invNormOffset], scalar[normOffset], 1);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessQueryRow(uint64_t queryRow)
    {
        const uint64_t bh = queryRow / seqLen_;
        const uint32_t head = static_cast<uint32_t>(bh % heads_);
        const uint64_t queryOffset = queryRow * headDim_;
        const uint64_t keyHeadOffset = bh * seqLen_ * headDim_;
        const uint64_t outputRowOffset = queryRow * seqLen_;

        LocalTensor<T> queryLocal = queryQueue_.AllocTensor<T>();
        CopyOneInputRow(queryLocal, queryGm_[queryOffset]);
        queryQueue_.EnQue(queryLocal);
        CopyScale(head);

        queryLocal = queryQueue_.DeQue<T>();
        LocalTensor<float> scaleLocal = scaleQueue_.DeQue<float>();
        LocalTensor<float> queryFloat = queryFloatBuf_.Get<float>();
        LocalTensor<float> tmp0 = tmp0Buf_.Get<float>();
        LocalTensor<float> tmp1 = tmp1Buf_.Get<float>();
        LocalTensor<float> work = reduceWorkBuf_.Get<float>();
        LocalTensor<float> scalar = scalarBuf_.Get<float>();

        Convert<T>::ToFloat(queryFloat, queryLocal, alignedHeadDim_);
        PipeBarrier<PIPE_V>();
        ComputeInverseNorm(queryFloat, tmp0, work, scalar, Q_SUM, Q_SUM_EPS, Q_NORM, Q_INV_NORM);
        Mins(scalar[SCALE_CLAMPED], scaleLocal, clampMax_, 1);
        PipeBarrier<PIPE_V>();
        Exp(scalar[LOGIT_SCALE], scalar[SCALE_CLAMPED], 1);
        PipeBarrier<PIPE_V>();

        for (uint32_t keyStart = 0; keyStart < seqLen_; keyStart += keyTileRows_) {
            const uint32_t rows = keyStart + keyTileRows_ <= seqLen_ ? keyTileRows_ : seqLen_ - keyStart;
            LocalTensor<T> keyLocal = keyQueue_.AllocTensor<T>();
            CopyKeyTile(keyLocal, keyGm_[keyHeadOffset + static_cast<uint64_t>(keyStart) * headDim_], rows);
            keyQueue_.EnQue(keyLocal);
            keyLocal = keyQueue_.DeQue<T>();

            LocalTensor<float> keyFloat = keyFloatBuf_.Get<float>();
            LocalTensor<float> outputFloat = outputFloatBuf_.Get<float>();
            Convert<T>::ToFloat(keyFloat, keyLocal, rows * alignedHeadDim_);
            PipeBarrier<PIPE_V>();
            for (uint32_t j = 0; j < rows; ++j) {
                LocalTensor<float> keyRow = keyFloat[j * alignedHeadDim_];
                ComputeInverseNorm(keyRow, tmp0, work, scalar, K_SUM, K_SUM_EPS, K_NORM, K_INV_NORM);
                Mul(tmp1, queryFloat, keyRow, alignedHeadDim_);
                PipeBarrier<PIPE_V>();
                ReduceSum<float>(scalar[DOT_SUM], tmp1, work, alignedHeadDim_);
                PipeBarrier<PIPE_V>();
                Mul(scalar[SCORE], scalar[DOT_SUM], scalar[Q_INV_NORM], 1);
                PipeBarrier<PIPE_V>();
                Mul(scalar[SCORE], scalar[SCORE], scalar[K_INV_NORM], 1);
                PipeBarrier<PIPE_V>();
                Mul(scalar[SCORE], scalar[SCORE], scalar[LOGIT_SCALE], 1);
                PipeBarrier<PIPE_V>();
                // V->S before scalar reads SCORE; S->V after SetValue so later
                // vector reads of outputFloat see it.
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
                outputFloat.SetValue(j, scalar.GetValue(SCORE));
                SetFlag<HardEvent::S_V>(EVENT_ID0);
                WaitFlag<HardEvent::S_V>(EVENT_ID0);
            }

            LocalTensor<T> outputLocal = outputQueue_.AllocTensor<T>();
            Convert<T>::FromFloat(outputLocal, outputFloat, rows);
            outputQueue_.EnQue(outputLocal);
            outputLocal = outputQueue_.DeQue<T>();
            CopyOutput(outputRowOffset + keyStart, outputLocal, rows);
            outputQueue_.FreeTensor(outputLocal);
            keyQueue_.FreeTensor(keyLocal);
        }

        scaleQueue_.FreeTensor(scaleLocal);
        queryQueue_.FreeTensor(queryLocal);
    }

private:
    TPipe pipe_;
    TQue<TPosition::VECIN, 1> queryQueue_;
    TQue<TPosition::VECIN, 1> keyQueue_;
    TQue<TPosition::VECIN, 1> scaleQueue_;
    TQue<TPosition::VECOUT, 1> outputQueue_;
    TBuf<TPosition::VECCALC> queryFloatBuf_;
    TBuf<TPosition::VECCALC> keyFloatBuf_;
    TBuf<TPosition::VECCALC> tmp0Buf_;
    TBuf<TPosition::VECCALC> tmp1Buf_;
    TBuf<TPosition::VECCALC> reduceWorkBuf_;
    TBuf<TPosition::VECCALC> outputFloatBuf_;
    TBuf<TPosition::VECCALC> scalarBuf_;
    GlobalTensor<T> queryGm_;
    GlobalTensor<T> keyGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<T> outputGm_;
    uint32_t heads_ = 0;
    uint32_t seqLen_ = 0;
    uint32_t headDim_ = 0;
    uint32_t alignedHeadDim_ = 0;
    uint32_t keyTileRows_ = 0;
    uint32_t usedCoreNum_ = 0;
    uint64_t totalQueryRows_ = 0;
    float clampMax_ = 0.0F;
    float eps_ = 0.0F;
};
} // namespace NsScaledCosineAttentionScore
#endif // SCALED_COSINE_ATTENTION_SCORE_IMPL_HPP_
