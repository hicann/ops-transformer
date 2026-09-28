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
 * \file attention_worker_combine_mxfp_dequant.h
 * \brief Ascend950 kernel for BS/K/H tiled MXFP8 and packed MXFP4 dequantization and accumulation.
 */

#ifndef ATTENTION_WORKER_COMBINE_MXFP_DEQUANT_H_
#define ATTENTION_WORKER_COMBINE_MXFP_DEQUANT_H_

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "attention_worker_combine_tiling_struct.h"
#include "../attention_worker_combine_common_utils.h"

namespace AttentionWorkerCombineRegbase {
using namespace AscendC;
using optiling::AttentionWorkerCombineRegbaseTilingData;

constexpr int64_t MXFP_SCALE_GROUP_SIZE = 32;
constexpr int64_t MXFP_SCALE_ALIGN = 2;
constexpr int64_t UB_BLOCK_BYTES = 32;
constexpr int64_t FP4_ELEMENTS_PER_BYTE = 2;

// E8M0 has no zero encoding: 0 is 2^-127, 255 is NaN.
__aicore__ inline float DecodeMxfpScale(uint8_t exponent)
{
    union {
        uint32_t bits;
        float value;
    } scale;
    scale.bits = exponent == 0 ? 0x00400000U : (exponent == 255 ? 0x7fc00000U : static_cast<uint32_t>(exponent) << 23);
    return scale.value;
}

enum class MxfpSplit {
    BS,
    K,
    H
};

template <typename TokenType, bool PackedFp4 = false, MxfpSplit Split = MxfpSplit::BS>
class KernelAttentionWorkerCombineMxfpDequant {
public:
    __aicore__ inline void Init(GM_ADDR scheduleContext, GM_ADDR expertScales, GM_ADDR layerId, GM_ADDR y,
                                GM_ADDR nextLayerId, GM_ADDR workspace,
                                const AttentionWorkerCombineRegbaseTilingData *tiling, TPipe *pipe)
    {
        t_ = tiling;
        context32_.SetGlobalBuffer((__gm__ uint32_t *)scheduleContext);
        context64_.SetGlobalBuffer((__gm__ uint64_t *)scheduleContext);
        microBatch_ = context32_.GetValue(GET_OFFSET_B32(ScheduleContext, attention.micro_batch_id));
        if (t_->needSchedule == 1) {
            const uint32_t count = context32_.GetValue(GET_OFFSET_B32(ScheduleContext, common.micro_batch_num));
            microBatch_ = (microBatch_ + 1) % count;
        }
        const uint64_t data = context64_.GetValue(GET_OFFSET_B64(ScheduleContext, attention.token_data_buf));
        const uint64_t info = context64_.GetValue(GET_OFFSET_B64(ScheduleContext, attention.token_info_buf));
        tokenRowBytes_ = PackedFp4 ? (t_->H + FP4_ELEMENTS_PER_BYTE - 1) / FP4_ELEMENTS_PER_BYTE : t_->H;
        tokens_.SetGlobalBuffer((__gm__ uint8_t *)(data + microBatch_ * t_->BS * t_->K * tokenRowBytes_));
        flags_.SetGlobalBuffer((__gm__ int32_t *)(info + microBatch_ * t_->BS * t_->K * sizeof(int32_t)));
        scales_.SetGlobalBuffer((__gm__ uint8_t *)expertScales);
        output_.SetGlobalBuffer((__gm__ bfloat16_t *)y);
        layer_.SetGlobalBuffer((__gm__ int32_t *)layerId);
        nextLayer_.SetGlobalBuffer((__gm__ int32_t *)nextLayerId);
        scaleStride_ = Ops::Base::CeilAlign<int64_t>((t_->H + MXFP_SCALE_GROUP_SIZE - 1) / MXFP_SCALE_GROUP_SIZE,
                                                     MXFP_SCALE_ALIGN);
        inputRowStride_ = Ops::Base::CeilAlign<int64_t>(
            PackedFp4 ? (t_->HSplitFactor + FP4_ELEMENTS_PER_BYTE - 1) / FP4_ELEMENTS_PER_BYTE : t_->HSplitFactor,
            UB_BLOCK_BYTES);
        pipe->InitBuffer(inputQueue_, 1, inputRowStride_ * t_->KSplitFactor);
        pipe->InitBuffer(outputQueue_, 1, t_->HSplitFactor * sizeof(bfloat16_t));
        pipe->InitBuffer(valueBuffer_, t_->HSplitFactor * sizeof(float));
        pipe->InitBuffer(sumBuffer_, t_->HSplitFactor * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const int64_t core = GetBlockIdx();
        if (core == 0) {
            nextLayer_.SetValue(0, layer_.GetValue(0) + 1);
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_ALL>(nextLayer_);
        }
        // BS/K split tokens across cores. H may additionally split a token when scheduling is off.
        const int64_t hCores = Split == MxfpSplit::H ? t_->HSplitCoreNum : 1;
        const int64_t bsCore = core / hCores;
        const int64_t hCore = core % hCores;
        const int64_t bsStart = bsCore * t_->mainCoreBsLoopNum;
        const int64_t bsCount = bsCore == t_->BsSplitCoreNum - 1 ? t_->tailCoreBsLoopNum : t_->mainCoreBsLoopNum;
        for (int64_t r = bsStart; r < bsStart + bsCount; ++r) {
            if (t_->needSchedule == 1) {
                WaitForRow(r);
            }
            if constexpr (Split == MxfpSplit::H) {
                const int64_t hStart = hCore * t_->mainCoreHLoopNum * t_->HSplitFactor;
                const int64_t hLoops = hCore == hCores - 1 ? t_->tailCoreHLoopNum : t_->mainCoreHLoopNum;
                for (int64_t loop = 0; loop < hLoops; ++loop) {
                    const int64_t h = hStart + loop * t_->HSplitFactor;
                    const int64_t count = t_->H - h < t_->HSplitFactor ? t_->H - h : t_->HSplitFactor;
                    ComputeTile(r, h, count);
                }
            } else {
                ComputeTile(r, 0, t_->H);
            }
            if (t_->needSchedule == 1) {
                ClearRow(r);
            }
        }
        if (t_->needSchedule == 1) {
            PipeBarrier<PIPE_ALL>();
            SyncAll();
            if (core == 0) {
                context32_.SetValue(GET_OFFSET_B32(ScheduleContext, attention.micro_batch_id), microBatch_);
                DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_ALL>(
                    context32_[GET_OFFSET_B32(ScheduleContext, attention.micro_batch_id)]);
            }
        }
    }

private:
    __aicore__ inline void WaitForRow(int64_t row)
    {
        bool ready = false;
        while (!ready) {
            ready = true;
            // Reuse token input storage; scan flags in chunks that fit the queue.
            const int64_t capacity = inputRowStride_ * t_->KSplitFactor / sizeof(int32_t);
            for (int64_t start = 0; start < t_->K && ready; start += capacity) {
                const int64_t count = t_->K - start < capacity ? t_->K - start : capacity;
                auto flags = inputQueue_.AllocTensor<int32_t>();
                DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
                DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
                DataCopyPad(flags, flags_[row * t_->K + start], copy, pad);
                inputQueue_.EnQue(flags);
                flags = inputQueue_.DeQue<int32_t>();
                for (int64_t k = 0; k < count; ++k) {
                    ready = ready && flags.GetValue(k) == 1;
                }
                inputQueue_.FreeTensor(flags);
            }
        }
    }

    __aicore__ inline void ClearRow(int64_t row)
    {
        // Complete token reads and result writes before releasing this row to the producer.
        PipeBarrier<PIPE_ALL>();
        const int64_t capacity = t_->HSplitFactor * sizeof(bfloat16_t) / sizeof(int32_t);
        for (int64_t start = 0; start < t_->K; start += capacity) {
            const int64_t count = t_->K - start < capacity ? t_->K - start : capacity;
            auto flags = outputQueue_.AllocTensor<int32_t>();
            Duplicate(flags, int32_t(0), count);
            outputQueue_.EnQue(flags);
            flags = outputQueue_.DeQue<int32_t>();
            DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(flags_[row * t_->K + start], flags, copy);
            outputQueue_.FreeTensor(flags);
        }
    }

    __aicore__ inline void DecodeFp4(LocalTensor<float> values, LocalTensor<uint8_t> input, int64_t count)
    {
        // DeQue completes MTE2 -> V. Finish previous vector reads before scalar UB access.
        const event_t vToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(vToS);
        WaitFlag<HardEvent::V_S>(vToS);
        // Each row is packed separately: low nibble = even h, high nibble = odd h.
        // All E2M1 values are exact in FP32; exponent zero includes +/-0 and +/-0.5.
        for (int64_t i = 0; i < count; ++i) {
            const uint8_t byte = input.GetValue(i / FP4_ELEMENTS_PER_BYTE);
            const uint8_t code = (byte >> ((i % FP4_ELEMENTS_PER_BYTE) * 4)) & 0x0f;
            const uint8_t magnitude = code & 7;
            float value = 0.0f;
            if (magnitude < 2) {
                value = static_cast<float>(magnitude) * 0.5f;
            } else {
                value = (1.0f + static_cast<float>(magnitude & 1) * 0.5f) *
                        static_cast<float>(1U << ((magnitude >> 1) - 1));
            }
            values.SetValue(i, (code & 8) ? -value : value);
        }
        const event_t sToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(sToV);
        WaitFlag<HardEvent::S_V>(sToV);
    }

    __aicore__ inline void ComputeTile(int64_t row, int64_t h, int64_t count)
    {
        auto sum = sumBuffer_.Get<float>();
        auto values = valueBuffer_.Get<float>();
        Duplicate(sum, 0.0f, count);
        PipeBarrier<PIPE_V>();
        const int64_t kTile = Split == MxfpSplit::BS ? t_->K : (Split == MxfpSplit::K ? t_->KSplitFactor : 1);
        for (int64_t kStart = 0; kStart < t_->K; kStart += kTile) {
            const int64_t rows = t_->K - kStart < kTile ? t_->K - kStart : kTile;
            auto input = inputQueue_.AllocTensor<uint8_t>();
            // BS/K: one 2D copy loads complete, contiguous GM rows into aligned UB rows.
            // H: rows=1, so a partial row needs no inter-row stride.
            DataCopyExtParams copy{
                static_cast<uint16_t>(rows),
                static_cast<uint32_t>(PackedFp4 ? (count + FP4_ELEMENTS_PER_BYTE - 1) / FP4_ELEMENTS_PER_BYTE : count),
                0, 0, 0};
            DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
            DataCopyPad(input,
                        tokens_[(row * t_->K + kStart) * tokenRowBytes_ + (PackedFp4 ? h / FP4_ELEMENTS_PER_BYTE : h)],
                        copy, pad);
            inputQueue_.EnQue(input);
            input = inputQueue_.DeQue<uint8_t>();
            for (int64_t i = 0; i < rows; ++i) {
                auto rowInput = input[i * inputRowStride_];
                if constexpr (PackedFp4) {
                    DecodeFp4(values, rowInput, count);
                } else {
                    Cast(values, rowInput.template ReinterpretCast<TokenType>(), RoundMode::CAST_NONE, count);
                }
                PipeBarrier<PIPE_V>();
                for (int64_t group = 0; group < count; group += MXFP_SCALE_GROUP_SIZE) {
                    const int64_t valid = count - group < MXFP_SCALE_GROUP_SIZE ? count - group : MXFP_SCALE_GROUP_SIZE;
                    const float scale = DecodeMxfpScale(scales_.GetValue((row * t_->K + kStart + i) * scaleStride_ +
                                                                         (h + group) / MXFP_SCALE_GROUP_SIZE));
                    Muls(values[group], values[group], scale, valid);
                }
                PipeBarrier<PIPE_V>();
                // Preserve expert order across K tiles: no partial-sum regrouping or intermediate BF16 cast.
                Add(sum, sum, values, count);
                PipeBarrier<PIPE_V>();
            }
            inputQueue_.FreeTensor(input);
        }
        auto out = outputQueue_.AllocTensor<bfloat16_t>();
        Cast(out, sum, RoundMode::CAST_RINT, count);
        outputQueue_.EnQue(out);
        out = outputQueue_.DeQue<bfloat16_t>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPad(output_[row * t_->H + h], out, copy);
        outputQueue_.FreeTensor(out);
    }

    const AttentionWorkerCombineRegbaseTilingData *t_ = nullptr;
    uint32_t microBatch_ = 0;
    int64_t scaleStride_ = 0;
    int64_t tokenRowBytes_ = 0;
    int64_t inputRowStride_ = 0;
    GlobalTensor<uint32_t> context32_;
    GlobalTensor<uint64_t> context64_;
    GlobalTensor<uint8_t> tokens_;
    GlobalTensor<uint8_t> scales_;
    GlobalTensor<int32_t> flags_, layer_, nextLayer_;
    GlobalTensor<bfloat16_t> output_;
    TQue<QuePosition::VECIN, 1> inputQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;
    TBuf<TPosition::VECCALC> valueBuffer_, sumBuffer_;
};

} // namespace AttentionWorkerCombineRegbase
#endif
