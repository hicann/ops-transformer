/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef BLOCK_EPILOGUE_RESCALE_O_FULL_QUANT_HPP
#define BLOCK_EPILOGUE_RESCALE_O_FULL_QUANT_HPP

#include "block_epilogue_full_quant.hpp"

namespace GbsaKernelArch22::Int8 {

template <class ElementO, bool singleRowTile, bool compactUb>
struct GbsaFullQuantVectorArch22<ElementO, singleRowTile, compactUb>::EpilogueRescaleO {
    static constexpr uint32_t OUT_ELEMENTS = BATCH_ROWS * FullQuantLayout::HEAD_DIM;
    static constexpr uint32_t UPDATE_ELEMENTS = 8192U;
    static_assert(singleRowTile || UPDATE_ELEMENTS * sizeof(float) <= BufferLayout::O_TEMP);
    static_assert(singleRowTile || BufferLayout::O_TEMP + UPDATE_ELEMENTS * sizeof(float) <= BufferLayout::P_INT32);

    GbsaFullQuantVectorArch22& state;

    __aicore__ inline void Update(AscendC::GlobalTensor<int32_t> output, uint32_t round)
    {
        if constexpr (singleRowTile) {
            if (state.totalRows_ == 0U) {
                return;
            }
            state.rowOffset_ = 0U;
            UpdateBatch(output, round);
            return;
        }
        // Match FP16's 8192-element O tile. S and O are used in separate
        // phases, so the wide path reuses [0,32KiB) for input and
        // [32KiB,64KiB) for the converted O tile.
        const uint32_t updateRows = UPDATE_ELEMENTS / state.alignedDim_ / 8U * 8U;
        for (state.rowOffset_ = 0U; state.rowOffset_ < state.totalRows_; state.rowOffset_ += updateRows) {
            state.rows_ =
                state.totalRows_ - state.rowOffset_ < updateRows ? state.totalRows_ - state.rowOffset_ : updateRows;
            UpdateBatch(output, round);
        }
    }

    __aicore__ inline void Store(AscendC::GlobalTensor<ElementO> output, uint64_t headStride = 0U)
    {
        if constexpr (singleRowTile) {
            if (state.totalRows_ == 0U) {
                return;
            }
            state.rowOffset_ = 0U;
            StoreBatch(output, headStride);
            return;
        }
        for (state.rowOffset_ = 0U; state.rowOffset_ < state.totalRows_; state.rowOffset_ += BATCH_ROWS) {
            state.rows_ =
                state.totalRows_ - state.rowOffset_ < BATCH_ROWS ? state.totalRows_ - state.rowOffset_ : BATCH_ROWS;
            StoreBatch(output, headStride);
        }
    }

    __aicore__ inline void StoreLse(AscendC::GlobalTensor<float> output, uint64_t headStride = 1U)
    {
        if constexpr (singleRowTile) {
            if (state.totalRows_ == 0U) {
                return;
            }
            state.rowOffset_ = 0U;
            StoreLseBatch(output, headStride);
            return;
        }
        for (state.rowOffset_ = 0U; state.rowOffset_ < state.totalRows_; state.rowOffset_ += BATCH_ROWS) {
            state.rows_ =
                state.totalRows_ - state.rowOffset_ < BATCH_ROWS ? state.totalRows_ - state.rowOffset_ : BATCH_ROWS;
            StoreLseBatch(output, headStride);
        }
    }

    __aicore__ inline void StorePartial(AscendC::GlobalTensor<float> output, AscendC::GlobalTensor<float> lse,
                                        uint32_t partialTask, uint32_t groupSize, uint32_t lseSubStride)
    {
        const uint32_t stride = (state.headDim_ + 15U) / 16U * 16U;
        for (state.rowOffset_ = 0U; state.rowOffset_ < state.totalRows_; state.rowOffset_ += BATCH_ROWS) {
            state.rows_ =
                state.totalRows_ - state.rowOffset_ < BATCH_ROWS ? state.totalRows_ - state.rowOffset_ : BATCH_ROWS;
            auto acc =
                state.template Ub<float>(BufferLayout::O_ACC + state.rowOffset_ * state.alignedDim_ * sizeof(float));
            NormalizeBatch(acc);
            state.template Sync<AscendC::HardEvent::V_MTE3>();
            const uint64_t offset =
                (static_cast<uint64_t>(partialTask) * groupSize + state.rowStart_ + state.rowOffset_) * stride;
            AscendC::DataCopyPad(output[offset], acc,
                                 AscendC::DataCopyExtParams(state.rows_, stride * sizeof(float),
                                                            (state.alignedDim_ - stride) / 8U, 0, 0));
            state.template Sync<AscendC::HardEvent::MTE3_V>();
            // LSE uses a padded slice per AIV, matching the existing FP16 combine contract.
            const uint64_t lseBase =
                (static_cast<uint64_t>(partialTask) * 2U + AscendC::GetSubBlockIdx()) * lseSubStride - state.rowStart_;
            StoreLseBatch(lse[lseBase]);
        }
    }

    __aicore__ inline void UpdateBatch(AscendC::GlobalTensor<int32_t> output, uint32_t round)
    {
        if (state.totalRows_ == 0U) {
            return;
        }
        const uint32_t inputOffset = singleRowTile && !compactUb ? (round % 2) * OUT_ELEMENTS * sizeof(int32_t) : 0U;
        auto input = state.template Ub<int32_t>(BufferLayout::O_INPUT + inputOffset);
        auto work = state.template Ub<float>(BufferLayout::O_TEMP);
        auto acc = state.template Ub<float>(BufferLayout::O_ACC + state.rowOffset_ * state.alignedDim_ * sizeof(float));
        // Match PV's compact aligned-D row layout; slot capacity remains bounded by max D.
        if (!singleRowTile || state.rowOffset_ != 0U || round >= (compactUb ? 1U : 2U)) {
            // Row batches and round t+2 reuse this DMA slot. PIPE_V alone does not
            // prevent MTE2 from overwriting input before the preceding Cast.
            state.template Sync<AscendC::HardEvent::V_MTE2>();
        }
        AscendC::DataCopy(input, output[(state.rowStart_ + state.rowOffset_) * state.alignedDim_],
                          state.rows_ * state.alignedDim_);
        state.template Sync<AscendC::HardEvent::MTE2_V>();
        AscendC::Cast(work, input, AscendC::RoundMode::CAST_NONE, state.rows_ * state.alignedDim_);
        AscendC::PipeBarrier<PIPE_V>();
        auto alpha = state.template Ub<float>(BufferLayout::ALPHA +
                                              (round % FullQuantLayout::STAGES) * BufferLayout::ALPHA_SLOT_BYTES +
                                              state.rowOffset_ * STAT_STRIDE * sizeof(float));
        // FP16's rescale epilogue repeats each column vector over valid rows.
        // Alpha is already broadcast to one 32-byte block per row by Softmax.
        const uint8_t rowStride = state.alignedDim_ / 8U;
        state.FullMask();
        for (uint32_t column = 0; column < state.alignedDim_; column += 64U) {
            const uint32_t remaining = state.alignedDim_ - column;
            if (remaining < 64U) {
                state.RowMask(remaining);
            }
            AscendC::Mul<float, false>(work[column], work[column],
                                       state.template Ub<float>(BufferLayout::V_SCALE)[column], uint64_t{0},
                                       state.rows_, AscendC::BinaryRepeatParams(1, 1, 1, rowStride, rowStride, 0));
            AscendC::Mul<float, false>(acc[column], acc[column], alpha, uint64_t{0}, state.rows_,
                                       AscendC::BinaryRepeatParams(1, 1, 0, rowStride, rowStride, 1));
        }
        state.FullMask();
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(acc, acc, work, state.rows_ * state.alignedDim_);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StoreBatch(AscendC::GlobalTensor<ElementO> output, uint64_t headStride = 0U)
    {
        auto acc = state.template Ub<float>(BufferLayout::O_ACC + state.rowOffset_ * state.alignedDim_ * sizeof(float));
        auto dst = state.template Ub<ElementO>(BufferLayout::O_TEMP);
        NormalizeBatch(acc);
        if (state.rows_ != 0) {
            AscendC::Cast(dst, acc, AscendC::RoundMode::CAST_RINT, state.rows_ * state.alignedDim_);
            state.template Sync<AscendC::HardEvent::V_MTE3>();
            const uint32_t copyAligned = (state.headDim_ + 15U) / 16U * 16U;
            AscendC::DataCopyExtParams copy{static_cast<uint16_t>(state.rows_),
                                            state.headDim_ * uint32_t{sizeof(ElementO)},
                                            (state.alignedDim_ - copyAligned) / 16U, 0, 0};
            if (headStride == 0U || headStride == state.headDim_) {
                AscendC::DataCopyPad(output[(state.rowStart_ + state.rowOffset_) * state.headDim_], dst, copy);
            } else {
                copy.blockCount = 1U;
                copy.srcStride = 0U;
                for (uint32_t row = 0U; row < state.rows_; ++row) {
                    AscendC::DataCopyPad(
                        output[static_cast<uint64_t>(state.rowStart_ + state.rowOffset_ + row) * headStride],
                        dst[row * state.alignedDim_], copy);
                }
            }
            state.template Sync<AscendC::HardEvent::MTE3_V>();
        }
    }

    // Reuse output scratch after Store has completed its MTE3 transfer.
    // Match FP16: LSE is [Tq, Nq, 1] or [B, Nq, Sq, 1]; empty rows use -FLT_MAX.
    __aicore__ inline void StoreLseBatch(AscendC::GlobalTensor<float> output, uint32_t headStride = 1U)
    {
        if (state.rows_ == 0U) {
            return;
        }
        auto result = state.template Ub<float>(BufferLayout::O_TEMP);
        auto maximum = result[BATCH_ROWS];
        if constexpr (singleRowTile) {
            state.template Sync<AscendC::HardEvent::V_S>();
            for (uint32_t row = 0; row < BATCH_ROWS; ++row) {
                const float sum = state.template Ub<float>(BufferLayout::RUNNING_SUM).GetValue(state.rowOffset_ + row);
                result.SetValue(row, sum > 0.0f ? sum : 1.0f);
                maximum.SetValue(
                    row, sum > 0.0f ?
                             state.template Ub<float>(BufferLayout::RUNNING_MAX).GetValue(state.rowOffset_ + row) :
                             -3.402823466e38f);
            }
            state.template Sync<AscendC::HardEvent::S_V>();
        } else {
            AscendC::Maxs(result,
                          state.template Ub<float>(BufferLayout::RUNNING_SUM + state.rowOffset_ * sizeof(float)), 1.0f,
                          BATCH_ROWS);
            AscendC::PipeBarrier<PIPE_V>();
            maximum = state.template Ub<float>(BufferLayout::RUNNING_MAX + state.rowOffset_ * sizeof(float));
        }
        AscendC::Ln(result, result, BATCH_ROWS);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(result, result, maximum, BATCH_ROWS);
        if (headStride == 1U) {
            state.template Sync<AscendC::HardEvent::V_MTE3>();
            AscendC::DataCopyPad(output[state.rowStart_ + state.rowOffset_], result,
                                 AscendC::DataCopyExtParams(1, state.rows_ * uint32_t{sizeof(float)}, 0, 0, 0));
        } else {
            // A separate aligned source block per row is required for strided scalar output.
            auto packed = result[BATCH_ROWS];
            if constexpr (singleRowTile) {
                state.template Sync<AscendC::HardEvent::V_S>();
                for (uint32_t row = 0U; row < state.rows_; ++row) {
                    packed.SetValue(row * STAT_STRIDE, result.GetValue(row));
                }
                state.template Sync<AscendC::HardEvent::S_MTE3>();
            } else {
                AscendC::Brcb(packed.template ReinterpretCast<uint32_t>(), result.template ReinterpretCast<uint32_t>(),
                              BATCH_ROWS / 8U, AscendC::BrcbRepeatParams(1, 8));
                state.template Sync<AscendC::HardEvent::V_MTE3>();
            }
            const AscendC::DataCopyExtParams copy{1, sizeof(float), 0, 0, 0};
            for (uint32_t row = 0U; row < state.rows_; ++row) {
                AscendC::DataCopyPad(
                    output[static_cast<uint64_t>(state.rowStart_ + state.rowOffset_ + row) * headStride],
                    packed[row * STAT_STRIDE], copy);
            }
        }
        state.template Sync<AscendC::HardEvent::MTE3_V>();
    }

    __aicore__ inline void NormalizeBatch(AscendC::LocalTensor<float> acc)
    {
        if constexpr (singleRowTile) {
            state.template Sync<AscendC::HardEvent::V_S>();
            for (uint32_t row = 0U; row < state.rows_; ++row) {
                const float sum = state.template Ub<float>(BufferLayout::RUNNING_SUM).GetValue(state.rowOffset_ + row);
                AscendC::Muls(acc[row * state.alignedDim_], acc[row * state.alignedDim_],
                              sum > 0.0f ? 1.0f / sum : 0.0f, state.alignedDim_);
            }
        } else {
            // A nonempty online-softmax row has sum >= 1. Empty rows have
            // zero O and zero sum, so denominator 1 preserves their zero O.
            auto safe = state.template Ub<float>(BufferLayout::REDUCE_SUM);
            auto broadcast = state.template Ub<float>(BufferLayout::P_INT32);
            AscendC::Maxs(safe, state.template Ub<float>(BufferLayout::RUNNING_SUM + state.rowOffset_ * sizeof(float)),
                          1.0f, BATCH_ROWS);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Brcb(broadcast.template ReinterpretCast<uint32_t>(), safe.template ReinterpretCast<uint32_t>(),
                          BATCH_ROWS / 8U, AscendC::BrcbRepeatParams(1, 8));
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t column = 0U; column < state.alignedDim_; column += 64U) {
                const uint32_t mask = state.alignedDim_ - column < 64U ? state.alignedDim_ - column : 64U;
                AscendC::Div(acc[column], acc[column], broadcast, mask, state.rows_,
                             AscendC::BinaryRepeatParams(1, 1, 0, state.alignedDim_ / 8U, state.alignedDim_ / 8U, 1));
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }
};

} // namespace GbsaKernelArch22::Int8

#endif
