/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef BLOCK_MMAD_QK_FULL_QUANT_HPP
#define BLOCK_MMAD_QK_FULL_QUANT_HPP

#include "block_mmad_full_quant.hpp"
#include "../../../gbsa_sparse_pair_copy.hpp"

namespace GbsaKernelArch22::Int8 {

template <bool singleRowTile>
struct GbsaFullQuantCubeArch22<singleRowTile>::BlockMmadQK {
    GbsaFullQuantCubeArch22& state;

    __aicore__ inline void LoadQuery(AscendC::GlobalTensor<int8_t> query, uint32_t rows, uint32_t head = 0,
                                     uint64_t headStride = 0)
    {
        state.rows_ = rows;
        state.rowRound_ = singleRowTile ? 16U : (rows + 15U) / 16U * 16U;
        state.qBytes_ = state.rowRound_ * FullQuantLayout::HEAD_DIM;
        state.pBytes_ = state.rowRound_ * FullQuantLayout::KV_TILE;
        state.cBytes_ = singleRowTile ? 16U * FullQuantLayout::QK_COLUMNS * sizeof(int32_t) :
                                        FullQuantLayout::L0C_SHARED_SLOT_BYTES;
        state.qkColumns_ = FullQuantLayout::QK_COLUMNS;
        while (state.rowRound_ * state.qkColumns_ * sizeof(int32_t) > state.cBytes_) {
            state.qkColumns_ /= 2U;
        }
        state.qkReduction_ = FullQuantLayout::L0B_SLOT_BYTES / state.qkColumns_;
        state.pvReduction_ = state.QkDim() * FullQuantLayout::PV_REDUCTION <= FullQuantLayout::L0B_SLOT_BYTES ?
                                 FullQuantLayout::PV_REDUCTION :
                                 FullQuantLayout::PV_REDUCTION / 2U;
        state.pvRows_ = state.cBytes_ / (state.QkDim() * sizeof(int32_t)) / 16U * 16U;
        state.pvRows_ = state.pvRows_ < FullQuantLayout::PV_ROWS ? state.pvRows_ : FullQuantLayout::PV_ROWS;
        const auto window = state.kvHeadMajor_ ? KvCopyWindow{state.HeadDim(), 0U} :
                                                 MakeKvCopyWindow(state.HeadDim(), state.kvHeads_, head);
        state.copyDim_ = window.channels;
        state.headOffset_ = window.headOffset;
        auto qL1 = state.L1(FullQuantLayout::L1_Q);
        auto qLayout = layout::zN::MakeLayout<int8_t>(state.RowsRound(), state.QkDim());
        // The previous task is drained before this call; Q in L1 has no remaining reader.
        if ((state.headDim_ % 32U) != 0U) {
            AscendC::InitConstValueParams<int32_t> clear(1, state.RowsRound() * state.QkDim() / 32U, 0, 0);
            AscendC::InitConstValue(qL1.template ReinterpretCast<int32_t>(), clear);
            AscendC::PipeBarrier<PIPE_MTE2>();
        }
        state.copyA_(qL1, query, qLayout,
                     layout::RowMajor(rows, state.HeadDim(), headStride == 0U ? state.HeadDim() : headStride));
        state.template Sync<AscendC::HardEvent::MTE2_MTE1>();
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2);
        const uint32_t reductionTile = singleRowTile ? state.QkDim() : state.qkReduction_;
        for (uint32_t start = 0; start < state.QkDim(); start += reductionTile) {
            const uint32_t count = state.QkDim() - start < reductionTile ? state.QkDim() - start : reductionTile;
            // Each MMAD K tile needs its own compact zZ row-fractal stride.
            state.loadA_(state.L0A(FullQuantLayout::L0A_Q + start * state.RowsRound()),
                         qL1[qLayout.GetOffset(MakeCoord(0U, start))],
                         layout::zZ::MakeLayout<int8_t>(state.RowsRound(), count), qLayout);
        }
        state.template Sync<AscendC::HardEvent::MTE1_M>();
    }

    __aicore__ inline void ReleaseQuery()
    {
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2);
    }

    template <bool isPaged, bool hasResidualBlock, class PlanType>
    __aicore__ inline void Qk(AscendC::GlobalTensor<int8_t> key, AscendC::GlobalTensor<int32_t> scores,
                              const Tile& tile, uint64_t kPageStride, uint64_t vPageStride, uint32_t heads,
                              uint32_t head, uint64_t kvStart,
                              const SparseSource<isPaged, hasResidualBlock, AscendC::GlobalTensor<int32_t>>& source,
                              PlanType& plan)
    {
        auto kLayout = layout::nZ::MakeLayout<int8_t>(state.KvDim(), FullQuantLayout::L1_KV_ROWS);
        // Each L1 slot gathers two compact 128-token chunks from selected sparse blocks.
        for (uint32_t half = 0;
             half < (TileColumns(tile) + FullQuantLayout::L1_KV_ROWS - 1U) / FullQuantLayout::L1_KV_ROWS; ++half) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(half);
            state.ClearTailSlot(L1K() + half * FullQuantLayout::KV_SLOT_BYTES, tile, half);
            for (uint32_t page = 0; page < 2; ++page) {
                const Page& entry = tile.pages[half * 2 + page];
                plan.begin[half * 2 + page] = plan.end[half * 2 + page] =
                    (half * 2 + page) * PlanType::RECORDS_PER_CHUNK;
                if (entry.validTokens == 0) {
                    continue;
                }
                auto dst = state.L1(L1K() + half * FullQuantLayout::KV_SLOT_BYTES)[kLayout.GetOffset(
                    MakeCoord(0U, page * FullQuantLayout::PAGE_TOKENS))];
                CopyKeyAndRecord(dst, key, kLayout, entry, half * 2 + page, plan, kPageStride, vPageStride, heads, head,
                                 kvStart, source);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(half);
        }
        // Keep all GQA rows together, shrinking N only when required by
        // the 32 KiB L0C slot. K is still loaded from GM once per KV tile.
        const uint32_t columns = QkColumns();
        const uint32_t reductionTile = singleRowTile ? FullQuantLayout::QK_REDUCTION : state.qkReduction_;
        const uint32_t reductionLoops = (state.QkDim() + reductionTile - 1U) / reductionTile;
        for (uint32_t half = 0;
             half < (TileColumns(tile) + FullQuantLayout::L1_KV_ROWS - 1U) / FullQuantLayout::L1_KV_ROWS; ++half) {
            for (uint32_t column = 0; column < HalfColumns(tile, half); column += columns) {
                const uint32_t tileColumns =
                    HalfColumns(tile, half) - column < columns ? HalfColumns(tile, half) - column : columns;
                const uint32_t cSlot = singleRowTile ? state.sCursor_++ % 2U : state.cCursor_++ % 2U;
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(cSlot);
                auto result = state.L0C(cSlot * state.CBytes());
                for (uint32_t reduction = 0; reduction < reductionLoops; ++reduction) {
                    const uint32_t start = reduction * reductionTile;
                    const uint32_t remaining = state.QkDim() - start;
                    const uint32_t count = remaining < reductionTile ? remaining : reductionTile;
                    const uint32_t bSlot = state.bCursor_++ % 2;
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(bSlot);
                    if (reduction == 0U && column == 0U) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(half);
                    }
                    auto src = state.L1(L1K() + half * FullQuantLayout::KV_SLOT_BYTES)[kLayout.GetOffset(
                        MakeCoord(state.headOffset_ + start, column))];
                    state.loadK_(state.L0B(bSlot), src, layout::nZ::MakeLayout<int8_t>(count, tileColumns), kLayout);
                    if (reduction + 1U == reductionLoops && column + tileColumns == HalfColumns(tile, half)) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(half);
                    }
                    state.template Sync<AscendC::HardEvent::MTE1_M>();
                    state.Mmad(result, state.L0A(FullQuantLayout::L0A_Q + start * state.RowsRound()), state.L0B(bSlot),
                               reduction == 0U, count, tileColumns, state.RowsRound());
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(bSlot);
                }
                state.template Sync<AscendC::HardEvent::M_FIX>();
                state.store_(scores[half * FullQuantLayout::L1_KV_ROWS + column], result,
                             layout::RowMajor(state.rows_, tileColumns, TileColumns(tile)),
                             layout::zN::MakeLayoutInL0C(MakeCoord(state.RowsRound(), tileColumns)));
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(cSlot);
            }
        }
    }

    template <class TensorLayout, class PlanType>
    struct KeyEmitter {
        AscendC::LocalTensor<int8_t>& dst;
        AscendC::GlobalTensor<int8_t>& src;
        const TensorLayout& layout;
        PlanType& plan;
        uint32_t chunk;
        uint32_t heads;
        uint32_t headDim;
        uint32_t copyDim;
        uint32_t headOffset;
        bool headMajor;
        bool packed;
        uint32_t packedBlockY;

        __aicore__ inline void operator()(const KvCopyRecord& r)
        {
            if (packed) {
                PackedKv::Emit<true>(dst, src, layout, r.dstRow, r.key, r.rows, r.ndNum, r.keyStride, headDim,
                                     packedBlockY);
            } else {
                GbsaSparsePairCopy::Emit<true>(dst, src, layout, r.dstRow, r.key - headOffset, r.rows, r.ndNum,
                                               r.keyStride, copyDim, headMajor ? headDim : heads * headDim);
            }
            plan.records[plan.end[chunk]++] = {r.value, r.valueStride, r.rows, r.dstRow, r.ndNum};
        }
    };

    template <class TensorLayout, bool isPaged, bool hasResidualBlock, class PlanType>
    __aicore__ inline void CopyKeyAndRecord(
        AscendC::LocalTensor<int8_t> dst, AscendC::GlobalTensor<int8_t> src, const TensorLayout& layout,
        const Page& part, uint32_t chunk, PlanType& plan, uint64_t kPageStride, uint64_t vPageStride, uint32_t heads,
        uint32_t head, uint64_t kvStart,
        const SparseSource<isPaged, hasResidualBlock, AscendC::GlobalTensor<int32_t>>& source)
    {
        source.ForEachKvRecord(part, kPageStride, vPageStride, heads, head, kvStart,
                               KeyEmitter<TensorLayout, PlanType>{dst, src, layout, plan, chunk, heads, state.HeadDim(),
                                                                  state.copyDim_, state.headOffset_, state.kvHeadMajor_,
                                                                  state.packedKv_, state.packedBlockY_});
    }

private:
    __aicore__ inline uint32_t L1K() const
    {
        return state.QBytes();
    }

    __aicore__ inline uint32_t QkColumns() const
    {
        return singleRowTile ? FullQuantLayout::QK_COLUMNS : state.qkColumns_;
    }
};

} // namespace GbsaKernelArch22::Int8

#endif
