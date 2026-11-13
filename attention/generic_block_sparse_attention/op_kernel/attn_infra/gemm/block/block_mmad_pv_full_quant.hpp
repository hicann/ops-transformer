/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef BLOCK_MMAD_PV_FULL_QUANT_HPP
#define BLOCK_MMAD_PV_FULL_QUANT_HPP

#include "block_mmad_full_quant.hpp"
#include "../../../gbsa_sparse_pair_copy.hpp"

namespace GbsaKernelArch22::Int8 {

template <bool singleRowTile>
struct GbsaFullQuantCubeArch22<singleRowTile>::BlockMmadPV {
    GbsaFullQuantCubeArch22& state;

    __aicore__ inline void PrefetchP(AscendC::GlobalTensor<int8_t> probability, uint32_t stage, const Tile& tile)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(4 + stage);
        state.copyA_(state.L1(L1P() + stage * PBytes()), probability,
                     layout::zN::MakeLayout<int8_t>(state.RowsRound(), TileColumns(tile)),
                     layout::RowMajor(state.rows_, TileColumns(tile)));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(4 + stage);
    }

    // The first half is independent of P and is queued before waiting for P-ready.
    // The second half is queued after starting the first PV MMAD.
    template <class PlanType>
    __aicore__ inline void PrefetchV(AscendC::GlobalTensor<int8_t> value, const Tile& tile, uint32_t heads,
                                     const PlanType& plan, uint32_t half)
    {
        auto vLayout = layout::zN::MakeLayout<int8_t>(FullQuantLayout::L1_KV_ROWS, state.KvDim());
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(2 + half);
        state.ClearTailSlot(L1V() + half * FullQuantLayout::KV_SLOT_BYTES, tile, half);
        for (uint32_t page = 0; page < 2; ++page) {
            const Page& entry = tile.pages[half * 2 + page];
            if (entry.validTokens == 0) {
                continue;
            }
            auto dst = state.L1(L1V() + half * FullQuantLayout::KV_SLOT_BYTES)[vLayout.GetOffset(
                MakeCoord(page * FullQuantLayout::PAGE_TOKENS, 0U))];
            CopyValue(dst, value, vLayout, half * 2 + page, plan, heads);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(2 + half);
    }

    template <class PlanType>
    __aicore__ inline void Pv(AscendC::GlobalTensor<int8_t> value, AscendC::GlobalTensor<int32_t> output,
                              const Tile& tile, uint32_t heads, uint32_t stage, const PlanType& plan)
    {
        auto vLayout = layout::zN::MakeLayout<int8_t>(FullQuantLayout::L1_KV_ROWS, state.KvDim());
        const uint32_t reduction = state.pvReduction_;
        const uint32_t reductionLoops = FullQuantLayout::PV_REDUCTION / reduction;
        const uint32_t rowTile = singleRowTile ? 16U : state.pvRows_;
        for (uint32_t row = 0; row < (singleRowTile ? 1U : state.rows_); row += rowTile) {
            const uint32_t actualRows =
                singleRowTile ? state.rows_ : (state.rows_ - row < rowTile ? state.rows_ - row : rowTile);
            const uint32_t matrixRows = singleRowTile ? 16U : (actualRows + 15U) / 16U * 16U;
            LoadP(stage, row, matrixRows, (singleRowTile || row + actualRows == state.rows_), reduction,
                  TileColumns(tile));
            const uint32_t cSlot = singleRowTile ? 2U + state.oCursor_++ % 2U : state.cCursor_++ % 2U;
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(cSlot);
            auto result = state.L0C(cSlot * state.CBytes());
            for (uint32_t half = 0;
                 half < (TileColumns(tile) + FullQuantLayout::L1_KV_ROWS - 1U) / FullQuantLayout::L1_KV_ROWS; ++half) {
                for (uint32_t part = 0; part * reduction < HalfColumns(tile, half); ++part) {
                    if (row == 0U && part == 0U) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(2 + half);
                    }
                    const uint32_t count = HalfColumns(tile, half) - part * reduction < reduction ?
                                               HalfColumns(tile, half) - part * reduction :
                                               reduction;
                    const uint32_t bSlot = state.bCursor_++ % 2;
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(bSlot);
                    auto src = state.L1(L1V() + half * FullQuantLayout::KV_SLOT_BYTES)[vLayout.GetOffset(
                        MakeCoord(part * reduction, state.headOffset_))];
                    state.loadV_(state.L0B(bSlot), src, layout::nZ::MakeLayout<int8_t>(count, state.QkDim()), vLayout);
                    if ((singleRowTile || row + actualRows == state.rows_) &&
                        (part + 1U) * reduction >= HalfColumns(tile, half)) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(2 + half);
                    }
                    state.template Sync<AscendC::HardEvent::MTE1_M>();
                    auto probability = state.L0A(
                        state.QBytes() + (half * FullQuantLayout::PV_REDUCTION + part * reduction) * matrixRows);
                    state.Mmad(result, probability, state.L0B(bSlot), half == 0U && part == 0U, count, state.QkDim(),
                               matrixRows);
                    if (row == 0U && part == 0U && half == 0U && TileColumns(tile) > FullQuantLayout::L1_KV_ROWS) {
                        PrefetchV(value, tile, heads, plan, 1);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(bSlot);
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(3);
            state.template Sync<AscendC::HardEvent::M_FIX>();
            state.store_(output[row * state.QkDim()], result, layout::RowMajor(actualRows, state.QkDim()),
                         layout::zN::MakeLayoutInL0C(MakeCoord(matrixRows, state.QkDim())));
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(cSlot);
        }
    }

    template <class TensorLayout, class PlanType>
    __aicore__ inline void CopyValue(AscendC::LocalTensor<int8_t> dst, AscendC::GlobalTensor<int8_t> src,
                                     const TensorLayout& layout, uint32_t chunk, const PlanType& plan, uint32_t heads)
    {
        for (uint32_t i = plan.begin[chunk]; i < plan.end[chunk]; ++i) {
            const auto& r = plan.records[i];
            if (state.packedKv_) {
                PackedKv::Emit<false>(dst, src, layout, r.dstRow, r.source, r.rows, r.ndNum, r.stride, state.HeadDim(),
                                      state.packedBlockY_);
            } else {
                GbsaSparsePairCopy::Emit<false>(dst, src, layout, r.dstRow, r.source - state.headOffset_, r.rows,
                                                r.ndNum, r.stride, state.copyDim_,
                                                state.kvHeadMajor_ ? state.HeadDim() : heads * state.HeadDim());
            }
        }
    }

    __aicore__ inline void LoadP(uint32_t stage, uint32_t row, uint32_t matrixRows, bool last, uint32_t reduction,
                                 uint32_t columns)
    {
        if (row == 0U) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(4 + stage);
        }
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(3);
        auto srcLayout = layout::zN::MakeLayout<int8_t>(state.RowsRound(), columns);
        const uint32_t packReduction = singleRowTile ? FullQuantLayout::PV_REDUCTION : reduction;
        for (uint32_t token = 0; token < columns; token += packReduction) {
            const uint32_t count = columns - token < packReduction ? columns - token : packReduction;
            state.loadA_(state.L0A(state.QBytes() + token * matrixRows),
                         state.L1(L1P() + stage * PBytes())[srcLayout.GetOffset(MakeCoord(row, token))],
                         layout::zZ::MakeLayout<int8_t>(matrixRows, count), srcLayout);
        }
        if (last) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(4 + stage);
        }
    }

private:
    __aicore__ inline uint32_t L1V() const
    {
        return state.QBytes() + FullQuantLayout::DOUBLE_BUFFER * FullQuantLayout::KV_SLOT_BYTES;
    }

    __aicore__ inline uint32_t L1P() const
    {
        return state.QBytes() + 2U * FullQuantLayout::DOUBLE_BUFFER * FullQuantLayout::KV_SLOT_BYTES;
    }

    __aicore__ inline uint32_t PBytes() const
    {
        return singleRowTile ? 16U * FullQuantLayout::KV_TILE : state.pBytes_;
    }
};

} // namespace GbsaKernelArch22::Int8

#endif
