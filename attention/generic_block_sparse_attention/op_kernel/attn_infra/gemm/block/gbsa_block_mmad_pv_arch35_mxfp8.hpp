/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_BLOCK_MMAD_PV_ARCH35_MXFP8_HPP
#define GBSA_BLOCK_MMAD_PV_ARCH35_MXFP8_HPP

#include "gbsa_block_mmad.hpp"
#include "../../arch/gbsa_cross_core_sync.hpp"
#include "../gbsa_gemm_dispatch_policy.hpp"
#include "../tile_common/gbsa_mxfp8_tile.hpp"

namespace NpuArch::Gemm::Block {

struct PvMxfp8FullBuffers {
    uint32_t KV_ROWS;
    uint32_t PV_A_OFFSET;
    uint32_t PV_A_SLOT_BYTES;
    uint32_t PV_B_OFFSET;
    uint32_t PV_B_SLOT_BYTES;
    uint32_t PV_COLUMNS;
    uint32_t PV_C_OFFSET;
    uint32_t PV_C_SLOT_BYTES;
    uint32_t PV_REDUCTION;
    uint32_t P_OFFSET;
    uint32_t P_SLOTS;
    uint32_t P_SLOT_BYTES;
    uint32_t V_OFFSET;
    uint32_t V_SLOTS;
    uint32_t V_SLOT_BYTES;
    uint32_t L0_B_EVENT_BASE;
};

template <bool Paged, const PvMxfp8FullBuffers* BufferConfig = nullptr>
struct BlockPVMxfp8Full {
public:
    struct DispatchPolicy {
        using ArchTag = Arch::AtlasA5;
        static constexpr bool PAGED_CACHE_FLAG = Paged;
        static constexpr uint32_t L0_STAGES = 2;
    };
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementA = fp8_e4m3fn_t;
    using ElementB = fp8_e4m3fn_t;
    using ElementC = float;
    using ElementStorage = uint8_t;
    using TileCopy = Tile::PackedTileCopyMxfp8Full<Paged, true>;
    using TileMmad = Tile::TileMmadTla<ArchTag, ElementA, layout::zN>;
    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
    using CopyGmToL1B = typename TileCopy::CopyGmToL1B;
    using CopyL0CToUB = typename TileCopy::CopyL0CToUB;
    static constexpr uint32_t L0_STAGES = DispatchPolicy::L0_STAGES;
    using Buffers = PvMxfp8FullBuffers;
    template <const Buffers* Binding>
    using WithBuffers = BlockPVMxfp8Full<Paged, Binding>;
    // The Kernel owns these constants; binding them in the type keeps
    // slot arithmetic and Cube tile loops constant after outlining.
    static constexpr Buffers layout = BufferConfig == nullptr ? Buffers{} : *BufferConfig;
    static constexpr uint32_t MAX_V_SLOTS = 2U;
    static constexpr uint32_t MAX_P_SLOTS = 3U;
    uint32_t HEAD_DIM, QUERY_TILE, VEC_ROWS, GROUP_ROWS;
    static constexpr uint32_t L0_TILE_N = 128U;
    static constexpr uint32_t L0_A_EVENT = 0;
    static constexpr uint32_t L0_C_EVENT = 2;
    __aicore__ static inline uint32_t ValueEvent(uint32_t slot)
    {
        return slot < 2U ? 2U + slot : 6U;
    }

    __aicore__ inline explicit BlockPVMxfp8Full(Arch::Resource<ArchTag>& resource, uint32_t dim, uint32_t queryTile,
                                                uint32_t vecRows, uint32_t groupRows)
        : HEAD_DIM(dim),
          QUERY_TILE(queryTile),
          VEC_ROWS(vecRows),
          GROUP_ROWS(groupRows),
          copyGmToL1V(dim, layout.KV_ROWS, groupRows)
    {
        static_assert(BufferConfig != nullptr, "Cube buffers must be bound by the Kernel");
        for (uint32_t i = 0; i < layout.V_SLOTS; ++i) {
            l1VTensor[i] = resource.l1Buf.template GetBufferByByte<uint8_t>(layout.V_OFFSET + i * layout.V_SLOT_BYTES);
        }
        for (uint32_t i = 0; i < layout.P_SLOTS; ++i) {
            l1PTensor[i] = resource.l1Buf.template GetBufferByByte<uint8_t>(layout.P_OFFSET + i * layout.P_SLOT_BYTES);
        }
        for (uint32_t i = 0; i < L0_STAGES; ++i) {
            l0ATensor[i] =
                resource.l0ABuf.template GetBufferByByte<ElementA>(layout.PV_A_OFFSET + i * layout.PV_A_SLOT_BYTES);
            l0BTensor[i] =
                resource.l0BBuf.template GetBufferByByte<ElementB>(layout.PV_B_OFFSET + i * layout.PV_B_SLOT_BYTES);
            l0CTensor[i] =
                resource.l0CBuf.template GetBufferByByte<ElementC>(layout.PV_C_OFFSET + i * layout.PV_C_SLOT_BYTES);
        }
        AllocEventID();
    }

    __aicore__ inline void AllocEventID()
    {
        for (uint32_t slot = 0; slot < layout.V_SLOTS; ++slot) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ValueEvent(slot));
        }
        for (uint32_t slot = 0; slot < L0_STAGES; ++slot) {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0_A_EVENT + slot);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(L0_C_EVENT + slot);
        }
    }

    // Prefetch V with QK; the delayed PV consumes a different slot.
    template <class TensorV>
    __aicore__ inline void LoadValue(TensorV const& source, const AscendC::GlobalTensor<uint64_t>& descriptor,
                                     uint32_t groupWords, uint32_t tokens, uint32_t sequence)
    {
        const uint32_t slot = sequence % layout.V_SLOTS;
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ValueEvent(slot));
        if (tokens % 64U != 0U) {
            // The PV reduction rounds up to 64 tokens. Zero P cannot mask a
            // stale FP8 NaN in V, so initialize the full fixed-pitch NZ tile
            // before the descriptors overwrite its valid rows.
            AscendC::InitConstValueParams<uint16_t> zero(1, layout.KV_ROWS * HEAD_DIM / 32U, 0, 0);
            AscendC::Fill(l1VTensor[slot].template ReinterpretCast<uint16_t>(), zero);
            AscendC::PipeBarrier<PIPE_MTE2>();
        }
        copyGmToL1V.template operator()<true>(l1VTensor[slot], source.data(), descriptor, groupWords, tokens,
                                              tla::get<0>(source.stride()), 0U);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(ValueEvent(slot));
    }

    // Reuse P across the two N slices of D192/256. Each slice has its own
    // FP32 L0C accumulator. P owns two 16KiB A slots; V shares the
    // two 32KiB B slots with QK through the same M_MTE1 events.
    template <class TensorV, class TensorUB>
    __aicore__ inline void operator()(TensorV const& gVTensorTla, const AscendC::GlobalTensor<uint64_t>& descriptor,
                                      uint32_t groupWords, uint32_t validKvs, uint32_t tileId, uint32_t queryRows,
                                      TensorUB const& ubOTensorTla, uint32_t pvFlag)
    {
        // Keep the common full-tile bounds and transfer sizes constant.
        // Partial tiles retain the original generic path and event sequence.
        if constexpr (layout.KV_ROWS == 512U || layout.KV_ROWS == 256U) {
            if (validKvs == layout.KV_ROWS) {
                Compute<true>(gVTensorTla, descriptor, groupWords, validKvs, tileId, queryRows, ubOTensorTla, pvFlag);
                return;
            }
        }
        Compute<false>(gVTensorTla, descriptor, groupWords, validKvs, tileId, queryRows, ubOTensorTla, pvFlag);
    }

private:
    template <bool FullTile, class TensorV, class TensorUB>
    __aicore__ inline void Compute(TensorV const& gVTensorTla, const AscendC::GlobalTensor<uint64_t>& descriptor,
                                   uint32_t groupWords, uint32_t validKvs, uint32_t tileId, uint32_t queryRows,
                                   TensorUB const& ubOTensorTla, uint32_t pvFlag)
    {
        const uint32_t tokens = FullTile ? layout.KV_ROWS : validKvs;
        const uint32_t reduction = FullTile ? layout.KV_ROWS : (tokens + 63) / 64 * 64;
        const uint32_t rows0 = (queryRows + 1) / 2;
        const uint32_t bankRows0 = (rows0 + 15) / 16 * 16;
        const uint32_t alignedRows = 2U * bankRows0; // Equal bank spans for split-M output.
        const uint32_t vSlot = tileId % layout.V_SLOTS;
        const uint32_t pSlot = tileId % layout.P_SLOTS;
        const uint32_t N_SLICES = (HEAD_DIM + layout.PV_COLUMNS - 1) / layout.PV_COLUMNS;
        const uint32_t firstC = tileId * N_SLICES % L0_STAGES;
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(ValueEvent(vSlot));
        for (uint32_t kvBegin = 0; kvBegin < reduction; kvBegin += layout.PV_REDUCTION) {
            const uint32_t chunk = FullTile ? layout.PV_REDUCTION : min(layout.PV_REDUCTION, reduction - kvBegin);
            const uint32_t aEvent = L0_A_EVENT + pL0BufId;
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(aEvent);
            copyL1ToL0A(l0ATensor[pL0BufId], l1PTensor[pSlot], alignedRows, chunk, alignedRows, 0, layout.KV_ROWS,
                        kvBegin);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(aEvent);
            for (uint32_t n = 0; n < N_SLICES; ++n) {
                const uint32_t channel = n * layout.PV_COLUMNS;
                const uint32_t columns = min(layout.PV_COLUMNS, HEAD_DIM - channel);
                const uint32_t cSlot = (firstC + n) % L0_STAGES;
                const uint32_t bEvent = layout.L0_B_EVENT_BASE + vL0BufId;
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(bEvent);
                copyL1ToL0B(l0BTensor[vL0BufId], l1VTensor[vSlot], columns, chunk, columns, layout.KV_ROWS, kvBegin,
                            channel);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(bEvent);
                if (n == 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(aEvent);
                }
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(bEvent);
                if (kvBegin == 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(L0_C_EVENT + cSlot);
                }
                MatmulPV(alignedRows, alignedRows, columns, chunk, cSlot, kvBegin == 0);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(bEvent);
                vL0BufId ^= 1;
            }
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(aEvent);
            pL0BufId ^= 1;
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ValueEvent(vSlot));
        Arch::CubeWaitBoth<PIPE_FIX>(pvFlag);
        for (uint32_t n = 0; n < N_SLICES; ++n) {
            const uint32_t channel = n * layout.PV_COLUMNS;
            const uint32_t columns = min(layout.PV_COLUMNS, HEAD_DIM - channel);
            const uint32_t cSlot = (firstC + n) % L0_STAGES;
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(L0_C_EVENT + cSlot);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(L0_C_EVENT + cSlot);
            auto ubOTile = GetTile(ubOTensorTla, tla::MakeCoord(0, channel), tla::MakeShape(VEC_ROWS, columns));
            copyL0CToUB(ubOTile, l0CTensor[cSlot], columns, queryRows);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(L0_C_EVENT + cSlot);
        }
    }

public:
    __aicore__ inline void FreeEventID()
    {
        for (uint32_t slot = 0; slot < layout.V_SLOTS; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ValueEvent(slot));
        }
        for (uint32_t slot = 0; slot < L0_STAGES; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0_A_EVENT + slot);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(L0_C_EVENT + slot);
        }
    }

private:
    __aicore__ inline void MatmulPV(uint32_t queryRows, uint32_t rows, uint32_t columns, uint32_t reduction,
                                    uint32_t cSlot, bool init)
    {
        auto a = tla::MakeTensor(l0ATensor[pL0BufId], tla::MakeLayout<ElementA, LayoutTagL0A>(rows, reduction),
                                 Arch::PositionL0A{});
        auto b = tla::MakeTensor(l0BTensor[vL0BufId], tla::MakeLayout<ElementB, LayoutTagL0B>(reduction, columns),
                                 Arch::PositionL0B{});
        auto c = tla::MakeTensor(l0CTensor[cSlot], tla::MakeLayoutL0C(rows, columns), Arch::PositionL0C{});
        tileMmad(c, a, b, queryRows, columns, reduction, init);
    }

    AscendC::LocalTensor<uint8_t> l1VTensor[MAX_V_SLOTS];
    AscendC::LocalTensor<uint8_t> l1PTensor[MAX_P_SLOTS];
    AscendC::LocalTensor<ElementA> l0ATensor[L0_STAGES];
    AscendC::LocalTensor<ElementB> l0BTensor[L0_STAGES];
    AscendC::LocalTensor<ElementC> l0CTensor[L0_STAGES];
    CopyGmToL1B copyGmToL1V;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL0CToUB copyL0CToUB;
    TileMmad tileMmad;
    uint32_t pL0BufId = 0;
    uint32_t vL0BufId = 0;
};

} // namespace NpuArch::Gemm::Block
#endif // GBSA_BLOCK_MMAD_PV_ARCH35_MXFP8_HPP
