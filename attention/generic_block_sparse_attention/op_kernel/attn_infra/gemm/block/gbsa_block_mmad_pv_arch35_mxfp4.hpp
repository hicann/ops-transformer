/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_BLOCK_MMAD_PV_ARCH35_MXFP4_HPP
#define GBSA_BLOCK_MMAD_PV_ARCH35_MXFP4_HPP

#include "gbsa_block_mmad.hpp"
#include "../../../arch35/gbsa_mxfp4_kernel_utils.hpp"
#include "gbsa_block_mmad_arch35_mxfp4_utils.hpp"
#include "../gbsa_gemm_dispatch_policy.hpp"
#include "../tile_common/gbsa_gemm_tile_copy.hpp"

namespace NpuArch::Gemm::Block {

template <bool Paged, class L1TileShape_, class L0TileShape_, class ElementA_, class ElementB_, class ElementC_,
          class ElementBias_, class TileCopy_, class TileMmad_>
struct BlockMmadTla<MmadAtlasA5GbsaPVMxfp4<Paged>, L1TileShape_, L0TileShape_, ElementA_, ElementB_, ElementC_,
                    ElementBias_, TileCopy_, TileMmad_> {
public:
    using DispatchPolicy = MmadAtlasA5GbsaPVMxfp4<Paged>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using ElementC = ElementC_;
    using ElementStorage = uint8_t;
    using TileCopy = TileCopy_;
    using TileMmad = TileMmad_;
    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
    using CopyGmToL1B = typename TileCopy::CopyGmToL1B;
    using CopyL0CToUB = typename TileCopy::CopyL0CToUB;
    static constexpr uint32_t L0_STAGES = DispatchPolicy::L0_STAGES;
    using Layout = GbsaKernelArch35::Mxfp4::L1TileHelper;
    using Descriptor = GbsaKernelArch35::Mxfp4::DescriptorTileHelper<Paged>;
    Layout layout;
    Descriptor descriptor;
    uint32_t HEAD_DIM, QUERY_TILE, VEC_ROWS, GROUP_ROWS;
    static constexpr uint32_t L0_TILE_N = 128U;
    static constexpr uint32_t L0_A_EVENT = 0;
    static constexpr uint32_t L0_B_EVENT = 4;
    static constexpr uint32_t L0_C_EVENT = 2;
    __aicore__ static inline uint32_t ValueEvent(uint32_t slot)
    {
        return slot < 2U ? 2U + slot : 6U;
    }

    __aicore__ inline explicit BlockMmadTla(Arch::Resource<ArchTag>& resource, const Layout& l1Info,
                                            const Descriptor& descriptorInfo)
        : layout(l1Info),
          descriptor(descriptorInfo),
          HEAD_DIM(l1Info.D),
          QUERY_TILE(l1Info.Q),
          VEC_ROWS(l1Info.R),
          GROUP_ROWS(descriptorInfo.ROWS),
          copyGmToL1V(l1Info, descriptorInfo)
    {
        for (uint32_t i = 0; i < Layout::V_SLOTS; ++i) {
            l1VTensor[i] = resource.l1Buf.template GetBufferByByte<uint8_t>(layout.V_OFFSET + i * layout.V_SLOT_BYTES);
        }
        l1VScaleTensor = resource.l1Buf.template GetBufferByByte<uint8_t>(layout.VSCALE_OFFSET);
        for (uint32_t i = 0; i < Layout::P_SLOTS; ++i) {
            l1PTensor[i] = resource.l1Buf.template GetBufferByByte<uint8_t>(layout.P_OFFSET + i * layout.P_SLOT_BYTES);
            l1PScaleTensor[i] =
                resource.l1Buf.template GetBufferByByte<uint8_t>(layout.PSCALE_OFFSET + i * layout.PSCALE_SLOT_BYTES);
        }
        for (uint32_t i = 0; i < L0_STAGES; ++i) {
            l0ATensor[i] =
                resource.l0ABuf.template GetBufferByByte<ElementA>(Layout::PV_A_OFFSET + i * Layout::PV_A_SLOT_BYTES);
            l0BTensor[i] =
                resource.l0BBuf.template GetBufferByByte<ElementB>(Layout::PV_B_OFFSET + i * Layout::PV_B_SLOT_BYTES);
            l0CTensor[i] =
                resource.l0CBuf.template GetBufferByByte<ElementC>(Layout::PV_C_OFFSET + i * Layout::PV_C_SLOT_BYTES);
        }
        AllocEventID();
        // Unused tail scales must remain finite when the final KV tile is padded.
        FillMxfp4L1(l1VScaleTensor, layout.VSCALE_SLOT_BYTES, 0x7f7f);
        // First InitializeValueScale joins this MTE2 fill before any PV.
    }

    __aicore__ inline void AllocEventID()
    {
        for (uint32_t slot = 0; slot < Layout::V_SLOTS; ++slot) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ValueEvent(slot));
        }
        for (uint32_t slot = 0; slot < L0_STAGES; ++slot) {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0_A_EVENT + slot);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0_B_EVENT + slot);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(L0_C_EVENT + slot);
        }
    }

    __aicore__ inline void InitializeValueScale(const AscendC::GlobalTensor<uint8_t>& source, uint64_t offset,
                                                uint32_t sourcePairs, uint32_t pairStride)
    {
        if (!hasValueScale) {
            LoadValueScale(source, offset, sourcePairs, pairStride);
        }
    }

    // External scales already repeat the same channel value along S. All V
    // data slots share this prefix, retained across tasks of the same head.
    __aicore__ inline void LoadValueScale(const AscendC::GlobalTensor<uint8_t>& source, uint64_t offset,
                                          uint32_t sourcePairs, uint32_t pairStride)
    {
        if (hasValueScale && offset == valueScaleHead) {
            return;
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(4);
        const uint32_t pairs = min(sourcePairs, layout.KV_ROWS / 64);
        auto gScale = source[offset].template ReinterpretCast<AscendC::fp8_e8m0_t>();
        auto gScaleTensor =
            tla::MakeTensor(gScale,
                            tla::MakeLayout(tla::MakeShape(tla::MakeShape(tla::Int<2>{}, pairs), HEAD_DIM),
                                            tla::MakeStride(tla::MakeStride(tla::Int<1>{}, pairStride), tla::Int<2>{})),
                            Arch::PositionGM{});
        auto l1ScaleLayout =
            tla::MakeMxScaleLayout<AscendC::fp8_e8m0_t, layout::nN, true>(layout.KV_ROWS / 32, HEAD_DIM);
        // The first page is full whenever selected tokens span multiple pages.
        // Repeat its externally supplied pairs; never read beyond that page.
        const uint32_t copies = DispatchPolicy::PAGED_CACHE_FLAG ? layout.KV_ROWS / 64 / pairs : 1U;
        for (uint32_t copy = 0; copy < copies; ++copy) {
            auto l1ScaleTensor =
                tla::MakeTensor(l1VScaleTensor[copy * pairs * 32].template ReinterpretCast<AscendC::fp8_e8m0_t>(),
                                l1ScaleLayout, Arch::PositionL1{});
            Tile::TileCopyTla<ArchTag, decltype(gScaleTensor), decltype(l1ScaleTensor)> copyScale;
            copyScale(l1ScaleTensor, gScaleTensor);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(4);
        valueScaleHead = offset;
        hasValueScale = true;
    }

    // Prefetch V with QK; the delayed PV consumes a different slot.
    template <class TensorV>
    __aicore__ inline void LoadValue(TensorV const& source, const AscendC::GlobalTensor<uint64_t>& descriptor,
                                     uint32_t groupWords, uint32_t tokens, uint32_t sequence)
    {
        const uint32_t slot = sequence % Layout::V_SLOTS;
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ValueEvent(slot));
        copyGmToL1V.template operator()<true>(l1VTensor[slot], source.data(), descriptor, groupWords, tokens,
                                              tla::get<0>(source.stride()), 0U);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(ValueEvent(slot));
    }

    // Reuse P across the two N slices of D192/256. Each slice has its own
    // FP32 L0C accumulator; P and V keep independent 16KiB double buffers.
    template <class TensorV, class TensorUB>
    __aicore__ inline void operator()(TensorV const& gVTensorTla, const AscendC::GlobalTensor<uint64_t>& descriptor,
                                      uint32_t groupWords, uint32_t validKvs, uint32_t tileId, uint32_t queryRows,
                                      TensorUB const& ubOTensorTla, uint32_t pvFlag)
    {
        // Keep the common full-tile bounds and transfer sizes constant.
        // Partial tiles retain the original generic path and event sequence.
        if (layout.KV_ROWS == 512) {
            if (validKvs == 512U) {
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
        const uint32_t tokens = FullTile ? 512U : validKvs;
        const uint32_t reduction = FullTile ? 512U : (tokens + 63) / 64 * 64;
        const uint32_t rows0 = (queryRows + 1) / 2;
        const uint32_t bankRows0 = (rows0 + 15) / 16 * 16;
        const uint32_t alignedRows = 2U * bankRows0; // Equal bank spans for split-M output.
        const uint32_t vSlot = tileId % Layout::V_SLOTS;
        const uint32_t pSlot = tileId % Layout::P_SLOTS;
        const uint32_t N_SLICES = (HEAD_DIM + layout.PV_COLUMNS - 1) / layout.PV_COLUMNS;
        const uint32_t firstC = tileId * N_SLICES % L0_STAGES;
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(ValueEvent(vSlot));
        for (uint32_t kvBegin = 0; kvBegin < reduction; kvBegin += Layout::PV_REDUCTION) {
            const uint32_t chunk = FullTile ? Layout::PV_REDUCTION : min(Layout::PV_REDUCTION, reduction - kvBegin);
            const uint32_t aEvent = L0_A_EVENT + pL0BufId;
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(aEvent);
            copyL1ToL0A(l0ATensor[pL0BufId], l1PTensor[pSlot], l1PScaleTensor[pSlot], alignedRows, chunk, alignedRows,
                        0, layout.KV_ROWS, kvBegin);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(aEvent);
            for (uint32_t n = 0; n < N_SLICES; ++n) {
                const uint32_t channel = n * layout.PV_COLUMNS;
                const uint32_t columns = min(layout.PV_COLUMNS, HEAD_DIM - channel);
                const uint32_t cSlot = (firstC + n) % L0_STAGES;
                const uint32_t bEvent = L0_B_EVENT + vL0BufId;
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(bEvent);
                copyL1ToL0B(l0BTensor[vL0BufId], l1VTensor[vSlot], l1VScaleTensor, columns, chunk, columns,
                            layout.KV_ROWS, kvBegin, channel);
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
        GbsaKernelArch35::Mxfp4::CubeWaitBoth<PIPE_FIX>(pvFlag);
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
        for (uint32_t slot = 0; slot < Layout::V_SLOTS; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ValueEvent(slot));
        }
        for (uint32_t slot = 0; slot < L0_STAGES; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0_A_EVENT + slot);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0_B_EVENT + slot);
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

    AscendC::LocalTensor<uint8_t> l1VTensor[Layout::V_SLOTS], l1VScaleTensor;
    uint64_t valueScaleHead = 0;
    bool hasValueScale = false;
    AscendC::LocalTensor<uint8_t> l1PTensor[Layout::P_SLOTS], l1PScaleTensor[Layout::P_SLOTS];
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
#endif // GBSA_BLOCK_MMAD_PV_ARCH35_MXFP4_HPP
