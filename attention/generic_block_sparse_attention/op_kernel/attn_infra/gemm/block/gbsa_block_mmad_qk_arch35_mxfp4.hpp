/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_BLOCK_MMAD_QK_ARCH35_MXFP4_HPP
#define GBSA_BLOCK_MMAD_QK_ARCH35_MXFP4_HPP

#include "gbsa_block_mmad.hpp"
#include "../../../arch35/gbsa_mxfp4_kernel_utils.hpp"
#include "gbsa_block_mmad_arch35_mxfp4_utils.hpp"
#include "../gbsa_gemm_dispatch_policy.hpp"
#include "../tile_common/gbsa_gemm_tile_copy.hpp"

namespace NpuArch::Gemm::Block {

template <bool Paged, class L1TileShape_, class L0TileShape_, class ElementA_, class ElementB_, class ElementC_,
          class ElementBias_, class TileCopy_, class TileMmad_>
struct BlockMmadTla<MmadAtlasA5GbsaQKMxfp4<Paged>, L1TileShape_, L0TileShape_, ElementA_, ElementB_, ElementC_,
                    ElementBias_, TileCopy_, TileMmad_> {
public:
    using DispatchPolicy = MmadAtlasA5GbsaQKMxfp4<Paged>;
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
    static constexpr uint32_t QUERY_EVENT_BASE = 4;
    static constexpr uint32_t L0_EVENT_BASE = 2;
    // Q stays in L0A[0] for the whole task. M_MTE1 event 6 protects
    // replacement after every preceding MMAD has finished reading that Q.
    // PV uses M_MTE1 events 0/1/4/5; QK's B slots use 2/3.
    static constexpr uint32_t QUERY_L0_EVENT = 6;
    // K slots use events 0/1/5; event 7 protects the remaining groups.
    // Q owns event 4; V owns events 2/3/6 in the same event directions.

    __aicore__ inline explicit BlockMmadTla(Arch::Resource<ArchTag>& resource, const Layout& l1Info,
                                            const Descriptor& descriptorInfo)
        : layout(l1Info),
          descriptor(descriptorInfo),
          HEAD_DIM(l1Info.D),
          QUERY_TILE(l1Info.Q),
          VEC_ROWS(l1Info.R),
          GROUP_ROWS(descriptorInfo.ROWS),
          copyGmToL1K(l1Info, descriptorInfo)
    {
        l1QTensor = resource.l1Buf.template GetBufferByByte<uint8_t>(Layout::Q_OFFSET);
        l1QScaleTensor = resource.l1Buf.template GetBufferByByte<uint8_t>(layout.QSCALE_OFFSET);
        for (uint32_t i = 0; i < L0_STAGES; ++i) {
            l0ATensor[i] =
                resource.l0ABuf.template GetBufferByByte<ElementA>(Layout::QK_A_OFFSET + i * Layout::QK_A_SLOT_BYTES);
            l0BTensor[i] =
                resource.l0BBuf.template GetBufferByByte<ElementB>(Layout::QK_B_OFFSET + i * Layout::QK_B_SLOT_BYTES);
            l0CTensor[i] =
                resource.l0CBuf.template GetBufferByByte<ElementC>(Layout::QK_C_OFFSET + i * Layout::QK_C_SLOT_BYTES);
        }
        for (uint32_t i = 0; i < Layout::K_SLOTS; ++i) {
            l1KTensor[i] = resource.l1Buf.template GetBufferByByte<uint8_t>(layout.K_OFFSET + i * layout.K_SLOT_BYTES);
            l1KScaleTensor[i] =
                resource.l1Buf.template GetBufferByByte<uint8_t>(layout.KSCALE_OFFSET + i * layout.KSCALE_SLOT_BYTES);
        }
        AllocEventID();
    }

    __aicore__ inline void AllocEventID()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(QUERY_EVENT_BASE);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(QUERY_L0_EVENT);
        for (uint32_t slot = 0; slot < Layout::K_SLOTS; ++slot) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(KeyEvent(slot));
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(7);
        for (uint32_t i = 0; i < L0_STAGES; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0_EVENT_BASE + i);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(i);
        }
    }

    template <class TensorQ>
    __aicore__ inline void LoadQuery(TensorQ const& gQTensorTla, const AscendC::GlobalTensor<uint8_t>& gQScale,
                                     uint32_t queryRows)
    {
        ReleaseQuery();
        qsActBaseTile = queryRows;
        qsActBaseTileAlign16 = (queryRows + 15) / 16 * 16;
        const uint32_t event = QUERY_EVENT_BASE;
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(event);
        CopyQGmToL1(gQTensorTla);
        const uint32_t headStride = tla::get<0>(gQTensorTla.stride()) / (HEAD_DIM / 2);
        copyGmToL1QScale(l1QScaleTensor, gQScale, 0, 0, queryRows, HEAD_DIM, headStride);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(event);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(event);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(QUERY_L0_EVENT);
        copyL1ToL0A(l0ATensor[0], l1QTensor, l1QScaleTensor, qsActBaseTileAlign16, HEAD_DIM, qsActBaseTileAlign16);
        // The first K LoadData and its MTE1_M token follow this Q load.
        queryLoaded = true;
    }

    __aicore__ inline void ReleaseQuery()
    {
        if (queryLoaded) {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(QUERY_L0_EVENT);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(QUERY_EVENT_BASE);
            queryLoaded = false;
        }
    }

    __aicore__ static inline uint32_t KeyEvent(uint32_t slot)
    {
        return slot < 2U ? slot : 5U;
    }

    template <class TensorK>
    __aicore__ inline void LoadKey(TensorK const& source, const AscendC::GlobalTensor<uint8_t>& scale,
                                   const AscendC::GlobalTensor<uint64_t>& descriptor, uint32_t groupWords,
                                   uint32_t tokens, uint32_t sequence)
    {
        using namespace GbsaKernelArch35::Mxfp4;
        const uint32_t slot = sequence % Layout::K_SLOTS;
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(KeyEvent(slot));
        LoadKeyGroup(source, scale, descriptor, groupWords, tokens, 0U, slot);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(KeyEvent(slot));
        if (tokens > GROUP_ROWS) {
            // Hide the first tail header's cache miss under the head DMA.
            Tile::PreloadMxfp4DescriptorHeader(descriptor[groupWords]);
        }
    }

    template <class TensorK, class TensorUB>
    __aicore__ inline void operator()(TensorK const& gKTensorTla, const AscendC::GlobalTensor<uint8_t>& gKScale,
                                      const AscendC::GlobalTensor<uint64_t>& descriptor, uint32_t groupWords,
                                      uint32_t validKvs, TensorUB const& ubSTensorTla, uint32_t sequence)
    {
        // Keep the common full-tile bounds and transfer sizes constant.
        // Partial tiles retain the original generic path and event sequence.
        if (layout.KV_ROWS == 512) {
            if (validKvs == 512U) {
                Compute<true>(gKTensorTla, gKScale, descriptor, groupWords, validKvs, ubSTensorTla, sequence);
                return;
            }
        }
        Compute<false>(gKTensorTla, gKScale, descriptor, groupWords, validKvs, ubSTensorTla, sequence);
    }

private:
    template <bool FullTile, class TensorK, class TensorUB>
    __aicore__ inline void Compute(TensorK const& gKTensorTla, const AscendC::GlobalTensor<uint8_t>& gKScale,
                                   const AscendC::GlobalTensor<uint64_t>& descriptor, uint32_t groupWords,
                                   uint32_t validKvs, TensorUB const& ubSTensorTla, uint32_t sequence)
    {
        const uint32_t tokens = FullTile ? 512U : validKvs;
        using namespace GbsaKernelArch35::Mxfp4;
        // Grants are queued after the preceding QK reads and scale waits.
        const uint32_t kL1BufId = sequence % Layout::K_SLOTS;
        const uint32_t groups = (tokens + GROUP_ROWS - 1U) / GROUP_ROWS;
        for (uint32_t pair = 0; pair < (groups + 1U) / 2U; ++pair) {
            CubePublishBoth<PIPE_MTE1>(static_cast<uint32_t>(Flag::KSCALE_FREE_BASE) + pair);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(KeyEvent(kL1BufId));
        for (uint32_t first = 0; first < tokens; first += L0_TILE_N) {
            const uint32_t group = first / GROUP_ROWS;
            const bool firstInGroup = first % GROUP_ROWS == 0U;
            if (firstInGroup) {
                if (group == 1U) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(7);
                    // This is a ready-token acknowledgement, not a slot release.
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(7);
                }
                AscendC::CrossCoreWaitFlag<4, PIPE_MTE1>(static_cast<uint32_t>(Flag::KSCALE_BASE) + group / 2U +
                                                         (group % 2U) * 16U);
            }
            const uint32_t columns = FullTile ? L0_TILE_N : min(L0_TILE_N, (tokens - first + 15U) / 16U * 16U);
            const uint32_t l0Event = L0_EVENT_BASE + qkL0BufId;
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0Event);
            LoadQKToL0(kL1BufId, first, columns);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0Event);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0Event);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(qkL0BufId);
            MatmulQK(columns);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0Event);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(qkL0BufId);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(qkL0BufId);
            auto ubSTile = GetTile(ubSTensorTla, tla::MakeCoord(0, first), tla::MakeShape(VEC_ROWS, columns));
            copyL0CToUB(ubSTile, l0CTensor[qkL0BufId], columns, qsActBaseTile);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(qkL0BufId);
            qkL0BufId ^= 1;
            if (first == 0U && tokens > GROUP_ROWS) {
                // Issue the first L0/MMAD before parsing the rest of the tile.
                // Only the tail-ready token needs acknowledgement; the whole
                // L1 slot remains owned until its final MTE1 read.
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(7);
                LoadKeyGroup<true>(gKTensorTla, gKScale, descriptor, groupWords, tokens, GROUP_ROWS, kL1BufId);
                for (uint32_t begin = 2U * GROUP_ROWS; begin < tokens; begin += GROUP_ROWS) {
                    LoadKeyGroup(gKTensorTla, gKScale, descriptor, groupWords, tokens, begin, kL1BufId);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(7);
            }
        }
        if (groups % 2U != 0U) {
            // An inactive bank still publishes the last pair's notification.
            AscendC::CrossCoreWaitFlag<4, PIPE_MTE1>(static_cast<uint32_t>(Flag::KSCALE_BASE) + groups / 2U + 16U);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(KeyEvent(kL1BufId));
    }

public:
    __aicore__ inline void FreeEventID()
    {
        ReleaseQuery();
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(QUERY_L0_EVENT);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(QUERY_EVENT_BASE);
        for (uint32_t slot = 0; slot < Layout::K_SLOTS; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(KeyEvent(slot));
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(7);
        for (uint32_t i = 0; i < L0_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0_EVENT_BASE + i);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(i);
        }
    }

private:
    template <bool HeaderReady = false, class TensorK>
    __aicore__ inline void LoadKeyGroup(TensorK const& gKTensorTla, const AscendC::GlobalTensor<uint8_t>& gKScale,
                                        const AscendC::GlobalTensor<uint64_t>& descriptor, uint32_t groupWords,
                                        uint32_t validKvs, uint32_t first, uint32_t kL1BufId)
    {
        copyGmToL1K.template CopyKeyAndScale<HeaderReady>(
            l1KTensor[kL1BufId], gKTensorTla.data(), l1KScaleTensor[kL1BufId], gKScale,
            descriptor[first / GROUP_ROWS * groupWords], groupWords, min(GROUP_ROWS, validKvs - first),
            tla::get<0>(gKTensorTla.stride()), first);
    }

    template <class TensorQ>
    __aicore__ inline void CopyQGmToL1(TensorQ const& gQTensorTla)
    {
        auto l1Layout = tla::MakeLayout<uint8_t, LayoutTagL1A>(qsActBaseTileAlign16, HEAD_DIM / 2);
        auto l1Tensor = tla::MakeTensor(l1QTensor, l1Layout, Arch::PositionL1{});
        auto gQTile = GetTile(gQTensorTla, tla::MakeCoord(0, 0), tla::MakeShape(qsActBaseTile, HEAD_DIM / 2));
        auto l1QTile = GetTile(l1Tensor, tla::MakeCoord(0, 0), tla::MakeShape(qsActBaseTile, HEAD_DIM / 2));
        Tile::TileCopyTla<ArchTag, decltype(gQTile), decltype(l1QTile)> copyGmToL1Q;
        copyGmToL1Q(l1QTile, gQTile);
    }

    __aicore__ inline void LoadQKToL0(uint32_t kBufId, uint32_t kvBegin, uint32_t columns)
    {
        copyL1ToL0B(l0BTensor[qkL0BufId], l1KTensor[kBufId], l1KScaleTensor[kBufId], columns, HEAD_DIM, layout.K_ROWS,
                    kvBegin);
    }

    __aicore__ inline void MatmulQK(uint32_t columns)
    {
        auto a = tla::MakeTensor(l0ATensor[0], tla::MakeLayout<ElementA, LayoutTagL0A>(qsActBaseTileAlign16, HEAD_DIM),
                                 Arch::PositionL0A{});
        auto b = tla::MakeTensor(l0BTensor[qkL0BufId], tla::MakeLayout<ElementB, LayoutTagL0B>(HEAD_DIM, columns),
                                 Arch::PositionL0B{});
        auto c = tla::MakeTensor(l0CTensor[qkL0BufId], tla::MakeLayoutL0C(qsActBaseTileAlign16, columns),
                                 Arch::PositionL0C{});
        tileMmad(c, a, b, qsActBaseTile, columns, HEAD_DIM, true);
    }

    AscendC::LocalTensor<uint8_t> l1QTensor, l1QScaleTensor;
    AscendC::LocalTensor<uint8_t> l1KTensor[Layout::K_SLOTS], l1KScaleTensor[Layout::K_SLOTS];
    AscendC::LocalTensor<ElementA> l0ATensor[L0_STAGES];
    AscendC::LocalTensor<ElementB> l0BTensor[L0_STAGES];
    AscendC::LocalTensor<ElementC> l0CTensor[L0_STAGES];
    typename TileCopy::CopyGmToL1Scale copyGmToL1QScale;
    CopyGmToL1B copyGmToL1K;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL0CToUB copyL0CToUB;
    TileMmad tileMmad;
    uint32_t qsActBaseTile = 0;
    uint32_t qsActBaseTileAlign16 = 0;
    uint32_t qkL0BufId = 0;
    bool queryLoaded = false;
};

} // namespace NpuArch::Gemm::Block
#endif // GBSA_BLOCK_MMAD_QK_ARCH35_MXFP4_HPP
