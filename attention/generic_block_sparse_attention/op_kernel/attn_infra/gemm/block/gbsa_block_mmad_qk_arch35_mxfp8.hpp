/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_BLOCK_MMAD_QK_ARCH35_MXFP8_HPP
#define GBSA_BLOCK_MMAD_QK_ARCH35_MXFP8_HPP

#include "gbsa_block_mmad.hpp"
#include "../../arch/gbsa_cross_core_sync.hpp"
#include "../gbsa_gemm_dispatch_policy.hpp"
#include "../tile_common/gbsa_mxfp8_tile.hpp"

namespace NpuArch::Gemm::Block {

struct QkMxfp8FullBuffers {
    uint32_t KSCALE_OFFSET;
    uint32_t KSCALE_SLOTS;
    uint32_t KSCALE_SLOT_BYTES;
    uint32_t KV_ROWS;
    uint32_t K_OFFSET;
    uint32_t K_ROWS;
    uint32_t K_SLOTS;
    uint32_t K_SLOT_BYTES;
    uint32_t QK_A_OFFSET;
    uint32_t QK_A_SLOT_BYTES;
    uint32_t QK_B_OFFSET;
    uint32_t QK_B_SLOT_BYTES;
    uint32_t QK_C_OFFSET;
    uint32_t QK_C_SLOT_BYTES;
    uint32_t QSCALE_OFFSET;
    uint32_t Q_OFFSET;
    uint32_t L0_B_EVENT_BASE;
    uint32_t SCALE_READY_FLAG;
    uint32_t SCALE_FREE_FLAG;
};

template <bool Paged, const QkMxfp8FullBuffers* BufferConfig = nullptr>
struct BlockQKMxfp8Full {
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
    using TileCopy = Tile::PackedTileCopyMxfp8Full<Paged, false>;
    using TileMmad = Tile::GbsaMxMmad;
    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
    using CopyGmToL1B = typename TileCopy::CopyGmToL1B;
    using CopyL0CToUB = typename TileCopy::CopyL0CToUB;
    static constexpr uint32_t L0_STAGES = DispatchPolicy::L0_STAGES;
    using Buffers = QkMxfp8FullBuffers;
    template <const Buffers* Binding>
    using WithBuffers = BlockQKMxfp8Full<Paged, Binding>;
    // The Kernel owns these constants; binding them in the type keeps
    // slot arithmetic and Cube tile loops constant after outlining.
    static constexpr Buffers layout = BufferConfig == nullptr ? Buffers{} : *BufferConfig;
    static constexpr uint32_t MAX_K_SLOTS = 1U;
    static constexpr uint32_t MAX_KSCALE_SLOTS = 3U;
    uint32_t HEAD_DIM, QUERY_TILE, VEC_ROWS, GROUP_ROWS;
    static constexpr uint32_t L0_TILE_N = 128U;
    static constexpr uint32_t QUERY_EVENT_BASE = 4;
    // Q stays in L0A[0] for the whole task. M_MTE1 event 6 protects
    // replacement after every preceding MMAD has finished reading that Q.
    // PV A slots use M_MTE1 events 0/1; QK and PV share B slots 2/3.
    static constexpr uint32_t QUERY_L0_EVENT = 6;
    // The K data slot uses event 0; event 7 protects the remaining groups.
    // Q owns event 4; V owns events 2/3 in the same event directions.

    __aicore__ inline explicit BlockQKMxfp8Full(Arch::Resource<ArchTag>& resource, uint32_t dim, uint32_t queryTile,
                                                uint32_t vecRows, uint32_t groupRows)
        : HEAD_DIM(dim),
          QUERY_TILE(queryTile),
          VEC_ROWS(vecRows),
          GROUP_ROWS(groupRows),
          copyGmToL1K(dim, layout.KV_ROWS, groupRows)
    {
        static_assert(BufferConfig != nullptr, "Cube buffers must be bound by the Kernel");
        l1QTensor = resource.l1Buf.template GetBufferByByte<uint8_t>(layout.Q_OFFSET);
        l1QScaleTensor = resource.l1Buf.template GetBufferByByte<uint8_t>(layout.QSCALE_OFFSET);
        for (uint32_t i = 0; i < L0_STAGES; ++i) {
            l0ATensor[i] =
                resource.l0ABuf.template GetBufferByByte<ElementA>(layout.QK_A_OFFSET + i * layout.QK_A_SLOT_BYTES);
            l0BTensor[i] =
                resource.l0BBuf.template GetBufferByByte<ElementB>(layout.QK_B_OFFSET + i * layout.QK_B_SLOT_BYTES);
            l0CTensor[i] =
                resource.l0CBuf.template GetBufferByByte<ElementC>(layout.QK_C_OFFSET + i * layout.QK_C_SLOT_BYTES);
        }
        for (uint32_t i = 0; i < layout.K_SLOTS; ++i) {
            l1KTensor[i] = resource.l1Buf.template GetBufferByByte<uint8_t>(layout.K_OFFSET + i * layout.K_SLOT_BYTES);
        }
        for (uint32_t i = 0; i < layout.KSCALE_SLOTS; ++i) {
            l1KScaleTensor[i] =
                resource.l1Buf.template GetBufferByByte<uint8_t>(layout.KSCALE_OFFSET + i * layout.KSCALE_SLOT_BYTES);
        }
        AllocEventID();
    }

    __aicore__ inline void AllocEventID()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(QUERY_EVENT_BASE);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(QUERY_L0_EVENT);
        for (uint32_t slot = 0; slot < layout.K_SLOTS; ++slot) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(KeyEvent(slot));
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(7);
        for (uint32_t i = 0; i < L0_STAGES; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(layout.L0_B_EVENT_BASE + i);
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
        const uint32_t headStride = tla::get<0>(gQTensorTla.stride()) / (HEAD_DIM);
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
        const uint32_t slot = sequence % layout.K_SLOTS;
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(KeyEvent(slot));
        scaleSlot = sequence % layout.KSCALE_SLOTS;
        LoadKeyGroup(source, scale, descriptor, groupWords, tokens, 0U, slot);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(KeyEvent(slot));
        if (tokens > GROUP_ROWS) {
            // Hide the first tail header's cache miss under the head DMA.
            Tile::PreloadMxfp8FullDescriptorHeader(descriptor[groupWords]);
        }
    }

    template <class TensorK, class TensorUB>
    __aicore__ inline void operator()(TensorK const& gKTensorTla, const AscendC::GlobalTensor<uint8_t>& gKScale,
                                      const AscendC::GlobalTensor<uint64_t>& descriptor, uint32_t groupWords,
                                      uint32_t validKvs, TensorUB const& ubSTensorTla, uint32_t sequence)
    {
        // Keep the common full-tile bounds and transfer sizes constant.
        // Partial tiles retain the original generic path and event sequence.
        if constexpr (layout.KV_ROWS == 512U || layout.KV_ROWS == 256U) {
            if (validKvs == layout.KV_ROWS) {
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
        const uint32_t tokens = FullTile ? layout.KV_ROWS : validKvs;
        // Grants are queued after the preceding QK reads and scale waits.
        const uint32_t kL1BufId = sequence % layout.K_SLOTS;
        const uint32_t groups = (tokens + GROUP_ROWS - 1U) / GROUP_ROWS;
        for (uint32_t pair = 0; pair < (groups + 1U) / 2U; ++pair) {
            Arch::CubePublishBoth<PIPE_MTE1>(layout.SCALE_FREE_FLAG + pair);
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
                AscendC::CrossCoreWaitFlag<4, PIPE_MTE1>(layout.SCALE_READY_FLAG + group / 2U + (group % 2U) * 16U);
            }
            const uint32_t columns = FullTile ? L0_TILE_N : min(L0_TILE_N, (tokens - first + 15U) / 16U * 16U);
            const uint32_t l0Event = layout.L0_B_EVENT_BASE + qkL0BufId;
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
            AscendC::CrossCoreWaitFlag<4, PIPE_MTE1>(layout.SCALE_READY_FLAG + groups / 2U + 16U);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(KeyEvent(kL1BufId));
    }

public:
    __aicore__ inline void FreeEventID()
    {
        ReleaseQuery();
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(QUERY_L0_EVENT);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(QUERY_EVENT_BASE);
        for (uint32_t slot = 0; slot < layout.K_SLOTS; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(KeyEvent(slot));
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(7);
        for (uint32_t i = 0; i < L0_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(layout.L0_B_EVENT_BASE + i);
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
            l1KTensor[kL1BufId], gKTensorTla.data(), l1KScaleTensor[scaleSlot], gKScale,
            descriptor[first / GROUP_ROWS * groupWords], groupWords, min(GROUP_ROWS, validKvs - first),
            tla::get<0>(gKTensorTla.stride()), first);
    }

    template <class TensorQ>
    __aicore__ inline void CopyQGmToL1(TensorQ const& gQTensorTla)
    {
        auto l1Layout = tla::MakeLayout<uint8_t, LayoutTagL1A>(qsActBaseTileAlign16, HEAD_DIM);
        auto l1Tensor = tla::MakeTensor(l1QTensor, l1Layout, Arch::PositionL1{});
        auto gQTile = GetTile(gQTensorTla, tla::MakeCoord(0, 0), tla::MakeShape(qsActBaseTile, HEAD_DIM));
        auto l1QTile = GetTile(l1Tensor, tla::MakeCoord(0, 0), tla::MakeShape(qsActBaseTile, HEAD_DIM));
        Tile::TileCopyTla<ArchTag, decltype(gQTile), decltype(l1QTile)> copyGmToL1Q;
        copyGmToL1Q(l1QTile, gQTile);
    }

    __aicore__ inline void LoadQKToL0(uint32_t kBufId, uint32_t kvBegin, uint32_t columns)
    {
        copyL1ToL0B(l0BTensor[qkL0BufId], l1KTensor[kBufId], l1KScaleTensor[scaleSlot], columns, HEAD_DIM,
                    layout.K_ROWS, kvBegin);
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
    AscendC::LocalTensor<uint8_t> l1KTensor[MAX_K_SLOTS], l1KScaleTensor[MAX_KSCALE_SLOTS];
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
    uint32_t scaleSlot = 0;
    bool queryLoaded = false;
};

} // namespace NpuArch::Gemm::Block
#endif // GBSA_BLOCK_MMAD_QK_ARCH35_MXFP8_HPP
