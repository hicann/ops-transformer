/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_MXFP8_PACKED_LOOKAHEAD_HPP
#define GBSA_MXFP8_PACKED_LOOKAHEAD_HPP
namespace NpuArch::Gemm::Block {
// Packed D192/256 Q stays in a disjoint L0A region for the whole query task.
// Grouped K lookahead uses independent L1 slots and their free/ready events.
template <class Base, uint32_t PackedDim = 0U>
struct GbsaMxPackedLookaheadQK : public Base {
    using ArchTag = typename Base::ArchTag;
    using ElementA = typename Base::ElementA;
    using ElementB = typename Base::ElementB;
    using ElementC = typename Base::ElementC;
    using TileCopy = typename Base::TileCopy;
    bool prefetched = false;
    bool qResident = false;
    bool residentQuery = false;
    bool reuseCsaQuery = false;
    static_assert(TileCopy::IS_PACKED_MX && Base::L0_STAGES == 2U &&
                      ((Base::L0_TILE_K == 256U && Base::L0_TILE_N == 128U) ||
                       (Base::L0_TILE_K == 128U && Base::L0_TILE_N == 256U)),
                  "Packed QK requires two32KiB L0B banks");

    __aicore__ inline GbsaMxPackedLookaheadQK(Arch::Resource<ArchTag>& resource, Mm1L1TileHelper& helper)
        : Base(resource, helper)
    {
        // Event4 is unused by the base QK/PV M_MTE1 protocols (0..3).
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
        if (this->l1BBufNum == 3U) {
            // K slots 0/1 use events 1/2; PV owns events 3/4.
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
        }
    }

    __aicore__ inline ~GbsaMxPackedLookaheadQK()
    {
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
        if (this->l1BBufNum == 3U) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
        }
    }

    __aicore__ inline void SetQueryTileCount(uint64_t tiles)
    {
        // A single D192 KV tile already loads Q only once in the base path.
        // Retain that path when there is no later tile to reuse the Q copy.
        reuseCsaQuery = tiles > 1U;
    }

    template <class TensorA>
    __aicore__ inline void loadQGM(TensorA& query, GemmCoord shape)
    {
        Base::loadQGM(query, shape);
        residentQuery = shape[1] == 256U || (shape[1] == 192U && reuseCsaQuery);
        if (residentQuery) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
        }
        qResident = false;
        if constexpr (Base::L0_TILE_N == 128U) {
            if (this->l1BBufNum == 3U) {
                // Q is ready in L1 and the preceding query has released its
                // disjoint L0A region. Prepare it before waiting for K records.
                PrepareGroupedQuery(shape[0], shape[1]);
            }
        }
    }

    __aicore__ inline void PrepareGroupedQuery(uint32_t rows, uint32_t embed)
    {
        auto l1Q =
            tla::MakeTensor(this->l1ATensor[0], tla::MakeLayout<ElementA, typename Base::LayoutTagL1A>(rows, embed),
                            Arch::PositionL1{});
        auto l0Q =
            tla::MakeTensor(this->l0ATensor[0][QueryL0Offset()],
                            tla::MakeLayout<ElementA, typename Base::LayoutTagL0A>(rows, embed), Arch::PositionL0A{});
        auto srcQ = GetTile(l1Q, tla::MakeCoord(0, 0), tla::MakeShape(rows, embed));
        this->copyL1ToL0A(l0Q, srcQ);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
        qResident = true;
    }

    __aicore__ inline uint32_t QueryL0Offset() const
    {
        // Up to M64 fits the existing upper 16 KiB. M128 reserves the upper
        // 32 KiB and PV places its two 16-KiB P slots in the lower half.
        return this->l1ATileM > 64U ? 32U * 1024U : 48U * 1024U;
    }

    __aicore__ inline void ReleaseQuery()
    {
        if (residentQuery) {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
        }
    }

    template <class TensorC>
    __aicore__ inline void EmitScore(TensorC& output, uint32_t rows, uint32_t columns, uint32_t slice, uint32_t bank,
                                     Arch::CrossCoreFlag& ready)
    {
        if (slice == 0U) {
            this->template WaitCrossCoreSync<4, PIPE_FIX>(ready);
        }
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(bank);
        auto c = tla::MakeTensor(this->l0CTensor[bank], tla::MakeLayoutL0C(rows, columns), Arch::PositionL0C{});
        using Copy = typename TileCopy::template CopyL0CToDst<TensorC>;
        if constexpr (AscendC::IsSameType<ElementC, float>::value) {
            auto dst =
                GetTile(output, tla::MakeCoord(0, slice * Base::L0_TILE_N), tla::MakeShape(rows, RoundUp(columns, 8U)));
            Copy copy;
            copy(dst, c);
        } else {
            const uint32_t perSubcore = RoundUp(rows, 2U) / 2U;
            auto dst = GetTile(output, tla::MakeCoord(0, slice * Base::L0_TILE_N),
                               tla::MakeShape(perSubcore, RoundUp(columns, 16U)));
            auto first = GetTile(c, tla::MakeCoord(0, 0), tla::MakeShape(perSubcore, columns));
            auto second = GetTile(c, tla::MakeCoord(perSubcore, 0), tla::MakeShape(perSubcore, columns));
            Copy copy0, copy1;
            copy0(dst, first, false);
            copy1(dst, second, true);
        }
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(bank);
    }

    __aicore__ inline bool HasPrefetched() const
    {
        return prefetched;
    }

    __aicore__ inline static uint32_t KSlotEvent(uint32_t slot)
    {
        return slot == 2U ? EVENT_ID5 : slot + 1U;
    }

    template <class TensorB>
    __aicore__ inline void PreloadGroup(TensorB& key, const Tile::GbsaMxPackedCursor& cursor, uint32_t columns,
                                        uint32_t embed, uint32_t group, uint32_t blockY, uint64_t tokenStride)
    {
        const uint32_t bank = group % this->l1BBufNum;
        const uint32_t event = KSlotEvent(bank);
        auto l1K =
            tla::MakeTensor(this->l1BTensor[bank],
                            tla::MakeLayout<ElementB, typename Base::LayoutTagL1B>(embed, columns), Arch::PositionL1{});
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(event);
        const auto saved = this->mxPackedCursor;
        this->mxPackedCursor = cursor.Group(group, columns);
        this->SparseKL1TileNLoad(key, l1K, {}, 0U, 128U, blockY, columns, embed, 0U, 0U, {}, 0U, 0U, tokenStride, 0U,
                                 0U, {}, 0U);
        this->mxPackedCursor = saved;
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(event);
    }

    // One online-softmax tile has one, two or four N128 QK MMADs at the actual D.
    // Each AIV has independent credits for its two groups. They occupy
    // disjoint L1 regions, so publishing group 2/3 need not wait for 0/1.
    // MTE1 returns each credit; the next tile reuses it in the same group order.
    template <uint32_t Groups, class TensorB, class TensorC>
    __aicore__ inline void ComputeGrouped(TensorB& key, TensorC& output, GemmCoord shape, uint32_t tile,
                                          uint32_t blockY, Arch::CrossCoreFlag ready, uint64_t tokenStride)
    {
        constexpr uint32_t embed = PackedDim != 0U ? PackedDim : 192U;
        const uint32_t rows = shape[0], columns = shape[1];
        const auto cursor = this->mxPackedCursor;
        if (!prefetched) {
            PreloadGroup(key, cursor, min(128U, columns), embed, 0U, blockY, tokenStride);
        }
        prefetched = false;
        auto l0Q =
            tla::MakeTensor(this->l0ATensor[0][QueryL0Offset()],
                            tla::MakeLayout<ElementA, typename Base::LayoutTagL0A>(rows, embed), Arch::PositionL0A{});
        if (!qResident) {
            PrepareGroupedQuery(rows, embed);
        }
        // AIV1 publishes one empty group when KV128 has only group 0.
        constexpr uint32_t notifyGroups = Groups == 1U ? 2U : Groups;
        for (uint32_t group = 0U; group < notifyGroups; ++group) {
            const uint32_t scaleFlag = 10U + group / 2U + (group % 2U) * 16U;
            // Even empty tail groups exchange credits; neither AIV can be
            // left blocked while draining a task or moving to the next batch.
            AscendC::CrossCoreWaitFlag<4, PIPE_MTE1>(scaleFlag);
            if (group * 128U < columns) {
                const uint32_t n = min(128U, columns - group * 128U);
                const uint32_t bank = group % Base::L0_STAGES;
                const uint32_t kSlot = group % this->l1BBufNum;
                const uint32_t kEvent = KSlotEvent(kSlot), bEvent = bank + 2U;
                auto l1K = tla::MakeTensor(this->l1BTensor[kSlot],
                                           tla::MakeLayout<ElementB, typename Base::LayoutTagL1B>(embed, n),
                                           Arch::PositionL1{});
                auto l0K = tla::MakeTensor(this->l0BTensor[bank],
                                           tla::MakeLayout<ElementB, typename Base::LayoutTagL0B>(embed, n),
                                           Arch::PositionL0B{});
                auto c = tla::MakeTensor(this->l0CTensor[bank], tla::MakeLayoutL0C(rows, n), Arch::PositionL0C{});
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kEvent);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(bEvent);
                this->copyL1ToL0B.SetScale(this->mxPackedScale[group * 128U * (embed / 32U)], embed / 64U);
                this->copyL1ToL0B(l0K, l1K);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(kEvent);
                AscendC::CrossCoreSetFlag<4, PIPE_MTE1>(scaleFlag);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(bEvent);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(bEvent);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(bank);
                this->tileMmad(c, l0Q, l0K, RoundUp(rows, 16U), n, embed, true);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(bEvent);
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(bank);
                EmitScore(output, rows, n, group, bank, ready);
                // Submit all remaining K groups after MMAD0. Group 3 reuses
                // slot 0 only after its MTE1 read releases the slot; MMAD0
                // continues from L0B. Each group retains its own ready event.
                if (group == 0U || this->l1BBufNum != 3U) {
                    const uint32_t begin = group + 1U;
                    const uint32_t end = this->l1BBufNum == 3U ? Groups : begin + 1U;
                    for (uint32_t next = begin; next < end && next * 128U < columns; ++next) {
                        PreloadGroup(key, cursor, min(128U, columns - next * 128U), embed, next, blockY, tokenStride);
                    }
                }
            } else {
                AscendC::CrossCoreSetFlag<4, PIPE_MTE1>(scaleFlag);
            }
        }
        qResident = true;
        this->template SetCrossCoreSync<4, PIPE_FIX>(ready);
    }

    template <class TensorB>
    __aicore__ inline void Preload(TensorB& key, const Tile::GbsaMxPackedCursor& cursor, uint32_t columns,
                                   uint32_t embed, uint32_t tile, uint32_t kvBaseTile, uint32_t blockY,
                                   uint64_t tokenStride)
    {
        if constexpr (Base::L0_TILE_N == 128U) {
            if (kvBaseTile == 512U || kvBaseTile == 256U || kvBaseTile == 128U) {
                PreloadGroup(key, cursor, min(128U, columns), embed, 0U, blockY, tokenStride);
                prefetched = true;
                return;
            }
        }
        const uint32_t bank = tile % this->l1BBufNum;
        const uint32_t event = bank + 1U;
        auto l1K =
            tla::MakeTensor(this->l1BTensor[bank],
                            tla::MakeLayout<ElementB, typename Base::LayoutTagL1B>(embed, columns), Arch::PositionL1{});
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(event);
        this->mxPackedCursor = cursor;
        this->SparseKL1TileNLoad(key, l1K, {}, tile, kvBaseTile, blockY, columns, embed, 0U, 0U, {}, 0U, 0U,
                                 tokenStride, 0U, 0U, {}, 0U);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(event);
        prefetched = true;
    }

    template <class TensorB, class TensorC>
    __aicore__ inline void operator()(TensorB& key, TensorC& output, AscendC::GlobalTensor<int32_t> indices,
                                      GemmCoord shape, uint32_t tile, uint32_t kvBaseTile, uint32_t blockY,
                                      uint64_t prefixA, uint64_t prefixB, Arch::CrossCoreFlag ready,
                                      uint64_t tokenStride)
    {
        if constexpr (Base::L0_TILE_N == 128U) {
            if (kvBaseTile == 512U) {
                ComputeGrouped<4U>(key, output, shape, tile, blockY, ready, tokenStride);
                return;
            }
            if (kvBaseTile == 256U) {
                ComputeGrouped<2U>(key, output, shape, tile, blockY, ready, tokenStride);
                return;
            }
            if (kvBaseTile == 128U) {
                ComputeGrouped<1U>(key, output, shape, tile, blockY, ready, tokenStride);
                return;
            }
        }
        if (!residentQuery) {
            Base::operator()(key, output, indices, shape, tile, kvBaseTile, blockY, prefixA, prefixB, ready,
                             tokenStride);
            return;
        }
        // Retain the non-grouped helper's compile-time D192/256 geometry.
        constexpr uint32_t embed = PackedDim != 0U ? PackedDim : (Base::L0_TILE_K == 256U ? 192U : 256U);
        const uint32_t rows = shape[0], columns = shape[1];
        const uint32_t slices = CeilDiv(columns, Base::L0_TILE_N);
        const uint32_t cStart = tile * CeilDiv(kvBaseTile, Base::L0_TILE_N);
        const uint32_t kBank = tile % this->l1BBufNum;
        const uint32_t kEvent = kBank + 1U;
        auto l1K =
            tla::MakeTensor(this->l1BTensor[kBank],
                            tla::MakeLayout<ElementB, typename Base::LayoutTagL1B>(embed, columns), Arch::PositionL1{});
        if (!prefetched)
            Preload(key, this->mxPackedCursor, columns, embed, tile, kvBaseTile, blockY, tokenStride);
        this->copyL1ToL0B.SetScale(this->mxPackedScale, embed / 64U);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kEvent);
        prefetched = false;

        constexpr uint32_t kSlices = (embed + Base::L0_TILE_K - 1U) / Base::L0_TILE_K;
        auto l1Q =
            tla::MakeTensor(this->l1ATensor[0], tla::MakeLayout<ElementA, typename Base::LayoutTagL1A>(rows, embed),
                            Arch::PositionL1{});
        for (uint32_t n = 0U; n < slices; ++n) {
            const uint32_t nAct = min(Base::L0_TILE_N, columns - n * Base::L0_TILE_N);
            const uint32_t cBank = (cStart + n) % Base::L0_STAGES;
            auto c = tla::MakeTensor(this->l0CTensor[cBank], tla::MakeLayoutL0C(rows, nAct), Arch::PositionL0C{});
            for (uint32_t k = 0U; k < kSlices; ++k) {
                const uint32_t kAct = min(Base::L0_TILE_K, embed - k * Base::L0_TILE_K);
                const uint32_t aBank = (prefixA + k) % Base::L0_STAGES;
                const uint32_t bBank = (prefixB + n * kSlices + k) % Base::L0_STAGES;
                const uint32_t bEvent = bBank + 2U;
                auto srcK =
                    GetTile(l1K, tla::MakeCoord(k * Base::L0_TILE_K, n * Base::L0_TILE_N), tla::MakeShape(kAct, nAct));
                auto l0K = tla::MakeTensor(this->l0BTensor[bBank],
                                           tla::MakeLayout<ElementB, typename Base::LayoutTagL0B>(kAct, nAct),
                                           Arch::PositionL0B{});
                // Q slices start at 48KiB; a second K slice starts at 56KiB.
                // D192: one K slice occupies 48..54KiB (at most 32 rows).
                // PV uses at most8KiB at0 and32KiB, so it cannot overwrite Q.
                auto l0Q = tla::MakeTensor(this->l0ATensor[0][48U * 1024U + k * 8192U],
                                           tla::MakeLayout<ElementA, typename Base::LayoutTagL0A>(rows, kAct),
                                           Arch::PositionL0A{});
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(bEvent);
                this->copyL1ToL0B(l0K, srcK);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(bEvent);
                if (n + 1U == slices && k + 1U == kSlices) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(kEvent);
                }
                if (n == 0U && !qResident) {
                    auto srcQ = GetTile(l1Q, tla::MakeCoord(0, k * Base::L0_TILE_K), tla::MakeShape(rows, kAct));
                    this->copyL1ToL0A(l0Q, srcQ);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(aBank);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(aBank);
                }
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(bEvent);
                if (k == 0U)
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(cBank);
                // Wide K128 uses the standard accumulating MMAD and its
                // documented small-MN barriers. SingleK256 stays initialized.
                this->tileMmad(c, l0Q, l0K, RoundUp(rows, 16U), nAct, kAct, k == 0U);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(bEvent);
            }
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(cBank);
            EmitScore(output, rows, nAct, n, cBank, ready);
        }
        qResident = true;
        this->template SetCrossCoreSync<4, PIPE_FIX>(ready);
    }
};

} // namespace NpuArch::Gemm::Block
#endif
