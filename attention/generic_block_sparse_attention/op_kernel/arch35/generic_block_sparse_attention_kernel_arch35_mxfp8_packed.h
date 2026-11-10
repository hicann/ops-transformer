/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_KERNEL_ARCH35_MXFP8_PACKED_H
#define GBSA_KERNEL_ARCH35_MXFP8_PACKED_H

#include "gbsa_mxfp8_constants.h"
#include "generic_block_sparse_attention_kernel_arch35_full_quant.h"
#include "gbsa_mxfp8_packed_qk_mmad.hpp"
#include "gbsa_mxfp8_packed_lookahead.hpp"
#include "gbsa_mxfp8_scale_cache_layout.h"
#include "gbsa_layout_utils.hpp"
#include "../attn_infra/gemm/tile_common/gbsa_build_kv_records_mxfp8.hpp"

namespace GbsaKernelArch35 {

// Small dense sparse blocks are compacted into a full attention tile. K/V
// stay in FP8; both AIVs gather their groups' E8M0 pairs into scale NZ, avoiding
// DN2NZ's 16-row padding between 4/8-token blocks. Layouts have independent keys.
template <class BlockMmadQK, class EpilogueOnlineSoftmax, class BlockMmadPV, class EpilogueRescaleO,
          uint32_t PackedDim = 0U, Format QFormat = Format::BSND, Format KvFormat = QFormat, bool IsPaged = false>
class GbsaMxfp8PackedKernelArch35
    : protected GbsaFullQuantKernelArch35<BlockMmadQK, EpilogueOnlineSoftmax, BlockMmadPV, EpilogueRescaleO,
                                          Format::TND, Format::BSND, true> {
protected:
    static_assert(PackedDim == 0U || PackedDim == 192U || PackedDim == 256U,
                  "Packed MXFP8 only supports D192 and D256; PackedDim=0 selects D192");
    static constexpr bool QHeadMajor = QFormat == Format::BNSD;
    static constexpr bool KvHeadMajor = KvFormat == Format::BNSD;
    static_assert(IsPaged || QFormat == KvFormat, "Dense MXFP8 Q/KV layouts must match");
    static_assert(!IsPaged || KvFormat != Format::TND, "Paged MXFP8 uses a batched page layout");
    using Base = GbsaFullQuantKernelArch35<BlockMmadQK, EpilogueOnlineSoftmax, BlockMmadPV, EpilogueRescaleO,
                                           Format::TND, Format::BSND, true>;
    using ElementQ = typename Base::ElementQ;
    using ArchTag = typename Base::ArchTag;
    using ElementK = typename Base::ElementK;
    using ElementV = typename Base::ElementV;
    using ElementS = typename Base::ElementS;
    using ElementP = typename Base::ElementP;
    using ElementOTmp = typename Base::ElementOTmp;
    using ElementO = typename EpilogueRescaleO::ElementO;
    using ElementLse = typename Base::ElementLse;
    using LayoutQ = typename Base::LayoutQ;
    using LayoutK = typename Base::LayoutK;
    using LayoutV = typename Base::LayoutV;
    using LayoutS = typename Base::LayoutS;
    using LayoutO = typename Base::LayoutO;
    using LayoutOTmp = typename Base::LayoutOTmp;
    using LayoutLse = typename Base::LayoutLse;
    using Cursor = NpuArch::Gemm::Tile::GbsaMxPackedCursor;
    using Base::resource;
    using Base::qHeads_;
    using Base::kvHeads_;
    using Base::embed_;
    using Base::blockShapeY_;
    using Base::qBlockNum_;
    using Base::topK_;
    using Base::scaleValue_;
    using Base::groupSize_;
    using Base::qBaseTile_;
    using Base::kvBaseTile_;
    using Base::pL1BufNum_;
    using Base::mm1L0ATotalStages_;
    using Base::mm1L0BTotalStages_;
    using Base::mm2L0ATotalStages_;
    using Base::mm2L0BTotalStages_;
    using Base::mm2L1AddrStart_;
    using Base::mm1L1TileHelper_;
    using Base::mm2L1TileHelper_;
    using Base::uBufTileHelper_;
    static constexpr uint32_t PRE_LAUNCH = Base::PRE_LAUNCH;
    static constexpr uint32_t RING = Base::MAX_CROSS_CORE_BUF_STAGES;
    static constexpr uint32_t UB_STAGES = Base::UB_S_OTMP_BUF_STAGES;
    static constexpr uint32_t DESCRIPTOR_FLAG = 7U;
    static constexpr uint32_t DESCRIPTOR_STAGES = 4U;
    static constexpr uint32_t DESCRIPTOR_AHEAD = 2U;
    static constexpr uint32_t GROUPED_RECORD_UB_STAGES = 2U;
    static constexpr uint32_t SCALE_FLAG = 10U;
    static constexpr uint32_t SCALE_L1_LARGE_BASE = NpuArch::Gemm::Tile::GBSA_MX_SCALE_BASE - RING * 4096U;

#ifdef __DAV_VEC__
    // Each AIV owns its record slots. Keep their rotation across tiles/tasks;
    // GM descriptor slots remain owned by Cube until delayed PV consumes them.
    uint32_t groupedRecordSlot_ = 0U;
    AscendC::GlobalTensor<int32_t> batchPages_;
    uint64_t pageStride_ = 0U;
#endif

    struct PendingTile {
        Cursor cursor;
        uint64_t outputOffset;
        uint32_t rows;
        uint32_t computeRows;
        uint64_t valueOffset;
        uint64_t valueScaleOffset;
        bool first;
        bool last;
    };

    __aicore__ inline uint64_t QueryRowOffset(const BatchSeqInfo& seq, uint32_t token, uint32_t head)
    {
        return GetTensorOffset<QFormat>(seq, token, head, qHeads_, 1U);
    }

    template <class Consumer>
    __aicore__ inline void ConsumePendingTile(const PendingTile& item, uint64_t sequence, uint64_t produced,
                                              Consumer& consumer, AscendC::GlobalTensor<ElementV>& values,
                                              AscendC::GlobalTensor<ElementO>& output,
                                              AscendC::GlobalTensor<ElementLse>& lse,
                                              AscendC::GlobalTensor<uint8_t>& valueScale,
                                              AscendC::LocalTensor<ElementOTmp> (&partialBuffers)[UB_STAGES],
                                              uint64_t tokenStride, uint64_t queryHeadStride)
    {
        const uint32_t pSlot = sequence % RING;
        const uint32_t oSlot = sequence % uBufTileHelper_.loUbBufNum;
        Arch::CrossCoreFlag pReady(pSlot + UB_STAGES), oReady(oSlot + UB_STAGES + RING);
        const GemmCoord shape{item.computeRows, embed_, item.rows};
#ifdef __DAV_CUBE__
        consumer.SetMxPacked(item.cursor);
        auto value = tla::MakeTensor(values[item.valueOffset],
                                     tla::MakeLayout<ElementV, LayoutV>(kvBaseTile_, tokenStride), Arch::PositionGM{});
        auto partial = tla::MakeTensor(partialBuffers[oSlot],
                                       tla::MakeLayout<ElementOTmp, LayoutOTmp>(RoundUp(item.computeRows, 16U), embed_),
                                       Arch::PositionUB{});
        AscendC::GlobalTensor<int32_t> unused;
        consumer(value, partial, unused, shape, sequence % 6U, kvBaseTile_, blockShapeY_,
                 produced * mm1L0ATotalStages_ + sequence * mm2L0ATotalStages_,
                 produced * mm1L0BTotalStages_ + sequence * mm2L0BTotalStages_, pReady, oReady, tokenStride);
        AscendC::CrossCoreSetFlag<4, PIPE_S>(DescriptorFlag(sequence % DESCRIPTOR_STAGES));
        AscendC::CrossCoreSetFlag<4, PIPE_S>(DescriptorFlag(sequence % DESCRIPTOR_STAGES) - 16U);
#endif
#ifdef __DAV_VEC__
        consumer.SetMxVScale(valueScale[item.valueScaleOffset]);
        auto out = tla::MakeTensor(output[item.outputOffset],
                                   tla::MakeLayout<ElementO, LayoutO>(qBaseTile_, queryHeadStride), Arch::PositionGM{});
        auto outLse = tla::MakeTensor(lse, tla::MakeLayout<ElementLse, LayoutLse>(qBaseTile_, 1), Arch::PositionGM{});
        consumer(out, outLse, shape, pSlot, sequence % 6U, item.first, item.last, oReady);
#endif
    }

    __aicore__ inline static uint32_t DescriptorFlag(uint32_t slot)
    {
        // Mode4 maps V1-local 0..15 to Cube 16..31. Only the Cube
        // endpoint adds16; V1 must keep its local 7/8/9/15 IDs.
        const uint32_t localFlag = slot < 3U ? DESCRIPTOR_FLAG + slot : 15U;
#ifdef __DAV_CUBE__
        return localFlag + 16U;
#else
        return localFlag;
#endif
    }

    __aicore__ inline void CalcPackedUb()
    {
        static_assert(AscendC::IsSameType<ElementS, float>::value, "Packed MXFP8 requires FP32 scores");
        static_assert(sizeof(ElementS) == GbsaMxScaleCache::SCORE_ELEMENT_BYTES &&
                          sizeof(ElementP) == GbsaMxScaleCache::PROBABILITY_ELEMENT_BYTES &&
                          sizeof(ElementOTmp) == sizeof(float),
                      "Host and kernel UB element sizes must agree");
        static_assert(UB_STAGES == GbsaMxScaleCache::UB_COMPUTE_STAGES && RING == GbsaMxScaleCache::UB_DELTA_STAGES &&
                          Epilogue::Block::UBufTileHelper::STATS_ROW_NUM == GbsaMxScaleCache::UB_STATS_ROWS,
                      "Host and kernel UB buffer counts must agree");
        // Head groups are split between the two AIVs. Statistics retain their
        // 64-row stride; the largest tier gives PV only one UB result slot.
        const uint32_t rows = GbsaMxScaleCache::VectorRows(qBaseTile_);
        const uint32_t pvStages = GbsaMxScaleCache::PvUbStages(qBaseTile_);
        constexpr uint32_t stats = GbsaMxScaleCache::UB_STATS_ROWS * sizeof(float);
        const uint32_t s = 0;
        const uint32_t p =
            static_cast<uint32_t>(GbsaMxScaleCache::ScoreUbSlotBytes(qBaseTile_, kvBaseTile_) * UB_STAGES);
        const uint32_t lo = p + rows * kvBaseTile_ * sizeof(ElementP) * UB_STAGES;
        const uint32_t outputSlotBytes = GbsaMxScaleCache::OutputUbSlotBytes(embed_, qBaseTile_);
        const uint32_t go = lo + outputSlotBytes * pvStages;
        const uint32_t lm = go + outputSlotBytes;
        const uint32_t gm = lm + stats;
        const uint32_t dm = gm + stats;
        const uint32_t ll = dm + stats * RING;
        const uint32_t gl = ll + stats;
        const uint32_t lse = gl + stats;
        const uint32_t mask = lse + 64U * 32U;
        uBufTileHelper_ = Epilogue::Block::UBufTileHelper(rows, kvBaseTile_, embed_, s, p, lo, go, lm, gm, dm, ll, gl,
                                                          lse, mask, pvStages);
        // Keep D192 rows compact within each D256-capacity slot. Cube Fixpipe
        // and Vector rescale must use the same reserved stride between slots.
        uBufTileHelper_.loUbColCapacity = GbsaMxScaleCache::BufferDim(embed_);
        uBufTileHelper_.mxVScaleStartOffset = mask;
    }

    __aicore__ inline static uint64_t Prefix(uint64_t iteration, uint64_t total, uint32_t qk, uint32_t pv, bool isQk)
    {
        if (isQk) {
            return iteration * qk + (iteration > PRE_LAUNCH ? iteration - PRE_LAUNCH : 0U) * pv;
        }
        const uint64_t submitted = iteration + PRE_LAUNCH + 1U < total ? iteration + PRE_LAUNCH + 1U : total;
        return submitted * qk + iteration * pv;
    }

#ifdef __DAV_VEC__
    // The sparse walk compacts only the current KV head. BSND pads each row;
    // BNSD pads each contiguous microblock. Neither needs a token map here.
    template <bool BlockPacked = false>
    static __simd_vf__ inline void GatherStreamScaleNz(__ubuf__ uint16_t* output, __ubuf__ uint16_t* cache,
                                                       uint16_t rows, uint16_t groups64, uint16_t blockShift = 0U,
                                                       uint16_t blockPairs = 16U)
    {
        using namespace AscendC::Reg;
        // Consecutive lanes are the existing NZ output order: 16 rows per
        // group, then the next group, followed by the second 16-row block.
        // D192 uses 96 lanes and D256 uses128, covering32 rows per iteration.
        RegTensor<int16_t> signedLanes;
        RegTensor<uint16_t> laneGroup, group, row, rowBias, groupBias, zero;
        RegTensor<uint16_t> lowMask;
        RegTensor<uint16_t> tokenIndex, offset, packed;
        uint32_t laneCount = uint32_t(groups64) * 32U;
        MaskReg all = UpdateMask<uint16_t>(laneCount);
        Arange(signedLanes, int16_t(0));
        Duplicate(zero, uint16_t(0), all);
        Duplicate(lowMask, uint16_t(15), all);
        ShiftRights(laneGroup, (RegTensor<uint16_t>&)signedLanes, int16_t(4), all);
        MaskReg second;
        Compares<uint16_t, AscendC::CMPMODE::GE>(second, laneGroup, groups64, all);
        Duplicate(groupBias, groups64, all);
        Select(groupBias, groupBias, zero, second);
        Sub(group, laneGroup, groupBias, all);
        Duplicate(rowBias, uint16_t(16), all);
        Select(rowBias, rowBias, zero, second);
        And(row, (RegTensor<uint16_t>&)signedLanes, lowMask, all);
        Add(row, row, rowBias, all);
        for (uint16_t baseRow = 0; baseRow < rows; baseRow += 32U) {
            const uint16_t remaining = rows - baseRow;
            // A one-block tail must store only48/64 elements, not96/128.
            uint32_t storeCount = (remaining <= 16U ? 16U : 32U) * uint32_t(groups64);
            MaskReg storeMask = UpdateMask<uint16_t>(storeCount);
            MaskReg valid;
            Compares<uint16_t, AscendC::CMPMODE::LT>(valid, row, remaining, storeMask);
            Adds(tokenIndex, row, baseRow, valid);
            if constexpr (BlockPacked) {
                RegTensor<uint16_t> block, inner, mask;
                ShiftRights(block, tokenIndex, int16_t(blockShift), valid);
                Duplicate(mask, uint16_t((1U << blockShift) - 1U), valid);
                And(inner, tokenIndex, mask, valid);
                Muls(offset, block, blockPairs, valid);
                Muls(inner, inner, groups64, valid);
                Add(offset, offset, inner, valid);
            } else {
                Muls(offset, tokenIndex, uint16_t(GbsaMxScaleCache::RAW_ROW_BYTES / sizeof(uint16_t)), valid);
            }
            Add(offset, offset, group, valid);
            Duplicate(packed, uint16_t(0), storeMask);
            Gather(packed, cache, offset, valid);
            StoreAlign(output + uint32_t(baseRow) * groups64, packed, storeMask);
        }
    }

    // Each source row is one BSND token: skip the other KV heads between
    // rows. DataCopyPad aligns each destination row to 32 bytes, independently
    // of span boundaries and the number of heads. Source points at this head.
    __aicore__ inline static void CopyHeadScaleRows(AscendC::LocalTensor<uint16_t> raw,
                                                    AscendC::GlobalTensor<uint16_t> source, uint32_t rows,
                                                    uint32_t rowPairs, uint32_t headPairs)
    {
        AscendC::DataCopyPad(raw, source,
                             AscendC::DataCopyExtParams(rows, headPairs * sizeof(uint16_t),
                                                        (rowPairs - headPairs) * sizeof(uint16_t), 0U, 0U),
                             AscendC::DataCopyPadExtParams<uint16_t>(false, 0U, 0U, 0U));
    }

    __aicore__ inline static uint32_t StreamRawFreeEvent(uint32_t slot)
    {
        return EVENT_ID3 + slot;
    }

    __aicore__ inline static uint32_t GroupedRawReadyEvent(uint32_t slot, uint32_t local)
    {
        // Packed scores and partial O stay in UB: their GM-input events are
        // unused. Reserve MTE2_V ID2 for VScale; Raw groups use 0/1/3/4/5/6.
        // S_V uses the same mapping, while V_S still protects a whole Raw tile.
        static_assert(GbsaMxScaleCache::GROUPED_RAW_STAGES * GbsaMxScaleCache::GROUPS_PER_AIV <= 7U,
                      "Raw group readiness must leave an event for VScale");
        const uint32_t event = slot * GbsaMxScaleCache::GROUPS_PER_AIV + local;
        return event < EVENT_ID2 ? event : event + 1U;
    }

    __aicore__ inline static uint32_t GroupedRawSlot(uint64_t sequence, uint32_t rawStages)
    {
        return rawStages == GbsaMxScaleCache::GROUPED_RAW_STAGES ? sequence % GbsaMxScaleCache::GROUPED_RAW_STAGES :
                                                                   sequence & 1U;
    }

    struct GroupedScaleConsumer {
        AscendC::GlobalTensor<uint16_t> scales;
        AscendC::LocalTensor<uint16_t> raw;
        uint32_t rowPairs;
        uint32_t headPairs;
        uint32_t blockRows;
        uint32_t outputRows = 0U;

        __aicore__ inline void CopyHeadMajorRun(uint32_t dstRow, uint64_t offset, uint32_t rows)
        {
            const uint32_t blockPairs = RoundUp(blockRows * headPairs, 16U);
            const uint32_t fullBlocks = rows / blockRows;
            const uint32_t fullRows = fullBlocks * blockRows;
            auto dst = raw[(dstRow / blockRows) * blockPairs];
            auto src = scales[offset / 64U];
            if (fullBlocks != 0U) {
                AscendC::DataCopyPad(
                    dst, src,
                    AscendC::DataCopyExtParams(fullBlocks, blockRows * headPairs * sizeof(uint16_t), 0U, 0U, 0U),
                    AscendC::DataCopyPadExtParams<uint16_t>(false, 0, 0, 0));
            }
            if (fullRows != rows) {
                // The sorted, unique valid prefix can have a partial block
                // only at its end. Copy its exact bytes; NZ masks padded rows.
                AscendC::DataCopyPad(
                    dst[fullBlocks * blockPairs], src[fullRows * headPairs],
                    AscendC::DataCopyExtParams(1U, (rows - fullRows) * headPairs * sizeof(uint16_t), 0U, 0U, 0U),
                    AscendC::DataCopyPadExtParams<uint16_t>(false, 0, 0, 0));
            }
        }

        __aicore__ inline void Record(uint64_t offset, uint32_t rows, uint64_t stride, uint32_t matrices)
        {
            // Contiguous FP8 K/V and KScale caches share their physical page
            // ordering. Byte offsets / 64 address uint16_t KScale pairs;
            // the base already selects this head in both KV layouts.
            if constexpr (KvHeadMajor) {
                if (matrices == 2U && rows == blockRows) {
                    const uint32_t bytes = rows * headPairs * sizeof(uint16_t);
                    const uint32_t blockPairs = RoundUp(blockRows * headPairs, 16U);
                    // BNSD uses the one GM stride for the gap between blocks;
                    // each block is contiguous and its UB end is padded to 32 B.
                    AscendC::DataCopyPad(raw[(outputRows / blockRows) * blockPairs], scales[offset / 64U],
                                         AscendC::DataCopyExtParams(2U, bytes, stride / 32U - bytes, 0U, 0U),
                                         AscendC::DataCopyPadExtParams<uint16_t>(false, 0, 0, 0));
                } else {
                    for (uint32_t matrix = 0U; matrix < matrices; ++matrix) {
                        CopyHeadMajorRun(outputRows + matrix * rows, offset + matrix * stride, rows);
                    }
                }
            } else {
                for (uint32_t matrix = 0U; matrix < matrices; ++matrix) {
                    CopyHeadScaleRows(
                        raw[(outputRows + matrix * rows) * (GbsaMxScaleCache::RAW_ROW_BYTES / sizeof(uint16_t))],
                        scales[(offset + matrix * stride) / 64U], rows, rowPairs, headPairs);
                }
            }
            outputRows += matrices * rows;
        }
    };

    // Three tile slots per AIV when UB permits, with separate Raw regions for its two
    // groups. Descriptor construction and scale DMA share this span walk;
    // K/V payloads are never staged in UB. Records live until delayed PV.
    __aicore__ inline void PrepareGroupedTile(Cursor cursor, AscendC::LocalTensor<uint64_t> recordsUb,
                                              AscendC::GlobalTensor<uint64_t> info,
                                              AscendC::GlobalTensor<uint16_t> scales,
                                              AscendC::LocalTensor<uint16_t> raw, uint32_t rows, uint32_t head,
                                              uint32_t packedBegin, uint64_t tokenStride, uint32_t slot,
                                              uint32_t rawSlot, bool positioned, AscendC::LocalTensor<uint16_t>* packed,
                                              AscendC::LocalTensor<uint8_t> l1Scale, bool publishEarly)
    {
        constexpr uint32_t GROUP = GbsaMxScaleCache::GROUP_TOKENS;
        const uint32_t lane = AscendC::GetSubBlockIdx();
        constexpr uint32_t groupWords = 8U + 2U * GROUP;
        const uint32_t localGroups = GbsaMxScaleCache::GroupsPerAiv(kvBaseTile_);
        const uint32_t headPairs = embed_ / 64U;
        const uint32_t rowPairs = KvHeadMajor ? headPairs : kvHeads_ * headPairs;
        const auto headScales = scales[KvHeadMajor ? 0U : head * headPairs];
        const uint32_t blockRows = GbsaMxScaleCache::RawBlockRows(blockShapeY_, KvHeadMajor);
        const uint32_t rawGroupPairs = GbsaMxScaleCache::RawCapacityBytes(GROUP) / sizeof(uint16_t);
        const auto* selectedGm = reinterpret_cast<const __gm__ int32_t*>(cursor.indices.GetPhyAddr());
        AscendC::CrossCoreWaitFlag<4, PIPE_S>(DescriptorFlag(slot));
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(StreamRawFreeEvent(rawSlot));
        for (uint32_t local = 0U; local < localGroups; ++local) {
            const uint32_t group = lane + 2U * local;
            const uint32_t groupRows = rows > group * GROUP ? min(GROUP, rows - group * GROUP) : 0U;
            auto groupRaw = raw[local * rawGroupPairs];
            const uint32_t recordEvent = EVENT_ID5 + groupedRecordSlot_;
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(recordEvent);
            auto recordBuffer = recordsUb[groupedRecordSlot_ * groupWords];
            auto* records = reinterpret_cast<__ubuf__ uint64_t*>(recordBuffer.GetPhyAddr());
            GroupedScaleConsumer consumer{headScales, groupRaw, rowPairs, headPairs, blockRows, 0U};
            uint32_t count = 0U;
            const uint32_t position = packedBegin + group * GROUP;
            const uint32_t firstBlock = position / blockShapeY_;
            const uint32_t blocks = groupRows / blockShapeY_;
            const bool fullGroup = positioned && cursor.contiguousRows == 0U && position % blockShapeY_ == 0U &&
                                   blocks * blockShapeY_ == groupRows && firstBlock + blocks <= cursor.count;
            if (fullGroup) {
                if constexpr (IsPaged) {
                    count = NpuArch::Gemm::Tile::BuildFullPagedKvRecordsMxfp8(
                        selectedGm + firstBlock, batchPages_.GetPhyAddr(), records, blocks, blockShapeY_, tokenStride,
                        pageStride_, consumer);
                } else {
                    count = NpuArch::Gemm::Tile::BuildFullKvRecordsMxfp8(selectedGm + firstBlock, records, blocks,
                                                                         blockShapeY_, tokenStride, consumer);
                }
            } else {
                // Partial groups and tasks outside the bounded fast walk keep
                // the cursor's clipping and bounded-memory fallback semantics.
                Cursor part = cursor;
                part.Advance(group * GROUP);
                while (consumer.outputRows < groupRows) {
                    uint32_t token = 0U, span = 0U;
                    if (!part.NextSpan(groupRows - consumer.outputRows, token, span)) {
                        break;
                    }
                    while (span != 0U) {
                        uint32_t rowsToCopy = span;
                        uint64_t offset = static_cast<uint64_t>(token) * tokenStride;
                        if constexpr (IsPaged) {
                            constexpr uint32_t PAGE = 128U;
                            rowsToCopy = min(span, PAGE - token % PAGE);
                            offset = static_cast<uint64_t>(static_cast<uint32_t>(batchPages_.GetValue(token / PAGE))) *
                                         pageStride_ +
                                     static_cast<uint64_t>(token % PAGE) * tokenStride;
                        }
                        records[8U + 2U * count] = offset;
                        records[9U + 2U * count] = static_cast<uint64_t>(rowsToCopy) << 1U;
                        ++count;
                        consumer.Record(offset, rowsToCopy, 0U, 1U);
                        token += rowsToCopy;
                        span -= rowsToCopy;
                    }
                }
            }
            // Publish this group's DMA completion before submitting the next
            // group's transfers, including when this group has no valid rows.
            const uint32_t rawReady = GroupedRawReadyEvent(rawSlot, local);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(rawReady);
            records[0] = count;
            records[1] = 0U;
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(recordEvent);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(recordEvent);
            AscendC::DataCopy(info[group * groupWords], recordBuffer, RoundUp(8U + 2U * count, 4U));
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(recordEvent);
            groupedRecordSlot_ = (groupedRecordSlot_ + 1U) % GROUPED_RECORD_UB_STAGES;
            AscendC::SetFlag<AscendC::HardEvent::S_V>(rawReady);
            if (publishEarly && local == 0U) {
                // Submit the first group's Vector/MTE3 work now so it can
                // overlap Scalar descriptor construction and MTE2 for group 2/3.
                PublishGroupedScale(rows, raw, packed[local], rawSlot, l1Scale, local);
            }
        }
        AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(DescriptorFlag(slot));
        if (publishEarly) {
            // Keep descriptor READY ahead of the second scale publication.
            // Both Raw reads must finish before the entire tile is reusable.
            if (localGroups == 2U) {
                PublishGroupedScale(rows, raw, packed[1U], rawSlot, l1Scale, 1U);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_S>(StreamRawFreeEvent(rawSlot));
        }
    }

    __aicore__ inline void PublishGroupedScale(uint32_t rows, AscendC::LocalTensor<uint16_t> raw,
                                               AscendC::LocalTensor<uint16_t> packed, uint32_t rawSlot,
                                               AscendC::LocalTensor<uint8_t> l1Scale, uint32_t local)
    {
        constexpr uint32_t GROUP = GbsaMxScaleCache::GROUP_TOKENS;
        const uint32_t groups = embed_ / 64U;
        const uint32_t rawGroupPairs = GbsaMxScaleCache::RawCapacityBytes(GROUP) / sizeof(uint16_t);
        const uint32_t blockRows = GbsaMxScaleCache::RawBlockRows(blockShapeY_, KvHeadMajor);
        const uint16_t blockShift = blockRows == 16U ? 4U : (blockRows == 8U ? 3U : 2U);
        const uint16_t blockPairs = RoundUp(blockRows * groups, 16U);
        const uint32_t rawReady = GroupedRawReadyEvent(rawSlot, local);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(rawReady);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(rawReady);
        const uint32_t group = AscendC::GetSubBlockIdx() + 2U * local;
        const uint32_t groupRows = rows > group * GROUP ? min(GROUP, rows - group * GROUP) : 0U;
        const uint32_t nzEvent = EVENT_ID5 + local;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(nzEvent);
        if (groupRows != 0U) {
            GatherStreamScaleNz<KvHeadMajor>(
                reinterpret_cast<__ubuf__ uint16_t*>(packed.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint16_t*>(raw.GetPhyAddr()) + local * rawGroupPairs, groupRows, groups,
                blockShift, blockPairs);
        }
        // The two groups have disjoint NZ and L1 storage. One credit per
        // local group avoids serializing group 2/3 behind MTE1 for 0/1.
        // Tile order is unchanged, and the previous same-group consumer
        // must release its credit before any later tile can publish it.
        AscendC::CrossCoreWaitFlag<4, PIPE_MTE3>(SCALE_FLAG + local);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(nzEvent);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(nzEvent);
        if (groupRows != 0U) {
            AscendC::DataCopy(l1Scale.ReinterpretCast<uint16_t>()[group * GROUP * groups], packed,
                              RoundUp(groupRows, 16U) * groups);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(nzEvent);
        AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(SCALE_FLAG + local);
    }

    __aicore__ inline void PublishGroupedTile(uint32_t rows, AscendC::LocalTensor<uint16_t> raw,
                                              AscendC::LocalTensor<uint16_t>* packed, uint32_t rawSlot,
                                              AscendC::LocalTensor<uint8_t> l1Scale)
    {
        for (uint32_t local = 0U; local < GbsaMxScaleCache::GroupsPerAiv(kvBaseTile_); ++local) {
            PublishGroupedScale(rows, raw, packed[local], rawSlot, l1Scale, local);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_S>(StreamRawFreeEvent(rawSlot));
    }

    __aicore__ inline void InitStreamScalePipeline(uint32_t rawStages)
    {
        for (uint32_t slot = 0U; slot < rawStages; ++slot) {
            AscendC::SetFlag<AscendC::HardEvent::V_S>(StreamRawFreeEvent(slot));
        }
        for (uint32_t slot = 0U; slot < GbsaMxScaleCache::NZ_STAGES; ++slot) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID5 + slot);
        }
    }

    __aicore__ inline void ReleaseStreamScalePipeline(uint32_t rawStages)
    {
        for (uint32_t slot = 0U; slot < rawStages; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(StreamRawFreeEvent(slot));
        }
        for (uint32_t slot = 0U; slot < GbsaMxScaleCache::NZ_STAGES; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID5 + slot);
        }
    }

#endif

public:
    __aicore__ inline void operator()(GbsaFullQuantKernelParamsArch35 const& params)
    {
        GET_TILING_DATA_WITH_STRUCT(GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData, data, params.tiling);
        const auto* td = &data;
        Base::FetchBaseShapeInfo(td, params.metaData);
        Base::CalcOnChipBufTileInfo(td);
        CalcPackedUb();
        const uint32_t coreNum = AscendC::GetBlockNum();
        const bool causal =
            (td->tilingKey / GbsaTilingKeyEncoding::MASK_DIVISOR) % GbsaTilingKeyEncoding::DECIMAL_RADIX != 0U;
        uint32_t coreIdx = AscendC::GetBlockIdx();
#ifdef __DAV_VEC__
        coreIdx /= AscendC::GetSubBlockNum();
#endif
        const auto* meta = reinterpret_cast<__gm__ const GbsaMetadata::Metadata*>(params.metaData);
        uint32_t fdScheduleFlags = 0U;
        if (!(causal ? GsaFd::ValidateMetadata<true>(meta, td, coreNum, &fdScheduleFlags) :
                       GsaFd::ValidateMetadata<false>(meta, td, coreNum, &fdScheduleFlags))) {
            return;
        }
        if (meta->saTotalTaskNum == 0 && params.sequsedQ != nullptr) {
            // There are no scheduled Q tokens. Confirm the actual lengths before
            // clearing all physical output, including padding, without a pipeline.
            AscendC::GlobalTensor<int32_t> actualQ;
            actualQ.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.sequsedQ));
            bool allQueriesEmpty = true;
            for (uint32_t batch = 0U; batch < td->batch; ++batch) {
                allQueriesEmpty = allQueriesEmpty && actualQ.GetValue(batch) == 0;
            }
            if (allQueriesEmpty) {
#ifdef __DAV_VEC__
                AscendC::TPipe emptyPipe;
                const uint64_t elements = td->totalQTokens * qHeads_ * embed_;
                const uint32_t vectorCores = coreNum * AscendC::GetSubBlockNum();
                const uint64_t coreElements = ((elements + vectorCores - 1U) / vectorCores + 15U) / 16U * 16U;
                const uint64_t begin = AscendC::GetBlockIdx() * coreElements;
                const uint64_t end = min(elements, begin + coreElements);
                AscendC::GlobalTensor<ElementO> output;
                output.SetGlobalBuffer(reinterpret_cast<__gm__ ElementO*>(params.o));
                for (uint64_t offset = begin; offset < end;) {
                    const uint64_t size = min(end - offset, uint64_t{1U << 30U});
                    auto target = output[offset];
                    AscendC::InitGlobalMemory(target, size, static_cast<ElementO>(0));
                    offset += size;
                }
#endif
                return;
            }
        }
        // The validated metadata flag is already local. Ordinary packed tasks
        // need not fetch the optional FD tiling fields to select their path.
        const bool fdEnabled = (fdScheduleFlags & GbsaMetadata::FD_SCHEDULE_ENABLED) != 0U && td->fdStaticEnabled != 0U;
        if (fdEnabled) {
            Run<true>(params, td, meta, coreNum, coreIdx, causal);
        } else {
            Run<false>(params, td, meta, coreNum, coreIdx, causal);
        }
    }

private:
    // Keep FD partial/combine state out of the ordinary packed loop. Both
    // specializations retain the same layout, buffers and runtime batch walk.
    template <bool fdEnabled>
    __aicore__ inline void Run(const GbsaFullQuantKernelParamsArch35& params,
                               const GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData* td,
                               __gm__ const GbsaMetadata::Metadata* meta, uint32_t coreNum, uint32_t coreIdx,
                               bool causal)
    {
        const uint32_t scaleCredits = GbsaMxScaleCache::GroupsPerAiv(kvBaseTile_);
        const uint32_t groups64 = embed_ / 64U;
        AscendC::GlobalTensor<ElementLse> gLse;
        gLse.SetGlobalBuffer(reinterpret_cast<__gm__ ElementLse*>(params.softmaxLse));
        AscendC::GlobalTensor<float> gPartialO, gPartialLse;
        if (fdEnabled) {
            gPartialO.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.workSpace + td->fdPartialOOffset));
            gPartialLse.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.workSpace + td->fdPartialLseOffset));
        }
        AscendC::GlobalTensor<uint16_t> kScale;
        kScale.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(params.kDequantScale));
        AscendC::GlobalTensor<int32_t> usedQ, usedKv, unusedIndices;
        usedQ.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.sequsedQ));
        usedKv.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.sequsedKv));
        AscendC::GlobalTensor<int64_t> cuQ, cuKv;
        cuQ.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(params.cuSeqLengths));
        cuKv.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(params.cuSeqLengthsKv));
        const uint32_t batchCount = td->batch;
        const uint64_t originalTokenStride = KvHeadMajor ? embed_ : static_cast<uint64_t>(kvHeads_) * embed_;
        const uint64_t queryHeadStride = QHeadMajor ? static_cast<uint64_t>(td->qSeqCapacity) * embed_ : embed_;
        const uint64_t queryScaleHeadStride =
            QHeadMajor ? static_cast<uint64_t>(td->qSeqCapacity) * groups64 : groups64;
        // Each core owns four independent descriptor slots. Their group headers
        // and records stay live until PV consumption; all sizes align to 64 bytes.
        const uint32_t descriptorTileWords = GbsaMxScaleCache::DescriptorTileWords(kvBaseTile_, blockShapeY_);
        const uint64_t workspaceCoreBytes = DESCRIPTOR_STAGES * descriptorTileWords * sizeof(uint64_t);
        const uint64_t workspaceCoreOffset = static_cast<uint64_t>(coreIdx) * workspaceCoreBytes;
        AscendC::GlobalTensor<uint64_t> descriptors;
        descriptors.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(params.workSpace + workspaceCoreOffset));
        AscendC::LocalTensor<ElementP> l1P[RING];
        AscendC::LocalTensor<ElementS> ubS[UB_STAGES];
        AscendC::LocalTensor<ElementOTmp> ubO[UB_STAGES];
        AscendC::LocalTensor<uint8_t> l1Scale[RING];
        Base::InitCrossCoreDstBuf(l1P, ubS, ubO);
        const uint32_t scaleL1SlotBytes = 4096U;
        const uint32_t scaleL1Base = SCALE_L1_LARGE_BASE;
        for (uint32_t i = 0; i < RING; ++i) {
            l1Scale[i] = resource.l1Buf.template GetBufferByByte<uint8_t>(scaleL1Base + i * scaleL1SlotBytes);
        }
        Base::template InitSyncFlags<4, 4, 4>();
#ifdef __DAV_CUBE__
        NpuArch::Gemm::Block::GbsaMxPackedLookaheadQK<BlockMmadQK, PackedDim> qk(resource, mm1L1TileHelper_);
        BlockMmadPV pv(resource, mm2L1AddrStart_, mm2L1TileHelper_);
        for (uint32_t i = 0; i < scaleCredits; ++i) {
            AscendC::CrossCoreSetFlag<4, PIPE_MTE1>(SCALE_FLAG + i);
            // Both AIVs receive independent credits for their two groups.
            AscendC::CrossCoreSetFlag<4, PIPE_MTE1>(SCALE_FLAG + 16U + i);
        }
        for (uint32_t i = 0; i < DESCRIPTOR_STAGES; ++i) {
            AscendC::CrossCoreSetFlag<4, PIPE_S>(DescriptorFlag(i));
            AscendC::CrossCoreSetFlag<4, PIPE_S>(DescriptorFlag(i) - 16U);
        }
#endif
#ifdef __DAV_VEC__
        EpilogueOnlineSoftmax softmax(resource, scaleValue_, uBufTileHelper_);
        EpilogueRescaleO rescale(resource, uBufTileHelper_);
        float alpha = 16.0f;
        if (params.pQuantScale != nullptr) {
            AscendC::GlobalTensor<float> pScale;
            pScale.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.pQuantScale));
            alpha = pScale.GetValue(0);
        }
        softmax.SetPQuantScale(alpha);
        rescale.SetPQuantScale(alpha);
        const uint32_t cacheOffset =
            RoundUp(uBufTileHelper_.mxVScaleStartOffset + GbsaMxScaleCache::ValueScaleUbBytes(embed_), 256U);
        // Raw -> NZ -> descriptor scratch. Grouped indices are read directly from GM.
        const uint32_t rawStages = GbsaMxScaleCache::PipelineRawStages(
            td->kvSeqCapacity, kvHeads_, embed_, blockShapeY_, kvBaseTile_, KvHeadMajor, qBaseTile_);
        const uint32_t rawTileTokens = GbsaMxScaleCache::RawTileTokens(kvBaseTile_);
        const uint32_t rawBytes = GbsaMxScaleCache::RawCapacityBytes(rawTileTokens);
        const uint32_t cacheBytes = rawStages * rawBytes;
        const uint32_t packedScaleBytes = GbsaMxScaleCache::NzScaleUbSlotBytes(embed_);
        const uint32_t auxBytes = GbsaMxScaleCache::PipelineAuxBytes(embed_, kvBaseTile_, rawStages);

        auto recordsUb = resource.ubBuf.template GetBufferByByte<uint64_t>(cacheOffset + cacheBytes + auxBytes);
        AscendC::LocalTensor<uint16_t> streamRaw[GbsaMxScaleCache::GROUPED_RAW_STAGES];
        AscendC::LocalTensor<uint16_t> streamPacked[GbsaMxScaleCache::NZ_STAGES];
        {
            for (uint32_t slot = 0U; slot < rawStages; ++slot) {
                streamRaw[slot] = resource.ubBuf.template GetBufferByByte<uint16_t>(cacheOffset + slot * rawBytes);
            }
            for (uint32_t slot = 0U; slot < GbsaMxScaleCache::NZ_STAGES; ++slot) {
                streamPacked[slot] = resource.ubBuf.template GetBufferByByte<uint16_t>(cacheOffset + cacheBytes +
                                                                                       slot * packedScaleBytes);
            }

            InitStreamScalePipeline(rawStages);
        }
        groupedRecordSlot_ = 0U;
        for (uint32_t slot = 0U; slot < GROUPED_RECORD_UB_STAGES; ++slot) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID5 + slot);
        }
#endif
        // AIV prepares the next task while AIC consumes the last two PV tiles.
        // Descriptor/P slots follow a continuous stream. AIV drains the old
        // output before starting the next softmax. Batch changes keep this stream live.
#ifdef __DAV_CUBE__
        auto& taskConsumer = pv;
#else
        auto& taskConsumer = rescale;
#endif
        // Metadata numbers every (batch-local Q token, KV head) in one
        // compact stream over all actual Q lengths. Each core owns task IDs
        // coreIdx, coreIdx + coreNum, ...; query tokens are never paired.
        const uint64_t taskCount = static_cast<uint32_t>(meta->saTotalTaskNum);
        const uint64_t tokenStride = originalTokenStride;
        AscendC::GlobalTensor<ElementQ> gQ;
        AscendC::GlobalTensor<ElementK> gK;
        AscendC::GlobalTensor<ElementV> gV;
        AscendC::GlobalTensor<ElementO> gO;
        AscendC::GlobalTensor<uint8_t> qScale, vScale;
        AscendC::GlobalTensor<int32_t> indices, counts;
        gQ.SetGlobalBuffer(reinterpret_cast<__gm__ ElementQ*>(params.q));
        gK.SetGlobalBuffer(reinterpret_cast<__gm__ ElementK*>(params.k));
        gV.SetGlobalBuffer(reinterpret_cast<__gm__ ElementV*>(params.v));
        gO.SetGlobalBuffer(reinterpret_cast<__gm__ ElementO*>(params.o));
        qScale.SetGlobalBuffer(params.qDequantScale);
        vScale.SetGlobalBuffer(params.vDequantScale);
        indices.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.sparseBlockIdx));
        counts.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.sparseBlockCount));
#ifdef __DAV_VEC__
        // Padding does not belong to the compact task stream. Clear it before
        // any live output accumulator can alias WriteEmptyOutput's scratch.
        if (params.sequsedQ != nullptr) {
            for (uint32_t batch = 0U; batch < batchCount; ++batch) {
                BatchSeqInfo seq{};
                ReadBatchSeqInfo<QFormat>(cuQ, usedQ, true, batch, seq, td->qSeqCapacity);
                for (uint64_t pad = static_cast<uint64_t>(seq.actualLen) * kvHeads_ + coreIdx;
                     pad < static_cast<uint64_t>(seq.storageLen) * kvHeads_; pad += coreNum) {
                    const uint64_t offset = QueryRowOffset(seq, pad / kvHeads_, (pad % kvHeads_) * groupSize_) * embed_;
                    auto out =
                        tla::MakeTensor(gO[offset], tla::MakeLayout<ElementO, LayoutO>(qBaseTile_, queryHeadStride),
                                        Arch::PositionGM{});
                    auto lse = tla::MakeTensor(gLse, tla::MakeLayout<ElementLse, LayoutLse>(qBaseTile_, 1),
                                               Arch::PositionGM{});
                    rescale.WriteEmptyOutput(out, lse, GemmCoord{groupSize_, embed_, 0});
                }
            }
        }
#endif
        PendingTile pending[RING];
        uint64_t produced = 0U;
        uint64_t consumed = 0U;
        // FD owns contiguous sparse-list ranges, possibly crossing base tasks.
        // Drain each partial before reusing its accumulator. Keep the prefill
        // cross-task pipeline unchanged when metadata does not select FD.
        const bool taskPipeline = !fdEnabled && static_cast<uint64_t>(coreIdx) + coreNum < taskCount;
        uint64_t taskLoopStart = coreIdx;
        uint64_t taskLoopEnd = taskCount;
        uint32_t taskLoopStep = coreNum;
        uint32_t scheduleFirstBlock = 0U;
        uint32_t scheduleLastBlock = 0U;
        if (fdEnabled) {
            taskLoopStart = taskCount;
            taskLoopStep = 1U;
            if (coreIdx < static_cast<uint32_t>(meta->fdActiveCoreNum)) {
                const __gm__ GbsaMetadata::DecodeSchedule& schedule = meta->decodeSchedules[coreIdx];
                taskLoopStart = static_cast<uint32_t>(schedule.baseTaskStart);
                taskLoopEnd = static_cast<uint32_t>(schedule.baseTaskEnd);
                scheduleFirstBlock = static_cast<uint32_t>(schedule.firstBlockStart);
                scheduleLastBlock = static_cast<uint32_t>(schedule.lastBlockEnd);
            }
        }
        uint32_t batch = 0U;
        uint64_t qLogicalStart = 0U;
        BatchSeqInfo qSeq{}, kvSeq{};
        if (batchCount != 0U) {
            ReadBatchSeqInfo<QFormat>(cuQ, usedQ, params.sequsedQ != nullptr, batch, qSeq, td->qSeqCapacity);
            ReadBatchSeqInfo<KvFormat>(cuKv, usedKv, params.sequsedKv != nullptr, batch, kvSeq, td->kvSeqCapacity);
        }
        uint32_t qLen = qSeq.actualLen;
        for (uint64_t task = taskLoopStart; task < taskLoopEnd; task += taskLoopStep) {
            uint32_t fdPartialTaskId = 0U;
            uint32_t fdPartialCount = 0U;
            const bool isFdPartial =
                fdEnabled && GsaFd::FindPartialTask(meta, task, coreIdx, fdPartialTaskId, fdPartialCount);
#ifdef __DAV_CUBE__
            while (taskPipeline && consumed < produced) {
                // Old PVs use the absolute addresses saved in pending, while
                // AIV prepares a new task that may belong to another batch.
                ConsumePendingTile(pending[consumed % RING], consumed, produced, taskConsumer, gV, gO, gLse, vScale,
                                   ubO, tokenStride, queryHeadStride);
                ++consumed;
            }
#endif
            const uint64_t globalQToken = task / kvHeads_;
            // Task IDs are monotonic on a core, so each batch prefix is visited
            // once. Empty batches advance without taking a queue position.
            while (batch < batchCount && globalQToken >= qLogicalStart + qLen) {
                qLogicalStart += qLen;
                ++batch;
                if (batch < batchCount) {
                    ReadBatchSeqInfo<QFormat>(cuQ, usedQ, params.sequsedQ != nullptr, batch, qSeq, td->qSeqCapacity);
                    ReadBatchSeqInfo<KvFormat>(cuKv, usedKv, params.sequsedKv != nullptr, batch, kvSeq,
                                               td->kvSeqCapacity);
                    qLen = qSeq.actualLen;
                }
            }
            if (batch >= batchCount) {
                break;
            }
            const uint32_t token = static_cast<uint32_t>(globalQToken - qLogicalStart);
            const uint32_t head = task % kvHeads_;
            const uint32_t kvLen = kvSeq.actualLen;
            // Metadata's task prefix counts actual tokens; sparse rows and tensor
            // addresses retain the physical cu prefix, including unused padding.
            const uint64_t qStorageToken = qSeq.storageStart + token;
            const uint64_t qHeadOffset = QueryRowOffset(qSeq, token, head * groupSize_);
            const uint64_t qOffset = qHeadOffset * embed_;
            const uint64_t kvHeadOffset =
                IsPaged ? static_cast<uint64_t>(head) * (KvHeadMajor ? td->blockSize : 1U) * embed_ :
                          GetTensorOffset<KvFormat>(kvSeq, 0U, head, kvHeads_, embed_);
            // Dense TND VScale is compact [B,Nkv,D,2], independent of cuKv.
            uint64_t vScaleOffset = (static_cast<uint64_t>(batch) * kvHeads_ + head) * embed_ * 2U;
            const uint64_t scaleBase = IsPaged ?
                                           (KvHeadMajor ? static_cast<uint64_t>(head) * td->blockSize * groups64 : 0U) :
                                       KvHeadMajor ? GetTensorOffset<KvFormat>(kvSeq, 0U, head, kvHeads_, groups64) :
                                                     kvSeq.storageStart * kvHeads_ * groups64;
            const uint64_t sparseRow = static_cast<uint64_t>(head) * qBlockNum_ + qStorageToken;
            const uint32_t visibleEnd = causal ? (kvLen < qLen ? 0U : kvLen - qLen + token + 1U) : kvLen;
            const uint32_t fullCount = min(static_cast<uint32_t>(counts.GetValue(sparseRow)), topK_);
            const uint32_t rawBegin = fdEnabled && task == taskLoopStart ? min(scheduleFirstBlock, fullCount) : 0U;
            const uint32_t rawEnd =
                fdEnabled && task + 1U == taskLoopEnd ? min(scheduleLastBlock, fullCount) : fullCount;
            const uint32_t count = rawEnd - rawBegin;
            // Capacity remains the physical row stride. Only the effective
            // sparse prefix is rebased; K/V and scale addresses remain global.
            Cursor cursor{indices[sparseRow * topK_ + rawBegin], count, blockShapeY_, visibleEnd, 0, 0};
            // The first count indices are a strictly increasing, unique prefix
            // inside this task's visible range. Only its last block can be partial.
            // AIC and both AIVs decode the same immutable inputs independently.
            uint64_t totalRows = 0U;
            bool directIndexMap = false;
            bool contiguousIndexMap = false;
            uint32_t firstToken = 0U;
            if (count != 0U && visibleEnd != 0U) {
                const uint32_t first = static_cast<uint32_t>(cursor.indices.GetValue(0U));
                const uint32_t last = static_cast<uint32_t>(cursor.indices.GetValue(count - 1U));
                totalRows = static_cast<uint64_t>(count) * blockShapeY_;
                if (visibleEnd % blockShapeY_ != 0U) {
                    const uint64_t lastStart = static_cast<uint64_t>(last) * blockShapeY_;
                    const uint64_t remaining = static_cast<uint64_t>(visibleEnd) - lastStart;
                    const uint64_t lastRows = remaining < blockShapeY_ ? remaining : blockShapeY_;
                    totalRows -= blockShapeY_ - lastRows;
                }
                // Preserve the bounded direct-position eligibility. Residency
                // is separate: grouped scalar readers do not need an index DMA.
                // Larger counts retain the cursor-based movement fallback.
                directIndexMap = GbsaMxScaleCache::DirectIndexMapFits(count);
                contiguousIndexMap = directIndexMap && last - first == count - 1U;
                firstToken = static_cast<uint64_t>(first) * blockShapeY_;
#ifdef __DAV_VEC__

#endif
            }
#ifdef __DAV_VEC__
            auto out = tla::MakeTensor(gO[qOffset], tla::MakeLayout<ElementO, LayoutO>(qBaseTile_, queryHeadStride),
                                       Arch::PositionGM{});
            auto lse = tla::MakeTensor(gLse, tla::MakeLayout<ElementLse, LayoutLse>(qBaseTile_, 1), Arch::PositionGM{});
#endif
            if (totalRows == 0U) {
                if (taskPipeline) {
                    // Empty-output scratch aliases the ordered O accumulator.
                    // Finish preceding consumers before clearing another task.
                    while (consumed < produced) {
                        ConsumePendingTile(pending[consumed % RING], consumed, produced, taskConsumer, gV, gO, gLse,
                                           vScale, ubO, tokenStride, queryHeadStride);
                        ++consumed;
                    }
                }
#ifdef __DAV_VEC__
                if (isFdPartial) {
                    rescale.WriteNeutralPartial(gPartialO, gPartialLse, fdPartialTaskId, groupSize_, embed_,
                                                td->fdLseSubStride);
                } else {
                    rescale.WriteEmptyOutput(out, lse, GemmCoord{groupSize_, embed_, 0});
                }
#endif
                continue;
            }
            if constexpr (IsPaged) {
                // V is per-channel: every page/group in this logical sequence
                // repeats the same scale. Resolve its first page only for live tasks.
                AscendC::GlobalTensor<int32_t> pages;
                pages.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.blockTable) +
                                      static_cast<uint64_t>(batch) * td->maxBlocksPerBatch);
                const uint64_t firstPage = static_cast<uint32_t>(pages.GetValue(0U));
                vScaleOffset = (firstPage * kvHeads_ * 2U + head * (KvHeadMajor ? 2U : 1U)) * embed_ * 2U;
#ifdef __DAV_VEC__
                batchPages_ = pages;
                pageStride_ = td->kStride0;
#endif
            }
            if (contiguousIndexMap) {
                cursor.contiguousBegin = firstToken;
                cursor.contiguousRows = static_cast<uint32_t>(totalRows);
            }
            const uint32_t computeRows = groupSize_;
            const uint32_t qkRows = groupSize_;
            const uint64_t tiles = (totalRows + kvBaseTile_ - 1U) / kvBaseTile_;
            const uint64_t taskBegin = taskPipeline ? produced : 0U;
#ifdef __DAV_VEC__
            rescale.SetMxVScale(vScale[vScaleOffset]);
            for (uint32_t prepare = 0U; prepare < DESCRIPTOR_AHEAD && prepare < tiles; ++prepare) {
                const uint32_t descriptorSlot = (taskBegin + prepare) % DESCRIPTOR_STAGES;
                const uint32_t futureRows = static_cast<uint32_t>(
                    totalRows - prepare * kvBaseTile_ < kvBaseTile_ ? totalRows - prepare * kvBaseTile_ : kvBaseTile_);
                const uint32_t rawSlot = GroupedRawSlot(taskBegin + prepare, rawStages);
                Cursor future = cursor;
                if (directIndexMap) {
                    future.raw = prepare * kvBaseTile_ / blockShapeY_;
                    future.inner = 0U;
                } else {
                    future.Advance(prepare * kvBaseTile_);
                }
                PrepareGroupedTile(future, recordsUb, descriptors[descriptorSlot * descriptorTileWords],
                                   kScale[scaleBase], streamRaw[rawSlot], futureRows, head, prepare * kvBaseTile_,
                                   tokenStride, descriptorSlot, rawSlot, directIndexMap, streamPacked,
                                   l1Scale[(taskBegin + prepare) % RING], prepare == 0U);
            }
#endif
#ifdef __DAV_CUBE__
            qk.SetQueryTileCount(max(tiles, static_cast<uint64_t>(2U)));
            qk.SetMxGeometry(embed_, queryScaleHeadStride, KvHeadMajor ? groups64 : kvHeads_ * groups64);
            qk.SetMxQScale(qScale[qHeadOffset * groups64 * 2U]);
            auto query = tla::MakeTensor(gQ[qOffset], tla::MakeLayout<ElementQ, LayoutQ>(qBaseTile_, queryHeadStride),
                                         Arch::PositionGM{});
            qk.loadQGM(query, GemmCoord{groupSize_, embed_, 0});
            auto key = tla::MakeTensor(gK[kvHeadOffset], tla::MakeLayout<ElementK, LayoutK>(tokenStride, kvBaseTile_),
                                       Arch::PositionGM{});
            auto value = tla::MakeTensor(gV[kvHeadOffset], tla::MakeLayout<ElementV, LayoutV>(kvBaseTile_, tokenStride),
                                         Arch::PositionGM{});
#endif
#ifdef __DAV_CUBE__
            Cursor nextCursor;
#endif
#ifdef __DAV_VEC__

            while (taskPipeline && consumed < taskBegin) {
                // Finish the preceding output only after submitting the
                // next task's descriptor/scale warmup. This also protects
                // the original online-softmax statistics before their reset.
                ConsumePendingTile(pending[consumed % RING], consumed, produced, taskConsumer, gV, gO, gLse, vScale,
                                   ubO, tokenStride, queryHeadStride);
                ++consumed;
            }
#endif
            for (uint64_t i = 0; i < tiles + (taskPipeline ? 0U : PRE_LAUNCH); ++i) {
                const uint64_t sequence = taskBegin + i;
                if (i < tiles) {
                    const uint32_t ringSlot = sequence % RING;
                    const uint32_t rows = static_cast<uint32_t>(
                        totalRows - i * kvBaseTile_ < kvBaseTile_ ? totalRows - i * kvBaseTile_ : kvBaseTile_);
#ifdef __DAV_CUBE__
                    pending[ringSlot].cursor = qk.HasPrefetched() ? nextCursor : cursor;
#else
                    pending[ringSlot].cursor = cursor;
#endif
                    pending[ringSlot].rows = rows;
                    if (directIndexMap) {
                        // Every grouped tile is divisible by the supported sparse block sizes.
                        cursor.raw = static_cast<uint32_t>((i + 1U) * kvBaseTile_ / blockShapeY_);
                        cursor.inner = 0U;
                    } else {
                        cursor.Advance(rows);
                    }
                    const uint32_t pSlot = sequence % pL1BufNum_;
                    const uint32_t sSlot = sequence % UB_STAGES;
                    Arch::CrossCoreFlag qkReady(sSlot), pReady(pSlot + UB_STAGES);
                    GemmCoord qkShape{qkRows, rows, embed_};
                    auto score = tla::MakeTensor(
                        ubS[sSlot], tla::MakeLayout<ElementS, LayoutS>(RoundUp(qkRows, 16U), RoundUp(rows, 16U)),
                        Arch::PositionUB{});
#ifdef __DAV_VEC__
                    {
                        // Fused descriptors have already submitted this Raw tile.
                        // Three Raw slots let prepare(i+2) use a different slot
                        // while the vector gathers of tile i are still in flight.
                        const uint32_t rawSlot = GroupedRawSlot(sequence, rawStages);
                        // Tile 0 was published during descriptor warmup.
                        if (i != 0U) {
                            PublishGroupedTile(rows, streamRaw[rawSlot], streamPacked, rawSlot, l1Scale[ringSlot]);
                        }
                    }
                    // Publish before either AIV waits for descriptor(i+2).
                    // Four slots let Cube finish QK(i), PV(i-2) and release the
                    // descriptor(i+2) slot without requiring softmax(i) on V1.
                    if (i + DESCRIPTOR_AHEAD < tiles) {
                        const uint64_t prepare = i + DESCRIPTOR_AHEAD;
                        const uint32_t descriptorSlot = (taskBegin + prepare) % DESCRIPTOR_STAGES;
                        const uint32_t futureRows = static_cast<uint32_t>(
                            totalRows - prepare * kvBaseTile_ < kvBaseTile_ ? totalRows - prepare * kvBaseTile_ :
                                                                              kvBaseTile_);
                        const uint32_t rawSlot = GroupedRawSlot(taskBegin + prepare, rawStages);
                        Cursor future = cursor;
                        if (directIndexMap) {
                            future.raw = prepare * kvBaseTile_ / blockShapeY_;
                            future.inner = 0U;
                        } else {
                            future.Advance(kvBaseTile_);
                        }
                        PrepareGroupedTile(future, recordsUb, descriptors[descriptorSlot * descriptorTileWords],
                                           kScale[scaleBase], streamRaw[rawSlot], futureRows, head,
                                           prepare * kvBaseTile_, tokenStride, descriptorSlot, rawSlot, directIndexMap,
                                           streamPacked, l1Scale[(taskBegin + prepare) % RING], false);
                    }
#endif
#ifdef __DAV_CUBE__
                    if (!qk.HasPrefetched()) {
                        const uint32_t descriptorSlot = sequence % DESCRIPTOR_STAGES;
                        AscendC::CrossCoreWaitFlag<4, PIPE_S>(DescriptorFlag(descriptorSlot));
                        AscendC::CrossCoreWaitFlag<4, PIPE_S>(DescriptorFlag(descriptorSlot) - 16U);
                        auto info = descriptors[descriptorSlot * descriptorTileWords];
                        pending[ringSlot].cursor.packedRecords = info.GetPhyAddr();
                        pending[ringSlot].cursor.packedRecordCount = 0U;
                        pending[ringSlot].cursor.packedRows = rows;
                        pending[ringSlot].cursor.packedGroupWords = GbsaMxScaleCache::DescriptorGroupWords();
                    }

                    qk.SetMxPacked(pending[ringSlot].cursor, l1Scale[ringSlot]);
                    qk(key, score, unusedIndices, qkShape, i % 6U, kvBaseTile_, blockShapeY_,
                       Prefix(i, tiles, mm1L0ATotalStages_, mm2L0ATotalStages_, true),
                       Prefix(i, tiles, mm1L0BTotalStages_, mm2L0BTotalStages_, true), qkReady, tokenStride);

                    if (i + 1U == tiles) {
                        qk.ReleaseQuery();
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
                    }
#endif
                    auto prob = tla::MakeTensor(l1P[pSlot],
                                                tla::MakeLayout<ElementP, layout::zN>(computeRows, RoundUp(rows, 64U)),
                                                Arch::PositionL1{});
#ifdef __DAV_VEC__
                    softmax(prob, GemmCoord{computeRows, rows, embed_}, i == 0U, sSlot, pSlot, qkReady, pReady,
                            RoundUp(rows, 64U));
#endif
                    if (taskPipeline) {
                        auto& item = pending[ringSlot];
                        item.outputOffset = qOffset;
                        item.rows = rows;
                        item.computeRows = computeRows;
                        item.valueOffset = kvHeadOffset;
                        item.valueScaleOffset = vScaleOffset;
                        item.first = i == 0U;
                        item.last = i + 1U == tiles;
                        ++produced;
                    }
                }
                if (taskPipeline && produced - consumed > PRE_LAUNCH) {
                    ConsumePendingTile(pending[consumed % RING], consumed, produced, taskConsumer, gV, gO, gLse, vScale,
                                       ubO, tokenStride, queryHeadStride);
                    ++consumed;
                } else if (!taskPipeline && i >= PRE_LAUNCH) {
                    const uint64_t done = i - PRE_LAUNCH;
                    const uint32_t pSlot = done % pL1BufNum_;
                    const uint32_t oSlot = done % uBufTileHelper_.loUbBufNum;
                    Arch::CrossCoreFlag pReady(pSlot + UB_STAGES), oReady(oSlot + UB_STAGES + pL1BufNum_);
                    GemmCoord pvShape{computeRows, embed_, pending[done % RING].rows};
#ifdef __DAV_CUBE__
                    pv.SetMxPacked(pending[done % RING].cursor);
                    auto partial = tla::MakeTensor(
                        ubO[oSlot], tla::MakeLayout<ElementOTmp, LayoutOTmp>(RoundUp(computeRows, 16U), embed_),
                        Arch::PositionUB{});
                    pv(value, partial, unusedIndices, pvShape, done % 6U, kvBaseTile_, blockShapeY_,
                       Prefix(done, tiles, mm1L0ATotalStages_, mm2L0ATotalStages_, false),
                       Prefix(done, tiles, mm1L0BTotalStages_, mm2L0BTotalStages_, false), pReady, oReady, tokenStride);
                    // PV has now consumed every scalar descriptor field.
                    // K/V data DMA no longer reads descriptor workspace.
                    AscendC::CrossCoreSetFlag<4, PIPE_S>(DescriptorFlag(done % DESCRIPTOR_STAGES));
                    AscendC::CrossCoreSetFlag<4, PIPE_S>(DescriptorFlag(done % DESCRIPTOR_STAGES) - 16U);
#endif
#ifdef __DAV_VEC__
                    if (isFdPartial) {
                        rescale.ProcessPartial(out, lse, pvShape, done % RING, done % 6U, done == 0U,
                                               done + 1U == tiles, oReady, gPartialO, gPartialLse, fdPartialTaskId,
                                               td->fdLseSubStride);
                    } else {
                        rescale(out, lse, pvShape, done % RING, done % 6U, done == 0U, done + 1U == tiles, oReady);
                    }
#endif
                }
#ifdef __DAV_CUBE__
                // Submit K(i+1) after current pending PV has submitted its V
                // copy. The descriptor is already produced by both AIVs at i-1, so
                // this wait never depends on softmax(i+1). PV keeps ownership
                // until its existing descriptor release; next QK skips wait.
                if (i + 1U < tiles) {
                    const uint64_t next = i + 1U;
                    const uint32_t nextRows = min(static_cast<uint64_t>(kvBaseTile_), totalRows - next * kvBaseTile_);
                    nextCursor = cursor;
                    const uint32_t slot = (taskBegin + next) % DESCRIPTOR_STAGES;
                    AscendC::CrossCoreWaitFlag<4, PIPE_S>(DescriptorFlag(slot));
                    AscendC::CrossCoreWaitFlag<4, PIPE_S>(DescriptorFlag(slot) - 16U);
                    auto info = descriptors[slot * descriptorTileWords];
                    nextCursor.packedRecords = info.GetPhyAddr();
                    nextCursor.packedRecordCount = 0U;
                    nextCursor.packedRows = nextRows;
                    nextCursor.packedGroupWords = GbsaMxScaleCache::DescriptorGroupWords();
                    qk.Preload(key, nextCursor, nextRows, embed_, next % 6U, kvBaseTile_, blockShapeY_, tokenStride);
                }
#endif
            }
        }
        // Only the end of this core's complete task stream drains its tail.
        if (taskPipeline) {
            while (consumed < produced) {
                ConsumePendingTile(pending[consumed % RING], consumed, produced, taskConsumer, gV, gO, gLse, vScale,
                                   ubO, tokenStride, queryHeadStride);
                ++consumed;
            }
        }
#ifdef __DAV_VEC__
        for (uint32_t i = 0; i < scaleCredits; ++i) {
            AscendC::CrossCoreWaitFlag<4, PIPE_MTE3>(SCALE_FLAG + i);
        }
        ReleaseStreamScalePipeline(rawStages);
        for (uint32_t i = 0; i < DESCRIPTOR_STAGES; ++i) {
            AscendC::CrossCoreWaitFlag<4, PIPE_S>(DescriptorFlag(i));
        }
        const uint32_t recordSlots = GROUPED_RECORD_UB_STAGES;
        for (uint32_t slot = 0U; slot < recordSlots; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID5 + slot);
        }
#endif
        Base::template ReleaseSyncFlags<4, 4, 4>();
        if (fdEnabled) {
            // Inactive FD cores must also reach this barrier. All partial GM
            // writes finish before the combine reuses the drained UB buffers.
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::SyncAll<false>();
#ifdef __DAV_VEC__
            GenericBlockSparseAttentionFdCombineArch35<ElementO, Arch::Resource<ArchTag>, false, QFormat> combine(
                resource);
            combine(meta, td, gPartialLse, gPartialO, gO, gLse, cuQ, usedQ, params.sequsedQ != nullptr);
#endif
        }
    }
};
} // namespace GbsaKernelArch35
#endif
