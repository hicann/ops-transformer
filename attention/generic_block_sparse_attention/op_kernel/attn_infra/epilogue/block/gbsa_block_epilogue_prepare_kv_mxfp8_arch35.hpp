/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_BLOCK_EPILOGUE_PREPARE_KV_MXFP8_ARCH35_HPP
#define GBSA_BLOCK_EPILOGUE_PREPARE_KV_MXFP8_ARCH35_HPP

#include "gbsa_block_epilogue_arch35_utils.hpp"
#include "../../gemm/tile_common/gbsa_mxfp8_tile.hpp"
#include "../../gemm/tile_common/gbsa_build_kv_records_mxfp8.hpp"

namespace NpuArch::Epilogue::Block {

namespace Mxfp8ScaleLayout {
constexpr uint32_t GROUP_TOKENS = Gemm::Tile::Mxfp8DescriptorLayout::GROUP_TOKENS;
constexpr uint32_t GROUPS_PER_AIV = 2U;
constexpr uint32_t RAW_STAGES = 2U;
constexpr uint32_t GROUPED_RAW_STAGES = 3U;
constexpr uint32_t NZ_STAGES = 2U;
constexpr uint32_t RAW_ROW_BYTES = 32U;

__aicore__ inline uint64_t NzScaleUbSlotBytes(uint64_t dim)
{
    return Mxfp8UbLayout::Align(GROUP_TOKENS * (Mxfp8UbLayout::BufferDim(dim) / 32U), 256U);
}

__aicore__ inline uint32_t GroupsPerAiv(uint64_t tile)
{
    // A 128-token tile still gives AIV1 one empty notification to balance READY/FREE.
    return static_cast<uint32_t>((tile / GROUP_TOKENS + 1U) / 2U);
}

__aicore__ inline uint64_t RawTileTokens(uint64_t tile)
{
    return GroupsPerAiv(tile) * GROUP_TOKENS;
}

__aicore__ inline uint32_t RawBlockRows(uint64_t blockY, bool headMajor)
{
    return headMajor ? static_cast<uint32_t>(blockY < 16U ? blockY : 16U) : 1U;
}

__aicore__ inline uint64_t RawCapacityBytes(uint64_t tokens)
{
    // Both layouts reserve 4 KiB per 128-token group. Each AIV tile slot
    // contains two groups for KV512 and one group for KV256/KV128.
    // BNSD keeps its compact payload at the beginning of each group region.
    return tokens * RAW_ROW_BYTES;
}

__aicore__ inline uint64_t PipelineAuxBytes(uint64_t dim)
{
    return NZ_STAGES * NzScaleUbSlotBytes(dim);
}
} // namespace Mxfp8ScaleLayout

// AIV producer for sparse K/V descriptors and current-head KScale. The kernel
// owns buffer allocation and scheduling; this component consumes those buffer
// views and keeps the record rotation and page context across tasks.
template <class Config>
class GbsaMxfp8KvProducer {
    static constexpr bool KvHeadMajor = Config::KV_HEAD_MAJOR;
    static constexpr bool IsPaged = Config::IS_PAGED;

public:
    static constexpr uint32_t DESCRIPTOR_FLAG = 7U;
    static constexpr uint32_t DESCRIPTOR_STAGES = Gemm::Tile::Mxfp8DescriptorLayout::RING_SLOTS;
    static constexpr uint32_t DESCRIPTOR_AHEAD = 2U;
    static constexpr uint32_t GROUPED_RECORD_UB_STAGES = 2U;
    static constexpr uint32_t SCALE_FLAG = 10U;

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

#ifdef __DAV_VEC__
    __aicore__ inline GbsaMxfp8KvProducer(const uint32_t& embed, const uint32_t& kvHeads, const uint32_t& blockShapeY,
                                          const uint32_t& kvBaseTile)
        : embed_(embed),
          kvHeads_(kvHeads),
          blockShapeY_(blockShapeY),
          kvBaseTile_(kvBaseTile)
    {}

    __aicore__ inline void SetPageTable(AscendC::GlobalTensor<int32_t> pages, uint64_t pageStride)
    {
        batchPages_ = pages;
        pageStride_ = pageStride;
    }

    __aicore__ inline void InitRecords()
    {
        groupedRecordSlot_ = 0U;
        for (uint32_t slot = 0U; slot < GROUPED_RECORD_UB_STAGES; ++slot) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID5 + slot);
        }
    }

private:
    using Cursor = Gemm::Tile::GbsaMxPackedCursor;

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
                Muls(offset, tokenIndex, uint16_t(Mxfp8ScaleLayout::RAW_ROW_BYTES / sizeof(uint16_t)), valid);
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
        static_assert(Mxfp8ScaleLayout::GROUPED_RAW_STAGES * Mxfp8ScaleLayout::GROUPS_PER_AIV <= 7U,
                      "Raw group readiness must leave an event for VScale");
        const uint32_t event = slot * Mxfp8ScaleLayout::GROUPS_PER_AIV + local;
        return event < EVENT_ID2 ? event : event + 1U;
    }

public:
    __aicore__ inline static uint32_t GroupedRawSlot(uint64_t sequence, uint32_t rawStages)
    {
        return rawStages == Mxfp8ScaleLayout::GROUPED_RAW_STAGES ? sequence % Mxfp8ScaleLayout::GROUPED_RAW_STAGES :
                                                                   sequence & 1U;
    }

private:
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
                        raw[(outputRows + matrix * rows) * (Mxfp8ScaleLayout::RAW_ROW_BYTES / sizeof(uint16_t))],
                        scales[(offset + matrix * stride) / 64U], rows, rowPairs, headPairs);
                }
            }
            outputRows += matrices * rows;
        }
    };

public:
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
        constexpr uint32_t GROUP = Mxfp8ScaleLayout::GROUP_TOKENS;
        const uint32_t lane = AscendC::GetSubBlockIdx();
        constexpr uint32_t groupWords = 8U + 2U * GROUP;
        const uint32_t localGroups = Mxfp8ScaleLayout::GroupsPerAiv(kvBaseTile_);
        const uint32_t headPairs = embed_ / 64U;
        const uint32_t rowPairs = KvHeadMajor ? headPairs : kvHeads_ * headPairs;
        const auto headScales = scales[KvHeadMajor ? 0U : head * headPairs];
        const uint32_t blockRows = Mxfp8ScaleLayout::RawBlockRows(blockShapeY_, KvHeadMajor);
        const uint32_t rawGroupPairs = Mxfp8ScaleLayout::RawCapacityBytes(GROUP) / sizeof(uint16_t);
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

private:
    __aicore__ inline void PublishGroupedScale(uint32_t rows, AscendC::LocalTensor<uint16_t> raw,
                                               AscendC::LocalTensor<uint16_t> packed, uint32_t rawSlot,
                                               AscendC::LocalTensor<uint8_t> l1Scale, uint32_t local)
    {
        constexpr uint32_t GROUP = Mxfp8ScaleLayout::GROUP_TOKENS;
        const uint32_t groups = embed_ / 64U;
        const uint32_t rawGroupPairs = Mxfp8ScaleLayout::RawCapacityBytes(GROUP) / sizeof(uint16_t);
        const uint32_t blockRows = Mxfp8ScaleLayout::RawBlockRows(blockShapeY_, KvHeadMajor);
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

public:
    __aicore__ inline void PublishGroupedTile(uint32_t rows, AscendC::LocalTensor<uint16_t> raw,
                                              AscendC::LocalTensor<uint16_t>* packed, uint32_t rawSlot,
                                              AscendC::LocalTensor<uint8_t> l1Scale)
    {
        for (uint32_t local = 0U; local < Mxfp8ScaleLayout::GroupsPerAiv(kvBaseTile_); ++local) {
            PublishGroupedScale(rows, raw, packed[local], rawSlot, l1Scale, local);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_S>(StreamRawFreeEvent(rawSlot));
    }

    __aicore__ inline void InitStreamScalePipeline(uint32_t rawStages)
    {
        for (uint32_t slot = 0U; slot < rawStages; ++slot) {
            AscendC::SetFlag<AscendC::HardEvent::V_S>(StreamRawFreeEvent(slot));
        }
        for (uint32_t slot = 0U; slot < Mxfp8ScaleLayout::NZ_STAGES; ++slot) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID5 + slot);
        }
    }

    __aicore__ inline void ReleaseStreamScalePipeline(uint32_t rawStages)
    {
        for (uint32_t slot = 0U; slot < rawStages; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(StreamRawFreeEvent(slot));
        }
        for (uint32_t slot = 0U; slot < Mxfp8ScaleLayout::NZ_STAGES; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID5 + slot);
        }
    }

private:
    // Non-owning geometry references stay valid for the enclosing kernel's Run().
    const uint32_t& embed_;
    const uint32_t& kvHeads_;
    const uint32_t& blockShapeY_;
    const uint32_t& kvBaseTile_;
    uint32_t groupedRecordSlot_ = 0U;
    AscendC::GlobalTensor<int32_t> batchPages_;
    uint64_t pageStride_ = 0U;
#endif
};

} // namespace NpuArch::Epilogue::Block

#endif // GBSA_BLOCK_EPILOGUE_PREPARE_KV_MXFP8_ARCH35_HPP
