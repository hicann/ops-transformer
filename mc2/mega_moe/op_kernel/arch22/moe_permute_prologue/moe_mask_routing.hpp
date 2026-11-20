/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MC2_MOE_KBIT_MASK_ROUTING_HPP
#define MC2_MOE_KBIT_MASK_ROUTING_HPP

#include "kernel_operator.h"
#include "moe_permute_prologue_tiling.h"
#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"

namespace MoePermute {

namespace MaskLayout = Mc2Tiling::MegaMoeMask;

struct MaskUbLayout {
    static constexpr uint32_t MAX_TOKENS = Mc2Tiling::MEGA_MOE_PERMUTE_CHUNK;
    static constexpr uint32_t MASK_BATCH = 8; // 单批专家数 = 计数块长度
    static constexpr uint32_t MAX_SLOTS = MAX_TOKENS * MaskLayout::MAX_TOPK;

    static constexpr uint32_t IDS_OFFSET = 0;
    static constexpr uint32_t IDS_BYTES = MAX_SLOTS * sizeof(int32_t);
    static constexpr uint32_t FLOATS_OFFSET = IDS_OFFSET + IDS_BYTES;
    static constexpr uint32_t FLOATS_BYTES = MAX_SLOTS * sizeof(float);
    static constexpr uint32_t IOTA_OFFSET = FLOATS_OFFSET + FLOATS_BYTES;
    static constexpr uint32_t IOTA_BYTES = MAX_SLOTS * sizeof(int32_t);
    static constexpr uint32_t SELECTED_OFFSET = IOTA_OFFSET + IOTA_BYTES;
    static constexpr uint32_t SELECTED_BYTES = MAX_SLOTS * sizeof(int32_t);
    static constexpr uint32_t BITS_OFFSET = SELECTED_OFFSET + SELECTED_BYTES;
    static constexpr uint32_t BITS_BYTES = MASK_BATCH * MAX_TOKENS;
    static constexpr uint32_t COUNTS_OFFSET = BITS_OFFSET + BITS_BYTES;
    static constexpr uint32_t COUNTS_BYTES = MASK_BATCH * sizeof(int32_t);
    static constexpr uint32_t ACTIVE_OFFSET = COUNTS_OFFSET + COUNTS_BYTES;
    static constexpr uint32_t ACTIVE_BYTES = MAX_TOKENS;
    static constexpr uint32_t ACTIVE_HALF_OFFSET = ACTIVE_OFFSET + ACTIVE_BYTES;
    static constexpr uint32_t ACTIVE_HALF_BYTES = MAX_TOKENS * sizeof(half);
    static constexpr uint32_t ACTIVE_FLOAT_OFFSET = ACTIVE_HALF_OFFSET + ACTIVE_HALF_BYTES;
    static constexpr uint32_t ACTIVE_FLOAT_BYTES = MAX_TOKENS * sizeof(float);
    static constexpr uint32_t COMBINE_INDICES_OFFSET = 144 * 1024;
    static constexpr uint32_t COMBINE_INDICES_BYTES = 128 * sizeof(int32_t);
    static constexpr uint32_t SEND_MASKS_OFFSET = COMBINE_INDICES_OFFSET + COMBINE_INDICES_BYTES;
    static constexpr uint32_t SEND_MASKS_BYTES = MASK_BATCH * MAX_TOKENS;
    static constexpr uint32_t MASK_BLOCK_END = SEND_MASKS_OFFSET + SEND_MASKS_BYTES;
    static constexpr uint32_t ROUTE_CACHE_FLOOR = 174 * 1024;

    static_assert(ACTIVE_FLOAT_OFFSET + ACTIVE_FLOAT_BYTES <= COMBINE_INDICES_OFFSET,
                  "active-mask scratch overlaps the combine index buffer");
    static_assert(MASK_BLOCK_END <= ROUTE_CACHE_FLOOR, "mask routing scratch overlaps the route cache");
};

template <class ArchTag>
class MoeMaskRouting {
public:
    static constexpr uint32_t MAX_TOKENS = MaskUbLayout::MAX_TOKENS;
    static constexpr uint32_t MAX_TOPK = MaskLayout::MAX_TOPK;
    static constexpr uint32_t MASK_BATCH = MaskUbLayout::MASK_BATCH;
    static constexpr uint32_t MAX_SLOTS = MaskUbLayout::MAX_SLOTS;
    static constexpr uint32_t SLOT_ALIGN = MaskLayout::SLOT_ALIGN;
    static constexpr uint32_t VEC_ALIGN = 256;

    static_assert(MAX_TOPK == 8, "mask packing uses one byte per token");
    static_assert(MaskLayout::MASK_STRIDE_ALIGN == Catlass::BYTE_PER_BLK);
    static_assert(MAX_TOKENS % MaskLayout::MASK_STRIDE_ALIGN == 0 && MAX_SLOTS % VEC_ALIGN == 0);

    CATLASS_DEVICE
    MoeMaskRouting(Catlass::Arch::Resource<ArchTag>& resource, uint32_t tokens, uint32_t topK, uint32_t experts,
                   uint32_t core, uint32_t cores)
        : tokens_(tokens),
          topK_(topK),
          experts_(experts),
          core_(core),
          cores_(cores),
          slots_(tokens * topK),
          maskStride_(RoundUp<Catlass::BYTE_PER_BLK>(tokens))
    {
        ids_ = resource.ubBuf.template GetBufferByByte<int32_t>(MaskUbLayout::IDS_OFFSET);
        floats_ = resource.ubBuf.template GetBufferByByte<float>(MaskUbLayout::FLOATS_OFFSET);
        iota_ = resource.ubBuf.template GetBufferByByte<int32_t>(MaskUbLayout::IOTA_OFFSET);
        selected_ = resource.ubBuf.template GetBufferByByte<int32_t>(MaskUbLayout::SELECTED_OFFSET);
        bits_ = resource.ubBuf.template GetBufferByByte<uint8_t>(MaskUbLayout::BITS_OFFSET);
        counts_ = resource.ubBuf.template GetBufferByByte<int32_t>(MaskUbLayout::COUNTS_OFFSET);
        active_ = resource.ubBuf.template GetBufferByByte<uint8_t>(MaskUbLayout::ACTIVE_OFFSET);
        activeHalf_ = resource.ubBuf.template GetBufferByByte<half>(MaskUbLayout::ACTIVE_HALF_OFFSET);
        activeFloat_ = resource.ubBuf.template GetBufferByByte<float>(MaskUbLayout::ACTIVE_FLOAT_OFFSET);
        sendMasks_ = resource.ubBuf.template GetBufferByByte<uint8_t>(MaskUbLayout::SEND_MASKS_OFFSET);
    }

    template <class PeerMemory>
    CATLASS_DEVICE void Generate(AscendC::GlobalTensor<int32_t> const& topkIds,
                                 AscendC::GlobalTensor<int32_t> const& counts, PeerMemory& peers, int64_t maskOffset,
                                 uint32_t expertsPerRank, AscendC::LocalTensor<int32_t> const& rankMask,
                                 bool hasRankMask, GM_ADDR activeMask)
    {
        if (core_ * MASK_BATCH >= experts_) {
            ResetMaskState();
            return;
        }

        const uint32_t lanes = RoundUp<VEC_ALIGN>(tokens_ * MAX_TOPK);
        LoadIdFloats(topkIds, lanes, activeMask);
        if (topK_ != MAX_TOPK) {
            ExpandPackedSlots(lanes);
        }
        if (activeMask != nullptr) {
            ApplyActiveMask(lanes);
        }

        const uint32_t epWorldSize = experts_ / expertsPerRank;
        for (uint32_t batchStart = core_ * MASK_BATCH; batchStart < experts_; batchStart += cores_ * MASK_BATCH) {
            EmitMaskBatch(peers, maskOffset, counts, expertsPerRank, epWorldSize, batchStart, lanes, rankMask,
                          hasRankMask);
        }
        ResetMaskState();
    }

    CATLASS_DEVICE void Expand(AscendC::GlobalTensor<uint8_t> const& wireMasks,
                               AscendC::GlobalTensor<int32_t> const& maskIndicesGm,
                               AscendC::LocalTensor<int32_t> const& routeCounts, uint32_t epWorldSize,
                               uint32_t expertsPerRank, uint32_t countStride)
    {
        const uint32_t lanes = RoundUp<VEC_ALIGN>(tokens_ * MAX_TOPK);
        const uint32_t slotStride = RoundUp<SLOT_ALIGN>(slots_);
        CreateSlotIndices(lanes);
        const uint32_t totalPairs = epWorldSize * expertsPerRank;
        for (uint32_t batchStart = core_; batchStart < totalPairs; batchStart += cores_ * MASK_BATCH) {
            const uint32_t remaining = (totalPairs - batchStart + cores_ - 1) / cores_;
            const uint32_t batchSize = remaining < MASK_BATCH ? remaining : MASK_BATCH;
            const uint32_t activePairs =
                LoadMasks(wireMasks, routeCounts, epWorldSize, countStride, batchStart, batchSize);
            if (activePairs == 0) {
                continue;
            }
            uint32_t bufferUsed = 0;
            for (uint32_t j = 0; j < batchSize; ++j) {
                if ((activePairs & (1U << j)) == 0) {
                    continue;
                }
                const uint32_t expertSourcePair = batchStart + j * cores_;
                const uint32_t reserved =
                    RoundUp<SLOT_ALIGN>(CountOfPair(routeCounts, expertSourcePair, epWorldSize, countStride));
                if (bufferUsed + reserved > MAX_SLOTS) {
                    Sync<AscendC::HardEvent::MTE3_V>();
                    bufferUsed = 0;
                }
                uint64_t count = 0;
                AscendC::GatherMask(selected_[bufferUsed], iota_,
                                    bits_[j * maskStride_].template ReinterpretCast<uint32_t>(), true, lanes,
                                    {1, 1, 0, 0}, count);
                Sync<AscendC::HardEvent::V_S>();
                Sync<AscendC::HardEvent::V_MTE3>();
                AscendC::DataCopyPad(maskIndicesGm[static_cast<uint64_t>(expertSourcePair) * slotStride],
                                     selected_[bufferUsed],
                                     {1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0});
                bufferUsed += reserved;
            }
            Sync<AscendC::HardEvent::MTE3_V>();
        }
        ResetMaskState();
    }

private:
    CATLASS_DEVICE static uint32_t CountOfPair(const AscendC::LocalTensor<int32_t>& routeCounts,
                                               uint32_t expertSourcePair, uint32_t epWorldSize, uint32_t countStride)
    {
        // expertSourcePair 按 [本地专家][来源卡] 编号，计数缓存按 [来源卡][本地专家] 排列。
        return static_cast<uint32_t>(
            routeCounts.GetValue((expertSourcePair % epWorldSize) * countStride + expertSourcePair / epWorldSize));
    }

    CATLASS_DEVICE static void ResetMaskState()
    {
        AscendC::SetMaskNorm();
        AscendC::ResetMask();
    }

    // 搬入本 chunk 的 topk_ids 与 x_active_mask，转成 float 供比较
    CATLASS_DEVICE void LoadIdFloats(AscendC::GlobalTensor<int32_t> const& topkIds, uint32_t lanes, GM_ADDR activeMask)
    {
        if (slots_ != lanes) {
            AscendC::Duplicate(ids_, -1, lanes);
        }
        Sync<AscendC::HardEvent::V_MTE2>();
        const uint32_t idBytes = static_cast<uint32_t>(slots_ * sizeof(int32_t));
        const uint32_t padSlots = (MAX_TOPK - slots_ % MAX_TOPK) % MAX_TOPK;
        AscendC::DataCopyPad(ids_, topkIds, {1, idBytes, 0, 0, 0},
                             {padSlots != 0, 0, static_cast<uint8_t>(padSlots), -1});
        if (activeMask != nullptr) {
            AscendC::GlobalTensor<uint8_t> active;
            active.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(activeMask));
            AscendC::DataCopyPad(active_, active, {1, tokens_, 0, 0, 0}, {});
        }
        Sync<AscendC::HardEvent::MTE2_V>();
        AscendC::Cast(floats_, ids_, AscendC::RoundMode::CAST_NONE, lanes);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // topK<8 时把紧凑槽位重排到 token*8+slot
    CATLASS_DEVICE void ExpandPackedSlots(uint32_t lanes)
    {
        auto offsets = iota_.template ReinterpretCast<uint32_t>();
        CreateSlotIndices(lanes);
        // Gather 使用字节偏移，元素下标乘以 sizeof(float) 后才能读取对应的专家 ID。
        AscendC::ShiftLeft(offsets, offsets, static_cast<uint32_t>(2), lanes);
        AscendC::PipeBarrier<PIPE_V>();
        auto gathered = selected_.template ReinterpretCast<float>();
        AscendC::Gather(gathered, floats_, offsets, static_cast<uint32_t>(0), lanes);
        AscendC::PipeBarrier<PIPE_V>();
        const uint32_t wordMask = ((1U << topK_) - 1U) * 0x01010101U;
        AscendC::Duplicate(bits_.template ReinterpretCast<uint32_t>(), wordMask, lanes / 32);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select(floats_, bits_, gathered, -1.0f, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, lanes);
        AscendC::PipeBarrier<PIPE_V>();
        Sync<AscendC::HardEvent::V_S>();
    }

    // 未激活 token 的槽位置 -1
    CATLASS_DEVICE void ApplyActiveMask(uint32_t lanes)
    {
        AscendC::Duplicate(activeFloat_, 0.0f, MAX_TOKENS);
        AscendC::Cast(activeHalf_, active_, AscendC::RoundMode::CAST_NONE, tokens_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(activeFloat_, activeHalf_, AscendC::RoundMode::CAST_NONE, tokens_);
        Sync<AscendC::HardEvent::V_S>();
        AscendC::CreateVecIndex(ids_, 0, lanes);
        AscendC::PipeBarrier<PIPE_V>();
        auto offsets = ids_.template ReinterpretCast<uint32_t>();
        AscendC::ShiftRight(offsets, offsets, static_cast<uint32_t>(3), lanes);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftLeft(offsets, offsets, static_cast<uint32_t>(2), lanes);
        AscendC::PipeBarrier<PIPE_V>();
        auto activeSlots = iota_.template ReinterpretCast<float>();
        AscendC::Gather(activeSlots, activeFloat_, offsets, static_cast<uint32_t>(0), lanes);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::CompareScalar(bits_, activeSlots, 0.0f, AscendC::CMPMODE::NE, lanes);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select(floats_, bits_, floats_, -1.0f, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, lanes);
        AscendC::PipeBarrier<PIPE_V>();
        Sync<AscendC::HardEvent::V_S>();
    }

    template <class PeerMemory>
    CATLASS_DEVICE void EmitMaskBatch(PeerMemory& peers, int64_t maskOffset,
                                      AscendC::GlobalTensor<int32_t> const& counts, uint32_t expertsPerRank,
                                      uint32_t epWorldSize, uint32_t batchStart, uint32_t lanes,
                                      AscendC::LocalTensor<int32_t> const& rankMask, bool hasRankMask)
    {
        for (uint32_t j = 0; j < MASK_BATCH; ++j) {
            uint64_t count = 0;
            const uint32_t expert = batchStart + j;
            if (expert < experts_) {
                auto expertMask = sendMasks_[j * maskStride_];
                AscendC::CompareScalar(expertMask, floats_, static_cast<float>(static_cast<int32_t>(expert)),
                                       AscendC::CMPMODE::EQ, lanes);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::GatherMask(selected_.template ReinterpretCast<float>(), floats_,
                                    expertMask.template ReinterpretCast<uint32_t>(), true, lanes, {1, 1, 0, 0}, count);
                Sync<AscendC::HardEvent::V_S>();
                const uint32_t dst = expert / expertsPerRank;
                const bool reachable = !hasRankMask || rankMask.GetValue(dst) == 0;
                if (count != 0 && reachable) {
                    Sync<AscendC::HardEvent::V_MTE3>();
                    AscendC::GlobalTensor<uint8_t> remote;
                    remote.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(peers(maskOffset, dst)));
                    const uint64_t offset =
                        (static_cast<uint64_t>(expert % expertsPerRank) * epWorldSize + peers.Rank()) * maskStride_;
                    AscendC::DataCopyPad(remote[offset], expertMask, {1, tokens_, 0, 0, 0});
                }
            }
            counts_.SetValue(j, static_cast<int32_t>(count));
        }
        Sync<AscendC::HardEvent::S_MTE3>();
        AscendC::DataCopy(counts[batchStart], counts_, MASK_BATCH);
        Sync<AscendC::HardEvent::MTE3_V>();
        Sync<AscendC::HardEvent::MTE3_S>();
    }

    // 将位图的 token * 8 + slot 坐标映射为紧凑路由下标 token * topK + slot。
    CATLASS_DEVICE void CreateSlotIndices(uint32_t lanes)
    {
        AscendC::CreateVecIndex(iota_, 0, lanes);
        AscendC::PipeBarrier<PIPE_V>();
        if (topK_ != MAX_TOPK) {
            AscendC::ShiftRight(ids_.template ReinterpretCast<uint32_t>(), iota_.template ReinterpretCast<uint32_t>(),
                                static_cast<uint32_t>(3), lanes);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(ids_, ids_, static_cast<int32_t>(topK_) - static_cast<int32_t>(MAX_TOPK), lanes);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(iota_, iota_, ids_, lanes);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    CATLASS_DEVICE uint32_t LoadMasks(AscendC::GlobalTensor<uint8_t> const& wireMasks,
                                      AscendC::LocalTensor<int32_t> const& routeCounts, uint32_t epWorldSize,
                                      uint32_t countStride, uint32_t batchStart, uint32_t batchSize)
    {
        uint32_t activePairs = 0;
        for (uint32_t j = 0; j < batchSize; ++j) {
            const uint32_t expertSourcePair = batchStart + j * cores_;
            if (CountOfPair(routeCounts, expertSourcePair, epWorldSize, countStride) == 0) {
                continue;
            }
            if (activePairs == 0) {
                Sync<AscendC::HardEvent::V_MTE2>();
            }
            activePairs |= 1U << j;
            AscendC::DataCopyPad(bits_[j * maskStride_],
                                 wireMasks[static_cast<uint64_t>(expertSourcePair) * maskStride_],
                                 {1, tokens_, 0, 0, 0}, {true, 0, static_cast<uint8_t>(maskStride_ - tokens_), 0});
        }
        if (activePairs != 0) {
            Sync<AscendC::HardEvent::MTE2_V>();
        }
        return activePairs;
    }

    template <AscendC::HardEvent EVENT>
    CATLASS_DEVICE static void Sync()
    {
        AscendC::SetFlag<EVENT>(EVENT_ID0);
        AscendC::WaitFlag<EVENT>(EVENT_ID0);
    }

    uint32_t tokens_, topK_, experts_, core_, cores_, slots_, maskStride_;
    AscendC::LocalTensor<int32_t> ids_, iota_, selected_, counts_;
    AscendC::LocalTensor<float> floats_, activeFloat_;
    AscendC::LocalTensor<half> activeHalf_;
    AscendC::LocalTensor<uint8_t> bits_, active_, sendMasks_;
};
} // namespace MoePermute
#endif
