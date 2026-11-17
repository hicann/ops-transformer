/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_BLOCK_EPILOGUE_PREPARE_KV_SCALE_ARCH35_MXFP4_HPP
#define GBSA_BLOCK_EPILOGUE_PREPARE_KV_SCALE_ARCH35_MXFP4_HPP

#include "gbsa_block_epilogue.hpp"
#include "../../../arch35/gbsa_mxfp4_kernel_utils.hpp"
#include "../gbsa_epilogue_dispatch_policy.hpp"
#include "gbsa_block_epilogue_arch35_mxfp4_utils.hpp"
#include "../../gemm/block/gbsa_block_mmad_arch35_mxfp4_utils.hpp"
#include "../../gemm/tile_common/gbsa_prepare_k_scale_mxfp4_a5.hpp"
#include "mxfp4_vf/vf_pack_k_scale_mxfp4.h"

namespace NpuArch::Epilogue::Block {

template <bool Paged>
class BlockEpilogue<EpilogueAtlasA5Mxfp4KvScale<Paged>> {
public:
    using DispatchPolicy = EpilogueAtlasA5Mxfp4KvScale<Paged>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    uint32_t D, Q, R;
    bool paddedScaleRows = false;
    using UbLayout = GbsaKernelArch35::Mxfp4::UBufTileHelper;
    UbLayout ub;
    using L1Layout = GbsaKernelArch35::Mxfp4::L1TileHelper;
    L1Layout l1;
    using Descriptor = GbsaKernelArch35::Mxfp4::DescriptorTileHelper<Paged>;
    Descriptor descriptor;
    static constexpr uint32_t SCALE_EVENT = 7;
    static constexpr uint32_t SCALE_ORDER_EVENT = 7;

    __aicore__ inline explicit BlockEpilogue(Arch::Resource<ArchTag>& resource, const L1Layout& l1Info,
                                             const UbLayout& ubInfo, const Descriptor& descriptorInfo)
        : D(l1Info.D),
          Q(l1Info.Q),
          R(l1Info.R),
          ub(ubInfo),
          l1(l1Info),
          descriptor(descriptorInfo),
          copyKScale(l1Info.D)
    {
        inputScaleUBTensor = resource.ubBuf.template GetBufferByByte<uint8_t>(ub.DESC_OFFSET);
        kScaleUBTensor = resource.ubBuf.template GetBufferByByte<uint8_t>(ub.KSCALE_OUTPUT_OFFSET);
        for (uint32_t i = 0; i < L1Layout::K_SLOTS; ++i) {
            kScaleL1Tensor[i] =
                resource.l1Buf.template GetBufferByByte<uint8_t>(l1.KSCALE_OFFSET + i * l1.KSCALE_SLOT_BYTES);
        }
        for (uint32_t i = 0; i < descriptor.INPUT_SLOTS; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(static_cast<event_t>(i));
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(SCALE_EVENT);
    }

    __aicore__ inline ~BlockEpilogue()
    {
        for (uint32_t i = 0; i < descriptor.INPUT_SLOTS; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(static_cast<event_t>(i));
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(SCALE_EVENT);
    }

    __aicore__ inline void BeginTile(const AscendC::GlobalTensor<uint8_t>& source, uint32_t sequence, uint32_t blockY,
                                     uint32_t sourceTokens, uint32_t tokenBytes)
    {
        gScale = source;
        inputTile = sequence % 3U;
        tokenStrideBytes = tokenBytes;
        // The existing input slot reserves 32B per token. A span whose scale
        // token stride is at most 32B fits without allocating another buffer.
        paddedScaleRows = UsePaddedScaleRows(blockY, tokenBytes / 16U);
        blockRows = !paddedScaleRows && tokenBytes / 16U <= 32U && blockY >= 4 && (blockY & (blockY - 1)) == 0 ?
                        min(blockY, 16U) :
                        1U;
        sourceBytes = sourceTokens * (D / 32);
    }

    __aicore__ inline void BeginGroup(uint32_t group, uint32_t)
    {
        inputSlot = inputTile * descriptor.GROUPS_PER_BANK + group / 2;
        copiedRows = 0;
        direct = false;
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(static_cast<event_t>(inputSlot));
    }

    __aicore__ inline void Record(uint64_t offset, uint32_t rows, uint64_t stride, uint32_t matrices)
    {
        if (!Paged && tokenStrideBytes == D / 2 && rows == descriptor.ROWS && matrices == 1 && offset % 16 == 0 &&
            (reinterpret_cast<uint64_t>(gScale.GetPhyAddr()) + offset / 16) % 32 == 0) {
            direct = true;
            return;
        }
        auto input = inputScaleUBTensor[inputSlot * descriptor.INPUT_BYTES];
        if (blockRows == 1U) {
            copyKScale.template CopyRecord<1>(input, gScale, offset, rows, stride, matrices, copiedRows, sourceBytes,
                                              tokenStrideBytes);
        } else if (blockRows == 4) {
            copyKScale.template CopyRecord<4>(input, gScale, offset, rows, stride, matrices, copiedRows, sourceBytes,
                                              tokenStrideBytes);
        } else if (blockRows == 8) {
            copyKScale.template CopyRecord<8>(input, gScale, offset, rows, stride, matrices, copiedRows, sourceBytes,
                                              tokenStrideBytes);
        } else if (blockRows == 16) {
            copyKScale.template CopyRecord<16>(input, gScale, offset, rows, stride, matrices, copiedRows, sourceBytes,
                                               tokenStrideBytes);
        } else {
            copyKScale.template CopyRecord<1>(input, gScale, offset, rows, stride, matrices, copiedRows, sourceBytes,
                                              tokenStrideBytes);
        }
        copiedRows += rows * matrices;
    }

    // Complete sparse blocks with contiguous token rows select the copy geometry
    // once per descriptor group, outside the record emission loop.
    template <uint32_t BlockRows>
    __aicore__ inline void Record(uint64_t offset, uint32_t rows, uint64_t stride, uint32_t matrices)
    {
        if (paddedScaleRows) {
            auto input = inputScaleUBTensor[inputSlot * descriptor.INPUT_BYTES];
            copyKScale.template CopyRecord<1>(input, gScale, offset, rows, stride, matrices, copiedRows, sourceBytes,
                                              tokenStrideBytes);
            copiedRows += rows * matrices;
            return;
        }
        if (!Paged && rows == descriptor.ROWS && matrices == 1U && offset % 16U == 0U &&
            (reinterpret_cast<uint64_t>(gScale.GetPhyAddr()) + offset / 16U) % 32U == 0U) {
            direct = true;
            return;
        }
        auto input = inputScaleUBTensor[inputSlot * descriptor.INPUT_BYTES];
        copyKScale.template CopyRecord<BlockRows>(input, gScale, offset, rows, stride, matrices, copiedRows,
                                                  sourceBytes, tokenStrideBytes);
        copiedRows += rows * matrices;
    }

    template <class Records>
    __aicore__ inline void EndGroup(Records records, uint32_t count, bool fullRecords)
    {
        // Tail/irregular blocks retain the shared descriptor builder's exact
        // merge and split decisions. Complete BNSD blocks emit directly.
        if (!fullRecords) {
            for (uint32_t i = 0; i < count; ++i) {
                const uint32_t word = 8 + i * 2;
                const uint64_t meta = records[word + 1];
                Record(records[word], static_cast<uint32_t>(meta) >> 1, meta >> 32, 1 + (meta & 1));
            }
        }
        groupRows[inputSlot] = direct ? 0U : copiedRows;
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(inputSlot));
    }

    __aicore__ inline void PrepareTile(uint32_t validKvs, uint32_t bank, uint32_t blockY, uint32_t sequence)
    {
        using namespace GbsaKernelArch35::Mxfp4;
        const uint32_t packRows =
            !paddedScaleRows && tokenStrideBytes / 16U <= 32U && blockY >= 4 && (blockY & (blockY - 1)) == 0 ?
                min(blockY, 16U) :
                1U;
        for (uint32_t first = 0; first < validKvs; first += 2U * descriptor.ROWS) {
            const uint32_t pair = first / (2U * descriptor.ROWS);
            const uint32_t group = first / descriptor.ROWS + bank;
            const bool active = group * descriptor.ROWS < validKvs;
            const uint32_t slot = sequence % 3U * descriptor.GROUPS_PER_BANK + pair;
            const uint32_t rows = active ? groupRows[slot] : 0U;
            if (rows == 0U) {
                if ((pendingScaleGroups & (1U << pair)) != 0U) {
                    // A Scalar wait must not consume a grant belonging to an
                    // earlier MTE3 wait on the same cross-core flag.
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(SCALE_ORDER_EVENT);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(SCALE_ORDER_EVENT);
                    pendingScaleGroups = 0U;
                }
                // Cube owns direct scale copies. No AIV L1 write needs MTE3,
                // but every group still consumes and returns one grant.
                AscendC::CrossCoreWaitFlag<4, PIPE_S>(static_cast<uint32_t>(Flag::KSCALE_FREE_BASE) + pair);
                AscendC::CrossCoreSetFlag<4, PIPE_S>(static_cast<uint32_t>(Flag::KSCALE_BASE) + pair);
                if (active) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(slot));
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(static_cast<event_t>(slot));
                }
            } else {
                AscendC::CrossCoreWaitFlag<4, PIPE_MTE3>(static_cast<uint32_t>(Flag::KSCALE_FREE_BASE) + pair);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(slot));
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(SCALE_EVENT);
                GbsaMxfp4VF::PackKScaleCallVF(kScaleUBTensor, inputScaleUBTensor[slot * descriptor.INPUT_BYTES], rows,
                                              packRows, D, tokenStrideBytes / 32U);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(SCALE_EVENT);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(SCALE_EVENT);
                const uint32_t bytes = (rows + 15) / 16 * 16 * (D / 32);
                AscendC::DataCopy(kScaleL1Tensor[sequence % L1Layout::K_SLOTS][group * descriptor.ROWS * (D / 32)],
                                  kScaleUBTensor, bytes);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(SCALE_EVENT);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(static_cast<event_t>(slot));
                AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(static_cast<uint32_t>(Flag::KSCALE_BASE) + pair);
                pendingScaleGroups |= 1U << pair;
            }
        }
    }

private:
    __aicore__ static constexpr bool UsePaddedScaleRows(uint32_t blockRows, uint32_t tokenScaleStride)
    {
        // Keep only the padded-row fallback required by unsupported span geometry.
        return tokenScaleStride > 32U || blockRows < 4U || (blockRows & (blockRows - 1U)) != 0U;
    }

    AscendC::LocalTensor<uint8_t> inputScaleUBTensor, kScaleUBTensor;
    AscendC::GlobalTensor<uint8_t> gScale;
    Gemm::Tile::CopyKScaleGmToUbMxfp4<Paged> copyKScale;
    AscendC::LocalTensor<uint8_t> kScaleL1Tensor[L1Layout::K_SLOTS];
    uint32_t groupRows[6];
    uint32_t inputTile = 0, inputSlot = 0, copiedRows = 0, blockRows = 1, sourceBytes = 0;
    uint32_t tokenStrideBytes = 0;
    uint32_t pendingScaleGroups = 0;
    bool direct = false;
};

} // namespace NpuArch::Epilogue::Block
#endif // GBSA_BLOCK_EPILOGUE_PREPARE_KV_SCALE_ARCH35_MXFP4_HPP
