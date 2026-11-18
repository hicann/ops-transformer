/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_BLOCK_EPILOGUE_PREPARE_K_SCALE_ARCH35_MXFP8_HPP
#define GBSA_BLOCK_EPILOGUE_PREPARE_K_SCALE_ARCH35_MXFP8_HPP

#include "gbsa_block_epilogue.hpp"
#include "../gbsa_epilogue_dispatch_policy.hpp"

#include "../../gemm/tile_common/gbsa_prepare_k_scale_mxfp8_a5.hpp"

namespace NpuArch::Epilogue::Block::GbsaMxfp8BaseVF {
using namespace AscendC;
using namespace AscendC::Reg;
// GM->UB pads each token's scale row to 32B. Retain E8M0 bytes as uint16
// pairs and transpose sixteen rows into [pair,16 rows] without scalar loads.
template <uint16_t D, bool HasTail>
__simd_vf__ inline void StaticPackKScale(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src, uint16_t rows)
{
    constexpr uint16_t pairs = D / 64;
    MaskReg all = CreateMask<uint16_t, MaskPattern::ALL>();
    RegTensor<uint16_t> position, high, low, gatherIndex, order, rowIndex, unit;
    RegTensor<uint16_t> first, second, packed0, packed1, combined, unused, output, actualRow;
    MaskReg valid;
    Arange((RegTensor<int16_t>&)position, 0);
    ShiftRights(high, position, static_cast<int16_t>(3), all);
    Muls(low, high, static_cast<uint16_t>(8), all);
    Sub(low, position, low, all);
    Muls(gatherIndex, low, static_cast<uint16_t>(16), all);
    Add(gatherIndex, gatherIndex, high, all);
    // First gather: [8 rows,16 pairs] -> [16 pairs,8 rows].
    ShiftRights(high, position, static_cast<int16_t>(4), all);
    Muls(high, high, static_cast<uint16_t>(16), all);
    Sub(rowIndex, position, high, all);
    ShiftRights(order, rowIndex, static_cast<int16_t>(3), all);
    Muls(low, low, static_cast<uint16_t>(2), all);
    Add(order, order, low, all);
    Add(order, order, high, all);
    uint32_t elements = 16 * pairs;
    MaskReg storeMask = UpdateMask<uint16_t>(elements);
    if constexpr (HasTail) {
        Duplicate(unit, static_cast<uint16_t>(0x7f7f));
    }
    for (uint16_t row = 0; row < rows; row += 16) {
        LoadAlign(first, src + row * 16);
        LoadAlign(second, src + (row + 8) * 16);
        Gather(packed0, first, gatherIndex);
        Gather(packed1, second, gatherIndex);
        Interleave(combined, unused, packed0, packed1);
        Gather(output, combined, order);
        if constexpr (HasTail) {
            Adds(actualRow, rowIndex, row, all);
            Compares<uint16_t, CMPMODE::LT>(valid, actualRow, rows, all);
            Select(output, output, unit, valid);
        }
        StoreAlign(dst + row * pairs, output, storeMask);
    }
}

// A block contains 4/8/16 consecutive tokens; only block boundaries are
// padded to 32B. One register covers all source bytes needed for 16 rows.
template <uint16_t D, uint16_t BlockRows, bool HasTail>
__simd_vf__ inline void StaticPackKScaleBlocks(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src, uint16_t rows)
{
    constexpr uint16_t pairs = D / 64;
    constexpr uint16_t blockPairs = (BlockRows * pairs + 15) / 16 * 16;
    constexpr uint16_t blockShift = BlockRows == 4 ? 2 : (BlockRows == 8 ? 3 : 4);
    MaskReg all = CreateMask<uint16_t, MaskPattern::ALL>();
    RegTensor<uint16_t> position, rowIndex, pairIndex, block, inner, index, unit;
    RegTensor<uint16_t> input, output, actualRow;
    MaskReg valid;
    Arange((RegTensor<int16_t>&)position, 0);
    ShiftRights(pairIndex, position, static_cast<int16_t>(4), all);
    Muls(rowIndex, pairIndex, static_cast<uint16_t>(16), all);
    Sub(rowIndex, position, rowIndex, all);
    ShiftRights(block, rowIndex, static_cast<int16_t>(blockShift), all);
    Muls(inner, block, BlockRows, all);
    Sub(inner, rowIndex, inner, all);
    Muls(inner, inner, pairs, all);
    Muls(index, block, blockPairs, all);
    Add(index, index, inner, all);
    Add(index, index, pairIndex, all);
    uint32_t elements = 16 * pairs;
    MaskReg storeMask = UpdateMask<uint16_t>(elements);
    if constexpr (HasTail) {
        Duplicate(unit, static_cast<uint16_t>(0x7f7f));
    }
    for (uint16_t row = 0; row < rows; row += 16) {
        LoadAlign(input, src + row / BlockRows * blockPairs);
        Gather(output, input, index);
        if constexpr (HasTail) {
            Adds(actualRow, rowIndex, row, all);
            Compares<uint16_t, CMPMODE::LT>(valid, actualRow, rows, all);
            Select(output, output, unit, valid);
        }
        StoreAlign(dst + row * pairs, output, storeMask);
    }
}

template <uint16_t D, uint16_t BlockRows>
__aicore__ inline void StaticPackKScaleBlocksCallVF(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src, uint16_t rows)
{
    if (rows % 16 == 0) {
        StaticPackKScaleBlocks<D, BlockRows, false>(dst, src, rows);
    } else {
        StaticPackKScaleBlocks<D, BlockRows, true>(dst, src, rows);
    }
}

// A span retains the other heads between adjacent token rows. At most 32B
// per row is reserved, so two registers cover sixteen rows of every supported
// span. Gather selects this head directly into the same zZ output as above.
template <uint16_t D, uint16_t BlockRows, bool HasTail>
__simd_vf__ inline void StaticPackKScaleSpans(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src, uint16_t rows,
                                              uint16_t stridePairs, uint16_t blockPairs)
{
    constexpr uint16_t pairs = D / 64;
    constexpr uint16_t blockShift = BlockRows == 4 ? 2 : (BlockRows == 8 ? 3 : 4);
    MaskReg all = CreateMask<uint16_t, MaskPattern::ALL>();
    RegTensor<uint16_t> position, rowIndex, pairIndex, block, inner, index, localIndex, unit;
    RegTensor<uint16_t> input0, input1, output0, output1, output, actualRow;
    MaskReg firstRegister, valid;
    Arange((RegTensor<int16_t>&)position, 0);
    ShiftRights(pairIndex, position, static_cast<int16_t>(4), all);
    Muls(rowIndex, pairIndex, static_cast<uint16_t>(16), all);
    Sub(rowIndex, position, rowIndex, all);
    ShiftRights(block, rowIndex, static_cast<int16_t>(blockShift), all);
    Muls(inner, block, BlockRows, all);
    Sub(inner, rowIndex, inner, all);
    Muls(inner, inner, stridePairs, all);
    Muls(index, block, blockPairs, all);
    Add(index, index, inner, all);
    Add(index, index, pairIndex, all);
    Compares<uint16_t, CMPMODE::LT>(firstRegister, index, static_cast<uint16_t>(128), all);
    Duplicate(localIndex, static_cast<uint16_t>(127));
    And(localIndex, index, localIndex, all);
    uint32_t elements = 16 * pairs;
    MaskReg storeMask = UpdateMask<uint16_t>(elements);
    if constexpr (HasTail) {
        Duplicate(unit, static_cast<uint16_t>(0x7f7f));
    }
    for (uint16_t row = 0; row < rows; row += 16) {
        LoadAlign(input0, src + row / BlockRows * blockPairs);
        LoadAlign(input1, src + row / BlockRows * blockPairs + 128);
        Gather(output0, input0, localIndex);
        Gather(output1, input1, localIndex);
        Select(output, output0, output1, firstRegister);
        if constexpr (HasTail) {
            Adds(actualRow, rowIndex, row, all);
            Compares<uint16_t, CMPMODE::LT>(valid, actualRow, rows, all);
            Select(output, output, unit, valid);
        }
        StoreAlign(dst + row * pairs, output, storeMask);
    }
}

template <uint16_t D, uint16_t BlockRows>
__aicore__ inline void StaticPackKScaleSpansCallVF(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src, uint16_t rows,
                                                   uint16_t stridePairs)
{
    const uint16_t blockPairs = ((BlockRows - 1U) * stridePairs + D / 64U + 15U) / 16U * 16U;
    if (rows % 16 == 0) {
        StaticPackKScaleSpans<D, BlockRows, false>(dst, src, rows, stridePairs, blockPairs);
    } else {
        StaticPackKScaleSpans<D, BlockRows, true>(dst, src, rows, stridePairs, blockPairs);
    }
}

template <uint16_t D>
__aicore__ inline void StaticPackKScaleCallVF(const LocalTensor<uint8_t>& dst, const LocalTensor<uint8_t>& src,
                                              uint16_t rows, uint16_t blockRows, uint16_t stridePairs)
{
    auto dstAddress = reinterpret_cast<__ubuf__ uint16_t*>(dst.GetPhyAddr());
    auto srcAddress = reinterpret_cast<__ubuf__ uint16_t*>(src.GetPhyAddr());
    if (stridePairs != D / 64U && blockRows != 1) {
        if (blockRows == 4) {
            StaticPackKScaleSpansCallVF<D, 4>(dstAddress, srcAddress, rows, stridePairs);
        } else if (blockRows == 8) {
            StaticPackKScaleSpansCallVF<D, 8>(dstAddress, srcAddress, rows, stridePairs);
        } else {
            StaticPackKScaleSpansCallVF<D, 16>(dstAddress, srcAddress, rows, stridePairs);
        }
        return;
    }
    if (blockRows == 4) {
        StaticPackKScaleBlocksCallVF<D, 4>(dstAddress, srcAddress, rows);
    } else if (blockRows == 8) {
        StaticPackKScaleBlocksCallVF<D, 8>(dstAddress, srcAddress, rows);
    } else if (blockRows == 16) {
        StaticPackKScaleBlocksCallVF<D, 16>(dstAddress, srcAddress, rows);
    } else if (rows % 16 == 0) {
        StaticPackKScale<D, false>(dstAddress, srcAddress, rows);
    } else {
        StaticPackKScale<D, true>(dstAddress, srcAddress, rows);
    }
}

__aicore__ inline void PackKScaleCallVF(const LocalTensor<uint8_t>& dst, const LocalTensor<uint8_t>& src, uint16_t rows,
                                        uint16_t blockRows, uint16_t D, uint16_t stridePairs)
{
    if (D == 64) {
        StaticPackKScaleCallVF<64>(dst, src, rows, blockRows, stridePairs);
    } else if (D == 128) {
        StaticPackKScaleCallVF<128>(dst, src, rows, blockRows, stridePairs);
    } else if (D == 192) {
        StaticPackKScaleCallVF<192>(dst, src, rows, blockRows, stridePairs);
    } else if (D == 256) {
        StaticPackKScaleCallVF<256>(dst, src, rows, blockRows, stridePairs);
    }
}

} // namespace NpuArch::Epilogue::Block::GbsaMxfp8BaseVF

namespace NpuArch::Epilogue::Block {

template <bool Paged, uint32_t TileTokens = 512U>
class PrepareKScaleMxfp8Full {
public:
    using ArchTag = Arch::AtlasA5;
    uint32_t D;
    struct Buffers {
        uint32_t rawOffset, nzOffset, rawStages, l1Offset, l1SlotBytes, l1Slots;
        uint32_t scaleReadyFlag, scaleFreeFlag;
    };
    Buffers buffers_;
    // Each AIV owns alternating groups. Three raw tiles are independent of
    // the L1 scale ring and the one NZ conversion buffer.
    struct RawLayout {
        uint32_t ROWS, GROUPS_PER_BANK, INPUT_BYTES, INPUT_SLOTS;
        __aicore__ explicit RawLayout(uint32_t rows, uint32_t rawStages)
            : ROWS(rows),
              GROUPS_PER_BANK(TileTokens / rows / 2U),
              INPUT_BYTES(rows * 32U),
              INPUT_SLOTS(rawStages * GROUPS_PER_BANK)
        {}
    };
    RawLayout descriptor;
    static constexpr uint32_t MAX_L1_SLOTS = 3U;
    static constexpr uint32_t SCALE_EVENT = 7;
    static constexpr uint32_t SCALE_ORDER_EVENT = 7;

    __aicore__ inline explicit PrepareKScaleMxfp8Full(Arch::Resource<ArchTag>& resource, uint32_t dim,
                                                      uint32_t groupRows, const Buffers& buffers)
        : D(dim),
          buffers_(buffers),
          descriptor(groupRows, buffers.rawStages),
          copyKScale(dim)
    {
        inputScaleUBTensor = resource.ubBuf.template GetBufferByByte<uint8_t>(buffers_.rawOffset);
        kScaleUBTensor = resource.ubBuf.template GetBufferByByte<uint8_t>(buffers_.nzOffset);
        for (uint32_t i = 0; i < buffers_.l1Slots; ++i) {
            kScaleL1Tensor[i] =
                resource.l1Buf.template GetBufferByByte<uint8_t>(buffers_.l1Offset + i * buffers_.l1SlotBytes);
        }
        for (uint32_t i = 0; i < descriptor.INPUT_SLOTS; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(static_cast<event_t>(i));
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(SCALE_EVENT);
    }

    __aicore__ inline ~PrepareKScaleMxfp8Full()
    {
        for (uint32_t i = 0; i < descriptor.INPUT_SLOTS; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(static_cast<event_t>(i));
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(SCALE_EVENT);
    }

    __aicore__ inline void BeginTile(const AscendC::GlobalTensor<uint8_t>& source, uint32_t sequence, uint32_t blockY,
                                     uint32_t sourceTokens, uint64_t tokenBytes)
    {
        gScale = source;
        inputTile = sequence % buffers_.rawStages;
        tokenStrideBytes = tokenBytes;
        // The existing input slot reserves 32B per token. A span whose scale
        // token stride is at most 32B fits without allocating another buffer.
        blockRows = tokenBytes / 32U <= 32U && blockY >= 4 && (blockY & (blockY - 1)) == 0 ? min(blockY, 16U) : 1U;
        sourceBytes = static_cast<uint64_t>(sourceTokens) * (D / 32U);
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
        if (!Paged && tokenStrideBytes == D && rows == descriptor.ROWS && matrices == 1 && offset % 32 == 0 &&
            (reinterpret_cast<uint64_t>(gScale.GetPhyAddr()) + offset / 32) % 32 == 0) {
            direct = true;
            return;
        }
        auto input = inputScaleUBTensor[RawOffset(inputSlot)];
        if (blockRows == 4) {
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
        if (!Paged && rows == descriptor.ROWS && matrices == 1U && offset % 32U == 0U &&
            (reinterpret_cast<uint64_t>(gScale.GetPhyAddr()) + offset / 32U) % 32U == 0U) {
            direct = true;
            return;
        }
        auto input = inputScaleUBTensor[RawOffset(inputSlot)];
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
                const uint32_t word = GbsaKernelArch35::Mxfp8Descriptor::HEADER_WORDS +
                                      i * GbsaKernelArch35::Mxfp8Descriptor::RECORD_WORDS;
                const uint64_t meta = records[word + 1];
                Record(records[word], static_cast<uint32_t>(meta) >> 1, meta >> 32, 1 + (meta & 1));
            }
        }
        groupRows[inputSlot] = direct ? 0U : copiedRows;
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(inputSlot));
    }

    __aicore__ inline void PrepareTile(uint32_t validKvs, uint32_t bank, uint32_t blockY, uint32_t sequence)
    {
        const uint32_t packRows =
            tokenStrideBytes / 32U <= 32U && blockY >= 4 && (blockY & (blockY - 1)) == 0 ? min(blockY, 16U) : 1U;
        // Row-by-row copies have already removed the GM gaps. Only the bounded
        // span path passes the physical stride (at most 16 pairs) into the VF.
        const uint16_t stridePairs = static_cast<uint16_t>(packRows == 1U ? D / 64U : tokenStrideBytes / 64U);
        for (uint32_t first = 0; first < validKvs; first += 2U * descriptor.ROWS) {
            const uint32_t pair = first / (2U * descriptor.ROWS);
            const uint32_t group = first / descriptor.ROWS + bank;
            const bool active = group * descriptor.ROWS < validKvs;
            const uint32_t slot = sequence % buffers_.rawStages * descriptor.GROUPS_PER_BANK + pair;
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
                AscendC::CrossCoreWaitFlag<4, PIPE_S>(buffers_.scaleFreeFlag + pair);
                AscendC::CrossCoreSetFlag<4, PIPE_S>(buffers_.scaleReadyFlag + pair);
                if (active) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(slot));
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(static_cast<event_t>(slot));
                }
            } else {
                AscendC::CrossCoreWaitFlag<4, PIPE_MTE3>(buffers_.scaleFreeFlag + pair);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(slot));
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(SCALE_EVENT);
                GbsaMxfp8BaseVF::PackKScaleCallVF(kScaleUBTensor, inputScaleUBTensor[RawOffset(slot)], rows, packRows,
                                                  D, stridePairs);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(SCALE_EVENT);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(SCALE_EVENT);
                const uint32_t bytes = (rows + 15) / 16 * 16 * (D / 32);
                AscendC::DataCopy(kScaleL1Tensor[sequence % buffers_.l1Slots][group * descriptor.ROWS * (D / 32)],
                                  kScaleUBTensor, bytes);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(SCALE_EVENT);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(static_cast<event_t>(slot));
                AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(buffers_.scaleReadyFlag + pair);
                pendingScaleGroups |= 1U << pair;
            }
        }
    }

private:
    __aicore__ inline uint32_t RawOffset(uint32_t slot) const
    {
        if constexpr (TileTokens == 256U) {
            return slot * 256U * 32U;
        }
        return slot * descriptor.INPUT_BYTES;
    }
    AscendC::LocalTensor<uint8_t> inputScaleUBTensor, kScaleUBTensor;
    AscendC::GlobalTensor<uint8_t> gScale;
    Gemm::Tile::CopyKScaleGmToUbMxfp8Full<Paged> copyKScale;
    AscendC::LocalTensor<uint8_t> kScaleL1Tensor[MAX_L1_SLOTS];
    uint32_t groupRows[6];
    uint32_t inputTile = 0, inputSlot = 0, copiedRows = 0, blockRows = 1;
    uint64_t sourceBytes = 0;
    uint64_t tokenStrideBytes = 0;
    uint32_t pendingScaleGroups = 0;
    bool direct = false;
};

} // namespace NpuArch::Epilogue::Block
#endif // GBSA_BLOCK_EPILOGUE_PREPARE_K_SCALE_ARCH35_MXFP8_HPP
