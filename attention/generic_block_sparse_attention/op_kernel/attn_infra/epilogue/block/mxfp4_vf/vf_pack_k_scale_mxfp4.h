/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_VF_PACK_K_SCALE_MXFP4_H
#define GBSA_VF_PACK_K_SCALE_MXFP4_H

#include "vf_mxfp4_common.h"

namespace NpuArch::Epilogue::Block::GbsaMxfp4VF {

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

} // namespace NpuArch::Epilogue::Block::GbsaMxfp4VF
#endif // GBSA_VF_PACK_K_SCALE_MXFP4_H
