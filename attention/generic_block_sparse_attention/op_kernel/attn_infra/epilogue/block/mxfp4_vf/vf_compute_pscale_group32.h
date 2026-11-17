/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_VF_COMPUTE_PSCALE_GROUP32_H
#define GBSA_VF_COMPUTE_PSCALE_GROUP32_H

#include "vf_mxfp4_common.h"

namespace NpuArch::Epilogue::Block::GbsaMxfp4VF {

// A short tile has one live group per row. Its compact statistics still use
// four columns, but neither the denominator nor zZ needs a group reduction.
__simd_vf__ inline void ComputeSingleGroupScale32(__ubuf__ float* localMax, __ubuf__ uint8_t* scales,
                                                  __ubuf__ float* tileG, __ubuf__ float* previousG,
                                                  __ubuf__ float* tileDm, __ubuf__ float* groupSum,
                                                  __ubuf__ float* tileSum, uint32_t validRows, float scaleOffset,
                                                  float scaleBias)
{
    RegTensor<float> local, sums, maximum, previous, zero, unit, exponent, weight, dm, safeG, safePrevious;
    RegTensor<uint32_t> index, zeroIndex;
    RegTensor<int32_t> integer;
    RegTensor<uint16_t> pairs, padding;
    MaskReg all = CreateMask<float, MaskPattern::ALL>();
    MaskReg scaleMask = CreateMask<uint16_t, MaskPattern::VL16>();
    MaskReg valid, validG, validPrevious, validBoth;
    uint32_t rows = validRows;
    MaskReg rowMask = UpdateMask<float>(rows);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    Duplicate(zero, 0.0f);
    Duplicate(unit, 127.0f);
    Duplicate(padding, static_cast<uint16_t>(0x7f7f));
    Duplicate(zeroIndex, static_cast<uint32_t>(0));
    Arange((RegTensor<int32_t>&)index, 0);
    Muls(index, index, static_cast<uint32_t>(4), all);
    Select(index, index, zeroIndex, rowMask);
    LoadAlign(local, localMax);
    LoadAlign(sums, groupSum);
    Gather(exponent, local, index);
    Gather(weight, sums, index);
    LoadAlign(maximum, tileG);
    LoadAlign(previous, previousG);
    Compares<float, CMPMODE::GT>(validG, maximum, NEG_INF, rowMask);
    Compares<float, CMPMODE::GT>(validPrevious, previous, NEG_INF, rowMask);
    MaskAnd(validBoth, validG, validPrevious, rowMask);
    Select(safeG, maximum, zero, validBoth);
    Select(safePrevious, previous, zero, validBoth);
    Sub(dm, safePrevious, safeG, rowMask);
    Muls(dm, dm, LN2, rowMask);
    Exp(dm, dm, rowMask);
    Select(dm, dm, zero, validBoth);
    StoreAlign(tileDm, dm, rowMask);

    Compares<float, CMPMODE::GT>(valid, exponent, NEG_INF, rowMask);
    Select(local, exponent, zero, valid);
    Select(safeG, maximum, zero, valid);
    Select(sums, weight, zero, valid);
    Sub(exponent, local, safeG, all);
    Adds(weight, exponent, scaleOffset, all);
    Muls(weight, weight, LN2, all);
    Exp(weight, weight, all);
    Mul(sums, sums, weight, all);
    StoreAlign(tileSum, sums, rowMask);
    Adds(exponent, exponent, scaleBias, all);
    Maxs(exponent, exponent, 0.0f, all);
    Select(exponent, exponent, unit, valid);
    // Pair zero contains the live exponent and one neutral exponent; pair
    // one is entirely neutral. Both 16-row blocks are fully initialized.
    Adds(exponent, exponent, static_cast<float>(0x7f00U), all);
    Cast<int32_t, float, INTEGER_ROUND>(integer, exponent, all);
    Pack(pairs, (RegTensor<uint32_t>&)integer);
    StoreAlign(reinterpret_cast<__ubuf__ uint16_t*>(scales), pairs, scaleMask);
    StoreAlign(reinterpret_cast<__ubuf__ uint16_t*>(scales) + 16, padding, scaleMask);
}

__aicore__ inline void SingleGroupPScaleCallVF(const LocalTensor<float>& localMax, const LocalTensor<uint8_t>& scale,
                                               const LocalTensor<float>& maximum, const LocalTensor<float>& previous,
                                               const LocalTensor<float>& dm, const LocalTensor<float>& groupSum,
                                               const LocalTensor<float>& tileSum, uint32_t rows, float scaleOffset,
                                               float scaleBias)
{
    ComputeSingleGroupScale32(reinterpret_cast<__ubuf__ float*>(localMax.GetPhyAddr()),
                              reinterpret_cast<__ubuf__ uint8_t*>(scale.GetPhyAddr()),
                              reinterpret_cast<__ubuf__ float*>(maximum.GetPhyAddr()),
                              reinterpret_cast<__ubuf__ float*>(previous.GetPhyAddr()),
                              reinterpret_cast<__ubuf__ float*>(dm.GetPhyAddr()),
                              reinterpret_cast<__ubuf__ float*>(groupSum.GetPhyAddr()),
                              reinterpret_cast<__ubuf__ float*>(tileSum.GetPhyAddr()), rows, scaleOffset, scaleBias);
}

// PScale = 2^(c-G-ceil(log2(CX))); groupSum already includes the ln(1.5) shift.
// G includes the current tile. Retain the preceding snapshot for dm and
// the delayed per-tile RescaleO consumer.
template <uint16_t N, uint16_t MaxRows>
__simd_vf__ inline void StaticComputeTileScale32(__ubuf__ float* localMax, __ubuf__ uint8_t* scales,
                                                 __ubuf__ float* tileG, __ubuf__ float* previousG,
                                                 __ubuf__ float* tileDm, __ubuf__ float* groupSum,
                                                 __ubuf__ float* tileSum, uint32_t validRows, float scaleOffset,
                                                 float scaleBias)
{
    constexpr uint16_t rowsPerLoad = 2048 / N;
    RegTensor<float> g, local, exponent, weight, sum, reduced;
    RegTensor<uint32_t> index;
    RegTensor<int32_t> integer;
    RegTensor<uint16_t> shortCode, high, pairs, unused, reordered, lower, upper;
    RegTensor<uint8_t> packed0, packed1, packed2, packed3;
    RegTensor<float> previous, maximum, zero, unit, code, safePrevious, safeMaximum;
    RegTensor<uint32_t> lanes, rowIndex;
    RegTensor<uint16_t> position, pair, row, half, low, order;
    UnalignReg sumStore0, sumStore1, sumStore2, sumStore3;
    __ubuf__ float* sumAddress;
    MaskReg valid, rowValid;
    MaskReg eight = CreateMask<float, MaskPattern::VL8>();
    MaskReg halfMask = CreateMask<float, MaskPattern::VL32>();
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    uint32_t rows = validRows;
    MaskReg mask = UpdateMask<float>(rows);
    MaskReg all = CreateMask<float, MaskPattern::ALL>();
    Duplicate(zero, 0.0f);
    Duplicate(unit, 127.0f);
    LoadAlign(previous, previousG);
    LoadAlign(maximum, tileG);
    MaskReg validPrevious, validMaximum, validBoth;
    Compares<float, CMPMODE::GT>(validPrevious, previous, NEG_INF, mask);
    Compares<float, CMPMODE::GT>(validMaximum, maximum, NEG_INF, mask);
    MaskAnd(validBoth, validPrevious, validMaximum, mask);
    Select(safePrevious, previous, zero, validBoth);
    Select(safeMaximum, maximum, zero, validBoth);
    Sub(code, safePrevious, safeMaximum, mask);
    Muls(code, code, LN2, mask);
    Exp(code, code, mask);
    Select(code, code, zero, validBoth);
    StoreAlign(tileDm, code, mask);

    // Consecutive FP32 c values hold eight (N=256) or four (N=128)
    // exponents per row. Broadcast G within each row using register Gather.
    Arange((RegTensor<int32_t>&)lanes, 0);
    ShiftRights(rowIndex, lanes, static_cast<int16_t>(N == 512 ? 4 : (N == 256 ? 3 : 2)), all);
    MaskReg pairMask = CreateMask<uint16_t, MaskPattern::ALL>();
    Arange((RegTensor<int16_t>&)position, 0);
    ShiftRights(pair, position, static_cast<int16_t>(4), pairMask);
    Muls(row, pair, static_cast<uint16_t>(16), pairMask);
    Sub(row, position, row, pairMask);
    if constexpr (N == 512) {
        // Four loads each contain four rows of sixteen exponents. Two levels
        // of interleave order their row banks as 0,2,1,3; Gather emits zZ.
        ShiftRights(half, row, static_cast<int16_t>(2), pairMask);
        Muls(low, half, static_cast<uint16_t>(4), pairMask);
        Sub(low, row, low, pairMask);
        Muls(order, low, static_cast<uint16_t>(32), pairMask);
        Muls(low, pair, static_cast<uint16_t>(4), pairMask);
        Add(order, order, low, pairMask);
        ShiftRights(low, half, static_cast<int16_t>(1), pairMask);
        Muls(high, low, static_cast<uint16_t>(2), pairMask);
        Sub(half, half, high, pairMask);
        Muls(half, half, static_cast<uint16_t>(2), pairMask);
        Add(order, order, half, pairMask);
        Add(order, order, low, pairMask);
    } else if constexpr (N == 256) {
        ShiftRights(half, row, static_cast<int16_t>(3), pairMask);
        Muls(low, half, static_cast<uint16_t>(8), pairMask);
        Sub(low, row, low, pairMask);
        Muls(low, low, static_cast<uint16_t>(8), pairMask);
        Muls(order, pair, static_cast<uint16_t>(2), pairMask);
        Add(order, order, low, pairMask);
        Add(order, order, half, pairMask);
    } else {
        Muls(order, row, static_cast<uint16_t>(2), pairMask);
        Add(order, order, pair, pairMask);
    }
    uint32_t pairCount = 16 * (N / 64);
    MaskReg storeMask = UpdateMask<uint16_t>(pairCount);
    Select(safeMaximum, maximum, zero, validMaximum);
    if constexpr (N * MaxRows <= 2048) {
        Duplicate(packed1, static_cast<uint8_t>(127));
    }
    if constexpr (N == 512 && MaxRows < 16) {
        Duplicate(packed2, static_cast<uint8_t>(127));
        Duplicate(packed3, static_cast<uint8_t>(127));
    }
    const uint16_t blocks = (validRows + 15) / 16;
    for (uint16_t block = 0; block < blocks; ++block) {
        const uint16_t blockRow = block * 16;
        Adds(index, rowIndex, static_cast<uint32_t>(blockRow), all);
        Gather(g, safeMaximum, index);
        LoadAlign(local, localMax + blockRow * (N / 32));
        LoadAlign(sum, groupSum + blockRow * (N / 32));
        Compares<uint32_t, CMPMODE::LT>(rowValid, index, validRows, all);
        Compares<float, CMPMODE::GT>(valid, local, NEG_INF, all);
        MaskAnd(valid, valid, rowValid, all);
        Select(local, local, zero, valid);
        Select(g, g, zero, valid);
        Select(sum, sum, zero, valid);
        Sub(exponent, local, g, all);
        // Compute the FP32 weight before the E8M0 lower clamp. Small groups can
        // contribute to rowsum even when their quantized PV scale is zero.
        Adds(weight, exponent, scaleOffset, all);
        Muls(weight, weight, LN2, all);
        Exp(weight, weight, all);
        Mul(sum, sum, weight, all);
        if constexpr (N == 512) {
            ReduceDataBlock<ReduceType::SUM>(reduced, sum, all);
            PairReduceElem<PairReduce::SUM>(sum, reduced, eight);
        } else if constexpr (N == 256) {
            ReduceDataBlock<ReduceType::SUM>(sum, sum, all);
        } else {
            PairReduceElem<PairReduce::SUM>(reduced, sum, all);
            PairReduceElem<PairReduce::SUM>(sum, reduced, halfMask);
        }
        sumAddress = tileSum + blockRow;
        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(sumAddress, sum, sumStore0, rowsPerLoad);
        StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(sumAddress, sumStore0, 0);
        Adds(exponent, exponent, scaleBias, all);
        Maxs(exponent, exponent, 0.0f, all);
        Select(exponent, exponent, unit, valid);
        Cast<int32_t, float, INTEGER_ROUND>(integer, exponent, all);
        Pack(shortCode, (RegTensor<uint32_t>&)integer);
        Pack(packed0, shortCode);
        if constexpr (N * MaxRows > 2048) {
            Adds(index, rowIndex, static_cast<uint32_t>(blockRow + rowsPerLoad), all);
            Gather(g, safeMaximum, index);
            LoadAlign(local, localMax + blockRow * (N / 32) + 64);
            LoadAlign(sum, groupSum + blockRow * (N / 32) + 64);
            Compares<uint32_t, CMPMODE::LT>(rowValid, index, validRows, all);
            Compares<float, CMPMODE::GT>(valid, local, NEG_INF, all);
            MaskAnd(valid, valid, rowValid, all);
            Select(local, local, zero, valid);
            Select(g, g, zero, valid);
            Select(sum, sum, zero, valid);
            Sub(exponent, local, g, all);
            // Compute the FP32 weight before the E8M0 lower clamp. Small groups can
            // contribute to rowsum even when their quantized PV scale is zero.
            Adds(weight, exponent, scaleOffset, all);
            Muls(weight, weight, LN2, all);
            Exp(weight, weight, all);
            Mul(sum, sum, weight, all);
            if constexpr (N == 512) {
                ReduceDataBlock<ReduceType::SUM>(reduced, sum, all);
                PairReduceElem<PairReduce::SUM>(sum, reduced, eight);
            } else if constexpr (N == 256) {
                ReduceDataBlock<ReduceType::SUM>(sum, sum, all);
            } else {
                PairReduceElem<PairReduce::SUM>(reduced, sum, all);
                PairReduceElem<PairReduce::SUM>(sum, reduced, halfMask);
            }
            sumAddress = tileSum + blockRow + rowsPerLoad;
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(sumAddress, sum, sumStore1, rowsPerLoad);
            StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(sumAddress, sumStore1, 0);
            Adds(exponent, exponent, scaleBias, all);
            Maxs(exponent, exponent, 0.0f, all);
            Select(exponent, exponent, unit, valid);
            Cast<int32_t, float, INTEGER_ROUND>(integer, exponent, all);
            Pack(shortCode, (RegTensor<uint32_t>&)integer);
            Pack(packed1, shortCode);
        }
        if constexpr (N == 512 && MaxRows == 16) {
            Adds(index, rowIndex, static_cast<uint32_t>(blockRow + 2 * rowsPerLoad), all);
            Gather(g, safeMaximum, index);
            LoadAlign(local, localMax + blockRow * (N / 32) + 2 * 64);
            LoadAlign(sum, groupSum + blockRow * (N / 32) + 2 * 64);
            Compares<uint32_t, CMPMODE::LT>(rowValid, index, validRows, all);
            Compares<float, CMPMODE::GT>(valid, local, NEG_INF, all);
            MaskAnd(valid, valid, rowValid, all);
            Select(local, local, zero, valid);
            Select(g, g, zero, valid);
            Select(sum, sum, zero, valid);
            Sub(exponent, local, g, all);
            // Compute the FP32 weight before the E8M0 lower clamp. Small groups can
            // contribute to rowsum even when their quantized PV scale is zero.
            Adds(weight, exponent, scaleOffset, all);
            Muls(weight, weight, LN2, all);
            Exp(weight, weight, all);
            Mul(sum, sum, weight, all);
            if constexpr (N == 512) {
                ReduceDataBlock<ReduceType::SUM>(reduced, sum, all);
                PairReduceElem<PairReduce::SUM>(sum, reduced, eight);
            } else if constexpr (N == 256) {
                ReduceDataBlock<ReduceType::SUM>(sum, sum, all);
            } else {
                PairReduceElem<PairReduce::SUM>(reduced, sum, all);
                PairReduceElem<PairReduce::SUM>(sum, reduced, halfMask);
            }
            sumAddress = tileSum + blockRow + 2 * rowsPerLoad;
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(sumAddress, sum, sumStore2, rowsPerLoad);
            StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(sumAddress, sumStore2, 0);
            Adds(exponent, exponent, scaleBias, all);
            Maxs(exponent, exponent, 0.0f, all);
            Select(exponent, exponent, unit, valid);
            Cast<int32_t, float, INTEGER_ROUND>(integer, exponent, all);
            Pack(shortCode, (RegTensor<uint32_t>&)integer);
            Pack(packed2, shortCode);
        }
        if constexpr (N == 512 && MaxRows == 16) {
            Adds(index, rowIndex, static_cast<uint32_t>(blockRow + 3 * rowsPerLoad), all);
            Gather(g, safeMaximum, index);
            LoadAlign(local, localMax + blockRow * (N / 32) + 3 * 64);
            LoadAlign(sum, groupSum + blockRow * (N / 32) + 3 * 64);
            Compares<uint32_t, CMPMODE::LT>(rowValid, index, validRows, all);
            Compares<float, CMPMODE::GT>(valid, local, NEG_INF, all);
            MaskAnd(valid, valid, rowValid, all);
            Select(local, local, zero, valid);
            Select(g, g, zero, valid);
            Select(sum, sum, zero, valid);
            Sub(exponent, local, g, all);
            // Compute the FP32 weight before the E8M0 lower clamp. Small groups can
            // contribute to rowsum even when their quantized PV scale is zero.
            Adds(weight, exponent, scaleOffset, all);
            Muls(weight, weight, LN2, all);
            Exp(weight, weight, all);
            Mul(sum, sum, weight, all);
            if constexpr (N == 512) {
                ReduceDataBlock<ReduceType::SUM>(reduced, sum, all);
                PairReduceElem<PairReduce::SUM>(sum, reduced, eight);
            } else if constexpr (N == 256) {
                ReduceDataBlock<ReduceType::SUM>(sum, sum, all);
            } else {
                PairReduceElem<PairReduce::SUM>(reduced, sum, all);
                PairReduceElem<PairReduce::SUM>(sum, reduced, halfMask);
            }
            sumAddress = tileSum + blockRow + 3 * rowsPerLoad;
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(sumAddress, sum, sumStore3, rowsPerLoad);
            StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(sumAddress, sumStore3, 0);
            Adds(exponent, exponent, scaleBias, all);
            Maxs(exponent, exponent, 0.0f, all);
            Select(exponent, exponent, unit, valid);
            Cast<int32_t, float, INTEGER_ROUND>(integer, exponent, all);
            Pack(shortCode, (RegTensor<uint32_t>&)integer);
            Pack(packed3, shortCode);
        }
        if constexpr (N == 512) {
            Interleave(lower, unused, (RegTensor<uint16_t>&)packed0, (RegTensor<uint16_t>&)packed1);
            Interleave(upper, unused, (RegTensor<uint16_t>&)packed2, (RegTensor<uint16_t>&)packed3);
            Interleave(pairs, unused, lower, upper);
            Gather(reordered, pairs, order);
        } else if constexpr (N == 256) {
            Interleave(pairs, unused, (RegTensor<uint16_t>&)packed0, (RegTensor<uint16_t>&)packed1);
            Gather(reordered, pairs, order);
        } else {
            Gather(reordered, (RegTensor<uint16_t>&)packed0, order);
        }
        // zZ: [rowBlock, pair, rowWithin16, exponentWithinPair].
        StoreAlign(reinterpret_cast<__ubuf__ uint16_t*>(scales) + blockRow * (N / 64), reordered, storeMask);
    }
}

template <uint16_t N>
__aicore__ inline void StaticPScaleCallVF(const LocalTensor<float>& localMax, const LocalTensor<uint8_t>& scale,
                                          const LocalTensor<float>& maximum, const LocalTensor<float>& previous,
                                          const LocalTensor<float>& dm, const LocalTensor<float>& groupSum,
                                          const LocalTensor<float>& tileSum, uint32_t rows, float scaleOffset,
                                          float scaleBias)
{
    auto localMaxAddress = reinterpret_cast<__ubuf__ float*>(localMax.GetPhyAddr());
    auto scaleAddress = reinterpret_cast<__ubuf__ uint8_t*>(scale.GetPhyAddr());
    auto maxAddress = reinterpret_cast<__ubuf__ float*>(maximum.GetPhyAddr());
    auto previousAddress = reinterpret_cast<__ubuf__ float*>(previous.GetPhyAddr());
    auto dmAddress = reinterpret_cast<__ubuf__ float*>(dm.GetPhyAddr());
    auto groupSumAddress = reinterpret_cast<__ubuf__ float*>(groupSum.GetPhyAddr());
    auto tileSumAddress = reinterpret_cast<__ubuf__ float*>(tileSum.GetPhyAddr());
    if (rows <= 4) {
        StaticComputeTileScale32<N, 4>(localMaxAddress, scaleAddress, maxAddress, previousAddress, dmAddress,
                                       groupSumAddress, tileSumAddress, rows, scaleOffset, scaleBias);
    } else if (rows <= 8) {
        StaticComputeTileScale32<N, 8>(localMaxAddress, scaleAddress, maxAddress, previousAddress, dmAddress,
                                       groupSumAddress, tileSumAddress, rows, scaleOffset, scaleBias);
    } else {
        StaticComputeTileScale32<N, 16>(localMaxAddress, scaleAddress, maxAddress, previousAddress, dmAddress,
                                        groupSumAddress, tileSumAddress, rows, scaleOffset, scaleBias);
    }
}

__aicore__ inline void PScaleCallVF(const LocalTensor<float>& localMax, const LocalTensor<uint8_t>& scale,
                                    const LocalTensor<float>& maximum, const LocalTensor<float>& previous,
                                    const LocalTensor<float>& dm, const LocalTensor<float>& groupSum,
                                    const LocalTensor<float>& tileSum, uint32_t rows, float scaleOffset,
                                    float scaleBias, uint16_t N)
{
    if (N == 512) {
        StaticPScaleCallVF<512>(localMax, scale, maximum, previous, dm, groupSum, tileSum, rows, scaleOffset,
                                scaleBias);
    } else if (N == 256) {
        StaticPScaleCallVF<256>(localMax, scale, maximum, previous, dm, groupSum, tileSum, rows, scaleOffset,
                                scaleBias);
    } else if (N == 128) {
        StaticPScaleCallVF<128>(localMax, scale, maximum, previous, dm, groupSum, tileSum, rows, scaleOffset,
                                scaleBias);
    }
}

} // namespace NpuArch::Epilogue::Block::GbsaMxfp4VF
#endif
