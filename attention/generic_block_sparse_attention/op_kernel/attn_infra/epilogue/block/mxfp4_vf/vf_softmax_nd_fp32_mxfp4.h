/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_VF_SOFTMAX_ND_FP32_MXFP4_H
#define GBSA_VF_SOFTMAX_ND_FP32_MXFP4_H

#include "vf_mxfp4_common.h"

namespace NpuArch::Epilogue::Block::GbsaMxfp4VF {

// CX group32: c = ceil(max(S) / ln2 - log2(CX)), P = exp(S - c*ln2 - ln(1.5)).
// FP32 softmax with independent group32 MXFP4 exponents. P uses zN;
// localGroupMax/groupSum use [row, group]. Only valid rows run this VF.
template <uint16_t R, uint16_t N, bool FullTile, uint16_t RowsPerLoop, bool BlockReduce = false, bool Prepare = true>
__simd_vf__ inline void StaticSoftmaxNdGroup32(__ubuf__ uint8_t* p, __ubuf__ float* s, __ubuf__ float* localGroupMax,
                                               __ubuf__ float* groupSum, __ubuf__ float* groupMax,
                                               __ubuf__ float* previousMax, uint16_t validRows, uint16_t validKv,
                                               float attentionScale, float negLog2Cx, uint16_t paddingRow,
                                               bool initializePadding)
{
    static_assert(RowsPerLoop == 1 || RowsPerLoop == 2);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    UnalignReg maxStore0, maxStore1, sumStore0, sumStore1;
    UnalignReg rowMaxStore;
    __ubuf__ float* rowMaxAddress = groupMax;
    RegTensor<float> zero, negative;
    RegTensor<uint8_t> padding;
    RegTensor<float> rowMax0, rowMax1, value0, value1, previous0, previous1;
    RegTensor<float> blockMax0, blockMax1, c00, c01, c10, c11;
    RegTensor<float> saved00, saved01, saved10, saved11, baseline0, baseline1;
    RegTensor<float> blockSum0, blockSum1, sum00, sum01, sum10, sum11;
    RegTensor<bfloat16_t> expanded0, expanded1, packed0, packed1;
    RegTensor<fp4x2_e2m1_t> quantized0, quantized1;
    RegTensor<uint32_t> columns, kvIndex, reduceIndex, broadcastIndex, pairIndex;
    MaskReg all = CreateMask<float, MaskPattern::ALL>();
    MaskReg one = CreateMask<float, MaskPattern::VL1>();
    MaskReg low = CreateMask<float, MaskPattern::VL32>();
    MaskReg firstFour = CreateMask<float, MaskPattern::VL4>();
    MaskReg firstEight = CreateMask<float, MaskPattern::VL8>();
    MaskReg pack64 = CreateMask<uint16_t, MaskPattern::VL64>();
    MaskReg nextFour, kvMask, valid0;
    Duplicate(zero, 0.0f);
    Duplicate(negative, NEG_INF);
    MaskXor(nextFour, firstFour, firstEight, all);
    Arange((RegTensor<int32_t>&)columns, 0);
    if constexpr (BlockReduce) {
        ShiftRights(kvIndex, columns, static_cast<int16_t>(2), all);
        Muls(kvIndex, kvIndex, static_cast<uint32_t>(4), all);
        Sub(kvIndex, columns, kvIndex, all);
        ShiftRights(reduceIndex, columns, static_cast<int16_t>(3), all);
        Muls(reduceIndex, reduceIndex, static_cast<uint32_t>(4), all);
        Add(reduceIndex, reduceIndex, kvIndex, all);
        ShiftRights(broadcastIndex, columns, static_cast<int16_t>(5), all);
        ShiftRights(pairIndex, columns, static_cast<int16_t>(1), all);
        Muls(pairIndex, pairIndex, static_cast<uint32_t>(2), all);
        Sub(pairIndex, columns, pairIndex, all);
        Muls(pairIndex, pairIndex, static_cast<uint32_t>(32), all);
    }
    const uint16_t planes = FullTile ? N / 64 : (validKv + 63) / 64;
    const uint16_t statsGroups = FullTile || N == 128 || validKv > 128 ? N / 32 : 4;
    if constexpr (Prepare) {
        if constexpr (!FullTile) {
            for (uint16_t i = 0; i < R * statsGroups / 64U; ++i) {
                StoreAlign(localGroupMax + i * 64U, negative, all);
            }
        }
        if (initializePadding) {
            Duplicate(padding, static_cast<uint8_t>(0));
            const uint16_t bankRows = (paddingRow + 15U) / 16U * 16U;
            const uint16_t repeats = ((bankRows - paddingRow) * 32U + 255U) / 256U;
            for (uint16_t plane = 0; plane < planes; ++plane) {
                uint32_t bytes = (bankRows - paddingRow) * 32U;
                for (uint16_t i = 0; i < repeats; ++i) {
                    MaskReg paddingMask = UpdateMask<uint8_t>(bytes);
                    StoreAlign(p + plane * R * 32U + paddingRow * 32U + i * 256U, padding, paddingMask);
                }
            }
        }
    }
    const uint16_t rowLoops = validRows / RowsPerLoop;
    for (uint16_t i = 0; i < rowLoops; ++i) {
        const uint16_t row = i * RowsPerLoop;
        __ubuf__ float* maxAddress0 = localGroupMax + row * statsGroups;
        __ubuf__ float* maxAddress1 = localGroupMax + (row + 1) * statsGroups;
        __ubuf__ float* sumAddress0 = groupSum + row * statsGroups;
        __ubuf__ float* sumAddress1 = groupSum + (row + 1) * statsGroups;
        // These accumulators change within the KV loop; reset each new row.
        Duplicate(rowMax0, NEG_INF);
        if constexpr (RowsPerLoop == 2) {
            Duplicate(rowMax1, NEG_INF);
        }
        for (uint16_t plane = 0; plane < planes; ++plane) {
            kvMask = all;
            if constexpr (!FullTile) {
                Adds(kvIndex, columns, static_cast<uint32_t>(plane * 64), all);
                Compares<uint32_t, CMPMODE::LT>(kvMask, kvIndex, static_cast<uint32_t>(validKv), all);
            }
            LoadAlign(value0, s + row * N + plane * 64);
            if constexpr (RowsPerLoop == 2) {
                LoadAlign(value1, s + (row + 1) * N + plane * 64);
            }
            Muls(value0, value0, attentionScale, all);
            if constexpr (RowsPerLoop == 2) {
                Muls(value1, value1, attentionScale, all);
            }
            if constexpr (!FullTile) {
                Select(value0, value0, negative, kvMask);
                if constexpr (RowsPerLoop == 2) {
                    Select(value1, value1, negative, kvMask);
                }
            }
            // Four 8-element block maxima form each group32 exponent.
            ReduceDataBlock<ReduceType::MAX>(blockMax0, value0, all);
            if constexpr (RowsPerLoop == 2) {
                ReduceDataBlock<ReduceType::MAX>(blockMax1, value1, all);
            }
            if constexpr (BlockReduce) {
                // Repeat each four block maxima to fill one data block.
                // The two resulting maxima are broadcast to their group32 halves.
                Gather(c00, blockMax0, reduceIndex);
                ReduceDataBlock<ReduceType::MAX>(blockMax0, c00, all);
                Gather(baseline0, blockMax0, broadcastIndex);
                if constexpr (RowsPerLoop == 2) {
                    Gather(c10, blockMax1, reduceIndex);
                    ReduceDataBlock<ReduceType::MAX>(blockMax1, c10, all);
                    Gather(baseline1, blockMax1, broadcastIndex);
                }
            } else {
                ReduceMax(c00, blockMax0, firstFour);
                if constexpr (RowsPerLoop == 2) {
                    ReduceMax(c10, blockMax1, firstFour);
                }
                ReduceMax(c01, blockMax0, nextFour);
                if constexpr (RowsPerLoop == 2) {
                    ReduceMax(c11, blockMax1, nextFour);
                }
                Duplicate(c00, c00, all);
                Duplicate(c01, c01, all);
                // Each half keeps one group32 exponent; both halves share the arithmetic.
                Select(baseline0, c00, c01, low);
            }
            Compares<float, CMPMODE::GT>(valid0, baseline0, NEG_INF, all);
            Select(baseline0, baseline0, zero, valid0);
            Muls(baseline0, baseline0, INV_LN2, all);
            Adds(baseline0, baseline0, negLog2Cx, all);
            Truncate<float, RoundMode::CAST_CEIL>(baseline0, baseline0, all);
            Select(saved00, baseline0, negative, valid0);
            if constexpr (BlockReduce) {
                Gather(saved01, saved00, pairIndex);
                StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(maxAddress0, saved01, maxStore0, 2);
            } else {
                Duplicate<float, HighLowPart::HIGHEST>(saved01, saved00, one);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(localGroupMax + row * statsGroups + plane * 2,
                                                                     saved00, one);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(localGroupMax + row * statsGroups + plane * 2 + 1,
                                                                     saved01, one);
            }
            Max(rowMax0, rowMax0, saved00, all);
            Muls(baseline0, baseline0, LN2, all);
            Adds(baseline0, baseline0, LN1P5, all);
            if constexpr (RowsPerLoop == 2) {
                if constexpr (!BlockReduce) {
                    Duplicate(c10, c10, all);
                    Duplicate(c11, c11, all);
                    Select(baseline1, c10, c11, low);
                }
                Compares<float, CMPMODE::GT>(valid0, baseline1, NEG_INF, all);
                Select(baseline1, baseline1, zero, valid0);
                Muls(baseline1, baseline1, INV_LN2, all);
                Adds(baseline1, baseline1, negLog2Cx, all);
                Truncate<float, RoundMode::CAST_CEIL>(baseline1, baseline1, all);
                Select(saved10, baseline1, negative, valid0);
                if constexpr (BlockReduce) {
                    Gather(saved11, saved10, pairIndex);
                    StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(maxAddress1, saved11, maxStore1, 2);
                } else {
                    Duplicate<float, HighLowPart::HIGHEST>(saved11, saved10, one);
                    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        localGroupMax + (row + 1) * statsGroups + plane * 2, saved10, one);
                    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        localGroupMax + (row + 1) * statsGroups + plane * 2 + 1, saved11, one);
                }
                Max(rowMax1, rowMax1, saved10, all);
                Muls(baseline1, baseline1, LN2, all);
                Adds(baseline1, baseline1, LN1P5, all);
            }
            Sub(value0, value0, baseline0, all);
            if constexpr (RowsPerLoop == 2) {
                Sub(value1, value1, baseline1, all);
            }
            Exp(value0, value0, all);
            if constexpr (RowsPerLoop == 2) {
                Exp(value1, value1, all);
            }
            if constexpr (!FullTile) {
                Select(value0, value0, zero, kvMask);
                if constexpr (RowsPerLoop == 2) {
                    Select(value1, value1, zero, kvMask);
                }
            }
            // FP32 denominator is reduced before either quantizing cast.
            ReduceDataBlock<ReduceType::SUM>(blockSum0, value0, all);
            if constexpr (RowsPerLoop == 2) {
                ReduceDataBlock<ReduceType::SUM>(blockSum1, value1, all);
            }
            if constexpr (BlockReduce) {
                PairReduceElem<PairReduce::SUM>(sum00, blockSum0, firstEight);
                PairReduceElem<PairReduce::SUM>(sum01, sum00, firstFour);
                StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(sumAddress0, sum01, sumStore0, 2);
                if constexpr (RowsPerLoop == 2) {
                    PairReduceElem<PairReduce::SUM>(sum10, blockSum1, firstEight);
                    PairReduceElem<PairReduce::SUM>(sum11, sum10, firstFour);
                    StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(sumAddress1, sum11, sumStore1, 2);
                }
            } else {
                ReduceSum(sum00, blockSum0, firstFour);
                if constexpr (RowsPerLoop == 2) {
                    ReduceSum(sum10, blockSum1, firstFour);
                }
                ReduceSum(sum01, blockSum0, nextFour);
                if constexpr (RowsPerLoop == 2) {
                    ReduceSum(sum11, blockSum1, nextFour);
                }
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(groupSum + row * statsGroups + plane * 2, sum00,
                                                                     one);
                if constexpr (RowsPerLoop == 2) {
                    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(groupSum + (row + 1) * statsGroups + plane * 2,
                                                                         sum10, one);
                }
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(groupSum + row * statsGroups + plane * 2 + 1,
                                                                     sum01, one);
                if constexpr (RowsPerLoop == 2) {
                    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        groupSum + (row + 1) * statsGroups + plane * 2 + 1, sum11, one);
                }
            }
            Cast<bfloat16_t, float, BF16_ROUND>(expanded0, value0, all);
            if constexpr (RowsPerLoop == 2) {
                Cast<bfloat16_t, float, BF16_ROUND>(expanded1, value1, all);
            }
            Pack((RegTensor<uint16_t>&)packed0, (RegTensor<uint32_t>&)expanded0);
            if constexpr (RowsPerLoop == 2) {
                Pack((RegTensor<uint16_t>&)packed1, (RegTensor<uint32_t>&)expanded1);
            }
            Cast<fp4x2_e2m1_t, bfloat16_t, FP4_ROUND>(quantized0, packed0, pack64);
            if constexpr (RowsPerLoop == 2) {
                Cast<fp4x2_e2m1_t, bfloat16_t, FP4_ROUND>(quantized1, packed1, pack64);
            }
            StoreAlign<uint8_t, StoreDist::DIST_PACK4_B32>(p + plane * R * 32 + row * 32,
                                                           (RegTensor<uint8_t>&)quantized0, pack64);
            if constexpr (RowsPerLoop == 2) {
                StoreAlign<uint8_t, StoreDist::DIST_PACK4_B32>(p + plane * R * 32 + (row + 1) * 32,
                                                               (RegTensor<uint8_t>&)quantized1, pack64);
            }
        }
        if constexpr (BlockReduce) {
            StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(maxAddress0, maxStore0, 0);
            StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(sumAddress0, sumStore0, 0);
            if constexpr (RowsPerLoop == 2) {
                StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(maxAddress1, maxStore1, 0);
                StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(sumAddress1, sumStore1, 0);
            }
        }
        // Merge the two lane-wise group accumulators once per completed row.
        ReduceMax(rowMax0, rowMax0, all);
        if constexpr (RowsPerLoop == 2) {
            ReduceMax(rowMax1, rowMax1, all);
        }
        LoadAlign<float, LoadDist::DIST_BRC_B32>(previous0, previousMax + row);
        if constexpr (RowsPerLoop == 2) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(previous1, previousMax + (row + 1));
        }
        Max(rowMax0, previous0, rowMax0, all);
        if constexpr (RowsPerLoop == 2) {
            Max(rowMax1, previous1, rowMax1, all);
        }
        if constexpr (BlockReduce) {
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(groupMax + row, rowMax0, one);
        } else {
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(rowMaxAddress, rowMax0, rowMaxStore, 1);
        }
        if constexpr (RowsPerLoop == 2) {
            if constexpr (BlockReduce) {
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(groupMax + row + 1, rowMax1, one);
            } else {
                StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(rowMaxAddress, rowMax1, rowMaxStore, 1);
            }
        }
    }
    if constexpr (!BlockReduce) {
        StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(rowMaxAddress, rowMaxStore, 0);
    }
}

// Dispatch outside VF: pairs expose independent rows; the odd tail runs once.
template <uint16_t R, uint16_t N, bool FullTile, bool BlockReduce = false>
__aicore__ inline void StaticSoftmaxCallVF(const LocalTensor<uint8_t>& p, const LocalTensor<float>& s,
                                           const LocalTensor<float>& localMax, const LocalTensor<float>& groupSum,
                                           const LocalTensor<float>& groupMax, const LocalTensor<float>& previousMax,
                                           uint16_t rows, uint16_t kvs, float scale, float negLog2Cx,
                                           bool initializePadding)
{
    auto pAddress = reinterpret_cast<__ubuf__ uint8_t*>(p.GetPhyAddr());
    auto sAddress = reinterpret_cast<__ubuf__ float*>(s.GetPhyAddr());
    auto maxAddress = reinterpret_cast<__ubuf__ float*>(localMax.GetPhyAddr());
    auto sumAddress = reinterpret_cast<__ubuf__ float*>(groupSum.GetPhyAddr());
    auto groupAddress = reinterpret_cast<__ubuf__ float*>(groupMax.GetPhyAddr());
    auto previousAddress = reinterpret_cast<__ubuf__ float*>(previousMax.GetPhyAddr());
    const uint16_t statsGroups = FullTile || N == 128 || kvs > 128 ? N / 32 : 4;
    const uint16_t pairedRows = rows / 2 * 2;
    if (pairedRows != 0) {
        StaticSoftmaxNdGroup32<R, N, FullTile, 2, BlockReduce>(pAddress, sAddress, maxAddress, sumAddress, groupAddress,
                                                               previousAddress, pairedRows, kvs, scale, negLog2Cx, rows,
                                                               initializePadding);
    }
    if (rows != pairedRows) {
        if (pairedRows == 0) {
            StaticSoftmaxNdGroup32<R, N, FullTile, 1, BlockReduce>(pAddress, sAddress, maxAddress, sumAddress,
                                                                   groupAddress, previousAddress, 1, kvs, scale,
                                                                   negLog2Cx, rows, initializePadding);
        } else {
            // The paired-row call initialized the whole bank. The final odd
            // row must preserve its completed statistics and padded P rows.
            StaticSoftmaxNdGroup32<R, N, FullTile, 1, BlockReduce, false>(
                pAddress + pairedRows * 32, sAddress + pairedRows * N, maxAddress + pairedRows * statsGroups,
                sumAddress + pairedRows * statsGroups, groupAddress + pairedRows, previousAddress + pairedRows, 1, kvs,
                scale, negLog2Cx, rows, false);
        }
    }
}

template <bool FullTile>
__aicore__ inline void SoftmaxCallVF(const LocalTensor<uint8_t>& p, const LocalTensor<float>& s,
                                     const LocalTensor<float>& localMax, const LocalTensor<float>& groupSum,
                                     const LocalTensor<float>& groupMax, const LocalTensor<float>& previousMax,
                                     uint16_t rows, uint16_t kvs, float scale, float negLog2Cx, uint16_t R, uint16_t N,
                                     bool initializePadding)
{
    if (N == 512) {
        if (rows <= 8 && kvs > 128) {
            StaticSoftmaxCallVF<16, 512, FullTile, true>(p, s, localMax, groupSum, groupMax, previousMax, rows, kvs,
                                                         scale, negLog2Cx, initializePadding);
        } else {
            StaticSoftmaxCallVF<16, 512, FullTile>(p, s, localMax, groupSum, groupMax, previousMax, rows, kvs, scale,
                                                   negLog2Cx, initializePadding);
        }
    } else if (N == 256) {
        StaticSoftmaxCallVF<32, 256, FullTile>(p, s, localMax, groupSum, groupMax, previousMax, rows, kvs, scale,
                                               negLog2Cx, initializePadding);
    } else if (N == 128) {
        StaticSoftmaxCallVF<64, 128, FullTile>(p, s, localMax, groupSum, groupMax, previousMax, rows, kvs, scale,
                                               negLog2Cx, initializePadding);
    }
}

} // namespace NpuArch::Epilogue::Block::GbsaMxfp4VF
#endif
