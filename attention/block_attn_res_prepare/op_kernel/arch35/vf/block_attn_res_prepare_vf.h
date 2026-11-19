/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_ATTN_RES_PREPARE_VF_H
#define BLOCK_ATTN_RES_PREPARE_VF_H

#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"

namespace BlockAttnResPrepareVF {

using HighLowPart = AscendC::Reg::HighLowPart;
using LoadDist = AscendC::Reg::LoadDist;
using MaskMergeMode = AscendC::Reg::MaskMergeMode;
using MaskPattern = AscendC::Reg::MaskPattern;
using MaskReg = AscendC::Reg::MaskReg;
using ReduceType = AscendC::Reg::ReduceType;
using StoreDist = AscendC::Reg::StoreDist;

template <typename T>
using RegTensor = AscendC::Reg::RegTensor<T>;

using AscendC::Reg::Add;
using AscendC::Reg::Adds;
using AscendC::Reg::Cast;
using AscendC::Reg::CreateMask;
using AscendC::Reg::Div;
using AscendC::Reg::Duplicate;
using AscendC::Reg::Exp;
using AscendC::Reg::LoadAlign;
using AscendC::Reg::Mul;
using AscendC::Reg::MulAddDst;
using AscendC::Reg::Muls;
using AscendC::Reg::Reduce;
using AscendC::Reg::RegTraitNumOne;
using AscendC::Reg::Sqrt;
using AscendC::Reg::StoreAlign;
using AscendC::Reg::Sub;
using AscendC::Reg::UpdateMask;

constexpr uint32_t FP32_REG_ELEMS = 64U;
// D-loop expansion factors; the cached-query path preserves the per-query accumulation order.
constexpr uint32_t BF16_STATS_D_UNROLL_FACTOR = 2U;
constexpr uint32_t BF16_GROUPED_STATS_D_UNROLL_FACTOR = 4U;
constexpr uint32_t THIRD_D_CHUNK_OFFSET = 2U * FP32_REG_ELEMS;
constexpr uint32_t FOURTH_D_CHUNK_OFFSET = 3U * FP32_REG_ELEMS;
constexpr uint32_t STAT_VECTOR_COUNT = 2U;
constexpr uint32_t TWO_S_ROWS = 2U;
constexpr uint32_t THREE_S_ROWS = 3U;
constexpr uint32_t FIRST_QUERY_ROW_INDEX = 0U;
constexpr uint32_t SECOND_QUERY_ROW_INDEX = 1U;
constexpr uint32_t THIRD_QUERY_ROW_INDEX = 2U;
constexpr uint32_t ONE_N_BLOCK = 1U;
constexpr uint32_t TWO_N_BLOCKS = 2U;
constexpr uint32_t THREE_N_BLOCKS = 3U;
constexpr uint32_t FOUR_N_BLOCKS = 4U;
constexpr uint32_t FIVE_N_BLOCKS = 5U;
constexpr uint32_t FIRST_BLOCK_INDEX = 0U;
constexpr uint32_t SECOND_BLOCK_INDEX = 1U;
constexpr uint32_t THIRD_BLOCK_INDEX = 2U;
constexpr uint32_t FOURTH_BLOCK_INDEX = 3U;
constexpr uint32_t FIFTH_BLOCK_INDEX = 4U;
constexpr uint32_t SCALAR_BLOCK_ELEMS = 8U;
constexpr uint32_t V_SQUARE_SUM_OFFSET = 0U;
constexpr uint32_t DOT_OR_WEIGHT_OFFSET = FP32_REG_ELEMS;
constexpr uint32_t LOGIT_MAX_OFFSET = STAT_VECTOR_COUNT * FP32_REG_ELEMS;
constexpr uint32_t EXP_SUM_OFFSET = LOGIT_MAX_OFFSET + SCALAR_BLOCK_ELEMS;
constexpr uint32_t STAT_UB_ELEMS = EXP_SUM_OFFSET + SCALAR_BLOCK_ELEMS;
constexpr uint32_t CACHED_QUERY_STAT_ROW_ELEMS = STAT_VECTOR_COUNT * SCALAR_BLOCK_ELEMS;
constexpr float FP32_LOWEST_FINITE = -3.4028234663852886e+38F;
constexpr AscendC::Reg::DivSpecificMode BARP_DIV_0ULP_FTZ_TRUE_MODE = {
    MaskMergeMode::ZEROING,
    true,
    AscendC::DivAlgo::PRECISION_0ULP_FTZ_TRUE,
};

constexpr AscendC::Reg::CastTrait BARP_CAST_BF16_TO_FP32 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

__simd_vf__ inline void InitializeEmptyOnlineSoftmax(__ubuf__ float* statAddr)
{
    RegTensor<float> logitMaxReg;
    RegTensor<float> expSumReg;
    MaskReg scalarMask = CreateMask<float, MaskPattern::VL1>();

    Duplicate(logitMaxReg, FP32_LOWEST_FINITE, scalarMask);
    Duplicate(expSumReg, 0.0F, scalarMask);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + LOGIT_MAX_OFFSET, logitMaxReg, scalarMask);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + EXP_SUM_OFFSET, expSumReg, scalarMask);
}

__simd_vf__ inline void FillZero(__ubuf__ float* outputAddr, uint32_t validD)
{
    RegTensor<float> zeroReg;
    MaskReg allMask = CreateMask<float, MaskPattern::ALL>();
    Duplicate(zeroReg, 0.0F, allMask);

    uint32_t remainingD = validD;
    const uint16_t loopCount = static_cast<uint16_t>((validD + FP32_REG_ELEMS - 1U) / FP32_REG_ELEMS);
    for (uint16_t loop = 0; loop < loopCount; ++loop) {
        MaskReg validMask = UpdateMask<float, RegTraitNumOne>(remainingD);
        const uint32_t offset = static_cast<uint32_t>(loop) * FP32_REG_ELEMS;
        StoreAlign<float, StoreDist::DIST_NORM>(outputAddr + offset, zeroReg, validMask);
    }
}

// These raw UB arguments are internal static byte offsets. UB offset zero is valid and must not be treated as nullptr.
template <typename BLOCK_RES_DTYPE>
__simd_callee__ inline void LoadInputAsFloat(RegTensor<float>& dst, __ubuf__ BLOCK_RES_DTYPE* src, MaskReg mask)
{
    if constexpr (std::is_same<BLOCK_RES_DTYPE, bfloat16_t>::value) {
        RegTensor<bfloat16_t> srcReg;
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(srcReg, src);
        Cast<float, bfloat16_t, BARP_CAST_BF16_TO_FP32>(dst, srcReg, mask);
    } else {
        LoadAlign<float, LoadDist::DIST_NORM>(dst, reinterpret_cast<__ubuf__ float*>(src));
    }
}

template <uint32_t S_ROWS>
__simd_callee__ inline __attribute__((always_inline)) void LoadSGroupQueries(__ubuf__ float* qAddr, uint32_t qRowStride,
                                                                             RegTensor<float>& q0, RegTensor<float>& q1,
                                                                             RegTensor<float>& q2, MaskReg mask)
{
    LoadInputAsFloat<float>(q0, qAddr, mask);
    LoadInputAsFloat<float>(q1, qAddr + qRowStride, mask);
    if constexpr (S_ROWS == THREE_S_ROWS) {
        LoadInputAsFloat<float>(q2, qAddr + THIRD_QUERY_ROW_INDEX * qRowStride, mask);
    }
}

template <uint32_t S_ROWS>
__simd_callee__ inline __attribute__((always_inline)) void AccumulateSRows(
    RegTensor<float>& acc0, RegTensor<float>& acc1, RegTensor<float>& acc2, RegTensor<float>& v,
    RegTensor<float>& factor0, RegTensor<float>& factor1, RegTensor<float>& factor2, MaskReg mask)
{
    MulAddDst<float, MaskMergeMode::ZEROING>(acc0, factor0, v, mask);
    MulAddDst<float, MaskMergeMode::ZEROING>(acc1, factor1, v, mask);
    if constexpr (S_ROWS == THREE_S_ROWS) {
        MulAddDst<float, MaskMergeMode::ZEROING>(acc2, factor2, v, mask);
    }
}

__simd_callee__ inline __attribute__((always_inline)) void MergeFourAccumulators(RegTensor<float>& acc0,
                                                                                 RegTensor<float>& acc1,
                                                                                 RegTensor<float>& acc2,
                                                                                 RegTensor<float>& acc3, MaskReg mask)
{
    Add<float, MaskMergeMode::ZEROING>(acc0, acc0, acc1, mask);
    Add<float, MaskMergeMode::ZEROING>(acc2, acc2, acc3, mask);
    Add<float, MaskMergeMode::ZEROING>(acc0, acc0, acc2, mask);
}

__simd_callee__ inline __attribute__((always_inline)) void AccumulateDotRow(__ubuf__ float* qAddr, RegTensor<float>& v,
                                                                            RegTensor<float>& acc, MaskReg validMask,
                                                                            MaskReg allMask)
{
    RegTensor<float> q;
    RegTensor<float> product;
    LoadInputAsFloat<float>(q, qAddr, validMask);
    Mul<float, MaskMergeMode::ZEROING>(product, q, v, validMask);
    Add<float, MaskMergeMode::ZEROING>(acc, acc, product, allMask);
}

__simd_callee__ inline __attribute__((always_inline)) void StoreSGroupStat(__ubuf__ float* statAddr, uint32_t nIndex,
                                                                           RegTensor<float>& squareSum,
                                                                           RegTensor<float>& dotSum, MaskReg scalarMask)
{
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + V_SQUARE_SUM_OFFSET + nIndex, squareSum,
                                                         scalarMask);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + DOT_OR_WEIGHT_OFFSET + nIndex, dotSum, scalarMask);
}

template <typename BLOCK_RES_DTYPE>
__simd_callee__ inline __attribute__((always_inline)) void AccumulateVPairDot(
    __ubuf__ float* qAddr, RegTensor<float>& v0, RegTensor<float>& v1, RegTensor<float>& acc0, RegTensor<float>& acc1,
    MaskReg validMask, MaskReg allMask)
{
    RegTensor<float> q;
    LoadInputAsFloat<float>(q, qAddr, validMask);
    if constexpr (std::is_same<BLOCK_RES_DTYPE, bfloat16_t>::value) {
        MulAddDst<float, MaskMergeMode::ZEROING>(acc0, q, v0, allMask);
        MulAddDst<float, MaskMergeMode::ZEROING>(acc1, q, v1, allMask);
    } else {
        RegTensor<float> product;
        Mul<float, MaskMergeMode::ZEROING>(product, q, v0, validMask);
        Add<float, MaskMergeMode::ZEROING>(acc0, acc0, product, allMask);
        Mul<float, MaskMergeMode::ZEROING>(product, q, v1, validMask);
        Add<float, MaskMergeMode::ZEROING>(acc1, acc1, product, allMask);
    }
}

__simd_callee__ inline __attribute__((always_inline)) void StoreVPairStat(
    __ubuf__ float* statAddr, uint32_t nIndex, RegTensor<float>& square0, RegTensor<float>& square1,
    RegTensor<float>& dotAcc0, RegTensor<float>& dotAcc1, MaskReg allMask, MaskReg scalarMask)
{
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + V_SQUARE_SUM_OFFSET + nIndex, square0, scalarMask);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + V_SQUARE_SUM_OFFSET + nIndex + SECOND_BLOCK_INDEX,
                                                         square1, scalarMask);
    RegTensor<float> dotSum;
    Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(dotSum, dotAcc0, allMask);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + DOT_OR_WEIGHT_OFFSET + nIndex, dotSum, scalarMask);
    Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(dotSum, dotAcc1, allMask);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + DOT_OR_WEIGHT_OFFSET + nIndex + SECOND_BLOCK_INDEX,
                                                         dotSum, scalarMask);
}

__simd_callee__ inline __attribute__((always_inline)) void LoadThreeSWeights(__ubuf__ float* weightAddr,
                                                                             uint32_t statRowStride,
                                                                             RegTensor<float>& weight0,
                                                                             RegTensor<float>& weight1,
                                                                             RegTensor<float>& weight2)
{
    LoadAlign<float, LoadDist::DIST_BRC_B32>(weight0, weightAddr);
    LoadAlign<float, LoadDist::DIST_BRC_B32>(weight1, weightAddr + statRowStride);
    LoadAlign<float, LoadDist::DIST_BRC_B32>(weight2, weightAddr + THIRD_QUERY_ROW_INDEX * statRowStride);
}

__simd_callee__ inline __attribute__((always_inline)) void AccumulateThreeSOutputs(
    RegTensor<float>& v, RegTensor<float>& weight0, RegTensor<float>& weight1, RegTensor<float>& weight2,
    RegTensor<float>& output0, RegTensor<float>& output1, RegTensor<float>& output2, MaskReg mask)
{
    MulAddDst<float, MaskMergeMode::ZEROING>(output0, v, weight0, mask);
    MulAddDst<float, MaskMergeMode::ZEROING>(output1, v, weight1, mask);
    MulAddDst<float, MaskMergeMode::ZEROING>(output2, v, weight2, mask);
}

__simd_callee__ inline __attribute__((always_inline)) void AccumulateCachedQueryPair(
    __ubuf__ float* qAddr, RegTensor<float>& v0, RegTensor<float>& v1, RegTensor<float>& dot0, RegTensor<float>& dot1,
    MaskReg mask)
{
    RegTensor<float> q;
    LoadInputAsFloat<float>(q, qAddr, mask);
    MulAddDst<float, MaskMergeMode::ZEROING>(dot0, q, v0, mask);
    LoadInputAsFloat<float>(q, qAddr + FP32_REG_ELEMS, mask);
    MulAddDst<float, MaskMergeMode::ZEROING>(dot1, q, v1, mask);
}

__simd_callee__ inline __attribute__((always_inline)) void FinalizeCachedQuery(
    __ubuf__ float* statAddr, RegTensor<float>& dot0, RegTensor<float>& dot1, RegTensor<float>& rms,
    RegTensor<float>& one, MaskReg allMask, MaskReg scalarMask)
{
    RegTensor<float> dotSum;
    RegTensor<float> logit;
    Add<float, MaskMergeMode::ZEROING>(dot0, dot0, dot1, allMask);
    Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(dotSum, dot0, allMask);
    Div<float, &BARP_DIV_0ULP_FTZ_TRUE_MODE>(logit, dotSum, rms, scalarMask);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr, logit, scalarMask);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + SCALAR_BLOCK_ELEMS, one, scalarMask);
}

template <typename BLOCK_RES_DTYPE>
__simd_vf__ inline void AccumulateSquareSumAndDot(__ubuf__ float* qAddr, __ubuf__ BLOCK_RES_DTYPE* vAddr,
                                                  __ubuf__ float* statAddr, uint32_t validD, uint32_t nIndex)
{
    RegTensor<float> qReg;
    RegTensor<float> vReg;
    RegTensor<float> squareReg;
    RegTensor<float> dotReg;
    RegTensor<float> squareAccReg;
    RegTensor<float> dotAccReg;
    RegTensor<float> squareSumReg;
    RegTensor<float> dotSumReg;

    MaskReg allMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg scalarMask = CreateMask<float, MaskPattern::VL1>();
    Duplicate(squareAccReg, 0.0F, allMask);
    Duplicate(dotAccReg, 0.0F, allMask);

    if constexpr (std::is_same<BLOCK_RES_DTYPE, bfloat16_t>::value) {
        // Independent accumulators shorten the dependency chains of the full-register loop.
        RegTensor<float> q1Reg;
        RegTensor<float> v1Reg;
        RegTensor<float> squareAcc1Reg;
        RegTensor<float> dotAcc1;
        Duplicate(squareAcc1Reg, 0.0F, allMask);
        Duplicate(dotAcc1, 0.0F, allMask);
        const uint16_t unrollCount = static_cast<uint16_t>(validD / (BF16_STATS_D_UNROLL_FACTOR * FP32_REG_ELEMS));
        for (uint16_t loop = 0; loop < unrollCount; ++loop) {
            const uint32_t offset = static_cast<uint32_t>(loop) * BF16_STATS_D_UNROLL_FACTOR * FP32_REG_ELEMS;
            LoadInputAsFloat<float>(qReg, qAddr + offset, allMask);
            LoadInputAsFloat<BLOCK_RES_DTYPE>(vReg, vAddr + offset, allMask);
            LoadInputAsFloat<float>(q1Reg, qAddr + offset + FP32_REG_ELEMS, allMask);
            LoadInputAsFloat<BLOCK_RES_DTYPE>(v1Reg, vAddr + offset + FP32_REG_ELEMS, allMask);
            MulAddDst<float, MaskMergeMode::ZEROING>(squareAccReg, vReg, vReg, allMask);
            MulAddDst<float, MaskMergeMode::ZEROING>(dotAccReg, qReg, vReg, allMask);
            MulAddDst<float, MaskMergeMode::ZEROING>(squareAcc1Reg, v1Reg, v1Reg, allMask);
            MulAddDst<float, MaskMergeMode::ZEROING>(dotAcc1, q1Reg, v1Reg, allMask);
        }
        // A masked ZEROING FMA would erase accumulated lanes outside the tail mask.
        uint32_t tailD = validD % (BF16_STATS_D_UNROLL_FACTOR * FP32_REG_ELEMS);
        const uint16_t tailLoopCount = static_cast<uint16_t>((tailD + FP32_REG_ELEMS - 1U) / FP32_REG_ELEMS);
        for (uint16_t loop = 0; loop < tailLoopCount; ++loop) {
            MaskReg validMask = UpdateMask<float, RegTraitNumOne>(tailD);
            const uint32_t offset = static_cast<uint32_t>(unrollCount) * BF16_STATS_D_UNROLL_FACTOR * FP32_REG_ELEMS +
                                    static_cast<uint32_t>(loop) * FP32_REG_ELEMS;
            LoadInputAsFloat<float>(qReg, qAddr + offset, validMask);
            LoadInputAsFloat<BLOCK_RES_DTYPE>(vReg, vAddr + offset, validMask);
            Mul<float, MaskMergeMode::ZEROING>(squareReg, vReg, vReg, validMask);
            Mul<float, MaskMergeMode::ZEROING>(dotReg, qReg, vReg, validMask);
            Add<float, MaskMergeMode::ZEROING>(squareAccReg, squareAccReg, squareReg, allMask);
            Add<float, MaskMergeMode::ZEROING>(dotAccReg, dotAccReg, dotReg, allMask);
        }
        Add<float, MaskMergeMode::ZEROING>(squareAccReg, squareAccReg, squareAcc1Reg, allMask);
        Add<float, MaskMergeMode::ZEROING>(dotAccReg, dotAccReg, dotAcc1, allMask);
    } else {
        uint32_t remainingD = validD;
        const uint16_t loopCount = static_cast<uint16_t>((validD + FP32_REG_ELEMS - 1U) / FP32_REG_ELEMS);
        for (uint16_t loop = 0; loop < loopCount; ++loop) {
            MaskReg validMask = UpdateMask<float, RegTraitNumOne>(remainingD);
            const uint32_t offset = static_cast<uint32_t>(loop) * FP32_REG_ELEMS;
            LoadInputAsFloat<float>(qReg, qAddr + offset, validMask);
            LoadInputAsFloat<BLOCK_RES_DTYPE>(vReg, vAddr + offset, validMask);
            Mul<float, MaskMergeMode::ZEROING>(squareReg, vReg, vReg, validMask);
            Mul<float, MaskMergeMode::ZEROING>(dotReg, qReg, vReg, validMask);
            Add<float, MaskMergeMode::ZEROING>(squareAccReg, squareAccReg, squareReg, allMask);
            Add<float, MaskMergeMode::ZEROING>(dotAccReg, dotAccReg, dotReg, allMask);
        }
    }

    Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(squareSumReg, squareAccReg, allMask);
    Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(dotSumReg, dotAccReg, allMask);

    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + V_SQUARE_SUM_OFFSET + nIndex, squareSumReg,
                                                         scalarMask);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + DOT_OR_WEIGHT_OFFSET + nIndex, dotSumReg,
                                                         scalarMask);
}

template <uint32_t S_ROWS, bool FIRST_S_GROUP>
__simd_callee__ inline __attribute__((always_inline)) void AccumulateBf16SGroup(
    __ubuf__ float* qAddr, __ubuf__ bfloat16_t* vAddr, uint32_t validD, uint32_t qRowStride,
    RegTensor<float>& dotAcc0Reg, RegTensor<float>& dotAcc1Reg, RegTensor<float>& dotAcc2Reg,
    RegTensor<float>& squareAccReg, MaskReg allMask)
{
    RegTensor<float> vReg;
    RegTensor<float> squareReg;
    RegTensor<float> q0Reg;
    RegTensor<float> q1Reg;
    RegTensor<float> q2Reg;
    RegTensor<float> v1Reg;
    RegTensor<float> v2Reg;
    RegTensor<float> v3Reg;
    // Q/dot numbers index S rows; Part1..Part3 index the D unroll chunks.
    RegTensor<float> q0Part1Reg;
    RegTensor<float> q0Part2Reg;
    RegTensor<float> q0Part3Reg;
    RegTensor<float> q1Part1Reg;
    RegTensor<float> q1Part2Reg;
    RegTensor<float> q1Part3Reg;
    RegTensor<float> q2Part1Reg;
    RegTensor<float> q2Part2Reg;
    RegTensor<float> q2Part3Reg;
    RegTensor<float> dotAcc0Part1Reg;
    RegTensor<float> dotAcc0Part2Reg;
    RegTensor<float> dotAcc0Part3Reg;
    RegTensor<float> dotAcc1Part1Reg;
    RegTensor<float> dotAcc1Part2Reg;
    RegTensor<float> dotAcc1Part3Reg;
    RegTensor<float> dotAcc2Part1Reg;
    RegTensor<float> dotAcc2Part2Reg;
    RegTensor<float> dotAcc2Part3Reg;
    RegTensor<float> squareAcc1Reg;
    RegTensor<float> squareAcc2Reg;
    RegTensor<float> squareAcc3Reg;
    Duplicate(dotAcc0Part1Reg, 0.0F, allMask);
    Duplicate(dotAcc0Part2Reg, 0.0F, allMask);
    Duplicate(dotAcc0Part3Reg, 0.0F, allMask);
    Duplicate(dotAcc1Part1Reg, 0.0F, allMask);
    Duplicate(dotAcc1Part2Reg, 0.0F, allMask);
    Duplicate(dotAcc1Part3Reg, 0.0F, allMask);
    if constexpr (S_ROWS == THREE_S_ROWS) {
        Duplicate(dotAcc2Part1Reg, 0.0F, allMask);
        Duplicate(dotAcc2Part2Reg, 0.0F, allMask);
        Duplicate(dotAcc2Part3Reg, 0.0F, allMask);
    }
    if constexpr (FIRST_S_GROUP) {
        Duplicate(squareAcc1Reg, 0.0F, allMask);
        Duplicate(squareAcc2Reg, 0.0F, allMask);
        Duplicate(squareAcc3Reg, 0.0F, allMask);
    }
    const uint16_t unrollCount = static_cast<uint16_t>(validD / (BF16_GROUPED_STATS_D_UNROLL_FACTOR * FP32_REG_ELEMS));
    for (uint16_t group = 0; group < unrollCount; ++group) {
        const uint32_t offset = static_cast<uint32_t>(group) * BF16_GROUPED_STATS_D_UNROLL_FACTOR * FP32_REG_ELEMS;
        LoadInputAsFloat<bfloat16_t>(vReg, vAddr + offset, allMask);
        LoadSGroupQueries<S_ROWS>(qAddr + offset, qRowStride, q0Reg, q1Reg, q2Reg, allMask);
        LoadInputAsFloat<bfloat16_t>(v1Reg, vAddr + offset + 1U * FP32_REG_ELEMS, allMask);
        LoadSGroupQueries<S_ROWS>(qAddr + offset + 1U * FP32_REG_ELEMS, qRowStride, q0Part1Reg, q1Part1Reg, q2Part1Reg,
                                  allMask);
        LoadInputAsFloat<bfloat16_t>(v2Reg, vAddr + offset + THIRD_D_CHUNK_OFFSET, allMask);
        LoadSGroupQueries<S_ROWS>(qAddr + offset + THIRD_D_CHUNK_OFFSET, qRowStride, q0Part2Reg, q1Part2Reg, q2Part2Reg,
                                  allMask);
        LoadInputAsFloat<bfloat16_t>(v3Reg, vAddr + offset + FOURTH_D_CHUNK_OFFSET, allMask);
        LoadSGroupQueries<S_ROWS>(qAddr + offset + FOURTH_D_CHUNK_OFFSET, qRowStride, q0Part3Reg, q1Part3Reg,
                                  q2Part3Reg, allMask);
        AccumulateSRows<S_ROWS>(dotAcc0Reg, dotAcc1Reg, dotAcc2Reg, vReg, q0Reg, q1Reg, q2Reg, allMask);
        if constexpr (FIRST_S_GROUP) {
            MulAddDst<float, MaskMergeMode::ZEROING>(squareAccReg, vReg, vReg, allMask);
        }
        AccumulateSRows<S_ROWS>(dotAcc0Part1Reg, dotAcc1Part1Reg, dotAcc2Part1Reg, v1Reg, q0Part1Reg, q1Part1Reg,
                                q2Part1Reg, allMask);
        if constexpr (FIRST_S_GROUP) {
            MulAddDst<float, MaskMergeMode::ZEROING>(squareAcc1Reg, v1Reg, v1Reg, allMask);
        }
        AccumulateSRows<S_ROWS>(dotAcc0Part2Reg, dotAcc1Part2Reg, dotAcc2Part2Reg, v2Reg, q0Part2Reg, q1Part2Reg,
                                q2Part2Reg, allMask);
        if constexpr (FIRST_S_GROUP) {
            MulAddDst<float, MaskMergeMode::ZEROING>(squareAcc2Reg, v2Reg, v2Reg, allMask);
        }
        AccumulateSRows<S_ROWS>(dotAcc0Part3Reg, dotAcc1Part3Reg, dotAcc2Part3Reg, v3Reg, q0Part3Reg, q1Part3Reg,
                                q2Part3Reg, allMask);
        if constexpr (FIRST_S_GROUP) {
            MulAddDst<float, MaskMergeMode::ZEROING>(squareAcc3Reg, v3Reg, v3Reg, allMask);
        }
    }
    uint32_t tailD = validD % (BF16_GROUPED_STATS_D_UNROLL_FACTOR * FP32_REG_ELEMS);
    const uint16_t tailLoopCount = static_cast<uint16_t>((tailD + FP32_REG_ELEMS - 1U) / FP32_REG_ELEMS);
    for (uint16_t loop = 0; loop < tailLoopCount; ++loop) {
        MaskReg validMask = UpdateMask<float, RegTraitNumOne>(tailD);
        const uint32_t offset =
            static_cast<uint32_t>(unrollCount) * BF16_GROUPED_STATS_D_UNROLL_FACTOR * FP32_REG_ELEMS +
            static_cast<uint32_t>(loop) * FP32_REG_ELEMS;
        LoadInputAsFloat<bfloat16_t>(vReg, vAddr + offset, validMask);
        AccumulateDotRow(qAddr + FIRST_QUERY_ROW_INDEX * qRowStride + offset, vReg, dotAcc0Reg, validMask, allMask);
        AccumulateDotRow(qAddr + SECOND_QUERY_ROW_INDEX * qRowStride + offset, vReg, dotAcc1Reg, validMask, allMask);
        if constexpr (S_ROWS == THREE_S_ROWS) {
            AccumulateDotRow(qAddr + THIRD_QUERY_ROW_INDEX * qRowStride + offset, vReg, dotAcc2Reg, validMask, allMask);
        }
        if constexpr (FIRST_S_GROUP) {
            Mul<float, MaskMergeMode::ZEROING>(squareReg, vReg, vReg, validMask);
            Add<float, MaskMergeMode::ZEROING>(squareAccReg, squareAccReg, squareReg, allMask);
        }
    }
    MergeFourAccumulators(dotAcc0Reg, dotAcc0Part1Reg, dotAcc0Part2Reg, dotAcc0Part3Reg, allMask);
    MergeFourAccumulators(dotAcc1Reg, dotAcc1Part1Reg, dotAcc1Part2Reg, dotAcc1Part3Reg, allMask);
    if constexpr (S_ROWS == THREE_S_ROWS) {
        MergeFourAccumulators(dotAcc2Reg, dotAcc2Part1Reg, dotAcc2Part2Reg, dotAcc2Part3Reg, allMask);
    }
    if constexpr (FIRST_S_GROUP) {
        MergeFourAccumulators(squareAccReg, squareAcc1Reg, squareAcc2Reg, squareAcc3Reg, allMask);
    }
}

template <uint32_t S_ROWS, bool FIRST_S_GROUP>
__simd_callee__ inline __attribute__((always_inline)) void AccumulateFp32SGroup(
    __ubuf__ float* qAddr, __ubuf__ float* vAddr, uint32_t validD, uint32_t qRowStride, RegTensor<float>& dotAcc0Reg,
    RegTensor<float>& dotAcc1Reg, RegTensor<float>& dotAcc2Reg, RegTensor<float>& squareAccReg, MaskReg allMask)
{
    RegTensor<float> vReg;
    RegTensor<float> squareReg;
    uint32_t remainingD = validD;
    const uint16_t loopCount = static_cast<uint16_t>((validD + FP32_REG_ELEMS - 1U) / FP32_REG_ELEMS);
    for (uint16_t loop = 0; loop < loopCount; ++loop) {
        MaskReg validMask = UpdateMask<float, RegTraitNumOne>(remainingD);
        const uint32_t offset = static_cast<uint32_t>(loop) * FP32_REG_ELEMS;
        LoadInputAsFloat<float>(vReg, vAddr + offset, validMask);
        AccumulateDotRow(qAddr + FIRST_QUERY_ROW_INDEX * qRowStride + offset, vReg, dotAcc0Reg, validMask, allMask);
        AccumulateDotRow(qAddr + SECOND_QUERY_ROW_INDEX * qRowStride + offset, vReg, dotAcc1Reg, validMask, allMask);
        if constexpr (S_ROWS == THREE_S_ROWS) {
            AccumulateDotRow(qAddr + THIRD_QUERY_ROW_INDEX * qRowStride + offset, vReg, dotAcc2Reg, validMask, allMask);
        }
        if constexpr (FIRST_S_GROUP) {
            Mul<float, MaskMergeMode::ZEROING>(squareReg, vReg, vReg, validMask);
            Add<float, MaskMergeMode::ZEROING>(squareAccReg, squareAccReg, squareReg, allMask);
        }
    }
}

template <typename BLOCK_RES_DTYPE, uint32_t S_ROWS, bool FIRST_S_GROUP>
__simd_vf__ inline void AccumulateSGroupStats(__ubuf__ float* qAddr, __ubuf__ BLOCK_RES_DTYPE* vAddr,
                                              __ubuf__ float* statAddr, __ubuf__ float* vSquareSumAddr, uint32_t validD,
                                              uint32_t nIndex, uint32_t qRowStride, uint32_t statRowStride)
{
    static_assert(S_ROWS == TWO_S_ROWS || S_ROWS == THREE_S_ROWS, "Grouped stats require two or three rows");
    RegTensor<float> dotAcc0Reg;
    RegTensor<float> dotAcc1Reg;
    RegTensor<float> dotAcc2Reg;
    RegTensor<float> dotSum0Reg;
    RegTensor<float> dotSum1Reg;
    RegTensor<float> dotSum2Reg;
    RegTensor<float> squareAccReg;
    RegTensor<float> vSquareSumReg;
    MaskReg allMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg scalarMask = CreateMask<float, MaskPattern::VL1>();
    Duplicate(dotAcc0Reg, 0.0F, allMask);
    Duplicate(dotAcc1Reg, 0.0F, allMask);
    if constexpr (S_ROWS == THREE_S_ROWS) {
        Duplicate(dotAcc2Reg, 0.0F, allMask);
    }
    if constexpr (FIRST_S_GROUP) {
        Duplicate(squareAccReg, 0.0F, allMask);
    }
    if constexpr (std::is_same<BLOCK_RES_DTYPE, bfloat16_t>::value) {
        AccumulateBf16SGroup<S_ROWS, FIRST_S_GROUP>(qAddr, vAddr, validD, qRowStride, dotAcc0Reg, dotAcc1Reg,
                                                    dotAcc2Reg, squareAccReg, allMask);
    } else {
        AccumulateFp32SGroup<S_ROWS, FIRST_S_GROUP>(qAddr, vAddr, validD, qRowStride, dotAcc0Reg, dotAcc1Reg,
                                                    dotAcc2Reg, squareAccReg, allMask);
    }
    Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(dotSum0Reg, dotAcc0Reg, allMask);
    Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(dotSum1Reg, dotAcc1Reg, allMask);
    if constexpr (S_ROWS == THREE_S_ROWS) {
        Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(dotSum2Reg, dotAcc2Reg, allMask);
    }
    if constexpr (FIRST_S_GROUP) {
        Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(vSquareSumReg, squareAccReg, allMask);
        StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(vSquareSumAddr + nIndex, vSquareSumReg, scalarMask);
    } else {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(vSquareSumReg, vSquareSumAddr + nIndex);
    }
    StoreSGroupStat(statAddr + FIRST_QUERY_ROW_INDEX * statRowStride, nIndex, vSquareSumReg, dotSum0Reg, scalarMask);
    StoreSGroupStat(statAddr + SECOND_QUERY_ROW_INDEX * statRowStride, nIndex, vSquareSumReg, dotSum1Reg, scalarMask);
    if constexpr (S_ROWS == THREE_S_ROWS) {
        StoreSGroupStat(statAddr + THIRD_QUERY_ROW_INDEX * statRowStride, nIndex, vSquareSumReg, dotSum2Reg,
                        scalarMask);
    }
}

template <typename BLOCK_RES_DTYPE, uint32_t S_ROWS, bool FIRST_S_GROUP>
__simd_vf__ inline void AccumulateVPairStats(__ubuf__ float* qAddr, __ubuf__ BLOCK_RES_DTYPE* v0Addr,
                                             __ubuf__ BLOCK_RES_DTYPE* v1Addr, __ubuf__ float* statAddr,
                                             __ubuf__ float* vSquareSumAddr, uint32_t validD, uint32_t nIndex,
                                             uint32_t qRowStride, uint32_t statRowStride)
{
    // Reusing V across S requires validD to be a multiple of FP32_REG_ELEMS.
    static_assert(S_ROWS == TWO_S_ROWS || S_ROWS == THREE_S_ROWS, "Grouped stats require two or three rows");
    RegTensor<float> v0Reg;
    RegTensor<float> v1Reg;
    RegTensor<float> dotReg;
    // S indexes query rows and N indexes the two V blocks processed together.
    RegTensor<float> dotAccS0N0Reg;
    RegTensor<float> dotAccS0N1Reg;
    RegTensor<float> dotAccS1N0Reg;
    RegTensor<float> dotAccS1N1Reg;
    RegTensor<float> dotAccS2N0Reg;
    RegTensor<float> dotAccS2N1Reg;
    RegTensor<float> squareAcc0Reg;
    RegTensor<float> squareAcc1Reg;
    RegTensor<float> vSquareSum0Reg;
    RegTensor<float> vSquareSum1Reg;
    MaskReg allMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg scalarMask = CreateMask<float, MaskPattern::VL1>();
    Duplicate(dotAccS0N0Reg, 0.0F, allMask);
    Duplicate(dotAccS0N1Reg, 0.0F, allMask);
    Duplicate(dotAccS1N0Reg, 0.0F, allMask);
    Duplicate(dotAccS1N1Reg, 0.0F, allMask);
    if constexpr (S_ROWS == THREE_S_ROWS) {
        Duplicate(dotAccS2N0Reg, 0.0F, allMask);
        Duplicate(dotAccS2N1Reg, 0.0F, allMask);
    }
    if constexpr (FIRST_S_GROUP) {
        Duplicate(squareAcc0Reg, 0.0F, allMask);
        Duplicate(squareAcc1Reg, 0.0F, allMask);
    }
    uint32_t remainingD = validD;
    const uint16_t loopCount = static_cast<uint16_t>((validD + FP32_REG_ELEMS - 1U) / FP32_REG_ELEMS);
    for (uint16_t loop = 0; loop < loopCount; ++loop) {
        MaskReg validMask = UpdateMask<float, RegTraitNumOne>(remainingD);
        const uint32_t offset = static_cast<uint32_t>(loop) * FP32_REG_ELEMS;
        LoadInputAsFloat<BLOCK_RES_DTYPE>(v0Reg, v0Addr + offset, validMask);
        LoadInputAsFloat<BLOCK_RES_DTYPE>(v1Reg, v1Addr + offset, validMask);
        AccumulateVPairDot<BLOCK_RES_DTYPE>(qAddr + FIRST_QUERY_ROW_INDEX * qRowStride + offset, v0Reg, v1Reg,
                                            dotAccS0N0Reg, dotAccS0N1Reg, validMask, allMask);
        AccumulateVPairDot<BLOCK_RES_DTYPE>(qAddr + SECOND_QUERY_ROW_INDEX * qRowStride + offset, v0Reg, v1Reg,
                                            dotAccS1N0Reg, dotAccS1N1Reg, validMask, allMask);
        if constexpr (S_ROWS == THREE_S_ROWS) {
            AccumulateVPairDot<BLOCK_RES_DTYPE>(qAddr + THIRD_QUERY_ROW_INDEX * qRowStride + offset, v0Reg, v1Reg,
                                                dotAccS2N0Reg, dotAccS2N1Reg, validMask, allMask);
        }
        if constexpr (FIRST_S_GROUP) {
            if constexpr (std::is_same<BLOCK_RES_DTYPE, bfloat16_t>::value) {
                MulAddDst<float, MaskMergeMode::ZEROING>(squareAcc0Reg, v0Reg, v0Reg, allMask);
                MulAddDst<float, MaskMergeMode::ZEROING>(squareAcc1Reg, v1Reg, v1Reg, allMask);
            } else {
                Mul<float, MaskMergeMode::ZEROING>(dotReg, v0Reg, v0Reg, validMask);
                Add<float, MaskMergeMode::ZEROING>(squareAcc0Reg, squareAcc0Reg, dotReg, allMask);
                Mul<float, MaskMergeMode::ZEROING>(dotReg, v1Reg, v1Reg, validMask);
                Add<float, MaskMergeMode::ZEROING>(squareAcc1Reg, squareAcc1Reg, dotReg, allMask);
            }
        }
    }
    if constexpr (FIRST_S_GROUP) {
        Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(vSquareSum0Reg, squareAcc0Reg, allMask);
        Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(vSquareSum1Reg, squareAcc1Reg, allMask);
        StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(vSquareSumAddr + nIndex, vSquareSum0Reg, scalarMask);
        StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(vSquareSumAddr + nIndex + SECOND_BLOCK_INDEX,
                                                             vSquareSum1Reg, scalarMask);
    } else {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(vSquareSum0Reg, vSquareSumAddr + nIndex);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(vSquareSum1Reg, vSquareSumAddr + nIndex + SECOND_BLOCK_INDEX);
    }
    StoreVPairStat(statAddr + FIRST_QUERY_ROW_INDEX * statRowStride, nIndex, vSquareSum0Reg, vSquareSum1Reg,
                   dotAccS0N0Reg, dotAccS0N1Reg, allMask, scalarMask);
    StoreVPairStat(statAddr + SECOND_QUERY_ROW_INDEX * statRowStride, nIndex, vSquareSum0Reg, vSquareSum1Reg,
                   dotAccS1N0Reg, dotAccS1N1Reg, allMask, scalarMask);
    if constexpr (S_ROWS == THREE_S_ROWS) {
        StoreVPairStat(statAddr + THIRD_QUERY_ROW_INDEX * statRowStride, nIndex, vSquareSum0Reg, vSquareSum1Reg,
                       dotAccS2N0Reg, dotAccS2N1Reg, allMask, scalarMask);
    }
}

template <typename BLOCK_RES_DTYPE>
__simd_vf__ inline void AccumulateDotWithCachedSquareSum(__ubuf__ float* qAddr, __ubuf__ BLOCK_RES_DTYPE* vAddr,
                                                         __ubuf__ float* statAddr, __ubuf__ float* vSquareSumAddr,
                                                         uint32_t validD, uint32_t nIndex)
{
    RegTensor<float> qReg;
    RegTensor<float> vReg;
    RegTensor<float> dotReg;
    RegTensor<float> dotAccReg;
    RegTensor<float> dotSumReg;
    RegTensor<float> vSquareSumReg;
    MaskReg allMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg scalarMask = CreateMask<float, MaskPattern::VL1>();
    Duplicate(dotAccReg, 0.0F, allMask);

    if constexpr (std::is_same<BLOCK_RES_DTYPE, bfloat16_t>::value) {
        RegTensor<float> q1Reg;
        RegTensor<float> v1Reg;
        RegTensor<float> dotAcc1Reg;
        Duplicate(dotAcc1Reg, 0.0F, allMask);
        const uint16_t unrollCount = static_cast<uint16_t>(validD / (BF16_STATS_D_UNROLL_FACTOR * FP32_REG_ELEMS));
        for (uint16_t loop = 0; loop < unrollCount; ++loop) {
            const uint32_t offset = static_cast<uint32_t>(loop) * BF16_STATS_D_UNROLL_FACTOR * FP32_REG_ELEMS;
            LoadInputAsFloat<float>(qReg, qAddr + offset, allMask);
            LoadInputAsFloat<BLOCK_RES_DTYPE>(vReg, vAddr + offset, allMask);
            LoadInputAsFloat<float>(q1Reg, qAddr + offset + FP32_REG_ELEMS, allMask);
            LoadInputAsFloat<BLOCK_RES_DTYPE>(v1Reg, vAddr + offset + FP32_REG_ELEMS, allMask);
            MulAddDst<float, MaskMergeMode::ZEROING>(dotAccReg, qReg, vReg, allMask);
            MulAddDst<float, MaskMergeMode::ZEROING>(dotAcc1Reg, q1Reg, v1Reg, allMask);
        }
        // Keep prior accumulator lanes when processing the final partial register.
        uint32_t tailD = validD % (BF16_STATS_D_UNROLL_FACTOR * FP32_REG_ELEMS);
        const uint16_t tailLoopCount = static_cast<uint16_t>((tailD + FP32_REG_ELEMS - 1U) / FP32_REG_ELEMS);
        for (uint16_t loop = 0; loop < tailLoopCount; ++loop) {
            MaskReg validMask = UpdateMask<float, RegTraitNumOne>(tailD);
            const uint32_t offset = static_cast<uint32_t>(unrollCount) * BF16_STATS_D_UNROLL_FACTOR * FP32_REG_ELEMS +
                                    static_cast<uint32_t>(loop) * FP32_REG_ELEMS;
            LoadInputAsFloat<float>(qReg, qAddr + offset, validMask);
            LoadInputAsFloat<BLOCK_RES_DTYPE>(vReg, vAddr + offset, validMask);
            Mul<float, MaskMergeMode::ZEROING>(dotReg, qReg, vReg, validMask);
            Add<float, MaskMergeMode::ZEROING>(dotAccReg, dotAccReg, dotReg, allMask);
        }
        Add<float, MaskMergeMode::ZEROING>(dotAccReg, dotAccReg, dotAcc1Reg, allMask);
    } else {
        uint32_t remainingD = validD;
        const uint16_t loopCount = static_cast<uint16_t>((validD + FP32_REG_ELEMS - 1U) / FP32_REG_ELEMS);
        for (uint16_t loop = 0; loop < loopCount; ++loop) {
            MaskReg validMask = UpdateMask<float, RegTraitNumOne>(remainingD);
            const uint32_t offset = static_cast<uint32_t>(loop) * FP32_REG_ELEMS;
            LoadInputAsFloat<float>(qReg, qAddr + offset, validMask);
            LoadInputAsFloat<BLOCK_RES_DTYPE>(vReg, vAddr + offset, validMask);
            Mul<float, MaskMergeMode::ZEROING>(dotReg, qReg, vReg, validMask);
            Add<float, MaskMergeMode::ZEROING>(dotAccReg, dotAccReg, dotReg, allMask);
        }
    }
    Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(dotSumReg, dotAccReg, allMask);
    LoadAlign<float, LoadDist::DIST_BRC_B32>(vSquareSumReg, vSquareSumAddr + nIndex);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + V_SQUARE_SUM_OFFSET + nIndex, vSquareSumReg,
                                                         scalarMask);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + DOT_OR_WEIGHT_OFFSET + nIndex, dotSumReg,
                                                         scalarMask);
}

__simd_vf__ inline void FinalizeSingleBlock(__ubuf__ float* statAddr, float reciprocalD, float eps)
{
    RegTensor<float> squareSumReg;
    RegTensor<float> dotReg;
    RegTensor<float> rmsReg;
    RegTensor<float> logitReg;
    RegTensor<float> oneReg;
    MaskReg scalarMask = CreateMask<float, MaskPattern::VL1>();

    LoadAlign<float, LoadDist::DIST_NORM>(squareSumReg, statAddr + V_SQUARE_SUM_OFFSET);
    LoadAlign<float, LoadDist::DIST_NORM>(dotReg, statAddr + DOT_OR_WEIGHT_OFFSET);
    Muls<float, float, MaskMergeMode::ZEROING>(squareSumReg, squareSumReg, reciprocalD, scalarMask);
    Adds<float, float, MaskMergeMode::ZEROING>(squareSumReg, squareSumReg, eps, scalarMask);
    Sqrt<float, MaskMergeMode::ZEROING>(rmsReg, squareSumReg, scalarMask);
    Div<float, &BARP_DIV_0ULP_FTZ_TRUE_MODE>(logitReg, dotReg, rmsReg, scalarMask);
    Duplicate(oneReg, 1.0F, scalarMask);

    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + LOGIT_MAX_OFFSET, logitReg, scalarMask);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + EXP_SUM_OFFSET, oneReg, scalarMask);
}

__simd_vf__ inline void FinalizeSoftmax(__ubuf__ float* statAddr, uint32_t validN, float reciprocalD, float eps)
{
    RegTensor<float> squareSumReg;
    RegTensor<float> dotReg;
    RegTensor<float> rmsReg;
    RegTensor<float> logitReg;
    RegTensor<float> logitMaxReg;
    RegTensor<float> logitMaxBrcReg;
    RegTensor<float> expReg;
    RegTensor<float> expSumReg;

    MaskReg allMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg scalarMask = CreateMask<float, MaskPattern::VL1>();
    uint32_t remainingN = validN;
    MaskReg validMask = UpdateMask<float, RegTraitNumOne>(remainingN);

    LoadAlign<float, LoadDist::DIST_NORM>(squareSumReg, statAddr + V_SQUARE_SUM_OFFSET);
    LoadAlign<float, LoadDist::DIST_NORM>(dotReg, statAddr + DOT_OR_WEIGHT_OFFSET);
    Muls<float, float, MaskMergeMode::ZEROING>(squareSumReg, squareSumReg, reciprocalD, validMask);
    Adds<float, float, MaskMergeMode::ZEROING>(squareSumReg, squareSumReg, eps, validMask);
    Sqrt<float, MaskMergeMode::ZEROING>(rmsReg, squareSumReg, validMask);
    Div<float, &BARP_DIV_0ULP_FTZ_TRUE_MODE>(logitReg, dotReg, rmsReg, validMask);

    Reduce<ReduceType::MAX, float, float, MaskMergeMode::ZEROING>(logitMaxReg, logitReg, validMask);
    Duplicate<float, HighLowPart::LOWEST, MaskMergeMode::ZEROING>(logitMaxBrcReg, logitMaxReg, allMask);
    Sub<float, MaskMergeMode::ZEROING>(expReg, logitReg, logitMaxBrcReg, validMask);
    Exp<float, MaskMergeMode::ZEROING>(expReg, expReg, validMask);
    Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(expSumReg, expReg, validMask);

    // E reuses the physical dot buffer after Z no longer needs to be retained.
    StoreAlign<float, StoreDist::DIST_NORM>(statAddr + DOT_OR_WEIGHT_OFFSET, expReg, validMask);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + LOGIT_MAX_OFFSET, logitMaxReg, scalarMask);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + EXP_SUM_OFFSET, expSumReg, scalarMask);
}

template <typename BLOCK_RES_DTYPE, bool FIRST_N_BLOCK>
__simd_vf__ inline void WeightedAccumulate(__ubuf__ BLOCK_RES_DTYPE* vAddr, __ubuf__ float* weightAddr,
                                           __ubuf__ float* outputAddr, uint32_t validD)
{
    RegTensor<float> vReg;
    RegTensor<float> weightReg;
    RegTensor<float> outputReg;
    LoadAlign<float, LoadDist::DIST_BRC_B32>(weightReg, weightAddr);

    uint32_t remainingD = validD;
    const uint16_t loopCount = static_cast<uint16_t>((validD + FP32_REG_ELEMS - 1U) / FP32_REG_ELEMS);
    for (uint16_t loop = 0; loop < loopCount; ++loop) {
        MaskReg validMask = UpdateMask<float, RegTraitNumOne>(remainingD);
        const uint32_t offset = static_cast<uint32_t>(loop) * FP32_REG_ELEMS;
        LoadInputAsFloat<BLOCK_RES_DTYPE>(vReg, vAddr + offset, validMask);
        if constexpr (FIRST_N_BLOCK) {
            Mul<float, MaskMergeMode::ZEROING>(outputReg, vReg, weightReg, validMask);
        } else {
            LoadAlign<float, LoadDist::DIST_NORM>(outputReg, outputAddr + offset);
            MulAddDst<float, MaskMergeMode::ZEROING>(outputReg, vReg, weightReg, validMask);
        }
        StoreAlign<float, StoreDist::DIST_NORM>(outputAddr + offset, outputReg, validMask);
    }
}

template <typename BLOCK_RES_DTYPE>
__simd_vf__ inline void CopySingleBlock(__ubuf__ BLOCK_RES_DTYPE* vAddr, __ubuf__ float* outputAddr, uint32_t validD)
{
    RegTensor<float> vReg;
    uint32_t remainingD = validD;
    const uint16_t loopCount = static_cast<uint16_t>((validD + FP32_REG_ELEMS - 1U) / FP32_REG_ELEMS);
    for (uint16_t loop = 0; loop < loopCount; ++loop) {
        MaskReg validMask = UpdateMask<float, RegTraitNumOne>(remainingD);
        const uint32_t offset = static_cast<uint32_t>(loop) * FP32_REG_ELEMS;
        LoadInputAsFloat<BLOCK_RES_DTYPE>(vReg, vAddr + offset, validMask);
        StoreAlign<float, StoreDist::DIST_NORM>(outputAddr + offset, vReg, validMask);
    }
}

template <typename BLOCK_RES_DTYPE, uint32_t N_BLOCKS, bool FIRST_N_GROUP>
__simd_vf__ inline void WeightedAccumulateTwoSRows(__ubuf__ BLOCK_RES_DTYPE* vAddr, uint32_t vRowStride,
                                                   __ubuf__ float* weight0Addr, __ubuf__ float* weight1Addr,
                                                   __ubuf__ float* output0Addr, __ubuf__ float* output1Addr,
                                                   uint32_t validD)
{
    static_assert(N_BLOCKS == ONE_N_BLOCK || N_BLOCKS == TWO_N_BLOCKS || N_BLOCKS == FOUR_N_BLOCKS,
                  "Two-row output supports one, two or four blocks");
    static_assert(N_BLOCKS != ONE_N_BLOCK || !FIRST_N_GROUP, "The single-block tail requires initialized outputs");
    RegTensor<float> v0Reg;
    RegTensor<float> v1Reg;
    RegTensor<float> v2Reg;
    RegTensor<float> v3Reg;
    RegTensor<float> weightS0N0Reg;
    RegTensor<float> weightS0N1Reg;
    RegTensor<float> weightS0N2Reg;
    RegTensor<float> weightS0N3Reg;
    RegTensor<float> weightS1N0Reg;
    RegTensor<float> weightS1N1Reg;
    RegTensor<float> weightS1N2Reg;
    RegTensor<float> weightS1N3Reg;
    RegTensor<float> output0Reg;
    RegTensor<float> output1Reg;
    LoadAlign<float, LoadDist::DIST_BRC_B32>(weightS0N0Reg, weight0Addr);
    if constexpr (N_BLOCKS >= TWO_N_BLOCKS) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(weightS0N1Reg, weight0Addr + SECOND_BLOCK_INDEX);
    }
    if constexpr (N_BLOCKS == FOUR_N_BLOCKS) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(weightS0N2Reg, weight0Addr + THIRD_BLOCK_INDEX);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(weightS0N3Reg, weight0Addr + FOURTH_BLOCK_INDEX);
    }
    LoadAlign<float, LoadDist::DIST_BRC_B32>(weightS1N0Reg, weight1Addr);
    if constexpr (N_BLOCKS >= TWO_N_BLOCKS) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(weightS1N1Reg, weight1Addr + SECOND_BLOCK_INDEX);
    }
    if constexpr (N_BLOCKS == FOUR_N_BLOCKS) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(weightS1N2Reg, weight1Addr + THIRD_BLOCK_INDEX);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(weightS1N3Reg, weight1Addr + FOURTH_BLOCK_INDEX);
    }

    uint32_t remainingD = validD;
    const uint16_t loopCount = static_cast<uint16_t>((validD + FP32_REG_ELEMS - 1U) / FP32_REG_ELEMS);
    for (uint16_t loop = 0; loop < loopCount; ++loop) {
        MaskReg validMask = UpdateMask<float, RegTraitNumOne>(remainingD);
        const uint32_t offset = static_cast<uint32_t>(loop) * FP32_REG_ELEMS;
        LoadInputAsFloat<BLOCK_RES_DTYPE>(v0Reg, vAddr + offset, validMask);
        if constexpr (N_BLOCKS >= TWO_N_BLOCKS) {
            LoadInputAsFloat<BLOCK_RES_DTYPE>(v1Reg, vAddr + vRowStride + offset, validMask);
        }
        if constexpr (N_BLOCKS == FOUR_N_BLOCKS) {
            LoadInputAsFloat<BLOCK_RES_DTYPE>(v2Reg, vAddr + THIRD_BLOCK_INDEX * vRowStride + offset, validMask);
            LoadInputAsFloat<BLOCK_RES_DTYPE>(v3Reg, vAddr + FOURTH_BLOCK_INDEX * vRowStride + offset, validMask);
        }
        if constexpr (FIRST_N_GROUP) {
            Mul<float, MaskMergeMode::ZEROING>(output0Reg, v0Reg, weightS0N0Reg, validMask);
            Mul<float, MaskMergeMode::ZEROING>(output1Reg, v0Reg, weightS1N0Reg, validMask);
        } else {
            LoadAlign<float, LoadDist::DIST_NORM>(output0Reg, output0Addr + offset);
            LoadAlign<float, LoadDist::DIST_NORM>(output1Reg, output1Addr + offset);
            MulAddDst<float, MaskMergeMode::ZEROING>(output0Reg, v0Reg, weightS0N0Reg, validMask);
            MulAddDst<float, MaskMergeMode::ZEROING>(output1Reg, v0Reg, weightS1N0Reg, validMask);
        }
        if constexpr (N_BLOCKS >= TWO_N_BLOCKS) {
            MulAddDst<float, MaskMergeMode::ZEROING>(output0Reg, v1Reg, weightS0N1Reg, validMask);
        }
        if constexpr (N_BLOCKS == FOUR_N_BLOCKS) {
            MulAddDst<float, MaskMergeMode::ZEROING>(output0Reg, v2Reg, weightS0N2Reg, validMask);
            MulAddDst<float, MaskMergeMode::ZEROING>(output0Reg, v3Reg, weightS0N3Reg, validMask);
        }
        if constexpr (N_BLOCKS >= TWO_N_BLOCKS) {
            MulAddDst<float, MaskMergeMode::ZEROING>(output1Reg, v1Reg, weightS1N1Reg, validMask);
        }
        if constexpr (N_BLOCKS == FOUR_N_BLOCKS) {
            MulAddDst<float, MaskMergeMode::ZEROING>(output1Reg, v2Reg, weightS1N2Reg, validMask);
            MulAddDst<float, MaskMergeMode::ZEROING>(output1Reg, v3Reg, weightS1N3Reg, validMask);
        }
        StoreAlign<float, StoreDist::DIST_NORM>(output0Addr + offset, output0Reg, validMask);
        StoreAlign<float, StoreDist::DIST_NORM>(output1Addr + offset, output1Reg, validMask);
    }
}

template <typename BLOCK_RES_DTYPE, uint32_t N_BLOCKS>
__simd_vf__ inline void WeightedAccumulateThreeSRows(__ubuf__ BLOCK_RES_DTYPE* vAddr, uint32_t vRowStride,
                                                     __ubuf__ float* weightAddr, uint32_t statRowStride,
                                                     __ubuf__ float* output0Addr, __ubuf__ float* output1Addr,
                                                     __ubuf__ float* output2Addr, uint32_t validD)
{
    static_assert(N_BLOCKS >= THREE_N_BLOCKS && N_BLOCKS <= FIVE_N_BLOCKS,
                  "Three-S outputs require three to five V rows");
    RegTensor<float> v0Reg;
    RegTensor<float> v1Reg;
    RegTensor<float> v2Reg;
    RegTensor<float> v3Reg;
    RegTensor<float> v4Reg;
    RegTensor<float> output0Reg;
    RegTensor<float> output1Reg;
    RegTensor<float> output2Reg;
    RegTensor<float> weightS0N0Reg;
    RegTensor<float> weightS0N1Reg;
    RegTensor<float> weightS0N2Reg;
    RegTensor<float> weightS0N3Reg;
    RegTensor<float> weightS0N4Reg;
    RegTensor<float> weightS1N0Reg;
    RegTensor<float> weightS1N1Reg;
    RegTensor<float> weightS1N2Reg;
    RegTensor<float> weightS1N3Reg;
    RegTensor<float> weightS1N4Reg;
    RegTensor<float> weightS2N0Reg;
    RegTensor<float> weightS2N1Reg;
    RegTensor<float> weightS2N2Reg;
    RegTensor<float> weightS2N3Reg;
    RegTensor<float> weightS2N4Reg;
    LoadThreeSWeights(weightAddr + FIRST_BLOCK_INDEX, statRowStride, weightS0N0Reg, weightS1N0Reg, weightS2N0Reg);
    LoadThreeSWeights(weightAddr + SECOND_BLOCK_INDEX, statRowStride, weightS0N1Reg, weightS1N1Reg, weightS2N1Reg);
    LoadThreeSWeights(weightAddr + THIRD_BLOCK_INDEX, statRowStride, weightS0N2Reg, weightS1N2Reg, weightS2N2Reg);
    if constexpr (N_BLOCKS >= FOUR_N_BLOCKS) {
        LoadThreeSWeights(weightAddr + FOURTH_BLOCK_INDEX, statRowStride, weightS0N3Reg, weightS1N3Reg, weightS2N3Reg);
    }
    if constexpr (N_BLOCKS == FIVE_N_BLOCKS) {
        LoadThreeSWeights(weightAddr + FIFTH_BLOCK_INDEX, statRowStride, weightS0N4Reg, weightS1N4Reg, weightS2N4Reg);
    }
    uint32_t remainingD = validD;
    const uint16_t loopCount = static_cast<uint16_t>((validD + FP32_REG_ELEMS - 1U) / FP32_REG_ELEMS);
    for (uint16_t loop = 0; loop < loopCount; ++loop) {
        MaskReg validMask = UpdateMask<float, RegTraitNumOne>(remainingD);
        const uint32_t offset = static_cast<uint32_t>(loop) * FP32_REG_ELEMS;
        LoadInputAsFloat<BLOCK_RES_DTYPE>(v0Reg, vAddr + offset, validMask);
        LoadInputAsFloat<BLOCK_RES_DTYPE>(v1Reg, vAddr + vRowStride + offset, validMask);
        LoadInputAsFloat<BLOCK_RES_DTYPE>(v2Reg, vAddr + THIRD_BLOCK_INDEX * vRowStride + offset, validMask);
        Mul<float, MaskMergeMode::ZEROING>(output0Reg, v0Reg, weightS0N0Reg, validMask);
        Mul<float, MaskMergeMode::ZEROING>(output1Reg, v0Reg, weightS1N0Reg, validMask);
        Mul<float, MaskMergeMode::ZEROING>(output2Reg, v0Reg, weightS2N0Reg, validMask);
        AccumulateThreeSOutputs(v1Reg, weightS0N1Reg, weightS1N1Reg, weightS2N1Reg, output0Reg, output1Reg, output2Reg,
                                validMask);
        AccumulateThreeSOutputs(v2Reg, weightS0N2Reg, weightS1N2Reg, weightS2N2Reg, output0Reg, output1Reg, output2Reg,
                                validMask);
        if constexpr (N_BLOCKS >= FOUR_N_BLOCKS) {
            LoadInputAsFloat<BLOCK_RES_DTYPE>(v3Reg, vAddr + FOURTH_BLOCK_INDEX * vRowStride + offset, validMask);
            AccumulateThreeSOutputs(v3Reg, weightS0N3Reg, weightS1N3Reg, weightS2N3Reg, output0Reg, output1Reg,
                                    output2Reg, validMask);
        }
        if constexpr (N_BLOCKS == FIVE_N_BLOCKS) {
            LoadInputAsFloat<BLOCK_RES_DTYPE>(v4Reg, vAddr + FIFTH_BLOCK_INDEX * vRowStride + offset, validMask);
            AccumulateThreeSOutputs(v4Reg, weightS0N4Reg, weightS1N4Reg, weightS2N4Reg, output0Reg, output1Reg,
                                    output2Reg, validMask);
        }
        StoreAlign<float, StoreDist::DIST_NORM>(output0Addr + offset, output0Reg, validMask);
        StoreAlign<float, StoreDist::DIST_NORM>(output1Addr + offset, output1Reg, validMask);
        StoreAlign<float, StoreDist::DIST_NORM>(output2Addr + offset, output2Reg, validMask);
    }
}

// Tiling requires D to be a multiple of FP32_REG_ELEMS.
// Preserve the two-VL dot/square accumulation order of the single-N row path.
__simd_vf__ inline void PrepareThreeCachedQueries(__ubuf__ float* qAddr, __ubuf__ bfloat16_t* vAddr,
                                                  __ubuf__ float* output, __ubuf__ float* stat, uint32_t d,
                                                  float reciprocalD, float eps, uint16_t tailLoops)
{
    MaskReg allMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg scalarMask = CreateMask<float, MaskPattern::VL1>();
    RegTensor<float> v0, v1, squareA, squareB, squareSum, rms, one;
    Duplicate(squareA, 0.0F, allMask);
    Duplicate(squareB, 0.0F, allMask);
    RegTensor<float> dotA0, dotB0;
    Duplicate(dotA0, 0.0F, allMask);
    Duplicate(dotB0, 0.0F, allMask);
    RegTensor<float> dotA1, dotB1;
    Duplicate(dotA1, 0.0F, allMask);
    Duplicate(dotB1, 0.0F, allMask);
    RegTensor<float> dotA2, dotB2;
    Duplicate(dotA2, 0.0F, allMask);
    Duplicate(dotB2, 0.0F, allMask);
    const uint16_t pairs = static_cast<uint16_t>(d / (BF16_STATS_D_UNROLL_FACTOR * FP32_REG_ELEMS));
    for (uint16_t loop = 0; loop < pairs; ++loop) {
        const uint32_t offset = static_cast<uint32_t>(loop) * BF16_STATS_D_UNROLL_FACTOR * FP32_REG_ELEMS;
        LoadInputAsFloat<bfloat16_t>(v0, vAddr + offset, allMask);
        LoadInputAsFloat<bfloat16_t>(v1, vAddr + offset + FP32_REG_ELEMS, allMask);
        MulAddDst<float, MaskMergeMode::ZEROING>(squareA, v0, v0, allMask);
        MulAddDst<float, MaskMergeMode::ZEROING>(squareB, v1, v1, allMask);
        AccumulateCachedQueryPair(qAddr + FIRST_QUERY_ROW_INDEX * d + offset, v0, v1, dotA0, dotB0, allMask);
        AccumulateCachedQueryPair(qAddr + SECOND_QUERY_ROW_INDEX * d + offset, v0, v1, dotA1, dotB1, allMask);
        AccumulateCachedQueryPair(qAddr + THIRD_QUERY_ROW_INDEX * d + offset, v0, v1, dotA2, dotB2, allMask);
        StoreAlign<float, StoreDist::DIST_NORM>(output + offset, v0, allMask);
        StoreAlign<float, StoreDist::DIST_NORM>(output + offset + FP32_REG_ELEMS, v1, allMask);
    }
    uint32_t tail = d % (BF16_STATS_D_UNROLL_FACTOR * FP32_REG_ELEMS);
    for (uint16_t loop = 0U; loop < tailLoops; ++loop) {
        MaskReg mask = UpdateMask<float, RegTraitNumOne>(tail);
        const uint32_t offset = static_cast<uint32_t>(pairs) * BF16_STATS_D_UNROLL_FACTOR * FP32_REG_ELEMS;
        LoadInputAsFloat<bfloat16_t>(v0, vAddr + offset, mask);
        RegTensor<float> product;
        Mul<float, MaskMergeMode::ZEROING>(product, v0, v0, mask);
        Add<float, MaskMergeMode::ZEROING>(squareA, squareA, product, allMask);
        AccumulateDotRow(qAddr + FIRST_QUERY_ROW_INDEX * d + offset, v0, dotA0, mask, allMask);
        AccumulateDotRow(qAddr + SECOND_QUERY_ROW_INDEX * d + offset, v0, dotA1, mask, allMask);
        AccumulateDotRow(qAddr + THIRD_QUERY_ROW_INDEX * d + offset, v0, dotA2, mask, allMask);
        StoreAlign<float, StoreDist::DIST_NORM>(output + offset, v0, mask);
    }
    Add<float, MaskMergeMode::ZEROING>(squareA, squareA, squareB, allMask);
    Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(squareSum, squareA, allMask);
    Muls<float, float, MaskMergeMode::ZEROING>(squareSum, squareSum, reciprocalD, scalarMask);
    Adds<float, float, MaskMergeMode::ZEROING>(squareSum, squareSum, eps, scalarMask);
    Sqrt<float, MaskMergeMode::ZEROING>(rms, squareSum, scalarMask);
    Duplicate(one, 1.0F, scalarMask);
    FinalizeCachedQuery(stat + FIRST_QUERY_ROW_INDEX * CACHED_QUERY_STAT_ROW_ELEMS, dotA0, dotB0, rms, one, allMask,
                        scalarMask);
    FinalizeCachedQuery(stat + SECOND_QUERY_ROW_INDEX * CACHED_QUERY_STAT_ROW_ELEMS, dotA1, dotB1, rms, one, allMask,
                        scalarMask);
    FinalizeCachedQuery(stat + THIRD_QUERY_ROW_INDEX * CACHED_QUERY_STAT_ROW_ELEMS, dotA2, dotB2, rms, one, allMask,
                        scalarMask);
}

} // namespace BlockAttnResPrepareVF

#endif // BLOCK_ATTN_RES_PREPARE_VF_H
