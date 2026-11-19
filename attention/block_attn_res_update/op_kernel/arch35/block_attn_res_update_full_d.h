/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_ATTN_RES_UPDATE_FULL_D_H
#define BLOCK_ATTN_RES_UPDATE_FULL_D_H

#include <type_traits>

#include "kernel_operator.h"
#include "tensor_api/tensor.h"
#include "block_attn_res_update_tiling_data.h"

namespace BlockAttnResUpdateOps {

namespace Reg = AscendC::Reg;

constexpr uint32_t BARU_FP32_BYTES = sizeof(float);
constexpr uint32_t BARU_BF16_BYTES = sizeof(bfloat16_t);
constexpr uint32_t BARU_FP32_ALIGN_ELEMENTS = 8;
constexpr uint32_t BARU_BF16_ALIGN_ELEMENTS = 16;
constexpr uint32_t BARU_FP32_ALIGN_MASK = BARU_FP32_ALIGN_ELEMENTS - 1U;
constexpr uint32_t BARU_BF16_ALIGN_MASK = BARU_BF16_ALIGN_ELEMENTS - 1U;
// One 256-byte vector register holds 64 FP32 elements; SHIFT and MASK implement division/modulo by 64.
constexpr uint32_t BARU_VREG_FP32_ELEMENTS = 64;
constexpr uint32_t BARU_VREG_FP32_SHIFT = 6;
constexpr uint32_t BARU_VREG_FP32_MASK = BARU_VREG_FP32_ELEMENTS - 1U;
// Each generic VF iteration processes two adjacent vector registers.
constexpr uint32_t BARU_VREG_PAIR_NUM = 2;
constexpr uint32_t BARU_VREG_PAIR_ELEMENTS = BARU_VREG_PAIR_NUM * BARU_VREG_FP32_ELEMENTS;
// Two disjoint UB regions are alternated so transfers for adjacent tiles do not reuse an in-flight buffer.
constexpr uint32_t BARU_BUFFER_NUM = 2;
// The stats UB tensor stores the online-softmax state and the Phase 1 score in separate T-major planes.
constexpr uint32_t BARU_LOGIT_MAX_PLANE_INDEX = 0;
constexpr uint32_t BARU_EXP_SUM_PLANE_INDEX = 1;
constexpr uint32_t BARU_SCORE_PLANE_INDEX = 2;
constexpr uint32_t BARU_STATS_PLANE_NUM = 3;

constexpr AscendC::Reg::CastTrait BARU_CAST_BF16_TO_FP32 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::Reg::CastTrait BARU_CAST_FP32_TO_BF16 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr AscendC::Reg::DivSpecificMode BARU_DIV_0ULP_FTZ_TRUE_MODE = {
    AscendC::Reg::MaskMergeMode::ZEROING,
    true,
    AscendC::DivAlgo::PRECISION_0ULP_FTZ_TRUE,
};

// Phase 1 computes p = partial + delta and
// score = dot(p, query) / sqrt(mean(p * p) + eps), then stores p and one score per T row.
// One-VREG specialization: the query is invariant across T, so load and tail-clear it once.
__simd_vf__ inline __attribute__((always_inline)) void BlockAttnResUpdatePhase1OneVLVF(
    __ubuf__ float* partial, __ubuf__ bfloat16_t* delta, __ubuf__ float* query, __ubuf__ float* stats, uint16_t tSize,
    uint32_t dSize, uint32_t dAlignFp32, uint32_t dAlignBf16, uint32_t statsTStride, float eps, float invD,
    uint16_t hasTail)
{
    Reg::MaskReg allMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg scalarMask = Reg::CreateMask<float, Reg::MaskPattern::VL1>();
    uint32_t remainingD = dSize;
    Reg::MaskReg dMask = Reg::UpdateMask<float>(remainingD);

    Reg::RegTensor<float> partialReg;
    Reg::RegTensor<float> deltaReg;
    Reg::RegTensor<float> queryReg;
    Reg::RegTensor<float> dotReg;
    Reg::RegTensor<float> squareSumReg;
    Reg::RegTensor<float> dotSumReg;
    Reg::RegTensor<float> rmsReg;
    Reg::RegTensor<float> scoreReg;
    Reg::RegTensor<bfloat16_t> deltaBf16Reg;

    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(queryReg, query);
    // Add below zeroes inactive partial lanes. Clear query padding once so the reductions can use allMask.
    for (uint16_t loop = 0; loop < hasTail; ++loop) {
        Reg::ShiftLefts((Reg::RegTensor<uint32_t>&)queryReg, (Reg::RegTensor<uint32_t>&)queryReg,
                        static_cast<int16_t>(0), dMask);
    }

    for (uint16_t tIdx = 0; tIdx < tSize; ++tIdx) {
        const uint32_t fp32TOffset = static_cast<uint32_t>(tIdx) * dAlignFp32;
        const uint32_t bf16TOffset = static_cast<uint32_t>(tIdx) * dAlignBf16;

        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg, partial + fp32TOffset);
        Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(deltaBf16Reg, delta + bf16TOffset);
        Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(deltaReg, deltaBf16Reg, dMask);
        Reg::Add<float>(partialReg, partialReg, deltaReg, dMask);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(partial + fp32TOffset, partialReg, dMask);

        // Form the dot product before reusing partialReg for the square product.
        Reg::Mul<float>(dotReg, partialReg, queryReg, allMask);
        Reg::Mul<float>(partialReg, partialReg, partialReg, allMask);
        Reg::Reduce<Reg::ReduceType::SUM>(squareSumReg, partialReg, allMask);
        Reg::Reduce<Reg::ReduceType::SUM>(dotSumReg, dotReg, allMask);
        Reg::Muls<float>(squareSumReg, squareSumReg, invD, scalarMask);
        Reg::Adds<float>(squareSumReg, squareSumReg, eps, scalarMask);
        Reg::Sqrt<float>(rmsReg, squareSumReg, scalarMask);
        Reg::Div<float, &BARU_DIV_0ULP_FTZ_TRUE_MODE>(scoreReg, dotSumReg, rmsReg, scalarMask);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
            stats + BARU_SCORE_PLANE_INDEX * statsTStride + tIdx, scoreReg, scalarMask);
    }
}

// Two-VREG Phase 1 specialization: both query vectors stay resident across the T loop, and the second is tail-cleared
// once.
__simd_vf__ inline __attribute__((always_inline)) void BlockAttnResUpdatePhase1TwoVLVF(
    __ubuf__ float* partial, __ubuf__ bfloat16_t* delta, __ubuf__ float* query, __ubuf__ float* stats, uint16_t tSize,
    uint32_t dSize, uint32_t dAlignFp32, uint32_t dAlignBf16, uint32_t statsTStride, float eps, float invD,
    uint16_t hasTail)
{
    Reg::MaskReg allMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg scalarMask = Reg::CreateMask<float, Reg::MaskPattern::VL1>();
    uint32_t tailD = dSize - BARU_VREG_FP32_ELEMENTS;
    Reg::MaskReg tailMask = Reg::UpdateMask<float>(tailD);

    Reg::RegTensor<float> partialReg0;
    Reg::RegTensor<float> partialReg1;
    Reg::RegTensor<float> deltaReg0;
    Reg::RegTensor<float> deltaReg1;
    Reg::RegTensor<float> queryReg0;
    Reg::RegTensor<float> queryReg1;
    Reg::RegTensor<float> dotReg;
    Reg::RegTensor<float> squareSumReg;
    Reg::RegTensor<float> dotSumReg;
    Reg::RegTensor<float> rmsReg;
    Reg::RegTensor<float> scoreReg;
    Reg::RegTensor<bfloat16_t> deltaBf16Reg0;
    Reg::RegTensor<bfloat16_t> deltaBf16Reg1;

    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(queryReg0, query);
    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(queryReg1, query + BARU_VREG_FP32_ELEMENTS);
    for (uint16_t loop = 0; loop < hasTail; ++loop) {
        Reg::ShiftLefts((Reg::RegTensor<uint32_t>&)queryReg1, (Reg::RegTensor<uint32_t>&)queryReg1,
                        static_cast<int16_t>(0), tailMask);
    }

    for (uint16_t tIdx = 0; tIdx < tSize; ++tIdx) {
        const uint32_t fp32TOffset = static_cast<uint32_t>(tIdx) * dAlignFp32;
        const uint32_t bf16TOffset = static_cast<uint32_t>(tIdx) * dAlignBf16;

        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0, partial + fp32TOffset);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg1, partial + fp32TOffset + BARU_VREG_FP32_ELEMENTS);
        Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(deltaBf16Reg0, delta + bf16TOffset);
        Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(deltaBf16Reg1,
                                                                   delta + bf16TOffset + BARU_VREG_FP32_ELEMENTS);
        Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(deltaReg0, deltaBf16Reg0, allMask);
        Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(deltaReg1, deltaBf16Reg1, tailMask);
        Reg::Add<float>(partialReg0, partialReg0, deltaReg0, allMask);
        Reg::Add<float>(partialReg1, partialReg1, deltaReg1, tailMask);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(partial + fp32TOffset, partialReg0, allMask);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(partial + fp32TOffset + BARU_VREG_FP32_ELEMENTS, partialReg1,
                                                          tailMask);

        // Fold the second vector directly into the first product. queryReg1 padding was cleared above, and the
        // masked Add zeroes partialReg1 padding, so the ZEROING-only MulAddDst can safely use allMask.
        Reg::Mul<float>(dotReg, partialReg0, queryReg0, allMask);
        Reg::Mul<float>(partialReg0, partialReg0, partialReg0, allMask);
        Reg::MulAddDst<float>(dotReg, partialReg1, queryReg1, allMask);
        Reg::MulAddDst<float>(partialReg0, partialReg1, partialReg1, allMask);
        Reg::Reduce<Reg::ReduceType::SUM>(squareSumReg, partialReg0, allMask);
        Reg::Reduce<Reg::ReduceType::SUM>(dotSumReg, dotReg, allMask);
        Reg::Muls<float>(squareSumReg, squareSumReg, invD, scalarMask);
        Reg::Adds<float>(squareSumReg, squareSumReg, eps, scalarMask);
        Reg::Sqrt<float>(rmsReg, squareSumReg, scalarMask);
        Reg::Div<float, &BARU_DIV_0ULP_FTZ_TRUE_MODE>(scoreReg, dotSumReg, rmsReg, scalarMask);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
            stats + BARU_SCORE_PLANE_INDEX * statsTStride + tIdx, scoreReg, scalarMask);
    }
}

// Generic Phase 1 path. D is the effective reduction length; aligned strides are used only for UB addressing.
__simd_vf__ inline __attribute__((always_inline)) void BlockAttnResUpdatePhase1VF(
    __ubuf__ float* partial, __ubuf__ bfloat16_t* delta, __ubuf__ float* query, __ubuf__ float* stats, uint16_t tSize,
    uint32_t dSize, uint32_t dAlignFp32, uint32_t dAlignBf16, uint32_t statsTStride, float eps, float invD,
    uint16_t hasMixedPair, uint16_t hasSingleRemainder, uint16_t hasRemainder, uint32_t remainderD)
{
    Reg::MaskReg allMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg scalarMask = Reg::CreateMask<float, Reg::MaskPattern::VL1>();
    Reg::MaskReg tailMask;

    Reg::RegTensor<float> partialReg0;
    Reg::RegTensor<float> partialReg1;
    Reg::RegTensor<float> deltaReg0;
    Reg::RegTensor<float> deltaReg1;
    Reg::RegTensor<float> queryReg0;
    Reg::RegTensor<float> queryReg1;
    Reg::RegTensor<float> squareAccReg0;
    Reg::RegTensor<float> squareAccReg1;
    Reg::RegTensor<float> dotAccReg0;
    Reg::RegTensor<float> dotAccReg1;
    Reg::RegTensor<float> squareSumReg;
    Reg::RegTensor<float> dotSumReg;
    Reg::RegTensor<float> rmsReg;
    Reg::RegTensor<float> scoreReg;
    Reg::RegTensor<bfloat16_t> deltaBf16Reg0;
    Reg::RegTensor<bfloat16_t> deltaBf16Reg1;

    const uint16_t fullLoops = static_cast<uint16_t>(dSize >> BARU_VREG_FP32_SHIFT);
    const uint16_t pairFullLoops = static_cast<uint16_t>(fullLoops >> 1U);
    uint32_t remainingD = remainderD;
    for (uint16_t loop = 0; loop < hasRemainder; ++loop) {
        tailMask = Reg::UpdateMask<float>(remainingD);
    }

    for (uint16_t tIdx = 0; tIdx < tSize; ++tIdx) {
        const uint32_t fp32TOffset = static_cast<uint32_t>(tIdx) * dAlignFp32;
        const uint32_t bf16TOffset = static_cast<uint32_t>(tIdx) * dAlignBf16;

        Reg::Duplicate(squareAccReg0, 0.0F, allMask);
        Reg::Duplicate(squareAccReg1, 0.0F, allMask);
        Reg::Duplicate(dotAccReg0, 0.0F, allMask);
        Reg::Duplicate(dotAccReg1, 0.0F, allMask);

        // Process every complete 2 * VL pair first. Query vectors are loaded near use instead of being kept
        // resident across the whole T loop.
        for (uint16_t loop = 0; loop < pairFullLoops; ++loop) {
            const uint32_t dOffset0 = static_cast<uint32_t>(loop) * BARU_VREG_PAIR_ELEMENTS;
            const uint32_t dOffset1 = dOffset0 + BARU_VREG_FP32_ELEMENTS;

            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0, partial + fp32TOffset + dOffset0);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg1, partial + fp32TOffset + dOffset1);
            Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(deltaBf16Reg0, delta + bf16TOffset + dOffset0);
            Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(deltaBf16Reg1, delta + bf16TOffset + dOffset1);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(queryReg0, query + dOffset0);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(queryReg1, query + dOffset1);

            Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(deltaReg0, deltaBf16Reg0, allMask);
            Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(deltaReg1, deltaBf16Reg1, allMask);
            Reg::Add<float>(partialReg0, partialReg0, deltaReg0, allMask);
            Reg::Add<float>(partialReg1, partialReg1, deltaReg1, allMask);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(partial + fp32TOffset + dOffset0, partialReg0, allMask);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(partial + fp32TOffset + dOffset1, partialReg1, allMask);

            Reg::MulAddDst<float>(squareAccReg0, partialReg0, partialReg0, allMask);
            Reg::MulAddDst<float>(squareAccReg1, partialReg1, partialReg1, allMask);
            Reg::MulAddDst<float>(dotAccReg0, partialReg0, queryReg0, allMask);
            Reg::MulAddDst<float>(dotAccReg1, partialReg1, queryReg1, allMask);
        }

        // Pair an odd complete vector with the tail to retain two independent accumulation chains.
        for (uint16_t loop = 0; loop < hasMixedPair; ++loop) {
            const uint32_t dOffset0 = (static_cast<uint32_t>(pairFullLoops) * BARU_VREG_PAIR_NUM +
                                       static_cast<uint32_t>(loop) * BARU_VREG_PAIR_NUM) *
                                      BARU_VREG_FP32_ELEMENTS;
            const uint32_t dOffset1 = dOffset0 + BARU_VREG_FP32_ELEMENTS;
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0, partial + fp32TOffset + dOffset0);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg1, partial + fp32TOffset + dOffset1);
            Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(deltaBf16Reg0, delta + bf16TOffset + dOffset0);
            Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(deltaBf16Reg1, delta + bf16TOffset + dOffset1);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(queryReg0, query + dOffset0);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(queryReg1, query + dOffset1);
            Reg::ShiftLefts((Reg::RegTensor<uint32_t>&)queryReg1, (Reg::RegTensor<uint32_t>&)queryReg1,
                            static_cast<int16_t>(0), tailMask);
            Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(deltaReg0, deltaBf16Reg0, allMask);
            Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(deltaReg1, deltaBf16Reg1, tailMask);
            Reg::Add<float>(partialReg0, partialReg0, deltaReg0, allMask);
            Reg::Add<float>(partialReg1, partialReg1, deltaReg1, tailMask);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(partial + fp32TOffset + dOffset0, partialReg0, allMask);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(partial + fp32TOffset + dOffset1, partialReg1, tailMask);
            Reg::MulAddDst<float>(squareAccReg0, partialReg0, partialReg0, allMask);
            Reg::MulAddDst<float>(squareAccReg1, partialReg1, partialReg1, allMask);
            Reg::MulAddDst<float>(dotAccReg0, partialReg0, queryReg0, allMask);
            Reg::MulAddDst<float>(dotAccReg1, partialReg1, queryReg1, allMask);
        }

        // Odd-full-only and tail-only are mutually exclusive and share one single-VL remainder body.
        for (uint16_t loop = 0; loop < hasSingleRemainder; ++loop) {
            const uint32_t dOffset =
                (static_cast<uint32_t>(pairFullLoops) * BARU_VREG_PAIR_NUM + loop) * BARU_VREG_FP32_ELEMENTS;
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0, partial + fp32TOffset + dOffset);
            Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(deltaBf16Reg0, delta + bf16TOffset + dOffset);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(queryReg0, query + dOffset);
            Reg::ShiftLefts((Reg::RegTensor<uint32_t>&)queryReg0, (Reg::RegTensor<uint32_t>&)queryReg0,
                            static_cast<int16_t>(0), tailMask);
            Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(deltaReg0, deltaBf16Reg0, tailMask);
            Reg::Add<float>(partialReg0, partialReg0, deltaReg0, tailMask);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(partial + fp32TOffset + dOffset, partialReg0, tailMask);
            // Add and ShiftLefts clear inactive lanes before the all-mask accumulation.
            Reg::MulAddDst<float>(squareAccReg0, partialReg0, partialReg0, allMask);
            Reg::MulAddDst<float>(dotAccReg0, partialReg0, queryReg0, allMask);
        }

        Reg::Add<float>(squareAccReg0, squareAccReg0, squareAccReg1, allMask);
        Reg::Add<float>(dotAccReg0, dotAccReg0, dotAccReg1, allMask);
        Reg::Reduce<Reg::ReduceType::SUM>(squareSumReg, squareAccReg0, allMask);
        Reg::Reduce<Reg::ReduceType::SUM>(dotSumReg, dotAccReg0, allMask);
        Reg::Muls<float>(squareSumReg, squareSumReg, invD, scalarMask);
        Reg::Adds<float>(squareSumReg, squareSumReg, eps, scalarMask);
        Reg::Sqrt<float>(rmsReg, squareSumReg, scalarMask);
        Reg::Div<float, &BARU_DIV_0ULP_FTZ_TRUE_MODE>(scoreReg, dotSumReg, rmsReg, scalarMask);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
            stats + BARU_SCORE_PLANE_INDEX * statsTStride + tIdx, scoreReg, scalarMask);
    }
}

// Phase 2 performs a numerically stable online-softmax merge:
// newMax = max(logitMax, score), alpha = exp(logitMax - newMax), beta = exp(score - newMax),
// h = (numerator * alpha + partial * beta) / (expSum * alpha + beta).
// One-VREG specialization: the fixed D body removes the inner loop and builds its mask once per VF call.
__simd_vf__ inline __attribute__((always_inline)) void BlockAttnResUpdatePhase2OneVLVF(
    __ubuf__ float* partial, __ubuf__ bfloat16_t* deltaH, __ubuf__ float* numerator, __ubuf__ float* stats,
    uint16_t tSize, uint32_t dSize, uint32_t dAlignFp32, uint32_t dAlignBf16, uint32_t statsTStride)
{
    Reg::MaskReg allMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    uint32_t remainingD = dSize;
    Reg::MaskReg dMask = Reg::UpdateMask<float>(remainingD);

    Reg::RegTensor<float> partialReg;
    Reg::RegTensor<float> partialTermReg;
    Reg::RegTensor<float> scoreReg;
    Reg::RegTensor<float> logitMaxReg;
    Reg::RegTensor<float> expSumReg;
    Reg::RegTensor<float> newMaxReg;
    Reg::RegTensor<float> alphaReg;
    Reg::RegTensor<float> betaReg;
    Reg::RegTensor<float> invDenomReg;
    Reg::RegTensor<float> numeratorReg;
    Reg::RegTensor<bfloat16_t> hBf16Reg;

    for (uint16_t tIdx = 0; tIdx < tSize; ++tIdx) {
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(logitMaxReg,
                                                           stats + BARU_LOGIT_MAX_PLANE_INDEX * statsTStride + tIdx);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(expSumReg,
                                                           stats + BARU_EXP_SUM_PLANE_INDEX * statsTStride + tIdx);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(scoreReg,
                                                           stats + BARU_SCORE_PLANE_INDEX * statsTStride + tIdx);
        Reg::Max<float>(newMaxReg, logitMaxReg, scoreReg, allMask);
        Reg::ExpSub<float>(alphaReg, logitMaxReg, newMaxReg, allMask);
        Reg::ExpSub<float>(betaReg, scoreReg, newMaxReg, allMask);
        Reg::MulDstAdd<float>(expSumReg, alphaReg, betaReg, allMask);
        Reg::Duplicate(invDenomReg, 1.0F, allMask);
        Reg::Div<float, &BARU_DIV_0ULP_FTZ_TRUE_MODE>(invDenomReg, invDenomReg, expSumReg, allMask);
        Reg::Mul<float>(alphaReg, alphaReg, invDenomReg, allMask);
        Reg::Mul<float>(betaReg, betaReg, invDenomReg, allMask);

        const uint32_t fp32TOffset = static_cast<uint32_t>(tIdx) * dAlignFp32;
        const uint32_t bf16TOffset = static_cast<uint32_t>(tIdx) * dAlignBf16;
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg, partial + fp32TOffset);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(numeratorReg, numerator + fp32TOffset);
        // Match the TwoVL/Generic order: normalize the weights before forming h.
        Reg::Mul<float>(partialTermReg, partialReg, betaReg, dMask);
        Reg::MulDstAdd<float>(numeratorReg, alphaReg, partialTermReg, dMask);
        Reg::Cast<bfloat16_t, float, BARU_CAST_FP32_TO_BF16>(hBf16Reg, numeratorReg, dMask);
        Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(deltaH + bf16TOffset, hBf16Reg, dMask);
    }
}

// Two-VREG Phase 2 specialization: two independent register chains expose the fixed pair to RVEC dual issue.
__simd_vf__ inline __attribute__((always_inline)) void BlockAttnResUpdatePhase2TwoVLVF(
    __ubuf__ float* partial, __ubuf__ bfloat16_t* deltaH, __ubuf__ float* numerator, __ubuf__ float* stats,
    uint16_t tSize, uint32_t dSize, uint32_t dAlignFp32, uint32_t dAlignBf16, uint32_t statsTStride)
{
    Reg::MaskReg allMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    uint32_t tailD = dSize - BARU_VREG_FP32_ELEMENTS;
    Reg::MaskReg tailMask = Reg::UpdateMask<float>(tailD);

    Reg::RegTensor<float> partialReg0;
    Reg::RegTensor<float> partialReg1;
    Reg::RegTensor<float> partialTermReg0;
    Reg::RegTensor<float> partialTermReg1;
    Reg::RegTensor<float> scoreReg;
    Reg::RegTensor<float> logitMaxReg;
    Reg::RegTensor<float> expSumReg;
    Reg::RegTensor<float> newMaxReg;
    Reg::RegTensor<float> alphaReg;
    Reg::RegTensor<float> betaReg;
    Reg::RegTensor<float> invDenomReg;
    Reg::RegTensor<float> numeratorReg0;
    Reg::RegTensor<float> numeratorReg1;
    Reg::RegTensor<bfloat16_t> hBf16Reg0;
    Reg::RegTensor<bfloat16_t> hBf16Reg1;

    for (uint16_t tIdx = 0; tIdx < tSize; ++tIdx) {
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(logitMaxReg,
                                                           stats + BARU_LOGIT_MAX_PLANE_INDEX * statsTStride + tIdx);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(expSumReg,
                                                           stats + BARU_EXP_SUM_PLANE_INDEX * statsTStride + tIdx);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(scoreReg,
                                                           stats + BARU_SCORE_PLANE_INDEX * statsTStride + tIdx);
        Reg::Max<float>(newMaxReg, logitMaxReg, scoreReg, allMask);
        Reg::ExpSub<float>(alphaReg, logitMaxReg, newMaxReg, allMask);
        Reg::ExpSub<float>(betaReg, scoreReg, newMaxReg, allMask);
        Reg::MulDstAdd<float>(expSumReg, alphaReg, betaReg, allMask);
        Reg::Duplicate(invDenomReg, 1.0F, allMask);
        Reg::Div<float, &BARU_DIV_0ULP_FTZ_TRUE_MODE>(invDenomReg, invDenomReg, expSumReg, allMask);
        Reg::Mul<float>(alphaReg, alphaReg, invDenomReg, allMask);
        Reg::Mul<float>(betaReg, betaReg, invDenomReg, allMask);

        const uint32_t fp32TOffset = static_cast<uint32_t>(tIdx) * dAlignFp32;
        const uint32_t bf16TOffset = static_cast<uint32_t>(tIdx) * dAlignBf16;
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0, partial + fp32TOffset);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg1, partial + fp32TOffset + BARU_VREG_FP32_ELEMENTS);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(numeratorReg0, numerator + fp32TOffset);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(numeratorReg1,
                                                        numerator + fp32TOffset + BARU_VREG_FP32_ELEMENTS);
        Reg::Mul<float>(partialTermReg0, partialReg0, betaReg, allMask);
        Reg::Mul<float>(partialTermReg1, partialReg1, betaReg, tailMask);
        Reg::MulDstAdd<float>(numeratorReg0, alphaReg, partialTermReg0, allMask);
        Reg::MulDstAdd<float>(numeratorReg1, alphaReg, partialTermReg1, tailMask);
        Reg::Cast<bfloat16_t, float, BARU_CAST_FP32_TO_BF16>(hBf16Reg0, numeratorReg0, allMask);
        Reg::Cast<bfloat16_t, float, BARU_CAST_FP32_TO_BF16>(hBf16Reg1, numeratorReg1, tailMask);
        Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(deltaH + bf16TOffset, hBf16Reg0, allMask);
        Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(deltaH + bf16TOffset + BARU_VREG_FP32_ELEMENTS,
                                                                   hBf16Reg1, tailMask);
    }
}

// Generic Phase 2 path computes the online-softmax merge coefficients and forms BF16 h.
// Alpha and beta are normalized once per T row so that each D loop needs only Mul + MulDstAdd.
// Phase 1 and Phase 2 run in program order on V, so delta is fully consumed before its UB region is reused for h.
__simd_vf__ inline __attribute__((always_inline)) void BlockAttnResUpdatePhase2VF(
    __ubuf__ float* partial, __ubuf__ bfloat16_t* deltaH, __ubuf__ float* numerator, __ubuf__ float* stats,
    uint16_t tSize, uint32_t dSize, uint32_t dAlignFp32, uint32_t dAlignBf16, uint32_t statsTStride, uint16_t fullLoops,
    uint16_t hasTail, uint16_t hasMixedPair, uint16_t hasOddFullOnly, uint16_t hasTailOnly)
{
    Reg::MaskReg allMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg tailMask;
    Reg::RegTensor<float> partialReg0;
    Reg::RegTensor<float> partialReg1;
    Reg::RegTensor<float> partialTermReg0;
    Reg::RegTensor<float> partialTermReg1;
    Reg::RegTensor<float> scoreReg;
    Reg::RegTensor<float> logitMaxReg;
    Reg::RegTensor<float> expSumReg;
    Reg::RegTensor<float> newMaxReg;
    Reg::RegTensor<float> alphaReg;
    Reg::RegTensor<float> betaReg;
    Reg::RegTensor<float> invDenomReg;
    Reg::RegTensor<float> numeratorReg0;
    Reg::RegTensor<float> numeratorReg1;
    Reg::RegTensor<bfloat16_t> hBf16Reg0;
    Reg::RegTensor<bfloat16_t> hBf16Reg1;

    const uint32_t tailD = dSize & BARU_VREG_FP32_MASK;
    const uint16_t pairFullLoops = static_cast<uint16_t>(fullLoops >> 1U);
    // Full vectors reuse allMask; only the true tail needs one dynamic mask for the whole VF call.
    uint32_t remainingD = tailD;
    for (uint16_t loop = 0; loop < hasTail; ++loop) {
        tailMask = Reg::UpdateMask<float>(remainingD);
    }

    for (uint16_t tIdx = 0; tIdx < tSize; ++tIdx) {
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(logitMaxReg,
                                                           stats + BARU_LOGIT_MAX_PLANE_INDEX * statsTStride + tIdx);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(expSumReg,
                                                           stats + BARU_EXP_SUM_PLANE_INDEX * statsTStride + tIdx);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(scoreReg,
                                                           stats + BARU_SCORE_PLANE_INDEX * statsTStride + tIdx);
        Reg::Max<float>(newMaxReg, logitMaxReg, scoreReg, allMask);
        Reg::ExpSub<float>(alphaReg, logitMaxReg, newMaxReg, allMask);
        Reg::ExpSub<float>(betaReg, scoreReg, newMaxReg, allMask);
        Reg::MulDstAdd<float>(expSumReg, alphaReg, betaReg, allMask);
        Reg::Duplicate(invDenomReg, 1.0F, allMask);
        Reg::Div<float, &BARU_DIV_0ULP_FTZ_TRUE_MODE>(invDenomReg, invDenomReg, expSumReg, allMask);
        Reg::Mul<float>(alphaReg, alphaReg, invDenomReg, allMask);
        Reg::Mul<float>(betaReg, betaReg, invDenomReg, allMask);
        const uint32_t fp32TOffset = static_cast<uint32_t>(tIdx) * dAlignFp32;
        const uint32_t bf16TOffset = static_cast<uint32_t>(tIdx) * dAlignBf16;
        // Separate register sets expose two adjacent D vectors to RVEC dual issue.
        for (uint16_t loop = 0; loop < pairFullLoops; ++loop) {
            const uint32_t dOffset0 = static_cast<uint32_t>(loop) * BARU_VREG_PAIR_ELEMENTS;
            const uint32_t dOffset1 = dOffset0 + BARU_VREG_FP32_ELEMENTS;
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0, partial + fp32TOffset + dOffset0);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg1, partial + fp32TOffset + dOffset1);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(numeratorReg0, numerator + fp32TOffset + dOffset0);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(numeratorReg1, numerator + fp32TOffset + dOffset1);
            Reg::Mul<float>(partialTermReg0, partialReg0, betaReg, allMask);
            Reg::Mul<float>(partialTermReg1, partialReg1, betaReg, allMask);
            Reg::MulDstAdd<float>(numeratorReg0, alphaReg, partialTermReg0, allMask);
            Reg::MulDstAdd<float>(numeratorReg1, alphaReg, partialTermReg1, allMask);
            Reg::Cast<bfloat16_t, float, BARU_CAST_FP32_TO_BF16>(hBf16Reg0, numeratorReg0, allMask);
            Reg::Cast<bfloat16_t, float, BARU_CAST_FP32_TO_BF16>(hBf16Reg1, numeratorReg1, allMask);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(deltaH + bf16TOffset + dOffset0, hBf16Reg0,
                                                                       allMask);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(deltaH + bf16TOffset + dOffset1, hBf16Reg1,
                                                                       allMask);
        }

        // Keep an odd full vector paired with the tail so the original two-vector ILP is preserved.
        for (uint16_t loop = 0; loop < hasMixedPair; ++loop) {
            const uint32_t dOffset0 = (static_cast<uint32_t>(pairFullLoops) * BARU_VREG_PAIR_NUM +
                                       static_cast<uint32_t>(loop) * BARU_VREG_PAIR_NUM) *
                                      BARU_VREG_FP32_ELEMENTS;
            const uint32_t dOffset1 = dOffset0 + BARU_VREG_FP32_ELEMENTS;
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0, partial + fp32TOffset + dOffset0);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg1, partial + fp32TOffset + dOffset1);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(numeratorReg0, numerator + fp32TOffset + dOffset0);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(numeratorReg1, numerator + fp32TOffset + dOffset1);
            Reg::Mul<float>(partialTermReg0, partialReg0, betaReg, allMask);
            Reg::Mul<float>(partialTermReg1, partialReg1, betaReg, tailMask);
            Reg::MulDstAdd<float>(numeratorReg0, alphaReg, partialTermReg0, allMask);
            Reg::MulDstAdd<float>(numeratorReg1, alphaReg, partialTermReg1, tailMask);
            Reg::Cast<bfloat16_t, float, BARU_CAST_FP32_TO_BF16>(hBf16Reg0, numeratorReg0, allMask);
            Reg::Cast<bfloat16_t, float, BARU_CAST_FP32_TO_BF16>(hBf16Reg1, numeratorReg1, tailMask);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(deltaH + bf16TOffset + dOffset0, hBf16Reg0,
                                                                       allMask);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(deltaH + bf16TOffset + dOffset1, hBf16Reg1,
                                                                       tailMask);
        }

        for (uint16_t loop = 0; loop < hasOddFullOnly; ++loop) {
            const uint32_t dOffset =
                (static_cast<uint32_t>(pairFullLoops) * BARU_VREG_PAIR_NUM + loop) * BARU_VREG_FP32_ELEMENTS;
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0, partial + fp32TOffset + dOffset);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(numeratorReg0, numerator + fp32TOffset + dOffset);
            Reg::Mul<float>(partialTermReg0, partialReg0, betaReg, allMask);
            Reg::MulDstAdd<float>(numeratorReg0, alphaReg, partialTermReg0, allMask);
            Reg::Cast<bfloat16_t, float, BARU_CAST_FP32_TO_BF16>(hBf16Reg0, numeratorReg0, allMask);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(deltaH + bf16TOffset + dOffset, hBf16Reg0,
                                                                       allMask);
        }

        for (uint16_t loop = 0; loop < hasTailOnly; ++loop) {
            const uint32_t dOffset = (static_cast<uint32_t>(fullLoops) + loop) * BARU_VREG_FP32_ELEMENTS;
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0, partial + fp32TOffset + dOffset);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(numeratorReg0, numerator + fp32TOffset + dOffset);
            Reg::Mul<float>(partialTermReg0, partialReg0, betaReg, tailMask);
            Reg::MulDstAdd<float>(numeratorReg0, alphaReg, partialTermReg0, tailMask);
            Reg::Cast<bfloat16_t, float, BARU_CAST_FP32_TO_BF16>(hBf16Reg0, numeratorReg0, tailMask);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(deltaH + bf16TOffset + dOffset, hBf16Reg0,
                                                                       tailMask);
        }
    }
}

// Normalize the BF16 merge output in place. Both the square reduction and gamma multiplication use FP32.
// This is equivalent to applying RMSNorm after the existing BF16 BlockAttnResUpdate output.
template <typename WeightT>
__simd_vf__ inline __attribute__((always_inline)) void BlockAttnResUpdateOutputNormVF(
    __ubuf__ bfloat16_t* h, __ubuf__ WeightT* weight, uint16_t tSize, uint32_t dSize, uint32_t dAlignBf16,
    uint16_t dLoops, uint16_t hasSingleVector, float eps, float invD)
{
    Reg::MaskReg allMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg scalarMask = Reg::CreateMask<float, Reg::MaskPattern::VL1>();
    Reg::RegTensor<bfloat16_t> hBf16Reg0;
    Reg::RegTensor<bfloat16_t> hBf16Reg1;
    Reg::RegTensor<float> hReg0;
    Reg::RegTensor<float> hReg1;
    Reg::RegTensor<float> squareAccReg0;
    Reg::RegTensor<float> squareAccReg1;
    Reg::RegTensor<float> squareAccReg2;
    Reg::RegTensor<float> squareAccReg3;
    Reg::RegTensor<float> rmsReg;
    Reg::RegTensor<float> oneReg;
    Reg::RegTensor<float> invRmsReg;
    Reg::RegTensor<float> invRmsBrcReg;
    Reg::RegTensor<float> weightReg0;
    Reg::RegTensor<float> weightReg1;
    const uint16_t pairLoops = static_cast<uint16_t>(dLoops >> 1U);
    // Shift by 2 to form groups of 4 vectors for the four independent square accumulators.
    const uint16_t reduceGroups = static_cast<uint16_t>(dLoops >> 2U);
    // The low 2 bits give the remaining 0-3 vectors after all complete groups.
    const uint16_t reduceTail = static_cast<uint16_t>(dLoops & 3U);

    for (uint16_t tIdx = 0; tIdx < tSize; ++tIdx) {
        const uint32_t rowOffset = static_cast<uint32_t>(tIdx) * dAlignBf16;
        Reg::Duplicate(squareAccReg0, 0.0F, allMask);
        Reg::Duplicate(squareAccReg1, 0.0F, allMask);
        Reg::Duplicate(squareAccReg2, 0.0F, allMask);
        Reg::Duplicate(squareAccReg3, 0.0F, allMask);
        uint32_t remainingD = dSize;
        // Four independent FMA chains hide the square-accumulation dependency.
        for (uint16_t group = 0; group < reduceGroups; ++group) {
            Reg::RegTensor<bfloat16_t> reduceBf16Reg0;
            Reg::RegTensor<float> reduceReg0;
            Reg::MaskReg reduceMask0 = Reg::UpdateMask<float>(remainingD);
            // group * 4 selects the first vector in this four-vector reduction group.
            const uint32_t reduceOffset0 = static_cast<uint32_t>(group) * 4U * BARU_VREG_FP32_ELEMENTS;
            Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(reduceBf16Reg0, h + rowOffset + reduceOffset0);
            Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(reduceReg0, reduceBf16Reg0, reduceMask0);
            Reg::MulAddDst<float>(squareAccReg0, reduceReg0, reduceReg0, allMask);
            Reg::RegTensor<bfloat16_t> reduceBf16Reg1;
            Reg::RegTensor<float> reduceReg1;
            Reg::MaskReg reduceMask1 = Reg::UpdateMask<float>(remainingD);
            // +1 selects the second vector within the four-vector reduction group.
            const uint32_t reduceOffset1 = (static_cast<uint32_t>(group) * 4U + 1U) * BARU_VREG_FP32_ELEMENTS;
            Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(reduceBf16Reg1, h + rowOffset + reduceOffset1);
            Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(reduceReg1, reduceBf16Reg1, reduceMask1);
            Reg::MulAddDst<float>(squareAccReg1, reduceReg1, reduceReg1, allMask);
            Reg::RegTensor<bfloat16_t> reduceBf16Reg2;
            Reg::RegTensor<float> reduceReg2;
            Reg::MaskReg reduceMask2 = Reg::UpdateMask<float>(remainingD);
            // +2 selects the third vector within the four-vector reduction group.
            const uint32_t reduceOffset2 = (static_cast<uint32_t>(group) * 4U + 2U) * BARU_VREG_FP32_ELEMENTS;
            Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(reduceBf16Reg2, h + rowOffset + reduceOffset2);
            Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(reduceReg2, reduceBf16Reg2, reduceMask2);
            Reg::MulAddDst<float>(squareAccReg2, reduceReg2, reduceReg2, allMask);
            Reg::RegTensor<bfloat16_t> reduceBf16Reg3;
            Reg::RegTensor<float> reduceReg3;
            Reg::MaskReg reduceMask3 = Reg::UpdateMask<float>(remainingD);
            // +3 selects the fourth vector within the four-vector reduction group.
            const uint32_t reduceOffset3 = (static_cast<uint32_t>(group) * 4U + 3U) * BARU_VREG_FP32_ELEMENTS;
            Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(reduceBf16Reg3, h + rowOffset + reduceOffset3);
            Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(reduceReg3, reduceBf16Reg3, reduceMask3);
            Reg::MulAddDst<float>(squareAccReg3, reduceReg3, reduceReg3, allMask);
        }
        for (uint16_t loop = 0; loop < reduceTail; ++loop) {
            Reg::MaskReg mask = Reg::UpdateMask<float>(remainingD);
            // Each unrolled reduction group processes 4 vectors; the tail starts after reduceGroups * 4 vectors.
            const uint32_t dOffset = (static_cast<uint32_t>(reduceGroups) * 4U + loop) * BARU_VREG_FP32_ELEMENTS;
            Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(hBf16Reg0, h + rowOffset + dOffset);
            Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(hReg0, hBf16Reg0, mask);
            Reg::MulAddDst<float>(squareAccReg0, hReg0, hReg0, allMask);
        }
        Reg::Add<float>(squareAccReg0, squareAccReg0, squareAccReg1, allMask);
        Reg::Add<float>(squareAccReg2, squareAccReg2, squareAccReg3, allMask);
        Reg::Add<float>(squareAccReg0, squareAccReg0, squareAccReg2, allMask);
        Reg::Reduce<Reg::ReduceType::SUM>(rmsReg, squareAccReg0, allMask);
        Reg::Muls<float>(rmsReg, rmsReg, invD, scalarMask);
        Reg::Adds<float>(rmsReg, rmsReg, eps, scalarMask);
        Reg::Sqrt<float>(rmsReg, rmsReg, scalarMask);
        Reg::Duplicate(oneReg, 1.0F, scalarMask);
        Reg::Div<float, &BARU_DIV_0ULP_FTZ_TRUE_MODE>(invRmsReg, oneReg, rmsReg, scalarMask);
        Reg::Duplicate<float>(invRmsBrcReg, invRmsReg, allMask);

        remainingD = dSize;
        for (uint16_t pairIdx = 0; pairIdx < pairLoops; ++pairIdx) {
            Reg::MaskReg mask0 = Reg::UpdateMask<float>(remainingD);
            Reg::MaskReg mask1 = Reg::UpdateMask<float>(remainingD);
            const uint32_t dOffset = static_cast<uint32_t>(pairIdx) * BARU_VREG_PAIR_ELEMENTS;
            Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(hBf16Reg0, h + rowOffset + dOffset);
            Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(
                hBf16Reg1, h + rowOffset + dOffset + BARU_VREG_FP32_ELEMENTS);
            Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(hReg0, hBf16Reg0, mask0);
            Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(hReg1, hBf16Reg1, mask1);
            if constexpr (std::is_same<WeightT, bfloat16_t>::value) {
                Reg::RegTensor<bfloat16_t> weightBf16Reg0;
                Reg::RegTensor<bfloat16_t> weightBf16Reg1;
                Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(weightBf16Reg0, weight + dOffset);
                Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(weightBf16Reg1,
                                                                           weight + dOffset + BARU_VREG_FP32_ELEMENTS);
                Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(weightReg0, weightBf16Reg0, mask0);
                Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(weightReg1, weightBf16Reg1, mask1);
            } else {
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(weightReg0, weight + dOffset);
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(weightReg1, weight + dOffset + BARU_VREG_FP32_ELEMENTS);
            }
            Reg::Mul<float>(hReg0, hReg0, invRmsBrcReg, mask0);
            Reg::Mul<float>(hReg1, hReg1, invRmsBrcReg, mask1);
            Reg::Mul<float>(hReg0, hReg0, weightReg0, mask0);
            Reg::Mul<float>(hReg1, hReg1, weightReg1, mask1);
            Reg::Cast<bfloat16_t, float, BARU_CAST_FP32_TO_BF16>(hBf16Reg0, hReg0, mask0);
            Reg::Cast<bfloat16_t, float, BARU_CAST_FP32_TO_BF16>(hBf16Reg1, hReg1, mask1);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(h + rowOffset + dOffset, hBf16Reg0, mask0);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(
                h + rowOffset + dOffset + BARU_VREG_FP32_ELEMENTS, hBf16Reg1, mask1);
        }
        for (uint16_t loop = 0; loop < hasSingleVector; ++loop) {
            Reg::MaskReg mask = Reg::UpdateMask<float>(remainingD);
            const uint32_t dOffset =
                static_cast<uint32_t>(pairLoops) * BARU_VREG_PAIR_ELEMENTS + loop * BARU_VREG_FP32_ELEMENTS;
            Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(hBf16Reg0, h + rowOffset + dOffset);
            Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(hReg0, hBf16Reg0, mask);
            if constexpr (std::is_same<WeightT, bfloat16_t>::value) {
                Reg::RegTensor<bfloat16_t> weightBf16Reg0;
                Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(weightBf16Reg0, weight + dOffset);
                Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(weightReg0, weightBf16Reg0, mask);
            } else {
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(weightReg0, weight + dOffset);
            }
            Reg::Mul<float>(hReg0, hReg0, invRmsBrcReg, mask);
            Reg::Mul<float>(hReg0, hReg0, weightReg0, mask);
            Reg::Cast<bfloat16_t, float, BARU_CAST_FP32_TO_BF16>(hBf16Reg0, hReg0, mask);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(h + rowOffset + dOffset, hBf16Reg0, mask);
        }
    }
}

template <bool SINGLE_TILE, bool APPLY_OUTPUT_NORM, class OUTPUT_NORM_WEIGHT_TYPE>
class BlockAttnResUpdateFullD {
public:
    // Multi-tile mode seeds one MTE3-to-MTE2 reuse token for each ping-pong buffer. The destructor drains the final
    // tokens; single-tile mode never reuses a buffer.
    __aicore__ inline __attribute__((always_inline)) BlockAttnResUpdateFullD()
    {
        if constexpr (!SINGLE_TILE) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
    }

    __aicore__ inline __attribute__((always_inline)) ~BlockAttnResUpdateFullD()
    {
        if constexpr (!SINGLE_TILE) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
    }

    __aicore__ inline __attribute__((always_inline)) void operator()(
        GM_ADDR partial, GM_ADDR delta, GM_ADDR query, GM_ADDR numerator, GM_ADDR logitMax, GM_ADDR expSum,
        GM_ADDR weight, GM_ADDR h, __tiling_data_ptr__ BlockAttnResUpdateTilingData* tilingData)
    {
        Init(partial, delta, query, numerator, logitMax, expSum, weight, h, tilingData);
        Process();
    }

private:
    __aicore__ inline __attribute__((always_inline)) void Init(
        GM_ADDR partial, GM_ADDR delta, GM_ADDR query, GM_ADDR numerator, GM_ADDR logitMax, GM_ADDR expSum,
        GM_ADDR weight, GM_ADDR h, __tiling_data_ptr__ BlockAttnResUpdateTilingData* tilingData)
    {
        tilingData_ = tilingData;

        partialGm_ = reinterpret_cast<__gm__ float*>(partial);
        deltaGm_ = reinterpret_cast<__gm__ bfloat16_t*>(delta);
        queryGm_ = reinterpret_cast<__gm__ float*>(query);
        numeratorGm_ = reinterpret_cast<__gm__ float*>(numerator);
        logitMaxGm_ = reinterpret_cast<__gm__ float*>(logitMax);
        expSumGm_ = reinterpret_cast<__gm__ float*>(expSum);
        weightGm_ = reinterpret_cast<__gm__ OUTPUT_NORM_WEIGHT_TYPE*>(weight);
        hGm_ = reinterpret_cast<__gm__ bfloat16_t*>(h);
        InitUbLayout();
    }

    __aicore__ inline __attribute__((always_inline)) void Process()
    {
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t usedCoreNum = tilingData_->usedCoreNum;
        if (blockIdx >= usedCoreNum) {
            return;
        }

        const uint32_t tPerCore = tilingData_->tPerCore;
        const int64_t coreTStart = static_cast<int64_t>(blockIdx) * static_cast<int64_t>(tPerCore);
        const uint32_t coreTSize = (blockIdx + 1U == usedCoreNum) ? tilingData_->lastTPerCore : tPerCore;
        const uint32_t dSize = tilingData_->dSize;
        const uint32_t tileT = tilingData_->tileT;
        const uint32_t dAlignFp32 = (dSize + BARU_FP32_ALIGN_MASK) & ~BARU_FP32_ALIGN_MASK;
        const uint32_t dAlignBf16 = (dSize + BARU_BF16_ALIGN_MASK) & ~BARU_BF16_ALIGN_MASK;
        const uint32_t statsTStride = tilingData_->statsTStride;

        auto queryGmLayout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1L, static_cast<int64_t>(dSize));
        auto queryGm = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(queryGm_), queryGmLayout);
        auto queryUbLayout =
            asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1L, static_cast<int64_t>(dAlignFp32));
        auto queryUbMem = asc::te::make_mem_ptr<asc::te::location::ub, float>(0);
        auto queryUb = asc::te::make_tensor(queryUbMem, queryUbLayout);
        // Start the core-invariant Phase 1 input first, then prepare the remaining scalar descriptors while MTE2 runs.
        asc::te::copy(queryUb, queryGm);
        [[maybe_unused]] auto weightUbMem =
            asc::te::make_mem_ptr<asc::te::location::ub, OUTPUT_NORM_WEIGHT_TYPE>(queryUbBytes_);

        // Keep runtime 0/1 loop bounds on the scalar side. Deriving them inside __simd_vf__ can trigger
        // HiIPUVectorLoopUnrollPass failures at -O2/-O3.
        const uint16_t fullDLoops = static_cast<uint16_t>(dSize >> BARU_VREG_FP32_SHIFT);
        const uint16_t hasDTail = static_cast<uint16_t>((dSize & BARU_VREG_FP32_MASK) != 0U);
        const uint16_t hasOddFullD = static_cast<uint16_t>(fullDLoops & 1U);
        const uint16_t hasMixedDPair = static_cast<uint16_t>(hasOddFullD & hasDTail);
        const uint16_t hasOddFullDOnly = static_cast<uint16_t>(hasOddFullD - hasMixedDPair);
        const uint16_t hasDTailOnly = static_cast<uint16_t>(hasDTail - hasMixedDPair);
        const uint16_t phase1HasSingleRemainder = static_cast<uint16_t>(hasOddFullDOnly + hasDTailOnly);
        const uint16_t phase1HasRemainder = static_cast<uint16_t>(hasMixedDPair + phase1HasSingleRemainder);
        const uint32_t phase1RemainderD = static_cast<uint32_t>(hasOddFullDOnly) * BARU_VREG_FP32_ELEMENTS +
                                          static_cast<uint32_t>(hasDTail) * (dSize & BARU_VREG_FP32_MASK);
        const float eps = tilingData_->eps;
        const float invD = tilingData_->invD;
        auto matrixGmLayout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(static_cast<int64_t>(coreTSize),
                                                                                     static_cast<int64_t>(dSize));
        auto statsGmLayout =
            asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1L, static_cast<int64_t>(coreTSize));
        auto partialGm = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(partialGm_ + coreTStart * static_cast<int64_t>(dSize)),
            matrixGmLayout);
        auto deltaGm = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(deltaGm_ + coreTStart * static_cast<int64_t>(dSize)),
            matrixGmLayout);
        auto numeratorGm = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(numeratorGm_ + coreTStart * static_cast<int64_t>(dSize)),
            matrixGmLayout);
        auto logitMaxGm =
            asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(logitMaxGm_ + coreTStart), statsGmLayout);
        auto expSumGm =
            asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(expSumGm_ + coreTStart), statsGmLayout);
        auto hGm = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(hGm_ + coreTStart * static_cast<int64_t>(dSize)),
            matrixGmLayout);

        auto fp32UbLayout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(static_cast<int64_t>(tileT),
                                                                                   static_cast<int64_t>(dAlignFp32));
        auto bf16UbLayout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(static_cast<int64_t>(tileT),
                                                                                   static_cast<int64_t>(dAlignBf16));
        auto statsUbLayout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(
            static_cast<int64_t>(BARU_STATS_PLANE_NUM), static_cast<int64_t>(statsTStride));

        // SINGLE_TILE executes one iteration; multi-tile mode alternates the two UB buffers until coreTSize is covered.
        for (uint32_t tileTStart = 0, bufferId = 0; SINGLE_TILE || tileTStart < coreTSize;) {
            const uint32_t remainingT = coreTSize - tileTStart;
            const uint32_t tileTSize = remainingT < tileT ? remainingT : tileT;
            const int64_t tileTCoord = static_cast<int64_t>(tileTStart);
            const auto tileShape = asc::te::make_shape(static_cast<int64_t>(tileTSize), static_cast<int64_t>(dSize));

            auto partialGmTile = partialGm.slice(asc::te::make_coord(tileTCoord, 0L), tileShape);
            auto deltaGmTile = deltaGm.slice(asc::te::make_coord(tileTCoord, 0L), tileShape);

            const uint64_t partialUbOffset = invariantUbBytes_ + static_cast<uint64_t>(bufferId) * bufferUbBytes_;
            const uint64_t deltaHUbOffset = partialUbOffset + partialUbBytes_;
            auto partialUbMem = asc::te::make_mem_ptr<asc::te::location::ub, float>(partialUbOffset);
            auto deltaHUbMem = asc::te::make_mem_ptr<asc::te::location::ub, bfloat16_t>(deltaHUbOffset);
            auto partialUbStorage = asc::te::make_tensor(partialUbMem, fp32UbLayout);
            auto deltaHUbStorage = asc::te::make_tensor(deltaHUbMem, bf16UbLayout);
            auto partialUb = partialUbStorage.slice(asc::te::make_coord(0L, 0L), tileShape);
            auto deltaHUb = deltaHUbStorage.slice(asc::te::make_coord(0L, 0L), tileShape);

            // Phase 1 copy-in. Prepare Phase 2 descriptors below while these MTE2 transfers are in flight.
            if constexpr (!SINGLE_TILE) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(bufferId);
            }
            asc::te::copy(partialUb, partialGmTile);
            asc::te::copy(deltaHUb, deltaGmTile);

            const auto statsShape = asc::te::make_shape(1L, static_cast<int64_t>(tileTSize));
            auto numeratorGmTile = numeratorGm.slice(asc::te::make_coord(tileTCoord, 0L), tileShape);
            auto logitMaxGmTile = logitMaxGm.slice(asc::te::make_coord(0L, tileTCoord), statsShape);
            auto expSumGmTile = expSumGm.slice(asc::te::make_coord(0L, tileTCoord), statsShape);
            auto hGmTile = hGm.slice(asc::te::make_coord(tileTCoord, 0L), tileShape);

            const uint64_t numeratorUbOffset = deltaHUbOffset + deltaHUbBytes_;
            const uint64_t statsUbOffset = numeratorUbOffset + partialUbBytes_;
            auto numeratorUbMem = asc::te::make_mem_ptr<asc::te::location::ub, float>(numeratorUbOffset);
            auto statsUbMem = asc::te::make_mem_ptr<asc::te::location::ub, float>(statsUbOffset);
            auto numeratorUbStorage = asc::te::make_tensor(numeratorUbMem, fp32UbLayout);
            auto statsUbStorage = asc::te::make_tensor(statsUbMem, statsUbLayout);
            auto numeratorUb = numeratorUbStorage.slice(asc::te::make_coord(0L, 0L), tileShape);
            auto logitMaxUb = statsUbStorage.slice(
                asc::te::make_coord(static_cast<int64_t>(BARU_LOGIT_MAX_PLANE_INDEX), 0L), statsShape);
            auto expSumUb = statsUbStorage.slice(
                asc::te::make_coord(static_cast<int64_t>(BARU_EXP_SUM_PLANE_INDEX), 0L), statsShape);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(bufferId);

            // Phase 1 compute.
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(bufferId);
            ComputePhase1(partialUbMem.get(), deltaHUbMem.get(), queryUbMem.get(), statsUbMem.get(), tileTSize, dSize,
                          dAlignFp32, dAlignBf16, statsTStride, eps, invD, hasDTail, hasMixedDPair,
                          phase1HasSingleRemainder, phase1HasRemainder, phase1RemainderD);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(bufferId);

            // Queue the partial copy-out as soon as its Phase 1 dependency is established.
            // MTE3 waits for Phase 1 while Scalar continues issuing the independent Phase 2 MTE2 copy-in.
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(bufferId);
            asc::te::copy(partialGmTile, partialUb);

            // Phase 2 copy-in can overlap Phase 1 compute because it writes disjoint UB regions.
            const uint32_t phase2EventId = bufferId + BARU_BUFFER_NUM;
            // Gamma is only consumed by Phase 2. Queue it after Phase 1 has started so its
            // transfer does not delay Phase 1's MTE2_V dependency; the Phase 2 event covers it.
            if constexpr (APPLY_OUTPUT_NORM) {
                if (SINGLE_TILE || tileTStart == 0) {
                    auto weightGm =
                        asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(weightGm_), queryGmLayout);
                    const uint32_t weightDAlign =
                        std::is_same<OUTPUT_NORM_WEIGHT_TYPE, bfloat16_t>::value ? dAlignBf16 : dAlignFp32;
                    auto weightUbLayout =
                        asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1L, static_cast<int64_t>(weightDAlign));
                    auto weightUb = asc::te::make_tensor(weightUbMem, weightUbLayout);
                    asc::te::copy(weightUb, weightGm);
                }
            }
            asc::te::copy(numeratorUb, numeratorGmTile);
            asc::te::copy(logitMaxUb, logitMaxGmTile);
            asc::te::copy(expSumUb, expSumGmTile);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(phase2EventId);

            // The partial copy-out and Phase 2 only read partial, so MTE3 and V may overlap.
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(phase2EventId);
            ComputePhase2(partialUbMem.get(), deltaHUbMem.get(), numeratorUbMem.get(), statsUbMem.get(),
                          weightUbMem.get(), tileTSize, dSize, dAlignFp32, dAlignBf16, statsTStride, invD, fullDLoops,
                          hasDTail, hasMixedDPair, hasOddFullDOnly, hasDTailOnly);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(phase2EventId);

            // Phase 2 copy-out closes this buffer's reuse dependency chain.
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(phase2EventId);
            asc::te::copy(hGmTile, deltaHUb);
            if constexpr (!SINGLE_TILE) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(bufferId);
            }
            if constexpr (SINGLE_TILE) {
                break;
            }
            tileTStart += tileTSize;
            bufferId ^= 1U;
        }
    }

    __aicore__ inline __attribute__((always_inline)) void ComputePhase1(
        __ubuf__ float* partial, __ubuf__ bfloat16_t* deltaH, __ubuf__ float* query, __ubuf__ float* stats,
        uint32_t rows, uint32_t dSize, uint32_t dAlignFp32, uint32_t dAlignBf16, uint32_t statsTStride, float eps,
        float invD, uint16_t hasDTail, uint16_t hasMixedDPair, uint16_t phase1HasSingleRemainder,
        uint16_t phase1HasRemainder, uint32_t phase1RemainderD)
    {
        // TwoVLVF is specialized for D spanning at most two FP32 vector-register widths.
        if (dSize <= BARU_VREG_FP32_ELEMENTS) {
            asc_vf_call<BlockAttnResUpdatePhase1OneVLVF>(partial, deltaH, query, stats, static_cast<uint16_t>(rows),
                                                         dSize, dAlignFp32, dAlignBf16, statsTStride, eps, invD,
                                                         hasDTail);
        } else if (dSize <= BARU_VREG_PAIR_ELEMENTS) {
            asc_vf_call<BlockAttnResUpdatePhase1TwoVLVF>(partial, deltaH, query, stats, static_cast<uint16_t>(rows),
                                                         dSize, dAlignFp32, dAlignBf16, statsTStride, eps, invD,
                                                         hasDTail);
        } else {
            asc_vf_call<BlockAttnResUpdatePhase1VF>(partial, deltaH, query, stats, static_cast<uint16_t>(rows), dSize,
                                                    dAlignFp32, dAlignBf16, statsTStride, eps, invD, hasMixedDPair,
                                                    phase1HasSingleRemainder, phase1HasRemainder, phase1RemainderD);
        }
    }

    __aicore__ inline __attribute__((always_inline)) void ComputePhase2(
        __ubuf__ float* partial, __ubuf__ bfloat16_t* deltaH, __ubuf__ float* numerator, __ubuf__ float* stats,
        __ubuf__ OUTPUT_NORM_WEIGHT_TYPE* weight, uint32_t rows, uint32_t dSize, uint32_t dAlignFp32,
        uint32_t dAlignBf16, uint32_t statsTStride, float invD, uint16_t fullDLoops, uint16_t hasDTail,
        uint16_t hasMixedDPair, uint16_t hasOddFullDOnly, uint16_t hasDTailOnly)
    {
        // TwoVLVF is specialized for D spanning at most two FP32 vector-register widths.
        if (dSize <= BARU_VREG_FP32_ELEMENTS) {
            asc_vf_call<BlockAttnResUpdatePhase2OneVLVF>(partial, deltaH, numerator, stats, static_cast<uint16_t>(rows),
                                                         dSize, dAlignFp32, dAlignBf16, statsTStride);
        } else if (dSize <= BARU_VREG_PAIR_ELEMENTS) {
            asc_vf_call<BlockAttnResUpdatePhase2TwoVLVF>(partial, deltaH, numerator, stats, static_cast<uint16_t>(rows),
                                                         dSize, dAlignFp32, dAlignBf16, statsTStride);
        } else {
            asc_vf_call<BlockAttnResUpdatePhase2VF>(partial, deltaH, numerator, stats, static_cast<uint16_t>(rows),
                                                    dSize, dAlignFp32, dAlignBf16, statsTStride, fullDLoops, hasDTail,
                                                    hasMixedDPair, hasOddFullDOnly, hasDTailOnly);
        }
        if constexpr (APPLY_OUTPUT_NORM) {
            asc_vf_call<BlockAttnResUpdateOutputNormVF<OUTPUT_NORM_WEIGHT_TYPE>>(
                deltaH, weight, static_cast<uint16_t>(rows), dSize, dAlignBf16,
                static_cast<uint16_t>(fullDLoops + hasDTail), static_cast<uint16_t>((fullDLoops + hasDTail) & 1U),
                tilingData_->outputNormEps, invD);
        }
    }

    __aicore__ inline __attribute__((always_inline)) void InitUbLayout()
    {
        const uint32_t dSize = tilingData_->dSize;
        const uint32_t dAlignFp32 = (dSize + BARU_FP32_ALIGN_MASK) & ~BARU_FP32_ALIGN_MASK;
        const uint32_t dAlignBf16 = (dSize + BARU_BF16_ALIGN_MASK) & ~BARU_BF16_ALIGN_MASK;
        queryUbBytes_ = static_cast<uint64_t>(dAlignFp32) * BARU_FP32_BYTES;
        invariantUbBytes_ = queryUbBytes_;
        if constexpr (APPLY_OUTPUT_NORM) {
            if constexpr (std::is_same<OUTPUT_NORM_WEIGHT_TYPE, bfloat16_t>::value) {
                invariantUbBytes_ += static_cast<uint64_t>(dAlignBf16) * BARU_BF16_BYTES;
            } else {
                invariantUbBytes_ += queryUbBytes_;
            }
        }
        partialUbBytes_ = static_cast<uint64_t>(tilingData_->tileT) * queryUbBytes_;
        deltaHUbBytes_ = static_cast<uint64_t>(tilingData_->tileT) * dAlignBf16 * BARU_BF16_BYTES;
        const uint64_t statsBytes =
            static_cast<uint64_t>(BARU_STATS_PLANE_NUM) * tilingData_->statsTStride * BARU_FP32_BYTES;
        const uint64_t numeratorUbBytes = partialUbBytes_;
        // Keep buffers contiguous; CalcUbBytes reserves tail padding only after the final stats allocation.
        bufferUbBytes_ = partialUbBytes_ + numeratorUbBytes + deltaHUbBytes_ + statsBytes;
    }

    uint64_t queryUbBytes_ = 0;
    uint64_t invariantUbBytes_ = 0;
    uint64_t partialUbBytes_ = 0;
    uint64_t deltaHUbBytes_ = 0;
    uint64_t bufferUbBytes_ = 0;

    __tiling_data_ptr__ BlockAttnResUpdateTilingData* tilingData_ = nullptr;
    __gm__ float* partialGm_ = nullptr;
    __gm__ bfloat16_t* deltaGm_ = nullptr;
    __gm__ float* queryGm_ = nullptr;
    __gm__ float* numeratorGm_ = nullptr;
    __gm__ float* logitMaxGm_ = nullptr;
    __gm__ float* expSumGm_ = nullptr;
    __gm__ OUTPUT_NORM_WEIGHT_TYPE* weightGm_ = nullptr;
    __gm__ bfloat16_t* hGm_ = nullptr;
};

} // namespace BlockAttnResUpdateOps

#endif // BLOCK_ATTN_RES_UPDATE_FULL_D_H
