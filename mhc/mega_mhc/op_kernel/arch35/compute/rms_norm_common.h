/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_TRANSFORMER_MHC_MEGA_MHC_RMS_NORM_COMMON_H
#define OPS_TRANSFORMER_MHC_MEGA_MHC_RMS_NORM_COMMON_H

#include "kernel_operator.h"
#include <cfloat>

namespace MegaMhc::Compute {
constexpr uint32_t kRegPairLanes = 128U;
constexpr uint32_t kRegLaneRounding = 63U;
constexpr uint16_t kReductionInitialStride = 32U;
constexpr float kNewtonHalf = 0.5f;
constexpr float kNewtonThreeHalves = 1.5f;
constexpr uint32_t kHeadCount = 4U;
constexpr uint32_t kHeadMatrixElements = 16U;
constexpr uint32_t kProjectionCoefficientElements = 24U;
constexpr uint32_t kPackedNzRowStride = 8U;
constexpr uint32_t kGateGroupElements = 8U;
constexpr uint32_t kPairedRows = 2U;

using namespace AscendC;

constexpr uint32_t kRegFloatLanes = 64;
constexpr uint32_t kRegCastVectorLanes = 64;

/** Build the low count lanes without changing the caller's loop increment (UpdateMask decrements its argument). */
__simd_callee__ inline Reg::MaskReg MaskForCount(uint32_t count)
{
    return Reg::UpdateMask<float>(count);
}

constexpr Reg::CastTrait kCastInputToFloat = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
                                              RoundMode::UNKNOWN};
constexpr Reg::CastTrait kCastFloatToInput = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING,
                                              RoundMode::CAST_RINT};

/** Compute sumOut=sum_i FP32(x[i])^2.
 * Full 128-element pairs use two independent FP32 chains; the tail uses the first chain. */
template <typename T>
__aicore__ inline void AccumulateRowSquares(__ubuf__ T* inputRow, uint32_t rowLength, __ubuf__ float* sumOut)
{
    const uint16_t pairedRepeats = static_cast<uint16_t>(rowLength / kRegPairLanes);
    const uint32_t pairedElements = static_cast<uint32_t>(pairedRepeats) * kRegPairLanes;
    const uint32_t tailElements = rowLength - pairedElements;
    const uint16_t tailRepeats = static_cast<uint16_t>((tailElements + kRegLaneRounding) / kRegFloatLanes);
    __VEC_SCOPE__
    {
        Reg::MaskReg mask;
        Reg::MaskReg fullMask = Compute::MaskForCount(kRegFloatLanes);
        Reg::RegTensor<T> input0, input1;
        Reg::RegTensor<float> value0, value1, partial, sum, sumOdd;
        Reg::Duplicate(sum, 0.0f, fullMask);
        Reg::Duplicate(sumOdd, 0.0f, fullMask);
        for (uint16_t repeat = 0U; repeat < pairedRepeats; ++repeat) {
            const uint32_t offset = static_cast<uint32_t>(repeat) * kRegPairLanes;
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input0, inputRow + offset);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input1, inputRow + offset + kRegFloatLanes);
            Reg::Cast<float, T, kCastInputToFloat>(value0, input0, fullMask);
            Reg::Cast<float, T, kCastInputToFloat>(value1, input1, fullMask);
            Reg::MulAddDst(sum, value0, value0, fullMask);
            Reg::MulAddDst(sumOdd, value1, value1, fullMask);
        }
        uint32_t remaining = tailElements;
        for (uint16_t repeat = 0U; repeat < tailRepeats; ++repeat) {
            const uint32_t offset = pairedElements + static_cast<uint32_t>(repeat) * kRegFloatLanes;
            mask = Reg::UpdateMask<float>(remaining);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input0, inputRow + offset);
            Reg::Cast<float, T, kCastInputToFloat>(value0, input0, mask);
            Reg::MulAddDst(sum, value0, value0, fullMask);
        }
        Reg::Add(sum, sum, sumOdd, fullMask);

        Reg::Reduce<Reg::ReduceType::SUM>(partial, sum, fullMask);
        Reg::Store<float>(sumOut, partial, 1U);
    }
}

/** Merge two compensated values: (left-leftCarry) += (right-rightCarry).
 * TwoSum retains the rounding residual instead of discarding it at each horizontal/tree or D-tile boundary.
 */
__simd_callee__ inline void AddCompensatedRegisters(Reg::RegTensor<float>& left, Reg::RegTensor<float>& leftCarry,
                                                    Reg::RegTensor<float>& right, Reg::RegTensor<float>& rightCarry,
                                                    Reg::MaskReg& mask)
{
    Reg::RegTensor<float> next, virtualRight, virtualLeft, leftError, rightError, limit, zero;
    Reg::MaskReg finite;
    Reg::Duplicate(limit, FLT_MAX, mask);
    Reg::Duplicate(zero, 0.0f, mask);
    Reg::Add(next, left, right, mask);
    Reg::Sub(virtualRight, next, left, mask);
    Reg::Sub(virtualLeft, next, virtualRight, mask);
    Reg::Sub(leftError, left, virtualLeft, mask);
    Reg::Sub(rightError, right, virtualRight, mask);
    Reg::Add(leftError, leftError, rightError, mask);
    Reg::Add(leftCarry, leftCarry, rightCarry, mask);
    Reg::Sub(leftCarry, leftCarry, leftError, mask);
    // An overflowing positive sum retains +Inf without an Inf-Inf compensation.
    Reg::Compare<float, CMPMODE::LE>(finite, next, limit, mask);
    Reg::Select(leftCarry, leftCarry, zero, finite);
    Reg::Adds(left, next, 0.0f, mask);
}

/** Accumulate (sumOut-carryOut)+=sum_i FP32(input[i])^2 across D tiles.
 * Kahan lane accumulation and TwoSum horizontal/tile merges preserve rounding residuals through the full row.
 * The caller zeroes both scalars once per row. */
template <typename T>
__aicore__ inline void AccumulateCompensatedSquares(__ubuf__ T* inputRow, uint32_t rowLength, __ubuf__ float* sumOut,
                                                    __ubuf__ float* carryOut)
{
    const uint16_t repeats = static_cast<uint16_t>((rowLength + kRegLaneRounding) / kRegFloatLanes);
    __VEC_SCOPE__
    {
        Reg::MaskReg mask;
        Reg::MaskReg full = MaskForCount(kRegFloatLanes);
        Reg::MaskReg scalar = MaskForCount(1U);
        Reg::RegTensor<T> input;
        Reg::RegTensor<uint32_t> indices, shifted;
        Reg::RegTensor<float> value, square, sum, carry, adjusted, next, difference;
        Reg::RegTensor<float> right, rightCarry, previous, previousCarry;
        Reg::RegTensor<float> limit, zero, ordinarySum;
        Reg::MaskReg finite;
        if constexpr (IsSameType<T, bfloat16_t>::value) {
            Reg::Duplicate(limit, FLT_MAX, full);
            Reg::Duplicate(zero, 0.0f, full);
            Reg::Duplicate(ordinarySum, 0.0f, full);
        }
        Reg::Duplicate(sum, 0.0f, full);
        Reg::Duplicate(carry, 0.0f, full);
        Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(indices), 0);
        Reg::Duplicate(shifted, 0U, full);
        uint32_t remaining = rowLength;
        for (uint16_t repeat = 0U; repeat < repeats; ++repeat) {
            mask = Reg::UpdateMask<float>(remaining);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input, inputRow + repeat * kRegFloatLanes);
            Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
            Reg::Mul(square, value, value, full);
            if constexpr (IsSameType<T, bfloat16_t>::value) {
                Reg::Add(ordinarySum, ordinarySum, square, full);
            }
            Reg::Sub(adjusted, square, carry, full);
            Reg::Add(next, sum, adjusted, full);
            Reg::Sub(difference, next, sum, full);
            Reg::Sub(carry, difference, adjusted, full);
            Reg::Adds(sum, next, 0.0f, full);
        }
        if constexpr (IsSameType<T, bfloat16_t>::value) {
            // The independent sum preserves overflow/NaN without extending the Kahan dependency chain.
            Reg::Compare<float, CMPMODE::LE>(finite, sum, limit, full);
            Reg::Select(sum, sum, ordinarySum, finite);
            Reg::Select(carry, carry, zero, finite);
        }
        for (uint16_t stride = kReductionInitialStride; stride != 0U; stride >>= 1U) {
            mask = MaskForCount(stride);
            Reg::Adds(shifted, indices, static_cast<uint32_t>(stride), mask);
            Reg::Gather(right, sum, shifted);
            Reg::Gather(rightCarry, carry, shifted);
            AddCompensatedRegisters(sum, carry, right, rightCarry, mask);
        }
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(previous, sumOut);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(previousCarry, carryOut);
        AddCompensatedRegisters(previous, previousCarry, sum, carry, scalar);
        Reg::Store<float>(sumOut, previous, 1U);
        Reg::Store<float>(carryOut, previousCarry, 1U);
    }
}

/** Accumulate FP32 scalars: accumulator+=partial. The caller determines the reduction order. */
__aicore__ inline void AddFp32Scalar(__ubuf__ float* accumulator, __ubuf__ float* partial)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> left;
        Reg::RegTensor<float> right;
        Reg::MaskReg mask = Compute::MaskForCount(1U);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(left, accumulator);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(right, partial);
        Reg::Add(left, left, right, mask);
        Reg::Store<float>(accumulator, left, 1U);
    }
}

/** Compute invRms=rsqrt(sumSquares/rowLength+epsilon), refined using fused reciprocal and square residuals.
 * COMPENSATED preserves sumCarry through division, epsilon addition and refinement; result is in lane zero. */
template <bool COMPENSATED = false>
__simd_callee__ inline void InverseRmsRegister(__ubuf__ float* sumSquares, float meanScale, float epsilon,
                                               Reg::RegTensor<float>& rms, __ubuf__ float* sumCarry = nullptr,
                                               float rowLength = 1.0f)
{
    Reg::RegTensor<float> sum;
    Reg::RegTensor<float> scale;
    Reg::RegTensor<float> eps;
    Reg::RegTensor<float> one;
    Reg::RegTensor<float> mean;
    Reg::RegTensor<float> negative;
    Reg::RegTensor<float> product;
    Reg::RegTensor<float> error;
    Reg::RegTensor<float> halfInverse;
    Reg::RegTensor<float> meanCarry;
    Reg::RegTensor<float> initialInverse, limit;
    Reg::MaskReg finite;
    Reg::MaskReg mask = Compute::MaskForCount(1);
    Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(sum, sumSquares);
    Reg::Duplicate(eps, epsilon, mask);
    Reg::Duplicate(one, 1.0f, mask);
    if constexpr (COMPENSATED) {
        Reg::RegTensor<float> carry;
        Reg::RegTensor<float> zeroCarry;
        // Keep the low part of (sum-carry)/length+epsilon through the inverse-square-root refinement.
        Reg::Duplicate(scale, rowLength, mask);
        Reg::Div(rms, sum, scale, mask);
        Reg::Muls(negative, scale, -1.0f, mask);
        Reg::Adds(meanCarry, sum, 0.0f, mask);
        Reg::MulAddDst(meanCarry, rms, negative, mask);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(carry, sumCarry);
        Reg::Sub(meanCarry, meanCarry, carry, mask);
        Reg::Div(meanCarry, meanCarry, scale, mask);
        Reg::Muls(meanCarry, meanCarry, -1.0f, mask);
        Reg::Duplicate(zeroCarry, 0.0f, mask);
        AddCompensatedRegisters(rms, meanCarry, eps, zeroCarry, mask);
    } else {
        Reg::Duplicate(scale, meanScale, mask);
        Reg::Mul(rms, sum, scale, mask);
        Reg::Add(rms, rms, eps, mask);
    }
    Reg::Adds(mean, rms, 0.0f, mask);
    Reg::RegTensor<float> reciprocal;
    Reg::RegTensor<float> correction;
    Reg::Duplicate(halfInverse, kNewtonHalf, mask);
    Reg::Duplicate(correction, kNewtonThreeHalves, mask);
    Reg::Duplicate(error, 1.0f, mask);
    Reg::Div(reciprocal, one, mean, mask);
    Reg::Sqrt(rms, reciprocal, mask);
    Reg::Adds(initialInverse, rms, 0.0f, mask);
    // First refine y=sqrt(1/mean), then recover the residual using fused products.
    Reg::Muls(product, mean, -kNewtonHalf, mask);
    Reg::Mul(product, product, rms, mask);
    Reg::MulAddDst(correction, product, rms, mask);
    Reg::Mul(rms, rms, correction, mask);
    Reg::Muls(negative, mean, -1.0f, mask);
    Reg::MulAddDst(error, negative, reciprocal, mask);
    Reg::Muls(product, rms, -1.0f, mask);
    Reg::MulAddDst(reciprocal, product, rms, mask);
    Reg::MulAddDst(error, mean, reciprocal, mask);
    if constexpr (COMPENSATED) {
        // The compensated mean represents mean-meanCarry.
        Reg::Mul(product, meanCarry, rms, mask);
        Reg::MulAddDst(error, product, rms, mask);
    }
    Reg::Mul(error, error, rms, mask);
    Reg::MulAddDst(rms, error, halfInverse, mask);
    // Preserve rsqrt(+Inf)=0 and NaN propagation instead of applying an invalid Newton correction.
    Reg::Duplicate(limit, FLT_MAX, mask);
    Reg::Compare<float, CMPMODE::LE>(finite, mean, limit, mask);
    Reg::Select(rms, rms, initialInverse, finite);
}

/** Finish one RMS statistic in UB: invRms=rsqrt(sumSquares/rowLength+epsilon), with Newton refinement. */
template <bool COMPENSATED = false>
__aicore__ inline void FinalizeInverseRms(__ubuf__ float* sumSquares, uint32_t rowLength, float epsilon,
                                          __ubuf__ float* invRmsOut, __ubuf__ float* sumCarry = nullptr)
{
    const float meanScale = 1.0f / static_cast<float>(rowLength);
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> inverse;
        InverseRmsRegister<COMPENSATED>(sumSquares, meanScale, epsilon, inverse, sumCarry,
                                        static_cast<float>(rowLength));
        Reg::Store<float>(invRmsOut, inverse, 1U);
    }
}

namespace RmsNormDetail {

/** Apply y[i]=Cast<T>(FP32(x[i])*invRms*gamma[i]).
 * FROM_SQUARES computes invRms from sumSquares; otherwise it consumes the precomputed statistic. */
template <typename T, bool FROM_SQUARES>
__aicore__ inline void ApplyRow(__ubuf__ T* inputRow, __ubuf__ float* gammaRow, uint32_t rowLength,
                                __ubuf__ float* invRms, __ubuf__ T* outputRow, float epsilon, uint32_t rowCount = 1U)
{
    const float meanScale = 1.0f / static_cast<float>(rowLength);
    const uint16_t fullRepeats = static_cast<uint16_t>(rowLength / kRegCastVectorLanes);
    const uint32_t tailElements = rowLength % kRegCastVectorLanes;
    const uint16_t tailRepeats = static_cast<uint16_t>((tailElements + kRegLaneRounding) / kRegFloatLanes);
    __VEC_SCOPE__
    {
        for (uint16_t row = 0U; row < static_cast<uint16_t>(rowCount); ++row) {
            Reg::MaskReg mask;
            Reg::RegTensor<T> input;
            Reg::RegTensor<float> value;
            Reg::RegTensor<float> gamma;
            Reg::RegTensor<float> inverse;
            Reg::RegTensor<float> scaled;
            Reg::RegTensor<T> output;
            if constexpr (FROM_SQUARES) {
                Reg::RegTensor<float> scalarInverse;
                InverseRmsRegister(invRms + row, meanScale, epsilon, scalarInverse);
                Reg::MaskReg fullMask = MaskForCount(kRegFloatLanes);
                Reg::Duplicate(inverse, scalarInverse, fullMask);
            } else {
                Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(inverse, invRms + row);
            }
            mask = Compute::MaskForCount(kRegFloatLanes);
            for (uint16_t repeat = 0; repeat < fullRepeats; ++repeat) {
                const uint32_t offset = repeat * Compute::kRegCastVectorLanes;
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input, inputRow + row * rowLength + offset);
                Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
                Reg::LoadAlign<float>(gamma, gammaRow + offset);
                Reg::Mul(scaled, value, inverse, mask);
                Reg::Mul(scaled, scaled, gamma, mask);
                Reg::Cast<T, float, kCastFloatToInput>(output, scaled, mask);
                Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(outputRow + row * rowLength + offset, output, mask);
            }
            mask = Compute::MaskForCount(tailElements);
            for (uint16_t repeat = 0U; repeat < tailRepeats; ++repeat) {
                const uint32_t offset = fullRepeats * Compute::kRegCastVectorLanes;
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input, inputRow + row * rowLength + offset);
                Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
                Reg::LoadAlign<float>(gamma, gammaRow + offset);
                Reg::Mul(scaled, value, inverse, mask);
                Reg::Mul(scaled, scaled, gamma, mask);
                Reg::Cast<T, float, kCastFloatToInput>(output, scaled, mask);
                Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(outputRow + row * rowLength + offset, output, mask);
            }
        }
    }
}

} // namespace RmsNormDetail

/** Normalize independent contiguous rows in one VF, preserving each row's inverse and rounding. */
template <typename T>
__aicore__ inline void ComputeRmsNormRowsFromSquares(__ubuf__ T* input, __ubuf__ float* weight, uint32_t length,
                                                     uint32_t rows, float epsilon, __ubuf__ float* sumSquares)
{
    RmsNormDetail::ApplyRow<T, true>(input, weight, length, sumSquares, input, epsilon, rows);
}

/** Normalize using the full-row square sum: y[i]=Cast<T>(FP32(x[i])*rsqrt(sum/length+eps)*weight[i]).
 * The caller supplies squares of the rounded stage input. RMS0/Mid/RMS1 share the same arithmetic. */
template <typename T>
__aicore__ inline void ComputeRmsNormFromSquares(__ubuf__ T* input, __ubuf__ float* weight, uint32_t length,
                                                 float epsilon, __ubuf__ float* sumSquares, __ubuf__ T* output)
{
    RmsNormDetail::ApplyRow<T, true>(input, weight, length, sumSquares, output, epsilon);
}

/** Apply a full-row inverse RMS to the current tile: y[i]=Cast<T>(FP32(x[i])*inverse*weight[i]).
 * Tile length controls this apply pass only; inverse must use the complete normalization axis. */
template <typename T>
__aicore__ inline void ApplyRmsNorm(__ubuf__ T* input, __ubuf__ float* weight, uint32_t length, __ubuf__ float* inverse,
                                    __ubuf__ T* output)
{
    RmsNormDetail::ApplyRow<T, false>(input, weight, length, inverse, output, 0.0f);
}

} // namespace MegaMhc::Compute

#endif // OPS_TRANSFORMER_MHC_MEGA_MHC_RMS_NORM_COMMON_H
