/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_TRANSFORMER_MHC_MEGA_MHC_PRE_H
#define OPS_TRANSFORMER_MHC_MEGA_MHC_PRE_H

#include "rms_norm_common.h"

namespace MegaMhc::Compute {
constexpr uint32_t kVectorSegmentsPerPair = 2U;

/** Cast and pack A while accumulating Pre statistics in one VF.
 * NZ[(c/8)*R*8+r*8+c%8]=FP32(x2[r,c]); ND[r*S+c]=FP32(x2[r,c]); squares[r]+=sum_c FP32(x2[r,c])^2.
 * Host supplies strides R/S. Two 64-lane halves retain the generic reduction order. */
template <typename T, bool PACKED_NZ>
__aicore__ inline void PrepareProjectionStageTile(__ubuf__ T* input, uint32_t rows, uint32_t kCount,
                                                  uint32_t stageStride, __ubuf__ float* stage, __ubuf__ float* squares)
{
    const uint16_t passes = static_cast<uint16_t>(kCount / kRegPairLanes);
    const uint16_t groups = static_cast<uint16_t>((rows + kRegLaneRounding) / kRegFloatLanes);
    __VEC_SCOPE__
    {
        Reg::MaskReg full = MaskForCount(kRegFloatLanes);
        Reg::MaskReg rowMask, validRows;
        Reg::RegTensor<T> rawLo, rawHi;
        Reg::RegTensor<float> lo, hi;
        Reg::RegTensor<float> indices, one, currentRow, packedSums, acc, square, sum, broadcast, previous;
        Reg::Arange(indices, 0.0f);
        Reg::Duplicate(one, 1.0f);
        for (uint16_t group = 0U; group < groups; ++group) {
            const uint32_t firstRow = static_cast<uint32_t>(group) * kRegFloatLanes;
            const uint16_t rowCount = static_cast<uint16_t>(AscendC::Std::min(rows - firstRow, kRegFloatLanes));
            Reg::Duplicate(currentRow, 0.0f);
            Reg::Duplicate(packedSums, 0.0f);
            for (uint16_t row = 0U; row < rowCount; ++row) {
                auto source = input + (firstRow + row) * kCount;
                auto destination = stage + (firstRow + row) * (PACKED_NZ ? kPackedNzRowStride : stageStride);
                Reg::Duplicate(acc, 0.0f);
                for (uint16_t pass = 0U; pass < passes; ++pass) {
                    const uint32_t column = static_cast<uint32_t>(pass) * kRegPairLanes;
                    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(rawLo, source + column);
                    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(rawHi, source + column + kRegFloatLanes);
                    Reg::Cast<float, T, kCastInputToFloat>(lo, rawLo, full);
                    Reg::Cast<float, T, kCastInputToFloat>(hi, rawHi, full);
                    Reg::MulAddDst(acc, lo, lo, full);
                    Reg::MulAddDst(acc, hi, hi, full);
                    if constexpr (PACKED_NZ) {
                        Reg::StoreAlign<float, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                            destination, lo, stageStride, kPackedNzRowStride * stageStride, full);
                        Reg::StoreAlign<float, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                            destination, hi, stageStride, kPackedNzRowStride * stageStride, full);
                    } else {
                        Reg::StoreAlign(destination + column, lo, full);
                        Reg::StoreAlign(destination + column + kRegFloatLanes, hi, full);
                    }
                }
                Reg::Reduce<Reg::ReduceType::SUM>(sum, acc, full);
                Reg::Duplicate(broadcast, sum, full);
                Reg::Compare<float, AscendC::CMPMODE::EQ>(rowMask, indices, currentRow, full);
                Reg::Select(packedSums, broadcast, packedSums, rowMask);
                Reg::Add(currentRow, currentRow, one, full);
            }
            uint32_t count = rowCount;
            validRows = Reg::UpdateMask<float>(count);
            Reg::LoadAlign(previous, squares + firstRow);
            Reg::Add(packedSums, packedSums, previous, validRows);
            Reg::StoreAlign(squares + firstRow, packedSums, validRows);
        }
    }
}

/** Prepare stage[r,c]=FP32(x2[r,c]) and squares[r]+=sum_c stage[r,c]^2.
 * ND uses the Host K stride; NZ offset=(c/8)*R*8+r*8+c%8. Paired rows retain independent ordered reductions.
 */
template <typename T, bool PACKED_NZ = false>
__aicore__ inline void PrepareProjectionTile(__ubuf__ T* input, uint32_t rows, uint32_t kCount, uint32_t stageStride,
                                             __ubuf__ float* stage, __ubuf__ float* squares)
{
    if (kCount % kRegPairLanes == 0U) {
        PrepareProjectionStageTile<T, PACKED_NZ>(input, rows, kCount, stageStride, stage, squares);
        return;
    }

    const uint16_t rowCount = static_cast<uint16_t>(rows);
    const uint16_t repeats = static_cast<uint16_t>((kCount + kRegLaneRounding) / kRegFloatLanes);
    const uint16_t pairedRows = rowCount / kPairedRows;
    const uint16_t tailRow = pairedRows * kPairedRows;
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> sum;
        Reg::RegTensor<float> value;

        Reg::RegTensor<float> partial;
        Reg::RegTensor<float> previous;
        Reg::RegTensor<T> raw;
        Reg::MaskReg mask;
        Reg::MaskReg fullMask = MaskForCount(kRegFloatLanes);
        Reg::MaskReg scalarMask = MaskForCount(1U);
        Reg::RegTensor<float> sumNext;
        Reg::RegTensor<float> valueNext;

        Reg::RegTensor<float> partialNext;
        Reg::RegTensor<float> previousNext;
        Reg::RegTensor<T> rawNext;
        // Interleave two rows while preserving each row's ordered FP32 reduction.
        for (uint16_t pair = 0; pair < pairedRows; ++pair) {
            const uint32_t row = pair * kPairedRows;
            Reg::Duplicate(sum, 0.0f, fullMask);

            auto inputRow = input + row * kCount;
            auto stageRow = stage + row * (PACKED_NZ ? kPackedNzRowStride : stageStride);
            uint32_t remaining = kCount;
            Reg::Duplicate(sumNext, 0.0f, fullMask);

            auto inputRowNext = input + (row + 1U) * kCount;
            auto stageRowNext = stage + (row + 1U) * (PACKED_NZ ? kPackedNzRowStride : stageStride);
            for (uint16_t repeat = 0; repeat < repeats; ++repeat) {
                const uint32_t column = repeat * kRegFloatLanes;
                mask = Reg::UpdateMask<float>(remaining);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(raw, inputRow + column);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(rawNext, inputRowNext + column);
                Reg::Cast<float, T, kCastInputToFloat>(value, raw, mask);
                if constexpr (PACKED_NZ) {
                    Reg::StoreAlign<float, Reg::DataCopyMode::DATA_BLOCK_COPY>(stageRow + column * stageStride, value,
                                                                               stageStride, mask);
                } else {
                    Reg::StoreAlign<float>(stageRow + column, value, mask);
                }
                Reg::MulAddDst(sum, value, value, fullMask);
                Reg::Cast<float, T, kCastInputToFloat>(valueNext, rawNext, mask);
                if constexpr (PACKED_NZ) {
                    Reg::StoreAlign<float, Reg::DataCopyMode::DATA_BLOCK_COPY>(stageRowNext + column * stageStride,
                                                                               valueNext, stageStride, mask);
                } else {
                    Reg::StoreAlign<float>(stageRowNext + column, valueNext, mask);
                }
                Reg::MulAddDst(sumNext, valueNext, valueNext, fullMask);
            }
            Reg::Reduce<Reg::ReduceType::SUM>(partial, sum, fullMask);
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(previous, squares + row);
            Reg::Add(partial, previous, partial, scalarMask);
            Reg::Store<float>(squares + row, partial, 1U);
            Reg::Reduce<Reg::ReduceType::SUM>(partialNext, sumNext, fullMask);
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(previousNext, squares + row + 1U);
            Reg::Add(partialNext, previousNext, partialNext, scalarMask);
            Reg::Store<float>(squares + row + 1U, partialNext, 1U);
        }
        for (uint16_t row = tailRow; row < rowCount; ++row) {
            Reg::Duplicate(sum, 0.0f, fullMask);

            auto inputRow = input + row * kCount;
            auto stageRow = stage + row * (PACKED_NZ ? kPackedNzRowStride : stageStride);
            uint32_t remaining = kCount;
            for (uint16_t repeat = 0; repeat < repeats; ++repeat) {
                const uint32_t column = repeat * kRegFloatLanes;
                mask = Reg::UpdateMask<float>(remaining);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(raw, inputRow + column);
                Reg::Cast<float, T, kCastInputToFloat>(value, raw, mask);
                if constexpr (PACKED_NZ) {
                    Reg::StoreAlign<float, Reg::DataCopyMode::DATA_BLOCK_COPY>(stageRow + column * stageStride, value,
                                                                               stageStride, mask);
                } else {
                    Reg::StoreAlign<float>(stageRow + column, value, mask);
                }
                Reg::MulAddDst(sum, value, value, fullMask);
            }
            Reg::Reduce<Reg::ReduceType::SUM>(partial, sum, fullMask);
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(previous, squares + row);
            Reg::Add(partial, previous, partial, scalarMask);
            Reg::Store<float>(squares + row, partial, 1U);
        }
    }
}

using namespace AscendC;

/** Scale projection rows in place in group zero using the complete input-row RMS.
 * Formula: z[r,q] = ordered_sum_g(P[g,r,q]) * rsqrt(ordered_sum_k(S[k,r])/K + eps).
 * Only projection results are scaled; the x2 values used by ComputeHInTile remain unchanged.
 * Use Sqrt/Div as in MhcPre; ordered partial addition is unchanged and rsqrt(+Inf)=0.
 */
__aicore__ inline void NormalizeProjectionRows(__ubuf__ float* partial, __ubuf__ float* rms, uint32_t rows,
                                               uint32_t rowStride, uint32_t groupStride, uint32_t groups,
                                               uint32_t rmsStride, uint32_t kGrid, uint32_t k, float eps)
{
    const uint16_t rowCount = static_cast<uint16_t>(rows);
    const uint16_t groupCount = static_cast<uint16_t>(groups);
    const uint16_t shardCount = static_cast<uint16_t>(kGrid);
    const float meanScale = 1.0f / static_cast<float>(k);
    __VEC_SCOPE__
    {
        Reg::MaskReg mask = MaskForCount(kProjectionCoefficientElements);
        Reg::RegTensor<float> sum, value, squares, mean, inverse, one;
        Reg::Duplicate(one, 1.0f, mask);
        for (uint16_t row = 0; row < rowCount; ++row) {
            auto partialRow = partial + row * rowStride;
            auto rmsRow = rms + row;
            Reg::Duplicate(sum, 0.0f, mask);
            for (uint16_t group = 0; group < groupCount; ++group) {
                Reg::LoadAlign(value, partialRow + group * groupStride);
                Reg::Add(sum, sum, value, mask);
            }
            Reg::Duplicate(squares, 0.0f, mask);
            for (uint16_t shard = 0; shard < shardCount; ++shard) {
                Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(value, rmsRow + shard * rmsStride);
                Reg::Add(squares, squares, value, mask);
            }
            Reg::Muls(mean, squares, meanScale, mask);
            Reg::Adds(mean, mean, eps, mask);
            Reg::Sqrt(inverse, mean, mask);
            Reg::Div(inverse, one, inverse, mask);
            Reg::Mul(sum, sum, inverse, mask);
            Reg::StoreAlign(partialRow, sum, mask);
        }
    }
}

/** Compute gates from packed [pre:4, post:4, res:16]: logits=alpha[branch]*normalized+bias;
 * pre=sigmoid(logits[0:4])+eps; post=hc_post_scale*sigmoid(logits[4:8]); res=logits[8:24].
 * This produces current pre coefficients; external shifted_prev_mix uses a separate caller-owned UB plane.
 * Keep separate Mul/Add rounding. Output planes are [pre:4*alignedRows, post:4*alignedRows, res:16*alignedRows]. */
__aicore__ inline void ComputeGateRows(__ubuf__ float* normalized, __ubuf__ float* alpha, __ubuf__ float* bias,
                                       float eps, float hcPostScale, uint32_t rows, uint32_t rowStride,
                                       uint32_t alignedRows, __ubuf__ float* coefficients)
{
    const uint16_t rowCount = static_cast<uint16_t>(rows);
    __VEC_SCOPE__
    {
        Reg::MaskReg headMask = MaskForCount(kHeadCount);
        Reg::MaskReg sigmoidMask = MaskForCount(kGateGroupElements);
        Reg::MaskReg resMask = MaskForCount(kHeadMatrixElements);
        Reg::MaskReg affineMask = MaskForCount(kProjectionCoefficientElements);
        Reg::RegTensor<uint32_t> indices, postIndices, resIndices;
        Reg::RegTensor<float> alphaPre, alphaPost, alphaRes, scale, offset;
        Reg::RegTensor<float> projected, logits, residual, negative, expValue, one, gate, post;
        Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(indices), 0);
        Reg::Adds(postIndices, indices, kHeadCount, headMask);
        Reg::Adds(resIndices, indices, kGateGroupElements, resMask);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(alphaPre, alpha);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(alphaPost, alpha + 1U);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(alphaRes, alpha + 2U);
        Reg::Select(scale, alphaPost, alphaRes, sigmoidMask);
        Reg::Select(scale, alphaPre, scale, headMask);
        Reg::Gather(offset, bias, indices, affineMask);
        Reg::Duplicate(one, 1.0f, sigmoidMask);
        for (uint16_t row = 0; row < rowCount; ++row) {
            auto normalizedRow = normalized + row * rowStride;
            auto preRow = coefficients + row * kHeadCount;
            auto postRow = preRow + alignedRows * kHeadCount;
            auto resRow = coefficients + alignedRows * kGateGroupElements + row * kHeadMatrixElements;
            Reg::Gather(projected, normalizedRow, indices, affineMask);
            Reg::Mul(logits, projected, scale, affineMask);
            Reg::Add(logits, logits, offset, affineMask);
            Reg::Gather(residual, logits, resIndices);
            Reg::Store<float>(resRow, residual, kHeadMatrixElements);
            Reg::Muls(negative, logits, -1.0f, sigmoidMask);
            Reg::Exp(expValue, negative, sigmoidMask);
            Reg::Add(expValue, expValue, one, sigmoidMask);
            Reg::Div(gate, one, expValue, sigmoidMask);
            Reg::Gather(post, gate, postIndices);
            Reg::Muls(post, post, hcPostScale, headMask);
            Reg::Store<float>(postRow, post, kHeadCount);
            Reg::Adds(gate, gate, eps, headMask);
            Reg::Store<float>(preRow, gate, kHeadCount);
        }
    }
}

/** Mix four heads: hIn[d]=Cast<T>(sum_n hPre[n]*FP32(x2[n,d])).
 * The caller supplies hPre from shifted_prev_mix when present, or the current sigmoid-plus-eps pre coefficients.
 * This function applies the selected coefficients; RMSNorm1 follows on the rounded hIn.
 * Accumulate sumSquares from rounded hIn using independent even/odd 64-column chains. */
template <typename T>
__aicore__ inline void ComputeHInTile(__ubuf__ float* hPre, __ubuf__ T* x2, uint32_t d, uint32_t dStart,
                                      uint32_t dCount, __ubuf__ T* hIn, __ubuf__ float* sumSquares)
{
    const uint16_t repeatTimes = static_cast<uint16_t>((dCount + kRegCastVectorLanes - 1U) / kRegCastVectorLanes);
    const uint16_t pairedRepeats = static_cast<uint16_t>(repeatTimes / kVectorSegmentsPerPair);
    const uint16_t tailStart = static_cast<uint16_t>(pairedRepeats * kVectorSegmentsPerPair);
    __VEC_SCOPE__
    {
        Reg::MaskReg mask;
        Reg::RegTensor<T> input;
        Reg::RegTensor<float> value;
        Reg::RegTensor<float> weight0, weight1, weight2, weight3;
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(weight0, hPre);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(weight1, hPre + 1);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(weight2, hPre + 2);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(weight3, hPre + 3);
        Reg::RegTensor<float> sum;
        Reg::RegTensor<T> result;
        Reg::RegTensor<float> squareSum, square;
        Reg::MaskReg fullMask = Compute::MaskForCount(kRegFloatLanes);
        Reg::Duplicate(squareSum, 0.0f, fullMask);
        uint32_t remaining = dCount;
        Reg::RegTensor<float> oddSquares;
        Reg::Duplicate(oddSquares, 0.0f, fullMask);
        auto x2Head0 = x2 + 0U * d + dStart;
        auto x2Head1 = x2 + 1U * d + dStart;
        auto x2Head2 = x2 + 2U * d + dStart;
        auto x2Head3 = x2 + 3U * d + dStart;
        for (uint16_t repeat = 0; repeat < pairedRepeats; ++repeat) {
            {
                const uint32_t tileOffset = (repeat * kVectorSegmentsPerPair + 0U) * kRegCastVectorLanes;
                mask = Reg::UpdateMask<float>(remaining);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input, x2Head0 + tileOffset);
                Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
                Reg::Mul(sum, value, weight0, mask);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input, x2Head1 + tileOffset);
                Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
                Reg::MulAddDst(sum, value, weight1, mask);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input, x2Head2 + tileOffset);
                Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
                Reg::MulAddDst(sum, value, weight2, mask);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input, x2Head3 + tileOffset);
                Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
                Reg::MulAddDst(sum, value, weight3, mask);
                Reg::Cast<T, float, kCastFloatToInput>(result, sum, mask);
                Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(hIn + tileOffset, result, mask);
                Reg::Cast<float, T, kCastInputToFloat>(value, result, mask);
                Reg::MulAddDst(squareSum, value, value, fullMask);
            }
            {
                const uint32_t tileOffset = (repeat * kVectorSegmentsPerPair + 1U) * kRegCastVectorLanes;
                mask = Reg::UpdateMask<float>(remaining);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input, x2Head0 + tileOffset);
                Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
                Reg::Mul(sum, value, weight0, mask);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input, x2Head1 + tileOffset);
                Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
                Reg::MulAddDst(sum, value, weight1, mask);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input, x2Head2 + tileOffset);
                Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
                Reg::MulAddDst(sum, value, weight2, mask);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input, x2Head3 + tileOffset);
                Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
                Reg::MulAddDst(sum, value, weight3, mask);
                Reg::Cast<T, float, kCastFloatToInput>(result, sum, mask);
                Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(hIn + tileOffset, result, mask);
                Reg::Cast<float, T, kCastInputToFloat>(value, result, mask);
                Reg::MulAddDst(oddSquares, value, value, fullMask);
            }
        }
        for (uint16_t repeat = tailStart; repeat < repeatTimes; ++repeat) {
            const uint32_t tileOffset = repeat * kRegCastVectorLanes;
            mask = Reg::UpdateMask<float>(remaining);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input, x2Head0 + tileOffset);
            Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
            Reg::Mul(sum, value, weight0, mask);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input, x2Head1 + tileOffset);
            Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
            Reg::MulAddDst(sum, value, weight1, mask);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input, x2Head2 + tileOffset);
            Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
            Reg::MulAddDst(sum, value, weight2, mask);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input, x2Head3 + tileOffset);
            Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
            Reg::MulAddDst(sum, value, weight3, mask);
            Reg::Cast<T, float, kCastFloatToInput>(result, sum, mask);
            Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(hIn + tileOffset, result, mask);
            Reg::Cast<float, T, kCastInputToFloat>(value, result, mask);
            Reg::MulAddDst(squareSum, value, value, fullMask);
        }
        Reg::Add(squareSum, squareSum, oddSquares, fullMask);

        Reg::Reduce<Reg::ReduceType::SUM>(square, squareSum, fullMask);
        Reg::Store<float>(sumSquares, square, 1U);
    }
}

} // namespace MegaMhc::Compute

#endif // OPS_TRANSFORMER_MHC_MEGA_MHC_PRE_H
