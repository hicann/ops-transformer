/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_TRANSFORMER_MHC_MEGA_MHC_SINKHORN_H
#define OPS_TRANSFORMER_MHC_MEGA_MHC_SINKHORN_H

#include "rms_norm_common.h"

namespace MegaMhc::Compute {

/** Reduce each independent four-lane row and broadcast its sum or maximum. */
template <bool MAXIMUM>
__simd_callee__ inline void SinkhornRowReduce(Reg::RegTensor<float>& result, Reg::RegTensor<float>& row,
                                              Reg::MaskReg& mask)
{
    Reg::RegTensor<float> even, odd, first, second;
    Reg::DeInterleave(even, odd, row, row);
    if constexpr (MAXIMUM) {
        Reg::Max(even, even, odd, mask);
    } else {
        Reg::Add(even, even, odd, mask);
    }
    Reg::DeInterleave(first, second, even, even);
    if constexpr (MAXIMUM) {
        Reg::Max(first, first, second, mask);
    } else {
        Reg::Add(first, first, second, mask);
    }
    Reg::Interleave(even, odd, first, first);
    Reg::Interleave(result, odd, even, even);
}

__simd_callee__ inline void SinkhornSoftmaxRow(Reg::RegTensor<float>& row, float eps, Reg::MaskReg& mask)
{
    Reg::RegTensor<float> reduction;
    SinkhornRowReduce<true>(reduction, row, mask);
    Reg::Sub(row, row, reduction, mask);
    Reg::Exp(row, row, mask);
    SinkhornRowReduce<false>(reduction, row, mask);
    Reg::Div(row, row, reduction, mask);
    Reg::Adds(row, row, eps, mask);
}

__simd_callee__ inline void SinkhornNormalizeRow(Reg::RegTensor<float>& row, float eps, Reg::MaskReg& mask)
{
    Reg::RegTensor<float> reduction;
    SinkhornRowReduce<false>(reduction, row, mask);
    Reg::Adds(reduction, reduction, eps, mask);
    Reg::Div(row, row, reduction, mask);
}

__simd_callee__ inline void SinkhornNormalizeColumns(Reg::RegTensor<float>& row0, Reg::RegTensor<float>& row1,
                                                     Reg::RegTensor<float>& row2, Reg::RegTensor<float>& row3,
                                                     float eps, Reg::MaskReg& mask)
{
    Reg::RegTensor<float> sum;
    Reg::Add(sum, row0, row1, mask);
    Reg::Add(sum, sum, row2, mask);
    Reg::Add(sum, sum, row3, mask);
    Reg::Adds(sum, sum, eps, mask);
    Reg::Div(row0, row0, sum, mask);
    Reg::Div(row1, row1, sum, mask);
    Reg::Div(row2, row2, sum, mask);
    Reg::Div(row3, row3, sum, mask);
}

/** In-place row-major [rows,4,4] logits -> Sinkhorn coefficients.
 * Four registers hold the four matrix rows for up to 16 tokens. All iterations remain in registers.
 * Zero iterations preserves logits; the positive iteration count comes from Host tiling.
 */
__aicore__ inline void ComputeSinkhorn4x4Rows(__ubuf__ float* matrices, uint32_t rows, float eps, uint32_t iterations)
{
    if (iterations == 0U) {
        return;
    }
    constexpr uint32_t tokensPerRegister = kRegFloatLanes / kHeadCount;
    const uint16_t batches = static_cast<uint16_t>((rows + tokensPerRegister - 1U) / tokensPerRegister);
    // Softmax and the first column normalization precede the loop; count only the remaining iterations.
    const uint16_t iterationCount = static_cast<uint16_t>(iterations - 1U);
    __VEC_SCOPE__
    {
        Reg::MaskReg full = MaskForCount(kRegFloatLanes);
        Reg::RegTensor<uint32_t> lane, token, column, index, row1Index, row2Index, row3Index, columnMask;
        Reg::RegTensor<int32_t> shift;
        Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(lane), 0);
        Reg::Duplicate(shift, 2, full);
        Reg::Duplicate(columnMask, kHeadCount - 1U, full);
        Reg::ShiftRight(token, lane, shift, full);
        Reg::And(column, lane, columnMask, full);
        Reg::Muls(index, token, kHeadMatrixElements, full);
        Reg::Add(index, index, column, full);
        // Gather/Scatter bases must remain 32B aligned, including the odd matrix rows.
        Reg::Adds(row1Index, index, kHeadCount, full);
        Reg::Adds(row2Index, index, 2U * kHeadCount, full);
        Reg::Adds(row3Index, index, 3U * kHeadCount, full);
        uint32_t remaining = rows * kHeadCount;
        for (uint16_t batch = 0U; batch < batches; ++batch) {
            Reg::MaskReg mask = Reg::UpdateMask<float>(remaining);
            auto base = matrices + static_cast<uint32_t>(batch) * tokensPerRegister * kHeadMatrixElements;
            Reg::RegTensor<float> row0, row1, row2, row3;
            Reg::Gather(row0, base, index, mask);
            Reg::Gather(row1, base, row1Index, mask);
            Reg::Gather(row2, base, row2Index, mask);
            Reg::Gather(row3, base, row3Index, mask);
            SinkhornSoftmaxRow(row0, eps, mask);
            SinkhornSoftmaxRow(row1, eps, mask);
            SinkhornSoftmaxRow(row2, eps, mask);
            SinkhornSoftmaxRow(row3, eps, mask);
            SinkhornNormalizeColumns(row0, row1, row2, row3, eps, mask);
            for (uint16_t iteration = 0U; iteration < iterationCount; ++iteration) {
                SinkhornNormalizeRow(row0, eps, mask);
                SinkhornNormalizeRow(row1, eps, mask);
                SinkhornNormalizeRow(row2, eps, mask);
                SinkhornNormalizeRow(row3, eps, mask);
                SinkhornNormalizeColumns(row0, row1, row2, row3, eps, mask);
            }
            Reg::Scatter(base, row0, index, mask);
            Reg::Scatter(base, row1, row1Index, mask);
            Reg::Scatter(base, row2, row2Index, mask);
            Reg::Scatter(base, row3, row3Index, mask);
        }
    }
}
} // namespace MegaMhc::Compute

#endif // OPS_TRANSFORMER_MHC_MEGA_MHC_SINKHORN_H
