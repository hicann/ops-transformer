/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_VF_MXFP4_COMMON_H
#define GBSA_VF_MXFP4_COMMON_H

#include "kernel_operator.h"

namespace NpuArch::Epilogue::Block::GbsaMxfp4VF {
using namespace AscendC;
using namespace AscendC::Reg;

constexpr float LN2 = 0.6931471806f;
constexpr float INV_LN2 = 1.4426950409f;
// CX is supplied by host tiling; only the additional log-domain shift is fixed.
constexpr float LN1P5 = 0.4054651081f;
constexpr float NEG_INF = -__builtin_inff();
constexpr float EMPTY_LSE = -3.402823466e+38f;
constexpr CastTrait BF16_ROUND = {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
constexpr CastTrait FP4_ROUND = {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
constexpr CastTrait INTEGER_ROUND = {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
constexpr CastTrait OUTPUT_RINT = {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

__simd_vf__ inline void InitInitialMax(__ubuf__ float* maximum, uint32_t rows)
{
    RegTensor<float> negative;
    MaskReg mask = UpdateMask<float>(rows);
    Duplicate(negative, NEG_INF);
    StoreAlign(maximum, negative, mask);
}

__aicore__ inline void InitInitialMaxCallVF(const LocalTensor<float>& maximum, uint32_t rows)
{
    InitInitialMax(reinterpret_cast<__ubuf__ float*>(maximum.GetPhyAddr()), rows);
}

__simd_vf__ inline void CopyVector(__ubuf__ float* dst, __ubuf__ float* src, uint32_t validRows)
{
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    RegTensor<float> value;
    MaskReg mask = UpdateMask<float>(validRows);
    LoadAlign(value, src);
    StoreAlign(dst, value, mask);
}

__aicore__ inline void CopyVectorCallVF(const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t rows)
{
    CopyVector(reinterpret_cast<__ubuf__ float*>(dst.GetPhyAddr()), reinterpret_cast<__ubuf__ float*>(src.GetPhyAddr()),
               rows);
}

template <uint16_t D, bool ClearStats = true>
__simd_vf__ inline void StaticInitTask(__ubuf__ float* output, __ubuf__ float* sum, __ubuf__ float* maximum,
                                       __ubuf__ float* groupMax, uint32_t rows)
{
    MaskReg all = CreateMask<float, MaskPattern::ALL>();
    // UpdateMask consumes its count argument; preserve the loop bound.
    uint32_t maskRows = rows;
    MaskReg rowMask = UpdateMask<float>(maskRows);
    RegTensor<float> zero, negative;
    Duplicate(zero, 0.0f);
    Duplicate(negative, NEG_INF);
    StoreAlign(sum, zero, rowMask);
    StoreAlign(maximum, negative, rowMask);
    if constexpr (ClearStats) {
        StoreAlign(groupMax, negative, rowMask);
        StoreAlign(groupMax + 64, negative, rowMask);
        StoreAlign(groupMax + 128, negative, rowMask);
        StoreAlign(groupMax + 192, negative, rowMask);
    }
    for (uint16_t row = 0; row < rows; ++row) {
        StoreAlign(output + row * D, zero, all);
        if constexpr (D >= 128) {
            StoreAlign(output + row * D + 64, zero, all);
        }
        if constexpr (D >= 192) {
            StoreAlign(output + row * D + 128, zero, all);
        }
        if constexpr (D == 256) {
            StoreAlign(output + row * D + 192, zero, all);
        }
    }
}

template <uint16_t D, bool ClearStats = true>
__aicore__ inline void StaticInitTaskCallVF(const LocalTensor<float>& output, const LocalTensor<float>& sum,
                                            const LocalTensor<float>& maximum, const LocalTensor<float>& groupMax,
                                            uint32_t rows)
{
    StaticInitTask<D, ClearStats>(reinterpret_cast<__ubuf__ float*>(output.GetPhyAddr()),
                                  reinterpret_cast<__ubuf__ float*>(sum.GetPhyAddr()),
                                  reinterpret_cast<__ubuf__ float*>(maximum.GetPhyAddr()),
                                  reinterpret_cast<__ubuf__ float*>(groupMax.GetPhyAddr()), rows);
}

template <bool ClearStats = true>
__aicore__ inline void InitTaskCallVF(const LocalTensor<float>& output, const LocalTensor<float>& sum,
                                      const LocalTensor<float>& maximum, const LocalTensor<float>& groupMax,
                                      uint32_t rows, uint16_t D)
{
    if (D == 64) {
        StaticInitTaskCallVF<64, ClearStats>(output, sum, maximum, groupMax, rows);
    } else if (D == 128) {
        StaticInitTaskCallVF<128, ClearStats>(output, sum, maximum, groupMax, rows);
    } else if (D == 192) {
        StaticInitTaskCallVF<192, ClearStats>(output, sum, maximum, groupMax, rows);
    } else if (D == 256) {
        StaticInitTaskCallVF<256, ClearStats>(output, sum, maximum, groupMax, rows);
    }
}

} // namespace NpuArch::Epilogue::Block::GbsaMxfp4VF
#endif
