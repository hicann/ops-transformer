/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_VF_RESCALE_O_FP32_PER_CHANNEL_H
#define GBSA_VF_RESCALE_O_FP32_PER_CHANNEL_H

#include "vf_mxfp4_common.h"

namespace NpuArch::Epilogue::Block::GbsaMxfp4VF {

template <typename OutputT>
__simd_vf__ inline void FillEmptyOutput(__ubuf__ OutputT* output, __ubuf__ float* lse, uint32_t elements, uint32_t rows,
                                        bool outputLse, bool packedLse)
{
    RegTensor<OutputT> zero;
    RegTensor<float> empty;
    RegTensor<uint32_t> indices;
    Duplicate(zero, static_cast<OutputT>(0));
    const uint32_t repeats = (elements + 127U) / 128U;
    for (uint32_t i = 0; i < repeats; ++i) {
        MaskReg mask = UpdateMask<OutputT>(elements);
        StoreAlign(output + i * 128U, zero, mask);
    }
    if (outputLse) {
        MaskReg mask = UpdateMask<float>(rows);
        Duplicate(empty, EMPTY_LSE);
        if (packedLse) {
            Arange((RegTensor<int32_t>&)indices, 0);
            Muls(indices, indices, static_cast<uint32_t>(8), mask);
            Scatter(lse, empty, indices, mask);
        } else {
            StoreAlign(lse, empty, mask);
        }
    }
}

template <typename OutputT>
__aicore__ inline void FillEmptyOutputCallVF(const LocalTensor<OutputT>& output, const LocalTensor<float>& lse,
                                             uint32_t elements, uint32_t rows, bool outputLse, bool packedLse)
{
    FillEmptyOutput<OutputT>(reinterpret_cast<__ubuf__ OutputT*>(output.GetPhyAddr()),
                             reinterpret_cast<__ubuf__ float*>(lse.GetPhyAddr()), elements, rows, outputLse, packedLse);
}

template <uint16_t D, uint16_t R>
__simd_vf__ inline void StaticRescaleOutput(__ubuf__ float* accumulator, __ubuf__ float* pv,
                                            __ubuf__ float* localRowsum, __ubuf__ float* outL, __ubuf__ float* tileDm,
                                            uint32_t validRows)
{
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    MaskReg all = CreateMask<float, MaskPattern::ALL>();
    uint32_t rows = validRows;
    MaskReg rowMask = UpdateMask<float>(rows);
    RegTensor<float> dm, previous, current;
    RegTensor<float> previous0, previous1, previous2, previous3;
    RegTensor<float> current0, current1, current2, current3;
    LoadAlign(dm, tileDm);
    LoadAlign(previous, outL);
    LoadAlign(current, localRowsum);
    Mul(previous, previous, dm, rowMask);
    Add(current, current, previous, rowMask);
    StoreAlign(outL, current, rowMask);
    for (uint16_t row = 0; row < validRows; ++row) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(dm, tileDm + row);
        LoadAlign(previous0, accumulator + row * D + 0);
        if constexpr (D >= 128) {
            LoadAlign(previous1, accumulator + row * D + 64);
        }
        if constexpr (D >= 192) {
            LoadAlign(previous2, accumulator + row * D + 128);
        }
        if constexpr (D >= 256) {
            LoadAlign(previous3, accumulator + row * D + 192);
        }
        LoadAlign(current0, pv + row * D + 0);
        if constexpr (D >= 128) {
            LoadAlign(current1, pv + row * D + 64);
        }
        if constexpr (D >= 192) {
            LoadAlign(current2, pv + row * D + 128);
        }
        if constexpr (D >= 256) {
            LoadAlign(current3, pv + row * D + 192);
        }
        Mul(previous0, previous0, dm, all);
        if constexpr (D >= 128) {
            Mul(previous1, previous1, dm, all);
        }
        if constexpr (D >= 192) {
            Mul(previous2, previous2, dm, all);
        }
        if constexpr (D >= 256) {
            Mul(previous3, previous3, dm, all);
        }
        Add(current0, current0, previous0, all);
        if constexpr (D >= 128) {
            Add(current1, current1, previous1, all);
        }
        if constexpr (D >= 192) {
            Add(current2, current2, previous2, all);
        }
        if constexpr (D >= 256) {
            Add(current3, current3, previous3, all);
        }
        StoreAlign(accumulator + row * D + 0, current0, all);
        if constexpr (D >= 128) {
            StoreAlign(accumulator + row * D + 64, current1, all);
        }
        if constexpr (D >= 192) {
            StoreAlign(accumulator + row * D + 128, current2, all);
        }
        if constexpr (D >= 256) {
            StoreAlign(accumulator + row * D + 192, current3, all);
        }
    }
}

__simd_vf__ inline void ComputeLse(__ubuf__ float* lse, __ubuf__ float* outG, __ubuf__ float* outL, uint32_t validRows,
                                   float log2CxCeil)
{
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    MaskReg all = CreateMask<float, MaskPattern::ALL>();
    all = UpdateMask<float>(validRows);
    RegTensor<float> g, l, logarithm, unit, empty;
    MaskReg valid;
    LoadAlign(g, outG);
    LoadAlign(l, outL);
    Duplicate(unit, 1.0f);
    Duplicate(empty, EMPTY_LSE);
    Compares<float, CMPMODE::GT>(valid, l, 0.0f, all);
    Select(l, l, unit, valid);
    Ln(logarithm, l, all);
    Adds(g, g, log2CxCeil, all);
    Muls(g, g, LN2, all);
    Adds(g, g, LN1P5, all);
    Add(g, g, logarithm, all);
    Select(g, g, empty, valid);
    StoreAlign(lse, g, all);
}

template <uint16_t D, uint16_t R, typename OutputT>
__simd_vf__ inline void StaticNormalizeOutput(__ubuf__ OutputT* output, __ubuf__ float* accumulator,
                                              __ubuf__ float* outL, uint16_t firstRow, uint16_t rowCount)
{
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    MaskReg all = CreateMask<float, MaskPattern::ALL>();
    MaskReg valid;
    RegTensor<float> denominator, zero, unit;
    RegTensor<float> value0, value1, value2, value3;
    RegTensor<OutputT> converted0, converted1, converted2, converted3;
    Duplicate(zero, 0.0f);
    Duplicate(unit, 1.0f);
    for (uint16_t localRow = 0; localRow < rowCount; ++localRow) {
        const uint16_t row = firstRow + localRow;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(denominator, outL + row);
        Compares<float, CMPMODE::GT>(valid, denominator, 0.0f, all);
        Select(denominator, denominator, unit, valid);
        LoadAlign(value0, accumulator + row * D + 0);
        if constexpr (D >= 128) {
            LoadAlign(value1, accumulator + row * D + 64);
        }
        if constexpr (D >= 192) {
            LoadAlign(value2, accumulator + row * D + 128);
        }
        if constexpr (D >= 256) {
            LoadAlign(value3, accumulator + row * D + 192);
        }
        Div(value0, value0, denominator, all);
        if constexpr (D >= 128) {
            Div(value1, value1, denominator, all);
        }
        if constexpr (D >= 192) {
            Div(value2, value2, denominator, all);
        }
        if constexpr (D >= 256) {
            Div(value3, value3, denominator, all);
        }
        Select(value0, value0, zero, valid);
        if constexpr (D >= 128) {
            Select(value1, value1, zero, valid);
        }
        if constexpr (D >= 192) {
            Select(value2, value2, zero, valid);
        }
        if constexpr (D >= 256) {
            Select(value3, value3, zero, valid);
        }
        Cast<OutputT, float, OUTPUT_RINT>(converted0, value0, all);
        if constexpr (D >= 128) {
            Cast<OutputT, float, OUTPUT_RINT>(converted1, value1, all);
        }
        if constexpr (D >= 192) {
            Cast<OutputT, float, OUTPUT_RINT>(converted2, value2, all);
        }
        if constexpr (D >= 256) {
            Cast<OutputT, float, OUTPUT_RINT>(converted3, value3, all);
        }
        StoreAlign<OutputT, StoreDist::DIST_PACK_B32>(output + localRow * D + 0, converted0, all);
        if constexpr (D >= 128) {
            StoreAlign<OutputT, StoreDist::DIST_PACK_B32>(output + localRow * D + 64, converted1, all);
        }
        if constexpr (D >= 192) {
            StoreAlign<OutputT, StoreDist::DIST_PACK_B32>(output + localRow * D + 128, converted2, all);
        }
        if constexpr (D >= 256) {
            StoreAlign<OutputT, StoreDist::DIST_PACK_B32>(output + localRow * D + 192, converted3, all);
        }
    }
}

__simd_vf__ inline void PackLse(__ubuf__ float* dst, __ubuf__ float* src, uint32_t rows)
{
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    MaskReg mask = UpdateMask<float>(rows);
    RegTensor<float> value;
    RegTensor<uint32_t> indices;
    Arange((RegTensor<int32_t>&)indices, 0);
    Muls(indices, indices, static_cast<uint32_t>(8), mask);
    LoadAlign(value, src);
    Scatter(dst, value, indices, mask);
}

__aicore__ inline void PackLseCallVF(const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t rows)
{
    PackLse(reinterpret_cast<__ubuf__ float*>(dst.GetPhyAddr()), reinterpret_cast<__ubuf__ float*>(src.GetPhyAddr()),
            rows);
}

template <uint16_t D, uint16_t R>
__aicore__ inline void StaticRescaleCallVF(const LocalTensor<float>& output, const LocalTensor<float>& pv,
                                           const LocalTensor<float>& rowSum, const LocalTensor<float>& sum,
                                           const LocalTensor<float>& dm, uint32_t rows)
{
    StaticRescaleOutput<D, R>(
        reinterpret_cast<__ubuf__ float*>(output.GetPhyAddr()), reinterpret_cast<__ubuf__ float*>(pv.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(rowSum.GetPhyAddr()), reinterpret_cast<__ubuf__ float*>(sum.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(dm.GetPhyAddr()), rows);
}

template <uint16_t D, uint16_t R, class OutputT>
__aicore__ inline void StaticNormalizeCallVF(const LocalTensor<OutputT>& result, const LocalTensor<float>& output,
                                             const LocalTensor<float>& sum, uint16_t first, uint16_t rows)
{
    StaticNormalizeOutput<D, R, OutputT>(reinterpret_cast<__ubuf__ OutputT*>(result.GetPhyAddr()),
                                         reinterpret_cast<__ubuf__ float*>(output.GetPhyAddr()),
                                         reinterpret_cast<__ubuf__ float*>(sum.GetPhyAddr()), first, rows);
}

__aicore__ inline void LseCallVF(const LocalTensor<float>& result, const LocalTensor<float>& maximum,
                                 const LocalTensor<float>& sum, uint32_t rows, float log2CxCeil)
{
    ComputeLse(reinterpret_cast<__ubuf__ float*>(result.GetPhyAddr()),
               reinterpret_cast<__ubuf__ float*>(maximum.GetPhyAddr()),
               reinterpret_cast<__ubuf__ float*>(sum.GetPhyAddr()), rows, log2CxCeil);
}

__aicore__ inline void RescaleCallVF(const LocalTensor<float>& output, const LocalTensor<float>& pv,
                                     const LocalTensor<float>& rowSum, const LocalTensor<float>& sum,
                                     const LocalTensor<float>& dm, uint32_t rows, uint16_t D, uint16_t R)
{
    if (D == 64) {
        StaticRescaleCallVF<64, 64>(output, pv, rowSum, sum, dm, rows);
    } else if (D == 128) {
        StaticRescaleCallVF<128, 64>(output, pv, rowSum, sum, dm, rows);
    } else if (D == 192) {
        StaticRescaleCallVF<192, 64>(output, pv, rowSum, sum, dm, rows);
    } else if (D == 256) {
        StaticRescaleCallVF<256, 64>(output, pv, rowSum, sum, dm, rows);
    }
}

template <class OutputT>
__aicore__ inline void NormalizeCallVF(const LocalTensor<OutputT>& result, const LocalTensor<float>& output,
                                       const LocalTensor<float>& sum, uint16_t first, uint16_t rows, uint16_t D,
                                       uint16_t R)
{
    if (D == 64) {
        StaticNormalizeCallVF<64, 64, OutputT>(result, output, sum, first, rows);
    } else if (D == 128) {
        StaticNormalizeCallVF<128, 64, OutputT>(result, output, sum, first, rows);
    } else if (D == 192) {
        StaticNormalizeCallVF<192, 64, OutputT>(result, output, sum, first, rows);
    } else if (D == 256) {
        StaticNormalizeCallVF<256, 64, OutputT>(result, output, sum, first, rows);
    }
}

} // namespace NpuArch::Epilogue::Block::GbsaMxfp4VF
#endif
