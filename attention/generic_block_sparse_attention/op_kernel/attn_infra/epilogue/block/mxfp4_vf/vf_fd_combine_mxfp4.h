/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_VF_FD_COMBINE_MXFP4_H
#define GBSA_VF_FD_COMBINE_MXFP4_H

#include "vf_mxfp4_common.h"

namespace NpuArch::Epilogue::Block::GbsaMxfp4VF {

// All partials of one head row are loaded before entering VF. Statistics are
// 32-byte spaced; the D channels remain contiguous FP32 until the final cast.
template <uint16_t D, typename OutputT, bool OutputLse>
__simd_vf__ inline void StaticCombinePartial(__ubuf__ OutputT* output, __ubuf__ float* lse, __ubuf__ float* numerator,
                                             __ubuf__ float* maximum, __ubuf__ float* sum, uint16_t splits,
                                             float log2CxCeil)
{
    MaskReg all = CreateMask<float, MaskPattern::ALL>();
    MaskReg valid;
    RegTensor<float> g, l, globalG, globalL, weight, zero, unit, negative;
    RegTensor<float> n0, n1, n2, n3, acc0, acc1, acc2, acc3;
    RegTensor<OutputT> result0, result1, result2, result3;
    Duplicate(zero, 0.0f);
    Duplicate(unit, 1.0f);
    Duplicate(negative, NEG_INF);
    Duplicate(globalG, NEG_INF);
    Duplicate(globalL, 0.0f);
    Duplicate(acc0, 0.0f);
    if constexpr (D >= 128) {
        Duplicate(acc1, 0.0f);
    }
    if constexpr (D >= 192) {
        Duplicate(acc2, 0.0f);
    }
    if constexpr (D >= 256) {
        Duplicate(acc3, 0.0f);
    }
    for (uint16_t split = 0; split < splits; ++split) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(g, maximum + split * 8);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(l, sum + split * 8);
        Compares<float, CMPMODE::GT>(valid, l, 0.0f, all);
        Select(g, g, negative, valid);
        Max(globalG, globalG, g, all);
    }
    // Empty partials are neutral, including the all-empty case.
    Compares<float, CMPMODE::GT>(valid, globalG, NEG_INF, all);
    Select(globalG, globalG, zero, valid);
    for (uint16_t split = 0; split < splits; ++split) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(g, maximum + split * 8);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(l, sum + split * 8);
        Compares<float, CMPMODE::GT>(valid, l, 0.0f, all);
        Select(g, g, globalG, valid);
        Sub(weight, g, globalG, all);
        Muls(weight, weight, LN2, all);
        Exp(weight, weight, all);
        Select(weight, weight, zero, valid);
        Mul(l, l, weight, all);
        Add(globalL, globalL, l, all);
        LoadAlign(n0, numerator + split * D + 0);
        Mul(n0, n0, weight, all);
        Add(acc0, acc0, n0, all);
        if constexpr (D >= 128) {
            LoadAlign(n1, numerator + split * D + 64);
            Mul(n1, n1, weight, all);
            Add(acc1, acc1, n1, all);
        }
        if constexpr (D >= 192) {
            LoadAlign(n2, numerator + split * D + 128);
            Mul(n2, n2, weight, all);
            Add(acc2, acc2, n2, all);
        }
        if constexpr (D >= 256) {
            LoadAlign(n3, numerator + split * D + 192);
            Mul(n3, n3, weight, all);
            Add(acc3, acc3, n3, all);
        }
    }
    Compares<float, CMPMODE::GT>(valid, globalL, 0.0f, all);
    Select(globalL, globalL, unit, valid);
    Div(acc0, acc0, globalL, all);
    Select(acc0, acc0, zero, valid);
    Cast<OutputT, float, OUTPUT_RINT>(result0, acc0, all);
    StoreAlign<OutputT, StoreDist::DIST_PACK_B32>(output + 0, result0, all);
    if constexpr (D >= 128) {
        Div(acc1, acc1, globalL, all);
        Select(acc1, acc1, zero, valid);
        Cast<OutputT, float, OUTPUT_RINT>(result1, acc1, all);
        StoreAlign<OutputT, StoreDist::DIST_PACK_B32>(output + 64, result1, all);
    }
    if constexpr (D >= 192) {
        Div(acc2, acc2, globalL, all);
        Select(acc2, acc2, zero, valid);
        Cast<OutputT, float, OUTPUT_RINT>(result2, acc2, all);
        StoreAlign<OutputT, StoreDist::DIST_PACK_B32>(output + 128, result2, all);
    }
    if constexpr (D >= 256) {
        Div(acc3, acc3, globalL, all);
        Select(acc3, acc3, zero, valid);
        Cast<OutputT, float, OUTPUT_RINT>(result3, acc3, all);
        StoreAlign<OutputT, StoreDist::DIST_PACK_B32>(output + 192, result3, all);
    }
    if constexpr (OutputLse) {
        Ln(globalL, globalL, all);
        Adds(globalG, globalG, log2CxCeil, all);
        Muls(globalG, globalG, LN2, all);
        Adds(globalG, globalG, LN1P5, all);
        Add(globalG, globalG, globalL, all);
        Duplicate(negative, EMPTY_LSE);
        Select(globalG, globalG, negative, valid);
        StoreAlign(lse, globalG, all);
    }
}

template <uint16_t D, typename OutputT, bool OutputLse>
__aicore__ inline void StaticCombinePartialCallVF(const LocalTensor<OutputT>& output, const LocalTensor<float>& lse,
                                                  const LocalTensor<float>& numerator,
                                                  const LocalTensor<float>& maximum, const LocalTensor<float>& sum,
                                                  uint16_t splits, float log2CxCeil)
{
    StaticCombinePartial<D, OutputT, OutputLse>(
        reinterpret_cast<__ubuf__ OutputT*>(output.GetPhyAddr()), reinterpret_cast<__ubuf__ float*>(lse.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(numerator.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(maximum.GetPhyAddr()), reinterpret_cast<__ubuf__ float*>(sum.GetPhyAddr()),
        splits, log2CxCeil);
}

template <typename OutputT, bool OutputLse>
__aicore__ inline void CombinePartialCallVF(const LocalTensor<OutputT>& output, const LocalTensor<float>& lse,
                                            const LocalTensor<float>& numerator, const LocalTensor<float>& maximum,
                                            const LocalTensor<float>& sum, uint16_t splits, float log2CxCeil,
                                            uint16_t D)
{
    if (D == 64) {
        StaticCombinePartialCallVF<64, OutputT, OutputLse>(output, lse, numerator, maximum, sum, splits, log2CxCeil);
    } else if (D == 128) {
        StaticCombinePartialCallVF<128, OutputT, OutputLse>(output, lse, numerator, maximum, sum, splits, log2CxCeil);
    } else if (D == 192) {
        StaticCombinePartialCallVF<192, OutputT, OutputLse>(output, lse, numerator, maximum, sum, splits, log2CxCeil);
    } else if (D == 256) {
        StaticCombinePartialCallVF<256, OutputT, OutputLse>(output, lse, numerator, maximum, sum, splits, log2CxCeil);
    }
}

} // namespace NpuArch::Epilogue::Block::GbsaMxfp4VF
#endif
