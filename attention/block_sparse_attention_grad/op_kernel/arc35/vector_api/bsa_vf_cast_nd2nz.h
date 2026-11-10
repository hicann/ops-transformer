/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include "kernel_tensor.h"
namespace AscendC {

using namespace Reg;
constexpr AscendC::Reg::CastTrait castTraitFp322Fp16Odd = {
    AscendC::Reg::RegLayout::ONE,
    AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};
constexpr AscendC::Reg::CastTrait castTraitFp322Fp16Even = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};
constexpr AscendC::Reg::CastTrait castTraitFp322Hif8Even = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};
constexpr AscendC::Reg::CastTrait castTraitFp322Hif8Odd = {
    AscendC::Reg::RegLayout::TWO,
    AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

// Enforce the finite saturating contract independently of cast lowering.
__simd_callee__ inline void ClampHif8Finite(RegTensor<float>& value, MaskReg mask)
{
    RegTensor<float> hi, lo;
    Duplicate(hi, 32768.0f);
    Duplicate(lo, -32768.0f);
    Min(value, value, hi, mask);
    Max(value, value, lo, mask);
}

template <typename T1>
__aicore__ inline void CastND2NZ(const LocalTensor<T1>& dstTensor, const LocalTensor<float>& srcTensor,
                                 const LocalTensor<uint8_t>& hif8GatherIndexes, const uint32_t srcM,
                                 const uint32_t srcN)
{
    const uint32_t blockSize = 32;
    const uint32_t blockN = blockSize / sizeof(T1); // 16
    const uint32_t fullExeSize = srcN;
    uint64_t srcLocalInt = srcTensor.GetPhyAddr();
    uint64_t dstLocalInt = dstTensor.GetPhyAddr();
    uint32_t blockStride = (srcM * blockN) * sizeof(T1) / blockSize + 1;
    uint32_t repeatStride = 1;
    if constexpr (IsSameType<T1, hifloat8_t>::value) {
        uint64_t indexesLocalInt = hif8GatherIndexes.GetPhyAddr();
        __VEC_SCOPE__
        {
            RegTensor<float> vregSrcEven;
            RegTensor<float> vregSrcOdd;
            RegTensor<hifloat8_t> vregCastEven;
            RegTensor<hifloat8_t> vregCastOdd;
            RegTensor<hifloat8_t> vregCastPacked;
            RegTensor<hifloat8_t> vregCastRes;
            RegTensor<uint8_t> vregIndexes;
            MaskReg pregFp32 = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregHif8 = CreateMask<hifloat8_t, MaskPattern::ALL>();
            uint32_t validN = fullExeSize;
            MaskReg pregValidN = UpdateMask<hifloat8_t>(validN);
            // The gather permutation is constant for the whole tile. Loading
            // it once avoids one UB register load for every source row.
            DataCopy<uint8_t, Reg::LoadDist::DIST_NORM>(vregIndexes, ((__ubuf__ uint8_t*&)indexesLocalInt));

            for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
                DataCopy<float, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B32>(
                    vregSrcEven, vregSrcOdd, ((__ubuf__ float*&)srcLocalInt), fullExeSize);
                ClampHif8Finite(vregSrcEven, pregFp32);
                ClampHif8Finite(vregSrcOdd, pregFp32);
                Cast<hifloat8_t, float, castTraitFp322Hif8Even>(vregCastEven, vregSrcEven, pregFp32);
                Cast<hifloat8_t, float, castTraitFp322Hif8Odd>(vregCastOdd, vregSrcOdd, pregFp32);
                Or((RegTensor<uint8_t>&)vregCastPacked, (RegTensor<uint8_t>&)vregCastEven,
                   (RegTensor<uint8_t>&)vregCastOdd, pregHif8);
                Gather<uint8_t>((RegTensor<uint8_t>&)vregCastRes, (RegTensor<uint8_t>&)vregCastPacked, vregIndexes);
                DataCopy<hifloat8_t, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ hifloat8_t*&)dstLocalInt), vregCastRes, blockStride, repeatStride, pregValidN);
            }
        }
    } else {
        __VEC_SCOPE__
        {
            RegTensor<float> vregSrcEven;
            RegTensor<float> vregSrcOdd;
            RegTensor<T1> vregCastEven;
            RegTensor<T1> vregCastOdd;
            RegTensor<T1> vregCastRes;
            uint32_t exeSize = fullExeSize;
            MaskReg pregFullExe = CreateMask<T1, MaskPattern::ALL>();
            MaskReg pregValidN = UpdateMask<T1>(exeSize);

            // [m,n] -> [n1,m1,16,16] -> [n1,m1*16,16] -> [n1,m1*16+1,16]
            for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
                DataCopy<float, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B32>(
                    vregSrcEven, vregSrcOdd, ((__ubuf__ float*&)srcLocalInt), fullExeSize);
                Cast<T1, float, castTraitFp322Fp16Even>(vregCastEven, vregSrcEven, pregFullExe);
                Cast<T1, float, castTraitFp322Fp16Odd>(vregCastOdd, vregSrcOdd, pregFullExe);
                // 0101: b16 0001: b32 1111: b8
                Or((RegTensor<uint16_t>&)vregCastRes, (RegTensor<uint16_t>&)vregCastEven,
                   (RegTensor<uint16_t>&)vregCastOdd, pregFullExe);
                // high 16bits represents stride with each 8 blocks（256B) low 16bits represent repeat stride
                DataCopy<T1, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T1*&)dstLocalInt), vregCastRes, blockStride, repeatStride, pregValidN);
            }
        }
    }
}

} // namespace AscendC
