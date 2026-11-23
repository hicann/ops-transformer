/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file vf_mul_sel_softmaxflashv2_cast_nz_dn_regbase_v2.h
 * \brief
 */

#ifndef MUL_SEL_SOFTMAXFLASHV2_CAST_NZ_DN_REGBASE_V2_H_
#define MUL_SEL_SOFTMAXFLASHV2_CAST_NZ_DN_REGBASE_V2_H_
#include "kernel_tensor.h"
#include "vf_store_exp.h"
namespace FaVectorApi {
using AscendC::LocalTensor;
using namespace AscendC;
using namespace Reg;

#define VMULSCVT false
#define DROPOUT false

template <typename T, typename T2, uint16_t ubN = 128>
__aicore__ inline void ProcessVec1DnNoUpdateRegbaseV2(const LocalTensor<T2>& dstTensor,
                                                      const LocalTensor<float>& expSumTensor,
                                                      const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor,
                                                      const LocalTensor<float>& expMaxTensor, const uint32_t m,
                                                      const uint32_t originN, const T scale, const T minValue,
                                                      float quantScale1)
{
    constexpr uint16_t repeatStride = 1;
    constexpr uint16_t blockStride = IsSameType<T2, int8_t>::value ? (ubN >> 1) : ubN;

    __ubuf__ T2* xExp = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ T2* xExp1 = xExp + (ubN * 8);
    __ubuf__ T* inputXLocalUB = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ float* expMaxFp32 = (__ubuf__ float*)expMaxTensor.GetPhyAddr();
    __ubuf__ float* newGlobalSum = (__ubuf__ float*)expSumTensor.GetPhyAddr();
    __ubuf__ T* newGlobalMax = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T* src0UbA = (__ubuf__ T*)inputXLocalUB;
    __ubuf__ T* src0UbB = (__ubuf__ T*)inputXLocalUB + m;
    __ubuf__ T* src0Ub1A = src0UbA + m * 2;
    __ubuf__ T* src0Ub1B = src0UbA + m * 3;
    __ubuf__ T* src0Ub2A = src0UbA + m * 4;
    __ubuf__ T* src0Ub2B = src0UbA + m * 5;
    __ubuf__ T* src0Ub3A = src0UbA + m * 6;
    __ubuf__ T* src0Ub3B = src0UbA + m * 7;

    __VEC_SCOPE__
    {
        Reg::RegTensor<T> vregXSumEven;
        Reg::RegTensor<T> vregXSumOdd;
        Reg::RegTensor<T> vregXSum1Even;
        Reg::RegTensor<T> vregXSum1Odd;
        Reg::RegTensor<T> vregXSum0;
        Reg::RegTensor<float> vregXSum0Even;
        Reg::RegTensor<float> vregXSum0Odd;
        Reg::RegTensor<T> vregXSum1;

        Reg::RegTensor<float> vregXExpF32Even;
        Reg::RegTensor<float> vregXExpF32Odd;

        Reg::RegTensor<T2> vregXExpEvenF16;
        Reg::RegTensor<T2> vregXExpOddF16;
        Reg::RegTensor<half> vregMulsResEven;
        Reg::RegTensor<half> vregMulsResOdd;
        Reg::RegTensor<half> vregMulsResEven1;
        Reg::RegTensor<half> vregMulsResOdd1;

        Reg::RegTensor<T> vregXExpEven;
        Reg::RegTensor<T> vregXExpOdd;
        Reg::RegTensor<T> vregXF32a;
        Reg::RegTensor<T> vregXF32b;
        Reg::RegTensor<T> vregXExpEven1;
        Reg::RegTensor<T> vregXExpOdd1;
        Reg::RegTensor<T2> vregXExpEvenF161;
        Reg::RegTensor<T2> vregXExpOddF161;

        Reg::RegTensor<T> vregXF321a;
        Reg::RegTensor<T> vregXF321b;
        Reg::RegTensor<T2> vregXExpF16Pack;
        Reg::RegTensor<T2> vregXExpF161Pack;
        Reg::RegTensor<T2> vregXExpF16Packa;
        Reg::RegTensor<T2> vregXExpF161Packa;
        Reg::MaskReg preg134;
        Reg::MaskReg preg135;
        Reg::MaskReg preg136;
        Reg::MaskReg pregAllB16;
        Reg::MaskReg pregAllB32;
        preg134 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        preg135 = Reg::CreateMask<half, Reg::MaskPattern::ALL>();
        uint32_t sreg92 = (uint32_t)(VECTOR_LENGTH_BYTES / sizeof(T2));
        preg136 = Reg::UpdateMask<T2>(sreg92);
        pregAllB16 = Reg::CreateMask<half, Reg::MaskPattern::ALL>();
        pregAllB32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::MaskReg preg108;
        Reg::RegTensor<T> src00a, src01a, src02a, src03a;
        Reg::RegTensor<T> src00b, src01b, src02b, src03b;
        Reg::RegTensor<T> max0a, max1a, max2a, max3a;
        Reg::RegTensor<T> max0b, max1b, max2b, max3b;
        Reg::RegTensor<T> vregMin;

        Reg::Duplicate<T, T>(max0a, minValue);
        Reg::Duplicate<T, T>(max0b, minValue);
        Reg::Duplicate<T, T>(max1a, minValue);
        Reg::Duplicate<T, T>(max1b, minValue);
        Reg::Duplicate<T, T>(max2a, minValue);
        Reg::Duplicate<T, T>(max2b, minValue);
        Reg::Duplicate<T, T>(max3a, minValue);
        Reg::Duplicate<T, T>(max3b, minValue);
        Reg::Duplicate<T, T>(vregMin, minValue);

        for (uint16_t i = originN; i < ubN; ++i) {
            Reg::DataCopy<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*)inputXLocalUB + i * m, vregMin, preg135);
        }
        mem_bar(VST_VLD);

        preg108 = pge_b16(PAT_ALL);
        for (uint16_t iterM = 0; iterM < uint16_t(ubN / 8); ++iterM) {
            auto aReg = AscendC::Reg::CreateAddrReg<T>(iterM, m * 8);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(src00a, src0UbA, aReg);
            Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max0a, max0a, src00a, preg108);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(src00b, src0UbB, aReg);
            Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max0b, max0b, src00b, preg108);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(src01a, src0Ub1A, aReg);
            Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max1a, max1a, src01a, preg108);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(src01b, src0Ub1B, aReg);
            Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max1b, max1b, src01b, preg108);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(src02a, src0Ub2A, aReg);
            Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max2a, max2a, src02a, preg108);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(src02b, src0Ub2B, aReg);
            Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max2b, max2b, src02b, preg108);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(src03a, src0Ub3A, aReg);
            Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max3a, max3a, src03a, preg108);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(src03b, src0Ub3B, aReg);
            Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max3b, max3b, src03b, preg108);
        }
        Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max0a, max0a, max1a, preg108);
        Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max0b, max0b, max1b, preg108);
        Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max2a, max2a, max3a, preg108);
        Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max2b, max2b, max3b, preg108);
        Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max0a, max0a, max2a, preg108);
        Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max0b, max0b, max2b, preg108);
        Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max0a, max0a, max0b, preg108);
        Reg::Muls<T, T, Reg::MaskMergeMode::ZEROING>(max0a, max0a, scale, preg108);

        Reg::DataCopy<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*)newGlobalMax, max0a, preg108);

        static constexpr Reg::CastTrait castTraitSum0 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                         Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTraitSum1 = {Reg::RegLayout::ONE, Reg::SatMode::NO_SAT,
                                                         Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        if constexpr (IsSameType<T2, int8_t>::value) {
            Reg::Duplicate<T, T>(vregXSumEven, 0);
            Reg::Duplicate<T, T>(vregXSumOdd, 0);
            Reg::Duplicate<T, T>(vregXSum1Even, 0);
            Reg::Duplicate<T, T>(vregXSum1Odd, 0);
            static constexpr Reg::CastTrait castTrait0 = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            for (uint16_t i0 = 0; i0 < uint16_t(ubN / 4); ++i0) {
                auto aReg = AscendC::Reg::CreateAddrReg<T>(i0, m);
                Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregXF32a, ((__ubuf__ T*)inputXLocalUB), aReg);
                Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregXF32b, ((__ubuf__ T*)inputXLocalUB + ubN * m / 2), aReg);
                Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregXF321a, ((__ubuf__ T*)inputXLocalUB + ubN * m / 4),
                                                           aReg);
                Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(
                    vregXF321b, ((__ubuf__ T*)inputXLocalUB + ubN * m / 2 + ubN * m / 4), aReg);

                Reg::Muls<T, T, Reg::MaskMergeMode::ZEROING>(vregXF32a, vregXF32a, scale, preg108);
                Reg::Muls<T, T, Reg::MaskMergeMode::ZEROING>(vregXF32b, vregXF32b, scale, preg108);
                Reg::FusedExpSub<T, T, Reg::RegLayout::ONE, Reg::MaskMergeMode::ZEROING>(vregXExpEven, vregXF32a, max0a,
                                                                                         preg134);
                Reg::FusedExpSub<T, T, Reg::RegLayout::ONE, Reg::MaskMergeMode::ZEROING>(vregXExpOdd, vregXF32b, max0a,
                                                                                         preg134);
                Reg::Muls<half, half, Reg::MaskMergeMode::ZEROING>(vregMulsResEven, vregXExpEven, (half)quantScale1,
                                                                   pregAllB16);
                Reg::Muls<half, half, Reg::MaskMergeMode::ZEROING>(vregMulsResOdd, vregXExpOdd, (half)quantScale1,
                                                                   pregAllB16);

                Reg::Muls<T, T, Reg::MaskMergeMode::ZEROING>(vregXF321a, vregXF321a, scale, preg108);
                Reg::Muls<T, T, Reg::MaskMergeMode::ZEROING>(vregXF321b, vregXF321b, scale, preg108);
                Reg::FusedExpSub<T, T, Reg::RegLayout::ONE, Reg::MaskMergeMode::ZEROING>(vregXExpEven1, vregXF321a,
                                                                                         max0a, preg134);
                Reg::FusedExpSub<T, T, Reg::RegLayout::ONE, Reg::MaskMergeMode::ZEROING>(vregXExpOdd1, vregXF321b,
                                                                                         max0a, preg134);
                Reg::Muls<half, half, Reg::MaskMergeMode::ZEROING>(vregMulsResEven1, vregXExpEven1, (half)quantScale1,
                                                                   pregAllB16);
                Reg::Muls<half, half, Reg::MaskMergeMode::ZEROING>(vregMulsResOdd1, vregXExpOdd1, (half)quantScale1,
                                                                   pregAllB16);

                Reg::Cast<T2, T, castTrait0>(vregXExpEvenF16, vregMulsResEven, preg135);
                Reg::Cast<T2, T, castTrait0>(vregXExpOddF16, vregMulsResOdd, preg135);
                Reg::DeInterleave<uint8_t>(
                    (Reg::RegTensor<uint8_t>&)vregXExpF16Pack, (Reg::RegTensor<uint8_t>&)vregXExpF16Packa,
                    (Reg::RegTensor<uint8_t>&)vregXExpEvenF16, (Reg::RegTensor<uint8_t>&)vregXExpOddF16);

                Reg::Cast<T2, T, castTrait0>(vregXExpEvenF161, vregMulsResEven1, preg135);
                Reg::Cast<T2, T, castTrait0>(vregXExpOddF161, vregMulsResOdd1, preg135);
                Reg::DeInterleave<uint8_t>(
                    (Reg::RegTensor<uint8_t>&)vregXExpF161Pack, (Reg::RegTensor<uint8_t>&)vregXExpF161Packa,
                    (Reg::RegTensor<uint8_t>&)vregXExpEvenF161, (Reg::RegTensor<uint8_t>&)vregXExpOddF161);

                Reg::DataCopy<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                    xExp, vregXExpF16Pack, blockStride, repeatStride, preg136);
                Reg::DataCopy<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                    xExp1, vregXExpF161Pack, blockStride, repeatStride, preg136);

                Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vregXSumEven, vregXExpEven, vregXSumEven, preg134);
                Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vregXSumOdd, vregXExpOdd, vregXSumOdd, preg134);
                Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vregXSum1Even, vregXExpEven1, vregXSum1Even, preg134);
                Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vregXSum1Odd, vregXExpOdd1, vregXSum1Odd, preg134);
                Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vregXSum0, vregXSumOdd, vregXSumEven, preg134);
                Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vregXSum1, vregXSum1Odd, vregXSum1Even, preg134);
                Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vregXSum0, vregXSum0, vregXSum1, preg134);

                Reg::Cast<float, half, castTraitSum0>(vregXSum0Even, vregXSum0, pregAllB16);
                Reg::Cast<float, half, castTraitSum1>(vregXSum0Odd, vregXSum0, pregAllB16);
            }
        } else {
            Reg::Duplicate<float, float>(vregXSum0Even, 0.0f);
            Reg::Duplicate<float, float>(vregXSum0Odd, 0.0f);
            for (uint16_t i0 = 0; i0 < uint16_t(ubN / 2); ++i0) {
                auto aReg = AscendC::Reg::CreateAddrReg<T>(i0, m);
                Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregXF32a, ((__ubuf__ T*)inputXLocalUB), aReg);
                Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregXF32b, ((__ubuf__ T*)inputXLocalUB + ubN * m / 2), aReg);

                Reg::Muls<T, T, Reg::MaskMergeMode::ZEROING>(vregXF32a, vregXF32a, scale, preg108);
                Reg::Muls<T, T, Reg::MaskMergeMode::ZEROING>(vregXF32b, vregXF32b, scale, preg108);

                Reg::FusedExpSub<T, T, Reg::RegLayout::ONE, Reg::MaskMergeMode::ZEROING>(vregXExpEven, vregXF32a, max0a,
                                                                                         preg134);
                Reg::FusedExpSub<T, T, Reg::RegLayout::ONE, Reg::MaskMergeMode::ZEROING>(vregXExpOdd, vregXF32b, max0a,
                                                                                         preg134);

                StoreExp(xExp, vregXExpEven, blockStride, repeatStride, preg136, quantScale1);
                StoreExp(xExp1, vregXExpOdd, blockStride, repeatStride, preg136, quantScale1);

                Reg::Cast<float, T, castTraitSum0>(vregXExpF32Even, vregXExpEven, pregAllB16);
                Reg::Cast<float, T, castTraitSum1>(vregXExpF32Odd, vregXExpEven, pregAllB16);
                Reg::Add<float, Reg::MaskMergeMode::ZEROING>(vregXSum0Even, vregXExpF32Even, vregXSum0Even, pregAllB32);
                Reg::Add<float, Reg::MaskMergeMode::ZEROING>(vregXSum0Odd, vregXExpF32Odd, vregXSum0Odd, pregAllB32);

                Reg::Cast<float, T, castTraitSum0>(vregXExpF32Even, vregXExpOdd, pregAllB16);
                Reg::Cast<float, T, castTraitSum1>(vregXExpF32Odd, vregXExpOdd, pregAllB16);
                Reg::Add<float, Reg::MaskMergeMode::ZEROING>(vregXSum0Even, vregXExpF32Even, vregXSum0Even, pregAllB32);
                Reg::Add<float, Reg::MaskMergeMode::ZEROING>(vregXSum0Odd, vregXExpF32Odd, vregXSum0Odd, pregAllB32);
            }
        }

        Reg::DataCopy<float, Reg::StoreDist::DIST_INTLV_B32>(newGlobalSum, vregXSum0Even, vregXSum0Odd, pregAllB32);
    }
}

template <typename T, typename T2, uint16_t ubN = 128>
__aicore__ inline void ProcessVec1DnUpdateRegbaseV2(const LocalTensor<T2>& dstTensor,
                                                    const LocalTensor<float>& expSumTensor,
                                                    const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor,
                                                    const LocalTensor<float>& expMaxTensor, const uint32_t m,
                                                    const uint32_t originN, const T scale, const T minValue,
                                                    float quantScale1)
{
    constexpr uint16_t repeatStride = 1;
    constexpr uint16_t blockStride = IsSameType<T2, int8_t>::value ? (ubN >> 1) : ubN;

    __ubuf__ T2* xExp = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ T2* xExp1 = xExp + (ubN * 8);
    __ubuf__ T* inputXLocalUB = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ float* expMaxFp32 = (__ubuf__ float*)expMaxTensor.GetPhyAddr();
    __ubuf__ float* newGlobalSum = (__ubuf__ float*)expSumTensor.GetPhyAddr();
    __ubuf__ T* newGlobalMax = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T* src0UbA = (__ubuf__ T*)inputXLocalUB;
    __ubuf__ T* src0UbB = (__ubuf__ T*)inputXLocalUB + m;
    __ubuf__ T* src0Ub1A = src0UbA + m * 2;
    __ubuf__ T* src0Ub1B = src0UbA + m * 3;
    __ubuf__ T* src0Ub2A = src0UbA + m * 4;
    __ubuf__ T* src0Ub2B = src0UbA + m * 5;
    __ubuf__ T* src0Ub3A = src0UbA + m * 6;
    __ubuf__ T* src0Ub3B = src0UbA + m * 7;

    __VEC_SCOPE__
    {
        Reg::RegTensor<T> vregXSumEven;
        Reg::RegTensor<T> vregXSumOdd;
        Reg::RegTensor<T> vregXSum1Even;
        Reg::RegTensor<T> vregXSum1Odd;
        Reg::RegTensor<T> vregXSum0;
        Reg::RegTensor<float> vregXSum0Even;
        Reg::RegTensor<float> vregXSum0Odd;
        Reg::RegTensor<T> vregXSum1;

        Reg::RegTensor<float> vregXExpF32Even;
        Reg::RegTensor<float> vregXExpF32Odd;

        Reg::RegTensor<T2> vregXExpEvenF16;
        Reg::RegTensor<T2> vregXExpOddF16;

        Reg::RegTensor<T> vregXExpEven;
        Reg::RegTensor<T> vregXExpOdd;
        Reg::RegTensor<T> vregXF32a;
        Reg::RegTensor<T> vregXF32b;
        Reg::RegTensor<T> vregXExpEven1;
        Reg::RegTensor<T> vregXExpOdd1;
        Reg::RegTensor<T2> vregXExpEvenF161;
        Reg::RegTensor<T2> vregXExpOddF161;
        Reg::RegTensor<half> vregMulsResEven;
        Reg::RegTensor<half> vregMulsResOdd;
        Reg::RegTensor<half> vregMulsResEven1;
        Reg::RegTensor<half> vregMulsResOdd1;

        Reg::RegTensor<T> vregXF321a;
        Reg::RegTensor<T> vregXF321b;
        Reg::RegTensor<float> vregXMaxF32Even;
        Reg::RegTensor<float> vregXMaxF32Odd;
        Reg::RegTensor<T> vregXMaxF16;
        Reg::RegTensor<T2> vregXExpF16Pack;
        Reg::RegTensor<T2> vregXExpF161Pack;
        Reg::RegTensor<T2> vregXExpF16Packa;
        Reg::RegTensor<T2> vregXExpF161Packa;

        Reg::MaskReg preg134;
        Reg::MaskReg preg135;
        Reg::MaskReg preg136;
        Reg::MaskReg pregAllB16;
        Reg::MaskReg pregAllB32;
        preg134 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        preg135 = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
        uint32_t sreg92 = (uint32_t)(VECTOR_LENGTH_BYTES / sizeof(T2));
        preg136 = Reg::UpdateMask<T2>(sreg92);
        pregAllB16 = Reg::CreateMask<half, Reg::MaskPattern::ALL>();
        pregAllB32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::MaskReg preg108;
        Reg::RegTensor<T> src00a, src01a, src02a, src03a;
        Reg::RegTensor<T> src00b, src01b, src02b, src03b;
        Reg::RegTensor<T> max0a, max1a, max2a, max3a;
        Reg::RegTensor<T> max0b, max1b, max2b, max3b;
        Reg::RegTensor<T> vregMin;

        Reg::Duplicate<T, T>(max0a, minValue);
        Reg::Duplicate<T, T>(max0b, minValue);
        Reg::Duplicate<T, T>(max1a, minValue);
        Reg::Duplicate<T, T>(max1b, minValue);
        Reg::Duplicate<T, T>(max2a, minValue);
        Reg::Duplicate<T, T>(max2b, minValue);
        Reg::Duplicate<T, T>(max3a, minValue);
        Reg::Duplicate<T, T>(max3b, minValue);
        Reg::Duplicate<T, T>(vregMin, minValue);
        for (uint16_t i = originN; i < ubN; ++i) {
            Reg::DataCopy<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*)inputXLocalUB + i * m, vregMin, preg135);
        }
        mem_bar(VST_VLD);

        preg108 = pge_b16(PAT_ALL);
        for (uint16_t iterM = 0; iterM < uint16_t(ubN / 8); ++iterM) {
            auto aReg = AscendC::Reg::CreateAddrReg<T>(iterM, m * 8);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(src00a, src0UbA, aReg);
            Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max0a, max0a, src00a, preg108);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(src00b, src0UbB, aReg);
            Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max0b, max0b, src00b, preg108);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(src01a, src0Ub1A, aReg);
            Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max1a, max1a, src01a, preg108);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(src01b, src0Ub1B, aReg);
            Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max1b, max1b, src01b, preg108);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(src02a, src0Ub2A, aReg);
            Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max2a, max2a, src02a, preg108);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(src02b, src0Ub2B, aReg);
            Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max2b, max2b, src02b, preg108);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(src03a, src0Ub3A, aReg);
            Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max3a, max3a, src03a, preg108);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(src03b, src0Ub3B, aReg);
            Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max3b, max3b, src03b, preg108);
        }
        Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregXMaxF16, ((__ubuf__ T*)newGlobalMax));
        Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max0a, max0a, max1a, preg108);
        Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max0b, max0b, max1b, preg108);
        Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max2a, max2a, max3a, preg108);
        Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max2b, max2b, max3b, preg108);
        Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max0a, max0a, max2a, preg108);
        Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max0b, max0b, max2b, preg108);
        Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max0a, max0a, max0b, preg108);
        Reg::Muls<T, T, Reg::MaskMergeMode::ZEROING>(max0a, max0a, scale, preg108);
        Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max0a, max0a, vregXMaxF16, preg108);

        Reg::FusedExpSub<T, T, Reg::RegLayout::ONE, Reg::MaskMergeMode::ZEROING>(vregXMaxF16, vregXMaxF16, max0a,
                                                                                 preg134);
        Reg::DataCopy<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*)newGlobalMax, max0a, preg108);
        static constexpr Reg::CastTrait castTraitSum0 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                         Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTraitSum1 = {Reg::RegLayout::ONE, Reg::SatMode::NO_SAT,
                                                         Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        Reg::Cast<float, half, castTraitSum0>(vregXMaxF32Even, vregXMaxF16, pregAllB16);
        Reg::Cast<float, half, castTraitSum1>(vregXMaxF32Odd, vregXMaxF16, pregAllB16);
        Reg::DataCopy<float, Reg::StoreDist::DIST_INTLV_B32>(expMaxFp32, vregXMaxF32Even, vregXMaxF32Odd, pregAllB32);
        if constexpr (IsSameType<T2, int8_t>::value) {
            Reg::Duplicate<T, T>(vregXSumEven, 0);
            Reg::Duplicate<T, T>(vregXSumOdd, 0);
            Reg::Duplicate<T, T>(vregXSum1Even, 0);
            Reg::Duplicate<T, T>(vregXSum1Odd, 0);
            static constexpr Reg::CastTrait castTrait0 = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            for (uint16_t i0 = 0; i0 < uint16_t(ubN / 4); ++i0) {
                auto aReg = AscendC::Reg::CreateAddrReg<T>(i0, m);
                Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregXF32a, ((__ubuf__ T*)inputXLocalUB), aReg);
                Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregXF32b, ((__ubuf__ T*)inputXLocalUB + ubN * m / 2), aReg);
                Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregXF321a, ((__ubuf__ T*)inputXLocalUB + ubN * m / 4),
                                                           aReg);
                Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(
                    vregXF321b, ((__ubuf__ T*)inputXLocalUB + ubN * m / 2 + ubN * m / 4), aReg);

                Reg::Muls<T, T, Reg::MaskMergeMode::ZEROING>(vregXF32a, vregXF32a, scale, preg108);
                Reg::Muls<T, T, Reg::MaskMergeMode::ZEROING>(vregXF32b, vregXF32b, scale, preg108);
                Reg::FusedExpSub<T, T, Reg::RegLayout::ONE, Reg::MaskMergeMode::ZEROING>(vregXExpEven, vregXF32a, max0a,
                                                                                         preg134);
                Reg::FusedExpSub<T, T, Reg::RegLayout::ONE, Reg::MaskMergeMode::ZEROING>(vregXExpOdd, vregXF32b, max0a,
                                                                                         preg134);
                Reg::Muls<half, half, Reg::MaskMergeMode::ZEROING>(vregMulsResEven, vregXExpEven, (half)quantScale1,
                                                                   pregAllB16);
                Reg::Muls<half, half, Reg::MaskMergeMode::ZEROING>(vregMulsResOdd, vregXExpOdd, (half)quantScale1,
                                                                   pregAllB16);

                Reg::Muls<T, T, Reg::MaskMergeMode::ZEROING>(vregXF321a, vregXF321a, scale, preg108);
                Reg::Muls<T, T, Reg::MaskMergeMode::ZEROING>(vregXF321b, vregXF321b, scale, preg108);
                Reg::FusedExpSub<T, T, Reg::RegLayout::ONE, Reg::MaskMergeMode::ZEROING>(vregXExpEven1, vregXF321a,
                                                                                         max0a, preg134);
                Reg::FusedExpSub<T, T, Reg::RegLayout::ONE, Reg::MaskMergeMode::ZEROING>(vregXExpOdd1, vregXF321b,
                                                                                         max0a, preg134);
                Reg::Muls<half, half, Reg::MaskMergeMode::ZEROING>(vregMulsResEven1, vregXExpEven1, (half)quantScale1,
                                                                   pregAllB16);
                Reg::Muls<half, half, Reg::MaskMergeMode::ZEROING>(vregMulsResOdd1, vregXExpOdd1, (half)quantScale1,
                                                                   pregAllB16);

                Reg::Cast<T2, T, castTrait0>(vregXExpEvenF16, vregMulsResEven, preg135);
                Reg::Cast<T2, T, castTrait0>(vregXExpOddF16, vregMulsResOdd, preg135);
                Reg::DeInterleave<uint8_t>(
                    (Reg::RegTensor<uint8_t>&)vregXExpF16Pack, (Reg::RegTensor<uint8_t>&)vregXExpF16Packa,
                    (Reg::RegTensor<uint8_t>&)vregXExpEvenF16, (Reg::RegTensor<uint8_t>&)vregXExpOddF16);

                Reg::Cast<T2, T, castTrait0>(vregXExpEvenF161, vregMulsResEven1, preg135);
                Reg::Cast<T2, T, castTrait0>(vregXExpOddF161, vregMulsResOdd1, preg135);
                Reg::DeInterleave<uint8_t>(
                    (Reg::RegTensor<uint8_t>&)vregXExpF161Pack, (Reg::RegTensor<uint8_t>&)vregXExpF161Packa,
                    (Reg::RegTensor<uint8_t>&)vregXExpEvenF161, (Reg::RegTensor<uint8_t>&)vregXExpOddF161);

                Reg::DataCopy<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                    xExp, vregXExpF16Pack, blockStride, repeatStride, preg136);
                Reg::DataCopy<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                    xExp1, vregXExpF161Pack, blockStride, repeatStride, preg136);

                Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vregXSumEven, vregXExpEven, vregXSumEven, preg134);
                Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vregXSumOdd, vregXExpOdd, vregXSumOdd, preg134);
                Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vregXSum1Even, vregXExpEven1, vregXSum1Even, preg134);
                Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vregXSum1Odd, vregXExpOdd1, vregXSum1Odd, preg134);

                Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vregXSum0, vregXSumOdd, vregXSumEven, preg134);
                Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vregXSum1, vregXSum1Odd, vregXSum1Even, preg134);
                Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vregXSum0, vregXSum0, vregXSum1, preg134);
                Reg::Cast<float, half, castTraitSum0>(vregXSum0Even, vregXSum0, pregAllB16);
                Reg::Cast<float, half, castTraitSum1>(vregXSum0Odd, vregXSum0, pregAllB16);
            }
        } else {
            Reg::Duplicate<float, float>(vregXSum0Even, 0.0f);
            Reg::Duplicate<float, float>(vregXSum0Odd, 0.0f);
            for (uint16_t i0 = 0; i0 < uint16_t(ubN / 2); ++i0) {
                auto aReg = AscendC::Reg::CreateAddrReg<T>(i0, m);
                Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregXF32a, ((__ubuf__ T*)inputXLocalUB), aReg);
                Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregXF32b, ((__ubuf__ T*)inputXLocalUB + ubN * m / 2), aReg);

                Reg::Muls<T, T, Reg::MaskMergeMode::ZEROING>(vregXF32a, vregXF32a, scale, preg108);
                Reg::Muls<T, T, Reg::MaskMergeMode::ZEROING>(vregXF32b, vregXF32b, scale, preg108);
                Reg::FusedExpSub<T, T, Reg::RegLayout::ONE, Reg::MaskMergeMode::ZEROING>(vregXExpEven, vregXF32a, max0a,
                                                                                         preg134);
                Reg::FusedExpSub<T, T, Reg::RegLayout::ONE, Reg::MaskMergeMode::ZEROING>(vregXExpOdd, vregXF32b, max0a,
                                                                                         preg134);

                StoreExp(xExp, vregXExpEven, blockStride, repeatStride, preg136, quantScale1);
                StoreExp(xExp1, vregXExpOdd, blockStride, repeatStride, preg136, quantScale1);

                Reg::Cast<float, T, castTraitSum0>(vregXExpF32Even, vregXExpEven, pregAllB16);
                Reg::Cast<float, T, castTraitSum1>(vregXExpF32Odd, vregXExpEven, pregAllB16);
                Reg::Add<float, Reg::MaskMergeMode::ZEROING>(vregXSum0Even, vregXExpF32Even, vregXSum0Even, pregAllB32);
                Reg::Add<float, Reg::MaskMergeMode::ZEROING>(vregXSum0Odd, vregXExpF32Odd, vregXSum0Odd, pregAllB32);

                Reg::Cast<float, T, castTraitSum0>(vregXExpF32Even, vregXExpOdd, pregAllB16);
                Reg::Cast<float, T, castTraitSum1>(vregXExpF32Odd, vregXExpOdd, pregAllB16);
                Reg::Add<float, Reg::MaskMergeMode::ZEROING>(vregXSum0Even, vregXExpF32Even, vregXSum0Even, pregAllB32);
                Reg::Add<float, Reg::MaskMergeMode::ZEROING>(vregXSum0Odd, vregXExpF32Odd, vregXSum0Odd, pregAllB32);
            }
        }
        Reg::RegTensor<float> vregL0;
        Reg::RegTensor<float> vregL01;
        Reg::DataCopy<float, Reg::LoadDist::DIST_DINTLV_B32>(vregL0, vregL01, ((__ubuf__ float*)newGlobalSum));
        Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(vregL0, vregXMaxF32Even, vregL0, pregAllB32);
        Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(vregL01, vregXMaxF32Odd, vregL01, pregAllB32);
        Reg::Add<float, Reg::MaskMergeMode::ZEROING>(vregL0, vregL0, vregXSum0Even, pregAllB32);
        Reg::Add<float, Reg::MaskMergeMode::ZEROING>(vregL01, vregL01, vregXSum0Odd, pregAllB32);
        Reg::DataCopy<float, Reg::StoreDist::DIST_INTLV_B32>(newGlobalSum, vregL0, vregL01, pregAllB32);
    }
}

/*
 * @ingroup ProcessVec1Vf
 * @brief compute max = reducemax, exp(x-max)/sum(exp(x-max))
 * @param [out] dstTensor, output LocalTensor
 * @param [out] expSumTensor, out sum(exp(x-max)) of last axis
 * @param [out] maxTensor, out max value of last axis
 * @param [in] srcTensor, input LocalTensor
 * @param [out] expMaxTensor, output expmax LocalTensor
 * @param [in] sharedTmpBuffer, input local temporary Tensor
 * @param [in] m, input rows
 * @param [in] n, input colums, should be 256 bytes aligned, the value is originN aligned to 64
 * @param [in] originN, input origin colums, support range: 0 < originN <= 128
 * @param [in] scale, scale value
 * @param [in] minValue, minimum value
 * @param [in] isUpdate, enable flash mode
 * @param [in] oriNRange, originN range
 */

template <typename T, typename T2, bool isUpdate = false, uint16_t ubN = 256>
__aicore__ inline void ProcessVec1VfDnRegbaseV2(const LocalTensor<T2>& dstTensor,
                                                const LocalTensor<float>& expSumTensor, const LocalTensor<T>& maxTensor,
                                                const LocalTensor<T>& srcTensor, const LocalTensor<float>& expMaxTensor,
                                                const uint32_t m, const uint32_t originN, const T scale,
                                                const T minValue, float quantScale1)
{
    if constexpr (!isUpdate) {
        ProcessVec1DnNoUpdateRegbaseV2<T, T2, ubN>(dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, m,
                                                   originN, scale, minValue, quantScale1);
    } else {
        ProcessVec1DnUpdateRegbaseV2<T, T2, ubN>(dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, m,
                                                 originN, scale, minValue, quantScale1);
    }
}
} // namespace FaVectorApi
#endif // MUL_SEL_SOFTMAXFLASHV2_CAST_NZ_DN_REGBASE_V2_H_
