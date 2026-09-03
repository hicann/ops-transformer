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
 * \file vf_flashupdate_new_regbase_v2.h
 * \brief
 */
#ifndef MY_FLASH_UPDATE_NEW_REGBASE_V2_INTERFACE_H
#define MY_FLASH_UPDATE_NEW_REGBASE_V2_INTERFACE_H

#include "kernel_tensor.h"

namespace FaVectorApi {
/* **************************************************************************************************
 * FlashUpdateV510
 * [s1, k] = [128, 128], fp16
 * ************************************************************************************************* */

/*
 * @ingroup FlashUpdateNoTailV510_VF
 * @brief compute, dstTensor = (preTensor + curTensor) * expMaxTensor
 * @param [out] dstTensor, output LocalTensor
 * @param [in] curTensor, input LocalTensor
 * @param [in] preTensor, input LocalTensor
 * @param [in] expMaxTensor, input LocalTensor
 * @param [in] m, input rows
 * @param [in] d, input colums, should be 64 aligned
 */
template <typename T, typename OUTPUT_T, typename MMOUTPUT_T, uint32_t dSize = 0>
__aicore__ inline void FlashUpdateNoTailV510_VF(const LocalTensor<T> &dstTensor, const LocalTensor<T> &curTensor,
                                                const LocalTensor<T> &preTensor, const LocalTensor<float> &expMaxTensor,
                                                const uint16_t m, const uint16_t d)
{
    __ubuf__ T *dstUb = (__ubuf__ T *)dstTensor.GetPhyAddr();
    __ubuf__ T *curUb = (__ubuf__ T *)curTensor.GetPhyAddr();
    __ubuf__ T *preUb = (__ubuf__ T *)preTensor.GetPhyAddr();
    __ubuf__ float *expMaxUb = (__ubuf__ float *)expMaxTensor.GetPhyAddr();

    constexpr uint16_t reduceSize = 1;

    __VEC_SCOPE__
    {
        Reg::RegTensor<float> vSrcRegMax;
        Reg::RegTensor<T> vSrcRegMaxB16;
        Reg::RegTensor<T> vSrcRegMaxB16Even;
        Reg::RegTensor<T> vSrcRegMaxB16Odd;
        Reg::RegTensor<T> vSrcRegPre;
        Reg::RegTensor<T> vSrcRegCur;
        Reg::RegTensor<T> vSrcRegMul;
        Reg::RegTensor<T> vDstRegAdd;

        Reg::RegTensor<MMOUTPUT_T> vregSrc;

        Reg::MaskReg maskRegAllB32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskRegAllB16 = Reg::CreateMask<T, Reg::MaskPattern::ALL>();

        Reg::MaskReg preg_d;
        uint32_t sreg_d = dSize;
        preg_d = Reg::UpdateMask<T>(sreg_d);

        for (uint16_t i = 0; i < m; ++i) {
            static constexpr Reg::CastTrait castTrait0 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            static constexpr Reg::CastTrait castTrait1 = {Reg::RegLayout::ONE, Reg::SatMode::NO_SAT,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(vSrcRegMax, expMaxUb + i * reduceSize);
            Reg::Cast<T, float, castTrait0>(vSrcRegMaxB16Even, vSrcRegMax, maskRegAllB32);
            Reg::Cast<T, float, castTrait1>(vSrcRegMaxB16Odd, vSrcRegMax, maskRegAllB32);
            Reg::Or<uint16_t, Reg::MaskMergeMode::ZEROING>((Reg::RegTensor<uint16_t> &)vSrcRegMaxB16,
                                                           (Reg::RegTensor<uint16_t> &)vSrcRegMaxB16Even,
                                                           (Reg::RegTensor<uint16_t> &)vSrcRegMaxB16Odd, preg_d);

            // high performance only support d=128
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vSrcRegPre, preUb + i * dSize);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vSrcRegCur, curUb + i * dSize);

            Reg::Mul<T, Reg::MaskMergeMode::ZEROING>(vSrcRegMul, vSrcRegMaxB16, vSrcRegPre, preg_d);
            Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vDstRegAdd, vSrcRegMul, vSrcRegCur, preg_d);
            Reg::DataCopy<OUTPUT_T, Reg::StoreDist::DIST_NORM_B16>(dstUb + i * dSize, vDstRegAdd, preg_d);
        }
    }
}

/*
 * @ingroup FlashUpdateLastNoTailV510_VF
 * @brief compute, dstTensor = (preTensor  + curTensor ) / expSumTensor
 * @param [out] dstTensor, output LocalTensor
 * @param [in] curTensor, input LocalTensor
 * @param [in] preTensor, input LocalTensor
 * @param [in] expMaxTensor, input LocalTensor
 * @param [in] expSumTensor, input LocalTensor
 * @param [in] m, input rows
 * @param [in] d, input colums, should be 32 bytes aligned
 */
template <typename T, typename OUTPUT_T, typename MMOUTPUT_T, uint32_t dSize = 0>
__aicore__ inline void FlashUpdateLastNoTailV510_VF(const LocalTensor<T> &dstTensor, const LocalTensor<T> &curTensor,
                                                    const LocalTensor<T> &preTensor,
                                                    const LocalTensor<float> &expMaxTensor,
                                                    const LocalTensor<float> &expSumTensor, const uint16_t m,
                                                    const uint16_t d)
{
    __ubuf__ T *dstUb = (__ubuf__ T *)dstTensor.GetPhyAddr();
    __ubuf__ T *curUb = (__ubuf__ T *)curTensor.GetPhyAddr();
    __ubuf__ T *preUb = (__ubuf__ T *)preTensor.GetPhyAddr();
    __ubuf__ float *expMaxUb = (__ubuf__ float *)expMaxTensor.GetPhyAddr();
    __ubuf__ float *expSumUb = (__ubuf__ float *)expSumTensor.GetPhyAddr();

    constexpr uint16_t reduceSize = 1;

    __VEC_SCOPE__
    {
        Reg::RegTensor<float> vSrcRegSum;
        Reg::RegTensor<float> vSrcRegMax;
        Reg::RegTensor<T> vSrcRegSumB16;
        Reg::RegTensor<T> vSrcRegSumB16Even;
        Reg::RegTensor<T> vSrcRegSumB16Odd;
        Reg::RegTensor<T> vSrcRegMaxB16;
        Reg::RegTensor<T> vSrcRegMaxB16Even;
        Reg::RegTensor<T> vSrcRegMaxB16Odd;
        Reg::RegTensor<T> vSrcRegPre;
        Reg::RegTensor<T> vSrcRegCur;
        Reg::RegTensor<T> vSrcRegMul;
        Reg::RegTensor<T> vSrcRegAdd;
        Reg::RegTensor<T> vDstRegDiv;

        Reg::RegTensor<MMOUTPUT_T> vregSrc;

        // false: normal mode; true: higher precision mode
        Reg::MaskReg preg_d;
        uint32_t sreg_d = dSize;
        preg_d = Reg::UpdateMask<T>(sreg_d);

        static constexpr Reg::DivSpecificMode mode = {Reg::MaskMergeMode::ZEROING, false};
        Reg::MaskReg maskRegAllB32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskRegAllB16 = Reg::CreateMask<T, Reg::MaskPattern::ALL>();

        for (uint16_t i = 0; i < m; ++i) {
            static constexpr Reg::CastTrait castTrait0 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            static constexpr Reg::CastTrait castTrait1 = {Reg::RegLayout::ONE, Reg::SatMode::NO_SAT,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(vSrcRegMax, expMaxUb + i * reduceSize);
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(vSrcRegSum, expSumUb + i * reduceSize);
            Reg::Cast<T, float, castTrait0>(vSrcRegMaxB16Even, vSrcRegMax, maskRegAllB32);
            Reg::Cast<T, float, castTrait1>(vSrcRegMaxB16Odd, vSrcRegMax, maskRegAllB32);
            Reg::Cast<T, float, castTrait0>(vSrcRegSumB16Even, vSrcRegSum, maskRegAllB32);
            Reg::Cast<T, float, castTrait1>(vSrcRegSumB16Odd, vSrcRegSum, maskRegAllB32);
            Reg::Or<uint16_t, Reg::MaskMergeMode::ZEROING>((Reg::RegTensor<uint16_t> &)vSrcRegMaxB16,
                                                           (Reg::RegTensor<uint16_t> &)vSrcRegMaxB16Even,
                                                           (Reg::RegTensor<uint16_t> &)vSrcRegMaxB16Odd, preg_d);
            Reg::Or<uint16_t, Reg::MaskMergeMode::ZEROING>((Reg::RegTensor<uint16_t> &)vSrcRegSumB16,
                                                           (Reg::RegTensor<uint16_t> &)vSrcRegSumB16Even,
                                                           (Reg::RegTensor<uint16_t> &)vSrcRegSumB16Odd, preg_d);

            // high performance only support d=128
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vSrcRegPre, preUb + i * dSize);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vSrcRegCur, curUb + i * dSize);

            Reg::Mul<T, Reg::MaskMergeMode::ZEROING>(vSrcRegMul, vSrcRegMaxB16, vSrcRegPre, preg_d);
            Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vSrcRegAdd, vSrcRegMul, vSrcRegCur, preg_d);
            Reg::Div<T, &mode>(vDstRegDiv, vSrcRegAdd, vSrcRegSumB16, preg_d);
            Reg::DataCopy<OUTPUT_T, Reg::StoreDist::DIST_NORM_B16>(dstUb + i * dSize, vDstRegDiv, preg_d);
        }
    }
}

/*
 * @ingroup FlashUpdateDivNoTailV510_VF
 * @brief compute, dstTensor = preTensor / expSumTensor
 * @param [out] dstTensor, output LocalTensor
 * @param [in] preTensor, input LocalTensor
 * @param [in] expSumTensor, input LocalTensor
 * @param [in] m, input rows
 * @param [in] d, input colums, should be 32 bytes aligned
 */
template <typename T, typename OUTPUT_T, typename MMOUTPUT_T, uint32_t dSize = 0>
__aicore__ inline void FlashUpdateDivNoTailV510_VF(const LocalTensor<T> &dstTensor, const LocalTensor<T> &preTensor,
                                                   const LocalTensor<float> &expSumTensor, const uint16_t m,
                                                   const uint16_t d)
{
    __ubuf__ T *dstUb = (__ubuf__ T *)dstTensor.GetPhyAddr();
    __ubuf__ T *preUb = (__ubuf__ T *)preTensor.GetPhyAddr();
    __ubuf__ float *expSumUb = (__ubuf__ float *)expSumTensor.GetPhyAddr();

    constexpr uint16_t reduceSize = 1;

    __VEC_SCOPE__
    {
        Reg::RegTensor<float> vSrcRegSum;
        Reg::RegTensor<T> vSrcRegSumB16Even;
        Reg::RegTensor<T> vSrcRegSumB16Odd;
        Reg::RegTensor<T> vSrcRegSumB16;
        Reg::RegTensor<T> vSrcRegPre;
        Reg::RegTensor<T> vDstRegDiv;

        Reg::RegTensor<MMOUTPUT_T> vregSrc;

        Reg::MaskReg preg_d;
        uint32_t sreg_d = dSize;
        preg_d = Reg::UpdateMask<T>(sreg_d);

        // false: normal mode; true: higher precision mode
        static constexpr Reg::DivSpecificMode mode = {Reg::MaskMergeMode::ZEROING, false};
        Reg::MaskReg maskRegAllB32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskRegAllB16 = Reg::CreateMask<T, Reg::MaskPattern::ALL>();

        for (uint16_t i = 0; i < m; ++i) {
            static constexpr Reg::CastTrait castTrait0 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            static constexpr Reg::CastTrait castTrait1 = {Reg::RegLayout::ONE, Reg::SatMode::NO_SAT,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(vSrcRegSum, expSumUb + i * reduceSize);
            Reg::Cast<T, float, castTrait0>(vSrcRegSumB16Even, vSrcRegSum, maskRegAllB32);
            Reg::Cast<T, float, castTrait1>(vSrcRegSumB16Odd, vSrcRegSum, maskRegAllB32);
            Reg::Or<uint16_t, Reg::MaskMergeMode::ZEROING>((Reg::RegTensor<uint16_t> &)vSrcRegSumB16,
                                                           (Reg::RegTensor<uint16_t> &)vSrcRegSumB16Even,
                                                           (Reg::RegTensor<uint16_t> &)vSrcRegSumB16Odd, preg_d);
            // high performance only support d=128
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vSrcRegPre, preUb + i * dSize);
            Reg::Div<T, &mode>(vDstRegDiv, vSrcRegPre, vSrcRegSumB16, preg_d);
            Reg::DataCopy<OUTPUT_T, Reg::StoreDist::DIST_NORM_B16>(dstUb + i * dSize, vDstRegDiv, preg_d);
        }
    }
}

template <typename T, typename OUTPUT_T, typename MMOUTPUT_T, uint32_t dSize = 0>
__aicore__ inline void FlashUpdateV510(const LocalTensor<T> &dstTensor, const LocalTensor<T> &curTensor,
                                       const LocalTensor<T> &preTensor, const LocalTensor<float> &expMaxTensor,
                                       const uint16_t m, const uint16_t d)
{
    static_assert(IsSameType<T, half>::value, "VF FlashUpdate High Performance, T must be half");
    // only support d=128 now, no tail
    FlashUpdateNoTailV510_VF<T, OUTPUT_T, MMOUTPUT_T, dSize>(dstTensor, curTensor, preTensor, expMaxTensor, m, d);
}

template <typename T, typename OUTPUT_T, typename MMOUTPUT_T, uint32_t dSize = 0>
__aicore__ inline void FlashUpdateLastV510(const LocalTensor<T> &dstTensor, const LocalTensor<T> &curTensor,
                                           const LocalTensor<T> &preTensor, const LocalTensor<float> &expMaxTensor,
                                           const LocalTensor<float> &expSumTensor, const uint16_t m, const uint16_t d)
{
    static_assert(IsSameType<T, half>::value, "VF FlashUpdate High Performance, T must be half");
    // only support d=128 now, no tail
    FlashUpdateLastNoTailV510_VF<T, OUTPUT_T, MMOUTPUT_T, dSize>(dstTensor, curTensor, preTensor, expMaxTensor,
                                                                 expSumTensor, m, d);
}

template <typename T, typename OUTPUT_T, typename MMOUTPUT_T, uint32_t dSize = 0>
__aicore__ inline void FlashUpdateDivV510(const LocalTensor<T> &dstTensor, const LocalTensor<T> &preTensor,
                                          const LocalTensor<float> &expSumTensor, const uint16_t m, const uint16_t d)
{
    static_assert(IsSameType<T, half>::value, "VF FlashUpdate High Performance, T must be half");
    // only support d=128 now, no tail
    FlashUpdateDivNoTailV510_VF<T, OUTPUT_T, MMOUTPUT_T, dSize>(dstTensor, preTensor, expSumTensor, m, d);
}

template <typename T, uint32_t srcD>
__aicore__ inline void InvalidLineUpdate(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
                                         const LocalTensor<T> &maxTensor, const uint16_t m, const uint16_t d,
                                         const T minValue, const T invalidValue)
{
    __ubuf__ T *dstUb = (__ubuf__ T *)dstTensor.GetPhyAddr();
    __ubuf__ T *srcUb = (__ubuf__ T *)srcTensor.GetPhyAddr();
    __ubuf__ T *maxUb = (__ubuf__ T *)maxTensor.GetPhyAddr();

    constexpr uint16_t floatRepSize = 64;
    uint16_t dLoops = d >> 6;

    __VEC_SCOPE__
    {
        RegTensor<float> vreg_invalid_value;
        RegTensor<float> vreg_max;
        RegTensor<float> vreg_input;
        RegTensor<float> vreg_input_brc;

        MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
        MaskReg preg_compare;

        Duplicate(vreg_invalid_value, invalidValue);
        for (uint16_t i = 0; i < m; ++i) {
            DataCopy<T, Reg::LoadDist::DIST_BRC_B32>(vreg_max, maxUb + i);
            CompareScalar<T, CMPMODE::EQ>(preg_compare, vreg_max, minValue, preg_all);
            for (uint16_t j = 0; j < dLoops; ++j) {
                DataCopy(vreg_input, srcUb + i * d + j * floatRepSize);
                Select(vreg_input_brc, vreg_invalid_value, vreg_input, preg_compare);
                DataCopy<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)dstUb + i * d + j * floatRepSize,
                                                           vreg_input_brc, preg_all);
            }
        }
    }
}

template <typename T>
__aicore__ inline void RowInvalidUpdateVF(const LocalTensor<T> &finalTensor, const LocalTensor<float> &maxTensor,
                                          const uint16_t m, const uint16_t d, int64_t dSize)
{
    __ubuf__ T *finalUb = (__ubuf__ T *)finalTensor.GetPhyAddr();
    __ubuf__ float *maxUb = (__ubuf__ float *)maxTensor.GetPhyAddr();

    constexpr uint16_t floatRepSize = 64; // 64: 一个寄存器可以存储64个float类型数据
    const uint16_t dLoops = d / floatRepSize;
    const uint16_t tailD = d % floatRepSize;
    uint32_t pltTailD = static_cast<uint32_t>(tailD);
    uint16_t hasTail = 0;
    if (tailD > 0) {
        hasTail = 1;
    }

    constexpr uint32_t tmpZero = 0x00000000; // zero value of fp16 and fp32
    const T zeroValue = *((T *)&tmpZero);
    constexpr uint32_t tmpMin = 0xFF7FFFFF; // min value of float
    const float minValue = *((float *)&tmpMin);

    __VEC_SCOPE__
    {
        Reg::RegTensor<float> vregMinValue;
        Reg::RegTensor<T> vregZeroValue;
        Reg::RegTensor<float> vregMax;
        Reg::RegTensor<T> vregFinal;
        Reg::RegTensor<T> vregFinalNew;

        Reg::MaskReg pregAll = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregTailD = Reg::UpdateMask<T>(pltTailD);
        Reg::MaskReg pregCompare;

        Reg::Duplicate<float, float>(vregMinValue, minValue);
        Reg::Duplicate<T, T>(vregZeroValue, zeroValue);
        for (uint16_t i = 0; i < m; ++i) {
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(vregMax, maxUb + i);
            Reg::Compare<float, CMPMODE::EQ>(pregCompare, vregMax, vregMinValue, pregAll);
            for (uint16_t j = 0; j < dLoops; ++j) {
                Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregFinal, finalUb + i * dSize + j * floatRepSize);
                Reg::Select<T>(vregFinalNew, vregZeroValue, vregFinal, pregCompare);
                Reg::DataCopy<T, Reg::StoreDist::DIST_NORM_B32>(finalUb + i * dSize + j * floatRepSize, vregFinalNew,
                                                                pregAll);
            }
            for (uint16_t t = 0; t < hasTail; ++t) {
                Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregFinal, finalUb + i * dSize + dLoops * floatRepSize);
                Reg::Select<T>(vregFinalNew, vregZeroValue, vregFinal, pregCompare);
                Reg::DataCopy<T, Reg::StoreDist::DIST_NORM_B32>(finalUb + i * dSize + dLoops * floatRepSize,
                                                                vregFinalNew, pregTailD);
            }
        }
    }
}

} // namespace FaVectorApi

#endif // MY_FLASH_UPDATE_NEW_REGBASE_V2_INTERFACE_H
