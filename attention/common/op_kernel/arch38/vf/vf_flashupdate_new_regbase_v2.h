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

constexpr float FP16_NOQUANT_RANGE_SCALE_BACK = 256.0f;

struct VfVlParam {
    uint16_t vlSize;
    uint16_t loopCount;
    uint16_t tailSize;
};

template <typename T, uint32_t dSize>
__aicore__ inline constexpr VfVlParam GetVfVlParam()
{
    constexpr uint16_t vectorLength = 256;
    constexpr uint16_t vlSize = vectorLength / sizeof(T);
    constexpr uint16_t loopCount = (dSize + vlSize - 1) / vlSize;
    return {vlSize, loopCount, static_cast<uint16_t>(dSize - (loopCount - 1) * vlSize)};
}

constexpr Reg::CastTrait castF16F32Zero = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
                                           RoundMode::UNKNOWN};
constexpr Reg::CastTrait castF16F32One = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
                                          RoundMode::UNKNOWN};
constexpr Reg::CastTrait castF32F16Zero = {Reg::RegLayout::ZERO, Reg::SatMode::SAT, Reg::MaskMergeMode::ZEROING,
                                           RoundMode::CAST_RINT};
constexpr Reg::CastTrait castF32F16One = {Reg::RegLayout::ONE, Reg::SatMode::SAT, Reg::MaskMergeMode::ZEROING,
                                          RoundMode::CAST_RINT};

template <typename T, typename OUTPUT_T, typename MMOUTPUT_T, uint32_t dSize = 0>
__aicore__ inline void FlashUpdateLoopV510Vf(const LocalTensor<T>& dstTensor, const LocalTensor<T>& curTensor,
                                             const LocalTensor<T>& preTensor, const LocalTensor<float>& expMaxTensor,
                                             const uint16_t m, const uint16_t d)
{
    static_assert(IsSameType<T, half>::value, "VF FlashUpdate High Performance, T must be half");
    static_assert(dSize > 0, "dSize must be a compile-time constant greater than 0");
    __ubuf__ T* dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ T* curUb = (__ubuf__ T*)curTensor.GetPhyAddr();
    __ubuf__ T* preUb = (__ubuf__ T*)preTensor.GetPhyAddr();
    __ubuf__ float* expMaxUb = (__ubuf__ float*)expMaxTensor.GetPhyAddr();

    constexpr uint16_t reduceSize = 1;
    constexpr VfVlParam vlParam = GetVfVlParam<T, dSize>();
    constexpr uint16_t fullChunks = vlParam.loopCount - 1;
    constexpr uint16_t tailOffset = fullChunks * vlParam.vlSize;

    __VEC_SCOPE__
    {
        Reg::RegTensor<float> vSrcRegExpMax;
        Reg::RegTensor<T> vSrcRegExpMaxB16;
        Reg::RegTensor<T> vSrcRegExpMaxB16Even;
        Reg::RegTensor<T> vSrcRegExpMaxB16Odd;
        Reg::RegTensor<T> vSrcRegPre;
        Reg::RegTensor<T> vSrcRegCur;
        Reg::RegTensor<T> vSrcRegMul;
        Reg::RegTensor<T> vDstRegAdd;

        Reg::MaskReg maskRegAllB32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskRegAllB16 = Reg::CreateMask<T, Reg::MaskPattern::ALL>();

        Reg::MaskReg pregFull;
        if constexpr (fullChunks > 0) {
            uint32_t fullCnt = vlParam.vlSize;
            pregFull = Reg::UpdateMask<T>(fullCnt);
        }
        uint32_t tailCnt = vlParam.tailSize;
        Reg::MaskReg pregTail = Reg::UpdateMask<T>(tailCnt);

        for (uint16_t i = 0; i < m; ++i) {
            static constexpr Reg::CastTrait castTrait0 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            static constexpr Reg::CastTrait castTrait1 = {Reg::RegLayout::ONE, Reg::SatMode::NO_SAT,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(vSrcRegExpMax, expMaxUb + i * reduceSize);
            Reg::Cast<T, float, castTrait0>(vSrcRegExpMaxB16Even, vSrcRegExpMax, maskRegAllB32);
            Reg::Cast<T, float, castTrait1>(vSrcRegExpMaxB16Odd, vSrcRegExpMax, maskRegAllB32);
            Reg::Or<uint16_t, Reg::MaskMergeMode::ZEROING>(
                (Reg::RegTensor<uint16_t>&)vSrcRegExpMaxB16, (Reg::RegTensor<uint16_t>&)vSrcRegExpMaxB16Even,
                (Reg::RegTensor<uint16_t>&)vSrcRegExpMaxB16Odd, maskRegAllB16);

            if constexpr (fullChunks > 0) {
                for (uint16_t j = 0; j < fullChunks; ++j) {
                    const uint32_t colOffset = i * dSize + j * vlParam.vlSize;
                    Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vSrcRegPre, preUb + colOffset);
                    Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vSrcRegCur, curUb + colOffset);
                    Reg::Mul<T, Reg::MaskMergeMode::ZEROING>(vSrcRegMul, vSrcRegExpMaxB16, vSrcRegPre, pregFull);
                    Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vDstRegAdd, vSrcRegMul, vSrcRegCur, pregFull);
                    Reg::DataCopy<OUTPUT_T, Reg::StoreDist::DIST_NORM_B16>(dstUb + colOffset, vDstRegAdd, pregFull);
                }
            }

            const uint32_t tailColOffset = i * dSize + tailOffset;
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vSrcRegPre, preUb + tailColOffset);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vSrcRegCur, curUb + tailColOffset);
            Reg::Mul<T, Reg::MaskMergeMode::ZEROING>(vSrcRegMul, vSrcRegExpMaxB16, vSrcRegPre, pregTail);
            Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vDstRegAdd, vSrcRegMul, vSrcRegCur, pregTail);
            Reg::DataCopy<OUTPUT_T, Reg::StoreDist::DIST_NORM_B16>(dstUb + tailColOffset, vDstRegAdd, pregTail);
        }
    }
}

template <typename T, typename OUTPUT_T, typename MMOUTPUT_T, uint32_t dSize = 0>
__aicore__ inline void FlashUpdateLastLoopV510Vf(const LocalTensor<T>& dstTensor, const LocalTensor<T>& curTensor,
                                                 const LocalTensor<T>& preTensor,
                                                 const LocalTensor<float>& expMaxTensor,
                                                 const LocalTensor<float>& expSumTensor, const uint16_t m,
                                                 const uint16_t d)
{
    static_assert(IsSameType<T, half>::value, "VF FlashUpdate High Performance, T must be half");
    static_assert(dSize > 0, "dSize must be a compile-time constant greater than 0");
    __ubuf__ T* dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ T* curUb = (__ubuf__ T*)curTensor.GetPhyAddr();
    __ubuf__ T* preUb = (__ubuf__ T*)preTensor.GetPhyAddr();
    __ubuf__ float* expMaxUb = (__ubuf__ float*)expMaxTensor.GetPhyAddr();
    __ubuf__ float* expSumUb = (__ubuf__ float*)expSumTensor.GetPhyAddr();

    constexpr uint16_t reduceSize = 1;
    constexpr VfVlParam vlParam = GetVfVlParam<T, dSize>();
    constexpr uint16_t fullChunks = vlParam.loopCount - 1;
    constexpr uint16_t tailOffset = fullChunks * vlParam.vlSize;

    __VEC_SCOPE__
    {
        Reg::RegTensor<float> vSrcRegSum;
        Reg::RegTensor<float> vSrcRegExpMax;
        Reg::RegTensor<T> vSrcRegSumB16;
        Reg::RegTensor<T> vSrcRegSumB16Even;
        Reg::RegTensor<T> vSrcRegSumB16Odd;
        Reg::RegTensor<T> vSrcRegExpMaxB16;
        Reg::RegTensor<T> vSrcRegExpMaxB16Even;
        Reg::RegTensor<T> vSrcRegExpMaxB16Odd;
        Reg::RegTensor<T> vSrcRegPre;
        Reg::RegTensor<T> vSrcRegCur;
        Reg::RegTensor<T> vSrcRegMul;
        Reg::RegTensor<T> vSrcRegAdd;
        Reg::RegTensor<T> vDstRegDiv;

        static constexpr Reg::DivSpecificMode mode = {Reg::MaskMergeMode::ZEROING, false};
        Reg::MaskReg maskRegAllB32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskRegAllB16 = Reg::CreateMask<T, Reg::MaskPattern::ALL>();

        Reg::MaskReg pregFull;
        if constexpr (fullChunks > 0) {
            uint32_t fullCnt = vlParam.vlSize;
            pregFull = Reg::UpdateMask<T>(fullCnt);
        }
        uint32_t tailCnt = vlParam.tailSize;
        Reg::MaskReg pregTail = Reg::UpdateMask<T>(tailCnt);

        for (uint16_t i = 0; i < m; ++i) {
            static constexpr Reg::CastTrait castTrait0 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            static constexpr Reg::CastTrait castTrait1 = {Reg::RegLayout::ONE, Reg::SatMode::NO_SAT,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(vSrcRegExpMax, expMaxUb + i * reduceSize);
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(vSrcRegSum, expSumUb + i * reduceSize);
            Reg::Cast<T, float, castTrait0>(vSrcRegExpMaxB16Even, vSrcRegExpMax, maskRegAllB32);
            Reg::Cast<T, float, castTrait1>(vSrcRegExpMaxB16Odd, vSrcRegExpMax, maskRegAllB32);
            Reg::Cast<T, float, castTrait0>(vSrcRegSumB16Even, vSrcRegSum, maskRegAllB32);
            Reg::Cast<T, float, castTrait1>(vSrcRegSumB16Odd, vSrcRegSum, maskRegAllB32);
            Reg::Or<uint16_t, Reg::MaskMergeMode::ZEROING>(
                (Reg::RegTensor<uint16_t>&)vSrcRegExpMaxB16, (Reg::RegTensor<uint16_t>&)vSrcRegExpMaxB16Even,
                (Reg::RegTensor<uint16_t>&)vSrcRegExpMaxB16Odd, maskRegAllB16);
            Reg::Or<uint16_t, Reg::MaskMergeMode::ZEROING>((Reg::RegTensor<uint16_t>&)vSrcRegSumB16,
                                                           (Reg::RegTensor<uint16_t>&)vSrcRegSumB16Even,
                                                           (Reg::RegTensor<uint16_t>&)vSrcRegSumB16Odd, maskRegAllB16);

            if constexpr (fullChunks > 0) {
                for (uint16_t j = 0; j < fullChunks; ++j) {
                    const uint32_t colOffset = i * dSize + j * vlParam.vlSize;
                    Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vSrcRegPre, preUb + colOffset);
                    Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vSrcRegCur, curUb + colOffset);
                    Reg::Mul<T, Reg::MaskMergeMode::ZEROING>(vSrcRegMul, vSrcRegExpMaxB16, vSrcRegPre, pregFull);
                    Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vSrcRegAdd, vSrcRegMul, vSrcRegCur, pregFull);
                    Reg::Div<T, &mode>(vDstRegDiv, vSrcRegAdd, vSrcRegSumB16, pregFull);
                    Reg::DataCopy<OUTPUT_T, Reg::StoreDist::DIST_NORM_B16>(dstUb + colOffset, vDstRegDiv, pregFull);
                }
            }

            const uint32_t tailColOffset = i * dSize + tailOffset;
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vSrcRegPre, preUb + tailColOffset);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vSrcRegCur, curUb + tailColOffset);
            Reg::Mul<T, Reg::MaskMergeMode::ZEROING>(vSrcRegMul, vSrcRegExpMaxB16, vSrcRegPre, pregTail);
            Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vSrcRegAdd, vSrcRegMul, vSrcRegCur, pregTail);
            Reg::Div<T, &mode>(vDstRegDiv, vSrcRegAdd, vSrcRegSumB16, pregTail);
            Reg::DataCopy<OUTPUT_T, Reg::StoreDist::DIST_NORM_B16>(dstUb + tailColOffset, vDstRegDiv, pregTail);
        }
    }
}

template <typename T, typename OUTPUT_T, typename MMOUTPUT_T, uint32_t dSize = 0>
__aicore__ inline void FlashUpdateDivLoopV510Vf(const LocalTensor<T>& dstTensor, const LocalTensor<T>& preTensor,
                                                const LocalTensor<float>& expSumTensor, const uint16_t m,
                                                const uint16_t d)
{
    static_assert(IsSameType<T, half>::value, "VF FlashUpdate High Performance, T must be half");
    static_assert(dSize > 0, "dSize must be a compile-time constant greater than 0");
    __ubuf__ T* dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ T* preUb = (__ubuf__ T*)preTensor.GetPhyAddr();
    __ubuf__ float* expSumUb = (__ubuf__ float*)expSumTensor.GetPhyAddr();

    constexpr uint16_t reduceSize = 1;
    constexpr VfVlParam vlParam = GetVfVlParam<T, dSize>();
    constexpr uint16_t fullChunks = vlParam.loopCount - 1;
    constexpr uint16_t tailOffset = fullChunks * vlParam.vlSize;

    __VEC_SCOPE__
    {
        Reg::RegTensor<float> vSrcRegSum;
        Reg::RegTensor<T> vSrcRegSumB16Even;
        Reg::RegTensor<T> vSrcRegSumB16Odd;
        Reg::RegTensor<T> vSrcRegSumB16;
        Reg::RegTensor<T> vSrcRegPre;
        Reg::RegTensor<T> vDstRegDiv;

        static constexpr Reg::DivSpecificMode mode = {Reg::MaskMergeMode::ZEROING, false};
        Reg::MaskReg maskRegAllB32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskRegAllB16 = Reg::CreateMask<T, Reg::MaskPattern::ALL>();

        Reg::MaskReg pregFull;
        if constexpr (fullChunks > 0) {
            uint32_t fullCnt = vlParam.vlSize;
            pregFull = Reg::UpdateMask<T>(fullCnt);
        }
        uint32_t tailCnt = vlParam.tailSize;
        Reg::MaskReg pregTail = Reg::UpdateMask<T>(tailCnt);

        for (uint16_t i = 0; i < m; ++i) {
            static constexpr Reg::CastTrait castTrait0 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            static constexpr Reg::CastTrait castTrait1 = {Reg::RegLayout::ONE, Reg::SatMode::NO_SAT,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(vSrcRegSum, expSumUb + i * reduceSize);
            Reg::Cast<T, float, castTrait0>(vSrcRegSumB16Even, vSrcRegSum, maskRegAllB32);
            Reg::Cast<T, float, castTrait1>(vSrcRegSumB16Odd, vSrcRegSum, maskRegAllB32);
            Reg::Or<uint16_t, Reg::MaskMergeMode::ZEROING>((Reg::RegTensor<uint16_t>&)vSrcRegSumB16,
                                                           (Reg::RegTensor<uint16_t>&)vSrcRegSumB16Even,
                                                           (Reg::RegTensor<uint16_t>&)vSrcRegSumB16Odd, maskRegAllB16);

            if constexpr (fullChunks > 0) {
                for (uint16_t j = 0; j < fullChunks; ++j) {
                    const uint32_t colOffset = i * dSize + j * vlParam.vlSize;
                    Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vSrcRegPre, preUb + colOffset);
                    Reg::Div<T, &mode>(vDstRegDiv, vSrcRegPre, vSrcRegSumB16, pregFull);
                    Reg::DataCopy<OUTPUT_T, Reg::StoreDist::DIST_NORM_B16>(dstUb + colOffset, vDstRegDiv, pregFull);
                }
            }

            const uint32_t tailColOffset = i * dSize + tailOffset;
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vSrcRegPre, preUb + tailColOffset);
            Reg::Div<T, &mode>(vDstRegDiv, vSrcRegPre, vSrcRegSumB16, pregTail);
            Reg::DataCopy<OUTPUT_T, Reg::StoreDist::DIST_NORM_B16>(dstUb + tailColOffset, vDstRegDiv, pregTail);
        }
    }
}

template <uint32_t dSize = 0>
__aicore__ inline void FlashUpdateLoopV510VfF32(const LocalTensor<float>& dstTensor, const LocalTensor<half>& curTensor,
                                                const LocalTensor<float>& preTensor,
                                                const LocalTensor<float>& expMaxTensor, const uint16_t m,
                                                const uint16_t d)
{
    static_assert(dSize > 0, "dSize must be a compile-time constant greater than 0");
    __ubuf__ float* dstUb = (__ubuf__ float*)dstTensor.GetPhyAddr();
    __ubuf__ half* curUb = (__ubuf__ half*)curTensor.GetPhyAddr();
    __ubuf__ float* preUb = (__ubuf__ float*)preTensor.GetPhyAddr();
    __ubuf__ float* expMaxUb = (__ubuf__ float*)expMaxTensor.GetPhyAddr();

    constexpr uint16_t vlSize = 128;
    constexpr uint16_t halfVlSize = 64;
    constexpr uint16_t loopCount = (dSize + vlSize - 1) / vlSize;

    __VEC_SCOPE__
    {
        Reg::RegTensor<float> vSrcRegExpMax;
        Reg::RegTensor<half> vSrcRegCurB16;
        Reg::MaskReg maskAllB16 = Reg::CreateMask<half, Reg::MaskPattern::ALL>();
        Reg::RegTensor<float> vSrcRegCurEven;
        Reg::RegTensor<float> vSrcRegCurOdd;
        Reg::RegTensor<float> vSrcRegPreEven;
        Reg::RegTensor<float> vSrcRegPreOdd;
        Reg::RegTensor<float> vSrcRegMulEven;
        Reg::RegTensor<float> vSrcRegMulOdd;
        Reg::RegTensor<float> vDstRegAddEven;
        Reg::RegTensor<float> vDstRegAddOdd;

        Reg::MaskReg maskAllF32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();

        for (uint16_t i = 0; i < m; ++i) {
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(vSrcRegExpMax, expMaxUb + i);
            for (uint16_t j = 0; j < loopCount; ++j) {
                const uint32_t colOffset = i * dSize + j * vlSize;
                Reg::DataCopy<half, Reg::LoadDist::DIST_NORM>(vSrcRegCurB16, curUb + colOffset);
                Reg::Cast<float, half, castF16F32Zero>(vSrcRegCurEven, vSrcRegCurB16, maskAllB16);
                Reg::Cast<float, half, castF16F32One>(vSrcRegCurOdd, vSrcRegCurB16, maskAllB16);
                Reg::DataCopy<float, Reg::LoadDist::DIST_NORM>(vSrcRegPreEven, preUb + colOffset);
                Reg::DataCopy<float, Reg::LoadDist::DIST_NORM>(vSrcRegPreOdd, preUb + colOffset + halfVlSize);
                Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(vSrcRegMulEven, vSrcRegExpMax, vSrcRegPreEven, maskAllF32);
                Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(vSrcRegMulOdd, vSrcRegExpMax, vSrcRegPreOdd, maskAllF32);
                Reg::Add<float, Reg::MaskMergeMode::ZEROING>(vDstRegAddEven, vSrcRegMulEven, vSrcRegCurEven,
                                                             maskAllF32);
                Reg::Add<float, Reg::MaskMergeMode::ZEROING>(vDstRegAddOdd, vSrcRegMulOdd, vSrcRegCurOdd, maskAllF32);
                Reg::DataCopy<float, Reg::StoreDist::DIST_NORM_B32>(dstUb + colOffset, vDstRegAddEven, maskAllF32);
                Reg::DataCopy<float, Reg::StoreDist::DIST_NORM_B32>(dstUb + colOffset + halfVlSize, vDstRegAddOdd,
                                                                    maskAllF32);
            }
        }
    }
}

template <uint32_t dSize = 0>
__aicore__ inline void FlashUpdateLastLoopV510VfF32(
    const LocalTensor<float>& dstTensor, const LocalTensor<half>& curTensor, const LocalTensor<float>& preTensor,
    const LocalTensor<float>& expMaxTensor, const LocalTensor<float>& expSumTensor, const uint16_t m, const uint16_t d)
{
    static_assert(dSize > 0, "dSize must be a compile-time constant greater than 0");
    __ubuf__ float* dstUb = (__ubuf__ float*)dstTensor.GetPhyAddr();
    __ubuf__ half* curUb = (__ubuf__ half*)curTensor.GetPhyAddr();
    __ubuf__ float* preUb = (__ubuf__ float*)preTensor.GetPhyAddr();
    __ubuf__ float* expMaxUb = (__ubuf__ float*)expMaxTensor.GetPhyAddr();
    __ubuf__ float* expSumUb = (__ubuf__ float*)expSumTensor.GetPhyAddr();

    constexpr uint16_t vlSize = 128;
    constexpr uint16_t halfVlSize = 64;
    constexpr uint16_t loopCount = (dSize + vlSize - 1) / vlSize;

    __VEC_SCOPE__
    {
        Reg::RegTensor<float> vSrcRegExpMax;
        Reg::RegTensor<half> vSrcRegCurB16;
        Reg::MaskReg maskAllB16 = Reg::CreateMask<half, Reg::MaskPattern::ALL>();
        Reg::RegTensor<float> vSrcRegSum;
        Reg::RegTensor<float> vSrcRegCurEven;
        Reg::RegTensor<float> vSrcRegCurOdd;
        Reg::RegTensor<float> vSrcRegPreEven;
        Reg::RegTensor<float> vSrcRegPreOdd;
        Reg::RegTensor<float> vSrcRegMulEven;
        Reg::RegTensor<float> vSrcRegMulOdd;
        Reg::RegTensor<float> vSrcRegAddEven;
        Reg::RegTensor<float> vSrcRegAddOdd;
        Reg::RegTensor<float> vDstRegDivEven;
        Reg::RegTensor<float> vDstRegDivOdd;
        Reg::RegTensor<float> vScaleBack;

        static constexpr Reg::DivSpecificMode mode = {Reg::MaskMergeMode::ZEROING, false};
        Reg::MaskReg maskAllF32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::Duplicate<float, float>(vScaleBack, FP16_NOQUANT_RANGE_SCALE_BACK);

        for (uint16_t i = 0; i < m; ++i) {
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(vSrcRegExpMax, expMaxUb + i);
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(vSrcRegSum, expSumUb + i);
            for (uint16_t j = 0; j < loopCount; ++j) {
                const uint32_t colOffset = i * dSize + j * vlSize;
                Reg::DataCopy<half, Reg::LoadDist::DIST_NORM>(vSrcRegCurB16, curUb + colOffset);
                Reg::Cast<float, half, castF16F32Zero>(vSrcRegCurEven, vSrcRegCurB16, maskAllB16);
                Reg::Cast<float, half, castF16F32One>(vSrcRegCurOdd, vSrcRegCurB16, maskAllB16);
                Reg::DataCopy<float, Reg::LoadDist::DIST_NORM>(vSrcRegPreEven, preUb + colOffset);
                Reg::DataCopy<float, Reg::LoadDist::DIST_NORM>(vSrcRegPreOdd, preUb + colOffset + halfVlSize);
                Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(vSrcRegMulEven, vSrcRegExpMax, vSrcRegPreEven, maskAllF32);
                Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(vSrcRegMulOdd, vSrcRegExpMax, vSrcRegPreOdd, maskAllF32);
                Reg::Add<float, Reg::MaskMergeMode::ZEROING>(vSrcRegAddEven, vSrcRegMulEven, vSrcRegCurEven,
                                                             maskAllF32);
                Reg::Add<float, Reg::MaskMergeMode::ZEROING>(vSrcRegAddOdd, vSrcRegMulOdd, vSrcRegCurOdd, maskAllF32);
                Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(vSrcRegAddEven, vSrcRegAddEven, vScaleBack, maskAllF32);
                Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(vSrcRegAddOdd, vSrcRegAddOdd, vScaleBack, maskAllF32);
                Reg::Div<float, &mode>(vDstRegDivEven, vSrcRegAddEven, vSrcRegSum, maskAllF32);
                Reg::Div<float, &mode>(vDstRegDivOdd, vSrcRegAddOdd, vSrcRegSum, maskAllF32);
                Reg::DataCopy<float, Reg::StoreDist::DIST_NORM_B32>(dstUb + colOffset, vDstRegDivEven, maskAllF32);
                Reg::DataCopy<float, Reg::StoreDist::DIST_NORM_B32>(dstUb + colOffset + halfVlSize, vDstRegDivOdd,
                                                                    maskAllF32);
            }
        }
    }
}

template <uint32_t dSize = 0>
__aicore__ inline void FlashUpdateDivLoopV510VfF32(const LocalTensor<float>& dstTensor,
                                                   const LocalTensor<float>& preTensor,
                                                   const LocalTensor<float>& expSumTensor, const uint16_t m,
                                                   const uint16_t d)
{
    static_assert(dSize > 0, "dSize must be a compile-time constant greater than 0");
    __ubuf__ float* dstUb = (__ubuf__ float*)dstTensor.GetPhyAddr();
    __ubuf__ float* preUb = (__ubuf__ float*)preTensor.GetPhyAddr();
    __ubuf__ float* expSumUb = (__ubuf__ float*)expSumTensor.GetPhyAddr();

    constexpr uint16_t vlSize = 128;
    constexpr uint16_t halfVlSize = 64;
    constexpr uint16_t loopCount = (dSize + vlSize - 1) / vlSize;

    __VEC_SCOPE__
    {
        Reg::RegTensor<float> vSrcRegSum;
        Reg::RegTensor<float> vSrcRegPreEven;
        Reg::RegTensor<float> vSrcRegPreOdd;
        Reg::RegTensor<float> vDstRegDivEven;
        Reg::RegTensor<float> vDstRegDivOdd;
        Reg::RegTensor<float> vScaleBack;

        static constexpr Reg::DivSpecificMode mode = {Reg::MaskMergeMode::ZEROING, false};
        Reg::MaskReg maskAllF32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::Duplicate<float, float>(vScaleBack, FP16_NOQUANT_RANGE_SCALE_BACK);

        for (uint16_t i = 0; i < m; ++i) {
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(vSrcRegSum, expSumUb + i);
            for (uint16_t j = 0; j < loopCount; ++j) {
                const uint32_t colOffset = i * dSize + j * vlSize;
                Reg::DataCopy<float, Reg::LoadDist::DIST_NORM>(vSrcRegPreEven, preUb + colOffset);
                Reg::DataCopy<float, Reg::LoadDist::DIST_NORM>(vSrcRegPreOdd, preUb + colOffset + halfVlSize);
                Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(vSrcRegPreEven, vSrcRegPreEven, vScaleBack, maskAllF32);
                Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(vSrcRegPreOdd, vSrcRegPreOdd, vScaleBack, maskAllF32);
                Reg::Div<float, &mode>(vDstRegDivEven, vSrcRegPreEven, vSrcRegSum, maskAllF32);
                Reg::Div<float, &mode>(vDstRegDivOdd, vSrcRegPreOdd, vSrcRegSum, maskAllF32);
                Reg::DataCopy<float, Reg::StoreDist::DIST_NORM_B32>(dstUb + colOffset, vDstRegDivEven, maskAllF32);
                Reg::DataCopy<float, Reg::StoreDist::DIST_NORM_B32>(dstUb + colOffset + halfVlSize, vDstRegDivOdd,
                                                                    maskAllF32);
            }
        }
    }
}

template <uint32_t dSize = 0>
__aicore__ inline void FlashUpdatePromoteV510VfF32(const LocalTensor<float>& dstTensor,
                                                   const LocalTensor<half>& curTensor, const uint16_t m,
                                                   const uint16_t d)
{
    static_assert(dSize > 0, "dSize must be a compile-time constant greater than 0");
    __ubuf__ float* dstUb = (__ubuf__ float*)dstTensor.GetPhyAddr();
    __ubuf__ half* curUb = (__ubuf__ half*)curTensor.GetPhyAddr();

    constexpr uint16_t vlSize = 128;
    constexpr uint16_t halfVlSize = 64;
    constexpr uint16_t loopCount = (dSize + vlSize - 1) / vlSize;

    __VEC_SCOPE__
    {
        Reg::RegTensor<half> vSrcRegCurB16;
        Reg::MaskReg maskAllB16 = Reg::CreateMask<half, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskAllF32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::RegTensor<float> vSrcRegCurEven;
        Reg::RegTensor<float> vSrcRegCurOdd;

        for (uint16_t i = 0; i < m; ++i) {
            for (uint16_t j = 0; j < loopCount; ++j) {
                const uint32_t colOffset = i * dSize + j * vlSize;
                Reg::DataCopy<half, Reg::LoadDist::DIST_NORM>(vSrcRegCurB16, curUb + colOffset);
                Reg::Cast<float, half, castF16F32Zero>(vSrcRegCurEven, vSrcRegCurB16, maskAllB16);
                Reg::Cast<float, half, castF16F32One>(vSrcRegCurOdd, vSrcRegCurB16, maskAllB16);
                Reg::DataCopy<float, Reg::StoreDist::DIST_NORM_B32>(dstUb + colOffset, vSrcRegCurEven, maskAllF32);
                Reg::DataCopy<float, Reg::StoreDist::DIST_NORM_B32>(dstUb + colOffset + halfVlSize, vSrcRegCurOdd,
                                                                    maskAllF32);
            }
        }
    }
}

template <uint32_t dSize = 0>
__aicore__ inline void FlashUpdateNarrowV510VfF32(const LocalTensor<half>& dstTensor,
                                                  const LocalTensor<float>& srcTensor, const uint16_t m,
                                                  const uint16_t d)
{
    static_assert(dSize > 0, "dSize must be a compile-time constant greater than 0");
    __ubuf__ half* dstUb = (__ubuf__ half*)dstTensor.GetPhyAddr();
    __ubuf__ float* srcUb = (__ubuf__ float*)srcTensor.GetPhyAddr();

    constexpr uint16_t vlSize = 128;
    constexpr uint16_t halfVlSize = 64;
    constexpr uint16_t loopCount = (dSize + vlSize - 1) / vlSize;

    __VEC_SCOPE__
    {
        Reg::RegTensor<float> vSrcRegEven;
        Reg::RegTensor<float> vSrcRegOdd;
        Reg::RegTensor<half> vDstRegEvenF16;
        Reg::RegTensor<half> vDstRegOddF16;
        Reg::RegTensor<half> vDstRegMerged;
        Reg::MaskReg maskAllF32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskAllB16 = Reg::CreateMask<half, Reg::MaskPattern::ALL>();

        for (uint16_t i = 0; i < m; ++i) {
            for (uint16_t j = 0; j < loopCount; ++j) {
                const uint32_t colOffset = i * dSize + j * vlSize;
                Reg::DataCopy<float, Reg::LoadDist::DIST_NORM>(vSrcRegEven, srcUb + colOffset);
                Reg::DataCopy<float, Reg::LoadDist::DIST_NORM>(vSrcRegOdd, srcUb + colOffset + halfVlSize);
                Reg::Cast<half, float, castF32F16Zero>(vDstRegEvenF16, vSrcRegEven, maskAllF32);
                Reg::Cast<half, float, castF32F16One>(vDstRegOddF16, vSrcRegOdd, maskAllF32);
                Reg::Or<uint16_t, Reg::MaskMergeMode::ZEROING>((Reg::RegTensor<uint16_t>&)vDstRegMerged,
                                                               (Reg::RegTensor<uint16_t>&)vDstRegEvenF16,
                                                               (Reg::RegTensor<uint16_t>&)vDstRegOddF16, maskAllB16);
                Reg::DataCopy<half, Reg::StoreDist::DIST_NORM_B16>(dstUb + colOffset, vDstRegMerged, maskAllB16);
            }
        }
    }
}

template <uint32_t dSize = 0>
__aicore__ inline void FlashUpdatePromoteV510VfD64(const LocalTensor<float>& dstTensor,
                                                   const LocalTensor<half>& curTensor, const uint16_t m,
                                                   const uint16_t d)
{
    static_assert(dSize > 0, "dSize must be a compile-time constant greater than 0");
    __ubuf__ float* dstUb = (__ubuf__ float*)dstTensor.GetPhyAddr();
    __ubuf__ half* curUb = (__ubuf__ half*)curTensor.GetPhyAddr();

    constexpr uint16_t vlSize = 64;
    constexpr uint16_t loopCount = (dSize + vlSize - 1) / vlSize;

    __VEC_SCOPE__
    {
        Reg::RegTensor<half> vSrcRegCurB16;
        Reg::RegTensor<float> vSrcRegCurF32;
        Reg::MaskReg maskAllF32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();

        for (uint16_t i = 0; i < m; ++i) {
            for (uint16_t j = 0; j < loopCount; ++j) {
                const uint32_t colOffset = i * dSize + j * vlSize;
                Reg::DataCopy<half, Reg::LoadDist::DIST_UNPACK_B16>(vSrcRegCurB16, curUb + colOffset);
                Reg::Cast<float, half, castF16F32Zero>(vSrcRegCurF32, vSrcRegCurB16, maskAllF32);
                Reg::DataCopy<float, Reg::StoreDist::DIST_NORM_B32>(dstUb + colOffset, vSrcRegCurF32, maskAllF32);
            }
        }
    }
}

template <uint32_t dSize = 0>
__aicore__ inline void FlashUpdateLoopV510VfD64(const LocalTensor<float>& dstTensor, const LocalTensor<half>& curTensor,
                                                const LocalTensor<float>& preTensor,
                                                const LocalTensor<float>& expMaxTensor, const uint16_t m,
                                                const uint16_t d)
{
    static_assert(dSize > 0, "dSize must be a compile-time constant greater than 0");
    __ubuf__ float* dstUb = (__ubuf__ float*)dstTensor.GetPhyAddr();
    __ubuf__ half* curUb = (__ubuf__ half*)curTensor.GetPhyAddr();
    __ubuf__ float* preUb = (__ubuf__ float*)preTensor.GetPhyAddr();
    __ubuf__ float* expMaxUb = (__ubuf__ float*)expMaxTensor.GetPhyAddr();

    constexpr uint16_t vlSize = 64;
    constexpr uint16_t loopCount = (dSize + vlSize - 1) / vlSize;

    __VEC_SCOPE__
    {
        Reg::RegTensor<float> vSrcRegExpMax;
        Reg::RegTensor<half> vSrcRegCurB16;
        Reg::RegTensor<float> vSrcRegCurF32;
        Reg::RegTensor<float> vSrcRegPre;
        Reg::RegTensor<float> vSrcRegMul;
        Reg::RegTensor<float> vDstRegAdd;
        Reg::MaskReg maskAllF32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();

        for (uint16_t i = 0; i < m; ++i) {
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(vSrcRegExpMax, expMaxUb + i);
            for (uint16_t j = 0; j < loopCount; ++j) {
                const uint32_t colOffset = i * dSize + j * vlSize;
                Reg::DataCopy<half, Reg::LoadDist::DIST_UNPACK_B16>(vSrcRegCurB16, curUb + colOffset);
                Reg::Cast<float, half, castF16F32Zero>(vSrcRegCurF32, vSrcRegCurB16, maskAllF32);
                Reg::DataCopy<float, Reg::LoadDist::DIST_NORM>(vSrcRegPre, preUb + colOffset);
                Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(vSrcRegMul, vSrcRegExpMax, vSrcRegPre, maskAllF32);
                Reg::Add<float, Reg::MaskMergeMode::ZEROING>(vDstRegAdd, vSrcRegMul, vSrcRegCurF32, maskAllF32);
                Reg::DataCopy<float, Reg::StoreDist::DIST_NORM_B32>(dstUb + colOffset, vDstRegAdd, maskAllF32);
            }
        }
    }
}

template <uint32_t dSize = 0>
__aicore__ inline void FlashUpdateLastLoopV510VfD64(
    const LocalTensor<float>& dstTensor, const LocalTensor<half>& curTensor, const LocalTensor<float>& preTensor,
    const LocalTensor<float>& expMaxTensor, const LocalTensor<float>& expSumTensor, const uint16_t m, const uint16_t d)
{
    static_assert(dSize > 0, "dSize must be a compile-time constant greater than 0");
    __ubuf__ float* dstUb = (__ubuf__ float*)dstTensor.GetPhyAddr();
    __ubuf__ half* curUb = (__ubuf__ half*)curTensor.GetPhyAddr();
    __ubuf__ float* preUb = (__ubuf__ float*)preTensor.GetPhyAddr();
    __ubuf__ float* expMaxUb = (__ubuf__ float*)expMaxTensor.GetPhyAddr();
    __ubuf__ float* expSumUb = (__ubuf__ float*)expSumTensor.GetPhyAddr();

    constexpr uint16_t vlSize = 64;
    constexpr uint16_t loopCount = (dSize + vlSize - 1) / vlSize;

    __VEC_SCOPE__
    {
        Reg::RegTensor<float> vSrcRegExpMax;
        Reg::RegTensor<float> vSrcRegSum;
        Reg::RegTensor<half> vSrcRegCurB16;
        Reg::RegTensor<float> vSrcRegCurF32;
        Reg::RegTensor<float> vSrcRegPre;
        Reg::RegTensor<float> vSrcRegMul;
        Reg::RegTensor<float> vSrcRegAdd;
        Reg::RegTensor<float> vDstRegDiv;
        Reg::RegTensor<float> vScaleBack;
        static constexpr Reg::DivSpecificMode mode = {Reg::MaskMergeMode::ZEROING, false};
        Reg::MaskReg maskAllF32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::Duplicate<float, float>(vScaleBack, FP16_NOQUANT_RANGE_SCALE_BACK);

        for (uint16_t i = 0; i < m; ++i) {
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(vSrcRegExpMax, expMaxUb + i);
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(vSrcRegSum, expSumUb + i);
            for (uint16_t j = 0; j < loopCount; ++j) {
                const uint32_t colOffset = i * dSize + j * vlSize;
                Reg::DataCopy<half, Reg::LoadDist::DIST_UNPACK_B16>(vSrcRegCurB16, curUb + colOffset);
                Reg::Cast<float, half, castF16F32Zero>(vSrcRegCurF32, vSrcRegCurB16, maskAllF32);
                Reg::DataCopy<float, Reg::LoadDist::DIST_NORM>(vSrcRegPre, preUb + colOffset);
                Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(vSrcRegMul, vSrcRegExpMax, vSrcRegPre, maskAllF32);
                Reg::Add<float, Reg::MaskMergeMode::ZEROING>(vSrcRegAdd, vSrcRegMul, vSrcRegCurF32, maskAllF32);
                Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(vSrcRegAdd, vSrcRegAdd, vScaleBack, maskAllF32);
                Reg::Div<float, &mode>(vDstRegDiv, vSrcRegAdd, vSrcRegSum, maskAllF32);
                Reg::DataCopy<float, Reg::StoreDist::DIST_NORM_B32>(dstUb + colOffset, vDstRegDiv, maskAllF32);
            }
        }
    }
}

template <uint32_t dSize = 0>
__aicore__ inline void FlashUpdateDivLoopV510VfD64(const LocalTensor<float>& dstTensor,
                                                   const LocalTensor<float>& preTensor,
                                                   const LocalTensor<float>& expSumTensor, const uint16_t m,
                                                   const uint16_t d)
{
    static_assert(dSize > 0, "dSize must be a compile-time constant greater than 0");
    __ubuf__ float* dstUb = (__ubuf__ float*)dstTensor.GetPhyAddr();
    __ubuf__ float* preUb = (__ubuf__ float*)preTensor.GetPhyAddr();
    __ubuf__ float* expSumUb = (__ubuf__ float*)expSumTensor.GetPhyAddr();

    constexpr uint16_t vlSize = 64;
    constexpr uint16_t loopCount = (dSize + vlSize - 1) / vlSize;

    __VEC_SCOPE__
    {
        Reg::RegTensor<float> vSrcRegSum;
        Reg::RegTensor<float> vSrcRegPre;
        Reg::RegTensor<float> vDstRegDiv;
        Reg::RegTensor<float> vScaleBack;
        static constexpr Reg::DivSpecificMode mode = {Reg::MaskMergeMode::ZEROING, false};
        Reg::MaskReg maskAllF32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::Duplicate<float, float>(vScaleBack, FP16_NOQUANT_RANGE_SCALE_BACK);

        for (uint16_t i = 0; i < m; ++i) {
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(vSrcRegSum, expSumUb + i);
            for (uint16_t j = 0; j < loopCount; ++j) {
                const uint32_t colOffset = i * dSize + j * vlSize;
                Reg::DataCopy<float, Reg::LoadDist::DIST_NORM>(vSrcRegPre, preUb + colOffset);
                Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(vSrcRegPre, vSrcRegPre, vScaleBack, maskAllF32);
                Reg::Div<float, &mode>(vDstRegDiv, vSrcRegPre, vSrcRegSum, maskAllF32);
                Reg::DataCopy<float, Reg::StoreDist::DIST_NORM_B32>(dstUb + colOffset, vDstRegDiv, maskAllF32);
            }
        }
    }
}

template <typename T, uint32_t srcD>
__aicore__ inline void InvalidLineUpdate(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
                                         const LocalTensor<T>& maxTensor, const uint16_t m, const uint16_t d,
                                         const T minValue, const T invalidValue)
{
    __ubuf__ T* dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ T* srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ T* maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();

    constexpr uint16_t floatRepSize = 64;
    uint16_t dLoops = d >> 6;

    __VEC_SCOPE__
    {
        Reg::RegTensor<float> vregInvalidValue;
        Reg::RegTensor<float> vregMax;
        Reg::RegTensor<float> vregInput;
        Reg::RegTensor<float> vregInputBrc;

        Reg::MaskReg pregAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregCompare;

        Reg::Duplicate(vregInvalidValue, invalidValue);
        for (uint16_t i = 0; i < m; ++i) {
            Reg::DataCopy<T, Reg::LoadDist::DIST_BRC_B32>(vregMax, maxUb + i);
            Reg::CompareScalar<T, CMPMODE::EQ>(pregCompare, vregMax, minValue, pregAll);
            for (uint16_t j = 0; j < dLoops; ++j) {
                Reg::DataCopy(vregInput, srcUb + i * d + j * floatRepSize);
                Reg::Select(vregInputBrc, vregInvalidValue, vregInput, pregCompare);
                Reg::DataCopy<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)dstUb + i * d + j * floatRepSize,
                                                                vregInputBrc, pregAll);
            }
        }
    }
}

template <typename T>
__aicore__ inline void RowInvalidUpdateVF(const LocalTensor<T>& finalTensor, const LocalTensor<float>& maxTensor,
                                          const uint16_t m, const uint16_t d, int64_t dSize)
{
    __ubuf__ T* finalUb = (__ubuf__ T*)finalTensor.GetPhyAddr();
    __ubuf__ float* maxUb = (__ubuf__ float*)maxTensor.GetPhyAddr();

    constexpr uint16_t floatRepSize = 64; // 64: 一个寄存器可以存储64个float类型数据
    const uint16_t dLoops = d / floatRepSize;
    const uint16_t tailD = d % floatRepSize;
    uint32_t pltTailD = static_cast<uint32_t>(tailD);
    uint16_t hasTail = 0;
    if (tailD > 0) {
        hasTail = 1;
    }

    constexpr uint32_t tmpZero = 0x00000000; // zero value of fp16 and fp32
    const T zeroValue = *((T*)&tmpZero);
    constexpr uint32_t tmpMin = 0xFF7FFFFF; // min value of float
    const float minValue = *((float*)&tmpMin);

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
