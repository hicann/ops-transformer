/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file mhc_pre_sinkhorn_backward_simd_vf.h
 * \brief mhc_pre_sinkhorn_backward
 */

#include "kernel_operator.h"

using namespace AscendC;
using namespace Reg;

namespace {
constexpr int32_t FP32_BYTE_SIZE = 4;
constexpr int32_t UB_BLOCK_BYTE_SIZE = 32;
constexpr int32_t VEC_LENGTH = 256;
constexpr int32_t BLOCKNUM_PER_VL = VEC_LENGTH / UB_BLOCK_BYTE_SIZE;
constexpr int32_t FP32_PER_VL = VEC_LENGTH / FP32_BYTE_SIZE;
} // namespace

template <typename T>
__aicore__ inline void ComputeGradSigmoidVf(LocalTensor<T> gradSigmoidLocal1, LocalTensor<T> gradSigmoidLocal2,
                                            LocalTensor<T> fusedHPre2AndHPost2Local, LocalTensor<T> invRmsLocal,
                                            LocalTensor<T> alphaLocal, LocalTensor<T> biasLocal, uint16_t repeatTimes,
                                            uint32_t totalElements)
{
    __local_mem__ T *gradSigmoidPtr1 = (__local_mem__ T *)gradSigmoidLocal1.GetPhyAddr();
    __local_mem__ T *gradSigmoidPtr2 = (__local_mem__ T *)gradSigmoidLocal2.GetPhyAddr();

    __local_mem__ T *invRmsPtr = (__local_mem__ T *)invRmsLocal.GetPhyAddr();
    __local_mem__ T *fusedHPre2AndHPost2Ptr = (__local_mem__ T *)fusedHPre2AndHPost2Local.GetPhyAddr();
    __local_mem__ T *alphaPtr = (__local_mem__ T *)alphaLocal.GetPhyAddr();
    __local_mem__ T *biasPtr = (__local_mem__ T *)biasLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        RegTensor<T> invRmsBrcbReg;
        RegTensor<T> fusedHPre2AndHPost2Reg;
        RegTensor<T> alphaReg;
        RegTensor<T> biasReg;
        RegTensor<T> gradSigmoidReg1;
        RegTensor<T> gradSigmoidReg2;
        RegTensor<T> mulMaskReg;
        RegTensor<T> tmpReg;
        RegTensor<T> onesReg;
        RegTensor<T> h2ValNormedReg;

        MaskReg mask = Reg::CreateMask<T, MaskPattern::ALL>();

        Reg::DataCopy<T, Reg::LoadDist::DIST_BLK>(alphaReg, alphaPtr);
        Reg::DataCopy<T, Reg::LoadDist::DIST_BLK>(biasReg, biasPtr);

        // 初始化 mulMaskReg 为 [1,1,1,1, 2,2,2,2, 1,1,1,1, 2,2,2,2, ...]
        Reg::Arange(mulMaskReg, 0.f);
        Reg::Muls(mulMaskReg, mulMaskReg, 0.25f, mask);
        Reg::Truncate<T, RoundMode::CAST_FLOOR, Reg::MaskMergeMode::ZEROING>(mulMaskReg, mulMaskReg, mask);

        Reg::Muls(tmpReg, mulMaskReg, 0.5f, mask);
        Reg::Truncate<T, RoundMode::CAST_FLOOR, Reg::MaskMergeMode::ZEROING>(tmpReg, tmpReg, mask);
        Reg::Muls(tmpReg, tmpReg, 2.0f, mask);

        Reg::Sub(mulMaskReg, mulMaskReg, tmpReg, mask);
        Reg::Adds(mulMaskReg, mulMaskReg, 1.0f, mask);

        // 初始化 onesReg 为 [1,1,1,1, ...]
        Reg::Duplicate(onesReg, 1.0f, mask);

        for (uint16_t i = 0; i < repeatTimes; i++) {
            Reg::MaskReg mask1 = Reg::UpdateMask<T>(totalElements);

            Reg::DataCopy<T, Reg::LoadDist::DIST_E2B_B32>(invRmsBrcbReg, invRmsPtr + i * BLOCKNUM_PER_VL);
            Reg::DataCopy(fusedHPre2AndHPost2Reg, fusedHPre2AndHPost2Ptr + i * FP32_PER_VL);

            Reg::Mul(h2ValNormedReg, fusedHPre2AndHPost2Reg, invRmsBrcbReg, mask);
            Reg::Mul(gradSigmoidReg1, h2ValNormedReg, alphaReg, mask);
            Reg::Add(gradSigmoidReg1, gradSigmoidReg1, biasReg, mask);

            Reg::Neg(gradSigmoidReg1, gradSigmoidReg1, mask);
            Reg::Exp(gradSigmoidReg1, gradSigmoidReg1, mask);
            Reg::Adds(gradSigmoidReg1, gradSigmoidReg1, 1.0f, mask);
            Reg::Div(gradSigmoidReg1, onesReg, gradSigmoidReg1, mask);
            Reg::Mul(tmpReg, gradSigmoidReg1, gradSigmoidReg1, mask);
            Reg::Sub(gradSigmoidReg1, gradSigmoidReg1, tmpReg, mask);
            Reg::Mul(gradSigmoidReg1, gradSigmoidReg1, mulMaskReg, mask);

            Reg::Mul(gradSigmoidReg2, gradSigmoidReg1, h2ValNormedReg, mask);

            Reg::DataCopy(gradSigmoidPtr1 + i * FP32_PER_VL, gradSigmoidReg1, mask1);
            Reg::DataCopy(gradSigmoidPtr2 + i * FP32_PER_VL, gradSigmoidReg2, mask1);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void ComputeGradPreVf(LocalTensor<T> gradHPreLocal, LocalTensor<U> xLocal,
                                        LocalTensor<U> gradHinLocal, int32_t n, uint16_t repeatTimes)
{
    __local_mem__ U *xPtr = (__local_mem__ U *)xLocal.GetPhyAddr();
    __local_mem__ U *hinGradPtr = (__local_mem__ U *)gradHinLocal.GetPhyAddr();
    __local_mem__ T *gradHPrePtr = (__local_mem__ T *)gradHPreLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        RegTensor<U> xReg0, xReg1, xReg2, xReg3;
        RegTensor<T> xFp32Reg0, xFp32Reg1, xFp32Reg2, xFp32Reg3;
        RegTensor<T> gradHinCastReg;
        RegTensor<U> tmpReg;
        RegTensor<U> gradHinReg;

        RegTensor<T> gradHPreReg0, gradHPreReg1, gradHPreReg2, gradHPreReg3;

        static constexpr Reg::CastTrait castTrait = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                     Reg::MaskMergeMode::MERGING, RoundMode::CAST_NONE};
        MaskReg mask = Reg::CreateMask<T, MaskPattern::ALL>();

        Reg::Duplicate(gradHPreReg0, 0);
        Reg::Duplicate(gradHPreReg1, 0);
        Reg::Duplicate(gradHPreReg2, 0);
        Reg::Duplicate(gradHPreReg3, 0);

        for (uint16_t i = 0; i < repeatTimes; i++) {
            Reg::DataCopy(xReg0, xPtr + i * FP32_PER_VL);
            Reg::DataCopy(xReg1, xPtr + (i + 1 * repeatTimes) * FP32_PER_VL);
            Reg::DataCopy(xReg2, xPtr + (i + 2 * repeatTimes) * FP32_PER_VL);
            Reg::DataCopy(xReg3, xPtr + (i + 3 * repeatTimes) * FP32_PER_VL);
            Reg::DataCopy(gradHinReg, hinGradPtr + i * FP32_PER_VL);

            Reg::Interleave(xReg0, tmpReg, xReg0, tmpReg);
            Reg::Interleave(xReg1, tmpReg, xReg1, tmpReg);
            Reg::Interleave(xReg2, tmpReg, xReg2, tmpReg);
            Reg::Interleave(xReg3, tmpReg, xReg3, tmpReg);
            Reg::Interleave(gradHinReg, tmpReg, gradHinReg, tmpReg);

            Reg::Cast<T, U, castTrait>(xFp32Reg0, xReg0, mask);
            Reg::Cast<T, U, castTrait>(xFp32Reg1, xReg1, mask);
            Reg::Cast<T, U, castTrait>(xFp32Reg2, xReg2, mask);
            Reg::Cast<T, U, castTrait>(xFp32Reg3, xReg3, mask);
            Reg::Cast<T, U, castTrait>(gradHinCastReg, gradHinReg, mask);

            Reg::Mul(xFp32Reg0, xFp32Reg0, gradHinCastReg, mask);
            Reg::Mul(xFp32Reg1, xFp32Reg1, gradHinCastReg, mask);
            Reg::Mul(xFp32Reg2, xFp32Reg2, gradHinCastReg, mask);
            Reg::Mul(xFp32Reg3, xFp32Reg3, gradHinCastReg, mask);

            Reg::Add(gradHPreReg0, gradHPreReg0, xFp32Reg0, mask);
            Reg::Add(gradHPreReg1, gradHPreReg1, xFp32Reg1, mask);
            Reg::Add(gradHPreReg2, gradHPreReg2, xFp32Reg2, mask);
            Reg::Add(gradHPreReg3, gradHPreReg3, xFp32Reg3, mask);
        }

        Reg::Reduce<Reg::ReduceType::SUM>(gradHPreReg0, gradHPreReg0, mask);
        Reg::Reduce<Reg::ReduceType::SUM>(gradHPreReg1, gradHPreReg1, mask);
        Reg::Reduce<Reg::ReduceType::SUM>(gradHPreReg2, gradHPreReg2, mask);
        Reg::Reduce<Reg::ReduceType::SUM>(gradHPreReg3, gradHPreReg3, mask);

        Reg::DataCopy<T, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(gradHPrePtr, gradHPreReg0, mask);
        Reg::DataCopy<T, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(gradHPrePtr + 1, gradHPreReg1, mask);
        Reg::DataCopy<T, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(gradHPrePtr + 2, gradHPreReg2, mask);
        Reg::DataCopy<T, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(gradHPrePtr + 3, gradHPreReg3, mask);
    }
}

template <typename T, typename U>
__aicore__ inline void ComputeGradXVf(LocalTensor<U> gradXLocal, LocalTensor<U> xLocal, LocalTensor<U> gradHinLocal,
                                      LocalTensor<T> hPreLocal, const T gradInvRmsVal, const T gradRMSNormVal,
                                      const int32_t c, uint16_t repeatTimes)
{
    __local_mem__ U *gradXPtr = (__local_mem__ U *)gradXLocal.GetPhyAddr();
    __local_mem__ U *xPtr = (__local_mem__ U *)xLocal.GetPhyAddr();
    __local_mem__ U *gradHinPtr = (__local_mem__ U *)gradHinLocal.GetPhyAddr();
    __local_mem__ T *hPrePtr = (__local_mem__ T *)hPreLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        RegTensor<U> xReg;
        RegTensor<T> xFp32Reg;
        RegTensor<T> gradXReg;
        RegTensor<U> gradHinReg, tmpUReg;
        RegTensor<T> gradHinCastReg;
        RegTensor<T> preReg;
        RegTensor<U> gradXCastReg, tmpTReg;
        RegTensor<T> tmpReg;

        MaskReg mask = Reg::CreateMask<T, MaskPattern::ALL>();
        MaskReg maskCast = Reg::CreateMask<U, MaskPattern::VL64>();

        static constexpr Reg::CastTrait castTrait1 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                      Reg::MaskMergeMode::MERGING, RoundMode::CAST_NONE};
        static constexpr Reg::CastTrait castTrait2 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                      Reg::MaskMergeMode::MERGING, RoundMode::CAST_RINT};

        Reg::DataCopy<T, Reg::LoadDist::DIST_BRC_B32>(preReg, hPrePtr);

        for (uint16_t i = 0; i < repeatTimes; i++) {
            Reg::DataCopy(xReg, xPtr + i * FP32_PER_VL);
            Reg::DataCopy(gradHinReg, gradHinPtr + i * FP32_PER_VL);

            Reg::Interleave(gradHinReg, tmpUReg, gradHinReg, tmpUReg);
            Reg::Interleave(xReg, tmpUReg, xReg, tmpUReg);
            Reg::Cast<T, U, castTrait1>(gradHinCastReg, gradHinReg, mask);
            Reg::Cast<T, U, castTrait1>(xFp32Reg, xReg, mask);

            Reg::Muls(gradXReg, xFp32Reg, gradInvRmsVal * gradRMSNormVal, mask);
            Reg::Mul(tmpReg, gradHinCastReg, preReg, mask);
            Reg::Add(gradXReg, tmpReg, gradXReg, mask);
            Reg::Cast<U, T, castTrait2>(gradXCastReg, gradXReg, mask);
            Reg::DeInterleave(gradXCastReg, tmpTReg, gradXCastReg, tmpTReg);
            Reg::DataCopy(gradXPtr + i * FP32_PER_VL, gradXCastReg, maskCast);
        }
    }
}
