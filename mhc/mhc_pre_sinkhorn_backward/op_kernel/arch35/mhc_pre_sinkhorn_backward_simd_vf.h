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
using namespace MicroAPI;

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

        MaskReg mask = MicroAPI::CreateMask<T, MaskPattern::ALL>();

        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BLK>(alphaReg, alphaPtr);
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BLK>(biasReg, biasPtr);

        // 初始化 mulMaskReg 为 [1,1,1,1, 2,2,2,2, 1,1,1,1, 2,2,2,2, ...]
        MicroAPI::Arange(mulMaskReg, 0.f);
        MicroAPI::Muls(mulMaskReg, mulMaskReg, 0.25f, mask);
        MicroAPI::Truncate<T, RoundMode::CAST_FLOOR, MicroAPI::MaskMergeMode::ZEROING>(mulMaskReg, mulMaskReg, mask);

        MicroAPI::Muls(tmpReg, mulMaskReg, 0.5f, mask);
        MicroAPI::Truncate<T, RoundMode::CAST_FLOOR, MicroAPI::MaskMergeMode::ZEROING>(tmpReg, tmpReg, mask);
        MicroAPI::Muls(tmpReg, tmpReg, 2.0f, mask);

        MicroAPI::Sub(mulMaskReg, mulMaskReg, tmpReg, mask);
        MicroAPI::Adds(mulMaskReg, mulMaskReg, 1.0f, mask);

        // 初始化 onesReg 为 [1,1,1,1, ...]
        MicroAPI::Duplicate(onesReg, 1.0f, mask);

        for (uint16_t i = 0; i < repeatTimes; i++) {
            MicroAPI::MaskReg mask1 = MicroAPI::UpdateMask<T>(totalElements);

            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_E2B_B32>(invRmsBrcbReg, invRmsPtr + i * BLOCKNUM_PER_VL);
            MicroAPI::DataCopy(fusedHPre2AndHPost2Reg, fusedHPre2AndHPost2Ptr + i * FP32_PER_VL);

            MicroAPI::Mul(h2ValNormedReg, fusedHPre2AndHPost2Reg, invRmsBrcbReg, mask);
            MicroAPI::Mul(gradSigmoidReg1, h2ValNormedReg, alphaReg, mask);
            MicroAPI::Add(gradSigmoidReg1, gradSigmoidReg1, biasReg, mask);

            MicroAPI::Neg(gradSigmoidReg1, gradSigmoidReg1, mask);
            MicroAPI::Exp(gradSigmoidReg1, gradSigmoidReg1, mask);
            MicroAPI::Adds(gradSigmoidReg1, gradSigmoidReg1, 1.0f, mask);
            MicroAPI::Div(gradSigmoidReg1, onesReg, gradSigmoidReg1, mask);
            MicroAPI::Mul(tmpReg, gradSigmoidReg1, gradSigmoidReg1, mask);
            MicroAPI::Sub(gradSigmoidReg1, gradSigmoidReg1, tmpReg, mask);
            MicroAPI::Mul(gradSigmoidReg1, gradSigmoidReg1, mulMaskReg, mask);

            MicroAPI::Mul(gradSigmoidReg2, gradSigmoidReg1, h2ValNormedReg, mask);

            MicroAPI::DataCopy(gradSigmoidPtr1 + i * FP32_PER_VL, gradSigmoidReg1, mask1);
            MicroAPI::DataCopy(gradSigmoidPtr2 + i * FP32_PER_VL, gradSigmoidReg2, mask1);
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

        static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                          MicroAPI::MaskMergeMode::MERGING, RoundMode::CAST_NONE};
        MaskReg mask = MicroAPI::CreateMask<T, MaskPattern::ALL>();

        MicroAPI::Duplicate(gradHPreReg0, 0);
        MicroAPI::Duplicate(gradHPreReg1, 0);
        MicroAPI::Duplicate(gradHPreReg2, 0);
        MicroAPI::Duplicate(gradHPreReg3, 0);

        for (uint16_t i = 0; i < repeatTimes; i++) {
            MicroAPI::DataCopy(xReg0, xPtr + i * FP32_PER_VL);
            MicroAPI::DataCopy(xReg1, xPtr + (i + 1 * repeatTimes) * FP32_PER_VL);
            MicroAPI::DataCopy(xReg2, xPtr + (i + 2 * repeatTimes) * FP32_PER_VL);
            MicroAPI::DataCopy(xReg3, xPtr + (i + 3 * repeatTimes) * FP32_PER_VL);
            MicroAPI::DataCopy(gradHinReg, hinGradPtr + i * FP32_PER_VL);

            MicroAPI::Interleave(xReg0, tmpReg, xReg0, tmpReg);
            MicroAPI::Interleave(xReg1, tmpReg, xReg1, tmpReg);
            MicroAPI::Interleave(xReg2, tmpReg, xReg2, tmpReg);
            MicroAPI::Interleave(xReg3, tmpReg, xReg3, tmpReg);
            MicroAPI::Interleave(gradHinReg, tmpReg, gradHinReg, tmpReg);

            MicroAPI::Cast<T, U, castTrait>(xFp32Reg0, xReg0, mask);
            MicroAPI::Cast<T, U, castTrait>(xFp32Reg1, xReg1, mask);
            MicroAPI::Cast<T, U, castTrait>(xFp32Reg2, xReg2, mask);
            MicroAPI::Cast<T, U, castTrait>(xFp32Reg3, xReg3, mask);
            MicroAPI::Cast<T, U, castTrait>(gradHinCastReg, gradHinReg, mask);

            MicroAPI::Mul(xFp32Reg0, xFp32Reg0, gradHinCastReg, mask);
            MicroAPI::Mul(xFp32Reg1, xFp32Reg1, gradHinCastReg, mask);
            MicroAPI::Mul(xFp32Reg2, xFp32Reg2, gradHinCastReg, mask);
            MicroAPI::Mul(xFp32Reg3, xFp32Reg3, gradHinCastReg, mask);

            MicroAPI::Add(gradHPreReg0, gradHPreReg0, xFp32Reg0, mask);
            MicroAPI::Add(gradHPreReg1, gradHPreReg1, xFp32Reg1, mask);
            MicroAPI::Add(gradHPreReg2, gradHPreReg2, xFp32Reg2, mask);
            MicroAPI::Add(gradHPreReg3, gradHPreReg3, xFp32Reg3, mask);
        }

        MicroAPI::Reduce<MicroAPI::ReduceType::SUM>(gradHPreReg0, gradHPreReg0, mask);
        MicroAPI::Reduce<MicroAPI::ReduceType::SUM>(gradHPreReg1, gradHPreReg1, mask);
        MicroAPI::Reduce<MicroAPI::ReduceType::SUM>(gradHPreReg2, gradHPreReg2, mask);
        MicroAPI::Reduce<MicroAPI::ReduceType::SUM>(gradHPreReg3, gradHPreReg3, mask);

        MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(gradHPrePtr, gradHPreReg0, mask);
        MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(gradHPrePtr + 1, gradHPreReg1, mask);
        MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(gradHPrePtr + 2, gradHPreReg2, mask);
        MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(gradHPrePtr + 3, gradHPreReg3, mask);
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

        MaskReg mask = MicroAPI::CreateMask<T, MaskPattern::ALL>();
        MaskReg maskCast = MicroAPI::CreateMask<U, MaskPattern::VL64>();

        static constexpr MicroAPI::CastTrait castTrait1 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                           MicroAPI::MaskMergeMode::MERGING, RoundMode::CAST_NONE};
        static constexpr MicroAPI::CastTrait castTrait2 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                           MicroAPI::MaskMergeMode::MERGING, RoundMode::CAST_RINT};

        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(preReg, hPrePtr);

        for (uint16_t i = 0; i < repeatTimes; i++) {
            MicroAPI::DataCopy(xReg, xPtr + i * FP32_PER_VL);
            MicroAPI::DataCopy(gradHinReg, gradHinPtr + i * FP32_PER_VL);

            MicroAPI::Interleave(gradHinReg, tmpUReg, gradHinReg, tmpUReg);
            MicroAPI::Interleave(xReg, tmpUReg, xReg, tmpUReg);
            MicroAPI::Cast<T, U, castTrait1>(gradHinCastReg, gradHinReg, mask);
            MicroAPI::Cast<T, U, castTrait1>(xFp32Reg, xReg, mask);

            MicroAPI::Muls(gradXReg, xFp32Reg, gradInvRmsVal * gradRMSNormVal, mask);
            MicroAPI::Mul(tmpReg, gradHinCastReg, preReg, mask);
            MicroAPI::Add(gradXReg, tmpReg, gradXReg, mask);
            MicroAPI::Cast<U, T, castTrait2>(gradXCastReg, gradXReg, mask);
            MicroAPI::DeInterleave(gradXCastReg, tmpTReg, gradXCastReg, tmpTReg);
            MicroAPI::DataCopy(gradXPtr + i * FP32_PER_VL, gradXCastReg, maskCast);
        }
    }
}
