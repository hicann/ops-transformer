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
 * \file situglu_activation.h
 * \brief SiTUGLU DEFAULT 和 LINEAR 门控激活实现。
 */

#ifndef MEGA_MOE_ARCH35_SITUGLU_ACTIVATION_H
#define MEGA_MOE_ARCH35_SITUGLU_ACTIVATION_H

#if defined(__DAV_C310__)
#include "activation_common.h"

/*
 * SiTUGLU 两段式 tanh：|x| < 0.6 使用多项式路径，其余使用 sigmoid 分解路径。
 * 宏在 SiTUGLU 的 VecScope 内原位展开，保持完整寄存器生命周期和指令顺序。
 */
#define MEGA_MOE_SITUGLU_COMPUTE_TANH_TWOPATH(result, zReg, msk, oneReg, absReg, sqrReg, polyReg, tmpReg, expReg, \
                                              c1Reg, c2Reg, cmpMask) \
    do { \
        AscendC::Reg::Mul(sqrReg, zReg, zReg, msk); \
        AscendC::Reg::Muls(polyReg, sqrReg, 0.0157396831f, msk); \
        AscendC::Reg::Adds(polyReg, polyReg, -0.0523039624f, msk); \
        AscendC::Reg::FusedMulDstAdd(polyReg, sqrReg, c2Reg, msk); \
        AscendC::Reg::FusedMulDstAdd(polyReg, sqrReg, c1Reg, msk); \
        AscendC::Reg::Mul(polyReg, polyReg, sqrReg, msk); \
        AscendC::Reg::FusedMulDstAdd(polyReg, zReg, zReg, msk); \
        AscendC::Reg::Muls(expReg, zReg, -2.0f, msk); \
        AscendC::Reg::Exp(expReg, expReg, msk); \
        AscendC::Reg::Adds(expReg, expReg, 1.0f, msk); \
        AscendC::Reg::Div(tmpReg, oneReg, expReg, msk); \
        AscendC::Reg::Muls(tmpReg, tmpReg, 2.0f, msk); \
        AscendC::Reg::Adds(tmpReg, tmpReg, -1.0f, msk); \
        AscendC::Reg::Abs(absReg, zReg, msk); \
        AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::GE>(cmpMask, absReg, 0.60000002384185791016f, msk); \
        AscendC::Reg::Select(result, tmpReg, polyReg, cmpMask); \
    } while (0)

namespace MegaMoeImpl {
namespace Activation {

// SiTUGLU 直接调用接口使用的最小参数集。
struct SituGluParams {
    float clampLimit;
    float beta;
    float invBeta;
    float alpha;
    float invAlpha;
};

template <typename InputType, bool FuseTopkWeight, bool IsLinear>
__aicore__ inline void RunSiTUGLU(const GatedActivationTileContext<InputType> &context, const SituGluParams &params)
{
    const float tanhC1 = -0.333327681f;
    const float tanhC2 = 0.133152977f;
    const float scalarOne = 1.0f;
    const float negScalarOne = -1.0f;
    bfloat16_t zeroValue = 0;
    const uint16_t rowLoopCount = context.rowLoopCount;
    const uint16_t fullVectorLoopCount = context.fullVectorLoopCount;
    const uint16_t needTailVectorCompute = context.needTailVectorCompute;
    const uint16_t needAdditionalPaddingStore = context.needAdditionalPaddingStore;
    uint32_t tailComputeMaskElementCount = context.tailComputeMaskElementCount;
    uint32_t tailStoreMaskElementCount = context.tailStoreMaskElementCount;
    uint32_t additionalPaddingStoreMaskElementCount = context.additionalPaddingStoreMaskElementCount;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<InputType> gateReg;
        AscendC::Reg::RegTensor<InputType> upReg;
        AscendC::Reg::RegTensor<float> gateFp32Reg;
        AscendC::Reg::RegTensor<float> upFp32Reg;
        AscendC::Reg::RegTensor<float> outputFp32Reg;
        AscendC::Reg::RegTensor<float> weightReg;
        AscendC::Reg::RegTensor<bfloat16_t> outputReg;
        AscendC::Reg::RegTensor<bfloat16_t> zeroReg;
        AscendC::Reg::RegTensor<float> negReg;
        AscendC::Reg::RegTensor<float> expReg;
        AscendC::Reg::RegTensor<float> denominatorReg;
        AscendC::Reg::RegTensor<float> sigmoidReg;
        AscendC::Reg::RegTensor<float> zReg;
        AscendC::Reg::RegTensor<float> betaReg;
        AscendC::Reg::RegTensor<float> oneReg;
        AscendC::Reg::RegTensor<float> absReg;
        AscendC::Reg::RegTensor<float> squareReg;
        AscendC::Reg::RegTensor<float> polynomialReg;
        AscendC::Reg::RegTensor<float> tanhTemporaryReg;
        AscendC::Reg::RegTensor<float> c1Reg;
        AscendC::Reg::RegTensor<float> c2Reg;
        AscendC::Reg::MaskReg compareMask;
        AscendC::Reg::MaskReg fullMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg tailComputeMask = AscendC::Reg::UpdateMask<float>(tailComputeMaskElementCount);
        AscendC::Reg::MaskReg tailStoreMask = AscendC::Reg::UpdateMask<float>(tailStoreMaskElementCount);
        AscendC::Reg::MaskReg additionalPaddingStoreMask =
            AscendC::Reg::UpdateMask<bfloat16_t>(additionalPaddingStoreMaskElementCount);
        AscendC::Reg::Duplicate(oneReg, scalarOne);
        AscendC::Reg::Duplicate(c1Reg, tanhC1);
        AscendC::Reg::Duplicate(c2Reg, tanhC2);

        for (uint16_t rowIndex = 0; rowIndex < rowLoopCount; rowIndex++) {
            if constexpr (FuseTopkWeight) {
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(
                    weightReg, context.topkWeights + rowIndex * INT32_PER_256B + WEIGHT_INDEX);
            }
            for (uint16_t vectorIndex = 0; vectorIndex < fullVectorLoopCount; vectorIndex++) {
                AscendC::Reg::AddrReg inputOffset = AscendC::Reg::CreateAddrReg<InputType>(
                    rowIndex, context.inputRowStrideElements, vectorIndex, VECTOR_LENGTH_FP32);
                AscendC::Reg::DataCopy<InputType, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(gateReg, context.gate,
                                                                                           inputOffset);
                AscendC::Reg::DataCopy<InputType, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(upReg, context.up,
                                                                                           inputOffset);
                AscendC::Reg::Cast<float, InputType, CAST_INPUT_TO_FP32>(gateFp32Reg, gateReg, fullMask);
                AscendC::Reg::Cast<float, InputType, CAST_INPUT_TO_FP32>(upFp32Reg, upReg, fullMask);
                AscendC::Reg::Mins(gateFp32Reg, gateFp32Reg, params.clampLimit, fullMask);
                AscendC::Reg::Mins(upFp32Reg, upFp32Reg, params.clampLimit, fullMask);
                AscendC::Reg::Maxs(upFp32Reg, upFp32Reg, -params.clampLimit, fullMask);

                AscendC::Reg::Muls(zReg, gateFp32Reg, params.invBeta, fullMask);
                MEGA_MOE_SITUGLU_COMPUTE_TANH_TWOPATH(sigmoidReg, zReg, fullMask, oneReg, absReg, squareReg,
                                                      polynomialReg, tanhTemporaryReg, expReg, c1Reg, c2Reg,
                                                      compareMask);
                AscendC::Reg::Muls(negReg, gateFp32Reg, negScalarOne, fullMask);
                AscendC::Reg::Exp(expReg, negReg, fullMask);
                AscendC::Reg::Adds(denominatorReg, expReg, scalarOne, fullMask);
                AscendC::Reg::Duplicate(betaReg, params.beta, fullMask);
                AscendC::Reg::Div(expReg, betaReg, denominatorReg, fullMask);
                AscendC::Reg::Mul(outputFp32Reg, sigmoidReg, expReg, fullMask);
                if constexpr (IsLinear) {
                    AscendC::Reg::Muls(gateFp32Reg, outputFp32Reg, scalarOne, fullMask);
                    AscendC::Reg::Muls(zReg, upFp32Reg, params.invAlpha, fullMask);
                    MEGA_MOE_SITUGLU_COMPUTE_TANH_TWOPATH(sigmoidReg, zReg, fullMask, oneReg, absReg, squareReg,
                                                          polynomialReg, tanhTemporaryReg, expReg, c1Reg, c2Reg,
                                                          compareMask);
                    AscendC::Reg::Muls(sigmoidReg, sigmoidReg, params.alpha, fullMask);
                    AscendC::Reg::Mul(outputFp32Reg, gateFp32Reg, sigmoidReg, fullMask);
                } else {
                    AscendC::Reg::Mul(outputFp32Reg, outputFp32Reg, upFp32Reg, fullMask);
                }
                if constexpr (FuseTopkWeight) {
                    AscendC::Reg::Mul(outputFp32Reg, outputFp32Reg, weightReg, fullMask);
                }
                AscendC::Reg::Cast<bfloat16_t, float, CAST_FP32_TO_BF16>(outputReg, outputFp32Reg, fullMask);
                AscendC::Reg::AddrReg outputOffset = AscendC::Reg::CreateAddrReg<bfloat16_t>(
                    rowIndex, context.outputRowStrideElements, vectorIndex, VECTOR_LENGTH_FP32);
                AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(context.output, outputReg,
                                                                                           outputOffset, fullMask);
            }

            AscendC::Reg::AddrReg tailInputOffset =
                AscendC::Reg::CreateAddrReg<InputType>(rowIndex, context.inputRowStrideElements);
            AscendC::Reg::AddrReg tailOutputOffset =
                AscendC::Reg::CreateAddrReg<bfloat16_t>(rowIndex, context.outputRowStrideElements);
            for (uint16_t tailIndex = 0; tailIndex < needTailVectorCompute; tailIndex++) {
                AscendC::Reg::DataCopy<InputType, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(gateReg, context.gateTail,
                                                                                           tailInputOffset);
                AscendC::Reg::DataCopy<InputType, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(upReg, context.upTail,
                                                                                           tailInputOffset);
                AscendC::Reg::Cast<float, InputType, CAST_INPUT_TO_FP32>(gateFp32Reg, gateReg, tailComputeMask);
                AscendC::Reg::Cast<float, InputType, CAST_INPUT_TO_FP32>(upFp32Reg, upReg, tailComputeMask);
                AscendC::Reg::Mins(gateFp32Reg, gateFp32Reg, params.clampLimit, tailComputeMask);
                AscendC::Reg::Mins(upFp32Reg, upFp32Reg, params.clampLimit, tailComputeMask);
                AscendC::Reg::Maxs(upFp32Reg, upFp32Reg, -params.clampLimit, tailComputeMask);

                AscendC::Reg::Muls(zReg, gateFp32Reg, params.invBeta, tailComputeMask);
                MEGA_MOE_SITUGLU_COMPUTE_TANH_TWOPATH(sigmoidReg, zReg, tailComputeMask, oneReg, absReg, squareReg,
                                                      polynomialReg, tanhTemporaryReg, expReg, c1Reg, c2Reg,
                                                      compareMask);
                AscendC::Reg::Muls(negReg, gateFp32Reg, negScalarOne, tailComputeMask);
                AscendC::Reg::Exp(expReg, negReg, tailComputeMask);
                AscendC::Reg::Adds(denominatorReg, expReg, scalarOne, tailComputeMask);
                AscendC::Reg::Duplicate(betaReg, params.beta, tailComputeMask);
                AscendC::Reg::Div(expReg, betaReg, denominatorReg, tailComputeMask);
                AscendC::Reg::Mul(outputFp32Reg, sigmoidReg, expReg, tailComputeMask);
                if constexpr (IsLinear) {
                    AscendC::Reg::Muls(gateFp32Reg, outputFp32Reg, scalarOne, tailComputeMask);
                    AscendC::Reg::Muls(zReg, upFp32Reg, params.invAlpha, tailComputeMask);
                    MEGA_MOE_SITUGLU_COMPUTE_TANH_TWOPATH(sigmoidReg, zReg, tailComputeMask, oneReg, absReg, squareReg,
                                                          polynomialReg, tanhTemporaryReg, expReg, c1Reg, c2Reg,
                                                          compareMask);
                    AscendC::Reg::Muls(sigmoidReg, sigmoidReg, params.alpha, tailComputeMask);
                    AscendC::Reg::Mul(outputFp32Reg, gateFp32Reg, sigmoidReg, tailComputeMask);
                } else {
                    AscendC::Reg::Mul(outputFp32Reg, outputFp32Reg, upFp32Reg, tailComputeMask);
                }
                if constexpr (FuseTopkWeight) {
                    AscendC::Reg::Mul(outputFp32Reg, outputFp32Reg, weightReg, tailComputeMask);
                }
                AscendC::Reg::Cast<bfloat16_t, float, CAST_FP32_TO_BF16>(outputReg, outputFp32Reg, tailComputeMask);
                AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                    context.outputTail, outputReg, tailOutputOffset, tailStoreMask);
            }
            for (uint16_t additionalPaddingIndex = 0; additionalPaddingIndex < needAdditionalPaddingStore;
                 additionalPaddingIndex++) {
                AscendC::Reg::Duplicate(zeroReg, zeroValue);
                AscendC::Reg::DataCopy<bfloat16_t>(context.additionalPaddingOutput, zeroReg, tailOutputOffset,
                                                   additionalPaddingStoreMask);
            }
        }
    }
}

} // namespace Activation
} // namespace MegaMoeImpl

#undef MEGA_MOE_SITUGLU_COMPUTE_TANH_TWOPATH

#endif // defined(__DAV_C310__)
#endif // MEGA_MOE_ARCH35_SITUGLU_ACTIVATION_H
