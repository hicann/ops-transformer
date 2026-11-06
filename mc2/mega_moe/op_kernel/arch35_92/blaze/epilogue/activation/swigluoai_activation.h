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
 * \file swigluoai_activation.h
 * \brief SwiGLUOAI 门控激活 Reg VF 实现。
 */

#ifndef MEGA_MOE_ARCH35_SWIGLUOAI_ACTIVATION_H
#define MEGA_MOE_ARCH35_SWIGLUOAI_ACTIVATION_H

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202))
#include "activation_common.h"
#include "basic_api/reg_compute/kernel_reg_compute_intf.h"

namespace MegaMoeImpl {
namespace Activation {

// SwiGLUOAI VF 调用接口使用的最小参数集。
struct SwiGluOaiParams {
    float clampLimit = 0.0F;
    float alpha = 0.0F;
    float beta = 0.0F;
};

#if (__NPU_ARCH__ != 3510) || !defined(__ASC_FTZ__)
template <typename InputType>
__simd_callee__ inline void LoadSwiGluOaiInputAsFp32(AscendC::Reg::RegTensor<float>& dstReg,
                                                     AscendC::Reg::RegTensor<InputType>& inputReg,
                                                     __ubuf__ InputType* src, AscendC::Reg::MaskReg computeMask)
{
    AscendC::Reg::LoadAlign<InputType, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(inputReg, src);
    AscendC::Reg::Cast<float, InputType, CAST_INPUT_TO_FP32>(dstReg, inputReg, computeMask);
}

__simd_callee__ inline void StoreSwiGluOaiFp32AsBf16(__ubuf__ bfloat16_t* dst,
                                                     AscendC::Reg::RegTensor<bfloat16_t>& outputReg,
                                                     AscendC::Reg::RegTensor<float>& srcReg,
                                                     AscendC::Reg::MaskReg computeMask, AscendC::Reg::MaskReg storeMask)
{
    AscendC::Reg::Cast<bfloat16_t, float, CAST_FP32_TO_BF16>(outputReg, srcReg, computeMask);
    AscendC::Reg::StoreAlign<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(dst, outputReg, storeMask);
}

__simd_callee__ inline void ComputeSwiGluOaiRegFlow(
    AscendC::Reg::RegTensor<float>& outputFp32Reg, AscendC::Reg::RegTensor<float>& gateFp32Reg,
    AscendC::Reg::RegTensor<float>& upFp32Reg, AscendC::Reg::RegTensor<float>& negReg,
    AscendC::Reg::RegTensor<float>& expReg, AscendC::Reg::RegTensor<float>& denominatorReg,
    AscendC::Reg::RegTensor<float>& sigmoidReg, AscendC::Reg::MaskReg computeMask, float clampLimit, float alpha,
    float beta)
{
    AscendC::Reg::Mins(gateFp32Reg, gateFp32Reg, clampLimit, computeMask);
    AscendC::Reg::Mins(upFp32Reg, upFp32Reg, clampLimit, computeMask);
    AscendC::Reg::Maxs(upFp32Reg, upFp32Reg, -clampLimit, computeMask);
    AscendC::Reg::Adds(upFp32Reg, upFp32Reg, beta, computeMask);
    AscendC::Reg::Muls(negReg, gateFp32Reg, -alpha, computeMask);
    AscendC::Reg::Exp(expReg, negReg, computeMask);
    AscendC::Reg::Adds(denominatorReg, expReg, 1.0F, computeMask);
    AscendC::Reg::Div(sigmoidReg, gateFp32Reg, denominatorReg, computeMask);
    AscendC::Reg::Mul(outputFp32Reg, sigmoidReg, upFp32Reg, computeMask);
}

template <typename InputType, bool TopkWeightsPrefetch>
__simd_vf__ inline void RunSwiGLUOAI(GatedActivationTileContext<InputType> context, SwiGluOaiParams params)
{
    __ubuf__ InputType* gate = context.gate;
    __ubuf__ InputType* up = context.up;
    __ubuf__ bfloat16_t* output = context.output;
    __ubuf__ float* topkWeights = context.topkWeights;
    __ubuf__ InputType* gateTail = context.gateTail;
    __ubuf__ InputType* upTail = context.upTail;
    __ubuf__ bfloat16_t* outputTail = context.outputTail;
    __ubuf__ bfloat16_t* additionalPaddingOutput = context.additionalPaddingOutput;

    const uint32_t inputRowStrideElements = context.inputRowStrideElements;
    const uint32_t outputRowStrideElements = context.outputRowStrideElements;
    const uint16_t rowLoopCount = context.rowLoopCount;
    const uint16_t fullVectorLoopCount = context.fullVectorLoopCount;
    const uint16_t needTailVectorCompute = context.needTailVectorCompute;
    const uint16_t needAdditionalPaddingStore = context.needAdditionalPaddingStore;
    uint32_t tailComputeMaskElementCount = context.tailComputeMaskElementCount;
    uint32_t tailStoreMaskElementCount = context.tailStoreMaskElementCount;
    uint32_t additionalPaddingStoreMaskElementCount = context.additionalPaddingStoreMaskElementCount;

    const float clampLimit = params.clampLimit;
    const float alpha = params.alpha;
    const float beta = params.beta;

    AscendC::Reg::RegTensor<InputType> gateReg;
    AscendC::Reg::RegTensor<InputType> upReg;
    AscendC::Reg::RegTensor<float> gateFp32Reg;
    AscendC::Reg::RegTensor<float> upFp32Reg;
    AscendC::Reg::RegTensor<float> negReg;
    AscendC::Reg::RegTensor<float> expReg;
    AscendC::Reg::RegTensor<float> denominatorReg;
    AscendC::Reg::RegTensor<float> sigmoidReg;
    AscendC::Reg::RegTensor<float> outputFp32Reg;
    AscendC::Reg::RegTensor<float> weightReg;
    AscendC::Reg::RegTensor<bfloat16_t> outputReg;
    AscendC::Reg::RegTensor<bfloat16_t> zeroReg;

    AscendC::Reg::MaskReg fullMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg tailComputeMask = AscendC::Reg::UpdateMask<float>(tailComputeMaskElementCount);
    AscendC::Reg::MaskReg tailStoreMask = AscendC::Reg::UpdateMask<float>(tailStoreMaskElementCount);
    AscendC::Reg::MaskReg additionalPaddingStoreMask =
        AscendC::Reg::UpdateMask<bfloat16_t>(additionalPaddingStoreMaskElementCount);

    for (uint16_t rowIndex = 0; rowIndex < rowLoopCount; ++rowIndex) {
        if constexpr (TopkWeightsPrefetch) {
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(
                weightReg, topkWeights + rowIndex * INT32_PER_256B + WEIGHT_INDEX);
        }

        const uint32_t inputRowOffset = static_cast<uint32_t>(rowIndex) * inputRowStrideElements;
        const uint32_t outputRowOffset = static_cast<uint32_t>(rowIndex) * outputRowStrideElements;
        for (uint16_t vectorIndex = 0; vectorIndex < fullVectorLoopCount; ++vectorIndex) {
            const uint32_t vectorOffset = static_cast<uint32_t>(vectorIndex) * VECTOR_LENGTH_FP32;
            LoadSwiGluOaiInputAsFp32<InputType>(gateFp32Reg, gateReg, gate + inputRowOffset + vectorOffset, fullMask);
            LoadSwiGluOaiInputAsFp32<InputType>(upFp32Reg, upReg, up + inputRowOffset + vectorOffset, fullMask);
            ComputeSwiGluOaiRegFlow(outputFp32Reg, gateFp32Reg, upFp32Reg, negReg, expReg, denominatorReg, sigmoidReg,
                                    fullMask, clampLimit, alpha, beta);
            if constexpr (TopkWeightsPrefetch) {
                AscendC::Reg::Mul(outputFp32Reg, outputFp32Reg, weightReg, fullMask);
            }
            StoreSwiGluOaiFp32AsBf16(output + outputRowOffset + vectorOffset, outputReg, outputFp32Reg, fullMask,
                                     fullMask);
        }

        for (uint16_t tailIndex = 0; tailIndex < needTailVectorCompute; ++tailIndex) {
            LoadSwiGluOaiInputAsFp32<InputType>(gateFp32Reg, gateReg, gateTail + inputRowOffset, tailComputeMask);
            LoadSwiGluOaiInputAsFp32<InputType>(upFp32Reg, upReg, upTail + inputRowOffset, tailComputeMask);
            ComputeSwiGluOaiRegFlow(outputFp32Reg, gateFp32Reg, upFp32Reg, negReg, expReg, denominatorReg, sigmoidReg,
                                    tailComputeMask, clampLimit, alpha, beta);
            if constexpr (TopkWeightsPrefetch) {
                AscendC::Reg::Mul(outputFp32Reg, outputFp32Reg, weightReg, tailComputeMask);
            }
            StoreSwiGluOaiFp32AsBf16(outputTail + outputRowOffset, outputReg, outputFp32Reg, tailComputeMask,
                                     tailStoreMask);
        }

        for (uint16_t additionalPaddingIndex = 0; additionalPaddingIndex < needAdditionalPaddingStore;
             ++additionalPaddingIndex) {
            AscendC::Reg::Duplicate(zeroReg, static_cast<bfloat16_t>(0));
            AscendC::Reg::StoreAlign<bfloat16_t>(additionalPaddingOutput + outputRowOffset, zeroReg,
                                                 additionalPaddingStoreMask);
        }
    }
}

#else
template <typename InputType, bool TopkWeightsPrefetch>
__simd_callee__ inline void ComputeSwiGluOai(__ubuf__ uint32_t*& output, __ubuf__ InputType*& gate,
                                             __ubuf__ InputType*& up, vector_float& weight, vector_bool computeMask,
                                             vector_bool storeMask, SwiGluOaiParams params)
{
    vector_float gateReg;
    vector_float upReg;
    vector_float resultReg;
    LoadActivationInput<InputType>(gateReg, gate, computeMask);
    LoadActivationInput<InputType>(upReg, up, computeMask);
    asc_min_scalar(gateReg, gateReg, params.clampLimit, computeMask);
    asc_min_scalar(upReg, upReg, params.clampLimit, computeMask);
    asc_max_scalar(upReg, upReg, -params.clampLimit, computeMask);
    asc_add_scalar(upReg, upReg, params.beta, computeMask);
    asc_mul_scalar(resultReg, gateReg, -params.alpha, computeMask);
    asc_exp(resultReg, resultReg, computeMask);
    asc_add_scalar(resultReg, resultReg, 1.0F, computeMask);
    asc_div(resultReg, gateReg, resultReg, computeMask);
    asc_mul(resultReg, resultReg, upReg, computeMask);
    if constexpr (TopkWeightsPrefetch) {
        asc_mul(resultReg, resultReg, weight, computeMask);
    }
    StoreActivationOutput(output, resultReg, computeMask, storeMask);
}

template <typename InputType, bool TopkWeightsPrefetch>
__simd_vf__ inline void RunSwiGLUOAI(GatedActivationTileContext<InputType> context, SwiGluOaiParams params)
{
    vector_bool fullMask = asc_create_mask_b32(PAT_ALL);
    vector_bool tailComputeMask;
    vector_bool tailStoreMask;
    if (context.needTailVectorCompute != 0U) {
        uint32_t tailComputeCount = context.tailComputeMaskElementCount;
        tailComputeMask = asc_update_mask_b32(tailComputeCount);
        if (context.tailStoreMaskElementCount == VECTOR_LENGTH_FP32) {
            tailStoreMask = fullMask;
        } else {
            uint32_t tailStoreCount = context.tailStoreMaskElementCount;
            tailStoreMask = asc_update_mask_b32(tailStoreCount);
        }
    }
    vector_bool additionalPaddingStoreMask;
    if (context.needAdditionalPaddingStore != 0U) {
        uint32_t paddingStoreCount = context.additionalPaddingStoreMaskElementCount;
        additionalPaddingStoreMask = asc_update_mask_b16(paddingStoreCount);
    }
    vector_float weightReg;
    __ubuf__ float* topkWeights = context.topkWeights;
    if constexpr (TopkWeightsPrefetch) {
        topkWeights += WEIGHT_INDEX;
    }

    for (uint16_t rowIndex = 0; rowIndex < context.rowLoopCount; ++rowIndex) {
        if constexpr (TopkWeightsPrefetch) {
            asc_loadalign_brc_elem_postupdate(weightReg, topkWeights, INT32_PER_256B);
        }
        const uint32_t inputRowOffset = static_cast<uint32_t>(rowIndex) * context.inputRowStrideElements;
        const uint32_t outputRowOffset = static_cast<uint32_t>(rowIndex) * context.outputRowStrideElements;
        __ubuf__ InputType* gate = context.gate + inputRowOffset;
        __ubuf__ InputType* up = context.up + inputRowOffset;
        __ubuf__ uint32_t* output = reinterpret_cast<__ubuf__ uint32_t*>(context.output + outputRowOffset);
        for (uint16_t vectorIndex = 0; vectorIndex < context.fullVectorLoopCount; ++vectorIndex) {
            ComputeSwiGluOai<InputType, TopkWeightsPrefetch>(output, gate, up, weightReg, fullMask, fullMask, params);
        }
        if (context.needTailVectorCompute != 0U) {
            ComputeSwiGluOai<InputType, TopkWeightsPrefetch>(output, gate, up, weightReg, tailComputeMask,
                                                             tailStoreMask, params);
        }
        if (context.needAdditionalPaddingStore != 0U) {
            vector_bfloat16_t zeroReg;
            asc_duplicate_scalar(zeroReg, static_cast<bfloat16_t>(0), additionalPaddingStoreMask);
            asc_storealign(reinterpret_cast<__ubuf__ bfloat16_t*>(output), zeroReg, additionalPaddingStoreMask);
        }
    }
}
#endif

} // namespace Activation
} // namespace MegaMoeImpl

#endif // Arch35/Arch92 激活核函数
#endif // MEGA_MOE_ARCH35_SWIGLUOAI_ACTIVATION_H
