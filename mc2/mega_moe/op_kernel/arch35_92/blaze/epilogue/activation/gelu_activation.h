/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_ARCH35_GELU_ACTIVATION_H
#define MEGA_MOE_ARCH35_GELU_ACTIVATION_H

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202))
#include "activation_common.h"

namespace MegaMoeImpl {
namespace Activation {

template <typename InputType>
struct GeluActivationTileContext {
    __ubuf__ InputType* input;
    __ubuf__ bfloat16_t* output;
    __ubuf__ float* topkWeights;
    uint32_t rowStrideElements;
    uint16_t rowCount;
    uint16_t validColumnCount;
};

// 与 arch35 独立 GELU 算子使用相同的 tanh 近似计算。
constexpr float GELU_NEG_SQRT_EIGHT_OVER_PI = -1.595769121 * 0.044715;
constexpr float GELU_CUBIC_RECIPROCAL = 1 / 0.044715;

#if (__NPU_ARCH__ != 3510) || !defined(__ASC_FTZ__)
template <typename InputType, bool TopkWeightsPrefetch>
__simd_vf__ inline void RunGeluActivation(GeluActivationTileContext<InputType> context)
{
    AscendC::Reg::RegTensor<InputType> inputReg;
    AscendC::Reg::RegTensor<float> x;
    AscendC::Reg::RegTensor<float> square;
    AscendC::Reg::RegTensor<float> cubic;
    AscendC::Reg::RegTensor<float> result;
    AscendC::Reg::RegTensor<float> weight;
    AscendC::Reg::RegTensor<bfloat16_t> outputReg;
    AscendC::Reg::MaskReg all = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();

    for (uint16_t row = 0; row < context.rowCount; ++row) {
        if constexpr (TopkWeightsPrefetch) {
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(
                weight, context.topkWeights + row * INT32_PER_256B + WEIGHT_INDEX);
        }
        __ubuf__ InputType* inputRow = context.input + row * context.rowStrideElements;
        __ubuf__ bfloat16_t* outputRow = context.output + row * context.rowStrideElements;
        for (uint16_t column = 0; column < context.rowStrideElements; column += VECTOR_LENGTH_FP32) {
            uint32_t valid = 0;
            if (column < context.validColumnCount) {
                valid = context.validColumnCount - column;
                valid = valid < VECTOR_LENGTH_FP32 ? valid : VECTOR_LENGTH_FP32;
            }
            if (valid == 0U) {
                AscendC::Reg::Duplicate(outputReg, static_cast<bfloat16_t>(0));
            } else {
                AscendC::Reg::MaskReg active = AscendC::Reg::UpdateMask<float>(valid);
                // 先将整个向量读入寄存器，再用 BF16 结果覆盖原输入位置。
                AscendC::Reg::LoadAlign<InputType, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(inputReg,
                                                                                            inputRow + column);
                AscendC::Reg::Cast<float, InputType, CAST_INPUT_TO_FP32>(x, inputReg, active);
                AscendC::Reg::Mul(square, x, x, active);
                AscendC::Reg::Mul(cubic, square, x, active);
                AscendC::Reg::Axpy(cubic, x, GELU_CUBIC_RECIPROCAL, active);
                AscendC::Reg::Muls(cubic, cubic, GELU_NEG_SQRT_EIGHT_OVER_PI, active);
                AscendC::Reg::Exp(cubic, cubic, active);
                AscendC::Reg::Adds(cubic, cubic, 1.0F, active);
                AscendC::Reg::Div(result, x, cubic, active);
                if constexpr (TopkWeightsPrefetch) {
                    AscendC::Reg::Mul(result, result, weight, active);
                }
                AscendC::Reg::Cast<bfloat16_t, float, CAST_FP32_TO_BF16>(outputReg, result, active);
                // ZEROING 转换将末尾无效列补零，确保 MX 分组中的无效列为零。
            }
            AscendC::Reg::StoreAlign<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(outputRow + column, outputReg,
                                                                                         all);
        }
    }
}

#else
// FTZ开启时原Reg Exp/Div直接使用指令；关闭FTZ的配置保留原VF及其补偿流程。
template <typename InputType, bool TopkWeightsPrefetch>
__simd_callee__ inline void ComputeGeluVector(__ubuf__ uint32_t*& output, __ubuf__ InputType*& input,
                                              vector_float& weight, vector_bool active, vector_bool storeMask)
{
    vector_float x;
    vector_float square;
    vector_float cubic;
    vector_float result;
    LoadActivationInput<InputType>(x, input, active);
    asc_mul(square, x, x, active);
    asc_mul(cubic, square, x, active);
    asc_axpy(cubic, x, GELU_CUBIC_RECIPROCAL, active);
    asc_mul_scalar(cubic, cubic, GELU_NEG_SQRT_EIGHT_OVER_PI, active);
    asc_exp(cubic, cubic, active);
    asc_add_scalar(cubic, cubic, 1.0F, active);
    asc_div(result, x, cubic, active);
    if constexpr (TopkWeightsPrefetch) {
        asc_mul(result, result, weight, active);
    }
    StoreActivationOutput(output, result, active, storeMask);
}

template <typename InputType, bool TopkWeightsPrefetch>
__simd_vf__ inline void RunGeluActivation(GeluActivationTileContext<InputType> context)
{
    constexpr uint32_t OUTPUT_STEP = VECTOR_LENGTH_FP32 * sizeof(bfloat16_t) / sizeof(uint32_t);
    const uint32_t fullVectorCount = context.validColumnCount / VECTOR_LENGTH_FP32;
    const uint32_t tailElementCount = context.validColumnCount % VECTOR_LENGTH_FP32;
    const uint32_t paddingVectorCount =
        context.rowStrideElements / VECTOR_LENGTH_FP32 - fullVectorCount - (tailElementCount != 0U ? 1U : 0U);
    vector_bool all = asc_create_mask_b32(PAT_ALL);
    vector_bool tailMask;
    if (tailElementCount != 0U) {
        uint32_t remaining = tailElementCount;
        tailMask = asc_update_mask_b32(remaining);
    }
    vector_float weight;

    for (uint16_t row = 0; row < context.rowCount; ++row) {
        if constexpr (TopkWeightsPrefetch) {
            asc_loadalign_brc_elem(weight, context.topkWeights + row * INT32_PER_256B + WEIGHT_INDEX);
        }
        __ubuf__ InputType* input = context.input + row * context.rowStrideElements;
        __ubuf__ uint32_t* output =
            reinterpret_cast<__ubuf__ uint32_t*>(context.output + row * context.rowStrideElements);
        for (uint32_t vectorIndex = 0; vectorIndex < fullVectorCount; ++vectorIndex) {
            ComputeGeluVector<InputType, TopkWeightsPrefetch>(output, input, weight, all, all);
        }
        if (tailElementCount != 0U) {
            ComputeGeluVector<InputType, TopkWeightsPrefetch>(output, input, weight, tailMask, all);
        }
        if (paddingVectorCount != 0U) {
            vector_bfloat16_t outputReg;
            asc_duplicate_scalar(outputReg, static_cast<bfloat16_t>(0), all);
            for (uint32_t vectorIndex = 0; vectorIndex < paddingVectorCount; ++vectorIndex) {
                asc_storealign_pack_postupdate(output, reinterpret_cast<vector_uint32_t&>(outputReg), OUTPUT_STEP, all);
            }
        }
    }
}
#endif

} // namespace Activation
} // namespace MegaMoeImpl

#endif // __NPU_ARCH__ 条件编译结束
#endif // MEGA_MOE_ARCH35_GELU_ACTIVATION_H 头文件保护结束
