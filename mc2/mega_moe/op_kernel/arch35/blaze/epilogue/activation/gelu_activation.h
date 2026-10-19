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

#if defined(__DAV_C310__)
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

} // namespace Activation
} // namespace MegaMoeImpl

#endif // __DAV_C310__ 条件编译结束
#endif // MEGA_MOE_ARCH35_GELU_ACTIVATION_H 头文件保护结束
