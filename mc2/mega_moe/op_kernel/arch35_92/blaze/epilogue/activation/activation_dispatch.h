/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_ACTIVATION_DISPATCH_H
#define MEGA_MOE_ACTIVATION_DISPATCH_H

#include "gelu_activation.h"
#include "situglu_activation.h"
#include "swiglu_activation.h"
#include "swigluoai_activation.h"

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202))
namespace MegaMoeImpl {
namespace Activation {
using ActMode = MegaMoeActMode;
using ActSubMode = MegaMoeActSubMode;

// MX 和 BF16 共用激活尾段配置，计算与存储掩码保持一致。
template <typename InputType>
__aicore__ inline void ConfigureGatedActivationTail(GatedActivationTileContext<InputType>& context,
                                                    uint16_t validColumnCount)
{
    const uint32_t vectorLength = Activation::VECTOR_LENGTH_FP32;
    context.fullVectorLoopCount = validColumnCount / vectorLength;
    context.needTailVectorCompute = 0U;
    context.needAdditionalPaddingStore = 0U;
    context.tailComputeMaskElementCount = 0U;
    context.tailStoreMaskElementCount = 0U;
    context.additionalPaddingStoreMaskElementCount = 0U;
    context.gateTail = context.gate;
    context.upTail = context.up;
    context.outputTail = context.output;
    context.additionalPaddingOutput = context.output;

    const uint32_t tailElementCount = validColumnCount % vectorLength;
    if (tailElementCount == 0U) {
        return;
    }

    context.tailComputeMaskElementCount = tailElementCount;
    context.needTailVectorCompute = 1U;
    const uint32_t tailAndPaddingElementCount =
        context.outputRowStrideElements - context.fullVectorLoopCount * vectorLength;
    if (tailAndPaddingElementCount <= vectorLength) {
        context.tailStoreMaskElementCount = tailAndPaddingElementCount;
    } else {
        context.needAdditionalPaddingStore = 1U;
        context.tailStoreMaskElementCount = vectorLength;
        context.additionalPaddingStoreMaskElementCount = tailAndPaddingElementCount - vectorLength;
    }
    const uint32_t tailColumnOffsetElements = context.fullVectorLoopCount * vectorLength;
    context.gateTail = context.gate + tailColumnOffsetElements;
    context.upTail = context.up + tailColumnOffsetElements;
    context.outputTail = context.output + tailColumnOffsetElements;
    context.additionalPaddingOutput = context.outputTail + context.needTailVectorCompute * vectorLength;
}

// 只分派激活数学计算，输出转换和通信量化由调用方分别处理。
template <typename InputType, bool Prefetch>
__aicore__ inline void RunGatedActivation(GatedActivationTileContext<InputType>& context,
                                          uint8_t mode, uint8_t subMode, float clampLimit, float alpha, float beta)
{
    const ActMode actMode = static_cast<ActMode>(mode);
    const ActSubMode actSubMode = static_cast<ActSubMode>(subMode);
    if (actMode == ActMode::SITU) {
        const float invBeta = beta != 0.0f ? 1.0f / beta : 1.0f;
        if (actSubMode == ActSubMode::LINEAR) {
            const float invAlpha = alpha != 0.0f ? 1.0f / alpha : 1.0f;
            const Activation::SituGluParams situGluParams{clampLimit, beta, invBeta, alpha,
                                                          invAlpha};
            Activation::RunSiTUGLU<InputType, Prefetch, true>(context, situGluParams);
        } else {
            const Activation::SituGluParams situGluParams{clampLimit, beta, invBeta, 1.0f, 1.0f};
            Activation::RunSiTUGLU<InputType, Prefetch, false>(context, situGluParams);
        }
    } else if (actMode == ActMode::SWIGLU_STEP) {
        Activation::RunSwiGLU<InputType, Prefetch, true>(context, clampLimit);
    } else if (actMode == ActMode::SWIGLU_OAI) {
        const Activation::SwiGluOaiParams swiGluOaiParams{clampLimit, alpha, beta};
        Activation::RunSwiGLUOAI<InputType, Prefetch>(context, swiGluOaiParams);
    } else {
        Activation::RunSwiGLU<InputType, Prefetch, false>(context, clampLimit);
    }
}
} // namespace Activation
} // namespace MegaMoeImpl
#endif
#endif
