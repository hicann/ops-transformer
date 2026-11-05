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
 * \file activation_common.h
 * \brief 门控激活公共数据结构、参数和常量。
 */

#ifndef MEGA_MOE_ARCH35_ACTIVATION_COMMON_H
#define MEGA_MOE_ARCH35_ACTIVATION_COMMON_H

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202))
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#include "basic_api/reg_compute/kernel_reg_compute_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../../../common/mega_moe_constants.h"
#if (__NPU_ARCH__ == 3510) && defined(__ASC_FTZ__)
#include <type_traits>
#include "c_api/asc_simd.h"
#endif

namespace MegaMoeImpl {
namespace Activation {

// 具体激活实现按 FP32 寄存器宽度划分主循环。
constexpr uint32_t VECTOR_LENGTH_FP32 = AscendC::VECTOR_REG_WIDTH / sizeof(float);

// 输入统一转换为 FP32，供各激活的寄存器计算流程使用。
constexpr AscendC::Reg::CastTrait CAST_INPUT_TO_FP32 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                        AscendC::Reg::MaskMergeMode::ZEROING,
                                                        AscendC::RoundMode::UNKNOWN};

// 当前激活结果统一写入 BF16 UB；量化流程在激活之后独立执行。
constexpr AscendC::Reg::CastTrait CAST_FP32_TO_BF16 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                                       AscendC::Reg::MaskMergeMode::ZEROING,
                                                       AscendC::RoundMode::CAST_RINT};

/*
 * 单个门控激活 tile 的完整执行上下文。
 *
 * Block Epilogue 在分派激活前直接计算地址、物理行距、循环次数和尾部 Mask，具体激活仅消费这些
 * 已物化的数据。此结构不保存 RegTensor、MaskReg 等 VF 内对象。
 */
template <typename InputType>
struct GatedActivationTileContext {
    __ubuf__ InputType* gate = nullptr;
    __ubuf__ InputType* up = nullptr;
    __ubuf__ bfloat16_t* output = nullptr;
    __ubuf__ float* topkWeights = nullptr;

    __ubuf__ InputType* gateTail = nullptr;
    __ubuf__ InputType* upTail = nullptr;
    __ubuf__ bfloat16_t* outputTail = nullptr;
    __ubuf__ bfloat16_t* additionalPaddingOutput = nullptr;

    uint32_t inputRowStrideElements;
    uint32_t outputRowStrideElements;
    uint16_t rowLoopCount;
    uint16_t fullVectorLoopCount;
    // 0/1 标志：直接作为尾 Vector 计算的 VF 循环上界。
    uint16_t needTailVectorCompute;
    // 0/1 标志：直接作为额外补零写回的 VF 循环上界。
    uint16_t needAdditionalPaddingStore;

    // UpdateMask 接收可写引用，具体激活需将以下计数复制到局部变量后再调用。
    uint32_t tailComputeMaskElementCount;
    uint32_t tailStoreMaskElementCount;
    uint32_t additionalPaddingStoreMaskElementCount;
};

#if (__NPU_ARCH__ == 3510) && defined(__ASC_FTZ__)
template <typename InputType>
__simd_callee__ inline void LoadActivationInput(vector_float& output, __ubuf__ InputType*& input,
                                                vector_bool computeMask)
{
    static_assert(std::is_same_v<InputType, bfloat16_t> || std::is_same_v<InputType, half>);
    using InputVector = std::conditional_t<std::is_same_v<InputType, bfloat16_t>, vector_bfloat16_t, vector_half>;
    InputVector inputReg;
    asc_loadalign_unpack_postupdate(inputReg, input, VECTOR_LENGTH_FP32);
    if constexpr (std::is_same_v<InputType, bfloat16_t>) {
        asc_bfloat162float(output, inputReg, computeMask, ASC_POSITION_EVEN);
    } else {
        asc_half2float(output, inputReg, computeMask, ASC_POSITION_EVEN);
    }
}

__simd_callee__ inline void StoreActivationOutput(__ubuf__ uint32_t*& output, vector_float input,
                                                  vector_bool computeMask, vector_bool storeMask)
{
    vector_bfloat16_t outputReg;
    asc_float2bfloat16_rn(outputReg, input, computeMask, ASC_POSITION_EVEN);
    constexpr uint32_t OUTPUT_STEP = VECTOR_LENGTH_FP32 * sizeof(bfloat16_t) / sizeof(uint32_t);
    asc_storealign_pack_postupdate(output, reinterpret_cast<vector_uint32_t&>(outputReg), OUTPUT_STEP, storeMask);
}
#endif

} // namespace Activation
} // namespace MegaMoeImpl

#endif // Arch35/Arch92 激活核函数
#endif // MEGA_MOE_ARCH35_ACTIVATION_COMMON_H
