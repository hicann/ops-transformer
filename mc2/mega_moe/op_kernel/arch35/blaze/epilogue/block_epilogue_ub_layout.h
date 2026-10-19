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
 * \file block_epilogue_ub_layout.h
 * \brief 激活及 MX 量化的 UB 布局。
 */

#ifndef BLOCK_EPILOGUE_UB_LAYOUT_H
#define BLOCK_EPILOGUE_UB_LAYOUT_H

#if defined(__DAV_C310__)
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../../common/mega_moe_constants.h"

namespace MegaMoeImpl {

struct ActivationMxQuantUbOffsets {
    uint32_t firstInputOffsetBytes;
    uint32_t firstInputElementCapacity;
    uint32_t activationOutputOffsetBytes;
    uint32_t activationOutputElementCapacity;
    uint32_t quantOutputOffsetBytes;
    uint32_t quantOutputElementCapacity;
    uint32_t quantScaleOffsetBytes;
    uint32_t quantScaleElementCapacity;
    uint32_t maxExpOffsetBytes;
    uint32_t inverseMxScaleOffsetBytes;
    uint32_t scaleElementCapacity;
    uint32_t topkWeightOffsetBytes;
    uint32_t topkWeightElementCapacity;
};

template <typename InputType, uint32_t TileM, uint32_t TileN, bool TopkWeightsPrefetch>
__aicore__ constexpr inline ActivationMxQuantUbOffsets BuildActivationMxQuantUbOffsets(bool isGelu)
{
    constexpr uint32_t tileElementCount = TileM * TileN;
    constexpr uint32_t tileBytes = tileElementCount * sizeof(InputType);
    ActivationMxQuantUbOffsets offsets{};
    offsets.firstInputElementCapacity = tileElementCount;
    // 两种激活共用相同物理边界：预取权重从此处开始，非预取的双槽在此处结束。
    offsets.topkWeightOffsetBytes = tileBytes * 2U;
    offsets.topkWeightElementCapacity = TileM * INT32_PER_256B;
    if (isGelu) {
        // 容量表示 LocalTensor 视图范围，逐行量化只写入其中的有效字节。
        offsets.activationOutputElementCapacity = (TopkWeightsPrefetch ? 1U : 2U) * tileElementCount;
        if constexpr (TopkWeightsPrefetch) {
            // 单个 BF16 输入槽之后，按 FP8 最大占用预留连续量化值和 scale 区域。
            offsets.quantOutputOffsetBytes = tileBytes;
            offsets.quantScaleOffsetBytes = tileBytes + tileElementCount * sizeof(int8_t);
        } else {
            // GELU 原位复用双槽；量化值位于行首，scale 位于该行 FP8 最大输出的末端。
            offsets.quantOutputOffsetBytes = 0U;
            offsets.quantScaleOffsetBytes = TileN * sizeof(int8_t);
        }
        offsets.quantOutputElementCapacity = offsets.topkWeightOffsetBytes - offsets.quantOutputOffsetBytes;
        offsets.quantScaleElementCapacity = offsets.topkWeightOffsetBytes - offsets.quantScaleOffsetBytes;
    } else {
        offsets.activationOutputElementCapacity = tileElementCount / ACTIVATION_N_HALF;
        offsets.quantOutputElementCapacity = offsets.activationOutputElementCapacity;
        offsets.quantOutputOffsetBytes = offsets.activationOutputElementCapacity * sizeof(bfloat16_t);
        offsets.quantScaleOffsetBytes =
            offsets.quantOutputOffsetBytes + offsets.quantOutputElementCapacity * sizeof(int8_t);
        offsets.scaleElementCapacity = offsets.activationOutputElementCapacity / AscendC::ONE_BLK_SIZE;
        offsets.quantScaleElementCapacity = offsets.scaleElementCapacity;
        offsets.maxExpOffsetBytes = offsets.quantScaleOffsetBytes + offsets.scaleElementCapacity * sizeof(int8_t);
        offsets.inverseMxScaleOffsetBytes = offsets.maxExpOffsetBytes + offsets.scaleElementCapacity * sizeof(uint16_t);
    }
    return offsets;
}

template <typename InputType>
struct ActivationMxQuantUbPointers {
    __ubuf__ InputType* firstInput;
    __ubuf__ InputType* secondInput;
    __ubuf__ bfloat16_t* activationOutput;
    __ubuf__ int8_t* quantOutput;
    __ubuf__ uint16_t* quantScale;
    __ubuf__ uint16_t* maxExp;
    __ubuf__ uint16_t* inverseMxScale;
    uint32_t selectedSlotOffsetBytes;
};

template <typename InputType, uint32_t MaxSingleElementCount>
__aicore__ inline ActivationMxQuantUbPointers<InputType> ResolveActivationMxQuantUbPointers(
    __ubuf__ InputType* firstInputBase, __ubuf__ bfloat16_t* activationOutputBase, __ubuf__ int8_t* quantOutputBase,
    __ubuf__ uint16_t* quantScaleBase, __ubuf__ uint16_t* maxExpBase, __ubuf__ uint16_t* inverseMxScaleBase,
    uint32_t validColumnCount, uint16_t pingpongIdx, bool isGelu)
{
    // 预取调用始终传入首槽；非预取按 ping/pong 下标切换完整输入槽。
    const uint32_t selectedSlotOffsetBytes = pingpongIdx == 1U ? MaxSingleElementCount * sizeof(InputType) : 0U;
    ActivationMxQuantUbPointers<InputType> pointers{};
    pointers.firstInput = firstInputBase + selectedSlotOffsetBytes / sizeof(InputType);
    pointers.activationOutput = activationOutputBase + selectedSlotOffsetBytes / sizeof(bfloat16_t);
    pointers.quantOutput = quantOutputBase + selectedSlotOffsetBytes;
    pointers.quantScale = (__ubuf__ uint16_t*)((__ubuf__ int8_t*)quantScaleBase + selectedSlotOffsetBytes);
    pointers.selectedSlotOffsetBytes = selectedSlotOffsetBytes;
    if (!isGelu) {
        // 每行输入中的 gate 和 up 两个半区相邻，仅门控量化使用 UB 中间值。
        pointers.secondInput = pointers.firstInput + validColumnCount;
        pointers.maxExp = maxExpBase + selectedSlotOffsetBytes / sizeof(uint16_t);
        pointers.inverseMxScale = inverseMxScaleBase + selectedSlotOffsetBytes / sizeof(uint16_t);
    }
    return pointers;
}

} // namespace MegaMoeImpl

#endif // defined(__DAV_C310__)
#endif // BLOCK_EPILOGUE_UB_LAYOUT_H
