/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHUNK_GATED_DELTA_RULE_STAGE2_VF_H
#define CHUNK_GATED_DELTA_RULE_STAGE2_VF_H

#include "kernel_tensor.h"

namespace ChunkGatedDeltaRule {
using namespace AscendC;
using namespace MicroAPI;

/*
 * FP32 状态缩放：scale = exp(gCumLast)，state = state * scale。
 * exp 结果保留在寄存器中，整条计算链只读写一次 state UB。
 */
__simd_vf__ inline void ScaleFp32StateByExpVF(__ubuf__ float *stateAddr, float gCumLast, uint32_t count)
{
    constexpr uint32_t elementsPerRepeat = VECTOR_REG_WIDTH / sizeof(float);
    RegTensor<float> scaleReg;
    RegTensor<float> stateReg;
    MaskReg allMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg mask;

    Duplicate(scaleReg, gCumLast);
    Exp<float, MaskMergeMode::ZEROING>(scaleReg, scaleReg, allMask);

    uint32_t remaining = count;
    uint16_t repeatTimes = static_cast<uint16_t>((count + elementsPerRepeat - 1) / elementsPerRepeat);
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = UpdateMask<float>(remaining);
        uint32_t offset = static_cast<uint32_t>(i) * elementsPerRepeat;
        LoadAlign(stateReg, stateAddr + offset);
        Mul(stateReg, stateReg, scaleReg, mask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(stateAddr + offset, stateReg, mask);
    }
}

/*
 * BF16 gated 状态缩放：scale = exp(gCumLast)，dst = BF16(FP32(src) * scale)。
 * Cast、Exp、Mul、Cast 在同一个 VF 中完成，FP32 中间结果不再写回临时 UB。
 */
__simd_vf__ inline void ScaleBf16StateByExpVF(__ubuf__ bfloat16_t *dstAddr, __ubuf__ bfloat16_t *srcAddr,
                                              float gCumLast, uint32_t count)
{
    constexpr uint32_t elementsPerRepeat = VECTOR_REG_WIDTH / sizeof(float);
    static constexpr CastTrait castBf16ToFp32 = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                                 RoundMode::UNKNOWN};
    static constexpr CastTrait castFp32ToBf16 = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING,
                                                 RoundMode::CAST_RINT};
    RegTensor<bfloat16_t> srcBf16Reg;
    RegTensor<bfloat16_t> dstBf16Reg;
    RegTensor<float> stateReg;
    RegTensor<float> scaleReg;
    MaskReg allMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg mask;

    Duplicate(scaleReg, gCumLast);
    Exp<float, MaskMergeMode::ZEROING>(scaleReg, scaleReg, allMask);

    uint32_t remaining = count;
    uint16_t repeatTimes = static_cast<uint16_t>((count + elementsPerRepeat - 1) / elementsPerRepeat);
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = UpdateMask<float>(remaining);
        uint32_t offset = static_cast<uint32_t>(i) * elementsPerRepeat;
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(srcBf16Reg, srcAddr + offset);
        Cast<float, bfloat16_t, castBf16ToFp32>(stateReg, srcBf16Reg, mask);
        Mul(stateReg, stateReg, scaleReg, mask);
        Cast<bfloat16_t, float, castFp32ToBf16>(dstBf16Reg, stateReg, mask);
        StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(dstAddr + offset, dstBf16Reg, mask);
    }
}

/* 无门控路径保持原 Cast -> Muls(1.0) -> Cast 顺序，并在寄存器中完成。 */
__simd_vf__ inline void ScaleBf16StateVF(__ubuf__ bfloat16_t *dstAddr, __ubuf__ bfloat16_t *srcAddr, uint32_t count)
{
    constexpr uint32_t elementsPerRepeat = VECTOR_REG_WIDTH / sizeof(float);
    static constexpr CastTrait castBf16ToFp32 = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                                 RoundMode::UNKNOWN};
    static constexpr CastTrait castFp32ToBf16 = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING,
                                                 RoundMode::CAST_RINT};
    RegTensor<bfloat16_t> srcBf16Reg;
    RegTensor<bfloat16_t> dstBf16Reg;
    RegTensor<float> stateReg;
    RegTensor<float> oneReg;
    MaskReg mask;

    Duplicate(oneReg, static_cast<float>(1.0f));
    uint32_t remaining = count;
    uint16_t repeatTimes = static_cast<uint16_t>((count + elementsPerRepeat - 1) / elementsPerRepeat);
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = UpdateMask<float>(remaining);
        uint32_t offset = static_cast<uint32_t>(i) * elementsPerRepeat;
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(srcBf16Reg, srcAddr + offset);
        Cast<float, bfloat16_t, castBf16ToFp32>(stateReg, srcBf16Reg, mask);
        Mul(stateReg, stateReg, oneReg, mask);
        Cast<bfloat16_t, float, castFp32ToBf16>(dstBf16Reg, stateReg, mask);
        StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(dstAddr + offset, dstBf16Reg, mask);
    }
}

} // namespace ChunkGatedDeltaRule

#endif // CHUNK_GATED_DELTA_RULE_STAGE2_VF_H
