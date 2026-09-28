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
 * \file vf_muls_cast_dpse.h
 */
#ifndef VF_MULS_CAST_DPSE_H
#define VF_MULS_CAST_DPSE_H

#include "kernel_tensor.h"

namespace AscendC {
#ifndef __CCE_KT_TEST__
constexpr static Reg::CastTrait DPSE_CAST_TRAIT_B32_TO_B16 = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::NO_SAT,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};

template <typename T, uint32_t dataSize>
__simd_vf__ inline void MulsCastDpseVF(uint64_t dstLocalInt, uint64_t srcLocalInt, float scaleValue)
{
    using namespace Reg;
    RegTensor<float> srcReg0;
    RegTensor<float> srcReg1;
    RegTensor<float> scaledReg0;
    RegTensor<float> scaledReg1;
    RegTensor<T> dstReg0;
    RegTensor<T> dstReg1;
    uint32_t sreg = dataSize;
    constexpr uint32_t regSize = VECTOR_REG_WIDTH / sizeof(float);
    constexpr uint16_t repeatTimes = (dataSize + regSize - 1) / regSize;
    constexpr uint16_t unrollFactor = 2;
    constexpr uint16_t unrollTimes = repeatTimes / unrollFactor;

    for (uint16_t i = 0; i < unrollTimes; ++i) {
        uint32_t offset0 = i * unrollFactor * regSize;
        uint32_t offset1 = offset0 + regSize;
        MaskReg mask0 = UpdateMask<float>(sreg);
        MaskReg mask1 = UpdateMask<float>(sreg);
        LoadAlign(srcReg0, (__ubuf__ float *&)srcLocalInt + offset0);
        LoadAlign(srcReg1, (__ubuf__ float *&)srcLocalInt + offset1);
        Muls(scaledReg0, srcReg0, scaleValue, mask0);
        Muls(scaledReg1, srcReg1, scaleValue, mask1);
        Cast<T, float, DPSE_CAST_TRAIT_B32_TO_B16>(dstReg0, scaledReg0, mask0);
        Cast<T, float, DPSE_CAST_TRAIT_B32_TO_B16>(dstReg1, scaledReg1, mask1);
        StoreAlign<T, StoreDist::DIST_PACK_B32>((__ubuf__ T *&)dstLocalInt + offset0, dstReg0, mask0);
        StoreAlign<T, StoreDist::DIST_PACK_B32>((__ubuf__ T *&)dstLocalInt + offset1, dstReg1, mask1);
    }

    if constexpr (repeatTimes % unrollFactor != 0) {
        RegTensor<float> srcRegTail;
        RegTensor<float> scaledRegTail;
        RegTensor<T> dstRegTail;
        MaskReg maskTail = UpdateMask<float>(sreg);
        constexpr uint32_t tailOffset = unrollTimes * unrollFactor * regSize;
        LoadAlign(srcRegTail, (__ubuf__ float *&)srcLocalInt + tailOffset);
        Muls(scaledRegTail, srcRegTail, scaleValue, maskTail);
        Cast<T, float, DPSE_CAST_TRAIT_B32_TO_B16>(dstRegTail, scaledRegTail, maskTail);
        StoreAlign<T, StoreDist::DIST_PACK_B32>((__ubuf__ T *&)dstLocalInt + tailOffset, dstRegTail, maskTail);
    }
}
#else
template <typename T, uint32_t dataSize>
__aicore__ inline void MulsCastDpseVF(uint64_t dstLocalInt, uint64_t srcLocalInt, float scaleValue)
{}
#endif
} // namespace AscendC

#endif // VF_MULS_CAST_DPSE_H
