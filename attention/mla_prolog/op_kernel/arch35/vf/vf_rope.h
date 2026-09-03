/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file vf_rope.h
 * \brief VF implementation for mla_prolog rope path (interleave-half mode).
 */

#ifndef VF_ROPE_H
#define VF_ROPE_H

#include "kernel_tensor.h"
#include "vf_comm.h"

namespace MlaProlog {

template <typename C>
__simd_vf__ inline void RopeVFImpl(__ubuf__ C *outputUb, __ubuf__ C *inputUb, __ubuf__ C *sinUb, __ubuf__ C *cosUb,
                                   uint32_t row, uint64_t srcStride, uint64_t dstStride, uint64_t sinCosStride)
{
    Reg::RegTensor<C> vregX;
    Reg::RegTensor<C> vregSin;
    Reg::RegTensor<C> vregCos;
    Reg::RegTensor<C> vregEven;
    Reg::RegTensor<C> vregOdd;
    Reg::RegTensor<C> vregHigh;
    Reg::RegTensor<C> vregLow;
    Reg::RegTensor<C> vregTemp;
    Reg::RegTensor<C> vregRes;

    Reg::MaskReg maskAll = Reg::CreateMask<C, Reg::MaskPattern::ALL>();
    Reg::MaskReg maskLowHalf = Reg::CreateMask<C, Reg::MaskPattern::H>();
    Reg::MaskReg maskHighHalf;
    Reg::Not(maskHighHalf, maskLowHalf, maskAll);

    for (uint16_t i = 0; i < row; ++i) {
        __ubuf__ C *curXUb = inputUb + i * srcStride;
        __ubuf__ C *curResUb = outputUb + i * dstStride;
        __ubuf__ C *curSinUb = sinUb + i * sinCosStride;
        __ubuf__ C *curCosUb = cosUb + i * sinCosStride;

        Reg::LoadAlign(vregX, curXUb);
        Reg::LoadAlign(vregSin, curSinUb);
        Reg::LoadAlign(vregCos, curCosUb);

        // vregEven = [evens(0..31), evens(32..63)]
        // vregOdd = [odds(0..31),  odds(32..63)]
        Reg::DeInterleave<C>(vregEven, vregOdd, vregX, vregX);

        // Part1 low  = cos * evens,        Part1 high preserved
        Reg::Mul(vregRes, vregCos, vregEven, maskLowHalf);
        // Part1 high = sin * evens,         Part1 low  preserved
        Reg::Mul(vregTemp, vregSin, vregEven, maskHighHalf);
        // Part2 low  = sin(-) * odds,      Part2 high preserved
        Reg::Mul(vregLow, vregSin, vregOdd, maskLowHalf);
        // Part2 high = cos * odds,          Part2 low  preserved
        Reg::Mul(vregHigh, vregCos, vregOdd, maskHighHalf);

        // Part1 = [cos_l*even + sin_l*odd, cos_u*odd + sin_u*even] = [y_lower, y_upper]
        Reg::Add(vregRes, vregRes, vregLow, maskLowHalf);
        Reg::Add(vregTemp, vregTemp, vregHigh, maskHighHalf);
        Reg::Move(vregRes, vregTemp, maskHighHalf);

        Reg::StoreAlign(curResUb, vregRes, maskAll);
    }
}

// col == 64
template <typename C>
__aicore__ inline void RotaryPosEmbVF(const LocalTensor<C> &outputLocal, const LocalTensor<C> &inputLocal,
                                      const LocalTensor<C> &cosLocal, const LocalTensor<C> &sinLocal, uint32_t row,
                                      uint64_t srcStride, uint64_t dstStride, uint64_t sinCosStride)
{
    __ubuf__ C *inputUb = (__ubuf__ C *)inputLocal.GetPhyAddr();
    __ubuf__ C *sinUb = (__ubuf__ C *)sinLocal.GetPhyAddr();
    __ubuf__ C *cosUb = (__ubuf__ C *)cosLocal.GetPhyAddr();
    __ubuf__ C *outputUb = (__ubuf__ C *)outputLocal.GetPhyAddr();

    RopeVFImpl(outputUb, inputUb, sinUb, cosUb, row, srcStride, dstStride, sinCosStride);
}
} // namespace MlaProlog

#endif // VF_ROPE_H
