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
 * \file vf_comm_arch92.h
 * \brief VF comm
 */

#ifndef VF_COMM_ARCH92_H
#define VF_COMM_ARCH92_H

#include "kernel_tensor.h"

namespace MlaProlog {

constexpr uint32_t ROPE_VF_COL = 64;

constexpr Reg::CastTrait castTraitB162B32 = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::UNKNOWN,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};

constexpr Reg::CastTrait castTraitB322B16 = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::NO_SAT,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};

template <typename B16T>
__simd_vf__ inline void CastF32ToB16NoSatVFImpl(__ubuf__ B16T* dstUb, __ubuf__ float* srcUb, uint32_t cnt)
{
    constexpr uint32_t repSize = 64; // VF fp32 寄存器宽度(256B / 4B)
    Reg::RegTensor<float> vregSrc;
    Reg::RegTensor<B16T> vregDst;
    Reg::MaskReg maskAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    const uint32_t fullLoop = cnt / repSize;
    for (uint32_t i = 0; i < fullLoop; ++i) {
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(vregSrc, srcUb + i * repSize);
        Reg::Cast<B16T, float, castTraitB322B16>(vregDst, vregSrc, maskAll);
        Reg::StoreAlign<B16T, Reg::StoreDist::DIST_PACK_B32>(dstUb + i * repSize, vregDst, maskAll);
    }
}

template <typename B16T>
__aicore__ inline void CastF32ToB16NoSat(const LocalTensor<B16T>& dstLocal, const LocalTensor<float>& srcLocal,
                                         uint32_t cnt)
{
    if (cnt == 0U) {
        return;
    }
    __ubuf__ float* srcUb = (__ubuf__ float*)srcLocal.GetPhyAddr();
    __ubuf__ B16T* dstUb = (__ubuf__ B16T*)dstLocal.GetPhyAddr();
    CastF32ToB16NoSatVFImpl<B16T>(dstUb, srcUb, cnt);
}

} // namespace MlaProlog

#endif // VF_COMM_ARCH92_H
