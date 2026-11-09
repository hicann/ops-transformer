/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef VF_GATHER_SIN_COS_ARCH92_H
#define VF_GATHER_SIN_COS_ARCH92_H

#include "kernel_tensor.h"
#include "vf_comm_arch92.h"

namespace MlaProlog {

// Sources are separate BF16 staging arrays; destinations are separate FP32 Sin/Cos arrays.
// Each iteration reads 128B from each source and writes one complete 256B destination row.
__simd_vf__ inline void GatherSinCosD64VFImpl(__ubuf__ float* cosDst, __ubuf__ float* sinDst,
                                              __ubuf__ bfloat16_t* cosSrc, __ubuf__ bfloat16_t* sinSrc, uint16_t rows)
{
    Reg::RegTensor<bfloat16_t> cosB16;
    Reg::RegTensor<bfloat16_t> sinB16;
    Reg::RegTensor<float> cosF32;
    Reg::RegTensor<float> sinF32;
    Reg::RegTensor<float> sinNeg;
    Reg::RegTensor<float> sinOut;
    Reg::MaskReg all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg lowHalf = Reg::CreateMask<float, Reg::MaskPattern::H>();

    for (uint16_t r = 0; r < rows; ++r) {
        const uint32_t offset = static_cast<uint32_t>(r) * 64U;
        Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(cosB16, cosSrc + offset);
        Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(sinB16, sinSrc + offset);
        Reg::Cast<float, bfloat16_t, castTraitB162B32>(cosF32, cosB16, all);
        Reg::Cast<float, bfloat16_t, castTraitB162B32>(sinF32, sinB16, all);
        Reg::Muls<float, float, Reg::MaskMergeMode::ZEROING>(sinNeg, sinF32, -1.0f, lowHalf);
        // ZEROING Muls does not preserve inactive lanes. Select the original right half explicitly.
        Reg::Select<float>(sinOut, sinNeg, sinF32, lowHalf);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(cosDst + offset, cosF32, all);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(sinDst + offset, sinOut, all);
    }
    // No row reads a preceding row's stores. Publish once for subsequent VF consumers.
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

__aicore__ inline void GatherSinCosD64VF(const LocalTensor<float>& cosDst, const LocalTensor<float>& sinDst,
                                         const LocalTensor<bfloat16_t>& cosSrc, const LocalTensor<bfloat16_t>& sinSrc,
                                         uint16_t rows)
{
    GatherSinCosD64VFImpl((__ubuf__ float*)cosDst.GetPhyAddr(), (__ubuf__ float*)sinDst.GetPhyAddr(),
                          (__ubuf__ bfloat16_t*)cosSrc.GetPhyAddr(), (__ubuf__ bfloat16_t*)sinSrc.GetPhyAddr(), rows);
}

} // namespace MlaProlog
#endif // VF_GATHER_SIN_COS_ARCH92_H
