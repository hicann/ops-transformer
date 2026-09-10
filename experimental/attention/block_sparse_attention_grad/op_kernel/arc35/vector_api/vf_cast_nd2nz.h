/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include "kernel_tensor.h"
namespace AscendC {

using namespace MicroAPI;
constexpr AscendC::MicroAPI::CastTrait castTraitFp322Fp16Odd = {
    AscendC::MicroAPI::RegLayout::ONE,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};
constexpr AscendC::MicroAPI::CastTrait castTraitFp322Fp16Even = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

template <typename T1>
__aicore__ inline void CastND2NZ(const LocalTensor<T1> &dstTensor, const LocalTensor<float> &srcTensor,
                                 const uint32_t srcM, const uint32_t srcN)
{
    const uint32_t blockSize = 32;
    const uint32_t blockN = blockSize / sizeof(T1); // 16
    const uint32_t fullExeSize = srcN;
    uint64_t srcLocalInt = srcTensor.GetPhyAddr();
    uint64_t dstLocalInt = dstTensor.GetPhyAddr();
    uint32_t blockStride = (srcM * blockN) * sizeof(T1) / blockSize;
    uint32_t repeatStride = 1;
    __VEC_SCOPE__
    {
        RegTensor<float> vregSrcEven;
        RegTensor<float> vregSrcOdd;
        RegTensor<T1> vregCastEven;
        RegTensor<T1> vregCastOdd;
        RegTensor<T1> vregCastRes;
        MaskReg pregFullExe = CreateMask<T1, MaskPattern::ALL>();

        // [m,n] -> [n1,m1,16,16] -> [n1,m1*16,16] -> [n1,m1*16+1,16]
        for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
            DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vregSrcEven, vregSrcOdd, ((__ubuf__ float *&)srcLocalInt), fullExeSize);
            Cast<T1, float, castTraitFp322Fp16Even>(vregCastEven, vregSrcEven, pregFullExe);
            Cast<T1, float, castTraitFp322Fp16Odd>(vregCastOdd, vregSrcOdd, pregFullExe);
            // 0101: b16 0001: b32 1111: b8
            Or((RegTensor<uint16_t> &)vregCastRes, (RegTensor<uint16_t> &)vregCastEven,
               (RegTensor<uint16_t> &)vregCastOdd, pregFullExe);
            // high 16bits represents stride with each 8 blocks（256B) low 16bits represent repeat stride
            DataCopy<T1, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T1 *&)dstLocalInt), vregCastRes, blockStride, repeatStride, pregFullExe);
        }
    }
}

} // namespace AscendC
