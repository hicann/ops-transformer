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
    // Softmax CastND2NZ (no inter-column +1 pad); DataCopy srcStride=0.
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

// Reorder an FP16/BF16 ND matrix to NZ without a type conversion.  Q and dO
// are already INPUT_TYPE after GM -> UB, unlike softmax results which are FP32.
template <typename T>
__aicore__ inline void TransdataND2NZ(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
                                      const uint32_t srcM, const uint32_t srcN)
{
    constexpr uint32_t BLOCK_SIZE = 32;
    const uint32_t blockN = BLOCK_SIZE / sizeof(T);
    uint64_t srcLocalInt = srcTensor.GetPhyAddr();
    uint64_t dstLocalInt = dstTensor.GetPhyAddr();
    // +1 32B pad between C0 columns; UB->L1 DataCopy must skip it via srcStride.
    const uint32_t blockStride = srcM * blockN * sizeof(T) / BLOCK_SIZE + 1;
    const uint32_t repeatStride = 1;

    __VEC_SCOPE__
    {
        RegTensor<T> vregSrc;
        MaskReg pregFullExe = CreateMask<T, MaskPattern::ALL>();

        // Same-dtype ND [m, n] -> NZ. Load a full ND row, scatter C0 blocks
        // with DATA_BLOCK_COPY. Do not deinterleave: that path is only for
        // FP32->FP16 CastND2NZ.
        for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); ++m) {
            LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                vregSrc, ((__ubuf__ T *&)srcLocalInt), srcN);
            DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T *&)dstLocalInt), vregSrc, blockStride, repeatStride, pregFullExe);
        }
    }
}

} // namespace AscendC
