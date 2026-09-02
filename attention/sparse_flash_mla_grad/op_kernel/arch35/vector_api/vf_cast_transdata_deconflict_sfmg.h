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
 * \file vf_cast_transdata.h
 */
#ifndef MY_CAST_TRANSDATA_DECONFLICT_INTERFACE_SFMG_H
#define MY_CAST_TRANSDATA_DECONFLICT_INTERFACE_SFMG_H

#include "kernel_tensor.h"

namespace AscendC {
#ifndef __CCE_KT_TEST__
using namespace MicroAPI;
constexpr AscendC::MicroAPI::CastTrait castTraitFp322Fp8Three = {
    AscendC::MicroAPI::RegLayout::THREE,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};
constexpr AscendC::MicroAPI::CastTrait castTraitFp322Fp8Two = {
    AscendC::MicroAPI::RegLayout::TWO,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};
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
/* **************************************************************************************************
 * cast + ND_2_NZ                                             *
 * ************************************************************************************************* */
/*
 * @ingroup BroadcastSub
 * @brief compute :res = ND_2_NZ(cast(fp32_x))
 * @param [out] dstTensor output LocalTensor
 * @param [in] srcTensor input src LocalTensor
 */
template <typename T1, typename T>
__simd_vf__ inline void CastTransdataDeconflictVFhalf(const uint32_t fullExeSize, uint64_t srcLocalInt,
                                                      uint64_t dstLocalInt, uint32_t blockStride, uint32_t repeatStride,
                                                      uint32_t srcM)
{
    RegTensor<T> vregSrcEven;
    RegTensor<T> vregSrcOdd;
    RegTensor<half> vregCastEven;
    RegTensor<half> vregCastOdd;
    RegTensor<half> vregCastRes;
    MaskReg pregFullExe = CreateMask<T1, MaskPattern::ALL>();

    // [m,n] -> [n1,m1,16,16] -> [n1,m1*16,16] -> [n1,m1*16+1,16]

    for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
        LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B32>(
            vregSrcEven, vregSrcOdd, ((__ubuf__ T *&)srcLocalInt), fullExeSize);
        Cast<T1, T, castTraitFp322Fp16Even>(vregCastEven, vregSrcEven, pregFullExe);
        Cast<T1, T, castTraitFp322Fp16Odd>(vregCastOdd, vregSrcOdd, pregFullExe);
        // 0101: b16 0001: b32 1111: b8
        Or((RegTensor<uint16_t> &)vregCastRes, (RegTensor<uint16_t> &)vregCastEven, (RegTensor<uint16_t> &)vregCastOdd,
           pregFullExe);
        // high 16bits represents stride with each 8 blocks（256B) low 16bits represent repeat stride
        StoreAlign<T1, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T1 *&)dstLocalInt), vregCastRes, blockStride, repeatStride, pregFullExe);
    }
}

template <typename T1, typename T>
__simd_vf__ inline void CastTransdataDeconflictVFbf16(const uint32_t fullExeSize, uint64_t srcLocalInt,
                                                      uint64_t dstLocalInt, uint32_t blockStride, uint32_t repeatStride,
                                                      uint32_t srcM)
{
    RegTensor<T> vregSrcEven;
    RegTensor<T> vregSrcOdd;
    RegTensor<bfloat16_t> vregCastEven;
    RegTensor<bfloat16_t> vregCastOdd;
    RegTensor<bfloat16_t> vregCastRes;
    MaskReg pregFullExe = CreateMask<T1, MaskPattern::ALL>();

    // [m,n] -> [n1,m1,16,16] -> [n1,m1*16,16] -> [n1,m1*16+1,16]

    for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
        LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B32>(
            vregSrcEven, vregSrcOdd, ((__ubuf__ T *&)srcLocalInt), fullExeSize);
        Cast<T1, T, castTraitFp322Fp16Even>(vregCastEven, vregSrcEven, pregFullExe);
        Cast<T1, T, castTraitFp322Fp16Odd>(vregCastOdd, vregSrcOdd, pregFullExe);
        // 0101: b16 0001: b32 1111: b8
        Or((RegTensor<uint16_t> &)vregCastRes, (RegTensor<uint16_t> &)vregCastEven, (RegTensor<uint16_t> &)vregCastOdd,
           pregFullExe);
        // high 16bits represents stride with each 8 blocks（256B) low 16bits represent repeat stride
        StoreAlign<T1, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T1 *&)dstLocalInt), vregCastRes, blockStride, repeatStride, pregFullExe);
    }
}

template <typename T1, typename T>
__simd_vf__ inline void CastTransdataDeconflictVFT(const uint32_t fullExeSize, uint64_t srcLocalInt,
                                                   uint64_t dstLocalInt, uint32_t blockStride, uint32_t repeatStride,
                                                   uint64_t dstLocalIntTail, uint32_t srcM)
{
    RegTensor<T> vregSrc;
    RegTensor<T> vregSrcTail;
    MaskReg pregFullExe = CreateMask<T, MaskPattern::ALL>();
    // [m,n] -> [n1,m1,16,16] -> [n1,m1*16,16] -> [n1,m1*16+1,16]
    for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ T1 *&)srcLocalInt), 64);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcTail, ((__ubuf__ T1 *&)srcLocalInt), 64);
        // high 16bits represents stride with each 8 blocks（256B) low 16bits represent repeat stride
        StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)dstLocalInt), vregSrc, blockStride, repeatStride, pregFullExe);
        StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)dstLocalIntTail), vregSrcTail, blockStride, repeatStride, pregFullExe);
    }
}

template <typename T1, typename T>
__simd_vf__ inline void CastTransdataDeconflictVFfp8(const uint32_t fullExeSize, uint64_t srcLocalInt,
                                                     uint64_t dstLocalInt, uint32_t blockStride, uint32_t repeatStride,
                                                     uint64_t selrIndexesInt, uint32_t srcM)
{
    RegTensor<T> vregSrcEven;
    RegTensor<T> vregSrcOdd;
    // fp8_e5m2_t
    RegTensor<T1> vregCastEven;
    RegTensor<T1> vregCastOdd;
    RegTensor<T1> vregCastResTmp;
    RegTensor<T1> vregCastRes;
    RegTensor<T1> vregIndexes;
    MaskReg preg_all = CreateMask<T, MaskPattern::ALL>();
    MaskReg preg_all_b8 = CreateMask<T1, MaskPattern::ALL>();

    for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
        LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B32>(
            vregSrcEven, vregSrcOdd, ((__ubuf__ T *&)srcLocalInt), fullExeSize);
        Cast<T1, T, castTraitFp322Fp16Even>(vregCastEven, vregSrcEven, preg_all);
        Cast<T1, T, castTraitFp322Fp8Two>(vregCastOdd, vregSrcOdd, preg_all);
        Or((RegTensor<uint8_t> &)vregCastResTmp, (RegTensor<uint8_t> &)vregCastEven, (RegTensor<uint8_t> &)vregCastOdd,
           preg_all_b8);
        LoadAlign<uint8_t, MicroAPI::LoadDist::DIST_NORM>((RegTensor<uint8_t> &)vregIndexes,
                                                          ((__ubuf__ uint8_t *&)selrIndexesInt));
        Gather<uint8_t>((RegTensor<uint8_t> &)vregCastRes, (RegTensor<uint8_t> &)vregCastResTmp,
                        (RegTensor<uint8_t> &)vregIndexes);
        StoreAlign<T1, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T1 *&)dstLocalInt), vregCastRes, blockStride, repeatStride, preg_all_b8);
    }
}

template <typename T1, typename T, uint32_t srcN>
__aicore__ inline void CastTransdataDeconflict(const LocalTensor<T1> &dstTensor, const LocalTensor<T> &srcTensor,
                                               const LocalTensor<uint8_t> &selrIndexesTensor, uint32_t srcM)
{
    const uint32_t blockSize = 32;
    const uint32_t blockN = blockSize / sizeof(T1);
    const uint32_t fullExeSize = srcN;
    uint64_t srcLocalInt = srcTensor.GetPhyAddr();
    uint64_t dstLocalInt = dstTensor.GetPhyAddr();
    uint32_t blockStride = (srcM * blockN) * sizeof(T1) / blockSize + 1;
    uint32_t repeatStride = 1;

    if constexpr (IsSameType<T1, half>::value) {
        CastTransdataDeconflictVFhalf<T1, T>(fullExeSize, srcLocalInt, dstLocalInt, blockStride, repeatStride, srcM);
    } else if constexpr (IsSameType<T1, bfloat16_t>::value) {
        CastTransdataDeconflictVFbf16<T1, T>(fullExeSize, srcLocalInt, dstLocalInt, blockStride, repeatStride, srcM);
    } else if constexpr (IsSameType<T1, T>::value && srcN == 128) {
        uint64_t dstLocalIntTail = dstTensor.GetPhyAddr() + (srcM + 1) * (srcN / 2) * sizeof(T1);
        CastTransdataDeconflictVFT<T1, T>(fullExeSize, srcLocalInt, dstLocalInt, blockStride, repeatStride,
                                          dstLocalIntTail, srcM);
    } else if constexpr (IsSameType<T1, fp8_e5m2_t>::value || IsSameType<T1, fp8_e4m3fn_t>::value) {
        uint64_t selrIndexesInt = selrIndexesTensor.GetPhyAddr();
        CastTransdataDeconflictVFfp8<T1, T>(fullExeSize, srcLocalInt, dstLocalInt, blockStride, repeatStride,
                                            selrIndexesInt, srcM);
    }
}
#else
template <typename T1, typename T, uint32_t srcN>
__aicore__ inline void CastTransdataDeconflict(const LocalTensor<T1> &dstTensor, const LocalTensor<T> &srcTensor,
                                               const LocalTensor<uint8_t> &selrIndexesTensor, uint32_t srcM)
{}
#endif
} // namespace AscendC

#endif // MY_CAST_TRANSDATA_DECONFLICT_INTERFACE_SFMG_H
