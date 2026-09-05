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
 * \file vf_rope.h
 * \brief
 */

#ifndef VF_ROPE_H
#define VF_ROPE_H

#include "kernel_operator.h"
#include "../compressor_comm.h"

using namespace AscendC;

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

template <typename T, typename ROPET>
__simd_vf__ void HalfModeRopeVF(__ubuf__ T *sinUb, __ubuf__ T *cosUb, __ubuf__ T *inUb, __ubuf__ ROPET *outUb,
                                uint32_t row, uint32_t col, uint32_t actualCol, uint64_t baseAddr)
{
    Reg::RegTensor<T> vregCos;
    Reg::RegTensor<T> vregHalfCos;
    Reg::RegTensor<T> vregSin;
    Reg::RegTensor<T> vregHalfSin;
    Reg::RegTensor<T> vregIn;
    Reg::RegTensor<T> vregHalfIn;
    Reg::RegTensor<T> vregOut;
    Reg::RegTensor<T> vregHalfOut;
    Reg::RegTensor<T> vregTemp;
    Reg::RegTensor<T> vregCastIn;
    Reg::RegTensor<ROPET> vregOutBf16;
    Reg::RegTensor<ROPET> vregOutHalfBf16;
    Reg::RegTensor<ROPET> vregCastOut;
    uint32_t maskValue = col / 2;
    Reg::MaskReg mask = Reg::UpdateMask<T>(maskValue);
    uint32_t halfCol = col / 2;

    for (uint32_t rIdx = 0; rIdx < row; rIdx++) {
        __ubuf__ T *curSinUb = sinUb + rIdx * col;
        __ubuf__ T *curCosUb = cosUb + rIdx * col;
        __ubuf__ T *curInUb = inUb + rIdx * actualCol;
        __ubuf__ ROPET *curOutUb = outUb + rIdx * actualCol;

        Reg::DataCopy(vregIn, curInUb + baseAddr);
        Reg::DataCopy(vregHalfIn, curInUb + baseAddr + halfCol);
        Reg::DataCopy(vregCos, curCosUb);
        Reg::DataCopy(vregHalfCos, curCosUb + halfCol);
        Reg::DataCopy(vregSin, curSinUb);
        Reg::DataCopy(vregHalfSin, curSinUb + halfCol);
        Reg::Mul(vregSin, vregSin, vregHalfIn, mask);
        Reg::Mul(vregHalfSin, vregHalfSin, vregIn, mask);
        Reg::Mul(vregCos, vregCos, vregIn, mask);
        Reg::Sub(vregOut, vregCos, vregSin, mask);
        Reg::Mul(vregHalfCos, vregHalfCos, vregHalfIn, mask);
        Reg::Add(vregHalfOut, vregHalfSin, vregHalfCos, mask);
        Reg::Cast<ROPET, T, castTraitB322B16>(vregOutBf16, vregOut, mask);
        Reg::DataCopy<ROPET, Reg::StoreDist::DIST_PACK_B32>(curOutUb + baseAddr, vregOutBf16, mask);
        Reg::Cast<ROPET, T, castTraitB322B16>(vregOutHalfBf16, vregHalfOut, mask);
        Reg::DataCopy<ROPET, Reg::StoreDist::DIST_PACK_B32>(curOutUb + baseAddr + halfCol, vregOutHalfBf16, mask);

        for (uint64_t dOffset = 0; dOffset < baseAddr; dOffset += 64) {
            uint32_t castMaskValue = min(baseAddr - dOffset, static_cast<uint64_t>(64));
            Reg::MaskReg castMask = Reg::UpdateMask<T>(castMaskValue);
            Reg::DataCopy(vregCastIn, curInUb + dOffset);
            Reg::Cast<ROPET, T, castTraitB322B16>(vregCastOut, vregCastIn, castMask);
            Reg::DataCopy<ROPET, Reg::StoreDist::DIST_PACK_B32>(curOutUb + dOffset, vregCastOut, castMask);
        }
    }
}

template <typename T, typename ROPET>
__simd_vf__ void InterleaveModeRopeVF(__ubuf__ T *sinUb, __ubuf__ T *cosUb, __ubuf__ T *inUb, __ubuf__ ROPET *outUb,
                                      uint32_t row, uint32_t col, uint32_t actualCol, uint64_t baseAddr)
{
    Reg::RegTensor<T> vregCos;
    Reg::RegTensor<T> vregSin;
    Reg::RegTensor<T> vregIn;
    Reg::RegTensor<T> vregOdd;
    Reg::RegTensor<T> vregEven;
    Reg::RegTensor<T> vregOut;
    Reg::RegTensor<T> vregTemp;
    Reg::RegTensor<T> vregCastIn;
    Reg::RegTensor<ROPET> vregOutBf16;
    Reg::RegTensor<ROPET> vregCastOut;
    uint32_t maskValue = col;
    Reg::MaskReg mask = Reg::UpdateMask<T>(maskValue);

    for (uint32_t rIdx = 0; rIdx < row; rIdx++) {
        __ubuf__ T *curSinUb = sinUb + rIdx * col;
        __ubuf__ T *curCosUb = cosUb + rIdx * col;
        __ubuf__ T *curInUb = inUb + rIdx * actualCol;
        __ubuf__ ROPET *curOutUb = outUb + rIdx * actualCol;

        Reg::DataCopy(vregIn, curInUb + baseAddr);
        Reg::DataCopy(vregCos, curCosUb);
        Reg::DataCopy(vregSin, curSinUb);
        Reg::Mul(vregCos, vregCos, vregIn, mask);
        Reg::DeInterleave<T>(vregEven, vregOdd, vregIn, vregTemp);
        Reg::Muls(vregOdd, vregOdd, static_cast<T>(-1.0), mask);
        Reg::Interleave<T>(vregIn, vregTemp, vregOdd, vregEven);
        Reg::Mul(vregSin, vregSin, vregIn, mask);
        Reg::Add(vregOut, vregCos, vregSin, mask);
        Reg::Cast<ROPET, T, castTraitB322B16>(vregOutBf16, vregOut, mask);
        Reg::DataCopy<ROPET, Reg::StoreDist::DIST_PACK_B32>(curOutUb + baseAddr, vregOutBf16, mask);
        for (uint64_t dOffset = 0; dOffset < baseAddr; dOffset += 64) {
            uint32_t castMaskValue = min(baseAddr - dOffset, static_cast<uint64_t>(64));
            Reg::MaskReg castMask = Reg::UpdateMask<T>(castMaskValue);
            Reg::DataCopy(vregCastIn, curInUb + dOffset);
            Reg::Cast<ROPET, T, castTraitB322B16>(vregCastOut, vregCastIn, castMask);
            Reg::DataCopy<ROPET, Reg::StoreDist::DIST_PACK_B32>(curOutUb + dOffset, vregCastOut, castMask);
        }
    }
}

template <Compressor::ROTARY_MODE MODE, typename T, typename ROPET>
__aicore__ inline void RopeVF(const LocalTensor<T> &sinTensor, const LocalTensor<T> &cosTensor,
                              const LocalTensor<T> &inTensor, const LocalTensor<ROPET> &outTensor, uint32_t row,
                              uint32_t col, uint32_t actualCol, uint64_t baseAddr)
{
    __ubuf__ T *sinUb = (__ubuf__ T *)sinTensor.GetPhyAddr();
    __ubuf__ T *cosUb = (__ubuf__ T *)cosTensor.GetPhyAddr();
    __ubuf__ T *inUb = (__ubuf__ T *)inTensor.GetPhyAddr();
    __ubuf__ ROPET *outUb = (__ubuf__ ROPET *)outTensor.GetPhyAddr();

    if constexpr (MODE == Compressor::ROTARY_MODE::HALF) {
        HalfModeRopeVF(sinUb, cosUb, inUb, outUb, row, col, actualCol, baseAddr);
    } else {
        InterleaveModeRopeVF(sinUb, cosUb, inUb, outUb, row, col, actualCol, baseAddr);
    }
}

#endif
