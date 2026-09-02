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
 * \file compute.h
 * \brief MicroAPI VF helpers for masked_causal_conv1d_backward
 */

#ifndef MASKED_CAUSAL_CONV1D_BACKWARD_VF_COMPUTE_H
#define MASKED_CAUSAL_CONV1D_BACKWARD_VF_COMPUTE_H

#include "kernel_operator.h"

using namespace AscendC;

namespace MaskedCausalConv1dBackwardVF {

constexpr MicroAPI::CastTrait castTraitB162B32 = {
    MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::UNKNOWN,
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};

constexpr MicroAPI::CastTrait castTraitB322B16 = {
    MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::NO_SAT,
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};

constexpr uint32_t REGSIZE = 256;
constexpr uint32_t B32_REP_SIZE = REGSIZE / sizeof(float); // 64 floats per vector reg

template <typename T>
__simd_vf__ void GradXAndWeightNoTail(__ubuf__ T *goAddr, __ubuf__ T *wAddr, __ubuf__ T *inAddr, __ubuf__ T *giAddr,
                                      __ubuf__ float *p0Addr, __ubuf__ float *p1Addr, __ubuf__ float *p2Addr,
                                      uint32_t sMain, uint32_t dimLen)
{
    MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    uint32_t dimLoopNum = dimLen / B32_REP_SIZE;
    uint32_t dimOff = 0;

    for (uint32_t dl = 0; dl < dimLoopNum; ++dl) {
        MicroAPI::RegTensor<T> w0B16, w1B16, w2B16;
        MicroAPI::RegTensor<float> w0B32, w1B32, w2B32;
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(w0B16, wAddr + 0 * dimLen + dimOff);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(w1B16, wAddr + 1 * dimLen + dimOff);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(w2B16, wAddr + 2 * dimLen + dimOff);
        MicroAPI::Cast<float, T, castTraitB162B32>(w0B32, w0B16, fullMask);
        MicroAPI::Cast<float, T, castTraitB162B32>(w1B32, w1B16, fullMask);
        MicroAPI::Cast<float, T, castTraitB162B32>(w2B32, w2B16, fullMask);

        for (uint32_t si = 0; si < sMain; ++si) {
            uint32_t r0 = (si << 1); // base row
            uint32_t r1 = r0 + 1U;   // next row

            // Six independent loads for two consecutive output rows
            MicroAPI::RegTensor<T> go0B16, go1B16, go2B16;
            MicroAPI::RegTensor<float> go0B32, go1B32, go2B32;
            MicroAPI::RegTensor<T> go3B16, go4B16, go5B16;
            MicroAPI::RegTensor<float> go3B32, go4B32, go5B32;
            MicroAPI::RegTensor<T> in0B16, in1B16;
            MicroAPI::RegTensor<float> in0B32, in1B32;

            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(go0B16, goAddr + r0 * dimLen + dimOff);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(go1B16, goAddr + (r0 + 1) * dimLen + dimOff);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(go2B16, goAddr + (r0 + 2) * dimLen + dimOff);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(go3B16, goAddr + r1 * dimLen + dimOff);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(go4B16, goAddr + (r1 + 1) * dimLen + dimOff);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(go5B16, goAddr + (r1 + 2) * dimLen + dimOff);

            MicroAPI::Cast<float, T, castTraitB162B32>(go0B32, go0B16, fullMask);
            MicroAPI::Cast<float, T, castTraitB162B32>(go1B32, go1B16, fullMask);
            MicroAPI::Cast<float, T, castTraitB162B32>(go2B32, go2B16, fullMask);
            MicroAPI::Cast<float, T, castTraitB162B32>(go3B32, go3B16, fullMask);
            MicroAPI::Cast<float, T, castTraitB162B32>(go4B32, go4B16, fullMask);
            MicroAPI::Cast<float, T, castTraitB162B32>(go5B32, go5B16, fullMask);

            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(in0B16, inAddr + r0 * dimLen + dimOff);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(in1B16, inAddr + r1 * dimLen + dimOff);
            MicroAPI::Cast<float, T, castTraitB162B32>(in0B32, in0B16, fullMask);
            MicroAPI::Cast<float, T, castTraitB162B32>(in1B32, in1B16, fullMask);

            // y0 = go0*w2 + go1*w1 + go2*w0
            MicroAPI::RegTensor<float> y0B32, y1B32, tmp0B32, tmp1B32, tmp2B32;
            MicroAPI::Mul(y0B32, go0B32, w2B32, fullMask);
            MicroAPI::MulAddDst(y0B32, go1B32, w1B32, fullMask);
            MicroAPI::MulAddDst(y0B32, go2B32, w0B32, fullMask);
            // y1 = go3*w2 + go4*w1 + go5*w0
            MicroAPI::Mul(y1B32, go3B32, w2B32, fullMask);
            MicroAPI::MulAddDst(y1B32, go4B32, w1B32, fullMask);
            MicroAPI::MulAddDst(y1B32, go5B32, w0B32, fullMask);

            // store gi rows
            MicroAPI::RegTensor<T> y0B16, y1B16;
            MicroAPI::Cast<T, float, castTraitB322B16>(y0B16, y0B32, fullMask);
            MicroAPI::Cast<T, float, castTraitB322B16>(y1B16, y1B32, fullMask);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(giAddr + r0 * dimLen + dimOff, y0B16, fullMask);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(giAddr + r1 * dimLen + dimOff, y1B16, fullMask);

            // products for row r0
            MicroAPI::Mul(tmp0B32, go0B32, in0B32, fullMask);
            MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(p2Addr + r0 * dimLen + dimOff, tmp0B32,
                                                                            fullMask);
            MicroAPI::Mul(tmp1B32, go1B32, in0B32, fullMask);
            MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(p1Addr + r0 * dimLen + dimOff, tmp1B32,
                                                                            fullMask);
            MicroAPI::Mul(tmp2B32, go2B32, in0B32, fullMask);
            MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(p0Addr + r0 * dimLen + dimOff, tmp2B32,
                                                                            fullMask);

            // products for row r1
            MicroAPI::Mul(tmp0B32, go3B32, in1B32, fullMask);
            MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(p2Addr + r1 * dimLen + dimOff, tmp0B32,
                                                                            fullMask);
            MicroAPI::Mul(tmp1B32, go4B32, in1B32, fullMask);
            MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(p1Addr + r1 * dimLen + dimOff, tmp1B32,
                                                                            fullMask);
            MicroAPI::Mul(tmp2B32, go5B32, in1B32, fullMask);
            MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(p0Addr + r1 * dimLen + dimOff, tmp2B32,
                                                                            fullMask);
        }
        dimOff += B32_REP_SIZE;
    }
}

template <typename T>
__simd_vf__ void GradXAndWeightWithTail(__ubuf__ T *goAddr, __ubuf__ T *wAddr, __ubuf__ T *inAddr, __ubuf__ T *giAddr,
                                        __ubuf__ float *p0Addr, __ubuf__ float *p1Addr, __ubuf__ float *p2Addr,
                                        uint32_t sMain, uint32_t dimLen)
{
    MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    uint32_t dimLoopNum = dimLen / B32_REP_SIZE;
    uint32_t dimOff = 0;

    for (uint32_t dl = 0; dl < dimLoopNum; ++dl) {
        MicroAPI::RegTensor<T> w0B16, w1B16, w2B16;
        MicroAPI::RegTensor<float> w0B32, w1B32, w2B32;
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(w0B16, wAddr + 0 * dimLen + dimOff);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(w1B16, wAddr + 1 * dimLen + dimOff);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(w2B16, wAddr + 2 * dimLen + dimOff);
        MicroAPI::Cast<float, T, castTraitB162B32>(w0B32, w0B16, fullMask);
        MicroAPI::Cast<float, T, castTraitB162B32>(w1B32, w1B16, fullMask);
        MicroAPI::Cast<float, T, castTraitB162B32>(w2B32, w2B16, fullMask);

        for (uint32_t si = 0; si < sMain; ++si) {
            uint32_t r0 = (si << 1);
            uint32_t r1 = r0 + 1U;

            MicroAPI::RegTensor<T> go0B16, go1B16, go2B16, go3B16, go4B16, go5B16;
            MicroAPI::RegTensor<float> go0B32, go1B32, go2B32, go3B32, go4B32, go5B32;
            MicroAPI::RegTensor<T> in0B16, in1B16;
            MicroAPI::RegTensor<float> in0B32, in1B32;

            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(go0B16, goAddr + r0 * dimLen + dimOff);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(go1B16, goAddr + (r0 + 1) * dimLen + dimOff);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(go2B16, goAddr + (r0 + 2) * dimLen + dimOff);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(go3B16, goAddr + r1 * dimLen + dimOff);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(go4B16, goAddr + (r1 + 1) * dimLen + dimOff);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(go5B16, goAddr + (r1 + 2) * dimLen + dimOff);
            MicroAPI::Cast<float, T, castTraitB162B32>(go0B32, go0B16, fullMask);
            MicroAPI::Cast<float, T, castTraitB162B32>(go1B32, go1B16, fullMask);
            MicroAPI::Cast<float, T, castTraitB162B32>(go2B32, go2B16, fullMask);
            MicroAPI::Cast<float, T, castTraitB162B32>(go3B32, go3B16, fullMask);
            MicroAPI::Cast<float, T, castTraitB162B32>(go4B32, go4B16, fullMask);
            MicroAPI::Cast<float, T, castTraitB162B32>(go5B32, go5B16, fullMask);

            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(in0B16, inAddr + r0 * dimLen + dimOff);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(in1B16, inAddr + r1 * dimLen + dimOff);
            MicroAPI::Cast<float, T, castTraitB162B32>(in0B32, in0B16, fullMask);
            MicroAPI::Cast<float, T, castTraitB162B32>(in1B32, in1B16, fullMask);

            MicroAPI::RegTensor<float> y0B32, y1B32, tmp0B32, tmp1B32, tmp2B32;
            // y0
            MicroAPI::Mul(y0B32, go0B32, w2B32, fullMask);
            MicroAPI::MulAddDst(y0B32, go1B32, w1B32, fullMask);
            MicroAPI::MulAddDst(y0B32, go2B32, w0B32, fullMask);
            // y1
            MicroAPI::Mul(y1B32, go3B32, w2B32, fullMask);
            MicroAPI::MulAddDst(y1B32, go4B32, w1B32, fullMask);
            MicroAPI::MulAddDst(y1B32, go5B32, w0B32, fullMask);

            MicroAPI::RegTensor<T> y0B16, y1B16;
            MicroAPI::Cast<T, float, castTraitB322B16>(y0B16, y0B32, fullMask);
            MicroAPI::Cast<T, float, castTraitB322B16>(y1B16, y1B32, fullMask);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(giAddr + r0 * dimLen + dimOff, y0B16, fullMask);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(giAddr + r1 * dimLen + dimOff, y1B16, fullMask);

            // products
            MicroAPI::Mul(tmp0B32, go0B32, in0B32, fullMask);
            MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(p2Addr + r0 * dimLen + dimOff, tmp0B32,
                                                                            fullMask);
            MicroAPI::Mul(tmp1B32, go1B32, in0B32, fullMask);
            MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(p1Addr + r0 * dimLen + dimOff, tmp1B32,
                                                                            fullMask);
            MicroAPI::Mul(tmp2B32, go2B32, in0B32, fullMask);
            MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(p0Addr + r0 * dimLen + dimOff, tmp2B32,
                                                                            fullMask);

            MicroAPI::Mul(tmp0B32, go3B32, in1B32, fullMask);
            MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(p2Addr + r1 * dimLen + dimOff, tmp0B32,
                                                                            fullMask);
            MicroAPI::Mul(tmp1B32, go4B32, in1B32, fullMask);
            MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(p1Addr + r1 * dimLen + dimOff, tmp1B32,
                                                                            fullMask);
            MicroAPI::Mul(tmp2B32, go5B32, in1B32, fullMask);
            MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(p0Addr + r1 * dimLen + dimOff, tmp2B32,
                                                                            fullMask);
        }

        // Tail row
        uint32_t r = (sMain << 1); // last row index
        MicroAPI::RegTensor<T> go0B16, go1B16, go2B16, inB16;
        MicroAPI::RegTensor<float> go0B32, go1B32, go2B32, inB32, yB32, tmp0B32, tmp1B32, tmp2B32;
        MicroAPI::RegTensor<T> yB16;

        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(go0B16, goAddr + r * dimLen + dimOff);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(go1B16, goAddr + (r + 1) * dimLen + dimOff);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(go2B16, goAddr + (r + 2) * dimLen + dimOff);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(inB16, inAddr + r * dimLen + dimOff);
        MicroAPI::Cast<float, T, castTraitB162B32>(go0B32, go0B16, fullMask);
        MicroAPI::Cast<float, T, castTraitB162B32>(go1B32, go1B16, fullMask);
        MicroAPI::Cast<float, T, castTraitB162B32>(go2B32, go2B16, fullMask);
        MicroAPI::Cast<float, T, castTraitB162B32>(inB32, inB16, fullMask);

        MicroAPI::Mul(yB32, go0B32, w2B32, fullMask);
        MicroAPI::MulAddDst(yB32, go1B32, w1B32, fullMask);
        MicroAPI::MulAddDst(yB32, go2B32, w0B32, fullMask);
        MicroAPI::Cast<T, float, castTraitB322B16>(yB16, yB32, fullMask);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(giAddr + r * dimLen + dimOff, yB16, fullMask);

        MicroAPI::Mul(tmp0B32, go0B32, inB32, fullMask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(p2Addr + r * dimLen + dimOff, tmp0B32,
                                                                        fullMask);
        MicroAPI::Mul(tmp1B32, go1B32, inB32, fullMask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(p1Addr + r * dimLen + dimOff, tmp1B32,
                                                                        fullMask);
        MicroAPI::Mul(tmp2B32, go2B32, inB32, fullMask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(p0Addr + r * dimLen + dimOff, tmp2B32,
                                                                        fullMask);

        dimOff += B32_REP_SIZE;
    }
}

template <typename T>
__simd_vf__ void CastRowB32ToDT(__ubuf__ float *srcAddr, __ubuf__ T *dstAddr, uint32_t dimLen)
{
    MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    uint32_t dimLoopNum = dimLen / B32_REP_SIZE;
    uint32_t dimOff = 0;
    for (uint32_t dl = 0; dl < dimLoopNum; ++dl) {
        MicroAPI::RegTensor<float> s;
        MicroAPI::RegTensor<T> dB16;
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(s, srcAddr + dimOff);
        MicroAPI::Cast<T, float, castTraitB322B16>(dB16, s, fullMask);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(dstAddr + dimOff, dB16, fullMask);
        dimOff += B32_REP_SIZE;
    }
}

template <typename T>
__aicore__ inline void DoGradXWeight(LocalTensor<T> &goUb, LocalTensor<T> &wUb, LocalTensor<T> &inUb,
                                     LocalTensor<T> &giUb, LocalTensor<float> &p0Ub, LocalTensor<float> &p1Ub,
                                     LocalTensor<float> &p2Ub, uint32_t sLen, uint32_t dimLen)
{
    __ubuf__ T *goAddr = (__ubuf__ T *)goUb.GetPhyAddr();
    __ubuf__ T *wAddr = (__ubuf__ T *)wUb.GetPhyAddr();
    __ubuf__ T *inAddr = (__ubuf__ T *)inUb.GetPhyAddr();
    __ubuf__ T *giAddr = (__ubuf__ T *)giUb.GetPhyAddr();
    __ubuf__ float *p0Addr = (__ubuf__ float *)p0Ub.GetPhyAddr();
    __ubuf__ float *p1Addr = (__ubuf__ float *)p1Ub.GetPhyAddr();
    __ubuf__ float *p2Addr = (__ubuf__ float *)p2Ub.GetPhyAddr();

    uint32_t sMain = sLen >> 1; // sLen/2
    if ((sLen & 1U) != 0U) {
        GradXAndWeightWithTail<T>(goAddr, wAddr, inAddr, giAddr, p0Addr, p1Addr, p2Addr, sMain, dimLen);
    } else {
        GradXAndWeightNoTail<T>(goAddr, wAddr, inAddr, giAddr, p0Addr, p1Addr, p2Addr, sMain, dimLen);
    }
}

template <typename T>
__aicore__ inline void DoCastToDT(LocalTensor<float> &inUb, uint32_t dimLen, LocalTensor<T> &outUb)
{
    __ubuf__ float *src = (__ubuf__ float *)inUb.GetPhyAddr();
    __ubuf__ T *dst = (__ubuf__ T *)outUb.GetPhyAddr();
    CastRowB32ToDT<T>(src, dst, dimLen);
}

} // namespace MaskedCausalConv1dBackwardVF

#endif
