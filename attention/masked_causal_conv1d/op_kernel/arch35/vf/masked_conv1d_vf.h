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
 * \file masked_conv1d_vf.h
 * \brief VF function for MaskedCausalConv1d kernel-width-3 causal convolution.
 *
 * ioQueue layout (unified prefix + x buffer):
 *   [bUbCur][sCur+2][ubFactorH]
 *   Row s   of batch b  =  x[s0 + s - 2]   (s=0,1 are prefix rows loaded from GM)
 *   Row s+2 of batch b  =  x[s0 + s]       (current token)
 *   Each row is ubFactorH elements wide (ubFactorH = k * H_REG, k >= 1).
 *
 * wBase layout: [3][ubFactorH]
 *   Tap t, H_REG chunk c: wBase[t * ubFactorH + c * H_REG]
 *
 * In-place write: y[s] is stored back to row s (overwrites x[s0+s-2], already consumed).
 * After VF: rows 0..sCur-1 hold y[0..sCur-1].
 *
 * Mask is applied at __aicore__ level (in Compute) to avoid b8 SLD.
 *
 * s×2 unroll: two __simd_vf__ variants split by sCur parity (modelled on compute.h).
 *   MaskedConv1dVFNoTail  — sCur even (sMain = sCur/2), no tail row.
 *   MaskedConv1dVFWithTail— sCur odd  (sMain = sCur/2), tail row unconditionally appended.
 *   CallMaskedConv1dVF (__aicore__) performs the parity check and dispatches;
 *   no runtime branch exists inside any __simd_vf__ function.
 *
 * Each main si iteration issues 6 independent LoadAlign (3 per s-slot) before any
 * compute, allowing the VF scheduler to maximally overlap load latency.
 */

#ifndef MASKED_CONV1D_VF_H
#define MASKED_CONV1D_VF_H

#include "kernel_operator.h"

using namespace AscendC;

namespace MaskedCausalConv1dNs {

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

// H_REG: VF FP32 register width in elements (256B / 4B = 64)
constexpr uint32_t H_REG = 64;

// ============================================================================
// MaskedConv1dVFNoTail: sCur is even, sMain = sCur/2.
//   Main loop only — no branch inside VF.
//   bStride = (2*sMain + 2) * ubFactorH  (sCur = 2*sMain)
// ============================================================================
template <typename T>
__simd_vf__ void MaskedConv1dVFNoTail(__ubuf__ T *ioBase, __ubuf__ T *wBase, uint32_t sMain, uint32_t bUbCur,
                                      uint32_t ubFactorH)
{
    Reg::MaskReg maskFull = Reg::CreateMask<float, Reg::MaskPattern::ALL>();

    uint32_t numHReg = ubFactorH / H_REG;
    uint32_t stride2 = 2 * ubFactorH;
    uint32_t stride3 = 3 * ubFactorH;
    uint32_t bStride = (2 * sMain + 2) * ubFactorH; // sCur=2*sMain, sCur+2 rows per batch

    for (uint32_t b = 0; b < bUbCur; ++b) {
        __ubuf__ T *bBase = ioBase + b * bStride;

        for (uint32_t c = 0; c < numHReg; ++c) {
            Reg::RegTensor<T> w0B16, w1B16, w2B16;
            Reg::RegTensor<float> w0F32, w1F32, w2F32;
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(w0B16, wBase + 0 * ubFactorH + c * H_REG);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(w1B16, wBase + 1 * ubFactorH + c * H_REG);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(w2B16, wBase + 2 * ubFactorH + c * H_REG);
            Reg::Cast<float, T, castTraitB162B32>(w0F32, w0B16, maskFull);
            Reg::Cast<float, T, castTraitB162B32>(w1F32, w1B16, maskFull);
            Reg::Cast<float, T, castTraitB162B32>(w2F32, w2B16, maskFull);

            __ubuf__ T *p = bBase + c * H_REG;

            for (uint32_t si = 0; si < sMain; ++si) {
                // 6 independent loads: 3 for s=2si, 3 for s=2si+1
                Reg::RegTensor<T> x0B16, x1B16, x2B16;
                Reg::RegTensor<float> x0F32, x1F32, x2F32;
                Reg::RegTensor<T> x3B16, x4B16, x5B16;
                Reg::RegTensor<float> x3F32, x4F32, x5F32;
                Reg::RegTensor<float> y0F32, y1F32, tmpF32;
                Reg::RegTensor<T> y0B16, y1B16;

                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x0B16, p);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x1B16, p + ubFactorH);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x2B16, p + stride2);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x3B16, p + ubFactorH);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x4B16, p + stride2);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x5B16, p + stride3);
                Reg::Cast<float, T, castTraitB162B32>(x0F32, x0B16, maskFull);
                Reg::Cast<float, T, castTraitB162B32>(x1F32, x1B16, maskFull);
                Reg::Cast<float, T, castTraitB162B32>(x2F32, x2B16, maskFull);
                Reg::Cast<float, T, castTraitB162B32>(x3F32, x3B16, maskFull);
                Reg::Cast<float, T, castTraitB162B32>(x4F32, x4B16, maskFull);
                Reg::Cast<float, T, castTraitB162B32>(x5F32, x5B16, maskFull);

                // y0 = x0*w0 + x1*w1 + x2*w2
                Reg::Mul(y0F32, x0F32, w0F32, maskFull);
                Reg::Mul(tmpF32, x1F32, w1F32, maskFull);
                Reg::Add(y0F32, y0F32, tmpF32, maskFull);
                Reg::Mul(tmpF32, x2F32, w2F32, maskFull);
                Reg::Add(y0F32, y0F32, tmpF32, maskFull);

                // y1 = x3*w0 + x4*w1 + x5*w2
                Reg::Mul(y1F32, x3F32, w0F32, maskFull);
                Reg::Mul(tmpF32, x4F32, w1F32, maskFull);
                Reg::Add(y1F32, y1F32, tmpF32, maskFull);
                Reg::Mul(tmpF32, x5F32, w2F32, maskFull);
                Reg::Add(y1F32, y1F32, tmpF32, maskFull);

                Reg::Cast<T, float, castTraitB322B16>(y0B16, y0F32, maskFull);
                Reg::Cast<T, float, castTraitB322B16>(y1B16, y1F32, maskFull);
                Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(p, y0B16, maskFull);
                Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(p + ubFactorH, y1B16, maskFull);

                p += 2 * ubFactorH;
            }
        }
    }
}

// ============================================================================
// MaskedConv1dVFWithTail: sCur is odd, sMain = sCur/2.
//   Main loop + one unconditional tail row — no branch inside VF.
//   bStride = (2*sMain + 3) * ubFactorH  (sCur = 2*sMain+1)
// ============================================================================
template <typename T>
__simd_vf__ void MaskedConv1dVFWithTail(__ubuf__ T *ioBase, __ubuf__ T *wBase, uint32_t sMain, uint32_t bUbCur,
                                        uint32_t ubFactorH)
{
    Reg::MaskReg maskFull = Reg::CreateMask<float, Reg::MaskPattern::ALL>();

    uint32_t numHReg = ubFactorH / H_REG;
    uint32_t stride2 = 2 * ubFactorH;
    uint32_t stride3 = 3 * ubFactorH;
    uint32_t bStride = (2 * sMain + 3) * ubFactorH; // sCur=2*sMain+1, sCur+2 rows per batch

    for (uint32_t b = 0; b < bUbCur; ++b) {
        __ubuf__ T *bBase = ioBase + b * bStride;

        for (uint32_t c = 0; c < numHReg; ++c) {
            Reg::RegTensor<T> w0B16, w1B16, w2B16;
            Reg::RegTensor<float> w0F32, w1F32, w2F32;
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(w0B16, wBase + 0 * ubFactorH + c * H_REG);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(w1B16, wBase + 1 * ubFactorH + c * H_REG);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(w2B16, wBase + 2 * ubFactorH + c * H_REG);
            Reg::Cast<float, T, castTraitB162B32>(w0F32, w0B16, maskFull);
            Reg::Cast<float, T, castTraitB162B32>(w1F32, w1B16, maskFull);
            Reg::Cast<float, T, castTraitB162B32>(w2F32, w2B16, maskFull);

            __ubuf__ T *p = bBase + c * H_REG;

            for (uint32_t si = 0; si < sMain; ++si) {
                // 6 independent loads: 3 for s=2si, 3 for s=2si+1
                Reg::RegTensor<T> x0B16, x1B16, x2B16;
                Reg::RegTensor<float> x0F32, x1F32, x2F32;
                Reg::RegTensor<T> x3B16, x4B16, x5B16;
                Reg::RegTensor<float> x3F32, x4F32, x5F32;
                Reg::RegTensor<float> y0F32, y1F32, tmpF32;
                Reg::RegTensor<T> y0B16, y1B16;

                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x0B16, p);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x1B16, p + ubFactorH);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x2B16, p + stride2);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x3B16, p + ubFactorH);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x4B16, p + stride2);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x5B16, p + stride3);
                Reg::Cast<float, T, castTraitB162B32>(x0F32, x0B16, maskFull);
                Reg::Cast<float, T, castTraitB162B32>(x1F32, x1B16, maskFull);
                Reg::Cast<float, T, castTraitB162B32>(x2F32, x2B16, maskFull);
                Reg::Cast<float, T, castTraitB162B32>(x3F32, x3B16, maskFull);
                Reg::Cast<float, T, castTraitB162B32>(x4F32, x4B16, maskFull);
                Reg::Cast<float, T, castTraitB162B32>(x5F32, x5B16, maskFull);

                // y0 = x0*w0 + x1*w1 + x2*w2
                Reg::Mul(y0F32, x0F32, w0F32, maskFull);
                Reg::Mul(tmpF32, x1F32, w1F32, maskFull);
                Reg::Add(y0F32, y0F32, tmpF32, maskFull);
                Reg::Mul(tmpF32, x2F32, w2F32, maskFull);
                Reg::Add(y0F32, y0F32, tmpF32, maskFull);

                // y1 = x3*w0 + x4*w1 + x5*w2
                Reg::Mul(y1F32, x3F32, w0F32, maskFull);
                Reg::Mul(tmpF32, x4F32, w1F32, maskFull);
                Reg::Add(y1F32, y1F32, tmpF32, maskFull);
                Reg::Mul(tmpF32, x5F32, w2F32, maskFull);
                Reg::Add(y1F32, y1F32, tmpF32, maskFull);

                Reg::Cast<T, float, castTraitB322B16>(y0B16, y0F32, maskFull);
                Reg::Cast<T, float, castTraitB322B16>(y1B16, y1F32, maskFull);
                Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(p, y0B16, maskFull);
                Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(p + ubFactorH, y1B16, maskFull);

                p += 2 * ubFactorH;
            }

            // Tail row — unconditional (this VF variant is only called when sCur is odd)
            {
                Reg::RegTensor<T> x0B16, x1B16, x2B16;
                Reg::RegTensor<float> x0F32, x1F32, x2F32, yF32, tmpF32;
                Reg::RegTensor<T> yB16;

                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x0B16, p);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x1B16, p + ubFactorH);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x2B16, p + stride2);
                Reg::Cast<float, T, castTraitB162B32>(x0F32, x0B16, maskFull);
                Reg::Cast<float, T, castTraitB162B32>(x1F32, x1B16, maskFull);
                Reg::Cast<float, T, castTraitB162B32>(x2F32, x2B16, maskFull);

                Reg::Mul(yF32, x0F32, w0F32, maskFull);
                Reg::Mul(tmpF32, x1F32, w1F32, maskFull);
                Reg::Add(yF32, yF32, tmpF32, maskFull);
                Reg::Mul(tmpF32, x2F32, w2F32, maskFull);
                Reg::Add(yF32, yF32, tmpF32, maskFull);

                Reg::Cast<T, float, castTraitB322B16>(yB16, yF32, maskFull);
                Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(p, yB16, maskFull);
            }
        }
    }
}

// ============================================================================
// CallMaskedConv1dVF: __aicore__ wrapper.
//   Parity check happens here (not inside VF) — dispatches to the correct
//   __simd_vf__ variant, each of which contains no runtime branches.
// ============================================================================
template <typename T>
__aicore__ inline void CallMaskedConv1dVF(LocalTensor<T> ioBuf, LocalTensor<T> wBuf, uint32_t sCur, uint32_t bUbCur,
                                          uint32_t ubFactorH)
{
    __ubuf__ T *ioBase = (__ubuf__ T *)ioBuf.GetPhyAddr();
    __ubuf__ T *wBase = (__ubuf__ T *)wBuf.GetPhyAddr();
    uint32_t sMain = sCur / 2;
    if (sCur & 1u) {
        MaskedConv1dVFWithTail<T>(ioBase, wBase, sMain, bUbCur, ubFactorH);
    } else {
        MaskedConv1dVFNoTail<T>(ioBase, wBase, sMain, bUbCur, ubFactorH);
    }
}

} // namespace MaskedCausalConv1dNs

#endif // MASKED_CONV1D_VF_H
