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
 * \file moe_gating_top_k_backward_vf.h
 * \brief VF functions for MoeGatingTopKBackward on Arch35
 */

#ifndef MOE_GATING_TOP_K_BACKWARD_VF_H
#define MOE_GATING_TOP_K_BACKWARD_VF_H

#include "kernel_operator.h"

namespace MoeGatingTopKBackwardNs {
using namespace AscendC;

constexpr Reg::CastTrait castTraitB322B16 = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::NO_SAT,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};

constexpr Reg::CastTrait castTraitB162B32 = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::UNKNOWN,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};

constexpr Reg::DivSpecificMode div0UlpMode = {
    Reg::MaskMergeMode::ZEROING,
    true,
};

__simd_vf__ void SigmoidGradFP32VF(__ubuf__ float *xNormAddr, __ubuf__ float *gradNormXAddr, __ubuf__ float *gradXAddr,
                                   uint32_t totalElements, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    Reg::RegTensor<float> xReg, gNormReg, tmpReg, dstReg;
    Reg::MaskReg preg = Reg::CreateMask<float>();

    for (uint16_t i = 0; i < repeatTimes; i++) {
        uint32_t remaining = totalElements - i * oneRepeatSize;
        uint32_t chunkElems = (remaining > oneRepeatSize) ? oneRepeatSize : remaining;
        preg = Reg::UpdateMask<float>(chunkElems);
        Reg::LoadAlign(xReg, xNormAddr + i * oneRepeatSize);
        Reg::LoadAlign(gNormReg, gradNormXAddr + i * oneRepeatSize);
        Reg::Muls(tmpReg, xReg, static_cast<float>(-1), preg);
        Reg::Adds(tmpReg, tmpReg, static_cast<float>(1), preg);
        Reg::Mul(tmpReg, tmpReg, xReg, preg);
        Reg::Mul(dstReg, tmpReg, gNormReg, preg);
        Reg::StoreAlign(gradXAddr + i * oneRepeatSize, dstReg, preg);
    }
}

template <typename T>
__simd_vf__ void SigmoidGradHalfVF(__ubuf__ float *xNormAddr, __ubuf__ float *gradNormXAddr, __ubuf__ T *gradXAddr,
                                   uint32_t totalElements, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    Reg::RegTensor<float> xReg, gNormReg, tmpReg, dstReg;
    Reg::RegTensor<T> outReg;
    Reg::MaskReg pregFp = Reg::CreateMask<float>();

    for (uint16_t i = 0; i < repeatTimes; i++) {
        uint32_t remaining = totalElements - i * oneRepeatSize;
        uint32_t chunkElems = (remaining > oneRepeatSize) ? oneRepeatSize : remaining;
        pregFp = Reg::UpdateMask<float>(chunkElems);
        Reg::LoadAlign(xReg, xNormAddr + i * oneRepeatSize);
        Reg::LoadAlign(gNormReg, gradNormXAddr + i * oneRepeatSize);
        Reg::Muls(tmpReg, xReg, static_cast<float>(-1), pregFp);
        Reg::Adds(tmpReg, tmpReg, static_cast<float>(1), pregFp);
        Reg::Mul(tmpReg, tmpReg, xReg, pregFp);
        Reg::Mul(dstReg, tmpReg, gNormReg, pregFp);

        Reg::Cast<T, float, castTraitB322B16>(outReg, dstReg, pregFp);
        Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(gradXAddr + i * oneRepeatSize, outReg, pregFp);
    }
}

template <typename T>
__simd_vf__ void CastGradYFlatVF(__ubuf__ T *srcAddr, __ubuf__ float *dstAddr, uint32_t totalElements,
                                 uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    Reg::MaskReg preg = Reg::CreateMask<float>();
    for (uint16_t i = 0; i < repeatTimes; i++) {
        uint32_t remaining = totalElements - i * oneRepeatSize;
        uint32_t chunkElems = (remaining > oneRepeatSize) ? oneRepeatSize : remaining;
        preg = Reg::UpdateMask<float>(chunkElems);
        Reg::RegTensor<T> srcB16;
        Reg::RegTensor<float> dstF32;
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(srcB16, srcAddr + i * oneRepeatSize);
        Reg::Cast<float, T, castTraitB162B32>(dstF32, srcB16, preg);
        Reg::StoreAlign(dstAddr + i * oneRepeatSize, dstF32, preg);
    }
}

template <typename T>
__simd_vf__ void CastGradYRowsVF(__ubuf__ T *srcAddr, __ubuf__ float *dstAddr, uint16_t curRows, uint16_t k,
                                 uint16_t srcRowStride, uint16_t dstRowStride)
{
    constexpr uint16_t chunkSize = VECTOR_REG_WIDTH / (2 * sizeof(T)); // 64 for half/bfloat16_t
    uint16_t repeatTimes = static_cast<uint16_t>((k + chunkSize - 1) / chunkSize);

    for (uint16_t row = 0; row < curRows; row++) {
        __ubuf__ T *srcRow = srcAddr + row * srcRowStride;
        __ubuf__ float *dstRow = dstAddr + row * dstRowStride;

        uint32_t remainingK = k;
        for (uint16_t c = 0; c < repeatTimes; c++) {
            uint16_t chunkOffset = c * chunkSize;
            Reg::MaskReg preg = Reg::UpdateMask<float>(remainingK);

            Reg::RegTensor<T> srcB16;
            Reg::RegTensor<float> dstF32;
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(srcB16, srcRow + chunkOffset);
            Reg::Cast<float, T, castTraitB162B32>(dstF32, srcB16, preg);
            Reg::StoreAlign(dstRow + chunkOffset, dstF32, preg);
        }
    }
}

__simd_vf__ void SigmoidRenormBackwardVF(__ubuf__ float *xNormBase, __ubuf__ int32_t *expertIdxBase,
                                         __ubuf__ float *gradYBase, __ubuf__ float *gradNormXBase, float eps,
                                         uint16_t curRows, uint16_t k, uint16_t kAlign, uint16_t n)
{
    constexpr uint16_t chunkSize = VECTOR_REG_WIDTH / sizeof(int32_t); // 64 for int32_t/float
    uint16_t repeatTimes = static_cast<uint16_t>((k + chunkSize - 1) / chunkSize);
    Reg::MaskReg maskLane0 = Reg::CreateMask<float, Reg::MaskPattern::VL1>();

    for (uint16_t row = 0; row < curRows; row++) {
        __ubuf__ int32_t *idxRow = expertIdxBase + row * kAlign;
        __ubuf__ float *xNormRow = xNormBase + row * n;
        __ubuf__ float *gradYRow = gradYBase + row * kAlign;
        __ubuf__ float *gradNormXRow = gradNormXBase + row * n;

        // ---- Phase 1a: accumulate global D across all chunks, add eps last. ----
        Reg::RegTensor<float> globalD;
        Reg::Duplicate(globalD, 0.0f, maskLane0);

        uint32_t remainingK1 = k;
        for (uint16_t c = 0; c < repeatTimes; c++) {
            uint16_t chunkOffset = c * chunkSize;
            Reg::MaskReg chunkMask = Reg::UpdateMask<int32_t>(remainingK1);

            Reg::RegTensor<int32_t> idxReg;
            Reg::RegTensor<uint32_t> idxU32Reg;
            Reg::RegTensor<float> wPrimeReg, chunkSum;

            Reg::LoadAlign(idxReg, idxRow + chunkOffset);
            idxU32Reg = (Reg::RegTensor<uint32_t> &)idxReg;
            Reg::DataCopyGather(wPrimeReg, xNormRow, idxU32Reg, chunkMask);

            Reg::ReduceSum(chunkSum, wPrimeReg, chunkMask);
            Reg::Add(globalD, globalD, chunkSum, maskLane0);
        }
        Reg::Adds(globalD, globalD, eps, maskLane0);

        // ---- Phase 1b: re-gather w', compute w'/D, multiply by gradY, accumulate betaNum. ----
        Reg::RegTensor<float> globalBetaNum;
        Reg::Duplicate(globalBetaNum, 0.0f, maskLane0);

        uint32_t remainingK1b = k;
        for (uint16_t c = 0; c < repeatTimes; c++) {
            uint16_t chunkOffset = c * chunkSize;
            Reg::MaskReg chunkMask = Reg::UpdateMask<int32_t>(remainingK1b);

            Reg::RegTensor<int32_t> idxReg;
            Reg::RegTensor<uint32_t> idxU32Reg;
            Reg::RegTensor<float> wPrimeReg, gradYReg, wNormReg, tmpReg, chunkSum;

            Reg::LoadAlign(idxReg, idxRow + chunkOffset);
            idxU32Reg = (Reg::RegTensor<uint32_t> &)idxReg;
            Reg::DataCopyGather(wPrimeReg, xNormRow, idxU32Reg, chunkMask);

            // Broadcast D (lane0) to every lane for this chunk's w'/D.
            Reg::RegTensor<float> bcastDReg1b;
            Reg::Duplicate<float, Reg::HighLowPart::LOWEST, Reg::MaskMergeMode::ZEROING>(bcastDReg1b, globalD,
                                                                                         chunkMask);
            Reg::Div<float, &div0UlpMode>(wNormReg, wPrimeReg, bcastDReg1b, chunkMask);

            Reg::LoadAlign(gradYReg, gradYRow + chunkOffset);
            Reg::Mul(tmpReg, gradYReg, wNormReg, chunkMask);
            Reg::ReduceSum(chunkSum, tmpReg, chunkMask);
            Reg::Add(globalBetaNum, globalBetaNum, chunkSum, maskLane0);
        }

        // ---- Phase 2: recompute gradWPrime per chunk, scatter. ----
        uint32_t remainingK2 = k;
        for (uint16_t c = 0; c < repeatTimes; c++) {
            uint16_t chunkOffset = c * chunkSize;
            Reg::MaskReg chunkMask = Reg::UpdateMask<int32_t>(remainingK2);

            Reg::RegTensor<int32_t> idxReg;
            Reg::RegTensor<uint32_t> idxU32Reg;
            Reg::RegTensor<float> gradYReg;

            Reg::LoadAlign(idxReg, idxRow + chunkOffset);
            idxU32Reg = (Reg::RegTensor<uint32_t> &)idxReg;
            Reg::LoadAlign(gradYReg, gradYRow + chunkOffset);

            // Broadcast the row-global D/betaNum (lane0) to every lane of this chunk.
            // betaNum already includes /D from Phase 1b.
            Reg::RegTensor<float> bcastDReg;
            Reg::Duplicate<float, Reg::HighLowPart::LOWEST, Reg::MaskMergeMode::ZEROING>(bcastDReg, globalD, chunkMask);
            Reg::RegTensor<float> gradWPrimeReg;
            Reg::Duplicate<float, Reg::HighLowPart::LOWEST, Reg::MaskMergeMode::ZEROING>(gradWPrimeReg, globalBetaNum,
                                                                                         chunkMask);
            Reg::Sub(gradWPrimeReg, gradYReg, gradWPrimeReg, chunkMask);
            Reg::Div<float, &div0UlpMode>(gradWPrimeReg, gradWPrimeReg, bcastDReg, chunkMask);

            Reg::DataCopyScatter(gradNormXRow, gradWPrimeReg, idxU32Reg, chunkMask);
        }
    }
}

template <typename T>
__aicore__ inline void CallSigmoidGradVF(LocalTensor<T> gradXOut, LocalTensor<float> xNorm,
                                         LocalTensor<float> gradNormX, uint32_t totalElements)
{
    __ubuf__ T *gradXAddr = (__ubuf__ T *)gradXOut.GetPhyAddr();
    __ubuf__ float *xNormAddr = (__ubuf__ float *)xNorm.GetPhyAddr();
    __ubuf__ float *gradNormXAddr = (__ubuf__ float *)gradNormX.GetPhyAddr();
    constexpr uint32_t oneRepeatSize = VECTOR_REG_WIDTH / sizeof(float);
    uint16_t repeatTimes = static_cast<uint16_t>((totalElements + oneRepeatSize - 1) / oneRepeatSize);
    if constexpr (IsSameType<T, float>::value) {
        SigmoidGradFP32VF(xNormAddr, gradNormXAddr, gradXAddr, totalElements, oneRepeatSize, repeatTimes);
    } else {
        SigmoidGradHalfVF<T>(xNormAddr, gradNormXAddr, gradXAddr, totalElements, oneRepeatSize, repeatTimes);
    }
}

template <typename T>
__aicore__ inline void CallCastGradYFlatVF(LocalTensor<T> gradYSrc, LocalTensor<float> dst, uint32_t totalElements)
{
    __ubuf__ T *srcAddr = (__ubuf__ T *)gradYSrc.GetPhyAddr();
    __ubuf__ float *dstAddr = (__ubuf__ float *)dst.GetPhyAddr();
    constexpr uint32_t oneRepeatSize = VECTOR_REG_WIDTH / sizeof(float);
    uint16_t repeatTimes = static_cast<uint16_t>((totalElements + oneRepeatSize - 1) / oneRepeatSize);
    CastGradYFlatVF<T>(srcAddr, dstAddr, totalElements, oneRepeatSize, repeatTimes);
}

template <typename T>
__aicore__ inline void CallCastGradYRowsVF(LocalTensor<T> gradYSrc, LocalTensor<float> dst, uint16_t curRows,
                                           uint16_t k, uint16_t srcRowStride, uint16_t dstRowStride)
{
    __ubuf__ T *srcAddr = (__ubuf__ T *)gradYSrc.GetPhyAddr();
    __ubuf__ float *dstAddr = (__ubuf__ float *)dst.GetPhyAddr();
    CastGradYRowsVF<T>(srcAddr, dstAddr, curRows, k, srcRowStride, dstRowStride);
}

__aicore__ inline void CallSigmoidRenormBackwardVF(LocalTensor<float> xNorm, LocalTensor<int32_t> expertIdx,
                                                   LocalTensor<float> gradY, LocalTensor<float> gradNormX, float eps,
                                                   uint16_t curRows, uint16_t k, uint16_t kAlign, uint16_t n)
{
    __ubuf__ float *xNormAddr = (__ubuf__ float *)xNorm.GetPhyAddr();
    __ubuf__ int32_t *expertIdxAddr = (__ubuf__ int32_t *)expertIdx.GetPhyAddr();
    __ubuf__ float *gradYAddr = (__ubuf__ float *)gradY.GetPhyAddr();
    __ubuf__ float *gradNormXAddr = (__ubuf__ float *)gradNormX.GetPhyAddr();
    SigmoidRenormBackwardVF(xNormAddr, expertIdxAddr, gradYAddr, gradNormXAddr, eps, curRows, k, kAlign, n);
}

} // namespace MoeGatingTopKBackwardNs
#endif // MOE_GATING_TOP_K_BACKWARD_VF_H
