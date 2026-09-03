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
 * \file vf_vec_mul_mat.h
 * \brief
 */

#ifndef VF_VEC_MUL_MAT_H
#define VF_VEC_MUL_MAT_H

#include "kernel_tensor.h"

namespace AscendC {

template <typename T>
__simd_vf__ void VecMulMatVFImpl(__ubuf__ T *outputUb, __ubuf__ T *rowVecUb, __ubuf__ T *matrixUb,
                                 const uint32_t floatRepSize, uint16_t dLoops, uint32_t dTail, uint16_t dTailLoop,
                                 const uint16_t row, const uint32_t col)
{
    Reg::RegTensor<T> vregMatrix;
    Reg::RegTensor<T> vregRowVec;
    Reg::RegTensor<T> vregOutput;
    Reg::MaskReg fullMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg tailMask;
    tailMask = Reg::UpdateMask<float>(dTail);
    constexpr static Reg::CastTrait castTraitPack2 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                      Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    constexpr static Reg::CastTrait castTraitF32ToHalf = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::CAST_ODD};

    uint32_t colOffset = 0;
    uint32_t rowOffset = 0;
    for (uint16_t j = 0; j < dLoops; j++) {
        Reg::LoadAlign<T, Reg::LoadDist::DIST_NORM>(vregRowVec, rowVecUb + colOffset);
        rowOffset = 0;
        for (uint16_t i = 0; i < row; i++) {
            Reg::LoadAlign<T, Reg::LoadDist::DIST_NORM>(vregMatrix, matrixUb + colOffset + rowOffset);
            Reg::Mul(vregOutput, vregMatrix, vregRowVec, fullMask);
            Reg::StoreAlign<T, Reg::StoreDist::DIST_NORM>(outputUb + colOffset + rowOffset, vregOutput, fullMask);
            rowOffset += col;
        }
        colOffset += floatRepSize;
    }

    if (dTailLoop > 0) {
        rowOffset = 0;
        Reg::LoadAlign<T, Reg::LoadDist::DIST_NORM>(vregRowVec, rowVecUb + dLoops * floatRepSize);
        for (uint16_t i = 0; i < row; i++) {
            Reg::LoadAlign<T, Reg::LoadDist::DIST_NORM>(vregMatrix, matrixUb + dLoops * floatRepSize + rowOffset);
            Reg::Mul(vregOutput, vregMatrix, vregRowVec, tailMask);
            Reg::StoreAlign<T, Reg::StoreDist::DIST_NORM>(outputUb + dLoops * floatRepSize + rowOffset, vregOutput,
                                                          tailMask);
            rowOffset += col;
        }
    }
}

template <typename T>
__aicore__ inline void VecMulMatVF(const LocalTensor<T> &matrixOutUb, const LocalTensor<T> &rowVecInUb,
                                   const LocalTensor<T> &matrixInUb, const uint16_t row, const uint32_t col)
{
    __ubuf__ float *matrixUb = (__ubuf__ float *)matrixInUb.GetPhyAddr();
    __ubuf__ float *rowVecUb = (__ubuf__ float *)rowVecInUb.GetPhyAddr();
    __ubuf__ float *outputUb = (__ubuf__ float *)matrixOutUb.GetPhyAddr();

    constexpr uint16_t floatRepSize = 64; // 一个寄存器能够存放64个FP32
    uint16_t dLoops = col / floatRepSize;
    uint32_t dTail = col % floatRepSize;
    uint16_t dTailLoop = dTail > 0 ? 1 : 0;

    VecMulMatVFImpl<T>(outputUb, rowVecUb, matrixUb, floatRepSize, dLoops, dTail, dTailLoop, row, col);
}

} // namespace AscendC

#endif // VF_VEC_MUL_MAT_H
