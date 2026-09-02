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
 * \file bsa_vec_sm_vf.h
 * \brief A5 SIMD_VF helpers for BSA softmax vector path.
 */
#ifndef BSA_VEC_SM_VF_H
#define BSA_VEC_SM_VF_H

#include "kernel_operator.h"

namespace BSASelectBlockMaskVF {
using namespace AscendC;
using namespace MicroAPI;

__simd_vf__ inline void DoSoftmaxSecondPassFp32VF(__ubuf__ float *scoreUb, __ubuf__ float *rowMaxUb,
                                                  __ubuf__ float *rowSumUb, float scaleValue, uint32_t row,
                                                  uint32_t col)
{
    RegTensor<float> scoreReg;
    RegTensor<float> maxReg;
    RegTensor<float> sumReg;
    RegTensor<float> tmpReg;

    constexpr uint32_t countPerRepeat = 64;
    for (uint32_t rowIdx = 0; rowIdx < row; ++rowIdx) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(maxReg, rowMaxUb + rowIdx);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(sumReg, rowSumUb + rowIdx);
        for (uint32_t colIdx = 0; colIdx < col; colIdx += countPerRepeat) {
            uint32_t curCount = (col - colIdx) >= countPerRepeat ? countPerRepeat : (col - colIdx);
            MaskReg preg = UpdateMask<float>(curCount);
            __ubuf__ float *scoreAddr = scoreUb + rowIdx * col + colIdx;
            LoadAlign(scoreReg, scoreAddr);
            Muls(scoreReg, scoreReg, scaleValue, preg);
            Sub(tmpReg, scoreReg, maxReg, preg);
            Exp(tmpReg, tmpReg, preg);
            Div(scoreReg, tmpReg, sumReg, preg);
            StoreAlign(scoreAddr, scoreReg, preg);
        }
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

__simd_vf__ inline void DoSoftmaxFirstPassExpFp32VF(__ubuf__ float *scoreUb, __ubuf__ float *rowMaxUb, uint32_t row,
                                                    uint32_t col)
{
    RegTensor<float> scoreReg;
    RegTensor<float> maxReg;
    RegTensor<float> tmpReg;

    constexpr uint32_t countPerRepeat = 64;
    for (uint32_t rowIdx = 0; rowIdx < row; ++rowIdx) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(maxReg, rowMaxUb + rowIdx);
        for (uint32_t colIdx = 0; colIdx < col; colIdx += countPerRepeat) {
            uint32_t curCount = (col - colIdx) >= countPerRepeat ? countPerRepeat : (col - colIdx);
            MaskReg preg = UpdateMask<float>(curCount);
            __ubuf__ float *scoreAddr = scoreUb + rowIdx * col + colIdx;
            LoadAlign(scoreReg, scoreAddr);
            Sub(tmpReg, scoreReg, maxReg, preg);
            Exp(scoreReg, tmpReg, preg);
            StoreAlign(scoreAddr, scoreReg, preg);
        }
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

__aicore__ inline void DoSoftmaxSecondPassFp32VfWrapper(const LocalTensor<float> &scoreUb,
                                                        const LocalTensor<float> &rowMaxUb,
                                                        const LocalTensor<float> &rowSumUb, float scaleValue,
                                                        uint32_t row, uint32_t col)
{
    __ubuf__ float *scoreAddr = (__ubuf__ float *)scoreUb.GetPhyAddr();
    __ubuf__ float *rowMaxAddr = (__ubuf__ float *)rowMaxUb.GetPhyAddr();
    __ubuf__ float *rowSumAddr = (__ubuf__ float *)rowSumUb.GetPhyAddr();
    DoSoftmaxSecondPassFp32VF(scoreAddr, rowMaxAddr, rowSumAddr, scaleValue, row, col);
}

__aicore__ inline void DoSoftmaxFirstPassExpFp32VfWrapper(const LocalTensor<float> &scoreUb,
                                                          const LocalTensor<float> &rowMaxUb, uint32_t row,
                                                          uint32_t col)
{
    __ubuf__ float *scoreAddr = (__ubuf__ float *)scoreUb.GetPhyAddr();
    __ubuf__ float *rowMaxAddr = (__ubuf__ float *)rowMaxUb.GetPhyAddr();
    DoSoftmaxFirstPassExpFp32VF(scoreAddr, rowMaxAddr, row, col);
}

} // namespace BSASelectBlockMaskVF

#endif // BSA_VEC_SM_VF_H
