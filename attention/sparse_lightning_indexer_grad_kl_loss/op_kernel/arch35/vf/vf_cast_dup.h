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
 * \file vf_cast_dup.h
 * \brief
 */
#ifndef VF_CAST_DUP_H
#define VF_CAST_DUP_H

#include "kernel_tensor.h"

namespace AscendC {
template <typename T, typename WEIGHT_T>
__simd_vf__ inline void CastDupBasicVF(__ubuf__ WEIGHT_T *dstUb, __ubuf__ T *srcUb, uint32_t m, uint32_t maskScaler)
{
    RegTensor<float> vreg_input;
    RegTensor<float> vreg_input_new;
    RegTensor<half> vreg_dw_half;
    RegTensor<half> vreg_dw_half_even;
    RegTensor<half> vreg_dw_half_odd;
    RegTensor<bfloat16_t> vreg_dw_bf;
    RegTensor<bfloat16_t> vreg_dw_bf_even;
    RegTensor<bfloat16_t> vreg_dw_bf_odd;
    RegTensor<float> vreg_zero;

    MaskReg pregFullExeB16 = CreateMask<WEIGHT_T, MaskPattern::ALL>();
    MaskReg preg_tail_n = UpdateMask<float>(m);
    MaskReg preg_tail = UpdateMask<WEIGHT_T>(maskScaler);

    Duplicate(vreg_zero, 0.0f);
    LoadAlign(vreg_input, srcUb);
    Select(vreg_input_new, vreg_input, vreg_zero, preg_tail_n);
    if constexpr (IsSameType<WEIGHT_T, half>::value) {
        Cast<WEIGHT_T, float, castTraitZero>(vreg_dw_half, vreg_input_new, pregFullExeB16);
        DeInterleave(vreg_dw_half_even, vreg_dw_half_odd, vreg_dw_half, vreg_dw_half);
        StoreAlign(((__ubuf__ WEIGHT_T *&)dstUb), vreg_dw_half_even, preg_tail);
    } else if constexpr (IsSameType<WEIGHT_T, bfloat16_t>::value) {
        Cast<WEIGHT_T, float, castTraitZero>(vreg_dw_bf, vreg_input_new, pregFullExeB16);
        DeInterleave(vreg_dw_bf_even, vreg_dw_bf_odd, vreg_dw_bf, vreg_dw_bf);
        StoreAlign(((__ubuf__ WEIGHT_T *&)dstUb), vreg_dw_bf_even, preg_tail);
    } else if constexpr (IsSameType<WEIGHT_T, float>::value) {
        StoreAlign(((__ubuf__ WEIGHT_T *&)dstUb), vreg_input_new, preg_tail);
    }
    StoreAlign(((__ubuf__ T *&)srcUb), vreg_zero, preg_tail_n);
}

template <typename T, typename WEIGHT_T>
__aicore__ inline void CastDupVf(const LocalTensor<WEIGHT_T> &dstTensor, const LocalTensor<T> &srcTensor,
                                 const uint32_t m)
{
    __ubuf__ WEIGHT_T *dstUb = (__ubuf__ WEIGHT_T *)dstTensor.GetPhyAddr();
    __ubuf__ T *srcUb = (__ubuf__ T *)srcTensor.GetPhyAddr();
    uint32_t maskScaler = m;

    CastDupBasicVF<T, WEIGHT_T>(dstUb, srcUb, m, maskScaler);
}
} // namespace AscendC

#endif // VF_CAST_DUP_H
