/* *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file vf_process_vec4_kl_loss_grad.h
 * \brief
 */
#ifndef VF_PROCESS_VEC4_KL_LOSS_GRAD_H
#define VF_PROCESS_VEC4_KL_LOSS_GRAD_H

#include "kernel_tensor.h"

namespace AscendC {
using namespace MicroAPI;

template <typename T>
__simd_vf__ inline void ProcessVec4BasicVF(__ubuf__ T *dstUb, __ubuf__ T *srcUb, __ubuf__ T *maxUb, __ubuf__ T *sumUb,
                                           float scaleValue, float pScaler, uint32_t m, const uint32_t nBaseSize)
{
    RegTensor<float> vreg_input_x1;
    RegTensor<float> vreg_input_x2;
    RegTensor<float> vreg_output_x1;
    RegTensor<float> vreg_output_x2;
    RegTensor<float> vreg_max;
    RegTensor<float> vreg_sum;
    RegTensor<float> vreg_tmp1;
    RegTensor<float> vreg_tmp2;

    MaskReg preg_all_b32 = CreateMask<float, MaskPattern::ALL>();

    Duplicate(vreg_output_x1, 0.0f, preg_all_b32);
    Duplicate(vreg_output_x2, 0.0f, preg_all_b32);
    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign(vreg_input_x1, srcUb + i * nBaseSize);
        LoadAlign(vreg_input_x2, srcUb + i * nBaseSize + floatRepSize);
        Muls(vreg_input_x1, vreg_input_x1, scaleValue, preg_all_b32);
        Muls(vreg_input_x2, vreg_input_x2, scaleValue, preg_all_b32);
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_max, maxUb + i);
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_sum, sumUb + i);
        ExpSub(vreg_tmp1, vreg_input_x1, vreg_max, preg_all_b32);
        ExpSub(vreg_tmp2, vreg_input_x2, vreg_max, preg_all_b32);
        Div(vreg_input_x1, vreg_tmp1, vreg_sum, preg_all_b32);
        Div(vreg_input_x2, vreg_tmp2, vreg_sum, preg_all_b32);
        Add(vreg_output_x1, vreg_output_x1, vreg_input_x1, preg_all_b32);
        Add(vreg_output_x2, vreg_output_x2, vreg_input_x2, preg_all_b32);
    }
    Muls(vreg_output_x1, vreg_output_x1, pScaler, preg_all_b32);
    Muls(vreg_output_x2, vreg_output_x2, pScaler, preg_all_b32);
    StoreAlign(((__ubuf__ T *&)dstUb), vreg_output_x1, preg_all_b32);
    StoreAlign(((__ubuf__ T *&)dstUb + floatRepSize), vreg_output_x2, preg_all_b32);
}

template <typename T>
__aicore__ inline void ProcessVec4Vf(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
                                     const LocalTensor<T> &maxTensor, const LocalTensor<T> &sumTensor,
                                     const float scaleValue, const float pScaler, const uint32_t m,
                                     const uint32_t nBaseSize)
{
    __ubuf__ T *dstUb = (__ubuf__ T *)dstTensor.GetPhyAddr();
    __ubuf__ T *srcUb = (__ubuf__ T *)srcTensor.GetPhyAddr();
    __ubuf__ T *maxUb = (__ubuf__ T *)maxTensor.GetPhyAddr();
    __ubuf__ T *sumUb = (__ubuf__ T *)sumTensor.GetPhyAddr();

    ProcessVec4BasicVF<T>(dstUb, srcUb, maxUb, sumUb, scaleValue, pScaler, m, nBaseSize);
}
} // namespace AscendC

#endif // VF_PROCESS_VEC4_KL_LOSS_GRAD_H
