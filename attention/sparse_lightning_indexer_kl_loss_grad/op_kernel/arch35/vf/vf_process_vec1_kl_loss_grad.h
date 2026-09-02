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
 * \file vf_process_vec1_kl_loss_grad.h
 * \brief
 */
#ifndef VF_PROCESS_VEC1_KL_LOSS_GRAD_H
#define VF_PROCESS_VEC1_KL_LOSS_GRAD_H

#include "kernel_tensor.h"

namespace AscendC {
constexpr uint16_t REDUCE_SIZE = 1;
constexpr uint32_t floatRepSize = 64;
using namespace MicroAPI;
constexpr static AscendC::MicroAPI::CastTrait castTraitB162B32Odd = {
    AscendC::MicroAPI::RegLayout::ONE,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};
constexpr static AscendC::MicroAPI::CastTrait castTraitB162B32Even = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

template <typename T, typename INPUT_T>
__simd_vf__ inline void ProcessVec1BasicVF(__ubuf__ T *dstUb, __ubuf__ T *srcUb, __ubuf__ INPUT_T *weightUb,
                                           const uint32_t m, const uint32_t nBaseSize)
{
    RegTensor<float> vreg_input_x1;
    RegTensor<float> vreg_input_x2;
    RegTensor<float> vreg_input_x3;
    RegTensor<float> vreg_input_x4;
    RegTensor<float> vreg_input_x1_new;
    RegTensor<float> vreg_input_x2_new;
    RegTensor<float> vreg_input_x3_new;
    RegTensor<float> vreg_input_x4_new;
    RegTensor<float> vreg_output_x1;
    RegTensor<float> vreg_output_x2;
    RegTensor<float> vreg_output_x3;
    RegTensor<float> vreg_output_x4;
    RegTensor<half> vreg_weight_half;
    RegTensor<bfloat16_t> vreg_weight_bf;
    RegTensor<float> vreg_weight;
    RegTensor<float> vreg_zero;

    MaskReg pregFullExeB16 = CreateMask<INPUT_T, MaskPattern::ALL>();
    MaskReg preg_all_b32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_relu;

    Duplicate(vreg_output_x1, 0, preg_all_b32);
    Duplicate(vreg_output_x2, 0, preg_all_b32);
    Duplicate(vreg_output_x3, 0, preg_all_b32);
    Duplicate(vreg_output_x4, 0, preg_all_b32);
    Duplicate(vreg_zero, 0.0f);
    for (uint16_t i = 0; i < m; ++i) {
        if constexpr (IsSameType<INPUT_T, half>::value) {
            LoadAlign<INPUT_T, MicroAPI::LoadDist::DIST_BRC_B16>(vreg_weight_half, weightUb + i);
            Cast<float, INPUT_T, castTraitB162B32Even>(vreg_weight, vreg_weight_half, pregFullExeB16);
        } else if constexpr (IsSameType<INPUT_T, bfloat16_t>::value) {
            LoadAlign<INPUT_T, MicroAPI::LoadDist::DIST_BRC_B16>(vreg_weight_bf, weightUb + i);
            Cast<float, INPUT_T, castTraitB162B32Even>(vreg_weight, vreg_weight_bf, pregFullExeB16);
        } else if constexpr (IsSameType<INPUT_T, float>::value) {
            LoadAlign<INPUT_T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_weight, weightUb + i);
        }

        LoadAlign(vreg_input_x1, srcUb + i * nBaseSize);
        LoadAlign(vreg_input_x2, srcUb + i * nBaseSize + floatRepSize);
        LoadAlign(vreg_input_x3, srcUb + i * nBaseSize + floatRepSize * 2);
        LoadAlign(vreg_input_x4, srcUb + i * nBaseSize + floatRepSize * 3);
        // SY ReLu计算
        Compare<T, AscendC::CMPMODE::GT>(preg_relu, vreg_input_x1, vreg_zero, preg_all_b32);
        Select(vreg_input_x1_new, vreg_input_x1, vreg_zero, preg_relu);
        Compare<T, AscendC::CMPMODE::GT>(preg_relu, vreg_input_x2, vreg_zero, preg_all_b32);
        Select(vreg_input_x2_new, vreg_input_x2, vreg_zero, preg_relu);
        Compare<T, AscendC::CMPMODE::GT>(preg_relu, vreg_input_x3, vreg_zero, preg_all_b32);
        Select(vreg_input_x3_new, vreg_input_x3, vreg_zero, preg_relu);
        Compare<T, AscendC::CMPMODE::GT>(preg_relu, vreg_input_x4, vreg_zero, preg_all_b32);
        Select(vreg_input_x4_new, vreg_input_x4, vreg_zero, preg_relu);
        // SY 乘weights
        Mul(vreg_input_x1, vreg_input_x1_new, vreg_weight, preg_all_b32);
        Mul(vreg_input_x2, vreg_input_x2_new, vreg_weight, preg_all_b32);
        Mul(vreg_input_x3, vreg_input_x3_new, vreg_weight, preg_all_b32);
        Mul(vreg_input_x4, vreg_input_x4_new, vreg_weight, preg_all_b32);
        // SY 累加和ReduceSum
        Add(vreg_output_x1, vreg_output_x1, vreg_input_x1, preg_all_b32);
        Add(vreg_output_x2, vreg_output_x2, vreg_input_x2, preg_all_b32);
        Add(vreg_output_x3, vreg_output_x3, vreg_input_x3, preg_all_b32);
        Add(vreg_output_x4, vreg_output_x4, vreg_input_x4, preg_all_b32);
    }
    StoreAlign(((__ubuf__ T *&)dstUb), vreg_output_x1, preg_all_b32);
    StoreAlign(((__ubuf__ T *&)dstUb + floatRepSize), vreg_output_x2, preg_all_b32);
    StoreAlign(((__ubuf__ T *&)dstUb + floatRepSize * 2), vreg_output_x3, preg_all_b32);
    StoreAlign(((__ubuf__ T *&)dstUb + floatRepSize * 3), vreg_output_x4, preg_all_b32);
}

template <typename T, typename INPUT_T>
__aicore__ inline void ProcessVec1Vf(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
                                     const LocalTensor<INPUT_T> &weightTensor, const uint32_t m,
                                     const uint32_t nBaseSize)
{
    __ubuf__ T *dstUb = (__ubuf__ T *)dstTensor.GetPhyAddr();
    __ubuf__ T *srcUb = (__ubuf__ T *)srcTensor.GetPhyAddr();
    __ubuf__ INPUT_T *weightUb = (__ubuf__ INPUT_T *)weightTensor.GetPhyAddr();

    ProcessVec1BasicVF<T, INPUT_T>(dstUb, srcUb, weightUb, m, nBaseSize);
}
} // namespace AscendC

#endif // VF_PROCESS_VEC1_KL_LOSS_GRAD_H
