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
 * \file vf_process_vec1_dense_kl_loss_grad.h
 * \brief
 */
#ifndef VF_PROCESS_VEC1_DENSE_KL_LOSS_GRAD_H
#define VF_PROCESS_VEC1_DENSE_KL_LOSS_GRAD_H

#include "kernel_tensor.h"

namespace AscendC {
constexpr uint16_t REDUCE_SIZE = 1;
constexpr uint16_t UNROLL_SIZE = 8;
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
                                           uint32_t loopNum, uint32_t tailSize, uint32_t repeatTime,
                                           const uint32_t nBaseSize)
{
    RegTensor<float> vreg_input_x1;
    RegTensor<float> vreg_input_x2;
    RegTensor<float> vreg_input_x3;
    RegTensor<float> vreg_input_x4;
    RegTensor<float> vreg_input_x5;
    RegTensor<float> vreg_input_x6;
    RegTensor<float> vreg_input_x7;
    RegTensor<float> vreg_input_x8;

    RegTensor<float> vreg_output_x18;
    RegTensor<float> vreg_output_x27;
    RegTensor<float> vreg_output_x36;
    RegTensor<float> vreg_output_x45;
    RegTensor<float> vreg_output_tail;

    RegTensor<float> vreg_weight1;
    RegTensor<float> vreg_weight2;
    RegTensor<float> vreg_weight3;
    RegTensor<float> vreg_weight4;
    RegTensor<float> vreg_weight5;
    RegTensor<float> vreg_weight6;
    RegTensor<float> vreg_weight7;
    RegTensor<float> vreg_weight8;

    MaskReg pregFullExeB16 = CreateMask<INPUT_T, MaskPattern::ALL>();
    MaskReg preg_all_b32 = CreateMask<float, MaskPattern::ALL>();

    for (uint16_t idx = 0; idx < repeatTime; idx++) {
        RegTensor<float> vreg_output;
        Duplicate(vreg_output, 0, preg_all_b32);
        Duplicate(vreg_output_tail, 0, preg_all_b32);
        for (uint16_t i = 0; i < loopNum; i++) {
            LoadAlign<INPUT_T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_weight1, weightUb + i * UNROLL_SIZE);
            LoadAlign(vreg_input_x1, srcUb + (i * UNROLL_SIZE + 0) * nBaseSize + idx * floatRepSize);
            Relu(vreg_input_x1, vreg_input_x1, preg_all_b32);
            StoreAlign(((__ubuf__ T *&)srcUb + (i * UNROLL_SIZE + 0) * nBaseSize + idx * floatRepSize), vreg_input_x1,
                       preg_all_b32);
            Mul(vreg_input_x1, vreg_input_x1, vreg_weight1, preg_all_b32);

            LoadAlign<INPUT_T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_weight2, weightUb + i * UNROLL_SIZE + 1);
            LoadAlign(vreg_input_x2, srcUb + (i * UNROLL_SIZE + 1) * nBaseSize + idx * floatRepSize);
            Relu(vreg_input_x2, vreg_input_x2, preg_all_b32);
            StoreAlign(((__ubuf__ T *&)srcUb + (i * UNROLL_SIZE + 1) * nBaseSize + idx * floatRepSize), vreg_input_x2,
                       preg_all_b32);
            Mul(vreg_input_x2, vreg_input_x2, vreg_weight2, preg_all_b32);

            LoadAlign<INPUT_T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_weight3, weightUb + i * UNROLL_SIZE + 2);
            LoadAlign(vreg_input_x3, srcUb + (i * UNROLL_SIZE + 2) * nBaseSize + idx * floatRepSize);
            Relu(vreg_input_x3, vreg_input_x3, preg_all_b32);
            StoreAlign(((__ubuf__ T *&)srcUb + (i * UNROLL_SIZE + 2) * nBaseSize + idx * floatRepSize), vreg_input_x3,
                       preg_all_b32);
            Mul(vreg_input_x3, vreg_input_x3, vreg_weight3, preg_all_b32);

            LoadAlign<INPUT_T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_weight4, weightUb + i * UNROLL_SIZE + 3);
            LoadAlign(vreg_input_x4, srcUb + (i * UNROLL_SIZE + 3) * nBaseSize + idx * floatRepSize);
            Relu(vreg_input_x4, vreg_input_x4, preg_all_b32);
            StoreAlign(((__ubuf__ T *&)srcUb + (i * UNROLL_SIZE + 3) * nBaseSize + idx * floatRepSize), vreg_input_x4,
                       preg_all_b32);
            Mul(vreg_input_x4, vreg_input_x4, vreg_weight4, preg_all_b32);

            LoadAlign<INPUT_T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_weight5, weightUb + i * UNROLL_SIZE + 4);
            LoadAlign(vreg_input_x5, srcUb + (i * UNROLL_SIZE + 4) * nBaseSize + idx * floatRepSize);
            Relu(vreg_input_x5, vreg_input_x5, preg_all_b32);
            StoreAlign(((__ubuf__ T *&)srcUb + (i * UNROLL_SIZE + 4) * nBaseSize + idx * floatRepSize), vreg_input_x5,
                       preg_all_b32);
            Mul(vreg_input_x5, vreg_input_x5, vreg_weight5, preg_all_b32);

            LoadAlign<INPUT_T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_weight6, weightUb + i * UNROLL_SIZE + 5);
            LoadAlign(vreg_input_x6, srcUb + (i * UNROLL_SIZE + 5) * nBaseSize + idx * floatRepSize);
            Relu(vreg_input_x6, vreg_input_x6, preg_all_b32);
            StoreAlign(((__ubuf__ T *&)srcUb + (i * UNROLL_SIZE + 5) * nBaseSize + idx * floatRepSize), vreg_input_x6,
                       preg_all_b32);
            Mul(vreg_input_x6, vreg_input_x6, vreg_weight6, preg_all_b32);

            LoadAlign<INPUT_T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_weight7, weightUb + i * UNROLL_SIZE + 6);
            LoadAlign(vreg_input_x7, srcUb + (i * UNROLL_SIZE + 6) * nBaseSize + idx * floatRepSize);
            Relu(vreg_input_x7, vreg_input_x7, preg_all_b32);
            StoreAlign(((__ubuf__ T *&)srcUb + (i * UNROLL_SIZE + 6) * nBaseSize + idx * floatRepSize), vreg_input_x7,
                       preg_all_b32);
            Mul(vreg_input_x7, vreg_input_x7, vreg_weight7, preg_all_b32);

            LoadAlign<INPUT_T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_weight8, weightUb + i * UNROLL_SIZE + 7);
            LoadAlign(vreg_input_x8, srcUb + (i * UNROLL_SIZE + 7) * nBaseSize + idx * floatRepSize);
            Relu(vreg_input_x8, vreg_input_x8, preg_all_b32);
            StoreAlign(((__ubuf__ T *&)srcUb + (i * UNROLL_SIZE + 7) * nBaseSize + idx * floatRepSize), vreg_input_x8,
                       preg_all_b32);
            Mul(vreg_input_x8, vreg_input_x8, vreg_weight8, preg_all_b32);

            Add(vreg_output_x18, vreg_input_x1, vreg_input_x8, preg_all_b32);
            Add(vreg_output_x27, vreg_input_x2, vreg_input_x7, preg_all_b32);
            Add(vreg_output_x36, vreg_input_x3, vreg_input_x6, preg_all_b32);
            Add(vreg_output_x45, vreg_input_x4, vreg_input_x5, preg_all_b32);
            Add(vreg_output_x18, vreg_output_x18, vreg_output_x27, preg_all_b32);
            Add(vreg_output_x36, vreg_output_x36, vreg_output_x45, preg_all_b32);
            Add(vreg_output_x18, vreg_output_x18, vreg_output_x36, preg_all_b32);
            Add(vreg_output, vreg_output, vreg_output_x18, preg_all_b32);
        }
        for (uint16_t j = 0; j < tailSize; j++) {
            LoadAlign<INPUT_T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_weight1, weightUb + loopNum * UNROLL_SIZE + j);
            LoadAlign(vreg_input_x1, srcUb + (loopNum * UNROLL_SIZE + j) * nBaseSize + idx * floatRepSize);
            Relu(vreg_input_x1, vreg_input_x1, preg_all_b32);
            StoreAlign(((__ubuf__ T *&)srcUb + (loopNum * UNROLL_SIZE + j) * nBaseSize + idx * floatRepSize),
                       vreg_input_x1, preg_all_b32);
            Mul(vreg_input_x1, vreg_input_x1, vreg_weight1, preg_all_b32);
            Add(vreg_output_tail, vreg_output_tail, vreg_input_x1, preg_all_b32);
        }
        Add(vreg_output, vreg_output, vreg_output_tail, preg_all_b32);
        StoreAlign(((__ubuf__ T *&)dstUb + idx * floatRepSize), vreg_output, preg_all_b32);
    }
}

template <typename T, typename INPUT_T>
__aicore__ inline void ProcessVec1Vf(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
                                     const LocalTensor<INPUT_T> &weightTensor, const uint32_t nSize,
                                     const uint32_t nBaseSize)
{
    __ubuf__ T *dstUb = (__ubuf__ T *)dstTensor.GetPhyAddr();
    __ubuf__ T *srcUb = (__ubuf__ T *)srcTensor.GetPhyAddr();
    __ubuf__ INPUT_T *weightUb = (__ubuf__ INPUT_T *)weightTensor.GetPhyAddr();
    uint32_t loopNum = nSize / UNROLL_SIZE;
    uint32_t tailSize = nSize % UNROLL_SIZE;
    uint32_t repeatTime = nBaseSize / floatRepSize;
    ProcessVec1BasicVF<T, INPUT_T>(dstUb, srcUb, weightUb, loopNum, tailSize, repeatTime, nBaseSize);
}
} // namespace AscendC

#endif // VF_PROCESS_VEC1_DENSE_KL_LOSS_GRAD_H
