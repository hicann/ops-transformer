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
 * \file vf_process_vec2_kl_loss_grad.h
 * \brief
 */
#ifndef VF_PROCESS_VEC2_KL_LOSS_GRAD_H
#define VF_PROCESS_VEC2_KL_LOSS_GRAD_H

#include "kernel_tensor.h"

namespace AscendC {
using namespace Reg;

template <typename T, bool isAlign>
__simd_vf__ inline void ProcessVec2BasicVF(__ubuf__ T* dstUb, __ubuf__ T* srcUb, __ubuf__ T* srcOtherUb,
                                           const uint32_t alignCnt, uint32_t tailSize, float minValue)
{
    RegTensor<float> vreg_input_other;
    RegTensor<float> vreg_input_other_new;
    RegTensor<float> vreg_input_x;
    RegTensor<float> vreg_input_x_new;
    RegTensor<float> vreg_tmp;
    RegTensor<float> vreg_tmp_sum;
    RegTensor<float> vreg_input_max;
    RegTensor<float> vreg_max;
    RegTensor<float> vreg_sum;
    RegTensor<float> vreg_input_sum;
    RegTensor<float> vreg_min;
    MaskReg preg_all_b32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_tail_n = UpdateMask<float>(tailSize);

    Duplicate(vreg_min, minValue);
    Duplicate(vreg_tmp, minValue);
    Duplicate(vreg_tmp_sum, 0.0f);

    for (uint16_t i = 0; i < alignCnt; ++i) {
        LoadAlign(vreg_input_x, srcUb + i * floatRepSize);
        LoadAlign(vreg_input_other, srcOtherUb + i * floatRepSize);
        Add(vreg_input_x, vreg_input_x, vreg_input_other, preg_all_b32);
        StoreAlign(((__ubuf__ T*&)srcUb + i * floatRepSize), vreg_input_x, preg_all_b32);
        Max(vreg_tmp, vreg_tmp, vreg_input_x, preg_all_b32);
    }
    if constexpr (!isAlign) {
        LoadAlign(vreg_input_x, srcUb + alignCnt * floatRepSize);
        LoadAlign(vreg_input_other, srcOtherUb + alignCnt * floatRepSize);
        Select(vreg_input_x_new, vreg_input_x, vreg_min, preg_tail_n);
        Select(vreg_input_other_new, vreg_input_other, vreg_min, preg_tail_n);
        Add(vreg_input_x_new, vreg_input_x_new, vreg_input_other_new, preg_all_b32);
        StoreAlign(((__ubuf__ T*&)srcUb + alignCnt * floatRepSize), vreg_input_x_new, preg_all_b32);
        Max(vreg_tmp, vreg_tmp, vreg_input_x_new, preg_all_b32);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    Reduce<Reg::ReduceType::MAX, float, float, Reg::MaskMergeMode::ZEROING>(vreg_input_max, vreg_tmp, preg_all_b32);
    Duplicate(vreg_max, vreg_input_max, preg_all_b32);
    for (uint16_t i = 0; i < alignCnt; ++i) {
        LoadAlign(vreg_input_x, srcUb + i * floatRepSize);
        ExpSub(vreg_tmp, vreg_input_x, vreg_max, preg_all_b32);
        StoreAlign(((__ubuf__ T*&)dstUb + i * floatRepSize), vreg_tmp, preg_all_b32);
        Add(vreg_tmp_sum, vreg_tmp, vreg_tmp_sum, preg_all_b32);
    }
    if constexpr (!isAlign) {
        LoadAlign(vreg_input_x_new, srcUb + alignCnt * floatRepSize);
        ExpSub(vreg_tmp, vreg_input_x_new, vreg_max, preg_all_b32);
        StoreAlign(((__ubuf__ T*&)dstUb + alignCnt * floatRepSize), vreg_tmp, preg_all_b32);
        Add(vreg_tmp_sum, vreg_tmp, vreg_tmp_sum, preg_all_b32);
    }
    Reduce<Reg::ReduceType::SUM, float, float, Reg::MaskMergeMode::ZEROING>(vreg_input_sum, vreg_tmp_sum, preg_all_b32);
    Duplicate(vreg_sum, vreg_input_sum, preg_all_b32);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    for (uint16_t i = 0; i < alignCnt; ++i) {
        LoadAlign(vreg_input_x, dstUb + i * floatRepSize);
        Div(vreg_tmp, vreg_input_x, vreg_sum, preg_all_b32);
        StoreAlign(((__ubuf__ T*&)dstUb + i * floatRepSize), vreg_tmp, preg_all_b32);
    }
    if constexpr (!isAlign) {
        LoadAlign(vreg_input_x_new, dstUb + alignCnt * floatRepSize);
        Div(vreg_tmp, vreg_input_x_new, vreg_sum, preg_all_b32);
        StoreAlign(((__ubuf__ T*&)dstUb + alignCnt * floatRepSize), vreg_tmp, preg_all_b32);
    }
}

template <typename T, bool isAlign>
__aicore__ inline void ProcessVec2Vf(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
                                     const LocalTensor<T>& srcOtherTensor, const uint32_t m)
{
    __ubuf__ T* dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ T* srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ T* srcOtherUb = (__ubuf__ T*)srcOtherTensor.GetPhyAddr();
    uint32_t alignCnt = m >> 6;
    uint32_t tailSize = m - (m >> 6 << 6);
    constexpr uint32_t tmpMin = 0xFF7FFFFF;
    float minValue = *((float*)&tmpMin);

    ProcessVec2BasicVF<T, isAlign>(dstUb, srcUb, srcOtherUb, alignCnt, tailSize, minValue);
}

template <typename T, bool isAlign>
__simd_vf__ inline void ProcessVec2SingleBasicVF(__ubuf__ T* dstUb, __ubuf__ T* srcUb, const uint32_t alignCnt,
                                                 uint32_t tailSize, float minValue)
{
    RegTensor<float> vregInput;
    RegTensor<float> vregInputTail;
    RegTensor<float> vregTmp;
    RegTensor<float> vregTmpSum;
    RegTensor<float> vregInputMax;
    RegTensor<float> vregMax;
    RegTensor<float> vregInputSum;
    RegTensor<float> vregSum;
    RegTensor<float> vregMin;
    MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
    MaskReg pregTail = UpdateMask<float>(tailSize);

    Duplicate(vregMin, minValue);
    Duplicate(vregTmp, minValue);
    Duplicate(vregTmpSum, 0.0f);
    for (uint16_t i = 0; i < alignCnt; ++i) {
        LoadAlign(vregInput, srcUb + i * floatRepSize);
        Max(vregTmp, vregTmp, vregInput, pregAll);
    }
    if constexpr (!isAlign) {
        LoadAlign(vregInput, srcUb + alignCnt * floatRepSize);
        Select(vregInputTail, vregInput, vregMin, pregTail);
        StoreAlign(((__ubuf__ T*&)srcUb + alignCnt * floatRepSize), vregInputTail, pregAll);
        Max(vregTmp, vregTmp, vregInputTail, pregAll);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    Reduce<Reg::ReduceType::MAX, float, float, Reg::MaskMergeMode::ZEROING>(vregInputMax, vregTmp, pregAll);
    Duplicate(vregMax, vregInputMax, pregAll);
    for (uint16_t i = 0; i < alignCnt; ++i) {
        LoadAlign(vregInput, srcUb + i * floatRepSize);
        ExpSub(vregTmp, vregInput, vregMax, pregAll);
        StoreAlign(((__ubuf__ T*&)dstUb + i * floatRepSize), vregTmp, pregAll);
        Add(vregTmpSum, vregTmp, vregTmpSum, pregAll);
    }
    if constexpr (!isAlign) {
        LoadAlign(vregInput, srcUb + alignCnt * floatRepSize);
        ExpSub(vregTmp, vregInput, vregMax, pregAll);
        StoreAlign(((__ubuf__ T*&)dstUb + alignCnt * floatRepSize), vregTmp, pregAll);
        Add(vregTmpSum, vregTmp, vregTmpSum, pregAll);
    }
    Reduce<Reg::ReduceType::SUM, float, float, Reg::MaskMergeMode::ZEROING>(vregInputSum, vregTmpSum, pregAll);
    Duplicate(vregSum, vregInputSum, pregAll);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    for (uint16_t i = 0; i < alignCnt; ++i) {
        LoadAlign(vregInput, dstUb + i * floatRepSize);
        Div(vregTmp, vregInput, vregSum, pregAll);
        StoreAlign(((__ubuf__ T*&)dstUb + i * floatRepSize), vregTmp, pregAll);
    }
    if constexpr (!isAlign) {
        LoadAlign(vregInput, dstUb + alignCnt * floatRepSize);
        Div(vregTmp, vregInput, vregSum, pregAll);
        StoreAlign(((__ubuf__ T*&)dstUb + alignCnt * floatRepSize), vregTmp, pregAll);
    }
}

template <typename T, bool isAlign>
__aicore__ inline void ProcessVec2SingleVf(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
                                           const uint32_t m)
{
    __ubuf__ T* dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ T* srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();
    uint32_t alignCnt = m >> 6;
    uint32_t tailSize = m - (m >> 6 << 6);
    constexpr uint32_t tmpMin = 0xFF7FFFFF;
    float minValue = *((float*)&tmpMin);
    ProcessVec2SingleBasicVF<T, isAlign>(dstUb, srcUb, alignCnt, tailSize, minValue);
}
} // namespace AscendC

#endif // VF_PROCESS_VEC2_KL_LOSS_GRAD_H
