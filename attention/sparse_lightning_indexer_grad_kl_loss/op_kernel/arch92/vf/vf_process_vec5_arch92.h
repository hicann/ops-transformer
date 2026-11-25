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
 * \file vf_process_vec5_arch92.h
 * \brief
 */
#ifndef VF_PROCESS_VEC5_ARCH92_H
#define VF_PROCESS_VEC5_ARCH92_H

#include "kernel_tensor.h"
#include "../sparse_lightning_indexer_grad_kl_loss_batch_loss_arch92.h"

namespace AscendC {
using namespace Reg;

// TwoSum keeps the rounding error of each tile or partial in a second FP32 register.
__simd_callee__ inline void AddLossComponent(RegTensor<float>& sum, RegTensor<float>& correction,
                                             RegTensor<float>& value, MaskReg& mask)
{
    RegTensor<float> next, delta, sumError, valueError;
    Add(next, sum, value, mask);
    Sub(delta, next, sum, mask);
    Sub(sumError, next, delta, mask);
    Sub(sumError, sum, sumError, mask);
    Sub(valueError, value, delta, mask);
    Add(sumError, sumError, valueError, mask);
    Add(correction, correction, sumError, mask);
    sum = next;
}

__simd_vf__ inline void InitBatchLossVF(__ubuf__ float* slot, float batchTag)
{
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> zero, tag;
    Duplicate(zero, 0.0F);
    Duplicate(tag, batchTag);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(slot, zero, mask);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(slot + 1, zero, mask);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(slot + 2, tag, mask);
}

__simd_vf__ inline void MergeBatchLossVF(__ubuf__ float* out, __ubuf__ float* slots, uint16_t count)
{
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    MaskReg valid, changed;
    RegTensor<float> sum, correction, zero, previous, tag, high, low, rounded;
    Duplicate(zero, 0.0F);
    Duplicate(sum, 0.0F);
    Duplicate(correction, 0.0F);
    Duplicate(previous, 0.0F);
    for (uint16_t i = 0; i < count; ++i) {
        __ubuf__ float* slot = slots + i * SligBatchLoss::SLOT_FLOATS;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(high, slot);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(low, slot + 1);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(tag, slot + 2);
        // Slots without a segment are zero. Round the running loss once per actual batch.
        Compare<float, CMPMODE::NE>(valid, tag, zero, mask);
        Select(tag, tag, previous, valid);
        Compare<float, CMPMODE::NE>(changed, tag, previous, mask);
        Add(rounded, sum, correction, mask);
        Select(sum, rounded, sum, changed);
        Select(correction, zero, correction, changed);
        previous = tag;
        AddLossComponent(sum, correction, high, mask);
        AddLossComponent(sum, correction, low, mask);
    }
    Add(rounded, sum, correction, mask);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(out, rounded, mask);
}

template <typename T, bool isAlign, bool hasOther = true, bool batchLoss = false>
__simd_vf__ inline void ProcessVec5BasicVF(__ubuf__ T* dstUb, __ubuf__ T* srcUb, __ubuf__ T* srcOtherUb,
                                           __ubuf__ T* reduceSumUb, uint32_t alignCnt, uint32_t tailSize,
                                           float maxValue, uint32_t lossSize)
{
    RegTensor<float> vreg_input_x1;
    RegTensor<float> vreg_input_x1_new;
    RegTensor<float> vreg_input_x2;
    RegTensor<float> vreg_input_x2_new;
    RegTensor<float> vreg_input_other1;
    RegTensor<float> vreg_input_other1_new;
    RegTensor<float> vreg_max;
    RegTensor<float> vreg_zero;
    RegTensor<float> vreg_tmp1;
    RegTensor<float> vreg_tmp2;
    RegTensor<float> vreg_output_x;
    RegTensor<float> vreg_output_x_new;
    RegTensor<float> vreg_output;
    RegTensor<float> vreg_dst;
    UnalignRegForStore ureg_dst;
    MaskReg preg_loss = UpdateMask<float>(lossSize);
    MaskReg preg_tail_n = UpdateMask<float>(tailSize);
    MaskReg preg_all_b32 = CreateMask<float, MaskPattern::ALL>();

    Duplicate(vreg_max, maxValue);
    Duplicate(vreg_zero, 0.0f);
    Duplicate(vreg_output_x, 0.0f);
    LoadAlign<T, Reg::LoadDist::DIST_BRC_B32>(vreg_dst, dstUb);
    if constexpr (hasOther) {
        for (uint16_t i = 0; i < alignCnt; ++i) {
            LoadAlign(vreg_input_x1, srcUb + i * floatRepSize);
            LoadAlign(vreg_input_other1, srcOtherUb + i * floatRepSize);
            Add(vreg_input_x1, vreg_input_x1, vreg_input_other1, preg_all_b32);
            StoreAlign(((__ubuf__ T*&)srcUb + i * floatRepSize), vreg_input_x1, preg_all_b32);
        }
        if constexpr (!isAlign) {
            LoadAlign(vreg_input_x1, srcUb + alignCnt * floatRepSize);
            LoadAlign(vreg_input_other1, srcOtherUb + alignCnt * floatRepSize);
            Select(vreg_input_x1_new, vreg_input_x1, vreg_zero, preg_tail_n);
            Select(vreg_input_other1_new, vreg_input_other1, vreg_zero, preg_tail_n);
            Add(vreg_input_x1_new, vreg_input_x1_new, vreg_input_other1_new, preg_tail_n);
            StoreAlign(((__ubuf__ T*&)srcUb + alignCnt * floatRepSize), vreg_input_x1_new, preg_tail_n);
        }
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    for (uint16_t i = 0; i < alignCnt; ++i) {
        LoadAlign(vreg_input_x1, reduceSumUb + i * floatRepSize);
        LoadAlign(vreg_input_x2, srcUb + i * floatRepSize);
        Max(vreg_input_x1, vreg_max, vreg_input_x1, preg_all_b32);
        Max(vreg_input_x2, vreg_max, vreg_input_x2, preg_all_b32);
        Log(vreg_tmp1, vreg_input_x1, preg_all_b32);
        Log(vreg_tmp2, vreg_input_x2, preg_all_b32);
        Sub(vreg_tmp2, vreg_tmp2, vreg_tmp1, preg_all_b32);
        Mul(vreg_tmp2, vreg_tmp2, vreg_input_x2, preg_all_b32);
        Add(vreg_output_x, vreg_tmp2, vreg_output_x, preg_all_b32);
    }
    if constexpr (!isAlign) {
        LoadAlign(vreg_input_x1, reduceSumUb + alignCnt * floatRepSize);
        LoadAlign(vreg_input_x2, srcUb + alignCnt * floatRepSize);
        Select(vreg_input_x1_new, vreg_input_x1, vreg_zero, preg_tail_n);
        Select(vreg_input_x2_new, vreg_input_x2, vreg_zero, preg_tail_n);
        Max(vreg_input_x1_new, vreg_max, vreg_input_x1_new, preg_all_b32);
        Max(vreg_input_x2_new, vreg_max, vreg_input_x2_new, preg_all_b32);
        Log(vreg_tmp1, vreg_input_x1_new, preg_all_b32);
        Log(vreg_tmp2, vreg_input_x2_new, preg_all_b32);
        Sub(vreg_tmp2, vreg_tmp2, vreg_tmp1, preg_all_b32);
        Mul(vreg_output_x_new, vreg_tmp2, vreg_input_x2_new, preg_all_b32);
        Add(vreg_output_x, vreg_output_x_new, vreg_output_x, preg_all_b32);
    }
    Reduce<Reg::ReduceType::SUM, float, float, Reg::MaskMergeMode::ZEROING>(vreg_output, vreg_output_x, preg_all_b32);
    if constexpr (batchLoss) {
        RegTensor<float> correction;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(correction, dstUb + 1);
        AddLossComponent(vreg_dst, correction, vreg_output, preg_all_b32);
        StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstUb, vreg_dst, preg_loss);
        StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstUb + 1, correction, preg_loss);
    } else {
        Add(vreg_dst, vreg_output, vreg_dst, preg_all_b32);
        StoreAlign(((__ubuf__ T*&)dstUb), vreg_dst, preg_loss);
    }
}

template <typename T, bool isAlign, bool hasOther = true, bool batchLoss = false>
__aicore__ inline void ProcessVec5Vf(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
                                     const LocalTensor<T>& srcOtherTensor, const LocalTensor<T>& reduceSumTensor,
                                     const uint32_t m)
{
    __ubuf__ T* dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ T* srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ T* srcOtherUb = (__ubuf__ T*)srcOtherTensor.GetPhyAddr();
    __ubuf__ T* reduceSumUb = (__ubuf__ T*)reduceSumTensor.GetPhyAddr();
    uint32_t alignCnt = m >> 6;
    uint32_t tailSize = m - (m >> 6 << 6);
    float maxValue = 1e-8;
    uint32_t lossSize = 1;

    ProcessVec5BasicVF<T, isAlign, hasOther, batchLoss>(dstUb, srcUb, srcOtherUb, reduceSumUb, alignCnt, tailSize,
                                                        maxValue, lossSize);
}
} // namespace AscendC

#endif // VF_PROCESS_VEC5_ARCH92_H
