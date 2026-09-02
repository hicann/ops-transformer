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
 * \file vf_process_vec6.h
 * \brief
 */
#ifndef VF_PROCESS_VEC6_H
#define VF_PROCESS_VEC6_H

#include "kernel_tensor.h"

namespace AscendC {
using namespace MicroAPI;
constexpr static AscendC::MicroAPI::CastTrait castTraitZero = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};
template <typename T, typename INPUT_T, typename WEIGHT_T, bool isAlign>
__simd_vf__ inline void ProcessVec6BasicVF(__ubuf__ INPUT_T *dstUb, __ubuf__ T *dwUb, __ubuf__ T *srcUb,
                                           __ubuf__ WEIGHT_T *weightUb, __ubuf__ T *reduceSumUb, __ubuf__ T *reluUb,
                                           uint32_t regCnt, uint32_t tailSize, uint32_t nIndexSize, uint32_t nBaseSize,
                                           uint32_t repeatSize, uint32_t blockStride)
{
    RegTensor<float> vreg_input_x1;
    RegTensor<float> vreg_input_x2;
    RegTensor<float> vreg_input_x1_new;
    RegTensor<float> vreg_input_x2_new;
    RegTensor<float> vreg_input_sub;
    RegTensor<float> vreg_input_sub_new;
    RegTensor<float> vreg_zero;
    RegTensor<half> vreg_weight_half;
    RegTensor<bfloat16_t> vreg_weight_bf;
    RegTensor<half> vreg_relugrad_half;
    RegTensor<bfloat16_t> vreg_relugrad_bf;
    RegTensor<half> vreg_relugrad_half_even;
    RegTensor<bfloat16_t> vreg_relugrad_bf_even;
    RegTensor<half> vreg_relugrad_half_odd;
    RegTensor<bfloat16_t> vreg_relugrad_bf_odd;
    RegTensor<float> vreg_weight;
    RegTensor<float> vreg_output_dw;
    RegTensor<float> vreg_out_dw;
    RegTensor<float> vreg_dst;
    RegTensor<float> vreg_tmp1;
    RegTensor<float> vreg_tmp2;
    UnalignRegForStore ureg_dst;
    MaskReg pregFullExeB16 = CreateMask<INPUT_T, MaskPattern::ALL>();
    MaskReg preg_tail_n = UpdateMask<float>(tailSize);
    MaskReg preg_all_b32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_relugrad;
    MaskReg preg_bf16 = UpdateMask<uint16_t>(repeatSize);

    Duplicate(vreg_zero, 0.0f);
    if constexpr (!isAlign) {
        LoadAlign(vreg_input_x1, srcUb + regCnt * floatRepSize);
        LoadAlign(vreg_input_x2, reduceSumUb + regCnt * floatRepSize);
        Select(vreg_input_x1_new, vreg_input_x1, vreg_zero, preg_tail_n);
        Select(vreg_input_x2_new, vreg_input_x2, vreg_zero, preg_tail_n);
        Sub(vreg_input_sub_new, vreg_input_x2_new, vreg_input_x1_new, preg_all_b32);
        StoreAlign(((__ubuf__ T *&)reduceSumUb + regCnt * floatRepSize), vreg_input_sub_new, preg_all_b32);
    } else {
        LoadAlign(vreg_input_x1, srcUb + regCnt * floatRepSize);
        LoadAlign(vreg_input_x2, reduceSumUb + regCnt * floatRepSize);
        Sub(vreg_input_sub, vreg_input_x2, vreg_input_x1, preg_all_b32);
        StoreAlign(((__ubuf__ T *&)reduceSumUb + regCnt * floatRepSize), vreg_input_sub, preg_all_b32);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    for (uint16_t i = 0; i < nIndexSize; ++i) {
        Duplicate(vreg_output_dw, 0.0f);
        if constexpr (IsSameType<WEIGHT_T, half>::value) {
            LoadAlign<WEIGHT_T, MicroAPI::LoadDist::DIST_BRC_B16>(vreg_weight_half, weightUb + i);
            Cast<float, WEIGHT_T, castTraitB162B32Even>(vreg_weight, vreg_weight_half, pregFullExeB16);
        } else if constexpr (IsSameType<WEIGHT_T, bfloat16_t>::value) {
            LoadAlign<WEIGHT_T, MicroAPI::LoadDist::DIST_BRC_B16>(vreg_weight_bf, weightUb + i);
            Cast<float, WEIGHT_T, castTraitB162B32Even>(vreg_weight, vreg_weight_bf, pregFullExeB16);
        } else if constexpr (IsSameType<WEIGHT_T, float>::value) {
            LoadAlign<WEIGHT_T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_weight, weightUb + i);
        }
        if constexpr (!isAlign) {
            LoadAlign(vreg_input_sub_new, reduceSumUb + regCnt * floatRepSize);
            LoadAlign(vreg_input_x1, reluUb + i * nBaseSize + regCnt * floatRepSize);
            Select(vreg_input_x1_new, vreg_input_x1, vreg_zero, preg_tail_n);
            Mul(vreg_tmp1, vreg_input_sub_new, vreg_input_x1_new, preg_all_b32);
            Add(vreg_output_dw, vreg_tmp1, vreg_output_dw, preg_all_b32);
            Mul(vreg_tmp1, vreg_input_sub_new, vreg_weight, preg_all_b32);
            Compare<T, AscendC::CMPMODE::GT>(preg_relugrad, vreg_input_x1_new, vreg_zero, preg_tail_n);
            Select(vreg_tmp2, vreg_tmp1, vreg_zero, preg_relugrad);
            if constexpr (IsSameType<INPUT_T, half>::value) {
                Cast<INPUT_T, float, castTraitZero>(vreg_relugrad_half, vreg_tmp2, pregFullExeB16);
                DeInterleave(vreg_relugrad_half_even, vreg_relugrad_half_odd, vreg_relugrad_half, vreg_relugrad_half);
                StoreAlign<INPUT_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ INPUT_T *&)dstUb), vreg_relugrad_half_even, blockStride, 1, preg_bf16);
            } else if constexpr (IsSameType<INPUT_T, bfloat16_t>::value) {
                Cast<INPUT_T, float, castTraitZero>(vreg_relugrad_bf, vreg_tmp2, pregFullExeB16);
                DeInterleave(vreg_relugrad_bf_even, vreg_relugrad_bf_odd, vreg_relugrad_bf, vreg_relugrad_bf);
                StoreAlign<INPUT_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ INPUT_T *&)dstUb), vreg_relugrad_bf_even, blockStride, 1, preg_bf16);
            }
        } else {
            LoadAlign(vreg_input_sub, reduceSumUb + regCnt * floatRepSize);
            LoadAlign(vreg_input_x1, reluUb + i * nBaseSize + regCnt * floatRepSize);
            Mul(vreg_tmp1, vreg_input_sub, vreg_input_x1, preg_all_b32);
            Add(vreg_output_dw, vreg_tmp1, vreg_output_dw, preg_all_b32);
            Mul(vreg_tmp1, vreg_input_sub, vreg_weight, preg_all_b32);
            Compare<T, AscendC::CMPMODE::GT>(preg_relugrad, vreg_input_x1, vreg_zero, preg_all_b32);
            Select(vreg_tmp2, vreg_tmp1, vreg_zero, preg_relugrad);
            if constexpr (IsSameType<INPUT_T, half>::value) {
                Cast<INPUT_T, float, castTraitZero>(vreg_relugrad_half, vreg_tmp2, pregFullExeB16);
                DeInterleave(vreg_relugrad_half_even, vreg_relugrad_half_odd, vreg_relugrad_half, vreg_relugrad_half);
                StoreAlign<INPUT_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ INPUT_T *&)dstUb), vreg_relugrad_half_even, blockStride, 1, preg_bf16);
            } else if constexpr (IsSameType<INPUT_T, bfloat16_t>::value) {
                Cast<INPUT_T, float, castTraitZero>(vreg_relugrad_bf, vreg_tmp2, pregFullExeB16);
                DeInterleave(vreg_relugrad_bf_even, vreg_relugrad_bf_odd, vreg_relugrad_bf, vreg_relugrad_bf);
                StoreAlign<INPUT_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ INPUT_T *&)dstUb), vreg_relugrad_bf_even, blockStride, 1, preg_bf16);
            }
        }
        Reduce<MicroAPI::ReduceType::SUM, float, float, MicroAPI::MaskMergeMode::ZEROING>(vreg_out_dw, vreg_output_dw,
                                                                                          preg_all_b32);
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_dst, (__ubuf__ T *&)dwUb);
        Add(vreg_dst, vreg_out_dw, vreg_dst, preg_all_b32);
        StoreUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)dwUb), vreg_dst, ureg_dst, 1);
    }
    StoreUnAlignPost<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)dwUb), ureg_dst, 0);
}

template <typename T, typename INPUT_T, typename WEIGHT_T, bool isAlign>
__aicore__ inline void ProcessVec6Vf(const LocalTensor<INPUT_T> &dstTensor, const LocalTensor<T> &dwTensor,
                                     const LocalTensor<T> &srcTensor, const LocalTensor<WEIGHT_T> &weightTensor,
                                     const LocalTensor<T> &reduceSumTensor, const LocalTensor<T> &reluTensor,
                                     const uint32_t m, const uint32_t nIndexSize, const uint32_t nBaseSize)
{
    uint32_t alignCnt = m >> 6;
    uint32_t tailSize = m - (m >> 6 << 6);
    uint32_t s2BaseSize = (m + 63) >> 6 << 6;
    uint32_t repeatSize = 64;
    uint32_t blockStride = nIndexSize;
    __ubuf__ T *srcUb = (__ubuf__ T *)srcTensor.GetPhyAddr();
    __ubuf__ T *reluUb = (__ubuf__ T *)reluTensor.GetPhyAddr();
    __ubuf__ T *reduceSumUb = (__ubuf__ T *)reduceSumTensor.GetPhyAddr();
    __ubuf__ WEIGHT_T *weightUb = (__ubuf__ WEIGHT_T *)weightTensor.GetPhyAddr();
    __ubuf__ T *dwUb = (__ubuf__ T *)dwTensor.GetPhyAddr();

    for (uint32_t regCnt = 0; regCnt < alignCnt; ++regCnt) {
        __ubuf__ INPUT_T *dstUb =
            (__ubuf__ INPUT_T *)dstTensor.GetPhyAddr() + (regCnt * nIndexSize) * (s2BaseSize >> 1);
        ProcessVec6BasicVF<T, INPUT_T, WEIGHT_T, true>(dstUb, dwUb, srcUb, weightUb, reduceSumUb, reluUb, regCnt,
                                                       tailSize, nIndexSize, nBaseSize, repeatSize, blockStride);
    }
    if constexpr (!isAlign) {
        __ubuf__ INPUT_T *dstUb =
            (__ubuf__ INPUT_T *)dstTensor.GetPhyAddr() + (alignCnt * nIndexSize) * (s2BaseSize >> 1);
        ProcessVec6BasicVF<T, INPUT_T, WEIGHT_T, false>(dstUb, dwUb, srcUb, weightUb, reduceSumUb, reluUb, alignCnt,
                                                        tailSize, nIndexSize, nBaseSize, repeatSize, blockStride);
    }
}
} // namespace AscendC

#endif // VF_PROCESS_VEC6_H
