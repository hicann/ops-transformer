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
 * \file vf_mul_sel_softmaxflashv2_cast_nz_dn_sfa.h
 * \brief
 */
#ifndef MUL_SEL_SOFTMAXFLASHV2_CAST_NZ_DN_SFA_H_
#define MUL_SEL_SOFTMAXFLASHV2_CAST_NZ_DN_SFA_H_

#include "kernel_tensor.h"
#include "../../arch35/vf/vf_basic_block_utils.h"

namespace FaVectorApi {
using AscendC::LocalTensor;
using namespace AscendC;
using namespace Reg;

template <typename T2, bool isUpdate>
__simd_vf__ inline void MqsmlaVec1Dn64VF(__ubuf__ T2* x_exp, __ubuf__ float* input_x_local_UB, __ubuf__ float* exp_max,
                                         __ubuf__ float* new_global_sum, __ubuf__ float* new_global_max,
                                         const uint32_t m, const uint32_t n, const uint32_t originN, const float scale,
                                         const float minValue, const uint32_t blockStride, const uint32_t repeatStride)
{
    __ubuf__ float* src_ub_0 = input_x_local_UB;
    __ubuf__ float* src_ub_1 = src_ub_0 + m;
    __ubuf__ float* src_ub_2 = src_ub_0 + m * 2;
    __ubuf__ float* src_ub_3 = src_ub_0 + m * 3;
    __ubuf__ float* src_ub_4 = src_ub_0 + m * 4;
    __ubuf__ float* src_ub_5 = src_ub_0 + m * 5;
    __ubuf__ float* src_ub_6 = src_ub_0 + m * 6;
    __ubuf__ float* src_ub_7 = src_ub_0 + m * 7;

    __ubuf__ T2* x_exp_0 = x_exp;
    __ubuf__ T2* x_exp_1 = x_exp_0 + (n * 4);

    MaskReg preg_input = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_out = CreateMask<T2, MaskPattern::ALL>();

    RegTensor<float> vreg_src_0, vreg_src_1, vreg_src_2, vreg_src_3;
    RegTensor<float> vreg_src_4, vreg_src_5, vreg_src_6, vreg_src_7;
    RegTensor<float> vreg_max_0;
    RegTensor<float> vreg_sum_0, vreg_sum_1, vreg_sum_2, vreg_sum_3;
    RegTensor<float> vreg_exp_0, vreg_exp_1, vreg_exp_2, vreg_exp_3;
    RegTensor<T2> vreg_exp_cast_0, vreg_exp_cast_1, vreg_exp_cast_2, vreg_exp_cast_3;
    RegTensor<T2> vreg_exp_pack_0, vreg_exp_pack_1;
    RegTensor<float> vreg_max_last;
    RegTensor<float> vreg_sum_last;
    RegTensor<float> vreg_exp_max;
    RegTensor<float> vreg_min;

    Duplicate(vreg_max_0, minValue);
    Duplicate(vreg_min, minValue);
    for (uint16_t i = originN; i < uint16_t(n); ++i) {
        StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ float*&)input_x_local_UB + i * m, vreg_min,
                                                         preg_input);
    }
    mem_bar(VST_VLD);

    for (uint16_t iter_m = 0; iter_m < uint16_t(n / 8); ++iter_m) {
        LoadAlign(vreg_src_0, src_ub_0 + iter_m * m * 8);
        LoadAlign(vreg_src_1, src_ub_1 + iter_m * m * 8);
        LoadAlign(vreg_src_2, src_ub_2 + iter_m * m * 8);
        LoadAlign(vreg_src_3, src_ub_3 + iter_m * m * 8);
        LoadAlign(vreg_src_4, src_ub_4 + iter_m * m * 8);
        LoadAlign(vreg_src_5, src_ub_5 + iter_m * m * 8);
        LoadAlign(vreg_src_6, src_ub_6 + iter_m * m * 8);
        LoadAlign(vreg_src_7, src_ub_7 + iter_m * m * 8);

        Max(vreg_src_0, vreg_src_0, vreg_src_1, preg_input);
        Max(vreg_src_2, vreg_src_2, vreg_src_3, preg_input);
        Max(vreg_src_4, vreg_src_4, vreg_src_5, preg_input);
        Max(vreg_src_6, vreg_src_6, vreg_src_7, preg_input);
        Max(vreg_src_0, vreg_src_0, vreg_src_2, preg_input);
        Max(vreg_src_4, vreg_src_4, vreg_src_6, preg_input);
        Max(vreg_src_0, vreg_src_0, vreg_src_4, preg_input);
        Max(vreg_max_0, vreg_max_0, vreg_src_0, preg_input);
    }

    Muls(vreg_max_0, vreg_max_0, scale, preg_input);
    if constexpr (isUpdate) {
        LoadAlign(vreg_max_last, new_global_max);
        Max(vreg_max_0, vreg_max_0, vreg_max_last, preg_input);
    }
    StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ float*&)new_global_max, vreg_max_0, preg_input);
    if constexpr (isUpdate) {
        FusedExpSub(vreg_exp_max, vreg_max_last, vreg_max_0, preg_input);
        StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ float*&)exp_max, vreg_exp_max, preg_input);
    }

    Duplicate<float, Reg::MaskMergeMode::ZEROING, float>(vreg_sum_0, 0, preg_input);
    Duplicate<float, Reg::MaskMergeMode::ZEROING, float>(vreg_sum_1, 0, preg_input);
    Duplicate<float, Reg::MaskMergeMode::ZEROING, float>(vreg_sum_2, 0, preg_input);
    Duplicate<float, Reg::MaskMergeMode::ZEROING, float>(vreg_sum_3, 0, preg_input);

    for (uint16_t i = 0; i < uint16_t(n / 4); ++i) {
        LoadAlign(vreg_src_0, input_x_local_UB + i * m);
        LoadAlign(vreg_src_1, input_x_local_UB + n * m / 2 + i * m);
        LoadAlign(vreg_src_2, input_x_local_UB + n * m / 4 + i * m);
        LoadAlign(vreg_src_3, input_x_local_UB + n * m / 2 + n * m / 4 + i * m);

        Muls(vreg_src_0, vreg_src_0, scale, preg_input);
        Muls(vreg_src_1, vreg_src_1, scale, preg_input);
        Muls(vreg_src_2, vreg_src_2, scale, preg_input);
        Muls(vreg_src_3, vreg_src_3, scale, preg_input);

        FusedExpSub(vreg_exp_0, vreg_src_0, vreg_max_0, preg_input);
        FusedExpSub(vreg_exp_1, vreg_src_1, vreg_max_0, preg_input);
        FusedExpSub(vreg_exp_2, vreg_src_2, vreg_max_0, preg_input);
        FusedExpSub(vreg_exp_3, vreg_src_3, vreg_max_0, preg_input);

        Add(vreg_sum_0, vreg_exp_0, vreg_sum_0, preg_input);
        Add(vreg_sum_1, vreg_exp_1, vreg_sum_1, preg_input);
        Add(vreg_sum_2, vreg_exp_2, vreg_sum_2, preg_input);
        Add(vreg_sum_3, vreg_exp_3, vreg_sum_3, preg_input);

        Cast<T2, float, castTraitZero>(vreg_exp_cast_0, vreg_exp_0, preg_input);
        Cast<T2, float, castTraitZero>(vreg_exp_cast_1, vreg_exp_1, preg_input);
        Cast<T2, float, castTraitZero>(vreg_exp_cast_2, vreg_exp_2, preg_input);
        Cast<T2, float, castTraitZero>(vreg_exp_cast_3, vreg_exp_3, preg_input);
        DeInterleave<T2, Reg::HighLowPart::LOWEST>(vreg_exp_pack_0, vreg_exp_cast_0, vreg_exp_cast_1);
        DeInterleave<T2, Reg::HighLowPart::LOWEST>(vreg_exp_pack_1, vreg_exp_cast_2, vreg_exp_cast_3);
        StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2*&)x_exp_0), vreg_exp_pack_0, blockStride, repeatStride, preg_out);
        StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2*&)x_exp_1), vreg_exp_pack_1, blockStride, repeatStride, preg_out);
    }

    Add(vreg_sum_0, vreg_sum_1, vreg_sum_0, preg_input);
    Add(vreg_sum_2, vreg_sum_3, vreg_sum_2, preg_input);
    Add(vreg_sum_0, vreg_sum_0, vreg_sum_2, preg_input);

    if constexpr (isUpdate) {
        LoadAlign(vreg_sum_last, new_global_sum);
        Mul(vreg_sum_last, vreg_exp_max, vreg_sum_last, preg_input);
        Add(vreg_sum_0, vreg_sum_last, vreg_sum_0, preg_input);
    }

    StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ float*&)new_global_sum, vreg_sum_0, preg_input);
}

template <typename T2, uint16_t s2Size, bool isUpdate>
__aicore__ inline void MqsmlaVec1Dn64(const LocalTensor<T2>& dstTensor, const LocalTensor<float>& expSumTensor,
                                      const LocalTensor<float>& maxTensor, const LocalTensor<float>& srcTensor,
                                      const LocalTensor<float>& expMaxTensor, const uint32_t m, const uint32_t n,
                                      const uint32_t originN, const float scale, const float minValue)
{
    constexpr uint32_t repeatStride = 1;
    constexpr uint32_t blockStride = (s2Size >> 1) | 0x1;

    __ubuf__ T2* x_exp = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ float* input_x_local_UB = (__ubuf__ float*)srcTensor.GetPhyAddr();
    __ubuf__ float* exp_max = (__ubuf__ float*)expMaxTensor.GetPhyAddr();
    __ubuf__ float* new_global_sum = (__ubuf__ float*)expSumTensor.GetPhyAddr();
    __ubuf__ float* new_global_max = (__ubuf__ float*)maxTensor.GetPhyAddr();

    MqsmlaVec1Dn64VF<T2, isUpdate>(x_exp, input_x_local_UB, exp_max, new_global_sum, new_global_max, m, n, originN,
                                   scale, minValue, blockStride, repeatStride);
}

template <typename T>
__simd_vf__ inline void InitSoftmaxFromSinksVF(__ubuf__ T* sumUb, __ubuf__ T* maxUb, __ubuf__ T* sinksUb,
                                               uint32_t sinksOffset, const T R0, uint32_t m)
{
    AscendC::Reg::RegTensor<T> vreg_sinks;
    AscendC::Reg::RegTensor<T> vreg_sum;
    AscendC::Reg::MaskReg preg_m = AscendC::Reg::UpdateMask<T>(m);
    AscendC::Reg::UnalignRegForLoad ureg;
    auto srcUbT = sinksUb + sinksOffset;
    AscendC::Reg::LoadUnAlignPre(ureg, srcUbT);
    AscendC::Reg::LoadUnAlign(vreg_sinks, ureg, srcUbT);
    AscendC::Reg::StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(maxUb, vreg_sinks, preg_m);
    AscendC::Reg::Duplicate<T, Reg::MaskMergeMode::ZEROING, T>(vreg_sum, R0, preg_m);
    AscendC::Reg::StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(sumUb, vreg_sum, preg_m);
}

template <typename T>
__aicore__ inline void InitSoftmaxFromSinks(const LocalTensor<T>& sumTensor, const LocalTensor<T>& maxTensor,
                                            const LocalTensor<T>& sinksTensor, uint32_t sinksOffset, const T R0,
                                            uint32_t m)
{
    __ubuf__ T* sumUb = (__ubuf__ T*)sumTensor.GetPhyAddr();
    __ubuf__ T* maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T* sinksUb = (__ubuf__ T*)sinksTensor.GetPhyAddr();
    InitSoftmaxFromSinksVF<T>(sumUb, maxUb, sinksUb, sinksOffset, R0, m);
}

template <typename T>
__simd_vf__ inline void ComputeLseVF(__ubuf__ T* outLseUb, __ubuf__ T* sumUb, __ubuf__ T* maxUb, uint32_t m)
{
    AscendC::Reg::RegTensor<T> vreg_outLse;
    AscendC::Reg::RegTensor<T> vreg_sum;
    AscendC::Reg::RegTensor<T> vreg_max;
    AscendC::Reg::RegTensor<T> vreg_tmp;
    AscendC::Reg::MaskReg preg_m = AscendC::Reg::UpdateMask<T>(m);
    AscendC::Reg::LoadAlign(vreg_sum, sumUb);
    AscendC::Reg::LoadAlign(vreg_max, maxUb);
    AscendC::Reg::UnalignRegForStore ureg;
    AscendC::Reg::Log<T, MaskMergeMode::ZEROING>(vreg_tmp, vreg_sum, preg_m);
    AscendC::Reg::Add(vreg_outLse, vreg_tmp, vreg_max, preg_m);
    AscendC::Reg::StoreAlign<T, Reg::StoreDist::DIST_NORM>(outLseUb, vreg_outLse, preg_m);
}

template <typename T>
__aicore__ inline void ComputeLse(const LocalTensor<T>& outLseTensor, const LocalTensor<T>& sumTensor,
                                  const LocalTensor<T>& maxTensor, uint32_t m)
{
    __ubuf__ T* outLseUb = (__ubuf__ T*)outLseTensor.GetPhyAddr();
    __ubuf__ T* sumUb = (__ubuf__ T*)sumTensor.GetPhyAddr();
    __ubuf__ T* maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    ComputeLseVF<T>(outLseUb, sumUb, maxUb, m);
}

} // namespace FaVectorApi
#endif
