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
 * \file vf_mul_sel_softmaxflashv2_cast_nz_dn_regbase_v2.h
 * \brief
 */

#ifndef MUL_SEL_SOFTMAXFLASHV2_CAST_NZ_DN_REGBASE_V2_H_
#define MUL_SEL_SOFTMAXFLASHV2_CAST_NZ_DN_REGBASE_V2_H_
#include "kernel_tensor.h"
namespace FaVectorApi {
using AscendC::LocalTensor;
using namespace AscendC;
using namespace Reg;

#define VMULSCVT false
#define DROPOUT false

template <typename T, typename T2, uint16_t ubN = 128>
__aicore__ inline void ProcessVec1DnNoUpdateRegbaseV2(const LocalTensor<T2> &dstTensor,
                                                      const LocalTensor<float> &expSumTensor,
                                                      const LocalTensor<T> &maxTensor, const LocalTensor<T> &srcTensor,
                                                      const LocalTensor<float> &expMaxTensor, const uint32_t m,
                                                      const uint32_t originN, const T scale, const T minValue,
                                                      float quantScale1)
{
    constexpr uint32_t stride = (ubN << 15) | (0x1 << 16) | 0x1;
    constexpr uint32_t blockN = 32;  // 一个block包含32个int8数据
    constexpr uint32_t blockU8 = 32; // 一个block有32个字节
    constexpr uint16_t repeatStride = 1;
    constexpr uint16_t blockStride = (ubN >> 1) * blockN * sizeof(T2) / blockU8;

    __ubuf__ T2 *x_exp = (__ubuf__ T2 *)dstTensor.GetPhyAddr();
    __ubuf__ T2 *x_exp_1 = x_exp + (ubN * 8);
    __ubuf__ T *input_x_local_UB = (__ubuf__ T *)srcTensor.GetPhyAddr();
    __ubuf__ float *exp_max_fp32 = (__ubuf__ float *)expMaxTensor.GetPhyAddr();
    __ubuf__ float *new_global_sum = (__ubuf__ float *)expSumTensor.GetPhyAddr();
    __ubuf__ T *new_global_max = (__ubuf__ T *)maxTensor.GetPhyAddr();
    __ubuf__ T *src0_ub = (__ubuf__ T *)input_x_local_UB;
    __ubuf__ T *src0_ub_1 = (__ubuf__ T *)input_x_local_UB + m;
    __ubuf__ T *src0_ub1 = src0_ub + m * 2;
    __ubuf__ T *src0_ub1_1 = src0_ub + m * 3;
    __ubuf__ T *src0_ub2 = src0_ub + m * 4;
    __ubuf__ T *src0_ub2_1 = src0_ub + m * 5;
    __ubuf__ T *src0_ub3 = src0_ub + m * 6;
    __ubuf__ T *src0_ub3_1 = src0_ub + m * 7;

    __VEC_SCOPE__
    {
        Reg::RegTensor<T> vreg_x_sum_even;
        Reg::RegTensor<T> vreg_x_sum_odd;
        Reg::RegTensor<T> vreg_x_sum_1_even;
        Reg::RegTensor<T> vreg_x_sum_1_odd;
        Reg::RegTensor<T> vreg_x_sum0;
        Reg::RegTensor<float> vreg_x_sum0_even;
        Reg::RegTensor<float> vreg_x_sum0_odd;
        Reg::RegTensor<T> vreg_x_sum1;
        Reg::RegTensor<T2> vreg_x_exp_even_f16;
        Reg::RegTensor<T2> vreg_x_exp_odd_f16;
        Reg::RegTensor<T2> vreg_x_exp_even_bf16;
        Reg::RegTensor<T2> vreg_x_exp_odd_bf16;
        Reg::RegTensor<half> vreg_muls_res_even;
        Reg::RegTensor<half> vreg_muls_res_odd;
        Reg::RegTensor<half> vreg_muls_res_even_1;
        Reg::RegTensor<half> vreg_muls_res_odd_1;

        Reg::RegTensor<T> vreg_x_exp_even;
        Reg::RegTensor<T> vreg_x_exp_odd;
        Reg::RegTensor<T> vreg_x_f32_a;
        Reg::RegTensor<T> vreg_x_f32_b;
        Reg::RegTensor<T> vreg_x_exp_even_1;
        Reg::RegTensor<T> vreg_x_exp_odd_1;
        Reg::RegTensor<T2> vreg_x_exp_even_f16_1;
        Reg::RegTensor<T2> vreg_x_exp_odd_f16_1;
        Reg::RegTensor<T2> vreg_x_exp_even_bf16_1;
        Reg::RegTensor<T2> vreg_x_exp_odd_bf16_1;

        Reg::RegTensor<T> vreg_x_f32_1_a;
        Reg::RegTensor<T> vreg_x_f32_1_b;
        Reg::RegTensor<T2> vreg_x_exp_f16_pack;
        Reg::RegTensor<T2> vreg_x_exp_f16_1_pack;
        Reg::RegTensor<T2> vreg_x_exp_f16_packa;
        Reg::RegTensor<T2> vreg_x_exp_f16_1_packa;
        Reg::RegTensor<T2> vreg_x_exp_bf16_pack;
        Reg::RegTensor<T2> vreg_x_exp_bf16_1_pack;
        Reg::RegTensor<T2> vreg_x_exp_bf16_packa;
        Reg::RegTensor<T2> vreg_x_exp_bf16_1_packa;
        Reg::MaskReg preg_100;
        Reg::MaskReg preg_101;
        Reg::MaskReg preg_134;
        Reg::MaskReg preg_135;
        Reg::MaskReg preg_136;
        Reg::MaskReg preg_all_b16;
        Reg::MaskReg preg_all_b32;
        preg_134 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        preg_135 = Reg::CreateMask<half, Reg::MaskPattern::ALL>();
        uint32_t sreg_92 = (uint32_t)256ULL;
        preg_136 = Reg::UpdateMask<T2>(sreg_92);
        preg_all_b16 = Reg::CreateMask<half, Reg::MaskPattern::ALL>();
        preg_all_b32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::MaskReg preg_108;
        Reg::RegTensor<T> src_00a, src_01a, src_02a, src_03a;
        Reg::RegTensor<T> src_00b, src_01b, src_02b, src_03b;
        Reg::RegTensor<T> src_10a, src_11a, src_12a, src_13a;
        Reg::RegTensor<T> src_10b, src_11b, src_12b, src_13b;
        Reg::RegTensor<T> max_0a, max_1a, max_2a, max_3a;
        Reg::RegTensor<T> max_0b, max_1b, max_2b, max_3b;
        Reg::RegTensor<T> vreg_min;

        Reg::Duplicate<T, T>(max_0a, minValue);
        Reg::Duplicate<T, T>(max_0b, minValue);
        Reg::Duplicate<T, T>(max_1a, minValue);
        Reg::Duplicate<T, T>(max_1b, minValue);
        Reg::Duplicate<T, T>(max_2a, minValue);
        Reg::Duplicate<T, T>(max_2b, minValue);
        Reg::Duplicate<T, T>(max_3a, minValue);
        Reg::Duplicate<T, T>(max_3b, minValue);
        Reg::Duplicate<T, T>(vreg_min, minValue);

        for (uint16_t i = originN; i < ubN; ++i) {
            Reg::DataCopy<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T *)input_x_local_UB + i * m, vreg_min, preg_135);
        } // 用最小值覆盖行方向（sInner方向）的脏数据

        preg_108 = pge_b16(PAT_ALL);
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 8); ++iter_m) {
            auto aReg = AscendC::Reg::CreateAddrReg<T>(iter_m, m * 8);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(src_00a, src0_ub, aReg);
            Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max_0a, max_0a, src_00a, preg_108);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(src_00b, src0_ub_1, aReg);
            Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max_0b, max_0b, src_00b, preg_108);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(src_01a, src0_ub1, aReg);
            Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max_1a, max_1a, src_01a, preg_108);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(src_01b, src0_ub1_1, aReg);
            Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max_1b, max_1b, src_01b, preg_108);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(src_02a, src0_ub2, aReg);
            Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max_2a, max_2a, src_02a, preg_108);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(src_02b, src0_ub2_1, aReg);
            Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max_2b, max_2b, src_02b, preg_108);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(src_03a, src0_ub3, aReg);
            Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max_3a, max_3a, src_03a, preg_108);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(src_03b, src0_ub3_1, aReg);
            Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max_3b, max_3b, src_03b, preg_108);
        }
        Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max_0a, max_0a, max_1a, preg_108);
        Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max_0b, max_0b, max_1b, preg_108);
        Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max_2a, max_2a, max_3a, preg_108);
        Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max_2b, max_2b, max_3b, preg_108);
        Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max_0a, max_0a, max_2a, preg_108);
        Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max_0b, max_0b, max_2b, preg_108);
        Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max_0a, max_0a, max_0b, preg_108);
        Reg::Muls<T, T, Reg::MaskMergeMode::ZEROING>(max_0a, max_0a, scale, preg_108);

        Reg::DataCopy<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T *)new_global_max, max_0a, preg_108);

        Reg::Duplicate<T, T>(vreg_x_sum_even, 0);
        Reg::Duplicate<T, T>(vreg_x_sum_odd, 0);
        Reg::Duplicate<T, T>(vreg_x_sum_1_even, 0);
        Reg::Duplicate<T, T>(vreg_x_sum_1_odd, 0);
        for (uint16_t i0 = 0; i0 < uint16_t(ubN / 4); ++i0) {
            auto aReg = AscendC::Reg::CreateAddrReg<T>(i0, m);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vreg_x_f32_a, ((__ubuf__ T *)input_x_local_UB), aReg);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vreg_x_f32_b, ((__ubuf__ T *)input_x_local_UB + ubN * m / 2),
                                                       aReg);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vreg_x_f32_1_a, ((__ubuf__ T *)input_x_local_UB + ubN * m / 4),
                                                       aReg);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(
                vreg_x_f32_1_b, ((__ubuf__ T *)input_x_local_UB + ubN * m / 2 + ubN * m / 4), aReg);

            Reg::Muls<T, T, Reg::MaskMergeMode::ZEROING>(vreg_x_f32_a, vreg_x_f32_a, scale, preg_108);
            Reg::Muls<T, T, Reg::MaskMergeMode::ZEROING>(vreg_x_f32_b, vreg_x_f32_b, scale, preg_108);

            Reg::FusedExpSub<T, T, Reg::RegLayout::ONE, Reg::MaskMergeMode::ZEROING>(vreg_x_exp_even, vreg_x_f32_a,
                                                                                     max_0a, preg_134);
            Reg::FusedExpSub<T, T, Reg::RegLayout::ONE, Reg::MaskMergeMode::ZEROING>(vreg_x_exp_odd, vreg_x_f32_b,
                                                                                     max_0a, preg_134);
            Reg::Muls<half, half, Reg::MaskMergeMode::ZEROING>(vreg_muls_res_even, vreg_x_exp_even, (half)quantScale1,
                                                               preg_all_b16);
            Reg::Muls<half, half, Reg::MaskMergeMode::ZEROING>(vreg_muls_res_odd, vreg_x_exp_odd, (half)quantScale1,
                                                               preg_all_b16);

            Reg::Muls<T, T, Reg::MaskMergeMode::ZEROING>(vreg_x_f32_1_a, vreg_x_f32_1_a, scale, preg_108);
            Reg::Muls<T, T, Reg::MaskMergeMode::ZEROING>(vreg_x_f32_1_b, vreg_x_f32_1_b, scale, preg_108);

            Reg::FusedExpSub<T, T, Reg::RegLayout::ONE, Reg::MaskMergeMode::ZEROING>(vreg_x_exp_even_1, vreg_x_f32_1_a,
                                                                                     max_0a, preg_134);
            Reg::FusedExpSub<T, T, Reg::RegLayout::ONE, Reg::MaskMergeMode::ZEROING>(vreg_x_exp_odd_1, vreg_x_f32_1_b,
                                                                                     max_0a, preg_134);
            Reg::Muls<half, half, Reg::MaskMergeMode::ZEROING>(vreg_muls_res_even_1, vreg_x_exp_even_1,
                                                               (half)quantScale1, preg_all_b16);
            Reg::Muls<half, half, Reg::MaskMergeMode::ZEROING>(vreg_muls_res_odd_1, vreg_x_exp_odd_1, (half)quantScale1,
                                                               preg_all_b16);

            static constexpr Reg::CastTrait castTrait0 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            Reg::Cast<T2, T, castTrait0>(vreg_x_exp_even_f16, vreg_muls_res_even, preg_135);
            Reg::Cast<T2, T, castTrait0>(vreg_x_exp_odd_f16, vreg_muls_res_odd, preg_135);
            Reg::DeInterleave<uint8_t>(
                (Reg::RegTensor<uint8_t> &)vreg_x_exp_f16_pack, (Reg::RegTensor<uint8_t> &)vreg_x_exp_f16_packa,
                (Reg::RegTensor<uint8_t> &)vreg_x_exp_even_f16, (Reg::RegTensor<uint8_t> &)vreg_x_exp_odd_f16);

            Reg::Cast<T2, T, castTrait0>(vreg_x_exp_even_f16_1, vreg_muls_res_even_1, preg_135);
            Reg::Cast<T2, T, castTrait0>(vreg_x_exp_odd_f16_1, vreg_muls_res_odd_1, preg_135);
            Reg::DeInterleave<uint8_t>(
                (Reg::RegTensor<uint8_t> &)vreg_x_exp_f16_1_pack, (Reg::RegTensor<uint8_t> &)vreg_x_exp_f16_1_packa,
                (Reg::RegTensor<uint8_t> &)vreg_x_exp_even_f16_1, (Reg::RegTensor<uint8_t> &)vreg_x_exp_odd_f16_1);

            Reg::DataCopy<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                x_exp, vreg_x_exp_f16_pack, blockStride, repeatStride, preg_136);
            Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vreg_x_sum_even, vreg_x_exp_even, vreg_x_sum_even, preg_134);
            Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vreg_x_sum_odd, vreg_x_exp_odd, vreg_x_sum_odd, preg_134);

            Reg::DataCopy<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                x_exp_1, vreg_x_exp_f16_1_pack, blockStride, repeatStride, preg_136);
            Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vreg_x_sum_1_even, vreg_x_exp_even_1, vreg_x_sum_1_even, preg_134);
            Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vreg_x_sum_1_odd, vreg_x_exp_odd_1, vreg_x_sum_1_odd, preg_134);
        }
        Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vreg_x_sum0, vreg_x_sum_odd, vreg_x_sum_even, preg_134);
        Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vreg_x_sum1, vreg_x_sum_1_odd, vreg_x_sum_1_even, preg_134);
        Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vreg_x_sum0, vreg_x_sum0, vreg_x_sum1, preg_134);
        static constexpr Reg::CastTrait castTrait0 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                      Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTrait1 = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                      Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        Reg::Cast<float, half, castTrait0>(vreg_x_sum0_even, vreg_x_sum0, preg_all_b16);
        Reg::Cast<float, half, castTrait1>(vreg_x_sum0_odd, vreg_x_sum0, preg_all_b16);
        Reg::DataCopy<float, Reg::StoreDist::DIST_INTLV_B32>(new_global_sum, vreg_x_sum0_even, vreg_x_sum0_odd,
                                                             preg_all_b32);
    }
}

template <typename T, typename T2, uint16_t ubN = 128>
__aicore__ inline void ProcessVec1DnUpdateRegbaseV2(const LocalTensor<T2> &dstTensor,
                                                    const LocalTensor<float> &expSumTensor,
                                                    const LocalTensor<T> &maxTensor, const LocalTensor<T> &srcTensor,
                                                    const LocalTensor<float> &expMaxTensor, const uint32_t m,
                                                    const uint32_t originN, const T scale, const T minValue,
                                                    float quantScale1)
{
    constexpr uint32_t stride = (ubN << 15) | (0x1 << 16) | 0x1;
    constexpr uint32_t blockN = 32;  // 一个block包含32个fp16数据
    constexpr uint32_t blockU8 = 32; // 一个block有32个字节
    constexpr uint16_t repeatStride = 1;
    constexpr uint16_t blockStride = (ubN >> 1) * blockN * sizeof(T2) / blockU8;

    __ubuf__ T2 *x_exp = (__ubuf__ T2 *)dstTensor.GetPhyAddr();
    __ubuf__ T2 *x_exp_1 = x_exp + (ubN * 8);
    __ubuf__ T *input_x_local_UB = (__ubuf__ T *)srcTensor.GetPhyAddr();
    __ubuf__ float *exp_max_fp32 = (__ubuf__ float *)expMaxTensor.GetPhyAddr();
    __ubuf__ float *new_global_sum = (__ubuf__ float *)expSumTensor.GetPhyAddr();
    __ubuf__ T *new_global_max = (__ubuf__ T *)maxTensor.GetPhyAddr();
    __ubuf__ T *src0_ub = (__ubuf__ T *)input_x_local_UB;
    __ubuf__ T *src0_ub_1 = (__ubuf__ T *)input_x_local_UB + m;
    __ubuf__ T *src0_ub1 = src0_ub + m * 2;
    __ubuf__ T *src0_ub1_1 = src0_ub + m * 3;
    __ubuf__ T *src0_ub2 = src0_ub + m * 4;
    __ubuf__ T *src0_ub2_1 = src0_ub + m * 5;
    __ubuf__ T *src0_ub3 = src0_ub + m * 6;
    __ubuf__ T *src0_ub3_1 = src0_ub + m * 7;

    __VEC_SCOPE__
    {
        Reg::RegTensor<T> vreg_x_sum_even;
        Reg::RegTensor<T> vreg_x_sum_odd;
        Reg::RegTensor<T> vreg_x_sum_1_even;
        Reg::RegTensor<T> vreg_x_sum_1_odd;
        Reg::RegTensor<T> vreg_x_sum0;
        Reg::RegTensor<float> vreg_x_sum0_even;
        Reg::RegTensor<float> vreg_x_sum0_odd;
        Reg::RegTensor<T> vreg_x_sum1;
        Reg::RegTensor<T2> vreg_x_exp_even_f16;
        Reg::RegTensor<T2> vreg_x_exp_odd_f16;
        Reg::RegTensor<T2> vreg_x_exp_even_bf16;
        Reg::RegTensor<T2> vreg_x_exp_odd_bf16;

        Reg::RegTensor<T> vreg_x_exp_even;
        Reg::RegTensor<T> vreg_x_exp_odd;
        Reg::RegTensor<T> vreg_x_f32_a;
        Reg::RegTensor<T> vreg_x_f32_b;
        Reg::RegTensor<T> vreg_x_exp_even_1;
        Reg::RegTensor<T> vreg_x_exp_odd_1;
        Reg::RegTensor<T2> vreg_x_exp_even_f16_1;
        Reg::RegTensor<T2> vreg_x_exp_odd_f16_1;
        Reg::RegTensor<T2> vreg_x_exp_even_bf16_1;
        Reg::RegTensor<T2> vreg_x_exp_odd_bf16_1;
        Reg::RegTensor<half> vreg_muls_res_even;
        Reg::RegTensor<half> vreg_muls_res_odd;
        Reg::RegTensor<half> vreg_muls_res_even_1;
        Reg::RegTensor<half> vreg_muls_res_odd_1;

        Reg::RegTensor<T> vreg_x_f32_1_a;
        Reg::RegTensor<T> vreg_x_f32_1_b;
        Reg::RegTensor<float> vreg_x_max_f32_even;
        Reg::RegTensor<float> vreg_x_max_f32_odd;
        Reg::RegTensor<T> vreg_x_max_f16;
        Reg::RegTensor<float> vreg_x_max_f32;
        Reg::RegTensor<float> vreg_x_max_f32_1;
        Reg::RegTensor<T2> vreg_x_exp_f16_pack;
        Reg::RegTensor<T2> vreg_x_exp_f16_1_pack;
        Reg::RegTensor<T2> vreg_x_exp_f16_packa;
        Reg::RegTensor<T2> vreg_x_exp_f16_1_packa;

        Reg::RegTensor<T2> vreg_x_exp_bf16_pack;
        Reg::RegTensor<T2> vreg_x_exp_bf16_1_pack;
        Reg::RegTensor<T2> vreg_x_exp_bf16_packa;
        Reg::RegTensor<T2> vreg_x_exp_bf16_1_packa;

        Reg::MaskReg preg_100;
        Reg::MaskReg preg_101;
        Reg::MaskReg preg_134;
        Reg::MaskReg preg_135;
        Reg::MaskReg preg_136;
        Reg::MaskReg preg_all_b16;
        Reg::MaskReg preg_all_b32;
        preg_134 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        preg_135 = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
        uint32_t sreg_92 = (uint32_t)256ULL;
        preg_136 = Reg::UpdateMask<T2>(sreg_92);
        preg_all_b16 = Reg::CreateMask<half, Reg::MaskPattern::ALL>();
        preg_all_b32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::MaskReg preg_108;
        Reg::RegTensor<T> src_00a, src_01a, src_02a, src_03a;
        Reg::RegTensor<T> src_00b, src_01b, src_02b, src_03b;
        Reg::RegTensor<T> src_10a, src_11a, src_12a, src_13a;
        Reg::RegTensor<T> src_10b, src_11b, src_12b, src_13b;
        Reg::RegTensor<T> max_0a, max_1a, max_2a, max_3a;
        Reg::RegTensor<T> max_0b, max_1b, max_2b, max_3b;
        Reg::RegTensor<T> vreg_min;

        Reg::Duplicate<T, T>(max_0a, minValue);
        Reg::Duplicate<T, T>(max_0b, minValue);
        Reg::Duplicate<T, T>(max_1a, minValue);
        Reg::Duplicate<T, T>(max_1b, minValue);
        Reg::Duplicate<T, T>(max_2a, minValue);
        Reg::Duplicate<T, T>(max_2b, minValue);
        Reg::Duplicate<T, T>(max_3a, minValue);
        Reg::Duplicate<T, T>(max_3b, minValue);
        Reg::Duplicate<T, T>(vreg_min, minValue);
        for (uint16_t i = originN; i < ubN; ++i) {
            Reg::DataCopy<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T *)input_x_local_UB + i * m, vreg_min, preg_135);
        }

        preg_108 = pge_b16(PAT_ALL);
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 8); ++iter_m) {
            auto aReg = AscendC::Reg::CreateAddrReg<T>(iter_m, m * 8);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(src_00a, src0_ub, aReg);
            Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max_0a, max_0a, src_00a, preg_108);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(src_00b, src0_ub_1, aReg);
            Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max_0b, max_0b, src_00b, preg_108);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(src_01a, src0_ub1, aReg);
            Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max_1a, max_1a, src_01a, preg_108);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(src_01b, src0_ub1_1, aReg);
            Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max_1b, max_1b, src_01b, preg_108);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(src_02a, src0_ub2, aReg);
            Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max_2a, max_2a, src_02a, preg_108);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(src_02b, src0_ub2_1, aReg);
            Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max_2b, max_2b, src_02b, preg_108);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(src_03a, src0_ub3, aReg);
            Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max_3a, max_3a, src_03a, preg_108);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(src_03b, src0_ub3_1, aReg);
            Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max_3b, max_3b, src_03b, preg_108);
        }
        Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vreg_x_max_f16, ((__ubuf__ T *)new_global_max));
        Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max_0a, max_0a, max_1a, preg_108);
        Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max_0b, max_0b, max_1b, preg_108);
        Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max_2a, max_2a, max_3a, preg_108);
        Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max_2b, max_2b, max_3b, preg_108);
        Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max_0a, max_0a, max_2a, preg_108);
        Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max_0b, max_0b, max_2b, preg_108);
        Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max_0a, max_0a, max_0b, preg_108);
        Reg::Muls<T, T, Reg::MaskMergeMode::ZEROING>(max_0a, max_0a, scale, preg_108);
        Reg::Max<T, Reg::MaskMergeMode::ZEROING>(max_0a, max_0a, vreg_x_max_f16, preg_108);

        Reg::FusedExpSub<T, T, Reg::RegLayout::ONE, Reg::MaskMergeMode::ZEROING>(vreg_x_max_f16, vreg_x_max_f16, max_0a,
                                                                                 preg_134);
        Reg::DataCopy<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T *)new_global_max, max_0a, preg_108);
        static constexpr Reg::CastTrait castTrait0 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                      Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTrait1 = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                      Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        Reg::Cast<float, half, castTrait0>(vreg_x_max_f32_even, vreg_x_max_f16, preg_all_b16);
        Reg::Cast<float, half, castTrait1>(vreg_x_max_f32_odd, vreg_x_max_f16, preg_all_b16);
        Reg::DataCopy<float, Reg::StoreDist::DIST_INTLV_B32>(exp_max_fp32, vreg_x_max_f32_even, vreg_x_max_f32_odd,
                                                             preg_all_b32);

        Reg::Duplicate<T, T>(vreg_x_sum_even, 0);
        Reg::Duplicate<T, T>(vreg_x_sum_odd, 0);
        Reg::Duplicate<T, T>(vreg_x_sum_1_even, 0);
        Reg::Duplicate<T, T>(vreg_x_sum_1_odd, 0);

        for (uint16_t i0 = 0; i0 < uint16_t(ubN / 4); ++i0) {
            auto aReg = AscendC::Reg::CreateAddrReg<T>(i0, m);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vreg_x_f32_a, ((__ubuf__ T *)input_x_local_UB), aReg);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vreg_x_f32_b, ((__ubuf__ T *)input_x_local_UB + ubN * m / 2),
                                                       aReg);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vreg_x_f32_1_a, ((__ubuf__ T *)input_x_local_UB + ubN * m / 4),
                                                       aReg);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(
                vreg_x_f32_1_b, ((__ubuf__ T *)input_x_local_UB + ubN * m / 2 + ubN * m / 4), aReg);

            Reg::Muls<T, T, Reg::MaskMergeMode::ZEROING>(vreg_x_f32_a, vreg_x_f32_a, scale, preg_108);
            Reg::Muls<T, T, Reg::MaskMergeMode::ZEROING>(vreg_x_f32_b, vreg_x_f32_b, scale, preg_108);

            Reg::FusedExpSub<T, T, Reg::RegLayout::ONE, Reg::MaskMergeMode::ZEROING>(vreg_x_exp_even, vreg_x_f32_a,
                                                                                     max_0a, preg_134);
            Reg::FusedExpSub<T, T, Reg::RegLayout::ONE, Reg::MaskMergeMode::ZEROING>(vreg_x_exp_odd, vreg_x_f32_b,
                                                                                     max_0a, preg_134);
            Reg::Muls<half, half, Reg::MaskMergeMode::ZEROING>(vreg_muls_res_even, vreg_x_exp_even, (half)quantScale1,
                                                               preg_all_b16);
            Reg::Muls<half, half, Reg::MaskMergeMode::ZEROING>(vreg_muls_res_odd, vreg_x_exp_odd, (half)quantScale1,
                                                               preg_all_b16);

            Reg::Muls<T, T, Reg::MaskMergeMode::ZEROING>(vreg_x_f32_1_a, vreg_x_f32_1_a, scale, preg_108);
            Reg::Muls<T, T, Reg::MaskMergeMode::ZEROING>(vreg_x_f32_1_b, vreg_x_f32_1_b, scale, preg_108);

            Reg::FusedExpSub<T, T, Reg::RegLayout::ONE, Reg::MaskMergeMode::ZEROING>(vreg_x_exp_even_1, vreg_x_f32_1_a,
                                                                                     max_0a, preg_134);
            Reg::FusedExpSub<T, T, Reg::RegLayout::ONE, Reg::MaskMergeMode::ZEROING>(vreg_x_exp_odd_1, vreg_x_f32_1_b,
                                                                                     max_0a, preg_134);

            Reg::Muls<half, half, Reg::MaskMergeMode::ZEROING>(vreg_muls_res_even_1, vreg_x_exp_even_1,
                                                               (half)quantScale1, preg_all_b16);
            Reg::Muls<half, half, Reg::MaskMergeMode::ZEROING>(vreg_muls_res_odd_1, vreg_x_exp_odd_1, (half)quantScale1,
                                                               preg_all_b16);

            static constexpr Reg::CastTrait castTrait0 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            Reg::Cast<T2, T, castTrait0>(vreg_x_exp_even_f16, vreg_muls_res_even, preg_135);
            Reg::Cast<T2, T, castTrait0>(vreg_x_exp_odd_f16, vreg_muls_res_odd, preg_135);
            Reg::DeInterleave<uint8_t>(
                (Reg::RegTensor<uint8_t> &)vreg_x_exp_f16_pack, (Reg::RegTensor<uint8_t> &)vreg_x_exp_f16_packa,
                (Reg::RegTensor<uint8_t> &)vreg_x_exp_even_f16, (Reg::RegTensor<uint8_t> &)vreg_x_exp_odd_f16);

            Reg::Cast<T2, T, castTrait0>(vreg_x_exp_even_f16_1, vreg_muls_res_even_1, preg_135);
            Reg::Cast<T2, T, castTrait0>(vreg_x_exp_odd_f16_1, vreg_muls_res_odd_1, preg_135);
            Reg::DeInterleave<uint8_t>(
                (Reg::RegTensor<uint8_t> &)vreg_x_exp_f16_1_pack, (Reg::RegTensor<uint8_t> &)vreg_x_exp_f16_1_packa,
                (Reg::RegTensor<uint8_t> &)vreg_x_exp_even_f16_1, (Reg::RegTensor<uint8_t> &)vreg_x_exp_odd_f16_1);

            Reg::DataCopy<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                x_exp, vreg_x_exp_f16_pack, blockStride, repeatStride, preg_136);
            Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vreg_x_sum_even, vreg_x_exp_even, vreg_x_sum_even, preg_134);
            Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vreg_x_sum_odd, vreg_x_exp_odd, vreg_x_sum_odd, preg_134);

            Reg::DataCopy<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                x_exp_1, vreg_x_exp_f16_1_pack, blockStride, repeatStride, preg_136);
            Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vreg_x_sum_1_even, vreg_x_exp_even_1, vreg_x_sum_1_even, preg_134);
            Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vreg_x_sum_1_odd, vreg_x_exp_odd_1, vreg_x_sum_1_odd, preg_134);
        }
        Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vreg_x_sum0, vreg_x_sum_odd, vreg_x_sum_even, preg_134);
        Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vreg_x_sum1, vreg_x_sum_1_odd, vreg_x_sum_1_even, preg_134);
        Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vreg_x_sum0, vreg_x_sum0, vreg_x_sum1, preg_134);
        Reg::Cast<float, half, castTrait0>(vreg_x_sum0_even, vreg_x_sum0, preg_all_b16);
        Reg::Cast<float, half, castTrait1>(vreg_x_sum0_odd, vreg_x_sum0, preg_all_b16);
        Reg::RegTensor<float> vreg_l0;
        Reg::RegTensor<float> vreg_l0_1;
        Reg::DataCopy<float, Reg::LoadDist::DIST_DINTLV_B32>(vreg_l0, vreg_l0_1, ((__ubuf__ float *)new_global_sum));
        Reg::DataCopy<float, Reg::LoadDist::DIST_DINTLV_B32>(vreg_x_max_f32, vreg_x_max_f32_1,
                                                             ((__ubuf__ float *)exp_max_fp32));
        Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(vreg_l0, vreg_x_max_f32, vreg_l0, preg_all_b32);
        Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(vreg_l0_1, vreg_x_max_f32_1, vreg_l0_1, preg_all_b32);
        Reg::Add<float, Reg::MaskMergeMode::ZEROING>(vreg_l0, vreg_l0, vreg_x_sum0_even, preg_all_b32);
        Reg::Add<float, Reg::MaskMergeMode::ZEROING>(vreg_l0_1, vreg_l0_1, vreg_x_sum0_odd, preg_all_b32);
        Reg::DataCopy<float, Reg::StoreDist::DIST_INTLV_B32>(new_global_sum, vreg_l0, vreg_l0_1, preg_all_b32);
    }
}

/*
 * @ingroup ProcessVec1Vf
 * @brief compute max = reducemax, exp(x-max)/sum(exp(x-max))
 * @param [out] dstTensor, output LocalTensor
 * @param [out] expSumTensor, out sum(exp(x-max)) of last axis
 * @param [out] maxTensor, out max value of last axis
 * @param [in] srcTensor, input LocalTensor
 * @param [out] expMaxTensor, output expmax LocalTensor
 * @param [in] sharedTmpBuffer, input local temporary Tensor
 * @param [in] m, input rows
 * @param [in] n, input colums, should be 256 bytes aligned, the value is originN aligned to 64
 * @param [in] originN, input origin colums, support range: 0 < originN <= 128
 * @param [in] scale, scale value
 * @param [in] minValue, minimum value
 * @param [in] isUpdate, enable flash mode
 * @param [in] oriNRange, originN range
 */

template <typename T, typename T2, bool isUpdate = false, uint16_t ubN = 256>
__aicore__ inline void ProcessVec1VfDnRegbaseV2(const LocalTensor<T2> &dstTensor,
                                                const LocalTensor<float> &expSumTensor, const LocalTensor<T> &maxTensor,
                                                const LocalTensor<T> &srcTensor, const LocalTensor<float> &expMaxTensor,
                                                const uint32_t m, const uint32_t originN, const T scale,
                                                const T minValue, float quantScale1)
{
    if constexpr (!isUpdate) {
        ProcessVec1DnNoUpdateRegbaseV2<T, T2, ubN>(dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, m,
                                                   originN, scale, minValue, quantScale1);
    } else {
        ProcessVec1DnUpdateRegbaseV2<T, T2, ubN>(dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, m,
                                                 originN, scale, minValue, quantScale1);
    }
}
} // namespace FaVectorApi
#endif // MUL_SEL_SOFTMAXFLASHV2_CAST_NZ_DN_REGBASE_V2_H_
